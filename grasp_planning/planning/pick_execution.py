"""Shared FR3 pick-execution helpers for debug grasp attempts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate
from grasp_planning.planning.fr3_motion_context import FR3MotionContext

from .admittance_controller import FR3AdmittanceController
from .goal_ik import GoalIKSolver
from .move_to_pose_controller import FR3MoveToPoseController
from .types import PoseCommand


@dataclass(frozen=True)
class PickExecutionResult:
    success: bool
    status: str
    message: str


def drive_robot_to_start_pose(sim, scene) -> None:
    """Actively settle the FR3 into a safe home pose before planning."""

    from grasp_planning.envs.fr3_cube_env import DEFAULT_ARM_START_JOINT_POS, DEFAULT_HAND_OPEN_WIDTH

    robot = scene["robot"]
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    arm_joint_names = tuple(DEFAULT_ARM_START_JOINT_POS.keys())
    arm_joint_ids = [joint_name_to_idx[name] for name in arm_joint_names]
    arm_targets = torch.tensor(
        [[DEFAULT_ARM_START_JOINT_POS[name] for name in arm_joint_names]], dtype=torch.float32, device=robot.device
    )
    hand_joint_names = tuple(
        name for name in robot.joint_names if name.startswith(("panda_finger_joint", "fr3_finger_joint"))
    )
    hand_target = float(DEFAULT_HAND_OPEN_WIDTH)
    physics_dt = sim.get_physics_dt()
    hand_joint_ids = [joint_name_to_idx[name] for name in hand_joint_names]

    for _ in range(max(1, int(1.5 / physics_dt))):
        robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_ids)
        if hand_joint_names:
            hand_targets = torch.full(
                (1, len(hand_joint_ids)),
                hand_target,
                dtype=torch.float32,
                device=robot.device,
            )
            robot.set_joint_position_target(hand_targets, joint_ids=hand_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)


def _build_controller(*, controller_type: str, robot, object_asset, scene, sim, fixed_gripper_width: float):
    if controller_type == "admittance":
        return FR3AdmittanceController(
            robot=robot,
            scene=scene,
            sim=sim,
            fixed_gripper_width=fixed_gripper_width,
        )
    return FR3MoveToPoseController(
        robot=robot,
        cube=object_asset,
        scene=scene,
        sim=sim,
        fixed_gripper_width=fixed_gripper_width,
    )


def move_to_pregrasp(
    *,
    sim,
    scene,
    robot,
    object_asset,
    world_grasp: WorldFrameGraspCandidate,
    controller_type: str,
    fixed_gripper_width: float,
) -> tuple[bool, str, tuple[float, float, float], tuple[float, float, float, float]]:
    """Move the arm to the grasp pregrasp pose."""

    context = FR3MotionContext(
        robot=robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=fixed_gripper_width,
    )
    if controller_type == "admittance":
        controller = _build_controller(
            controller_type=controller_type,
            robot=robot,
            object_asset=object_asset,
            scene=scene,
            sim=sim,
            fixed_gripper_width=fixed_gripper_width,
        )
        result = controller.move_to_pose(
            position_w=world_grasp.pregrasp_position_w,
            orientation_xyzw=world_grasp.orientation_xyzw,
        )
        if not result.success:
            return False, result.message, world_grasp.pregrasp_position_w, world_grasp.orientation_xyzw
    else:
        solver = GoalIKSolver(context, locked_joint_names=("panda_joint1", "fr3_joint1"))
        goal_q = solver.solve(
            PoseCommand(
                position_w=world_grasp.pregrasp_position_w,
                orientation_xyzw=world_grasp.orientation_xyzw,
            ),
            restore_start_state=False,
        )
        if goal_q is None:
            return (
                False,
                "No IK solution found for the requested pregrasp pose.",
                world_grasp.pregrasp_position_w,
                world_grasp.orientation_xyzw,
            )

    tcp_position_w, tcp_quat_w = context.get_tcp_pose_w()
    actual_tcp_position_w = tuple(float(v) for v in tcp_position_w[0].tolist())
    actual_tcp_orientation_xyzw = (
        float(tcp_quat_w[0, 1].item()),
        float(tcp_quat_w[0, 2].item()),
        float(tcp_quat_w[0, 3].item()),
        float(tcp_quat_w[0, 0].item()),
    )
    return True, "ok", actual_tcp_position_w, actual_tcp_orientation_xyzw


def _build_vertical_tcp_waypoints(
    *,
    start_tcp_position_w: tuple[float, float, float],
    target_tcp_z: float,
    num_waypoints: int,
) -> list[tuple[float, float, float]]:
    start_x, start_y, start_z = start_tcp_position_w
    if num_waypoints <= 1:
        return [(start_x, start_y, float(target_tcp_z))]
    return [
        (
            start_x,
            start_y,
            float(start_z + (target_tcp_z - start_z) * (step_idx / num_waypoints)),
        )
        for step_idx in range(1, num_waypoints + 1)
    ]


def _execute_tcp_waypoint_sequence(
    *,
    controller,
    tcp_positions_w: list[tuple[float, float, float]],
    tcp_orientation_xyzw: tuple[float, float, float, float],
) -> bool:
    pose_sequence = []
    for tcp_position_w in tcp_positions_w:
        grasp_position_w, grasp_orientation_xyzw = FR3MotionContext.tcp_pose_to_grasp_pose(
            tcp_position_w,
            tcp_orientation_xyzw,
        )
        pose_sequence.append((grasp_position_w, grasp_orientation_xyzw))
    if hasattr(controller, "move_through_poses"):
        return bool(controller.move_through_poses(pose_sequence).success)

    for grasp_position_w, grasp_orientation_xyzw in pose_sequence:
        result = controller.move_to_pose(position_w=grasp_position_w, orientation_xyzw=grasp_orientation_xyzw)
        if not result.success:
            return False
    return True


def _command_gripper_width(
    *,
    sim,
    scene,
    robot,
    width: float,
    duration_s: float,
) -> None:
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    hand_joint_names = tuple(
        name for name in robot.joint_names if name.startswith(("panda_finger_joint", "fr3_finger_joint"))
    )
    if not hand_joint_names:
        return
    hand_joint_ids = [joint_name_to_idx[name] for name in hand_joint_names]
    physics_dt = sim.get_physics_dt()
    steps = max(1, int(duration_s / physics_dt))
    hand_targets = torch.full(
        (1, len(hand_joint_ids)),
        float(width),
        dtype=torch.float32,
        device=robot.device,
    )
    for _ in range(steps):
        robot.set_joint_position_target(hand_targets, joint_ids=hand_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)


def _servo_tcp_line(
    *,
    sim,
    scene,
    robot,
    start_tcp_position_w: tuple[float, float, float],
    target_tcp_position_w: tuple[float, float, float],
    tcp_orientation_xyzw: tuple[float, float, float, float],
    fixed_gripper_width: float,
    duration_s: float,
    max_joint_delta_rad: float = 0.015,
    z_tolerance_m: float = 0.015,
    max_extra_duration_s: float = 8.0,
    lock_joint1: bool = True,
    carried_object=None,
) -> bool:
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

    context = FR3MotionContext(
        robot=robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=fixed_gripper_width,
    )
    ik_controller = DifferentialIKController(
        cfg=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1,
        device=context.device,
    )
    steps = max(1, int(float(duration_s) / context.physics_dt))
    carried_offset_w = None
    carried_quat_w = None
    if carried_object is not None:
        tcp_pos_w, _ = context.get_tcp_pose_w()
        object_pose_w = carried_object.data.root_link_pose_w.clone()
        carried_offset_w = object_pose_w[:, :3] - tcp_pos_w
        carried_quat_w = object_pose_w[:, 3:7].clone()
    max_steps = steps + max(0, int(float(max_extra_duration_s) / context.physics_dt))
    for step_idx in range(1, max_steps + 1):
        alpha = min(float(step_idx) / float(steps), 1.0)
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        tcp_position_w = tuple(
            float((1.0 - smooth_alpha) * start_tcp_position_w[i] + smooth_alpha * target_tcp_position_w[i])
            for i in range(3)
        )
        grasp_position_w, grasp_orientation_xyzw = FR3MotionContext.tcp_pose_to_grasp_pose(
            tcp_position_w,
            tcp_orientation_xyzw,
        )
        q_before = context.get_arm_q()
        q_des = context.command_pose_via_differential_ik(
            ik_controller,
            PoseCommand(position_w=grasp_position_w, orientation_xyzw=grasp_orientation_xyzw),
        )
        q_limited = q_before + torch.clamp(q_des - q_before, min=-max_joint_delta_rad, max=max_joint_delta_rad)
        if lock_joint1 and q_limited.shape[1] > 0:
            q_limited[:, 0] = q_before[:, 0]
        context.command_arm(q_limited)
        context.command_fixed_gripper()
        scene.write_data_to_sim()
        sim.step()
        scene.update(context.physics_dt)
        tcp_pos_w, _ = context.get_tcp_pose_w()
        if carried_object is not None and carried_offset_w is not None and carried_quat_w is not None:
            object_pose_w = torch.cat((tcp_pos_w + carried_offset_w, carried_quat_w), dim=1)
            carried_object.write_root_pose_to_sim(object_pose_w)
            zero_velocity = torch.zeros((1, 6), dtype=torch.float32, device=robot.device)
            carried_object.write_root_velocity_to_sim(zero_velocity)
        actual_tcp_z = float(tcp_pos_w[0, 2].item())
        if alpha >= 1.0 and abs(actual_tcp_z - target_tcp_position_w[2]) <= z_tolerance_m:
            return True
    return False


def execute_vertical_pick_sequence(
    *,
    sim,
    scene,
    robot,
    object_asset,
    start_tcp_position_w: tuple[float, float, float],
    start_tcp_orientation_xyzw: tuple[float, float, float, float],
    target_tcp_z: float,
    target_tcp_orientation_xyzw: tuple[float, float, float, float],
    open_gripper_width: float,
    closed_gripper_width: float,
    controller_type: str,
) -> PickExecutionResult:
    """Execute a direct grasp move, close, and direct retreat sequence."""

    target_tcp_position_w = (
        float(start_tcp_position_w[0]),
        float(start_tcp_position_w[1]),
        float(target_tcp_z),
    )
    if not _servo_tcp_line(
        sim=sim,
        scene=scene,
        robot=robot,
        start_tcp_position_w=start_tcp_position_w,
        target_tcp_position_w=target_tcp_position_w,
        tcp_orientation_xyzw=target_tcp_orientation_xyzw,
        fixed_gripper_width=open_gripper_width,
        duration_s=3.0,
    ):
        return PickExecutionResult(False, "approach_failed", "Servo grasp approach failed before gripper close.")

    _command_gripper_width(
        sim=sim,
        scene=scene,
        robot=robot,
        width=closed_gripper_width,
        duration_s=1.2,
    )

    if not _servo_tcp_line(
        sim=sim,
        scene=scene,
        robot=robot,
        start_tcp_position_w=target_tcp_position_w,
        target_tcp_position_w=start_tcp_position_w,
        tcp_orientation_xyzw=target_tcp_orientation_xyzw,
        fixed_gripper_width=closed_gripper_width,
        duration_s=3.0,
    ):
        return PickExecutionResult(False, "retreat_failed", "Vertical retreat failed after gripper close.")
    return PickExecutionResult(True, "ok", "Pick sequence executed.")


def execute_pick_from_world_grasp(
    *,
    sim,
    scene,
    robot,
    object_asset,
    world_grasp: WorldFrameGraspCandidate,
    controller_type: str,
    fixed_gripper_width: float,
    closed_gripper_width: float,
    pregrasp_only: bool,
) -> PickExecutionResult:
    """Run pregrasp and optionally a simple vertical pick sequence."""

    floor_clearance_m = 0.05
    if world_grasp.pregrasp_position_w[2] <= floor_clearance_m:
        return PickExecutionResult(
            False,
            "invalid_pregrasp",
            (
                "Requested pregrasp is too close to or below the floor: "
                f"pregrasp_position_w={world_grasp.pregrasp_position_w} required_min_z={floor_clearance_m:.3f}"
            ),
        )

    ok, message, actual_tcp_position_w, actual_tcp_orientation_xyzw = move_to_pregrasp(
        sim=sim,
        scene=scene,
        robot=robot,
        object_asset=object_asset,
        world_grasp=world_grasp,
        controller_type=controller_type,
        fixed_gripper_width=fixed_gripper_width,
    )
    if not ok:
        return PickExecutionResult(False, "pregrasp_failed", message)
    if pregrasp_only:
        return PickExecutionResult(True, "ok", "Pregrasp reached.")

    grasp_tcp_position_w, grasp_tcp_orientation_xyzw = FR3MotionContext.grasp_pose_to_tcp_pose(
        world_grasp.position_w,
        world_grasp.orientation_xyzw,
    )
    min_tcp_z_m = 0.005
    if grasp_tcp_position_w[2] <= min_tcp_z_m:
        grasp_tcp_position_w = (
            grasp_tcp_position_w[0],
            grasp_tcp_position_w[1],
            min_tcp_z_m,
        )

    return execute_vertical_pick_sequence(
        sim=sim,
        scene=scene,
        robot=robot,
        object_asset=object_asset,
        start_tcp_position_w=actual_tcp_position_w,
        start_tcp_orientation_xyzw=actual_tcp_orientation_xyzw,
        target_tcp_z=grasp_tcp_position_w[2],
        target_tcp_orientation_xyzw=actual_tcp_orientation_xyzw,
        open_gripper_width=world_grasp.gripper_width / 2.0,
        closed_gripper_width=closed_gripper_width,
        controller_type=controller_type,
    )
