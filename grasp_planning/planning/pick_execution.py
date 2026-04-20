"""Shared FR3 pick-execution helpers for debug grasp attempts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate
from grasp_planning.planning.fr3_motion_context import FR3MotionContext
from grasp_planning.robot_naming import arm_joint_names_for_prefix, infer_robot_name_prefix_from_joint_names

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

    from grasp_planning.envs.fr3_cube_env import DEFAULT_ARM_START_JOINT_POS, DEFAULT_HAND_START_WIDTH

    robot = scene["robot"]
    robot_name_prefix = infer_robot_name_prefix_from_joint_names(tuple(robot.joint_names))
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    arm_joint_names = arm_joint_names_for_prefix(robot_name_prefix)
    arm_joint_ids = [joint_name_to_idx[name] for name in arm_joint_names]
    arm_targets = torch.tensor(
        [[DEFAULT_ARM_START_JOINT_POS[name.replace(f"{robot_name_prefix}_", "fr3_", 1)] for name in arm_joint_names]],
        dtype=torch.float32,
        device=robot.device,
    )
    hand_joint_names = tuple(name for name in robot.joint_names if name.startswith(f"{robot_name_prefix}_finger_joint"))
    hand_target = float(DEFAULT_HAND_START_WIDTH)
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
        solver = GoalIKSolver(context)
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
    for tcp_position_w in tcp_positions_w:
        grasp_position_w, grasp_orientation_xyzw = FR3MotionContext.tcp_pose_to_grasp_pose(
            tcp_position_w,
            tcp_orientation_xyzw,
        )
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
    robot_name_prefix = infer_robot_name_prefix_from_joint_names(tuple(robot.joint_names))
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    hand_joint_names = tuple(name for name in robot.joint_names if name.startswith(f"{robot_name_prefix}_finger_joint"))
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


def execute_vertical_pick_sequence(
    *,
    sim,
    scene,
    robot,
    object_asset,
    start_tcp_position_w: tuple[float, float, float],
    start_tcp_orientation_xyzw: tuple[float, float, float, float],
    target_tcp_z: float,
    open_gripper_width: float,
    closed_gripper_width: float,
    controller_type: str,
) -> PickExecutionResult:
    """Execute a simple vertical descent-close-retreat sequence."""

    descent_waypoints = _build_vertical_tcp_waypoints(
        start_tcp_position_w=start_tcp_position_w,
        target_tcp_z=target_tcp_z,
        num_waypoints=12,
    )
    open_controller = _build_controller(
        controller_type=controller_type,
        robot=robot,
        object_asset=object_asset,
        scene=scene,
        sim=sim,
        fixed_gripper_width=open_gripper_width,
    )
    if not _execute_tcp_waypoint_sequence(
        controller=open_controller,
        tcp_positions_w=descent_waypoints,
        tcp_orientation_xyzw=start_tcp_orientation_xyzw,
    ):
        return PickExecutionResult(False, "approach_failed", "Vertical approach failed before gripper close.")

    _command_gripper_width(
        sim=sim,
        scene=scene,
        robot=robot,
        width=closed_gripper_width,
        duration_s=0.6,
    )

    retreat_controller = _build_controller(
        controller_type=controller_type,
        robot=robot,
        object_asset=object_asset,
        scene=scene,
        sim=sim,
        fixed_gripper_width=closed_gripper_width,
    )
    retreat_waypoints = list(reversed(descent_waypoints[:-1])) + [start_tcp_position_w]
    if not _execute_tcp_waypoint_sequence(
        controller=retreat_controller,
        tcp_positions_w=retreat_waypoints,
        tcp_orientation_xyzw=start_tcp_orientation_xyzw,
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
    return execute_vertical_pick_sequence(
        sim=sim,
        scene=scene,
        robot=robot,
        object_asset=object_asset,
        start_tcp_position_w=actual_tcp_position_w,
        start_tcp_orientation_xyzw=actual_tcp_orientation_xyzw,
        target_tcp_z=world_grasp.position_w[2],
        open_gripper_width=world_grasp.gripper_width / 2.0,
        closed_gripper_width=closed_gripper_width,
        controller_type=controller_type,
    )
