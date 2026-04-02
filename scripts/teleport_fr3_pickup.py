"""Standalone teleport-based FR3 pickup debug script.

This script is intentionally separate from the planner/admittance launch path.
It solves joint states for pregrasp/grasp, teleports the arm to those poses,
then performs a real gripper close and a real joint-space retreat.
"""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleport the FR3 to grasp poses for isolated pickup debugging.")
parser.add_argument(
    "--fr3-usd",
    type=str,
    default="",
    help="Optional override for the Franka Research 3 USD asset path or Omniverse URL.",
)
parser.add_argument(
    "--run-seconds",
    type=float,
    default=0.0,
    help="Optional wall-clock duration to keep the simulation alive after execution. Use 0 for until interrupted.",
)
parser.add_argument(
    "--grasp-face",
    type=str,
    default="pos_z",
    choices=("pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z"),
    help="Cube face grasp to use.",
)
parser.add_argument(
    "--grasp-label",
    type=str,
    default="",
    choices=("+x", "-x", "+y", "-y", "+z", "-z"),
    help="Optional raw face label override. Prefer --grasp-face for normal use.",
)
parser.add_argument(
    "--pregrasp-offset",
    type=float,
    default=0.20,
    help="Pregrasp offset in meters applied opposite the grasp approach direction.",
)
parser.add_argument(
    "--tcp-to-grasp-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, -0.045),
    metavar=("X", "Y", "Z"),
    help="Override the fixed TCP-to-grasp-center offset used for pose conversion.",
)
parser.add_argument(
    "--close-duration",
    type=float,
    default=0.6,
    help="Duration in seconds for the physical gripper close.",
)
parser.add_argument(
    "--retreat-duration",
    type=float,
    default=1.5,
    help="Duration in seconds for the real joint-space retreat back to pregrasp.",
)
parser.add_argument(
    "--hold-after-close",
    type=float,
    default=0.2,
    help="Extra hold time in seconds at grasp after the gripper closes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

from grasp_planning import CubeFaceGraspGenerator  # noqa: E402
from grasp_planning.envs import make_fr3_cube_scene_cfg  # noqa: E402
from grasp_planning.envs.fr3_cube_env import (  # noqa: E402
    DEFAULT_ARM_START_JOINT_POS,
    DEFAULT_CUBE_CFG,
    DEFAULT_HAND_START_JOINT_POS,
)
from grasp_planning.planning.fr3_motion_context import FR3MotionContext  # noqa: E402
from grasp_planning.planning.types import PoseCommand  # noqa: E402
from grasp_planning.scene_defaults import (  # noqa: E402
    CUBE_ORIENTATION_XYZW,
    CUBE_POSITION,
    ROBOT_BASE_ORIENTATION_XYZW,
    ROBOT_BASE_POSITION,
)


def resolve_fr3_usd_path() -> str:
    """Return the user-supplied FR3 asset path or the built-in Isaac asset URL."""

    if args_cli.fr3_usd:
        return args_cli.fr3_usd

    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Unable to resolve Isaac asset root for the built-in FR3 asset.")
    return assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd"


def configure_grasp_tcp_calibration() -> None:
    """Apply the runtime TCP-to-grasp-center calibration used by pose conversion."""

    tcp_to_grasp_offset = tuple(float(value) for value in args_cli.tcp_to_grasp_offset)
    FR3MotionContext._TCP_TO_GRASP_CENTER_OFFSET = tcp_to_grasp_offset
    print(f"[INFO]: Using TCP-to-grasp-center offset {tcp_to_grasp_offset}", flush=True)


def build_scene() -> tuple[sim_utils.SimulationContext, InteractiveScene]:
    """Create the simulator and populate the FR3 cube scene."""

    print("[INFO]: Creating simulation context...", flush=True)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim._app_control_on_stop_handle = None
    sim._disable_app_control_on_stop_handle = True
    sim.set_camera_view([1.6, -1.2, 1.0], [0.35, 0.0, 0.3])
    fr3_usd_path = resolve_fr3_usd_path()
    print(f"[INFO]: FR3 asset path: {fr3_usd_path}", flush=True)

    scene_cfg = make_fr3_cube_scene_cfg(
        fr3_asset_path=fr3_usd_path,
        cube_position=CUBE_POSITION,
        cube_orientation_xyzw=CUBE_ORIENTATION_XYZW,
        robot_base_position=ROBOT_BASE_POSITION,
        robot_base_orientation_xyzw=ROBOT_BASE_ORIENTATION_XYZW,
    )
    scene = InteractiveScene(scene_cfg)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    sim.reset()
    scene.reset()
    print("[INFO]: Scene ready.", flush=True)
    return sim, scene


def drive_robot_to_start_pose(sim, scene) -> None:
    """Actively settle the FR3 into a safe home pose before solving debug poses."""

    robot = scene["robot"]
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    arm_joint_names = tuple(DEFAULT_ARM_START_JOINT_POS.keys())
    arm_joint_ids = [joint_name_to_idx[name] for name in arm_joint_names]
    arm_targets = torch.tensor(
        [[DEFAULT_ARM_START_JOINT_POS[name] for name in arm_joint_names]],
        dtype=torch.float32,
        device=robot.device,
    )
    hand_joint_names = tuple(name for name in robot.joint_names if name.startswith("fr3_finger_joint"))
    hand_joint_ids = [joint_name_to_idx[name] for name in hand_joint_names]
    hand_target = float(DEFAULT_HAND_START_JOINT_POS["fr3_finger_joint.*"])
    physics_dt = sim.get_physics_dt()

    for _ in range(max(1, int(1.5 / physics_dt))):
        robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_ids)
        if hand_joint_ids:
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


def build_selected_grasp():
    """Return the requested face grasp candidate for the configured cube."""

    label_from_face = {
        "pos_x": "+x",
        "neg_x": "-x",
        "pos_y": "+y",
        "neg_y": "-y",
        "pos_z": "+z",
        "neg_z": "-z",
    }
    requested_label = args_cli.grasp_label or label_from_face[args_cli.grasp_face]
    grasp_generator = CubeFaceGraspGenerator(
        cube_size=DEFAULT_CUBE_CFG.size,
        pregrasp_offset=args_cli.pregrasp_offset,
    )
    candidates = grasp_generator.generate(
        cube_position_w=CUBE_POSITION,
        cube_orientation_xyzw=CUBE_ORIENTATION_XYZW,
        robot_base_position_w=ROBOT_BASE_POSITION,
    )
    for grasp in candidates:
        if grasp.label == requested_label:
            return grasp
    raise RuntimeError(f"Cube grasp generator did not produce the requested grasp '{requested_label}'.")


def snapshot_cube_state(cube) -> torch.Tensor:
    """Snapshot the cube root state so solving motions can be undone."""

    return cube.data.root_state_w.clone()


def restore_cube_state(*, sim, scene, cube, root_state_w: torch.Tensor) -> None:
    """Restore the cube pose/velocity after IK solving."""

    cube.write_root_state_to_sim(root_state_w)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())


def solve_pose_to_q(context: FR3MotionContext, pose: PoseCommand) -> torch.Tensor:
    """Solve a pose into arm joints without stepping physics."""

    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

    q_start = context.get_arm_q()
    q_current = q_start.clone()
    zero_vel = torch.zeros_like(q_current)
    lower, upper = context.get_joint_limits()
    cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(cfg=cfg, num_envs=1, device=context.device)
    best_q = q_start.clone()
    best_position_error = float("inf")
    best_orientation_error = float("inf")
    target_position_w = torch.tensor([pose.position_w], dtype=torch.float32, device=context.device)
    target_orientation_xyzw = torch.tensor([pose.orientation_xyzw], dtype=torch.float32, device=context.device)

    try:
        for _ in range(220):
            context.robot.write_joint_state_to_sim(q_current, zero_vel, joint_ids=context.arm_joint_ids)
            context.scene.update(0.0)

            desired_q = context.command_pose_via_differential_ik(ik_controller, pose)
            if context.joint_limits_are_usable(lower, upper):
                desired_q = torch.max(torch.min(desired_q, upper), lower)
            context.robot.write_joint_state_to_sim(desired_q, zero_vel, joint_ids=context.arm_joint_ids)
            context.scene.update(0.0)

            pos_error, rot_error = context.compute_pose_error(target_position_w, target_orientation_xyzw)
            pos_norm = float(torch.linalg.norm(pos_error).item())
            rot_norm = float(torch.linalg.norm(rot_error).item())
            q_current = desired_q.clone()
            if pos_norm + rot_norm < best_position_error + best_orientation_error:
                best_position_error = pos_norm
                best_orientation_error = rot_norm
                best_q = q_current.clone()
            if pos_norm <= 0.025 and rot_norm <= 0.08:
                print(
                    f"[INFO]: Offline IK converged with position_error={pos_norm:.4f} orientation_error={rot_norm:.4f}",
                    flush=True,
                )
                return q_current.detach().clone()
    finally:
        context.robot.write_joint_state_to_sim(q_start, zero_vel, joint_ids=context.arm_joint_ids)
        context.scene.update(0.0)

    if best_position_error <= 0.045 and best_orientation_error <= 0.14:
        print(
            "[WARN]: Offline IK accepted approximate solution. "
            f"best_position_error={best_position_error:.4f} best_orientation_error={best_orientation_error:.4f}",
            flush=True,
        )
        return best_q.detach().clone()

    raise RuntimeError(
        "Failed to solve teleport joint state for pose "
        f"position={pose.position_w} orientation_xyzw={pose.orientation_xyzw}. "
        f"best_position_error={best_position_error:.4f} best_orientation_error={best_orientation_error:.4f}"
    )


def teleport_arm_and_hand(*, sim, scene, context: FR3MotionContext, arm_q: torch.Tensor, hand_width: float) -> None:
    """Hard-set the arm and hand joints to the requested state."""

    arm_vel = torch.zeros_like(arm_q)
    context.robot.write_joint_state_to_sim(arm_q, arm_vel, joint_ids=context.arm_joint_ids)
    if context.hand_joint_ids.numel() > 0:
        hand_q = torch.full(
            (1, int(context.hand_joint_ids.numel())),
            float(hand_width),
            dtype=torch.float32,
            device=context.device,
        )
        hand_vel = torch.zeros_like(hand_q)
        context.robot.write_joint_state_to_sim(hand_q, hand_vel, joint_ids=context.hand_joint_ids)
        context.robot.set_joint_position_target(hand_q, joint_ids=context.hand_joint_ids)
    context.robot.set_joint_position_target(arm_q, joint_ids=context.arm_joint_ids)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())


def command_gripper_width(*, sim, scene, context: FR3MotionContext, width: float, duration_s: float) -> None:
    """Animate the fingers to the requested width."""

    if context.hand_joint_ids.numel() == 0:
        return
    hand_targets = torch.full(
        (1, int(context.hand_joint_ids.numel())),
        float(width),
        dtype=torch.float32,
        device=context.device,
    )
    arm_hold = context.get_arm_q()
    physics_dt = sim.get_physics_dt()
    steps = max(1, int(duration_s / physics_dt))
    for _ in range(steps):
        context.robot.set_joint_position_target(arm_hold, joint_ids=context.arm_joint_ids)
        context.robot.set_joint_position_target(hand_targets, joint_ids=context.hand_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)


def animate_arm_to_q(
    *, sim, scene, context: FR3MotionContext, target_q: torch.Tensor, hand_width: float, duration_s: float
) -> None:
    """Retreat with real motion by interpolating joint targets over time."""

    start_q = context.get_arm_q()
    physics_dt = sim.get_physics_dt()
    steps = max(1, int(duration_s / physics_dt))
    hand_targets = (
        torch.full(
            (1, int(context.hand_joint_ids.numel())),
            float(hand_width),
            dtype=torch.float32,
            device=context.device,
        )
        if context.hand_joint_ids.numel() > 0
        else None
    )
    for step_idx in range(1, steps + 1):
        alpha = float(step_idx) / float(steps)
        arm_targets = start_q + (target_q - start_q) * alpha
        context.robot.set_joint_position_target(arm_targets, joint_ids=context.arm_joint_ids)
        if hand_targets is not None:
            context.robot.set_joint_position_target(hand_targets, joint_ids=context.hand_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)


def log_current_tcp_pose(context: FR3MotionContext, label: str) -> None:
    """Print the current TCP pose for debugging."""

    tcp_position_w, tcp_quat_w = context.get_tcp_pose_w()
    tcp_position = tuple(float(v) for v in tcp_position_w[0].tolist())
    tcp_orientation_xyzw = (
        float(tcp_quat_w[0, 1].item()),
        float(tcp_quat_w[0, 2].item()),
        float(tcp_quat_w[0, 3].item()),
        float(tcp_quat_w[0, 0].item()),
    )
    print(
        f"[INFO]: {label} TCP pose position={tcp_position} orientation_xyzw={tcp_orientation_xyzw}",
        flush=True,
    )


def run() -> None:
    """Teleport to pregrasp and grasp, then physically close and retreat."""

    configure_grasp_tcp_calibration()
    sim, scene = build_scene()
    physics_dt = sim.get_physics_dt()
    for _ in range(max(1, int(0.1 / physics_dt))):
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)

    drive_robot_to_start_pose(sim, scene)
    robot = scene["robot"]
    cube = scene["cube"]
    grasp = build_selected_grasp()
    context = FR3MotionContext(
        robot=robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=float(DEFAULT_HAND_START_JOINT_POS["fr3_finger_joint.*"]),
    )
    print(
        "[INFO]: Teleport pickup grasp "
        f"label={grasp.label} pregrasp={grasp.pregrasp_position_w} grasp={grasp.position_w} "
        f"orientation_xyzw={grasp.orientation_xyzw} gripper_width={grasp.gripper_width}",
        flush=True,
    )
    if grasp.pregrasp_position_w[2] <= 0.05:
        raise RuntimeError(
            "Requested pregrasp is too close to or below the floor: "
            f"grasp_face={args_cli.grasp_face}, pregrasp_position_w={grasp.pregrasp_position_w}."
        )

    cube_root_state_w = snapshot_cube_state(cube)
    pregrasp_q = solve_pose_to_q(
        context,
        PoseCommand(position_w=grasp.pregrasp_position_w, orientation_xyzw=grasp.orientation_xyzw),
    )
    restore_cube_state(sim=sim, scene=scene, cube=cube, root_state_w=cube_root_state_w)

    grasp_q = solve_pose_to_q(
        context,
        PoseCommand(position_w=grasp.position_w, orientation_xyzw=grasp.orientation_xyzw),
    )
    restore_cube_state(sim=sim, scene=scene, cube=cube, root_state_w=cube_root_state_w)

    open_width = float(grasp.gripper_width / 2.0)
    teleport_arm_and_hand(sim=sim, scene=scene, context=context, arm_q=pregrasp_q, hand_width=open_width)
    log_current_tcp_pose(context, "Teleported pregrasp")

    teleport_arm_and_hand(sim=sim, scene=scene, context=context, arm_q=grasp_q, hand_width=open_width)
    log_current_tcp_pose(context, "Teleported grasp")

    command_gripper_width(
        sim=sim,
        scene=scene,
        context=context,
        width=0.0,
        duration_s=args_cli.close_duration,
    )
    if args_cli.hold_after_close > 0.0:
        command_gripper_width(
            sim=sim,
            scene=scene,
            context=context,
            width=0.0,
            duration_s=args_cli.hold_after_close,
        )

    animate_arm_to_q(
        sim=sim,
        scene=scene,
        context=context,
        target_q=pregrasp_q,
        hand_width=0.0,
        duration_s=args_cli.retreat_duration,
    )
    log_current_tcp_pose(context, "Retreated")

    elapsed_s = 0.0
    while args_cli.run_seconds <= 0.0 or elapsed_s < args_cli.run_seconds:
        try:
            if sim.is_stopped():
                break
            if not sim.is_playing():
                sim.step()
                continue
            scene.write_data_to_sim()
            sim.step()
            scene.update(physics_dt)
            elapsed_s += physics_dt
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
    except Exception:
        print("[ERROR]: Unhandled exception in teleport_fr3_pickup.py", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        simulation_app.close()
