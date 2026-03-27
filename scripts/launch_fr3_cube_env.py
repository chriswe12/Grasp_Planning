"""Launch an Isaac Lab scene with a ground plane, FR3 robot, and graspable cube."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Launch an FR3 + cube motion-planning environment in Isaac Lab.")
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
    help="Optional wall-clock duration to keep the simulation alive. Use 0 for until interrupted.",
)
parser.add_argument(
    "--target-pos",
    type=float,
    nargs=3,
    default=(0.45, 0.0, 0.35),
    metavar=("X", "Y", "Z"),
    help="Primary target TCP position in world coordinates.",
)
parser.add_argument(
    "--target-quat",
    type=float,
    nargs=4,
    default=(0.0, 1.0, 0.0, 0.0),
    metavar=("X", "Y", "Z", "W"),
    help="Target TCP orientation as a quaternion in xyzw order.",
)
parser.add_argument(
    "--test-multi-targets",
    action="store_true",
    help="Run a short three-target planner smoke test instead of a single pose.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
import torch  # noqa: E402

from grasp_planning import CubeFaceGraspGenerator, FR3MoveToPoseController  # noqa: E402
from grasp_planning.envs import make_fr3_cube_scene_cfg  # noqa: E402
from grasp_planning.envs.fr3_cube_env import DEFAULT_ARM_START_JOINT_POS, DEFAULT_CUBE_CFG, DEFAULT_HAND_START_JOINT_POS  # noqa: E402
from grasp_planning.planning.fr3_motion_context import FR3MotionContext  # noqa: E402


ROBOT_BASE_POSITION = (0.0, 0.0, 0.0)
ROBOT_BASE_ORIENTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
CUBE_POSITION = (0.45, 0.0, 0.025)
CUBE_ORIENTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
SMOKE_TEST_TCP_TARGETS = (
    (-0.30, 0.00, 0.60),
    (-0.18, 0.14, 0.60),
    (-0.18, -0.14, 0.58),
)


def resolve_fr3_usd_path() -> str:
    """Return the user-supplied FR3 asset path or the built-in Isaac asset URL."""

    if args_cli.fr3_usd:
        return args_cli.fr3_usd

    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Unable to resolve Isaac asset root for the built-in FR3 asset.")
    return assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd"


def build_scene() -> tuple[sim_utils.SimulationContext, InteractiveScene]:
    """Create the simulator and populate the FR3 cube scene."""

    print("[INFO]: Creating simulation context...", flush=True)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim._app_control_on_stop_handle = None
    sim._disable_app_control_on_stop_handle = True
    sim.set_camera_view([1.6, -1.2, 1.0], [0.35, 0.0, 0.3])
    print("[INFO]: Resolving FR3 asset path...", flush=True)
    fr3_usd_path = resolve_fr3_usd_path()
    print(f"[INFO]: FR3 asset path: {fr3_usd_path}", flush=True)

    scene_cfg = make_fr3_cube_scene_cfg(
        fr3_asset_path=fr3_usd_path,
        cube_position=CUBE_POSITION,
        cube_orientation_xyzw=CUBE_ORIENTATION_XYZW,
        robot_base_position=ROBOT_BASE_POSITION,
        robot_base_orientation_xyzw=ROBOT_BASE_ORIENTATION_XYZW,
    )
    print("[INFO]: Creating interactive scene...", flush=True)
    scene = InteractiveScene(scene_cfg)
    print("[INFO]: Waiting for stage assets to finish loading...", flush=True)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    print("[INFO]: Resetting simulator...", flush=True)
    sim.reset()
    print("[INFO]: Simulator reset finished.", flush=True)
    print("[INFO]: Resetting scene buffers...", flush=True)
    scene.reset()
    print("[INFO]: Scene ready.", flush=True)
    return sim, scene


def build_target_sequence(
    current_orientation_xyzw: tuple[float, float, float, float],
) -> list[tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    """Return the ordered move-to-pose targets in the grasp frame expected by the controller."""

    if args_cli.test_multi_targets:
        return [
            FR3MotionContext.tcp_pose_to_grasp_pose(
                position,
                current_orientation_xyzw,
            )
            for position in SMOKE_TEST_TCP_TARGETS
        ]
    if tuple(args_cli.target_pos) == (0.45, 0.0, 0.35) and tuple(args_cli.target_quat) == (0.0, 1.0, 0.0, 0.0):
        grasp_generator = CubeFaceGraspGenerator(cube_size=DEFAULT_CUBE_CFG.size)
        grasp = grasp_generator.generate(
            cube_position_w=CUBE_POSITION,
            cube_orientation_xyzw=CUBE_ORIENTATION_XYZW,
            robot_base_position_w=ROBOT_BASE_POSITION,
        )[0]
        pregrasp_position_w = (
            grasp.pregrasp_position_w[0],
            grasp.pregrasp_position_w[1],
            grasp.pregrasp_position_w[2] + 0.2,
        )
        return [(pregrasp_position_w, grasp.orientation_xyzw)]
    return [FR3MotionContext.tcp_pose_to_grasp_pose(tuple(args_cli.target_pos), tuple(args_cli.target_quat))]


def drive_robot_to_start_pose(sim, scene) -> None:
    """Actively settle the FR3 into a safe home pose before planning."""

    robot = scene["robot"]
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    arm_joint_names = tuple(DEFAULT_ARM_START_JOINT_POS.keys())
    arm_joint_ids = [joint_name_to_idx[name] for name in arm_joint_names]
    arm_targets = torch.tensor([[
        DEFAULT_ARM_START_JOINT_POS[name]
        for name in arm_joint_names
    ]], dtype=torch.float32, device=robot.device)
    hand_joint_names = tuple(name for name in robot.joint_names if name.startswith("fr3_finger_joint"))
    hand_target = float(DEFAULT_HAND_START_JOINT_POS["fr3_finger_joint.*"])
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

    settled_joint_pos = {
        name: float(robot.data.joint_pos[0, joint_name_to_idx[name]].item())
        for name in arm_joint_names
    }
    print(f"[INFO]: Settled FR3 start joints: {settled_joint_pos}", flush=True)


def run() -> None:
    """Plan and execute one or more move-to-pose requests, then keep the app alive if requested."""

    sim, scene = build_scene()
    physics_dt = sim.get_physics_dt()
    robot = scene["robot"]
    cube = scene["cube"]
    warmup_steps = max(1, int(0.1 / physics_dt))
    for _ in range(warmup_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)
    drive_robot_to_start_pose(sim, scene)

    controller = FR3MoveToPoseController(robot=robot, cube=cube, scene=scene, sim=sim)
    print(
        "[INFO]: Planner resolved "
        f"ee_body={controller.ee_body_name}, arm_joints={controller.arm_joint_names}, "
        f"hand_joints={controller.hand_joint_names}.",
        flush=True,
    )

    current_tcp_position_w, current_orientation_xyzw = controller.get_current_tcp_pose()
    print(
        "[INFO]: Current TCP pose "
        f"position={current_tcp_position_w} orientation_xyzw={current_orientation_xyzw}",
        flush=True,
    )
    targets = build_target_sequence(current_orientation_xyzw)
    for index, (position_w, orientation_xyzw) in enumerate(targets, start=1):
        if args_cli.test_multi_targets:
            target_tcp_position_w = SMOKE_TEST_TCP_TARGETS[index - 1]
            target_tcp_orientation_xyzw = current_orientation_xyzw
        elif tuple(args_cli.target_pos) == (0.45, 0.0, 0.35) and tuple(args_cli.target_quat) == (0.0, 1.0, 0.0, 0.0):
            target_tcp_position_w, target_tcp_orientation_xyzw = FR3MotionContext.grasp_pose_to_tcp_pose(
                position_w,
                orientation_xyzw,
            )
        else:
            target_tcp_position_w = tuple(args_cli.target_pos)
            target_tcp_orientation_xyzw = tuple(args_cli.target_quat)
        print(
            "[INFO]: Executing target "
            f"{index}/{len(targets)} tcp_position={target_tcp_position_w} "
            f"tcp_orientation_xyzw={target_tcp_orientation_xyzw} "
            f"grasp_position={position_w} grasp_orientation_xyzw={orientation_xyzw}",
            flush=True,
        )
        result = controller.move_to_pose(position_w=position_w, orientation_xyzw=orientation_xyzw)
        print(
            "[INFO]: Move request finished. "
            f"target_index={index}, status={result.status}, success={result.success}, message={result.message}",
            flush=True,
        )
        if not result.success:
            break
        actual_tcp_position_w, actual_tcp_orientation_xyzw = controller.get_current_tcp_pose()
        print(
            "[INFO]: Final TCP pose after target "
            f"{index}: position={actual_tcp_position_w} orientation_xyzw={actual_tcp_orientation_xyzw} "
            f"(initial_tcp_position={current_tcp_position_w})",
            flush=True,
        )
        for _ in range(max(1, int(0.25 / physics_dt))):
            scene.write_data_to_sim()
            sim.step()
            scene.update(physics_dt)

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
    finally:
        simulation_app.close()
