"""Launch an Isaac Lab scene with a ground plane, FR3 robot, and graspable cube."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Launch an FR3 + cube motion-planning environment in Isaac Lab.")
parser.add_argument(
    "--fr3-usd",
    type=str,
    default="",
    help="Optional override for the Franka Research 3 USD asset path or Omniverse URL.",
)
parser.add_argument(
    "--controller",
    type=str,
    default="planner",
    choices=("planner", "admittance"),
    help="Execution controller: conservative joint-space planner or Isaac-side admittance controller.",
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
parser.add_argument(
    "--grasp-face",
    type=str,
    default="pos_z",
    choices=("pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z"),
    help="Cube face grasp to use for the default pregrasp-only debug path.",
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
    "--pregrasp-only",
    action="store_true",
    help="Stop after reaching pregrasp instead of running the full approach-close-retreat pickup sequence.",
)
parser.add_argument(
    "--tcp-to-grasp-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, -0.045),
    metavar=("X", "Y", "Z"),
    help="Override the fixed TCP-to-grasp-center offset used for pose conversion and pickup execution.",
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

from grasp_planning import (  # noqa: E402
    CubeFaceGraspGenerator,
    FR3AdmittanceController,
    FR3MoveToPoseController,
    FR3PickController,
)
from grasp_planning.envs import make_fr3_cube_scene_cfg  # noqa: E402
from grasp_planning.envs.fr3_cube_env import (  # noqa: E402
    DEFAULT_ARM_START_JOINT_POS,
    DEFAULT_CUBE_CFG,
    DEFAULT_HAND_START_WIDTH,
)
from grasp_planning.planning.fr3_motion_context import FR3MotionContext  # noqa: E402
from grasp_planning.planning.goal_ik import GoalIKSolver  # noqa: E402
from grasp_planning.planning.types import PoseCommand  # noqa: E402
from grasp_planning.robot_naming import (  # noqa: E402
    arm_joint_names_for_prefix,
    infer_robot_name_prefix_from_joint_names,
)
from grasp_planning.scene_defaults import (  # noqa: E402
    CUBE_ORIENTATION_XYZW,
    CUBE_POSITION,
    ROBOT_BASE_ORIENTATION_XYZW,
    ROBOT_BASE_POSITION,
)

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


def configure_grasp_tcp_calibration() -> None:
    """Apply the runtime TCP-to-grasp-center calibration used by planning and pickup."""

    tcp_to_grasp_offset = tuple(float(value) for value in args_cli.tcp_to_grasp_offset)
    FR3MotionContext._TCP_TO_GRASP_CENTER_OFFSET = tcp_to_grasp_offset
    FR3PickController._TCP_TO_GRASP_CENTER_OFFSET = tcp_to_grasp_offset
    print(
        f"[INFO]: Using TCP-to-grasp-center offset {tcp_to_grasp_offset}",
        flush=True,
    )


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
    return [FR3MotionContext.tcp_pose_to_grasp_pose(tuple(args_cli.target_pos), tuple(args_cli.target_quat))]


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


def run_pick_sequence(sim, scene, robot, cube) -> None:
    """Run the debug pick flow: pregrasp first, then optionally approach, close, and retreat."""

    grasp = build_selected_grasp()
    floor_clearance_m = 0.05
    if grasp.pregrasp_position_w[2] <= floor_clearance_m:
        raise RuntimeError(
            "Requested pregrasp is too close to or below the floor: "
            f"grasp_face={args_cli.grasp_face}, pregrasp_position_w={grasp.pregrasp_position_w}, "
            f"required_min_z={floor_clearance_m:.3f}. Increase --pregrasp-offset or choose another face."
        )
    print(
        "[INFO]: Selected debug grasp "
        f"label={grasp.label} pregrasp={grasp.pregrasp_position_w} grasp={grasp.position_w} "
        f"normal_w={grasp.normal_w} orientation_xyzw={grasp.orientation_xyzw} gripper_width={grasp.gripper_width}",
        flush=True,
    )
    fixed_gripper_width = float(DEFAULT_HAND_START_WIDTH)
    context = FR3MotionContext(
        robot=robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=fixed_gripper_width,
    )
    if args_cli.controller == "admittance":
        pregrasp_controller = FR3AdmittanceController(
            robot=robot,
            scene=scene,
            sim=sim,
            fixed_gripper_width=fixed_gripper_width,
        )
        print(
            "[INFO]: Pregrasp controller resolved "
            f"type=admittance ee_body={pregrasp_controller.ee_body_name}, "
            f"arm_joints={pregrasp_controller.arm_joint_names}, hand_joints={pregrasp_controller.hand_joint_names}.",
            flush=True,
        )
        pregrasp_result = pregrasp_controller.move_to_pose(
            position_w=grasp.pregrasp_position_w,
            orientation_xyzw=grasp.orientation_xyzw,
        )
        print(
            "[INFO]: Pregrasp admittance move finished. "
            f"status={pregrasp_result.status} success={pregrasp_result.success} message={pregrasp_result.message}",
            flush=True,
        )
        if not pregrasp_result.success:
            return
    else:
        solver = GoalIKSolver(context)
        print(
            "[INFO]: Pregrasp controller resolved "
            f"type=goal_ik ee_body={context.ee_body_name}, arm_joints={context.arm_joint_names}, "
            f"hand_joints={context.hand_joint_names}.",
            flush=True,
        )
        goal_q = solver.solve(
            PoseCommand(
                position_w=grasp.pregrasp_position_w,
                orientation_xyzw=grasp.orientation_xyzw,
            ),
            restore_start_state=False,
        )
        print(
            f"[INFO]: Pregrasp IK finished. success={goal_q is not None}",
            flush=True,
        )
        if goal_q is None:
            return
    target_tcp_position_w, target_tcp_orientation_xyzw = FR3MotionContext.grasp_pose_to_tcp_pose(
        grasp.pregrasp_position_w,
        grasp.orientation_xyzw,
    )
    print(
        "[INFO]: Pregrasp target "
        f"grasp_position={grasp.pregrasp_position_w} grasp_orientation_xyzw={grasp.orientation_xyzw} "
        f"tcp_position={target_tcp_position_w} tcp_orientation_xyzw={target_tcp_orientation_xyzw}",
        flush=True,
    )
    tcp_position_w, tcp_quat_w = context.get_tcp_pose_w()
    actual_tcp_position_w = tuple(float(v) for v in tcp_position_w[0].tolist())
    actual_tcp_orientation_xyzw = (
        float(tcp_quat_w[0, 1].item()),
        float(tcp_quat_w[0, 2].item()),
        float(tcp_quat_w[0, 3].item()),
        float(tcp_quat_w[0, 0].item()),
    )
    print(
        "[INFO]: Actual TCP pose after pregrasp move "
        f"position={actual_tcp_position_w} orientation_xyzw={actual_tcp_orientation_xyzw}",
        flush=True,
    )
    if args_cli.pregrasp_only:
        return

    execute_vertical_pick_sequence(
        sim=sim,
        scene=scene,
        robot=robot,
        cube=cube,
        start_tcp_position_w=actual_tcp_position_w,
        start_tcp_orientation_xyzw=actual_tcp_orientation_xyzw,
        target_tcp_z=grasp.position_w[2],
        open_gripper_width=grasp.gripper_width / 2.0,
        closed_gripper_width=0.0,
    )


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
    label: str,
) -> bool:
    for index, tcp_position_w in enumerate(tcp_positions_w, start=1):
        grasp_position_w, grasp_orientation_xyzw = FR3MotionContext.tcp_pose_to_grasp_pose(
            tcp_position_w,
            tcp_orientation_xyzw,
        )
        print(
            f"[INFO]: {label} waypoint "
            f"{index}/{len(tcp_positions_w)} tcp_position={tcp_position_w} tcp_orientation_xyzw={tcp_orientation_xyzw} "
            f"grasp_position={grasp_position_w} grasp_orientation_xyzw={grasp_orientation_xyzw}",
            flush=True,
        )
        result = controller.move_to_pose(position_w=grasp_position_w, orientation_xyzw=grasp_orientation_xyzw)
        print(
            f"[INFO]: {label} waypoint result "
            f"index={index} status={result.status} success={result.success} message={result.message}",
            flush=True,
        )
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
    cube,
    start_tcp_position_w: tuple[float, float, float],
    start_tcp_orientation_xyzw: tuple[float, float, float, float],
    target_tcp_z: float,
    open_gripper_width: float,
    closed_gripper_width: float,
) -> None:
    def build_waypoint_controller(fixed_gripper_width: float):
        if args_cli.controller == "admittance":
            return FR3AdmittanceController(
                robot=robot,
                scene=scene,
                sim=sim,
                fixed_gripper_width=fixed_gripper_width,
            )
        return FR3MoveToPoseController(
            robot=robot,
            cube=cube,
            scene=scene,
            sim=sim,
            fixed_gripper_width=fixed_gripper_width,
        )

    descent_waypoints = _build_vertical_tcp_waypoints(
        start_tcp_position_w=start_tcp_position_w,
        target_tcp_z=target_tcp_z,
        num_waypoints=12,
    )
    open_controller = build_waypoint_controller(open_gripper_width)
    if not _execute_tcp_waypoint_sequence(
        controller=open_controller,
        tcp_positions_w=descent_waypoints,
        tcp_orientation_xyzw=start_tcp_orientation_xyzw,
        label="Vertical",
    ):
        print("[WARN]: Vertical approach failed before gripper close.", flush=True)
        return

    _command_gripper_width(
        sim=sim,
        scene=scene,
        robot=robot,
        width=closed_gripper_width,
        duration_s=0.6,
    )

    retreat_controller = build_waypoint_controller(closed_gripper_width)
    retreat_waypoints = list(reversed(descent_waypoints[:-1])) + [start_tcp_position_w]
    if not _execute_tcp_waypoint_sequence(
        controller=retreat_controller,
        tcp_positions_w=retreat_waypoints,
        tcp_orientation_xyzw=start_tcp_orientation_xyzw,
        label="Retreat",
    ):
        print("[WARN]: Vertical retreat failed after gripper close.", flush=True)


def drive_robot_to_start_pose(sim, scene) -> None:
    """Actively settle the FR3 into a safe home pose before planning."""

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

    settled_joint_pos = {
        name: float(robot.data.joint_pos[0, joint_name_to_idx[name]].item()) for name in arm_joint_names
    }
    print(f"[INFO]: Settled FR3 start joints: {settled_joint_pos}", flush=True)


def run() -> None:
    """Run either the debug pick flow or explicit move-to-pose targets, then keep the app alive if requested."""

    configure_grasp_tcp_calibration()
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

    default_target_pos = (0.45, 0.0, 0.35)
    default_target_quat = (0.0, 1.0, 0.0, 0.0)
    using_default_target = (
        tuple(args_cli.target_pos) == default_target_pos and tuple(args_cli.target_quat) == default_target_quat
    )
    if not args_cli.test_multi_targets and using_default_target:
        run_pick_sequence(sim, scene, robot, cube)
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
        return

    if args_cli.controller == "admittance":
        controller = FR3AdmittanceController(robot=robot, scene=scene, sim=sim)
    else:
        controller = FR3MoveToPoseController(robot=robot, cube=cube, scene=scene, sim=sim)
    print(
        "[INFO]: Controller resolved "
        f"ee_body={controller.ee_body_name}, arm_joints={controller.arm_joint_names}, "
        f"hand_joints={controller.hand_joint_names}.",
        flush=True,
    )

    current_tcp_position_w, current_orientation_xyzw = controller.get_current_tcp_pose()
    print(
        f"[INFO]: Current TCP pose position={current_tcp_position_w} orientation_xyzw={current_orientation_xyzw}",
        flush=True,
    )
    targets = build_target_sequence(current_orientation_xyzw)
    for index, (position_w, orientation_xyzw) in enumerate(targets, start=1):
        if args_cli.test_multi_targets:
            target_tcp_position_w = SMOKE_TEST_TCP_TARGETS[index - 1]
            target_tcp_orientation_xyzw = current_orientation_xyzw
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
            "[INFO]: Planner resolved "
            f"ee_body={controller.ee_body_name}, arm_joints={controller.arm_joint_names}, "
            f"hand_joints={controller.hand_joint_names}.",
            flush=True,
        )

        current_tcp_position_w, current_orientation_xyzw = controller.get_current_tcp_pose()
        print(
            f"[INFO]: Current TCP pose position={current_tcp_position_w} orientation_xyzw={current_orientation_xyzw}",
            flush=True,
        )
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
    except Exception as exc:
        import traceback

        print(f"[ERROR]: Unhandled exception: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
