"""Run a saved Fabrica grasp bundle through an Isaac pickup attempt."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Load saved grasps, sample a part placement, and execute a pickup in Isaac."
)
parser.add_argument("--input-json", type=Path, required=True, help="Input grasp bundle, typically from stage 2.")
parser.add_argument(
    "--part-usd",
    type=str,
    default="",
    help=(
        "Optional prebuilt bundle-local USD asset. By default the runner converts the saved bundle-local mesh to USD "
        "so Isaac uses the same frame as stage 2."
    ),
)
parser.add_argument(
    "--use-provided-part-usd",
    action="store_true",
    help="Use --part-usd directly instead of generating a bundle-local USD from the input bundle.",
)
parser.add_argument("--fr3-usd", type=str, default="", help="Optional override for the Franka Panda USD path.")
parser.add_argument(
    "--controller",
    type=str,
    default="moveit",
    choices=("moveit",),
    help="Execution controller. Isaac pickup execution uses MoveIt-planned waypoints.",
)
parser.add_argument("--pregrasp-offset", type=float, default=0.20, help="Pregrasp offset in meters.")
parser.add_argument("--grasp-id", type=str, default="", help="Optional explicit grasp id to execute.")
parser.add_argument(
    "--gripper-width-clearance",
    type=float,
    default=0.01,
    help="Clearance added to the saved grasp jaw width for the open approach width.",
)
parser.add_argument("--close-width", type=float, default=0.0, help="Finger joint target width for close.")
parser.add_argument(
    "--object-mass-kg",
    type=float,
    default=None,
    help="Optional target object mass in kg. Mutually exclusive with --object-density-kg-m3.",
)
parser.add_argument(
    "--object-density-kg-m3",
    type=float,
    default=None,
    help="Optional target object density in kg/m^3. Defaults to PLA-like density when mass is not set.",
)
parser.add_argument(
    "--success-height-margin-m",
    type=float,
    default=0.05,
    help="Minimum object lift height required to report Isaac pickup success.",
)
parser.add_argument(
    "--tcp-to-grasp-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Override the fixed TCP-to-grasp-center offset used for pose conversion and pickup execution.",
)
parser.add_argument("--support-face", type=str, default="", help="Optional explicit support face.")
parser.add_argument("--yaw-deg", type=float, default=None, help="Optional explicit pickup yaw in degrees.")
parser.add_argument("--xy-world", type=str, default="", help="Optional explicit world XY as x,y.")
parser.add_argument("--random-support-face", action="store_true", help="Sample support face from allowed set.")
parser.add_argument("--random-yaw", action="store_true", help="Sample yaw from allowed set.")
parser.add_argument(
    "--allowed-support-faces",
    type=str,
    default="pos_x,neg_x,pos_y,neg_y,neg_z",
    help="Comma-separated support faces used by the random sampler.",
)
parser.add_argument(
    "--allowed-yaw-deg",
    type=str,
    default="0,90,180,270",
    help="Comma-separated yaw values used by the random sampler.",
)
parser.add_argument("--xy-min-world", type=str, default="-0.45,-0.05", help="Random placement XY lower bound.")
parser.add_argument("--xy-max-world", type=str, default="-0.35,0.05", help="Random placement XY upper bound.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument(
    "--detailed-finger-contact-gap-m",
    type=float,
    default=0.002,
    help="Detailed Franka finger contact gap used during the ground recheck.",
)
parser.add_argument(
    "--gripper-collision-model",
    type=str,
    default="",
    help="Gripper collision model for the pickup-pose ground recheck. Defaults to bundle metadata or franka_hand.",
)
parser.add_argument("--pregrasp-only", action="store_true", help="Stop after reaching pregrasp.")
parser.add_argument("--moveit-frame-id", type=str, default="base", help="MoveIt planning frame.")
parser.add_argument(
    "--moveit-target-position-signs",
    type=str,
    default="1,1,1",
    help="Comma-separated x,y,z signs applied to world grasp positions before MoveIt planning.",
)
parser.add_argument("--moveit-planning-group", type=str, default="fr3_arm", help="MoveIt planning group.")
parser.add_argument("--moveit-pose-link", type=str, default="fr3_hand_tcp", help="MoveIt pose link.")
parser.add_argument("--moveit-namespace", type=str, default="", help="Optional MoveIt namespace.")
parser.add_argument(
    "--moveit-joint-names",
    type=str,
    default="",
    help="Optional comma-separated arm joint names. Defaults to fr3_joint1..fr3_joint7.",
)
parser.add_argument(
    "--moveit-start-joint-positions",
    type=str,
    default="",
    help="Optional comma-separated MoveIt start joint positions for direct planning.",
)
parser.add_argument("--moveit-pipeline-id", type=str, default="", help="Optional MoveIt planning pipeline id.")
parser.add_argument("--moveit-planner-id", type=str, default="", help="Optional MoveIt planner id.")
parser.add_argument("--moveit-wait-for-moveit-timeout-s", type=float, default=15.0)
parser.add_argument("--moveit-ik-timeout-s", type=float, default=2.0)
parser.add_argument("--moveit-planning-time-s", type=float, default=5.0)
parser.add_argument("--moveit-num-planning-attempts", type=int, default=5)
parser.add_argument("--moveit-velocity-scale", type=float, default=0.05)
parser.add_argument("--moveit-acceleration-scale", type=float, default=0.05)
parser.add_argument(
    "--moveit-execution-speed-rad-s",
    type=float,
    default=0.35,
    help="Maximum Isaac joint playback speed for MoveIt waypoint execution.",
)
parser.add_argument(
    "--moveit-grasp-settle-time-s",
    type=float,
    default=0.0,
    help="Seconds to hold the final MoveIt grasp waypoint before closing the gripper.",
)
parser.add_argument(
    "--gripper-close-duration-s",
    type=float,
    default=1.5,
    help="Nominal gripper close command duration before contact/target settling.",
)
parser.add_argument(
    "--gripper-close-max-duration-s",
    type=float,
    default=10.0,
    help="Maximum gripper close duration before declaring close failure/contact.",
)
parser.add_argument(
    "--postclose-hold-s",
    type=float,
    default=1.0,
    help="Seconds to hold the closed gripper at the grasp waypoint before lift.",
)
parser.add_argument(
    "--moveit-plan-json",
    type=Path,
    default=None,
    help="Precomputed MoveIt joint waypoint plan. Used when IsaacLab Python cannot import ROS2.",
)
parser.add_argument(
    "--moveit-motion-sequence-json",
    type=Path,
    default=None,
    help="Optional arbitrary MoveIt segment sequence to execute continuously instead of a pickup.",
)
parser.add_argument(
    "--moveit-lift-height-m",
    type=float,
    default=0.08,
    help="Lift height for the MoveIt lift pose target, matching real_execution.lift_height_m by default.",
)
parser.add_argument("--moveit-allow-collisions", action="store_true")
parser.add_argument(
    "--run-seconds",
    type=float,
    default=0.0,
    help="Optional wall-clock duration to keep the simulation alive. Use 0 for until interrupted.",
)
parser.add_argument(
    "--attempt-artifact",
    type=Path,
    default=Path("artifacts/isaac_pick_attempt.json"),
    help="Optional JSON artifact for the selected attempt.",
)
parser.add_argument("--record-video", type=Path, default=None, help="Optional MP4/AVI path for Isaac RGB video.")
parser.add_argument("--video-fps", type=float, default=30.0, help="Recorded video frame rate.")
parser.add_argument("--video-width", type=int, default=960, help="Recorded video width in pixels.")
parser.add_argument("--video-height", type=int, default=540, help="Recorded video height in pixels.")
parser.add_argument(
    "--video-camera-eye",
    type=float,
    nargs=3,
    default=(1.6, -1.2, 1.0),
    metavar=("X", "Y", "Z"),
    help="Isaac recording camera eye position.",
)
parser.add_argument(
    "--video-camera-target",
    type=float,
    nargs=3,
    default=(0.35, 0.0, 0.3),
    metavar=("X", "Y", "Z"),
    help="Isaac recording camera look-at target.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.object_mass_kg is not None and args_cli.object_density_kg_m3 is not None:
    parser.error("--object-mass-kg and --object-density-kg-m3 are mutually exclusive.")
if args_cli.object_mass_kg is not None and args_cli.object_mass_kg <= 0.0:
    parser.error("--object-mass-kg must be > 0.")
if args_cli.object_density_kg_m3 is not None and args_cli.object_density_kg_m3 <= 0.0:
    parser.error("--object-density-kg-m3 must be > 0.")
if args_cli.record_video is not None:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
from isaaclab.sim.schemas import schemas_cfg  # noqa: E402
from isaaclab.sim.utils import bind_physics_material  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

from grasp_planning import (  # noqa: E402
    accepted_grasps,
    build_pickup_pose_world,
    evaluate_saved_grasps_against_pickup_pose,
    load_grasp_bundle,
    sample_pickup_placement_spec,
    saved_grasp_to_world_grasp,
    score_grasps,
    select_first_feasible_grasp,
)
from grasp_planning.controllers.fr3_pick_controller import FR3PickController  # noqa: E402
from grasp_planning.envs import (  # noqa: E402
    DEFAULT_PART_DENSITY_KG_M3,
    ISAAC_MIN_CONTACT_OFFSET_M,
    make_fr3_part_scene_cfg,
)
from grasp_planning.envs.franka_collisions import expose_franka_mesh_collisions  # noqa: E402
from grasp_planning.grasping.fabrica_grasp_debug import load_stl_mesh  # noqa: E402
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.mujoco.scene_builder import write_temporary_triangle_mesh_stl  # noqa: E402
from grasp_planning.planning.fr3_motion_context import FR3MotionContext  # noqa: E402
from grasp_planning.planning.pick_execution import (  # noqa: E402
    drive_robot_to_start_pose,
    execute_moveit_joint_trajectory_sequence,
    execute_pick_from_moveit_joint_trajectories,
)
from grasp_planning.ros2.moveit_pose_commander import (  # noqa: E402
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    rclpy,
)
from grasp_planning.ros2.moveit_world_grasp import world_grasp_pose_targets  # noqa: E402
from grasp_planning.scene_defaults import ROBOT_BASE_ORIENTATION_XYZW, ROBOT_BASE_POSITION  # noqa: E402
from grasp_planning.start_poses import (  # noqa: E402
    DEFAULT_HAND_OPEN_WIDTH,
    KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
    kuka_isaac_to_moveit_joint_positions,
    kuka_moveit_to_isaac_joint_positions,
)
from grasp_planning.video import OpenCvVideoWriter  # noqa: E402


class GraspSelectionFailure(RuntimeError):
    def __init__(self, *, status: str, message: str, world_grasp=None) -> None:
        super().__init__(message)
        self.status = status
        self.world_grasp = world_grasp


def _effective_object_density_kg_m3() -> float | None:
    if args_cli.object_mass_kg is not None:
        return None
    if args_cli.object_density_kg_m3 is not None:
        return float(args_cli.object_density_kg_m3)
    return float(DEFAULT_PART_DENSITY_KG_M3)


def _object_mass_properties_cfg():
    if args_cli.object_mass_kg is not None:
        return sim_utils.MassPropertiesCfg(mass=float(args_cli.object_mass_kg))
    density_kg_m3 = _effective_object_density_kg_m3()
    if density_kg_m3 is None:
        return None
    return sim_utils.MassPropertiesCfg(density=float(density_kg_m3))


def _bind_high_friction_contact_material(stage, *, root_paths: tuple[str, ...]) -> int:
    material_path = "/World/Looks/high_friction_contact_material"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=3.0,
        dynamic_friction=2.5,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    material_cfg.func(material_path, material_cfg)
    bound_count = 0
    for root_path in root_paths:
        prim = stage.GetPrimAtPath(root_path)
        if not prim.IsValid():
            print(f"[WARN]: Cannot bind contact material; prim does not exist: {root_path}", flush=True)
            continue
        try:
            bind_physics_material(root_path, material_path, stage=stage, stronger_than_descendants=True)
            bound_count += 1
        except ValueError as exc:
            print(f"[WARN]: Cannot bind contact material to {root_path}: {exc}", flush=True)
    return bound_count


def _parse_vec2(raw: str) -> tuple[float, float]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 2:
        raise ValueError(f"Expected exactly 2 comma-separated values, got '{raw}'.")
    return values


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one comma-separated float, got '{raw}'.")
    return values


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one comma-separated token, got '{raw}'.")
    return values


def _moveit_joint_names_from_args() -> tuple[str, ...]:
    if not str(args_cli.moveit_joint_names).strip():
        return MoveItPoseCommanderConfig().joint_names
    return _parse_str_tuple(str(args_cli.moveit_joint_names))


def _using_kuka_lbr_moveit_joints() -> bool:
    joint_names = _moveit_joint_names_from_args()
    return len(joint_names) == 7 and all(name == f"lbr_A{index}" for index, name in enumerate(joint_names, start=1))


def _moveit_target_position_signs_from_args() -> tuple[float, float, float]:
    signs = _parse_float_tuple(str(args_cli.moveit_target_position_signs))
    if len(signs) != 3:
        raise ValueError(
            f"--moveit-target-position-signs must contain exactly 3 comma-separated values for x,y,z, got {len(signs)}."
        )
    return (float(signs[0]), float(signs[1]), float(signs[2]))


def _moveit_waypoint_to_isaac(waypoint: tuple[float, ...]) -> tuple[float, ...]:
    if _using_kuka_lbr_moveit_joints():
        return kuka_moveit_to_isaac_joint_positions(waypoint)
    return tuple(float(value) for value in waypoint)


def _isaac_waypoint_to_moveit(waypoint: tuple[float, ...]) -> tuple[float, ...]:
    if _using_kuka_lbr_moveit_joints():
        return kuka_isaac_to_moveit_joint_positions(waypoint)
    return tuple(float(value) for value in waypoint)


def _moveit_start_joint_positions_from_args() -> tuple[float, ...] | None:
    if not str(args_cli.moveit_start_joint_positions).strip():
        return None
    positions = _parse_float_tuple(str(args_cli.moveit_start_joint_positions))
    joint_names = _moveit_joint_names_from_args()
    if len(positions) != len(joint_names):
        raise ValueError(
            f"--moveit-start-joint-positions must match the configured MoveIt joint-name count ({len(joint_names)})."
        )
    return positions


def _approach_open_gripper_width(selected_world_grasp=None) -> float:
    del selected_world_grasp
    if _using_kuka_lbr_moveit_joints():
        return float(KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M)
    return float(DEFAULT_HAND_OPEN_WIDTH)


def _write_kuka_configured_start_state(scene) -> None:
    """Apply Isaac Lab's configured articulation defaults before the first rendered step."""

    if not _using_kuka_lbr_moveit_joints():
        return
    robot = scene["robot"]
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    scene.write_data_to_sim()
    configured = {
        name: float(joint_pos[0, index]) for index, name in enumerate(robot.joint_names) if name.startswith("joint")
    }
    print(f"[INFO]: Applied configured KUKA articulation start state: {configured}", flush=True)


def _prepare_robot_start_pose(sim, scene, *, hand_open_width: float, step_callback=None) -> None:
    if _using_kuka_lbr_moveit_joints():
        print(
            "[INFO]: KUKA spawned at its configured articulation start state; "
            "skipping the legacy active start-pose drive.",
            flush=True,
        )
        return
    print("[INFO]: Driving robot to start pose...", flush=True)
    drive_robot_to_start_pose(
        sim,
        scene,
        hand_open_width=hand_open_width,
        step_callback=step_callback,
    )


def _effective_close_gripper_width(selected_grasp) -> float:
    requested_close_width = float(args_cli.close_width)
    if requested_close_width > 0.0 or not _using_kuka_lbr_moveit_joints():
        return requested_close_width
    jaw_width = float(getattr(selected_grasp, "jaw_width", 0.0))
    if not np.isfinite(jaw_width) or jaw_width <= 0.0:
        return requested_close_width
    return min(0.001, jaw_width)


def _is_generated_kuka_y_gripper_usd(path: str | Path) -> bool:
    resolved = str(path).replace("\\", "/")
    return "kuka_iiwa7_y_gripper" in resolved


def resolve_fr3_usd_path() -> str:
    if args_cli.fr3_usd:
        return args_cli.fr3_usd
    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Unable to resolve Isaac asset root for the built-in Franka Factory asset.")
    return assets_root_path + "/Isaac/IsaacLab/Factory/franka_mimic.usd"


def configure_grasp_tcp_calibration() -> None:
    tcp_to_grasp_offset = tuple(float(value) for value in args_cli.tcp_to_grasp_offset)
    FR3MotionContext._TCP_TO_GRASP_CENTER_OFFSET = tcp_to_grasp_offset
    FR3PickController._TCP_TO_GRASP_CENTER_OFFSET = tcp_to_grasp_offset


def _gripper_collision_model_from_args_or_bundle(bundle) -> str:
    if str(args_cli.gripper_collision_model).strip():
        return str(args_cli.gripper_collision_model)
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    return str(metadata.get("gripper_collision_model", "franka_hand"))


def _moveit_config_from_args() -> MoveItPoseCommanderConfig:
    return MoveItPoseCommanderConfig(
        planning_group=str(args_cli.moveit_planning_group),
        pose_link=str(args_cli.moveit_pose_link),
        joint_names=_moveit_joint_names_from_args(),
        moveit_namespace=str(args_cli.moveit_namespace),
        pipeline_id=str(args_cli.moveit_pipeline_id),
        planner_id=str(args_cli.moveit_planner_id),
        wait_for_moveit_timeout_s=float(args_cli.moveit_wait_for_moveit_timeout_s),
        ik_timeout_s=float(args_cli.moveit_ik_timeout_s),
        fk_timeout_s=float(args_cli.moveit_ik_timeout_s),
        planning_time_s=float(args_cli.moveit_planning_time_s),
        num_planning_attempts=int(args_cli.moveit_num_planning_attempts),
        velocity_scale=float(args_cli.moveit_velocity_scale),
        acceleration_scale=float(args_cli.moveit_acceleration_scale),
        post_execute_sleep_s=0.0,
        avoid_collisions=not bool(args_cli.moveit_allow_collisions),
    )


def _trajectory_waypoints_for_joints(trajectory, *, joint_names: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    joint_trajectory = trajectory.joint_trajectory
    source_joint_names = tuple(str(name) for name in joint_trajectory.joint_names)
    name_to_index = {name: index for index, name in enumerate(source_joint_names)}
    missing = [joint_name for joint_name in joint_names if joint_name not in name_to_index]
    if missing:
        raise RuntimeError(f"MoveIt trajectory is missing arm joints: {missing}.")
    ordered_indices = [name_to_index[name] for name in joint_names]
    waypoints = tuple(
        tuple(float(point.positions[index]) for index in ordered_indices) for point in tuple(joint_trajectory.points)
    )
    if not waypoints:
        raise RuntimeError("MoveIt returned a trajectory with no points.")
    return waypoints


def _plan_moveit_target_sequence(
    *,
    targets,
    labels: tuple[str, ...],
    start_joint_positions: tuple[float, ...],
) -> dict[str, tuple[tuple[float, ...], ...]]:
    if rclpy is None:
        raise RuntimeError("ROS2 MoveIt dependencies are unavailable. Source the ROS2 / MoveIt workspace first.")
    initialized_here = False
    commander = None
    try:
        if not rclpy.ok():
            print("[INFO]: Initializing ROS2 client for MoveIt planning.", flush=True)
            rclpy.init()
            initialized_here = True
        moveit_config = _moveit_config_from_args()
        print(
            f"[INFO]: Connecting to MoveIt group={moveit_config.planning_group} link={moveit_config.pose_link}.",
            flush=True,
        )
        commander = MoveItPoseCommander(moveit_config, node_name="isaac_moveit_trajectory")
        commander.wait_for_moveit(require_execute=False)
        planned: dict[str, tuple[tuple[float, ...], ...]] = {}
        current_start = start_joint_positions
        for label in labels:
            print(f"[INFO]: Requesting MoveIt plan for {label}.", flush=True)
            trajectory, message = commander.plan_to_pose(
                targets[label],
                label=f"isaac_{label}",
                start_joint_positions=current_start,
            )
            if trajectory is None:
                raise RuntimeError(f"MoveIt failed to plan {label}: {message}")
            waypoints = _trajectory_waypoints_for_joints(trajectory, joint_names=moveit_config.joint_names)
            print(f"[INFO]: MoveIt plan for {label} returned {len(waypoints)} waypoints.", flush=True)
            planned[label] = tuple(_moveit_waypoint_to_isaac(waypoint) for waypoint in waypoints)
            current_start = waypoints[-1]
        return planned
    finally:
        if commander is not None:
            commander.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()


def _plan_moveit_joint_trajectories(
    *,
    world_grasp,
    start_joint_positions: tuple[float, ...],
) -> dict[str, tuple[tuple[float, ...], ...]]:
    print("[INFO]: Building MoveIt pose targets for Isaac attempt.", flush=True)
    targets = world_grasp_pose_targets(
        world_grasp,
        frame_id=str(args_cli.moveit_frame_id),
        lift_height_m=float(args_cli.moveit_lift_height_m),
        position_signs=_moveit_target_position_signs_from_args(),
        tcp_to_grasp_offset=tuple(float(value) for value in args_cli.tcp_to_grasp_offset),
    )
    print("[INFO]: Built MoveIt pose targets for Isaac attempt.", flush=True)
    labels = ("pregrasp",) if args_cli.pregrasp_only else ("pregrasp", "grasp", "lift")
    return _plan_moveit_target_sequence(
        targets=targets,
        labels=labels,
        start_joint_positions=start_joint_positions,
    )


def _current_isaac_arm_joint_positions(*, sim, scene, robot, fixed_gripper_width: float) -> tuple[float, ...]:
    context = FR3MotionContext(
        robot=robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=fixed_gripper_width,
    )
    return tuple(float(value) for value in context.get_arm_q()[0].tolist())


def _print_moveit_joint_trajectory_summary(trajectories: dict[str, tuple[tuple[float, ...], ...]]) -> None:
    for label, waypoints in trajectories.items():
        print(
            f"[INFO]: MoveIt Isaac trajectory {label}: waypoints={len(waypoints)}",
            flush=True,
        )


def _load_moveit_plan_json(path: Path) -> tuple[dict[str, object], dict[str, tuple[tuple[float, ...], ...]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected MoveIt plan JSON object in '{path}'.")
    raw_trajectories = payload.get("trajectories")
    if not isinstance(raw_trajectories, dict):
        raise ValueError(f"MoveIt plan JSON '{path}' is missing a trajectories object.")
    trajectories: dict[str, tuple[tuple[float, ...], ...]] = {}
    for label, raw_waypoints in raw_trajectories.items():
        if not isinstance(raw_waypoints, list):
            raise ValueError(f"MoveIt plan trajectory '{label}' must be a list of waypoints.")
        waypoints = []
        for raw_waypoint in raw_waypoints:
            if not isinstance(raw_waypoint, list | tuple):
                raise ValueError(f"MoveIt plan trajectory '{label}' contains a non-list waypoint.")
            waypoints.append(_moveit_waypoint_to_isaac(tuple(float(value) for value in raw_waypoint)))
        trajectories[str(label)] = tuple(waypoints)
    return payload, trajectories


def _mesh_in_bundle_local_frame(bundle) -> object:
    mesh_global = load_stl_mesh(bundle.target_stl_path, scale=bundle.stl_scale)
    rotation = bundle.local_frame_orientation_xyzw_world
    object_pose_world = type("BundlePose", (), {})()
    object_pose_world.rotation_world_from_object = None
    # Use the same row-vector convention as ObjectWorldPose.transform_points_to_world.
    from grasp_planning.grasping.fabrica_grasp_debug import TriangleMesh, quat_to_rotmat_xyzw

    rot = quat_to_rotmat_xyzw(rotation)
    translation = np.asarray(bundle.local_frame_origin_world, dtype=float)
    vertices_local = (np.asarray(mesh_global.vertices_obj, dtype=float) - translation[None, :]) @ rot
    return TriangleMesh(vertices_obj=vertices_local, faces=np.asarray(mesh_global.faces, dtype=np.int64))


def _bundle_execution_pose_world(bundle) -> ObjectWorldPose | None:
    metadata = dict(bundle.metadata)
    raw_pose = metadata.get("execution_world_pose")
    if not isinstance(raw_pose, dict):
        return None
    position_world = raw_pose.get("position_world")
    orientation_xyzw_world = raw_pose.get("orientation_xyzw_world")
    if not isinstance(position_world, (list, tuple)) or not isinstance(orientation_xyzw_world, (list, tuple)):
        return None
    if len(position_world) != 3 or len(orientation_xyzw_world) != 4:
        return None
    return ObjectWorldPose(
        position_world=tuple(float(v) for v in position_world),
        orientation_xyzw_world=tuple(float(v) for v in orientation_xyzw_world),
    )


def _explicit_pose_spec(object_pose_world: ObjectWorldPose):
    return type(
        "PlacementSpec",
        (),
        {
            "support_face": "explicit_pose",
            "yaw_deg": 0.0,
            "xy_world": tuple(float(v) for v in object_pose_world.position_world[:2]),
        },
    )()


def _resolve_placement_spec():
    explicit_xy = _parse_vec2(args_cli.xy_world) if args_cli.xy_world else None
    if args_cli.support_face and args_cli.yaw_deg is not None and explicit_xy is not None:
        from grasp_planning.grasping.fabrica_grasp_debug import PickupPlacementSpec

        return PickupPlacementSpec(
            support_face=args_cli.support_face,
            yaw_deg=float(args_cli.yaw_deg),
            xy_world=explicit_xy,
        )

    rng = np.random.default_rng(args_cli.seed)
    return sample_pickup_placement_spec(
        rng=rng,
        allowed_support_faces=_parse_str_tuple(args_cli.allowed_support_faces),
        allowed_yaw_deg=_parse_float_tuple(args_cli.allowed_yaw_deg),
        xy_min_world=_parse_vec2(args_cli.xy_min_world),
        xy_max_world=_parse_vec2(args_cli.xy_max_world),
    )


def _resolve_placement_and_pose(bundle, mesh_local):
    bundle_pose_world = _bundle_execution_pose_world(bundle)
    has_pickup_override = bool(args_cli.support_face or args_cli.yaw_deg is not None or args_cli.xy_world)
    if bundle_pose_world is not None and not has_pickup_override:
        return _explicit_pose_spec(bundle_pose_world), bundle_pose_world

    placement_spec = _resolve_placement_spec()
    object_pose_world = build_pickup_pose_world(
        mesh_local,
        support_face=placement_spec.support_face,
        yaw_deg=placement_spec.yaw_deg,
        xy_world=placement_spec.xy_world,
    )
    return placement_spec, object_pose_world


def _mesh_collision_cfg():
    return schemas_cfg.ConvexDecompositionPropertiesCfg()


class IsaacVideoRecorder:
    def __init__(self, *, camera: Camera, sim, output_path: Path, fps: float, width: int, height: int) -> None:
        self._camera = camera
        self._sim = sim
        self._writer = OpenCvVideoWriter(output_path, fps=fps, width=width, height=height)
        self._capture_interval_s = 1.0 / float(fps)
        self._next_capture_time_s = 0.0
        self._elapsed_s = 0.0
        self.output_path = str(output_path)

    @property
    def frame_count(self) -> int:
        return int(self._writer.frame_count)

    def set_view(self, *, eye: tuple[float, float, float], target: tuple[float, float, float]) -> None:
        eye_tensor = torch.tensor([eye], dtype=torch.float32, device=self._sim.device)
        target_tensor = torch.tensor([target], dtype=torch.float32, device=self._sim.device)
        self._camera.set_world_poses_from_view(eye_tensor, target_tensor)

    def capture(self, *, force: bool = False) -> None:
        physics_dt = float(self._sim.get_physics_dt())
        self._elapsed_s += physics_dt
        if not force and self._elapsed_s + 1.0e-9 < self._next_capture_time_s:
            return
        self._camera.update(dt=physics_dt)
        raw_rgb = self._camera.data.output.get("rgb")
        if raw_rgb is None:
            return
        frame = raw_rgb[0]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        self._writer.append_rgb(np.asarray(frame))
        if force:
            self._next_capture_time_s = max(self._next_capture_time_s, self._elapsed_s + self._capture_interval_s)
            return
        while self._next_capture_time_s <= self._elapsed_s + 1.0e-9:
            self._next_capture_time_s += self._capture_interval_s

    def close(self) -> None:
        self._writer.close()


def _make_video_camera() -> Camera | None:
    if args_cli.record_video is None:
        return None
    sim_utils.create_prim("/World/ExecutionBenchmarkVideo", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/ExecutionBenchmarkVideo/CameraSensor",
        update_period=0.0,
        height=int(args_cli.video_height),
        width=int(args_cli.video_width),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e5),
        ),
    )
    return Camera(cfg=camera_cfg)


def resolve_part_usd_path(*, bundle, mesh_local) -> str:
    if args_cli.use_provided_part_usd:
        if not args_cli.part_usd:
            raise ValueError("--part-usd is required when --use-provided-part-usd is set.")
        print(
            "[WARN]: Using provided part USD directly. It must already be authored in the saved bundle-local frame.",
            flush=True,
        )
        return args_cli.part_usd

    output_dir = Path("artifacts/isaac_bundle_assets").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_usd = output_dir / f"{args_cli.input_json.stem}_bundle_local.usd"
    temp_stl = write_temporary_triangle_mesh_stl(
        mesh_local,
        prefix=f"{args_cli.input_json.stem}_bundle_local_",
        dir=output_dir,
    )
    converter_cfg = MeshConverterCfg(
        asset_path=str(temp_stl),
        usd_dir=str(output_dir),
        usd_file_name=output_usd.name,
        force_usd_conversion=True,
        make_instanceable=False,
        scale=(1.0, 1.0, 1.0),
        mass_props=_object_mass_properties_cfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=ISAAC_MIN_CONTACT_OFFSET_M,
            rest_offset=0.0,
        ),
        mesh_collision_props=_mesh_collision_cfg(),
    )
    converter = MeshConverter(converter_cfg)
    converted_path = str(Path(converter.usd_path).resolve())
    print(
        f"[INFO]: Generated bundle-local Isaac part USD from bundle={bundle.target_stl_path} output={converted_path}",
        flush=True,
    )
    try:
        temp_stl.unlink()
    except FileNotFoundError:
        pass
    return converted_path


def build_scene(
    *, object_pose_world, part_usd_path: str
) -> tuple[sim_utils.SimulationContext, InteractiveScene, Camera | None]:
    print("[INFO]: Creating simulation context...", flush=True)
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim._app_control_on_stop_handle = None
    sim._disable_app_control_on_stop_handle = True
    sim.set_camera_view([1.6, -1.2, 1.0], [0.35, 0.0, 0.3])

    print("[INFO]: Building Franka Panda + part scene config...", flush=True)
    fr3_usd_path = resolve_fr3_usd_path()
    print(f"[INFO]: Isaac robot USD: {fr3_usd_path}", flush=True)
    scene_cfg = make_fr3_part_scene_cfg(
        fr3_asset_path=fr3_usd_path,
        part_usd_path=part_usd_path,
        part_position=object_pose_world.position_world,
        part_orientation_xyzw=object_pose_world.orientation_xyzw_world,
        part_mass_kg=None if args_cli.object_mass_kg is None else float(args_cli.object_mass_kg),
        part_density_kg_m3=_effective_object_density_kg_m3(),
        robot_base_position=ROBOT_BASE_POSITION,
        robot_base_orientation_xyzw=ROBOT_BASE_ORIENTATION_XYZW,
    )
    print("[INFO]: Creating interactive scene...", flush=True)
    scene = InteractiveScene(scene_cfg)
    video_camera = _make_video_camera()
    print("[INFO]: Waiting for stage assets to finish loading...", flush=True)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    if _is_generated_kuka_y_gripper_usd(fr3_usd_path):
        print(
            "[INFO]: Skipping Franka visual mesh collision exposure for generated KUKA/Y-gripper USD; "
            "using authored collision hulls.",
            flush=True,
        )
    else:
        enabled_collision_count, _enabled_collision_paths = expose_franka_mesh_collisions(
            mesh_path_patterns=(
                r"(?:panda|fr3)_hand",
                r"(?:panda|fr3)_leftfinger",
                r"(?:panda|fr3)_rightfinger",
                r"finger",
                r"gripper",
            )
        )
        if enabled_collision_count > 0:
            print(
                f"[INFO]: Enabled Isaac collision on {enabled_collision_count} Franka gripper mesh prims.",
                flush=True,
            )
        else:
            print(
                "[WARN]: No Franka gripper mesh prims were found while enabling Isaac robot collisions; "
                "gripper-object contacts may be missing.",
                flush=True,
            )
    stage = omni.usd.get_context().get_stage()
    bound_contact_material_count = _bind_high_friction_contact_material(
        stage,
        root_paths=("/World/envs/env_0/Robot", "/World/envs/env_0/Part"),
    )
    print(
        "[INFO]: Bound high-friction Isaac contact material "
        f"to {bound_contact_material_count} robot/part collision subtrees.",
        flush=True,
    )
    print("[INFO]: Resetting simulator...", flush=True)
    sim.reset()
    print("[INFO]: Resetting scene buffers...", flush=True)
    scene.reset()
    _write_kuka_configured_start_state(scene)
    print("[INFO]: Scene ready.", flush=True)
    return sim, scene, video_camera


def _write_attempt_artifact(
    *,
    bundle,
    placement_spec,
    object_pose_world,
    statuses,
    selected_world_grasp,
    execution_result,
    video_recorder=None,
) -> None:
    artifact = {
        "input_json": str(args_cli.input_json),
        "target_stl_path": bundle.target_stl_path,
        "part_usd": getattr(args_cli, "resolved_part_usd", args_cli.part_usd),
        "placement": {
            "support_face": placement_spec.support_face,
            "yaw_deg": placement_spec.yaw_deg,
            "xy_world": list(placement_spec.xy_world),
            "object_position_world": list(object_pose_world.position_world),
            "object_orientation_xyzw_world": list(object_pose_world.orientation_xyzw_world),
        },
        "counts": {
            "saved": len(bundle.candidates),
            "ground_feasible": len(accepted_grasps(statuses)),
        },
        "selected_grasp_id": None if selected_world_grasp is None else selected_world_grasp.grasp_id,
        "selected_world_grasp": None
        if selected_world_grasp is None
        else {
            "position_w": list(selected_world_grasp.position_w),
            "orientation_xyzw": list(selected_world_grasp.orientation_xyzw),
            "normal_w": list(selected_world_grasp.normal_w),
            "pregrasp_position_w": list(selected_world_grasp.pregrasp_position_w),
            "jaw_width": selected_world_grasp.jaw_width,
            "gripper_width": selected_world_grasp.gripper_width,
        },
        "execution": {
            "controller": args_cli.controller,
            "success": execution_result.success,
            "status": execution_result.status,
            "message": execution_result.message,
        },
    }
    object_lift_height_m = getattr(execution_result, "object_lift_height_m", None)
    target_lift_height_m = getattr(execution_result, "target_lift_height_m", None)
    if object_lift_height_m is not None:
        artifact["execution"]["object_lift_height_m"] = object_lift_height_m
    if target_lift_height_m is not None:
        artifact["execution"]["target_lift_height_m"] = target_lift_height_m
    diagnostics = getattr(execution_result, "diagnostics", None)
    if diagnostics:
        artifact["execution"]["diagnostics"] = dict(diagnostics)
    if video_recorder is not None:
        artifact["video"] = {
            "path": video_recorder.output_path,
            "frame_count": video_recorder.frame_count,
            "fps": float(args_cli.video_fps),
            "width": int(args_cli.video_width),
            "height": int(args_cli.video_height),
        }
    artifact["moveit"] = {
        "frame_id": args_cli.moveit_frame_id,
        "target_position_signs": list(_moveit_target_position_signs_from_args()),
        "tcp_to_grasp_offset": [float(value) for value in args_cli.tcp_to_grasp_offset],
        "planning_group": args_cli.moveit_planning_group,
        "pose_link": args_cli.moveit_pose_link,
        "namespace": args_cli.moveit_namespace,
        "joint_names": list(_moveit_joint_names_from_args()),
        "pipeline_id": args_cli.moveit_pipeline_id,
        "planner_id": args_cli.moveit_planner_id,
        "lift_height_m": args_cli.moveit_lift_height_m,
        "allow_collisions": bool(args_cli.moveit_allow_collisions),
        "plan_json": None if args_cli.moveit_plan_json is None else str(args_cli.moveit_plan_json),
    }
    output = args_cli.attempt_artifact
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def _candidate_world_grasp(grasp, object_pose_world):
    return saved_grasp_to_world_grasp(
        grasp,
        object_pose_world,
        pregrasp_offset=args_cli.pregrasp_offset,
        gripper_width_clearance=args_cli.gripper_width_clearance,
    )


def _select_executable_grasp(feasible_grasps, statuses, object_pose_world):
    candidates = _ordered_executable_grasp_candidates(feasible_grasps, statuses, object_pose_world)
    if not candidates:
        return None, None
    return candidates[0]


def _ordered_executable_grasp_candidates(feasible_grasps, statuses, object_pose_world):
    min_pregrasp_z = 0.05
    if args_cli.grasp_id:
        selected = next((grasp for grasp in feasible_grasps if grasp.grasp_id == args_cli.grasp_id), None)
        if selected is None:
            raise GraspSelectionFailure(
                status="requested_grasp_not_feasible",
                message=f"Requested grasp id '{args_cli.grasp_id}' is not ground-feasible for this pickup pose.",
            )
        world_grasp = _candidate_world_grasp(selected, object_pose_world)
        if world_grasp.pregrasp_position_w[2] <= min_pregrasp_z:
            raise GraspSelectionFailure(
                status="invalid_pregrasp",
                message=(
                    f"Requested grasp id '{args_cli.grasp_id}' has unsafe pregrasp height: "
                    f"pregrasp_position_w={world_grasp.pregrasp_position_w} required_min_z={min_pregrasp_z:.3f}"
                ),
                world_grasp=world_grasp,
            )
        return [(selected, world_grasp)]

    ordered = sorted(
        feasible_grasps, key=lambda grasp: float("-inf") if grasp.score is None else grasp.score, reverse=True
    )
    candidates = []
    for grasp in ordered:
        world_grasp = _candidate_world_grasp(grasp, object_pose_world)
        if world_grasp.pregrasp_position_w[2] > min_pregrasp_z:
            candidates.append((grasp, world_grasp))
    if candidates:
        return candidates

    fallback = select_first_feasible_grasp(statuses)
    if fallback is None:
        return []
    world_fallback = _candidate_world_grasp(fallback, object_pose_world)
    if world_fallback.pregrasp_position_w[2] <= min_pregrasp_z:
        return []
    return [(fallback, world_fallback)]


def run() -> None:
    configure_grasp_tcp_calibration()
    print("[INFO]: Loading grasp bundle...", flush=True)
    bundle = load_grasp_bundle(args_cli.input_json)
    print(f"[INFO]: Loaded {len(bundle.candidates)} saved grasps from '{args_cli.input_json}'.", flush=True)
    print("[INFO]: Loading local STL mesh for placement and filtering...", flush=True)
    mesh_local = _mesh_in_bundle_local_frame(bundle)
    args_cli.resolved_part_usd = resolve_part_usd_path(bundle=bundle, mesh_local=mesh_local)
    print("[INFO]: Resolving pickup placement...", flush=True)
    placement_spec, object_pose_world = _resolve_placement_and_pose(bundle, mesh_local)
    print(
        "[INFO]: Pickup placement "
        f"support_face={placement_spec.support_face} yaw_deg={placement_spec.yaw_deg:.1f} "
        f"xy_world={placement_spec.xy_world} object_pose_world={object_pose_world.position_world} "
        f"orientation_xyzw={object_pose_world.orientation_xyzw_world}",
        flush=True,
    )

    print("[INFO]: Rechecking saved grasps against the selected pickup pose...", flush=True)
    gripper_collision_model = _gripper_collision_model_from_args_or_bundle(bundle)
    statuses = evaluate_saved_grasps_against_pickup_pose(
        bundle.candidates,
        object_pose_world=object_pose_world,
        contact_gap_m=args_cli.detailed_finger_contact_gap_m,
        gripper_collision_model=gripper_collision_model,
    )
    rescored_feasible = score_grasps(accepted_grasps(statuses), mesh_local=mesh_local)
    rescored_by_id = {grasp.grasp_id: grasp for grasp in rescored_feasible}
    statuses = [
        type(entry)(
            grasp=rescored_by_id.get(entry.grasp.grasp_id, entry.grasp),
            status=entry.status,
            reason=entry.reason,
        )
        for entry in statuses
    ]
    feasible_grasps = accepted_grasps(statuses)
    try:
        executable_candidates = _ordered_executable_grasp_candidates(feasible_grasps, statuses, object_pose_world)
    except GraspSelectionFailure as exc:
        result = SimpleNamespace(success=False, status=exc.status, message=str(exc))
        _write_attempt_artifact(
            bundle=bundle,
            placement_spec=placement_spec,
            object_pose_world=object_pose_world,
            statuses=statuses,
            selected_world_grasp=exc.world_grasp,
            execution_result=result,
        )
        raise RuntimeError(result.message) from exc
    print(
        f"[INFO]: Ground recheck complete. feasible={len(feasible_grasps)} / saved={len(bundle.candidates)}",
        flush=True,
    )
    if not executable_candidates:
        result = SimpleNamespace(
            success=False,
            status="no_feasible_grasp",
            message="No saved grasp survives the pickup-ground recheck for the sampled placement.",
        )
        _write_attempt_artifact(
            bundle=bundle,
            placement_spec=placement_spec,
            object_pose_world=object_pose_world,
            statuses=statuses,
            selected_world_grasp=None,
            execution_result=result,
        )
        raise RuntimeError(result.message)
    selected_grasp, selected_world_grasp = executable_candidates[0]
    print(
        "[INFO]: Initial grasp candidate "
        f"id={selected_grasp.grasp_id} grasp_w={selected_world_grasp.position_w} "
        f"pregrasp_w={selected_world_grasp.pregrasp_position_w} "
        f"orientation_xyzw={selected_world_grasp.orientation_xyzw} "
        f"gripper_width={selected_world_grasp.gripper_width:.4f}",
        flush=True,
    )

    sim, scene, video_camera = build_scene(
        object_pose_world=object_pose_world, part_usd_path=args_cli.resolved_part_usd
    )
    physics_dt = sim.get_physics_dt()
    video_recorder = None
    if video_camera is not None:
        video_recorder = IsaacVideoRecorder(
            camera=video_camera,
            sim=sim,
            output_path=args_cli.record_video,
            fps=float(args_cli.video_fps),
            width=int(args_cli.video_width),
            height=int(args_cli.video_height),
        )
        video_recorder.set_view(
            eye=tuple(float(value) for value in args_cli.video_camera_eye),
            target=tuple(float(value) for value in args_cli.video_camera_target),
        )
        video_recorder.capture(force=True)

    def _record_step() -> None:
        if video_recorder is not None:
            video_recorder.capture()

    approach_open_gripper_width = _approach_open_gripper_width(selected_world_grasp)
    print("[INFO]: Warming up simulation...", flush=True)
    for _ in range(max(1, int(0.1 / physics_dt))):
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)
        _record_step()
    _prepare_robot_start_pose(
        sim,
        scene,
        hand_open_width=approach_open_gripper_width,
        step_callback=_record_step,
    )

    robot = scene["robot"]
    part = scene["part"]
    part_pose_w = part.data.root_link_pose_w[0]
    print(
        "[INFO]: Spawned Isaac part pose "
        f"position=({float(part_pose_w[0]):.6f}, {float(part_pose_w[1]):.6f}, {float(part_pose_w[2]):.6f}) "
        f"orientation_wxyz=({float(part_pose_w[3]):.6f}, {float(part_pose_w[4]):.6f}, "
        f"{float(part_pose_w[5]):.6f}, {float(part_pose_w[6]):.6f})",
        flush=True,
    )
    if args_cli.moveit_motion_sequence_json is not None:
        sequence_payload = json.loads(args_cli.moveit_motion_sequence_json.read_text(encoding="utf-8"))
        labels = tuple(str(value) for value in sequence_payload["labels"])
        trajectories = {
            str(label): tuple(
                _moveit_waypoint_to_isaac(tuple(float(value) for value in waypoint)) for waypoint in waypoints
            )
            for label, waypoints in dict(sequence_payload["trajectories"]).items()
        }
        execution_result = execute_moveit_joint_trajectory_sequence(
            sim=sim,
            scene=scene,
            robot=robot,
            moveit_joint_trajectories=trajectories,
            labels=labels,
            fixed_gripper_width=approach_open_gripper_width,
            max_joint_speed_rad_s=float(args_cli.moveit_execution_speed_rad_s),
            step_callback=_record_step,
        )
        if video_recorder is not None:
            video_recorder.capture(force=True)
            video_recorder.close()
        args_cli.attempt_artifact.parent.mkdir(parents=True, exist_ok=True)
        args_cli.attempt_artifact.write_text(
            json.dumps(
                {
                    "execution": {
                        "success": bool(execution_result.success),
                        "status": execution_result.status,
                        "message": execution_result.message,
                        "diagnostics": dict(execution_result.diagnostics),
                    },
                    "sequence_plan": str(args_cli.moveit_motion_sequence_json),
                    "video_path": None if video_recorder is None else video_recorder.output_path,
                    "video_frame_count": 0 if video_recorder is None else video_recorder.frame_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not execution_result.success:
            raise SystemExit(1)
        return
    print("[INFO]: Executing pick attempt...", flush=True)
    if args_cli.moveit_plan_json is not None:
        print(f"[INFO]: Loading precomputed MoveIt plan from {args_cli.moveit_plan_json}.", flush=True)
        moveit_plan_payload, moveit_joint_trajectories = _load_moveit_plan_json(args_cli.moveit_plan_json)
        planned_grasp_id = moveit_plan_payload.get("selected_grasp_id")
        if planned_grasp_id not in ("", None, selected_grasp.grasp_id):
            raise RuntimeError(
                "Precomputed MoveIt plan grasp id "
                f"'{planned_grasp_id}' does not match selected Isaac grasp '{selected_grasp.grasp_id}'."
            )
    else:
        start_joint_positions = _moveit_start_joint_positions_from_args()
        if start_joint_positions is None:
            start_joint_positions = _current_isaac_arm_joint_positions(
                sim=sim,
                scene=scene,
                robot=robot,
                fixed_gripper_width=approach_open_gripper_width,
            )
            start_joint_positions = _isaac_waypoint_to_moveit(start_joint_positions)
        moveit_joint_trajectories = None
        planning_errors: list[str] = []
        for candidate_grasp, candidate_world_grasp in executable_candidates:
            print(
                f"[INFO]: Planning Isaac attempt with MoveIt for grasp {candidate_grasp.grasp_id}.",
                flush=True,
            )
            try:
                moveit_joint_trajectories = _plan_moveit_joint_trajectories(
                    world_grasp=candidate_world_grasp,
                    start_joint_positions=start_joint_positions,
                )
            except RuntimeError as exc:
                planning_errors.append(f"{candidate_grasp.grasp_id}: {exc}")
                print(f"[WARN]: MoveIt rejected grasp {candidate_grasp.grasp_id}: {exc}", flush=True)
                continue
            selected_grasp = candidate_grasp
            selected_world_grasp = candidate_world_grasp
            print(f"[INFO]: Selected MoveIt-planned grasp {selected_grasp.grasp_id}.", flush=True)
            break
        if moveit_joint_trajectories is None:
            tried = ", ".join(planning_errors[:8])
            if len(planning_errors) > 8:
                tried += f", ... ({len(planning_errors)} total)"
            result = SimpleNamespace(
                success=False,
                status="moveit_planning_failed",
                message=f"MoveIt could not plan any ground-feasible Isaac grasp. Tried: {tried}",
            )
            _write_attempt_artifact(
                bundle=bundle,
                placement_spec=placement_spec,
                object_pose_world=object_pose_world,
                statuses=statuses,
                selected_world_grasp=selected_world_grasp,
                execution_result=result,
            )
            raise RuntimeError(result.message)
    effective_close_width = _effective_close_gripper_width(selected_grasp)
    print(
        "[INFO]: Isaac gripper command sequence "
        f"approach_open_width={approach_open_gripper_width:.4f} "
        f"candidate_grasp_width={selected_world_grasp.gripper_width:.4f} "
        f"requested_close_width={float(args_cli.close_width):.4f} "
        f"effective_close_width={effective_close_width:.4f}.",
        flush=True,
    )
    _print_moveit_joint_trajectory_summary(moveit_joint_trajectories)
    execution_result = execute_pick_from_moveit_joint_trajectories(
        sim=sim,
        scene=scene,
        robot=robot,
        object_asset=part,
        moveit_joint_trajectories=moveit_joint_trajectories,
        open_gripper_width=approach_open_gripper_width,
        closed_gripper_width=effective_close_width,
        pregrasp_only=bool(args_cli.pregrasp_only),
        success_height_margin_m=float(args_cli.success_height_margin_m),
        max_joint_speed_rad_s=float(args_cli.moveit_execution_speed_rad_s),
        grasp_settle_time_s=float(args_cli.moveit_grasp_settle_time_s),
        gripper_close_duration_s=float(args_cli.gripper_close_duration_s),
        gripper_close_max_duration_s=float(args_cli.gripper_close_max_duration_s),
        postclose_hold_s=float(args_cli.postclose_hold_s),
        selected_gripper_width_m=float(selected_world_grasp.jaw_width),
        step_callback=_record_step,
    )
    if video_recorder is not None:
        video_recorder.capture(force=True)
    _write_attempt_artifact(
        bundle=bundle,
        placement_spec=placement_spec,
        object_pose_world=object_pose_world,
        statuses=statuses,
        selected_world_grasp=selected_world_grasp,
        execution_result=execution_result,
        video_recorder=video_recorder,
    )
    if video_recorder is not None:
        video_recorder.close()

    print(
        "[INFO]: Fabrica Isaac pickup attempt "
        f"support_face={placement_spec.support_face} yaw_deg={placement_spec.yaw_deg:.1f} "
        f"saved={len(bundle.candidates)} feasible={len(feasible_grasps)} "
        f"selected={selected_grasp.grasp_id} status={execution_result.status} success={execution_result.success}",
        flush=True,
    )
    if not execution_result.success:
        raise SystemExit(1)

    if args_cli.headless and args_cli.run_seconds <= 0.0:
        return

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
    run_error: BaseException | None = None
    try:
        run()
    except SystemExit as exc:
        run_error = exc
    except BaseException as exc:
        run_error = exc
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    finally:
        try:
            simulation_app.close()
        except SystemExit:
            if run_error is None:
                raise
    if run_error is not None:
        raise run_error
