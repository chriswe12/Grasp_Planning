"""Run one fixed KUKA/Y-gripper Isaac pickup without pipeline grasp switching."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT_JSON = Path("artifacts/plumbers_block0_kuka_pipeline_stage2_ground_feasible.json")
DEFAULT_PLAN_JSON = Path("artifacts/plumbers_block0_kuka_isaac_pick_attempt_moveit_plan.json")
DEFAULT_ROBOT_USD = Path("assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda")
DEFAULT_OBJECT_POSITION = (0.4252643585205078, 0.05988234281539917, 0.04)
DEFAULT_OBJECT_ORIENTATION_XYZW = (-0.7071067811865475, 0.0, 0.0, 0.7071067811865476)
DEFAULT_MOVEIT_JOINT_NAMES = tuple(f"lbr_A{index}" for index in range(1, 8))
DEFAULT_KUKA_MOVEIT_ARM_START_JOINT_VALUES = (
    0.0,
    0.5,
    0.0,
    -1.3962634015954636,
    0.0,
    1.1,
    0.0,
)
DEFAULT_KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M = 0.084


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--grasp-id", default="g1013")
    parser.add_argument("--robot-usd", type=Path, default=DEFAULT_ROBOT_USD)
    parser.add_argument(
        "--kuka-arm-actuator-profile",
        choices=("working", "source-usd"),
        default="source-usd",
        help=(
            "Arm actuator profile for KUKA IsaacLab execution. "
            "'source-usd' uses original USD arm stiffness/effort with default damping 80; "
            "'working' uses the old stiff tracking profile."
        ),
    )
    parser.add_argument(
        "--kuka-arm-damping",
        type=float,
        default=None,
        help="Override KUKA arm actuator damping for every arm joint while keeping the selected profile's stiffness/effort.",
    )
    parser.add_argument(
        "--part-usd",
        type=Path,
        default=None,
        help="Optional bundle-local part USD. If omitted, the script regenerates one from --input-json.",
    )
    parser.add_argument("--generated-usd-dir", type=Path, default=Path("artifacts/fixed_kuka_pick_assets"))
    parser.add_argument("--attempt-artifact", type=Path, default=Path("artifacts/fixed_kuka_isaac_pick_attempt.json"))
    parser.add_argument("--object-position", type=float, nargs=3, default=DEFAULT_OBJECT_POSITION)
    parser.add_argument("--object-orientation-xyzw", type=float, nargs=4, default=DEFAULT_OBJECT_ORIENTATION_XYZW)
    parser.add_argument(
        "--object-yaw-deg",
        type=float,
        default=None,
        help="Optional extra world-Z yaw applied to --object-orientation-xyzw.",
    )
    parser.add_argument("--object-mass-kg", type=float, default=None)
    parser.add_argument("--object-density-kg-m3", type=float, default=1240.0)
    parser.add_argument("--pregrasp-offset", type=float, default=0.10)
    parser.add_argument("--gripper-width-clearance", type=float, default=0.01)
    parser.add_argument("--open-width", type=float, default=DEFAULT_KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M)
    parser.add_argument("--close-width", type=float, default=0.0)
    parser.add_argument("--lift-height-m", type=float, default=0.08)
    parser.add_argument("--success-height-margin-m", type=float, default=0.05)
    parser.add_argument("--tcp-to-grasp-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--moveit-plan-json",
        type=Path,
        default=DEFAULT_PLAN_JSON,
        help="Optional precomputed MoveIt plan. If the path does not exist, this script plans through MoveIt.",
    )
    parser.add_argument("--force-replan", action="store_true")
    parser.add_argument(
        "--allow-unvalidated-plan-json",
        action="store_true",
        help="Allow loading an old MoveIt plan JSON that does not record the exact object/grasp/offset metadata.",
    )
    parser.add_argument("--moveit-frame-id", default="lbr_link_0")
    parser.add_argument("--moveit-target-position-signs", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--moveit-planning-group", default="arm")
    parser.add_argument("--moveit-pose-link", default="gripper_tcp")
    parser.add_argument("--moveit-namespace", default="/lbr")
    parser.add_argument("--moveit-joint-names", default=",".join(DEFAULT_MOVEIT_JOINT_NAMES))
    parser.add_argument(
        "--moveit-start-joint-positions",
        default=",".join(str(value) for value in DEFAULT_KUKA_MOVEIT_ARM_START_JOINT_VALUES),
    )
    parser.add_argument("--moveit-pipeline-id", default="")
    parser.add_argument("--moveit-planner-id", default="")
    parser.add_argument("--moveit-wait-for-moveit-timeout-s", type=float, default=15.0)
    parser.add_argument("--moveit-ik-timeout-s", type=float, default=2.0)
    parser.add_argument("--moveit-planning-time-s", type=float, default=5.0)
    parser.add_argument("--moveit-num-planning-attempts", type=int, default=5)
    parser.add_argument("--moveit-velocity-scale", type=float, default=0.05)
    parser.add_argument("--moveit-acceleration-scale", type=float, default=0.05)
    parser.add_argument("--moveit-execution-speed-rad-s", type=float, default=0.60)
    parser.add_argument("--preclose-hold-s", type=float, default=1.0)
    parser.add_argument("--postclose-hold-s", type=float, default=1.0)
    parser.add_argument("--gripper-close-duration-s", type=float, default=1.5)
    parser.add_argument("--gripper-close-max-duration-s", type=float, default=10.0)
    parser.add_argument("--require-close-target", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--allow-contact-stall-before-target",
        action="store_true",
        help="Allow lift after a gripper contact stall even if the commanded close target was not reached.",
    )
    parser.add_argument("--run-seconds-after", type=float, default=0.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
from isaaclab.sim.schemas import schemas_cfg  # noqa: E402
from isaaclab.sim.utils import bind_physics_material  # noqa: E402

from grasp_planning.envs import (  # noqa: E402
    DEFAULT_PART_DENSITY_KG_M3,
    ISAAC_MIN_CONTACT_OFFSET_M,
    make_fr3_part_scene_cfg,
)
from grasp_planning.envs.franka_collisions import expose_franka_mesh_collisions  # noqa: E402
from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    TriangleMesh,
    load_grasp_bundle,
    load_stl_mesh,
    quat_to_rotmat_xyzw,
)
from grasp_planning.grasping.grasp_transforms import saved_grasp_to_world_grasp  # noqa: E402
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.mujoco.scene_builder import write_temporary_triangle_mesh_stl  # noqa: E402
from grasp_planning.planning.fr3_motion_context import FR3MotionContext  # noqa: E402
from grasp_planning.planning.pick_execution import (  # noqa: E402
    GRIPPER_CLOSE_SETTLE_DURATION_S,
    PickExecutionResult,
    _command_gripper_width,
    _execute_moveit_waypoint_segment,
    _hold_arm_waypoint,
    _hold_arm_waypoint_until_settled,
    _moveit_waypoint_tensor,
    _object_root_z,
    _validate_object_lift,
    drive_robot_to_start_pose,
)
from grasp_planning.planning.trajectory_executor import TrajectoryExecutor  # noqa: E402
from grasp_planning.ros2.moveit_pose_commander import (  # noqa: E402
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    rclpy,
)
from grasp_planning.ros2.moveit_world_grasp import world_grasp_pose_targets  # noqa: E402
from grasp_planning.scene_defaults import ROBOT_BASE_ORIENTATION_XYZW, ROBOT_BASE_POSITION  # noqa: E402
from grasp_planning.start_poses import (  # noqa: E402
    gripper_joint_target_from_width,
    kuka_moveit_to_isaac_joint_positions,
)

ATTEMPT_ARTIFACT_WRITTEN = False


def _resolve_file(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return str(resolved)


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _quat_multiply_xyzw(lhs: tuple[float, float, float, float], rhs: tuple[float, float, float, float]):
    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _normalized_quat_xyzw(values) -> tuple[float, float, float, float]:
    quat = np.asarray(tuple(float(value) for value in values), dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Object orientation quaternion must have nonzero norm.")
    return tuple(float(value) for value in quat / norm)


def _object_orientation_xyzw_from_args() -> tuple[float, float, float, float]:
    base = _normalized_quat_xyzw(args_cli.object_orientation_xyzw)
    if args_cli.object_yaw_deg is None:
        return base
    half_yaw = np.deg2rad(float(args_cli.object_yaw_deg)) * 0.5
    yaw = (0.0, 0.0, float(np.sin(half_yaw)), float(np.cos(half_yaw)))
    return _normalized_quat_xyzw(_quat_multiply_xyzw(yaw, base))


def _moveit_joint_names() -> tuple[str, ...]:
    return _parse_csv_strings(args_cli.moveit_joint_names)


def _moveit_start_joint_positions() -> tuple[float, ...]:
    positions = _parse_csv_floats(args_cli.moveit_start_joint_positions)
    joint_names = _moveit_joint_names()
    if len(positions) != len(joint_names):
        raise ValueError(f"--moveit-start-joint-positions has {len(positions)} values, expected {len(joint_names)}.")
    return positions


def _using_kuka_moveit_joints() -> bool:
    return _moveit_joint_names() == DEFAULT_MOVEIT_JOINT_NAMES


def _moveit_waypoint_to_isaac(waypoint: tuple[float, ...]) -> tuple[float, ...]:
    if _using_kuka_moveit_joints():
        return kuka_moveit_to_isaac_joint_positions(waypoint)
    return tuple(float(value) for value in waypoint)


def _mesh_in_bundle_local_frame(bundle) -> TriangleMesh:
    mesh_global = load_stl_mesh(bundle.target_stl_path, scale=bundle.stl_scale)
    rot = quat_to_rotmat_xyzw(bundle.local_frame_orientation_xyzw_world)
    translation = np.asarray(bundle.local_frame_origin_world, dtype=float)
    vertices_local = (np.asarray(mesh_global.vertices_obj, dtype=float) - translation[None, :]) @ rot
    return TriangleMesh(vertices_obj=vertices_local, faces=np.asarray(mesh_global.faces, dtype=np.int64))


def _mesh_collision_cfg():
    return schemas_cfg.ConvexDecompositionPropertiesCfg()


def _object_mass_properties_cfg():
    if args_cli.object_mass_kg is not None:
        return sim_utils.MassPropertiesCfg(mass=float(args_cli.object_mass_kg))
    density = (
        DEFAULT_PART_DENSITY_KG_M3 if args_cli.object_density_kg_m3 is None else float(args_cli.object_density_kg_m3)
    )
    return sim_utils.MassPropertiesCfg(density=density)


def _generate_part_usd(bundle, mesh_local: TriangleMesh) -> str:
    output_dir = args_cli.generated_usd_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_usd = output_dir / f"{args_cli.input_json.stem}_fixed_pick_bundle_local.usd"
    temp_stl = write_temporary_triangle_mesh_stl(
        mesh_local,
        prefix=f"{args_cli.input_json.stem}_fixed_pick_",
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
    try:
        temp_stl.unlink()
    except FileNotFoundError:
        pass
    converted_path = str(Path(converter.usd_path).resolve())
    print(f"[INFO]: Generated fixed-pick part USD: {converted_path}", flush=True)
    return converted_path


def _part_usd_path(bundle, mesh_local: TriangleMesh) -> str:
    if args_cli.part_usd is not None:
        print("[WARN]: Using provided part USD. It must be in the bundle-local frame.", flush=True)
        return _resolve_file(args_cli.part_usd)
    return _generate_part_usd(bundle, mesh_local)


def _bind_high_friction_contact_material(stage, *, root_paths: tuple[str, ...]) -> int:
    material_path = "/World/Looks/fixed_pick_high_friction_contact_material"
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
        if not stage.GetPrimAtPath(root_path).IsValid():
            print(f"[WARN]: Cannot bind contact material; missing prim: {root_path}", flush=True)
            continue
        bind_physics_material(root_path, material_path, stage=stage, stronger_than_descendants=True)
        bound_count += 1
    return bound_count


def _is_generated_kuka_y_gripper_usd(path: str | Path) -> bool:
    resolved = str(path).replace("\\", "/")
    return "kuka_iiwa7_y_gripper" in resolved


def _build_scene(*, part_usd_path: str, object_pose_world: ObjectWorldPose):
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
    sim.set_camera_view([1.1, -0.9, 0.65], [0.42, 0.06, 0.08])

    scene_cfg = make_fr3_part_scene_cfg(
        fr3_asset_path=_resolve_file(args_cli.robot_usd),
        part_usd_path=part_usd_path,
        part_position=object_pose_world.position_world,
        part_orientation_xyzw=object_pose_world.orientation_xyzw_world,
        part_mass_kg=None if args_cli.object_mass_kg is None else float(args_cli.object_mass_kg),
        part_density_kg_m3=args_cli.object_density_kg_m3,
        robot_base_position=ROBOT_BASE_POSITION,
        robot_base_orientation_xyzw=ROBOT_BASE_ORIENTATION_XYZW,
        kuka_arm_actuator_profile=args_cli.kuka_arm_actuator_profile.replace("-", "_"),
        kuka_arm_damping_override=args_cli.kuka_arm_damping,
    )
    scene = InteractiveScene(scene_cfg)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    if _is_generated_kuka_y_gripper_usd(args_cli.robot_usd):
        print(
            "[INFO]: Skipping Franka visual mesh collision exposure for generated KUKA/Y-gripper USD; "
            "using authored collision hulls.",
            flush=True,
        )
    else:
        enabled_count, _ = expose_franka_mesh_collisions(
            mesh_path_patterns=(
                r"(?:panda|fr3)_hand",
                r"(?:panda|fr3)_leftfinger",
                r"(?:panda|fr3)_rightfinger",
                r"finger",
                r"gripper",
            )
        )
        print(f"[INFO]: Enabled collision on {enabled_count} robot gripper mesh prims.", flush=True)
    bound_count = _bind_high_friction_contact_material(
        omni.usd.get_context().get_stage(),
        root_paths=("/World/envs/env_0/Robot", "/World/envs/env_0/Part"),
    )
    print(f"[INFO]: Bound high-friction contact material to {bound_count} subtrees.", flush=True)
    sim.reset()
    scene.reset()
    return sim, scene


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


def _moveit_config() -> MoveItPoseCommanderConfig:
    return MoveItPoseCommanderConfig(
        planning_group=str(args_cli.moveit_planning_group),
        pose_link=str(args_cli.moveit_pose_link),
        joint_names=_moveit_joint_names(),
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
        avoid_collisions=False,
    )


def _plan_moveit(world_grasp) -> dict[str, tuple[tuple[float, ...], ...]]:
    if rclpy is None:
        raise RuntimeError("ROS2 MoveIt dependencies are unavailable. Pass --moveit-plan-json or source MoveIt.")
    targets = world_grasp_pose_targets(
        world_grasp,
        frame_id=str(args_cli.moveit_frame_id),
        lift_height_m=float(args_cli.lift_height_m),
        position_signs=tuple(float(v) for v in args_cli.moveit_target_position_signs),
        tcp_to_grasp_offset=tuple(float(v) for v in args_cli.tcp_to_grasp_offset),
    )
    initialized_here = False
    commander = None
    try:
        if not rclpy.ok():
            rclpy.init()
            initialized_here = True
        moveit_config = _moveit_config()
        commander = MoveItPoseCommander(moveit_config, node_name="fixed_kuka_pick_moveit")
        commander.wait_for_moveit(require_execute=False)
        planned: dict[str, tuple[tuple[float, ...], ...]] = {}
        current_start = _moveit_start_joint_positions()
        for label in ("pregrasp", "grasp", "lift"):
            print(f"[INFO]: Planning fixed {label} pose through MoveIt.", flush=True)
            trajectory, message = commander.plan_to_pose(
                targets[label],
                label=f"fixed_kuka_pick_{label}",
                start_joint_positions=current_start,
            )
            if trajectory is None:
                raise RuntimeError(f"MoveIt failed to plan {label}: {message}")
            moveit_waypoints = _trajectory_waypoints_for_joints(trajectory, joint_names=moveit_config.joint_names)
            planned[label] = tuple(_moveit_waypoint_to_isaac(waypoint) for waypoint in moveit_waypoints)
            current_start = moveit_waypoints[-1]
            print(f"[INFO]: MoveIt fixed {label} plan returned {len(moveit_waypoints)} waypoints.", flush=True)
        return planned
    finally:
        if commander is not None:
            commander.destroy_node()
        if initialized_here:
            rclpy.shutdown()


def _load_plan_json(
    path: Path, *, expected_grasp_id: str
) -> tuple[dict[str, object], dict[str, tuple[tuple[float, ...], ...]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan_grasp_id = str(payload.get("selected_grasp_id", ""))
    if plan_grasp_id and plan_grasp_id != expected_grasp_id:
        raise ValueError(f"Plan grasp id '{plan_grasp_id}' does not match requested '{expected_grasp_id}'.")
    _merge_sibling_attempt_metadata(payload, path=path)
    raw_trajectories = payload.get("trajectories")
    if not isinstance(raw_trajectories, dict):
        raise ValueError(f"MoveIt plan JSON '{path}' is missing trajectories.")
    trajectories: dict[str, tuple[tuple[float, ...], ...]] = {}
    for label in ("pregrasp", "grasp", "lift"):
        raw_waypoints = raw_trajectories.get(label)
        if not isinstance(raw_waypoints, list):
            raise ValueError(f"MoveIt plan JSON '{path}' is missing trajectory '{label}'.")
        trajectories[label] = tuple(
            _moveit_waypoint_to_isaac(tuple(float(v) for v in waypoint)) for waypoint in raw_waypoints
        )
    return payload, trajectories


def _sibling_attempt_artifact_path(plan_path: Path) -> Path | None:
    stem = plan_path.stem
    suffix = "_moveit_plan"
    if not stem.endswith(suffix):
        return None
    return plan_path.with_name(f"{stem[: -len(suffix)]}{plan_path.suffix}")


def _merge_sibling_attempt_metadata(payload: dict[str, object], *, path: Path) -> None:
    if isinstance(payload.get("selected_world_grasp"), dict) and isinstance(payload.get("moveit"), dict):
        return
    sibling = _sibling_attempt_artifact_path(path)
    if sibling is None or not sibling.is_file():
        return
    try:
        attempt = json.loads(sibling.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN]: Could not read sibling attempt metadata from {sibling}: {exc}", flush=True)
        return
    if payload.get("selected_grasp_id") != attempt.get("selected_grasp_id"):
        return
    if not isinstance(payload.get("selected_world_grasp"), dict) and isinstance(
        attempt.get("selected_world_grasp"),
        dict,
    ):
        payload["selected_world_grasp"] = attempt["selected_world_grasp"]
    if not isinstance(payload.get("moveit"), dict) and isinstance(attempt.get("moveit"), dict):
        payload["moveit"] = attempt["moveit"]
    payload.setdefault("metadata_source", str(sibling))


def _vector_close(actual, expected, *, tolerance: float = 1.0e-5) -> bool:
    if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
        return False
    return all(abs(float(a) - float(e)) <= float(tolerance) for a, e in zip(actual, expected, strict=True))


def _plan_validation_error(payload: dict[str, object], world_grasp) -> str | None:
    selected_world_grasp = payload.get("selected_world_grasp")
    if not isinstance(selected_world_grasp, dict):
        return "cached plan JSON has no selected_world_grasp metadata"
    if not _vector_close(selected_world_grasp.get("position_w"), world_grasp.position_w):
        return "cached plan JSON was planned for a different grasp position"
    if not _vector_close(selected_world_grasp.get("orientation_xyzw"), world_grasp.orientation_xyzw):
        return "cached plan JSON was planned for a different grasp orientation"

    moveit = payload.get("moveit")
    if not isinstance(moveit, dict):
        return "cached plan JSON has no MoveIt metadata"
    if str(moveit.get("frame_id", "")) != str(args_cli.moveit_frame_id):
        return "cached plan JSON uses a different MoveIt frame"
    if str(moveit.get("pose_link", "")) != str(args_cli.moveit_pose_link):
        return "cached plan JSON uses a different MoveIt pose link"
    if not _vector_close(moveit.get("target_position_signs"), args_cli.moveit_target_position_signs):
        return "cached plan JSON uses different target position signs"
    if not _vector_close(moveit.get("tcp_to_grasp_offset"), args_cli.tcp_to_grasp_offset):
        return "cached plan JSON uses a different TCP-to-grasp offset"
    return None


def _resolve_moveit_trajectories(
    world_grasp,
) -> tuple[str, dict[str, object] | None, dict[str, tuple[tuple[float, ...], ...]]]:
    plan_json = args_cli.moveit_plan_json
    if not args_cli.force_replan and plan_json is not None and plan_json.expanduser().is_file():
        payload, trajectories = _load_plan_json(
            plan_json.expanduser().resolve(), expected_grasp_id=world_grasp.grasp_id
        )
        validation_error = _plan_validation_error(payload, world_grasp)
        if validation_error is not None:
            message = (
                f"{validation_error}: {plan_json}. Re-run with --force-replan, or pass "
                "--allow-unvalidated-plan-json if you intentionally want to replay this exact cached joint path."
            )
            if not args_cli.allow_unvalidated_plan_json:
                raise ValueError(message)
            print(f"[WARN]: {message}", flush=True)
        print(f"[INFO]: Loaded fixed MoveIt plan JSON: {plan_json}", flush=True)
        return str(plan_json), payload, trajectories
    trajectories = _plan_moveit(world_grasp)
    return "planned_live", None, trajectories


def _contact_stall_matches_selected_grasp(close_diagnostics: dict[str, object], world_grasp) -> bool:
    final_positions = close_diagnostics.get("gripper_close_final_joint_positions")
    if not isinstance(final_positions, list) or not final_positions:
        close_diagnostics["gripper_close_contact_stall_accept_reason"] = "missing final finger positions"
        return False

    final_step_delta = close_diagnostics.get("gripper_close_final_max_step_delta")
    if final_step_delta is not None and float(final_step_delta) > 1.0e-4:
        close_diagnostics["gripper_close_contact_stall_accept_reason"] = (
            f"finger still moving at {float(final_step_delta):.6f} m/step"
        )
        return False

    final_close = max(abs(float(value)) for value in final_positions)
    selected_jaw_width_m = float(world_grasp.jaw_width)
    expected_close = abs(gripper_joint_target_from_width("left_finger_joint", selected_jaw_width_m))
    tolerance = 0.003
    close_diagnostics["gripper_close_contact_stall_max_abs_joint_position"] = float(final_close)
    close_diagnostics["gripper_close_contact_stall_expected_min_joint_position"] = float(expected_close)
    close_diagnostics["gripper_close_contact_stall_expected_tolerance_m"] = float(tolerance)
    close_diagnostics["gripper_close_contact_stall_selected_jaw_width_m"] = selected_jaw_width_m
    accepted = final_close + tolerance >= expected_close
    close_diagnostics["gripper_close_contact_stall_accepted"] = bool(accepted)
    if not accepted:
        close_diagnostics["gripper_close_contact_stall_accept_reason"] = (
            f"finger only closed to {final_close:.4f} m, expected at least {expected_close:.4f} m "
            f"for selected jaw width {selected_jaw_width_m:.4f} m"
        )
    return accepted


def _run_fixed_sequence(*, sim, scene, world_grasp, moveit_joint_trajectories) -> PickExecutionResult:
    robot = scene["robot"]
    part = scene["part"]
    physics_dt = sim.get_physics_dt()
    initial_object_z = _object_root_z(part)
    observed_lift_object_max_z = None
    capture_lift_object_z = False

    def _step_callback() -> None:
        nonlocal observed_lift_object_max_z
        if not capture_lift_object_z:
            return
        object_z = _object_root_z(part)
        if object_z is None:
            return
        observed_lift_object_max_z = (
            object_z if observed_lift_object_max_z is None else max(observed_lift_object_max_z, object_z)
        )

    print("[INFO]: Warming up fixed scene.", flush=True)
    for _ in range(max(1, int(0.1 / physics_dt))):
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)

    print(f"[INFO]: Driving robot to start with open_width={float(args_cli.open_width):.4f}.", flush=True)
    drive_robot_to_start_pose(sim, scene, hand_open_width=float(args_cli.open_width), step_callback=_step_callback)
    context = FR3MotionContext(robot=robot, scene=scene, sim=sim, fixed_gripper_width=float(args_cli.open_width))
    print(
        "[INFO]: Fixed motion context "
        f"ee_body={context.ee_body_name} arm_joints={list(context.arm_joint_names)} "
        f"hand_joints={list(context.hand_joint_names)}.",
        flush=True,
    )
    executor = TrajectoryExecutor(
        context,
        max_joint_speed_rad_s=float(args_cli.moveit_execution_speed_rad_s),
        step_callback=_step_callback,
    )
    first_waypoint = _moveit_waypoint_tensor(
        context=context,
        moveit_joint_trajectories=moveit_joint_trajectories,
        label="pregrasp",
        index=0,
    )
    context.reset_joint_state(first_waypoint, steps=5)

    diagnostics: dict[str, object] = {
        "open_gripper_width_m": float(args_cli.open_width),
        "closed_gripper_width_m": float(args_cli.close_width),
        "moveit_execution_speed_rad_s": float(args_cli.moveit_execution_speed_rad_s),
        "preclose_hold_s": float(args_cli.preclose_hold_s),
        "postclose_hold_s": float(args_cli.postclose_hold_s),
    }

    for label in ("pregrasp", "grasp"):
        print(f"[INFO]: Executing fixed {label} trajectory.", flush=True)
        ok, detail = _execute_moveit_waypoint_segment(
            context=context,
            executor=executor,
            moveit_joint_trajectories=moveit_joint_trajectories,
            label=label,
        )
        diagnostics[f"{label}_settled"] = bool(ok)
        diagnostics[f"{label}_detail"] = detail
        if not ok:
            return PickExecutionResult(
                False,
                f"moveit_{label}_failed",
                f"Fixed {label} trajectory did not settle: {detail}",
                diagnostics=diagnostics,
            )

    grasp_waypoint = _moveit_waypoint_tensor(
        context=context,
        moveit_joint_trajectories=moveit_joint_trajectories,
        label="grasp",
    )
    if float(args_cli.preclose_hold_s) > 0.0:
        diagnostics.update(
            _hold_arm_waypoint_until_settled(
                context=context,
                waypoint=grasp_waypoint,
                duration_s=float(args_cli.preclose_hold_s),
                tolerance_rad=0.015,
                step_callback=_step_callback,
            )
        )

    print(
        f"[INFO]: Closing gripper at fixed grasp before lift, close_width={float(args_cli.close_width):.4f}.",
        flush=True,
    )
    close_diagnostics = _command_gripper_width(
        sim=sim,
        scene=scene,
        robot=robot,
        width=float(args_cli.close_width),
        duration_s=float(args_cli.gripper_close_duration_s),
        max_duration_s=float(args_cli.gripper_close_max_duration_s),
        hold_context=context,
        hold_arm_waypoint=grasp_waypoint,
        settle_duration_s=GRIPPER_CLOSE_SETTLE_DURATION_S,
        min_contact_motion_m=max(
            0.001, min(0.003, 0.125 * abs(float(args_cli.open_width) - float(args_cli.close_width)))
        ),
        force_joint_state=False,
        step_callback=_step_callback,
    )
    diagnostics.update(dict(close_diagnostics))
    close_status = str(close_diagnostics.get("gripper_close_status", "unknown"))
    close_is_acceptable = close_status == "target_reached"
    if close_status in {"contact_stalled", "max_duration_elapsed"}:
        close_is_acceptable = bool(args_cli.allow_contact_stall_before_target) or _contact_stall_matches_selected_grasp(
            diagnostics,
            world_grasp,
        )
    print(
        "[INFO]: Fixed gripper close finished "
        f"status={close_status} duration_s={float(close_diagnostics.get('gripper_close_duration_s', 0.0)):.3f} "
        f"final_error={close_diagnostics.get('gripper_close_final_max_position_error', 'n/a')}.",
        flush=True,
    )
    if not close_is_acceptable:
        return PickExecutionResult(
            False,
            "gripper_close_failed",
            "Fixed gripper close did not reach the target or a plausible selected-grasp contact: "
            f"status={close_status}, reason={diagnostics.get('gripper_close_contact_stall_accept_reason', 'n/a')}.",
            diagnostics=diagnostics,
        )

    context.fixed_gripper_width = float(args_cli.close_width)
    if float(args_cli.postclose_hold_s) > 0.0:
        print(f"[INFO]: Holding closed gripper for {float(args_cli.postclose_hold_s):.2f}s before lift.", flush=True)
        _hold_arm_waypoint(
            context=context,
            waypoint=grasp_waypoint,
            duration_s=float(args_cli.postclose_hold_s),
            step_callback=_step_callback,
        )

    print("[INFO]: Executing fixed lift trajectory after close.", flush=True)
    capture_lift_object_z = True
    try:
        ok, detail = _execute_moveit_waypoint_segment(
            context=context,
            executor=executor,
            moveit_joint_trajectories=moveit_joint_trajectories,
            label="lift",
        )
    finally:
        capture_lift_object_z = False
    diagnostics["lift_settled"] = bool(ok)
    diagnostics["lift_detail"] = detail
    if not ok:
        return PickExecutionResult(False, "moveit_lift_failed", f"Fixed lift failed: {detail}", diagnostics=diagnostics)

    lift_result = _validate_object_lift(
        object_asset=part,
        initial_object_z=initial_object_z,
        success_height_margin_m=float(args_cli.success_height_margin_m),
        observed_object_max_z=observed_lift_object_max_z,
        extra_diagnostics=diagnostics,
    )
    if lift_result is not None:
        return lift_result
    return PickExecutionResult(True, "ok", "Fixed KUKA pick sequence completed.", diagnostics=diagnostics)


def _write_artifact(
    *, bundle, part_usd_path: str, object_pose_world, world_grasp, plan_source: str, plan_payload, result
) -> None:
    global ATTEMPT_ARTIFACT_WRITTEN
    artifact = {
        "script": "scripts/run_fixed_kuka_pick_in_isaac.py",
        "input_json": str(args_cli.input_json),
        "part_usd": part_usd_path,
        "target_mesh_path": bundle.target_stl_path,
        "selected_grasp_id": world_grasp.grasp_id,
        "object_pose_world": {
            "position_world": list(object_pose_world.position_world),
            "orientation_xyzw_world": list(object_pose_world.orientation_xyzw_world),
        },
        "selected_world_grasp": {
            "position_w": list(world_grasp.position_w),
            "orientation_xyzw": list(world_grasp.orientation_xyzw),
            "pregrasp_position_w": list(world_grasp.pregrasp_position_w),
            "jaw_width": float(world_grasp.jaw_width),
            "gripper_width": float(world_grasp.gripper_width),
        },
        "moveit": {
            "plan_source": plan_source,
            "plan_json_grasp_id": None if plan_payload is None else plan_payload.get("selected_grasp_id"),
            "frame_id": args_cli.moveit_frame_id,
            "target_position_signs": list(args_cli.moveit_target_position_signs),
            "tcp_to_grasp_offset": list(args_cli.tcp_to_grasp_offset),
            "planning_group": args_cli.moveit_planning_group,
            "pose_link": args_cli.moveit_pose_link,
            "namespace": args_cli.moveit_namespace,
            "joint_names": list(_moveit_joint_names()),
            "start_joint_positions": list(_moveit_start_joint_positions()),
        },
        "execution": {
            "success": bool(result.success),
            "status": result.status,
            "message": result.message,
            "object_lift_height_m": result.object_lift_height_m,
            "target_lift_height_m": result.target_lift_height_m,
            "diagnostics": dict(result.diagnostics or {}),
        },
    }
    output = args_cli.attempt_artifact.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    ATTEMPT_ARTIFACT_WRITTEN = True
    print(f"[INFO]: Wrote fixed-pick artifact: {output}", flush=True)


def main() -> None:
    bundle = load_grasp_bundle(args_cli.input_json)
    grasp = next((candidate for candidate in bundle.candidates if candidate.grasp_id == args_cli.grasp_id), None)
    if grasp is None:
        raise RuntimeError(f"Grasp id '{args_cli.grasp_id}' was not found in {args_cli.input_json}.")

    object_pose_world = ObjectWorldPose(
        position_world=tuple(float(value) for value in args_cli.object_position),
        orientation_xyzw_world=_object_orientation_xyzw_from_args(),
    )
    world_grasp = saved_grasp_to_world_grasp(
        grasp,
        object_pose_world,
        pregrasp_offset=float(args_cli.pregrasp_offset),
        gripper_width_clearance=float(args_cli.gripper_width_clearance),
    )
    print(
        "[INFO]: Fixed pick grasp "
        f"id={world_grasp.grasp_id} grasp_w={world_grasp.position_w} "
        f"pregrasp_w={world_grasp.pregrasp_position_w} width={world_grasp.gripper_width:.4f}.",
        flush=True,
    )

    FR3MotionContext._TCP_TO_GRASP_CENTER_OFFSET = tuple(float(v) for v in args_cli.tcp_to_grasp_offset)
    plan_source, plan_payload, moveit_joint_trajectories = _resolve_moveit_trajectories(world_grasp)
    mesh_local = _mesh_in_bundle_local_frame(bundle)
    part_usd_path = _part_usd_path(bundle, mesh_local)
    sim, scene = _build_scene(part_usd_path=part_usd_path, object_pose_world=object_pose_world)

    result = _run_fixed_sequence(
        sim=sim,
        scene=scene,
        world_grasp=world_grasp,
        moveit_joint_trajectories=moveit_joint_trajectories,
    )
    _write_artifact(
        bundle=bundle,
        part_usd_path=part_usd_path,
        object_pose_world=object_pose_world,
        world_grasp=world_grasp,
        plan_source=plan_source,
        plan_payload=plan_payload,
        result=result,
    )
    print(f"[RESULT]: success={result.success} status={result.status} message={result.message}", flush=True)
    if float(args_cli.run_seconds_after) > 0.0:
        steps = int(float(args_cli.run_seconds_after) / sim.get_physics_dt())
        for _ in range(max(1, steps)):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
    if not result.success:
        raise RuntimeError(result.message)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR]: Fixed KUKA pick failed: {exc}", flush=True)
        traceback.print_exc()
        if ATTEMPT_ARTIFACT_WRITTEN:
            print("[INFO]: Keeping detailed fixed-pick artifact from failed run.", flush=True)
        else:
            failure = SimpleNamespace(
                success=False,
                status="exception",
                message=str(exc),
                object_lift_height_m=None,
                target_lift_height_m=float(args_cli.success_height_margin_m),
                diagnostics={},
            )
            args_cli.attempt_artifact.parent.mkdir(parents=True, exist_ok=True)
            args_cli.attempt_artifact.write_text(
                json.dumps(
                    {"script": "scripts/run_fixed_kuka_pick_in_isaac.py", "execution": failure.__dict__}, indent=2
                ),
                encoding="utf-8",
            )
        os._exit(1)
    finally:
        simulation_app.close()
