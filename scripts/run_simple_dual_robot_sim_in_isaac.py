#!/usr/bin/env python3
"""Replay a dual-MoveIt holder/pickup plan in a two-KUKA Isaac scene."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan-json", type=Path, required=True)
parser.add_argument(
    "--robot-usd",
    type=Path,
    default=Path("assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda"),
)
parser.add_argument(
    "--attempt-artifact",
    type=Path,
    default=Path("artifacts/dual_grasp_planning/plumbers_block/simple_dual_robot_sim_attempt.json"),
)
parser.add_argument("--object-density-kg-m3", type=float, default=1240.0)
parser.add_argument("--static-friction", type=float, default=5.0)
parser.add_argument("--dynamic-friction", type=float, default=4.0)
parser.add_argument("--gripper-effort-limit", type=float, default=200.0)
parser.add_argument(
    "--critical-damping-ratio",
    "--arm-critical-damping-ratio",
    dest="critical_damping_ratio",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--unloaded-max-joint-speed-rad-s",
    type=float,
    default=1.00,
    help="Maximum arm speed before either gripper closes on a part.",
)
parser.add_argument(
    "--loaded-max-joint-speed-rad-s",
    type=float,
    default=0.70,
    help="Maximum inserter speed while transporting the grasped part.",
)
parser.add_argument(
    "--max-joint-speed-rad-s",
    type=float,
    default=None,
    help="Compatibility override that applies one maximum speed to every segment.",
)
parser.add_argument(
    "--trajectory-waypoint-tolerance-rad",
    type=float,
    default=0.030,
    help="Maximum final joint tracking error accepted for non-contact MoveIt segments.",
)
parser.add_argument(
    "--contact-pose-tolerance-rad",
    type=float,
    default=0.005,
    help=(
        "Maximum final joint tracking error before closing either gripper. "
        "This must be tighter than the general transit tolerance."
    ),
)
parser.add_argument(
    "--trajectory-final-settle-time-s",
    type=float,
    default=2.0,
    help="Maximum time to hold each final arm waypoint before declaring it unsettled.",
)
parser.add_argument("--grasp-settle-time-s", type=float, default=1.0)
parser.add_argument("--gripper-close-duration-s", type=float, default=3.0)
parser.add_argument("--gripper-contact-preload-m", type=float, default=0.0004)
parser.add_argument(
    "--close-width",
    type=float,
    default=0.0,
    help=(
        "Requested closed gripper width. The default 0.0 uses the same KUKA "
        "effective close command as run_pipeline: min(0.001 m, selected jaw width)."
    ),
)
parser.add_argument(
    "--gripper-close-max-duration-s",
    type=float,
    default=10.0,
)
parser.add_argument(
    "--finger-contact-min-force-n",
    type=float,
    default=0.25,
    help=(
        "Minimum filtered force required on each finger to accept bilateral "
        "contact with that gripper's selected object."
    ),
)
parser.add_argument("--postclose-hold-s", type=float, default=1.0)
parser.add_argument("--final-hold-s", type=float, default=2.0)
parser.add_argument(
    "--holder-only",
    action="store_true",
    help="Execute only holder pregrasp/grasp/close and hold for stress testing.",
)
parser.add_argument(
    "--holder-sequence-json",
    type=Path,
    default=None,
    help=(
        "JSON containing ordered holder-only plan paths. Executes every case "
        "in one Isaac scene without resetting the holder articulation."
    ),
)
parser.add_argument("--record-video", type=Path, default=None)
parser.add_argument("--video-fps", type=float, default=30.0)
parser.add_argument("--video-width", type=int, default=960)
parser.add_argument("--video-height", type=int, default=540)
parser.add_argument(
    "--video-camera-eye",
    type=float,
    nargs=3,
    default=(1.55, -1.20, 0.95),
)
parser.add_argument(
    "--video-camera-target",
    type=float,
    nargs=3,
    default=(0.52, -0.05, 0.20),
)
parser.add_argument("--base-position-tolerance-m", type=float, default=0.025)
parser.add_argument(
    "--base-orientation-tolerance-rad",
    type=float,
    default=0.20,
)
parser.add_argument(
    "--incoming-position-tolerance-m",
    type=float,
    default=0.040,
)
parser.add_argument(
    "--incoming-orientation-tolerance-rad",
    type=float,
    default=0.20,
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.object_density_kg_m3 <= 0.0:
    parser.error("--object-density-kg-m3 must be positive.")
if args_cli.static_friction <= 0.0 or args_cli.dynamic_friction <= 0.0:
    parser.error("--static-friction and --dynamic-friction must be positive.")
if args_cli.gripper_effort_limit <= 0.0:
    parser.error("--gripper-effort-limit must be positive.")
if args_cli.critical_damping_ratio <= 0.0:
    parser.error("--critical-damping-ratio must be positive.")
if args_cli.gripper_contact_preload_m < 0.0:
    parser.error("--gripper-contact-preload-m must be non-negative.")
if args_cli.trajectory_waypoint_tolerance_rad <= 0.0:
    parser.error("--trajectory-waypoint-tolerance-rad must be positive.")
if args_cli.contact_pose_tolerance_rad <= 0.0:
    parser.error("--contact-pose-tolerance-rad must be positive.")
if args_cli.contact_pose_tolerance_rad > args_cli.trajectory_waypoint_tolerance_rad:
    parser.error("--contact-pose-tolerance-rad must be no larger than --trajectory-waypoint-tolerance-rad.")
if args_cli.trajectory_final_settle_time_s <= 0.0:
    parser.error("--trajectory-final-settle-time-s must be positive.")
if args_cli.unloaded_max_joint_speed_rad_s <= 0.0:
    parser.error("--unloaded-max-joint-speed-rad-s must be positive.")
if args_cli.loaded_max_joint_speed_rad_s <= 0.0:
    parser.error("--loaded-max-joint-speed-rad-s must be positive.")
if args_cli.close_width < 0.0:
    parser.error("--close-width must be non-negative.")
if args_cli.finger_contact_min_force_n < 0.0:
    parser.error("--finger-contact-min-force-n must be non-negative.")
if not 0.0 < args_cli.base_orientation_tolerance_rad <= math.pi:
    parser.error("--base-orientation-tolerance-rad must be in (0, pi].")
if not 0.0 < args_cli.incoming_orientation_tolerance_rad <= math.pi:
    parser.error("--incoming-orientation-tolerance-rad must be in (0, pi].")
if args_cli.max_joint_speed_rad_s is not None and args_cli.max_joint_speed_rad_s <= 0.0:
    parser.error("--max-joint-speed-rad-s must be positive when provided.")
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

from grasp_planning.envs.fr3_part_env import (  # noqa: E402
    ISAAC_MIN_CONTACT_OFFSET_M,
    make_dual_kuka_assembly_scene_cfg,
)
from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    quat_to_rotmat_xyzw,
    rotmat_to_quat_xyzw,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import (  # noqa: E402
    TriangleMesh,
)
from grasp_planning.grasping.mesh_io import load_triangle_mesh  # noqa: E402
from grasp_planning.mujoco.scene_builder import (  # noqa: E402
    write_temporary_triangle_mesh_stl,
)
from grasp_planning.pipeline.dual_robot_simple_sim import (  # noqa: E402
    DEFAULT_FLOOR_Z_WORLD_M,
    source_local_subassembly_mesh,
)
from grasp_planning.planning.fr3_motion_context import (  # noqa: E402
    FR3MotionContext,
)
from grasp_planning.planning.pick_execution import (  # noqa: E402
    GRIPPER_CLOSE_SETTLE_DURATION_S,
    _command_gripper_width,
    _kuka_contact_stall_matches_grasp_width,
)
from grasp_planning.planning.trajectory_executor import (  # noqa: E402
    TrajectoryExecutor,
)
from grasp_planning.planning.types import JointTrajectory  # noqa: E402
from grasp_planning.start_poses import (  # noqa: E402
    KUKA_MOVEIT_ARM_START_JOINT_VALUES,
    KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
    gripper_joint_target_from_width,
    kuka_gripper_approach_width,
    kuka_moveit_to_isaac_joint_positions,
)

ROLE_ROBOT_NAMES = {
    "holder": "lbr_one",
    "inserter": "lbr_two",
}
from grasp_planning.video import OpenCvVideoWriter  # noqa: E402


class IsaacVideoRecorder:
    def __init__(self, *, camera: Camera, sim, output_path: Path) -> None:
        self._camera = camera
        self._sim = sim
        self._writer = OpenCvVideoWriter(
            output_path,
            fps=float(args_cli.video_fps),
            width=int(args_cli.video_width),
            height=int(args_cli.video_height),
        )
        self._capture_interval_s = 1.0 / float(args_cli.video_fps)
        self._next_capture_time_s = 0.0
        self._elapsed_s = 0.0
        self.output_path = str(output_path.expanduser().resolve())

    @property
    def frame_count(self) -> int:
        return int(self._writer.frame_count)

    def set_view(self) -> None:
        eye = torch.tensor([args_cli.video_camera_eye], dtype=torch.float32, device=self._sim.device)
        target = torch.tensor([args_cli.video_camera_target], dtype=torch.float32, device=self._sim.device)
        self._camera.set_world_poses_from_view(eye, target)

    def capture(self, *, force: bool = False) -> None:
        dt = float(self._sim.get_physics_dt())
        self._elapsed_s += dt
        if not force and self._elapsed_s + 1.0e-9 < self._next_capture_time_s:
            return
        self._camera.update(dt=dt)
        raw = self._camera.data.output.get("rgb")
        if raw is None:
            return
        frame = raw[0]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        self._writer.append_rgb(np.asarray(frame))
        while self._next_capture_time_s <= self._elapsed_s + 1.0e-9:
            self._next_capture_time_s += self._capture_interval_s

    def close(self) -> None:
        self._writer.close()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in '{path}'.")
    return payload


def _vec(
    raw: object,
    *,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw)  # type: ignore[arg-type]
    if len(values) != length:
        raise ValueError(f"{field_name} must contain {length} values, got {len(values)}.")
    return values


def _pose(
    raw: dict[str, object],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]:
    return (
        _vec(
            raw["position_world_m"],
            length=3,
            field_name="position_world_m",
        ),
        _vec(
            raw["orientation_xyzw_world"],
            length=4,
            field_name="orientation_xyzw_world",
        ),
    )


def _source_local_mesh(object_payload: dict[str, object]) -> TriangleMesh:
    mesh = load_triangle_mesh(
        str(object_payload["mesh_path"]),
        scale=float(object_payload["mesh_scale"]),
    )
    source_pose_assembly = dict(object_payload["source_pose_assembly"])
    source_position = np.asarray(
        source_pose_assembly["position_world_m"],
        dtype=float,
    )
    source_rotation = quat_to_rotmat_xyzw(
        _vec(
            source_pose_assembly["orientation_xyzw_world"],
            length=4,
            field_name="source_pose_assembly.orientation",
        )
    )
    vertices_source = (mesh.vertices_obj - source_position[None, :]) @ source_rotation
    return TriangleMesh(
        vertices_obj=vertices_source,
        faces=mesh.faces,
    )


def _mesh_collision_cfg():
    return schemas_cfg.ConvexDecompositionPropertiesCfg()


def _convert_part_to_usd(
    *,
    name: str,
    object_payload: dict[str, object],
    output_dir: Path,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_usd = output_dir / f"{name}_source_local.usd"
    mesh_source = _source_local_mesh(object_payload)
    temp_stl = write_temporary_triangle_mesh_stl(
        mesh_source,
        prefix=f"{name}_source_local_",
        dir=output_dir,
    )
    converter = MeshConverter(
        MeshConverterCfg(
            asset_path=str(temp_stl),
            usd_dir=str(output_dir),
            usd_file_name=output_usd.name,
            force_usd_conversion=True,
            make_instanceable=False,
            scale=(1.0, 1.0, 1.0),
            mass_props=sim_utils.MassPropertiesCfg(density=float(args_cli.object_density_kg_m3)),
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
    )
    try:
        temp_stl.unlink()
    except FileNotFoundError:
        pass
    return str(Path(converter.usd_path).resolve())


def _convert_subassembly_to_usd(
    *,
    subassembly_payload: dict[str, object],
    output_dir: Path,
) -> str:
    """Convert the assembled prefix into one rigid compound Isaac asset."""

    output_dir.mkdir(parents=True, exist_ok=True)
    part_ids = tuple(str(value) for value in subassembly_payload.get("part_ids", []))
    if not part_ids:
        raise ValueError("Subassembly payload has no part IDs.")
    output_usd = output_dir / ("subassembly_" + "_".join(part_ids) + "_source_local.usd")
    mesh_source = source_local_subassembly_mesh(subassembly_payload)
    temp_stl = write_temporary_triangle_mesh_stl(
        mesh_source,
        prefix="subassembly_source_local_",
        dir=output_dir,
    )
    converter = MeshConverter(
        MeshConverterCfg(
            asset_path=str(temp_stl),
            usd_dir=str(output_dir),
            usd_file_name=output_usd.name,
            force_usd_conversion=True,
            make_instanceable=False,
            scale=(1.0, 1.0, 1.0),
            mass_props=sim_utils.MassPropertiesCfg(density=float(args_cli.object_density_kg_m3)),
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
    )
    try:
        temp_stl.unlink()
    except FileNotFoundError:
        pass
    return str(Path(converter.usd_path).resolve())


def _bind_high_friction_material(
    stage,
    *,
    static_friction: float,
    dynamic_friction: float,
) -> int:
    material_path = "/World/Looks/dual_grasp_high_friction"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(static_friction),
        dynamic_friction=float(dynamic_friction),
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    material_cfg.func(material_path, material_cfg)
    bound = 0
    for root_path in (
        "/World/envs/env_0/HolderRobot",
        "/World/envs/env_0/InserterRobot",
        "/World/envs/env_0/BasePart",
        "/World/envs/env_0/IncomingPart",
    ):
        bind_physics_material(
            root_path,
            material_path,
            stage=stage,
            stronger_than_descendants=True,
        )
        bound += 1
    return bound


def _trajectory(
    *,
    context: FR3MotionContext,
    raw: dict[str, object],
    label: str,
) -> JointTrajectory:
    role = "holder" if "holder" in label else "inserter"
    expected_names = tuple(f"{ROLE_ROBOT_NAMES[role]}_A{index}" for index in range(1, 8))
    names = tuple(str(value) for value in raw["joint_names"])  # type: ignore[index]
    if names != expected_names:
        raise ValueError(f"{label} joint names do not match {expected_names}: {names}")
    waypoints = []
    for raw_waypoint in raw["waypoints"]:  # type: ignore[index]
        moveit_waypoint = _vec(
            raw_waypoint,
            length=7,
            field_name=f"{label}.waypoint",
        )
        isaac_waypoint = kuka_moveit_to_isaac_joint_positions(moveit_waypoint)
        waypoints.append(
            torch.tensor(
                [isaac_waypoint],
                dtype=torch.float32,
                device=context.device,
            )
        )
    if not waypoints:
        raise ValueError(f"{label} has no waypoints.")
    return JointTrajectory(
        waypoints=waypoints,
        dt=context.physics_dt,
    )


def _trajectory_group(
    *,
    context: FR3MotionContext,
    segments: tuple[tuple[str, dict[str, object]], ...],
) -> JointTrajectory:
    """Join consecutive MoveIt segments without stopping at their boundary."""

    waypoints: list[torch.Tensor] = []
    for label, raw in segments:
        segment = _trajectory(context=context, raw=raw, label=label)
        for waypoint in segment.waypoints:
            if waypoints and float(torch.max(torch.abs(waypoint - waypoints[-1])).item()) <= 1.0e-9:
                continue
            waypoints.append(waypoint)
    if not waypoints:
        raise ValueError("A grouped MoveIt trajectory must contain at least one waypoint.")
    return JointTrajectory(waypoints=waypoints, dt=context.physics_dt)


def _execute_segments(
    *,
    context: FR3MotionContext,
    segments: tuple[tuple[str, dict[str, object]], ...],
    max_joint_speed_rad_s: float,
    waypoint_tolerance_rad: float,
    step_callback: Callable[[], None] | None = None,
) -> tuple[torch.Tensor, str, float]:
    trajectory = _trajectory_group(context=context, segments=segments)
    executor = TrajectoryExecutor(
        context,
        waypoint_tolerance_rad=float(waypoint_tolerance_rad),
        max_joint_speed_rad_s=float(max_joint_speed_rad_s),
        final_settle_steps=max(
            1,
            int(float(args_cli.trajectory_final_settle_time_s) / context.physics_dt),
        ),
        step_callback=step_callback,
    )
    started_at = time.perf_counter()
    ok, detail = executor.execute(trajectory)
    duration_s = time.perf_counter() - started_at
    labels = [label for label, _raw in segments]
    if not ok:
        raise RuntimeError(f"{' -> '.join(labels)} execution failed: {detail}")
    return (
        trajectory.waypoints[-1].clone(),
        f"continuous segments {' -> '.join(labels)}: {detail}",
        duration_s,
    )


def _execute_segment(
    *,
    context: FR3MotionContext,
    raw: dict[str, object],
    label: str,
    max_joint_speed_rad_s: float,
    waypoint_tolerance_rad: float,
    step_callback: Callable[[], None] | None = None,
) -> tuple[torch.Tensor, str, float]:
    return _execute_segments(
        context=context,
        segments=((label, raw),),
        max_joint_speed_rad_s=max_joint_speed_rad_s,
        waypoint_tolerance_rad=waypoint_tolerance_rad,
        step_callback=step_callback,
    )


def _close_gripper(
    *,
    context: FR3MotionContext,
    arm_waypoint: torch.Tensor,
    selected_jaw_width_m: float,
    label: str,
    contact_role: str,
    step_callback: Callable[[], None] | None = None,
) -> dict[str, object]:
    requested_close_width_m = float(args_cli.close_width)
    commanded_width_m = (
        requested_close_width_m if requested_close_width_m > 0.0 else min(0.001, float(selected_jaw_width_m))
    )
    settle_steps = max(
        1,
        int(float(args_cli.grasp_settle_time_s) / context.physics_dt),
    )
    for _ in range(settle_steps):
        if step_callback is not None:
            step_callback()
        context.command_arm(arm_waypoint)
        context.command_fixed_gripper()
        context.scene.write_data_to_sim()
        context.sim.step()
        context.scene.update(context.physics_dt)
    diagnostics = _command_gripper_width(
        sim=context.sim,
        scene=context.scene,
        robot=context.robot,
        width=commanded_width_m,
        duration_s=float(args_cli.gripper_close_duration_s),
        max_duration_s=float(args_cli.gripper_close_max_duration_s),
        hold_context=context,
        hold_arm_waypoint=arm_waypoint,
        settle_duration_s=GRIPPER_CLOSE_SETTLE_DURATION_S,
        min_contact_motion_m=0.001,
        force_joint_state=False,
        stop_on_contact=lambda: _filtered_bilateral_contact_matches_selected_object(
            _finger_contact_snapshot(context.scene, role=contact_role),
            minimum_force_n=float(args_cli.finger_contact_min_force_n),
        ),
        contact_preload_m=float(args_cli.gripper_contact_preload_m),
        contact_hold_width_m=float(commanded_width_m),
        step_callback=step_callback,
    )
    status = str(diagnostics.get("gripper_close_status", "unknown"))
    width_matched = _kuka_contact_stall_matches_grasp_width(
        diagnostics,
        float(selected_jaw_width_m),
    )
    joint_names = diagnostics.get("gripper_close_joint_names")
    final_positions = diagnostics.get("gripper_close_final_joint_positions")
    if (
        not width_matched
        and isinstance(joint_names, list)
        and isinstance(final_positions, list)
        and "left_finger_joint" in joint_names
        and len(final_positions) == len(joint_names)
    ):
        left_index = joint_names.index("left_finger_joint")
        final_close = abs(float(final_positions[left_index]))
        selected_close = abs(
            gripper_joint_target_from_width(
                "left_finger_joint",
                float(selected_jaw_width_m),
            )
        )
        width_matched = final_close + 0.003 >= selected_close
        diagnostics["selected_width_geometry_match"] = bool(width_matched)
        diagnostics["selected_width_final_close_m"] = final_close
        diagnostics["selected_width_expected_close_m"] = selected_close
    width_rejection_reason = str(
        diagnostics.get(
            "gripper_close_contact_stall_accept_reason",
            "selected-width geometry did not match",
        )
    )
    filtered_contacts = _finger_contact_snapshot(
        context.scene,
        role=contact_role,
    )
    filtered_contact_matched = _filtered_bilateral_contact_matches_selected_object(
        filtered_contacts,
        minimum_force_n=float(args_cli.finger_contact_min_force_n),
    )
    matched = bool(width_matched) or filtered_contact_matched
    diagnostics["selected_jaw_width_m"] = float(selected_jaw_width_m)
    diagnostics["requested_close_width_m"] = requested_close_width_m
    diagnostics["transport_command_width_m"] = commanded_width_m
    diagnostics["selected_width_geometry_match"] = bool(width_matched)
    diagnostics["selected_object_filtered_finger_contacts"] = filtered_contacts
    diagnostics["selected_object_contact_min_force_n"] = float(args_cli.finger_contact_min_force_n)
    diagnostics["selected_object_bilateral_contact_match"] = filtered_contact_matched
    diagnostics["selected_contact_acceptance"] = (
        "selected_width_geometry"
        if bool(width_matched)
        else "filtered_bilateral_object_contact"
        if filtered_contact_matched
        else "rejected"
    )
    if filtered_contact_matched and not bool(width_matched):
        diagnostics["selected_width_geometry_rejection_reason"] = width_rejection_reason
        diagnostics["gripper_close_contact_stall_accept_reason"] = (
            f"both {contact_role} fingers contact the selected object above "
            f"{float(args_cli.finger_contact_min_force_n):.4f} N"
        )
    diagnostics["selected_width_contact_matched"] = matched
    if status not in {
        "target_reached",
        "contact_stalled",
        "contact_latched",
        "max_duration_elapsed",
    } or not bool(matched):
        raise RuntimeError(
            f"{label} did not establish selected-width contact: "
            f"status={status} matched={matched} diagnostics={diagnostics}"
        )
    transport_hold_width_m = commanded_width_m
    contact_hold_positions = diagnostics.get("gripper_close_contact_hold_joint_positions")
    if (
        isinstance(joint_names, list)
        and isinstance(contact_hold_positions, list)
        and "left_finger_joint" in joint_names
        and len(contact_hold_positions) == len(joint_names)
    ):
        left_index = joint_names.index("left_finger_joint")
        transport_hold_width_m = max(
            0.0,
            min(
                KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
                KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M - 2.0 * abs(float(contact_hold_positions[left_index])),
            ),
        )
    diagnostics["transport_hold_width_m"] = float(transport_hold_width_m)
    context.fixed_gripper_width = float(transport_hold_width_m)
    for _ in range(
        max(
            1,
            int(float(args_cli.postclose_hold_s) / context.physics_dt),
        )
    ):
        if step_callback is not None:
            step_callback()
        context.command_arm(arm_waypoint)
        context.command_fixed_gripper()
        context.scene.write_data_to_sim()
        context.sim.step()
        context.scene.update(context.physics_dt)
    return diagnostics


def _root_pose(asset) -> dict[str, list[float]]:
    pose = asset.data.root_link_pose_w[0]
    return {
        "position_world_m": [float(pose[index].item()) for index in range(3)],
        "orientation_wxyz_world": [float(pose[index].item()) for index in range(3, 7)],
    }


def _hand_joint_positions(context: FR3MotionContext) -> dict[str, float]:
    values = context.get_hand_q()[0]
    return {name: float(values[index].item()) for index, name in enumerate(context.hand_joint_names)}


def _finger_contact_snapshot(scene: InteractiveScene, *, role: str) -> dict[str, object]:
    contacts: dict[str, object] = {}
    for side in ("left", "right"):
        sensor = scene[f"{role}_{side}_finger_contact"]
        force_matrix = sensor.data.force_matrix_w
        if force_matrix is None:
            contacts[side] = {
                "filtered_force_world_n": [0.0, 0.0, 0.0],
                "filtered_force_norm_n": 0.0,
                "available": False,
            }
            continue
        flattened = force_matrix[0].reshape(-1, 3)
        force = torch.sum(flattened, dim=0)
        contacts[side] = {
            "filtered_force_world_n": [float(value) for value in force.tolist()],
            "filtered_force_norm_n": float(torch.linalg.norm(force).item()),
            "available": True,
        }
    return contacts


def _filtered_bilateral_contact_matches_selected_object(
    contacts: dict[str, object],
    *,
    minimum_force_n: float,
) -> bool:
    """Accept only two-sided contacts filtered to the role's intended object."""

    threshold = float(minimum_force_n)
    if threshold < 0.0:
        raise ValueError("minimum_force_n must be non-negative.")
    for side in ("left", "right"):
        raw_contact = contacts.get(side)
        if not isinstance(raw_contact, dict) or not bool(raw_contact.get("available", False)):
            return False
        try:
            force_norm_n = float(raw_contact["filtered_force_norm_n"])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(force_norm_n) or force_norm_n < threshold:
            return False
    return True


def _distance(
    first: list[float] | tuple[float, ...],
    second: list[float] | tuple[float, ...],
) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))


def _quaternion_distance_rad(
    first: list[float] | tuple[float, ...],
    second: list[float] | tuple[float, ...],
) -> float:
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    first_array /= max(float(np.linalg.norm(first_array)), 1.0e-12)
    second_array /= max(float(np.linalg.norm(second_array)), 1.0e-12)
    cosine = float(np.clip(abs(np.dot(first_array, second_array)), 0.0, 1.0))
    return float(2.0 * math.acos(cosine))


def _pose_matrix(
    position: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = quat_to_rotmat_xyzw(orientation_xyzw)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def _finite_symmetry_powers(raw_matrix: object) -> tuple[np.ndarray, ...]:
    """Expand one finite source-frame symmetry without trusting an order tag."""

    symmetry = np.asarray(raw_matrix, dtype=float)
    identity = np.eye(4, dtype=float)
    if symmetry.shape != (4, 4) or not np.all(np.isfinite(symmetry)):
        return (identity,)
    powers: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    current = identity
    for _ in range(16):
        key = tuple(round(float(value), 8) for value in current.reshape(-1))
        if key in seen:
            break
        seen.add(key)
        powers.append(current)
        current = current @ symmetry
    return tuple(powers)


def _expected_incoming_preinsertion_poses(
    plan: dict[str, object],
) -> tuple[dict[str, object], ...]:
    """Return symmetry-valid incoming targets for the selected held-base pose."""

    selected_objects = plan.get("objects")
    if not isinstance(selected_objects, dict):
        raise ValueError("Dual task is missing objects.")
    selected_base = selected_objects.get("subassembly", selected_objects.get("base"))
    if not isinstance(selected_base, dict):
        raise ValueError("Dual task is missing the held base/subassembly.")
    selected_base_position, selected_base_orientation = _pose(dict(selected_base["source_pose_world"]))

    raw_ranked = plan.get("ranked_pair_candidates")
    candidates = [plan]
    if isinstance(raw_ranked, list):
        candidates.extend(value for value in raw_ranked if isinstance(value, dict))

    expected: list[dict[str, object]] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        candidate_objects = candidate.get("objects")
        if not isinstance(candidate_objects, dict):
            continue
        candidate_base = candidate_objects.get(
            "subassembly",
            candidate_objects.get("base"),
        )
        incoming = candidate_objects.get("incoming")
        if not isinstance(candidate_base, dict) or not isinstance(incoming, dict):
            continue
        candidate_base_position, candidate_base_orientation = _pose(dict(candidate_base["source_pose_world"]))
        if _distance(candidate_base_position, selected_base_position) > 1.0e-7:
            continue
        if (
            _quaternion_distance_rad(
                candidate_base_orientation,
                selected_base_orientation,
            )
            > 1.0e-7
        ):
            continue
        position, orientation = _pose(dict(incoming["preinsertion_source_pose_world"]))
        raw_transition = candidate.get("transition_symmetry")
        transition = dict(raw_transition) if isinstance(raw_transition, dict) else {}
        symmetry_powers = _finite_symmetry_powers(transition.get("incoming_symmetry_source_m", np.eye(4)))
        expected_matrix = _pose_matrix(position, orientation)
        for symmetry_power, symmetry_matrix in enumerate(symmetry_powers):
            equivalent_matrix = expected_matrix @ symmetry_matrix
            equivalent_position = tuple(float(value) for value in equivalent_matrix[:3, 3])
            equivalent_orientation = rotmat_to_quat_xyzw(equivalent_matrix[:3, :3])
            key = tuple(round(value, 9) for value in (*equivalent_position, *equivalent_orientation))
            if key in seen:
                continue
            seen.add(key)
            expected.append(
                {
                    "execution_candidate_id": str(
                        candidate.get(
                            "execution_candidate_id",
                            candidate.get("pair_id", ""),
                        )
                    ),
                    "transition_id": str(candidate.get("transition_id", "")),
                    "symmetry_power": symmetry_power,
                    "position_world_m": list(equivalent_position),
                    "orientation_xyzw_world": list(equivalent_orientation),
                }
            )
    if not expected:
        raise ValueError("Dual task has no pre-insertion target for the selected held-base pose.")
    return tuple(expected)


def _motion_snapshot(
    *,
    context: FR3MotionContext,
    target_arm: torch.Tensor,
    target_pose: dict[str, object],
    base_part,
    incoming_part,
) -> dict[str, object]:
    actual_arm = context.get_arm_q()[0]
    target_arm_flat = target_arm[0]
    joint_errors = torch.abs(actual_arm - target_arm_flat)
    tcp_position, tcp_orientation_wxyz = context.get_tcp_pose_w()
    actual_position = np.asarray(tcp_position[0].tolist(), dtype=float)
    actual_orientation = np.asarray(
        tcp_orientation_wxyz[0].tolist(),
        dtype=float,
    )
    target_position = np.asarray(
        _vec(
            target_pose["position_world_m"],
            length=3,
            field_name="target_pose.position_world_m",
        ),
        dtype=float,
    )
    target_orientation_xyzw = _vec(
        target_pose["orientation_xyzw_world"],
        length=4,
        field_name="target_pose.orientation_xyzw_world",
    )
    target_orientation = np.asarray(
        (
            target_orientation_xyzw[3],
            target_orientation_xyzw[0],
            target_orientation_xyzw[1],
            target_orientation_xyzw[2],
        ),
        dtype=float,
    )
    actual_orientation /= max(float(np.linalg.norm(actual_orientation)), 1.0e-12)
    target_orientation /= max(float(np.linalg.norm(target_orientation)), 1.0e-12)
    quaternion_dot = float(np.clip(abs(np.dot(actual_orientation, target_orientation)), 0.0, 1.0))
    return {
        "target_joint_positions_rad": [float(value) for value in target_arm_flat.tolist()],
        "actual_joint_positions_rad": [float(value) for value in actual_arm.tolist()],
        "max_joint_error_rad": float(torch.max(joint_errors).item()),
        "actual_tcp_pose": {
            "position_world_m": [float(value) for value in actual_position],
            "orientation_wxyz_world": [float(value) for value in actual_orientation],
        },
        "target_tcp_pose": {
            "position_world_m": [float(value) for value in target_position],
            "orientation_xyzw_world": [float(value) for value in target_orientation_xyzw],
        },
        "tcp_position_error_m": float(np.linalg.norm(actual_position - target_position)),
        "tcp_orientation_error_rad": float(2.0 * math.acos(quaternion_dot)),
        "base_pose": _root_pose(base_part),
        "incoming_pose": _root_pose(incoming_part),
    }


def _write_attempt(
    *,
    plan_path: Path,
    plan: dict[str, object],
    result: dict[str, object],
) -> Path:
    output = args_cli.attempt_artifact.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "dual_robot_simple_isaac_attempt",
                "plan_json": str(plan_path),
                "assembly": plan.get("assembly"),
                "step_id": plan.get("step_id"),
                "pair_id": plan.get("pair_id"),
                "result": result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main() -> int:
    global ROLE_ROBOT_NAMES

    main_started_at = time.perf_counter()
    plan_path = args_cli.plan_json.expanduser().resolve()
    holder_sequence: list[tuple[Path, dict[str, object]]] = []
    if args_cli.holder_sequence_json is not None:
        sequence_path = args_cli.holder_sequence_json.expanduser().resolve()
        sequence_payload = _read_json(sequence_path)
        for raw_case in sequence_payload.get("cases", []):  # type: ignore[union-attr]
            case = dict(raw_case)
            case_plan_path = Path(str(case["plan"])).expanduser().resolve()
            holder_sequence.append((case_plan_path, _read_json(case_plan_path)))
        if not holder_sequence:
            raise ValueError(f"No cases in holder sequence '{sequence_path}'.")
        plan_path, plan = holder_sequence[0]
    else:
        plan = _read_json(plan_path)
    if plan.get("kind") != "dual_robot_simple_sim_task":
        raise ValueError(f"Unsupported plan kind in '{plan_path}': {plan.get('kind')}")
    raw_roles = plan.get("roles")
    roles = dict(raw_roles) if isinstance(raw_roles, dict) else {}
    role_robot_names: dict[str, str] = {}
    for role in ("holder", "inserter"):
        raw_role = roles.get(role)
        if not isinstance(raw_role, dict):
            raise ValueError(f"Dual plan is missing role metadata for '{role}'.")
        role_robot_names[role] = str(raw_role.get("robot", ""))
    if set(role_robot_names.values()) != {"lbr_one", "lbr_two"}:
        raise ValueError(f"Dual plan roles must assign lbr_one and lbr_two exactly once; got {role_robot_names}.")
    ROLE_ROBOT_NAMES = role_robot_names
    objects = dict(plan["objects"])
    base_payload = dict(objects["base"])
    raw_subassembly = objects.get("subassembly")
    subassembly_payload = dict(raw_subassembly) if isinstance(raw_subassembly, dict) else None
    incoming_payload = dict(objects["incoming"])
    layout = dict(plan.get("layout", {}))
    holder_robot_base_position = _vec(
        layout.get("holder_base_world_m", (0.0, -0.42, 0.0)),
        length=3,
        field_name="layout.holder_base_world_m",
    )
    inserter_robot_base_position = _vec(
        layout.get("inserter_base_world_m", (0.0, 0.42, 0.0)),
        length=3,
        field_name="layout.inserter_base_world_m",
    )
    floor_z_world_m = float(
        layout.get(
            "pickup_floor_z_world_m",
            DEFAULT_FLOOR_Z_WORLD_M,
        )
    )
    trajectories = {str(name): dict(value) for name, value in dict(plan["trajectories"]).items()}
    targets = {str(name): dict(value) for name, value in dict(plan["targets"]).items()}
    output_dir = plan_path.parent / "simple_dual_robot_sim_isaac_assets"
    if subassembly_payload is None:
        base_usd = _convert_part_to_usd(
            name=f"part_{base_payload['part_id']}_base",
            object_payload=base_payload,
            output_dir=output_dir,
        )
        base_pose_payload = dict(base_payload["source_pose_world"])
        assembled_part_ids_before = [str(base_payload["part_id"])]
    else:
        base_usd = _convert_subassembly_to_usd(
            subassembly_payload=subassembly_payload,
            output_dir=output_dir,
        )
        base_pose_payload = dict(subassembly_payload["source_pose_world"])
        assembled_part_ids_before = [
            str(value)
            for value in subassembly_payload["part_ids"]  # type: ignore[index]
        ]
    incoming_usd = _convert_part_to_usd(
        name=f"part_{incoming_payload['part_id']}_incoming",
        object_payload=incoming_payload,
        output_dir=output_dir,
    )
    base_position, base_orientation = _pose(base_pose_payload)
    pickup_position, pickup_orientation = _pose(dict(incoming_payload["pickup_source_pose_world"]))
    expected_preinsertion_poses = _expected_incoming_preinsertion_poses(plan)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
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
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
    )
    sim._app_control_on_stop_handle = None
    sim._disable_app_control_on_stop_handle = True
    sim.set_camera_view([1.65, -1.25, 1.15], [0.48, 0.0, 0.25])
    scene_cfg = make_dual_kuka_assembly_scene_cfg(
        robot_asset_path=str(args_cli.robot_usd.expanduser().resolve()),
        base_part_usd_path=base_usd,
        incoming_part_usd_path=incoming_usd,
        base_part_position=base_position,
        base_part_orientation_xyzw=base_orientation,
        incoming_part_position=pickup_position,
        incoming_part_orientation_xyzw=pickup_orientation,
        ground_height_m=floor_z_world_m,
        holder_robot_base_position=holder_robot_base_position,
        inserter_robot_base_position=inserter_robot_base_position,
        part_density_kg_m3=float(args_cli.object_density_kg_m3),
        kuka_hand_effort_limit_sim=float(args_cli.gripper_effort_limit),
    )
    scene = InteractiveScene(scene_cfg)
    video_camera = None
    if args_cli.record_video is not None:
        sim_utils.create_prim("/World/DualHolderStressVideo", "Xform")
        video_camera = Camera(
            cfg=CameraCfg(
                prim_path="/World/DualHolderStressVideo/CameraSensor",
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
        )
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    bound_count = _bind_high_friction_material(
        omni.usd.get_context().get_stage(),
        static_friction=float(args_cli.static_friction),
        dynamic_friction=float(args_cli.dynamic_friction),
    )
    print(
        f"[DUAL-SIM-ISAAC] Bound high-friction material to {bound_count} scene roots.",
        flush=True,
    )
    sim.reset()
    scene.reset()
    video_recorder = None
    if video_camera is not None:
        video_recorder = IsaacVideoRecorder(
            camera=video_camera,
            sim=sim,
            output_path=args_cli.record_video,
        )
        video_recorder.set_view()
        video_recorder.capture(force=True)

    holder_robot = scene["holder_robot"]
    inserter_robot = scene["inserter_robot"]
    base_part = scene["base_part"]
    incoming_part = scene["incoming_part"]
    plan_grasps = dict(plan["grasps"])
    holder_approach_width_m = kuka_gripper_approach_width(
        float(dict(plan_grasps["holder"])["jaw_width_m"])
    )
    inserter_approach_width_m = kuka_gripper_approach_width(
        float(dict(plan_grasps["inserter_pickup"])["jaw_width_m"])
    )
    holder_context = FR3MotionContext(
        robot=holder_robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=holder_approach_width_m,
        critical_damping_ratio=float(args_cli.critical_damping_ratio),
    )
    inserter_context = FR3MotionContext(
        robot=inserter_robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=inserter_approach_width_m,
        critical_damping_ratio=float(args_cli.critical_damping_ratio),
    )
    holder_damping = holder_context.refresh_critical_joint_damping(required=True)
    inserter_damping = inserter_context.refresh_critical_joint_damping(required=True)
    print(
        "[DUAL-SIM-ISAAC] Applied configuration-adaptive critical damping "
        f"zeta={float(args_cli.critical_damping_ratio):.3f} to both robots.",
        flush=True,
    )
    pickup_root_state = torch.zeros(
        (1, 13),
        dtype=torch.float32,
        device=incoming_part.device,
    )
    pickup_root_state[0, :3] = torch.tensor(
        pickup_position,
        dtype=torch.float32,
        device=incoming_part.device,
    )
    pickup_root_state[0, 3:7] = torch.tensor(
        (
            pickup_orientation[3],
            pickup_orientation[0],
            pickup_orientation[1],
            pickup_orientation[2],
        ),
        dtype=torch.float32,
        device=incoming_part.device,
    )

    def _pin_incoming_at_pickup() -> None:
        incoming_part.write_root_state_to_sim(pickup_root_state)

    def _capture_step_callback() -> None:
        if video_recorder is not None:
            video_recorder.capture()

    def _pickup_step_callback() -> None:
        _pin_incoming_at_pickup()
        _capture_step_callback()

    initial_holder = _trajectory(
        context=holder_context,
        raw=trajectories["holder_pregrasp"],
        label="holder_pregrasp",
    ).waypoints[0]
    if args_cli.holder_only:
        initial_inserter = torch.tensor(
            kuka_moveit_to_isaac_joint_positions(KUKA_MOVEIT_ARM_START_JOINT_VALUES),
            dtype=torch.float32,
            device=inserter_robot.device,
        )
    else:
        initial_inserter = _trajectory(
            context=inserter_context,
            raw=trajectories["inserter_pickup_pregrasp"],
            label="inserter_pickup_pregrasp",
        ).waypoints[0]
    holder_context.reset_joint_state(initial_holder, steps=5)
    inserter_context.reset_joint_state(initial_inserter, steps=5)
    for _ in range(20):
        _pin_incoming_at_pickup()
        holder_context.command_arm(initial_holder)
        holder_context.command_fixed_gripper()
        inserter_context.command_arm(initial_inserter)
        inserter_context.command_fixed_gripper()
        scene.write_data_to_sim()
        sim.step()
        scene.update(float(sim.get_physics_dt()))
        if video_recorder is not None:
            video_recorder.capture()

    initial_base_pose = _root_pose(base_part)
    initial_incoming_pose = _root_pose(incoming_part)
    execution_started_at = time.perf_counter()
    result: dict[str, object] = {
        "success": False,
        "status": "running",
        "initial_base_pose": initial_base_pose,
        "initial_incoming_pose": initial_incoming_pose,
        "assembled_part_ids_before": assembled_part_ids_before,
        "setup_duration_s": execution_started_at - main_started_at,
        "motion_speed_limits_rad_s": {
            "unloaded": float(
                args_cli.max_joint_speed_rad_s
                if args_cli.max_joint_speed_rad_s is not None
                else args_cli.unloaded_max_joint_speed_rad_s
            ),
            "loaded": float(
                args_cli.max_joint_speed_rad_s
                if args_cli.max_joint_speed_rad_s is not None
                else args_cli.loaded_max_joint_speed_rad_s
            ),
        },
        "contact_physics": {
            "static_friction": float(args_cli.static_friction),
            "dynamic_friction": float(args_cli.dynamic_friction),
            "gripper_effort_limit": float(args_cli.gripper_effort_limit),
            "finger_contact_min_force_n": float(args_cli.finger_contact_min_force_n),
            "gripper_contact_preload_m": float(args_cli.gripper_contact_preload_m),
            "gripper_close_duration_s": float(args_cli.gripper_close_duration_s),
        },
        "joint_drive_damping": {
            "holder": holder_damping,
            "inserter": inserter_damping,
        },
        "tracking_tolerances_rad": {
            "transit": float(args_cli.trajectory_waypoint_tolerance_rad),
            "contact_pose": float(args_cli.contact_pose_tolerance_rad),
        },
        "trajectory_playback": {
            "mode": "continuous_moveit_polyline_with_velocity_feedforward",
            "settle_boundaries": [
                "holder_grasp",
                "inserter_pickup_grasp",
                "inserter_preinsertion",
            ],
        },
        "steps": [],
    }
    unloaded_speed = float(
        args_cli.max_joint_speed_rad_s
        if args_cli.max_joint_speed_rad_s is not None
        else args_cli.unloaded_max_joint_speed_rad_s
    )
    loaded_speed = float(
        args_cli.max_joint_speed_rad_s
        if args_cli.max_joint_speed_rad_s is not None
        else args_cli.loaded_max_joint_speed_rad_s
    )

    def _tolerance_for_segment(label: str) -> float:
        if label in {"holder_grasp", "inserter_pickup_grasp"}:
            return float(args_cli.contact_pose_tolerance_rad)
        return float(args_cli.trajectory_waypoint_tolerance_rad)

    def _step_record(
        *,
        label: str,
        context: FR3MotionContext,
        target_arm: torch.Tensor,
        detail: str,
        duration_s: float,
        segment_labels: tuple[str, ...] = (),
    ) -> dict[str, object]:
        record = {
            "label": label,
            "ok": True,
            "detail": detail,
            "duration_s": float(duration_s),
            "maximum_joint_speed_rad_s": (
                unloaded_speed
                if label
                in {
                    "holder_pregrasp",
                    "holder_grasp",
                    "inserter_pickup_pregrasp",
                    "inserter_pickup_grasp",
                }
                else loaded_speed
            ),
            "joint_tracking_tolerance_rad": _tolerance_for_segment(label),
            "tracking": _motion_snapshot(
                context=context,
                target_arm=target_arm,
                target_pose=targets[label],
                base_part=base_part,
                incoming_part=incoming_part,
            ),
        }
        if segment_labels:
            record["continuous_segment_labels"] = list(segment_labels)
        return record

    if holder_sequence:
        sequence_results: list[dict[str, object]] = []
        for case_index, (case_plan_path, case_plan) in enumerate(
            holder_sequence,
            start=1,
        ):
            case_objects = dict(case_plan["objects"])
            case_base = dict(case_objects["base"])
            case_subassembly = case_objects.get("subassembly")
            if isinstance(case_subassembly, dict):
                case_pose_payload = dict(case_subassembly["source_pose_world"])
            else:
                case_pose_payload = dict(case_base["source_pose_world"])
            case_position, case_orientation = _pose(case_pose_payload)
            base_state = base_part.data.root_state_w.clone()
            base_state[0, :3] = torch.tensor(
                case_position,
                dtype=torch.float32,
                device=base_part.device,
            )
            base_state[0, 3:7] = torch.tensor(
                (
                    case_orientation[3],
                    case_orientation[0],
                    case_orientation[1],
                    case_orientation[2],
                ),
                dtype=torch.float32,
                device=base_part.device,
            )
            base_state[0, 7:] = 0.0

            # Release the previous object before relocating the next test pose.
            release_steps = max(1, int(0.5 / holder_context.physics_dt))
            for _ in range(release_steps):
                holder_context.command_fixed_gripper()
                holder_context.scene.write_data_to_sim()
                holder_context.sim.step()
                holder_context.scene.update(holder_context.physics_dt)
                if video_recorder is not None:
                    video_recorder.capture()
            base_part.write_root_state_to_sim(base_state)
            for _ in range(10):
                base_part.write_root_state_to_sim(base_state)
                holder_context.command_fixed_gripper()
                holder_context.scene.write_data_to_sim()
                holder_context.sim.step()
                holder_context.scene.update(holder_context.physics_dt)
                if video_recorder is not None:
                    video_recorder.capture()

            trajectories = {str(name): dict(value) for name, value in dict(case_plan["trajectories"]).items()}
            targets = {str(name): dict(value) for name, value in dict(case_plan["targets"]).items()}
            case_grasps = dict(case_plan["grasps"])
            holder_context.fixed_gripper_width = kuka_gripper_approach_width(
                float(dict(case_grasps["holder"])["jaw_width_m"])
            )
            inserter_context.fixed_gripper_width = kuka_gripper_approach_width(
                float(dict(case_grasps["inserter_pickup"])["jaw_width_m"])
            )
            case_initial_base_pose = _root_pose(base_part)
            case_result: dict[str, object] = {
                "index": case_index,
                "plan": str(case_plan_path),
                "pair_id": case_plan.get("pair_id"),
                "holder_grasp_id": dict(case_plan["grasps"])["holder"]["grasp_id"],  # type: ignore[index]
                "base_position_world_m": list(case_position),
                "success": False,
                "steps": [],
                "video_start_frame": (0 if video_recorder is None else video_recorder.frame_count),
            }
            try:
                holder_labels = ("holder_pregrasp", "holder_grasp")
                holder_last, detail, duration_s = _execute_segments(
                    context=holder_context,
                    segments=tuple((f"case_{case_index}_{label}", trajectories[label]) for label in holder_labels),
                    max_joint_speed_rad_s=unloaded_speed,
                    waypoint_tolerance_rad=_tolerance_for_segment("holder_grasp"),
                    step_callback=(None if video_recorder is None else video_recorder.capture),
                )
                case_result["steps"].append(  # type: ignore[union-attr]
                    _step_record(
                        label="holder_grasp",
                        context=holder_context,
                        target_arm=holder_last,
                        detail=detail,
                        duration_s=duration_s,
                        segment_labels=holder_labels,
                    )
                )
                case_result["holder_close"] = _close_gripper(
                    context=holder_context,
                    arm_waypoint=holder_last,
                    selected_jaw_width_m=float(
                        dict(case_plan["grasps"])["holder"]["jaw_width_m"]  # type: ignore[index]
                    ),
                    label=f"holder_case_{case_index}",
                    contact_role="holder",
                    step_callback=(None if video_recorder is None else video_recorder.capture),
                )
                hold_steps = max(
                    1,
                    int(float(args_cli.final_hold_s) / holder_context.physics_dt),
                )
                for _ in range(hold_steps):
                    holder_context.command_arm(holder_last)
                    holder_context.scene.write_data_to_sim()
                    holder_context.sim.step()
                    holder_context.scene.update(holder_context.physics_dt)
                    if video_recorder is not None:
                        video_recorder.capture()
                case_result["success"] = True
                case_result["message"] = "holder cycle completed"
            except Exception as exc:
                case_result["message"] = str(exc)
                case_result["traceback"] = traceback.format_exc()
            case_result["final_base_pose"] = _root_pose(base_part)
            case_result["base_displacement_m"] = _distance(
                case_initial_base_pose["position_world_m"],
                case_result["final_base_pose"]["position_world_m"],  # type: ignore[index]
            )
            case_result["video_end_frame"] = 0 if video_recorder is None else video_recorder.frame_count
            sequence_results.append(case_result)

        result.update(
            {
                "success": all(bool(case["success"]) for case in sequence_results),
                "status": ("ok" if all(bool(case["success"]) for case in sequence_results) else "partial_failure"),
                "message": (
                    f"Executed {len(sequence_results)} holder cases continuously; "
                    f"{sum(bool(case['success']) for case in sequence_results)} succeeded."
                ),
                "holder_sequence": sequence_results,
                "execution_duration_s": time.perf_counter() - execution_started_at,
                "main_duration_s": time.perf_counter() - main_started_at,
            }
        )
        if video_recorder is not None:
            video_recorder.capture(force=True)
            result["video_path"] = video_recorder.output_path
            result["video_frame_count"] = video_recorder.frame_count
            video_recorder.close()
        output = _write_attempt(
            plan_path=plan_path,
            plan=plan,
            result=result,
        )
        print(
            f"[DUAL-SIM-ISAAC] continuous holder sequence: {result['message']} artifact={output}",
            flush=True,
        )
        return 0

    try:
        holder_labels = ("holder_pregrasp", "holder_grasp")
        holder_last, detail, duration_s = _execute_segments(
            context=holder_context,
            segments=tuple((label, trajectories[label]) for label in holder_labels),
            max_joint_speed_rad_s=unloaded_speed,
            waypoint_tolerance_rad=_tolerance_for_segment("holder_grasp"),
            step_callback=_pickup_step_callback,
        )
        result["steps"].append(
            _step_record(
                label="holder_grasp",
                context=holder_context,
                target_arm=holder_last,
                detail=detail,
                duration_s=duration_s,
                segment_labels=holder_labels,
            )
        )
        holder_close = _close_gripper(
            context=holder_context,
            arm_waypoint=holder_last,
            selected_jaw_width_m=float(
                dict(plan["grasps"])["holder"]["jaw_width_m"]  # type: ignore[index]
            ),
            label="holder",
            contact_role="holder",
            step_callback=_pickup_step_callback,
        )
        result["holder_close"] = holder_close
        result["after_holder_close"] = {
            "tracking": _motion_snapshot(
                context=holder_context,
                target_arm=holder_last,
                target_pose=targets["holder_grasp"],
                base_part=base_part,
                incoming_part=incoming_part,
            ),
            "hand_joint_positions": _hand_joint_positions(holder_context),
            "finger_contacts": _finger_contact_snapshot(
                scene,
                role="holder",
            ),
        }

        if args_cli.holder_only:
            final_steps = max(
                1,
                int(float(args_cli.final_hold_s) / float(sim.get_physics_dt())),
            )
            for _ in range(final_steps):
                holder_context.command_arm(holder_last)
                inserter_context.command_arm(initial_inserter)
                inserter_context.command_fixed_gripper()
                scene.write_data_to_sim()
                sim.step()
                scene.update(float(sim.get_physics_dt()))
            result.update(
                {
                    "success": True,
                    "status": "ok",
                    "message": "Holder-only simulation completed.",
                    "final_base_pose": _root_pose(base_part),
                    "final_incoming_pose": _root_pose(incoming_part),
                    "execution_duration_s": time.perf_counter() - execution_started_at,
                }
            )
            if video_recorder is not None:
                video_recorder.capture(force=True)
                result["video_path"] = video_recorder.output_path
                result["video_frame_count"] = video_recorder.frame_count
                video_recorder.close()
            output = _write_attempt(
                plan_path=plan_path,
                plan=plan,
                result=result,
            )
            print(
                f"[DUAL-SIM-ISAAC] Holder-only success. Attempt: {output}",
                flush=True,
            )
            return 0

        pickup_labels = (
            "inserter_pickup_pregrasp",
            "inserter_pickup_grasp",
        )
        inserter_last, detail, duration_s = _execute_segments(
            context=inserter_context,
            segments=tuple((label, trajectories[label]) for label in pickup_labels),
            max_joint_speed_rad_s=unloaded_speed,
            waypoint_tolerance_rad=_tolerance_for_segment("inserter_pickup_grasp"),
            step_callback=_pickup_step_callback,
        )
        result["steps"].append(
            _step_record(
                label="inserter_pickup_grasp",
                context=inserter_context,
                target_arm=inserter_last,
                detail=detail,
                duration_s=duration_s,
                segment_labels=pickup_labels,
            )
        )
        inserter_close = _close_gripper(
            context=inserter_context,
            arm_waypoint=inserter_last,
            selected_jaw_width_m=float(
                dict(plan["grasps"])["inserter_pickup"][  # type: ignore[index]
                    "jaw_width_m"
                ]
            ),
            label="inserter",
            contact_role="inserter",
            step_callback=_pickup_step_callback,
        )
        result["inserter_close"] = inserter_close
        result["after_inserter_close"] = {
            "tracking": _motion_snapshot(
                context=inserter_context,
                target_arm=inserter_last,
                target_pose=targets["inserter_pickup_grasp"],
                base_part=base_part,
                incoming_part=incoming_part,
            ),
            "incoming_pose": _root_pose(incoming_part),
            "hand_joint_positions": _hand_joint_positions(inserter_context),
            "finger_contacts": _finger_contact_snapshot(
                scene,
                role="inserter",
            ),
        }

        transport_labels = (
            "inserter_pickup_lift",
            "inserter_above_preinsertion",
            "inserter_preinsertion",
        )
        inserter_last, detail, duration_s = _execute_segments(
            context=inserter_context,
            segments=tuple((label, trajectories[label]) for label in transport_labels),
            max_joint_speed_rad_s=loaded_speed,
            waypoint_tolerance_rad=_tolerance_for_segment("inserter_preinsertion"),
            step_callback=_capture_step_callback,
        )
        result["steps"].append(
            _step_record(
                label="inserter_preinsertion",
                context=inserter_context,
                target_arm=inserter_last,
                detail=detail,
                duration_s=duration_s,
                segment_labels=transport_labels,
            )
        )
        result["after_inserter_preinsertion"] = {
            "incoming_pose": _root_pose(incoming_part),
            "hand_joint_positions": _hand_joint_positions(inserter_context),
        }

        final_steps = max(
            1,
            int(float(args_cli.final_hold_s) / float(sim.get_physics_dt())),
        )
        for _ in range(final_steps):
            holder_context.command_arm(holder_last)
            holder_context.command_fixed_gripper()
            inserter_context.command_arm(inserter_last)
            inserter_context.command_fixed_gripper()
            scene.write_data_to_sim()
            sim.step()
            scene.update(float(sim.get_physics_dt()))
            if video_recorder is not None:
                video_recorder.capture()

        final_base_pose = _root_pose(base_part)
        final_incoming_pose = _root_pose(incoming_part)
        base_displacement = _distance(
            initial_base_pose["position_world_m"],
            final_base_pose["position_world_m"],
        )
        incoming_transport_distance = _distance(
            initial_incoming_pose["position_world_m"],
            final_incoming_pose["position_world_m"],
        )
        base_orientation_error = _quaternion_distance_rad(
            final_base_pose["orientation_wxyz_world"],
            (
                base_orientation[3],
                base_orientation[0],
                base_orientation[1],
                base_orientation[2],
            ),
        )
        incoming_pose_errors: list[dict[str, object]] = []
        for expected_pose in expected_preinsertion_poses:
            expected_orientation_xyzw = tuple(
                float(value)
                for value in expected_pose["orientation_xyzw_world"]  # type: ignore[union-attr]
            )
            position_error = _distance(
                final_incoming_pose["position_world_m"],
                expected_pose["position_world_m"],  # type: ignore[arg-type]
            )
            orientation_error = _quaternion_distance_rad(
                final_incoming_pose["orientation_wxyz_world"],
                (
                    expected_orientation_xyzw[3],
                    expected_orientation_xyzw[0],
                    expected_orientation_xyzw[1],
                    expected_orientation_xyzw[2],
                ),
            )
            incoming_pose_errors.append(
                {
                    **expected_pose,
                    "position_error_m": position_error,
                    "orientation_error_rad": orientation_error,
                }
            )
        incoming_position_tolerance = float(args_cli.incoming_position_tolerance_m)
        incoming_orientation_tolerance = float(args_cli.incoming_orientation_tolerance_rad)
        best_incoming_pose = min(
            incoming_pose_errors,
            key=lambda value: max(
                float(value["position_error_m"]) / incoming_position_tolerance,
                float(value["orientation_error_rad"]) / incoming_orientation_tolerance,
            ),
        )
        incoming_preinsertion_error = float(best_incoming_pose["position_error_m"])
        incoming_preinsertion_orientation_error = float(best_incoming_pose["orientation_error_rad"])
        result.update(
            {
                "final_base_pose": final_base_pose,
                "final_incoming_pose": final_incoming_pose,
                "expected_base_orientation_xyzw_world": list(base_orientation),
                "expected_incoming_preinsertion_position_world_m": list(
                    expected_preinsertion_poses[0]["position_world_m"]  # type: ignore[arg-type]
                ),
                "expected_incoming_preinsertion_orientation_xyzw_world": list(
                    expected_preinsertion_poses[0]["orientation_xyzw_world"]  # type: ignore[arg-type]
                ),
                "expected_incoming_preinsertion_poses": list(expected_preinsertion_poses),
                "matched_incoming_preinsertion_pose": best_incoming_pose,
                "base_displacement_m": base_displacement,
                "base_orientation_error_rad": base_orientation_error,
                "incoming_transport_distance_m": (incoming_transport_distance),
                "incoming_preinsertion_position_error_m": (incoming_preinsertion_error),
                "incoming_preinsertion_orientation_error_rad": (incoming_preinsertion_orientation_error),
            }
        )
        if base_displacement > float(args_cli.base_position_tolerance_m):
            raise RuntimeError(
                f"Base moved {base_displacement:.4f} m; allowed {float(args_cli.base_position_tolerance_m):.4f} m."
            )
        if base_orientation_error > float(args_cli.base_orientation_tolerance_rad):
            raise RuntimeError(
                "Base rotated away from its held pose; "
                f"error={base_orientation_error:.4f} rad, allowed="
                f"{float(args_cli.base_orientation_tolerance_rad):.4f} rad."
            )
        if incoming_transport_distance < 0.15:
            raise RuntimeError(
                f"Incoming part did not travel with the inserter; distance={incoming_transport_distance:.4f} m."
            )
        if incoming_preinsertion_error > float(args_cli.incoming_position_tolerance_m):
            raise RuntimeError(
                "Incoming part did not reach the pre-insertion pose; "
                f"error={incoming_preinsertion_error:.4f} m, allowed="
                f"{float(args_cli.incoming_position_tolerance_m):.4f} m."
            )
        if incoming_preinsertion_orientation_error > incoming_orientation_tolerance:
            raise RuntimeError(
                "Incoming part did not reach a symmetry-equivalent pre-insertion orientation; "
                f"error={incoming_preinsertion_orientation_error:.4f} rad, allowed="
                f"{incoming_orientation_tolerance:.4f} rad."
            )
        result.update(
            {
                "success": True,
                "status": "ok",
                "message": (
                    "Holder maintained the assembled prefix while the inserter "
                    f"picked part {plan.get('incoming_part_id')} and transported "
                    "it to pre-insertion."
                ),
            }
        )
    except Exception as exc:
        result.update(
            {
                "success": False,
                "status": "failed",
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "final_base_pose": _root_pose(base_part),
                "final_incoming_pose": _root_pose(incoming_part),
            }
        )

    result["execution_duration_s"] = time.perf_counter() - execution_started_at
    result["main_duration_s"] = time.perf_counter() - main_started_at
    if video_recorder is not None:
        video_recorder.capture(force=True)
        result["video_path"] = video_recorder.output_path
        result["video_frame_count"] = video_recorder.frame_count
        video_recorder.close()
    output = _write_attempt(
        plan_path=plan_path,
        plan=plan,
        result=result,
    )
    print(
        f"[DUAL-SIM-ISAAC] success={result['success']} status={result['status']} artifact={output}",
        flush=True,
    )
    if not bool(result["success"]):
        print(
            f"[DUAL-SIM-ISAAC] failure: {result.get('message')}",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
