"""Shared Fabrica-style planning pipeline for sim, pitl, and real flows."""

from __future__ import annotations

import glob
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from grasp_planning.grasping import AntipodalGraspGeneratorConfig, AntipodalMeshGraspGenerator
from grasp_planning.grasping.collision import (
    GRIPPER_COLLISION_MODEL_FRANKA,
    GRIPPER_COLLISION_MODEL_KUKA_Y,
    normalize_gripper_collision_model_name,
)
from grasp_planning.grasping.fabrica_grasp_debug import (
    DEFAULT_CONTACT_APPROACH_OFFSETS_M,
    DEFAULT_CONTACT_LATERAL_OFFSETS_M,
    GRASP_SCORING_ALGORITHM_VERSION,
    CandidateStatus,
    PickupPlacementSpec,
    SavedGraspBundle,
    SavedGraspCandidate,
    accepted_grasps,
    build_pickup_pose_world,
    canonicalize_target_mesh,
    evaluate_saved_grasps_against_pickup_pose,
    filter_grasps_against_assembly,
    load_assembly_obstacle_mesh,
    load_asset_mesh,
    quat_to_rotmat_xyzw,
    relative_asset_mesh_path,
    resolve_asset_mesh_path,
    rotmat_to_quat_xyzw,
    save_grasp_bundle,
    score_grasps,
    serialize_saved_candidate,
    write_debug_html,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import SurfaceSample
from grasp_planning.grasping.mesh_io import DEFAULT_ASSET_MESH_DIR
from grasp_planning.grasping.world_constraints import ObjectWorldPose

DEFAULT_REACHABILITY_PROXY_SCORE_WEIGHT = 0.0
DEFAULT_REACHABILITY_PROXY_HAND_OFFSET_M = 0.10
DEFAULT_REACHABILITY_PROXY_COMFORT_RADIUS_M = 0.50
DEFAULT_REACHABILITY_PROXY_COMFORT_BAND_M = 0.18
DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_ZERO_MIN_M = 0.20
DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_FULL_MIN_M = 0.35
DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_FULL_MAX_M = 0.65
DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_ZERO_MAX_M = 0.85
DEFAULT_REACHABILITY_PROXY_OBJECT_CLEARANCE_M = 0.04
DEFAULT_REACHABILITY_PROXY_FLOOR_CLEARANCE_M = 0.05


@dataclass(frozen=True)
class GeometryConfig:
    target_mesh_path: str
    mesh_scale: float = 1.0
    assembly_glob: str | None = None
    assembly_obstacle_paths: tuple[str, ...] | None = None
    assembly_obstacle_sweep_vector_m: tuple[float, float, float] | None = None
    assembly_obstacle_metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class PlanningConfig:
    stage1_cache_enabled: bool = True
    stage1_cache_dir: str = "artifacts/stage1_cache"
    num_surface_samples: int = 1024
    min_jaw_width: float = 0.002
    max_jaw_width: float = 0.09
    antipodal_cosine_threshold: float = 0.984807753012208
    roll_angles_rad: tuple[float, ...] = (0.0,)
    max_pair_checks: int = 40960
    detailed_finger_contact_gap_m: float = 0.002
    gripper_collision_model: str = GRIPPER_COLLISION_MODEL_FRANKA
    floor_clearance_margin_m: float = 0.0
    skip_stage1_collision_checks: bool = False
    stage1_pose_upright_axis_enabled: bool = True
    top_grasp_score_weight: float = 0.35
    regrasp_transfer_top_grasp_score_weight: float = 0.85
    reachability_proxy_score_weight: float = DEFAULT_REACHABILITY_PROXY_SCORE_WEIGHT
    reachability_proxy_hand_offset_m: float = DEFAULT_REACHABILITY_PROXY_HAND_OFFSET_M
    symmetry_pickup_enabled: bool = False
    symmetry_asset_path: str = ""
    symmetry_max_transforms: int = 0
    symmetry_next_orientation_limit: int = 24
    contact_lateral_offsets_m: tuple[float, ...] = DEFAULT_CONTACT_LATERAL_OFFSETS_M
    contact_approach_offsets_m: tuple[float, ...] = DEFAULT_CONTACT_APPROACH_OFFSETS_M
    rng_seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gripper_collision_model",
            normalize_gripper_collision_model_name(self.gripper_collision_model),
        )

    def to_generator_config(
        self,
        *,
        upright_approach_axes_obj: tuple[tuple[float, float, float], ...] = (),
    ) -> AntipodalGraspGeneratorConfig:
        return AntipodalGraspGeneratorConfig(
            num_surface_samples=self.num_surface_samples,
            min_jaw_width=self.min_jaw_width,
            max_jaw_width=self.max_jaw_width,
            antipodal_cosine_threshold=self.antipodal_cosine_threshold,
            roll_angles_rad=self.roll_angles_rad,
            upright_approach_axes_obj=upright_approach_axes_obj,
            max_pair_checks=self.max_pair_checks,
            detailed_finger_contact_gap_m=self.detailed_finger_contact_gap_m,
            gripper_collision_model=self.gripper_collision_model,
            rng_seed=self.rng_seed,
        )


def _robot_metadata_for_planning(planning: PlanningConfig) -> dict[str, object]:
    if planning.gripper_collision_model == GRIPPER_COLLISION_MODEL_KUKA_Y:
        return {
            "robot_model": "kuka_iiwa7",
            "gripper_model": GRIPPER_COLLISION_MODEL_KUKA_Y,
            "tcp_link": "gripper_tcp",
            "tcp_offset_m": [0.0, 0.0, 0.1455],
        }
    return {
        "robot_model": "franka_fr3",
        "gripper_model": planning.gripper_collision_model,
        "tcp_link": "fr3_hand_tcp",
    }


@dataclass(frozen=True)
class PickupPoseConfig:
    support_face: str
    yaw_deg: float
    xy_world: tuple[float, float]

    def to_spec(self) -> PickupPlacementSpec:
        return PickupPlacementSpec(
            support_face=self.support_face,
            yaw_deg=self.yaw_deg,
            xy_world=self.xy_world,
        )


@dataclass(frozen=True)
class ExecutionWorldPoseConfig:
    position_world: tuple[float, float, float]
    orientation_xyzw_world: tuple[float, float, float, float]

    def to_object_pose_world(self) -> ObjectWorldPose:
        return ObjectWorldPose(
            position_world=self.position_world,
            orientation_xyzw_world=self.orientation_xyzw_world,
        )


@dataclass(frozen=True)
class MujocoPipelineConfig:
    enabled: bool = False
    python_executable: str = ""
    robot_config: str = ""
    simulation_config: str = ""
    controller: str = "native"
    grasp_id: str = ""
    pregrasp_offset: float | None = None
    gripper_width_clearance: float | None = None
    contact_gap_m: float | None = None
    object_mass_kg: float | None = None
    object_density_kg_m3: float | None = None
    object_scale: float | None = None
    lift_height_m: float | None = None
    success_height_margin_m: float | None = None
    attempt_artifact: str = "artifacts/mujoco_pick_attempt.json"
    viewer: bool = True
    viewer_left_ui: bool = False
    viewer_right_ui: bool = False
    viewer_no_realtime: bool = False
    viewer_hold_seconds: float = 8.0
    viewer_block_at_end: bool = False
    keep_generated_scene: bool = False
    moveit_frame_id: str = "base"
    moveit_planning_group: str = "fr3_arm"
    moveit_pose_link: str = "fr3_hand_tcp"
    moveit_namespace: str = ""
    moveit_pipeline_id: str = ""
    moveit_planner_id: str = ""
    moveit_wait_for_moveit_timeout_s: float = 15.0
    moveit_ik_timeout_s: float = 2.0
    moveit_planning_time_s: float = 5.0
    moveit_num_planning_attempts: int = 5
    moveit_ik_candidate_count: int = 1
    moveit_ik_beam_width: int = 1
    moveit_ik_seed_perturbation_rad: float = 0.35
    moveit_ik_dedup_tolerance_rad: float = 0.05
    moveit_ik_joint_weights: tuple[float, ...] = ()
    moveit_velocity_scale: float = 0.05
    moveit_acceleration_scale: float = 0.05
    moveit_execute_timeout_s: float = 120.0
    moveit_allow_collisions: bool = False
    regrasp_fallback_enabled: bool = True
    force_regrasp_fallback: bool = False
    regrasp_plan_artifact: str = "artifacts/mujoco_regrasp_plan.json"
    regrasp_html_artifact: str = ""
    regrasp_staging_xy_world: tuple[float, float] | None = None
    regrasp_staging_xy_offsets_m: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.06, 0.0),
        (-0.06, 0.0),
        (0.0, 0.06),
        (0.0, -0.06),
        (0.12, 0.0),
        (-0.12, 0.0),
        (0.0, 0.12),
        (0.0, -0.12),
        (0.06, 0.06),
        (0.06, -0.06),
        (-0.06, 0.06),
        (-0.06, -0.06),
    )
    regrasp_max_placement_options: int = 18
    regrasp_moveit_max_candidate_plans: int = 36
    regrasp_moveit_transfer_candidates_per_placement: int = 3
    regrasp_moveit_final_candidates_per_placement: int = 3
    regrasp_yaw_angles_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    regrasp_max_orientations: int = 24
    regrasp_min_facet_area_m2: float = 0.0
    regrasp_stability_margin_m: float = 0.0
    regrasp_coplanar_tolerance_m: float = 1.0e-6


@dataclass(frozen=True)
class IsaacPipelineConfig:
    enabled: bool = False
    python_executable: str = ""
    part_usd: str = ""
    fr3_usd: str = ""
    controller: str = "moveit"
    grasp_id: str = ""
    grasp_rank: int = 1
    pregrasp_offset: float | None = None
    gripper_width_clearance: float | None = None
    contact_gap_m: float | None = None
    lift_height_m: float = 0.08
    success_height_margin_m: float = 0.05
    close_width: float = 0.0
    object_mass_kg: float | None = None
    object_density_kg_m3: float | None = None
    tcp_to_grasp_offset: tuple[float, float, float] | None = None
    attempt_artifact: str = "artifacts/isaac_pick_attempt.json"
    pregrasp_only: bool = False
    run_seconds: float = 0.0
    headless: bool = False
    moveit_frame_id: str = "base"
    moveit_target_position_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    moveit_planning_group: str = "fr3_arm"
    moveit_pose_link: str = "fr3_hand_tcp"
    moveit_namespace: str = ""
    moveit_joint_names: tuple[str, ...] = ()
    moveit_start_joint_positions: tuple[float, ...] = ()
    moveit_pipeline_id: str = ""
    moveit_planner_id: str = ""
    moveit_wait_for_moveit_timeout_s: float = 15.0
    moveit_ik_timeout_s: float = 2.0
    moveit_planning_time_s: float = 5.0
    moveit_num_planning_attempts: int = 5
    moveit_ik_candidate_count: int = 1
    moveit_ik_beam_width: int = 1
    moveit_ik_seed_perturbation_rad: float = 0.35
    moveit_ik_dedup_tolerance_rad: float = 0.05
    moveit_ik_joint_weights: tuple[float, ...] = ()
    moveit_velocity_scale: float = 0.05
    moveit_acceleration_scale: float = 0.05
    moveit_execution_speed_rad_s: float = 0.35
    moveit_grasp_settle_time_s: float = 0.0
    gripper_close_duration_s: float = 1.5
    gripper_close_max_duration_s: float = 10.0
    postclose_hold_s: float = 1.0
    moveit_allow_collisions: bool = False
    record_video: str = ""
    video_fps: float = 30.0
    video_width: int = 960
    video_height: int = 540
    video_camera_eye: tuple[float, float, float] = (1.6, -1.2, 1.0)
    video_camera_target: tuple[float, float, float] = (0.35, 0.0, 0.3)


@dataclass(frozen=True)
class Ros2Config:
    pose_base_topic: str = ""
    frame_id: str = "world"
    timeout_s: float = 10.0
    assembly_name: str = ""
    part_id: int | None = None
    position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RealExecutionConfig:
    enabled: bool = False
    grasp_id: str = ""
    attempt_artifact: str = "artifacts/real_robot_pick_attempt.json"
    planning_group: str = "fr3_arm"
    pose_link: str = "fr3_hand_tcp"
    moveit_namespace: str = ""
    joint_names: tuple[str, ...] = ()
    frame_id: str = "base"
    wait_for_moveit_timeout_s: float = 15.0
    ik_timeout_s: float = 2.0
    planning_time_s: float = 5.0
    num_planning_attempts: int = 5
    velocity_scale: float = 0.05
    acceleration_scale: float = 0.05
    execute_timeout_s: float = 120.0
    post_execute_sleep_s: float = 0.5
    pregrasp_offset_m: float = 0.10
    gripper_width_clearance_m: float = 0.01
    lift_height_m: float = 0.08
    require_confirmation: bool = True
    stop_after: str = "pregrasp"
    allow_collisions: bool = False
    planning_scene_obstacles: tuple[dict[str, object], ...] = ()
    gripper_enabled: bool = False
    gripper_client: str = "franka"
    gripper_grasp_action: str = "/fr3_gripper/grasp"
    gripper_move_action: str = "/fr3_gripper/move"
    gripper_command_action: str = "/gripper_controller/gripper_cmd"
    gripper_command_position_mode: str = "width"
    gripper_command_max_effort: float = 30.0
    gripper_trigger_open_service: str = "/gripper_controller/open"
    gripper_trigger_close_service: str = "/gripper_controller/close"
    gripper_trigger_stop_service: str = "/gripper_controller/stop"
    gripper_open_width: float = 0.08
    gripper_grasp_speed: float = 0.03
    gripper_grasp_force: float = 30.0
    gripper_epsilon_inner: float = 0.002
    gripper_epsilon_outer: float = 0.08
    gripper_timeout_s: float = 10.0
    grasp_settle_time_s: float = 0.5


@dataclass(frozen=True)
class Stage1Result:
    bundle: SavedGraspBundle
    target_mesh_local: object
    target_pose_in_obj_world: ObjectWorldPose
    obstacle_mesh_world: object | None
    collision_backend_name: str
    raw_candidate_count: int
    raw_candidates: tuple[SavedGraspCandidate, ...] = ()
    surface_samples: tuple[SurfaceSample, ...] = ()


@dataclass(frozen=True)
class GroundRecheckResult:
    source_bundle: SavedGraspBundle
    accepted_bundle: SavedGraspBundle
    mesh_local: object
    pickup_pose_world: ObjectWorldPose
    pickup_spec: PickupPlacementSpec | None
    statuses: list[CandidateStatus]
    accepted: list[SavedGraspCandidate]


def _mesh_in_source_frame(mesh_obj_world, source_frame_pose_obj_world: ObjectWorldPose):
    rotation_obj_world_from_source = source_frame_pose_obj_world.rotation_world_from_object
    translation_obj_world_from_source = source_frame_pose_obj_world.translation_world
    vertices_source = (
        np.asarray(mesh_obj_world.vertices_obj, dtype=float) - translation_obj_world_from_source[None, :]
    ) @ rotation_obj_world_from_source
    return type(mesh_obj_world)(
        vertices_obj=vertices_source,
        faces=np.asarray(mesh_obj_world.faces, dtype=np.int64),
    )


def _source_frame_pose_from_bundle(bundle: SavedGraspBundle) -> ObjectWorldPose:
    return ObjectWorldPose(
        position_world=bundle.source_frame_origin_obj_world,
        orientation_xyzw_world=bundle.source_frame_orientation_xyzw_obj_world,
    )


def _minus_z_axis_in_source_frame(object_pose_world: ObjectWorldPose) -> tuple[float, float, float]:
    axis_source = object_pose_world.rotation_world_from_object.T @ np.array([0.0, 0.0, -1.0], dtype=float)
    norm = float(np.linalg.norm(axis_source))
    if norm < 1.0e-12:
        return (0.0, 0.0, -1.0)
    return tuple(float(value) for value in (axis_source / norm).tolist())


def _unique_axes(
    axes: tuple[tuple[float, float, float], ...],
    *,
    tolerance: float = 1.0e-8,
) -> tuple[tuple[float, float, float], ...]:
    unique: list[tuple[float, float, float]] = []
    for axis in axes:
        vector = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm < 1.0e-12:
            continue
        normalized = tuple(float(value) for value in (vector / norm).tolist())
        if all(float(np.linalg.norm(np.asarray(normalized) - np.asarray(existing))) > tolerance for existing in unique):
            unique.append(normalized)
    return tuple(unique)


def _stage1_upright_approach_axes(
    *,
    source_frame_pose_obj_world: ObjectWorldPose,
    extra_axes_obj: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return _unique_axes(
        (
            (0.0, 0.0, -1.0),
            _minus_z_axis_in_source_frame(source_frame_pose_obj_world),
            *extra_axes_obj,
        )
    )


def _axes_difference(
    axes: tuple[tuple[float, float, float], ...],
    base_axes: tuple[tuple[float, float, float], ...],
    *,
    tolerance: float = 1.0e-8,
) -> tuple[tuple[float, float, float], ...]:
    extras: list[tuple[float, float, float]] = []
    base_arrays = [np.asarray(axis, dtype=float) for axis in base_axes]
    for axis in axes:
        axis_array = np.asarray(axis, dtype=float)
        if all(float(np.linalg.norm(axis_array - base_axis)) > tolerance for base_axis in base_arrays):
            extras.append(axis)
    return tuple(extras)


def _candidate_score_sort_key(candidate: SavedGraspCandidate) -> tuple[float, str]:
    return (float("-inf") if candidate.score is None else float(candidate.score), candidate.grasp_id)


def _sorted_scored_candidates(candidates: tuple[SavedGraspCandidate, ...] | list[SavedGraspCandidate]):
    return tuple(sorted(candidates, key=_candidate_score_sort_key, reverse=True))


def _axes_metadata_payload(axes: tuple[tuple[float, float, float], ...]) -> list[list[float]]:
    return [[float(value) for value in axis] for axis in axes]


_STAGE1_CACHE_SCHEMA_VERSION = 14


def _path_cache_record(path: str | Path) -> dict[str, object]:
    resolved = resolve_asset_mesh_path(path)
    stat = resolved.stat()
    return {
        "path": relative_asset_mesh_path(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _explicit_assembly_obstacle_paths(geometry: GeometryConfig) -> tuple[str, ...] | None:
    if geometry.assembly_obstacle_paths is None:
        return None
    return tuple(str(path) for path in geometry.assembly_obstacle_paths)


def _assembly_obstacle_mode(geometry: GeometryConfig) -> str:
    return "explicit" if geometry.assembly_obstacle_paths is not None else "glob"


def _assembly_obstacle_sweep_vector(geometry: GeometryConfig) -> tuple[float, float, float] | None:
    if geometry.assembly_obstacle_sweep_vector_m is None:
        return None
    vector = tuple(float(value) for value in geometry.assembly_obstacle_sweep_vector_m)
    if len(vector) != 3:
        raise ValueError("assembly_obstacle_sweep_vector_m must contain exactly three values.")
    if float(np.linalg.norm(np.asarray(vector, dtype=float))) < 1.0e-12:
        return None
    return vector


def _assembly_obstacle_sweep_distance_m(geometry: GeometryConfig) -> float:
    vector = _assembly_obstacle_sweep_vector(geometry)
    if vector is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def _resolved_assembly_obstacle_paths(geometry: GeometryConfig, planning: PlanningConfig) -> list[Path]:
    if planning.skip_stage1_collision_checks:
        return []
    if geometry.assembly_obstacle_paths is not None:
        raw_paths = [resolve_asset_mesh_path(path) for path in geometry.assembly_obstacle_paths]
    else:
        if not geometry.assembly_glob:
            return []
        pattern_path = Path(geometry.assembly_glob).expanduser()
        if pattern_path.is_absolute():
            raw_paths = [Path(path) for path in glob.glob(str(pattern_path))]
        else:
            raw_paths = list(DEFAULT_ASSET_MESH_DIR.glob(geometry.assembly_glob))

    target_resolved = resolve_asset_mesh_path(geometry.target_mesh_path).resolve()
    obstacle_paths: list[Path] = []
    seen: set[Path] = set()
    for path in raw_paths:
        resolved = resolve_asset_mesh_path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            raise FileNotFoundError(f"Assembly obstacle mesh not found at '{resolved}'.")
        if resolved == target_resolved:
            continue
        obstacle_paths.append(resolved)
    return sorted(obstacle_paths)


def _assembly_cache_records(geometry: GeometryConfig, planning: PlanningConfig) -> list[dict[str, object]]:
    return [_path_cache_record(path) for path in _resolved_assembly_obstacle_paths(geometry, planning)]


def _stage1_cache_key_payload(
    *,
    geometry: GeometryConfig,
    planning: PlanningConfig,
    source_frame_pose_obj_world: ObjectWorldPose | None,
    upright_approach_axes_obj: tuple[tuple[float, float, float], ...],
) -> dict[str, object]:
    source_frame_payload = None
    if source_frame_pose_obj_world is not None:
        source_frame_payload = {
            "position_world": [float(v) for v in source_frame_pose_obj_world.position_world],
            "orientation_xyzw_world": [float(v) for v in source_frame_pose_obj_world.orientation_xyzw_world],
        }
    payload = {
        "schema_version": _STAGE1_CACHE_SCHEMA_VERSION,
        "algorithm": "fabrica_stage1_antipodal_v1",
        "grasp_scoring_algorithm": GRASP_SCORING_ALGORITHM_VERSION,
        "geometry": {
            "target_mesh": _path_cache_record(geometry.target_mesh_path),
            "mesh_scale": float(geometry.mesh_scale),
            "assembly_glob": geometry.assembly_glob,
            "assembly_obstacle_mode": _assembly_obstacle_mode(geometry),
            "assembly_obstacle_paths": (
                None
                if geometry.assembly_obstacle_paths is None
                else [str(path) for path in geometry.assembly_obstacle_paths]
            ),
            "assembly_obstacle_sweep_vector_m": (
                None
                if _assembly_obstacle_sweep_vector(geometry) is None
                else [float(value) for value in _assembly_obstacle_sweep_vector(geometry)]
            ),
            "assembly_obstacle_metadata": geometry.assembly_obstacle_metadata or {},
            "assembly_meshes": _assembly_cache_records(geometry, planning),
            "source_frame_pose_obj_world": source_frame_payload,
        },
        "planning": {
            "num_surface_samples": int(planning.num_surface_samples),
            "min_jaw_width": float(planning.min_jaw_width),
            "max_jaw_width": float(planning.max_jaw_width),
            "antipodal_cosine_threshold": float(planning.antipodal_cosine_threshold),
            "roll_angles_rad": [float(v) for v in planning.roll_angles_rad],
            "max_pair_checks": int(planning.max_pair_checks),
            "detailed_finger_contact_gap_m": float(planning.detailed_finger_contact_gap_m),
            "gripper_collision_model": planning.gripper_collision_model,
            "skip_stage1_collision_checks": bool(planning.skip_stage1_collision_checks),
            "contact_lateral_offsets_m": [float(v) for v in planning.contact_lateral_offsets_m],
            "contact_approach_offsets_m": [float(v) for v in planning.contact_approach_offsets_m],
            "rng_seed": int(planning.rng_seed),
        },
    }
    if upright_approach_axes_obj:
        payload["planning"]["upright_approach_axes_obj"] = [
            [float(value) for value in axis] for axis in upright_approach_axes_obj
        ]
    return payload


def _stage1_cache_path(
    *,
    geometry: GeometryConfig,
    planning: PlanningConfig,
    source_frame_pose_obj_world: ObjectWorldPose | None,
    upright_approach_axes_obj: tuple[tuple[float, float, float], ...],
) -> tuple[Path, str, dict[str, object]]:
    key_payload = _stage1_cache_key_payload(
        geometry=geometry,
        planning=planning,
        source_frame_pose_obj_world=source_frame_pose_obj_world,
        upright_approach_axes_obj=upright_approach_axes_obj,
    )
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    stem = Path(relative_asset_mesh_path(geometry.target_mesh_path)).with_suffix("").as_posix()
    safe_stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)[-96:] or "object"
    return Path(planning.stage1_cache_dir) / f"{safe_stem}_{key[:16]}.json", key, key_payload


def _saved_candidate_to_cache_payload(candidate: SavedGraspCandidate) -> dict[str, object]:
    return {
        "grasp_id": candidate.grasp_id,
        "grasp_pose_obj": {
            "position": list(candidate.grasp_position_obj),
            "orientation_xyzw": list(candidate.grasp_orientation_xyzw_obj),
        },
        "contact_points_obj": [list(candidate.contact_point_a_obj), list(candidate.contact_point_b_obj)],
        "contact_normals_obj": [list(candidate.contact_normal_a_obj), list(candidate.contact_normal_b_obj)],
        "jaw_width": float(candidate.jaw_width),
        "roll_angle_rad": float(candidate.roll_angle_rad),
        "contact_patch_offset_local": [
            float(candidate.contact_patch_lateral_offset_m),
            float(candidate.contact_patch_approach_offset_m),
        ],
        "score": candidate.score,
        "score_components": candidate.score_components,
        "metadata": candidate.metadata or {},
    }


def _saved_candidate_from_cache_payload(item: dict[str, object]) -> SavedGraspCandidate:
    contact_patch_offset_local = item.get("contact_patch_offset_local", [0.0, 0.0])
    return SavedGraspCandidate(
        grasp_id=str(item["grasp_id"]),
        grasp_position_obj=tuple(float(v) for v in item["grasp_pose_obj"]["position"]),  # type: ignore[index]
        grasp_orientation_xyzw_obj=tuple(
            float(v)
            for v in item["grasp_pose_obj"]["orientation_xyzw"]  # type: ignore[index]
        ),
        contact_point_a_obj=tuple(float(v) for v in item["contact_points_obj"][0]),  # type: ignore[index]
        contact_point_b_obj=tuple(float(v) for v in item["contact_points_obj"][1]),  # type: ignore[index]
        contact_normal_a_obj=tuple(float(v) for v in item["contact_normals_obj"][0]),  # type: ignore[index]
        contact_normal_b_obj=tuple(float(v) for v in item["contact_normals_obj"][1]),  # type: ignore[index]
        jaw_width=float(item["jaw_width"]),
        roll_angle_rad=float(item["roll_angle_rad"]),
        contact_patch_lateral_offset_m=float(contact_patch_offset_local[0]),  # type: ignore[index]
        contact_patch_approach_offset_m=float(contact_patch_offset_local[1]),  # type: ignore[index]
        score=None if item.get("score") is None else float(item["score"]),
        score_components=(
            None
            if item.get("score_components") is None
            else {str(k): float(v) for k, v in dict(item["score_components"]).items()}  # type: ignore[arg-type]
        ),
        metadata=dict(item.get("metadata", {})) or None,
    )


def _surface_sample_to_cache_payload(sample: SurfaceSample) -> dict[str, object]:
    return {
        "point_obj": list(sample.point_obj),
        "normal_obj": list(sample.normal_obj),
        "face_index": int(sample.face_index),
    }


def _surface_sample_from_cache_payload(item: dict[str, object]) -> SurfaceSample:
    return SurfaceSample(
        point_obj=tuple(float(v) for v in item["point_obj"]),  # type: ignore[arg-type]
        normal_obj=tuple(float(v) for v in item["normal_obj"]),  # type: ignore[arg-type]
        face_index=int(item["face_index"]),
    )


def _load_stage1_cache(
    *,
    cache_path: Path,
    cache_key: str,
    target_mesh_local,
    target_pose_in_obj_world: ObjectWorldPose,
    obstacle_mesh_world,
) -> Stage1Result | None:
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != _STAGE1_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("cache_key") != cache_key:
        return None
    bundle_payload = dict(payload["bundle"])
    metadata = dict(bundle_payload.get("metadata", {}))
    metadata.update(
        {
            "stage1_cache_hit": True,
            "stage1_cache_path": str(cache_path),
            "stage1_cache_key": cache_key,
        }
    )
    kept_candidates = tuple(
        _saved_candidate_from_cache_payload(dict(item)) for item in bundle_payload.get("candidates", [])
    )
    target_payload = dict(bundle_payload["target"])
    bundle = SavedGraspBundle(
        target_mesh_path=str(target_payload["mesh_path"]),
        mesh_scale=float(target_payload["mesh_scale"]),
        source_frame_origin_obj_world=tuple(float(v) for v in target_payload["source_frame_origin_obj_world"]),
        source_frame_orientation_xyzw_obj_world=tuple(
            float(v) for v in target_payload["source_frame_orientation_xyzw_obj_world"]
        ),
        candidates=kept_candidates,
        metadata=metadata,
    )
    raw_candidates = tuple(
        _saved_candidate_from_cache_payload(dict(item)) for item in payload.get("raw_candidates", [])
    )
    surface_samples = tuple(
        _surface_sample_from_cache_payload(dict(item)) for item in payload.get("surface_samples", [])
    )
    return Stage1Result(
        bundle=bundle,
        target_mesh_local=target_mesh_local,
        target_pose_in_obj_world=target_pose_in_obj_world,
        obstacle_mesh_world=obstacle_mesh_world,
        collision_backend_name=str(payload.get("collision_backend_name", metadata.get("collision_backend", ""))),
        raw_candidate_count=int(payload.get("raw_candidate_count", len(raw_candidates))),
        raw_candidates=raw_candidates,
        surface_samples=surface_samples,
    )


def _write_stage1_cache(
    *,
    cache_path: Path,
    cache_key: str,
    cache_key_payload: dict[str, object],
    result: Stage1Result,
) -> None:
    payload = {
        "schema_version": _STAGE1_CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "cache_key_payload": cache_key_payload,
        "collision_backend_name": result.collision_backend_name,
        "raw_candidate_count": int(result.raw_candidate_count),
        "bundle": {
            "target": {
                "mesh_path": result.bundle.target_mesh_path,
                "mesh_scale": float(result.bundle.mesh_scale),
                "source_frame_origin_obj_world": list(result.bundle.source_frame_origin_obj_world),
                "source_frame_orientation_xyzw_obj_world": list(result.bundle.source_frame_orientation_xyzw_obj_world),
            },
            "metadata": result.bundle.metadata,
            "candidates": [_saved_candidate_to_cache_payload(candidate) for candidate in result.bundle.candidates],
        },
        "raw_candidates": [_saved_candidate_to_cache_payload(candidate) for candidate in result.raw_candidates],
        "surface_samples": [_surface_sample_to_cache_payload(sample) for sample in result.surface_samples],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _augment_stage1_result_with_extra_upright_axes(
    result: Stage1Result,
    *,
    planning: PlanningConfig,
    all_upright_approach_axes_obj: tuple[tuple[float, float, float], ...],
    base_upright_approach_axes_obj: tuple[tuple[float, float, float], ...],
    extra_upright_approach_axes_obj: tuple[tuple[float, float, float], ...],
) -> Stage1Result:
    if not extra_upright_approach_axes_obj:
        return result

    extra_object_candidates = AntipodalMeshGraspGenerator(
        planning.to_generator_config(upright_approach_axes_obj=extra_upright_approach_axes_obj)
    ).generate_additional_upright_roll_candidates(
        result.target_mesh_local,
        (candidate.to_object_frame_candidate() for candidate in result.raw_candidates),
    )
    extra_raw_candidates = [
        serialize_saved_candidate(f"g{result.raw_candidate_count + index:04d}", candidate)
        for index, candidate in enumerate(extra_object_candidates, start=1)
    ]
    scored_extra_raw = (
        score_grasps(extra_raw_candidates, mesh_local=result.target_mesh_local) if extra_raw_candidates else []
    )
    if not extra_raw_candidates:
        kept_extra_candidates = []
    elif planning.skip_stage1_collision_checks:
        kept_extra_candidates = list(scored_extra_raw)
    else:
        kept_extra_candidates = filter_grasps_against_assembly(
            extra_raw_candidates,
            object_pose_world=result.target_pose_in_obj_world,
            obstacle_mesh_world=result.obstacle_mesh_world,
            contact_gap_m=planning.detailed_finger_contact_gap_m,
            gripper_collision_model=planning.gripper_collision_model,
            contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
            contact_approach_offsets_m=planning.contact_approach_offsets_m,
        )
        kept_extra_candidates = score_grasps(kept_extra_candidates, mesh_local=result.target_mesh_local)

    merged_raw_candidates = _sorted_scored_candidates([*result.raw_candidates, *scored_extra_raw])
    merged_kept_candidates = _sorted_scored_candidates([*result.bundle.candidates, *kept_extra_candidates])
    raw_candidate_count = result.raw_candidate_count + len(scored_extra_raw)
    metadata = dict(result.bundle.metadata)
    metadata.update(
        {
            "upright_approach_axes_obj": _axes_metadata_payload(all_upright_approach_axes_obj),
            "stage1_cache_base_upright_approach_axes_obj": _axes_metadata_payload(base_upright_approach_axes_obj),
            "stage1_cache_pose_upright_axes_obj": _axes_metadata_payload(extra_upright_approach_axes_obj),
            "stage1_cache_augmented": True,
            "stage1_cache_augmented_raw_candidate_count": len(scored_extra_raw),
            "stage1_cache_augmented_assembly_feasible_count": len(kept_extra_candidates),
            "raw_candidate_count": raw_candidate_count,
            "assembly_feasible_count": len(merged_kept_candidates),
            "scored_feasible_count": len(merged_kept_candidates),
        }
    )
    return replace(
        result,
        bundle=replace(result.bundle, candidates=merged_kept_candidates, metadata=metadata),
        raw_candidate_count=raw_candidate_count,
        raw_candidates=merged_raw_candidates,
    )


def generate_stage1_result(
    *,
    geometry: GeometryConfig,
    planning: PlanningConfig,
    source_frame_pose_obj_world: ObjectWorldPose | None = None,
    upright_approach_axes_obj: tuple[tuple[float, float, float], ...] = (),
) -> Stage1Result:
    target_mesh_obj_world = load_asset_mesh(geometry.target_mesh_path, scale=geometry.mesh_scale)
    if source_frame_pose_obj_world is None:
        target_mesh_local, target_pose_in_obj_world = canonicalize_target_mesh(target_mesh_obj_world)
    else:
        target_pose_in_obj_world = source_frame_pose_obj_world
        target_mesh_local = _mesh_in_source_frame(target_mesh_obj_world, target_pose_in_obj_world)
    base_upright_approach_axes_obj = _stage1_upright_approach_axes(
        source_frame_pose_obj_world=target_pose_in_obj_world,
        extra_axes_obj=(),
    )
    all_upright_approach_axes_obj = _stage1_upright_approach_axes(
        source_frame_pose_obj_world=target_pose_in_obj_world,
        extra_axes_obj=upright_approach_axes_obj,
    )
    extra_upright_approach_axes_obj = _axes_difference(
        all_upright_approach_axes_obj,
        base_upright_approach_axes_obj,
    )

    cache_path = None
    cache_key = ""
    cache_key_payload: dict[str, object] = {}
    if planning.stage1_cache_enabled:
        cache_path, cache_key, cache_key_payload = _stage1_cache_path(
            geometry=geometry,
            planning=planning,
            source_frame_pose_obj_world=source_frame_pose_obj_world,
            upright_approach_axes_obj=base_upright_approach_axes_obj,
        )
        if cache_path.exists():
            obstacle_mesh_world = None
            if not planning.skip_stage1_collision_checks:
                obstacle_mesh_world, _ = load_assembly_obstacle_mesh(
                    assembly_glob=geometry.assembly_glob,
                    assembly_paths=_explicit_assembly_obstacle_paths(geometry),
                    obstacle_sweep_vector_m=_assembly_obstacle_sweep_vector(geometry),
                    target_stl_path=geometry.target_mesh_path,
                    stl_scale=geometry.mesh_scale,
                )
            try:
                cached = _load_stage1_cache(
                    cache_path=cache_path,
                    cache_key=cache_key,
                    target_mesh_local=target_mesh_local,
                    target_pose_in_obj_world=target_pose_in_obj_world,
                    obstacle_mesh_world=obstacle_mesh_world,
                )
            except (KeyError, TypeError, ValueError):
                cached = None
            if cached is not None:
                return _augment_stage1_result_with_extra_upright_axes(
                    cached,
                    planning=planning,
                    all_upright_approach_axes_obj=all_upright_approach_axes_obj,
                    base_upright_approach_axes_obj=base_upright_approach_axes_obj,
                    extra_upright_approach_axes_obj=extra_upright_approach_axes_obj,
                )

    generator = AntipodalMeshGraspGenerator(
        planning.to_generator_config(upright_approach_axes_obj=base_upright_approach_axes_obj)
    )
    raw_candidates = generator.generate(target_mesh_local)
    surface_samples = tuple(getattr(generator, "last_surface_samples", ()))
    serialized_raw = [
        serialize_saved_candidate(f"g{index:04d}", candidate) for index, candidate in enumerate(raw_candidates, start=1)
    ]
    scored_raw = score_grasps(serialized_raw, mesh_local=target_mesh_local)

    obstacle_mesh_world = None
    obstacle_paths: tuple[str, ...] = ()
    if planning.skip_stage1_collision_checks:
        kept_candidates = list(scored_raw)
    else:
        obstacle_mesh_world, obstacle_paths = load_assembly_obstacle_mesh(
            assembly_glob=geometry.assembly_glob,
            assembly_paths=_explicit_assembly_obstacle_paths(geometry),
            obstacle_sweep_vector_m=_assembly_obstacle_sweep_vector(geometry),
            target_stl_path=geometry.target_mesh_path,
            stl_scale=geometry.mesh_scale,
        )
        kept_candidates = filter_grasps_against_assembly(
            serialized_raw,
            object_pose_world=target_pose_in_obj_world,
            obstacle_mesh_world=obstacle_mesh_world,
            contact_gap_m=planning.detailed_finger_contact_gap_m,
            gripper_collision_model=planning.gripper_collision_model,
            contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
            contact_approach_offsets_m=planning.contact_approach_offsets_m,
        )
        kept_candidates = score_grasps(kept_candidates, mesh_local=target_mesh_local)

    bundle = SavedGraspBundle(
        target_mesh_path=relative_asset_mesh_path(geometry.target_mesh_path),
        mesh_scale=geometry.mesh_scale,
        source_frame_origin_obj_world=target_pose_in_obj_world.position_world,
        source_frame_orientation_xyzw_obj_world=target_pose_in_obj_world.orientation_xyzw_world,
        candidates=tuple(kept_candidates),
        metadata={
            **_robot_metadata_for_planning(planning),
            "assembly_glob": geometry.assembly_glob,
            "assembly_obstacle_mode": _assembly_obstacle_mode(geometry),
            "assembly_obstacle_sweep_vector_m": (
                None
                if _assembly_obstacle_sweep_vector(geometry) is None
                else list(_assembly_obstacle_sweep_vector(geometry))
            ),
            "assembly_obstacle_sweep_distance_m": _assembly_obstacle_sweep_distance_m(geometry),
            "assembly_obstacle_metadata": geometry.assembly_obstacle_metadata or {},
            "collision_backend": generator.collision_backend_name,
            "gripper_collision_model": planning.gripper_collision_model,
            "stage1_collision_checks_skipped": planning.skip_stage1_collision_checks,
            "stage1_cache_enabled": planning.stage1_cache_enabled,
            "stage1_cache_hit": False,
            "stage1_cache_path": None if cache_path is None else str(cache_path),
            "stage1_cache_key": cache_key or None,
            "grasp_scoring_algorithm": GRASP_SCORING_ALGORITHM_VERSION,
            "num_surface_samples": planning.num_surface_samples,
            "surface_sample_count": len(surface_samples),
            "raw_candidate_count": len(serialized_raw),
            "assembly_feasible_count": len(kept_candidates),
            "scored_feasible_count": len(kept_candidates),
            "assembly_obstacle_paths": list(obstacle_paths),
            "contact_lateral_offsets_m": list(planning.contact_lateral_offsets_m),
            "contact_approach_offsets_m": list(planning.contact_approach_offsets_m),
            "upright_approach_axes_obj": _axes_metadata_payload(base_upright_approach_axes_obj),
            "stage1_cache_base_upright_approach_axes_obj": _axes_metadata_payload(base_upright_approach_axes_obj),
            "stage1_cache_pose_upright_axes_obj": [],
            "stage1_cache_augmented": False,
            "stage1_cache_augmented_raw_candidate_count": 0,
            "stage1_cache_augmented_assembly_feasible_count": 0,
        },
    )
    result = Stage1Result(
        bundle=bundle,
        target_mesh_local=target_mesh_local,
        target_pose_in_obj_world=target_pose_in_obj_world,
        obstacle_mesh_world=obstacle_mesh_world,
        collision_backend_name=generator.collision_backend_name,
        raw_candidate_count=len(serialized_raw),
        raw_candidates=tuple(scored_raw),
        surface_samples=surface_samples,
    )
    if planning.stage1_cache_enabled and cache_path is not None:
        _write_stage1_cache(
            cache_path=cache_path,
            cache_key=cache_key,
            cache_key_payload=cache_key_payload,
            result=result,
        )
    return _augment_stage1_result_with_extra_upright_axes(
        result,
        planning=planning,
        all_upright_approach_axes_obj=all_upright_approach_axes_obj,
        base_upright_approach_axes_obj=base_upright_approach_axes_obj,
        extra_upright_approach_axes_obj=extra_upright_approach_axes_obj,
    )


def write_stage1_artifacts(
    result: Stage1Result, *, geometry: GeometryConfig, planning: PlanningConfig, output_json: Path, output_html: Path
) -> None:
    save_grasp_bundle(result.bundle, output_json)
    obstacle_mesh_local = None
    if result.obstacle_mesh_world is not None:
        obstacle_mesh_local = _mesh_in_source_frame(result.obstacle_mesh_world, result.target_pose_in_obj_world)
    obstacle_metadata = dict(result.bundle.metadata.get("assembly_obstacle_metadata", {}) or {})
    metadata_lines = [
        f"target_mesh:      {relative_asset_mesh_path(geometry.target_mesh_path)}",
        f"assembly_glob:    {geometry.assembly_glob}",
        f"obstacle_mode:    {_assembly_obstacle_mode(geometry)}",
        f"obstacle_count:   {len(result.bundle.metadata.get('assembly_obstacle_paths', []))}",
        f"sweep_vector_m:   {result.bundle.metadata.get('assembly_obstacle_sweep_vector_m')}",
        f"sweep_distance_m: {float(result.bundle.metadata.get('assembly_obstacle_sweep_distance_m', 0.0)):.6f}",
        f"precedence_plan:  {obstacle_metadata.get('precedence_plan_path', 'none')}",
        f"assembled_before: {obstacle_metadata.get('already_assembled_part_ids', [])}",
        f"collision_backend:{result.collision_backend_name}",
        f"gripper_model:   {planning.gripper_collision_model}",
        f"stage1_collision:{'skipped' if planning.skip_stage1_collision_checks else 'enabled'}",
        f"raw_candidates:   {result.raw_candidate_count}",
        f"assembly_feasible:{len(result.bundle.candidates)}",
        f"contact_offsets_x:{tuple(planning.contact_lateral_offsets_m)}",
        f"contact_offsets_z:{tuple(planning.contact_approach_offsets_m)}",
        f"local_origin_src: {tuple(round(v, 6) for v in result.target_pose_in_obj_world.position_world)}",
    ]
    write_debug_html(
        title="Fabrica Assembly-Feasible Grasps",
        subtitle="Offline assembly collision screening. Candidates are stored and visualized in the target part-local frame.",
        mesh_local=result.target_mesh_local,
        candidate_statuses=[
            CandidateStatus(
                grasp=candidate,
                status="accepted",
                reason="assembly_skipped" if planning.skip_stage1_collision_checks else "assembly_clear",
            )
            for candidate in result.bundle.candidates
        ],
        output_html=output_html,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
        obstacle_mesh_local=obstacle_mesh_local,
        metadata_lines=metadata_lines,
        gripper_collision_model=planning.gripper_collision_model,
    )


def _score_grasps_for_world_top_approach(
    grasps: list[SavedGraspCandidate],
    *,
    mesh_local,
    object_pose_world: ObjectWorldPose,
    top_grasp_score_weight: float,
    reachability_proxy_score_weight: float = DEFAULT_REACHABILITY_PROXY_SCORE_WEIGHT,
    reachability_proxy_hand_offset_m: float = DEFAULT_REACHABILITY_PROXY_HAND_OFFSET_M,
) -> list[SavedGraspCandidate]:
    object_scored = score_grasps(grasps, mesh_local=mesh_local)
    top_weight = min(1.0, max(0.0, float(top_grasp_score_weight)))
    reachability_weight = min(1.0, max(0.0, float(reachability_proxy_score_weight)))
    world_weight = top_weight + reachability_weight
    if world_weight > 1.0:
        top_weight /= world_weight
        reachability_weight /= world_weight
        world_weight = 1.0
    object_weight = 1.0 - world_weight
    if top_weight <= 0.0 and reachability_weight <= 0.0:
        return object_scored

    world_scored: list[SavedGraspCandidate] = []
    for grasp in object_scored:
        grasp_rot_obj = quat_to_rotmat_xyzw(grasp.grasp_orientation_xyzw_obj)
        approach_axis_world = object_pose_world.rotation_world_from_object @ grasp_rot_obj[:, 2]
        top_down_score = min(1.0, max(0.0, float(-approach_axis_world[2])))
        reachability = _runtime_reachability_proxy_components(
            grasp,
            mesh_local=mesh_local,
            object_pose_world=object_pose_world,
            approach_axis_world=approach_axis_world,
            hand_offset_m=reachability_proxy_hand_offset_m,
        )
        object_score = 0.0 if grasp.score is None else float(grasp.score)
        combined_score = (
            object_weight * object_score
            + top_weight * top_down_score
            + reachability_weight * reachability["reachability_proxy"]
        )
        score_components = dict(grasp.score_components or {})
        score_components["object_score"] = object_score
        score_components["top_down_approach"] = top_down_score
        score_components["world_approach_z"] = float(approach_axis_world[2])
        score_components["top_grasp_score_weight"] = top_weight
        score_components["reachability_proxy_score_weight"] = reachability_weight
        score_components["world_object_score_weight"] = object_weight
        score_components.update(reachability)
        score_components["score"] = float(combined_score)
        world_scored.append(
            replace(
                grasp,
                score=float(combined_score),
                score_components=score_components,
            )
        )
    return sorted(
        world_scored,
        key=lambda candidate: (
            float("-inf") if candidate.score is None else float(candidate.score),
            candidate.grasp_id,
        ),
        reverse=True,
    )


def _runtime_reachability_proxy_components(
    grasp: SavedGraspCandidate,
    *,
    mesh_local,
    object_pose_world: ObjectWorldPose,
    approach_axis_world: np.ndarray,
    hand_offset_m: float,
) -> dict[str, float]:
    object_center_world = _mesh_aabb_center_world(mesh_local, object_pose_world)
    grasp_position_world = object_pose_world.transform_points_to_world(
        np.asarray([grasp.grasp_position_obj], dtype=float)
    )[0]
    backoff_axis_world = -_normalize_or_zero(np.asarray(approach_axis_world, dtype=float))
    hand_position_world = grasp_position_world + backoff_axis_world * max(0.0, float(hand_offset_m))

    base_xy = np.zeros(2, dtype=float)
    object_xy = object_center_world[:2]
    hand_xy = hand_position_world[:2]
    object_from_base_xy = object_xy - base_xy
    object_radius_m = float(np.linalg.norm(object_from_base_xy))
    hand_radius_m = float(np.linalg.norm(hand_xy - base_xy))
    object_direction_xy = _normalize_or_zero(object_from_base_xy)
    backoff_xy = _normalize_or_zero(backoff_axis_world[:2])
    hand_side = 0.0 if np.linalg.norm(object_direction_xy) < 1.0e-9 else float(np.dot(backoff_xy, object_direction_xy))
    hand_side = float(np.clip(hand_side, -1.0, 1.0))
    target_side = float(
        np.clip(
            (DEFAULT_REACHABILITY_PROXY_COMFORT_RADIUS_M - object_radius_m)
            / max(DEFAULT_REACHABILITY_PROXY_COMFORT_BAND_M, 1.0e-9),
            -1.0,
            1.0,
        )
    )
    if target_side >= 0.0:
        raw_side_score = 0.5 * (1.0 + hand_side)
    else:
        raw_side_score = 0.5 * (1.0 - hand_side)
    side_strength = abs(target_side)
    side_score = (1.0 - side_strength) + side_strength * raw_side_score
    hand_radial_score = _trapezoid_score(
        hand_radius_m,
        zero_below=DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_ZERO_MIN_M,
        full_from=DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_FULL_MIN_M,
        full_to=DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_FULL_MAX_M,
        zero_above=DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_ZERO_MAX_M,
    )
    pregrasp_clearance_score, pregrasp_clearance_m, pregrasp_floor_margin_m = _pregrasp_clearance_score(
        hand_position_world,
        mesh_local=mesh_local,
        object_pose_world=object_pose_world,
    )
    reachability_proxy = 0.65 * hand_radial_score + 0.25 * _clamp01(side_score) + 0.10 * pregrasp_clearance_score
    return {
        "reachability_proxy": _clamp01(reachability_proxy),
        "reachability_hand_radial": hand_radial_score,
        "reachability_side": _clamp01(side_score),
        "reachability_pregrasp_clearance": pregrasp_clearance_score,
        "reachability_hand_radius_m": hand_radius_m,
        "reachability_object_radius_m": object_radius_m,
        "reachability_hand_side": hand_side,
        "reachability_target_side": target_side,
        "reachability_hand_offset_m": max(0.0, float(hand_offset_m)),
        "reachability_pregrasp_clearance_m": pregrasp_clearance_m,
        "reachability_pregrasp_floor_margin_m": pregrasp_floor_margin_m,
        "reachability_pregrasp_x_world": float(hand_position_world[0]),
        "reachability_pregrasp_y_world": float(hand_position_world[1]),
        "reachability_pregrasp_z_world": float(hand_position_world[2]),
    }


def _mesh_aabb_center_world(mesh_local, object_pose_world: ObjectWorldPose) -> np.ndarray:
    vertices_obj = getattr(mesh_local, "vertices_obj", None)
    if vertices_obj is None:
        return object_pose_world.translation_world
    vertices = np.asarray(vertices_obj, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] != 3:
        return object_pose_world.translation_world
    center_obj = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    return object_pose_world.transform_points_to_world(np.asarray([center_obj], dtype=float))[0]


def _pregrasp_clearance_score(
    pregrasp_position_world: np.ndarray,
    *,
    mesh_local,
    object_pose_world: ObjectWorldPose,
) -> tuple[float, float, float]:
    floor_margin_m = float(pregrasp_position_world[2])
    floor_score = _clamp01(floor_margin_m / max(DEFAULT_REACHABILITY_PROXY_FLOOR_CLEARANCE_M, 1.0e-9))
    vertices_obj = getattr(mesh_local, "vertices_obj", None)
    if vertices_obj is None:
        return floor_score, 1.0e9, floor_margin_m
    vertices = np.asarray(vertices_obj, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] != 3:
        return floor_score, 1.0e9, floor_margin_m
    vertices_world = object_pose_world.transform_points_to_world(vertices)
    bounds_min = vertices_world.min(axis=0)
    bounds_max = vertices_world.max(axis=0)
    outside = np.maximum(np.maximum(bounds_min - pregrasp_position_world, pregrasp_position_world - bounds_max), 0.0)
    distance_m = float(np.linalg.norm(outside))
    object_score = _clamp01(distance_m / max(DEFAULT_REACHABILITY_PROXY_OBJECT_CLEARANCE_M, 1.0e-9))
    return floor_score * object_score, distance_m, floor_margin_m


def _normalize_or_zero(vec: np.ndarray) -> np.ndarray:
    array = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm < 1.0e-9:
        return np.zeros_like(array)
    return array / norm


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _trapezoid_score(
    value: float,
    *,
    zero_below: float,
    full_from: float,
    full_to: float,
    zero_above: float,
) -> float:
    value = float(value)
    if value <= zero_below or value >= zero_above:
        return 0.0
    if full_from <= value <= full_to:
        return 1.0
    if value < full_from:
        return _clamp01((value - zero_below) / max(full_from - zero_below, 1.0e-9))
    return _clamp01((zero_above - value) / max(zero_above - full_to, 1.0e-9))


def _symmetry_part_key(target_mesh_path: str) -> tuple[str, str] | None:
    relative_path = Path(relative_asset_mesh_path(target_mesh_path))
    parts = relative_path.parts
    if "fabrica" not in parts:
        return None
    fabrica_index = parts.index("fabrica")
    if len(parts) <= fabrica_index + 2:
        return None
    return parts[fabrica_index + 1], Path(parts[fabrica_index + 2]).stem


def _configured_symmetry_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "assets":
        return (DEFAULT_ASSET_MESH_DIR.parent / path).resolve()
    return (DEFAULT_ASSET_MESH_DIR / path).resolve()


def _default_symmetry_path(target_mesh_path: str) -> Path | None:
    key = _symmetry_part_key(target_mesh_path)
    if key is None:
        return None
    assembly, _part_id = key
    return (DEFAULT_ASSET_MESH_DIR / "obj" / "fabrica" / assembly / "symmetries.json").resolve()


def _identity_symmetry_record() -> dict[str, object]:
    return {
        "name": "identity",
        "type": "identity",
        "description": "Identity",
        "matrix_obj": np.eye(4, dtype=float).tolist(),
        "angle_deg": 0.0,
        "source": "identity",
    }


def _clean_symmetry_record(record: dict[str, object], *, translation_scale: float = 1.0) -> dict[str, object] | None:
    try:
        matrix = np.asarray(record.get("matrix_obj"), dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    matrix = matrix.copy()
    matrix[:3, 3] *= float(translation_scale)
    name = str(record.get("name") or "symmetry")
    return {
        "name": name,
        "type": str(record.get("type") or "finite_rotation"),
        "description": str(record.get("description") or name),
        "source": str(record.get("source") or "unknown"),
        "angle_deg": float(record.get("angle_deg", 0.0) or 0.0),
        "matrix_obj": [[float(value) for value in row] for row in matrix.tolist()],
    }


def _is_identity_symmetry(record: dict[str, object]) -> bool:
    if str(record.get("type", "")) == "identity" or str(record.get("name", "")) == "identity":
        return True
    return bool(np.allclose(np.asarray(record["matrix_obj"], dtype=float), np.eye(4), atol=1.0e-9))


def _positive_finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _symmetry_asset_mesh_scale(payload: dict[str, object], part_payload: dict[str, object]) -> float | None:
    return _positive_finite_float(part_payload.get("mesh_scale", payload.get("mesh_scale")))


def _symmetry_matrix_in_source_frame(bundle: SavedGraspBundle, record: dict[str, object]) -> np.ndarray:
    matrix_obj = np.asarray(record["matrix_obj"], dtype=float)
    source_frame_pose = _source_frame_pose_from_bundle(bundle)
    rotation_obj_from_source = source_frame_pose.rotation_world_from_object
    translation_obj_from_source = source_frame_pose.translation_world

    obj_from_source = np.eye(4, dtype=float)
    obj_from_source[:3, :3] = rotation_obj_from_source
    obj_from_source[:3, 3] = translation_obj_from_source

    source_from_obj = np.eye(4, dtype=float)
    source_from_obj[:3, :3] = rotation_obj_from_source.T
    source_from_obj[:3, 3] = -(rotation_obj_from_source.T @ translation_obj_from_source)

    return source_from_obj @ matrix_obj @ obj_from_source


def _symmetry_records_for_bundle(
    bundle: SavedGraspBundle,
    planning: PlanningConfig,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    metadata: dict[str, object] = {
        "symmetry_pickup_enabled": bool(planning.symmetry_pickup_enabled),
        "symmetry_pickup_load_status": "disabled",
    }
    if not planning.symmetry_pickup_enabled:
        return (), metadata

    part_key = _symmetry_part_key(bundle.target_mesh_path)
    if part_key is None:
        metadata["symmetry_pickup_load_status"] = "unsupported_mesh_path"
        return (), metadata
    assembly, part_id = part_key
    raw_path = str(planning.symmetry_asset_path).strip()
    symmetry_path = _configured_symmetry_path(raw_path) if raw_path else _default_symmetry_path(bundle.target_mesh_path)
    metadata.update(
        {
            "symmetry_pickup_assembly": assembly,
            "symmetry_pickup_part_id": part_id,
            "symmetry_pickup_source_path": None if symmetry_path is None else str(symmetry_path),
        }
    )
    if symmetry_path is None or not symmetry_path.exists():
        metadata["symmetry_pickup_load_status"] = "missing_file"
        return (), metadata

    payload = json.loads(symmetry_path.read_text(encoding="utf-8"))
    part_payload = dict(dict(payload.get("parts", {})).get(part_id, {}) or {})
    raw_records = part_payload.get("symmetries", [])
    if not isinstance(raw_records, list):
        metadata["symmetry_pickup_load_status"] = "missing_part"
        return (), metadata

    bundle_mesh_scale = _positive_finite_float(bundle.mesh_scale)
    if bundle_mesh_scale is None:
        metadata["symmetry_pickup_load_status"] = "invalid_bundle_mesh_scale"
        metadata["symmetry_pickup_bundle_mesh_scale"] = float(bundle.mesh_scale)
        return (), metadata

    asset_mesh_scale = _symmetry_asset_mesh_scale(payload, part_payload)
    if asset_mesh_scale is None:
        if not math.isclose(bundle_mesh_scale, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
            metadata.update(
                {
                    "symmetry_pickup_load_status": "missing_mesh_scale",
                    "symmetry_pickup_bundle_mesh_scale": bundle_mesh_scale,
                }
            )
            return (), metadata
        asset_mesh_scale = 1.0
    translation_scale = bundle_mesh_scale / asset_mesh_scale

    cleaned = [
        record
        for item in raw_records
        if isinstance(item, dict)
        for record in [_clean_symmetry_record(item, translation_scale=translation_scale)]
        if record
    ]
    if not cleaned:
        metadata["symmetry_pickup_load_status"] = "no_valid_records"
        return (), metadata

    identity_records = [record for record in cleaned if _is_identity_symmetry(record)] or [_identity_symmetry_record()]
    nonidentity_records = [record for record in cleaned if not _is_identity_symmetry(record)]
    max_transforms = int(planning.symmetry_max_transforms)
    if max_transforms > 0:
        nonidentity_records = nonidentity_records[:max_transforms]
    records = tuple([identity_records[0], *nonidentity_records])
    metadata.update(
        {
            "symmetry_pickup_load_status": "loaded",
            "symmetry_pickup_available_transform_count": len(cleaned),
            "symmetry_pickup_transform_count": len(nonidentity_records),
            "symmetry_pickup_transform_names": [str(record["name"]) for record in nonidentity_records],
            "symmetry_pickup_asset_mesh_scale": asset_mesh_scale,
            "symmetry_pickup_bundle_mesh_scale": bundle_mesh_scale,
            "symmetry_pickup_translation_scale": translation_scale,
        }
    )
    return records, metadata


def _safe_symmetry_id_fragment(name: str) -> str:
    fragment = "".join(char if char.isalnum() or char in "._-" else "_" for char in name.strip())
    return fragment[:80] or "symmetry"


def _normalize_vector_tuple(vector: object) -> tuple[float, float, float]:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        return tuple(float(value) for value in array.tolist())  # type: ignore[return-value]
    return tuple(float(value) for value in (array / norm).tolist())  # type: ignore[return-value]


def _symmetry_pickup_metadata(candidate: SavedGraspCandidate, record: dict[str, object]) -> dict[str, object]:
    metadata = dict(candidate.metadata or {})
    metadata.update(
        {
            "symmetry_pickup_parent_grasp_id": candidate.grasp_id,
            "symmetry_pickup_name": str(record["name"]),
            "symmetry_pickup_description": str(record["description"]),
            "symmetry_pickup_source": str(record["source"]),
            "symmetry_pickup_angle_deg": float(record["angle_deg"]),
            "symmetry_pickup_matrix_obj": record["matrix_obj"],
            "symmetry_pickup_is_identity": _is_identity_symmetry(record),
        }
    )
    return metadata


def _candidate_with_identity_symmetry(
    candidate: SavedGraspCandidate,
    record: dict[str, object],
) -> SavedGraspCandidate:
    return replace(candidate, metadata=_symmetry_pickup_metadata(candidate, record))


def _candidate_transformed_by_symmetry(
    candidate: SavedGraspCandidate,
    record: dict[str, object],
    bundle: SavedGraspBundle,
) -> SavedGraspCandidate:
    matrix = _symmetry_matrix_in_source_frame(bundle, record)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    def transform_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
        transformed = rotation @ np.asarray(point, dtype=float) + translation
        return tuple(float(value) for value in transformed.tolist())

    def transform_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
        return _normalize_vector_tuple(rotation @ np.asarray(vector, dtype=float))

    grasp_rotation = rotation @ quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj)
    return SavedGraspCandidate(
        grasp_id=f"{candidate.grasp_id}__sym_{_safe_symmetry_id_fragment(str(record['name']))}",
        grasp_position_obj=transform_point(candidate.grasp_position_obj),
        grasp_orientation_xyzw_obj=rotmat_to_quat_xyzw(grasp_rotation),
        contact_point_a_obj=transform_point(candidate.contact_point_a_obj),
        contact_point_b_obj=transform_point(candidate.contact_point_b_obj),
        contact_normal_a_obj=transform_vector(candidate.contact_normal_a_obj),
        contact_normal_b_obj=transform_vector(candidate.contact_normal_b_obj),
        jaw_width=candidate.jaw_width,
        roll_angle_rad=candidate.roll_angle_rad,
        contact_patch_lateral_offset_m=candidate.contact_patch_lateral_offset_m,
        contact_patch_approach_offset_m=candidate.contact_patch_approach_offset_m,
        score=candidate.score,
        score_components=None if candidate.score_components is None else dict(candidate.score_components),
        metadata=_symmetry_pickup_metadata(candidate, record),
    )


def _rounded_grasp_values(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(float(value), 7) for value in values)


def _grasp_geometry_key(candidate: SavedGraspCandidate) -> tuple[float, ...]:
    quat = np.asarray(candidate.grasp_orientation_xyzw_obj, dtype=float)
    if quat[3] < 0.0:
        quat = -quat
    return _rounded_grasp_values(
        tuple(candidate.grasp_position_obj)
        + tuple(float(value) for value in quat.tolist())
        + tuple(candidate.contact_point_a_obj)
        + tuple(candidate.contact_point_b_obj)
        + tuple(candidate.contact_normal_a_obj)
        + tuple(candidate.contact_normal_b_obj)
        + (
            float(candidate.jaw_width),
            float(candidate.contact_patch_lateral_offset_m),
            float(candidate.contact_patch_approach_offset_m),
        )
    )


def _symmetry_pickup_candidates(
    bundle: SavedGraspBundle,
    planning: PlanningConfig,
) -> tuple[tuple[SavedGraspCandidate, ...], tuple[dict[str, object], ...], dict[str, object]]:
    records, metadata = _symmetry_records_for_bundle(bundle, planning)
    if not records:
        metadata.update(
            {
                "symmetry_pickup_source_candidate_count": len(bundle.candidates),
                "symmetry_pickup_expanded_candidate_count": len(bundle.candidates),
                "symmetry_pickup_deduplicated_candidate_count": 0,
            }
        )
        return tuple(bundle.candidates), records, metadata

    identity = records[0]
    transforms = records[1:]
    expanded: list[SavedGraspCandidate] = []
    seen: set[tuple[float, ...]] = set()
    deduplicated = 0
    for candidate in bundle.candidates:
        for expanded_candidate in (
            _candidate_with_identity_symmetry(candidate, identity),
            *(_candidate_transformed_by_symmetry(candidate, record, bundle) for record in transforms),
        ):
            key = _grasp_geometry_key(expanded_candidate)
            if key in seen:
                deduplicated += 1
                continue
            seen.add(key)
            expanded.append(expanded_candidate)
    metadata.update(
        {
            "symmetry_pickup_source_candidate_count": len(bundle.candidates),
            "symmetry_pickup_expanded_candidate_count": len(expanded),
            "symmetry_pickup_derived_candidate_count": max(0, len(expanded) - len(bundle.candidates)),
            "symmetry_pickup_deduplicated_candidate_count": deduplicated,
        }
    )
    return tuple(expanded), records, metadata


def _symmetry_parent_summaries(candidates: list[SavedGraspCandidate]) -> list[dict[str, object]]:
    by_parent: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        metadata = dict(candidate.metadata or {})
        parent_id = str(metadata.get("symmetry_pickup_parent_grasp_id", candidate.grasp_id))
        summary = by_parent.setdefault(
            parent_id,
            {
                "parent_grasp_id": parent_id,
                "feasible_variant_count": 0,
                "feasible_symmetry_names": set(),
            },
        )
        summary["feasible_variant_count"] = int(summary["feasible_variant_count"]) + 1
        summary["feasible_symmetry_names"].add(str(metadata.get("symmetry_pickup_name", "identity")))
    normalized: list[dict[str, object]] = []
    for summary in by_parent.values():
        normalized.append(
            {
                "parent_grasp_id": summary["parent_grasp_id"],
                "feasible_variant_count": summary["feasible_variant_count"],
                "feasible_symmetry_names": sorted(summary["feasible_symmetry_names"]),
            }
        )
    return sorted(normalized, key=lambda item: (-int(item["feasible_variant_count"]), str(item["parent_grasp_id"])))


def _object_pose_after_symmetry(base_pose: ObjectWorldPose, matrix: np.ndarray) -> ObjectWorldPose:
    rotation = base_pose.rotation_world_from_object @ matrix[:3, :3]
    translation = base_pose.rotation_world_from_object @ matrix[:3, 3] + base_pose.translation_world
    return ObjectWorldPose(
        position_world=tuple(float(value) for value in translation.tolist()),
        orientation_xyzw_world=rotmat_to_quat_xyzw(rotation),
    )


def _rotation_angle_rad(rotation: np.ndarray) -> float:
    cosine = 0.5 * (float(np.trace(rotation)) - 1.0)
    return float(math.acos(min(1.0, max(-1.0, cosine))))


def _point_to_world(point_obj: tuple[float, float, float], pose: ObjectWorldPose) -> np.ndarray:
    return pose.rotation_world_from_object @ np.asarray(point_obj, dtype=float) + pose.translation_world


def _symmetry_next_orientation_options(
    accepted: list[SavedGraspCandidate],
    *,
    bundle: SavedGraspBundle,
    pickup_pose_world: ObjectWorldPose,
    final_pose_world: ObjectWorldPose,
    symmetry_records: tuple[dict[str, object], ...],
    limit: int,
) -> list[dict[str, object]]:
    if not accepted or not symmetry_records or limit <= 0:
        return []
    options: list[dict[str, object]] = []
    for grasp in accepted:
        grasp_rotation_obj = quat_to_rotmat_xyzw(grasp.grasp_orientation_xyzw_obj)
        pickup_grasp_rotation = pickup_pose_world.rotation_world_from_object @ grasp_rotation_obj
        pickup_grasp_position = _point_to_world(grasp.grasp_position_obj, pickup_pose_world)
        pickup_metadata = dict(grasp.metadata or {})
        for record in symmetry_records:
            final_pose = _object_pose_after_symmetry(
                final_pose_world,
                _symmetry_matrix_in_source_frame(bundle, record),
            )
            final_grasp_rotation = final_pose.rotation_world_from_object @ grasp_rotation_obj
            wrist_rotation_rad = _rotation_angle_rad(final_grasp_rotation @ pickup_grasp_rotation.T)
            final_grasp_position = _point_to_world(grasp.grasp_position_obj, final_pose)
            translation_m = float(np.linalg.norm(final_grasp_position - pickup_grasp_position))
            options.append(
                {
                    "pickup_grasp_id": grasp.grasp_id,
                    "parent_grasp_id": str(pickup_metadata.get("symmetry_pickup_parent_grasp_id", grasp.grasp_id)),
                    "pickup_symmetry_name": str(pickup_metadata.get("symmetry_pickup_name", "identity")),
                    "final_symmetry_name": str(record["name"]),
                    "final_symmetry_description": str(record["description"]),
                    "rank_score": float(wrist_rotation_rad + translation_m),
                    "wrist_rotation_deg": float(math.degrees(wrist_rotation_rad)),
                    "grasp_translation_m": translation_m,
                    "final_object_pose": {
                        "position_world": list(final_pose.position_world),
                        "orientation_xyzw_world": list(final_pose.orientation_xyzw_world),
                    },
                }
            )
    return sorted(options, key=lambda item: (float(item["rank_score"]), str(item["pickup_grasp_id"])))[:limit]


def recheck_stage2_result(
    *,
    bundle: SavedGraspBundle,
    pickup_spec: PickupPlacementSpec | None,
    planning: PlanningConfig,
    object_pose_world: ObjectWorldPose | None = None,
) -> GroundRecheckResult:
    mesh_obj_world = load_asset_mesh(bundle.target_mesh_path, scale=bundle.mesh_scale)
    mesh_local = _mesh_in_source_frame(mesh_obj_world, _source_frame_pose_from_bundle(bundle))
    if object_pose_world is None:
        if pickup_spec is None:
            raise ValueError("Either pickup_spec or object_pose_world must be provided.")
        pickup_pose_world = build_pickup_pose_world(
            mesh_local,
            support_face=pickup_spec.support_face,
            yaw_deg=pickup_spec.yaw_deg,
            xy_world=pickup_spec.xy_world,
        )
    else:
        pickup_pose_world = object_pose_world
    pickup_candidates, symmetry_records, symmetry_metadata = _symmetry_pickup_candidates(bundle, planning)
    source_bundle = replace(
        bundle,
        candidates=tuple(pickup_candidates),
        metadata={**dict(bundle.metadata), **symmetry_metadata},
    )
    statuses = evaluate_saved_grasps_against_pickup_pose(
        pickup_candidates,
        object_pose_world=pickup_pose_world,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
        gripper_collision_model=planning.gripper_collision_model,
        floor_clearance_margin_m=planning.floor_clearance_margin_m,
        contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
        contact_approach_offsets_m=planning.contact_approach_offsets_m,
    )
    accepted = _score_grasps_for_world_top_approach(
        accepted_grasps(statuses),
        mesh_local=mesh_local,
        object_pose_world=pickup_pose_world,
        top_grasp_score_weight=planning.top_grasp_score_weight,
        reachability_proxy_score_weight=planning.reachability_proxy_score_weight,
        reachability_proxy_hand_offset_m=planning.reachability_proxy_hand_offset_m,
    )
    rescored_by_id = {grasp.grasp_id: grasp for grasp in accepted}
    rescored_statuses = [
        CandidateStatus(
            grasp=rescored_by_id.get(entry.grasp.grasp_id, entry.grasp),
            status=entry.status,
            reason=entry.reason,
        )
        for entry in statuses
    ]
    final_pose_world = _source_frame_pose_from_bundle(bundle)
    next_orientation_options = _symmetry_next_orientation_options(
        accepted,
        bundle=bundle,
        pickup_pose_world=pickup_pose_world,
        final_pose_world=final_pose_world,
        symmetry_records=symmetry_records,
        limit=int(planning.symmetry_next_orientation_limit),
    )
    metadata = dict(source_bundle.metadata)
    metadata.update(
        {
            "pickup_support_face": None if pickup_spec is None else pickup_spec.support_face,
            "pickup_yaw_deg": None if pickup_spec is None else float(pickup_spec.yaw_deg),
            "pickup_xy_world": None if pickup_spec is None else list(pickup_spec.xy_world),
            "execution_world_pose": {
                "position_world": list(pickup_pose_world.position_world),
                "orientation_xyzw_world": list(pickup_pose_world.orientation_xyzw_world),
            },
            "nominal_assembly_world_pose": {
                "position_world": list(final_pose_world.position_world),
                "orientation_xyzw_world": list(final_pose_world.orientation_xyzw_world),
            },
            "ground_original_input_count": len(bundle.candidates),
            "ground_input_count": len(pickup_candidates),
            "ground_feasible_count": len(accepted),
            "gripper_collision_model": planning.gripper_collision_model,
            **_robot_metadata_for_planning(planning),
            "symmetry_pickup_feasible_count": len(accepted),
            "symmetry_pickup_parent_summaries": _symmetry_parent_summaries(accepted),
            "symmetry_next_orientation_options": next_orientation_options,
            "top_grasp_score_weight": planning.top_grasp_score_weight,
            "reachability_proxy_score_weight": planning.reachability_proxy_score_weight,
            "reachability_proxy_hand_offset_m": planning.reachability_proxy_hand_offset_m,
        }
    )
    accepted_bundle = SavedGraspBundle(
        target_mesh_path=bundle.target_mesh_path,
        mesh_scale=bundle.mesh_scale,
        source_frame_origin_obj_world=bundle.source_frame_origin_obj_world,
        source_frame_orientation_xyzw_obj_world=bundle.source_frame_orientation_xyzw_obj_world,
        candidates=tuple(accepted),
        metadata=metadata,
    )
    return GroundRecheckResult(
        source_bundle=source_bundle,
        accepted_bundle=accepted_bundle,
        mesh_local=mesh_local,
        pickup_pose_world=pickup_pose_world,
        pickup_spec=pickup_spec,
        statuses=rescored_statuses,
        accepted=accepted,
    )


def write_stage2_artifacts(
    result: GroundRecheckResult, *, planning: PlanningConfig, output_json: Path, output_html: Path
) -> None:
    save_grasp_bundle(result.accepted_bundle, output_json)
    from grasp_planning.grasping.fabrica_grasp_debug import ground_plane_overlay_obj

    write_debug_html(
        title="Fabrica Pickup Ground Recheck",
        subtitle="Saved assembly-feasible grasps rechecked against the pickup-ground constraint. The HTML view is rendered in the selected execution-world pose.",
        mesh_local=result.mesh_local,
        candidate_statuses=result.statuses,
        output_html=output_html,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
        ground_plane=ground_plane_overlay_obj(
            result.mesh_local, object_pose_world=result.pickup_pose_world, enabled=True
        ),
        display_object_pose_world=result.pickup_pose_world,
        metadata_lines=[
            f"target_mesh:      {relative_asset_mesh_path(result.source_bundle.target_mesh_path)}",
            f"input_grasps:     {result.accepted_bundle.metadata.get('ground_original_input_count', len(result.source_bundle.candidates))}",
            f"pickup_expanded:  {len(result.source_bundle.candidates)}",
            f"ground_feasible:  {len(result.accepted)}",
            f"symmetry_pickup:  {result.accepted_bundle.metadata.get('symmetry_pickup_load_status', 'disabled')}",
            f"symmetry_variants:{result.accepted_bundle.metadata.get('symmetry_pickup_derived_candidate_count', 0)}",
            f"support_face:     {result.pickup_spec.support_face if result.pickup_spec is not None else 'explicit_pose'}",
            f"pickup_yaw_deg:   {float(result.pickup_spec.yaw_deg):.1f}"
            if result.pickup_spec is not None
            else "pickup_yaw_deg:   n/a",
            f"contact_offsets_x:{tuple(planning.contact_lateral_offsets_m)}",
            f"contact_offsets_z:{tuple(planning.contact_approach_offsets_m)}",
            f"floor_clearance: {planning.floor_clearance_margin_m:.6f} m",
            f"top_score_weight: {planning.top_grasp_score_weight:.3f}",
            f"reach_score_wt:   {planning.reachability_proxy_score_weight:.3f}",
            f"reach_hand_off:   {planning.reachability_proxy_hand_offset_m:.3f} m",
            f"pickup_pos_w:     {tuple(round(v, 6) for v in result.pickup_pose_world.position_world)}",
        ],
        gripper_collision_model=planning.gripper_collision_model,
    )
