"""Shared Fabrica-style planning pipeline for sim, pitl, and real flows."""

from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from grasp_planning.grasping import AntipodalGraspGeneratorConfig, AntipodalMeshGraspGenerator
from grasp_planning.grasping.fabrica_grasp_debug import (
    DEFAULT_CONTACT_APPROACH_OFFSETS_M,
    DEFAULT_CONTACT_LATERAL_OFFSETS_M,
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
    save_grasp_bundle,
    score_grasps,
    serialize_saved_candidate,
    write_debug_html,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import SurfaceSample
from grasp_planning.grasping.mesh_io import DEFAULT_ASSET_MESH_DIR
from grasp_planning.grasping.world_constraints import ObjectWorldPose


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
    floor_clearance_margin_m: float = 0.0
    skip_stage1_collision_checks: bool = False
    top_grasp_score_weight: float = 0.35
    regrasp_transfer_top_grasp_score_weight: float = 0.85
    contact_lateral_offsets_m: tuple[float, ...] = DEFAULT_CONTACT_LATERAL_OFFSETS_M
    contact_approach_offsets_m: tuple[float, ...] = DEFAULT_CONTACT_APPROACH_OFFSETS_M
    rng_seed: int = 0

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
            rng_seed=self.rng_seed,
        )


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
    moveit_planner_id: str = ""
    moveit_wait_for_moveit_timeout_s: float = 15.0
    moveit_ik_timeout_s: float = 2.0
    moveit_planning_time_s: float = 5.0
    moveit_num_planning_attempts: int = 5
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
    controller: str = "admittance"
    grasp_id: str = ""
    pregrasp_offset: float | None = None
    gripper_width_clearance: float | None = None
    contact_gap_m: float | None = None
    close_width: float = 0.0
    tcp_to_grasp_offset: tuple[float, float, float] | None = None
    attempt_artifact: str = "artifacts/isaac_pick_attempt.json"
    pregrasp_only: bool = False
    run_seconds: float = 0.0
    headless: bool = False


@dataclass(frozen=True)
class Ros2Config:
    debug_frame_topic: str = ""
    frame_id: str = "world"
    timeout_s: float = 10.0
    object_id: str = ""


@dataclass(frozen=True)
class RealExecutionConfig:
    enabled: bool = False
    grasp_id: str = ""
    attempt_artifact: str = "artifacts/real_robot_pick_attempt.json"
    planning_group: str = "fr3_arm"
    pose_link: str = "fr3_hand_tcp"
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
    gripper_enabled: bool = False
    gripper_grasp_action: str = "/fr3_gripper/grasp"
    gripper_move_action: str = "/fr3_gripper/move"
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


_STAGE1_CACHE_SCHEMA_VERSION = 3


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
    }


def _saved_candidate_from_cache_payload(item: dict[str, object]) -> SavedGraspCandidate:
    contact_patch_offset_local = item.get("contact_patch_offset_local", [0.0, 0.0])
    return SavedGraspCandidate(
        grasp_id=str(item["grasp_id"]),
        grasp_position_obj=tuple(float(v) for v in item["grasp_pose_obj"]["position"]),  # type: ignore[index]
        grasp_orientation_xyzw_obj=tuple(
            float(v) for v in item["grasp_pose_obj"]["orientation_xyzw"]  # type: ignore[index]
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
                "source_frame_orientation_xyzw_obj_world": list(
                    result.bundle.source_frame_orientation_xyzw_obj_world
                ),
            },
            "metadata": result.bundle.metadata,
            "candidates": [_saved_candidate_to_cache_payload(candidate) for candidate in result.bundle.candidates],
        },
        "raw_candidates": [_saved_candidate_to_cache_payload(candidate) for candidate in result.raw_candidates],
        "surface_samples": [_surface_sample_to_cache_payload(sample) for sample in result.surface_samples],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    all_upright_approach_axes_obj = _stage1_upright_approach_axes(
        source_frame_pose_obj_world=target_pose_in_obj_world,
        extra_axes_obj=upright_approach_axes_obj,
    )

    cache_path = None
    cache_key = ""
    cache_key_payload: dict[str, object] = {}
    if planning.stage1_cache_enabled:
        cache_path, cache_key, cache_key_payload = _stage1_cache_path(
            geometry=geometry,
            planning=planning,
            source_frame_pose_obj_world=source_frame_pose_obj_world,
            upright_approach_axes_obj=all_upright_approach_axes_obj,
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
                return cached

    generator = AntipodalMeshGraspGenerator(
        planning.to_generator_config(upright_approach_axes_obj=all_upright_approach_axes_obj)
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
            "stage1_collision_checks_skipped": planning.skip_stage1_collision_checks,
            "stage1_cache_enabled": planning.stage1_cache_enabled,
            "stage1_cache_hit": False,
            "stage1_cache_path": None if cache_path is None else str(cache_path),
            "stage1_cache_key": cache_key or None,
            "num_surface_samples": planning.num_surface_samples,
            "surface_sample_count": len(surface_samples),
            "raw_candidate_count": len(serialized_raw),
            "assembly_feasible_count": len(kept_candidates),
            "scored_feasible_count": len(kept_candidates),
            "assembly_obstacle_paths": list(obstacle_paths),
            "contact_lateral_offsets_m": list(planning.contact_lateral_offsets_m),
            "contact_approach_offsets_m": list(planning.contact_approach_offsets_m),
            "upright_approach_axes_obj": [
                [float(value) for value in axis] for axis in all_upright_approach_axes_obj
            ],
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
    return result


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
    )


def _score_grasps_for_world_top_approach(
    grasps: list[SavedGraspCandidate],
    *,
    mesh_local,
    object_pose_world: ObjectWorldPose,
    top_grasp_score_weight: float,
) -> list[SavedGraspCandidate]:
    object_scored = score_grasps(grasps, mesh_local=mesh_local)
    weight = min(1.0, max(0.0, float(top_grasp_score_weight)))
    if weight <= 0.0:
        return object_scored

    world_scored: list[SavedGraspCandidate] = []
    for grasp in object_scored:
        grasp_rot_obj = quat_to_rotmat_xyzw(grasp.grasp_orientation_xyzw_obj)
        approach_axis_world = object_pose_world.rotation_world_from_object @ grasp_rot_obj[:, 2]
        top_down_score = min(1.0, max(0.0, float(-approach_axis_world[2])))
        object_score = 0.0 if grasp.score is None else float(grasp.score)
        combined_score = (1.0 - weight) * object_score + weight * top_down_score
        score_components = dict(grasp.score_components or {})
        score_components["object_score"] = object_score
        score_components["top_down_approach"] = top_down_score
        score_components["world_approach_z"] = float(approach_axis_world[2])
        score_components["top_grasp_score_weight"] = weight
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
    statuses = evaluate_saved_grasps_against_pickup_pose(
        bundle.candidates,
        object_pose_world=pickup_pose_world,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
        floor_clearance_margin_m=planning.floor_clearance_margin_m,
        contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
        contact_approach_offsets_m=planning.contact_approach_offsets_m,
    )
    accepted = _score_grasps_for_world_top_approach(
        accepted_grasps(statuses),
        mesh_local=mesh_local,
        object_pose_world=pickup_pose_world,
        top_grasp_score_weight=planning.top_grasp_score_weight,
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
    metadata = dict(bundle.metadata)
    metadata.update(
        {
            "pickup_support_face": None if pickup_spec is None else pickup_spec.support_face,
            "pickup_yaw_deg": None if pickup_spec is None else float(pickup_spec.yaw_deg),
            "pickup_xy_world": None if pickup_spec is None else list(pickup_spec.xy_world),
            "execution_world_pose": {
                "position_world": list(pickup_pose_world.position_world),
                "orientation_xyzw_world": list(pickup_pose_world.orientation_xyzw_world),
            },
            "ground_input_count": len(bundle.candidates),
            "ground_feasible_count": len(accepted),
            "top_grasp_score_weight": planning.top_grasp_score_weight,
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
        source_bundle=bundle,
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
            f"input_grasps:     {len(result.source_bundle.candidates)}",
            f"ground_feasible:  {len(result.accepted)}",
            f"support_face:     {result.pickup_spec.support_face if result.pickup_spec is not None else 'explicit_pose'}",
            f"pickup_yaw_deg:   {float(result.pickup_spec.yaw_deg):.1f}"
            if result.pickup_spec is not None
            else "pickup_yaw_deg:   n/a",
            f"contact_offsets_x:{tuple(planning.contact_lateral_offsets_m)}",
            f"contact_offsets_z:{tuple(planning.contact_approach_offsets_m)}",
            f"floor_clearance: {planning.floor_clearance_margin_m:.6f} m",
            f"top_score_weight: {planning.top_grasp_score_weight:.3f}",
            f"pickup_pos_w:     {tuple(round(v, 6) for v in result.pickup_pose_world.position_world)}",
        ],
    )
