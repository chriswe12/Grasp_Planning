#!/usr/bin/env python3
"""Batch benchmark grasp generation over Fabrica OBJ parts and stable orientations."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.collision import (  # noqa: E402
    BoxCollisionPrimitive,
    FrankaHandFingerCollisionModel,
    GraspCollisionEvaluator,
    MeshCollisionPrimitive,
    trimesh_fcl_backend_available,
)
from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    CandidateStatus,
    SavedGraspCandidate,
    candidate_payload,
    canonicalize_target_mesh,
    evaluate_saved_grasps_against_pickup_pose,
    ground_plane_overlay_obj,
    linear_sweep_triangle_mesh,
    load_asset_mesh,
    quat_to_rotmat_xyzw,
    relative_asset_mesh_path,
    transform_primitive_to_world,
    unique_edges,
    write_debug_html,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh  # noqa: E402
from grasp_planning.grasping.mesh_io import resolve_mesh_path  # noqa: E402
from grasp_planning.pipeline import (  # noqa: E402
    GeometryConfig,
    HandoverFallbackResult,
    HandoverGraspPair,
    PlanningConfig,
    generate_stage1_result,
    plan_handover_fallback,
    plan_mujoco_regrasp_fallback,
    recheck_stage2_result,
    write_handover_fallback_result,
    write_mujoco_regrasp_debug_html,
    write_mujoco_regrasp_plan,
    write_stage1_artifacts,
    write_stage2_artifacts,
)
from grasp_planning.pipeline.stable_orientations import (  # noqa: E402
    StableOrientation,
    StableOrientationConfig,
    enumerate_stable_orientations,
    stable_orientation_payload,
    stable_orientation_result_payload,
)
from scripts.run_grasp_pipeline import _planning_config  # noqa: E402
from scripts.write_part_frame_debug_html import write_part_frame_debug_html  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "grasp_generation_benchmark.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "grasp_generation_benchmark"
SUCCESS_STATUSES = {"direct_success", "fallback_success", "handover_fallback_success"}
FAILED_GRASP_HTML_LIMIT = 100
FAILED_GRASP_STAGE1_PASS_EXAMPLE_LIMIT = 20
FAILED_GRASP_DISPLAY_FRAME = "stage2_floor_pose"
ALL_GRASP_OVERVIEW_MAX_MESH_EDGES = 5000
ALL_GRASP_OVERVIEW_MAX_OBSTACLE_EDGES = 5000
ALL_GRASP_OVERVIEW_MARKER_LENGTH_M = 0.025
HANDOVER_PAIR_HTML_LIMIT = 100
HANDOVER_PAIR_ACCEPTED_HTML_LIMIT = 30


@dataclass(frozen=True)
class TargetSpec:
    assembly: str
    part_id: str
    target_mesh_path: str
    assembly_glob: str
    assembly_obstacle_paths: tuple[str, ...] | None = None
    precedence_plan_path: str = ""
    selected_assembly_order: tuple[str, ...] = ()
    already_assembled_part_ids: tuple[str, ...] = ()
    pre_insertion_poses_path: str = ""
    pre_insertion_role: str = ""
    insertion_sweep_vector_m: tuple[float, float, float] | None = None
    insertion_sweep_distance_m: float = 0.0
    final_to_pre_insertion_translation_m: tuple[float, float, float] | None = None


def _precedence_target_state(*, assembly_dir: Path, part_id: str) -> dict[str, object]:
    plan_path = assembly_dir / "precedence_plan.json"
    if not plan_path.is_file():
        return {
            "assembly_obstacle_paths": None,
            "precedence_plan_path": "",
            "selected_assembly_order": (),
            "already_assembled_part_ids": (),
        }
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    orders = payload.get("forward_assembly_orders")
    if not isinstance(orders, list) or not orders:
        raise ValueError(f"Precedence plan '{plan_path}' does not contain forward_assembly_orders.")
    selected_order = tuple(str(item) for item in orders[0])
    if part_id not in selected_order:
        raise ValueError(f"Part '{part_id}' is not present in selected precedence order from '{plan_path}'.")
    part_index = selected_order.index(part_id)
    already_assembled = selected_order[:part_index]
    obstacle_paths = []
    for obstacle_part_id in already_assembled:
        obstacle_path = assembly_dir / f"{obstacle_part_id}.obj"
        if not obstacle_path.is_file():
            raise FileNotFoundError(
                f"Precedence plan '{plan_path}' references missing obstacle part '{obstacle_path}'."
            )
        obstacle_paths.append(relative_asset_mesh_path(obstacle_path))
    return {
        "assembly_obstacle_paths": tuple(obstacle_paths),
        "precedence_plan_path": relative_asset_mesh_path(plan_path),
        "selected_assembly_order": selected_order,
        "already_assembled_part_ids": already_assembled,
    }


def _target_assembly_obstacle_metadata(target: TargetSpec) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if target.precedence_plan_path:
        metadata.update(
            {
                "precedence_plan_path": target.precedence_plan_path,
                "selected_assembly_order": list(target.selected_assembly_order),
                "already_assembled_part_ids": list(target.already_assembled_part_ids),
                "current_part_index": len(target.already_assembled_part_ids),
            }
        )
    if target.pre_insertion_poses_path:
        metadata.update(
            {
                "pre_insertion_poses_path": target.pre_insertion_poses_path,
                "pre_insertion_role": target.pre_insertion_role,
                "insertion_sweep_vector_m": None
                if target.insertion_sweep_vector_m is None
                else list(target.insertion_sweep_vector_m),
                "insertion_sweep_distance_m": target.insertion_sweep_distance_m,
                "final_to_pre_insertion_translation_m": None
                if target.final_to_pre_insertion_translation_m is None
                else list(target.final_to_pre_insertion_translation_m),
            }
        )
    return metadata


def _pre_insertion_target_state(*, assembly_dir: Path, part_id: str) -> dict[str, object]:
    poses_path = assembly_dir / "pre_insertion_poses.json"
    if not poses_path.is_file():
        return {
            "pre_insertion_poses_path": "",
            "pre_insertion_role": "",
            "insertion_sweep_vector_m": None,
            "insertion_sweep_distance_m": 0.0,
            "final_to_pre_insertion_translation_m": None,
        }
    payload = json.loads(poses_path.read_text(encoding="utf-8"))
    parts = payload.get("parts")
    if not isinstance(parts, dict):
        raise ValueError(f"Pre-insertion poses file '{poses_path}' does not contain a parts mapping.")
    if part_id not in parts:
        raise ValueError(f"Part '{part_id}' is not present in pre-insertion poses file '{poses_path}'.")
    part_payload = dict(parts[part_id])
    transform = np.asarray(part_payload.get("final_to_pre_insertion_transform_m"), dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"Part '{part_id}' in '{poses_path}' has an invalid final_to_pre_insertion_transform_m.")
    if not np.allclose(transform[:3, :3], np.eye(3), atol=1.0e-9):
        raise ValueError(
            f"Part '{part_id}' in '{poses_path}' has a rotational pre-insertion transform; "
            "linear swept obstacles currently support translation-only insertions."
        )
    translation = tuple(float(value) for value in transform[:3, 3])
    raw_vector = part_payload.get("pre_to_final_insertion_vector_m")
    if raw_vector is None:
        sweep_vector = None
        distance_m = 0.0
    else:
        vector_array = np.asarray(raw_vector, dtype=float)
        if vector_array.shape != (3,):
            raise ValueError(f"Part '{part_id}' in '{poses_path}' has an invalid pre_to_final_insertion_vector_m.")
        if not np.allclose(vector_array + np.asarray(translation, dtype=float), np.zeros(3), atol=1.0e-6):
            raise ValueError(
                f"Part '{part_id}' in '{poses_path}' has inconsistent final_to_pre and pre_to_final vectors."
            )
        distance_m = float(np.linalg.norm(vector_array))
        sweep_vector = None if distance_m < 1.0e-12 else tuple(float(value) for value in vector_array)
    return {
        "pre_insertion_poses_path": relative_asset_mesh_path(poses_path),
        "pre_insertion_role": str(part_payload.get("role", "")),
        "insertion_sweep_vector_m": sweep_vector,
        "insertion_sweep_distance_m": distance_m,
        "final_to_pre_insertion_translation_m": translation,
    }


@dataclass(frozen=True)
class FallbackBenchmarkConfig:
    enabled: bool = True
    yaw_angles_deg: tuple[float, ...] = (0.0,)
    max_orientations: int = 24
    max_placement_options: int = 18
    min_facet_area_m2: float = 1.0e-6
    stability_margin_m: float = 0.0
    coplanar_tolerance_m: float = 1.0e-6
    staging_xy_offsets_m: tuple[tuple[float, float], ...] = ((0.0, 0.0),)


@dataclass(frozen=True)
class HandoverFallbackBenchmarkConfig:
    enabled: bool = True
    max_final_candidates: int = 40
    max_transfer_candidates: int = 80
    max_pair_checks: int = 1000
    max_accepted_pairs: int = 24
    max_rejected_pairs: int = 100
    transfer_floor_clearance_margin_m: float = 0.0


@dataclass(frozen=True)
class AssemblyObstaclePart:
    part_id: str
    mesh_path: str
    mesh_world: TriangleMesh
    scene: object


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in '{path}'.")
    return payload


def _tuple_floats(values: object, *, expected_len: int | None = None) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a list/tuple of floats, got {values!r}.")
    result = tuple(float(value) for value in values)
    if expected_len is not None and len(result) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(result)}.")
    return result


def _tuple_float_pairs(values: object) -> tuple[tuple[float, float], ...]:
    if values in ("", None):
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a list/tuple of [x, y] pairs, got {values!r}.")
    return tuple(_tuple_floats(value, expected_len=2) for value in values)  # type: ignore[arg-type]


def _fallback_config(payload: dict[str, object]) -> FallbackBenchmarkConfig:
    raw = dict(payload.get("fallback", {}))
    return FallbackBenchmarkConfig(
        enabled=bool(raw.get("enabled", True)),
        yaw_angles_deg=_tuple_floats(raw.get("yaw_angles_deg", [0.0])),
        max_orientations=int(raw.get("max_orientations", 24)),
        max_placement_options=int(raw.get("max_placement_options", 18)),
        min_facet_area_m2=float(raw.get("min_facet_area_m2", 1.0e-6)),
        stability_margin_m=float(raw.get("stability_margin_m", 0.0)),
        coplanar_tolerance_m=float(raw.get("coplanar_tolerance_m", 1.0e-6)),
        staging_xy_offsets_m=_tuple_float_pairs(raw.get("staging_xy_offsets_m", [[0.0, 0.0]])) or ((0.0, 0.0),),
    )


def _handover_fallback_config(payload: dict[str, object]) -> HandoverFallbackBenchmarkConfig:
    raw = dict(payload.get("handover_fallback", {}))
    return HandoverFallbackBenchmarkConfig(
        enabled=bool(raw.get("enabled", True)),
        max_final_candidates=int(raw.get("max_final_candidates", 40)),
        max_transfer_candidates=int(raw.get("max_transfer_candidates", 80)),
        max_pair_checks=int(raw.get("max_pair_checks", 1000)),
        max_accepted_pairs=int(raw.get("max_accepted_pairs", 24)),
        max_rejected_pairs=int(raw.get("max_rejected_pairs", 100)),
        transfer_floor_clearance_margin_m=float(raw.get("transfer_floor_clearance_margin_m", 0.0)),
    )


def _stable_orientation_config(payload: dict[str, object]) -> StableOrientationConfig:
    raw = dict(payload.get("stable_orientations", {}))
    return StableOrientationConfig(
        robust_tilt_deg=float(raw.get("robust_tilt_deg", 5.0)),
        min_support_area_m2=float(raw.get("min_support_area_m2", 1.0e-6)),
        min_support_area_fraction=float(raw.get("min_support_area_fraction", 0.01)),
        coplanar_tolerance_m=float(raw.get("coplanar_tolerance_m", 1.0e-6)),
        xy_world=_tuple_floats(raw.get("xy_world", [0.0, 0.0]), expected_len=2),  # type: ignore[arg-type]
    )


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value)) or "item"


def _candidate_payload(candidate: SavedGraspCandidate) -> dict[str, object]:
    return {
        "grasp_id": candidate.grasp_id,
        "grasp_pose_obj": {
            "position": list(candidate.grasp_position_obj),
            "orientation_xyzw": list(candidate.grasp_orientation_xyzw_obj),
        },
        "contact_points_obj": [list(candidate.contact_point_a_obj), list(candidate.contact_point_b_obj)],
        "contact_normals_obj": [list(candidate.contact_normal_a_obj), list(candidate.contact_normal_b_obj)],
        "jaw_width": candidate.jaw_width,
        "roll_angle_rad": candidate.roll_angle_rad,
        "contact_patch_offset_local": [
            candidate.contact_patch_lateral_offset_m,
            candidate.contact_patch_approach_offset_m,
        ],
        "score": candidate.score,
        "score_components": candidate.score_components,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _git_metadata() -> dict[str, object]:
    def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    commit = _run_git(["rev-parse", "--short", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "",
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.name == "stage1_cache":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _target_spec_from_asset_path(path: Path, *, target_root: str) -> TargetSpec:
    resolved = resolve_mesh_path(path)
    assets_relative = Path(relative_asset_mesh_path(resolved))
    root_relative = Path(target_root)
    try:
        rel = assets_relative.relative_to(root_relative)
    except ValueError as exc:
        raise ValueError(f"Target '{assets_relative}' is not under target_root '{target_root}'.") from exc
    if len(rel.parts) < 2:
        raise ValueError(f"Target '{assets_relative}' must be under an assembly directory.")
    assembly = rel.parts[0]
    part_id = rel.stem
    precedence_state = _precedence_target_state(assembly_dir=resolved.parent, part_id=part_id)
    pre_insertion_state = _pre_insertion_target_state(assembly_dir=resolved.parent, part_id=part_id)
    return TargetSpec(
        assembly=assembly,
        part_id=part_id,
        target_mesh_path=assets_relative.as_posix(),
        assembly_glob=(root_relative / assembly / "*.obj").as_posix(),
        assembly_obstacle_paths=precedence_state["assembly_obstacle_paths"],  # type: ignore[arg-type]
        precedence_plan_path=str(precedence_state["precedence_plan_path"]),
        selected_assembly_order=precedence_state["selected_assembly_order"],  # type: ignore[arg-type]
        already_assembled_part_ids=precedence_state["already_assembled_part_ids"],  # type: ignore[arg-type]
        pre_insertion_poses_path=str(pre_insertion_state["pre_insertion_poses_path"]),
        pre_insertion_role=str(pre_insertion_state["pre_insertion_role"]),
        insertion_sweep_vector_m=pre_insertion_state["insertion_sweep_vector_m"],  # type: ignore[arg-type]
        insertion_sweep_distance_m=float(pre_insertion_state["insertion_sweep_distance_m"]),
        final_to_pre_insertion_translation_m=pre_insertion_state[  # type: ignore[arg-type]
            "final_to_pre_insertion_translation_m"
        ],
    )


def _discover_targets(payload: dict[str, object], args: argparse.Namespace) -> list[TargetSpec]:
    target_root = str(payload.get("target_root", "obj/fabrica")).strip() or "obj/fabrica"
    root_path = resolve_mesh_path(target_root)
    if args.target:
        raw_paths = [Path(value) for value in args.target]
        specs = [_target_spec_from_asset_path(path, target_root=target_root) for path in raw_paths]
    else:
        specs = [
            _target_spec_from_asset_path(path, target_root=target_root)
            for path in sorted(root_path.glob("*/*.obj"))
            if path.is_file()
        ]
    if args.assembly:
        allowed = {str(value) for value in args.assembly}
        specs = [spec for spec in specs if spec.assembly in allowed]
    if args.part:
        allowed_parts = {Path(str(value)).stem for value in args.part}
        specs = [spec for spec in specs if spec.part_id in allowed_parts]
    specs = sorted(specs, key=lambda spec: (spec.assembly, spec.part_id, spec.target_mesh_path))
    if args.limit_parts is not None:
        specs = specs[: max(0, int(args.limit_parts))]
    return specs


def _reason_counts(statuses: Iterable[CandidateStatus]) -> dict[str, int]:
    return dict(Counter(entry.reason for entry in statuses))


def _best_candidate_payload(candidates: Iterable[SavedGraspCandidate]) -> dict[str, object] | None:
    candidates_tuple = tuple(candidates)
    if not candidates_tuple:
        return None
    best = candidates_tuple[0]
    return {
        "grasp_id": best.grasp_id,
        "score": best.score,
        "contact_patch_offset_local": [
            best.contact_patch_lateral_offset_m,
            best.contact_patch_approach_offset_m,
        ],
    }


def _candidate_score(candidate: SavedGraspCandidate) -> float:
    return float(candidate.score) if candidate.score is not None else float("-inf")


def _status_sort_key(entry: CandidateStatus) -> tuple[float, str]:
    return (-_candidate_score(entry.grasp), entry.grasp.grasp_id)


def _candidate_contact_key(candidate: SavedGraspCandidate, *, tolerance_m: float = 1.0e-5) -> tuple[object, ...]:
    def _point_key(point: tuple[float, float, float]) -> tuple[int, int, int]:
        return tuple(int(round(float(value) / tolerance_m)) for value in point)

    contact_pair = tuple(
        sorted(
            (
                _point_key(candidate.contact_point_a_obj),
                _point_key(candidate.contact_point_b_obj),
            )
        )
    )
    jaw_width_key = int(round(float(candidate.jaw_width) / tolerance_m))
    return contact_pair, jaw_width_key


def _unique_contact_statuses(
    statuses: Iterable[CandidateStatus],
    *,
    limit: int = FAILED_GRASP_HTML_LIMIT,
) -> list[CandidateStatus]:
    selected: list[CandidateStatus] = []
    seen: set[tuple[object, ...]] = set()
    for entry in sorted(statuses, key=_status_sort_key):
        key = _candidate_contact_key(entry.grasp)
        if key in seen:
            continue
        seen.add(key)
        selected.append(entry)
        if len(selected) >= max(0, int(limit)):
            break
    return selected


def _top_rejected_statuses(
    statuses: Iterable[CandidateStatus],
    *,
    limit: int = FAILED_GRASP_HTML_LIMIT,
) -> list[CandidateStatus]:
    rejected = [entry for entry in statuses if entry.status != "accepted"]
    return _unique_contact_statuses(rejected, limit=limit)


def _stage1_passed_stage2_failure_statuses(
    stage2,
    *,
    limit: int = FAILED_GRASP_STAGE1_PASS_EXAMPLE_LIMIT,
) -> list[CandidateStatus]:
    if stage2 is None:
        return []
    return _unique_contact_statuses(
        [
            CandidateStatus(
                grasp=entry.grasp,
                status="stage1_pass",
                reason=f"floor: {entry.reason} (stage1 passed)",
            )
            for entry in stage2.statuses
            if entry.status != "accepted"
        ],
        limit=limit,
    )


def _mesh_in_object_frame(mesh_world: TriangleMesh, object_pose_world) -> TriangleMesh:
    rotation = object_pose_world.rotation_world_from_object
    translation = object_pose_world.translation_world
    vertices_obj = (np.asarray(mesh_world.vertices_obj, dtype=float) - translation) @ rotation
    return TriangleMesh(vertices_obj=vertices_obj, faces=np.asarray(mesh_world.faces, dtype=np.int64))


def _minus_z_axis_in_object_frame(object_pose_world) -> tuple[float, float, float]:
    axis_obj = object_pose_world.rotation_world_from_object.T @ np.array([0.0, 0.0, -1.0], dtype=float)
    norm = float(np.linalg.norm(axis_obj))
    if norm < 1.0e-12:
        return (0.0, 0.0, -1.0)
    return tuple(float(value) for value in (axis_obj / norm).tolist())


def _unique_axes(axes: Iterable[tuple[float, float, float]], *, tolerance: float = 1.0e-8) -> tuple[tuple[float, float, float], ...]:
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


def _upright_approach_axes_obj(
    *,
    source_frame_pose_obj_world,
    orientations: Iterable[StableOrientation],
) -> tuple[tuple[float, float, float], ...]:
    return _unique_axes(
        (
            (0.0, 0.0, -1.0),
            _minus_z_axis_in_object_frame(source_frame_pose_obj_world),
            *(_minus_z_axis_in_object_frame(orientation.object_pose_world) for orientation in orientations),
        )
    )


def _assembly_obstacle_parts(
    *,
    target: TargetSpec,
    mesh_scale: float,
    contact_gap_m: float,
) -> list[AssemblyObstaclePart]:
    target_resolved = resolve_mesh_path(target.target_mesh_path)
    evaluator = GraspCollisionEvaluator(FrankaHandFingerCollisionModel(contact_gap_m=contact_gap_m))
    parts: list[AssemblyObstaclePart] = []
    if target.assembly_obstacle_paths is None:
        obstacle_paths = sorted(REPO_ROOT.joinpath("assets").glob(target.assembly_glob))
    else:
        obstacle_paths = [resolve_mesh_path(path) for path in target.assembly_obstacle_paths]
    for path in obstacle_paths:
        resolved = resolve_mesh_path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Assembly obstacle mesh not found at '{resolved}'.")
        if resolved == target_resolved:
            continue
        mesh_world = load_asset_mesh(resolved, scale=mesh_scale)
        if target.insertion_sweep_vector_m is not None:
            mesh_world = linear_sweep_triangle_mesh(mesh_world, target.insertion_sweep_vector_m)
        parts.append(
            AssemblyObstaclePart(
                part_id=resolved.stem,
                mesh_path=relative_asset_mesh_path(resolved),
                mesh_world=mesh_world,
                scene=evaluator.build_scene(mesh_world),
            )
        )
    return parts


def _candidate_collides_with_scene(
    candidate: SavedGraspCandidate,
    *,
    object_pose_world,
    scene: object,
    contact_gap_m: float,
    lateral_offset_m: float,
    approach_offset_m: float,
) -> bool:
    candidate_obj = candidate.to_object_frame_candidate()
    grasp_rotmat = quat_to_rotmat_xyzw(candidate_obj.grasp_orientation_xyzw_obj)
    collision_model = FrankaHandFingerCollisionModel(
        contact_gap_m=contact_gap_m,
        contact_patch_lateral_offset_m=lateral_offset_m,
        contact_patch_approach_offset_m=approach_offset_m,
    )
    for primitive_obj in collision_model.primitives_for_grasp(
        grasp_rotmat=grasp_rotmat,
        contact_point_a=np.asarray(candidate_obj.contact_point_a_obj, dtype=float),
        contact_point_b=np.asarray(candidate_obj.contact_point_b_obj, dtype=float),
    ):
        primitive_world = transform_primitive_to_world(primitive_obj, object_pose_world)
        if isinstance(primitive_world, BoxCollisionPrimitive) and scene.intersects_box(primitive_world):
            return True
        if isinstance(primitive_world, MeshCollisionPrimitive) and scene.intersects_mesh(primitive_world):
            return True
    return False


def _colliding_obstacle_parts(
    candidate: SavedGraspCandidate,
    *,
    object_pose_world,
    obstacle_parts: list[AssemblyObstaclePart],
    contact_gap_m: float,
    lateral_offset_m: float,
    approach_offset_m: float,
) -> list[str]:
    return [
        part.part_id
        for part in obstacle_parts
        if _candidate_collides_with_scene(
            candidate,
            object_pose_world=object_pose_world,
            scene=part.scene,
            contact_gap_m=contact_gap_m,
            lateral_offset_m=lateral_offset_m,
            approach_offset_m=approach_offset_m,
        )
    ]


def _stage1_assembly_failure_statuses(
    *,
    target: TargetSpec,
    stage1,
    planning: PlanningConfig,
    mesh_scale: float,
    limit: int = FAILED_GRASP_HTML_LIMIT,
    max_candidates_to_scan: int | None = None,
) -> tuple[list[CandidateStatus], list[AssemblyObstaclePart]]:
    accepted_ids = {candidate.grasp_id for candidate in stage1.bundle.candidates}
    raw_candidates = sorted(
        (candidate for candidate in stage1.raw_candidates if candidate.grasp_id not in accepted_ids),
        key=lambda candidate: (-_candidate_score(candidate), candidate.grasp_id),
    )
    scan_limit = max_candidates_to_scan if max_candidates_to_scan is not None else max(int(limit) * 8, int(limit))
    raw_candidates = raw_candidates[: max(0, int(scan_limit))]
    if not raw_candidates:
        return [], []
    if planning.skip_stage1_collision_checks:
        return [], []

    obstacle_parts = _assembly_obstacle_parts(
        target=target,
        mesh_scale=mesh_scale,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
    )
    statuses: list[CandidateStatus] = []
    seen_contact_keys: set[tuple[object, ...]] = set()
    for candidate in raw_candidates:
        collisions = _colliding_obstacle_parts(
            candidate,
            object_pose_world=stage1.target_pose_in_obj_world,
            obstacle_parts=obstacle_parts,
            contact_gap_m=planning.detailed_finger_contact_gap_m,
            lateral_offset_m=float(candidate.contact_patch_lateral_offset_m),
            approach_offset_m=float(candidate.contact_patch_approach_offset_m),
        )
        if collisions:
            shown = ", ".join(collisions[:4])
            suffix = "" if len(collisions) <= 4 else f", +{len(collisions) - 4} more"
            constraint_kind = "part_sweep" if target.insertion_sweep_vector_m is not None else "part"
            reason = f"{constraint_kind}: {shown}{suffix}"
        else:
            constraint_kind = "part_sweep" if target.insertion_sweep_vector_m is not None else "part"
            reason = f"{constraint_kind}: no_collision_at_saved_offset"

        contact_key = _candidate_contact_key(candidate)
        if contact_key in seen_contact_keys:
            continue
        seen_contact_keys.add(contact_key)
        statuses.append(CandidateStatus(grasp=candidate, status="rejected", reason=reason))
        if len(statuses) >= max(0, int(limit)):
            break
    return statuses, obstacle_parts


def _constraint_counts(statuses: Iterable[CandidateStatus]) -> dict[str, int]:
    return dict(Counter(entry.reason for entry in statuses))


def _raw_stage1_payload(stage1, target: TargetSpec) -> dict[str, object]:
    return {
        "target_mesh_path": target.target_mesh_path,
        "assembly_glob": target.assembly_glob,
        "assembly_obstacle_paths": None
        if target.assembly_obstacle_paths is None
        else list(target.assembly_obstacle_paths),
        "precedence_plan_path": target.precedence_plan_path,
        "selected_assembly_order": list(target.selected_assembly_order),
        "already_assembled_part_ids": list(target.already_assembled_part_ids),
        "pre_insertion_poses_path": target.pre_insertion_poses_path,
        "pre_insertion_role": target.pre_insertion_role,
        "insertion_sweep_vector_m": None
        if target.insertion_sweep_vector_m is None
        else list(target.insertion_sweep_vector_m),
        "insertion_sweep_distance_m": target.insertion_sweep_distance_m,
        "final_to_pre_insertion_translation_m": None
        if target.final_to_pre_insertion_translation_m is None
        else list(target.final_to_pre_insertion_translation_m),
        "raw_candidate_count": stage1.raw_candidate_count,
        "scored_raw_candidate_count": len(stage1.raw_candidates),
        "candidates": [_candidate_payload(candidate) for candidate in stage1.raw_candidates],
    }


def _status_for_orientation(
    *,
    stage1_count: int,
    stage2_count: int,
    fallback_found: bool,
    handover_found: bool,
    fallback_enabled: bool,
    handover_enabled: bool,
) -> str:
    if stage1_count <= 0:
        return "stage1_failed"
    if stage2_count > 0:
        return "direct_success"
    if fallback_found:
        return "fallback_success"
    if handover_found:
        return "handover_fallback_success"
    if fallback_enabled or handover_enabled:
        return "stage2_failed_fallback_failed"
    return "stage2_failed_no_fallback"


def _fallback_summary(plan) -> dict[str, object] | None:
    if plan is None:
        return None
    return {
        "transfer_grasp_id": plan.transfer_grasp.grasp_id,
        "transfer_grasp_score": plan.transfer_grasp.score,
        "final_grasp_id": plan.final_grasp.grasp_id,
        "final_grasp_score": plan.final_grasp.score,
        "placement_option_count": len(plan.placement_options),
        "selected_support_stability_margin_m": plan.support_facet.stability_margin_m,
        "metadata": plan.metadata,
    }


def _handover_summary(result: HandoverFallbackResult | None) -> dict[str, object] | None:
    if result is None or result.selected_pair is None:
        return None
    selected = result.selected_pair
    return {
        "transfer_grasp_id": selected.transfer_grasp.grasp_id,
        "transfer_grasp_score": selected.transfer_grasp.score,
        "final_grasp_id": selected.final_grasp.grasp_id,
        "final_grasp_score": selected.final_grasp.score,
        "accepted_pair_count": len(result.accepted_pairs),
        "rejected_pair_count_displayed": len(result.rejected_pairs),
        "metadata": result.metadata,
    }


def _orientation_row(
    *,
    target: TargetSpec,
    orientation: StableOrientation,
    status: str,
    stage1_raw_count: int,
    stage1_count: int,
    stage2_count: int,
    best_direct: dict[str, object] | None,
    fallback_summary: dict[str, object] | None,
    handover_summary: dict[str, object] | None = None,
    error: str = "",
) -> dict[str, object]:
    return {
        "assembly": target.assembly,
        "part_id": target.part_id,
        "target_mesh_path": target.target_mesh_path,
        "orientation_id": orientation.orientation_id,
        "status": status,
        "com_method": orientation.com_method,
        "support_area_m2": orientation.area_m2,
        "stability_margin_m": orientation.stability_margin_m,
        "com_height_m": orientation.com_height_m,
        "max_stable_tilt_deg": orientation.max_stable_tilt_deg,
        "stage1_raw_count": stage1_raw_count,
        "stage1_assembly_feasible_count": stage1_count,
        "stage2_ground_feasible_count": stage2_count,
        "best_direct_grasp_id": "" if best_direct is None else str(best_direct.get("grasp_id", "")),
        "best_direct_score": "" if best_direct is None else best_direct.get("score", ""),
        "fallback_found": fallback_summary is not None,
        "fallback_transfer_grasp_id": "" if fallback_summary is None else fallback_summary["transfer_grasp_id"],
        "fallback_final_grasp_id": "" if fallback_summary is None else fallback_summary["final_grasp_id"],
        "fallback_placement_option_count": 0 if fallback_summary is None else fallback_summary["placement_option_count"],
        "handover_fallback_found": handover_summary is not None,
        "handover_transfer_grasp_id": "" if handover_summary is None else handover_summary["transfer_grasp_id"],
        "handover_final_grasp_id": "" if handover_summary is None else handover_summary["final_grasp_id"],
        "handover_accepted_pair_count": 0 if handover_summary is None else handover_summary["accepted_pair_count"],
        "error": error,
    }


def _write_summary_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "assembly",
        "part_id",
        "target_mesh_path",
        "orientation_id",
        "status",
        "com_method",
        "support_area_m2",
        "stability_margin_m",
        "com_height_m",
        "max_stable_tilt_deg",
        "stage1_raw_count",
        "stage1_assembly_feasible_count",
        "stage2_ground_feasible_count",
        "best_direct_grasp_id",
        "best_direct_score",
        "fallback_found",
        "fallback_transfer_grasp_id",
        "fallback_final_grasp_id",
        "fallback_placement_option_count",
        "handover_fallback_found",
        "handover_transfer_grasp_id",
        "handover_final_grasp_id",
        "handover_accepted_pair_count",
        "handover_transfer_floor_clearance_margin_m",
        "error",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_summary_md(output_path: Path, *, rows: list[dict[str, object]], part_records: list[dict[str, object]]) -> None:
    status_counts = Counter(str(row["status"]) for row in rows)
    part_status_counts = Counter(str(part.get("status", "unknown")) for part in part_records)
    direct = status_counts.get("direct_success", 0)
    fallback = status_counts.get("fallback_success", 0)
    handover = status_counts.get("handover_fallback_success", 0)
    total = len(rows)
    generation_success = direct + fallback + handover
    lines = [
        "# Grasp Generation Benchmark Summary",
        "",
        f"- parts: {len(part_records)}",
        f"- orientations: {total}",
        f"- direct successes: {direct}",
        f"- fallback successes: {fallback}",
        f"- handover fallback successes: {handover}",
        f"- generation successes: {generation_success}",
        f"- generation success rate: {generation_success / total:.3f}" if total else "- generation success rate: n/a",
        "",
        "## Orientation Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Part Status Counts", ""])
    for status, count in sorted(part_status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Failed Orientations", ""])
    failed_rows = [row for row in rows if str(row["status"]) not in SUCCESS_STATUSES]
    if not failed_rows:
        lines.append("- none")
    else:
        for row in failed_rows:
            lines.append(
                f"- {row['target_mesh_path']} {row['orientation_id']}: {row['status']} "
                f"(stage1={row['stage1_assembly_feasible_count']}, stage2={row['stage2_ground_feasible_count']})"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relative_link(output_dir: Path, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _write_index_html(
    output_path: Path,
    *,
    output_dir: Path,
    rows: list[dict[str, object]],
    part_records: list[dict[str, object]],
    browser_parts: list[dict[str, object]],
) -> None:
    status_counts = Counter(str(row["status"]) for row in rows)
    part_status_counts = Counter(str(part.get("status", "unknown")) for part in part_records)
    data = {
        "title": "Grasp Generation Benchmark",
        "subtitle": "Select an assembly and part, then step through stable orientations and their benchmark artifacts.",
        "summary": {
            "part_count": len(part_records),
            "orientation_count": len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "part_status_counts": dict(sorted(part_status_counts.items())),
        },
        "parts": browser_parts,
    }
    data_json = json.dumps(data, indent=2)
    document = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Grasp Generation Benchmark Browser</title>
  <style>
    :root {
      --bg: #f6f4ee;
      --panel: #fffdf8;
      --ink: #1f2522;
      --muted: #68716c;
      --line: #d9d4c7;
      --mesh: #2f6f5e;
      --floor: #2563eb;
      --success: #15803d;
      --fallback: #1d4ed8;
      --fail: #b91c1c;
      --franka: #d97706;
      --hand: #8f5a12;
      --contact-a: #c8452d;
      --contact-b: #1f7c60;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    .layout { display: grid; grid-template-columns: 390px minmax(0, 1fr); min-height: 100vh; }
    aside { border-right: 1px solid var(--line); background: var(--panel); padding: 20px; overflow: auto; }
    main { padding: 18px; overflow: auto; }
    h1 { margin: 0 0 8px; font-size: 25px; line-height: 1.15; }
    .subtitle { margin: 0 0 14px; color: var(--muted); font-size: 14px; line-height: 1.45; }
    .summary { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 14px; }
    .metric { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 8px 10px; }
    .metric strong { display: block; font-size: 12px; color: var(--muted); }
    .metric span { display: block; margin-top: 3px; font-family: "IBM Plex Mono", monospace; font-size: 13px; }
    .selectors { display: grid; gap: 8px; margin-bottom: 12px; }
    label { display: grid; gap: 4px; color: var(--muted); font-size: 12px; }
    select { width: 100%; border: 1px solid var(--line); border-radius: 8px; background: #fff; color: var(--ink); padding: 9px 10px; font: inherit; }
    .controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 14px; }
    button { border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 8px; padding: 10px 12px; font: inherit; cursor: pointer; }
    button:hover { border-color: var(--mesh); }
    .orientation-list { display: grid; gap: 8px; }
    .orientation-item { width: 100%; text-align: left; border-radius: 8px; }
    .orientation-item.active { border-color: var(--mesh); box-shadow: 0 0 0 2px rgba(47,111,94,0.14); }
    .item-title { display: flex; justify-content: space-between; gap: 8px; font-weight: 700; }
    .item-meta { margin-top: 5px; color: var(--muted); font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.4; }
    .status { border-radius: 999px; padding: 2px 7px; font-size: 12px; white-space: nowrap; }
    .direct_success { background: #dcfce7; color: var(--success); }
    .fallback_success { background: #dbeafe; color: var(--fallback); }
    .failed { background: #fee2e2; color: var(--fail); }
    .card { border: 1px solid var(--line); background: rgba(255,253,248,0.96); border-radius: 8px; padding: 14px; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.6fr); gap: 16px; align-items: start; }
    #scene { width: 100%; aspect-ratio: 1.45 / 1; display: block; border-radius: 8px; background: linear-gradient(180deg, #ffffff, #ebe7dc); }
    .kv { white-space: pre-wrap; font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.55; margin: 0; }
    .link-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .link-list a { border: 1px solid var(--line); border-radius: 8px; background: #fff; color: #2563eb; padding: 6px 8px; font-size: 13px; text-decoration: none; }
    .link-list a:hover { text-decoration: underline; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; color: var(--muted); font-size: 13px; }
    .legend span { display: inline-flex; align-items: center; gap: 7px; }
    .swatch { width: 13px; height: 13px; border-radius: 999px; display: inline-block; }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1 id="title"></h1>
      <p id="subtitle" class="subtitle"></p>
      <div id="summary" class="summary"></div>
      <div class="selectors">
        <label>Assembly<select id="assemblySelect"></select></label>
        <label>Part<select id="partSelect"></select></label>
      </div>
      <div class="controls">
        <button id="prevPartBtn" type="button">Previous Part</button>
        <button id="nextPartBtn" type="button">Next Part</button>
        <button id="prevOrientationBtn" type="button">Previous Orientation</button>
        <button id="nextOrientationBtn" type="button">Next Orientation</button>
        <button id="solidBtn" type="button">Solid Mesh</button>
        <button id="resetBtn" type="button">Reset View</button>
      </div>
      <div id="orientationList" class="orientation-list"></div>
    </aside>
    <main>
      <div class="grid">
        <section class="card">
          <svg id="scene" viewBox="0 0 1100 760"></svg>
          <div class="legend">
            <span><i class="swatch" style="background: var(--mesh)"></i>Target mesh</span>
            <span><i class="swatch" style="background: var(--floor)"></i>Ground</span>
            <span><i class="swatch" style="background: var(--franka)"></i>Finger boxes</span>
            <span><i class="swatch" style="background: var(--hand)"></i>Hand mesh</span>
            <span><i class="swatch" style="background: var(--contact-a)"></i>Contact A</span>
            <span><i class="swatch" style="background: var(--contact-b)"></i>Contact B</span>
          </div>
        </section>
        <section class="card">
          <pre id="details" class="kv"></pre>
          <div id="linkList" class="link-list"></div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const assemblySelect = document.getElementById("assemblySelect");
    const partSelect = document.getElementById("partSelect");
    const scene = document.getElementById("scene");
    const details = document.getElementById("details");
    const linkList = document.getElementById("linkList");
    const list = document.getElementById("orientationList");
    document.getElementById("title").textContent = data.title;
    document.getElementById("subtitle").textContent = data.subtitle;
    const initialView = { yaw: -0.72, pitch: 0.52, zoom: 1.0, panX: 0, panY: 0 };
    const state = { assembly: "", partKey: "", index: 0, solid: false, dragging: false, dragMode: "rotate", pointerId: null, lastX: 0, lastY: 0, ...initialView };
    function metric(label, value) {
      const node = document.createElement("div");
      node.className = "metric";
      node.innerHTML = `<strong>${label}</strong><span>${value}</span>`;
      return node;
    }
    function renderSummary() {
      const summary = document.getElementById("summary");
      const statusText = Object.entries(data.summary.status_counts).map(([key, value]) => `${key}:${value}`).join(" ");
      summary.replaceChildren(
        metric("Parts", data.summary.part_count),
        metric("Orientations", data.summary.orientation_count),
        metric("Statuses", statusText || "none"),
        metric("Browser Parts", data.parts.length),
      );
    }
    function assemblies() {
      return [...new Set(data.parts.map((part) => part.assembly))].sort();
    }
    function partsForAssembly() {
      return data.parts.filter((part) => part.assembly === state.assembly);
    }
    function currentPartIndex() {
      return partsForAssembly().findIndex((part) => part.key === state.partKey);
    }
    function currentPart() {
      return data.parts.find((part) => part.key === state.partKey) || partsForAssembly()[0] || data.parts[0] || null;
    }
    function currentFrame() {
      const part = currentPart();
      return part ? part.frames[state.index] || null : null;
    }
    function setAssembly(value) {
      state.assembly = value;
      const parts = partsForAssembly();
      state.partKey = parts.length ? parts[0].key : "";
      state.index = 0;
      populatePartSelect();
      render();
    }
    function setPart(value, preserveOrientationIndex = false) {
      state.partKey = value;
      if (!preserveOrientationIndex) {
        state.index = 0;
      }
      render();
    }
    function populateAssemblySelect() {
      assemblySelect.replaceChildren();
      assemblies().forEach((assembly) => {
        const option = document.createElement("option");
        option.value = assembly;
        option.textContent = assembly;
        assemblySelect.appendChild(option);
      });
      state.assembly = assemblySelect.value || assemblies()[0] || "";
    }
    function populatePartSelect() {
      partSelect.replaceChildren();
      partsForAssembly().forEach((part) => {
        const successes = part.frames.filter((frame) => frame.status === "direct_success" || frame.status === "fallback_success" || frame.status === "handover_fallback_success").length;
        const option = document.createElement("option");
        option.value = part.key;
        option.textContent = `${part.part_id} (${successes}/${part.frames.length})`;
        partSelect.appendChild(option);
      });
      if (![...partSelect.options].some((option) => option.value === state.partKey)) {
        state.partKey = partSelect.options.length ? partSelect.options[0].value : "";
      }
      partSelect.value = state.partKey;
    }
    function statusClass(status) {
      if (status === "direct_success") return "direct_success";
      if (status === "fallback_success" || status === "handover_fallback_success") return "fallback_success";
      return "failed";
    }
    function fmt(value, digits = 3) {
      if (value === null || value === undefined || value === "") return "n/a";
      return Number(value).toFixed(digits);
    }
    function allPoints(part, frame) {
      if (!frame) return part ? part.vertices_local : [[0, 0, 0]];
      const points = [...frame.mesh_vertices_world, ...frame.floor_world];
      const grasp = frame.best_grasp;
      if (grasp) {
        points.push(
          grasp.grasp_position_obj,
          grasp.contact_point_a_obj,
          grasp.contact_point_b_obj,
          ...grasp.franka_hand_vertices_obj,
          ...grasp.franka_left_boxes.flatMap((box) => box.corners),
          ...grasp.franka_right_boxes.flatMap((box) => box.corners),
        );
      }
      return points;
    }
    function bounds(points) {
      return points.reduce((acc, point) => {
        point.forEach((value, axis) => {
          acc.min[axis] = Math.min(acc.min[axis], value);
          acc.max[axis] = Math.max(acc.max[axis], value);
        });
        return acc;
      }, { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
    }
    function rotate(point, center) {
      const shifted = point.map((value, axis) => value - center[axis]);
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];
      return [x1, cp * y1 + sp * z1, -sp * y1 + cp * z1];
    }
    function projection(part, frame) {
      const b = bounds(allPoints(part, frame));
      const center = b.min.map((value, axis) => 0.5 * (value + b.max[axis]));
      const extent = Math.max(...b.max.map((value, axis) => value - b.min[axis]), 0.08);
      return { center, scale: 560 / extent };
    }
    function project(point, projectionBounds) {
      const [x, y, z] = rotate(point, projectionBounds.center);
      const scale = projectionBounds.scale * state.zoom;
      return { x: 550 + state.panX + x * scale, y: 380 + state.panY - y * scale, depth: z };
    }
    function add(tag, attrs) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      scene.appendChild(node);
      return node;
    }
    function line(a, b, options, projectionBounds) {
      const pa = project(a, projectionBounds), pb = project(b, projectionBounds);
      add("line", { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, stroke: options.stroke || "#333", "stroke-width": options.width || 2, "stroke-opacity": options.opacity ?? 1, "stroke-dasharray": options.dash || "" });
    }
    function point(p, options, projectionBounds) {
      const pp = project(p, projectionBounds);
      add("circle", { cx: pp.x, cy: pp.y, r: options.radius || 5, fill: options.fill || "#000", stroke: "#fff", "stroke-width": options.strokeWidth || 1.5, "fill-opacity": options.opacity ?? 1 });
    }
    function polygon(points, options, projectionBounds) {
      const projected = points.map((p) => project(p, projectionBounds));
      add("polygon", { points: projected.map((p) => `${p.x},${p.y}`).join(" "), fill: options.fill || "none", "fill-opacity": options.fillOpacity ?? 1, stroke: options.stroke || "none", "stroke-width": options.strokeWidth || 1, "stroke-opacity": options.strokeOpacity ?? 1 });
    }
    function label(p, text, color, projectionBounds) {
      const pp = project(p, projectionBounds);
      const node = add("text", { x: pp.x + 8, y: pp.y - 8, fill: color, "font-size": 14, "font-family": "IBM Plex Mono, monospace", "font-weight": 700 });
      node.textContent = text;
    }
    function drawMesh(part, frame, projectionBounds) {
      if (state.solid) {
        part.faces.map((face) => {
          const points = face.map((index) => frame.mesh_vertices_world[index]);
          const rotated = points.map((p) => rotate(p, projectionBounds.center));
          const edgeA = rotated[1].map((value, axis) => value - rotated[0][axis]);
          const edgeB = rotated[2].map((value, axis) => value - rotated[0][axis]);
          const normal = [
            edgeA[1] * edgeB[2] - edgeA[2] * edgeB[1],
            edgeA[2] * edgeB[0] - edgeA[0] * edgeB[2],
            edgeA[0] * edgeB[1] - edgeA[1] * edgeB[0],
          ];
          const depth = rotated.reduce((sum, p) => sum + p[2], 0) / rotated.length;
          return { points, normal, depth };
        }).filter((face) => face.normal[2] > 0).sort((a, b) => a.depth - b.depth).forEach((face) => {
          polygon(face.points, { fill: "#7aa392", fillOpacity: 0.86, stroke: "#2f6f5e", strokeWidth: 0.8, strokeOpacity: 0.45 }, projectionBounds);
        });
      }
      part.edges.forEach(([a, b]) => line(frame.mesh_vertices_world[a], frame.mesh_vertices_world[b], { stroke: "#2f6f5e", width: 1.5, opacity: 0.82 }, projectionBounds));
    }
    function drawBox(corners, color, projectionBounds) {
      [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([a, b]) => {
        line(corners[a], corners[b], { stroke: color, width: 1.6, opacity: 0.78 }, projectionBounds);
      });
    }
    function drawHand(grasp, projectionBounds) {
      grasp.franka_hand_faces.forEach((face) => {
        line(grasp.franka_hand_vertices_obj[face[0]], grasp.franka_hand_vertices_obj[face[1]], { stroke: "#8f5a12", width: 0.9, opacity: 0.28 }, projectionBounds);
        line(grasp.franka_hand_vertices_obj[face[1]], grasp.franka_hand_vertices_obj[face[2]], { stroke: "#8f5a12", width: 0.9, opacity: 0.28 }, projectionBounds);
        line(grasp.franka_hand_vertices_obj[face[2]], grasp.franka_hand_vertices_obj[face[0]], { stroke: "#8f5a12", width: 0.9, opacity: 0.28 }, projectionBounds);
      });
    }
    function drawGrasp(grasp, projectionBounds) {
      grasp.franka_left_boxes.forEach((box) => drawBox(box.corners, "#d97706", projectionBounds));
      grasp.franka_right_boxes.forEach((box) => drawBox(box.corners, "#d97706", projectionBounds));
      drawHand(grasp, projectionBounds);
      line(grasp.contact_point_a_obj, grasp.contact_point_b_obj, { stroke: "#15803d", width: 3, opacity: 0.95 }, projectionBounds);
      point(grasp.grasp_position_obj, { fill: "#15803d", radius: 7 }, projectionBounds);
      point(grasp.contact_point_a_obj, { fill: "#c8452d", radius: 6 }, projectionBounds);
      point(grasp.contact_point_b_obj, { fill: "#1f7c60", radius: 6 }, projectionBounds);
      label(grasp.grasp_position_obj, grasp.grasp_id, "#15803d", projectionBounds);
    }
    function renderList() {
      const part = currentPart();
      list.replaceChildren();
      if (!part || !part.frames.length) {
        const empty = document.createElement("div");
        empty.className = "item-meta";
        empty.textContent = "No stable orientations for this part.";
        list.appendChild(empty);
        return;
      }
      part.frames.forEach((frame, index) => {
        const grasp = frame.best_grasp;
        const button = document.createElement("button");
        button.type = "button";
        button.className = `orientation-item${index === state.index ? " active" : ""}`;
        button.innerHTML = `
          <div class="item-title">
            <span>${frame.orientation.orientation_id}</span>
            <span class="status ${statusClass(frame.status)}">${frame.status}</span>
          </div>
          <div class="item-meta">
            feasible=${frame.stage2_ground_feasible_count} tilt=${fmt(frame.orientation.max_stable_tilt_deg, 1)}deg<br>
            best=${grasp ? `${grasp.grasp_id} score=${fmt(grasp.score, 3)}` : "none"}
          </div>
        `;
        button.addEventListener("click", () => { state.index = index; render(); });
        list.appendChild(button);
      });
    }
    function linkLabel(key) {
      return key.replace(/_html$/, "").replace(/_/g, " ");
    }
    function renderDetails(part, frame) {
      linkList.replaceChildren();
      if (!part) {
        details.textContent = "No browser data was generated.";
        return;
      }
      if (!frame) {
        details.textContent = [`target: ${part.target_mesh_path}`, "No stable orientations."].join("\\n");
        return;
      }
      Object.entries(frame.links || {}).filter(([, value]) => value).forEach(([key, value]) => {
        const anchor = document.createElement("a");
        anchor.href = value;
        anchor.textContent = linkLabel(key);
        linkList.appendChild(anchor);
      });
      const grasp = frame.best_grasp;
      details.textContent = [
        `target:             ${part.target_mesh_path}`,
        `precedence_plan:    ${part.precedence_plan_path || "none"}`,
        `assembled_before:   ${JSON.stringify(part.already_assembled_part_ids || [])}`,
        `obstacle_paths:     ${JSON.stringify(part.assembly_obstacle_paths === null ? "glob" : (part.assembly_obstacle_paths || []))}`,
        `pre_insert_path:    ${part.pre_insertion_poses_path || "none"}`,
        `sweep_vector_m:     ${JSON.stringify(part.insertion_sweep_vector_m || null)}`,
        `sweep_distance_m:   ${fmt(part.insertion_sweep_distance_m || 0, 6)}`,
        `orientation:        ${frame.orientation.orientation_id}`,
        `status:             ${frame.status}`,
        `stage1_feasible:    ${frame.stage1_assembly_feasible_count}`,
        `stage2_feasible:    ${frame.stage2_ground_feasible_count}`,
        `normal_obj:         (${frame.orientation.normal_obj.map((v) => fmt(v, 4)).join(", ")})`,
        `support_area_m2:    ${fmt(frame.orientation.area_m2, 6)}`,
        `stability_margin_m: ${fmt(frame.orientation.stability_margin_m, 6)}`,
        `com_height_m:       ${fmt(frame.orientation.com_height_m, 6)}`,
        `max_stable_tilt:    ${fmt(frame.orientation.max_stable_tilt_deg, 3)} deg`,
        `com_method:         ${frame.orientation.com_method}`,
        "",
        grasp ? `best_grasp:        ${grasp.grasp_id}` : "best_grasp:        none",
        grasp ? `best_score:        ${fmt(grasp.score, 6)}` : "",
        grasp ? `jaw_width_m:       ${fmt(grasp.jaw_width, 6)}` : "",
        grasp ? `roll_angle_rad:    ${fmt(grasp.roll_angle_rad, 6)}` : "",
        grasp && grasp.score_components ? `object_score:      ${fmt(grasp.score_components.object_score, 6)}` : "",
        grasp && grasp.score_components ? `top_down_score:    ${fmt(grasp.score_components.top_down_approach, 6)}` : "",
        "",
        frame.fallback ? `fallback_final:    ${frame.fallback.final_grasp_id}` : "",
        frame.handover_fallback ? `handover_final:    ${frame.handover_fallback.final_grasp_id}` : "",
        frame.handover_fallback ? `handover_transfer: ${frame.handover_fallback.transfer_grasp_id}` : "",
        frame.error ? `error:             ${frame.error}` : "",
      ].filter((line) => line !== "").join("\\n");
    }
    function renderScene(part, frame) {
      scene.replaceChildren();
      if (!part || !frame) return;
      const b = projection(part, frame);
      polygon(frame.floor_world, { fill: "#2563eb", fillOpacity: 0.13, stroke: "#2563eb", strokeWidth: 1.8, strokeOpacity: 0.85 }, b);
      for (let i = 0; i < frame.floor_world.length; i += 1) {
        line(frame.floor_world[i], frame.floor_world[(i + 1) % frame.floor_world.length], { stroke: "#2563eb", width: 1.8, opacity: 0.85, dash: "8 5" }, b);
      }
      drawMesh(part, frame, b);
      if (frame.best_grasp) {
        drawGrasp(frame.best_grasp, b);
      } else {
        label(frame.mesh_vertices_world[0], "no direct stage-2 grasp", "#b91c1c", b);
      }
    }
    function render() {
      assemblySelect.value = state.assembly;
      partSelect.value = state.partKey;
      const part = currentPart();
      if (part && state.index >= part.frames.length) state.index = 0;
      const frame = currentFrame();
      renderList();
      renderScene(part, frame);
      renderDetails(part, frame);
    }
    function stepOrientation(delta) {
      const part = currentPart();
      if (!part || !part.frames.length) return;
      state.index = (state.index + delta + part.frames.length) % part.frames.length;
      render();
    }
    function stepPart(delta) {
      const parts = partsForAssembly();
      if (!parts.length) return;
      const currentIndex = currentPartIndex();
      const nextIndex = ((currentIndex < 0 ? 0 : currentIndex) + delta + parts.length) % parts.length;
      const nextPart = parts[nextIndex];
      const frameCount = nextPart.frames.length;
      state.partKey = nextPart.key;
      state.index = frameCount ? Math.min(state.index, frameCount - 1) : 0;
      render();
    }
    assemblySelect.addEventListener("change", () => setAssembly(assemblySelect.value));
    partSelect.addEventListener("change", () => setPart(partSelect.value));
    document.getElementById("prevPartBtn").addEventListener("click", () => stepPart(-1));
    document.getElementById("nextPartBtn").addEventListener("click", () => stepPart(1));
    document.getElementById("prevOrientationBtn").addEventListener("click", () => stepOrientation(-1));
    document.getElementById("nextOrientationBtn").addEventListener("click", () => stepOrientation(1));
    document.getElementById("solidBtn").addEventListener("click", () => {
      state.solid = !state.solid;
      document.getElementById("solidBtn").textContent = state.solid ? "Wireframe Mesh" : "Solid Mesh";
      render();
    });
    document.getElementById("resetBtn").addEventListener("click", () => {
      Object.assign(state, initialView);
      render();
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") { event.preventDefault(); stepOrientation(-1); }
      if (event.key === "ArrowRight") { event.preventDefault(); stepOrientation(1); }
      if (event.key === "ArrowUp") { event.preventDefault(); stepPart(-1); }
      if (event.key === "ArrowDown") { event.preventDefault(); stepPart(1); }
    });
    scene.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 && event.button !== 1 && event.button !== 2) return;
      event.preventDefault();
      state.dragging = true;
      state.dragMode = event.button === 1 || event.button === 2 || event.shiftKey ? "pan" : "rotate";
      state.pointerId = event.pointerId;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      scene.setPointerCapture(event.pointerId);
    });
    function stopDragging() {
      state.dragging = false;
      state.pointerId = null;
    }
    scene.addEventListener("pointerup", (event) => { if (state.pointerId === event.pointerId) stopDragging(); });
    scene.addEventListener("pointercancel", stopDragging);
    scene.addEventListener("pointermove", (event) => {
      if (!state.dragging || (state.pointerId !== null && state.pointerId !== event.pointerId)) return;
      const dx = event.clientX - state.lastX;
      const dy = event.clientY - state.lastY;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      if (state.dragMode === "pan") {
        state.panX += dx;
        state.panY += dy;
      } else {
        state.yaw += dx * 0.01;
        state.pitch -= dy * 0.01;
      }
      render();
    });
    scene.addEventListener("wheel", (event) => {
      event.preventDefault();
      state.zoom = Math.max(0.3, Math.min(5.0, state.zoom * (event.deltaY < 0 ? 1.08 : 1 / 1.08)));
      render();
    }, { passive: false });
    scene.addEventListener("contextmenu", (event) => event.preventDefault());
    renderSummary();
    populateAssemblySelect();
    populatePartSelect();
    render();
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document.replace("__DATA_JSON__", data_json), encoding="utf-8")


def _world_floor_corners(vertices_world: np.ndarray) -> list[list[float]]:
    mins = vertices_world.min(axis=0)
    maxs = vertices_world.max(axis=0)
    extents = np.maximum(maxs - mins, 1.0e-3)
    padding = max(0.35 * float(np.max(extents[:2])), 0.08)
    return [
        [float(mins[0] - padding), float(mins[1] - padding), 0.0],
        [float(maxs[0] + padding), float(mins[1] - padding), 0.0],
        [float(maxs[0] + padding), float(maxs[1] + padding), 0.0],
        [float(mins[0] - padding), float(maxs[1] + padding), 0.0],
    ]


def _part_orientation_frame(
    *,
    stage1,
    planning: PlanningConfig,
    target: TargetSpec,
    orientation: StableOrientation,
    status: str,
    stage2=None,
    fallback_summary: dict[str, object] | None = None,
    handover_summary: dict[str, object] | None = None,
    links: dict[str, str] | None = None,
    error: str = "",
) -> dict[str, object]:
    mesh_vertices_world = orientation.object_pose_world.transform_points_to_world(
        np.asarray(stage1.target_mesh_local.vertices_obj, dtype=float)
    )
    best_grasp_payload = None
    if stage2 is not None and stage2.accepted:
        best_grasp_payload = candidate_payload(
            [CandidateStatus(grasp=stage2.accepted[0], status="accepted", reason="best_direct_grasp")],
            contact_gap_m=planning.detailed_finger_contact_gap_m,
            object_pose_world=orientation.object_pose_world,
        )[0]
    return {
        "target": {
            "assembly": target.assembly,
            "part_id": target.part_id,
            "target_mesh_path": target.target_mesh_path,
            "assembly_obstacle_paths": None
            if target.assembly_obstacle_paths is None
            else list(target.assembly_obstacle_paths),
            "precedence_plan_path": target.precedence_plan_path,
            "selected_assembly_order": list(target.selected_assembly_order),
            "already_assembled_part_ids": list(target.already_assembled_part_ids),
            "pre_insertion_poses_path": target.pre_insertion_poses_path,
            "pre_insertion_role": target.pre_insertion_role,
            "insertion_sweep_vector_m": None
            if target.insertion_sweep_vector_m is None
            else list(target.insertion_sweep_vector_m),
            "insertion_sweep_distance_m": target.insertion_sweep_distance_m,
            "final_to_pre_insertion_translation_m": None
            if target.final_to_pre_insertion_translation_m is None
            else list(target.final_to_pre_insertion_translation_m),
        },
        "orientation": stable_orientation_payload(orientation),
        "status": status,
        "error": error,
        "stage1_assembly_feasible_count": len(stage1.bundle.candidates),
        "stage2_ground_feasible_count": 0 if stage2 is None else len(stage2.accepted),
        "stage2_reason_counts": {} if stage2 is None else _reason_counts(stage2.statuses),
        "fallback": fallback_summary,
        "handover_fallback": handover_summary,
        "links": links or {},
        "mesh_vertices_world": [[float(v) for v in vertex] for vertex in mesh_vertices_world.tolist()],
        "floor_world": _world_floor_corners(mesh_vertices_world),
        "best_grasp": best_grasp_payload,
    }


def _part_browser_payload(
    *,
    target: TargetSpec,
    stage1,
    orientation_frames: list[dict[str, object]],
) -> dict[str, object]:
    mesh_local = stage1.target_mesh_local
    return {
        "key": f"{target.assembly}/{target.part_id}",
        "assembly": target.assembly,
        "part_id": target.part_id,
        "target_mesh_path": target.target_mesh_path,
        "assembly_obstacle_paths": None
        if target.assembly_obstacle_paths is None
        else list(target.assembly_obstacle_paths),
        "precedence_plan_path": target.precedence_plan_path,
        "selected_assembly_order": list(target.selected_assembly_order),
        "already_assembled_part_ids": list(target.already_assembled_part_ids),
        "pre_insertion_poses_path": target.pre_insertion_poses_path,
        "pre_insertion_role": target.pre_insertion_role,
        "insertion_sweep_vector_m": None
        if target.insertion_sweep_vector_m is None
        else list(target.insertion_sweep_vector_m),
        "insertion_sweep_distance_m": target.insertion_sweep_distance_m,
        "final_to_pre_insertion_translation_m": None
        if target.final_to_pre_insertion_translation_m is None
        else list(target.final_to_pre_insertion_translation_m),
        "vertices_local": [
            [float(value) for value in vertex] for vertex in np.asarray(mesh_local.vertices_obj, dtype=float).tolist()
        ],
        "faces": [[int(value) for value in face] for face in np.asarray(mesh_local.faces, dtype=np.int64).tolist()],
        "edges": unique_edges(mesh_local.faces),
        "frames": orientation_frames,
    }


def _sample_edges_for_overview(edges: list[tuple[int, int]], max_edges: int) -> list[tuple[int, int]]:
    if max_edges <= 0 or len(edges) <= max_edges:
        return edges
    indices = np.linspace(0, len(edges) - 1, int(max_edges), dtype=np.int64)
    return [edges[int(index)] for index in indices]


def _compact_vertices_for_edges(
    vertices: list[list[float]],
    edges: list[tuple[int, int]],
) -> tuple[list[list[float]], list[tuple[int, int]]]:
    old_to_new: dict[int, int] = {}
    compact_vertices: list[list[float]] = []
    compact_edges: list[tuple[int, int]] = []
    for start, end in edges:
        remapped = []
        for old_index in (int(start), int(end)):
            if old_index not in old_to_new:
                old_to_new[old_index] = len(compact_vertices)
                compact_vertices.append(vertices[old_index])
            remapped.append(old_to_new[old_index])
        compact_edges.append((remapped[0], remapped[1]))
    return compact_vertices, compact_edges


def _bounds_corners(points: list[list[float]]) -> list[list[float]]:
    if not points:
        return []
    array = np.asarray(points, dtype=float)
    mins = array.min(axis=0)
    maxs = array.max(axis=0)
    return [
        [float(x), float(y), float(z)]
        for x in (mins[0], maxs[0])
        for y in (mins[1], maxs[1])
        for z in (mins[2], maxs[2])
    ]


def _transform_points_for_overview(points_obj: np.ndarray, object_pose_world) -> list[list[float]]:
    return [
        [round(float(value), 6) for value in point]
        for point in object_pose_world.transform_points_to_world(np.asarray(points_obj, dtype=float)).tolist()
    ]


def _mesh_overview_payload(mesh_local: TriangleMesh, object_pose_world, *, max_edges: int) -> dict[str, object]:
    vertices = _transform_points_for_overview(np.asarray(mesh_local.vertices_obj, dtype=float), object_pose_world)
    edges_original = unique_edges(mesh_local.faces)
    edges = _sample_edges_for_overview(edges_original, max_edges)
    vertices, edges = _compact_vertices_for_edges(vertices, edges)
    return {
        "vertices": vertices,
        "edges": [[int(a), int(b)] for a, b in edges],
        "edge_count_original": len(edges_original),
    }


def _marker_point(point_obj: np.ndarray, object_pose_world) -> list[float]:
    return _transform_points_for_overview(np.asarray([point_obj], dtype=float), object_pose_world)[0]


def _candidate_marker_payload(
    entry: CandidateStatus,
    object_pose_world,
    *,
    raw_pickup_by_id: dict[str, CandidateStatus] | None = None,
) -> dict[str, object]:
    candidate = entry.grasp
    contact_a = np.asarray(candidate.contact_point_a_obj, dtype=float)
    contact_b = np.asarray(candidate.contact_point_b_obj, dtype=float)
    center = np.asarray(candidate.grasp_position_obj, dtype=float)
    approach_axis = quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj)[:, 2]
    marker_offset = np.asarray(approach_axis, dtype=float) * float(ALL_GRASP_OVERVIEW_MARKER_LENGTH_M)
    raw_pickup_entry = None if raw_pickup_by_id is None else raw_pickup_by_id.get(candidate.grasp_id)
    raw_pickup_status = "unknown" if raw_pickup_entry is None else raw_pickup_entry.status
    raw_pickup_reason = "" if raw_pickup_entry is None else raw_pickup_entry.reason
    return {
        "grasp_id": candidate.grasp_id,
        "status": entry.status,
        "reason": entry.reason,
        "raw_pickup_status": raw_pickup_status,
        "raw_pickup_reason": raw_pickup_reason,
        "raw_pickup_feasible": raw_pickup_status == "accepted",
        "score": None if candidate.score is None else round(float(candidate.score), 6),
        "roll_angle_rad": round(float(candidate.roll_angle_rad), 6),
        "jaw_width": round(float(candidate.jaw_width), 6),
        "center": _marker_point(center, object_pose_world),
        "contact_a": _marker_point(contact_a, object_pose_world),
        "contact_b": _marker_point(contact_b, object_pose_world),
        "post_a": _marker_point(contact_a - marker_offset, object_pose_world),
        "post_b": _marker_point(contact_b - marker_offset, object_pose_world),
    }


def _all_generated_grasp_statuses(stage1, stage2=None) -> list[CandidateStatus]:
    stage1_ids = {candidate.grasp_id for candidate in stage1.bundle.candidates}
    stage2_by_id = {} if stage2 is None else {entry.grasp.grasp_id: entry for entry in stage2.statuses}
    rejected_reason = (
        "part_sweep: stage1 insertion-swept assembly rejected"
        if (stage1.bundle.metadata or {}).get("assembly_obstacle_sweep_vector_m") is not None
        else "part: stage1 assembly rejected"
    )
    statuses: list[CandidateStatus] = []
    for candidate in sorted(stage1.raw_candidates, key=lambda grasp: (-_candidate_score(grasp), grasp.grasp_id)):
        if candidate.grasp_id not in stage1_ids:
            statuses.append(CandidateStatus(grasp=candidate, status="stage1_rejected", reason=rejected_reason))
            continue
        stage2_entry = stage2_by_id.get(candidate.grasp_id)
        if stage2_entry is None:
            statuses.append(CandidateStatus(grasp=candidate, status="stage1_pass", reason="stage1 passed; no stage2 status"))
        elif stage2_entry.status == "accepted":
            statuses.append(CandidateStatus(grasp=stage2_entry.grasp, status="accepted", reason=stage2_entry.reason))
        else:
            statuses.append(
                CandidateStatus(
                    grasp=stage2_entry.grasp,
                    status="stage2_rejected",
                    reason=f"floor: {stage2_entry.reason}",
                )
            )
    return statuses


def _raw_pickup_statuses_for_pose(
    stage1,
    *,
    object_pose_world,
    planning: PlanningConfig,
    floor_clearance_margin_m: float,
) -> list[CandidateStatus]:
    return evaluate_saved_grasps_against_pickup_pose(
        stage1.raw_candidates,
        object_pose_world=object_pose_world,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
        floor_clearance_margin_m=floor_clearance_margin_m,
        contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
        contact_approach_offsets_m=planning.contact_approach_offsets_m,
    )


def _write_all_generated_grasps_overview_html(
    output_html: Path,
    *,
    title: str,
    subtitle: str,
    mesh_local: TriangleMesh,
    candidate_statuses: list[CandidateStatus],
    object_pose_world,
    ground_plane: dict[str, object] | None = None,
    obstacle_mesh_local: TriangleMesh | None = None,
    raw_pickup_statuses: list[CandidateStatus] | None = None,
    metadata_lines: list[str] | None = None,
) -> dict[str, int]:
    target_payload = _mesh_overview_payload(mesh_local, object_pose_world, max_edges=ALL_GRASP_OVERVIEW_MAX_MESH_EDGES)
    obstacle_payload = {"vertices": [], "edges": [], "edge_count_original": 0, "bounds": []}
    if obstacle_mesh_local is not None:
        obstacle_payload = _mesh_overview_payload(
            obstacle_mesh_local,
            object_pose_world,
            max_edges=ALL_GRASP_OVERVIEW_MAX_OBSTACLE_EDGES,
        )
        obstacle_payload["bounds"] = _bounds_corners(obstacle_payload["vertices"])  # type: ignore[index]
    status_counts = dict(Counter(entry.status for entry in candidate_statuses))
    raw_pickup_by_id = (
        {}
        if raw_pickup_statuses is None
        else {entry.grasp.grasp_id: entry for entry in raw_pickup_statuses}
    )
    raw_pickup_counts = dict(Counter(entry.status for entry in raw_pickup_by_id.values()))
    data = {
        "title": title,
        "subtitle": subtitle,
        "metadata_lines": metadata_lines or [],
        "target": target_payload,
        "obstacle": obstacle_payload,
        "ground_plane": (
            None
            if ground_plane is None
            else {
                "corners": [
                    _marker_point(np.asarray(point, dtype=float), object_pose_world)
                    for point in ground_plane["corners_obj"]
                ]
            }
        ),
        "status_counts": status_counts,
        "raw_pickup_counts": raw_pickup_counts,
        "markers": [
            _candidate_marker_payload(entry, object_pose_world, raw_pickup_by_id=raw_pickup_by_id)
            for entry in candidate_statuses
        ],
    }
    data_json = json.dumps(data, indent=2)
    document = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>All Generated Grasp Markers</title>
  <style>
    :root {
      --bg: #f6f4ee;
      --panel: #fffdf8;
      --ink: #1f2522;
      --muted: #68716c;
      --line: #d9d4c7;
      --mesh: #2f6f5e;
      --obstacle: #64748b;
      --ground: #2563eb;
      --stage1: #b91c1c;
      --stage2: #d97706;
      --accepted: #15803d;
      --pass: #1d4ed8;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    .layout { display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: 100vh; }
    aside { border-right: 1px solid var(--line); background: var(--panel); padding: 18px; overflow: auto; }
    main { padding: 18px; overflow: hidden; }
    h1 { margin: 0 0 8px; font-size: 24px; line-height: 1.15; }
    .subtitle { margin: 0 0 14px; color: var(--muted); font-size: 14px; line-height: 1.45; }
    .panel { border: 1px solid var(--line); background: rgba(255,253,248,0.96); border-radius: 8px; padding: 12px; margin-bottom: 12px; }
    .checks { display: grid; gap: 8px; }
    label { display: flex; align-items: center; justify-content: space-between; gap: 12px; font-size: 13px; color: var(--muted); }
    input[type="checkbox"] { width: 18px; height: 18px; }
    input[type="range"] { width: 100%; }
    button { border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 8px; padding: 8px 10px; font: inherit; cursor: pointer; }
    button:hover { border-color: var(--mesh); }
    .controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .segmented { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
    .segmented.selection { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .segmented button.active { border-color: var(--mesh); background: #e7f2ef; color: #1f5d4f; }
    .range-row { display: grid; gap: 8px; }
    .inline-controls { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: center; }
    .small-note { margin: 8px 0 0; color: var(--muted); font-size: 12px; line-height: 1.35; }
    .kv { white-space: pre-wrap; font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.55; margin: 0; }
    .legend { display: grid; gap: 7px; font-size: 13px; color: var(--muted); }
    .legend span { display: flex; align-items: center; gap: 8px; }
    .swatch { width: 14px; height: 14px; border-radius: 999px; display: inline-block; }
    .canvas-wrap { height: calc(100vh - 36px); border: 1px solid var(--line); border-radius: 8px; background: linear-gradient(180deg, #ffffff, #ebe7dc); overflow: hidden; }
    canvas { display: block; width: 100%; height: 100%; cursor: grab; }
    @media (max-width: 1050px) {
      .layout { grid-template-columns: 1fr; }
      main { overflow: visible; }
      .canvas-wrap { height: 70vh; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1 id="title"></h1>
      <p id="subtitle" class="subtitle"></p>
      <section class="panel legend">
        <span><i class="swatch" style="background: var(--stage1)"></i>Stage 1 assembly rejected</span>
        <span><i class="swatch" style="background: var(--stage2)"></i>Stage 1 passed, stage 2 floor rejected</span>
        <span><i class="swatch" style="background: var(--accepted)"></i>Accepted in this orientation</span>
        <span><i class="swatch" style="background: var(--pass)"></i>Stage 1 passed, no stage 2 status</span>
      </section>
      <section class="panel">
        <div class="segmented selection" id="selectionModeButtons">
          <button type="button" class="active" data-selection-mode="all">All Generated</button>
          <button type="button" data-selection-mode="stage1">Passed Stage 1</button>
          <button type="button" data-selection-mode="stage2">Passed Stage 2</button>
          <button type="button" data-selection-mode="pickup">Simple Pickup</button>
        </div>
        <p id="selectionNote" class="small-note"></p>
      </section>
      <section class="panel">
        <div class="segmented" id="rollModeButtons">
          <button type="button" class="active" data-roll-mode="all">All Rolls</button>
          <button type="button" data-roll-mode="zero">0 Roll</button>
          <button type="button" data-roll-mode="three">3 Rolls</button>
        </div>
        <p id="rollNote" class="small-note"></p>
      </section>
      <section class="panel range-row">
        <label>Shown <output id="samplePctLabel">100%</output></label>
        <input id="samplePct" type="range" min="0" max="100" value="100" step="1">
        <div class="inline-controls">
          <span id="seedLabel" class="small-note">seed: 1</span>
          <button id="newSeedBtn" type="button">New Seed</button>
        </div>
      </section>
      <section class="panel controls">
        <button id="prevBtn" type="button">Previous</button>
        <button id="nextBtn" type="button">Next</button>
        <button id="resetBtn" type="button">Reset View</button>
        <button id="meshBtn" type="button">Mesh On</button>
      </section>
      <section class="panel">
        <pre id="details" class="kv"></pre>
      </section>
    </aside>
    <main>
      <div class="canvas-wrap"><canvas id="scene"></canvas></div>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    const details = document.getElementById("details");
    document.getElementById("title").textContent = data.title;
    document.getElementById("subtitle").textContent = data.subtitle;
    const samplePct = document.getElementById("samplePct");
    const samplePctLabel = document.getElementById("samplePctLabel");
    const seedLabel = document.getElementById("seedLabel");
    const selectionNote = document.getElementById("selectionNote");
    const rollNote = document.getElementById("rollNote");
    const selectionButtons = Array.from(document.querySelectorAll("#selectionModeButtons button"));
    const rollButtons = Array.from(document.querySelectorAll("#rollModeButtons button"));
    const state = {
      yaw: -0.72,
      pitch: 0.52,
      zoom: 1,
      panX: 0,
      panY: 0,
      selected: 0,
      dragging: false,
      dragMode: "rotate",
      lastX: 0,
      lastY: 0,
      showMesh: true,
      selectionMode: "all",
      rollMode: "all",
      samplePct: 100,
      sampleSeed: 1,
    };
    const colors = { stage1_rejected: "#b91c1c", stage2_rejected: "#d97706", accepted: "#15803d", stage1_pass: "#1d4ed8" };
    const TAU = Math.PI * 2;
    function normalizeRoll(value) {
      let roll = Number(value || 0) % TAU;
      if (roll < 0) roll += TAU;
      if (Math.abs(roll - TAU) < 1e-6) roll = 0;
      return roll;
    }
    function rollKey(value) { return String(Math.round(normalizeRoll(value) * 1e6)); }
    const rollBucketMap = new Map();
    data.markers.forEach((marker) => {
      const key = rollKey(marker.roll_angle_rad);
      if (!rollBucketMap.has(key)) rollBucketMap.set(key, normalizeRoll(marker.roll_angle_rad));
    });
    const rollBuckets = Array.from(rollBucketMap.entries())
      .map(([key, angle]) => ({ key, angle }))
      .sort((a, b) => a.angle - b.angle);
    function angularDistance(a, b) {
      const diff = Math.abs(a - b) % TAU;
      return Math.min(diff, TAU - diff);
    }
    function chooseNearestRollBucket(target, usedKeys) {
      let best = null;
      let bestDistance = Infinity;
      rollBuckets.forEach((bucket) => {
        if (usedKeys.has(bucket.key)) return;
        const distance = angularDistance(bucket.angle, target);
        if (distance < bestDistance) {
          best = bucket;
          bestDistance = distance;
        }
      });
      return best;
    }
    function rollBucketsForMode() {
      if (state.rollMode === "all") return null;
      if (!rollBuckets.length) return [];
      if (state.rollMode === "zero") {
        const exactZero = rollBuckets.filter((bucket) => angularDistance(bucket.angle, 0) <= 1e-4);
        if (exactZero.length) return exactZero;
        const nearest = chooseNearestRollBucket(0, new Set());
        return nearest ? [nearest] : [];
      }
      const used = new Set();
      const selected = [];
      [0, TAU / 3, (2 * TAU) / 3].forEach((target) => {
        const bucket = chooseNearestRollBucket(target, used);
        if (bucket) {
          selected.push(bucket);
          used.add(bucket.key);
        }
      });
      return selected;
    }
    function updateRollNote() {
      const buckets = rollBucketsForMode();
      if (buckets === null) {
        rollNote.textContent = `${rollBuckets.length} roll buckets available.`;
        return;
      }
      const degrees = buckets.map((bucket) => `${(bucket.angle * 180 / Math.PI).toFixed(1)} deg`).join(", ");
      rollNote.textContent = buckets.length ? `showing roll bucket(s): ${degrees}` : "no roll bucket selected.";
    }
    function hashString(text) {
      let hash = 2166136261;
      for (let i = 0; i < text.length; i += 1) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
      }
      return hash >>> 0;
    }
    function randomRank(marker) {
      return hashString(`${state.sampleSeed}:${marker.grasp_id}`) / 4294967296;
    }
    function selectionVisible(marker) {
      if (state.selectionMode === "stage1") return marker.status !== "stage1_rejected";
      if (state.selectionMode === "stage2") return marker.status === "accepted";
      if (state.selectionMode === "pickup") return Boolean(marker.raw_pickup_feasible);
      return true;
    }
    function updateSelectionNote() {
      const counts = {
        all: data.markers.length,
        stage1: data.markers.filter((marker) => marker.status !== "stage1_rejected").length,
        stage2: data.markers.filter((marker) => marker.status === "accepted").length,
        pickup: data.markers.filter((marker) => marker.raw_pickup_feasible).length,
      };
      selectionNote.textContent = `all ${counts.all}, stage1 ${counts.stage1}, stage2 ${counts.stage2}, simple pickup ${counts.pickup}`;
    }
    function rollVisible(marker, activeRollKeys) {
      if (activeRollKeys === null) return true;
      return activeRollKeys.has(rollKey(marker.roll_angle_rad));
    }
    function markerEligible(marker, activeRollKeys) {
      if (!selectionVisible(marker)) return false;
      if (!rollVisible(marker, activeRollKeys)) return false;
      return true;
    }
    function eligibleMarkers() {
      const activeRollBuckets = rollBucketsForMode();
      const activeRollKeys = activeRollBuckets === null ? null : new Set(activeRollBuckets.map((bucket) => bucket.key));
      return data.markers.filter((marker) => markerEligible(marker, activeRollKeys));
    }
    function visibleMarkers() {
      const eligible = eligibleMarkers();
      if (state.samplePct >= 100) return eligible;
      if (state.samplePct <= 0 || !eligible.length) return [];
      const count = Math.ceil(eligible.length * state.samplePct / 100);
      return eligible
        .map((marker) => ({ marker, rank: randomRank(marker) }))
        .sort((a, b) => a.rank - b.rank || a.marker.grasp_id.localeCompare(b.marker.grasp_id))
        .slice(0, count)
        .map((entry) => entry.marker);
    }
    function countBy(markers, key) {
      return markers.reduce((acc, marker) => {
        const value = marker[key] || "unknown";
        acc[value] = (acc[value] || 0) + 1;
        return acc;
      }, {});
    }
    function markerTopMid(marker) {
      return marker.contact_a.map((value, axis) => 0.5 * (value + marker.contact_b[axis]));
    }
    function markerHandMid(marker) {
      return marker.post_a.map((value, axis) => 0.5 * (value + marker.post_b[axis]));
    }
    function markerStemEnd(marker) {
      const topMid = markerTopMid(marker);
      const handMid = markerHandMid(marker);
      return handMid.map((value, axis) => value + 0.75 * (value - topMid[axis]));
    }
    function allPoints() {
      return [
        ...data.target.vertices,
        ...data.obstacle.vertices,
        ...(data.obstacle.bounds || []),
        ...(data.ground_plane ? data.ground_plane.corners : []),
        ...data.markers.flatMap((m) => [m.center, m.contact_a, m.contact_b, m.post_a, m.post_b, markerStemEnd(m)]),
      ];
    }
    const bounds = allPoints().reduce((acc, point) => {
      point.forEach((value, axis) => { acc.min[axis] = Math.min(acc.min[axis], value); acc.max[axis] = Math.max(acc.max[axis], value); });
      return acc;
    }, { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
    const center = bounds.min.map((value, axis) => 0.5 * (value + bounds.max[axis]));
    const extent = Math.max(...bounds.max.map((value, axis) => value - bounds.min[axis]), 0.12);
    function resize() {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }
    function rotate(point) {
      const shifted = point.map((value, axis) => value - center[axis]);
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];
      return [x1, cp * y1 + sp * z1, -sp * y1 + cp * z1];
    }
    function project(point) {
      const rect = canvas.getBoundingClientRect();
      const [x, y, z] = rotate(point);
      const scale = (0.68 * Math.min(rect.width, rect.height) / extent) * state.zoom;
      return { x: rect.width * 0.5 + state.panX + x * scale, y: rect.height * 0.5 + state.panY - y * scale, depth: z };
    }
    function line(a, b, color, width = 1, alpha = 1) {
      const pa = project(a), pb = project(b);
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    function polygon(points, color, alpha) {
      if (!points.length) return;
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.beginPath();
      points.forEach((point, index) => {
        const p = project(point);
        if (index === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    function drawEdges(vertices, edges, color, width, alpha) {
      edges.forEach(([a, b]) => line(vertices[a], vertices[b], color, width, alpha));
    }
    function drawMarker(marker, selected) {
      const color = colors[marker.status] || "#111827";
      const topMid = markerTopMid(marker);
      const handMid = markerHandMid(marker);
      const stemEnd = markerStemEnd(marker);
      line(marker.contact_a, marker.post_a, color, selected ? 2.8 : 1.15, selected ? 1 : 0.58);
      line(marker.contact_b, marker.post_b, color, selected ? 2.8 : 1.15, selected ? 1 : 0.58);
      line(marker.post_a, marker.post_b, color, selected ? 2.8 : 1.15, selected ? 1 : 0.58);
      line(handMid, stemEnd, color, selected ? 2.8 : 1.15, selected ? 1 : 0.58);
      if (selected) {
        const p = project(topMid);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    function draw() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      if (data.ground_plane) {
        polygon(data.ground_plane.corners, "#2563eb", 0.12);
        for (let i = 0; i < data.ground_plane.corners.length; i += 1) {
          line(data.ground_plane.corners[i], data.ground_plane.corners[(i + 1) % data.ground_plane.corners.length], "#2563eb", 1.4, 0.75);
        }
      }
      if (state.showMesh) {
        drawEdges(data.obstacle.vertices, data.obstacle.edges, "#64748b", 0.8, 0.22);
        drawEdges(data.target.vertices, data.target.edges, "#2f6f5e", 1.1, 0.55);
      }
      const markers = visibleMarkers();
      markers.forEach((marker, index) => drawMarker(marker, index === state.selected));
      renderDetails();
    }
    function renderDetails() {
      const markers = visibleMarkers();
      const eligible = eligibleMarkers();
      if (state.selected >= markers.length) state.selected = 0;
      const marker = markers[state.selected];
      details.textContent = [
        ...data.metadata_lines,
        "",
        `raw_markers:      ${data.markers.length}`,
        `eligible_markers: ${eligible.length}`,
        `visible_markers:  ${markers.length}`,
        `status_counts:    ${JSON.stringify(data.status_counts)}`,
        `visible_status:   ${JSON.stringify(countBy(markers, "status"))}`,
        `raw_pickup:       ${JSON.stringify(data.raw_pickup_counts || {})}`,
        `visible_pickup:   ${JSON.stringify(countBy(markers, "raw_pickup_status"))}`,
        `selection_mode:   ${state.selectionMode}`,
        `roll_mode:        ${state.rollMode}`,
        `sample_percent:   ${state.samplePct}%`,
        `sample_seed:      ${state.sampleSeed}`,
        `target_edges:     ${data.target.edges.length}/${data.target.edge_count_original}`,
        `obstacle_edges:   ${data.obstacle.edges.length}/${data.obstacle.edge_count_original}`,
        "",
        marker ? `selected:         ${marker.grasp_id}` : "selected:         none",
        marker ? `status:           ${marker.status}` : "",
        marker ? `reason:           ${marker.reason}` : "",
        marker ? `raw_pickup:       ${marker.raw_pickup_status}` : "",
        marker ? `raw_pickup_reason:${marker.raw_pickup_reason}` : "",
        marker ? `score:            ${marker.score === null ? "n/a" : marker.score}` : "",
        marker ? `roll_angle_rad:   ${marker.roll_angle_rad}` : "",
        marker ? `jaw_width_m:      ${marker.jaw_width}` : "",
      ].filter((line) => line !== "").join("\\n");
    }
    function selectDelta(delta) {
      const markers = visibleMarkers();
      if (!markers.length) return;
      state.selected = (state.selected + delta + markers.length) % markers.length;
      draw();
    }
    function selectNearest(x, y) {
      const markers = visibleMarkers();
      let bestIndex = -1, bestDistance = Infinity;
      markers.forEach((marker, index) => {
        const p = project(markerTopMid(marker));
        const dist = Math.hypot(p.x - x, p.y - y);
        if (dist < bestDistance) { bestDistance = dist; bestIndex = index; }
      });
      if (bestIndex >= 0 && bestDistance < 24) {
        state.selected = bestIndex;
        draw();
      }
    }
    function updateSampleLabels() {
      samplePctLabel.textContent = `${state.samplePct}%`;
      seedLabel.textContent = `seed: ${state.sampleSeed}`;
    }
    function resetSelectionAndDraw() {
      state.selected = 0;
      updateSelectionNote();
      updateRollNote();
      updateSampleLabels();
      draw();
    }
    selectionButtons.forEach((button) => {
      button.addEventListener("click", () => {
        state.selectionMode = button.dataset.selectionMode || "all";
        selectionButtons.forEach((item) => item.classList.toggle("active", item === button));
        resetSelectionAndDraw();
      });
    });
    rollButtons.forEach((button) => {
      button.addEventListener("click", () => {
        state.rollMode = button.dataset.rollMode || "all";
        rollButtons.forEach((item) => item.classList.toggle("active", item === button));
        resetSelectionAndDraw();
      });
    });
    samplePct.addEventListener("input", () => {
      state.samplePct = Number(samplePct.value);
      resetSelectionAndDraw();
    });
    document.getElementById("newSeedBtn").addEventListener("click", () => {
      state.sampleSeed += 1;
      resetSelectionAndDraw();
    });
    document.getElementById("prevBtn").addEventListener("click", () => selectDelta(-1));
    document.getElementById("nextBtn").addEventListener("click", () => selectDelta(1));
    document.getElementById("resetBtn").addEventListener("click", () => { Object.assign(state, { yaw: -0.72, pitch: 0.52, zoom: 1, panX: 0, panY: 0 }); draw(); });
    document.getElementById("meshBtn").addEventListener("click", (event) => { state.showMesh = !state.showMesh; event.target.textContent = state.showMesh ? "Mesh On" : "Mesh Off"; draw(); });
    window.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") { event.preventDefault(); selectDelta(-1); }
      if (event.key === "ArrowRight" || event.key === "ArrowDown") { event.preventDefault(); selectDelta(1); }
    });
    canvas.addEventListener("pointerdown", (event) => {
      const rect = canvas.getBoundingClientRect();
      if (event.button === 0 && !event.shiftKey) selectNearest(event.clientX - rect.left, event.clientY - rect.top);
      state.dragging = true;
      state.dragMode = event.button === 1 || event.shiftKey ? "pan" : "rotate";
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
      canvas.style.cursor = state.dragMode === "pan" ? "move" : "grabbing";
    });
    canvas.addEventListener("pointerup", (event) => { state.dragging = false; canvas.releasePointerCapture(event.pointerId); canvas.style.cursor = "grab"; });
    canvas.addEventListener("pointercancel", () => { state.dragging = false; canvas.style.cursor = "grab"; });
    canvas.addEventListener("pointermove", (event) => {
      if (!state.dragging) return;
      const dx = event.clientX - state.lastX, dy = event.clientY - state.lastY;
      state.lastX = event.clientX; state.lastY = event.clientY;
      if (state.dragMode === "pan") { state.panX += dx; state.panY += dy; }
      else { state.yaw += dx * 0.01; state.pitch -= dy * 0.01; }
      draw();
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      state.zoom = Math.max(0.25, Math.min(5, state.zoom * (event.deltaY < 0 ? 1.08 : 1 / 1.08)));
      draw();
    }, { passive: false });
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    window.addEventListener("resize", resize);
    updateSelectionNote();
    updateRollNote();
    updateSampleLabels();
    resize();
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(document.replace("__DATA_JSON__", data_json), encoding="utf-8")
    return status_counts


def _write_all_generated_grasps_html(
    output_html: Path,
    *,
    target: TargetSpec,
    orientation: StableOrientation,
    status: str,
    stage1,
    planning: PlanningConfig,
    simple_pickup_floor_clearance_margin_m: float,
    stage2=None,
    error: str = "",
) -> dict[str, int]:
    display_pose = orientation.object_pose_world if stage2 is None else stage2.pickup_pose_world
    obstacle_mesh_local = None
    if stage1.obstacle_mesh_world is not None:
        obstacle_mesh_local = _mesh_in_object_frame(stage1.obstacle_mesh_world, stage1.target_pose_in_obj_world)
    candidate_statuses = _all_generated_grasp_statuses(stage1, stage2=stage2)
    raw_pickup_statuses = _raw_pickup_statuses_for_pose(
        stage1,
        object_pose_world=display_pose,
        planning=planning,
        floor_clearance_margin_m=simple_pickup_floor_clearance_margin_m,
    )
    raw_pickup_counts = Counter(entry.status for entry in raw_pickup_statuses)
    metadata_lines = [
        f"target_mesh:      {target.target_mesh_path}",
        f"assembly:         {target.assembly}",
        f"part_id:          {target.part_id}",
        f"precedence_plan:  {target.precedence_plan_path or 'none'}",
        f"assembled_before: {list(target.already_assembled_part_ids)}",
        f"obstacle_paths:   {list(target.assembly_obstacle_paths) if target.assembly_obstacle_paths is not None else 'glob'}",
        f"pre_insert_path:  {target.pre_insertion_poses_path or 'none'}",
        f"sweep_vector_m:   {target.insertion_sweep_vector_m}",
        f"sweep_distance_m: {target.insertion_sweep_distance_m:.6f}",
        f"orientation:      {orientation.orientation_id}",
        f"benchmark_status: {status}",
        f"display_frame:    {FAILED_GRASP_DISPLAY_FRAME}",
        f"raw_candidates:   {len(stage1.raw_candidates)}",
        f"stage1_feasible:  {len(stage1.bundle.candidates)}",
        f"stage2_feasible:  {0 if stage2 is None else len(stage2.accepted)}",
        f"raw_pickup_ok:    {raw_pickup_counts.get('accepted', 0)}",
        f"floor_clearance:  {planning.floor_clearance_margin_m:.6f} m",
        f"simple_pickup_floor_clearance: {simple_pickup_floor_clearance_margin_m:.6f} m",
        "marker_shape:     open grasp side; hand-side crossbar plus stem",
    ]
    if error:
        metadata_lines.append(f"error:            {error}")
    return _write_all_generated_grasps_overview_html(
        output_html,
        title=f"All Generated Grasps: {target.assembly}/{target.part_id} {orientation.orientation_id}",
        subtitle="Every generated grasp as a lightweight goalpost marker, colored by the first stage that rejected it.",
        mesh_local=stage1.target_mesh_local,
        candidate_statuses=candidate_statuses,
        object_pose_world=display_pose,
        ground_plane=ground_plane_overlay_obj(stage1.target_mesh_local, object_pose_world=display_pose, enabled=True),
        obstacle_mesh_local=obstacle_mesh_local,
        raw_pickup_statuses=raw_pickup_statuses,
        metadata_lines=metadata_lines,
    )


def _handover_pair_payload(pair: HandoverGraspPair, *, planning: PlanningConfig) -> dict[str, object]:
    transfer_payload = candidate_payload(
        [CandidateStatus(grasp=pair.transfer_grasp, status=pair.status, reason=pair.reason)],
        contact_gap_m=planning.detailed_finger_contact_gap_m,
    )[0]
    final_payload = candidate_payload(
        [CandidateStatus(grasp=pair.final_grasp, status=pair.status, reason=pair.reason)],
        contact_gap_m=planning.detailed_finger_contact_gap_m,
    )[0]
    return {
        "status": pair.status,
        "reason": pair.reason,
        "score": pair.score,
        "transfer": transfer_payload,
        "final": final_payload,
        "metadata": pair.metadata,
    }


def _write_handover_grasp_pairs_html(
    output_html: Path,
    *,
    target: TargetSpec,
    orientation: StableOrientation,
    status: str,
    stage1,
    planning: PlanningConfig,
    result: HandoverFallbackResult,
    limit: int = HANDOVER_PAIR_HTML_LIMIT,
) -> dict[str, int]:
    accepted_limit = min(HANDOVER_PAIR_ACCEPTED_HTML_LIMIT, max(0, int(limit)))
    accepted_pairs = list(result.accepted_pairs[:accepted_limit])
    rejected_pairs = list(result.rejected_pairs[: max(0, int(limit) - len(accepted_pairs))])
    pairs = accepted_pairs + rejected_pairs
    status_counts = dict(Counter(pair.status for pair in pairs))
    mesh_local = stage1.target_mesh_local
    data = {
        "title": f"Handover Fallback Pairs: {target.assembly}/{target.part_id} {orientation.orientation_id}",
        "subtitle": "Planning-only reverse handover search. The final hand is assembly-feasible; the transfer hand is floor-feasible in the current pose.",
        "target_mesh_path": target.target_mesh_path,
        "metadata_lines": [
            f"benchmark_status: {status}",
            f"precedence_plan:  {target.precedence_plan_path or 'none'}",
            f"assembled_before: {list(target.already_assembled_part_ids)}",
            f"obstacle_paths:   {list(target.assembly_obstacle_paths) if target.assembly_obstacle_paths is not None else 'glob'}",
            f"pre_insert_path:  {target.pre_insertion_poses_path or 'none'}",
            f"sweep_vector_m:   {target.insertion_sweep_vector_m}",
            f"sweep_distance_m: {target.insertion_sweep_distance_m:.6f}",
            f"raw_candidates:   {len(stage1.raw_candidates)}",
            f"stage1_feasible:  {len(stage1.bundle.candidates)}",
            f"accepted_pairs:   {len(result.accepted_pairs)}",
            f"rejected_shown:   {len(rejected_pairs)}",
            f"checked_pairs:    {result.metadata.get('checked_pair_count', 0)}",
            f"rejection_counts: {result.metadata.get('rejection_counts', {})}",
            f"floor_counts:     {result.transfer_floor_status_counts}",
        ],
        "vertices_obj": [[float(v) for v in vertex] for vertex in np.asarray(mesh_local.vertices_obj, dtype=float).tolist()],
        "edges": unique_edges(mesh_local.faces),
        "faces": [[int(value) for value in face] for face in np.asarray(mesh_local.faces, dtype=np.int64).tolist()],
        "status_counts": status_counts,
        "pairs": [_handover_pair_payload(pair, planning=planning) for pair in pairs],
    }
    data_json = json.dumps(data, indent=2)
    document = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Handover Fallback Pairs</title>
  <style>
    :root {
      --bg: #f6f4ee;
      --panel: #fffdf8;
      --ink: #1f2522;
      --muted: #68716c;
      --line: #d9d4c7;
      --mesh: #2f6f5e;
      --transfer: #1d4ed8;
      --final: #15803d;
      --reject: #b91c1c;
      --hand: #8f5a12;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    .layout { display: grid; grid-template-columns: 390px minmax(0, 1fr); min-height: 100vh; }
    aside { border-right: 1px solid var(--line); background: var(--panel); padding: 18px; overflow: auto; }
    main { padding: 18px; overflow: hidden; }
    h1 { margin: 0 0 8px; font-size: 24px; line-height: 1.15; }
    .subtitle { margin: 0 0 14px; color: var(--muted); font-size: 14px; line-height: 1.45; }
    .panel { border: 1px solid var(--line); background: rgba(255,253,248,0.96); border-radius: 8px; padding: 12px; margin-bottom: 12px; }
    .controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    button { border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 8px; padding: 8px 10px; font: inherit; cursor: pointer; }
    button:hover { border-color: var(--mesh); }
    .pair-list { display: grid; gap: 8px; }
    .pair-item { text-align: left; border-radius: 8px; }
    .pair-item.active { border-color: var(--mesh); box-shadow: 0 0 0 2px rgba(47,111,94,0.14); }
    .status { border-radius: 999px; padding: 2px 7px; font-size: 12px; white-space: nowrap; }
    .accepted { background: #dcfce7; color: var(--final); }
    .rejected { background: #fee2e2; color: var(--reject); }
    .item-title { display: flex; justify-content: space-between; gap: 8px; font-weight: 700; }
    .item-meta { margin-top: 5px; color: var(--muted); font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.4; }
    .kv { white-space: pre-wrap; font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.55; margin: 0; }
    .canvas-wrap { height: calc(100vh - 36px); border: 1px solid var(--line); border-radius: 8px; background: linear-gradient(180deg, #ffffff, #ebe7dc); overflow: hidden; }
    canvas { display: block; width: 100%; height: 100%; cursor: grab; }
    @media (max-width: 1050px) {
      .layout { grid-template-columns: 1fr; }
      main { overflow: visible; }
      .canvas-wrap { height: 70vh; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1 id="title"></h1>
      <p id="subtitle" class="subtitle"></p>
      <section class="panel controls">
        <button id="prevBtn" type="button">Previous</button>
        <button id="nextBtn" type="button">Next</button>
        <button id="resetBtn" type="button">Reset View</button>
        <button id="meshBtn" type="button">Mesh On</button>
      </section>
      <section class="panel"><pre id="details" class="kv"></pre></section>
      <section id="pairList" class="pair-list"></section>
    </aside>
    <main>
      <div class="canvas-wrap"><canvas id="scene"></canvas></div>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    const details = document.getElementById("details");
    const pairList = document.getElementById("pairList");
    document.getElementById("title").textContent = data.title;
    document.getElementById("subtitle").textContent = data.subtitle;
    const state = { yaw: -0.72, pitch: 0.52, zoom: 1, panX: 0, panY: 0, selected: 0, dragging: false, dragMode: "rotate", lastX: 0, lastY: 0, showMesh: true };
    function allPairPoints(pair) {
      if (!pair) return [];
      return [pair.transfer, pair.final].flatMap((grasp) => [
        grasp.grasp_position_obj,
        grasp.contact_point_a_obj,
        grasp.contact_point_b_obj,
        ...grasp.franka_hand_vertices_obj,
        ...grasp.franka_left_boxes.flatMap((box) => box.corners),
        ...grasp.franka_right_boxes.flatMap((box) => box.corners),
      ]);
    }
    function allPoints() {
      return [...data.vertices_obj, ...data.pairs.flatMap(allPairPoints)];
    }
    const bounds = allPoints().reduce((acc, point) => {
      point.forEach((value, axis) => { acc.min[axis] = Math.min(acc.min[axis], value); acc.max[axis] = Math.max(acc.max[axis], value); });
      return acc;
    }, { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
    const center = bounds.min.map((value, axis) => 0.5 * (value + bounds.max[axis]));
    const extent = Math.max(...bounds.max.map((value, axis) => value - bounds.min[axis]), 0.12);
    function resize() {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }
    function rotate(point) {
      const shifted = point.map((value, axis) => value - center[axis]);
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];
      return [x1, cp * y1 + sp * z1, -sp * y1 + cp * z1];
    }
    function project(point) {
      const rect = canvas.getBoundingClientRect();
      const [x, y, z] = rotate(point);
      const scale = (0.68 * Math.min(rect.width, rect.height) / extent) * state.zoom;
      return { x: rect.width * 0.5 + state.panX + x * scale, y: rect.height * 0.5 + state.panY - y * scale, depth: z };
    }
    function line(a, b, color, width = 1, alpha = 1) {
      const pa = project(a), pb = project(b);
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    function point(p, color, radius = 4) {
      const pp = project(p);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(pp.x, pp.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    function drawBox(corners, color, alpha) {
      [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([a, b]) => line(corners[a], corners[b], color, 1.3, alpha));
    }
    function drawHand(grasp, color, alpha) {
      grasp.franka_hand_faces.forEach((face) => {
        line(grasp.franka_hand_vertices_obj[face[0]], grasp.franka_hand_vertices_obj[face[1]], color, 0.75, alpha * 0.45);
        line(grasp.franka_hand_vertices_obj[face[1]], grasp.franka_hand_vertices_obj[face[2]], color, 0.75, alpha * 0.45);
        line(grasp.franka_hand_vertices_obj[face[2]], grasp.franka_hand_vertices_obj[face[0]], color, 0.75, alpha * 0.45);
      });
      grasp.franka_left_boxes.forEach((box) => drawBox(box.corners, color, alpha));
      grasp.franka_right_boxes.forEach((box) => drawBox(box.corners, color, alpha));
      line(grasp.contact_point_a_obj, grasp.contact_point_b_obj, color, 2.4, alpha);
      point(grasp.grasp_position_obj, color, 4.5);
    }
    function renderList() {
      pairList.replaceChildren();
      data.pairs.forEach((pair, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `pair-item${index === state.selected ? " active" : ""}`;
        button.innerHTML = `<div class="item-title"><span>${pair.final.grasp_id} / ${pair.transfer.grasp_id}</span><span class="status ${pair.status}">${pair.status}</span></div><div class="item-meta">${pair.reason}<br>final=${pair.final.score ?? "n/a"} transfer=${pair.transfer.score ?? "n/a"}</div>`;
        button.addEventListener("click", () => { state.selected = index; draw(); renderList(); });
        pairList.appendChild(button);
      });
    }
    function draw() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      if (state.showMesh) {
        data.edges.forEach(([a, b]) => line(data.vertices_obj[a], data.vertices_obj[b], "#2f6f5e", 1, 0.42));
      }
      const pair = data.pairs[state.selected];
      if (pair) {
        const rejected = pair.status !== "accepted";
        drawHand(pair.final, rejected ? "#64748b" : "#15803d", rejected ? 0.55 : 0.9);
        drawHand(pair.transfer, rejected ? "#b91c1c" : "#1d4ed8", rejected ? 0.9 : 0.9);
      }
      renderDetails();
    }
    function renderDetails() {
      if (state.selected >= data.pairs.length) state.selected = 0;
      const pair = data.pairs[state.selected];
      details.textContent = [
        ...data.metadata_lines,
        "",
        `shown_pairs:    ${data.pairs.length}`,
        `status_counts:  ${JSON.stringify(data.status_counts)}`,
        "",
        pair ? `status:         ${pair.status}` : "status:         none",
        pair ? `reason:         ${pair.reason}` : "",
        pair ? `final_grasp:    ${pair.final.grasp_id}` : "",
        pair ? `transfer_grasp: ${pair.transfer.grasp_id}` : "",
        pair ? `pair_score:     ${pair.score}` : "",
      ].filter((line) => line !== "").join("\\n");
    }
    function selectDelta(delta) {
      if (!data.pairs.length) return;
      state.selected = (state.selected + delta + data.pairs.length) % data.pairs.length;
      draw();
      renderList();
    }
    document.getElementById("prevBtn").addEventListener("click", () => selectDelta(-1));
    document.getElementById("nextBtn").addEventListener("click", () => selectDelta(1));
    document.getElementById("resetBtn").addEventListener("click", () => { Object.assign(state, { yaw: -0.72, pitch: 0.52, zoom: 1, panX: 0, panY: 0 }); draw(); });
    document.getElementById("meshBtn").addEventListener("click", (event) => { state.showMesh = !state.showMesh; event.target.textContent = state.showMesh ? "Mesh On" : "Mesh Off"; draw(); });
    window.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") { event.preventDefault(); selectDelta(-1); }
      if (event.key === "ArrowRight" || event.key === "ArrowDown") { event.preventDefault(); selectDelta(1); }
    });
    canvas.addEventListener("pointerdown", (event) => {
      state.dragging = true;
      state.dragMode = event.button === 1 || event.shiftKey ? "pan" : "rotate";
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
      canvas.style.cursor = state.dragMode === "pan" ? "move" : "grabbing";
    });
    canvas.addEventListener("pointerup", (event) => { state.dragging = false; canvas.releasePointerCapture(event.pointerId); canvas.style.cursor = "grab"; });
    canvas.addEventListener("pointercancel", () => { state.dragging = false; canvas.style.cursor = "grab"; });
    canvas.addEventListener("pointermove", (event) => {
      if (!state.dragging) return;
      const dx = event.clientX - state.lastX, dy = event.clientY - state.lastY;
      state.lastX = event.clientX; state.lastY = event.clientY;
      if (state.dragMode === "pan") { state.panX += dx; state.panY += dy; }
      else { state.yaw += dx * 0.01; state.pitch -= dy * 0.01; }
      draw();
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      state.zoom = Math.max(0.25, Math.min(5, state.zoom * (event.deltaY < 0 ? 1.08 : 1 / 1.08)));
      draw();
    }, { passive: false });
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    window.addEventListener("resize", resize);
    renderList();
    resize();
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(document.replace("__DATA_JSON__", data_json), encoding="utf-8")
    return status_counts


def _write_failed_grasps_html(
    output_html: Path,
    *,
    target: TargetSpec,
    orientation: StableOrientation,
    status: str,
    stage1,
    planning: PlanningConfig,
    mesh_scale: float,
    stage2=None,
    error: str = "",
    limit: int = FAILED_GRASP_HTML_LIMIT,
    stage1_failure_statuses: list[CandidateStatus] | None = None,
    stage1_failure_obstacle_parts: list[AssemblyObstaclePart] | None = None,
) -> tuple[int, dict[str, int]]:
    obstacle_mesh_local = None
    display_pose = orientation.object_pose_world
    ground_plane = ground_plane_overlay_obj(stage1.target_mesh_local, object_pose_world=display_pose, enabled=True)
    failure_stage = "stage2_floor"
    obstacle_paths: list[str] = []
    stage1_rejected_count = 0
    stage1_pass_count = 0

    if stage1_failure_statuses is None or stage1_failure_obstacle_parts is None:
        candidate_statuses, obstacle_parts = _stage1_assembly_failure_statuses(
            target=target,
            stage1=stage1,
            planning=planning,
            mesh_scale=mesh_scale,
            limit=limit,
        )
    else:
        candidate_statuses = stage1_failure_statuses
        obstacle_parts = stage1_failure_obstacle_parts
    if candidate_statuses:
        stage1_pass_example_limit = min(FAILED_GRASP_STAGE1_PASS_EXAMPLE_LIMIT, max(0, int(limit)))
        stage1_rejected_limit = max(0, int(limit) - stage1_pass_example_limit)
        selected_stage1_failures = candidate_statuses[:stage1_rejected_limit]
        stage1_pass_statuses = _stage1_passed_stage2_failure_statuses(
            stage2,
            limit=max(0, int(limit) - len(selected_stage1_failures)),
        )
        candidate_statuses = selected_stage1_failures + stage1_pass_statuses
        stage1_rejected_count = len(selected_stage1_failures)
        stage1_pass_count = len(stage1_pass_statuses)
        obstacle_paths = [part.mesh_path for part in obstacle_parts]
        failure_stage = "stage1_assembly_preferred_with_stage1_pass_examples"
        if stage1.obstacle_mesh_world is not None:
            obstacle_mesh_local = _mesh_in_object_frame(stage1.obstacle_mesh_world, stage1.target_pose_in_obj_world)
    elif stage2 is not None:
        candidate_statuses = _stage1_passed_stage2_failure_statuses(stage2, limit=limit)
        stage1_pass_count = len(candidate_statuses)
        display_pose = stage2.pickup_pose_world
        ground_plane = ground_plane_overlay_obj(stage2.mesh_local, object_pose_world=display_pose, enabled=True)
        failure_stage = "stage2_floor_stage1_pass_examples"
    else:
        top_candidates = sorted(stage1.bundle.candidates, key=lambda candidate: (-_candidate_score(candidate), candidate.grasp_id))
        candidate_statuses = _unique_contact_statuses(
            [
                CandidateStatus(grasp=candidate, status="rejected", reason=f"orientation_error: {error or 'unknown'}")
                for candidate in top_candidates
            ],
            limit=limit,
        )
        failure_stage = "orientation_error"

    constraint_counts = _constraint_counts(candidate_statuses)
    metadata_lines = [
        f"target_mesh:      {target.target_mesh_path}",
        f"assembly:         {target.assembly}",
        f"part_id:          {target.part_id}",
        f"precedence_plan:  {target.precedence_plan_path or 'none'}",
        f"assembled_before: {list(target.already_assembled_part_ids)}",
        f"obstacle_paths:   {list(target.assembly_obstacle_paths) if target.assembly_obstacle_paths is not None else 'glob'}",
        f"pre_insert_path:  {target.pre_insertion_poses_path or 'none'}",
        f"sweep_vector_m:   {target.insertion_sweep_vector_m}",
        f"sweep_distance_m: {target.insertion_sweep_distance_m:.6f}",
        f"orientation:      {orientation.orientation_id}",
        f"benchmark_status: {status}",
        f"failure_stage:    {failure_stage}",
        f"display_frame:    {FAILED_GRASP_DISPLAY_FRAME}",
        f"displayed_grasps: {len(candidate_statuses)} of max {limit}",
        f"stage1_rejected_displayed: {stage1_rejected_count}",
        f"stage1_pass_displayed:     {stage1_pass_count}",
        f"stage1_feasible:  {len(stage1.bundle.candidates)}",
        f"floor_clearance:  {planning.floor_clearance_margin_m:.6f} m",
        (
            "selection_policy: prefer unique stage-1 assembly failures, plus a comparison slice "
            "of stage-1-passed/stage-2-failed grasps"
        ),
        "render_limits:    target_edges<=8000, obstacle_edges<=8000",
        f"constraints:      {constraint_counts}",
    ]
    if obstacle_paths:
        metadata_lines.append(f"obstacle_parts:   {obstacle_paths}")
    if error:
        metadata_lines.append(f"error:            {error}")

    write_debug_html(
        title=f"Failed Grasps: {target.assembly}/{target.part_id} {orientation.orientation_id}",
        subtitle=(
            "Top failed grasp candidates rendered in the selected floor pose; "
            "the unsatisfied constraint is encoded in each candidate reason."
        ),
        mesh_local=stage1.target_mesh_local,
        candidate_statuses=candidate_statuses,
        output_html=output_html,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
        ground_plane=ground_plane,
        obstacle_mesh_local=obstacle_mesh_local,
        metadata_lines=metadata_lines,
        display_object_pose_world=display_pose,
        max_mesh_edges=8000,
        max_obstacle_edges=8000,
    )
    return len(candidate_statuses), constraint_counts


def _write_part_orientations_html(
    output_html: Path,
    *,
    target: TargetSpec,
    stage1,
    orientation_frames: list[dict[str, object]],
) -> None:
    mesh_local = stage1.target_mesh_local
    data = {
        "title": f"{target.assembly}/{target.part_id} Stable Orientations",
        "subtitle": "One frame per stable orientation. The grasp shown is the highest-scoring direct stage-2 grasp when one exists.",
        "target_mesh_path": target.target_mesh_path,
        "precedence_plan_path": target.precedence_plan_path,
        "assembly_obstacle_paths": None
        if target.assembly_obstacle_paths is None
        else list(target.assembly_obstacle_paths),
        "selected_assembly_order": list(target.selected_assembly_order),
        "already_assembled_part_ids": list(target.already_assembled_part_ids),
        "pre_insertion_poses_path": target.pre_insertion_poses_path,
        "pre_insertion_role": target.pre_insertion_role,
        "insertion_sweep_vector_m": None
        if target.insertion_sweep_vector_m is None
        else list(target.insertion_sweep_vector_m),
        "insertion_sweep_distance_m": target.insertion_sweep_distance_m,
        "final_to_pre_insertion_translation_m": None
        if target.final_to_pre_insertion_translation_m is None
        else list(target.final_to_pre_insertion_translation_m),
        "vertices_local": [[float(v) for v in vertex] for vertex in np.asarray(mesh_local.vertices_obj, dtype=float).tolist()],
        "faces": [[int(value) for value in face] for face in np.asarray(mesh_local.faces, dtype=np.int64).tolist()],
        "edges": unique_edges(mesh_local.faces),
        "frames": orientation_frames,
    }
    data_json = json.dumps(data, indent=2)
    document = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stable Orientation Winners</title>
  <style>
    :root {
      --bg: #f6f4ee;
      --panel: #fffdf8;
      --ink: #1f2522;
      --muted: #68716c;
      --line: #d9d4c7;
      --mesh: #2f6f5e;
      --floor: #2563eb;
      --success: #15803d;
      --fallback: #1d4ed8;
      --fail: #b91c1c;
      --franka: #d97706;
      --hand: #8f5a12;
      --contact-a: #c8452d;
      --contact-b: #1f7c60;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    .layout { display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: 100vh; }
    aside { border-right: 1px solid var(--line); background: var(--panel); padding: 20px; overflow: auto; }
    main { padding: 18px; overflow: auto; }
    h1 { margin: 0 0 8px; font-size: 25px; line-height: 1.15; }
    .subtitle { margin: 0 0 16px; color: var(--muted); font-size: 14px; line-height: 1.45; }
    .controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 14px; }
    button { border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 8px; padding: 10px 12px; font: inherit; cursor: pointer; }
    button:hover { border-color: var(--mesh); }
    .orientation-list { display: grid; gap: 8px; }
    .orientation-item { width: 100%; text-align: left; border-radius: 8px; }
    .orientation-item.active { border-color: var(--mesh); box-shadow: 0 0 0 2px rgba(47,111,94,0.14); }
    .item-title { display: flex; justify-content: space-between; gap: 8px; font-weight: 700; }
    .item-meta { margin-top: 5px; color: var(--muted); font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.4; }
    .status { border-radius: 999px; padding: 2px 7px; font-size: 12px; white-space: nowrap; }
    .direct_success { background: #dcfce7; color: var(--success); }
    .fallback_success { background: #dbeafe; color: var(--fallback); }
    .failed { background: #fee2e2; color: var(--fail); }
    .card { border: 1px solid var(--line); background: rgba(255,253,248,0.96); border-radius: 8px; padding: 14px; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.6fr); gap: 16px; align-items: start; }
    #scene { width: 100%; aspect-ratio: 1.45 / 1; display: block; border-radius: 8px; background: linear-gradient(180deg, #ffffff, #ebe7dc); }
    .kv { white-space: pre-wrap; font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.55; margin: 0; }
    .link-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .link-list a { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 6px 8px; font-size: 13px; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; color: var(--muted); font-size: 13px; }
    .legend span { display: inline-flex; align-items: center; gap: 7px; }
    .swatch { width: 13px; height: 13px; border-radius: 999px; display: inline-block; }
    a { color: #2563eb; text-decoration: none; margin-right: 10px; }
    a:hover { text-decoration: underline; }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1 id="title"></h1>
      <p id="subtitle" class="subtitle"></p>
      <div class="controls">
        <button id="prevBtn" type="button">Previous</button>
        <button id="nextBtn" type="button">Next</button>
        <button id="solidBtn" type="button">Solid Mesh</button>
        <button id="resetBtn" type="button">Reset View</button>
      </div>
      <div id="orientationList" class="orientation-list"></div>
    </aside>
    <main>
      <div class="grid">
        <section class="card">
          <svg id="scene" viewBox="0 0 1100 760"></svg>
          <div class="legend">
            <span><i class="swatch" style="background: var(--mesh)"></i>Target mesh</span>
            <span><i class="swatch" style="background: var(--floor)"></i>Ground</span>
            <span><i class="swatch" style="background: var(--franka)"></i>Finger boxes</span>
            <span><i class="swatch" style="background: var(--hand)"></i>Hand mesh</span>
            <span><i class="swatch" style="background: var(--contact-a)"></i>Contact A</span>
            <span><i class="swatch" style="background: var(--contact-b)"></i>Contact B</span>
          </div>
        </section>
        <section class="card">
          <pre id="details" class="kv"></pre>
          <div id="linkList" class="link-list"></div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const scene = document.getElementById("scene");
    const details = document.getElementById("details");
    const linkList = document.getElementById("linkList");
    const list = document.getElementById("orientationList");
    document.getElementById("title").textContent = data.title;
    document.getElementById("subtitle").textContent = data.subtitle;
    const initialView = { yaw: -0.72, pitch: 0.52, zoom: 1.0, panX: 0, panY: 0 };
    const state = { index: 0, solid: false, dragging: false, dragMode: "rotate", pointerId: null, lastX: 0, lastY: 0, ...initialView };
    function statusClass(status) {
      if (status === "direct_success") return "direct_success";
      if (status === "fallback_success") return "fallback_success";
      return "failed";
    }
    function currentFrame() {
      return data.frames[state.index] || null;
    }
    function fmt(value, digits = 3) {
      if (value === null || value === undefined || value === "") return "n/a";
      return Number(value).toFixed(digits);
    }
    function allPoints(frame) {
      const points = [...frame.mesh_vertices_world, ...frame.floor_world];
      const grasp = frame.best_grasp;
      if (grasp) {
        points.push(
          grasp.grasp_position_obj,
          grasp.contact_point_a_obj,
          grasp.contact_point_b_obj,
          ...grasp.franka_hand_vertices_obj,
          ...grasp.franka_left_boxes.flatMap((box) => box.corners),
          ...grasp.franka_right_boxes.flatMap((box) => box.corners),
        );
      }
      return points;
    }
    function bounds(points) {
      return points.reduce((acc, point) => {
        point.forEach((value, axis) => {
          acc.min[axis] = Math.min(acc.min[axis], value);
          acc.max[axis] = Math.max(acc.max[axis], value);
        });
        return acc;
      }, { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
    }
    function rotate(point, center) {
      const shifted = point.map((value, axis) => value - center[axis]);
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];
      return [x1, cp * y1 + sp * z1, -sp * y1 + cp * z1];
    }
    function projection(frame) {
      const b = bounds(allPoints(frame));
      const center = b.min.map((value, axis) => 0.5 * (value + b.max[axis]));
      const extent = Math.max(...b.max.map((value, axis) => value - b.min[axis]), 0.08);
      return { center, scale: 560 / extent };
    }
    function project(point, projectionBounds) {
      const [x, y, z] = rotate(point, projectionBounds.center);
      const scale = projectionBounds.scale * state.zoom;
      return { x: 550 + state.panX + x * scale, y: 380 + state.panY - y * scale, depth: z };
    }
    function add(tag, attrs) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      scene.appendChild(node);
      return node;
    }
    function line(a, b, options, projectionBounds) {
      const pa = project(a, projectionBounds), pb = project(b, projectionBounds);
      add("line", { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, stroke: options.stroke || "#333", "stroke-width": options.width || 2, "stroke-opacity": options.opacity ?? 1, "stroke-dasharray": options.dash || "" });
    }
    function point(p, options, projectionBounds) {
      const pp = project(p, projectionBounds);
      add("circle", { cx: pp.x, cy: pp.y, r: options.radius || 5, fill: options.fill || "#000", stroke: "#fff", "stroke-width": options.strokeWidth || 1.5, "fill-opacity": options.opacity ?? 1 });
    }
    function polygon(points, options, projectionBounds) {
      const projected = points.map((p) => project(p, projectionBounds));
      add("polygon", { points: projected.map((p) => `${p.x},${p.y}`).join(" "), fill: options.fill || "none", "fill-opacity": options.fillOpacity ?? 1, stroke: options.stroke || "none", "stroke-width": options.strokeWidth || 1, "stroke-opacity": options.strokeOpacity ?? 1 });
    }
    function label(p, text, color, projectionBounds) {
      const pp = project(p, projectionBounds);
      const node = add("text", { x: pp.x + 8, y: pp.y - 8, fill: color, "font-size": 14, "font-family": "IBM Plex Mono, monospace", "font-weight": 700 });
      node.textContent = text;
    }
    function drawMesh(frame, projectionBounds) {
      if (state.solid) {
        data.faces.map((face) => {
          const points = face.map((index) => frame.mesh_vertices_world[index]);
          const rotated = points.map((p) => rotate(p, projectionBounds.center));
          const edgeA = rotated[1].map((value, axis) => value - rotated[0][axis]);
          const edgeB = rotated[2].map((value, axis) => value - rotated[0][axis]);
          const normal = [
            edgeA[1] * edgeB[2] - edgeA[2] * edgeB[1],
            edgeA[2] * edgeB[0] - edgeA[0] * edgeB[2],
            edgeA[0] * edgeB[1] - edgeA[1] * edgeB[0],
          ];
          const depth = rotated.reduce((sum, p) => sum + p[2], 0) / rotated.length;
          return { points, normal, depth };
        }).filter((face) => face.normal[2] > 0).sort((a, b) => a.depth - b.depth).forEach((face) => {
          polygon(face.points, { fill: "#7aa392", fillOpacity: 0.86, stroke: "#2f6f5e", strokeWidth: 0.8, strokeOpacity: 0.45 }, projectionBounds);
        });
      }
      data.edges.forEach(([a, b]) => line(frame.mesh_vertices_world[a], frame.mesh_vertices_world[b], { stroke: "#2f6f5e", width: 1.5, opacity: 0.82 }, projectionBounds));
    }
    function drawBox(corners, color, projectionBounds) {
      [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([a, b]) => {
        line(corners[a], corners[b], { stroke: color, width: 1.6, opacity: 0.78 }, projectionBounds);
      });
    }
    function drawHand(grasp, projectionBounds) {
      grasp.franka_hand_faces.forEach((face) => {
        line(grasp.franka_hand_vertices_obj[face[0]], grasp.franka_hand_vertices_obj[face[1]], { stroke: "#8f5a12", width: 0.9, opacity: 0.28 }, projectionBounds);
        line(grasp.franka_hand_vertices_obj[face[1]], grasp.franka_hand_vertices_obj[face[2]], { stroke: "#8f5a12", width: 0.9, opacity: 0.28 }, projectionBounds);
        line(grasp.franka_hand_vertices_obj[face[2]], grasp.franka_hand_vertices_obj[face[0]], { stroke: "#8f5a12", width: 0.9, opacity: 0.28 }, projectionBounds);
      });
    }
    function drawGrasp(grasp, projectionBounds) {
      grasp.franka_left_boxes.forEach((box) => drawBox(box.corners, "#d97706", projectionBounds));
      grasp.franka_right_boxes.forEach((box) => drawBox(box.corners, "#d97706", projectionBounds));
      drawHand(grasp, projectionBounds);
      line(grasp.contact_point_a_obj, grasp.contact_point_b_obj, { stroke: "#15803d", width: 3, opacity: 0.95 }, projectionBounds);
      point(grasp.grasp_position_obj, { fill: "#15803d", radius: 7 }, projectionBounds);
      point(grasp.contact_point_a_obj, { fill: "#c8452d", radius: 6 }, projectionBounds);
      point(grasp.contact_point_b_obj, { fill: "#1f7c60", radius: 6 }, projectionBounds);
      label(grasp.grasp_position_obj, grasp.grasp_id, "#15803d", projectionBounds);
    }
    function renderList() {
      list.replaceChildren();
      data.frames.forEach((frame, index) => {
        const grasp = frame.best_grasp;
        const button = document.createElement("button");
        button.type = "button";
        button.className = `orientation-item${index === state.index ? " active" : ""}`;
        button.innerHTML = `
          <div class="item-title">
            <span>${frame.orientation.orientation_id}</span>
            <span class="status ${statusClass(frame.status)}">${frame.status}</span>
          </div>
          <div class="item-meta">
            feasible=${frame.stage2_ground_feasible_count} tilt=${fmt(frame.orientation.max_stable_tilt_deg, 1)}deg<br>
            best=${grasp ? `${grasp.grasp_id} score=${fmt(grasp.score, 3)}` : "none"}
          </div>
        `;
        button.addEventListener("click", () => { state.index = index; render(); });
        list.appendChild(button);
      });
    }
    function renderDetails(frame) {
      const grasp = frame.best_grasp;
      linkList.replaceChildren();
      Object.entries(frame.links || {}).filter(([, value]) => value).forEach(([key, value]) => {
        const anchor = document.createElement("a");
        anchor.href = value;
        anchor.textContent = key;
        linkList.appendChild(anchor);
      });
      details.textContent = [
        `target:             ${data.target_mesh_path}`,
        `precedence_plan:    ${data.precedence_plan_path || "none"}`,
        `assembled_before:   ${JSON.stringify(data.already_assembled_part_ids || [])}`,
        `obstacle_paths:     ${JSON.stringify(data.assembly_obstacle_paths === null ? "glob" : (data.assembly_obstacle_paths || []))}`,
        `pre_insert_path:    ${data.pre_insertion_poses_path || "none"}`,
        `sweep_vector_m:     ${JSON.stringify(data.insertion_sweep_vector_m || null)}`,
        `sweep_distance_m:   ${fmt(data.insertion_sweep_distance_m || 0, 6)}`,
        `orientation:        ${frame.orientation.orientation_id}`,
        `status:             ${frame.status}`,
        `stage1_feasible:    ${frame.stage1_assembly_feasible_count}`,
        `stage2_feasible:    ${frame.stage2_ground_feasible_count}`,
        `normal_obj:         (${frame.orientation.normal_obj.map((v) => fmt(v, 4)).join(", ")})`,
        `support_area_m2:    ${fmt(frame.orientation.area_m2, 6)}`,
        `stability_margin_m: ${fmt(frame.orientation.stability_margin_m, 6)}`,
        `com_height_m:       ${fmt(frame.orientation.com_height_m, 6)}`,
        `max_stable_tilt:    ${fmt(frame.orientation.max_stable_tilt_deg, 3)} deg`,
        `com_method:         ${frame.orientation.com_method}`,
        "",
        grasp ? `best_grasp:        ${grasp.grasp_id}` : "best_grasp:        none",
        grasp ? `best_score:        ${fmt(grasp.score, 6)}` : "",
        grasp ? `jaw_width_m:       ${fmt(grasp.jaw_width, 6)}` : "",
        grasp ? `roll_angle_rad:    ${fmt(grasp.roll_angle_rad, 6)}` : "",
        grasp && grasp.score_components ? `object_score:      ${fmt(grasp.score_components.object_score, 6)}` : "",
        grasp && grasp.score_components ? `top_down_score:    ${fmt(grasp.score_components.top_down_approach, 6)}` : "",
        "",
        frame.fallback ? `fallback_final:    ${frame.fallback.final_grasp_id}` : "",
        frame.error ? `error:             ${frame.error}` : "",
      ].filter((line) => line !== "").join("\\n");
    }
    function renderScene(frame) {
      scene.replaceChildren();
      const b = projection(frame);
      polygon(frame.floor_world, { fill: "#2563eb", fillOpacity: 0.13, stroke: "#2563eb", strokeWidth: 1.8, strokeOpacity: 0.85 }, b);
      for (let i = 0; i < frame.floor_world.length; i += 1) {
        line(frame.floor_world[i], frame.floor_world[(i + 1) % frame.floor_world.length], { stroke: "#2563eb", width: 1.8, opacity: 0.85, dash: "8 5" }, b);
      }
      drawMesh(frame, b);
      if (frame.best_grasp) {
        drawGrasp(frame.best_grasp, b);
      } else {
        label(frame.mesh_vertices_world[0], "no direct stage-2 grasp", "#b91c1c", b);
      }
    }
    function render() {
      const frame = currentFrame();
      if (!frame) {
        details.textContent = "No stable orientations.";
        return;
      }
      renderList();
      renderScene(frame);
      renderDetails(frame);
    }
    function step(delta) {
      if (!data.frames.length) return;
      state.index = (state.index + delta + data.frames.length) % data.frames.length;
      render();
    }
    document.getElementById("prevBtn").addEventListener("click", () => step(-1));
    document.getElementById("nextBtn").addEventListener("click", () => step(1));
    document.getElementById("solidBtn").addEventListener("click", () => {
      state.solid = !state.solid;
      document.getElementById("solidBtn").textContent = state.solid ? "Wireframe Mesh" : "Solid Mesh";
      render();
    });
    document.getElementById("resetBtn").addEventListener("click", () => {
      Object.assign(state, initialView);
      render();
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") { event.preventDefault(); step(-1); }
      if (event.key === "ArrowRight" || event.key === "ArrowDown") { event.preventDefault(); step(1); }
    });
    scene.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 && event.button !== 1 && event.button !== 2) return;
      event.preventDefault();
      state.dragging = true;
      state.dragMode = event.button === 1 || event.button === 2 || event.shiftKey ? "pan" : "rotate";
      state.pointerId = event.pointerId;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      scene.setPointerCapture(event.pointerId);
    });
    function stopDragging() {
      state.dragging = false;
      state.pointerId = null;
    }
    scene.addEventListener("pointerup", (event) => { if (state.pointerId === event.pointerId) stopDragging(); });
    scene.addEventListener("pointercancel", stopDragging);
    scene.addEventListener("pointermove", (event) => {
      if (!state.dragging || (state.pointerId !== null && state.pointerId !== event.pointerId)) return;
      const dx = event.clientX - state.lastX;
      const dy = event.clientY - state.lastY;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      if (state.dragMode === "pan") {
        state.panX += dx;
        state.panY += dy;
      } else {
        state.yaw += dx * 0.01;
        state.pitch -= dy * 0.01;
      }
      render();
    });
    scene.addEventListener("wheel", (event) => {
      event.preventDefault();
      state.zoom = Math.max(0.3, Math.min(5.0, state.zoom * (event.deltaY < 0 ? 1.08 : 1 / 1.08)));
      render();
    }, { passive: false });
    scene.addEventListener("contextmenu", (event) => event.preventDefault());
    render();
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(document.replace("__DATA_JSON__", data_json), encoding="utf-8")


def _write_orientation_details(
    output_path: Path,
    *,
    target: TargetSpec,
    orientation: StableOrientation,
    status: str,
    stage1,
    stage2=None,
    fallback_summary: dict[str, object] | None = None,
    handover_summary: dict[str, object] | None = None,
    error: str = "",
) -> None:
    payload = {
        "target": {
            "assembly": target.assembly,
            "part_id": target.part_id,
            "target_mesh_path": target.target_mesh_path,
            "assembly_glob": target.assembly_glob,
            "assembly_obstacle_paths": None
            if target.assembly_obstacle_paths is None
            else list(target.assembly_obstacle_paths),
            "precedence_plan_path": target.precedence_plan_path,
            "selected_assembly_order": list(target.selected_assembly_order),
            "already_assembled_part_ids": list(target.already_assembled_part_ids),
            "pre_insertion_poses_path": target.pre_insertion_poses_path,
            "pre_insertion_role": target.pre_insertion_role,
            "insertion_sweep_vector_m": None
            if target.insertion_sweep_vector_m is None
            else list(target.insertion_sweep_vector_m),
            "insertion_sweep_distance_m": target.insertion_sweep_distance_m,
            "final_to_pre_insertion_translation_m": None
            if target.final_to_pre_insertion_translation_m is None
            else list(target.final_to_pre_insertion_translation_m),
        },
        "orientation": stable_orientation_payload(orientation),
        "status": status,
        "stage1": {
            "raw_candidate_count": stage1.raw_candidate_count,
            "assembly_feasible_count": len(stage1.bundle.candidates),
            "cache_hit": bool((stage1.bundle.metadata or {}).get("stage1_cache_hit")),
            "cache_path": (stage1.bundle.metadata or {}).get("stage1_cache_path"),
        },
        "stage2": None
        if stage2 is None
        else {
            "ground_feasible_count": len(stage2.accepted),
            "reason_counts": _reason_counts(stage2.statuses),
            "best_direct_grasp": _best_candidate_payload(stage2.accepted),
        },
        "fallback": fallback_summary,
        "handover_fallback": handover_summary,
        "error": error,
    }
    _write_json(output_path, payload)


def _apply_cli_overrides(payload: dict[str, object], args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    effective = deepcopy(payload)
    benchmark = dict(effective.get("benchmark", {}))
    benchmark["output_dir"] = str(output_dir)
    effective["benchmark"] = benchmark
    if args.robust_tilt_deg is not None:
        stable = dict(effective.get("stable_orientations", {}))
        stable["robust_tilt_deg"] = float(args.robust_tilt_deg)
        effective["stable_orientations"] = stable
    planning = dict(effective.get("planning", {}))
    if args.no_stage1_cache:
        planning["stage1_cache_enabled"] = False
    if args.skip_stage1_collision_checks:
        planning["skip_stage1_collision_checks"] = True
    effective["planning"] = planning
    if args.fallback_enabled is not None:
        fallback = dict(effective.get("fallback", {}))
        fallback["enabled"] = bool(args.fallback_enabled)
        effective["fallback"] = fallback
        handover_fallback = dict(effective.get("handover_fallback", {}))
        handover_fallback["enabled"] = bool(args.fallback_enabled)
        effective["handover_fallback"] = handover_fallback
    return effective


def _benchmark_one_target(
    *,
    target: TargetSpec,
    planning: PlanningConfig,
    stable_config: StableOrientationConfig,
    fallback_config: FallbackBenchmarkConfig,
    handover_config: HandoverFallbackBenchmarkConfig,
    mesh_scale: float,
    output_dir: Path,
    target_index: int,
    target_count: int,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object] | None]:
    part_dir = output_dir / "parts" / _safe_id(target.assembly) / _safe_id(target.part_id)
    stage1_dir = part_dir / "stage1"
    orientations_dir = part_dir / "orientations"
    part_record: dict[str, object] = {
        "assembly": target.assembly,
        "part_id": target.part_id,
        "target_mesh_path": target.target_mesh_path,
        "assembly_glob": target.assembly_glob,
        "assembly_obstacle_paths": None
        if target.assembly_obstacle_paths is None
        else list(target.assembly_obstacle_paths),
        "precedence_plan_path": target.precedence_plan_path,
        "selected_assembly_order": list(target.selected_assembly_order),
        "already_assembled_part_ids": list(target.already_assembled_part_ids),
        "pre_insertion_poses_path": target.pre_insertion_poses_path,
        "pre_insertion_role": target.pre_insertion_role,
        "insertion_sweep_vector_m": None
        if target.insertion_sweep_vector_m is None
        else list(target.insertion_sweep_vector_m),
        "insertion_sweep_distance_m": target.insertion_sweep_distance_m,
        "final_to_pre_insertion_translation_m": None
        if target.final_to_pre_insertion_translation_m is None
        else list(target.final_to_pre_insertion_translation_m),
        "status": "pending",
    }
    rows: list[dict[str, object]] = []
    orientation_frames: list[dict[str, object]] = []
    stage1_failure_debug: tuple[list[CandidateStatus], list[AssemblyObstaclePart]] | None = None
    geometry = GeometryConfig(
        target_mesh_path=target.target_mesh_path,
        mesh_scale=mesh_scale,
        assembly_glob=target.assembly_glob,
        assembly_obstacle_paths=target.assembly_obstacle_paths,
        assembly_obstacle_sweep_vector_m=target.insertion_sweep_vector_m,
        assembly_obstacle_metadata=_target_assembly_obstacle_metadata(target),
    )

    def _stage1_failure_debug() -> tuple[list[CandidateStatus], list[AssemblyObstaclePart]]:
        nonlocal stage1_failure_debug
        if stage1_failure_debug is None:
            stage1_failure_debug = _stage1_assembly_failure_statuses(
                target=target,
                stage1=stage1,
                planning=planning,
                mesh_scale=mesh_scale,
                limit=FAILED_GRASP_HTML_LIMIT,
            )
        return stage1_failure_debug

    try:
        target_mesh_obj_world = load_asset_mesh(target.target_mesh_path, scale=mesh_scale)
        target_mesh_local, target_pose_in_obj_world = canonicalize_target_mesh(target_mesh_obj_world)
        orientation_result = enumerate_stable_orientations(target_mesh_local, stable_config)
        stable_json = part_dir / "stable_orientations.json"
        _write_json(stable_json, stable_orientation_result_payload(orientation_result))
        upright_approach_axes = _upright_approach_axes_obj(
            source_frame_pose_obj_world=target_pose_in_obj_world,
            orientations=orientation_result.orientations,
        )
        part_record.update(
            {
                "stable_orientations_json": str(stable_json),
                "stable_orientation_count": len(orientation_result.orientations),
                "rejected_orientation_candidate_count": len(orientation_result.rejected_candidates),
                "com_method": orientation_result.com_method,
                "upright_approach_axis_count": len(upright_approach_axes),
            }
        )
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        part_record.update({"status": "orientation_generation_error", "error": error})
        print(f"  orientation_generation_error: {error}", flush=True)
        return part_record, rows, None

    print(
        f"[{target_index}/{target_count}] {target.target_mesh_path}: generating stage 1 "
        f"with {len(upright_approach_axes)} upright roll reference axes.",
        flush=True,
    )
    try:
        stage1 = generate_stage1_result(
            geometry=geometry,
            planning=planning,
            upright_approach_axes_obj=upright_approach_axes,
        )
        stage1_json = stage1_dir / "grasps.json"
        stage1_html = stage1_dir / "grasps.html"
        write_stage1_artifacts(stage1, geometry=geometry, planning=planning, output_json=stage1_json, output_html=stage1_html)
        _write_json(stage1_dir / "raw_grasps.json", _raw_stage1_payload(stage1, target))
        part_frame_html = part_dir / "part_frame.html"
        write_part_frame_debug_html(input_json=stage1_json, output_html=part_frame_html)
        part_record.update(
            {
                "stage1_json": str(stage1_json),
                "stage1_html": str(stage1_html),
                "part_frame_html": str(part_frame_html),
                "stage1_raw_count": stage1.raw_candidate_count,
                "stage1_assembly_feasible_count": len(stage1.bundle.candidates),
                "stage1_cache_hit": bool((stage1.bundle.metadata or {}).get("stage1_cache_hit")),
                "upright_approach_axis_count": len(upright_approach_axes),
            }
        )
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        part_record.update({"status": "stage1_error", "error": error})
        print(f"  stage1_error: {error}", flush=True)
        return part_record, rows, None

    if not orientation_result.orientations:
        part_record.update({"status": "no_stable_orientations"})
        print("  no_stable_orientations", flush=True)
        return part_record, rows, _part_browser_payload(target=target, stage1=stage1, orientation_frames=orientation_frames)

    for orientation in orientation_result.orientations:
        orientation_dir = orientations_dir / orientation.orientation_id
        stage2_json = orientation_dir / "stage2.json"
        stage2_html = orientation_dir / "stage2.html"
        fallback_json = orientation_dir / "fallback_plan.json"
        fallback_html = orientation_dir / "fallback_plan.html"
        handover_json = orientation_dir / "handover_fallback_plan.json"
        handover_html = orientation_dir / "handover_fallback_pairs.html"
        print(f"  {orientation.orientation_id}: stage 2.", flush=True)
        try:
            stage2 = recheck_stage2_result(
                bundle=stage1.bundle,
                pickup_spec=None,
                planning=planning,
                object_pose_world=orientation.object_pose_world,
            )
            write_stage2_artifacts(stage2, planning=planning, output_json=stage2_json, output_html=stage2_html)
            plan = None
            handover_result = None
            if fallback_config.enabled and len(stage1.bundle.candidates) > 0 and not stage2.accepted:
                plan = plan_mujoco_regrasp_fallback(
                    stage1=stage1,
                    direct_stage2=stage2,
                    planning=planning,
                    force=False,
                    staging_xy_world=(
                        float(orientation.object_pose_world.position_world[0]),
                        float(orientation.object_pose_world.position_world[1]),
                    ),
                    staging_xy_offsets_m=fallback_config.staging_xy_offsets_m,
                    yaw_angles_deg=fallback_config.yaw_angles_deg,
                    max_orientations=fallback_config.max_orientations,
                    max_placement_options=fallback_config.max_placement_options,
                    min_facet_area_m2=fallback_config.min_facet_area_m2,
                    stability_margin_m=fallback_config.stability_margin_m,
                    coplanar_tolerance_m=fallback_config.coplanar_tolerance_m,
                )
                if plan is not None:
                    write_mujoco_regrasp_plan(plan, fallback_json, input_stage2_json=stage2_json)
                    write_mujoco_regrasp_debug_html(
                        plan=plan,
                        stage1=stage1,
                        planning=planning,
                        output_html=fallback_html,
                    )
            fallback_summary = _fallback_summary(plan)
            if (
                handover_config.enabled
                and len(stage1.bundle.candidates) > 0
                and not stage2.accepted
                and fallback_summary is None
            ):
                handover_result = plan_handover_fallback(
                    stage1=stage1,
                    direct_stage2=stage2,
                    planning=planning,
                    max_final_candidates=handover_config.max_final_candidates,
                    max_transfer_candidates=handover_config.max_transfer_candidates,
                    max_pair_checks=handover_config.max_pair_checks,
                    max_accepted_pairs=handover_config.max_accepted_pairs,
                    max_rejected_pairs=handover_config.max_rejected_pairs,
                    transfer_floor_clearance_margin_m=handover_config.transfer_floor_clearance_margin_m,
                )
                if handover_result is not None:
                    write_handover_fallback_result(
                        handover_result,
                        handover_json,
                        input_stage2_json=stage2_json,
                    )
                    _write_handover_grasp_pairs_html(
                        handover_html,
                        target=target,
                        orientation=orientation,
                        status="handover_fallback_success"
                        if handover_result.selected_pair is not None
                        else "handover_fallback_failed",
                        stage1=stage1,
                        planning=planning,
                        result=handover_result,
                    )
            handover_summary = _handover_summary(handover_result)
            status = _status_for_orientation(
                stage1_count=len(stage1.bundle.candidates),
                stage2_count=len(stage2.accepted),
                fallback_found=fallback_summary is not None,
                handover_found=handover_summary is not None,
                fallback_enabled=fallback_config.enabled,
                handover_enabled=handover_config.enabled,
            )
            row = _orientation_row(
                target=target,
                orientation=orientation,
                status=status,
                stage1_raw_count=stage1.raw_candidate_count,
                stage1_count=len(stage1.bundle.candidates),
                stage2_count=len(stage2.accepted),
                best_direct=_best_candidate_payload(stage2.accepted),
                fallback_summary=fallback_summary,
                handover_summary=handover_summary,
            )
            row_links = {
                "stage2_json": _relative_link(output_dir, stage2_json),
                "stage2_html": _relative_link(output_dir, stage2_html),
                "fallback_json": _relative_link(output_dir, fallback_json) if fallback_summary is not None else "",
                "fallback_html": _relative_link(output_dir, fallback_html) if fallback_summary is not None else "",
                "handover_json": _relative_link(output_dir, handover_json) if handover_result is not None else "",
                "handover_html": _relative_link(output_dir, handover_html) if handover_result is not None else "",
            }
            if handover_result is not None:
                row["handover_pair_count_displayed"] = len(handover_result.accepted_pairs) + len(
                    handover_result.rejected_pairs
                )
                row["handover_rejection_counts"] = handover_result.metadata.get("rejection_counts", {})
                row["handover_transfer_floor_status_counts"] = handover_result.transfer_floor_status_counts
                row["handover_checked_pair_count"] = handover_result.metadata.get("checked_pair_count", 0)
                row["handover_transfer_floor_clearance_margin_m"] = handover_result.metadata.get(
                    "transfer_floor_clearance_margin_m",
                    handover_config.transfer_floor_clearance_margin_m,
                )
            all_grasps_html = orientation_dir / "all_generated_grasps.html"
            try:
                overview_counts = _write_all_generated_grasps_html(
                    all_grasps_html,
                    target=target,
                    orientation=orientation,
                    status=status,
                    stage1=stage1,
                    planning=planning,
                    simple_pickup_floor_clearance_margin_m=handover_config.transfer_floor_clearance_margin_m,
                    stage2=stage2,
                )
                row_links["all_generated_grasps_html"] = _relative_link(output_dir, all_grasps_html)
                row["all_generated_grasp_count"] = sum(overview_counts.values())
                row["all_generated_grasp_status_counts"] = overview_counts
            except Exception as debug_exc:
                row["all_generated_grasps_error"] = "".join(
                    traceback.format_exception_only(type(debug_exc), debug_exc)
                ).strip()
            if status not in SUCCESS_STATUSES:
                failed_html = orientation_dir / "failed_grasps.html"
                try:
                    stage1_failure_statuses, stage1_failure_obstacle_parts = _stage1_failure_debug()
                    failed_count, failed_constraint_counts = _write_failed_grasps_html(
                        failed_html,
                        target=target,
                        orientation=orientation,
                        status=status,
                        stage1=stage1,
                        planning=planning,
                        mesh_scale=mesh_scale,
                        stage2=stage2,
                        stage1_failure_statuses=stage1_failure_statuses,
                        stage1_failure_obstacle_parts=stage1_failure_obstacle_parts,
                    )
                    row_links["failed_grasps_html"] = _relative_link(output_dir, failed_html)
                    row["failed_grasp_count_displayed"] = failed_count
                    row["failed_constraint_counts"] = failed_constraint_counts
                    row["failed_grasp_display_frame"] = FAILED_GRASP_DISPLAY_FRAME
                except Exception as debug_exc:
                    row["failed_grasps_error"] = "".join(
                        traceback.format_exception_only(type(debug_exc), debug_exc)
                    ).strip()
            row["links"] = row_links
            rows.append(row)
            orientation_frames.append(
                _part_orientation_frame(
                    stage1=stage1,
                    planning=planning,
                    target=target,
                    orientation=orientation,
                    status=status,
                    stage2=stage2,
                    fallback_summary=fallback_summary,
                    handover_summary=handover_summary,
                    links=row_links,
                )
            )
            _write_orientation_details(
                orientation_dir / "details.json",
                target=target,
                orientation=orientation,
                status=status,
                stage1=stage1,
                stage2=stage2,
                fallback_summary=fallback_summary,
                handover_summary=handover_summary,
            )
            print(
                f"    {status}: stage2={len(stage2.accepted)} fallback={fallback_summary is not None} "
                f"handover={handover_summary is not None}",
                flush=True,
            )
        except Exception as exc:
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            row = _orientation_row(
                target=target,
                orientation=orientation,
                status="orientation_error",
                stage1_raw_count=stage1.raw_candidate_count,
                stage1_count=len(stage1.bundle.candidates),
                stage2_count=0,
                best_direct=None,
                fallback_summary=None,
                error=error,
            )
            row_links: dict[str, str] = {}
            failed_html = orientation_dir / "failed_grasps.html"
            all_grasps_html = orientation_dir / "all_generated_grasps.html"
            try:
                overview_counts = _write_all_generated_grasps_html(
                    all_grasps_html,
                    target=target,
                    orientation=orientation,
                    status="orientation_error",
                    stage1=stage1,
                    planning=planning,
                    simple_pickup_floor_clearance_margin_m=handover_config.transfer_floor_clearance_margin_m,
                    error=error,
                )
                row_links["all_generated_grasps_html"] = _relative_link(output_dir, all_grasps_html)
                row["all_generated_grasp_count"] = sum(overview_counts.values())
                row["all_generated_grasp_status_counts"] = overview_counts
            except Exception as debug_exc:
                row["all_generated_grasps_error"] = "".join(
                    traceback.format_exception_only(type(debug_exc), debug_exc)
                ).strip()
            try:
                stage1_failure_statuses, stage1_failure_obstacle_parts = _stage1_failure_debug()
                failed_count, failed_constraint_counts = _write_failed_grasps_html(
                    failed_html,
                    target=target,
                    orientation=orientation,
                    status="orientation_error",
                    stage1=stage1,
                    planning=planning,
                    mesh_scale=mesh_scale,
                    error=error,
                    stage1_failure_statuses=stage1_failure_statuses,
                    stage1_failure_obstacle_parts=stage1_failure_obstacle_parts,
                )
                row_links["failed_grasps_html"] = _relative_link(output_dir, failed_html)
                row["failed_grasp_count_displayed"] = failed_count
                row["failed_constraint_counts"] = failed_constraint_counts
                row["failed_grasp_display_frame"] = FAILED_GRASP_DISPLAY_FRAME
            except Exception as debug_exc:
                row["failed_grasps_error"] = "".join(traceback.format_exception_only(type(debug_exc), debug_exc)).strip()
            row["links"] = row_links
            rows.append(row)
            orientation_frames.append(
                _part_orientation_frame(
                    stage1=stage1,
                    planning=planning,
                    target=target,
                    orientation=orientation,
                    status="orientation_error",
                    links=row_links,
                    error=error,
                )
            )
            _write_orientation_details(
                orientation_dir / "details.json",
                target=target,
                orientation=orientation,
                status="orientation_error",
                stage1=stage1,
                error=error,
            )
            print(f"    orientation_error: {error}", flush=True)

    orientations_html = part_dir / "orientations.html"
    _write_part_orientations_html(
        orientations_html,
        target=target,
        stage1=stage1,
        orientation_frames=orientation_frames,
    )
    part_statuses = Counter(str(row["status"]) for row in rows if row["assembly"] == target.assembly and row["part_id"] == target.part_id)
    if any(part_statuses.get(status, 0) for status in SUCCESS_STATUSES):
        part_status = "has_generation_success"
    elif part_statuses:
        part_status = "all_orientations_failed"
    else:
        part_status = "no_orientation_rows"
    part_record.update(
        {
            "status": part_status,
            "orientation_status_counts": dict(part_statuses),
            "orientations_html": str(orientations_html),
        }
    )
    return part_record, rows, _part_browser_payload(target=target, stage1=stage1, orientation_frames=orientation_frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark grasp generation across Fabrica OBJ parts and stable poses.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Benchmark YAML config path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override benchmark output directory.")
    parser.add_argument("--clean", action="store_true", help="Remove stale benchmark artifacts, preserving stage1_cache.")
    parser.add_argument("--assembly", action="append", default=[], help="Restrict to an assembly name. Repeatable.")
    parser.add_argument("--part", action="append", default=[], help="Restrict to a part id/stem. Repeatable.")
    parser.add_argument("--target", action="append", default=[], help="Restrict to a target mesh path. Repeatable.")
    parser.add_argument("--limit-parts", type=int, default=None, help="Cap the number of targets after filtering.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of part-level benchmark workers to run concurrently. Default: 1.",
    )
    parser.add_argument("--robust-tilt-deg", type=float, default=None, help="Override stable orientation robust tilt.")
    parser.add_argument("--no-stage1-cache", action="store_true", help="Disable stage-1 cache use for this run.")
    parser.add_argument(
        "--skip-stage1-collision-checks",
        action="store_true",
        help="Bypass only stage-1 assembly collision filtering.",
    )
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument("--fallback", dest="fallback_enabled", action="store_true", default=None)
    fallback_group.add_argument("--no-fallback", dest="fallback_enabled", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1.")
    payload = _load_yaml(args.config)
    configured_output_dir = Path(str(dict(payload.get("benchmark", {})).get("output_dir", DEFAULT_OUTPUT_DIR)))
    output_dir = (args.output_dir or configured_output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    payload = _apply_cli_overrides(payload, args, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        _clean_output_dir(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if not trimesh_fcl_backend_available():
        raise RuntimeError(
            "trimesh/FCL collision backend is unavailable. Install python-fcl/native FCL before running the benchmark."
        )

    planning = _planning_config(payload)
    stable_config = _stable_orientation_config(payload)
    fallback_config = _fallback_config(payload)
    handover_config = _handover_fallback_config(payload)
    mesh_scale = float(dict(payload.get("geometry", {})).get("mesh_scale", 1.0))
    targets = _discover_targets(payload, args)
    if not targets:
        raise RuntimeError("No benchmark targets matched the requested filters.")

    _write_yaml(output_dir / "benchmark_config.yaml", payload)
    provenance = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(args.config),
        "cli_args": _json_safe(vars(args)),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "git": _git_metadata(),
        "collision_backend": "trimesh_fcl",
        "target_count": len(targets),
        "targets": [target.target_mesh_path for target in targets],
    }

    part_records: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    browser_parts: list[dict[str, object]] = []
    if args.jobs == 1 or len(targets) == 1:
        for index, target in enumerate(targets, start=1):
            part_record, part_rows, browser_part = _benchmark_one_target(
                target=target,
                planning=planning,
                stable_config=stable_config,
                fallback_config=fallback_config,
                handover_config=handover_config,
                mesh_scale=mesh_scale,
                output_dir=output_dir,
                target_index=index,
                target_count=len(targets),
            )
            part_records.append(part_record)
            rows.extend(part_rows)
            if browser_part is not None:
                browser_parts.append(browser_part)
    else:
        worker_count = min(int(args.jobs), len(targets))
        print(f"[BENCHMARK] Running {len(targets)} targets with {worker_count} part-level workers.", flush=True)
        completed: dict[int, tuple[dict[str, object], list[dict[str, object]], dict[str, object] | None]] = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _benchmark_one_target,
                    target=target,
                    planning=planning,
                    stable_config=stable_config,
                    fallback_config=fallback_config,
                    handover_config=handover_config,
                    mesh_scale=mesh_scale,
                    output_dir=output_dir,
                    target_index=index,
                    target_count=len(targets),
                ): index
                for index, target in enumerate(targets, start=1)
            }
            for future in as_completed(futures):
                index = futures[future]
                completed[index] = future.result()
                print(f"[BENCHMARK] Completed target {index}/{len(targets)}.", flush=True)
        for index in sorted(completed):
            part_record, part_rows, browser_part = completed[index]
            part_records.append(part_record)
            rows.extend(part_rows)
            if browser_part is not None:
                browser_parts.append(browser_part)

    results = {
        "schema_version": 1,
        "provenance": provenance,
        "config": payload,
        "parts": part_records,
        "orientations": rows,
        "summary": {
            "part_count": len(part_records),
            "orientation_count": len(rows),
            "orientation_status_counts": dict(Counter(str(row["status"]) for row in rows)),
            "part_status_counts": dict(Counter(str(part.get("status", "unknown")) for part in part_records)),
            "failed_grasp_pages_written": sum(
                1
                for row in rows
                if isinstance(row.get("links"), dict) and row["links"].get("failed_grasps_html")
            ),
            "all_generated_grasp_pages_written": sum(
                1
                for row in rows
                if isinstance(row.get("links"), dict) and row["links"].get("all_generated_grasps_html")
            ),
            "handover_fallback_pages_written": sum(
                1
                for row in rows
                if isinstance(row.get("links"), dict) and row["links"].get("handover_html")
            ),
            "failed_grasp_display_frames": dict(
                Counter(
                    str(row.get("failed_grasp_display_frame", "unknown"))
                    for row in rows
                    if isinstance(row.get("links"), dict) and row["links"].get("failed_grasps_html")
                )
            ),
        },
    }
    _write_json(output_dir / "results.json", results)
    _write_summary_csv(output_dir / "summary.csv", rows)
    _write_summary_md(output_dir / "summary.md", rows=rows, part_records=part_records)
    _write_index_html(
        output_dir / "index.html",
        output_dir=output_dir,
        rows=rows,
        part_records=part_records,
        browser_parts=browser_parts,
    )

    print(f"[BENCHMARK] Wrote results to {output_dir / 'results.json'}", flush=True)
    print(f"[BENCHMARK] Wrote summary to {output_dir / 'summary.md'}", flush=True)
    print(f"[BENCHMARK] Wrote index to {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
