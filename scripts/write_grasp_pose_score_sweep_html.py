#!/usr/bin/env python3
"""Write an HTML report showing how grasp scores change across floor poses."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping import (  # noqa: E402
    AntipodalGraspGeneratorConfig,
    SavedGraspCandidate,
    TriangleMesh,
    score_grasps,
)
from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    CandidateStatus,
    candidate_payload,
    evaluate_saved_grasps_against_pickup_pose,
    fmt_vec,
    ground_plane_overlay_obj,
    pickup_pose_for_support_face,
    quat_to_rotmat_xyzw,
    unique_edges,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.pipeline import PlanningConfig  # noqa: E402
from grasp_planning.pipeline.fabrica_pipeline import (  # noqa: E402
    DEFAULT_REACHABILITY_PROXY_COMFORT_BAND_M,
    DEFAULT_REACHABILITY_PROXY_COMFORT_RADIUS_M,
    DEFAULT_REACHABILITY_PROXY_FLOOR_CLEARANCE_M,
    DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_FULL_MAX_M,
    DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_FULL_MIN_M,
    DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_ZERO_MAX_M,
    DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_ZERO_MIN_M,
    DEFAULT_REACHABILITY_PROXY_OBJECT_CLEARANCE_M,
    _runtime_reachability_proxy_components,
)
from scripts.run_grasp_pipeline import _planning_config  # noqa: E402
from scripts.write_grasp_score_comparison_html import (  # noqa: E402
    _generator_config_from_payload,
    _load_yaml,
    generate_candidates_for_mesh,
    load_candidates_from_bundle,
)

SUPPORT_FACE_OPTIONS = ("neg_z", "pos_z", "neg_x", "pos_x", "neg_y", "pos_y")
SORT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("runtime_rank", "Runtime rank"),
    ("object_rank", "Object rank"),
    ("score_gain", "Most score gained"),
    ("score_loss", "Most score lost"),
)
SORT_OPTION_IDS = tuple(option_id for option_id, _ in SORT_OPTIONS)


@dataclass(frozen=True)
class FloorPoseRecord:
    pose_index: int
    label: str
    support_face: str
    yaw_deg: float
    xy_world: tuple[float, float]
    object_pose_world: ObjectWorldPose


def _parse_csv_floats(raw: str | None, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if raw in ("", None):
        return default
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one numeric CSV value.")
    return values


def _parse_support_faces(raw: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw in ("", None):
        return default
    if str(raw).strip().lower() == "all":
        return SUPPORT_FACE_OPTIONS
    faces = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not faces:
        raise ValueError("Expected at least one support face.")
    invalid = [face for face in faces if face not in SUPPORT_FACE_OPTIONS]
    if invalid:
        raise ValueError(f"Unsupported support face(s): {invalid}. Use one of {SUPPORT_FACE_OPTIONS} or 'all'.")
    return faces


def _base_pickup_defaults(config_payload: dict[str, object]) -> tuple[str, tuple[float, float]]:
    raw = config_payload.get("pickup_pose")
    if not isinstance(raw, dict):
        return "neg_z", (0.5, 0.0)
    xy_raw = raw.get("xy_world", (0.5, 0.0))
    if not isinstance(xy_raw, (list, tuple)) or len(xy_raw) != 2:
        xy_world = (0.5, 0.0)
    else:
        xy_world = (float(xy_raw[0]), float(xy_raw[1]))
    return str(raw.get("support_face", "neg_z")), xy_world


def _default_x_values(base_x: float) -> tuple[float, float, float]:
    return (max(0.05, float(base_x) - 0.25), float(base_x), float(base_x) + 0.30)


def _default_y_values(base_y: float) -> tuple[float, float, float]:
    return (float(base_y) - 0.12, float(base_y), float(base_y) + 0.12)


def build_floor_pose_records(
    *,
    mesh_local: TriangleMesh,
    support_faces: Iterable[str],
    yaw_deg_values: Iterable[float],
    x_values: Iterable[float],
    y_values: Iterable[float],
) -> list[FloorPoseRecord]:
    poses: list[FloorPoseRecord] = []
    for support_face in support_faces:
        for yaw_deg in yaw_deg_values:
            for x_world in x_values:
                for y_world in y_values:
                    pose = pickup_pose_for_support_face(
                        mesh_local,
                        support_face=support_face,
                        yaw_deg=float(yaw_deg),
                        xy_world=(float(x_world), float(y_world)),
                    )
                    label = (
                        f"{len(poses) + 1:02d} "
                        f"{support_face} yaw={float(yaw_deg):.0f} "
                        f"xy=({float(x_world):.2f},{float(y_world):+.2f})"
                    )
                    poses.append(
                        FloorPoseRecord(
                            pose_index=len(poses),
                            label=label,
                            support_face=support_face,
                            yaw_deg=float(yaw_deg),
                            xy_world=(float(x_world), float(y_world)),
                            object_pose_world=pose,
                        )
                    )
    if not poses:
        raise ValueError("Pose sweep produced no poses.")
    return poses


def _score_value(candidate: SavedGraspCandidate) -> float:
    return float("-inf") if candidate.score is None else float(candidate.score)


def _object_scored_pool(
    candidates: Iterable[SavedGraspCandidate],
    *,
    mesh_local: TriangleMesh,
    initial_pose_world: ObjectWorldPose,
    planning: PlanningConfig,
    max_candidates: int,
) -> list[SavedGraspCandidate]:
    object_scored = score_grasps(candidates, mesh_local=mesh_local)
    statuses = evaluate_saved_grasps_against_pickup_pose(
        object_scored,
        object_pose_world=initial_pose_world,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
        floor_clearance_margin_m=planning.floor_clearance_margin_m,
        contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
        contact_approach_offsets_m=planning.contact_approach_offsets_m,
    )
    accepted_by_id = {entry.grasp.grasp_id: entry.grasp for entry in statuses if entry.status == "accepted"}
    object_scored = [
        accepted_by_id[candidate.grasp_id] for candidate in object_scored if candidate.grasp_id in accepted_by_id
    ]
    if max_candidates > 0:
        return object_scored[:max_candidates]
    return object_scored


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _score_prescored_grasps_for_runtime_pose(
    grasps: Iterable[SavedGraspCandidate],
    *,
    mesh_local: TriangleMesh,
    object_pose_world: ObjectWorldPose,
    planning: PlanningConfig,
) -> list[SavedGraspCandidate]:
    top_weight = _clamp01(planning.top_grasp_score_weight)
    reachability_weight = _clamp01(planning.reachability_proxy_score_weight)
    world_weight = top_weight + reachability_weight
    if world_weight > 1.0:
        top_weight /= world_weight
        reachability_weight /= world_weight
        world_weight = 1.0
    object_weight = 1.0 - world_weight

    scored: list[SavedGraspCandidate] = []
    for grasp in grasps:
        grasp_rot_obj = quat_to_rotmat_xyzw(grasp.grasp_orientation_xyzw_obj)
        approach_axis_world = object_pose_world.rotation_world_from_object @ grasp_rot_obj[:, 2]
        top_down_score = _clamp01(-float(approach_axis_world[2]))
        reachability = _runtime_reachability_proxy_components(
            grasp,
            mesh_local=mesh_local,
            object_pose_world=object_pose_world,
            approach_axis_world=approach_axis_world,
            hand_offset_m=planning.reachability_proxy_hand_offset_m,
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
        scored.append(replace(grasp, score=float(combined_score), score_components=score_components))
    return sorted(
        scored,
        key=lambda candidate: (
            float("-inf") if candidate.score is None else float(candidate.score),
            candidate.grasp_id,
        ),
        reverse=True,
    )


def _pose_score_records(
    candidate_pool: list[SavedGraspCandidate],
    *,
    mesh_local: TriangleMesh,
    pose_record: FloorPoseRecord,
    planning: PlanningConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    statuses = evaluate_saved_grasps_against_pickup_pose(
        candidate_pool,
        object_pose_world=pose_record.object_pose_world,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
        floor_clearance_margin_m=planning.floor_clearance_margin_m,
        contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
        contact_approach_offsets_m=planning.contact_approach_offsets_m,
    )
    runtime_scored = _score_prescored_grasps_for_runtime_pose(
        candidate_pool,
        mesh_local=mesh_local,
        object_pose_world=pose_record.object_pose_world,
        planning=planning,
    )
    runtime_by_id = {candidate.grasp_id: candidate for candidate in runtime_scored}
    runtime_rank = {candidate.grasp_id: rank for rank, candidate in enumerate(runtime_scored, start=1)}
    status_by_id = {entry.grasp.grasp_id: entry for entry in statuses}
    object_by_id = {candidate.grasp_id: candidate for candidate in candidate_pool}

    records: list[dict[str, object]] = []
    for candidate in candidate_pool:
        entry = status_by_id[candidate.grasp_id]
        runtime = runtime_by_id[candidate.grasp_id]
        object_score = _score_value(object_by_id[candidate.grasp_id])
        runtime_score = _score_value(runtime)
        components = dict(runtime.score_components or {})
        record = {
            "pose_index": pose_record.pose_index,
            "grasp_id": candidate.grasp_id,
            "status": entry.status,
            "reason": entry.reason,
            "runtime_rank": runtime_rank[candidate.grasp_id],
            "object_score": object_score,
            "score": runtime_score,
            "score_delta": float(runtime_score - object_score),
            "top_down_approach": components.get("top_down_approach"),
            "reachability_proxy": components.get("reachability_proxy"),
            "reachability_hand_radial": components.get("reachability_hand_radial"),
            "reachability_side": components.get("reachability_side"),
            "reachability_hand_radius_m": components.get("reachability_hand_radius_m"),
            "reachability_hand_side": components.get("reachability_hand_side"),
            "reachability_target_side": components.get("reachability_target_side"),
            "reachability_pregrasp_x_world": components.get("reachability_pregrasp_x_world"),
            "reachability_pregrasp_y_world": components.get("reachability_pregrasp_y_world"),
            "reachability_pregrasp_z_world": components.get("reachability_pregrasp_z_world"),
            "components": components,
        }
        records.append(record)

    accepted_runtime = [
        candidate for candidate in runtime_scored if status_by_id[candidate.grasp_id].status == "accepted"
    ]
    accepted_scores = [_score_value(candidate) for candidate in accepted_runtime]
    top = accepted_runtime[0] if accepted_runtime else None
    top_scored = runtime_scored[0] if runtime_scored else None
    summary = {
        "pose_index": pose_record.pose_index,
        "top_grasp_id": None if top is None else top.grasp_id,
        "top_score": None if top is None else _score_value(top),
        "top_scored_grasp_id": None if top_scored is None else top_scored.grasp_id,
        "top_scored_score": None if top_scored is None else _score_value(top_scored),
        "accepted_count": len(accepted_runtime),
        "candidate_count": len(candidate_pool),
        "average_score": None if not accepted_scores else float(np.mean(accepted_scores)),
    }
    return records, summary


def _select_display_ids(
    candidate_pool: list[SavedGraspCandidate],
    pose_records_by_id: dict[str, list[dict[str, object]]],
    *,
    max_display: int,
    top_per_pose_ids: Iterable[str],
    allowed_ids: set[str] | None = None,
) -> list[str]:
    if max_display <= 0:
        return []
    selected: list[str] = []
    seen: set[str] = set()

    def add(grasp_id: str) -> None:
        if allowed_ids is not None and grasp_id not in allowed_ids:
            return
        if len(selected) >= max_display or grasp_id in seen:
            return
        seen.add(grasp_id)
        selected.append(grasp_id)

    for grasp_id in top_per_pose_ids:
        add(grasp_id)

    def best_runtime_score(grasp_id: str) -> float:
        best = float("-inf")
        for record in pose_records_by_id.get(grasp_id, []):
            score = record.get("score")
            if score is not None:
                best = max(best, float(score))
        return best

    for candidate in sorted(
        candidate_pool,
        key=lambda item: (best_runtime_score(item.grasp_id), _score_value(item), item.grasp_id),
        reverse=True,
    ):
        add(candidate.grasp_id)
    return selected


def _world_pose_payload(record: FloorPoseRecord) -> dict[str, object]:
    rotation = record.object_pose_world.rotation_world_from_object
    return {
        "pose_index": record.pose_index,
        "label": record.label,
        "support_face": record.support_face,
        "yaw_deg": record.yaw_deg,
        "xy_world": [float(record.xy_world[0]), float(record.xy_world[1])],
        "position_world": fmt_vec(record.object_pose_world.position_world),
        "orientation_xyzw_world": fmt_vec(record.object_pose_world.orientation_xyzw_world),
        "rotation_world_from_object": [[float(v) for v in row] for row in rotation.tolist()],
    }


def _world_bounds(
    *,
    mesh_local: TriangleMesh,
    poses: list[FloorPoseRecord],
    pose_records_by_id: dict[str, list[dict[str, object]]],
) -> tuple[list[float], list[float]]:
    points = [np.zeros(3, dtype=float)]
    vertices = np.asarray(mesh_local.vertices_obj, dtype=float)
    for pose in poses:
        points.extend(pose.object_pose_world.transform_points_to_world(vertices))
    for pose_records in pose_records_by_id.values():
        for record in pose_records:
            if record.get("reachability_pregrasp_x_world") is None:
                continue
            points.append(
                np.array(
                    [
                        float(record["reachability_pregrasp_x_world"]),
                        float(record["reachability_pregrasp_y_world"]),
                        float(record["reachability_pregrasp_z_world"]),
                    ],
                    dtype=float,
                )
            )
    array = np.asarray(points, dtype=float)
    mins = array.min(axis=0)
    maxs = array.max(axis=0)
    padding = np.maximum(0.10 * (maxs - mins), np.array([0.08, 0.08, 0.04], dtype=float))
    return (mins - padding).tolist(), (maxs + padding).tolist()


def _slider_bounds(values: Iterable[float], *, fallback_center: float, fallback_radius: float) -> dict[str, float]:
    unique_values = [float(value) for value in values]
    if not unique_values:
        return {
            "min": float(fallback_center - fallback_radius),
            "max": float(fallback_center + fallback_radius),
            "value": float(fallback_center),
        }
    min_value = min(unique_values)
    max_value = max(unique_values)
    center = float(unique_values[len(unique_values) // 2])
    if abs(max_value - min_value) < 1.0e-9:
        min_value = center - fallback_radius
        max_value = center + fallback_radius
    return {"min": float(min_value), "max": float(max_value), "value": center}


def _interactive_controls_payload(pose_records: list[FloorPoseRecord]) -> dict[str, object]:
    x_values = [record.xy_world[0] for record in pose_records]
    y_values = [record.xy_world[1] for record in pose_records]
    yaw_values = [record.yaw_deg for record in pose_records]
    default_pose = pose_records[0]
    return {
        "x": {
            **_slider_bounds(x_values, fallback_center=default_pose.xy_world[0], fallback_radius=0.25),
            "step": 0.005,
        },
        "y": {
            **_slider_bounds(y_values, fallback_center=default_pose.xy_world[1], fallback_radius=0.12),
            "step": 0.005,
        },
        "yaw_deg": {
            "min": 0.0,
            "max": 360.0,
            "value": float(yaw_values[0] if yaw_values else default_pose.yaw_deg),
            "step": 1.0,
        },
        "hand_roll_deg": {"min": -180.0, "max": 180.0, "value": 0.0, "step": 1.0},
    }


def _scoring_payload(planning: PlanningConfig) -> dict[str, float]:
    return {
        "top_grasp_score_weight": float(planning.top_grasp_score_weight),
        "reachability_proxy_score_weight": float(planning.reachability_proxy_score_weight),
        "reachability_proxy_hand_offset_m": float(planning.reachability_proxy_hand_offset_m),
        "floor_clearance_margin_m": float(planning.floor_clearance_margin_m),
        "reachability_comfort_radius_m": DEFAULT_REACHABILITY_PROXY_COMFORT_RADIUS_M,
        "reachability_comfort_band_m": DEFAULT_REACHABILITY_PROXY_COMFORT_BAND_M,
        "reachability_hand_radius_zero_min_m": DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_ZERO_MIN_M,
        "reachability_hand_radius_full_min_m": DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_FULL_MIN_M,
        "reachability_hand_radius_full_max_m": DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_FULL_MAX_M,
        "reachability_hand_radius_zero_max_m": DEFAULT_REACHABILITY_PROXY_HAND_RADIUS_ZERO_MAX_M,
        "reachability_object_clearance_m": DEFAULT_REACHABILITY_PROXY_OBJECT_CLEARANCE_M,
        "reachability_floor_clearance_m": DEFAULT_REACHABILITY_PROXY_FLOOR_CLEARANCE_M,
    }


def build_pose_score_sweep_payload(
    *,
    mesh_local: TriangleMesh,
    candidates: Iterable[SavedGraspCandidate],
    source_label: str,
    planning: PlanningConfig,
    pose_records: list[FloorPoseRecord],
    max_candidates: int,
    max_display: int,
    top_per_pose: int,
) -> dict[str, object]:
    candidate_list = list(candidates)
    candidate_pool = _object_scored_pool(
        candidate_list,
        mesh_local=mesh_local,
        initial_pose_world=pose_records[0].object_pose_world,
        planning=planning,
        max_candidates=max_candidates,
    )
    if not candidate_pool:
        raise RuntimeError("No grasp candidates were available for pose score sweep.")

    object_rank = {candidate.grasp_id: rank for rank, candidate in enumerate(candidate_pool, start=1)}
    object_by_id = {candidate.grasp_id: candidate for candidate in candidate_pool}
    pose_records_by_id: dict[str, list[dict[str, object]]] = {candidate.grasp_id: [] for candidate in candidate_pool}
    pose_summaries: list[dict[str, object]] = []
    top_per_pose_ids: list[str] = []

    for pose_record in pose_records:
        records, summary = _pose_score_records(
            candidate_pool,
            mesh_local=mesh_local,
            pose_record=pose_record,
            planning=planning,
        )
        pose_summaries.append(summary)
        ranked_records = sorted(records, key=lambda item: int(item["runtime_rank"]))
        accepted_sorted = [record for record in ranked_records if record["status"] == "accepted"]
        top_records = accepted_sorted[: max(0, top_per_pose)]
        top_per_pose_ids.extend(str(record["grasp_id"]) for record in top_records)
        for record in records:
            pose_records_by_id[str(record["grasp_id"])].append(record)

    feasible_ids = {
        grasp_id
        for grasp_id, records in pose_records_by_id.items()
        if any(record["status"] == "accepted" for record in records)
    }

    display_ids = _select_display_ids(
        candidate_pool,
        pose_records_by_id,
        max_display=max_display,
        top_per_pose_ids=top_per_pose_ids,
        allowed_ids=feasible_ids,
    )
    base_payload_by_id = {
        str(item["grasp_id"]): item
        for item in candidate_payload(
            [
                CandidateStatus(grasp=object_by_id[grasp_id], status="accepted", reason="pose_sweep_candidate")
                for grasp_id in display_ids
            ],
            contact_gap_m=planning.detailed_finger_contact_gap_m,
        )
    }

    display_candidates: list[dict[str, object]] = []
    for display_index, grasp_id in enumerate(display_ids, start=1):
        base_payload = dict(base_payload_by_id[grasp_id])
        object_candidate = object_by_id[grasp_id]
        pose_scores = pose_records_by_id[grasp_id]
        accepted_scores = [float(record["score"]) for record in pose_scores if record["score"] is not None]
        deltas = [float(record["score_delta"]) for record in pose_scores if record["score_delta"] is not None]
        best_record = max(
            pose_scores,
            key=lambda record: float("-inf") if record["score"] is None else float(record["score"]),
        )
        base_payload.update(
            {
                "display_rank": display_index,
                "object_rank": object_rank[grasp_id],
                "object_score": _score_value(object_candidate),
                "object_components": dict(object_candidate.score_components or {}),
                "pose_scores": pose_scores,
                "best_pose_index": int(best_record["pose_index"]),
                "best_score": None if not accepted_scores else max(accepted_scores),
                "worst_score_delta": None if not deltas else min(deltas),
                "best_score_delta": None if not deltas else max(deltas),
            }
        )
        display_candidates.append(base_payload)

    bounds_min_world, bounds_max_world = _world_bounds(
        mesh_local=mesh_local,
        poses=pose_records,
        pose_records_by_id={grasp_id: pose_records_by_id[grasp_id] for grasp_id in display_ids},
    )
    faces = np.asarray(mesh_local.faces, dtype=np.int64)
    first_pose = pose_records[0]
    return {
        "title": "Grasp Runtime Pose Score Sweep",
        "subtitle": "One generated grasp set rescored across many floor poses using the runtime top-down and reachability terms.",
        "source_label": source_label,
        "candidate_count": len(candidate_list),
        "scored_candidate_count": len(candidate_pool),
        "display_count": len(display_candidates),
        "pose_count": len(pose_records),
        "sort_options": [{"id": option_id, "label": label} for option_id, label in SORT_OPTIONS],
        "sort_by": "runtime_rank",
        "metadata_lines": [
            f"source:           {source_label}",
            f"input_candidates: {len(candidate_list)}",
            f"scored_candidates:{len(candidate_pool)}",
            f"display_count:    {len(display_candidates)}",
            f"pose_count:       {len(pose_records)}",
            f"top_score_weight: {planning.top_grasp_score_weight:.3f}",
            f"reach_score_wt:   {planning.reachability_proxy_score_weight:.3f}",
            f"reach_hand_off:   {planning.reachability_proxy_hand_offset_m:.3f} m",
            f"floor_clearance:  {planning.floor_clearance_margin_m:.6f} m",
        ],
        "vertices_obj": [[float(v) for v in vertex] for vertex in np.asarray(mesh_local.vertices_obj).tolist()],
        "faces": [[int(v) for v in face] for face in faces.tolist()],
        "edges": [[int(a), int(b)] for a, b in unique_edges(faces)],
        "ground_plane_overlay": ground_plane_overlay_obj(
            mesh_local,
            object_pose_world=first_pose.object_pose_world,
            enabled=True,
        ),
        "bounds_min_world": [float(v) for v in bounds_min_world],
        "bounds_max_world": [float(v) for v in bounds_max_world],
        "interactive_controls": _interactive_controls_payload(pose_records),
        "scoring": _scoring_payload(planning),
        "poses": [_world_pose_payload(record) for record in pose_records],
        "pose_summaries": pose_summaries,
        "candidates": display_candidates,
    }


def _html_document(data_json: str) -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Grasp Runtime Pose Score Sweep</title>
  <style>
    :root {
      --bg: #f3efe4;
      --panel: #fffaf0;
      --ink: #1e1d1a;
      --accent: #b43f2c;
      --accent-soft: #e8b59f;
      --muted: #6f6a5f;
      --mesh: #4f6b5f;
      --ground: #2563eb;
      --accepted: #15803d;
      --rejected: #b91c1c;
      --contact-a: #c8452d;
      --contact-b: #1f7c60;
      --franka: #d97706;
      --hand: #8f5a12;
      --base: #111827;
      --pregrasp: #7c3aed;
      --gain: #168354;
      --loss: #bf4938;
      --flat: #6d7d76;
      --line: #d9ceb8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff8e8 0, transparent 30%),
        linear-gradient(135deg, #f7f2e7 0%, #efe7d4 100%);
    }
    .layout { display: grid; grid-template-columns: 410px minmax(0, 1fr); min-height: 100vh; }
    .sidebar { border-right: 1px solid var(--line); background: rgba(255,250,240,0.94); padding: 20px 18px; overflow: auto; }
    .title { margin: 0 0 8px; font-size: 27px; line-height: 1.1; }
    .subtitle { margin: 0 0 16px; color: var(--muted); font-size: 14px; line-height: 1.45; }
    .controls { display: flex; flex-wrap: wrap; gap: 9px; margin-bottom: 12px; align-items: center; }
    button, select { border: 1px solid var(--line); background: white; color: var(--ink); border-radius: 999px; padding: 9px 12px; font: inherit; }
    button { cursor: pointer; }
    button:hover, select:hover { border-color: var(--accent); }
    button:disabled { cursor: default; color: var(--muted); background: #f7f1e5; }
    button:disabled:hover { border-color: var(--line); }
    .field { display: grid; gap: 5px; width: 100%; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    .field select { width: 100%; border-radius: 10px; text-transform: none; letter-spacing: 0; color: var(--ink); }
    .sliders { display: grid; gap: 10px; width: 100%; border: 1px solid var(--line); border-radius: 14px; padding: 11px 12px; background: rgba(255,255,255,0.68); margin-bottom: 12px; }
    .slider-row { display: grid; grid-template-columns: 54px minmax(0, 1fr) 82px; gap: 10px; align-items: center; font-family: "IBM Plex Mono", monospace; font-size: 12px; color: var(--muted); }
    .slider-row input[type="range"] { width: 100%; accent-color: var(--accent); }
    .slider-row output { text-align: right; color: var(--ink); }
    .pose-card { border: 1px solid var(--line); border-radius: 14px; background: rgba(255,255,255,0.72); padding: 10px 12px; margin-bottom: 12px; font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.5; color: var(--muted); }
    .list { display: grid; gap: 9px; margin-bottom: 18px; }
    .item { border: 1px solid var(--line); border-radius: 14px; padding: 11px 13px; background: rgba(255,255,255,0.72); cursor: pointer; transition: border-color 120ms ease, box-shadow 120ms ease; text-align: left; }
    .item:hover { border-color: var(--accent-soft); box-shadow: 0 8px 18px rgba(85,65,42,0.08); }
    .item.active { border-color: var(--accent); box-shadow: 0 10px 24px rgba(180,63,44,0.16); background: #fff; }
    .item-rank { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
    .item-main { display: flex; justify-content: space-between; align-items: baseline; margin-top: 5px; gap: 10px; }
    .item-label { font-size: 19px; font-weight: 700; }
    .item-score { font-family: "IBM Plex Mono", monospace; font-size: 13px; white-space: nowrap; }
    .item-meta { margin-top: 7px; color: var(--muted); font-size: 12px; font-family: "IBM Plex Mono", monospace; line-height: 1.45; }
    .status.accepted { color: var(--accepted); }
    .status.rejected { color: var(--rejected); }
    .delta.gain { color: var(--gain); }
    .delta.loss { color: var(--loss); }
    .delta.flat { color: var(--flat); }
    .main { padding: 18px; overflow: auto; }
    .cards { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(330px, 0.75fr); gap: 18px; align-items: start; }
    .card { border: 1px solid var(--line); border-radius: 20px; background: rgba(255,250,240,0.88); padding: 16px; box-shadow: 0 14px 32px rgba(72,51,28,0.08); }
    .card h2 { margin: 0 0 12px; font-size: 16px; letter-spacing: 0.03em; text-transform: uppercase; }
    #scene {
      width: 100%;
      height: auto;
      aspect-ratio: 1.25 / 1;
      display: block;
      background:
        radial-gradient(circle at 20% 18%, rgba(255,255,255,0.9), rgba(255,255,255,0.55) 35%, rgba(233,226,208,0.65)),
        linear-gradient(180deg, rgba(255,255,255,0.2), rgba(223,214,194,0.18));
      border-radius: 16px;
    }
    .legend { display: flex; flex-wrap: wrap; gap: 11px; margin-top: 12px; font-size: 13px; color: var(--muted); }
    .legend span { display: inline-flex; align-items: center; gap: 7px; }
    .swatch { width: 14px; height: 14px; border-radius: 999px; display: inline-block; }
    .kv { white-space: pre-wrap; font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.55; margin: 0; }
    .caption { margin-top: 10px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    @media (max-width: 1120px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .cards { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1 id="title" class="title"></h1>
      <p id="subtitle" class="subtitle"></p>
      <div class="controls">
        <button id="prevPoseBtn" type="button">Prev Pose</button>
        <button id="nextPoseBtn" type="button">Next Pose</button>
        <button id="prevGraspBtn" type="button">Prev Grasp</button>
        <button id="nextGraspBtn" type="button">Next Grasp</button>
        <button id="meshModeBtn" type="button">Solid Mesh</button>
        <button id="feasibleOnlyBtn" type="button" disabled>Default-Roll Feasible: On</button>
        <label class="field" for="poseSelect">Pose<select id="poseSelect"></select></label>
        <label class="field" for="sortSelect">Sort<select id="sortSelect"></select></label>
      </div>
      <div class="sliders" aria-label="Live scoring sliders">
        <label class="slider-row" for="xSlider"><span>x</span><input id="xSlider" type="range"><output id="xValue"></output></label>
        <label class="slider-row" for="ySlider"><span>y</span><input id="ySlider" type="range"><output id="yValue"></output></label>
        <label class="slider-row" for="objectYawSlider"><span>yaw</span><input id="objectYawSlider" type="range"><output id="objectYawValue"></output></label>
        <label class="slider-row" for="handRollSlider"><span>hand roll</span><input id="handRollSlider" type="range"><output id="handRollValue"></output></label>
      </div>
      <div id="poseSummary" class="pose-card"></div>
      <div id="graspList" class="list"></div>
    </aside>
    <main class="main">
      <div class="cards">
        <section class="card">
          <h2>World Pose</h2>
          <svg id="scene" viewBox="0 0 960 760"></svg>
          <div class="legend">
            <span><i class="swatch" style="background: var(--mesh)"></i>Object mesh</span>
            <span><i class="swatch" style="background: var(--ground)"></i>Floor</span>
            <span><i class="swatch" style="background: var(--base)"></i>Robot base</span>
            <span><i class="swatch" style="background: var(--franka)"></i>Gripper finger boxes</span>
            <span><i class="swatch" style="background: var(--hand)"></i>Gripper mesh</span>
            <span><i class="swatch" style="background: #0f766e"></i>3x3 contact grid</span>
            <span><i class="swatch" style="background: var(--pregrasp)"></i>Pregrasp proxy</span>
          </div>
          <p class="caption">Left drag rotates, middle drag pans, scroll zooms. Up/Down changes pose; Left/Right changes grasp.</p>
        </section>
        <section class="card">
          <h2>Selection</h2>
          <pre id="details" class="kv"></pre>
        </section>
      </div>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const title = document.getElementById("title");
    const subtitle = document.getElementById("subtitle");
    const graspList = document.getElementById("graspList");
    const poseSummary = document.getElementById("poseSummary");
    const scene = document.getElementById("scene");
    const details = document.getElementById("details");
    const poseSelect = document.getElementById("poseSelect");
    const sortSelect = document.getElementById("sortSelect");
    const prevPoseBtn = document.getElementById("prevPoseBtn");
    const nextPoseBtn = document.getElementById("nextPoseBtn");
    const prevGraspBtn = document.getElementById("prevGraspBtn");
    const nextGraspBtn = document.getElementById("nextGraspBtn");
    const meshModeBtn = document.getElementById("meshModeBtn");
    const feasibleOnlyBtn = document.getElementById("feasibleOnlyBtn");
    const xSlider = document.getElementById("xSlider");
    const ySlider = document.getElementById("ySlider");
    const objectYawSlider = document.getElementById("objectYawSlider");
    const handRollSlider = document.getElementById("handRollSlider");
    const xValue = document.getElementById("xValue");
    const yValue = document.getElementById("yValue");
    const objectYawValue = document.getElementById("objectYawValue");
    const handRollValue = document.getElementById("handRollValue");
    title.textContent = data.title || "Grasp Runtime Pose Score Sweep";
    subtitle.textContent = data.subtitle || "";
    data.poses.forEach((pose) => {
      const option = document.createElement("option");
      option.value = String(pose.pose_index);
      option.textContent = pose.label;
      poseSelect.appendChild(option);
    });
    (data.sort_options || []).forEach((optionInfo) => {
      const option = document.createElement("option");
      option.value = optionInfo.id;
      option.textContent = optionInfo.label;
      option.selected = optionInfo.id === data.sort_by;
      sortSelect.appendChild(option);
    });
    const state = {
      poseIndex: 0,
      selectedIndex: 0,
      yaw: -0.82,
      pitch: 0.56,
      zoom: 1.0,
      panX: 0,
      panY: 0,
      dragging: false,
      dragMode: "rotate",
      lastPointerX: 0,
      lastPointerY: 0,
      pointerId: null,
      meshRenderMode: "wireframe",
      defaultRollFeasibleOnly: true,
      sortBy: data.sort_by || "runtime_rank",
      objectX: 0,
      objectY: 0,
      objectYawDeg: 0,
      supportFace: "neg_z",
      handRollDeg: 0,
    };
    const worldMin = data.bounds_min_world || [-0.2, -0.2, -0.05];
    const worldMax = data.bounds_max_world || [0.8, 0.2, 0.3];
    const worldCenter = worldMin.map((value, axis) => 0.5 * (value + worldMax[axis]));
    const worldExtent = Math.max(...worldMax.map((value, axis) => value - worldMin[axis]), 0.2);
    const baseScale = 520 / worldExtent;
    function setupSlider(slider, cfg) {
      slider.min = String(cfg.min);
      slider.max = String(cfg.max);
      slider.step = String(cfg.step || 1);
      slider.value = String(cfg.value);
    }
    function syncSliderOutputs() {
      xValue.textContent = `${Number(state.objectX).toFixed(3)} m`;
      yValue.textContent = `${Number(state.objectY).toFixed(3)} m`;
      objectYawValue.textContent = `${Number(state.objectYawDeg).toFixed(1)} deg`;
      handRollValue.textContent = `${Number(state.handRollDeg).toFixed(1)} deg`;
    }
    function setSlidersFromState() {
      xSlider.value = String(state.objectX);
      ySlider.value = String(state.objectY);
      objectYawSlider.value = String(state.objectYawDeg);
      handRollSlider.value = String(state.handRollDeg);
      syncSliderOutputs();
    }
    function setStateFromPose(pose, resetRoll = true) {
      state.objectX = Number(pose.xy_world[0]);
      state.objectY = Number(pose.xy_world[1]);
      state.objectYawDeg = Number(pose.yaw_deg);
      state.supportFace = pose.support_face || "neg_z";
      if (resetRoll) state.handRollDeg = 0;
      setSlidersFromState();
    }
    setupSlider(xSlider, data.interactive_controls.x);
    setupSlider(ySlider, data.interactive_controls.y);
    setupSlider(objectYawSlider, data.interactive_controls.yaw_deg);
    setupSlider(handRollSlider, data.interactive_controls.hand_roll_deg);
    setStateFromPose(data.poses[0] || { xy_world: [0, 0], yaw_deg: 0, support_face: "neg_z" });
    function degToRad(deg) { return Number(deg) * Math.PI / 180.0; }
    function matMul(a, b) {
      return [
        [
          a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
          a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
          a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ],
        [
          a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
          a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
          a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ],
        [
          a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
          a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
          a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ],
      ];
    }
    function matVec(m, v) {
      return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
      ];
    }
    function rotX(angle) {
      const c = Math.cos(angle), s = Math.sin(angle);
      return [[1, 0, 0], [0, c, -s], [0, s, c]];
    }
    function rotY(angle) {
      const c = Math.cos(angle), s = Math.sin(angle);
      return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
    }
    function rotZ(angle) {
      const c = Math.cos(angle), s = Math.sin(angle);
      return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
    }
    function supportFaceRotation(face) {
      if (face === "pos_x") return rotY(Math.PI / 2);
      if (face === "neg_x") return rotY(-Math.PI / 2);
      if (face === "pos_y") return rotX(-Math.PI / 2);
      if (face === "neg_y") return rotX(Math.PI / 2);
      if (face === "pos_z") return rotX(Math.PI);
      return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    }
    function currentPose() {
      const rotation = matMul(rotZ(degToRad(state.objectYawDeg)), supportFaceRotation(state.supportFace));
      const rotated = data.vertices_obj.map((point) => matVec(rotation, point));
      const minZ = Math.min(...rotated.map((point) => point[2]));
      return {
        pose_index: state.poseIndex,
        label: `live ${state.supportFace} yaw=${Number(state.objectYawDeg).toFixed(1)} xy=(${Number(state.objectX).toFixed(3)},${Number(state.objectY).toFixed(3)}) roll=${Number(state.handRollDeg).toFixed(1)}`,
        support_face: state.supportFace,
        yaw_deg: Number(state.objectYawDeg),
        xy_world: [Number(state.objectX), Number(state.objectY)],
        position_world: [Number(state.objectX), Number(state.objectY), -minZ],
        rotation_world_from_object: rotation,
      };
    }
    function vecAdd(a, b) { return a.map((value, axis) => value + b[axis]); }
    function vecSub(a, b) { return a.map((value, axis) => value - b[axis]); }
    function vecScale(v, scale) { return v.map((value) => value * scale); }
    function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
    function norm(v) { return Math.hypot(...v); }
    function normalize(v) {
      const length = norm(v);
      if (length < 1e-9) return v.map(() => 0);
      return v.map((value) => value / length);
    }
    function rotateVectorAroundAxis(vector, axis, angle) {
      const unit = normalize(axis);
      const c = Math.cos(angle), s = Math.sin(angle);
      return vecAdd(
        vecAdd(vecScale(vector, c), vecScale([
          unit[1] * vector[2] - unit[2] * vector[1],
          unit[2] * vector[0] - unit[0] * vector[2],
          unit[0] * vector[1] - unit[1] * vector[0],
        ], s)),
        vecScale(unit, dot(unit, vector) * (1 - c))
      );
    }
    function rotatePointAroundAxis(point, origin, axis, angle) {
      return vecAdd(origin, rotateVectorAroundAxis(vecSub(point, origin), axis, angle));
    }
    function adjustedGeometry(candidate, rollDeg = state.handRollDeg) {
      const angle = degToRad(rollDeg);
      const axis = normalize(candidate.closing_axis_obj);
      const origin = [
        0.5 * (candidate.contact_point_a_obj[0] + candidate.contact_point_b_obj[0]),
        0.5 * (candidate.contact_point_a_obj[1] + candidate.contact_point_b_obj[1]),
        0.5 * (candidate.contact_point_a_obj[2] + candidate.contact_point_b_obj[2]),
      ];
      const rotatePoint = (point) => rotatePointAroundAxis(point, origin, axis, angle);
      return {
        approach_axis_obj: rotateVectorAroundAxis(candidate.gripper_z_axis_obj, axis, angle),
        hand_vertices_obj: (candidate.franka_hand_vertices_obj || []).map(rotatePoint),
        left_boxes: (candidate.franka_left_boxes || []).map((box) => ({ name: box.name, corners: box.corners.map(rotatePoint) })),
        right_boxes: (candidate.franka_right_boxes || []).map((box) => ({ name: box.name, corners: box.corners.map(rotatePoint) })),
        left_grid: (candidate.franka_left_contact_grid_obj || []).map(rotatePoint),
        right_grid: (candidate.franka_right_contact_grid_obj || []).map(rotatePoint),
        left_anchor: rotatePoint(candidate.franka_left_tip_anchor_obj),
        right_anchor: rotatePoint(candidate.franka_right_tip_anchor_obj),
      };
    }
    function objectAabbCenterObj() {
      const mins = [Infinity, Infinity, Infinity], maxs = [-Infinity, -Infinity, -Infinity];
      data.vertices_obj.forEach((point) => point.forEach((value, axis) => {
        mins[axis] = Math.min(mins[axis], value);
        maxs[axis] = Math.max(maxs[axis], value);
      }));
      return mins.map((value, axis) => 0.5 * (value + maxs[axis]));
    }
    const objectCenterObj = objectAabbCenterObj();
    function worldObjectBounds(pose = currentPose()) {
      const vertices = transformPoints(data.vertices_obj, pose);
      const mins = [Infinity, Infinity, Infinity], maxs = [-Infinity, -Infinity, -Infinity];
      vertices.forEach((point) => point.forEach((value, axis) => {
        mins[axis] = Math.min(mins[axis], value);
        maxs[axis] = Math.max(maxs[axis], value);
      }));
      return { mins, maxs };
    }
    function trapezoidScore(value, zeroBelow, fullFrom, fullTo, zeroAbove) {
      if (value <= zeroBelow || value >= zeroAbove) return 0.0;
      if (fullFrom <= value && value <= fullTo) return 1.0;
      if (value < fullFrom) return clamp((value - zeroBelow) / Math.max(fullFrom - zeroBelow, 1e-9), 0, 1);
      return clamp((zeroAbove - value) / Math.max(zeroAbove - fullTo, 1e-9), 0, 1);
    }
    function liveReachability(candidate, geometry, pose) {
      const scoring = data.scoring;
      const graspWorld = transformPoint(candidate.grasp_position_obj, pose);
      const objectCenterWorld = transformPoint(objectCenterObj, pose);
      const approachAxisWorld = matVec(pose.rotation_world_from_object, geometry.approach_axis_obj);
      const backoffAxisWorld = vecScale(normalize(approachAxisWorld), -1);
      const handWorld = vecAdd(graspWorld, vecScale(backoffAxisWorld, Math.max(0, scoring.reachability_proxy_hand_offset_m)));
      const objectRadius = Math.hypot(objectCenterWorld[0], objectCenterWorld[1]);
      const handRadius = Math.hypot(handWorld[0], handWorld[1]);
      const objectDir = normalize([objectCenterWorld[0], objectCenterWorld[1], 0]);
      const backoffXY = normalize([backoffAxisWorld[0], backoffAxisWorld[1], 0]);
      const handSide = norm(objectDir) < 1e-9 ? 0.0 : clamp(dot(backoffXY, objectDir), -1, 1);
      const targetSide = clamp(
        (scoring.reachability_comfort_radius_m - objectRadius) / Math.max(scoring.reachability_comfort_band_m, 1e-9),
        -1,
        1
      );
      const rawSideScore = targetSide >= 0 ? 0.5 * (1.0 + handSide) : 0.5 * (1.0 - handSide);
      const sideStrength = Math.abs(targetSide);
      const sideScore = (1.0 - sideStrength) + sideStrength * rawSideScore;
      const handRadialScore = trapezoidScore(
        handRadius,
        scoring.reachability_hand_radius_zero_min_m,
        scoring.reachability_hand_radius_full_min_m,
        scoring.reachability_hand_radius_full_max_m,
        scoring.reachability_hand_radius_zero_max_m
      );
      const bounds = worldObjectBounds(pose);
      const outside = [0, 1, 2].map((axis) => Math.max(bounds.mins[axis] - handWorld[axis], handWorld[axis] - bounds.maxs[axis], 0));
      const objectClearanceM = norm(outside);
      const objectClearanceScore = clamp(objectClearanceM / Math.max(scoring.reachability_object_clearance_m, 1e-9), 0, 1);
      const floorScore = clamp(handWorld[2] / Math.max(scoring.reachability_floor_clearance_m, 1e-9), 0, 1);
      const pregraspClearance = floorScore * objectClearanceScore;
      const proxy = clamp(0.65 * handRadialScore + 0.25 * clamp(sideScore, 0, 1) + 0.10 * pregraspClearance, 0, 1);
      return {
        approach_axis_world: approachAxisWorld,
        reachability_proxy: proxy,
        reachability_hand_radial: handRadialScore,
        reachability_side: clamp(sideScore, 0, 1),
        reachability_pregrasp_clearance: pregraspClearance,
        reachability_hand_radius_m: handRadius,
        reachability_object_radius_m: objectRadius,
        reachability_hand_side: handSide,
        reachability_target_side: targetSide,
        reachability_pregrasp_clearance_m: objectClearanceM,
        reachability_pregrasp_floor_margin_m: handWorld[2],
        reachability_pregrasp_x_world: handWorld[0],
        reachability_pregrasp_y_world: handWorld[1],
        reachability_pregrasp_z_world: handWorld[2],
      };
    }
    function liveFloorStatus(candidate, geometry, pose) {
      const points = [
        ...geometry.hand_vertices_obj,
        ...geometry.left_boxes.flatMap((box) => box.corners),
        ...geometry.right_boxes.flatMap((box) => box.corners),
      ];
      if (!points.length) return { status: "accepted", reason: "live_floor_unchecked", min_z_world: null };
      const minZ = Math.min(...transformPoints(points, pose).map((point) => point[2]));
      const margin = Number(data.scoring.floor_clearance_margin_m || 0);
      const accepted = minZ >= margin - 1e-8;
      return {
        status: accepted ? "accepted" : "rejected",
        reason: accepted ? "live_floor_clearance" : "live_floor_collision",
        min_z_world: minZ,
      };
    }
    function liveRecord(candidate) {
      const pose = currentPose();
      const geometry = adjustedGeometry(candidate);
      const reachability = liveReachability(candidate, geometry, pose);
      const floor = liveFloorStatus(candidate, geometry, pose);
      let topWeight = clamp(Number(data.scoring.top_grasp_score_weight || 0), 0, 1);
      let reachWeight = clamp(Number(data.scoring.reachability_proxy_score_weight || 0), 0, 1);
      let worldWeight = topWeight + reachWeight;
      if (worldWeight > 1.0) {
        topWeight /= worldWeight;
        reachWeight /= worldWeight;
        worldWeight = 1.0;
      }
      const objectWeight = 1.0 - worldWeight;
      const topDown = clamp(-Number(reachability.approach_axis_world[2]), 0, 1);
      const objectScore = Number(candidate.object_score || 0);
      const score = objectWeight * objectScore + topWeight * topDown + reachWeight * reachability.reachability_proxy;
      const components = {
        ...candidate.object_components,
        object_score: objectScore,
        top_down_approach: topDown,
        world_approach_z: Number(reachability.approach_axis_world[2]),
        top_grasp_score_weight: topWeight,
        reachability_proxy_score_weight: reachWeight,
        world_object_score_weight: objectWeight,
        hand_roll_offset_deg: Number(state.handRollDeg),
        ...reachability,
        score,
      };
      return {
        pose_index: state.poseIndex,
        grasp_id: candidate.grasp_id,
        status: floor.status,
        reason: floor.reason,
        min_z_world: floor.min_z_world,
        runtime_rank: null,
        object_score: objectScore,
        score,
        score_delta: score - objectScore,
        top_down_approach: topDown,
        ...reachability,
        components,
      };
    }
    let liveRecordCacheKey = "";
    let liveRecordCache = { records: [], byId: new Map() };
    function liveStateKey() {
      return [
        Number(state.objectX).toFixed(4),
        Number(state.objectY).toFixed(4),
        Number(state.objectYawDeg).toFixed(3),
        state.supportFace,
        Number(state.handRollDeg).toFixed(3),
      ].join("|");
    }
    function computeCurrentRecords() {
      const records = data.candidates.map((candidate) => ({ candidate, record: liveRecord(candidate) }));
      records.sort((a, b) => Number(b.record.score) - Number(a.record.score) || compareId(a.candidate, b.candidate));
      records.forEach((item, index) => { item.record.runtime_rank = index + 1; });
      return { records, byId: new Map(records.map((item) => [item.candidate.grasp_id, item.record])) };
    }
    function currentRecords() {
      const key = liveStateKey();
      if (key !== liveRecordCacheKey) {
        liveRecordCacheKey = key;
        liveRecordCache = computeCurrentRecords();
      }
      return liveRecordCache.records;
    }
    function liveRecordFor(candidate) {
      currentRecords();
      return liveRecordCache.byId.get(candidate.grasp_id) || liveRecord(candidate);
    }
    let defaultRollStatusCacheKey = "";
    let defaultRollStatusCache = new Map();
    function defaultRollStateKey() {
      return [
        Number(state.objectX).toFixed(4),
        Number(state.objectY).toFixed(4),
        Number(state.objectYawDeg).toFixed(3),
        state.supportFace,
      ].join("|");
    }
    function defaultRollFloorStatus(candidate) {
      const key = defaultRollStateKey();
      if (key !== defaultRollStatusCacheKey) {
        defaultRollStatusCacheKey = key;
        defaultRollStatusCache = new Map();
      }
      if (!defaultRollStatusCache.has(candidate.grasp_id)) {
        defaultRollStatusCache.set(
          candidate.grasp_id,
          liveFloorStatus(candidate, adjustedGeometry(candidate, 0), currentPose())
        );
      }
      return defaultRollStatusCache.get(candidate.grasp_id);
    }
    function currentSummary() {
      const records = currentRecords();
      const accepted = records.filter((item) => item.record.status === "accepted");
      const defaultFeasible = records.filter((item) => defaultRollFloorStatus(item.candidate).status === "accepted");
      const topClear = accepted[0];
      const topScored = records[0];
      const acceptedScores = accepted.map((item) => item.record.score);
      return {
        top_grasp_id: topClear ? topClear.candidate.grasp_id : null,
        top_score: topClear ? topClear.record.score : null,
        top_scored_grasp_id: topScored ? topScored.candidate.grasp_id : null,
        top_scored_score: topScored ? topScored.record.score : null,
        default_feasible_count: defaultFeasible.length,
        accepted_count: accepted.length,
        candidate_count: data.candidates.length,
        average_score: acceptedScores.length ? acceptedScores.reduce((sum, value) => sum + value, 0) / acceptedScores.length : null,
      };
    }
    function poseScore(candidate) { return liveRecordFor(candidate); }
    function fmt(value, digits = 6) {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "n/a";
      return Number(value).toFixed(digits);
    }
    function fmtVec(vec) {
      if (!vec) return "n/a";
      return `(${vec.map((value) => value >= 0 ? `+${Number(value).toFixed(4)}` : Number(value).toFixed(4)).join(", ")})`;
    }
    function statusClass(score) {
      if (score === null || score === undefined) return "flat";
      if (Number(score) > 1e-6) return "gain";
      if (Number(score) < -1e-6) return "loss";
      return "flat";
    }
    function scoreDeltaColor(value) {
      if (value === null || value === undefined) return "#6d7d76";
      if (Number(value) > 1e-6) return "#168354";
      if (Number(value) < -1e-6) return "#bf4938";
      return "#6d7d76";
    }
    function componentValue(components, key, digits = 6) {
      return components && components[key] !== undefined && components[key] !== null ? fmt(components[key], digits) : "n/a";
    }
    function transformPoint(point, pose = currentPose()) {
      const r = pose.rotation_world_from_object;
      return [
        r[0][0] * point[0] + r[0][1] * point[1] + r[0][2] * point[2] + pose.position_world[0],
        r[1][0] * point[0] + r[1][1] * point[1] + r[1][2] * point[2] + pose.position_world[1],
        r[2][0] * point[0] + r[2][1] * point[1] + r[2][2] * point[2] + pose.position_world[2],
      ];
    }
    function transformPoints(points, pose = currentPose()) {
      return (points || []).map((point) => transformPoint(point, pose));
    }
    function candidateWorldPoint(candidate, key) {
      return transformPoint(candidate[key]);
    }
    function pregraspPoint(record) {
      if (record.reachability_pregrasp_x_world === null || record.reachability_pregrasp_x_world === undefined) return null;
      return [record.reachability_pregrasp_x_world, record.reachability_pregrasp_y_world, record.reachability_pregrasp_z_world].map(Number);
    }
    function compareId(a, b) {
      return String(a.grasp_id).localeCompare(String(b.grasp_id));
    }
    function orderedCandidates() {
      const ranked = currentRecords();
      if (state.sortBy === "runtime_rank") return ranked.map((item) => item.candidate);
      const candidates = data.candidates.slice();
      if (state.sortBy === "object_rank") return candidates.sort((a, b) => Number(a.object_rank) - Number(b.object_rank) || compareId(a, b));
      if (state.sortBy === "score_gain") return candidates.sort((a, b) => {
        const as = poseScore(a).score_delta ?? -Infinity;
        const bs = poseScore(b).score_delta ?? -Infinity;
        return Number(bs) - Number(as) || compareId(a, b);
      });
      if (state.sortBy === "score_loss") return candidates.sort((a, b) => {
        const as = poseScore(a).score_delta ?? Infinity;
        const bs = poseScore(b).score_delta ?? Infinity;
        return Number(as) - Number(bs) || compareId(a, b);
      });
      return ranked.map((item) => item.candidate);
    }
    function visibleCandidates() {
      const candidates = orderedCandidates();
      return state.defaultRollFeasibleOnly
        ? candidates.filter((candidate) => defaultRollFloorStatus(candidate).status === "accepted")
        : candidates;
    }
    function rotate(point) {
      const shifted = point.map((value, axis) => value - worldCenter[axis]);
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];
      return [x1, cp * y1 + sp * z1, -sp * y1 + cp * z1];
    }
    function project(point) {
      const [x, y, z] = rotate(point);
      const scale = baseScale * state.zoom;
      return { x: 480 + state.panX + x * scale, y: 380 + state.panY - y * scale, depth: z };
    }
    function wrapAngle(angle) {
      const tau = Math.PI * 2;
      let wrapped = angle % tau;
      if (wrapped <= -Math.PI) wrapped += tau;
      else if (wrapped > Math.PI) wrapped -= tau;
      return wrapped;
    }
    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }
    function addSvg(tag, attrs) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => {
        if (value !== null && value !== undefined) node.setAttribute(key, String(value));
      });
      scene.appendChild(node);
      return node;
    }
    function drawLine(a, b, options = {}) {
      const pa = project(a), pb = project(b);
      addSvg("line", { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, stroke: options.stroke || "#555", "stroke-width": options.strokeWidth || 2, "stroke-opacity": options.opacity ?? 1, "stroke-dasharray": options.dash || "" });
    }
    function drawPoint(point, options = {}) {
      const p = project(point);
      addSvg("circle", { cx: p.x, cy: p.y, r: options.radius || 6, fill: options.fill || "#000", "fill-opacity": options.opacity ?? 1, stroke: options.stroke || "white", "stroke-width": options.strokeWidth || 2 });
    }
    function drawPolygon(points, options = {}) {
      const projected = points.map((point) => project(point));
      addSvg("polygon", { points: projected.map((point) => `${point.x},${point.y}`).join(" "), fill: options.fill || "none", "fill-opacity": options.fillOpacity ?? 1, stroke: options.stroke || "none", "stroke-width": options.strokeWidth || 1, "stroke-opacity": options.strokeOpacity ?? 1 });
    }
    function drawLabel(point, text, fill, dx = 8, dy = -8) {
      const p = project(point);
      const node = addSvg("text", { x: p.x + dx, y: p.y + dy, fill, "font-size": 14, "font-family": "IBM Plex Mono, monospace", "font-weight": 600 });
      node.textContent = text;
    }
    function drawBox(corners, color) {
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      const worldCorners = transformPoints(corners);
      edges.forEach(([s,e]) => drawLine(worldCorners[s], worldCorners[e], { stroke: color, strokeWidth: 1.8, opacity: 0.8 }));
    }
    function shadeColor(hex, factor) {
      const clean = hex.replace("#", "");
      const value = Number.parseInt(clean, 16);
      const r = (value >> 16) & 255;
      const g = (value >> 8) & 255;
      const b = value & 255;
      const scale = clamp(factor, 0, 1.4);
      return `#${[r,g,b].map((channel) => clamp(Math.round(channel * scale), 0, 255)).map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
    }
    function drawMeshEdges(vertices, edges, stroke, width, opacity) {
      edges.forEach(([start, end]) => drawLine(vertices[start], vertices[end], { stroke, strokeWidth: width, opacity }));
    }
    function drawTargetMesh(worldVertices) {
      if (state.meshRenderMode === "solid") {
        const faces = data.faces.map((face) => {
          const points = face.map((index) => worldVertices[index]);
          const rotated = points.map((point) => rotate(point));
          const edgeA = rotated[1].map((value, axis) => value - rotated[0][axis]);
          const edgeB = rotated[2].map((value, axis) => value - rotated[0][axis]);
          const normal = [
            edgeA[1] * edgeB[2] - edgeA[2] * edgeB[1],
            edgeA[2] * edgeB[0] - edgeA[0] * edgeB[2],
            edgeA[0] * edgeB[1] - edgeA[1] * edgeB[0],
          ];
          const depth = rotated.reduce((sum, point) => sum + point[2], 0) / rotated.length;
          return { points, normal, depth };
        });
        faces
          .filter((face) => face.normal[2] > 0)
          .sort((a, b) => a.depth - b.depth)
          .forEach((face) => {
            const norm = Math.hypot(face.normal[0], face.normal[1], face.normal[2]) || 1;
            const light = 0.45 + 0.55 * (face.normal[2] / norm);
            drawPolygon(face.points, { fill: shadeColor("#4f6b5f", 0.7 + light * 0.45), fillOpacity: 0.92, stroke: "#32453d", strokeWidth: 1.2, strokeOpacity: 0.55 });
          });
        return;
      }
      drawMeshEdges(worldVertices, data.edges, "#4f6b5f", 2.0, 0.8);
    }
    function drawFloor() {
      const pad = Math.max((worldMax[0] - worldMin[0]), (worldMax[1] - worldMin[1])) * 0.08;
      const corners = [
        [worldMin[0] - pad, worldMin[1] - pad, 0],
        [worldMax[0] + pad, worldMin[1] - pad, 0],
        [worldMax[0] + pad, worldMax[1] + pad, 0],
        [worldMin[0] - pad, worldMax[1] + pad, 0],
      ];
      drawPolygon(corners, { fill: "#2563eb", fillOpacity: 0.12, stroke: "#2563eb", strokeWidth: 2, strokeOpacity: 0.65 });
      for (let i = 0; i < corners.length; i += 1) drawLine(corners[i], corners[(i + 1) % corners.length], { stroke: "#2563eb", strokeWidth: 2, opacity: 0.75, dash: "10 6" });
    }
    function drawBase() {
      drawPoint([0, 0, 0], { fill: "#111827", radius: 7, stroke: "white", strokeWidth: 2 });
      drawLine([0, 0, 0], [0.08, 0, 0], { stroke: "#dc2626", strokeWidth: 2.4, opacity: 0.9 });
      drawLine([0, 0, 0], [0, 0.08, 0], { stroke: "#16a34a", strokeWidth: 2.4, opacity: 0.9 });
      drawLabel([0, 0, 0], "base", "#111827", 9, -9);
    }
    function drawHandMesh(candidate, geometry) {
      if (!candidate.franka_hand_faces || !candidate.franka_hand_vertices_obj) return;
      const vertices = transformPoints(geometry.hand_vertices_obj);
      candidate.franka_hand_faces.forEach((face) => {
        drawLine(vertices[face[0]], vertices[face[1]], { stroke: "#8f5a12", strokeWidth: 1.1, opacity: 0.35 });
        drawLine(vertices[face[1]], vertices[face[2]], { stroke: "#8f5a12", strokeWidth: 1.1, opacity: 0.35 });
        drawLine(vertices[face[2]], vertices[face[0]], { stroke: "#8f5a12", strokeWidth: 1.1, opacity: 0.35 });
      });
    }
    function drawContactGrid(gridPoints, selectedPoint, gridColor, selectedColor) {
      transformPoints(gridPoints || []).forEach((point) => {
        drawPoint(point, { fill: gridColor, radius: 2.4, opacity: 0.8, stroke: "white", strokeWidth: 0.8 });
      });
      if (selectedPoint) drawPoint(transformPoint(selectedPoint), { fill: selectedColor, radius: 4.2, opacity: 1.0, stroke: "white", strokeWidth: 1.2 });
    }
    function renderPoseSummary() {
      const pose = currentPose();
      const summary = currentSummary();
      poseSummary.textContent = [
        `pose:       ${pose.label}`,
        `support:    ${pose.support_face}`,
        `yaw_deg:    ${fmt(pose.yaw_deg, 1)}`,
        `xy_world:   (${fmt(pose.xy_world[0], 3)}, ${fmt(pose.xy_world[1], 3)})`,
        `top_clear:  ${summary.top_grasp_id || "none"} ${fmt(summary.top_score, 6)}`,
        `top_scored: ${summary.top_scored_grasp_id || "none"} ${fmt(summary.top_scored_score, 6)}`,
        `shown:      ${summary.default_feasible_count} default-roll feasible`,
        `live_clear: ${summary.accepted_count} / ${summary.candidate_count}`,
        `avg_score:  ${fmt(summary.average_score, 6)}`,
      ].join("\\n");
    }
    function renderList() {
      graspList.replaceChildren();
      const candidates = visibleCandidates();
      candidates.forEach((candidate, index) => {
        const record = poseScore(candidate);
        const defaultStatus = defaultRollFloorStatus(candidate);
        const delta = record.score_delta;
        const item = document.createElement("button");
        item.type = "button";
        item.className = `item${index === state.selectedIndex ? " active" : ""}`;
        const runtimeRank = record.runtime_rank === null || record.runtime_rank === undefined ? "n/a" : `#${record.runtime_rank}`;
        item.innerHTML = `
          <div class="item-rank">${runtimeRank} ${candidate.grasp_id} object #${candidate.object_rank}</div>
          <div class="item-main">
            <div class="item-label status ${record.status || "rejected"}">${record.status || "missing"}</div>
            <div class="item-score delta ${statusClass(delta)}">delta=${fmt(delta, 4)}</div>
          </div>
          <div class="item-meta">runtime=${fmt(record.score, 4)} object=${fmt(candidate.object_score, 4)}<br>default=${defaultStatus.status} live=${record.status}<br>reach=${fmt(record.reachability_proxy, 4)} top=${fmt(record.top_down_approach, 4)}<br>hand_r=${fmt(record.reachability_hand_radius_m, 4)} side=${fmt(record.reachability_hand_side, 3)}</div>
        `;
        item.addEventListener("click", () => { state.selectedIndex = index; render(); });
        graspList.appendChild(item);
      });
    }
    function scoreTrace(candidate) {
      return candidate.pose_scores.map((record, index) => {
        const pose = data.poses[index];
        const rank = record.runtime_rank === null || record.runtime_rank === undefined ? "n/a" : `#${record.runtime_rank}`;
        return `${String(index + 1).padStart(2, "0")} ${pose.support_face} yaw=${fmt(pose.yaw_deg, 0)} xy=(${fmt(pose.xy_world[0], 2)},${fmt(pose.xy_world[1], 2)}) ${record.status} rank=${rank} score=${fmt(record.score, 5)} delta=${fmt(record.score_delta, 5)} reach=${fmt(record.reachability_proxy, 4)}`;
      }).join("\\n");
    }
    function renderScene(candidate) {
      scene.replaceChildren();
      const worldVertices = transformPoints(data.vertices_obj);
      drawFloor();
      drawBase();
      drawTargetMesh(worldVertices);
      const record = poseScore(candidate);
      const geometry = adjustedGeometry(candidate);
      geometry.left_boxes.forEach((box) => drawBox(box.corners, "#d97706"));
      geometry.right_boxes.forEach((box) => drawBox(box.corners, "#d97706"));
      drawHandMesh(candidate, geometry);
      drawContactGrid(geometry.left_grid, geometry.left_anchor, "#0f766e", "#14b8a6");
      drawContactGrid(geometry.right_grid, geometry.right_anchor, "#0f766e", "#14b8a6");
      const contactA = candidateWorldPoint(candidate, "contact_point_a_obj");
      const contactB = candidateWorldPoint(candidate, "contact_point_b_obj");
      const graspCenter = candidateWorldPoint(candidate, "grasp_position_obj");
      const pregrasp = pregraspPoint(record);
      const deltaColor = scoreDeltaColor(record.score_delta);
      drawLine(contactA, contactB, { stroke: deltaColor, strokeWidth: 3, opacity: 0.95 });
      drawPoint(graspCenter, { fill: record.status === "accepted" ? "#15803d" : "#b91c1c", radius: 7 });
      drawPoint(contactA, { fill: "#c8452d", radius: 6 });
      drawPoint(contactB, { fill: "#1f7c60", radius: 6 });
      if (pregrasp) {
        drawLine(graspCenter, pregrasp, { stroke: "#7c3aed", strokeWidth: 2.4, opacity: 0.85, dash: "6 5" });
        drawPoint(pregrasp, { fill: "#7c3aed", radius: 6, stroke: "white", strokeWidth: 1.5 });
      }
      drawLabel(graspCenter, `${candidate.grasp_id} score=${fmt(record.score, 3)}`, deltaColor);
    }
    function renderDetails(candidate) {
      const pose = currentPose();
      const record = poseScore(candidate);
      const defaultStatus = defaultRollFloorStatus(candidate);
      const components = record.components || {};
      const pregrasp = pregraspPoint(record);
      details.textContent = [
        ...data.metadata_lines,
        "",
        `pose:             ${pose.label}`,
        `pose_position_w:  ${fmtVec(pose.position_world)}`,
        `pose_quat_xyzw:   ${fmtVec(pose.orientation_xyzw_world)}`,
        "",
        `grasp_id:         ${candidate.grasp_id}`,
        `default_roll:     ${defaultStatus.status}`,
        `default_reason:   ${defaultStatus.reason}`,
        `live_status:      ${record.status}`,
        `live_reason:      ${record.reason}`,
        `live_min_z_world: ${fmt(record.min_z_world, 6)}`,
        `object_rank:      ${candidate.object_rank}`,
        `runtime_rank:     ${record.runtime_rank ?? "n/a"}`,
        `object_score:     ${fmt(candidate.object_score, 6)}`,
        `runtime_score:    ${fmt(record.score, 6)}`,
        `score_delta:      ${fmt(record.score_delta, 6)}`,
        `top_down:         ${fmt(record.top_down_approach, 6)}`,
        `reachability:     ${fmt(record.reachability_proxy, 6)}`,
        `reach_hand_rad:   ${fmt(record.reachability_hand_radial, 6)}`,
        `reach_side:       ${fmt(record.reachability_side, 6)}`,
        `hand_radius_m:    ${fmt(record.reachability_hand_radius_m, 6)}`,
        `hand_side:        ${fmt(record.reachability_hand_side, 6)}`,
        `target_side:      ${fmt(record.reachability_target_side, 6)}`,
        `pregrasp_world:   ${fmtVec(pregrasp)}`,
        "",
        "object-frame score:",
        JSON.stringify(candidate.object_components, null, 2),
        "",
        "runtime components:",
        JSON.stringify(components, null, 2),
        "",
        "score trace:",
        scoreTrace(candidate),
        "",
        `jaw_width:        ${fmt(candidate.jaw_width, 6)} m`,
        `roll_angle_rad:   ${fmt(candidate.roll_angle_rad, 6)}`,
        `contact_offset_x: ${fmt(candidate.contact_patch_lateral_offset_m, 6)} m`,
        `contact_offset_z: ${fmt(candidate.contact_patch_approach_offset_m, 6)} m`,
      ].join("\\n");
    }
    function render() {
      poseSelect.value = String(state.poseIndex);
      const candidates = visibleCandidates();
      renderPoseSummary();
      if (candidates.length === 0) {
        graspList.replaceChildren();
        scene.replaceChildren();
        details.textContent = [...data.metadata_lines, "No candidates to display for this pose/filter."].join("\\n");
        return;
      }
      if (state.selectedIndex >= candidates.length) state.selectedIndex = 0;
      const candidate = candidates[state.selectedIndex];
      renderList();
      renderScene(candidate);
      renderDetails(candidate);
    }
    let sceneRenderPending = false;
    function renderCurrentScene() {
      const candidates = visibleCandidates();
      if (candidates.length === 0) {
        scene.replaceChildren();
        return;
      }
      if (state.selectedIndex >= candidates.length) state.selectedIndex = 0;
      renderScene(candidates[state.selectedIndex]);
    }
    function scheduleSceneRender() {
      if (sceneRenderPending) return;
      sceneRenderPending = true;
      window.requestAnimationFrame(() => {
        sceneRenderPending = false;
        renderCurrentScene();
      });
    }
    function stepPose(delta) {
      state.poseIndex = (state.poseIndex + delta + data.poses.length) % data.poses.length;
      state.selectedIndex = 0;
      setStateFromPose(data.poses[state.poseIndex]);
      render();
    }
    function stepGrasp(delta) {
      const candidates = visibleCandidates();
      if (candidates.length === 0) return;
      state.selectedIndex = (state.selectedIndex + delta + candidates.length) % candidates.length;
      render();
    }
    window.addEventListener("keydown", (event) => {
      if (event.key === "PageUp" || event.key === "ArrowUp") { event.preventDefault(); stepPose(-1); }
      if (event.key === "PageDown" || event.key === "ArrowDown") { event.preventDefault(); stepPose(1); }
      if (event.key === "ArrowLeft") { event.preventDefault(); stepGrasp(-1); }
      if (event.key === "ArrowRight") { event.preventDefault(); stepGrasp(1); }
    });
    prevPoseBtn.addEventListener("click", () => stepPose(-1));
    nextPoseBtn.addEventListener("click", () => stepPose(1));
    prevGraspBtn.addEventListener("click", () => stepGrasp(-1));
    nextGraspBtn.addEventListener("click", () => stepGrasp(1));
    poseSelect.addEventListener("change", () => {
      state.poseIndex = Number(poseSelect.value);
      setStateFromPose(data.poses[state.poseIndex]);
      state.selectedIndex = 0;
      render();
    });
    xSlider.addEventListener("input", () => {
      state.objectX = Number(xSlider.value);
      syncSliderOutputs();
      render();
    });
    ySlider.addEventListener("input", () => {
      state.objectY = Number(ySlider.value);
      syncSliderOutputs();
      render();
    });
    objectYawSlider.addEventListener("input", () => {
      state.objectYawDeg = Number(objectYawSlider.value);
      syncSliderOutputs();
      render();
    });
    handRollSlider.addEventListener("input", () => {
      state.handRollDeg = Number(handRollSlider.value);
      syncSliderOutputs();
      render();
    });
    sortSelect.addEventListener("change", () => {
      const current = visibleCandidates()[state.selectedIndex];
      state.sortBy = sortSelect.value;
      const candidates = visibleCandidates();
      const nextIndex = current ? candidates.findIndex((candidate) => candidate.grasp_id === current.grasp_id) : 0;
      state.selectedIndex = Math.max(0, nextIndex);
      render();
    });
    meshModeBtn.addEventListener("click", () => {
      state.meshRenderMode = state.meshRenderMode === "wireframe" ? "solid" : "wireframe";
      meshModeBtn.textContent = state.meshRenderMode === "wireframe" ? "Solid Mesh" : "Wireframe Mesh";
      renderCurrentScene();
    });
    feasibleOnlyBtn.title = "The list is fixed to grasps whose default hand roll clears the floor at the current x/y/yaw pose.";
    scene.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 && event.button !== 1) return;
      event.preventDefault();
      state.dragging = true;
      state.dragMode = event.button === 1 ? "pan" : "rotate";
      state.lastPointerX = event.clientX;
      state.lastPointerY = event.clientY;
      state.pointerId = event.pointerId;
      scene.setPointerCapture(event.pointerId);
      scene.style.cursor = state.dragMode === "pan" ? "move" : "grabbing";
    });
    function stopDragging() {
      state.dragging = false;
      state.pointerId = null;
      scene.style.cursor = "grab";
    }
    scene.addEventListener("pointerup", (event) => {
      if (state.pointerId === event.pointerId) stopDragging();
    });
    scene.addEventListener("pointercancel", () => { stopDragging(); });
    scene.addEventListener("pointermove", (event) => {
      if (!state.dragging || (state.pointerId !== null && event.pointerId !== state.pointerId)) return;
      const dx = event.clientX - state.lastPointerX;
      const dy = event.clientY - state.lastPointerY;
      state.lastPointerX = event.clientX;
      state.lastPointerY = event.clientY;
      if (state.dragMode === "pan") {
        state.panX += dx;
        state.panY += dy;
      } else {
        state.yaw = wrapAngle(state.yaw + dx * 0.01);
        state.pitch = wrapAngle(state.pitch - dy * 0.01);
      }
      scheduleSceneRender();
    });
    scene.addEventListener("wheel", (event) => {
      event.preventDefault();
      const zoomFactor = event.deltaY < 0 ? 1.08 : 1 / 1.08;
      state.zoom = clamp(state.zoom * zoomFactor, 0.35, 4.0);
      scheduleSceneRender();
    }, { passive: false });
    scene.style.cursor = "grab";
    scene.addEventListener("contextmenu", (event) => event.preventDefault());
    render();
  </script>
</body>
</html>
""".replace("__DATA_JSON__", data_json)


def write_pose_score_sweep_html(output_html: Path, payload: dict[str, object]) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, sort_keys=True)
    output_html.write_text(_html_document(data_json), encoding="utf-8")


def _source_from_args(
    args: argparse.Namespace,
) -> tuple[TriangleMesh, list[SavedGraspCandidate], str, dict[str, object]]:
    config_payload: dict[str, object] = {}
    if args.config is not None:
        config_payload = _load_yaml(args.config)

    if args.input_json is not None:
        mesh_local, candidates, source_label = load_candidates_from_bundle(args.input_json)
        return mesh_local, candidates, source_label, config_payload

    raw_geometry = dict(config_payload.get("geometry", {}))
    target_mesh_path = args.target_mesh or raw_geometry.get("target_mesh_path")
    if target_mesh_path in ("", None):
        raise ValueError("Provide --input-json, --target-mesh, or --config with geometry.target_mesh_path.")
    mesh_scale = float(args.mesh_scale if args.mesh_scale is not None else raw_geometry.get("mesh_scale", 1.0))
    generator_config = _generator_config_from_payload(config_payload)
    if args.num_surface_samples is not None:
        generator_config = AntipodalGraspGeneratorConfig(
            num_surface_samples=int(args.num_surface_samples),
            min_jaw_width=generator_config.min_jaw_width,
            max_jaw_width=generator_config.max_jaw_width,
            antipodal_cosine_threshold=generator_config.antipodal_cosine_threshold,
            roll_angles_rad=generator_config.roll_angles_rad,
            max_pair_checks=generator_config.max_pair_checks,
            detailed_finger_contact_gap_m=generator_config.detailed_finger_contact_gap_m,
            rng_seed=generator_config.rng_seed,
        )
    if args.max_pair_checks is not None:
        generator_config = AntipodalGraspGeneratorConfig(
            num_surface_samples=generator_config.num_surface_samples,
            min_jaw_width=generator_config.min_jaw_width,
            max_jaw_width=generator_config.max_jaw_width,
            antipodal_cosine_threshold=generator_config.antipodal_cosine_threshold,
            roll_angles_rad=generator_config.roll_angles_rad,
            max_pair_checks=int(args.max_pair_checks),
            detailed_finger_contact_gap_m=generator_config.detailed_finger_contact_gap_m,
            rng_seed=generator_config.rng_seed,
        )
    mesh_local, candidates, source_label = generate_candidates_for_mesh(
        target_mesh_path=str(target_mesh_path),
        mesh_scale=mesh_scale,
        generator_config=generator_config,
    )
    return mesh_local, candidates, source_label, config_payload


def _planning_from_config(config_payload: dict[str, object]) -> PlanningConfig:
    if config_payload:
        return _planning_config(config_payload)
    return PlanningConfig()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML report that rescores one grasp set across many object floor poses.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-json", type=Path, help="Existing saved grasp bundle JSON to rescore.")
    source.add_argument("--target-mesh", help="Asset-relative or absolute mesh path to generate grasps for.")
    parser.add_argument("--config", type=Path, help="Pipeline YAML to reuse geometry/planning generation settings.")
    parser.add_argument(
        "--mesh-scale", type=float, help="Mesh scale for --target-mesh; overrides config geometry.mesh_scale."
    )
    parser.add_argument("--num-surface-samples", type=int, help="Override generation sample count.")
    parser.add_argument("--max-pair-checks", type=int, help="Override generation pair-check cap.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=300,
        help="Maximum object-score-ranked candidates to rescore across all poses; <=0 means all.",
    )
    parser.add_argument("--max-display", type=int, default=120, help="Maximum candidates embedded in the HTML.")
    parser.add_argument("--top-per-pose", type=int, default=5, help="Always keep this many top grasps from each pose.")
    parser.add_argument(
        "--support-faces",
        help="Comma-separated support faces or 'all'. Defaults to config pickup_pose.support_face or neg_z.",
    )
    parser.add_argument("--yaw-deg", default="0,90,180,270", help="Comma-separated yaw angles in degrees.")
    parser.add_argument("--x-values", help="Comma-separated object x positions in world/base frame.")
    parser.add_argument("--y-values", help="Comma-separated object y positions in world/base frame.")
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("artifacts/grasp_pose_score_sweep.html"),
        help="Output HTML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mesh_local, candidates, source_label, config_payload = _source_from_args(args)
    if not candidates:
        raise RuntimeError("No grasp candidates were available for pose score sweep.")
    planning = _planning_from_config(config_payload)
    default_support_face, base_xy = _base_pickup_defaults(config_payload)
    support_faces = _parse_support_faces(args.support_faces, default=(default_support_face,))
    yaw_values = _parse_csv_floats(args.yaw_deg, default=(0.0, 90.0, 180.0, 270.0))
    x_values = _parse_csv_floats(args.x_values, default=_default_x_values(base_xy[0]))
    y_values = _parse_csv_floats(args.y_values, default=_default_y_values(base_xy[1]))
    pose_records = build_floor_pose_records(
        mesh_local=mesh_local,
        support_faces=support_faces,
        yaw_deg_values=yaw_values,
        x_values=x_values,
        y_values=y_values,
    )
    payload = build_pose_score_sweep_payload(
        mesh_local=mesh_local,
        candidates=list(candidates),
        source_label=source_label,
        planning=planning,
        pose_records=pose_records,
        max_candidates=int(args.max_candidates),
        max_display=int(args.max_display),
        top_per_pose=int(args.top_per_pose),
    )
    write_pose_score_sweep_html(args.output_html, payload)
    print(
        f"Wrote {args.output_html} with {payload['display_count']} displayed grasps, "
        f"{payload['scored_candidate_count']} scored grasps, and {payload['pose_count']} poses."
    )


if __name__ == "__main__":
    main()
