"""Planning-only in-air handover fallback helpers."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from grasp_planning.grasping.collision import (
    BoxCollisionPrimitive,
    FrankaHandFingerCollisionModel,
    MeshCollisionPrimitive,
    trimesh,
)
from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspCandidate,
    accepted_grasps,
    evaluate_saved_grasps_against_pickup_pose,
    quat_to_rotmat_xyzw,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose

from .fabrica_pipeline import GroundRecheckResult, PlanningConfig, Stage1Result

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HandoverGraspPair:
    transfer_grasp: SavedGraspCandidate
    final_grasp: SavedGraspCandidate
    status: str
    reason: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HandoverFallbackResult:
    target_mesh_path: str
    mesh_scale: float
    source_frame_origin_obj_world: tuple[float, float, float]
    source_frame_orientation_xyzw_obj_world: tuple[float, float, float, float]
    initial_object_pose_world: ObjectWorldPose
    accepted_pairs: tuple[HandoverGraspPair, ...]
    rejected_pairs: tuple[HandoverGraspPair, ...]
    transfer_floor_status_counts: dict[str, int]
    metadata: dict[str, object]

    @property
    def selected_pair(self) -> HandoverGraspPair | None:
        return self.accepted_pairs[0] if self.accepted_pairs else None


def _candidate_score(candidate: SavedGraspCandidate) -> float:
    return float(candidate.score) if candidate.score is not None else float("-inf")


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


def _pose_payload(pose: ObjectWorldPose) -> dict[str, object]:
    return {
        "position_world": list(pose.position_world),
        "orientation_xyzw_world": list(pose.orientation_xyzw_world),
    }


def _pair_payload(pair: HandoverGraspPair) -> dict[str, object]:
    return {
        "status": pair.status,
        "reason": pair.reason,
        "score": pair.score,
        "transfer_grasp": _candidate_payload(pair.transfer_grasp),
        "final_grasp": _candidate_payload(pair.final_grasp),
        "metadata": pair.metadata,
    }


def _contact_pair_key(
    candidate: SavedGraspCandidate, *, tolerance_m: float = 1.0e-5
) -> tuple[tuple[int, int, int], ...]:
    points = (
        tuple(int(round(float(value) / tolerance_m)) for value in candidate.contact_point_a_obj),
        tuple(int(round(float(value) / tolerance_m)) for value in candidate.contact_point_b_obj),
    )
    return tuple(sorted(points))


def _same_contact_pair(left: SavedGraspCandidate, right: SavedGraspCandidate) -> bool:
    return _contact_pair_key(left) == _contact_pair_key(right)


def _candidate_primitives(
    candidate: SavedGraspCandidate,
    *,
    contact_gap_m: float,
) -> tuple[BoxCollisionPrimitive | MeshCollisionPrimitive, ...]:
    model = FrankaHandFingerCollisionModel(
        contact_gap_m=contact_gap_m,
        contact_patch_lateral_offset_m=float(candidate.contact_patch_lateral_offset_m),
        contact_patch_approach_offset_m=float(candidate.contact_patch_approach_offset_m),
    )
    return model.primitives_for_grasp(
        grasp_rotmat=quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj),
        contact_point_a=np.asarray(candidate.contact_point_a_obj, dtype=float),
        contact_point_b=np.asarray(candidate.contact_point_b_obj, dtype=float),
    )


def _primitive_mesh(primitive: BoxCollisionPrimitive | MeshCollisionPrimitive):
    if trimesh is None:
        raise RuntimeError("trimesh is required for handover hand collision checks.")
    if isinstance(primitive, BoxCollisionPrimitive):
        return trimesh.creation.box(extents=2.0 * primitive.half_extents, transform=primitive.transform_matrix_obj())
    return trimesh.Trimesh(vertices=primitive.vertices_obj, faces=primitive.faces, process=False)


def _collision_manager_for_primitives(
    fixed_primitives: tuple[BoxCollisionPrimitive | MeshCollisionPrimitive, ...],
):
    if trimesh is None:
        raise RuntimeError("trimesh is required for handover hand collision checks.")
    manager = trimesh.collision.CollisionManager()
    for index, primitive in enumerate(fixed_primitives):
        manager.add_object(f"fixed_{index}", _primitive_mesh(primitive))
    return manager


def _manager_collides(
    manager,
    query_primitives: tuple[BoxCollisionPrimitive | MeshCollisionPrimitive, ...],
) -> bool:
    return any(
        bool(manager.in_collision_single(_primitive_mesh(primitive), return_data=False))
        for primitive in query_primitives
    )


def _pair_score(transfer_grasp: SavedGraspCandidate, final_grasp: SavedGraspCandidate) -> float:
    final_score = _candidate_score(final_grasp)
    transfer_score = _candidate_score(transfer_grasp)
    return float(final_score * 1.0e6 + transfer_score)


def _sorted_candidates(candidates: Iterable[SavedGraspCandidate]) -> list[SavedGraspCandidate]:
    return sorted(candidates, key=lambda candidate: (-_candidate_score(candidate), candidate.grasp_id))


def plan_handover_fallback(
    *,
    stage1: Stage1Result,
    direct_stage2: GroundRecheckResult,
    planning: PlanningConfig,
    max_final_candidates: int = 40,
    max_transfer_candidates: int = 80,
    max_pair_checks: int = 1000,
    max_accepted_pairs: int = 24,
    max_rejected_pairs: int = 100,
    transfer_floor_clearance_margin_m: float | None = None,
) -> HandoverFallbackResult | None:
    """Search reverse handover pairs after direct pickup and floor regrasp fail.

    Final grasps come from stage-1 assembly-feasible candidates. Transfer grasps
    come from raw candidates that can pick the object from the current floor
    pose. Pair feasibility is tested by full Franka hand/finger collision in the
    object frame.
    """

    raw_candidates = tuple(stage1.raw_candidates) if stage1.raw_candidates else tuple(stage1.bundle.candidates)
    final_candidates = tuple(_sorted_candidates(stage1.bundle.candidates)[: max(0, int(max_final_candidates))])
    if not raw_candidates or not final_candidates:
        return None

    floor_clearance_margin_m = (
        planning.floor_clearance_margin_m
        if transfer_floor_clearance_margin_m is None
        else float(transfer_floor_clearance_margin_m)
    )
    floor_statuses = evaluate_saved_grasps_against_pickup_pose(
        raw_candidates,
        object_pose_world=direct_stage2.pickup_pose_world,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
        floor_clearance_margin_m=floor_clearance_margin_m,
        contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
        contact_approach_offsets_m=planning.contact_approach_offsets_m,
    )
    transfer_candidates = tuple(
        _sorted_candidates(accepted_grasps(floor_statuses))[: max(0, int(max_transfer_candidates))]
    )
    floor_counts = dict(Counter(entry.reason if entry.status != "accepted" else "accepted" for entry in floor_statuses))
    if not transfer_candidates:
        return HandoverFallbackResult(
            target_mesh_path=stage1.bundle.target_mesh_path,
            mesh_scale=stage1.bundle.mesh_scale,
            source_frame_origin_obj_world=stage1.bundle.source_frame_origin_obj_world,
            source_frame_orientation_xyzw_obj_world=stage1.bundle.source_frame_orientation_xyzw_obj_world,
            initial_object_pose_world=direct_stage2.pickup_pose_world,
            accepted_pairs=(),
            rejected_pairs=(),
            transfer_floor_status_counts=floor_counts,
            metadata={
                "checked_pair_count": 0,
                "limit_reached": False,
                "final_candidate_count": len(final_candidates),
                "transfer_floor_feasible_count": 0,
                "raw_candidate_count": len(raw_candidates),
                "rejection_counts": {},
                "transfer_floor_clearance_margin_m": floor_clearance_margin_m,
            },
        )

    accepted_pairs: list[HandoverGraspPair] = []
    rejected_pairs: list[HandoverGraspPair] = []
    rejection_counts: Counter[str] = Counter()
    final_primitives_by_id: dict[str, tuple[BoxCollisionPrimitive | MeshCollisionPrimitive, ...]] = {}
    checked_pair_count = 0
    limit_reached = False

    for final_grasp in final_candidates:
        final_primitives = final_primitives_by_id.setdefault(
            final_grasp.grasp_id,
            _candidate_primitives(final_grasp, contact_gap_m=planning.detailed_finger_contact_gap_m),
        )
        final_manager = _collision_manager_for_primitives(final_primitives)
        for transfer_grasp in transfer_candidates:
            if checked_pair_count >= max(0, int(max_pair_checks)):
                limit_reached = True
                break
            checked_pair_count += 1
            score = _pair_score(transfer_grasp, final_grasp)
            metadata = {
                "final_score": final_grasp.score,
                "transfer_score": transfer_grasp.score,
                "checked_pair_index": checked_pair_count,
            }
            if _same_contact_pair(transfer_grasp, final_grasp):
                reason = "same_contact_pair"
                rejection_counts[reason] += 1
                if len(rejected_pairs) < max(0, int(max_rejected_pairs)):
                    rejected_pairs.append(
                        HandoverGraspPair(
                            transfer_grasp=transfer_grasp,
                            final_grasp=final_grasp,
                            status="rejected",
                            reason=reason,
                            score=score,
                            metadata=metadata,
                        )
                    )
                continue

            transfer_primitives = _candidate_primitives(
                transfer_grasp,
                contact_gap_m=planning.detailed_finger_contact_gap_m,
            )
            if _manager_collides(final_manager, transfer_primitives):
                reason = "hand_collision"
                rejection_counts[reason] += 1
                if len(rejected_pairs) < max(0, int(max_rejected_pairs)):
                    rejected_pairs.append(
                        HandoverGraspPair(
                            transfer_grasp=transfer_grasp,
                            final_grasp=final_grasp,
                            status="rejected",
                            reason=reason,
                            score=score,
                            metadata=metadata,
                        )
                    )
                continue

            accepted_pairs.append(
                HandoverGraspPair(
                    transfer_grasp=transfer_grasp,
                    final_grasp=final_grasp,
                    status="accepted",
                    reason="transfer_floor_and_hand_clear",
                    score=score,
                    metadata=metadata,
                )
            )
            if len(accepted_pairs) >= max(1, int(max_accepted_pairs)) and len(rejected_pairs) >= max_rejected_pairs:
                limit_reached = True
                break
        if limit_reached:
            break

    accepted_pairs = sorted(
        accepted_pairs,
        key=lambda pair: (
            -_candidate_score(pair.final_grasp),
            -_candidate_score(pair.transfer_grasp),
            pair.final_grasp.grasp_id,
            pair.transfer_grasp.grasp_id,
        ),
    )
    return HandoverFallbackResult(
        target_mesh_path=stage1.bundle.target_mesh_path,
        mesh_scale=stage1.bundle.mesh_scale,
        source_frame_origin_obj_world=stage1.bundle.source_frame_origin_obj_world,
        source_frame_orientation_xyzw_obj_world=stage1.bundle.source_frame_orientation_xyzw_obj_world,
        initial_object_pose_world=direct_stage2.pickup_pose_world,
        accepted_pairs=tuple(accepted_pairs[: max(0, int(max_accepted_pairs))]),
        rejected_pairs=tuple(rejected_pairs[: max(0, int(max_rejected_pairs))]),
        transfer_floor_status_counts=floor_counts,
        metadata={
            "checked_pair_count": checked_pair_count,
            "limit_reached": limit_reached,
            "final_candidate_count": len(final_candidates),
            "transfer_floor_feasible_count": len(transfer_candidates),
            "raw_candidate_count": len(raw_candidates),
            "rejection_counts": dict(rejection_counts),
            "transfer_floor_clearance_margin_m": floor_clearance_margin_m,
            "max_final_candidates": int(max_final_candidates),
            "max_transfer_candidates": int(max_transfer_candidates),
            "max_pair_checks": int(max_pair_checks),
        },
    )


def write_handover_fallback_result(
    result: HandoverFallbackResult,
    output_json: str | Path,
    *,
    input_stage2_json: str | Path | None = None,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "target": {
            "mesh_path": result.target_mesh_path,
            "mesh_scale": result.mesh_scale,
            "source_frame_origin_obj_world": list(result.source_frame_origin_obj_world),
            "source_frame_orientation_xyzw_obj_world": list(result.source_frame_orientation_xyzw_obj_world),
        },
        "input_stage2_json": None if input_stage2_json is None else str(input_stage2_json),
        "initial_object_pose_world": _pose_payload(result.initial_object_pose_world),
        "selected_pair": None if result.selected_pair is None else _pair_payload(result.selected_pair),
        "accepted_pairs": [_pair_payload(pair) for pair in result.accepted_pairs],
        "rejected_pairs": [_pair_payload(pair) for pair in result.rejected_pairs],
        "transfer_floor_status_counts": result.transfer_floor_status_counts,
        "metadata": result.metadata,
    }
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
