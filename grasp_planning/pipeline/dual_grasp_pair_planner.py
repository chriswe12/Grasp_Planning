"""Bounded geometric pairing of holder and inserter end-effector grasps."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from grasp_planning.grasping.collision import (
    GRIPPER_COLLISION_MODEL_KUKA_Y,
    BoxCollisionPrimitive,
    MeshCollisionPrimitive,
    make_gripper_collision_model,
    trimesh,
    trimesh_fcl_backend_available,
)
from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspBundle,
    SavedGraspCandidate,
    linear_sweep_triangle_mesh,
    quat_to_rotmat_xyzw,
    rotmat_to_quat_xyzw,
    save_grasp_bundle,
    transform_primitive_to_world,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh
from grasp_planning.grasping.mesh_io import load_triangle_mesh
from grasp_planning.grasping.world_constraints import ObjectWorldPose

from .assembly_sequence import AssemblySequence, AssemblySequenceStep
from .fabrica_pipeline import GeometryConfig, PlanningConfig, generate_stage1_result
from .holder_state_feasibility import HolderStateFeasibilityResult
from .transition_symmetry import (
    TransitionSymmetryCandidate,
    compile_step_transition_symmetries,
    expand_grasp_candidates_by_symmetry,
    load_assembly_symmetry_records,
)

SCHEMA_VERSION = 2

REASON_ACCEPTED = "accepted"
REASON_ASSEMBLY_INSERTION_SWEEP_COLLISION = "assembly_insertion_sweep_collision"
REASON_INSERTER_RETREAT_COLLISION = "inserter_retreat_collision"
REASON_INSERTER_TABLE_COLLISION = "inserter_table_collision"
REASON_INSERTER_CLEARANCE_MARGIN_FAILED = "inserter_clearance_margin_failed"
REASON_END_EFFECTOR_SWEEP_COLLISION = "end_effector_sweep_collision"
REASON_PAIR_CLEARANCE_MARGIN_FAILED = "pair_clearance_margin_failed"


def _score(candidate: SavedGraspCandidate) -> float:
    return float(candidate.score) if candidate.score is not None else 0.0


def _candidate_sort_key(candidate: SavedGraspCandidate) -> tuple[float, str]:
    return (-_score(candidate), candidate.grasp_id)


@dataclass(frozen=True)
class DualGraspPairConfig:
    """Search, robustness, scoring, and debugger limits for Stage 3."""

    max_holder_candidates_per_step: int = 80
    max_inserter_candidates_per_step: int = 160
    max_candidates_per_cluster: int = 2
    contact_position_bin_m: float = 0.025
    axis_bin_deg: float = 30.0
    max_pair_checks: int = 4000
    max_accepted_pairs: int = 48
    max_rejected_pairs: int = 200
    max_collision_diagnostics_per_step: int = 24
    max_pairs_per_holder: int = 4
    max_pairs_per_inserter: int = 4
    matrix_unary_rejections_per_side: int = 12
    table_clearance_margin_m: float = 0.002
    geometry_clearance_margin_m: float = 0.0
    retreat_distance_m: float = 0.05
    path_samples: int = 21
    holder_score_weight: float = 0.45
    inserter_score_weight: float = 0.45
    clearance_score_weight: float = 0.10
    clearance_score_saturation_m: float = 0.05
    transition_symmetry_enabled: bool = False
    transition_symmetry_asset_path: str = ""
    transition_symmetry_geometry_tolerance_m: float = 0.001
    transition_symmetry_max_partial_assembly_transforms: int = 0
    transition_symmetry_max_incoming_transforms: int = 0

    def __post_init__(self) -> None:
        positive_ints = {
            "max_holder_candidates_per_step": self.max_holder_candidates_per_step,
            "max_inserter_candidates_per_step": self.max_inserter_candidates_per_step,
            "max_candidates_per_cluster": self.max_candidates_per_cluster,
            "max_pair_checks": self.max_pair_checks,
            "max_accepted_pairs": self.max_accepted_pairs,
            "max_pairs_per_holder": self.max_pairs_per_holder,
            "max_pairs_per_inserter": self.max_pairs_per_inserter,
            "path_samples": self.path_samples,
        }
        for name, value in positive_ints.items():
            if int(value) < 1:
                raise ValueError(f"pair_planning.{name} must be >= 1.")
        if self.path_samples < 3:
            raise ValueError("pair_planning.path_samples must be >= 3.")
        if (
            self.max_rejected_pairs < 0
            or self.max_collision_diagnostics_per_step < 0
            or self.matrix_unary_rejections_per_side < 0
            or self.transition_symmetry_max_partial_assembly_transforms < 0
            or self.transition_symmetry_max_incoming_transforms < 0
        ):
            raise ValueError("Pair rejection/debugger limits must be >= 0.")
        positive_floats = {
            "contact_position_bin_m": self.contact_position_bin_m,
            "axis_bin_deg": self.axis_bin_deg,
            "clearance_score_saturation_m": self.clearance_score_saturation_m,
        }
        for name, value in positive_floats.items():
            if float(value) <= 0.0:
                raise ValueError(f"pair_planning.{name} must be > 0.")
        nonnegative_floats = {
            "table_clearance_margin_m": self.table_clearance_margin_m,
            "geometry_clearance_margin_m": self.geometry_clearance_margin_m,
            "retreat_distance_m": self.retreat_distance_m,
            "holder_score_weight": self.holder_score_weight,
            "inserter_score_weight": self.inserter_score_weight,
            "clearance_score_weight": self.clearance_score_weight,
            "transition_symmetry_geometry_tolerance_m": (self.transition_symmetry_geometry_tolerance_m),
        }
        for name, value in nonnegative_floats.items():
            if float(value) < 0.0:
                raise ValueError(f"pair_planning.{name} must be >= 0.")
        if self.holder_score_weight + self.inserter_score_weight + self.clearance_score_weight <= 0.0:
            raise ValueError("At least one pair score weight must be positive.")

    def to_payload(self) -> dict[str, object]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}


@dataclass(frozen=True)
class InserterCandidateStatus:
    candidate: SavedGraspCandidate = field(repr=False, compare=False)
    status: str
    reason: str
    minimum_clearance_m: float | None
    details: dict[str, object] = field(default_factory=dict)

    @property
    def grasp_id(self) -> str:
        return self.candidate.grasp_id

    def to_payload(self) -> dict[str, object]:
        return {
            "grasp_id": self.grasp_id,
            "status": self.status,
            "reason": self.reason,
            "score": self.candidate.score,
            "minimum_clearance_m": self.minimum_clearance_m,
            "details": self.details,
        }


@dataclass(frozen=True)
class InserterGraspLibrary:
    step_id: str
    step_index: int
    incoming_part_id: str
    bundle: SavedGraspBundle
    source_frame_pose_assembly: ObjectWorldPose
    candidate_statuses: tuple[InserterCandidateStatus, ...]
    raw_candidate_count: int
    assembly_insertion_feasible_count: int
    collision_backend_name: str
    source_stage1_cache_key: str | None
    retreat_translation_assembly_m: tuple[float, float, float]

    @property
    def accepted_candidates(self) -> tuple[SavedGraspCandidate, ...]:
        return self.bundle.candidates

    @property
    def reason_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(status.reason for status in self.candidate_statuses).items()))

    def reference_payload(self, *, source_artifact: str) -> dict[str, object]:
        return {
            "artifact": source_artifact,
            "candidate_frame": "canonical incoming-part source frame",
            "source_frame_pose_assembly": {
                "position": list(self.source_frame_pose_assembly.position_world),
                "orientation_xyzw": list(self.source_frame_pose_assembly.orientation_xyzw_world),
            },
            "raw_candidate_count": self.raw_candidate_count,
            "assembly_insertion_feasible_count": self.assembly_insertion_feasible_count,
            "accepted_candidate_count": len(self.accepted_candidates),
            "reason_counts": self.reason_counts,
            "source_stage1_cache_key": self.source_stage1_cache_key,
        }


@dataclass(frozen=True)
class DualGraspPairEvaluation:
    pair_id: str
    holder_grasp_id: str
    inserter_grasp_id: str
    status: str
    reason: str
    score: float
    holder_score: float
    inserter_score: float
    clearance_score: float
    minimum_clearance_m: float | None
    collision_check: str
    details: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "holder_grasp_id": self.holder_grasp_id,
            "inserter_grasp_id": self.inserter_grasp_id,
            "status": self.status,
            "reason": self.reason,
            "score": self.score,
            "score_components": {
                "holder": self.holder_score,
                "inserter": self.inserter_score,
                "clearance": self.clearance_score,
            },
            "minimum_clearance_m": self.minimum_clearance_m,
            "collision_check": self.collision_check,
            "details": self.details,
        }


@dataclass(frozen=True)
class UnaryCandidateReference:
    grasp_id: str
    status: str
    reason: str
    score: float | None
    source_rank: int
    shortlisted: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "grasp_id": self.grasp_id,
            "status": self.status,
            "reason": self.reason,
            "score": self.score,
            "source_rank": self.source_rank,
            "shortlisted": self.shortlisted,
        }


@dataclass(frozen=True)
class DualGraspPairStepResult:
    step_id: str
    step_index: int
    incoming_part_id: str
    assembled_part_ids_before: tuple[str, ...]
    final_to_pre_translation_assembly_m: tuple[float, float, float]
    retreat_translation_assembly_m: tuple[float, float, float]
    holder_candidates: tuple[UnaryCandidateReference, ...]
    inserter_candidates: tuple[UnaryCandidateReference, ...]
    shortlisted_holder_ids: tuple[str, ...]
    shortlisted_inserter_ids: tuple[str, ...]
    matrix_holder_ids: tuple[str, ...]
    matrix_inserter_ids: tuple[str, ...]
    evaluations: tuple[DualGraspPairEvaluation, ...]
    retained_pair_ids: tuple[str, ...]
    detailed_rejected_pair_ids: tuple[str, ...]
    transition_candidates: tuple[TransitionSymmetryCandidate, ...]
    transition_symmetry_metadata: dict[str, object]
    metadata: dict[str, object]

    @property
    def retained_pairs(self) -> tuple[DualGraspPairEvaluation, ...]:
        retained = set(self.retained_pair_ids)
        return tuple(evaluation for evaluation in self.evaluations if evaluation.pair_id in retained)

    @property
    def reason_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(evaluation.reason for evaluation in self.evaluations).items()))

    def to_payload(
        self,
        *,
        holder_source_artifact: str,
        inserter_source_artifact: str,
    ) -> dict[str, object]:
        evaluation_by_id = {evaluation.pair_id: evaluation for evaluation in self.evaluations}
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "dual_grasp_pairs_step",
            "generated_by": "scripts/build_dual_grasp_pairs.py",
            "step_id": self.step_id,
            "step_index": self.step_index,
            "incoming_part_id": self.incoming_part_id,
            "assembled_part_ids_before": list(self.assembled_part_ids_before),
            "motion": {
                "insertion_translation_start_m": list(self.final_to_pre_translation_assembly_m),
                "insertion_translation_end_m": [0.0, 0.0, 0.0],
                "retreat_translation_end_m": list(self.retreat_translation_assembly_m),
            },
            "transition_symmetry": {
                **self.transition_symmetry_metadata,
                "candidates": [candidate.to_payload() for candidate in self.transition_candidates],
            },
            "candidate_sources": {
                "holder": {
                    "artifact": holder_source_artifact,
                    "candidate_collection": "candidates",
                    "state_selector": self.step_id,
                },
                "inserter": {
                    "artifact": inserter_source_artifact,
                    "candidate_collection": "candidates",
                },
            },
            "holder_candidates": [candidate.to_payload() for candidate in self.holder_candidates],
            "inserter_candidates": [candidate.to_payload() for candidate in self.inserter_candidates],
            "shortlisted_holder_ids": list(self.shortlisted_holder_ids),
            "shortlisted_inserter_ids": list(self.shortlisted_inserter_ids),
            "matrix_holder_ids": list(self.matrix_holder_ids),
            "matrix_inserter_ids": list(self.matrix_inserter_ids),
            "evaluations": [evaluation.to_payload() for evaluation in self.evaluations],
            "retained_pair_ids": list(self.retained_pair_ids),
            "retained_pairs": [evaluation_by_id[pair_id].to_payload() for pair_id in self.retained_pair_ids],
            "detailed_rejected_pair_ids": list(self.detailed_rejected_pair_ids),
            "reason_counts": self.reason_counts,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DualGraspPairPlanningResult:
    assembly: str
    base_part_id: str
    selected_order: tuple[str, ...]
    config: DualGraspPairConfig
    holder_feasibility: HolderStateFeasibilityResult = field(
        repr=False,
        compare=False,
    )
    inserter_libraries: tuple[InserterGraspLibrary, ...] = field(
        repr=False,
        compare=False,
    )
    steps: tuple[DualGraspPairStepResult, ...]

    @property
    def inserter_libraries_by_step(self) -> dict[str, InserterGraspLibrary]:
        return {library.step_id: library for library in self.inserter_libraries}


@dataclass(frozen=True)
class _CollisionQuery:
    collides: bool
    obstacle_names: tuple[str, ...]
    minimum_distance_m: float | None


@dataclass(frozen=True)
class _CandidateGeometry:
    final_meshes: tuple[object, ...]
    swept_meshes: tuple[object, ...]
    swept_bounds: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class _TransitionCandidateGeometry:
    geometry: _CandidateGeometry
    insertion_translation_m: tuple[float, float, float]
    retreat_translation_m: tuple[float, float, float]
    status: str
    reason: str
    minimum_clearance_m: float | None
    details: dict[str, object] = field(default_factory=dict)


def _triangle_mesh_to_trimesh(mesh: TriangleMesh):
    if trimesh is None:
        raise RuntimeError("trimesh is required for dual-grasp pair planning.")
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices_obj, dtype=float),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def _primitive_to_trimesh(
    primitive: BoxCollisionPrimitive | MeshCollisionPrimitive,
):
    if trimesh is None:
        raise RuntimeError("trimesh is required for dual-grasp pair planning.")
    if isinstance(primitive, BoxCollisionPrimitive):
        return trimesh.creation.box(
            extents=2.0 * np.asarray(primitive.half_extents, dtype=float),
            transform=primitive.transform_matrix_obj(),
        )
    return trimesh.Trimesh(
        vertices=np.asarray(primitive.vertices_obj, dtype=float),
        faces=np.asarray(primitive.faces, dtype=np.int64),
        process=False,
    )


def _manager_for_meshes(
    named_meshes: Iterable[tuple[str, object]],
):
    if trimesh is None:
        raise RuntimeError("trimesh is required for dual-grasp pair planning.")
    manager = trimesh.collision.CollisionManager()
    count = 0
    for name, mesh in named_meshes:
        manager.add_object(name, mesh)
        count += 1
    return manager if count else None


def _query_manager(manager, query_meshes: tuple[object, ...]) -> _CollisionQuery:
    if manager is None:
        return _CollisionQuery(False, (), None)
    collides = False
    names: set[str] = set()
    minimum_distance = math.inf
    for query_mesh in query_meshes:
        hit, hit_names = manager.in_collision_single(query_mesh, return_names=True)
        if hit:
            collides = True
            names.update(str(name) for name in hit_names)
        minimum_distance = min(
            minimum_distance,
            float(manager.min_distance_single(query_mesh)),
        )
    return _CollisionQuery(
        collides=collides,
        obstacle_names=tuple(sorted(names)),
        minimum_distance_m=(None if math.isinf(minimum_distance) else minimum_distance),
    )


def _candidate_meshes_assembly(
    candidate: SavedGraspCandidate,
    *,
    source_pose_assembly: ObjectWorldPose,
    planning: PlanningConfig,
    model_cache: dict[tuple[float, float], object],
) -> tuple[object, ...]:
    offset_key = (
        float(candidate.contact_patch_lateral_offset_m),
        float(candidate.contact_patch_approach_offset_m),
    )
    model = model_cache.get(offset_key)
    if model is None:
        model = make_gripper_collision_model(
            planning.gripper_collision_model,
            contact_gap_m=planning.detailed_finger_contact_gap_m,
            contact_patch_lateral_offset_m=offset_key[0],
            contact_patch_approach_offset_m=offset_key[1],
        )
        model_cache[offset_key] = model
    candidate_obj = candidate.to_object_frame_candidate()
    primitives = model.primitives_for_grasp(
        grasp_rotmat=quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj),
        contact_point_a=np.asarray(
            candidate_obj.contact_point_a_obj,
            dtype=float,
        ),
        contact_point_b=np.asarray(
            candidate_obj.contact_point_b_obj,
            dtype=float,
        ),
        grasp_center=np.asarray(candidate_obj.grasp_position_obj, dtype=float),
    )
    return tuple(
        _primitive_to_trimesh(transform_primitive_to_world(primitive, source_pose_assembly)) for primitive in primitives
    )


def _translated_meshes(
    meshes: tuple[object, ...],
    translation_m: tuple[float, float, float] | np.ndarray,
) -> tuple[object, ...]:
    translation = np.asarray(translation_m, dtype=float)
    translated = []
    for source in meshes:
        mesh = source.copy()
        mesh.apply_translation(translation)
        translated.append(mesh)
    return tuple(translated)


def _swept_meshes(
    meshes: tuple[object, ...],
    translation_m: tuple[float, float, float] | np.ndarray,
) -> tuple[object, ...]:
    swept = []
    for mesh in meshes:
        triangle_mesh = TriangleMesh(
            vertices_obj=np.asarray(mesh.vertices, dtype=float),
            faces=np.asarray(mesh.faces, dtype=np.int64),
        )
        swept.append(_triangle_mesh_to_trimesh(linear_sweep_triangle_mesh(triangle_mesh, translation_m)))
    return tuple(swept)


def _bounds(meshes: tuple[object, ...]) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.vstack([np.asarray(mesh.vertices, dtype=float) for mesh in meshes])
    return vertices.min(axis=0), vertices.max(axis=0)


def _aabb_distance_lower_bound(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> float:
    first_min, first_max = first
    second_min, second_max = second
    separation = np.maximum(
        np.maximum(second_min - first_max, first_min - second_max),
        0.0,
    )
    return float(np.linalg.norm(separation))


def _minimum_table_clearance(
    meshes: tuple[object, ...],
    translations: tuple[tuple[float, float, float], ...],
    *,
    table_z_m: float,
) -> float:
    return min(
        float(
            np.min(np.asarray(mesh.vertices, dtype=float)[:, 2])
            + min(float(translation[2]) for translation in translations)
            - table_z_m
        )
        for mesh in meshes
    )


def _minimum(*values: float | None) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(finite) if finite else None


def _inserter_candidate(
    candidate: SavedGraspCandidate,
    *,
    incoming_part_id: str,
) -> SavedGraspCandidate:
    source_id = candidate.grasp_id
    suffix = source_id[1:] if source_id.startswith("g") else source_id
    return replace(
        candidate,
        grasp_id=f"i{incoming_part_id}_{suffix}",
        metadata={
            **(candidate.metadata or {}),
            "candidate_role": "assembly_inserter",
            "incoming_part_id": incoming_part_id,
            "source_grasp_id": source_id,
        },
    )


def _retreat_translation(
    step: AssemblySequenceStep,
    *,
    distance_m: float,
) -> tuple[float, float, float]:
    final_to_pre = np.asarray(
        step.final_to_pre_insertion_translation_m,
        dtype=float,
    )
    norm = float(np.linalg.norm(final_to_pre))
    if norm < 1.0e-12 or distance_m <= 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(float(value) for value in final_to_pre / norm * distance_m)


def generate_inserter_grasp_library(
    *,
    sequence: AssemblySequence,
    step: AssemblySequenceStep,
    planning: PlanningConfig,
    config: DualGraspPairConfig,
) -> InserterGraspLibrary:
    """Generate insertion-filtered KUKA grasps and apply table/retreat checks."""

    if not step.holder_base_available:
        raise ValueError(f"Step '{step.step_id}' has no available holder base and cannot be paired.")
    if planning.gripper_collision_model != GRIPPER_COLLISION_MODEL_KUKA_Y:
        raise ValueError(
            f"Dual-grasp pair planning requires planning.gripper_collision_model='{GRIPPER_COLLISION_MODEL_KUKA_Y}'."
        )
    if not trimesh_fcl_backend_available():
        raise RuntimeError("trimesh with python-fcl is required for inserter grasp filtering.")

    part = sequence.parts_by_id[step.incoming_part_id]
    obstacle_paths = tuple(
        str(sequence.parts_by_id[part_id].resolved_mesh_path) for part_id in step.assembled_part_ids_before
    )
    stage1 = generate_stage1_result(
        geometry=GeometryConfig(
            target_mesh_path=str(part.resolved_mesh_path),
            mesh_scale=sequence.mesh_scale,
            assembly_glob=None,
            assembly_obstacle_paths=obstacle_paths,
            assembly_obstacle_sweep_vector_m=(step.pre_to_final_insertion_vector_m),
            assembly_obstacle_metadata={
                "assembly": sequence.assembly,
                "step_id": step.step_id,
                "incoming_part_id": step.incoming_part_id,
                "assembled_part_ids_before": list(step.assembled_part_ids_before),
                "motion_equivalence": (
                    "incoming gripper moves final_to_pre while static assembly is swept by pre_to_final"
                ),
            },
        ),
        planning=replace(planning, skip_stage1_collision_checks=False),
    )
    symmetry_metadata: dict[str, object] = {
        "symmetry_pickup_enabled": bool(planning.symmetry_pickup_enabled),
        "symmetry_pickup_load_status": "disabled",
    }
    force_full_assembly_recheck = False
    if planning.symmetry_pickup_enabled:
        records_by_part, asset_metadata = load_assembly_symmetry_records(
            sequence,
            symmetry_asset_path=(planning.symmetry_asset_path or None),
            max_nonidentity=int(planning.symmetry_max_transforms),
        )
        expanded_raw, expansion_metadata = expand_grasp_candidates_by_symmetry(
            stage1.raw_candidates,
            source_pose_assembly=stage1.target_pose_in_obj_world,
            symmetry_records=records_by_part[step.incoming_part_id],
        )
        symmetry_metadata = {
            "symmetry_pickup_enabled": True,
            "symmetry_pickup_load_status": asset_metadata["load_status"],
            "symmetry_pickup_source_path": asset_metadata["source_path"],
            "symmetry_pickup_asset_mesh_scale": asset_metadata.get("asset_mesh_scale"),
            "symmetry_pickup_translation_scale": asset_metadata.get("translation_scale"),
            **expansion_metadata,
        }
        source_raw_candidates = expanded_raw
        # Symmetry variants can move the gripper relative to the assembly.
        # Re-run the complete assembly sweep instead of inheriting the parent
        # candidate's canonical Stage-1 environment result.
        source_filtered_candidates = expanded_raw
        force_full_assembly_recheck = True
    else:
        source_raw_candidates = stage1.raw_candidates
        source_filtered_candidates = stage1.bundle.candidates
    renamed_raw = tuple(
        _inserter_candidate(
            candidate,
            incoming_part_id=step.incoming_part_id,
        )
        for candidate in source_raw_candidates
    )
    renamed_filtered = tuple(
        _inserter_candidate(
            candidate,
            incoming_part_id=step.incoming_part_id,
        )
        for candidate in source_filtered_candidates
    )
    filtered_by_id = {candidate.grasp_id: candidate for candidate in renamed_filtered}
    source_pose = stage1.target_pose_in_obj_world
    retreat = _retreat_translation(
        step,
        distance_m=config.retreat_distance_m,
    )
    insertion_translation = step.final_to_pre_insertion_translation_m
    insertion_distance = float(np.linalg.norm(np.asarray(insertion_translation, dtype=float)))
    retreat_extends_past_preinsertion = (
        float(np.linalg.norm(np.asarray(retreat, dtype=float))) > insertion_distance + 1.0e-12
    )
    motion_translations = (
        (0.0, 0.0, 0.0),
        insertion_translation,
        retreat,
    )
    part_meshes = {
        part_id: _triangle_mesh_to_trimesh(
            load_triangle_mesh(
                sequence.parts_by_id[part_id].resolved_mesh_path,
                scale=sequence.mesh_scale,
            )
        )
        for part_id in step.assembled_part_ids_before
    }
    assembled_manager = _manager_for_meshes(part_meshes.items())
    model_cache: dict[tuple[float, float], object] = {}
    statuses: list[InserterCandidateStatus] = []
    accepted: list[SavedGraspCandidate] = []
    for raw_candidate in renamed_raw:
        candidate = filtered_by_id.get(raw_candidate.grasp_id)
        if candidate is None:
            statuses.append(
                InserterCandidateStatus(
                    candidate=raw_candidate,
                    status="rejected",
                    reason=REASON_ASSEMBLY_INSERTION_SWEEP_COLLISION,
                    minimum_clearance_m=None,
                    details={
                        "filter": "existing_stage1_assembly_insertion_sweep",
                    },
                )
            )
            continue

        final_meshes = _candidate_meshes_assembly(
            candidate,
            source_pose_assembly=source_pose,
            planning=planning,
            model_cache=model_cache,
        )
        retreat_swept = _swept_meshes(final_meshes, retreat)
        if force_full_assembly_recheck:
            assembled_query_meshes = (
                *_swept_meshes(final_meshes, insertion_translation),
                *retreat_swept,
            )
        elif config.geometry_clearance_margin_m > 0.0:
            assembled_query_meshes = (
                *_swept_meshes(final_meshes, insertion_translation),
                *retreat_swept,
            )
        elif retreat_extends_past_preinsertion:
            assembled_query_meshes = retreat_swept
        else:
            # Existing Stage 1 already collision-checked the full insertion
            # sweep. A shorter retreat retraces a subset of the same geometry.
            assembled_query_meshes = ()
        assembled_query = _query_manager(
            assembled_manager,
            assembled_query_meshes,
        )
        table_clearance = _minimum_table_clearance(
            final_meshes,
            motion_translations,
            table_z_m=sequence.table_z_assembly_m,
        )
        minimum_clearance = _minimum(
            assembled_query.minimum_distance_m,
            table_clearance,
        )
        if assembled_query.collides:
            statuses.append(
                InserterCandidateStatus(
                    candidate=candidate,
                    status="rejected",
                    reason=(
                        REASON_ASSEMBLY_INSERTION_SWEEP_COLLISION
                        if force_full_assembly_recheck
                        else REASON_INSERTER_RETREAT_COLLISION
                    ),
                    minimum_clearance_m=minimum_clearance,
                    details={
                        "obstacle_part_ids": list(assembled_query.obstacle_names),
                        "retreat_translation_assembly_m": list(retreat),
                        "retreat_extends_past_preinsertion": (retreat_extends_past_preinsertion),
                    },
                )
            )
            continue
        if (
            assembled_query.minimum_distance_m is not None
            and assembled_query.minimum_distance_m < config.geometry_clearance_margin_m
        ):
            statuses.append(
                InserterCandidateStatus(
                    candidate=candidate,
                    status="rejected",
                    reason=REASON_INSERTER_CLEARANCE_MARGIN_FAILED,
                    minimum_clearance_m=minimum_clearance,
                    details={
                        "obstacle_type": "assembled_parts",
                        "required_clearance_m": (config.geometry_clearance_margin_m),
                    },
                )
            )
            continue
        if table_clearance < -1.0e-9:
            statuses.append(
                InserterCandidateStatus(
                    candidate=candidate,
                    status="rejected",
                    reason=REASON_INSERTER_TABLE_COLLISION,
                    minimum_clearance_m=minimum_clearance,
                    details={"obstacle_type": "table"},
                )
            )
            continue
        if table_clearance < config.table_clearance_margin_m:
            statuses.append(
                InserterCandidateStatus(
                    candidate=candidate,
                    status="rejected",
                    reason=REASON_INSERTER_CLEARANCE_MARGIN_FAILED,
                    minimum_clearance_m=minimum_clearance,
                    details={
                        "obstacle_type": "table",
                        "required_clearance_m": (config.table_clearance_margin_m),
                    },
                )
            )
            continue
        statuses.append(
            InserterCandidateStatus(
                candidate=candidate,
                status="accepted",
                reason=REASON_ACCEPTED,
                minimum_clearance_m=minimum_clearance,
                details={
                    "insertion_translation_assembly_m": list(insertion_translation),
                    "retreat_translation_assembly_m": list(retreat),
                },
            )
        )
        accepted.append(candidate)

    accepted_tuple = tuple(sorted(accepted, key=_candidate_sort_key))
    metadata = {
        **stage1.bundle.metadata,
        **symmetry_metadata,
        "artifact_kind": "inserter_candidate_library",
        "planning_stage": "dual_robot_stage_3_unary",
        "generated_by": "scripts/build_dual_grasp_pairs.py",
        "assembly": sequence.assembly,
        "step_id": step.step_id,
        "incoming_part_id": step.incoming_part_id,
        "assembled_part_ids_before": list(step.assembled_part_ids_before),
        "raw_candidate_count": len(renamed_raw),
        "assembly_insertion_feasible_count": len(renamed_filtered),
        "table_retreat_feasible_count": len(accepted_tuple),
        "candidate_ids_renamed": True,
        "table_filter_applied": True,
        "retreat_filter_applied": True,
        "pair_filter_applied": False,
        "insertion_translation_assembly_m": list(insertion_translation),
        "retreat_translation_assembly_m": list(retreat),
    }
    bundle = replace(
        stage1.bundle,
        candidates=accepted_tuple,
        metadata=metadata,
    )
    return InserterGraspLibrary(
        step_id=step.step_id,
        step_index=step.step_index,
        incoming_part_id=step.incoming_part_id,
        bundle=bundle,
        source_frame_pose_assembly=source_pose,
        candidate_statuses=tuple(statuses),
        raw_candidate_count=len(renamed_raw),
        assembly_insertion_feasible_count=len(renamed_filtered),
        collision_backend_name=stage1.collision_backend_name,
        source_stage1_cache_key=(
            None
            if stage1.bundle.metadata.get("stage1_cache_key") is None
            else str(stage1.bundle.metadata["stage1_cache_key"])
        ),
        retreat_translation_assembly_m=retreat,
    )


def generate_inserter_grasp_libraries(
    *,
    sequence: AssemblySequence,
    planning: PlanningConfig,
    config: DualGraspPairConfig,
) -> tuple[InserterGraspLibrary, ...]:
    return tuple(
        generate_inserter_grasp_library(
            sequence=sequence,
            step=step,
            planning=planning,
            config=config,
        )
        for step in sequence.steps
        if step.holder_base_available
    )


def _axis_bins(axis: np.ndarray, *, bin_rad: float) -> tuple[int, int]:
    normalized = np.asarray(axis, dtype=float)
    normalized /= max(float(np.linalg.norm(normalized)), 1.0e-12)
    azimuth = math.atan2(float(normalized[1]), float(normalized[0]))
    elevation = math.asin(float(np.clip(normalized[2], -1.0, 1.0)))
    return (
        int(round(azimuth / bin_rad)),
        int(round(elevation / bin_rad)),
    )


def _candidate_cluster_key(
    candidate: SavedGraspCandidate,
    *,
    source_pose_assembly: ObjectWorldPose,
    config: DualGraspPairConfig,
) -> tuple[int, ...]:
    rotation_assembly = source_pose_assembly.rotation_world_from_object @ quat_to_rotmat_xyzw(
        candidate.grasp_orientation_xyzw_obj
    )
    contact_center_source = 0.5 * (
        np.asarray(candidate.contact_point_a_obj, dtype=float) + np.asarray(candidate.contact_point_b_obj, dtype=float)
    )
    contact_center_assembly = (
        source_pose_assembly.rotation_world_from_object @ contact_center_source + source_pose_assembly.translation_world
    )
    position_bins = tuple(int(round(float(value) / config.contact_position_bin_m)) for value in contact_center_assembly)
    axis_bin_rad = math.radians(config.axis_bin_deg)
    jaw_bins = _axis_bins(rotation_assembly[:, 1], bin_rad=axis_bin_rad)
    approach_bins = _axis_bins(
        rotation_assembly[:, 2],
        bin_rad=axis_bin_rad,
    )
    return (*position_bins, *jaw_bins, *approach_bins)


def _diverse_shortlist(
    candidates: Iterable[SavedGraspCandidate],
    *,
    source_pose_assembly: ObjectWorldPose,
    config: DualGraspPairConfig,
    limit: int,
) -> tuple[SavedGraspCandidate, ...]:
    cluster_counts: Counter[tuple[int, ...]] = Counter()
    selected: list[SavedGraspCandidate] = []
    for candidate in sorted(candidates, key=_candidate_sort_key):
        cluster = _candidate_cluster_key(
            candidate,
            source_pose_assembly=source_pose_assembly,
            config=config,
        )
        if cluster_counts[cluster] >= config.max_candidates_per_cluster:
            continue
        cluster_counts[cluster] += 1
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _candidate_references(
    statuses: Iterable[tuple[SavedGraspCandidate, str, str]],
    *,
    shortlisted_ids: set[str],
) -> tuple[UnaryCandidateReference, ...]:
    sorted_statuses = sorted(
        statuses,
        key=lambda item: _candidate_sort_key(item[0]),
    )
    return tuple(
        UnaryCandidateReference(
            grasp_id=candidate.grasp_id,
            status=status,
            reason=reason,
            score=candidate.score,
            source_rank=rank,
            shortlisted=candidate.grasp_id in shortlisted_ids,
        )
        for rank, (candidate, status, reason) in enumerate(
            sorted_statuses,
            start=1,
        )
    )


def _candidate_geometry(
    candidate: SavedGraspCandidate,
    *,
    source_pose_assembly: ObjectWorldPose,
    planning: PlanningConfig,
    translations: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    model_cache: dict[tuple[float, float], object],
) -> _CandidateGeometry:
    final_meshes = _candidate_meshes_assembly(
        candidate,
        source_pose_assembly=source_pose_assembly,
        planning=planning,
        model_cache=model_cache,
    )
    swept_meshes = tuple(mesh for translation in translations for mesh in _swept_meshes(final_meshes, translation))
    return _CandidateGeometry(
        final_meshes=final_meshes,
        swept_meshes=swept_meshes,
        swept_bounds=_bounds(swept_meshes),
    )


def _transition_source_pose(
    transition: TransitionSymmetryCandidate,
) -> ObjectWorldPose:
    matrix = transition.final_source_matrix
    return ObjectWorldPose(
        position_world=tuple(float(value) for value in matrix[:3, 3]),
        orientation_xyzw_world=rotmat_to_quat_xyzw(matrix[:3, :3]),
    )


def _transition_motion_translations(
    transition: TransitionSymmetryCandidate,
    *,
    retreat_distance_m: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    insertion = transition.preinsertion_source_matrix[:3, 3] - transition.final_source_matrix[:3, 3]
    norm = float(np.linalg.norm(insertion))
    if norm < 1.0e-12 or float(retreat_distance_m) <= 0.0:
        retreat = np.zeros(3, dtype=float)
    else:
        retreat = insertion / norm * float(retreat_distance_m)
    return (
        tuple(float(value) for value in insertion),
        tuple(float(value) for value in retreat),
    )


def _transition_candidate_geometry(
    *,
    candidate: SavedGraspCandidate,
    transition: TransitionSymmetryCandidate,
    planning: PlanningConfig,
    config: DualGraspPairConfig,
    assembled_manager: object,
    table_z_m: float,
    model_cache: dict[tuple[float, float], object],
) -> _TransitionCandidateGeometry:
    insertion, retreat = _transition_motion_translations(
        transition,
        retreat_distance_m=config.retreat_distance_m,
    )
    geometry = _candidate_geometry(
        candidate,
        source_pose_assembly=_transition_source_pose(transition),
        planning=planning,
        translations=(insertion, retreat),
        model_cache=model_cache,
    )
    assembled_query = _query_manager(
        assembled_manager,
        geometry.swept_meshes,
    )
    table_clearance = _minimum_table_clearance(
        geometry.final_meshes,
        ((0.0, 0.0, 0.0), insertion, retreat),
        table_z_m=table_z_m,
    )
    minimum_clearance = _minimum(
        assembled_query.minimum_distance_m,
        table_clearance,
    )
    details: dict[str, object] = {
        "assembled_prefix_obstacle_part_ids": list(assembled_query.obstacle_names),
        "assembled_prefix_minimum_clearance_m": (assembled_query.minimum_distance_m),
        "table_minimum_clearance_m": table_clearance,
        "insertion_translation_assembly_m": list(insertion),
        "retreat_translation_assembly_m": list(retreat),
    }
    if assembled_query.collides:
        return _TransitionCandidateGeometry(
            geometry=geometry,
            insertion_translation_m=insertion,
            retreat_translation_m=retreat,
            status="rejected",
            reason=REASON_ASSEMBLY_INSERTION_SWEEP_COLLISION,
            minimum_clearance_m=minimum_clearance,
            details=details,
        )
    if (
        assembled_query.minimum_distance_m is not None
        and assembled_query.minimum_distance_m < config.geometry_clearance_margin_m
    ):
        return _TransitionCandidateGeometry(
            geometry=geometry,
            insertion_translation_m=insertion,
            retreat_translation_m=retreat,
            status="rejected",
            reason=REASON_INSERTER_CLEARANCE_MARGIN_FAILED,
            minimum_clearance_m=minimum_clearance,
            details=details,
        )
    if table_clearance < -1.0e-9:
        return _TransitionCandidateGeometry(
            geometry=geometry,
            insertion_translation_m=insertion,
            retreat_translation_m=retreat,
            status="rejected",
            reason=REASON_INSERTER_TABLE_COLLISION,
            minimum_clearance_m=minimum_clearance,
            details=details,
        )
    if table_clearance < config.table_clearance_margin_m:
        return _TransitionCandidateGeometry(
            geometry=geometry,
            insertion_translation_m=insertion,
            retreat_translation_m=retreat,
            status="rejected",
            reason=REASON_INSERTER_CLEARANCE_MARGIN_FAILED,
            minimum_clearance_m=minimum_clearance,
            details=details,
        )
    return _TransitionCandidateGeometry(
        geometry=geometry,
        insertion_translation_m=insertion,
        retreat_translation_m=retreat,
        status="accepted",
        reason=REASON_ACCEPTED,
        minimum_clearance_m=minimum_clearance,
        details=details,
    )


def _first_pair_collision(
    *,
    holder_meshes: tuple[object, ...],
    inserter_final_meshes: tuple[object, ...],
    insertion_translation_m: tuple[float, float, float],
    retreat_translation_m: tuple[float, float, float],
    samples: int,
    clearance_margin_m: float,
) -> dict[str, object]:
    holder_manager = _manager_for_meshes((f"holder_{index}", mesh) for index, mesh in enumerate(holder_meshes))
    phases = (
        (
            "insertion",
            lambda progress: np.asarray(insertion_translation_m, dtype=float) * (1.0 - progress),
        ),
        (
            "retreat",
            lambda progress: np.asarray(retreat_translation_m, dtype=float) * progress,
        ),
    )
    for phase, translation_for_progress in phases:
        for progress in np.linspace(0.0, 1.0, int(samples)):
            translation = translation_for_progress(float(progress))
            query = _query_manager(
                holder_manager,
                _translated_meshes(inserter_final_meshes, translation),
            )
            margin_failed = query.minimum_distance_m is not None and query.minimum_distance_m < clearance_margin_m
            if query.collides or margin_failed:
                return {
                    "first_failing_phase": phase,
                    "first_failing_progress": float(progress),
                    "first_failing_translation_assembly_m": [float(value) for value in translation],
                    "colliding_holder_primitives": list(query.obstacle_names),
                }
    return {
        "first_failing_phase": None,
        "first_failing_progress": None,
        "first_failing_translation_assembly_m": None,
        "colliding_holder_primitives": [],
    }


def _clearance_score(
    minimum_clearance_m: float | None,
    *,
    config: DualGraspPairConfig,
) -> float:
    if minimum_clearance_m is None:
        return 0.0
    usable = max(
        0.0,
        float(minimum_clearance_m) - config.geometry_clearance_margin_m,
    )
    return min(1.0, usable / config.clearance_score_saturation_m)


def _pair_score(
    holder: SavedGraspCandidate,
    inserter: SavedGraspCandidate,
    *,
    clearance_score: float,
    config: DualGraspPairConfig,
) -> float:
    total_weight = config.holder_score_weight + config.inserter_score_weight + config.clearance_score_weight
    return float(
        (
            config.holder_score_weight * _score(holder)
            + config.inserter_score_weight * _score(inserter)
            + config.clearance_score_weight * clearance_score
        )
        / total_weight
    )


def _pair_id(
    step: AssemblySequenceStep,
    holder: SavedGraspCandidate,
    inserter: SavedGraspCandidate,
) -> str:
    return f"p{step.step_index:03d}_{holder.grasp_id}_{inserter.grasp_id}"


def _evaluate_pair(
    *,
    step: AssemblySequenceStep,
    holder: SavedGraspCandidate,
    inserter: SavedGraspCandidate,
    holder_geometry: _CandidateGeometry,
    inserter_geometry: _CandidateGeometry,
    inserter_manager_cache: dict[str, object],
    config: DualGraspPairConfig,
    inserter_manager_cache_key: str | None = None,
) -> DualGraspPairEvaluation:
    lower_bound = _aabb_distance_lower_bound(
        holder_geometry.swept_bounds,
        inserter_geometry.swept_bounds,
    )
    pair_id = _pair_id(step, holder, inserter)
    if lower_bound > config.geometry_clearance_margin_m:
        clearance_component = _clearance_score(
            lower_bound,
            config=config,
        )
        return DualGraspPairEvaluation(
            pair_id=pair_id,
            holder_grasp_id=holder.grasp_id,
            inserter_grasp_id=inserter.grasp_id,
            status="accepted",
            reason=REASON_ACCEPTED,
            score=_pair_score(
                holder,
                inserter,
                clearance_score=clearance_component,
                config=config,
            ),
            holder_score=_score(holder),
            inserter_score=_score(inserter),
            clearance_score=clearance_component,
            minimum_clearance_m=lower_bound,
            collision_check="aabb_separation_proof",
            details={
                "minimum_clearance_is_lower_bound": True,
                "aabb_distance_lower_bound_m": lower_bound,
            },
        )

    manager_key = inserter.grasp_id if inserter_manager_cache_key is None else inserter_manager_cache_key
    inserter_manager = inserter_manager_cache.get(manager_key)
    if inserter_manager is None:
        inserter_manager = _manager_for_meshes(
            (f"inserter_sweep_{index}", mesh) for index, mesh in enumerate(inserter_geometry.swept_meshes)
        )
        inserter_manager_cache[manager_key] = inserter_manager
    exact = _query_manager(inserter_manager, holder_geometry.final_meshes)
    clearance_component = _clearance_score(
        exact.minimum_distance_m,
        config=config,
    )
    details = {
        "minimum_clearance_is_lower_bound": False,
        "aabb_distance_lower_bound_m": lower_bound,
        "colliding_inserter_sweep_primitives": list(exact.obstacle_names),
    }
    if exact.collides or (
        exact.minimum_distance_m is not None and exact.minimum_distance_m < config.geometry_clearance_margin_m
    ):
        details["sampled_collision_diagnostic"] = False
        reason = REASON_END_EFFECTOR_SWEEP_COLLISION if exact.collides else REASON_PAIR_CLEARANCE_MARGIN_FAILED
        return DualGraspPairEvaluation(
            pair_id=pair_id,
            holder_grasp_id=holder.grasp_id,
            inserter_grasp_id=inserter.grasp_id,
            status="rejected",
            reason=reason,
            score=_pair_score(
                holder,
                inserter,
                clearance_score=clearance_component,
                config=config,
            ),
            holder_score=_score(holder),
            inserter_score=_score(inserter),
            clearance_score=clearance_component,
            minimum_clearance_m=exact.minimum_distance_m,
            collision_check="exact_fcl",
            details=details,
        )
    return DualGraspPairEvaluation(
        pair_id=pair_id,
        holder_grasp_id=holder.grasp_id,
        inserter_grasp_id=inserter.grasp_id,
        status="accepted",
        reason=REASON_ACCEPTED,
        score=_pair_score(
            holder,
            inserter,
            clearance_score=clearance_component,
            config=config,
        ),
        holder_score=_score(holder),
        inserter_score=_score(inserter),
        clearance_score=clearance_component,
        minimum_clearance_m=exact.minimum_distance_m,
        collision_check="exact_fcl",
        details=details,
    )


def _validate_retained_pair_transitions(
    evaluations: tuple[DualGraspPairEvaluation, ...],
    *,
    retained_pair_ids: tuple[str, ...],
    transition_candidates: tuple[TransitionSymmetryCandidate, ...],
    sequence: AssemblySequence,
    step: AssemblySequenceStep,
    holders_by_id: dict[str, SavedGraspCandidate],
    inserters_by_id: dict[str, SavedGraspCandidate],
    holder_geometries: dict[str, _CandidateGeometry],
    planning: PlanningConfig,
    config: DualGraspPairConfig,
    model_cache: dict[tuple[float, float], object],
) -> tuple[DualGraspPairEvaluation, ...]:
    """Attach transition compatibility, validating variants for retained pairs."""

    retained = set(retained_pair_ids)
    identity_transitions = tuple(transition for transition in transition_candidates if transition.is_identity)
    if not identity_transitions:
        raise RuntimeError(f"Step '{step.step_id}' has no identity transition candidate.")
    needs_transformed_checks = any(not transition.is_identity for transition in transition_candidates)
    assembled_manager = None
    if needs_transformed_checks:
        assembled_manager = _manager_for_meshes(
            (
                part_id,
                _triangle_mesh_to_trimesh(
                    load_triangle_mesh(
                        sequence.parts_by_id[part_id].resolved_mesh_path,
                        scale=sequence.mesh_scale,
                    )
                ),
            )
            for part_id in step.assembled_part_ids_before
        )

    transition_geometry_cache: dict[
        tuple[str, str],
        _TransitionCandidateGeometry,
    ] = {}
    inserter_manager_cache: dict[str, object] = {}
    annotated: list[DualGraspPairEvaluation] = []
    for evaluation in evaluations:
        if evaluation.status != "accepted":
            annotated.append(evaluation)
            continue
        if evaluation.pair_id not in retained:
            identity_ids = [transition.transition_id for transition in identity_transitions]
            annotated.append(
                replace(
                    evaluation,
                    details={
                        **evaluation.details,
                        "compatible_transition_ids": identity_ids,
                        "transition_validation": {
                            transition_id: {
                                "status": "accepted",
                                "reason": REASON_ACCEPTED,
                                "validation_source": ("canonical_unary_and_pair_collision_checks"),
                                "gripper_sweep_checked": True,
                                "assembled_prefix_checked": True,
                                "table_checked": True,
                                "holder_gripper_checked": True,
                                "robot_path_checked": False,
                                "collision_check": (evaluation.collision_check),
                                "minimum_clearance_m": (evaluation.minimum_clearance_m),
                            }
                            for transition_id in identity_ids
                        },
                        "transition_validation_policy": (
                            "non-retained pairs remain identity-only; "
                            "nonidentity corridors are validated only for "
                            "the bounded retained fallback set"
                        ),
                    },
                )
            )
            continue
        holder = holders_by_id[evaluation.holder_grasp_id]
        inserter = inserters_by_id[evaluation.inserter_grasp_id]
        compatible_ids: list[str] = []
        validations: dict[str, object] = {}
        for transition in transition_candidates:
            transition_id = transition.transition_id
            if transition.is_identity:
                compatible_ids.append(transition_id)
                validations[transition_id] = {
                    "status": "accepted",
                    "reason": REASON_ACCEPTED,
                    "validation_source": ("canonical_unary_and_pair_collision_checks"),
                    "gripper_sweep_checked": True,
                    "assembled_prefix_checked": True,
                    "table_checked": True,
                    "holder_gripper_checked": True,
                    "robot_path_checked": False,
                    "collision_check": evaluation.collision_check,
                    "minimum_clearance_m": evaluation.minimum_clearance_m,
                }
                continue

            geometry_key = (inserter.grasp_id, transition_id)
            transition_geometry = transition_geometry_cache.get(geometry_key)
            if transition_geometry is None:
                transition_geometry = _transition_candidate_geometry(
                    candidate=inserter,
                    transition=transition,
                    planning=planning,
                    config=config,
                    assembled_manager=assembled_manager,
                    table_z_m=sequence.table_z_assembly_m,
                    model_cache=model_cache,
                )
                transition_geometry_cache[geometry_key] = transition_geometry
            if transition_geometry.status != "accepted":
                validations[transition_id] = {
                    "status": "rejected",
                    "reason": transition_geometry.reason,
                    "validation_source": "transformed_corridor_fcl",
                    "gripper_sweep_checked": True,
                    "assembled_prefix_checked": True,
                    "table_checked": True,
                    "holder_gripper_checked": False,
                    "robot_path_checked": False,
                    "minimum_clearance_m": (transition_geometry.minimum_clearance_m),
                    "details": transition_geometry.details,
                }
                continue

            pair_result = _evaluate_pair(
                step=step,
                holder=holder,
                inserter=inserter,
                holder_geometry=holder_geometries[holder.grasp_id],
                inserter_geometry=transition_geometry.geometry,
                inserter_manager_cache=inserter_manager_cache,
                config=config,
                inserter_manager_cache_key=(f"{inserter.grasp_id}::{transition_id}"),
            )
            status = pair_result.status
            if status == "accepted":
                compatible_ids.append(transition_id)
            validations[transition_id] = {
                "status": status,
                "reason": pair_result.reason,
                "validation_source": "transformed_corridor_fcl",
                "gripper_sweep_checked": True,
                "assembled_prefix_checked": True,
                "table_checked": True,
                "holder_gripper_checked": True,
                "robot_path_checked": False,
                "collision_check": pair_result.collision_check,
                "minimum_clearance_m": _minimum(
                    transition_geometry.minimum_clearance_m,
                    pair_result.minimum_clearance_m,
                ),
                "details": {
                    **transition_geometry.details,
                    "holder_pair": pair_result.details,
                },
            }

        annotated.append(
            replace(
                evaluation,
                details={
                    **evaluation.details,
                    "compatible_transition_ids": compatible_ids,
                    "transition_validation": validations,
                    "transition_validation_policy": (
                        "identity inherits canonical checks; every "
                        "nonidentity corridor is rechecked against the "
                        "assembled prefix, table, and selected holder gripper"
                    ),
                },
            )
        )
    return tuple(annotated)


def _retain_diverse_pairs(
    evaluations: Iterable[DualGraspPairEvaluation],
    *,
    config: DualGraspPairConfig,
) -> tuple[str, ...]:
    holder_counts: Counter[str] = Counter()
    inserter_counts: Counter[str] = Counter()
    retained: list[str] = []
    compatible = sorted(
        (evaluation for evaluation in evaluations if evaluation.status == "accepted"),
        key=lambda evaluation: (-evaluation.score, evaluation.pair_id),
    )
    for evaluation in compatible:
        if (
            holder_counts[evaluation.holder_grasp_id] >= config.max_pairs_per_holder
            or inserter_counts[evaluation.inserter_grasp_id] >= config.max_pairs_per_inserter
        ):
            continue
        retained.append(evaluation.pair_id)
        holder_counts[evaluation.holder_grasp_id] += 1
        inserter_counts[evaluation.inserter_grasp_id] += 1
        if len(retained) >= config.max_accepted_pairs:
            break
    return tuple(retained)


def _matrix_ids(
    references: tuple[UnaryCandidateReference, ...],
    *,
    shortlisted_ids: tuple[str, ...],
    rejection_limit: int,
) -> tuple[str, ...]:
    rejected = [reference.grasp_id for reference in references if reference.status != "accepted"][:rejection_limit]
    return (*shortlisted_ids, *rejected)


def plan_dual_grasp_pairs(
    *,
    sequence: AssemblySequence,
    holder_feasibility: HolderStateFeasibilityResult,
    inserter_libraries: tuple[InserterGraspLibrary, ...],
    planning: PlanningConfig,
    config: DualGraspPairConfig,
) -> DualGraspPairPlanningResult:
    """Plan bounded, ranked end-effector pairs for every holder-active step."""

    if planning.gripper_collision_model != GRIPPER_COLLISION_MODEL_KUKA_Y:
        raise ValueError("Dual-grasp pair planning requires the KUKA Y-gripper collision model.")
    if not trimesh_fcl_backend_available():
        raise RuntimeError("trimesh with python-fcl is required for dual-grasp pair planning.")
    if (
        holder_feasibility.assembly != sequence.assembly
        or holder_feasibility.base_part_id != sequence.base_part_id
        or holder_feasibility.selected_order != sequence.selected_order
    ):
        raise ValueError("Holder feasibility result does not match the assembly sequence.")
    libraries_by_step = {library.step_id: library for library in inserter_libraries}
    expected_step_ids = {step.step_id for step in sequence.steps if step.holder_base_available}
    if set(libraries_by_step) != expected_step_ids:
        raise ValueError(
            "Inserter libraries must resolve exactly once for every "
            f"holder-active step: expected {sorted(expected_step_ids)}, "
            f"got {sorted(libraries_by_step)}."
        )

    holder_by_id = {candidate.grasp_id: candidate for candidate in holder_feasibility.candidates}
    holder_state_by_step = {state.step_id: state for state in holder_feasibility.states}
    model_cache: dict[tuple[float, float], object] = {}
    step_results: list[DualGraspPairStepResult] = []
    for step in sequence.steps:
        if not step.holder_base_available:
            continue
        holder_state = holder_state_by_step[step.step_id]
        library = libraries_by_step[step.step_id]
        if library.step_index != step.step_index or library.incoming_part_id != step.incoming_part_id:
            raise ValueError(f"Inserter library '{library.step_id}' does not match its sequence step.")
        transition_candidates, transition_symmetry_metadata = compile_step_transition_symmetries(
            sequence=sequence,
            step=step,
            incoming_source_pose_assembly=(library.source_frame_pose_assembly),
            symmetry_asset_path=(config.transition_symmetry_asset_path or None),
            geometry_tolerance_m=(config.transition_symmetry_geometry_tolerance_m),
            max_partial_assembly_transforms=(config.transition_symmetry_max_partial_assembly_transforms),
            max_incoming_transforms=(config.transition_symmetry_max_incoming_transforms),
            yaw_only=True,
            enabled=bool(config.transition_symmetry_enabled),
        )
        holder_result_by_id = {result.grasp_id: result for result in holder_state.candidate_results}
        holder_accepted = tuple(holder_by_id[grasp_id] for grasp_id in holder_state.accepted_grasp_ids)
        holder_shortlist = _diverse_shortlist(
            holder_accepted,
            source_pose_assembly=(holder_feasibility.source_frame_pose_assembly),
            config=config,
            limit=config.max_holder_candidates_per_step,
        )
        inserter_shortlist = _diverse_shortlist(
            library.accepted_candidates,
            source_pose_assembly=library.source_frame_pose_assembly,
            config=config,
            limit=config.max_inserter_candidates_per_step,
        )
        holder_shortlisted_ids = tuple(candidate.grasp_id for candidate in holder_shortlist)
        inserter_shortlisted_ids = tuple(candidate.grasp_id for candidate in inserter_shortlist)
        holder_references = _candidate_references(
            (
                (
                    candidate,
                    holder_result_by_id[candidate.grasp_id].status,
                    holder_result_by_id[candidate.grasp_id].reason,
                )
                for candidate in holder_feasibility.candidates
            ),
            shortlisted_ids=set(holder_shortlisted_ids),
        )
        inserter_references = _candidate_references(
            (
                (
                    status.candidate,
                    status.status,
                    status.reason,
                )
                for status in library.candidate_statuses
            ),
            shortlisted_ids=set(inserter_shortlisted_ids),
        )

        holder_geometries = {
            candidate.grasp_id: _candidate_geometry(
                candidate,
                source_pose_assembly=(holder_feasibility.source_frame_pose_assembly),
                planning=planning,
                translations=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                model_cache=model_cache,
            )
            for candidate in holder_shortlist
        }
        inserter_geometries = {
            candidate.grasp_id: _candidate_geometry(
                candidate,
                source_pose_assembly=library.source_frame_pose_assembly,
                planning=planning,
                translations=(
                    step.final_to_pre_insertion_translation_m,
                    library.retreat_translation_assembly_m,
                ),
                model_cache=model_cache,
            )
            for candidate in inserter_shortlist
        }
        combinations = sorted(
            (
                (
                    -(
                        config.holder_score_weight * _score(holder)
                        + config.inserter_score_weight * _score(inserter)
                        + config.clearance_score_weight
                    ),
                    holder.grasp_id,
                    inserter.grasp_id,
                    holder,
                    inserter,
                )
                for holder in holder_shortlist
                for inserter in inserter_shortlist
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        inserter_manager_cache: dict[str, object] = {}
        evaluations_list = [
            _evaluate_pair(
                step=step,
                holder=holder,
                inserter=inserter,
                holder_geometry=holder_geometries[holder.grasp_id],
                inserter_geometry=inserter_geometries[inserter.grasp_id],
                inserter_manager_cache=inserter_manager_cache,
                config=config,
            )
            for _, _, _, holder, inserter in combinations[: config.max_pair_checks]
        ]
        diagnostic_ids = {
            evaluation.pair_id
            for evaluation in sorted(
                (item for item in evaluations_list if item.status == "rejected"),
                key=lambda item: (-item.score, item.pair_id),
            )[: config.max_collision_diagnostics_per_step]
        }
        evaluations = tuple(
            (
                replace(
                    evaluation,
                    details={
                        **evaluation.details,
                        **_first_pair_collision(
                            holder_meshes=holder_geometries[evaluation.holder_grasp_id].final_meshes,
                            inserter_final_meshes=inserter_geometries[evaluation.inserter_grasp_id].final_meshes,
                            insertion_translation_m=(step.final_to_pre_insertion_translation_m),
                            retreat_translation_m=(library.retreat_translation_assembly_m),
                            samples=config.path_samples,
                            clearance_margin_m=(config.geometry_clearance_margin_m),
                        ),
                        "sampled_collision_diagnostic": True,
                    },
                )
                if evaluation.pair_id in diagnostic_ids
                else evaluation
            )
            for evaluation in evaluations_list
        )
        retained_pair_ids = _retain_diverse_pairs(
            evaluations,
            config=config,
        )
        evaluations = _validate_retained_pair_transitions(
            evaluations,
            retained_pair_ids=retained_pair_ids,
            transition_candidates=transition_candidates,
            sequence=sequence,
            step=step,
            holders_by_id=holder_by_id,
            inserters_by_id={candidate.grasp_id: candidate for candidate in inserter_shortlist},
            holder_geometries=holder_geometries,
            planning=planning,
            config=config,
            model_cache=model_cache,
        )
        retained_pair_id_set = set(retained_pair_ids)
        compatible_pair_transition_count = sum(
            len(
                evaluation.details.get(
                    "compatible_transition_ids",
                    (),
                )
            )
            for evaluation in evaluations
            if evaluation.pair_id in retained_pair_id_set
        )
        checked_pair_transition_count = sum(
            len(
                dict(
                    evaluation.details.get(
                        "transition_validation",
                        {},
                    )
                )
            )
            for evaluation in evaluations
            if evaluation.pair_id in retained_pair_id_set
        )
        identity_only_nonretained_pair_count = sum(
            evaluation.status == "accepted" and evaluation.pair_id not in retained_pair_id_set
            for evaluation in evaluations
        )
        transition_symmetry_metadata = {
            **transition_symmetry_metadata,
            "pair_conditioned_validation": (
                "retained grasp pairs rechecked against assembled prefix, table, and holder gripper"
            ),
            "checked_retained_pair_transition_count": (checked_pair_transition_count),
            "compatible_retained_pair_transition_count": (compatible_pair_transition_count),
            "identity_only_nonretained_pair_count": (identity_only_nonretained_pair_count),
        }
        rejected = sorted(
            (evaluation for evaluation in evaluations if evaluation.status == "rejected"),
            key=lambda evaluation: (-evaluation.score, evaluation.pair_id),
        )
        exact_count = sum(evaluation.collision_check == "exact_fcl" for evaluation in evaluations)
        compatible_count = sum(evaluation.status == "accepted" for evaluation in evaluations)
        step_results.append(
            DualGraspPairStepResult(
                step_id=step.step_id,
                step_index=step.step_index,
                incoming_part_id=step.incoming_part_id,
                assembled_part_ids_before=step.assembled_part_ids_before,
                final_to_pre_translation_assembly_m=(step.final_to_pre_insertion_translation_m),
                retreat_translation_assembly_m=(library.retreat_translation_assembly_m),
                holder_candidates=holder_references,
                inserter_candidates=inserter_references,
                shortlisted_holder_ids=holder_shortlisted_ids,
                shortlisted_inserter_ids=inserter_shortlisted_ids,
                matrix_holder_ids=_matrix_ids(
                    holder_references,
                    shortlisted_ids=holder_shortlisted_ids,
                    rejection_limit=(config.matrix_unary_rejections_per_side),
                ),
                matrix_inserter_ids=_matrix_ids(
                    inserter_references,
                    shortlisted_ids=inserter_shortlisted_ids,
                    rejection_limit=(config.matrix_unary_rejections_per_side),
                ),
                evaluations=evaluations,
                retained_pair_ids=retained_pair_ids,
                detailed_rejected_pair_ids=tuple(
                    evaluation.pair_id for evaluation in rejected[: config.max_rejected_pairs]
                ),
                transition_candidates=transition_candidates,
                transition_symmetry_metadata=transition_symmetry_metadata,
                metadata={
                    "holder_unary_accepted_count": len(holder_accepted),
                    "inserter_unary_accepted_count": len(library.accepted_candidates),
                    "holder_shortlist_count": len(holder_shortlist),
                    "inserter_shortlist_count": len(inserter_shortlist),
                    "possible_shortlisted_pair_count": (len(holder_shortlist) * len(inserter_shortlist)),
                    "checked_pair_count": len(evaluations),
                    "pair_check_limit_reached": (len(combinations) > len(evaluations)),
                    "broadphase_clear_count": (len(evaluations) - exact_count),
                    "exact_fcl_pair_check_count": exact_count,
                    "compatible_pair_count": compatible_count,
                    "rejected_pair_count": (len(evaluations) - compatible_count),
                    "retained_pair_count": len(retained_pair_ids),
                    "checked_retained_pair_transition_count": (checked_pair_transition_count),
                    "compatible_retained_pair_transition_count": (compatible_pair_transition_count),
                    "identity_only_nonretained_pair_count": (identity_only_nonretained_pair_count),
                    "retention_policy": {
                        "max_pairs_per_holder": (config.max_pairs_per_holder),
                        "max_pairs_per_inserter": (config.max_pairs_per_inserter),
                    },
                },
            )
        )

    return DualGraspPairPlanningResult(
        assembly=sequence.assembly,
        base_part_id=sequence.base_part_id,
        selected_order=sequence.selected_order,
        config=config,
        holder_feasibility=holder_feasibility,
        inserter_libraries=inserter_libraries,
        steps=tuple(step_results),
    )


def inserter_artifact_name(step: AssemblySequenceStep | InserterGraspLibrary) -> str:
    return f"inserter_candidates_{step.step_id}.json"


def pair_artifact_name(step: AssemblySequenceStep | DualGraspPairStepResult) -> str:
    return f"dual_grasp_pairs_{step.step_id}.json"


def pair_html_name(step: AssemblySequenceStep | DualGraspPairStepResult) -> str:
    return f"dual_grasp_pairs_{step.step_id}.html"


def write_inserter_grasp_library(
    library: InserterGraspLibrary,
    output_path: str | Path,
) -> None:
    save_grasp_bundle(library.bundle, output_path)


def write_dual_grasp_pair_step_json(
    step_result: DualGraspPairStepResult,
    output_path: str | Path,
    *,
    holder_source_artifact: str,
    inserter_source_artifact: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            step_result.to_payload(
                holder_source_artifact=holder_source_artifact,
                inserter_source_artifact=inserter_source_artifact,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def dual_grasp_pair_summary_payload(
    result: DualGraspPairPlanningResult,
) -> dict[str, object]:
    libraries = result.inserter_libraries_by_step
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "dual_grasp_pair_summary",
        "generated_by": "scripts/build_dual_grasp_pairs.py",
        "assembly": result.assembly,
        "base_part_id": result.base_part_id,
        "selected_order": list(result.selected_order),
        "configuration": result.config.to_payload(),
        "holder_source_artifact": "holder_state_feasibility.json",
        "steps": [
            {
                "step_id": step.step_id,
                "step_index": step.step_index,
                "incoming_part_id": step.incoming_part_id,
                "inserter_source": libraries[step.step_id].reference_payload(
                    source_artifact=inserter_artifact_name(libraries[step.step_id])
                ),
                "pair_artifact": pair_artifact_name(step),
                "pair_html": pair_html_name(step),
                "holder_unary_accepted_count": step.metadata["holder_unary_accepted_count"],
                "inserter_unary_accepted_count": step.metadata["inserter_unary_accepted_count"],
                "checked_pair_count": step.metadata["checked_pair_count"],
                "compatible_pair_count": step.metadata["compatible_pair_count"],
                "retained_pair_count": step.metadata["retained_pair_count"],
                "transition_symmetry_enabled": bool(step.transition_symmetry_metadata.get("enabled", False)),
                "transition_candidate_count": len(step.transition_candidates),
                "reason_counts": step.reason_counts,
                "selected_pair_id": (None if not step.retained_pair_ids else step.retained_pair_ids[0]),
            }
            for step in result.steps
        ],
        "scope_boundary": {
            "checked": ("end-effector geometry, table, assembled prefix, incoming part sweep, insertion and retreat"),
            "not_checked": ("robot IK, robot links, trajectories, forces, friction, simultaneous execution"),
        },
    }


def write_dual_grasp_pair_summary_json(
    result: DualGraspPairPlanningResult,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dual_grasp_pair_summary_payload(result), indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DualGraspPairConfig",
    "DualGraspPairEvaluation",
    "DualGraspPairPlanningResult",
    "DualGraspPairStepResult",
    "InserterCandidateStatus",
    "InserterGraspLibrary",
    "UnaryCandidateReference",
    "dual_grasp_pair_summary_payload",
    "generate_inserter_grasp_libraries",
    "generate_inserter_grasp_library",
    "inserter_artifact_name",
    "pair_artifact_name",
    "pair_html_name",
    "plan_dual_grasp_pairs",
    "write_dual_grasp_pair_step_json",
    "write_dual_grasp_pair_summary_json",
    "write_inserter_grasp_library",
]
