"""Per-state collision feasibility for a table-supported assembly holder."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from grasp_planning.grasping.collision import (
    BoxCollisionPrimitive,
    MeshCollisionPrimitive,
    gripper_collision_check_gaps,
    make_gripper_collision_model,
    trimesh,
    trimesh_fcl_backend_available,
)
from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspCandidate,
    linear_sweep_triangle_mesh,
    quat_to_rotmat_xyzw,
    transform_primitive_to_world,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh
from grasp_planning.grasping.mesh_io import load_triangle_mesh
from grasp_planning.grasping.world_constraints import ObjectWorldPose

from .assembly_sequence import AssemblySequence, AssemblySequenceStep
from .fabrica_pipeline import PlanningConfig, Stage1Result

SCHEMA_VERSION = 1
REASON_ACCEPTED = "accepted"
REASON_BASE_NOT_AVAILABLE = "base_not_available"
REASON_BASE_COLLISION = "base_collision"
REASON_TABLE_COLLISION = "table_collision"
REASON_ASSEMBLED_PART_COLLISION = "assembled_part_collision"
REASON_HOLDER_PREGRASP_COLLISION = "holder_pregrasp_collision"
REASON_HOLDER_APPROACH_SWEEP_COLLISION = "holder_approach_sweep_collision"
REASON_INCOMING_PART_SWEEP_COLLISION = "incoming_part_sweep_collision"
REASON_CLEARANCE_MARGIN_FAILED = "clearance_margin_failed"


@dataclass(frozen=True)
class HolderFeasibilityConfig:
    pregrasp_offset_m: float = 0.05
    table_clearance_margin_m: float = 0.002
    geometry_clearance_margin_m: float = 0.0
    incoming_path_samples: int = 21

    def __post_init__(self) -> None:
        if self.pregrasp_offset_m < 0.0:
            raise ValueError("holder_feasibility.pregrasp_offset_m must be >= 0.")
        if self.table_clearance_margin_m < 0.0:
            raise ValueError("holder_feasibility.table_clearance_margin_m must be >= 0.")
        if self.geometry_clearance_margin_m < 0.0:
            raise ValueError("holder_feasibility.geometry_clearance_margin_m must be >= 0.")
        if self.incoming_path_samples < 3:
            raise ValueError("holder_feasibility.incoming_path_samples must be >= 3.")


@dataclass(frozen=True)
class HolderCandidateFeasibility:
    grasp_id: str
    status: str
    reason: str
    minimum_clearance_m: float | None
    details: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "grasp_id": self.grasp_id,
            "status": self.status,
            "reason": self.reason,
            "minimum_clearance_m": self.minimum_clearance_m,
            "details": self.details,
        }


@dataclass(frozen=True)
class HolderStateFeasibility:
    step_id: str
    step_index: int
    incoming_part_id: str
    holder_base_available: bool
    assembled_part_ids_before: tuple[str, ...]
    static_obstacle_part_ids: tuple[str, ...]
    incoming_final_to_pre_translation_m: tuple[float, float, float]
    candidate_results: tuple[HolderCandidateFeasibility, ...]
    reason_counts: dict[str, int]

    @property
    def accepted_grasp_ids(self) -> tuple[str, ...]:
        return tuple(result.grasp_id for result in self.candidate_results if result.status == "accepted")

    def to_payload(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "step_index": self.step_index,
            "incoming_part_id": self.incoming_part_id,
            "holder_base_available": self.holder_base_available,
            "assembled_part_ids_before": list(self.assembled_part_ids_before),
            "obstacle_motion_specs": [
                {
                    "part_id": part_id,
                    "motion": "static",
                    "translation_start_m": [0.0, 0.0, 0.0],
                    "translation_end_m": [0.0, 0.0, 0.0],
                }
                for part_id in self.static_obstacle_part_ids
            ]
            + [
                {
                    "part_id": self.incoming_part_id,
                    "motion": "linear_insertion",
                    "translation_start_m": list(self.incoming_final_to_pre_translation_m),
                    "translation_end_m": [0.0, 0.0, 0.0],
                }
            ],
            "reason_counts": self.reason_counts,
            "accepted_grasp_ids": list(self.accepted_grasp_ids),
            "candidate_results": [result.to_payload() for result in self.candidate_results],
        }


@dataclass(frozen=True)
class HolderStateFeasibilityResult:
    assembly: str
    base_part_id: str
    base_part_source: str
    selected_order: tuple[str, ...]
    table_z_assembly_m: float
    source_frame_pose_assembly: ObjectWorldPose
    config: HolderFeasibilityConfig
    candidates: tuple[SavedGraspCandidate, ...]
    states: tuple[HolderStateFeasibility, ...]
    collision_backend_name: str
    source_holder_cache_key: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "holder_state_feasibility",
            "generated_by": "scripts/build_holder_state_feasibility.py",
            "assembly": self.assembly,
            "base_part_id": self.base_part_id,
            "base_part_source": self.base_part_source,
            "selected_order": list(self.selected_order),
            "table": {
                "z_assembly_m": self.table_z_assembly_m,
                "clearance_margin_m": self.config.table_clearance_margin_m,
            },
            "frame_contract": {
                "candidate_frame": "canonical base-part source frame",
                "source_frame_pose": "base source frame expressed in the Fabrica assembly asset frame",
                "collision_frame": "scaled Fabrica assembly asset frame",
            },
            "source_frame_pose_assembly": {
                "position": list(self.source_frame_pose_assembly.position_world),
                "orientation_xyzw": list(self.source_frame_pose_assembly.orientation_xyzw_world),
            },
            "configuration": {
                "pregrasp_offset_m": self.config.pregrasp_offset_m,
                "table_clearance_margin_m": self.config.table_clearance_margin_m,
                "geometry_clearance_margin_m": self.config.geometry_clearance_margin_m,
                "incoming_path_samples": self.config.incoming_path_samples,
            },
            "collision_backend": self.collision_backend_name,
            "source_holder_cache_key": self.source_holder_cache_key,
            "candidate_count": len(self.candidates),
            "candidates": {candidate.grasp_id: _candidate_payload(candidate) for candidate in self.candidates},
            "states": [state.to_payload() for state in self.states],
        }


@dataclass(frozen=True)
class _CollisionQuery:
    collides: bool
    obstacle_names: tuple[str, ...]
    minimum_distance_m: float | None


@dataclass(frozen=True)
class _PreparedHolderCandidate:
    candidate: SavedGraspCandidate
    gripper_meshes: tuple[object, ...]
    pregrasp_meshes: tuple[object, ...]
    approach_swept_meshes: tuple[object, ...]
    grasp_to_pregrasp_translation_assembly_m: tuple[float, float, float]
    static_minimum_clearance_m: float | None
    static_failure: HolderCandidateFeasibility | None


def _candidate_payload(candidate: SavedGraspCandidate) -> dict[str, object]:
    return {
        "grasp_id": candidate.grasp_id,
        "grasp_pose_obj": {
            "position": list(candidate.grasp_position_obj),
            "orientation_xyzw": list(candidate.grasp_orientation_xyzw_obj),
        },
        "contact_points_obj": [
            list(candidate.contact_point_a_obj),
            list(candidate.contact_point_b_obj),
        ],
        "contact_normals_obj": [
            list(candidate.contact_normal_a_obj),
            list(candidate.contact_normal_b_obj),
        ],
        "jaw_width": candidate.jaw_width,
        "roll_angle_rad": candidate.roll_angle_rad,
        "contact_patch_offset_local": [
            candidate.contact_patch_lateral_offset_m,
            candidate.contact_patch_approach_offset_m,
        ],
        "score": candidate.score,
        "score_components": candidate.score_components,
        "metadata": candidate.metadata or {},
    }


def _triangle_mesh_to_trimesh(mesh: TriangleMesh):
    if trimesh is None:
        raise RuntimeError("trimesh is required for holder state feasibility.")
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices_obj, dtype=float),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def _primitive_to_trimesh(primitive: BoxCollisionPrimitive | MeshCollisionPrimitive):
    if trimesh is None:
        raise RuntimeError("trimesh is required for holder state feasibility.")
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


def _manager_for_parts(part_meshes: dict[str, TriangleMesh], part_ids: tuple[str, ...]):
    if trimesh is None:
        raise RuntimeError("trimesh is required for holder state feasibility.")
    manager = trimesh.collision.CollisionManager()
    for part_id in part_ids:
        manager.add_object(part_id, _triangle_mesh_to_trimesh(part_meshes[part_id]))
    return manager if part_ids else None


def _query_manager(manager, query_meshes: tuple[object, ...]) -> _CollisionQuery:
    if manager is None:
        return _CollisionQuery(collides=False, obstacle_names=(), minimum_distance_m=None)
    collides = False
    names: set[str] = set()
    minimum_distance = math.inf
    for query_mesh in query_meshes:
        hit, hit_names = manager.in_collision_single(query_mesh, return_names=True)
        if hit:
            collides = True
            names.update(str(name) for name in hit_names)
        distance = float(manager.min_distance_single(query_mesh))
        minimum_distance = min(minimum_distance, distance)
    return _CollisionQuery(
        collides=collides,
        obstacle_names=tuple(sorted(names)),
        minimum_distance_m=None if math.isinf(minimum_distance) else minimum_distance,
    )


def _candidate_meshes_assembly(
    candidate: SavedGraspCandidate,
    *,
    source_pose_assembly: ObjectWorldPose,
    planning: PlanningConfig,
    model_cache: dict[tuple[float, float], tuple[object, ...]],
) -> tuple[object, ...]:
    opening_meshes = _candidate_opening_meshes_assembly(
        candidate,
        source_pose_assembly=source_pose_assembly,
        planning=planning,
        model_cache=model_cache,
    )
    combined: list[object] = []
    base_added = False
    for meshes in opening_meshes:
        for mesh_index, mesh in enumerate(meshes):
            if mesh_index == 0:
                if base_added:
                    continue
                base_added = True
            combined.append(mesh)
    return tuple(combined)


def _candidate_opening_meshes_assembly(
    candidate: SavedGraspCandidate,
    *,
    source_pose_assembly: ObjectWorldPose,
    planning: PlanningConfig,
    model_cache: dict[tuple[float, float], tuple[object, ...]],
) -> tuple[tuple[object, ...], ...]:
    """Return distinct contact and approach meshes without merging their fingers."""

    offset_key = (
        float(candidate.contact_patch_lateral_offset_m),
        float(candidate.contact_patch_approach_offset_m),
    )
    models = model_cache.get(offset_key)
    if models is None:
        models = tuple(
            make_gripper_collision_model(
                planning.gripper_collision_model,
                contact_gap_m=gap_m,
                contact_patch_lateral_offset_m=offset_key[0],
                contact_patch_approach_offset_m=offset_key[1],
            )
            for gap_m in gripper_collision_check_gaps(planning.detailed_finger_contact_gap_m)
        )
        model_cache[offset_key] = models
    candidate_obj = candidate.to_object_frame_candidate()
    candidate_meshes: list[tuple[object, ...]] = []
    for model in models:
        primitives = model.primitives_for_grasp(
            grasp_rotmat=quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj),
            contact_point_a=np.asarray(candidate_obj.contact_point_a_obj, dtype=float),
            contact_point_b=np.asarray(candidate_obj.contact_point_b_obj, dtype=float),
            grasp_center=np.asarray(candidate_obj.grasp_position_obj, dtype=float),
        )
        candidate_meshes.append(
            tuple(
                _primitive_to_trimesh(transform_primitive_to_world(primitive, source_pose_assembly))
                for primitive in primitives
            )
        )
    return tuple(candidate_meshes)


def _translated_meshes(query_meshes: tuple[object, ...], translation_m: np.ndarray) -> tuple[object, ...]:
    translated = []
    for query_mesh in query_meshes:
        mesh = query_mesh.copy()
        mesh.apply_translation(np.asarray(translation_m, dtype=float))
        translated.append(mesh)
    return tuple(translated)


def _swept_meshes(query_meshes: tuple[object, ...], translation_m: np.ndarray) -> tuple[object, ...]:
    swept = []
    for query_mesh in query_meshes:
        mesh = TriangleMesh(
            vertices_obj=np.asarray(query_mesh.vertices, dtype=float),
            faces=np.asarray(query_mesh.faces, dtype=np.int64),
        )
        swept.append(_triangle_mesh_to_trimesh(linear_sweep_triangle_mesh(mesh, translation_m)))
    return tuple(swept)


def _minimum_table_clearance(query_meshes: tuple[object, ...], *, table_z_m: float) -> float:
    return min(
        float(np.min(np.asarray(query_mesh.vertices, dtype=float)[:, 2]) - table_z_m) for query_mesh in query_meshes
    )


def _result(
    candidate: SavedGraspCandidate,
    *,
    status: str,
    reason: str,
    minimum_clearance_m: float | None,
    **details: object,
) -> HolderCandidateFeasibility:
    return HolderCandidateFeasibility(
        grasp_id=candidate.grasp_id,
        status=status,
        reason=reason,
        minimum_clearance_m=minimum_clearance_m,
        details=details,
    )


def _minimum(*values: float | None) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(finite) if finite else None


def _first_incoming_failure(
    *,
    gripper_meshes: tuple[object, ...],
    incoming_mesh_final: TriangleMesh,
    final_to_pre_translation_m: tuple[float, float, float],
    samples: int,
    clearance_margin_m: float,
) -> tuple[float | None, tuple[float, float, float] | None]:
    if trimesh is None:
        return None, None
    gripper_manager = trimesh.collision.CollisionManager()
    for index, mesh in enumerate(gripper_meshes):
        gripper_manager.add_object(f"gripper_{index}", mesh)
    final_mesh = _triangle_mesh_to_trimesh(incoming_mesh_final)
    final_to_pre = np.asarray(final_to_pre_translation_m, dtype=float)
    for progress in np.linspace(0.0, 1.0, int(samples)):
        translation = final_to_pre * (1.0 - float(progress))
        incoming_mesh = final_mesh.copy()
        incoming_mesh.apply_translation(translation)
        if bool(gripper_manager.in_collision_single(incoming_mesh)):
            return float(progress), tuple(float(value) for value in translation)
        if clearance_margin_m > 0.0:
            distance = float(gripper_manager.min_distance_single(incoming_mesh))
            if distance < clearance_margin_m:
                return float(progress), tuple(float(value) for value in translation)
    return None, None


def _prepare_holder_candidate(
    candidate: SavedGraspCandidate,
    *,
    table_z_m: float,
    source_pose_assembly: ObjectWorldPose,
    base_manager,
    planning: PlanningConfig,
    config: HolderFeasibilityConfig,
    model_cache: dict[tuple[float, float], tuple[object, ...]],
) -> _PreparedHolderCandidate:
    opening_meshes = _candidate_opening_meshes_assembly(
        candidate,
        source_pose_assembly=source_pose_assembly,
        planning=planning,
        model_cache=model_cache,
    )
    if not opening_meshes:
        raise RuntimeError("Holder candidate produced no gripper collision geometry.")
    contact_meshes = opening_meshes[0]
    approach_meshes = opening_meshes[-1]
    gripper_meshes = tuple(
        [*contact_meshes, *approach_meshes[1:]]
        if len(opening_meshes) > 1
        else contact_meshes
    )
    rotation_assembly_from_source = source_pose_assembly.rotation_world_from_object
    approach_axis_source = quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj)[:, 2]
    approach_axis_assembly = rotation_assembly_from_source @ approach_axis_source
    grasp_to_pregrasp = -approach_axis_assembly * config.pregrasp_offset_m
    # The gripper is physically at the approach opening during pregrasp and
    # the Cartesian approach. The selected contact state exists only at the
    # final grasp pose.
    pregrasp_meshes = _translated_meshes(approach_meshes, grasp_to_pregrasp)
    approach_swept_meshes = _swept_meshes(approach_meshes, grasp_to_pregrasp)

    def prepared(
        *,
        minimum_clearance_m: float | None,
        failure: HolderCandidateFeasibility | None,
    ) -> _PreparedHolderCandidate:
        return _PreparedHolderCandidate(
            candidate=candidate,
            gripper_meshes=gripper_meshes,
            pregrasp_meshes=pregrasp_meshes,
            approach_swept_meshes=approach_swept_meshes,
            grasp_to_pregrasp_translation_assembly_m=tuple(float(value) for value in grasp_to_pregrasp),
            static_minimum_clearance_m=minimum_clearance_m,
            static_failure=failure,
        )

    table_clearance = _minimum_table_clearance(gripper_meshes, table_z_m=table_z_m)
    minimum_clearance: float | None = table_clearance
    if table_clearance < -1.0e-9:
        return prepared(
            minimum_clearance_m=table_clearance,
            failure=_result(
                candidate,
                status="rejected",
                reason=REASON_TABLE_COLLISION,
                minimum_clearance_m=table_clearance,
                obstacle_type="table",
            ),
        )
    if table_clearance < config.table_clearance_margin_m:
        return prepared(
            minimum_clearance_m=table_clearance,
            failure=_result(
                candidate,
                status="rejected",
                reason=REASON_CLEARANCE_MARGIN_FAILED,
                minimum_clearance_m=table_clearance,
                obstacle_type="table",
                required_clearance_m=config.table_clearance_margin_m,
            ),
        )

    # The contact-state fingertips are expected to touch their own grasped
    # base. Checking that zero-gap mesh against the base makes FCL report the
    # two intended contacts as collisions. The approach opening still checks
    # the identical palm plus both fingers against the target, while the
    # contact state remains active for the table and every unrelated object.
    base_query = _query_manager(base_manager, approach_meshes)
    minimum_clearance = _minimum(minimum_clearance, base_query.minimum_distance_m)
    if base_query.collides:
        return prepared(
            minimum_clearance_m=minimum_clearance,
            failure=_result(
                candidate,
                status="rejected",
                reason=REASON_BASE_COLLISION,
                minimum_clearance_m=minimum_clearance,
                obstacle_type="base_part",
                obstacle_part_ids=list(base_query.obstacle_names),
            ),
        )
    if base_query.minimum_distance_m is not None and base_query.minimum_distance_m < config.geometry_clearance_margin_m:
        return prepared(
            minimum_clearance_m=minimum_clearance,
            failure=_result(
                candidate,
                status="rejected",
                reason=REASON_CLEARANCE_MARGIN_FAILED,
                minimum_clearance_m=minimum_clearance,
                obstacle_type="base_part",
                obstacle_part_ids=list(base_query.obstacle_names),
                required_clearance_m=config.geometry_clearance_margin_m,
            ),
        )

    pregrasp_table_clearance = _minimum_table_clearance(pregrasp_meshes, table_z_m=table_z_m)
    minimum_clearance = _minimum(minimum_clearance, pregrasp_table_clearance)
    if pregrasp_table_clearance < -1.0e-9:
        return prepared(
            minimum_clearance_m=minimum_clearance,
            failure=_result(
                candidate,
                status="rejected",
                reason=REASON_HOLDER_PREGRASP_COLLISION,
                minimum_clearance_m=minimum_clearance,
                obstacle_type="table",
                pregrasp_translation_assembly_m=list(grasp_to_pregrasp),
            ),
        )
    if pregrasp_table_clearance < config.table_clearance_margin_m:
        return prepared(
            minimum_clearance_m=minimum_clearance,
            failure=_result(
                candidate,
                status="rejected",
                reason=REASON_CLEARANCE_MARGIN_FAILED,
                minimum_clearance_m=minimum_clearance,
                obstacle_type="table_at_pregrasp",
                required_clearance_m=config.table_clearance_margin_m,
                pregrasp_translation_assembly_m=list(grasp_to_pregrasp),
            ),
        )

    pregrasp_base_query = _query_manager(base_manager, pregrasp_meshes)
    minimum_clearance = _minimum(minimum_clearance, pregrasp_base_query.minimum_distance_m)
    if pregrasp_base_query.collides:
        return prepared(
            minimum_clearance_m=minimum_clearance,
            failure=_result(
                candidate,
                status="rejected",
                reason=REASON_HOLDER_PREGRASP_COLLISION,
                minimum_clearance_m=minimum_clearance,
                obstacle_type="base_part",
                obstacle_part_ids=list(pregrasp_base_query.obstacle_names),
                pregrasp_translation_assembly_m=list(grasp_to_pregrasp),
            ),
        )

    approach_base_query = _query_manager(base_manager, approach_swept_meshes)
    minimum_clearance = _minimum(minimum_clearance, approach_base_query.minimum_distance_m)
    if approach_base_query.collides:
        return prepared(
            minimum_clearance_m=minimum_clearance,
            failure=_result(
                candidate,
                status="rejected",
                reason=REASON_HOLDER_APPROACH_SWEEP_COLLISION,
                minimum_clearance_m=minimum_clearance,
                obstacle_type="base_part",
                obstacle_part_ids=list(approach_base_query.obstacle_names),
                pregrasp_translation_assembly_m=list(grasp_to_pregrasp),
            ),
        )
    if (
        approach_base_query.minimum_distance_m is not None
        and approach_base_query.minimum_distance_m < config.geometry_clearance_margin_m
    ):
        return prepared(
            minimum_clearance_m=minimum_clearance,
            failure=_result(
                candidate,
                status="rejected",
                reason=REASON_CLEARANCE_MARGIN_FAILED,
                minimum_clearance_m=minimum_clearance,
                obstacle_type="holder_approach_sweep_vs_base",
                obstacle_part_ids=list(approach_base_query.obstacle_names),
                required_clearance_m=config.geometry_clearance_margin_m,
                pregrasp_translation_assembly_m=list(grasp_to_pregrasp),
            ),
        )
    return prepared(minimum_clearance_m=minimum_clearance, failure=None)


def _evaluate_candidate_for_state(
    prepared: _PreparedHolderCandidate,
    *,
    step: AssemblySequenceStep,
    part_meshes: dict[str, TriangleMesh],
    assembled_manager,
    incoming_sweep_manager,
    config: HolderFeasibilityConfig,
) -> HolderCandidateFeasibility:
    candidate = prepared.candidate
    if prepared.static_failure is not None:
        return prepared.static_failure
    gripper_meshes = prepared.gripper_meshes
    minimum_clearance = prepared.static_minimum_clearance_m
    assembled_query = _query_manager(assembled_manager, gripper_meshes)
    minimum_clearance = _minimum(minimum_clearance, assembled_query.minimum_distance_m)
    if assembled_query.collides:
        return _result(
            candidate,
            status="rejected",
            reason=REASON_ASSEMBLED_PART_COLLISION,
            minimum_clearance_m=minimum_clearance,
            obstacle_type="assembled_parts",
            obstacle_part_ids=list(assembled_query.obstacle_names),
        )
    if (
        assembled_query.minimum_distance_m is not None
        and assembled_query.minimum_distance_m < config.geometry_clearance_margin_m
    ):
        return _result(
            candidate,
            status="rejected",
            reason=REASON_CLEARANCE_MARGIN_FAILED,
            minimum_clearance_m=minimum_clearance,
            obstacle_type="assembled_parts",
            obstacle_part_ids=list(assembled_query.obstacle_names),
            required_clearance_m=config.geometry_clearance_margin_m,
        )

    grasp_to_pregrasp = np.asarray(
        prepared.grasp_to_pregrasp_translation_assembly_m,
        dtype=float,
    )
    pregrasp_query = _query_manager(assembled_manager, prepared.pregrasp_meshes)
    minimum_clearance = _minimum(minimum_clearance, pregrasp_query.minimum_distance_m)
    if pregrasp_query.collides:
        return _result(
            candidate,
            status="rejected",
            reason=REASON_HOLDER_PREGRASP_COLLISION,
            minimum_clearance_m=minimum_clearance,
            obstacle_type="assembled_parts",
            obstacle_part_ids=list(pregrasp_query.obstacle_names),
            pregrasp_translation_assembly_m=list(grasp_to_pregrasp),
        )

    approach_query = _query_manager(assembled_manager, prepared.approach_swept_meshes)
    minimum_clearance = _minimum(minimum_clearance, approach_query.minimum_distance_m)
    if approach_query.collides:
        return _result(
            candidate,
            status="rejected",
            reason=REASON_HOLDER_APPROACH_SWEEP_COLLISION,
            minimum_clearance_m=minimum_clearance,
            obstacle_type="assembled_parts",
            obstacle_part_ids=list(approach_query.obstacle_names),
            pregrasp_translation_assembly_m=list(grasp_to_pregrasp),
        )
    if (
        approach_query.minimum_distance_m is not None
        and approach_query.minimum_distance_m < config.geometry_clearance_margin_m
    ):
        return _result(
            candidate,
            status="rejected",
            reason=REASON_CLEARANCE_MARGIN_FAILED,
            minimum_clearance_m=minimum_clearance,
            obstacle_type="holder_approach_sweep",
            required_clearance_m=config.geometry_clearance_margin_m,
            pregrasp_translation_assembly_m=list(grasp_to_pregrasp),
        )

    incoming_query = _query_manager(incoming_sweep_manager, gripper_meshes)
    minimum_clearance = _minimum(minimum_clearance, incoming_query.minimum_distance_m)
    incoming_margin_failed = (
        incoming_query.minimum_distance_m is not None
        and incoming_query.minimum_distance_m < config.geometry_clearance_margin_m
    )
    if incoming_query.collides or incoming_margin_failed:
        progress, translation = _first_incoming_failure(
            gripper_meshes=gripper_meshes,
            incoming_mesh_final=part_meshes[step.incoming_part_id],
            final_to_pre_translation_m=step.final_to_pre_insertion_translation_m,
            samples=config.incoming_path_samples,
            clearance_margin_m=config.geometry_clearance_margin_m,
        )
        reason = REASON_INCOMING_PART_SWEEP_COLLISION if incoming_query.collides else REASON_CLEARANCE_MARGIN_FAILED
        return _result(
            candidate,
            status="rejected",
            reason=reason,
            minimum_clearance_m=minimum_clearance,
            obstacle_type="incoming_part_sweep",
            incoming_part_id=step.incoming_part_id,
            first_failing_insertion_progress=progress,
            first_failing_incoming_translation_assembly_m=(None if translation is None else list(translation)),
            required_clearance_m=config.geometry_clearance_margin_m,
        )

    return _result(
        candidate,
        status="accepted",
        reason=REASON_ACCEPTED,
        minimum_clearance_m=minimum_clearance,
        pregrasp_translation_assembly_m=list(grasp_to_pregrasp),
    )


def evaluate_holder_state_feasibility(
    *,
    sequence: AssemblySequence,
    holder_library: Stage1Result,
    planning: PlanningConfig,
    config: HolderFeasibilityConfig,
) -> HolderStateFeasibilityResult:
    """Filter one raw holder library against every selected-order state."""

    if not trimesh_fcl_backend_available():
        raise RuntimeError("trimesh with python-fcl is required for holder state feasibility.")
    if str(holder_library.bundle.metadata.get("base_part_id")) != sequence.base_part_id:
        raise ValueError(
            "Holder library base part does not match the assembly sequence: "
            f"{holder_library.bundle.metadata.get('base_part_id')!r} != {sequence.base_part_id!r}."
        )

    source_pose_assembly = ObjectWorldPose(
        position_world=holder_library.bundle.source_frame_origin_obj_world,
        orientation_xyzw_world=holder_library.bundle.source_frame_orientation_xyzw_obj_world,
    )
    part_meshes = {
        part.part_id: load_triangle_mesh(part.resolved_mesh_path, scale=sequence.mesh_scale) for part in sequence.parts
    }
    candidates = tuple(holder_library.bundle.candidates)
    states: list[HolderStateFeasibility] = []
    model_cache: dict[tuple[float, float], tuple[object, ...]] = {}
    base_manager = _manager_for_parts(
        part_meshes,
        (sequence.base_part_id,),
    )
    prepared_candidates = tuple(
        _prepare_holder_candidate(
            candidate,
            table_z_m=sequence.table_z_assembly_m,
            source_pose_assembly=source_pose_assembly,
            base_manager=base_manager,
            planning=planning,
            config=config,
            model_cache=model_cache,
        )
        for candidate in candidates
    )

    for step in sequence.steps:
        static_part_ids = tuple(
            part_id for part_id in step.assembled_part_ids_before if part_id != sequence.base_part_id
        )
        if not step.holder_base_available:
            candidate_results = tuple(
                _result(
                    prepared.candidate,
                    status="not_applicable",
                    reason=REASON_BASE_NOT_AVAILABLE,
                    minimum_clearance_m=None,
                )
                for prepared in prepared_candidates
            )
        else:
            assembled_manager = _manager_for_parts(part_meshes, static_part_ids)
            incoming_final = part_meshes[step.incoming_part_id]
            incoming_swept = linear_sweep_triangle_mesh(
                incoming_final,
                step.final_to_pre_insertion_translation_m,
            )
            incoming_sweep_manager = _manager_for_parts(
                {"incoming_sweep": incoming_swept},
                ("incoming_sweep",),
            )
            candidate_results = tuple(
                _evaluate_candidate_for_state(
                    prepared,
                    step=step,
                    part_meshes=part_meshes,
                    assembled_manager=assembled_manager,
                    incoming_sweep_manager=incoming_sweep_manager,
                    config=config,
                )
                for prepared in prepared_candidates
            )
        reason_counts = dict(sorted(Counter(result.reason for result in candidate_results).items()))
        states.append(
            HolderStateFeasibility(
                step_id=step.step_id,
                step_index=step.step_index,
                incoming_part_id=step.incoming_part_id,
                holder_base_available=step.holder_base_available,
                assembled_part_ids_before=step.assembled_part_ids_before,
                static_obstacle_part_ids=static_part_ids,
                incoming_final_to_pre_translation_m=step.final_to_pre_insertion_translation_m,
                candidate_results=candidate_results,
                reason_counts=reason_counts,
            )
        )

    return HolderStateFeasibilityResult(
        assembly=sequence.assembly,
        base_part_id=sequence.base_part_id,
        base_part_source=sequence.base_part_source,
        selected_order=sequence.selected_order,
        table_z_assembly_m=sequence.table_z_assembly_m,
        source_frame_pose_assembly=source_pose_assembly,
        config=config,
        candidates=candidates,
        states=tuple(states),
        collision_backend_name="trimesh_fcl",
        source_holder_cache_key=(
            None
            if holder_library.bundle.metadata.get("stage1_cache_key") is None
            else str(holder_library.bundle.metadata["stage1_cache_key"])
        ),
    )


def write_holder_state_feasibility_json(
    result: HolderStateFeasibilityResult,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_payload(), indent=2) + "\n", encoding="utf-8")


__all__ = [
    "HolderCandidateFeasibility",
    "HolderFeasibilityConfig",
    "HolderStateFeasibility",
    "HolderStateFeasibilityResult",
    "evaluate_holder_state_feasibility",
    "write_holder_state_feasibility_json",
]
