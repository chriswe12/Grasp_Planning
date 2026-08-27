"""Generate a reusable, state-independent holder-grasp library for one base part."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from grasp_planning.grasping.collision import GRIPPER_COLLISION_MODEL_KUKA_Y, GRIPPER_COLLISION_MODEL_PDZ
from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspCandidate,
    save_grasp_bundle,
)

from .assembly_sequence import AssemblySequence
from .fabrica_pipeline import GeometryConfig, PlanningConfig, Stage1Result, generate_stage1_result
from .holder_grasp_debug_html import write_holder_grasp_debug_html


def _holder_candidate(candidate: SavedGraspCandidate, *, base_part_id: str) -> SavedGraspCandidate:
    source_grasp_id = candidate.grasp_id
    if source_grasp_id.startswith("g") and source_grasp_id[1:].isdigit():
        holder_grasp_id = f"h{source_grasp_id[1:]}"
    else:
        holder_grasp_id = f"h_{source_grasp_id}"
    return replace(
        candidate,
        grasp_id=holder_grasp_id,
        metadata={
            **(candidate.metadata or {}),
            "candidate_role": "assembly_holder",
            "base_part_id": base_part_id,
            "source_grasp_id": source_grasp_id,
        },
    )


def generate_holder_grasp_library(
    *,
    sequence: AssemblySequence,
    planning: PlanningConfig,
) -> Stage1Result:
    """Generate scored base contacts without filtering against assembly states.

    The existing Stage-1 generator supplies object/self collision rejection,
    scoring, detailed KUKA collision geometry, and cache invalidation. Assembly,
    table, incoming-part, and robot-pair filters intentionally belong to later
    dual-robot stages.
    """

    if planning.gripper_collision_model not in {GRIPPER_COLLISION_MODEL_KUKA_Y, GRIPPER_COLLISION_MODEL_PDZ}:
        raise ValueError(
            "Holder grasp generation requires a mesh collision model: "
            f"'{GRIPPER_COLLISION_MODEL_KUKA_Y}' or '{GRIPPER_COLLISION_MODEL_PDZ}'."
        )

    base_part = sequence.parts_by_id[sequence.base_part_id]
    holder_planning = replace(planning, skip_stage1_collision_checks=True)
    result = generate_stage1_result(
        geometry=GeometryConfig(
            target_mesh_path=str(base_part.resolved_mesh_path),
            mesh_scale=sequence.mesh_scale,
            assembly_glob=None,
            assembly_obstacle_paths=(),
        ),
        planning=holder_planning,
    )

    candidates = tuple(
        _holder_candidate(candidate, base_part_id=sequence.base_part_id) for candidate in result.bundle.candidates
    )
    raw_candidates = tuple(
        _holder_candidate(candidate, base_part_id=sequence.base_part_id) for candidate in result.raw_candidates
    )
    metadata = {
        **result.bundle.metadata,
        "artifact_kind": "holder_base_candidate_library",
        "planning_stage": "dual_robot_stage_1",
        "generated_by": "scripts/build_holder_grasp_library.py",
        "assembly": sequence.assembly,
        "base_part_id": sequence.base_part_id,
        "base_part_source": sequence.base_part_source,
        "base_part_role": base_part.role,
        "base_part_touches_table": base_part.touches_table,
        "base_part_order_index": sequence.base_part_order_index,
        "first_holder_step_index": sequence.first_holder_step_index,
        "selected_assembly_order": list(sequence.selected_order),
        "holder_candidate_count": len(candidates),
        "state_filter_applied": False,
        "table_filter_applied": False,
        "incoming_part_sweep_filter_applied": False,
        "robot_pair_filter_applied": False,
        "frame_contract": {
            "candidate_frame": "canonical base-part source frame",
            "source_frame_pose": "base source frame expressed in the Fabrica assembly asset frame",
            "assembly_frame": "Fabrica OBJ assembled coordinates scaled by mesh_scale",
        },
        "table_context": {
            "z_assembly_m": sequence.table_z_assembly_m,
            "contact_tolerance_m": sequence.table_contact_tolerance_m,
            "contact_part_ids": list(sequence.table_contact_part_ids),
        },
    }
    return replace(
        result,
        bundle=replace(result.bundle, candidates=candidates, metadata=metadata),
        raw_candidates=raw_candidates,
    )


def write_holder_grasp_library_artifacts(
    result: Stage1Result,
    *,
    sequence: AssemblySequence,
    planning: PlanningConfig,
    output_json: str | Path,
    output_html: str | Path,
) -> None:
    """Write the Stage-1 holder library and its interactive visual debugger."""

    save_grasp_bundle(result.bundle, output_json)
    base_part = sequence.parts_by_id[sequence.base_part_id]
    metadata_lines = [
        f"assembly:          {sequence.assembly}",
        f"base_part:         {sequence.base_part_id}",
        f"base_asset_role:   {base_part.role}",
        f"base_selection:    {sequence.base_part_source}",
        f"selected_order:    {' -> '.join(sequence.selected_order)}",
        f"first_holder_step: {sequence.first_holder_step_index}",
        f"base_touches_table:{base_part.touches_table}",
        f"collision_backend: {result.collision_backend_name}",
        f"gripper_model:     {planning.gripper_collision_model}",
        "state_filter:      not applied (Stage 2)",
        "table_filter:      not applied (Stage 2)",
        "incoming_sweep:    not applied (Stage 2)",
        f"raw_candidates:    {result.raw_candidate_count}",
        f"holder_candidates: {len(result.bundle.candidates)}",
        f"contact_offsets_x: {tuple(planning.contact_lateral_offsets_m)}",
        f"contact_offsets_z: {tuple(planning.contact_approach_offsets_m)}",
        f"source_origin_asm: {tuple(round(v, 6) for v in result.target_pose_in_obj_world.position_world)}",
    ]
    source_pose = result.target_pose_in_obj_world
    source_origin = source_pose.translation_world
    source_rotation = source_pose.rotation_world_from_object
    base_extent = np.ptp(np.asarray(result.target_mesh_local.vertices_obj, dtype=float), axis=0)
    table_radius = max(0.15, 3.0 * float(np.max(base_extent[:2])))
    table_corners_assembly = np.asarray(
        [
            [source_origin[0] - table_radius, source_origin[1] - table_radius, sequence.table_z_assembly_m],
            [source_origin[0] + table_radius, source_origin[1] - table_radius, sequence.table_z_assembly_m],
            [source_origin[0] + table_radius, source_origin[1] + table_radius, sequence.table_z_assembly_m],
            [source_origin[0] - table_radius, source_origin[1] + table_radius, sequence.table_z_assembly_m],
        ],
        dtype=float,
    )
    table_corners_local = (table_corners_assembly - source_origin[None, :]) @ source_rotation
    write_holder_grasp_debug_html(
        title="Fabrica Base Holder Candidate Library",
        subtitle=(
            f"Stage 1: scored {planning.gripper_collision_model} contacts on the designated base only. "
            "Assembly-state, table, insertion-sweep, and robot-pair filters have not yet been applied."
        ),
        mesh_local=result.target_mesh_local,
        candidates=result.bundle.candidates,
        output_html=output_html,
        metadata_lines=metadata_lines,
        table_plane_local=np.round(table_corners_local, 6).tolist(),
        gripper_collision_model=planning.gripper_collision_model,
    )
