from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import SavedGraspCandidate
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.pipeline.assembly_sequence import (
    AssemblyPartSpec,
    AssemblySequence,
    AssemblySequenceStep,
)
from grasp_planning.pipeline.transition_symmetry import (
    compile_step_transition_symmetries,
    expand_grasp_candidates_by_symmetry,
    load_assembly_symmetry_records,
)


def _write_triangle(path: Path, *, x_offset: float = 0.0) -> None:
    path.write_text(
        "\n".join(
            (
                f"v {x_offset + 0.1} 0 0",
                f"v {x_offset - 0.1} 0 0",
                f"v {x_offset} 0.1 0",
                f"v {x_offset} -0.1 0",
                "f 1 3 2",
                "f 1 2 4",
                "",
            )
        ),
        encoding="utf-8",
    )


def _record(name: str, matrix: np.ndarray) -> dict[str, object]:
    return {
        "name": name,
        "type": "identity" if name == "identity" else "finite_rotation",
        "description": name,
        "source": "test",
        "angle_deg": 0.0 if name == "identity" else 180.0,
        "matrix_obj": matrix.tolist(),
    }


def _fixture(tmp_path: Path) -> tuple[AssemblySequence, AssemblySequenceStep]:
    paths = {part_id: tmp_path / f"{part_id}.obj" for part_id in ("2", "1", "0")}
    for path in paths.values():
        _write_triangle(path)
    parts = tuple(
        AssemblyPartSpec(
            part_id=part_id,
            mesh_path=str(path),
            role="moving_part",
            bounds_min_assembly_m=(-0.1, -0.1, 0.0),
            bounds_max_assembly_m=(0.1, 0.1, 0.0),
            table_clearance_m=0.0,
            touches_table=True,
            vertex_count=4,
            face_count=2,
            asset_record={},
            resolved_mesh_path=path,
        )
        for part_id, path in paths.items()
    )
    final_to_pre = np.eye(4)
    final_to_pre[0, 3] = -0.4
    step = AssemblySequenceStep(
        step_id="step_001_part_0",
        step_index=1,
        incoming_part_id="0",
        incoming_part_role="moving_part",
        assembled_part_ids_before=("2",),
        assembled_part_ids_after=("2", "0"),
        base_part_status="assembled",
        holder_base_available=True,
        final_to_pre_insertion_transform_m=tuple(tuple(float(value) for value in row) for row in final_to_pre),
        final_to_pre_insertion_translation_m=(-0.4, 0.0, 0.0),
        pre_to_final_insertion_vector_m=(0.4, 0.0, 0.0),
        insertion_distance_m=0.4,
        disassembly_path_waypoints=None,
    )
    sequence = AssemblySequence(
        assembly="fixture",
        base_part_id="2",
        base_part_source="selected_order[0]",
        base_part_order_index=0,
        first_holder_step_index=1,
        selected_order=("2", "1", "0"),
        mesh_scale=1.0,
        table_z_assembly_m=0.0,
        table_contact_tolerance_m=1.0e-6,
        table_contact_part_ids=("2",),
        parts=parts,
        steps=(step,),
        precedence_plan_record={},
        pre_insertion_poses_record={},
        warnings=(),
        source_assembly_dir=tmp_path,
    )
    identity = np.eye(4)
    half_turn = np.diag((-1.0, -1.0, 1.0, 1.0))
    payload = {
        "schema_version": 1,
        "assembly": "fixture",
        "mesh_scale": 1.0,
        "parts": {
            "2": {"symmetries": [_record("identity", identity), _record("z180", half_turn)]},
            "1": {"symmetries": [_record("identity", identity)]},
            "0": {"symmetries": [_record("identity", identity), _record("z180", half_turn)]},
        },
    }
    (tmp_path / "symmetries.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return sequence, step


def _candidate() -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id="g0001",
        grasp_position_obj=(0.1, 0.0, 0.0),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(0.1, -0.01, 0.0),
        contact_point_b_obj=(0.1, 0.01, 0.0),
        contact_normal_a_obj=(0.0, 1.0, 0.0),
        contact_normal_b_obj=(0.0, -1.0, 0.0),
        jaw_width=0.02,
        roll_angle_rad=0.0,
        score=0.8,
    )


def test_pickup_grasp_symmetry_preserves_parent_and_transforms_geometry(
    tmp_path: Path,
) -> None:
    sequence, _ = _fixture(tmp_path)
    records, metadata = load_assembly_symmetry_records(sequence)
    expanded, expansion = expand_grasp_candidates_by_symmetry(
        (_candidate(),),
        source_pose_assembly=ObjectWorldPose(
            position_world=(0.0, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        ),
        symmetry_records=records["0"],
    )

    assert metadata["load_status"] == "loaded"
    assert len(expanded) == 2
    transformed = next(candidate for candidate in expanded if "__sym_" in candidate.grasp_id)
    assert np.allclose(transformed.grasp_position_obj, (-0.1, 0.0, 0.0))
    assert transformed.metadata["symmetry_pickup_parent_grasp_id"] == "g0001"
    assert transformed.metadata["symmetry_pickup_name"] == "z180"
    assert expansion["symmetry_pickup_derived_candidate_count"] == 1


def test_partial_assembly_symmetry_creates_opposite_preinsertion_corridor(
    tmp_path: Path,
) -> None:
    sequence, step = _fixture(tmp_path)
    candidates, metadata = compile_step_transition_symmetries(
        sequence=sequence,
        step=step,
        incoming_source_pose_assembly=ObjectWorldPose(
            position_world=(0.0, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        ),
        geometry_tolerance_m=1.0e-7,
    )

    assert metadata["accepted_partial_assembly_symmetry_count"] == 2
    assert metadata["raw_transition_combination_count"] == 4
    assert metadata["deduplicated_transition_combination_count"] == 2
    assert len(candidates) == 2
    opposite = next(
        candidate
        for candidate in candidates
        if candidate.partial_assembly_symmetry_name == "identity"
        and candidate.incoming_destination_symmetry_name == "z180"
    )
    assert np.allclose(
        opposite.preinsertion_source_matrix[:3, 3],
        (0.4, 0.0, 0.0),
    )
    assert np.allclose(opposite.final_source_matrix[:3, 3], (0.0, 0.0, 0.0))


def test_added_prefix_part_removes_broken_partial_assembly_symmetry(
    tmp_path: Path,
) -> None:
    sequence, step = _fixture(tmp_path)
    later_step = AssemblySequenceStep(
        **{
            **step.__dict__,
            "step_id": "step_002_part_0",
            "step_index": 2,
            "assembled_part_ids_before": ("2", "1"),
            "assembled_part_ids_after": ("2", "1", "0"),
        }
    )
    candidates, metadata = compile_step_transition_symmetries(
        sequence=sequence,
        step=later_step,
        incoming_source_pose_assembly=ObjectWorldPose(
            position_world=(0.0, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        ),
        geometry_tolerance_m=1.0e-7,
    )

    assert metadata["accepted_partial_assembly_symmetry_count"] == 1
    assert len(candidates) == 2
    assert {candidate.partial_assembly_symmetry_name for candidate in candidates} == {"identity"}
    assert any(
        rejection["reason"] == "assembled_prefix_not_equivalent" and rejection["part_id"] == "1"
        for rejection in metadata["rejected_partial_assembly_symmetries"]
    )


def test_incoming_candidate_limit_does_not_truncate_equivalence_proof(
    tmp_path: Path,
) -> None:
    sequence, step = _fixture(tmp_path)
    symmetry_path = tmp_path / "symmetries.json"
    payload = json.loads(symmetry_path.read_text(encoding="utf-8"))
    quarter_turn = np.asarray(
        (
            (0.0, -1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    payload["parts"]["0"]["symmetries"].insert(
        1,
        _record("z90", quarter_turn),
    )
    symmetry_path.write_text(json.dumps(payload), encoding="utf-8")

    candidates, metadata = compile_step_transition_symmetries(
        sequence=sequence,
        step=step,
        incoming_source_pose_assembly=ObjectWorldPose(
            position_world=(0.0, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        ),
        geometry_tolerance_m=1.0e-7,
        max_incoming_transforms=1,
    )

    assert metadata["incoming_symmetry_candidate_count"] == 2
    assert metadata["incoming_symmetry_match_record_count"] == 3
    assert metadata["accepted_partial_assembly_symmetry_count"] == 2
    assert any(
        candidate.partial_assembly_symmetry_name == "z180" and candidate.incoming_equivalence_symmetry_name == "z180"
        for candidate in candidates
    )
