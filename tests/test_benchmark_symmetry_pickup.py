from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from grasp_planning.grasping.fabrica_grasp_debug import SavedGraspCandidate
from scripts.run_grasp_generation_benchmark import _apply_cli_overrides, _best_candidate_payload
from scripts.run_grasp_pipeline import _planning_config


def _candidate() -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id="g0001__sym_turn_z",
        grasp_position_obj=(0.0, 0.0, 0.0),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(0.0, -0.01, 0.0),
        contact_point_b_obj=(0.0, 0.01, 0.0),
        contact_normal_a_obj=(0.0, 1.0, 0.0),
        contact_normal_b_obj=(0.0, -1.0, 0.0),
        jaw_width=0.02,
        roll_angle_rad=0.0,
        score=0.7,
        metadata={
            "symmetry_pickup_parent_grasp_id": "g0001",
            "symmetry_pickup_name": "turn_z",
        },
    )


def test_benchmark_cli_overrides_enable_symmetry_pickup() -> None:
    args = Namespace(
        robust_tilt_deg=None,
        no_stage1_cache=False,
        skip_stage1_collision_checks=False,
        symmetry_pickup_enabled=True,
        symmetry_asset_path="assets/obj/fabrica/beam/symmetries.json",
        symmetry_max_transforms=3,
        symmetry_next_orientation_limit=5,
        fallback_enabled=None,
    )

    payload = _apply_cli_overrides({"planning": {}}, args, Path("artifacts/test"))
    planning = _planning_config(payload)

    assert planning.symmetry_pickup_enabled is True
    assert planning.symmetry_asset_path == "assets/obj/fabrica/beam/symmetries.json"
    assert planning.symmetry_max_transforms == 3
    assert planning.symmetry_next_orientation_limit == 5


def test_best_candidate_payload_reports_symmetry_parent() -> None:
    payload = _best_candidate_payload([_candidate()])

    assert payload is not None
    assert payload["grasp_id"] == "g0001__sym_turn_z"
    assert payload["parent_grasp_id"] == "g0001"
    assert payload["pickup_symmetry_name"] == "turn_z"
    assert payload["metadata"]["symmetry_pickup_name"] == "turn_z"
