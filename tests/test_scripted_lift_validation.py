from __future__ import annotations

import math

import numpy as np
import pytest

from grasp_planning.rl.scripted_lift_validation import (
    ScriptedLiftThresholds,
    balanced_subset_indices,
    build_liftability_records,
    classify_liftability_evidence,
    classify_scripted_lift,
    contiguous_subset_indices,
    resolve_explicit_target_indices,
    summarize_scripted_lifts,
)


def _classify(**overrides: float) -> str:
    values = {
        "commanded_lift_m": 0.06,
        "tcp_final_lift_m": 0.058,
        "object_final_lift_m": 0.052,
        "object_peak_lift_m": 0.055,
        "settle_translation_m": 0.001,
        "settle_rotation_rad": math.radians(1.0),
        "relative_drift_m": 0.004,
    }
    values.update(overrides)
    return classify_scripted_lift(**values, thresholds=ScriptedLiftThresholds())


def test_classification_separates_setup_and_retention_failures() -> None:
    assert _classify() == "success"
    assert _classify(settle_translation_m=0.004) == "object_unstable_before_grasp"
    assert _classify(tcp_final_lift_m=0.03) == "arm_lift_failed"
    assert _classify(object_final_lift_m=0.01, object_peak_lift_m=0.02) == "object_not_lifted"
    assert _classify(object_final_lift_m=0.02, object_peak_lift_m=0.05) == "object_dropped_after_peak"
    assert _classify(relative_drift_m=0.04) == "object_not_retained_near_tcp"


def test_lift_height_threshold_can_define_a_practical_retry_profile() -> None:
    measurements = {
        "commanded_lift_m": 0.06,
        "tcp_final_lift_m": 0.058,
        "object_final_lift_m": 0.0386,
        "object_peak_lift_m": 0.0397,
        "settle_translation_m": 0.0013,
        "settle_rotation_rad": math.radians(0.1),
        "relative_drift_m": 0.0127,
    }

    assert classify_scripted_lift(
        **measurements,
        thresholds=ScriptedLiftThresholds(),
    ) == "object_not_lifted"
    assert classify_scripted_lift(
        **measurements,
        thresholds=ScriptedLiftThresholds(minimum_final_lift_m=0.035),
    ) == "success"


def test_summary_excludes_simulator_invalid_attempts_from_grasp_rate() -> None:
    rows = [
        {"part_id": "a", "status": "success"},
        {"part_id": "a", "status": "object_not_lifted"},
        {"part_id": "b", "status": "arm_lift_failed"},
    ]
    summary = summarize_scripted_lifts(rows)
    assert summary["attempts"] == 3
    assert summary["valid_attempts"] == 2
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["per_part"]["b"]["valid_attempts"] == 0


def test_threshold_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        ScriptedLiftThresholds(minimum_final_lift_m=0.0)
    with pytest.raises(ValueError, match="must not exceed one"):
        ScriptedLiftThresholds(minimum_arm_lift_fraction=1.1)


def test_balanced_subset_round_robins_across_parts() -> None:
    indices = balanced_subset_indices(
        np.asarray(["a", "a", "a", "b", "b", "c"]),
        5,
    )
    assert indices.tolist() == [0, 3, 5, 1, 4]


def test_balanced_subset_can_select_next_grasp_per_part() -> None:
    part_ids = np.asarray(["a", "a", "b", "b", "c", "c"])

    indices = balanced_subset_indices(part_ids, 3, start_round=1)

    assert indices.tolist() == [1, 3, 5]


def test_explicit_target_indices_are_ordered_and_strict() -> None:
    available = np.asarray(["target_a", "target_b", "target_c"])

    assert resolve_explicit_target_indices(available, ["target_c", "target_a"]).tolist() == [2, 0]
    with pytest.raises(ValueError, match="absent"):
        resolve_explicit_target_indices(available, ["missing"])
    with pytest.raises(ValueError, match="unique"):
        resolve_explicit_target_indices(available, ["target_a", "target_a"])


def test_liftability_evidence_prefers_success_and_keeps_invalid_separate() -> None:
    assert classify_liftability_evidence([]) == "untested"
    assert classify_liftability_evidence(["object_unstable_before_grasp"]) == "setup_invalid"
    assert classify_liftability_evidence(["object_not_lifted"]) == "not_liftable"
    assert classify_liftability_evidence(["object_not_lifted", "success"]) == "liftable"
    with pytest.raises(ValueError, match="Unknown"):
        classify_liftability_evidence(["mystery"])


def test_liftability_records_preserve_untested_catalog_targets() -> None:
    records = build_liftability_records(
        np.asarray(["a", "b", "c"]),
        [
            {"target_id": "a", "status": "object_not_lifted"},
            {"target_id": "a", "status": "success"},
            {"target_id": "b", "status": "arm_lift_failed"},
        ],
    )

    assert [record["label"] for record in records] == ["liftable", "setup_invalid", "untested"]
    assert records[0]["attempt_count"] == 2
    with pytest.raises(ValueError, match="absent"):
        build_liftability_records(np.asarray(["a"]), [{"target_id": "missing", "status": "success"}])


def test_contiguous_subset_indices_bounds_the_last_batch() -> None:
    assert contiguous_subset_indices(25, offset=12, limit=12).tolist() == list(range(12, 24))
    assert contiguous_subset_indices(25, offset=24, limit=12).tolist() == [24]
    with pytest.raises(ValueError, match="offset"):
        contiguous_subset_indices(25, offset=25, limit=12)
