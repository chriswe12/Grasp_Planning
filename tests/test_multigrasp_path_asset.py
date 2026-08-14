from __future__ import annotations

import pytest

from isaac_rl.scripts.build_multigrasp_path_asset import (
    _is_multipart_planned_manifest,
    _target_gripper_apertures,
)


def test_legacy_planned_manifest_ignores_empty_multipart_placeholders() -> None:
    manifest = {"schema_version": 3, "parts": [], "split": {}}
    targets = [{"target_id": "orientation_000__g1"}]

    assert not _is_multipart_planned_manifest(manifest, targets)


def test_multipart_planned_manifest_requires_complete_split_metadata() -> None:
    manifest = {"schema_version": 5, "parts": [{"part_id": "0"}], "split": {}}
    targets = [
        {
            "target_id": "part_0__orientation_000__g1",
            "part_id": "0",
            "local_orientation_id": "orientation_000",
            "split": "train",
        }
    ]

    with pytest.raises(ValueError, match="split names"):
        _is_multipart_planned_manifest(manifest, targets)


def test_complete_multipart_planned_manifest_is_detected() -> None:
    manifest = {
        "schema_version": 5,
        "parts": [{"part_id": "0"}],
        "split": {"names": ["train", "validation", "test"]},
    }
    targets = [
        {
            "target_id": "part_0__orientation_000__g1",
            "part_id": "0",
            "local_orientation_id": "orientation_000",
            "split": "validation",
        }
    ]

    assert _is_multipart_planned_manifest(manifest, targets)


def test_target_gripper_apertures_require_jaw_width_plus_ten_mm() -> None:
    jaw_widths, approach_widths = _target_gripper_apertures(
        [
            {
                "target_id": "target_a",
                "world_grasp": {
                    "jaw_width_m": 0.042,
                    "gripper_width_m": 0.052,
                },
            }
        ]
    )

    assert jaw_widths.tolist() == pytest.approx([0.042])
    assert approach_widths.tolist() == pytest.approx([0.052])

    with pytest.raises(ValueError, match="jaw width \+ 10 mm"):
        _target_gripper_apertures(
            [
                {
                    "target_id": "bad",
                    "world_grasp": {
                        "jaw_width_m": 0.042,
                        "gripper_width_m": 0.084,
                    },
                }
            ]
        )
