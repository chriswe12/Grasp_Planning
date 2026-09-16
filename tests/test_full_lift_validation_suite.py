"""Tests for exact-coverage aggregation of full scripted-lift suites."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.verify_full_lift_validation import verify_and_merge_suite, write_outputs


def _catalog(path: Path) -> None:
    np.savez_compressed(
        path,
        target_ids=np.asarray(["train_a", "validation_a", "test_a"]),
        part_ids=np.asarray(["part_0", "part_1", "part_1"]),
        split_ids=np.asarray(["train", "validation", "test"]),
    )


def _report(
    logs_root: Path,
    *,
    job_id: int,
    target_id: str,
    part_id: str,
    split: str,
    offset: int,
    status: str,
) -> None:
    report_dir = logs_root / "scripted_lift" / f"job_{job_id}"
    report_dir.mkdir(parents=True)
    payload = {
        "metadata": {
            "dataset_sha256": "dataset-hash",
            "dataset_shard": 0,
            "catalog_split": split,
            "target_offset": offset,
        },
        "summary": {},
        "attempts": [{"target_id": target_id, "part_id": part_id, "status": status}],
    }
    (report_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _manifest() -> dict[str, object]:
    return {
        "suite_id": "test-suite",
        "dataset_sha256": "dataset-hash",
        "expected_target_count": 3,
        "runs": [
            {
                "label": "train",
                "job_id": 1,
                "dataset_shard": 0,
                "catalog_split": "train",
                "target_offset": 0,
                "expected_targets": 1,
            },
            {
                "label": "validation",
                "job_id": 2,
                "dataset_shard": 0,
                "catalog_split": "validation",
                "target_offset": 0,
                "expected_targets": 1,
            },
            {
                "label": "test",
                "job_id": 3,
                "dataset_shard": 0,
                "catalog_split": "test",
                "target_offset": 0,
                "expected_targets": 1,
            },
        ],
    }


def test_full_suite_requires_and_writes_exact_catalog_coverage(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.npz"
    logs_root = tmp_path / "logs"
    _catalog(catalog_path)
    _report(
        logs_root,
        job_id=1,
        target_id="train_a",
        part_id="part_0",
        split="train",
        offset=0,
        status="success",
    )
    _report(
        logs_root,
        job_id=2,
        target_id="validation_a",
        part_id="part_1",
        split="validation",
        offset=0,
        status="object_not_lifted",
    )
    _report(
        logs_root,
        job_id=3,
        target_id="test_a",
        part_id="part_1",
        split="test",
        offset=0,
        status="object_unstable_before_grasp",
    )

    verification, rows = verify_and_merge_suite(
        _manifest(), logs_root=logs_root, catalog_path=catalog_path
    )
    assert verification["exact_catalog_coverage"] is True
    assert verification["summary"]["status_counts"] == {
        "object_not_lifted": 1,
        "object_unstable_before_grasp": 1,
        "success": 1,
    }
    output_dir = tmp_path / "merged"
    write_outputs(output_dir, verification=verification, rows=rows)
    assert (output_dir / "attempts.csv").is_file()
    assert "Targets: 3/3" in (output_dir / "report.md").read_text(encoding="utf-8")


def test_full_suite_rejects_duplicate_target_coverage(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.npz"
    logs_root = tmp_path / "logs"
    _catalog(catalog_path)
    for job_id, split in ((1, "train"), (2, "validation"), (3, "test")):
        _report(
            logs_root,
            job_id=job_id,
            target_id="train_a",
            part_id="part_0",
            split=split,
            offset=0,
            status="success",
        )
    with pytest.raises(ValueError, match="belongs to"):
        verify_and_merge_suite(_manifest(), logs_root=logs_root, catalog_path=catalog_path)


def test_full_lift_submission_is_bounded_and_report_only() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "euler/submit_full_lift_validation.sh"
    ).read_text(encoding="utf-8")
    assert 'BATCH_SIZE="${LIFT_BATCH_SIZE:-96}"' in source
    assert "--target-offset" in source
    assert "--max-targets" in source
    assert "--no-lift-videos" in source
    assert "--lift-speed-m-s 0.050" in source
    assert "watch_full_lift_validation.sh" in source
