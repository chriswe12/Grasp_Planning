#!/usr/bin/env python3
"""Verify and merge one complete Euler scripted-lift suite."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.rl.scripted_lift_validation import summarize_scripted_lifts  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=REPO_ROOT / "logs/euler",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/fabrica_all_v1/merged/goal_catalog.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def verify_and_merge_suite(
    manifest: dict[str, object],
    *,
    logs_root: Path,
    catalog_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Validate exact target coverage and return one merged result table."""

    runs = list(manifest.get("runs", []))
    if not runs:
        raise ValueError("Lift-suite manifest contains no runs.")
    with np.load(catalog_path, allow_pickle=False) as catalog:
        catalog_target_ids = catalog["target_ids"].astype(str)
        catalog_split_ids = catalog["split_ids"].astype(str)
    if len(set(catalog_target_ids.tolist())) != len(catalog_target_ids):
        raise ValueError("Catalog target IDs are not unique.")
    split_by_target = dict(zip(catalog_target_ids.tolist(), catalog_split_ids.tolist(), strict=True))

    merged_rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    for raw_run in runs:
        run = dict(raw_run)
        job_id = int(run["job_id"])
        report_path = logs_root / "scripted_lift" / f"job_{job_id}" / "summary.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing pulled report for job {job_id}: {report_path}")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        metadata = dict(payload["metadata"])
        attempts = [dict(row) for row in payload["attempts"]]
        expected = int(run["expected_targets"])
        if len(attempts) != expected:
            raise ValueError(f"Job {job_id} produced {len(attempts)} attempts; expected {expected}.")
        expected_metadata = {
            "dataset_sha256": str(manifest["dataset_sha256"]),
            "dataset_shard": int(run["dataset_shard"]),
            "catalog_split": str(run["catalog_split"]),
            "target_offset": int(run["target_offset"]),
        }
        for name, expected_value in expected_metadata.items():
            actual_value = metadata.get(name)
            if actual_value != expected_value:
                raise ValueError(
                    f"Job {job_id} metadata {name}={actual_value!r}; expected {expected_value!r}."
                )
        for row in attempts:
            target_id = str(row["target_id"])
            expected_split = str(run["catalog_split"])
            if target_id not in split_by_target:
                raise ValueError(f"Job {job_id} returned unknown target {target_id!r}.")
            if split_by_target[target_id] != expected_split:
                raise ValueError(
                    f"Target {target_id!r} belongs to {split_by_target[target_id]!r}, "
                    f"not {expected_split!r}."
                )
            row["source_job_id"] = job_id
            row["dataset_shard"] = int(run["dataset_shard"])
            row["catalog_split"] = expected_split
            merged_rows.append(row)
        run_summaries.append(
            {
                "job_id": job_id,
                "label": str(run["label"]),
                "attempts": len(attempts),
                "status_counts": dict(Counter(str(row["status"]) for row in attempts)),
            }
        )

    observed_ids = [str(row["target_id"]) for row in merged_rows]
    duplicates = sorted(target_id for target_id, count in Counter(observed_ids).items() if count > 1)
    missing = sorted(set(catalog_target_ids.tolist()) - set(observed_ids))
    extra = sorted(set(observed_ids) - set(catalog_target_ids.tolist()))
    if duplicates or missing or extra:
        raise ValueError(
            "Lift suite does not exactly cover the catalog: "
            f"duplicates={len(duplicates)}, missing={len(missing)}, extra={len(extra)}."
        )
    expected_total = int(manifest["expected_target_count"])
    if len(merged_rows) != expected_total or expected_total != len(catalog_target_ids):
        raise ValueError(
            f"Coverage mismatch: rows={len(merged_rows)}, manifest={expected_total}, "
            f"catalog={len(catalog_target_ids)}."
        )

    overall = summarize_scripted_lifts(merged_rows)
    by_split = {
        split: summarize_scripted_lifts(
            row for row in merged_rows if str(row["catalog_split"]) == split
        )
        for split in ("train", "validation", "test")
    }
    verification = {
        "schema_version": 1,
        "suite_id": manifest["suite_id"],
        "dataset_sha256": manifest["dataset_sha256"],
        "catalog_path": str(catalog_path),
        "exact_catalog_coverage": True,
        "expected_targets": expected_total,
        "observed_targets": len(merged_rows),
        "jobs": len(runs),
        "summary": overall,
        "by_split": by_split,
        "runs": run_summaries,
    }
    return verification, merged_rows


def write_outputs(
    output_dir: Path,
    *,
    verification: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with (output_dir / "attempts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = dict(verification["summary"])
    report = [
        "# Full Fabrica scripted-lift screen",
        "",
        f"- Suite: `{verification['suite_id']}`",
        f"- Exact catalog coverage: {verification['exact_catalog_coverage']}",
        f"- Targets: {verification['observed_targets']}/{verification['expected_targets']}",
        f"- Euler jobs: {verification['jobs']}",
        f"- Simulator-invalid setups: {summary['simulator_invalid_attempts']}",
        f"- Valid physical attempts: {summary['valid_attempts']}",
        f"- Retained pickups: {summary['successes']}",
        f"- Retained-pickup rate: {100.0 * float(summary['success_rate']):.1f}%",
        "",
        "## Outcomes",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    report.extend(f"| {name} | {count} |" for name, count in summary["status_counts"].items())
    report.extend(
        [
            "",
            "## Primary split",
            "",
            "| Split | Attempts | Setup invalid | Valid | Success | Rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split, split_summary_raw in dict(verification["by_split"]).items():
        split_summary = dict(split_summary_raw)
        report.append(
            f"| {split} | {split_summary['attempts']} | "
            f"{split_summary['simulator_invalid_attempts']} | {split_summary['valid_attempts']} | "
            f"{split_summary['successes']} | {100.0 * float(split_summary['success_rate']):.1f}% |"
        )
    report.extend(
        [
            "",
            "This is a first-pass permissive scripted feasibility screen. Simulator-invalid setups are",
            "not counted as physical failures. A failed first attempt can be retried diagnostically before",
            "the target is excluded from lift-reward training.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    try:
        manifest = json.loads(args.manifest.expanduser().resolve().read_text(encoding="utf-8"))
        verification, rows = verify_and_merge_suite(
            manifest,
            logs_root=args.logs_root.expanduser().resolve(),
            catalog_path=args.catalog.expanduser().resolve(),
        )
        output_dir = args.output_dir.expanduser().resolve()
        write_outputs(output_dir, verification=verification, rows=rows)
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"[LIFT-SUITE] targets={len(rows)} exact_catalog_coverage=true")
    print(f"[LIFT-SUITE] report={output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
