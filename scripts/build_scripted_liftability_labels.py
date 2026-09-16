#!/usr/bin/env python3
"""Build catalog-aligned liftability labels from scripted Isaac trial reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.rl.fabrica_dataset import sha256_file  # noqa: E402
from grasp_planning.rl.scripted_lift_validation import build_liftability_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "isaac_rl/data/fabrica_all_v1/merged/goal_catalog.npz",
    )
    parser.add_argument(
        "--attempt-root",
        type=Path,
        action="append",
        required=True,
        help="Directory recursively containing attempts.csv files; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _attempt_paths(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        if root.is_file():
            if root.name != "attempts.csv":
                raise ValueError(f"Attempt file must be named attempts.csv: {root}")
            paths.add(root)
        elif root.is_dir():
            paths.update(path.resolve() for path in root.rglob("attempts.csv"))
        else:
            raise FileNotFoundError(root)
    if not paths:
        raise FileNotFoundError("No attempts.csv files found below the requested roots.")
    return sorted(paths)


def main() -> None:
    args = _parser().parse_args()
    catalog_path = args.catalog.expanduser().resolve()
    attempt_paths = _attempt_paths(args.attempt_root)
    attempts: list[dict[str, str]] = []
    for path in attempt_paths:
        with path.open(newline="", encoding="utf-8") as stream:
            attempts.extend(dict(row) for row in csv.DictReader(stream))

    with np.load(catalog_path, allow_pickle=False) as catalog:
        target_ids = catalog["target_ids"].astype(str)
        part_ids = catalog["part_ids"].astype(str)
        split_ids = catalog["split_ids"].astype(str)
    records = build_liftability_records(target_ids, attempts)
    labels = np.asarray([record["label"] for record in records])
    label_counts = dict(sorted(Counter(labels.tolist()).items()))
    split_label_counts = {
        split: dict(sorted(Counter(labels[split_ids == split].tolist()).items()))
        for split in sorted(set(split_ids.tolist()))
    }
    catalog_sha256 = sha256_file(catalog_path)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT / "artifacts/scripted_grasp_lift_validation" / f"liftability_labels_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "liftability_labels.npz",
        schema_version=np.asarray(1, dtype=np.int64),
        source_catalog_path=np.asarray(str(catalog_path)),
        source_catalog_sha256=np.asarray(catalog_sha256),
        target_ids=target_ids,
        part_ids=part_ids,
        split_ids=split_ids,
        labels=labels,
        label_codes=np.asarray([record["label_code"] for record in records], dtype=np.int8),
        attempt_counts=np.asarray([record["attempt_count"] for record in records], dtype=np.int32),
        success_counts=np.asarray([record["success_count"] for record in records], dtype=np.int32),
        physical_failure_counts=np.asarray(
            [record["physical_failure_count"] for record in records], dtype=np.int32
        ),
        simulator_invalid_counts=np.asarray(
            [record["simulator_invalid_count"] for record in records], dtype=np.int32
        ),
    )
    with (output_dir / "liftability_labels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "schema_version": 1,
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": catalog_sha256,
        "attempt_files": [str(path) for path in attempt_paths],
        "attempt_rows": len(attempts),
        "catalog_targets": len(records),
        "label_counts": label_counts,
        "split_label_counts": split_label_counts,
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Scripted liftability labels",
        "",
        f"- Catalog targets: {len(records)}",
        f"- Scripted attempt rows: {len(attempts)}",
        f"- Source attempt files: {len(attempt_paths)}",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    report.extend(f"| {label} | {count} |" for label, count in label_counts.items())
    report.extend(
        [
            "",
            "## By catalog split",
            "",
            "| Split | Liftable | Not liftable | Setup invalid | Untested |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split, counts in split_label_counts.items():
        report.append(
            f"| {split} | {counts.get('liftable', 0)} | {counts.get('not_liftable', 0)} "
            f"| {counts.get('setup_invalid', 0)} | {counts.get('untested', 0)} |"
        )
    report.extend(
        [
            "",
            "A target is `liftable` if any approved scripted trial succeeded. A valid physical failure",
            "with no success is `not_liftable`. Targets having only simulator/setup-invalid trials are",
            "`setup_invalid`; targets without evidence remain `untested`.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[LIFTABILITY] labels={label_counts} attempts={len(attempts)} files={len(attempt_paths)}")
    print(f"[LIFTABILITY] report={output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
