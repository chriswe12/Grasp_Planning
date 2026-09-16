#!/usr/bin/env python3
"""Compare benchmark result JSON files while ignoring non-outcome metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

IGNORED_KEYS = {
    "git",
    "python_executable",
    "python_version",
    "stage1_cache_enabled",
    "stage1_cache_hit",
    "timestamp",
    "timings_s",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(value: Any, *, output_root: Path) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if key in IGNORED_KEYS:
                continue
            if key == "cli_args" and isinstance(child, dict):
                normalized[key] = {
                    arg_key: _normalize(arg_value, output_root=output_root)
                    for arg_key, arg_value in child.items()
                    if arg_key not in {"jobs", "no_stage1_cache", "output_dir"}
                }
                continue
            if key == "benchmark" and isinstance(child, dict):
                normalized[key] = {
                    bench_key: _normalize(bench_value, output_root=output_root)
                    for bench_key, bench_value in child.items()
                    if bench_key != "output_dir"
                }
                continue
            normalized[key] = _normalize(child, output_root=output_root)
        return normalized
    if isinstance(value, list):
        return [_normalize(child, output_root=output_root) for child in value]
    if isinstance(value, str):
        root = output_root.as_posix()
        return value.replace(root, "<OUTPUT>")
    return value


def _first_difference(left: Any, right: Any, *, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: type differs ({type(left).__name__} != {type(right).__name__})"
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return f"{path}: keys differ (left_only={sorted(left_keys - right_keys)}, right_only={sorted(right_keys - left_keys)})"
        for key in sorted(left_keys):
            difference = _first_difference(left[key], right[key], path=f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length differs ({len(left)} != {len(right)})"
        for index, (left_child, right_child) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_child, right_child, path=f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return f"{path}: value differs ({left!r} != {right!r})"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="Baseline results.json path.")
    parser.add_argument("right", type=Path, help="New results.json path.")
    parser.add_argument(
        "--left-output-root",
        type=Path,
        default=None,
        help="Output root to normalize in the baseline. Defaults to left parent.",
    )
    parser.add_argument(
        "--right-output-root",
        type=Path,
        default=None,
        help="Output root to normalize in the new result. Defaults to right parent.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left_root = (args.left_output_root or args.left.parent).resolve()
    right_root = (args.right_output_root or args.right.parent).resolve()
    left = _normalize(_load_json(args.left), output_root=left_root)
    right = _normalize(_load_json(args.right), output_root=right_root)
    difference = _first_difference(left, right)
    if difference is not None:
        print(f"Benchmark outcomes differ: {difference}", file=sys.stderr)
        return 1
    print("Benchmark outcomes match after normalizing provenance, output paths, cache-use metadata, jobs, and timings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
