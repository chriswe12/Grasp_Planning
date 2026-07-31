#!/usr/bin/env python3
"""Build the reusable Stage-1 holder-grasp library for one Fabrica base part."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.pipeline import (  # noqa: E402
    compile_assembly_sequence,
    generate_holder_grasp_library,
    write_holder_grasp_library_artifacts,
)
from scripts.run_grasp_pipeline import _planning_config  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dual_grasp_planning.yaml"


def _load_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level YAML mapping in '{path}'.")
    return payload


def _repo_path(raw_path: object) -> Path:
    path = Path(str(raw_path)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--assembly", help="Override assembly.name from YAML.")
    parser.add_argument(
        "--base-part-id",
        help="Override the default holder base, forward_assembly_orders[0][0].",
    )
    parser.add_argument("--output-dir", type=Path, help="Override the assembly artifact directory.")
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    payload = _load_config(config_path)
    assembly_raw = dict(payload.get("assembly", {}))
    artifacts_raw = dict(payload.get("artifacts", {}))

    assembly_name = str(args.assembly or assembly_raw["name"])
    configured_base_part_id = assembly_raw.get("base_part_id")
    base_part_id = args.base_part_id if args.base_part_id is not None else configured_base_part_id
    asset_root = _repo_path(assembly_raw.get("asset_root", "assets/obj/fabrica"))
    sequence = compile_assembly_sequence(
        asset_root / assembly_name,
        base_part_id=base_part_id,
        mesh_scale=float(assembly_raw.get("mesh_scale", 0.01)),
        table_z_assembly_m=float(assembly_raw.get("table_z_assembly_m", 0.0)),
        table_contact_tolerance_m=float(assembly_raw.get("table_contact_tolerance_m", 1.0e-6)),
    )

    planning = _planning_config(payload)
    planning = replace(planning, stage1_cache_dir=str(_repo_path(planning.stage1_cache_dir)))
    result = generate_holder_grasp_library(sequence=sequence, planning=planning)

    if args.output_dir is None:
        output_root = _repo_path(artifacts_raw.get("output_root", "artifacts/dual_grasp_planning"))
        output_dir = output_root / assembly_name
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_json = output_dir / "holder_base_candidates.json"
    output_html = output_dir / "holder_base_candidates.html"
    write_holder_grasp_library_artifacts(
        result,
        sequence=sequence,
        planning=planning,
        output_json=output_json,
        output_html=output_html,
    )

    print(f"assembly: {sequence.assembly}")
    print(f"base part: {sequence.base_part_id}")
    print(f"base selection: {sequence.base_part_source}")
    print(f"raw holder candidates: {result.raw_candidate_count}")
    print(f"saved holder candidates: {len(result.bundle.candidates)}")
    print(f"base source origin in assembly frame: {result.target_pose_in_obj_world.position_world}")
    print(f"json: {output_json}")
    print(f"html: {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
