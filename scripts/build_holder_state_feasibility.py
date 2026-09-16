#!/usr/bin/env python3
"""Build Stage-2 per-state holder feasibility artifacts."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.pipeline import (  # noqa: E402
    HolderFeasibilityConfig,
    compile_assembly_sequence,
    evaluate_holder_state_feasibility,
    generate_holder_grasp_library,
    write_holder_state_debug_artifacts,
    write_holder_state_feasibility_json,
)
from scripts.build_holder_grasp_library import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    _load_config,
    _repo_path,
)
from scripts.run_grasp_pipeline import _planning_config  # noqa: E402


def _holder_feasibility_config(payload: dict[str, object]) -> HolderFeasibilityConfig:
    raw = dict(payload.get("holder_feasibility", {}))
    return HolderFeasibilityConfig(
        pregrasp_offset_m=float(raw.get("pregrasp_offset_m", 0.05)),
        table_clearance_margin_m=float(raw.get("table_clearance_margin_m", 0.002)),
        geometry_clearance_margin_m=float(raw.get("geometry_clearance_margin_m", 0.0)),
        incoming_path_samples=int(raw.get("incoming_path_samples", 21)),
    )


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
    holder_library = generate_holder_grasp_library(sequence=sequence, planning=planning)
    feasibility = evaluate_holder_state_feasibility(
        sequence=sequence,
        holder_library=holder_library,
        planning=planning,
        config=_holder_feasibility_config(payload),
    )

    if args.output_dir is None:
        output_root = _repo_path(artifacts_raw.get("output_root", "artifacts/dual_grasp_planning"))
        output_dir = output_root / assembly_name
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_json = output_dir / "holder_state_feasibility.json"
    write_holder_state_feasibility_json(feasibility, output_json)
    matrix_html, state_htmls = write_holder_state_debug_artifacts(
        feasibility,
        sequence,
        output_dir,
    )

    print(f"assembly: {sequence.assembly}")
    print(f"base part: {sequence.base_part_id} ({sequence.base_part_source})")
    print(f"holder candidates: {len(feasibility.candidates)}")
    for state in feasibility.states:
        print(
            f"step {state.step_index} incoming {state.incoming_part_id}: "
            f"{len(state.accepted_grasp_ids)} accepted {state.reason_counts}"
        )
    print(f"json: {output_json}")
    print(f"matrix html: {matrix_html}")
    print(f"state html files: {len(state_htmls)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
