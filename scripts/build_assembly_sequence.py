#!/usr/bin/env python3
"""Compile and visualize the selected Fabrica assembly sequence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.pipeline.assembly_sequence import (  # noqa: E402
    DEFAULT_ASSET_ROOT,
    compile_assembly_sequence,
    write_assembly_sequence_json,
)
from grasp_planning.pipeline.assembly_sequence_debug_html import (  # noqa: E402
    DEFAULT_MAX_EDGES_PER_PART,
    DEFAULT_MAX_FACES_PER_PART,
    write_assembly_sequence_html,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile forward_assembly_orders[0] into explicit JSON states and an interactive HTML viewer."
    )
    parser.add_argument("--assembly", required=True, help="Fabrica assembly directory name.")
    parser.add_argument(
        "--base-part-id",
        help="Override the default holder base, forward_assembly_orders[0][0].",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
        help=f"Root containing Fabrica assembly directories (default: {DEFAULT_ASSET_ROOT}).",
    )
    parser.add_argument("--mesh-scale", type=float, default=0.01, help="Scale applied to OBJ coordinates.")
    parser.add_argument(
        "--table-z",
        type=float,
        default=0.0,
        help="Table-plane Z in scaled assembly-asset coordinates.",
    )
    parser.add_argument(
        "--table-contact-tolerance",
        type=float,
        default=1.0e-6,
        help="Absolute distance used to classify a mesh as touching the table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory (default: artifacts/dual_grasp_planning/<assembly>).",
    )
    parser.add_argument(
        "--max-edges-per-part",
        type=int,
        default=DEFAULT_MAX_EDGES_PER_PART,
        help="Maximum mesh edges embedded per part in the HTML viewer; <=0 keeps every edge.",
    )
    parser.add_argument(
        "--max-faces-per-part",
        type=int,
        default=DEFAULT_MAX_FACES_PER_PART,
        help="Maximum mesh faces embedded per part in the HTML viewer; <=0 keeps every face.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    asset_root = args.asset_root.expanduser()
    if not asset_root.is_absolute():
        asset_root = (REPO_ROOT / asset_root).resolve()
    assembly_dir = asset_root / str(args.assembly)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "artifacts" / "dual_grasp_planning" / str(args.assembly)
    elif not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    sequence = compile_assembly_sequence(
        assembly_dir,
        base_part_id=args.base_part_id,
        mesh_scale=float(args.mesh_scale),
        table_z_assembly_m=float(args.table_z),
        table_contact_tolerance_m=float(args.table_contact_tolerance),
        repo_root=REPO_ROOT,
    )
    json_path = output_dir / "assembly_sequence.json"
    html_path = output_dir / "assembly_sequence.html"
    write_assembly_sequence_json(sequence, json_path)
    write_assembly_sequence_html(
        sequence,
        html_path,
        max_edges_per_part=int(args.max_edges_per_part),
        max_faces_per_part=int(args.max_faces_per_part),
    )

    print(f"Assembly:            {sequence.assembly}")
    print(f"Selected order:       {' -> '.join(sequence.selected_order)}")
    print(f"Base part:            {sequence.base_part_id}")
    print(f"Base selection:       {sequence.base_part_source}")
    print(f"Base order index:      {sequence.base_part_order_index}")
    print(f"First holder step:     {sequence.first_holder_step_index}")
    print(f"Table contact parts:   {list(sequence.table_contact_part_ids)}")
    print(f"JSON artifact:         {json_path}")
    print(f"HTML artifact:         {html_path}")
    if sequence.warnings:
        print("Warnings:")
        for warning in sequence.warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
