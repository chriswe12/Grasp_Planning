"""Generate assembly-feasible grasp candidates for a Fabrica part and save them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    DEFAULT_CONTACT_APPROACH_OFFSETS_M,
    DEFAULT_CONTACT_LATERAL_OFFSETS_M,
    relative_asset_mesh_path,
)
from grasp_planning.pipeline import (  # noqa: E402
    GeometryConfig,
    PlanningConfig,
    generate_stage1_result,
    write_stage1_artifacts,
)


def _parse_rolls(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one roll angle.")
    return values


def _parse_offsets(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one offset value.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate assembly-feasible Fabrica grasps for an OBJ/STL part and save them."
    )
    parser.add_argument(
        "--mesh-path",
        "--stl-path",
        dest="mesh_path",
        type=Path,
        required=True,
        help="Target OBJ/STL path, relative to assets or absolute.",
    )
    parser.add_argument("--assembly-glob", type=str, required=True, help="Sibling assembly glob under assets/.")
    parser.add_argument(
        "--mesh-scale", "--stl-scale", dest="mesh_scale", type=float, default=0.001, help="Uniform mesh scale."
    )
    parser.add_argument("--num-samples", type=int, default=1024, help="Number of surface samples.")
    parser.add_argument("--min-jaw-width", type=float, default=0.002, help="Minimum jaw width in meters.")
    parser.add_argument("--max-jaw-width", type=float, default=0.09, help="Maximum jaw width in meters.")
    parser.add_argument(
        "--antipodal-cosine-threshold",
        type=float,
        default=0.984807753012208,
        help="Minimum antipodal cosine alignment.",
    )
    parser.add_argument(
        "--roll-angles-rad", type=_parse_rolls, default=(0.0,), help="Comma-separated roll angles in radians."
    )
    parser.add_argument("--max-pair-checks", type=int, default=40960, help="Maximum nearby sample pairs to evaluate.")
    parser.add_argument(
        "--detailed-finger-contact-gap-m", type=float, default=0.002, help="Detailed Franka finger contact gap."
    )
    parser.add_argument(
        "--contact-lateral-offsets-m",
        type=_parse_offsets,
        default=DEFAULT_CONTACT_LATERAL_OFFSETS_M,
        help="Comma-separated pad-local lateral contact offsets to try when center contact collides.",
    )
    parser.add_argument(
        "--contact-approach-offsets-m",
        type=_parse_offsets,
        default=DEFAULT_CONTACT_APPROACH_OFFSETS_M,
        help="Comma-separated pad-local approach offsets to try when center contact collides.",
    )
    parser.add_argument("--rng-seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--output-html", type=Path, required=True, help="Output HTML path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    planning = PlanningConfig(
        num_surface_samples=args.num_samples,
        min_jaw_width=args.min_jaw_width,
        max_jaw_width=args.max_jaw_width,
        antipodal_cosine_threshold=args.antipodal_cosine_threshold,
        roll_angles_rad=args.roll_angles_rad,
        max_pair_checks=args.max_pair_checks,
        detailed_finger_contact_gap_m=args.detailed_finger_contact_gap_m,
        contact_lateral_offsets_m=args.contact_lateral_offsets_m,
        contact_approach_offsets_m=args.contact_approach_offsets_m,
        rng_seed=args.rng_seed,
    )
    geometry = GeometryConfig(
        target_mesh_path=str(args.mesh_path),
        mesh_scale=args.mesh_scale,
        assembly_glob=args.assembly_glob,
    )
    stage1 = generate_stage1_result(geometry=geometry, planning=planning)
    write_stage1_artifacts(
        stage1,
        geometry=geometry,
        planning=planning,
        output_json=args.output_json,
        output_html=args.output_html,
    )

    print(
        "[INFO] Assembly grasp export: "
        f"kept {len(stage1.bundle.candidates)} / {stage1.raw_candidate_count} candidates "
        f"for target '{relative_asset_mesh_path(args.mesh_path)}'.",
        flush=True,
    )
    print(f"[INFO] Wrote grasp JSON to: {args.output_json.resolve()}", flush=True)
    print(f"[INFO] Wrote HTML debug view to: {args.output_html.resolve()}", flush=True)


if __name__ == "__main__":
    main()
