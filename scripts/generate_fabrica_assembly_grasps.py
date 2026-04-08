"""Generate assembly-feasible grasp candidates for a Fabrica part and save them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping import AntipodalGraspGeneratorConfig, AntipodalMeshGraspGenerator  # noqa: E402
from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    CandidateStatus,
    DEFAULT_CONTACT_APPROACH_OFFSETS_M,
    DEFAULT_CONTACT_LATERAL_OFFSETS_M,
    SavedGraspBundle,
    canonicalize_target_mesh,
    filter_grasps_against_assembly,
    load_assembly_obstacle_mesh,
    load_stl_mesh,
    relative_stl_path,
    save_grasp_bundle,
    serialize_saved_candidate,
    shifted_mesh,
    write_debug_html,
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
    parser = argparse.ArgumentParser(description="Generate assembly-feasible Fabrica grasps and save them to JSON.")
    parser.add_argument("--stl-path", type=Path, required=True, help="Target STL path, relative to assets/stl or absolute.")
    parser.add_argument("--assembly-glob", type=str, required=True, help="Sibling assembly glob under assets/stl.")
    parser.add_argument("--stl-scale", type=float, default=0.001, help="Uniform STL scale.")
    parser.add_argument("--num-samples", type=int, default=1024, help="Number of surface samples.")
    parser.add_argument("--min-jaw-width", type=float, default=0.002, help="Minimum jaw width in meters.")
    parser.add_argument("--max-jaw-width", type=float, default=0.09, help="Maximum jaw width in meters.")
    parser.add_argument("--antipodal-cosine-threshold", type=float, default=0.984807753012208, help="Minimum antipodal cosine alignment.")
    parser.add_argument("--roll-angles-rad", type=_parse_rolls, default=(0.0,), help="Comma-separated roll angles in radians.")
    parser.add_argument("--max-pair-checks", type=int, default=40960, help="Maximum nearby sample pairs to evaluate.")
    parser.add_argument("--detailed-finger-contact-gap-m", type=float, default=0.002, help="Detailed Franka finger contact gap.")
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
    config = AntipodalGraspGeneratorConfig(
        num_surface_samples=args.num_samples,
        min_jaw_width=args.min_jaw_width,
        max_jaw_width=args.max_jaw_width,
        antipodal_cosine_threshold=args.antipodal_cosine_threshold,
        roll_angles_rad=args.roll_angles_rad,
        max_pair_checks=args.max_pair_checks,
        detailed_finger_contact_gap_m=args.detailed_finger_contact_gap_m,
        rng_seed=args.rng_seed,
    )
    target_mesh_global = load_stl_mesh(args.stl_path, scale=args.stl_scale)
    target_mesh_local, target_pose_world = canonicalize_target_mesh(target_mesh_global)
    generator = AntipodalMeshGraspGenerator(config)
    raw_candidates = generator.generate(target_mesh_local)
    serialized_raw = [serialize_saved_candidate(f"g{index:04d}", candidate) for index, candidate in enumerate(raw_candidates, start=1)]

    obstacle_mesh_world, obstacle_paths = load_assembly_obstacle_mesh(
        assembly_glob=args.assembly_glob,
        target_stl_path=args.stl_path,
        stl_scale=args.stl_scale,
    )
    kept_candidates = filter_grasps_against_assembly(
        serialized_raw,
        object_pose_world=target_pose_world,
        obstacle_mesh_world=obstacle_mesh_world,
        contact_gap_m=args.detailed_finger_contact_gap_m,
        contact_lateral_offsets_m=args.contact_lateral_offsets_m,
        contact_approach_offsets_m=args.contact_approach_offsets_m,
    )

    bundle = SavedGraspBundle(
        target_stl_path=relative_stl_path(args.stl_path),
        stl_scale=args.stl_scale,
        local_frame_origin_world=target_pose_world.position_world,
        local_frame_orientation_xyzw_world=target_pose_world.orientation_xyzw_world,
        candidates=tuple(kept_candidates),
        metadata={
            "assembly_glob": args.assembly_glob,
            "collision_backend": generator.collision_backend_name,
            "num_surface_samples": args.num_samples,
            "raw_candidate_count": len(serialized_raw),
            "assembly_feasible_count": len(kept_candidates),
            "assembly_obstacle_paths": list(obstacle_paths),
            "contact_lateral_offsets_m": list(args.contact_lateral_offsets_m),
            "contact_approach_offsets_m": list(args.contact_approach_offsets_m),
        },
    )
    save_grasp_bundle(bundle, args.output_json)

    obstacle_mesh_local = None
    if obstacle_mesh_world is not None:
        obstacle_mesh_local = shifted_mesh(obstacle_mesh_world, -target_pose_world.translation_world)
    write_debug_html(
        title="Fabrica Assembly-Feasible Grasps",
        subtitle="Offline assembly collision screening. Candidates are stored and visualized in the target part-local frame.",
        mesh_local=target_mesh_local,
        candidate_statuses=[CandidateStatus(grasp=candidate, status="accepted", reason="assembly_clear") for candidate in kept_candidates],
        output_html=args.output_html,
        contact_gap_m=args.detailed_finger_contact_gap_m,
        obstacle_mesh_local=obstacle_mesh_local,
        metadata_lines=[
            f"target_stl:       {relative_stl_path(args.stl_path)}",
            f"assembly_glob:    {args.assembly_glob}",
            f"collision_backend:{generator.collision_backend_name}",
            f"raw_candidates:   {len(serialized_raw)}",
            f"assembly_feasible:{len(kept_candidates)}",
            f"contact_offsets_x:{tuple(args.contact_lateral_offsets_m)}",
            f"contact_offsets_z:{tuple(args.contact_approach_offsets_m)}",
            f"local_origin_w:   {tuple(round(v, 6) for v in target_pose_world.position_world)}",
        ],
    )

    print(
        "[INFO] Assembly grasp export: "
        f"kept {len(kept_candidates)} / {len(serialized_raw)} candidates "
        f"for target '{relative_stl_path(args.stl_path)}' against {len(obstacle_paths)} obstacle meshes.",
        flush=True,
    )
    print(f"[INFO] Wrote grasp JSON to: {args.output_json.resolve()}", flush=True)
    print(f"[INFO] Wrote HTML debug view to: {args.output_html.resolve()}", flush=True)


if __name__ == "__main__":
    main()
