#!/usr/bin/env python3
"""Build Stage-3 holder/inserter end-effector pair artifacts."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.pipeline import (  # noqa: E402
    DualGraspPairConfig,
    compile_assembly_sequence,
    evaluate_holder_state_feasibility,
    generate_holder_grasp_library,
    generate_inserter_grasp_libraries,
    inserter_artifact_name,
    pair_artifact_name,
    plan_dual_grasp_pairs,
    write_assembly_sequence_html,
    write_assembly_sequence_json,
    write_dual_grasp_pair_debug_artifacts,
    write_dual_grasp_pair_step_json,
    write_dual_grasp_pair_summary_json,
    write_holder_grasp_library_artifacts,
    write_holder_state_feasibility_json,
    write_inserter_grasp_library,
)
from scripts.build_holder_grasp_library import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    _load_config,
    _repo_path,
)
from scripts.build_holder_state_feasibility import (  # noqa: E402
    _holder_feasibility_config,
)
from scripts.run_grasp_pipeline import _planning_config  # noqa: E402


def _pair_config(payload: dict[str, object]) -> DualGraspPairConfig:
    raw = dict(payload.get("pair_planning", {}))
    return DualGraspPairConfig(
        max_holder_candidates_per_step=int(raw.get("max_holder_candidates_per_step", 80)),
        max_inserter_candidates_per_step=int(raw.get("max_inserter_candidates_per_step", 512)),
        max_candidates_per_cluster=int(raw.get("max_candidates_per_cluster", 2)),
        contact_position_bin_m=float(raw.get("contact_position_bin_m", 0.025)),
        axis_bin_deg=float(raw.get("axis_bin_deg", 30.0)),
        max_pair_checks=int(raw.get("max_pair_checks", 4000)),
        max_accepted_pairs=int(raw.get("max_accepted_pairs", 256)),
        max_rejected_pairs=int(raw.get("max_rejected_pairs", 200)),
        max_collision_diagnostics_per_step=int(raw.get("max_collision_diagnostics_per_step", 24)),
        max_pairs_per_holder=int(raw.get("max_pairs_per_holder", 4)),
        max_pairs_per_inserter=int(raw.get("max_pairs_per_inserter", 4)),
        matrix_unary_rejections_per_side=int(raw.get("matrix_unary_rejections_per_side", 12)),
        table_clearance_margin_m=float(raw.get("table_clearance_margin_m", 0.002)),
        geometry_clearance_margin_m=float(raw.get("geometry_clearance_margin_m", 0.0)),
        retreat_distance_m=float(raw.get("retreat_distance_m", 0.05)),
        path_samples=int(raw.get("path_samples", 21)),
        holder_score_weight=float(raw.get("holder_score_weight", 0.45)),
        inserter_score_weight=float(raw.get("inserter_score_weight", 0.45)),
        clearance_score_weight=float(raw.get("clearance_score_weight", 0.10)),
        clearance_score_saturation_m=float(raw.get("clearance_score_saturation_m", 0.05)),
        transition_symmetry_enabled=bool(raw.get("transition_symmetry_enabled", False)),
        transition_symmetry_asset_path=str(raw.get("transition_symmetry_asset_path", "")),
        transition_symmetry_geometry_tolerance_m=float(raw.get("transition_symmetry_geometry_tolerance_m", 0.001)),
        transition_symmetry_max_partial_assembly_transforms=int(
            raw.get(
                "transition_symmetry_max_partial_assembly_transforms",
                0,
            )
        ),
        transition_symmetry_max_incoming_transforms=int(raw.get("transition_symmetry_max_incoming_transforms", 0)),
        adaptive_inserter_shortlist=bool(raw.get("adaptive_inserter_shortlist", True)),
        prefer_aabb_clear_inserter_candidates=bool(
            raw.get("prefer_aabb_clear_inserter_candidates", True)
        ),
        balance_inserter_approach_directions=bool(
            raw.get("balance_inserter_approach_directions", True)
        ),
        balance_inserter_symmetry_transforms=bool(
            raw.get("balance_inserter_symmetry_transforms", True)
        ),
        exact_pair_clearance_ranking=bool(raw.get("exact_pair_clearance_ranking", True)),
        stage3_worker_count=int(raw.get("stage3_worker_count", 1)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--assembly", help="Override assembly.name from YAML.")
    parser.add_argument(
        "--base-part-id",
        help="Override the default holder base, forward_assembly_orders[0][0].",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the assembly artifact directory.",
    )
    parser.add_argument(
        "--max-inserter-candidates-per-step",
        type=int,
        help="Override the Stage-3 unary inserter cap for this build.",
    )
    parser.add_argument(
        "--max-pair-checks",
        type=int,
        help="Override the per-step holder/inserter pair-check budget.",
    )
    parser.add_argument(
        "--skip-exact-pair-clearance-ranking",
        action="store_true",
        help=(
            "Keep exact collision rejection but skip exact distance queries "
            "when no positive pair-clearance margin is configured."
        ),
    )
    args = parser.parse_args(argv)

    payload = _load_config(args.config.expanduser().resolve())
    assembly_raw = dict(payload.get("assembly", {}))
    artifacts_raw = dict(payload.get("artifacts", {}))
    assembly_name = str(args.assembly or assembly_raw["name"])
    configured_base = assembly_raw.get("base_part_id")
    base_part_id = args.base_part_id if args.base_part_id is not None else configured_base
    sequence = compile_assembly_sequence(
        _repo_path(assembly_raw.get("asset_root", "assets/obj/fabrica")) / assembly_name,
        base_part_id=base_part_id,
        mesh_scale=float(assembly_raw.get("mesh_scale", 0.01)),
        table_z_assembly_m=float(assembly_raw.get("table_z_assembly_m", 0.0)),
        table_contact_tolerance_m=float(assembly_raw.get("table_contact_tolerance_m", 1.0e-6)),
    )
    planning = _planning_config(payload)
    planning = replace(
        planning,
        stage1_cache_dir=str(_repo_path(planning.stage1_cache_dir)),
    )
    pair_config = _pair_config(payload)
    if args.max_inserter_candidates_per_step is not None:
        pair_config = replace(
            pair_config,
            max_inserter_candidates_per_step=args.max_inserter_candidates_per_step,
        )
    if args.max_pair_checks is not None:
        pair_config = replace(pair_config, max_pair_checks=args.max_pair_checks)
    if args.skip_exact_pair_clearance_ranking:
        pair_config = replace(pair_config, exact_pair_clearance_ranking=False)

    print(
        f"[Stage 1] loading/generating holder library for base {sequence.base_part_id}",
        flush=True,
    )
    holder_library = generate_holder_grasp_library(
        sequence=sequence,
        planning=planning,
    )
    print(
        f"[Stage 2] evaluating {len(holder_library.bundle.candidates)} holder candidates against assembly states",
        flush=True,
    )
    holder_feasibility = evaluate_holder_state_feasibility(
        sequence=sequence,
        holder_library=holder_library,
        planning=planning,
        config=_holder_feasibility_config(payload),
    )
    print(
        "[Stage 3] generating insertion-filtered grasp libraries "
        f"(workers={'auto' if pair_config.stage3_worker_count == 0 else pair_config.stage3_worker_count}, "
        f"adaptive_shortlist={pair_config.adaptive_inserter_shortlist})",
        flush=True,
    )
    inserter_started = time.monotonic()
    inserter_libraries = generate_inserter_grasp_libraries(
        sequence=sequence,
        planning=planning,
        config=pair_config,
    )
    inserter_elapsed = time.monotonic() - inserter_started
    for library in inserter_libraries:
        print(
            f"  {library.step_id}: {library.raw_candidate_count} raw, "
            f"{library.assembly_insertion_feasible_count} assembly/insertion "
            f"feasible, {len(library.accepted_candidates)} table/retreat feasible, "
            "approaches="
            f"{library.bundle.metadata.get('accepted_approach_direction_counts_assembly', {})}",
            flush=True,
        )
    print(
        f"[Stage 3] insertion filtering completed in {inserter_elapsed:.2f}s",
        flush=True,
    )
    print("[Stage 3] pairing shortlisted end effectors", flush=True)
    pairing_started = time.monotonic()
    result = plan_dual_grasp_pairs(
        sequence=sequence,
        holder_feasibility=holder_feasibility,
        inserter_libraries=inserter_libraries,
        planning=planning,
        config=pair_config,
    )
    pairing_elapsed = time.monotonic() - pairing_started
    print(
        f"[Stage 3] pair/transition checks completed in {pairing_elapsed:.2f}s",
        flush=True,
    )

    if args.output_dir is None:
        output_dir = (
            _repo_path(
                artifacts_raw.get(
                    "output_root",
                    "artifacts/dual_grasp_planning",
                )
            )
            / assembly_name
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_assembly_sequence_json(sequence, output_dir / "assembly_sequence.json")
    write_assembly_sequence_html(sequence, output_dir / "assembly_sequence.html")
    write_holder_grasp_library_artifacts(
        holder_library,
        sequence=sequence,
        planning=planning,
        output_json=output_dir / "holder_base_candidates.json",
        output_html=output_dir / "holder_base_candidates.html",
    )
    holder_source = output_dir / "holder_state_feasibility.json"
    write_holder_state_feasibility_json(
        holder_feasibility,
        holder_source,
    )
    for library in inserter_libraries:
        write_inserter_grasp_library(
            library,
            output_dir / inserter_artifact_name(library),
        )
    for step in result.steps:
        write_dual_grasp_pair_step_json(
            step,
            output_dir / pair_artifact_name(step),
            holder_source_artifact=holder_source.name,
            inserter_source_artifact=inserter_artifact_name(result.inserter_libraries_by_step[step.step_id]),
        )
    summary_json = output_dir / "dual_grasp_pair_summary.json"
    write_dual_grasp_pair_summary_json(result, summary_json)
    summary_html, step_htmls = write_dual_grasp_pair_debug_artifacts(
        result,
        sequence,
        output_dir,
    )

    print(f"assembly: {result.assembly}", flush=True)
    for step in result.steps:
        print(
            f"{step.step_id}: "
            f"{step.metadata['holder_shortlist_count']} holders x "
            f"{step.metadata['inserter_shortlist_count']} inserters, "
            f"{step.metadata['checked_pair_count']} checked, "
            f"{step.metadata['compatible_pair_count']} compatible, "
            f"{step.metadata['retained_pair_count']} retained",
            flush=True,
        )
    print(f"summary json: {summary_json}", flush=True)
    print(f"summary html: {summary_html}", flush=True)
    print(f"step html files: {len(step_htmls)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
