#!/usr/bin/env python3
"""Build Stage-3 holder/inserter end-effector pair artifacts."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import CandidateStatus  # noqa: E402
from grasp_planning.grasping.mesh_io import load_triangle_mesh  # noqa: E402
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
    write_holder_state_debug_artifacts,
    write_holder_state_feasibility_json,
    write_inserter_grasp_library,
)
from grasp_planning.pipeline.fabrica_pipeline import _mesh_in_source_frame  # noqa: E402
from grasp_planning.pipeline.holder_grasp_debug_html import (  # noqa: E402
    write_holder_grasp_debug_html,
)
from grasp_planning.pipeline.inserter_unary_debug_html import (  # noqa: E402
    write_inserter_unary_debug_html,
)
from scripts.build_holder_grasp_library import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    _load_config,
    _repo_path,
)
from scripts.build_holder_state_feasibility import (  # noqa: E402
    _holder_feasibility_config,
)
from scripts.run_grasp_generation_benchmark import (  # noqa: E402
    _write_all_generated_grasps_overview_html,
)
from scripts.run_grasp_pipeline import _planning_config  # noqa: E402


def _write_index_html(*, output_dir: Path, result, summary_html: Path, step_htmls: tuple[Path, ...]) -> Path:
    """Write the stable offline-build landing page expected by benchmark users."""

    rows = []
    for step in result.steps:
        metadata = step.metadata
        rows.append(
            "<tr>"
            f"<td>{html.escape(step.step_id)}</td>"
            f"<td>{int(metadata['holder_shortlist_count'])}</td>"
            f"<td>{int(metadata['inserter_shortlist_count'])}</td>"
            f"<td>{int(metadata['compatible_pair_count'])}</td>"
            f"<td>{int(metadata['retained_pair_count'])}</td>"
            f"<td><a href=\"{html.escape(pair_artifact_name(step).replace('.json', '.html'))}\">pairs</a> · "
            f"<a href=\"inserter_population_{html.escape(step.step_id)}.html\">all unary grasps</a> · "
            f"<a href=\"inserter_unary_debug_{html.escape(step.step_id)}.html\">constraint debugger</a> · "
            f"<a href=\"inserter_failures_{html.escape(step.step_id)}.html\">unary failures</a></td>"
            "</tr>"
        )
    page = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>PDZ offline grasp build</title>
<style>body{{font:15px system-ui;margin:3rem;max-width:900px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}}a{{color:#0759b5}}</style></head>
<body><h1>{html.escape(result.assembly)} offline grasp build</h1>
<p>Three-stage PDZ collision planning: Stage 1 grasps, Stage 2 holder-state feasibility, Stage 3 insertion and pair checks.</p>
<p><a href=\"{summary_html.name}\">Stage 3 summary</a> · <a href=\"holder_base_candidates.html\">Stage 1 holder grasps</a> · <a href=\"holder_validity_matrix.html\">Stage 2 holder-state debugger</a></p>
<p><small>\"holders\" is the Stage-2 holder-state pass count only. \"inserters\" is the Stage-3 unary insertion pass count; zero inserters means that assembly step cannot proceed.</small></p>
<table><thead><tr><th>Assembly step</th><th>holders<br><small>Stage 2</small></th><th>inserters<br><small>Stage 3 unary</small></th><th>compatible</th><th>retained</th><th></th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    path = output_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def _failure_sort_key(status) -> tuple[float, float, str]:
    """Show the least-penetrating failures first, then the best grasp score."""

    clearance = status.minimum_clearance_m
    penetration = float("inf") if clearance is None else max(0.0, -float(clearance))
    score = float(status.candidate.score) if status.candidate.score is not None else float("-inf")
    return (penetration, -score, status.grasp_id)


def _diverse_failure_statuses(library, *, per_reason: int = 60):
    """Keep an inspectable, spatially diverse slice of rejected unary grasps."""

    selected = []
    by_reason = {}
    for status in library.candidate_statuses:
        if status.status == "rejected":
            by_reason.setdefault(status.reason, []).append(status)
    for reason in sorted(by_reason):
        used_bins = set()
        for status in sorted(by_reason[reason], key=_failure_sort_key):
            candidate = status.candidate
            # Preserve contact-normal families as well as position/roll.  A
            # single rounded or highly scored region must not hide valid flat
            # face grasp families in the failure debugger.
            cell = tuple(int(np.floor(float(value) / 0.025)) for value in candidate.grasp_position_obj)
            def normal_family(normal) -> tuple[int, int]:
                value = np.asarray(normal, dtype=float)
                axis = int(np.argmax(np.abs(value)))
                return axis, 1 if value[axis] >= 0.0 else -1
            contacts = tuple(sorted((normal_family(candidate.contact_normal_a_obj), normal_family(candidate.contact_normal_b_obj))))
            key = (cell, int(np.floor(float(candidate.roll_angle_rad) / (np.pi / 6.0))), contacts)
            if key in used_bins:
                continue
            used_bins.add(key)
            selected.append(status)
            if sum(item.reason == reason for item in selected) >= per_reason:
                break
    return tuple(sorted(selected, key=lambda status: (status.reason, *_failure_sort_key(status))))


def _failure_candidate_payload(status, rank: int) -> dict[str, object]:
    candidate = status.candidate
    return {
        "rank": rank,
        "grasp_id": candidate.grasp_id,
        "position": list(candidate.grasp_position_obj),
        "orientation_xyzw": list(candidate.grasp_orientation_xyzw_obj),
        "contact_a": list(candidate.contact_point_a_obj),
        "contact_b": list(candidate.contact_point_b_obj),
        "jaw_width": candidate.jaw_width,
        "roll_angle_rad": candidate.roll_angle_rad,
        "score": candidate.score,
        "reason": status.reason,
        "minimum_clearance_m": status.minimum_clearance_m,
        "penetration_m": None if status.minimum_clearance_m is None else max(0.0, -status.minimum_clearance_m),
        "details": status.details,
    }


def _write_inserter_failure_debug(*, library, sequence, planning, output_dir: Path) -> tuple[Path, Path]:
    """Persist the failed unary grasps that normal bundle serialization omits."""

    failures = _diverse_failure_statuses(library)
    json_path = output_dir / f"inserter_failures_{library.step_id}.json"
    json_path.write_text(
        json.dumps(
            {
                "kind": "inserter_unary_failure_debug",
                "step_id": library.step_id,
                "incoming_part_id": library.incoming_part_id,
                "selection": {
                    "per_reason_limit": 60,
                    "ranking": "least penetration first, then grasp score",
                    "diversity": "one grasp per 25 mm grasp-center cell, 30 degree roll bin, and contact-normal family per reason",
                },
                "reason_counts": library.reason_counts,
                "failures": [_failure_candidate_payload(status, rank) for rank, status in enumerate(failures, start=1)],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    # The existing grasp viewer provides a quick visual inspection of the
    # candidate and the exact PDZ collision hulls.  Keep the unary failure
    # information inside the candidate metadata shown in its details panel.
    visual_candidates = tuple(
        replace(
            status.candidate,
            metadata={
                **(status.candidate.metadata or {}),
                "unary_failure": {
                    "reason": status.reason,
                    "minimum_clearance_m": status.minimum_clearance_m,
                    "penetration_m": None if status.minimum_clearance_m is None else max(0.0, -status.minimum_clearance_m),
                    "details": status.details,
                },
            },
        )
        for status in failures
    )
    source = library.source_frame_pose_assembly
    # Candidates are in Stage-1's canonical source frame, not the raw OBJ
    # frame.  Render that same mesh frame so sampled contacts lie on the mesh.
    mesh = _mesh_in_source_frame(
        load_triangle_mesh(library.bundle.target_mesh_path, scale=library.bundle.mesh_scale),
        source,
    )
    table_corners_world = np.asarray(
        [[-0.5, -0.5, sequence.table_z_assembly_m], [-0.5, 0.5, sequence.table_z_assembly_m], [0.5, 0.5, sequence.table_z_assembly_m], [0.5, -0.5, sequence.table_z_assembly_m]],
        dtype=float,
    )
    table_corners_local = (source.rotation_world_from_object.T @ (table_corners_world - source.translation_world).T).T.tolist()
    html_path = output_dir / f"inserter_failures_{library.step_id}.html"
    write_holder_grasp_debug_html(
        title=f"Inserter unary failures: {library.step_id}",
        subtitle="Diverse rejected candidates; table is shown in the incoming-part source frame.",
        mesh_local=mesh,
        candidates=visual_candidates,
        output_html=html_path,
        metadata_lines=[
            f"raw candidates: {library.raw_candidate_count}",
            f"rejected candidates shown: {len(visual_candidates)}",
            "ranked by least known penetration; one candidate per spatial/roll bin",
        ],
        table_plane_local=table_corners_local,
        gripper_collision_model=planning.gripper_collision_model,
    )
    return json_path, html_path


def _write_inserter_population_debug(*, library, sequence, output_dir: Path) -> Path:
    """Write the all-grasp marker viewer with pass/fail and percentage controls."""

    source = library.source_frame_pose_assembly
    mesh = _mesh_in_source_frame(
        load_triangle_mesh(library.bundle.target_mesh_path, scale=library.bundle.mesh_scale), source
    )
    statuses = [
        CandidateStatus(
            grasp=status.candidate,
            status=("accepted" if status.status == "accepted" else "rejected"),
            reason=status.reason,
        )
        for status in library.candidate_statuses
        if status.status != "not_evaluated"
    ]
    path = output_dir / f"inserter_population_{library.step_id}.html"
    _write_all_generated_grasps_overview_html(
        path,
        title=f"{sequence.assembly} / {library.step_id} inserter population",
        subtitle="All unary-evaluated grasps. Use the pass/fail buttons and Shown slider to inspect the full population.",
        mesh_local=mesh,
        candidate_statuses=statuses,
        object_pose_world=source,
        metadata_lines=[
            f"raw candidates: {library.raw_candidate_count}",
            f"unary evaluated: {len(statuses)}",
            f"unary accepted: {len(library.accepted_candidates)}",
            "red/amber markers rejected; green markers passed all unary constraints",
        ],
    )
    return path


def _write_inserter_unary_constraint_debug(*, library, sequence, planning, pair_config, output_dir: Path) -> Path:
    """Write the full Stage-3 explorer: overview filters plus a detailed scene."""

    path = output_dir / f"inserter_unary_debug_{library.step_id}.html"
    write_inserter_unary_debug_html(
        library=library,
        sequence=sequence,
        planning=planning,
        pair_config=pair_config,
        output_path=path,
    )
    return path


def _pair_config(payload: dict[str, object]) -> DualGraspPairConfig:
    raw = dict(payload.get("pair_planning", {}))
    contact_pairs = tuple(
        tuple(float(value) for value in pair)
        for pair in raw.get("inserter_contact_offset_pairs_m", DualGraspPairConfig().inserter_contact_offset_pairs_m)
    )
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
        inserter_contact_offset_pairs_m=contact_pairs,
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
    write_holder_state_debug_artifacts(
        holder_feasibility,
        sequence,
        output_dir,
    )
    for library in inserter_libraries:
        write_inserter_grasp_library(
            library,
            output_dir / inserter_artifact_name(library),
        )
        _write_inserter_failure_debug(
            library=library,
            sequence=sequence,
            planning=planning,
            output_dir=output_dir,
        )
        _write_inserter_population_debug(
            library=library,
            sequence=sequence,
            output_dir=output_dir,
        )
        _write_inserter_unary_constraint_debug(
            library=library,
            sequence=sequence,
            planning=planning,
            pair_config=pair_config,
            output_dir=output_dir,
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
    index_html = _write_index_html(
        output_dir=output_dir,
        result=result,
        summary_html=summary_html,
        step_htmls=tuple(step_htmls),
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
    print(f"index html: {index_html}", flush=True)
    print(f"step html files: {len(step_htmls)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
