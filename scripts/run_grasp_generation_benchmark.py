#!/usr/bin/env python3
"""Batch benchmark grasp generation over Fabrica OBJ parts and stable orientations."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.collision import trimesh_fcl_backend_available  # noqa: E402
from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    CandidateStatus,
    SavedGraspCandidate,
    candidate_payload,
    relative_asset_mesh_path,
    unique_edges,
)
from grasp_planning.grasping.mesh_io import resolve_mesh_path  # noqa: E402
from grasp_planning.pipeline import (  # noqa: E402
    GeometryConfig,
    PlanningConfig,
    generate_stage1_result,
    plan_mujoco_regrasp_fallback,
    recheck_stage2_result,
    write_mujoco_regrasp_debug_html,
    write_mujoco_regrasp_plan,
    write_stage1_artifacts,
    write_stage2_artifacts,
)
from grasp_planning.pipeline.stable_orientations import (  # noqa: E402
    StableOrientation,
    StableOrientationConfig,
    enumerate_stable_orientations,
    stable_orientation_payload,
    stable_orientation_result_payload,
)
from scripts.run_grasp_pipeline import _planning_config  # noqa: E402
from scripts.write_part_frame_debug_html import write_part_frame_debug_html  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "grasp_generation_benchmark.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "grasp_generation_benchmark"


@dataclass(frozen=True)
class TargetSpec:
    assembly: str
    part_id: str
    target_mesh_path: str
    assembly_glob: str


@dataclass(frozen=True)
class FallbackBenchmarkConfig:
    enabled: bool = True
    yaw_angles_deg: tuple[float, ...] = (0.0,)
    max_orientations: int = 24
    max_placement_options: int = 18
    min_facet_area_m2: float = 1.0e-6
    stability_margin_m: float = 0.0
    coplanar_tolerance_m: float = 1.0e-6
    staging_xy_offsets_m: tuple[tuple[float, float], ...] = ((0.0, 0.0),)


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in '{path}'.")
    return payload


def _tuple_floats(values: object, *, expected_len: int | None = None) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a list/tuple of floats, got {values!r}.")
    result = tuple(float(value) for value in values)
    if expected_len is not None and len(result) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(result)}.")
    return result


def _tuple_float_pairs(values: object) -> tuple[tuple[float, float], ...]:
    if values in ("", None):
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a list/tuple of [x, y] pairs, got {values!r}.")
    return tuple(_tuple_floats(value, expected_len=2) for value in values)  # type: ignore[arg-type]


def _fallback_config(payload: dict[str, object]) -> FallbackBenchmarkConfig:
    raw = dict(payload.get("fallback", {}))
    return FallbackBenchmarkConfig(
        enabled=bool(raw.get("enabled", True)),
        yaw_angles_deg=_tuple_floats(raw.get("yaw_angles_deg", [0.0])),
        max_orientations=int(raw.get("max_orientations", 24)),
        max_placement_options=int(raw.get("max_placement_options", 18)),
        min_facet_area_m2=float(raw.get("min_facet_area_m2", 1.0e-6)),
        stability_margin_m=float(raw.get("stability_margin_m", 0.0)),
        coplanar_tolerance_m=float(raw.get("coplanar_tolerance_m", 1.0e-6)),
        staging_xy_offsets_m=_tuple_float_pairs(raw.get("staging_xy_offsets_m", [[0.0, 0.0]])) or ((0.0, 0.0),),
    )


def _stable_orientation_config(payload: dict[str, object]) -> StableOrientationConfig:
    raw = dict(payload.get("stable_orientations", {}))
    return StableOrientationConfig(
        robust_tilt_deg=float(raw.get("robust_tilt_deg", 5.0)),
        min_support_area_m2=float(raw.get("min_support_area_m2", 1.0e-6)),
        min_support_area_fraction=float(raw.get("min_support_area_fraction", 0.01)),
        coplanar_tolerance_m=float(raw.get("coplanar_tolerance_m", 1.0e-6)),
        xy_world=_tuple_floats(raw.get("xy_world", [0.0, 0.0]), expected_len=2),  # type: ignore[arg-type]
    )


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value)) or "item"


def _candidate_payload(candidate: SavedGraspCandidate) -> dict[str, object]:
    return {
        "grasp_id": candidate.grasp_id,
        "grasp_pose_obj": {
            "position": list(candidate.grasp_position_obj),
            "orientation_xyzw": list(candidate.grasp_orientation_xyzw_obj),
        },
        "contact_points_obj": [list(candidate.contact_point_a_obj), list(candidate.contact_point_b_obj)],
        "contact_normals_obj": [list(candidate.contact_normal_a_obj), list(candidate.contact_normal_b_obj)],
        "jaw_width": candidate.jaw_width,
        "roll_angle_rad": candidate.roll_angle_rad,
        "contact_patch_offset_local": [
            candidate.contact_patch_lateral_offset_m,
            candidate.contact_patch_approach_offset_m,
        ],
        "score": candidate.score,
        "score_components": candidate.score_components,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _git_metadata() -> dict[str, object]:
    def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    commit = _run_git(["rev-parse", "--short", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "",
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.name == "stage1_cache":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _target_spec_from_asset_path(path: Path, *, target_root: str) -> TargetSpec:
    resolved = resolve_mesh_path(path)
    assets_relative = Path(relative_asset_mesh_path(resolved))
    root_relative = Path(target_root)
    try:
        rel = assets_relative.relative_to(root_relative)
    except ValueError as exc:
        raise ValueError(f"Target '{assets_relative}' is not under target_root '{target_root}'.") from exc
    if len(rel.parts) < 2:
        raise ValueError(f"Target '{assets_relative}' must be under an assembly directory.")
    assembly = rel.parts[0]
    part_id = rel.stem
    return TargetSpec(
        assembly=assembly,
        part_id=part_id,
        target_mesh_path=assets_relative.as_posix(),
        assembly_glob=(root_relative / assembly / "*.obj").as_posix(),
    )


def _discover_targets(payload: dict[str, object], args: argparse.Namespace) -> list[TargetSpec]:
    target_root = str(payload.get("target_root", "obj/fabrica")).strip() or "obj/fabrica"
    root_path = resolve_mesh_path(target_root)
    if args.target:
        raw_paths = [Path(value) for value in args.target]
        specs = [_target_spec_from_asset_path(path, target_root=target_root) for path in raw_paths]
    else:
        specs = [
            _target_spec_from_asset_path(path, target_root=target_root)
            for path in sorted(root_path.glob("*/*.obj"))
            if path.is_file()
        ]
    if args.assembly:
        allowed = {str(value) for value in args.assembly}
        specs = [spec for spec in specs if spec.assembly in allowed]
    if args.part:
        allowed_parts = {Path(str(value)).stem for value in args.part}
        specs = [spec for spec in specs if spec.part_id in allowed_parts]
    specs = sorted(specs, key=lambda spec: (spec.assembly, spec.part_id, spec.target_mesh_path))
    if args.limit_parts is not None:
        specs = specs[: max(0, int(args.limit_parts))]
    return specs


def _reason_counts(statuses: Iterable[CandidateStatus]) -> dict[str, int]:
    return dict(Counter(entry.reason for entry in statuses))


def _best_candidate_payload(candidates: Iterable[SavedGraspCandidate]) -> dict[str, object] | None:
    candidates_tuple = tuple(candidates)
    if not candidates_tuple:
        return None
    best = candidates_tuple[0]
    return {
        "grasp_id": best.grasp_id,
        "score": best.score,
        "contact_patch_offset_local": [
            best.contact_patch_lateral_offset_m,
            best.contact_patch_approach_offset_m,
        ],
    }


def _raw_stage1_payload(stage1, target: TargetSpec) -> dict[str, object]:
    return {
        "target_mesh_path": target.target_mesh_path,
        "assembly_glob": target.assembly_glob,
        "raw_candidate_count": stage1.raw_candidate_count,
        "scored_raw_candidate_count": len(stage1.raw_candidates),
        "candidates": [_candidate_payload(candidate) for candidate in stage1.raw_candidates],
    }


def _status_for_orientation(*, stage1_count: int, stage2_count: int, fallback_found: bool, fallback_enabled: bool) -> str:
    if stage1_count <= 0:
        return "stage1_failed"
    if stage2_count > 0:
        return "direct_success"
    if fallback_found:
        return "fallback_success"
    if fallback_enabled:
        return "stage2_failed_fallback_failed"
    return "stage2_failed_no_fallback"


def _fallback_summary(plan) -> dict[str, object] | None:
    if plan is None:
        return None
    return {
        "transfer_grasp_id": plan.transfer_grasp.grasp_id,
        "transfer_grasp_score": plan.transfer_grasp.score,
        "final_grasp_id": plan.final_grasp.grasp_id,
        "final_grasp_score": plan.final_grasp.score,
        "placement_option_count": len(plan.placement_options),
        "selected_support_stability_margin_m": plan.support_facet.stability_margin_m,
        "metadata": plan.metadata,
    }


def _orientation_row(
    *,
    target: TargetSpec,
    orientation: StableOrientation,
    status: str,
    stage1_raw_count: int,
    stage1_count: int,
    stage2_count: int,
    best_direct: dict[str, object] | None,
    fallback_summary: dict[str, object] | None,
    error: str = "",
) -> dict[str, object]:
    return {
        "assembly": target.assembly,
        "part_id": target.part_id,
        "target_mesh_path": target.target_mesh_path,
        "orientation_id": orientation.orientation_id,
        "status": status,
        "com_method": orientation.com_method,
        "support_area_m2": orientation.area_m2,
        "stability_margin_m": orientation.stability_margin_m,
        "com_height_m": orientation.com_height_m,
        "max_stable_tilt_deg": orientation.max_stable_tilt_deg,
        "stage1_raw_count": stage1_raw_count,
        "stage1_assembly_feasible_count": stage1_count,
        "stage2_ground_feasible_count": stage2_count,
        "best_direct_grasp_id": "" if best_direct is None else str(best_direct.get("grasp_id", "")),
        "best_direct_score": "" if best_direct is None else best_direct.get("score", ""),
        "fallback_found": fallback_summary is not None,
        "fallback_transfer_grasp_id": "" if fallback_summary is None else fallback_summary["transfer_grasp_id"],
        "fallback_final_grasp_id": "" if fallback_summary is None else fallback_summary["final_grasp_id"],
        "fallback_placement_option_count": 0 if fallback_summary is None else fallback_summary["placement_option_count"],
        "error": error,
    }


def _write_summary_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "assembly",
        "part_id",
        "target_mesh_path",
        "orientation_id",
        "status",
        "com_method",
        "support_area_m2",
        "stability_margin_m",
        "com_height_m",
        "max_stable_tilt_deg",
        "stage1_raw_count",
        "stage1_assembly_feasible_count",
        "stage2_ground_feasible_count",
        "best_direct_grasp_id",
        "best_direct_score",
        "fallback_found",
        "fallback_transfer_grasp_id",
        "fallback_final_grasp_id",
        "fallback_placement_option_count",
        "error",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_summary_md(output_path: Path, *, rows: list[dict[str, object]], part_records: list[dict[str, object]]) -> None:
    status_counts = Counter(str(row["status"]) for row in rows)
    part_status_counts = Counter(str(part.get("status", "unknown")) for part in part_records)
    direct = status_counts.get("direct_success", 0)
    fallback = status_counts.get("fallback_success", 0)
    total = len(rows)
    generation_success = direct + fallback
    lines = [
        "# Grasp Generation Benchmark Summary",
        "",
        f"- parts: {len(part_records)}",
        f"- orientations: {total}",
        f"- direct successes: {direct}",
        f"- fallback successes: {fallback}",
        f"- generation successes: {generation_success}",
        f"- generation success rate: {generation_success / total:.3f}" if total else "- generation success rate: n/a",
        "",
        "## Orientation Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Part Status Counts", ""])
    for status, count in sorted(part_status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Failed Orientations", ""])
    failed_rows = [row for row in rows if str(row["status"]) not in {"direct_success", "fallback_success"}]
    if not failed_rows:
        lines.append("- none")
    else:
        for row in failed_rows:
            lines.append(
                f"- {row['target_mesh_path']} {row['orientation_id']}: {row['status']} "
                f"(stage1={row['stage1_assembly_feasible_count']}, stage2={row['stage2_ground_feasible_count']})"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relative_link(output_dir: Path, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _write_index_html(
    output_path: Path,
    *,
    output_dir: Path,
    rows: list[dict[str, object]],
    part_records: list[dict[str, object]],
) -> None:
    status_counts = Counter(str(row["status"]) for row in rows)
    part_by_key = {(part["assembly"], part["part_id"]): part for part in part_records}
    table_rows: list[str] = []
    for row in rows:
        key = (row["assembly"], row["part_id"])
        part = part_by_key.get(key, {})
        links = row.get("links", {}) if isinstance(row.get("links"), dict) else {}
        stage1_link = _relative_link(output_dir, Path(str(part.get("stage1_html")))) if part.get("stage1_html") else ""
        orientations_link = (
            _relative_link(output_dir, Path(str(part.get("orientations_html"))))
            if part.get("orientations_html")
            else ""
        )
        stage2_link = str(links.get("stage2_html", ""))
        fallback_link = str(links.get("fallback_html", ""))
        link_html = " ".join(
            item
            for item in (
                f'<a href="{html.escape(stage1_link)}">stage1</a>' if stage1_link else "",
                f'<a href="{html.escape(orientations_link)}">orientations</a>' if orientations_link else "",
                f'<a href="{html.escape(stage2_link)}">stage2</a>' if stage2_link else "",
                f'<a href="{html.escape(fallback_link)}">fallback</a>' if fallback_link else "",
            )
            if item
        )
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['assembly']))}</td>"
            f"<td>{html.escape(str(row['part_id']))}</td>"
            f"<td>{html.escape(str(row['orientation_id']))}</td>"
            f"<td><span class=\"status {html.escape(str(row['status']))}\">{html.escape(str(row['status']))}</span></td>"
            f"<td>{html.escape(str(row['stage1_assembly_feasible_count']))}</td>"
            f"<td>{html.escape(str(row['stage2_ground_feasible_count']))}</td>"
            f"<td>{html.escape(str(row['max_stable_tilt_deg']))}</td>"
            f"<td>{link_html}</td>"
            "</tr>"
        )
    if not table_rows:
        table_rows.append("<tr><td colspan=\"8\">No orientation rows were generated.</td></tr>")
    counts_html = "".join(
        f"<li><strong>{html.escape(status)}</strong>: {count}</li>" for status, count in sorted(status_counts.items())
    )
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Grasp Generation Benchmark</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; color: #1f2933; background: #f7f7f4; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 12px; font-size: 28px; }}
    .summary {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }}
    .summary section {{ border: 1px solid #d8d8cf; background: #fff; border-radius: 8px; padding: 14px 16px; min-width: 220px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d8d8cf; }}
    th, td {{ border-bottom: 1px solid #e3e3dc; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #ecece4; position: sticky; top: 0; }}
    a {{ color: #2563eb; text-decoration: none; margin-right: 8px; }}
    a:hover {{ text-decoration: underline; }}
    .status {{ border-radius: 999px; padding: 3px 8px; background: #e5e7eb; white-space: nowrap; }}
    .direct_success {{ background: #dcfce7; color: #166534; }}
    .fallback_success {{ background: #dbeafe; color: #1d4ed8; }}
    .stage1_failed, .stage2_failed_fallback_failed, .stage2_failed_no_fallback {{ background: #fee2e2; color: #991b1b; }}
  </style>
</head>
<body>
  <main>
    <h1>Grasp Generation Benchmark</h1>
    <div class="summary">
      <section><strong>Parts</strong><br>{len(part_records)}</section>
      <section><strong>Orientations</strong><br>{len(rows)}</section>
      <section><strong>Status Counts</strong><ul>{counts_html}</ul></section>
    </div>
    <table>
      <thead>
        <tr>
          <th>Assembly</th><th>Part</th><th>Orientation</th><th>Status</th>
          <th>Stage 1</th><th>Stage 2</th><th>Max Tilt Deg</th><th>Links</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def _world_floor_corners(vertices_world: np.ndarray) -> list[list[float]]:
    mins = vertices_world.min(axis=0)
    maxs = vertices_world.max(axis=0)
    extents = np.maximum(maxs - mins, 1.0e-3)
    padding = max(0.35 * float(np.max(extents[:2])), 0.08)
    return [
        [float(mins[0] - padding), float(mins[1] - padding), 0.0],
        [float(maxs[0] + padding), float(mins[1] - padding), 0.0],
        [float(maxs[0] + padding), float(maxs[1] + padding), 0.0],
        [float(mins[0] - padding), float(maxs[1] + padding), 0.0],
    ]


def _part_orientation_frame(
    *,
    stage1,
    planning: PlanningConfig,
    target: TargetSpec,
    orientation: StableOrientation,
    status: str,
    stage2=None,
    fallback_summary: dict[str, object] | None = None,
    links: dict[str, str] | None = None,
    error: str = "",
) -> dict[str, object]:
    mesh_vertices_world = orientation.object_pose_world.transform_points_to_world(
        np.asarray(stage1.target_mesh_local.vertices_obj, dtype=float)
    )
    best_grasp_payload = None
    if stage2 is not None and stage2.accepted:
        best_grasp_payload = candidate_payload(
            [CandidateStatus(grasp=stage2.accepted[0], status="accepted", reason="best_direct_grasp")],
            contact_gap_m=planning.detailed_finger_contact_gap_m,
            object_pose_world=orientation.object_pose_world,
        )[0]
    return {
        "target": {
            "assembly": target.assembly,
            "part_id": target.part_id,
            "target_mesh_path": target.target_mesh_path,
        },
        "orientation": stable_orientation_payload(orientation),
        "status": status,
        "error": error,
        "stage1_assembly_feasible_count": len(stage1.bundle.candidates),
        "stage2_ground_feasible_count": 0 if stage2 is None else len(stage2.accepted),
        "stage2_reason_counts": {} if stage2 is None else _reason_counts(stage2.statuses),
        "fallback": fallback_summary,
        "links": links or {},
        "mesh_vertices_world": [[float(v) for v in vertex] for vertex in mesh_vertices_world.tolist()],
        "floor_world": _world_floor_corners(mesh_vertices_world),
        "best_grasp": best_grasp_payload,
    }


def _write_part_orientations_html(
    output_html: Path,
    *,
    target: TargetSpec,
    stage1,
    orientation_frames: list[dict[str, object]],
) -> None:
    mesh_local = stage1.target_mesh_local
    data = {
        "title": f"{target.assembly}/{target.part_id} Stable Orientations",
        "subtitle": "One frame per stable orientation. The grasp shown is the highest-scoring direct stage-2 grasp when one exists.",
        "target_mesh_path": target.target_mesh_path,
        "vertices_local": [[float(v) for v in vertex] for vertex in np.asarray(mesh_local.vertices_obj, dtype=float).tolist()],
        "faces": [[int(value) for value in face] for face in np.asarray(mesh_local.faces, dtype=np.int64).tolist()],
        "edges": unique_edges(mesh_local.faces),
        "frames": orientation_frames,
    }
    data_json = json.dumps(data, indent=2)
    document = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stable Orientation Winners</title>
  <style>
    :root {
      --bg: #f6f4ee;
      --panel: #fffdf8;
      --ink: #1f2522;
      --muted: #68716c;
      --line: #d9d4c7;
      --mesh: #2f6f5e;
      --floor: #2563eb;
      --success: #15803d;
      --fallback: #1d4ed8;
      --fail: #b91c1c;
      --franka: #d97706;
      --hand: #8f5a12;
      --contact-a: #c8452d;
      --contact-b: #1f7c60;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    .layout { display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: 100vh; }
    aside { border-right: 1px solid var(--line); background: var(--panel); padding: 20px; overflow: auto; }
    main { padding: 18px; overflow: auto; }
    h1 { margin: 0 0 8px; font-size: 25px; line-height: 1.15; }
    .subtitle { margin: 0 0 16px; color: var(--muted); font-size: 14px; line-height: 1.45; }
    .controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 14px; }
    button { border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 8px; padding: 10px 12px; font: inherit; cursor: pointer; }
    button:hover { border-color: var(--mesh); }
    .orientation-list { display: grid; gap: 8px; }
    .orientation-item { width: 100%; text-align: left; border-radius: 8px; }
    .orientation-item.active { border-color: var(--mesh); box-shadow: 0 0 0 2px rgba(47,111,94,0.14); }
    .item-title { display: flex; justify-content: space-between; gap: 8px; font-weight: 700; }
    .item-meta { margin-top: 5px; color: var(--muted); font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.4; }
    .status { border-radius: 999px; padding: 2px 7px; font-size: 12px; white-space: nowrap; }
    .direct_success { background: #dcfce7; color: var(--success); }
    .fallback_success { background: #dbeafe; color: var(--fallback); }
    .failed { background: #fee2e2; color: var(--fail); }
    .card { border: 1px solid var(--line); background: rgba(255,253,248,0.96); border-radius: 8px; padding: 14px; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.6fr); gap: 16px; align-items: start; }
    #scene { width: 100%; aspect-ratio: 1.45 / 1; display: block; border-radius: 8px; background: linear-gradient(180deg, #ffffff, #ebe7dc); }
    .kv { white-space: pre-wrap; font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.55; margin: 0; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; color: var(--muted); font-size: 13px; }
    .legend span { display: inline-flex; align-items: center; gap: 7px; }
    .swatch { width: 13px; height: 13px; border-radius: 999px; display: inline-block; }
    a { color: #2563eb; text-decoration: none; margin-right: 10px; }
    a:hover { text-decoration: underline; }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1 id="title"></h1>
      <p id="subtitle" class="subtitle"></p>
      <div class="controls">
        <button id="prevBtn" type="button">Previous</button>
        <button id="nextBtn" type="button">Next</button>
        <button id="solidBtn" type="button">Solid Mesh</button>
        <button id="resetBtn" type="button">Reset View</button>
      </div>
      <div id="orientationList" class="orientation-list"></div>
    </aside>
    <main>
      <div class="grid">
        <section class="card">
          <svg id="scene" viewBox="0 0 1100 760"></svg>
          <div class="legend">
            <span><i class="swatch" style="background: var(--mesh)"></i>Target mesh</span>
            <span><i class="swatch" style="background: var(--floor)"></i>Ground</span>
            <span><i class="swatch" style="background: var(--franka)"></i>Finger boxes</span>
            <span><i class="swatch" style="background: var(--hand)"></i>Hand mesh</span>
            <span><i class="swatch" style="background: var(--contact-a)"></i>Contact A</span>
            <span><i class="swatch" style="background: var(--contact-b)"></i>Contact B</span>
          </div>
        </section>
        <section class="card">
          <pre id="details" class="kv"></pre>
        </section>
      </div>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const scene = document.getElementById("scene");
    const details = document.getElementById("details");
    const list = document.getElementById("orientationList");
    document.getElementById("title").textContent = data.title;
    document.getElementById("subtitle").textContent = data.subtitle;
    const initialView = { yaw: -0.72, pitch: 0.52, zoom: 1.0, panX: 0, panY: 0 };
    const state = { index: 0, solid: false, dragging: false, dragMode: "rotate", pointerId: null, lastX: 0, lastY: 0, ...initialView };
    function statusClass(status) {
      if (status === "direct_success") return "direct_success";
      if (status === "fallback_success") return "fallback_success";
      return "failed";
    }
    function currentFrame() {
      return data.frames[state.index] || null;
    }
    function fmt(value, digits = 3) {
      if (value === null || value === undefined || value === "") return "n/a";
      return Number(value).toFixed(digits);
    }
    function allPoints(frame) {
      const points = [...frame.mesh_vertices_world, ...frame.floor_world];
      const grasp = frame.best_grasp;
      if (grasp) {
        points.push(
          grasp.grasp_position_obj,
          grasp.contact_point_a_obj,
          grasp.contact_point_b_obj,
          ...grasp.franka_hand_vertices_obj,
          ...grasp.franka_left_boxes.flatMap((box) => box.corners),
          ...grasp.franka_right_boxes.flatMap((box) => box.corners),
        );
      }
      return points;
    }
    function bounds(points) {
      return points.reduce((acc, point) => {
        point.forEach((value, axis) => {
          acc.min[axis] = Math.min(acc.min[axis], value);
          acc.max[axis] = Math.max(acc.max[axis], value);
        });
        return acc;
      }, { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
    }
    function rotate(point, center) {
      const shifted = point.map((value, axis) => value - center[axis]);
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];
      return [x1, cp * y1 + sp * z1, -sp * y1 + cp * z1];
    }
    function projection(frame) {
      const b = bounds(allPoints(frame));
      const center = b.min.map((value, axis) => 0.5 * (value + b.max[axis]));
      const extent = Math.max(...b.max.map((value, axis) => value - b.min[axis]), 0.08);
      return { center, scale: 560 / extent };
    }
    function project(point, projectionBounds) {
      const [x, y, z] = rotate(point, projectionBounds.center);
      const scale = projectionBounds.scale * state.zoom;
      return { x: 550 + state.panX + x * scale, y: 380 + state.panY - y * scale, depth: z };
    }
    function add(tag, attrs) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      scene.appendChild(node);
      return node;
    }
    function line(a, b, options, projectionBounds) {
      const pa = project(a, projectionBounds), pb = project(b, projectionBounds);
      add("line", { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, stroke: options.stroke || "#333", "stroke-width": options.width || 2, "stroke-opacity": options.opacity ?? 1, "stroke-dasharray": options.dash || "" });
    }
    function point(p, options, projectionBounds) {
      const pp = project(p, projectionBounds);
      add("circle", { cx: pp.x, cy: pp.y, r: options.radius || 5, fill: options.fill || "#000", stroke: "#fff", "stroke-width": options.strokeWidth || 1.5, "fill-opacity": options.opacity ?? 1 });
    }
    function polygon(points, options, projectionBounds) {
      const projected = points.map((p) => project(p, projectionBounds));
      add("polygon", { points: projected.map((p) => `${p.x},${p.y}`).join(" "), fill: options.fill || "none", "fill-opacity": options.fillOpacity ?? 1, stroke: options.stroke || "none", "stroke-width": options.strokeWidth || 1, "stroke-opacity": options.strokeOpacity ?? 1 });
    }
    function label(p, text, color, projectionBounds) {
      const pp = project(p, projectionBounds);
      const node = add("text", { x: pp.x + 8, y: pp.y - 8, fill: color, "font-size": 14, "font-family": "IBM Plex Mono, monospace", "font-weight": 700 });
      node.textContent = text;
    }
    function drawMesh(frame, projectionBounds) {
      if (state.solid) {
        data.faces.map((face) => {
          const points = face.map((index) => frame.mesh_vertices_world[index]);
          const rotated = points.map((p) => rotate(p, projectionBounds.center));
          const edgeA = rotated[1].map((value, axis) => value - rotated[0][axis]);
          const edgeB = rotated[2].map((value, axis) => value - rotated[0][axis]);
          const normal = [
            edgeA[1] * edgeB[2] - edgeA[2] * edgeB[1],
            edgeA[2] * edgeB[0] - edgeA[0] * edgeB[2],
            edgeA[0] * edgeB[1] - edgeA[1] * edgeB[0],
          ];
          const depth = rotated.reduce((sum, p) => sum + p[2], 0) / rotated.length;
          return { points, normal, depth };
        }).filter((face) => face.normal[2] > 0).sort((a, b) => a.depth - b.depth).forEach((face) => {
          polygon(face.points, { fill: "#7aa392", fillOpacity: 0.86, stroke: "#2f6f5e", strokeWidth: 0.8, strokeOpacity: 0.45 }, projectionBounds);
        });
      }
      data.edges.forEach(([a, b]) => line(frame.mesh_vertices_world[a], frame.mesh_vertices_world[b], { stroke: "#2f6f5e", width: 1.5, opacity: 0.82 }, projectionBounds));
    }
    function drawBox(corners, color, projectionBounds) {
      [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([a, b]) => {
        line(corners[a], corners[b], { stroke: color, width: 1.6, opacity: 0.78 }, projectionBounds);
      });
    }
    function drawHand(grasp, projectionBounds) {
      grasp.franka_hand_faces.forEach((face) => {
        line(grasp.franka_hand_vertices_obj[face[0]], grasp.franka_hand_vertices_obj[face[1]], { stroke: "#8f5a12", width: 0.9, opacity: 0.28 }, projectionBounds);
        line(grasp.franka_hand_vertices_obj[face[1]], grasp.franka_hand_vertices_obj[face[2]], { stroke: "#8f5a12", width: 0.9, opacity: 0.28 }, projectionBounds);
        line(grasp.franka_hand_vertices_obj[face[2]], grasp.franka_hand_vertices_obj[face[0]], { stroke: "#8f5a12", width: 0.9, opacity: 0.28 }, projectionBounds);
      });
    }
    function drawGrasp(grasp, projectionBounds) {
      grasp.franka_left_boxes.forEach((box) => drawBox(box.corners, "#d97706", projectionBounds));
      grasp.franka_right_boxes.forEach((box) => drawBox(box.corners, "#d97706", projectionBounds));
      drawHand(grasp, projectionBounds);
      line(grasp.contact_point_a_obj, grasp.contact_point_b_obj, { stroke: "#15803d", width: 3, opacity: 0.95 }, projectionBounds);
      point(grasp.grasp_position_obj, { fill: "#15803d", radius: 7 }, projectionBounds);
      point(grasp.contact_point_a_obj, { fill: "#c8452d", radius: 6 }, projectionBounds);
      point(grasp.contact_point_b_obj, { fill: "#1f7c60", radius: 6 }, projectionBounds);
      label(grasp.grasp_position_obj, grasp.grasp_id, "#15803d", projectionBounds);
    }
    function renderList() {
      list.replaceChildren();
      data.frames.forEach((frame, index) => {
        const grasp = frame.best_grasp;
        const button = document.createElement("button");
        button.type = "button";
        button.className = `orientation-item${index === state.index ? " active" : ""}`;
        button.innerHTML = `
          <div class="item-title">
            <span>${frame.orientation.orientation_id}</span>
            <span class="status ${statusClass(frame.status)}">${frame.status}</span>
          </div>
          <div class="item-meta">
            feasible=${frame.stage2_ground_feasible_count} tilt=${fmt(frame.orientation.max_stable_tilt_deg, 1)}deg<br>
            best=${grasp ? `${grasp.grasp_id} score=${fmt(grasp.score, 3)}` : "none"}
          </div>
        `;
        button.addEventListener("click", () => { state.index = index; render(); });
        list.appendChild(button);
      });
    }
    function renderDetails(frame) {
      const grasp = frame.best_grasp;
      const links = Object.entries(frame.links || {}).filter(([, value]) => value).map(([key, value]) => `${key}: ${value}`);
      details.textContent = [
        `target:             ${data.target_mesh_path}`,
        `orientation:        ${frame.orientation.orientation_id}`,
        `status:             ${frame.status}`,
        `stage1_feasible:    ${frame.stage1_assembly_feasible_count}`,
        `stage2_feasible:    ${frame.stage2_ground_feasible_count}`,
        `normal_obj:         (${frame.orientation.normal_obj.map((v) => fmt(v, 4)).join(", ")})`,
        `support_area_m2:    ${fmt(frame.orientation.area_m2, 6)}`,
        `stability_margin_m: ${fmt(frame.orientation.stability_margin_m, 6)}`,
        `com_height_m:       ${fmt(frame.orientation.com_height_m, 6)}`,
        `max_stable_tilt:    ${fmt(frame.orientation.max_stable_tilt_deg, 3)} deg`,
        `com_method:         ${frame.orientation.com_method}`,
        "",
        grasp ? `best_grasp:        ${grasp.grasp_id}` : "best_grasp:        none",
        grasp ? `best_score:        ${fmt(grasp.score, 6)}` : "",
        grasp ? `jaw_width_m:       ${fmt(grasp.jaw_width, 6)}` : "",
        grasp ? `roll_angle_rad:    ${fmt(grasp.roll_angle_rad, 6)}` : "",
        grasp && grasp.score_components ? `object_score:      ${fmt(grasp.score_components.object_score, 6)}` : "",
        grasp && grasp.score_components ? `top_down_score:    ${fmt(grasp.score_components.top_down_approach, 6)}` : "",
        "",
        frame.fallback ? `fallback_final:    ${frame.fallback.final_grasp_id}` : "",
        frame.error ? `error:             ${frame.error}` : "",
        "",
        ...links,
      ].filter((line) => line !== "").join("\\n");
    }
    function renderScene(frame) {
      scene.replaceChildren();
      const b = projection(frame);
      polygon(frame.floor_world, { fill: "#2563eb", fillOpacity: 0.13, stroke: "#2563eb", strokeWidth: 1.8, strokeOpacity: 0.85 }, b);
      for (let i = 0; i < frame.floor_world.length; i += 1) {
        line(frame.floor_world[i], frame.floor_world[(i + 1) % frame.floor_world.length], { stroke: "#2563eb", width: 1.8, opacity: 0.85, dash: "8 5" }, b);
      }
      drawMesh(frame, b);
      if (frame.best_grasp) {
        drawGrasp(frame.best_grasp, b);
      } else {
        label(frame.mesh_vertices_world[0], "no direct stage-2 grasp", "#b91c1c", b);
      }
    }
    function render() {
      const frame = currentFrame();
      if (!frame) {
        details.textContent = "No stable orientations.";
        return;
      }
      renderList();
      renderScene(frame);
      renderDetails(frame);
    }
    function step(delta) {
      if (!data.frames.length) return;
      state.index = (state.index + delta + data.frames.length) % data.frames.length;
      render();
    }
    document.getElementById("prevBtn").addEventListener("click", () => step(-1));
    document.getElementById("nextBtn").addEventListener("click", () => step(1));
    document.getElementById("solidBtn").addEventListener("click", () => {
      state.solid = !state.solid;
      document.getElementById("solidBtn").textContent = state.solid ? "Wireframe Mesh" : "Solid Mesh";
      render();
    });
    document.getElementById("resetBtn").addEventListener("click", () => {
      Object.assign(state, initialView);
      render();
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") { event.preventDefault(); step(-1); }
      if (event.key === "ArrowRight" || event.key === "ArrowDown") { event.preventDefault(); step(1); }
    });
    scene.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 && event.button !== 1 && event.button !== 2) return;
      event.preventDefault();
      state.dragging = true;
      state.dragMode = event.button === 1 || event.button === 2 || event.shiftKey ? "pan" : "rotate";
      state.pointerId = event.pointerId;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      scene.setPointerCapture(event.pointerId);
    });
    function stopDragging() {
      state.dragging = false;
      state.pointerId = null;
    }
    scene.addEventListener("pointerup", (event) => { if (state.pointerId === event.pointerId) stopDragging(); });
    scene.addEventListener("pointercancel", stopDragging);
    scene.addEventListener("pointermove", (event) => {
      if (!state.dragging || (state.pointerId !== null && state.pointerId !== event.pointerId)) return;
      const dx = event.clientX - state.lastX;
      const dy = event.clientY - state.lastY;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      if (state.dragMode === "pan") {
        state.panX += dx;
        state.panY += dy;
      } else {
        state.yaw += dx * 0.01;
        state.pitch -= dy * 0.01;
      }
      render();
    });
    scene.addEventListener("wheel", (event) => {
      event.preventDefault();
      state.zoom = Math.max(0.3, Math.min(5.0, state.zoom * (event.deltaY < 0 ? 1.08 : 1 / 1.08)));
      render();
    }, { passive: false });
    scene.addEventListener("contextmenu", (event) => event.preventDefault());
    render();
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(document.replace("__DATA_JSON__", data_json), encoding="utf-8")


def _write_orientation_details(
    output_path: Path,
    *,
    target: TargetSpec,
    orientation: StableOrientation,
    status: str,
    stage1,
    stage2=None,
    fallback_summary: dict[str, object] | None = None,
    error: str = "",
) -> None:
    payload = {
        "target": {
            "assembly": target.assembly,
            "part_id": target.part_id,
            "target_mesh_path": target.target_mesh_path,
            "assembly_glob": target.assembly_glob,
        },
        "orientation": stable_orientation_payload(orientation),
        "status": status,
        "stage1": {
            "raw_candidate_count": stage1.raw_candidate_count,
            "assembly_feasible_count": len(stage1.bundle.candidates),
            "cache_hit": bool((stage1.bundle.metadata or {}).get("stage1_cache_hit")),
            "cache_path": (stage1.bundle.metadata or {}).get("stage1_cache_path"),
        },
        "stage2": None
        if stage2 is None
        else {
            "ground_feasible_count": len(stage2.accepted),
            "reason_counts": _reason_counts(stage2.statuses),
            "best_direct_grasp": _best_candidate_payload(stage2.accepted),
        },
        "fallback": fallback_summary,
        "error": error,
    }
    _write_json(output_path, payload)


def _apply_cli_overrides(payload: dict[str, object], args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    effective = deepcopy(payload)
    benchmark = dict(effective.get("benchmark", {}))
    benchmark["output_dir"] = str(output_dir)
    effective["benchmark"] = benchmark
    if args.robust_tilt_deg is not None:
        stable = dict(effective.get("stable_orientations", {}))
        stable["robust_tilt_deg"] = float(args.robust_tilt_deg)
        effective["stable_orientations"] = stable
    planning = dict(effective.get("planning", {}))
    if args.no_stage1_cache:
        planning["stage1_cache_enabled"] = False
    if args.skip_stage1_collision_checks:
        planning["skip_stage1_collision_checks"] = True
    effective["planning"] = planning
    if args.fallback_enabled is not None:
        fallback = dict(effective.get("fallback", {}))
        fallback["enabled"] = bool(args.fallback_enabled)
        effective["fallback"] = fallback
    return effective


def _benchmark_one_target(
    *,
    target: TargetSpec,
    planning: PlanningConfig,
    stable_config: StableOrientationConfig,
    fallback_config: FallbackBenchmarkConfig,
    mesh_scale: float,
    output_dir: Path,
    target_index: int,
    target_count: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    part_dir = output_dir / "parts" / _safe_id(target.assembly) / _safe_id(target.part_id)
    stage1_dir = part_dir / "stage1"
    orientations_dir = part_dir / "orientations"
    part_record: dict[str, object] = {
        "assembly": target.assembly,
        "part_id": target.part_id,
        "target_mesh_path": target.target_mesh_path,
        "assembly_glob": target.assembly_glob,
        "status": "pending",
    }
    rows: list[dict[str, object]] = []
    orientation_frames: list[dict[str, object]] = []
    geometry = GeometryConfig(
        target_mesh_path=target.target_mesh_path,
        mesh_scale=mesh_scale,
        assembly_glob=target.assembly_glob,
    )

    print(
        f"[{target_index}/{target_count}] {target.target_mesh_path}: generating stage 1.",
        flush=True,
    )
    try:
        stage1 = generate_stage1_result(geometry=geometry, planning=planning)
        stage1_json = stage1_dir / "grasps.json"
        stage1_html = stage1_dir / "grasps.html"
        write_stage1_artifacts(stage1, geometry=geometry, planning=planning, output_json=stage1_json, output_html=stage1_html)
        _write_json(stage1_dir / "raw_grasps.json", _raw_stage1_payload(stage1, target))
        part_frame_html = part_dir / "part_frame.html"
        write_part_frame_debug_html(input_json=stage1_json, output_html=part_frame_html)
        part_record.update(
            {
                "stage1_json": str(stage1_json),
                "stage1_html": str(stage1_html),
                "part_frame_html": str(part_frame_html),
                "stage1_raw_count": stage1.raw_candidate_count,
                "stage1_assembly_feasible_count": len(stage1.bundle.candidates),
                "stage1_cache_hit": bool((stage1.bundle.metadata or {}).get("stage1_cache_hit")),
            }
        )
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        part_record.update({"status": "stage1_error", "error": error})
        print(f"  stage1_error: {error}", flush=True)
        return part_record, rows

    try:
        orientation_result = enumerate_stable_orientations(stage1.target_mesh_local, stable_config)
        stable_json = part_dir / "stable_orientations.json"
        _write_json(stable_json, stable_orientation_result_payload(orientation_result))
        part_record.update(
            {
                "stable_orientations_json": str(stable_json),
                "stable_orientation_count": len(orientation_result.orientations),
                "rejected_orientation_candidate_count": len(orientation_result.rejected_candidates),
                "com_method": orientation_result.com_method,
            }
        )
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        part_record.update({"status": "orientation_generation_error", "error": error})
        print(f"  orientation_generation_error: {error}", flush=True)
        return part_record, rows

    if not orientation_result.orientations:
        part_record.update({"status": "no_stable_orientations"})
        print("  no_stable_orientations", flush=True)
        return part_record, rows

    for orientation in orientation_result.orientations:
        orientation_dir = orientations_dir / orientation.orientation_id
        stage2_json = orientation_dir / "stage2.json"
        stage2_html = orientation_dir / "stage2.html"
        fallback_json = orientation_dir / "fallback_plan.json"
        fallback_html = orientation_dir / "fallback_plan.html"
        print(f"  {orientation.orientation_id}: stage 2.", flush=True)
        try:
            stage2 = recheck_stage2_result(
                bundle=stage1.bundle,
                pickup_spec=None,
                planning=planning,
                object_pose_world=orientation.object_pose_world,
            )
            write_stage2_artifacts(stage2, planning=planning, output_json=stage2_json, output_html=stage2_html)
            plan = None
            if fallback_config.enabled and len(stage1.bundle.candidates) > 0 and not stage2.accepted:
                plan = plan_mujoco_regrasp_fallback(
                    stage1=stage1,
                    direct_stage2=stage2,
                    planning=planning,
                    force=False,
                    staging_xy_world=(
                        float(orientation.object_pose_world.position_world[0]),
                        float(orientation.object_pose_world.position_world[1]),
                    ),
                    staging_xy_offsets_m=fallback_config.staging_xy_offsets_m,
                    yaw_angles_deg=fallback_config.yaw_angles_deg,
                    max_orientations=fallback_config.max_orientations,
                    max_placement_options=fallback_config.max_placement_options,
                    min_facet_area_m2=fallback_config.min_facet_area_m2,
                    stability_margin_m=fallback_config.stability_margin_m,
                    coplanar_tolerance_m=fallback_config.coplanar_tolerance_m,
                )
                if plan is not None:
                    write_mujoco_regrasp_plan(plan, fallback_json, input_stage2_json=stage2_json)
                    write_mujoco_regrasp_debug_html(
                        plan=plan,
                        stage1=stage1,
                        planning=planning,
                        output_html=fallback_html,
                    )
            fallback_summary = _fallback_summary(plan)
            status = _status_for_orientation(
                stage1_count=len(stage1.bundle.candidates),
                stage2_count=len(stage2.accepted),
                fallback_found=fallback_summary is not None,
                fallback_enabled=fallback_config.enabled,
            )
            row = _orientation_row(
                target=target,
                orientation=orientation,
                status=status,
                stage1_raw_count=stage1.raw_candidate_count,
                stage1_count=len(stage1.bundle.candidates),
                stage2_count=len(stage2.accepted),
                best_direct=_best_candidate_payload(stage2.accepted),
                fallback_summary=fallback_summary,
            )
            row_links = {
                "stage2_json": _relative_link(output_dir, stage2_json),
                "stage2_html": _relative_link(output_dir, stage2_html),
                "fallback_json": _relative_link(output_dir, fallback_json) if fallback_summary is not None else "",
                "fallback_html": _relative_link(output_dir, fallback_html) if fallback_summary is not None else "",
            }
            row["links"] = row_links
            rows.append(row)
            orientation_frames.append(
                _part_orientation_frame(
                    stage1=stage1,
                    planning=planning,
                    target=target,
                    orientation=orientation,
                    status=status,
                    stage2=stage2,
                    fallback_summary=fallback_summary,
                    links=row_links,
                )
            )
            _write_orientation_details(
                orientation_dir / "details.json",
                target=target,
                orientation=orientation,
                status=status,
                stage1=stage1,
                stage2=stage2,
                fallback_summary=fallback_summary,
            )
            print(f"    {status}: stage2={len(stage2.accepted)} fallback={fallback_summary is not None}", flush=True)
        except Exception as exc:
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            row = _orientation_row(
                target=target,
                orientation=orientation,
                status="orientation_error",
                stage1_raw_count=stage1.raw_candidate_count,
                stage1_count=len(stage1.bundle.candidates),
                stage2_count=0,
                best_direct=None,
                fallback_summary=None,
                error=error,
            )
            row["links"] = {}
            rows.append(row)
            orientation_frames.append(
                _part_orientation_frame(
                    stage1=stage1,
                    planning=planning,
                    target=target,
                    orientation=orientation,
                    status="orientation_error",
                    links={},
                    error=error,
                )
            )
            _write_orientation_details(
                orientation_dir / "details.json",
                target=target,
                orientation=orientation,
                status="orientation_error",
                stage1=stage1,
                error=error,
            )
            print(f"    orientation_error: {error}", flush=True)

    orientations_html = part_dir / "orientations.html"
    _write_part_orientations_html(
        orientations_html,
        target=target,
        stage1=stage1,
        orientation_frames=orientation_frames,
    )
    part_statuses = Counter(str(row["status"]) for row in rows if row["assembly"] == target.assembly and row["part_id"] == target.part_id)
    if part_statuses.get("direct_success", 0) or part_statuses.get("fallback_success", 0):
        part_status = "has_generation_success"
    elif part_statuses:
        part_status = "all_orientations_failed"
    else:
        part_status = "no_orientation_rows"
    part_record.update(
        {
            "status": part_status,
            "orientation_status_counts": dict(part_statuses),
            "orientations_html": str(orientations_html),
        }
    )
    return part_record, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark grasp generation across Fabrica OBJ parts and stable poses.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Benchmark YAML config path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override benchmark output directory.")
    parser.add_argument("--clean", action="store_true", help="Remove stale benchmark artifacts, preserving stage1_cache.")
    parser.add_argument("--assembly", action="append", default=[], help="Restrict to an assembly name. Repeatable.")
    parser.add_argument("--part", action="append", default=[], help="Restrict to a part id/stem. Repeatable.")
    parser.add_argument("--target", action="append", default=[], help="Restrict to a target mesh path. Repeatable.")
    parser.add_argument("--limit-parts", type=int, default=None, help="Cap the number of targets after filtering.")
    parser.add_argument("--robust-tilt-deg", type=float, default=None, help="Override stable orientation robust tilt.")
    parser.add_argument("--no-stage1-cache", action="store_true", help="Disable stage-1 cache use for this run.")
    parser.add_argument(
        "--skip-stage1-collision-checks",
        action="store_true",
        help="Bypass only stage-1 assembly collision filtering.",
    )
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument("--fallback", dest="fallback_enabled", action="store_true", default=None)
    fallback_group.add_argument("--no-fallback", dest="fallback_enabled", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_yaml(args.config)
    configured_output_dir = Path(str(dict(payload.get("benchmark", {})).get("output_dir", DEFAULT_OUTPUT_DIR)))
    output_dir = (args.output_dir or configured_output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    payload = _apply_cli_overrides(payload, args, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        _clean_output_dir(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if not trimesh_fcl_backend_available():
        raise RuntimeError(
            "trimesh/FCL collision backend is unavailable. Install python-fcl/native FCL before running the benchmark."
        )

    planning = _planning_config(payload)
    stable_config = _stable_orientation_config(payload)
    fallback_config = _fallback_config(payload)
    mesh_scale = float(dict(payload.get("geometry", {})).get("mesh_scale", 1.0))
    targets = _discover_targets(payload, args)
    if not targets:
        raise RuntimeError("No benchmark targets matched the requested filters.")

    _write_yaml(output_dir / "benchmark_config.yaml", payload)
    provenance = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(args.config),
        "cli_args": _json_safe(vars(args)),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "git": _git_metadata(),
        "collision_backend": "trimesh_fcl",
        "target_count": len(targets),
        "targets": [target.target_mesh_path for target in targets],
    }

    part_records: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for index, target in enumerate(targets, start=1):
        part_record, part_rows = _benchmark_one_target(
            target=target,
            planning=planning,
            stable_config=stable_config,
            fallback_config=fallback_config,
            mesh_scale=mesh_scale,
            output_dir=output_dir,
            target_index=index,
            target_count=len(targets),
        )
        part_records.append(part_record)
        rows.extend(part_rows)

    results = {
        "schema_version": 1,
        "provenance": provenance,
        "config": payload,
        "parts": part_records,
        "orientations": rows,
        "summary": {
            "part_count": len(part_records),
            "orientation_count": len(rows),
            "orientation_status_counts": dict(Counter(str(row["status"]) for row in rows)),
            "part_status_counts": dict(Counter(str(part.get("status", "unknown")) for part in part_records)),
        },
    }
    _write_json(output_dir / "results.json", results)
    _write_summary_csv(output_dir / "summary.csv", rows)
    _write_summary_md(output_dir / "summary.md", rows=rows, part_records=part_records)
    _write_index_html(output_dir / "index.html", output_dir=output_dir, rows=rows, part_records=part_records)

    print(f"[BENCHMARK] Wrote results to {output_dir / 'results.json'}", flush=True)
    print(f"[BENCHMARK] Wrote summary to {output_dir / 'summary.md'}", flush=True)
    print(f"[BENCHMARK] Wrote index to {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
