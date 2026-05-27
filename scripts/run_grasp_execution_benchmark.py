#!/usr/bin/env python3
"""Execute saved benchmark grasps in MuJoCo and/or Isaac with per-attempt video."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "grasp_execution_benchmark.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "grasp_execution_benchmark"
BACKENDS = ("mujoco", "isaac")


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in '{path}'.")
    return payload


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _jsonl_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True))
        stream.write("\n")


def _effective_python_command(raw_value: object) -> list[str]:
    value = str(raw_value or "").strip()
    if value:
        return shlex.split(value)
    if sys.executable:
        return [sys.executable]
    raise RuntimeError("Could not determine a Python executable.")


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    if env.get("TERM", "") in {"", "dumb"}:
        env["TERM"] = "xterm"
    repo_path = str(REPO_ROOT)
    pythonpath = env.get("PYTHONPATH", "")
    entries = [entry for entry in pythonpath.split(os.pathsep) if entry]
    if repo_path not in entries:
        env["PYTHONPATH"] = os.pathsep.join([repo_path, *entries])
    return env


def _safe_id(value: object) -> str:
    text = str(value).strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _as_list(raw: object) -> list[str]:
    if raw in ("", None):
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw]
    return [str(raw)]


def _optional_int(raw: object) -> int | None:
    if raw in ("", None):
        return None
    return int(raw)


def _resolve_path(path_value: object, *, base_dir: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _backend_list(raw: object) -> tuple[str, ...]:
    normalized = str(raw or "mujoco").strip().lower()
    if normalized == "both":
        return BACKENDS
    if normalized in BACKENDS:
        return (normalized,)
    raise ValueError(f"Unsupported backend '{raw}'. Expected mujoco, isaac, or both.")


def _record_video_enabled(raw: object) -> bool:
    value = str(raw or "all").strip().lower()
    if value == "all":
        return True
    if value == "none":
        return False
    raise ValueError("execution_benchmark.record_video must be 'all' or 'none'.")


def _filtered_orientation_rows(
    *,
    generation_results: dict[str, object],
    generation_root: Path,
    assemblies: set[str],
    parts: set[str],
    orientations: set[str],
    limit_orientations: int | None,
) -> list[dict[str, object]]:
    raw_rows = generation_results.get("orientations", [])
    if not isinstance(raw_rows, list):
        raise ValueError("Generation results JSON is missing an orientations list.")
    rows: list[dict[str, object]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        assembly = str(raw_row.get("assembly", ""))
        part_id = str(raw_row.get("part_id", ""))
        orientation_id = str(raw_row.get("orientation_id", ""))
        if assemblies and assembly not in assemblies:
            continue
        if parts and part_id not in parts:
            continue
        if orientations and orientation_id not in orientations:
            continue
        links = raw_row.get("links", {})
        if not isinstance(links, dict) or not links.get("stage2_json"):
            continue
        stage2_json = _resolve_path(links["stage2_json"], base_dir=generation_root)
        if not stage2_json.is_file():
            continue
        row = dict(raw_row)
        row["stage2_json_path"] = str(stage2_json)
        rows.append(row)
        if limit_orientations is not None and len(rows) >= limit_orientations:
            break
    return rows


def _attempt_specs_for_row(
    *,
    row: dict[str, object],
    backends: tuple[str, ...],
    grasp_ids: set[str],
    max_grasps_per_orientation: int | None,
) -> list[dict[str, object]]:
    stage2_json = Path(str(row["stage2_json_path"]))
    bundle = load_grasp_bundle(stage2_json)
    candidates = [candidate for candidate in bundle.candidates if not grasp_ids or candidate.grasp_id in grasp_ids]
    if max_grasps_per_orientation is not None:
        candidates = candidates[: max(0, int(max_grasps_per_orientation))]
    specs: list[dict[str, object]] = []
    for candidate in candidates:
        for backend in backends:
            specs.append(
                {
                    "assembly": str(row.get("assembly", "")),
                    "part_id": str(row.get("part_id", "")),
                    "target_mesh_path": str(row.get("target_mesh_path", "")),
                    "orientation_id": str(row.get("orientation_id", "")),
                    "generation_status": str(row.get("status", "")),
                    "stage2_json": str(stage2_json),
                    "backend": backend,
                    "grasp_id": candidate.grasp_id,
                    "grasp_score": candidate.score,
                    "stage2_ground_feasible_count": int(
                        row.get("stage2_ground_feasible_count", len(bundle.candidates)) or 0
                    ),
                }
            )
    return specs


def _attempt_key(spec: dict[str, object]) -> str:
    return "|".join(
        [
            str(spec["backend"]),
            str(spec["assembly"]),
            str(spec["part_id"]),
            str(spec["orientation_id"]),
            str(spec["grasp_id"]),
            str(spec["stage2_json"]),
        ]
    )


def _attempt_dir(output_dir: Path, spec: dict[str, object]) -> Path:
    return (
        output_dir
        / "parts"
        / _safe_id(spec["assembly"])
        / _safe_id(spec["part_id"])
        / "orientations"
        / _safe_id(spec["orientation_id"])
        / _safe_id(spec["backend"])
        / _safe_id(spec["grasp_id"])
    )


def _append_optional(command: list[str], flag: str, value: object) -> None:
    if value in ("", None):
        return
    command.extend([flag, str(value)])


def _mujoco_command(
    *,
    cfg: dict[str, object],
    spec: dict[str, object],
    attempt_artifact: Path,
    video_path: Path | None,
) -> list[str]:
    command = [
        *_effective_python_command(cfg.get("python_executable", "")),
        "scripts/run_fabrica_grasp_in_mujoco.py",
        "--input-json",
        str(spec["stage2_json"]),
        "--robot-config",
        str(cfg.get("robot_config", "configs/mujoco_fr3_with_hand.json")),
        "--attempt-artifact",
        str(attempt_artifact),
        "--controller",
        str(cfg.get("controller", "native")),
        "--grasp-id",
        str(spec["grasp_id"]),
    ]
    _append_optional(command, "--simulation-config", cfg.get("simulation_config"))
    _append_optional(command, "--pregrasp-offset", cfg.get("pregrasp_offset"))
    _append_optional(command, "--gripper-width-clearance", cfg.get("gripper_width_clearance"))
    _append_optional(command, "--contact-gap-m", cfg.get("contact_gap_m"))
    _append_optional(command, "--object-mass-kg", cfg.get("object_mass_kg"))
    _append_optional(command, "--object-scale", cfg.get("object_scale"))
    _append_optional(command, "--lift-height-m", cfg.get("lift_height_m"))
    _append_optional(command, "--success-height-margin-m", cfg.get("success_height_margin_m"))
    if bool(cfg.get("keep_generated_scene", False)):
        command.append("--keep-generated-scene")
    if video_path is not None:
        video = dict(cfg.get("video", {}) or {})
        command.extend(
            [
                "--record-video",
                str(video_path),
                "--video-fps",
                str(video.get("fps", 30.0)),
                "--video-width",
                str(video.get("width", 960)),
                "--video-height",
                str(video.get("height", 540)),
                "--video-camera-azimuth",
                str(video.get("camera_azimuth", 135.0)),
                "--video-camera-elevation",
                str(video.get("camera_elevation", -25.0)),
                "--video-camera-distance",
                str(video.get("camera_distance", 1.45)),
            ]
        )
        lookat = video.get("camera_lookat", [0.35, 0.0, 0.28])
        command.extend(["--video-camera-lookat", *(str(value) for value in lookat)])
    return command


def _isaac_command(
    *,
    cfg: dict[str, object],
    spec: dict[str, object],
    attempt_artifact: Path,
    video_path: Path | None,
) -> list[str]:
    command = [
        *_effective_python_command(cfg.get("python_executable", "/media/pdz/Elements1/IsaacLab/isaaclab.sh -p")),
        "scripts/run_fabrica_grasp_in_isaac.py",
        "--input-json",
        str(spec["stage2_json"]),
        "--controller",
        str(cfg.get("controller", "planner")),
        "--attempt-artifact",
        str(attempt_artifact),
        "--grasp-id",
        str(spec["grasp_id"]),
        "--close-width",
        str(cfg.get("close_width", 0.0)),
        "--run-seconds",
        str(cfg.get("run_seconds", 0.0)),
    ]
    if bool(cfg.get("headless", True)):
        command.append("--headless")
    _append_optional(command, "--part-usd", cfg.get("part_usd"))
    if bool(cfg.get("use_provided_part_usd", False)):
        command.append("--use-provided-part-usd")
    _append_optional(command, "--fr3-usd", cfg.get("fr3_usd"))
    _append_optional(command, "--pregrasp-offset", cfg.get("pregrasp_offset"))
    _append_optional(command, "--gripper-width-clearance", cfg.get("gripper_width_clearance"))
    _append_optional(command, "--detailed-finger-contact-gap-m", cfg.get("contact_gap_m"))
    tcp_offset = cfg.get("tcp_to_grasp_offset")
    if tcp_offset not in ("", None):
        command.extend(["--tcp-to-grasp-offset", *(str(value) for value in tcp_offset)])
    if video_path is not None:
        video = dict(cfg.get("video", {}) or {})
        command.extend(
            [
                "--record-video",
                str(video_path),
                "--video-fps",
                str(video.get("fps", 30.0)),
                "--video-width",
                str(video.get("width", 960)),
                "--video-height",
                str(video.get("height", 540)),
            ]
        )
        eye = video.get("camera_eye", [1.6, -1.2, 1.0])
        target = video.get("camera_target", [0.35, 0.0, 0.3])
        command.extend(["--video-camera-eye", *(str(value) for value in eye)])
        command.extend(["--video-camera-target", *(str(value) for value in target)])
    return command


def _load_json_if_present(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _execution_summary(backend: str, artifact: dict[str, object] | None) -> dict[str, object]:
    if artifact is None:
        return {"success": False, "status": "artifact_missing", "message": "Attempt artifact was not written."}
    if backend == "isaac":
        execution = artifact.get("execution", {})
        video = artifact.get("video", {})
    else:
        execution = artifact.get("result", {})
        video = execution if isinstance(execution, dict) else {}
    if not isinstance(execution, dict):
        execution = {}
    if not isinstance(video, dict):
        video = {}
    return {
        "success": bool(execution.get("success", False)),
        "status": str(execution.get("status", "unknown")),
        "message": str(execution.get("message", "")),
        "lift_height_m": execution.get("lift_height_m"),
        "target_lift_height_m": execution.get("target_lift_height_m"),
        "video_path": video.get("path") or execution.get("video_path"),
        "video_frame_count": int(video.get("frame_count") or execution.get("video_frame_count") or 0),
    }


def _run_attempt(
    *,
    spec: dict[str, object],
    output_dir: Path,
    payload: dict[str, object],
    record_video: bool,
) -> dict[str, object]:
    backend = str(spec["backend"])
    attempt_dir = _attempt_dir(output_dir, spec)
    attempt_artifact = attempt_dir / "attempt.json"
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    video_path = attempt_dir / "attempt.mp4" if record_video else None
    attempt_dir.mkdir(parents=True, exist_ok=True)

    if backend == "mujoco":
        command = _mujoco_command(
            cfg=dict(payload.get("mujoco", {}) or {}),
            spec=spec,
            attempt_artifact=attempt_artifact,
            video_path=video_path,
        )
    elif backend == "isaac":
        command = _isaac_command(
            cfg=dict(payload.get("isaac", {}) or {}),
            spec=spec,
            attempt_artifact=attempt_artifact,
            video_path=video_path,
        )
    else:
        raise ValueError(f"Unsupported backend '{backend}'.")

    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=REPO_ROOT, env=_subprocess_env(), stdout=stdout, stderr=stderr)
    duration_s = time.perf_counter() - started
    artifact = _load_json_if_present(attempt_artifact)
    summary = _execution_summary(backend, artifact)
    if completed.returncode != 0 and summary["status"] == "artifact_missing":
        summary["status"] = "runner_failed"
        summary["message"] = f"Runner exited with code {completed.returncode}; see stderr.log."
    return {
        **spec,
        "attempt_key": _attempt_key(spec),
        "returncode": int(completed.returncode),
        "duration_s": float(duration_s),
        "attempt_artifact": str(attempt_artifact),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "command": command,
        **summary,
    }


def _relative(output_dir: Path, path_value: object) -> str:
    if path_value in ("", None):
        return ""
    try:
        return os.path.relpath(str(path_value), output_dir)
    except ValueError:
        return str(path_value)


def _write_summary_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "assembly",
        "part_id",
        "orientation_id",
        "backend",
        "grasp_id",
        "success",
        "status",
        "returncode",
        "duration_s",
        "lift_height_m",
        "target_lift_height_m",
        "video_frame_count",
        "video_path",
        "attempt_artifact",
        "stage2_json",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def _write_index_html(path: Path, *, output_dir: Path, records: list[dict[str, object]]) -> None:
    status_counts = Counter(str(record.get("status", "unknown")) for record in records)
    backend_counts = Counter(str(record.get("backend", "unknown")) for record in records)
    rows_html = []
    for record in records:
        duration_text = f"{float(record.get('duration_s', 0.0)):.1f}"
        video = _relative(output_dir, record.get("video_path"))
        video_html = ""
        if video:
            video_html = (
                f'<video controls preload="metadata" width="320" src="{html.escape(video)}"></video>'
                f'<div><a href="{html.escape(video)}">video</a></div>'
            )
        artifact = _relative(output_dir, record.get("attempt_artifact"))
        stderr = _relative(output_dir, record.get("stderr_log"))
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('assembly', '')))}</td>"
            f"<td>{html.escape(str(record.get('part_id', '')))}</td>"
            f"<td>{html.escape(str(record.get('orientation_id', '')))}</td>"
            f"<td>{html.escape(str(record.get('backend', '')))}</td>"
            f"<td>{html.escape(str(record.get('grasp_id', '')))}</td>"
            f"<td>{html.escape(str(record.get('success', '')))}</td>"
            f"<td>{html.escape(str(record.get('status', '')))}</td>"
            f"<td>{html.escape(duration_text)}</td>"
            f"<td>{html.escape(str(record.get('video_frame_count', '')))}</td>"
            f"<td>{video_html}</td>"
            f'<td><a href="{html.escape(artifact)}">artifact</a><br><a href="{html.escape(stderr)}">stderr</a></td>'
            f"<td>{html.escape(str(record.get('message', '')))}</td>"
            "</tr>"
        )
    summary = "<br>".join(
        [
            f"attempts: {len(records)}",
            f"successes: {sum(1 for record in records if record.get('success'))}",
            "backends: " + html.escape(json.dumps(dict(backend_counts), sort_keys=True)),
            "statuses: " + html.escape(json.dumps(dict(status_counts), sort_keys=True)),
        ]
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Grasp Execution Benchmark</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: white; }}
    video {{ max-width: 320px; background: #111827; }}
    .summary {{ margin: 0 0 16px; line-height: 1.5; }}
  </style>
</head>
<body>
  <h1>Grasp Execution Benchmark</h1>
  <p class="summary">{summary}</p>
  <table>
    <thead>
      <tr>
        <th>Assembly</th><th>Part</th><th>Orientation</th><th>Backend</th><th>Grasp</th>
        <th>Success</th><th>Status</th><th>Seconds</th><th>Frames</th><th>Video</th><th>Artifacts</th><th>Message</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def _apply_cli_overrides(payload: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    effective = dict(payload)
    benchmark = dict(effective.get("execution_benchmark", {}) or {})
    selection = dict(effective.get("selection", {}) or {})
    if args.output_dir is not None:
        benchmark["output_dir"] = str(args.output_dir)
    if args.backend is not None:
        benchmark["backend"] = args.backend
    if args.record_video is not None:
        benchmark["record_video"] = args.record_video
    if args.no_resume:
        benchmark["resume"] = False
    if args.assembly:
        selection["assemblies"] = args.assembly
    if args.part:
        selection["parts"] = args.part
    if args.orientation:
        selection["orientations"] = args.orientation
    if args.grasp_id:
        selection["grasp_ids"] = args.grasp_id
    if args.limit_orientations is not None:
        selection["limit_orientations"] = args.limit_orientations
    if args.max_grasps_per_orientation is not None:
        selection["max_grasps_per_orientation"] = args.max_grasps_per_orientation
    if args.limit_attempts is not None:
        selection["limit_attempts"] = args.limit_attempts
    effective["execution_benchmark"] = benchmark
    effective["selection"] = selection
    return effective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--generation-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=("mujoco", "isaac", "both"), default=None)
    parser.add_argument("--record-video", choices=("all", "none"), default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--assembly", action="append", default=[])
    parser.add_argument("--part", action="append", default=[])
    parser.add_argument("--orientation", action="append", default=[])
    parser.add_argument("--grasp-id", action="append", default=[])
    parser.add_argument("--limit-orientations", type=int, default=None)
    parser.add_argument("--max-grasps-per-orientation", type=int, default=None)
    parser.add_argument("--limit-attempts", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _apply_cli_overrides(_load_yaml(args.config), args)
    benchmark_cfg = dict(payload.get("execution_benchmark", {}) or {})
    selection = dict(payload.get("selection", {}) or {})
    output_dir = Path(str(benchmark_cfg.get("output_dir", DEFAULT_OUTPUT_DIR)))
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    generation_results_path = args.generation_results or Path(
        str(payload.get("generation_results", "artifacts/grasp_generation_benchmark/results.json"))
    )
    if not generation_results_path.is_absolute():
        generation_results_path = (REPO_ROOT / generation_results_path).resolve()
    generation_root = generation_results_path.parent
    generation_results = json.loads(generation_results_path.read_text(encoding="utf-8"))
    if not isinstance(generation_results, dict):
        raise ValueError(f"Expected object in generation results '{generation_results_path}'.")

    backends = _backend_list(benchmark_cfg.get("backend", "mujoco"))
    record_video = _record_video_enabled(benchmark_cfg.get("record_video", "all"))
    assemblies = set(_as_list(selection.get("assemblies")))
    parts = set(_as_list(selection.get("parts")))
    orientations = set(_as_list(selection.get("orientations")))
    grasp_ids = set(_as_list(selection.get("grasp_ids")))
    limit_orientations = _optional_int(selection.get("limit_orientations"))
    max_grasps = _optional_int(selection.get("max_grasps_per_orientation"))
    limit_attempts = _optional_int(selection.get("limit_attempts"))
    resume = bool(benchmark_cfg.get("resume", True)) and not bool(args.no_resume)

    if not backends:
        raise RuntimeError("No execution backends are enabled.")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(output_dir / "execution_benchmark_config.yaml", payload)

    orientation_rows = _filtered_orientation_rows(
        generation_results=generation_results,
        generation_root=generation_root,
        assemblies=assemblies,
        parts=parts,
        orientations=orientations,
        limit_orientations=limit_orientations,
    )
    specs: list[dict[str, object]] = []
    for row in orientation_rows:
        specs.extend(
            _attempt_specs_for_row(
                row=row,
                backends=backends,
                grasp_ids=grasp_ids,
                max_grasps_per_orientation=max_grasps,
            )
        )
        if limit_attempts is not None and len(specs) >= limit_attempts:
            specs = specs[:limit_attempts]
            break
    if not specs:
        raise RuntimeError("No execution attempts matched the requested filters.")

    jsonl_path = output_dir / "attempts.jsonl"
    existing_records = _jsonl_records(jsonl_path)
    existing_by_key = {str(record.get("attempt_key", "")): record for record in existing_records}
    completed_success = {
        key
        for key, record in existing_by_key.items()
        if key and (not args.rerun_failed or bool(record.get("success", False)))
    }

    records = list(existing_records)
    print(
        f"[EXEC-BENCH] attempts={len(specs)} backends={backends} record_video={record_video} "
        f"resume={resume} output_dir={output_dir}",
        flush=True,
    )
    for index, spec in enumerate(specs, start=1):
        key = _attempt_key(spec)
        if resume and key in completed_success:
            print(f"[EXEC-BENCH] skip {index}/{len(specs)} {key}", flush=True)
            continue
        print(
            f"[EXEC-BENCH] run {index}/{len(specs)} "
            f"{spec['assembly']}/{spec['part_id']} {spec['orientation_id']} "
            f"{spec['backend']} grasp={spec['grasp_id']}",
            flush=True,
        )
        record = _run_attempt(spec=spec, output_dir=output_dir, payload=payload, record_video=record_video)
        _append_jsonl(jsonl_path, record)
        records.append(record)
        print(
            f"[EXEC-BENCH] done success={record.get('success')} status={record.get('status')} "
            f"returncode={record.get('returncode')} frames={record.get('video_frame_count')}",
            flush=True,
        )

    latest_by_key = {str(record.get("attempt_key", "")): record for record in records if record.get("attempt_key")}
    final_records = list(latest_by_key.values())
    results = {
        "schema_version": 1,
        "provenance": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(args.config),
            "generation_results": str(generation_results_path),
            "python_executable": sys.executable,
            "cli_args": _json_safe(vars(args)),
        },
        "summary": {
            "attempt_count": len(final_records),
            "success_count": sum(1 for record in final_records if record.get("success")),
            "backend_counts": dict(Counter(str(record.get("backend", "unknown")) for record in final_records)),
            "status_counts": dict(Counter(str(record.get("status", "unknown")) for record in final_records)),
            "video_count": sum(1 for record in final_records if record.get("video_path")),
        },
        "attempts": final_records,
    }
    _write_json(output_dir / "results.json", results)
    _write_summary_csv(output_dir / "summary.csv", final_records)
    _write_index_html(output_dir / "index.html", output_dir=output_dir, records=final_records)
    print(f"[EXEC-BENCH] Wrote {output_dir / 'results.json'}", flush=True)
    print(f"[EXEC-BENCH] Wrote {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
