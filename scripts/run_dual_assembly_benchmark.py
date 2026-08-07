#!/usr/bin/env python3
"""Run a resumable, video-recorded benchmark over every dual assembly step."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dual_assembly_benchmark.yaml"
TERMINAL_STATUSES = {"success", "failed", "interrupted"}


def _read_mapping(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level mapping in '{path}'.")
    return payload


def _read_json_mapping(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _jsonl_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("case_id"):
            records.append(payload)
    return records


def _latest_records(records: Iterable[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    return {str(record["case_id"]): dict(record) for record in records}


def _safe_id(value: object) -> str:
    text = str(value).strip() or "unknown"
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in text)


def _float_triplet(raw: object, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{field_name} must contain exactly three values.")
    return tuple(float(value) for value in raw)  # type: ignore[return-value]


def _selected_steps(
    *,
    artifact_dir: Path,
    selected_parts: set[str],
) -> tuple[dict[str, object], ...]:
    sequence_path = artifact_dir / "assembly_sequence.json"
    payload = _read_json_mapping(sequence_path)
    if payload is None:
        raise FileNotFoundError(f"Assembly sequence does not exist: {sequence_path}")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError(f"Assembly sequence '{sequence_path}' has no steps list.")
    steps = tuple(
        dict(raw_step)
        for raw_step in raw_steps
        if isinstance(raw_step, dict)
        and bool(raw_step.get("holder_base_available", False))
        and (not selected_parts or str(raw_step.get("incoming_part_id", "")) in selected_parts)
    )
    if not steps:
        raise ValueError("No holder-active assembly steps matched the benchmark selection.")
    for step in steps:
        pair_path = artifact_dir / f"dual_grasp_pairs_{step['step_id']}.json"
        if not pair_path.is_file():
            raise FileNotFoundError(f"Stage-3 artifact is missing for benchmark step '{step['step_id']}': {pair_path}")
    return steps


def _case_specs(
    *,
    payload: Mapping[str, object],
    selected_parts: set[str] | None = None,
    selected_placements: set[str] | None = None,
    selected_orientations: set[str] | None = None,
    limit_cases: int | None = None,
) -> tuple[dict[str, object], ...]:
    benchmark = dict(payload.get("benchmark", {}) or {})
    assembly = str(benchmark.get("assembly", "plumbers_block"))
    artifact_root = (REPO_ROOT / str(benchmark.get("artifact_root", "artifacts/dual_grasp_planning"))).resolve()
    artifact_dir = artifact_root / assembly
    selection = dict(payload.get("selection", {}) or {})
    configured_parts = {str(value) for value in selection.get("incoming_parts", []) or []}
    parts = configured_parts if selected_parts is None else set(selected_parts)
    steps = _selected_steps(artifact_dir=artifact_dir, selected_parts=parts)

    base = dict(payload.get("base", {}) or {})
    assembly_xyz = _float_triplet(base.get("position_world_m", (0.55, 0.0, -0.03)), field_name="base.position_world_m")
    assembly_yaw_deg = float(base.get("yaw_deg", 0.0))
    if abs(assembly_yaw_deg) > 1.0e-9:
        raise ValueError("The dual assembly benchmark requires an upright, zero-yaw base.")

    placements = payload.get("placements")
    orientations = payload.get("orientations")
    if not isinstance(placements, list) or not placements:
        raise ValueError("dual benchmark placements must be a non-empty list.")
    if not isinstance(orientations, list) or not orientations:
        raise ValueError("dual benchmark orientations must be a non-empty list.")

    specs: list[dict[str, object]] = []
    for step in steps:
        part_id = str(step["incoming_part_id"])
        step_id = str(step["step_id"])
        for raw_placement in placements:
            if not isinstance(raw_placement, dict):
                raise ValueError("Each benchmark placement must be a mapping.")
            placement_id = _safe_id(raw_placement.get("id", "placement"))
            if selected_placements is not None and placement_id not in selected_placements:
                continue
            pickup_x = float(raw_placement["x"])
            pickup_y = float(raw_placement["y"])
            if pickup_x <= 0.0:
                raise ValueError(f"Placement '{placement_id}' is not in front of the robots: x={pickup_x}.")
            inserter_arm = "lbr_one" if pickup_y < assembly_xyz[1] else "lbr_two"
            holder_arm = "lbr_two" if inserter_arm == "lbr_one" else "lbr_one"
            for raw_orientation in orientations:
                if not isinstance(raw_orientation, dict):
                    raise ValueError("Each benchmark orientation must be a mapping.")
                orientation_id = _safe_id(raw_orientation.get("id", "orientation"))
                if selected_orientations is not None and orientation_id not in selected_orientations:
                    continue
                rpy = _float_triplet(
                    raw_orientation.get("rpy_deg", (0.0, 0.0, 0.0)),
                    field_name=f"orientations.{orientation_id}.rpy_deg",
                )
                case_id = _safe_id(f"{step_id}__part_{part_id}__{placement_id}__{orientation_id}")
                specs.append(
                    {
                        "case_id": case_id,
                        "assembly": assembly,
                        "step_id": step_id,
                        "step_index": int(step.get("step_index", len(specs) + 1)),
                        "incoming_part_id": part_id,
                        "placement_id": placement_id,
                        "pickup_x": pickup_x,
                        "pickup_y": pickup_y,
                        "orientation_id": orientation_id,
                        "pickup_roll_deg": rpy[0],
                        "pickup_pitch_deg": rpy[1],
                        "pickup_yaw_deg": rpy[2],
                        "assembly_x": assembly_xyz[0],
                        "assembly_y": assembly_xyz[1],
                        "assembly_z": assembly_xyz[2],
                        "assembly_yaw_deg": assembly_yaw_deg,
                        "floor_z": float(base.get("floor_z_world_m", assembly_xyz[2])),
                        "inserter_arm": inserter_arm,
                        "holder_arm": holder_arm,
                    }
                )
    if not specs:
        raise ValueError("No benchmark cases matched the requested part, placement, and orientation filters.")
    effective_limit = limit_cases
    if effective_limit is None and selection.get("limit_cases") not in (None, ""):
        effective_limit = int(selection["limit_cases"])
    if effective_limit is not None:
        if effective_limit < 1:
            raise ValueError("limit_cases must be at least one.")
        specs = specs[:effective_limit]
    return tuple(specs)


def _output_dir(payload: Mapping[str, object], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    benchmark = dict(payload.get("benchmark", {}) or {})
    raw = Path(str(benchmark.get("output_dir", "artifacts/dual_assembly_benchmark/plumbers_block")))
    return raw.expanduser().resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def _case_paths(output_dir: Path, spec: Mapping[str, object]) -> dict[str, Path]:
    case_dir = output_dir / "cases" / str(spec["case_id"])
    return {
        "case_dir": case_dir,
        "plan": case_dir / "plan.json",
        "attempt": case_dir / "attempt.json",
        "video": case_dir / "scene.webm",
        "image": case_dir / "failure_scene.svg",
        "log": case_dir / "run.log",
        "case": case_dir / "case.json",
    }


def _command(
    *,
    payload: Mapping[str, object],
    spec: Mapping[str, object],
    paths: Mapping[str, Path],
) -> list[str]:
    benchmark = dict(payload.get("benchmark", {}) or {})
    physics = dict(payload.get("physics", {}) or {})
    command = [
        str(REPO_ROOT / "run_simple_dual_robot.sh"),
        "--mode",
        "sim",
        "--assembly",
        str(spec["assembly"]),
        "--incoming-part-id",
        str(spec["incoming_part_id"]),
        "--max-pair-attempts",
        str(benchmark.get("max_pair_attempts", 256)),
        "--joint-rank-candidates",
        str(benchmark.get("joint_rank_candidates", 8)),
        "--assembly-x",
        str(spec["assembly_x"]),
        "--assembly-y",
        str(spec["assembly_y"]),
        "--assembly-z",
        str(spec["assembly_z"]),
        "--assembly-yaw-deg",
        str(spec["assembly_yaw_deg"]),
        "--pickup-x",
        str(spec["pickup_x"]),
        "--pickup-y",
        str(spec["pickup_y"]),
        "--pickup-roll-deg",
        str(spec["pickup_roll_deg"]),
        "--pickup-pitch-deg",
        str(spec["pickup_pitch_deg"]),
        "--pickup-yaw-deg",
        str(spec["pickup_yaw_deg"]),
        "--inserter-arm",
        "auto",
        "--floor-z",
        str(spec["floor_z"]),
        "--plan-output",
        str(paths["plan"]),
        "--attempt-output",
        str(paths["attempt"]),
        "--record-video",
        str(paths["video"]),
        "--static-friction",
        str(physics.get("static_friction", 5.0)),
        "--dynamic-friction",
        str(physics.get("dynamic_friction", 4.0)),
        "--gripper-effort-limit",
        str(physics.get("gripper_effort_limit", 200.0)),
        "--critical-damping-ratio",
        str(physics.get("critical_damping_ratio", 1.0)),
        "--gripper-close-duration-s",
        str(physics.get("gripper_close_duration_s", 3.0)),
        "--finger-contact-min-force-n",
        str(physics.get("finger_contact_min_force_n", 0.25)),
        "--gripper-contact-preload-m",
        str(physics.get("gripper_contact_preload_m", 0.0004)),
        "--ros-domain-id",
        str(benchmark.get("ros_domain_id", 0)),
        "--headless",
        "--no-planning-debug-gui",
    ]
    if benchmark.get("isaac_python"):
        command.extend(["--isaac-python", str(benchmark["isaac_python"])])
    return command


def _tail(path: Path, *, max_characters: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    return text[-max_characters:]


def _attempt_result(path: Path) -> dict[str, object]:
    payload = _read_json_mapping(path)
    if payload is None:
        return {}
    result = payload.get("result")
    return dict(result) if isinstance(result, dict) else {}


def _plan_summary(path: Path) -> dict[str, object]:
    payload = _read_json_mapping(path)
    if payload is None:
        return {}
    return {
        "pair_id": str(payload.get("pair_id", "")),
        "transition_id": str(payload.get("transition_id", "")),
        "execution_candidate_id": str(payload.get("execution_candidate_id", "")),
        "selection_score": payload.get("selection_score"),
    }


def _meaningful_log_failure(path: Path, *, returncode: int | None = None) -> str:
    """Return the last actionable failure instead of runner cleanup output."""

    lines = [line.strip() for line in _tail(path, max_characters=20000).splitlines() if line.strip()]
    preferred_prefixes = (
        "ValueError:",
        "RuntimeError:",
        "FileNotFoundError:",
        "AssertionError:",
        "[DUAL-SIM-ISAAC] failure:",
    )
    for line in reversed(lines):
        if line.startswith(preferred_prefixes) or (line.startswith("[DUAL-SIM-PLAN]") and "failed" in line.lower()):
            return line.removeprefix("[DUAL-SIM-ISAAC] failure:").strip()
    ignored_fragments = (
        "Stopping MoveIt stack",
        "Stopping background ROS",
        "Simulation App Shutting Down",
    )
    for line in reversed(lines):
        if not any(fragment in line for fragment in ignored_fragments):
            return line
    suffix = "" if returncode is None else f" with code {returncode}"
    return f"Runner exited{suffix} before writing an attempt artifact."


def _failure_phase(
    message: str,
    *,
    attempt_result: Mapping[str, object] | None = None,
    has_plan: bool = False,
) -> tuple[str, str]:
    """Classify a failure into the user-visible assembly phase."""

    normalized = message.lower()
    result = dict(attempt_result or {})
    if any(
        token in normalized
        for token in (
            "grounded pickup pose",
            "pickup-floor",
            "pickup floor",
            "no compatible grasp pairs",
            "inserter_pickup_pregrasp: ik",
            "inserter_pickup_grasp: ik",
        )
    ):
        return "incoming_grasp_planning", "Incoming-part grasp planning"
    if any(
        token in normalized
        for token in (
            "moveit could not plan any",
            "could not plan any of",
            "planning failed with code",
            "ik preflight",
        )
    ):
        return "moveit_candidate_planning", "MoveIt candidate planning"
    if any(
        token in normalized
        for token in (
            "holder_pregrasp",
            "holder_grasp",
            "holder did not establish",
            "holder close",
            "base grasp",
        )
    ):
        return "holder_base_grasp", "Holder/base grasp"
    if any(
        token in normalized
        for token in (
            "inserter did not establish",
            "incoming grasp",
            "inserter close",
            "selected-width contact",
        )
    ):
        return "incoming_part_grasp", "Incoming-part grasp"
    if any(
        token in normalized
        for token in (
            "pre-insertion",
            "preinsertion",
            "did not travel with the inserter",
            "transport",
            "base moved",
        )
    ):
        return "transition", "Transport to pre-insertion"
    if any(token in normalized for token in ("moveit stack", "service unavailable", "failed to connect", "ros domain")):
        return "setup", "MoveIt/Isaac setup"

    if result:
        if "holder_close" not in result:
            return "holder_base_grasp", "Holder/base grasp"
        if "inserter_close" not in result:
            return "incoming_part_grasp", "Incoming-part grasp"
        if "after_inserter_preinsertion" not in result:
            return "transition", "Transport to pre-insertion"
        return "execution_validation", "Final execution validation"
    if has_plan:
        return "execution_start", "Isaac execution startup"
    return "planning", "Grasp/transition planning"


def _quaternion_rotation_matrix(raw_xyzw: object) -> np.ndarray:
    values = np.asarray(raw_xyzw, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        return np.eye(3, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=float)
    x, y, z, w = values / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _rpy_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = (math.radians(float(value)) for value in (roll_deg, pitch_deg, yaw_deg))
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)), dtype=float)
    ry = np.asarray(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), dtype=float)
    rz = np.asarray(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)), dtype=float)
    return rz @ ry @ rx


def _load_obj_mesh(path: Path, *, scale: float) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.strip().split()
        if len(fields) >= 4 and fields[0] == "v":
            vertices.append(tuple(float(value) * float(scale) for value in fields[1:4]))
        elif len(fields) >= 4 and fields[0] == "f":
            face = [int(value.split("/", 1)[0]) - 1 for value in fields[1:]]
            triangles.extend((face[0], face[index], face[index + 1]) for index in range(1, len(face) - 1))
    if not vertices or not triangles:
        raise ValueError(f"Could not load drawable geometry from '{path}'.")
    return np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=int)


def _artifact_dir(payload: Mapping[str, object], spec: Mapping[str, object]) -> Path:
    benchmark = dict(payload.get("benchmark", {}) or {})
    root = Path(str(benchmark.get("artifact_root", "artifacts/dual_grasp_planning"))).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root.resolve() / str(spec["assembly"])


def _sequence_payload(payload: Mapping[str, object], spec: Mapping[str, object]) -> dict[str, object]:
    sequence = _read_json_mapping(_artifact_dir(payload, spec) / "assembly_sequence.json")
    if sequence is None:
        raise FileNotFoundError("The assembly sequence is unavailable for failure-scene rendering.")
    return sequence


def _incoming_source_frame(
    *,
    payload: Mapping[str, object],
    spec: Mapping[str, object],
    plan: Mapping[str, object] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, float]:
    sequence = _sequence_payload(payload, spec)
    part_payload = dict(dict(sequence["parts"])[str(spec["incoming_part_id"])])  # type: ignore[arg-type]
    mesh_path = REPO_ROOT / str(part_payload["mesh_path"])
    mesh_scale = float(sequence.get("mesh_scale", 0.01))
    source_position: object = (0.0, 0.0, 0.0)
    source_orientation: object = (0.0, 0.0, 0.0, 1.0)
    pickup_position = np.asarray((spec["pickup_x"], spec["pickup_y"], spec["floor_z"]), dtype=float)
    pickup_rotation = _rpy_rotation_matrix(
        float(spec["pickup_roll_deg"]),
        float(spec["pickup_pitch_deg"]),
        float(spec["pickup_yaw_deg"]),
    )

    if plan is not None:
        incoming = dict(dict(plan["objects"])["incoming"])  # type: ignore[arg-type]
        source_pose = dict(incoming.get("source_pose_assembly", {}) or {})
        pickup_pose = dict(incoming.get("pickup_source_pose_world", {}) or {})
        source_position = source_pose.get("position_world_m", source_position)
        source_orientation = source_pose.get("orientation_xyzw_world", source_orientation)
        pickup_position = np.asarray(pickup_pose.get("position_world_m", pickup_position), dtype=float)
        pickup_rotation = _quaternion_rotation_matrix(pickup_pose.get("orientation_xyzw_world", (0.0, 0.0, 0.0, 1.0)))
        mesh_path = Path(str(incoming.get("mesh_path", mesh_path))).expanduser().resolve()
        mesh_scale = float(incoming.get("mesh_scale", mesh_scale))
    else:
        library = _read_json_mapping(_artifact_dir(payload, spec) / f"inserter_candidates_{spec['step_id']}.json")
        if library is not None:
            target = dict(library.get("target", {}) or {})
            source_position = target.get("source_frame_origin_obj_world", source_position)
            source_orientation = target.get("source_frame_orientation_xyzw_obj_world", source_orientation)

    vertices, triangles = _load_obj_mesh(mesh_path, scale=mesh_scale)
    source_rotation = _quaternion_rotation_matrix(source_orientation)
    local_vertices = (vertices - np.asarray(source_position, dtype=float)) @ source_rotation
    world_vertices = local_vertices @ pickup_rotation.T
    if plan is None:
        pickup_position[2] = float(spec["floor_z"]) - float(np.min(world_vertices[:, 2]))
    world_vertices += pickup_position
    return world_vertices, triangles, pickup_position, mesh_path, mesh_scale


def _project_scene(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = np.asarray((1.65, -1.25, 1.15), dtype=float)
    target = np.asarray((0.48, 0.0, 0.15), dtype=float)
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0), dtype=float))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = points - target
    return relative @ right, relative @ up, relative @ forward


def _render_failure_scene(
    *,
    path: Path,
    payload: Mapping[str, object],
    spec: Mapping[str, object],
    plan_path: Path,
    failure_label: str,
    message: str,
) -> None:
    """Render the planned initial world scene when Isaac produced no video."""

    plan = _read_json_mapping(plan_path)
    if plan is not None and not isinstance(plan.get("objects"), dict):
        plan = None
    sequence = _sequence_payload(payload, spec)
    steps = [dict(value) for value in sequence.get("steps", []) if isinstance(value, dict)]
    step = next((value for value in steps if str(value.get("step_id")) == str(spec["step_id"])), None)
    if step is None:
        raise ValueError(f"Assembly step '{spec['step_id']}' is absent from the sequence.")
    parts = dict(sequence["parts"])
    mesh_scale = float(sequence.get("mesh_scale", 0.01))
    assembly_rotation = _rpy_rotation_matrix(0.0, 0.0, float(spec["assembly_yaw_deg"]))
    assembly_translation = np.asarray((spec["assembly_x"], spec["assembly_y"], spec["assembly_z"]), dtype=float)
    drawables: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    palette = ("#30d6a4", "#54c7e8", "#5b8ff9", "#66d184")
    for index, part_id in enumerate(step.get("assembled_part_ids_before", [])):
        part_payload = dict(parts[str(part_id)])
        mesh_path = (REPO_ROOT / str(part_payload["mesh_path"])).resolve()
        vertices, triangles = _load_obj_mesh(mesh_path, scale=mesh_scale)
        vertices = vertices @ assembly_rotation.T + assembly_translation
        drawables.append((f"assembled part {part_id}", palette[index % len(palette)], vertices, triangles))
    incoming_vertices, incoming_triangles, pickup_position, _, _ = _incoming_source_frame(
        payload=payload,
        spec=spec,
        plan=plan,
    )
    drawables.append((f"incoming part {spec['incoming_part_id']}", "#ff9e43", incoming_vertices, incoming_triangles))

    holder_base = np.asarray((0.0, 0.42 if spec["holder_arm"] == "lbr_two" else -0.42, 0.0), dtype=float)
    inserter_base = np.asarray((0.0, 0.42 if spec["inserter_arm"] == "lbr_two" else -0.42, 0.0), dtype=float)
    if plan is not None:
        layout = dict(plan.get("layout", {}) or {})
        holder_base = np.asarray(layout.get("holder_base_world_m", holder_base), dtype=float)
        inserter_base = np.asarray(layout.get("inserter_base_world_m", inserter_base), dtype=float)
    floor_corners = np.asarray(
        (
            (-0.10, -0.62, spec["floor_z"]),
            (0.90, -0.62, spec["floor_z"]),
            (0.90, 0.62, spec["floor_z"]),
            (-0.10, 0.62, spec["floor_z"]),
        ),
        dtype=float,
    )
    all_points = np.concatenate(
        [floor_corners, holder_base[None, :], inserter_base[None, :], pickup_position[None, :]]
        + [vertices for _, _, vertices, _ in drawables],
        axis=0,
    )
    projected_x, projected_y, _ = _project_scene(all_points)
    content_width, content_height = 900.0, 385.0
    x_span = max(float(np.ptp(projected_x)), 1.0e-6)
    y_span = max(float(np.ptp(projected_y)), 1.0e-6)
    scale = min(content_width / x_span, content_height / y_span)
    center_x = 480.0 - scale * float((np.min(projected_x) + np.max(projected_x)) * 0.5)
    center_y = 270.0 + scale * float((np.min(projected_y) + np.max(projected_y)) * 0.5)

    def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px, py, depth = _project_scene(points)
        return np.column_stack((center_x + scale * px, center_y - scale * py)), depth

    floor_xy, _ = project(floor_corners)
    svg: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">',
        '<rect width="960" height="540" fill="#0a0f18"/>',
        '<defs><filter id="shadow"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-opacity=".45"/></filter></defs>',
        f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in floor_xy)}" fill="#18283a" stroke="#36506b" stroke-width="2"/>',
    ]
    faces: list[tuple[float, str]] = []
    for label, color, vertices, triangles in drawables:
        vertex_xy, vertex_depth = project(vertices)
        stride = max(1, int(math.ceil(len(triangles) / 650)))
        for triangle in triangles[::stride]:
            polygon = vertex_xy[triangle]
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in polygon)
            faces.append(
                (
                    float(np.mean(vertex_depth[triangle])),
                    f'<polygon points="{points}" fill="{color}" fill-opacity=".72" stroke="#07101b" stroke-opacity=".48" stroke-width=".55"/>',
                )
            )
        center_xy, _ = project(np.mean(vertices, axis=0, keepdims=True))
        svg.append(
            f'<text x="{center_xy[0, 0]:.1f}" y="{center_xy[0, 1]:.1f}" fill="{color}" '
            f'font-family="system-ui,sans-serif" font-size="15" font-weight="700">{html.escape(label)}</text>'
        )
    svg[4:4] = [shape for _, shape in sorted(faces, key=lambda item: item[0], reverse=True)]

    for label, color, point in (
        (f"{spec['holder_arm']} holder", "#ffd166", holder_base),
        (f"{spec['inserter_arm']} inserter", "#b18cff", inserter_base),
    ):
        xy, _ = project(point[None, :])
        svg.extend(
            (
                f'<circle cx="{xy[0, 0]:.1f}" cy="{xy[0, 1]:.1f}" r="9" fill="{color}" filter="url(#shadow)"/>',
                f'<text x="{xy[0, 0] + 13:.1f}" y="{xy[0, 1] + 5:.1f}" fill="{color}" font-family="system-ui,sans-serif" font-size="14" font-weight="700">{html.escape(label)}</text>',
            )
        )
    svg.extend(
        (
            '<rect x="18" y="16" width="924" height="68" rx="12" fill="#111b29" stroke="#2a3b52"/>',
            f'<text x="36" y="43" fill="#ff7188" font-family="system-ui,sans-serif" font-size="18" font-weight="800">Failed at: {html.escape(failure_label)}</text>',
            f'<text x="36" y="68" fill="#b6c5d8" font-family="system-ui,sans-serif" font-size="13">{html.escape(message[:145])}</text>',
            f'<text x="24" y="520" fill="#8ea1b8" font-family="system-ui,sans-serif" font-size="12">Planned initial scene · pickup ({float(spec["pickup_x"]):.2f}, {float(spec["pickup_y"]):.2f}) m · RPY ({float(spec["pickup_roll_deg"]):.0f}, {float(spec["pickup_pitch_deg"]):.0f}, {float(spec["pickup_yaw_deg"]):.0f})°</text>',
            "</svg>",
        )
    )
    _atomic_write_text(path, "\n".join(svg) + "\n")


def _terminate_process_group(process: subprocess.Popen[str], *, timeout_s: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=timeout_s)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)


def _run_case(
    *,
    payload: Mapping[str, object],
    spec: Mapping[str, object],
    paths: Mapping[str, Path],
) -> tuple[dict[str, object], bool]:
    paths["case_dir"].mkdir(parents=True, exist_ok=True)
    for key in ("plan", "attempt", "video", "image", "case"):
        paths[key].unlink(missing_ok=True)
    command = _command(payload=payload, spec=spec, paths=paths)
    started_at = time.time()
    interrupted = False
    with paths["log"].open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = int(process.wait())
        except KeyboardInterrupt:
            interrupted = True
            print("\n[DUAL-BENCH] Interrupt received; stopping the active case cleanly...", flush=True)
            _terminate_process_group(process)
            returncode = int(process.returncode if process.returncode is not None else 130)

    attempt_result = _attempt_result(paths["attempt"])
    plan_summary = _plan_summary(paths["plan"])
    video_exists = paths["video"].is_file() and paths["video"].stat().st_size > 0
    if interrupted:
        status = "interrupted"
        success = False
        message = "Benchmark interrupted by the user during this case."
    elif attempt_result:
        success = bool(attempt_result.get("success", False)) and returncode == 0
        status = "success" if success else "failed"
        message = str(attempt_result.get("message", ""))
    else:
        success = False
        status = "failed"
        message = _meaningful_log_failure(
            paths["log"],
            returncode=returncode,
        )
    failure_stage, failure_phase_label = (
        ("", "")
        if success
        else _failure_phase(
            message,
            attempt_result=attempt_result,
            has_plan=paths["plan"].is_file(),
        )
    )
    image_path = ""
    image_error = ""
    if not success and not video_exists:
        try:
            _render_failure_scene(
                path=paths["image"],
                payload=payload,
                spec=spec,
                plan_path=paths["plan"],
                failure_label=failure_phase_label,
                message=message,
            )
            image_path = str(paths["image"])
        except (FileNotFoundError, KeyError, TypeError, ValueError, OSError) as exc:
            image_error = str(exc)
    record = {
        **dict(spec),
        **plan_summary,
        "status": status,
        "success": success,
        "failure_stage": failure_stage,
        "failure_phase_label": failure_phase_label,
        "result_status": str(attempt_result.get("status", "")),
        "message": message,
        "returncode": returncode,
        "duration_s": max(0.0, time.time() - started_at),
        "started_at": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        "finished_at": datetime.now(tz=timezone.utc).isoformat(),
        "plan_json": str(paths["plan"]) if paths["plan"].is_file() else "",
        "attempt_json": str(paths["attempt"]) if paths["attempt"].is_file() else "",
        "video_path": str(paths["video"]) if video_exists else "",
        "image_path": image_path,
        "image_error": image_error,
        "log_path": str(paths["log"]),
        "command": command,
    }
    _atomic_write_json(paths["case"], record)
    return record, interrupted


def _relative(output_dir: Path, raw_path: object) -> str:
    if not raw_path:
        return ""
    path = Path(str(raw_path))
    try:
        return os.path.relpath(path, output_dir)
    except ValueError:
        return str(path)


def _video_mime(path: str) -> str:
    return "video/webm" if Path(path).suffix.lower() == ".webm" else "video/mp4"


def _ffmpeg_executable(payload: Mapping[str, object]) -> Path:
    configured = os.environ.get("IMAGEIO_FFMPEG_EXE", "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file():
            return candidate
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return Path(system_ffmpeg).resolve()
    benchmark = dict(payload.get("benchmark", {}) or {})
    isaac_python = Path(str(benchmark.get("isaac_python", ""))).expanduser()
    isaac_sim = isaac_python.parent / "_isaac_sim"
    candidates = sorted(isaac_sim.glob("kit/python/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"))
    if candidates:
        return candidates[-1].resolve()
    raise FileNotFoundError(
        "Could not find ffmpeg. Set IMAGEIO_FFMPEG_EXE or install ffmpeg before repairing old videos."
    )


def _convert_mp4_to_webm(*, ffmpeg: Path, source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    temporary.unlink(missing_ok=True)
    subprocess.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libvpx",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-b:v",
            "2M",
            "-pix_fmt",
            "yuv420p",
            str(temporary),
        ],
        check=True,
    )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no browser video for '{source}'.")
    os.replace(temporary, target)


def _repair_browser_videos(
    *,
    payload: Mapping[str, object],
    output_dir: Path,
    specs: tuple[dict[str, object], ...],
    latest: dict[str, dict[str, object]],
    events_path: Path,
) -> tuple[int, int]:
    ffmpeg = _ffmpeg_executable(payload)
    converted = 0
    skipped = 0
    for index, spec in enumerate(specs, start=1):
        case_id = str(spec["case_id"])
        record = latest.get(case_id)
        if not record:
            skipped += 1
            continue
        raw_video = str(record.get("video_path", ""))
        source = Path(raw_video) if raw_video else _case_paths(output_dir, spec)["case_dir"] / "scene.mp4"
        if not source.is_file() or source.stat().st_size == 0:
            skipped += 1
            continue
        if source.suffix.lower() == ".webm":
            skipped += 1
            continue
        target = source.with_suffix(".webm")
        print(f"[DUAL-BENCH] Convert video {index}/{len(specs)} {case_id}", flush=True)
        if not target.is_file() or target.stat().st_size == 0:
            _convert_mp4_to_webm(ffmpeg=ffmpeg, source=source, target=target)
        updated = {
            **record,
            "video_path": str(target),
            "original_video_path": str(source),
        }
        case_path = _case_paths(output_dir, spec)["case"]
        if case_path.is_file():
            _atomic_write_json(case_path, updated)
        _append_jsonl(events_path, updated)
        latest[case_id] = updated
        converted += 1
        _refresh_outputs(output_dir=output_dir, specs=specs, latest=latest)
    return converted, skipped


def _repair_failure_evidence(
    *,
    payload: Mapping[str, object],
    output_dir: Path,
    specs: tuple[dict[str, object], ...],
    latest: dict[str, dict[str, object]],
    events_path: Path,
) -> tuple[int, int]:
    """Backfill precise failure labels and scene stills for completed cases."""

    repaired = 0
    skipped = 0
    for index, spec in enumerate(specs, start=1):
        case_id = str(spec["case_id"])
        record = latest.get(case_id)
        if not record or bool(record.get("success", False)):
            skipped += 1
            continue
        paths = _case_paths(output_dir, spec)
        attempt_result = _attempt_result(paths["attempt"])
        message = str(attempt_result.get("message", "")) if attempt_result else _meaningful_log_failure(paths["log"])
        failure_stage, failure_phase_label = _failure_phase(
            message,
            attempt_result=attempt_result,
            has_plan=paths["plan"].is_file(),
        )
        raw_video = str(record.get("video_path", ""))
        video_exists = bool(raw_video) and Path(raw_video).is_file() and Path(raw_video).stat().st_size > 0
        image_path = ""
        image_error = ""
        if not video_exists:
            print(f"[DUAL-BENCH] Render failure scene {index}/{len(specs)} {case_id}", flush=True)
            try:
                _render_failure_scene(
                    path=paths["image"],
                    payload=payload,
                    spec=spec,
                    plan_path=paths["plan"],
                    failure_label=failure_phase_label,
                    message=message,
                )
                image_path = str(paths["image"])
            except (FileNotFoundError, KeyError, TypeError, ValueError, OSError) as exc:
                image_error = str(exc)
        updated = {
            **record,
            "message": message,
            "failure_stage": failure_stage,
            "failure_phase_label": failure_phase_label,
            "image_path": image_path,
            "image_error": image_error,
        }
        _atomic_write_json(paths["case"], updated)
        _append_jsonl(events_path, updated)
        latest[case_id] = updated
        repaired += 1
    _refresh_outputs(output_dir=output_dir, specs=specs, latest=latest)
    return repaired, skipped


def _group_rows(records: list[dict[str, object]], field: str) -> str:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if str(record.get("status", "")) in TERMINAL_STATUSES:
            grouped[str(record.get(field, "unknown"))].append(record)
    rows = []
    for key, values in sorted(grouped.items()):
        completed = len(values)
        successes = sum(bool(value.get("success", False)) for value in values)
        rate = 100.0 * successes / completed if completed else 0.0
        rows.append(f"<tr><td>{html.escape(key)}</td><td>{completed}</td><td>{successes}</td><td>{rate:.1f}%</td></tr>")
    return "".join(rows) or '<tr><td colspan="4">No completed cases yet.</td></tr>'


def _write_html(
    path: Path,
    *,
    specs: tuple[dict[str, object], ...],
    latest: Mapping[str, Mapping[str, object]],
    output_dir: Path,
) -> None:
    records = [dict(latest.get(str(spec["case_id"]), {**spec, "status": "pending"})) for spec in specs]
    terminal = [record for record in records if str(record.get("status")) in TERMINAL_STATUSES]
    successes = sum(bool(record.get("success", False)) for record in terminal)
    failures = sum(str(record.get("status")) == "failed" for record in terminal)
    interrupted = sum(str(record.get("status")) == "interrupted" for record in terminal)
    durations = [float(record.get("duration_s", 0.0)) for record in terminal if record.get("duration_s") is not None]
    median_duration = statistics.median(durations) if durations else 0.0
    cards = []
    for record in records:
        status = str(record.get("status", "pending"))
        video = _relative(output_dir, record.get("video_path"))
        image = _relative(output_dir, record.get("image_path"))
        plan = _relative(output_dir, record.get("plan_json"))
        attempt = _relative(output_dir, record.get("attempt_json"))
        log = _relative(output_dir, record.get("log_path"))
        links = " ".join(
            f'<a href="{html.escape(target)}">{label}</a>'
            for label, target in (("plan", plan), ("attempt", attempt), ("log", log))
            if target
        )
        if video:
            media_html = (
                f'<video controls preload="metadata"><source src="{html.escape(video)}" '
                f'type="{_video_mime(video)}"></video>'
            )
        elif image:
            media_html = (
                f'<img class="failure-image" src="{html.escape(image)}" '
                f'alt="Planned scene at {html.escape(str(record.get("failure_phase_label", "failure")))}">'
            )
        else:
            media_html = '<div class="no-video">No recording or scene image</div>'
        message = str(record.get("message", ""))
        failure_phase = str(record.get("failure_phase_label", ""))
        failure_row = f"<dt>failed at</dt><dd>{html.escape(failure_phase)}</dd>" if failure_phase else ""
        cards.append(
            f'<article class="case {html.escape(status)}" '
            f'data-part="{html.escape(str(record.get("incoming_part_id", "")))}" '
            f'data-status="{html.escape(status)}">'
            f"{media_html}"
            f'<div class="case-body"><div class="case-title">Part {html.escape(str(record.get("incoming_part_id", "")))} · '
            f"{html.escape(str(record.get('placement_id', '')))} · "
            f"{html.escape(str(record.get('orientation_id', '')))}</div>"
            f'<div class="badge">{html.escape(status)}</div>'
            f"<dl><dt>arms</dt><dd>{html.escape(str(record.get('holder_arm', '')))} holds · "
            f"{html.escape(str(record.get('inserter_arm', '')))} inserts</dd>"
            f"<dt>pickup</dt><dd>({float(record.get('pickup_x', 0.0)):.2f}, "
            f"{float(record.get('pickup_y', 0.0)):.2f}) m</dd>"
            f"<dt>RPY</dt><dd>({float(record.get('pickup_roll_deg', 0.0)):.0f}, "
            f"{float(record.get('pickup_pitch_deg', 0.0)):.0f}, "
            f"{float(record.get('pickup_yaw_deg', 0.0)):.0f})°</dd>"
            f"<dt>pair</dt><dd>{html.escape(str(record.get('pair_id', '')))}</dd>"
            f"{failure_row}<dt>time</dt><dd>{float(record.get('duration_s', 0.0)):.1f} s</dd></dl>"
            f'<p class="message">{html.escape(message)}</p><div class="links">{links}</div></div></article>'
        )
    part_options = "".join(
        f'<option value="{html.escape(part)}">Part {html.escape(part)}</option>'
        for part in sorted({str(spec["incoming_part_id"]) for spec in specs})
    )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dual-Arm Assembly Benchmark</title><style>
:root{{--bg:#0a0f18;--panel:#121a27;--line:#27364a;--text:#e9f0fa;--muted:#93a4ba;--green:#3ddc97;--red:#ff647c;--amber:#ffca6a;--blue:#69a8ff}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 20% 0,#18253a 0,var(--bg) 42%);color:var(--text);font:14px Inter,system-ui,sans-serif}}
header{{padding:30px clamp(18px,4vw,64px) 18px}} h1{{font-size:clamp(28px,4vw,46px);margin:0 0 8px}} .subtitle,.muted{{color:var(--muted)}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;padding:0 clamp(18px,4vw,64px) 24px}} .stat{{background:linear-gradient(145deg,#172235,#101722);border:1px solid var(--line);border-radius:16px;padding:17px}} .stat strong{{display:block;font-size:27px;margin-top:5px}}
.content{{padding:0 clamp(18px,4vw,64px) 50px}} .controls{{position:sticky;top:0;z-index:5;background:#0a0f18e8;backdrop-filter:blur(10px);padding:12px 0;display:flex;gap:10px;flex-wrap:wrap}} select{{background:var(--panel);color:var(--text);border:1px solid var(--line);padding:9px 12px;border-radius:9px}}
.tables{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:16px 0 28px}} .table-panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}} h2{{margin:0 0 12px}} table{{width:100%;border-collapse:collapse}} th,td{{padding:7px;border-bottom:1px solid #263346;text-align:left}}
   .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}} .case{{background:var(--panel);border:1px solid var(--line);border-radius:15px;overflow:hidden;box-shadow:0 16px 40px #0005}} .case.success{{border-color:#2c7d5d}} .case.failed{{border-color:#8f3345}} .case.interrupted{{border-color:#9b7733}} video,.no-video,.failure-image{{width:100%;aspect-ratio:16/9;background:#070a0f;display:flex;align-items:center;justify-content:center;color:var(--muted);object-fit:contain}} .case-body{{padding:14px}} .case-title{{font-weight:700;font-size:16px;padding-right:80px}} .badge{{float:right;margin-top:-22px;padding:4px 9px;border-radius:999px;background:#26364d}} .success .badge{{color:var(--green)}} .failed .badge{{color:var(--red)}} .interrupted .badge{{color:var(--amber)}} dl{{display:grid;grid-template-columns:65px 1fr;gap:5px;margin:12px 0}} dt{{color:var(--muted)}} dd{{margin:0;overflow-wrap:anywhere}} .message{{color:var(--muted);min-height:38px}} a{{color:var(--blue);text-decoration:none}} .links{{display:flex;gap:12px}} .hidden{{display:none}}
</style></head><body><header><h1>Dual-Arm Assembly Benchmark</h1><div class="subtitle">Incremental results · base fixed upright at workspace center · nearest arm inserts</div></header>
<section class="stats"><div class="stat">Progress<strong>{len(terminal)} / {len(specs)}</strong></div><div class="stat">Success<strong>{successes}</strong></div><div class="stat">Failed<strong>{failures}</strong></div><div class="stat">Interrupted<strong>{interrupted}</strong></div><div class="stat">Success rate<strong>{(100.0 * successes / len(terminal) if terminal else 0.0):.1f}%</strong></div><div class="stat">Median runtime<strong>{median_duration:.0f}s</strong></div></section>
<main class="content"><div class="controls"><select id="part"><option value="">All parts</option>{part_options}</select><select id="status"><option value="">All statuses</option><option>success</option><option>failed</option><option>interrupted</option><option>running</option><option>pending</option></select><span class="muted">Updated {html.escape(datetime.now().astimezone().isoformat(timespec="seconds"))}</span></div>
<div class="tables"><section class="table-panel"><h2>By part</h2><table><tr><th>Part</th><th>Runs</th><th>Pass</th><th>Rate</th></tr>{_group_rows(records, "incoming_part_id")}</table></section><section class="table-panel"><h2>By placement</h2><table><tr><th>Placement</th><th>Runs</th><th>Pass</th><th>Rate</th></tr>{_group_rows(records, "placement_id")}</table></section><section class="table-panel"><h2>By orientation</h2><table><tr><th>Orientation</th><th>Runs</th><th>Pass</th><th>Rate</th></tr>{_group_rows(records, "orientation_id")}</table></section></div>
<section class="grid">{"".join(cards)}</section></main><script>
const part=document.querySelector('#part'),status=document.querySelector('#status'); function apply(){{document.querySelectorAll('.case').forEach(card=>card.classList.toggle('hidden',(part.value&&card.dataset.part!==part.value)||(status.value&&card.dataset.status!==status.value)))}} part.onchange=apply;status.onchange=apply;
</script></body></html>"""
    _atomic_write_text(path, body)


def _write_csv(path: Path, specs: tuple[dict[str, object], ...], latest: Mapping[str, Mapping[str, object]]) -> None:
    fields = (
        "case_id",
        "incoming_part_id",
        "placement_id",
        "orientation_id",
        "pickup_x",
        "pickup_y",
        "pickup_roll_deg",
        "pickup_pitch_deg",
        "pickup_yaw_deg",
        "holder_arm",
        "inserter_arm",
        "status",
        "success",
        "failure_stage",
        "failure_phase_label",
        "result_status",
        "duration_s",
        "pair_id",
        "transition_id",
        "message",
        "video_path",
        "image_path",
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for spec in specs:
            record = dict(latest.get(str(spec["case_id"]), {**spec, "status": "pending"}))
            writer.writerow(
                {
                    field: (
                        record.get(field, "").replace("\x00", "")
                        if isinstance(record.get(field, ""), str)
                        else record.get(field, "")
                    )
                    for field in fields
                }
            )
    os.replace(temporary, path)


def _refresh_outputs(
    *,
    output_dir: Path,
    specs: tuple[dict[str, object], ...],
    latest: Mapping[str, Mapping[str, object]],
) -> None:
    records = [dict(latest.get(str(spec["case_id"]), {**spec, "status": "pending"})) for spec in specs]
    terminal = [record for record in records if str(record.get("status")) in TERMINAL_STATUSES]
    summary = {
        "schema_version": 1,
        "kind": "dual_assembly_benchmark_summary",
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        "case_count": len(specs),
        "completed_count": len(terminal),
        "success_count": sum(bool(record.get("success", False)) for record in terminal),
        "failed_count": sum(str(record.get("status")) == "failed" for record in terminal),
        "interrupted_count": sum(str(record.get("status")) == "interrupted" for record in terminal),
        "records": records,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "summary.csv", specs, latest)
    _write_html(output_dir / "index.html", specs=specs, latest=latest, output_dir=output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parts", nargs="*", default=None)
    parser.add_argument("--placements", nargs="*", default=None)
    parser.add_argument("--orientations", nargs="*", default=None)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--repair-videos",
        action="store_true",
        help="Convert legacy MP4V recordings to browser-playable WebM and refresh the dashboard.",
    )
    parser.add_argument(
        "--repair-failure-evidence",
        action="store_true",
        help="Backfill precise failure phases and scene stills for cases without video.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    payload = _read_mapping(config_path)
    output_dir = _output_dir(payload, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_parts = None if args.parts is None else {str(value) for value in args.parts}
    selected_placements = None if args.placements is None else {_safe_id(value) for value in args.placements}
    selected_orientations = None if args.orientations is None else {_safe_id(value) for value in args.orientations}
    specs = _case_specs(
        payload=payload,
        selected_parts=selected_parts,
        selected_placements=selected_placements,
        selected_orientations=selected_orientations,
        limit_cases=args.limit_cases,
    )
    manifest = {
        "schema_version": 1,
        "kind": "dual_assembly_benchmark_manifest",
        "created_or_refreshed_at": datetime.now(tz=timezone.utc).isoformat(),
        "config_path": str(config_path),
        "case_count": len(specs),
        "cases": list(specs),
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    _atomic_write_text(output_dir / "config_snapshot.yaml", yaml.safe_dump(payload, sort_keys=False))

    events_path = output_dir / "events.jsonl"
    if args.no_resume:
        _atomic_write_text(events_path, "")
    latest = _latest_records(_jsonl_records(events_path))
    _refresh_outputs(output_dir=output_dir, specs=specs, latest=latest)
    print(
        f"[DUAL-BENCH] cases={len(specs)} output={output_dir} dashboard={output_dir / 'index.html'}",
        flush=True,
    )
    if args.dry_run:
        print("[DUAL-BENCH] Dry run complete; manifest/dashboard written.", flush=True)
        return 0
    if args.repair_videos:
        converted, skipped = _repair_browser_videos(
            payload=payload,
            output_dir=output_dir,
            specs=specs,
            latest=latest,
            events_path=events_path,
        )
        print(
            f"[DUAL-BENCH] Browser video repair complete: converted={converted} skipped={skipped}. "
            f"Open {output_dir / 'index.html'}",
            flush=True,
        )
        return 0
    if args.repair_failure_evidence:
        repaired, skipped = _repair_failure_evidence(
            payload=payload,
            output_dir=output_dir,
            specs=specs,
            latest=latest,
            events_path=events_path,
        )
        print(
            f"[DUAL-BENCH] Failure evidence repair complete: repaired={repaired} skipped={skipped}. "
            f"Open {output_dir / 'index.html'}",
            flush=True,
        )
        return 0

    retry_failed = bool(args.retry_failed)
    for index, spec in enumerate(specs, start=1):
        case_id = str(spec["case_id"])
        prior = latest.get(case_id, {})
        prior_status = str(prior.get("status", ""))
        if prior_status == "success" or (prior_status == "failed" and not retry_failed):
            print(f"[DUAL-BENCH] Skip {index}/{len(specs)} {case_id}: {prior_status}", flush=True)
            continue
        paths = _case_paths(output_dir, spec)
        running = {
            **spec,
            "status": "running",
            "success": False,
            "message": "Planning/Isaac execution is active.",
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "plan_json": str(paths["plan"]),
            "attempt_json": str(paths["attempt"]),
            "video_path": str(paths["video"]),
            "image_path": "",
            "log_path": str(paths["log"]),
        }
        _append_jsonl(events_path, running)
        latest[case_id] = running
        _refresh_outputs(output_dir=output_dir, specs=specs, latest=latest)
        print(
            f"[DUAL-BENCH] Run {index}/{len(specs)} {case_id} "
            f"holder={spec['holder_arm']} inserter={spec['inserter_arm']}",
            flush=True,
        )
        record, interrupted = _run_case(payload=payload, spec=spec, paths=paths)
        _append_jsonl(events_path, record)
        latest[case_id] = record
        _refresh_outputs(output_dir=output_dir, specs=specs, latest=latest)
        print(
            f"[DUAL-BENCH] Saved {case_id}: status={record['status']} duration={float(record['duration_s']):.1f}s",
            flush=True,
        )
        if interrupted:
            print(
                f"[DUAL-BENCH] Partial benchmark is safe at {output_dir / 'index.html'}. "
                "Run the same command to resume.",
                flush=True,
            )
            return 130

    print(f"[DUAL-BENCH] Complete. Open {output_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
