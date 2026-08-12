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
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dual_assembly_benchmark.yaml"
TERMINAL_STATUSES = {"success", "failed", "interrupted"}
FAILURE_STAGE_ORDER = {
    "setup": 0,
    "planning": 1,
    "incoming_grasp_planning": 2,
    "moveit_candidate_planning": 3,
    "execution_start": 4,
    "holder_base_grasp": 5,
    "incoming_part_grasp": 6,
    "transition": 7,
    "execution_validation": 8,
    "success": 9,
    "pending": 10,
}
FAILURE_STAGE_LABELS = {
    "setup": "MoveIt/Isaac setup",
    "planning": "Grasp/transition planning",
    "incoming_grasp_planning": "Incoming-part grasp planning",
    "moveit_candidate_planning": "MoveIt candidate planning",
    "execution_start": "Isaac execution startup",
    "holder_base_grasp": "Holder/base grasp",
    "incoming_part_grasp": "Incoming-part grasp",
    "transition": "Transport to pre-insertion",
    "execution_validation": "Final execution validation",
    "success": "Successful",
    "pending": "Not completed",
}


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


def _failed_case_specs_from_summary(
    specs: Iterable[Mapping[str, object]],
    *,
    summary_path: Path,
    failure_stages: set[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Select failed cases, optionally restricted to prior failure stages."""

    resolved_summary = summary_path.expanduser().resolve()
    summary = _read_json_mapping(resolved_summary)
    if summary is None:
        raise FileNotFoundError(f"Prior benchmark summary does not exist or is invalid: {resolved_summary}")
    raw_records = summary.get("records")
    if not isinstance(raw_records, list):
        raise ValueError(f"Prior benchmark summary has no records list: {resolved_summary}")
    selected_failure_stages = {str(stage).strip() for stage in (failure_stages or set()) if str(stage).strip()}
    failed_ids = {
        str(record.get("case_id", ""))
        for record in raw_records
        if isinstance(record, dict)
        and str(record.get("status", "")) == "failed"
        and record.get("case_id")
        and (not selected_failure_stages or str(record.get("failure_stage", "")) in selected_failure_stages)
    }
    if not failed_ids:
        stage_suffix = (
            "" if not selected_failure_stages else f" in failure stage(s) {', '.join(sorted(selected_failure_stages))}"
        )
        raise ValueError(f"Prior benchmark summary contains no failed cases{stage_suffix}: {resolved_summary}")

    available = {str(spec["case_id"]): dict(spec) for spec in specs}
    missing = sorted(failed_ids.difference(available))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise ValueError(
            f"Prior summary contains {len(missing)} failed case(s) outside the current benchmark matrix: "
            f"{preview}{suffix}."
        )
    return tuple(spec for case_id, spec in available.items() if case_id in failed_ids)


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
    sequence = _read_json_mapping(artifact_dir / "assembly_sequence.json")
    if sequence is None:
        raise FileNotFoundError(f"Assembly sequence does not exist: {artifact_dir / 'assembly_sequence.json'}")
    part_assets = dict(sequence.get("parts", {}) or {})
    mesh_scale = float(sequence.get("mesh_scale", 0.01))

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
        part_asset = dict(part_assets.get(part_id, {}) or {})
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
                        "incoming_mesh_path": str(part_asset.get("mesh_path", "")),
                        "incoming_mesh_scale": mesh_scale,
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
        "thumbnail": case_dir / "scene_thumbnail.jpg",
        "image": case_dir / "failure_scene.svg",
        "log": case_dir / "run.log",
        "case": case_dir / "case.json",
    }


def _command(
    *,
    payload: Mapping[str, object],
    spec: Mapping[str, object],
    paths: Mapping[str, Path],
    planning_only: bool = False,
    ik_only: bool = False,
    ik_collision_diagnostics: bool = False,
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
        "--ik-solver",
        str(benchmark.get("ik_solver", "kdl")),
        "--ik-timeout-s",
        str(benchmark.get("ik_timeout_s", 0.35)),
        "--exact-ik-candidates",
        str(benchmark.get("exact_ik_candidates", 7)),
        "--exact-ik-beam-width",
        str(benchmark.get("exact_ik_beam_width", 4)),
        "--exact-ik-seed-perturbation-rad",
        str(benchmark.get("exact_ik_seed_perturbation_rad", 0.60)),
        "--pickup-approach-ik-steps",
        str(benchmark.get("pickup_approach_ik_steps", 5)),
        "--planning-time-s",
        str(benchmark.get("planning_time_s", 15.0)),
        "--planning-attempts",
        str(benchmark.get("planning_attempts", 16)),
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
        "--ros-domain-id",
        str(benchmark.get("ros_domain_id", 0)),
        "--headless",
        "--no-planning-debug-gui",
    ]
    if planning_only:
        command.append("--planning-only")
        if ik_only:
            command.extend(("--ik-only", "--skip-joint-space-ranking"))
        if ik_collision_diagnostics:
            command.append("--ik-collision-diagnostics")
    else:
        command.extend(
            [
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
            ]
        )
    if not planning_only and benchmark.get("isaac_python"):
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
    moveit = dict(payload.get("moveit", {}) or {})
    ik_preflight = dict(payload.get("ik_preflight", {}) or moveit.get("ik_preflight", {}) or {})
    collision_diagnostics = dict(ik_preflight.get("collision_diagnostics", {}) or {})
    target_diagnostics: dict[str, dict[str, object]] = {}
    failure_target_counts: dict[str, int] = {}
    role_records = dict(ik_preflight.get("records", {}) or {})
    for role, raw_records in role_records.items():
        if not isinstance(raw_records, list):
            continue
        for role_record in raw_records:
            if not isinstance(role_record, Mapping):
                continue
            for target_record in role_record.get("targets", []):
                if not isinstance(target_record, Mapping):
                    continue
                target = str(target_record.get("target", "unknown"))
                key = f"{role}:{target}"
                aggregate = target_diagnostics.setdefault(
                    key,
                    {
                        "role": str(role),
                        "target": target,
                        "evaluations": 0,
                        "failed_evaluations": 0,
                        "seed_attempts": 0,
                        "ik_requests": 0,
                        "kinematic_cache_hits": 0,
                        "kinematic_cache_misses": 0,
                        "collision_disabled_ik_solutions": 0,
                        "kinematic_or_numerical_failures": 0,
                        "valid_states": 0,
                        "invalid_states": 0,
                        "invalid_states_without_contacts": 0,
                        "contact_class_counts": {},
                        "contact_pair_counts": {},
                    },
                )
                aggregate["evaluations"] = int(aggregate["evaluations"]) + 1
                aggregate["seed_attempts"] = int(aggregate["seed_attempts"]) + int(
                    target_record.get("seed_attempts", 0)
                )
                aggregate["ik_requests"] = int(aggregate["ik_requests"]) + int(
                    target_record.get("ik_requests", target_record.get("seed_attempts", 0))
                )
                aggregate["kinematic_cache_hits"] = int(aggregate["kinematic_cache_hits"]) + int(
                    target_record.get("kinematic_cache_hits", 0)
                )
                aggregate["kinematic_cache_misses"] = int(aggregate["kinematic_cache_misses"]) + int(
                    target_record.get("kinematic_cache_misses", 0)
                )
                if not bool(target_record.get("ok", False)):
                    aggregate["failed_evaluations"] = int(aggregate["failed_evaluations"]) + 1
                    failure_target_counts[key] = int(failure_target_counts.get(key, 0)) + 1
                diagnostic = dict(target_record.get("collision_diagnostics", {}) or {})
                for field in (
                    "collision_disabled_ik_solutions",
                    "kinematic_or_numerical_failures",
                    "valid_states",
                    "invalid_states",
                    "invalid_states_without_contacts",
                ):
                    aggregate[field] = int(aggregate[field]) + int(diagnostic.get(field, 0))
                for field in ("contact_class_counts", "contact_pair_counts"):
                    destination = aggregate[field]
                    assert isinstance(destination, dict)
                    for name, count in dict(diagnostic.get(field, {}) or {}).items():
                        destination[str(name)] = int(destination.get(str(name), 0)) + int(count)
    return {
        "pair_id": str(payload.get("pair_id", "")),
        "transition_id": str(payload.get("transition_id", "")),
        "execution_candidate_id": str(payload.get("execution_candidate_id", "")),
        "selection_score": payload.get("selection_score"),
        "ik_seed_calls": int(ik_preflight.get("ik_seed_calls", 0)),
        "ik_solutions_found": int(ik_preflight.get("ik_solutions_found", 0)),
        "ik_collision_diagnostics": collision_diagnostics,
        "ik_target_diagnostics": target_diagnostics,
        "ik_failure_target_counts": failure_target_counts,
    }


def _merge_count_mapping(destination: dict[str, int], source: object) -> None:
    if not isinstance(source, Mapping):
        return
    for key, value in source.items():
        destination[str(key)] = int(destination.get(str(key), 0)) + int(value)


def _aggregate_ik_diagnostics(records: list[dict[str, object]]) -> dict[str, object]:
    aggregate: dict[str, object] = {
        "case_count": 0,
        "ik_requests": 0,
        "kinematic_cache_hits": 0,
        "kinematic_cache_misses": 0,
        "ik_request_duration_s": 0.0,
        "collision_disabled_ik_solutions": 0,
        "kinematic_or_numerical_failures": 0,
        "state_validity_requests": 0,
        "valid_states": 0,
        "invalid_states": 0,
        "invalid_states_without_contacts": 0,
        "contact_class_counts": {},
        "contact_pair_counts": {},
        "failure_target_counts": {},
        "target_diagnostics": {},
    }
    for record in records:
        diagnostics = record.get("ik_collision_diagnostics")
        if not isinstance(diagnostics, Mapping) or not diagnostics:
            continue
        aggregate["case_count"] = int(aggregate["case_count"]) + 1
        for field in (
            "ik_requests",
            "kinematic_cache_hits",
            "kinematic_cache_misses",
            "collision_disabled_ik_solutions",
            "kinematic_or_numerical_failures",
            "state_validity_requests",
            "valid_states",
            "invalid_states",
            "invalid_states_without_contacts",
        ):
            aggregate[field] = int(aggregate[field]) + int(diagnostics.get(field, 0))
        aggregate["ik_request_duration_s"] = float(aggregate["ik_request_duration_s"]) + float(
            diagnostics.get("ik_request_duration_s", 0.0)
        )
        _merge_count_mapping(aggregate["contact_class_counts"], diagnostics.get("contact_class_counts"))
        _merge_count_mapping(aggregate["contact_pair_counts"], diagnostics.get("contact_pair_counts"))
        _merge_count_mapping(aggregate["failure_target_counts"], record.get("ik_failure_target_counts"))
        targets = record.get("ik_target_diagnostics")
        if not isinstance(targets, Mapping):
            continue
        all_targets = aggregate["target_diagnostics"]
        assert isinstance(all_targets, dict)
        for key, raw_target in targets.items():
            if not isinstance(raw_target, Mapping):
                continue
            target = all_targets.setdefault(
                str(key),
                {
                    "role": str(raw_target.get("role", "")),
                    "target": str(raw_target.get("target", "")),
                    "evaluations": 0,
                    "failed_evaluations": 0,
                    "seed_attempts": 0,
                    "ik_requests": 0,
                    "kinematic_cache_hits": 0,
                    "kinematic_cache_misses": 0,
                    "collision_disabled_ik_solutions": 0,
                    "kinematic_or_numerical_failures": 0,
                    "valid_states": 0,
                    "invalid_states": 0,
                    "invalid_states_without_contacts": 0,
                    "contact_class_counts": {},
                    "contact_pair_counts": {},
                },
            )
            for field in (
                "evaluations",
                "failed_evaluations",
                "seed_attempts",
                "ik_requests",
                "kinematic_cache_hits",
                "kinematic_cache_misses",
                "collision_disabled_ik_solutions",
                "kinematic_or_numerical_failures",
                "valid_states",
                "invalid_states",
                "invalid_states_without_contacts",
            ):
                target[field] = int(target[field]) + int(raw_target.get(field, 0))
            _merge_count_mapping(target["contact_class_counts"], raw_target.get("contact_class_counts"))
            _merge_count_mapping(target["contact_pair_counts"], raw_target.get("contact_pair_counts"))
    return aggregate


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
    planning_only: bool = False,
    ik_only: bool = False,
    ik_collision_diagnostics: bool = False,
) -> tuple[dict[str, object], bool]:
    paths["case_dir"].mkdir(parents=True, exist_ok=True)
    for key in ("plan", "attempt", "video", "thumbnail", "image", "case"):
        paths[key].unlink(missing_ok=True)
    command = _command(
        payload=payload,
        spec=spec,
        paths=paths,
        planning_only=planning_only,
        ik_only=ik_only,
        ik_collision_diagnostics=ik_collision_diagnostics,
    )
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
    elif planning_only and returncode == 0 and paths["plan"].is_file():
        success = True
        status = "success"
        message = "MoveIt candidate planning succeeded; Isaac execution was intentionally skipped."
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
    thumbnail_path = ""
    thumbnail_error = ""
    if video_exists:
        try:
            _extract_video_thumbnail(
                ffmpeg=_ffmpeg_executable(payload),
                source=paths["video"],
                target=paths["thumbnail"],
            )
            thumbnail_path = str(paths["thumbnail"])
        except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError, OSError) as exc:
            thumbnail_error = str(exc)
    if not video_exists and (not success or planning_only):
        try:
            _render_failure_scene(
                path=paths["image"],
                payload=payload,
                spec=spec,
                plan_path=paths["plan"],
                failure_label=(
                    "MoveIt candidate planning succeeded" if planning_only and success else failure_phase_label
                ),
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
        "planning_only": planning_only,
        "ik_only": bool(ik_only),
        "ik_collision_diagnostics_enabled": bool(ik_collision_diagnostics),
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
        "thumbnail_path": thumbnail_path,
        "thumbnail_error": thumbnail_error,
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


def _extract_video_thumbnail(*, ffmpeg: Path, source: Path, target: Path) -> None:
    """Extract a browser-compatible scene poster from a recorded video."""

    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    temporary.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
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
            "-ss",
            "0.5",
            "-frames:v",
            "1",
            "-vf",
            "scale=640:-2:flags=lanczos",
            "-q:v",
            "3",
            str(temporary),
        ],
        check=True,
    )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no scene thumbnail for '{source}'.")
    os.replace(temporary, target)


def _repair_browser_videos(
    *,
    payload: Mapping[str, object],
    output_dir: Path,
    specs: tuple[dict[str, object], ...],
    latest: dict[str, dict[str, object]],
    events_path: Path,
) -> tuple[int, int, int]:
    ffmpeg = _ffmpeg_executable(payload)
    converted = 0
    thumbnails = 0
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
        target = source
        changed = False
        if source.suffix.lower() != ".webm":
            target = source.with_suffix(".webm")
            print(f"[DUAL-BENCH] Convert video {index}/{len(specs)} {case_id}", flush=True)
            if not target.is_file() or target.stat().st_size == 0:
                _convert_mp4_to_webm(ffmpeg=ffmpeg, source=source, target=target)
            converted += 1
            changed = True
        thumbnail = _case_paths(output_dir, spec)["thumbnail"]
        if not thumbnail.is_file() or thumbnail.stat().st_size == 0:
            print(f"[DUAL-BENCH] Extract thumbnail {index}/{len(specs)} {case_id}", flush=True)
            _extract_video_thumbnail(ffmpeg=ffmpeg, source=target, target=thumbnail)
            thumbnails += 1
            changed = True
        needs_update = (
            changed
            or str(record.get("video_path", "")) != str(target)
            or str(record.get("thumbnail_path", "")) != str(thumbnail)
        )
        if not needs_update:
            skipped += 1
            continue
        updated = {
            **record,
            "video_path": str(target),
            "thumbnail_path": str(thumbnail),
        }
        if target != source:
            updated["original_video_path"] = str(source)
        case_path = _case_paths(output_dir, spec)["case"]
        if case_path.is_file():
            _atomic_write_json(case_path, updated)
        _append_jsonl(events_path, updated)
        latest[case_id] = updated
    _refresh_outputs(output_dir=output_dir, specs=specs, latest=latest)
    return converted, thumbnails, skipped


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


def _friendly_id(value: object) -> str:
    return str(value).replace("_", " ").strip().title()


def _failure_key(record: Mapping[str, object]) -> str:
    if bool(record.get("success", False)):
        return "success"
    stage = str(record.get("failure_stage", "")).strip()
    if stage:
        return stage
    return "pending" if str(record.get("status", "")) not in TERMINAL_STATUSES else "planning"


def _failure_label(record: Mapping[str, object]) -> str:
    key = _failure_key(record)
    configured = str(record.get("failure_phase_label", "")).strip()
    return configured or FAILURE_STAGE_LABELS.get(key, _friendly_id(key))


def _mesh_icon_svg(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    rotation: np.ndarray,
    label: str,
    color: str,
    show_axes: bool = False,
) -> str:
    """Return a compact isometric mesh thumbnail for a dashboard guide."""

    points = np.asarray(vertices, dtype=float)
    faces = np.asarray(triangles, dtype=int)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not len(faces):
        raise ValueError("A guide mesh requires finite vertices and triangles.")
    points = (points - np.mean(points, axis=0)) @ np.asarray(rotation, dtype=float).T
    extent = max(float(np.ptp(points, axis=0).max()), 1.0e-6)
    right = np.asarray((0.70710678, -0.70710678, 0.0), dtype=float)
    up = np.asarray((-0.40824829, -0.40824829, 0.81649658), dtype=float)
    depth_axis = np.asarray((0.57735027, 0.57735027, 0.57735027), dtype=float)
    axis_points = np.zeros((0, 3), dtype=float)
    if show_axes:
        axis_points = np.vstack((np.zeros(3), np.asarray(rotation, dtype=float).T * (extent * 0.72)))
    fit_points = np.vstack((points, axis_points)) if len(axis_points) else points
    fit_x = fit_points @ right
    fit_y = fit_points @ up
    scale = min(178.0 / max(float(np.ptp(fit_x)), 1.0e-6), 92.0 / max(float(np.ptp(fit_y)), 1.0e-6))
    center_x = 110.0 - scale * float((np.min(fit_x) + np.max(fit_x)) * 0.5)
    center_y = 62.0 + scale * float((np.min(fit_y) + np.max(fit_y)) * 0.5)

    def project(raw: np.ndarray) -> np.ndarray:
        return np.column_stack((center_x + scale * (raw @ right), center_y - scale * (raw @ up)))

    projected = project(points)
    depth = points @ depth_axis
    stride = max(1, int(math.ceil(len(faces) / 180)))
    polygons = []
    for triangle in faces[::stride]:
        polygon = projected[triangle]
        polygons.append(
            (
                float(np.mean(depth[triangle])),
                f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in polygon)}" '
                f'fill="{color}" fill-opacity=".78" stroke="#07101b" stroke-width=".7"/>',
            )
        )
    shapes = [shape for _, shape in sorted(polygons, key=lambda item: item[0], reverse=True)]
    if show_axes:
        projected_axes = project(axis_points)
        origin = projected_axes[0]
        for index, (axis_label, axis_color) in enumerate(
            (("X", "#ff647c"), ("Y", "#3ddc97"), ("Z", "#69a8ff")), start=1
        ):
            endpoint = projected_axes[index]
            shapes.extend(
                (
                    f'<line x1="{origin[0]:.1f}" y1="{origin[1]:.1f}" x2="{endpoint[0]:.1f}" '
                    f'y2="{endpoint[1]:.1f}" stroke="{axis_color}" stroke-width="2.4"/>',
                    f'<circle cx="{endpoint[0]:.1f}" cy="{endpoint[1]:.1f}" r="3.2" fill="{axis_color}"/>',
                    f'<text x="{endpoint[0] + 4:.1f}" y="{endpoint[1] - 3:.1f}" fill="{axis_color}" '
                    f'font-family="system-ui,sans-serif" font-size="10" font-weight="800">{axis_label}</text>',
                )
            )
    return (
        '<svg class="guide-svg" viewBox="0 0 220 126" role="img" '
        f'aria-label="{html.escape(label, quote=True)}"><title>{html.escape(label)}</title>'
        '<rect width="220" height="126" rx="10" fill="#0a111d"/>'
        '<path d="M16 106H204" stroke="#344963" stroke-width="1" stroke-dasharray="4 4"/>'
        f"{''.join(shapes)}</svg>"
    )


def _orientation_icon_svg(record: Mapping[str, object]) -> str:
    half = np.asarray((0.72, 0.38, 0.27), dtype=float)
    vertices = np.asarray(
        [
            (x * half[0], y * half[1], z * half[2])
            for x, y, z in (
                (-1, -1, -1),
                (1, -1, -1),
                (1, 1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, 1, 1),
            )
        ],
        dtype=float,
    )
    triangles = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ),
        dtype=int,
    )
    rpy = (
        float(record.get("pickup_roll_deg", 0.0)),
        float(record.get("pickup_pitch_deg", 0.0)),
        float(record.get("pickup_yaw_deg", 0.0)),
    )
    label = f"{_friendly_id(record.get('orientation_id', 'orientation'))}: RPY {rpy[0]:.0f}, {rpy[1]:.0f}, {rpy[2]:.0f} degrees"
    return _mesh_icon_svg(
        vertices,
        triangles,
        rotation=_rpy_rotation_matrix(*rpy),
        label=label,
        color="#ff9e43",
        show_axes=True,
    )


def _part_icon_svg(record: Mapping[str, object]) -> str:
    raw_path = Path(str(record.get("incoming_mesh_path", ""))).expanduser()
    mesh_path = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
    try:
        vertices, triangles = _load_obj_mesh(mesh_path.resolve(), scale=float(record.get("incoming_mesh_scale", 0.01)))
    except (FileNotFoundError, OSError, ValueError):
        vertices = np.asarray(
            ((-1, -0.5, -0.4), (1, -0.5, -0.4), (1, 0.5, -0.4), (-1, 0.5, -0.4), (0, 0, 0.8)),
            dtype=float,
        )
        triangles = np.asarray(((0, 1, 2), (0, 2, 3), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)), dtype=int)
    return _mesh_icon_svg(
        vertices,
        triangles,
        rotation=_rpy_rotation_matrix(18.0, 0.0, 30.0),
        label=f"Incoming part {record.get('incoming_part_id', '')}",
        color="#49d7aa",
    )


def _placement_icon_svg(record: Mapping[str, object]) -> str:
    pickup_x = float(record.get("pickup_x", 0.0))
    pickup_y = float(record.get("pickup_y", 0.0))
    assembly_x = float(record.get("assembly_x", 0.55))
    assembly_y = float(record.get("assembly_y", 0.0))
    inserter = str(record.get("inserter_arm", "lbr_two"))
    holder = str(record.get("holder_arm", "lbr_one"))

    def project(x: float, y: float) -> tuple[float, float]:
        return 18.0 + 184.0 * x / 0.75, 70.0 - 112.0 * y / 0.84

    pickup = project(pickup_x, pickup_y)
    assembly = project(assembly_x, assembly_y)
    arm_one = project(0.0, -0.42)
    arm_two = project(0.0, 0.42)
    inserter_base = arm_one if inserter == "lbr_one" else arm_two
    holder_base = arm_one if holder == "lbr_one" else arm_two
    label = f"{_friendly_id(record.get('placement_id', 'placement'))}: pickup {pickup_x:.2f}, {pickup_y:.2f} metres"
    return f'''<svg class="guide-svg" viewBox="0 0 220 140" role="img" aria-label="{html.escape(label, quote=True)}">
<title>{html.escape(label)}</title><rect width="220" height="140" rx="10" fill="#0a111d"/>
<path d="M18 10V130M18 70H207" stroke="#31445d" stroke-width="1"/>
<path d="M18 70H205" stroke="#69a8ff" stroke-width="1.5" marker-end="url(#none)"/>
<line x1="{holder_base[0]:.1f}" y1="{holder_base[1]:.1f}" x2="{assembly[0]:.1f}" y2="{assembly[1]:.1f}" stroke="#ffd166" stroke-width="2" stroke-dasharray="4 3"/>
<line x1="{inserter_base[0]:.1f}" y1="{inserter_base[1]:.1f}" x2="{pickup[0]:.1f}" y2="{pickup[1]:.1f}" stroke="#b18cff" stroke-width="2" stroke-dasharray="4 3"/>
<circle cx="{arm_one[0]:.1f}" cy="{arm_one[1]:.1f}" r="7" fill="{"#b18cff" if inserter == "lbr_one" else "#ffd166"}"/><text x="29" y="126" fill="#9db0c7" font-size="9" font-family="system-ui">LBR one</text>
<circle cx="{arm_two[0]:.1f}" cy="{arm_two[1]:.1f}" r="7" fill="{"#b18cff" if inserter == "lbr_two" else "#ffd166"}"/><text x="29" y="19" fill="#9db0c7" font-size="9" font-family="system-ui">LBR two</text>
<rect x="{assembly[0] - 6:.1f}" y="{assembly[1] - 6:.1f}" width="12" height="12" rx="2" fill="#30d6a4"/><text x="{assembly[0] - 18:.1f}" y="{assembly[1] + 19:.1f}" fill="#7eeac8" font-size="9" font-family="system-ui">assembly</text>
<circle cx="{pickup[0]:.1f}" cy="{pickup[1]:.1f}" r="7" fill="#ff9e43" stroke="#ffe0bd"/><text x="{pickup[0] - 8:.1f}" y="{pickup[1] - 11:.1f}" fill="#ffc98d" font-size="9" font-family="system-ui">pick</text>
<text x="191" y="65" fill="#69a8ff" font-size="10" font-family="system-ui">+X</text></svg>'''


def _failure_stage_icon_svg(stage: str, label: str) -> str:
    stages = ("Plan", "Start", "Hold", "Pick", "Move", "Check")
    stage_index = {
        "setup": 0,
        "planning": 0,
        "incoming_grasp_planning": 0,
        "moveit_candidate_planning": 0,
        "execution_start": 1,
        "holder_base_grasp": 2,
        "incoming_part_grasp": 3,
        "transition": 4,
        "execution_validation": 5,
        "success": 5,
        "pending": 0,
    }.get(stage, 0)
    circles = []
    for index, name in enumerate(stages):
        x = 18 + index * 37
        completed = index < stage_index or stage == "success"
        current = index == stage_index and stage != "success"
        color = "#3ddc97" if completed else "#ff647c" if current else "#34445a"
        circles.append(
            f'<circle cx="{x}" cy="48" r="7" fill="{color}"/><text x="{x}" y="70" text-anchor="middle" fill="#9db0c7" font-size="8" font-family="system-ui">{name}</text>'
        )
    line_color = "#3ddc97" if stage == "success" else "#445a76"
    return (
        f'<svg class="guide-svg stage-svg" viewBox="0 0 220 82" role="img" aria-label="{html.escape(label, quote=True)}">'
        f'<title>{html.escape(label)}</title><rect width="220" height="82" rx="10" fill="#0a111d"/>'
        f'<line x1="18" y1="48" x2="203" y2="48" stroke="{line_color}" stroke-width="3"/>{"".join(circles)}</svg>'
    )


def _arm_icon_svg(inserter_arm: str) -> str:
    representative = {
        "pickup_x": 0.55,
        "pickup_y": -0.24 if inserter_arm == "lbr_one" else 0.24,
        "assembly_x": 0.55,
        "assembly_y": 0.0,
        "inserter_arm": inserter_arm,
        "holder_arm": "lbr_two" if inserter_arm == "lbr_one" else "lbr_one",
        "placement_id": f"{inserter_arm}_inserts",
    }
    return _placement_icon_svg(representative)


def _guide_button(*, filter_id: str, value: str, title: str, detail: str, image: str) -> str:
    return (
        f'<button class="guide-item" type="button" data-filter-id="{html.escape(filter_id, quote=True)}" '
        f'data-value="{html.escape(value, quote=True)}">{image}<span class="guide-item-text">'
        f"<strong>{html.escape(title)}</strong><small>{html.escape(detail)}</small>"
        '<span class="guide-count">0 shown</span></span></button>'
    )


def _visual_guides(specs: tuple[dict[str, object], ...], records: list[dict[str, object]]) -> str:
    unique_parts: dict[str, dict[str, object]] = {}
    unique_placements: dict[str, dict[str, object]] = {}
    unique_orientations: dict[str, dict[str, object]] = {}
    for spec in specs:
        unique_parts.setdefault(str(spec["incoming_part_id"]), spec)
        unique_placements.setdefault(str(spec["placement_id"]), spec)
        unique_orientations.setdefault(str(spec["orientation_id"]), spec)

    part_cards = []
    for part_id, spec in unique_parts.items():
        part_cards.append(
            _guide_button(
                filter_id="part",
                value=part_id,
                title=f"Part {part_id}",
                detail=f"Assembly step {int(spec.get('step_index', 0))}",
                image=_part_icon_svg(spec),
            )
        )
    placement_cards = []
    for placement_id, spec in unique_placements.items():
        placement_cards.append(
            _guide_button(
                filter_id="placement",
                value=placement_id,
                title=_friendly_id(placement_id),
                detail=f"pickup ({float(spec['pickup_x']):.2f}, {float(spec['pickup_y']):.2f}) m",
                image=_placement_icon_svg(spec),
            )
        )
    orientation_cards = []
    for orientation_id, spec in unique_orientations.items():
        orientation_cards.append(
            _guide_button(
                filter_id="orientation",
                value=orientation_id,
                title=_friendly_id(orientation_id),
                detail=(
                    f"RPY ({float(spec['pickup_roll_deg']):.0f}, {float(spec['pickup_pitch_deg']):.0f}, "
                    f"{float(spec['pickup_yaw_deg']):.0f}) degrees"
                ),
                image=_orientation_icon_svg(spec),
            )
        )
    present_failure_keys = sorted(
        {_failure_key(record) for record in records}, key=lambda key: FAILURE_STAGE_ORDER.get(key, 99)
    )
    failure_cards = [
        _guide_button(
            filter_id="failure",
            value=stage,
            title=FAILURE_STAGE_LABELS.get(stage, _friendly_id(stage)),
            detail="Click to filter cases at this outcome",
            image=_failure_stage_icon_svg(stage, FAILURE_STAGE_LABELS.get(stage, _friendly_id(stage))),
        )
        for stage in present_failure_keys
    ]
    inserter_cards = [
        _guide_button(
            filter_id="inserter",
            value=arm,
            title=f"{_friendly_id(arm)} inserts",
            detail="The other arm holds the assembled base",
            image=_arm_icon_svg(arm),
        )
        for arm in sorted({str(spec["inserter_arm"]) for spec in specs})
    ]
    status_cards = [
        _guide_button(
            filter_id="status",
            value=status,
            title=_friendly_id(status),
            detail="Benchmark case result",
            image=_failure_stage_icon_svg("success" if status == "success" else "pending", _friendly_id(status)),
        )
        for status in sorted({str(record.get("status", "pending")) for record in records})
    ]

    panels = (
        ("failure", "Failure phase", "The highlighted node is where the case stopped.", failure_cards),
        ("part", "Incoming part", "Actual OBJ geometry for each assembly step.", part_cards),
        (
            "placement",
            "Pickup location",
            "Top-down world view: orange is pickup, green is the fixed assembly.",
            placement_cards,
        ),
        (
            "orientation",
            "Pickup orientation",
            "The asymmetric block and local XYZ axes show the configured source-frame RPY.",
            orientation_cards,
        ),
        ("inserter", "Arm assignment", "Purple reaches the pickup; yellow holds the assembly.", inserter_cards),
        ("status", "Case status", "Terminal and unfinished benchmark outcomes.", status_cards),
    )
    return "".join(
        f'<section class="guide-panel" data-guide="{key}"><div class="guide-description"><strong>{title}</strong>'
        f'<span>{description}</span></div><div class="guide-grid">{"".join(cards)}</div></section>'
        for key, title, description, cards in panels
    )


def _select_options(values: Iterable[tuple[str, str]]) -> str:
    return "".join(
        f'<option value="{html.escape(value, quote=True)}">{html.escape(label)}</option>' for value, label in values
    )


def _dashboard_css() -> str:
    return """
:root{--bg:#0a0f18;--panel:#121a27;--panel2:#172235;--line:#27364a;--text:#e9f0fa;--muted:#93a4ba;--green:#3ddc97;--red:#ff647c;--amber:#ffca6a;--blue:#69a8ff}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at 20% 0,#18253a 0,var(--bg) 42%);color:var(--text);font:14px Inter,system-ui,sans-serif} button,select,input{font:inherit}
header{padding:30px clamp(18px,4vw,64px) 18px} h1{font-size:clamp(28px,4vw,46px);margin:0 0 8px}.subtitle,.muted{color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;padding:0 clamp(18px,4vw,64px) 24px}.stat{background:linear-gradient(145deg,#172235,#101722);border:1px solid var(--line);border-radius:16px;padding:17px}.stat strong{display:block;font-size:27px;margin-top:5px}
.content{padding:0 clamp(18px,4vw,64px) 50px}.control-bar{position:sticky;top:0;z-index:5;background:#0a0f18f2;backdrop-filter:blur(10px);border-block:1px solid #26364a;padding:12px 0}.control-row{display:flex;gap:10px;flex-wrap:wrap;align-items:end}.control-row+.control-row{margin-top:10px}.control{display:grid;gap:4px;min-width:145px}.control.wide{flex:1;min-width:240px}.control span{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}select,input{width:100%;background:var(--panel);color:var(--text);border:1px solid var(--line);padding:9px 11px;border-radius:9px}.reset{background:#21324a;color:var(--text);border:1px solid #405574;padding:9px 14px;border-radius:9px;cursor:pointer}.reset:hover{background:#2c4260}
.analysis{margin:20px 0 28px}.analysis-head{display:flex;align-items:end;gap:12px;flex-wrap:wrap;margin-bottom:14px}.analysis-head h2{margin:0 auto 0 0}.guide-wrap,.breakdown{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:15px}.guide-wrap{margin-bottom:14px}.guide-panel[hidden]{display:none}.guide-description{display:flex;gap:10px;align-items:baseline;margin-bottom:12px}.guide-description span{color:var(--muted)}.guide-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:10px}.guide-item{display:block;padding:0;text-align:left;overflow:hidden;background:#0e1622;color:var(--text);border:1px solid #2a3a50;border-radius:12px;cursor:pointer}.guide-item:hover,.guide-item.active{border-color:var(--blue);box-shadow:0 0 0 1px #69a8ff55}.guide-svg{display:block;width:100%;height:112px}.guide-item-text{display:grid;gap:3px;padding:9px 10px}.guide-item small{color:var(--muted);min-height:30px}.guide-count{color:var(--blue);font-size:11px}
.breakdown{overflow:auto}.breakdown table{width:100%;border-collapse:collapse;min-width:760px}.breakdown th,.breakdown td{padding:9px;border-bottom:1px solid #263346;text-align:left}.breakdown th{color:#b8c7da;font-size:11px;text-transform:uppercase;letter-spacing:.05em}.breakdown tbody tr{cursor:pointer}.breakdown tbody tr:hover{background:#1a2738}.rate{font-variant-numeric:tabular-nums}.failure-mix{color:var(--muted);max-width:320px}.empty-row{text-align:center;color:var(--muted);padding:24px!important}
.cases-head{display:flex;align-items:baseline;gap:10px;margin:22px 0 12px}.cases-head h2{margin:0}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}.case{background:var(--panel);border:1px solid var(--line);border-radius:15px;overflow:hidden;box-shadow:0 16px 40px #0005}.case.success{border-color:#2c7d5d}.case.failed{border-color:#8f3345}.case.interrupted{border-color:#9b7733}.video-shell,.failure-image-button{position:relative;display:block;width:100%;aspect-ratio:16/9;background:#070a0f}video,.no-video,.failure-image{width:100%;aspect-ratio:16/9;background:#070a0f;display:flex;align-items:center;justify-content:center;color:var(--muted);object-fit:contain}.video-load{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);border:1px solid #8db9ff;background:#102440e8;color:#eef6ff;border-radius:999px;padding:11px 18px;font-weight:700;cursor:pointer;box-shadow:0 6px 24px #0009}.video-load:hover{background:#1b3b67}.video-open{position:absolute;right:10px;top:9px;background:#07101dcc;border-radius:7px;padding:5px 8px;font-size:11px}.failure-image-button{border:0;padding:0;cursor:zoom-in;color:var(--text)}.image-expand{position:absolute;right:10px;bottom:9px;background:#07101ddd;border:1px solid #ffffff38;border-radius:7px;padding:6px 9px;font-size:11px}.failure-image-button:hover .image-expand{background:#1b3b67}
.case-body{padding:14px}.case-title{font-weight:700;font-size:16px;padding-right:80px}.badge{float:right;margin-top:-22px;padding:4px 9px;border-radius:999px;background:#26364d}.success .badge{color:var(--green)}.failed .badge{color:var(--red)}.interrupted .badge{color:var(--amber)}.failure-callout{margin:11px 0 0;padding:8px 10px;border-left:3px solid var(--red);background:#351721;color:#ff9aaa;font-weight:700}dl{display:grid;grid-template-columns:65px 1fr;gap:5px;margin:12px 0}dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}.message{color:var(--muted);min-height:38px}a{color:var(--blue);text-decoration:none}.links{display:flex;gap:12px}.hidden{display:none!important}
#image-viewer{width:96vw;height:94vh;max-width:none;max-height:none;padding:44px 16px 16px;border:1px solid #536b8b;border-radius:14px;background:#080d15f5;color:var(--text)}#image-viewer::backdrop{background:#02050bd9;backdrop-filter:blur(4px)}#image-viewer img{display:block;width:100%;height:calc(100% - 28px);object-fit:contain}.viewer-close{position:absolute;right:14px;top:10px;border:1px solid #7188a7;background:#18263b;color:var(--text);border-radius:999px;width:34px;height:34px;font-size:22px;cursor:pointer}#viewer-caption{text-align:center;color:var(--muted);padding-top:7px}
@media(max-width:700px){.control{min-width:calc(50% - 6px)}.control.wide{min-width:100%}.guide-description{display:grid}.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
"""


def _dashboard_script() -> str:
    return """
const cards=Array.from(document.querySelectorAll('.case'));
const grid=document.querySelector('#case-grid');
const filters={part:document.querySelector('#part'),status:document.querySelector('#status'),failure:document.querySelector('#failure'),placement:document.querySelector('#placement'),orientation:document.querySelector('#orientation'),inserter:document.querySelector('#inserter')};
const search=document.querySelector('#search'),caseSort=document.querySelector('#case-sort'),groupBy=document.querySelector('#group-by'),groupSort=document.querySelector('#group-sort'),guideDimension=document.querySelector('#guide-dimension');
const terminalStatuses=new Set(['success','failed','interrupted']);
function matches(card,ignoreFilter=''){for(const [key,control] of Object.entries(filters)){if(key!==ignoreFilter&&control.value&&card.dataset[key]!==control.value)return false}const query=search.value.trim().toLowerCase();return !query||card.dataset.search.includes(query)}
function visibleCards(){return cards.filter(card=>matches(card))}
function median(values){if(!values.length)return 0;const ordered=[...values].sort((a,b)=>a-b),middle=Math.floor(ordered.length/2);return ordered.length%2?ordered[middle]:(ordered[middle-1]+ordered[middle])/2}
function sortCards(){const definitions={failure:['failureOrder',1,true],status:['statusOrder',1,true],part:['partOrder',1,true],placement:['placementOrder',1,true],orientation:['orientationOrder',1,true],duration_desc:['duration',-1,true],duration_asc:['duration',1,true],case:['case',1,false]};const [key,direction,numeric]=definitions[caseSort.value];cards.sort((a,b)=>{const av=a.dataset[key],bv=b.dataset[key];const compared=numeric?Number(av)-Number(bv):av.localeCompare(bv);return direction*compared||a.dataset.case.localeCompare(b.dataset.case)}).forEach(card=>grid.append(card))}
function updateStats(shown){const completed=shown.filter(card=>terminalStatuses.has(card.dataset.status)),success=completed.filter(card=>card.dataset.success==='true').length,failed=completed.filter(card=>card.dataset.status==='failed').length,durations=completed.map(card=>Number(card.dataset.duration)).filter(Number.isFinite);document.querySelector('#shown-count').textContent=shown.length;document.querySelector('#shown-success').textContent=success;document.querySelector('#shown-failed').textContent=failed;document.querySelector('#shown-rate').textContent=`${completed.length?(100*success/completed.length).toFixed(1):'0.0'}%`;document.querySelector('#shown-median').textContent=`${median(durations).toFixed(0)}s`;document.querySelector('#case-count').textContent=`${shown.length} shown`}
function groupLabel(card,key){return card.dataset[`${key}Label`]||card.dataset[key]||'Unknown'}
function updateBreakdown(shown){const key=groupBy.value,groups=new Map();for(const card of shown){const value=card.dataset[key]||'unknown';if(!groups.has(value))groups.set(value,{value,label:groupLabel(card,key),cards:[]});groups.get(value).cards.push(card)}let rows=Array.from(groups.values()).map(group=>{const completed=group.cards.filter(card=>terminalStatuses.has(card.dataset.status)),success=completed.filter(card=>card.dataset.success==='true').length,failed=completed.filter(card=>card.dataset.status==='failed').length,durations=completed.map(card=>Number(card.dataset.duration)).filter(Number.isFinite),failureCounts=new Map();for(const card of completed.filter(card=>card.dataset.status!=='success')){failureCounts.set(card.dataset.failureLabel,(failureCounts.get(card.dataset.failureLabel)||0)+1)}const failureMix=Array.from(failureCounts.entries()).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([label,count])=>`${label}: ${count}`).join(' · ')||'—';return{...group,cases:group.cards.length,completed:completed.length,success,failed,rate:completed.length?100*success/completed.length:0,median:median(durations),failureMix}});const sorters={label:(a,b)=>a.label.localeCompare(b.label),cases_desc:(a,b)=>b.cases-a.cases||a.label.localeCompare(b.label),failures_desc:(a,b)=>b.failed-a.failed||a.label.localeCompare(b.label),rate_asc:(a,b)=>a.rate-b.rate||a.label.localeCompare(b.label),duration_desc:(a,b)=>b.median-a.median||a.label.localeCompare(b.label)};rows.sort(sorters[groupSort.value]);const body=document.querySelector('#breakdown-body');body.replaceChildren();if(!rows.length){const row=body.insertRow();const cell=row.insertCell();cell.colSpan=7;cell.className='empty-row';cell.textContent='No cases match the current filters.';return}for(const group of rows){const row=body.insertRow();for(const text of [group.label,group.cases,group.success,group.failed,`${group.rate.toFixed(1)}%`,`${group.median.toFixed(0)}s`,group.failureMix]){const cell=row.insertCell();cell.textContent=text}row.cells[4].className='rate';row.cells[6].className='failure-mix';row.title=`Filter ${groupLabel(group.cards[0],key)}`;row.addEventListener('click',()=>{filters[key].value=group.value;apply()})}}
function updateGuide(shown){const dimension=guideDimension.value==='follow'?groupBy.value:guideDimension.value;document.querySelectorAll('.guide-panel').forEach(panel=>panel.hidden=panel.dataset.guide!==dimension);for(const button of document.querySelectorAll('.guide-item')){const control=filters[button.dataset.filterId],facetCards=cards.filter(card=>matches(card,button.dataset.filterId)),count=facetCards.filter(card=>card.dataset[button.dataset.filterId]===button.dataset.value).length;button.querySelector('.guide-count').textContent=`${count} shown`;button.classList.toggle('active',control.value===button.dataset.value)}}
function apply(){sortCards();for(const card of cards)card.classList.toggle('hidden',!matches(card));const shown=visibleCards();updateStats(shown);updateBreakdown(shown);updateGuide(shown)}
for(const control of Object.values(filters))control.addEventListener('change',apply);search.addEventListener('input',apply);caseSort.addEventListener('change',apply);groupBy.addEventListener('change',apply);groupSort.addEventListener('change',()=>updateBreakdown(visibleCards()));guideDimension.addEventListener('change',()=>updateGuide(visibleCards()));
document.querySelector('#reset-filters').addEventListener('click',()=>{for(const control of Object.values(filters))control.value='';search.value='';caseSort.value='failure';groupBy.value='failure';groupSort.value='failures_desc';guideDimension.value='follow';apply()});
document.querySelectorAll('.guide-item').forEach(button=>button.addEventListener('click',()=>{const control=filters[button.dataset.filterId];control.value=control.value===button.dataset.value?'':button.dataset.value;apply()}));
function loadAndPlay(shell){const video=shell.querySelector('video'),source=video.querySelector('source'),button=shell.querySelector('.video-load');if(!source.src){source.src=source.dataset.src;video.load()}button.hidden=true;video.play().catch(()=>{button.hidden=false;button.textContent='Playback failed · open video'})}document.querySelectorAll('.video-shell').forEach(shell=>shell.querySelector('.video-load').addEventListener('click',()=>loadAndPlay(shell)));
const viewer=document.querySelector('#image-viewer'),viewerImage=document.querySelector('#viewer-image'),viewerCaption=document.querySelector('#viewer-caption');document.querySelectorAll('.failure-image-button').forEach(button=>button.addEventListener('click',()=>{viewerImage.src=button.dataset.fullSrc;viewerImage.alt=button.dataset.caption;viewerCaption.textContent=button.dataset.caption;viewer.showModal()}));viewer.querySelector('.viewer-close').addEventListener('click',()=>viewer.close());viewer.addEventListener('click',event=>{if(event.target===viewer)viewer.close()});
apply();
"""


def _ik_diagnostic_report_html(records: list[dict[str, object]]) -> str:
    diagnostics = _aggregate_ik_diagnostics(records)
    if not int(diagnostics["case_count"]):
        return ""

    def sorted_counts(raw: object) -> list[tuple[str, int]]:
        if not isinstance(raw, Mapping):
            return []
        return sorted(((str(key), int(value)) for key, value in raw.items()), key=lambda item: (-item[1], item[0]))

    target_rows = []
    targets = diagnostics["target_diagnostics"]
    assert isinstance(targets, Mapping)
    for key, raw_target in sorted(
        targets.items(),
        key=lambda item: (-int(item[1].get("failed_evaluations", 0)), str(item[0])),
    ):
        target = dict(raw_target)
        top_contact = sorted_counts(target.get("contact_pair_counts"))[:1]
        target_rows.append(
            "<tr>"
            f"<td>{html.escape(str(key))}</td>"
            f"<td>{int(target.get('evaluations', 0))}</td>"
            f"<td>{int(target.get('failed_evaluations', 0))}</td>"
            f"<td>{int(target.get('seed_attempts', 0))}</td>"
            f"<td>{int(target.get('ik_requests', 0))}</td>"
            f"<td>{int(target.get('kinematic_cache_hits', 0))}</td>"
            f"<td>{int(target.get('collision_disabled_ik_solutions', 0))}</td>"
            f"<td>{int(target.get('kinematic_or_numerical_failures', 0))}</td>"
            f"<td>{int(target.get('invalid_states', 0))}</td>"
            f"<td>{int(target.get('valid_states', 0))}</td>"
            f"<td>{html.escape(f'{top_contact[0][0]} ({top_contact[0][1]})' if top_contact else '—')}</td>"
            "</tr>"
        )
    pair_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in sorted_counts(diagnostics["contact_pair_counts"])[:30]
    )
    class_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in sorted_counts(diagnostics["contact_class_counts"])
    )
    return f"""
<section class="analysis"><div class="analysis-head"><div><h2>Exact IK and collision diagnostics</h2>
<span class="muted">Collision-disabled KDL IK separates geometry/numerical failure from MoveIt state invalidity. Contacts cover the complete two-arm robot and work surface used by exact preflight; target-part AABBs are not active in this phase.</span></div></div>
<section class="stats"><div class="stat">Diagnostic cases<strong>{int(diagnostics['case_count'])}</strong></div>
<div class="stat">IK requests<strong>{int(diagnostics['ik_requests'])}</strong></div>
<div class="stat">IK cache hits<strong>{int(diagnostics['kinematic_cache_hits'])}</strong></div>
<div class="stat">IK states returned<strong>{int(diagnostics['collision_disabled_ik_solutions'])}</strong></div>
<div class="stat">No IK returned<strong>{int(diagnostics['kinematic_or_numerical_failures'])}</strong></div>
<div class="stat">Collision-invalid<strong>{int(diagnostics['invalid_states'])}</strong></div>
<div class="stat">Valid states<strong>{int(diagnostics['valid_states'])}</strong></div></section>
<section class="breakdown"><h3>Failure and validity by exact target</h3><table><thead><tr><th>Role / target</th><th>Evaluated</th><th>Stopped here</th><th>Seed evaluations</th><th>IK calls</th><th>IK cache hits</th><th>IK returned</th><th>No IK</th><th>Invalid</th><th>Valid</th><th>Top contact</th></tr></thead><tbody>{''.join(target_rows)}</tbody></table></section>
<div class="control-row"><section class="breakdown" style="flex:1"><h3>Collision classes</h3><table><thead><tr><th>Class</th><th>Contacts</th></tr></thead><tbody>{class_rows}</tbody></table></section>
<section class="breakdown" style="flex:2"><h3>Exact colliding body pairs (top 30)</h3><table><thead><tr><th>MoveIt bodies</th><th>Contacts</th></tr></thead><tbody>{pair_rows}</tbody></table></section></div></section>
"""


def _write_html(
    path: Path,
    *,
    specs: tuple[dict[str, object], ...],
    latest: Mapping[str, Mapping[str, object]],
    output_dir: Path,
) -> None:
    records = [{**spec, **dict(latest.get(str(spec["case_id"]), {}))} for spec in specs]
    for record in records:
        record.setdefault("status", "pending")
    terminal = [record for record in records if str(record.get("status")) in TERMINAL_STATUSES]
    successes = sum(bool(record.get("success", False)) for record in terminal)
    failures = sum(str(record.get("status")) == "failed" for record in terminal)
    durations = [float(record.get("duration_s", 0.0)) for record in terminal if record.get("duration_s") is not None]
    median_duration = statistics.median(durations) if durations else 0.0
    cards = []
    placement_order = {
        value: index for index, value in enumerate(dict.fromkeys(str(spec["placement_id"]) for spec in specs))
    }
    orientation_order = {
        value: index for index, value in enumerate(dict.fromkeys(str(spec["orientation_id"]) for spec in specs))
    }
    part_order = {
        value: index for index, value in enumerate(dict.fromkeys(str(spec["incoming_part_id"]) for spec in specs))
    }
    status_order = {"failed": 0, "interrupted": 1, "success": 2, "running": 3, "pending": 4}
    for record in records:
        status = str(record.get("status", "pending"))
        video = _relative(output_dir, record.get("video_path"))
        thumbnail = _relative(output_dir, record.get("thumbnail_path"))
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
            poster = f' poster="{html.escape(thumbnail)}"' if thumbnail else ""
            media_html = (
                f'<div class="video-shell"><video class="lazy-video" controls preload="none"{poster}>'
                f'<source data-src="{html.escape(video)}" type="{_video_mime(video)}"></video>'
                f'<button class="video-load" type="button">▶ Play recording</button>'
                f'<a class="video-open" href="{html.escape(video)}" target="_blank">open video</a></div>'
            )
        elif image:
            image_alt = f"Planned scene at {str(record.get('failure_phase_label', 'failure'))}"
            media_html = (
                f'<button class="failure-image-button" type="button" '
                f'data-full-src="{html.escape(image)}" data-caption="{html.escape(image_alt)}" '
                f'aria-label="Enlarge {html.escape(image_alt)}">'
                f'<img class="failure-image" src="{html.escape(image)}" alt="{html.escape(image_alt)}">'
                f'<span class="image-expand">⛶ Enlarge scene</span></button>'
            )
        else:
            media_html = '<div class="no-video">No recording or scene image</div>'
        message = str(record.get("message", ""))
        failure_key = _failure_key(record)
        failure_label = _failure_label(record)
        failure_row = f"<dt>failed at</dt><dd>{html.escape(failure_label)}</dd>" if status != "success" else ""
        failure_callout = (
            f'<div class="failure-callout">Failed at: {html.escape(failure_label)}</div>' if status == "failed" else ""
        )
        raw_case_ik_diagnostics = record.get("ik_collision_diagnostics", {})
        case_ik_diagnostics = (
            dict(raw_case_ik_diagnostics) if isinstance(raw_case_ik_diagnostics, Mapping) else {}
        )
        top_contact_pairs = sorted(
            (
                (str(name), int(count))
                for name, count in dict(case_ik_diagnostics.get("contact_pair_counts", {}) or {}).items()
            ),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        ik_rows = ""
        if case_ik_diagnostics:
            contact_text = " · ".join(f"{name} ({count})" for name, count in top_contact_pairs) or "none"
            ik_rows = (
                f"<dt>IK calls</dt><dd>{int(case_ik_diagnostics.get('ik_requests', 0))}</dd>"
                f"<dt>IK cache</dt><dd>{int(case_ik_diagnostics.get('kinematic_cache_hits', 0))} reused · "
                f"{int(case_ik_diagnostics.get('kinematic_cache_misses', 0))} solved</dd>"
                f"<dt>IK split</dt><dd>{int(case_ik_diagnostics.get('kinematic_or_numerical_failures', 0))} no IK · "
                f"{int(case_ik_diagnostics.get('invalid_states', 0))} collision-invalid · "
                f"{int(case_ik_diagnostics.get('valid_states', 0))} valid</dd>"
                f"<dt>contacts</dt><dd>{html.escape(contact_text)}</dd>"
            )
        placement_label = (
            f"{_friendly_id(record.get('placement_id', ''))} "
            f"({float(record.get('pickup_x', 0.0)):.2f}, {float(record.get('pickup_y', 0.0)):.2f} m)"
        )
        orientation_label = (
            f"{_friendly_id(record.get('orientation_id', ''))} · RPY "
            f"({float(record.get('pickup_roll_deg', 0.0)):.0f}, {float(record.get('pickup_pitch_deg', 0.0)):.0f}, "
            f"{float(record.get('pickup_yaw_deg', 0.0)):.0f})°"
        )
        search_text = " ".join(
            str(record.get(field, ""))
            for field in (
                "case_id",
                "incoming_part_id",
                "placement_id",
                "orientation_id",
                "failure_stage",
                "failure_phase_label",
                "holder_arm",
                "inserter_arm",
                "pair_id",
                "transition_id",
                "message",
            )
        ).lower()
        cards.append(
            f'<article class="case {html.escape(status)}" '
            f'data-part="{html.escape(str(record.get("incoming_part_id", "")))}" '
            f'data-part-label="Part {html.escape(str(record.get("incoming_part_id", "")))}" '
            f'data-part-order="{part_order.get(str(record.get("incoming_part_id", "")), 999)}" '
            f'data-status="{html.escape(status)}" data-status-label="{html.escape(_friendly_id(status))}" '
            f'data-status-order="{status_order.get(status, 99)}" data-success="{str(bool(record.get("success", False))).lower()}" '
            f'data-failure="{html.escape(failure_key)}" data-failure-label="{html.escape(failure_label, quote=True)}" '
            f'data-failure-order="{FAILURE_STAGE_ORDER.get(failure_key, 99)}" '
            f'data-placement="{html.escape(str(record.get("placement_id", "")))}" '
            f'data-placement-label="{html.escape(placement_label, quote=True)}" '
            f'data-placement-order="{placement_order.get(str(record.get("placement_id", "")), 999)}" '
            f'data-orientation="{html.escape(str(record.get("orientation_id", "")))}" '
            f'data-orientation-label="{html.escape(orientation_label, quote=True)}" '
            f'data-orientation-order="{orientation_order.get(str(record.get("orientation_id", "")), 999)}" '
            f'data-inserter="{html.escape(str(record.get("inserter_arm", "")))}" '
            f'data-inserter-label="{html.escape(_friendly_id(record.get("inserter_arm", "")))} inserts" '
            f'data-duration="{float(record.get("duration_s", 0.0)):.6f}" '
            f'data-case="{html.escape(str(record.get("case_id", "")), quote=True)}" '
            f'data-search="{html.escape(search_text, quote=True)}">'
            f"{media_html}"
            f'<div class="case-body"><div class="case-title">Part {html.escape(str(record.get("incoming_part_id", "")))} · '
            f"{html.escape(str(record.get('placement_id', '')))} · "
            f"{html.escape(str(record.get('orientation_id', '')))}</div>"
            f'<div class="badge">{html.escape(status)}</div>'
            f"{failure_callout}"
            f"<dl><dt>arms</dt><dd>{html.escape(str(record.get('holder_arm', '')))} holds · "
            f"{html.escape(str(record.get('inserter_arm', '')))} inserts</dd>"
            f"<dt>pickup</dt><dd>({float(record.get('pickup_x', 0.0)):.2f}, "
            f"{float(record.get('pickup_y', 0.0)):.2f}) m</dd>"
            f"<dt>RPY</dt><dd>({float(record.get('pickup_roll_deg', 0.0)):.0f}, "
            f"{float(record.get('pickup_pitch_deg', 0.0)):.0f}, "
            f"{float(record.get('pickup_yaw_deg', 0.0)):.0f})°</dd>"
            f"<dt>pair</dt><dd>{html.escape(str(record.get('pair_id', '')))}</dd>"
            f"{ik_rows}{failure_row}<dt>time</dt><dd>{float(record.get('duration_s', 0.0)):.1f} s</dd></dl>"
            f'<p class="message">{html.escape(message)}</p><div class="links">{links}</div></div></article>'
        )
    part_options = _select_options((part, f"Part {part}") for part in part_order)
    placement_options = _select_options(
        (
            placement,
            next(
                (
                    f"{_friendly_id(placement)} ({float(spec['pickup_x']):.2f}, {float(spec['pickup_y']):.2f} m)"
                    for spec in specs
                    if str(spec["placement_id"]) == placement
                ),
                _friendly_id(placement),
            ),
        )
        for placement in placement_order
    )
    orientation_options = _select_options(
        (
            orientation,
            next(
                (
                    f"{_friendly_id(orientation)} · RPY ({float(spec['pickup_roll_deg']):.0f}, "
                    f"{float(spec['pickup_pitch_deg']):.0f}, {float(spec['pickup_yaw_deg']):.0f})°"
                    for spec in specs
                    if str(spec["orientation_id"]) == orientation
                ),
                _friendly_id(orientation),
            ),
        )
        for orientation in orientation_order
    )
    present_failures = sorted(
        {_failure_key(record) for record in records}, key=lambda key: FAILURE_STAGE_ORDER.get(key, 99)
    )
    failure_options = _select_options(
        (key, FAILURE_STAGE_LABELS.get(key, _friendly_id(key))) for key in present_failures
    )
    inserter_options = _select_options(
        (arm, f"{_friendly_id(arm)} inserts") for arm in sorted({str(spec["inserter_arm"]) for spec in specs})
    )
    visual_guides = _visual_guides(specs, records)
    diagnostic_report = _ik_diagnostic_report_html(records)
    css = _dashboard_css()
    script = _dashboard_script()
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dual-Arm Assembly Benchmark</title><style>{css}</style></head><body>
<header><h1>Dual-Arm Assembly Benchmark</h1><div class="subtitle">Find where cases failed, compare groups, and inspect every recorded or rendered scene.</div></header>
<section class="stats"><div class="stat">Progress<strong>{len(terminal)} / {len(specs)}</strong></div><div class="stat">Cases shown<strong id="shown-count">{len(records)}</strong></div><div class="stat">Shown success<strong id="shown-success">{successes}</strong></div><div class="stat">Shown failed<strong id="shown-failed">{failures}</strong></div><div class="stat">Shown pass rate<strong id="shown-rate">{(100.0 * successes / len(terminal) if terminal else 0.0):.1f}%</strong></div><div class="stat">Shown median<strong id="shown-median">{median_duration:.0f}s</strong></div></section>
<main class="content">{diagnostic_report}<section class="control-bar" aria-label="Benchmark filters and sorting">
<div class="control-row"><label class="control wide"><span>Search case, pair, or error</span><input id="search" type="search" placeholder="e.g. IK failed, part 3, transition"></label>
<label class="control"><span>Part</span><select id="part"><option value="">All parts</option>{part_options}</select></label>
<label class="control"><span>Status</span><select id="status"><option value="">All statuses</option><option value="success">Success</option><option value="failed">Failed</option><option value="interrupted">Interrupted</option><option value="running">Running</option><option value="pending">Pending</option></select></label>
<label class="control"><span>Failed at</span><select id="failure"><option value="">All phases/outcomes</option>{failure_options}</select></label>
<label class="control"><span>Inserter</span><select id="inserter"><option value="">Either arm</option>{inserter_options}</select></label></div>
<div class="control-row"><label class="control"><span>Pickup location</span><select id="placement"><option value="">All locations</option>{placement_options}</select></label>
<label class="control"><span>Part orientation</span><select id="orientation"><option value="">All orientations</option>{orientation_options}</select></label>
<label class="control"><span>Case order</span><select id="case-sort"><option value="failure">Failure phase</option><option value="status">Status</option><option value="part">Assembly step / part</option><option value="placement">Pickup location</option><option value="orientation">Part orientation</option><option value="duration_desc">Runtime: longest first</option><option value="duration_asc">Runtime: shortest first</option><option value="case">Case ID</option></select></label>
<button class="reset" id="reset-filters" type="button">Reset view</button><span class="muted">Updated {html.escape(datetime.now().astimezone().isoformat(timespec="seconds"))}</span></div></section>
<section class="analysis"><div class="analysis-head"><h2>Group analysis</h2>
<label class="control"><span>Group statistics by</span><select id="group-by"><option value="failure">Failure phase</option><option value="part">Incoming part</option><option value="placement">Pickup location</option><option value="orientation">Part orientation</option><option value="inserter">Inserter arm</option><option value="status">Status</option></select></label>
<label class="control"><span>Group row order</span><select id="group-sort"><option value="failures_desc">Most failures</option><option value="cases_desc">Most cases</option><option value="rate_asc">Lowest pass rate</option><option value="duration_desc">Slowest median</option><option value="label">Name</option></select></label>
<label class="control"><span>Visual guide</span><select id="guide-dimension"><option value="follow">Follow statistics group</option><option value="failure">Failure phase</option><option value="part">Incoming part</option><option value="placement">Pickup location</option><option value="orientation">Part orientation</option><option value="inserter">Inserter arm</option><option value="status">Status</option></select></label></div>
<section class="guide-wrap">{visual_guides}</section>
<section class="breakdown"><table><thead><tr><th>Group</th><th>Cases</th><th>Pass</th><th>Failed</th><th>Pass rate</th><th>Median</th><th>Top failure phases</th></tr></thead><tbody id="breakdown-body"></tbody></table><div class="muted">Click a group row or visual tile to filter the case gallery.</div></section></section>
<div class="cases-head"><h2>Cases</h2><span class="muted" id="case-count">{len(records)} shown</span></div><section class="grid" id="case-grid">{"".join(cards)}</section></main>
<dialog id="image-viewer"><button class="viewer-close" type="button" aria-label="Close enlarged scene">×</button><img id="viewer-image" alt=""><div id="viewer-caption"></div></dialog><script>{script}</script></body></html>"""
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
        "thumbnail_path",
        "image_path",
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for spec in specs:
            record = {**spec, **dict(latest.get(str(spec["case_id"]), {}))}
            record.setdefault("status", "pending")
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
    records = [{**spec, **dict(latest.get(str(spec["case_id"]), {}))} for spec in specs]
    for record in records:
        record.setdefault("status", "pending")
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
        "ik_collision_diagnostics": _aggregate_ik_diagnostics(terminal),
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
    parser.add_argument(
        "--failed-from-summary",
        type=Path,
        default=None,
        help=(
            "Run exactly the cases whose latest status is failed in a prior "
            "benchmark summary; use a different --output-dir for a clean fixed-case report."
        ),
    )
    parser.add_argument(
        "--failure-stages",
        nargs="+",
        default=None,
        help=(
            "With --failed-from-summary, select only failures in these recorded stages, "
            "for example: moveit_candidate_planning."
        ),
    )
    parser.add_argument(
        "--planning-only",
        action="store_true",
        help="Stop each case after a complete MoveIt candidate plan; do not launch Isaac.",
    )
    parser.add_argument(
        "--ik-only",
        action="store_true",
        help="Run exact complete-state IK only; skip transition pre-ranking, OMPL, mock motion, and Isaac.",
    )
    parser.add_argument(
        "--ik-collision-diagnostics",
        action="store_true",
        help=(
            "With --ik-only, retain per-target validity totals and exact colliding body names from "
            "the normal cached-kinematics complete-state preflight."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--repair-videos",
        action="store_true",
        help="Convert legacy recordings, extract scene posters, and refresh lazy browser playback.",
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
    if args.ik_collision_diagnostics and not args.ik_only:
        raise ValueError("--ik-collision-diagnostics requires --ik-only.")
    if args.ik_only:
        args.planning_only = True
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
    failed_from_summary = None
    failure_stages = None if args.failure_stages is None else {str(value) for value in args.failure_stages}
    if failure_stages and args.failed_from_summary is None:
        raise ValueError("--failure-stages requires --failed-from-summary.")
    if args.failed_from_summary is not None:
        failed_from_summary = args.failed_from_summary.expanduser().resolve()
        if failed_from_summary == output_dir / "summary.json":
            raise ValueError("--failed-from-summary must differ from the selected output directory's summary.json.")
        specs = _failed_case_specs_from_summary(
            specs,
            summary_path=failed_from_summary,
            failure_stages=failure_stages,
        )
    manifest = {
        "schema_version": 1,
        "kind": "dual_assembly_benchmark_manifest",
        "created_or_refreshed_at": datetime.now(tz=timezone.utc).isoformat(),
        "config_path": str(config_path),
        "case_count": len(specs),
        "failed_from_summary": "" if failed_from_summary is None else str(failed_from_summary),
        "failure_stages": sorted(failure_stages or set()),
        "planning_only": bool(args.planning_only),
        "ik_only": bool(args.ik_only),
        "ik_collision_diagnostics": bool(args.ik_collision_diagnostics),
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
        converted, thumbnails, skipped = _repair_browser_videos(
            payload=payload,
            output_dir=output_dir,
            specs=specs,
            latest=latest,
            events_path=events_path,
        )
        print(
            f"[DUAL-BENCH] Browser video repair complete: converted={converted} "
            f"thumbnails={thumbnails} skipped={skipped}. "
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
            "message": (
                "MoveIt candidate planning is active." if args.planning_only else "Planning/Isaac execution is active."
            ),
            "planning_only": bool(args.planning_only),
            "ik_only": bool(args.ik_only),
            "ik_collision_diagnostics_enabled": bool(args.ik_collision_diagnostics),
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "plan_json": str(paths["plan"]),
            "attempt_json": "" if args.planning_only else str(paths["attempt"]),
            "video_path": "" if args.planning_only else str(paths["video"]),
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
        record, interrupted = _run_case(
            payload=payload,
            spec=spec,
            paths=paths,
            planning_only=bool(args.planning_only),
            ik_only=bool(args.ik_only),
            ik_collision_diagnostics=bool(args.ik_collision_diagnostics),
        )
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
