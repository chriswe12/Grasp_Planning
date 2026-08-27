#!/usr/bin/env python3
"""Execute saved benchmark grasps in MuJoCo and/or Isaac with per-attempt video."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle  # noqa: E402
from grasp_planning.grasping.grasp_transforms import saved_grasp_to_world_grasp  # noqa: E402
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.ros2.moveit_pose_commander import (  # noqa: E402
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    rclpy,
)
from grasp_planning.ros2.moveit_world_grasp import world_grasp_pose_targets  # noqa: E402
from grasp_planning.start_poses import DEFAULT_ARM_START_JOINT_VALUES, DEFAULT_MOVEIT_ARM_JOINT_NAMES  # noqa: E402

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


def _optional_vec2(raw: object) -> tuple[float, float] | None:
    if raw in ("", None):
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        values: object = [part.strip() for part in stripped.split(",") if part.strip()]
    else:
        values = raw
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected world XY as a list/tuple or comma-separated string, got {raw!r}.")
    parsed = tuple(float(value) for value in values)
    if len(parsed) != 2:
        raise ValueError(f"Expected exactly 2 world XY values, got {len(parsed)} from {raw!r}.")
    return parsed


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
    return rows


def _attempt_specs_for_row(
    *,
    row: dict[str, object],
    backends: tuple[str, ...],
    grasp_ids: set[str],
    max_grasps_per_orientation: int | None,
    placement_xy_world: tuple[float, float] | None,
    max_gripper_width_m: float | None,
    gripper_width_clearance_m: float,
) -> list[dict[str, object]]:
    stage2_json = Path(str(row["stage2_json_path"]))
    bundle = load_grasp_bundle(stage2_json)
    candidates = [candidate for candidate in bundle.candidates if not grasp_ids or candidate.grasp_id in grasp_ids]
    if max_gripper_width_m is not None:
        max_width = float(max_gripper_width_m)
        clearance = float(gripper_width_clearance_m)
        candidates = [candidate for candidate in candidates if float(candidate.jaw_width) + clearance <= max_width]
    candidates = sorted(
        candidates,
        key=lambda candidate: (
            float("-inf") if candidate.score is None else float(candidate.score),
            str(candidate.grasp_id),
        ),
        reverse=True,
    )
    if max_grasps_per_orientation is not None:
        candidates = candidates[: max(0, int(max_grasps_per_orientation))]
    specs: list[dict[str, object]] = []
    for grasp_rank, candidate in enumerate(candidates, start=1):
        for backend in backends:
            specs.append(
                {
                    "assembly": str(row.get("assembly", "")),
                    "part_id": str(row.get("part_id", "")),
                    "target_mesh_path": str(row.get("target_mesh_path", "")),
                    "orientation_id": str(row.get("orientation_id", "")),
                    "generation_status": str(row.get("status", "")),
                    "stage2_json": str(stage2_json),
                    "source_stage2_json": str(stage2_json),
                    "placement_mode": "bundle_pose_shift_xy" if placement_xy_world is not None else "bundle_pose",
                    "placement_xy_world": list(placement_xy_world) if placement_xy_world is not None else None,
                    "backend": backend,
                    "grasp_id": candidate.grasp_id,
                    "grasp_rank": grasp_rank,
                    "grasp_score": candidate.score,
                    "stage2_ground_feasible_count": int(
                        row.get("stage2_ground_feasible_count", len(bundle.candidates)) or 0
                    ),
                }
            )
    return specs


def _attempt_specs_for_rows(
    *,
    rows: list[dict[str, object]],
    backends: tuple[str, ...],
    grasp_ids: set[str],
    max_grasps_per_orientation: int | None,
    placement_xy_world: tuple[float, float] | None,
    max_gripper_width_m: float | None,
    gripper_width_clearance_m: float,
    limit_orientations: int | None,
    limit_attempts: int | None,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    counted_orientations = 0
    orientation_limit = None if limit_orientations is None else max(0, int(limit_orientations))
    attempt_limit = None if limit_attempts is None else max(0, int(limit_attempts))
    for row in rows:
        row_specs = _attempt_specs_for_row(
            row=row,
            backends=backends,
            grasp_ids=grasp_ids,
            max_grasps_per_orientation=max_grasps_per_orientation,
            placement_xy_world=placement_xy_world,
            max_gripper_width_m=max_gripper_width_m,
            gripper_width_clearance_m=gripper_width_clearance_m,
        )
        if not row_specs:
            continue
        if orientation_limit is not None and counted_orientations >= orientation_limit:
            break
        counted_orientations += 1
        specs.extend(row_specs)
        if attempt_limit is not None and len(specs) >= attempt_limit:
            return specs[:attempt_limit]
    return specs


def _attempt_key(spec: dict[str, object]) -> str:
    placement_xy = spec.get("placement_xy_world")
    placement_key = "bundle"
    if placement_xy not in ("", None):
        xy = _optional_vec2(placement_xy)
        if xy is not None:
            placement_key = f"xy={xy[0]:.9g},{xy[1]:.9g}"
    return "|".join(
        [
            str(spec["backend"]),
            str(spec["assembly"]),
            str(spec["part_id"]),
            str(spec["orientation_id"]),
            str(spec["grasp_id"]),
            str(spec.get("placement_mode", "bundle_pose")),
            placement_key,
            str(spec["stage2_json"]),
        ]
    )


def _attempt_placement_id(spec: dict[str, object]) -> str:
    mode = str(spec.get("placement_mode", "bundle_pose"))
    placement_xy = spec.get("placement_xy_world")
    xy = None if placement_xy in ("", None) else _optional_vec2(placement_xy)
    if xy is None:
        return mode
    return f"{mode}_x{xy[0]:.9g}_y{xy[1]:.9g}"


def _attempt_dir(output_dir: Path, spec: dict[str, object]) -> Path:
    return (
        output_dir
        / "parts"
        / _safe_id(spec["assembly"])
        / _safe_id(spec["part_id"])
        / "orientations"
        / _safe_id(spec["orientation_id"])
        / _safe_id(spec["backend"])
        / _safe_id(_attempt_placement_id(spec))
        / _safe_id(spec["grasp_id"])
    )


def _execution_stage2_stem(spec: dict[str, object]) -> str:
    digest = hashlib.sha1(_attempt_key(spec).encode("utf-8")).hexdigest()[:12]
    pieces = [
        "stage2_execution_pose",
        str(spec["assembly"]),
        str(spec["part_id"]),
        str(spec["orientation_id"]),
        str(spec["backend"]),
        _attempt_placement_id(spec),
        str(spec["grasp_id"]),
        digest,
    ]
    return _safe_id("_".join(pieces))


def _records_for_attempt_keys(
    records: list[dict[str, object]], attempt_keys: list[str] | set[str]
) -> list[dict[str, object]]:
    attempt_key_set = set(attempt_keys)
    return [record for record in records if str(record.get("attempt_key", "")) in attempt_key_set]


def _latest_records_for_attempt_keys(
    records: list[dict[str, object]], attempt_keys: list[str]
) -> list[dict[str, object]]:
    attempt_key_set = set(attempt_keys)
    latest_by_key = {
        str(record.get("attempt_key", "")): record
        for record in records
        if str(record.get("attempt_key", "")) in attempt_key_set
    }
    return [latest_by_key[key] for key in attempt_keys if key in latest_by_key]


def _append_optional(command: list[str], flag: str, value: object) -> None:
    if value in ("", None):
        return
    command.extend([flag, str(value)])


def _optional_float(value: object, default: float) -> float:
    if value in ("", None):
        return float(default)
    return float(value)


def _optional_float_or_none(value: object) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _optional_string_tuple(value: object, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value in ("", None):
        return default
    if isinstance(value, str):
        values: object = [part.strip() for part in value.split(",") if part.strip()]
    else:
        values = value
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a string list or comma-separated string, got {value!r}.")
    parsed = tuple(str(item) for item in values)
    if not parsed:
        return default
    if any(not item for item in parsed):
        raise ValueError(f"String lists must not contain empty values: {value!r}.")
    return parsed


def _optional_float_tuple(value: object, default: tuple[float, ...] = ()) -> tuple[float, ...]:
    if value in ("", None):
        return default
    if isinstance(value, str):
        values: object = [part.strip() for part in value.split(",") if part.strip()]
    else:
        values = value
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a float list or comma-separated string, got {value!r}.")
    return tuple(float(item) for item in values)


def _moveit_joint_names_from_cfg(cfg: dict[str, object]) -> tuple[str, ...]:
    return _optional_string_tuple(cfg.get("moveit_joint_names"), DEFAULT_MOVEIT_ARM_JOINT_NAMES)


def _moveit_start_joint_positions_from_cfg(cfg: dict[str, object]) -> tuple[float, ...]:
    joint_names = _moveit_joint_names_from_cfg(cfg)
    if cfg.get("moveit_start_joint_positions") in ("", None):
        if cfg.get("moveit_joint_names") in ("", None):
            start_positions = DEFAULT_ARM_START_JOINT_VALUES
        else:
            start_positions = tuple(0.0 for _ in joint_names)
    else:
        start_positions = _optional_float_tuple(cfg.get("moveit_start_joint_positions"))
    if len(start_positions) != len(joint_names):
        raise ValueError(
            f"moveit_start_joint_positions must match the configured MoveIt joint-name count ({len(joint_names)})."
        )
    return start_positions


def _moveit_target_position_signs_from_cfg(cfg: dict[str, object]) -> tuple[float, float, float]:
    signs = _optional_float_tuple(cfg.get("moveit_target_position_signs"), (1.0, 1.0, 1.0))
    if len(signs) != 3:
        raise ValueError(f"moveit_target_position_signs must contain exactly 3 values, got {len(signs)}.")
    return (float(signs[0]), float(signs[1]), float(signs[2]))


def _tcp_to_grasp_offset_from_cfg(cfg: dict[str, object]) -> tuple[float, float, float]:
    values = cfg.get("tcp_to_grasp_offset", (0.0, 0.0, 0.0))
    if values in ("", None):
        return (0.0, 0.0, 0.0)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("isaac.tcp_to_grasp_offset must contain exactly 3 values.")
    return tuple(float(value) for value in values)


def _configured_gripper_width_clearance_m(payload: dict[str, object], backends: tuple[str, ...]) -> float:
    values: list[float] = []
    for backend in backends:
        backend_cfg = dict(payload.get(backend, {}) or {})
        raw = backend_cfg.get("gripper_width_clearance")
        values.append(_optional_float(raw, 0.01))
    return max(values) if values else 0.01


def _object_pose_from_bundle_metadata(bundle, *, bundle_path: Path) -> ObjectWorldPose:
    raw_pose = bundle.metadata.get("execution_world_pose")
    if not isinstance(raw_pose, dict):
        raise RuntimeError(f"Stage-2 bundle '{bundle_path}' is missing metadata.execution_world_pose.")
    position_world = raw_pose.get("position_world")
    orientation_xyzw_world = raw_pose.get("orientation_xyzw_world")
    if not isinstance(position_world, (list, tuple)) or len(position_world) != 3:
        raise RuntimeError(f"Stage-2 bundle '{bundle_path}' has an invalid execution_world_pose.position_world.")
    if not isinstance(orientation_xyzw_world, (list, tuple)) or len(orientation_xyzw_world) != 4:
        raise RuntimeError(
            f"Stage-2 bundle '{bundle_path}' has an invalid execution_world_pose.orientation_xyzw_world."
        )
    return ObjectWorldPose(
        position_world=tuple(float(value) for value in position_world),
        orientation_xyzw_world=tuple(float(value) for value in orientation_xyzw_world),
    )


def _moveit_config_from_benchmark_cfg(cfg: dict[str, object]) -> MoveItPoseCommanderConfig:
    return MoveItPoseCommanderConfig(
        planning_group=str(cfg.get("moveit_planning_group", "fr3_arm")),
        pose_link=str(cfg.get("moveit_pose_link", "fr3_hand_tcp")),
        joint_names=_moveit_joint_names_from_cfg(cfg),
        moveit_namespace=str(cfg.get("moveit_namespace", "")),
        pipeline_id=str(cfg.get("moveit_pipeline_id", "")),
        planner_id=str(cfg.get("moveit_planner_id", "")),
        wait_for_moveit_timeout_s=_optional_float(cfg.get("moveit_wait_for_moveit_timeout_s"), 15.0),
        ik_timeout_s=_optional_float(cfg.get("moveit_ik_timeout_s"), 2.0),
        fk_timeout_s=_optional_float(cfg.get("moveit_ik_timeout_s"), 2.0),
        planning_time_s=_optional_float(cfg.get("moveit_planning_time_s"), 5.0),
        num_planning_attempts=int(_optional_float(cfg.get("moveit_num_planning_attempts"), 5.0)),
        velocity_scale=_optional_float(cfg.get("moveit_velocity_scale"), 0.05),
        acceleration_scale=_optional_float(cfg.get("moveit_acceleration_scale"), 0.05),
        post_execute_sleep_s=0.0,
        avoid_collisions=not bool(cfg.get("moveit_allow_collisions", False)),
    )


def _trajectory_waypoints_for_joints(trajectory, *, joint_names: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    joint_trajectory = trajectory.joint_trajectory
    source_joint_names = tuple(str(name) for name in joint_trajectory.joint_names)
    name_to_index = {name: index for index, name in enumerate(source_joint_names)}
    missing = [joint_name for joint_name in joint_names if joint_name not in name_to_index]
    if missing:
        raise RuntimeError(f"MoveIt trajectory is missing arm joints: {missing}.")
    ordered_indices = [name_to_index[name] for name in joint_names]
    waypoints = tuple(
        tuple(float(point.positions[index]) for index in ordered_indices) for point in tuple(joint_trajectory.points)
    )
    if not waypoints:
        raise RuntimeError("MoveIt returned a trajectory with no points.")
    return waypoints


def _preplan_isaac_moveit(
    *,
    cfg: dict[str, object],
    spec: dict[str, object],
    execution_stage2_json: Path,
    plan_path: Path,
) -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 MoveIt dependencies are unavailable. Source the ROS2 / MoveIt workspace first.")
    bundle = load_grasp_bundle(execution_stage2_json)
    selected = next((grasp for grasp in bundle.candidates if grasp.grasp_id == str(spec["grasp_id"])), None)
    if selected is None:
        raise RuntimeError(f"Requested Isaac grasp id '{spec['grasp_id']}' is not present in {execution_stage2_json}.")
    object_pose_world = _object_pose_from_bundle_metadata(bundle, bundle_path=execution_stage2_json)
    world_grasp = saved_grasp_to_world_grasp(
        selected,
        object_pose_world,
        pregrasp_offset=_optional_float(cfg.get("pregrasp_offset"), 0.20),
        gripper_width_clearance=_optional_float(cfg.get("gripper_width_clearance"), 0.01),
    )
    min_pregrasp_z = 0.05
    if world_grasp.pregrasp_position_w[2] <= min_pregrasp_z:
        raise RuntimeError(
            f"Requested Isaac grasp id '{selected.grasp_id}' has unsafe pregrasp height: "
            f"pregrasp_position_w={world_grasp.pregrasp_position_w} required_min_z={min_pregrasp_z:.3f}"
        )
    targets = world_grasp_pose_targets(
        world_grasp,
        frame_id=str(cfg.get("moveit_frame_id", "base")),
        lift_height_m=_optional_float(cfg.get("moveit_lift_height_m", cfg.get("lift_height_m")), 0.08),
        position_signs=_moveit_target_position_signs_from_cfg(cfg),
        tcp_to_grasp_offset=_tcp_to_grasp_offset_from_cfg(cfg),
    )
    labels = ("pregrasp",) if bool(cfg.get("pregrasp_only", False)) else ("pregrasp", "grasp", "lift")
    initialized_here = False
    commander = None
    try:
        if not rclpy.ok():
            rclpy.init()
            initialized_here = True
        commander = MoveItPoseCommander(_moveit_config_from_benchmark_cfg(cfg), node_name="isaac_benchmark_moveit")
        commander.wait_for_moveit(require_execute=False)
        planned: dict[str, tuple[tuple[float, ...], ...]] = {}
        current_start = _moveit_start_joint_positions_from_cfg(cfg)
        joint_names = _moveit_joint_names_from_cfg(cfg)
        for label in labels:
            print(f"[EXEC-BENCH] preplan Isaac {label} with MoveIt/cuMotion", flush=True)
            trajectory, message = commander.plan_to_pose(
                targets[label],
                label=f"isaac_benchmark_{label}",
                start_joint_positions=current_start,
            )
            if trajectory is None:
                raise RuntimeError(f"MoveIt failed to preplan Isaac {label}: {message}")
            waypoints = _trajectory_waypoints_for_joints(trajectory, joint_names=joint_names)
            planned[label] = waypoints
            current_start = waypoints[-1]
    finally:
        if commander is not None:
            commander.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()

    _write_json(
        plan_path,
        {
            "selected_grasp_id": selected.grasp_id,
            "joint_names": list(_moveit_joint_names_from_cfg(cfg)),
            "start_joint_positions": list(_moveit_start_joint_positions_from_cfg(cfg)),
            "trajectories": {label: [list(waypoint) for waypoint in waypoints] for label, waypoints in planned.items()},
            "moveit": {
                "frame_id": str(cfg.get("moveit_frame_id", "base")),
                "target_position_signs": list(_moveit_target_position_signs_from_cfg(cfg)),
                "tcp_to_grasp_offset": list(_tcp_to_grasp_offset_from_cfg(cfg)),
                "planning_group": str(cfg.get("moveit_planning_group", "fr3_arm")),
                "pose_link": str(cfg.get("moveit_pose_link", "fr3_hand_tcp")),
                "namespace": str(cfg.get("moveit_namespace", "")),
                "pipeline_id": str(cfg.get("moveit_pipeline_id", "")),
                "planner_id": str(cfg.get("moveit_planner_id", "")),
                "lift_height_m": _optional_float(cfg.get("moveit_lift_height_m", cfg.get("lift_height_m")), 0.08),
                "allow_collisions": bool(cfg.get("moveit_allow_collisions", False)),
            },
        },
    )


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
        str(spec.get("execution_stage2_json") or spec["stage2_json"]),
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
    if cfg.get("object_mass_kg") not in (None, "") and cfg.get("object_density_kg_m3") not in (None, ""):
        raise ValueError("mujoco.object_mass_kg and object_density_kg_m3 are mutually exclusive.")
    _append_optional(command, "--object-mass-kg", cfg.get("object_mass_kg"))
    _append_optional(command, "--object-density-kg-m3", cfg.get("object_density_kg_m3"))
    _append_optional(command, "--object-scale", cfg.get("object_scale"))
    _append_optional(command, "--lift-height-m", cfg.get("lift_height_m"))
    _append_optional(command, "--success-height-margin-m", cfg.get("success_height_margin_m"))
    if bool(cfg.get("keep_generated_scene", False)):
        command.append("--keep-generated-scene")
    if str(cfg.get("controller", "native")) == "moveit":
        command.extend(
            [
                "--moveit-frame-id",
                str(cfg.get("moveit_frame_id", "base")),
                "--moveit-planning-group",
                str(cfg.get("moveit_planning_group", "fr3_arm")),
                "--moveit-pose-link",
                str(cfg.get("moveit_pose_link", "fr3_hand_tcp")),
                "--moveit-namespace",
                str(cfg.get("moveit_namespace", "")),
                "--moveit-pipeline-id",
                str(cfg.get("moveit_pipeline_id", "")),
                "--moveit-planner-id",
                str(cfg.get("moveit_planner_id", "")),
                "--moveit-wait-for-moveit-timeout-s",
                str(cfg.get("moveit_wait_for_moveit_timeout_s", 15.0)),
                "--moveit-ik-timeout-s",
                str(cfg.get("moveit_ik_timeout_s", 2.0)),
                "--moveit-planning-time-s",
                str(cfg.get("moveit_planning_time_s", 5.0)),
                "--moveit-num-planning-attempts",
                str(cfg.get("moveit_num_planning_attempts", 5)),
                "--moveit-velocity-scale",
                str(cfg.get("moveit_velocity_scale", 0.05)),
                "--moveit-acceleration-scale",
                str(cfg.get("moveit_acceleration_scale", 0.05)),
                "--moveit-execute-timeout-s",
                str(cfg.get("moveit_execute_timeout_s", 120.0)),
            ]
        )
        if bool(cfg.get("moveit_allow_collisions", False)):
            command.append("--moveit-allow-collisions")
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
    controller = str(cfg.get("controller", "moveit"))
    if controller != "moveit":
        raise ValueError(f"Unsupported Isaac benchmark controller '{controller}'. Only 'moveit' is supported.")
    command = [
        *_effective_python_command(cfg.get("python_executable", "/media/pdz/Elements1/IsaacLab/isaaclab.sh -p")),
        "scripts/run_fabrica_grasp_in_isaac.py",
        "--input-json",
        str(spec.get("execution_stage2_json") or spec["stage2_json"]),
        "--controller",
        controller,
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
    if bool(cfg.get("pregrasp_only", False)):
        command.append("--pregrasp-only")
    _append_optional(command, "--part-usd", cfg.get("part_usd"))
    if bool(cfg.get("use_provided_part_usd", False)):
        command.append("--use-provided-part-usd")
    _append_optional(command, "--fr3-usd", cfg.get("fr3_usd"))
    _append_optional(command, "--moveit-plan-json", spec.get("moveit_plan_json") or cfg.get("moveit_plan_json"))
    _append_optional(command, "--pregrasp-offset", cfg.get("pregrasp_offset"))
    _append_optional(command, "--gripper-width-clearance", cfg.get("gripper_width_clearance"))
    _append_optional(command, "--detailed-finger-contact-gap-m", cfg.get("contact_gap_m"))
    _append_optional(command, "--gripper-collision-model", cfg.get("gripper_collision_model"))
    if cfg.get("object_mass_kg") not in (None, "") and cfg.get("object_density_kg_m3") not in (None, ""):
        raise ValueError("isaac.object_mass_kg and object_density_kg_m3 are mutually exclusive.")
    _append_optional(command, "--object-mass-kg", cfg.get("object_mass_kg"))
    _append_optional(command, "--object-density-kg-m3", cfg.get("object_density_kg_m3"))
    _append_optional(command, "--success-height-margin-m", cfg.get("success_height_margin_m"))
    if cfg.get("tcp_to_grasp_offset") not in ("", None):
        command.extend(["--tcp-to-grasp-offset", *(str(value) for value in _tcp_to_grasp_offset_from_cfg(cfg))])
    command.extend(
        [
            "--moveit-frame-id",
            str(cfg.get("moveit_frame_id", "base")),
            "--moveit-target-position-signs",
            ",".join(str(value) for value in _moveit_target_position_signs_from_cfg(cfg)),
            "--moveit-planning-group",
            str(cfg.get("moveit_planning_group", "fr3_arm")),
            "--moveit-pose-link",
            str(cfg.get("moveit_pose_link", "fr3_hand_tcp")),
            "--moveit-namespace",
            str(cfg.get("moveit_namespace", "")),
            "--moveit-joint-names",
            ",".join(_moveit_joint_names_from_cfg(cfg)),
            "--moveit-start-joint-positions",
            ",".join(str(value) for value in _moveit_start_joint_positions_from_cfg(cfg)),
            "--moveit-pipeline-id",
            str(cfg.get("moveit_pipeline_id", "")),
            "--moveit-planner-id",
            str(cfg.get("moveit_planner_id", "")),
            "--moveit-wait-for-moveit-timeout-s",
            str(cfg.get("moveit_wait_for_moveit_timeout_s", 15.0)),
            "--moveit-ik-timeout-s",
            str(cfg.get("moveit_ik_timeout_s", 2.0)),
            "--moveit-planning-time-s",
            str(cfg.get("moveit_planning_time_s", 5.0)),
            "--moveit-num-planning-attempts",
            str(cfg.get("moveit_num_planning_attempts", 5)),
            "--moveit-velocity-scale",
            str(cfg.get("moveit_velocity_scale", 0.05)),
            "--moveit-acceleration-scale",
            str(cfg.get("moveit_acceleration_scale", 0.05)),
            "--moveit-lift-height-m",
            str(cfg.get("moveit_lift_height_m", cfg.get("lift_height_m", 0.08))),
            "--moveit-execution-speed-rad-s",
            str(cfg.get("moveit_execution_speed_rad_s", 0.35)),
            "--moveit-grasp-settle-time-s",
            str(cfg.get("moveit_grasp_settle_time_s", 0.0)),
            "--gripper-close-duration-s",
            str(cfg.get("gripper_close_duration_s", 1.5)),
            "--gripper-close-max-duration-s",
            str(cfg.get("gripper_close_max_duration_s", 10.0)),
            "--postclose-hold-s",
            str(cfg.get("postclose_hold_s", 1.0)),
        ]
    )
    if bool(cfg.get("moveit_allow_collisions", False)):
        command.append("--moveit-allow-collisions")
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


def _read_text_tail(path: Path, *, max_bytes: int = 8192) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - int(max_bytes)))
        return stream.read().decode("utf-8", errors="replace")


def _missing_artifact_summary(*, returncode: int, stderr_path: Path) -> dict[str, object]:
    stderr_tail = _read_text_tail(stderr_path)
    if int(returncode) != 0:
        return {
            "success": False,
            "status": "runner_failed",
            "message": f"Runner exited with code {int(returncode)}; see stderr.log.",
        }
    if "Traceback (most recent call last)" in stderr_tail:
        return {
            "success": False,
            "status": "runner_failed",
            "message": "Runner exited without writing an attempt artifact after a Python traceback; see stderr.log.",
        }
    return {"success": False, "status": "artifact_missing", "message": "Attempt artifact was not written."}


def _prepare_execution_stage2_json(
    *,
    source_stage2_json: Path,
    attempt_dir: Path,
    placement_xy_world: tuple[float, float] | None,
    execution_stem: str = "stage2_execution_pose",
) -> tuple[Path, dict[str, object] | None, dict[str, object] | None]:
    if placement_xy_world is None:
        return source_stage2_json, None, None

    payload = json.loads(source_stage2_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in stage-2 bundle '{source_stage2_json}'.")
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Stage-2 bundle '{source_stage2_json}' has non-object metadata.")
    raw_pose = metadata.get("execution_world_pose")
    if not isinstance(raw_pose, dict):
        raise ValueError(
            f"Execution benchmark placement.xy_world requires metadata.execution_world_pose in '{source_stage2_json}'."
        )
    position_world = raw_pose.get("position_world")
    orientation_xyzw_world = raw_pose.get("orientation_xyzw_world")
    if not isinstance(position_world, (list, tuple)) or len(position_world) != 3:
        raise ValueError(f"Stage-2 bundle '{source_stage2_json}' has an invalid execution_world_pose.position_world.")
    if not isinstance(orientation_xyzw_world, (list, tuple)) or len(orientation_xyzw_world) != 4:
        raise ValueError(
            f"Stage-2 bundle '{source_stage2_json}' has an invalid execution_world_pose.orientation_xyzw_world."
        )

    original_pose = {
        "position_world": [float(value) for value in position_world],
        "orientation_xyzw_world": [float(value) for value in orientation_xyzw_world],
    }
    execution_pose = {
        "position_world": [float(placement_xy_world[0]), float(placement_xy_world[1]), float(position_world[2])],
        "orientation_xyzw_world": [float(value) for value in orientation_xyzw_world],
    }
    metadata["execution_world_pose"] = execution_pose
    metadata["execution_benchmark_source_stage2_json"] = str(source_stage2_json)
    metadata["execution_benchmark_original_execution_world_pose"] = original_pose
    metadata["execution_benchmark_placement_xy_world"] = [float(placement_xy_world[0]), float(placement_xy_world[1])]

    execution_stage2_json = attempt_dir / f"{_safe_id(execution_stem)}.json"
    _write_json(execution_stage2_json, payload)
    return execution_stage2_json, original_pose, execution_pose


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
        "lift_height_m": execution.get("lift_height_m", execution.get("object_lift_height_m")),
        "target_lift_height_m": execution.get("target_lift_height_m"),
        "video_path": video.get("path") or execution.get("video_path"),
        "video_frame_count": int(video.get("frame_count") or execution.get("video_frame_count") or 0),
    }


def _write_isaac_preplan_failure_artifact(
    *,
    path: Path,
    run_spec: dict[str, object],
    cfg: dict[str, object],
    message: str,
    duration_s: float,
    stdout_path: Path,
    stderr_path: Path,
    plan_path: Path,
) -> None:
    _write_json(
        path,
        {
            "attempt": _json_safe(run_spec),
            "execution": {
                "controller": str(cfg.get("controller", "moveit")),
                "success": False,
                "status": "moveit_preplan_failed",
                "message": message,
                "duration_s": float(duration_s),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            },
            "moveit": {
                "frame_id": str(cfg.get("moveit_frame_id", "base")),
                "planning_group": str(cfg.get("moveit_planning_group", "fr3_arm")),
                "pose_link": str(cfg.get("moveit_pose_link", "fr3_hand_tcp")),
                "namespace": str(cfg.get("moveit_namespace", "")),
                "joint_names": list(_moveit_joint_names_from_cfg(cfg)),
                "pipeline_id": str(cfg.get("moveit_pipeline_id", "")),
                "planner_id": str(cfg.get("moveit_planner_id", "")),
                "lift_height_m": _optional_float(cfg.get("moveit_lift_height_m", cfg.get("lift_height_m")), 0.08),
                "allow_collisions": bool(cfg.get("moveit_allow_collisions", False)),
                "plan_json": str(plan_path),
            },
        },
    )


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
    video_path = attempt_dir / "attempt.webm" if record_video else None
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt_artifact.unlink(missing_ok=True)
    if video_path is not None:
        video_path.unlink(missing_ok=True)

    run_spec = dict(spec)
    placement_xy_world = _optional_vec2(spec.get("placement_xy_world"))
    source_stage2_json = Path(str(spec.get("source_stage2_json") or spec["stage2_json"]))
    execution_stage2_json, original_pose, execution_pose = _prepare_execution_stage2_json(
        source_stage2_json=source_stage2_json,
        attempt_dir=attempt_dir,
        placement_xy_world=placement_xy_world,
        execution_stem=_execution_stage2_stem(run_spec),
    )
    run_spec["source_stage2_json"] = str(source_stage2_json)
    run_spec["execution_stage2_json"] = str(execution_stage2_json)
    if original_pose is not None:
        run_spec["original_execution_world_pose"] = original_pose
    if execution_pose is not None:
        run_spec["execution_world_pose"] = execution_pose
        run_spec["execution_object_position_world"] = execution_pose["position_world"]

    if backend == "mujoco":
        command = _mujoco_command(
            cfg=dict(payload.get("mujoco", {}) or {}),
            spec=run_spec,
            attempt_artifact=attempt_artifact,
            video_path=video_path,
        )
    elif backend == "isaac":
        isaac_cfg = dict(payload.get("isaac", {}) or {})
        controller = str(isaac_cfg.get("controller", "moveit"))
        if controller != "moveit":
            raise ValueError(f"Unsupported Isaac benchmark controller '{controller}'. Only 'moveit' is supported.")
        plan_path = attempt_dir / "moveit_plan.json"
        plan_path.unlink(missing_ok=True)
        preplan_started = time.perf_counter()
        try:
            _preplan_isaac_moveit(
                cfg=isaac_cfg,
                spec=run_spec,
                execution_stage2_json=execution_stage2_json,
                plan_path=plan_path,
            )
        except Exception as exc:
            stdout_path.write_text("", encoding="utf-8")
            with stderr_path.open("w", encoding="utf-8") as stderr:
                traceback.print_exception(type(exc), exc, exc.__traceback__, file=stderr)
            duration_s = float(time.perf_counter() - preplan_started)
            _write_isaac_preplan_failure_artifact(
                path=attempt_artifact,
                run_spec=run_spec,
                cfg=isaac_cfg,
                message=str(exc),
                duration_s=duration_s,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                plan_path=plan_path,
            )
            return {
                **run_spec,
                "attempt_key": _attempt_key(spec),
                "returncode": 1,
                "duration_s": duration_s,
                "attempt_artifact": str(attempt_artifact),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "command": [],
                "success": False,
                "status": "moveit_preplan_failed",
                "message": str(exc),
                "lift_height_m": None,
                "target_lift_height_m": None,
                "video_path": None,
                "video_frame_count": 0,
            }
        run_spec["moveit_plan_json"] = str(plan_path)
        command = _isaac_command(
            cfg=isaac_cfg,
            spec=run_spec,
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
    if summary["status"] == "artifact_missing":
        summary.update(_missing_artifact_summary(returncode=int(completed.returncode), stderr_path=stderr_path))
    return {
        **run_spec,
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
        "grasp_rank",
        "grasp_score",
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
        "source_stage2_json",
        "execution_stage2_json",
        "placement_mode",
        "placement_xy_world",
        "execution_object_position_world",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def _record_success(record: dict[str, object]) -> bool:
    return bool(record.get("success"))


def _record_status(record: dict[str, object]) -> str:
    return str(record.get("status", "unknown") or "unknown")


def _record_tone(record: dict[str, object]) -> str:
    if _record_success(record):
        return "ok"
    status = _record_status(record)
    if status in {"artifact_missing", "runner_failed"}:
        return "bad"
    if "failed" in status or "collision" in status or "invalid" in status:
        return "fail"
    return "warn"


def _rate(successes: int, total: int) -> str:
    return "0.0%" if total <= 0 else f"{100.0 * successes / total:.1f}%"


def _option_tags(values: list[str]) -> str:
    options = ['<option value="">All</option>']
    options.extend(
        f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>' for value in values
    )
    return "".join(options)


def _breakdown_html(counter: Counter[str], total: int) -> str:
    if not counter:
        return '<div class="muted">No attempts.</div>'
    rows = []
    for label, count in counter.most_common():
        pct = 0.0 if total <= 0 else 100.0 * count / total
        rows.append(
            '<div class="bar-row">'
            f'<div class="bar-label">{html.escape(str(label))}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{pct:.3f}%"></div>'
            "</div>"
            f'<div class="bar-value">{count}</div>'
            "</div>"
        )
    return "".join(rows)


def _overview_cell(records: list[dict[str, object]], output_dir: Path) -> str:
    if not records:
        return '<td class="matrix-cell empty"><span class="muted">not run</span></td>'
    success_count = sum(1 for record in records if _record_success(record))
    if success_count == len(records):
        tone = "ok"
    elif success_count == 0:
        tone = "fail"
    else:
        tone = "mixed"
    statuses = ", ".join(sorted({_record_status(record) for record in records}))
    chips = []
    for record in sorted(records, key=lambda item: int(item.get("grasp_rank") or 0)):
        anchor = html.escape(str(record.get("_anchor", "")), quote=True)
        title = html.escape(
            f"{record.get('grasp_id', '')} | {_record_status(record)} | {record.get('message', '')}",
            quote=True,
        )
        chips.append(
            f'<a class="chip {_record_tone(record)}" href="#{anchor}" title="{title}">'
            f"{html.escape(str(record.get('grasp_rank', '')))}:{html.escape(str(record.get('grasp_id', '')))}</a>"
        )
    video_links = []
    for record in records:
        video = _relative(output_dir, record.get("video_path"))
        if video:
            video_links.append(f'<a href="{html.escape(video)}">video</a>')
    videos = f'<div class="cell-links">{" ".join(video_links[:3])}</div>' if video_links else ""
    return (
        f'<td class="matrix-cell {tone}">'
        f'<div class="cell-score">{success_count}/{len(records)} ok</div>'
        f'<div class="cell-status">{html.escape(statuses)}</div>'
        f'<div class="cell-chips">{"".join(chips)}</div>'
        f"{videos}</td>"
    )


def _write_overview_html(path: Path, *, output_dir: Path, records: list[dict[str, object]]) -> None:
    records = [dict(record, _anchor=f"attempt-{index}") for index, record in enumerate(records, start=1)]
    backend_values = sorted({str(record.get("backend", "")) for record in records if record.get("backend", "") != ""})
    assembly_values = sorted(
        {str(record.get("assembly", "")) for record in records if record.get("assembly", "") != ""}
    )
    part_values = sorted({str(record.get("part_id", "")) for record in records if record.get("part_id", "") != ""})
    status_values = sorted({_record_status(record) for record in records})
    success_count = sum(1 for record in records if _record_success(record))
    status_counts = Counter(_record_status(record) for record in records)

    backend_panels = []
    for backend in backend_values:
        backend_records = [record for record in records if str(record.get("backend", "")) == backend]
        backend_successes = sum(1 for record in backend_records if _record_success(record))
        backend_panels.append(
            '<section class="panel">'
            f"<h3>{html.escape(backend)}</h3>"
            f'<div class="panel-number">{backend_successes}/{len(backend_records)}</div>'
            f'<div class="muted">{_rate(backend_successes, len(backend_records))} success</div>'
            f"{_breakdown_html(Counter(_record_status(record) for record in backend_records), len(backend_records))}"
            "</section>"
        )

    matrix: dict[tuple[str, str, str], dict[str, list[dict[str, object]]]] = {}
    for record in records:
        key = (
            str(record.get("assembly", "")),
            str(record.get("part_id", "")),
            str(record.get("orientation_id", "")),
        )
        backend = str(record.get("backend", "unknown"))
        matrix.setdefault(key, {}).setdefault(backend, []).append(record)
    matrix_rows = []
    for (assembly, part_id, orientation_id), by_backend in sorted(matrix.items()):
        row_records = [record for backend_records in by_backend.values() for record in backend_records]
        row_success = bool(row_records) and all(_record_success(record) for record in row_records)
        statuses = " ".join(sorted({_record_status(record) for record in row_records}))
        backends = " ".join(sorted(by_backend))
        search = " ".join(
            [
                assembly,
                part_id,
                orientation_id,
                backends,
                statuses,
                " ".join(str(record.get("grasp_id", "")) for record in row_records),
            ]
        )
        cells = "".join(_overview_cell(by_backend.get(backend, []), output_dir) for backend in backend_values)
        matrix_rows.append(
            f'<tr data-assembly="{html.escape(assembly, quote=True)}" data-part="{html.escape(part_id, quote=True)}" '
            f'data-orientation="{html.escape(orientation_id, quote=True)}" '
            f'data-backends="{html.escape(backends, quote=True)}" data-statuses="{html.escape(statuses, quote=True)}" '
            f'data-success="{str(row_success).lower()}" data-search="{html.escape(search, quote=True)}">'
            f"<th>{html.escape(assembly)}</th><th>{html.escape(part_id)}</th><th>{html.escape(orientation_id)}</th>{cells}</tr>"
        )
    matrix_head = "".join(f"<th>{html.escape(backend)}</th>" for backend in backend_values)

    failures = [record for record in records if not _record_success(record)]
    failure_rows = []
    grouped_failures: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in failures:
        grouped_failures.setdefault((str(record.get("backend", "")), _record_status(record)), []).append(record)
    for (backend, status), group in sorted(grouped_failures.items(), key=lambda item: (-len(item[1]), item[0])):
        sample = group[0]
        failure_rows.append(
            "<tr>"
            f'<td>{html.escape(backend)}</td><td><span class="pill fail">{html.escape(status)}</span></td>'
            f"<td>{len(group)}</td>"
            f"<td>{html.escape(', '.join(sorted({str(record.get('part_id', '')) for record in group}))[:160])}</td>"
            f"<td>{html.escape(', '.join(sorted({str(record.get('orientation_id', '')) for record in group}))[:160])}</td>"
            f'<td><a href="#{html.escape(str(sample.get("_anchor", "")), quote=True)}">sample</a></td>'
            f"<td>{html.escape(str(sample.get('message', '')))}</td>"
            "</tr>"
        )
    failure_table = (
        '<table class="compact"><thead><tr><th>Backend</th><th>Status</th><th>Count</th><th>Parts</th>'
        "<th>Orientations</th><th>Jump</th><th>Sample Message</th></tr></thead>"
        f"<tbody>{''.join(failure_rows)}</tbody></table>"
        if failure_rows
        else '<div class="muted">No failures.</div>'
    )

    detail_rows = []
    for record in records:
        artifact = _relative(output_dir, record.get("attempt_artifact"))
        video = _relative(output_dir, record.get("video_path"))
        data_search = " ".join(
            str(record.get(key, ""))
            for key in ("assembly", "part_id", "orientation_id", "backend", "grasp_id", "status", "message")
        )
        detail_rows.append(
            f'<tr id="{html.escape(str(record.get("_anchor", "")), quote=True)}" '
            f'data-assembly="{html.escape(str(record.get("assembly", "")), quote=True)}" '
            f'data-part="{html.escape(str(record.get("part_id", "")), quote=True)}" '
            f'data-backend="{html.escape(str(record.get("backend", "")), quote=True)}" '
            f'data-statuses="{html.escape(_record_status(record), quote=True)}" '
            f'data-success="{str(_record_success(record)).lower()}" '
            f'data-search="{html.escape(data_search, quote=True)}">'
            f"<td>{html.escape(str(record.get('assembly', '')))}</td>"
            f"<td>{html.escape(str(record.get('part_id', '')))}</td>"
            f"<td>{html.escape(str(record.get('orientation_id', '')))}</td>"
            f"<td>{html.escape(str(record.get('backend', '')))}</td>"
            f"<td>{html.escape(str(record.get('grasp_id', '')))}</td>"
            f"<td>{html.escape(str(record.get('grasp_rank', '')))}</td>"
            f'<td><span class="pill {_record_tone(record)}">{html.escape(_record_status(record))}</span></td>'
            f"<td>{html.escape(str(record.get('message', '')))}</td>"
            f'<td><a href="{html.escape(artifact)}">artifact</a>'
            + (f'<br><a href="{html.escape(video)}">video</a>' if video else "")
            + "</td></tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Grasp Execution Overview</title>
  <style>
    :root {{ --ok:#15803d; --ok-bg:#dcfce7; --fail:#b91c1c; --fail-bg:#fee2e2; --bad:#7f1d1d; --bad-bg:#fecaca; --warn:#b45309; --warn-bg:#fef3c7; --mixed-bg:#e0f2fe; --line:#e5e7eb; --muted:#6b7280; }}
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    h1 {{ margin: 0 0 4px; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: white; position: sticky; top: 0; z-index: 2; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .summary-grid, .panel-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 16px 0; }}
    .stat, .panel {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #f9fafb; }}
    .stat-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .stat-value, .panel-number {{ font-size: 26px; font-weight: 700; }}
    .filters {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: end; border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin: 16px 0; }}
    .filters label {{ display: grid; gap: 4px; color: var(--muted); font-size: 12px; }}
    .filters select, .filters input[type=search] {{ border: 1px solid #d1d5db; border-radius: 6px; min-height: 32px; padding: 4px 8px; font: inherit; }}
    .filters input[type=search] {{ min-width: 260px; }}
    .matrix-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
    .matrix-table th {{ position: static; background: #f9fafb; }}
    .matrix-cell {{ min-width: 190px; border-left: 1px solid var(--line); }}
    .matrix-cell.ok {{ background: var(--ok-bg); }}
    .matrix-cell.fail {{ background: var(--fail-bg); }}
    .matrix-cell.mixed {{ background: var(--mixed-bg); }}
    .matrix-cell.empty {{ background: #f3f4f6; }}
    .cell-score {{ font-weight: 700; }}
    .cell-status, .cell-links {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
    .cell-chips {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }}
    .chip, .pill {{ display: inline-flex; border-radius: 999px; padding: 2px 7px; font-size: 12px; text-decoration: none; border: 1px solid transparent; }}
    .chip.ok, .pill.ok {{ color: var(--ok); background: var(--ok-bg); border-color: #86efac; }}
    .chip.fail, .pill.fail {{ color: var(--fail); background: var(--fail-bg); border-color: #fca5a5; }}
    .chip.bad, .pill.bad {{ color: var(--bad); background: var(--bad-bg); border-color: #f87171; }}
    .chip.warn, .pill.warn {{ color: var(--warn); background: var(--warn-bg); border-color: #fcd34d; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(90px, 1fr) 2fr 40px; gap: 8px; align-items: center; margin: 7px 0; }}
    .bar-label, .bar-value {{ color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ height: 9px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: #2563eb; }}
    .attempts tbody tr:target {{ outline: 3px solid #2563eb; outline-offset: -3px; background: #eff6ff; }}
    .hidden {{ display: none !important; }}
  </style>
</head>
<body>
  <h1>Grasp Execution Overview</h1>
  <p class="muted">High-level success/failure view. The original video table is still available at <a href="index.html">index.html</a>.</p>
  <section class="summary-grid">
    <div class="stat"><div class="stat-label">Attempts</div><div class="stat-value">{len(records)}</div></div>
    <div class="stat"><div class="stat-label">Successes</div><div class="stat-value">{success_count}</div><div class="muted">{_rate(success_count, len(records))}</div></div>
    <div class="stat"><div class="stat-label">Failures</div><div class="stat-value">{len(records) - success_count}</div></div>
    <div class="stat"><div class="stat-label">Videos</div><div class="stat-value">{sum(1 for record in records if record.get("video_path"))}</div></div>
  </section>
  <section class="filters">
    <label>Assembly<select id="assemblyFilter">{_option_tags(assembly_values)}</select></label>
    <label>Part<select id="partFilter">{_option_tags(part_values)}</select></label>
    <label>Backend<select id="backendFilter">{_option_tags(backend_values)}</select></label>
    <label>Status<select id="statusFilter">{_option_tags(status_values)}</select></label>
    <label>Search<input id="searchFilter" type="search" placeholder="orientation, grasp id, message"></label>
    <label><span>Only Failures</span><input id="failureOnlyFilter" type="checkbox"></label>
  </section>
  <h2>Backend Breakdown</h2>
  <div class="panel-grid">
    {"".join(backend_panels)}
    <section class="panel"><h3>Status Breakdown</h3>{_breakdown_html(status_counts, len(records))}</section>
  </div>
  <h2>Assembly / Part / Orientation Matrix</h2>
  <div class="matrix-wrap"><table class="matrix-table"><thead><tr><th>Assembly</th><th>Part</th><th>Orientation</th>{matrix_head}</tr></thead><tbody>{"".join(matrix_rows)}</tbody></table></div>
  <h2>Failure Groups</h2>
  {failure_table}
  <h2>Attempt Index</h2>
  <table class="attempts"><thead><tr><th>Assembly</th><th>Part</th><th>Orientation</th><th>Backend</th><th>Grasp</th><th>Rank</th><th>Status</th><th>Message</th><th>Links</th></tr></thead><tbody>{"".join(detail_rows)}</tbody></table>
  <script>
    const controls = {{
      assembly: document.getElementById("assemblyFilter"),
      part: document.getElementById("partFilter"),
      backend: document.getElementById("backendFilter"),
      status: document.getElementById("statusFilter"),
      search: document.getElementById("searchFilter"),
      failureOnly: document.getElementById("failureOnlyFilter"),
    }};
    function tokenMatch(value, token) {{
      return !token || (value || "") === token || (` ${{value || ""}} `).includes(` ${{token}} `);
    }}
    function visible(row) {{
      const search = controls.search.value.trim().toLowerCase();
      if (controls.assembly.value && row.dataset.assembly !== controls.assembly.value) return false;
      if (controls.part.value && row.dataset.part !== controls.part.value) return false;
      if (controls.backend.value && !tokenMatch(row.dataset.backends || row.dataset.backend, controls.backend.value)) return false;
      if (controls.status.value && !tokenMatch(row.dataset.statuses || row.dataset.status, controls.status.value)) return false;
      if (controls.failureOnly.checked && row.dataset.success === "true") return false;
      if (search && !(row.dataset.search || "").toLowerCase().includes(search)) return false;
      return true;
    }}
    function applyFilters() {{
      document.querySelectorAll(".matrix-table tbody tr, .attempts tbody tr").forEach((row) => row.classList.toggle("hidden", !visible(row)));
    }}
    Object.values(controls).forEach((control) => control.addEventListener("input", applyFilters));
    applyFilters();
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def _write_index_html(path: Path, *, output_dir: Path, records: list[dict[str, object]]) -> None:
    status_counts = Counter(str(record.get("status", "unknown")) for record in records)
    backend_counts = Counter(str(record.get("backend", "unknown")) for record in records)
    rows_html = []
    for record in records:
        duration_text = f"{float(record.get('duration_s', 0.0)):.1f}"
        video = _relative(output_dir, record.get("video_path"))
        video_html = ""
        if video:
            suffix = Path(video).suffix.lower()
            video_type = "video/webm" if suffix == ".webm" else "video/mp4" if suffix == ".mp4" else ""
            type_attr = f' type="{video_type}"' if video_type else ""
            video_html = (
                f'<video controls preload="metadata" width="320">'
                f'<source src="{html.escape(video)}"{type_attr}>'
                f"</video>"
                f'<div><a href="{html.escape(video)}">video</a></div>'
            )
        artifact = _relative(output_dir, record.get("attempt_artifact"))
        stderr = _relative(output_dir, record.get("stderr_log"))
        placement = record.get("execution_object_position_world") or record.get("placement_xy_world") or ""
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('assembly', '')))}</td>"
            f"<td>{html.escape(str(record.get('part_id', '')))}</td>"
            f"<td>{html.escape(str(record.get('orientation_id', '')))}</td>"
            f"<td>{html.escape(str(record.get('backend', '')))}</td>"
            f"<td>{html.escape(str(record.get('grasp_id', '')))}</td>"
            f"<td>{html.escape(str(record.get('grasp_rank', '')))}</td>"
            f"<td>{html.escape(str(record.get('grasp_score', '')))}</td>"
            f"<td>{html.escape(str(placement))}</td>"
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
        <th>Assembly</th><th>Part</th><th>Orientation</th><th>Backend</th><th>Grasp</th><th>Rank</th><th>Score</th><th>Object Position</th>
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
    placement = dict(effective.get("placement", {}) or {})
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
    if args.placement_xy_world is not None:
        placement["xy_world"] = list(_optional_vec2(args.placement_xy_world) or ())
    if args.use_bundle_placement:
        placement["xy_world"] = None
    effective["execution_benchmark"] = benchmark
    effective["selection"] = selection
    effective["placement"] = placement
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
    parser.add_argument(
        "--placement-xy-world",
        type=str,
        default=None,
        help="Execution object XY as x,y. Keeps each bundle's saved orientation and Z; default comes from config.",
    )
    parser.add_argument(
        "--use-bundle-placement",
        action="store_true",
        help="Disable benchmark XY relocation and execute with each stage-2 bundle's saved execution_world_pose.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _apply_cli_overrides(_load_yaml(args.config), args)
    benchmark_cfg = dict(payload.get("execution_benchmark", {}) or {})
    selection = dict(payload.get("selection", {}) or {})
    placement_cfg = dict(payload.get("placement", {}) or {})
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
    max_gripper_width_m = _optional_float_or_none(selection.get("max_gripper_width_m"))
    gripper_width_clearance_m = _configured_gripper_width_clearance_m(payload, backends)
    placement_xy_world = _optional_vec2(placement_cfg.get("xy_world"))
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
    )
    specs = _attempt_specs_for_rows(
        rows=orientation_rows,
        backends=backends,
        grasp_ids=grasp_ids,
        max_grasps_per_orientation=max_grasps,
        placement_xy_world=placement_xy_world,
        max_gripper_width_m=max_gripper_width_m,
        gripper_width_clearance_m=gripper_width_clearance_m,
        limit_orientations=limit_orientations,
        limit_attempts=limit_attempts,
    )
    if not specs:
        raise RuntimeError("No execution attempts matched the requested filters.")

    attempt_keys = [_attempt_key(spec) for spec in specs]
    jsonl_path = output_dir / "attempts.jsonl"
    if resume:
        existing_records = _records_for_attempt_keys(_jsonl_records(jsonl_path), attempt_keys)
    else:
        existing_records = []
        if jsonl_path.exists():
            jsonl_path.write_text("", encoding="utf-8")
    existing_by_key = {str(record.get("attempt_key", "")): record for record in existing_records}
    completed_success = {
        key
        for key, record in existing_by_key.items()
        if key and (not args.rerun_failed or bool(record.get("success", False)))
    }

    records = list(existing_records)
    print(
        f"[EXEC-BENCH] attempts={len(specs)} backends={backends} record_video={record_video} "
        f"resume={resume} placement_xy_world={placement_xy_world} output_dir={output_dir}",
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

    final_records = _latest_records_for_attempt_keys(records, attempt_keys)
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
    _write_overview_html(output_dir / "overview.html", output_dir=output_dir, records=final_records)
    print(f"[EXEC-BENCH] Wrote {output_dir / 'results.json'}", flush=True)
    print(f"[EXEC-BENCH] Wrote {output_dir / 'index.html'}", flush=True)
    print(f"[EXEC-BENCH] Wrote {output_dir / 'overview.html'}", flush=True)


if __name__ == "__main__":
    main()
