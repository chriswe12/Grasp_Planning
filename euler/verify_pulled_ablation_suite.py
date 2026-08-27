#!/usr/bin/env python3
"""Verify that a completed Euler ablation suite was pulled intact."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

FATAL_PATTERN = re.compile(
    r"Traceback|OutOfMemory|CUDA out of memory|NCCL|rank[^\n]*failed|"
    r"(?:^|[^A-Za-z0-9_])(?:nan|inf)(?:[^A-Za-z0-9_]|$)",
    flags=re.IGNORECASE | re.MULTILINE,
)
PROGRESS_PATTERN = re.compile(r"epoch:\s+(\d+)/(\d+)")


def _load_yaml(path: Path) -> dict[str, Any]:
    # RL-Games serializes tuple-valued Hydra fields with !!python/tuple tags.
    # BaseLoader deliberately keeps every scalar/container as plain data and
    # does not construct Python objects, while still allowing us to inspect the
    # two top-level string contracts needed by this verifier.
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "suite_id",
        "task",
        "experiment_family",
        "gpu_count",
        "num_envs_per_rank",
        "max_iterations",
        "seed",
        "runs",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Manifest is missing keys: {', '.join(missing)}")
    if not isinstance(payload["runs"], list) or not payload["runs"]:
        raise ValueError("Manifest runs must be a nonempty list")
    return payload


def _required_command_fragments(manifest: dict[str, Any], run: dict[str, Any]) -> tuple[str, ...]:
    return (
        f"--task {manifest['task']}",
        f"--num_envs {manifest['num_envs_per_rank']}",
        f"--max_iterations {manifest['max_iterations']}",
        f"--seed {manifest['seed']}",
        f"--sim2real_profile {run['sim2real_profile']}",
        f"--policy-context {run['policy_context']}",
        f"--experiment-name {run['experiment']}",
    )


def _verify_memory_csv(path: Path, *, max_iterations: int, min_free_mib: float) -> list[str]:
    errors: list[str] = []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return [f"{path}: no memory samples"]
    required = ("epoch", "allocated_mib", "reserved_mib", "device_used_mib", "device_free_mib")
    missing = [key for key in required if key not in rows[0]]
    if missing:
        return [f"{path}: missing columns {missing}"]
    for row_index, row in enumerate(rows, start=2):
        for key in required:
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                errors.append(f"{path}:{row_index}: invalid {key}={row.get(key)!r}")
                continue
            if not math.isfinite(value):
                errors.append(f"{path}:{row_index}: non-finite {key}={value}")
    final_epoch = int(float(rows[-1]["epoch"]))
    if final_epoch != max_iterations:
        errors.append(f"{path}: final epoch {final_epoch}, expected {max_iterations}")
    minimum_free = min(float(row["device_free_mib"]) for row in rows)
    if minimum_free < min_free_mib:
        errors.append(f"{path}: minimum free VRAM {minimum_free:.1f} MiB below {min_free_mib:.1f} MiB")
    return errors


def _verify_tensorboard_events(paths: list[Path], *, max_iterations: int) -> list[str]:
    errors: list[str] = []
    scalar_events: dict[str, list[Any]] = {}
    for path in paths:
        if path.stat().st_size == 0:
            errors.append(f"empty TensorBoard event file {path}")
            continue
        try:
            accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
            accumulator.Reload()
        except Exception as exc:  # pragma: no cover - exact parser errors vary by TensorBoard release
            errors.append(f"could not read TensorBoard event {path}: {exc}")
            continue
        for tag in accumulator.Tags().get("scalars", []):
            scalar_events.setdefault(tag, []).extend(accumulator.Scalars(tag))
    required_tags = (
        "rewards/iter",
        "Episode/success_rate",
        "Episode/collision_rate",
        "losses/completion_aux_loss",
    )
    for tag in required_tags:
        if not scalar_events.get(tag):
            errors.append(f"TensorBoard lacks required scalar {tag!r}")
    for tag, events in scalar_events.items():
        for event in events:
            if not math.isfinite(float(event.value)):
                errors.append(f"TensorBoard scalar {tag!r} is non-finite at step {event.step}")
    reward_events = scalar_events.get("rewards/iter", [])
    if reward_events:
        final_reward_step = max(int(event.step) for event in reward_events)
        if final_reward_step != max_iterations:
            errors.append(
                f"TensorBoard final rewards/iter step {final_reward_step}, expected {max_iterations}"
            )
    return errors


def verify_run(
    manifest: dict[str, Any],
    run: dict[str, Any],
    *,
    logs_root: Path,
    min_free_mib: float,
) -> dict[str, Any]:
    job_id = int(run["job_id"])
    gpu_count = int(manifest["gpu_count"])
    max_iterations = int(manifest["max_iterations"])
    errors: list[str] = []
    stdout_path = logs_root / f"slurm-{job_id}.out"
    stderr_path = logs_root / f"slurm-{job_id}.err"
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    if not stdout:
        errors.append(f"missing or empty {stdout_path}")
    if not stderr_path.is_file():
        errors.append(f"missing {stderr_path}")
    for fragment in _required_command_fragments(manifest, run):
        if fragment not in stdout:
            errors.append(f"Slurm output lacks command fragment {fragment!r}")
    completion_count = len(re.findall(r"^Training time: ", stdout, flags=re.MULTILINE))
    if completion_count != gpu_count:
        errors.append(f"found {completion_count} rank completion markers, expected {gpu_count}")
    progress = PROGRESS_PATTERN.findall(stdout)
    if not progress:
        errors.append("no PPO epoch progress marker")
    else:
        last_epoch, total_epochs = (int(value) for value in progress[-1])
        if (last_epoch, total_epochs) != (max_iterations, max_iterations):
            errors.append(
                f"last progress marker is {last_epoch}/{total_epochs}, expected "
                f"{max_iterations}/{max_iterations}"
            )
    fatal_matches = sorted({match.group(0).strip() for match in FATAL_PATTERN.finditer(stdout + "\n" + stderr)})
    if fatal_matches:
        errors.append(f"fatal log patterns: {fatal_matches}")

    run_dir = logs_root / "rl_games" / str(manifest["experiment_family"]) / str(run["experiment"])
    params_dir = run_dir / "params"
    required_files = (
        params_dir / "agent.yaml",
        params_dir / "env.yaml",
        params_dir / "sim2real_profile.yaml",
    )
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty {path}")
    checkpoints = sorted((run_dir / "nn").glob("*.pth"))
    if not checkpoints:
        errors.append(f"no checkpoint under {run_dir / 'nn'}")
    for path in checkpoints:
        if path.stat().st_size == 0:
            errors.append(f"empty checkpoint {path}")
    events = sorted((run_dir / "summaries").glob("events.out.tfevents.*"))
    if not events:
        errors.append(f"no TensorBoard event under {run_dir / 'summaries'}")
    else:
        errors.extend(_verify_tensorboard_events(events, max_iterations=max_iterations))
    memory_csvs = sorted(run_dir.glob("gpu_memory_rank_*.csv"))
    if len(memory_csvs) != gpu_count:
        errors.append(f"found {len(memory_csvs)} rank memory CSVs, expected {gpu_count}")
    for path in memory_csvs:
        errors.extend(
            _verify_memory_csv(path, max_iterations=max_iterations, min_free_mib=min_free_mib)
        )

    if all(path.is_file() for path in required_files):
        env_cfg = _load_yaml(params_dir / "env.yaml")
        profile_cfg = _load_yaml(params_dir / "sim2real_profile.yaml")
        if env_cfg.get("policy_context_mode") != run["policy_context"]:
            errors.append(
                f"serialized context {env_cfg.get('policy_context_mode')!r}, "
                f"expected {run['policy_context']!r}"
            )
        if profile_cfg.get("profile") != run["sim2real_profile"]:
            errors.append(
                f"serialized profile {profile_cfg.get('profile')!r}, "
                f"expected {run['sim2real_profile']!r}"
            )

    memory_analysis_path = logs_root / "metrics" / f"gpu-{job_id}.training-memory.json"
    memory_analysis: dict[str, Any] | None = None
    if not memory_analysis_path.is_file():
        errors.append(f"missing {memory_analysis_path}")
    else:
        memory_analysis = json.loads(memory_analysis_path.read_text(encoding="utf-8"))
        if memory_analysis.get("status") != "PASS":
            errors.append(f"memory analysis status is {memory_analysis.get('status')!r}")
        if len(memory_analysis.get("ranks", [])) != gpu_count:
            errors.append(
                f"memory analysis has {len(memory_analysis.get('ranks', []))} ranks, expected {gpu_count}"
            )

    gpu_summary_path = logs_root / "metrics" / f"gpu-{job_id}.summary.txt"
    if not gpu_summary_path.is_file():
        errors.append(f"missing {gpu_summary_path}")
    elif f"gpu_count={gpu_count}" not in gpu_summary_path.read_text(encoding="utf-8"):
        errors.append(f"{gpu_summary_path} does not report gpu_count={gpu_count}")

    return {
        "label": run["label"],
        "job_id": job_id,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "run_dir": str(run_dir),
        "checkpoint_count": len(checkpoints),
        "event_count": len(events),
        "memory_csv_count": len(memory_csvs),
        "completion_marker_count": completion_count,
        "memory_analysis": memory_analysis,
    }


def verify_suite(manifest: dict[str, Any], *, logs_root: Path, min_free_mib: float) -> dict[str, Any]:
    runs = [
        verify_run(manifest, run, logs_root=logs_root, min_free_mib=min_free_mib)
        for run in manifest["runs"]
    ]
    return {
        "suite_id": manifest["suite_id"],
        "status": "PASS" if all(run["status"] == "PASS" for run in runs) else "FAIL",
        "logs_root": str(logs_root),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--logs-root", type=Path, default=Path("logs/euler"))
    parser.add_argument("--min-free-mib", type=float, default=1024.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    manifest = _load_manifest(args.manifest)
    payload = verify_suite(manifest, logs_root=args.logs_root, min_free_mib=args.min_free_mib)
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
