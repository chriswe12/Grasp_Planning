from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import yaml
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

MODULE_PATH = Path(__file__).parents[1] / "euler" / "verify_pulled_ablation_suite.py"
SPEC = importlib.util.spec_from_file_location("verify_pulled_ablation_suite", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def _manifest() -> dict[str, object]:
    return {
        "suite_id": "test-suite",
        "task": "Test-Task-v0",
        "experiment_family": "test_family",
        "gpu_count": 3,
        "num_envs_per_rank": 224,
        "max_iterations": 3,
        "seed": 42,
        "runs": [
            {
                "label": "baseline",
                "job_id": 123,
                "experiment": "test_experiment",
                "sim2real_profile": "combined_sim2real",
                "policy_context": "action",
            }
        ],
    }


def _write_complete_fixture(logs_root: Path) -> None:
    run_dir = logs_root / "rl_games/test_family/test_experiment"
    params_dir = run_dir / "params"
    params_dir.mkdir(parents=True)
    (run_dir / "nn").mkdir()
    (run_dir / "summaries").mkdir()
    command = (
        "--task Test-Task-v0 --num_envs 224 --max_iterations 3 --seed 42 "
        "--sim2real_profile combined_sim2real --policy-context action "
        "--experiment-name test_experiment"
    )
    stdout = f"[INFO] Running mode=train: {command}\nfps total: 1 epoch: 3/3\n" + "Training time: 1\n" * 3
    (logs_root / "slurm-123.out").write_text(stdout, encoding="utf-8")
    (logs_root / "slurm-123.err").write_text("", encoding="utf-8")
    (params_dir / "agent.yaml").write_text("params: {}\n", encoding="utf-8")
    (params_dir / "env.yaml").write_text(
        yaml.safe_dump({"policy_context_mode": "action"}), encoding="utf-8"
    )
    (params_dir / "sim2real_profile.yaml").write_text(
        "profile: combined_sim2real\n"
        "overrides:\n"
        "  live_rgb_gamma: !!python/tuple\n"
        "  - 0.9\n"
        "  - 1.1\n",
        encoding="utf-8",
    )
    (run_dir / "nn/policy.pth").write_bytes(b"checkpoint")
    writer = EventFileWriter(str(run_dir / "summaries"))
    for tag, value in (
        ("rewards/iter", 1.0),
        ("Episode/success_rate", 0.5),
        ("Episode/collision_rate", 0.0),
        ("losses/completion_aux_loss", 0.1),
    ):
        writer.add_event(
            Event(
                wall_time=1.0,
                step=3,
                summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]),
            )
        )
    writer.flush()
    writer.close()
    fieldnames = (
        "epoch",
        "allocated_mib",
        "reserved_mib",
        "device_used_mib",
        "device_free_mib",
    )
    for rank in range(3):
        with (run_dir / f"gpu_memory_rank_{rank}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "epoch": 3,
                    "allocated_mib": 8000,
                    "reserved_mib": 9000,
                    "device_used_mib": 19000,
                    "device_free_mib": 4500,
                }
            )
    metrics = logs_root / "metrics"
    metrics.mkdir()
    (metrics / "gpu-123.training-memory.json").write_text(
        json.dumps({"status": "PASS", "ranks": [{}, {}, {}]}), encoding="utf-8"
    )
    (metrics / "gpu-123.summary.txt").write_text("gpu_count=3\n", encoding="utf-8")


def test_complete_pulled_suite_passes(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    result = VERIFIER.verify_suite(_manifest(), logs_root=tmp_path, min_free_mib=1024.0)
    assert result["status"] == "PASS"
    assert result["runs"][0]["completion_marker_count"] == 3
    assert result["runs"][0]["memory_csv_count"] == 3


def test_missing_rank_and_nonfinite_memory_fail_closed(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    stdout_path = tmp_path / "slurm-123.out"
    stdout_path.write_text(
        stdout_path.read_text(encoding="utf-8").replace("Training time: 1\n", "", 1),
        encoding="utf-8",
    )
    memory_path = tmp_path / "rl_games/test_family/test_experiment/gpu_memory_rank_0.csv"
    memory_path.write_text(
        memory_path.read_text(encoding="utf-8").replace("19000", "nan"),
        encoding="utf-8",
    )
    result = VERIFIER.verify_suite(_manifest(), logs_root=tmp_path, min_free_mib=1024.0)
    assert result["status"] == "FAIL"
    errors = "\n".join(result["runs"][0]["errors"])
    assert "2 rank completion markers" in errors
    assert "non-finite device_used_mib" in errors


def test_nonfinite_or_incomplete_tensorboard_fails_closed(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    summaries = tmp_path / "rl_games/test_family/test_experiment/summaries"
    writer = EventFileWriter(str(summaries))
    writer.add_event(
        Event(
            wall_time=2.0,
            step=2,
            summary=Summary(
                value=[
                    Summary.Value(tag="rewards/iter", simple_value=float("nan")),
                ]
            ),
        )
    )
    writer.flush()
    writer.close()
    result = VERIFIER.verify_suite(_manifest(), logs_root=tmp_path, min_free_mib=1024.0)
    assert result["status"] == "FAIL"
    errors = "\n".join(result["runs"][0]["errors"])
    assert "non-finite" in errors
