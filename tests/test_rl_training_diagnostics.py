"""Tests for distributed PPO batching and completion-head diagnostics."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from grasp_planning.rl.completion_diagnostics import CompletionDiagnostics
from grasp_planning.rl.ppo_batching import resolve_local_minibatch_size

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_distributed_observer_drops_nonzero_rank_episode_tensors() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("rl_games.common.algo_observer")
    from grasp_planning.rl.distributed_observer import DistributedSafeIsaacAlgoObserver

    class FakeAlgo:
        games_to_track = 10
        ppo_device = "cpu"
        writer = None
        global_rank = 2

    observer = DistributedSafeIsaacAlgoObserver()
    observer.after_init(FakeAlgo())
    observer.process_infos(
        {"episode": {"success": torch.tensor(1.0)}},
        torch.tensor([0]),
    )
    assert observer.ep_infos == []
    assert observer.direct_info == {}


def test_four_gpu_batch_preserves_single_gpu_effective_minibatch() -> None:
    local_minibatch = resolve_local_minibatch_size(
        rollout_batch_size_per_rank=224 * 64,
        target_global_minibatch_size=1024,
        world_size=4,
    )
    assert local_minibatch == 256
    assert local_minibatch * 4 == 1024
    assert (224 * 64 // local_minibatch) * 2 == 112


def test_batch_resolver_uses_valid_divisor_for_non_power_of_two_world_size() -> None:
    assert resolve_local_minibatch_size(
        rollout_batch_size_per_rank=256 * 64,
        target_global_minibatch_size=1024,
        world_size=6,
    ) == 128


def test_completion_diagnostics_exclude_ambiguous_labels() -> None:
    diagnostics = CompletionDiagnostics(threshold=0.95)
    diagnostics.update(
        probabilities=[0.99, 0.80, 0.10, 0.20],
        ready=[True, True, False, False],
        supervised=[True, True, True, False],
    )
    summary = diagnostics.summary()
    assert summary["supervised_samples"] == 3
    assert summary["ambiguous_samples"] == 1
    assert summary["precision"] == 1.0
    assert summary["recall"] == 0.5
    assert summary["false_positive_rate"] == 0.0
    assert summary["brier_score"] == pytest.approx((0.01**2 + 0.20**2 + 0.10**2) / 3)


def _write_memory_log(path: Path, *, growth_mib_per_epoch: float) -> None:
    fieldnames = [
        "epoch",
        "timestamp_unix",
        "allocated_mib",
        "reserved_mib",
        "epoch_peak_allocated_mib",
        "epoch_peak_reserved_mib",
        "device_used_mib",
        "device_free_mib",
        "device_total_mib",
        "actor_gradient_buffer_mib",
        "central_gradient_buffer_mib",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, 21):
            used = 20_000.0 + growth_mib_per_epoch * epoch
            writer.writerow(
                {
                    "epoch": epoch,
                    "timestamp_unix": epoch,
                    "allocated_mib": used - 1000.0,
                    "reserved_mib": used - 500.0,
                    "epoch_peak_allocated_mib": used - 900.0,
                    "epoch_peak_reserved_mib": used - 400.0,
                    "device_used_mib": used,
                    "device_free_mib": 24_564.0 - used,
                    "device_total_mib": 24_564.0,
                    "actor_gradient_buffer_mib": 24.0,
                    "central_gradient_buffer_mib": 1.0,
                }
            )


def test_memory_analyzer_rejects_a_long_run_with_linear_growth(tmp_path: Path) -> None:
    _write_memory_log(tmp_path / "gpu_memory_rank_0.csv", growth_mib_per_epoch=2.0)
    output_path = tmp_path / "analysis.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "euler/analyze_training_memory.py"),
            str(tmp_path),
            "--warmup-epochs",
            "2",
            "--min-samples",
            "10",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "FAIL"


def test_memory_analyzer_accepts_flat_memory_with_headroom(tmp_path: Path) -> None:
    _write_memory_log(tmp_path / "gpu_memory_rank_0.csv", growth_mib_per_epoch=0.0)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "euler/analyze_training_memory.py"),
            str(tmp_path),
            "--warmup-epochs",
            "2",
            "--min-samples",
            "10",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "PASS"
