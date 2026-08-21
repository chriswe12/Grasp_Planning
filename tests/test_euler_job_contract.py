"""Regression tests for the Euler container launch contract."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_euler_pythonpath_exposes_outer_project_package() -> None:
    config = _read("euler/euler.env")
    assert "EULER_CONTAINER_PYTHONPATH=/workspace/grasping_rl:" in config
    assert "EULER_LOGIN=euler-gimenol" in config
    assert "EULER_PROJECT_DIR=/cluster/home/gimenol/grasping_rl" in config


def test_submit_supports_explicit_gpu_benchmark_resources() -> None:
    source = _read("euler/submit.sh")
    for option in (
        "--gpu-type",
        "--gpu-count",
        "--gpu-memory",
        "--cpus-per-gpu",
        "--memory-per-cpu",
        "--time-limit",
    ):
        assert option in source
    assert '--gpus="${gpu_type}:${gpu_count}"' in source
    assert '--gres="gpumem:${gpu_memory}"' in source
    assert 'EULER_SKIP_SYNC:-0' in source
    assert '[[ "${mode}" == "smoke" && "${gpu_count}" != "1" ]]' in source
    assert "--gpu-count greater than one configures the Slurm ranks automatically" in source


def test_batch_job_launches_one_slurm_task_per_allocated_gpu() -> None:
    source = _read("euler/job.sbatch")
    assert "srun" in source
    assert '--ntasks="${requested_gpu_count}"' in source
    assert "--gpus-per-task=1" in source
    assert "--gpu-bind=single:1" in source
    assert "euler/slurm_rank_launcher.sh" in source
    assert "--distributed" in source
    assert "completion_count < requested_gpu_count" in source
    assert 'APPTAINERENV_ISAAC_RL_EXPERIMENT_NAME=' in source


def test_batch_job_samples_and_summarizes_every_allocated_gpu() -> None:
    source = _read("euler/job.sbatch")
    assert '--id="${visible_gpu_selector}"' in source
    assert "gpu_count=%s" in source
    assert "gpu_%s_peak_memory_mib=%s" in source
    assert "APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in source
    assert "analyze_training_memory.py" in source
    assert "gpu_memory_rank_*.csv" in source


def test_rl_games_entrypoint_records_distributed_batch_contract() -> None:
    source = _read("isaac_rl/scripts/rl_games/train.py")
    assert 'world_size = int(os.getenv("WORLD_SIZE", "1"))' in source
    assert 'os.getenv("ISAAC_RL_EXPERIMENT_NAME")' in source
    assert 'if global_rank == 0:' in source
    assert '"environments_per_rank": None' in source
    assert "global_rollout_batch_size = rollout_batch_size * world_size" in source
    assert "global rollout batch={global_rollout_batch_size}" in source
    assert "resolve_local_minibatch_size" in source
    assert "effective global minibatch={effective_global_minibatch_size}" in source
    assert "optimizer updates/epoch={optimizer_updates_per_epoch}" in source


def test_rl_games_entrypoint_does_not_retain_nonzero_rank_episode_tensors() -> None:
    source = _read("isaac_rl/scripts/rl_games/train.py")
    observer_source = _read("grasp_planning/rl/distributed_observer.py")
    assert "class DistributedSafeIsaacAlgoObserver(IsaacAlgoObserver)" in observer_source
    assert 'getattr(algo, "global_rank", 0)' in observer_source
    assert "if not self._collect_training_statistics:" in observer_source
    assert "self.ep_infos.clear()" in observer_source
    assert "Runner(DistributedSafeIsaacAlgoObserver())" in source


def test_completion_ppo_reuses_distributed_gradient_buffers() -> None:
    source = _read(
        "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/agents/completion_ppo.py"
    )
    assert "class _ReusableGradientAllReduce" in source
    assert "torch.cat(all_grads_list)" not in source
    assert "_central_value_calc_gradients_with_reusable_buffer" in source
    assert "gpu_memory_rank_{self.global_rank}.csv" in source


def test_slurm_launcher_isolates_each_gpu_before_apptainer_starts() -> None:
    source = _read("euler/slurm_rank_launcher.sh")
    assert 'APPTAINERENV_CUDA_VISIBLE_DEVICES="${task_gpu}"' in source
    assert 'APPTAINERENV_RANK="${SLURM_PROCID}"' in source
    assert 'APPTAINERENV_WORLD_SIZE="${SLURM_NTASKS}"' in source
    assert "APPTAINERENV_LOCAL_RANK=0" in source
    assert 'APPTAINERENV_ISAAC_RL_ORIGINAL_LOCAL_RANK="${SLURM_LOCALID}"' in source
    assert 'exec "$@"' in source


def test_python_entrypoints_bootstrap_repository_root_before_isaac() -> None:
    expected_parent_depths = {
        "isaac_rl/scripts/rl_games/train.py": 3,
        "isaac_rl/scripts/rl_games/evaluate_multigrasp.py": 3,
        "isaac_rl/scripts/smoke_env.py": 2,
    }
    for relative_path, parent_depth in expected_parent_depths.items():
        source = _read(relative_path)
        bootstrap = f"REPO_ROOT = Path(__file__).resolve().parents[{parent_depth}]"
        assert bootstrap in source
        assert source.index(bootstrap) < source.index("from isaaclab.app import AppLauncher")


def test_batch_job_requires_preflight_and_application_completion_markers() -> None:
    source = _read("euler/job.sbatch")
    assert "[PREFLIGHT] project imports OK" in source
    assert "import grasp_planning" in source
    assert "'^Training time: '" in source
    assert "'^\\[SMOKE\\] steps='" in source
    assert "Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0" in source


def test_image_validation_uses_current_multipart_catalog() -> None:
    source = _read("euler/push_image.sh")
    assert "--task Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0" in source


def test_watchers_reject_masked_application_failures() -> None:
    validation_watcher = _read("euler/watch_validate_and_pull.sh")
    generic_watcher = _read("euler/watch_and_pull.sh")
    assert '[[ -z "${local_run_dir}" ]]' in validation_watcher
    assert "created no training run" in validation_watcher
    assert "training completion marker is absent" in validation_watcher
    assert "completion marker is absent" in generic_watcher
