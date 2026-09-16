"""Regression tests for the Euler container launch contract."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_euler_pythonpath_exposes_outer_project_package() -> None:
    config = _read("euler/euler.env.example")
    assert "EULER_CONTAINER_PYTHONPATH=/workspace/grasping_rl:" in config
    assert "EULER_LOGIN=your-euler-login" in config
    assert "EULER_PROJECT_DIR=/cluster/home/your-user/grasping_rl" in config


def test_submit_supports_explicit_gpu_benchmark_resources() -> None:
    source = _read("euler/submit.sh")
    for option in (
        "--gpu-type",
        "--gpu-count",
        "--gpu-memory",
        "--cpus-per-gpu",
        "--memory-per-cpu",
        "--time-limit",
        "--afterok",
        "--exclude",
    ):
        assert option in source
    assert '--gpus="${gpu_type}:${gpu_count}"' in source
    assert '--gres="gpumem:${gpu_memory}"' in source
    assert "EULER_SKIP_SYNC:-0" in source
    assert '[[ "${mode}" =~ ^(smoke|lift)$ && "${gpu_count}" != "1" ]]' in source
    assert "--gpu-count greater than one configures the Slurm ranks automatically" in source
    assert "smoke|lift|probe|train" in source
    assert '--dependency="afterok:${afterok_job_id}"' in source
    assert "--kill-on-invalid-dep=yes" in source
    assert '--exclude="${exclude_nodes}"' in source


def test_batch_job_launches_one_slurm_task_per_allocated_gpu() -> None:
    source = _read("euler/job.sbatch")
    assert "srun" in source
    assert '--ntasks="${requested_gpu_count}"' in source
    assert "--gpus-per-task=1" in source
    assert "--gpu-bind=single:1" in source
    assert "euler/slurm_rank_launcher.sh" in source
    assert "--distributed" in source
    assert "completion_count < requested_gpu_count" in source
    assert "APPTAINERENV_ISAAC_RL_EXPERIMENT_NAME=" in source


def test_batch_job_samples_and_summarizes_every_allocated_gpu() -> None:
    source = _read("euler/job.sbatch")
    assert '--id="${visible_gpu_selector}"' in source
    assert "gpu_count=%s" in source
    assert "gpu_%s_peak_memory_mib=%s" in source
    assert "APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in source
    assert "analyze_training_memory.py" in source
    assert "gpu_memory_rank_*.csv" in source


def test_distributed_jobs_do_not_reuse_writable_kit_caches() -> None:
    source = _read("euler/job.sbatch")
    launcher_source = _read("euler/slurm_rank_launcher.sh")
    assert "if (( requested_gpu_count > 1 )); then" in source
    assert '"${EULER_CACHE_DIR}/cache/torch/" "${local_cache}/cache/torch/"' in source
    assert "if (( requested_gpu_count == 1 )); then" in source
    assert "Keeping distributed writable Kit caches job-local" in source
    assert 'flock -s "${cache_lock}" rsync' in source
    assert 'euler/slurm_rank_launcher.sh\n        "${local_cache}"' in source
    assert 'rank_cache="${cache_root}/rank_${SLURM_PROCID}"' in launcher_source
    assert '"${rank_cache}/cache/kit"' in launcher_source
    assert '"${rank_cache}/cache/ov"' in launcher_source
    assert '"${rank_cache}/cache/computecache"' in launcher_source
    assert '"${rank_cache}/data"' in launcher_source
    assert 'runtime_arg="${rank_cache}${runtime_arg#"${cache_root}"}"' in launcher_source
    assert 'exec "${runtime_args[@]}"' in launcher_source


def test_rl_games_entrypoint_records_distributed_batch_contract() -> None:
    source = _read("isaac_rl/scripts/rl_games/train.py")
    assert 'world_size = int(os.getenv("WORLD_SIZE", "1"))' in source
    assert 'os.getenv("ISAAC_RL_EXPERIMENT_NAME")' in source
    assert "if global_rank == 0:" in source
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
    source = _read("isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/agents/completion_ppo.py")
    assert "class _ReusableGradientAllReduce" in source
    assert "torch.cat(all_grads_list)" not in source
    assert "_central_value_calc_gradients_with_reusable_buffer" in source
    assert "gpu_memory_rank_{self.global_rank}.csv" in source


def test_lift_dls_slices_environment_and_joint_dimensions_separately() -> None:
    source = _read("isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env.py")
    compact_source = "".join(source.split())
    assert "[env_ids,self.context.ee_jacobi_body_idx][:,:,self.arm_ids]" in compact_source
    assert "[env_ids,self.context.ee_jacobi_body_idx,:,self.arm_ids]" not in compact_source


def test_lift_mode_sizes_physx_patch_buffer_for_production_environment_count() -> None:
    source = _read("isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env.py")
    init_start = source.index("def __init__")
    init_end = source.index("def _setup_scene", init_start)
    init_source = source[init_start:init_end]
    assert "cfg.sim.physx.gpu_max_rigid_patch_count" in init_source
    assert "2**19" in init_source
    assert "cfg.sim.physx.gpu_found_lost_pairs_capacity" not in init_source


def test_lift_gravity_updates_use_full_physx_view_tensor() -> None:
    source = _read("isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env.py")
    assert "(int(part.root_physx_view.count), 1)" in source
    assert "torch.full((cpu_ids.numel(), 1)" not in source


def test_lift_physics_only_enables_the_selected_part_clone() -> None:
    source = _read("isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env.py")
    assert "def _enable_only_selected_part_simulation" in source
    assert "part.root_physx_view.set_disable_simulations(disabled, cpu_env_ids)" in source
    assert "selected_parts = self.target_part_indices[self.target_index[env_ids]]" in source
    assert "self._enable_only_selected_part_simulation(env_ids)" in source


def test_lift_part_types_do_not_expand_the_startup_broadphase() -> None:
    source = _read("isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env.py")
    setup_start = source.index("def _setup_scene")
    setup_end = source.index("def _active_part_pose", setup_start)
    setup_source = source[setup_start:setup_end]
    assert "part_cfg.init_state.pos" not in setup_source
    assert "self.parts.append(RigidObject(part_cfg))" in setup_source


def test_lift_material_uses_scene_default_without_per_clone_usd_binding() -> None:
    source = _read("isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env.py")
    assert "cfg.sim.physics_material = sim_utils.RigidBodyMaterialCfg(" in source
    assert "bind_physics_material" not in source
    assert "policy_lift_high_friction" not in source


def test_lift_gravity_updates_skip_inactive_part_clones() -> None:
    source = _read("isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env.py")
    gravity_method = source.split("def _set_part_gravity_disabled", maxsplit=1)[1].split(
        "def _enable_only_selected_part_simulation", maxsplit=1
    )[0]
    assert "for part_index, part in enumerate(self.parts):" in gravity_method
    assert "local_mask = selected_parts == part_index" in gravity_method
    assert "selected_ids = env_ids[local_mask]" in gravity_method


def test_lift_approach_fixture_is_not_rewritten_every_physics_substep() -> None:
    source = _read("isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env.py")
    pre_step = source.split("def _pre_physics_step", maxsplit=1)[1].split("def _begin_lift_attempts", maxsplit=1)[0]
    apply_action = source.split("def _apply_action", maxsplit=1)[1].split("def _advance_lift_phases", maxsplit=1)[0]
    assert "self._restore_selected_part_fixture(approach_ids)" in pre_step
    assert "self.lift_phase == self._LIFT_PHASE_CLOSE" in apply_action
    assert "fixture_mask = normal_mask" not in apply_action


def test_smoke_does_not_wrap_dynamic_lift_steps_in_inference_mode() -> None:
    source = _read("isaac_rl/scripts/smoke_env.py")
    assert "torch.inference_mode" not in source


def test_slurm_launcher_isolates_each_gpu_before_apptainer_starts() -> None:
    source = _read("euler/slurm_rank_launcher.sh")
    batch_source = _read("euler/job.sbatch")
    train_source = _read("isaac_rl/scripts/rl_games/train.py")
    assert 'APPTAINERENV_CUDA_VISIBLE_DEVICES="${task_gpu}"' in source
    assert 'APPTAINERENV_RANK="${SLURM_PROCID}"' in source
    assert 'APPTAINERENV_WORLD_SIZE="${SLURM_NTASKS}"' in source
    assert "APPTAINERENV_LOCAL_RANK=0" in source
    assert 'APPTAINERENV_ISAAC_RL_ORIGINAL_LOCAL_RANK="${SLURM_LOCALID}"' in source
    assert 'exec "${runtime_args[@]}"' in source
    assert "APPTAINERENV_ISAAC_RL_STARTUP_LOCK=" in batch_source
    assert "APPTAINERENV_ISAAC_RL_DISTRIBUTED_READY_DIR=" in batch_source
    assert ".isaac-startup-${node_name}.lock" in batch_source
    assert "startup_lock_handle = _acquire_distributed_startup_lock()" in train_source
    assert train_source.index("startup_lock_handle = _acquire_distributed_startup_lock()") < train_source.index(
        "app_launcher = AppLauncher(args_cli)"
    )
    assert train_source.index("env = gym.make(") < train_source.index(
        "_release_distributed_startup_lock(startup_lock_handle)",
        train_source.index("env = gym.make("),
    )
    assert "def _wait_for_distributed_environment_barrier" in train_source
    assert train_source.index(
        "_wait_for_distributed_environment_barrier(global_rank, world_size)"
    ) < train_source.index(
        'runner.run({"train": True',
    )


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
    assert "'^\\[LIFT\\] report='" in source
    assert "Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0" in source


def test_bulk_lift_jobs_can_disable_per_target_videos() -> None:
    source = _read("euler/job.sbatch")
    assert '[[ "${lift_arg}" == "--no-lift-videos" ]]' in source
    assert "lift_record_videos=0" in source
    assert "if (( lift_record_videos )); then" in source


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


def test_fabrica_weekend_suite_is_large_reproducible_and_pullable() -> None:
    source = _read("euler/submit_fabrica_weekend_suite.sh")
    assert 'NUM_ENVS="${WEEKEND_NUM_ENVS:-224}"' in source
    assert 'MAX_ITERATIONS="${WEEKEND_MAX_ITERATIONS:-10000}"' in source
    assert 'GPU_COUNT="${WEEKEND_GPU_COUNT:-4}"' in source
    assert 'GLOBAL_MINIBATCH_SIZE="${WEEKEND_GLOBAL_MINIBATCH_SIZE:-$((GPU_COUNT * 256))}"' in source
    assert 'TASK="Grasp-Visual-Servo-RGBD-FabricaAll-Direct-v0"' in source
    assert "submit_one baseline combined_sim2real baseline 0" in source
    assert "submit_one improved combined_sim2real long_run_improved 1" in source
    assert "submit_one clutter-improved combined_clutter long_run_improved 1" in source
    assert "watch_ablation_suite.sh" in source
    assert "write_manifest" in source


def test_submit_selects_supported_fabrica_rank_counts_and_six_rank_batch() -> None:
    source = _read("euler/submit.sh")
    assert '[[ "${gpu_count}" != "4" && "${gpu_count}" != "6" ]]' in source
    assert "--global_minibatch_size 1536" in source
