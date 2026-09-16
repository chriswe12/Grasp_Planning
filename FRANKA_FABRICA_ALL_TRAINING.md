# All-part Panda / ZED Mini training

This workflow generates Fabrica grasps for the Panda hand, checks them in the randomized pencil lab, and trains the standalone visual alignment/completion policy. Offline dynamic lift validation filters the grasp targets; the learned task itself is still visual alignment and completion, not a new online lift policy. Real ZED intrinsics/extrinsics and hardware transfer remain provisional.

## Dataset construction

```bash
python3 scripts/run_grasp_generation_benchmark.py --config configs/franka_fabrica_all.yaml --jobs 4 --skip-stage1-collision-checks
bash scripts/franka_isaac_python.sh isaac_rl/scripts/build_franka_all_catalog.py --headless
bash scripts/franka_isaac_python.sh isaac_rl/scripts/validate_franka_all_lifts.py --headless
python3 scripts/verify_franka_all_catalog.py --catalog isaac_rl/data/franka_fabrica_all/catalog_lift_validated.npz --output artifacts/franka_all_20260915/catalog_audit.json
```

Use the repo Isaac container for the two Isaac commands. All 46 original meshes are attempted. Assembly-only obstacle screening is skipped because these are isolated-part training scenes; object/gripper/floor collisions remain checked. The optional `benchmark.write_full_grasp_html: false` avoids hundreds of megabytes of duplicated gripper meshes per orientation, while retaining the exact saved stage-1/stage-2 JSON and compact diagnostics. Other pipeline callers retain their existing HTML behavior.

For thin parts rejected solely by the 10 mm floor margin, retry the same generator with `configs/franka_fabrica_all_tabletop.yaml` and explicit `--target obj/fabrica/ASSEMBLY/ID.obj`. This uses 2 mm geometric clearance; the builder prefers completed recovery outputs and still requires actual driven collision checks and dynamic closure/lift/hold success. Do not lower physical validation thresholds to force coverage.

Each target references its original saved stage-2 grasp and source hash. The builder samples up to 96 target poses per part, diversified across stable orientations and source grasps, with two grounded table placements. All generated source grasps remain available; the training catalog is a physically validated subset. Its audit lists excluded parts explicitly.

Each environment owns exactly one actual part mesh and only samples that part's matching targets. Distributed assignment covers eligible parts across ranks. Whole gamepad parts are test-only, a deterministic fifth of other parts validation-only, and remaining source-grasp groups stay within a single split even across orientations/placements. Canonical goal images are fixed; live colors, materials, room lighting, exposure and white balance vary by episode.

## Complete catalog and physical validation

The September 15 build generated ground-feasible candidates for all 46 parts. The normal collision decomposition blocked narrow accessible regions on `duct/1` and `cooling_manifold/1`. Rebuilding only these two collision assets with `--collision-quality fine` resolved the false collisions; the original bundle-local mesh, source grasp and physical mass remain unchanged. `--asset-catalog` can reuse the verified fine collider for additional source grasps.

`scripts/merge_franka_validated_catalogs.py` only adds previously missing parts and checks exact bundle-local vertices/faces and mass. Its `--stage approach` mode clears physical labels so a merged catalog must pass fresh physical validation. The complete approach input is `isaac_rl/data/franka_fabrica_all_complete/approach_catalog.npz`; the final physical subset is `catalog.npz` in the same directory. Original catalogs and diagnostic outputs remain preserved. The final audit accepted 1,616 targets across all 46 parts (865 train, 445 validation, 306 test), rejecting 20 of the 1,636 approach candidates in the stronger full-mesh lift test. Catalog SHA256: `6c520e2791f8b022e90915a0070f21add15137509e302300a366d8d60d4f3696`.

Physical protocol `exact_goal_dynamic_close_lift_clearance_hold_v2` uses free dynamic objects and 20 N finger effort caps. It commands a lift of `max(0.08, largest_part_extent + 0.04)` metres, then requires more than 5 cm root lift, more than 1 cm clearance of the entire original mesh, bilateral finger contact, less than 3 N arm/hand contact, less than 3 mm initial settling drift and less than 5 mm final TCP tracking error. Holding is measured over the final 0.5 seconds. Mesh support extrema are evaluated using its convex-hull vertices, not the approximate collision hull. The earlier fixed 8 cm test could leave a large part's far edge on the table as it rotated. This validates lift and hold, not preservation of the object's in-hand orientation.

## Long Euler run

Stage `isaac_rl/data/franka_fabrica_all_complete/catalog.npz` with `EULER_FRANKA_CATALOG` through `euler/stage_franka.sh`. First run a four-GPU, three-epoch PPO gate on that exact production catalog and verify its checkpoints and validation report. Diagnostic two-part checkpoints must not seed production.

`euler/submit_franka_long.py` submits six sequential continuation jobs by default: 32 environments per GPU on four GPUs, 64 rollout steps, 1024 global minibatch, checkpoint every 100 epochs, and final epoch6144 (approximately 50.3 million transitions). Every job resumes its verified predecessor with optimizer state, then evaluates every held-out validation target on one GPU. Only the final segment additionally evaluates the test split. A fixed large budget does not guarantee convergence; use per-part and macro validation scores to judge learning.

An optional `--startup-epochs 256 --startup-time-limit 04:00:00` adds a short initial continuation while preserving the final epoch/transition budget. Completed initial gates are verified with Slurm accounting and resumed without a new `afterok` dependency, because the controller may have purged their completed records; pending/running predecessors still require `afterok`. Strict checkpoint metadata and catalog identity checks remain mandatory.

The launcher records each submitted job immediately and starts `euler/watch_franka_chain.py` detached. It watches and pulls each completed job, retaining intermediate/final checkpoints and failure logs locally; a failed predecessor prevents dependent training. The workstation must remain on and connected for downloads. The watcher records the best validation checkpoint without using test results for selection.

```bash
python3 euler/submit_franka_long.py --config /absolute/path/to/prepared/project/euler/euler.env \
  --catalog isaac_rl/data/franka_fabrica_all_complete/catalog.npz \
  --catalog-audit artifacts/franka_all_20260915/final_catalog_audit.json \
  --initial-job PRODUCTION_PPO_GATE_JOB_ID \
  --output artifacts/franka_all_20260915/long_training_chain.json
```

Check `long_training_chain.json`, its `.watch.log`, and per-rank GPU/RAM telemetry. A queued Slurm job is not evidence that training started; require real PPO epochs, finite checkpoint/loss/optimizer verification, distinct GPU identities and a bounded memory-stability observation.

## September 15 active chain

Deployment: `artifacts/franka_euler/20260915_140300/project/euler/euler.env`. Production gate `14277304` completed successfully on four Quadro RTX 6000 GPUs, including 24,576 transitions, 45 actor optimizer updates, matching actor/critic hashes across ranks, and complete 445-target validation. Its three-epoch policy had zero validation successes; this is an execution gate, not a trained-policy quality result.

The active chain is `artifacts/franka_all_20260915/long_training_chain_start256.json`. It starts with job `14283250` to epoch 256 in the four-hour queue, then jobs `14283251`, `14283252`, `14283254`, `14283256`, `14283258`, `14283261` to epochs 1024, 2048, 3072, 4096, 5120 and 6144 with 12-hour limits. The final budget remains 50,331,648 transitions. The original six unstarted jobs were cancelled before this replacement; their manifest is marked superseded.

Requests allow the compatible RTX-only GPUhe pool. The initial four-hour job has no GPU-generation preference; subsequent jobs have a soft preference for the observed RTX 4090 node generation (`gpuhe&EPYC_9554`). Actual allocation/backend GPU names and UUIDs are authoritative; the original submission GPU type can differ after an in-place scheduler update. Submission alone does not establish optimizer execution.

```bash
tail -f artifacts/franka_all_20260915/long_training_chain_start256.watch.log
ssh euler-gimenol 'squeue -u gimenol'
```

The watcher retains the last PPO epoch while a segment's frozen validation runs. Its next segment starts only after the job completes successfully. Startup proof is written to `long_start_monitor.json`, `long_checkpoint_verified.json` and `long_memory_verified.json` in the same artifact directory.

The actual 46-part wrist-view gallery is `artifacts/franka_all_20260915/inspection/index.html`.

A separate local capacity check resumed the exact production gate from epoch 3 to 6 with 64 environments on an RTX 4090. It verified 24 additional actor optimizer updates, finite checkpoint/loss tensors and all 29 eligible training parts; device memory stayed at 12,892 MiB used with 11,190 MiB free over the three measured epochs. This supports a smaller-GPU-count fallback if needed, but does not replace the long-run memory observation or seed the active chain. Evidence: `artifacts/franka_all_20260915/capacity64_verified.json`.

## Verified long-run startup, September 15 at 17:09 CEST

Job `14283250` started at 16:08:30 CEST on `eu-lo-g3-052`, using four distinct Quadro RTX 6000 GPUs and 128 total environments. All ranks reached epoch 105. The downloaded epoch-100 checkpoint contains 819,200 transitions, 286 finite tensors and 1,551 new actor optimizer updates since the production gate; catalog identity and all 29 eligible training-part geometry assignments passed verification. Final cross-rank weight agreement is checked at segment completion, not inferred from this in-progress checkpoint.

All four ranks passed the memory gate over epochs 29–105 (77 samples after 25 warmup epochs): device-used and reserved-memory slopes were exactly zero; live tensor-memory slopes were effectively zero. Minimum free VRAM was 16,516 MiB. Host RSS stayed bounded: rank 0 released about 821 MiB, while other ranks grew only 4.6–5.5 MiB across the observation. This is a bounded stability result, not a guarantee against every future leak. Both job logs had no application/PhysX/CUDA/NCCL error signatures in the final scan. The actual first training RGB image was inspected and showed colored parts, pencil marks, shadows and lighting variation without white-out.

Evidence:
- `artifacts/franka_all_20260915/long_start_monitor.json`
- `artifacts/franka_all_20260915/long_checkpoint_verified.json`
- `artifacts/franka_all_20260915/long_memory_verified.json`
- `artifacts/franka_all_20260915/long_start_snapshot/2026-09-15_16-08-45_job_14283250_4gpu/nn/last_franka_zed_ep_100_rew_1.9035815.pth`
- `artifacts/franka_all_20260915/long_start_preview/first_training_rgb.png`

The detached chain watcher remains active (PID 127185 at verification). Only one training job is running; six continuation jobs wait on successful predecessors. At the observed throughput, the full 50.3-million-transition budget is roughly 40–50 hours including typical startup/evaluation overhead, plus queue waits; allocation speed and policy-dependent episode length can change this estimate. The first allocation has a four-hour limit (20:08:30 CEST deadline); later allocations have 12-hour limits. The final epoch budget remains 6144, not 256.

Manual pull from the repository root:

```bash
EULER_CONFIG_PATH="$PWD/artifacts/franka_euler/20260915_140300/project/euler/euler.env" \
EULER_LOCAL_RESULTS_DIR="$PWD/artifacts/franka_all_20260915/euler_results" \
bash euler/pull_results.sh
```

Automatic pulls happen after each segment. Whole-part holdouts remain excluded from optimization: all 46 parts have validated grasps, while 29 train and 17 are held out for generalization. Real ZED Mini calibration and hardware transfer remain unvalidated; checkpoint health does not establish held-out task success.

## Unattended checkpoint pulls in tmux

On September 15 at 22:22 CEST, the detached watcher was handed over to tmux session `franka-checkpoint-pull`. Its supervisor is `artifacts/franka_all_20260915/checkpoint_pull_supervisor.sh`; it holds a single-instance lock and restarts the chain watcher after an unexpected nonzero exit. Network download failures are retried by the existing pull helper. The second tmux pane displays the existing `.watch.log`. Closing the viewer or this chat does not stop the supervisor; the local workstation must remain on and connected. The handoff touched only local watcher processes, not any training jobs.

The completed epoch-256 checkpoint is already downloaded. The replacement watcher replayed the completed gate/first segment successfully and advanced to running job 14283251. Updated supervisor/watcher PIDs and tmux name are recorded in the chain manifest.

```bash
tmux attach -t franka-checkpoint-pull
```
