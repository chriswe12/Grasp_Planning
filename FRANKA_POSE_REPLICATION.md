# Franka pose-randomized Clutter-v5 adaptation

> September 21 update: the original bank below keeps object placement fixed. See [independent placement randomization](FRANKA_PLACEMENT_RANDOMIZATION.md) for the new shared-tabletop XY/yaw bank with unchanged canonical goals. Existing checkpoints were trained with the original bank.

The final recipe uses **15 Hz policy control and 120 Hz GPU physics**, the Panda with the supplied camera mount, provisional ZED Mini intrinsics, and the pencil lab scene. It starts fresh; previous straight-approach and rejected v1 weights are not resumed.

## Historical selection

Selected Clutter-v5, Euler job 11419483, epoch 5000, Isaac RL commit `93365b9`. The matched comparison used 125 held-out targets, three distances and three repeats per profile (1,125 episodes per profile).

| Recipe | Nominal | Clutter | Depth stress | Mean |
| --- | ---: | ---: | ---: | ---: |
| Combined-v4 | 61.2% | 26.6% | 27.6% | 38.5% |
| **Clutter-v5** | **69.2%** | **50.2%** | 50.8% | **56.7%** |
| Depth-v5A | 68.2% | 38.0% | 44.6% | 50.3% |
| Depth-v5B | 68.1% | 42.0% | **52.4%** | 54.2% |

Evidence: `logs/euler/evaluations/euler_policy_benchmark/comparison.md` and `artifacts/franka_pose_replication_20260916/historical_comparison.json`. These are historical source results, not expected Franka success rates. Incompatible experiments were not ranked together.

## Final v3 reset contract

Catalog: `isaac_rl/data/franka_clutter_v5_pose_v3/catalog.npz`, SHA256 `bd7f1e6c0f05be762ea4045188f9d48b9d4e401b333a25e4be7e3d007e292623`.

- 1,615 targets across all 46 parts: 864 train, 445 validation, 306 test. There are 29 optimization parts and 17 whole-part holdouts.
- 133,041 accepted reset states. One target (`plumbers_block__part_2__orientation_004__g1034__p0`) was excluded because no safe positive start survived; no part was removed.
- Perturbed starts cover five approach distances (progress 0, .25, .5, .75, .94), eight balanced rotation axes, rotations tapering from 15 to 5 degrees, and lateral offsets sampled between half and full of the 10-to-3 mm tapered range.
- Separate pools contain ready starts, isolated position boundaries, isolated orientation negatives, physically checked exact goals, and validated nominal waypoints. Every runtime reset uses an accepted bank entry. There is no unchecked exact-goal or original-waypoint fallback.
- Each accepted state satisfies joint limits and contact below 0.5 N over 12 physics steps. Generated IK states also require agreement within 1 mm and 0.015 rad; original nominal waypoints are checked directly. Missing boundary classes fall back to validated negative path states, never ready states.

The superseded v1 adaptation had an all-positive boundary pool. Its chain was cancelled and its weights are excluded. V3 fixes the labels and extends physics checks to every reset mode.

## Recipe and deliberate adaptations

`configs/franka_clutter_v5_replica.json` records source settings and hashes. The runtime restores the source reset mixture (55% path, 15% nominal, 15% ready, 15% boundary at full difficulty), failure replay, binary completion rewards, dense pose rewards, stable completion, source network/PPO settings, generic RGBD/calibration noise, motion response/delay, and randomized PD gains. Existing scene colors, lighting, materials and props remain varied. Goal images stay canonical.

At 15 Hz, gamma/tau are squared, time costs and slew limits are adjusted to the policy period, and four 30 Hz completion frames become two 15 Hz frames (about 133 ms). The curriculum advances by global experience: at 256 environments, warm-up ends around epoch 500 and full difficulty around epoch 6,000. It increases approach range and the fraction of perturbed path starts; ready/boundary variation is present during warm-up. Bank angles are fixed per distance, without unvalidated joint interpolation.

The Panda TCP is perturbed relative to a fixed object/goal, whereas the source also moved the object/goal laterally. Pencil lab props replace the old t-slot clutter geometry. D405-specific disparity statistics and the source 2% image-repeat augmentation are disabled; camera delays are quantized to 15 Hz. Failure-replay scores rebuild each allocation, while policy, optimizer and curriculum resume. These are adaptations requiring new held-out evaluation.

Evaluation uses all held-out targets at far/mid/close perturbed starts. Close uses .94 progress (about 6 mm approach displacement); historical close used .84 (about 16 mm), so the resulting scores are not directly equivalent. Checkpoint selection averages complete far/mid/close macro-part scores; incomplete condition sets are excluded.

## Verification and videos

Final evidence is under `artifacts/franka_pose_replication_20260916/v3/`:

- `catalog_audit.json`: full catalog/source/split/reset coverage audit.
- `runtime_validation/reset_runtime_audit.json`: 512 actual resets at 15/120 Hz; all use validated states. The 272 perturbed path samples measured 5.15–15.07 degrees and 1.55–9.94 mm lateral offsets across all eight axes.
- `runtime_validation/label_audit.json`: 94 boundary samples included 18 positives and 76 not-ready cases, including 44 isolated orientation negatives and 20 isolated position negatives. All 78 ready samples were positive.
- `ppo_smoke_verified.json`: 64-environment, three-epoch PPO smoke; 12,288 transitions, 286 finite tensors, actual optimizer updates and all 29 training geometries.
- `videos/index.html`: two actual smoke-policy replays, each 254 frames at 15 fps. Both start around 100 mm / 15 degrees and time out. They demonstrate scene/reset behavior, not learned performance. Both videos were encoded and decoded with matching frame counts; the rendered scene was visually inspected.

Focused recipe and evaluation-selection tests passed. Source observation/curriculum/completion and Euler launch checks also passed. Final four-GPU startup proof is recorded below.

## Training and unattended downloads

Full job **14345512** resumes the verified epoch-53 checkpoint and runs to epoch 1,000, followed by successful-predecessor continuations through epoch 10,000. The short job14338504 and its original pending chain were cancelled during the user-requested immediate promotion. Four Quadro RTX 6000 GPUs × 64 environments × 64 rollout steps × 10,000 epochs = **163,840,000 transitions**, matching the source experience budget. Only one job trains at a time. The full learning-rate schedule spans all 10,000 epochs. Full segments request 24-hour allocations; this is not a runtime guarantee.

Manifest: `artifacts/franka_pose_replication_20260916/v3/full_training_chain.json`.

```bash
tmux attach -t franka-checkpoint-pull
# Detach without stopping downloads: Ctrl-b, then d
tail -F artifacts/franka_pose_replication_20260916/v3/full_training_chain.watch.log
```

The detached single-instance supervisor downloads each completed segment and retries failures. A separate `franka-pose-startup-check` tmux session checks at least 102 recorded epochs, downloads an immutable checkpoint, and verifies finite weights plus memory on all four ranks. Each cluster segment also checks checkpoint agreement; segments with sufficient samples require a 25-warmup/75-measured-epoch memory gate before dependent jobs can proceed. Keep this workstation powered and connected for local downloads.

## Remaining limits

This policy learns visual alignment and completion timing. The object is kinematic and fingers stay open; offline grasp lift validation does not establish online pickup success. Real ZED Mini calibration, physical pickup and hardware transfer remain unvalidated. Historical source success rates do not establish the new policy's quality; use the new held-out evaluations.

## Live startup verified — September 16, 2026, 13:32 CEST

Historical startup proof from job14338504 (subsequently stopped for full-training promotion): Its epoch32 checkpoint was downloaded with matching SHA256 before/after transfer and independently checked:524288transitions,286finite tensors,1020actor optimizer steps, finite losses, four distinct physical GPUs and all29training geometries. Actual `run.json` confirms fresh initialization, the final v3 catalog,256total environments and15Hz/120Hz. Final cross-rank weight agreement remains a segment-completion check.

Memory passed on epochs11–32 (22post-warm-up samples on each rank). Device and reserved-memory slopes were zero; live-tensor memory was flat within rounding. Minimum free VRAM was14380.625MiB. Host RSS increased4.0–9.8MiB per worker over this window, against approximately11.5GiB in use. This is bounded startup evidence, not a lifetime leak guarantee. The separate102-epoch monitor remains active, as does the segment-end memory gate. No application/PhysX/CUDA/NCCL error signatures appeared in the inspected job logs. The actual cluster RGB preview was downloaded and visually checked.

The proof files below were refreshed through epoch53 before handoff; the original epoch32 figures above record the initial observation. Proofs: `v3/short_checkpoint_verified.json`, `v3/short_memory_verified.json`, `v3/short_snapshot_integrity.json`, and `v3/cluster_preview/` under the artifact root. The usable checkpoint is `v3/short_start_snapshot/2026-09-16_12-45-04_job_14338504_4gpu/nn/franka_zed.pth`. The detached downloader and extended startup monitor were both verified alive; complete-segment automatic downloads remain pending until the first segment finishes.

## Immediate full-training promotion — September 16, 2026

At the user's request, replaced the128-epoch startup allocation with a24-hour allocation to epoch1000. Resume checkpoint `resume_promotions/epoch_53/epoch_53.pth` is an immutable verified copy:SHA256 `49d61548b49aabbde2273f7c0f8a07a02f3272443d8590afd37eb050055b9a12`, epoch53,868352transitions,1692actor optimizer steps. Updated memory proof covers43post-warm-up samples per rank. The old pending continuations were cancelled before the replacement chain was submitted, preventing duplicate chains.

New jobs:14345512→1000,14345531→2000,14345536→3000,14345538→4000,14345541→5000,14345548→6000,14345555→7000,14345574→8000,14345578→9000,14345579→10000. First job used an explicit afterany handoff from the stopped startup job and a verified checkpoint; later jobs retain strict afterok plus completed-predecessor verification. Total budget,15Hz/120Hz and256environments are unchanged. `full_training_chain.json` is authoritative; `training_chain.json` is marked superseded. Both tmux sessions now monitor the new chain.

Full resume verified at14:19CEST: all four ranks reached epoch59/1000. The downloaded epoch58 checkpoint has950272transitions,286finite tensors and160new actor optimizer updates beyond the epoch53 source. Exact checkpoint-transfer hashes,15Hz/120Hz,256envs, epoch1000limit and curriculum offset3392 were verified. Initial resumed memory samples (epochs56–59) passed with flat device memory and at least14400.625MiBfree; this is a short resume check, supplementing the43sample pre-handoff proof. Extended monitoring and automatic downloads remain active. Current evidence: `full_resume_verified.json`, `full_resume_integrity.json`, `full_resume_memory.json` in the v3 artifact directory. The final queue contains one running full job and nine dependent continuations; no old job remains active.
