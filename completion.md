# Visual-Servo Training Revision Completion

## Real D405 policy deployment path (2026-08-24)

Real execution can now use the stage-2 grasp as its single source of truth,
plan and execute the pregrasp through MoveIt, run the D405 PPO policy only for
the final approach through collision-checking MoveIt Servo, and return control
after the learned four-frame low-speed completion gate. Gripper closing remains
a separate explicitly approved step; dry-run, motion approval, and close
approval are independent configuration gates.

The deployed observation path uses synchronized rectified RGB8/aligned Z16,
converts ROS millimetres to metres once, reuses the shared validity-aware
preprocessor, appends zero privileged placeholders, and uses deterministic
`mus`. Live optical twists are rotated into `lbr_link_0` through timestamped TF.
Every data, TF, controller, deadman, workspace, joint, force, or timeout fault
latches zero motion. See `configs/visual_servo_real_d405.yaml` and
`REAL_D405_PPO_DEPLOYMENT_IMPLEMENTATION.md` for the remaining hardware gates.

Date: 2026-08-14 through 2026-08-21
Source branch: `grasping-rl`
Source commit at start: `5159a1959c0cba818e4210d9e517f25c323df47f`
Change status at start: local-only in an already dirty worktree

## 2026-08-21 Final Verification And Publication

The nested `isaac_rl` repository was committed as `93365b9` on its `master`
branch. The parent `grasping-rl` branch contains the camera, observation,
sim-to-real profile, visual workspace, Euler deployment, diagnostics, assets,
tests, plans, and architecture documentation that support that environment.

Final host verification completed with 360 passing tests and two expected
RL-Games dependency skips. The nested Isaac test suite completed in the Isaac
Python environment with 25 passing tests and one optional skip. Ruff, Python
compilation, shell syntax, and Git whitespace checks passed. During this final
pass, the standalone T-slot preview helper's late-import boundary was corrected
so STEP-to-USD conversion receives `schemas_cfg` explicitly instead of relying
on a function-local import.

Four-GPU baseline job `11417200` supplied the first live check of the
distributed-observer memory fix. Across epochs 29--34, all four ranks remained
flat at approximately 9,972--9,996 MiB PyTorch allocated, with 2.58--2.78 GiB
device memory free. In particular, ranks one through three no longer showed
the previous approximately 2.2 MiB/epoch asymmetric growth. This is strong
early evidence rather than the formal long-run verdict: the analyzer requires
150 post-warm-up samples before reporting `PASS`.

## 2026-08-20 Euler launch reliability fix

Euler job `11219636` did not create a training run. Its pulled stderr showed
`ModuleNotFoundError: No module named 'grasp_planning'`, while Isaac Sim's
`python.sh` masked the child failure and caused Slurm to record `COMPLETED 0:0`.

The deployment now exports `/workspace/grasping_rl` in the container
`PYTHONPATH`, bootstraps the repository root in the train, evaluation, and smoke
entrypoints, and validates `grasp_planning` before simulator startup. Batch jobs
also require explicit import and application completion markers, so a masked
Python failure returns a failed Slurm status. Both watcher variants reject a
Slurm-only success when the expected run or completion marker is missing and
point to the pulled stdout/stderr logs.

## 2026-08-21 T-slot scale correction

The canonical T-slot workspace remains active for goal capture, training,
playback, and evaluation. The previous prototype's 10 mm slots, 41 mm lands,
and 51 mm pitch appeared too large in the close wrist view. The authored land
centers now use 25.5 mm pitch while retaining 20.5 mm-wide aluminum lands,
leaving approximately 5 mm dark slots. Collision remains the flat z=0 plane.

The visual material, scene, workspace, and sim-to-real profile identifiers were
bumped. The multipart catalog/contact sheet were regenerated with the smaller
canonical T-slot.

## 2026-08-21 Distributed episode-statistics memory leak fix

Four-GPU job `11408451` showed that the earlier reusable gradient buffers were
not the complete memory fix. Rank zero remained flat at approximately
9,972 MiB allocated, while ranks one through three retained approximately
2.2 MiB per PPO epoch. The asymmetry traced the remaining leak to RL-Games'
stock `IsaacAlgoObserver`: every rank appended CUDA-backed episode dictionaries,
but RL-Games calls the statistics writer and its `ep_infos.clear()` only on
global rank zero.

Training now uses a distributed-safe observer. Nonzero ranks do not collect
episode logging tensors because they have no writer, while rank zero preserves
the existing TensorBoard metrics and clears all observer state normally. A
behavior test verifies that an episode tensor presented to rank two is not
retained. Job `11408451` was cancelled and its partial checkpoints and logs
were pulled before the fix.

The corrected small-pitch catalog and actual `combined_sim2real` task were also
rendered through the complete 72 x 128 x 8 policy preprocessing path. The
verification stage contained zero clutter/distractor prims; clutter remains a
preview-only experiment and is not part of baseline training.

## 2026-08-20 T-slot workspace and live appearance integration

> Updated on 2026-08-21: the T-slot remains canonical, but its rendered pattern
> is now half-scale. See the correction above.

The accepted T-slot/palette preview has been moved out of `.cache` and wired
into goal capture and the actual RL environment. The canonical reference now
uses muted brown PLA parts, matte yellow fingers, and a neutral aluminum plate
with 10 mm slots, 41 mm lands, and 51 mm pitch. The tracked plate USD has no
physics or collision schema. `/World/GroundPlane` is invisible to the renderer
but remains the sole flat z=0 workspace collision surface, so grooves affect RGB
and depth without creating reset or policy collisions.

Training-only part color is selected independently per environment from a
weighted 24-color muted FDM palette. Each selected part preset also receives
bounded continuous HSV/value and roughness jitter. The live aluminum response
is mostly neutral with rarer cool/dim and warm/bright presets plus independent
continuous color/roughness jitter. T-slot geometry is sampled once per cloned
environment as 60% nominal, 20% phase-shifted, and 20% continuously rotated
through the unique -90 to +90 degree range. The stress profile broadens that to
40/25/35. This models changes in table-relative appearance as the movable part
is presented at different yaw angles while retaining one flat collision plane.
The target catalog is never randomized and always uses the canonical part,
background, and layout.
Existing bounded key-light rotation continues to move shadow direction over
training, while the calibrated camera uncertainty remains at +/-1% scale,
+/-1.5 pixels, and +/-1 degree rather than the preview's excessive 3--5% idea.

The material, scene, and workspace profile identifiers were bumped and the
full 1,256-image goal catalog was regenerated. All targets passed: zero pose,
rotation, or image-quality failures, with worst capture mismatch 0.588 mm and
0.0969 degrees. The new contact sheet is under
`artifacts/plumbers_block_catalog_debug/goal_rgb_contact_sheet.png`.

Verification: 346 repository tests passed with one optional RL-Games import
skip outside Isaac; Ruff passed. A four-target canonical capture and the full
1,256-target capture both passed. An eight-environment combined-profile smoke
selected distinct part colors plus nominal, phase-shifted, and rotated T-slot
layouts, completed one policy step, and reported zero collisions.

## Outcome

The multipart visual-servo training system now trains motion, pose-error
features, and autonomous completion as one shared visual policy. It has a
collision-safe reset mixture with explicit already-successful and boundary
examples, variable episode budgets, an easy-to-full curriculum, lower and
smoother commanded velocity, graded contact-risk shaping, live-only visual and
physical scene randomization, online part-balanced hard-target replay, and a
held-out periodic validation/checkpoint-selection workflow.

This is an architecture and observation-contract change. Training must start
fresh; checkpoints from before this revision are not load-compatible.

## 2026-08-19 D405 Sim-to-Real Extension

The active camera and observation contract is now v7/v3. Rendering and
camera-frame control share the same corrected hand-eye rotation, and live plus
goal RGB-D pass through one validity-aware area-resize/normalization path. The
full multipart catalogue was rerendered after freezing this contract: all
1,256 targets passed, with 1,012 train, 125 validation, and 119 test targets;
the worst capture mismatch was 0.588 mm and 0.0969 degrees.

The live-only D405 model no longer treats metric depth pixels as independent.
It perturbs disparity with episode-stable low-frequency structure, an AR-like
temporal component, and a smaller independent residual, then applies
quantization, range invalidation, sparse/edge dropout, and horizontal stereo
boundary failures. A coupled RGB-D affine warp approximates bounded intrinsic
and small hand-eye uncertainty until device-specific calibration is measured.

At the 15 Hz policy boundary, the environment now samples live-frame latency
and rare repeats, motion-command delay, actuator response scale/bias/filtering,
and arm stiffness/damping. Completion hold remains immediate and is never put
through the motion delay. Fifteen percent of combined-profile environments are
clean. Object mass and contact friction are intentionally unchanged because
the part is kinematic and this policy terminates before grasp contact.

Nine named profiles make ablation and stress results reproducible:
`nominal`, `sensor_only`, `camera_uncertainty`, `timing_control`, `appearance`,
`combined_sim2real`, `combined_clutter`, `combined_depth_robust`, and
`stress_test`. Training defaults to the combined
profile; evaluation defaults to nominal. Exact overrides, camera profile, and
observation profile are written to run YAML and TensorBoard text and embedded
in evaluation JSON/Markdown/CSV outputs.

The provisional profile is based on documented D405/D400 properties: 18 mm
stereo baseline, 7--50 cm operating range, 1/32-pixel subpixel disparity, and
0.1 mm close-range depth units. It is versioned
`d405_documented_provisional_v6_15hz`; real plane captures should later replace the
range values without changing the observation interface.

New diagnostics include
`scripts/render_d405_randomization_grid.py` and
`artifacts/d405_randomization/provisional_sensor_grid.png`. The exact remaining
recording-dependent work is fitting the real sensor distribution, checking
persistent bracket/cable silhouettes, offline real-frame policy replay, and
safe closed-loop robot validation.

Verification for this extension: 329 repository tests passed with one optional
RL-Games import skip outside the Isaac environment; Ruff passed; a four-
environment multipart Isaac smoke completed two steps with zero collisions.
The additional forced full-strength smoke mode was added, but its second GPU
launch was blocked by the execution service's approval/usage limit.

## 2026-08-21 Controlled Clutter And Depth-Robust Training Variants

Two controlled variants now extend the unchanged `combined_sim2real` baseline.
`combined_clutter` places one to three deterministic peripheral colored props
in 60% of cloned environments while retaining 40% clean environments. Props
affect rendered RGB and depth but have no collision or rigid-body schemas; they
remain outside the nominal target/approach corridor and the workspace collision
surface stays the flat `/World/GroundPlane`. This is visual distractor training,
not a physical-clutter avoidance controller.

`combined_depth_robust` keeps clutter disabled and changes only depth-error
ranges. It moderately expands metric scale/bias/residual noise, structured
fixed and temporal disparity error, stereo-boundary mismatch, sparse/edge
dropout, and depth-patch loss. It preserves 15% clean episodes plus the current
RGB, camera-warp, timing, control, appearance, reset, reward, and PPO settings.
The stronger values bracket uncertainty before the two real D405 units are
measured and are not claimed as camera specifications.

The active profile namespace is now
`d405_documented_provisional_v6_15hz`. Exact one-environment scene/policy renders
are saved under `artifacts/training_visual_check_clutter/` and
`artifacts/training_visual_check_depth_robust/`. Four-environment, two-step
Isaac smokes completed for both variants; clutter instantiated in two of four
environments with five total props, and both runs reported zero collision rate.

## Implementation Plan Used

1. Preserve the collision invariant first: only exact nominal waypoints and
   fully authored rotation variants may initialize multipart episodes.
2. Put continuous error variation around those safe states by translating the
   active object and final TCP goal together, with the displacement capped by
   stored per-state clearance.
3. Introduce explicit path, no-noise, ready, and completion-boundary reset
   modes plus reset-dependent time budgets.
4. Add a step-based curriculum that expands distance, perturbation, appearance,
   and hard-target replay without changing the rollout/minibatch geometry.
5. Share learned geometric features between motion, completion, and pose-error
   heads while ensuring privileged labels cannot enter the action path.
6. Slow and smooth commands near predicted completion, and align the training
   completion gate with deterministic deployment.
7. Add contact-risk shaping and online failure-driven sampling without breaking
   per-part target balance.
8. Add live-only sensor/occlusion and physical light, shadow, part, and ground
   randomization; keep catalog goals canonical.
9. Evaluate saved checkpoints on the validation split, select by safety-aware
   held-out score, and leave the test split untouched until final reporting.
10. Verify pure logic, policy isolation/gradients, simulator resets, one real PPO
    update, checkpoint creation, and TensorBoard metrics.

## Reset Distribution And Safety

The multipart training mixture is:

| Reset mode | Fraction | State and purpose | Timeout |
|---|---:|---|---:|
| Path | 55% | Authored path waypoint; continuous collision-capped stable-part XY/yaw error; authored gripper rotation enabled by curriculum | 4--12 s by progress |
| No noise | 15% | Exact nominal authored waypoint with zero XY, rotation, and joint noise | 4--12 s by progress |
| Ready | 15% | Final nominal orientation with continuous 0--3.5 mm XY error | 1.5 s |
| Boundary | 15% | One of the last three non-final nominal waypoints, or a fully validated final five-degree rotation | 2.5 s |

Twenty-five percent of ready resets are the exact successful pose, or 3.75% of
all resets at the configured mixture. This provides dense positive examples for
the completion classifier without making them the majority.

Multipart arm states still use the 32 exact collision-validated trajectory
waypoints. Path progress and XY error are sampled continuously, but arbitrary
joint-space interpolation is intentionally not used: the asset proves the
authored endpoints safe, not every interpolated joint configuration. Rotation
is likewise either nominal or a fully authored variant.

For part-pose estimation error, the robot remains on an authored nominal state
while the stable active object and final TCP goal transform together. Only
support-plane XY translation and world-Z yaw are sampled: the catalog resting
height and roll/pitch remain fixed, and the kinematic part remains nonmoving.
The canonical goal RGB-D also remains fixed because it describes the correct
grasp relative to the part, while the live camera and physics see the perturbed
actual part pose.

The combined maximum point displacement from translation and yaw is limited to:

```text
stored clearance - 1.0 mm authored minimum - 0.1 mm guard
```

The yaw term uses the conservative bound `2 * part_xy_radius * sin(|yaw|/2)`.
The nominal-clearance table is used for nominal/ready states and the matching
target/axis/waypoint table for rotated states. No-noise, ready, and boundary
resets retain zero object yaw so their completion labels and authored safety
meaning remain intact. No reset mode deliberately starts in contact. The live
simulator smoke reported zero collision rate.

## Curriculum And Episode Timing

The curriculum uses the global simulator-step counter:

- steps 0--16,000: ordinary path resets remain in progress 0.70--0.94; path
  pose perturbation, appearance randomization, and hard-target replay are zero;
- steps 16,000--192,000: minimum progress ramps from 0.70 to 0.0 and
  perturbation, appearance, and replay strength ramp from 0 to 1;
- after step 192,000: the full path/error/appearance distribution is active and
  up to 25% of balanced samples are failure-replay samples.

At a 64-step PPO horizon, full difficulty is reached at about epoch 3,000. The
curriculum does not change `num_envs`, horizon length, rollout batch size, or
minibatch size.

Per-environment timeout tensors replace one fixed training horizon. Close path
states receive about 4 s, far states 12 s, ready states 1.5 s, and boundary
states 2.5 s. Isaac Lab still uses 12 s as the maximum allocated horizon.

## Shared Policy And Autonomous Completion

The actor observation contains:

```text
72 * 128 * 8 RGB-D values = 73,728
previous motion action    =      6
privileged training labels =     8
total                     = 73,742
```

The final eight pose/completion labels are sliced away before action inference.
Only the image pair and previous executed motion action are deployment inputs.

The network now has:

- a 256-value fused visual latent;
- a shared 128-value geometric feature trunk;
- a motion head using visual latent, geometric features, and previous action;
- a completion head using visual latent and geometric features;
- a six-axis pose-error head using the same geometric features;
- the existing centralized critic, which remains privileged during training.

Pose error is an auxiliary weighted smooth-L1 objective and completion is a
masked, positive-weighted binary-cross-entropy objective. They do not directly
supervise motion commands; their shared trunk shapes the visual features PPO
can use. Automated tests confirm that changing only privileged labels leaves
motion, completion, and value outputs exactly unchanged, while all three heads
backpropagate into the shared representation.

The policy still decides when to finish. Merely satisfying geometric tolerances
does not terminate an episode. Ground-truth completion labels are positive only
inside 4 mm and 3 degrees with no unsafe contact, negative outside 6 mm or 4
degrees or in unsafe contact, and ignored in the ambiguity band.

Stochastic PPO emits a Bernoulli stop action. Deterministic deployment uses the
head probability and requires `p(done) >= 0.95`. Both paths require four
consecutive stop frames while TCP speed is below 0.005 m/s and 0.03 rad/s.

## Motion And Reward Changes

- Maximum normalized action change: 0.50 per 15 Hz environment step, preserving
  the earlier 30 Hz slew limit per second.
- Linear command scale: 0.04 m/s, reduced from 0.05 m/s.
- Angular command scale: 0.24 rad/s, reduced from 0.30 rad/s.
- Completion confidence starts slowing Gaussian motion mean and standard
  deviation at 0.70 and reaches a 0.25 scale floor at certainty.
- A stop candidate holds zero twist immediately while the stability/streak
  gate decides whether to end the episode.

The existing potential-difference reward remains the motion teacher. Correct
declared completion pays +50, premature completion and unsafe contact cost -50,
timeout costs -15, and divergence costs -25. Correct completion from a
synthetic ready reset is scaled to +10 so easy examples do not dominate PPO.

Contact risk now adds a squared per-step penalty from 0.05 N up to the 1 N
unsafe-contact threshold with weight 2. This is graded contact-severity shaping,
not an unvalidated geometric distance oracle.

## Visual Randomization

Only live observations and the physical live scene are randomized. Goal RGB-D
catalog images stay canonical.

Sensor-space variation includes exposure, contrast, gamma, white balance,
vignette, noise, blur, depth scale/bias/noise/quantization, edge dropout, 6%
RGB patch occlusion, and 4% depth patch dropout. Patches cover 0.5--3% of the
image. RGB uses the live-image mean as an occluder and depth uses maximum range;
black rectangles are not introduced.

At a 120-step cadence, the shared physical scene varies key-light yaw/pitch,
intensity, angle, temperature, dome lighting, part hue/brightness/roughness,
finger brightness, and ground hue/brightness/roughness. Rotating the key light
changes cast-shadow position. Randomization strength follows the curriculum.

The active catalog was not regenerated by this change. It already carries the
required `jaw_width_plus_10mm_v1` approach profile and the environment rejects a
mismatched catalog. The current checked-in asset contains 1,256 validated
targets: 1,012 train, 125 validation, and 119 test, with RGB shape
`(1256, 144, 256, 3)` and rotation resets `(1256, 8, 32, 7)`.

## Failure-Driven Sampling

Every terminal episode updates an exponential moving failure score for its
target. Timeout and premature stop contribute 1.0, divergence 1.25, collision
1.5, and correct completion 0.0. At full curriculum, 25% of a balanced batch is
redrawn from these scores with a 0.10 exploration floor and power 1.5.

Replacement happens only among targets belonging to the same part as the
original slot, preserving part balance. Scores are intentionally online
environment state and restart at zero for a fresh process; validation reports
provide the durable per-target failure record.

## Periodic Validation And Checkpoint Selection

`euler/watch_validate_and_pull.sh` watches a running Euler job. At the chosen
epoch interval (default usage: 1,000), it:

1. discovers the active Euler run directory from its Slurm log;
2. pulls the matching periodic checkpoint only;
3. runs deterministic far/mid/close evaluation over all 125 validation targets
   on the local GPU while Euler continues training;
4. records termination counts/rates and final completion confidence;
5. ranks all completed validation checkpoints;
6. pulls all durable Euler results when the job finishes.

The selection score is:

```text
success - collision - 0.50 * premature - 0.25 * divergence
        - 0.10 * (timeout + horizon)
```

Outputs live below
`evaluations/periodic_validation/{epoch_<N>,checkpoint_selection.json,checkpoint_selection.md,best_checkpoint.txt}`.
The selector rejects non-validation summaries. `local_policy.sh` and
`benchmark_multipart_policy.py` prefer `best_checkpoint.txt` when it exists.
The 119-target test split is reserved for one final report after selection.

## Changed Paths

- `isaac_rl/.../isaac_rl_env.py` and `isaac_rl_env_cfg.py`: reset mixture,
  curriculum, timeouts, previous-action observation, speed limiting, completion
  gating, risk reward, failure replay, and metrics.
- `isaac_rl/.../training_curriculum.py`: pure curriculum, reset-mode, timeout,
  failure-replay, and failure-score helpers.
- `isaac_rl/.../reset_position_sampling.py`: continuous per-environment
  clearance-capped position profiles.
- `isaac_rl/.../object_pose_sampling.py`: stable support-plane XY/yaw pose
  estimation error with a combined conservative collision-clearance budget.
- `isaac_rl/.../agents/resnet_rgbd_network.py` and both RL-Games YAML files:
  shared geometry/context architecture and completion-linked motion slowdown.
- `grasp_planning/rl/live_observation_randomization.py`: curriculum strength and
  realistic small RGB/depth patches.
- `grasp_planning/rl/scene_appearance_randomization.py`: physical hue variation
  and curriculum-scaled light/material samples.
- `isaac_rl/scripts/rl_games/evaluate_multigrasp.py`: termination counts/rates
  and completion-confidence reporting.
- `euler/watch_validate_and_pull.sh` and
  `euler/select_validation_checkpoint.py`: periodic validation and selection.
- `euler/submit.sh`, `euler/local_policy.sh`, and
  `isaac_rl/scripts/benchmark_multipart_policy.py`: workflow discovery and
  automatic use of the validation-selected checkpoint.
- `isaac_rl/scripts/smoke_env.py`: reset-mode/timeout diagnostics and multi-frame
  completion declaration.
- `tests/test_training_curriculum.py`, `test_reset_position_sampling.py`,
  `test_live_observation_randomization.py`,
  `test_scene_appearance_randomization.py`,
  `test_select_validation_checkpoint.py`, and
  `test_resnet_rgbd_network.py`: focused regression coverage.
- `isaac_rl/README.md` and `euler/README.md`: updated contracts and commands.

## Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q`: 318 passed, one
  host-only skip because RL-Games is available only in the Isaac image.
- Isaac-image network tests: 3 passed, covering privileged-label isolation,
  previous-action context, and gradients through every shared head.
- A 256-environment CUDA tensor check passed for the vectorized RGB/depth patch
  masks, with no per-environment Python loop or GPU synchronization.
- Focused curriculum/randomization/selector tests: 24 passed.
- Python compilation and Ruff checks passed for the changed Python paths.
- Shell syntax checks passed for the modified Euler scripts.
- RTX 4090 multipart smoke, eight environments, two steps:
  `(policy, critic, action) = ((8, 73742), (8, 26), (8, 7))`; path, boundary,
  and ready/exact modes present; reset budgets 1.5--6.33 s; collision rate 0.
- One complete local PPO epoch with eight environments:
  rollout batch 512, minibatch 512, 84 FPS, checkpoint written successfully.
  TensorBoard contained finite actor, critic, central-value, pose auxiliary,
  completion auxiliary, curriculum, reset, contact, and failure-score metrics.
- The first local PPO attempt hit the known NVIDIA 570/container cuDNN mismatch.
  Rerunning with the existing local-only `ISAAC_RL_DISABLE_CUDNN=1` workaround
  completed. Euler does not use this workaround.

Not yet run: a new 256-environment Euler probe, a long convergence run, and the
periodic watcher against a live new-architecture Euler job. Those are the next
operational validations, not missing implementation pieces.

## Commands To Run Next

First confirm the revised architecture at the known 256-environment setting:

```bash
./euler/submit.sh probe \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
    --num_envs 256 \
    --max_iterations 5 \
    --headless \
    --enable_cameras
```

After that probe completes safely, start a fresh run without `--checkpoint`:

```bash
./euler/submit.sh train \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
    --num_envs 256 \
    --max_iterations 10000 \
    --headless \
    --enable_cameras
```

Use the validation-aware command printed by submission, not both watchers:

```bash
./euler/watch_validate_and_pull.sh JOB_ID 1000
```

TensorBoard remains available with:

```bash
./euler/local_policy.sh tensorboard
```

After validation selects the checkpoint, run the test split exactly once:

```bash
ISAAC_RL_POLICY=multipart ./euler/local_policy.sh evaluate \
    --catalog_split test \
    --runs_per_target 3
```

## Risks And Follow-Up

- The collision guarantee is a reset-state guarantee for the modeled
  hand/finger-to-active-part and ground clearances. It does not prove every
  policy action collision-free or introduce arbitrary arm-link SDF shaping.
- Online failure scores are not checkpoint state; a restarted training process
  relearns them.
- Key-light direction and intensity remain global at each slow cadence. Part
  palette and T-slot appearance are now independent per environment and stable
  for the episode.
- Validation runs require the local PC and GPU to remain available. They do not
  consume the one-GPU Euler allocation and therefore can overlap cluster
  training.
- The new run should be compared by held-out validation success and termination
  breakdown, not training return alone.

## 2026-08-20 Euler GPU Selection And Capacity Probes

Euler deployment now targets the `gimenol` staff login and persistent paths in
`/cluster/home/gimenol` and `/cluster/scratch/gimenol`. `euler/submit.sh`
accepts per-job GPU type/count, minimum GPU memory, CPU-per-GPU,
memory-per-CPU, and time-limit selectors while retaining one RTX 4090 as the
safe default. The submission validates the six-GPU/48-core shareholder ceiling
before calling `sbatch` and logs both requested and assigned hardware. GPU counts
greater than one now launch one Slurm task and RL-Games process per allocated
GPU on the same node. Each task receives a one-GPU cgroup before Apptainer and
Vulkan start; each rank owns `--num_envs` environments, a simulator, rollout
buffer, and model replica, while RL-Games synchronizes gradients. The job uses
one shared rank-stable experiment name, writes configuration/checkpoints from rank zero,
requires a completion marker from every rank, samples VRAM on every GPU, and
reports the global rollout batch for watcher ETA calculations.

Two matched single-node probes completed from the same synchronized snapshot.
Job `11327083` used two RTX 4090 GPUs with 224 envs per rank (448 total),
completed both ranks, averaged 2,243 FPS over epochs 2--5, and peaked at
21,992--22,328 MiB per GPU. Job `11327086` used four GPUs with 224 envs per
rank (896 total), completed all four ranks, averaged 4,506 FPS, and peaked at
21,874--22,354 MiB. Relative to the 1,263 FPS one-GPU baseline these are 1.78x
and 3.57x aggregate throughput, with essentially unchanged FPS per simulated
environment.

An earlier torchrun design exposed multiple Vulkan GPUs to every Isaac process,
which fails for the USD-RT RGB-D camera when a process selects logical GPU 1.
Remapping inside a torchrun child was also too late for Vulkan. The corrected
launcher uses `srun --gpus-per-task=1 --gpu-bind=single:1`, forwards each task's
rank and one-GPU visibility through Apptainer, and lets every renderer use its
private logical `cuda:0` while RL-Games/NCCL retain the global ranks.

The current camera-enabled multipart randomization completed a five-iteration,
256-environment RTX 4090 probe in job `11319881`: peak VRAM was 22,656 of
24,564 MiB, stable total throughput was 1,249--1,281 FPS, and PPO training time
was 75.9 seconds. This confirms 256 envs can run but leaves only 7.8% memory;
224 envs is the long-run recommendation on a 24 GB card. Target probes are
queued for RTX PRO 6000 96 GB at 512, 1,024, and 1,280 envs and for A100 80 GB
at 256 envs. Their measured results, rather than proportional VRAM scaling,
will determine the larger production setting. The durable benchmark table and
commands live in `euler/GPU_BENCHMARKS.md`.

## 2026-08-25 Policy Context And Busy-Background Ablations

Added three explicit deployment-measurable actor-context contracts: preceding
applied action (6 values), action plus normalized camera-frame TCP twist (12),
and action/twist plus continuous base-from-camera rotation (18). Training,
evaluation, video recording, smoke testing, checkpoint metadata, and the real
D405 runtime now agree on the selected context layout. The real runtime derives
the twist from the timestamped TCP pose stream and transforms it with the live
camera TF. Privileged pose/completion labels remain rollout-only supervision,
and the learned completion head remains visual-only.

The controlled four-run suite is prepared for three RTX 4090 ranks per policy,
224 environments per rank, and 3,810 epochs per run. This preserves
approximately the same rollout and optimizer-update experience as 256
environments for 10,000 single-GPU epochs while retaining the safer measured
224-environment per-GPU VRAM margin and fitting the 48-hour job limit. Full
contracts, monitoring commands, and verification are recorded in
`POLICY_CONTEXT_ABLATIONS.md`.

Jobs `11733536`, `11733541`, `11733542`, and `11733543` were submitted by
mistake and immediately cancelled at zero elapsed runtime. No ablation training
is active; `euler/submit_policy_context_ablations.sh` is ready for an explicit
future launch.

## 2026-08-25 Shared 15 Hz Policy Contract

All task and policy-context variants now run the wrist observation, actor, and
command boundary at 15 Hz. Isaac keeps 120 Hz physics with decimation 8. The
real D405 continues streaming synchronized RGB and aligned depth at about
30 Hz, but a timestamp gate accepts at most every other frame for inference
and command publication. Training metadata and deployment sidecars record
`policy_rate_hz: 15.0`, and the real runtime rejects a mismatched rate.

Time-dependent settings were converted rather than silently changing physical
behavior: normalized slew is 0.50 per step, response alpha starts at 0.91,
observation/action delay is at most one step, appearance refresh is every 60
steps, per-time penalties scale with the policy period, and PPO uses
`gamma=0.9801`, `tau=0.9025`. Four completion decisions remain required, now
an intentional roughly 267 ms stability hold.

Verification completed with 48 focused tests and a two-environment,
two-step Isaac smoke using the largest 18-value actor context. Isaac reported
physics `0.008333 s`, render/environment `0.066667 s`,
`policy_rate_hz=15.0`, and zero collision rate. No Euler job was submitted or
started for this change.

## 2026-08-25 Stable Actual Part Versus Nominal Pose Estimate

Multipart path resets now model error in the perceived part frame, including
continuous horizontal translation and in-plane yaw. This is not arbitrary
physical pose randomization. Every actual part begins from its validated
catalog resting pose and is transformed only within the support plane: world Z
and roll/pitch are unchanged, so floor support and resting stability remain
valid. Parts remain kinematic and therefore do not drift after reset.

The robot stays at the trajectory waypoint authored from the nominal estimated
part pose. The physical part, live camera render, physics collision geometry,
and privileged correct final TCP target use the stable perturbed actual pose.
The target follows the same rigid transform as the part, preserving the exact
grasp in part coordinates. The canonical goal image is intentionally not
perturbed: it continues to show the desired relative grasp, while the live
image contains the pose-estimation discrepancy that the policy must correct.

Object yaw tapers from at most 10 degrees far from the goal to 3 degrees near
it and is curriculum-scaled. For each exact target, authored gripper rotation,
and path waypoint, translation plus the worst-case yaw-induced surface motion
must fit inside the stored collision clearance after retaining the 1.0 mm
authored minimum and 0.1 mm guard. No-noise, ready, and boundary reset modes
keep object yaw zero.

The isolated four-case inspection artifact is
`artifacts/object_pose_randomization_check/object_pose_randomization_check.png`.
It compares nominal, translation-only, yaw-only, and combined resets; all four
were collision-free, canonical goal RGB was byte-identical, and the target
transform in part coordinates was invariant to numerical precision. Focused
logic/profile tests passed (`25 passed`), Python compilation and Ruff passed,
and a 64-environment, two-step production multipart Isaac smoke exercised the
full reset mixture with XY offsets up to 7.75 mm and object yaw from -7.86 to
+7.35 degrees at zero collision rate.

## 2026-08-25 PDZ Gripper And MuJoCo Goal Catalog Migration

The visual-servo task now uses the real PDZ parallel gripper consistently in
grasp generation, MoveIt planning, reset-trajectory IK, collision validation,
goal rendering, Isaac simulation, and deployment metadata. The aperture is
12--76 mm; contact grasps stop at 62 mm and the approach state adds exactly
5 mm per finger. The active TCP is the named `pdz_gripper_tcp` frame at
`[0, 0, 0.1355]` in the PDZ base with a -90 degree local-Z rotation. The D405
optical frame remains attached to link 7 at
`[0.009, -0.050560254038, 0.097927071163]` with the measured 848x480
intrinsics.

The rebuilt multipart data flow found 661 PDZ geometry candidates, validated
482 through MoveIt, retained 481 Cartesian approach trajectories, and retained
393 targets after eight-reset collision validation against the active part and
flat support plane. The final split is 313 train, 38 validation, and 42 test
targets across all five parts. Each retained target has eight 32-waypoint
rotation-reset trajectories and at least 1 mm modeled clearance.

Goal RGB-D is now rendered with the experimental MuJoCo Filament backend, not
with the former Isaac/Y-gripper capture. URDF import runs in a stock-MuJoCo
subprocess because the experimental preload cannot safely iterate `MjSpec`
strings; the parent Filament process then compiles the canonical MJCF, adds the
exact D405 camera, hides only the camera enclosure surfaces containing the
optical origin, and renders the PDZ fingers, white contact pads, part, and
T-slot surface. All 393 captures passed the strict TCP and depth-quality
checks. The final arrays are RGB `(393, 144, 256, 3)`, depth
`(393, 144, 256)`, and resets `(393, 8, 32, 7)`.

Verification completed with 406 host tests passing (two RL-Games-only tests
skipped there), all nine skipped tests passing inside the Isaac environment,
a four-environment production Isaac smoke, and real PPO updates for all four
prepared configurations. The smoke loaded 313/393 training targets, PDZ USD,
eight collision-validated resets, 15 Hz timing, stable support-plane part-pose
estimate error, and D405 sensor/control randomization with zero collision rate.

The object-pose inspection artifact was regenerated after the PDZ migration,
so `artifacts/object_pose_randomization_check/object_pose_randomization_check.png`
now shows the PDZ articulation in the external scene and live policy RGB. All
four nominal/translation/yaw cases remain collision-free, their canonical goal
RGB is byte-identical, and their part-relative target invariance error remains
below `0.00001 mm`.

The nested Isaac extension suite also passes 27 tests with one optional skip.
Rotation-reset loading now fails closed on mismatched robot and approach-gripper
profiles in addition to its existing schema, target-ID, IK, and collision checks.

## 2026-08-25 PDZ Three-GPU Ablation Launch

The synchronized PDZ/Filament/15 Hz/support-plane-error snapshot passed a
three-epoch distributed preflight (`11746133`) on three RTX 4090 GPUs with 224
environments per rank. All three ranks completed, throughput reached 1,856 FPS,
and measured peak device memory was 19,506 MiB of 24,564 MiB.

Four full 3,810-epoch jobs were then submitted with three RTX 4090 ranks and
224 environments per rank: baseline `combined_sim2real` plus `action`
(`11746645`), busy background plus `action` (`11746654`),
`combined_sim2real` plus `action_twist` (`11746657`), and
`combined_sim2real` plus `action_twist_rotation` (`11746661`). The first three
started immediately. Euler subsequently admitted the fourth while those jobs
were still running, so all four controlled ablations now train concurrently.

Early live verification passed the first checkpoint boundary for baseline,
background, and velocity. Baseline reward improved from -12.08 at epoch 50 to
+13.04 by epoch 88; background improved from -9.66 to -0.68 by epoch 77; and
velocity saved its first checkpoint at epoch 50 with best reward -11.70.
Per-rank live PyTorch allocations were flat after startup. Baseline rank 0 made
one early 3,420 MiB cached-reservation step and then stayed flat, leaving about
4 GiB reusable inside that cache. Several GPUs later made isolated 64 MiB
driver-reported steps without any PyTorch allocated/reserved change. The worst
observed device-free value is 1,091 MiB on that cached baseline rank; other
active ranks retain roughly 4.5--4.8 GiB. A deliberately short-window slope
check initially failed baseline, while background and velocity passed. At the
epoch-200 and post-curriculum rechecks, the production 150-sample analyzer
passed all three ranks of baseline, background, and velocity. No job has
emitted an OOM, traceback, NCCL error, non-finite scalar, or rank failure.
Velocity subsequently crossed epoch 100, wrote its periodic checkpoint, and
all three ranks remained exactly flat through epochs 94--101. The isolated
64 MiB steps are therefore not a per-checkpoint leak.

Live TensorBoard inspection found zero non-finite values across all 90 scalar
tags in each active job. The path-reset XY/yaw and visual-randomization scales
remain intentionally zero during the 16,000-policy-step warmup (250 PPO epochs
at horizon 64), then ramp to full strength at 192,000 steps (epoch 3,000).
The 3,810-epoch schedule therefore includes 810 epochs at full curriculum
strength. Live events from baseline, background, and velocity confirm that
curriculum, perturbation, visual strength, and object yaw are exactly zero
through epoch 250 and become nonzero at epoch 251.

The velocity-rotation job also passed distributed startup, its epoch-100 and
epoch-200 checkpoints, and serialized-contract inspection: three ranks each
use 224 environments, `action_twist_rotation`, decimation 8 / 15 Hz, and the
`combined_sim2real` profile. Its 90 scalar streams are finite and its three
ranks retain 4.6--4.7 GiB free. Like the earlier jobs, two ranks made one
isolated 64 MiB driver-only step while PyTorch allocations and reservations
stayed flat. After epoch 260, the production 150-sample memory analyzer passed
all three ranks, and live events confirmed the zero-through-250/nonzero-from-
251 curriculum transition. All four jobs have therefore passed the same
startup, repeated-checkpoint, finite-metric, memory, and curriculum gates.

Each job has a detached `watch_and_pull.sh` tmux session protected by a system
sleep inhibitor. Result pulls now take an exclusive local `flock` and retry
after transient SSH/rsync failures, preventing concurrent watchers from
updating the same checkpoint or TensorBoard event file. Final job outcomes and
pulled-artifact verification remain pending while training runs.

An independent suite supervisor, `euler/watch_ablation_suite.sh`, is also active
in tmux session `euler_suite_20260825_165140` using the tracked manifest
`euler/ablation_suite_20260825-165140.json`. After every job becomes terminal it
performs another serialized/retrying pull and writes a final local verification
report. `euler/verify_pulled_ablation_suite.py` requires `COMPLETED/0:0`, three
rank completion markers, epoch 3,810/3,810, at least one checkpoint and event
file, matching serialized profile/context, three finite rank-memory CSVs with
at least 1,024 MiB free, and a passing long-run memory analysis. It also parses
the TensorBoard protobufs, requires the core reward/success/collision/completion
scalars, rejects every non-finite scalar, and requires `rewards/iter` to reach
step 3,810. Its pass and fail-closed fixture tests cover tagged Hydra YAML,
non-finite memory/scalars, incomplete ranks, and empty/incomplete artifacts.

## 2026-08-26 All-Fabrica PDZ Grasp Catalog

The low-cap `fabrica_all_v1` rollout now inventories all 46 configured Fabrica
OBJ parts at mesh scale `0.01` and generates globally namespaced grasp IDs
across all seven assemblies. The all-part benchmark evaluated 264 stable
orientations: 69 produced direct stage-2 grasps, 188 had no stage-2 direct
grasp, and seven failed stage 1. Geometry selection retained 38 feasible parts
and explicitly excluded eight parts rather than weakening the physical filters:
all five configured `beam` parts, `cooling_manifold__part_1`, and
`duct__part_0`/`duct__part_1`.

The merged geometry manifest selected 1,940 diverse targets across 65 retained
orientations, with 6,255 alternates after enforcing the PDZ 12 mm minimum and
66 mm maximum training jaw widths. MoveIt validated 1,875 targets; straight
Cartesian path generation retained 1,869. Collision-safe rotation-reset
generation retained 1,488 targets and rejected 381 while preserving the strict
1 mm modeled clearance. Every retained target has eight 32-waypoint reset
variants; the worst rotation-reset IK residual is 0.0469 mm / 0.02732 degrees.

MuJoCo Filament captured and strictly reloaded one canonical D405 RGB-D goal
for every retained target. The finalized arrays are RGB
`(1488, 144, 256, 3)`, depth `(1488, 144, 256)`, nominal paths
`(1488, 32, 7)`, and rotation resets `(1488, 8, 32, 7)`. Ordered target IDs
match exactly across goal, path, and rotation-reset assets. The final primary
held-out-grasp split contains 1,210 train, 137 validation, and 141 test targets;
no `(assembly_name, local_part_id, grasp_id)` group crosses a split. All 38
represented parts retain at least eight final targets, and
`isaac_rl/data/fabrica_all_v1/dataset_index.json` reports
`training_ready: true`.

Generated benchmark and catalog artifacts remain outside normal Git history at
`artifacts/grasp_generation_benchmark_pdz/` (about 35 GiB) and
`isaac_rl/data/fabrica_all_v1/` (about 345 MiB). The legacy plumbers-block
catalog is unchanged. Sharded Isaac scene/task integration described as the
next phase of the implementation plan is still separate from this completed
grasp-generation catalog.
