# Visual-Servo Training Revision Completion

Date: 2026-08-14
Source branch: `grasping-rl`
Source commit at start: `5159a1959c0cba818e4210d9e517f25c323df47f`
Change status: local-only in an already dirty worktree

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
| Path | 55% | Authored path waypoint; continuous collision-capped XY error; authored rotation enabled by curriculum | 4--12 s by progress |
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

For XY error, the robot remains on an authored state while the active object and
final TCP goal translate together. The magnitude is limited to:

```text
stored clearance - 1.0 mm authored minimum - 0.1 mm guard
```

The nominal-clearance table is used for nominal/ready states and the matching
target/axis/waypoint table for rotated states. No reset mode deliberately starts
in contact. The live eight-environment simulator smoke reported zero collision
rate.

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

- Maximum normalized action change: 0.25 per environment step.
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
- Physical material/light randomization is global across cloned environments at
  each cadence because the rendered materials and lights are shared.
- Validation runs require the local PC and GPU to remain available. They do not
  consume the one-GPU Euler allocation and therefore can overlap cluster
  training.
- The new run should be compared by held-out validation success and termination
  breakdown, not training return alone.
