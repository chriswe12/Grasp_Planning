# Policy Context And Background Ablations

Date: 2026-08-25
Seed: `42`

Status: PDZ three-GPU preflight passed; the full suite is submitted with
persistent automatic result watchers.

## Controlled Runs

All four prepared policies use the same task, goal catalogue, PPO configuration,
training split, reset/curriculum logic, camera model, completion head, and
random seed. Each run uses three GPUs with 224 environments per rank and 3,810
PPO epochs. This preserves approximately the same rollout exposure as 256
environments for 10,000 single-GPU epochs:
`3 * 224 * 3,810 ~= 256 * 10,000` environment-epochs.

The three-rank batch resolver selects a divisible minibatch of 256 per rank,
giving an effective global minibatch of 768 and 112 optimizer updates per PPO
epoch. Thus the suite intentionally has the same rollout exposure but about
33% more, smaller-batch optimizer updates than the older single-GPU reference.

All four policies share an explicit 15 Hz policy/observation/control contract.
Isaac physics remains at 120 Hz and uses decimation 8, so each policy action
and rendered wrist RGB-D observation covers `1/15 s`. The real D405 streams
remain at approximately 30 Hz; deployment consumes at most every other
synchronized frame and publishes policy commands at 15 Hz.

All four also share the same stable-part pose-estimation-error distribution.
The actual part begins in its validated catalog resting pose and varies only in
support-plane XY/yaw, while Z and roll/pitch stay fixed. The robot trajectory is
conditioned on the nominal estimated pose; the live image and collision scene
contain the stable actual pose; and the correct TCP target remains rigidly
attached to the actual part. This reset change is therefore controlled across
the ablations rather than becoming a fifth experimental variable.

| Run | Sim-to-real profile | Actor context | Question |
|---|---|---|---|
| baseline | `combined_sim2real` | previous applied action (6) | Current reference |
| background | `combined_busy_background` | previous applied action (6) | Does peripheral office/factory clutter improve robustness? |
| velocity | `combined_sim2real` | action + camera-frame TCP twist (12) | Does measured motion state improve control? |
| velocity-rotation | `combined_sim2real` | action + twist + base-from-camera rotation 6D (18) | Does absolute wrist orientation add useful context beyond twist? |

The completion head remains visual-only in every run. Pose error, completion
ground truth, joint state, and other simulator-only values are never connected
to the actor action path.

## Context Contract

- Previous action: the six normalized motion values actually applied after
  filtering and safety gating.
- TCP twist: measured linear and angular velocity of the TCP parent link,
  expressed in the optical-camera frame. Linear velocity is divided by
  `0.04 m/s`, angular velocity by `0.24 rad/s`, and values are clipped to
  `[-5, 5]`.
- Camera rotation: `R_base_from_camera` encoded continuously as the first two
  rotation-matrix columns concatenated into six values. A quaternion was not
  used because of its sign ambiguity.

The real D405 runtime reconstructs the same context from the timestamped TCP
pose stream and the live camera TF. Checkpoint sidecars record
`policy_context_mode`, `policy_context_size`, and the resulting full network
input size, preventing a checkpoint from being loaded with the wrong layout.
They also bind `policy_rate_hz: 15.0` and reject checkpoints with a different
rate contract.

## 15 Hz Timing Equivalence

The rate change preserves the important physical-time behavior of the former
30 Hz setup:

- the normalized action slew limit is `0.50` per 15 Hz step, preserving the
  former maximum slew per second of `0.25` per 30 Hz step;
- response filtering uses alpha `0.91`, the two-step time-equivalent of the
  former `0.70` response;
- simulated observation and command delays are zero or one 15 Hz step;
- scene appearance changes every 60 policy steps, still every four seconds;
- per-step collision, action, and living penalties are scaled by elapsed
  policy time;
- PPO uses `gamma=0.9801` and `tau=0.9025`, time-equivalent to `0.99/0.95` at
  30 Hz.

Completion still requires four consecutive policy decisions. At 15 Hz this is
an intentional approximately 267 ms stability hold before termination or
gripper permission.

## Submitted Euler Suite

The exact PDZ/15 Hz/stable-pose-error source snapshot was synchronized and the
suite was submitted on 2026-08-25:

| Job ID | Run | Profile | Context |
|---|---|---|---|
| `11746645` | baseline | `combined_sim2real` | `action` |
| `11746654` | background | `combined_busy_background` | `action` |
| `11746657` | velocity | `combined_sim2real` | `action_twist` |
| `11746661` | velocity-rotation | `combined_sim2real` | `action_twist_rotation` |

Each run requests three RTX 4090 GPUs, 224 environments per rank, 3,810 PPO
epochs, and at most 48 hours. Euler admitted baseline, background, and velocity
concurrently; velocity-rotation is eligible but held by
`QOSMaxCpuPerUserLimit` until one of those 24-core allocations ends. A detached
`euler_pull_<job-id>` tmux session and sleep inhibitor is active for each job,
running `euler/watch_and_pull.sh` until the job terminates and its results are
present locally. Result pulls are mutually exclusive and retry transient
SSH/rsync failures indefinitely.

The tracked suite manifest is
`euler/ablation_suite_20260825-165140.json`. A second detached supervisor,
`euler_suite_20260825_165140`, runs `euler/watch_ablation_suite.sh` and
independently performs a final pull and fail-closed artifact verification after
all four jobs are terminal. Its report will be written to
`logs/euler/ablation-suite-20260825-165140.verification.json`. Verification
parses every TensorBoard scalar, rejects non-finite values, and requires the
reward stream to reach epoch 3,810. TensorBoard compares the runs after they are
pulled:

```bash
./euler/local_policy.sh tensorboard
```

Evaluation and debug-video commands must pass the checkpoint's matching
`--policy-context` value. The baseline and background checkpoints use
`action`, velocity uses `action_twist`, and velocity-rotation uses
`action_twist_rotation`. The 15 Hz rate is part of the task configuration and
does not require an additional command-line flag.

## Verification Before Submission

- 15 Hz timing, context, profile, deployment-runtime, safety, curriculum, and
  observation-randomization tests: 48 passed.
- Final two-environment Isaac smoke with the largest
  `action_twist_rotation` context reported physics `0.008333 s`, render and
  environment step `0.066667 s`, `policy_rate_hz=15.0`, two completed steps,
  and zero collision rate.
- Focused policy/runtime/safety/profile tests: 40 passed, one optional
  RL-Games import test skipped under system Python.
- Final focused ablation/background/runtime set: 35 passed.
- Isaac smoke for `action_twist_rotation`: policy observation `(2, 73754)`,
  critic `(2, 26)`, action `(2, 7)`, two steps completed.
- Isaac smoke for `combined_busy_background` + `action`: policy observation
  `(2, 73742)`, active background and clutter observed, one step completed.
- One complete local RL-Games PPO epoch for `action_twist_rotation`; checkpoint
  written and `Training time` completion marker emitted.
- Euler preflight `11746133` completed three epochs across three RTX 4090 GPUs
  at 224 environments per rank. All ranks emitted `Training time`; observed
  throughput reached 1,856 FPS, and peak device memory was 19,506 MiB of
  24,564 MiB with no OOM or rank failure.
- Baseline, background, and velocity each crossed epoch 50 and wrote a durable
  checkpoint. Live PyTorch allocations/reservations are flat after their
  startup caches. Isolated 64 MiB driver-reported steps occurred on several
  GPUs without tensor/reservation growth; baseline rank 0 retains 1,091 MiB
  device-free plus roughly 4 GiB reusable inside its PyTorch cache, while the
  other ranks retain about 4.5--4.8 GiB device-free. Epoch 200 and curriculum
  activation at epoch 250 are explicit memory rechecks. Inspection found no
  non-finite value across all 90 TensorBoard scalar tags per run and no OOM,
  traceback, NCCL, or rank failure.
- Curriculum warmup lasts 16,000 policy steps (epoch 250 with horizon 64), so
  path-pose and visual-randomization strength is intentionally zero in the
  initial scalar samples. It reaches full strength at epoch 3,000 and remains
  full for the final 810 epochs.
- The suite-level verifier has pass and fail-closed fixture coverage; the full
  host suite currently passes 409 tests with two environment-only skips.
- A mistaken submission produced jobs `11733536`, `11733541`, `11733542`, and
  `11733543`; all four were cancelled at zero elapsed runtime before starting.
  They are historical cancelled records, not training runs.
