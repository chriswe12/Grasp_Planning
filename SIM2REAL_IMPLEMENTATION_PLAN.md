# D405 Visual-Servo Sim-to-Real Implementation Plan

## Implementation Status (2026-08-19)

Implemented without real-camera recordings:

- One corrected v7 camera rotation for both rendering and action transforms.
- Shared validity-aware aligned RGB-D resize, range handling, normalization,
  and optional plumb-bob rectification for deployment input.
- D405/D400-inspired disparity-space depth error with fixed spatial structure,
  temporal correlation, independent residual error, quantization, invalid
  range, sparse dropout, and horizontal stereo-boundary failures.
- Coupled RGB/depth calibration warp, clean-environment mixture, frame delay,
  frame repeats, action delay, actuator response, and joint gain variation.
- Physical light/shadow/T-slot randomization retained, with a weighted
  per-environment part palette. The canonical render-only T-slot pattern is
  half the earlier prototype scale over the unchanged flat collision plane.
- Named `nominal`, ablation, combined, clutter, depth-robust, and stress profiles selectable from
  training and evaluation, with exact parameters saved in YAML, TensorBoard,
  Markdown, JSON, and CSV artifacts.
- Unit/source-contract coverage and a complete repository test pass.
- A fail-closed real deployment path that keeps stage-2/MoveIt planning to
  pregrasp, runs deterministic D405 policy twists through collision-checking
  MoveIt Servo until learned completion, and independently gates gripper close.
  The checked-in sink remains dry-run and hardware acceptance is still pending.

Provisional assumptions use the documented 18 mm D405 stereo baseline,
7--50 cm operating range, D400 1/32-pixel subpixel disparity behavior, and a
0.1 mm close-range depth unit. These values are intentionally versioned as
`d405_documented_provisional_v6_15hz` and must later be replaced by measurements of
the actual serial-numbered camera.

Official references used for the provisional envelope:

- [D405 product specifications](https://www.realsenseai.com/products/stereo-depth-camera-d405/)
  for the 7--50 cm ideal range, global shutter, and accuracy rating.
- [D400-series product-family datasheet](https://dev.realsenseai.com/download/42003/)
  for the D405's 18 mm stereo baseline.
- [D400 subpixel-linearity white paper](https://dev.realsenseai.com/docs/white-paper-subpixel-linearity-improvement-for-intel-realsense-depth-cameras/)
  for 1/32-pixel subpixel disparity and the 100 micrometre close-range depth-
  unit recommendation.
- [RealSense SDK projection documentation](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/)
  for Z16 scaling and zero-valued invalid depth.
- [D400 depth post-processing white paper](https://dev.realsenseai.com/docs/depth-post-processing-for-intel-realsense-depth-camera-d400-series/)
  for spatial/temporal correlation and edge-aware filtering behavior.

Still recording-dependent and therefore deferred:

- Fitting exact bias/noise/dropout/correlation ranges from plane captures.
- Checking which bracket, fastener, cable, or wrist-cover silhouettes are
  persistently visible and adding only those meshes.
- Real-frame replay, simulated-versus-real diagnostic grids, and real closed-
  loop safety validation.

Object mass and contact friction variation is not active because the current
alignment environment keeps the target kinematic and stops before grasp
contact; those parameters would not alter its rollouts. Arm stiffness and
damping variation is active because it changes tracking behavior.

## Objective

Prepare the multipart RGB-D visual-servo policy for transfer from Isaac Sim to
the physical KUKA iiwa7/Y-gripper setup with RealSense D405 serial
`260322275185`.

The implementation order is:

```text
action-frame correctness
-> shared preprocessing
-> real D405 measurement
-> correlated disparity-depth model
-> camera uncertainty
-> timing and control mismatch
-> profile-based evaluation
-> regenerate v7 catalogue
-> compute probes
-> training
-> offline real replay
-> safe robot test
```

## Current Baseline

The multipart task already contains:

- Continuous collision-safe nominal-to-actual part-frame error: stable
  support-plane XY/yaw perturbations plus authored gripper-rotation resets.
- No-noise, ready, boundary, and already-successful reset cases.
- Variable episode durations.
- Curriculum-driven reset and visual-randomization strength.
- Failure-driven target replay.
- Live-only RGB exposure, contrast, gamma, white-balance, vignette, blur, and
  noise randomization.
- RGB patch occlusion.
- Depth scale, bias, constant metric noise, quantization, edge dropout, and
  connected patch dropout.
- Physical key-light, shadow, and small-pitch T-slot appearance/layout
  randomization, plus a weighted 24-color live part palette, over the same flat
  collision plane.
- Learned completion probability with a low-speed, consecutive-frame stop
  gate.

The active nominal camera profile is v7:

- D405 native resolution: `848 x 480`.
- Intrinsics: `fx=436.3104`, `fy=435.6493`, `cx=418.6266`,
  `cy=236.5121` pixels.
- Camera origin in `lbr_link_ee`: `(55.667, 9.000, 70.776) mm`.
- Camera origin in Isaac `link7`: `(55.667, 9.000, 105.776) mm`.
- A 180-degree tool-Z correction is applied to the optical orientation.

## Phase 0: Correctness Blockers

**Priority:** Critical
**Difficulty:** Low to medium

### 0.1 Unify rendered and control camera frames

Rendering uses the corrected v7 camera pose, but `isaac_rl_env.py` currently
constructs `rotation_tcp_from_camera` directly from the original calibration
matrix. That bypasses the 180-degree v7 correction.

Change the policy action-frame transform to consume the same central camera
pose helper used by the renderer.

Acceptance criteria:

- Rendered optical axes and policy action axes match exactly.
- Positive and negative camera-frame translations move in the expected image
  directions.
- Positive and negative camera-frame rotations use the expected axes.
- Unit tests cover all six signed motion components.

### 0.2 Make v7 the single source of truth

Refactor `grasp_planning/d405_wrist_camera.py` to expose:

- `T_link7_camera`.
- Camera position.
- Camera rotation matrix.
- Camera quaternion.
- Native intrinsics.
- Distortion metadata.
- Depth range and units.
- Camera-profile identifier.

Goal rendering, training, playback, evaluation, and real inference must consume
these helpers rather than reconstructing transforms independently.

## Phase 1: Shared RGB-D Preprocessing Contract

**Priority:** Critical
**Difficulty:** Medium

Create `grasp_planning/rl/d405_observation.py` and define the deployment
pipeline:

```text
848x480 D405 RGB and depth
-> rectify using K and distortion
-> align depth to color
-> convert depth units to metres
-> identify invalid pixels
-> resize to 256x144
-> area-downsample to 128x72
-> encode invalid depth
-> normalize RGB and depth
-> concatenate live and goal observations
```

Requirements:

- Keep Isaac goal and live images undistorted.
- Rectify real D405 images before shared preprocessing.
- Define one invalid-depth representation.
- Preserve validity while resizing instead of averaging invalid zero values
  into valid depth.
- Store preprocessing metadata in the camera profile.
- Reject goal catalogues with incompatible observation profiles.

Tests:

- Known synthetic inputs produce the expected normalized tensor.
- Real and simulated inputs match after the modality-specific rectification
  step.
- Invalid depth survives resizing correctly.
- Metric-depth normalization is reversible within quantization error.
- No NaNs or infinities reach the actor.

## Phase 2: Measure the Real D405 Distribution

**Priority:** High
**Difficulty:** Low to medium

Add `scripts/analyze_d405_depth_capture.py`.

Capture approximately 100 static RGB-D frames at:

```text
7 cm, 10 cm, 15 cm, 25 cm, and 40 cm
```

Estimate:

- Mean depth bias versus distance.
- Noise standard deviation versus distance.
- Disparity-domain noise.
- Spatial correlation length.
- Temporal correlation.
- Fixed-pattern error.
- Invalid-pixel frequency and persistence.
- Edge-error width.
- Exposure and RGB statistics.
- Actual deployed depth units.

Outputs:

```text
artifacts/d405_calibration/d405_noise_profile.json
artifacts/d405_calibration/depth_statistics.csv
artifacts/d405_calibration/error_plots/
```

This capture calibrates simulation ranges. It is not demonstration or action
training data.

## Phase 3: Correlated D405 Depth Model

**Priority:** High
**Difficulty:** High

Extend `grasp_planning/rl/live_observation_randomization.py`.

### 3.1 Disparity-space error

Model stereo error using:

```text
disparity = focal_length * baseline / depth
noisy_disparity = disparity + disparity_error
noisy_depth = focal_length * baseline / noisy_disparity
```

This makes metric depth error grow naturally with distance.

### 3.2 Episode-stable components

Sample per episode:

- Global disparity bias.
- Residual depth scale and offset.
- Low-frequency fixed-pattern map.

### 3.3 Spatial correlation

Combine:

- Small-scale correlated fields.
- Medium-scale correlated fields.
- Large low-frequency bias.
- A smaller independent-pixel component.

Generate low-resolution random fields and upsample them to the `128 x 72`
policy resolution for batched GPU efficiency.

### 3.4 Temporal correlation

Maintain per-environment state:

```text
error_t = rho * error_t-1 + sqrt(1 - rho^2) * innovation_t
```

Reset only environments whose episodes ended.

### 3.5 Stereo boundary failures

Around strong depth gradients:

- Expand uncertainty regions, with stronger horizontal expansion.
- Sample foreground depth, background depth, or invalid depth.
- Add occasional short horizontal streaks.
- Persist some missing regions for multiple frames.

### 3.6 Range and quantization

- Degrade or invalidate measurements below approximately 7 cm.
- Use the actual deployed depth unit.
- Keep a configurable 50 cm maximum.
- Quantize after continuous noise.

### 3.7 Optional texture dependence

After measurement, optionally increase uncertainty for locally textureless,
dark, or saturated RGB regions. Keep this disabled until real data justifies
it.

Tests:

- Zero strength is exactly identity.
- Metric error grows with depth.
- Neighboring pixels have positive covariance.
- Consecutive frames have the configured temporal correlation.
- Resetting one environment does not reset another.
- Disparity remains valid and depth contains no NaNs.
- The model remains practical with 256 environments.

## Phase 4: Bounded Camera Uncertainty

**Priority:** High
**Difficulty:** Medium to high

### 4.1 Extrinsic uncertainty

Randomize only the live rendered camera per episode around v7:

```text
translation: approximately +/-2 to 3 mm
rotation: approximately +/-0.5 to 1.0 degrees
```

The controller continues using nominal v7. This models hand-eye calibration
error. The goal catalogue remains nominal and deterministic.

### 4.2 Intrinsic uncertainty

Apply a consistent image-space warp to RGB, depth, and validity:

```text
fx and fy: +/-0.5 to 1.0 percent
cx and cy: +/-2 to 4 native pixels
```

This avoids rebuilding camera renderers per environment.

### 4.3 Curriculum and clean mixture

- Begin with zero uncertainty.
- Ramp uncertainty with the existing visual curriculum.
- Retain 15 to 20 percent clean or weakly randomized environments after the
  curriculum reaches full strength.

## Phase 5: Camera and Control Timing

**Priority:** High
**Difficulty:** Medium

Implement in the direct RL environment.

### 5.1 Observation history

- Maintain a short live RGB-D history.
- Sample a delay of zero to one 15 Hz policy step.
- Repeat an old frame with approximately 1 to 3 percent probability.
- Never delay the static goal image.

### 5.2 Motion-command history

- Sample a delay of zero to one 15 Hz policy step.
- Do not add a two-step delay to the provisional 15 Hz profile.
- Delay the six motion commands, not the immediate completion hold.

### 5.3 Controller response

Randomize per episode:

- Linear and angular response scale.
- Low-pass response coefficient.
- Small persistent motion-command bias.
- Optional controller gain and damping scale.

Begin with approximately `+/-10 to 15 percent` response uncertainty.

Tests:

- Queues reset independently.
- Completion immediately suppresses motion.
- Delayed motion remains within action bounds.
- No action from the previous episode leaks into a reset environment.

## Phase 6: Appearance Randomization Calibration

**Priority:** Medium
**Difficulty:** Low

Retain the existing RGB, light, shadow, part, finger, and ground
randomizations. Do not broaden them without evidence.

Create diagnostic grids containing:

```text
canonical simulation
weak randomization
medium randomization
full randomization
real D405 examples
```

Adjust ranges only when real images lie outside the simulated distribution.

## Phase 7: Persistent Visible Hardware

**Priority:** Conditional
**Difficulty:** Medium

Inspect real D405 frames for visible:

- Camera bracket.
- Fasteners.
- Cable.
- Gripper adapter.
- Wrist cover.
- Self-occluding hardware near image boundaries.

If visible, add simplified non-colliding meshes matching position and
silhouette. Do not include the orange inspector box in training images.

## Phase 8: Modest Physics Randomization

**Priority:** Medium to low
**Difficulty:** Medium

Initial ranges:

```text
object mass:       +/-15 percent
object friction:   +/-20 percent
gripper friction:  +/-15 percent
joint damping:     +/-10 percent
controller gains:  +/-10 percent
```

Requirements:

- Keep all values physically valid.
- Preserve collision-safe reset guarantees.
- Do not randomize object scale.
- Log the sampled distributions.
- Prefer setup/reset-time changes over unstable per-step physics mutations.

## Phase 9: Regenerate the Canonical v7 Catalogue

**Priority:** Critical
**Difficulty:** Medium

After camera and preprocessing behavior is frozen:

1. Render all 1,119 goal targets.
2. Validate TCP position and rotation.
3. Validate target IDs, part IDs, and splits.
4. Generate the complete RGB contact sheet.
5. Confirm train, validation, and test counts.
6. Reject v5/v6 catalogues automatically.
7. Keep goals canonical, without live observation randomization.

The physical part always starts from a validated stable catalog resting pose.
Pose-estimate error may change only support-plane XY and yaw; it must not change
world Z or roll/pitch. The live render and physics use this stable actual pose,
the robot path represents the nominal estimate, the final TCP target remains
rigidly attached to the actual part, and the canonical goal stays unchanged.

Rotation and position reset assets require regeneration only if collision or
robot/part geometry changes.

## Phase 10: Reproducible Randomization Profiles

**Priority:** High
**Difficulty:** Medium

Define:

```text
nominal
sensor_only
camera_uncertainty
timing_control
appearance
combined_sim2real
stress_test
```

Every profile records:

- Stable identifier.
- Exact parameter ranges.
- Random seed.
- Camera and preprocessing profile.
- Depth-noise profile.
- Training/evaluation mode.

Write the selected profile to training logs, checkpoints, evaluation JSON,
videos, and TensorBoard configuration text.

## Phase 11: Evaluation and Checkpoint Selection

**Priority:** High
**Difficulty:** Medium to high

Periodic evaluation should run:

1. Nominal validation.
2. Nominal held-out test.
3. Sensor-only randomization.
4. Camera uncertainty.
5. Timing and controller uncertainty.
6. Combined sim-to-real randomization.
7. Stress testing.

Report:

- Overall and per-part success.
- Success by initial distance, rotation, and positional offset.
- Premature and missed completion.
- Collision and timeout rates.
- Final position and rotation error.
- Completion calibration.
- Performance by corruption severity.

Select the best checkpoint using a composite score based on combined success,
collisions, premature completion, and worst-part performance instead of only
nominal mean success.

Initial gates:

- No NaNs or renderer failures.
- Nominal success no more than three percentage points below the clean
  baseline.
- Premature-completion rate below 1 percent.
- Collision rate below 1 percent.
- No part fails catastrophically relative to the aggregate.
- Combined-randomization results improve through training.

Absolute success thresholds should be set after the first nominal v7
benchmark.

## Phase 12: Offline Real-Frame Replay

**Priority:** High
**Difficulty:** Medium

Add `isaac_rl/scripts/benchmark_real_rgbd_replay.py`.

For recorded synchronized real RGB-D frames and synthetic catalogue goals,
report:

- Predicted camera-frame velocity.
- Completion probability.
- Auxiliary position and rotation predictions.
- Frame-to-frame command smoothness.
- False completion on unfinished examples.
- Missed completion on manually labelled ready examples.

This is a preprocessing and catastrophic-domain-gap check, not a closed-loop
success benchmark.

## Phase 13: Training Sequence

**Priority:** High
**Difficulty:** Operational

1. Run a one-environment smoke test.
2. Run a five-iteration 64-environment probe.
3. Run a five-iteration 192-environment probe.
4. Run a five-iteration 256-environment probe.
5. Record VRAM and FPS after temporal depth buffers are enabled.
6. Train one nominal v7 baseline.
7. Train the combined sim-to-real profile.
8. Run periodic multi-profile validation.
9. Compare at least two random seeds.
10. Benchmark validation and test splits offline.
11. Generate representative success and failure videos.

Retain the existing reset curriculum, completion supervision, reset mixture,
and failure replay.

## Phase 14: Safe First Real-Robot Test

**Priority:** Critical for deployment
**Difficulty:** Medium

Initial safety configuration:

- 25 percent commanded speed.
- No gripper closing.
- Workspace and joint limits.
- Force/contact abort.
- Human emergency stop.
- Completion hold enabled.
- Full RGB-D and command recording.

Test order:

```text
already-ready poses
-> very close poses
-> small translation errors
-> small rotation errors
-> combined close errors
-> middle-distance starts
-> far starts
```

Do not begin real testing from random far poses.

## Documentation and Required Artifacts

Update:

- `isaac_rl/README.md`.
- `completion.md`.
- Camera and observation profile metadata.
- Euler probe/train command examples.
- Companion-wiki agent changelog.

Produce:

```text
D405 noise analysis report
v7 full goal catalogue
v7 contact sheet
randomization profile definitions
randomization preview grids
unit and integration test results
VRAM and FPS probe table
nominal and randomized benchmarks
real-frame replay report
representative success and failure videos
```

## Explicitly Deferred Methods

Do not add these before the calibrated domain-randomization baseline is tested:

- GAN or CycleGAN image translation.
- Supervised real-world action learning.
- Adversarial feature adaptation.
- Large arbitrary camera-pose changes.
- Heavy image cutout.
- Random object scaling.
- Extremely broad color randomization.

Only revisit them if measured real-world failures remain after preprocessing,
camera, depth, and timing mismatches are addressed.
