# Mixed-renderer Franka reference bank

The mixed bank was inspected and training was authorized on September 21.
The builder/preview workflow below itself performs no optimization. The separate
Euler launch is referenced by `configs/franka_mixed_next_run.json`;
consult its verification status rather than treating submission as training success.
Existing catalogs/checkpoints remain intact.

## Training configuration

`configs/franka_mixed_next_run.json` points to the new catalog. The training task
reads its self-describing contract; do not resume an old checkpoint silently.
The old architecture and 15 Hz policy / 120 Hz physics are retained.

The replacement `franka_clutter_v7_zed_calibrated` bank uses the recorded ZED Mini
SN13829658 rectified LEFT calibration: 672×376, fx=fy=338.6001587,
cx=332.0467834, cy=184.4311218, baseline=0.06286765 m. Its field of view is
89.55°×58.08°, replacing the provisional 76.22°×47.39° optics. All 7,960 images
were regenerated in both renderers; the live Isaac camera uses the same profile.
The profile is embedded as `camera_profile_data` and checked against its digest
by training and deployment. An explicit mismatched override is rejected.
Old catalogs still resolve the original profile, so existing checkpoint replay
does not silently change. The grasp, reset, mount, TCP, and physics data remain
unchanged; this is not a new extrinsic calibration.

The original pending jobs 14784992/14785066 were cancelled without executing.
Replacement launch evidence lives under `artifacts/franka_calibrated_training_20260921`.
The comparison with the user's recorded video is under
`artifacts/franka_zed_calibrated_20260921/recording/index.html`.

At each episode reset the reference is selected independently of live object
placement and live appearance:

- 20% refreshed canonical blue Isaac reference;
- 30% independently colored Isaac references;
- 50% independently colored MuJoCo references.

Each target has four additional cached RGB-D images (two from each renderer),
with four distinct palette colors. Live color is not synchronized with them.
The selected goal stays constant during the episode. Variants stay on CPU as
float16; only selected reset images are transferred to GPU as float32.

MuJoCo uses the actual Isaac visual triangles, current PhysX link/object poses,
calibrated camera transform and finger aperture. It does not run another physics
controller or solve another IK problem. Its lab uses base colors from the same
materials, with textures/decals simplified and hard shadows disabled. Isaac live
scenes retain the pencil table and floor configuration. Both renderers use
optical-Z metres and the shared 128x72 packing and validity convention.

The builder compares near-surface interior depth between renderers, failing if
median disagreement exceeds 5 mm. Boundary pixels and distant background are
excluded from this diagnostic, so it is not a universal per-pixel error bound.

## Verified depth preprocessing correction

The pinned CUDA/PyTorch runtime reproduced cross-environment contamination after
`grid_sample` returned NHWC depth with strides `(36864, 256, 1, 36864)`.
Three constant-depth inputs at 0.2, 0.5 and 0.8 m were packed as 0.2, 0.2 and
0.2 m by the legacy adaptive-area path. Canonicalizing the singleton-channel
stride with squeeze/unsqueeze preserves 0.2, 0.5 and 0.8 m independently, without
an extra full-image copy. `contiguous()` alone can be a no-op for that layout.
This is a preprocessing/runtime-layout defect, not established evidence of an
Isaac annotator bug or real-camera depth bias.

New catalogs explicitly declare `rgbd_packing: dense_depth_channel_v2` and
`depth_source: radial_to_optical_z_v1`. Goal RGB-D is regenerated, including the
canonical blue reference. Radial depth is converted using each tile's intrinsics,
then the shared optical-Z range/validity packing is applied. Old catalog rendering
keeps its explicit legacy layout for reproducible old-checkpoint diagnostics;
new training must use the new catalog. Hardware is already single-camera optical
Z and does not require an empirical depth-value offset.

The mixed bank is digest-checked. Loading its contract into an old checkpoint
is not permitted as an ordinary resume. The old policy's recorded successes do
not establish correct independent depth observations across its training batch.
The defect is a plausible transfer problem, not proof of the cause of every
previous failure.

## Build and inspect

Inside the task's Isaac runtime, with `mujoco==3.3.7` and `MUJOCO_GL=egl`:

```bash
bash scripts/franka_isaac_python.sh isaac_rl/scripts/build_franka_mixed_goals.py \
  --output isaac_rl/data/franka_clutter_v6_mixed/catalog.npz \
  --gallery artifacts/franka_mixed_goals_20260921/full --headless --device cuda:0
python3 scripts/report_franka_mixed_goals.py artifacts/franka_mixed_goals_20260921/full
bash scripts/franka_isaac_python.sh isaac_rl/scripts/check_franka_mixed_goals.py \
  --catalog isaac_rl/data/franka_clutter_v6_mixed/catalog.npz \
  --output artifacts/franka_mixed_goals_20260921/smoke.json --headless --device cuda:0
```

The builder supports `--parts ... --limit-per-part 2` for small previews. It writes
a separate catalog atomically only after all render checks pass. Rendering is an
offline dependency; training/deployment consume cached images and do not need
MuJoCo running alongside Isaac or the robot.

For future deployment, a selection can explicitly set `goal_variant_index` to
0 (canonical blue), 1–2 (Isaac) or 3–4 (MuJoCo). No random reference changes occur
inside the hardware policy loop. Color-render overrides and nonzero cached
variant selection are mutually exclusive. The active hardware selection is
not changed by creating this bank.

## Deployment evidence and remaining preparation

`scripts/audit_franka_deployment_records.py` reads existing JSON sessions without
connecting to the robot or camera. Its output distinguishes validity coverage
from depth accuracy, commanded actions from measured motion, and raw camera
pixels from cropped/resized policy inputs.

Recorded evidence includes calibrated ZED Mini intrinsics, policy timing, depth
coverage, and some TCP/controller feedback. The original run logger does not
save RGB-D sequences. One older RGB-D snapshot is available. This permits spatial
inspection of holes but not calibration of temporal depth noise, stable hole
persistence, or absolute depth bias. Those require synchronized raw frames and
known surfaces/poses. Do not fit a generic depth correction to coverage alone.

Before another long run/deployment:

1. Verify the real workspace in robot-base coordinates. The latest recorded TCP
   is around Y=-0.50 m; the placement bank samples objects at Y=-0.185..+0.185 m.
   TCP is not object pose, but the frame/coverage discrepancy needs resolving.
2. Confirm hand-to-camera rotation/translation and TCP/aperture. Training uses a
   Panda arm; hardware is FR3 plus Panda hand, with FR3 kinematics in Servo.
3. Separate controller versions when analyzing tracking. Earlier sessions barely
   moved despite commands; later feedback shows real displacement. Do not fit
   one latency/tracking model to all sessions indiscriminately.
4. Record synchronized rectified RGB, raw optical-Z depth, confidence, camera
   timestamps and robot feedback, including stationary and moving sequences.
5. Use a known plane at several distances to measure depth bias; use repeated
   stationary frames for noise and spatial/temporal hole statistics. Preserve
   structured missing-depth augmentation rather than replacing it with IID holes.
6. Compare the next policy against fixed-blue and original-placement baselines
   on held-out placements and hardware shadow-inference examples. Real logged
   rollouts without success labels are diagnostics, not demonstrations of correct
   actions or verified alignment.
