# Real D405 PPO Deployment Implementation Specification

## Status and purpose

Repository implementation status (2026-08-24):

- The lightweight deterministic policy runtime, strict goal provider,
  synchronized RGB-D intake, TF action transform, completion/safety supervisor,
  persistent step artifacts, dry-run sink, and MoveIt Servo sink are
  implemented.
- Real execution can now plan normally to the stage-2 pregrasp, use the policy
  only for pregrasp-to-grasp alignment, return on learned completion, and gate
  gripper closure before an optional MoveIt lift.
- The aligned LBR launch has an opt-in Servo node configured for speed-unit
  commands at `gripper_tcp` with shared-scene collision checking.
- Hardware acceptance, live-camera soak/replay, checkpoint parity in the
  production Isaac environment, approved cell limits, and the external
  `mv_launch` stream-profile changes remain required before real motion.

This document is an implementation handoff for running the trained multipart
RGB-D PPO visual-servo policy on the physical KUKA iiwa7/Y-gripper system with
RealSense D405 serial `260322275185` (`realsense_1`).

The camera streams described below are compatible with the implemented policy
path. Robot motion remains disabled by checked-in defaults until the remaining
acceptance tests and site-specific approvals in this document are satisfied.

This specification is narrower than `SIM2REAL_IMPLEMENTATION_PLAN.md`. That
file covers training-time randomization and sim-to-real evaluation. This file
covers the production data path from a live ROS image to a safely applied
policy command.

## Non-negotiable deployment rules

- Use only D405 serial `260322275185` with the checkpoint trained for the
  `realsense_1` camera profile. Do not silently substitute `realsense_2`.
- The policy must receive the same observation representation used in
  training: `128 x 72`, eight image channels, followed by six previous applied
  motion actions.
- The live image and goal image are each RGB-D with four channels. The eight
  image channels are ordered as `live RGBD, goal RGBD`.
- The ROS RealSense `16UC1` image is in **millimetres**. Convert it with
  `depth_m = depth_z16 * 0.001`. Do not apply the provisional sensor-internal
  `0.0001 m/unit` value from `D405WristCameraConfig` to a ROS depth image.
- Use synchronized color and aligned-to-color depth. Never combine unrelated
  latest frames merely because both are available.
- Do not rectify an `image_rect` image a second time.
- RGB may use linear interpolation during rectification. Depth must use
  nearest-neighbour rectification.
- Use deterministic policy output. Do not sample Gaussian/Bernoulli actions on
  the robot.
- The completion decision comes from the learned completion probability and
  deployment-available motion measurements. It must not use privileged
  geometric error.
- Any stale/malformed observation, stale transform, failed safety check, lost
  command consumer, or operator stop must produce a latched zero-motion hold.
- Initial real tests must not close the gripper.

## Existing sources of truth

The implementation must reuse these files rather than duplicating constants or
preprocessing:

- `grasp_planning/d405_wrist_camera.py`
  - camera serial/profile identity;
  - native `848 x 480` calibration;
  - trained intrinsics and distortion;
  - reliable depth interval;
  - training/Isaac camera-frame helpers.
- `grasp_planning/rl/d405_observation.py`
  - validity-aware RGB-D resizing;
  - invalid-depth handling;
  - metric-depth normalization;
  - live/goal RGB-D packing;
  - optional raw-image rectification.
- `isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/agents/rl_games_multipart_ppo_cfg.yaml`
  - network dimensions and PPO player configuration.
- `isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/agents/resnet_rgbd_network.py`
  - exact actor input slicing and output heads.
- `isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/agents/completion_model.py`
  - deterministic output semantics and completion probability.
- `isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env.py`
  - action slew limiting, camera-frame action convention, completion gating,
    and observation order.
- `isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env_cfg.py`
  - action scales and completion thresholds.
- `isaac_rl/scripts/rl_games/evaluate_multigrasp.py`
  - known-working RL-Games registration and checkpoint restoration sequence.
- `isaac_rl/data/plumbers_block/goal_catalog.npz`
  - canonical goal observations and target identities.

The external camera/pose producer is the `mv_launch` stack described by:

- `docs/realsense_streaming_settings.md` in `mv_launch`;
- `launch/zed_realsense_trio.launch.py` in `mv_launch`;
- the RealSense topic and QoS notes supplied with the deployment stack.

## End-to-end data and control flow

```text
RealSense D405 serial 260322275185
  -> 848x480 RGB8 at approximately 30 Hz
  -> 848x480 aligned Z16 depth at approximately 30 Hz
  -> RGB/depth timestamp synchronization
  -> timestamp gate accepts at most 15 synchronized frames per second
  -> Z16 millimetres to float metres
  -> validity-aware shared preprocessing to 128x72 RGB-D
  -> select canonical 128x72 goal RGB-D by target_id
  -> concatenate live RGBD + goal RGBD + previous applied action
  -> append zero-valued inference placeholders for privileged training labels
  -> deterministic RL-Games actor forward pass
  -> six normalized camera-frame motion means + completion probability
  -> clamp and reproduce training-time action slew limit
  -> learned completion hold/gate
  -> scale to camera-frame Cartesian twist
  -> transform camera optical twist to the command frame using live TF
  -> safety supervisor
  -> dry-run recorder or real Cartesian command adapter
```

## 1. Camera launch changes in `mv_launch`

### 1.1 Explicit stream profiles

Do not depend on librealsense defaults. Configure `realsense_1` explicitly:

```text
rgb_camera.color_profile:=848x480x30
depth_module.depth_profile:=848x480x30
rgb_camera.color_format:=RGB8
depth_module.depth_format:=Z16
align_depth.enable:=true
enable_sync:=true
```

If the installed wrapper uses legacy parameter names, map these values to the
equivalent parameters and test the resulting messages. Do not change the
requested profile silently if a profile is rejected.

### 1.2 Rectification interpolation

The two `image_proc::RectifyNode` instances must have different interpolation
parameters:

```text
color RectifyNode: interpolation = 1  # linear
depth RectifyNode: interpolation = 0  # nearest neighbour
```

Nearest-neighbour depth rectification prevents interpolation across foreground
edges, background edges, and invalid zero pixels.

### 1.3 QoS and synchronization

Keep the existing camera-info QoS override that makes the rectify nodes receive
`CameraInfo`. The policy subscriber should use sensor-data-compatible QoS,
`KEEP_LAST(1)`, and no backlog.

The policy must process an RGB/depth pair only when:

- both images have the expected shape and encoding;
- both refer to the color optical pixel grid;
- their timestamps match, or their absolute skew is below a configurable
  maximum initially set to `0.010 s`;
- the pair is newer than the last accepted pair.

Prefer exact synchronization when aligned depth and color carry identical
timestamps. Otherwise use approximate synchronization with a bounded queue.

## 2. ROS topics and message contract

The deployment node should subscribe to:

```text
/realsense_1/camera/color/image_rect
/realsense_1/camera/color/camera_info
/realsense_1/camera/aligned_depth_to_color/image_rect
/realsense_1/camera/aligned_depth_to_color/camera_info
/left/ee_pose
```

Expected types:

```text
sensor_msgs/msg/Image
sensor_msgs/msg/CameraInfo
geometry_msgs/msg/PoseStamped
```

Expected image contract:

| Field | Color | Depth |
|---|---|---|
| Size | `848 x 480` | `848 x 480` |
| Encoding | `rgb8` | `16UC1` |
| Grid | color optical | aligned to color optical |
| Rate | approximately 30 Hz | approximately 30 Hz |
| Rectification | already rectified | already rectified, nearest-neighbour |

The policy only requires `realsense_1`. It must not inherit the existing
three-camera tracker's readiness rule requiring the ZED and `realsense_2`.

### 2.1 Startup camera validation

Before enabling inference, validate and report:

- the connected serial is `260322275185`;
- actual color/depth profile and rate;
- image encodings and endianness;
- aligned depth dimensions equal color dimensions;
- both `CameraInfo` messages are present;
- aligned-depth intrinsics/pixel grid match the color stream;
- the rectified projection intrinsics agree with the trained profile within a
  configurable tolerance;
- no duplicate publishers are producing the selected image topics.

The trained native calibration is:

```text
width  = 848
height = 480
fx = 436.3104248046875
fy = 435.6492614746094
cx = 418.62664794921875
cy = 236.5121307373047
distortion model = plumb_bob
D = [-0.05201759934425354,
      0.05433472618460655,
      0.0002693705027922988,
      0.0008704775245860219,
     -0.017724450677633286]
```

For an already rectified image, compare the effective rectified projection
matrix, normally `P[:3, :3]`, to the trained pinhole intrinsics. Do not apply
the distortion coefficients again to `image_rect`.

## 3. RGB-D conversion and preprocessing

### 3.1 ROS conversion

Convert without changing channel order:

```python
rgb_uint8 = ...       # HWC, rgb8
depth_z16 = ...       # HW, uint16
rgb_float = rgb_uint8.astype(np.float32) / 255.0
depth_m = depth_z16.astype(np.float32) * 0.001
```

Zero depth remains invalid. Non-finite values, values below `0.07 m`, and
values at or beyond `0.50 m` are handled by the shared preprocessing contract.

Do not use `D405WristCameraConfig.depth_unit_m == 0.0001` for ROS images. That
value describes a provisional sensor-internal close-range setting, whereas the
RealSense ROS wrapper republishes Z16 depth in millimetres.

### 3.2 Rectification choices

There are two supported input paths. Implement one configured path at a time:

1. Recommended production path:
   - subscribe to the two `image_rect` topics;
   - skip `rectify_aligned_rgbd_numpy()`;
   - call `preprocess_aligned_rgbd_torch()` directly.
2. Diagnostic raw path:
   - subscribe to raw color and aligned raw depth;
   - call `rectify_aligned_rgbd_numpy()` exactly once;
   - then call `preprocess_aligned_rgbd_torch()`.

Reject configurations that subscribe to `image_rect` while also enabling the
repository rectification helper.

### 3.3 Policy image construction

Use `D405ObservationPreprocessCfg.from_camera(D405WristCameraConfig())` and
`preprocess_aligned_rgbd_torch()` for the live pair. The helper accepts the
native input size and produces one `128 x 72 x 4` RGB-D tensor. Do not create a
separate, slightly different deployment resize implementation.

Load the selected goal from the canonical catalogue, run it through the same
shared resize/packing contract if the catalogue stores render-resolution data,
and keep it static for the active target.

The combined image tensor is NHWC before flattening:

```text
live RGBD: [1, 72, 128, 4]
goal RGBD: [1, 72, 128, 4]
combined:  [1, 72, 128, 8]
flattened image values: 72 * 128 * 8 = 73728
```

## 4. Goal catalogue and target selection

Add a goal-catalogue provider rather than indexing the NPZ ad hoc in the ROS
callback.

Requirements:

- load `isaac_rl/data/plumbers_block/goal_catalog.npz` once at startup;
- reuse existing catalogue validation and split-selection helpers;
- expose available `target_id`, part ID, grasp ID, and split metadata;
- select exactly one explicit `target_id` for a real trial;
- reject unknown or ambiguous target IDs;
- reject a catalogue whose camera profile, observation profile, target IDs, or
  validation metadata does not match the checkpoint/run metadata;
- publish/log the selected target and goal image before enabling motion;
- require an explicit operator reset before changing targets.

Do not randomize the goal image during deployment.

## 5. Policy input and checkpoint loading

### 5.1 Exact observation layout

The actor network was built with:

```text
image values:              73728
previous-action context:       6
privileged pose target:         6
privileged completion target:   2
total RL-Games input:       73742
```

Only the first `73734` values are deployment inputs:

```text
[flattened live+goal RGB-D, previous applied six-dimensional action]
```

The last eight values are training labels. During deterministic inference,
append eight zeros and call the model with `is_train=False`. The actor must not
read real geometric target error or a simulator success flag.

### 5.2 Previous-action semantics

The six context values must be the normalized action that was actually accepted
by the deployment action filter on the preceding control step. They are not:

- the raw network output before filtering;
- the Cartesian twist after physical-unit scaling;
- the last received command when the safety supervisor replaced it with zero.

When a safety hold or terminal completion is latched, reset the action context
to zero before a new trial.

### 5.3 Checkpoint loader

Factor a deployment-safe loader from the known-working registration sequence in
`evaluate_multigrasp.py`:

- register `grasp_rgbd_resnet18`;
- register `grasp_completion_hybrid`;
- load the multipart RL-Games YAML;
- instantiate with input shape `(73742,)` and seven outputs;
- restore the selected checkpoint;
- set evaluation mode and disable gradients;
- use deterministic outputs;
- run on the configured CUDA device, with an explicit CPU fallback only for
  offline tests.

Do not start Isaac Sim merely to load the actor on the robot PC.

Add a parity test: for the same checkpoint and saved observation tensor, the
new lightweight loader must produce the same deterministic motion mean and
completion probability as `evaluate_multigrasp.py` within floating-point
tolerance.

### 5.4 Checkpoint compatibility checks

At startup, validate or require sidecar metadata for:

- network dimensions;
- camera profile identifier;
- observation profile identifier;
- goal-catalogue identity/hash;
- action scales;
- completion thresholds;
- source commit when available.

Fail closed on a mismatch. Do not merely print a warning and continue.

## 6. Interpreting policy output

Use the deterministic `mus` output from `GraspCompletionModel`:

```text
mus[0:6] = normalized camera-frame motion mean
mus[6]   = completion probability in [0, 1]
```

Do not use sampled `actions`. Clamp the six motion values to `[-1, 1]` and
reproduce the training-time normalized slew limit:

```python
delta = clamp(requested - previous_applied, -0.25, 0.25)
filtered = previous_applied + delta
```

The nominal physical scales are:

```text
linear:  0.04 m/s for normalized magnitude 1
angular: 0.24 rad/s for normalized magnitude 1
```

Apply additional first-robot-test speed limiting after these scales, initially
`25%`, in the safety supervisor. Log both the policy-requested and actually
applied commands.

## 7. Camera-frame command conversion

The first three motion values are linear velocity in the current camera optical
frame; the next three are angular velocity in that frame.

At every accepted observation timestamp:

1. obtain the current transform from camera optical frame to the controller's
   accepted command frame;
2. rotate linear and angular vectors separately;
3. preserve units and timestamps;
4. send the transformed twist to the command adapter.

Do not apply the Isaac-only 180-degree mount correction again to a live TF
transform. The real TF tree must already describe the physical camera optical
frame. The Isaac correction exists to reconcile generated Isaac robot axes
with the calibrated MoveIt mount axes.

The implementation must include an axis/sign diagnostic in dry-run mode:

- positive camera X points image-right;
- positive camera Y points image-down;
- positive camera Z points forward into the scene;
- all six signed unit commands transform in the expected robot-base direction.

## 8. TF and physical mount validation

The downloaded D405 frame reference describes:

```text
pdz_gripper_base_link
  -> camera_bottom_screw_frame
  -> camera_link
  -> camera_depth_frame
  -> camera_depth_optical_frame
```

It also states that nominal internal RealSense transforms and
driver-published internal transforms must not be enabled simultaneously.

The policy training configuration is expressed through `lbr_link_ee` and Isaac
`link7`, not directly through `pdz_gripper_base_link`. Therefore the
implementation must compare composed transforms, not raw translation triples.

Add a startup/diagnostic tool that:

- resolves the complete live transform from the active robot base or flange to
  the selected camera optical frame;
- records every frame name used in the chain;
- compares the composed physical optical pose with the calibrated training
  pose after accounting for the documented parent-frame relationship;
- reports translation error in millimetres and rotation error in degrees;
- refuses motion above configured tolerances;
- detects duplicate TF publishers for the same child frame.

Use one source for internal stream transforms:

- offline/RViz: nominal RealSense description; or
- real driver: calibrated driver-published stream transforms.

## 9. Real command adapter

The camera stack supplies observations and flange poses but does not by itself
provide the robot command interface. Implement an explicit adapter boundary:

```text
VisualServoCommandSink
  - DryRunCommandSink
  - RealCartesianServoCommandSink
```

The real implementation should use the supported hardware Cartesian servo
interface, preferably MoveIt Servo or the already approved LBR Cartesian
controller. Do not introduce an unreviewed direct joint-velocity controller.

The adapter must provide:

- command frame and timestamp validation;
- zero-command/hold;
- watchdog timeout;
- controller-enabled/healthy feedback;
- acknowledgement that a command consumer exists;
- joint, velocity, acceleration, and workspace limits;
- clean activation and deactivation;
- a synchronous emergency-stop path independent of policy inference.

The first implementation may stop at `DryRunCommandSink`, but real motion must
remain impossible until the real sink and its safety checks are explicitly
selected.

## 10. Completion state machine

Reproduce deployment-available training behavior:

```text
candidate threshold:       probability >= 0.95
required stable frames:    4 consecutive policy steps
maximum linear speed:      0.005 m/s
maximum angular speed:     0.03 rad/s
```

Behavior:

1. As soon as probability reaches `0.95`, command zero motion for that frame.
2. Estimate or read current TCP linear and angular speed.
3. Increment the streak only while probability and both speed gates pass.
4. Reset the streak when any gate fails.
5. After four valid frames, enter a latched `COMPLETED_HOLD` state.
6. Remain stopped until an explicit operator reset/new-trial command.

The speed measurement must come from the robot or timestamped flange motion,
not privileged target error. If speed is estimated from `/left/ee_pose`, use a
filtered, timestamp-aware estimate and reject stale/non-monotonic poses.

The state machine should expose at least:

```text
DISARMED
READY
RUNNING
CANDIDATE_HOLD
COMPLETED_HOLD
SAFETY_HOLD
FAULT
```

## 11. Safety supervisor

Safety must be outside the neural network and must override every policy
command.

Required checks:

- operator arm/start and independent emergency stop;
- command deadman/watchdog;
- maximum RGB/depth timestamp age;
- maximum RGB/depth skew;
- maximum flange-pose/TF age;
- minimum valid-depth fraction;
- finite observation and finite model output;
- output and slew bounds;
- Cartesian linear/angular speed bounds;
- configured Cartesian workspace bounds;
- joint position/velocity/acceleration limits;
- controller health and command-subscriber presence;
- force/contact abort when the available robot interface exposes it;
- maximum trial duration;
- explicit rejection of gripper-close commands during initial testing.

Any violation commands zero immediately and latches `SAFETY_HOLD` or `FAULT`.
Recovery requires an explicit operator action; fresh data alone must not resume
motion.

## 12. Runtime configuration and launch

Add a checked-in YAML configuration, for example:

```text
configs/visual_servo_real_d405.yaml
```

It should contain:

- checkpoint path;
- goal catalogue path and target ID;
- expected camera/observation profile identifiers;
- camera serial;
- all image, info, pose, TF, and command topics;
- rectified/raw input mode;
- image synchronization tolerance;
- control rate and data-age limits;
- model device;
- action scales and first-test speed fraction;
- completion thresholds;
- workspace and joint safety limits;
- dry-run versus real command sink;
- output/log directory;
- operator-control service/topic names.

Add one user-facing launcher, for example:

```text
scripts/run_d405_ppo_visual_servo.py
```

Default behavior must be dry-run, no gripper close, and zero motion. Real motion
must require an explicit configuration setting and command-line confirmation.

## 13. Diagnostics and persistent artifacts

Every trial must create a persistent run directory containing:

```text
configuration snapshot
checkpoint identity/hash
git commit and dirty status
camera and observation profile
selected target metadata and goal preview
live CameraInfo and TF-chain snapshot
per-step timestamps and synchronization skew
completion probability and state-machine state
raw policy motion mean
filtered normalized action
scaled/transformed requested twist
actually applied or suppressed command
safety events and hold reason
RGB/depth validity statistics
optional synchronized RGB/depth recordings or rosbag
```

Add a live diagnostic display or low-rate preview showing:

- rectified live RGB;
- normalized/colored live depth and invalid mask;
- selected goal RGB/depth;
- completion probability and streak;
- requested/applied command;
- current runtime state and safety reason.

Diagnostic rendering must not block the 15 Hz policy/command loop. Camera
receipt and freshness monitoring remain at the native approximately 30 Hz
stream rate.

## 14. Tests required before hardware motion

### 14.1 Unit tests

- `16UC1` value `250` becomes `0.250 m`.
- RGB remains RGB rather than being converted to BGR.
- invalid/near/far depth follows the shared validity rules.
- native `848 x 480` input produces `1 x 72 x 128 x 4`.
- live+goal produces exactly 73,728 flattened image values.
- full inference observation has exactly 73,742 values.
- privileged inference placeholders cannot influence deterministic output.
- action slew limiting matches `isaac_rl_env.py`.
- completion state transitions match the simulator implementation.
- stale data and malformed tensors always produce zero command.
- terminal and safety holds are latched.

### 14.2 Checkpoint parity test

Store one deterministic observation fixture and compare:

```text
Isaac evaluation player output
vs.
lightweight deployment loader output
```

Compare all six motion means and completion probability.

### 14.3 ROS integration tests

Using recorded or synthetic publishers:

- validate QoS compatibility;
- validate exact/approximate synchronization;
- drop and delay individual topics;
- inject wrong encodings, dimensions, frame IDs, and intrinsics;
- inject stale pose/TF;
- verify every fault holds zero and is reported;
- verify the node never requires ZED or `realsense_2` readiness.

### 14.4 Live-camera dry-run

With the robot command sink disabled:

- run for at least ten minutes;
- confirm approximately 30 Hz synchronized camera receipt and 15 Hz accepted
  policy observations;
- measure end-to-end image-to-output latency and jitter;
- confirm no growing message queue or memory use;
- confirm the observed depth range and invalid fraction are plausible;
- inspect live/goal previews for channel order, orientation, FOV, and scale;
- exercise camera disconnect/reconnect and verify latched hold behavior.

### 14.5 Offline real-frame replay

Record synchronized real frames from already-ready, near, and clearly
unfinished poses. Report:

- motion means and completion probability;
- temporal command smoothness;
- false completion on unfinished frames;
- missed completion on ready frames;
- preprocessing and inference latency.

This is a domain-gap and catastrophic-output check, not proof of closed-loop
success.

## 15. Staged hardware acceptance

Real motion may be enabled only in this order:

1. dry-run with live camera and stationary robot;
2. dry-run while the robot is moved manually/under an existing safe controller;
3. 25% policy speed, no gripper close, already-ready pose;
4. very close translation-only offsets;
5. very close rotation-only offsets;
6. combined close offsets;
7. middle-distance starts;
8. far starts only after close/middle trials meet agreed success and safety
   thresholds.

Every stage requires an emergency-stop operator, workspace limits, command
watchdog, and persistent recording.

## 16. Suggested implementation layout

The branch may adjust names to match local conventions, but responsibilities
should remain separated:

```text
grasp_planning/ros2/d405_rgbd_subscriber.py
    synchronized image/camera-info intake and validation

grasp_planning/rl/d405_policy_runtime.py
    goal loading, observation assembly, checkpoint loading, deterministic actor

grasp_planning/ros2/visual_servo_safety.py
    state machine, completion gate, data watchdogs, safety overrides

grasp_planning/ros2/visual_servo_command_sink.py
    dry-run and real Cartesian command adapters

scripts/check_d405_policy_tf.py
    composed TF/camera-pose validation and signed-axis diagnostics

scripts/run_d405_ppo_visual_servo.py
    single user-facing deployment entrypoint

configs/visual_servo_real_d405.yaml
    explicit deployment configuration, safe by default

tests/test_d405_policy_runtime.py
tests/test_visual_servo_safety.py
tests/test_d405_ros_contract.py
    unit, parity, and ROS contract coverage
```

Avoid putting model loading, image synchronization, TF lookup, controller
publishing, and safety state in one ROS callback.

## 17. Definition of done

The implementation is complete only when all of the following are true:

- `mv_launch` explicitly produces `848 x 480 x 30` RGB8/Z16 streams for serial
  `260322275185`.
- Aligned depth is on the color grid and depth rectification uses nearest
  neighbour.
- ROS Z16 is demonstrably converted from millimetres to metres exactly once.
- The live preprocessing path uses the repository shared helper and produces
  the exact trained tensor layout.
- The goal target is explicit and catalogue/checkpoint compatibility is
  enforced.
- A lightweight deterministic checkpoint loader matches Isaac evaluation
  output on a saved fixture.
- No privileged target error enters the actor or completion state machine.
- Camera-frame actions are transformed using a verified live optical TF.
- Completion behavior matches the probability, low-speed, and consecutive-step
  contract.
- Dry-run is the default and all data/TF/controller faults command a latched
  hold.
- Ten-minute live-camera dry-run and offline replay pass.
- Staged robot tests are recorded and reviewed before increasing speed or
  enabling gripper close.

## Known external decisions still required

The implementation branch must resolve these from the robot stack rather than
guessing:

- the supported Cartesian servo command interface and topic/service names;
- the authoritative measured TCP speed source;
- the operator arm/reset/deadman interface;
- the available force/contact signal and abort mechanism;
- approved Cartesian workspace and joint limits;
- the exact relationship between `pdz_gripper_base_link`, `lbr_link_ee`, and
  the real driver optical frames;
- where the production checkpoint and trial artifacts are stored.

These decisions may change adapter configuration, but they do not change the
camera, preprocessing, model-input, or learned-completion contracts above.
