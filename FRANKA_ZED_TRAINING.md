# Panda + ZED Mini learning pilot

For the implemented Fabrica bundle adapter, physical lift filtering and
persistent training launch, see [FRANKA_FABRICA_TRAINING.md](FRANKA_FABRICA_TRAINING.md).
The cuboid workflow below remains a separate smoke fixture. Older schema-2
catalogs need rebuilding after the addition of arm-link contact monitoring.

`Grasp-Franka-ZEDMini-RGBD-Direct-v0` is a GPU visual-servo alignment task on
the plain white table. It uses the Panda arm and Panda hand from the supplied
unplug scene, the exact supplied wrist-camera pose, and a separate ZED Mini
profile. It reuses the existing ResNet RGB-D actor and hybrid motion/completion
PPO. This pilot aligns an open hand to a fixed cuboid; it does not yet learn or
validate physical grasp closure and lifting.

## Run locally

From this repository, use Isaac Lab 2.3.2 / Isaac Sim 5.1:

```bash
ISAAC_LAB=/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh

# Generate the 12 pilot goals, approach resets and goal RGB-D images.
"$ISAAC_LAB" -p isaac_rl/scripts/prepare_franka_zed_catalog.py --headless

# Bounded physics/controller/reset check; no learning.
"$ISAAC_LAB" -p isaac_rl/scripts/check_franka_zed.py --headless

# Construct the policy and check inference; no optimizer steps.
ISAAC_RL_DISABLE_CUDNN=1 "$ISAAC_LAB" -p isaac_rl/scripts/train_franka_zed.py \
  --headless --num-envs 8 --dry-run

# Explicitly starts training when you choose to run it.
ISAAC_RL_DISABLE_CUDNN=1 "$ISAAC_LAB" -p isaac_rl/scripts/train_franka_zed.py \
  --headless --num-envs 16 --iterations 2000
```

The cuDNN override uses the existing project's local-driver workaround.
The normal ImageNet ResNet18 weights must be cached or downloadable.
`--no-pretrained` is only for an offline model smoke: shared early ResNet layers
are frozen, so random frozen features are unsuitable as the intended baseline.
An iteration count is a run limit, not a convergence guarantee.

The same scripts run inside `isaac-lab-euler:2.3.2` using the Docker command in
[the scene guide](FRANKA_TRAINING_SCENE.md), substituting the script and flags.
Mount the project at `/workspace/project` and optionally mount the host
`~/.cache/torch/hub/checkpoints` at `/root/.cache/torch/hub/checkpoints` read-only.
No Euler task wiring or jobs were added for this new task.

## Camera contract

`configs/franka_zed_mini.json` is **provisional**, based on the source camera's
3.06 focal length / 4.8 horizontal aperture: 672x376, fx=fy=428.4, centered
principal point. Simulation renders 256x144 and supplies 128x72 RGB-D to the
actor. The mount remains `panda_hand` to ROS optical frame, position
(-0.1115, 0.0481, 0.0151) m, WXYZ quaternion
(0.6435702, -0.2768679, 0.2724621, -0.6594891).

The initial usable depth interval is 0.1–1.0 m. This is a working training
choice, not a measured reliable range. RGB and optical-Z metric depth must be
aligned to rectified LEFT. Invalid/out-of-range depth is excluded during area
resizing and represented by normalized depth 1. The renderer's centered,
square-pixel image is reprojected to the profile intrinsics before resizing;
pixels outside its view become invalid. No D405-specific noise is applied.
Only simple live RGB gain randomization is currently enabled. Simulation does
not reproduce the ZED stereo matcher, its failure modes, latency or occlusions.

On the computer with the ZED SDK and connected Mini:

```bash
python3 scripts/export_zed_mini_profile.py \
  --output configs/franka_zed_mini_measured.json --resolution VGA
```

Use `--camera-profile configs/franka_zed_mini_measured.json` with **both** the
builder and trainer/checker, rebuilding the catalog first. The exporter checks
camera identity and retains the supplied mount/TCP for later physical checking.
It uses SDK rectified LEFT calibration, because the saved `.conf` values are
raw calibration ([Stereolabs calibration documentation](https://www.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration)).
The local `SN33137761.conf` has a 120.08 mm baseline and was not used as Mini
calibration. The provisional Mini baseline is 63 mm.

Catalogs contain an explicit robot/scene/camera/processing/controller contract.
Changing the camera profile requires new goal images. Checkpoint resumes
require a matching `.contract.json` sidecar, written next to checkpoints after
a normal training exit. Existing KUKA/D405 checkpoints are not accepted as a
matching resume. The task does not silently reinterpret old PDZ grasp labels.

## Controller and learning setup

- Physics 120 Hz; policy 15 Hz. Six camera-frame velocity actions plus one
  completion action; nominal per-axis limits 0.04 m/s and 0.24 rad/s.
- Damped least-squares IK includes the physical TCP offset of 0.1034 m along
  hand Z. Joint speed is clipped to 1 rad/s. Each physics step writes both a
  short position target and velocity feedforward to the implicit PhysX PD
  drives, avoiding the slow response of tiny position increments alone.
- Panda high-PD arm gains: stiffness 400, damping 80; torque caps 87 Nm on
  joints 1–4 and 12 Nm on 5–7. Finger runtime gains are 2000/100, cap 200 N.
  Arm gravity is disabled, approximating compensation. These are simulation
  baseline parameters, not identified real Franka dynamics or grip-force tuning.
- Inputs: live and goal RGB-D plus previous six actions. Privileged pose and
  completion labels occupy the existing training-only observation slots;
  the shared actor separates them from its action inputs. Critic state has
  joints, velocities, pose errors and previous actions.
- Reward combines pose-potential improvement, small time/action costs and
  graded completion reward. Success requires declared completion within 4 mm
  and 3 degrees without detected unsafe hand contact. Hand/finger contact at
  3 N, large divergence, and 8-second timeout terminate episodes.
- Resets use only sampled IK/contact-checked approach waypoints; 15% start at
  the goal for completion supervision. The default 12-target cuboid catalog
  has 10 training, one validation and one test target. This is a tiny readiness
  dataset, not a meaningful generalization benchmark.

## Custom object starting point

The builder accepts `--goal-spec path/to/goals.json`, with one collision-enabled
object USD and at least three targets. `object_pose` and `goal_pose` are
environment-local XYZ + WXYZ; `goal_pose` describes the hand orientation and
TCP position. The USD must already use the intended object-local frame.

```json
{
  "object_usd_path": "object.usd",
  "gripper_open_width_m": 0.06,
  "targets": [
    {"object_pose": [0.43, 0.0, 0.02, 1, 0, 0, 0],
     "goal_pose": [0.43, 0.0, 0.02, 0, 1, 0, 0]}
  ]
}
```

The example shows the record format; supply at least three actual targets.
The default trainer supports the same object via `--object-usd` with the exact
catalog path/contract and matching `--gripper-open-width` if changed.
The implemented Fabrica adapter uses the saved stage-2 bundles and their
existing frame and grasp semantics directly. Its first validated custom mesh
is `plumbers_block/0`; see the linked Fabrica guide for that complete workflow.

## Verified and still missing

On RTX 4090 in the pinned container, catalog generation accepted all 12 goals
and all 12 sampled waypoints per goal. Goal images visibly contain the blue
part. These are discrete GPU IK and hand-contact checks; neither continuous
swept collision checking nor whole-arm collision preflight is established.

The paired 12-episode test started at mean 65.4 mm error. Zero control timed
out in every episode; the privileged reference controller completed all 12
with mean/median final position error 3.21/3.09 mm and mean rotation error
0.146 degrees, without detected hand collisions. Exact-goal resets had less
than 1 micrometre position error. Mean goal/live RGB-D difference was 0.00694.
See `artifacts/franka_zed/check.json` for per-episode results. These results
validate the simulation action path, not learned-policy performance.

Four CPU regression tests cover depth validity, calibrated image/depth
alignment, TCP Jacobian offset sign, and rejection of a non-Mini baseline.
Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_zed_mini.py -q`;
an unrelated auto-loaded host pytest plugin otherwise interrupts collection.

The shared policy/player also passed a GPU forward-inference dry run on eight
environments, producing finite seven-dimensional actions with zero optimizer
steps. Evidence: `logs/franka_zed_dry_run/20260908_153824/dry_run.json`.

After explicit user authorization, the two-epoch PPO smoke passed on eight
GPU environments (1,024 transitions). A first run exposed unflushed TensorBoard
events on fast shutdown; the Franka trainer now closes its writer before Isaac
shutdown. The repeat passed with finite actor/critic/auxiliary losses, finite
checkpoint tensors and a matching camera/task sidecar. Evidence:
`logs/franka_zed_smoke/20260908_155029/smoke_verification.json`; checkpoint:
`logs/franka_zed_smoke/20260908_155029/nn/franka_zed.pth`.
AMP reduced its initial scale to 8192; the saved actor optimizer records one
completed update during this very short run. This is an optimizer plumbing
check, not evidence of policy convergence. No full training or cluster job ran.

Still needed before claiming the full learning migration:

1. Confirm whether the physical arm is Panda or **FR3 with Panda hand**. The
   latter needs its actual articulation, TCP and matching robot-valid catalogs.
2. Measure rectified LEFT intrinsics at the active resolution, hand-to-camera
   and hand-to-TCP transforms, table/mount dimensions, depth validity and latency.
3. Validate real controller response and force/velocity limits; identify useful
   actuator/noise/delay randomization. Inspector settling is not this GPU path.
4. Expand the implemented single-part Fabrica adapter to representative parts
   and placements, with adequate held-out-object coverage and global planning.
5. Extend the implemented offline dynamic closure/lift checks to imperfect
   policy stops and online lift rewards if required. During alignment training,
   the task holds the object kinematic and keeps the gripper open.
6. Run full training and held-out learned-policy evaluation after the local
   optimizer smoke. Two iterations establish plumbing readiness, not policy
   convergence or generalization.
7. Wire and verify a real ZED inference/control bridge and the new Euler launch
   path before deployment. Existing hardware execution gates remain in place.
