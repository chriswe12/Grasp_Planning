# Real-Robot Deployment Requirements

This guide describes what must be installed or copied to a new computer to run
the consolidated pipeline on the KUKA iiwa7 cell. It covers deterministic
MoveIt grasping and optional D405 policy-assisted approach. Training is out of
scope.

The supported operator boundary is:

```bash
source ./setup_robot_env.sh
./run_pipeline.sh --help
```

## Deployment Matrix

| Requirement | Blind MoveIt grasp | Policy approach |
| --- | --- | --- |
| Tracked repository and robot/part assets | required | required |
| ROS 2 Humble, MoveIt 2 and ros2_control | required | required |
| KUKA FRI 1.15 LBR workspace | required | required |
| Repository ROS overlay and `fp_debug_msgs` | required | required |
| External `servo_gripper` controller | required unless explicitly skipped before grasp | required |
| Perception `DebugPoseItem` stream | required | required |
| Base Python dependencies | required | required |
| PyTorch, torchvision and Pillow | not required | required |
| NVIDIA GPU and compatible CUDA-enabled PyTorch | not required | required for the reviewed 15 Hz path |
| Compressed RGB-D and camera-to-robot TF | not required | required |
| MuJoCo Filament runtime goal renderer | not required | required |
| Selected checkpoint and deployment sidecar | not required | required |
| Isaac Sim, Isaac Lab, `rl_games`, TensorBoard | not required | not required |

## Supported Host Baseline

The current hardware path targets:

- Ubuntu 22.04 x86-64;
- ROS 2 Humble;
- Python 3.10 or newer;
- Fast DDS through `rmw_fastrtps_cpp` and UDPv4;
- KUKA Sunrise FRI client SDK 1.15; and
- the repository's PDZ gripper model and `pdz_gripper_tcp` by default.

Install ROS and native build/collision dependencies. Use `rosdep` afterward to
resolve the complete package manifests:

```bash
sudo apt install \
  ros-humble-desktop \
  ros-humble-moveit \
  ros-humble-moveit-servo \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-rmw-fastrtps-cpp \
  ros-dev-tools \
  python3-pip \
  python3-venv \
  libccd-dev \
  libfcl-dev
```

`pick_ik` is optional. The launchers use KDL by default; install
`ros-humble-pick-ik` only when selecting `--ik-solver pick_ik`.

## Clone And Python Environment

```bash
git clone --recurse-submodules \
  https://github.com/chriswe12/Grasp_Planning.git
cd Grasp_Planning

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

`--system-site-packages` keeps the ROS-installed `rclpy` and message packages
visible to the virtual environment.

For blind MoveIt grasping:

```bash
python3 -m pip install -e .
```

For policy-assisted grasping:

```bash
python3 -m pip install -e '.[deployment]'
```

The deployment extra adds Pillow, PyTorch and torchvision. Do not install the
`training` extra on the robot computer. The public entrypoint uses
`PIPELINE_PYTHON`, then `python3`, then `python`; pin it to the virtual
environment for predictable operation:

```bash
export PIPELINE_PYTHON="$PWD/.venv/bin/python"
```

## KUKA LBR Underlay

The inspected hardware workspace uses these sources:

| Repository | Revision |
| --- | --- |
| `lbr-stack/lbr_fri_ros2_stack` | `8128459f0cbb5d3cce8f68d942c086292c9a3ecc` |
| `lbr-stack/fri` | `dc7d613d42571e458dd62ea814a0a47a209487d8` |
| `lbr-stack/lbr_fri_idl` | `e422bff2b3d70ec47853145b35645980df623a60` |

Bootstrap the 1.15 workspace using the upstream manifest, then check out the
revisions above if exact reproduction matters:

```bash
source /opt/ros/humble/setup.bash
mkdir -p /path/to/lbr-stack/src
cd /path/to/lbr-stack

vcs import src --input \
  https://raw.githubusercontent.com/lbr-stack/lbr_fri_ros2_stack/humble/lbr_fri_ros2_stack/repos-fri-1.15.yaml
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

The Grasp Planning repository owns the PDZ URDF/SRDF, controller YAML, Servo
YAML and per-arm FRI endpoint configurations. The inspected legacy LBR source
checkout still has old local description edits, so a clean-upstream physical
bringup must be validated before treating the hashes alone as a complete cell
qualification.

## Repository ROS Overlay

The repository pins `fp_debug_msgs` in `ros2_ws/dependencies.repos`.

```bash
cd /path/to/Grasp_Planning
bash scripts/download_ros2_dependencies.sh

source /opt/ros/humble/setup.bash
source /path/to/lbr-stack/install/setup.bash

cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build \
  --packages-select fp_debug_msgs robot_integration_ros \
  --symlink-install
cd ..
```

Set the non-default underlay path before sourcing the common environment:

```bash
export LBR_WORKSPACE=/path/to/lbr-stack
source ./setup_robot_env.sh
```

## Required External ROS Systems

### Robot and MoveIt

The launchers require the LBR arm joint state, controller manager, MoveIt IK
and planning services, and trajectory execution action. Policy execution also
starts collision-checking MoveIt Servo and the estimated force-torque
broadcaster.

Cell defaults are:

| Side | Robot | FRI peer | UDP port |
| --- | --- | --- | --- |
| left | `lbr_one` | `192.170.10.2` | `30200` |
| right | `lbr_two` | `192.170.20.2` | `30201` |

The SmartPAD FRI application must use a 10 ms send period and position client
command mode. Network addresses and safety limits are site configuration, not
portable defaults.

### Grippers

The external `servo_gripper` ROS package is not a submodule of this repository
and must be installed separately on the gripper computer. The pipeline expects:

```text
/left/gripper_controller/calibrate
/left/gripper_controller/open
/left/gripper_controller/close
/left/gripper_controller/stop
/left/gripper_controller/position_command
/left/gripper_controller/position

/right/gripper_controller/calibrate
/right/gripper_controller/open
/right/gripper_controller/close
/right/gripper_controller/stop
/right/gripper_controller/position_command
/right/gripper_controller/position
```

The service type is `std_srvs/srv/Trigger`; command and feedback topics use
`std_msgs/msg/Float64`. Normalized position is closure fraction (`0.0` fully
open, `1.0` fully closed), mapped to the calibrated physical 74-7 mm stroke.
Known USB adapter identities are `5B3D047592` for left and `5B3D044069` for
right. Calibrate with empty jaws; the grasp pipeline deliberately does not
calibrate automatically.

### Perception

Real execution requires:

```text
/perception/fp/pose_base/fused/assembly
  fp_debug_msgs/msg/DebugPoseItem
```

The selected assembly and numeric part ID must match the pipeline request. The
publisher may run on another host, but every host must use the same
`ROS_DOMAIN_ID`, compatible DDS settings, multicast-capable networking and
synchronized clocks.

## Dual Planning Artifacts

The default dual real path consumes the offline holder/inserter catalog under:

```text
artifacts/dual_grasp_planning/plumbers_block/
```

`artifacts/` is ignored by Git. Copy a validated catalog or regenerate it:

```bash
python3 scripts/build_assembly_sequence.py --assembly plumbers_block
python3 scripts/build_holder_grasp_library.py \
  --config configs/dual_grasp_planning.yaml
python3 scripts/build_holder_state_feasibility.py \
  --config configs/dual_grasp_planning.yaml
python3 scripts/build_dual_grasp_pairs.py \
  --config configs/dual_grasp_planning.yaml
```

The single-object pipeline generates its stage-1/stage-2 grasp bundle from the
live perceived part and does not require this dual catalog.

## Policy Checkpoints

Policy weights are intentionally ignored by Git. Copy the selected `.pth` and
its hash-bearing `.deployment.json` sidecar into:

```text
.cache/d405_policy_deployment/
```

For example:

```text
pdz-velocity-rotation-v1.pth
pdz-velocity-rotation-v1.deployment.json
```

The registry validates the checkpoint hash, observation context, camera
profile and gripper embodiment before motion. The old stored goal catalog is
not required by current PDZ deployment: the selected live grasp goal is
rendered on demand.

## Compressed RGB-D Contract

Policy deployment subscribes directly to `sensor_msgs/msg/CompressedImage`:

```text
/<camera>/camera/color/image_rect/compressed
/<camera>/camera/aligned_depth_to_color/image_rect/compressedDepth
```

- Color must decode as JPEG or PNG RGB.
- Depth must be lossless `16UC1; compressedDepth` PNG in millimetres.
- RGB and aligned depth dimensions must match.
- Both messages need non-empty frame IDs and acceptable timestamp skew.
- CameraInfo is diagnostic-only; camera serial and factory calibration do not
  gate execution.
- A live TF connection from the camera optical frame to the robot command frame
  remains mandatory in normal deployment.

The camera host normally needs the RealSense ROS driver and compressed image
transport publisher plugins. Cross-host testing also needs clock
synchronization; disabling publisher-header age does not disable the local
frame-receipt watchdog.

## MuJoCo Filament Goal Renderer

Policy deployment uses MuJoCo only to render the RGB-D goal for the exact grasp
accepted by MoveIt. Isaac Sim and Isaac Lab are not started.

Required software:

- the Python `mujoco` package installed by this project;
- an ABI-compatible experimental `libmujoco_filament.so`;
- the matching Filament asset directory;
- Vulkan loader support; and
- a working Vulkan ICD. The wrapper defaults to Mesa Lavapipe at
  `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`.

Point the launcher at a persistent installation rather than its development
defaults under `/tmp`:

```bash
export MUJOCO_FILAMENT_LIBRARY=/opt/mujoco-filament/lib/libmujoco_filament.so
export MUJOCO_FILAMENT_ASSETS_DIR=/opt/mujoco-filament/assets
```

The repository does not currently build or download the experimental Filament
library. A new deployment must copy a compatible validated build or add an
external build step. `scripts/run_mujoco_filament.sh` verifies both paths before
starting the renderer.

### Geometry Inputs

The runtime renderer consumes:

- the tracked PDZ robot URDF at
  `assets/urdf/kuka_iiwa7_pdz_gripper/urdf/kuka_iiwa7_pdz_gripper.urdf`;
- tracked black-finger and white-pad visual meshes under
  `assets/urdf/kuka_iiwa7_pdz_gripper/meshes/visual/`;
- the selected part mesh rebuilt from the stage-2 bundle into a temporary STL;
- the exact MoveIt joint state and selected jaw approach width;
- the configured D405 camera profile and calibrated wrist transform; and
- a procedural 0.65 x 0.60 m small-pitch T-slot surface with 25.5 mm pitch and
  20.5 mm aluminum lands.

No external texture image files are required. Materials are scalar PBR values
authored into the temporary MJCF.

### Canonical PDZ Render Materials

These values are defined once in
`grasp_planning/rl/goal_catalog_profiles.py` and are shared by training goal
capture and on-demand deployment rendering:

| Render material | Physical/visual role | RGB | Metallic | Roughness | Emission |
| --- | --- | --- | ---: | ---: | ---: |
| `part_canonical` | muted brown printed part | `(0.372549, 0.282353, 0.227451)` | 0.00 | 0.96 | 0.00 |
| `pdz_finger_black` | black PLA finger bodies | `(0.025, 0.030, 0.035)` | 0.00 | 0.82 | 0.00 |
| `pdz_contact_white` | white TPU contact pads | `(0.95, 0.96, 0.97)` | 0.00 | 0.90 | 0.00 |
| `tslot_aluminum` | aluminum T-slot lands | `(0.58, 0.61, 0.64)` | 0.72 | 0.58 | 0.00 |
| `tslot_slot` | dark recessed slots/backing | `(0.12, 0.135, 0.15)` | 0.25 | 0.72 | 0.00 |

PDZ rendering restores the authored visual meshes after URDF import, removes
the camera-enclosure surfaces that contain the optical origin, binds the pad
geometries separately from the finger bodies, disables ambient occlusion and
hard cast shadows, and uses Filament environment illumination with fallback
head-light intensity 1000 and environment intensity 6500.

Legacy Y-gripper checkpoints retain their separate older material and six-fill
light contract. Do not render a PDZ checkpoint with the legacy Y embodiment.

### Render Output And Validation

Each attempt writes `policy_goal_<grasp-id>.npz` containing RGB, metric depth,
camera/material/workspace/scene profile identifiers, requested MoveIt/TCP
state and measured render FK. Rendering aborts before robot motion when:

- the rendered TCP differs by more than 2 mm or 1 degree from the accepted
  MoveIt target;
- the depth image has less than 0.01 m standard deviation;
- required geometry/material assets are missing; or
- the Filament launcher exits without producing a validated artifact.

## Non-Moving Verification

After installation:

```bash
source ./setup_robot_env.sh

python3 -c 'import fcl, mujoco, numpy, scipy, trimesh, yaml'
ros2 pkg prefix lbr_bringup
ros2 pkg prefix moveit_ros_move_group
ros2 pkg prefix moveit_servo
ros2 pkg prefix robot_integration_ros
ros2 pkg prefix fp_debug_msgs

./run_pipeline.sh --list-policies
./run_pipeline.sh --workflow single-object --mode real --robots left \
  --policy velocity-rotation --part-id 0 --generate-config-only
```

Start mock MoveIt before connecting hardware and verify the PDZ model, planning
scene, controller names, passive gripper state and camera TF tree in RViz.

## Portability Gaps

A clean clone still needs four non-Git inputs:

1. the external LBR underlay;
2. the external `servo_gripper` controller workspace;
3. selected policy checkpoint files and sidecars; and
4. an ABI-compatible MuJoCo Filament library and asset directory.

Dual execution additionally needs a copied or regenerated planning catalog.
Until these are pinned or bootstrapped, the repository is not a one-command
clone-to-hardware deployment.
