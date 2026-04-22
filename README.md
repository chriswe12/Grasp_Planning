# Grasp Planning

YAML-driven Fabrica grasp planning with three pipeline modes behind one entrypoint:
- `sim`: offline execution-world pose from config, then MuJoCo execution
- `pitl`: ROS2 perception pose intake, then MuJoCo execution
- `real`: ROS2 perception pose intake, planning, and optional real-robot execution from the stage-2 bundle

## Entry Point

```bash
./run_pipeline.sh --mode sim
./run_pipeline.sh --mode pitl
./run_pipeline.sh --mode real
./run_pipeline.sh --mode sim --headless
```

Default configs:
- `configs/grasp_pipeline_sim.yaml`
- `configs/grasp_pipeline_pitl.yaml`
- `configs/grasp_pipeline_real.yaml`

`sim` and `pitl` both run stage 1, write stage-1 artifacts, run stage 2, write stage-2 artifacts, then execute from the stage-2 bundle. `real` writes the same stage artifacts and can optionally execute the selected grasp on hardware when `real_execution.enabled: true`.

## ROS2 Workspace

The repo now contains a dedicated ROS2 workspace for hardware-facing integration:

- `ros2_ws/src/robot_integration_ros`

This keeps the real-robot entrypoints and ROS packaging under `colcon`, while the rest of the project stays a normal Python repo.

Build and source it as an overlay on top of your FR3 / MoveIt workspace:

```bash
source /opt/ros/<distro>/setup.bash
source /path/to/your/fr3_moveit_ws/install/setup.bash

cd ros2_ws
colcon build --packages-select robot_integration_ros --symlink-install
source install/setup.bash
```

Once sourced, you can run the real-robot EE mover with:

```bash
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --execute
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --keep-current-orientation --execute
```

### Two-Shell Launch For Real Hardware

Use two fresh terminals. In both terminals, use the same private ROS domain so you do not collide with other `move_group` instances on the network.

Terminal 1: launch the FR3 MoveIt stack

```bash
source /opt/ros/humble/setup.bash
source /home/pdz/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1

ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=<robot_ip> use_fake_hardware:=false
```

Terminal 2: source the robot integration overlay and run the script

```bash
source /opt/ros/humble/setup.bash
source /home/pdz/franka_ros2_ws/install/setup.bash
source /media/pdz/Elements1/robot_integration/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
```

Optional check:

```bash
ros2 control list_controllers
```

You want `fr3_arm_controller` to be `active`.

Plan only:

```bash
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --keep-current-orientation
```

Execute on hardware:

```bash
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --execute
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --keep-current-orientation --execute
```

## Standalone Real-Robot EE Motion

For direct MoveIt-controlled FR3 end-effector testing there is a standalone script:

```bash
python scripts/move_real_robot_ee.py --x 0.35 --y 0.00 --z 0.40
python scripts/move_real_robot_ee.py --x 0.35 --y 0.00 --z 0.40 --execute
python scripts/move_real_robot_ee.py --x 0.32 --y -0.10 --z 0.28 --roll 3.14159 --pitch 0.0 --yaw 1.5708 --execute
```

Notes:
- The canonical ROS2-packaged entrypoint is now `ros2 run robot_integration_ros move_real_robot_ee ...`.
- `scripts/move_real_robot_ee.py` is a source-tree shim around the same workspace package.
- The script assumes the FR3 MoveIt stack is already running elsewhere and exposing `/compute_ik`, `/plan_kinematic_path`, and `/execute_trajectory`.
- `--keep-current-orientation` queries the current EE pose from MoveIt FK and reuses only its orientation while applying your requested `x/y/z`.
- Without `--execute`, the script only checks IK and planning. Add `--execute` to move the hardware.
- The default motion scales are intentionally slow: `velocity_scale=0.05` and `acceleration_scale=0.05`.
- `Ctrl-C` only sends a best-effort cancel to MoveIt. Do not rely on it as a safety stop; use the robot-side stop/pause for emergency interruption.
- If you do not provide orientation, the script uses the thesis neutral/top-down orientation: `roll=pi`, `pitch=0`, `yaw=pi/2`.
- The default planning frame is `base` and the default end-effector link is `fr3_hand_tcp`. Override these if your MoveIt setup uses different names.

## Setup

Bootstrap the MuJoCo assets:

```bash
bash scripts/download_required_assets.sh
```

This does two things:
- sparse-clones the required MuJoCo Menagerie assets under `.cache/robot_descriptions/mujoco_menagerie`
- builds `.cache/generated_mujoco_models/fr3_with_panda_hand.xml`

The pipeline expects the vendored Franka hand collision mesh at:
- `assets/urdf/franka_description/meshes/robot_ee/franka_hand_black/collision/hand.stl`

## Config Layout

Pipeline configs:
- `configs/grasp_pipeline_sim.yaml`
- `configs/grasp_pipeline_pitl.yaml`
- `configs/grasp_pipeline_real.yaml`

Shared MuJoCo execution config:
- `configs/mujoco_simulation.yaml`

Real hardware execution config:
- `real_execution` block inside `configs/grasp_pipeline_real.yaml`

Use `configs/mujoco_simulation.yaml` to tune:
- grasp approach settings such as `pregrasp_offset_m` and `gripper_width_clearance_m`
- scene contact settings such as object mass, friction, `solref`, `solimp`, margin, and gap
- robot timing and speed such as `timestep_s`, `control_substeps`, `speed_scale`, IK and trajectory settings
- gripper actuation and settle behavior such as `open_ctrl`, `closed_ctrl`, and `close_steps`

## Repo Shape

Kept code is limited to the pipeline product:
- `run_pipeline.sh`
- `scripts/run_grasp_pipeline.py`
- `scripts/run_fabrica_grasp_in_mujoco.py`
- `scripts/build_mujoco_fr3_hand_models.py`
- `scripts/download_required_assets.sh`
- `grasp_planning/grasping/`
- `grasp_planning/pipeline/`
- `grasp_planning/ros2/`
- `grasp_planning/mujoco/`

Fabrica OBJ assets live under `assets/obj/fabrica/`.

## Notes

- The default Fabrica OBJ scale in the pipeline configs is `0.01`.
- The MuJoCo runner uses the exact `execution_world_pose` stored in the stage-2 bundle unless you override placement explicitly.
- ROS2 dual-topic intake is enabled when `object_id`, `local_frame_offset_topic`, and `execution_frame_topic` are all configured in the pipeline YAML.
