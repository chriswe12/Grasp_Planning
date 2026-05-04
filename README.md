# Grasp Planning

YAML-driven Fabrica grasp planning with three pipeline modes behind one entrypoint:
- `sim`: offline execution-world pose from config, then optional MuJoCo and/or Isaac execution
- `pitl`: ROS2 perception pose intake, then optional MuJoCo and/or Isaac execution
- `real`: ROS2 perception pose intake, planning, and optional real-robot execution from the stage-2 bundle

## Entry Point

```bash
./run_pipeline.sh --mode sim
./run_pipeline.sh --mode pitl
./run_pipeline.sh --mode real
./run_pipeline.sh --mode sim --headless
./run_pipeline.sh --mode sim --backend isaac --headless
```

Default configs:
- `configs/grasp_pipeline_sim.yaml`
- `configs/grasp_pipeline_pitl.yaml`
- `configs/grasp_pipeline_real.yaml`

`sim` and `pitl` both run stage 1, write stage-1 artifacts, run stage 2, write stage-2 artifacts, then execute from the stage-2 bundle with whichever simulation backends are enabled. Use `--backend {config,mujoco,isaac,both,none}` to override the YAML for one run. `real` writes the same stage artifacts and can optionally execute the selected grasp on hardware when `real_execution.enabled: true`.

For `pitl` and `real`, the planning local frame is defined from the OBJ itself by subtracting the arithmetic mean of all OBJ vertices. The ROS2 `fp_debug_msgs/msg/DebugFrame` subscriber then treats the selected `pose_item.pose_base` as the world pose of that centroid-centered local frame.

## ROS2 Workspace

The repo now contains a dedicated ROS2 workspace for hardware-facing integration:

- `ros2_ws/src/robot_integration_ros`
- `ros2_ws/dependencies.repos` for pinned external ROS2 package sources

This keeps the real-robot entrypoints and ROS packaging under `colcon`, while the rest of the project stays a normal Python repo.

Before building the overlay, fetch the pinned ROS2 source dependency used by `pitl` and `real` mode DebugFrame intake:

```bash
bash scripts/download_ros2_dependencies.sh
```

Build and source it as an overlay on top of your FR3 / MoveIt workspace:

```bash
source /opt/ros/<distro>/setup.bash
source /path/to/your/fr3_moveit_ws/install/setup.bash

cd ros2_ws
colcon build --packages-select fp_debug_msgs robot_integration_ros --symlink-install
source install/setup.bash
```

Once sourced, you can run the real-robot EE mover with:

```bash
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --execute
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --keep-current-orientation --execute
```

### Two-Shell Launch For Real Hardware

Use normal ROS2 discovery unless you deliberately need an isolated domain. The repo launcher defaults to `ROS_DOMAIN_ID=0` and clears localhost-only discovery settings.

Terminal 1: launch the FR3 MoveIt stack

```bash
source /opt/ros/humble/setup.bash
source /home/pdz/franka_ros2_ws/install/setup.bash

ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=<robot_ip> use_fake_hardware:=false
```

Terminal 2: source the robot integration overlay and run the script

```bash
source /opt/ros/humble/setup.bash
source /home/pdz/franka_ros2_ws/install/setup.bash
source /media/pdz/Elements1/perception_bag_test/ros2_ws/install/setup.bash
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

For Isaac execution, build the repo-specific Isaac container:

```bash
./docker_env.sh build
```

The container includes ROS2 Jazzy and a built `fp_debug_msgs` overlay for `pitl`
pose intake. The top-level pipeline runs with the container venv at
`/opt/grasp-pipeline-venv/bin/python`; Isaac execution still runs through
`/isaac-sim/python.sh` from the `isaac_execution.python_executable` config.

Bootstrap the MuJoCo assets:

```bash
bash scripts/download_required_assets.sh
```

This does two things:
- sparse-clones the required MuJoCo Menagerie assets under `.cache/robot_descriptions/mujoco_menagerie`
- builds `.cache/generated_mujoco_models/fr3_with_panda_hand.xml`

Bootstrap the pinned ROS2 message dependency used by `pitl` / `real` DebugFrame intake:

```bash
bash scripts/download_ros2_dependencies.sh
```

This does two things:
- imports `ros2_ws/dependencies.repos` when `vcstool` is available
- ensures `ros2_ws/src/fp_debug_msgs` is present and pinned even without `vcstool`

Override `FP_DEBUG_MSGS_REMOTE` and `FP_DEBUG_MSGS_REF` if you need to bootstrap from a mirror, a local bare repo, or a different pinned ref.

The pipeline expects the vendored Franka hand collision mesh at:
- `assets/urdf/franka_description/meshes/robot_ee/franka_hand_black/collision/hand.stl`

## Config Layout

Pipeline configs:
- `configs/grasp_pipeline_sim.yaml`
- `configs/grasp_pipeline_sim_isaac.yaml`
- `configs/grasp_pipeline_pitl.yaml`
- `configs/grasp_pipeline_pitl_isaac.yaml`
- `configs/grasp_pipeline_real.yaml`

Shared MuJoCo execution config:
- `configs/mujoco_simulation.yaml`

Isaac execution config:
- `isaac_execution` block inside `configs/grasp_pipeline_sim.yaml`
- `isaac_execution` block inside `configs/grasp_pipeline_pitl.yaml`
- `scripts/run_fabrica_grasp_in_isaac.py`
- `scripts/convert_stl_to_usd.py`

Mesh/frame debug view:

```bash
./scripts/write_part_frame_debug_html.py \
  --input-json artifacts/pitl_pipeline_stage2_ground_feasible.json \
  --output-html artifacts/part_frame_debug.html
```

This writes a mesh-only HTML view showing the saved bundle-local part, its
area-weighted centroid, and the transformed execution/world pose when the input
bundle contains `metadata.execution_world_pose`.

Real hardware execution config:
- `real_execution` block inside `configs/grasp_pipeline_real.yaml`

Use the `planning` block in `configs/grasp_pipeline_*.yaml` to tune grasp generation and filtering:
- `roll_angle_step_deg` expands roll samples over a full 360 degrees. For example, `15.0` generates 24 roll angles from 0 through 345 degrees.
- `detailed_finger_contact_gap_m` changes the gripper contact geometry used during detailed checks.
- `floor_clearance_margin_m` is a stage-2 filtering margin: the full hand/finger collision geometry must stay at least this far above the world `z=0` floor. This does not change MuJoCo execution settings.
- `top_grasp_score_weight` is applied during stage-2 scoring after the real/execution pose is known. It boosts grasps whose pregrasp-to-grasp approach is top-down in world coordinates, with movement mostly along `-Z`.
- `skip_stage1_collision_checks: true` keeps all generated stage-1 grasps and skips offline assembly collision filtering. For a one-off run, pass `--skip-stage1-collision-checks`.

Use `configs/mujoco_simulation.yaml` to tune:
- grasp approach settings such as `pregrasp_offset_m` and `gripper_width_clearance_m`
- scene contact settings such as object mass, friction, `solref`, `solimp`, margin, and gap
- robot timing and speed such as `timestep_s`, `control_substeps`, `speed_scale`, IK and trajectory settings
- gripper actuation and settle behavior such as `open_ctrl`, `closed_ctrl`, and `close_steps`

MuJoCo can either use its native damped-IK arm controller or MoveIt-planned arm trajectories:
- default: `mujoco_execution.controller: "native"`
- MoveIt planning with MuJoCo physics/viewer: set `mujoco_execution.controller: "moveit"`

The MoveIt-backed MuJoCo path requires the FR3 MoveIt stack to be running and sourced, but only uses MoveIt planning services. MoveIt plans `pregrasp`, `grasp`, and `lift` joint trajectories from the stage-2 bundle; MuJoCo still executes those joint waypoints, closes the gripper, simulates contacts, and evaluates pickup success by object lift height.

For Isaac execution, use the Isaac-only config or set `isaac_execution.enabled: true`. The runner generates a collision-enabled bundle-local USD from the stage-2 bundle by default, so the spawned Isaac asset uses the same frame as the ground recheck. Disable `mujoco_execution.enabled` if you want Isaac only.

Run Isaac-backed sim through the container:

```bash
./docker_env.sh run ./run_pipeline.sh --mode sim --config configs/grasp_pipeline_sim_isaac.yaml --headless
```

If you want the MuJoCo backend instead, bootstrap its generated robot XML first:

```bash
./docker_env.sh run bash scripts/download_required_assets.sh
```

## Repo Shape

Kept code is limited to the pipeline product:
- `run_pipeline.sh`
- `docker_env.sh`
- `Dockerfile`
- `scripts/run_grasp_pipeline.py`
- `scripts/run_fabrica_grasp_in_mujoco.py`
- `scripts/run_fabrica_grasp_in_isaac.py`
- `scripts/convert_stl_to_usd.py`
- `scripts/build_mujoco_fr3_hand_models.py`
- `scripts/download_required_assets.sh`
- `scripts/download_ros2_dependencies.sh`
- `grasp_planning/grasping/`
- `grasp_planning/pipeline/`
- `grasp_planning/ros2/`
- `grasp_planning/mujoco/`
- `grasp_planning/envs/`
- `grasp_planning/planning/`

Fabrica OBJ assets live under `assets/obj/fabrica/`.

## Notes

- The default Fabrica OBJ scale in the pipeline configs is `0.01`.
- The MuJoCo runner uses the exact `execution_world_pose` stored in the stage-2 bundle unless you override placement explicitly.
- `pitl` and `real` use one ROS2 subscriber: set `ros2.debug_frame_topic` and `ros2.object_id` in the pipeline YAML before running those modes.
