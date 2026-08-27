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

The default `sim` config reproduces corrected KUKA execution-benchmark run 3: `plumbers_block/0` at stable `orientation_002`, world pose `[0.5, 0.0, 0.04]` with quaternion `[-0.7071067811865475, 0.0, 0.0, 0.7071067811865476]`. It regenerates benchmark-equivalent candidates, dynamically selects the highest-scoring stage-2 grasp, and opens the Isaac GUI. Start `./start_lbr_moveit.sh` in another terminal before running `./run_pipeline.sh --mode sim`; pass `--headless` only when the GUI is not wanted.

For `pitl` and `real`, the planning local frame is defined from the OBJ itself by subtracting the arithmetic mean of all OBJ vertices. The ROS2 `fp_debug_msgs/msg/DebugPoseItem` subscriber then treats `pose_base` as the world pose of that centroid-centered local frame when its Fabrica assembly name and part id match the requested object.

## Grasp Generation Benchmark

Run the standalone benchmark to evaluate grasp generation over Fabrica OBJ parts and robust stable orientations without executing in MuJoCo, Isaac, MoveIt, or hardware:

```bash
python scripts/run_grasp_generation_benchmark.py --limit-parts 1
python scripts/run_grasp_generation_benchmark.py --assembly plumbers_block --clean
```

The default config is `configs/grasp_generation_benchmark.yaml`; outputs go to `artifacts/grasp_generation_benchmark/` with `results.json`, `summary.csv`, `summary.md`, `index.html`, per-part stage artifacts, stable-orientation metadata, and optional generation-only fallback plans. The benchmark requires the same collision backend as normal stage-1 filtering.

## Grasp Execution Benchmark

After running the generation benchmark, execute selected stage-2 feasible grasps in MuJoCo and/or Isaac with per-attempt artifacts and videos:

```bash
python scripts/run_grasp_execution_benchmark.py --assembly beam --part 0 --limit-orientations 1 --max-grasps-per-orientation 2
python scripts/run_grasp_execution_benchmark.py --backend both --assembly beam --part 0 --orientation orientation_003 --max-grasps-per-orientation 1
python scripts/run_grasp_execution_benchmark.py --backend both --assembly beam --part 0 --orientation orientation_003 --max-grasps-per-orientation 9 --placement-xy-world 0.5,0.0 --no-resume
python scripts/run_grasp_execution_benchmark.py --backend mujoco --record-video all --limit-attempts 10
```

For the KUKA iiwa7 Isaac path, start the LBR mock state/controller stack and namespaced MoveIt planning server in one terminal:

```bash
./start_lbr_moveit.sh
```

The helper launches `robot_state_publisher`, mock `ros2_control`, and MoveGroup together. They all receive a repo-local MoveIt description derived from the same authoritative URDF as the Isaac USD, so MoveIt plans to `gripper_tcp` without the former target-Y reflection or 35 mm TCP compensation.

Then execute the highest-scored direct grasp for every `plumbers_block` orientation that has a stage-2 feasible grasp:

```bash
export ROS_LOG_DIR=/tmp/ros-log
source /opt/ros/humble/setup.bash
source /home/pdz/lbr-stack/install/setup.bash
source ros2_ws/install/setup.bash
python3 scripts/run_grasp_execution_benchmark.py \
  --backend isaac \
  --assembly plumbers_block \
  --max-grasps-per-orientation 1 \
  --no-resume \
  --output-dir artifacts/grasp_execution_benchmark_kuka_plumbers_block_corrected
```

The KUKA conversion currently applies only to the benchmark's `isaac` block. Do not use `--backend mujoco` or `--backend both` for this run. Omit `--max-grasps-per-orientation 1` only when intentionally running every direct feasible grasp; the current `plumbers_block` generation artifacts contain thousands of such candidates.

The default config is `configs/grasp_execution_benchmark.yaml`; outputs go to `artifacts/grasp_execution_benchmark/` with resumable `attempts.jsonl`, `results.json`, `summary.csv`, `index.html`, per-attempt `attempt.json`, logs, and browser-playable `attempt.webm` when video recording is enabled. The benchmark consumes the generation benchmark's stage-2 bundles and runs direct stage-2 feasible grasps in descending score order; orientations with no stage-2 feasible grasp are skipped rather than converted into regrasp attempts. By default, execution attempts keep each stage-2 bundle's saved stable orientation and Z height but shift object XY to the normal robot workspace at `[0.5, 0.0]`; pass `--placement-xy-world x,y` to choose another table location or `--use-bundle-placement` to use the generation bundle pose verbatim.

## ROS2 Workspace

The repo now contains a dedicated ROS2 workspace for hardware-facing integration:

- `ros2_ws/src/robot_integration_ros`
- `ros2_ws/dependencies.repos` for pinned external ROS2 package sources

This keeps the real-robot entrypoints and ROS packaging under `colcon`, while the rest of the project stays a normal Python repo.

Before building the overlay, fetch the pinned ROS2 source dependency used by `pitl` and `real` mode `DebugPoseItem` intake:

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

For local Isaac execution, use the local Isaac Lab launcher. The default
Isaac configs call:

```bash
/media/pdz/Elements1/IsaacLab/isaaclab.sh -p
```

through `isaac_execution.python_executable`. This field may include launcher
arguments, so a local Isaac Lab command such as
`/media/pdz/Elements1/IsaacLab/isaaclab.sh -p` is supported. Docker remains an
optional helper for reproducing the older containerized environment, not a
requirement for Isaac sim execution.

Bootstrap the MuJoCo assets:

```bash
bash scripts/download_required_assets.sh
```

This does two things:
- sparse-clones the required MuJoCo Menagerie assets under `.cache/robot_descriptions/mujoco_menagerie`
- builds `.cache/generated_mujoco_models/fr3_with_panda_hand.xml`

Bootstrap the pinned ROS2 message dependency used by `pitl` / `real` `DebugPoseItem` intake:

```bash
bash scripts/download_ros2_dependencies.sh
```

This does two things:
- imports `ros2_ws/dependencies.repos` when `vcstool` is available
- ensures `ros2_ws/src/fp_debug_msgs` is present and pinned even without `vcstool`

Override `FP_DEBUG_MSGS_REMOTE` and `FP_DEBUG_MSGS_REF` if you need to bootstrap from a mirror, a local bare repo, or a different pinned ref.

The pipeline expects the vendored Franka hand collision mesh at:
- `assets/urdf/franka_description/meshes/robot_ee/franka_hand_black/collision/hand.stl`

The KUKA iiwa7 Y-gripper configs use the local gripper meshes and generated robot USD/URDF:
- `assets/urdf/kuka_iiwa7_y_gripper/meshes/{hand.STL,left_finger.STL,right_finger.STL}`
- `assets/urdf/kuka_iiwa7_y_gripper/urdf/kuka_iiwa7_y_gripper.urdf`
- `assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda`

The URDF above is the KUKA kinematic source of truth. Its calibrated `gripper_tcp` is `0.1813 m` along local Z from `lbr_link_7`: `0.0308 m` from link 7 to the gripper base plus `0.1505 m` from the gripper base to the TCP. `python3 scripts/build_kuka_moveit_description.py` regenerates the repo-local MoveIt/ros2_control xacro while retaining the LBR hardware joint and link names required by the controller stack. The checked-in Isaac USD and the MoveIt description are covered by an FK-equivalence regression test.

For `gripper_collision_model: kuka_y_gripper`, saved bundles identify `robot_model: kuka_iiwa7`, `gripper_model: kuka_y_gripper`, and `tcp_link: gripper_tcp`.

The visual-servo RL task uses the PDZ parallel gripper instead. Its source
URDF and meshes are under `assets/urdf/kuka_iiwa7_pdz_gripper/` and
`assets/urdf/pdz_gripper_description/`, and its Isaac asset is
`assets/usd/kuka_iiwa7_pdz_gripper/kuka_iiwa7_pdz_gripper.usd`. The aperture
is 12--76 mm; goal contacts are limited to 62 mm so the approach command can
add 5 mm per finger. The named TCP is 135.5 mm above the PDZ base and rotated
-90 degrees about local Z. The checked-in MoveIt xacro and SRDF use that same
named frame, and `configs/grasp_generation_benchmark_pdz.yaml` records the
matching collision-generation contract.

RL goal RGB-D is generated from the collision-validated PDZ trajectories with
MuJoCo Filament via `python3 isaac_rl/scripts/prepare_plumbers_block_catalog.py
--stage mujoco`. The renderer uses the deployed D405 intrinsics and wrist
transform, hides only the camera enclosure surfaces that contain the optical
origin, and retains the visible gripper fingers/pads, part, and T-slot surface.

### Copying Local State To A New Worktree

Tracked assets are checked out by Git, but the MuJoCo cache, pinned ROS2 source dependency, and `colcon` build/install/log directories are ignored local state. After creating a new worktree, copy those directories from this worktree with:

```bash
git worktree add /path/to/new-worktree <branch>
scripts/copy_worktree_local_state.sh --to /path/to/new-worktree
```

You can also opt into an automatic `post-checkout` hook for future worktrees:

```bash
git config core.hooksPath .githooks
git config grasp-planning.localStateSource "$(pwd -P)"
```

Git does not enable repository hooks from tracked files by default. Once enabled, the hook runs after `git worktree add` performs the initial checkout and copies the ignored local state from `grasp-planning.localStateSource`.

## Config Layout

Pipeline configs:
- `configs/grasp_pipeline_sim.yaml`
- `configs/grasp_pipeline_sim_cumotion.yaml`
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

### D405 policy-assisted real grasp approach

The real executor can keep normal MoveIt IK/path planning for the selected
stage-2 pregrasp and replace only the final pregrasp-to-grasp segment with the
trained D405 PPO policy:

```text
stage-2 bundle target
-> MoveIt plan/execute to pregrasp
-> synchronized D405 RGB-D + explicit catalogue target
-> deterministic policy twists
-> live optical-frame TF into lbr_link_0
-> safety supervisor
-> collision-checking MoveIt Servo
-> four-frame learned completion hold
-> explicitly approved gripper close
-> optional MoveIt lift
```

Set these fields in the selected real-pipeline YAML:

```yaml
real_execution:
  grasp_approach_controller: "d405_policy"
  visual_servo_config: "configs/visual_servo_real_d405.yaml"
  stop_after: "grasp"
```

Then fill the checkpoint, checkpoint-metadata sidecar, and exact `target_id` in
`configs/visual_servo_real_d405.yaml`. Startup fails if the target's part or
grasp ID differs from the selected stage-2 bundle, if the D405 serial/profile
does not match, or if the checkpoint/catalogue hashes and observation/action
contracts do not match the sidecar.

After reviewing and selecting a checkpoint, create its sidecar with:

```bash
python3 scripts/write_d405_checkpoint_metadata.py \
  --checkpoint /path/to/reviewed-checkpoint.pth \
  --output /path/to/reviewed-checkpoint.deployment.json \
  --source-commit "$(git rev-parse HEAD)"
```

The robot-side Python environment needs the lightweight policy dependencies but
does not need Isaac Sim:

```bash
python3 -m pip install -e '.[deployment]'
```

The checked-in deployment config is intentionally non-moving:
`command_sink: dry_run`, `real_motion_approved: false`, and
`allow_gripper_close_on_completion: false`. Start the aligned LBR stack with
the opt-in Servo node for dry-run/status checks and later staged motion:

```bash
./start_lbr_moveit.sh --mode hardware --servo
python3 scripts/run_d405_ppo_visual_servo.py \
  --config configs/visual_servo_real_d405.yaml \
  --expected-part-id 0 \
  --expected-grasp-id g1973
```

The dry-run preflight also requires fresh deadman and E-stop heartbeats. For a
stationary, non-moving dry-run only, these can be synthetic publishers in two
additional terminals:

```bash
ros2 topic pub -r 10 /d405_visual_servo/deadman std_msgs/msg/Bool '{data: true}'
ros2 topic pub -r 10 /d405_visual_servo/emergency_stop std_msgs/msg/Bool '{data: false}'
```

Do not use synthetic operator signals for real motion. Connect those topics to
the reviewed physical operator controls, set a real `WrenchStamped`
`force_topic`, and keep all three streams publishing as heartbeats; stale
joint, force, deadman, or E-stop data latches a zero-motion hold.

MoveIt Servo consumes speed-unit `TwistStamped` commands on
`/lbr/servo_node/delta_twist_cmds`, commands `gripper_tcp` in `lbr_link_0`, and
monitors the same planning scene published by MoveGroup. It therefore applies
joint/singularity and self/scene-collision slowdown or halt behavior during the
learned approach. A full motion planner is still used for pregrasp and lift;
calling a global planner for every 15 Hz policy step is not supported.

Real Servo output additionally requires `command_sink: moveit_servo`,
`real_motion_approved: true`, the CLI/caller real-motion confirmation, live
deadman and emergency-stop topics, reviewed workspace/joint/force limits, and
a healthy Servo status/command subscriber. Gripper closing occurs only after
the learned completion probability is at least `0.95` for four consecutive
low-speed frames and both `gripper_enabled` and
`allow_gripper_close_on_completion` are true.

After the standalone dry-run and guarded no-close trials pass, select
`d405_policy` in `configs/grasp_pipeline_real_lbr_iiwa7.yaml` and run the normal
real pipeline. It performs MoveIt pregrasp planning first, starts the policy
approach, closes only after the completion gate, deactivates Servo, and then
uses MoveIt for the configured lift:

```bash
./run_pipeline.sh --mode real --config configs/grasp_pipeline_real_lbr_iiwa7.yaml
```

Use the `planning` block in `configs/grasp_pipeline_*.yaml` to tune grasp generation and filtering:
- `stage1_cache_enabled` and `stage1_cache_dir` cache the generated stage-1 grasps plus surface samples per object mesh and stage-1 planning settings. Cache hits still write the normal stage artifacts.
- `roll_angle_step_deg` expands roll samples over a full 360 degrees. For example, `15.0` generates 24 roll angles from 0 through 345 degrees.
- `stage1_pose_upright_axis_enabled` adds a live-pose-derived world-upright roll sample during stage 1. Stage-1 caching stores the pose-independent base grasps and augments cache hits with only the missing per-run upright roll variants, so real/PITL pose jitter does not force a full regeneration.
- `detailed_finger_contact_gap_m` changes the gripper contact geometry used during detailed checks.
- `floor_clearance_margin_m` is a stage-2 filtering margin: the full hand/finger collision geometry must stay at least this far above the world `z=0` floor. This does not change MuJoCo execution settings.
- `top_grasp_score_weight` is applied during stage-2 scoring after the real/execution pose is known. It boosts grasps whose pregrasp-to-grasp approach is top-down in world coordinates, with movement mostly along `-Z`.
- `regrasp_transfer_top_grasp_score_weight` applies the same top-down score to the regrasp transfer pickup in the initial pose. It defaults higher than `top_grasp_score_weight` because the first pickup is more sensitive to wrist orientation than setting the object back down.
- `skip_stage1_collision_checks: true` keeps all generated stage-1 grasps and skips offline assembly collision filtering. For a one-off run, pass `--skip-stage1-collision-checks`.

Use `configs/mujoco_simulation.yaml` to tune:
- grasp approach settings such as `pregrasp_offset_m` and `gripper_width_clearance_m`
- scene contact settings such as object mass, friction, `solref`, `solimp`, margin, and gap
- robot timing and speed such as `timestep_s`, `control_substeps`, `speed_scale`, adaptive final-approach slowdown, IK, and trajectory settings
- gripper actuation and settle behavior such as `open_ctrl`, `closed_ctrl`, and `close_steps`

MuJoCo also has an optional one-placement regrasp fallback for the case where stage 1 finds assembly-feasible grasps but stage 2 rejects all of them because of the floor. Configure it in the `mujoco_execution` block:
- `regrasp_fallback_enabled: true`
- `force_regrasp_fallback: true` or `./run_pipeline.sh ... --force-regrasp-fallback` to test the fallback even when direct stage-2 grasps exist; forced mode skips the surface currently on the floor and fails clearly if no feasible different-surface plan exists
- `regrasp_plan_artifact` for the JSON plan containing the transfer grasp, staging pose, and final grasp
- `regrasp_html_artifact` for the side-by-side debug view of the initial/staging poses, floor plane, all evaluated grasp markers, and highlighted planned transfer/final grasps
- `regrasp_staging_xy_offsets_m` samples multiple table XY locations around `regrasp_staging_xy_world`
- `regrasp_max_placement_options` caps how many feasible placement options are written into the regrasp plan
- `regrasp_moveit_max_candidate_plans`, `regrasp_moveit_transfer_candidates_per_placement`, and `regrasp_moveit_final_candidates_per_placement` cap the runtime MoveIt candidate plans scored before MuJoCo execution

The fallback computes convex-hull support facets, checks homogeneous-COM stability for candidate resting poses, samples staging XY locations, searches for a final assembly-feasible grasp in each staging pose, and then searches raw stage-1 grasps for a transfer grasp that is floor-feasible in both the initial and staging poses. MuJoCo can execute this fallback either with native IK or with MoveIt-planned transfer/place/final-pick trajectories. With MoveIt, execution first plans and scores a capped set of placement/transfer/final-grasp combinations, ranks them by joint path length, joint jumps, and the static placement score, then executes the cheapest candidate in MuJoCo and falls back to the next cheapest if needed. During regrasp transport, the runner inserts higher lift/rotate/translate waypoints before descending to the staging placement; tune that with `robot.regrasp_transport_clearance_m` in `configs/mujoco_simulation.yaml`. The MuJoCo attempt artifact includes `planned_candidates`, `attempts`, and trajectory diagnostics for debugging path choice.

For a planning-only regrasp visualization that uses a known different-surface case, run:

```bash
./run_pipeline.sh --mode sim --config configs/grasp_pipeline_sim_plumbers_regrasp.yaml --backend none --headless
```

This writes `artifacts/plumbers_mujoco_regrasp_plan.html` without launching MuJoCo execution.

MuJoCo can either use its native damped-IK arm controller or MoveIt-planned arm trajectories:
- default sim config: `mujoco_execution.controller: "moveit"`
- native MuJoCo IK: set `mujoco_execution.controller: "native"`
- cuMotion through MoveIt: use `configs/grasp_pipeline_sim_cumotion.yaml`

The MoveIt-backed MuJoCo path requires the FR3 MoveIt stack to be running and sourced, but only uses MoveIt planning services. For direct pickups, MoveIt plans `pregrasp`, `grasp`, and `lift` trajectories from the stage-2 bundle. For regrasp fallback, it plans transfer, staging placement, retreat, and final-pick trajectories. MuJoCo still executes those joint waypoints, closes/opens the gripper, simulates contacts, and evaluates pickup success by object lift height.

The cuMotion sim config keeps MoveIt as the ROS2 integration layer and asks `/plan_kinematic_path` to use `moveit_pipeline_id: "isaac_ros_cumotion"` and `moveit_planner_id: "cuMotion"`. This requires a running MoveIt stack that has NVIDIA Isaac ROS cuMotion installed and configured under those identifiers. If your local MoveIt config uses different names, override `moveit_pipeline_id` or `moveit_planner_id` in the YAML.

Start the local fake-hardware cuMotion + MoveIt stack in one terminal:

```bash
./start_cumotion_moveit.sh
```

Leave that script running, then run the MuJoCo sim from another terminal.

Run MuJoCo sim with cuMotion-backed MoveIt planning:

```bash
./run_pipeline.sh --mode sim --config configs/grasp_pipeline_sim_cumotion.yaml --backend mujoco --headless
```

The default sim config uses Isaac execution; other configs can opt in with `isaac_execution.enabled: true`. The runner generates a collision-enabled bundle-local USD from the stage-2 bundle by default, so the spawned Isaac asset uses the same frame as the ground recheck. With no `isaac_execution.fr3_usd` override it uses Isaac Lab's Factory Franka mimic USD because that asset has manipulation-ready finger contact geometry. It also exposes the spawned gripper mesh prims as PhysX collision geometry before simulation reset, then validates success from the part lift height using `isaac_execution.success_height_margin_m`. Disable `mujoco_execution.enabled` if you want Isaac only. Isaac direct pickups use `isaac_execution.controller: "moveit"`: MoveIt plans the same `pregrasp`, `grasp`, and `lift` pose targets used by real execution, then Isaac streams the returned joint waypoints in simulation.

For the KUKA iiwa7 Y-gripper path, use the default sim config, `configs/grasp_pipeline_sim_isaac.yaml`, `configs/grasp_pipeline_gazebo_lbr_iiwa7.yaml`, or `configs/grasp_pipeline_real_lbr_iiwa7.yaml`. These configs use `gripper_collision_model: kuka_y_gripper`; the grasp-generation benchmark config uses the same gripper model and a 3x3 contact-offset grid with max lateral offset `0.002916666666666667 m` and max approach offset `0.0030833333333333333 m`.

### KUKA iiwa7 Real Hardware Runbook

This runbook uses the working three-host network layout:

- `192.170.20.1`: pipeline, LBR ROS2 control, MoveIt, and RViz computer
- `192.170.20.2`: KUKA controller / FRI peer
- `192.170.20.3`: gripper computer

All ROS2 processes on `.1` and `.3` must start with the same `ROS_DOMAIN_ID`, non-localhost discovery, and compatible DDS configuration. Environment changes do not affect processes that are already running; restart a process after changing its ROS environment.

On the pipeline computer, source the repository helper in every new ROS2 terminal:

```bash
cd /media/pdz/Elements1/Grasp_Planning_kuka_iiwa_7
source ./setup_ros2_hardware_env.sh
```

It sources ROS Humble, `/home/pdz/lbr-stack`, and the repository overlay, then selects domain `0`, network discovery, and Fast DDS. It also sets `GRASP_KEEP_ROS_DISCOVERY_ENV=1` so `run_pipeline.sh` preserves this hardware network configuration.

Before launching ROS, start the FRI client application from the KUKA SmartPAD with the configured client IP and a 10 ms send period. Use position control or joint impedance with client command mode `POSITION`. Place the real arm in a collision-free starting configuration; real execution plans from the current `/lbr/joint_states` and does not reset the physical robot to the Isaac ready pose.

Terminal 1 on `.1`: start the physical robot, trajectory controller, robot-state publisher, and namespaced MoveIt server from the aligned description.

```bash
cd /media/pdz/Elements1/Grasp_Planning_kuka_iiwa_7
source ./setup_ros2_hardware_env.sh
./start_lbr_moveit.sh --mode hardware
```

Terminal 1 on the gripper computer `.3`: start the endpoint gripper controller. Put these exports in its launch wrapper or service environment to avoid repeating them manually.

```bash
cd /home/s3c/Workspaces
source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset ROS_DISCOVERY_SERVER ROS_STATIC_PEERS ROS_AUTOMATIC_DISCOVERY_RANGE
unset CYCLONEDDS_URI FASTRTPS_DEFAULT_PROFILES_FILE

ros2 launch servo_gripper gripper.launch.py
```

Terminal 3 on `.1`: for the fixed corrected-benchmark run-3 test pose, publish the expected Fabrica `plumbers_block/0` perception pose. The temporary left-robot correction in `configs/grasp_pipeline_real_lbr_iiwa7.yaml` subtracts `0.840 m` from the received world Y coordinate before planning, so the example perception Y of `0.840 m` becomes a MoveIt target Y of `0.0 m`.

```bash
cd /media/pdz/Elements1/Grasp_Planning_kuka_iiwa_7
source ./setup_ros2_hardware_env.sh
ros2 topic pub -r 2 \
  /perception/fp/pose_base/fused/assembly \
  fp_debug_msgs/msg/DebugPoseItem \
  "{assembly_name: plumbers_block, part_id: 0, mode: config_test, score: 1.0, pose_base: {pose: {position: {x: 0.5, y: 0.84, z: 0.04}, orientation: {x: -0.7071067811865475, y: 0.0, z: 0.0, w: 0.7071067811865476}}}}"
```

Terminal 4 on `.1`: verify discovery before enabling hardware execution.

```bash
cd /media/pdz/Elements1/Grasp_Planning_kuka_iiwa_7
source ./setup_ros2_hardware_env.sh

ros2 service type /gripper_controller/open
ros2 service type /gripper_controller/close
ros2 service type /gripper_controller/stop
ros2 service list | grep -E '^/lbr/(compute_ik|plan_kinematic_path)$'
ros2 action list | grep '^/lbr/execute_trajectory$'
ros2 topic list | grep -E '^/(lbr/joint_states|perception/fp/pose_base/fused/assembly)$'
```

All three gripper type commands must print `std_srvs/srv/Trigger`, and every `grep` must return its requested interface. Then run:

```bash
./run_pipeline.sh \
  --mode real \
  --config configs/grasp_pipeline_real_lbr_iiwa7.yaml
```

Review the printed grasp target and type `yes` only when the workspace is clear. The current KUKA config opens the gripper, moves through pregrasp and grasp, closes, lifts `0.08 m`, and stops while holding the object. It retains confirmation and `0.05` velocity/acceleration scaling.

The ROS2 package also exposes `fp_debug_msgs/action/GraspAssembly`, whose success
contract includes transporting the insertion part to its pre-assembly pose. The
current real executor only implements pickup and lift, so the action server is
deliberately blocked before any hardware subprocess or motion starts. You can
start it for ROS graph and client-integration checks:

```bash
source ./setup_ros2_hardware_env.sh
ros2 run robot_integration_ros grasp_assembly_action_server \
  --config configs/grasp_pipeline_real_lbr_iiwa7.yaml
```

Then send a current single-robot goal:

```bash
ros2 action send_goal --feedback \
  /grasp_assembly \
  fp_debug_msgs/action/GraspAssembly \
  "{assembly_name: plumbers_block, base_part_id: 4, insertion_part_id: 0, holder_robot: right, inserter_robot: left}"
```

For now, goal validation uses only `assembly_name`, `insertion_part_id`, and
`inserter_robot`, and requires `inserter_robot: left`; `base_part_id` and
`holder_robot` remain reserved for the future assembly flow. Without `--execute`,
valid goals abort with `EXECUTION_DISABLED`. Even with `--execute`, they abort
with `TRANSPORT_UNSUPPORTED` before pose intake, planning, gripper commands, or
arm motion. This prevents a pickup-only lift from being reported as successful
completion of the stronger action contract. Use the direct real pipeline command
above when intentionally testing pickup and lift before pre-assembly transport
is implemented.

Manual gripper commands from any correctly configured ROS2 terminal:

```bash
# Release the object.
ros2 service call /gripper_controller/open std_srvs/srv/Trigger "{}"

# Stop the gripper motor immediately.
ros2 service call /gripper_controller/stop std_srvs/srv/Trigger "{}"
```

Stop the pose publisher after the pipeline reports that it received the pose. Shut down the pipeline first, then MoveIt, the hardware launch, and finally the SmartPAD FRI application.

#### Why The ROS Environment Must Match

ROS2 domain and RMW selection happen when each process starts. `ROS_DOMAIN_ID` separates independent DDS graphs, so domain `42` cannot discover domain `0`. `ROS_LOCALHOST_ONLY=1` prevents discovery across `.1` and `.3`. DDS discovery also has to select the correct network interface on a multihomed robot computer.

Using the same middleware implementation is not a fundamental ROS2 requirement: Fast DDS and Cyclone DDS both implement DDS/RTPS and can interoperate in principle. In this setup, however, the gripper was running Cyclone DDS while the pipeline computer had only the Fast DDS RMW installed, and remote service discovery did not succeed reliably. Domain and RMW settings were corrected together, so the session did not prove that the vendor mismatch alone caused the failure. Standardizing both hosts on the already-installed `rmw_fastrtps_cpp` removes that variable and its vendor-specific discovery and interface differences.

The alternatives are to install Cyclone DDS RMW on `.1` and run every process with Cyclone DDS, or deliberately configure mixed-vendor discovery, multicast routing, interfaces, and firewall rules. A Fast DDS discovery server can replace multicast when the network blocks it, but every participant still needs a consistent discovery-server configuration. For this two-computer ROS graph, one shared environment in launch wrappers or system services is simpler than exporting variables interactively and does not require changing middleware on every run.

Run Isaac-backed sim locally:

```bash
./run_pipeline.sh --mode sim
```

If you want the MuJoCo backend instead, bootstrap its generated robot XML first:

```bash
bash scripts/download_required_assets.sh
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
- `scripts/run_grasp_generation_benchmark.py`
- `scripts/download_required_assets.sh`
- `scripts/download_ros2_dependencies.sh`
- `configs/grasp_generation_benchmark.yaml`
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
- `pitl` and `real` use one ROS2 subscriber: set `ros2.pose_base_topic`, `ros2.assembly_name`, and numeric `ros2.part_id` in the pipeline YAML before running those modes. The configured topic must publish `fp_debug_msgs/msg/DebugPoseItem`. Optional `ros2.position_offset_m: [x, y, z]` adds a fixed world-axis translation to the received pose before planning; the left KUKA real config currently uses `[0.0, -0.840, 0.0]`.
