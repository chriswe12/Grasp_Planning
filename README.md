# Fabrica Grasp Planning And Dual-Arm Assembly

This repository plans collision-checked parallel-jaw grasps for Fabrica parts
and executes them with Franka Research 3 or KUKA iiwa7 robots. It contains two
related workflows:

1. The **single-object pipeline** generates grasps, filters them at a known
   world pose, and can execute a selected stage-2 bundle in MuJoCo, Isaac, or
   on a real robot.
2. The **dual-arm assembly pipeline** selects one grasp that holds the partial
   assembly and another that picks and transports the incoming part to a
   symmetry-aware pre-insertion pose.

Both workflows are YAML-driven and write JSON/HTML artifacts that explain what
was selected and why alternatives were rejected. Generated artifacts under
`artifacts/` are local outputs and are not the source code.

## Start Here

Choose the entrypoint that matches the job:

| Goal | Entrypoint | Main config |
| --- | --- | --- |
| Generate and execute one grasp | `./run_pipeline.sh --mode sim` | `configs/grasp_pipeline_sim.yaml` |
| Use a ROS2 perception pose in simulation | `./run_pipeline.sh --mode pitl` | `configs/grasp_pipeline_pitl.yaml` |
| Plan or execute one real-robot pickup | `./run_pipeline.sh --mode real` | `configs/grasp_pipeline_real.yaml` |
| Build dual holder/inserter planning artifacts | `scripts/build_dual_grasp_pairs.py` | `configs/dual_grasp_planning.yaml` |
| Plan and run the dual-arm vertical slice in Isaac | `./run_simple_dual_robot.sh --mode sim` | `configs/dual_grasp_planning.yaml` |
| Preflight or run the guarded dual-arm hardware slice | `./run_simple_dual_robot.sh --mode real` | `configs/dual_grasp_planning.yaml` |
| Evaluate grasp generation without execution | `scripts/run_grasp_generation_benchmark.py` | `configs/grasp_generation_benchmark.yaml` |
| Execute saved benchmark grasps | `scripts/run_grasp_execution_benchmark.py` | `configs/grasp_execution_benchmark.yaml` |

The single-object and dual-arm pipelines share grasp generation and collision
geometry, but their numbered stages are different. When this README says
"dual Stage 3," it means holder/inserter pair construction, not the
single-object stage-2 bundle.

## What Is Implemented

The single-object path supports:

- object-local antipodal grasp generation and scoring;
- assembly and world-pose collision filtering;
- finite-symmetry pickup-grasp expansion;
- stage-2 bundles as the shared MuJoCo, Isaac, and real-execution contract;
- optional MuJoCo regrasp fallback; and
- planning-only and execution benchmarks.

The dual-arm path supports:

- compiling an authored assembly order into explicit partial-assembly states;
- reusable holder grasps checked against each changing assembly state;
- incoming-part grasps and collision-checked holder/inserter pairs;
- two distinct uses of symmetry:
  - **pickup symmetry** creates equivalent object-local grasp choices before
    the incoming part is grasped;
  - **transition symmetry** creates equivalent final/pre-insertion corridors
    after the grasp is fixed;
- distance/rotation ranking followed by exact MoveIt IK and full trajectory
  planning with candidate fallback;
- execution of the validated vertical slice in Isaac; and
- guarded real-robot planning/execution through pre-insertion.

The dual-arm path does **not** yet execute constrained insertion, release the
incoming part, retreat, or coordinate an arbitrary complete assembly. Real
execution stops at the configured phase, no later than pre-insertion. Read
`DUAL_ROBOT_HOLD_GRASPING_PLAN.md` for implementation status and
`DUAL_ROBOT_TRANSITION_SYMMETRY_PLAN.md` for the symmetry/frame contract.

## Architecture And Artifact Flow

The single-object path is:

```text
Fabrica OBJ + assembly metadata + YAML
              |
              v
stage 1: object-local grasp generation and assembly filtering
              |
              v
stage 2: execution-pose floor filtering and scoring
              |
              v
saved stage-2 bundle
       |             |              |
       v             v              v
    MuJoCo         Isaac         real MoveIt
```

MuJoCo, Isaac, and the real executor consume the same saved stage-2 bundle;
do not create backend-specific grasp serialization.

The dual-arm path is:

```text
Fabrica assembly order, pre-insertion poses, meshes, and symmetries
              |
              v
dual Stage 0: explicit partial-assembly sequence
              |
              v
dual Stage 1: reusable holder-grasp library
              |
              v
dual Stage 2: holder feasibility in every assembly state
              |
              v
dual Stage 3: incoming grasps + collision-checked grasp pairs
              |       + symmetry-equivalent transition corridors
              v
runtime layout ranking -> MoveIt IK/path fallback -> Isaac or guarded real run
```

For `plumbers_block`, the selected order is `2 -> 0 -> 3 -> 1 -> 4`.
Part `2` is the initial base. Holder-active insertion steps therefore use
incoming parts `0`, `3`, `1`, and `4`.

## Installation

Install the Python package and test dependencies:

```bash
python3 -m pip install -e ".[test]"
```

Simulation and real-robot planning additionally require ROS2 and a compatible
MoveIt workspace. The current KUKA launchers default to ROS Humble and
`/home/pdz/lbr-stack`. Isaac commands default to
`/media/pdz/Elements1/IsaacLab/isaaclab.sh`; override `--lbr-ws` or
`--isaac-python` when using another installation.

Fetch the pinned ROS2 message dependency and build the repository overlay:

```bash
bash scripts/download_ros2_dependencies.sh

source /opt/ros/humble/setup.bash
source /home/pdz/lbr-stack/install/setup.bash
cd ros2_ws
colcon build --packages-select fp_debug_msgs robot_integration_ros --symlink-install
cd ..
```

Source the dual-arm environment helper in terminals where commands are run
manually:

```bash
source ./setup_dual_robot_env.sh
```

The one-command dual runner sources the standard locations itself.

## Single-Object Quick Start

The user-facing single-object entrypoint has three modes:

- `sim`: take the execution-world pose from YAML, then optionally execute in
  MuJoCo and/or Isaac;
- `pitl`: wait for a ROS2 perception pose, then optionally execute in MuJoCo
  and/or Isaac; and
- `real`: wait for the same ROS2 pose, write the same planning artifacts, and
  optionally execute on hardware.

```bash
./run_pipeline.sh --mode sim
./run_pipeline.sh --mode pitl
./run_pipeline.sh --mode real
./run_pipeline.sh --mode sim --headless
./run_pipeline.sh --mode sim --backend isaac --headless
```

Default configs are `configs/grasp_pipeline_sim.yaml`,
`configs/grasp_pipeline_pitl.yaml`, and `configs/grasp_pipeline_real.yaml`.
Use `--backend {config,mujoco,isaac,both,none}` to override simulation backend
selection for one run.

`sim` and `pitl` both run stages 1 and 2 before executing the resulting
stage-2 bundle. `real` writes the same artifacts and executes only when
`real_execution.enabled: true`. The checked-in real config remains safe by
default: execution is disabled, confirmation is required, motion stops at
pregrasp, and gripper actuation is disabled.

The default sim config reproduces corrected KUKA execution-benchmark run 3:
`plumbers_block/0` at stable `orientation_002`. Start
`./start_lbr_moveit.sh` in another terminal, then run
`./run_pipeline.sh --mode sim`. Pass `--headless` when the Isaac GUI is not
wanted.

For `pitl` and `real`, the planning local frame is the OBJ frame translated by
the arithmetic mean of all OBJ vertices. A matching
`fp_debug_msgs/msg/DebugPoseItem.pose_base` is treated as the world pose of
that centroid-centered frame.

## Dual-Arm Quick Start

### 1. Build the offline planning artifacts

Run the four dual stages in order after changing meshes, assembly metadata,
grasp settings, pair settings, or symmetry settings:

```bash
python3 scripts/build_assembly_sequence.py --assembly plumbers_block
python3 scripts/build_holder_grasp_library.py \
  --config configs/dual_grasp_planning.yaml
python3 scripts/build_holder_state_feasibility.py \
  --config configs/dual_grasp_planning.yaml
python3 scripts/build_dual_grasp_pairs.py \
  --config configs/dual_grasp_planning.yaml
```

`build_dual_grasp_pairs.py` can build/load the earlier stages, but listing all
four commands makes the dependency order explicit. Outputs are written to
`artifacts/dual_grasp_planning/plumbers_block/`. The most useful entrypoints
are:

- `assembly_sequence.html` for the authored sequence and insertion poses;
- `holder_base_candidates.html` for reusable base grasps;
- `holder_validity_matrix.html` for per-state holder feasibility;
- `dual_grasp_pair_summary.html` for per-step holder/inserter pairs; and
- `dual_robot_pair_score_debug.html` after running
  `python3 scripts/build_dual_robot_pair_score_debug.py`.

Stage-3 JSON files contain `transition_symmetry.candidates`. Regenerate these
files after enabling or changing transition symmetry; the runtime runner does
not silently rebuild stale Stage-3 artifacts.

### 2. Run the dual-arm Isaac vertical slice

For the first holder-active step, run:

```bash
./run_simple_dual_robot.sh \
  --mode sim \
  --assembly plumbers_block \
  --incoming-part-id 0
```

This one command starts the shared dual-arm mock MoveIt stack, ranks fresh
pair/transition tasks for the requested layout, performs exact IK preflight,
plans the full target sequence with fallback, starts Isaac, and stops the
MoveIt stack when finished. Add `--headless` for a non-GUI run or
`--record-video /tmp/dual_part_0.mp4` to save a video.

Do not pass `--holder-only` when testing transition symmetry; that option
intentionally skips the inserter transport and pre-insertion phases.

The other selected-order incoming parts are `3`, `1`, and `4`:

```bash
./run_simple_dual_robot.sh --mode sim --incoming-part-id 3
./run_simple_dual_robot.sh --mode sim --incoming-part-id 1
./run_simple_dual_robot.sh --mode sim --incoming-part-id 4
```

The plan and attempt artifacts are written next to the Stage-3 artifacts. For
part `0`, inspect the chosen grasp pair and transition with:

```bash
jq '{
  pair_id,
  transition_id,
  is_identity: .transition_symmetry.is_identity,
  insertion_vector: .transition_symmetry.pre_to_final_translation_assembly_m,
  transition_motion_score,
  transition_motion_components
}' artifacts/dual_grasp_planning/plumbers_block/simple_dual_robot_sim_plan_part_0.json
```

The selected pickup grasp retains its parent and symmetry provenance. Once the
gripper closes, its part-to-TCP transform is fixed: transition fallback may
choose another compatible destination corridor, but cannot silently change
the grasp or regrasp the part.

### 3. Inspect or run MoveIt separately

To keep MoveIt running for RViz or repeated planner calls:

```bash
./start_dual_lbr_moveit.sh --mode mock --rviz
```

In another terminal:

```bash
source ./setup_dual_robot_env.sh
python3 scripts/smoke_test_dual_lbr_moveit.py
./run_simple_dual_robot.sh --mode sim --reuse-moveit --incoming-part-id 0
```

The shared MoveIt model contains `arm_one`, `arm_two`, and `both_arms` in one
planning scene, so cross-arm collisions remain enabled. Physical role mapping
is `lbr_one` at `Y=-0.42 m` as holder and `lbr_two` at `Y=+0.42 m` as inserter.

### 4. Real dual-arm safety boundary

Without `--execute`, real mode performs only non-moving target IK checks:

```bash
./run_simple_dual_robot.sh --mode real --incoming-part-id 0
```

Hardware execution requires the correctly configured hardware MoveIt stack,
explicit `--execute`, confirmation unless `--yes` is supplied, and currently
`--allow-objectless-planning` because the simple runtime scene does not yet
contain exact object collision meshes. The latter is a known safety limitation,
not a convenience flag. Review `./run_simple_dual_robot.sh --help` and the
KUKA hardware runbook below before enabling motion.

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

The same entrypoint also exposes a dual holder/inserter adapter for the
validated first `plumbers_block` step. Start the shared dual MoveIt stack
separately, then run either perception-in-the-loop Isaac:

```bash
ros2 run robot_integration_ros grasp_assembly_action_server \
  --dual-mode pitl \
  --config configs/dual_grasp_planning.yaml \
  --headless
```

or guarded real execution:

```bash
ros2 run robot_integration_ros grasp_assembly_action_server \
  --dual-mode real \
  --config configs/dual_grasp_planning.yaml \
  --execute \
  --allow-objectless-planning
```

The dual adapter uses both base and insertion `DebugPoseItem` poses and all goal
fields. Its current validated role mapping is `holder_robot: left` (`lbr_one`)
and `inserter_robot: right` (`lbr_two`). It reaches pre-insertion but does not
perform insertion, release, or retreat.

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

## Development And Verification

Run the same Python checks used by CI:

```bash
python3 -m pip install -e ".[dev,test]"
pre-commit run --all-files
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

The FCL-backed collision tests require the native `libccd`/`libfcl` packages
and `python-fcl`. ROS2-independent tests use mocks where possible; the dual
MoveIt smoke test is separate because it requires a live shared MoveIt stack:

```bash
./start_dual_lbr_moveit.sh --mode mock
# In a second sourced terminal:
python3 scripts/smoke_test_dual_lbr_moveit.py
```

## Repo Shape

The main source areas are:

- `run_pipeline.sh` - single-object `sim`, `pitl`, and `real` wrapper;
- `run_simple_dual_robot.sh` - one-command dual-arm sim/real vertical slice;
- `start_lbr_moveit.sh` and `start_dual_lbr_moveit.sh` - single/shared MoveIt
  launchers;
- `configs/` - single-object, benchmark, backend, and dual-arm YAML settings;
- `grasp_planning/grasping/` - mesh loading, antipodal generation, scoring,
  transforms, and gripper collision geometry;
- `grasp_planning/pipeline/` - single-object stages, regrasp planning, dual
  assembly stages, pair scoring, and transition symmetry;
- `grasp_planning/planning/` - backend-neutral execution data and helpers;
- `grasp_planning/mujoco/` and `grasp_planning/envs/` - MuJoCo and Isaac
  execution support;
- `grasp_planning/ros2/` - perception, MoveIt, gripper, multi-IK, and guarded
  real execution adapters;
- `scripts/` - build, run, benchmark, debug, and model-generation commands;
- `ros2_ws/src/robot_integration_ros/` - ROS2 package, launch descriptions,
  MoveIt configuration, and action-server entrypoints;
- `assets/obj/fabrica/` - per-assembly OBJ meshes, precedence plans,
  pre-insertion poses, and finite symmetry assets;
- `assets/urdf/` and `assets/usd/` - robot and gripper models; and
- `tests/` - unit/regression coverage, including ROS-independent MoveIt mocks.

Deep dual-arm implementation notes live in
`DUAL_ROBOT_HOLD_GRASPING_PLAN.md`; the shorter
`DUAL_ROBOT_TRANSITION_SYMMETRY_PLAN.md` is the authoritative explanation of
the two symmetry boundaries. `KUKA_dual_arm_CHEATSHEET.md` is an operational
lab reference, not the architecture source of truth.

## Notes

- The default Fabrica OBJ scale in the pipeline configs is `0.01`.
- The MuJoCo runner uses the exact `execution_world_pose` stored in the stage-2 bundle unless you override placement explicitly.
- `pitl` and `real` use one ROS2 subscriber: set `ros2.pose_base_topic`, `ros2.assembly_name`, and numeric `ros2.part_id` in the pipeline YAML before running those modes. The configured topic must publish `fp_debug_msgs/msg/DebugPoseItem`. Optional `ros2.position_offset_m: [x, y, z]` adds a fixed world-axis translation to the received pose before planning; the left KUKA real config currently uses `[0.0, -0.840, 0.0]`.
