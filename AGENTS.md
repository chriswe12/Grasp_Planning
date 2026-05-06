# AGENTS.md

## Purpose

This repository is for the YAML-driven Fabrica grasp-planning pipeline on Franka Research 3 with MuJoCo or Isaac execution and ROS2 pose intake.

Current scope:
- one user-facing entrypoint: `run_pipeline.sh`
- repo-specific Isaac container helper: `docker_env.sh`
- three pipeline modes: `sim`, `pitl`, `real`
- shared stage-1 and stage-2 Fabrica planning in `grasp_planning/pipeline/`
- MuJoCo execution from the stage-2 bundle in `scripts/run_fabrica_grasp_in_mujoco.py`
- Isaac execution from the stage-2 bundle in `scripts/run_fabrica_grasp_in_isaac.py`
- ROS2 pose intake in `grasp_planning/ros2/`
- optional real-robot execution from the stage-2 bundle in `grasp_planning/ros2/real_grasp_executor.py`
- internal ROS2 workspace for hardware-facing nodes in `ros2_ws/src/robot_integration_ros/`
- MuJoCo robot model generation from Menagerie assets in `scripts/build_mujoco_fr3_hand_models.py`

## Main Files

- `run_pipeline.sh`
- `docker_env.sh`
- `Dockerfile`
- `scripts/run_grasp_pipeline.py`
- `scripts/run_fabrica_grasp_in_mujoco.py`
- `scripts/run_fabrica_grasp_in_isaac.py`
- `scripts/convert_stl_to_usd.py`
- `scripts/build_mujoco_fr3_hand_models.py`
- `scripts/download_required_assets.sh`
- `grasp_planning/pipeline/fabrica_pipeline.py`
- `grasp_planning/mujoco/runner.py`
- `grasp_planning/mujoco/scene_builder.py`
- `grasp_planning/ros2/pose_listener.py`
- `grasp_planning/ros2/real_grasp_executor.py`
- `grasp_planning/ros2/franka_gripper_client.py`
- `grasp_planning/ros2/moveit_pose_commander.py`
- `ros2_ws/src/robot_integration_ros/robot_integration_ros/move_real_robot_ee.py`

## Environment Notes

- `run_pipeline.sh` should resolve `PIPELINE_PYTHON`, then `python3`, then `python`.
- `sim` uses `execution_world_pose` from YAML and can execute in MuJoCo and/or Isaac.
- `pitl` waits on ROS2 topics, writes stage artifacts, then can execute in MuJoCo and/or Isaac.
- `run_pipeline.sh --backend {config,mujoco,isaac,both,none}` overrides sim/pitl execution backend for one run.
- `configs/grasp_pipeline_sim_isaac.yaml` and `configs/grasp_pipeline_pitl_isaac.yaml` are Isaac-only convenience configs with MuJoCo disabled.
- `real` uses the same ROS2 intake path, writes the same stage artifacts, and can optionally execute on hardware when `real_execution.enabled: true`.
- Planning knobs: `roll_angle_step_deg` expands a full 360 degree roll sweep; `floor_clearance_margin_m` and `top_grasp_score_weight` are stage-2 world-pose filters/scorers; `--skip-stage1-collision-checks` bypasses only stage-1 assembly collision filtering.
- The MuJoCo path consumes the stage-2 bundle as the source of truth.
- The Isaac path consumes the stage-2 bundle as the source of truth.
- The real-robot path also consumes the stage-2 bundle as the source of truth; do not create a second grasp serialization path.
- The MuJoCo object mesh must be rebuilt in the saved bundle-local frame before execution.
- MuJoCo can optionally use MoveIt for planning only via `mujoco_execution.controller: "moveit"`; MuJoCo still executes the planned joint waypoints and owns physics/viewer/contact evaluation.
- MuJoCo regrasp fallback is geometry-filtered first, then MoveIt-ranked at execution time: do not choose staging poses only by static placement score when MoveIt trajectories are available.
- Regrasp fallback artifacts are split: `*_regrasp_plan.json/html` explain candidate resting poses and grasps; the MuJoCo attempt artifact records ranked `planned_candidates`, execution `attempts`, and trajectory diagnostics.
- Do not add MoveIt planning to Isaac execution without an explicit request. The supported Isaac controllers are currently `admittance` and `planner`.
- Isaac execution generates a collision-enabled bundle-local USD from the stage-2 bundle by default; only use a provided USD if it is already authored in the saved bundle-local frame.
- The vendored Franka hand collision mesh lives at `assets/urdf/franka_description/meshes/robot_ee/franka_hand_black/collision/hand.stl`.
- MuJoCo Menagerie `franka_fr3` is arm-only; use `scripts/build_mujoco_fr3_hand_models.py` to generate the local FR3+Panda-hand XML under `.cache/generated_mujoco_models/`.
- Hardware-facing ROS2 code depends on an external FR3 / MoveIt workspace being sourced before running repo-local ROS2 nodes.
- `run_pipeline.sh` resets ROS discovery to `ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}` and clears localhost-only/static discovery unless `GRASP_KEEP_ROS_DISCOVERY_ENV=1`.
- Keep `configs/grasp_pipeline_real.yaml` safe by default: `real_execution.enabled: false`, `require_confirmation: true`, `stop_after: "pregrasp"`, `gripper_enabled: false`.

## General Guidance

- Keep changes aligned with the three pipeline modes only.
- Do not reintroduce the retired simulator stack, its configs, or its container setup.
- Prefer updating `configs/mujoco_simulation.yaml` when exposing new MuJoCo tuning knobs.
- Prefer updating the `isaac_execution` block in the sim/pitl configs when exposing Isaac tuning knobs.
- Prefer updating the `real_execution` block in `configs/grasp_pipeline_real.yaml` when exposing hardware execution knobs.
- Avoid committing `__pycache__` files.
