# AGENTS.md

## Purpose

This repository is for the YAML-driven Fabrica grasp-planning pipeline on Franka Research 3 with MuJoCo execution and ROS2 pose intake.

Current scope:
- one user-facing entrypoint: `run_pipeline.sh`
- three pipeline modes: `sim`, `pitl`, `real`
- shared stage-1 and stage-2 Fabrica planning in `grasp_planning/pipeline/`
- MuJoCo execution from the stage-2 bundle in `scripts/run_fabrica_grasp_in_mujoco.py`
- ROS2 pose intake in `grasp_planning/ros2/`
- MuJoCo robot model generation from Menagerie assets in `scripts/build_mujoco_fr3_hand_models.py`

## Main Files

- `run_pipeline.sh`
- `scripts/run_grasp_pipeline.py`
- `scripts/run_fabrica_grasp_in_mujoco.py`
- `scripts/build_mujoco_fr3_hand_models.py`
- `scripts/download_required_assets.sh`
- `grasp_planning/pipeline/fabrica_pipeline.py`
- `grasp_planning/grasping/fabrica_grasp_debug.py`
- `grasp_planning/grasping/mesh_antipodal_grasp_generator.py`
- `grasp_planning/mujoco/runner.py`
- `grasp_planning/mujoco/scene_builder.py`
- `grasp_planning/ros2/pose_listener.py`

## Environment Notes

- `run_pipeline.sh` should resolve `PIPELINE_PYTHON`, then `python3`, then `python`.
- `sim` uses `execution_world_pose` from YAML and executes in MuJoCo.
- `pitl` waits on ROS2 topics, writes stage artifacts, then executes in MuJoCo.
- `real` uses the same ROS2 intake path but remains planning-only.
- The MuJoCo path consumes the stage-2 bundle as the source of truth.
- The MuJoCo object mesh must be rebuilt in the saved bundle-local frame before execution.
- The vendored Franka hand collision mesh lives at `assets/urdf/franka_description/meshes/robot_ee/franka_hand_black/collision/hand.stl`.
- MuJoCo Menagerie `franka_fr3` is arm-only; use `scripts/build_mujoco_fr3_hand_models.py` to generate the local FR3+Panda-hand XML under `.cache/generated_mujoco_models/`.

## General Guidance

- Keep changes aligned with the three pipeline modes only.
- Do not reintroduce the retired simulator stack, its configs, or its container setup.
- Prefer updating `configs/mujoco_simulation.yaml` when exposing new MuJoCo tuning knobs.
- Avoid committing `__pycache__` files.
