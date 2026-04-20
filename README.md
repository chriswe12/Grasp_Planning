# Grasp Planning

YAML-driven Fabrica grasp planning with three pipeline modes behind one entrypoint:
- `sim`: offline execution-world pose from config, then MuJoCo execution
- `pitl`: ROS2 perception pose intake, then MuJoCo execution
- `real`: ROS2 perception pose intake, planning only

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

`sim` and `pitl` both run stage 1, write stage-1 artifacts, run stage 2, write stage-2 artifacts, then execute the selected grasp in MuJoCo from the stage-2 bundle. `real` stops after writing the planning artifacts.

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
