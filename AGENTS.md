# AGENTS.md

## Purpose

This repository is for task-aware grasp planning on Franka Research 3 in Isaac Sim / Isaac Lab.

Current scope:
- FR3 + cube scene setup in Isaac Lab,
- one dynamic cube with fixed pose from the launcher,
- a debug pickup path in the launcher with optional `--pregrasp-only`,
- a standalone teleport-based pickup debug script,
- a separate object-frame antipodal grasp debug path for procedural mesh geometry and STL input,
- a separate two-stage Fabrica grasp workflow for offline assembly filtering and pickup-ground recheck,
- a new integrated Fabrica-to-Isaac pickup path that loads saved grasps, converts STL to a simulator asset, rechecks floor feasibility, and executes the first feasible grasp,
- pickup can work in sim with tuned admittance, but the stack is still experimental.

## Main Files

- `scripts/launch_fr3_cube_env.py`
- `scripts/debug_cube_grasps.py`
- `scripts/debug_mesh_antipodal_grasps.py`
- `scripts/generate_fabrica_assembly_grasps.py`
- `scripts/check_fabrica_ground_feasible_grasps.py`
- `scripts/convert_stl_to_usd.py`
- `scripts/run_fabrica_pickup_in_isaac.py`
- `scripts/teleport_fr3_pickup.py`
- `scripts/inspect_fr3_tcp_geometry.py`
- `scripts/diagnose_fr3_top_grasp.py`
- `grasp_planning/envs/fr3_cube_env.py`
- `grasp_planning/envs/fr3_part_env.py`
- `grasp_planning/scene_defaults.py`
- `Dockerfile`
- `docker_env.sh`

## Environment Notes

- The launcher uses Isaac Sim's built-in FR3 asset by default.
- Shared cube and robot defaults live in `grasp_planning/scene_defaults.py`.
- The FR3 is spawned via `ArticulationCfg` from a USD path.
- The default launcher debug flow runs pregrasp, approach, close, and retreat; use `--pregrasp-only` to stop after pregrasp.
- The launcher exposes face selection with `--grasp-face {pos_x,neg_x,pos_y,neg_y,pos_z,neg_z}`, `--pregrasp-offset`, and `--tcp-to-grasp-offset`.
- The fixed TCP offset `(0, 0, -0.045)` was verified from the spawned Isaac asset; do not keep re-tuning it blindly.
- The main residual problem was arm drive tracking under load, not offline IK or TCP frame conversion.
- Keep environment work separate from controller work when possible.
- Keep the mesh antipodal grasp path separate from the existing cube-face grasp path.
- Mesh antipodal grasp defaults now live in `configs/mesh_antipodal_grasp_debug.yaml`; CLI flags should stay as per-run overrides.
- STL files for the mesh antipodal debug path live under `assets/stl/`; relative `--stl-path` values resolve there.
- Fabrica assembly STL files are assumed to already be in shared global coordinates.
- The Fabrica two-stage path saves grasps in the target part-local frame so the offline assembly stage and the pickup-ground stage use the same grasp coordinates.
- The integrated Isaac pickup path also consumes those saved part-local grasps; world-frame execution targets are derived at runtime from the sampled object pose.
- The planner collision checker must fail fast if the robot asset has no collision-enabled `UsdGeom.Gprim`s; do not silently skip scene queries and pretend all states are valid.
- Fabrica contact-offset refinement is part of the saved grasp definition: the stored grasp pose already includes the accepted finger-pad offset, and both Fabrica stages search a 5x5 inset grid on the rubber-tip contact patch.
- Fabrica grasp scoring is geometric-only over already-feasible grasps: antipodal alignment, centering, local contact support, and COM offset. Collision and approach checks stay as upstream hard filters.
- Stage 1 and stage 2 Fabrica HTML viewers are score-sorted; stage 2 renders in the selected pickup world pose, not the canonical saved local frame.
- `scripts/check_fabrica_ground_feasible_grasps.py` accepts `--support-face`, `--yaw-deg`, and `--xy-world` overrides; use those instead of editing `HARDCODED_PICKUP_SPECS` for one-off pose checks.
- `scripts/run_fabrica_pickup_in_isaac.py` does not use `HARDCODED_PICKUP_SPECS`; it requires an explicit pose or samples support face / yaw / XY directly.
- Negative `--xy-world` values must be passed as `--xy-world=-0.2,0.0` or `--xy-world "-0.2,0.0"` so `argparse` does not treat them as flags.
- The MuJoCo grasp path should use the stage-2 bundle pickup pose as the source of truth; only override XY/face/yaw deliberately.
- MuJoCo execution must rebuild the object mesh in the saved bundle-local frame; using the raw assembly-global STL misaligns saved grasps and world targets.
- The mesh antipodal generator now KD-preselects nearby sample pairs within `max_jaw_width`; `max_pair_checks` applies after that preselection, not to the full Cartesian pair set.
- For YAML roll sampling, prefer `generator.roll_step_deg`; do not casually claim legacy `roll_angles_deg` / `roll_angles_rad` YAML compatibility without checking merged-default precedence.
- Mesh antipodal finger collision is evaluated per rolled grasp pose with an FCL-backed `trimesh` scene built once per `generate(mesh)` call.
- The mesh antipodal runtime filter now uses detailed Franka finger boxes plus a Franka hand mesh palm check, with a configurable `generator.detailed_finger_contact_gap_m`.
- Keep the hand-mesh dependency lazily loaded so generator construction and config/debug imports still work without assets or `trimesh`.
- Do not hand-author part USDs for Isaac if you can avoid it; `scripts/convert_stl_to_usd.py` now uses Isaac Lab's `MeshConverter`, which fixed the previous scene-creation shutdown on custom part assets.
- The current remaining integrated-pipeline issue is controller/pregrasp convergence under load; asset loading and offline-to-online grasp bridging are working.
- MuJoCo Menagerie `franka_fr3` / `franka_fr3_v2` are arm-only; use `scripts/build_mujoco_fr3_hand_models.py` to generate the local FR3+Panda-hand XMLs under `.cache/generated_mujoco_models/`.

## Docker Notes

- Base image: `nvcr.io/nvidia/isaac-sim:5.1.0`
- Install Isaac Lab on top of that image with:
  `isaaclab==2.3.2.post1`
- Use Isaac's Python inside the container:
  `/isaac-sim/python.sh`
- The repo root is mounted inside the container at:
  `/workspace/Grasp_Planning`

Do not install `isaaclab[isaacsim]` on top of the Isaac Sim container image.
- `docker_env.sh run` now checks host GPU runtime availability up front and launches Docker with `--runtime=nvidia`, `--gpus all`, and explicit NVIDIA capability env vars.

## GUI Notes

GUI mode in Docker requires:
- `DISPLAY`
- `/tmp/.X11-unix`
- Xauthority mount

Before GUI launch on the host:

```bash
xhost +SI:localuser:root
```

Inside the container, the common launcher path is:

```bash
/isaac-sim/python.sh scripts/launch_fr3_cube_env.py
```

## General Guidance

- Import Isaac/Omniverse modules only after creating the app through `AppLauncher`.
- Avoid committing `__pycache__` files.
- Keep stable infrastructure changes isolated from experimental controller or grasping code.
