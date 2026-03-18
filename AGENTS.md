# AGENTS.md

## Purpose

This repository is for task-aware grasp planning on Franka Research 3 in Isaac Sim / Isaac Lab.

Current stable scope:
- scene setup only,
- FR3 in scene,
- ground plane,
- one dynamic cube with fixed pose from the launcher,
- no controller, motion planning, or grasp execution in the stable launch path.

## Main Files

- `scripts/launch_fr3_cube_env.py`
- `grasp_planning/envs/fr3_cube_env.py`
- `Dockerfile`
- `docker_env.sh`

## Environment Notes

- The launcher uses Isaac Sim's built-in FR3 asset by default.
- The cube pose is defined in `scripts/launch_fr3_cube_env.py`.
- The current stable setup loads the FR3 as a scene USD asset.
- Keep environment work separate from controller work when possible.

## Docker Notes

- Base image: `nvcr.io/nvidia/isaac-sim:5.1.0`
- Install Isaac Lab on top of that image with:
  `isaaclab==2.3.2.post1`
- Use Isaac's Python inside the container:
  `/isaac-sim/python.sh`
- The repo root is mounted inside the container at:
  `/workspace/Grasp_Planning`

Do not install `isaaclab[isaacsim]` on top of the Isaac Sim container image.

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
