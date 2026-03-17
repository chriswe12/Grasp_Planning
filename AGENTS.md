# AGENTS.md

## Project Summary

This repository is for task-aware grasp planning on Franka Research 3 using Isaac Sim / Isaac Lab.

Current implemented scope:
- environment setup only,
- single FR3 robot in scene,
- ground plane,
- single dynamic cube with fixed pose defined in the launcher,
- no robot control, grasp execution, motion planning, or perception pipeline yet.

Primary entrypoint:
- `scripts/launch_fr3_cube_env.py`

Environment config:
- `grasp_planning/envs/fr3_cube_env.py`

## Current Environment Behavior

- The launcher uses Isaac Sim's built-in FR3 USD by default.
- Default FR3 asset path resolves through Isaac assets root to:
  `Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd`
- The cube pose is hard-coded in `scripts/launch_fr3_cube_env.py`.
- The FR3 is currently loaded as a scene USD asset, not as a controlled articulation interface.
- This was done intentionally because the articulation-based setup was causing the app to shut down during `sim.reset()`.

## Docker Setup

Main files:
- `Dockerfile`
- `docker_env.sh`

Important implementation details:
- Base image: `nvcr.io/nvidia/isaac-sim:5.1.0`
- Isaac Lab install: `isaaclab==2.3.2.post1` with `--no-deps`
- Python inside container should use Isaac's interpreter:
  `/isaac-sim/python.sh`

Important lessons:
- Do not install `isaaclab[isaacsim]` on top of the Isaac Sim container image.
  - That tries to download the pip-packaged Isaac Sim stack again.
  - It triggers huge downloads such as `isaacsim-extscache-kit` and is slow and fragile.
- Do not upgrade pip/setuptools/wheel inside the Isaac Sim container unless necessary.
  - Doing so previously broke Isaac Sim's bundled Torch environment.
- `USER root` is required before `apt-get` in this base image.
- Ubuntu 24.04 package naming matters in the Isaac Sim 5.1 image.
  - `libasound2t64` is correct there, not `libasound2`.
- Keep Docker layers cacheable.
  - Failed layers are not cached.
  - `--no-cache-dir` makes retries slower because pip cannot reuse downloads.

## GUI / X11 Learnings

The container can run GUI mode, but X11 auth must be correct.

Required pieces:
- `DISPLAY`
- `/tmp/.X11-unix` mount
- Xauthority file mount
- `XAUTHORITY` inside container

`docker_env.sh` already handles:
- mounting repo root to `/workspace/Grasp_Planning`,
- mounting `/tmp/.X11-unix`,
- mounting Xauthority file to `/tmp/.docker.xauth`,
- exporting `DISPLAY`,
- exporting `XAUTHORITY=/tmp/.docker.xauth` when available.

Host command typically needed before GUI launch:
- `xhost +SI:localuser:root`

Why:
- the container runs as `root`,
- X11 access control otherwise blocks the GUI client even when the socket is mounted.

You can revoke access later with:
- `xhost -SI:localuser:root`

Useful GUI diagnostics inside the container:
- `echo $DISPLAY`
- `echo $XAUTHORITY`
- `ls /tmp/.X11-unix`
- `xeyes`

If `xeyes` cannot open a window, Isaac GUI will not work either.

## Isaac Lab Runtime Learnings

- Import Isaac/Omniverse modules only after creating `SimulationApp` through `AppLauncher`.
- Use `/isaac-sim/python.sh` for all launcher execution inside the container.
- The launcher currently sets:
  - `sim._app_control_on_stop_handle = None`
  - `sim._disable_app_control_on_stop_handle = True`
  to reduce standalone app lifecycle interference.
- The launcher also waits for stage loading to finish before `sim.reset()`.

## Important Failure History

These issues already happened and should not be reintroduced without a reason:

1. Installing `isaaclab[isaacsim]` in the Isaac Sim image
- caused redundant multi-GB downloads,
- often timed out on `pypi.nvidia.com`.

2. Upgrading pip/setuptools/wheel in the image
- broke bundled Torch packaging files inside Isaac Sim.

3. Missing Xauthority mount
- caused GUI startup with no usable default window,
- app opened and then shut down immediately.

4. FR3 as manual `ArticulationCfg`
- app shut down during `sim.reset()`.
- A simpler USD scene asset path is stable for the current environment-only stage.

## Recommended Commands

Build image:

```bash
./docker_env.sh build
```

Run container with GUI support:

```bash
xhost +SI:localuser:root
./docker_env.sh run
```

Inside container:

```bash
export PYTHONPATH=/workspace/Grasp_Planning:/isaac-sim/kit/python/lib/python3.11/site-packages/isaaclab/source/isaaclab
/isaac-sim/python.sh scripts/launch_fr3_cube_env.py
```

Headless run:

```bash
export PYTHONPATH=/workspace/Grasp_Planning:/isaac-sim/kit/python/lib/python3.11/site-packages/isaaclab/source/isaaclab
/isaac-sim/python.sh scripts/launch_fr3_cube_env.py --headless
```

Timed run:

```bash
/isaac-sim/python.sh scripts/launch_fr3_cube_env.py --run-seconds 30
```

## Repo Notes

- Do not assume controller or grasp-generation code is part of the stable environment path.
- Keep environment changes isolated from controller development when possible.
- Avoid committing `__pycache__` content.
