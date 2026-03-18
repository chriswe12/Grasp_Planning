# Grasp_Planning

Minimal Isaac Lab scaffold for task-aware grasp planning experiments on Franka Research 3.

Current scope:
- interactive environment only,
- ground plane + dynamic cube + FR3 scene,
- no controller or grasp execution yet.

Main entrypoint:

```bash
python scripts/launch_fr3_cube_env.py
```

Standalone grasp debug viewer:

```bash
python scripts/debug_cube_grasps.py
```

The debug viewer is browser-based. It writes a self-contained HTML file to
`artifacts/cube_grasp_debug.html`; open that file from the host browser.

Docker build:

```bash
./docker_env.sh build
```

Docker run for containerized Isaac execution:

```bash
./docker_env.sh run
```

Inside the container:

```bash
export PYTHONPATH=/workspace/Grasp_Planning:/isaac-sim/kit/python/lib/python3.11/site-packages/isaaclab/source/isaaclab
/isaac-sim/python.sh scripts/launch_fr3_cube_env.py --headless
```

For GUI mode inside the container:

```bash
export PYTHONPATH=/workspace/Grasp_Planning:/isaac-sim/kit/python/lib/python3.11/site-packages/isaaclab/source/isaaclab
/isaac-sim/python.sh scripts/launch_fr3_cube_env.py
```

To run for a fixed duration instead of until interrupted:

```bash
export PYTHONPATH=/workspace/Grasp_Planning:/isaac-sim/kit/python/lib/python3.11/site-packages/isaaclab/source/isaaclab
/isaac-sim/python.sh scripts/launch_fr3_cube_env.py --run-seconds 30
```

To inspect the deterministic cube grasps with a selectable ranked list:

```bash
python scripts/debug_cube_grasps.py --cube-position 0.45,0.0,0.025 --cube-orientation-xyzw 0,0,0,1
```

Viewer controls:
- left mouse drag rotates the scene,
- middle mouse drag pans the scene,
- mouse wheel zooms,
- arrow keys or `Prev` / `Next` switch the selected grasp.

Current grasp convention for the cube generator:
- each candidate represents a symmetric parallel-jaw pinch grasp,
- `position_w` is the cube-center pinch midpoint,
- the selected face label (`+x`, `-y`, etc.) determines the approach side and gripper orientation,
- the debug viewer derives the two finger locations from that grasp pose and `gripper_width`.

To override the built-in Isaac FR3 asset URL with another USD:

```bash
/isaac-sim/python.sh scripts/launch_fr3_cube_env.py --fr3-usd /absolute/path/to/fr3.usd
```

Container lifecycle helpers:

```bash
./docker_env.sh stop
./docker_env.sh remove
```

Host compatibility checks used for the Docker setup:
- architecture: `x86_64`,
- OS: Ubuntu 22.04.5,
- NVIDIA driver module: `570.211.01`,
- Docker: `29.1.3`,
- NVIDIA Container Toolkit: `1.18.1`.

Notes:
- the cube pose is defined directly in `scripts/launch_fr3_cube_env.py`,
- `scripts/debug_cube_grasps.py` is intended for local debug visualization and writes generated output into `artifacts/`,
- by default the launcher uses Isaac Sim's built-in FR3 asset:
  `Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd`,
- `--fr3-usd` is optional and only needed to override that default,
- later controller work can replace the hard-coded cube pose with an externally provided object pose,
- the launcher now spawns the FR3 as an `ArticulationCfg` and includes an experimental grasp controller path,
- the current planner / pickup path does not work reliably yet; the arm moves, but the cube is not picked successfully,
- the Dockerfile is based on `nvcr.io/nvidia/isaac-sim:5.1.0` and installs the minimal Isaac Lab `2.3.2.post1` runtime needed for this repo on top of Isaac Sim,
- `docker_env.sh` mounts the repo root to `/workspace/Grasp_Planning` inside the container,
- for GUI mode, `docker_env.sh run` grants the container's root user temporary X11 access with `xhost +SI:localuser:root` when `DISPLAY` and `xhost` are available,
- that X11 access is revoked automatically when `docker_env.sh run` exits.
