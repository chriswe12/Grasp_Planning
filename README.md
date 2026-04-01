# Grasp_Planning

Minimal Isaac Lab scaffold for task-aware grasp planning experiments on Franka Research 3.

Current scope:
- interactive FR3 + cube environment,
- ground plane + dynamic cube + FR3 scene,
- experimental move-to-pose and pickup debug paths in the launcher,
- a standalone teleport-based pickup debug script for isolating path-planning issues,
- a separate minimal object-frame antipodal grasp generator for procedural mesh and STL debug,
- pickup is now possible in sim with tuning, but the stack is still experimental and not yet robust.

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

Standalone mesh antipodal grasp debug viewer:

```bash
python scripts/debug_mesh_antipodal_grasps.py --geometry cube
python scripts/debug_mesh_antipodal_grasps.py --geometry cylinder
python scripts/debug_mesh_antipodal_grasps.py --geometry stl --stl-path my_part.stl --stl-scale 0.001
```

This viewer is also browser-based. It writes a self-contained HTML file to
`artifacts/mesh_antipodal_grasp_debug.html`; open that file from the host browser.
Relative STL paths are resolved under `assets/stl/`.

Docker build:

```bash
./docker_env.sh build
```

Docker run for containerized Isaac execution:

```bash
./docker_env.sh run
```

This opens an interactive shell inside the container immediately.

Inside the container:

```bash
/isaac-sim/python.sh scripts/launch_fr3_cube_env.py --headless
```

To run the new Isaac-side admittance controller instead of the joint-space planner:

```bash
/isaac-sim/python.sh scripts/launch_fr3_cube_env.py --controller admittance --headless
```

Standalone teleport-based pickup debug path:

```bash
/isaac-sim/python.sh scripts/teleport_fr3_pickup.py --headless
```

FR3 TCP geometry inspection from the spawned asset:

```bash
/isaac-sim/python.sh scripts/inspect_fr3_tcp_geometry.py --headless
```

Systematic top-grasp diagnosis for offline IK vs controller tracking:

```bash
/isaac-sim/python.sh scripts/diagnose_fr3_top_grasp.py --headless --baselines-only
```

For GUI mode inside the container:

```bash
/isaac-sim/python.sh scripts/launch_fr3_cube_env.py
```

To run for a fixed duration instead of until interrupted:

```bash
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
- `Solid Mesh` toggles between wireframe and filled mesh rendering,
- arrow keys or `Prev` / `Next` switch the selected grasp.

Mesh antipodal grasp path:
- lives separately from the existing cube-face grasp generator,
- uses object geometry only and returns grasps in the object frame,
- samples surface points and normals on a triangle mesh,
- uses a KD-tree to find nearby sampled contact pairs within the jaw-width limit,
- applies `max_pair_checks` after that KD-tree preselection,
- filters on jaw width, antipodal consistency, coarse finger-box collision, and a Franka hand-mesh palm check,
- evaluates gripper collision per rolled grasp pose, not once per unrolled contact pair,
- uses an FCL-backed `trimesh` collision scene built once per `generate(mesh)` call,
- can export typed grasp candidates with pose, contacts, normals, and jaw width.

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
- `scripts/debug_mesh_antipodal_grasps.py` is a separate local viewer for the new object-frame antipodal grasp path and supports procedural cube/cylinder meshes plus STL input from `assets/stl/`,
- the new mesh grasp generator lives under `grasp_planning/grasping/mesh_antipodal_grasp_generator.py` and is intentionally separate from the existing cube-face path,
- by default the launcher uses Isaac Sim's built-in FR3 asset:
  `Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd`,
- `--fr3-usd` is optional and only needed to override that default,
- later controller work can replace the hard-coded cube pose with an externally provided object pose,
- the launcher now spawns the FR3 as an `ArticulationCfg` and includes an experimental grasp controller path,
- `--controller admittance` uses an Isaac-only Cartesian admittance loop adapted from the upstream ROS2/libfranka controller,
- the best current pickup path is the launcher with `--controller admittance`; it can pick up the cube in sim, but is still sensitive to gains and not yet robust,
- the standalone teleport script is the cleanest way to debug grasp geometry because it bypasses arm path planning,
- the fixed `fr3_hand_tcp -> finger-midpoint` offset is verified from the spawned Isaac asset as approximately `(0, 0, -0.045)`,
- the main residual error found during debugging was low-level arm joint tracking under load, not the TCP offset or the offline IK solve,
- the Dockerfile is based on `nvcr.io/nvidia/isaac-sim:5.1.0` and installs the minimal Isaac Lab `2.3.2.post1` runtime needed for this repo on top of Isaac Sim,
- `docker_env.sh` mounts the repo root to `/workspace/Grasp_Planning` inside the container,
- the container exports `PYTHONPATH` automatically for the mounted workspace and Isaac Lab source tree,
- for GUI mode, `docker_env.sh run` grants the container's root user temporary X11 access with `xhost +SI:localuser:root` when `DISPLAY` and `xhost` are available,
- that X11 access is revoked automatically when `docker_env.sh run` exits.
## Mesh Antipodal Grasp Debug

The arbitrary-object grasp debug path is driven by `scripts/debug_mesh_antipodal_grasps.py`.

Default settings live in `configs/mesh_antipodal_grasp_debug.yaml`. Run with the default config:

```bash
python scripts/debug_mesh_antipodal_grasps.py
```

To evaluate an STL from `assets/stl/`, either edit the YAML file or override selected values on the command line:

```bash
python scripts/debug_mesh_antipodal_grasps.py --geometry stl --stl-path my_part.stl --stl-scale 0.001
```

CLI flags override the YAML values for that run only.

Finger collision dimensions use semantic names:
- `finger_extent_lateral`
- `finger_extent_closing`
- `finger_extent_approach`

Matching CLI overrides are:
- `--finger-extent-lateral`
- `--finger-extent-closing`
- `--finger-extent-approach`

Legacy YAML keys `finger_depth`, `finger_length`, and `finger_thickness` still map to the new semantic fields when the new keys are absent. Legacy CLI flags with those names are also accepted as aliases for per-run overrides.

The mesh antipodal debug path now requires `trimesh` with FCL support (`python-fcl`) for collision checking. In Docker this is provided by the repo `Dockerfile`; on a host install you need the native FCL libraries plus `trimesh` / `python-fcl`.

For roll sampling in YAML, set `generator.roll_step_deg`.
This generates roll samples at `0, step, 2*step, ...` up to but excluding `360`.
Use `360` for a single `0 deg` sample.
For per-run overrides, `--roll-angles-rad` still works from the CLI.
Do not rely on legacy YAML `roll_angles_deg` or `roll_angles_rad` keys while `roll_step_deg` is present, because the merged config currently gives `roll_step_deg` precedence.
The HTML viewer renders the same grasp-frame convention used by the generator collision check:
- local `x`: lateral
- local `y`: closing
- local `z`: approach
- purple: current coarse finger boxes used by the runtime collision filter
- orange/brown: Franka finger boxes and hand mesh overlays for geometry debugging
