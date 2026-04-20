# Grasp_Planning

Minimal Isaac Lab scaffold for task-aware grasp planning experiments on Franka Research 3.

Current scope:
- interactive FR3 + cube environment,
- ground plane + dynamic cube + FR3 scene,
- experimental move-to-pose and pickup debug paths in the launcher,
- a standalone teleport-based pickup debug script for isolating path-planning issues,
- a separate minimal object-frame antipodal grasp generator for procedural mesh and mesh-asset debug,
- a shared Fabrica-style planning pipeline for offline assembly filtering and pickup-ground rechecking,
- a YAML-driven local pipeline path and a ROS2-backed planning-only path behind one bash launcher,
- an integrated Fabrica-to-Isaac pickup path that reloads saved grasps, rechecks floor feasibility for the execution pose, and executes the first feasible grasp,
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

Unified planning entrypoint:

```bash
./run_pipeline.sh --mode local
./run_pipeline.sh --mode real
```

The wrapper defaults to:
- `configs/grasp_pipeline_local.yaml` for `--mode local`
- `configs/grasp_pipeline_real.yaml` for `--mode real`

It prefers `/isaac-sim/python.sh` inside the Isaac container and falls back to `python3` or `python` on the host.

`configs/grasp_pipeline_real.yaml` keeps the legacy single-topic pose listener runnable by default.
If `ros2.object_id`, `ros2.local_frame_offset_topic`, and `ros2.execution_frame_topic` are all set,
real mode switches to a dual-topic perception path that:
- subscribes to both topics concurrently,
- composes `.obj -> saved_local` with `saved_local -> execution_world`,
- stays planning-only.

The local config is the current default path for OBJ-based Fabrica assets under `assets/obj/`.
The default `mesh_scale` for those OBJ assets is `0.01`; `1.0` is too large for the current grasp-width thresholds.

Standalone mesh antipodal grasp debug viewer:

```bash
python scripts/debug_mesh_antipodal_grasps.py --geometry cube
python scripts/debug_mesh_antipodal_grasps.py --geometry cylinder
python scripts/debug_mesh_antipodal_grasps.py --geometry stl --stl-path my_part.stl --stl-scale 0.001
```

This viewer is also browser-based. It writes a self-contained HTML file to
`artifacts/mesh_antipodal_grasp_debug.html`; open that file from the host browser.
Relative STL paths are resolved under `assets/stl/`.

Fabrica two-stage grasp workflow:

```bash
python scripts/generate_fabrica_assembly_grasps.py \
  --mesh-path obj/fabrica/beam/2.obj \
  --assembly-glob 'obj/fabrica/beam/*.obj' \
  --mesh-scale 0.01 \
  --num-samples 204 \
  --antipodal-cosine-threshold 0.984807753012208 \
  --min-jaw-width 0.002 \
  --max-jaw-width 0.09 \
  --output-json artifacts/fabrica_beam_2_assembly_grasps.json \
  --output-html artifacts/fabrica_beam_2_assembly_grasps.html

python scripts/check_fabrica_ground_feasible_grasps.py \
  --input-json artifacts/fabrica_beam_2_assembly_grasps.json \
  --output-json artifacts/fabrica_beam_2_ground_feasible.json \
  --output-html artifacts/fabrica_beam_2_ground_feasible.html

python scripts/check_fabrica_ground_feasible_grasps.py \
  --input-json artifacts/fabrica_beam_2_assembly_grasps.json \
  --output-json artifacts/fabrica_beam_2_ground_feasible_neg_z.json \
  --output-html artifacts/fabrica_beam_2_ground_feasible_neg_z.html \
  --support-face neg_z \
  --yaw-deg 0 \
  --xy-world 0.0,0.0
```

Stage 1 generates grasps on the target part, filters them against sibling assembly meshes from the same Fabrica assembly, scores the surviving grasps geometrically, and saves the accepted grasps plus an HTML viewer in score order.
Stage 2 reloads those saved grasps, applies a pickup pose, filters them against the pickup ground plane only, rescoring any surviving grasps before export.
The saved grasp JSON stays in the target part-local frame, but the stage-2 HTML now renders the part and grasps in the selected pickup world pose so support-face and yaw overrides are visually obvious.

Current geometry conventions:
- Fabrica assets now live under `assets/obj/`
- individual OBJ files are authored in shared assembly coordinates
- stage 1 canonicalizes the chosen target part into a saved local frame
- stage 2 applies `local -> execution_world`
- the floor is the execution-world plane with normal `+z` at `z=0`

Integrated Fabrica-to-Isaac pickup flow:

```bash
/isaac-sim/python.sh scripts/convert_stl_to_usd.py \
  --stl-path Fabrica/printing/beam/2.stl \
  --stl-scale 0.001 \
  --output-usd artifacts/converted/beam_2.usd \
  --headless \
  --device cuda

/isaac-sim/python.sh scripts/run_fabrica_pickup_in_isaac.py \
  --input-json /workspace/Grasp_Planning/artifacts/fabrica_beam_2_assembly_grasps.json \
  --part-usd /workspace/Grasp_Planning/artifacts/converted/beam_2.usd \
  --controller admittance \
  --support-face neg_y \
  --yaw-deg 90 \
  --xy-world=-0.5,0.0 \
  --headless \
  --device cuda \
  --run-seconds 5
```

This path:
- consumes the saved stage-1 grasp JSON,
- converts the STL to an Isaac-native rigid mesh asset with Isaac Lab's `MeshConverter`,
- places the part in sim at the requested or sampled support pose,
- rechecks the saved grasps against the floor for that exact pose,
- selects the first feasible grasp and attempts execution,
- writes an attempt artifact to `artifacts/isaac_pick_attempt.json`.

Notes:
- If `--xy-world` starts with a negative value, pass it as `--xy-world=-0.5,0.0` or quote it.
- The current remaining failure mode is controller/pregrasp convergence, not STL-to-USD conversion or grasp loading.

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
./run_pipeline.sh --mode local
```

If `local_simulation.enabled: true` and `local_simulation.part_usd` is set in `configs/grasp_pipeline_local.yaml`,
the local pipeline writes stage-1 and stage-2 artifacts first, then hands off to Isaac for the execution attempt.

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

Fabrica assembly / pickup path:
- `scripts/generate_fabrica_assembly_grasps.py` is the offline stage,
- `scripts/check_fabrica_ground_feasible_grasps.py` is the pickup-ground recheck stage,
- `scripts/run_grasp_pipeline.py` is the shared YAML-driven orchestration entrypoint,
- `run_pipeline.sh` is the user-facing launcher for `local` and `real` modes,
- shared pipeline code lives under `grasp_planning/pipeline/`,
- ROS2 object-pose listening lives under `grasp_planning/ros2/`,
- shared utilities and viewer generation live in `grasp_planning/grasping/fabrica_grasp_debug.py`,
- assembly OBJ files are assumed to already be in a shared global coordinate system,
- the target part is recentered into a canonical local frame before grasps are saved,
- saved grasp JSON remains in that local frame so stage 1 and stage 2 talk in the same coordinates,
- saved grasp bundles now also store the source-frame origin/orientation used to define that local frame,
- stage 1, stage 2, and the stage-1 HTML obstacle view all honor that stored source-frame rotation, not just translation,
- saved grasp poses already include any accepted finger-pad contact offset refinement; downstream consumers should execute the stored pose directly rather than reapplying the offset,
- both stages refine infeasible center-contact grasps over a 5x5 grid on the Franka rubber tip contact patch, with equal inset spacing from the pad edges in lateral and approach directions,
- Fabrica scoring is geometric-only over already-feasible grasps: antipodal alignment, centering, local contact support, and COM offset in the closing-plane,
- stage 1 and stage 2 HTML viewers list grasps in score order,
- `scripts/check_fabrica_ground_feasible_grasps.py` accepts `--support-face`, `--yaw-deg`, and `--xy-world` overrides for one-off pickup-pose checks,
- `scripts/check_fabrica_ground_feasible_grasps.py` and `scripts/run_fabrica_pickup_in_isaac.py` also accept explicit object world pose overrides,
- accepted and rejected grasps are both rendered in the ground-recheck HTML viewer, and the viewer can be toggled to show accepted grasps only.

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
- `scripts/debug_mesh_antipodal_grasps.py` also supports assembly obstacle overlays with `--assembly-glob` and keeps those overlays in the target object frame for HTML visualization,
- the Fabrica two-stage scripts are local debug tools; they do not move the robot and do not depend on Isaac,
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

Detailed Franka finger collision now uses:
- `detailed_finger_contact_gap_m`

Matching CLI override:
- `--detailed-finger-contact-gap-m`

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
