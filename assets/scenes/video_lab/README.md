# Video-derived modular lab

Editable Blender reconstruction of `/home/pdz/Downloads/Video.mov`, the newest
video in Downloads on 2026-09-14. The 11.08 s handheld clip is 320 × 568 pixels.
This is a modeled visual approximation, not a photogrammetric scan or a measured
digital twin. Six sampled frames are preserved in `reference/`.

The table matches the visible warm off-white laminate, center panel seam, thin
edge banding, gray steel frame, black rear grommets, tape remnants, and clustered
handling marks. The background includes white partitions, aluminum framing, an
open doorway, outlets, dark textured flooring, and a neighboring workstation.
Unseen geometry is an estimate. No robot, cables attached to a robot, robot
shadows, or task object is baked into the environment texture.

## Open and inspect

From the repository root:

```bash
blender assets/scenes/video_lab/video_lab.blend
```

The file opens in the overview camera. Select `Camera_Table_Detail` for a closer
look. The packed textures make the `.blend` self-contained. The construction
uses Blender 5.2.1, standard materials, and Cycles; no add-ons are required.

- `video_lab.blend`: editable master, with optional preview props visible.
- `preview_overview.png`, `preview_table_detail.png`: CPU Cycles renders.
- `preview_clean.png`: environment with all table props hidden.
- `preview_seed_7.png`, `preview_seed_23.png`: actual randomized scene renders.
- `environment.usdc`: static table and background; **no props, robot, or lights**.
- `props/*.usdc`: five independent dynamic mock assets, local bottom at z=0.
- `lighting.usdc`, `cameras.usdc`: optional reference lighting and inspection cameras.
- `preview.usda`: assembled USD inspection scene, including props and lights.
- `manifest.json`: dimensions, root paths, collision contract, prop sizes and masses.
- `validation.json`: structural verification results and simulation limitation.

Keep the USD files beside `textures/`. USD materials use portable
`UsdPreviewSurface` with albedo and roughness textures. Blender's fine procedural
micro-bump is not reproduced by the USD export. The USD environment has not yet
been visually validated in Isaac; renderer exposure and lighting need tuning
there independently of Blender.

## Edit dimensions and rebuild

`scene_config.json` exposes the table width, depth, height, thickness, seam,
color, roughness, wall dimensions and randomization ranges. The nominal table is
**1.60 × 0.80 × 0.74 m**, with a 25 mm top. These are estimates, not measurements.
A metric table measurement and calibrated camera would improve fidelity.

```bash
blender -b --threads 12 --python scripts/blender/build_video_lab.py -- \
  --output assets/scenes/video_lab --render --samples 96 --resolution 1440
```

Rebuilding overwrites generated assets in the output folder. Save hand edits to
another `.blend` first. Alternate dimensions update the table/support plane and
legs; the room, wear locations and nominal prop staging are independently
editable and should be reviewed after large dimensional changes. Surface texture
synthesis is deterministic and contains no source-video pixels.

## Collections and coordinate contract

| Collection | Contents / purpose |
| --- | --- |
| `Table` | Two laminate panels, edge bands, frame, feet |
| `Surface_Wear` | Separate marks, tape and grommets; individually editable |
| `Room` | Floor, partitions, doorway and electrical details |
| `Background_Furniture` | Neighboring workstation and floor cables |
| `Optional_Props` | Blue flange, brown bracket, green cap, tape roll, pencil |
| `Lighting`, `Cameras` | Independent lights and inspection viewpoints |
| `Anchors` | Estimated left/right robot mounts and central task origin |
| `Collision` | Hidden simple table and floor support meshes |

Units are meters, Z is up. The tabletop is **z=0**, floor **z=-0.74** nominal.
X runs along the long edge; +Y points toward the rear partition. Mount guides
are `(-0.60, 0.23, 0)` and `(0.60, 0.23, 0)`; neither is calibrated. Robots and
real task assets should be inserted through the normal project workflow.

The visible table seam and small surface details do not change the planar
support collision. Only table and floor have static collision; background
furniture and walls are visual. Collision meshes use USD `purpose=guide`, while
render geometry has no physics schema. The room footprint is 4.6 × 4.4 m; use
at least 5 m clone spacing or remove/crop the room and its floor for dense RL.
Do not spawn the old tabletop or a second z=0 ground beneath this asset.

Each prop USD has its own rigid-body root, mass and mesh collisions. Its component
meshes use convex hulls: holes in the flange, bracket and tape roll are visual,
**not insertion-accurate collision geometry**. The props are intended as optional
mock grasping/clutter assets. Replace their collision representation if those
openings matter to a task. Blender props remain freely movable authoring objects;
dynamic physics is authored in their separate USD files.

## Seeded Blender randomization

```bash
blender -b assets/scenes/video_lab/video_lab.blend --threads 12 \
  --python scripts/blender/randomize_video_lab.py -- \
  --seed 23 --output /tmp/video_lab_seed23.blend --render /tmp/video_lab_seed23.png
```

Add `--no-props` for an empty tabletop. A JSON sample is written next to the
variant `.blend`. The randomizer is also embedded as a Blender text block:
set the scene custom property `randomization_seed`, select the embedded
`randomize_video_lab.py` in the Text Editor, and press Run Script. The default
seed is zero.

Sampling changes table tint/roughness, wall tint, light power/temperature,
exposure, camera position/focal length, wear visibility, zero-to-five props,
prop poses, and modest prop hue/brightness/roughness. Repeated seeds reproduce
the same result without accumulating previous changes. Prop footprints are
kept inside the table, separated from one another, and excluded from estimated
robot-mount and central-task discs. A crowded configuration may fit fewer props
than requested. These are authoring ranges, not distributions measured from
the real lab. Table geometry and the support plane are not randomized implicitly.

`scene_config.json` is read during rebuilding. For an already open `.blend`, its
scene custom property `randomization_config` holds the active ranges.

## Opt-in Isaac / USD use

The helper `grasp_planning/rl/video_lab_scene.py` adds references and appearance
overrides without changing existing training tasks. After AppLauncher, before
starting simulation, and using your task's stage:

```python
from grasp_planning.rl.video_lab_scene import (
    add_environment, randomize_environment, add_mock_props,
)

add_environment(stage, "/World/envs/env_0/Lab")
randomize_environment(stage, "/World/envs/env_0/Lab", seed=23)
# Optional; leave out when spawning actual dataset objects through the task:
layout = add_mock_props(stage, "/World/envs/env_0/MockProps", seed=23)
```

The helper references absolute resolved asset paths for the live stage. The
shipped standalone USD package itself uses relative paths and is portable.
Material overrides are local to each non-instanceable environment reference;
randomizing one does not edit another environment or the source USD.
Runtime `randomize_environment` covers table tint/roughness, wall tint and wear.
Keep live light, camera, sensor and task-object randomization in the existing
RL profile; the broader Blender randomizer is for authoring/exported variants.
Lights are not automatically spawned by this helper. Reuse the task's lighting,
or explicitly reference `lighting.usdc` once for scene inspection.

Call `add_mock_props` only during scene construction. During training reset their
poses/velocities with Isaac's RigidObject state APIs using `sample_prop_layout`;
do not rebuild physics prims during stepping. The returned positions are local
to the supplied root. Add the correct environment origin in the simulator.
No training was started and no existing RL task/config was switched to this room.

## Verification

```bash
blender -b --python-exit-code 1 --python scripts/blender/validate_video_lab.py
```

This checks relative asset dependencies, units, support-plane bounds, separate
prop physics, 200 randomized layouts, deterministic Blender/USD sampling,
independent material overrides and unchanged source USD. Cycles renders were
inspected visually. Isaac contact/dynamics and RTX appearance are **not yet
verified**: the NVIDIA driver was unavailable in the authoring session.
