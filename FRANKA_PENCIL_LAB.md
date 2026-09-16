# Franka / ZED Mini in the pencil-mark lab

The video-derived pencil scene is integrated into `FrankaZedEnv` as an optional
training scene. Its catalog is `isaac_rl/data/franka_fabrica_pencil/catalog.npz`.
Passing this catalog to the existing trainer, controller checker or recorder
automatically loads the lab configuration stored in its contract. Plain-table
catalogs keep their existing scene and contract.

The room is translated by `(0.48, -0.10, 0)` so the existing robot base at the
environment origin occupies the lab's `(-0.48, 0.10, 0)` mount. Grasp paths,
object poses, TCP, camera mount and intrinsics are unchanged. The former tabletop
is removed; the lab supplies its tabletop and floor colliders. Parallel rooms
use 6 m spacing. USD metre/Z-up metadata and enabled manifest colliders are checked
when loading. Cable-width repairs are authored in the runtime stage.

Optional props use their authored positions and are collision-enabled kinematic
distractors, fixed between resets. These are not free dynamic clutter. They are
enabled in the inspection catalog. The blue Fabrica target remains kinematic
during the alignment task and the Panda fingers remain open, as before.

## Build and inspect

Use the Isaac Lab interpreter/container configured for this repository:

```bash
ISAAC_LAB=/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh
"$ISAAC_LAB" -p isaac_rl/scripts/rerender_franka_catalog.py --headless \
  --source isaac_rl/data/franka_fabrica_plumbers_v2/catalog_training_ready.npz \
  --output isaac_rl/data/franka_fabrica_pencil/catalog.npz \
  --lab-assets assets/scenes/video_lab_pencil --lab-props
"$ISAAC_LAB" -p scripts/record_franka_scene_inspection.py --headless \
  --catalog isaac_rl/data/franka_fabrica_pencil/catalog.npz \
  --output artifacts/franka_pencil_lab/recordings
python3 scripts/package_franka_scene_inspection.py \
  --recording-dir artifacts/franka_pencil_lab/recordings \
  --goal-images isaac_rl/data/franka_fabrica_pencil \
  --title 'Franka in the pencil-mark lab'
```

The packager needs Pillow and FFmpeg (`--ffmpeg /path/to/ffmpeg` or the optional
`imageio-ffmpeg` Python package). It verifies video decoding before writing the
local gallery. The high-resolution inspection run is at
`artifacts/franka_pencil_lab/recordings_hq/index.html` (native 960 x 640 overview,
256 x 144 wrist RGB, 1280 x 720 composed MP4). The initial smaller-camera gallery
is retained at `artifacts/franka_pencil_lab/recordings/index.html`.

The lab migration checks every saved approach using the actual 120 Hz joint
drives and arm/hand/finger contact sensors before recapturing goal RGB-D.
It preserves original source bundle hashes, grasp IDs and split assignments.
Physical lift labels are retained as `source_lift_validated`, while
`lift_validated` is cleared; the changed physical scene needs a new lift check.
`lab_approach_validated` records the new approach checks.

## Appearance and compatibility

The training launcher now defaults to
`isaac_rl/data/franka_fabrica_pencil_randomized/catalog.npz`. Its
`appearance_randomization` contract embeds
`configs/franka_pencil_randomization.json`. Each environment samples on reset;
the sample stays fixed within that episode. The profile varies:

- Target and prop colors across ten colors, including white and charcoal.
- Target roughness 0.28–0.85 and table roughness 0.28–0.82.
- Table brightness/channel tint, wall brightness and wear-mark visibility.
- Local light intensity 1,800–6,500, position, radius 7–32 cm and color
  temperature 3,500–7,500 K. Intensity is a renderer setting, not measured lux.
- Live observation exposure ±0.25 EV and channel gains 0.94–1.06.

Lights are linked to their own room; shared ambient fill stays fixed. Robot,
camera, geometry and collision properties retain their existing configuration.
Saved goal RGB-D remains the canonical pencil-scene reference, deliberately
independent of each live appearance. The policy must align across that difference.
Raw wrist-camera inspection panels show the physical render before observation
exposure/channel gains.

Create the randomized catalog with
`python3 scripts/prepare_franka_appearance_catalog.py`. It preserves every
non-contract array, including source IDs, splits, poses and goal images. Changing
the profile requires rebuilding this catalog and fresh launch-gate reports;
checkpoints with other contracts cannot be resumed into it.

The actual Isaac gallery is `artifacts/franka_randomized/recordings/index.html`:
12 appearance samples, six goal views and four zero/scripted videos. Validation
reports in `artifacts/franka_randomized/` cover reproducible resets, unchanged
canonical goals, light/material isolation, all 52 controller targets, and a
16-environment PPO smoke (3 epochs, 3,072 transitions, 286 finite tensors).

`--lab-seed N` applies the existing seeded table/wall appearance helper once at
setup. All clones use the same sample as the goal renders. This does not add
per-episode randomization. Omit it for the authored pencil appearance. Generate
a separate catalog after changing this seed, props, layout or assets.

`lab_scene` in the catalog contract records the asset directory, a hash of its
USD/texture/manifest inputs, room translation, appearance seed and prop mode.
Changes fail the normal goal/checkpoint contract check. Original Blender/USD
fallbacks and the old training catalog are not overwritten. The prepared
`artifacts/video_lab_franka/pencil_franka.usda` remains a separate static preview;
it is not used as the training articulation or camera definition.

## Validation artifacts

- `isaac_rl/data/franka_fabrica_pencil/catalog.rerender.json`: 52 new-scene approach checks and RGB-D captures.
- `artifacts/franka_pencil_lab/recordings/recording.json`: physics-stepped zero/scripted episode outcomes.
- `artifacts/franka_pencil_lab/controller_check.json`: parallel all-target reference/zero check.
- `artifacts/franka_pencil_lab/training_smoke.json`: bounded optimizer/checkpoint smoke.

On 2026-09-14, all 52 driven approaches had zero measured arm/hand contact force.
The parallel 52-environment reference check achieved 52/52 successes without
collisions (mean final position error 3.27 mm), and zero actions produced 52
timeouts. The four-environment PPO smoke completed three epochs/768 transitions,
with 286 finite checkpoint tensors and three actor optimizer updates. It used the
new lab catalog; this was a bounded smoke, not a new full training run.

The lab dimensions are estimated from video and camera calibration remains
provisional. The randomized run launched on 2026-09-14 at 14:59 local as
`franka-fabrica-zed-20260914_145932` (16 GPU environments, 2,000 epochs).
Its epoch-50 checkpoint passed verification: 51,200 transitions, 286 finite
tensors, finite losses and 96 completed actor optimizer updates. Evidence:
`logs/franka_fabrica/20260914_145932/startup_check.json`.
Current full-run details are recorded in
`artifacts/franka_fabrica/latest_launch.json`. Before reusing physical grasp/lift
claims, run the existing lift validator in this lab.

GPU scaling subsequently selected 32 environments and compatible cuDNN/NCCL
libraries. The previous run was replaced by
`franka-fabrica-zed-20260914_155609`; see `FRANKA_TRAINING_SCALING.md` for
measured throughput, resume validation and the single available GPU limitation.
