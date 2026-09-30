# Visual-Servo Training And Data

Run commands from the repository root. The standalone Isaac Lab project is
the `isaac_rl/` Git submodule; its [README](../isaac_rl/README.md) covers the
KUKA/D405 task and model. This guide consolidates the parent repository's
Franka training contracts. See [Euler](../euler/README.md) for cluster setup
and the [Franka workbench](franka-policy.md) for deployment.

## Select The Task

- Bundle-driven Isaac execution runs MoveIt-planned joint waypoints. It is
  not the standalone RL training environment.
- Franka training uses `isaac_rl/scripts/train_franka_zed.py` and
  `FrankaZedEnv`, with a Panda simulation model, Panda hand, and ZED RGB-D.
  Hardware deployment uses FR3 and must separately validate model/control compatibility.
- The online task learns visual alignment and completion. Its kinematic part
  and open fingers do not train dynamic pickup/lift. Offline lift validation
  checks catalog grasps, not learned hardware grasp success.

Use the actual catalog and checkpoint metadata to identify a run. Experiment
names such as pencil, clutter, all-parts, or pair-only are not interchangeable
task contracts. Local catalogs and checkpoints are not supplied by cloning.

## Prepare A Franka Catalog

The all-parts preparation path is:

```bash
python3 scripts/run_grasp_generation_benchmark.py \
  --config configs/franka_fabrica_all.yaml --jobs 4 --skip-stage1-collision-checks
bash scripts/franka_isaac_python.sh isaac_rl/scripts/build_franka_all_catalog.py --headless
bash scripts/franka_isaac_python.sh isaac_rl/scripts/validate_franka_all_lifts.py --headless
python3 scripts/verify_franka_all_catalog.py \
  --catalog isaac_rl/data/franka_fabrica_all/catalog_lift_validated.npz \
  --output artifacts/franka_catalog_audit.json
```

Skipping stage-1 assembly collision checks is intentional for isolated parts;
it does not bypass object/gripper or floor filtering. Geometry clearance is
not a substitute for physical lift validation. Inspect coverage and exclusion
reports rather than assuming every Fabrica part survived validation.

Catalog rows must keep source bundle/mesh/frame identity, target poses,
grasp IDs, and train/validation/test splits aligned. Variants of the same
underlying grasp must stay in the same split. Rebuild and audit when these
contracts change; do not silently substitute a different catalog for a resume.

Useful recipe builders include `scripts/prepare_franka_fast_training.py` and
`scripts/build_franka_pair_only_catalog.py`. Inspect their `--help` before
choosing inputs and distinct output paths. A two-image/pair-only network
changes architecture; it is not checkpoint-compatible with enhanced fusion
merely because the image resolution matches.

## Camera And Observation Contract

Camera profiles under `configs/franka_zed_mini_sn13829658*.json` bind calibrated
rectified-left intrinsics and image dimensions. Catalogs embed camera profile
data and its digest; deployment must agree with checkpoint metadata. Measured
intrinsics do not establish a measured physical camera mount.

RGB and optical-axis Z depth in metres share the rectified-left image frame.
Invalid normalized depth uses the far-range value. Area downsampling must
respect validity and keep environments/channels independent. Preserve the
`rgbd_packing=dense_depth_channel_v2` and
`depth_source=radial_to_optical_z_v1` contracts where declared. Legacy packing
is only for reproducing explicitly compatible legacy artifacts, not a repair
via arbitrary depth offsets.

Supported training profiles include 128x72, 256x144, and 384x216 observations.
Changing resolution requires matching goal references, camera metadata,
network shapes, and deployment support; resizing live images alone is not
a compatible migration.

## Goals, Placement, And Resets

The mixed-goal recipe combines canonical Isaac, colored Isaac, and MuJoCo
references (20/30/50 percent in that recipe). Live appearance is independently
sampled and the selected goal stays fixed for the episode. Goal variants are
cached offline; MuJoCo is not an online training controller.

Placement randomization uses a finite, validated bank of absolute world
placements, with grounded object geometry and transformed TCP targets. The
common Franka workspace recipe uses x=0.33-0.58 m, y=+/-0.185 m, and
yaw=+/-90 degrees; read the selected recipe rather than applying these ranges
to every catalog. Do not substitute unchecked reset poses when IK or physics
validation rejects a state. Report per-target coverage and exclusions.

The `configs/franka_clutter_v5_replica.json` recipe mixes path, nominal, ready,
and boundary reset states. Completion positives/negatives require physical
label validation. Recipe changes to resets, supervision, goal banks, or
network architecture must be explicit when comparing or resuming runs.

## Policy And Evaluation

The hybrid policy emits six camera-frame motion actions and a completion
decision. The standard timing contract is 120 Hz physics and 15 Hz policy
control; standard velocity scales are 0.04 m/s and 0.24 rad/s. Recipe-selected
deployable context may include previous actions, measured camera-frame twist,
and rotation context. Metadata binds context dimensions and control rate.
Ground-truth pose errors and completion labels are training supervision and
critic inputs, never deployed actor inputs.

Placement/video evaluation can accept certified object symmetry. Position
and angular error must use the same equivalent TCP candidate, not independently
best candidates. Keep nominal metrics alongside symmetry-aware ones. Finite
symmetries need source/frame validation; continuous symmetry requires explicit
certification. An evaluation setting does not itself change a trained policy's
objective. Check the recipe and checkpoint for training-time symmetry settings.

## Run And Resume

```bash
bash scripts/franka_isaac_python.sh isaac_rl/scripts/train_franka_zed.py --help
```

Choose an explicit catalog, split, run directory, and camera profile. The
trainer's default cube catalog is not an implicit selection of an all-parts
or clutter experiment. Resume only with compatible catalog, camera, packing,
network/context dimensions, and checkpoint sidecar metadata.

For multi-GPU work, follow the [Euler guide](../euler/README.md). Stage an
isolated snapshot, smoke-test imports/assets and distributed execution, check
global rollout/minibatch accounting and finite optimizer updates, then submit
only when training is authorized. A queued job is not a running or validated
training result. Evaluate held-out targets with per-part coverage and failure
categories rather than only an aggregate success rate.
