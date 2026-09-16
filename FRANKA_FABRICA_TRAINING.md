# Fabrica grasps in the Panda/ZED task

For the all-part Panda catalog and longer Euler continuation workflow, see [FRANKA_FABRICA_ALL_TRAINING.md](FRANKA_FABRICA_ALL_TRAINING.md). The commands below describe the original single-part pilot.

For GPU-count benchmarks, cuDNN acceleration, checkpoint resume and multi-GPU
launching, see [`FRANKA_TRAINING_SCALING.md`](FRANKA_TRAINING_SCALING.md).

For the integrated video-derived pencil lab and its episode gallery, see
[`FRANKA_PENCIL_LAB.md`](FRANKA_PENCIL_LAB.md). It is selected by its own catalog;
the plain-table catalog below remains available.

The launcher now selects the pencil lab with per-episode colors, materials and
lighting from `configs/franka_pencil_randomization.json`, using
`isaac_rl/data/franka_fabrica_pencil_randomized/catalog.npz`. See the lab document
above for ranges and rendered examples. Current run/container details live in
`artifacts/franka_fabrica/latest_launch.json`.

The previous plain-table appearance was `panda_hand_wrist_rgbd_white_table_blue_part_keylight_v2`:
a matte-blue target, white table, directional key light, lower ambient fill and
ambient occlusion. The first pilot used an uncolored imported USD under uniform
dome lighting and was superseded after visual inspection. Its checkpoints remain
in `logs/franka_fabrica/20260914_093354/`; do not resume them with the v2 catalog.
The previous run `franka-fabrica-zed-20260914_103922` was no longer running when
the lab integration began; its log ended at epoch 1,547/2,000.

The stage-2 adapter now trains on actual `plumbers_block/0` grasp bundles from
`artifacts/fabrica_franka_pilot`, with the Panda arm/hand and supplied ZED Mini
mount in the lab. This remains a single-part baseline. Physical
FR3 hardware and measured ZED calibration are not implied by the Panda preset.

## Data and validation

The planner produced 348 grasp-orientation pairs (331 distinct grasp IDs).
The adapter selected 41 candidates across three stable orientations and
created two placements per candidate: 82 targets. It preserves original grasp
IDs, bundle hashes, object frame and per-grasp jaw widths. The shared
`saved_grasp_to_world_grasp` applies contact-patch offsets; the adapter also
accounts for the planner's 103.65 mm contact height versus the task's 103.4 mm
TCP offset. Object geometry is reconstructed in the saved bundle-local frame
and converted to a convex-decomposition collision USD.

Seventy targets passed GPU IK, arm/hand/finger contact monitoring, driven
approach checks at 120 Hz and object-only camera visibility checks. Sixty-two
then passed free-object closure, an 8 cm scripted lift and a half-second hold
above 5 cm, with bilateral object contact and a 20 N per-finger drive cap.
Those lift checks start at the exact grasp; they do not certify a learned
policy's imperfect final pose or hardware execution. Simulated object mass is
0.1401 kg, estimated using the existing 1240 kg/m3 density assumption.

The subsequent visual-servo reference check exposed finger collisions for
five source grasp IDs. Both placements of each affected ID were excluded:
`g0076`, `g0463`, `g2517`, `g3283`, `g3286`. Collision thresholds were preserved.
The original 62-target catalog and failure reports remain available.
The resulting training catalog has **52 targets: 33 train, 12 validation,
7 test**, with no source grasp ID shared across splits. These are held-out
grasps of the same part, not held-out objects or assemblies.
The final paired check completed all 52 reference-controller episodes without
collision (mean final error 3.27 mm); zero commands timed out on all 52.

Artifacts:

- `isaac_rl/data/franka_fabrica_plumbers/source_manifest.json`: source lineage.
- `catalog.npz` / `catalog.json`: 70 approach/image-checked targets and diagnostics.
- `catalog_lift_validated.npz`: 62 physically checked targets.
- `isaac_rl/data/franka_fabrica_plumbers/catalog_training_ready.npz` / `.filter.json`: original retained targets and exclusions.
- `isaac_rl/data/franka_fabrica_plumbers_v2/catalog_training_ready.npz`: previous plain-table 52-target catalog with refreshed RGB-D.
- `isaac_rl/data/franka_fabrica_pencil_randomized/catalog.npz`: current 52-target lab catalog with per-episode appearance profile and canonical goals.
- `artifacts/franka_randomized/`: current appearance, controller and optimizer checks plus rendered gallery.
- `catalog_training_ready.rerender.json` in the v2 directory: source hash, appearance contracts and per-target image checks.
- `artifacts/franka_fabrica/lifts.json`: per-target physical lift results.
- `artifacts/franka_fabrica_v2/controller_check_ready.json`: final paired controller check.
- `artifacts/franka_fabrica_v2/training_smoke.json`: optimizer/checkpoint verification.
- `artifacts/franka_fabrica/latest_launch.json`: persistent run/container details.

## Reproduce preparation

Use Isaac Lab 2.3.2 / Isaac Sim 5.1; scripts enable cameras automatically.

```bash
ISAAC_LAB=/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh
# Refresh the existing v1 catalog for the appearance-only change:
"$ISAAC_LAB" -p isaac_rl/scripts/rerender_franka_catalog.py --headless \
  --source isaac_rl/data/franka_fabrica_plumbers/catalog_training_ready.npz \
  --output isaac_rl/data/franka_fabrica_plumbers_v2/catalog_training_ready.npz
"$ISAAC_LAB" -p isaac_rl/scripts/check_franka_zed.py --headless \
  --catalog isaac_rl/data/franka_fabrica_plumbers_v2/catalog_training_ready.npz \
  --output artifacts/franka_fabrica_v2/controller_check_ready.json
```

The checker writes a `passed` verdict even when a failure is followed by
Isaac shutdown; the process exit status alone is insufficient. The launcher
requires a passing controller report and optimizer smoke report for the exact
catalog SHA256. Randomized catalogs additionally require a passing rendered
appearance/isolation report for that hash. Catalog generation is an offline operation and does not modify
the source grasp bundles or run the hardware execution pipeline.

## Train and monitor

After readiness reports pass:

```bash
python3 scripts/launch_franka_fabrica_training.py --num-envs 32 --iterations 2000
cat artifacts/franka_fabrica/latest_launch.json
docker ps --filter label=codex.task=franka-fabrica-zed-training
```

To reproduce the optimizer gate, run `train_franka_zed.py` with this catalog,
`--iterations 3 --num-envs 16 --headless`, then use
`python3 scripts/verify_franka_training_run.py --run <run-directory>
--output artifacts/franka_randomized/training_smoke.json`.

The launcher starts a detached Docker container and prevents a second active
launch with the same task label. Closing the terminal or conversation does
not stop it. Outputs are bind-mounted into `logs/franka_fabrica/<launch-time>/`:
`session.log`, `launch.json`, training logs, checkpoints and contract sidecars.
After training exits successfully, the container evaluates its best checkpoint
on validation and test targets, then writes `COMPLETED`. Use
`docker logs -f <container-name>` to watch, or `docker stop <container-name>`
to stop. It does not automatically restart after a host reboot.

```bash
docker logs -f <container-name-from-latest_launch.json>
```

The actor learns six camera-frame motion commands and completion; closure/lift
is currently an offline data-quality check, not an online training phase or
reward. The camera remains provisional; appearance variation is described in
`FRANKA_PENCIL_LAB.md`.
The next extensions are broader part/placement coverage, rotational reset
perturbations, physical lift validation from imperfect policy stops, calibrated
ZED noise/latency and the appropriate real-robot controller/model.

## Appearance contract and inspection

`make_franka_part_material_cfg()` is shared by the primitive preview and imported
USD spawn. Binding the material explicitly avoids losing the blue appearance when
replacing the placeholder with a Fabrica asset. The shared KUKA render configuration
is unchanged. The white table and Panda body materials are retained.

The refresh utility permits only a scene-profile change; it verifies all other
contract fields and source hashes, and preserves all geometric arrays, split IDs,
and physical-validation labels. It recaptures goal RGB-D through Isaac; it does not
recolor cached images. Old scene contracts fail on load. For changes to geometry,
physics, camera or TCP, use the full catalog builder and physical validators.

The blue material is provisional until real part appearance is specified. Measured
ZED calibration remains pending. Inspector runs opened before these edits must be
restarted to spawn the corrected scene. Use `scripts/record_franka_scene_inspection.py`
for goal views and zero/scripted episode frames. Corrected gallery:
`artifacts/franka_scene_inspection_v2/index.html`.
