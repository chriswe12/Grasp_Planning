# Fabrica-All Visual-Servo RL Dataset Implementation Plan

## Purpose

Build a substantially larger, leakage-safe visual-servo RL dataset from every
Fabrica assembly in this repository, then provide a second expansion path for
additional real or procedurally generated industrial parts.

This document is an implementation handoff. A new agent should be able to use
it to make the changes without rediscovering the current dataset pipeline or
making a monolithic 46-part Isaac scene that exhausts VRAM.

## Current State

The active multipart task is plumbers-block-specific:

- dataset root: `isaac_rl/data/plumbers_block/`
- configured parts: `0`, `1`, `2`, `3`, `4`
- raw selected manifest targets: 661
- final collision/rotation/render-validated goal targets: 393
- goal observations: RGB-D rendered by MuJoCo Filament
- live observations: RGB-D rendered by Isaac RTX and randomized online
- policy input: live RGB-D plus goal RGB-D at `128x72`
- reset variants: eight collision-validated variants per retained target
- split behavior: grouped by `(part_id, grasp_id)` so one local grasp cannot
  cross train/validation/test through different stable orientations

Relevant current entry points:

- `isaac_rl/scripts/prepare_plumbers_block_catalog.py`
- `isaac_rl/scripts/build_assembly_multigrasp_manifest.py`
- `isaac_rl/scripts/build_multigrasp_manifest.py`
- `isaac_rl/scripts/plan_multigrasp_targets.py`
- `isaac_rl/scripts/build_multigrasp_path_asset.py`
- `isaac_rl/scripts/build_multigrasp_rotation_reset_asset.py`
- `isaac_rl/scripts/capture_multigrasp_goal_catalog_mujoco.py`
- `isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/multigrasp_catalog.py`
- `isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env.py`
- `isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env_cfg.py`

The parent repository and `isaac_rl/` are separate Git repositories. Both are
currently dirty and contain unrelated user work. Preserve all existing changes,
inspect overlapping diffs before editing, and do not reset either repository.

## Fabrica Inventory

All current Fabrica OBJ assets use the established mesh scale `0.01`.

| Assembly | Part IDs | Count |
|---|---|---:|
| `beam` | `0, 1, 2, 3, 6` | 5 |
| `car` | `0, 1, 2, 3, 4, 5` | 6 |
| `cooling_manifold` | `0, 1, 2, 3, 4, 5, 6` | 7 |
| `duct` | `0, 1, 2, 3, 4, 5, 6, 7` | 8 |
| `gamepad` | `0, 1, 2, 3, 4, 5` | 6 |
| `plumbers_block` | `0, 1, 2, 3, 4` | 5 |
| `stool_circular` | `0, 1, 2, 3, 4, 5, 6, 7, 8` | 9 |
| **Total** | | **46** |

Some meshes are much larger or denser than the current plumbers-block assets.
For example, car, duct, beam, gamepad, and stool contain parts whose largest
dimension exceeds the PDZ opening. They can still expose graspable thin
features, but inclusion must be decided by the existing geometry, jaw-width,
ground, MoveIt, reset-collision, and goal-capture filters. Do not force every
part to contribute a target.

The first four assemblies inspected already expose at least 138 stable support
orientations before grasp feasibility filtering. Direct convex-hull processing
of the high-detail gamepad render meshes was slow enough to justify a cached
simplified planning representation.

## Dataset Products

### Product A: `fabrica_all_v1`

The first production dataset should cover every feasible Fabrica part and use
the existing PDZ gripper, D405 camera, material, T-slot, and 15 Hz policy
contracts.

Recommended pilot settings:

- 32 selected diverse targets per feasible part orientation
- 128 alternates per part orientation
- eight collision-validated reset variants
- at least four distinct valid reset variants per retained target
- current 10 mm total approach clearance contract
- current maximum training jaw-width and PDZ collision model
- one canonical MuJoCo Filament RGB-D goal per target

Recommended full settings after the pilot succeeds:

- 64 selected diverse targets per feasible part orientation
- 256 alternates per part orientation
- increase to 128/512 only for orientations where farthest-point diversity
  continues to add genuinely different contact, approach, roll, or jaw-width
  targets

Expected scale after all filters:

- pilot: approximately 2,000-4,000 final targets
- full: approximately 4,000-8,000 final targets
- goal catalog storage: roughly 0.5-1.5 GiB at the current per-target size
- planning JSON, caches, USDs, diagnostics, and reset assets may consume several
  additional GiB

These are estimates. The builder must report actual yield at every filter stage
and per assembly/part/orientation.

### Product B: `fabrica_extended_v1`

After `fabrica_all_v1` trains successfully, add approximately 50-300 additional
physically plausible parts. Favor the geometry distribution expected in the
real workcell:

- brackets and plates
- pipe fittings and elbows
- knobs, handles, and connectors
- cylindrical and prismatic industrial parts
- additional real CAD parts expected in deployment
- parametric variants with explicit, physically meaningful dimensions

Do not mix arbitrary internet meshes into the production dataset without scale,
license/source, watertightness/COM, size, stable-pose, and gripper-feasibility
metadata. Procedural or public-data parts should be marked as pretraining data,
and the original Fabrica validation/test sets must remain identifiable.

## Non-Negotiable Dataset Contracts

### Globally unique identifiers

The current `part_0__orientation_...` format collides across assemblies. Use a
stable namespace everywhere:

```text
part key:       beam__part_0
orientation:   beam__part_0__orientation_003
target:        beam__part_0__orientation_003__g0147
```

Requirements:

- IDs must not depend on array order or shard assignment.
- Use the same IDs in manifests, plans, paths, rotation resets, goal catalogs,
  metrics, videos, and checkpoint metadata.
- `part_names` should contain the globally unique part keys.
- Preserve local fields separately: `assembly_name`, `local_part_id`,
  `local_orientation_id`, and `grasp_id`.
- Bump the multipart manifest/catalog schema when adding merged-assembly fields.
- Maintain backward compatibility for the existing plumbers-block schema or
  provide a clear one-time migration path and tests.

### Exact aligned-array contract

For every finalized shard:

```text
goal_catalog.target_ids
    == paths.target_ids
    == rotation_resets.target_ids
```

The order must match exactly. Finalization must remove a failed target from all
target-aligned artifacts, not only from RGB-D images.

### Visual and robot profiles

Do not bypass the existing profile checks. Every catalog/shard must record and
match:

- PDZ robot profile
- approach-gripper profile
- D405 goal camera profile
- RGB-D observation profile
- visual material profile
- visual scene profile
- T-slot workspace profile
- MuJoCo Filament goal renderer profile

Use the shared black-finger, white-non-emissive-pad, muted-part, and canonical
T-slot material contract already implemented in the repository.

### Feasibility and safety

A target can enter training only after passing:

1. stable support-pose validation;
2. diverse grasp selection;
3. jaw-width and 10 mm approach-clearance limits;
4. gripper/object/ground geometry filtering;
5. MoveIt plan validation;
6. path-asset generation;
7. collision-safe rotation/position reset generation;
8. MuJoCo goal capture and strict reload validation.

Never create an episode that starts in collision. A part or orientation with no
passing targets should be excluded with a machine-readable reason.

## Split And Evaluation Design

One split cannot answer both known-object grasp generalization and unseen-object
generalization. Preserve both protocols.

### Primary production split: held-out grasps

- Every feasible part may appear in training.
- Group by `(assembly_name, local_part_id, grasp_id)`.
- All stable-orientation appearances of the same physical grasp stay in one
  split.
- Use approximately 80/10/10 train/validation/test.
- This tests new grasps on known parts and is the split used for the final
  all-Fabrica production policy.

### OOD evaluation split: held-out objects/assemblies

Create an additional split scheme in metadata and evaluation tools:

- held-out-part split: complete parts are absent from training;
- held-out-assembly split: one or more complete assemblies are absent from
  training;
- do not report these as production all-Fabrica results because that policy
  intentionally trains on all assemblies;
- use a separate ablation checkpoint or dataset view to measure visual
  goal-conditioned generalization.

Suggested artifact fields:

```text
split_ids                    # primary held-out-grasp split
part_holdout_split_ids       # secondary evaluation scheme
assembly_holdout_split_ids   # secondary evaluation scheme
split_scheme_metadata_json   # seeds, grouping rules, counts
```

Update catalog validation and evaluation code so every reported metric names
the split scheme explicitly.

## Image-Diversity Design

Do not multiply the static catalog with hundreds of nearly identical goal
renders. The goal image is the deterministic definition of the desired grasp;
most image diversity belongs in the live observation stream.

### Canonical goal data

- Keep one strict canonical RGB-D goal per target.
- Keep camera pose, intrinsics, crop, depth normalization, gripper aperture,
  object pose, and material semantics fixed.
- Optionally support 2-4 mild goal-render appearance variants later, but store
  the canonical variant separately and do not change goal geometry.

### Online live diversity

Continue generating effectively unlimited live views during training:

- weighted part colors and roughness;
- T-slot appearance and geometry variation;
- light intensity, color, direction, and shadow position;
- clean, cluttered, and busy office/factory backgrounds;
- D405 RGB exposure, gamma, contrast, white balance, blur, noise, vignette, and
  patch occlusion;
- correlated disparity/depth bias, spatial and temporal noise, quantization,
  edge mismatch, structured dropout, and calibration warp;
- camera/mount uncertainty within the established calibration ranges;
- continuous robot reset progress, TCP position/rotation errors, object-pose
  estimate errors, and variable episode lengths.

Retain a nonzero clean-episode fraction. Extreme corruption must not become the
only training distribution.

## Scalable Artifact Layout

Use a versioned root and retain per-assembly intermediate artifacts:

```text
isaac_rl/data/fabrica_all_v1/
  dataset_index.json
  dataset_config.yaml
  inventory.json
  assemblies/
    beam/
      sources/
      usd/
      manifest.json
      plans/
      exclusions.json
    car/
    ...
  merged/
    manifest.json
    split_report.json
  shards/
    shard_00/
      goal_catalog.npz
      paths.npz
      rotation_resets.npz
      part_inventory.json
      usd/
    shard_01/
    ...
  reports/
    yield_by_stage.json
    yield_by_assembly_part_orientation.csv
    excluded_parts.json
    contact_sheets/
```

Large generated caches and rendered datasets should remain outside ordinary Git
history. Track configs, schemas, reports, and small manifests as appropriate;
sync large artifacts to Euler scratch/project storage explicitly.

## Why Training Must Use Shards

The current Isaac environment creates every configured part in every parallel
environment and moves inactive objects away. With five parts and 224-256
environments this is manageable. With 46 parts it would create more than ten
thousand rigid objects on each GPU and waste substantial VRAM and simulation
time.

Do not implement the large run as one 46-part scene per GPU.

Instead:

- divide parts into target-balanced shards;
- load only the shard assigned to a distributed rank;
- keep the same local environment count and rollout horizon on every rank;
- synchronize PPO gradients across ranks;
- ensure every global PPO iteration receives samples from all active shards;
- partition by expected validated target count and asset cost, not simply by
  the number of part files;
- record shard membership and weights in checkpoint metadata.

Recommended first deployment:

- four GPUs;
- four balanced shards, each containing approximately 10-13 parts;
- use fewer parts on a shard containing unusually heavy meshes;
- run a VRAM/throughput probe before the full training job;
- if seven or eight GPUs are readily available, one assembly per rank is a
  convenient diagnostic configuration, but equal rank weighting must not
  unintentionally oversample small assemblies.

Single-GPU sequential shard training is acceptable only as a smoke test. It is
not the preferred production method because repeated shard switching can cause
forgetting and changes the PPO data distribution between updates.

## Required Implementation Work

### Phase 0: preserve and characterize the current baseline

1. Record the current plumbers-block catalog signatures, counts, schema, and
   validation reports.
2. Add/retain a plumbers-block build smoke test so genericization cannot break
   the existing dataset.
3. Do not regenerate or overwrite the current production catalog until the new
   versioned pipeline passes its pilot.

### Phase 1: inventory and simplified planning meshes

1. Add a generic inventory tool, for example:

   `isaac_rl/scripts/build_fabrica_rl_inventory.py`

2. Enumerate assembly directories and numeric OBJ part IDs deterministically.
3. Record mesh bounds, vertex/face counts, scale, source path, and source hash.
4. Build cached simplified planning/collision meshes for high-detail parts.
5. Keep original render meshes for goal/live visual fidelity.
6. Validate that simplification preserves scale, frame, support surfaces, and
   grasp-relevant geometry within configured tolerances.
7. Write explicit exclusion reasons for invalid or unsupported meshes.

### Phase 2: generic per-assembly catalog preparation

1. Refactor `prepare_plumbers_block_catalog.py` into reusable library functions
   without deleting the existing wrapper.
2. Add a generic entry point, for example:

   `isaac_rl/scripts/prepare_fabrica_catalog.py`

3. Accept a YAML dataset configuration containing assemblies, part IDs, paths,
   scales, target caps, alternates, split seeds, and output root.
4. Generalize source generation so it does not hardcode `plumbers_block` or
   `PART_IDS`.
5. Preserve explicit resumable stages:

   `inventory -> sources -> manifest -> plan -> paths -> rotation -> mujoco -> finalize -> shard`

6. Every stage must support resume and must never treat the mere existence of a
   stale artifact as proof that it matches the active config/profile.

### Phase 3: namespaced manifests and split schemas

1. Update `build_assembly_multigrasp_manifest.py` to create globally unique
   part/orientation/target IDs.
2. Add per-target assembly and local-part fields.
3. Merge per-assembly manifests into one deterministic global manifest.
4. Implement the primary grouped-grasp split and secondary part/assembly
   holdout schemes.
5. Add schema validators and reports for uniqueness, grouping, coverage, and
   target counts.
6. Preserve greedy farthest-point grasp selection. More targets must mean more
   contact/approach/roll/width diversity, not adjacent duplicates.

### Phase 4: planning, reset, and goal-capture generalization

1. Make plan, path, reset, and capture scripts consume namespaced part keys and
   per-part USD paths.
2. Verify plan-cache keys include assembly, part, mesh hash, robot profile,
   planning parameters, and target ID.
3. Ensure rotation-reset cache keys cannot collide between assemblies.
4. Keep the exact target-alignment contract through finalization.
5. Extend diagnostic reports with per-stage and per-part rejection reasons.
6. Generate contact sheets per assembly and a bounded global overview rather
   than one unreadably large sheet.

### Phase 5: shard builder and dataset index

1. Add a deterministic shard builder that balances:

   - retained target count;
   - number of loaded part assets;
   - mesh/asset complexity;
   - assembly coverage.

2. Each shard must be a standalone valid multipart catalog with its own exact
   aligned goal/path/reset arrays and part USD inventory.
3. `dataset_index.json` must record:

   - dataset version and source config hash;
   - global and per-shard counts;
   - shard membership;
   - split schemes and seeds;
   - robot/camera/material/scene profiles;
   - hashes of shard catalogs and assets;
   - excluded parts/orientations and reasons.

4. Add a verifier that reloads every shard and checks union/disjointness against
   the global manifest.

### Phase 6: rank-aware Isaac/RL loading

1. Remove the plumbers-block-only dataset constants from the new task path.
2. Add explicit dataset-root/index/shard configuration. Do not silently infer a
   random dataset from the working directory.
3. Select the shard deterministically from distributed rank or an explicit
   `--dataset-shard` override.
4. Add a new registered task for the sharded Fabrica dataset rather than
   changing the existing plumbers-block task underneath old checkpoints.
5. Configure `part_names`, `part_usd_paths`, catalog, and reset paths from the
   chosen shard before scene construction.
6. Assert that all ranks use the same dataset version, split scheme, policy
   context, observation contract, batch dimensions, and optimizer schedule.
7. Log per-assembly and per-part sampling/success without creating thousands of
   TensorBoard series. Put detailed per-target data in offline evaluation
   artifacts.
8. Ensure global sampling does not overweight a shard merely because it has
   fewer targets.

### Phase 7: Euler integration

1. Add dataset staging/sync that avoids transferring unchanged GiB-scale files
   on every submission.
2. Stage data in an appropriate Euler project/scratch location and keep the
   repository path/config separate from large generated artifacts.
3. Add a four-GPU probe command that performs at least:

   - shard selection on every rank;
   - environment reset for each part represented on the rank;
   - one rollout and PPO update;
   - checkpoint save and reload;
   - VRAM and throughput reporting per rank.

4. Only after the probe passes, submit the full training job.
5. Keep existing watch/validate/pull behavior and include the dataset index and
   shard map beside checkpoints.
6. Remember that `max_iterations` is the number of synchronized global PPO
   updates, not a separate independent iteration count per GPU.

### Phase 8: evaluation

At minimum, produce:

- aggregate train/validation/test metrics under the held-out-grasp split;
- per-assembly and per-part metrics;
- far/mid/close reset metrics;
- completion precision/recall and premature-completion rate;
- collision and timeout rates;
- clean, sensor-randomized, cluttered, busy-background, and depth-robust
  conditions;
- a held-out-part checkpoint/evaluation;
- a held-out-assembly checkpoint/evaluation;
- browser-safe success and failure videos from multiple assemblies.

Do not infer a success rate from a few debug videos. Use the offline evaluator
with fixed seeds and report confidence intervals or at least numerator and
denominator.

### Phase 9: extended parts

1. Define an external-part manifest schema containing:

   - stable unique part key;
   - mesh source and license/provenance;
   - mesh scale and units;
   - density or mass assumptions;
   - category and real/procedural flag;
   - expected real availability;
   - render mesh and simplified planning mesh hashes.

2. Run the same feasibility pipeline and produce the same exclusion reports.
3. Keep Fabrica and external/procedural source labels in every target.
4. Add source-balanced sampling so thousands of procedural parts do not drown
   out the 46 physically relevant Fabrica parts.
5. Keep a production fine-tuning phase weighted toward the actual deployment
   parts and visual distribution.

## Testing Requirements

Add unit tests for:

- assembly/part/orientation/target namespacing;
- uniqueness across assemblies with repeated local part/grasp IDs;
- grouped split leakage prevention;
- held-out-part and held-out-assembly isolation;
- deterministic split and shard assignment;
- merged-manifest counts and ordering;
- exact aligned-array finalization;
- shard union and disjointness;
- catalog schema compatibility and profile checks;
- rank-to-shard mapping;
- configuration failures for missing shard assets or mismatched profiles;
- simplified-mesh frame/scale preservation;
- plumbers-block wrapper backward compatibility.

Add integration/smoke tests for:

- one small assembly through every CPU stage;
- a two-assembly merged manifest with deliberately colliding local IDs;
- one small MuJoCo goal-capture subset;
- one single-GPU sharded task reset/step;
- one multi-GPU PPO update and checkpoint reload on Euler.

## Pilot And Rollout Strategy

Do not immediately plan and render the maximum dataset.

### Pilot A: two assemblies

Use `plumbers_block` plus `beam` with 8-16 selected targets per orientation.
This must prove namespacing, merged splits, cross-assembly plans, exact aligned
arrays, goal capture, shard verification, and environment loading.

### Pilot B: all assemblies, low cap

Use all seven assemblies with 32 selected targets and 128 alternates per
orientation. Produce the actual yield/cost report and four target-balanced
shards.

### Pilot C: distributed training proof

Run four GPUs for enough updates to expose initialization OOM, growing VRAM,
rank imbalance, broken checkpointing, or a dataset shard that never resets
successfully. Inspect per-rank FPS and VRAM rather than only aggregate FPS.

### Full run

Increase the target cap only after inspecting diversity and validated yield.
Train using all four shards, run periodic fixed-seed validation, retain best and
last checkpoints, and automatically pull the final reports/checkpoints.

## Acceptance Criteria

`fabrica_all_v1` is complete only when:

- all seven Fabrica assemblies appear in the source inventory;
- every one of the 46 parts is either represented by validated targets or has
  an explicit exclusion reason;
- all IDs are globally unique and stable across rebuilds;
- no grouped grasp leaks across the primary split;
- secondary held-out-object split schemes pass isolation tests;
- every shard passes strict catalog/path/reset reload validation;
- no retained reset starts in collision;
- visual/robot/camera/profile checks pass without bypasses;
- clean and randomized observation sheets look correct for multiple assemblies;
- a four-GPU rollout/update/save/reload smoke test completes without OOM;
- per-rank FPS, peak VRAM, target counts, and shard weights are recorded;
- a resumable build command and Euler training command are documented;
- generated dataset reports and checkpoint metadata identify the exact dataset
  version and hashes.

`fabrica_extended_v1` additionally requires provenance, scale, physical-quality
gates, and source-balanced sampling for every extra part.

## Risks And Decisions To Preserve

- More static goal images are not a replacement for online live-image domain
  randomization.
- Exact MuJoCo/Isaac pixel equality is neither achievable nor desirable; preserve
  semantic material/camera/geometry parity and randomized live illumination.
- Loading 46 parts in every environment is not acceptable as the default design.
- Never reduce validation thresholds merely to increase catalog size.
- Do not split the same local grasp across train/validation/test through stable
  orientation aliases.
- Do not use arbitrary geometry scaling as deployment data unless the physical
  part scale truly varies.
- Do not treat a planning-only grasp as training-ready before MoveIt, reset
  collision, and goal-capture validation.
- Keep old plumbers-block tasks/checkpoints reproducible while introducing the
  new versioned sharded task.

## Documentation Handoff

When implementation changes durable behavior:

1. update repository docs and `completion.md` with actual, not estimated,
   counts and commands;
2. create or update the required companion-wiki changelog under
   `../mt_wiki/agent-changelogs/` following its template;
3. record source repo, parent and nested `isaac_rl` commits, changed paths,
   dataset version/hashes, tests, Euler probe results, risks, and exclusions;
4. do not directly edit durable wiki pages from this code repository.
