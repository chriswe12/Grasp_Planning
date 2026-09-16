# Warmer, lightly weathered tabletop — separate appearance study

This is a separate version of the approved scene. The original package remains
unchanged at `../video_lab/`, including its Blender file, textures and USD assets.
`fallback_hashes.json` records SHA-256 hashes of every original file.

Added to the tabletop: localized amber discoloration, gray contact smudges,
fine wipe streaks, small dirt specks, an interrupted residue ring, and faint
edge/seam grime. Color and roughness vary together. Lighting, camera poses,
room geometry, robot anchors, collision and all optional props are retained.

From the repository root:

```bash
# New version
blender assets/scenes/video_lab_weathered/video_lab.blend

# Original fallback
blender assets/scenes/video_lab/video_lab.blend
```

Compare `preview_table_detail.png` with `comparison_original_detail.png`: both
use the same camera, lights and props. `preview_overview.png` shows the room.

## Dial the dirt back

In the Shader Editor for material `Table_Laminate`, adjust the Value node named
`Table_Weathering_Strength`: **0 = original**, **1 = this version**, with values
between them for less dirt. The same node controls color and roughness.
Alternatively, use Blender's Python console:

```python
bpy.data.materials['Table_Laminate'].node_tree.nodes['Table_Weathering_Strength'].outputs[0].default_value = 0.5
```

Textures are packed in the new `.blend`. The embedded scene randomizer still
works and retains the dirt-control setting. As before, invoking that randomizer
replaces tabletop roughness with its sampled roughness value; the dirt color
variation and strength control remain active.

## Regenerate or export a different dirt pattern

```bash
blender -b --threads 12 --python-exit-code 1 \
  --python scripts/blender/weather_video_lab.py -- \
  --seed 194 --amount 1.3 --render
```

This reads the original scene and overwrites only the separate weathered output.
`--seed` varies surface irregularity and specks; `--amount` scales the authored
wear (0–2). Use `--output` to keep additional alternatives. Do not rebuild this
folder with the original scene builder, which does not author this weathering.

The new USD environment differs from the original only in its two tabletop
texture file paths. It contains the authored weathered appearance at full
strength; moving the Blender slider does not update USD automatically. Re-run
the weathering command with another amount for a new portable USD appearance.
Use this folder as `asset_dir` with the existing opt-in RL helper. Keep textures
alongside the USD files. For coordinates, physics limitations and broader
randomization usage see `../video_lab/README.md`.

The original package hashes, USD dependency resolution, unchanged geometry and
physics, packed textures, material strength endpoints and existing randomizer
compatibility are checked in `validation.json`. Isaac simulation was not rerun.
