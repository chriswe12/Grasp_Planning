# Pencil-mark tabletop variant

This version adds the graphite-like marks highlighted in the user's screenshot:
short parallel scratches, cross-hatching, retraced fixture outlines and scribble
fragments, concentrated on the right-hand work area. These are more numerous and
higher contrast than the earlier faint scuffs.

Open from the repository root:

```bash
blender assets/scenes/video_lab_pencil/video_lab.blend
```

The prior `../video_lab/` and `../video_lab_weathered/` packages are unchanged.
Their file hashes are recorded in `fallback_hashes.json` and checked during build.

Compare `preview_table_detail.png` with `comparison_before_detail.png`; both use
the same camera, lights and props. `preview_overview.png` shows the whole room.

## Adjust / remove / randomize

- Toggle collection **Pencil_Marks** to hide the added pencil strokes independently.
- Five `Pencil_Graphite_*` materials vary graphite pressure/color across strokes.
- The embedded `randomize_video_lab.py` traverses the Surface_Wear collection's
  children, so its existing wear-visibility sampling includes the new marks.
- In USD, `/Lab/Handling_Scuff_PencilMarks` references `pencil_marks.usdc`. Hide this
  root to disable the new marks. The existing runtime appearance helper recognizes
  its `Handling_Scuff` prefix and includes it in wear visibility randomization.
- Geometry, room, camera, props and table/floor collision remain as before. The
  new strokes are thin flat render-only meshes, with no collision or rigid body.
  They can affect rendered RGB/depth at their tiny 0.13 mm visual offset.

To regenerate another reproducible arrangement, density or contrast:

```bash
blender -b --threads 12 --python-exit-code 1 \
  --python scripts/blender/pencil_video_lab.py -- \
  --seed 42 --density 1.0 --contrast 1.0 --render
```

Density range is 0.1–3; contrast range is 0–1. Lower contrast makes the strokes
closer to the laminate color. Changing the seed varies strokes, pressure and
placement. Use `--output` for additional alternatives. This command overwrites
only the separate pencil-variant output, never its source package.

The packed `.blend` is standalone. Keep `pencil_marks.usdc`, `textures/` and the
other USD files together when moving the USD package. Pass this directory as
`asset_dir` to the existing optional RL scene helper. The manifest and
`pencil_config.json` describe the authored variant.

See `../video_lab/README.md` for baseline coordinates and physics limitations,
and `../video_lab_weathered/README.md` for the underlying dirt-material control.
No Isaac simulation or training was started for this appearance edit.
