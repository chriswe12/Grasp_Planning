Put `.stl` files for `scripts/debug_mesh_antipodal_grasps.py` in this folder.

Examples:

```bash
python scripts/debug_mesh_antipodal_grasps.py --geometry stl --stl-path my_part.stl
python scripts/debug_mesh_antipodal_grasps.py --geometry stl --stl-path my_part.stl --stl-scale 0.001
```

Notes:

- Relative `--stl-path` values are resolved relative to this folder.
- `--stl-scale 0.001` is useful for STL files authored in millimeters.
- The grasp generator expects a closed triangle mesh with consistent outward winding.
- Default settings now live in `configs/mesh_antipodal_grasp_debug.yaml`.
