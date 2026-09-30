#!/usr/bin/env python3
"""Sample a mesh to PCD and launch stock GPD's selected-grasp viewer."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    canonicalize_target_mesh,
    load_asset_mesh,
    relative_asset_mesh_path,
)
from grasp_planning.grasping.gpd_grasp_generator import (  # noqa: E402
    sample_mesh_surface,
    write_ascii_pcd,
    write_normals_csv,
)


def _safe_stem(path: str | Path) -> str:
    stem = Path(str(path)).with_suffix("").as_posix()
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)[-96:] or "mesh"


def _resolve_detect_grasps(path: str) -> str:
    if path:
        return path
    found = shutil.which("detect_grasps")
    return "" if found is None else found


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a repo mesh as a synthetic point cloud and open stock GPD's grasp viewer."
    )
    parser.add_argument(
        "--target-mesh",
        required=True,
        help="Input OBJ/STL path, relative to assets/ or absolute.",
    )
    parser.add_argument("--mesh-scale", type=float, default=0.01, help="Uniform scale applied while loading the mesh.")
    parser.add_argument("--num-samples", type=int, default=4096, help="Number of surface points to sample.")
    parser.add_argument("--rng-seed", type=int, default=0, help="Deterministic point sampling seed.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/gpd_viewer"),
        help="Directory for generated PCD and normals CSV.",
    )
    parser.add_argument(
        "--detect-grasps",
        default="",
        help="Path to stock GPD detect_grasps. If omitted, PATH is searched.",
    )
    parser.add_argument(
        "--gpd-config",
        default="",
        help="Path to a GPD cfg file such as /path/to/gpd/cfg/eigen_params.cfg.",
    )
    parser.add_argument(
        "--gpd-working-dir",
        default="",
        help="Optional working directory for detect_grasps, usually the GPD build directory.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only write the PCD/normals artifacts and print the GPD command.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mesh_obj_world = load_asset_mesh(args.target_mesh, scale=float(args.mesh_scale))
    mesh_local, source_pose = canonicalize_target_mesh(mesh_obj_world)
    samples = sample_mesh_surface(mesh_local, num_samples=int(args.num_samples), rng_seed=int(args.rng_seed))

    artifact_dir = args.artifact_dir
    if not artifact_dir.is_absolute():
        artifact_dir = REPO_ROOT / artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(args.target_mesh)
    pcd_path = artifact_dir / f"{stem}.pcd"
    normals_path = artifact_dir / f"{stem}_normals.csv"
    write_ascii_pcd(pcd_path, samples)
    write_normals_csv(normals_path, samples)

    print(f"[INFO] Wrote synthetic GPD point cloud: {pcd_path}", flush=True)
    print(f"[INFO] Wrote matching normals CSV:       {normals_path}", flush=True)
    print(
        "[INFO] Mesh source frame origin in loaded mesh frame: "
        f"{tuple(round(float(v), 6) for v in source_pose.position_world)}",
        flush=True,
    )
    print(f"[INFO] Target mesh: {relative_asset_mesh_path(args.target_mesh)}", flush=True)

    detect_grasps = _resolve_detect_grasps(args.detect_grasps)
    if not detect_grasps or not args.gpd_config:
        print("[INFO] GPD viewer not launched because detect_grasps or --gpd-config is missing.", flush=True)
        print(
            f"Run:\n  /path/to/gpd/build/detect_grasps /path/to/gpd/cfg/eigen_params.cfg {pcd_path}",
            flush=True,
        )
        return

    command = [detect_grasps, args.gpd_config, str(pcd_path)]
    print(f"[INFO] Running GPD viewer: {' '.join(command)}", flush=True)
    if args.no_run:
        return
    subprocess.run(
        command,
        cwd=None if not args.gpd_working_dir else args.gpd_working_dir,
        check=True,
    )


if __name__ == "__main__":
    main()
