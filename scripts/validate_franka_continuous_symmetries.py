#!/usr/bin/env python3
"""Validate inertia-axis hints against actual mesh surfaces at non-grid angles.

This is a deterministic geometric audit, not a proof of appearance or assembly
equivalence. Unlike nearest-vertex comparison, point-to-triangle distance does
not reject a cylinder simply because its angular tessellation is finite.
"""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ANGLES = [7.3, 17, 31, 43, 79, 137, 181.7, 223, 281, 337]
CONTINUOUS_TOLERANCE = 0.00015  # Separate, explicit tessellation allowance for round meshes.


def validate(task):
    key, mesh_path, scale, hint, tolerance, *angles = task
    angles = angles[0] if angles else ANGLES
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    mesh.vertices *= scale
    axis = np.asarray(hint["axis_obj"], dtype=float)
    axis /= np.linalg.norm(axis)
    center = mesh.center_mass if mesh.is_watertight else mesh.centroid
    samples, _ = trimesh.sample.sample_surface(mesh, 2048, seed=42023)
    probes = np.concatenate((np.asarray(mesh.vertices)[:: max(1, len(mesh.vertices) // 256)], samples[:256]))
    full = np.concatenate((np.asarray(mesh.vertices), samples))
    checked = []
    accepted = True
    for stage, points in [("probe", probes), ("full_vertices_and_surface", full)]:
        for angle in angles:
            r = Rotation.from_rotvec(axis * np.deg2rad(angle)).as_matrix()
            rotated = (points - center) @ r.T + center
            distances = []
            for start in range(0, len(points), 256):
                _, ds, _ = trimesh.proximity.closest_point(mesh, rotated[start : start + 256])
                distances.extend(ds.tolist())
            peak = float(np.max(distances))
            checked.append(dict(stage=stage, angle_deg=angle, samples=len(points), max_surface_error_m=peak))
            if not np.isfinite(peak) or peak > tolerance:
                accepted = False
                break
        if not accepted:
            break
    result = dict(
        part=key,
        accepted=accepted,
        type="continuous_axial",
        axis_obj=axis.tolist(),
        center_obj_m=center.tolist(),
        mesh_scale=scale,
        mesh_path=str(mesh_path.relative_to(ROOT)),
        mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        max_surface_error_m=max(r["max_surface_error_m"] for r in checked),
        checks=checked,
    )
    print(key, "ACCEPT" if accepted else "reject", round(result["max_surface_error_m"] * 1000, 5), "mm", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coverage", type=Path, default=ROOT / "artifacts/franka_symmetry_evaluation_20260923/catalog_coverage.json"
    )
    p.add_argument("--output", type=Path, default=ROOT / "assets/obj/fabrica/continuous_symmetries.json")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    parts = sorted({r["part"] for r in json.loads(args.coverage.read_text())["targets"]})
    tasks = []
    for key in parts:
        assembly, part = key.split("__part_")
        data = json.loads((ROOT / f"assets/obj/fabrica/{assembly}/symmetries.json").read_text())
        for hint in data["parts"][part].get("continuous_symmetries", []):
            tasks.append(
                (
                    key,
                    ROOT / f"assets/obj/fabrica/{assembly}/{part}.obj",
                    data["mesh_scale"],
                    hint,
                    CONTINUOUS_TOLERANCE,
                )
            )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(validate, tasks))
    finite_tasks = []
    for task, record in zip(tasks, records):
        if not record["accepted"]:
            for angle in sorted(set(range(30, 360, 30)) | {45, 135, 225, 315}):
                finite_tasks.append((*task[:4], CONTINUOUS_TOLERANCE, [angle]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        finite_records = list(pool.map(validate, finite_tasks))
    for record, task in zip(finite_records, finite_tasks):
        record["type"] = "finite_rotation"
        record["angle_deg"] = task[-1][0]
        record["name"] = f"surface_axis_{record['angle_deg']}_deg"
        r = Rotation.from_rotvec(np.asarray(record["axis_obj"]) * np.deg2rad(record["angle_deg"])).as_matrix()
        c = np.asarray(record["center_obj_m"])
        t = np.eye(4)
        t[:3, :3] = r
        t[:3, 3] = c - r @ c
        record["matrix_obj"] = t.tolist()
    payload = dict(
        schema_version=1,
        method="rotated_full_vertices_and_seeded_surface_to_triangles",
        tolerance_m=CONTINUOUS_TOLERANCE,
        angles_deg=ANGLES,
        surface_samples=2048,
        limitation="Sampled geometric validation; no texture, material or functional equivalence claim.",
        finite_rotations={key: [r for r in finite_records if r["part"] == key] for key in parts},
        parts={key: [r for r in records if r["part"] == key] for key in parts},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
