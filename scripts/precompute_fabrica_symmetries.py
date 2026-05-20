#!/usr/bin/env python3
"""Precompute object-frame rotational symmetries for Fabrica OBJ assets.

The detector is intentionally offline and conservative: it generates a bounded
set of candidate proper rotations, validates each candidate against the loaded
mesh geometry, writes accepted transforms to per-assembly ``symmetries.json``
files, and emits an HTML report with accepted transforms plus the closest
rejected candidates for manual inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

try:
    import trimesh
except Exception as exc:  # pragma: no cover - import-time dependency guard
    raise RuntimeError("trimesh is required for symmetry precomputation.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = REPO_ROOT / "assets" / "obj" / "fabrica"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "fabrica_symmetries"
DEFAULT_MESH_SCALE = 0.01
DEFAULT_ORDERS = (2, 3, 4, 6)


@dataclass(frozen=True)
class DetectionConfig:
    mesh_scale: float = DEFAULT_MESH_SCALE
    tolerance_m: float = 0.001
    sample_count: int = 10_000
    max_validation_vertices: int = 5_000
    visual_sample_count: int = 1_400
    validation_seed: int = 17
    visual_seed: int = 23
    orders: tuple[int, ...] = DEFAULT_ORDERS
    max_candidate_axes: int = 48
    max_face_axes: int = 12
    axis_grid_samples: int = 0
    near_miss_count: int = 12
    coverage_multiplier: float = 1.5
    max_distance_multiplier: float = 6.0


@dataclass(frozen=True)
class AxisRecord:
    name: str
    source: str
    axis: np.ndarray


@dataclass(frozen=True)
class CenterRecord:
    name: str
    point: np.ndarray


@dataclass(frozen=True)
class CandidateTransform:
    name: str
    source: str
    matrix: np.ndarray
    axis: np.ndarray | None
    center: np.ndarray | None
    angle_deg: float
    order: int | None
    step: int | None


def _clean_float(value: float, *, digits: int = 10) -> float:
    rounded = round(float(value), digits)
    if abs(rounded) < 10**-digits:
        return 0.0
    return rounded


def _vector_payload(values: Sequence[float] | np.ndarray, *, digits: int = 10) -> list[float]:
    return [_clean_float(float(value), digits=digits) for value in values]


def _matrix_payload(matrix: np.ndarray, *, digits: int = 10) -> list[list[float]]:
    return [_vector_payload(row, digits=digits) for row in np.asarray(matrix, dtype=float)]


def _normalize_axis(axis: Sequence[float] | np.ndarray) -> np.ndarray | None:
    values = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12 or not np.isfinite(norm):
        return None
    values = values / norm
    first_nonzero = int(np.argmax(np.abs(values)))
    if values[first_nonzero] < 0.0:
        values = -values
    return values


def _add_axis(records: list[AxisRecord], name: str, source: str, axis: Sequence[float] | np.ndarray) -> None:
    normalized = _normalize_axis(axis)
    if normalized is None:
        return
    for existing in records:
        if abs(float(np.dot(existing.axis, normalized))) > 0.99999:
            return
    records.append(AxisRecord(name=name, source=source, axis=normalized))


def _basis_combo_axes(records: list[AxisRecord], prefix: str, source: str, basis: np.ndarray) -> None:
    basis = np.asarray(basis, dtype=float)
    if basis.shape != (3, 3):
        return
    axes = [_normalize_axis(basis[:, index]) for index in range(3)]
    if any(axis is None for axis in axes):
        return
    e0, e1, e2 = [axis for axis in axes if axis is not None]
    pair_specs = [
        ("xy_sum", e0 + e1),
        ("xy_diff", e0 - e1),
        ("xz_sum", e0 + e2),
        ("xz_diff", e0 - e2),
        ("yz_sum", e1 + e2),
        ("yz_diff", e1 - e2),
    ]
    for suffix, axis in pair_specs:
        _add_axis(records, f"{prefix}_{suffix}", source, axis)
    for signs in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)):
        axis = signs[0] * e0 + signs[1] * e1 + signs[2] * e2
        _add_axis(records, f"{prefix}_diag_{signs[0]}_{signs[1]}_{signs[2]}", source, axis)


def _fibonacci_axes(count: int) -> Iterable[np.ndarray]:
    if count <= 0:
        return
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        z_value = 1.0 - 2.0 * ((index + 0.5) / count)
        radius = math.sqrt(max(0.0, 1.0 - z_value * z_value))
        theta = golden_angle * index
        yield np.array([math.cos(theta) * radius, math.sin(theta) * radius, z_value], dtype=float)


def _candidate_axes(mesh: "trimesh.Trimesh", config: DetectionConfig) -> list[AxisRecord]:
    records: list[AxisRecord] = []
    object_basis = np.eye(3)
    for index, axis_name in enumerate(("x", "y", "z")):
        _add_axis(records, f"object_{axis_name}", "object_frame", object_basis[:, index])
    _basis_combo_axes(records, "object", "object_frame_combinations", object_basis)

    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) >= 3:
        centered = vertices - np.mean(vertices, axis=0)
        try:
            _, eigenvectors = np.linalg.eigh(np.cov(centered.T))
            pca_basis = eigenvectors[:, ::-1]
            for index in range(3):
                _add_axis(records, f"pca_{index}", "vertex_pca", pca_basis[:, index])
            _basis_combo_axes(records, "pca", "vertex_pca_combinations", pca_basis)
        except np.linalg.LinAlgError:
            pass

    try:
        obb_transform = np.asarray(mesh.bounding_box_oriented.primitive.transform, dtype=float)
        obb_basis = obb_transform[:3, :3]
        for index in range(3):
            _add_axis(records, f"obb_{index}", "oriented_bounds", obb_basis[:, index])
        _basis_combo_axes(records, "obb", "oriented_bounds_combinations", obb_basis)
    except Exception:
        pass

    try:
        face_order = np.argsort(np.asarray(mesh.area_faces, dtype=float))[::-1]
        face_axes = 0
        for face_index in face_order:
            if face_axes >= config.max_face_axes:
                break
            normal = np.asarray(mesh.face_normals[int(face_index)], dtype=float)
            before = len(records)
            _add_axis(records, f"face_normal_{int(face_index)}", "large_face_normal", normal)
            if len(records) > before:
                face_axes += 1
    except Exception:
        pass

    try:
        symmetry_axis = getattr(mesh, "symmetry_axis", None)
        if symmetry_axis is not None:
            _add_axis(records, "trimesh_symmetry_axis", "trimesh", np.asarray(symmetry_axis, dtype=float))
    except Exception:
        pass

    for index, axis in enumerate(_fibonacci_axes(config.axis_grid_samples)):
        _add_axis(records, f"grid_{index}", "fibonacci_grid", axis)

    return records[: max(1, config.max_candidate_axes)]


def _candidate_centers(mesh: "trimesh.Trimesh", tolerance_m: float) -> list[CenterRecord]:
    records: list[CenterRecord] = []

    def add_center(name: str, point: Sequence[float] | np.ndarray) -> None:
        values = np.asarray(point, dtype=float).reshape(3)
        if not np.all(np.isfinite(values)):
            return
        for existing in records:
            if np.linalg.norm(existing.point - values) <= max(tolerance_m * 0.25, 1e-8):
                return
        records.append(CenterRecord(name=name, point=values))

    bounds = np.asarray(mesh.bounds, dtype=float)
    add_center("bounds_center", 0.5 * (bounds[0] + bounds[1]))
    try:
        add_center("center_mass", np.asarray(mesh.center_mass, dtype=float))
    except Exception:
        pass
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices):
        add_center("vertex_mean", np.mean(vertices, axis=0))
    add_center("object_origin", np.zeros(3, dtype=float))
    return records


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x_value, y_value, z_value = axis
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    one_minus_cosine = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x_value * x_value * one_minus_cosine,
                x_value * y_value * one_minus_cosine - z_value * sine,
                x_value * z_value * one_minus_cosine + y_value * sine,
            ],
            [
                y_value * x_value * one_minus_cosine + z_value * sine,
                cosine + y_value * y_value * one_minus_cosine,
                y_value * z_value * one_minus_cosine - x_value * sine,
            ],
            [
                z_value * x_value * one_minus_cosine - y_value * sine,
                z_value * y_value * one_minus_cosine + x_value * sine,
                cosine + z_value * z_value * one_minus_cosine,
            ],
        ],
        dtype=float,
    )


def _transform_about_center(axis: np.ndarray, center: np.ndarray, angle_rad: float) -> np.ndarray:
    rotation = _rotation_matrix(axis, angle_rad)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = center - rotation @ center
    return matrix


def _apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _load_mesh(path: Path, *, scale: float) -> "trimesh.Trimesh":
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No geometry found in scene '{path}'.")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh for '{path}', got {type(loaded).__name__}.")
    mesh = loaded.copy()
    if scale <= 0.0:
        raise ValueError("mesh_scale must be > 0.")
    mesh.apply_scale(float(scale))
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh '{path}' has no vertices or faces.")
    return mesh


def _sample_surface(mesh: "trimesh.Trimesh", count: int, *, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 3), dtype=float)
    samples, _ = trimesh.sample.sample_surface(mesh, int(count), seed=seed)
    return np.asarray(samples, dtype=float)


def _validation_vertices(vertices: np.ndarray, *, max_count: int, seed: int) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if max_count <= 0 or len(vertices) <= max_count:
        return vertices
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(vertices), size=int(max_count), replace=False))
    return vertices[indices]


def _cloud_spacing(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    distances, _ = cKDTree(points).query(points, k=2)
    return float(np.percentile(distances[:, 1], 95.0))


def _distance_metrics(distances: np.ndarray) -> dict[str, float]:
    distances = np.asarray(distances, dtype=float)
    if len(distances) == 0:
        return {"mean_m": 0.0, "p95_m": 0.0, "p99_m": 0.0, "max_m": 0.0}
    return {
        "mean_m": float(np.mean(distances)),
        "p95_m": float(np.percentile(distances, 95.0)),
        "p99_m": float(np.percentile(distances, 99.0)),
        "max_m": float(np.max(distances)),
    }


def _evaluate_transform(
    candidate: CandidateTransform,
    *,
    vertices: np.ndarray,
    vertex_tree: cKDTree,
    validation_points: np.ndarray,
    validation_tree: cKDTree,
    base_tolerance_m: float,
    effective_tolerance_m: float,
    max_distance_m: float,
) -> dict[str, Any]:
    transformed_vertices = _apply_transform(vertices, candidate.matrix)
    vertex_distances, _ = vertex_tree.query(transformed_vertices)
    vertex_metrics = _distance_metrics(vertex_distances)
    if vertex_metrics["p99_m"] > base_tolerance_m:
        return {
            "accepted": False,
            "validation_mode": "vertex_prefilter",
            "effective_tolerance_m": float(effective_tolerance_m),
            "max_allowed_distance_m": float(max_distance_m),
            "vertex_mean_m": vertex_metrics["mean_m"],
            "vertex_p95_m": vertex_metrics["p95_m"],
            "vertex_p99_m": vertex_metrics["p99_m"],
            "vertex_max_m": vertex_metrics["max_m"],
            **vertex_metrics,
        }
    transformed = _apply_transform(validation_points, candidate.matrix)
    distances, _ = validation_tree.query(transformed)
    metrics = _distance_metrics(distances)
    accepted = (
        vertex_metrics["p99_m"] <= base_tolerance_m
        and metrics["p99_m"] <= effective_tolerance_m
        and metrics["max_m"] <= max_distance_m
    )
    return {
        "accepted": bool(accepted),
        "validation_mode": "surface",
        "effective_tolerance_m": float(effective_tolerance_m),
        "max_allowed_distance_m": float(max_distance_m),
        "vertex_mean_m": vertex_metrics["mean_m"],
        "vertex_p95_m": vertex_metrics["p95_m"],
        "vertex_p99_m": vertex_metrics["p99_m"],
        "vertex_max_m": vertex_metrics["max_m"],
        **metrics,
    }


def _candidate_transforms(
    axes: Sequence[AxisRecord],
    centers: Sequence[CenterRecord],
    config: DetectionConfig,
) -> Iterable[CandidateTransform]:
    for axis_record in axes:
        for center_record in centers:
            for order in config.orders:
                if order < 2:
                    continue
                for step in range(1, order):
                    angle_deg = 360.0 * step / order
                    matrix = _transform_about_center(axis_record.axis, center_record.point, math.radians(angle_deg))
                    yield CandidateTransform(
                        name=f"{axis_record.name}_{center_record.name}_order{order}_step{step}",
                        source=f"{axis_record.source}:{center_record.name}",
                        matrix=matrix,
                        axis=axis_record.axis,
                        center=center_record.point,
                        angle_deg=angle_deg,
                        order=order,
                        step=step,
                    )


def _candidate_payload(candidate: CandidateTransform, validation: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": candidate.name,
        "type": "finite_rotation" if candidate.order else "identity",
        "source": candidate.source,
        "angle_deg": _clean_float(candidate.angle_deg, digits=8),
        "order": candidate.order,
        "step": candidate.step,
        "matrix_obj": _matrix_payload(candidate.matrix),
        "translation_obj_m": _vector_payload(candidate.matrix[:3, 3]),
        "validation": {
            key: _clean_float(value, digits=10) if isinstance(value, float) else value
            for key, value in validation.items()
        },
    }
    if candidate.axis is not None:
        payload["axis_obj"] = _vector_payload(candidate.axis)
    if candidate.center is not None:
        payload["center_obj_m"] = _vector_payload(candidate.center)
    return payload


def _matrix_is_duplicate(matrix: np.ndarray, accepted: Sequence[dict[str, Any]], *, tolerance: float) -> bool:
    for record in accepted:
        existing = np.asarray(record["matrix_obj"], dtype=float)
        if np.allclose(existing, matrix, atol=max(tolerance * 0.1, 1e-8), rtol=0.0):
            return True
    return False


def _transform_effect_is_duplicate(
    matrix: np.ndarray,
    accepted: Sequence[dict[str, Any]],
    *,
    probe_points: np.ndarray,
    tolerance: float,
) -> bool:
    transformed = _apply_transform(probe_points, matrix)
    for record in accepted:
        existing = np.asarray(record["matrix_obj"], dtype=float)
        existing_transformed = _apply_transform(probe_points, existing)
        distances = np.linalg.norm(transformed - existing_transformed, axis=1)
        if float(np.percentile(distances, 99.0)) <= tolerance:
            return True
    return False


def _continuous_symmetry_payload(mesh: "trimesh.Trimesh") -> list[dict[str, Any]]:
    continuous: list[dict[str, Any]] = []
    try:
        symmetry_kind = getattr(mesh, "symmetry", None)
    except Exception:
        symmetry_kind = None
    if symmetry_kind == "radial":
        try:
            axis = _normalize_axis(np.asarray(mesh.symmetry_axis, dtype=float))
        except Exception:
            axis = None
        if axis is not None:
            continuous.append(
                {
                    "type": "continuous_radial",
                    "axis_obj": _vector_payload(axis),
                    "note": "Trimesh inertia-based radial symmetry candidate; finite samples are validated separately.",
                }
            )
    elif symmetry_kind == "spherical":
        continuous.append(
            {
                "type": "continuous_spherical",
                "note": "Trimesh inertia-based spherical symmetry candidate; finite samples are validated separately.",
            }
        )
    return continuous


def detect_mesh_symmetries(
    mesh: "trimesh.Trimesh",
    *,
    assembly: str,
    part_id: str,
    mesh_path: str,
    config: DetectionConfig,
) -> dict[str, Any]:
    surface_points = _sample_surface(mesh, config.sample_count, seed=config.validation_seed)
    mesh_vertices = np.asarray(mesh.vertices, dtype=float)
    vertices = _validation_vertices(
        mesh_vertices,
        max_count=config.max_validation_vertices,
        seed=config.validation_seed + 1009,
    )
    validation_points = np.vstack([vertices, surface_points])
    validation_tree = cKDTree(validation_points)
    vertex_tree = cKDTree(vertices)
    probe_stride = max(1, len(validation_points) // 2_000)
    probe_points = validation_points[::probe_stride]
    spacing_m = _cloud_spacing(surface_points)
    effective_tolerance_m = float(config.tolerance_m + config.coverage_multiplier * spacing_m)
    max_distance_m = max(
        float(config.tolerance_m * config.max_distance_multiplier),
        float(effective_tolerance_m * 2.5),
    )

    identity = CandidateTransform(
        name="identity",
        source="identity",
        matrix=np.eye(4, dtype=float),
        axis=None,
        center=None,
        angle_deg=0.0,
        order=None,
        step=None,
    )
    identity_validation = _evaluate_transform(
        identity,
        vertices=vertices,
        vertex_tree=vertex_tree,
        validation_points=validation_points,
        validation_tree=validation_tree,
        base_tolerance_m=config.tolerance_m,
        effective_tolerance_m=effective_tolerance_m,
        max_distance_m=max_distance_m,
    )
    accepted: list[dict[str, Any]] = [_candidate_payload(identity, identity_validation)]
    rejected: list[dict[str, Any]] = []
    axes = _candidate_axes(mesh, config)
    centers = _candidate_centers(mesh, config.tolerance_m)
    candidates_tested = 0

    for candidate in _candidate_transforms(axes, centers, config):
        candidates_tested += 1
        validation = _evaluate_transform(
            candidate,
            vertices=vertices,
            vertex_tree=vertex_tree,
            validation_points=validation_points,
            validation_tree=validation_tree,
            base_tolerance_m=config.tolerance_m,
            effective_tolerance_m=effective_tolerance_m,
            max_distance_m=max_distance_m,
        )
        payload = _candidate_payload(candidate, validation)
        if validation["accepted"]:
            if not _matrix_is_duplicate(
                candidate.matrix,
                accepted,
                tolerance=config.tolerance_m,
            ) and not _transform_effect_is_duplicate(
                candidate.matrix,
                accepted,
                probe_points=probe_points,
                tolerance=effective_tolerance_m,
            ):
                accepted.append(payload)
            continue
        rejected.append(payload)

    rejected.sort(key=lambda record: (record["validation"]["p99_m"], record["validation"]["max_m"]))
    near_misses = rejected[: config.near_miss_count]
    visual_points = _sample_surface(mesh, config.visual_sample_count, seed=config.visual_seed)
    bounds = np.asarray(mesh.bounds, dtype=float)

    return {
        "assembly": assembly,
        "part_id": part_id,
        "mesh_path": mesh_path,
        "mesh_scale": float(config.mesh_scale),
        "frame": "object",
        "pose_equivalence": "T_world_object_equivalent = T_world_object @ matrix_obj",
        "bounds_obj_m": [_vector_payload(bounds[0]), _vector_payload(bounds[1])],
        "extent_m": _clean_float(float(np.max(bounds[1] - bounds[0]))),
        "tolerance_m": float(config.tolerance_m),
        "effective_tolerance_m": _clean_float(effective_tolerance_m),
        "sample_spacing_p95_m": _clean_float(spacing_m),
        "candidate_summary": {
            "axes_tested": len(axes),
            "centers_tested": len(centers),
            "candidates_tested": candidates_tested,
            "accepted_count": len(accepted),
            "near_miss_count": len(near_misses),
            "orders": list(config.orders),
            "axis_sources": sorted({axis.source for axis in axes}),
            "center_names": [center.name for center in centers],
            "mesh_vertices": int(len(mesh_vertices)),
            "validation_vertices": int(len(vertices)),
            "surface_samples": int(len(surface_points)),
        },
        "continuous_symmetries": _continuous_symmetry_payload(mesh),
        "symmetries": accepted,
        "near_misses": near_misses,
        "visual_points_obj": [_vector_payload(point, digits=6) for point in visual_points],
    }


def detect_part_symmetries(path: Path, *, assembly: str, part_id: str, config: DetectionConfig) -> dict[str, Any]:
    mesh = _load_mesh(path, scale=config.mesh_scale)
    mesh_path = path.relative_to(REPO_ROOT / "assets").as_posix()
    return detect_mesh_symmetries(mesh, assembly=assembly, part_id=part_id, mesh_path=mesh_path, config=config)


def _asset_payload(assembly: str, parts: Sequence[dict[str, Any]], config: DetectionConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_by": "scripts/precompute_fabrica_symmetries.py",
        "assembly": assembly,
        "mesh_scale": float(config.mesh_scale),
        "tolerance_m": float(config.tolerance_m),
        "frame": "object",
        "pose_equivalence": "T_world_object_equivalent = T_world_object @ matrix_obj",
        "notes": [
            "Only proper rotations are considered; reflections are intentionally excluded.",
            "Identity is always included. Other transforms are accepted only after geometry validation.",
            "Runtime code should treat these as precomputed candidates, not as proof of functional assembly equivalence.",
        ],
        "parts": {
            str(part["part_id"]): {
                key: part[key]
                for key in (
                    "mesh_path",
                    "frame",
                    "pose_equivalence",
                    "bounds_obj_m",
                    "extent_m",
                    "tolerance_m",
                    "effective_tolerance_m",
                    "sample_spacing_p95_m",
                    "candidate_summary",
                    "continuous_symmetries",
                    "symmetries",
                )
            }
            for part in parts
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _numeric_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (0, f"{int(path.stem):08d}")
    except ValueError:
        return (1, path.stem)


def _selected_assemblies(asset_root: Path, names: Sequence[str]) -> list[Path]:
    if names:
        assemblies = [asset_root / name for name in names]
    else:
        assemblies = [path for path in asset_root.iterdir() if path.is_dir()]
    missing = [path for path in assemblies if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Assembly directories not found: {', '.join(str(path) for path in missing)}")
    return sorted(assemblies, key=lambda path: path.name)


def _selected_parts(assembly_dir: Path, part_ids: Sequence[str]) -> list[Path]:
    if part_ids:
        parts = [assembly_dir / f"{part_id}.obj" for part_id in part_ids]
    else:
        parts = sorted(assembly_dir.glob("*.obj"), key=_numeric_sort_key)
    missing = [path for path in parts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Part OBJ files not found: {', '.join(str(path) for path in missing)}")
    return parts


def _html_document(data_json: str) -> str:
    return (
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fabrica Symmetry Inspection</title>
  <style>
    :root {
      --bg: #f6f4ee;
      --panel: #fffdf8;
      --ink: #1f2522;
      --muted: #68716c;
      --line: #d9d4c7;
      --mesh: #475569;
      --accepted: #0f766e;
      --near: #b91c1c;
      --axis: #d97706;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    .layout { display: grid; grid-template-columns: 380px minmax(0, 1fr); min-height: 100vh; }
    aside { border-right: 1px solid var(--line); background: var(--panel); padding: 18px; overflow: auto; }
    main { padding: 18px; overflow: auto; }
    h1 { margin: 0 0 8px; font-size: 24px; line-height: 1.15; }
    .subtitle { margin: 0 0 14px; color: var(--muted); font-size: 13px; line-height: 1.4; }
    label { display: grid; gap: 5px; margin-bottom: 10px; color: var(--muted); font-size: 12px; }
    select { width: 100%; border: 1px solid var(--line); border-radius: 8px; background: #fff; color: var(--ink); padding: 8px; font: inherit; }
    .controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 12px 0; }
    button { border: 1px solid var(--line); border-radius: 8px; background: #fff; color: var(--ink); padding: 9px 10px; font: inherit; cursor: pointer; }
    button:hover { border-color: var(--accepted); }
    .list { display: grid; gap: 8px; margin-top: 12px; }
    .item { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 9px; text-align: left; cursor: pointer; }
    .item.active { border-color: var(--accepted); box-shadow: 0 0 0 2px rgba(15,118,110,0.13); }
    .item.near.active { border-color: var(--near); box-shadow: 0 0 0 2px rgba(185,28,28,0.13); }
    .item-title { display: flex; justify-content: space-between; gap: 8px; font-weight: 700; font-size: 13px; }
    .item-meta { margin-top: 4px; color: var(--muted); font-family: "IBM Plex Mono", monospace; font-size: 11px; line-height: 1.35; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr); gap: 16px; align-items: start; }
    .panel { border: 1px solid var(--line); border-radius: 8px; background: rgba(255,253,248,0.96); padding: 14px; }
    #scene { width: 100%; aspect-ratio: 1.45 / 1; display: block; border-radius: 8px; background: linear-gradient(180deg, #ffffff, #ebe7dc); }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; color: var(--muted); font-size: 12px; }
    .legend span { display: inline-flex; align-items: center; gap: 7px; }
    .swatch { width: 12px; height: 12px; border-radius: 999px; display: inline-block; }
    .kv { white-space: pre-wrap; font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.55; margin: 0; }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>Fabrica Symmetry Inspection</h1>
      <p class="subtitle">Accepted transforms overlay in green. Closest rejected candidates overlay in red.</p>
      <label>Assembly<select id="assemblySelect"></select></label>
      <label>Part<select id="partSelect"></select></label>
      <label>Transform<select id="transformSelect"></select></label>
      <div class="controls">
        <button id="prevPartBtn" type="button">Prev Part</button>
        <button id="nextPartBtn" type="button">Next Part</button>
        <button id="prevTransformBtn" type="button">Prev Sym</button>
        <button id="nextTransformBtn" type="button">Next Sym</button>
        <button id="nearBtn" type="button">Near Misses: On</button>
        <button id="resetBtn" type="button">Reset View</button>
      </div>
      <div id="transformList" class="list"></div>
    </aside>
    <main>
      <div class="grid">
        <section class="panel">
          <svg id="scene" viewBox="0 0 1100 760"></svg>
          <div class="legend">
            <span><i class="swatch" style="background: var(--mesh)"></i>Original</span>
            <span><i class="swatch" style="background: var(--accepted)"></i>Accepted transform</span>
            <span><i class="swatch" style="background: var(--near)"></i>Rejected near miss</span>
            <span><i class="swatch" style="background: var(--axis)"></i>Rotation axis</span>
          </div>
        </section>
        <section class="panel">
          <pre id="details" class="kv"></pre>
        </section>
      </div>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const assemblySelect = document.getElementById("assemblySelect");
    const partSelect = document.getElementById("partSelect");
    const transformSelect = document.getElementById("transformSelect");
    const transformList = document.getElementById("transformList");
    const details = document.getElementById("details");
    const scene = document.getElementById("scene");
    const initialView = { yaw: -0.72, pitch: 0.52, zoom: 1.0, panX: 0, panY: 0 };
    const state = {
      assembly: "",
      partIndex: 0,
      transformIndex: 0,
      showNear: true,
      dragging: false,
      dragMode: "rotate",
      pointerId: null,
      lastX: 0,
      lastY: 0,
      ...initialView,
    };
    const assemblies = [...new Set(data.parts.map((part) => part.assembly))];
    function option(label, value) {
      const node = document.createElement("option");
      node.value = value;
      node.textContent = label;
      return node;
    }
    function partsForAssembly() {
      return data.parts.filter((part) => part.assembly === state.assembly);
    }
    function currentPart() {
      const parts = partsForAssembly();
      return parts[Math.max(0, Math.min(state.partIndex, parts.length - 1))] || null;
    }
    function transformsForPart(part) {
      if (!part) return [];
      const accepted = part.symmetries.map((record) => ({ ...record, group: "accepted" }));
      const near = state.showNear ? part.near_misses.map((record) => ({ ...record, group: "near" })) : [];
      return [...accepted, ...near];
    }
    function currentTransform() {
      const part = currentPart();
      const transforms = transformsForPart(part);
      return transforms[Math.max(0, Math.min(state.transformIndex, transforms.length - 1))] || null;
    }
    function fmt(value, digits = 4) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return Number(value).toFixed(digits);
    }
    function applyMatrix(point, matrix) {
      return [
        point[0] * matrix[0][0] + point[1] * matrix[0][1] + point[2] * matrix[0][2] + matrix[0][3],
        point[0] * matrix[1][0] + point[1] * matrix[1][1] + point[2] * matrix[1][2] + matrix[1][3],
        point[0] * matrix[2][0] + point[1] * matrix[2][1] + point[2] * matrix[2][2] + matrix[2][3],
      ];
    }
    function allPoints(part, transform) {
      const points = [...part.visual_points_obj];
      if (transform) points.push(...part.visual_points_obj.map((point) => applyMatrix(point, transform.matrix_obj)));
      if (transform && transform.center_obj_m && transform.axis_obj) {
        points.push(transform.center_obj_m);
        const extent = part.extent_m || 0.1;
        points.push(transform.center_obj_m.map((value, index) => value + transform.axis_obj[index] * extent * 0.7));
        points.push(transform.center_obj_m.map((value, index) => value - transform.axis_obj[index] * extent * 0.7));
      }
      return points;
    }
    function boundsFor(points) {
      return points.reduce((acc, point) => {
        point.forEach((value, index) => {
          acc.min[index] = Math.min(acc.min[index], value);
          acc.max[index] = Math.max(acc.max[index], value);
        });
        return acc;
      }, { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
    }
    function rotate(point, center) {
      const shifted = point.map((value, index) => value - center[index]);
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];
      return [x1, cp * y1 + sp * z1, -sp * y1 + cp * z1];
    }
    function project(point, center, scale) {
      const [x, y, z] = rotate(point, center);
      return { x: 550 + state.panX + x * scale, y: 380 + state.panY - y * scale, depth: z };
    }
    function addSvg(tag, attrs) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      scene.appendChild(node);
      return node;
    }
    function drawPoint(point, center, scale, fill, radius, opacity) {
      const projected = project(point, center, scale);
      addSvg("circle", { cx: projected.x, cy: projected.y, r: radius, fill, "fill-opacity": opacity });
    }
    function drawLine(a, b, center, scale, stroke, width) {
      const pa = project(a, center, scale);
      const pb = project(b, center, scale);
      addSvg("line", { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, stroke, "stroke-width": width, "stroke-opacity": 0.95 });
    }
    function renderScene() {
      scene.replaceChildren();
      const part = currentPart();
      const transform = currentTransform();
      if (!part || !transform) return;
      const points = allPoints(part, transform);
      const bounds = boundsFor(points);
      const center = bounds.min.map((value, index) => 0.5 * (value + bounds.max[index]));
      const extent = Math.max(...bounds.max.map((value, index) => value - bounds.min[index]), 0.04);
      const scale = 560 / extent * state.zoom;
      part.visual_points_obj.forEach((point) => drawPoint(point, center, scale, "#475569", 2.2, 0.28));
      const transformedPoints = part.visual_points_obj.map((point) => applyMatrix(point, transform.matrix_obj));
      const color = transform.group === "near" ? "#b91c1c" : "#0f766e";
      transformedPoints.forEach((point) => drawPoint(point, center, scale, color, 2.3, 0.48));
      if (transform.center_obj_m && transform.axis_obj) {
        const axisExtent = Math.max(part.extent_m || 0.04, 0.04) * 0.85;
        const start = transform.center_obj_m.map((value, index) => value - transform.axis_obj[index] * axisExtent);
        const end = transform.center_obj_m.map((value, index) => value + transform.axis_obj[index] * axisExtent);
        drawLine(start, end, center, scale, "#d97706", 3);
        drawPoint(transform.center_obj_m, center, scale, "#d97706", 4.5, 1.0);
      }
    }
    function renderDetails() {
      const part = currentPart();
      const transform = currentTransform();
      if (!part || !transform) {
        details.textContent = "";
        return;
      }
      const validation = transform.validation || {};
      details.textContent = [
        `assembly: ${part.assembly}`,
        `part: ${part.part_id}`,
        `mesh: ${part.mesh_path}`,
        `accepted symmetries: ${part.symmetries.length}`,
        `near misses shown: ${state.showNear ? part.near_misses.length : 0}`,
        `candidates tested: ${part.candidate_summary.candidates_tested}`,
        `axes tested: ${part.candidate_summary.axes_tested}`,
        `vertices checked: ${part.candidate_summary.validation_vertices} / ${part.candidate_summary.mesh_vertices}`,
        `surface samples: ${part.candidate_summary.surface_samples}`,
        `centers: ${part.candidate_summary.center_names.join(", ")}`,
        "",
        `selected: ${transform.name}`,
        `group: ${transform.group}`,
        `angle_deg: ${fmt(transform.angle_deg, 3)}`,
        `axis_obj: ${transform.axis_obj ? transform.axis_obj.map((v) => fmt(v, 4)).join(", ") : "n/a"}`,
        `center_obj_m: ${transform.center_obj_m ? transform.center_obj_m.map((v) => fmt(v, 5)).join(", ") : "n/a"}`,
        `validation_mode: ${validation.validation_mode || "n/a"}`,
        `vertex_p99_m: ${fmt(validation.vertex_p99_m, 6)}`,
        `p99_m: ${fmt(validation.p99_m, 6)}`,
        `max_m: ${fmt(validation.max_m, 6)}`,
        `effective_tol_m: ${fmt(validation.effective_tolerance_m, 6)}`,
        "",
        `matrix_obj:`,
        JSON.stringify(transform.matrix_obj),
      ].join("\\n");
    }
    function renderTransformControls() {
      const part = currentPart();
      const transforms = transformsForPart(part);
      transformSelect.replaceChildren();
      transformList.replaceChildren();
      transforms.forEach((transform, index) => {
        const label = `${transform.group === "near" ? "near" : "ok"} ${transform.name}`;
        transformSelect.appendChild(option(label, String(index)));
        const item = document.createElement("button");
        item.type = "button";
        item.className = `item ${transform.group === "near" ? "near" : ""} ${index === state.transformIndex ? "active" : ""}`;
        item.innerHTML = `<div class="item-title"><span>${transform.name}</span><span>${transform.group}</span></div>
          <div class="item-meta">angle ${fmt(transform.angle_deg, 1)} deg | p99 ${fmt(transform.validation?.p99_m, 5)} | max ${fmt(transform.validation?.max_m, 5)}</div>`;
        item.addEventListener("click", () => {
          state.transformIndex = index;
          renderAll();
        });
        transformList.appendChild(item);
      });
      state.transformIndex = Math.max(0, Math.min(state.transformIndex, transforms.length - 1));
      transformSelect.value = String(state.transformIndex);
    }
    function renderPartControls() {
      const parts = partsForAssembly();
      partSelect.replaceChildren();
      parts.forEach((part, index) => {
        partSelect.appendChild(option(`${part.part_id} (${part.symmetries.length} accepted)`, String(index)));
      });
      state.partIndex = Math.max(0, Math.min(state.partIndex, parts.length - 1));
      partSelect.value = String(state.partIndex);
    }
    function renderAll() {
      renderPartControls();
      renderTransformControls();
      renderScene();
      renderDetails();
      document.getElementById("nearBtn").textContent = `Near Misses: ${state.showNear ? "On" : "Off"}`;
    }
    assemblies.forEach((assembly) => assemblySelect.appendChild(option(assembly, assembly)));
    state.assembly = assemblies[0] || "";
    assemblySelect.value = state.assembly;
    assemblySelect.addEventListener("change", () => {
      state.assembly = assemblySelect.value;
      state.partIndex = 0;
      state.transformIndex = 0;
      renderAll();
    });
    partSelect.addEventListener("change", () => {
      state.partIndex = Number(partSelect.value);
      state.transformIndex = 0;
      renderAll();
    });
    transformSelect.addEventListener("change", () => {
      state.transformIndex = Number(transformSelect.value);
      renderAll();
    });
    function stepPart(delta) {
      const parts = partsForAssembly();
      state.partIndex = (state.partIndex + delta + parts.length) % parts.length;
      state.transformIndex = 0;
      renderAll();
    }
    function stepTransform(delta) {
      const transforms = transformsForPart(currentPart());
      state.transformIndex = (state.transformIndex + delta + transforms.length) % transforms.length;
      renderAll();
    }
    document.getElementById("prevPartBtn").addEventListener("click", () => stepPart(-1));
    document.getElementById("nextPartBtn").addEventListener("click", () => stepPart(1));
    document.getElementById("prevTransformBtn").addEventListener("click", () => stepTransform(-1));
    document.getElementById("nextTransformBtn").addEventListener("click", () => stepTransform(1));
    document.getElementById("nearBtn").addEventListener("click", () => {
      state.showNear = !state.showNear;
      state.transformIndex = 0;
      renderAll();
    });
    document.getElementById("resetBtn").addEventListener("click", () => {
      Object.assign(state, initialView);
      renderAll();
    });
    scene.addEventListener("pointerdown", (event) => {
      state.dragging = true;
      state.pointerId = event.pointerId;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      state.dragMode = event.button === 1 || event.shiftKey ? "pan" : "rotate";
      scene.setPointerCapture(event.pointerId);
    });
    scene.addEventListener("pointermove", (event) => {
      if (!state.dragging || event.pointerId !== state.pointerId) return;
      const dx = event.clientX - state.lastX;
      const dy = event.clientY - state.lastY;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      if (state.dragMode === "pan") {
        state.panX += dx;
        state.panY += dy;
      } else {
        state.yaw += dx * 0.008;
        state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch + dy * 0.008));
      }
      renderScene();
    });
    scene.addEventListener("pointerup", (event) => {
      if (event.pointerId === state.pointerId) state.dragging = false;
    });
    scene.addEventListener("wheel", (event) => {
      event.preventDefault();
      state.zoom = Math.max(0.15, Math.min(8, state.zoom * Math.exp(-event.deltaY * 0.001)));
      renderScene();
    }, { passive: false });
    window.addEventListener("keydown", (event) => {
      if (event.key === "ArrowUp") stepPart(-1);
      else if (event.key === "ArrowDown") stepPart(1);
      else if (event.key === "ArrowLeft") stepTransform(-1);
      else if (event.key === "ArrowRight") stepTransform(1);
    });
    renderAll();
  </script>
</body>
</html>
"""
    ).replace("__DATA_JSON__", data_json)


def write_symmetry_report_html(path: Path, report_payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(report_payload, separators=(",", ":"))
    path.write_text(_html_document(data_json), encoding="utf-8")


def _parse_orders(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("orders must be a comma-separated list of integers >= 2")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute rotational symmetries for Fabrica OBJ parts.")
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--assembly", action="append", default=[], help="Assembly name to process. Repeatable.")
    parser.add_argument("--part", action="append", default=[], help="Part id/stem to process. Repeatable.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-html", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--mesh-scale", type=float, default=DEFAULT_MESH_SCALE)
    parser.add_argument("--tolerance-m", type=float, default=0.001)
    parser.add_argument("--sample-count", type=int, default=10_000)
    parser.add_argument("--max-validation-vertices", type=int, default=5_000)
    parser.add_argument("--visual-sample-count", type=int, default=1_400)
    parser.add_argument("--orders", type=_parse_orders, default=DEFAULT_ORDERS)
    parser.add_argument("--max-candidate-axes", type=int, default=48)
    parser.add_argument("--max-face-axes", type=int, default=12)
    parser.add_argument("--axis-grid-samples", type=int, default=0)
    parser.add_argument("--near-miss-count", type=int, default=12)
    parser.add_argument(
        "--no-write-assets", action="store_true", help="Only write report artifacts, not symmetries.json."
    )
    parser.add_argument(
        "--write-partial-assets",
        action="store_true",
        help="Allow --part runs to overwrite an assembly symmetries.json with only the selected parts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    asset_root = args.asset_root.resolve()
    output_root = args.output_root.resolve()
    report_html = (args.report_html or output_root / "index.html").resolve()
    report_json = (args.report_json or output_root / "report.json").resolve()
    config = DetectionConfig(
        mesh_scale=float(args.mesh_scale),
        tolerance_m=float(args.tolerance_m),
        sample_count=int(args.sample_count),
        max_validation_vertices=int(args.max_validation_vertices),
        visual_sample_count=int(args.visual_sample_count),
        orders=tuple(args.orders),
        max_candidate_axes=int(args.max_candidate_axes),
        max_face_axes=int(args.max_face_axes),
        axis_grid_samples=int(args.axis_grid_samples),
        near_miss_count=int(args.near_miss_count),
    )

    all_parts: list[dict[str, Any]] = []
    assembly_payloads: dict[str, list[dict[str, Any]]] = {}
    write_assets = not args.no_write_assets and (not args.part or args.write_partial_assets)
    if args.part and not args.no_write_assets and not args.write_partial_assets:
        print(
            "[SYMMETRY] --part was provided; skipping symmetries.json writes. "
            "Use --write-partial-assets to overwrite with a partial asset file.",
            flush=True,
        )
    for assembly_dir in _selected_assemblies(asset_root, args.assembly):
        assembly = assembly_dir.name
        parts: list[dict[str, Any]] = []
        for part_path in _selected_parts(assembly_dir, args.part):
            part_id = part_path.stem
            print(f"[SYMMETRY] {assembly}/{part_id}: detecting...", flush=True)
            result = detect_part_symmetries(part_path, assembly=assembly, part_id=part_id, config=config)
            parts.append(result)
            all_parts.append(result)
            print(
                "[SYMMETRY] "
                f"{assembly}/{part_id}: {len(result['symmetries'])} accepted, "
                f"{result['candidate_summary']['candidates_tested']} candidates",
                flush=True,
            )
        assembly_payloads[assembly] = parts
        if write_assets:
            _write_json(assembly_dir / "symmetries.json", _asset_payload(assembly, parts, config))

    report_payload = {
        "schema_version": 1,
        "generated_by": "scripts/precompute_fabrica_symmetries.py",
        "asset_root": str(asset_root),
        "config": {
            "mesh_scale": config.mesh_scale,
            "tolerance_m": config.tolerance_m,
            "sample_count": config.sample_count,
            "max_validation_vertices": config.max_validation_vertices,
            "visual_sample_count": config.visual_sample_count,
            "orders": list(config.orders),
            "max_candidate_axes": config.max_candidate_axes,
            "max_face_axes": config.max_face_axes,
            "axis_grid_samples": config.axis_grid_samples,
            "near_miss_count": config.near_miss_count,
        },
        "assemblies": sorted(assembly_payloads),
        "parts": all_parts,
    }
    _write_json(report_json, report_payload)
    write_symmetry_report_html(report_html, report_payload)
    print(f"[SYMMETRY] Wrote report JSON to {report_json}", flush=True)
    print(f"[SYMMETRY] Wrote report HTML to {report_html}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
