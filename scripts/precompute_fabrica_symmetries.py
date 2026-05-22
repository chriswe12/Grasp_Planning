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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
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
VALIDATION_MODES = ("mesh", "geometric")


@dataclass(frozen=True)
class DetectionConfig:
    mesh_scale: float = DEFAULT_MESH_SCALE
    tolerance_m: float = 0.001
    validation_mode: str = "mesh"
    max_mesh_probe_vertices: int = 2_000
    skip_geometric_vertex_prefilter: bool = False
    sample_count: int = 40_000
    max_validation_vertices: int = 5_000
    visual_sample_count: int = 1_400
    max_visual_faces: int = 4_000
    max_visual_edges: int = 6_000
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
class MeshConnectivityContext:
    vertices: np.ndarray
    faces: np.ndarray
    face_set: frozenset[tuple[int, int, int]]
    vertex_tree: cKDTree
    probe_indices: np.ndarray
    original_vertex_count: int
    original_face_count: int
    weld_tolerance_m: float


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


def _face_set(faces: np.ndarray) -> frozenset[tuple[int, int, int]]:
    return frozenset(tuple(sorted(int(value) for value in face)) for face in np.asarray(faces, dtype=np.int64).tolist())


def _build_mesh_connectivity_context(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    tolerance_m: float,
    max_probe_vertices: int,
) -> MeshConnectivityContext:
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    weld_tolerance_m = max(float(tolerance_m) * 1.0e-3, 1.0e-9)
    vertex_keys = np.round(vertices / weld_tolerance_m).astype(np.int64)
    _, inverse = np.unique(vertex_keys, axis=0, return_inverse=True)

    welded_vertices = np.zeros((int(np.max(inverse)) + 1, 3), dtype=float)
    counts = np.zeros(len(welded_vertices), dtype=float)
    np.add.at(welded_vertices, inverse, vertices)
    np.add.at(counts, inverse, 1.0)
    welded_vertices /= counts[:, None]

    welded_faces = inverse[faces]
    keep = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 2] != welded_faces[:, 0])
    )
    welded_faces = _compact_faces(welded_faces[keep])
    probe_stride = max(1, len(welded_vertices) // max(1, int(max_probe_vertices)))
    return MeshConnectivityContext(
        vertices=welded_vertices,
        faces=welded_faces,
        face_set=_face_set(welded_faces),
        vertex_tree=cKDTree(welded_vertices),
        probe_indices=np.arange(0, len(welded_vertices), probe_stride, dtype=np.int64),
        original_vertex_count=int(len(vertices)),
        original_face_count=int(len(faces)),
        weld_tolerance_m=float(weld_tolerance_m),
    )


def _bijective_vertex_match(
    transformed_vertices: np.ndarray,
    mesh_context: MeshConnectivityContext,
    *,
    tolerance_m: float,
) -> tuple[np.ndarray | None, int, int, dict[str, float]]:
    distances, _ = mesh_context.vertex_tree.query(transformed_vertices)
    metrics = _distance_metrics(distances)
    candidate_columns = mesh_context.vertex_tree.query_ball_point(transformed_vertices, r=float(tolerance_m))
    edge_count = int(sum(len(columns) for columns in candidate_columns))
    vertex_count = int(len(mesh_context.vertices))
    if edge_count < vertex_count:
        return None, 0, edge_count, metrics

    indptr = np.zeros(vertex_count + 1, dtype=np.int64)
    indices = np.empty(edge_count, dtype=np.int64)
    weights = np.empty(edge_count, dtype=float)
    cursor = 0
    for row, columns in enumerate(candidate_columns):
        cursor_next = cursor + len(columns)
        indptr[row] = cursor
        indices[cursor:cursor_next] = columns
        if columns:
            deltas = mesh_context.vertices[np.asarray(columns, dtype=np.int64)] - transformed_vertices[row]
            weights[cursor:cursor_next] = np.linalg.norm(deltas, axis=1) + 1.0e-15
        cursor = cursor_next
    indptr[vertex_count] = cursor
    graph = csr_matrix((weights, indices, indptr), shape=(vertex_count, vertex_count))
    try:
        row_indices, column_indices = min_weight_full_bipartite_matching(graph)
    except ValueError:
        return None, 0, edge_count, metrics
    unique_matches = int(len(column_indices))
    if unique_matches != vertex_count or len(row_indices) != vertex_count:
        return None, unique_matches, edge_count, metrics

    matching = np.empty(vertex_count, dtype=np.int64)
    matching[np.asarray(row_indices, dtype=np.int64)] = np.asarray(column_indices, dtype=np.int64)
    matched_distances = np.linalg.norm(transformed_vertices - mesh_context.vertices[matching], axis=1)
    return matching.astype(np.int64), unique_matches, edge_count, _distance_metrics(matched_distances)


def _evaluate_mesh_transform(
    candidate: CandidateTransform,
    *,
    mesh_context: MeshConnectivityContext,
    tolerance_m: float,
) -> dict[str, Any]:
    probe_vertices = mesh_context.vertices[mesh_context.probe_indices]
    transformed_probe_vertices = _apply_transform(probe_vertices, candidate.matrix)
    probe_distances, _ = mesh_context.vertex_tree.query(transformed_probe_vertices)
    probe_metrics = _distance_metrics(probe_distances)
    if probe_metrics["max_m"] > tolerance_m:
        return {
            "accepted": False,
            "validation_mode": "mesh_connectivity_probe",
            "effective_tolerance_m": float(tolerance_m),
            "max_allowed_distance_m": float(tolerance_m),
            "mesh_vertices": int(len(mesh_context.vertices)),
            "mesh_faces": int(len(mesh_context.face_set)),
            "mesh_probe_vertices": int(len(probe_vertices)),
            "mesh_weld_tolerance_m": float(mesh_context.weld_tolerance_m),
            "mesh_unique_vertex_matches": 0,
            "mesh_reject_reason": "probe_vertex_distance",
            "mesh_missing_faces": len(mesh_context.face_set),
            "mesh_extra_faces": 0,
            "vertex_mean_m": probe_metrics["mean_m"],
            "vertex_p95_m": probe_metrics["p95_m"],
            "vertex_p99_m": probe_metrics["p99_m"],
            "vertex_max_m": probe_metrics["max_m"],
            **probe_metrics,
        }

    transformed_vertices = _apply_transform(mesh_context.vertices, candidate.matrix)
    vertex_map, unique_matches, candidate_edges, metrics = _bijective_vertex_match(
        transformed_vertices,
        mesh_context,
        tolerance_m=tolerance_m,
    )
    vertex_count = int(len(mesh_context.vertices))
    base_payload: dict[str, Any] = {
        "validation_mode": "mesh_connectivity",
        "effective_tolerance_m": float(tolerance_m),
        "max_allowed_distance_m": float(tolerance_m),
        "mesh_vertices": vertex_count,
        "mesh_faces": int(len(mesh_context.face_set)),
        "mesh_probe_vertices": int(len(probe_vertices)),
        "mesh_weld_tolerance_m": float(mesh_context.weld_tolerance_m),
        "mesh_candidate_vertex_edges": int(candidate_edges),
        "mesh_unique_vertex_matches": unique_matches,
        "vertex_mean_m": metrics["mean_m"],
        "vertex_p95_m": metrics["p95_m"],
        "vertex_p99_m": metrics["p99_m"],
        "vertex_max_m": metrics["max_m"],
        **metrics,
    }
    if vertex_map is None and metrics["max_m"] > tolerance_m:
        return {
            **base_payload,
            "accepted": False,
            "mesh_reject_reason": "vertex_distance",
            "mesh_missing_faces": len(mesh_context.face_set),
            "mesh_extra_faces": 0,
        }
    if vertex_map is None:
        return {
            **base_payload,
            "accepted": False,
            "mesh_reject_reason": "no_bijective_vertex_map",
            "mesh_missing_faces": len(mesh_context.face_set),
            "mesh_extra_faces": 0,
        }

    remapped_faces = vertex_map[mesh_context.faces]
    remapped_face_set = _face_set(remapped_faces)
    missing_faces = int(len(mesh_context.face_set - remapped_face_set))
    extra_faces = int(len(remapped_face_set - mesh_context.face_set))
    return {
        **base_payload,
        "accepted": missing_faces == 0 and extra_faces == 0,
        "mesh_reject_reason": "none" if missing_faces == 0 and extra_faces == 0 else "face_connectivity",
        "mesh_missing_faces": missing_faces,
        "mesh_extra_faces": extra_faces,
    }


def _evaluate_geometric_transform(
    candidate: CandidateTransform,
    *,
    vertices: np.ndarray,
    vertex_tree: cKDTree,
    validation_points: np.ndarray,
    validation_tree: cKDTree,
    base_tolerance_m: float,
    effective_tolerance_m: float,
    max_distance_m: float,
    skip_vertex_prefilter: bool,
) -> dict[str, Any]:
    transformed_vertices = _apply_transform(vertices, candidate.matrix)
    vertex_distances, _ = vertex_tree.query(transformed_vertices)
    vertex_metrics = _distance_metrics(vertex_distances)
    if not skip_vertex_prefilter and vertex_metrics["p99_m"] > base_tolerance_m:
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
        (skip_vertex_prefilter or vertex_metrics["p99_m"] <= base_tolerance_m)
        and metrics["p99_m"] <= effective_tolerance_m
        and metrics["max_m"] <= max_distance_m
    )
    return {
        "accepted": bool(accepted),
        "validation_mode": "geometric_surface_no_vertex_prefilter" if skip_vertex_prefilter else "geometric_surface",
        "effective_tolerance_m": float(effective_tolerance_m),
        "max_allowed_distance_m": float(max_distance_m),
        "vertex_prefilter_skipped": bool(skip_vertex_prefilter),
        "vertex_mean_m": vertex_metrics["mean_m"],
        "vertex_p95_m": vertex_metrics["p95_m"],
        "vertex_p99_m": vertex_metrics["p99_m"],
        "vertex_max_m": vertex_metrics["max_m"],
        **metrics,
    }


def _evaluate_transform(
    candidate: CandidateTransform,
    *,
    config: DetectionConfig,
    mesh_context: MeshConnectivityContext,
    geometric_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if config.validation_mode == "mesh":
        return _evaluate_mesh_transform(candidate, mesh_context=mesh_context, tolerance_m=config.tolerance_m)
    if config.validation_mode == "geometric":
        if geometric_context is None:
            raise ValueError("geometric validation context is required for validation_mode='geometric'")
        return _evaluate_geometric_transform(
            candidate,
            vertices=geometric_context["vertices"],
            vertex_tree=geometric_context["vertex_tree"],
            validation_points=geometric_context["validation_points"],
            validation_tree=geometric_context["validation_tree"],
            base_tolerance_m=config.tolerance_m,
            effective_tolerance_m=geometric_context["effective_tolerance_m"],
            max_distance_m=geometric_context["max_distance_m"],
            skip_vertex_prefilter=config.skip_geometric_vertex_prefilter,
        )
    raise ValueError(f"Unsupported validation_mode '{config.validation_mode}'.")


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


def _angle_description(angle_deg: float) -> str:
    rounded = round(float(angle_deg))
    if abs(float(angle_deg) - rounded) <= 1.0e-7:
        angle_text = str(int(rounded))
    else:
        angle_text = f"{float(angle_deg):.3f}".rstrip("0").rstrip(".")
    return f"{angle_text}°"


def _axis_description(axis: np.ndarray | None) -> str:
    if axis is None:
        return "no axis"
    normalized = _normalize_axis(axis)
    if normalized is None:
        return "unknown axis"
    cardinal_axes = (
        ("+X", np.array([1.0, 0.0, 0.0], dtype=float)),
        ("+Y", np.array([0.0, 1.0, 0.0], dtype=float)),
        ("+Z", np.array([0.0, 0.0, 1.0], dtype=float)),
        ("-X", np.array([-1.0, 0.0, 0.0], dtype=float)),
        ("-Y", np.array([0.0, -1.0, 0.0], dtype=float)),
        ("-Z", np.array([0.0, 0.0, -1.0], dtype=float)),
    )
    for label, basis in cardinal_axes:
        if float(np.dot(normalized, basis)) >= 1.0 - 1.0e-6:
            return label

    components = [
        (name, float(value))
        for name, value in zip(("X", "Y", "Z"), normalized, strict=True)
        if abs(float(value)) >= 1.0e-6
    ]
    if len(components) in (2, 3):
        magnitudes = [abs(value) for _, value in components]
        if max(magnitudes) - min(magnitudes) <= 1.0e-6:
            signs = [f"{'+' if value >= 0.0 else '-'}{name}" for name, value in components]
            return f"{' '.join(signs)} diagonal"

    return str(_vector_payload(normalized, digits=4))


def _center_description(candidate: CandidateTransform) -> str:
    if candidate.center is None:
        return ""
    center_name = candidate.source.split(":")[-1].replace("_", " ")
    return f" ({center_name})"


def _transform_description(candidate: CandidateTransform) -> str:
    if candidate.order is None:
        return "Identity"
    return f"{_angle_description(candidate.angle_deg)} about {_axis_description(candidate.axis)}{_center_description(candidate)}"


def _candidate_payload(candidate: CandidateTransform, validation: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": candidate.name,
        "type": "finite_rotation" if candidate.order else "identity",
        "description": _transform_description(candidate),
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


def _unique_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i0, i1, i2 in np.asarray(faces, dtype=np.int64).tolist():
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            edges.add(tuple(sorted((int(a), int(b)))))
    return sorted(edges)


def _sample_sequence(values: Sequence[Any], max_count: int) -> list[Any]:
    if max_count <= 0 or len(values) <= max_count:
        return list(values)
    indices = np.linspace(0, len(values) - 1, int(max_count), dtype=np.int64)
    return [values[int(index)] for index in indices]


def _compact_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return faces.reshape((0, 3))
    face_keys = np.sort(faces, axis=1)
    _, unique_indices = np.unique(face_keys, axis=0, return_index=True)
    return faces[np.sort(unique_indices)]


def _cluster_mesh_for_visuals(
    vertices: np.ndarray, faces: np.ndarray, *, max_faces: int
) -> tuple[np.ndarray, np.ndarray]:
    if max_faces <= 0 or len(faces) <= max_faces:
        return vertices, faces

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    bounds_min = np.min(vertices, axis=0)
    bounds_max = np.max(vertices, axis=0)
    extent = np.maximum(bounds_max - bounds_min, 1.0e-12)

    def clustered(divisions: int) -> tuple[np.ndarray, np.ndarray] | None:
        normalized = np.clip((vertices - bounds_min) / extent, 0.0, 1.0)
        cells = np.minimum(np.floor(normalized * float(divisions)).astype(np.int64), divisions - 1)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)

        clustered_vertices = np.zeros((int(np.max(inverse)) + 1, 3), dtype=float)
        counts = np.zeros(len(clustered_vertices), dtype=float)
        np.add.at(clustered_vertices, inverse, vertices)
        np.add.at(counts, inverse, 1.0)
        clustered_vertices /= counts[:, None]

        clustered_faces = inverse[faces]
        keep = (
            (clustered_faces[:, 0] != clustered_faces[:, 1])
            & (clustered_faces[:, 1] != clustered_faces[:, 2])
            & (clustered_faces[:, 2] != clustered_faces[:, 0])
        )
        clustered_faces = _compact_faces(clustered_faces[keep])
        if len(clustered_faces) == 0:
            return None
        return clustered_vertices, clustered_faces

    best: tuple[np.ndarray, np.ndarray] | None = None
    low, high = 2, 128
    while low <= high:
        middle = (low + high) // 2
        candidate = clustered(middle)
        face_count = len(candidate[1]) if candidate is not None else 0
        if candidate is not None and face_count <= max_faces:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if best is not None:
        return best

    sampled_faces = np.asarray(_sample_sequence(faces.tolist(), max_faces), dtype=np.int64)
    return vertices, sampled_faces


def _visual_mesh_payload(mesh: "trimesh.Trimesh", config: DetectionConfig) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    original_face_count = int(len(faces))
    original_edge_count = len(_unique_edges(faces))
    display_vertices, sampled_faces = _cluster_mesh_for_visuals(vertices, faces, max_faces=config.max_visual_faces)
    if sampled_faces.size == 0:
        return {
            "visual_mesh_vertices_obj": [],
            "visual_mesh_faces": [],
            "visual_mesh_edges": [],
            "visual_mesh_original_faces": original_face_count,
            "visual_mesh_original_edges": original_edge_count,
        }

    used_indices = sorted({int(index) for face in sampled_faces.tolist() for index in face})
    index_map = {old_index: new_index for new_index, old_index in enumerate(used_indices)}
    compact_faces = np.asarray(
        [[index_map[int(index)] for index in face] for face in sampled_faces.tolist()],
        dtype=np.int64,
    )
    edges = _sample_sequence(_unique_edges(compact_faces), config.max_visual_edges)
    return {
        "visual_mesh_vertices_obj": [_vector_payload(display_vertices[index], digits=6) for index in used_indices],
        "visual_mesh_faces": [[int(value) for value in face] for face in compact_faces.tolist()],
        "visual_mesh_edges": [[int(a), int(b)] for a, b in edges],
        "visual_mesh_original_faces": original_face_count,
        "visual_mesh_original_edges": original_edge_count,
    }


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
    if config.validation_mode not in VALIDATION_MODES:
        raise ValueError(f"validation_mode must be one of {', '.join(VALIDATION_MODES)}.")
    mesh_vertices = np.asarray(mesh.vertices, dtype=float)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
    mesh_context = _build_mesh_connectivity_context(
        mesh_vertices,
        mesh_faces,
        tolerance_m=config.tolerance_m,
        max_probe_vertices=config.max_mesh_probe_vertices,
    )
    geometric_context: dict[str, Any] | None = None
    if config.validation_mode == "geometric":
        surface_points = _sample_surface(mesh, config.sample_count, seed=config.validation_seed)
        vertices = _validation_vertices(
            mesh_vertices,
            max_count=config.max_validation_vertices,
            seed=config.validation_seed + 1009,
        )
        validation_points = np.vstack([vertices, surface_points])
        spacing_m = _cloud_spacing(surface_points)
        effective_tolerance_m = float(config.tolerance_m + config.coverage_multiplier * spacing_m)
        max_distance_m = max(
            float(config.tolerance_m * config.max_distance_multiplier),
            float(effective_tolerance_m * 2.5),
        )
        geometric_context = {
            "vertices": vertices,
            "vertex_tree": cKDTree(vertices),
            "validation_points": validation_points,
            "validation_tree": cKDTree(validation_points),
            "surface_sample_count": int(len(surface_points)),
            "validation_vertex_count": int(len(vertices)),
            "spacing_m": float(spacing_m),
            "effective_tolerance_m": float(effective_tolerance_m),
            "max_distance_m": float(max_distance_m),
        }
        probe_points = validation_points[:: max(1, len(validation_points) // 2_000)]
    else:
        surface_points = np.empty((0, 3), dtype=float)
        spacing_m = 0.0
        effective_tolerance_m = float(config.tolerance_m)
        max_distance_m = float(config.tolerance_m)
        probe_points = mesh_context.vertices[:: max(1, len(mesh_context.vertices) // 2_000)]

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
        config=config,
        mesh_context=mesh_context,
        geometric_context=geometric_context,
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
            config=config,
            mesh_context=mesh_context,
            geometric_context=geometric_context,
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
    bounds = np.asarray(mesh.bounds, dtype=float)
    visual_mesh = _visual_mesh_payload(mesh, config)

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
            "validation_mode": config.validation_mode,
            "skip_geometric_vertex_prefilter": bool(config.skip_geometric_vertex_prefilter),
            "orders": list(config.orders),
            "axis_sources": sorted({axis.source for axis in axes}),
            "center_names": [center.name for center in centers],
            "mesh_vertices": int(len(mesh_vertices)),
            "mesh_faces": int(len(mesh_faces)),
            "mesh_welded_vertices": int(len(mesh_context.vertices)),
            "mesh_welded_faces": int(len(mesh_context.face_set)),
            "mesh_probe_vertices": int(len(mesh_context.probe_indices)),
            "validation_vertices": int(
                geometric_context["validation_vertex_count"]
                if geometric_context is not None
                else len(mesh_context.vertices)
            ),
            "surface_samples": int(len(surface_points)),
            "visual_faces": len(visual_mesh["visual_mesh_faces"]),
            "visual_edges": len(visual_mesh["visual_mesh_edges"]),
        },
        "continuous_symmetries": _continuous_symmetry_payload(mesh),
        "symmetries": accepted,
        "near_misses": near_misses,
        **visual_mesh,
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
      --motion: #2563eb;
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
    input[type="range"] { width: 100%; accent-color: var(--accepted); }
    .range-row { display: grid; grid-template-columns: minmax(0, 1fr) 48px; gap: 8px; align-items: center; }
    .range-value { color: var(--ink); font-family: "IBM Plex Mono", monospace; font-size: 11px; text-align: right; }
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
      <p class="subtitle">Accepted transforms overlay as meshes in green. Closest rejected candidates overlay in red.</p>
      <label>Assembly<select id="assemblySelect"></select></label>
      <label>Part<select id="partSelect"></select></label>
      <label>Transform<select id="transformSelect"></select></label>
      <label>Progress
        <span class="range-row">
          <input id="progressSlider" type="range" min="0" max="100" step="1" value="100">
          <span id="progressValue" class="range-value">100%</span>
        </span>
      </label>
      <div class="controls">
        <button id="prevPartBtn" type="button">Prev Part</button>
        <button id="nextPartBtn" type="button">Next Part</button>
        <button id="prevTransformBtn" type="button">Prev Sym</button>
        <button id="nextTransformBtn" type="button">Next Sym</button>
        <button id="meshModeBtn" type="button">Wireframe Mesh</button>
        <button id="nearBtn" type="button">Near Misses: On</button>
        <button id="resetBtn" type="button">Reset View</button>
      </div>
      <div id="transformList" class="list"></div>
    </aside>
    <main>
      <div class="grid">
        <section class="panel">
          <canvas id="scene" width="1100" height="760"></canvas>
          <div class="legend">
            <span><i class="swatch" style="background: var(--mesh)"></i>Original</span>
            <span><i class="swatch" style="background: var(--accepted)"></i>Accepted transform</span>
            <span><i class="swatch" style="background: var(--near)"></i>Rejected near miss</span>
            <span><i class="swatch" style="background: var(--motion)"></i>Point motion</span>
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
    const progressSlider = document.getElementById("progressSlider");
    const progressValue = document.getElementById("progressValue");
    const transformList = document.getElementById("transformList");
    const details = document.getElementById("details");
    const scene = document.getElementById("scene");
    const initialView = { yaw: -0.72, pitch: 0.52, zoom: 1.0, panX: 0, panY: 0 };
    const state = {
      assembly: "",
      partIndex: 0,
      transformIndex: 0,
      transformProgress: 1.0,
      solidMesh: true,
      showNear: true,
      dragging: false,
      dragMode: "rotate",
      pointerId: null,
      lastX: 0,
      lastY: 0,
      ...initialView,
    };
    const ctx = scene.getContext("2d");
    const sceneWidth = scene.width;
    const sceneHeight = scene.height;
    let renderSceneQueued = false;
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
    function normalizeVector(vector) {
      const norm = Math.hypot(vector[0], vector[1], vector[2]);
      if (!Number.isFinite(norm) || norm <= 1.0e-12) return null;
      return vector.map((value) => value / norm);
    }
    function rotationMatrix(axisRaw, angleRad) {
      const axis = normalizeVector(axisRaw);
      if (!axis) return null;
      const [x, y, z] = axis;
      const c = Math.cos(angleRad);
      const s = Math.sin(angleRad);
      const oneMinusC = 1.0 - c;
      return [
        [c + x * x * oneMinusC, x * y * oneMinusC - z * s, x * z * oneMinusC + y * s],
        [y * x * oneMinusC + z * s, c + y * y * oneMinusC, y * z * oneMinusC - x * s],
        [z * x * oneMinusC - y * s, z * y * oneMinusC + x * s, c + z * z * oneMinusC],
      ];
    }
    function matrixAboutCenter(axis, center, angleDeg) {
      const rotation = rotationMatrix(axis, angleDeg * Math.PI / 180.0);
      if (!rotation) return null;
      const rotatedCenter = [
        rotation[0][0] * center[0] + rotation[0][1] * center[1] + rotation[0][2] * center[2],
        rotation[1][0] * center[0] + rotation[1][1] * center[1] + rotation[1][2] * center[2],
        rotation[2][0] * center[0] + rotation[2][1] * center[1] + rotation[2][2] * center[2],
      ];
      return [
        [rotation[0][0], rotation[0][1], rotation[0][2], center[0] - rotatedCenter[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], center[1] - rotatedCenter[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], center[2] - rotatedCenter[2]],
        [0.0, 0.0, 0.0, 1.0],
      ];
    }
    function interpolatedMatrix(matrix, progress) {
      const identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
      ];
      return identity.map((row, rowIndex) =>
        row.map((value, columnIndex) => value + progress * (matrix[rowIndex][columnIndex] - value))
      );
    }
    function matrixAtProgress(transform, progress) {
      const clamped = Math.max(0.0, Math.min(1.0, Number(progress)));
      if (transform?.axis_obj && transform?.center_obj_m && Number.isFinite(Number(transform.angle_deg))) {
        const matrix = matrixAboutCenter(transform.axis_obj, transform.center_obj_m, Number(transform.angle_deg) * clamped);
        if (matrix) return matrix;
      }
      return interpolatedMatrix(transform.matrix_obj, clamped);
    }
    function meshVertices(part) {
      return part.visual_mesh_vertices_obj || [];
    }
    function transformedVertices(part, matrix) {
      return meshVertices(part).map((point) => applyMatrix(point, matrix));
    }
    function allPoints(part, transform) {
      const vertices = meshVertices(part);
      const points = [...vertices];
      if (transform) {
        points.push(...transformedVertices(part, transform.matrix_obj));
        const progressMatrix = matrixAtProgress(transform, state.transformProgress);
        points.push(...transformedVertices(part, progressMatrix));
      }
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
      return { x: sceneWidth * 0.5 + state.panX + x * scale, y: sceneHeight * 0.5 + state.panY - y * scale, depth: z };
    }
    function drawPoint(point, center, scale, fill, radius, opacity) {
      const projected = project(point, center, scale);
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.arc(projected.x, projected.y, radius, 0, Math.PI * 2.0);
      ctx.fill();
      ctx.restore();
    }
    function drawLine(a, b, center, scale, stroke, width) {
      const pa = project(a, center, scale);
      const pb = project(b, center, scale);
      drawProjectedLine(pa, pb, stroke, width, 0.95);
    }
    function drawProjectedLine(a, b, stroke, width, opacity) {
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.restore();
    }
    function drawPolygon(points, center, scale, options) {
      const projected = points.map((point) => project(point, center, scale));
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(projected[0].x, projected[0].y);
      projected.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
      ctx.closePath();
      if (options.fill) {
        ctx.globalAlpha = options.fillOpacity ?? 1;
        ctx.fillStyle = options.fill;
        ctx.fill();
      }
      if (options.stroke) {
        ctx.globalAlpha = options.strokeOpacity ?? 1;
        ctx.strokeStyle = options.stroke;
        ctx.lineWidth = options.strokeWidth || 1;
        ctx.stroke();
      }
      ctx.restore();
    }
    function shadeColor(hex, factor) {
      const clean = hex.replace("#", "");
      const value = Number.parseInt(clean, 16);
      const channels = [(value >> 16) & 255, (value >> 8) & 255, value & 255];
      return `#${channels.map((channel) =>
        Math.max(0, Math.min(255, Math.round(channel * factor))).toString(16).padStart(2, "0")
      ).join("")}`;
    }
    function drawMeshEdges(vertices, edges, center, scale, stroke, width, opacity) {
      edges.forEach(([start, end]) => {
        if (!vertices[start] || !vertices[end]) return;
        const pa = project(vertices[start], center, scale);
        const pb = project(vertices[end], center, scale);
        drawProjectedLine(pa, pb, stroke, width, opacity);
      });
    }
    function drawMesh(part, vertices, center, scale, options) {
      const faces = part.visual_mesh_faces || [];
      const edges = part.visual_mesh_edges || [];
      if (state.solidMesh) {
        faces.map((face) => {
          const points = face.map((index) => vertices[index]);
          if (points.length !== 3 || points.some((point) => !point)) return null;
          const rotated = points.map((point) => rotate(point, center));
          const edgeA = rotated[1].map((value, axis) => value - rotated[0][axis]);
          const edgeB = rotated[2].map((value, axis) => value - rotated[0][axis]);
          const normal = [
            edgeA[1] * edgeB[2] - edgeA[2] * edgeB[1],
            edgeA[2] * edgeB[0] - edgeA[0] * edgeB[2],
            edgeA[0] * edgeB[1] - edgeA[1] * edgeB[0],
          ];
          const depth = rotated.reduce((sum, point) => sum + point[2], 0) / rotated.length;
          return { points, normal, depth };
        }).filter((face) =>
          face && Number.isFinite(face.depth) && face.normal.every(Number.isFinite) && face.normal[2] > 0
        ).sort((a, b) => a.depth - b.depth).forEach((face) => {
          const norm = Math.hypot(face.normal[0], face.normal[1], face.normal[2]) || 1;
          const light = 0.45 + 0.55 * (face.normal[2] / norm);
          drawPolygon(face.points, center, scale, {
            fill: shadeColor(options.fill, 0.7 + light * 0.45),
            fillOpacity: options.fillOpacity,
            stroke: options.stroke,
            strokeWidth: options.strokeWidth,
            strokeOpacity: options.strokeOpacity,
          });
        });
      }
      drawMeshEdges(vertices, edges, center, scale, options.stroke, options.edgeWidth, options.edgeOpacity);
    }
    function drawArrowLine(a, b, stroke, width, opacity) {
      drawProjectedLine(a, b, stroke, width, opacity);
      const angle = Math.atan2(b.y - a.y, b.x - a.x);
      const size = Math.max(7, width * 4.0);
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.fillStyle = stroke;
      ctx.beginPath();
      ctx.moveTo(b.x, b.y);
      ctx.lineTo(b.x - size * Math.cos(angle - 0.42), b.y - size * Math.sin(angle - 0.42));
      ctx.lineTo(b.x - size * Math.cos(angle + 0.42), b.y - size * Math.sin(angle + 0.42));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
    function drawMotionMarkers(part, transform, center, scale) {
      if (!transform || transform.type === "identity") return;
      const points = meshVertices(part);
      const finalMatrix = transform.matrix_obj;
      const stride = Math.max(1, Math.floor(points.length / 24));
      for (let index = 0; index < points.length; index += stride) {
        const start = points[index];
        const end = applyMatrix(start, finalMatrix);
        const distance = Math.hypot(end[0] - start[0], end[1] - start[1], end[2] - start[2]);
        if (distance <= Math.max(1.0e-5, (part?.extent_m || 0.04) * 0.01)) continue;
        const pa = project(start, center, scale);
        const pb = project(end, center, scale);
        drawArrowLine(pa, pb, "#2563eb", 1.4, 0.45);
      }
    }
    function renderScene() {
      ctx.clearRect(0, 0, sceneWidth, sceneHeight);
      try {
        const part = currentPart();
        const transform = currentTransform();
        if (!part || !transform) return;
        const points = allPoints(part, transform);
        if (!points.length) return;
        const bounds = boundsFor(points);
        const center = bounds.min.map((value, index) => 0.5 * (value + bounds.max[index]));
        const extent = Math.max(...bounds.max.map((value, index) => value - bounds.min[index]), 0.04);
        const scale = 560 / extent * state.zoom;
        drawMesh(part, meshVertices(part), center, scale, {
          fill: "#64748b",
          fillOpacity: 0.24,
          stroke: "#475569",
          strokeWidth: 0.7,
          strokeOpacity: 0.28,
          edgeWidth: 1.1,
          edgeOpacity: 0.42,
        });
        drawMotionMarkers(part, transform, center, scale);
        const progressMatrix = matrixAtProgress(transform, state.transformProgress);
        const transformed = transformedVertices(part, progressMatrix);
        const color = transform.group === "near" ? "#b91c1c" : "#0f766e";
        drawMesh(part, transformed, center, scale, {
          fill: color,
          fillOpacity: transform.group === "near" ? 0.58 : 0.68,
          stroke: color,
          strokeWidth: 0.8,
          strokeOpacity: 0.46,
          edgeWidth: 1.3,
          edgeOpacity: 0.78,
        });
        if (transform.center_obj_m && transform.axis_obj) {
          const axisExtent = Math.max(part.extent_m || 0.04, 0.04) * 0.85;
          const start = transform.center_obj_m.map((value, index) => value - transform.axis_obj[index] * axisExtent);
          const end = transform.center_obj_m.map((value, index) => value + transform.axis_obj[index] * axisExtent);
          drawLine(start, end, center, scale, "#d97706", 3);
          drawPoint(transform.center_obj_m, center, scale, "#d97706", 4.5, 1.0);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        ctx.save();
        ctx.fillStyle = "#b91c1c";
        ctx.font = "14px IBM Plex Mono, monospace";
        ctx.fillText(`Render error: ${message}`, 32, 48);
        ctx.restore();
      }
    }
    function scheduleRenderScene() {
      if (renderSceneQueued) return;
      renderSceneQueued = true;
      window.requestAnimationFrame(() => {
        renderSceneQueued = false;
        renderScene();
      });
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
        `validation mode: ${part.candidate_summary.validation_mode}`,
        `vertex prefilter skipped: ${part.candidate_summary.skip_geometric_vertex_prefilter}`,
        `candidates tested: ${part.candidate_summary.candidates_tested}`,
        `axes tested: ${part.candidate_summary.axes_tested}`,
        `vertices checked: ${part.candidate_summary.validation_vertices} / ${part.candidate_summary.mesh_vertices}`,
        `mesh graph: ${part.candidate_summary.mesh_welded_vertices} vertices / ${part.candidate_summary.mesh_welded_faces} faces`,
        `mesh probe: ${part.candidate_summary.mesh_probe_vertices} vertices`,
        `surface samples: ${part.candidate_summary.surface_samples}`,
        `visual faces: ${part.candidate_summary.visual_faces} / ${part.candidate_summary.mesh_faces}`,
        `visual edges: ${part.candidate_summary.visual_edges} / ${part.visual_mesh_original_edges ?? "n/a"}`,
        `centers: ${part.candidate_summary.center_names.join(", ")}`,
        "",
        `selected: ${transform.name}`,
        `transform: ${transform.description || "n/a"}`,
        `group: ${transform.group}`,
        `progress: ${fmt(state.transformProgress * 100.0, 1)}%`,
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
        const readable = transform.description || transform.name;
        const label = `${transform.group === "near" ? "near" : "ok"} ${readable}`;
        transformSelect.appendChild(option(label, String(index)));
        const item = document.createElement("button");
        item.type = "button";
        item.className = `item ${transform.group === "near" ? "near" : ""} ${index === state.transformIndex ? "active" : ""}`;
        item.innerHTML = `<div class="item-title"><span>${readable}</span><span>${transform.group}</span></div>
          <div class="item-meta">${transform.name} | p99 ${fmt(transform.validation?.p99_m, 5)} | max ${fmt(transform.validation?.max_m, 5)}</div>`;
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
      document.getElementById("meshModeBtn").textContent = state.solidMesh ? "Wireframe Mesh" : "Solid Mesh";
      progressSlider.value = String(Math.round(state.transformProgress * 100.0));
      progressValue.textContent = `${Math.round(state.transformProgress * 100.0)}%`;
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
    progressSlider.addEventListener("input", () => {
      state.transformProgress = Number(progressSlider.value) / 100.0;
      progressValue.textContent = `${Math.round(state.transformProgress * 100.0)}%`;
      scheduleRenderScene();
      renderDetails();
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
    document.getElementById("meshModeBtn").addEventListener("click", () => {
      state.solidMesh = !state.solidMesh;
      renderAll();
    });
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
      scheduleRenderScene();
    });
    scene.addEventListener("pointerup", (event) => {
      if (event.pointerId === state.pointerId) state.dragging = false;
    });
    scene.addEventListener("wheel", (event) => {
      event.preventDefault();
      state.zoom = Math.max(0.15, Math.min(8, state.zoom * Math.exp(-event.deltaY * 0.001)));
      scheduleRenderScene();
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
    parser.add_argument(
        "--validation-mode",
        choices=VALIDATION_MODES,
        default=DetectionConfig.validation_mode,
        help="Use strict welded mesh connectivity validation or the older sampled geometric validation.",
    )
    parser.add_argument(
        "--max-mesh-probe-vertices",
        type=int,
        default=DetectionConfig.max_mesh_probe_vertices,
        help="Vertex-count budget for strict mesh validation's cheap probe before full graph matching.",
    )
    parser.add_argument(
        "--skip-geometric-vertex-prefilter",
        action="store_true",
        help="In geometric mode, skip the OBJ-vertex p99 prefilter and decide from sampled surface validation only.",
    )
    parser.add_argument("--sample-count", type=int, default=DetectionConfig.sample_count)
    parser.add_argument("--max-validation-vertices", type=int, default=5_000)
    parser.add_argument("--visual-sample-count", type=int, default=1_400)
    parser.add_argument(
        "--max-visual-faces",
        type=int,
        default=DetectionConfig.max_visual_faces,
        help="Maximum mesh faces embedded in the HTML report per part. Use 0 for no cap.",
    )
    parser.add_argument(
        "--max-visual-edges",
        type=int,
        default=DetectionConfig.max_visual_edges,
        help="Maximum mesh edges embedded in the HTML report per part. Use 0 for no cap.",
    )
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
        validation_mode=str(args.validation_mode),
        max_mesh_probe_vertices=int(args.max_mesh_probe_vertices),
        skip_geometric_vertex_prefilter=bool(args.skip_geometric_vertex_prefilter),
        sample_count=int(args.sample_count),
        max_validation_vertices=int(args.max_validation_vertices),
        visual_sample_count=int(args.visual_sample_count),
        max_visual_faces=int(args.max_visual_faces),
        max_visual_edges=int(args.max_visual_edges),
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
            "validation_mode": config.validation_mode,
            "max_mesh_probe_vertices": config.max_mesh_probe_vertices,
            "skip_geometric_vertex_prefilter": config.skip_geometric_vertex_prefilter,
            "sample_count": config.sample_count,
            "max_validation_vertices": config.max_validation_vertices,
            "visual_sample_count": config.visual_sample_count,
            "max_visual_faces": config.max_visual_faces,
            "max_visual_edges": config.max_visual_edges,
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
