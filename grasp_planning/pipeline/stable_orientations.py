"""Stable support orientation enumeration for grasp-generation benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from grasp_planning.grasping.fabrica_grasp_debug import (
    mesh_area_weighted_triangle_centroid,
    rotmat_to_quat_xyzw,
    trimesh,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose


@dataclass(frozen=True)
class StableOrientationConfig:
    robust_tilt_deg: float = 5.0
    min_support_area_m2: float = 1.0e-6
    min_support_area_fraction: float = 0.01
    coplanar_tolerance_m: float = 1.0e-6
    xy_world: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.robust_tilt_deg < 0.0 or self.robust_tilt_deg >= 90.0:
            raise ValueError("robust_tilt_deg must be >= 0 and < 90.")
        if self.min_support_area_m2 < 0.0:
            raise ValueError("min_support_area_m2 must be >= 0.")
        if self.min_support_area_fraction < 0.0:
            raise ValueError("min_support_area_fraction must be >= 0.")
        if self.coplanar_tolerance_m < 0.0:
            raise ValueError("coplanar_tolerance_m must be >= 0.")
        if len(self.xy_world) != 2:
            raise ValueError("xy_world must contain exactly two values.")
        object.__setattr__(self, "robust_tilt_deg", float(self.robust_tilt_deg))
        object.__setattr__(self, "min_support_area_m2", float(self.min_support_area_m2))
        object.__setattr__(self, "min_support_area_fraction", float(self.min_support_area_fraction))
        object.__setattr__(self, "coplanar_tolerance_m", float(self.coplanar_tolerance_m))
        object.__setattr__(self, "xy_world", tuple(float(value) for value in self.xy_world))


@dataclass(frozen=True)
class StableOrientationCandidate:
    candidate_id: str
    normal_obj: tuple[float, float, float]
    area_m2: float
    vertex_indices: tuple[int, ...]
    vertices_obj: tuple[tuple[float, float, float], ...]
    com_obj: tuple[float, float, float]
    com_projection_obj: tuple[float, float, float]
    stability_margin_m: float
    com_height_m: float
    robust_required_margin_m: float
    max_stable_tilt_deg: float
    com_method: str
    rejection_reason: str


@dataclass(frozen=True)
class StableOrientation:
    orientation_id: str
    object_pose_world: ObjectWorldPose
    normal_obj: tuple[float, float, float]
    area_m2: float
    vertex_indices: tuple[int, ...]
    vertices_obj: tuple[tuple[float, float, float], ...]
    com_obj: tuple[float, float, float]
    com_projection_obj: tuple[float, float, float]
    stability_margin_m: float
    com_height_m: float
    robust_required_margin_m: float
    max_stable_tilt_deg: float
    com_method: str


@dataclass(frozen=True)
class StableOrientationResult:
    orientations: tuple[StableOrientation, ...]
    rejected_candidates: tuple[StableOrientationCandidate, ...]
    com_obj: tuple[float, float, float]
    com_method: str
    raw_facet_count: int
    area_threshold_m2: float
    robust_tilt_deg: float


@dataclass(frozen=True)
class _FacetGroup:
    normal: np.ndarray
    offset: float
    area: float
    vertex_indices: tuple[int, ...]


@dataclass(frozen=True)
class _SupportMetrics:
    facet: _FacetGroup
    vertices_obj: np.ndarray
    com_obj: np.ndarray
    com_projection_obj: np.ndarray
    stability_margin_m: float
    com_height_m: float
    robust_required_margin_m: float
    max_stable_tilt_deg: float
    com_method: str


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-10:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def _triangle_area(vertices: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])))


def _convex_hull_facets(mesh_local: object, *, coplanar_tolerance_m: float) -> list[_FacetGroup]:
    vertices = np.asarray(mesh_local.vertices_obj, dtype=float)
    if len(vertices) < 4:
        return []
    try:
        hull = ConvexHull(vertices)
    except QhullError:
        return []

    groups: list[dict[str, object]] = []
    for simplex, equation in zip(hull.simplices, hull.equations, strict=True):
        normal = _normalize(np.asarray(equation[:3], dtype=float))
        offset = float(equation[3])
        area = _triangle_area(vertices[np.asarray(simplex, dtype=np.int64)])
        if area <= 1.0e-12:
            continue
        matched: dict[str, object] | None = None
        for group in groups:
            group_normal = np.asarray(group["normal"], dtype=float)
            group_offset = float(group["offset"])
            if (
                float(np.dot(normal, group_normal)) >= 1.0 - 1.0e-6
                and abs(offset - group_offset) <= float(coplanar_tolerance_m)
            ):
                matched = group
                break
        if matched is None:
            groups.append(
                {
                    "normal": normal,
                    "offset": offset,
                    "area": area,
                    "indices": set(int(index) for index in simplex),
                }
            )
        else:
            matched["area"] = float(matched["area"]) + area
            matched["indices"].update(int(index) for index in simplex)  # type: ignore[union-attr]

    return [
        _FacetGroup(
            normal=np.asarray(group["normal"], dtype=float),
            offset=float(group["offset"]),
            area=float(group["area"]),
            vertex_indices=tuple(sorted(int(index) for index in group["indices"])),
        )
        for group in groups
    ]


def _mesh_center_of_mass(mesh_local: object) -> tuple[np.ndarray, str]:
    vertices = np.asarray(mesh_local.vertices_obj, dtype=float)
    faces = np.asarray(mesh_local.faces, dtype=np.int64)
    if trimesh is not None:
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            center_mass = np.asarray(mesh.center_mass, dtype=float)
            if bool(getattr(mesh, "is_volume", False)) and center_mass.shape == (3,) and np.all(np.isfinite(center_mass)):
                return center_mass, "volume"
        except Exception:
            pass
    return np.asarray(mesh_area_weighted_triangle_centroid(mesh_local), dtype=float), "surface_centroid_fallback"


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _normalize(normal)
    axis = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(axis, normal))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0], dtype=float)
    basis_u = _normalize(axis - float(np.dot(axis, normal)) * normal)
    basis_v = _normalize(np.cross(normal, basis_u))
    return basis_u, basis_v


def _ordered_polygon_2d(points_2d: np.ndarray) -> np.ndarray:
    if len(points_2d) < 3:
        return points_2d
    try:
        hull = ConvexHull(points_2d)
    except QhullError:
        return points_2d
    return points_2d[np.asarray(hull.vertices, dtype=np.int64)]


def _convex_polygon_margin(point_2d: np.ndarray, polygon_2d: np.ndarray) -> float:
    if len(polygon_2d) < 3:
        return float("-inf")
    signed_distances: list[float] = []
    for index, start in enumerate(polygon_2d):
        end = polygon_2d[(index + 1) % len(polygon_2d)]
        edge = end - start
        edge_len = float(np.linalg.norm(edge))
        if edge_len <= 1.0e-12:
            continue
        rel = point_2d - start
        signed_distances.append(float((edge[0] * rel[1] - edge[1] * rel[0]) / edge_len))
    if not signed_distances:
        return float("-inf")
    return min(signed_distances)


def _rotation_aligning_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _normalize(source)
    target = _normalize(target)
    dot = min(1.0, max(-1.0, float(np.dot(source, target))))
    if dot > 1.0 - 1.0e-10:
        return np.eye(3, dtype=float)
    if dot < -1.0 + 1.0e-10:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(axis, source))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = _normalize(axis - float(np.dot(axis, source)) * source)
        return _axis_angle_to_rotmat(axis, math.pi)
    axis = np.cross(source, target)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + skew + skew @ skew * (1.0 / (1.0 + dot))


def _axis_angle_to_rotmat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = [float(v) for v in axis]
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def _pose_for_support_facet(
    *,
    mesh_local: object,
    normal_obj: tuple[float, float, float],
    xy_world: tuple[float, float],
) -> ObjectWorldPose:
    rotation_world_from_obj = _rotation_aligning_vectors(
        np.asarray(normal_obj, dtype=float),
        np.array([0.0, 0.0, -1.0], dtype=float),
    )
    vertices = np.asarray(mesh_local.vertices_obj, dtype=float)
    rotated_vertices = vertices @ rotation_world_from_obj.T
    min_z = float(rotated_vertices[:, 2].min())
    translation = (float(xy_world[0]), float(xy_world[1]), float(-min_z))
    return ObjectWorldPose(
        position_world=translation,
        orientation_xyzw_world=rotmat_to_quat_xyzw(rotation_world_from_obj),
    )


def _support_metrics(
    *,
    mesh_local: object,
    facet: _FacetGroup,
    com_obj: np.ndarray,
    com_method: str,
    robust_tilt_deg: float,
) -> _SupportMetrics:
    vertices = np.asarray(mesh_local.vertices_obj, dtype=float)
    facet_vertices = vertices[np.asarray(facet.vertex_indices, dtype=np.int64)]
    signed_distance = float(np.dot(facet.normal, com_obj) + facet.offset)
    com_projection_obj = com_obj - signed_distance * facet.normal
    com_height_m = max(0.0, -signed_distance)
    basis_u, basis_v = _plane_basis(facet.normal)
    origin = facet_vertices[0]
    facet_points_2d = np.column_stack(
        ((facet_vertices - origin[None, :]) @ basis_u, (facet_vertices - origin[None, :]) @ basis_v)
    )
    polygon_2d = _ordered_polygon_2d(facet_points_2d)
    com_2d = np.asarray(
        [float(np.dot(com_projection_obj - origin, basis_u)), float(np.dot(com_projection_obj - origin, basis_v))],
        dtype=float,
    )
    margin = _convex_polygon_margin(com_2d, polygon_2d)
    robust_required_margin_m = float(com_height_m * math.tan(math.radians(float(robust_tilt_deg))))
    if margin < 0.0:
        max_tilt_deg = 0.0
    elif com_height_m <= 1.0e-12:
        max_tilt_deg = 90.0
    else:
        max_tilt_deg = math.degrees(math.atan(float(margin) / com_height_m))
    return _SupportMetrics(
        facet=facet,
        vertices_obj=facet_vertices,
        com_obj=com_obj,
        com_projection_obj=com_projection_obj,
        stability_margin_m=float(margin),
        com_height_m=float(com_height_m),
        robust_required_margin_m=float(robust_required_margin_m),
        max_stable_tilt_deg=float(max_tilt_deg),
        com_method=com_method,
    )


def _candidate_from_metrics(
    candidate_id: str,
    metrics: _SupportMetrics,
    *,
    rejection_reason: str,
) -> StableOrientationCandidate:
    return StableOrientationCandidate(
        candidate_id=candidate_id,
        normal_obj=tuple(float(v) for v in metrics.facet.normal),
        area_m2=float(metrics.facet.area),
        vertex_indices=metrics.facet.vertex_indices,
        vertices_obj=tuple(tuple(float(v) for v in vertex) for vertex in metrics.vertices_obj),
        com_obj=tuple(float(v) for v in metrics.com_obj),
        com_projection_obj=tuple(float(v) for v in metrics.com_projection_obj),
        stability_margin_m=float(metrics.stability_margin_m),
        com_height_m=float(metrics.com_height_m),
        robust_required_margin_m=float(metrics.robust_required_margin_m),
        max_stable_tilt_deg=float(metrics.max_stable_tilt_deg),
        com_method=metrics.com_method,
        rejection_reason=rejection_reason,
    )


def enumerate_stable_orientations(
    mesh_local: object,
    config: StableOrientationConfig | None = None,
) -> StableOrientationResult:
    cfg = config or StableOrientationConfig()
    com_obj, com_method = _mesh_center_of_mass(mesh_local)
    facets = _convex_hull_facets(mesh_local, coplanar_tolerance_m=cfg.coplanar_tolerance_m)
    largest_area_m2 = max((float(facet.area) for facet in facets), default=0.0)
    area_threshold_m2 = max(cfg.min_support_area_m2, largest_area_m2 * cfg.min_support_area_fraction)
    stable_metrics: list[_SupportMetrics] = []
    rejected: list[StableOrientationCandidate] = []

    sorted_facets = sorted(
        facets,
        key=lambda facet: (
            -float(facet.area),
            tuple(round(float(value), 9) for value in facet.normal),
            facet.vertex_indices,
        ),
    )
    for candidate_index, facet in enumerate(sorted_facets):
        metrics = _support_metrics(
            mesh_local=mesh_local,
            facet=facet,
            com_obj=com_obj,
            com_method=com_method,
            robust_tilt_deg=cfg.robust_tilt_deg,
        )
        candidate_id = f"facet_{candidate_index:03d}"
        if float(facet.area) < area_threshold_m2:
            rejected.append(_candidate_from_metrics(candidate_id, metrics, rejection_reason="support_area_too_small"))
            continue
        if metrics.stability_margin_m < 0.0:
            rejected.append(_candidate_from_metrics(candidate_id, metrics, rejection_reason="com_outside_support"))
            continue
        if metrics.stability_margin_m + 1.0e-12 < metrics.robust_required_margin_m:
            rejected.append(_candidate_from_metrics(candidate_id, metrics, rejection_reason="tilt_margin_too_small"))
            continue
        stable_metrics.append(metrics)

    stable_metrics = sorted(
        stable_metrics,
        key=lambda metrics: (
            -float(metrics.facet.area),
            -float(metrics.max_stable_tilt_deg),
            tuple(round(float(value), 9) for value in metrics.facet.normal),
            metrics.facet.vertex_indices,
        ),
    )
    orientations = tuple(
        StableOrientation(
            orientation_id=f"orientation_{index:03d}",
            object_pose_world=_pose_for_support_facet(
                mesh_local=mesh_local,
                normal_obj=tuple(float(v) for v in metrics.facet.normal),
                xy_world=cfg.xy_world,
            ),
            normal_obj=tuple(float(v) for v in metrics.facet.normal),
            area_m2=float(metrics.facet.area),
            vertex_indices=metrics.facet.vertex_indices,
            vertices_obj=tuple(tuple(float(v) for v in vertex) for vertex in metrics.vertices_obj),
            com_obj=tuple(float(v) for v in metrics.com_obj),
            com_projection_obj=tuple(float(v) for v in metrics.com_projection_obj),
            stability_margin_m=float(metrics.stability_margin_m),
            com_height_m=float(metrics.com_height_m),
            robust_required_margin_m=float(metrics.robust_required_margin_m),
            max_stable_tilt_deg=float(metrics.max_stable_tilt_deg),
            com_method=metrics.com_method,
        )
        for index, metrics in enumerate(stable_metrics)
    )
    return StableOrientationResult(
        orientations=orientations,
        rejected_candidates=tuple(rejected),
        com_obj=tuple(float(v) for v in com_obj),
        com_method=com_method,
        raw_facet_count=len(facets),
        area_threshold_m2=float(area_threshold_m2),
        robust_tilt_deg=float(cfg.robust_tilt_deg),
    )


def pose_payload(pose: ObjectWorldPose) -> dict[str, object]:
    return {
        "position_world": list(pose.position_world),
        "orientation_xyzw_world": list(pose.orientation_xyzw_world),
    }


def stable_orientation_payload(orientation: StableOrientation) -> dict[str, object]:
    return {
        "orientation_id": orientation.orientation_id,
        "object_pose_world": pose_payload(orientation.object_pose_world),
        "normal_obj": list(orientation.normal_obj),
        "area_m2": orientation.area_m2,
        "vertex_indices": list(orientation.vertex_indices),
        "vertices_obj": [list(vertex) for vertex in orientation.vertices_obj],
        "com_obj": list(orientation.com_obj),
        "com_projection_obj": list(orientation.com_projection_obj),
        "stability_margin_m": orientation.stability_margin_m,
        "com_height_m": orientation.com_height_m,
        "robust_required_margin_m": orientation.robust_required_margin_m,
        "max_stable_tilt_deg": orientation.max_stable_tilt_deg,
        "com_method": orientation.com_method,
    }


def stable_orientation_candidate_payload(candidate: StableOrientationCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "normal_obj": list(candidate.normal_obj),
        "area_m2": candidate.area_m2,
        "vertex_indices": list(candidate.vertex_indices),
        "vertices_obj": [list(vertex) for vertex in candidate.vertices_obj],
        "com_obj": list(candidate.com_obj),
        "com_projection_obj": list(candidate.com_projection_obj),
        "stability_margin_m": candidate.stability_margin_m,
        "com_height_m": candidate.com_height_m,
        "robust_required_margin_m": candidate.robust_required_margin_m,
        "max_stable_tilt_deg": candidate.max_stable_tilt_deg,
        "com_method": candidate.com_method,
        "rejection_reason": candidate.rejection_reason,
    }


def stable_orientation_result_payload(result: StableOrientationResult) -> dict[str, object]:
    return {
        "com_obj": list(result.com_obj),
        "com_method": result.com_method,
        "raw_facet_count": result.raw_facet_count,
        "area_threshold_m2": result.area_threshold_m2,
        "robust_tilt_deg": result.robust_tilt_deg,
        "stable_orientation_count": len(result.orientations),
        "rejected_candidate_count": len(result.rejected_candidates),
        "orientations": [stable_orientation_payload(orientation) for orientation in result.orientations],
        "rejected_candidates": [
            stable_orientation_candidate_payload(candidate) for candidate in result.rejected_candidates
        ],
    }
