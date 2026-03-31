"""Minimal antipodal grasp generation from object geometry only."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def _rotmat_to_quat_xyzw(rotmat: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotmat))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotmat[2, 1] - rotmat[1, 2]) / s
        y = (rotmat[0, 2] - rotmat[2, 0]) / s
        z = (rotmat[1, 0] - rotmat[0, 1]) / s
    elif rotmat[0, 0] > rotmat[1, 1] and rotmat[0, 0] > rotmat[2, 2]:
        s = math.sqrt(1.0 + rotmat[0, 0] - rotmat[1, 1] - rotmat[2, 2]) * 2.0
        w = (rotmat[2, 1] - rotmat[1, 2]) / s
        x = 0.25 * s
        y = (rotmat[0, 1] + rotmat[1, 0]) / s
        z = (rotmat[0, 2] + rotmat[2, 0]) / s
    elif rotmat[1, 1] > rotmat[2, 2]:
        s = math.sqrt(1.0 + rotmat[1, 1] - rotmat[0, 0] - rotmat[2, 2]) * 2.0
        w = (rotmat[0, 2] - rotmat[2, 0]) / s
        x = (rotmat[0, 1] + rotmat[1, 0]) / s
        y = 0.25 * s
        z = (rotmat[1, 2] + rotmat[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotmat[2, 2] - rotmat[0, 0] - rotmat[1, 1]) * 2.0
        w = (rotmat[1, 0] - rotmat[0, 1]) / s
        x = (rotmat[0, 2] + rotmat[2, 0]) / s
        y = (rotmat[1, 2] + rotmat[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=float)
    quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat)


def _axis_angle_to_rotmat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = [float(v) for v in axis]
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def _triangle_points_from_barycentric(vertices: np.ndarray, barycentric_uv: np.ndarray) -> np.ndarray:
    u = barycentric_uv[:, 0:1]
    v = barycentric_uv[:, 1:2]
    w = 1.0 - u - v
    return u * vertices[:, 0, :] + v * vertices[:, 1, :] + w * vertices[:, 2, :]


@dataclass(frozen=True)
class TriangleMesh:
    """Triangle mesh in the object frame."""

    vertices_obj: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_obj, dtype=float)
        faces = np.asarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices_obj must have shape (N, 3).")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3).")
        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError("Triangle mesh must be non-empty.")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("faces contains an out-of-range vertex index.")
        object.__setattr__(self, "vertices_obj", vertices.copy())
        object.__setattr__(self, "faces", faces.copy())

    @property
    def face_vertices(self) -> np.ndarray:
        return self.vertices_obj[self.faces]


@dataclass(frozen=True)
class SurfaceSample:
    """Sampled object-surface point with an associated outward normal."""

    point_obj: tuple[float, float, float]
    normal_obj: tuple[float, float, float]
    face_index: int


@dataclass(frozen=True)
class ObjectFrameGraspCandidate:
    """Minimal parallel-jaw grasp candidate expressed in object coordinates."""

    grasp_position_obj: tuple[float, float, float]
    grasp_orientation_xyzw_obj: tuple[float, float, float, float]
    contact_point_a_obj: tuple[float, float, float]
    contact_point_b_obj: tuple[float, float, float]
    contact_normal_a_obj: tuple[float, float, float]
    contact_normal_b_obj: tuple[float, float, float]
    jaw_width: float
    roll_angle_rad: float

    def closing_axis_obj(self) -> tuple[float, float, float]:
        point_a = np.asarray(self.contact_point_a_obj, dtype=float)
        point_b = np.asarray(self.contact_point_b_obj, dtype=float)
        return tuple(float(v) for v in _normalize(point_b - point_a))


@dataclass(frozen=True)
class AntipodalGraspGeneratorConfig:
    """Config for a minimal object-frame antipodal grasp generator."""

    num_surface_samples: int = 192
    min_jaw_width: float = 0.02
    max_jaw_width: float = 0.08
    antipodal_cosine_threshold: float = 0.94
    roll_angles_rad: tuple[float, ...] = (0.0,)
    max_pair_checks: int = 4096
    finger_depth: float = 0.008
    finger_thickness: float = 0.01
    finger_length: float = 0.012
    finger_clearance: float = 0.002
    contact_patch_radius: float = 0.006
    collision_sample_count: int = 256
    rng_seed: int = 0


class AntipodalMeshGraspGenerator:
    """Generate minimal antipodal parallel-jaw grasps from object geometry."""

    def __init__(self, config: AntipodalGraspGeneratorConfig | None = None) -> None:
        self._config = config or AntipodalGraspGeneratorConfig()

    def generate(self, mesh: TriangleMesh) -> list[ObjectFrameGraspCandidate]:
        surface_samples = self.sample_surface(mesh, num_samples=self._config.num_surface_samples)
        collision_points = self._surface_collision_points(mesh)
        pair_indices = self._candidate_pair_indices(surface_samples)
        candidates: list[ObjectFrameGraspCandidate] = []
        seen_keys: set[tuple[float, ...]] = set()

        for i, j in pair_indices:
            sample_a, sample_b, closing_axis = self._canonicalize_pair(surface_samples[i], surface_samples[j])
            point_a = np.asarray(sample_a.point_obj, dtype=float)
            point_b = np.asarray(sample_b.point_obj, dtype=float)
            normal_a = np.asarray(sample_a.normal_obj, dtype=float)
            normal_b = np.asarray(sample_b.normal_obj, dtype=float)
            jaw_width = float(np.linalg.norm(point_b - point_a))
            if jaw_width < self._config.min_jaw_width or jaw_width > self._config.max_jaw_width:
                continue
            if not self._is_antipodal(closing_axis=closing_axis, normal_a=normal_a, normal_b=normal_b):
                continue

            grasp_center = 0.5 * (point_a + point_b)
            base_rotmat = self._base_grasp_rotmat(closing_axis)
            if not self._passes_finger_clearance(
                collision_points=collision_points,
                contact_point_a=point_a,
                contact_point_b=point_b,
                contact_normal_a=normal_a,
                contact_normal_b=normal_b,
                closing_axis=closing_axis,
            ):
                continue

            for roll_angle_rad in self._config.roll_angles_rad:
                roll_rotmat = _axis_angle_to_rotmat(closing_axis, float(roll_angle_rad))
                grasp_rotmat = roll_rotmat @ base_rotmat
                grasp_quat = _rotmat_to_quat_xyzw(grasp_rotmat)
                candidate = ObjectFrameGraspCandidate(
                    grasp_position_obj=tuple(float(v) for v in grasp_center),
                    grasp_orientation_xyzw_obj=grasp_quat,
                    contact_point_a_obj=sample_a.point_obj,
                    contact_point_b_obj=sample_b.point_obj,
                    contact_normal_a_obj=sample_a.normal_obj,
                    contact_normal_b_obj=sample_b.normal_obj,
                    jaw_width=jaw_width,
                    roll_angle_rad=float(roll_angle_rad),
                )
                dedupe_key = (
                    *np.round(np.asarray(candidate.grasp_position_obj, dtype=float), 6).tolist(),
                    *np.round(np.asarray(candidate.contact_point_a_obj, dtype=float), 6).tolist(),
                    *np.round(np.asarray(candidate.contact_point_b_obj, dtype=float), 6).tolist(),
                    round(float(candidate.roll_angle_rad), 6),
                )
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                candidates.append(candidate)

        return candidates

    def sample_surface(self, mesh: TriangleMesh, *, num_samples: int) -> list[SurfaceSample]:
        if num_samples <= 0:
            return []
        face_vertices = mesh.face_vertices
        edge_ab = face_vertices[:, 1, :] - face_vertices[:, 0, :]
        edge_ac = face_vertices[:, 2, :] - face_vertices[:, 0, :]
        raw_normals = np.cross(edge_ab, edge_ac)
        double_areas = np.linalg.norm(raw_normals, axis=1)
        valid_mask = double_areas > 1.0e-10
        if not np.any(valid_mask):
            raise ValueError("Triangle mesh does not contain any non-degenerate faces.")

        valid_face_vertices = face_vertices[valid_mask]
        valid_face_indices = np.nonzero(valid_mask)[0]
        valid_normals = raw_normals[valid_mask] / double_areas[valid_mask][:, None]
        weights = double_areas[valid_mask] / np.sum(double_areas[valid_mask])

        rng = np.random.default_rng(self._config.rng_seed)
        sampled_face_slots = rng.choice(len(valid_face_indices), size=num_samples, replace=True, p=weights)
        barycentric = rng.random((num_samples, 2))
        reflected = barycentric.sum(axis=1) > 1.0
        barycentric[reflected] = 1.0 - barycentric[reflected]
        sampled_vertices = valid_face_vertices[sampled_face_slots]
        sampled_points = _triangle_points_from_barycentric(sampled_vertices, barycentric)
        sampled_normals = valid_normals[sampled_face_slots]

        return [
            SurfaceSample(
                point_obj=tuple(float(v) for v in sampled_points[sample_idx]),
                normal_obj=tuple(float(v) for v in sampled_normals[sample_idx]),
                face_index=int(valid_face_indices[sampled_face_slots[sample_idx]]),
            )
            for sample_idx in range(num_samples)
        ]

    def _surface_collision_points(self, mesh: TriangleMesh) -> np.ndarray:
        face_vertices = mesh.face_vertices
        centroids = face_vertices.mean(axis=1)
        vertices = mesh.vertices_obj
        extra_samples = self.sample_surface(mesh, num_samples=self._config.collision_sample_count)
        sampled_points = np.array([sample.point_obj for sample in extra_samples], dtype=float)
        return np.vstack((vertices, centroids, sampled_points))

    def _candidate_pair_indices(self, samples: list[SurfaceSample]) -> list[tuple[int, int]]:
        if len(samples) < 2:
            return []

        points = np.array([sample.point_obj for sample in samples], dtype=float)
        tree = cKDTree(points)
        max_radius = self._config.max_jaw_width
        pair_indices: list[tuple[int, int]] = []
        seen_pairs: set[tuple[int, int]] = set()

        for i, point in enumerate(points):
            neighbors = tree.query_ball_point(point, r=max_radius)
            for j in neighbors:
                if j <= i:
                    continue
                pair = (i, j)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                pair_indices.append(pair)

        if len(pair_indices) <= self._config.max_pair_checks:
            return pair_indices
        step = max(1, len(pair_indices) // self._config.max_pair_checks)
        return pair_indices[::step][: self._config.max_pair_checks]

    def _canonicalize_pair(
        self,
        sample_a: SurfaceSample,
        sample_b: SurfaceSample,
    ) -> tuple[SurfaceSample, SurfaceSample, np.ndarray]:
        point_a = np.asarray(sample_a.point_obj, dtype=float)
        point_b = np.asarray(sample_b.point_obj, dtype=float)
        closing_axis = _normalize(point_b - point_a)
        dot_a = float(np.dot(np.asarray(sample_a.normal_obj, dtype=float), closing_axis))
        dot_b = float(np.dot(np.asarray(sample_b.normal_obj, dtype=float), closing_axis))
        if dot_a > 0.0 or dot_b < 0.0:
            return sample_b, sample_a, -closing_axis
        return sample_a, sample_b, closing_axis

    def _is_antipodal(self, *, closing_axis: np.ndarray, normal_a: np.ndarray, normal_b: np.ndarray) -> bool:
        threshold = self._config.antipodal_cosine_threshold
        inward_alignment_a = float(np.dot(-normal_a, closing_axis))
        inward_alignment_b = float(np.dot(normal_b, closing_axis))
        normal_opposition = float(np.dot(normal_a, normal_b))
        return (
            inward_alignment_a >= threshold
            and inward_alignment_b >= threshold
            and normal_opposition <= -threshold
        )

    def _base_grasp_rotmat(self, closing_axis: np.ndarray) -> np.ndarray:
        gripper_y = _normalize(closing_axis)
        reference_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(gripper_y, reference_axis))) > 0.95:
            reference_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        gripper_x = reference_axis - float(np.dot(reference_axis, gripper_y)) * gripper_y
        gripper_x = _normalize(gripper_x)
        gripper_z = _normalize(np.cross(gripper_x, gripper_y))
        gripper_x = _normalize(np.cross(gripper_y, gripper_z))
        return np.column_stack((gripper_x, gripper_y, gripper_z))

    def _passes_finger_clearance(
        self,
        *,
        collision_points: np.ndarray,
        contact_point_a: np.ndarray,
        contact_point_b: np.ndarray,
        contact_normal_a: np.ndarray,
        contact_normal_b: np.ndarray,
        closing_axis: np.ndarray,
    ) -> bool:
        box_a = self._finger_box(
            contact_point=contact_point_a,
            contact_normal=contact_normal_a,
            closing_axis=closing_axis,
            invert_closing_axis=False,
        )
        box_b = self._finger_box(
            contact_point=contact_point_b,
            contact_normal=contact_normal_b,
            closing_axis=closing_axis,
            invert_closing_axis=True,
        )
        exclusion_radius = self._config.contact_patch_radius
        distances_to_a = np.linalg.norm(collision_points - contact_point_a[None, :], axis=1)
        distances_to_b = np.linalg.norm(collision_points - contact_point_b[None, :], axis=1)
        collision_mask = (distances_to_a > exclusion_radius) & (distances_to_b > exclusion_radius)
        relevant_points = collision_points[collision_mask]
        return not (self._points_in_box(relevant_points, *box_a) or self._points_in_box(relevant_points, *box_b))

    def _finger_box(
        self,
        *,
        contact_point: np.ndarray,
        contact_normal: np.ndarray,
        closing_axis: np.ndarray,
        invert_closing_axis: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        finger_x = _normalize(contact_normal)
        finger_y = -closing_axis if invert_closing_axis else closing_axis
        finger_y = finger_y - float(np.dot(finger_y, finger_x)) * finger_x
        if np.linalg.norm(finger_y) <= 1.0e-8:
            reference_axis = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(reference_axis, finger_x))) > 0.95:
                reference_axis = np.array([1.0, 0.0, 0.0], dtype=float)
            finger_y = reference_axis - float(np.dot(reference_axis, finger_x)) * finger_x
            if invert_closing_axis:
                finger_y = -finger_y
        finger_y = _normalize(finger_y)
        finger_z = _normalize(np.cross(finger_x, finger_y))
        rotation = np.column_stack((finger_x, finger_y, finger_z))
        half_extents = 0.5 * np.array(
            [
                self._config.finger_depth,
                self._config.finger_length,
                self._config.finger_thickness,
            ],
            dtype=float,
        )
        # Keep the box centered near the contact plane so broad finger pads are
        # conservatively rejected when nearby object geometry crowds the pair.
        center = contact_point + finger_x * (0.5 * self._config.finger_clearance)
        return center, rotation, half_extents

    @staticmethod
    def _points_in_box(points: np.ndarray, center: np.ndarray, rotation: np.ndarray, half_extents: np.ndarray) -> bool:
        if len(points) == 0:
            return False
        local_points = (points - center[None, :]) @ rotation
        within = np.all(np.abs(local_points) <= (half_extents[None, :] + 1.0e-9), axis=1)
        return bool(np.any(within))


def export_grasp_candidates_json(
    candidates: Iterable[ObjectFrameGraspCandidate],
    output_path: str | Path,
) -> None:
    """Write object-frame grasp candidates to a JSON file."""

    payload = [
        {
            "grasp_pose_obj": {
                "position": [float(v) for v in candidate.grasp_position_obj],
                "orientation_xyzw": [float(v) for v in candidate.grasp_orientation_xyzw_obj],
            },
            "contact_points_obj": [
                [float(v) for v in candidate.contact_point_a_obj],
                [float(v) for v in candidate.contact_point_b_obj],
            ],
            "contact_normals_obj": [
                [float(v) for v in candidate.contact_normal_a_obj],
                [float(v) for v in candidate.contact_normal_b_obj],
            ],
            "jaw_width": float(candidate.jaw_width),
            "roll_angle_rad": float(candidate.roll_angle_rad),
        }
        for candidate in candidates
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
