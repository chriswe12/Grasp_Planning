"""Deterministic grasp generation for a box-like cube."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import math
import numpy as np


@dataclass(frozen=True)
class GraspCandidate:
    """World-frame grasp pose and metadata for a parallel gripper."""

    position_w: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    normal_w: tuple[float, float, float]
    pregrasp_offset: float
    gripper_width: float
    score: float
    label: str

    @property
    def pregrasp_position_w(self) -> tuple[float, float, float]:
        """Pose reached before the straight-line approach."""

        normal = np.asarray(self.normal_w, dtype=float)
        position = np.asarray(self.position_w, dtype=float)
        pregrasp = position - normal * self.pregrasp_offset
        return tuple(float(v) for v in pregrasp)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def _quat_to_rotmat_xyzw(quat_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


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


def _make_gripper_orientation(normal_w: np.ndarray) -> tuple[float, float, float, float]:
    """Construct a gripper orientation from the face normal.

    The gripper x-axis points along the approach direction toward the object.
    The z-axis is kept as aligned with world up as possible.
    """

    approach_axis = _normalize(normal_w)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(approach_axis, world_up))) > 0.95:
        tangent_seed = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        tangent_seed = world_up

    gripper_y = _normalize(np.cross(tangent_seed, approach_axis))
    gripper_z = _normalize(np.cross(approach_axis, gripper_y))
    rotmat = np.column_stack((approach_axis, gripper_y, gripper_z))
    return _rotmat_to_quat_xyzw(rotmat)


class CubeFaceGraspGenerator:
    """Generate and rank face-centered grasps for a cube."""

    def __init__(
        self,
        *,
        cube_size: tuple[float, float, float],
        pregrasp_offset: float = 0.10,
        finger_clearance: float = 0.01,
    ) -> None:
        self._cube_size = np.asarray(cube_size, dtype=float)
        self._half_extents = self._cube_size / 2.0
        self._pregrasp_offset = float(pregrasp_offset)
        self._finger_clearance = float(finger_clearance)

    def generate(
        self,
        *,
        cube_position_w: tuple[float, float, float],
        cube_orientation_xyzw: tuple[float, float, float, float],
        robot_base_position_w: tuple[float, float, float],
    ) -> list[GraspCandidate]:
        """Return face-centered grasp candidates ordered by descending score."""

        rotmat = _quat_to_rotmat_xyzw(cube_orientation_xyzw)
        cube_position = np.asarray(cube_position_w, dtype=float)
        robot_base_position = np.asarray(robot_base_position_w, dtype=float)
        face_defs = (
            ("+x", np.array([1.0, 0.0, 0.0]), self._half_extents[0], self._cube_size[1]),
            ("-x", np.array([-1.0, 0.0, 0.0]), self._half_extents[0], self._cube_size[1]),
            ("+y", np.array([0.0, 1.0, 0.0]), self._half_extents[1], self._cube_size[0]),
            ("-y", np.array([0.0, -1.0, 0.0]), self._half_extents[1], self._cube_size[0]),
            ("+z", np.array([0.0, 0.0, 1.0]), self._half_extents[2], self._cube_size[0]),
            ("-z", np.array([0.0, 0.0, -1.0]), self._half_extents[2], self._cube_size[0]),
        )

        candidates: list[GraspCandidate] = []
        for label, normal_obj, face_offset, closing_span in face_defs:
            point_obj = normal_obj * face_offset
            point_w = cube_position + rotmat @ point_obj
            normal_w = _normalize(rotmat @ normal_obj)
            orientation = _make_gripper_orientation(-normal_w)
            score = self._score_candidate(
                point_w=point_w,
                normal_w=normal_w,
                robot_base_position_w=robot_base_position,
                label=label,
            )
            candidates.append(
                GraspCandidate(
                    position_w=tuple(float(v) for v in cube_position),
                    orientation_xyzw=orientation,
                    normal_w=tuple(float(v) for v in normal_w),
                    pregrasp_offset=self._pregrasp_offset,
                    gripper_width=float(closing_span + self._finger_clearance),
                    score=score,
                    label=label,
                )
            )

        return sorted(candidates, key=lambda grasp: grasp.score, reverse=True)

    @staticmethod
    def _score_candidate(
        *,
        point_w: np.ndarray,
        normal_w: np.ndarray,
        robot_base_position_w: np.ndarray,
        label: str,
    ) -> float:
        distance_score = -float(np.linalg.norm(point_w - robot_base_position_w))
        side_grasp_bonus = 0.25 if label in {"+x", "-x", "+y", "-y"} else 0.0
        top_grasp_penalty = -0.15 if label == "+z" else 0.0
        underside_penalty = -2.0 if label == "-z" else 0.0
        horizontal_bonus = 0.1 * (1.0 - abs(float(normal_w[2])))
        return distance_score + side_grasp_bonus + top_grasp_penalty + underside_penalty + horizontal_bonus
