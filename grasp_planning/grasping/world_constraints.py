"""World-frame post-filters for object-frame grasp candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .collision import (
    BoxCollisionPrimitive,
    FingerBoxGripperCollisionModel,
    FingerBoxWithHandMeshCollisionModel,
    FrankaHandFingerCollisionModel,
    MeshCollisionPrimitive,
)
from .finger_geometry import finger_box_corners
from .mesh_antipodal_grasp_generator import ObjectFrameGraspCandidate


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def _quat_to_rotmat_xyzw(quat_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=float)
    if quat.shape != (4,):
        raise ValueError("Quaternion must have shape (4,).")
    quat = quat / np.linalg.norm(quat)
    x, y, z, w = [float(v) for v in quat]
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


@dataclass(frozen=True)
class ObjectWorldPose:
    """Rigid transform from object frame to world frame."""

    position_world: tuple[float, float, float]
    orientation_xyzw_world: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        position = np.asarray(self.position_world, dtype=float)
        orientation = np.asarray(self.orientation_xyzw_world, dtype=float)
        if position.shape != (3,):
            raise ValueError("position_world must have shape (3,).")
        if orientation.shape != (4,):
            raise ValueError("orientation_xyzw_world must have shape (4,).")
        if float(np.linalg.norm(orientation)) < 1.0e-8:
            raise ValueError("orientation_xyzw_world must be non-zero.")
        object.__setattr__(self, "position_world", tuple(float(v) for v in position))
        object.__setattr__(self, "orientation_xyzw_world", tuple(float(v) for v in orientation / np.linalg.norm(orientation)))

    @property
    def rotation_world_from_object(self) -> np.ndarray:
        return _quat_to_rotmat_xyzw(self.orientation_xyzw_world)

    @property
    def translation_world(self) -> np.ndarray:
        return np.asarray(self.position_world, dtype=float)

    def transform_points_to_world(self, points_obj: np.ndarray) -> np.ndarray:
        points = np.asarray(points_obj, dtype=float)
        return points @ self.rotation_world_from_object.T + self.translation_world


@dataclass(frozen=True)
class HalfSpaceWorldConstraint:
    """Keep all tested geometry on or above a world-frame plane."""

    normal_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    offset_world: float = 0.0

    def __post_init__(self) -> None:
        normal = _normalize(np.asarray(self.normal_world, dtype=float))
        object.__setattr__(self, "normal_world", tuple(float(v) for v in normal))
        object.__setattr__(self, "offset_world", float(self.offset_world))

    def signed_distance_world(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=float)
        return points @ np.asarray(self.normal_world, dtype=float) + float(self.offset_world)


_DEFAULT_GROUND_PLANE = HalfSpaceWorldConstraint()


class WorldCollisionConstraintEvaluator:
    """Evaluate world-frame half-space constraints for object-frame grasps."""

    def __init__(
        self,
        collision_model: FingerBoxGripperCollisionModel
        | FingerBoxWithHandMeshCollisionModel
        | FrankaHandFingerCollisionModel,
    ) -> None:
        self._collision_model = collision_model

    def filter_grasps_above_plane(
        self,
        candidates: Iterable[ObjectFrameGraspCandidate],
        *,
        object_pose_world: ObjectWorldPose,
        plane_constraint: HalfSpaceWorldConstraint = _DEFAULT_GROUND_PLANE,
    ) -> list[ObjectFrameGraspCandidate]:
        accepted: list[ObjectFrameGraspCandidate] = []
        for candidate in candidates:
            if self.is_grasp_above_plane(
                candidate,
                object_pose_world=object_pose_world,
                plane_constraint=plane_constraint,
            ):
                accepted.append(candidate)
        return accepted

    def is_grasp_above_plane(
        self,
        candidate: ObjectFrameGraspCandidate,
        *,
        object_pose_world: ObjectWorldPose,
        plane_constraint: HalfSpaceWorldConstraint = _DEFAULT_GROUND_PLANE,
    ) -> bool:
        grasp_rotmat_obj = _quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj)
        contact_point_a = np.asarray(candidate.contact_point_a_obj, dtype=float)
        contact_point_b = np.asarray(candidate.contact_point_b_obj, dtype=float)

        for primitive_obj in self._collision_model.primitives_for_grasp(
            grasp_rotmat=grasp_rotmat_obj,
            contact_point_a=contact_point_a,
            contact_point_b=contact_point_b,
        ):
            primitive_world = _transform_primitive_to_world(primitive_obj, object_pose_world)
            if _primitive_penetrates_plane(primitive_world, plane_constraint):
                return False
        return True


def filter_grasp_candidates_above_plane(
    candidates: Iterable[ObjectFrameGraspCandidate],
    *,
    object_pose_world: ObjectWorldPose,
    collision_model: FingerBoxGripperCollisionModel
    | FingerBoxWithHandMeshCollisionModel
    | FrankaHandFingerCollisionModel,
    plane_constraint: HalfSpaceWorldConstraint = _DEFAULT_GROUND_PLANE,
) -> list[ObjectFrameGraspCandidate]:
    """Convenience wrapper for world-frame plane filtering."""

    evaluator = WorldCollisionConstraintEvaluator(collision_model)
    return evaluator.filter_grasps_above_plane(
        candidates,
        object_pose_world=object_pose_world,
        plane_constraint=plane_constraint,
    )


def _transform_primitive_to_world(
    primitive_obj: BoxCollisionPrimitive | MeshCollisionPrimitive,
    object_pose_world: ObjectWorldPose,
) -> BoxCollisionPrimitive | MeshCollisionPrimitive:
    rotation_world_from_object = object_pose_world.rotation_world_from_object
    translation_world = object_pose_world.translation_world
    if isinstance(primitive_obj, BoxCollisionPrimitive):
        return BoxCollisionPrimitive(
            name=primitive_obj.name,
            center_obj=rotation_world_from_object @ primitive_obj.center_obj + translation_world,
            rotation_obj=rotation_world_from_object @ primitive_obj.rotation_obj,
            half_extents=primitive_obj.half_extents,
        )
    return MeshCollisionPrimitive(
        name=primitive_obj.name,
        vertices_obj=object_pose_world.transform_points_to_world(primitive_obj.vertices_obj),
        faces=primitive_obj.faces,
    )


def _primitive_penetrates_plane(
    primitive_world: BoxCollisionPrimitive | MeshCollisionPrimitive,
    plane_constraint: HalfSpaceWorldConstraint,
) -> bool:
    if isinstance(primitive_world, BoxCollisionPrimitive):
        vertices_world = finger_box_corners(
            primitive_world.center_obj,
            primitive_world.rotation_obj,
            primitive_world.half_extents,
        )
    else:
        vertices_world = primitive_world.vertices_obj
    distances = plane_constraint.signed_distance_world(vertices_world)
    return bool(np.any(distances < 0.0))
