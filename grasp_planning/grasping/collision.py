"""Extensible mesh collision checks for object-frame grasps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .finger_geometry import finger_box_corners, finger_boxes_from_grasp

try:
    import trimesh
    from trimesh.collision import CollisionManager
except Exception:  # pragma: no cover - optional dependency path
    trimesh = None
    CollisionManager = None


class TriangleMeshLike(Protocol):
    vertices_obj: np.ndarray
    faces: np.ndarray

    @property
    def face_vertices(self) -> np.ndarray: ...


@dataclass(frozen=True)
class BoxCollisionPrimitive:
    """A box primitive expressed in object coordinates."""

    name: str
    center_obj: np.ndarray
    rotation_obj: np.ndarray
    half_extents: np.ndarray

    def aabb_bounds_obj(self) -> tuple[np.ndarray, np.ndarray]:
        corners = finger_box_corners(self.center_obj, self.rotation_obj, self.half_extents)
        return corners.min(axis=0), corners.max(axis=0)

    def transform_matrix_obj(self) -> np.ndarray:
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = self.rotation_obj
        transform[:3, 3] = self.center_obj
        return transform


@dataclass(frozen=True)
class FingerBoxGripperCollisionModel:
    """Collision model using the current pair of finger boxes."""

    finger_extent_lateral: float
    finger_extent_closing: float
    finger_extent_approach: float
    finger_clearance: float

    def primitives_for_grasp(
        self,
        *,
        grasp_rotmat: np.ndarray,
        contact_point_a: np.ndarray,
        contact_point_b: np.ndarray,
    ) -> tuple[BoxCollisionPrimitive, ...]:
        box_a, box_b = finger_boxes_from_grasp(
            grasp_rotmat=grasp_rotmat,
            contact_point_a=contact_point_a,
            contact_point_b=contact_point_b,
            finger_extent_lateral=self.finger_extent_lateral,
            finger_extent_closing=self.finger_extent_closing,
            finger_extent_approach=self.finger_extent_approach,
            finger_clearance=self.finger_clearance,
        )
        return (
            BoxCollisionPrimitive(
                name="finger_a",
                center_obj=np.asarray(box_a[0], dtype=float),
                rotation_obj=np.asarray(box_a[1], dtype=float),
                half_extents=np.asarray(box_a[2], dtype=float),
            ),
            BoxCollisionPrimitive(
                name="finger_b",
                center_obj=np.asarray(box_b[0], dtype=float),
                rotation_obj=np.asarray(box_b[1], dtype=float),
                half_extents=np.asarray(box_b[2], dtype=float),
            ),
        )


class MeshCollisionScene(Protocol):
    """Prepared mesh acceleration structure for primitive queries."""

    def intersects_box(
        self,
        primitive: BoxCollisionPrimitive,
    ) -> bool: ...


class MeshCollisionBackend(Protocol):
    """Factory for prepared mesh collision scenes."""

    backend_name: str

    def build_scene(self, mesh: TriangleMeshLike) -> MeshCollisionScene: ...


class TrimeshFclMeshCollisionScene:
    """Mesh collision scene backed by trimesh and FCL."""

    def __init__(self, mesh: TriangleMeshLike) -> None:
        if trimesh is None or CollisionManager is None:
            raise RuntimeError("trimesh/FCL collision backend is unavailable.")
        self._mesh = trimesh.Trimesh(vertices=mesh.vertices_obj, faces=mesh.faces, process=False)
        self._manager = CollisionManager()
        self._manager.add_object("object", self._mesh)

    def intersects_box(
        self,
        primitive: BoxCollisionPrimitive,
    ) -> bool:
        box_mesh = trimesh.creation.box(extents=2.0 * primitive.half_extents)
        result = self._manager.in_collision_single(
            box_mesh,
            transform=primitive.transform_matrix_obj(),
            return_data=False,
        )
        return bool(result)


class TrimeshFclMeshCollisionBackend:
    backend_name = "trimesh_fcl"

    def build_scene(self, mesh: TriangleMeshLike) -> MeshCollisionScene:
        return TrimeshFclMeshCollisionScene(mesh)


class GraspCollisionEvaluator:
    """Collision evaluator that can grow from static grasp checks to trajectories."""

    def __init__(
        self,
        collision_model: FingerBoxGripperCollisionModel,
        backend: MeshCollisionBackend | None = None,
    ) -> None:
        self._collision_model = collision_model
        self._backend = backend or self._default_backend()

    @property
    def backend_name(self) -> str:
        return self._backend.backend_name

    def build_scene(self, mesh: TriangleMeshLike) -> MeshCollisionScene:
        return self._backend.build_scene(mesh)

    def is_grasp_collision_free(
        self,
        *,
        scene: MeshCollisionScene,
        grasp_rotmat: np.ndarray,
        contact_point_a: np.ndarray,
        contact_point_b: np.ndarray,
    ) -> bool:
        for primitive in self._collision_model.primitives_for_grasp(
            grasp_rotmat=grasp_rotmat,
            contact_point_a=contact_point_a,
            contact_point_b=contact_point_b,
        ):
            if scene.intersects_box(primitive):
                return False
        return True

    @staticmethod
    def _default_backend() -> MeshCollisionBackend:
        if trimesh is None or CollisionManager is None:
            raise RuntimeError(
                "trimesh with FCL support is required for mesh grasp collision checks. "
                "Install 'trimesh' and 'python-fcl', and ensure native FCL libraries are available."
            )
        return TrimeshFclMeshCollisionBackend()
