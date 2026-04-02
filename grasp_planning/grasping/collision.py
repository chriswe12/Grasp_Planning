"""Extensible mesh collision checks for object-frame grasps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
class MeshCollisionPrimitive:
    """A triangle mesh primitive expressed in object coordinates."""

    name: str
    vertices_obj: np.ndarray
    faces: np.ndarray


CollisionPrimitive = BoxCollisionPrimitive | MeshCollisionPrimitive


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


@dataclass(frozen=True)
class FingerBoxWithHandMeshCollisionModel:
    """Collision model using the existing coarse finger boxes plus the Franka hand mesh."""

    finger_extent_lateral: float
    finger_extent_closing: float
    finger_extent_approach: float
    finger_clearance: float
    hand_vertices_local: np.ndarray | None = None
    hand_faces: np.ndarray | None = None
    hand_to_contact_offset_m: float = 58.4e-3 + 45.25e-3

    def __post_init__(self) -> None:
        if self.hand_vertices_local is not None and self.hand_faces is not None:
            object.__setattr__(self, "hand_vertices_local", np.asarray(self.hand_vertices_local, dtype=float))
            object.__setattr__(self, "hand_faces", np.asarray(self.hand_faces, dtype=np.int64))

    def primitives_for_grasp(
        self,
        *,
        grasp_rotmat: np.ndarray,
        contact_point_a: np.ndarray,
        contact_point_b: np.ndarray,
    ) -> tuple[CollisionPrimitive, ...]:
        box_model = FingerBoxGripperCollisionModel(
            finger_extent_lateral=self.finger_extent_lateral,
            finger_extent_closing=self.finger_extent_closing,
            finger_extent_approach=self.finger_extent_approach,
            finger_clearance=self.finger_clearance,
        )
        finger_primitives = list(
            box_model.primitives_for_grasp(
                grasp_rotmat=grasp_rotmat,
                contact_point_a=contact_point_a,
                contact_point_b=contact_point_b,
            )
        )
        hand_origin = 0.5 * (np.asarray(contact_point_a, dtype=float) + np.asarray(contact_point_b, dtype=float)) - (
            np.asarray(grasp_rotmat, dtype=float)[:, 2] * float(self.hand_to_contact_offset_m)
        )
        hand_vertices_local = self.hand_vertices_local
        hand_faces = self.hand_faces
        if hand_vertices_local is None or hand_faces is None:
            hand_vertices_local, hand_faces = _load_franka_hand_mesh()
        hand_vertices_obj = hand_origin[None, :] + hand_vertices_local @ np.asarray(grasp_rotmat, dtype=float).T
        finger_primitives.append(
            MeshCollisionPrimitive(
                name="franka_hand",
                vertices_obj=hand_vertices_obj,
                faces=hand_faces,
            )
        )
        return tuple(finger_primitives)


@dataclass(frozen=True)
class _CollisionBoxSpec:
    name: str
    center_local: np.ndarray
    size_local: np.ndarray
    rpy_local: np.ndarray


def _rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rot_z @ rot_y @ rot_x


_FRANKA_LEFT_FINGER_BOX_SPECS = (
    _CollisionBoxSpec(
        name="left_screw_mount",
        center_local=np.array([0.0, 18.5e-3, 11.0e-3], dtype=float),
        size_local=np.array([22.0e-3, 15.0e-3, 20.0e-3], dtype=float),
        rpy_local=np.zeros(3, dtype=float),
    ),
    _CollisionBoxSpec(
        name="left_carriage_sledge",
        center_local=np.array([0.0, 6.8e-3, 2.2e-3], dtype=float),
        size_local=np.array([22.0e-3, 8.8e-3, 3.8e-3], dtype=float),
        rpy_local=np.zeros(3, dtype=float),
    ),
    _CollisionBoxSpec(
        name="left_diagonal_finger",
        center_local=np.array([0.0, 15.9e-3, 28.35e-3], dtype=float),
        size_local=np.array([17.5e-3, 7.0e-3, 23.5e-3], dtype=float),
        rpy_local=np.array([np.pi / 6.0, 0.0, 0.0], dtype=float),
    ),
    _CollisionBoxSpec(
        name="left_rubber_tip",
        center_local=np.array([0.0, 7.58e-3, 45.25e-3], dtype=float),
        size_local=np.array([17.5e-3, 15.2e-3, 18.5e-3], dtype=float),
        rpy_local=np.zeros(3, dtype=float),
    ),
)

_FRANKA_RIGHT_FINGER_BOX_SPECS = (
    _CollisionBoxSpec(
        name="right_screw_mount",
        center_local=np.array([0.0, 18.5e-3, 11.0e-3], dtype=float),
        size_local=np.array([22.0e-3, 15.0e-3, 20.0e-3], dtype=float),
        rpy_local=np.zeros(3, dtype=float),
    ),
    _CollisionBoxSpec(
        name="right_carriage_sledge",
        center_local=np.array([0.0, 6.8e-3, 2.2e-3], dtype=float),
        size_local=np.array([22.0e-3, 8.8e-3, 3.8e-3], dtype=float),
        rpy_local=np.zeros(3, dtype=float),
    ),
    _CollisionBoxSpec(
        name="right_diagonal_finger",
        center_local=np.array([0.0, 15.9e-3, 28.35e-3], dtype=float),
        size_local=np.array([17.5e-3, 7.0e-3, 23.5e-3], dtype=float),
        rpy_local=np.array([-np.pi / 6.0, 0.0, np.pi], dtype=float),
    ),
    _CollisionBoxSpec(
        name="right_rubber_tip",
        center_local=np.array([0.0, 7.58e-3, 45.25e-3], dtype=float),
        size_local=np.array([17.5e-3, 15.2e-3, 18.5e-3], dtype=float),
        rpy_local=np.zeros(3, dtype=float),
    ),
)

_FRANKA_FINGERTIP_CONTACT_Z_M = 45.25e-3
_FRANKA_HAND_MESH_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "urdf"
    / "franka_description"
    / "meshes"
    / "robot_ee"
    / "franka_hand_black"
    / "collision"
    / "hand.stl"
)


def _load_franka_hand_mesh() -> tuple[np.ndarray, np.ndarray]:
    if trimesh is None:
        raise RuntimeError("trimesh is required to load the Franka hand collision mesh.")
    if not _FRANKA_HAND_MESH_PATH.is_file():
        raise FileNotFoundError(f"Franka hand collision mesh not found at '{_FRANKA_HAND_MESH_PATH}'.")
    mesh = trimesh.load(_FRANKA_HAND_MESH_PATH, force="mesh")
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)


@dataclass(frozen=True)
class FrankaHandFingerCollisionModel:
    """Collision model using Franka finger boxes plus the hand collision mesh."""

    hand_vertices_local: np.ndarray | None = None
    hand_faces: np.ndarray | None = None
    contact_gap_m: float = 0.002

    def __post_init__(self) -> None:
        if self.hand_vertices_local is not None and self.hand_faces is not None:
            object.__setattr__(self, "hand_vertices_local", np.asarray(self.hand_vertices_local, dtype=float))
            object.__setattr__(self, "hand_faces", np.asarray(self.hand_faces, dtype=np.int64))
        object.__setattr__(self, "contact_gap_m", float(self.contact_gap_m))

    def primitives_for_grasp(
        self,
        *,
        grasp_rotmat: np.ndarray,
        contact_point_a: np.ndarray,
        contact_point_b: np.ndarray,
    ) -> tuple[CollisionPrimitive, ...]:
        left_rotmat = np.asarray(grasp_rotmat, dtype=float)
        right_rotmat = left_rotmat @ _rpy_to_rotmat(0.0, 0.0, np.pi)
        closing_axis = left_rotmat[:, 1]
        fingertip_offset_left = left_rotmat @ np.array([0.0, 0.0, _FRANKA_FINGERTIP_CONTACT_Z_M], dtype=float)
        fingertip_offset_right = right_rotmat @ np.array([0.0, 0.0, _FRANKA_FINGERTIP_CONTACT_Z_M], dtype=float)

        left_origin = np.asarray(contact_point_b, dtype=float) - fingertip_offset_left + closing_axis * self.contact_gap_m
        right_origin = np.asarray(contact_point_a, dtype=float) - fingertip_offset_right - closing_axis * self.contact_gap_m
        hand_origin = 0.5 * (left_origin - left_rotmat[:, 2] * 58.4e-3 + right_origin - right_rotmat[:, 2] * 58.4e-3)
        hand_vertices_local = self.hand_vertices_local
        hand_faces = self.hand_faces
        if hand_vertices_local is None or hand_faces is None:
            hand_vertices_local, hand_faces = _load_franka_hand_mesh()
        hand_vertices_obj = hand_origin[None, :] + hand_vertices_local @ left_rotmat.T

        primitives: list[CollisionPrimitive] = [
            MeshCollisionPrimitive(
                name="franka_hand",
                vertices_obj=hand_vertices_obj,
                faces=hand_faces,
            )
        ]
        for origin, rotmat, specs in (
            (left_origin, left_rotmat, _FRANKA_LEFT_FINGER_BOX_SPECS),
            (right_origin, right_rotmat, _FRANKA_RIGHT_FINGER_BOX_SPECS),
        ):
            for spec in specs:
                primitives.append(
                    BoxCollisionPrimitive(
                        name=spec.name,
                        center_obj=origin + rotmat @ spec.center_local,
                        rotation_obj=rotmat @ _rpy_to_rotmat(*spec.rpy_local),
                        half_extents=0.5 * spec.size_local,
                    )
                )
        return tuple(primitives)


class MeshCollisionScene(Protocol):
    """Prepared mesh acceleration structure for primitive queries."""

    def intersects_box(
        self,
        primitive: BoxCollisionPrimitive,
    ) -> bool: ...

    def intersects_mesh(
        self,
        primitive: MeshCollisionPrimitive,
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

    def intersects_mesh(
        self,
        primitive: MeshCollisionPrimitive,
    ) -> bool:
        mesh = trimesh.Trimesh(vertices=primitive.vertices_obj, faces=primitive.faces, process=False)
        result = self._manager.in_collision_single(mesh, return_data=False)
        return bool(result)


class TrimeshFclMeshCollisionBackend:
    backend_name = "trimesh_fcl"

    def build_scene(self, mesh: TriangleMeshLike) -> MeshCollisionScene:
        return TrimeshFclMeshCollisionScene(mesh)


class GraspCollisionEvaluator:
    """Collision evaluator that can grow from static grasp checks to trajectories."""

    def __init__(
        self,
        collision_model: FingerBoxGripperCollisionModel
        | FingerBoxWithHandMeshCollisionModel
        | FrankaHandFingerCollisionModel,
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
            if isinstance(primitive, BoxCollisionPrimitive) and scene.intersects_box(primitive):
                return False
            if isinstance(primitive, MeshCollisionPrimitive) and scene.intersects_mesh(primitive):
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
