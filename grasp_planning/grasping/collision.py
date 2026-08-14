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


def trimesh_fcl_backend_available() -> bool:
    """Return whether the trimesh collision backend is actually usable."""

    if trimesh is None or CollisionManager is None:
        return False
    try:
        CollisionManager()
    except Exception:
        return False
    return True


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
        grasp_center: np.ndarray | None = None,
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
        grasp_center: np.ndarray | None = None,
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
                grasp_center=grasp_center,
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
_FRANKA_HAND_MESH_CACHE: tuple[np.ndarray, np.ndarray] | None = None
_KUKA_Y_GRIPPER_MESH_DIR = Path(__file__).resolve().parents[2] / "assets" / "urdf" / "kuka_iiwa7_y_gripper" / "meshes"
_KUKA_Y_GRIPPER_MESH_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_KUKA_Y_GRIPPER_TCP_TO_GRASP_CENTER_M = np.array([0.0, 0.0, 0.1505], dtype=float)
_KUKA_Y_GRIPPER_MESH_NAMES = {
    "base": "hand.STL",
    "left_finger": "left_finger.STL",
    "right_finger": "right_finger.STL",
}
_KUKA_Y_FINGERTIP_REFERENCE_MIN_Z_M = 0.08


def _load_franka_hand_mesh() -> tuple[np.ndarray, np.ndarray]:
    global _FRANKA_HAND_MESH_CACHE
    if _FRANKA_HAND_MESH_CACHE is not None:
        return _FRANKA_HAND_MESH_CACHE
    if trimesh is None:
        raise RuntimeError("trimesh is required to load the Franka hand collision mesh.")
    if not _FRANKA_HAND_MESH_PATH.is_file():
        raise FileNotFoundError(f"Franka hand collision mesh not found at '{_FRANKA_HAND_MESH_PATH}'.")
    mesh = trimesh.load(_FRANKA_HAND_MESH_PATH, force="mesh")
    _FRANKA_HAND_MESH_CACHE = (np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64))
    return _FRANKA_HAND_MESH_CACHE


def _load_kuka_y_gripper_mesh(key: str) -> tuple[np.ndarray, np.ndarray]:
    cached = _KUKA_Y_GRIPPER_MESH_CACHE.get(key)
    if cached is not None:
        return cached
    if trimesh is None:
        raise RuntimeError("trimesh is required to load the KUKA Y-gripper collision meshes.")
    mesh_name = _KUKA_Y_GRIPPER_MESH_NAMES[key]
    mesh_path = _KUKA_Y_GRIPPER_MESH_DIR / mesh_name
    if not mesh_path.is_file():
        raise FileNotFoundError(f"KUKA Y-gripper collision mesh not found at '{mesh_path}'.")
    mesh = trimesh.load(mesh_path, force="mesh")
    if key == "base":
        vertices, faces = _convex_hull_mesh(mesh)
    else:
        vertices, faces = _component_convex_hull_mesh(mesh, min_area_fraction=0.05, min_z_extent_m=0.02)
    _KUKA_Y_GRIPPER_MESH_CACHE[key] = (vertices, faces)
    return vertices, faces


def _convex_hull_mesh(mesh) -> tuple[np.ndarray, np.ndarray]:
    hull = mesh.convex_hull
    return np.asarray(hull.vertices, dtype=float) * 0.001, np.asarray(hull.faces, dtype=np.int64)


def _component_convex_hull_mesh(
    mesh,
    *,
    min_area_fraction: float,
    min_z_extent_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    components = mesh.split(only_watertight=False)
    total_area = max(float(mesh.area), 1.0e-12)
    kept = []
    for component in components:
        bounds_m = np.asarray(component.bounds, dtype=float) * 0.001
        z_extent_m = float(bounds_m[1, 2] - bounds_m[0, 2])
        if float(component.area) / total_area >= float(min_area_fraction) and z_extent_m >= float(min_z_extent_m):
            kept.append(component)
    if not kept:
        kept = [component for component in components if float(component.area) / total_area >= float(min_area_fraction)]
    if not kept:
        kept = [mesh]
    vertices_out: list[np.ndarray] = []
    faces_out: list[np.ndarray] = []
    for component in kept:
        hull = component.convex_hull
        offset = sum(len(vertices) for vertices in vertices_out)
        vertices_out.append(np.asarray(hull.vertices, dtype=float) * 0.001)
        faces_out.append(np.asarray(hull.faces, dtype=np.int64) + offset)
    return np.vstack(vertices_out), np.vstack(faces_out)


def _place_kuka_y_left_finger_for_grasp(vertices_local: np.ndarray, half_opening_m: float) -> np.ndarray:
    shifted = np.asarray(vertices_local, dtype=float).copy()
    fingertip_vertices = _kuka_y_fingertip_reference_vertices(shifted)
    inner_y = float(np.max(fingertip_vertices[:, 1]))
    shifted[:, 1] += -float(half_opening_m) - inner_y
    return shifted


def _place_kuka_y_right_finger_for_grasp(vertices_local: np.ndarray, half_opening_m: float) -> np.ndarray:
    shifted = np.asarray(vertices_local, dtype=float).copy()
    fingertip_vertices = _kuka_y_fingertip_reference_vertices(shifted)
    inner_y = float(np.min(fingertip_vertices[:, 1]))
    shifted[:, 1] += float(half_opening_m) - inner_y
    return shifted


def _kuka_y_fingertip_reference_vertices(vertices_local: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices_local, dtype=float)
    high_z = vertices[:, 2] >= _KUKA_Y_FINGERTIP_REFERENCE_MIN_Z_M
    if np.any(high_z):
        return vertices[high_z]
    return vertices


@dataclass(frozen=True)
class FrankaHandFingerCollisionModel:
    """Collision model using Franka finger boxes plus the hand collision mesh."""

    hand_vertices_local: np.ndarray | None = None
    hand_faces: np.ndarray | None = None
    contact_gap_m: float = 0.002
    contact_patch_lateral_offset_m: float = 0.0
    contact_patch_approach_offset_m: float = 0.0

    def __post_init__(self) -> None:
        if self.hand_vertices_local is not None and self.hand_faces is not None:
            object.__setattr__(self, "hand_vertices_local", np.asarray(self.hand_vertices_local, dtype=float))
            object.__setattr__(self, "hand_faces", np.asarray(self.hand_faces, dtype=np.int64))
        object.__setattr__(self, "contact_gap_m", float(self.contact_gap_m))
        object.__setattr__(self, "contact_patch_lateral_offset_m", float(self.contact_patch_lateral_offset_m))
        object.__setattr__(self, "contact_patch_approach_offset_m", float(self.contact_patch_approach_offset_m))

    def primitives_for_grasp(
        self,
        *,
        grasp_rotmat: np.ndarray,
        contact_point_a: np.ndarray,
        contact_point_b: np.ndarray,
        grasp_center: np.ndarray | None = None,
    ) -> tuple[CollisionPrimitive, ...]:
        left_rotmat = np.asarray(grasp_rotmat, dtype=float)
        right_rotmat = left_rotmat @ _rpy_to_rotmat(0.0, 0.0, np.pi)
        closing_axis = left_rotmat[:, 1]
        fingertip_contact_offset_left = np.array(
            [
                self.contact_patch_lateral_offset_m,
                0.0,
                _FRANKA_FINGERTIP_CONTACT_Z_M + self.contact_patch_approach_offset_m,
            ],
            dtype=float,
        )
        fingertip_contact_offset_right = np.array(
            [
                -self.contact_patch_lateral_offset_m,
                0.0,
                _FRANKA_FINGERTIP_CONTACT_Z_M + self.contact_patch_approach_offset_m,
            ],
            dtype=float,
        )
        fingertip_offset_left = left_rotmat @ fingertip_contact_offset_left
        fingertip_offset_right = right_rotmat @ fingertip_contact_offset_right

        left_origin = (
            np.asarray(contact_point_b, dtype=float) - fingertip_offset_left + closing_axis * self.contact_gap_m
        )
        right_origin = (
            np.asarray(contact_point_a, dtype=float) - fingertip_offset_right - closing_axis * self.contact_gap_m
        )
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


@dataclass(frozen=True)
class KukaYGripperCollisionModel:
    """Collision model using the generated KUKA/Y-gripper STL meshes."""

    base_vertices_local: np.ndarray | None = None
    base_faces: np.ndarray | None = None
    left_finger_vertices_local: np.ndarray | None = None
    left_finger_faces: np.ndarray | None = None
    right_finger_vertices_local: np.ndarray | None = None
    right_finger_faces: np.ndarray | None = None
    contact_gap_m: float = 0.002
    contact_patch_lateral_offset_m: float = 0.0
    contact_patch_approach_offset_m: float = 0.0
    tcp_to_grasp_center_m: tuple[float, float, float] = tuple(float(v) for v in _KUKA_Y_GRIPPER_TCP_TO_GRASP_CENTER_M)

    def __post_init__(self) -> None:
        for vertices_field, faces_field in (
            ("base_vertices_local", "base_faces"),
            ("left_finger_vertices_local", "left_finger_faces"),
            ("right_finger_vertices_local", "right_finger_faces"),
        ):
            vertices = getattr(self, vertices_field)
            faces = getattr(self, faces_field)
            if vertices is not None and faces is not None:
                object.__setattr__(self, vertices_field, np.asarray(vertices, dtype=float))
                object.__setattr__(self, faces_field, np.asarray(faces, dtype=np.int64))
        object.__setattr__(self, "contact_gap_m", float(self.contact_gap_m))
        object.__setattr__(self, "contact_patch_lateral_offset_m", float(self.contact_patch_lateral_offset_m))
        object.__setattr__(self, "contact_patch_approach_offset_m", float(self.contact_patch_approach_offset_m))
        tcp = np.asarray(self.tcp_to_grasp_center_m, dtype=float)
        if tcp.shape != (3,):
            raise ValueError("tcp_to_grasp_center_m must contain exactly three values.")
        object.__setattr__(self, "tcp_to_grasp_center_m", tuple(float(v) for v in tcp))

    def primitives_for_grasp(
        self,
        *,
        grasp_rotmat: np.ndarray,
        contact_point_a: np.ndarray,
        contact_point_b: np.ndarray,
        grasp_center: np.ndarray | None = None,
    ) -> tuple[MeshCollisionPrimitive, ...]:
        rotmat = np.asarray(grasp_rotmat, dtype=float)
        contact_a = np.asarray(contact_point_a, dtype=float)
        contact_b = np.asarray(contact_point_b, dtype=float)
        jaw_width = float(np.linalg.norm(contact_b - contact_a))
        half_opening_m = 0.5 * jaw_width + float(self.contact_gap_m)
        if grasp_center is None:
            tcp_center_obj = (
                0.5 * (contact_a + contact_b)
                - rotmat[:, 0] * float(self.contact_patch_lateral_offset_m)
                - rotmat[:, 2] * float(self.contact_patch_approach_offset_m)
            )
        else:
            tcp_center_obj = np.asarray(grasp_center, dtype=float)
        base_origin_obj = tcp_center_obj - rotmat @ np.asarray(self.tcp_to_grasp_center_m, dtype=float)

        base_vertices, base_faces = self._mesh("base")
        left_vertices, left_faces = self._mesh("left_finger")
        right_vertices, right_faces = self._mesh("right_finger")
        left_shifted = _place_kuka_y_left_finger_for_grasp(left_vertices, half_opening_m)
        right_shifted = _place_kuka_y_right_finger_for_grasp(right_vertices, half_opening_m)
        return (
            self._mesh_to_object(
                name="kuka_y_gripper_base",
                vertices_local=base_vertices,
                faces=base_faces,
                origin_obj=base_origin_obj,
                rotmat=rotmat,
            ),
            self._mesh_to_object(
                name="kuka_y_left_finger",
                vertices_local=left_shifted,
                faces=left_faces,
                origin_obj=base_origin_obj,
                rotmat=rotmat,
            ),
            self._mesh_to_object(
                name="kuka_y_right_finger",
                vertices_local=right_shifted,
                faces=right_faces,
                origin_obj=base_origin_obj,
                rotmat=rotmat,
            ),
        )

    def _mesh(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if key == "base" and self.base_vertices_local is not None and self.base_faces is not None:
            return self.base_vertices_local, self.base_faces
        if key == "left_finger" and self.left_finger_vertices_local is not None and self.left_finger_faces is not None:
            return self.left_finger_vertices_local, self.left_finger_faces
        if (
            key == "right_finger"
            and self.right_finger_vertices_local is not None
            and self.right_finger_faces is not None
        ):
            return self.right_finger_vertices_local, self.right_finger_faces
        return _load_kuka_y_gripper_mesh(key)

    @staticmethod
    def _mesh_to_object(
        *,
        name: str,
        vertices_local: np.ndarray,
        faces: np.ndarray,
        origin_obj: np.ndarray,
        rotmat: np.ndarray,
    ) -> MeshCollisionPrimitive:
        vertices = np.asarray(vertices_local, dtype=float)
        return MeshCollisionPrimitive(
            name=name,
            vertices_obj=np.asarray(origin_obj, dtype=float)[None, :] + vertices @ np.asarray(rotmat, dtype=float).T,
            faces=np.asarray(faces, dtype=np.int64),
        )


GripperCollisionModel = (
    FingerBoxGripperCollisionModel
    | FingerBoxWithHandMeshCollisionModel
    | FrankaHandFingerCollisionModel
    | KukaYGripperCollisionModel
)

GRIPPER_COLLISION_MODEL_FRANKA = "franka_hand"
GRIPPER_COLLISION_MODEL_KUKA_Y = "kuka_y_gripper"
SUPPORTED_GRIPPER_COLLISION_MODELS = (GRIPPER_COLLISION_MODEL_FRANKA, GRIPPER_COLLISION_MODEL_KUKA_Y)


def normalize_gripper_collision_model_name(name: str) -> str:
    normalized = str(name or GRIPPER_COLLISION_MODEL_FRANKA).strip().lower().replace("-", "_")
    aliases = {
        "franka": GRIPPER_COLLISION_MODEL_FRANKA,
        "franka_hand": GRIPPER_COLLISION_MODEL_FRANKA,
        "panda": GRIPPER_COLLISION_MODEL_FRANKA,
        "panda_hand": GRIPPER_COLLISION_MODEL_FRANKA,
        "kuka": GRIPPER_COLLISION_MODEL_KUKA_Y,
        "lbr": GRIPPER_COLLISION_MODEL_KUKA_Y,
        "lbr_iiwa7": GRIPPER_COLLISION_MODEL_KUKA_Y,
        "kuka_iiwa7_y_gripper": GRIPPER_COLLISION_MODEL_KUKA_Y,
        "lbr_iiwa7_y_gripper": GRIPPER_COLLISION_MODEL_KUKA_Y,
        "kuka_y_gripper": GRIPPER_COLLISION_MODEL_KUKA_Y,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "Unsupported gripper collision model "
            f"'{name}'. Expected one of: {', '.join(SUPPORTED_GRIPPER_COLLISION_MODELS)}."
        ) from exc


def make_gripper_collision_model(
    name: str = GRIPPER_COLLISION_MODEL_FRANKA,
    *,
    contact_gap_m: float = 0.002,
    contact_patch_lateral_offset_m: float = 0.0,
    contact_patch_approach_offset_m: float = 0.0,
) -> GripperCollisionModel:
    normalized = normalize_gripper_collision_model_name(name)
    if normalized == GRIPPER_COLLISION_MODEL_FRANKA:
        return FrankaHandFingerCollisionModel(
            contact_gap_m=contact_gap_m,
            contact_patch_lateral_offset_m=contact_patch_lateral_offset_m,
            contact_patch_approach_offset_m=contact_patch_approach_offset_m,
        )
    if normalized == GRIPPER_COLLISION_MODEL_KUKA_Y:
        return KukaYGripperCollisionModel(
            contact_gap_m=contact_gap_m,
            contact_patch_lateral_offset_m=contact_patch_lateral_offset_m,
            contact_patch_approach_offset_m=contact_patch_approach_offset_m,
        )
    raise AssertionError(f"Unhandled gripper collision model '{normalized}'.")


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

    def minimum_distance_to_box(
        self,
        primitive: BoxCollisionPrimitive,
    ) -> float: ...

    def minimum_distance_to_mesh(
        self,
        primitive: MeshCollisionPrimitive,
    ) -> float: ...


class MeshCollisionBackend(Protocol):
    """Factory for prepared mesh collision scenes."""

    backend_name: str

    def build_scene(self, mesh: TriangleMeshLike) -> MeshCollisionScene: ...


class TrimeshFclMeshCollisionScene:
    """Mesh collision scene backed by trimesh and FCL."""

    def __init__(self, mesh: TriangleMeshLike) -> None:
        if not trimesh_fcl_backend_available():
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

    def minimum_distance_to_box(self, primitive: BoxCollisionPrimitive) -> float:
        """Return the non-negative separation between the scene and one box."""

        box_mesh = trimesh.creation.box(extents=2.0 * primitive.half_extents)
        return float(
            self._manager.min_distance_single(
                box_mesh,
                transform=primitive.transform_matrix_obj(),
                return_data=False,
            )
        )

    def minimum_distance_to_mesh(self, primitive: MeshCollisionPrimitive) -> float:
        """Return the non-negative separation between the scene and one mesh."""

        mesh = trimesh.Trimesh(
            vertices=primitive.vertices_obj,
            faces=primitive.faces,
            process=False,
        )
        return float(self._manager.min_distance_single(mesh, return_data=False))


class TrimeshFclMeshCollisionBackend:
    backend_name = "trimesh_fcl"

    def build_scene(self, mesh: TriangleMeshLike) -> MeshCollisionScene:
        return TrimeshFclMeshCollisionScene(mesh)


class GraspCollisionEvaluator:
    """Collision evaluator that can grow from static grasp checks to trajectories."""

    def __init__(
        self,
        collision_model: GripperCollisionModel,
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
        grasp_center: np.ndarray | None = None,
    ) -> bool:
        for primitive in self._collision_model.primitives_for_grasp(
            grasp_rotmat=grasp_rotmat,
            contact_point_a=contact_point_a,
            contact_point_b=contact_point_b,
            grasp_center=grasp_center,
        ):
            if isinstance(primitive, BoxCollisionPrimitive) and scene.intersects_box(primitive):
                return False
            if isinstance(primitive, MeshCollisionPrimitive) and scene.intersects_mesh(primitive):
                return False
        return True

    @staticmethod
    def _default_backend() -> MeshCollisionBackend:
        if not trimesh_fcl_backend_available():
            raise RuntimeError(
                "trimesh with FCL support is required for mesh grasp collision checks. "
                "Install 'trimesh' and 'python-fcl', and ensure native FCL libraries are available."
            )
        return TrimeshFclMeshCollisionBackend()
