"""Shared helpers for Fabrica grasp generation, export, and debug visualization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from .collision import (
    BoxCollisionPrimitive,
    FrankaHandFingerCollisionModel,
    GraspCollisionEvaluator,
    MeshCollisionPrimitive,
)
from .finger_geometry import finger_box_corners
from .mesh_antipodal_grasp_generator import ObjectFrameGraspCandidate, TriangleMesh
from .mesh_io import load_triangle_mesh, relative_mesh_path, resolve_mesh_path
from .world_constraints import ObjectWorldPose, WorldCollisionConstraintEvaluator

try:
    import trimesh
except Exception:  # pragma: no cover - optional dependency path
    trimesh = None


REPO_ROOT = Path(__file__).resolve().parents[2]
FRANKA_HAND_MESH_PATH = (
    REPO_ROOT
    / "assets"
    / "urdf"
    / "franka_description"
    / "meshes"
    / "robot_ee"
    / "franka_hand_black"
    / "collision"
    / "hand.stl"
)
SCHEMA_VERSION = 2
FRANKA_CONTACT_PATCH_LATERAL_SIZE_M = 17.5e-3
FRANKA_CONTACT_PATCH_APPROACH_SIZE_M = 18.5e-3
DEFAULT_CONTACT_GRID_RESOLUTION = 5


@dataclass(frozen=True)
class SavedGraspCandidate:
    grasp_id: str
    grasp_position_obj: tuple[float, float, float]
    grasp_orientation_xyzw_obj: tuple[float, float, float, float]
    contact_point_a_obj: tuple[float, float, float]
    contact_point_b_obj: tuple[float, float, float]
    contact_normal_a_obj: tuple[float, float, float]
    contact_normal_b_obj: tuple[float, float, float]
    jaw_width: float
    roll_angle_rad: float
    contact_patch_lateral_offset_m: float = 0.0
    contact_patch_approach_offset_m: float = 0.0
    score: float | None = None
    score_components: dict[str, float] | None = None

    def to_object_frame_candidate(self) -> ObjectFrameGraspCandidate:
        return ObjectFrameGraspCandidate(
            grasp_position_obj=self.grasp_position_obj,
            grasp_orientation_xyzw_obj=self.grasp_orientation_xyzw_obj,
            contact_point_a_obj=self.contact_point_a_obj,
            contact_point_b_obj=self.contact_point_b_obj,
            contact_normal_a_obj=self.contact_normal_a_obj,
            contact_normal_b_obj=self.contact_normal_b_obj,
            jaw_width=self.jaw_width,
            roll_angle_rad=self.roll_angle_rad,
        )


@dataclass(frozen=True)
class SavedGraspBundle:
    target_mesh_path: str
    mesh_scale: float
    source_frame_origin_obj_world: tuple[float, float, float]
    source_frame_orientation_xyzw_obj_world: tuple[float, float, float, float]
    candidates: tuple[SavedGraspCandidate, ...]
    metadata: dict[str, object]

    @property
    def target_stl_path(self) -> str:
        return self.target_mesh_path

    @property
    def stl_scale(self) -> float:
        return self.mesh_scale

    @property
    def local_frame_origin_world(self) -> tuple[float, float, float]:
        return self.source_frame_origin_obj_world

    @property
    def local_frame_orientation_xyzw_world(self) -> tuple[float, float, float, float]:
        return self.source_frame_orientation_xyzw_obj_world


@dataclass(frozen=True)
class CandidateStatus:
    grasp: SavedGraspCandidate
    status: str
    reason: str


@dataclass(frozen=True)
class PickupPlacementSpec:
    support_face: str
    yaw_deg: float
    xy_world: tuple[float, float]


@dataclass(frozen=True)
class _CollisionBoxSpec:
    name: str
    center_local: tuple[float, float, float]
    size_local: tuple[float, float, float]
    rpy_local: tuple[float, float, float] = (0.0, 0.0, 0.0)


_FRANKA_LEFT_FINGER_BOX_SPECS = (
    _CollisionBoxSpec("screw_mount", (0.0, 18.5e-3, 11.0e-3), (22.0e-3, 15.0e-3, 20.0e-3)),
    _CollisionBoxSpec("carriage_sledge", (0.0, 6.8e-3, 2.2e-3), (22.0e-3, 8.8e-3, 3.8e-3)),
    _CollisionBoxSpec(
        "diagonal_finger", (0.0, 15.9e-3, 28.35e-3), (17.5e-3, 7.0e-3, 23.5e-3), (math.pi / 6.0, 0.0, 0.0)
    ),
    _CollisionBoxSpec("rubber_tip", (0.0, 7.58e-3, 45.25e-3), (17.5e-3, 15.2e-3, 18.5e-3)),
)
_FRANKA_RIGHT_FINGER_BOX_SPECS = (
    _CollisionBoxSpec("screw_mount", (0.0, 18.5e-3, 11.0e-3), (22.0e-3, 15.0e-3, 20.0e-3)),
    _CollisionBoxSpec("carriage_sledge", (0.0, 6.8e-3, 2.2e-3), (22.0e-3, 8.8e-3, 3.8e-3)),
    _CollisionBoxSpec(
        "diagonal_finger", (0.0, 15.9e-3, 28.35e-3), (17.5e-3, 7.0e-3, 23.5e-3), (-math.pi / 6.0, 0.0, math.pi)
    ),
    _CollisionBoxSpec("rubber_tip", (0.0, 7.58e-3, 45.25e-3), (17.5e-3, 15.2e-3, 18.5e-3)),
)
_FRANKA_FINGER_JOINT_Z_M = 58.4e-3
_FRANKA_TIP_CONTACT_Z_M = 45.25e-3
_FRANKA_HAND_MESH_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def fmt_vec(vec: Iterable[float]) -> list[float]:
    return [round(float(value), 6) for value in vec]


def _equally_spaced_offsets(size_m: float, resolution: int) -> tuple[float, ...]:
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    step = float(size_m) / float(resolution + 1)
    half_extent = 0.5 * float(size_m)
    return tuple(float(-half_extent + step * (index + 1)) for index in range(resolution))


DEFAULT_CONTACT_LATERAL_OFFSETS_M = _equally_spaced_offsets(
    FRANKA_CONTACT_PATCH_LATERAL_SIZE_M,
    DEFAULT_CONTACT_GRID_RESOLUTION,
)
DEFAULT_CONTACT_APPROACH_OFFSETS_M = _equally_spaced_offsets(
    FRANKA_CONTACT_PATCH_APPROACH_SIZE_M,
    DEFAULT_CONTACT_GRID_RESOLUTION,
)

DEFAULT_GRASP_SCORING_SIGMA_CENTER_M = 0.01
DEFAULT_GRASP_SCORING_SIGMA_COM_M = 0.02
DEFAULT_GRASP_SCORING_SUPPORT_TARGET = 80
DEFAULT_GRASP_SCORING_CONTACT_RADIUS_M = (
    0.5 * math.hypot(FRANKA_CONTACT_PATCH_LATERAL_SIZE_M, FRANKA_CONTACT_PATCH_APPROACH_SIZE_M) + 0.003
)


def quat_to_rotmat_xyzw(quat_xyzw: tuple[float, float, float, float]) -> np.ndarray:
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


def rotmat_to_quat_xyzw(rotmat: np.ndarray) -> tuple[float, float, float, float]:
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


def rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rot_z @ rot_y @ rot_x


def resolve_stl_path(path: str | Path) -> Path:
    return resolve_mesh_path(path)


def relative_stl_path(path: str | Path) -> str:
    return relative_mesh_path(path)


def resolve_asset_mesh_path(path: str | Path) -> Path:
    return resolve_mesh_path(path)


def relative_asset_mesh_path(path: str | Path) -> str:
    return relative_mesh_path(path)


def load_stl_mesh(path: str | Path, *, scale: float) -> TriangleMesh:
    return load_triangle_mesh(path, scale=scale)


def load_asset_mesh(path: str | Path, *, scale: float) -> TriangleMesh:
    return load_triangle_mesh(path, scale=scale)


def unique_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i0, i1, i2 in np.asarray(faces, dtype=np.int64).tolist():
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            edges.add(tuple(sorted((int(a), int(b)))))
    return sorted(edges)


def mesh_aabb_center(mesh: TriangleMesh) -> np.ndarray:
    return 0.5 * (
        np.asarray(mesh.vertices_obj, dtype=float).min(axis=0) + np.asarray(mesh.vertices_obj, dtype=float).max(axis=0)
    )


def shifted_mesh(mesh: TriangleMesh, offset: np.ndarray) -> TriangleMesh:
    return TriangleMesh(
        vertices_obj=np.asarray(mesh.vertices_obj, dtype=float) + np.asarray(offset, dtype=float),
        faces=np.asarray(mesh.faces, dtype=np.int64),
    )


def combine_triangle_meshes(meshes: list[TriangleMesh]) -> TriangleMesh | None:
    if not meshes:
        return None
    vertices_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    vertex_offset = 0
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices_obj, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        vertices_list.append(vertices)
        faces_list.append(faces + vertex_offset)
        vertex_offset += len(vertices)
    return TriangleMesh(vertices_obj=np.vstack(vertices_list), faces=np.vstack(faces_list))


def canonicalize_target_mesh(global_mesh: TriangleMesh) -> tuple[TriangleMesh, ObjectWorldPose]:
    center_world = mesh_aabb_center(global_mesh)
    local_mesh = shifted_mesh(global_mesh, -center_world)
    return local_mesh, ObjectWorldPose(
        position_world=tuple(float(v) for v in center_world),
        orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
    )


def transform_mesh_to_world(mesh: TriangleMesh, object_pose_world: ObjectWorldPose) -> TriangleMesh:
    rotation = object_pose_world.rotation_world_from_object
    translation = object_pose_world.translation_world
    vertices_world = np.asarray(mesh.vertices_obj, dtype=float) @ rotation.T + translation
    return TriangleMesh(vertices_obj=vertices_world, faces=np.asarray(mesh.faces, dtype=np.int64))


def load_assembly_obstacle_mesh(
    *,
    assembly_glob: str | None,
    target_stl_path: str | Path | None,
    stl_scale: float,
) -> tuple[TriangleMesh | None, tuple[str, ...]]:
    if not assembly_glob:
        return None, ()
    target_resolved = None if target_stl_path is None else resolve_mesh_path(target_stl_path)
    resolved_paths = []
    for path in sorted(REPO_ROOT.joinpath("assets").glob(assembly_glob)):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if target_resolved is not None and resolved == target_resolved:
            continue
        resolved_paths.append(resolved)
    meshes = [load_triangle_mesh(path, scale=stl_scale) for path in resolved_paths]
    return combine_triangle_meshes(meshes), tuple(relative_mesh_path(path) for path in resolved_paths)


def _box_corners_from_pose(
    center_obj: np.ndarray, rotation_obj: np.ndarray, size_xyz: tuple[float, float, float]
) -> np.ndarray:
    half_extents = 0.5 * np.asarray(size_xyz, dtype=float)
    return finger_box_corners(center_obj, rotation_obj, half_extents)


def _load_franka_hand_mesh() -> tuple[np.ndarray, np.ndarray]:
    global _FRANKA_HAND_MESH_CACHE
    if _FRANKA_HAND_MESH_CACHE is not None:
        return _FRANKA_HAND_MESH_CACHE
    if trimesh is None:
        raise RuntimeError("trimesh is required to load the Franka hand collision mesh.")
    mesh = trimesh.load(FRANKA_HAND_MESH_PATH, force="mesh")
    _FRANKA_HAND_MESH_CACHE = (np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64))
    return _FRANKA_HAND_MESH_CACHE


def franka_collision_geometry(
    *,
    grasp_rotmat: np.ndarray,
    grasp_center: np.ndarray,
    contact_point_a: np.ndarray,
    contact_point_b: np.ndarray,
    contact_gap_m: float,
    contact_patch_lateral_offset_m: float = 0.0,
    contact_patch_approach_offset_m: float = 0.0,
) -> dict[str, object]:
    fingertip_contact_offset_left = np.array(
        [contact_patch_lateral_offset_m, 0.0, _FRANKA_TIP_CONTACT_Z_M + contact_patch_approach_offset_m],
        dtype=float,
    )
    fingertip_contact_offset_right = np.array(
        [-contact_patch_lateral_offset_m, 0.0, _FRANKA_TIP_CONTACT_Z_M + contact_patch_approach_offset_m],
        dtype=float,
    )
    closing_axis = np.asarray(grasp_rotmat, dtype=float)[:, 1]
    right_finger_rotmat = grasp_rotmat @ rpy_to_rotmat(0.0, 0.0, math.pi)
    left_finger_origin = (
        np.asarray(contact_point_b, dtype=float)
        - grasp_rotmat @ fingertip_contact_offset_left
        + closing_axis * float(contact_gap_m)
    )
    right_finger_origin = (
        np.asarray(contact_point_a, dtype=float)
        - right_finger_rotmat @ fingertip_contact_offset_right
        - closing_axis * float(contact_gap_m)
    )
    hand_origin_left = left_finger_origin - grasp_rotmat @ np.array([0.0, 0.0, _FRANKA_FINGER_JOINT_Z_M], dtype=float)
    hand_origin_right = right_finger_origin - right_finger_rotmat @ np.array(
        [0.0, 0.0, _FRANKA_FINGER_JOINT_Z_M], dtype=float
    )

    def _boxes_for_finger(
        prefix: str, contact_origin_obj: np.ndarray, base_rotmat: np.ndarray, specs: tuple[_CollisionBoxSpec, ...]
    ) -> list[dict[str, object]]:
        boxes: list[dict[str, object]] = []
        for spec in specs:
            local_rotmat = rpy_to_rotmat(*spec.rpy_local)
            world_rotmat = base_rotmat @ local_rotmat
            center_obj = contact_origin_obj + base_rotmat @ np.asarray(spec.center_local, dtype=float)
            boxes.append(
                {
                    "name": f"{prefix}_{spec.name}",
                    "corners": [
                        fmt_vec(corner.tolist())
                        for corner in _box_corners_from_pose(center_obj, world_rotmat, spec.size_local)
                    ],
                }
            )
        return boxes

    def _contact_grid_points(
        contact_origin_obj: np.ndarray, base_rotmat: np.ndarray, lateral_sign: float
    ) -> list[list[float]]:
        points: list[list[float]] = []
        for lateral_offset_m in DEFAULT_CONTACT_LATERAL_OFFSETS_M:
            for approach_offset_m in DEFAULT_CONTACT_APPROACH_OFFSETS_M:
                local_offset = np.array(
                    [lateral_sign * lateral_offset_m, 0.0, _FRANKA_TIP_CONTACT_Z_M + approach_offset_m],
                    dtype=float,
                )
                point = contact_origin_obj + base_rotmat @ local_offset
                points.append(fmt_vec(point.tolist()))
        return points

    left_tip_anchor = left_finger_origin + grasp_rotmat @ np.array([0.0, 0.0, _FRANKA_TIP_CONTACT_Z_M], dtype=float)
    right_tip_anchor = right_finger_origin + right_finger_rotmat @ np.array(
        [0.0, 0.0, _FRANKA_TIP_CONTACT_Z_M], dtype=float
    )
    hand_origin = 0.5 * (hand_origin_left + hand_origin_right)
    hand_vertices_local, hand_faces = _load_franka_hand_mesh()
    hand_vertices_obj = hand_origin[None, :] + hand_vertices_local @ grasp_rotmat.T
    return {
        "franka_left_boxes": _boxes_for_finger("left", left_finger_origin, grasp_rotmat, _FRANKA_LEFT_FINGER_BOX_SPECS),
        "franka_right_boxes": _boxes_for_finger(
            "right", right_finger_origin, right_finger_rotmat, _FRANKA_RIGHT_FINGER_BOX_SPECS
        ),
        "franka_hand_origin_obj": fmt_vec(hand_origin.tolist()),
        "franka_hand_reference_obj": fmt_vec(np.asarray(grasp_center, dtype=float).tolist()),
        "franka_hand_vertices_obj": [fmt_vec(vertex.tolist()) for vertex in hand_vertices_obj],
        "franka_hand_faces": [[int(v) for v in face] for face in hand_faces.tolist()],
        "franka_left_tip_anchor_obj": fmt_vec(left_tip_anchor.tolist()),
        "franka_right_tip_anchor_obj": fmt_vec(right_tip_anchor.tolist()),
        "franka_left_contact_grid_obj": _contact_grid_points(left_finger_origin, grasp_rotmat, 1.0),
        "franka_right_contact_grid_obj": _contact_grid_points(right_finger_origin, right_finger_rotmat, -1.0),
        "franka_left_anchor_error_m": round(
            float(np.linalg.norm(left_tip_anchor - np.asarray(contact_point_b, dtype=float))), 8
        ),
        "franka_right_anchor_error_m": round(
            float(np.linalg.norm(right_tip_anchor - np.asarray(contact_point_a, dtype=float))), 8
        ),
        "contact_patch_lateral_offset_m": round(float(contact_patch_lateral_offset_m), 6),
        "contact_patch_approach_offset_m": round(float(contact_patch_approach_offset_m), 6),
    }


def serialize_saved_candidate(grasp_id: str, candidate: ObjectFrameGraspCandidate) -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=tuple(float(v) for v in candidate.grasp_position_obj),
        grasp_orientation_xyzw_obj=tuple(float(v) for v in candidate.grasp_orientation_xyzw_obj),
        contact_point_a_obj=tuple(float(v) for v in candidate.contact_point_a_obj),
        contact_point_b_obj=tuple(float(v) for v in candidate.contact_point_b_obj),
        contact_normal_a_obj=tuple(float(v) for v in candidate.contact_normal_a_obj),
        contact_normal_b_obj=tuple(float(v) for v in candidate.contact_normal_b_obj),
        jaw_width=float(candidate.jaw_width),
        roll_angle_rad=float(candidate.roll_angle_rad),
        contact_patch_lateral_offset_m=0.0,
        contact_patch_approach_offset_m=0.0,
    )


def save_grasp_bundle(bundle: SavedGraspBundle, output_path: str | Path) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "target": {
            "mesh_path": bundle.target_mesh_path,
            "mesh_scale": bundle.mesh_scale,
            "source_frame_origin_obj_world": list(bundle.source_frame_origin_obj_world),
            "source_frame_orientation_xyzw_obj_world": list(bundle.source_frame_orientation_xyzw_obj_world),
            "stl_path": bundle.target_mesh_path,
            "stl_scale": bundle.mesh_scale,
            "local_frame_origin_world": list(bundle.source_frame_origin_obj_world),
            "local_frame_orientation_xyzw_world": list(bundle.source_frame_orientation_xyzw_obj_world),
        },
        "metadata": bundle.metadata,
        "candidates": [
            {
                "grasp_id": candidate.grasp_id,
                "grasp_pose_obj": {
                    "position": list(candidate.grasp_position_obj),
                    "orientation_xyzw": list(candidate.grasp_orientation_xyzw_obj),
                },
                "contact_points_obj": [list(candidate.contact_point_a_obj), list(candidate.contact_point_b_obj)],
                "contact_normals_obj": [list(candidate.contact_normal_a_obj), list(candidate.contact_normal_b_obj)],
                "jaw_width": candidate.jaw_width,
                "roll_angle_rad": candidate.roll_angle_rad,
                "contact_patch_offset_local": [
                    candidate.contact_patch_lateral_offset_m,
                    candidate.contact_patch_approach_offset_m,
                ],
                "score": candidate.score,
                "score_components": candidate.score_components,
            }
            for candidate in bundle.candidates
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_grasp_bundle(path: str | Path) -> SavedGraspBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in {1, SCHEMA_VERSION}:
        raise ValueError(f"Unsupported grasp bundle schema version: {payload.get('schema_version')}")
    target = payload["target"]
    candidates = []
    for item in payload["candidates"]:
        contact_patch_offset_local = item.get("contact_patch_offset_local", [0.0, 0.0])
        candidates.append(
            SavedGraspCandidate(
                grasp_id=str(item["grasp_id"]),
                grasp_position_obj=tuple(float(v) for v in item["grasp_pose_obj"]["position"]),
                grasp_orientation_xyzw_obj=tuple(float(v) for v in item["grasp_pose_obj"]["orientation_xyzw"]),
                contact_point_a_obj=tuple(float(v) for v in item["contact_points_obj"][0]),
                contact_point_b_obj=tuple(float(v) for v in item["contact_points_obj"][1]),
                contact_normal_a_obj=tuple(float(v) for v in item["contact_normals_obj"][0]),
                contact_normal_b_obj=tuple(float(v) for v in item["contact_normals_obj"][1]),
                jaw_width=float(item["jaw_width"]),
                roll_angle_rad=float(item["roll_angle_rad"]),
                contact_patch_lateral_offset_m=float(contact_patch_offset_local[0]),
                contact_patch_approach_offset_m=float(contact_patch_offset_local[1]),
                score=None if item.get("score") is None else float(item["score"]),
                score_components=(
                    None
                    if item.get("score_components") is None
                    else {str(k): float(v) for k, v in dict(item["score_components"]).items()}
                ),
            )
        )
    return SavedGraspBundle(
        target_mesh_path=str(target.get("mesh_path", target["stl_path"])),
        mesh_scale=float(target.get("mesh_scale", target["stl_scale"])),
        source_frame_origin_obj_world=tuple(
            float(v) for v in target.get("source_frame_origin_obj_world", target["local_frame_origin_world"])
        ),
        source_frame_orientation_xyzw_obj_world=tuple(
            float(v)
            for v in target.get(
                "source_frame_orientation_xyzw_obj_world",
                target["local_frame_orientation_xyzw_world"],
            )
        ),
        candidates=tuple(candidates),
        metadata=dict(payload.get("metadata", {})),
    )


def object_point_to_world(point_obj: np.ndarray, object_pose_world: ObjectWorldPose) -> np.ndarray:
    return (
        object_pose_world.rotation_world_from_object @ np.asarray(point_obj, dtype=float)
        + object_pose_world.translation_world
    )


def _display_point(
    point: Iterable[float],
    *,
    object_pose_world: ObjectWorldPose | None,
) -> list[float]:
    point_arr = np.asarray(tuple(float(v) for v in point), dtype=float)
    if object_pose_world is None:
        return fmt_vec(point_arr.tolist())
    return fmt_vec(object_point_to_world(point_arr, object_pose_world).tolist())


def world_point_to_object(point_world: np.ndarray, object_pose_world: ObjectWorldPose) -> np.ndarray:
    rotation_world_from_object = object_pose_world.rotation_world_from_object
    return rotation_world_from_object.T @ (np.asarray(point_world, dtype=float) - object_pose_world.translation_world)


def transform_primitive_to_world(
    primitive_obj: BoxCollisionPrimitive | MeshCollisionPrimitive,
    object_pose_world: ObjectWorldPose,
) -> BoxCollisionPrimitive | MeshCollisionPrimitive:
    rotation = object_pose_world.rotation_world_from_object
    translation = object_pose_world.translation_world
    if isinstance(primitive_obj, BoxCollisionPrimitive):
        return BoxCollisionPrimitive(
            name=primitive_obj.name,
            center_obj=rotation @ primitive_obj.center_obj + translation,
            rotation_obj=rotation @ primitive_obj.rotation_obj,
            half_extents=primitive_obj.half_extents,
        )
    return MeshCollisionPrimitive(
        name=primitive_obj.name,
        vertices_obj=primitive_obj.vertices_obj @ rotation.T + translation,
        faces=primitive_obj.faces,
    )


def _candidate_with_contact_offset(
    candidate: SavedGraspCandidate,
    *,
    lateral_offset_m: float,
    approach_offset_m: float,
) -> SavedGraspCandidate:
    grasp_rotmat = quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj)
    previous_offset_obj = grasp_rotmat @ np.array(
        [
            float(candidate.contact_patch_lateral_offset_m),
            0.0,
            float(candidate.contact_patch_approach_offset_m),
        ],
        dtype=float,
    )
    updated_offset_obj = grasp_rotmat @ np.array(
        [float(lateral_offset_m), 0.0, float(approach_offset_m)],
        dtype=float,
    )
    grasp_position_obj = (
        np.asarray(candidate.grasp_position_obj, dtype=float) + previous_offset_obj - updated_offset_obj
    )
    return SavedGraspCandidate(
        grasp_id=candidate.grasp_id,
        grasp_position_obj=tuple(float(v) for v in grasp_position_obj),
        grasp_orientation_xyzw_obj=candidate.grasp_orientation_xyzw_obj,
        contact_point_a_obj=candidate.contact_point_a_obj,
        contact_point_b_obj=candidate.contact_point_b_obj,
        contact_normal_a_obj=candidate.contact_normal_a_obj,
        contact_normal_b_obj=candidate.contact_normal_b_obj,
        jaw_width=candidate.jaw_width,
        roll_angle_rad=candidate.roll_angle_rad,
        contact_patch_lateral_offset_m=float(lateral_offset_m),
        contact_patch_approach_offset_m=float(approach_offset_m),
        score=candidate.score,
        score_components=None if candidate.score_components is None else dict(candidate.score_components),
    )


@dataclass(frozen=True)
class _MeshNeighborhoodIndex:
    vertices_obj: np.ndarray
    vertex_normals_obj: np.ndarray
    tree: cKDTree
    center_of_mass_obj: np.ndarray


def _mesh_vertex_normals(mesh: TriangleMesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices_obj, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.zeros_like(vertices)
    triangles = vertices[faces]
    raw_face_normals = np.cross(triangles[:, 1, :] - triangles[:, 0, :], triangles[:, 2, :] - triangles[:, 0, :])
    for face_index, face in enumerate(faces):
        face_normal = raw_face_normals[face_index]
        for vertex_index in face:
            normals[int(vertex_index)] += face_normal
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-12
    normals[valid] /= lengths[valid][:, None]
    if np.any(~valid):
        normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=float)
    return normals


def _mesh_surface_centroid(mesh: TriangleMesh) -> np.ndarray:
    triangles = np.asarray(mesh.face_vertices, dtype=float)
    raw_normals = np.cross(triangles[:, 1, :] - triangles[:, 0, :], triangles[:, 2, :] - triangles[:, 0, :])
    double_areas = np.linalg.norm(raw_normals, axis=1)
    valid = double_areas > 1.0e-12
    if not np.any(valid):
        return np.asarray(mesh.vertices_obj, dtype=float).mean(axis=0)
    centroids = triangles.mean(axis=1)
    weights = double_areas[valid]
    return np.average(centroids[valid], axis=0, weights=weights)


def _mesh_center_of_mass(mesh: TriangleMesh) -> np.ndarray:
    if trimesh is not None:
        try:
            tri_mesh = trimesh.Trimesh(vertices=mesh.vertices_obj, faces=mesh.faces, process=False)
            center_mass = np.asarray(tri_mesh.center_mass, dtype=float)
            if center_mass.shape == (3,) and np.all(np.isfinite(center_mass)):
                return center_mass
        except Exception:
            pass
    return _mesh_surface_centroid(mesh)


def _build_mesh_neighborhood_index(mesh: TriangleMesh) -> _MeshNeighborhoodIndex:
    vertices = np.asarray(mesh.vertices_obj, dtype=float)
    return _MeshNeighborhoodIndex(
        vertices_obj=vertices,
        vertex_normals_obj=_mesh_vertex_normals(mesh),
        tree=cKDTree(vertices),
        center_of_mass_obj=_mesh_center_of_mass(mesh),
    )


def _contact_neighborhood_indices(
    index: _MeshNeighborhoodIndex,
    contact_point_obj: np.ndarray,
    *,
    radius_m: float,
) -> np.ndarray:
    indices = index.tree.query_ball_point(np.asarray(contact_point_obj, dtype=float), r=float(radius_m))
    if indices:
        return np.asarray(indices, dtype=np.int64)
    _, nearest_index = index.tree.query(np.asarray(contact_point_obj, dtype=float), k=1)
    return np.asarray([int(nearest_index)], dtype=np.int64)


def _project_onto_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=float)
    return np.asarray(vec, dtype=float) - float(np.dot(vec, normal)) * normal


def _grasp_score_components(
    candidate: SavedGraspCandidate,
    *,
    mesh_index: _MeshNeighborhoodIndex,
    sigma_center_m: float = DEFAULT_GRASP_SCORING_SIGMA_CENTER_M,
    sigma_com_m: float = DEFAULT_GRASP_SCORING_SIGMA_COM_M,
    support_target: int = DEFAULT_GRASP_SCORING_SUPPORT_TARGET,
    contact_radius_m: float = DEFAULT_GRASP_SCORING_CONTACT_RADIUS_M,
) -> dict[str, float]:
    grasp_center = np.asarray(candidate.grasp_position_obj, dtype=float)
    contact_right = np.asarray(candidate.contact_point_a_obj, dtype=float)
    contact_left = np.asarray(candidate.contact_point_b_obj, dtype=float)
    normal_right = np.asarray(candidate.contact_normal_a_obj, dtype=float)
    normal_left = np.asarray(candidate.contact_normal_b_obj, dtype=float)
    closing_axis = contact_left - contact_right
    closing_norm = float(np.linalg.norm(closing_axis))
    if closing_norm < 1.0e-12:
        raise ValueError(f"Candidate '{candidate.grasp_id}' has coincident contact points.")
    closing_axis /= closing_norm

    left_alignment = max(0.0, float(np.dot(normal_left, closing_axis)))
    right_alignment = max(0.0, float(np.dot(normal_right, -closing_axis)))
    s_align = 0.5 * (left_alignment + right_alignment)

    contact_midpoint = 0.5 * (contact_left + contact_right)
    center_offset_plane = _project_onto_plane(contact_midpoint - grasp_center, closing_axis)
    d_center = float(np.linalg.norm(center_offset_plane))
    s_center = math.exp(-((d_center * d_center) / (sigma_center_m * sigma_center_m)))

    left_indices = _contact_neighborhood_indices(mesh_index, contact_left, radius_m=contact_radius_m)
    right_indices = _contact_neighborhood_indices(mesh_index, contact_right, radius_m=contact_radius_m)
    n_left = int(left_indices.size)
    n_right = int(right_indices.size)
    s_support = min(1.0, float(n_left + n_right) / float(max(1, support_target)))

    com_offset_plane = _project_onto_plane(mesh_index.center_of_mass_obj - grasp_center, closing_axis)
    d_com = float(np.linalg.norm(com_offset_plane))
    s_com = math.exp(-((d_com * d_com) / (sigma_com_m * sigma_com_m)))

    total = 0.40 * s_align + 0.25 * s_center + 0.20 * s_support + 0.15 * s_com
    total = min(1.0, max(0.0, total))
    return {
        "antipodal_alignment": float(s_align),
        "centering": float(s_center),
        "contact_support": float(s_support),
        "com_offset": float(s_com),
        "contact_count_left": float(n_left),
        "contact_count_right": float(n_right),
        "center_offset_plane_m": float(d_center),
        "com_offset_plane_m": float(d_com),
        "score": float(total),
    }


def score_grasps(
    grasps: Iterable[SavedGraspCandidate],
    *,
    mesh_local: TriangleMesh,
    sigma_center_m: float = DEFAULT_GRASP_SCORING_SIGMA_CENTER_M,
    sigma_com_m: float = DEFAULT_GRASP_SCORING_SIGMA_COM_M,
    support_target: int = DEFAULT_GRASP_SCORING_SUPPORT_TARGET,
    contact_radius_m: float = DEFAULT_GRASP_SCORING_CONTACT_RADIUS_M,
) -> list[SavedGraspCandidate]:
    mesh_index = _build_mesh_neighborhood_index(mesh_local)
    scored: list[SavedGraspCandidate] = []
    for grasp in grasps:
        components = _grasp_score_components(
            grasp,
            mesh_index=mesh_index,
            sigma_center_m=sigma_center_m,
            sigma_com_m=sigma_com_m,
            support_target=support_target,
            contact_radius_m=contact_radius_m,
        )
        scored.append(
            SavedGraspCandidate(
                grasp_id=grasp.grasp_id,
                grasp_position_obj=grasp.grasp_position_obj,
                grasp_orientation_xyzw_obj=grasp.grasp_orientation_xyzw_obj,
                contact_point_a_obj=grasp.contact_point_a_obj,
                contact_point_b_obj=grasp.contact_point_b_obj,
                contact_normal_a_obj=grasp.contact_normal_a_obj,
                contact_normal_b_obj=grasp.contact_normal_b_obj,
                jaw_width=grasp.jaw_width,
                roll_angle_rad=grasp.roll_angle_rad,
                contact_patch_lateral_offset_m=grasp.contact_patch_lateral_offset_m,
                contact_patch_approach_offset_m=grasp.contact_patch_approach_offset_m,
                score=components["score"],
                score_components=components,
            )
        )
    return sorted(
        scored,
        key=lambda candidate: (
            float("-inf") if candidate.score is None else float(candidate.score),
            candidate.grasp_id,
        ),
        reverse=True,
    )


def _ordered_contact_offset_pairs(
    candidate: SavedGraspCandidate,
    *,
    contact_lateral_offsets_m: tuple[float, ...],
    contact_approach_offsets_m: tuple[float, ...],
) -> list[tuple[float, float]]:
    def _lateral_sort_key(value: float) -> tuple[float, int]:
        if abs(value) < 1.0e-12:
            return (0.0, 0)
        return (abs(value), 0 if value > 0.0 else 1)

    ordered_laterals = sorted((float(v) for v in contact_lateral_offsets_m), key=_lateral_sort_key)

    ordered = [(float(candidate.contact_patch_lateral_offset_m), float(candidate.contact_patch_approach_offset_m))]
    for lateral in ordered_laterals:
        for approach in contact_approach_offsets_m:
            pair = (float(lateral), float(approach))
            if pair not in ordered:
                ordered.append(pair)
    return ordered


def _assembly_collision_free_for_offset(
    candidate: SavedGraspCandidate,
    *,
    object_pose_world: ObjectWorldPose,
    obstacle_scene,
    contact_gap_m: float,
    lateral_offset_m: float,
    approach_offset_m: float,
    hand_vertices_local: np.ndarray,
    hand_faces: np.ndarray,
) -> bool:
    candidate_obj = candidate.to_object_frame_candidate()
    grasp_rotmat_obj = quat_to_rotmat_xyzw(candidate_obj.grasp_orientation_xyzw_obj)
    collision_model = FrankaHandFingerCollisionModel(
        hand_vertices_local=hand_vertices_local,
        hand_faces=hand_faces,
        contact_gap_m=contact_gap_m,
        contact_patch_lateral_offset_m=lateral_offset_m,
        contact_patch_approach_offset_m=approach_offset_m,
    )
    for primitive_obj in collision_model.primitives_for_grasp(
        grasp_rotmat=grasp_rotmat_obj,
        contact_point_a=np.asarray(candidate_obj.contact_point_a_obj, dtype=float),
        contact_point_b=np.asarray(candidate_obj.contact_point_b_obj, dtype=float),
    ):
        primitive_world = transform_primitive_to_world(primitive_obj, object_pose_world)
        if isinstance(primitive_world, BoxCollisionPrimitive) and obstacle_scene.intersects_box(primitive_world):
            return False
        if isinstance(primitive_world, MeshCollisionPrimitive) and obstacle_scene.intersects_mesh(primitive_world):
            return False
    return True


def filter_grasps_against_assembly(
    candidates: Iterable[SavedGraspCandidate],
    *,
    object_pose_world: ObjectWorldPose,
    obstacle_mesh_world: TriangleMesh | None,
    contact_gap_m: float,
    contact_lateral_offsets_m: tuple[float, ...] = DEFAULT_CONTACT_LATERAL_OFFSETS_M,
    contact_approach_offsets_m: tuple[float, ...] = DEFAULT_CONTACT_APPROACH_OFFSETS_M,
) -> list[SavedGraspCandidate]:
    if obstacle_mesh_world is None:
        return list(candidates)
    hand_vertices_local, hand_faces = _load_franka_hand_mesh()
    obstacle_scene = GraspCollisionEvaluator(
        FrankaHandFingerCollisionModel(
            hand_vertices_local=hand_vertices_local,
            hand_faces=hand_faces,
            contact_gap_m=contact_gap_m,
        )
    ).build_scene(obstacle_mesh_world)
    kept: list[SavedGraspCandidate] = []
    for candidate in candidates:
        for lateral_offset_m, approach_offset_m in _ordered_contact_offset_pairs(
            candidate,
            contact_lateral_offsets_m=contact_lateral_offsets_m,
            contact_approach_offsets_m=contact_approach_offsets_m,
        ):
            if _assembly_collision_free_for_offset(
                candidate,
                object_pose_world=object_pose_world,
                obstacle_scene=obstacle_scene,
                contact_gap_m=contact_gap_m,
                lateral_offset_m=lateral_offset_m,
                approach_offset_m=approach_offset_m,
                hand_vertices_local=hand_vertices_local,
                hand_faces=hand_faces,
            ):
                kept.append(
                    _candidate_with_contact_offset(
                        candidate,
                        lateral_offset_m=lateral_offset_m,
                        approach_offset_m=approach_offset_m,
                    )
                )
                break
    return kept


def evaluate_grasps_against_ground(
    candidates: Iterable[SavedGraspCandidate],
    *,
    object_pose_world: ObjectWorldPose,
    contact_gap_m: float,
    contact_lateral_offsets_m: tuple[float, ...] = DEFAULT_CONTACT_LATERAL_OFFSETS_M,
    contact_approach_offsets_m: tuple[float, ...] = DEFAULT_CONTACT_APPROACH_OFFSETS_M,
) -> list[CandidateStatus]:
    statuses: list[CandidateStatus] = []
    hand_vertices_local, hand_faces = _load_franka_hand_mesh()
    for candidate in candidates:
        accepted_candidate: SavedGraspCandidate | None = None
        used_refinement = False
        for lateral_offset_m, approach_offset_m in _ordered_contact_offset_pairs(
            candidate,
            contact_lateral_offsets_m=contact_lateral_offsets_m,
            contact_approach_offsets_m=contact_approach_offsets_m,
        ):
            evaluator = WorldCollisionConstraintEvaluator(
                FrankaHandFingerCollisionModel(
                    hand_vertices_local=hand_vertices_local,
                    hand_faces=hand_faces,
                    contact_gap_m=contact_gap_m,
                    contact_patch_lateral_offset_m=lateral_offset_m,
                    contact_patch_approach_offset_m=approach_offset_m,
                )
            )
            object_candidate = candidate.to_object_frame_candidate()
            if evaluator.is_grasp_above_plane(object_candidate, object_pose_world=object_pose_world):
                accepted_candidate = _candidate_with_contact_offset(
                    candidate,
                    lateral_offset_m=lateral_offset_m,
                    approach_offset_m=approach_offset_m,
                )
                used_refinement = (
                    abs(lateral_offset_m - candidate.contact_patch_lateral_offset_m) > 1.0e-9
                    or abs(approach_offset_m - candidate.contact_patch_approach_offset_m) > 1.0e-9
                )
                break
        if accepted_candidate is not None:
            statuses.append(
                CandidateStatus(
                    grasp=accepted_candidate,
                    status="accepted",
                    reason="clear_of_ground_offset" if used_refinement else "clear_of_ground",
                )
            )
        else:
            statuses.append(CandidateStatus(grasp=candidate, status="rejected", reason="ground_collision"))
    return statuses


def build_pickup_pose_world(
    mesh_local: TriangleMesh,
    *,
    support_face: str,
    yaw_deg: float,
    xy_world: tuple[float, float],
) -> ObjectWorldPose:
    return pickup_pose_for_support_face(
        mesh_local,
        support_face=support_face,
        yaw_deg=yaw_deg,
        xy_world=xy_world,
    )


def evaluate_saved_grasps_against_pickup_pose(
    grasps: Iterable[SavedGraspCandidate],
    *,
    object_pose_world: ObjectWorldPose,
    contact_gap_m: float,
    contact_lateral_offsets_m: tuple[float, ...] = DEFAULT_CONTACT_LATERAL_OFFSETS_M,
    contact_approach_offsets_m: tuple[float, ...] = DEFAULT_CONTACT_APPROACH_OFFSETS_M,
) -> list[CandidateStatus]:
    return evaluate_grasps_against_ground(
        grasps,
        object_pose_world=object_pose_world,
        contact_gap_m=contact_gap_m,
        contact_lateral_offsets_m=contact_lateral_offsets_m,
        contact_approach_offsets_m=contact_approach_offsets_m,
    )


def accepted_grasps(statuses: Iterable[CandidateStatus]) -> list[SavedGraspCandidate]:
    return [entry.grasp for entry in statuses if entry.status == "accepted"]


def select_first_feasible_grasp(statuses: Iterable[CandidateStatus]) -> SavedGraspCandidate | None:
    best: SavedGraspCandidate | None = None
    for entry in statuses:
        if entry.status == "accepted":
            if best is None:
                best = entry.grasp
                continue
            best_score = float("-inf") if best.score is None else float(best.score)
            grasp_score = float("-inf") if entry.grasp.score is None else float(entry.grasp.score)
            if grasp_score > best_score:
                best = entry.grasp
    return best


def sample_pickup_placement_spec(
    *,
    rng: np.random.Generator,
    allowed_support_faces: tuple[str, ...],
    allowed_yaw_deg: tuple[float, ...],
    xy_min_world: tuple[float, float],
    xy_max_world: tuple[float, float],
) -> PickupPlacementSpec:
    if not allowed_support_faces:
        raise ValueError("allowed_support_faces must be non-empty.")
    if not allowed_yaw_deg:
        raise ValueError("allowed_yaw_deg must be non-empty.")
    x_min, y_min = [float(v) for v in xy_min_world]
    x_max, y_max = [float(v) for v in xy_max_world]
    if x_max < x_min or y_max < y_min:
        raise ValueError("xy_max_world must be >= xy_min_world component-wise.")
    support_face = str(rng.choice(np.asarray(allowed_support_faces, dtype=object)))
    yaw_deg = float(rng.choice(np.asarray(allowed_yaw_deg, dtype=float)))
    xy_world = (
        float(rng.uniform(x_min, x_max)),
        float(rng.uniform(y_min, y_max)),
    )
    return PickupPlacementSpec(
        support_face=support_face,
        yaw_deg=yaw_deg,
        xy_world=xy_world,
    )


def _rotation_for_support_face(face: str) -> np.ndarray:
    if face == "pos_x":
        return rpy_to_rotmat(0.0, math.pi / 2.0, 0.0)
    if face == "neg_x":
        return rpy_to_rotmat(0.0, -math.pi / 2.0, 0.0)
    if face == "pos_y":
        return rpy_to_rotmat(-math.pi / 2.0, 0.0, 0.0)
    if face == "neg_y":
        return rpy_to_rotmat(math.pi / 2.0, 0.0, 0.0)
    if face == "pos_z":
        return rpy_to_rotmat(math.pi, 0.0, 0.0)
    if face == "neg_z":
        return np.eye(3, dtype=float)
    raise ValueError(f"Unsupported support face '{face}'.")


def pickup_pose_for_support_face(
    mesh_local: TriangleMesh,
    *,
    support_face: str,
    yaw_deg: float = 0.0,
    xy_world: tuple[float, float] = (0.0, 0.0),
) -> ObjectWorldPose:
    base_rot = _rotation_for_support_face(support_face)
    yaw_rot = rpy_to_rotmat(0.0, 0.0, math.radians(float(yaw_deg)))
    rotation = yaw_rot @ base_rot
    rotated_vertices = np.asarray(mesh_local.vertices_obj, dtype=float) @ rotation.T
    min_z = float(rotated_vertices[:, 2].min())
    translation = np.array([float(xy_world[0]), float(xy_world[1]), -min_z], dtype=float)
    return ObjectWorldPose(
        position_world=tuple(float(v) for v in translation),
        orientation_xyzw_world=rotmat_to_quat_xyzw(rotation),
    )


def ground_plane_overlay_obj(
    state_mesh: TriangleMesh,
    *,
    object_pose_world: ObjectWorldPose,
    enabled: bool,
    padding_scale: float = 6.0,
    min_radius_m: float = 0.2,
) -> dict[str, object] | None:
    if not enabled:
        return None
    mins = np.asarray(state_mesh.vertices_obj, dtype=float).min(axis=0)
    maxs = np.asarray(state_mesh.vertices_obj, dtype=float).max(axis=0)
    extents = np.maximum(maxs - mins, 1.0e-3)
    radius = max(0.5 * float(np.max(extents)) * float(padding_scale), float(min_radius_m))
    plane_points_world = np.array(
        [[-radius, -radius, 0.0], [radius, -radius, 0.0], [radius, radius, 0.0], [-radius, radius, 0.0]],
        dtype=float,
    )
    plane_points_obj = np.array(
        [world_point_to_object(point_world, object_pose_world) for point_world in plane_points_world], dtype=float
    )
    return {"corners_obj": [fmt_vec(point.tolist()) for point in plane_points_obj]}


def candidate_payload(
    candidate_statuses: Iterable[CandidateStatus],
    *,
    contact_gap_m: float,
    object_pose_world: ObjectWorldPose | None = None,
) -> list[dict[str, object]]:
    status_list = list(candidate_statuses)
    status_list.sort(
        key=lambda entry: (
            0 if entry.status == "accepted" else 1,
            -(float(entry.grasp.score) if entry.grasp.score is not None else float("-inf")),
            entry.grasp.grasp_id,
        )
    )
    payload: list[dict[str, object]] = []
    for rank, entry in enumerate(status_list, start=1):
        candidate = entry.grasp.to_object_frame_candidate()
        point_a = np.asarray(candidate.contact_point_a_obj, dtype=float)
        point_b = np.asarray(candidate.contact_point_b_obj, dtype=float)
        center = np.asarray(candidate.grasp_position_obj, dtype=float)
        rotation = quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj)
        closing_axis = (point_b - point_a) / np.linalg.norm(point_b - point_a)
        geometry = franka_collision_geometry(
            grasp_rotmat=rotation,
            grasp_center=center,
            contact_point_a=point_a,
            contact_point_b=point_b,
            contact_gap_m=contact_gap_m,
            contact_patch_lateral_offset_m=entry.grasp.contact_patch_lateral_offset_m,
            contact_patch_approach_offset_m=entry.grasp.contact_patch_approach_offset_m,
        )
        payload.append(
            {
                "rank": rank,
                "grasp_id": entry.grasp.grasp_id,
                "status": entry.status,
                "reason": entry.reason,
                "grasp_position_obj": _display_point(candidate.grasp_position_obj, object_pose_world=object_pose_world),
                "grasp_orientation_xyzw_obj": fmt_vec(candidate.grasp_orientation_xyzw_obj),
                "contact_point_a_obj": _display_point(
                    candidate.contact_point_a_obj, object_pose_world=object_pose_world
                ),
                "contact_point_b_obj": _display_point(
                    candidate.contact_point_b_obj, object_pose_world=object_pose_world
                ),
                "contact_normal_a_obj": fmt_vec(candidate.contact_normal_a_obj),
                "contact_normal_b_obj": fmt_vec(candidate.contact_normal_b_obj),
                "jaw_width": round(float(candidate.jaw_width), 6),
                "roll_angle_rad": round(float(candidate.roll_angle_rad), 6),
                "score": None if entry.grasp.score is None else round(float(entry.grasp.score), 6),
                "score_components": entry.grasp.score_components,
                "contact_patch_lateral_offset_m": round(float(entry.grasp.contact_patch_lateral_offset_m), 6),
                "contact_patch_approach_offset_m": round(float(entry.grasp.contact_patch_approach_offset_m), 6),
                "closing_axis_obj": fmt_vec(closing_axis.tolist()),
                "gripper_x_axis_obj": fmt_vec(rotation[:, 0].tolist()),
                "gripper_y_axis_obj": fmt_vec(rotation[:, 1].tolist()),
                "gripper_z_axis_obj": fmt_vec(rotation[:, 2].tolist()),
                **(
                    geometry
                    if object_pose_world is None
                    else {
                        **geometry,
                        "franka_left_boxes": [
                            {
                                "name": box["name"],
                                "corners": [
                                    _display_point(corner, object_pose_world=object_pose_world)
                                    for corner in box["corners"]
                                ],
                            }
                            for box in geometry["franka_left_boxes"]
                        ],
                        "franka_right_boxes": [
                            {
                                "name": box["name"],
                                "corners": [
                                    _display_point(corner, object_pose_world=object_pose_world)
                                    for corner in box["corners"]
                                ],
                            }
                            for box in geometry["franka_right_boxes"]
                        ],
                        "franka_hand_origin_obj": _display_point(
                            geometry["franka_hand_origin_obj"], object_pose_world=object_pose_world
                        ),
                        "franka_hand_reference_obj": _display_point(
                            geometry["franka_hand_reference_obj"], object_pose_world=object_pose_world
                        ),
                        "franka_hand_vertices_obj": [
                            _display_point(vertex, object_pose_world=object_pose_world)
                            for vertex in geometry["franka_hand_vertices_obj"]
                        ],
                        "franka_left_tip_anchor_obj": _display_point(
                            geometry["franka_left_tip_anchor_obj"], object_pose_world=object_pose_world
                        ),
                        "franka_right_tip_anchor_obj": _display_point(
                            geometry["franka_right_tip_anchor_obj"], object_pose_world=object_pose_world
                        ),
                        "franka_left_contact_grid_obj": [
                            _display_point(point, object_pose_world=object_pose_world)
                            for point in geometry["franka_left_contact_grid_obj"]
                        ],
                        "franka_right_contact_grid_obj": [
                            _display_point(point, object_pose_world=object_pose_world)
                            for point in geometry["franka_right_contact_grid_obj"]
                        ],
                    }
                ),
            }
        )
    return payload


def write_debug_html(
    *,
    title: str,
    subtitle: str,
    mesh_local: TriangleMesh,
    candidate_statuses: Iterable[CandidateStatus],
    output_html: str | Path,
    contact_gap_m: float,
    ground_plane: dict[str, object] | None = None,
    obstacle_mesh_local: TriangleMesh | None = None,
    metadata_lines: list[str] | None = None,
    display_object_pose_world: ObjectWorldPose | None = None,
) -> None:
    mesh_vertices_display = (
        [fmt_vec(vertex) for vertex in mesh_local.vertices_obj.tolist()]
        if display_object_pose_world is None
        else [
            fmt_vec(object_point_to_world(vertex, display_object_pose_world).tolist())
            for vertex in np.asarray(mesh_local.vertices_obj, dtype=float)
        ]
    )
    obstacle_vertices_display = (
        []
        if obstacle_mesh_local is None
        else (
            [fmt_vec(vertex) for vertex in obstacle_mesh_local.vertices_obj.tolist()]
            if display_object_pose_world is None
            else [
                fmt_vec(object_point_to_world(vertex, display_object_pose_world).tolist())
                for vertex in np.asarray(obstacle_mesh_local.vertices_obj, dtype=float)
            ]
        )
    )
    ground_plane_display = (
        ground_plane
        if ground_plane is None or display_object_pose_world is None
        else {
            "corners_obj": [
                _display_point(point, object_pose_world=display_object_pose_world)
                for point in ground_plane["corners_obj"]
            ]
        }
    )
    data = {
        "title": title,
        "subtitle": subtitle,
        "vertices_obj": mesh_vertices_display,
        "edges": unique_edges(mesh_local.faces),
        "faces": [[int(v) for v in face] for face in mesh_local.faces.tolist()],
        "obstacle_vertices_obj": obstacle_vertices_display,
        "obstacle_edges": [] if obstacle_mesh_local is None else unique_edges(obstacle_mesh_local.faces),
        "ground_plane_overlay": ground_plane_display,
        "metadata_lines": metadata_lines or [],
        "candidates": candidate_payload(
            candidate_statuses,
            contact_gap_m=contact_gap_m,
            object_pose_world=display_object_pose_world,
        ),
    }
    data_json = json.dumps(data, indent=2)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fabrica Grasp Debug</title>
  <style>
    :root {
      --bg: #f3efe4;
      --panel: #fffaf0;
      --ink: #1e1d1a;
      --accent: #b43f2c;
      --accent-soft: #e8b59f;
      --muted: #6f6a5f;
      --mesh: #4f6b5f;
      --obstacle: #64748b;
      --ground: #2563eb;
      --accepted: #15803d;
      --rejected: #b91c1c;
      --contact-a: #c8452d;
      --contact-b: #1f7c60;
      --franka: #d97706;
      --hand: #8f5a12;
      --axis: #1397a6;
      --line: #d9ceb8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff8e8 0, transparent 30%),
        linear-gradient(135deg, #f7f2e7 0%, #efe7d4 100%);
    }
    .layout { display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: 100vh; }
    .sidebar { border-right: 1px solid var(--line); background: rgba(255,250,240,0.92); padding: 20px 18px; overflow: auto; }
    .title { margin: 0 0 8px; font-size: 28px; line-height: 1.1; }
    .subtitle { margin: 0 0 18px; color: var(--muted); font-size: 14px; line-height: 1.5; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
    button { border: 1px solid var(--line); background: white; color: var(--ink); border-radius: 999px; padding: 10px 14px; font: inherit; cursor: pointer; }
    button:hover { border-color: var(--accent); }
    .list { display: grid; gap: 10px; margin-bottom: 18px; }
    .item { border: 1px solid var(--line); border-radius: 16px; padding: 12px 14px; background: rgba(255,255,255,0.7); cursor: pointer; transition: transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease; }
    .item:hover { transform: translateY(-1px); border-color: var(--accent-soft); box-shadow: 0 8px 18px rgba(85,65,42,0.08); }
    .item.active { border-color: var(--accent); box-shadow: 0 10px 24px rgba(180,63,44,0.18); background: #fff; }
    .item-rank { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
    .item-main { display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px; gap: 10px; }
    .item-label { font-size: 20px; font-weight: 700; }
    .item-score { font-family: "IBM Plex Mono", monospace; font-size: 13px; }
    .item-meta { margin-top: 8px; color: var(--muted); font-size: 13px; font-family: "IBM Plex Mono", monospace; }
    .status.accepted { color: var(--accepted); }
    .status.rejected { color: var(--rejected); }
    .main { padding: 18px; overflow: auto; }
    .cards { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr); gap: 18px; align-items: start; }
    .card { border: 1px solid var(--line); border-radius: 20px; background: rgba(255,250,240,0.88); padding: 16px; box-shadow: 0 14px 32px rgba(72,51,28,0.08); }
    .card h2 { margin: 0 0 12px; font-size: 16px; letter-spacing: 0.03em; text-transform: uppercase; }
    #scene {
      width: 100%;
      height: auto;
      aspect-ratio: 1.25 / 1;
      display: block;
      background:
        radial-gradient(circle at 20% 18%, rgba(255,255,255,0.9), rgba(255,255,255,0.55) 35%, rgba(233,226,208,0.65)),
        linear-gradient(180deg, rgba(255,255,255,0.2), rgba(223,214,194,0.18));
      border-radius: 16px;
    }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; font-size: 13px; color: var(--muted); }
    .legend span { display: inline-flex; align-items: center; gap: 8px; }
    .swatch { width: 14px; height: 14px; border-radius: 999px; display: inline-block; }
    .kv { white-space: pre-wrap; font-family: "IBM Plex Mono", monospace; font-size: 13px; line-height: 1.55; margin: 0; }
    .caption { margin-top: 10px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .cards { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1 id="title" class="title"></h1>
      <p id="subtitle" class="subtitle"></p>
      <div class="controls">
        <button id="prevBtn" type="button">Prev</button>
        <button id="nextBtn" type="button">Next</button>
        <button id="meshModeBtn" type="button">Solid Mesh</button>
        <button id="acceptedOnlyBtn" type="button">Accepted Only: Off</button>
      </div>
      <div id="graspList" class="list"></div>
    </aside>
    <main class="main">
      <div class="cards">
        <section class="card">
          <h2>Object Frame</h2>
          <svg id="scene" viewBox="0 0 960 760"></svg>
          <div class="legend">
            <span><i class="swatch" style="background: var(--mesh)"></i>Target mesh</span>
            <span><i class="swatch" style="background: var(--obstacle)"></i>Assembly obstacles</span>
            <span><i class="swatch" style="background: var(--ground)"></i>Ground plane</span>
            <span><i class="swatch" style="background: var(--accepted)"></i>Accepted</span>
            <span><i class="swatch" style="background: var(--rejected)"></i>Rejected</span>
            <span><i class="swatch" style="background: var(--franka)"></i>Franka finger boxes</span>
            <span><i class="swatch" style="background: var(--hand)"></i>Franka hand mesh</span>
            <span><i class="swatch" style="background: #0f766e"></i>5x5 contact grid</span>
          </div>
          <p class="caption">Left drag rotates, middle drag pans, scroll zooms, and arrow keys switch candidates.</p>
        </section>
        <section class="card">
          <h2>Selection</h2>
          <pre id="details" class="kv"></pre>
        </section>
      </div>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const title = document.getElementById("title");
    const subtitle = document.getElementById("subtitle");
    const graspList = document.getElementById("graspList");
    const scene = document.getElementById("scene");
    const details = document.getElementById("details");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const meshModeBtn = document.getElementById("meshModeBtn");
    const acceptedOnlyBtn = document.getElementById("acceptedOnlyBtn");
    title.textContent = data.title;
    subtitle.textContent = data.subtitle;
    const state = {
      selectedIndex: 0,
      yaw: -0.82,
      pitch: 0.56,
      zoom: 1.0,
      panX: 0,
      panY: 0,
      dragging: false,
      dragMode: "rotate",
      lastPointerX: 0,
      lastPointerY: 0,
      pointerId: null,
      meshRenderMode: "wireframe",
      acceptedOnly: false,
    };
    function visibleCandidates() {
      return state.acceptedOnly ? data.candidates.filter((candidate) => candidate.status === "accepted") : data.candidates;
    }
    const points = [
      ...data.vertices_obj,
      ...data.obstacle_vertices_obj,
      ...(data.ground_plane_overlay ? data.ground_plane_overlay.corners_obj : []),
      ...data.candidates.flatMap((candidate) => [candidate.grasp_position_obj, candidate.contact_point_a_obj, candidate.contact_point_b_obj, ...candidate.franka_hand_vertices_obj, ...candidate.franka_left_boxes.flatMap((box) => box.corners), ...candidate.franka_right_boxes.flatMap((box) => box.corners)]),
    ];
    const bounds = points.reduce((acc, point) => {
      point.forEach((value, axis) => { acc.min[axis] = Math.min(acc.min[axis], value); acc.max[axis] = Math.max(acc.max[axis], value); });
      return acc;
    }, { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
    const center = bounds.min.map((value, axis) => 0.5 * (value + bounds.max[axis]));
    const extent = Math.max(...bounds.max.map((value, axis) => value - bounds.min[axis]), 0.18);
    const baseScale = 520 / extent;
    function rotate(point) {
      const shifted = point.map((value, axis) => value - center[axis]);
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];
      return [x1, cp * y1 - sp * z1, sp * y1 + cp * z1];
    }
    function project(point) {
      const [x, y, z] = rotate(point);
      const scale = baseScale * state.zoom;
      return { x: 480 + state.panX + x * scale, y: 380 + state.panY - y * scale, depth: z };
    }
    function wrapAngle(angle) {
      const tau = Math.PI * 2;
      let wrapped = angle % tau;
      if (wrapped <= -Math.PI) wrapped += tau;
      else if (wrapped > Math.PI) wrapped -= tau;
      return wrapped;
    }
    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }
    function fmtVec(vec) {
      return `(${vec.map((value) => value >= 0 ? `+${value.toFixed(4)}` : value.toFixed(4)).join(", ")})`;
    }
    function addSvg(tag, attrs) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      scene.appendChild(node);
      return node;
    }
    function drawLine(a, b, options = {}) {
      const pa = project(a), pb = project(b);
      addSvg("line", { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, stroke: options.stroke || "#555", "stroke-width": options.strokeWidth || 2, "stroke-opacity": options.opacity ?? 1, "stroke-dasharray": options.dash || "", "marker-end": options.markerEnd || "" });
    }
    function drawPoint(point, options = {}) {
      const p = project(point);
      addSvg("circle", { cx: p.x, cy: p.y, r: options.radius || 6, fill: options.fill || "#000", "fill-opacity": options.opacity ?? 1, stroke: options.stroke || "white", "stroke-width": options.strokeWidth || 2 });
    }
    function drawPolygon(points, options = {}) {
      const projected = points.map((point) => project(point));
      addSvg("polygon", { points: projected.map((point) => `${point.x},${point.y}`).join(" "), fill: options.fill || "none", "fill-opacity": options.fillOpacity ?? 1, stroke: options.stroke || "none", "stroke-width": options.strokeWidth || 1, "stroke-opacity": options.strokeOpacity ?? 1 });
    }
    function drawLabel(point, text, fill, dx = 8, dy = -8) {
      const p = project(point);
      const node = addSvg("text", { x: p.x + dx, y: p.y + dy, fill, "font-size": 15, "font-family": "IBM Plex Mono, monospace", "font-weight": 600 });
      node.textContent = text;
    }
    function drawBox(corners, color) {
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      edges.forEach(([s,e]) => drawLine(corners[s], corners[e], { stroke: color, strokeWidth: 1.8, opacity: 0.8 }));
    }
    function shadeColor(hex, factor) {
      const clean = hex.replace("#", "");
      const value = Number.parseInt(clean, 16);
      const r = (value >> 16) & 255;
      const g = (value >> 8) & 255;
      const b = value & 255;
      const scale = clamp(factor, 0, 1.4);
      return `#${[r,g,b].map((channel) => clamp(Math.round(channel * scale), 0, 255)).map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
    }
    function drawMeshEdges(vertices, edges, stroke, width, opacity) {
      edges.forEach(([start, end]) => drawLine(vertices[start], vertices[end], { stroke, strokeWidth: width, opacity }));
    }
    function drawTargetMesh() {
      if (state.meshRenderMode === "solid") {
        const faces = data.faces.map((face) => {
          const points = face.map((index) => data.vertices_obj[index]);
          const rotated = points.map((point) => rotate(point));
          const edgeA = rotated[1].map((value, axis) => value - rotated[0][axis]);
          const edgeB = rotated[2].map((value, axis) => value - rotated[0][axis]);
          const normal = [
            edgeA[1] * edgeB[2] - edgeA[2] * edgeB[1],
            edgeA[2] * edgeB[0] - edgeA[0] * edgeB[2],
            edgeA[0] * edgeB[1] - edgeA[1] * edgeB[0],
          ];
          const depth = rotated.reduce((sum, point) => sum + point[2], 0) / rotated.length;
          return { points, normal, depth };
        });
        faces
          .filter((face) => face.normal[2] > 0)
          .sort((a, b) => a.depth - b.depth)
          .forEach((face) => {
            const norm = Math.hypot(face.normal[0], face.normal[1], face.normal[2]) || 1;
            const light = 0.45 + 0.55 * (face.normal[2] / norm);
            drawPolygon(face.points, { fill: shadeColor("#4f6b5f", 0.7 + light * 0.45), fillOpacity: 0.92, stroke: "#32453d", strokeWidth: 1.2, strokeOpacity: 0.55 });
          });
        return;
      }
      drawMeshEdges(data.vertices_obj, data.edges, "#4f6b5f", 2.0, 0.8);
    }
    function drawHandMesh(candidate) {
      candidate.franka_hand_faces.forEach((face) => {
        drawLine(candidate.franka_hand_vertices_obj[face[0]], candidate.franka_hand_vertices_obj[face[1]], { stroke: "#8f5a12", strokeWidth: 1.1, opacity: 0.35 });
        drawLine(candidate.franka_hand_vertices_obj[face[1]], candidate.franka_hand_vertices_obj[face[2]], { stroke: "#8f5a12", strokeWidth: 1.1, opacity: 0.35 });
        drawLine(candidate.franka_hand_vertices_obj[face[2]], candidate.franka_hand_vertices_obj[face[0]], { stroke: "#8f5a12", strokeWidth: 1.1, opacity: 0.35 });
      });
    }
    function drawContactGrid(gridPoints, selectedPoint, gridColor, selectedColor) {
      gridPoints.forEach((point) => {
        drawPoint(point, { fill: gridColor, radius: 2.4, opacity: 0.8, stroke: "white", strokeWidth: 0.8 });
      });
      drawPoint(selectedPoint, { fill: selectedColor, radius: 4.2, opacity: 1.0, stroke: "white", strokeWidth: 1.2 });
    }
    function renderList() {
      graspList.replaceChildren();
      visibleCandidates().forEach((candidate, index) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = `item${index === state.selectedIndex ? " active" : ""}`;
        item.innerHTML = `
          <div class="item-rank">${candidate.grasp_id}</div>
          <div class="item-main">
            <div class="item-label status ${candidate.status}">${candidate.status}</div>
            <div class="item-score">score=${candidate.score === null ? "n/a" : candidate.score.toFixed(3)}</div>
          </div>
          <div class="item-meta">reason=${candidate.reason}<br>roll=${candidate.roll_angle_rad.toFixed(3)} w=${candidate.jaw_width.toFixed(4)} center=${fmtVec(candidate.grasp_position_obj)}</div>
        `;
        item.addEventListener("click", () => { state.selectedIndex = index; render(); });
        graspList.appendChild(item);
      });
    }
    function renderScene(candidate) {
      scene.replaceChildren();
      const defs = addSvg("defs", {});
      const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
      marker.setAttribute("id", "arrow");
      marker.setAttribute("markerWidth", "8");
      marker.setAttribute("markerHeight", "8");
      marker.setAttribute("refX", "7");
      marker.setAttribute("refY", "4");
      marker.setAttribute("orient", "auto");
      marker.innerHTML = '<path d="M0,0 L8,4 L0,8 z" fill="currentColor"></path>';
      defs.appendChild(marker);
      if (data.ground_plane_overlay) {
        const corners = data.ground_plane_overlay.corners_obj;
        drawPolygon(corners, { fill: "#2563eb", fillOpacity: 0.16, stroke: "#2563eb", strokeWidth: 2, strokeOpacity: 0.75 });
        for (let i = 0; i < corners.length; i += 1) drawLine(corners[i], corners[(i + 1) % corners.length], { stroke: "#2563eb", strokeWidth: 2, opacity: 0.9, dash: "10 6" });
        drawLabel(corners[0], "z=0 plane", "#2563eb", 10, -8);
      }
      drawMeshEdges(data.obstacle_vertices_obj, data.obstacle_edges, "#64748b", 1.4, 0.45);
      drawTargetMesh();
      candidate.franka_left_boxes.forEach((box) => drawBox(box.corners, "#d97706"));
      candidate.franka_right_boxes.forEach((box) => drawBox(box.corners, "#d97706"));
      drawHandMesh(candidate);
      drawContactGrid(candidate.franka_left_contact_grid_obj, candidate.franka_left_tip_anchor_obj, "#0f766e", "#14b8a6");
      drawContactGrid(candidate.franka_right_contact_grid_obj, candidate.franka_right_tip_anchor_obj, "#0f766e", "#14b8a6");
      const statusColor = candidate.status === "accepted" ? "#15803d" : "#b91c1c";
      drawLine(candidate.contact_point_a_obj, candidate.contact_point_b_obj, { stroke: statusColor, strokeWidth: 3, opacity: 0.9 });
      drawPoint(candidate.grasp_position_obj, { fill: statusColor, radius: 7 });
      drawPoint(candidate.contact_point_a_obj, { fill: "#c8452d", radius: 6 });
      drawPoint(candidate.contact_point_b_obj, { fill: "#1f7c60", radius: 6 });
      drawLabel(candidate.grasp_position_obj, candidate.grasp_id, statusColor);
    }
    function renderDetails(candidate) {
      details.textContent = [
        ...data.metadata_lines,
        `grasp_id:         ${candidate.grasp_id}`,
        `status:           ${candidate.status}`,
        `reason:           ${candidate.reason}`,
        `score:            ${candidate.score === null ? "n/a" : candidate.score.toFixed(6)}`,
        `score_align:      ${candidate.score_components ? candidate.score_components.antipodal_alignment.toFixed(6) : "n/a"}`,
        `score_center:     ${candidate.score_components ? candidate.score_components.centering.toFixed(6) : "n/a"}`,
        `score_support:    ${candidate.score_components ? candidate.score_components.contact_support.toFixed(6) : "n/a"}`,
        `score_com:        ${candidate.score_components ? candidate.score_components.com_offset.toFixed(6) : "n/a"}`,
        `jaw_width:        ${candidate.jaw_width.toFixed(6)} m`,
        `roll_angle_rad:   ${candidate.roll_angle_rad.toFixed(6)}`,
        `contact_offset_x: ${candidate.contact_patch_lateral_offset_m.toFixed(6)} m`,
        `contact_offset_z: ${candidate.contact_patch_approach_offset_m.toFixed(6)} m`,
        `grasp_position:   (${candidate.grasp_position_obj.join(", ")})`,
        `contact_a:        (${candidate.contact_point_a_obj.join(", ")})`,
        `contact_b:        (${candidate.contact_point_b_obj.join(", ")})`,
      ].join("\\n");
    }
    function render() {
      const candidates = visibleCandidates();
      if (candidates.length === 0) {
        details.textContent = [...data.metadata_lines, "No candidates to display."].join("\\n");
        graspList.replaceChildren();
        scene.replaceChildren();
        return;
      }
      if (state.selectedIndex >= candidates.length) {
        state.selectedIndex = 0;
      }
      const candidate = candidates[state.selectedIndex];
      renderList();
      renderScene(candidate);
      renderDetails(candidate);
    }
    window.addEventListener("keydown", (event) => {
      const candidates = visibleCandidates();
      if (candidates.length === 0) return;
      if (event.key === "ArrowUp" || event.key === "ArrowLeft") { event.preventDefault(); state.selectedIndex = (state.selectedIndex - 1 + candidates.length) % candidates.length; render(); }
      if (event.key === "ArrowDown" || event.key === "ArrowRight") { event.preventDefault(); state.selectedIndex = (state.selectedIndex + 1) % candidates.length; render(); }
    });
    prevBtn.addEventListener("click", () => {
      const candidates = visibleCandidates();
      if (candidates.length === 0) return;
      state.selectedIndex = (state.selectedIndex - 1 + candidates.length) % candidates.length;
      render();
    });
    nextBtn.addEventListener("click", () => {
      const candidates = visibleCandidates();
      if (candidates.length === 0) return;
      state.selectedIndex = (state.selectedIndex + 1) % candidates.length;
      render();
    });
    meshModeBtn.addEventListener("click", () => {
      state.meshRenderMode = state.meshRenderMode === "wireframe" ? "solid" : "wireframe";
      meshModeBtn.textContent = state.meshRenderMode === "wireframe" ? "Solid Mesh" : "Wireframe Mesh";
      render();
    });
    acceptedOnlyBtn.addEventListener("click", () => {
      state.acceptedOnly = !state.acceptedOnly;
      state.selectedIndex = 0;
      acceptedOnlyBtn.textContent = `Accepted Only: ${state.acceptedOnly ? "On" : "Off"}`;
      render();
    });
    scene.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 && event.button !== 1) return;
      event.preventDefault();
      state.dragging = true;
      state.dragMode = event.button === 1 ? "pan" : "rotate";
      state.lastPointerX = event.clientX;
      state.lastPointerY = event.clientY;
      state.pointerId = event.pointerId;
      scene.setPointerCapture(event.pointerId);
      scene.style.cursor = state.dragMode === "pan" ? "move" : "grabbing";
    });
    function stopDragging() {
      state.dragging = false;
      state.pointerId = null;
      scene.style.cursor = "grab";
    }
    scene.addEventListener("pointerup", (event) => {
      if (state.pointerId === event.pointerId) stopDragging();
    });
    scene.addEventListener("pointercancel", () => { stopDragging(); });
    scene.addEventListener("pointermove", (event) => {
      if (!state.dragging || (state.pointerId !== null && event.pointerId !== state.pointerId)) return;
      const dx = event.clientX - state.lastPointerX;
      const dy = event.clientY - state.lastPointerY;
      state.lastPointerX = event.clientX;
      state.lastPointerY = event.clientY;
      if (state.dragMode === "pan") {
        state.panX += dx;
        state.panY += dy;
      } else {
        state.yaw = wrapAngle(state.yaw + dx * 0.01);
        state.pitch = wrapAngle(state.pitch - dy * 0.01);
      }
      render();
    });
    scene.addEventListener("wheel", (event) => {
      event.preventDefault();
      const zoomFactor = event.deltaY < 0 ? 1.08 : 1 / 1.08;
      state.zoom = clamp(state.zoom * zoomFactor, 0.35, 4.0);
      render();
    }, { passive: false });
    scene.style.cursor = "grab";
    scene.addEventListener("contextmenu", (event) => event.preventDefault());
    render();
  </script>
</body>
</html>
"""
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html.replace("__DATA_JSON__", data_json), encoding="utf-8")
