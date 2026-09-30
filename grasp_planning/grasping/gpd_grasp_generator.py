"""External GPD-backed grasp proposal generation for object meshes.

This module keeps GPD at the proposal boundary: it samples an object-frame point
cloud from the mesh, calls a GPD-compatible command, then converts returned
grasp frames into the same object-frame candidates used by the rest of the
pipeline.
"""

from __future__ import annotations

import json
import math
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from .collision import GRIPPER_COLLISION_MODEL_FRANKA, GraspCollisionEvaluator, make_gripper_collision_model
from .fabrica_grasp_debug import rotmat_to_quat_xyzw
from .mesh_antipodal_grasp_generator import ObjectFrameGraspCandidate, SurfaceSample, TriangleMesh


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-10:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def _orthonormalize(rotmat: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotmat, dtype=float))
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _triangle_points_from_barycentric(vertices: np.ndarray, barycentric_uv: np.ndarray) -> np.ndarray:
    u = barycentric_uv[:, 0:1]
    v = barycentric_uv[:, 1:2]
    w = 1.0 - u - v
    return u * vertices[:, 0, :] + v * vertices[:, 1, :] + w * vertices[:, 2, :]


def sample_mesh_surface(mesh: TriangleMesh, *, num_samples: int, rng_seed: int) -> tuple[SurfaceSample, ...]:
    """Sample object-frame mesh surface points and outward face normals."""

    if num_samples <= 0:
        return ()
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

    rng = np.random.default_rng(int(rng_seed))
    sampled_face_slots = rng.choice(len(valid_face_indices), size=int(num_samples), replace=True, p=weights)
    barycentric = rng.random((int(num_samples), 2))
    reflected = barycentric.sum(axis=1) > 1.0
    barycentric[reflected] = 1.0 - barycentric[reflected]
    sampled_vertices = valid_face_vertices[sampled_face_slots]
    sampled_points = _triangle_points_from_barycentric(sampled_vertices, barycentric)
    sampled_normals = valid_normals[sampled_face_slots]

    return tuple(
        SurfaceSample(
            point_obj=tuple(float(v) for v in sampled_points[sample_idx]),
            normal_obj=tuple(float(v) for v in sampled_normals[sample_idx]),
            face_index=int(valid_face_indices[sampled_face_slots[sample_idx]]),
        )
        for sample_idx in range(int(num_samples))
    )


def write_ascii_pcd(path: str | Path, samples: Iterable[SurfaceSample]) -> None:
    """Write surface sample points to an ASCII PCD file that GPD can read."""

    sample_list = tuple(samples)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {len(sample_list)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(sample_list)}",
        "DATA ascii",
    ]
    rows = ["{:.9g} {:.9g} {:.9g}".format(*sample.point_obj) for sample in sample_list]
    output.write_text("\n".join([*header, *rows]) + "\n", encoding="utf-8")


def write_normals_csv(path: str | Path, samples: Iterable[SurfaceSample]) -> None:
    """Write one normal per sampled PCD point in GPD's optional CSV format."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = ["{:.9g},{:.9g},{:.9g}".format(*sample.normal_obj) for sample in samples]
    output.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


@dataclass(frozen=True)
class GpdGraspGeneratorConfig:
    """Configuration for invoking an external GPD-compatible grasp source."""

    executable: str = ""
    config_path: str = ""
    command_template: str = ""
    working_dir: str = ""
    output_json: str = ""
    artifact_dir: str = "artifacts/gpd"
    keep_artifacts: bool = False
    timeout_s: float = 120.0
    num_pointcloud_samples: int = 2048
    min_jaw_width: float = 0.002
    max_jaw_width: float = 0.09
    detailed_finger_contact_gap_m: float = 0.002
    rng_seed: int = 0
    check_target_collision: bool = True
    gripper_collision_model: str = GRIPPER_COLLISION_MODEL_FRANKA


class ExternalGpdGraspGenerator:
    """Generate object-frame grasp candidates with an external GPD command."""

    def __init__(self, config: GpdGraspGeneratorConfig) -> None:
        self._config = config
        self._last_surface_samples: tuple[SurfaceSample, ...] = ()
        self._collision_evaluator = (
            GraspCollisionEvaluator(
                make_gripper_collision_model(
                    self._config.gripper_collision_model, contact_gap_m=self._config.detailed_finger_contact_gap_m
                )
            )
            if self._config.check_target_collision
            else None
        )

    @property
    def collision_backend_name(self) -> str:
        if self._collision_evaluator is None:
            return "gpd_external_no_target_collision"
        return self._collision_evaluator.backend_name

    @property
    def last_surface_samples(self) -> tuple[SurfaceSample, ...]:
        return self._last_surface_samples

    def generate(self, mesh: TriangleMesh) -> list[ObjectFrameGraspCandidate]:
        surface_samples = sample_mesh_surface(
            mesh,
            num_samples=self._config.num_pointcloud_samples,
            rng_seed=self._config.rng_seed,
        )
        self._last_surface_samples = surface_samples
        raw_payload = self._run_gpd(surface_samples)
        candidates = self._candidates_from_payload(raw_payload, mesh=mesh)
        if self._collision_evaluator is None:
            return candidates
        collision_scene = self._collision_evaluator.build_scene(mesh)
        return [
            candidate
            for candidate in candidates
            if self._collision_evaluator.is_grasp_collision_free(
                scene=collision_scene,
                grasp_rotmat=self._rotmat_from_candidate(candidate),
                contact_point_a=np.asarray(candidate.contact_point_a_obj, dtype=float),
                contact_point_b=np.asarray(candidate.contact_point_b_obj, dtype=float),
            )
        ]

    def _artifact_paths(self, temp_dir: Path) -> tuple[Path, Path, Path]:
        if self._config.keep_artifacts or self._config.output_json:
            artifact_dir = Path(self._config.artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            pcd_path = artifact_dir / "gpd_stage1_pointcloud.pcd"
            normals_path = artifact_dir / "gpd_stage1_normals.csv"
            output_path = (
                Path(self._config.output_json) if self._config.output_json else artifact_dir / "gpd_grasps.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return pcd_path, normals_path, output_path
        return temp_dir / "gpd_stage1_pointcloud.pcd", temp_dir / "gpd_stage1_normals.csv", temp_dir / "gpd_grasps.json"

    def _command(self, *, pcd_path: Path, normals_path: Path, output_json_path: Path) -> list[str]:
        values = {
            "executable": self._config.executable,
            "config": self._config.config_path,
            "pcd": str(pcd_path),
            "normals": str(normals_path),
            "output_json": str(output_json_path),
        }
        if self._config.command_template:
            return shlex.split(self._config.command_template.format(**values))
        if not self._config.executable:
            raise ValueError("planning.grasp_generator='gpd' requires either gpd.command_template or gpd.executable.")
        command = [self._config.executable]
        if self._config.config_path:
            command.append(self._config.config_path)
        command.extend([str(pcd_path), str(normals_path)])
        return command

    def _run_gpd(self, surface_samples: tuple[SurfaceSample, ...]) -> object:
        with tempfile.TemporaryDirectory(prefix="grasp_gpd_") as tmp:
            temp_dir = Path(tmp)
            pcd_path, normals_path, output_json_path = self._artifact_paths(temp_dir)
            write_ascii_pcd(pcd_path, surface_samples)
            write_normals_csv(normals_path, surface_samples)
            if output_json_path.exists():
                output_json_path.unlink()
            command = self._command(
                pcd_path=pcd_path.resolve(),
                normals_path=normals_path.resolve(),
                output_json_path=output_json_path.resolve(),
            )
            completed = subprocess.run(
                command,
                cwd=None if not self._config.working_dir else self._config.working_dir,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=float(self._config.timeout_s),
            )
            if output_json_path.is_file() and output_json_path.stat().st_size > 0:
                return json.loads(output_json_path.read_text(encoding="utf-8"))
            try:
                return json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "GPD command did not write a JSON output file or JSON stdout. "
                    "Use a wrapper/patched GPD binary that emits a grasp JSON payload."
                ) from exc

    def _candidates_from_payload(self, payload: object, *, mesh: TriangleMesh) -> list[ObjectFrameGraspCandidate]:
        items = _extract_grasp_items(payload)
        mesh_index = _MeshContactIndex.from_mesh(mesh)
        candidates: list[ObjectFrameGraspCandidate] = []
        seen_keys: set[tuple[float, ...]] = set()
        for item in items:
            candidate = self._candidate_from_item(item, mesh_index=mesh_index)
            if candidate is None:
                continue
            key = (
                *np.round(np.asarray(candidate.grasp_position_obj, dtype=float), 6).tolist(),
                *np.round(np.asarray(candidate.grasp_orientation_xyzw_obj, dtype=float), 6).tolist(),
                round(float(candidate.jaw_width), 6),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(candidate)
        return candidates

    def _candidate_from_item(
        self,
        item: dict[str, object],
        *,
        mesh_index: "_MeshContactIndex",
    ) -> ObjectFrameGraspCandidate | None:
        rotmat = _rotmat_from_gpd_item(item)
        center = _vector_from_item(item, "center", "position", "top", "surface")
        if center is None:
            pose = item.get("grasp_pose_obj")
            if isinstance(pose, dict):
                center = _vector_from_item(pose, "position")
        if center is None:
            raise ValueError("GPD grasp item is missing a center/position/top/surface vector.")

        jaw_width = _float_from_item(item, "jaw_width", "width", "grasp_width", "aperture")
        if jaw_width is None:
            raise ValueError("GPD grasp item is missing jaw width.")
        if jaw_width < self._config.min_jaw_width or jaw_width > self._config.max_jaw_width:
            return None

        contact_points = _contact_points_from_item(item)
        contact_normals = _contact_normals_from_item(item)
        closing_axis = _normalize(rotmat[:, 1])
        if contact_points is None:
            center_arr = np.asarray(center, dtype=float)
            right_guess = center_arr - 0.5 * float(jaw_width) * closing_axis
            left_guess = center_arr + 0.5 * float(jaw_width) * closing_axis
            contact_a, normal_a = mesh_index.nearest_contact(right_guess)
            contact_b, normal_b = mesh_index.nearest_contact(left_guess)
        else:
            contact_a = np.asarray(contact_points[0], dtype=float)
            contact_b = np.asarray(contact_points[1], dtype=float)
            if contact_normals is None:
                _, normal_a = mesh_index.nearest_contact(contact_a)
                _, normal_b = mesh_index.nearest_contact(contact_b)
            else:
                normal_a = np.asarray(contact_normals[0], dtype=float)
                normal_b = np.asarray(contact_normals[1], dtype=float)

        contact_a, contact_b, normal_a, normal_b = _canonicalize_contacts(
            contact_a=contact_a,
            contact_b=contact_b,
            normal_a=normal_a,
            normal_b=normal_b,
            closing_axis=closing_axis,
        )
        normal_a = _normal_facing(normal_a, -closing_axis)
        normal_b = _normal_facing(normal_b, closing_axis)
        width_from_contacts = float(np.linalg.norm(contact_b - contact_a))
        if width_from_contacts > 1.0e-8:
            jaw_width = width_from_contacts
        if jaw_width < self._config.min_jaw_width or jaw_width > self._config.max_jaw_width:
            return None

        center_arr = 0.5 * (contact_a + contact_b)
        return ObjectFrameGraspCandidate(
            grasp_position_obj=tuple(float(v) for v in center_arr),
            grasp_orientation_xyzw_obj=rotmat_to_quat_xyzw(rotmat),
            contact_point_a_obj=tuple(float(v) for v in contact_a),
            contact_point_b_obj=tuple(float(v) for v in contact_b),
            contact_normal_a_obj=tuple(float(v) for v in _normalize(normal_a)),
            contact_normal_b_obj=tuple(float(v) for v in _normalize(normal_b)),
            jaw_width=float(jaw_width),
            roll_angle_rad=0.0,
        )

    @staticmethod
    def _rotmat_from_candidate(candidate: ObjectFrameGraspCandidate) -> np.ndarray:
        x, y, z, w = [float(v) for v in candidate.grasp_orientation_xyzw_obj]
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
class _MeshContactIndex:
    vertices_obj: np.ndarray
    normals_obj: np.ndarray
    tree: cKDTree

    @classmethod
    def from_mesh(cls, mesh: TriangleMesh) -> "_MeshContactIndex":
        vertices = np.asarray(mesh.vertices_obj, dtype=float)
        return cls(vertices_obj=vertices, normals_obj=_mesh_vertex_normals(mesh), tree=cKDTree(vertices))

    def nearest_contact(self, point_obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, index = self.tree.query(np.asarray(point_obj, dtype=float), k=1)
        return self.vertices_obj[int(index)].copy(), self.normals_obj[int(index)].copy()


def _mesh_vertex_normals(mesh: TriangleMesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices_obj, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.zeros_like(vertices)
    triangles = vertices[faces]
    raw_face_normals = np.cross(triangles[:, 1, :] - triangles[:, 0, :], triangles[:, 2, :] - triangles[:, 0, :])
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            normals[int(vertex_index)] += raw_face_normals[face_index]
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-12
    normals[valid] /= lengths[valid][:, None]
    if np.any(~valid):
        normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=float)
    return normals


def _extract_grasp_items(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("grasps", "candidates", "selected_grasps"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
        else:
            raise ValueError("GPD JSON payload must contain a grasps/candidates/selected_grasps list.")
    else:
        raise ValueError("GPD JSON payload must be a list or mapping.")
    if not all(isinstance(item, dict) for item in items):
        raise ValueError("Each GPD grasp item must be a mapping.")
    return [dict(item) for item in items]


def _vector_from_item(item: dict[str, object], *keys: str) -> np.ndarray | None:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"GPD field '{key}' must contain exactly 3 values.")
        return arr
    return None


def _float_from_item(item: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        if item.get(key) is None:
            continue
        return float(item[key])
    return None


def _rotmat_from_gpd_item(item: dict[str, object]) -> np.ndarray:
    pose = item.get("grasp_pose_obj")
    if isinstance(pose, dict) and pose.get("orientation_xyzw") is not None:
        return _rotmat_from_quat_xyzw(np.asarray(pose["orientation_xyzw"], dtype=float))
    if item.get("orientation_xyzw") is not None:
        return _rotmat_from_quat_xyzw(np.asarray(item["orientation_xyzw"], dtype=float))
    if item.get("rotation_matrix") is not None:
        rotmat = np.asarray(item["rotation_matrix"], dtype=float)
        if rotmat.shape != (3, 3):
            raise ValueError("GPD rotation_matrix must have shape (3, 3).")
        return _orthonormalize(rotmat)

    approach = _vector_from_item(item, "approach")
    binormal = _vector_from_item(item, "binormal")
    axis = _vector_from_item(item, "axis")
    if approach is None or binormal is None or axis is None:
        raise ValueError(
            "GPD grasp item must provide orientation_xyzw, rotation_matrix, or approach/binormal/axis vectors."
        )
    gpd_rotmat = _orthonormalize(np.column_stack((_normalize(approach), _normalize(binormal), _normalize(axis))))
    approach_axis = gpd_rotmat[:, 0]
    closing_axis = gpd_rotmat[:, 1]
    lateral_axis = _normalize(np.cross(closing_axis, approach_axis))
    return _orthonormalize(np.column_stack((lateral_axis, closing_axis, approach_axis)))


def _rotmat_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    if quat_xyzw.shape != (4,):
        raise ValueError("orientation_xyzw must contain exactly 4 values.")
    x, y, z, w = [float(v) for v in quat_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-12:
        raise ValueError("orientation_xyzw has near-zero norm.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _contact_points_from_item(item: dict[str, object]) -> tuple[np.ndarray, np.ndarray] | None:
    value = item.get("contact_points_obj") or item.get("contact_points")
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2, 3):
        raise ValueError("contact_points/contact_points_obj must have shape (2, 3).")
    return arr[0], arr[1]


def _contact_normals_from_item(item: dict[str, object]) -> tuple[np.ndarray, np.ndarray] | None:
    value = item.get("contact_normals_obj") or item.get("contact_normals")
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2, 3):
        raise ValueError("contact_normals/contact_normals_obj must have shape (2, 3).")
    return arr[0], arr[1]


def _canonicalize_contacts(
    *,
    contact_a: np.ndarray,
    contact_b: np.ndarray,
    normal_a: np.ndarray,
    normal_b: np.ndarray,
    closing_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis_from_contacts = contact_b - contact_a
    if np.linalg.norm(axis_from_contacts) < 1.0e-10:
        return contact_a, contact_b, normal_a, normal_b
    if float(np.dot(axis_from_contacts, closing_axis)) < 0.0:
        return contact_b, contact_a, normal_b, normal_a
    return contact_a, contact_b, normal_a, normal_b


def _normal_facing(normal: np.ndarray, direction: np.ndarray) -> np.ndarray:
    normal = _normalize(normal)
    direction = _normalize(direction)
    if float(np.dot(normal, direction)) < 0.0:
        return -normal
    return normal
