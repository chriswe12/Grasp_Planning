"""Generic triangle-mesh IO helpers for grasp-planning assets."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .mesh_antipodal_grasp_generator import TriangleMesh

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_MESH_DIR = REPO_ROOT / "assets"

try:
    import trimesh
except Exception:  # pragma: no cover - optional dependency path
    trimesh = None


def resolve_mesh_path(path: str | Path) -> Path:
    mesh_path = Path(path).expanduser()
    if not mesh_path.is_absolute():
        mesh_path = (DEFAULT_ASSET_MESH_DIR / mesh_path).resolve()
    else:
        mesh_path = mesh_path.resolve()
    return mesh_path


def relative_mesh_path(path: str | Path) -> str:
    resolved = resolve_mesh_path(path)
    try:
        return str(resolved.relative_to(DEFAULT_ASSET_MESH_DIR))
    except ValueError:
        return str(resolved)


def _dedupe_triangle_vertices(triangles: np.ndarray) -> TriangleMesh:
    vertex_map: dict[tuple[float, float, float], int] = {}
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for triangle in np.asarray(triangles, dtype=float):
        face: list[int] = []
        for vertex in triangle:
            key = tuple(float(value) for value in vertex)
            vertex_index = vertex_map.get(key)
            if vertex_index is None:
                vertex_index = len(vertices)
                vertex_map[key] = vertex_index
                vertices.append([float(value) for value in vertex])
            face.append(vertex_index)
        faces.append(face)
    return TriangleMesh(vertices_obj=np.array(vertices, dtype=float), faces=np.array(faces, dtype=np.int64))


def _load_ascii_stl(path: Path, *, scale: float) -> TriangleMesh:
    triangles: list[np.ndarray] = []
    current_vertices: list[list[float]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("vertex"):
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Malformed ASCII STL vertex at line {line_number} in '{path}'.")
            current_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(current_vertices) == 3:
                triangles.append(np.asarray(current_vertices, dtype=float) * scale)
                current_vertices = []
    if current_vertices:
        raise ValueError(f"Incomplete triangle in ASCII STL '{path}'.")
    if not triangles:
        raise ValueError(f"No triangles found in ASCII STL '{path}'.")
    return _dedupe_triangle_vertices(np.stack(triangles, axis=0))


def _load_binary_stl(path: Path, *, scale: float) -> TriangleMesh:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Binary STL '{path}' is too short.")
    triangle_count = struct.unpack_from("<I", data, offset=80)[0]
    expected_size = 84 + triangle_count * 50
    if len(data) != expected_size:
        raise ValueError(
            f"Binary STL '{path}' has size {len(data)} bytes, expected {expected_size} bytes for {triangle_count} triangles."
        )
    triangles = np.empty((triangle_count, 3, 3), dtype=float)
    offset = 84
    for triangle_index in range(triangle_count):
        offset += 12
        vertices = struct.unpack_from("<9f", data, offset=offset)
        triangles[triangle_index, :, :] = np.asarray(vertices, dtype=float).reshape(3, 3) * scale
        offset += 36
        offset += 2
    if triangle_count == 0:
        raise ValueError(f"Binary STL '{path}' does not contain any triangles.")
    return _dedupe_triangle_vertices(triangles)


def _load_obj_mesh(path: Path, *, scale: float) -> TriangleMesh:
    if trimesh is None:
        raise RuntimeError("trimesh is required to load OBJ meshes.")
    mesh = trimesh.load(path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=float) * float(scale)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return TriangleMesh(vertices_obj=vertices, faces=faces)


def load_triangle_mesh(path: str | Path, *, scale: float = 1.0) -> TriangleMesh:
    mesh_path = resolve_mesh_path(path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found at '{mesh_path}'.")
    if scale <= 0.0:
        raise ValueError("scale must be > 0.")
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        return _load_obj_mesh(mesh_path, scale=scale)
    if suffix == ".stl":
        data = mesh_path.read_bytes()
        if len(data) >= 84:
            triangle_count = struct.unpack_from("<I", data, offset=80)[0]
            if len(data) == 84 + triangle_count * 50:
                return _load_binary_stl(mesh_path, scale=scale)
        return _load_ascii_stl(mesh_path, scale=scale)
    raise ValueError(f"Unsupported mesh format '{mesh_path.suffix}' for '{mesh_path}'.")
