"""Generate a browser-based HTML debug view for mesh antipodal grasps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping import (  # noqa: E402
    AntipodalGraspGeneratorConfig,
    AntipodalMeshGraspGenerator,
    ObjectFrameGraspCandidate,
    TriangleMesh,
)


DEFAULT_OUTPUT_HTML = REPO_ROOT / "artifacts" / "mesh_antipodal_grasp_debug.html"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "mesh_antipodal_grasp_debug.yaml"
DEFAULT_STL_DIR = REPO_ROOT / "assets" / "stl"

_DEFAULT_CONFIG = {
    "geometry": {
        "type": "cube",
        "cube_side": 0.05,
        "cylinder_radius": 0.02,
        "cylinder_height": 0.05,
        "cylinder_segments": 24,
        "stl_path": None,
        "stl_scale": 1.0,
    },
    "generator": {
        "num_samples": 192,
        "min_jaw_width": 0.02,
        "max_jaw_width": 0.08,
        "antipodal_cosine_threshold": 0.94,
        "roll_angles_rad": [0.0],
        "max_pair_checks": 4096,
        "finger_depth": 0.008,
        "finger_length": 0.012,
        "finger_thickness": 0.01,
        "finger_clearance": 0.002,
        "contact_patch_radius": 0.006,
        "collision_sample_count": 256,
        "rng_seed": 0,
    },
    "output_html": str(DEFAULT_OUTPUT_HTML),
}


def _quat_to_rotmat_xyzw(quat_xyzw: tuple[float, float, float, float]) -> np.ndarray:
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


def _parse_rolls(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one roll angle.")
    return values


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: '{path}'.")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file '{path}' must contain a YAML mapping at the top level.")
    return loaded


def _coerce_roll_angles(value: object) -> tuple[float, ...]:
    if isinstance(value, str):
        return _parse_rolls(value)
    if isinstance(value, (list, tuple)):
        values = tuple(float(item) for item in value)
        if not values:
            raise ValueError("roll_angles_rad must contain at least one value.")
        return values
    raise ValueError("roll_angles_rad must be a comma-separated string or a YAML list of numbers.")


def _config_from_sources(args: argparse.Namespace) -> argparse.Namespace:
    config_path = args.config.expanduser().resolve()
    merged = _deep_merge(_DEFAULT_CONFIG, _load_yaml_config(config_path))

    geometry = merged.get("geometry")
    generator = merged.get("generator")
    if not isinstance(geometry, dict) or not isinstance(generator, dict):
        raise ValueError("Config must define 'geometry' and 'generator' mappings.")

    resolved = argparse.Namespace(
        config=config_path,
        geometry=str(geometry.get("type", _DEFAULT_CONFIG["geometry"]["type"])),
        cube_side=float(geometry.get("cube_side", _DEFAULT_CONFIG["geometry"]["cube_side"])),
        cylinder_radius=float(geometry.get("cylinder_radius", _DEFAULT_CONFIG["geometry"]["cylinder_radius"])),
        cylinder_height=float(geometry.get("cylinder_height", _DEFAULT_CONFIG["geometry"]["cylinder_height"])),
        cylinder_segments=int(geometry.get("cylinder_segments", _DEFAULT_CONFIG["geometry"]["cylinder_segments"])),
        stl_path=None if geometry.get("stl_path") in (None, "") else Path(str(geometry["stl_path"])),
        stl_scale=float(geometry.get("stl_scale", _DEFAULT_CONFIG["geometry"]["stl_scale"])),
        num_samples=int(generator.get("num_samples", _DEFAULT_CONFIG["generator"]["num_samples"])),
        min_jaw_width=float(generator.get("min_jaw_width", _DEFAULT_CONFIG["generator"]["min_jaw_width"])),
        max_jaw_width=float(generator.get("max_jaw_width", _DEFAULT_CONFIG["generator"]["max_jaw_width"])),
        antipodal_cosine_threshold=float(
            generator.get(
                "antipodal_cosine_threshold",
                _DEFAULT_CONFIG["generator"]["antipodal_cosine_threshold"],
            )
        ),
        roll_angles_rad=_coerce_roll_angles(generator.get("roll_angles_rad", _DEFAULT_CONFIG["generator"]["roll_angles_rad"])),
        max_pair_checks=int(generator.get("max_pair_checks", _DEFAULT_CONFIG["generator"]["max_pair_checks"])),
        finger_depth=float(generator.get("finger_depth", _DEFAULT_CONFIG["generator"]["finger_depth"])),
        finger_length=float(generator.get("finger_length", _DEFAULT_CONFIG["generator"]["finger_length"])),
        finger_thickness=float(generator.get("finger_thickness", _DEFAULT_CONFIG["generator"]["finger_thickness"])),
        finger_clearance=float(generator.get("finger_clearance", _DEFAULT_CONFIG["generator"]["finger_clearance"])),
        contact_patch_radius=float(
            generator.get("contact_patch_radius", _DEFAULT_CONFIG["generator"]["contact_patch_radius"])
        ),
        collision_sample_count=int(
            generator.get("collision_sample_count", _DEFAULT_CONFIG["generator"]["collision_sample_count"])
        ),
        rng_seed=int(generator.get("rng_seed", _DEFAULT_CONFIG["generator"]["rng_seed"])),
        output_html=Path(str(merged.get("output_html", _DEFAULT_CONFIG["output_html"]))),
    )

    for name in (
        "geometry",
        "cube_side",
        "cylinder_radius",
        "cylinder_height",
        "cylinder_segments",
        "stl_path",
        "stl_scale",
        "num_samples",
        "min_jaw_width",
        "max_jaw_width",
        "antipodal_cosine_threshold",
        "roll_angles_rad",
        "max_pair_checks",
        "finger_depth",
        "finger_length",
        "finger_thickness",
        "finger_clearance",
        "contact_patch_radius",
        "collision_sample_count",
        "rng_seed",
        "output_html",
    ):
        override = getattr(args, name)
        if override is not None:
            setattr(resolved, name, override)

    if resolved.geometry not in {"cube", "cylinder", "stl"}:
        raise ValueError(f"Unsupported geometry type '{resolved.geometry}'. Expected cube, cylinder, or stl.")
    return resolved


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

    return TriangleMesh(
        vertices_obj=np.array(vertices, dtype=float),
        faces=np.array(faces, dtype=np.int64),
    )


def _load_ascii_stl(path: Path, *, scale: float) -> TriangleMesh:
    triangles: list[np.ndarray] = []
    current_vertices: list[list[float]] = []

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("vertex"):
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
        # Skip the stored facet normal; the grasp generator recomputes normals from winding.
        offset += 12
        vertices = struct.unpack_from("<9f", data, offset=offset)
        triangles[triangle_index, :, :] = np.asarray(vertices, dtype=float).reshape(3, 3) * scale
        offset += 36
        offset += 2

    if triangle_count == 0:
        raise ValueError(f"Binary STL '{path}' does not contain any triangles.")
    return _dedupe_triangle_vertices(triangles)


def _load_stl_mesh(path: Path, *, scale: float) -> TriangleMesh:
    stl_path = path.expanduser()
    if not stl_path.is_absolute():
        stl_path = (DEFAULT_STL_DIR / stl_path).resolve()
    else:
        stl_path = stl_path.resolve()
    if not stl_path.is_file():
        raise FileNotFoundError(
            f"STL file not found at '{stl_path}'. Place it under '{DEFAULT_STL_DIR}' or pass an absolute path."
        )
    if scale <= 0.0:
        raise ValueError("--stl-scale must be > 0.")

    data = stl_path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, offset=80)[0]
        if len(data) == 84 + triangle_count * 50:
            return _load_binary_stl(stl_path, scale=scale)
    return _load_ascii_stl(stl_path, scale=scale)


def _make_cube_mesh(side_length: float) -> TriangleMesh:
    half = 0.5 * float(side_length)
    vertices = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return TriangleMesh(vertices_obj=vertices, faces=faces)


def _make_cylinder_mesh(radius: float, height: float, radial_segments: int) -> TriangleMesh:
    if radial_segments < 3:
        raise ValueError("radial_segments must be at least 3.")
    half_height = 0.5 * float(height)
    angles = np.linspace(0.0, 2.0 * np.pi, num=radial_segments, endpoint=False)

    vertices: list[list[float]] = []
    for z in (-half_height, half_height):
        for angle in angles:
            vertices.append([float(radius * np.cos(angle)), float(radius * np.sin(angle)), z])
    bottom_center_index = len(vertices)
    vertices.append([0.0, 0.0, -half_height])
    top_center_index = len(vertices)
    vertices.append([0.0, 0.0, half_height])

    faces: list[list[int]] = []
    for idx in range(radial_segments):
        next_idx = (idx + 1) % radial_segments
        bottom_a = idx
        bottom_b = next_idx
        top_a = radial_segments + idx
        top_b = radial_segments + next_idx
        faces.append([bottom_a, bottom_b, top_b])
        faces.append([bottom_a, top_b, top_a])
        faces.append([bottom_center_index, bottom_b, bottom_a])
        faces.append([top_center_index, top_a, top_b])

    return TriangleMesh(vertices_obj=np.array(vertices, dtype=float), faces=np.array(faces, dtype=np.int64))


def _build_mesh(args: argparse.Namespace) -> TriangleMesh:
    if args.geometry == "stl":
        if not args.stl_path:
            raise ValueError("--stl-path is required when --geometry stl.")
        return _load_stl_mesh(args.stl_path, scale=args.stl_scale)
    if args.geometry == "cube":
        return _make_cube_mesh(side_length=args.cube_side)
    return _make_cylinder_mesh(radius=args.cylinder_radius, height=args.cylinder_height, radial_segments=args.cylinder_segments)


def _unique_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i0, i1, i2 in faces.tolist():
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            edge = tuple(sorted((int(a), int(b))))
            edges.add(edge)
    return sorted(edges)


def _finger_box_corners(
    candidate: ObjectFrameGraspCandidate,
    *,
    depth: float,
    length: float,
    thickness: float,
    clearance: float,
) -> tuple[np.ndarray, np.ndarray]:
    point_a = np.asarray(candidate.contact_point_a_obj, dtype=float)
    point_b = np.asarray(candidate.contact_point_b_obj, dtype=float)
    normal_a = np.asarray(candidate.contact_normal_a_obj, dtype=float)
    normal_b = np.asarray(candidate.contact_normal_b_obj, dtype=float)
    closing_axis = (point_b - point_a) / np.linalg.norm(point_b - point_a)

    def build(contact_point: np.ndarray, contact_normal: np.ndarray, invert_closing_axis: bool) -> np.ndarray:
        finger_x = contact_normal / np.linalg.norm(contact_normal)
        finger_y = -closing_axis if invert_closing_axis else closing_axis
        finger_y = finger_y - float(np.dot(finger_y, finger_x)) * finger_x
        finger_y /= np.linalg.norm(finger_y)
        finger_z = np.cross(finger_x, finger_y)
        finger_z /= np.linalg.norm(finger_z)
        rotation = np.column_stack((finger_x, finger_y, finger_z))
        half_extents = 0.5 * np.array([depth, length, thickness], dtype=float)
        center = contact_point + finger_x * (0.5 * clearance)
        signs = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        )
        local_corners = signs * half_extents[None, :]
        return center[None, :] + local_corners @ rotation.T

    return build(point_a, normal_a, False), build(point_b, normal_b, True)


parser = argparse.ArgumentParser(description="Generate an HTML debug view for object-frame antipodal grasps.")
parser.add_argument(
    "--config",
    type=Path,
    default=DEFAULT_CONFIG_PATH,
    help=f"YAML config path. Default: {DEFAULT_CONFIG_PATH}",
)
parser.add_argument(
    "--geometry",
    choices=("cube", "cylinder", "stl"),
    default=None,
    help="Mesh source: procedural cube/cylinder or an STL file.",
)
parser.add_argument("--cube-side", type=float, default=None, help="Cube side length in meters.")
parser.add_argument("--cylinder-radius", type=float, default=None, help="Cylinder radius in meters.")
parser.add_argument("--cylinder-height", type=float, default=None, help="Cylinder height in meters.")
parser.add_argument("--cylinder-segments", type=int, default=None, help="Cylinder radial segment count.")
parser.add_argument(
    "--stl-path",
    type=Path,
    default=None,
    help=f"Path to an STL file. Relative paths are resolved under {DEFAULT_STL_DIR}.",
)
parser.add_argument(
    "--stl-scale",
    type=float,
    default=None,
    help="Uniform scale applied to STL vertices after loading. Use this to convert units to meters if needed.",
)
parser.add_argument("--num-samples", type=int, default=None, help="Number of surface samples.")
parser.add_argument("--min-jaw-width", type=float, default=None, help="Minimum jaw width in meters.")
parser.add_argument("--max-jaw-width", type=float, default=None, help="Maximum jaw width in meters.")
parser.add_argument("--antipodal-cosine-threshold", type=float, default=None, help="Minimum cosine alignment for antipodal normals.")
parser.add_argument("--roll-angles-rad", type=_parse_rolls, default=None, help="Comma-separated roll angles in radians.")
parser.add_argument("--max-pair-checks", type=int, default=None, help="Maximum contact pairs to evaluate.")
parser.add_argument("--finger-depth", type=float, default=None, help="Finger box depth in meters.")
parser.add_argument("--finger-length", type=float, default=None, help="Finger box length along the closing axis.")
parser.add_argument("--finger-thickness", type=float, default=None, help="Finger box thickness in meters.")
parser.add_argument("--finger-clearance", type=float, default=None, help="Contact-plane finger clearance in meters.")
parser.add_argument("--contact-patch-radius", type=float, default=None, help="Ignored contact neighborhood radius.")
parser.add_argument("--collision-sample-count", type=int, default=None, help="Auxiliary surface samples for coarse clearance checks.")
parser.add_argument("--rng-seed", type=int, default=None, help="Random seed for deterministic sampling.")
parser.add_argument("--output-html", type=Path, default=None, help=f"Output HTML path. Default from YAML: {DEFAULT_OUTPUT_HTML}")


@dataclass(frozen=True)
class _ViewerState:
    mesh: TriangleMesh
    edges: list[tuple[int, int]]
    candidates: list[ObjectFrameGraspCandidate]
    config: AntipodalGraspGeneratorConfig
    geometry_name: str


def _fmt_vec(vec: tuple[float, ...] | list[float]) -> list[float]:
    return [round(float(value), 6) for value in vec]


def _build_payload(state: _ViewerState) -> dict[str, object]:
    vertices = [[round(float(v), 6) for v in vertex] for vertex in state.mesh.vertices_obj]
    candidates = []
    for index, candidate in enumerate(state.candidates, start=1):
        point_a = np.asarray(candidate.contact_point_a_obj, dtype=float)
        point_b = np.asarray(candidate.contact_point_b_obj, dtype=float)
        center = np.asarray(candidate.grasp_position_obj, dtype=float)
        normal_a = np.asarray(candidate.contact_normal_a_obj, dtype=float)
        normal_b = np.asarray(candidate.contact_normal_b_obj, dtype=float)
        closing_axis = (point_b - point_a) / np.linalg.norm(point_b - point_a)
        rotation = _quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj)
        finger_box_a, finger_box_b = _finger_box_corners(
            candidate,
            depth=state.config.finger_depth,
            length=state.config.finger_length,
            thickness=state.config.finger_thickness,
            clearance=state.config.finger_clearance,
        )
        candidates.append(
            {
                "rank": index,
                "grasp_position_obj": _fmt_vec(candidate.grasp_position_obj),
                "grasp_orientation_xyzw_obj": _fmt_vec(candidate.grasp_orientation_xyzw_obj),
                "contact_point_a_obj": _fmt_vec(candidate.contact_point_a_obj),
                "contact_point_b_obj": _fmt_vec(candidate.contact_point_b_obj),
                "contact_normal_a_obj": _fmt_vec(candidate.contact_normal_a_obj),
                "contact_normal_b_obj": _fmt_vec(candidate.contact_normal_b_obj),
                "jaw_width": round(float(candidate.jaw_width), 6),
                "roll_angle_rad": round(float(candidate.roll_angle_rad), 6),
                "closing_axis_obj": _fmt_vec(closing_axis.tolist()),
                "gripper_x_axis_obj": _fmt_vec(rotation[:, 0].tolist()),
                "gripper_y_axis_obj": _fmt_vec(rotation[:, 1].tolist()),
                "gripper_z_axis_obj": _fmt_vec(rotation[:, 2].tolist()),
                "finger_box_a": [_fmt_vec(corner.tolist()) for corner in finger_box_a],
                "finger_box_b": [_fmt_vec(corner.tolist()) for corner in finger_box_b],
                "contact_midpoint_error": round(float(np.linalg.norm(center - 0.5 * (point_a + point_b))), 8),
            }
        )

    return {
        "geometry_name": state.geometry_name,
        "vertices_obj": vertices,
        "edges": state.edges,
        "faces": [[int(v) for v in face] for face in state.mesh.faces.tolist()],
        "triangle_count": int(len(state.mesh.faces)),
        "vertex_count": int(len(state.mesh.vertices_obj)),
        "candidate_count": int(len(state.candidates)),
        "config": {
            "num_surface_samples": state.config.num_surface_samples,
            "min_jaw_width": state.config.min_jaw_width,
            "max_jaw_width": state.config.max_jaw_width,
            "antipodal_cosine_threshold": state.config.antipodal_cosine_threshold,
            "roll_angles_rad": [float(v) for v in state.config.roll_angles_rad],
            "finger_depth": state.config.finger_depth,
            "finger_length": state.config.finger_length,
            "finger_thickness": state.config.finger_thickness,
            "finger_clearance": state.config.finger_clearance,
        },
        "candidates": candidates,
    }


def _html_document(payload: dict[str, object]) -> str:
    data_json = json.dumps(payload, indent=2)
    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mesh Antipodal Grasp Debug</title>
  <style>
    :root {{
      --bg: #f3efe4;
      --panel: #fffaf0;
      --ink: #1e1d1a;
      --accent: #b43f2c;
      --accent-soft: #e8b59f;
      --muted: #6f6a5f;
      --line: #d9ceb8;
      --mesh: #4f6b5f;
      --contact-a: #c8452d;
      --contact-b: #1f7c60;
      --box: #6d3cc6;
      --axis: #1397a6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff8e8 0, transparent 30%),
        linear-gradient(135deg, #f7f2e7 0%, #efe7d4 100%);
    }}
    .layout {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid var(--line);
      background: rgba(255, 250, 240, 0.92);
      padding: 20px 18px;
      overflow: auto;
    }}
    .title {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.1;
    }}
    .subtitle {{
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
    }}
    button {{
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 999px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{
      border-color: var(--accent);
    }}
    .list {{
      display: grid;
      gap: 10px;
      margin-bottom: 18px;
    }}
    .item {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.7);
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
    }}
    .item:hover {{
      transform: translateY(-1px);
      border-color: var(--accent-soft);
      box-shadow: 0 8px 18px rgba(85, 65, 42, 0.08);
    }}
    .item.active {{
      border-color: var(--accent);
      box-shadow: 0 10px 24px rgba(180, 63, 44, 0.18);
      background: #fff;
    }}
    .item-rank {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .item-main {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-top: 6px;
      gap: 10px;
    }}
    .item-label {{
      font-size: 22px;
      font-weight: 700;
    }}
    .item-score {{
      font-family: "IBM Plex Mono", monospace;
      font-size: 14px;
    }}
    .item-meta {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      font-family: "IBM Plex Mono", monospace;
    }}
    .main {{
      padding: 18px;
      overflow: auto;
    }}
    .cards {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
      gap: 18px;
      align-items: start;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(255, 250, 240, 0.88);
      padding: 16px;
      box-shadow: 0 14px 32px rgba(72, 51, 28, 0.08);
    }}
    .card h2 {{
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    #scene {{
      width: 100%;
      height: auto;
      aspect-ratio: 1.25 / 1;
      display: block;
      background:
        radial-gradient(circle at 20% 18%, rgba(255,255,255,0.9), rgba(255,255,255,0.55) 35%, rgba(233,226,208,0.65)),
        linear-gradient(180deg, rgba(255,255,255,0.2), rgba(223,214,194,0.18));
      border-radius: 16px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 12px;
      font-size: 13px;
      color: var(--muted);
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      display: inline-block;
    }}
    .details {{
      display: grid;
      gap: 14px;
    }}
    .kv {{
      font-family: "IBM Plex Mono", monospace;
      font-size: 13px;
      line-height: 1.55;
      white-space: pre-wrap;
      margin: 0;
    }}
    .caption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    @media (max-width: 1100px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      .cards {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1 class="title">Mesh Antipodal Grasp Debug</h1>
      <p class="subtitle">
        Object-frame antipodal grasp candidates generated from procedural object geometry only.
        Select a candidate to inspect contacts, normals, gripper axes, and the coarse finger boxes.
      </p>
      <div class="controls">
        <button id="prevBtn" type="button">Prev</button>
        <button id="nextBtn" type="button">Next</button>
        <button id="meshModeBtn" type="button">Solid Mesh</button>
      </div>
      <div id="graspList" class="list"></div>
    </aside>
    <main class="main">
      <div class="cards">
        <section class="card">
          <h2>Object Frame</h2>
          <svg id="scene" viewBox="0 0 960 760" aria-label="Mesh antipodal grasp debug scene"></svg>
          <div class="legend">
            <span><i class="swatch" style="background: var(--mesh)"></i>Mesh wireframe</span>
            <span><i class="swatch" style="background: var(--accent)"></i>Grasp center</span>
            <span><i class="swatch" style="background: var(--contact-a)"></i>Contact A / normal</span>
            <span><i class="swatch" style="background: var(--contact-b)"></i>Contact B / normal</span>
            <span><i class="swatch" style="background: var(--box)"></i>Finger boxes</span>
            <span><i class="swatch" style="background: var(--axis)"></i>Gripper frame axes</span>
          </div>
          <p class="caption">
            The viewer uses the same coarse finger-box geometry as the object-only clearance filter.
          </p>
        </section>
        <section class="card">
          <h2>Selection</h2>
          <div class="details">
            <div>
              <h2 style="margin-bottom:10px;">Summary</h2>
              <pre id="summary" class="kv"></pre>
            </div>
            <div>
              <h2 style="margin-bottom:10px;">Geometry</h2>
              <pre id="geometry" class="kv"></pre>
            </div>
            <div>
              <h2 style="margin-bottom:10px;">Generator</h2>
              <pre id="config" class="kv"></pre>
            </div>
          </div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const scene = document.getElementById("scene");
    const graspList = document.getElementById("graspList");
    const summary = document.getElementById("summary");
    const geometry = document.getElementById("geometry");
    const configView = document.getElementById("config");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const meshModeBtn = document.getElementById("meshModeBtn");

    const state = {{
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
    }};

    const objectPoints = [
      ...data.vertices_obj,
      ...data.candidates.flatMap((candidate) => [
        candidate.grasp_position_obj,
        candidate.contact_point_a_obj,
        candidate.contact_point_b_obj,
        ...candidate.finger_box_a,
        ...candidate.finger_box_b,
      ]),
    ];

    const bounds = objectPoints.reduce((acc, point) => {{
      point.forEach((value, axis) => {{
        acc.min[axis] = Math.min(acc.min[axis], value);
        acc.max[axis] = Math.max(acc.max[axis], value);
      }});
      return acc;
    }}, {{ min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] }});
    const center = bounds.min.map((value, axis) => 0.5 * (value + bounds.max[axis]));
    const extent = Math.max(...bounds.max.map((value, axis) => value - bounds.min[axis]), 0.18);
    const baseScale = 520 / extent;

    function rotate(point) {{
      const shifted = point.map((value, axis) => value - center[axis]);
      const cy = Math.cos(state.yaw);
      const sy = Math.sin(state.yaw);
      const cp = Math.cos(state.pitch);
      const sp = Math.sin(state.pitch);
      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];
      const x2 = x1;
      const y2 = cp * y1 - sp * z1;
      const z2 = sp * y1 + cp * z1;
      return [x2, y2, z2];
    }}

    function project(point) {{
      const [x, y, z] = rotate(point);
      const scale = baseScale * state.zoom;
      return {{
        x: 480 + state.panX + x * scale,
        y: 380 + state.panY - y * scale,
        depth: z,
      }};
    }}

    function wrapAngle(angle) {{
      const tau = Math.PI * 2;
      let wrapped = angle % tau;
      if (wrapped <= -Math.PI) {{
        wrapped += tau;
      }} else if (wrapped > Math.PI) {{
        wrapped -= tau;
      }}
      return wrapped;
    }}

    function fmtVec(vec) {{
      return `(${vec.map((value) => value >= 0 ? `+${value.toFixed(4)}` : value.toFixed(4)).join(", ")})`;
    }}

    function addSvg(tag, attrs) {{
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      scene.appendChild(node);
      return node;
    }}

    function drawLine(a, b, options = {{}}) {{
      const pa = project(a);
      const pb = project(b);
      addSvg("line", {{
        x1: pa.x,
        y1: pa.y,
        x2: pb.x,
        y2: pb.y,
        stroke: options.stroke || "#555",
        "stroke-width": options.strokeWidth || 2,
        "stroke-opacity": options.opacity ?? 1,
        "stroke-dasharray": options.dash || "",
        "marker-end": options.markerEnd || "",
      }});
    }}

    function shadeColor(hex, factor) {{
      const clean = hex.replace("#", "");
      const value = Number.parseInt(clean, 16);
      const r = (value >> 16) & 255;
      const g = (value >> 8) & 255;
      const b = value & 255;
      const scale = clamp(factor, 0, 1.4);
      const next = [r, g, b]
        .map((channel) => clamp(Math.round(channel * scale), 0, 255))
        .map((channel) => channel.toString(16).padStart(2, "0"))
        .join("");
      return `#${next}`;
    }}

    function drawPolygon(points, options = {{}}) {{
      const projected = points.map((point) => project(point));
      addSvg("polygon", {{
        points: projected.map((point) => `${point.x},${point.y}`).join(" "),
        fill: options.fill || "none",
        "fill-opacity": options.fillOpacity ?? 1,
        stroke: options.stroke || "none",
        "stroke-width": options.strokeWidth || 1,
        "stroke-opacity": options.strokeOpacity ?? 1,
      }});
    }}

    function drawPoint(point, options = {{}}) {{
      const p = project(point);
      addSvg("circle", {{
        cx: p.x,
        cy: p.y,
        r: options.radius || 6,
        fill: options.fill || "#000",
        "fill-opacity": options.opacity ?? 1,
        stroke: options.stroke || "white",
        "stroke-width": options.strokeWidth || 2,
      }});
    }}

    function drawLabel(point, text, fill, dx = 10, dy = -10) {{
      const p = project(point);
      const node = addSvg("text", {{
        x: p.x + dx,
        y: p.y + dy,
        fill,
        "font-size": 16,
        "font-family": "IBM Plex Mono, monospace",
        "font-weight": 600,
      }});
      node.textContent = text;
    }}

    function drawArrow(origin, vector, length, color, width, label = null, dx = 8, dy = -8) {{
      const target = origin.map((value, axis) => value + vector[axis] * length);
      drawLine(origin, target, {{ stroke: color, strokeWidth: width, markerEnd: "url(#arrow)" }});
      if (label) {{
        drawLabel(target, label, color, dx, dy);
      }}
    }}

    function drawFingerBox(corners, color) {{
      const edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
      ];
      edges.forEach(([start, end]) => {{
        drawLine(corners[start], corners[end], {{ stroke: color, strokeWidth: 2, opacity: 0.8 }});
      }});
    }}

    function drawMesh() {{
      if (state.meshRenderMode === "solid") {{
        const faces = data.faces.map((face) => {{
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
          return {{ points, normal, depth }};
        }});
        faces
          .filter((face) => face.normal[2] > 0)
          .sort((a, b) => a.depth - b.depth)
          .forEach((face) => {{
            const norm = Math.hypot(face.normal[0], face.normal[1], face.normal[2]) || 1;
            const light = 0.45 + 0.55 * (face.normal[2] / norm);
            drawPolygon(face.points, {{
              fill: shadeColor("#4f6b5f", 0.7 + light * 0.45),
              fillOpacity: 0.92,
              stroke: "#32453d",
              strokeWidth: 1.2,
              strokeOpacity: 0.55,
            }});
          }});
        return;
      }}

      data.edges.forEach(([start, end]) => {{
        drawLine(data.vertices_obj[start], data.vertices_obj[end], {{ stroke: "#4f6b5f", strokeWidth: 2, opacity: 0.75 }});
      }});
    }}

    function renderList() {{
      graspList.replaceChildren();
      data.candidates.forEach((candidate, index) => {{
        const item = document.createElement("button");
        item.type = "button";
        item.className = `item${index === state.selectedIndex ? " active" : ""}`;
        item.innerHTML = `
          <div class="item-rank">Candidate ${candidate.rank}</div>
          <div class="item-main">
            <div class="item-label">w=${candidate.jaw_width.toFixed(4)}</div>
            <div class="item-score">roll=${candidate.roll_angle_rad.toFixed(3)}</div>
          </div>
          <div class="item-meta">center=${fmtVec(candidate.grasp_position_obj)}<br>closing=${fmtVec(candidate.closing_axis_obj)}</div>
        `;
        item.addEventListener("click", () => {{
          state.selectedIndex = index;
          render();
        }});
        graspList.appendChild(item);
      }});
    }}

    function renderDetails(candidate) {{
      summary.textContent =
        `geometry:        ${data.geometry_name}\\n` +
        `candidate:       ${candidate.rank} / ${data.candidate_count}\\n` +
        `jaw_width:       ${candidate.jaw_width.toFixed(6)} m\\n` +
        `roll_angle_rad:  ${candidate.roll_angle_rad.toFixed(6)}\\n` +
        `midpoint_error:  ${candidate.contact_midpoint_error.toFixed(8)}`;

      geometry.textContent =
        `grasp_position_obj:      ${fmtVec(candidate.grasp_position_obj)}\\n` +
        `grasp_orientation_xyzw:  ${fmtVec(candidate.grasp_orientation_xyzw_obj)}\\n` +
        `contact_point_a_obj:     ${fmtVec(candidate.contact_point_a_obj)}\\n` +
        `contact_point_b_obj:     ${fmtVec(candidate.contact_point_b_obj)}\\n` +
        `contact_normal_a_obj:    ${fmtVec(candidate.contact_normal_a_obj)}\\n` +
        `contact_normal_b_obj:    ${fmtVec(candidate.contact_normal_b_obj)}\\n` +
        `closing_axis_obj:        ${fmtVec(candidate.closing_axis_obj)}\\n` +
        `gripper_x_axis_obj:      ${fmtVec(candidate.gripper_x_axis_obj)}\\n` +
        `gripper_y_axis_obj:      ${fmtVec(candidate.gripper_y_axis_obj)}\\n` +
        `gripper_z_axis_obj:      ${fmtVec(candidate.gripper_z_axis_obj)}`;

      configView.textContent =
        `vertices:        ${data.vertex_count}\\n` +
        `triangles:       ${data.triangle_count}\\n` +
        `samples:         ${data.config.num_surface_samples}\\n` +
        `jaw_limits:      [${data.config.min_jaw_width.toFixed(4)}, ${data.config.max_jaw_width.toFixed(4)}]\\n` +
        `antipodal_cos:   ${data.config.antipodal_cosine_threshold.toFixed(4)}\\n` +
        `rolls:           ${data.config.roll_angles_rad.map((v) => Number(v).toFixed(3)).join(", ")}\\n` +
        `finger_dims:     (${data.config.finger_depth.toFixed(4)}, ${data.config.finger_length.toFixed(4)}, ${data.config.finger_thickness.toFixed(4)})\\n` +
        `finger_clear:    ${data.config.finger_clearance.toFixed(4)}`;
    }}

    function renderScene(candidate) {{
      scene.replaceChildren();
      const defs = addSvg("defs", {{}});
      const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
      marker.setAttribute("id", "arrow");
      marker.setAttribute("markerWidth", "8");
      marker.setAttribute("markerHeight", "8");
      marker.setAttribute("refX", "7");
      marker.setAttribute("refY", "4");
      marker.setAttribute("orient", "auto");
      marker.innerHTML = '<path d="M0,0 L8,4 L0,8 z" fill="currentColor"></path>';
      defs.appendChild(marker);

      drawMesh();

      drawFingerBox(candidate.finger_box_a, "#6d3cc6");
      drawFingerBox(candidate.finger_box_b, "#6d3cc6");
      drawLine(candidate.contact_point_a_obj, candidate.contact_point_b_obj, {{ stroke: "#b43f2c", strokeWidth: 3, opacity: 0.9 }});

      drawPoint(candidate.grasp_position_obj, {{ fill: "#b43f2c", radius: 7 }});
      drawPoint(candidate.contact_point_a_obj, {{ fill: "#c8452d", radius: 6 }});
      drawPoint(candidate.contact_point_b_obj, {{ fill: "#1f7c60", radius: 6 }});

      drawArrow(candidate.contact_point_a_obj, candidate.contact_normal_a_obj, 0.03, "#c8452d", 2.5, "nA");
      drawArrow(candidate.contact_point_b_obj, candidate.contact_normal_b_obj, 0.03, "#1f7c60", 2.5, "nB");
      drawArrow(candidate.grasp_position_obj, candidate.gripper_x_axis_obj, 0.035, "#b43f2c", 2.5, "x");
      drawArrow(candidate.grasp_position_obj, candidate.gripper_y_axis_obj, 0.035, "#6d3cc6", 2.5, "y");
      drawArrow(candidate.grasp_position_obj, candidate.gripper_z_axis_obj, 0.035, "#1397a6", 2.5, "z");

      drawLabel(candidate.grasp_position_obj, "g", "#b43f2c");
      drawLabel(candidate.contact_point_a_obj, "A", "#c8452d", 10, 14);
      drawLabel(candidate.contact_point_b_obj, "B", "#1f7c60", 10, 14);
    }}

    function render() {{
      const candidate = data.candidates[state.selectedIndex];
      renderList();
      renderDetails(candidate);
      renderScene(candidate);
    }}

    function clamp(value, min, max) {{
      return Math.min(max, Math.max(min, value));
    }}

    prevBtn.addEventListener("click", () => {{
      state.selectedIndex = (state.selectedIndex - 1 + data.candidates.length) % data.candidates.length;
      render();
    }});

    nextBtn.addEventListener("click", () => {{
      state.selectedIndex = (state.selectedIndex + 1) % data.candidates.length;
      render();
    }});

    meshModeBtn.addEventListener("click", () => {{
      state.meshRenderMode = state.meshRenderMode === "wireframe" ? "solid" : "wireframe";
      meshModeBtn.textContent = state.meshRenderMode === "wireframe" ? "Solid Mesh" : "Wireframe Mesh";
      render();
    }});

    window.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowUp" || event.key === "ArrowLeft") {{
        event.preventDefault();
        prevBtn.click();
      }}
      if (event.key === "ArrowDown" || event.key === "ArrowRight") {{
        event.preventDefault();
        nextBtn.click();
      }}
    }});

    scene.addEventListener("pointerdown", (event) => {{
      if (event.button !== 0 && event.button !== 1) {{
        return;
      }}
      event.preventDefault();
      state.dragging = true;
      state.dragMode = event.button === 1 ? "pan" : "rotate";
      state.lastPointerX = event.clientX;
      state.lastPointerY = event.clientY;
      state.pointerId = event.pointerId;
      scene.setPointerCapture(event.pointerId);
      scene.style.cursor = state.dragMode === "pan" ? "move" : "grabbing";
    }});

    function stopDragging() {{
      state.dragging = false;
      state.pointerId = null;
      scene.style.cursor = "grab";
    }}

    scene.addEventListener("pointerup", (event) => {{
      if (state.pointerId === event.pointerId) {{
        stopDragging();
      }}
    }});

    scene.addEventListener("pointercancel", () => {{
      stopDragging();
    }});

    scene.addEventListener("pointermove", (event) => {{
      if (!state.dragging || (state.pointerId !== null && event.pointerId !== state.pointerId)) {{
        return;
      }}
      const dx = event.clientX - state.lastPointerX;
      const dy = event.clientY - state.lastPointerY;
      state.lastPointerX = event.clientX;
      state.lastPointerY = event.clientY;
      if (state.dragMode === "pan") {{
        state.panX += dx;
        state.panY += dy;
      }} else {{
        state.yaw = wrapAngle(state.yaw + dx * 0.01);
        state.pitch = wrapAngle(state.pitch - dy * 0.01);
      }}
      render();
    }});

    scene.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const zoomFactor = event.deltaY < 0 ? 1.08 : 1 / 1.08;
      state.zoom = clamp(state.zoom * zoomFactor, 0.35, 4.0);
      render();
    }}, {{ passive: false }});

    scene.style.cursor = "grab";
    scene.addEventListener("contextmenu", (event) => event.preventDefault());

    if (data.candidates.length > 0) {{
      render();
    }} else {{
      graspList.textContent = "No candidates passed the current filters.";
      summary.textContent = "No candidates generated.";
      geometry.textContent = "";
      configView.textContent = "";
    }}
  </script>
</body>
</html>
"""
    return template.replace("{{", "{").replace("}}", "}").replace("__DATA_JSON__", data_json)


def _build_viewer_state(args: argparse.Namespace) -> _ViewerState:
    config = AntipodalGraspGeneratorConfig(
        num_surface_samples=args.num_samples,
        min_jaw_width=args.min_jaw_width,
        max_jaw_width=args.max_jaw_width,
        antipodal_cosine_threshold=args.antipodal_cosine_threshold,
        roll_angles_rad=args.roll_angles_rad,
        max_pair_checks=args.max_pair_checks,
        finger_depth=args.finger_depth,
        finger_length=args.finger_length,
        finger_thickness=args.finger_thickness,
        finger_clearance=args.finger_clearance,
        contact_patch_radius=args.contact_patch_radius,
        collision_sample_count=args.collision_sample_count,
        rng_seed=args.rng_seed,
    )
    mesh = _build_mesh(args)
    generator = AntipodalMeshGraspGenerator(config)
    candidates = generator.generate(mesh)
    return _ViewerState(
        mesh=mesh,
        edges=_unique_edges(mesh.faces),
        candidates=candidates,
        config=config,
        geometry_name=args.geometry,
    )


def main() -> None:
    args = _config_from_sources(parser.parse_args())
    state = _build_viewer_state(args)
    mins = state.mesh.vertices_obj.min(axis=0)
    maxs = state.mesh.vertices_obj.max(axis=0)
    extents = maxs - mins
    print(
        "[INFO] Mesh summary: "
        f"vertices={len(state.mesh.vertices_obj)} triangles={len(state.mesh.faces)} "
        f"extents_m=({extents[0]:.6f}, {extents[1]:.6f}, {extents[2]:.6f})",
        flush=True,
    )
    if np.max(np.abs(extents)) > 2.0:
        print(
            "[WARN] Mesh extents look very large for meter units. "
            "If the STL was authored in millimeters, retry with --stl-scale 0.001.",
            flush=True,
        )
    if not state.candidates:
        print(
            "[WARN] No grasp candidates passed the current filters. "
            "If scale is correct, try more samples and a looser antipodal threshold, "
            "for example: --num-samples 1024 --max-pair-checks 60000 "
            "--antipodal-cosine-threshold 0.8 --min-jaw-width 0.005",
            flush=True,
        )
    output_path = args.output_html.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_document(_build_payload(state)), encoding="utf-8")
    print(f"[INFO] Wrote HTML mesh grasp debug view to: {output_path}")


if __name__ == "__main__":
    main()
