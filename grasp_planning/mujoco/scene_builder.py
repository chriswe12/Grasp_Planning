"""Build temporary MuJoCo scenes for object pickup validation."""

from __future__ import annotations

import os
import struct
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import TriangleMesh
from grasp_planning.grasping.world_constraints import ObjectWorldPose


def _quat_xyzw_to_wxyz_text(quat_xyzw: tuple[float, float, float, float]) -> str:
    x, y, z, w = [float(v) for v in quat_xyzw]
    return f"{w:.9g} {x:.9g} {y:.9g} {z:.9g}"


@dataclass(frozen=True)
class MujocoObjectSceneConfig:
    """MuJoCo object/ground contact parameters for pickup validation."""

    object_body_name: str = "target_object"
    object_geom_name: str = "target_object_geom"
    ground_geom_name: str = "ground"
    object_mass_kg: float = 0.15
    object_friction: tuple[float, float, float] = (2.2, 0.08, 0.01)
    ground_friction: tuple[float, float, float] = (1.5, 0.04, 0.002)
    object_condim: int = 6
    object_solref: tuple[float, float] = (0.003, 1.0)
    object_solimp: tuple[float, float, float] = (0.95, 0.99, 0.001)
    object_margin: float = 0.001
    object_gap: float = 0.0005
    floor_half_extent_xy_m: float = 2.0
    floor_thickness_m: float = 0.1


def _format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _ensure_top_level_child(root: ET.Element, tag: str) -> ET.Element:
    child = root.find(tag)
    if child is None:
        child = ET.SubElement(root, tag)
    return child


def _ensure_nested_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def _ensure_named_child(parent: ET.Element, tag: str, *, name: str) -> ET.Element:
    for child in parent.findall(tag):
        if child.get("name") == name:
            return child
    return ET.SubElement(parent, tag, {"name": name})


def build_scene_xml_text(
    *,
    robot_xml_path: str | Path,
    object_mesh_path: str | Path,
    object_pose_world: ObjectWorldPose,
    object_scale: float,
    scene_cfg: MujocoObjectSceneConfig,
) -> str:
    """Return a full MuJoCo XML scene with a robot, ground plane, and one object mesh."""

    robot_xml = Path(robot_xml_path).expanduser().resolve()
    if not robot_xml.is_file():
        raise FileNotFoundError(f"MuJoCo robot XML not found at '{robot_xml}'.")
    object_mesh = Path(object_mesh_path).expanduser().resolve()
    if not object_mesh.is_file():
        raise FileNotFoundError(f"Object mesh not found at '{object_mesh}'.")
    if object_scale <= 0.0:
        raise ValueError("object_scale must be > 0.")

    root = ET.parse(robot_xml).getroot()
    if root.tag != "mujoco":
        raise ValueError(f"Expected MuJoCo XML root <mujoco>, got <{root.tag}> in '{robot_xml}'.")

    compiler = _ensure_top_level_child(root, "compiler")
    compiler.set("angle", compiler.get("angle", "radian"))

    option = _ensure_top_level_child(root, "option")
    option.set("gravity", option.get("gravity", "0 0 -9.81"))
    option.set("timestep", option.get("timestep", "0.002"))
    option.set("iterations", option.get("iterations", "100"))
    option.set("cone", option.get("cone", "elliptic"))

    visual = _ensure_top_level_child(root, "visual")
    headlight = _ensure_nested_child(visual, "headlight")
    headlight.set("active", headlight.get("active", "1"))
    headlight.set("ambient", headlight.get("ambient", "0.18 0.2 0.24"))
    headlight.set("diffuse", headlight.get("diffuse", "0.55 0.55 0.55"))
    headlight.set("specular", headlight.get("specular", "0.12 0.12 0.12"))

    asset = _ensure_top_level_child(root, "asset")
    mesh_name = "target_object_mesh"
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": mesh_name,
            "file": os.path.relpath(object_mesh, start=robot_xml.parent),
            "scale": _format_vec((object_scale, object_scale, object_scale)),
        },
    )

    worldbody = _ensure_top_level_child(root, "worldbody")
    # A simple key/fill/rim setup makes the object and hand easier to read in the viewer.
    for light_name, attributes in (
        (
            "scene_key_light",
            {
                "pos": "1.8 -1.6 3.2",
                "dir": "-0.45 0.35 -1",
                "directional": "true",
                "diffuse": "0.95 0.9 0.82",
                "specular": "0.25 0.25 0.25",
                "castshadow": "true",
            },
        ),
        (
            "scene_fill_light",
            {
                "pos": "-2 1 2.1",
                "dir": "0.55 -0.2 -1",
                "directional": "true",
                "diffuse": "0.45 0.5 0.62",
                "specular": "0.08 0.08 0.08",
            },
        ),
        (
            "scene_rim_light",
            {
                "pos": "-0.4 -2.5 2.6",
                "dir": "0.1 0.9 -0.9",
                "directional": "true",
                "diffuse": "0.3 0.32 0.38",
                "specular": "0.12 0.12 0.12",
            },
        ),
    ):
        light = _ensure_named_child(worldbody, "light", name=light_name)
        for key, value in attributes.items():
            light.set(key, value)
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": scene_cfg.ground_geom_name,
            "type": "plane",
            "size": _format_vec(
                (scene_cfg.floor_half_extent_xy_m, scene_cfg.floor_half_extent_xy_m, scene_cfg.floor_thickness_m)
            ),
            "pos": "0 0 0",
            "friction": _format_vec(scene_cfg.ground_friction),
            "rgba": "0.3 0.32 0.36 1",
        },
    )
    object_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": scene_cfg.object_body_name,
            "pos": _format_vec(object_pose_world.position_world),
            "quat": _quat_xyzw_to_wxyz_text(object_pose_world.orientation_xyzw_world),
        },
    )
    ET.SubElement(object_body, "freejoint", {"name": f"{scene_cfg.object_body_name}_freejoint"})
    ET.SubElement(
        object_body,
        "geom",
        {
            "name": scene_cfg.object_geom_name,
            "type": "mesh",
            "mesh": mesh_name,
            "mass": f"{scene_cfg.object_mass_kg:.9g}",
            "friction": _format_vec(scene_cfg.object_friction),
            "condim": str(int(scene_cfg.object_condim)),
            "solref": _format_vec(scene_cfg.object_solref),
            "solimp": _format_vec(scene_cfg.object_solimp),
            "margin": f"{scene_cfg.object_margin:.9g}",
            "gap": f"{scene_cfg.object_gap:.9g}",
            "rgba": "0.86 0.34 0.26 1",
        },
    )

    return ET.tostring(root, encoding="unicode")


def write_temporary_scene_xml(
    *,
    robot_xml_path: str | Path,
    object_mesh_path: str | Path,
    object_pose_world: ObjectWorldPose,
    object_scale: float,
    scene_cfg: MujocoObjectSceneConfig,
) -> Path:
    """Write a temporary scene XML alongside the robot XML so relative robot assets still resolve."""

    robot_xml = Path(robot_xml_path).expanduser().resolve()
    xml_text = build_scene_xml_text(
        robot_xml_path=robot_xml,
        object_mesh_path=object_mesh_path,
        object_pose_world=object_pose_world,
        object_scale=object_scale,
        scene_cfg=scene_cfg,
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"{robot_xml.stem}_grasp_eval_",
        suffix=".xml",
        dir=robot_xml.parent,
        delete=False,
    ) as handle:
        handle.write(xml_text)
        return Path(handle.name)


def write_temporary_triangle_mesh_stl(
    mesh: TriangleMesh,
    *,
    prefix: str = "mujoco_object_",
    dir: str | Path | None = None,
) -> Path:
    """Write a temporary binary STL for a triangle mesh and return its path."""

    directory = None if dir is None else str(Path(dir).expanduser().resolve())
    vertices = np.asarray(mesh.vertices_obj, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    with tempfile.NamedTemporaryFile(mode="wb", prefix=prefix, suffix=".stl", dir=directory, delete=False) as handle:
        handle.write(b"bundle-local-mesh".ljust(80, b"\0"))
        handle.write(struct.pack("<I", int(faces.shape[0])))
        for face in faces:
            tri = vertices[face]
            edge_a = tri[1] - tri[0]
            edge_b = tri[2] - tri[0]
            normal = np.cross(edge_a, edge_b)
            norm = float(np.linalg.norm(normal))
            if norm > 1.0e-12:
                normal = normal / norm
            else:
                normal = np.zeros(3, dtype=np.float32)
            handle.write(struct.pack("<3f", float(normal[0]), float(normal[1]), float(normal[2])))
            for vertex in tri:
                handle.write(struct.pack("<3f", float(vertex[0]), float(vertex[1]), float(vertex[2])))
            handle.write(struct.pack("<H", 0))
        return Path(handle.name)
