#!/usr/bin/env python3
"""Build a KUKA iiwa7 + PDZ gripper URDF from tracked source assets."""

from __future__ import annotations

import copy
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.isaac_visual_materials import (  # noqa: E402
    VISUAL_SERVO_CONTACT_PAD_COLOR,
    VISUAL_SERVO_FINGER_COLOR,
)

KUKA_SOURCE = (
    REPO_ROOT
    / "assets"
    / "urdf"
    / "kuka_iiwa7_y_gripper"
    / "urdf"
    / "kuka_iiwa7_y_gripper.urdf"
)
PDZ_SOURCE_ROOT = REPO_ROOT / "assets" / "urdf" / "pdz_gripper_description"
PDZ_SOURCE = PDZ_SOURCE_ROOT / "urdf" / "pdz_gripper.urdf"
OUTPUT_ROOT = REPO_ROOT / "assets" / "urdf" / "kuka_iiwa7_pdz_gripper"
OUTPUT_URDF = OUTPUT_ROOT / "urdf" / "kuka_iiwa7_pdz_gripper.urdf"
PDZ_TCP_Z_M = 0.1355  # Slim(3) pad vertical midpoint in the flange frame.
PDZ_TCP_YAW_RAD = -0.5 * np.pi  # Planner Y-closing frame from the PDZ X-closing body frame.

ARM_LINKS = {"base_link", *(f"link{index}" for index in range(1, 8))}
ARM_JOINTS = {f"joint{index}" for index in range(1, 8)}
PDZ_LINK_MASSES_KG = {
    "pdz_gripper_base_link": 0.8,
    "pdz_gripper_left_finger_link": 0.08,
    "pdz_gripper_right_finger_link": 0.08,
}
PDZ_COLLISION_MESHES = {
    "pdz_gripper_base_link": ("base.stl",),
    "pdz_gripper_left_finger_link": ("left_finger.stl", "left_pad_8mm.stl"),
    "pdz_gripper_right_finger_link": ("right_finger.stl", "right_pad_8mm.stl"),
}


def _fmt(value: float) -> str:
    if abs(float(value)) < 5.0e-12:
        value = 0.0
    return f"{float(value):.9g}"


def _fmt_vec(values: np.ndarray) -> str:
    return " ".join(_fmt(value) for value in values)


def _indent(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    child_indent = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_indent
        for child in element:
            _indent(child, level + 1)
        if not element.tail or not element.tail.strip():
            element.tail = indent
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def _copy_meshes() -> None:
    output_meshes = OUTPUT_ROOT / "meshes"
    # This directory is generated exclusively from the two source packages.
    # Recreate it so removed vendor meshes cannot survive a package update and
    # remain installable through the generated MoveIt description.
    if output_meshes.is_dir():
        shutil.rmtree(output_meshes)
    output_meshes.mkdir(parents=True, exist_ok=True)
    for mesh in (KUKA_SOURCE.parent.parent / "meshes").glob("*.STL"):
        if mesh.name in {"hand.STL", "left_finger.STL", "right_finger.STL"}:
            continue
        shutil.copy2(mesh, output_meshes / mesh.name)
    for mesh_kind in ("visual", "collision"):
        source_dir = PDZ_SOURCE_ROOT / "meshes" / mesh_kind
        destination_dir = output_meshes / mesh_kind
        destination_dir.mkdir(parents=True, exist_ok=True)
        for mesh in source_dir.glob("*.stl"):
            shutil.copy2(mesh, destination_dir / mesh.name)


def _rewrite_arm_meshes(link: ET.Element) -> None:
    for mesh in link.findall(".//mesh"):
        source_name = Path(str(mesh.get("filename"))).name
        mesh.set("filename", f"../meshes/{source_name}")


def _rewrite_pdz_meshes(link: ET.Element) -> None:
    for tag in ("visual", "collision"):
        for block in link.findall(tag):
            for mesh in block.findall(".//mesh"):
                filename = str(mesh.get("filename"))
                if filename.startswith("package://realsense2_description/"):
                    block.set("remove", "true")
                    continue
                source_name = Path(filename).name
                mesh.set("filename", f"../meshes/{tag}/{source_name}")
    for block in list(link):
        if block.get("remove") == "true":
            link.remove(block)


def _rewrite_pdz_visual_materials(link: ET.Element) -> None:
    """Author the canonical black-finger/white-pad colors in the source URDF."""

    if str(link.get("name")) not in {
        "pdz_gripper_left_finger_link",
        "pdz_gripper_right_finger_link",
    }:
        return
    for visual in link.findall("visual"):
        name = str(visual.get("name", "")).lower()
        mesh = visual.find(".//mesh")
        mesh_name = str(mesh.get("filename", "")).lower() if mesh is not None else ""
        is_pad = "tpu_pad" in name or "pad_8mm" in mesh_name
        material = visual.find("material")
        if material is None:
            material = ET.SubElement(visual, "material")
        material.set("name", "pdz_contact_white" if is_pad else "pdz_finger_black")
        color = material.find("color")
        if color is None:
            color = ET.SubElement(material, "color")
        rgb = VISUAL_SERVO_CONTACT_PAD_COLOR if is_pad else VISUAL_SERVO_FINGER_COLOR
        color.set("rgba", " ".join(_fmt(value) for value in (*rgb, 1.0)))


def _mesh_bounds(link_name: str) -> tuple[np.ndarray, np.ndarray]:
    meshes = []
    for mesh_name in PDZ_COLLISION_MESHES[link_name]:
        mesh = trimesh.load_mesh(PDZ_SOURCE_ROOT / "meshes" / "collision" / mesh_name)
        vertices = np.asarray(mesh.vertices, dtype=float) * 0.001
        meshes.append(vertices)
    vertices = np.vstack(meshes)
    return vertices.min(axis=0), vertices.max(axis=0)


def _add_inertial(link: ET.Element) -> None:
    link_name = str(link.get("name"))
    if link_name not in PDZ_LINK_MASSES_KG or link.find("inertial") is not None:
        return
    mass = PDZ_LINK_MASSES_KG[link_name]
    lower, upper = _mesh_bounds(link_name)
    center = 0.5 * (lower + upper)
    dx, dy, dz = np.maximum(upper - lower, 1.0e-6)
    inertia = (
        mass * (dy * dy + dz * dz) / 12.0,
        mass * (dx * dx + dz * dz) / 12.0,
        mass * (dx * dx + dy * dy) / 12.0,
    )
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", {"xyz": _fmt_vec(center), "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": _fmt(mass)})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": _fmt(inertia[0]),
            "ixy": "0",
            "ixz": "0",
            "iyy": _fmt(inertia[1]),
            "iyz": "0",
            "izz": _fmt(inertia[2]),
        },
    )
    link.insert(0, inertial)


def _ensure_camera_visual(link: ET.Element) -> None:
    """Give Isaac's imported D405 body a visual prim matching its collision.

    Isaac's URDF importer emits a physics-layer reference for every rigid-link
    visual.  The upstream D405 `camera_link` is a body with collision/inertia
    but no visual, which otherwise leaves that generated reference unresolved.
    """

    if str(link.get("name")) != "camera_link" or link.find("visual") is not None:
        return
    collision = link.find("collision")
    if collision is None:
        return
    visual = copy.deepcopy(collision)
    visual.tag = "visual"
    visual.attrib.pop("name", None)
    link.append(visual)


def _build_robot() -> ET.Element:
    kuka = ET.parse(KUKA_SOURCE).getroot()
    pdz = ET.parse(PDZ_SOURCE).getroot()
    robot = ET.Element("robot", {"name": "kuka_iiwa7_pdz_gripper"})

    for link in kuka.findall("link"):
        if str(link.get("name")) in ARM_LINKS:
            arm_link = copy.deepcopy(link)
            _rewrite_arm_meshes(arm_link)
            robot.append(arm_link)

    for link in pdz.findall("link"):
        if str(link.get("name")) == "world":
            continue
        pdz_link = copy.deepcopy(link)
        _rewrite_pdz_meshes(pdz_link)
        _rewrite_pdz_visual_materials(pdz_link)
        _add_inertial(pdz_link)
        _ensure_camera_visual(pdz_link)
        robot.append(pdz_link)

    for joint in kuka.findall("joint"):
        if str(joint.get("name")) in ARM_JOINTS:
            robot.append(copy.deepcopy(joint))

    mount = ET.Element("joint", {"name": "pdz_gripper_mount_joint", "type": "fixed"})
    ET.SubElement(mount, "origin", {"xyz": "0 0 0.035", "rpy": "0 0 0"})
    ET.SubElement(mount, "parent", {"link": "link7"})
    ET.SubElement(mount, "child", {"link": "pdz_gripper_base_link"})
    robot.append(mount)

    for joint in pdz.findall("joint"):
        if str(joint.get("name")) == "pdz_gripper_mount_joint":
            continue
        pdz_joint = copy.deepcopy(joint)
        if str(pdz_joint.get("name")) == "pdz_gripper_tcp_joint":
            origin = pdz_joint.find("origin")
            if origin is None:
                origin = ET.SubElement(pdz_joint, "origin")
            origin.set("xyz", f"0 0 {_fmt(PDZ_TCP_Z_M)}")
            origin.set("rpy", f"0 0 {_fmt(PDZ_TCP_YAW_RAD)}")
        if str(pdz_joint.get("name")) in {
            "pdz_gripper_left_finger_joint",
            "pdz_gripper_right_finger_joint",
        }:
            dynamics = pdz_joint.find("dynamics")
            if dynamics is None:
                dynamics = ET.SubElement(pdz_joint, "dynamics")
            dynamics.attrib.update({"damping": "10.0", "friction": "0.0"})
        robot.append(pdz_joint)

    return robot


def main() -> None:
    if not KUKA_SOURCE.is_file():
        raise FileNotFoundError(KUKA_SOURCE)
    if not PDZ_SOURCE.is_file():
        raise FileNotFoundError(PDZ_SOURCE)
    _copy_meshes()
    robot = _build_robot()
    _indent(robot)
    OUTPUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(OUTPUT_URDF, encoding="utf-8", xml_declaration=True)
    OUTPUT_URDF.write_text(OUTPUT_URDF.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    print(OUTPUT_URDF)


if __name__ == "__main__":
    main()
