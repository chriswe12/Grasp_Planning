#!/usr/bin/env python3
"""Build a hardware-aligned LBR iiwa 7 R800 + Y-gripper URDF and USD."""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import trimesh
except Exception as exc:  # pragma: no cover - dependency error path
    raise SystemExit(f"trimesh is required to build robot assets: {exc}") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOADS_DIR = Path("/home/pdz/Downloads")
DEFAULT_ARM_SOURCE_ROOT = DEFAULT_DOWNLOADS_DIR / "lbr_iiwa7_r800_zimmer_recreate_2026-06-03"
DEFAULT_ARM_URDF = DEFAULT_ARM_SOURCE_ROOT / "urdf" / "lbr_iiwa7_r800_zimmer.urdf"
DEFAULT_ARM_MESH_DIR = DEFAULT_ARM_SOURCE_ROOT / "meshes"
DEFAULT_ASSET_ROOT = REPO_ROOT / "assets" / "urdf" / "kuka_iiwa7_y_gripper"
DEFAULT_USD_ROOT = REPO_ROOT / "assets" / "usd" / "kuka_iiwa7_y_gripper"

ROBOT_NAME = "kuka_iiwa7_y_gripper"
GRIPPER_MESH_SCALE = 0.001
FINGER_TRAVEL_M = 0.04
GRIPPER_TCP_LINK = "gripper_tcp"
ISAAC_MIN_CONTACT_OFFSET_M = 1.0e-5
# The canonical lbr-stack EE frame is 35 mm above link_7. The physical Y-gripper
# mount sits 4.2 mm back along EE-local Z, while its grasp center is calibrated
# 150.5 mm above the gripper base. Keep these transforms separate so MoveIt and
# Isaac share the same EE, hand-mount, and TCP contract.
LBR_LINK_7_TO_EE_XYZ = (0.0, 0.0, 0.035)
GRIPPER_EE_MOUNT_OFFSET_XYZ = (0.0, 0.0, -0.0042)
GRIPPER_TCP_XYZ = (0.0, 0.0, 0.1505)
GRIPPER_TCP_RPY = (0.0, 0.0, 0.0)
GRIPPER_TCP_MASS_KG = 1.0e-3
GRIPPER_TCP_DIAGONAL_INERTIA = (1.0e-7, 1.0e-7, 1.0e-7)
GRIPPER_MOUNT_XYZ = tuple(
    ee_value + mount_offset
    for ee_value, mount_offset in zip(LBR_LINK_7_TO_EE_XYZ, GRIPPER_EE_MOUNT_OFFSET_XYZ, strict=True)
)
GRIPPER_MOUNT_RPY = (0.0, 0.0, 0.0)
USD_JOINT_TARGET_POSITION_DEG = {
    "joint2": 41.0,
    "joint4": 80.0,
    "joint6": 51.0,
}

ARM_LINKS = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7")
ARM_MESHES = {
    "base_link": "base_link.STL",
    "link1": "link1.STL",
    "link2": "link2.STL",
    "link3": "link3.STL",
    "link4": "link4.STL",
    "link5": "link5.STL",
    "link6": "link6.STL",
    "link7": "link7.STL",
}
GRIPPER_LINKS = {"gripper_base_link", "left_finger_link", "right_finger_link"}
GRIPPER_FINGER_LINKS = {"left_finger_link", "right_finger_link"}

USD_MATERIALS = {
    "arm_silver": (0.65, 0.62, 0.59),
    "gripper_body": (0.55, 0.56, 0.58),
    "finger_dark": (0.17, 0.20, 0.22),
}


@dataclass(frozen=True)
class MeshSpec:
    link_name: str
    source_path: Path
    output_name: str
    scale: float
    color_rgb: tuple[float, float, float]
    local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class LinkInertial:
    mass_kg: float
    center: tuple[float, float, float]
    diagonal_inertia: tuple[float, float, float]
    # URDF tensor entries ordered as (ixy, ixz, iyz).
    off_diagonal_inertia: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None
    mimic: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0
    mimic_usd_multiplier: float | None = None


# The downloaded CAD URDF uses alternate per-link frames and joint coordinate
# signs.  These fixed transforms re-express its meshes and inertials in the
# canonical lbr-stack link frames at q=0.  Kinematics below use the lbr-stack
# joint origins and axes directly; no Cartesian reflection is involved.
ARM_SOURCE_TO_LBR_LINK_FRAME = {
    "base_link": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "link1": ((0.0, 0.0, 0.1900), (0.0, 0.0, 0.0)),
    "link2": ((0.0, 0.0105, -0.0025), (-math.pi / 2.0, 0.0, 0.0)),
    "link3": ((0.0, 0.0, 0.1893), (0.0, 0.0, 0.0)),
    "link4": ((0.0, -0.0105, -0.0032), (-math.pi / 2.0, 0.0, 0.0)),
    "link5": ((0.0, 0.0, 0.1886), (0.0, 0.0, 0.0)),
    "link6": ((0.0, 0.0707, -0.0039), (-math.pi / 2.0, 0.0, 0.0)),
    "link7": ((0.0, 0.0, 0.0311), (0.0, 0.0, 0.0)),
}

LBR_HARDWARE_JOINT_KINEMATICS = (
    ((0.0, 0.0, 0.1475), (0.0, 0.0, 1.0)),
    ((0.0, -0.0105, 0.1925), (0.0, 1.0, 0.0)),
    ((0.0, 0.0105, 0.2075), (0.0, 0.0, 1.0)),
    ((0.0, 0.0105, 0.1925), (0.0, -1.0, 0.0)),
    ((0.0, -0.0105, 0.2075), (0.0, 0.0, 1.0)),
    ((0.0, -0.0707, 0.1925), (0.0, 1.0, 0.0)),
    ((0.0, 0.0707, 0.0910), (0.0, 0.0, 1.0)),
)


def _material_name(spec: MeshSpec) -> str:
    if spec.link_name in ARM_LINKS:
        return "arm_silver"
    if spec.link_name == "gripper_base_link":
        return "gripper_body"
    return "finger_dark"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-dir", type=Path, default=DEFAULT_DOWNLOADS_DIR)
    parser.add_argument("--arm-urdf", type=Path, default=DEFAULT_ARM_URDF)
    parser.add_argument("--arm-mesh-dir", type=Path, default=DEFAULT_ARM_MESH_DIR)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--usd-root", type=Path, default=DEFAULT_USD_ROOT)
    return parser.parse_args()


def _fmt(value: float) -> str:
    value = float(value)
    if abs(value) < 5.0e-12:
        value = 0.0
    return f"{value:.9g}"


def _fmt_vec(values: tuple[float, float, float] | np.ndarray) -> str:
    return " ".join(_fmt(float(value)) for value in values)


def _rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rot_z @ rot_y @ rot_x


def _transform_from_xyz_rpy(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rpy_to_rotmat(*rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def _quat_wxyz_from_rotmat(rotmat: np.ndarray) -> tuple[float, float, float, float]:
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
    quat = np.array([w, x, y, z], dtype=float)
    quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat)


def _parse_float_triplet(raw: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not raw:
        return default
    values = tuple(float(value) for value in raw.split())
    if len(values) != 3:
        raise ValueError(f"Expected 3 floats, got '{raw}'.")
    return values


def _inertia_tensor(inertial: LinkInertial) -> np.ndarray:
    ixx, iyy, izz = inertial.diagonal_inertia
    ixy, ixz, iyz = inertial.off_diagonal_inertia
    return np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ],
        dtype=float,
    )


def _principal_inertia_frame(inertial: LinkInertial) -> tuple[np.ndarray, np.ndarray]:
    """Return principal moments and their right-handed axes in the link frame."""

    tensor = _inertia_tensor(inertial)
    off_diagonal = tensor - np.diag(np.diag(tensor))
    if np.allclose(off_diagonal, 0.0, rtol=0.0, atol=5.0e-12):
        return np.diag(tensor).copy(), np.eye(3, dtype=float)

    moments, axes = np.linalg.eigh(tensor)
    # Eigenvector signs are arbitrary. Canonicalize them before enforcing a
    # proper rotation so generated USD remains stable across rebuilds.
    for column in range(3):
        largest_component = int(np.argmax(np.abs(axes[:, column])))
        if axes[largest_component, column] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return moments, axes


def _load_arm_model(arm_urdf: Path) -> tuple[dict[str, ET.Element], dict[str, LinkInertial], list[JointSpec]]:
    root = ET.parse(arm_urdf).getroot()
    links: dict[str, ET.Element] = {}
    inertials: dict[str, LinkInertial] = {}
    for link in root.findall("link"):
        name = str(link.get("name"))
        if name not in ARM_LINKS:
            continue
        links[name] = copy.deepcopy(link)
        inertial = link.find("inertial")
        if inertial is None:
            raise ValueError(f"Arm link '{name}' has no inertial block.")
        origin = inertial.find("origin")
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass is None or inertia is None:
            raise ValueError(f"Arm link '{name}' has incomplete inertial data.")
        center = _parse_float_triplet(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0))
        inertials[name] = LinkInertial(
            mass_kg=float(mass.get("value", "0")),
            center=center,
            diagonal_inertia=(
                float(inertia.get("ixx", "0")),
                float(inertia.get("iyy", "0")),
                float(inertia.get("izz", "0")),
            ),
            off_diagonal_inertia=(
                float(inertia.get("ixy", "0")),
                float(inertia.get("ixz", "0")),
                float(inertia.get("iyz", "0")),
            ),
        )

    joints: list[JointSpec] = []
    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = str(parent_el.get("link"))
        child = str(child_el.get("link"))
        if parent not in ARM_LINKS or child not in ARM_LINKS:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints.append(
            JointSpec(
                name=str(joint.get("name")),
                joint_type=str(joint.get("type")),
                parent=parent,
                child=child,
                xyz=_parse_float_triplet(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0)),
                rpy=_parse_float_triplet(None if origin is None else origin.get("rpy"), (0.0, 0.0, 0.0)),
                axis=_parse_float_triplet(None if axis is None else axis.get("xyz"), (0.0, 0.0, 1.0)),
                lower=None if limit is None or limit.get("lower") is None else float(limit.get("lower")),
                upper=None if limit is None or limit.get("upper") is None else float(limit.get("upper")),
                effort=None if limit is None or limit.get("effort") is None else float(limit.get("effort")),
                velocity=None if limit is None or limit.get("velocity") is None else float(limit.get("velocity")),
            )
        )
    if set(links) != set(ARM_LINKS):
        missing = sorted(set(ARM_LINKS) - set(links))
        raise ValueError(f"Missing expected arm links in '{arm_urdf}': {missing}")
    return links, inertials, joints


def _gripper_mesh_specs(downloads_dir: Path) -> tuple[MeshSpec, ...]:
    specs = []
    for name, link, mass, color in (
        ("hand.STL", "gripper_base_link", 0.8, (0.55, 0.56, 0.58)),
        ("left_finger.STL", "left_finger_link", 0.08, (0.17, 0.20, 0.22)),
        ("right_finger.STL", "right_finger_link", 0.08, (0.17, 0.20, 0.22)),
    ):
        path = downloads_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required top-level Downloads mesh '{path}'. "
                "This script intentionally does not fall back to lowercase hand.stl or nested folders."
            )
        specs.append(MeshSpec(link, path, name, GRIPPER_MESH_SCALE, color))
    return tuple(specs)


def _all_mesh_specs(downloads_dir: Path, arm_mesh_dir: Path) -> tuple[MeshSpec, ...]:
    arm_specs = []
    for link in ARM_LINKS:
        mesh_name = ARM_MESHES[link]
        path = arm_mesh_dir / mesh_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required LBR arm mesh '{path}'.")
        local_xyz, local_rpy = ARM_SOURCE_TO_LBR_LINK_FRAME[link]
        arm_specs.append(
            MeshSpec(
                link,
                path,
                mesh_name,
                1.0,
                (0.65, 0.62, 0.59),
                local_xyz=local_xyz,
                local_rpy=local_rpy,
            )
        )
    return tuple(arm_specs) + _gripper_mesh_specs(downloads_dir)


def _copy_meshes(mesh_specs: tuple[MeshSpec, ...], mesh_dir: Path) -> dict[str, Path]:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for spec in mesh_specs:
        destination = mesh_dir / spec.output_name
        shutil.copy2(spec.source_path, destination)
        copied[spec.link_name] = destination
    return copied


def _load_scaled_mesh(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, force="mesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh '{path}' has no triangle geometry.")
    return np.asarray(mesh.vertices, dtype=float) * float(scale), np.asarray(mesh.faces, dtype=np.int64)


def _box_inertia(mass_kg: float, extents_m: np.ndarray) -> tuple[float, float, float]:
    dx, dy, dz = [float(max(value, 1.0e-6)) for value in extents_m]
    return (
        mass_kg * (dy * dy + dz * dz) / 12.0,
        mass_kg * (dx * dx + dz * dz) / 12.0,
        mass_kg * (dx * dx + dy * dy) / 12.0,
    )


def _gripper_inertials(mesh_specs: tuple[MeshSpec, ...], mesh_paths: dict[str, Path]) -> dict[str, LinkInertial]:
    inertials: dict[str, LinkInertial] = {}
    for spec in mesh_specs:
        if spec.link_name not in {"gripper_base_link", "left_finger_link", "right_finger_link"}:
            continue
        mass = 0.8 if spec.link_name == "gripper_base_link" else 0.08
        vertices, _faces = _load_scaled_mesh(mesh_paths[spec.link_name], spec.scale)
        bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
        center = tuple(float(v) for v in 0.5 * (bounds[0] + bounds[1]))
        inertia = _box_inertia(mass, bounds[1] - bounds[0])
        inertials[spec.link_name] = LinkInertial(mass_kg=mass, center=center, diagonal_inertia=inertia)
    return inertials


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


def _rewrite_link_for_output(link: ET.Element, *, spec: MeshSpec, inertial: LinkInertial) -> None:
    name = str(link.get("name"))
    mesh_name = ARM_MESHES[name]
    for tag in ("visual", "collision"):
        block = link.find(tag)
        if block is None:
            continue
        origin = block.find("origin")
        if origin is None:
            origin = ET.SubElement(block, "origin")
        origin.set("xyz", _fmt_vec(spec.local_xyz))
        origin.set("rpy", _fmt_vec(spec.local_rpy))
        for mesh in block.findall(".//mesh"):
            mesh.set("filename", f"../meshes/{mesh_name}")
            mesh.set("scale", "1 1 1")

    inertial_el = link.find("inertial")
    if inertial_el is not None:
        origin = inertial_el.find("origin")
        if origin is None:
            origin = ET.SubElement(inertial_el, "origin")
        origin.set("xyz", _fmt_vec(inertial.center))
        origin.set("rpy", "0 0 0")
        mass = inertial_el.find("mass")
        if mass is not None:
            mass.set("value", _fmt(inertial.mass_kg))
        inertia = inertial_el.find("inertia")
        if inertia is not None:
            ixx, iyy, izz = inertial.diagonal_inertia
            ixy, ixz, iyz = inertial.off_diagonal_inertia
            inertia.attrib.update(
                {
                    "ixx": _fmt(ixx),
                    "ixy": _fmt(ixy),
                    "ixz": _fmt(ixz),
                    "iyy": _fmt(iyy),
                    "iyz": _fmt(iyz),
                    "izz": _fmt(izz),
                }
            )


def _make_gripper_link_urdf(spec: MeshSpec, inertial: LinkInertial) -> ET.Element:
    link = ET.Element("link", {"name": spec.link_name})
    inertial_el = ET.SubElement(link, "inertial")
    ET.SubElement(inertial_el, "origin", {"xyz": _fmt_vec(inertial.center), "rpy": "0 0 0"})
    ET.SubElement(inertial_el, "mass", {"value": _fmt(inertial.mass_kg)})
    ixx, iyy, izz = inertial.diagonal_inertia
    ixy, ixz, iyz = inertial.off_diagonal_inertia
    ET.SubElement(
        inertial_el,
        "inertia",
        {
            "ixx": _fmt(ixx),
            "ixy": _fmt(ixy),
            "ixz": _fmt(ixz),
            "iyy": _fmt(iyy),
            "iyz": _fmt(iyz),
            "izz": _fmt(izz),
        },
    )
    for tag in ("visual", "collision"):
        block = ET.SubElement(link, tag)
        ET.SubElement(block, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(block, "geometry")
        ET.SubElement(
            geometry,
            "mesh",
            {
                "filename": f"../meshes/{spec.output_name}",
                "scale": f"{GRIPPER_MESH_SCALE} {GRIPPER_MESH_SCALE} {GRIPPER_MESH_SCALE}",
            },
        )
        if tag == "visual":
            material = ET.SubElement(block, "material", {"name": _material_name(spec)})
            ET.SubElement(material, "color", {"rgba": f"{_fmt_vec(spec.color_rgb)} 1"})
    return link


def _make_tcp_link_urdf() -> ET.Element:
    link = ET.Element("link", {"name": GRIPPER_TCP_LINK})
    inertial_el = ET.SubElement(link, "inertial")
    ET.SubElement(inertial_el, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial_el, "mass", {"value": _fmt(GRIPPER_TCP_MASS_KG)})
    ixx, iyy, izz = GRIPPER_TCP_DIAGONAL_INERTIA
    ET.SubElement(
        inertial_el,
        "inertia",
        {"ixx": _fmt(ixx), "ixy": "0", "ixz": "0", "iyy": _fmt(iyy), "iyz": "0", "izz": _fmt(izz)},
    )
    return link


def _make_joint_xml(joint: JointSpec) -> ET.Element:
    el = ET.Element("joint", {"name": joint.name, "type": joint.joint_type})
    ET.SubElement(el, "origin", {"xyz": _fmt_vec(joint.xyz), "rpy": _fmt_vec(joint.rpy)})
    ET.SubElement(el, "parent", {"link": joint.parent})
    ET.SubElement(el, "child", {"link": joint.child})
    if joint.joint_type != "fixed":
        ET.SubElement(el, "axis", {"xyz": _fmt_vec(joint.axis)})
        if joint.lower is not None and joint.upper is not None:
            ET.SubElement(
                el,
                "limit",
                {
                    "lower": _fmt(joint.lower),
                    "upper": _fmt(joint.upper),
                    "effort": _fmt(100.0 if joint.effort is None else joint.effort),
                    "velocity": _fmt(0.2 if joint.velocity is None else joint.velocity),
                },
            )
        ET.SubElement(el, "dynamics", {"damping": "10.0", "friction": "0.0"})
    if joint.mimic:
        ET.SubElement(
            el,
            "mimic",
            {
                "joint": joint.mimic,
                "multiplier": _fmt(joint.mimic_multiplier),
                "offset": _fmt(joint.mimic_offset),
            },
        )
    return el


def _full_joint_specs(arm_joints: list[JointSpec]) -> list[JointSpec]:
    return [
        *arm_joints,
        JointSpec(
            name="gripper_mount_joint",
            joint_type="fixed",
            parent="link7",
            child="gripper_base_link",
            xyz=GRIPPER_MOUNT_XYZ,
            rpy=GRIPPER_MOUNT_RPY,
            axis=(0.0, 0.0, 1.0),
        ),
        JointSpec(
            name="gripper_tcp_joint",
            joint_type="fixed",
            parent="gripper_base_link",
            child=GRIPPER_TCP_LINK,
            xyz=GRIPPER_TCP_XYZ,
            rpy=GRIPPER_TCP_RPY,
            axis=(0.0, 0.0, 1.0),
        ),
        JointSpec(
            name="left_finger_joint",
            joint_type="prismatic",
            parent="gripper_base_link",
            child="left_finger_link",
            xyz=(0.0, 0.0, 0.0),
            rpy=(0.0, 0.0, 0.0),
            axis=(0.0, 1.0, 0.0),
            lower=0.0,
            upper=FINGER_TRAVEL_M,
            effort=100.0,
            velocity=0.2,
        ),
        JointSpec(
            name="right_finger_joint",
            joint_type="prismatic",
            parent="gripper_base_link",
            child="right_finger_link",
            xyz=(0.0, 0.0, 0.0),
            rpy=(0.0, 0.0, 0.0),
            axis=(0.0, -1.0, 0.0),
            lower=0.0,
            upper=FINGER_TRAVEL_M,
            effort=100.0,
            velocity=0.2,
            mimic="left_finger_joint",
            mimic_multiplier=1.0,
            mimic_offset=0.0,
            mimic_usd_multiplier=1.0,
        ),
    ]


def _hardware_aligned_arm_joints(source_joints: list[JointSpec]) -> list[JointSpec]:
    """Return lbr-stack iiwa7 joint frames while preserving source limits."""

    if len(source_joints) != len(LBR_HARDWARE_JOINT_KINEMATICS):
        raise ValueError(f"Expected 7 source arm joints, got {len(source_joints)}.")
    aligned = []
    for index, (source, (xyz, axis)) in enumerate(
        zip(source_joints, LBR_HARDWARE_JOINT_KINEMATICS, strict=True),
        start=1,
    ):
        aligned.append(
            JointSpec(
                name=f"joint{index}",
                joint_type="revolute",
                parent="base_link" if index == 1 else f"link{index - 1}",
                child=f"link{index}",
                xyz=xyz,
                rpy=(0.0, 0.0, 0.0),
                axis=axis,
                lower=source.lower,
                upper=source.upper,
                effort=source.effort,
                velocity=source.velocity,
            )
        )
    return aligned


def _hardware_aligned_arm_inertials(
    source_inertials: dict[str, LinkInertial],
) -> dict[str, LinkInertial]:
    aligned: dict[str, LinkInertial] = {}
    for link_name, source in source_inertials.items():
        xyz, rpy = ARM_SOURCE_TO_LBR_LINK_FRAME[link_name]
        rotation = _rpy_to_rotmat(*rpy)
        center = rotation @ np.asarray(source.center, dtype=float) + np.asarray(xyz, dtype=float)
        inertia = rotation @ _inertia_tensor(source) @ rotation.T
        aligned[link_name] = LinkInertial(
            mass_kg=source.mass_kg,
            center=tuple(float(value) for value in center),
            diagonal_inertia=tuple(float(value) for value in np.diag(inertia)),
            off_diagonal_inertia=(
                float(inertia[0, 1]),
                float(inertia[0, 2]),
                float(inertia[1, 2]),
            ),
        )
    return aligned


def _write_urdf(
    *,
    urdf_path: Path,
    arm_links: dict[str, ET.Element],
    arm_joints: list[JointSpec],
    mesh_specs: tuple[MeshSpec, ...],
    inertials: dict[str, LinkInertial],
) -> None:
    robot = ET.Element("robot", {"name": ROBOT_NAME})
    mesh_specs_by_link = {spec.link_name: spec for spec in mesh_specs}
    for link_name in ARM_LINKS:
        link = copy.deepcopy(arm_links[link_name])
        _rewrite_link_for_output(
            link,
            spec=mesh_specs_by_link[link_name],
            inertial=inertials[link_name],
        )
        robot.append(link)
    for spec in mesh_specs:
        if spec.link_name in ARM_LINKS:
            continue
        robot.append(_make_gripper_link_urdf(spec, inertials[spec.link_name]))
    robot.append(_make_tcp_link_urdf())
    for joint in _full_joint_specs(arm_joints):
        robot.append(_make_joint_xml(joint))
    _indent(robot)
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)
    urdf_path.write_text(urdf_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")


def _mesh_payload(spec: MeshSpec, mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = _load_scaled_mesh(mesh_path, spec.scale)
    transform = _transform_from_xyz_rpy(spec.local_xyz, spec.local_rpy)
    vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
    return vertices, faces


def _compute_link_world_transforms(joints: list[JointSpec]) -> dict[str, np.ndarray]:
    transforms = {"base_link": np.eye(4, dtype=float)}
    unresolved = list(joints)
    while unresolved:
        next_unresolved: list[JointSpec] = []
        progressed = False
        for joint in unresolved:
            if joint.parent not in transforms:
                next_unresolved.append(joint)
                continue
            transforms[joint.child] = transforms[joint.parent] @ _transform_from_xyz_rpy(joint.xyz, joint.rpy)
            progressed = True
        if not progressed:
            names = ", ".join(joint.name for joint in next_unresolved)
            raise ValueError(f"Could not resolve link transforms for joints: {names}")
        unresolved = next_unresolved
    return transforms


def _axis_token_and_limits(joint: JointSpec) -> tuple[str, float | None, float | None]:
    axis = np.asarray(joint.axis, dtype=float)
    idx = int(np.argmax(np.abs(axis)))
    token = ("X", "Y", "Z")[idx]
    sign = 1.0 if axis[idx] >= 0.0 else -1.0
    lower, upper = joint.lower, joint.upper
    if lower is not None and upper is not None:
        if joint.joint_type == "revolute":
            lower, upper = math.degrees(lower), math.degrees(upper)
        if sign < 0.0:
            lower, upper = -upper, -lower
    return token, lower, upper


def _usd_joint_coordinate_sign(joint: JointSpec) -> float:
    axis = np.asarray(joint.axis, dtype=float)
    idx = int(np.argmax(np.abs(axis)))
    return 1.0 if axis[idx] >= 0.0 else -1.0


def _usd_mimic_api_axis(joint: JointSpec) -> str:
    axis_token, _, _ = _axis_token_and_limits(joint)
    if joint.joint_type == "prismatic":
        # Isaac Sim 5.1's PhysxMimicJointAPI only allows rotX/rotY/rotZ
        # instance names. For single-DOF prismatic joints PhysX uses the
        # joint axis, so the token only needs to be a valid mimic API instance.
        return f"rot{axis_token}"
    if joint.joint_type == "revolute":
        return f"rot{axis_token}"
    raise ValueError(f"Joint '{joint.name}' cannot use a PhysX mimic axis for type '{joint.joint_type}'.")


def _write_int_array(handle, name: str, values: np.ndarray, indent: str) -> None:
    handle.write(f"{indent}int[] {name} = [\n")
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    lines = []
    for start in range(0, len(flat), 24):
        lines.append(f"{indent}    " + ", ".join(str(int(value)) for value in flat[start : start + 24]))
    handle.write(",\n".join(lines))
    if lines:
        handle.write("\n")
    handle.write(f"{indent}]\n")


def _write_points(handle, vertices: np.ndarray, indent: str) -> None:
    handle.write(f"{indent}point3f[] points = [\n")
    lines = [f"{indent}    ({_fmt(x)}, {_fmt(y)}, {_fmt(z)})" for x, y, z in np.asarray(vertices, dtype=float)]
    handle.write(",\n".join(lines))
    if lines:
        handle.write("\n")
    handle.write(f"{indent}]\n")


def _write_materials(handle, indent: str) -> None:
    handle.write(f'{indent}def Scope "Looks"\n')
    handle.write(f"{indent}{{\n")
    for name, color in USD_MATERIALS.items():
        handle.write(f'{indent}    def Material "{name}"\n')
        handle.write(f"{indent}    {{\n")
        handle.write(f"{indent}        token outputs:surface.connect = </{ROBOT_NAME}/Looks/{name}/Shader.outputs:surface>\n")
        handle.write(f'{indent}        def Shader "Shader"\n')
        handle.write(f"{indent}        {{\n")
        handle.write(f'{indent}            uniform token info:id = "UsdPreviewSurface"\n')
        handle.write(f"{indent}            color3f inputs:diffuseColor = ({', '.join(_fmt(value) for value in color)})\n")
        handle.write(f"{indent}            float inputs:roughness = 0.55\n")
        handle.write(f"{indent}            float inputs:metallic = 0\n")
        handle.write(f"{indent}            token outputs:surface\n")
        handle.write(f"{indent}        }}\n")
        handle.write(f"{indent}    }}\n")
    handle.write(f"{indent}}}\n")


def _write_mesh_geometry(handle, vertices: np.ndarray, faces: np.ndarray, indent: str) -> None:
    counts = np.full((len(faces),), 3, dtype=np.int64)
    _write_points(handle, vertices, indent)
    _write_int_array(handle, "faceVertexCounts", counts, indent)
    _write_int_array(handle, "faceVertexIndices", faces.reshape(-1), indent)


def _write_visual_mesh(handle, spec: MeshSpec, vertices: np.ndarray, faces: np.ndarray, indent: str) -> None:
    color = ", ".join(_fmt(value) for value in spec.color_rgb)
    material_name = _material_name(spec)
    handle.write(f'{indent}def Mesh "{spec.link_name}_visual_mesh" (\n')
    handle.write(f'{indent}    prepend apiSchemas = ["MaterialBindingAPI"]\n')
    handle.write(f"{indent})\n{indent}{{\n")
    handle.write(f"{indent}    uniform token subdivisionScheme = \"none\"\n")
    handle.write(f"{indent}    bool doubleSided = true\n")
    handle.write(f"{indent}    rel material:binding = </{ROBOT_NAME}/Looks/{material_name}> (\n")
    handle.write(f'{indent}        bindMaterialAs = "strongerThanDescendants"\n')
    handle.write(f"{indent}    )\n")
    handle.write(f"{indent}    color3f[] primvars:displayColor = [({color})] (\n")
    handle.write(f"{indent}        interpolation = \"constant\"\n")
    handle.write(f"{indent}    )\n")
    _write_mesh_geometry(handle, vertices, faces, f"{indent}    ")
    handle.write(f"{indent}}}\n")


def _convex_hull_payload(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    hull = mesh.convex_hull
    return (
        np.asarray(hull.vertices, dtype=float),
        np.asarray(hull.faces, dtype=np.int64),
    )


def _component_convex_hull_payloads(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    min_area_fraction: float = 0.05,
    min_z_extent_m: float = 0.02,
    split_axis: int | None = None,
    max_hull_extent_m: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    total_area = max(float(mesh.area), 1.0e-12)
    kept = []
    for component in components:
        bounds = np.asarray(component.bounds, dtype=float)
        z_extent_m = float(bounds[1, 2] - bounds[0, 2])
        if float(component.area) / total_area >= float(min_area_fraction) and z_extent_m >= float(min_z_extent_m):
            kept.append(component)
    if not kept:
        kept = [component for component in components if float(component.area) / total_area >= float(min_area_fraction)]
    if not kept:
        kept = [mesh]

    payloads = []
    for component in kept:
        payloads.extend(
            _split_component_convex_hulls(
                component,
                split_axis=split_axis,
                max_hull_extent_m=max_hull_extent_m,
            )
        )
    return payloads


def _split_component_convex_hulls(
    component: trimesh.Trimesh,
    *,
    split_axis: int | None,
    max_hull_extent_m: float | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    bounds = np.asarray(component.bounds, dtype=float)
    hull_count = 1
    if split_axis is not None and max_hull_extent_m is not None and max_hull_extent_m > 0.0:
        extent_m = float(bounds[1, split_axis] - bounds[0, split_axis])
        hull_count = max(1, min(4, int(math.ceil(extent_m / float(max_hull_extent_m)))))
    if hull_count <= 1 or len(component.faces) < 16:
        hull = component.convex_hull
        return [(np.asarray(hull.vertices, dtype=float), np.asarray(hull.faces, dtype=np.int64))]

    centers = np.asarray(component.triangles_center, dtype=float)[:, split_axis]
    edges = np.linspace(float(bounds[0, split_axis]), float(bounds[1, split_axis]), hull_count + 1)
    payloads: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(hull_count):
        lower = edges[index]
        upper = edges[index + 1]
        if index == hull_count - 1:
            face_indices = np.flatnonzero((centers >= lower) & (centers <= upper))
        else:
            face_indices = np.flatnonzero((centers >= lower) & (centers < upper))
        if len(face_indices) < 4:
            continue
        submesh = component.submesh([face_indices], append=True, repair=False)
        if len(submesh.vertices) < 4 or len(submesh.faces) < 4:
            continue
        hull = submesh.convex_hull
        payloads.append((np.asarray(hull.vertices, dtype=float), np.asarray(hull.faces, dtype=np.int64)))
    if payloads:
        return payloads
    hull = component.convex_hull
    return [(np.asarray(hull.vertices, dtype=float), np.asarray(hull.faces, dtype=np.int64))]


def _collision_payloads(spec: MeshSpec, vertices: np.ndarray, faces: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if spec.link_name in GRIPPER_FINGER_LINKS:
        return _component_convex_hull_payloads(
            vertices,
            faces,
            min_area_fraction=0.02,
            min_z_extent_m=0.0,
            split_axis=2,
            max_hull_extent_m=0.03,
        )
    return [_convex_hull_payload(vertices, faces)]


def _write_collision_mesh(handle, spec: MeshSpec, vertices: np.ndarray, faces: np.ndarray, indent: str) -> None:
    collision_payloads = _collision_payloads(spec, vertices, faces)
    for index, (collision_vertices, collision_faces) in enumerate(collision_payloads):
        mesh_name = (
            f"{spec.link_name}_collision_mesh"
            if len(collision_payloads) == 1
            else f"{spec.link_name}_collision_mesh_{index}"
        )
        handle.write(f'{indent}def Mesh "{mesh_name}" (\n')
        handle.write(
            f'{indent}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", '
            f'"PhysicsMeshCollisionAPI", "PhysxConvexHullCollisionAPI"]\n'
        )
        handle.write(f"{indent})\n{indent}{{\n")
        handle.write(f"{indent}    uniform token subdivisionScheme = \"none\"\n")
        handle.write(f"{indent}    uniform token purpose = \"guide\"\n")
        handle.write(f"{indent}    token visibility = \"invisible\"\n")
        handle.write(f"{indent}    bool doubleSided = true\n")
        handle.write(f"{indent}    bool physics:collisionEnabled = true\n")
        handle.write(f"{indent}    float physxCollision:contactOffset = {_fmt(ISAAC_MIN_CONTACT_OFFSET_M)}\n")
        handle.write(f"{indent}    float physxCollision:restOffset = 0\n")
        handle.write(f'{indent}    uniform token physics:approximation = "convexHull"\n')
        _write_mesh_geometry(handle, collision_vertices, collision_faces, f"{indent}    ")
        handle.write(f"{indent}}}\n")


def _write_link(
    handle,
    *,
    spec: MeshSpec,
    mesh_path: Path,
    inertial: LinkInertial,
    transform: np.ndarray,
    indent: str,
) -> None:
    translation = transform[:3, 3]
    quat = _quat_wxyz_from_rotmat(transform[:3, :3])
    principal_moments, principal_axes = _principal_inertia_frame(inertial)
    principal_axes_quat = _quat_wxyz_from_rotmat(principal_axes)
    ixx, iyy, izz = principal_moments
    api_schemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI"]
    if spec.link_name == "base_link":
        api_schemas.extend(["PhysicsArticulationRootAPI", "PhysxArticulationAPI"])
    quoted_api_schemas = ", ".join(f'"{api}"' for api in api_schemas)
    handle.write(f'{indent}def Xform "{spec.link_name}" (\n')
    handle.write(f"{indent}    prepend apiSchemas = [{quoted_api_schemas}]\n")
    handle.write(f"{indent})\n{indent}{{\n")
    handle.write(f"{indent}    double3 xformOp:translate = ({_fmt(translation[0])}, {_fmt(translation[1])}, {_fmt(translation[2])})\n")
    handle.write(f"{indent}    quatf xformOp:orient = ({_fmt(quat[0])}, {_fmt(quat[1])}, {_fmt(quat[2])}, {_fmt(quat[3])})\n")
    handle.write(f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n')
    handle.write(f"{indent}    bool physics:rigidBodyEnabled = true\n")
    handle.write(f"{indent}    bool physxRigidBody:disableGravity = false\n")
    if spec.link_name == "base_link":
        handle.write(f"{indent}    bool physxArticulation:articulationEnabled = true\n")
        handle.write(f"{indent}    bool physxArticulation:enabledSelfCollisions = false\n")
    handle.write(f"{indent}    float physics:mass = {_fmt(inertial.mass_kg)}\n")
    handle.write(
        f"{indent}    point3f physics:centerOfMass = "
        f"({_fmt(inertial.center[0])}, {_fmt(inertial.center[1])}, {_fmt(inertial.center[2])})\n"
    )
    handle.write(f"{indent}    vector3f physics:diagonalInertia = ({_fmt(ixx)}, {_fmt(iyy)}, {_fmt(izz)})\n")
    handle.write(
        f"{indent}    quatf physics:principalAxes = "
        f"({_fmt(principal_axes_quat[0])}, {_fmt(principal_axes_quat[1])}, "
        f"{_fmt(principal_axes_quat[2])}, {_fmt(principal_axes_quat[3])})\n"
    )
    vertices, faces = _mesh_payload(spec, mesh_path)
    _write_visual_mesh(handle, spec, vertices, faces, f"{indent}    ")
    _write_collision_mesh(handle, spec, vertices, faces, f"{indent}    ")
    handle.write(f"{indent}}}\n")


def _write_tcp_link(handle, *, transform: np.ndarray, indent: str) -> None:
    translation = transform[:3, 3]
    quat = _quat_wxyz_from_rotmat(transform[:3, :3])
    ixx, iyy, izz = GRIPPER_TCP_DIAGONAL_INERTIA
    handle.write(f'{indent}def Xform "{GRIPPER_TCP_LINK}" (\n')
    handle.write(
        f'{indent}    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI"]\n'
    )
    handle.write(f"{indent})\n{indent}{{\n")
    handle.write(f"{indent}    double3 xformOp:translate = ({_fmt(translation[0])}, {_fmt(translation[1])}, {_fmt(translation[2])})\n")
    handle.write(f"{indent}    quatf xformOp:orient = ({_fmt(quat[0])}, {_fmt(quat[1])}, {_fmt(quat[2])}, {_fmt(quat[3])})\n")
    handle.write(f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n')
    handle.write(f"{indent}    bool physics:rigidBodyEnabled = true\n")
    handle.write(f"{indent}    bool physxRigidBody:disableGravity = false\n")
    handle.write(f"{indent}    float physics:mass = {_fmt(GRIPPER_TCP_MASS_KG)}\n")
    handle.write(f"{indent}    point3f physics:centerOfMass = (0, 0, 0)\n")
    handle.write(f"{indent}    vector3f physics:diagonalInertia = ({_fmt(ixx)}, {_fmt(iyy)}, {_fmt(izz)})\n")
    handle.write(f"{indent}}}\n")


def _write_drive_api(handle, joint: JointSpec, indent: str) -> None:
    drive_name = "angular" if joint.joint_type == "revolute" else "linear"
    stiffness = 2000.0 if joint.joint_type == "revolute" else 5000.0
    damping = 200.0 if joint.joint_type == "revolute" else 200.0
    target_position = (
        _usd_joint_coordinate_sign(joint) * USD_JOINT_TARGET_POSITION_DEG.get(joint.name, 0.0)
        if joint.joint_type == "revolute"
        else 0.0
    )
    handle.write(f"{indent}    token drive:{drive_name}:physics:type = \"force\"\n")
    handle.write(f"{indent}    float drive:{drive_name}:physics:stiffness = {_fmt(stiffness)}\n")
    handle.write(f"{indent}    float drive:{drive_name}:physics:damping = {_fmt(damping)}\n")
    handle.write(f"{indent}    float drive:{drive_name}:physics:targetPosition = {_fmt(target_position)}\n")
    handle.write(f"{indent}    float drive:{drive_name}:physics:targetVelocity = 0\n")


def _write_joint(handle, joint: JointSpec, indent: str) -> None:
    if joint.joint_type == "fixed":
        usd_type = "PhysicsFixedJoint"
        api_schemas = ""
    elif joint.joint_type == "revolute":
        usd_type = "PhysicsRevoluteJoint"
        schemas = ["PhysicsJointStateAPI:angular"]
        if joint.mimic:
            schemas.append(f"PhysxMimicJointAPI:{_usd_mimic_api_axis(joint)}")
        else:
            schemas.insert(0, "PhysicsDriveAPI:angular")
        api_schemas = ' (\n{}    prepend apiSchemas = [{}]\n{})'.format(
            indent,
            ", ".join(f'"{schema}"' for schema in schemas),
            indent,
        )
    elif joint.joint_type == "prismatic":
        usd_type = "PhysicsPrismaticJoint"
        schemas = ["PhysicsJointStateAPI:linear"]
        if joint.mimic:
            schemas.append(f"PhysxMimicJointAPI:{_usd_mimic_api_axis(joint)}")
        else:
            schemas.insert(0, "PhysicsDriveAPI:linear")
        api_schemas = ' (\n{}    prepend apiSchemas = [{}]\n{})'.format(
            indent,
            ", ".join(f'"{schema}"' for schema in schemas),
            indent,
        )
    else:
        raise ValueError(f"Unsupported joint type '{joint.joint_type}' for joint '{joint.name}'.")

    local_rot0 = _quat_wxyz_from_rotmat(_rpy_to_rotmat(*joint.rpy))
    axis_token, lower, upper = _axis_token_and_limits(joint)
    handle.write(f'{indent}def {usd_type} "{joint.name}"{api_schemas}\n')
    handle.write(f"{indent}{{\n")
    handle.write(f"{indent}    rel physics:body0 = </{ROBOT_NAME}/{joint.parent}>\n")
    handle.write(f"{indent}    rel physics:body1 = </{ROBOT_NAME}/{joint.child}>\n")
    if joint.joint_type != "fixed":
        handle.write(f"{indent}    uniform token physics:axis = \"{axis_token}\"\n")
        if lower is not None and upper is not None:
            handle.write(f"{indent}    float physics:lowerLimit = {_fmt(lower)}\n")
            handle.write(f"{indent}    float physics:upperLimit = {_fmt(upper)}\n")
    handle.write(f"{indent}    point3f physics:localPos0 = ({_fmt_vec(joint.xyz).replace(' ', ', ')})\n")
    handle.write(f"{indent}    point3f physics:localPos1 = (0, 0, 0)\n")
    handle.write(
        f"{indent}    quatf physics:localRot0 = "
        f"({_fmt(local_rot0[0])}, {_fmt(local_rot0[1])}, {_fmt(local_rot0[2])}, {_fmt(local_rot0[3])})\n"
    )
    handle.write(f"{indent}    quatf physics:localRot1 = (1, 0, 0, 0)\n")
    if joint.mimic:
        mimic_axis = _usd_mimic_api_axis(joint)
        mimic_multiplier = joint.mimic_usd_multiplier
        if mimic_multiplier is None:
            mimic_multiplier = joint.mimic_multiplier
        handle.write(
            f"{indent}    rel physxMimicJoint:{mimic_axis}:referenceJoint = "
            f"</{ROBOT_NAME}/joints/{joint.mimic}>\n"
        )
        handle.write(f"{indent}    float physxMimicJoint:{mimic_axis}:gearing = {_fmt(mimic_multiplier)}\n")
        handle.write(f"{indent}    float physxMimicJoint:{mimic_axis}:offset = {_fmt(joint.mimic_offset)}\n")
    if joint.joint_type in {"revolute", "prismatic"}:
        if not joint.mimic:
            _write_drive_api(handle, joint, indent)
        state_name = "angular" if joint.joint_type == "revolute" else "linear"
        state_position = (
            _usd_joint_coordinate_sign(joint) * USD_JOINT_TARGET_POSITION_DEG.get(joint.name, 0.0)
            if joint.joint_type == "revolute"
            else 0.0
        )
        handle.write(f"{indent}    float state:{state_name}:physics:position = {_fmt(state_position)}\n")
        handle.write(f"{indent}    float state:{state_name}:physics:velocity = 0\n")
    handle.write(f"{indent}}}\n")


def _write_world_fixed_base_joint(handle, indent: str) -> None:
    handle.write(f'{indent}def PhysicsFixedJoint "world_fixed_base_joint"\n')
    handle.write(f"{indent}{{\n")
    handle.write(f"{indent}    rel physics:body1 = </{ROBOT_NAME}/base_link>\n")
    handle.write(f"{indent}    float physics:breakForce = inf\n")
    handle.write(f"{indent}    float physics:breakTorque = inf\n")
    handle.write(f"{indent}    point3f physics:localPos0 = (0, 0, 0)\n")
    handle.write(f"{indent}    point3f physics:localPos1 = (0, 0, 0)\n")
    handle.write(f"{indent}    quatf physics:localRot0 = (1, 0, 0, 0)\n")
    handle.write(f"{indent}    quatf physics:localRot1 = (1, 0, 0, 0)\n")
    handle.write(f"{indent}}}\n")


def _write_usd(
    *,
    usd_path: Path,
    mesh_specs: tuple[MeshSpec, ...],
    mesh_paths: dict[str, Path],
    inertials: dict[str, LinkInertial],
    joints: list[JointSpec],
) -> None:
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    transforms = _compute_link_world_transforms(joints)
    with usd_path.open("w", encoding="utf-8") as handle:
        handle.write("#usda 1.0\n")
        handle.write("(\n")
        handle.write(f"    defaultPrim = \"{ROBOT_NAME}\"\n")
        handle.write("    metersPerUnit = 1\n")
        handle.write("    upAxis = \"Z\"\n")
        handle.write(")\n\n")
        handle.write(f'def Xform "{ROBOT_NAME}"\n')
        handle.write("{\n")
        handle.write("    double3 xformOp:translate = (0, 0, 0)\n")
        handle.write("    quatf xformOp:orient = (1, 0, 0, 0)\n")
        handle.write('    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n\n')
        _write_materials(handle, "    ")
        handle.write("\n")
        for spec in mesh_specs:
            _write_link(
                handle,
                spec=spec,
                mesh_path=mesh_paths[spec.link_name],
                inertial=inertials[spec.link_name],
                transform=transforms[spec.link_name],
                indent="    ",
            )
            handle.write("\n")
        _write_tcp_link(handle, transform=transforms[GRIPPER_TCP_LINK], indent="    ")
        handle.write("\n")
        handle.write('    def Scope "joints"\n')
        handle.write("    {\n")
        _write_world_fixed_base_joint(handle, "        ")
        handle.write("\n")
        for joint in joints:
            _write_joint(handle, joint, "        ")
            handle.write("\n")
        handle.write("    }\n")
        handle.write("}\n")


def main() -> None:
    args = _parse_args()
    downloads_dir = args.downloads_dir.expanduser().resolve()
    arm_urdf = args.arm_urdf.expanduser().resolve()
    arm_mesh_dir = args.arm_mesh_dir.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    usd_root = args.usd_root.expanduser().resolve()

    arm_links, source_arm_inertials, source_arm_joints = _load_arm_model(arm_urdf)
    arm_inertials = _hardware_aligned_arm_inertials(source_arm_inertials)
    arm_joints = _hardware_aligned_arm_joints(source_arm_joints)
    mesh_specs = _all_mesh_specs(downloads_dir, arm_mesh_dir)
    mesh_paths = _copy_meshes(mesh_specs, asset_root / "meshes")
    inertials = {**arm_inertials, **_gripper_inertials(mesh_specs, mesh_paths)}
    joints = _full_joint_specs(arm_joints)

    urdf_path = asset_root / "urdf" / f"{ROBOT_NAME}.urdf"
    usd_path = usd_root / f"{ROBOT_NAME}.usda"
    _write_urdf(
        urdf_path=urdf_path,
        arm_links=arm_links,
        arm_joints=arm_joints,
        mesh_specs=mesh_specs,
        inertials=inertials,
    )
    _write_usd(
        usd_path=usd_path,
        mesh_specs=mesh_specs,
        mesh_paths=mesh_paths,
        inertials=inertials,
        joints=joints,
    )
    print(urdf_path)
    print(usd_path)


if __name__ == "__main__":
    main()
