#!/usr/bin/env python3
"""Build the hardware-canonical MoveIt/ros2_control iiwa7 description."""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_URDF = REPO_ROOT / "assets" / "urdf" / "kuka_iiwa7_y_gripper" / "urdf" / "kuka_iiwa7_y_gripper.urdf"
DEFAULT_OUTPUT_XACRO = (
    REPO_ROOT / "ros2_ws" / "src" / "robot_integration_ros" / "urdf" / "iiwa7_y_gripper_moveit.urdf.xacro"
)
DEFAULT_DUAL_OUTPUT_XACRO = (
    REPO_ROOT / "ros2_ws" / "src" / "robot_integration_ros" / "urdf" / "dual_iiwa7_y_gripper_moveit.urdf.xacro"
)
XACRO_NS = "http://www.ros.org/wiki/xacro"
PACKAGE_MESH_PREFIX = "package://robot_integration_ros/meshes/"
LBR_HARDWARE_EE_OFFSET_M = 0.035
DUAL_ARM_BASE_Y_M = {
    "lbr_one": -0.42,
    "lbr_two": 0.42,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-urdf", type=Path, default=DEFAULT_SOURCE_URDF)
    parser.add_argument("--output-xacro", type=Path, default=DEFAULT_OUTPUT_XACRO)
    parser.add_argument("--dual-output-xacro", type=Path, default=DEFAULT_DUAL_OUTPUT_XACRO)
    return parser.parse_args()


def _link_name(name: str) -> str:
    if name == "base_link":
        return "$(arg robot_name)_link_0"
    if name.startswith("link") and name[4:].isdigit():
        return f"$(arg robot_name)_link_{name[4:]}"
    return name


def _joint_name(name: str) -> str:
    if name.startswith("joint") and name[5:].isdigit():
        return f"$(arg robot_name)_A{name[5:]}"
    return name


def _rename_description_element(element: ET.Element) -> ET.Element:
    renamed = copy.deepcopy(element)
    if renamed.tag == "link":
        renamed.set("name", _link_name(str(renamed.get("name"))))
    elif renamed.tag == "joint":
        renamed.set("name", _joint_name(str(renamed.get("name"))))
        parent = renamed.find("parent")
        child = renamed.find("child")
        if parent is not None:
            parent.set("link", _link_name(str(parent.get("link"))))
        if child is not None:
            child.set("link", _link_name(str(child.get("link"))))
        mimic = renamed.find("mimic")
        if mimic is not None and mimic.get("joint"):
            mimic.set("joint", _joint_name(str(mimic.get("joint"))))

    for mesh in renamed.iter("mesh"):
        filename = str(mesh.get("filename", ""))
        if filename.startswith("../meshes/"):
            mesh.set("filename", PACKAGE_MESH_PREFIX + Path(filename).name)
    return renamed


def _dual_link_name(robot_name: str, name: str) -> str:
    if name == "base_link":
        return f"{robot_name}_link_0"
    if name.startswith("link") and name[4:].isdigit():
        return f"{robot_name}_link_{name[4:]}"
    return f"{robot_name}_{name}"


def _dual_joint_name(robot_name: str, name: str) -> str:
    if name.startswith("joint") and name[5:].isdigit():
        return f"{robot_name}_A{name[5:]}"
    return f"{robot_name}_{name}"


def _rename_dual_description_element(element: ET.Element, *, robot_name: str) -> ET.Element:
    renamed = copy.deepcopy(element)
    if renamed.tag == "link":
        renamed.set("name", _dual_link_name(robot_name, str(renamed.get("name"))))
    elif renamed.tag == "joint":
        renamed.set("name", _dual_joint_name(robot_name, str(renamed.get("name"))))
        parent = renamed.find("parent")
        child = renamed.find("child")
        if parent is not None:
            parent.set("link", _dual_link_name(robot_name, str(parent.get("link"))))
        if child is not None:
            child.set("link", _dual_link_name(robot_name, str(child.get("link"))))
        mimic = renamed.find("mimic")
        if mimic is not None and mimic.get("joint"):
            mimic.set("joint", _dual_joint_name(robot_name, str(mimic.get("joint"))))

    for material in renamed.iter("material"):
        material_name = str(material.get("name", ""))
        if material_name:
            material.set("name", f"{robot_name}_{material_name}")
    for mesh in renamed.iter("mesh"):
        filename = str(mesh.get("filename", ""))
        if filename.startswith("../meshes/"):
            mesh.set("filename", PACKAGE_MESH_PREFIX + Path(filename).name)
    return renamed


def _append_hardware_ee(root: ET.Element, *, robot_name: str) -> None:
    ET.SubElement(root, "link", {"name": f"{robot_name}_link_ee"})
    hardware_ee_joint = ET.SubElement(
        root,
        "joint",
        {"name": f"{robot_name}_joint_ee", "type": "fixed"},
    )
    ET.SubElement(hardware_ee_joint, "parent", {"link": f"{robot_name}_link_7"})
    ET.SubElement(hardware_ee_joint, "child", {"link": f"{robot_name}_link_ee"})
    ET.SubElement(
        hardware_ee_joint,
        "origin",
        {"xyz": f"0 0 {LBR_HARDWARE_EE_OFFSET_M:g}", "rpy": "0 0 0"},
    )


def build_moveit_xacro(*, source_urdf: Path, output_xacro: Path) -> Path:
    source_root = ET.parse(source_urdf).getroot()
    arm_joints = [
        joint
        for joint in source_root.findall("joint")
        if str(joint.get("name", "")).startswith("joint") and str(joint.get("name", ""))[5:].isdigit()
    ]
    if len(arm_joints) != 7:
        raise ValueError(f"Expected 7 arm joints in '{source_urdf}', found {len(arm_joints)}.")

    ET.register_namespace("xacro", XACRO_NS)
    # The SRDF and MoveIt configuration package are named "iiwa7". The names
    # must match exactly or MoveIt silently drops semantic groups.
    root = ET.Element("robot", {"name": "iiwa7"})
    root.append(
        ET.Comment(
            " Generated from assets/urdf/kuka_iiwa7_y_gripper/urdf/kuka_iiwa7_y_gripper.urdf; do not hand-edit. "
        )
    )
    ET.SubElement(
        root,
        f"{{{XACRO_NS}}}include",
        {"filename": "$(find lbr_description)/ros2_control/lbr_system_interface.xacro"},
    )
    for name, default in (
        ("robot_name", "lbr"),
        ("mode", "mock"),
        ("system_config_path", "$(find lbr_description)/ros2_control/lbr_system_config.yaml"),
        (
            "initial_joint_positions_path",
            "$(find lbr_description)/ros2_control/initial_joint_positions.yaml",
        ),
    ):
        ET.SubElement(root, f"{{{XACRO_NS}}}arg", {"name": name, "default": default})

    ET.SubElement(root, f"{{{XACRO_NS}}}property", {"name": "PI", "value": "3.1415926535897931"})
    ET.SubElement(
        root,
        f"{{{XACRO_NS}}}property",
        {
            "name": "joint_limits_path",
            "value": "$(find lbr_description)/urdf/iiwa7/joint_limits.yaml",
        },
    )
    ET.SubElement(
        root,
        f"{{{XACRO_NS}}}property",
        {"name": "joint_limits", "value": "${xacro.load_yaml(joint_limits_path)}"},
    )

    ET.SubElement(root, "link", {"name": "world"})
    world_joint = ET.SubElement(root, "joint", {"name": "$(arg robot_name)_world_joint", "type": "fixed"})
    ET.SubElement(world_joint, "parent", {"link": "world"})
    ET.SubElement(world_joint, "child", {"link": "$(arg robot_name)_link_0"})

    for element in source_root:
        if element.tag in {"link", "joint"}:
            root.append(_rename_description_element(element))

    # lbr_ros2_control's force/torque estimator is configured against
    # <robot_name>_link_ee. Keep that canonical frame at 35 mm above link_7;
    # the generated asset independently carries the calibrated hand-mount
    # offset relative to this EE frame.
    ET.SubElement(root, "link", {"name": "$(arg robot_name)_link_ee"})
    hardware_ee_joint = ET.SubElement(
        root,
        "joint",
        {"name": "$(arg robot_name)_joint_ee", "type": "fixed"},
    )
    ET.SubElement(hardware_ee_joint, "parent", {"link": "$(arg robot_name)_link_7"})
    ET.SubElement(hardware_ee_joint, "child", {"link": "$(arg robot_name)_link_ee"})
    ET.SubElement(
        hardware_ee_joint,
        "origin",
        {"xyz": f"0 0 {LBR_HARDWARE_EE_OFFSET_M:g}", "rpy": "0 0 0"},
    )

    ET.SubElement(
        root,
        f"{{{XACRO_NS}}}lbr_system_interface",
        {
            "robot_name": "$(arg robot_name)",
            "mode": "$(arg mode)",
            "joint_limits": "${joint_limits}",
            "system_config_path": "$(arg system_config_path)",
            "initial_joint_positions_path": "$(arg initial_joint_positions_path)",
        },
    )

    ET.indent(root, space="  ")
    output_xacro.parent.mkdir(parents=True, exist_ok=True)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    output_xacro.write_bytes(xml_bytes + b"\n")
    return output_xacro


def build_dual_moveit_xacro(*, source_urdf: Path, output_xacro: Path) -> Path:
    """Build one MoveIt robot model containing two fully prefixed iiwa7/Y-gripper chains."""

    source_root = ET.parse(source_urdf).getroot()
    arm_joints = [
        joint
        for joint in source_root.findall("joint")
        if str(joint.get("name", "")).startswith("joint") and str(joint.get("name", ""))[5:].isdigit()
    ]
    if len(arm_joints) != 7:
        raise ValueError(f"Expected 7 arm joints in '{source_urdf}', found {len(arm_joints)}.")

    ET.register_namespace("xacro", XACRO_NS)
    root = ET.Element("robot", {"name": "lbr_dual_arm"})
    root.append(
        ET.Comment(
            " Generated from assets/urdf/kuka_iiwa7_y_gripper/urdf/kuka_iiwa7_y_gripper.urdf; "
            "both arms carry the calibrated Y-gripper; do not hand-edit. "
        )
    )
    ET.SubElement(
        root,
        f"{{{XACRO_NS}}}include",
        {"filename": "$(find lbr_description)/ros2_control/lbr_system_interface.xacro"},
    )
    for name, default in (
        ("mode", "mock"),
        (
            "lbr_one_system_config_path",
            "$(find robot_integration_ros)/config/lbr_one_system_config.yaml",
        ),
        (
            "lbr_two_system_config_path",
            "$(find robot_integration_ros)/config/lbr_two_system_config.yaml",
        ),
        (
            "initial_joint_positions_path",
            "$(find robot_integration_ros)/config/dual_lbr_initial_joint_positions.yaml",
        ),
    ):
        ET.SubElement(root, f"{{{XACRO_NS}}}arg", {"name": name, "default": default})

    ET.SubElement(root, f"{{{XACRO_NS}}}property", {"name": "PI", "value": "3.1415926535897931"})
    ET.SubElement(
        root,
        f"{{{XACRO_NS}}}property",
        {
            "name": "joint_limits_path",
            "value": "$(find lbr_description)/urdf/iiwa7/joint_limits.yaml",
        },
    )
    ET.SubElement(
        root,
        f"{{{XACRO_NS}}}property",
        {"name": "joint_limits", "value": "${xacro.load_yaml(joint_limits_path)}"},
    )

    ET.SubElement(root, "link", {"name": "base_link"})
    for robot_name, base_y_m in DUAL_ARM_BASE_Y_M.items():
        base_joint = ET.SubElement(root, "joint", {"name": f"{robot_name}_base_joint", "type": "fixed"})
        ET.SubElement(base_joint, "origin", {"xyz": f"0 {base_y_m:g} 0", "rpy": "0 0 0"})
        ET.SubElement(base_joint, "parent", {"link": "base_link"})
        ET.SubElement(base_joint, "child", {"link": f"{robot_name}_link_0"})

        for element in source_root:
            if element.tag in {"link", "joint"}:
                root.append(_rename_dual_description_element(element, robot_name=robot_name))

        _append_hardware_ee(root, robot_name=robot_name)
        ET.SubElement(
            root,
            f"{{{XACRO_NS}}}lbr_system_interface",
            {
                "robot_name": robot_name,
                "mode": "$(arg mode)",
                "joint_limits": "${joint_limits}",
                "system_config_path": f"$(arg {robot_name}_system_config_path)",
                "initial_joint_positions_path": "$(arg initial_joint_positions_path)",
            },
        )

    ET.indent(root, space="  ")
    output_xacro.parent.mkdir(parents=True, exist_ok=True)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    output_xacro.write_bytes(xml_bytes + b"\n")
    return output_xacro


def main() -> None:
    args = _parse_args()
    outputs = (
        build_moveit_xacro(source_urdf=args.source_urdf, output_xacro=args.output_xacro),
        build_dual_moveit_xacro(source_urdf=args.source_urdf, output_xacro=args.dual_output_xacro),
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
