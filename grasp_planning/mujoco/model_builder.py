"""Build local MuJoCo FR3 models augmented with the Menagerie Panda hand."""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

_HAND_ATTACH_POS = "0 0 0.107"
_HAND_ATTACH_QUAT = "0.9238795 0 0 -0.3826834"
_HAND_EE_SITE_NAME = "gripper"
_HAND_PAD_FRICTION = "4.0 0.12 0.02"
_HAND_PAD_PRIORITY = "2"
_HAND_PAD_CONDIM = "6"
_HAND_GRIPPER_KP = 300.0
_HAND_GRIPPER_KV = 20.0
_HAND_GRIPPER_FORCE_LIMIT = 300.0
_HAND_GRIPPER_GAIN = (0.04 * _HAND_GRIPPER_KP) / 255.0
_HAND_CLASS_RENAMES = {
    "panda": "franka_hand",
    "finger": "franka_hand/finger",
    "visual": "franka_hand/visual",
    "collision": "franka_hand/collision",
    "fingertip_pad_collision_1": "franka_hand/fingertip_pad_collision_1",
    "fingertip_pad_collision_2": "franka_hand/fingertip_pad_collision_2",
    "fingertip_pad_collision_3": "franka_hand/fingertip_pad_collision_3",
    "fingertip_pad_collision_4": "franka_hand/fingertip_pad_collision_4",
    "fingertip_pad_collision_5": "franka_hand/fingertip_pad_collision_5",
}


def _append_children(dst: ET.Element, src: ET.Element) -> None:
    for child in list(src):
        dst.append(copy.deepcopy(child))


def _ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def _extract_panda_hand_body(hand_root: ET.Element) -> ET.Element:
    hand_body = hand_root.find("./worldbody/body[@name='hand']")
    if hand_body is None:
        raise RuntimeError("Could not locate Panda hand body in hand.xml")
    hand_body = _rename_hand_classes(hand_body)
    hand_body.set("name", "franka_hand")
    hand_body.set("pos", _HAND_ATTACH_POS)
    hand_body.set("quat", _HAND_ATTACH_QUAT)
    ET.SubElement(hand_body, "site", {"name": _HAND_EE_SITE_NAME, "pos": "0 0 0.1"})
    return hand_body


def _merge_assets(dst_asset: ET.Element, hand_asset: ET.Element) -> None:
    existing_materials = {el.get("name") for el in dst_asset.findall("material")}
    for material in hand_asset.findall("material"):
        name = material.get("name")
        if name == "off_white" and name not in existing_materials:
            dst_asset.append(copy.deepcopy(material))
            existing_materials.add(name)
    for mesh in hand_asset.findall("mesh"):
        dst_asset.append(copy.deepcopy(mesh))


def _rename_hand_classes(root: ET.Element) -> ET.Element:
    root = copy.deepcopy(root)
    for element in root.iter():
        class_attr = element.get("class")
        if class_attr in _HAND_CLASS_RENAMES:
            element.set("class", _HAND_CLASS_RENAMES[class_attr])
        childclass_attr = element.get("childclass")
        if childclass_attr in _HAND_CLASS_RENAMES:
            element.set("childclass", _HAND_CLASS_RENAMES[childclass_attr])
        if element.get("body1") == "hand":
            element.set("body1", "franka_hand")
        if element.get("body2") == "hand":
            element.set("body2", "franka_hand")
        class_attr = element.get("class", "")
        if element.tag == "default" and class_attr.startswith("franka_hand/fingertip_pad_collision_"):
            for child in element.findall("geom"):
                child.set("friction", _HAND_PAD_FRICTION)
                child.set("priority", _HAND_PAD_PRIORITY)
                child.set("condim", _HAND_PAD_CONDIM)
    return root


def _retune_hand_actuator(root: ET.Element) -> None:
    """Increase the Panda-hand tendon actuator strength for firmer grasping."""

    actuator = root.find("./actuator/general[@name='actuator8']")
    if actuator is None:
        return
    actuator.set("forcerange", f"{-_HAND_GRIPPER_FORCE_LIMIT:.9g} {_HAND_GRIPPER_FORCE_LIMIT:.9g}")
    actuator.set("gainprm", f"{_HAND_GRIPPER_GAIN:.9g} 0 0")
    actuator.set("biasprm", f"0 {-_HAND_GRIPPER_KP:.9g} {-_HAND_GRIPPER_KV:.9g}")


def build_fr3_with_panda_hand_xml(
    *,
    arm_xml_path: str | Path,
    panda_hand_xml_path: str | Path,
    output_xml_path: str | Path,
) -> Path:
    """Create a combined MuJoCo XML with an FR3/FR3v2 arm and Panda hand."""

    arm_xml_path = Path(arm_xml_path).expanduser().resolve()
    panda_hand_xml_path = Path(panda_hand_xml_path).expanduser().resolve()
    output_xml_path = Path(output_xml_path).expanduser().resolve()

    arm_tree = ET.parse(arm_xml_path)
    arm_root = arm_tree.getroot()
    hand_root = ET.parse(panda_hand_xml_path).getroot()
    _retune_hand_actuator(hand_root)

    output_dir = output_xml_path.parent

    compiler = _ensure_child(arm_root, "compiler")
    compiler.set("meshdir", ".")

    hand_default = hand_root.find("./default/default[@class='panda']")
    if hand_default is None:
        raise RuntimeError("Could not locate Panda hand default class in hand.xml")
    hand_default = _rename_hand_classes(hand_default)
    dst_default = _ensure_child(arm_root, "default")
    dst_default.append(copy.deepcopy(hand_default))

    hand_asset = hand_root.find("./asset")
    if hand_asset is None:
        raise RuntimeError("Could not locate Panda hand asset block in hand.xml")
    hand_asset = _rename_hand_classes(hand_asset)

    dst_asset = _ensure_child(arm_root, "asset")
    for mesh in dst_asset.findall("mesh"):
        file_attr = mesh.get("file")
        if file_attr:
            mesh.set(
                "file",
                os.path.relpath(Path(arm_xml_path.parent, "assets", file_attr).resolve(), start=output_dir),
            )
    for mesh in hand_asset.findall("mesh"):
        file_attr = mesh.get("file")
        if file_attr:
            mesh.set(
                "file",
                os.path.relpath(Path(panda_hand_xml_path.parent, "assets", file_attr).resolve(), start=output_dir),
            )
    _merge_assets(dst_asset, hand_asset)

    worldbody = arm_root.find("./worldbody")
    if worldbody is None:
        raise RuntimeError("Arm XML has no <worldbody>")
    link7 = worldbody.find(".//body[@name='fr3_link7']")
    if link7 is None:
        link7 = worldbody.find(".//body[@name='fr3v2_link7']")
    if link7 is None:
        raise RuntimeError("Could not locate FR3 terminal link body to attach the hand")
    link7.append(_extract_panda_hand_body(hand_root))

    for tag in ("contact", "tendon", "equality", "actuator"):
        src = hand_root.find(f"./{tag}")
        if src is not None:
            src = _rename_hand_classes(src)
            dst = _ensure_child(arm_root, tag)
            _append_children(dst, src)

    output_xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(arm_tree, space="  ")
    arm_tree.write(output_xml_path, encoding="utf-8", xml_declaration=False)
    return output_xml_path
