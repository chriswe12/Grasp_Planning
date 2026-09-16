from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "isaac_rl/scripts/capture_multigrasp_goal_catalog_mujoco.py"
SPEC = importlib.util.spec_from_file_location("_mujoco_goal_capture_scene", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CAPTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE)


def _imported_gripper_root() -> ET.Element:
    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    for side in ("left", "right"):
        ET.SubElement(
            asset,
            "mesh",
            name=f"{side}_finger",
            file=f"../meshes/collision/{side}_finger.stl",
        )
        ET.SubElement(
            asset,
            "mesh",
            name=f"{side}_pad_8mm",
            file=f"../meshes/collision/{side}_pad_8mm.stl",
        )
        body = ET.SubElement(
            worldbody,
            "body",
            name=f"pdz_gripper_{side}_finger_link",
        )
        ET.SubElement(body, "geom", type="mesh", mesh=f"{side}_finger")
        ET.SubElement(
            body,
            "geom",
            name=f"{side}_tpu_pad",
            type="mesh",
            mesh=f"{side}_pad_8mm",
        )
    return root


def test_mujoco_capture_restores_detailed_gripper_visual_meshes() -> None:
    root = _imported_gripper_root()
    urdf = REPO_ROOT / (
        "assets/urdf/kuka_iiwa7_pdz_gripper/urdf/"
        "kuka_iiwa7_pdz_gripper.urdf"
    )

    CAPTURE._restore_pdz_gripper_visual_meshes(root, urdf)

    meshes = {
        mesh.get("name"): mesh for mesh in root.findall("asset/mesh")
    }
    for side in ("left", "right"):
        finger = root.find(
            f"worldbody/body[@name='pdz_gripper_{side}_finger_link']"
        )
        assert finger is not None
        assert finger.find("geom").get("mesh") == f"pdz_visual_{side}_finger"
        assert finger.find(f"geom[@name='{side}_tpu_pad']").get("mesh") == (
            f"pdz_visual_{side}_pad"
        )
        assert "/meshes/visual/" in meshes[f"pdz_visual_{side}_finger"].get(
            "file"
        )
        assert "/meshes/visual/" in meshes[f"pdz_visual_{side}_pad"].get("file")


def test_mujoco_capture_uses_environment_and_headlight_without_scene_lights() -> None:
    root = _imported_gripper_root()

    CAPTURE._author_filament_fallback_lighting(root)

    assert root.findall("worldbody//light") == []
    headlight = root.find("visual/headlight")
    assert headlight is not None
    assert headlight.get("active") == "1"
    numerics = {
        numeric.get("name"): float(numeric.get("data"))
        for numeric in root.findall("custom/numeric")
    }
    assert numerics["filament.ao.enabled"] == 0.0
    assert numerics["filament.fallback.head_light_intensity"] > 0.0
    assert numerics["filament.fallback.environment_light_intensity"] > 0.0


def test_mujoco_material_xml_uses_low_legacy_specular_response() -> None:
    asset = ET.Element("asset")

    CAPTURE._add_material(
        asset,
        name="matte_test",
        color=(0.5, 0.5, 0.5),
        metallic=0.0,
        roughness=0.9,
    )

    material = asset.find("material")
    assert material is not None
    assert float(material.get("specular")) <= 0.05
    assert float(material.get("shininess")) <= 0.05


def test_mujoco_tslot_matches_canonical_isaac_metric_dimensions() -> None:
    worldbody = ET.Element("worldbody")

    CAPTURE._author_canonical_tslot_surface(worldbody)

    lands = worldbody.findall("geom[@material='tslot_aluminum']")
    assert len(lands) == 25
    land_half_size = tuple(float(value) for value in lands[0].get("size").split())
    assert 2.0 * land_half_size[0] == CAPTURE.CANONICAL_TSLOT_LAND_WIDTH_M
    x_positions = [float(land.get("pos").split()[0]) for land in lands]
    assert x_positions[1] - x_positions[0] == pytest.approx(
        CAPTURE.CANONICAL_TSLOT_PITCH_M
    )
    assert CAPTURE.CANONICAL_TSLOT_SLOT_WIDTH_M == pytest.approx(0.005)

    backing = worldbody.find("geom[@name='tslot_backing']")
    assert backing is not None
    backing_half_size = tuple(float(value) for value in backing.get("size").split())
    assert tuple(2.0 * value for value in backing_half_size[:2]) == (0.65, 0.60)
