from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
)
from grasp_planning.isaac_visual_materials import (
    VISUAL_SERVO_CONTACT_PAD_COLOR,
    VISUAL_SERVO_CONTACT_PAD_ROUGHNESS,
    VISUAL_SERVO_FINGER_COLOR,
    VISUAL_SERVO_FINGER_ROUGHNESS,
    VISUAL_SERVO_MATERIAL_PROFILE,
    VISUAL_SERVO_PART_COLOR,
    VISUAL_SERVO_PART_ROUGHNESS,
    classify_robot_finger_geometry_material,
)
from grasp_planning.isaac_visual_scene import (
    VISUAL_SERVO_DIRECT_LIGHT_SAMPLES,
    VISUAL_SERVO_DL_DENOISER_ENABLED,
    VISUAL_SERVO_DOME_INTENSITY,
    VISUAL_SERVO_GROUND_COLOR,
    VISUAL_SERVO_KEY_INTENSITY,
    VISUAL_SERVO_SCENE_PROFILE,
)
from grasp_planning.rl.visual_servo_dataset import MmapVisualServoFrameDataset
from scripts.build_visual_servo_training_cache import _area_resize
from grasp_planning.rl.goal_catalog_profiles import GOAL_FILAMENT_MATERIALS


def test_canonical_scene_is_small_tslot_with_pdz_black_white_fingers_and_a_stronger_shape_key() -> None:
    assert "dome_directional_dlaa" in VISUAL_SERVO_SCENE_PROFILE
    assert "small_tslot" in VISUAL_SERVO_SCENE_PROFILE
    assert "pdz_black_whitepads" in VISUAL_SERVO_MATERIAL_PROFILE
    assert max(VISUAL_SERVO_GROUND_COLOR) <= 0.10
    assert VISUAL_SERVO_KEY_INTENSITY > VISUAL_SERVO_DOME_INTENSITY


def test_canonical_scene_uses_denoised_four_sample_direct_lighting() -> None:
    assert "4spp_dldenoise" in VISUAL_SERVO_SCENE_PROFILE
    assert VISUAL_SERVO_DIRECT_LIGHT_SAMPLES == 4
    assert VISUAL_SERVO_DL_DENOISER_ENABLED is True


def test_cache_resize_averages_pixels_instead_of_stride_sampling() -> None:
    image = np.asarray(
        [
            [0, 100, 0, 100],
            [100, 200, 100, 200],
            [0, 100, 0, 100],
            [100, 200, 100, 200],
        ],
        dtype=np.float32,
    )

    resized = _area_resize(image, height=2, width=2)

    np.testing.assert_allclose(resized, np.full((2, 2), 100.0, dtype=np.float32))


def test_legacy_nearest_sampled_training_cache_is_rejected(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "resampling": "stride",
                "observation_profile": "legacy",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema 2 area filtering"):
        MmapVisualServoFrameDataset(tmp_path, split="train")


def test_observation_profile_records_area_filtered_policy_shape() -> None:
    assert D405_VISUAL_SERVO_OBSERVATION_PROFILE == "rgbd_render_256x144_valid_area_128x72_d405_range_v3"


def test_goal_renderer_uses_canonical_colors_with_filament_matte_calibration() -> None:
    assert GOAL_FILAMENT_MATERIALS["part_canonical"].color == VISUAL_SERVO_PART_COLOR
    assert GOAL_FILAMENT_MATERIALS["part_canonical"].roughness > VISUAL_SERVO_PART_ROUGHNESS
    assert GOAL_FILAMENT_MATERIALS["pdz_finger_black"].color == VISUAL_SERVO_FINGER_COLOR
    assert GOAL_FILAMENT_MATERIALS["pdz_finger_black"].roughness > VISUAL_SERVO_FINGER_ROUGHNESS
    pad = GOAL_FILAMENT_MATERIALS["pdz_contact_white"]
    assert pad.color == VISUAL_SERVO_CONTACT_PAD_COLOR
    assert pad.roughness > VISUAL_SERVO_CONTACT_PAD_ROUGHNESS
    assert pad.emission == 0.0


def test_finger_material_classifier_binds_leaf_pad_geometry_white() -> None:
    root = "/World/envs/env_0/Robot/pdz_gripper_left_finger_link"
    assert classify_robot_finger_geometry_material(root, "Xform") is None
    assert (
        classify_robot_finger_geometry_material(
            f"{root}/visuals/left_finger/node_STL_BINARY_0", "Mesh"
        )
        == "black_pla"
    )
    assert (
        classify_robot_finger_geometry_material(
            f"{root}/visuals/left_tpu_pad/node_STL_BINARY_0", "Mesh"
        )
        == "white_contact_pad"
    )
    assert (
        classify_robot_finger_geometry_material(
            "/World/envs/env_0/Robot/link7/visuals/node_STL_BINARY_0", "Mesh"
        )
        is None
    )


def test_generated_pdz_urdf_defaults_to_black_fingers_and_white_pads() -> None:
    urdf = (
        Path(__file__).resolve().parents[1]
        / "assets/urdf/kuka_iiwa7_pdz_gripper/urdf/kuka_iiwa7_pdz_gripper.urdf"
    )
    root = ET.parse(urdf).getroot()
    observed: dict[str, tuple[float, float, float]] = {}
    for link_name in (
        "pdz_gripper_left_finger_link",
        "pdz_gripper_right_finger_link",
    ):
        link = root.find(f"link[@name='{link_name}']")
        assert link is not None
        for visual in link.findall("visual"):
            material = visual.find("material")
            color = visual.find("material/color")
            assert material is not None and color is not None
            observed[str(material.get("name"))] = tuple(
                float(value) for value in str(color.get("rgba")).split()[:3]
            )
    assert observed["pdz_finger_black"] == VISUAL_SERVO_FINGER_COLOR
    assert observed["pdz_contact_white"] == VISUAL_SERVO_CONTACT_PAD_COLOR
