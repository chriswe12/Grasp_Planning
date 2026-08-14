from __future__ import annotations

import json

import numpy as np
import pytest

from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
)
from grasp_planning.isaac_visual_materials import VISUAL_SERVO_MATERIAL_PROFILE
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


def test_canonical_scene_is_dark_matte_with_a_stronger_shape_key() -> None:
    assert "dome_directional_dlaa" in VISUAL_SERVO_SCENE_PROFILE
    assert "readable_black_fingers" in VISUAL_SERVO_MATERIAL_PROFILE
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
    assert D405_VISUAL_SERVO_OBSERVATION_PROFILE == "rgbd_render_256x144_area_128x72_v2"
