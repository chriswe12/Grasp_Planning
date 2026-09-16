from __future__ import annotations

import math

import pytest

from grasp_planning.isaac_visual_materials import (
    VISUAL_SERVO_CONTACT_PAD_COLOR,
    VISUAL_SERVO_CONTACT_PAD_ROUGHNESS,
    VISUAL_SERVO_FINGER_COLOR,
    VISUAL_SERVO_FINGER_ROUGHNESS,
    VISUAL_SERVO_PART_COLOR,
    VISUAL_SERVO_WORK_SURFACE_COLOR,
)
from grasp_planning.isaac_visual_scene import (
    VISUAL_SERVO_DOME_INTENSITY,
    VISUAL_SERVO_KEY_INTENSITY,
    VISUAL_SERVO_KEY_ROTATION_WXYZ,
)
from grasp_planning.rl.scene_appearance_randomization import (
    SceneAppearanceRandomizationCfg,
    sample_scene_appearance,
)


def test_midpoint_sample_preserves_canonical_direction_and_brightness() -> None:
    cfg = SceneAppearanceRandomizationCfg(
        key_angle_deg=(8.0, 8.0),
        part_roughness=(0.8, 0.8),
        gripper_canonical_fraction=1.0,
        finger_roughness=(VISUAL_SERVO_FINGER_ROUGHNESS, VISUAL_SERVO_FINGER_ROUGHNESS),
        pad_roughness=(VISUAL_SERVO_CONTACT_PAD_ROUGHNESS, VISUAL_SERVO_CONTACT_PAD_ROUGHNESS),
        ground_roughness=(0.9, 0.9),
    )

    sample = sample_scene_appearance(cfg, [0.5] * 19)

    assert sample.key_yaw_delta_deg == pytest.approx(0.0)
    assert sample.key_pitch_delta_deg == pytest.approx(0.0)
    assert sample.key_orientation_wxyz == pytest.approx(VISUAL_SERVO_KEY_ROTATION_WXYZ)
    assert sample.key_intensity == pytest.approx(VISUAL_SERVO_KEY_INTENSITY)
    assert sample.dome_intensity == pytest.approx(VISUAL_SERVO_DOME_INTENSITY)
    assert sample.part_color == pytest.approx(VISUAL_SERVO_PART_COLOR)
    assert sample.finger_color == pytest.approx(VISUAL_SERVO_FINGER_COLOR)
    assert sample.finger_roughness == pytest.approx(VISUAL_SERVO_FINGER_ROUGHNESS)
    assert sample.pad_color == pytest.approx(VISUAL_SERVO_CONTACT_PAD_COLOR)
    assert sample.pad_roughness == pytest.approx(VISUAL_SERVO_CONTACT_PAD_ROUGHNESS)
    assert sample.ground_color == pytest.approx(VISUAL_SERVO_WORK_SURFACE_COLOR)


def test_extreme_samples_move_shadows_and_stay_bounded() -> None:
    cfg = SceneAppearanceRandomizationCfg()

    low_values = [0.0] * 19
    high_values = [1.0] * 19
    low_values[18] = 1.0
    low = sample_scene_appearance(cfg, low_values)
    high = sample_scene_appearance(cfg, high_values)

    assert low.key_yaw_delta_deg == -35.0
    assert high.key_yaw_delta_deg == 35.0
    assert low.key_pitch_delta_deg == -15.0
    assert high.key_pitch_delta_deg == 15.0
    assert low.key_orientation_wxyz != pytest.approx(high.key_orientation_wxyz)
    assert math.sqrt(sum(value * value for value in low.key_orientation_wxyz)) == pytest.approx(1.0)
    assert math.sqrt(sum(value * value for value in high.key_orientation_wxyz)) == pytest.approx(1.0)
    assert low.key_intensity == pytest.approx(0.7 * VISUAL_SERVO_KEY_INTENSITY)
    assert high.key_intensity == pytest.approx(1.3 * VISUAL_SERVO_KEY_INTENSITY)
    for color in (
        low.key_color,
        high.key_color,
        low.part_color,
        high.part_color,
        low.finger_color,
        high.finger_color,
        low.pad_color,
        high.pad_color,
        low.ground_color,
        high.ground_color,
    ):
        assert all(0.0 <= channel <= 1.0 for channel in color)


@pytest.mark.parametrize(
    "cfg, message",
    (
        (SceneAppearanceRandomizationCfg(interval_steps=0), "interval_steps"),
        (
            SceneAppearanceRandomizationCfg(key_yaw_delta_deg=(1.0, -1.0)),
            "key_yaw_delta_deg",
        ),
        (
            SceneAppearanceRandomizationCfg(part_roughness=(-0.1, 0.8)),
            "part_roughness",
        ),
        (
            SceneAppearanceRandomizationCfg(pad_roughness=(0.6, 1.1)),
            "pad_roughness",
        ),
        (
            SceneAppearanceRandomizationCfg(gripper_canonical_fraction=1.1),
            "gripper_canonical_fraction",
        ),
    ),
)
def test_invalid_scene_appearance_config_is_rejected(
    cfg: SceneAppearanceRandomizationCfg,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_scene_appearance_requires_exact_unit_sample_vector() -> None:
    cfg = SceneAppearanceRandomizationCfg()

    with pytest.raises(ValueError, match="Expected 19"):
        sample_scene_appearance(cfg, [0.5] * 18)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sample_scene_appearance(cfg, [0.5] * 18 + [1.1])


def test_hue_samples_change_part_and_ground_color() -> None:
    cfg = SceneAppearanceRandomizationCfg(
        part_color_scale=(1.0, 1.0),
        ground_color_scale=(1.0, 1.0),
    )
    low_values = [0.5] * 19
    high_values = [0.5] * 19
    low_values[11] = 0.0
    low_values[12] = 0.0
    high_values[11] = 1.0
    high_values[12] = 1.0

    low = sample_scene_appearance(cfg, low_values)
    high = sample_scene_appearance(cfg, high_values)

    assert low.part_color != pytest.approx(high.part_color)
    assert low.ground_color != pytest.approx(high.ground_color)


def test_gripper_samples_stay_identifiable_but_change_material_response() -> None:
    cfg = SceneAppearanceRandomizationCfg()
    low_values = [0.0] * 19
    high_values = [1.0] * 19
    low_values[18] = 1.0
    low = sample_scene_appearance(cfg, low_values)
    high = sample_scene_appearance(cfg, high_values)

    assert low.finger_color != pytest.approx(high.finger_color)
    assert low.finger_roughness == pytest.approx(0.30)
    assert high.finger_roughness == pytest.approx(0.70)
    assert low.pad_color != pytest.approx(high.pad_color)
    assert low.pad_roughness == pytest.approx(0.55)
    assert high.pad_roughness == pytest.approx(0.88)
    assert max(high.finger_color) <= 0.07
    assert min(low.pad_color) > 0.80
    assert not low.gripper_canonical
    assert not high.gripper_canonical


def test_gripper_canonical_mixture_preserves_exact_materials() -> None:
    cfg = SceneAppearanceRandomizationCfg(gripper_canonical_fraction=0.20)
    values = [1.0] * 19
    values[18] = 0.10

    sample = sample_scene_appearance(cfg, values)

    assert sample.gripper_canonical
    assert sample.finger_color == pytest.approx(VISUAL_SERVO_FINGER_COLOR)
    assert sample.finger_roughness == pytest.approx(VISUAL_SERVO_FINGER_ROUGHNESS)
    assert sample.pad_color == pytest.approx(VISUAL_SERVO_CONTACT_PAD_COLOR)
    assert sample.pad_roughness == pytest.approx(VISUAL_SERVO_CONTACT_PAD_ROUGHNESS)
