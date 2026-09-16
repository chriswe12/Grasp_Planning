from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/object_pose_sampling.py"
)
SPEC = importlib.util.spec_from_file_location("object_pose_sampling", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
object_pose_sampling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(object_pose_sampling)
apply_planar_object_pose_delta = object_pose_sampling.apply_planar_object_pose_delta
sample_collision_safe_yaw_offsets_from_profile = (
    object_pose_sampling.sample_collision_safe_yaw_offsets_from_profile
)
yaw_offset_profile = object_pose_sampling.yaw_offset_profile


def _rotation_z(yaw: float) -> torch.Tensor:
    return torch.tensor(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_yaw_profile_tapers_from_far_to_near() -> None:
    result = yaw_offset_profile(
        torch.tensor([0.0, 0.5, 1.0]),
        far_yaw_rad=math.radians(10.0),
        near_yaw_rad=math.radians(3.0),
        exponent=1.0,
    )

    assert torch.rad2deg(result).tolist() == pytest.approx([10.0, 6.5, 3.0])


def test_yaw_and_translation_jointly_preserve_validated_clearance() -> None:
    clearance = torch.tensor([0.050, 0.010, 0.004, 0.050])
    translation = torch.tensor([0.010, 0.006, 0.0029, 0.010])
    radius = torch.tensor([0.080, 0.080, 0.080, 0.080])
    zero = torch.tensor([False, False, False, True])

    yaw, requested, safe_cap = sample_collision_safe_yaw_offsets_from_profile(
        torch.full((4,), math.radians(15.0)),
        clearance,
        translation,
        radius,
        zero,
        minimum_collision_clearance_m=0.001,
        clearance_guard_m=0.0001,
        magnitude_unit_samples=torch.ones(4),
        sign_unit_samples=torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )

    rotational_displacement = 2.0 * radius * torch.sin(0.5 * yaw.abs())
    assert torch.all(translation + rotational_displacement <= clearance - 0.0011 + 1.0e-8)
    assert yaw[0] < 0.0
    assert yaw[1] > 0.0
    assert abs(float(yaw[2])) < math.radians(0.01)
    assert yaw[3] == 0.0
    assert requested[3] == 0.0
    assert safe_cap[1] < math.radians(15.0)


def test_rigid_part_delta_preserves_target_transform_in_part_frame() -> None:
    object_position = torch.tensor([[0.40, 0.10, 0.02]])
    object_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    target_position = torch.tensor([[0.44, 0.11, 0.08]])
    target_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    translation = torch.tensor([[0.007, -0.004, 0.0]])
    yaw = torch.tensor([math.radians(12.0)])

    moved_object_p, moved_object_q, moved_target_p, moved_target_q = apply_planar_object_pose_delta(
        object_position,
        object_quaternion,
        target_position,
        target_quaternion,
        translation,
        yaw,
    )

    nominal_relative = target_position[0] - object_position[0]
    moved_relative_in_part = _rotation_z(-float(yaw[0])) @ (moved_target_p[0] - moved_object_p[0])
    torch.testing.assert_close(moved_relative_in_part, nominal_relative)
    torch.testing.assert_close(moved_object_p, object_position + translation)
    torch.testing.assert_close(moved_object_q, moved_target_q)
    assert moved_object_q[0, 0] == pytest.approx(math.cos(0.5 * float(yaw[0])))


def test_planar_part_delta_rejects_unstable_vertical_translation() -> None:
    with pytest.raises(ValueError, match="cannot change world Z"):
        apply_planar_object_pose_delta(
            torch.zeros((1, 3)),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.zeros((1, 3)),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.001]]),
            torch.zeros(1),
        )
