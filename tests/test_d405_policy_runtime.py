from __future__ import annotations

import json

import numpy as np
import torch

from grasp_planning.d405_wrist_camera import D405_VISUAL_SERVO_CAMERA_PROFILE
from grasp_planning.rl.d405_policy_runtime import (
    POLICY_FULL_INPUT_SIZE,
    POLICY_IMAGE_VALUE_COUNT,
    D405RuntimeGoal,
    assemble_policy_observation,
    ros_depth_z16_to_metres,
    write_checkpoint_metadata_template,
)
from grasp_planning.rl.policy_context import policy_observation_size


def test_ros_z16_millimetres_are_converted_exactly_once() -> None:
    depth = ros_depth_z16_to_metres(np.asarray([[0, 250, 500]], dtype=np.uint16))

    np.testing.assert_allclose(depth, [[0.0, 0.250, 0.500]], atol=1.0e-7)
    assert depth.dtype == np.float32


def test_policy_observation_has_exact_deployment_and_placeholder_layout() -> None:
    live = torch.zeros((1, 72, 128, 4), dtype=torch.float32)
    goal = torch.ones((1, 72, 128, 4), dtype=torch.float32)
    previous = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6)

    observation = assemble_policy_observation(live, goal, previous)

    assert tuple(observation.shape) == (1, POLICY_FULL_INPUT_SIZE)
    assert POLICY_IMAGE_VALUE_COUNT == 73_728
    packed_image = observation[:, :POLICY_IMAGE_VALUE_COUNT].view(1, 72, 128, 8)
    assert torch.count_nonzero(packed_image[..., :4]) == 0
    assert torch.all(packed_image[..., 4:] == 1.0)
    torch.testing.assert_close(
        observation[0, POLICY_IMAGE_VALUE_COUNT : POLICY_IMAGE_VALUE_COUNT + 6],
        torch.tensor(previous),
    )
    assert torch.count_nonzero(observation[:, -8:]) == 0


def test_policy_observation_rejects_non_finite_action_context() -> None:
    image = torch.zeros((1, 72, 128, 4), dtype=torch.float32)

    try:
        assemble_policy_observation(image, image, (0.0, 0.0, 0.0, 0.0, 0.0, float("nan")))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Expected non-finite deployment context to be rejected.")


def test_policy_observation_supports_velocity_rotation_context() -> None:
    image = torch.zeros((1, 72, 128, 4), dtype=torch.float32)
    observation = assemble_policy_observation(
        image,
        image,
        (0.0,) * 6,
        policy_context_mode="action_twist_rotation",
        normalized_tcp_twist_camera=(0.1,) * 6,
        rotation_base_from_camera=np.eye(3),
    )

    assert tuple(observation.shape) == (
        1,
        policy_observation_size(
            "action_twist_rotation",
            image_value_count=POLICY_IMAGE_VALUE_COUNT,
        ),
    )


def test_checkpoint_sidecar_binds_completion_gate(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")

    output = write_checkpoint_metadata_template(
        output_path=tmp_path / "deployment.json",
        checkpoint_path=checkpoint,
        completion_probability_threshold=0.97,
        completion_required_consecutive_steps=6,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["completion_probability_threshold"] == 0.97
    assert payload["completion_required_consecutive_steps"] == 6
    assert payload["policy_context_mode"] == "action"
    assert payload["policy_context_size"] == 6
    assert payload["policy_rate_hz"] == 15.0
    assert "goal_catalog_sha256" not in payload


def test_runtime_goal_loads_one_on_demand_render_without_catalogue(tmp_path) -> None:
    path = tmp_path / "runtime_goal.npz"
    np.savez_compressed(
        path,
        goal_id=np.asarray("runtime__part_2__g0584"),
        part_id=np.asarray("2"),
        grasp_id=np.asarray("g0584"),
        jaw_width_m=np.asarray(0.04, dtype=np.float32),
        goal_rgb=np.full((144, 256, 3), 127, dtype=np.uint8),
        goal_depth=np.full((144, 256), 0.25, dtype=np.float32),
        goal_camera_profile=np.asarray(D405_VISUAL_SERVO_CAMERA_PROFILE),
        goal_observation_profile=np.asarray("rgbd_render_256x144_valid_area_128x72_d405_range_v3"),
        render_validation_passed=np.asarray(True, dtype=np.bool_),
    )

    loaded = D405RuntimeGoal(path).load(expected_grasp_id="g0584", expected_part_id="2")

    assert loaded.goal_id == "runtime__part_2__g0584"
    assert loaded.grasp_id == "g0584"
    assert tuple(loaded.goal_rgbd.shape) == (1, 72, 128, 4)
