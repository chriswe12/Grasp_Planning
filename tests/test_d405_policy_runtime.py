from __future__ import annotations

import json

import numpy as np
import torch

from grasp_planning.rl.d405_policy_runtime import (
    POLICY_FULL_INPUT_SIZE,
    POLICY_IMAGE_VALUE_COUNT,
    assemble_policy_observation,
    ros_depth_z16_to_metres,
    write_checkpoint_metadata_template,
)
from grasp_planning.rl.policy_context import policy_observation_size
from grasp_planning.rl.policy_timing import POLICY_RATE_HZ


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


def test_policy_observation_packs_twist_and_camera_rotation_context() -> None:
    image = torch.zeros((1, 72, 128, 4), dtype=torch.float32)
    previous = np.arange(6, dtype=np.float32)
    twist = -previous
    rotation = np.eye(3, dtype=np.float32)

    observation = assemble_policy_observation(
        image,
        image,
        previous,
        policy_context_mode="action_twist_rotation",
        normalized_tcp_twist_camera=twist,
        rotation_base_from_camera=rotation,
    )

    expected_size = policy_observation_size(
        "action_twist_rotation",
        image_value_count=POLICY_IMAGE_VALUE_COUNT,
    )
    assert tuple(observation.shape) == (1, expected_size)
    context = observation[0, POLICY_IMAGE_VALUE_COUNT : POLICY_IMAGE_VALUE_COUNT + 18]
    torch.testing.assert_close(context[:6], torch.from_numpy(previous))
    torch.testing.assert_close(context[6:12], torch.from_numpy(twist))
    torch.testing.assert_close(context[12:], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_checkpoint_sidecar_binds_completion_gate(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    catalog = tmp_path / "catalog.npz"
    checkpoint.write_bytes(b"checkpoint")
    catalog.write_bytes(b"catalog")

    output = write_checkpoint_metadata_template(
        output_path=tmp_path / "deployment.json",
        checkpoint_path=checkpoint,
        goal_catalog_path=catalog,
        completion_probability_threshold=0.97,
        completion_required_consecutive_steps=6,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["completion_probability_threshold"] == 0.97
    assert payload["completion_required_consecutive_steps"] == 6
    assert payload["policy_context_mode"] == "action"
    assert payload["policy_context_size"] == 6
    assert payload["policy_rate_hz"] == POLICY_RATE_HZ
    assert payload["action_delta_limit"] == 0.50


def test_checkpoint_sidecar_records_extended_policy_context(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    catalog = tmp_path / "catalog.npz"
    checkpoint.write_bytes(b"checkpoint")
    catalog.write_bytes(b"catalog")

    output = write_checkpoint_metadata_template(
        output_path=tmp_path / "deployment.json",
        checkpoint_path=checkpoint,
        goal_catalog_path=catalog,
        policy_context_mode="action_twist_rotation",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["policy_context_mode"] == "action_twist_rotation"
    assert payload["policy_context_size"] == 18
    assert payload["network_input_size"] == POLICY_FULL_INPUT_SIZE + 12


def test_checkpoint_sidecar_rejects_non_contract_policy_rate(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    catalog = tmp_path / "catalog.npz"
    checkpoint.write_bytes(b"checkpoint")
    catalog.write_bytes(b"catalog")

    try:
        write_checkpoint_metadata_template(
            output_path=tmp_path / "deployment.json",
            checkpoint_path=checkpoint,
            goal_catalog_path=catalog,
            policy_rate_hz=30.0,
        )
    except ValueError as exc:
        assert "15.0 Hz" in str(exc)
    else:
        raise AssertionError("Expected non-contract policy rate to be rejected.")
