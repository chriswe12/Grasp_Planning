from __future__ import annotations

import numpy as np
import pytest
import torch

from grasp_planning.rl.policy_context import (
    assemble_policy_context_torch,
    resolve_policy_context,
    rotation_matrix_to_6d_numpy,
    rotation_matrix_to_6d_torch,
)


@pytest.mark.parametrize(
    ("mode", "size"),
    (("action", 6), ("action_twist", 12), ("action_twist_rotation", 18)),
)
def test_policy_context_sizes_are_explicit(mode: str, size: int) -> None:
    assert resolve_policy_context(mode).size == size


def test_rotation_6d_is_first_two_columns_not_row_interleaved() -> None:
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    expected = np.asarray([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)

    np.testing.assert_array_equal(rotation_matrix_to_6d_numpy(rotation), expected)
    torch.testing.assert_close(rotation_matrix_to_6d_torch(torch.from_numpy(rotation)), torch.from_numpy(expected))


def test_full_policy_context_preserves_requested_layout() -> None:
    previous = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    twist = -previous
    rotation = torch.eye(3).unsqueeze(0)

    context = assemble_policy_context_torch(
        "action_twist_rotation",
        previous,
        normalized_tcp_twist_camera=twist,
        rotation_base_from_camera=rotation,
    )

    assert tuple(context.shape) == (1, 18)
    torch.testing.assert_close(context[:, :6], previous)
    torch.testing.assert_close(context[:, 6:12], twist)
    torch.testing.assert_close(
        context[:, 12:],
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
    )
