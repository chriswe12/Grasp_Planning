"""Deployment-measurable actor-context layouts for visual-servo policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

POLICY_CONTEXT_ACTION = "action"
POLICY_CONTEXT_ACTION_TWIST = "action_twist"
POLICY_CONTEXT_ACTION_TWIST_ROTATION = "action_twist_rotation"
POLICY_CONTEXT_CHOICES = (
    POLICY_CONTEXT_ACTION,
    POLICY_CONTEXT_ACTION_TWIST,
    POLICY_CONTEXT_ACTION_TWIST_ROTATION,
)

POLICY_PREVIOUS_ACTION_SIZE = 6
POLICY_TCP_TWIST_SIZE = 6
POLICY_CAMERA_ROTATION_SIZE = 6


@dataclass(frozen=True)
class PolicyContextSpec:
    """One explicit, checkpoint-compatible actor-context contract."""

    name: str
    size: int
    uses_tcp_twist: bool
    uses_camera_rotation: bool


def resolve_policy_context(mode: str) -> PolicyContextSpec:
    normalized = str(mode).strip().lower()
    if normalized == POLICY_CONTEXT_ACTION:
        return PolicyContextSpec(normalized, 6, False, False)
    if normalized == POLICY_CONTEXT_ACTION_TWIST:
        return PolicyContextSpec(normalized, 12, True, False)
    if normalized == POLICY_CONTEXT_ACTION_TWIST_ROTATION:
        return PolicyContextSpec(normalized, 18, True, True)
    raise ValueError(
        f"Unknown policy context '{mode}'. Expected one of {', '.join(POLICY_CONTEXT_CHOICES)}."
    )


def policy_observation_size(
    mode: str,
    *,
    image_value_count: int,
    privileged_label_size: int = 8,
) -> int:
    """Return the full rollout observation size for one context contract."""

    if image_value_count <= 0 or privileged_label_size < 0:
        raise ValueError("Observation component sizes must be non-negative and images must be present.")
    return int(image_value_count) + resolve_policy_context(mode).size + int(privileged_label_size)


def rotation_matrix_to_6d_torch(rotation: torch.Tensor) -> torch.Tensor:
    """Return the first two matrix columns, concatenated as a continuous 6D rotation."""

    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrix must end in 3x3, got {tuple(rotation.shape)}.")
    return rotation[..., :, :2].transpose(-2, -1).reshape(*rotation.shape[:-2], 6)


def rotation_matrix_to_6d_numpy(rotation: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """NumPy equivalent of :func:`rotation_matrix_to_6d_torch`."""

    matrix = np.asarray(rotation, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Camera rotation context must be one finite 3x3 matrix.")
    return matrix[:, :2].T.reshape(6)


def assemble_policy_context_torch(
    mode: str,
    previous_applied_action: torch.Tensor,
    *,
    normalized_tcp_twist_camera: torch.Tensor | None = None,
    rotation_base_from_camera: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assemble a batched actor context using only deployment-available state."""

    spec = resolve_policy_context(mode)
    if previous_applied_action.shape[-1] != POLICY_PREVIOUS_ACTION_SIZE:
        raise ValueError("Previous action context must end in six values.")
    values = [previous_applied_action]
    if spec.uses_tcp_twist:
        if normalized_tcp_twist_camera is None or normalized_tcp_twist_camera.shape != previous_applied_action.shape:
            raise ValueError("This policy context requires a normalized six-value camera-frame TCP twist.")
        values.append(normalized_tcp_twist_camera)
    if spec.uses_camera_rotation:
        if rotation_base_from_camera is None or rotation_base_from_camera.shape[-2:] != (3, 3):
            raise ValueError("This policy context requires a base-from-camera 3x3 rotation matrix.")
        values.append(rotation_matrix_to_6d_torch(rotation_base_from_camera))
    context = torch.cat(values, dim=-1)
    if context.shape[-1] != spec.size or not torch.isfinite(context).all():
        raise ValueError("Policy context has an invalid shape or contains non-finite values.")
    return context


__all__ = [
    "POLICY_CAMERA_ROTATION_SIZE",
    "POLICY_CONTEXT_ACTION",
    "POLICY_CONTEXT_ACTION_TWIST",
    "POLICY_CONTEXT_ACTION_TWIST_ROTATION",
    "POLICY_CONTEXT_CHOICES",
    "POLICY_PREVIOUS_ACTION_SIZE",
    "POLICY_TCP_TWIST_SIZE",
    "PolicyContextSpec",
    "assemble_policy_context_torch",
    "policy_observation_size",
    "resolve_policy_context",
    "rotation_matrix_to_6d_numpy",
    "rotation_matrix_to_6d_torch",
]
