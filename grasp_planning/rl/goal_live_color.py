"""Goal/live part-color relationship sampling for RGB-D visual servoing."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from grasp_planning.isaac_visual_materials import VISUAL_SERVO_PART_PALETTE

COLOR_RELATIONSHIP_MATCH = 0
COLOR_RELATIONSHIP_SIMILAR = 1
COLOR_RELATIONSHIP_DIFFERENT = 2
COLOR_RELATIONSHIP_NAMES = ("match", "similar", "different")


@dataclass(frozen=True)
class GoalLiveColorPairs:
    goal_variant_slots: torch.Tensor
    goal_palette_indices: torch.Tensor
    live_palette_indices: torch.Tensor
    relationship_codes: torch.Tensor


def sample_goal_live_color_pairs(
    count: int,
    available_goal_palette_indices: torch.Tensor,
    *,
    match_fraction: float = 0.25,
    similar_fraction: float = 0.20,
    device: torch.device | str | None = None,
) -> GoalLiveColorPairs:
    """Sample 25% matching, 20% similar, and 55% clearly different pairs."""

    sample_count = int(count)
    if sample_count < 1:
        raise ValueError("count must be positive.")
    if not 0.0 <= match_fraction <= 1.0 or not 0.0 <= similar_fraction <= 1.0:
        raise ValueError("Color relationship fractions must lie in [0, 1].")
    if match_fraction + similar_fraction > 1.0:
        raise ValueError("match_fraction + similar_fraction must not exceed 1.")
    target_device = torch.device(device) if device is not None else available_goal_palette_indices.device
    available = available_goal_palette_indices.to(device=target_device, dtype=torch.long).flatten()
    if available.numel() < 3 or torch.unique(available).numel() != available.numel():
        raise ValueError("At least three unique goal palette indices are required.")
    palette_size = len(VISUAL_SERVO_PART_PALETTE)
    if torch.any((available < 0) | (available >= palette_size)):
        raise ValueError("A goal palette index lies outside VISUAL_SERVO_PART_PALETTE.")

    palette_rgb = torch.tensor(
        [entry.color for entry in VISUAL_SERVO_PART_PALETTE],
        dtype=torch.float32,
        device=target_device,
    )
    variant_slots = torch.randint(available.numel(), (sample_count,), device=target_device)
    goal_indices = available[variant_slots]
    distances = torch.linalg.norm(palette_rgb[goal_indices, None, :] - palette_rgb[None, :, :], dim=-1)
    ordering = torch.argsort(distances, dim=1)
    similar_candidates = ordering[:, 1:5]
    different_candidates = ordering[:, palette_size // 2 :]
    similar_choice = similar_candidates[
        torch.arange(sample_count, device=target_device),
        torch.randint(similar_candidates.shape[1], (sample_count,), device=target_device),
    ]
    different_choice = different_candidates[
        torch.arange(sample_count, device=target_device),
        torch.randint(different_candidates.shape[1], (sample_count,), device=target_device),
    ]
    draws = torch.rand(sample_count, device=target_device)
    relationship = torch.full((sample_count,), COLOR_RELATIONSHIP_DIFFERENT, dtype=torch.long, device=target_device)
    relationship[draws < match_fraction] = COLOR_RELATIONSHIP_MATCH
    relationship[(draws >= match_fraction) & (draws < match_fraction + similar_fraction)] = COLOR_RELATIONSHIP_SIMILAR
    live_indices = torch.where(
        relationship == COLOR_RELATIONSHIP_MATCH,
        goal_indices,
        torch.where(
            relationship == COLOR_RELATIONSHIP_SIMILAR,
            similar_choice,
            different_choice,
        ),
    )
    return GoalLiveColorPairs(
        goal_variant_slots=variant_slots,
        goal_palette_indices=goal_indices,
        live_palette_indices=live_indices,
        relationship_codes=relationship,
    )


__all__ = [
    "COLOR_RELATIONSHIP_DIFFERENT",
    "COLOR_RELATIONSHIP_MATCH",
    "COLOR_RELATIONSHIP_NAMES",
    "COLOR_RELATIONSHIP_SIMILAR",
    "GoalLiveColorPairs",
    "sample_goal_live_color_pairs",
]
