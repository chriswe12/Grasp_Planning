"""Deterministic multi-branch IK seeds for the seven-axis KUKA iiwa arms."""

from __future__ import annotations

import math
from typing import Sequence

KUKA_IIWA_JOINT_LOWER_LIMITS_RAD = (-2.97, -2.09, -2.97, -2.09, -2.97, -2.09, -3.05)
KUKA_IIWA_JOINT_UPPER_LIMITS_RAD = (2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05)
KUKA_IIWA_A7_NEAR_LIMIT_BRANCH_RAD = 3.0


def kuka_iiwa_ik_seed_candidates(
    start_joint_positions: Sequence[float],
    *,
    preferred_joint_positions: Sequence[float] | None = None,
    candidate_count: int = 7,
    perturbation_rad: float = 0.60,
) -> tuple[tuple[float, ...], ...]:
    """Return bounded, deterministic seeds that probe useful iiwa IK branches."""

    start = tuple(float(value) for value in start_joint_positions)
    lower = KUKA_IIWA_JOINT_LOWER_LIMITS_RAD
    upper = KUKA_IIWA_JOINT_UPPER_LIMITS_RAD
    if len(start) != len(lower):
        raise ValueError(f"Expected {len(lower)} iiwa joints, got {len(start)}.")
    if any(value < low or value > high for value, low, high in zip(start, lower, upper)):
        raise ValueError("iiwa IK start state is outside the configured joint limits.")

    count = max(1, int(candidate_count))
    seeds: list[tuple[float, ...]] = []

    def add_seed(raw_seed: Sequence[float]) -> None:
        if len(seeds) >= count:
            return
        seed = tuple(float(value) for value in raw_seed)
        if len(seed) != len(start) or any(
            value < low or value > high for value, low, high in zip(seed, lower, upper)
        ):
            return
        if not any(max(abs(a - b) for a, b in zip(seed, old)) < 1.0e-9 for old in seeds):
            seeds.append(seed)

    add_seed(start)
    if preferred_joint_positions is not None:
        if len(preferred_joint_positions) != len(start):
            raise ValueError(f"Expected {len(start)} preferred iiwa joints, got {len(preferred_joint_positions)}.")
        add_seed(preferred_joint_positions)

    for offset in (math.pi, -math.pi):
        branch = list(start)
        branch[-1] += offset
        add_seed(branch)
    for target in (KUKA_IIWA_A7_NEAR_LIMIT_BRANCH_RAD, -KUKA_IIWA_A7_NEAR_LIMIT_BRANCH_RAD):
        branch = list(start)
        branch[-1] = target
        add_seed(branch)
    for joint_index, scale in ((0, 1.0), (2, 1.0), (3, 0.75), (4, 1.0)):
        for direction in (1.0, -1.0):
            branch = list(start)
            branch[joint_index] += direction * float(perturbation_rad) * scale
            add_seed(branch)

    golden_ratio = 0.5 * (1.0 + math.sqrt(5.0))
    sample_index = 1
    while len(seeds) < count and sample_index <= count * 8:
        add_seed(
            tuple(
                value
                + float(perturbation_rad)
                * ((((sample_index * (joint_index + 1)) / golden_ratio) % 1.0) * 2.0 - 1.0)
                for joint_index, value in enumerate(start)
            )
        )
        sample_index += 1
    return tuple(seeds)
