from __future__ import annotations

import torch

from grasp_planning.rl.lift_reward import (
    lift_outcome_reward,
    physical_pickup_success,
    retained_lift_quality,
)


def test_retained_lift_quality_rejects_close_bump_and_rewards_retention() -> None:
    quality = retained_lift_quality(
        final_lift_m=torch.tensor([0.005, 0.040, 0.040, 0.040]),
        peak_lift_m=torch.tensor([0.005, 0.040, 0.060, 0.040]),
        relative_drift_m=torch.tensor([0.0, 0.0, 0.0, 0.030]),
        arm_lift_ok=torch.tensor([True, True, True, True]),
        minimum_credit_lift_m=0.005,
        full_credit_lift_m=0.040,
        drift_scale_m=0.015,
        drop_scale_m=0.010,
    )

    assert quality[0] == 0.0
    assert quality[1] == 1.0
    assert quality[2] < 0.03
    assert 0.25 < quality[3] < 0.30


def test_pickup_success_requires_height_retention_and_arm_motion() -> None:
    result = physical_pickup_success(
        final_lift_m=torch.tensor([0.025, 0.024, 0.030, 0.030]),
        peak_lift_m=torch.tensor([0.030, 0.030, 0.050, 0.030]),
        relative_drift_m=torch.tensor([0.030, 0.0, 0.0, 0.0]),
        arm_lift_ok=torch.tensor([True, True, True, False]),
        minimum_final_lift_m=0.025,
        maximum_relative_drift_m=0.030,
        maximum_peak_drop_m=0.015,
    )

    assert result.tolist() == [True, False, False, False]


def test_lift_outcome_truth_table_only_penalizes_neither() -> None:
    reward = lift_outcome_reward(
        geometric_success=torch.tensor([True, False, True, False]),
        pickup_success=torch.tensor([False, True, True, False]),
        lift_quality=torch.tensor([0.0, 1.0, 1.0, 0.0]),
        lift_quality_reward=25.0,
        geometric_lift_bonus=10.0,
        neither_penalty=30.0,
    )

    # Geometric-only reward is emitted at commit, so the later outcome is 0.
    assert reward.tolist() == [0.0, 25.0, 35.0, -30.0]


def test_partial_motion_without_pickup_cannot_turn_neither_into_positive_reward() -> None:
    reward = lift_outcome_reward(
        geometric_success=torch.tensor([False, True]),
        pickup_success=torch.tensor([False, False]),
        lift_quality=torch.tensor([1.0, 1.0]),
        lift_quality_reward=55.0,
        geometric_lift_bonus=15.0,
        neither_penalty=30.0,
    )

    # A failed pickup stays negative outside the target region. A target-pose
    # commit was rewarded earlier and is therefore neutral at outcome time.
    assert reward.tolist() == [-30.0, 0.0]
