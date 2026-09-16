from __future__ import annotations

import pytest

from grasp_planning.rl.policy_timing import (
    PHYSICS_RATE_HZ,
    POLICY_DECIMATION,
    POLICY_PERIOD_S,
    POLICY_RATE_HZ,
    PolicyRateGate,
    temporal_reward_scale,
)


def test_shared_policy_rate_is_exactly_15_hz() -> None:
    assert POLICY_RATE_HZ == 15.0
    assert POLICY_PERIOD_S == pytest.approx(1.0 / 15.0)
    assert PHYSICS_RATE_HZ / POLICY_DECIMATION == POLICY_RATE_HZ
    assert POLICY_DECIMATION == 8


def test_rate_gate_accepts_every_other_30_hz_camera_frame() -> None:
    gate = PolicyRateGate()
    accepted = [gate.accept(index / 30.0) for index in range(7)]

    assert accepted == [True, False, True, False, True, False, True]


def test_rate_gate_rejects_timestamp_regression() -> None:
    gate = PolicyRateGate()
    assert gate.accept(1.0)

    with pytest.raises(ValueError, match="monotonically"):
        gate.accept(0.9)


def test_temporal_penalties_double_when_rate_halves_from_30_to_15_hz() -> None:
    assert temporal_reward_scale(1.0 / 30.0) == pytest.approx(1.0)
    assert temporal_reward_scale(1.0 / 15.0) == pytest.approx(2.0)
