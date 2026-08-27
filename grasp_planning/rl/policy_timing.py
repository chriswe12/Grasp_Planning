"""Shared simulation, training, and deployment timing contract."""

from __future__ import annotations

import math

POLICY_RATE_HZ = 15.0
POLICY_PERIOD_S = 1.0 / POLICY_RATE_HZ
PHYSICS_RATE_HZ = 120.0
POLICY_DECIMATION = int(round(PHYSICS_RATE_HZ / POLICY_RATE_HZ))
REFERENCE_REWARD_RATE_HZ = 30.0

if not math.isclose(PHYSICS_RATE_HZ / POLICY_DECIMATION, POLICY_RATE_HZ, abs_tol=1.0e-12):
    raise RuntimeError("The policy rate must be an integer decimation of the physics rate.")


class PolicyRateGate:
    """Accept timestamped sensor frames at no more than the policy rate."""

    def __init__(self, rate_hz: float = POLICY_RATE_HZ) -> None:
        if not math.isfinite(float(rate_hz)) or float(rate_hz) <= 0.0:
            raise ValueError("Policy rate must be finite and positive.")
        self.rate_hz = float(rate_hz)
        self.period_s = 1.0 / self.rate_hz
        self.reset()

    def reset(self) -> None:
        self.last_accepted_stamp_s: float | None = None

    def accept(self, stamp_s: float) -> bool:
        stamp = float(stamp_s)
        if not math.isfinite(stamp):
            raise ValueError("Policy frame timestamp must be finite.")
        previous = self.last_accepted_stamp_s
        if previous is None:
            self.last_accepted_stamp_s = stamp
            return True
        if stamp < previous - 1.0e-9:
            raise ValueError("Policy frame timestamps must increase monotonically.")
        if stamp - previous < self.period_s - 1.0e-6:
            return False
        self.last_accepted_stamp_s = stamp
        return True


def temporal_reward_scale(step_dt_s: float) -> float:
    """Scale per-step penalties to preserve their reference cost per second."""

    step_dt = float(step_dt_s)
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("Policy step duration must be finite and positive.")
    return step_dt * REFERENCE_REWARD_RATE_HZ


__all__ = [
    "PHYSICS_RATE_HZ",
    "POLICY_DECIMATION",
    "POLICY_PERIOD_S",
    "POLICY_RATE_HZ",
    "PolicyRateGate",
    "REFERENCE_REWARD_RATE_HZ",
    "temporal_reward_scale",
]
