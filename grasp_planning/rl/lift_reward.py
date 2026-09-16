"""Pure reward contracts for terminal close-and-lift visual-servo trials."""

from __future__ import annotations

import torch


def smoothstep01(value: torch.Tensor) -> torch.Tensor:
    """Return a cubic smoothstep after clamping ``value`` to ``[0, 1]``."""

    bounded = value.clamp(0.0, 1.0)
    return bounded.square() * (3.0 - 2.0 * bounded)


def retained_lift_quality(
    final_lift_m: torch.Tensor,
    peak_lift_m: torch.Tensor,
    relative_drift_m: torch.Tensor,
    arm_lift_ok: torch.Tensor,
    *,
    minimum_credit_lift_m: float,
    full_credit_lift_m: float,
    drift_scale_m: float,
    drop_scale_m: float,
) -> torch.Tensor:
    """Score retained lift height, TCP-relative drift, and post-peak drop.

    Object height is measured from the instant immediately before the upward
    motion, so contact-induced motion while closing cannot earn lift credit.
    """

    if not 0.0 <= minimum_credit_lift_m < full_credit_lift_m:
        raise ValueError("Lift credit thresholds must satisfy 0 <= minimum < full.")
    if drift_scale_m <= 0.0 or drop_scale_m <= 0.0:
        raise ValueError("Lift drift and drop scales must be positive.")
    if not (final_lift_m.shape == peak_lift_m.shape == relative_drift_m.shape == arm_lift_ok.shape):
        raise ValueError("All lift-quality tensors must have the same shape.")

    height = smoothstep01(
        (final_lift_m - float(minimum_credit_lift_m)) / (float(full_credit_lift_m) - float(minimum_credit_lift_m))
    )
    drift = torch.exp(-torch.square(relative_drift_m / float(drift_scale_m)))
    peak_drop = torch.relu(peak_lift_m - final_lift_m)
    retention = torch.exp(-torch.square(peak_drop / float(drop_scale_m)))
    quality = height * (0.25 + 0.75 * drift) * retention
    return quality * arm_lift_ok.bool().to(dtype=quality.dtype)


def physical_pickup_success(
    final_lift_m: torch.Tensor,
    peak_lift_m: torch.Tensor,
    relative_drift_m: torch.Tensor,
    arm_lift_ok: torch.Tensor,
    *,
    minimum_final_lift_m: float,
    maximum_relative_drift_m: float,
    maximum_peak_drop_m: float,
) -> torch.Tensor:
    """Return the permissive physical-pickup outcome used by PPO."""

    if minimum_final_lift_m <= 0.0:
        raise ValueError("minimum_final_lift_m must be positive.")
    if maximum_relative_drift_m <= 0.0 or maximum_peak_drop_m <= 0.0:
        raise ValueError("Pickup drift and drop thresholds must be positive.")
    peak_drop = torch.relu(peak_lift_m - final_lift_m)
    return (
        arm_lift_ok.bool()
        & (final_lift_m >= float(minimum_final_lift_m))
        & (relative_drift_m <= float(maximum_relative_drift_m))
        & (peak_drop <= float(maximum_peak_drop_m))
    )


def lift_outcome_reward(
    geometric_success: torch.Tensor,
    pickup_success: torch.Tensor,
    lift_quality: torch.Tensor,
    *,
    lift_quality_reward: float,
    geometric_lift_bonus: float,
    neither_penalty: float,
) -> torch.Tensor:
    """Reward lifting, add a target-pose bonus, and penalize only neither.

    Geometric success without pickup is deliberately not penalized; its
    positive reward is issued when the policy commits to the lift attempt.
    """

    if not (geometric_success.shape == pickup_success.shape == lift_quality.shape):
        raise ValueError("Lift outcome tensors must have the same shape.")
    if lift_quality_reward < 0.0 or geometric_lift_bonus < 0.0 or neither_penalty < 0.0:
        raise ValueError("Lift reward weights must be non-negative.")
    geometric = geometric_success.bool()
    pickup = pickup_success.bool()
    quality = lift_quality.clamp(0.0, 1.0)
    pickup_quality = pickup.to(dtype=quality.dtype) * quality
    reward = float(lift_quality_reward) * pickup_quality
    reward += float(geometric_lift_bonus) * geometric.to(dtype=quality.dtype) * pickup_quality
    reward -= float(neither_penalty) * (~geometric & ~pickup).to(dtype=quality.dtype)
    return reward


__all__ = [
    "lift_outcome_reward",
    "physical_pickup_success",
    "retained_lift_quality",
    "smoothstep01",
]
