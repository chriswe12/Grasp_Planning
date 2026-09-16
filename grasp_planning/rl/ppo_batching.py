"""Helpers for preserving PPO batch semantics across distributed ranks."""

from __future__ import annotations


def largest_divisor_at_most(value: int, limit: int) -> int:
    """Return the largest positive divisor of ``value`` no larger than ``limit``."""

    if value < 1:
        raise ValueError("value must be positive")
    if limit < 1:
        raise ValueError("limit must be positive")
    upper = min(value, limit)
    for candidate in range(upper, 0, -1):
        if value % candidate == 0:
            return candidate
    raise AssertionError("one is always a valid divisor")


def resolve_local_minibatch_size(
    *,
    rollout_batch_size_per_rank: int,
    target_global_minibatch_size: int,
    world_size: int,
) -> int:
    """Resolve a divisible per-rank minibatch near a target global batch.

    RL-Games interprets ``minibatch_size`` independently on every distributed
    rank and averages those gradients. Dividing the desired global minibatch
    by the world size therefore preserves the single-GPU optimization
    contract instead of multiplying the effective gradient batch by the
    number of GPUs.
    """

    if rollout_batch_size_per_rank < 1:
        raise ValueError("rollout_batch_size_per_rank must be positive")
    if target_global_minibatch_size < 1:
        raise ValueError("target_global_minibatch_size must be positive")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    target_per_rank = max(1, round(target_global_minibatch_size / world_size))
    return largest_divisor_at_most(rollout_batch_size_per_rank, target_per_rank)


__all__ = ["largest_divisor_at_most", "resolve_local_minibatch_size"]
