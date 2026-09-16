"""Deterministic part assignment and geometry-matched target sampling tables."""
import numpy as np


def part_assignment(target_parts, *, num_envs, rank=0, world_size=1, sequential=False):
    parts = np.asarray(target_parts, dtype=np.int64)
    eligible = np.unique(parts)
    if not len(eligible) or num_envs < 1:
        raise ValueError('Need targets and a positive environment count')
    if num_envs * world_size < len(eligible):
        raise ValueError(f'{len(eligible)} parts need at least that many total environment slots')
    if sequential:
        # Same fixed geometry per slot, then cycle through that part's targets.
        mapping = eligible[np.arange(num_envs) % len(eligible)]
    else:
        mapping = eligible[(rank * num_envs + np.arange(num_envs)) % len(eligible)]
    buckets = [np.flatnonzero(parts == part) for part in mapping]
    counts = np.asarray([len(b) for b in buckets], dtype=np.int64)
    padded = np.zeros((num_envs, int(counts.max())), dtype=np.int64)
    for i, bucket in enumerate(buckets):
        padded[i, :len(bucket)] = bucket
    return mapping, padded, counts
