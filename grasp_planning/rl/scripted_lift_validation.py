"""Pure reporting contracts for scripted grasp-lift validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

LIFTABILITY_LABEL_CODES = {
    "untested": 0,
    "liftable": 1,
    "not_liftable": 2,
    "setup_invalid": 3,
}
SIMULATOR_INVALID_STATUSES = frozenset({"object_unstable_before_grasp", "arm_lift_failed"})
PHYSICAL_FAILURE_STATUSES = frozenset(
    {"object_not_lifted", "object_dropped_after_peak", "object_not_retained_near_tcp"}
)


@dataclass(frozen=True)
class ScriptedLiftThresholds:
    """Thresholds separating simulator/setup failures from grasp retention."""

    minimum_final_lift_m: float = 0.04
    minimum_arm_lift_fraction: float = 0.80
    maximum_settle_translation_m: float = 0.003
    maximum_settle_rotation_rad: float = 0.05235987755982989  # 3 degrees
    maximum_peak_drop_m: float = 0.015
    maximum_relative_drift_m: float = 0.030

    def __post_init__(self) -> None:
        positive = {
            "minimum_final_lift_m": self.minimum_final_lift_m,
            "minimum_arm_lift_fraction": self.minimum_arm_lift_fraction,
            "maximum_settle_translation_m": self.maximum_settle_translation_m,
            "maximum_settle_rotation_rad": self.maximum_settle_rotation_rad,
            "maximum_peak_drop_m": self.maximum_peak_drop_m,
            "maximum_relative_drift_m": self.maximum_relative_drift_m,
        }
        invalid = [name for name, value in positive.items() if float(value) <= 0.0]
        if invalid:
            raise ValueError(f"Scripted lift thresholds must be positive: {invalid}")
        if float(self.minimum_arm_lift_fraction) > 1.0:
            raise ValueError("minimum_arm_lift_fraction must not exceed one.")


def classify_scripted_lift(
    *,
    commanded_lift_m: float,
    tcp_final_lift_m: float,
    object_final_lift_m: float,
    object_peak_lift_m: float,
    settle_translation_m: float,
    settle_rotation_rad: float,
    relative_drift_m: float,
    thresholds: ScriptedLiftThresholds,
) -> str:
    """Classify one attempt without letting simulator failures blame a grasp."""

    if (
        float(settle_translation_m) > thresholds.maximum_settle_translation_m
        or float(settle_rotation_rad) > thresholds.maximum_settle_rotation_rad
    ):
        return "object_unstable_before_grasp"
    if float(tcp_final_lift_m) < thresholds.minimum_arm_lift_fraction * float(commanded_lift_m):
        return "arm_lift_failed"
    if float(object_final_lift_m) >= thresholds.minimum_final_lift_m:
        if float(relative_drift_m) > thresholds.maximum_relative_drift_m:
            return "object_not_retained_near_tcp"
        if float(object_peak_lift_m) - float(object_final_lift_m) > thresholds.maximum_peak_drop_m:
            return "object_dropped_after_peak"
        return "success"
    if float(object_peak_lift_m) >= thresholds.minimum_final_lift_m:
        return "object_dropped_after_peak"
    return "object_not_lifted"


def summarize_scripted_lifts(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate overall and per-part scripted-lift outcomes."""

    records = [dict(row) for row in rows]
    status_counts = Counter(str(row["status"]) for row in records)
    simulator_invalid = status_counts["object_unstable_before_grasp"] + status_counts["arm_lift_failed"]
    valid_attempts = len(records) - simulator_invalid
    successes = status_counts["success"]
    by_part: dict[str, dict[str, object]] = {}
    for part_id in sorted({str(row["part_id"]) for row in records}):
        selected = [row for row in records if str(row["part_id"]) == part_id]
        counts = Counter(str(row["status"]) for row in selected)
        invalid = counts["object_unstable_before_grasp"] + counts["arm_lift_failed"]
        valid = len(selected) - invalid
        by_part[part_id] = {
            "attempts": len(selected),
            "valid_attempts": valid,
            "successes": counts["success"],
            "success_rate": counts["success"] / valid if valid else 0.0,
            "status_counts": dict(sorted(counts.items())),
        }
    return {
        "attempts": len(records),
        "simulator_invalid_attempts": simulator_invalid,
        "valid_attempts": valid_attempts,
        "successes": successes,
        "success_rate": successes / valid_attempts if valid_attempts else 0.0,
        "status_counts": dict(sorted(status_counts.items())),
        "per_part": by_part,
    }


def classify_liftability_evidence(statuses: Iterable[str]) -> str:
    """Reduce repeated scripted trials to a conservative target label."""

    normalized = tuple(str(status) for status in statuses)
    known = {"success", *SIMULATOR_INVALID_STATUSES, *PHYSICAL_FAILURE_STATUSES}
    unknown = sorted(set(normalized) - known)
    if unknown:
        raise ValueError(f"Unknown scripted-lift statuses: {unknown}")
    if "success" in normalized:
        return "liftable"
    if any(status in PHYSICAL_FAILURE_STATUSES for status in normalized):
        return "not_liftable"
    if any(status in SIMULATOR_INVALID_STATUSES for status in normalized):
        return "setup_invalid"
    return "untested"


def build_liftability_records(
    catalog_target_ids: np.ndarray,
    attempts: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Align attempt evidence to a complete catalog without dropping untested targets."""

    target_ids = np.asarray(catalog_target_ids).astype(str)
    if target_ids.ndim != 1 or len(set(target_ids.tolist())) != len(target_ids):
        raise ValueError("Catalog target IDs must be a unique one-dimensional array.")
    evidence: dict[str, list[str]] = {target_id: [] for target_id in target_ids.tolist()}
    for attempt in attempts:
        target_id = str(attempt["target_id"])
        if target_id not in evidence:
            raise ValueError(f"Attempt target {target_id!r} is absent from the catalog.")
        evidence[target_id].append(str(attempt["status"]))
    records: list[dict[str, object]] = []
    for target_id in target_ids.tolist():
        statuses = evidence[target_id]
        label = classify_liftability_evidence(statuses)
        records.append(
            {
                "target_id": target_id,
                "label": label,
                "label_code": LIFTABILITY_LABEL_CODES[label],
                "attempt_count": len(statuses),
                "success_count": statuses.count("success"),
                "physical_failure_count": sum(status in PHYSICAL_FAILURE_STATUSES for status in statuses),
                "simulator_invalid_count": sum(status in SIMULATOR_INVALID_STATUSES for status in statuses),
            }
        )
    return records


def balanced_subset_indices(
    part_ids: np.ndarray,
    limit: int,
    *,
    start_round: int = 0,
) -> np.ndarray:
    """Select a deterministic round-robin subset across available parts.

    ``start_round`` skips that many targets within every part.  It permits
    repeatable physical screens of different catalog grasps without letting
    large part families dominate a bounded simulator batch.
    """

    normalized = np.asarray(part_ids).astype(str)
    if normalized.ndim != 1:
        raise ValueError("part_ids must be a one-dimensional array.")
    if limit < 0:
        raise ValueError("limit must be non-negative.")
    if start_round < 0:
        raise ValueError("start_round must be non-negative.")
    if limit == 0 or limit >= len(normalized):
        if start_round == 0:
            return np.arange(len(normalized), dtype=np.int64)
    groups = {
        part_id: np.flatnonzero(normalized == part_id).astype(np.int64).tolist()[start_round:]
        for part_id in sorted(set(normalized.tolist()))
    }
    selected: list[int] = []
    while len(selected) < limit and any(groups.values()):
        for part_id in groups:
            if groups[part_id] and len(selected) < limit:
                selected.append(groups[part_id].pop(0))
    return np.asarray(selected, dtype=np.int64)


def resolve_explicit_target_indices(
    available_target_ids: np.ndarray,
    requested_target_ids: Iterable[str],
) -> np.ndarray:
    """Resolve unique target IDs to catalog indices without silent omission."""

    available = np.asarray(available_target_ids).astype(str)
    if available.ndim != 1:
        raise ValueError("available_target_ids must be a one-dimensional array.")
    requested = tuple(str(value) for value in requested_target_ids)
    if not requested:
        raise ValueError("At least one target ID is required.")
    if len(set(requested)) != len(requested):
        raise ValueError("Requested target IDs must be unique.")
    index_by_id = {target_id: index for index, target_id in enumerate(available.tolist())}
    missing = [target_id for target_id in requested if target_id not in index_by_id]
    if missing:
        raise ValueError(f"Requested target IDs are absent from the selected catalog split: {missing}")
    return np.asarray([index_by_id[target_id] for target_id in requested], dtype=np.int64)


def contiguous_subset_indices(target_count: int, *, offset: int, limit: int) -> np.ndarray:
    """Return one bounded, non-overlapping catalog batch."""

    if target_count <= 0:
        raise ValueError("target_count must be positive.")
    if offset < 0 or offset >= target_count:
        raise ValueError(f"offset must be in [0, {target_count - 1}].")
    if limit <= 0:
        raise ValueError("limit must be positive.")
    return np.arange(offset, min(offset + limit, target_count), dtype=np.int64)


__all__ = [
    "LIFTABILITY_LABEL_CODES",
    "PHYSICAL_FAILURE_STATUSES",
    "SIMULATOR_INVALID_STATUSES",
    "ScriptedLiftThresholds",
    "balanced_subset_indices",
    "build_liftability_records",
    "classify_liftability_evidence",
    "classify_scripted_lift",
    "contiguous_subset_indices",
    "resolve_explicit_target_indices",
    "summarize_scripted_lifts",
]
