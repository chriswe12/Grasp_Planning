"""Bounded multi-IK search for sequential MoveIt pose targets."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Mapping, Sequence

import numpy as np

from .moveit_pose_commander import PoseTarget


@dataclass(frozen=True)
class MultiIkPlanningConfig:
    candidate_count: int = 1
    beam_width: int = 1
    seed_perturbation_rad: float = 0.35
    dedup_tolerance_rad: float = 0.05
    joint_weights: tuple[float, ...] = ()

    @property
    def enabled(self) -> bool:
        return int(self.candidate_count) > 1


@dataclass(frozen=True)
class MultiIkSequencePlan:
    trajectories: Mapping[str, tuple[tuple[float, ...], ...]]
    joint_path_cost: float
    terminal_joint_positions: tuple[float, ...]
    diagnostics: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _PartialPlan:
    trajectories: dict[str, tuple[tuple[float, ...], ...]]
    cost: float
    terminal: tuple[float, ...]
    diagnostics: tuple[dict[str, object], ...]


def _wrapped_joint_delta(lhs: Sequence[float], rhs: Sequence[float]) -> np.ndarray:
    delta = np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float)
    return (delta + pi) % (2.0 * pi) - pi


def _joint_weights(config: MultiIkPlanningConfig, joint_count: int) -> np.ndarray:
    if not config.joint_weights:
        return np.ones(joint_count, dtype=float)
    weights = np.asarray(config.joint_weights, dtype=float)
    if weights.size != joint_count:
        raise ValueError(f"Expected {joint_count} multi-IK joint weights, got {weights.size}.")
    if np.any(weights <= 0.0):
        raise ValueError("Multi-IK joint weights must all be positive.")
    return weights


def _seed_candidates(
    start: tuple[float, ...],
    *,
    candidate_count: int,
    perturbation_rad: float,
) -> tuple[tuple[float, ...], ...]:
    """Return deterministic low-discrepancy seeds around the current state."""
    count = max(1, int(candidate_count))
    seeds = [start]
    joint_count = len(start)
    if joint_count == 0:
        return tuple(seeds)
    golden_ratio = 0.5 * (1.0 + np.sqrt(5.0))
    for sample_index in range(1, count):
        offsets = []
        for joint_index in range(joint_count):
            phase = ((sample_index * (joint_index + 1) / golden_ratio) % 1.0) * 2.0 - 1.0
            offsets.append(float(perturbation_rad) * float(phase))
        seeds.append(tuple(float(value + offset) for value, offset in zip(start, offsets)))
    return tuple(seeds)


def _is_distinct(
    candidate: tuple[float, ...],
    accepted: Sequence[tuple[float, ...]],
    *,
    tolerance_rad: float,
) -> bool:
    return all(
        float(np.max(np.abs(_wrapped_joint_delta(candidate, other)))) >= float(tolerance_rad) for other in accepted
    )


def _trajectory_cost(
    start: tuple[float, ...],
    waypoints: tuple[tuple[float, ...], ...],
    *,
    weights: np.ndarray,
) -> float:
    points = (start,) + waypoints
    return float(
        sum(
            np.linalg.norm(_wrapped_joint_delta(next_point, point) * weights)
            for point, next_point in zip(points, points[1:])
        )
    )


def plan_pose_sequence_multi_ik(
    commander,
    *,
    targets: Mapping[str, PoseTarget],
    labels: tuple[str, ...],
    start_joint_positions: Sequence[float],
    joint_names: tuple[str, ...],
    config: MultiIkPlanningConfig,
    label_prefix: str,
) -> MultiIkSequencePlan:
    """Plan a pose sequence while retaining the cheapest partial joint paths."""
    start = tuple(float(value) for value in start_joint_positions)
    if len(start) != len(joint_names):
        raise ValueError(f"Expected {len(joint_names)} start joints, got {len(start)}.")
    weights = _joint_weights(config, len(start))
    beam = [_PartialPlan(trajectories={}, cost=0.0, terminal=start, diagnostics=())]

    for label in labels:
        expanded: list[_PartialPlan] = []
        failures: list[str] = []
        for parent_index, parent in enumerate(beam):
            ik_solutions: list[tuple[float, ...]] = []
            for seed_index, seed in enumerate(
                _seed_candidates(
                    parent.terminal,
                    candidate_count=config.candidate_count,
                    perturbation_rad=config.seed_perturbation_rad,
                )
            ):
                joints, message = commander.compute_ik(targets[label], seed_joint_positions=seed)
                if joints is None:
                    failures.append(f"parent={parent_index} seed={seed_index}: {message}")
                    continue
                solution = tuple(float(value) for value in joints)
                if _is_distinct(
                    solution,
                    ik_solutions,
                    tolerance_rad=config.dedup_tolerance_rad,
                ):
                    ik_solutions.append(solution)

            for solution_index, solution in enumerate(ik_solutions):
                trajectory, message = commander.plan_to_joint_positions(
                    solution,
                    label=f"{label_prefix}_{label}_ik{solution_index}",
                    start_joint_positions=parent.terminal,
                )
                if trajectory is None:
                    failures.append(f"parent={parent_index} ik={solution_index}: {message}")
                    continue
                joint_trajectory = trajectory.joint_trajectory
                source_names = tuple(str(name) for name in joint_trajectory.joint_names)
                indices = {name: index for index, name in enumerate(source_names)}
                missing = [name for name in joint_names if name not in indices]
                if missing:
                    failures.append(f"parent={parent_index} ik={solution_index}: missing joints {missing}")
                    continue
                waypoints = tuple(
                    tuple(float(point.positions[indices[name]]) for name in joint_names)
                    for point in tuple(joint_trajectory.points)
                )
                if not waypoints:
                    failures.append(f"parent={parent_index} ik={solution_index}: empty trajectory")
                    continue
                edge_cost = _trajectory_cost(parent.terminal, waypoints, weights=weights)
                diagnostic = {
                    "label": label,
                    "parent_rank": parent_index,
                    "ik_solution_index": solution_index,
                    "distinct_ik_solution_count": len(ik_solutions),
                    "waypoint_count": len(waypoints),
                    "edge_joint_path_cost": edge_cost,
                    "cumulative_joint_path_cost": parent.cost + edge_cost,
                }
                expanded.append(
                    _PartialPlan(
                        trajectories={**parent.trajectories, label: waypoints},
                        cost=parent.cost + edge_cost,
                        terminal=waypoints[-1],
                        diagnostics=parent.diagnostics + (diagnostic,),
                    )
                )

        if not expanded:
            detail = "; ".join(failures[:8])
            raise RuntimeError(f"Multi-IK planning failed at {label}. {detail}")
        expanded.sort(key=lambda item: item.cost)
        beam = expanded[: max(1, int(config.beam_width))]

    best = beam[0]
    return MultiIkSequencePlan(
        trajectories=best.trajectories,
        joint_path_cost=best.cost,
        terminal_joint_positions=best.terminal,
        diagnostics=best.diagnostics,
    )


__all__ = [
    "MultiIkPlanningConfig",
    "MultiIkSequencePlan",
    "plan_pose_sequence_multi_ik",
]
