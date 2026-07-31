"""Frame-aware reachability proxies for ordering dual-grasp planning attempts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import (
    quat_to_rotmat_xyzw,
    rotmat_to_quat_xyzw,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _rotation_z(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


@dataclass(frozen=True)
class MovableFrame:
    position_world_m: tuple[float, float, float]
    yaw_deg: float = 0.0

    def to_payload(self) -> dict[str, object]:
        return {
            "position_world_m": list(self.position_world_m),
            "yaw_deg": self.yaw_deg,
        }


@dataclass(frozen=True)
class TaskTargetPose:
    name: str
    position_world_m: tuple[float, float, float]
    orientation_xyzw_world: tuple[float, float, float, float]

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "position_world_m": list(self.position_world_m),
            "orientation_xyzw_world": list(self.orientation_xyzw_world),
        }


@dataclass(frozen=True)
class ReachabilityProxyConfig:
    shoulder_offset_base_m: tuple[float, float, float] = (0.0, 0.0, 0.34)
    minimum_reach_m: float = 0.15
    comfort_reach_m: float = 0.55
    maximum_reach_m: float = 0.95
    minimum_height_base_m: float = -0.05
    comfort_height_base_m: float = 0.25
    maximum_height_base_m: float = 0.85
    front_zero_m: float = -0.15
    front_full_m: float = 0.15
    distance_weight: float = 0.45
    height_weight: float = 0.20
    front_weight: float = 0.20
    approach_weight: float = 0.15
    target_min_weight: float = 0.60
    target_mean_weight: float = 0.40
    ownership_margin_m: float = 0.25
    ownership_weight: float = 0.70
    noncrossing_weight: float = 0.30
    offline_pair_weight: float = 0.40
    reachability_weight: float = 0.45
    layout_weight: float = 0.15

    def __post_init__(self) -> None:
        if not (0.0 <= self.minimum_reach_m < self.comfort_reach_m < self.maximum_reach_m):
            raise ValueError("Reach limits must satisfy minimum < comfort < maximum.")
        if not (self.minimum_height_base_m < self.comfort_height_base_m < self.maximum_height_base_m):
            raise ValueError("Height limits must satisfy minimum < comfort < maximum.")
        if self.front_zero_m >= self.front_full_m:
            raise ValueError("front_zero_m must be less than front_full_m.")
        if self.ownership_margin_m <= 0.0:
            raise ValueError("ownership_margin_m must be > 0.")
        weight_groups = (
            (
                self.distance_weight,
                self.height_weight,
                self.front_weight,
                self.approach_weight,
            ),
            (self.target_min_weight, self.target_mean_weight),
            (self.ownership_weight, self.noncrossing_weight),
            (
                self.offline_pair_weight,
                self.reachability_weight,
                self.layout_weight,
            ),
        )
        for weights in weight_groups:
            if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
                raise ValueError("Reachability score weights must be nonnegative with a positive sum in each group.")

    def to_payload(self) -> dict[str, object]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}


def transform_target_pose(
    *,
    position_parent_m: tuple[float, float, float],
    orientation_xyzw_parent: tuple[float, float, float, float],
    parent_frame_world: MovableFrame,
    name: str,
) -> TaskTargetPose:
    rotation = _rotation_z(parent_frame_world.yaw_deg)
    position = rotation @ np.asarray(position_parent_m, dtype=float) + np.asarray(
        parent_frame_world.position_world_m, dtype=float
    )
    orientation = rotation @ quat_to_rotmat_xyzw(orientation_xyzw_parent)
    return TaskTargetPose(
        name=name,
        position_world_m=tuple(float(value) for value in position),
        orientation_xyzw_world=tuple(float(value) for value in rotmat_to_quat_xyzw(orientation)),
    )


def _triangle_score(
    value: float,
    *,
    minimum: float,
    comfort: float,
    maximum: float,
) -> float:
    if value <= minimum or value >= maximum:
        return 0.0
    if value <= comfort:
        return _clamp01((value - minimum) / (comfort - minimum))
    return _clamp01((maximum - value) / (maximum - comfort))


def robot_shoulder_world(
    robot_base_world: MovableFrame,
    *,
    config: ReachabilityProxyConfig,
) -> np.ndarray:
    return _rotation_z(robot_base_world.yaw_deg) @ np.asarray(config.shoulder_offset_base_m, dtype=float) + np.asarray(
        robot_base_world.position_world_m, dtype=float
    )


def workspace_pose_score(
    target: TaskTargetPose,
    *,
    robot_base_world: MovableFrame,
    config: ReachabilityProxyConfig,
) -> dict[str, float | bool]:
    """Score one TCP target without claiming kinematic feasibility."""

    base_position = np.asarray(robot_base_world.position_world_m, dtype=float)
    shoulder_world = robot_shoulder_world(robot_base_world, config=config)
    target_position = np.asarray(target.position_world_m, dtype=float)
    target_from_shoulder = target_position - shoulder_world
    distance = float(np.linalg.norm(target_from_shoulder))
    rotation_world_from_base = _rotation_z(robot_base_world.yaw_deg)
    target_base = rotation_world_from_base.T @ (target_position - base_position)
    shoulder_to_target_direction = target_from_shoulder / max(distance, 1.0e-12)
    approach_world = quat_to_rotmat_xyzw(target.orientation_xyzw_world)[:, 2]
    approach_alignment = float(np.dot(approach_world, shoulder_to_target_direction))

    distance_score = _triangle_score(
        distance,
        minimum=config.minimum_reach_m,
        comfort=config.comfort_reach_m,
        maximum=config.maximum_reach_m,
    )
    height_score = _triangle_score(
        float(target_base[2]),
        minimum=config.minimum_height_base_m,
        comfort=config.comfort_height_base_m,
        maximum=config.maximum_height_base_m,
    )
    front_score = _clamp01((float(target_base[0]) - config.front_zero_m) / (config.front_full_m - config.front_zero_m))
    approach_score = _clamp01((approach_alignment + 0.20) / 1.20)
    inside_reach_shell = config.minimum_reach_m < distance < config.maximum_reach_m
    inside_height_band = config.minimum_height_base_m < float(target_base[2]) < config.maximum_height_base_m
    total_weight = config.distance_weight + config.height_weight + config.front_weight + config.approach_weight
    score = (
        config.distance_weight * distance_score
        + config.height_weight * height_score
        + config.front_weight * front_score
        + config.approach_weight * approach_score
    ) / total_weight
    if not inside_reach_shell or not inside_height_band:
        score = 0.0
    return {
        "score": float(score),
        "distance_score": distance_score,
        "height_score": height_score,
        "front_score": front_score,
        "approach_score": approach_score,
        "distance_m": distance,
        "height_base_m": float(target_base[2]),
        "front_base_m": float(target_base[0]),
        "approach_alignment": approach_alignment,
        "inside_reach_shell": inside_reach_shell,
        "inside_height_band": inside_height_band,
    }


def arm_target_set_score(
    targets: Iterable[TaskTargetPose],
    *,
    robot_base_world: MovableFrame,
    config: ReachabilityProxyConfig,
) -> dict[str, object]:
    target_scores = tuple(
        (
            target,
            workspace_pose_score(
                target,
                robot_base_world=robot_base_world,
                config=config,
            ),
        )
        for target in targets
    )
    if not target_scores:
        return {
            "score": 0.0,
            "minimum_target_score": 0.0,
            "mean_target_score": 0.0,
            "targets": [],
        }
    scores = [float(score["score"]) for _, score in target_scores]
    minimum_score = min(scores)
    mean_score = float(np.mean(scores))
    total_weight = config.target_min_weight + config.target_mean_weight
    score = (config.target_min_weight * minimum_score + config.target_mean_weight * mean_score) / total_weight
    return {
        "score": float(score),
        "minimum_target_score": minimum_score,
        "mean_target_score": mean_score,
        "targets": [{"name": target.name, **target_score} for target, target_score in target_scores],
    }


def _segments_cross_xy(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    def orientation(
        first: np.ndarray,
        second: np.ndarray,
        third: np.ndarray,
    ) -> float:
        return float((second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0]))

    a = orientation(first_start, first_end, second_start)
    b = orientation(first_start, first_end, second_end)
    c = orientation(second_start, second_end, first_start)
    d = orientation(second_start, second_end, first_end)
    return a * b < 0.0 and c * d < 0.0


def pair_layout_score(
    *,
    offline_pair_score: float,
    holder_targets: Iterable[TaskTargetPose],
    inserter_targets: Iterable[TaskTargetPose],
    holder_grasp_target: TaskTargetPose,
    inserter_grasp_target: TaskTargetPose,
    holder_robot_base_world: MovableFrame,
    inserter_robot_base_world: MovableFrame,
    config: ReachabilityProxyConfig,
) -> dict[str, object]:
    """Rank one Stage-3 pair for a movable cell layout."""

    holder = arm_target_set_score(
        holder_targets,
        robot_base_world=holder_robot_base_world,
        config=config,
    )
    inserter = arm_target_set_score(
        inserter_targets,
        robot_base_world=inserter_robot_base_world,
        config=config,
    )
    holder_score = float(holder["score"])
    inserter_score = float(inserter["score"])
    reachability_score = (
        config.target_min_weight * min(holder_score, inserter_score)
        + config.target_mean_weight * 0.5 * (holder_score + inserter_score)
    ) / (config.target_min_weight + config.target_mean_weight)

    holder_shoulder = robot_shoulder_world(
        holder_robot_base_world,
        config=config,
    )
    inserter_shoulder = robot_shoulder_world(
        inserter_robot_base_world,
        config=config,
    )
    holder_target = np.asarray(
        holder_grasp_target.position_world_m,
        dtype=float,
    )
    inserter_target = np.asarray(
        inserter_grasp_target.position_world_m,
        dtype=float,
    )
    holder_own = float(np.linalg.norm(holder_target - holder_shoulder))
    holder_other = float(np.linalg.norm(holder_target - inserter_shoulder))
    inserter_own = float(np.linalg.norm(inserter_target - inserter_shoulder))
    inserter_other = float(np.linalg.norm(inserter_target - holder_shoulder))
    holder_ownership = _clamp01(0.5 + (holder_other - holder_own) / (2.0 * config.ownership_margin_m))
    inserter_ownership = _clamp01(0.5 + (inserter_other - inserter_own) / (2.0 * config.ownership_margin_m))
    ownership_score = (
        config.target_min_weight * min(holder_ownership, inserter_ownership)
        + config.target_mean_weight * 0.5 * (holder_ownership + inserter_ownership)
    ) / (config.target_min_weight + config.target_mean_weight)
    segments_cross = _segments_cross_xy(
        holder_shoulder,
        holder_target,
        inserter_shoulder,
        inserter_target,
    )
    noncrossing_score = 0.0 if segments_cross else 1.0
    layout_score = (config.ownership_weight * ownership_score + config.noncrossing_weight * noncrossing_score) / (
        config.ownership_weight + config.noncrossing_weight
    )
    total_weight = config.offline_pair_weight + config.reachability_weight + config.layout_weight
    combined_score = (
        config.offline_pair_weight * _clamp01(offline_pair_score)
        + config.reachability_weight * reachability_score
        + config.layout_weight * layout_score
    ) / total_weight
    return {
        "score": float(combined_score),
        "offline_pair_score": _clamp01(offline_pair_score),
        "reachability_score": float(reachability_score),
        "layout_score": float(layout_score),
        "holder": holder,
        "inserter": inserter,
        "ownership_score": float(ownership_score),
        "holder_ownership_score": holder_ownership,
        "inserter_ownership_score": inserter_ownership,
        "noncrossing_score": noncrossing_score,
        "segments_cross_xy": segments_cross,
    }


__all__ = [
    "MovableFrame",
    "ReachabilityProxyConfig",
    "TaskTargetPose",
    "arm_target_set_score",
    "pair_layout_score",
    "robot_shoulder_world",
    "transform_target_pose",
    "workspace_pose_score",
]
