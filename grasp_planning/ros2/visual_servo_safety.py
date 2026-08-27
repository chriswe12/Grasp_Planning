"""Fail-closed safety and learned-completion gates for real visual servoing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class VisualServoState(str, Enum):
    DISARMED = "DISARMED"
    READY = "READY"
    RUNNING = "RUNNING"
    CANDIDATE_HOLD = "CANDIDATE_HOLD"
    COMPLETED_HOLD = "COMPLETED_HOLD"
    SAFETY_HOLD = "SAFETY_HOLD"
    FAULT = "FAULT"


@dataclass(frozen=True)
class VisualServoSafetyConfig:
    max_image_age_s: float = 0.15
    enforce_source_image_age: bool = True
    max_image_skew_s: float = 0.010
    max_pose_age_s: float = 0.15
    max_tf_age_s: float = 0.15
    max_servo_status_age_s: float = 0.25
    minimum_valid_depth_fraction: float = 0.20
    maximum_trial_duration_s: float = 15.0
    completion_probability_threshold: float = 0.95
    completion_required_consecutive_steps: int = 4
    completion_max_linear_speed_m_s: float = 0.005
    completion_max_angular_speed_rad_s: float = 0.03
    workspace_min_xyz_m: tuple[float, float, float] = (-math.inf, -math.inf, -math.inf)
    workspace_max_xyz_m: tuple[float, float, float] = (math.inf, math.inf, math.inf)
    joint_position_limits_rad: tuple[tuple[float, float], ...] = ()
    max_joint_velocity_rad_s: tuple[float, ...] = ()
    max_joint_acceleration_rad_s2: tuple[float, ...] = ()
    force_abort_threshold_n: float = math.inf
    require_deadman: bool = True
    require_force_measurement: bool = True

    def validate(self) -> None:
        positive_fields = {
            "max_image_age_s": self.max_image_age_s,
            "max_image_skew_s": self.max_image_skew_s,
            "max_pose_age_s": self.max_pose_age_s,
            "max_tf_age_s": self.max_tf_age_s,
            "max_servo_status_age_s": self.max_servo_status_age_s,
            "maximum_trial_duration_s": self.maximum_trial_duration_s,
            "completion_max_linear_speed_m_s": self.completion_max_linear_speed_m_s,
            "completion_max_angular_speed_rad_s": self.completion_max_angular_speed_rad_s,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0.0 <= self.minimum_valid_depth_fraction <= 1.0:
            raise ValueError("minimum_valid_depth_fraction must lie in [0, 1].")
        if not 0.0 < self.completion_probability_threshold <= 1.0:
            raise ValueError("completion_probability_threshold must lie in (0, 1].")
        if self.completion_required_consecutive_steps < 1:
            raise ValueError("completion_required_consecutive_steps must be positive.")
        if any(low >= high for low, high in zip(self.workspace_min_xyz_m, self.workspace_max_xyz_m, strict=True)):
            raise ValueError("Every workspace minimum must be below its maximum.")
        if any(low >= high for low, high in self.joint_position_limits_rad):
            raise ValueError("Every joint-position minimum must be below its maximum.")
        if any(value <= 0.0 for value in self.max_joint_velocity_rad_s):
            raise ValueError("Joint-velocity limits must be positive.")
        if any(value <= 0.0 for value in self.max_joint_acceleration_rad_s2):
            raise ValueError("Joint-acceleration limits must be positive.")
        joint_count = len(self.joint_position_limits_rad)
        for name, values in (
            ("max_joint_velocity_rad_s", self.max_joint_velocity_rad_s),
            ("max_joint_acceleration_rad_s2", self.max_joint_acceleration_rad_s2),
        ):
            if values and len(values) != joint_count:
                raise ValueError(f"{name} must match the configured joint-position limit count.")


@dataclass(frozen=True)
class VisualServoSafetySample:
    now_s: float
    color_stamp_s: float
    depth_stamp_s: float
    pose_stamp_s: float
    tf_stamp_s: float
    valid_depth_fraction: float
    requested_normalized_action: tuple[float, float, float, float, float, float]
    completion_probability: float
    tcp_position_m: tuple[float, float, float]
    tcp_linear_speed_m_s: float
    tcp_angular_speed_rad_s: float
    joint_positions_rad: tuple[float, ...] = ()
    joint_velocities_rad_s: tuple[float, ...] = ()
    joint_accelerations_rad_s2: tuple[float, ...] = ()
    force_norm_n: float | None = None
    deadman_active: bool = False
    emergency_stop_active: bool = False
    command_consumer_exists: bool = False
    servo_healthy: bool = False
    servo_status_age_s: float | None = None


@dataclass(frozen=True)
class VisualServoSafetyDecision:
    state: VisualServoState
    applied_normalized_action: tuple[float, float, float, float, float, float]
    completion_streak: int
    reason: str
    terminal: bool


def slew_limit_normalized_action(
    requested: Sequence[float],
    previous_applied: Sequence[float],
    *,
    delta_limit: float = 0.25,
) -> tuple[float, float, float, float, float, float]:
    requested_array = np.asarray(requested, dtype=np.float64)
    previous_array = np.asarray(previous_applied, dtype=np.float64)
    if requested_array.shape != (6,) or previous_array.shape != (6,):
        raise ValueError("Visual-servo actions must contain exactly six values.")
    if not np.isfinite(requested_array).all() or not np.isfinite(previous_array).all():
        raise ValueError("Visual-servo actions must be finite.")
    if not math.isfinite(float(delta_limit)) or float(delta_limit) <= 0.0:
        raise ValueError("delta_limit must be finite and positive.")
    requested_array = np.clip(requested_array, -1.0, 1.0)
    delta = np.clip(requested_array - previous_array, -float(delta_limit), float(delta_limit))
    return tuple(float(value) for value in np.clip(previous_array + delta, -1.0, 1.0))  # type: ignore[return-value]


class PoseVelocityEstimator:
    """Timestamp-aware finite-difference TCP speed estimate with low-pass filtering."""

    def __init__(self, *, smoothing_alpha: float = 0.35, maximum_dt_s: float = 0.25) -> None:
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must lie in (0, 1].")
        if maximum_dt_s <= 0.0:
            raise ValueError("maximum_dt_s must be positive.")
        self._alpha = float(smoothing_alpha)
        self._maximum_dt_s = float(maximum_dt_s)
        self.reset()

    def reset(self) -> None:
        self._previous: tuple[float, np.ndarray, np.ndarray] | None = None
        self._linear_velocity = np.zeros(3, dtype=np.float64)
        self._angular_velocity = np.zeros(3, dtype=np.float64)
        self._linear_speed = 0.0
        self._angular_speed = 0.0

    @property
    def linear_velocity_m_s(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._linear_velocity)  # type: ignore[return-value]

    @property
    def angular_velocity_rad_s(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._angular_velocity)  # type: ignore[return-value]

    def update(
        self,
        *,
        stamp_s: float,
        position_m: Sequence[float],
        orientation_xyzw: Sequence[float],
    ) -> tuple[float, float]:
        position = np.asarray(position_m, dtype=np.float64)
        quaternion = np.asarray(orientation_xyzw, dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("Pose velocity estimation requires XYZ and XYZW values.")
        if not math.isfinite(float(stamp_s)) or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise ValueError("Pose velocity inputs must be finite.")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-12:
            raise ValueError("Pose quaternion norm is zero.")
        quaternion /= norm
        if self._previous is None:
            self._previous = (float(stamp_s), position, quaternion)
            return self._linear_speed, self._angular_speed
        previous_stamp, previous_position, previous_quaternion = self._previous
        dt = float(stamp_s) - previous_stamp
        if dt <= 0.0:
            raise ValueError("Pose timestamps must increase monotonically.")
        self._previous = (float(stamp_s), position, quaternion)
        if dt > self._maximum_dt_s:
            self._linear_velocity.fill(0.0)
            self._angular_velocity.fill(0.0)
            self._linear_speed = 0.0
            self._angular_speed = 0.0
            return self._linear_speed, self._angular_speed
        linear_velocity = (position - previous_position) / dt
        previous_conjugate = previous_quaternion.copy()
        previous_conjugate[:3] *= -1.0
        x1, y1, z1, w1 = quaternion
        x2, y2, z2, w2 = previous_conjugate
        delta = np.asarray(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ),
            dtype=np.float64,
        )
        if delta[3] < 0.0:
            delta *= -1.0
        vector_norm = float(np.linalg.norm(delta[:3]))
        if vector_norm <= 1.0e-12:
            angular_velocity = np.zeros(3, dtype=np.float64)
        else:
            angle = 2.0 * math.atan2(vector_norm, float(np.clip(delta[3], -1.0, 1.0)))
            angular_velocity = delta[:3] * (angle / (vector_norm * dt))
        self._linear_velocity += self._alpha * (linear_velocity - self._linear_velocity)
        self._angular_velocity += self._alpha * (angular_velocity - self._angular_velocity)
        self._linear_speed = float(np.linalg.norm(self._linear_velocity))
        self._angular_speed = float(np.linalg.norm(self._angular_velocity))
        return self._linear_speed, self._angular_speed


class VisualServoSafetySupervisor:
    """Latch faults and gate learned completion independently of the policy."""

    ZERO_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __init__(self, config: VisualServoSafetyConfig) -> None:
        config.validate()
        self.config = config
        self.state = VisualServoState.DISARMED
        self.reason = "not armed"
        self.completion_streak = 0
        self._trial_start_s: float | None = None

    def mark_ready(self) -> None:
        if self.state != VisualServoState.DISARMED:
            raise RuntimeError(f"Cannot mark ready from state {self.state.value}.")
        self.state = VisualServoState.READY
        self.reason = "preflight complete"

    def arm(self, *, now_s: float) -> None:
        if self.state != VisualServoState.READY:
            raise RuntimeError(f"Cannot arm from state {self.state.value}.")
        self._trial_start_s = float(now_s)
        self.completion_streak = 0
        self.state = VisualServoState.RUNNING
        self.reason = "running"

    def reset(self) -> None:
        self.state = VisualServoState.DISARMED
        self.reason = "not armed"
        self.completion_streak = 0
        self._trial_start_s = None

    def latch_fault(self, reason: str, *, fault: bool = False) -> VisualServoSafetyDecision:
        self.state = VisualServoState.FAULT if fault else VisualServoState.SAFETY_HOLD
        self.reason = str(reason)
        self.completion_streak = 0
        return self._decision(self.ZERO_ACTION, terminal=True)

    def evaluate(self, sample: VisualServoSafetySample) -> VisualServoSafetyDecision:
        if self.state in {
            VisualServoState.COMPLETED_HOLD,
            VisualServoState.SAFETY_HOLD,
            VisualServoState.FAULT,
        }:
            return self._decision(self.ZERO_ACTION, terminal=True)
        if self.state not in {VisualServoState.RUNNING, VisualServoState.CANDIDATE_HOLD}:
            return self.latch_fault(f"safety evaluation attempted from {self.state.value}", fault=True)

        violation = self._violation(sample)
        if violation is not None:
            return self.latch_fault(violation)

        if sample.completion_probability >= self.config.completion_probability_threshold:
            stable = (
                sample.tcp_linear_speed_m_s <= self.config.completion_max_linear_speed_m_s
                and sample.tcp_angular_speed_rad_s <= self.config.completion_max_angular_speed_rad_s
            )
            self.completion_streak = self.completion_streak + 1 if stable else 0
            if self.completion_streak >= self.config.completion_required_consecutive_steps:
                self.state = VisualServoState.COMPLETED_HOLD
                self.reason = "learned completion gate satisfied"
                return self._decision(self.ZERO_ACTION, terminal=True)
            self.state = VisualServoState.CANDIDATE_HOLD
            self.reason = "completion candidate; holding for low-speed stability"
            return self._decision(self.ZERO_ACTION, terminal=False)

        self.completion_streak = 0
        self.state = VisualServoState.RUNNING
        self.reason = "running"
        applied = slew_limit_normalized_action(sample.requested_normalized_action, self.ZERO_ACTION, delta_limit=1.0)
        return self._decision(applied, terminal=False)

    def _violation(self, sample: VisualServoSafetySample) -> str | None:
        finite_scalars = (
            sample.now_s,
            sample.color_stamp_s,
            sample.depth_stamp_s,
            sample.pose_stamp_s,
            sample.tf_stamp_s,
            sample.valid_depth_fraction,
            sample.completion_probability,
            sample.tcp_linear_speed_m_s,
            sample.tcp_angular_speed_rad_s,
        )
        arrays = (
            sample.requested_normalized_action,
            sample.tcp_position_m,
            sample.joint_positions_rad,
            sample.joint_velocities_rad_s,
            sample.joint_accelerations_rad_s2,
        )
        if not all(math.isfinite(float(value)) for value in finite_scalars) or not all(
            np.isfinite(np.asarray(values, dtype=np.float64)).all() for values in arrays
        ):
            return "non-finite observation, state, or policy output"
        if sample.emergency_stop_active:
            return "operator emergency stop is active"
        if self.config.require_deadman and not sample.deadman_active:
            return "operator deadman is not active"
        if not sample.command_consumer_exists:
            return "MoveIt Servo command consumer is unavailable"
        if not sample.servo_healthy:
            return "MoveIt Servo reported an unsafe or unavailable state"
        if sample.servo_status_age_s is not None and sample.servo_status_age_s > self.config.max_servo_status_age_s:
            return "MoveIt Servo status is stale"
        if abs(sample.color_stamp_s - sample.depth_stamp_s) > self.config.max_image_skew_s:
            return "RGB/depth timestamps exceed the synchronization tolerance"
        if (
            self.config.enforce_source_image_age
            and sample.now_s - min(sample.color_stamp_s, sample.depth_stamp_s)
            > self.config.max_image_age_s
        ):
            return "RGB-D observation is stale"
        if sample.now_s - sample.pose_stamp_s > self.config.max_pose_age_s:
            return "TCP pose is stale"
        if sample.now_s - sample.tf_stamp_s > self.config.max_tf_age_s:
            return "camera-to-command transform is stale"
        if max(sample.color_stamp_s, sample.depth_stamp_s, sample.pose_stamp_s, sample.tf_stamp_s) - sample.now_s > 0.05:
            return "sensor timestamp is implausibly in the future"
        if sample.valid_depth_fraction < self.config.minimum_valid_depth_fraction:
            return "valid depth fraction is below the configured minimum"
        if self._trial_start_s is None or sample.now_s - self._trial_start_s > self.config.maximum_trial_duration_s:
            return "visual-servo trial timed out"
        if any(abs(value) > 1.0 + 1.0e-6 for value in sample.requested_normalized_action):
            return "policy action exceeds normalized bounds"
        if any(
            value < low or value > high
            for value, low, high in zip(
                sample.tcp_position_m,
                self.config.workspace_min_xyz_m,
                self.config.workspace_max_xyz_m,
                strict=True,
            )
        ):
            return "TCP is outside the configured Cartesian workspace"
        if self.config.joint_position_limits_rad:
            if len(sample.joint_positions_rad) != len(self.config.joint_position_limits_rad):
                return "joint-position sample does not match configured limits"
            if any(
                value < low or value > high
                for value, (low, high) in zip(
                    sample.joint_positions_rad,
                    self.config.joint_position_limits_rad,
                    strict=True,
                )
            ):
                return "joint position exceeds a configured limit"
        for values, limits, label in (
            (sample.joint_velocities_rad_s, self.config.max_joint_velocity_rad_s, "velocity"),
            (sample.joint_accelerations_rad_s2, self.config.max_joint_acceleration_rad_s2, "acceleration"),
        ):
            if limits:
                if len(values) != len(limits):
                    return f"joint-{label} sample does not match configured limits"
                if any(abs(value) > limit for value, limit in zip(values, limits, strict=True)):
                    return f"joint {label} exceeds a configured limit"
        if self.config.require_force_measurement and sample.force_norm_n is None:
            return "required force measurement is unavailable"
        if sample.force_norm_n is not None:
            if not math.isfinite(float(sample.force_norm_n)):
                return "force measurement is non-finite"
            if sample.force_norm_n > self.config.force_abort_threshold_n:
                return "force/contact abort threshold exceeded"
        return None

    def _decision(
        self,
        action: tuple[float, float, float, float, float, float],
        *,
        terminal: bool,
    ) -> VisualServoSafetyDecision:
        return VisualServoSafetyDecision(
            state=self.state,
            applied_normalized_action=action,
            completion_streak=self.completion_streak,
            reason=self.reason,
            terminal=terminal,
        )


__all__ = [
    "PoseVelocityEstimator",
    "VisualServoSafetyConfig",
    "VisualServoSafetyDecision",
    "VisualServoSafetySample",
    "VisualServoSafetySupervisor",
    "VisualServoState",
    "slew_limit_normalized_action",
]
