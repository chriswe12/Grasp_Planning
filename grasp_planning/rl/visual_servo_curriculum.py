"""Simulator-independent first-curriculum geometry and dataset helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from grasp_planning.rl.policy_timing import POLICY_RATE_HZ


@dataclass(frozen=True)
class VisualServoCurriculumConfig:
    policy_hz: float = POLICY_RATE_HZ
    approach_duration_s: float = 2.0
    capture_duration_s: float = 1.50
    settle_duration_s: float = 1.0
    precision_duration_s: float = 1.50
    translation_gain: float = 2.0
    rotation_gain: float = 2.0
    max_linear_velocity_m_s: float = 0.05
    max_angular_velocity_rad_s: float = 0.30
    dls_damping: float = 0.05
    initial_joint_noise_rad: float = 0.025
    object_translation_xy_m: float = 0.015
    object_yaw_deg: float = 15.0
    success_position_tolerance_m: float = 0.006
    success_rotation_tolerance_deg: float = 4.0
    funnel_near_progress: float = 0.65
    funnel_far_translation_gain: float = 2.00
    funnel_near_translation_gain: float = 2.50
    funnel_far_rotation_gain: float = 0.35
    funnel_near_rotation_gain: float = 3.00
    funnel_linear_derivative_gain: float = 0.50
    funnel_angular_derivative_gain: float = 0.25
    funnel_far_half_width_m: float = 0.008
    funnel_near_half_width_m: float = 0.0005
    funnel_approach_slow_error_m: float = 0.004
    funnel_approach_stop_error_m: float = 0.012
    precision_start_progress: float = 0.995
    precision_translation_gain: float = 3.0
    precision_rotation_gain: float = 2.0
    precision_linear_derivative_gain: float = 0.70
    precision_angular_derivative_gain: float = 0.35
    precision_max_linear_velocity_m_s: float = 0.012
    precision_max_angular_velocity_rad_s: float = 0.10
    precision_max_command_lead_m: float = 0.002

    def __post_init__(self) -> None:
        if self.policy_hz <= 0.0 or self.approach_duration_s <= 0.0:
            raise ValueError("Policy rate and approach duration must be positive.")
        if self.capture_duration_s < 0.0 or self.settle_duration_s < 0.0 or self.precision_duration_s < 0.0:
            raise ValueError("Capture, settle, and precision durations must be nonnegative.")
        if self.max_linear_velocity_m_s <= 0.0 or self.max_angular_velocity_rad_s <= 0.0:
            raise ValueError("Velocity limits must be positive.")
        if self.dls_damping <= 0.0:
            raise ValueError("DLS damping must be positive.")
        if not 0.0 < self.funnel_near_progress < 1.0:
            raise ValueError("Funnel near progress must lie strictly between zero and one.")
        if not 0.0 < self.precision_start_progress <= 1.0:
            raise ValueError("Precision start progress must lie in (0, 1].")
        if not 0.0 <= self.funnel_near_half_width_m <= self.funnel_far_half_width_m:
            raise ValueError("Funnel half widths must be nonnegative and shrink near the grasp.")
        if self.funnel_linear_derivative_gain < 0.0 or self.funnel_angular_derivative_gain < 0.0:
            raise ValueError("Funnel derivative gains must be nonnegative.")
        if not 0.0 <= self.funnel_approach_slow_error_m < self.funnel_approach_stop_error_m:
            raise ValueError("Funnel approach thresholds must be ordered and nonnegative.")

    @property
    def policy_dt_s(self) -> float:
        return 1.0 / self.policy_hz

    @property
    def step_count(self) -> int:
        total_duration_s = (
            self.capture_duration_s
            + self.approach_duration_s
            + self.settle_duration_s
            + self.precision_duration_s
        )
        return max(2, int(round(total_duration_s * self.policy_hz)) + 1)


def smooth_trajectory_progress(elapsed_s: float, duration_s: float) -> tuple[float, float]:
    """Return quintic trajectory progress and progress rate in 1/s."""

    duration = float(duration_s)
    if duration <= 0.0:
        raise ValueError("Trajectory duration must be positive.")
    alpha = float(np.clip(float(elapsed_s) / duration, 0.0, 1.0))
    progress = alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))
    if alpha <= 0.0 or alpha >= 1.0:
        return progress, 0.0
    progress_rate = 30.0 * alpha**2 * (1.0 - alpha) ** 2 / duration
    return progress, progress_rate


def _normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("Quaternion norm must be nonzero.")
    quaternion = quaternion / norm
    return quaternion if quaternion[3] >= 0.0 else -quaternion


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _quat_conjugate_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    return np.array([-x, -y, -z, w], dtype=np.float64)


def interpolate_pose(
    start_position: np.ndarray,
    start_orientation_xyzw: np.ndarray,
    goal_position: np.ndarray,
    goal_orientation_xyzw: np.ndarray,
    progress: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate position and shortest-path quaternion at normalized progress."""

    alpha = float(np.clip(progress, 0.0, 1.0))
    start_position = np.asarray(start_position, dtype=np.float64)
    goal_position = np.asarray(goal_position, dtype=np.float64)
    q0 = _normalize_quaternion_xyzw(start_orientation_xyzw)
    q1 = _normalize_quaternion_xyzw(goal_orientation_xyzw)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        quaternion = _normalize_quaternion_xyzw((1.0 - alpha) * q0 + alpha * q1)
    else:
        angle = float(np.arccos(dot))
        quaternion = (
            np.sin((1.0 - alpha) * angle) * q0 + np.sin(alpha * angle) * q1
        ) / np.sin(angle)
    return (1.0 - alpha) * start_position + alpha * goal_position, quaternion


def pose_error_twist(
    current_position: np.ndarray,
    current_orientation_xyzw: np.ndarray,
    target_position: np.ndarray,
    target_orientation_xyzw: np.ndarray,
) -> np.ndarray:
    """Return a world-frame SE(3) small-angle error [translation, rotation]."""

    current_q = _normalize_quaternion_xyzw(current_orientation_xyzw)
    target_q = _normalize_quaternion_xyzw(target_orientation_xyzw)
    error_q = _quat_multiply_xyzw(target_q, _quat_conjugate_xyzw(current_q))
    if error_q[3] < 0.0:
        error_q = -error_q
    vector_norm = float(np.linalg.norm(error_q[:3]))
    if vector_norm <= 1.0e-9:
        rotation_error = 2.0 * error_q[:3]
    else:
        angle = 2.0 * np.arctan2(vector_norm, float(error_q[3]))
        rotation_error = error_q[:3] * (angle / vector_norm)
    return np.concatenate(
        (np.asarray(target_position, dtype=np.float64) - np.asarray(current_position, dtype=np.float64), rotation_error)
    )


def clamp_twist(twist: np.ndarray, *, max_linear: float, max_angular: float) -> np.ndarray:
    result = np.asarray(twist, dtype=np.float64).copy()
    for segment, limit in ((slice(0, 3), max_linear), (slice(3, 6), max_angular)):
        norm = float(np.linalg.norm(result[segment]))
        if norm > float(limit):
            result[segment] *= float(limit) / norm
    return result


def _rotation_matrix_from_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion_xyzw(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _smoothstep01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def alignment_funnel_expert_twist(
    *,
    nominal_twist: np.ndarray,
    pose_error: np.ndarray,
    grasp_orientation_xyzw: np.ndarray,
    trajectory_progress: float,
    config: VisualServoCurriculumConfig,
    measured_twist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return a phase-aware expert action and scalar diagnostics.

    Grasp-frame x/y are transverse to the local-z approach axis.  The allowed
    transverse error shrinks near the object.  Nominal approach motion is
    slowed or stopped when the TCP lies outside that funnel, while feedback
    gains rise only during the final part of the approach.
    """

    progress = float(np.clip(trajectory_progress, 0.0, 1.0))
    near_phase = _smoothstep01(
        (progress - config.funnel_near_progress) / (1.0 - config.funnel_near_progress)
    )
    rotation_world_from_grasp = _rotation_matrix_from_quaternion_xyzw(
        grasp_orientation_xyzw
    )
    position_error_world = np.asarray(pose_error[:3], dtype=np.float64)
    position_error_grasp = rotation_world_from_grasp.T @ position_error_world
    transverse_error = position_error_grasp[:2]
    transverse_error_norm = float(np.linalg.norm(transverse_error))

    half_width = (
        config.funnel_far_half_width_m * (1.0 - near_phase)
        + config.funnel_near_half_width_m * near_phase
    )
    if transverse_error_norm <= half_width or transverse_error_norm <= 1.0e-12:
        controlled_transverse_error = np.zeros(2, dtype=np.float64)
    else:
        controlled_transverse_error = transverse_error * (
            (transverse_error_norm - half_width) / transverse_error_norm
        )

    translation_gain = (
        config.funnel_far_translation_gain * (1.0 - near_phase)
        + config.funnel_near_translation_gain * near_phase
    )
    rotation_gain = (
        config.funnel_far_rotation_gain * (1.0 - near_phase)
        + config.funnel_near_rotation_gain * near_phase
    )
    controlled_position_error_grasp = position_error_grasp.copy()
    controlled_position_error_grasp[:2] = controlled_transverse_error
    correction_world = rotation_world_from_grasp @ (
        translation_gain * controlled_position_error_grasp
    )

    corridor_excess = max(0.0, transverse_error_norm - half_width)
    slow_excess = max(
        0.0,
        config.funnel_approach_slow_error_m - config.funnel_far_half_width_m,
    )
    stop_excess = max(
        slow_excess + 1.0e-6,
        config.funnel_approach_stop_error_m - config.funnel_far_half_width_m,
    )
    if corridor_excess <= slow_excess:
        approach_scale = 1.0
    elif corridor_excess >= stop_excess:
        approach_scale = 0.0
    else:
        approach_scale = 1.0 - (
            corridor_excess - slow_excess
        ) / (stop_excess - slow_excess)
        approach_scale = _smoothstep01(approach_scale)

    nominal = np.asarray(nominal_twist, dtype=np.float64).copy()
    nominal[:3] *= approach_scale
    measured = (
        nominal.copy()
        if measured_twist is None
        else np.asarray(measured_twist, dtype=np.float64)
    )
    velocity_error = nominal - measured
    residual_raw = np.concatenate(
        (correction_world, rotation_gain * np.asarray(pose_error[3:], dtype=np.float64))
    )
    residual_raw[:3] += config.funnel_linear_derivative_gain * velocity_error[:3]
    residual_raw[3:] += config.funnel_angular_derivative_gain * velocity_error[3:]
    full = clamp_twist(
        nominal + residual_raw,
        max_linear=config.max_linear_velocity_m_s,
        max_angular=config.max_angular_velocity_rad_s,
    )
    residual = full - np.asarray(nominal_twist, dtype=np.float64)
    diagnostics = {
        "near_phase": near_phase,
        "funnel_half_width_m": half_width,
        "transverse_error_m": transverse_error_norm,
        "approach_scale": approach_scale,
        "translation_gain": translation_gain,
        "rotation_gain": rotation_gain,
        "measured_linear_speed_m_s": float(np.linalg.norm(measured[:3])),
        "measured_angular_speed_rad_s": float(np.linalg.norm(measured[3:])),
    }
    return full, residual, diagnostics


def precision_docking_expert_twist(
    *,
    pose_error: np.ndarray,
    measured_twist: np.ndarray,
    config: VisualServoCurriculumConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return a low-speed Cartesian PD command for exact final-pose docking."""

    error = np.asarray(pose_error, dtype=np.float64)
    measured = np.asarray(measured_twist, dtype=np.float64)
    command = np.concatenate(
        (
            config.precision_translation_gain * error[:3]
            - config.precision_linear_derivative_gain * measured[:3],
            config.precision_rotation_gain * error[3:]
            - config.precision_angular_derivative_gain * measured[3:],
        )
    )
    command = clamp_twist(
        command,
        max_linear=config.precision_max_linear_velocity_m_s,
        max_angular=config.precision_max_angular_velocity_rad_s,
    )
    return command, {
        "position_error_m": float(np.linalg.norm(error[:3])),
        "rotation_error_rad": float(np.linalg.norm(error[3:])),
        "measured_linear_speed_m_s": float(np.linalg.norm(measured[:3])),
        "measured_angular_speed_rad_s": float(np.linalg.norm(measured[3:])),
    }


def expert_twist(
    *,
    nominal_twist: np.ndarray,
    pose_error: np.ndarray,
    config: VisualServoCurriculumConfig,
    grasp_orientation_xyzw: np.ndarray | None = None,
    trajectory_progress: float | None = None,
    measured_twist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded full and residual expert twists in world coordinates.

    Supplying grasp orientation and progress selects the alignment-funnel
    controller.  The two-argument behavior remains available for old datasets
    and callers.
    """

    if grasp_orientation_xyzw is not None and trajectory_progress is not None:
        full, residual, _ = alignment_funnel_expert_twist(
            nominal_twist=nominal_twist,
            pose_error=pose_error,
            grasp_orientation_xyzw=grasp_orientation_xyzw,
            trajectory_progress=trajectory_progress,
            config=config,
            measured_twist=measured_twist,
        )
        return full, residual

    residual = np.asarray(pose_error, dtype=np.float64).copy()
    residual[:3] *= config.translation_gain
    residual[3:] *= config.rotation_gain
    full = clamp_twist(
        np.asarray(nominal_twist, dtype=np.float64) + residual,
        max_linear=config.max_linear_velocity_m_s,
        max_angular=config.max_angular_velocity_rad_s,
    )
    return full, full - np.asarray(nominal_twist, dtype=np.float64)


def write_episode_npz(
    output_dir: str | Path,
    *,
    episode_index: int,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    config: VisualServoCurriculumConfig,
) -> tuple[Path, Path]:
    """Write one compressed episode and adjacent JSON metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{episode_index:06d}"
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"
    np.savez_compressed(npz_path, **arrays)
    payload = {
        "schema_version": 1,
        "episode_index": int(episode_index),
        "split": "validation" if episode_index % 10 == 0 else "train",
        "config": asdict(config),
        **metadata,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return npz_path, json_path
