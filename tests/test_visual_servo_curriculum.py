from __future__ import annotations

import numpy as np

from grasp_planning.rl.visual_servo_curriculum import (
    VisualServoCurriculumConfig,
    alignment_funnel_expert_twist,
    expert_twist,
    interpolate_pose,
    pose_error_twist,
    precision_docking_expert_twist,
    smooth_trajectory_progress,
    write_episode_npz,
)


def test_interpolate_pose_endpoints_and_midpoint() -> None:
    start = np.array([0.0, 0.0, 0.1])
    goal = np.array([0.0, 0.0, 0.0])
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    midpoint, midpoint_q = interpolate_pose(start, quaternion, goal, quaternion, 0.5)
    assert np.allclose(midpoint, [0.0, 0.0, 0.05])
    assert np.allclose(midpoint_q, quaternion)


def test_pose_error_and_expert_store_residual_consistently() -> None:
    config = VisualServoCurriculumConfig(max_linear_velocity_m_s=1.0, max_angular_velocity_rad_s=1.0)
    error = pose_error_twist(
        np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([0.01, -0.02, 0.03]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    nominal = np.array([0.0, 0.0, -0.04, 0.0, 0.0, 0.0])
    full, residual = expert_twist(nominal_twist=nominal, pose_error=error, config=config)
    assert np.allclose(full, nominal + residual)
    assert np.allclose(residual[:3], 2.0 * error[:3])


def test_episode_writer_splits_by_episode(tmp_path) -> None:
    config = VisualServoCurriculumConfig()
    npz_path, json_path = write_episode_npz(
        tmp_path,
        episode_index=10,
        arrays={"action": np.zeros((2, 6), dtype=np.float32)},
        metadata={"success": True},
        config=config,
    )
    assert np.load(npz_path)["action"].shape == (2, 6)
    assert '"split": "validation"' in json_path.read_text(encoding="utf-8")


def test_smooth_trajectory_progress_starts_and_stops_without_velocity_jump() -> None:
    assert smooth_trajectory_progress(0.0, 2.0) == (0.0, 0.0)
    midpoint, midpoint_rate = smooth_trajectory_progress(1.0, 2.0)
    assert np.isclose(midpoint, 0.5)
    assert np.isclose(midpoint_rate, 0.9375)
    assert smooth_trajectory_progress(2.0, 2.0) == (1.0, 0.0)
    assert smooth_trajectory_progress(3.0, 2.0) == (1.0, 0.0)


def test_alignment_funnel_stops_approach_when_object_is_not_between_fingers() -> None:
    config = VisualServoCurriculumConfig()
    full, _, diagnostics = alignment_funnel_expert_twist(
        nominal_twist=np.array([0.0, 0.0, -0.04, 0.0, 0.0, 0.0]),
        pose_error=np.array([0.015, 0.0, 0.0, 0.0, 0.0, 0.0]),
        grasp_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        trajectory_progress=0.2,
        config=config,
    )
    assert diagnostics["approach_scale"] == 0.0
    assert np.isclose(full[2], 0.0)
    assert full[0] > 0.0


def test_alignment_funnel_defers_small_correction_until_near_grasp() -> None:
    config = VisualServoCurriculumConfig()
    arguments = {
        "nominal_twist": np.zeros(6),
        "pose_error": np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "grasp_orientation_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
        "config": config,
    }
    far, _, far_debug = alignment_funnel_expert_twist(
        trajectory_progress=0.2, **arguments
    )
    near, _, near_debug = alignment_funnel_expert_twist(
        trajectory_progress=0.95, **arguments
    )
    assert np.isclose(far[0], 0.0)
    assert near[0] > 0.0
    assert far_debug["funnel_half_width_m"] > near_debug["funnel_half_width_m"]
    assert far_debug["translation_gain"] < near_debug["translation_gain"]


def test_alignment_funnel_derivative_term_opposes_tcp_overshoot() -> None:
    config = VisualServoCurriculumConfig()
    arguments = {
        "nominal_twist": np.zeros(6),
        "pose_error": np.array([0.0, 0.0, -0.003, 0.0, 0.0, 0.0]),
        "grasp_orientation_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
        "trajectory_progress": 1.0,
        "config": config,
    }
    undamped, _, _ = alignment_funnel_expert_twist(**arguments)
    damped, _, debug = alignment_funnel_expert_twist(
        measured_twist=np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0]),
        **arguments,
    )
    assert abs(damped[2]) < abs(undamped[2])
    assert np.isclose(debug["measured_linear_speed_m_s"], 0.02)


def test_precision_docking_is_slow_and_derivative_damped() -> None:
    config = VisualServoCurriculumConfig()
    command, debug = precision_docking_expert_twist(
        pose_error=np.array([0.01, 0.0, -0.004, 0.0, 0.0, 0.1]),
        measured_twist=np.array([0.02, 0.0, -0.01, 0.0, 0.0, 0.05]),
        config=config,
    )
    assert np.linalg.norm(command[:3]) <= config.precision_max_linear_velocity_m_s + 1.0e-12
    assert np.linalg.norm(command[3:]) <= config.precision_max_angular_velocity_rad_s + 1.0e-12
    assert command[2] < 0.0
    assert np.isclose(debug["measured_linear_speed_m_s"], np.sqrt(0.0005))
