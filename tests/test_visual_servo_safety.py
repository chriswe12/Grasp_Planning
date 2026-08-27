from __future__ import annotations

from dataclasses import replace

import numpy as np

from grasp_planning.ros2.visual_servo_safety import (
    PoseVelocityEstimator,
    VisualServoSafetyConfig,
    VisualServoSafetySample,
    VisualServoSafetySupervisor,
    VisualServoState,
    slew_limit_normalized_action,
)


def _sample(**changes) -> VisualServoSafetySample:
    sample = VisualServoSafetySample(
        now_s=10.0,
        color_stamp_s=9.98,
        depth_stamp_s=9.981,
        pose_stamp_s=9.99,
        tf_stamp_s=9.98,
        valid_depth_fraction=0.75,
        requested_normalized_action=(0.2, -0.2, 0.1, 0.0, 0.0, 0.0),
        completion_probability=0.25,
        tcp_position_m=(0.5, 0.0, 0.3),
        tcp_linear_speed_m_s=0.001,
        tcp_angular_speed_rad_s=0.005,
        joint_positions_rad=(0.0,) * 7,
        joint_velocities_rad_s=(0.0,) * 7,
        joint_accelerations_rad_s2=(0.0,) * 7,
        force_norm_n=1.0,
        deadman_active=True,
        emergency_stop_active=False,
        command_consumer_exists=True,
        servo_healthy=True,
        servo_status_age_s=0.01,
    )
    return replace(sample, **changes)


def _supervisor() -> VisualServoSafetySupervisor:
    config = VisualServoSafetyConfig(
        workspace_min_xyz_m=(0.2, -0.6, 0.02),
        workspace_max_xyz_m=(0.8, 0.6, 0.9),
        joint_position_limits_rad=((-3.0, 3.0),) * 7,
        max_joint_velocity_rad_s=(1.0,) * 7,
        max_joint_acceleration_rad_s2=(2.0,) * 7,
        force_abort_threshold_n=10.0,
    )
    supervisor = VisualServoSafetySupervisor(config)
    supervisor.mark_ready()
    supervisor.arm(now_s=9.0)
    return supervisor


def test_action_slew_limit_matches_training_contract() -> None:
    action = slew_limit_normalized_action((1.0, -1.0, 0.1, 0.0, 0.5, -0.5), (0.0,) * 6)

    np.testing.assert_allclose(action, (0.25, -0.25, 0.1, 0.0, 0.25, -0.25))


def test_completion_requires_four_low_speed_frames_and_holds_immediately() -> None:
    supervisor = _supervisor()

    for streak in range(1, 4):
        decision = supervisor.evaluate(_sample(completion_probability=0.96))
        assert decision.state == VisualServoState.CANDIDATE_HOLD
        assert decision.completion_streak == streak
        assert decision.applied_normalized_action == (0.0,) * 6
        assert not decision.terminal

    decision = supervisor.evaluate(_sample(completion_probability=0.96))

    assert decision.state == VisualServoState.COMPLETED_HOLD
    assert decision.terminal


def test_stale_image_latches_zero_safety_hold() -> None:
    supervisor = _supervisor()

    decision = supervisor.evaluate(_sample(color_stamp_s=9.0, depth_stamp_s=9.0))
    repeated = supervisor.evaluate(_sample())

    assert decision.state == VisualServoState.SAFETY_HOLD
    assert decision.applied_normalized_action == (0.0,) * 6
    assert repeated.state == VisualServoState.SAFETY_HOLD
    assert repeated.terminal


def test_pose_velocity_estimator_uses_translation_and_quaternion_delta() -> None:
    estimator = PoseVelocityEstimator(smoothing_alpha=1.0)
    estimator.update(stamp_s=1.0, position_m=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0))

    linear, angular = estimator.update(
        stamp_s=1.1,
        position_m=(0.01, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, np.sin(0.05), np.cos(0.05)),
    )

    assert abs(linear - 0.1) < 1.0e-9
    assert abs(angular - 1.0) < 1.0e-9
    np.testing.assert_allclose(estimator.linear_velocity_m_s, (0.1, 0.0, 0.0), atol=1.0e-9)
    np.testing.assert_allclose(estimator.angular_velocity_rad_s, (0.0, 0.0, 1.0), atol=1.0e-9)
