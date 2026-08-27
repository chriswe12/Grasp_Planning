from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import yaml

from grasp_planning.d405_wrist_camera import pdz_camera_rotation_in_link7
from grasp_planning.ros2 import d405_visual_servo
from grasp_planning.ros2.d405_rgbd_subscriber import SynchronizedD405Frame
from grasp_planning.ros2.d405_visual_servo import D405VisualServoDeploymentConfig


def _write_config(tmp_path: Path, **changes) -> Path:
    files = {}
    for key, name in (
        ("checkpoint_path", "checkpoint.pth"),
        ("checkpoint_metadata_path", "checkpoint.json"),
        ("agent_config_path", "agent.yaml"),
        ("goal_observation_path", "runtime_goal.npz"),
    ):
        path = tmp_path / name
        path.write_bytes(b"fixture")
        files[key] = path.name
    payload = {
        **files,
        "goal_renderer_launcher": "run_mujoco_filament.sh",
        "goal_renderer_python_command": "python3",
        "goal_renderer_script": "renderer.py",
        "goal_renderer_robot_urdf": "robot.urdf",
        "goal_renderer_backend": "filament",
        "command_sink": "dry_run",
        "require_deadman": False,
        "require_force_measurement": False,
        **changes,
    }
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(yaml.safe_dump({"visual_servo": payload}), encoding="utf-8")
    return config_path


def test_dry_run_config_resolves_artifact_paths_relative_to_yaml(tmp_path: Path) -> None:
    config = D405VisualServoDeploymentConfig.from_yaml(_write_config(tmp_path))

    assert config.command_sink == "dry_run"
    assert config.checkpoint_path == (tmp_path / "checkpoint.pth").resolve()
    assert config.goal_observation_path == (tmp_path / "runtime_goal.npz").resolve()
    assert not config.real_motion_approved
    assert config.image_transport == "raw"
    assert config.tcp_frame == "pdz_gripper_tcp"
    assert config.policy_rate_hz == 15.0
    assert config.max_joint_state_age_s == 0.15
    assert config.max_force_age_s == 0.15
    assert config.max_operator_signal_age_s == 0.25
    assert config.goal_renderer_backend == "filament"
    assert config.goal_renderer_launcher == (tmp_path / "run_mujoco_filament.sh").resolve()
    assert config.goal_renderer_robot_urdf == (tmp_path / "robot.urdf").resolve()
    assert config.expected_camera_serial == "260522275434"
    assert not config.allow_pdz_camera_rotation_fallback


def test_moveit_servo_config_fails_closed_without_explicit_motion_approval(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, command_sink="moveit_servo")

    try:
        D405VisualServoDeploymentConfig.from_yaml(config_path)
    except ValueError as exc:
        assert "real_motion_approved" in str(exc)
    else:
        raise AssertionError("Expected MoveIt Servo output to require explicit real-motion approval.")


def test_moveit_servo_config_allows_reviewed_cell_without_application_heartbeats(
    tmp_path: Path,
) -> None:
    config = D405VisualServoDeploymentConfig.from_yaml(
        _write_config(
            tmp_path,
            command_sink="moveit_servo",
            real_motion_approved=True,
            expected_joint_names=["lbr_A1"],
            joint_position_limits_rad=[[-2.97, 2.97]],
            force_topic="/lbr/force_torque_broadcaster/wrench",
            force_abort_threshold_n=10.0,
            require_force_measurement=True,
            require_deadman=False,
            deadman_topic="",
            emergency_stop_topic="",
        )
    )

    assert config.command_sink == "moveit_servo"
    assert not config.require_deadman
    assert config.deadman_topic == ""
    assert config.emergency_stop_topic == ""


def test_required_deadman_still_requires_a_topic(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        command_sink="moveit_servo",
        real_motion_approved=True,
        expected_joint_names=["lbr_A1"],
        joint_position_limits_rad=[[-2.97, 2.97]],
        force_topic="/lbr/force_torque_broadcaster/wrench",
        force_abort_threshold_n=10.0,
        require_force_measurement=True,
        require_deadman=True,
        deadman_topic="",
        emergency_stop_topic="",
    )

    try:
        D405VisualServoDeploymentConfig.from_yaml(config_path)
    except ValueError as exc:
        assert "deadman_topic" in str(exc)
    else:
        raise AssertionError("Expected a required deadman to require a topic.")


def test_tcp_pose_feedback_is_derived_from_robot_tf(monkeypatch) -> None:
    transform = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=500_000_000)),
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.4, y=-0.1, z=0.3),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )
    node = object.__new__(d405_visual_servo.D405VisualServoNode)
    node.config = SimpleNamespace(command_frame="lbr_link_0", tcp_frame="gripper_tcp")
    node.tf_buffer = mock.Mock()
    node.tf_buffer.lookup_transform.return_value = transform
    node.velocity_estimator = mock.Mock()
    node.velocity_estimator.update.return_value = (0.012, 0.034)
    node.velocity_estimator.linear_velocity_m_s = (0.01, 0.0, 0.0)
    node.velocity_estimator.angular_velocity_rad_s = (0.0, 0.02, 0.0)
    node._feedback_lock = d405_visual_servo.threading.Lock()
    node.pose_stamp_s = None
    node.tcp_position_m = None
    node.tcp_linear_speed_m_s = 0.0
    node.tcp_angular_speed_rad_s = 0.0
    node.tcp_twist_command = (0.0,) * 6
    node._fault = mock.Mock()
    monkeypatch.setattr(d405_visual_servo, "Time", mock.Mock(return_value="latest"))

    node._refresh_tcp_pose_from_tf()

    node.tf_buffer.lookup_transform.assert_called_once_with(
        "lbr_link_0",
        "gripper_tcp",
        "latest",
    )
    node.velocity_estimator.update.assert_called_once_with(
        stamp_s=12.5,
        position_m=(0.4, -0.1, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert node.pose_stamp_s == 12.5
    assert node.tcp_position_m == (0.4, -0.1, 0.3)
    assert node.tcp_linear_speed_m_s == 0.012
    assert node.tcp_angular_speed_rad_s == 0.034
    assert node.tcp_twist_command == (0.01, 0.0, 0.0, 0.0, 0.02, 0.0)
    node._fault.assert_not_called()


def test_policy_step_refreshes_clock_after_inference_and_snapshots_feedback() -> None:
    node = object.__new__(d405_visual_servo.D405VisualServoNode)
    node.config = SimpleNamespace(
        expected_joint_names=("lbr_A1",),
        require_force_measurement=True,
        require_deadman=False,
        emergency_stop_topic="",
        max_joint_state_age_s=0.5,
        max_force_age_s=0.5,
        max_operator_signal_age_s=0.25,
        linear_action_scale_m_s=0.04,
        angular_action_scale_rad_s=0.24,
        first_test_speed_fraction=0.25,
        command_frame="lbr_link_0",
        command_sink="moveit_servo",
    )
    node._feedback_lock = d405_visual_servo.threading.Lock()
    node.pose_stamp_s = 9.99
    node.tcp_position_m = (0.5, 0.0, 0.2)
    node.tcp_linear_speed_m_s = 0.0
    node.tcp_angular_speed_rad_s = 0.0
    node.tcp_twist_command = (0.0,) * 6
    node.joint_positions_rad = (0.0,)
    node.joint_velocities_rad_s = (0.0,)
    node.joint_accelerations_rad_s2 = (0.0,)
    node.joint_stamp_s = 9.99
    node.force_norm_n = 1.0
    node.force_stamp_s = 9.99
    node.deadman_active = True
    node.deadman_receipt_s = None
    node.emergency_stop_active = False
    node.emergency_stop_receipt_s = None
    node.now_seconds = mock.Mock(side_effect=[10.0, 10.3])

    inference = SimpleNamespace(
        valid_depth_fraction=0.8,
        requested_normalized_action=(0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
        filtered_normalized_action=(0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
        completion_probability=0.1,
    )

    def infer_and_receive_new_feedback(rgb, depth, **context):
        del rgb, depth
        assert context["tcp_twist_camera"] == pytest.approx((0.0,) * 6)
        np.testing.assert_allclose(context["rotation_base_from_camera"], np.eye(3))
        # This reproduces the multithreaded runtime: feedback advances while
        # CUDA inference is blocking the control callback.
        with node._feedback_lock:
            node.pose_stamp_s = 10.25
            node.joint_stamp_s = 10.25
            node.force_stamp_s = 10.25
        return inference

    node.runtime = SimpleNamespace(
        infer=mock.Mock(side_effect=infer_and_receive_new_feedback),
        commit_applied_action=mock.Mock(),
        policy_context_mode="action_twist",
    )
    decision = SimpleNamespace(
        applied_normalized_action=(0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
        completion_streak=0,
        state=d405_visual_servo.VisualServoState.RUNNING,
        reason="running",
        terminal=False,
    )
    node.supervisor = SimpleNamespace(evaluate=mock.Mock(return_value=decision))
    health = SimpleNamespace(
        consumer_exists=True,
        healthy=True,
        status_code=0,
        status_text="no_warning",
        status_age_s=0.0,
    )
    node.sink = SimpleNamespace(
        is_real=False,
        health=mock.Mock(return_value=health),
        send_twist=mock.Mock(return_value=True),
    )
    node._lookup_camera_rotation = mock.Mock(return_value=(np.eye(3), 10.25))
    node.latest_frame_receipt_s = 9.99
    node.camera_rotation_source = "pdz_cad_link7_fallback"
    node._last_command_twist = (0.0,) * 6
    node.nonzero_command_sent = False
    node.step_count = 0
    node.terminal = False
    node._record_step = mock.Mock()
    frame = SynchronizedD405Frame(
        rgb_uint8=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_z16=np.zeros((2, 2), dtype=np.uint16),
        color_stamp_s=9.90,
        depth_stamp_s=9.90,
        camera_frame_id="camera_color_optical_frame",
    )

    node._process_frame(frame)

    safety_sample = node.supervisor.evaluate.call_args.args[0]
    assert safety_sample.now_s == 10.3
    assert safety_sample.pose_stamp_s == 10.25
    assert safety_sample.now_s - safety_sample.pose_stamp_s == pytest.approx(0.05)
    step = node._record_step.call_args.args[0]
    assert step["joint_state_age_s"] == pytest.approx(0.05)
    assert step["force_age_s"] == pytest.approx(0.05)
    assert step["policy_inference_duration_s"] >= 0.0
    assert step["control_preparation_duration_s"] >= 0.0


def test_rgbd_watchdog_bounds_original_sensor_age_not_only_receipt_age() -> None:
    frame = SynchronizedD405Frame(
        rgb_uint8=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_z16=np.zeros((2, 2), dtype=np.uint16),
        color_stamp_s=9.40,
        depth_stamp_s=9.41,
        camera_frame_id="camera_color_optical_frame",
    )

    violation = d405_visual_servo.rgbd_watchdog_violation(
        now_s=10.0,
        latest_frame=frame,
        latest_receipt_s=9.99,
        maximum_age_s=0.50,
    )

    assert violation == "RGB-D source frame is stale"


def test_rgbd_watchdog_accepts_fresh_compressed_frame() -> None:
    frame = SynchronizedD405Frame(
        rgb_uint8=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_z16=np.zeros((2, 2), dtype=np.uint16),
        color_stamp_s=9.75,
        depth_stamp_s=9.76,
        camera_frame_id="camera_color_optical_frame",
    )

    violation = d405_visual_servo.rgbd_watchdog_violation(
        now_s=10.0,
        latest_frame=frame,
        latest_receipt_s=9.98,
        maximum_age_s=0.50,
    )

    assert violation is None


def test_rgbd_watchdog_can_warn_only_for_source_age_but_keeps_receipt_watchdog() -> None:
    frame = SynchronizedD405Frame(
        rgb_uint8=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_z16=np.zeros((2, 2), dtype=np.uint16),
        color_stamp_s=9.40,
        depth_stamp_s=9.41,
        camera_frame_id="camera_color_optical_frame",
    )

    accepted = d405_visual_servo.rgbd_watchdog_violation(
        now_s=10.0,
        latest_frame=frame,
        latest_receipt_s=9.99,
        maximum_age_s=0.50,
        enforce_source_image_age=False,
    )
    stalled = d405_visual_servo.rgbd_watchdog_violation(
        now_s=10.0,
        latest_frame=frame,
        latest_receipt_s=9.40,
        maximum_age_s=0.50,
        enforce_source_image_age=False,
    )

    assert accepted is None
    assert stalled == "RGB-D command watchdog expired"


def test_disconnected_camera_tf_can_use_opt_in_pdz_rotation_fallback(monkeypatch) -> None:
    class TestTransformException(Exception):
        pass

    link7_transform = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=9, nanosec=990_000_000)),
        transform=SimpleNamespace(
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )
    frame = SynchronizedD405Frame(
        rgb_uint8=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_z16=np.zeros((2, 2), dtype=np.uint16),
        color_stamp_s=9.40,
        depth_stamp_s=9.41,
        camera_frame_id="camera_color_optical_frame",
    )
    logger = mock.Mock()
    node = object.__new__(d405_visual_servo.D405VisualServoNode)
    node.config = SimpleNamespace(
        command_frame="lbr_link_0",
        camera_optical_frame="camera_color_optical_frame",
        transform_timeout_s=0.05,
        allow_pdz_camera_rotation_fallback=True,
        expected_camera_profile=d405_visual_servo.D405_VISUAL_SERVO_CAMERA_PROFILE,
    )
    node.tf_buffer = mock.Mock()
    node.tf_buffer.lookup_transform.side_effect = [
        TestTransformException("two unconnected trees"),
        link7_transform,
    ]
    node._camera_rotation_fallback_warned = False
    node.camera_rotation_source = "unresolved"
    node.get_logger = mock.Mock(return_value=logger)
    node.now_seconds = mock.Mock(return_value=10.0)
    fake_time = mock.Mock(side_effect=["image-time", "latest-time"])
    fake_duration = mock.Mock(return_value="timeout")
    monkeypatch.setattr(d405_visual_servo, "TransformException", TestTransformException)
    monkeypatch.setattr(d405_visual_servo, "Time", fake_time)
    monkeypatch.setattr(d405_visual_servo, "Duration", fake_duration)

    rotation, stamp_s = node._lookup_camera_rotation(frame=frame)

    np.testing.assert_allclose(rotation, pdz_camera_rotation_in_link7(), atol=1.0e-12)
    assert stamp_s == 9.99
    assert node.camera_rotation_source == "pdz_cad_link7_fallback"
    assert "audited PDZ CAD optical rotation" in logger.warning.call_args.args[0]
    assert node.tf_buffer.lookup_transform.call_args_list == [
        mock.call("lbr_link_0", "camera_color_optical_frame", "image-time", timeout="timeout"),
        mock.call("lbr_link_0", "lbr_link_7", "latest-time", timeout="timeout"),
    ]


def test_camera_contract_warns_for_trained_intrinsics_difference_but_continues() -> None:
    live_projection = (
        437.42193603515625,
        0.0,
        427.5228576660156,
        0.0,
        0.0,
        436.81134033203125,
        238.6438446044922,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    info = SimpleNamespace(width=848, height=480, p=live_projection)
    logger = mock.Mock()
    node = object.__new__(d405_visual_servo.D405VisualServoNode)
    node.config = SimpleNamespace(
        intrinsics_tolerance_px=3.0,
        color_topic="/color",
        depth_topic="/depth",
    )
    node.color_camera_info = info
    node.depth_camera_info = info
    node.get_logger = mock.Mock(return_value=logger)
    node.get_publishers_info_by_topic = mock.Mock(return_value=[object()])

    node.validate_camera_contract()

    assert logger.warning.call_count == 2
    assert all(
        "continuing with the live CameraInfo" in call.args[0]
        for call in logger.warning.call_args_list
    )


def test_camera_serial_difference_warns_but_does_not_gate_routing(monkeypatch) -> None:
    response = SimpleNamespace(
        values=[SimpleNamespace(string_value="_260522275434")]
    )
    future = mock.Mock()
    future.done.return_value = True
    future.exception.return_value = None
    future.result.return_value = response
    client = mock.Mock()
    client.wait_for_service.return_value = True
    client.call_async.return_value = future
    logger = mock.Mock()
    node = object.__new__(d405_visual_servo.D405VisualServoNode)
    node.config = SimpleNamespace(
        camera_parameter_node="/realsense_1/camera",
        camera_serial_parameter="serial_no",
        expected_camera_serial="different-serial",
    )
    node._serial_client = client
    node.get_logger = mock.Mock(return_value=logger)
    request_type = SimpleNamespace(Request=lambda: SimpleNamespace(names=[]))
    ros = SimpleNamespace(spin_until_future_complete=mock.Mock())
    monkeypatch.setattr(d405_visual_servo, "GetParameters", request_type)
    monkeypatch.setattr(d405_visual_servo, "rclpy", ros)

    serial = node.query_and_validate_camera_serial(timeout_s=2.0)

    assert serial == "260522275434"
    assert "continuing because camera routing is determined by topics" in (
        logger.warning.call_args.args[0]
    )


def test_policy_preparation_strict_loads_runtime_before_ros_node_creation(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    runtime = object()

    with mock.patch.object(d405_visual_servo, "D405PolicyRuntime", return_value=runtime) as runtime_type:
        preparation = d405_visual_servo.prepare_d405_policy_visual_servo(
            config_path=config_path,
            expected_grasp_id="g1973",
            expected_part_id="0",
        )

    assert preparation.runtime is runtime
    assert preparation.expected_grasp_id == "g1973"
    assert preparation.expected_part_id == "0"
    assert runtime_type.call_args.kwargs["checkpoint_path"] == (tmp_path / "checkpoint.pth").resolve()
    assert runtime_type.call_args.kwargs["expected_grasp_id"] == "g1973"
    assert runtime_type.call_args.kwargs["expected_part_id"] == "0"


def test_policy_preparation_can_bind_runtime_goal_for_live_stage2_fallback(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    fallback_goal = tmp_path / "fallback_goal.npz"
    fallback_goal.write_bytes(b"fixture")

    with mock.patch.object(d405_visual_servo, "D405PolicyRuntime") as runtime_type:
        preparation = d405_visual_servo.prepare_d405_policy_visual_servo(
            config_path=config_path,
            expected_grasp_id="g0001",
            expected_part_id="0",
            goal_observation_path_override=fallback_goal,
        )

    assert preparation.config.goal_observation_path == fallback_goal.resolve()
    assert runtime_type.call_args.kwargs["goal_observation_path"] == fallback_goal.resolve()
    assert runtime_type.call_args.kwargs["expected_grasp_id"] == "g0001"


def test_armed_policy_loop_uses_four_thread_executor(monkeypatch, tmp_path: Path) -> None:
    config = SimpleNamespace(
        command_sink="dry_run",
        startup_timeout_s=1.0,
        command_frame="lbr_link_0",
        allow_gripper_close_on_completion=False,
    )
    supervisor = SimpleNamespace(
        state=d405_visual_servo.VisualServoState.COMPLETED_HOLD,
        reason="completed",
        mark_ready=mock.Mock(),
        arm=mock.Mock(),
    )
    sink = SimpleNamespace(
        is_real=False,
        activate=mock.Mock(),
        hold=mock.Mock(),
        deactivate=mock.Mock(),
    )
    runtime = SimpleNamespace(goal=SimpleNamespace(goal_id="goal"))
    node = SimpleNamespace(
        terminal=False,
        failure_message="",
        supervisor=supervisor,
        sink=sink,
        runtime=runtime,
        nonzero_command_sent=False,
        step_count=0,
        run_directory=tmp_path,
        basic_preflight_ready=mock.Mock(return_value=True),
        basic_preflight_missing_inputs=mock.Mock(return_value=()),
        validate_camera_contract=mock.Mock(),
        query_and_validate_camera_serial=mock.Mock(return_value="camera"),
        now_seconds=mock.Mock(return_value=10.0),
        write_summary=mock.Mock(),
        destroy_node=mock.Mock(),
        _fault=mock.Mock(),
        policy_rate_gate=SimpleNamespace(reset=mock.Mock()),
    )
    preparation = SimpleNamespace(
        config=config,
        runtime=runtime,
        expected_grasp_id="g1",
        expected_part_id="0",
    )

    class FakeExecutor:
        instance = None

        def __init__(self, *, num_threads: int) -> None:
            self.num_threads = num_threads
            self.node = None
            self.spin_count = 0
            self.removed = False
            self.shutdown_timeout_s = None
            FakeExecutor.instance = self

        def add_node(self, added_node) -> None:
            self.node = added_node

        def spin_once(self, *, timeout_sec: float) -> None:
            assert timeout_sec == 0.05
            self.spin_count += 1
            self.node.terminal = True

        def remove_node(self, removed_node) -> None:
            assert removed_node is self.node
            self.removed = True

        def shutdown(self, *, timeout_sec: float) -> None:
            self.shutdown_timeout_s = timeout_sec

    monkeypatch.setattr(d405_visual_servo, "D405VisualServoNode", mock.Mock(return_value=node))
    monkeypatch.setattr(d405_visual_servo, "MultiThreadedExecutor", FakeExecutor)
    monkeypatch.setattr(d405_visual_servo.rclpy, "ok", mock.Mock(return_value=True))

    result = d405_visual_servo.run_d405_policy_visual_servo(
        config_path=tmp_path / "unused.yaml",
        expected_grasp_id="g1",
        expected_part_id="0",
        allow_real_motion=False,
        preparation=preparation,
    )

    executor = FakeExecutor.instance
    assert executor is not None
    assert executor.num_threads == 4
    assert executor.spin_count == 1
    assert executor.removed
    assert executor.shutdown_timeout_s == 1.0
    assert result.completed
    supervisor.mark_ready.assert_called_once_with()
    supervisor.arm.assert_called_once_with(now_s=10.0)
