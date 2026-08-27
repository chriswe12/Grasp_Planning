"""Real D405 policy loop with TF, safety supervision, and MoveIt Servo output."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    D405WristCameraConfig,
)
from grasp_planning.rl.d405_policy_runtime import EXPECTED_D405_SERIAL, D405PolicyRuntime
from grasp_planning.rl.policy_timing import POLICY_RATE_HZ, PolicyRateGate
from grasp_planning.ros2.d405_rgbd_subscriber import (
    D405RgbdSubscriber,
    SynchronizedD405Frame,
    ros_stamp_seconds,
)
from grasp_planning.ros2.visual_servo_command_sink import (
    DryRunCommandSink,
    MoveItServoCommandSink,
)
from grasp_planning.ros2.visual_servo_safety import (
    PoseVelocityEstimator,
    VisualServoSafetyConfig,
    VisualServoSafetySample,
    VisualServoSafetySupervisor,
    VisualServoState,
)

try:  # pragma: no cover - exercised only in a sourced ROS2 environment
    import rclpy
    from geometry_msgs.msg import PoseStamped, WrenchStamped
    from rcl_interfaces.srv import GetParameters
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, JointState
    from std_msgs.msg import Bool
    from tf2_ros import Buffer, TransformException, TransformListener
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    PoseStamped = None
    WrenchStamped = None
    GetParameters = None
    Duration = None
    Node = object
    qos_profile_sensor_data = None
    Time = None
    CameraInfo = None
    JointState = None
    Bool = None
    Buffer = None
    TransformException = Exception
    TransformListener = None


def _tuple_floats(raw: object, *, count: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != count:
        raise ValueError(f"{field_name} must contain exactly {count} values.")
    values = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field_name} must contain only finite values.")
    return values


def _tuple_pairs(raw: object, *, field_name: str) -> tuple[tuple[float, float], ...]:
    if raw in (None, "", ()):
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of [minimum, maximum] pairs.")
    return tuple(
        _tuple_floats(item, count=2, field_name=f"{field_name}[{index}]")  # type: ignore[arg-type]
        for index, item in enumerate(raw)
    )


def _resolve_config_path(raw: object, *, base_dir: Path) -> Path:
    value = str(raw or "").strip()
    if not value:
        return Path("")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


@dataclass(frozen=True)
class D405VisualServoDeploymentConfig:
    checkpoint_path: Path
    checkpoint_metadata_path: Path
    agent_config_path: Path
    goal_catalog_path: Path
    target_id: str
    model_device: str = "cuda:0"
    expected_camera_serial: str = EXPECTED_D405_SERIAL
    expected_camera_profile: str = D405_VISUAL_SERVO_CAMERA_PROFILE
    expected_observation_profile: str = D405_VISUAL_SERVO_OBSERVATION_PROFILE
    color_topic: str = "/realsense_1/camera/color/image_rect"
    depth_topic: str = "/realsense_1/camera/aligned_depth_to_color/image_rect"
    color_camera_info_topic: str = "/realsense_1/camera/color/camera_info"
    depth_camera_info_topic: str = "/realsense_1/camera/aligned_depth_to_color/camera_info"
    pose_topic: str = "/left/ee_pose"
    joint_state_topic: str = "/lbr/joint_states"
    force_topic: str = ""
    deadman_topic: str = "/d405_visual_servo/deadman"
    emergency_stop_topic: str = "/d405_visual_servo/emergency_stop"
    camera_parameter_node: str = "/realsense_1/camera"
    camera_serial_parameter: str = "serial_no"
    camera_optical_frame: str = "realsense_1_color_optical_frame"
    command_frame: str = "lbr_link_0"
    command_sink: str = "dry_run"
    moveit_servo_twist_topic: str = "/lbr/servo_node/delta_twist_cmds"
    moveit_servo_status_topic: str = "/lbr/servo_node/status"
    moveit_servo_start_service: str = "/lbr/servo_node/start_servo"
    moveit_servo_stop_service: str = "/lbr/servo_node/stop_servo"
    real_motion_approved: bool = False
    allow_gripper_close_on_completion: bool = False
    first_test_speed_fraction: float = 0.25
    linear_action_scale_m_s: float = 0.04
    angular_action_scale_rad_s: float = 0.24
    action_delta_limit: float = 0.50
    policy_rate_hz: float = POLICY_RATE_HZ
    startup_timeout_s: float = 30.0
    transform_timeout_s: float = 0.05
    intrinsics_tolerance_px: float = 3.0
    expected_joint_names: tuple[str, ...] = ()
    max_image_age_s: float = 0.15
    max_image_skew_s: float = 0.010
    max_pose_age_s: float = 0.15
    max_tf_age_s: float = 0.15
    max_servo_status_age_s: float = 0.25
    max_joint_state_age_s: float = 0.15
    max_force_age_s: float = 0.15
    max_operator_signal_age_s: float = 0.25
    minimum_valid_depth_fraction: float = 0.20
    maximum_trial_duration_s: float = 15.0
    completion_probability_threshold: float = 0.95
    completion_required_consecutive_steps: int = 4
    completion_max_linear_speed_m_s: float = 0.005
    completion_max_angular_speed_rad_s: float = 0.03
    workspace_min_xyz_m: tuple[float, float, float] = (-2.0, -2.0, -0.2)
    workspace_max_xyz_m: tuple[float, float, float] = (2.0, 2.0, 2.0)
    joint_position_limits_rad: tuple[tuple[float, float], ...] = ()
    max_joint_velocity_rad_s: tuple[float, ...] = ()
    max_joint_acceleration_rad_s2: tuple[float, ...] = ()
    force_abort_threshold_n: float = math.inf
    require_deadman: bool = True
    require_force_measurement: bool = True
    output_root: Path = Path("artifacts/d405_visual_servo_runs")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "D405VisualServoDeploymentConfig":
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"Expected a YAML mapping in '{config_path}'.")
        payload = dict(raw.get("visual_servo", raw))
        base_dir = config_path.parent
        config = cls(
            checkpoint_path=_resolve_config_path(payload.get("checkpoint_path"), base_dir=base_dir),
            checkpoint_metadata_path=_resolve_config_path(
                payload.get("checkpoint_metadata_path"),
                base_dir=base_dir,
            ),
            agent_config_path=_resolve_config_path(payload.get("agent_config_path"), base_dir=base_dir),
            goal_catalog_path=_resolve_config_path(payload.get("goal_catalog_path"), base_dir=base_dir),
            target_id=str(payload.get("target_id", "")),
            model_device=str(payload.get("model_device", "cuda:0")),
            expected_camera_serial=str(payload.get("expected_camera_serial", EXPECTED_D405_SERIAL)),
            expected_camera_profile=str(
                payload.get("expected_camera_profile", D405_VISUAL_SERVO_CAMERA_PROFILE)
            ),
            expected_observation_profile=str(
                payload.get("expected_observation_profile", D405_VISUAL_SERVO_OBSERVATION_PROFILE)
            ),
            color_topic=str(payload.get("color_topic", cls.color_topic)),
            depth_topic=str(payload.get("depth_topic", cls.depth_topic)),
            color_camera_info_topic=str(
                payload.get("color_camera_info_topic", cls.color_camera_info_topic)
            ),
            depth_camera_info_topic=str(
                payload.get("depth_camera_info_topic", cls.depth_camera_info_topic)
            ),
            pose_topic=str(payload.get("pose_topic", cls.pose_topic)),
            joint_state_topic=str(payload.get("joint_state_topic", cls.joint_state_topic)),
            force_topic=str(payload.get("force_topic", "")),
            deadman_topic=str(payload.get("deadman_topic", cls.deadman_topic)),
            emergency_stop_topic=str(
                payload.get("emergency_stop_topic", cls.emergency_stop_topic)
            ),
            camera_parameter_node=str(
                payload.get("camera_parameter_node", cls.camera_parameter_node)
            ),
            camera_serial_parameter=str(
                payload.get("camera_serial_parameter", cls.camera_serial_parameter)
            ),
            camera_optical_frame=str(payload.get("camera_optical_frame", cls.camera_optical_frame)),
            command_frame=str(payload.get("command_frame", cls.command_frame)),
            command_sink=str(payload.get("command_sink", "dry_run")).strip().lower(),
            moveit_servo_twist_topic=str(
                payload.get("moveit_servo_twist_topic", cls.moveit_servo_twist_topic)
            ),
            moveit_servo_status_topic=str(
                payload.get("moveit_servo_status_topic", cls.moveit_servo_status_topic)
            ),
            moveit_servo_start_service=str(
                payload.get("moveit_servo_start_service", cls.moveit_servo_start_service)
            ),
            moveit_servo_stop_service=str(
                payload.get("moveit_servo_stop_service", cls.moveit_servo_stop_service)
            ),
            real_motion_approved=bool(payload.get("real_motion_approved", False)),
            allow_gripper_close_on_completion=bool(
                payload.get("allow_gripper_close_on_completion", False)
            ),
            first_test_speed_fraction=float(payload.get("first_test_speed_fraction", 0.25)),
            linear_action_scale_m_s=float(payload.get("linear_action_scale_m_s", 0.04)),
            angular_action_scale_rad_s=float(payload.get("angular_action_scale_rad_s", 0.24)),
            action_delta_limit=float(payload.get("action_delta_limit", 0.50)),
            policy_rate_hz=float(payload.get("policy_rate_hz", POLICY_RATE_HZ)),
            startup_timeout_s=float(payload.get("startup_timeout_s", 30.0)),
            transform_timeout_s=float(payload.get("transform_timeout_s", 0.05)),
            intrinsics_tolerance_px=float(payload.get("intrinsics_tolerance_px", 3.0)),
            expected_joint_names=tuple(str(value) for value in payload.get("expected_joint_names", ())),
            max_image_age_s=float(payload.get("max_image_age_s", 0.15)),
            max_image_skew_s=float(payload.get("max_image_skew_s", 0.010)),
            max_pose_age_s=float(payload.get("max_pose_age_s", 0.15)),
            max_tf_age_s=float(payload.get("max_tf_age_s", 0.15)),
            max_servo_status_age_s=float(payload.get("max_servo_status_age_s", 0.25)),
            max_joint_state_age_s=float(payload.get("max_joint_state_age_s", 0.15)),
            max_force_age_s=float(payload.get("max_force_age_s", 0.15)),
            max_operator_signal_age_s=float(payload.get("max_operator_signal_age_s", 0.25)),
            minimum_valid_depth_fraction=float(payload.get("minimum_valid_depth_fraction", 0.20)),
            maximum_trial_duration_s=float(payload.get("maximum_trial_duration_s", 15.0)),
            completion_probability_threshold=float(
                payload.get("completion_probability_threshold", 0.95)
            ),
            completion_required_consecutive_steps=int(
                payload.get("completion_required_consecutive_steps", 4)
            ),
            completion_max_linear_speed_m_s=float(
                payload.get("completion_max_linear_speed_m_s", 0.005)
            ),
            completion_max_angular_speed_rad_s=float(
                payload.get("completion_max_angular_speed_rad_s", 0.03)
            ),
            workspace_min_xyz_m=_tuple_floats(
                payload.get("workspace_min_xyz_m", [-2.0, -2.0, -0.2]),
                count=3,
                field_name="workspace_min_xyz_m",
            ),  # type: ignore[arg-type]
            workspace_max_xyz_m=_tuple_floats(
                payload.get("workspace_max_xyz_m", [2.0, 2.0, 2.0]),
                count=3,
                field_name="workspace_max_xyz_m",
            ),  # type: ignore[arg-type]
            joint_position_limits_rad=_tuple_pairs(
                payload.get("joint_position_limits_rad", ()),
                field_name="joint_position_limits_rad",
            ),
            max_joint_velocity_rad_s=tuple(
                float(value) for value in payload.get("max_joint_velocity_rad_s", ())
            ),
            max_joint_acceleration_rad_s2=tuple(
                float(value) for value in payload.get("max_joint_acceleration_rad_s2", ())
            ),
            force_abort_threshold_n=float(payload.get("force_abort_threshold_n", math.inf)),
            require_deadman=bool(payload.get("require_deadman", True)),
            require_force_measurement=bool(payload.get("require_force_measurement", True)),
            output_root=_resolve_config_path(
                payload.get("output_root", "../artifacts/d405_visual_servo_runs"),
                base_dir=base_dir,
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.command_sink not in {"dry_run", "moveit_servo"}:
            raise ValueError("command_sink must be 'dry_run' or 'moveit_servo'.")
        if self.expected_camera_serial != EXPECTED_D405_SERIAL:
            raise ValueError(
                f"This policy supports only D405 serial {EXPECTED_D405_SERIAL}; got {self.expected_camera_serial}."
            )
        if not self.target_id:
            raise ValueError("target_id must select one explicit goal catalogue entry.")
        if not 0.0 < self.first_test_speed_fraction <= 1.0:
            raise ValueError("first_test_speed_fraction must lie in (0, 1].")
        if not math.isclose(self.policy_rate_hz, POLICY_RATE_HZ, abs_tol=1.0e-9):
            raise ValueError(
                f"This checkpoint/runtime contract requires policy_rate_hz={POLICY_RATE_HZ:.1f}."
            )
        for path_name in (
            "checkpoint_path",
            "checkpoint_metadata_path",
            "agent_config_path",
            "goal_catalog_path",
        ):
            path = getattr(self, path_name)
            if not path.is_file():
                raise FileNotFoundError(f"{path_name} does not exist: {path}")
        if self.command_sink == "moveit_servo":
            if not self.real_motion_approved:
                raise ValueError("MoveIt Servo output requires real_motion_approved: true.")
            if not self.expected_joint_names or not self.joint_position_limits_rad:
                raise ValueError("Real motion requires explicit joint names and position limits.")
            if len(self.expected_joint_names) != len(self.joint_position_limits_rad):
                raise ValueError("Joint names and joint-position limits must have equal length.")
            if not self.require_force_measurement or not self.force_topic:
                raise ValueError("Real motion requires force supervision and a non-empty force_topic.")
            if not self.require_deadman or not self.deadman_topic or not self.emergency_stop_topic:
                raise ValueError("Real motion requires explicit deadman and emergency-stop topics.")
        for name in ("max_joint_state_age_s", "max_force_age_s", "max_operator_signal_age_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        self.safety_config().validate()

    def safety_config(self) -> VisualServoSafetyConfig:
        return VisualServoSafetyConfig(
            max_image_age_s=self.max_image_age_s,
            max_image_skew_s=self.max_image_skew_s,
            max_pose_age_s=self.max_pose_age_s,
            max_tf_age_s=self.max_tf_age_s,
            max_servo_status_age_s=self.max_servo_status_age_s,
            minimum_valid_depth_fraction=self.minimum_valid_depth_fraction,
            maximum_trial_duration_s=self.maximum_trial_duration_s,
            completion_probability_threshold=self.completion_probability_threshold,
            completion_required_consecutive_steps=self.completion_required_consecutive_steps,
            completion_max_linear_speed_m_s=self.completion_max_linear_speed_m_s,
            completion_max_angular_speed_rad_s=self.completion_max_angular_speed_rad_s,
            workspace_min_xyz_m=self.workspace_min_xyz_m,
            workspace_max_xyz_m=self.workspace_max_xyz_m,
            joint_position_limits_rad=self.joint_position_limits_rad,
            max_joint_velocity_rad_s=self.max_joint_velocity_rad_s,
            max_joint_acceleration_rad_s2=self.max_joint_acceleration_rad_s2,
            force_abort_threshold_n=self.force_abort_threshold_n,
            require_deadman=self.require_deadman,
            require_force_measurement=self.require_force_measurement,
        )


@dataclass(frozen=True)
class D405VisualServoRunResult:
    completed: bool
    state: str
    message: str
    target_id: str
    motion_applied: bool
    allow_gripper_close: bool
    step_count: int
    run_directory: Path


def _rotation_matrix_from_quaternion_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        raise ValueError("TF quaternion norm is zero.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


class D405VisualServoNode(Node):  # pragma: no cover - ROS integration path
    """Single-trial ROS node. Policy, safety, and command adapter stay separate."""

    def __init__(
        self,
        config: D405VisualServoDeploymentConfig,
        *,
        expected_grasp_id: str,
        expected_part_id: str,
    ) -> None:
        if rclpy is None or CameraInfo is None or Buffer is None:
            raise RuntimeError("ROS2 image/TF dependencies are unavailable. Source the ROS2 workspace first.")
        super().__init__("d405_ppo_visual_servo")
        self.config = config
        self.runtime = D405PolicyRuntime(
            checkpoint_path=config.checkpoint_path,
            checkpoint_metadata_path=config.checkpoint_metadata_path,
            agent_config_path=config.agent_config_path,
            goal_catalog_path=config.goal_catalog_path,
            target_id=config.target_id,
            expected_grasp_id=expected_grasp_id,
            expected_part_id=expected_part_id,
            device=config.model_device,
            expected_camera_profile=config.expected_camera_profile,
            expected_observation_profile=config.expected_observation_profile,
            linear_action_scale_m_s=config.linear_action_scale_m_s,
            angular_action_scale_rad_s=config.angular_action_scale_rad_s,
            action_delta_limit=config.action_delta_limit,
            policy_rate_hz=config.policy_rate_hz,
            completion_probability_threshold=config.completion_probability_threshold,
            completion_required_consecutive_steps=config.completion_required_consecutive_steps,
        )
        self.supervisor = VisualServoSafetySupervisor(config.safety_config())
        self.velocity_estimator = PoseVelocityEstimator()
        self.policy_rate_gate = PolicyRateGate(config.policy_rate_hz)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.color_camera_info = None
        self.depth_camera_info = None
        self.latest_frame: SynchronizedD405Frame | None = None
        self.latest_frame_receipt_s: float | None = None
        self.pose_stamp_s: float | None = None
        self.tcp_position_m: tuple[float, float, float] | None = None
        self.tcp_linear_speed_m_s = 0.0
        self.tcp_angular_speed_rad_s = 0.0
        self.tcp_twist_command = np.zeros(6, dtype=np.float64)
        self.joint_positions_rad: tuple[float, ...] = ()
        self.joint_velocities_rad_s: tuple[float, ...] = ()
        self.joint_accelerations_rad_s2: tuple[float, ...] = ()
        self.joint_stamp_s: float | None = None
        self._previous_joint_stamp_s: float | None = None
        self._previous_joint_velocities: np.ndarray | None = None
        self.force_norm_n: float | None = None
        self.force_stamp_s: float | None = None
        self.deadman_active = not config.require_deadman
        self.deadman_receipt_s: float | None = None
        self.emergency_stop_active = False
        self.emergency_stop_receipt_s: float | None = None
        self.armed = False
        self.terminal = False
        self.step_count = 0
        self._inference_active = False
        self.failure_message = ""
        self.nonzero_command_sent = False
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.run_directory = config.output_root / f"{timestamp}-{config.target_id}"
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self._steps_path = self.run_directory / "steps.jsonl"

        self.rgbd_subscriber = D405RgbdSubscriber(
            self,
            color_topic=config.color_topic,
            depth_topic=config.depth_topic,
            maximum_skew_s=config.max_image_skew_s,
            callback=self._on_rgbd,
        )
        self.create_subscription(
            CameraInfo,
            config.color_camera_info_topic,
            self._on_color_camera_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            config.depth_camera_info_topic,
            self._on_depth_camera_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(PoseStamped, config.pose_topic, self._on_pose, qos_profile_sensor_data)
        self.create_subscription(JointState, config.joint_state_topic, self._on_joint_state, qos_profile_sensor_data)
        if config.force_topic:
            self.create_subscription(WrenchStamped, config.force_topic, self._on_force, qos_profile_sensor_data)
        if config.deadman_topic:
            self.create_subscription(Bool, config.deadman_topic, self._on_deadman, qos_profile_sensor_data)
        if config.emergency_stop_topic:
            self.create_subscription(Bool, config.emergency_stop_topic, self._on_emergency_stop, qos_profile_sensor_data)
        self._serial_client = self.create_client(
            GetParameters,
            f"{config.camera_parameter_node.rstrip('/')}/get_parameters",
        )
        if config.command_sink == "moveit_servo":
            self.sink = MoveItServoCommandSink(
                self,
                twist_topic=config.moveit_servo_twist_topic,
                status_topic=config.moveit_servo_status_topic,
                start_service=config.moveit_servo_start_service,
                stop_service=config.moveit_servo_stop_service,
            )
        else:
            self.sink = DryRunCommandSink()

    def now_seconds(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1.0e-9

    def _on_color_camera_info(self, message) -> None:
        self.color_camera_info = message

    def _on_depth_camera_info(self, message) -> None:
        self.depth_camera_info = message

    def _on_pose(self, message) -> None:
        stamp_s = ros_stamp_seconds(message.header.stamp)
        position = (
            float(message.pose.position.x),
            float(message.pose.position.y),
            float(message.pose.position.z),
        )
        quaternion = (
            float(message.pose.orientation.x),
            float(message.pose.orientation.y),
            float(message.pose.orientation.z),
            float(message.pose.orientation.w),
        )
        try:
            linear, angular = self.velocity_estimator.update(
                stamp_s=stamp_s,
                position_m=position,
                orientation_xyzw=quaternion,
            )
        except ValueError as exc:
            self._fault(f"invalid TCP pose stream: {exc}")
            return
        self.pose_stamp_s = stamp_s
        self.tcp_position_m = position
        self.tcp_linear_speed_m_s = linear
        self.tcp_angular_speed_rad_s = angular
        self.tcp_twist_command = np.asarray(
            self.velocity_estimator.linear_velocity_m_s + self.velocity_estimator.angular_velocity_rad_s,
            dtype=np.float64,
        )

    def _on_joint_state(self, message) -> None:
        names = tuple(str(name) for name in message.name)
        if self.config.expected_joint_names:
            try:
                indices = tuple(names.index(name) for name in self.config.expected_joint_names)
            except ValueError as exc:
                self._fault(f"joint state is missing a configured arm joint: {exc}")
                return
        else:
            indices = tuple(range(len(names)))
        try:
            positions = tuple(float(message.position[index]) for index in indices)
            velocities = tuple(float(message.velocity[index]) for index in indices)
        except IndexError:
            self._fault("joint state position/velocity arrays do not match their names")
            return
        stamp_s = ros_stamp_seconds(message.header.stamp)
        acceleration = np.zeros(len(velocities), dtype=np.float64)
        velocity_array = np.asarray(velocities, dtype=np.float64)
        if self._previous_joint_stamp_s is not None and self._previous_joint_velocities is not None:
            dt = stamp_s - self._previous_joint_stamp_s
            if 0.0 < dt <= 0.25:
                acceleration = (velocity_array - self._previous_joint_velocities) / dt
        self._previous_joint_stamp_s = stamp_s
        self._previous_joint_velocities = velocity_array
        self.joint_stamp_s = stamp_s
        self.joint_positions_rad = positions
        self.joint_velocities_rad_s = velocities
        self.joint_accelerations_rad_s2 = tuple(float(value) for value in acceleration)

    def _on_force(self, message) -> None:
        force = message.wrench.force
        self.force_norm_n = math.sqrt(float(force.x) ** 2 + float(force.y) ** 2 + float(force.z) ** 2)
        self.force_stamp_s = ros_stamp_seconds(message.header.stamp)

    def _on_deadman(self, message) -> None:
        self.deadman_active = bool(message.data)
        self.deadman_receipt_s = self.now_seconds()
        if self.armed and not self.deadman_active:
            self._fault("operator deadman released")

    def _on_emergency_stop(self, message) -> None:
        self.emergency_stop_active = bool(message.data)
        self.emergency_stop_receipt_s = self.now_seconds()
        if self.emergency_stop_active:
            self._fault("operator emergency stop activated")

    def _on_rgbd(self, frame: SynchronizedD405Frame) -> None:
        self.latest_frame = frame
        self.latest_frame_receipt_s = self.now_seconds()
        if not self.armed or self.terminal or self._inference_active:
            return
        frame_stamp_s = max(frame.color_stamp_s, frame.depth_stamp_s)
        if not self.policy_rate_gate.accept(frame_stamp_s):
            return
        self._inference_active = True
        try:
            self._process_frame(frame)
        except Exception as exc:
            self._fault(f"visual-servo frame failed: {exc}")
        finally:
            self._inference_active = False

    def _lookup_camera_rotation(self, *, frame: SynchronizedD405Frame):
        if frame.camera_frame_id != self.config.camera_optical_frame:
            raise ValueError(
                f"Color frame '{frame.camera_frame_id}' does not match configured optical frame "
                f"'{self.config.camera_optical_frame}'."
            )
        stamp_ns = int(round(max(frame.color_stamp_s, frame.depth_stamp_s) * 1.0e9))
        transform = self.tf_buffer.lookup_transform(
            self.config.command_frame,
            self.config.camera_optical_frame,
            Time(nanoseconds=stamp_ns),
            timeout=Duration(seconds=self.config.transform_timeout_s),
        )
        rotation = transform.transform.rotation
        matrix = _rotation_matrix_from_quaternion_xyzw((rotation.x, rotation.y, rotation.z, rotation.w))
        transform_stamp_s = ros_stamp_seconds(transform.header.stamp)
        if transform_stamp_s <= 0.0:
            transform_stamp_s = max(frame.color_stamp_s, frame.depth_stamp_s)
        return matrix, transform_stamp_s

    def _process_frame(self, frame: SynchronizedD405Frame) -> None:
        if self.pose_stamp_s is None or self.tcp_position_m is None:
            raise RuntimeError("TCP pose is unavailable.")
        now_s = self.now_seconds()
        stream_violation = self._auxiliary_stream_violation(now_s)
        if stream_violation is not None:
            raise RuntimeError(stream_violation)
        rotation_command_from_camera, transform_stamp_s = self._lookup_camera_rotation(frame=frame)
        rotation_camera_from_command = rotation_command_from_camera.T
        tcp_twist_camera = np.concatenate(
            (
                rotation_camera_from_command @ self.tcp_twist_command[:3],
                rotation_camera_from_command @ self.tcp_twist_command[3:],
            )
        )
        inference = self.runtime.infer(
            frame.rgb_uint8,
            frame.depth_z16,
            tcp_twist_camera=tcp_twist_camera,
            rotation_base_from_camera=rotation_command_from_camera,
        )
        sink_health = self.sink.health(now_s=now_s)
        decision = self.supervisor.evaluate(
            VisualServoSafetySample(
                now_s=now_s,
                color_stamp_s=frame.color_stamp_s,
                depth_stamp_s=frame.depth_stamp_s,
                pose_stamp_s=self.pose_stamp_s,
                tf_stamp_s=transform_stamp_s,
                valid_depth_fraction=inference.valid_depth_fraction,
                requested_normalized_action=inference.filtered_normalized_action,
                completion_probability=inference.completion_probability,
                tcp_position_m=self.tcp_position_m,
                tcp_linear_speed_m_s=self.tcp_linear_speed_m_s,
                tcp_angular_speed_rad_s=self.tcp_angular_speed_rad_s,
                joint_positions_rad=self.joint_positions_rad,
                joint_velocities_rad_s=self.joint_velocities_rad_s,
                joint_accelerations_rad_s2=self.joint_accelerations_rad_s2,
                force_norm_n=self.force_norm_n,
                deadman_active=self.deadman_active,
                emergency_stop_active=self.emergency_stop_active,
                command_consumer_exists=sink_health.consumer_exists,
                servo_healthy=sink_health.healthy,
                servo_status_age_s=sink_health.status_age_s,
            )
        )
        normalized = np.asarray(decision.applied_normalized_action, dtype=np.float64)
        camera_twist = normalized.copy()
        camera_twist[:3] *= self.config.linear_action_scale_m_s * self.config.first_test_speed_fraction
        camera_twist[3:] *= self.config.angular_action_scale_rad_s * self.config.first_test_speed_fraction
        command_twist = np.concatenate(
            (
                rotation_command_from_camera @ camera_twist[:3],
                rotation_command_from_camera @ camera_twist[3:],
            )
        )
        sent = self.sink.send_twist(command_twist, frame_id=self.config.command_frame, stamp_s=now_s)
        if not sent:
            self._fault("MoveIt Servo command was not accepted by a live consumer")
            return
        if self.sink.is_real and bool(np.any(np.abs(command_twist) > 0.0)):
            self.nonzero_command_sent = True
        self.runtime.commit_applied_action(decision.applied_normalized_action)
        self.step_count += 1
        self._record_step(
            {
                "step": self.step_count,
                "now_s": now_s,
                "color_stamp_s": frame.color_stamp_s,
                "depth_stamp_s": frame.depth_stamp_s,
                "image_skew_s": abs(frame.color_stamp_s - frame.depth_stamp_s),
                "valid_depth_fraction": inference.valid_depth_fraction,
                "completion_probability": inference.completion_probability,
                "completion_streak": decision.completion_streak,
                "state": decision.state.value,
                "reason": decision.reason,
                "raw_policy_action": list(inference.requested_normalized_action),
                "filtered_policy_action": list(inference.filtered_normalized_action),
                "applied_normalized_action": list(decision.applied_normalized_action),
                "camera_twist": camera_twist.tolist(),
                "command_twist": command_twist.tolist(),
                "command_sink": self.config.command_sink,
                "servo_status_code": sink_health.status_code,
                "servo_status": sink_health.status_text,
                "tcp_position_m": list(self.tcp_position_m),
                "tcp_linear_speed_m_s": self.tcp_linear_speed_m_s,
                "tcp_angular_speed_rad_s": self.tcp_angular_speed_rad_s,
                "tcp_twist_camera": tcp_twist_camera.tolist(),
                "force_norm_n": self.force_norm_n,
            }
        )
        if decision.terminal:
            self.sink.hold(frame_id=self.config.command_frame, stamp_s=now_s)
            self.runtime.reset_action_context()
            self.terminal = True

    def _record_step(self, payload: Mapping[str, object]) -> None:
        with self._steps_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(payload), sort_keys=True) + "\n")

    def _fault(self, message: str) -> None:
        if self.terminal:
            return
        self.failure_message = str(message)
        decision = self.supervisor.latch_fault(self.failure_message)
        now_s = self.now_seconds()
        try:
            self.sink.hold(frame_id=self.config.command_frame, stamp_s=now_s)
        finally:
            self.runtime.reset_action_context()
            self.terminal = True
        self._record_step(
            {
                "step": self.step_count,
                "now_s": now_s,
                "state": decision.state.value,
                "reason": decision.reason,
                "applied_normalized_action": [0.0] * 6,
            }
        )

    def basic_preflight_ready(self) -> bool:
        return (
            self.color_camera_info is not None
            and self.depth_camera_info is not None
            and self.latest_frame is not None
            and self.pose_stamp_s is not None
            and self.tcp_position_m is not None
            and (not self.config.expected_joint_names or bool(self.joint_positions_rad))
            and (not self.config.expected_joint_names or self.joint_stamp_s is not None)
            and (
                not self.config.require_force_measurement
                or (self.force_norm_n is not None and self.force_stamp_s is not None)
            )
            and (
                not self.config.require_deadman
                or (self.deadman_active and self.deadman_receipt_s is not None)
            )
            and (
                not self.config.emergency_stop_topic
                or self.emergency_stop_receipt_s is not None
            )
            and not self.emergency_stop_active
        )

    def _auxiliary_stream_violation(self, now_s: float) -> str | None:
        checks = (
            (
                bool(self.config.expected_joint_names),
                self.joint_stamp_s,
                self.config.max_joint_state_age_s,
                "joint state",
            ),
            (
                self.config.require_force_measurement,
                self.force_stamp_s,
                self.config.max_force_age_s,
                "force measurement",
            ),
            (
                self.config.require_deadman,
                self.deadman_receipt_s,
                self.config.max_operator_signal_age_s,
                "deadman signal",
            ),
            (
                bool(self.config.emergency_stop_topic),
                self.emergency_stop_receipt_s,
                self.config.max_operator_signal_age_s,
                "emergency-stop signal",
            ),
        )
        for required, stamp_s, maximum_age_s, label in checks:
            if not required:
                continue
            if stamp_s is None:
                return f"required {label} is unavailable"
            age_s = float(now_s) - float(stamp_s)
            if age_s < -0.05:
                return f"{label} timestamp is implausibly in the future"
            if age_s > float(maximum_age_s):
                return f"{label} is stale"
        return None

    def validate_camera_contract(self) -> None:
        camera = D405WristCameraConfig()
        for label, info in (
            ("color", self.color_camera_info),
            ("aligned depth", self.depth_camera_info),
        ):
            if info is None:
                raise RuntimeError(f"{label} CameraInfo is unavailable.")
            if int(info.width) != camera.width or int(info.height) != camera.height:
                raise ValueError(
                    f"{label} CameraInfo must be {camera.width}x{camera.height}, got {info.width}x{info.height}."
                )
            projection = tuple(float(value) for value in info.p)
            actual = (projection[0], projection[5], projection[2], projection[6])
            expected = (camera.fx, camera.fy, camera.cx, camera.cy)
            if any(abs(lhs - rhs) > self.config.intrinsics_tolerance_px for lhs, rhs in zip(actual, expected, strict=True)):
                raise ValueError(f"{label} rectified intrinsics {actual} do not match trained intrinsics {expected}.")
        color_projection = np.asarray(self.color_camera_info.p, dtype=np.float64)
        depth_projection = np.asarray(self.depth_camera_info.p, dtype=np.float64)
        if not np.allclose(
            color_projection,
            depth_projection,
            atol=self.config.intrinsics_tolerance_px,
            rtol=0.0,
        ):
            raise ValueError("Aligned-depth CameraInfo does not use the color projection grid.")
        for topic in (self.config.color_topic, self.config.depth_topic):
            publisher_count = len(self.get_publishers_info_by_topic(topic))
            if publisher_count != 1:
                raise RuntimeError(f"Expected exactly one publisher on '{topic}', found {publisher_count}.")

    def query_and_validate_camera_serial(self, *, timeout_s: float) -> str:
        if not self._serial_client.wait_for_service(timeout_sec=float(timeout_s)):
            raise RuntimeError(
                f"Camera parameter service for '{self.config.camera_parameter_node}' is unavailable."
            )
        request = GetParameters.Request()
        request.names = [self.config.camera_serial_parameter]
        future = self._serial_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(timeout_s))
        if not future.done() or future.exception() is not None:
            raise RuntimeError("Failed to query the connected D405 serial parameter.")
        response = future.result()
        if response is None or len(response.values) != 1:
            raise RuntimeError("Camera serial parameter response is malformed.")
        serial = str(response.values[0].string_value).strip().lstrip("_")
        if serial != self.config.expected_camera_serial:
            raise RuntimeError(
                f"Connected camera serial '{serial}' does not match required serial "
                f"'{self.config.expected_camera_serial}'."
            )
        return serial

    def watchdog(self) -> None:
        if not self.armed or self.terminal:
            return
        now_s = self.now_seconds()
        stream_violation = self._auxiliary_stream_violation(now_s)
        if stream_violation is not None:
            self._fault(stream_violation)
            return
        if self.latest_frame_receipt_s is None or now_s - self.latest_frame_receipt_s > self.config.max_image_age_s:
            self._fault("RGB-D command watchdog expired")
            return
        health = self.sink.health(now_s=now_s)
        if not health.consumer_exists or not health.healthy:
            self._fault(f"MoveIt Servo health watchdog failed: {health.status_text}")
        elif health.status_age_s is not None and health.status_age_s > self.config.max_servo_status_age_s:
            self._fault("MoveIt Servo status watchdog expired")

    def write_summary(self, result: D405VisualServoRunResult, *, camera_serial: str) -> None:
        config_payload = asdict(self.config)
        for key, value in tuple(config_payload.items()):
            if isinstance(value, Path):
                config_payload[key] = str(value)
        payload = {
            "result": {
                **asdict(result),
                "run_directory": str(result.run_directory),
            },
            "config": config_payload,
            "camera_serial": camera_serial,
            "checkpoint_sha256": self.runtime.checkpoint_sha256,
            "goal_catalog_sha256": self.runtime.catalog.sha256,
            "goal": {
                "target_id": self.runtime.goal.target_id,
                "part_id": self.runtime.goal.part_id,
                "orientation_id": self.runtime.goal.orientation_id,
                "grasp_id": self.runtime.goal.grasp_id,
                "split_id": self.runtime.goal.split_id,
            },
        }
        (self.run_directory / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_d405_policy_visual_servo(
    *,
    config_path: str | Path,
    expected_grasp_id: str,
    expected_part_id: str,
    allow_real_motion: bool,
) -> D405VisualServoRunResult:  # pragma: no cover - ROS integration path
    if rclpy is None:
        raise RuntimeError("ROS2 is unavailable. Source the ROS2 and MoveIt workspaces first.")
    config = D405VisualServoDeploymentConfig.from_yaml(config_path)
    if config.command_sink == "moveit_servo" and not allow_real_motion:
        raise RuntimeError("Real MoveIt Servo output requires explicit caller confirmation.")
    initialized_here = False
    if not rclpy.ok():
        rclpy.init()
        initialized_here = True
    node = D405VisualServoNode(
        config,
        expected_grasp_id=expected_grasp_id,
        expected_part_id=expected_part_id,
    )
    camera_serial = ""
    try:
        deadline = time.monotonic() + config.startup_timeout_s
        while time.monotonic() < deadline and not node.basic_preflight_ready() and not node.terminal:
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.terminal:
            raise RuntimeError(node.failure_message)
        if not node.basic_preflight_ready():
            raise TimeoutError("D405 visual-servo preflight did not receive all required topics.")
        node.validate_camera_contract()
        camera_serial = node.query_and_validate_camera_serial(timeout_s=config.startup_timeout_s)
        node.supervisor.mark_ready()
        node.sink.activate(timeout_s=config.startup_timeout_s)
        if node.sink.is_real:
            node.sink.wait_until_healthy(timeout_s=config.startup_timeout_s, frame_id=config.command_frame)
        node.supervisor.arm(now_s=node.now_seconds())
        node.policy_rate_gate.reset()
        node.armed = True
        while not node.terminal:
            rclpy.spin_once(node, timeout_sec=0.02)
            node.watchdog()
        completed = node.supervisor.state == VisualServoState.COMPLETED_HOLD
        message = node.supervisor.reason if completed else (node.failure_message or node.supervisor.reason)
        result = D405VisualServoRunResult(
            completed=completed,
            state=node.supervisor.state.value,
            message=message,
            target_id=config.target_id,
            motion_applied=node.nonzero_command_sent,
            allow_gripper_close=bool(
                completed
                and node.sink.is_real
                and config.allow_gripper_close_on_completion
            ),
            step_count=node.step_count,
            run_directory=node.run_directory,
        )
        node.write_summary(result, camera_serial=camera_serial)
        return result
    except Exception as exc:
        node._fault(str(exc))
        result = D405VisualServoRunResult(
            completed=False,
            state=node.supervisor.state.value,
            message=str(exc),
            target_id=config.target_id,
            motion_applied=node.nonzero_command_sent,
            allow_gripper_close=False,
            step_count=node.step_count,
            run_directory=node.run_directory,
        )
        node.write_summary(result, camera_serial=camera_serial)
        return result
    finally:
        try:
            node.sink.hold(frame_id=config.command_frame, stamp_s=node.now_seconds())
            node.sink.deactivate(timeout_s=min(config.startup_timeout_s, 5.0))
        finally:
            node.destroy_node()
            if initialized_here and rclpy.ok():
                rclpy.shutdown()


__all__ = [
    "D405VisualServoDeploymentConfig",
    "D405VisualServoRunResult",
    "D405VisualServoNode",
    "run_d405_policy_visual_servo",
]
