"""Real D405 policy loop with TF, safety supervision, and MoveIt Servo output."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    D405WristCameraConfig,
    camera_mount_profile_from_camera_profile,
    camera_rotation_in_link7,
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
    from geometry_msgs.msg import WrenchStamped
    from rcl_interfaces.srv import GetParameters
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.duration import Duration
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, JointState
    from std_msgs.msg import Bool
    from tf2_ros import Buffer, TransformException, TransformListener
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    WrenchStamped = None
    GetParameters = None
    MutuallyExclusiveCallbackGroup = None
    Duration = None
    MultiThreadedExecutor = None
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
class _RobotFeedbackSnapshot:
    pose_stamp_s: float | None
    tcp_position_m: tuple[float, float, float] | None
    tcp_linear_speed_m_s: float
    tcp_angular_speed_rad_s: float
    tcp_twist_command: tuple[float, float, float, float, float, float]
    joint_positions_rad: tuple[float, ...]
    joint_velocities_rad_s: tuple[float, ...]
    joint_accelerations_rad_s2: tuple[float, ...]
    joint_stamp_s: float | None
    force_norm_n: float | None
    force_stamp_s: float | None
    deadman_active: bool
    deadman_receipt_s: float | None
    emergency_stop_active: bool
    emergency_stop_receipt_s: float | None


def rgbd_watchdog_violation(
    *,
    now_s: float,
    latest_frame: SynchronizedD405Frame | None,
    latest_receipt_s: float | None,
    maximum_age_s: float,
    enforce_source_image_age: bool = True,
) -> str | None:
    if latest_frame is None or latest_receipt_s is None:
        return "RGB-D command watchdog has no frame"
    source_stamp_s = min(float(latest_frame.color_stamp_s), float(latest_frame.depth_stamp_s))
    source_age_s = float(now_s) - source_stamp_s
    receipt_age_s = float(now_s) - float(latest_receipt_s)
    if source_age_s < -0.05:
        return "RGB-D source timestamp is implausibly in the future"
    if enforce_source_image_age and source_age_s > float(maximum_age_s):
        return "RGB-D source frame is stale"
    if receipt_age_s > float(maximum_age_s):
        return "RGB-D command watchdog expired"
    return None


@dataclass(frozen=True)
class D405VisualServoDeploymentConfig:
    checkpoint_path: Path
    checkpoint_metadata_path: Path
    agent_config_path: Path
    goal_observation_path: Path
    goal_renderer_launcher: Path
    goal_renderer_python_command: str
    goal_renderer_script: Path
    goal_renderer_robot_urdf: Path
    goal_renderer_backend: str = "filament"
    goal_renderer_timeout_s: float = 240.0
    model_device: str = "cuda:0"
    expected_camera_serial: str = EXPECTED_D405_SERIAL
    expected_camera_profile: str = D405_VISUAL_SERVO_CAMERA_PROFILE
    expected_observation_profile: str = D405_VISUAL_SERVO_OBSERVATION_PROFILE
    image_transport: str = "raw"
    color_topic: str = "/realsense_1/camera/color/image_rect"
    depth_topic: str = "/realsense_1/camera/aligned_depth_to_color/image_rect"
    color_camera_info_topic: str = "/realsense_1/camera/color/camera_info"
    depth_camera_info_topic: str = "/realsense_1/camera/aligned_depth_to_color/camera_info"
    joint_state_topic: str = "/lbr/joint_states"
    force_topic: str = ""
    deadman_topic: str = "/d405_visual_servo/deadman"
    emergency_stop_topic: str = "/d405_visual_servo/emergency_stop"
    camera_parameter_node: str = "/realsense_1/camera"
    camera_serial_parameter: str = "serial_no"
    camera_optical_frame: str = "realsense_1_color_optical_frame"
    allow_camera_topic_frame_alias: bool = False
    allow_pdz_camera_rotation_fallback: bool = False
    command_frame: str = "lbr_link_0"
    tcp_frame: str = "pdz_gripper_tcp"
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
    action_delta_limit: float = 0.25
    policy_rate_hz: float = POLICY_RATE_HZ
    startup_timeout_s: float = 30.0
    transform_timeout_s: float = 0.05
    intrinsics_tolerance_px: float = 3.0
    expected_joint_names: tuple[str, ...] = ()
    max_image_age_s: float = 0.15
    enforce_source_image_age: bool = True
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
            goal_observation_path=_resolve_config_path(
                payload.get("goal_observation_path"),
                base_dir=base_dir,
            ),
            goal_renderer_launcher=_resolve_config_path(
                payload.get(
                    "goal_renderer_launcher",
                    "../scripts/run_mujoco_filament.sh",
                ),
                base_dir=base_dir,
            ),
            goal_renderer_python_command=str(
                payload.get("goal_renderer_python_command", "python3")
            ).strip(),
            goal_renderer_script=_resolve_config_path(
                payload.get("goal_renderer_script", "../scripts/render_d405_policy_goal.py"),
                base_dir=base_dir,
            ),
            goal_renderer_robot_urdf=_resolve_config_path(
                payload.get(
                    "goal_renderer_robot_urdf",
                    "../assets/urdf/kuka_iiwa7_pdz_gripper/urdf/kuka_iiwa7_pdz_gripper.urdf",
                ),
                base_dir=base_dir,
            ),
            goal_renderer_backend=str(
                payload.get("goal_renderer_backend", "filament")
            ).strip().lower(),
            goal_renderer_timeout_s=float(payload.get("goal_renderer_timeout_s", 240.0)),
            model_device=str(payload.get("model_device", "cuda:0")),
            expected_camera_serial=str(payload.get("expected_camera_serial", EXPECTED_D405_SERIAL)),
            expected_camera_profile=str(
                payload.get("expected_camera_profile", D405_VISUAL_SERVO_CAMERA_PROFILE)
            ),
            expected_observation_profile=str(
                payload.get("expected_observation_profile", D405_VISUAL_SERVO_OBSERVATION_PROFILE)
            ),
            image_transport=str(payload.get("image_transport", "raw")).strip().lower(),
            color_topic=str(payload.get("color_topic", cls.color_topic)),
            depth_topic=str(payload.get("depth_topic", cls.depth_topic)),
            color_camera_info_topic=str(
                payload.get("color_camera_info_topic", cls.color_camera_info_topic)
            ),
            depth_camera_info_topic=str(
                payload.get("depth_camera_info_topic", cls.depth_camera_info_topic)
            ),
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
            allow_camera_topic_frame_alias=bool(
                payload.get("allow_camera_topic_frame_alias", False)
            ),
            allow_pdz_camera_rotation_fallback=bool(
                payload.get("allow_pdz_camera_rotation_fallback", False)
            ),
            command_frame=str(payload.get("command_frame", cls.command_frame)),
            tcp_frame=str(payload.get("tcp_frame", cls.tcp_frame)),
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
            action_delta_limit=float(payload.get("action_delta_limit", 0.25)),
            policy_rate_hz=float(payload.get("policy_rate_hz", POLICY_RATE_HZ)),
            startup_timeout_s=float(payload.get("startup_timeout_s", 30.0)),
            transform_timeout_s=float(payload.get("transform_timeout_s", 0.05)),
            intrinsics_tolerance_px=float(payload.get("intrinsics_tolerance_px", 3.0)),
            expected_joint_names=tuple(str(value) for value in payload.get("expected_joint_names", ())),
            max_image_age_s=float(payload.get("max_image_age_s", 0.15)),
            enforce_source_image_age=bool(payload.get("enforce_source_image_age", True)),
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
        if self.image_transport not in {"raw", "compressed"}:
            raise ValueError("image_transport must be 'raw' or 'compressed'.")
        for field_name in (
            "color_topic",
            "depth_topic",
            "color_camera_info_topic",
            "depth_camera_info_topic",
            "camera_optical_frame",
            "command_frame",
            "tcp_frame",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty.")
        if not 0.0 < self.first_test_speed_fraction <= 1.0:
            raise ValueError("first_test_speed_fraction must lie in (0, 1].")
        if not math.isfinite(self.policy_rate_hz) or self.policy_rate_hz <= 0.0:
            raise ValueError("policy_rate_hz must be finite and positive.")
        for path_name in (
            "checkpoint_path",
            "checkpoint_metadata_path",
            "agent_config_path",
        ):
            path = getattr(self, path_name)
            if not path.is_file():
                raise FileNotFoundError(f"{path_name} does not exist: {path}")
        if str(self.goal_observation_path) not in {"", "."} and not self.goal_observation_path.is_file():
            raise FileNotFoundError(
                f"goal_observation_path does not exist: {self.goal_observation_path}"
            )
        if not self.goal_renderer_python_command:
            raise ValueError("goal_renderer_python_command must be non-empty.")
        if self.goal_renderer_backend != "filament":
            raise ValueError("goal_renderer_backend must be 'filament'.")
        if self.goal_renderer_timeout_s <= 0.0:
            raise ValueError("Goal renderer timeout must be positive.")
        if self.command_sink == "moveit_servo":
            if not self.real_motion_approved:
                raise ValueError("MoveIt Servo output requires real_motion_approved: true.")
            if not self.expected_joint_names or not self.joint_position_limits_rad:
                raise ValueError("Real motion requires explicit joint names and position limits.")
            if len(self.expected_joint_names) != len(self.joint_position_limits_rad):
                raise ValueError("Joint names and joint-position limits must have equal length.")
            if not self.require_force_measurement or not self.force_topic:
                raise ValueError("Real motion requires force supervision and a non-empty force_topic.")
            if self.require_deadman and not self.deadman_topic:
                raise ValueError("require_deadman requires a non-empty deadman_topic.")
        for name in ("max_joint_state_age_s", "max_force_age_s", "max_operator_signal_age_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        self.safety_config().validate()

    def safety_config(self) -> VisualServoSafetyConfig:
        return VisualServoSafetyConfig(
            max_image_age_s=self.max_image_age_s,
            enforce_source_image_age=self.enforce_source_image_age,
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
    goal_id: str
    motion_applied: bool
    allow_gripper_close: bool
    step_count: int
    run_directory: Path


@dataclass(frozen=True)
class D405VisualServoPreparation:
    config: D405VisualServoDeploymentConfig
    runtime: D405PolicyRuntime
    expected_grasp_id: str
    expected_part_id: str


def prepare_d405_policy_visual_servo(
    *,
    config_path: str | Path,
    expected_grasp_id: str,
    expected_part_id: str,
    goal_observation_path_override: str | Path = "",
) -> D405VisualServoPreparation:
    """Strict-load policy assets before any gripper or arm motion begins."""

    config = D405VisualServoDeploymentConfig.from_yaml(config_path)
    if str(goal_observation_path_override).strip():
        config = replace(
            config,
            goal_observation_path=Path(goal_observation_path_override).expanduser().resolve(),
        )
    if not config.goal_observation_path.is_file():
        raise FileNotFoundError(
            "A runtime-rendered goal observation is required before policy loading: "
            f"{config.goal_observation_path}"
        )
    runtime = D405PolicyRuntime(
        checkpoint_path=config.checkpoint_path,
        checkpoint_metadata_path=config.checkpoint_metadata_path,
        agent_config_path=config.agent_config_path,
        goal_observation_path=config.goal_observation_path,
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
    return D405VisualServoPreparation(
        config=config,
        runtime=runtime,
        expected_grasp_id=str(expected_grasp_id),
        expected_part_id=str(expected_part_id),
    )


def preflight_d405_policy_visual_servo(
    preparation: D405VisualServoPreparation,
) -> str:  # pragma: no cover - ROS integration path
    """Verify every live policy input and Servo endpoint before robot motion."""

    if rclpy is None:
        raise RuntimeError("ROS2 is unavailable. Source the ROS2 and MoveIt workspaces first.")
    config = preparation.config
    initialized_here = False
    if not rclpy.ok():
        rclpy.init()
        initialized_here = True
    node = D405VisualServoNode(
        config,
        expected_grasp_id=preparation.expected_grasp_id,
        expected_part_id=preparation.expected_part_id,
        prepared_runtime=preparation.runtime,
    )
    try:
        deadline = time.monotonic() + config.startup_timeout_s
        while time.monotonic() < deadline and not node.basic_preflight_ready() and not node.terminal:
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.terminal:
            raise RuntimeError(node.failure_message)
        if not node.basic_preflight_ready():
            missing = "; ".join(node.basic_preflight_missing_inputs())
            raise TimeoutError(f"D405 visual-servo preflight missing: {missing}.")
        node.validate_camera_contract()
        serial = node.query_and_validate_camera_serial(timeout_s=config.startup_timeout_s)
        assert node.latest_frame is not None
        node._lookup_camera_rotation(frame=node.latest_frame)
        node.sink.preflight(timeout_s=config.startup_timeout_s)
        return serial
    finally:
        node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()


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
        prepared_runtime: D405PolicyRuntime | None = None,
    ) -> None:
        if rclpy is None or CameraInfo is None or Buffer is None:
            raise RuntimeError("ROS2 image/TF dependencies are unavailable. Source the ROS2 workspace first.")
        super().__init__("d405_ppo_visual_servo")
        self.config = config
        self.runtime = prepared_runtime or D405PolicyRuntime(
            checkpoint_path=config.checkpoint_path,
            checkpoint_metadata_path=config.checkpoint_metadata_path,
            agent_config_path=config.agent_config_path,
            goal_observation_path=config.goal_observation_path,
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
        # Inference and command/watchdog decisions must remain serialized, but
        # they must not block high-rate joint/force/TF/Servo feedback callbacks.
        # The latter stay in the node's distinct default callback group.
        self._control_callback_group = MutuallyExclusiveCallbackGroup()
        self._feedback_lock = threading.Lock()
        self.velocity_estimator = PoseVelocityEstimator()
        self.policy_rate_gate = PolicyRateGate(config.policy_rate_hz)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.color_camera_info = None
        self.depth_camera_info = None
        self.latest_frame: SynchronizedD405Frame | None = None
        self.latest_frame_receipt_s: float | None = None
        self._last_stale_source_warning_s: float | None = None
        self._camera_rotation_fallback_warned = False
        self.camera_rotation_source = "unresolved"
        self.pose_stamp_s: float | None = None
        self.tcp_position_m: tuple[float, float, float] | None = None
        self.tcp_linear_speed_m_s = 0.0
        self.tcp_angular_speed_rad_s = 0.0
        self.tcp_twist_command = (0.0,) * 6
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
        self._last_command_twist = (0.0,) * 6
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.run_directory = config.output_root / f"{timestamp}-{self.runtime.goal.grasp_id}"
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self._steps_path = self.run_directory / "steps.jsonl"

        self.rgbd_subscriber = D405RgbdSubscriber(
            self,
            color_topic=config.color_topic,
            depth_topic=config.depth_topic,
            image_transport=config.image_transport,
            maximum_skew_s=config.max_image_skew_s,
            callback=self._on_rgbd,
            callback_group=self._control_callback_group,
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
        self.create_subscription(JointState, config.joint_state_topic, self._on_joint_state, qos_profile_sensor_data)
        if config.force_topic:
            self.create_subscription(WrenchStamped, config.force_topic, self._on_force, qos_profile_sensor_data)
        if config.deadman_topic:
            self.create_subscription(Bool, config.deadman_topic, self._on_deadman, qos_profile_sensor_data)
        if config.emergency_stop_topic:
            self.create_subscription(Bool, config.emergency_stop_topic, self._on_emergency_stop, qos_profile_sensor_data)
        self._tcp_pose_timer = self.create_timer(0.01, self._refresh_tcp_pose_from_tf)
        self._watchdog_timer = self.create_timer(
            0.02,
            self.watchdog,
            callback_group=self._control_callback_group,
        )
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
        if not config.enforce_source_image_age:
            self.get_logger().warning(
                "TEST MODE: RGB-D publisher-header age is warning-only; the local frame-receipt "
                "watchdog, RGB/depth skew check, and all other motion safety checks remain active."
            )
        if config.allow_pdz_camera_rotation_fallback:
            self.get_logger().warning(
                "TEST MODE: if the live optical TF tree is disconnected, camera-frame command "
                "rotation will fall back to the audited PDZ CAD mount composed with current "
                "lbr_link_7. The MoveIt, renderer, and policy assets use the same PDZ embodiment."
            )

    def now_seconds(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1.0e-9

    def _robot_feedback_snapshot(self) -> _RobotFeedbackSnapshot:
        with self._feedback_lock:
            return _RobotFeedbackSnapshot(
                pose_stamp_s=self.pose_stamp_s,
                tcp_position_m=self.tcp_position_m,
                tcp_linear_speed_m_s=self.tcp_linear_speed_m_s,
                tcp_angular_speed_rad_s=self.tcp_angular_speed_rad_s,
                tcp_twist_command=self.tcp_twist_command,
                joint_positions_rad=self.joint_positions_rad,
                joint_velocities_rad_s=self.joint_velocities_rad_s,
                joint_accelerations_rad_s2=self.joint_accelerations_rad_s2,
                joint_stamp_s=self.joint_stamp_s,
                force_norm_n=self.force_norm_n,
                force_stamp_s=self.force_stamp_s,
                deadman_active=self.deadman_active,
                deadman_receipt_s=self.deadman_receipt_s,
                emergency_stop_active=self.emergency_stop_active,
                emergency_stop_receipt_s=self.emergency_stop_receipt_s,
            )

    def _on_color_camera_info(self, message) -> None:
        self.color_camera_info = message

    def _on_depth_camera_info(self, message) -> None:
        self.depth_camera_info = message

    def _refresh_tcp_pose_from_tf(self) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.config.command_frame,
                self.config.tcp_frame,
                Time(),
            )
        except TransformException:
            return
        stamp_s = ros_stamp_seconds(transform.header.stamp)
        if stamp_s <= 0.0:
            return
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        position = (
            float(translation.x),
            float(translation.y),
            float(translation.z),
        )
        quaternion = (
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        )
        error: ValueError | None = None
        with self._feedback_lock:
            if self.pose_stamp_s is not None and stamp_s <= self.pose_stamp_s:
                return
            try:
                linear, angular = self.velocity_estimator.update(
                    stamp_s=stamp_s,
                    position_m=position,
                    orientation_xyzw=quaternion,
                )
            except ValueError as exc:
                error = exc
            else:
                self.pose_stamp_s = stamp_s
                self.tcp_position_m = position
                self.tcp_linear_speed_m_s = linear
                self.tcp_angular_speed_rad_s = angular
                self.tcp_twist_command = (
                    *self.velocity_estimator.linear_velocity_m_s,
                    *self.velocity_estimator.angular_velocity_rad_s,
                )
        if error is not None:
            self._fault(f"invalid TCP TF stream: {error}")

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
        velocity_array = np.asarray(velocities, dtype=np.float64)
        with self._feedback_lock:
            acceleration = np.zeros(len(velocities), dtype=np.float64)
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
        force_norm_n = math.sqrt(float(force.x) ** 2 + float(force.y) ** 2 + float(force.z) ** 2)
        force_stamp_s = ros_stamp_seconds(message.header.stamp)
        with self._feedback_lock:
            self.force_norm_n = force_norm_n
            self.force_stamp_s = force_stamp_s

    def _on_deadman(self, message) -> None:
        deadman_active = bool(message.data)
        with self._feedback_lock:
            self.deadman_active = deadman_active
            self.deadman_receipt_s = self.now_seconds()
        if self.armed and not deadman_active:
            self._fault("operator deadman released")

    def _on_emergency_stop(self, message) -> None:
        emergency_stop_active = bool(message.data)
        with self._feedback_lock:
            self.emergency_stop_active = emergency_stop_active
            self.emergency_stop_receipt_s = self.now_seconds()
        if emergency_stop_active:
            self._fault("operator emergency stop activated")

    def _on_rgbd(self, frame: SynchronizedD405Frame) -> None:
        receipt_s = self.now_seconds()
        self.latest_frame = frame
        self.latest_frame_receipt_s = receipt_s
        source_age_s = receipt_s - min(frame.color_stamp_s, frame.depth_stamp_s)
        if (
            not self.config.enforce_source_image_age
            and source_age_s > self.config.max_image_age_s
            and (
                self._last_stale_source_warning_s is None
                or receipt_s - self._last_stale_source_warning_s >= 2.0
            )
        ):
            self.get_logger().warning(
                "TEST MODE: accepting RGB-D frame with publisher-header age "
                f"{source_age_s * 1000.0:.1f} ms; configured limit is "
                f"{self.config.max_image_age_s * 1000.0:.1f} ms."
            )
            self._last_stale_source_warning_s = receipt_s
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
        if (
            frame.camera_frame_id != self.config.camera_optical_frame
            and not self.config.allow_camera_topic_frame_alias
        ):
            raise ValueError(
                f"Color frame '{frame.camera_frame_id}' does not match configured optical frame "
                f"'{self.config.camera_optical_frame}'."
            )
        stamp_ns = int(round(max(frame.color_stamp_s, frame.depth_stamp_s) * 1.0e9))
        try:
            transform = self.tf_buffer.lookup_transform(
                self.config.command_frame,
                self.config.camera_optical_frame,
                Time(nanoseconds=stamp_ns),
                timeout=Duration(seconds=self.config.transform_timeout_s),
            )
        except TransformException as direct_error:
            if not self.config.allow_pdz_camera_rotation_fallback:
                raise
            try:
                link7_transform = self.tf_buffer.lookup_transform(
                    self.config.command_frame,
                    "lbr_link_7",
                    Time(),
                    timeout=Duration(seconds=self.config.transform_timeout_s),
                )
            except TransformException as fallback_error:
                raise RuntimeError(
                    "Live camera TF is disconnected and the PDZ link7 fallback also failed: "
                    f"live={direct_error}; fallback={fallback_error}"
                ) from fallback_error
            link7_rotation = link7_transform.transform.rotation
            rotation_command_from_link7 = _rotation_matrix_from_quaternion_xyzw(
                (
                    link7_rotation.x,
                    link7_rotation.y,
                    link7_rotation.z,
                    link7_rotation.w,
                )
            )
            camera_cfg = D405WristCameraConfig(
                mount_profile=camera_mount_profile_from_camera_profile(
                    self.config.expected_camera_profile
                )
            )
            rotation_command_from_camera = (
                rotation_command_from_link7 @ camera_rotation_in_link7(camera_cfg)
            )
            transform_stamp_s = ros_stamp_seconds(link7_transform.header.stamp)
            if transform_stamp_s <= 0.0:
                transform_stamp_s = self.now_seconds()
            self.camera_rotation_source = "pdz_cad_link7_fallback"
            if not self._camera_rotation_fallback_warned:
                self.get_logger().warning(
                    "TEST MODE: live camera TF is disconnected; using current "
                    "lbr_link_0<-lbr_link_7 with the audited PDZ CAD optical rotation. "
                    f"Original TF error: {direct_error}"
                )
                self._camera_rotation_fallback_warned = True
            return rotation_command_from_camera, transform_stamp_s
        rotation = transform.transform.rotation
        matrix = _rotation_matrix_from_quaternion_xyzw((rotation.x, rotation.y, rotation.z, rotation.w))
        transform_stamp_s = ros_stamp_seconds(transform.header.stamp)
        if transform_stamp_s <= 0.0:
            transform_stamp_s = max(frame.color_stamp_s, frame.depth_stamp_s)
        self.camera_rotation_source = "live_tf"
        return matrix, transform_stamp_s

    def _process_frame(self, frame: SynchronizedD405Frame) -> None:
        feedback = self._robot_feedback_snapshot()
        if feedback.pose_stamp_s is None or feedback.tcp_position_m is None:
            raise RuntimeError("TCP pose is unavailable.")
        pre_inference_now_s = self.now_seconds()
        stream_violation = self._auxiliary_stream_violation(
            pre_inference_now_s,
            feedback=feedback,
        )
        if stream_violation is not None:
            raise RuntimeError(stream_violation)
        control_preparation_started_s = time.perf_counter()
        rotation_command_from_camera, transform_stamp_s = self._lookup_camera_rotation(frame=frame)
        rotation_camera_from_command = rotation_command_from_camera.T
        tcp_twist_command = np.asarray(feedback.tcp_twist_command, dtype=np.float64)
        tcp_twist_camera = np.concatenate(
            (
                rotation_camera_from_command @ tcp_twist_command[:3],
                rotation_camera_from_command @ tcp_twist_command[3:],
            )
        )
        inference_started_s = time.perf_counter()
        inference = self.runtime.infer(
            frame.rgb_uint8,
            frame.depth_z16,
            tcp_twist_camera=tcp_twist_camera,
            rotation_base_from_camera=rotation_command_from_camera,
        )
        inference_duration_s = time.perf_counter() - inference_started_s
        # Feedback callbacks continue to run during CUDA inference. Refresh the
        # ROS clock and atomically snapshot their latest state only after the
        # blocking inference/TF work; otherwise fresh feedback appears to be in
        # the future relative to a pre-inference timestamp.
        now_s = self.now_seconds()
        feedback = self._robot_feedback_snapshot()
        if feedback.pose_stamp_s is None or feedback.tcp_position_m is None:
            raise RuntimeError("TCP pose became unavailable after policy inference.")
        stream_violation = self._auxiliary_stream_violation(now_s, feedback=feedback)
        if stream_violation is not None:
            raise RuntimeError(stream_violation)
        sink_health = self.sink.health(now_s=now_s)
        control_preparation_duration_s = time.perf_counter() - control_preparation_started_s
        decision = self.supervisor.evaluate(
            VisualServoSafetySample(
                now_s=now_s,
                color_stamp_s=frame.color_stamp_s,
                depth_stamp_s=frame.depth_stamp_s,
                pose_stamp_s=feedback.pose_stamp_s,
                tf_stamp_s=transform_stamp_s,
                valid_depth_fraction=inference.valid_depth_fraction,
                requested_normalized_action=inference.filtered_normalized_action,
                completion_probability=inference.completion_probability,
                tcp_position_m=feedback.tcp_position_m,
                tcp_linear_speed_m_s=feedback.tcp_linear_speed_m_s,
                tcp_angular_speed_rad_s=feedback.tcp_angular_speed_rad_s,
                joint_positions_rad=feedback.joint_positions_rad,
                joint_velocities_rad_s=feedback.joint_velocities_rad_s,
                joint_accelerations_rad_s2=feedback.joint_accelerations_rad_s2,
                force_norm_n=feedback.force_norm_n,
                deadman_active=feedback.deadman_active,
                emergency_stop_active=feedback.emergency_stop_active,
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
        self._last_command_twist = tuple(float(value) for value in command_twist)
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
                "image_source_age_s": now_s - min(frame.color_stamp_s, frame.depth_stamp_s),
                "image_receipt_age_s": (
                    None
                    if self.latest_frame_receipt_s is None
                    else now_s - self.latest_frame_receipt_s
                ),
                "image_skew_s": abs(frame.color_stamp_s - frame.depth_stamp_s),
                "joint_state_age_s": (
                    None if feedback.joint_stamp_s is None else now_s - feedback.joint_stamp_s
                ),
                "force_age_s": (
                    None if feedback.force_stamp_s is None else now_s - feedback.force_stamp_s
                ),
                "tcp_pose_age_s": now_s - feedback.pose_stamp_s,
                "camera_transform_age_s": now_s - transform_stamp_s,
                "policy_inference_duration_s": inference_duration_s,
                "control_preparation_duration_s": control_preparation_duration_s,
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
                "camera_rotation_source": self.camera_rotation_source,
                "servo_status_code": sink_health.status_code,
                "servo_status": sink_health.status_text,
                "servo_status_age_s": sink_health.status_age_s,
                "tcp_position_m": list(feedback.tcp_position_m),
                "tcp_linear_speed_m_s": feedback.tcp_linear_speed_m_s,
                "tcp_angular_speed_rad_s": feedback.tcp_angular_speed_rad_s,
                "tcp_twist_camera": tcp_twist_camera.tolist(),
                "policy_context_mode": self.runtime.policy_context_mode,
                "force_norm_n": feedback.force_norm_n,
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
        feedback = self._robot_feedback_snapshot()
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
                "image_source_age_s": (
                    None
                    if self.latest_frame is None
                    else now_s
                    - min(self.latest_frame.color_stamp_s, self.latest_frame.depth_stamp_s)
                ),
                "image_receipt_age_s": (
                    None
                    if self.latest_frame_receipt_s is None
                    else now_s - self.latest_frame_receipt_s
                ),
                "joint_state_age_s": (
                    None if feedback.joint_stamp_s is None else now_s - feedback.joint_stamp_s
                ),
                "force_age_s": (
                    None if feedback.force_stamp_s is None else now_s - feedback.force_stamp_s
                ),
                "tcp_pose_age_s": (
                    None if feedback.pose_stamp_s is None else now_s - feedback.pose_stamp_s
                ),
            }
        )

    def basic_preflight_ready(self) -> bool:
        return not self.basic_preflight_missing_inputs()

    def basic_preflight_missing_inputs(self) -> tuple[str, ...]:
        missing: list[str] = []
        feedback = self._robot_feedback_snapshot()
        if self.color_camera_info is None:
            missing.append(f"color CameraInfo ({self.config.color_camera_info_topic})")
        if self.depth_camera_info is None:
            missing.append(f"depth CameraInfo ({self.config.depth_camera_info_topic})")
        if self.latest_frame is None:
            missing.append(
                f"synchronized {self.config.image_transport} RGB-D "
                f"({self.config.color_topic}, {self.config.depth_topic})"
            )
        if feedback.pose_stamp_s is None or feedback.tcp_position_m is None:
            missing.append(f"TCP TF ({self.config.command_frame} <- {self.config.tcp_frame})")
        if self.config.expected_joint_names and (
            not feedback.joint_positions_rad or feedback.joint_stamp_s is None
        ):
            missing.append(f"joint state ({self.config.joint_state_topic})")
        if self.config.require_force_measurement and (
            feedback.force_norm_n is None or feedback.force_stamp_s is None
        ):
            missing.append(f"force measurement ({self.config.force_topic})")
        if self.config.require_deadman and (
            not feedback.deadman_active or feedback.deadman_receipt_s is None
        ):
            missing.append(f"active deadman ({self.config.deadman_topic})")
        if self.config.emergency_stop_topic and feedback.emergency_stop_receipt_s is None:
            missing.append(f"emergency-stop heartbeat ({self.config.emergency_stop_topic})")
        if feedback.emergency_stop_active:
            missing.append("emergency stop is active")
        return tuple(missing)

    def _auxiliary_stream_violation(
        self,
        now_s: float,
        *,
        feedback: _RobotFeedbackSnapshot | None = None,
    ) -> str | None:
        feedback = feedback or self._robot_feedback_snapshot()
        checks = (
            (
                bool(self.config.expected_joint_names),
                feedback.joint_stamp_s,
                self.config.max_joint_state_age_s,
                "joint state",
            ),
            (
                self.config.require_force_measurement,
                feedback.force_stamp_s,
                self.config.max_force_age_s,
                "force measurement",
            ),
            (
                self.config.require_deadman,
                feedback.deadman_receipt_s,
                self.config.max_operator_signal_age_s,
                "deadman signal",
            ),
            (
                bool(self.config.emergency_stop_topic),
                feedback.emergency_stop_receipt_s,
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
                delta = tuple(lhs - rhs for lhs, rhs in zip(actual, expected, strict=True))
                self.get_logger().warning(
                    f"{label} rectified intrinsics {actual} differ from trained intrinsics "
                    f"{expected} by {delta} px; continuing with the live CameraInfo."
                )
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
            self.get_logger().warning(
                f"Camera parameter service for '{self.config.camera_parameter_node}' is unavailable; "
                "continuing with the configured RGB-D topics, CameraInfo, and TF chain."
            )
            return ""
        request = GetParameters.Request()
        request.names = [self.config.camera_serial_parameter]
        future = self._serial_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(timeout_s))
        if not future.done() or future.exception() is not None:
            self.get_logger().warning(
                "Failed to query the connected D405 serial parameter; continuing with "
                "the configured RGB-D topics, CameraInfo, and TF chain."
            )
            return ""
        response = future.result()
        if response is None or len(response.values) != 1:
            self.get_logger().warning(
                "Camera serial parameter response is malformed; continuing with the "
                "configured RGB-D topics, CameraInfo, and TF chain."
            )
            return ""
        serial = str(response.values[0].string_value).strip().lstrip("_")
        if self.config.expected_camera_serial and serial != self.config.expected_camera_serial:
            self.get_logger().warning(
                f"Connected camera serial '{serial}' differs from configured serial "
                f"'{self.config.expected_camera_serial}'; continuing because camera routing "
                "is determined by topics, CameraInfo, and TF."
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
        rgbd_violation = rgbd_watchdog_violation(
            now_s=now_s,
            latest_frame=self.latest_frame,
            latest_receipt_s=self.latest_frame_receipt_s,
            maximum_age_s=self.config.max_image_age_s,
            enforce_source_image_age=self.config.enforce_source_image_age,
        )
        if rgbd_violation is not None:
            self._fault(rgbd_violation)
            return
        health = self.sink.health(now_s=now_s)
        if not health.consumer_exists or not health.healthy:
            self._fault(f"MoveIt Servo health watchdog failed: {health.status_text}")
        elif health.status_age_s is not None and health.status_age_s > self.config.max_servo_status_age_s:
            self._fault("MoveIt Servo status watchdog expired")
        elif not self.sink.send_twist(
            self._last_command_twist,
            frame_id=self.config.command_frame,
            stamp_s=now_s,
        ):
            self._fault("MoveIt Servo command refresh was not accepted by a live consumer")

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
            "policy_context_mode": self.runtime.policy_context_mode,
            "policy_context_size": self.runtime.policy_context_size,
            "network_input_size": self.runtime.network_input_size,
            "goal_observation_sha256": self.runtime.goal_observation.sha256,
            "goal": {
                "goal_id": self.runtime.goal.goal_id,
                "part_id": self.runtime.goal.part_id,
                "grasp_id": self.runtime.goal.grasp_id,
            },
        }
        (self.run_directory / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_d405_policy_visual_servo(
    *,
    config_path: str | Path,
    expected_grasp_id: str,
    expected_part_id: str,
    allow_real_motion: bool,
    preparation: D405VisualServoPreparation | None = None,
) -> D405VisualServoRunResult:  # pragma: no cover - ROS integration path
    if rclpy is None:
        raise RuntimeError("ROS2 is unavailable. Source the ROS2 and MoveIt workspaces first.")
    if preparation is None:
        preparation = prepare_d405_policy_visual_servo(
            config_path=config_path,
            expected_grasp_id=expected_grasp_id,
            expected_part_id=expected_part_id,
        )
    if preparation.expected_grasp_id != str(expected_grasp_id) or preparation.expected_part_id != str(
        expected_part_id
    ):
        raise ValueError("Prepared D405 policy identity does not match the requested stage-2 grasp.")
    config = preparation.config
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
        prepared_runtime=preparation.runtime,
    )
    camera_serial = ""
    executor = None
    try:
        deadline = time.monotonic() + config.startup_timeout_s
        while time.monotonic() < deadline and not node.basic_preflight_ready() and not node.terminal:
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.terminal:
            raise RuntimeError(node.failure_message)
        if not node.basic_preflight_ready():
            missing = "; ".join(node.basic_preflight_missing_inputs())
            raise TimeoutError(f"D405 visual-servo preflight missing: {missing}.")
        node.validate_camera_contract()
        camera_serial = node.query_and_validate_camera_serial(timeout_s=config.startup_timeout_s)
        node.supervisor.mark_ready()
        node.sink.activate(timeout_s=config.startup_timeout_s)
        if node.sink.is_real:
            node.sink.wait_until_healthy(timeout_s=config.startup_timeout_s, frame_id=config.command_frame)
        node.supervisor.arm(now_s=node.now_seconds())
        node.policy_rate_gate.reset()
        node.armed = True
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        while not node.terminal:
            executor.spin_once(timeout_sec=0.05)
        completed = node.supervisor.state == VisualServoState.COMPLETED_HOLD
        message = node.supervisor.reason if completed else (node.failure_message or node.supervisor.reason)
        result = D405VisualServoRunResult(
            completed=completed,
            state=node.supervisor.state.value,
            message=message,
            goal_id=node.runtime.goal.goal_id,
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
            goal_id=node.runtime.goal.goal_id,
            motion_applied=node.nonzero_command_sent,
            allow_gripper_close=False,
            step_count=node.step_count,
            run_directory=node.run_directory,
        )
        node.write_summary(result, camera_serial=camera_serial)
        return result
    finally:
        if executor is not None:
            executor.remove_node(node)
            executor.shutdown(timeout_sec=1.0)
        try:
            node.sink.hold(frame_id=config.command_frame, stamp_s=node.now_seconds())
            node.sink.deactivate(timeout_s=min(config.startup_timeout_s, 5.0))
        finally:
            node.destroy_node()
            if initialized_here and rclpy.ok():
                rclpy.shutdown()


__all__ = [
    "D405VisualServoDeploymentConfig",
    "D405VisualServoPreparation",
    "D405VisualServoRunResult",
    "D405VisualServoNode",
    "prepare_d405_policy_visual_servo",
    "preflight_d405_policy_visual_servo",
    "rgbd_watchdog_violation",
    "run_d405_policy_visual_servo",
]
