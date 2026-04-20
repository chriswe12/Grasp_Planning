"""Headless MoveIt2 joint-space planning helpers for FR3."""

from __future__ import annotations

import copy
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetMotionPlan
from rclpy.node import Node


def _ensure_ros_log_dir() -> str:
    log_dir = os.environ.get("ROS_LOG_DIR", "")
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        return log_dir
    fallback = Path.cwd() / ".cache" / "ros_logs"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["ROS_LOG_DIR"] = str(fallback)
    return str(fallback)


def _run_xacro(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class MoveItPlannerConfig:
    """Configuration for the headless FR3 MoveIt planner."""

    group_name: str = "fr3_arm"
    pipeline_id: str = "move_group"
    planner_id: str = ""
    allowed_planning_time: float = 5.0
    num_planning_attempts: int = 1
    max_velocity_scaling_factor: float = 0.2
    max_acceleration_scaling_factor: float = 0.2
    goal_tolerance: float = 1.0e-3
    arm_joint_names: tuple[str, ...] = (
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    )
    robot_ip: str = "127.0.0.1"
    use_fake_hardware: bool = True
    fake_sensor_commands: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MoveItJointPlan:
    """A MoveIt joint trajectory result in a lightweight repo-local form."""

    joint_names: tuple[str, ...]
    points: tuple[tuple[float, ...], ...]
    time_from_start_s: tuple[float, ...]
    planning_time_s: float


def _patched_ompl_planning_config(package_share: Path) -> dict:
    ompl = copy.deepcopy(_load_yaml(package_share / "config" / "ompl_planning.yaml"))
    if "panda_arm" in ompl and "fr3_arm" not in ompl:
        ompl["fr3_arm"] = copy.deepcopy(ompl["panda_arm"])
    if "panda_arm_hand" in ompl and "fr3_manipulator" not in ompl:
        ompl["fr3_manipulator"] = copy.deepcopy(ompl["panda_arm_hand"])
    return ompl


def _build_move_group_params(cfg: MoveItPlannerConfig) -> dict:
    franka_description_share = Path(get_package_share_directory("franka_description"))
    moveit_share = Path(get_package_share_directory("franka_fr3_moveit_config"))
    xacro_bin = "xacro"

    robot_description = _run_xacro(
        [
            xacro_bin,
            str(franka_description_share / "robots" / "fr3" / "fr3.urdf.xacro"),
            "hand:=true",
            "arm_id:=fr3",
            f"robot_ip:={cfg.robot_ip}",
            f"use_fake_hardware:={'true' if cfg.use_fake_hardware else 'false'}",
            f"fake_sensor_commands:={'true' if cfg.fake_sensor_commands else 'false'}",
            "ros2_control:=false",
        ]
    )
    robot_description_semantic = _run_xacro(
        [
            xacro_bin,
            str(moveit_share / "srdf" / "fr3_arm.srdf.xacro"),
            "hand:=true",
            "arm_id:=fr3",
        ]
    )
    kinematics_yaml = _load_yaml(moveit_share / "config" / "kinematics.yaml")
    ompl_yaml = _patched_ompl_planning_config(moveit_share)
    params = {
        "robot_description": robot_description,
        "robot_description_semantic": robot_description_semantic,
        "robot_description_kinematics": kinematics_yaml,
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "allow_trajectory_execution": False,
            "request_adapters": (
                "default_planner_request_adapters/AddTimeOptimalParameterization "
                "default_planner_request_adapters/ResolveConstraintFrames "
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ),
            "start_state_max_bounds_error": 0.1,
        },
        "planning_scene_monitor": {
            "publish_planning_scene": False,
            "publish_geometry_updates": False,
            "publish_state_updates": False,
            "publish_transforms_updates": False,
        },
    }
    params.update(ompl_yaml)
    return {"/move_group": {"ros__parameters": params}}


class MoveItHeadlessFr3Server:
    """Manage a headless `move_group` process for FR3 planning."""

    def __init__(self, cfg: MoveItPlannerConfig, *, log_path: str | Path | None = None) -> None:
        self._cfg = cfg
        self._params_file = Path(
            tempfile.NamedTemporaryFile(prefix="fr3_move_group_", suffix=".yaml", delete=False).name
        )
        self._log_path = Path(log_path) if log_path is not None else Path("artifacts/move_group_headless.log")
        self._proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        params = _build_move_group_params(self._cfg)
        self._params_file.write_text(yaml.safe_dump(params), encoding="utf-8")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self._log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update(self._cfg.extra_env)
        env["ROS_LOG_DIR"] = _ensure_ros_log_dir()
        self._proc = subprocess.Popen(
            [
                "ros2",
                "run",
                "moveit_ros_move_group",
                "move_group",
                "--ros-args",
                "--params-file",
                str(self._params_file),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5.0)
            self._proc = None
        try:
            self._params_file.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "MoveItHeadlessFr3Server":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


class MoveItJointPlanner:
    """Thin client over MoveIt's `/plan_kinematic_path` service."""

    def __init__(self) -> None:
        _ensure_ros_log_dir()
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("mujoco_moveit_joint_planner")
        self._client = self._node.create_client(GetMotionPlan, "/plan_kinematic_path")

    def close(self) -> None:
        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def wait_until_ready(self, timeout_sec: float = 30.0) -> bool:
        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            if self._client.wait_for_service(timeout_sec=0.5):
                return True
        return False

    def plan_joint_path(
        self,
        *,
        start_positions: Sequence[float],
        goal_positions: Sequence[float],
        cfg: MoveItPlannerConfig,
    ) -> MoveItJointPlan:
        if len(start_positions) != len(cfg.arm_joint_names) or len(goal_positions) != len(cfg.arm_joint_names):
            raise ValueError("Start/goal positions must match MoveIt arm joint count.")

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request
        motion_request.group_name = cfg.group_name
        motion_request.pipeline_id = cfg.pipeline_id
        motion_request.planner_id = cfg.planner_id
        motion_request.num_planning_attempts = int(cfg.num_planning_attempts)
        motion_request.allowed_planning_time = float(cfg.allowed_planning_time)
        motion_request.max_velocity_scaling_factor = float(cfg.max_velocity_scaling_factor)
        motion_request.max_acceleration_scaling_factor = float(cfg.max_acceleration_scaling_factor)
        motion_request.start_state.joint_state.name = list(cfg.arm_joint_names)
        motion_request.start_state.joint_state.position = [float(v) for v in start_positions]

        constraints = Constraints()
        constraints.joint_constraints = []
        for joint_name, position in zip(cfg.arm_joint_names, goal_positions):
            constraint = JointConstraint()
            constraint.joint_name = joint_name
            constraint.position = float(position)
            constraint.tolerance_above = float(cfg.goal_tolerance)
            constraint.tolerance_below = float(cfg.goal_tolerance)
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        motion_request.goal_constraints = [constraints]

        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=cfg.allowed_planning_time + 5.0)
        if future.result() is None:
            raise RuntimeError("MoveIt planning service did not return a result.")
        response = future.result().motion_plan_response
        if response.error_code.val != 1:
            raise RuntimeError(f"MoveIt planning failed with error code {response.error_code.val}.")

        trajectory = response.trajectory.joint_trajectory
        return MoveItJointPlan(
            joint_names=tuple(trajectory.joint_names),
            points=tuple(tuple(float(v) for v in point.positions) for point in trajectory.points),
            time_from_start_s=tuple(
                float(point.time_from_start.sec) + 1.0e-9 * float(point.time_from_start.nanosec)
                for point in trajectory.points
            ),
            planning_time_s=float(response.planning_time),
        )
