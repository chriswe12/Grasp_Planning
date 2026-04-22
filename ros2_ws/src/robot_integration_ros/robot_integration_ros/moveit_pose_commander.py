"""MoveIt-based helpers for sending FR3 end-effector pose goals."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from moveit_msgs.action import ExecuteTrajectory
    from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
    from moveit_msgs.srv import GetMotionPlan, GetPositionFK, GetPositionIK
    from rclpy.action import ActionClient
    from rclpy.node import Node
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    PoseStamped = None
    ExecuteTrajectory = None
    Constraints = None
    JointConstraint = None
    MoveItErrorCodes = None
    GetMotionPlan = None
    GetPositionFK = None
    GetPositionIK = None
    ActionClient = None
    Node = object


DEFAULT_FR3_MOVEIT_RPY = (math.pi, 0.0, math.pi / 2.0)


def normalize_quaternion_xyzw(quaternion_xyzw: Sequence[float]) -> tuple[float, float, float, float]:
    if len(quaternion_xyzw) != 4:
        raise ValueError(f"Expected 4 quaternion values, got {len(quaternion_xyzw)}.")
    qx, qy, qz, qw = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1.0e-12:
        raise ValueError("Quaternion norm is zero.")
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    half_roll = 0.5 * float(roll)
    half_pitch = 0.5 * float(pitch)
    half_yaw = 0.5 * float(yaw)

    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)

    return normalize_quaternion_xyzw(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


@dataclass(frozen=True)
class PoseTarget:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    frame_id: str = "base"

    @classmethod
    def from_rpy(
        cls,
        *,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        frame_id: str = "base",
    ) -> "PoseTarget":
        qx, qy, qz, qw = quaternion_from_rpy(roll, pitch, yaw)
        return cls(x=float(x), y=float(y), z=float(z), qx=qx, qy=qy, qz=qz, qw=qw, frame_id=str(frame_id))

    @classmethod
    def from_quaternion(
        cls,
        *,
        x: float,
        y: float,
        z: float,
        quaternion_xyzw: Sequence[float],
        frame_id: str = "base",
    ) -> "PoseTarget":
        qx, qy, qz, qw = normalize_quaternion_xyzw(quaternion_xyzw)
        return cls(x=float(x), y=float(y), z=float(z), qx=qx, qy=qy, qz=qz, qw=qw, frame_id=str(frame_id))

    @property
    def position_xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def orientation_xyzw(self) -> tuple[float, float, float, float]:
        return (self.qx, self.qy, self.qz, self.qw)


@dataclass(frozen=True)
class MoveItPoseCommanderConfig:
    planning_group: str = "fr3_arm"
    pose_link: str = "fr3_hand_tcp"
    joint_names: tuple[str, ...] = field(default_factory=lambda: tuple(f"fr3_joint{i}" for i in range(1, 8)))
    ik_service_name: str = "/compute_ik"
    planning_service_name: str = "/plan_kinematic_path"
    fk_service_name: str = "/compute_fk"
    execute_action_name: str = "/execute_trajectory"
    planner_id: str = ""
    wait_for_moveit_timeout_s: float = 15.0
    ik_timeout_s: float = 2.0
    fk_timeout_s: float = 2.0
    planning_time_s: float = 5.0
    num_planning_attempts: int = 5
    velocity_scale: float = 0.05
    acceleration_scale: float = 0.05
    execute_timeout_s: float = 120.0
    post_execute_sleep_s: float = 0.5
    avoid_collisions: bool = True


class MoveItPoseCommander(Node):
    """Small synchronous MoveIt client for terminal-driven pose goals."""

    def __init__(self, config: MoveItPoseCommanderConfig, *, node_name: str = "fr3_moveit_pose_commander") -> None:
        if (
            rclpy is None
            or PoseStamped is None
            or GetPositionIK is None
            or GetMotionPlan is None
            or GetPositionFK is None
            or ExecuteTrajectory is None
        ):
            raise RuntimeError(
                "ROS2 MoveIt dependencies are unavailable. Source the ROS2 / MoveIt workspace before running this."
            )

        super().__init__(node_name)
        self._config = config
        self._ik_client = self.create_client(GetPositionIK, config.ik_service_name)
        self._plan_client = self.create_client(GetMotionPlan, config.planning_service_name)
        self._fk_client = self.create_client(GetPositionFK, config.fk_service_name)
        self._execute_client = ActionClient(self, ExecuteTrajectory, config.execute_action_name)
        self._active_goal_handle = None

    @property
    def config(self) -> MoveItPoseCommanderConfig:
        return self._config

    def wait_for_moveit(self) -> None:
        self.get_logger().info("Waiting for MoveIt services and actions.")
        if not self._ik_client.wait_for_service(timeout_sec=self.config.wait_for_moveit_timeout_s):
            raise RuntimeError(f"MoveIt IK service '{self.config.ik_service_name}' is unavailable.")
        if not self._plan_client.wait_for_service(timeout_sec=self.config.wait_for_moveit_timeout_s):
            raise RuntimeError(f"MoveIt planning service '{self.config.planning_service_name}' is unavailable.")
        if not self._fk_client.wait_for_service(timeout_sec=self.config.wait_for_moveit_timeout_s):
            raise RuntimeError(f"MoveIt FK service '{self.config.fk_service_name}' is unavailable.")
        if not self._execute_client.wait_for_server(timeout_sec=self.config.wait_for_moveit_timeout_s):
            raise RuntimeError(f"MoveIt execute action '{self.config.execute_action_name}' is unavailable.")
        self.get_logger().info("MoveIt connection ready.")

    def move_to_pose(self, target: PoseTarget, *, label: str, execute: bool) -> tuple[bool, str]:
        self.get_logger().info(
            f"[{label}] Target frame={target.frame_id} "
            f"xyz=({target.x:.4f}, {target.y:.4f}, {target.z:.4f}) "
            f"quat=({target.qx:.5f}, {target.qy:.5f}, {target.qz:.5f}, {target.qw:.5f})"
        )

        joints, message = self.compute_ik(target)
        if joints is None:
            return False, f"{label}: {message}"

        trajectory, message = self.plan_to_joint_positions(joints, label=label)
        if trajectory is None:
            return False, f"{label}: {message}"

        point_count = len(tuple(trajectory.joint_trajectory.points))
        if not execute:
            return True, f"{label}: plan ready with {point_count} trajectory points"

        return self.execute_trajectory(trajectory, label=label)

    def get_current_pose(self, *, frame_id: str) -> PoseTarget:
        request = GetPositionFK.Request()
        request.header.frame_id = str(frame_id)
        request.fk_link_names = [self.config.pose_link]
        request.robot_state.is_diff = True

        future = self._fk_client.call_async(request)
        response = self._wait_for_future(
            future,
            timeout_s=self.config.fk_timeout_s + 3.0,
            label="FK request",
        )
        if response is None:
            raise RuntimeError("FK response was None")
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"FK failed with code={response.error_code.val}")
        if not response.pose_stamped:
            raise RuntimeError("FK response did not include a pose")

        pose_msg = response.pose_stamped[0]
        return PoseTarget.from_quaternion(
            x=pose_msg.pose.position.x,
            y=pose_msg.pose.position.y,
            z=pose_msg.pose.position.z,
            quaternion_xyzw=(
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w,
            ),
            frame_id=pose_msg.header.frame_id or str(frame_id),
        )

    def compute_ik(self, target: PoseTarget) -> tuple[list[float] | None, str]:
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.config.planning_group
        request.ik_request.ik_link_name = self.config.pose_link
        request.ik_request.pose_stamped = self._pose_stamped(target)
        request.ik_request.avoid_collisions = bool(self.config.avoid_collisions)

        timeout_seconds = max(float(self.config.ik_timeout_s), 0.0)
        request.ik_request.timeout.sec = int(timeout_seconds)
        request.ik_request.timeout.nanosec = int((timeout_seconds % 1.0) * 1.0e9)

        future = self._ik_client.call_async(request)
        try:
            response = self._wait_for_future(future, timeout_s=self.config.ik_timeout_s + 3.0, label="IK request")
        except Exception as exc:
            return None, f"IK call failed: {exc}"

        if response is None:
            return None, "IK response was None"
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, f"IK failed with code={response.error_code.val}"

        name_to_position = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        missing_joints = [joint_name for joint_name in self.config.joint_names if joint_name not in name_to_position]
        if missing_joints:
            return None, f"IK solution missing joints: {missing_joints}"

        return [float(name_to_position[joint_name]) for joint_name in self.config.joint_names], "ok"

    def plan_to_joint_positions(self, joint_positions: Sequence[float], *, label: str):
        if len(tuple(joint_positions)) != len(self.config.joint_names):
            return None, f"Expected {len(self.config.joint_names)} joint targets, got {len(tuple(joint_positions))}"

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request
        motion_request.group_name = self.config.planning_group
        motion_request.num_planning_attempts = int(self.config.num_planning_attempts)
        motion_request.allowed_planning_time = float(self.config.planning_time_s)
        motion_request.max_velocity_scaling_factor = float(self.config.velocity_scale)
        motion_request.max_acceleration_scaling_factor = float(self.config.acceleration_scale)
        motion_request.start_state.is_diff = True
        if self.config.planner_id:
            motion_request.planner_id = self.config.planner_id

        goal = Constraints()
        goal.name = str(label)
        for joint_name, position in zip(self.config.joint_names, joint_positions):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = str(joint_name)
            joint_constraint.position = float(position)
            joint_constraint.tolerance_above = 0.001
            joint_constraint.tolerance_below = 0.001
            joint_constraint.weight = 1.0
            goal.joint_constraints.append(joint_constraint)

        motion_request.goal_constraints.append(goal)

        future = self._plan_client.call_async(request)
        try:
            response = self._wait_for_future(
                future,
                timeout_s=self.config.planning_time_s + 5.0,
                label="motion-planning request",
            )
        except Exception as exc:
            return None, f"Planning call failed: {exc}"

        if response is None:
            return None, "Planning response was None"
        if response.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, f"Planning failed with code={response.motion_plan_response.error_code.val}"

        return response.motion_plan_response.trajectory, "ok"

    def execute_trajectory(self, trajectory, *, label: str) -> tuple[bool, str]:
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory

        send_future = self._execute_client.send_goal_async(goal)
        try:
            goal_handle = self._wait_for_future(send_future, timeout_s=5.0, label="execute goal submission")
        except Exception as exc:
            return False, f"{label}: failed to send execute goal: {exc}"

        if goal_handle is None or not goal_handle.accepted:
            return False, f"{label}: execute goal was rejected"

        result_future = goal_handle.get_result_async()
        self._active_goal_handle = goal_handle
        try:
            result_wrapper = self._wait_for_future(
                result_future,
                timeout_s=self.config.execute_timeout_s,
                label="trajectory execution",
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            return False, f"{label}: execution failed: {exc}"
        finally:
            self._active_goal_handle = None

        result = result_wrapper.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return False, f"{label}: execution returned code={result.error_code.val}"

        time.sleep(max(float(self.config.post_execute_sleep_s), 0.0))
        return True, f"{label}: execution complete"

    def cancel_current_execution(self) -> tuple[bool, str]:
        goal_handle = self._active_goal_handle
        if goal_handle is None:
            return False, "Interrupt received, but no trajectory execution was active."

        cancel_future = goal_handle.cancel_goal_async()
        try:
            cancel_response = self._wait_for_future(cancel_future, timeout_s=5.0, label="trajectory cancel")
        except Exception as exc:
            return False, f"Interrupt received, but trajectory cancel failed: {exc}"

        if cancel_response is not None and tuple(getattr(cancel_response, "goals_canceling", ())):
            return (
                True,
                "Interrupt received. Sent trajectory cancel request; the robot should hold its current pose if the action server honors cancellation.",
            )
        return False, "Interrupt received, but the trajectory cancel request was not accepted."

    def _wait_for_future(self, future, *, timeout_s: float, label: str):
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(timeout_s))
        if not future.done():
            raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s")
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f"{label} raised {exception!r}")
        return future.result()

    def _pose_stamped(self, target: PoseTarget) -> PoseStamped:
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = str(target.frame_id)
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = float(target.x)
        pose_stamped.pose.position.y = float(target.y)
        pose_stamped.pose.position.z = float(target.z)
        pose_stamped.pose.orientation.x = float(target.qx)
        pose_stamped.pose.orientation.y = float(target.qy)
        pose_stamped.pose.orientation.z = float(target.qz)
        pose_stamped.pose.orientation.w = float(target.qw)
        return pose_stamped


__all__ = [
    "DEFAULT_FR3_MOVEIT_RPY",
    "MoveItPoseCommander",
    "MoveItPoseCommanderConfig",
    "PoseTarget",
    "normalize_quaternion_xyzw",
    "quaternion_from_rpy",
    "rclpy",
]
