"""MoveIt-based helpers for sending FR3 end-effector pose goals."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

try:
    import rclpy
    from geometry_msgs.msg import Pose, PoseStamped
    from moveit_msgs.action import ExecuteTrajectory
    from moveit_msgs.msg import (
        AttachedCollisionObject,
        CollisionObject,
        Constraints,
        JointConstraint,
        MoveItErrorCodes,
        PlanningScene,
    )
    from moveit_msgs.srv import (
        ApplyPlanningScene,
        GetMotionPlan,
        GetPositionFK,
        GetPositionIK,
        GetStateValidity,
        QueryPlannerInterfaces,
    )
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from shape_msgs.msg import SolidPrimitive
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    Pose = None
    PoseStamped = None
    ApplyPlanningScene = None
    AttachedCollisionObject = None
    CollisionObject = None
    ExecuteTrajectory = None
    Constraints = None
    JointConstraint = None
    MoveItErrorCodes = None
    PlanningScene = None
    GetMotionPlan = None
    GetPositionFK = None
    GetPositionIK = None
    GetStateValidity = None
    QueryPlannerInterfaces = None
    SolidPrimitive = None
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


def _normalize_ros_namespace(namespace: str) -> str:
    value = str(namespace).strip()
    if value in {"", "/"}:
        return ""
    return "/" + "/".join(part for part in value.split("/") if part)


def _normalize_ros_name(name: str) -> str:
    value = str(name).strip()
    if not value:
        raise ValueError("ROS service/action name must not be empty.")
    if not value.startswith("/"):
        value = f"/{value}"
    return "/" + "/".join(part for part in value.split("/") if part)


def _ros_name_with_namespace(name: str, *, namespace: str) -> str:
    ros_name = _normalize_ros_name(name)
    ros_namespace = _normalize_ros_namespace(namespace)
    if not ros_namespace:
        return ros_name
    if ros_name == ros_namespace or ros_name.startswith(f"{ros_namespace}/"):
        return ros_name
    return f"{ros_namespace}{ros_name}"


def _tuple3_floats(raw: object, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{field_name} must contain exactly three numeric values.")
    return tuple(float(value) for value in raw)


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
    moveit_namespace: str = ""
    ik_service_name: str = "/compute_ik"
    planning_service_name: str = "/plan_kinematic_path"
    query_planner_interface_service_name: str = "/query_planner_interface"
    fk_service_name: str = "/compute_fk"
    apply_planning_scene_service_name: str = "/apply_planning_scene"
    state_validity_service_name: str = "/check_state_validity"
    execute_action_name: str = "/execute_trajectory"
    pipeline_id: str = ""
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

    def __post_init__(self) -> None:
        namespace = _normalize_ros_namespace(self.moveit_namespace)
        object.__setattr__(self, "moveit_namespace", namespace)
        object.__setattr__(self, "joint_names", tuple(str(name) for name in self.joint_names))
        for field_name in (
            "ik_service_name",
            "planning_service_name",
            "query_planner_interface_service_name",
            "fk_service_name",
            "apply_planning_scene_service_name",
            "state_validity_service_name",
            "execute_action_name",
        ):
            object.__setattr__(
                self,
                field_name,
                _ros_name_with_namespace(str(getattr(self, field_name)), namespace=namespace),
            )


class MoveItPoseCommander(Node):
    """Small synchronous MoveIt client for terminal-driven pose goals."""

    def __init__(self, config: MoveItPoseCommanderConfig, *, node_name: str = "fr3_moveit_pose_commander") -> None:
        if (
            rclpy is None
            or PoseStamped is None
            or GetPositionIK is None
            or GetMotionPlan is None
            or GetPositionFK is None
            or GetStateValidity is None
            or QueryPlannerInterfaces is None
            or ApplyPlanningScene is None
            or ExecuteTrajectory is None
        ):
            raise RuntimeError(
                "ROS2 MoveIt dependencies are unavailable. Source the ROS2 / MoveIt workspace before running this."
            )

        super().__init__(node_name)
        self._config = config
        self._ik_client = self.create_client(GetPositionIK, config.ik_service_name)
        self._plan_client = self.create_client(GetMotionPlan, config.planning_service_name)
        self._planner_query_client = self.create_client(
            QueryPlannerInterfaces,
            config.query_planner_interface_service_name,
        )
        self._fk_client = self.create_client(GetPositionFK, config.fk_service_name)
        self._apply_planning_scene_client = self.create_client(
            ApplyPlanningScene,
            config.apply_planning_scene_service_name,
        )
        self._state_validity_client = self.create_client(
            GetStateValidity,
            config.state_validity_service_name,
        )
        self._execute_client = ActionClient(self, ExecuteTrajectory, config.execute_action_name)
        self._active_goal_handle = None

    @property
    def config(self) -> MoveItPoseCommanderConfig:
        return self._config

    def wait_for_moveit(self, *, require_execute: bool = True) -> None:
        self.get_logger().info("Waiting for MoveIt services and actions.")
        if not self._ik_client.wait_for_service(timeout_sec=self.config.wait_for_moveit_timeout_s):
            raise RuntimeError(f"MoveIt IK service '{self.config.ik_service_name}' is unavailable.")
        if not self._plan_client.wait_for_service(timeout_sec=self.config.wait_for_moveit_timeout_s):
            raise RuntimeError(f"MoveIt planning service '{self.config.planning_service_name}' is unavailable.")
        if self.config.pipeline_id and not self._planner_query_client.wait_for_service(
            timeout_sec=self.config.wait_for_moveit_timeout_s
        ):
            raise RuntimeError(
                f"MoveIt planner-query service '{self.config.query_planner_interface_service_name}' is unavailable."
            )
        if not self._fk_client.wait_for_service(timeout_sec=self.config.wait_for_moveit_timeout_s):
            raise RuntimeError(f"MoveIt FK service '{self.config.fk_service_name}' is unavailable.")
        if require_execute and not self._execute_client.wait_for_server(
            timeout_sec=self.config.wait_for_moveit_timeout_s
        ):
            raise RuntimeError(f"MoveIt execute action '{self.config.execute_action_name}' is unavailable.")
        if self.config.pipeline_id:
            self._validate_requested_pipeline()
        self.get_logger().info("MoveIt connection ready.")

    def apply_planning_scene_obstacles(
        self,
        obstacles: Sequence[Mapping[str, object]],
        *,
        default_frame_id: str,
    ) -> tuple[bool, str]:
        if not obstacles:
            return True, "No planning-scene obstacles configured."
        if (
            Pose is None
            or CollisionObject is None
            or PlanningScene is None
            or ApplyPlanningScene is None
            or SolidPrimitive is None
        ):
            return False, "MoveIt planning-scene message types are unavailable."
        if not self._apply_planning_scene_client.wait_for_service(timeout_sec=self.config.wait_for_moveit_timeout_s):
            return (
                False,
                f"MoveIt planning-scene service '{self.config.apply_planning_scene_service_name}' is unavailable.",
            )

        try:
            collision_objects = [
                self._collision_object_from_obstacle_spec(obstacle, default_frame_id=default_frame_id)
                for obstacle in obstacles
            ]
        except Exception as exc:
            return False, f"Invalid planning-scene obstacle config: {exc}"

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = collision_objects

        request = ApplyPlanningScene.Request()
        request.scene = scene
        try:
            response = self._wait_for_future(
                self._apply_planning_scene_client.call_async(request),
                timeout_s=self.config.wait_for_moveit_timeout_s,
                label="planning-scene apply",
            )
        except Exception as exc:
            return False, f"Planning-scene apply failed: {exc}"
        if response is None:
            return False, "Planning-scene apply response was None."
        if not bool(getattr(response, "success", False)):
            return False, "MoveIt rejected the planning-scene update."
        return True, f"Applied {len(collision_objects)} planning-scene obstacle(s)."

    def apply_planning_scene_robot_state(
        self,
        robot_state: Mapping[str, float],
    ) -> tuple[bool, str]:
        """Update passive or unreported joints in the shared MoveIt scene."""

        state_items = tuple((str(name), float(value)) for name, value in robot_state.items())
        if not state_items:
            return True, "No planning-scene robot-state joints requested."
        names = tuple(name for name, _value in state_items)
        if len(names) != len(set(names)):
            return False, "Planning-scene robot state contains duplicate joint names."
        if PlanningScene is None or ApplyPlanningScene is None:
            return False, "MoveIt planning-scene message types are unavailable."
        if not self._apply_planning_scene_client.wait_for_service(timeout_sec=self.config.wait_for_moveit_timeout_s):
            return (
                False,
                f"MoveIt planning-scene service '{self.config.apply_planning_scene_service_name}' is unavailable.",
            )

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.joint_state.name = list(names)
        scene.robot_state.joint_state.position = [value for _name, value in state_items]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        try:
            response = self._wait_for_future(
                self._apply_planning_scene_client.call_async(request),
                timeout_s=self.config.wait_for_moveit_timeout_s,
                label="planning-scene robot-state apply",
            )
        except Exception as exc:
            return False, f"Planning-scene robot-state apply failed: {exc}"
        if response is None:
            return False, "Planning-scene robot-state apply response was None."
        if not bool(getattr(response, "success", False)):
            return False, "MoveIt rejected the planning-scene robot-state update."
        return True, f"Applied {len(state_items)} planning-scene robot-state joint(s)."

    def apply_planning_scene_attached_obstacles(
        self,
        obstacles: Sequence[Mapping[str, object]],
        *,
        default_frame_id: str,
    ) -> tuple[bool, str]:
        """Attach collision boxes to robot links for loaded-path planning."""

        if not obstacles:
            return True, "No attached planning-scene obstacles configured."
        if AttachedCollisionObject is None or PlanningScene is None or ApplyPlanningScene is None:
            return False, "MoveIt attached-collision message types are unavailable."
        if not self._apply_planning_scene_client.wait_for_service(timeout_sec=self.config.wait_for_moveit_timeout_s):
            return (
                False,
                f"MoveIt planning-scene service '{self.config.apply_planning_scene_service_name}' is unavailable.",
            )

        attached_objects = []
        try:
            for obstacle in obstacles:
                link_name = str(obstacle.get("link_name", "")).strip()
                if not link_name:
                    raise ValueError("Each attached collision object requires a non-empty link_name.")
                collision_object = self._collision_object_from_obstacle_spec(
                    obstacle,
                    default_frame_id=link_name or default_frame_id,
                )
                attached = AttachedCollisionObject()
                attached.link_name = link_name
                attached.touch_links = [str(value) for value in obstacle.get("touch_links", ())]
                attached.object = collision_object
                attached_objects.append(attached)
        except Exception as exc:
            return False, f"Invalid attached planning-scene obstacle config: {exc}"

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = attached_objects
        request = ApplyPlanningScene.Request()
        request.scene = scene
        try:
            response = self._wait_for_future(
                self._apply_planning_scene_client.call_async(request),
                timeout_s=self.config.wait_for_moveit_timeout_s,
                label="attached planning-scene apply",
            )
        except Exception as exc:
            return False, f"Attached planning-scene apply failed: {exc}"
        if response is None:
            return False, "Attached planning-scene apply response was None."
        if not bool(getattr(response, "success", False)):
            return False, "MoveIt rejected the attached planning-scene update."
        return True, f"Attached {len(attached_objects)} planning-scene obstacle(s)."

    def remove_planning_scene_attached_obstacles(
        self,
        obstacles: Sequence[Mapping[str, object]],
        *,
        default_frame_id: str,
    ) -> tuple[bool, str]:
        """Detach collision objects and remove the world copies MoveIt restores.

        An attached-object ``REMOVE`` is a detach operation in MoveIt.  The
        detached object can therefore reappear in the world collision set and
        poison the next planning attempt unless the world object with the same
        id is removed explicitly after the detach succeeds.
        """

        if not obstacles:
            return True, "No attached planning-scene obstacles requested for removal."
        if (
            AttachedCollisionObject is None
            or CollisionObject is None
            or PlanningScene is None
            or ApplyPlanningScene is None
        ):
            return False, "MoveIt attached-collision message types are unavailable."
        if not self._apply_planning_scene_client.wait_for_service(timeout_sec=self.config.wait_for_moveit_timeout_s):
            return (
                False,
                f"MoveIt planning-scene service '{self.config.apply_planning_scene_service_name}' is unavailable.",
            )

        attached_objects = []
        obstacle_ids = []
        for obstacle in obstacles:
            obstacle_id = str(obstacle.get("id", "")).strip()
            link_name = str(obstacle.get("link_name", "")).strip()
            if not obstacle_id or not link_name:
                return False, "Attached collision-object removal requires id and link_name."
            obstacle_ids.append(obstacle_id)
            attached = AttachedCollisionObject()
            attached.link_name = link_name
            attached.object = CollisionObject()
            attached.object.header.frame_id = str(obstacle.get("frame_id", default_frame_id) or default_frame_id)
            attached.object.id = obstacle_id
            attached.object.operation = CollisionObject.REMOVE
            attached_objects.append(attached)

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = attached_objects
        request = ApplyPlanningScene.Request()
        request.scene = scene
        try:
            response = self._wait_for_future(
                self._apply_planning_scene_client.call_async(request),
                timeout_s=self.config.wait_for_moveit_timeout_s,
                label="attached planning-scene removal",
            )
        except Exception as exc:
            return False, f"Attached planning-scene removal failed: {exc}"
        if response is None:
            return False, "Attached planning-scene removal response was None."
        if not bool(getattr(response, "success", False)):
            return False, "MoveIt rejected the attached planning-scene removal."
        world_ok, world_message = self.remove_planning_scene_obstacles(
            obstacle_ids,
            default_frame_id=default_frame_id,
        )
        if not world_ok:
            return (
                False,
                f"Detached {len(attached_objects)} planning-scene obstacle(s), "
                f"but could not remove their world copies: {world_message}",
            )
        return (
            True,
            f"Detached and removed {len(attached_objects)} planning-scene obstacle(s) from the world.",
        )

    def remove_planning_scene_obstacles(
        self,
        obstacle_ids: Sequence[str],
        *,
        default_frame_id: str,
    ) -> tuple[bool, str]:
        normalized_ids = tuple(
            obstacle_id for obstacle_id in (str(value).strip() for value in obstacle_ids) if obstacle_id
        )
        if not normalized_ids:
            return True, "No planning-scene obstacles requested for removal."
        if CollisionObject is None or PlanningScene is None or ApplyPlanningScene is None:
            return False, "MoveIt planning-scene message types are unavailable."
        if not self._apply_planning_scene_client.wait_for_service(timeout_sec=self.config.wait_for_moveit_timeout_s):
            return (
                False,
                f"MoveIt planning-scene service '{self.config.apply_planning_scene_service_name}' is unavailable.",
            )

        collision_objects = []
        for obstacle_id in normalized_ids:
            collision_object = CollisionObject()
            collision_object.header.frame_id = str(default_frame_id)
            collision_object.id = obstacle_id
            collision_object.operation = CollisionObject.REMOVE
            collision_objects.append(collision_object)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = collision_objects

        request = ApplyPlanningScene.Request()
        request.scene = scene
        try:
            response = self._wait_for_future(
                self._apply_planning_scene_client.call_async(request),
                timeout_s=self.config.wait_for_moveit_timeout_s,
                label="planning-scene removal",
            )
        except Exception as exc:
            return False, f"Planning-scene removal failed: {exc}"
        if response is None:
            return False, "Planning-scene removal response was None."
        if not bool(getattr(response, "success", False)):
            return False, "MoveIt rejected the planning-scene removal."
        return True, f"Removed {len(collision_objects)} planning-scene obstacle(s)."

    def _collision_object_from_obstacle_spec(self, obstacle: Mapping[str, object], *, default_frame_id: str):
        obstacle_id = str(obstacle.get("id", "")).strip()
        if not obstacle_id:
            raise ValueError("Each planning-scene obstacle requires a non-empty id.")
        obstacle_type = str(obstacle.get("type", "box")).strip().lower()
        if obstacle_type != "box":
            raise ValueError(f"Unsupported obstacle type '{obstacle_type}'. Only 'box' is currently supported.")

        size_m = _tuple3_floats(obstacle.get("size_m", ()), field_name=f"{obstacle_id}.size_m")
        if any(value <= 0.0 for value in size_m):
            raise ValueError(f"{obstacle_id}.size_m values must be positive.")
        xyz = _tuple3_floats(obstacle.get("xyz", (0.0, 0.0, 0.0)), field_name=f"{obstacle_id}.xyz")
        quaternion_raw = obstacle.get("quaternion_xyzw")
        if quaternion_raw is None:
            rpy = _tuple3_floats(obstacle.get("rpy", (0.0, 0.0, 0.0)), field_name=f"{obstacle_id}.rpy")
            qx, qy, qz, qw = quaternion_from_rpy(*rpy)
        else:
            qx, qy, qz, qw = normalize_quaternion_xyzw(tuple(float(value) for value in quaternion_raw))  # type: ignore[arg-type]

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = xyz
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(size_m)

        collision_object = CollisionObject()
        collision_object.header.frame_id = str(obstacle.get("frame_id", default_frame_id) or default_frame_id)
        collision_object.id = obstacle_id
        collision_object.primitives.append(primitive)
        collision_object.primitive_poses.append(pose)
        collision_object.operation = CollisionObject.ADD
        return collision_object

    def _validate_requested_pipeline(self) -> None:
        requested = str(self.config.pipeline_id)
        response = self._wait_for_future(
            self._planner_query_client.call_async(self._planner_query_request()),
            timeout_s=self.config.wait_for_moveit_timeout_s,
            label="planner-interface query",
        )
        available = tuple(
            str(interface.pipeline_id)
            for interface in tuple(getattr(response, "planner_interfaces", ()))
            if str(interface.pipeline_id)
        )
        if requested in available:
            return
        detail = ", ".join(available) if available else "none"
        raise RuntimeError(
            f"Requested MoveIt planning pipeline '{requested}' is unavailable. "
            f"Available planning pipeline ids: {detail}."
        )

    def _planner_query_request(self):
        if QueryPlannerInterfaces is None:
            raise RuntimeError(
                "MoveIt planner-query service type is unavailable. "
                "Source the ROS2 / MoveIt workspace before validating planning pipelines."
            )
        return QueryPlannerInterfaces.Request()

    def move_to_pose(self, target: PoseTarget, *, label: str, execute: bool) -> tuple[bool, str]:
        self.get_logger().info(
            f"[{label}] Target frame={target.frame_id} "
            f"xyz=({target.x:.4f}, {target.y:.4f}, {target.z:.4f}) "
            f"quat=({target.qx:.5f}, {target.qy:.5f}, {target.qz:.5f}, {target.qw:.5f})"
        )

        trajectory, message = self.plan_to_pose(target, label=label)
        if trajectory is None:
            return False, f"{label}: {message}"

        point_count = len(tuple(trajectory.joint_trajectory.points))
        if not execute:
            return True, f"{label}: plan ready with {point_count} trajectory points"

        return self.execute_trajectory(trajectory, label=label)

    def plan_to_pose(
        self,
        target: PoseTarget,
        *,
        label: str,
        start_joint_positions: Sequence[float] | None = None,
        start_robot_state: Mapping[str, float] | None = None,
    ):
        if start_joint_positions is not None and start_robot_state is not None:
            return None, "Provide either start_joint_positions or start_robot_state, not both"
        joints, message = self.compute_ik(
            target,
            seed_joint_positions=start_joint_positions,
            seed_robot_state=start_robot_state,
        )
        if joints is None:
            return None, message
        return self.plan_to_joint_positions(
            joints,
            label=label,
            start_joint_positions=start_joint_positions,
            start_robot_state=start_robot_state,
        )

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

    def compute_ik(
        self,
        target: PoseTarget,
        *,
        seed_joint_positions: Sequence[float] | None = None,
        seed_robot_state: Mapping[str, float] | None = None,
        avoid_collisions: bool | None = None,
    ) -> tuple[list[float] | None, str]:
        if seed_joint_positions is not None and seed_robot_state is not None:
            return None, "Provide either seed_joint_positions or seed_robot_state, not both"
        if seed_joint_positions is not None:
            seed_joint_positions = tuple(float(value) for value in seed_joint_positions)
            if len(seed_joint_positions) != len(self.config.joint_names):
                return (
                    None,
                    f"Expected {len(self.config.joint_names)} seed joint positions, got {len(seed_joint_positions)}",
                )
        complete_seed_items: tuple[tuple[str, float], ...] = ()
        if seed_robot_state is not None:
            complete_seed_items = tuple((str(name), float(value)) for name, value in seed_robot_state.items())
            names = tuple(name for name, _value in complete_seed_items)
            if len(names) != len(set(names)):
                return None, "seed_robot_state contains duplicate joint names"
            missing = [joint_name for joint_name in self.config.joint_names if joint_name not in names]
            if missing:
                return None, f"Complete IK seed is missing active-group joints: {missing}"
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.config.planning_group
        request.ik_request.ik_link_name = self.config.pose_link
        request.ik_request.pose_stamped = self._pose_stamped(target)
        request.ik_request.avoid_collisions = bool(
            self.config.avoid_collisions if avoid_collisions is None else avoid_collisions
        )
        # With no explicit seed, make the empty RobotState a diff so MoveIt
        # resolves it against the current shared planning-scene state. A full
        # empty state otherwise produces conversion errors and can lose the
        # stationary second arm during dual-arm collision-aware IK.
        request.ik_request.robot_state.is_diff = True
        if complete_seed_items:
            # Merge a complete dual-arm hypothetical state into the shared
            # scene so the inactive arm remains at the candidate state being
            # validated. Unspecified gripper joints retain their scene values.
            request.ik_request.robot_state.joint_state.name = [name for name, _value in complete_seed_items]
            request.ik_request.robot_state.joint_state.position = [value for _name, value in complete_seed_items]
        elif seed_joint_positions is not None:
            request.ik_request.robot_state.is_diff = False
            request.ik_request.robot_state.joint_state.name = list(self.config.joint_names)
            request.ik_request.robot_state.joint_state.position = list(seed_joint_positions)

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

    def check_state_validity(
        self,
        robot_state: Mapping[str, float],
        *,
        group_name: str = "",
    ) -> tuple[dict[str, object] | None, str]:
        """Return full-scene validity and exact MoveIt collision contacts."""

        state_items = tuple((str(name), float(value)) for name, value in robot_state.items())
        names = tuple(name for name, _value in state_items)
        if not state_items:
            return None, "State-validity robot_state must not be empty"
        if len(names) != len(set(names)):
            return None, "State-validity robot_state contains duplicate joint names"
        if not self._state_validity_client.wait_for_service(timeout_sec=self.config.wait_for_moveit_timeout_s):
            return None, f"MoveIt state-validity service '{self.config.state_validity_service_name}' is unavailable."

        request = GetStateValidity.Request()
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = [name for name, _value in state_items]
        request.robot_state.joint_state.position = [value for _name, value in state_items]
        request.group_name = str(group_name)
        future = self._state_validity_client.call_async(request)
        try:
            response = self._wait_for_future(
                future,
                timeout_s=self.config.ik_timeout_s + 3.0,
                label="state-validity request",
            )
        except Exception as exc:
            return None, f"State-validity call failed: {exc}"
        if response is None:
            return None, "State-validity response was None"

        contacts = []
        for contact in response.contacts:
            contacts.append(
                {
                    "body_1": str(contact.contact_body_1),
                    "body_type_1": int(contact.body_type_1),
                    "body_2": str(contact.contact_body_2),
                    "body_type_2": int(contact.body_type_2),
                    "depth_m": float(contact.depth),
                    "position_world_m": [
                        float(contact.position.x),
                        float(contact.position.y),
                        float(contact.position.z),
                    ],
                    "normal_world": [
                        float(contact.normal.x),
                        float(contact.normal.y),
                        float(contact.normal.z),
                    ],
                }
            )
        return {
            "valid": bool(response.valid),
            "contacts": contacts,
            "cost_source_count": len(response.cost_sources),
            "constraint_result_count": len(response.constraint_result),
        }, "ok"

    def plan_to_joint_positions(
        self,
        joint_positions: Sequence[float],
        *,
        label: str,
        start_joint_positions: Sequence[float] | None = None,
        start_robot_state: Mapping[str, float] | None = None,
    ):
        if start_joint_positions is not None and start_robot_state is not None:
            return None, "Provide either start_joint_positions or start_robot_state, not both"
        joint_positions = tuple(float(value) for value in joint_positions)
        if len(joint_positions) != len(self.config.joint_names):
            return None, f"Expected {len(self.config.joint_names)} joint targets, got {len(joint_positions)}"
        if start_joint_positions is not None:
            start_joint_positions = tuple(float(value) for value in start_joint_positions)
            if len(start_joint_positions) != len(self.config.joint_names):
                return (
                    None,
                    f"Expected {len(self.config.joint_names)} start joint positions, got {len(start_joint_positions)}",
                )
        complete_start_items: tuple[tuple[str, float], ...] = ()
        if start_robot_state is not None:
            complete_start_items = tuple((str(name), float(value)) for name, value in start_robot_state.items())
            names = tuple(name for name, _value in complete_start_items)
            if len(names) != len(set(names)):
                return None, "start_robot_state contains duplicate joint names"
            missing = [joint_name for joint_name in self.config.joint_names if joint_name not in names]
            if missing:
                return None, f"Complete planning start state is missing active-group joints: {missing}"

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request
        motion_request.group_name = self.config.planning_group
        motion_request.num_planning_attempts = int(self.config.num_planning_attempts)
        motion_request.allowed_planning_time = float(self.config.planning_time_s)
        motion_request.max_velocity_scaling_factor = float(self.config.velocity_scale)
        motion_request.max_acceleration_scaling_factor = float(self.config.acceleration_scale)
        motion_request.start_state.is_diff = True
        if complete_start_items:
            motion_request.start_state.joint_state.name = [name for name, _value in complete_start_items]
            motion_request.start_state.joint_state.position = [value for _name, value in complete_start_items]
        elif start_joint_positions is not None:
            motion_request.start_state.is_diff = False
            motion_request.start_state.joint_state.name = list(self.config.joint_names)
            motion_request.start_state.joint_state.position = list(start_joint_positions)
        if self.config.pipeline_id:
            motion_request.pipeline_id = self.config.pipeline_id
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
