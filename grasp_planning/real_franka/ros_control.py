"""Read FR3 feedback; publish bounded Cartesian twists to watchdog-protected Servo."""

import time

import numpy as np

from .core import rotation


class RosControl:
    def __init__(self):
        import rclpy
        from control_msgs.msg import JointTrajectoryControllerState
        from franka_msgs.msg import FrankaRobotState
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from tf2_ros import Buffer, TransformListener

        from grasp_planning.ros2.visual_servo_command_sink import MoveItServoCommandSink

        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("franka_visual_policy")
        self.buffer = Buffer()
        # The policy samples at 15 Hz, while dynamic TF arrives at 100 Hz.
        # A 100-message default queue accumulated old transforms within our
        # bounded spin budget. Consume the newest arm snapshot; do not replay
        # historical feedback. Leave static TF's transient-local QoS unchanged.
        self.listener = TransformListener(
            self.buffer,
            self.node,
            qos=QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.msg = None
        self.received = 0
        self.width = None
        self.width_received = 0
        self.controller_feedback = None
        self.controller_received = 0
        self.controller_sub = self.node.create_subscription(
            JointTrajectoryControllerState,
            "/fr3_arm_controller/controller_state",
            self._controller,
            qos_profile_sensor_data,
        )
        self.grip_sub = self.node.create_subscription(
            JointState, "/franka_gripper/joint_states", self._gripper, qos_profile_sensor_data
        )
        self.sub = self.node.create_subscription(
            FrankaRobotState, "/franka_robot_state_broadcaster/robot_state", self._state, qos_profile_sensor_data
        )
        self.sink = MoveItServoCommandSink(
            self.node,
            twist_topic="/franka_policy/command",
            status_topic="/franka_policy_servo/status",
            start_service="/franka_policy_servo/start_servo",
            stop_service="/franka_policy_servo/stop_servo",
        )

    def _gripper(self, msg):
        if len(msg.position) == 2 and np.isfinite(msg.position).all():
            self.width = float(sum(msg.position))
            self.width_received = time.monotonic()

    def _state(self, msg):
        self.msg = msg
        self.received = time.monotonic()

    def _controller(self, msg):
        self.controller_received = time.monotonic()
        self.controller_feedback = dict(
            joint_names=list(msg.joint_names),
            reference_positions=list(msg.reference.positions),
            reference_velocities=list(msg.reference.velocities),
            measured_positions=list(msg.feedback.positions),
            measured_velocities=list(msg.feedback.velocities),
            commanded_efforts=list(msg.output.effort),
            commanded_velocities=list(msg.output.velocities),
        )

    def spin(self):
        # One callback per 15 Hz tick lets high-rate feedback starve TF callbacks.
        # Drain a bounded batch so freshness checks see the newest queued data.
        deadline = time.monotonic() + 0.004
        for _ in range(128):
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
            if time.monotonic() >= deadline:
                break

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def state(self):
        self.spin()
        if self.msg is None or time.monotonic() - self.received > 0.2:
            raise RuntimeError("No fresh Franka feedback; connect and enable FCI")
        m = self.msg
        age = self.now() - (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        if not -0.05 <= age <= 0.2:
            raise RuntimeError("Stale Franka source timestamp")
        from rclpy.time import Time

        t = self.buffer.lookup_transform("fr3_link0", "fr3_hand_tcp", Time())
        tcp_age = self.now() - (t.header.stamp.sec + t.header.stamp.nanosec * 1e-9)
        if not -0.05 <= tcp_age <= 0.2:
            raise RuntimeError(f"Stale TCP transform: age {tcp_age:.3f} s (maximum 0.200 s)")
        q = t.transform.rotation
        p = t.transform.translation
        wrench = m.o_f_ext_hat_k.wrench
        dq = np.asarray(m.measured_joint_state.velocity)
        if dq.shape != (7,) or not np.isfinite(dq).all():
            raise RuntimeError("Invalid joint velocity feedback")
        return dict(
            position=np.array([p.x, p.y, p.z]),
            tcp_transform_age_s=tcp_age,
            rotation=rotation([q.w, q.x, q.y, q.z]),
            force=np.array([wrench.force.x, wrench.force.y, wrench.force.z]),
            torque=np.array([wrench.torque.x, wrench.torque.y, wrench.torque.z]),
            joint_velocity=dq,
            desired_joint_velocity=list(m.desired_joint_state.velocity),
            desired_joint_acceleration=list(m.ddq_d),
            control_command_success_rate=m.control_command_success_rate,
            robot_mode=m.robot_mode,
            joints=list(m.measured_joint_state.position),
            collision=any(m.collision_indicators.is_joint_collision)
            or any(getattr(m.collision_indicators.is_cartesian_linear_collision, k) for k in ("x", "y", "z"))
            or any(getattr(m.collision_indicators.is_cartesian_angular_collision, k) for k in ("x", "y", "z")),
            controller_feedback=self.controller_feedback,
            controller_feedback_age_s=time.monotonic() - self.controller_received if self.controller_feedback else None,
        )

    def wait(self, seconds=8):
        # Arm/TF and gripper have independent publishers and DDS discovery.
        # Arm readiness alone can precede the first gripper sample.
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                state = self.state()
                self.require_fresh_aperture()
                return state
            except Exception:
                self.spin()
        state = self.state()
        self.require_fresh_aperture()
        return state

    def require_fresh_aperture(self):
        if self.width is None or time.monotonic() - self.width_received > 0.3:
            raise RuntimeError("No fresh gripper aperture feedback")

    def check_width(self, expected):
        self.require_fresh_aperture()
        if abs(self.width - expected) > 0.003:
            raise RuntimeError(
                f"Open fingers to the selected training aperture: {expected * 1000:.1f} mm (currently {self.width * 1000:.1f} mm)"
            )

    def activate(self):
        self.require_velocity_controller()
        self.sink.preflight(timeout_s=5)
        self.require_single_command_sources()
        self.sink.activate(timeout_s=3)
        self.sink.wait_until_healthy(timeout_s=3, frame_id="fr3_link0")

    def require_single_command_sources(self):
        for topic in (
            "/franka_policy/command",
            "/franka_policy_servo/joint_trajectory",
            "/fr3_arm_controller/joint_trajectory",
        ):
            count = self.node.count_publishers(topic)
            if count != 1:
                raise RuntimeError(
                    f"Expected one command publisher on {topic}; found {count}. Stop duplicate connections."
                )

    def require_velocity_controller(self):
        from controller_manager_msgs.srv import ListControllers

        client = self.node.create_client(ListControllers, "/controller_manager/list_controllers")
        try:
            if not client.wait_for_service(timeout_sec=3):
                raise RuntimeError("Controller manager unavailable")
            future = client.call_async(ListControllers.Request())
            deadline = time.monotonic() + 3
            while not future.done() and time.monotonic() < deadline:
                self.rclpy.spin_once(self.node, timeout_sec=0.01)
            if not future.done() or future.result() is None:
                raise RuntimeError("Controller inspection timed out")
            validate_velocity_controller(future.result().controller)
        finally:
            self.node.destroy_client(client)

    def send(self, v):
        self.require_single_command_sources()
        health = self.sink.health(now_s=self.now())
        if not health.healthy or health.status_code is None or health.status_age_s is None or health.status_age_s > 0.2:
            raise RuntimeError(f"Servo not healthy: {health}")
        if not self.sink.send_twist(v, frame_id="fr3_link0", stamp_s=self.now()):
            raise RuntimeError("Servo did not accept twist")

    def stop(self):
        if self.sink.active:
            self.sink.deactivate(timeout_s=1)

    def close(self):
        try:
            self.stop()
        finally:
            self.node.destroy_node()


def validate_velocity_controller(controllers):
    expected = {f"fr3_joint{i}/velocity" for i in range(1, 8)}
    for controller in controllers:
        if controller.name == "fr3_arm_controller":
            if controller.state == "active" and set(controller.claimed_interfaces) == expected:
                return
    raise RuntimeError("Native velocity controller is not active. Stop and restart ./franka_policy.sh connect")


def check_state(state, start, cfg):
    if state["robot_mode"] not in (1, 2):
        raise RuntimeError(f"Robot not idle/moving: mode {state['robot_mode']}")
    if state["collision"]:
        raise RuntimeError("Franka collision flag")
    p = state["position"]
    if not np.isfinite(p).all() or not ((p >= cfg["workspace_min_m"]) & (p <= cfg["workspace_max_m"])).all():
        raise RuntimeError("TCP outside configured base-frame workspace")
    if np.linalg.norm(p - start["position"]) > cfg["max_displacement_m"]:
        raise RuntimeError("Session displacement limit")
    angle = np.arccos(np.clip((np.trace(start["rotation"].T @ state["rotation"]) - 1) / 2, -1, 1))
    if not np.isfinite(angle) or angle > cfg["max_rotation_rad"]:
        raise RuntimeError("Session rotation limit")
    if not np.isfinite(state["force"]).all() or np.linalg.norm(state["force"]) > cfg["external_force_stop_n"]:
        raise RuntimeError(
            f"External force limit: {np.linalg.norm(state['force']):.2f} N (limit {cfg['external_force_stop_n']:.2f} N)"
        )
    if not np.isfinite(state["torque"]).all() or np.linalg.norm(state["torque"]) > cfg["external_torque_stop_nm"]:
        raise RuntimeError(
            f"External torque limit: {np.linalg.norm(state['torque']):.2f} Nm "
            f"(limit {cfg['external_torque_stop_nm']:.2f} Nm)"
        )


def close_gripper(cfg, width, opening=False, stop_event=None, monitor=None):
    """Explicit operator-only action, never called automatically by policy completion."""
    import rclpy
    from franka_msgs.action import Grasp, Move
    from rclpy.action import ActionClient

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("franka_policy_manual_close")
    action = Move if opening else Grasp
    client = ActionClient(node, action, "/franka_gripper/move" if opening else "/franka_gripper/grasp")
    handle = None
    try:
        if not client.wait_for_server(timeout_sec=3):
            raise RuntimeError("Gripper action server missing")
        if not 0 <= width <= 0.08 or not 0 < cfg["manual_gripper_force_n"] <= 20:
            raise ValueError("Invalid grasp width/force")
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Gripper action cancelled")
        if monitor:
            monitor()
        goal = action.Goal()
        goal.width = float(width)
        goal.speed = 0.02
        if not opening:
            goal.force = float(cfg["manual_gripper_force_n"])
            goal.epsilon.inner = 0.003
            goal.epsilon.outer = 0.003
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5)
        if not future.done() or not future.result().accepted:
            raise RuntimeError("Gripper goal not accepted")
        handle = future.result()
        result = handle.get_result_async()
        deadline = time.monotonic() + 8
        while not result.done():
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Gripper action cancelled")
            if time.monotonic() >= deadline:
                raise RuntimeError("Gripper action timed out")
            if monitor:
                monitor()
            rclpy.spin_once(node, timeout_sec=0.02)
        if result.result().status != 4 or not result.result().result.success:
            raise RuntimeError(str(result.result().result))
        return str(result.result().result)
    except Exception:
        if handle is not None:
            cancel = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancel, timeout_sec=2)
        raise
    finally:
        node.destroy_node()
