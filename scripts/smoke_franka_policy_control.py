#!/usr/bin/env python3
"""Motion/expiry regression on an already running fake launch, domain 79 ONLY."""

import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import TwistStamped
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_planning.real_franka.ros_control import RosControl, validate_velocity_controller


def main():
    assert os.environ.get("ROS_DOMAIN_ID") == "79", "Smoke test requires isolated ROS_DOMAIN_ID=79"
    rclpy.init()
    node = rclpy.create_node("franka_fake_control_smoke")
    latest = {}
    buffer = Buffer()
    _listener = TransformListener(buffer, node)
    _subscriptions = [
        node.create_subscription(JointState, "/joint_states", lambda m: latest.update(joints=m), 10),
        node.create_subscription(
            JointTrajectoryControllerState,
            "/fr3_arm_controller/controller_state",
            lambda m: latest.update(controller=m),
            10,
        ),
        node.create_subscription(Int8, "/franka_policy_servo/status", lambda m: latest.update(status=m.data), 10),
    ]
    pub = node.create_publisher(TwistStamped, "/franka_policy/command", 10)

    def spin(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.002)

    def call(kind, name, request):
        client = node.create_client(kind, name)
        try:
            assert client.wait_for_service(timeout_sec=10), name
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=5)
            assert future.done(), name
            return future.result()
        finally:
            node.destroy_client(client)

    def send(speed):
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        msg.twist.linear.z = speed
        pub.publish(msg)

    def position():
        t = buffer.lookup_transform("fr3_link0", "fr3_hand_tcp", rclpy.time.Time()).transform.translation
        return np.array([t.x, t.y, t.z])

    allowed = False
    try:
        # Check the actual loaded URDF before publishing even a zero command.
        req = GetParameters.Request(names=["robot_description"])
        description = call(GetParameters, "/controller_manager/get_parameters", req).values[0].string_value
        assert (
            "components/GenericSystem" in description and "franka_hardware/FrankaHardwareInterface" not in description
        )
        assert 'name="calculate_dynamics"' in description
        controllers = call(ListControllers, "/controller_manager/list_controllers", ListControllers.Request())
        validate_velocity_controller(controllers.controller)
        allowed = True
        assert call(Trigger, "/franka_policy_servo/start_servo", Trigger.Request()).success
        spin(0.5)
        policy_check = RosControl.__new__(RosControl)
        policy_check.node = node
        policy_check.require_single_command_sources()
        start = position()
        outputs = []
        for _ in range(30):
            send(0.005)
            spin(1 / 15)
            outputs.append(list(latest["controller"].output.velocities))
        end = position()
        spin(0.7)  # no more policy commands: independent watchdog must stop
        settled = position()
        spin(0.4)
        final = position()
        dq = np.array(latest["controller"].feedback.velocities)
        result = dict(
            fake_hardware=True,
            command_speed_m_s=0.005,
            command_duration_s=2.0,
            tcp_delta_m=(end - start).tolist(),
            stop_drift_m=float(np.linalg.norm(final - settled)),
            stopped_joint_speed_max_rad_s=float(np.max(np.abs(dq))),
            output_velocity_max_rad_s=float(np.max(np.abs(outputs))),
            servo_status=latest.get("status"),
            claimed_interfaces=[c.claimed_interfaces for c in controllers.controller if c.name == "fr3_arm_controller"][
                0
            ],
        )
        print(json.dumps(result, indent=2), flush=True)
        assert 0.003 < end[2] - start[2] < 0.02, "Fake TCP did not follow velocity command"
        assert result["stop_drift_m"] < 0.0001 and result["stopped_joint_speed_max_rad_s"] < 0.001
        assert result["output_velocity_max_rad_s"] > 0.001
        # Freeze only the domain-79 fake guard, while it has forwarded motion.
        # This proves JTC's own timeout works even if the guard process stalls.
        guards = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                command = (proc / "cmdline").read_bytes().split(b"\0")
                environment = (proc / "environ").read_bytes().split(b"\0")
            except OSError:
                continue
            if b"ROS_DOMAIN_ID=79" in environment and any(c.endswith(b"/franka_policy_watchdog.py") for c in command):
                guards.append(int(proc.name))
        assert len(guards) == 1, f"Expected exactly one isolated fake watchdog: {guards}"
        for _ in range(8):
            send(0.005)
            spin(1 / 15)
        assert max(abs(v) for v in latest["controller"].output.velocities) > 0.001
        os.kill(guards[0], signal.SIGSTOP)
        try:
            spin(0.6)
            stopped = max(abs(v) for v in latest["controller"].output.velocities)
            print(json.dumps({"guard_stall_timeout_velocity_rad_s": stopped}), flush=True)
            assert stopped < 1e-6, "Controller retained velocity after guard stalled"
        finally:
            os.kill(guards[0], signal.SIGCONT)
    finally:
        if allowed:
            send(0.0)
            spin(0.3)
            call(Trigger, "/franka_policy_servo/stop_servo", Trigger.Request())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
