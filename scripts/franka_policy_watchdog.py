#!/usr/bin/env python3
"""Independent command-expiry guard between MoveIt Servo and the FR3 controller.

A stale producer never leaves a last nonzero trajectory velocity latched. This
node continues to run if the camera, browser, or policy process exits.
"""

import math
import time
from copy import deepcopy

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class Guard(Node):
    def __init__(self):
        super().__init__("franka_policy_watchdog")
        self.last_command = None
        self.latest_joints = None
        self.last_target = None
        self.ever_armed = False
        self.halted = True
        self.twist = self.create_publisher(TwistStamped, "/franka_policy_servo/delta_twist_cmds", 10)
        self.trajectory = self.create_publisher(JointTrajectory, "/fr3_arm_controller/joint_trajectory", 10)
        self.create_subscription(TwistStamped, "/franka_policy/command", self.command, qos_profile_sensor_data)
        self.create_subscription(JointState, "/joint_states", self.joints, qos_profile_sensor_data)
        self.create_subscription(JointTrajectory, "/franka_policy_servo/joint_trajectory", self.target, 10)
        self.create_timer(0.01, self.tick)

    def command(self, msg):
        t = msg.twist
        v = [t.linear.x, t.linear.y, t.linear.z, t.angular.x, t.angular.y, t.angular.z]
        age = self.get_clock().now().nanoseconds * 1e-9 - (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        if (
            msg.header.frame_id != "fr3_link0"
            or not all(map(math.isfinite, v))
            or not -0.02 <= age <= 0.15
            or math.hypot(*v[:3]) > 0.04001
            or math.hypot(*v[3:]) > 0.24001
        ):
            self.last_command = None
            return
        self.last_command = time.monotonic()
        self.ever_armed = True
        self.halted = False
        self.twist.publish(msg)

    def joints(self, msg):
        values = dict(zip(msg.name, msg.position))
        names = [f"fr3_joint{i}" for i in range(1, 8)]
        if all(n in values and math.isfinite(values[n]) for n in names):
            self.latest_joints = (time.monotonic(), [values[n] for n in names])

    def target(self, msg):
        if not msg.points:
            return
        self.last_target = msg
        if self.fresh():
            # Native velocity mode consumes Servo's already filtered velocity.
            # Re-splining its 10 ms position waypoint from measured joints would
            # attenuate velocity again. The hardware driver limits acceleration
            # and jerk at 1 kHz; JTC supplies independent command expiry.
            target = deepcopy(msg)
            target.header.stamp.sec = 0
            target.header.stamp.nanosec = 0
            target.points = [target.points[-1]]
            target.points[0].time_from_start.sec = 0
            target.points[0].time_from_start.nanosec = 0
            self.trajectory.publish(target)

    def fresh(self):
        return self.last_command is not None and time.monotonic() - self.last_command < 0.15

    def tick(self):
        if not self.ever_armed or self.fresh():
            return
        # Repeat the hold, including after the policy process has crashed.
        names = [f"fr3_joint{i}" for i in range(1, 8)]
        if self.latest_joints and time.monotonic() - self.latest_joints[0] < 0.2:
            positions = self.latest_joints[1]
        elif self.last_target:
            values = dict(zip(self.last_target.joint_names, self.last_target.points[-1].positions))
            if not all(n in values for n in names):
                return
            positions = [values[n] for n in names]
        else:
            return
        if not self.halted:
            self.get_logger().warn("Policy command expired: holding current joint position")
            self.halted = True
        msg = JointTrajectory()
        # Zero timestamp means begin on receipt, avoiding a zero-duration stop
        # being rejected as already in the past after DDS transport latency.
        msg.joint_names = names
        p = JointTrajectoryPoint()
        p.positions = positions
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        # Zero native velocity immediately; the driver applies its rate limiter.
        # A repeated 100 ms spline would continually restart the deceleration.
        p.time_from_start.nanosec = 0
        msg.points = [p]
        self.trajectory.publish(msg)
        zero = TwistStamped()
        zero.header.stamp = self.get_clock().now().to_msg()
        zero.header.frame_id = "fr3_link0"
        self.twist.publish(zero)


def main():
    rclpy.init()
    node = Guard()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
