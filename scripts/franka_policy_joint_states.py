#!/usr/bin/env python3
"""Forward partial arm/hand states immediately, preserving source timestamps.

MoveIt and robot_state_publisher already merge partial joint messages. A periodic
joint_state_publisher here reduced 1 kHz arm feedback to 50 Hz and restamped it.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


class JointStateRelay(Node):
    def __init__(self):
        super().__init__("franka_policy_joint_states")
        # Reliable output matches the standard MoveIt CurrentStateMonitor.
        self.output = self.create_publisher(JointState, "/joint_states", 10)
        self.arm = self.create_subscription(
            JointState, "/franka/joint_states", self.output.publish, qos_profile_sensor_data
        )
        self.hand = self.create_subscription(
            JointState, "/franka_gripper/joint_states", self.output.publish, qos_profile_sensor_data
        )


def main():
    rclpy.init()
    node = JointStateRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
