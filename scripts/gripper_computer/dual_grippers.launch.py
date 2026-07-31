"""Launch two conservative ST3215 gripper controllers in stable ROS namespaces."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _gripper_node(*, role: str, port_argument: str) -> Node:
    return Node(
        package="servo_gripper",
        executable="open_close_node",
        namespace=role,
        name="gripper_controller",
        output="screen",
        parameters=[
            {
                "port": LaunchConfiguration(port_argument),
                "servo_id": ParameterValue(
                    LaunchConfiguration(f"{role}_servo_id"),
                    value_type=int,
                ),
                "speed": ParameterValue(
                    LaunchConfiguration("speed"),
                    value_type=int,
                ),
                "torque_limit": ParameterValue(
                    LaunchConfiguration("torque_limit"),
                    value_type=int,
                ),
                "close_direction": ParameterValue(
                    LaunchConfiguration(f"{role}_close_direction"),
                    value_type=int,
                ),
            }
        ],
    )


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "lbr_one_port",
                default_value=("/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D047592-if00"),
            ),
            DeclareLaunchArgument(
                "lbr_two_port",
                default_value=("/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D044069-if00"),
            ),
            DeclareLaunchArgument("lbr_one_servo_id", default_value="1"),
            DeclareLaunchArgument("lbr_two_servo_id", default_value="1"),
            DeclareLaunchArgument("lbr_one_close_direction", default_value="1"),
            DeclareLaunchArgument("lbr_two_close_direction", default_value="1"),
            DeclareLaunchArgument("speed", default_value="820"),
            DeclareLaunchArgument("torque_limit", default_value="500"),
            _gripper_node(role="lbr_one", port_argument="lbr_one_port"),
            _gripper_node(role="lbr_two", port_argument="lbr_two_port"),
        ]
    )
