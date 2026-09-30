"""FR3 state + current-pose trajectory hold + Cartesian Servo. No planned motion."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_planning.real_franka.control_preflight import acquire_robot_lock, require_driver_manifest

_robot_lock = None


def setup(context):
    global _robot_lock
    ip = LaunchConfiguration("robot_ip").perform(context)
    fake = LaunchConfiguration("fake").perform(context)
    state_only = LaunchConfiguration("state_only").perform(context) == "true"
    if fake != "true":
        if not state_only:
            require_driver_manifest(Path(__file__).resolve().parents[1])
        _robot_lock = acquire_robot_lock(ip)
    desc = Path(get_package_share_directory("franka_description"))
    move = Path(get_package_share_directory("franka_fr3_moveit_config"))
    urdf = xacro.process_file(
        str(desc / "robots/fr3/fr3.urdf.xacro"),
        mappings=dict(
            hand="true",
            arm_id="fr3",
            robot_ip=ip,
            use_fake_hardware=fake,
            fake_sensor_commands=fake,
            ros2_control="true",
        ),
    ).toxml()
    if fake == "true":
        # GenericSystem must integrate velocity commands for the smoke test to
        # measure motion; otherwise a velocity controller only changes a field.
        tree = ET.fromstring(urdf)
        for hardware in tree.findall(".//ros2_control/hardware"):
            if hardware.findtext("plugin") in ("mock_components/GenericSystem", "fake_components/GenericSystem"):
                ET.SubElement(hardware, "param", name="calculate_dynamics").text = "true"
        urdf = ET.tostring(tree, encoding="unicode")
    srdf = xacro.process_file(str(desc / "robots/fr3/fr3.srdf.xacro"), mappings={"hand": "true"}).toxml()
    model = {"robot_description": urdf, "robot_description_semantic": srdf}
    kin = yaml.safe_load((move / "config/kinematics.yaml").read_text())
    root = Path(__file__).resolve().parents[1]
    if fake != "true" and not state_only:
        hardware = Path(get_package_share_directory("franka_hardware"))
        expected = root / ".cache/franka_velocity_driver/install/franka_hardware/share/franka_hardware"
        if hardware.resolve() != expected.resolve():
            raise RuntimeError("Rate-limited velocity driver missing: run scripts/build_franka_velocity_driver.sh")
    controller = str(root / "configs/franka_policy_controllers.yaml")
    servo = {
        "moveit_servo": dict(
            command_in_type="speed_units",
            scale=dict(linear=1.0, rotational=1.0, joint=1.0),
            publish_period=0.01,
            low_latency_mode=False,
            command_out_type="trajectory_msgs/JointTrajectory",
            publish_joint_positions=True,
            publish_joint_velocities=True,
            publish_joint_accelerations=False,
            smoothing_filter_plugin_name="online_signal_smoothing::ButterworthFilterPlugin",
            is_primary_planning_scene_monitor=True,
            move_group_name="fr3_arm",
            planning_frame="fr3_link0",
            ee_frame_name="fr3_hand_tcp",
            robot_link_command_frame="fr3_link0",
            incoming_command_timeout=0.15,
            num_outgoing_halt_msgs_to_publish=10,
            lower_singularity_threshold=17.0,
            hard_stop_singularity_threshold=30.0,
            joint_limit_margin=0.1,
            leaving_singularity_threshold_multiplier=2.0,
            cartesian_command_in_topic="~/delta_twist_cmds",
            joint_command_in_topic="~/delta_joint_cmds",
            joint_topic="/joint_states",
            status_topic="~/status",
            command_out_topic="/franka_policy_servo/joint_trajectory",
            check_collisions=True,
            collision_check_rate=30.0,
            self_collision_proximity_threshold=0.01,
            scene_collision_proximity_threshold=0.02,
        )
    }
    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": urdf, "publish_frequency": 100.0}],
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[controller, {"robot_description": urdf, "arm_id": "fr3"}],
            remappings=[("joint_states", "franka/joint_states")],
            output="screen",
            on_exit=Shutdown(reason="Franka hardware process exited"),
        ),
        Node(
            package="moveit_servo",
            executable="servo_node_main",
            name="franka_policy_servo",
            parameters=[model, kin, servo],
            output="screen",
        ),
    ]
    controllers = ["joint_state_broadcaster", "franka_robot_state_broadcaster"]
    if not state_only:
        controllers.append("fr3_arm_controller")
    for name in controllers:
        if name == "franka_robot_state_broadcaster" and fake == "true":
            continue
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[name, "--controller-manager-timeout", "30"],
            )
        )
    nodes.append(
        ExecuteProcess(
            cmd=["/usr/bin/python3", str(Path(__file__).with_name("franka_policy_joint_states.py"))],
            output="screen",
        )
    )
    if fake == "true":
        nodes.append(
            Node(
                package="franka_gripper",
                executable="fake_gripper_state_publisher.py",
                name="franka_gripper",
                parameters=[{"joint_names": ["fr3_finger_joint1", "fr3_finger_joint2"], "state_publish_rate": 50}],
            )
        )
    if fake != "true":
        nodes.append(
            Node(
                package="franka_gripper",
                executable="franka_gripper_node",
                name="franka_gripper",
                parameters=[{"robot_ip": ip, "joint_names": ["fr3_finger_joint1", "fr3_finger_joint2"]}],
                output="screen",
            )
        )
    nodes.append(
        ExecuteProcess(
            cmd=["/usr/bin/python3", str(Path(__file__).with_name("franka_policy_watchdog.py"))], output="screen"
        )
    )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="192.168.1.200"),
            DeclareLaunchArgument("fake", default_value="false"),
            DeclareLaunchArgument(
                "state_only",
                default_value="false",
                description="Read feedback without activating an arm command controller",
            ),
            OpaqueFunction(function=setup),
        ]
    )
