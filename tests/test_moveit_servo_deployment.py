from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_aligned_moveit_servo_uses_speed_units_tcp_and_collision_checking() -> None:
    path = (
        REPO_ROOT
        / "ros2_ws/src/robot_integration_ros/config/iiwa7_y_gripper_moveit_servo.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    servo = payload["/**/servo_node"]["ros__parameters"]["moveit_servo"]

    assert servo["command_in_type"] == "speed_units"
    assert servo["move_group_name"] == "arm"
    assert servo["planning_frame"] == "lbr_link_0"
    assert servo["robot_link_command_frame"] == "lbr_link_0"
    assert servo["ee_frame_name"] == "gripper_tcp"
    assert servo["command_out_type"] == "trajectory_msgs/JointTrajectory"
    assert servo["command_out_topic"] == "/lbr/joint_trajectory_controller/joint_trajectory"
    assert servo["publish_period"] == 0.02
    assert servo["check_collisions"] is True
    assert servo["is_primary_planning_scene_monitor"] is False

    pdz_path = (
        REPO_ROOT
        / "ros2_ws/src/robot_integration_ros/config/iiwa7_pdz_gripper_moveit_servo.yaml"
    )
    pdz_servo = yaml.safe_load(pdz_path.read_text(encoding="utf-8"))[
        "/**/servo_node"
    ]["ros__parameters"]["moveit_servo"]
    assert pdz_servo["ee_frame_name"] == "pdz_gripper_tcp"
    assert pdz_servo["check_collisions"] is True


def test_aligned_launch_and_shell_wrapper_expose_opt_in_servo() -> None:
    launch = (
        REPO_ROOT
        / "ros2_ws/src/robot_integration_ros/launch/aligned_lbr_moveit.launch.py"
    ).read_text(encoding="utf-8")
    wrapper = (REPO_ROOT / "start_lbr_moveit.sh").read_text(encoding="utf-8")

    assert 'package="moveit_servo"' in launch
    assert 'LaunchConfiguration("servo")' in launch
    assert "--servo" in wrapper
    assert "--gripper-model" in wrapper
    assert 'GRIPPER_MODEL="pdz_gripper"' in wrapper
    assert 'default_value="pdz_gripper"' in launch
    assert 'servo:="$([[ "${SERVO}" -eq 1 ]]' in wrapper
    assert "ctrl_cfg_pkg:=robot_integration_ros" in wrapper
    assert 'controller_config="config/single_lbr_controllers.yaml"' in wrapper
    assert 'controller_config="config/single_lbr_controllers_pdz_gripper.yaml"' in wrapper


def test_single_arm_servo_receives_complete_robot_state() -> None:
    semantic = (
        REPO_ROOT
        / "ros2_ws/src/robot_integration_ros/config/iiwa7_y_gripper.srdf.xacro"
    ).read_text(encoding="utf-8")
    legacy_controllers = yaml.safe_load(
        (
            REPO_ROOT
            / "ros2_ws/src/robot_integration_ros/config/single_lbr_controllers.yaml"
        ).read_text(encoding="utf-8")
    )

    assert '<passive_joint name="left_finger_joint"/>' in semantic
    pdz_semantic = (
        REPO_ROOT
        / "ros2_ws/src/robot_integration_ros/config/iiwa7_pdz_gripper.srdf.xacro"
    ).read_text(encoding="utf-8")
    assert '<passive_joint name="pdz_gripper_left_finger_joint"/>' in pdz_semantic
    manager = legacy_controllers["/**/controller_manager"]["ros__parameters"]
    trajectory = legacy_controllers["/**/joint_trajectory_controller"]["ros__parameters"]
    assert manager["update_rate"] == 100
    servo = yaml.safe_load(
        (
            REPO_ROOT
            / "ros2_ws/src/robot_integration_ros/config/iiwa7_y_gripper_moveit_servo.yaml"
        ).read_text(encoding="utf-8")
    )["/**/servo_node"]["ros__parameters"]["moveit_servo"]
    assert servo["publish_period"] >= 2.0 / manager["update_rate"]
    assert "/**/joint_state_broadcaster" not in legacy_controllers
    pdz_controllers = yaml.safe_load(
        (
            REPO_ROOT
            / "ros2_ws/src/robot_integration_ros/config/single_lbr_controllers_pdz_gripper.yaml"
        ).read_text(encoding="utf-8")
    )
    assert "/**/joint_state_broadcaster" not in pdz_controllers
    assert trajectory["joints"] == [f"lbr_A{index}" for index in range(1, 8)]
    assert trajectory["allow_partial_joints_goal"] is False

    launch = (
        REPO_ROOT
        / "ros2_ws/src/robot_integration_ros/launch/aligned_lbr_moveit.launch.py"
    ).read_text(encoding="utf-8")
    assert 'executable="gripper_joint_state_bridge"' in launch
    assert '"physical_sides": gripper_side if mode == "hardware" else ""' in launch


def test_single_arm_policy_uses_existing_lbr_force_broadcaster() -> None:
    deployment = yaml.safe_load(
        (REPO_ROOT / "configs/visual_servo_real_d405.yaml").read_text(encoding="utf-8")
    )["visual_servo"]
    launch = (
        REPO_ROOT
        / "ros2_ws/src/robot_integration_ros/launch/aligned_lbr_moveit.launch.py"
    ).read_text(encoding="utf-8")

    assert deployment["force_topic"] == "/lbr/force_torque_broadcaster/wrench"
    assert deployment["require_force_measurement"] is True
    assert 'controller="force_torque_broadcaster"' in launch


def test_dual_moveit_servo_has_one_collision_checked_route_per_arm() -> None:
    deployment = yaml.safe_load(
        (
            REPO_ROOT
            / "ros2_ws/src/robot_integration_ros/config/dual_iiwa7_pdz_gripper_moveit_servo.yaml"
        ).read_text(encoding="utf-8")
    )
    one = deployment["/**/lbr_one_servo_node"]["ros__parameters"]["moveit_servo"]
    two = deployment["/**/lbr_two_servo_node"]["ros__parameters"]["moveit_servo"]

    assert one["move_group_name"] == "arm_one"
    assert one["ee_frame_name"] == "lbr_one_pdz_gripper_tcp"
    assert one["command_out_topic"].endswith("/lbr_one_joint_trajectory_controller/joint_trajectory")
    assert two["move_group_name"] == "arm_two"
    assert two["ee_frame_name"] == "lbr_two_pdz_gripper_tcp"
    assert two["command_out_topic"].endswith("/lbr_two_joint_trajectory_controller/joint_trajectory")
    assert one["check_collisions"] is True
    assert two["check_collisions"] is True

    launch = (
        REPO_ROOT
        / "ros2_ws/src/robot_integration_ros/launch/dual_aligned_lbr_moveit.launch.py"
    ).read_text(encoding="utf-8")
    wrapper = (REPO_ROOT / "start_dual_lbr_moveit.sh").read_text(encoding="utf-8")
    assert "lbr_one_mode" in launch and "lbr_two_mode" in launch
    assert 'LaunchConfiguration("robots")' in launch
    assert 'LaunchConfiguration("servo")' in launch
    assert "--robots" in wrapper and "--servo" in wrapper


def test_dual_hardware_uses_split_controller_managers_and_servo_outputs() -> None:
    hardware_servo = yaml.safe_load(
        (
            REPO_ROOT
            / "ros2_ws/src/robot_integration_ros/config/dual_iiwa7_pdz_gripper_moveit_servo_hardware.yaml"
        ).read_text(encoding="utf-8")
    )
    one = hardware_servo["/**/lbr_one_servo_node"]["ros__parameters"]["moveit_servo"]
    two = hardware_servo["/**/lbr_two_servo_node"]["ros__parameters"]["moveit_servo"]
    assert one["command_out_topic"] == (
        "/lbr_dual_arm/lbr_one_control/lbr_one_joint_trajectory_controller/joint_trajectory"
    )
    assert two["command_out_topic"] == (
        "/lbr_dual_arm/lbr_two_control/lbr_two_joint_trajectory_controller/joint_trajectory"
    )

    controllers = yaml.safe_load(
        (
            REPO_ROOT
            / "ros2_ws/src/robot_integration_ros/config/dual_lbr_moveit_controllers_hardware.yaml"
        ).read_text(encoding="utf-8")
    )["moveit_simple_controller_manager"]
    assert controllers["controller_names"] == [
        "lbr_one_control/lbr_one_joint_trajectory_controller",
        "lbr_two_control/lbr_two_joint_trajectory_controller",
    ]

    launch = (
        REPO_ROOT
        / "ros2_ws/src/robot_integration_ros/launch/dual_aligned_lbr_moveit.launch.py"
    ).read_text(encoding="utf-8")
    assert "cannot safely share one ResourceManager" in launch
    assert 'package="joint_state_publisher"' in launch
    assert 'executable="gripper_joint_state_bridge"' in launch
    assert '"physical_sides": physical_sides' in launch
    assert 'f"{robot_namespace}/{robot}_control"' in launch
