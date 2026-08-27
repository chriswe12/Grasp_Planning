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
    assert servo["check_collisions"] is True
    assert servo["is_primary_planning_scene_monitor"] is False


def test_aligned_launch_and_shell_wrapper_expose_opt_in_servo() -> None:
    launch = (
        REPO_ROOT
        / "ros2_ws/src/robot_integration_ros/launch/aligned_lbr_moveit.launch.py"
    ).read_text(encoding="utf-8")
    wrapper = (REPO_ROOT / "start_lbr_moveit.sh").read_text(encoding="utf-8")

    assert 'package="moveit_servo"' in launch
    assert 'LaunchConfiguration("servo")' in launch
    assert "--servo" in wrapper
    assert 'servo:="$([[ "${SERVO}" -eq 1 ]]' in wrapper
