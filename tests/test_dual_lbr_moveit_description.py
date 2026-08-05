from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from scripts.build_kuka_moveit_description import (
    DEFAULT_DUAL_OUTPUT_XACRO,
    DEFAULT_SOURCE_URDF,
    DUAL_ARM_BASE_Y_M,
    LBR_HARDWARE_EE_OFFSET_M,
    XACRO_NS,
    build_dual_moveit_xacro,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "ros2_ws/src/robot_integration_ros"
SRDF_PATH = PACKAGE_ROOT / "config/dual_iiwa7_y_gripper.srdf"
CONTROLLERS_PATH = PACKAGE_ROOT / "config/dual_lbr_controllers.yaml"
MOVEIT_CONTROLLERS_PATH = PACKAGE_ROOT / "config/dual_lbr_moveit_controllers.yaml"
RVIZ_PATH = PACKAGE_ROOT / "config/dual_lbr_moveit.rviz"
INITIAL_JOINT_POSITIONS_PATH = PACKAGE_ROOT / "config/dual_lbr_initial_joint_positions.yaml"


def _named_elements(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    return {str(element.get("name")): element for element in root.findall(tag)}


def test_generated_dual_xacro_is_reproducible_and_has_two_complete_y_grippers(tmp_path: Path) -> None:
    output = build_dual_moveit_xacro(
        source_urdf=DEFAULT_SOURCE_URDF,
        output_xacro=tmp_path / "dual_iiwa7_y_gripper_moveit.urdf.xacro",
    )
    assert output.read_text(encoding="utf-8") == DEFAULT_DUAL_OUTPUT_XACRO.read_text(encoding="utf-8")

    root = ET.parse(output).getroot()
    assert root.get("name") == "lbr_dual_arm"
    links = _named_elements(root, "link")
    joints = _named_elements(root, "joint")
    assert len(links) == len(root.findall("link"))
    assert len(joints) == len(root.findall("joint"))

    for robot_name, expected_y_m in DUAL_ARM_BASE_Y_M.items():
        base_joint = joints[f"{robot_name}_base_joint"]
        assert base_joint.find("parent").get("link") == "base_link"
        assert base_joint.find("child").get("link") == f"{robot_name}_link_0"
        xyz = [float(value) for value in base_joint.find("origin").get("xyz").split()]
        assert xyz == [0.0, expected_y_m, 0.0]
        assert base_joint.find("origin").get("rpy") == "0 0 0"

        assert f"{robot_name}_gripper_base_link" in links
        assert f"{robot_name}_left_finger_link" in links
        assert f"{robot_name}_right_finger_link" in links
        assert f"{robot_name}_gripper_tcp" in links

        mount_joint = joints[f"{robot_name}_gripper_mount_joint"]
        tcp_joint = joints[f"{robot_name}_gripper_tcp_joint"]
        mount_z = float(mount_joint.find("origin").get("xyz").split()[2])
        tcp_z = float(tcp_joint.find("origin").get("xyz").split()[2])
        mount_rpy = [float(value) for value in mount_joint.find("origin").get("rpy").split()]
        tcp_rpy = [float(value) for value in tcp_joint.find("origin").get("rpy").split()]
        assert math.isclose(mount_z, 0.0308, abs_tol=1.0e-9)
        assert math.isclose(tcp_z, 0.1455, abs_tol=1.0e-9)
        assert math.isclose(mount_z + tcp_z, 0.1763, abs_tol=1.0e-9)
        assert mount_rpy == pytest.approx([0.0, 0.0, math.pi], abs=1.0e-8)
        assert tcp_rpy == pytest.approx([0.0, 0.0, math.pi], abs=1.0e-8)

        mimic = joints[f"{robot_name}_right_finger_joint"].find("mimic")
        assert mimic.get("joint") == f"{robot_name}_left_finger_joint"
        assert joints[f"{robot_name}_joint_ee"].find("origin").get("xyz") == (f"0 0 {LBR_HARDWARE_EE_OFFSET_M:g}")

    system_interfaces = root.findall(f"{{{XACRO_NS}}}lbr_system_interface")
    assert {element.get("robot_name") for element in system_interfaces} == {"lbr_one", "lbr_two"}
    initial_positions_arg = root.find(f"{{{XACRO_NS}}}arg[@name='initial_joint_positions_path']")
    assert initial_positions_arg is not None
    assert initial_positions_arg.get("default") == (
        "$(find robot_integration_ros)/config/dual_lbr_initial_joint_positions.yaml"
    )


def test_dual_mock_start_matches_shared_moveit_start_pose() -> None:
    from grasp_planning.start_poses import KUKA_MOVEIT_ARM_START_JOINT_VALUES

    degrees = yaml.safe_load(INITIAL_JOINT_POSITIONS_PATH.read_text(encoding="utf-8"))
    radians = tuple(math.radians(float(degrees[f"A{index}"])) for index in range(1, 8))

    assert radians == pytest.approx(KUKA_MOVEIT_ARM_START_JOINT_VALUES)


def test_dual_srdf_uses_gripper_tcp_groups_and_keeps_cross_arm_collisions_enabled() -> None:
    root = ET.parse(SRDF_PATH).getroot()
    groups = _named_elements(root, "group")
    arm_one_chain = groups["arm_one"].find("chain")
    arm_two_chain = groups["arm_two"].find("chain")
    assert (arm_one_chain.get("base_link"), arm_one_chain.get("tip_link")) == (
        "lbr_one_link_0",
        "lbr_one_gripper_tcp",
    )
    assert (arm_two_chain.get("base_link"), arm_two_chain.get("tip_link")) == (
        "lbr_two_link_0",
        "lbr_two_gripper_tcp",
    )
    assert {group.get("name") for group in groups["both_arms"].findall("group")} == {"arm_one", "arm_two"}

    disabled_pairs = [(element.get("link1"), element.get("link2")) for element in root.findall("disable_collisions")]
    assert disabled_pairs
    assert all(
        (str(link1).startswith("lbr_one_") and str(link2).startswith("lbr_one_"))
        or (str(link1).startswith("lbr_two_") and str(link2).startswith("lbr_two_"))
        for link1, link2 in disabled_pairs
    )
    assert {joint.get("name") for joint in root.findall("passive_joint")} == {
        "lbr_one_left_finger_joint",
        "lbr_two_left_finger_joint",
    }


def test_dual_controller_and_hardware_configs_are_separate_per_arm() -> None:
    controllers = yaml.safe_load(CONTROLLERS_PATH.read_text(encoding="utf-8"))
    controller_params = controllers["/**/controller_manager"]["ros__parameters"]
    assert "lbr_one_joint_trajectory_controller" in controller_params
    assert "lbr_two_joint_trajectory_controller" in controller_params
    extra_joints = controllers["/**/joint_state_broadcaster"]["ros__parameters"]["extra_joints"]
    assert extra_joints == ["lbr_one_left_finger_joint", "lbr_two_left_finger_joint"]

    one_joints = controllers["/**/lbr_one_joint_trajectory_controller"]["ros__parameters"]["joints"]
    two_joints = controllers["/**/lbr_two_joint_trajectory_controller"]["ros__parameters"]["joints"]
    assert one_joints == [f"lbr_one_A{index}" for index in range(1, 8)]
    assert two_joints == [f"lbr_two_A{index}" for index in range(1, 8)]
    assert set(one_joints).isdisjoint(two_joints)

    moveit_controllers = yaml.safe_load(MOVEIT_CONTROLLERS_PATH.read_text(encoding="utf-8"))
    manager = moveit_controllers["moveit_simple_controller_manager"]
    assert manager["controller_names"] == [
        "lbr_one_joint_trajectory_controller",
        "lbr_two_joint_trajectory_controller",
    ]
    assert manager["lbr_one_joint_trajectory_controller"]["joints"] == one_joints
    assert manager["lbr_two_joint_trajectory_controller"]["joints"] == two_joints

    expected_hardware = {
        "lbr_one": (30200, "192.170.10.2"),
        "lbr_two": (30201, "192.170.20.2"),
    }
    for robot_name, (expected_port, expected_host) in expected_hardware.items():
        config = yaml.safe_load((PACKAGE_ROOT / f"config/{robot_name}_system_config.yaml").read_text(encoding="utf-8"))
        assert config["hardware"]["port_id"] == expected_port
        assert config["hardware"]["remote_host"] == expected_host
        assert config["estimated_ft_sensor"]["chain_root"] == f"{robot_name}_link_0"
        assert config["estimated_ft_sensor"]["chain_tip"] == f"{robot_name}_link_ee"


def test_dual_launch_assets_and_wrapper_are_installed() -> None:
    setup_text = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
    launch_text = (PACKAGE_ROOT / "launch/dual_aligned_lbr_moveit.launch.py").read_text(encoding="utf-8")
    wrapper_text = (REPO_ROOT / "start_dual_lbr_moveit.sh").read_text(encoding="utf-8")

    for installed_path in (
        "launch/dual_aligned_lbr_moveit.launch.py",
        "config/dual_iiwa7_y_gripper.srdf",
        "config/dual_lbr_controllers.yaml",
        "config/dual_lbr_initial_joint_positions.yaml",
        "config/dual_lbr_kinematics.yaml",
        "config/dual_lbr_moveit_controllers.yaml",
        "urdf/dual_iiwa7_y_gripper_moveit.urdf.xacro",
    ):
        assert f'"{installed_path}"' in setup_text

    assert 'robot_name="lbr_dual_arm"' in launch_text
    assert "dual_lbr_initial_joint_positions.yaml" in launch_text
    assert 'controller="lbr_one_joint_trajectory_controller"' in launch_text
    assert 'controller="lbr_two_joint_trajectory_controller"' in launch_text
    assert "dual_aligned_lbr_moveit.launch.py" in wrapper_text
    assert "--mode hardware" in wrapper_text
    assert "setsid ros2 launch" in wrapper_text
    assert 'kill -TERM -- "-${process_group}"' in wrapper_text
    assert 'kill -KILL -- "-${process_group}"' in wrapper_text


def test_dual_rviz_uses_one_stable_compound_group_motion_planning_display() -> None:
    config = yaml.safe_load(RVIZ_PATH.read_text(encoding="utf-8"))
    displays = config["Visualization Manager"]["Displays"]
    motion_planning_displays = [
        display for display in displays if display.get("Class") == "moveit_rviz_plugin/MotionPlanning"
    ]

    assert len(motion_planning_displays) == 1
    display = motion_planning_displays[0]
    assert display["Planning Request"]["Planning Group"] == "both_arms"
    assert display["Name"] == "Both Arms MotionPlanning"
    assert all(display["Move Group Namespace"] == "lbr_dual_arm" for display in motion_planning_displays)

    marker_displays = [display for display in displays if display.get("Class") == "rviz_default_plugins/MarkerArray"]
    assert len(marker_displays) == 1
    marker_display = marker_displays[0]
    assert marker_display["Name"] == "Perceived Part AABBs"
    assert marker_display["Marker Topic"]["Value"] == "/grasp_assembly/debug_aabbs"
