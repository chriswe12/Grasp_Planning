from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from grasp_planning.start_poses import KUKA_MOVEIT_TO_ISAAC_JOINT_SIGNS
from scripts.build_kuka_moveit_description import (
    DEFAULT_SOURCE_URDF,
    LBR_HARDWARE_EE_OFFSET_M,
    XACRO_NS,
    build_moveit_xacro,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ISAAC_USD = REPO_ROOT / "assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda"
CHECKED_MOVEIT_XACRO = (
    REPO_ROOT / "ros2_ws" / "src" / "robot_integration_ros" / "urdf" / "iiwa7_y_gripper_moveit.urdf.xacro"
)
KUKA_CONFIGS = (
    "configs/grasp_execution_benchmark.yaml",
    "configs/grasp_pipeline_pitl_isaac.yaml",
    "configs/grasp_pipeline_sim.yaml",
    "configs/grasp_pipeline_sim_isaac.yaml",
    "configs/grasp_pipeline_sim_isaac_plumbers_block.yaml",
)
LBR_HARDWARE_JOINT_KINEMATICS = (
    ("0 0 0.1475", "0 0 1"),
    ("0 -0.0105 0.1925", "0 1 0"),
    ("0 0.0105 0.2075", "0 0 1"),
    ("0 0.0105 0.1925", "0 -1 0"),
    ("0 -0.0105 0.2075", "0 0 1"),
    ("0 -0.0707 0.1925", "0 1 0"),
    ("0 0.0707 0.091", "0 0 1"),
)


def _floats(raw: str) -> np.ndarray:
    return np.asarray([float(value.strip()) for value in raw.split(",")], dtype=float)


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=float,
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _rotation_from_rpy(raw: str) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in raw.split())
    rx = _rotation_from_axis_angle(np.array([1.0, 0.0, 0.0]), roll)
    ry = _rotation_from_axis_angle(np.array([0.0, 1.0, 0.0]), pitch)
    rz = _rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)
    return rz @ ry @ rx


def _rotation_from_quat_wxyz(values: np.ndarray) -> np.ndarray:
    w, x, y, z = values / np.linalg.norm(values)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def _urdf_fk(root: ET.Element, joint_positions: tuple[float, ...]) -> np.ndarray:
    joints = {str(joint.get("name")): joint for joint in root.findall("joint")}
    result = np.eye(4, dtype=float)
    values = {f"joint{index}": value for index, value in enumerate(joint_positions, start=1)}
    values.update({"gripper_mount_joint": 0.0, "gripper_tcp_joint": 0.0})
    for name in (*values.keys(),):
        joint = joints[name]
        origin = joint.find("origin")
        xyz = np.asarray([float(value) for value in origin.get("xyz", "0 0 0").split()], dtype=float)
        rotation = _rotation_from_rpy(origin.get("rpy", "0 0 0"))
        result = result @ _transform(xyz, rotation)
        axis_element = joint.find("axis")
        if axis_element is not None:
            axis = np.asarray([float(value) for value in axis_element.get("xyz").split()], dtype=float)
            result = result @ _transform(np.zeros(3), _rotation_from_axis_angle(axis, values[name]))
    return result


def _usd_joint_block(text: str, name: str) -> str:
    match = re.search(
        rf'def Physics(?:Revolute|Fixed)Joint "{re.escape(name)}".*?^        \}}',
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, name
    return match.group(0)


def _usd_tuple(block: str, field: str) -> np.ndarray:
    match = re.search(rf"{re.escape(field)} = \(([^)]+)\)", block)
    assert match is not None, field
    return _floats(match.group(1))


def _usd_fk(text: str, joint_positions: tuple[float, ...]) -> np.ndarray:
    result = np.eye(4, dtype=float)
    values = {f"joint{index}": value for index, value in enumerate(joint_positions, start=1)}
    values.update({"gripper_mount_joint": 0.0, "gripper_tcp_joint": 0.0})
    for name in values:
        block = _usd_joint_block(text, name)
        local0 = _transform(
            _usd_tuple(block, "physics:localPos0"),
            _rotation_from_quat_wxyz(_usd_tuple(block, "physics:localRot0")),
        )
        local1 = _transform(
            _usd_tuple(block, "physics:localPos1"),
            _rotation_from_quat_wxyz(_usd_tuple(block, "physics:localRot1")),
        )
        result = result @ local0
        axis_match = re.search(r'physics:axis = "([XYZ])"', block)
        if axis_match is not None:
            axis = np.zeros(3, dtype=float)
            axis["XYZ".index(axis_match.group(1))] = 1.0
            result = result @ _transform(np.zeros(3), _rotation_from_axis_angle(axis, values[name]))
        result = result @ np.linalg.inv(local1)
    return result


def test_generated_moveit_xacro_uses_lbr_hardware_kinematics_and_grasp_tcp(tmp_path: Path) -> None:
    output = build_moveit_xacro(
        source_urdf=DEFAULT_SOURCE_URDF,
        output_xacro=tmp_path / "iiwa7_y_gripper_moveit.urdf.xacro",
    )
    assert output.read_text(encoding="utf-8") == CHECKED_MOVEIT_XACRO.read_text(encoding="utf-8")
    generated = ET.parse(output).getroot()
    generated_joints = {str(joint.get("name")): joint for joint in generated.findall("joint")}

    for index, (expected_xyz, expected_axis) in enumerate(LBR_HARDWARE_JOINT_KINEMATICS, start=1):
        generated_joint = generated_joints[f"$(arg robot_name)_A{index}"]
        assert generated_joint.find("origin").get("xyz") == expected_xyz
        assert generated_joint.find("origin").get("rpy") == "0 0 0"
        assert generated_joint.find("axis").get("xyz") == expected_axis

    gripper_mount_joint = generated_joints["gripper_mount_joint"]
    assert gripper_mount_joint.find("origin").get("xyz") == "0 0 0.0308"
    tcp_joint = generated_joints["gripper_tcp_joint"]
    assert tcp_joint.find("origin").get("xyz") == "0 0 0.1505"
    mount_z = float(gripper_mount_joint.find("origin").get("xyz").split()[2])
    tcp_z = float(tcp_joint.find("origin").get("xyz").split()[2])
    assert math.isclose(mount_z + tcp_z, 0.1813, abs_tol=1.0e-9)
    hardware_ee_joint = generated_joints["$(arg robot_name)_joint_ee"]
    assert hardware_ee_joint.get("type") == "fixed"
    assert hardware_ee_joint.find("parent").get("link") == "$(arg robot_name)_link_7"
    assert hardware_ee_joint.find("child").get("link") == "$(arg robot_name)_link_ee"
    assert hardware_ee_joint.find("origin").get("xyz") == f"0 0 {LBR_HARDWARE_EE_OFFSET_M:g}"
    assert generated.find("link[@name='$(arg robot_name)_link_ee']") is not None
    mesh_filenames = [str(mesh.get("filename")) for mesh in generated.iter("mesh")]
    assert mesh_filenames
    assert all(filename.startswith("package://robot_integration_ros/meshes/") for filename in mesh_filenames)
    assert generated.find(f"{{{XACRO_NS}}}lbr_system_interface") is not None


def test_hardware_canonical_urdf_fk_matches_generated_isaac_usd() -> None:
    source = ET.parse(DEFAULT_SOURCE_URDF).getroot()
    usd_text = ISAAC_USD.read_text(encoding="utf-8")
    samples = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.5, 0.0, -1.3962634015954636, 0.0, 1.1, 0.0),
        (0.2, -0.4, 0.7, -1.1, 0.3, 0.8, -0.5),
    )
    for source_positions in samples:
        isaac_positions = tuple(
            sign * value for sign, value in zip(KUKA_MOVEIT_TO_ISAAC_JOINT_SIGNS, source_positions, strict=True)
        )
        urdf_pose = _urdf_fk(source, source_positions)
        usd_pose = _usd_fk(usd_text, isaac_positions)
        np.testing.assert_allclose(usd_pose, urdf_pose, atol=1.0e-5)


def test_kuka_configs_no_longer_apply_model_mismatch_compensation() -> None:
    for relative_path in KUKA_CONFIGS:
        payload = yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        config = payload.get("isaac", payload.get("isaac_execution"))
        assert config["tcp_to_grasp_offset"] == [0.0, 0.0, 0.0], relative_path
        assert config["moveit_target_position_signs"] == [1.0, 1.0, 1.0], relative_path


def test_aligned_launch_and_description_are_installed_by_ros_package() -> None:
    package_root = REPO_ROOT / "ros2_ws/src/robot_integration_ros"
    setup_text = (package_root / "setup.py").read_text(encoding="utf-8")
    launch_text = (package_root / "launch/aligned_lbr_moveit.launch.py").read_text(encoding="utf-8")
    assert '"launch/aligned_lbr_moveit.launch.py"' in setup_text
    assert '"config/iiwa7_y_gripper.srdf.xacro"' in setup_text
    assert '"urdf/iiwa7_y_gripper_moveit.urdf.xacro"' in setup_text
    assert "MoveItConfigsBuilder" in launch_text
    assert "robot_description_semantic" in launch_text
    assert 'choices=["mock", "hardware"]' in launch_text
    assert "RVizMixin.arg_rviz()" in launch_text
    assert "moveit_configs.robot_description" in launch_text
    assert 'IfCondition(LaunchConfiguration("rviz"))' in launch_text


def test_start_lbr_moveit_wrapper_exposes_aligned_rviz_flag() -> None:
    wrapper_text = (REPO_ROOT / "start_lbr_moveit.sh").read_text(encoding="utf-8")
    assert "--rviz" in wrapper_text
    assert "rviz:=" in wrapper_text
