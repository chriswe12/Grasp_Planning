from pathlib import Path

from setuptools import setup

package_name = "robot_integration_ros"
kuka_mesh_root = Path("../../../assets/urdf/kuka_iiwa7_y_gripper/meshes")
pdz_mesh_root = Path("../../../assets/urdf/kuka_iiwa7_pdz_gripper/meshes")
pdz_mesh_data_files = [
    (
        f"share/{package_name}/meshes/pdz_gripper/{path.relative_to(pdz_mesh_root).parent}",
        [str(path)],
    )
    for path in sorted(pdz_mesh_root.rglob("*"))
    if path.is_file()
]


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/launch",
            [
                "launch/aligned_lbr_moveit.launch.py",
                "launch/dual_aligned_lbr_moveit.launch.py",
            ],
        ),
        (
            f"share/{package_name}/config",
            [
                "config/dual_iiwa7_y_gripper.srdf",
                "config/dual_iiwa7_pdz_gripper.srdf",
                "config/dual_iiwa7_y_gripper_moveit_servo.yaml",
                "config/dual_iiwa7_y_gripper_moveit_servo_hardware.yaml",
                "config/dual_iiwa7_pdz_gripper_moveit_servo.yaml",
                "config/dual_iiwa7_pdz_gripper_moveit_servo_hardware.yaml",
                "config/dual_lbr_controllers.yaml",
                "config/dual_lbr_controllers_pdz_gripper.yaml",
                "config/dual_lbr_initial_joint_positions.yaml",
                "config/dual_lbr_joint_limits.yaml",
                "config/dual_lbr_kinematics.yaml",
                "config/dual_lbr_kinematics_pick_ik.yaml",
                "config/dual_lbr_moveit.rviz",
                "config/dual_lbr_moveit_controllers.yaml",
                "config/dual_lbr_moveit_controllers_hardware.yaml",
                "config/iiwa7_y_gripper.srdf.xacro",
                "config/iiwa7_y_gripper_moveit_servo.yaml",
                "config/iiwa7_pdz_gripper.srdf.xacro",
                "config/iiwa7_pdz_gripper_moveit_servo.yaml",
                "config/lbr_10_system_config.yaml",
                "config/lbr_20_system_config.yaml",
                "config/lbr_one_system_config.yaml",
                "config/lbr_two_system_config.yaml",
                "config/single_lbr_controllers.yaml",
                "config/single_lbr_controllers_pdz_gripper.yaml",
            ],
        ),
        (
            f"share/{package_name}/urdf",
            [
                "urdf/dual_iiwa7_y_gripper_moveit.urdf.xacro",
                "urdf/dual_iiwa7_pdz_gripper_moveit.urdf.xacro",
                "urdf/iiwa7_y_gripper_moveit.urdf.xacro",
                "urdf/iiwa7_pdz_gripper_moveit.urdf.xacro",
            ],
        ),
        (f"share/{package_name}/meshes", [str(path) for path in sorted(kuka_mesh_root.glob("*.STL"))]),
        *pdz_mesh_data_files,
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Repository Maintainer",
    maintainer_email="noreply@example.com",
    description="ROS2 real-robot integration helpers for the Fabrica grasp-planning repository.",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gripper_joint_state_bridge = robot_integration_ros.gripper_joint_state_bridge:main",
            "grasp_assembly_action_server = robot_integration_ros.grasp_assembly_action_server:main",
            "move_real_robot_ee = robot_integration_ros.move_real_robot_ee:main",
        ],
    },
)
