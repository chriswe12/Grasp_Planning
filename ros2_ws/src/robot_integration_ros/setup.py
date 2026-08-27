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
        (f"share/{package_name}/launch", ["launch/aligned_lbr_moveit.launch.py"]),
        (
            f"share/{package_name}/config",
            [
                "config/iiwa7_y_gripper.srdf.xacro",
                "config/iiwa7_pdz_gripper.srdf.xacro",
                "config/iiwa7_y_gripper_moveit_servo.yaml",
            ],
        ),
        (
            f"share/{package_name}/urdf",
            [
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
            "grasp_assembly_action_server = robot_integration_ros.grasp_assembly_action_server:main",
            "move_real_robot_ee = robot_integration_ros.move_real_robot_ee:main",
        ],
    },
)
