from setuptools import setup

package_name = "robot_integration_ros"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
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
            "move_real_robot_ee = robot_integration_ros.move_real_robot_ee:main",
        ],
    },
)
