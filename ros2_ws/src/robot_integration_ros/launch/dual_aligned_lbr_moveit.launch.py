"""Launch two side-by-side iiwa7 arms with one selected gripper model."""

from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from lbr_bringup.moveit import LBRMoveGroupMixin
from lbr_bringup.ros2_control import LBRROS2ControlMixin
from lbr_bringup.rviz import RVizMixin
from moveit_configs_utils import MoveItConfigsBuilder


def _package_path(relative_path: str) -> str:
    return str(Path(get_package_share_directory("robot_integration_ros")) / relative_path)


def _launch_setup(context: LaunchContext):
    mode = LaunchConfiguration("mode").perform(context)
    robot_namespace = LaunchConfiguration("robot_namespace").perform(context)
    ik_solver = LaunchConfiguration("ik_solver").perform(context)
    gripper_model = LaunchConfiguration("gripper_model").perform(context)
    description_path = _package_path(f"urdf/dual_iiwa7_{gripper_model}_moveit.urdf.xacro")
    semantic_description_path = _package_path(f"config/dual_iiwa7_{gripper_model}.srdf")
    kinematics_path = _package_path(
        "config/dual_lbr_kinematics_pick_ik.yaml" if ik_solver == "pick_ik" else "config/dual_lbr_kinematics.yaml"
    )
    joint_limits_path = _package_path("config/dual_lbr_joint_limits.yaml")
    moveit_controllers_path = _package_path("config/dual_lbr_moveit_controllers.yaml")
    initial_joint_positions_path = _package_path("config/dual_lbr_initial_joint_positions.yaml")

    moveit_configs = (
        MoveItConfigsBuilder(robot_name="lbr_dual_arm", package_name="iiwa7_moveit_config")
        .robot_description(
            file_path=description_path,
            mappings={
                "mode": mode,
                "initial_joint_positions_path": initial_joint_positions_path,
            },
        )
        .robot_description_semantic(file_path=semantic_description_path)
        .robot_description_kinematics(file_path=kinematics_path)
        .joint_limits(file_path=joint_limits_path)
        .trajectory_execution(file_path=moveit_controllers_path)
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .to_moveit_configs()
    )
    robot_description = moveit_configs.robot_description

    robot_state_publisher = LBRROS2ControlMixin.node_robot_state_publisher(
        robot_name=robot_namespace,
        robot_description=robot_description,
        use_sim_time=False,
    )
    ros2_control_node = LBRROS2ControlMixin.node_ros2_control(
        robot_name=robot_namespace,
        use_sim_time=False,
        robot_description=robot_description,
    )
    spawners = [
        LBRROS2ControlMixin.node_controller_spawner(
            robot_name=robot_namespace,
            controller="joint_state_broadcaster",
        ),
        LBRROS2ControlMixin.node_controller_spawner(
            robot_name=robot_namespace,
            controller="lbr_one_joint_trajectory_controller",
        ),
        LBRROS2ControlMixin.node_controller_spawner(
            robot_name=robot_namespace,
            controller="lbr_two_joint_trajectory_controller",
        ),
    ]
    controller_event_handler = RegisterEventHandler(OnProcessStart(target_action=ros2_control_node, on_start=spawners))

    move_group = LBRMoveGroupMixin.node_move_group(
        namespace=robot_namespace,
        parameters=[
            moveit_configs.to_dict(),
            LBRMoveGroupMixin.params_move_group(),
            {"use_sim_time": False},
        ],
    )
    rviz = RVizMixin.node_rviz(
        rviz_cfg_pkg="robot_integration_ros",
        rviz_cfg="config/dual_lbr_moveit.rviz",
        parameters=[
            moveit_configs.robot_description,
            moveit_configs.robot_description_semantic,
            moveit_configs.robot_description_kinematics,
            moveit_configs.planning_pipelines,
            moveit_configs.joint_limits,
            {"use_sim_time": False},
        ],
        remappings=[
            ("display_planned_path", PathJoinSubstitution([robot_namespace, "display_planned_path"])),
            ("joint_states", PathJoinSubstitution([robot_namespace, "joint_states"])),
            ("monitored_planning_scene", PathJoinSubstitution([robot_namespace, "monitored_planning_scene"])),
            ("planning_scene", PathJoinSubstitution([robot_namespace, "planning_scene"])),
            ("planning_scene_world", PathJoinSubstitution([robot_namespace, "planning_scene_world"])),
            ("robot_description", PathJoinSubstitution([robot_namespace, "robot_description"])),
            (
                "robot_description_semantic",
                PathJoinSubstitution([robot_namespace, "robot_description_semantic"]),
            ),
            ("recognized_object_array", PathJoinSubstitution([robot_namespace, "recognized_object_array"])),
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    return [robot_state_publisher, ros2_control_node, controller_event_handler, move_group, rviz]


def generate_launch_description() -> LaunchDescription:
    description = LaunchDescription()
    description.add_action(
        DeclareLaunchArgument(
            "gripper_model",
            default_value="y_gripper",
            choices=["y_gripper", "pdz_gripper"],
            description="End-effector model carried by both arms.",
        )
    )
    description.add_action(
        DeclareLaunchArgument(
            "mode",
            default_value="mock",
            choices=["mock", "hardware"],
            description="Run both arms against mock ros2_control or their physical FRI interfaces.",
        )
    )
    description.add_action(
        DeclareLaunchArgument(
            "robot_namespace",
            default_value="lbr_dual_arm",
            description="Shared namespace for dual-arm control, MoveIt, and planning-scene topics.",
        )
    )
    description.add_action(
        DeclareLaunchArgument(
            "ik_solver",
            default_value="kdl",
            choices=["pick_ik", "kdl"],
            description="MoveIt IK plugin for both redundant iiwa7 arms.",
        )
    )
    description.add_action(
        DeclareLaunchArgument(
            "ctrl_cfg_pkg",
            default_value="robot_integration_ros",
            description="Package containing the dual-arm ros2_control configuration.",
        )
    )
    description.add_action(
        DeclareLaunchArgument(
            "ctrl_cfg",
            default_value="config/dual_lbr_controllers.yaml",
            description="Dual-arm ros2_control configuration relative to ctrl_cfg_pkg.",
        )
    )
    description.add_action(RVizMixin.arg_rviz())
    description.add_action(LBRMoveGroupMixin.arg_allow_trajectory_execution())
    description.add_action(LBRMoveGroupMixin.arg_capabilities())
    description.add_action(LBRMoveGroupMixin.arg_disable_capabilities())
    description.add_action(LBRMoveGroupMixin.arg_monitor_dynamics())
    description.add_action(LBRMoveGroupMixin.args_publish_monitored_planning_scene())
    description.add_action(OpaqueFunction(function=_launch_setup))
    return description
