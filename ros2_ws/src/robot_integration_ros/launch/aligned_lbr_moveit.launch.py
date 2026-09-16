"""Launch iiwa7 mock/hardware control and MoveIt from hardware-canonical kinematics."""

from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from lbr_bringup.description import LBRDescriptionMixin
from lbr_bringup.moveit import LBRMoveGroupMixin
from lbr_bringup.ros2_control import LBRROS2ControlMixin
from lbr_bringup.rviz import RVizMixin
from moveit_configs_utils import MoveItConfigsBuilder


def _resolved_package_path(package_name: str, relative_path: str) -> str:
    return str(Path(get_package_share_directory(package_name)) / relative_path)


def _launch_setup(context: LaunchContext):
    mode = LaunchConfiguration("mode").perform(context)
    robot_name = LaunchConfiguration("robot_name").perform(context)
    gripper_model = LaunchConfiguration("gripper_model").perform(context)
    gripper_side = LaunchConfiguration("gripper_side").perform(context)
    controller_config = LaunchConfiguration("ctrl_cfg").perform(context)
    if controller_config == "auto":
        controller_config = (
            "config/single_lbr_controllers_pdz_gripper.yaml"
            if gripper_model == "pdz_gripper"
            else "config/single_lbr_controllers.yaml"
        )
        context.launch_configurations["ctrl_cfg"] = controller_config
    system_config_path = _resolved_package_path(
        LaunchConfiguration("sys_cfg_pkg").perform(context),
        LaunchConfiguration("sys_cfg").perform(context),
    )
    initial_joint_positions_path = _resolved_package_path(
        LaunchConfiguration("sys_cfg_pkg").perform(context),
        LaunchConfiguration("init_jnt_pos").perform(context),
    )
    description_path = _resolved_package_path(
        "robot_integration_ros",
        f"urdf/iiwa7_{gripper_model}_moveit.urdf.xacro",
    )
    semantic_description_path = _resolved_package_path(
        "robot_integration_ros",
        f"config/iiwa7_{gripper_model}.srdf.xacro",
    )
    servo_config_path = _resolved_package_path(
        "robot_integration_ros",
        f"config/iiwa7_{gripper_model}_moveit_servo.yaml",
    )

    moveit_configs = (
        MoveItConfigsBuilder(robot_name="iiwa7", package_name="iiwa7_moveit_config")
        .robot_description(
            file_path=description_path,
            mappings={
                "robot_name": robot_name,
                "mode": mode,
                "system_config_path": system_config_path,
                "initial_joint_positions_path": initial_joint_positions_path,
            },
        )
        .robot_description_semantic(
            file_path=semantic_description_path,
            mappings={"robot_name": robot_name},
        )
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .to_moveit_configs()
    )
    robot_description = moveit_configs.robot_description

    robot_state_publisher = LBRROS2ControlMixin.node_robot_state_publisher(
        robot_name=robot_name,
        robot_description=robot_description,
        use_sim_time=False,
    )
    ros2_control_node = LBRROS2ControlMixin.node_ros2_control(
        robot_name=robot_name,
        use_sim_time=False,
        robot_description=robot_description,
    )
    spawners = [
        LBRROS2ControlMixin.node_controller_spawner(
            robot_name=robot_name,
            controller="joint_state_broadcaster",
        ),
        LBRROS2ControlMixin.node_controller_spawner(
            robot_name=robot_name,
            controller=LaunchConfiguration("ctrl"),
        ),
    ]
    if mode == "hardware":
        spawners[1:1] = [
            LBRROS2ControlMixin.node_controller_spawner(
                robot_name=robot_name,
                controller="force_torque_broadcaster",
            ),
            LBRROS2ControlMixin.node_controller_spawner(
                robot_name=robot_name,
                controller="lbr_state_broadcaster",
            ),
        ]

    controller_event_handler = RegisterEventHandler(OnProcessStart(target_action=ros2_control_node, on_start=spawners))
    gripper_joint_state_bridge = Node(
        package="robot_integration_ros",
        executable="gripper_joint_state_bridge",
        name="gripper_joint_state_bridge",
        namespace=robot_name,
        output="screen",
        parameters=[
            {
                "layout": "single",
                "gripper_model": gripper_model,
                "single_side": gripper_side,
                "physical_sides": gripper_side if mode == "hardware" else "",
                "left_feedback_topic": "/left/gripper_controller/position",
                "right_feedback_topic": "/right/gripper_controller/position",
                "publish_rate_hz": 20.0,
                "feedback_stale_warning_s": 1.0,
                "use_sim_time": False,
            }
        ],
    )
    move_group = LBRMoveGroupMixin.node_move_group(
        namespace=robot_name,
        parameters=[
            moveit_configs.to_dict(),
            LBRMoveGroupMixin.params_move_group(),
            {"use_sim_time": False},
        ],
    )
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        namespace=robot_name,
        output="screen",
        parameters=[
            moveit_configs.robot_description,
            moveit_configs.robot_description_semantic,
            moveit_configs.robot_description_kinematics,
            moveit_configs.joint_limits,
            servo_config_path,
            {"use_sim_time": False},
        ],
        condition=IfCondition(LaunchConfiguration("servo")),
    )
    rviz = RVizMixin.node_rviz(
        rviz_cfg_pkg="iiwa7_moveit_config",
        rviz_cfg="config/moveit.rviz",
        parameters=[
            moveit_configs.robot_description,
            moveit_configs.robot_description_semantic,
            moveit_configs.robot_description_kinematics,
            moveit_configs.planning_pipelines,
            moveit_configs.joint_limits,
            {"use_sim_time": False},
        ],
        remappings=[
            ("display_planned_path", PathJoinSubstitution([robot_name, "display_planned_path"])),
            ("joint_states", PathJoinSubstitution([robot_name, "joint_states"])),
            ("monitored_planning_scene", PathJoinSubstitution([robot_name, "monitored_planning_scene"])),
            ("planning_scene", PathJoinSubstitution([robot_name, "planning_scene"])),
            ("planning_scene_world", PathJoinSubstitution([robot_name, "planning_scene_world"])),
            ("robot_description", PathJoinSubstitution([robot_name, "robot_description"])),
            ("robot_description_semantic", PathJoinSubstitution([robot_name, "robot_description_semantic"])),
            ("recognized_object_array", PathJoinSubstitution([robot_name, "recognized_object_array"])),
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    return [
        robot_state_publisher,
        ros2_control_node,
        controller_event_handler,
        gripper_joint_state_bridge,
        move_group,
        servo_node,
        rviz,
    ]


def generate_launch_description() -> LaunchDescription:
    description = LaunchDescription()
    description.add_action(
        DeclareLaunchArgument(
            "gripper_model",
            default_value="pdz_gripper",
            choices=["y_gripper", "pdz_gripper"],
            description="End-effector model; PDZ matches current planning and policy training.",
        )
    )
    description.add_action(
        DeclareLaunchArgument(
            "mode",
            default_value="mock",
            choices=["mock", "hardware"],
            description="Run against mock ros2_control or the physical LBR FRI interface.",
        )
    )
    description.add_action(LBRDescriptionMixin.arg_robot_name())
    description.add_action(
        DeclareLaunchArgument(
            "gripper_side",
            default_value="left",
            choices=["left", "right"],
            description="Persistent servo-gripper namespace paired with this standalone arm.",
        )
    )
    description.add_action(RVizMixin.arg_rviz())
    description.add_action(
        DeclareLaunchArgument(
            "servo",
            default_value="false",
            choices=["true", "false"],
            description="Start collision-checking MoveIt Servo for D405 policy commands.",
        )
    )
    description.add_action(LBRROS2ControlMixin.arg_sys_cfg_pkg())
    description.add_action(LBRROS2ControlMixin.arg_sys_cfg())
    description.add_action(LBRROS2ControlMixin.arg_init_jnt_pos())
    description.add_action(
        DeclareLaunchArgument(
            "ctrl_cfg_pkg",
            default_value="robot_integration_ros",
            description="Package containing the standalone ros2_control configuration.",
        )
    )
    description.add_action(
        DeclareLaunchArgument(
            "ctrl_cfg",
            default_value="auto",
            description="Controller YAML relative to ctrl_cfg_pkg; auto follows gripper_model.",
        )
    )
    description.add_action(LBRROS2ControlMixin.arg_ctrl())
    description.add_action(LBRMoveGroupMixin.arg_allow_trajectory_execution())
    description.add_action(LBRMoveGroupMixin.arg_capabilities())
    description.add_action(LBRMoveGroupMixin.arg_disable_capabilities())
    description.add_action(LBRMoveGroupMixin.arg_monitor_dynamics())
    description.add_action(LBRMoveGroupMixin.args_publish_monitored_planning_scene())
    description.add_action(OpaqueFunction(function=_launch_setup))
    return description
