"""Launch two side-by-side iiwa7 arms with one selected gripper model."""

from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from lbr_bringup.moveit import LBRMoveGroupMixin
from lbr_bringup.ros2_control import LBRROS2ControlMixin
from lbr_bringup.rviz import RVizMixin
from moveit_configs_utils import MoveItConfigsBuilder


def _package_path(relative_path: str) -> str:
    return str(Path(get_package_share_directory("robot_integration_ros")) / relative_path)


def _launch_setup(context: LaunchContext):
    mode = LaunchConfiguration("mode").perform(context)
    robots = LaunchConfiguration("robots").perform(context)
    robot_namespace = LaunchConfiguration("robot_namespace").perform(context)
    ik_solver = LaunchConfiguration("ik_solver").perform(context)
    gripper_model = LaunchConfiguration("gripper_model").perform(context)
    controller_config_package = LaunchConfiguration("ctrl_cfg_pkg").perform(context)
    controller_config = LaunchConfiguration("ctrl_cfg").perform(context)
    if controller_config == "auto":
        controller_config = (
            "config/dual_lbr_controllers_pdz_gripper.yaml"
            if gripper_model == "pdz_gripper"
            else "config/dual_lbr_controllers.yaml"
        )
        context.launch_configurations["ctrl_cfg"] = controller_config
    description_path = _package_path(f"urdf/dual_iiwa7_{gripper_model}_moveit.urdf.xacro")
    semantic_description_path = _package_path(f"config/dual_iiwa7_{gripper_model}.srdf")
    kinematics_path = _package_path(
        "config/dual_lbr_kinematics_pick_ik.yaml" if ik_solver == "pick_ik" else "config/dual_lbr_kinematics.yaml"
    )
    joint_limits_path = _package_path("config/dual_lbr_joint_limits.yaml")
    moveit_controllers_path = _package_path(
        "config/dual_lbr_moveit_controllers_hardware.yaml"
        if mode == "hardware"
        else "config/dual_lbr_moveit_controllers.yaml"
    )
    initial_joint_positions_path = _package_path("config/dual_lbr_initial_joint_positions.yaml")
    servo_suffix = "_hardware" if mode == "hardware" else ""
    servo_config_path = _package_path(
        f"config/dual_iiwa7_{gripper_model}_moveit_servo{servo_suffix}.yaml"
    )

    lbr_one_mode = mode if mode == "mock" or robots in {"left", "both"} else "mock"
    lbr_two_mode = mode if mode == "mock" or robots in {"right", "both"} else "mock"
    physical_sides = "" if mode == "mock" else robots.replace("both", "left,right")

    moveit_configs = (
        MoveItConfigsBuilder(robot_name="lbr_dual_arm", package_name="iiwa7_moveit_config")
        .robot_description(
            file_path=description_path,
            mappings={
                "mode": mode,
                "lbr_one_mode": lbr_one_mode,
                "lbr_two_mode": lbr_two_mode,
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
    control_actions = []
    if mode == "mock":
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
        control_actions.extend(
            (
                ros2_control_node,
                RegisterEventHandler(OnProcessStart(target_action=ros2_control_node, on_start=spawners)),
            )
        )
    else:
        # The upstream LBR hardware interface requires the literal resource
        # names auxiliary_sensor and estimated_ft_sensor. Two hardware systems
        # therefore cannot safely share one ResourceManager. Keep one manager
        # per arm and merge their joint states for the shared MoveIt model.
        controller_config_path = str(
            Path(get_package_share_directory(controller_config_package)) / controller_config
        )
        single_description_path = _package_path(f"urdf/iiwa7_{gripper_model}_moveit.urdf.xacro")
        source_topics = []
        for robot, control_mode, system_config_name in (
            ("lbr_one", lbr_one_mode, "lbr_one_system_config.yaml"),
            ("lbr_two", lbr_two_mode, "lbr_two_system_config.yaml"),
        ):
            control_namespace = f"{robot_namespace}/{robot}_control"
            source_topics.append(f"/{control_namespace}/joint_states")
            control_robot_description = {
                "robot_description": ParameterValue(
                    Command(
                        [
                            FindExecutable(name="xacro"),
                            " ",
                            single_description_path,
                            " robot_name:=",
                            robot,
                            " mode:=",
                            control_mode,
                            " system_config_path:=",
                            _package_path(f"config/{system_config_name}"),
                            " initial_joint_positions_path:=",
                            initial_joint_positions_path,
                        ]
                    ),
                    value_type=str,
                )
            }
            control_node = Node(
                package="controller_manager",
                executable="ros2_control_node",
                namespace=control_namespace,
                output="screen",
                parameters=[
                    {"use_sim_time": False},
                    controller_config_path,
                    control_robot_description,
                ],
                remappings=[("~/robot_description", "robot_description")],
            )
            arm_spawners = [
                LBRROS2ControlMixin.node_controller_spawner(
                    robot_name=control_namespace,
                    controller="joint_state_broadcaster",
                ),
                LBRROS2ControlMixin.node_controller_spawner(
                    robot_name=control_namespace,
                    controller=f"{robot}_joint_trajectory_controller",
                ),
            ]
            if control_mode == "hardware":
                arm_spawners.append(
                    LBRROS2ControlMixin.node_controller_spawner(
                        robot_name=control_namespace,
                        controller=f"{robot}_force_torque_broadcaster",
                    )
                )
            control_actions.extend(
                (
                    control_node,
                    RegisterEventHandler(OnProcessStart(target_action=control_node, on_start=arm_spawners)),
                )
            )
        control_actions.append(
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                name="joint_state_aggregator",
                namespace=robot_namespace,
                output="screen",
                parameters=[
                    robot_description,
                    {
                        "source_list": source_topics,
                        "rate": 100.0,
                    },
                ],
            )
        )

    gripper_joint_state_bridge = Node(
        package="robot_integration_ros",
        executable="gripper_joint_state_bridge",
        name="gripper_joint_state_bridge",
        namespace=robot_namespace,
        output="screen",
        parameters=[
            {
                "layout": "dual",
                "gripper_model": gripper_model,
                "physical_sides": physical_sides,
                "left_feedback_topic": "/left/gripper_controller/position",
                "right_feedback_topic": "/right/gripper_controller/position",
                "publish_rate_hz": 20.0,
                "feedback_stale_warning_s": 1.0,
                "use_sim_time": False,
            }
        ],
    )

    move_group = LBRMoveGroupMixin.node_move_group(
        namespace=robot_namespace,
        parameters=[
            moveit_configs.to_dict(),
            LBRMoveGroupMixin.params_move_group(),
            {"use_sim_time": False},
        ],
    )
    servo_nodes = [
        Node(
            package="moveit_servo",
            executable="servo_node_main",
            name=f"{robot}_servo_node",
            namespace=robot_namespace,
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
        for robot in ("lbr_one", "lbr_two")
    ]
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
    return [
        robot_state_publisher,
        *control_actions,
        gripper_joint_state_bridge,
        move_group,
        *servo_nodes,
        rviz,
    ]


def generate_launch_description() -> LaunchDescription:
    description = LaunchDescription()
    description.add_action(
        DeclareLaunchArgument(
            "gripper_model",
            default_value="pdz_gripper",
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
            "robots",
            default_value="both",
            choices=["left", "right", "both"],
            description=(
                "Physical arms connected in hardware mode. An inactive arm remains "
                "as a mock collision participant in the shared MoveIt scene."
            ),
        )
    )
    description.add_action(
        DeclareLaunchArgument(
            "servo",
            default_value="false",
            choices=["true", "false"],
            description="Start one collision-checking MoveIt Servo node per arm.",
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
            default_value="auto",
            description="Dual-arm controller YAML relative to ctrl_cfg_pkg; auto follows gripper_model.",
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
