"""Standalone terminal entrypoint for MoveIt-driven FR3 end-effector moves."""

from __future__ import annotations

import argparse
import sys

from .moveit_pose_commander import (
    DEFAULT_FR3_MOVEIT_RPY,
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    PoseTarget,
    normalize_quaternion_xyzw,
    rclpy,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute a single FR3 end-effector pose target through MoveIt.",
        epilog=(
            "Examples:\n"
            "  ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40\n"
            "  ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --execute\n"
            "  ros2 run robot_integration_ros move_real_robot_ee --x 0.32 --y -0.10 --z 0.28 --roll 3.14159 --pitch 0 --yaw 1.5708 --execute\n"
            "  ros2 run robot_integration_ros move_real_robot_ee --x 0.32 --y -0.10 --z 0.28 --qx 0 --qy 1 --qz 0 --qw 0 --execute"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--x", required=True, type=float, help="Target x position in the planning frame.")
    parser.add_argument("--y", required=True, type=float, help="Target y position in the planning frame.")
    parser.add_argument("--z", required=True, type=float, help="Target z position in the planning frame.")
    parser.add_argument("--frame-id", default="base", help="Planning frame for the target pose.")
    parser.add_argument(
        "--roll",
        type=float,
        default=None,
        help="Target roll in radians. Defaults to the thesis neutral orientation when no quaternion is given.",
    )
    parser.add_argument("--pitch", type=float, default=None, help="Target pitch in radians.")
    parser.add_argument("--yaw", type=float, default=None, help="Target yaw in radians.")
    parser.add_argument("--qx", type=float, default=None, help="Target quaternion x component.")
    parser.add_argument("--qy", type=float, default=None, help="Target quaternion y component.")
    parser.add_argument("--qz", type=float, default=None, help="Target quaternion z component.")
    parser.add_argument("--qw", type=float, default=None, help="Target quaternion w component.")
    parser.add_argument(
        "--keep-current-orientation",
        action="store_true",
        help="Keep the current end-effector orientation from MoveIt FK and only change position.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send the planned trajectory to the real robot. Without this flag the script only validates IK and planning.",
    )
    parser.add_argument("--label", default="ee_target", help="Label used in MoveIt goal constraints and logging.")
    parser.add_argument("--planning-group", default="fr3_arm", help="MoveIt planning group name.")
    parser.add_argument("--pose-link", default="fr3_hand_tcp", help="IK end-effector link name.")
    parser.add_argument("--planner-id", default="", help="Optional MoveIt planner id. Empty means use the default.")
    parser.add_argument("--ik-timeout-s", type=float, default=2.0, help="IK timeout in seconds.")
    parser.add_argument("--planning-time-s", type=float, default=5.0, help="Allowed planning time in seconds.")
    parser.add_argument("--num-planning-attempts", type=int, default=5, help="Number of MoveIt planning attempts.")
    parser.add_argument("--velocity-scale", type=float, default=0.05, help="MoveIt velocity scaling factor.")
    parser.add_argument("--acceleration-scale", type=float, default=0.05, help="MoveIt acceleration scaling factor.")
    parser.add_argument(
        "--wait-for-moveit-timeout-s",
        type=float,
        default=15.0,
        help="Timeout while waiting for MoveIt services and actions to appear.",
    )
    parser.add_argument(
        "--execute-timeout-s",
        type=float,
        default=120.0,
        help="Timeout while waiting for MoveIt trajectory execution to finish.",
    )
    parser.add_argument(
        "--post-execute-sleep-s",
        type=float,
        default=0.5,
        help="Extra settle time after successful execution.",
    )
    parser.add_argument(
        "--allow-collisions",
        action="store_true",
        help="Disable MoveIt's avoid-collisions flag for IK. Leave unset for normal operation.",
    )
    return parser


def pose_target_from_args(
    args: argparse.Namespace,
    *,
    current_orientation_xyzw: tuple[float, float, float, float] | None = None,
) -> PoseTarget:
    quaternion_values = (args.qx, args.qy, args.qz, args.qw)
    explicit_rpy = any(value is not None for value in (args.roll, args.pitch, args.yaw))

    if args.keep_current_orientation:
        if explicit_rpy or any(value is not None for value in quaternion_values):
            raise ValueError(
                "--keep-current-orientation cannot be combined with explicit --roll/--pitch/--yaw or --qx/--qy/--qz/--qw."
            )
        if current_orientation_xyzw is None:
            raise ValueError("Current orientation is required when --keep-current-orientation is set.")
        return PoseTarget.from_quaternion(
            x=args.x,
            y=args.y,
            z=args.z,
            quaternion_xyzw=current_orientation_xyzw,
            frame_id=args.frame_id,
        )

    if any(value is not None for value in quaternion_values):
        if not all(value is not None for value in quaternion_values):
            raise ValueError("When using quaternion orientation, provide all of --qx --qy --qz --qw.")
        return PoseTarget.from_quaternion(
            x=args.x,
            y=args.y,
            z=args.z,
            quaternion_xyzw=normalize_quaternion_xyzw(quaternion_values),
            frame_id=args.frame_id,
        )

    default_roll, default_pitch, default_yaw = DEFAULT_FR3_MOVEIT_RPY
    return PoseTarget.from_rpy(
        x=args.x,
        y=args.y,
        z=args.z,
        roll=default_roll if args.roll is None else args.roll,
        pitch=default_pitch if args.pitch is None else args.pitch,
        yaw=default_yaw if args.yaw is None else args.yaw,
        frame_id=args.frame_id,
    )


def commander_config_from_args(args: argparse.Namespace) -> MoveItPoseCommanderConfig:
    return MoveItPoseCommanderConfig(
        planning_group=args.planning_group,
        pose_link=args.pose_link,
        planner_id=args.planner_id,
        wait_for_moveit_timeout_s=args.wait_for_moveit_timeout_s,
        ik_timeout_s=args.ik_timeout_s,
        planning_time_s=args.planning_time_s,
        num_planning_attempts=args.num_planning_attempts,
        velocity_scale=args.velocity_scale,
        acceleration_scale=args.acceleration_scale,
        execute_timeout_s=args.execute_timeout_s,
        post_execute_sleep_s=args.post_execute_sleep_s,
        avoid_collisions=not args.allow_collisions,
    )


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_argument_parser()
    args, ros_args = parser.parse_known_args(raw_argv)

    if rclpy is None:
        print("ROS2 MoveIt dependencies are unavailable. Source the ROS2 / MoveIt workspace first.", file=sys.stderr)
        return 1

    config = commander_config_from_args(args)
    rclpy.init(args=ros_args)
    node = None
    try:
        node = MoveItPoseCommander(config)
        node.wait_for_moveit()
        current_orientation_xyzw = None
        if args.keep_current_orientation:
            current_pose = node.get_current_pose(frame_id=args.frame_id)
            current_orientation_xyzw = current_pose.orientation_xyzw
            print(
                "Using current orientation "
                f"({current_pose.qx:.5f}, {current_pose.qy:.5f}, {current_pose.qz:.5f}, {current_pose.qw:.5f})",
                file=sys.stdout,
            )
        try:
            target = pose_target_from_args(args, current_orientation_xyzw=current_orientation_xyzw)
        except ValueError as exc:
            parser.error(str(exc))
        ok, message = node.move_to_pose(target, label=args.label, execute=bool(args.execute))
        print(message, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1
    except KeyboardInterrupt:
        if node is None:
            print("Interrupt received before the ROS2 node was ready.", file=sys.stderr)
            return 130
        cancelled, message = node.cancel_current_execution()
        print(message, file=sys.stderr)
        return 130 if cancelled or "no trajectory execution was active" in message.lower() else 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


__all__ = [
    "build_argument_parser",
    "commander_config_from_args",
    "main",
    "pose_target_from_args",
]
