#!/usr/bin/env python3
"""Verify IK and planning for both arms in the running dual-LBR MoveIt stack."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moveit_msgs.srv import GetStateValidity  # noqa: E402

from grasp_planning.ros2.moveit_pose_commander import (  # noqa: E402
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    PoseTarget,
    rclpy,
)

ARM_SETTINGS = (
    ("arm_one", "lbr_one_gripper_tcp", tuple(f"lbr_one_A{index}" for index in range(1, 8))),
    ("arm_two", "lbr_two_gripper_tcp", tuple(f"lbr_two_A{index}" for index in range(1, 8))),
)
KNOWN_CROSS_ARM_COLLISION = (
    -math.pi / 2.0,
    math.radians(25.0),
    0.0,
    math.pi / 2.0,
    0.0,
    0.0,
    0.0,
    math.pi / 2.0,
    math.radians(25.0),
    0.0,
    math.pi / 2.0,
    0.0,
    0.0,
    0.0,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moveit-namespace", default="/lbr_dual_arm")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument(
        "--delta-z",
        type=float,
        default=0.01,
        help="Small Cartesian Z displacement planned from each arm's current TCP pose.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the two small trajectories. Intended for mock mode unless explicitly risk-assessed.",
    )
    return parser.parse_args()


def _verify_cross_arm_collision(commander: MoveItPoseCommander, *, moveit_namespace: str) -> None:
    endpoint = f"{moveit_namespace.rstrip('/')}/check_state_validity"
    client = commander.create_client(GetStateValidity, endpoint)
    if not client.wait_for_service(timeout_sec=commander.config.wait_for_moveit_timeout_s):
        raise RuntimeError(f"MoveIt state-validity service '{endpoint}' is unavailable.")

    request = GetStateValidity.Request()
    request.group_name = "both_arms"
    request.robot_state.is_diff = False
    request.robot_state.joint_state.name = [
        *(f"lbr_one_A{index}" for index in range(1, 8)),
        *(f"lbr_two_A{index}" for index in range(1, 8)),
        "lbr_one_left_finger_joint",
        "lbr_two_left_finger_joint",
    ]
    request.robot_state.joint_state.position = [*KNOWN_CROSS_ARM_COLLISION, 0.0, 0.0]
    future = client.call_async(request)
    rclpy.spin_until_future_complete(commander, future, timeout_sec=5.0)
    if not future.done():
        raise RuntimeError("Cross-arm state-validity request timed out.")
    response = future.result()
    if response is None:
        raise RuntimeError("Cross-arm state-validity response was empty.")
    if response.valid:
        raise RuntimeError("Known overlapping dual-arm state was incorrectly reported collision-free.")

    cross_contacts = [
        (contact.contact_body_1, contact.contact_body_2)
        for contact in response.contacts
        if (str(contact.contact_body_1).startswith("lbr_one_") and str(contact.contact_body_2).startswith("lbr_two_"))
        or (str(contact.contact_body_1).startswith("lbr_two_") and str(contact.contact_body_2).startswith("lbr_one_"))
    ]
    if not cross_contacts:
        raise RuntimeError(
            "The known invalid state did not report a contact between lbr_one and lbr_two; "
            f"contacts={[(contact.contact_body_1, contact.contact_body_2) for contact in response.contacts]}"
        )
    print(f"cross_arm_collision_check: valid={response.valid} contacts={cross_contacts[:4]}")


def main() -> None:
    args = _parse_args()
    if rclpy is None:
        raise RuntimeError("ROS2 dependencies are unavailable. Source ROS2 and both workspaces first.")

    rclpy.init()
    try:
        for index, (planning_group, pose_link, joint_names) in enumerate(ARM_SETTINGS, start=1):
            commander = MoveItPoseCommander(
                MoveItPoseCommanderConfig(
                    planning_group=planning_group,
                    pose_link=pose_link,
                    joint_names=joint_names,
                    moveit_namespace=str(args.moveit_namespace),
                    planning_time_s=5.0,
                    num_planning_attempts=5,
                    velocity_scale=0.05,
                    acceleration_scale=0.05,
                ),
                node_name=f"dual_lbr_smoke_{planning_group}",
            )
            try:
                commander.wait_for_moveit(require_execute=bool(args.execute))
                current = commander.get_current_pose(frame_id=str(args.frame_id))
                target = PoseTarget.from_quaternion(
                    x=current.x,
                    y=current.y,
                    z=current.z + float(args.delta_z),
                    quaternion_xyzw=current.orientation_xyzw,
                    frame_id=str(args.frame_id),
                )
                ok, message = commander.move_to_pose(
                    target,
                    label=f"{planning_group}_smoke",
                    execute=bool(args.execute),
                )
                print(
                    f"{planning_group}: current=({current.x:.4f}, {current.y:.4f}, {current.z:.4f}) "
                    f"target_z={target.z:.4f} ok={ok} message={message}"
                )
                if not ok:
                    raise RuntimeError(f"{planning_group} smoke check failed: {message}")
                if index == 1:
                    _verify_cross_arm_collision(
                        commander,
                        moveit_namespace=str(args.moveit_namespace),
                    )
            finally:
                commander.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
