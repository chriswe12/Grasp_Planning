from __future__ import annotations

import unittest
from unittest import mock

from grasp_planning.grasping import ObjectWorldPose
from grasp_planning.ros2 import pose_listener


class _Point:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _Quaternion:
    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class _Pose:
    def __init__(self, position: tuple[float, float, float], orientation: tuple[float, float, float, float]) -> None:
        self.position = _Point(*position)
        self.orientation = _Quaternion(*orientation)


class _PoseStamped:
    def __init__(self, pose: tuple[tuple[float, float, float], tuple[float, float, float, float]]) -> None:
        self.pose = _Pose(*pose)


class _PoseItem:
    def __init__(
        self,
        *,
        assembly_name: str,
        part_id: int,
        score: float,
        pose_base: tuple[tuple[float, float, float], tuple[float, float, float, float]],
    ) -> None:
        self.assembly_name = assembly_name
        self.part_id = part_id
        self.score = score
        self.pose_base = _PoseStamped(pose_base)


class _FakeNode:
    def __init__(self) -> None:
        self.status_messages: list[str] = []
        self.destroyed = False
        self.latest_pose = None

    def publish_status(self, text: str) -> None:
        self.status_messages.append(text)

    def destroy_node(self) -> None:
        self.destroyed = True


class _DebugNode(_FakeNode):
    pass


class Ros2WaitLoopTests(unittest.TestCase):
    def test_debug_pose_item_listener_callback_updates_latest_pose_for_matching_part(self) -> None:
        listener = pose_listener._DebugPoseItemListener.__new__(pose_listener._DebugPoseItemListener)
        listener._assembly_name = "cooling_manifold"
        listener._part_id = 2
        listener._latest_pose = None

        listener._on_debug_pose_item(
            _PoseItem(
                assembly_name="cooling_manifold",
                part_id=2,
                score=0.8,
                pose_base=((0.4, 0.1, 0.2), (0.0, 0.0, 0.0, 1.0)),
            )
        )

        self.assertEqual(
            listener.latest_pose,
            ObjectWorldPose(
                position_world=(0.4, 0.1, 0.2),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            ),
        )

    def test_wait_for_debug_pose_item_message_returns_after_spin_updates_pose(self) -> None:
        fake_rclpy = mock.Mock()
        fake_rclpy.ok.return_value = False
        node = _DebugNode()

        def spin_once(node_obj, timeout_sec: float) -> None:
            node_obj.latest_pose = ObjectWorldPose(
                position_world=(0.6, 0.2, 0.15),
                orientation_xyzw_world=(0.0, 0.0, 0.70710678, 0.70710678),
            )

        fake_rclpy.spin_once.side_effect = spin_once

        with (
            mock.patch.object(pose_listener, "rclpy", fake_rclpy),
            mock.patch.object(pose_listener, "DebugPoseItem", object()),
            mock.patch.object(pose_listener, "_DebugPoseItemListener", return_value=node),
        ):
            pose = pose_listener.wait_for_debug_pose_item_message(
                topic_name="/perception/fp/pose_base/fused/assembly",
                message_type="fp_debug_msgs/msg/DebugPoseItem",
                assembly_name="cooling_manifold",
                part_id=2,
                timeout_s=0.5,
            )

        self.assertEqual(pose, node.latest_pose)
        self.assertTrue(node.destroyed)
        fake_rclpy.init.assert_called_once()
        fake_rclpy.shutdown.assert_called_once()

    def test_wait_for_debug_pose_item_message_requires_assembly_and_part(self) -> None:
        fake_rclpy = mock.Mock()

        with (
            mock.patch.object(pose_listener, "rclpy", fake_rclpy),
            mock.patch.object(pose_listener, "DebugPoseItem", object()),
        ):
            with self.assertRaises(ValueError):
                pose_listener.wait_for_debug_pose_item_message(
                    topic_name="/perception/fp/pose_base/fused/assembly",
                    message_type="fp_debug_msgs/msg/DebugPoseItem",
                    assembly_name="",
                    part_id=2,
                    timeout_s=0.5,
                )
            with self.assertRaises(ValueError):
                pose_listener.wait_for_debug_pose_item_message(
                    topic_name="/perception/fp/pose_base/fused/assembly",
                    message_type="fp_debug_msgs/msg/DebugPoseItem",
                    assembly_name="cooling_manifold",
                    part_id=-1,
                    timeout_s=0.5,
                )


if __name__ == "__main__":
    unittest.main()
