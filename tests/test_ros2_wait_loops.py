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


class _PoseItem:
    def __init__(
        self,
        *,
        object_id: str,
        score: float,
        pose_base: tuple[tuple[float, float, float], tuple[float, float, float, float]],
    ) -> None:
        self.object_id = object_id
        self.score = score
        self.pose_base = _Pose(*pose_base)


class _DebugFrame:
    def __init__(self, pose_items) -> None:
        self.pose_items = tuple(pose_items)


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
    def test_debug_frame_listener_callback_updates_latest_pose_for_matching_object(self) -> None:
        listener = pose_listener._DebugFramePoseListener.__new__(pose_listener._DebugFramePoseListener)
        listener._object_id = "cooling_screw"
        listener._latest_pose = None

        listener._on_debug_frame(
            _DebugFrame(
                [
                    _PoseItem(
                        object_id="pb_top",
                        score=1.0,
                        pose_base=((9.0, 9.0, 9.0), (0.0, 0.0, 0.0, 1.0)),
                    ),
                    _PoseItem(
                        object_id="cooling_screw",
                        score=0.8,
                        pose_base=((0.4, 0.1, 0.2), (0.0, 0.0, 0.0, 1.0)),
                    ),
                ]
            )
        )

        self.assertEqual(
            listener.latest_pose,
            ObjectWorldPose(
                position_world=(0.4, 0.1, 0.2),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            ),
        )

    def test_wait_for_debug_frame_pose_message_returns_after_spin_updates_pose(self) -> None:
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
            mock.patch.object(pose_listener, "DebugFrame", object()),
            mock.patch.object(pose_listener, "_DebugFramePoseListener", return_value=node),
        ):
            pose = pose_listener.wait_for_debug_frame_pose_message(
                topic_name="/perception/fp/debug_frame/zed2i_2",
                message_type="fp_debug_msgs/msg/DebugFrame",
                object_id="cooling_screw",
                timeout_s=0.5,
            )

        self.assertEqual(pose, node.latest_pose)
        self.assertTrue(node.destroyed)
        fake_rclpy.init.assert_called_once()
        fake_rclpy.shutdown.assert_called_once()

    def test_wait_for_debug_frame_pose_message_requires_object_id(self) -> None:
        fake_rclpy = mock.Mock()

        with (
            mock.patch.object(pose_listener, "rclpy", fake_rclpy),
            mock.patch.object(pose_listener, "DebugFrame", object()),
        ):
            with self.assertRaises(ValueError):
                pose_listener.wait_for_debug_frame_pose_message(
                    topic_name="/perception/fp/debug_frame/zed2i_2",
                    message_type="fp_debug_msgs/msg/DebugFrame",
                    object_id="",
                    timeout_s=0.5,
                )


if __name__ == "__main__":
    unittest.main()
