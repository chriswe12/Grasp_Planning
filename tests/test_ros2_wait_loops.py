from __future__ import annotations

import unittest
from unittest import mock

from grasp_planning.grasping import ObjectWorldPose
from grasp_planning.ros2 import pose_listener


class _Vector:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _Vector3Stamped:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.vector = _Vector(x, y, z)


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
    def __init__(self, pose_items, *, stamp=None) -> None:
        self.pose_items = tuple(pose_items)
        self.stamp = stamp


class _Stamp:
    def __init__(self, sec: int, nanosec: int) -> None:
        self.sec = sec
        self.nanosec = nanosec


class _Header:
    def __init__(self, stamp: _Stamp) -> None:
        self.stamp = stamp


class _StampedVector3Stamped(_Vector3Stamped):
    def __init__(self, x: float, y: float, z: float, *, sec: int, nanosec: int) -> None:
        super().__init__(x, y, z)
        self.header = _Header(_Stamp(sec, nanosec))


class _FakeNode:
    def __init__(self) -> None:
        self.status_messages: list[str] = []
        self.destroyed = False
        self.latest_pose = None

    def publish_status(self, text: str) -> None:
        self.status_messages.append(text)

    def destroy_node(self) -> None:
        self.destroyed = True


class _VectorNode(_FakeNode):
    pass


class _DebugNode(_FakeNode):
    pass


class _RealFramePairNode(_FakeNode):
    def __init__(self) -> None:
        super().__init__()
        self.latest_pair = None


class Ros2WaitLoopTests(unittest.TestCase):
    def test_vector_listener_callback_updates_latest_pose(self) -> None:
        listener = pose_listener._Vector3StampedListener.__new__(pose_listener._Vector3StampedListener)
        listener._latest_pose = None

        listener._on_vector3_stamped(_Vector3Stamped(0.1, -0.2, 0.3))

        self.assertEqual(
            listener.latest_pose,
            ObjectWorldPose(
                position_world=(0.1, -0.2, 0.3),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            ),
        )

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

    def test_wait_for_centroid_offset_message_returns_after_spin_updates_pose(self) -> None:
        fake_rclpy = mock.Mock()
        fake_rclpy.ok.return_value = False
        node = _VectorNode()

        def spin_once(node_obj, timeout_sec: float) -> None:
            node_obj.latest_pose = ObjectWorldPose(
                position_world=(0.25, -0.05, 0.4),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            )

        fake_rclpy.spin_once.side_effect = spin_once

        with (
            mock.patch.object(pose_listener, "rclpy", fake_rclpy),
            mock.patch.object(pose_listener, "Vector3Stamped", object()),
            mock.patch.object(pose_listener, "_Vector3StampedListener", return_value=node),
        ):
            pose = pose_listener.wait_for_centroid_offset_message(
                topic_name="/perception/fp/mesh_centroid_offset/cooling_screw",
                message_type="geometry_msgs/msg/Vector3Stamped",
                timeout_s=0.5,
            )

        self.assertEqual(pose, node.latest_pose)
        self.assertTrue(node.destroyed)
        fake_rclpy.init.assert_called_once()
        fake_rclpy.shutdown.assert_called_once()

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

    def test_real_frame_pair_listener_pairs_messages_by_matching_stamp(self) -> None:
        listener = pose_listener._RealFramePairListener.__new__(pose_listener._RealFramePairListener)
        listener._object_id = "cooling_screw"
        listener._latest_source_pose = None
        listener._latest_execution_pose = None
        listener._latest_source_stamp = None
        listener._latest_execution_stamp = None
        listener._source_by_stamp = {}
        listener._execution_by_stamp = {}

        listener._on_vector3_stamped(_StampedVector3Stamped(0.1, 0.2, 0.3, sec=11, nanosec=7))
        self.assertIsNone(listener.latest_pair)

        listener._on_debug_frame(
            _DebugFrame(
                [
                    _PoseItem(
                        object_id="cooling_screw",
                        score=0.8,
                        pose_base=((0.6, 0.2, 0.15), (0.0, 0.0, 0.0, 1.0)),
                    )
                ],
                stamp=_Stamp(11, 7),
            )
        )

        self.assertIsNotNone(listener.latest_pair)
        assert listener.latest_pair is not None
        self.assertEqual(listener.latest_pair.source_frame_pose_obj_world.position_world, (0.1, 0.2, 0.3))
        self.assertEqual(listener.latest_pair.execution_pose_world.position_world, (0.6, 0.2, 0.15))

    def test_wait_for_real_frame_pair_messages_returns_after_spin_updates_pair(self) -> None:
        fake_rclpy = mock.Mock()
        fake_rclpy.ok.return_value = False
        node = _RealFramePairNode()

        def spin_once(node_obj, timeout_sec: float) -> None:
            node_obj.latest_pair = pose_listener.RealFramePair(
                source_frame_pose_obj_world=ObjectWorldPose(
                    position_world=(0.25, -0.05, 0.4),
                    orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
                ),
                execution_pose_world=ObjectWorldPose(
                    position_world=(0.6, 0.2, 0.15),
                    orientation_xyzw_world=(0.0, 0.0, 0.70710678, 0.70710678),
                ),
            )

        fake_rclpy.spin_once.side_effect = spin_once

        with (
            mock.patch.object(pose_listener, "rclpy", fake_rclpy),
            mock.patch.object(pose_listener, "Vector3Stamped", object()),
            mock.patch.object(pose_listener, "DebugFrame", object()),
            mock.patch.object(pose_listener, "_RealFramePairListener", return_value=node),
        ):
            pair = pose_listener.wait_for_real_frame_pair_messages(
                source_topic_name="/perception/fp/mesh_centroid_offset/cooling_screw",
                source_message_type="geometry_msgs/msg/Vector3Stamped",
                execution_topic_name="/perception/fp/debug_frame/zed2i_2",
                execution_message_type="fp_debug_msgs/msg/DebugFrame",
                object_id="cooling_screw",
                timeout_s=0.5,
            )

        self.assertEqual(pair, node.latest_pair)
        self.assertTrue(node.destroyed)
        fake_rclpy.init.assert_called_once()
        fake_rclpy.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
