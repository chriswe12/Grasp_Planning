from __future__ import annotations

import math
import unittest

import torch

from grasp_planning.planning.fr3_motion_context import (
    FR3MotionContext,
    grasp_pose_to_tcp_pose,
    tcp_pose_to_grasp_pose,
)


class _FakeRobotData:
    def __init__(self) -> None:
        self.joint_pos_limits = torch.tensor(
            [[[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0], [-4.0, 4.0], [-5.0, 5.0], [-6.0, 6.0], [-7.0, 7.0]]],
            dtype=torch.float32,
        )


class _FakeRobot:
    def __init__(self) -> None:
        self.data = _FakeRobotData()
        self.device = "cpu"


class FR3MotionContextTests(unittest.TestCase):
    def test_get_joint_limits_prefers_joint_pos_limits(self) -> None:
        context = FR3MotionContext.__new__(FR3MotionContext)
        context.robot = _FakeRobot()
        context.arm_joint_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)

        lower, upper = context.get_joint_limits()

        torch.testing.assert_close(lower, torch.tensor([[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]]))
        torch.testing.assert_close(upper, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]))

    def test_joint_state_within_limits_allows_small_tolerance(self) -> None:
        context = FR3MotionContext.__new__(FR3MotionContext)
        context.robot = _FakeRobot()
        context.arm_joint_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)

        q = torch.tensor([[1.0005, 2.0005, 3.0005, 4.0005, 5.0005, 6.0005, 7.0005]], dtype=torch.float32)

        self.assertTrue(context.joint_state_within_limits(q, tolerance=1.0e-3))
        self.assertFalse(context.joint_state_within_limits(q, tolerance=1.0e-5))

    def test_tcp_pose_to_grasp_pose_removes_fixed_tcp_transform(self) -> None:
        grasp_position_w, grasp_orientation_xyzw = tcp_pose_to_grasp_pose(
            position_w=(0.45, 0.0, 0.35),
            orientation_xyzw=(0.0, 1.0, 0.0, 0.0),
            grasp_to_tcp_quat_wxyz=FR3MotionContext._GRASP_TO_TCP_QUAT_WXYZ,
            tcp_to_grasp_center_offset=FR3MotionContext._TCP_TO_GRASP_CENTER_OFFSET,
        )

        self.assertAlmostEqual(grasp_position_w[0], 0.45, places=6)
        self.assertAlmostEqual(grasp_position_w[1], 0.0, places=6)
        self.assertAlmostEqual(grasp_position_w[2], 0.395, places=6)
        self.assertAlmostEqual(grasp_orientation_xyzw[0], 0.0, places=6)
        self.assertAlmostEqual(abs(grasp_orientation_xyzw[1]), math.sqrt(0.5), places=6)
        self.assertAlmostEqual(grasp_orientation_xyzw[2], 0.0, places=6)
        self.assertAlmostEqual(abs(grasp_orientation_xyzw[3]), math.sqrt(0.5), places=6)

    def test_grasp_and_tcp_pose_conversion_roundtrip(self) -> None:
        tcp_position_w = (0.12, -0.04, 0.81)
        tcp_orientation_xyzw = (0.2, 0.3, -0.1, 0.9273618495495703)

        grasp_position_w, grasp_orientation_xyzw = tcp_pose_to_grasp_pose(
            position_w=tcp_position_w,
            orientation_xyzw=tcp_orientation_xyzw,
            grasp_to_tcp_quat_wxyz=FR3MotionContext._GRASP_TO_TCP_QUAT_WXYZ,
            tcp_to_grasp_center_offset=FR3MotionContext._TCP_TO_GRASP_CENTER_OFFSET,
        )
        roundtrip_position_w, roundtrip_orientation_xyzw = grasp_pose_to_tcp_pose(
            position_w=grasp_position_w,
            orientation_xyzw=grasp_orientation_xyzw,
            grasp_to_tcp_quat_wxyz=FR3MotionContext._GRASP_TO_TCP_QUAT_WXYZ,
            tcp_to_grasp_center_offset=FR3MotionContext._TCP_TO_GRASP_CENTER_OFFSET,
        )

        for actual, expected in zip(roundtrip_position_w, tcp_position_w):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(roundtrip_orientation_xyzw, tcp_orientation_xyzw):
            self.assertAlmostEqual(actual, expected, places=6)


if __name__ == "__main__":
    unittest.main()
