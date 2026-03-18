from __future__ import annotations

import unittest
from unittest import mock

import torch

from grasp_planning.controllers.fr3_pick_controller import FR3PickController, quat_xyzw_to_wxyz
from grasp_planning.grasping import GraspCandidate


class _FakeRootPhysxView:
    def get_jacobians(self) -> torch.Tensor:
        return torch.zeros((1, 12, 6, 9), dtype=torch.float32)


class _FakeRobotData:
    def __init__(self) -> None:
        self.joint_pos = torch.zeros((1, 9), dtype=torch.float32)
        self.body_state_w = torch.zeros((1, 13, 13), dtype=torch.float32)
        self.body_pose_w = torch.zeros((1, 13, 7), dtype=torch.float32)
        self.root_pose_w = torch.zeros((1, 7), dtype=torch.float32)
        self.body_pose_w[0, 12, 3] = 1.0


class _FakeRobot:
    def __init__(self) -> None:
        self.device = "cpu"
        self.is_fixed_base = True
        self.joint_names = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
            "fr3_finger_joint1",
            "fr3_finger_joint2",
        ]
        self.body_names = [
            "fr3_link0",
            "fr3_link1",
            "fr3_link2",
            "fr3_link3",
            "fr3_link4",
            "fr3_link5",
            "fr3_link6",
            "fr3_link7",
            "fr3_link8",
            "fr3_hand",
            "fr3_leftfinger",
            "fr3_rightfinger",
            "fr3_hand_tcp",
        ]
        self.data = _FakeRobotData()
        self.root_physx_view = _FakeRootPhysxView()

    def set_joint_position_target(self, *_args, **_kwargs) -> None:
        return None


class FR3PickControllerTests(unittest.TestCase):
    def test_quat_xyzw_to_wxyz(self) -> None:
        quat = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
        converted = quat_xyzw_to_wxyz(quat)
        expected = torch.tensor([[0.4, 0.1, 0.2, 0.3]], dtype=torch.float32)
        torch.testing.assert_close(converted, expected)

    def test_resolves_validated_fr3_schema(self) -> None:
        fake_robot = _FakeRobot()
        grasp = GraspCandidate(
            position_w=(0.45, 0.0, 0.025),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            normal_w=(1.0, 0.0, 0.0),
            pregrasp_offset=0.1,
            gripper_width=0.06,
            score=1.0,
            label="+x",
        )

        with mock.patch.object(FR3PickController, "_build_ik_controller", return_value=object()):
            controller = FR3PickController(robot=fake_robot, grasp=grasp, physics_dt=0.01)

        self.assertEqual(controller.ee_body_name, "fr3_hand_tcp")
        self.assertEqual(controller.arm_joint_names, tuple(fake_robot.joint_names[:7]))
        self.assertEqual(controller.hand_joint_names, tuple(fake_robot.joint_names[7:]))
        self.assertEqual(controller._ee_jacobi_body_idx, 11)
        self.assertEqual(controller._arm_joint_ids.tolist(), list(range(7)))
        self.assertEqual(controller._hand_joint_ids.tolist(), [7, 8])

    def test_interpolate_phase_pose_uses_geodesic_orientation(self) -> None:
        fake_robot = _FakeRobot()
        grasp = GraspCandidate(
            position_w=(1.0, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 1.0, 0.0),
            normal_w=(1.0, 0.0, 0.0),
            pregrasp_offset=0.1,
            gripper_width=0.06,
            score=1.0,
            label="+x",
        )
        with mock.patch.object(FR3PickController, "_build_ik_controller", return_value=object()):
            controller = FR3PickController(robot=fake_robot, grasp=grasp, physics_dt=0.01)
        controller._phase_elapsed_s = controller._durations.pregrasp_s / 2.0
        pos, quat_xyzw = controller._interpolate_phase_pose(
            target_position_w=(1.0, 0.0, 0.0),
            target_orientation_xyzw=(0.0, 0.0, 1.0, 0.0),
            duration_s=controller._durations.pregrasp_s,
        )
        self.assertAlmostEqual(pos[0], 0.5, places=4)
        self.assertAlmostEqual(pos[1], 0.0, places=4)
        self.assertAlmostEqual(pos[2], 0.0, places=4)
        self.assertAlmostEqual(quat_xyzw[2], 0.7071067, places=4)
        self.assertAlmostEqual(quat_xyzw[3], 0.7071067, places=4)


if __name__ == "__main__":
    unittest.main()
