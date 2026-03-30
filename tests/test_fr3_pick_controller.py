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
        self.last_joint_targets = []
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
        self.last_joint_targets.append((_args, _kwargs))
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
        expected_quat_wxyz = controller._phase_start_quat_w
        self.assertAlmostEqual(pos[0], 0.5, places=4)
        self.assertAlmostEqual(pos[1], 0.0, places=4)
        self.assertAlmostEqual(pos[2], -0.0225, places=4)
        self.assertEqual(len(quat_xyzw), 4)
        self.assertAlmostEqual(sum(component * component for component in quat_xyzw), 1.0, places=4)
        self.assertFalse(torch.allclose(expected_quat_wxyz, torch.tensor([[1.0, 0.0, 0.0, 0.0]])))

    def test_limit_joint_delta_per_step(self) -> None:
        current_joint_pos = torch.zeros((1, 7), dtype=torch.float32)
        desired_joint_pos = torch.full((1, 7), 0.5, dtype=torch.float32)

        limited = FR3PickController._limit_joint_delta(
            current_joint_pos,
            desired_joint_pos,
            max_joint_delta_rad=0.04,
        )

        torch.testing.assert_close(limited, torch.full((1, 7), 0.04, dtype=torch.float32))

    def test_compute_line_tracking_target_projects_lateral_error(self) -> None:
        desired_line_point, lateral_error = FR3PickController._compute_line_tracking_target(
            current_position_w=(0.12, 0.03, 0.0),
            start_position_w=(0.0, 0.0, 0.0),
            target_position_w=(0.2, 0.0, 0.0),
            tau=0.5,
        )

        self.assertAlmostEqual(desired_line_point[0], 0.1, places=4)
        self.assertAlmostEqual(desired_line_point[1], 0.0, places=4)
        self.assertAlmostEqual(desired_line_point[2], 0.0, places=4)
        self.assertAlmostEqual(lateral_error[0], 0.0, places=4)
        self.assertAlmostEqual(lateral_error[1], 0.03, places=4)
        self.assertAlmostEqual(lateral_error[2], 0.0, places=4)

    def test_start_phase_approach_initializes_distinct_tcp_targets(self) -> None:
        fake_robot = _FakeRobot()
        fake_robot.data.body_pose_w[0, 12, :3] = torch.tensor([-0.43, 0.0, 0.17], dtype=torch.float32)
        fake_robot.data.body_pose_w[0, 12, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        grasp = GraspCandidate(
            position_w=(-0.45, 0.0, 0.025),
            orientation_xyzw=(0.0, 0.70710678, 0.0, 0.70710678),
            normal_w=(0.0, 0.0, -1.0),
            pregrasp_offset=0.2,
            gripper_width=0.06,
            score=1.0,
            label="+z",
        )

        with mock.patch.object(FR3PickController, "_build_ik_controller", return_value=object()):
            controller = FR3PickController(robot=fake_robot, grasp=grasp, physics_dt=0.01, start_phase="approach")

        self.assertAlmostEqual(controller._pregrasp_tcp_position_w[0], -0.43, places=4)
        self.assertAlmostEqual(controller._pregrasp_tcp_position_w[1], 0.0, places=4)
        self.assertAlmostEqual(controller._pregrasp_tcp_position_w[2], 0.17, places=4)
        self.assertAlmostEqual(controller._grasp_tcp_position_w[0], -0.43, places=4)
        self.assertAlmostEqual(controller._grasp_tcp_position_w[1], 0.0, places=4)
        self.assertAlmostEqual(controller._grasp_tcp_position_w[2], 0.025, places=4)


if __name__ == "__main__":
    unittest.main()
