from __future__ import annotations

import unittest

import torch

from grasp_planning.planning.admittance_controller import (
    AdmittanceStepState,
    AdmittanceControllerCfg,
    FR3AdmittanceController,
    integrate_admittance_step,
)


class _FakeRobotData:
    def __init__(self) -> None:
        self.body_net_forces_w = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, -2.0, 3.5]]],
            dtype=torch.float32,
        )
        self.body_net_torques_w = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.4, -0.5, 0.6]]],
            dtype=torch.float32,
        )


class _FakeRobot:
    def __init__(self) -> None:
        self.device = "cpu"
        self.data = _FakeRobotData()


class _FakeContext:
    def __init__(self) -> None:
        self.device = "cpu"
        self.ee_body_idx = 1
        self.robot = _FakeRobot()


class AdmittanceControllerMathTests(unittest.TestCase):
    def test_virtual_state_moves_toward_target_without_external_force(self) -> None:
        next_state = integrate_admittance_step(
            state=AdmittanceStepState(
                position_w=(0.0, 0.0, 0.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                twist=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            target_position_w=(0.1, 0.0, 0.0),
            target_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            external_wrench=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            stiffness_diag=(250.0, 250.0, 250.0, 10.0, 10.0, 8.0),
            inertia_diag=(4.0, 2.0, 3.0, 0.075, 0.2, 0.001),
            dt=0.01,
        )

        self.assertGreater(next_state.position_w[0], 0.0)
        self.assertGreater(next_state.twist[0], 0.0)

    def test_external_wrench_pushes_virtual_state(self) -> None:
        next_state = integrate_admittance_step(
            state=AdmittanceStepState(
                position_w=(0.0, 0.0, 0.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                twist=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            target_position_w=(0.0, 0.0, 0.0),
            target_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            external_wrench=(4.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            stiffness_diag=(250.0, 250.0, 250.0, 10.0, 10.0, 8.0),
            inertia_diag=(4.0, 2.0, 3.0, 0.075, 0.2, 0.001),
            dt=0.01,
        )

        self.assertGreater(next_state.position_w[0], 0.0)
        self.assertGreater(next_state.twist[0], 0.0)

    def test_extracts_end_effector_wrench_from_robot_data(self) -> None:
        controller = FR3AdmittanceController.__new__(FR3AdmittanceController)
        controller._context = _FakeContext()

        wrench = controller._estimate_external_wrench()

        expected = torch.tensor([[1.0, -2.0, 3.5, 0.4, -0.5, 0.6]], dtype=torch.float32)
        torch.testing.assert_close(wrench.cpu(), expected)

    def test_missing_wrench_channels_fall_back_to_zero(self) -> None:
        controller = FR3AdmittanceController.__new__(FR3AdmittanceController)
        controller._context = _FakeContext()
        controller._context.robot.data = object()

        wrench = controller._estimate_external_wrench()

        torch.testing.assert_close(wrench.cpu(), torch.zeros((1, 6), dtype=torch.float32))

    def test_step_uses_zero_wrench_by_default(self) -> None:
        controller = FR3AdmittanceController.__new__(FR3AdmittanceController)
        controller._context = _FakeContext()
        controller._cfg = AdmittanceControllerCfg()
        controller._filtered_wrench = torch.zeros((1, 6), dtype=torch.float32)
        controller._twist = torch.zeros((1, 6), dtype=torch.float32)
        controller._reference_position = torch.zeros((1, 3), dtype=torch.float32)
        controller._reference_orientation = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        controller._virtual_position = controller._reference_position.clone()
        controller._virtual_orientation = controller._reference_orientation.clone()
        controller._target_position_w = (0.0, 0.0, 0.0)
        controller._target_orientation_xyzw = (0.0, 0.0, 0.0, 1.0)
        controller._stiffness = torch.eye(6, dtype=torch.float32)
        controller._inertia = torch.eye(6, dtype=torch.float32)
        controller._damping = torch.eye(6, dtype=torch.float32)
        controller._ik_controller = unittest.mock.Mock()
        controller._context.physics_dt = 0.01
        controller._context.command_pose_via_differential_ik = unittest.mock.Mock(return_value=torch.zeros((1, 7)))
        controller._context.get_joint_limits = unittest.mock.Mock(
            return_value=(torch.full((1, 7), -1.0), torch.full((1, 7), 1.0))
        )
        controller._context.joint_limits_are_usable = unittest.mock.Mock(return_value=True)
        controller._context.hold_position = unittest.mock.Mock()
        controller._estimate_external_wrench = unittest.mock.Mock(
            return_value=torch.tensor([[1.0, -2.0, 3.5, 0.4, -0.5, 0.6]], dtype=torch.float32)
        )

        controller.step()

        torch.testing.assert_close(controller._filtered_wrench, torch.zeros((1, 6), dtype=torch.float32))
        controller._context.hold_position.assert_called_once()
        _q_des, = controller._context.hold_position.call_args.args
        self.assertEqual(controller._context.hold_position.call_args.kwargs["steps"], controller._cfg.inner_settle_steps)


if __name__ == "__main__":
    unittest.main()
