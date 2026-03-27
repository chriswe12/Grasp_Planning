from __future__ import annotations

import unittest

import torch

from grasp_planning.planning.admittance_controller import (
    AdmittanceStepState,
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


if __name__ == "__main__":
    unittest.main()
