from __future__ import annotations

import torch

from grasp_planning.planning.trajectory_executor import TrajectoryExecutor
from grasp_planning.planning.types import JointTrajectory


class _FakeSim:
    def step(self) -> None:
        pass


class _FakeScene:
    def write_data_to_sim(self) -> None:
        pass

    def update(self, _dt: float) -> None:
        pass


class _TrackingContext:
    physics_dt = 0.01
    arm_joint_names = ("joint1",)

    def __init__(self, *, velocity_decay: float = 1.0) -> None:
        self.sim = _FakeSim()
        self.scene = _FakeScene()
        self.q = torch.zeros((1, 1), dtype=torch.float32)
        self.qd = torch.zeros_like(self.q)
        self.velocity_decay = float(velocity_decay)
        self.position_commands: list[float] = []
        self.velocity_commands: list[float] = []

    def get_arm_q(self) -> torch.Tensor:
        return self.q.clone()

    def get_arm_qd(self) -> torch.Tensor:
        return self.qd.clone()

    def command_arm(self, q: torch.Tensor) -> None:
        self.q = q.clone()
        self.position_commands.append(float(q.item()))

    def command_arm_velocity(self, qd: torch.Tensor) -> None:
        self.qd = qd.clone() * self.velocity_decay
        self.velocity_commands.append(float(qd.item()))

    def command_fixed_gripper(self) -> None:
        pass


def test_continuous_path_respects_speed_limit_without_stopping_at_internal_point() -> None:
    context = _TrackingContext()
    executor = TrajectoryExecutor(
        context,
        max_joint_speed_rad_s=0.5,
        final_settle_steps=2,
    )

    ok, detail = executor.execute(
        JointTrajectory(
            waypoints=[
                torch.tensor([[0.5]], dtype=torch.float32),
                torch.tensor([[1.0]], dtype=torch.float32),
            ],
            dt=context.physics_dt,
        )
    )

    assert ok, detail
    assert max(abs(value) for value in context.velocity_commands) <= 0.5 + 1.0e-6
    moving_commands = context.velocity_commands[:-1]
    middle = min(
        range(len(context.position_commands) - 1),
        key=lambda index: abs(context.position_commands[index] - 0.5),
    )
    assert abs(moving_commands[middle]) > 0.1
    assert context.velocity_commands[-1] == 0.0
    assert context.position_commands[-1] == 1.0


def test_settle_requires_low_velocity_as_well_as_position_error() -> None:
    context = _TrackingContext()
    executor = TrajectoryExecutor(
        context,
        velocity_tolerance_rad_s=0.05,
        max_joint_speed_rad_s=0.5,
        final_settle_steps=2,
    )
    original_command_velocity = context.command_arm_velocity

    def keep_moving(qd: torch.Tensor) -> None:
        original_command_velocity(qd)
        if torch.count_nonzero(qd) == 0:
            context.qd.fill_(0.2)

    context.command_arm_velocity = keep_moving  # type: ignore[method-assign]

    ok, detail = executor.execute(
        JointTrajectory(
            waypoints=[torch.tensor([[0.5]], dtype=torch.float32)],
            dt=context.physics_dt,
        )
    )

    assert not ok
    assert "last_max_joint_speed=0.2000" in detail


def test_large_move_is_not_silently_compressed_past_speed_limit() -> None:
    context = _TrackingContext()
    executor = TrajectoryExecutor(
        context,
        max_joint_speed_rad_s=0.5,
        final_settle_steps=2,
    )

    ok, detail = executor.execute(
        JointTrajectory(
            waypoints=[torch.tensor([[1.0]], dtype=torch.float32)],
            dt=context.physics_dt,
        )
    )

    assert ok, detail
    assert len(context.velocity_commands) > 300
    assert max(abs(value) for value in context.velocity_commands) <= 0.5 + 1.0e-6
