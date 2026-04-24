from __future__ import annotations

from pathlib import Path

from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate
from grasp_planning.pipeline import RealExecutionConfig
from grasp_planning.ros2 import real_grasp_executor


class _FakeCommander:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def move_to_pose(self, target, *, label: str, execute: bool) -> tuple[bool, str]:
        self.calls.append((label, target.frame_id))
        return True, f"{label} ok"


class _FakeGripper:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def open(self, *, width: float) -> tuple[bool, str]:
        self.calls.append(("open", width))
        return True, "open ok"

    def close(self, *, width: float) -> tuple[bool, str]:
        self.calls.append(("close", width))
        return True, "close ok"


def _world_grasp() -> WorldFrameGraspCandidate:
    return WorldFrameGraspCandidate(
        grasp_id="g0001",
        position_w=(0.4, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        normal_w=(0.0, 0.0, 1.0),
        pregrasp_offset=0.1,
        pregrasp_position_w=(0.4, 0.0, 0.1),
        gripper_width=0.03,
        jaw_width=0.02,
        roll_angle_rad=0.0,
        contact_point_a_w=(0.39, 0.0, 0.2),
        contact_point_b_w=(0.41, 0.0, 0.2),
    )


def test_execute_selected_world_grasp_stops_at_pregrasp() -> None:
    commander = _FakeCommander()
    config = RealExecutionConfig(enabled=True, stop_after="pregrasp", frame_id="base", gripper_enabled=False)

    result, steps = real_grasp_executor._execute_selected_world_grasp(
        commander=commander,
        gripper=None,
        world_grasp=_world_grasp(),
        config=config,
        attempt_artifact_path=Path("artifacts/test_attempt.json"),
    )

    assert result.success is True
    assert result.status == "stopped_at_pregrasp"
    assert result.pregrasp_reached is True
    assert result.grasp_reached is False
    assert commander.calls == [("pregrasp", "base")]
    assert [step["name"] for step in steps] == ["pregrasp"]


def test_execute_selected_world_grasp_runs_full_sequence_with_gripper() -> None:
    commander = _FakeCommander()
    gripper = _FakeGripper()
    config = RealExecutionConfig(enabled=True, stop_after="full", frame_id="base", gripper_enabled=True)

    result, steps = real_grasp_executor._execute_selected_world_grasp(
        commander=commander,
        gripper=gripper,
        world_grasp=_world_grasp(),
        config=config,
        attempt_artifact_path=Path("artifacts/test_attempt.json"),
    )

    assert result.success is True
    assert result.status == "completed"
    assert result.pregrasp_reached is True
    assert result.grasp_reached is True
    assert result.lift_reached is True
    assert commander.calls == [("pregrasp", "base"), ("grasp", "base"), ("lift", "base")]
    assert gripper.calls == [("open", 0.08), ("close", 0.02)]
    assert [step["name"] for step in steps] == ["open_gripper", "pregrasp", "grasp", "close_gripper", "lift"]
