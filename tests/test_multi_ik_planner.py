from __future__ import annotations

from types import SimpleNamespace

import pytest

from grasp_planning.ros2.moveit_pose_commander import PoseTarget
from grasp_planning.ros2.multi_ik_planner import MultiIkPlanningConfig, plan_pose_sequence_multi_ik


class _Commander:
    def __init__(self) -> None:
        self.ik_calls: list[tuple[float, ...]] = []
        self.plan_calls: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    def compute_ik(self, target, *, seed_joint_positions):
        seed = tuple(float(value) for value in seed_joint_positions)
        self.ik_calls.append(seed)
        # Create two stable IK branches based on the first seed joint.
        return ([0.8, 0.05] if seed[0] >= 0.0 else [-0.3, 0.3]), "ok"

    def plan_to_joint_positions(self, joints, *, label, start_joint_positions):
        goal = tuple(float(value) for value in joints)
        start = tuple(float(value) for value in start_joint_positions)
        self.plan_calls.append((start, goal))
        midpoint = tuple(0.5 * (lhs + rhs) for lhs, rhs in zip(start, goal))
        trajectory = SimpleNamespace(
            joint_trajectory=SimpleNamespace(
                joint_names=["A1", "A2"],
                points=[
                    SimpleNamespace(positions=midpoint),
                    SimpleNamespace(positions=goal),
                ],
            )
        )
        return trajectory, "ok"


class _BoundedCommander(_Commander):
    def compute_ik(self, target, *, seed_joint_positions):
        seed = tuple(float(value) for value in seed_joint_positions)
        self.ik_calls.append(seed)
        return ([0.0] if seed[0] >= 0.0 else [-3.0]), "ok"

    def plan_to_joint_positions(self, joints, *, label, start_joint_positions):
        del label
        goal = tuple(float(value) for value in joints)
        start = tuple(float(value) for value in start_joint_positions)
        self.plan_calls.append((start, goal))
        return (
            SimpleNamespace(
                joint_trajectory=SimpleNamespace(
                    joint_names=["A1"],
                    points=[SimpleNamespace(positions=goal)],
                )
            ),
            "ok",
        )


def _target() -> PoseTarget:
    return PoseTarget.from_quaternion(
        x=0.4,
        y=0.2,
        z=0.5,
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="base",
    )


def test_multi_ik_search_deduplicates_solutions_and_selects_cheapest_path() -> None:
    commander = _Commander()
    result = plan_pose_sequence_multi_ik(
        commander,
        targets={"pregrasp": _target()},
        labels=("pregrasp",),
        start_joint_positions=(0.0, 0.0),
        joint_names=("A1", "A2"),
        config=MultiIkPlanningConfig(
            candidate_count=8,
            beam_width=2,
            seed_perturbation_rad=1.0,
            dedup_tolerance_rad=0.05,
        ),
        label_prefix="test",
    )

    assert len(commander.ik_calls) == 8
    assert len(commander.plan_calls) == 2
    assert result.terminal_joint_positions == pytest.approx((-0.3, 0.3))
    assert result.diagnostics[0]["distinct_ik_solution_count"] == 2


def test_joint_weights_change_which_ik_branch_is_selected() -> None:
    commander = _Commander()
    result = plan_pose_sequence_multi_ik(
        commander,
        targets={"target": _target()},
        labels=("target",),
        start_joint_positions=(0.0, 0.0),
        joint_names=("A1", "A2"),
        config=MultiIkPlanningConfig(
            candidate_count=8,
            beam_width=1,
            seed_perturbation_rad=1.0,
            joint_weights=(0.1, 10.0),
        ),
        label_prefix="weighted",
    )

    assert result.terminal_joint_positions == pytest.approx((0.8, 0.05))


def test_multi_segment_search_uses_parent_terminal_as_next_start() -> None:
    commander = _Commander()
    result = plan_pose_sequence_multi_ik(
        commander,
        targets={"pregrasp": _target(), "grasp": _target()},
        labels=("pregrasp", "grasp"),
        start_joint_positions=(0.0, 0.0),
        joint_names=("A1", "A2"),
        config=MultiIkPlanningConfig(candidate_count=4, beam_width=2, seed_perturbation_rad=1.0),
        label_prefix="sequence",
    )

    assert tuple(result.trajectories) == ("pregrasp", "grasp")
    assert any(start != (0.0, 0.0) for start, _goal in commander.plan_calls[2:])


def test_joint_weight_count_must_match_robot() -> None:
    with pytest.raises(ValueError, match="Expected 2 multi-IK joint weights"):
        plan_pose_sequence_multi_ik(
            _Commander(),
            targets={"target": _target()},
            labels=("target",),
            start_joint_positions=(0.0, 0.0),
            joint_names=("A1", "A2"),
            config=MultiIkPlanningConfig(candidate_count=2, joint_weights=(1.0,)),
            label_prefix="bad",
        )


def test_explicit_half_turn_seed_is_filtered_by_bounded_joint_limits() -> None:
    commander = _Commander()
    plan_pose_sequence_multi_ik(
        commander,
        targets={"target": _target()},
        labels=("target",),
        start_joint_positions=(0.0, 1.0),
        joint_names=("A1", "A2"),
        config=MultiIkPlanningConfig(
            candidate_count=2,
            seed_offsets_rad=((0.0, 3.141592653589793), (0.0, -3.141592653589793)),
            joint_lower_limits_rad=(-2.97, -3.05),
            joint_upper_limits_rad=(2.97, 3.05),
        ),
        label_prefix="bounded_seed",
    )

    assert commander.ik_calls == pytest.approx(
        [
            (0.0, 1.0),
            (0.0, 1.0 - 3.141592653589793),
        ]
    )


def test_bounded_joint_cost_does_not_wrap_across_position_limits() -> None:
    commander = _BoundedCommander()
    result = plan_pose_sequence_multi_ik(
        commander,
        targets={"target": _target()},
        labels=("target",),
        start_joint_positions=(3.0,),
        joint_names=("A1",),
        config=MultiIkPlanningConfig(
            candidate_count=2,
            seed_offsets_rad=((-6.0,),),
            joint_lower_limits_rad=(-3.05,),
            joint_upper_limits_rad=(3.05,),
            continuous_joints=(False,),
        ),
        label_prefix="bounded_cost",
    )

    assert result.terminal_joint_positions == pytest.approx((0.0,))


def test_unspecified_joint_topology_preserves_legacy_wrapped_cost() -> None:
    commander = _BoundedCommander()
    result = plan_pose_sequence_multi_ik(
        commander,
        targets={"target": _target()},
        labels=("target",),
        start_joint_positions=(3.0,),
        joint_names=("A1",),
        config=MultiIkPlanningConfig(
            candidate_count=2,
            seed_offsets_rad=((-6.0,),),
        ),
        label_prefix="legacy_wrapped_cost",
    )

    assert result.terminal_joint_positions == pytest.approx((-3.0,))
