from __future__ import annotations

import math

from grasp_planning.pipeline.cartesian_waypoint_ik import (
    WaypointChainIkResult,
    WaypointIkResult,
    interpolate_pose_targets,
    resolve_ik,
    slerp_quaternion_xyzw,
    solve_cartesian_waypoint_chain,
)
from grasp_planning.ros2.moveit_pose_commander import PoseTarget


def _pose(x: float, y: float, z: float, quaternion_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id: str = "base_link") -> PoseTarget:
    return PoseTarget.from_quaternion(x=x, y=y, z=z, quaternion_xyzw=quaternion_xyzw, frame_id=frame_id)


def test_slerp_quaternion_is_normalized_and_returns_endpoints_at_bounds() -> None:
    start = (0.0, 0.0, 0.0, 1.0)
    end = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)  # 90 deg about Z

    at_start = slerp_quaternion_xyzw(start, end, 0.0)
    at_end = slerp_quaternion_xyzw(start, end, 1.0)
    midpoint = slerp_quaternion_xyzw(start, end, 0.5)

    for quaternion in (at_start, at_end, midpoint):
        norm = math.sqrt(sum(component**2 for component in quaternion))
        assert math.isclose(norm, 1.0, abs_tol=1e-9)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(at_start, start))
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(at_end, end))
    # 45 degrees about Z at the midpoint.
    assert math.isclose(midpoint[2], math.sin(math.pi / 8), abs_tol=1e-9)
    assert math.isclose(midpoint[3], math.cos(math.pi / 8), abs_tol=1e-9)


def test_slerp_quaternion_takes_the_shortest_arc() -> None:
    start = (0.0, 0.0, 0.0, 1.0)
    end = (0.0, 0.0, 0.0, -1.0)  # same rotation as start, opposite sign

    midpoint = slerp_quaternion_xyzw(start, end, 0.5)

    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(midpoint, start))


def test_interpolate_pose_targets_spaces_positions_linearly() -> None:
    start = _pose(0.0, 0.0, 0.0)
    end = _pose(1.0, 2.0, -1.0)

    waypoints = interpolate_pose_targets(start, end, num_waypoints=10)

    assert len(waypoints) == 10
    assert waypoints[4].position_xyz == (0.5, 1.0, -0.5)
    last = waypoints[-1]
    assert math.isclose(last.x, end.x, abs_tol=1e-9)
    assert math.isclose(last.y, end.y, abs_tol=1e-9)
    assert math.isclose(last.z, end.z, abs_tol=1e-9)
    assert last.orientation_xyzw == end.orientation_xyzw


def test_interpolate_pose_targets_rejects_mismatched_frames() -> None:
    start = _pose(0.0, 0.0, 0.0, frame_id="lbr_one_link_0")
    end = _pose(1.0, 0.0, 0.0, frame_id="base_link")

    try:
        interpolate_pose_targets(start, end, num_waypoints=5)
    except ValueError as exc:
        assert "frame_id" in str(exc)
    else:
        raise AssertionError("Expected mismatched frame_id to fail.")


def test_interpolate_pose_targets_rejects_non_positive_waypoint_count() -> None:
    start = _pose(0.0, 0.0, 0.0)
    end = _pose(1.0, 0.0, 0.0)

    try:
        interpolate_pose_targets(start, end, num_waypoints=0)
    except ValueError as exc:
        assert "num_waypoints" in str(exc)
    else:
        raise AssertionError("Expected num_waypoints=0 to fail.")


class _SeedChainingCommander:
    """Fake solver whose 'solution' encodes the seed it was given.

    Each successful solve bumps joint 0 by +0.1 relative to its seed (or
    from an all-zero seed when none is given), so a test can prove that
    waypoint N really was seeded with waypoint N-1's own result rather than
    the original seed or a fixed one.
    """

    def __init__(self, *, fail_at_calls: frozenset[int] = frozenset()) -> None:
        self.fail_at_calls = fail_at_calls
        self.calls: list[tuple[PoseTarget, tuple[float, ...] | None]] = []

    def compute_ik(self, target, *, seed_joint_positions=None):
        call_number = len(self.calls) + 1
        self.calls.append((target, tuple(seed_joint_positions) if seed_joint_positions is not None else None))
        if call_number in self.fail_at_calls:
            return None, f"synthetic failure at call {call_number}"
        base = list(seed_joint_positions) if seed_joint_positions is not None else [0.0] * 7
        base[0] = round(base[0] + 0.1, 9)
        return base, "ok"


def test_solve_cartesian_waypoint_chain_seeds_each_step_from_the_previous_solution() -> None:
    commander = _SeedChainingCommander()
    start = _pose(0.0, 0.0, 0.0)
    end = _pose(1.0, 0.0, 0.0)

    result = solve_cartesian_waypoint_chain(commander=commander, start=start, end=end, num_waypoints=10)

    assert isinstance(result, WaypointChainIkResult)
    assert result.success is True
    assert result.first_failure is None
    assert result.final_joint_positions is not None
    assert math.isclose(result.final_joint_positions[0], 1.0, abs_tol=1e-9)
    # Waypoint k's seed is exactly waypoint (k-1)'s solved joint_positions.
    for previous, current in zip(commander.calls, commander.calls[1:]):
        _, previous_seed = previous
        current_target, current_seed = current
        assert current_seed is not None
        assert math.isclose(current_seed[0], (previous_seed[0] if previous_seed else 0.0) + 0.1, abs_tol=1e-9)
    assert commander.calls[0][1] is None  # first waypoint uses the caller's seed (none, here)
    assert len(result.waypoints) == 10
    assert result.waypoints[-1].fraction == 1.0


def test_solve_cartesian_waypoint_chain_uses_caller_supplied_initial_seed() -> None:
    commander = _SeedChainingCommander()
    start = _pose(0.0, 0.0, 0.0)
    end = _pose(1.0, 0.0, 0.0)
    initial_seed = tuple(float(value) for value in range(7))

    solve_cartesian_waypoint_chain(
        commander=commander,
        start=start,
        end=end,
        num_waypoints=3,
        seed_joint_positions=initial_seed,
    )

    assert commander.calls[0][1] == initial_seed


def test_solve_cartesian_waypoint_chain_stops_at_first_failure_by_default() -> None:
    commander = _SeedChainingCommander(fail_at_calls=frozenset({3}))
    start = _pose(0.0, 0.0, 0.0)
    end = _pose(1.0, 0.0, 0.0)

    result = solve_cartesian_waypoint_chain(commander=commander, start=start, end=end, num_waypoints=10)

    assert result.success is False
    assert len(result.waypoints) == 3
    assert result.first_failure is not None
    assert result.first_failure.waypoint_index == 3
    assert result.final_joint_positions is None
    assert len(commander.calls) == 3


def test_solve_cartesian_waypoint_chain_can_continue_past_a_failure() -> None:
    commander = _SeedChainingCommander(fail_at_calls=frozenset({3}))
    start = _pose(0.0, 0.0, 0.0)
    end = _pose(1.0, 0.0, 0.0)

    result = solve_cartesian_waypoint_chain(
        commander=commander,
        start=start,
        end=end,
        num_waypoints=5,
        stop_on_failure=False,
    )

    assert len(result.waypoints) == 5
    assert result.success is False
    assert [waypoint.ok for waypoint in result.waypoints] == [True, True, False, True, True]
    # Waypoint 4 must have been re-seeded from waypoint 2's last successful
    # solution (0.2), not from waypoint 3's failure.
    _, waypoint_4_seed = commander.calls[3]
    assert waypoint_4_seed is not None
    assert math.isclose(waypoint_4_seed[0], 0.2, abs_tol=1e-9)


class _DirectOnlyCommander:
    """A minimal commander exposing only compute_ik, like MoveItPoseCommander
    callers use for the existing single-shot 'direct' strategy."""

    def __init__(self) -> None:
        self.calls: list[tuple[PoseTarget, tuple[float, ...] | None]] = []

    def compute_ik(self, target, *, seed_joint_positions=None):
        seed = tuple(seed_joint_positions) if seed_joint_positions is not None else None
        self.calls.append((target, seed))
        return [1.0] * 7, "direct ok"


def test_resolve_ik_direct_strategy_is_a_pure_passthrough() -> None:
    commander = _DirectOnlyCommander()
    target = _pose(1.0, 0.0, 0.0)
    seed = (0.1,) * 7

    joints, message = resolve_ik(commander, target, strategy="direct", seed_joint_positions=seed)

    assert joints == [1.0] * 7
    assert message == "direct ok"
    assert commander.calls == [(target, seed)]


def test_resolve_ik_rejects_unknown_strategy() -> None:
    commander = _DirectOnlyCommander()
    target = _pose(1.0, 0.0, 0.0)

    try:
        resolve_ik(commander, target, strategy="teleport")
    except ValueError as exc:
        assert "teleport" in str(exc)
    else:
        raise AssertionError("Expected an unknown strategy to raise.")


class _WaypointCapableCommander(_SeedChainingCommander):
    def __init__(self, *, current_pose: PoseTarget, **kwargs) -> None:
        super().__init__(**kwargs)
        self._current_pose = current_pose
        self.fk_calls: list[str] = []

    def get_current_pose(self, *, frame_id: str) -> PoseTarget:
        self.fk_calls.append(frame_id)
        return self._current_pose


def test_resolve_ik_cartesian_waypoints_starts_from_get_current_pose_when_no_start_given() -> None:
    current = _pose(0.0, 0.0, 0.0)
    target = _pose(1.0, 0.0, 0.0, frame_id="base_link")
    commander = _WaypointCapableCommander(current_pose=current)

    joints, message = resolve_ik(commander, target, strategy="cartesian_waypoints", num_waypoints=10)

    assert commander.fk_calls == ["base_link"]
    assert joints is not None
    assert math.isclose(joints[0], 1.0, abs_tol=1e-9)
    assert "cartesian_waypoint_chain succeeded" in message


def test_resolve_ik_cartesian_waypoints_uses_explicit_start_without_fk() -> None:
    start = _pose(0.0, 0.0, 0.0)
    target = _pose(1.0, 0.0, 0.0)
    commander = _WaypointCapableCommander(current_pose=_pose(99.0, 99.0, 99.0))

    joints, _ = resolve_ik(commander, target, strategy="cartesian_waypoints", start=start, num_waypoints=5)

    assert commander.fk_calls == []
    assert joints is not None


def test_resolve_ik_cartesian_waypoints_reports_the_failing_waypoint() -> None:
    start = _pose(0.0, 0.0, 0.0)
    target = _pose(1.0, 0.0, 0.0)
    commander = _WaypointCapableCommander(current_pose=start, fail_at_calls=frozenset({2}))

    joints, message = resolve_ik(commander, target, strategy="cartesian_waypoints", start=start, num_waypoints=10)

    assert joints is None
    assert "waypoint 2/10" in message


def test_waypoint_ik_result_is_plain_data() -> None:
    result = WaypointIkResult(
        waypoint_index=1,
        fraction=0.1,
        ok=True,
        message="ok",
        joint_positions=(0.0,) * 7,
    )
    assert result.ok is True
    assert len(result.joint_positions) == 7
