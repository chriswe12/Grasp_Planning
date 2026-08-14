"""Alternative IK strategy: walk to a target through interpolated waypoints.

The dual-arm preflight (`dual_real_grasp_executor.py`) solves IK for a target
in one shot: it hands the configured `kinematics_solver` plugin a single seed
(the robot's current state, or `KUKA_MOVEIT_ARM_START_JOINT_VALUES` on retry)
and one far-away Cartesian goal. The default plugin
(`kdl_kinematics_plugin/KDLKinematicsPlugin`, see
`ros2_ws/src/robot_integration_ros/config/dual_lbr_kinematics.yaml`) is a
local Newton/Jacobian solver with only a 0.05 s timeout: for a redundant
7-DOF iiwa7, it can easily fail to converge on a large jump even though the
target pose is kinematically reachable, simply because its one seed is too
far from any solution.

This module tests that hypothesis without touching the configured plugin.
It linearly interpolates the Cartesian pose (position lerp, orientation
slerp) into `num_waypoints` steps between a start and an end pose, then
solves IK one waypoint at a time, seeding each solve with the previous
waypoint's solved joints: solve waypoint 1 from the last known joints, seed
waypoint 2 with waypoint 1's solution, and so on up to the real target. Each
step is a small, easy correction from an already-valid nearby seed, which is
a much easier problem for the same local solver. No motion planning or
execution happens here; this only chains `compute_ik` calls to see whether a
solution exists at all.

If the walk reaches the target where the direct single-shot call failed,
the failure was a seeding/local-minima artifact of the configured IK plugin,
not a real kinematic or collision limit. If the walk also fails, that is
stronger evidence of a genuine limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from grasp_planning.ros2.moveit_pose_commander import PoseTarget


def slerp_quaternion_xyzw(
    start: tuple[float, float, float, float],
    end: tuple[float, float, float, float],
    fraction: float,
) -> tuple[float, float, float, float]:
    """Spherically interpolate two xyzw quaternions along their shortest arc."""
    x0, y0, z0, w0 = start
    x1, y1, z1, w1 = end

    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
    if dot < 0.0:
        x1, y1, z1, w1 = -x1, -y1, -z1, -w1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))

    if dot > 0.9995:
        # Endpoints are nearly identical; slerp's sin(theta) denominator
        # would be near zero. Linear interpolation plus renormalization is
        # numerically stable and visually indistinguishable here.
        x = x0 + (x1 - x0) * fraction
        y = y0 + (y1 - y0) * fraction
        z = z0 + (z1 - z0) * fraction
        w = w0 + (w1 - w0) * fraction
    else:
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * fraction
        s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
        s1 = math.sin(theta) / sin_theta_0
        x = s0 * x0 + s1 * x1
        y = s0 * y0 + s1 * y1
        z = s0 * z0 + s1 * z1
        w = s0 * w0 + s1 * w1

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / norm, y / norm, z / norm, w / norm)


def interpolate_pose_targets(
    start: PoseTarget,
    end: PoseTarget,
    *,
    num_waypoints: int,
) -> tuple[PoseTarget, ...]:
    """Return `num_waypoints` poses from `start` to `end`, excluding `start`.

    Waypoint `num_waypoints` equals `end` (up to floating point), so the
    final entry is always the real target being tested.
    """
    if num_waypoints < 1:
        raise ValueError(f"num_waypoints must be at least 1; got {num_waypoints}.")
    if start.frame_id != end.frame_id:
        raise ValueError(
            "start/end frame_id must match to linearly interpolate a "
            f"Cartesian path; got {start.frame_id!r} and {end.frame_id!r}."
        )

    waypoints = []
    for index in range(1, num_waypoints + 1):
        fraction = index / num_waypoints
        qx, qy, qz, qw = slerp_quaternion_xyzw(start.orientation_xyzw, end.orientation_xyzw, fraction)
        waypoints.append(
            PoseTarget.from_quaternion(
                x=start.x + (end.x - start.x) * fraction,
                y=start.y + (end.y - start.y) * fraction,
                z=start.z + (end.z - start.z) * fraction,
                quaternion_xyzw=(qx, qy, qz, qw),
                frame_id=start.frame_id,
            )
        )
    return tuple(waypoints)


class _IkCommander(Protocol):
    def compute_ik(
        self,
        target: PoseTarget,
        *,
        seed_joint_positions: Sequence[float] | None = None,
    ) -> tuple[list[float] | None, str]: ...


class _IkAndFkCommander(_IkCommander, Protocol):
    def get_current_pose(self, *, frame_id: str) -> PoseTarget: ...


IK_STRATEGIES = ("direct", "cartesian_waypoints")


@dataclass(frozen=True)
class WaypointIkResult:
    waypoint_index: int
    fraction: float
    ok: bool
    message: str
    joint_positions: tuple[float, ...] | None


@dataclass(frozen=True)
class WaypointChainIkResult:
    waypoints: tuple[WaypointIkResult, ...]

    @property
    def success(self) -> bool:
        return len(self.waypoints) > 0 and all(waypoint.ok for waypoint in self.waypoints)

    @property
    def first_failure(self) -> WaypointIkResult | None:
        for waypoint in self.waypoints:
            if not waypoint.ok:
                return waypoint
        return None

    @property
    def final_joint_positions(self) -> tuple[float, ...] | None:
        if not self.waypoints:
            return None
        last = self.waypoints[-1]
        return last.joint_positions if last.ok else None


def solve_cartesian_waypoint_chain(
    *,
    commander: _IkCommander,
    start: PoseTarget,
    end: PoseTarget,
    num_waypoints: int = 10,
    seed_joint_positions: Sequence[float] | None = None,
    stop_on_failure: bool = True,
) -> WaypointChainIkResult:
    """Solve IK for `end` by chaining `num_waypoints` interpolated solves.

    `seed_joint_positions` seeds only the first waypoint; every later
    waypoint is seeded with the previous waypoint's own solved joints. Pass
    `stop_on_failure=False` to keep solving past a failed waypoint (reusing
    the last successful seed) instead of aborting the chain, to see whether
    the walk can recover further along.
    """
    waypoint_targets = interpolate_pose_targets(start, end, num_waypoints=num_waypoints)
    seed = tuple(float(value) for value in seed_joint_positions) if seed_joint_positions is not None else None

    results: list[WaypointIkResult] = []
    for index, target in enumerate(waypoint_targets, start=1):
        joints, message = commander.compute_ik(target, seed_joint_positions=seed)
        resolved = tuple(float(value) for value in joints) if joints is not None else None
        results.append(
            WaypointIkResult(
                waypoint_index=index,
                fraction=index / num_waypoints,
                ok=resolved is not None,
                message=str(message),
                joint_positions=resolved,
            )
        )
        if resolved is not None:
            seed = resolved
        elif stop_on_failure:
            break
    return WaypointChainIkResult(tuple(results))


def resolve_ik(
    commander: _IkAndFkCommander,
    target: PoseTarget,
    *,
    strategy: str = "direct",
    seed_joint_positions: Sequence[float] | None = None,
    num_waypoints: int = 10,
    start: PoseTarget | None = None,
) -> tuple[list[float] | None, str]:
    """Resolve IK for `target` using the requested strategy.

    This is the one modular seam both the single-arm (`real_grasp_executor.py`)
    and dual-arm (`dual_real_grasp_executor.py`) real executors call through,
    so switching strategies is a config change in either routine rather than
    a code change in each:

    - `"direct"` - today's behavior: one `commander.compute_ik` call. Return
      value and failure message are passed through unchanged.
    - `"cartesian_waypoints"` - walk to `target` through `num_waypoints`
      linearly interpolated Cartesian poses (see module docstring), starting
      from `start` if given, otherwise from `commander.get_current_pose()`.
      Returns the final waypoint's joints on a fully successful walk, or
      `None` with a message naming the first waypoint that failed.

    Both strategies return the same `(joints_or_None, message)` shape as
    `MoveItPoseCommander.compute_ik`, so callers do not need to branch.
    """
    if strategy == "direct":
        return commander.compute_ik(target, seed_joint_positions=seed_joint_positions)
    if strategy != "cartesian_waypoints":
        raise ValueError(f"Unknown ik_strategy {strategy!r}; expected one of {IK_STRATEGIES}.")

    start_pose = start if start is not None else commander.get_current_pose(frame_id=target.frame_id)
    chain_result = solve_cartesian_waypoint_chain(
        commander=commander,
        start=start_pose,
        end=target,
        num_waypoints=num_waypoints,
        seed_joint_positions=seed_joint_positions,
    )
    if chain_result.success:
        final_joints = chain_result.final_joint_positions
        assert final_joints is not None
        return (
            list(final_joints),
            f"cartesian_waypoint_chain succeeded over {len(chain_result.waypoints)} waypoint(s)",
        )
    failure = chain_result.first_failure
    assert failure is not None
    return (
        None,
        f"cartesian_waypoint_chain failed at waypoint {failure.waypoint_index}/{num_waypoints} "
        f"(fraction={failure.fraction:.2f}): {failure.message}",
    )


__all__ = [
    "IK_STRATEGIES",
    "WaypointChainIkResult",
    "WaypointIkResult",
    "interpolate_pose_targets",
    "resolve_ik",
    "slerp_quaternion_xyzw",
    "solve_cartesian_waypoint_chain",
]
