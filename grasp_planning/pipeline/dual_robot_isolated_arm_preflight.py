"""Debug helper: does the other KUKA arm's presence explain an IK failure?

The coordinated dual-arm real preflight (`dual_real_grasp_executor.py`)
always resolves each arm's IK against the shared `/lbr_dual_arm` planning
scene, which includes wherever the other arm currently sits. That is
correct for the real pipeline, but it means a failing target's console
message ("IK failed with code=-31") does not say *why*: it could be that
this arm genuinely cannot reach that pose, or it could be that the other
arm's current configuration is blocking it.

This module answers that question for one already-selected candidate by
running the exact same per-role preflight targets twice against the same
live MoveIt stack:

  coupled  - the default scene (what `dual_real_grasp_executor.py` uses).
  isolated - the same scene, but with every `arm_one` <-> `arm_two` link
             pair temporarily marked collision-allowed in the move_group's
             allowed collision matrix (ACM), so each arm's IK/motion plan
             is computed "as if the other robot weren't there." Each arm's
             own self-collision and collision with the table/scene are
             still checked.

A target whose result flips between the two modes is being blocked by
arm-arm interference. A target that fails identically in both modes is not
explained by the other arm's presence at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from grasp_planning.ros2.dual_real_grasp_executor import (
    _motion_sequence_through,
    _preflight_targets,
)


def links_by_prefix(entry_names: Sequence[str], prefix: str) -> tuple[str, ...]:
    """Return every ACM entry name that belongs to one robot, e.g. `lbr_one`.

    Reads the prefix out of the live ACM instead of hard-coding the dual
    URDF's link list, so this stays correct if the description changes.
    """
    marker = f"{prefix}_"
    return tuple(name for name in entry_names if str(name).startswith(marker))


def disable_pairwise_collisions(
    entry_names: Sequence[str],
    entry_rows: Sequence[Sequence[bool]],
    group_a: Sequence[str],
    group_b: Sequence[str],
) -> tuple[tuple[str, ...], tuple[tuple[bool, ...], ...]]:
    """Return an ACM with every `group_a` x `group_b` link pair allowed.

    Every other entry is preserved exactly, including pairs already allowed
    for other reasons (e.g. `Adjacent`/`Never` pairs from the SRDF). Names in
    `group_a`/`group_b` that are missing from `entry_names` are appended
    first (with an all-False row) so isolation still applies even for a link
    the source ACM never mentioned.
    """
    names = list(entry_names)
    rows = [list(row) for row in entry_rows]

    for row in rows:
        if len(row) < len(names):
            row.extend([False] * (len(names) - len(row)))
    while len(rows) < len(names):
        rows.append([False] * len(names))

    index = {name: position for position, name in enumerate(names)}
    for name in (*group_a, *group_b):
        if name in index:
            continue
        index[name] = len(names)
        names.append(name)
        for row in rows:
            row.append(False)
        rows.append([False] * len(names))

    width = len(names)
    for row in rows:
        if len(row) < width:
            row.extend([False] * (width - len(row)))

    for link_a in group_a:
        row_index = index[link_a]
        for link_b in group_b:
            col_index = index[link_b]
            rows[row_index][col_index] = True
            rows[col_index][row_index] = True

    return tuple(names), tuple(tuple(row) for row in rows)


@dataclass(frozen=True)
class TargetIsolationResult:
    role: str
    target_name: str
    coupled_ok: bool
    coupled_message: str
    isolated_ok: bool
    isolated_message: str

    @property
    def diverges(self) -> bool:
        return self.coupled_ok != self.isolated_ok


def _run_preflight_pass(
    *,
    candidate: Mapping[str, object],
    commanders: Mapping[str, object],
    frame_id: str,
    stop_after: str,
    ik_strategy: str,
    cartesian_waypoint_count: int,
) -> dict[str, tuple[bool, str]]:
    results: dict[str, tuple[bool, str]] = {}

    def _record(*, name: str, role: str, ok: bool, message: str, **_ignored: object) -> None:
        del role
        prefix = "preflight_"
        if name.startswith(prefix):
            results[name[len(prefix) :]] = (bool(ok), str(message))

    _preflight_targets(
        plan=candidate,
        commanders=commanders,
        frame_id=frame_id,
        record=_record,
        stop_on_failure=False,
        stop_after=stop_after,
        ik_strategy=ik_strategy,
        cartesian_waypoint_count=cartesian_waypoint_count,
    )
    return results


def compare_candidate_arm_isolation(
    *,
    candidate: Mapping[str, object],
    commanders: Mapping[str, object],
    frame_id: str,
    stop_after: str = "inserter_preinsertion",
    ik_strategy: str = "direct",
    cartesian_waypoint_count: int = 10,
) -> tuple[TargetIsolationResult, ...]:
    """Run one candidate's preflight targets coupled, then isolated.

    `commanders` must have "holder" and "inserter" `MoveItPoseCommander`-like
    entries sharing the same live planning scene (any one of them can read
    or write the scene-wide ACM; "holder" is used here only because
    `dual_real_grasp_executor.py` already uses it for other shared scene
    edits). The original ACM is always restored before returning, even if a
    preflight call raises.

    `ik_strategy` is orthogonal to the coupled/isolated collision comparison
    this module runs: it controls how each individual IK call is resolved
    (see `grasp_planning/pipeline/cartesian_waypoint_ik.py`) and is applied
    identically to both the coupled and isolated passes, so a target that
    still diverges between them is not explained by the IK strategy choice.
    """
    reference_commander = commanders["holder"]
    baseline_names, baseline_rows = reference_commander.get_allowed_collision_matrix()

    coupled = _run_preflight_pass(
        candidate=candidate,
        commanders=commanders,
        frame_id=frame_id,
        stop_after=stop_after,
        ik_strategy=ik_strategy,
        cartesian_waypoint_count=cartesian_waypoint_count,
    )

    holder_links = links_by_prefix(baseline_names, "lbr_one")
    inserter_links = links_by_prefix(baseline_names, "lbr_two")
    if not holder_links or not inserter_links:
        raise RuntimeError(
            "Could not find lbr_one_*/lbr_two_* entries in the live allowed "
            "collision matrix; is the connected MoveIt stack really running "
            "the dual_iiwa7_y_gripper SRDF?"
        )

    isolated_names, isolated_rows = disable_pairwise_collisions(
        baseline_names,
        baseline_rows,
        holder_links,
        inserter_links,
    )
    ok, message = reference_commander.apply_allowed_collision_matrix(isolated_names, isolated_rows)
    if not ok:
        raise RuntimeError(f"Could not disable cross-arm collisions: {message}")

    try:
        isolated = _run_preflight_pass(
            candidate=candidate,
            commanders=commanders,
            frame_id=frame_id,
            stop_after=stop_after,
            ik_strategy=ik_strategy,
            cartesian_waypoint_count=cartesian_waypoint_count,
        )
    finally:
        restore_ok, restore_message = reference_commander.apply_allowed_collision_matrix(
            baseline_names,
            baseline_rows,
        )
        if not restore_ok:
            raise RuntimeError(f"Could not restore the original collision matrix: {restore_message}")

    results = []
    for role, target_name in _motion_sequence_through(stop_after):
        coupled_ok, coupled_message = coupled.get(target_name, (False, "not attempted"))
        isolated_ok, isolated_message = isolated.get(target_name, (False, "not attempted"))
        results.append(
            TargetIsolationResult(
                role=role,
                target_name=target_name,
                coupled_ok=coupled_ok,
                coupled_message=coupled_message,
                isolated_ok=isolated_ok,
                isolated_message=isolated_message,
            )
        )
    return tuple(results)


__all__ = [
    "TargetIsolationResult",
    "compare_candidate_arm_isolation",
    "disable_pairwise_collisions",
    "links_by_prefix",
]
