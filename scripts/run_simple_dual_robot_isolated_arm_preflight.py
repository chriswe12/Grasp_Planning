#!/usr/bin/env python3
"""Check whether the other KUKA arm's presence explains a dual-arm IK failure.

`run_simple_dual_robot_real.py` always resolves each arm's IK/motion plan
against the shared `/lbr_dual_arm` planning scene, which includes wherever
the other arm currently sits (see `dual_real_grasp_executor.py`). That is the
correct, conservative behavior for real execution, but when every candidate
fails identically at the same target (e.g. "IK failed with code=-31" for all
256 ranked pairs at `inserter_preinsertion`), that console output alone does
not say *why*: the target pose could be genuinely unreachable for that one
arm, or the other arm's current configuration could be blocking it.

This script answers that question for one already-selected candidate. It
loads the same saved `dual_robot_simple_sim_task` plan JSON used by
`run_simple_dual_robot_real.py` (built with `build_simple_dual_robot_task.py`)
and, against the same live `/lbr_dual_arm` MoveIt stack, runs every per-role
preflight target twice:

  coupled  - the default scene, collision-aware against both arms (what the
             real pipeline actually uses).
  isolated - the same scene, but with every arm_one <-> arm_two link pair
             temporarily marked collision-allowed in the move_group's
             allowed collision matrix, so each arm's IK is computed "as if
             the other robot weren't there." Self-collision and collision
             with the table/scene are still checked for each arm.

Nothing is executed and nothing physically moves; this is IK preflight only,
exactly like `run_simple_dual_robot_real.py` without `--execute`. The
allowed collision matrix is restored to what the move_group reported before
this script ran the isolated pass, even if the isolated pass raises.

Usage (after starting the mock or hardware dual MoveIt stack, e.g. via
`start_dual_lbr_moveit.sh`):

    python3 scripts/run_simple_dual_robot_isolated_arm_preflight.py \\
        --plan-json artifacts/dual_grasp_planning/plumbers_block/simple_dual_robot_real_task_bag_pose.json \\
        --stop-after inserter_preinsertion

Read the printed per-target table: a target marked DIVERGES failed coupled
but passed isolated, meaning the other arm's presence is the reason it
failed. A target that fails in both columns is not explained by the other
arm at all; look at that arm's own reachability, the target pose, or the
frame conversion instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.pipeline.cartesian_waypoint_ik import IK_STRATEGIES  # noqa: E402
from grasp_planning.pipeline.dual_robot_isolated_arm_preflight import (  # noqa: E402
    TargetIsolationResult,
    compare_candidate_arm_isolation,
)
from grasp_planning.ros2.dual_real_grasp_executor import (  # noqa: E402
    ROLE_SPECS,
    STOP_AFTER_CHOICES,
    DualRealExecutionConfig,
    _make_commander,
    _ranked_candidate_plans,
    _work_surface_obstacle,
    load_and_validate_dual_plan,
)
from grasp_planning.ros2.moveit_pose_commander import rclpy  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=Path("artifacts/dual_grasp_planning/plumbers_block/simple_dual_robot_sim_plan.json"),
        help="Same saved dual_robot_simple_sim_task plan used by run_simple_dual_robot_real.py.",
    )
    parser.add_argument(
        "--attempt-artifact",
        type=Path,
        default=Path("artifacts/dual_grasp_planning/plumbers_block/simple_dual_robot_isolated_arm_preflight.json"),
    )
    parser.add_argument("--moveit-namespace", default="/lbr_dual_arm")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--candidate-rank",
        type=int,
        default=1,
        help="1-based rank into the plan's ranked_pair_candidates (default: 1, the top-ranked pair).",
    )
    selector.add_argument(
        "--pair-id",
        default="",
        help="Select a candidate by pair_id or execution_candidate_id instead of --candidate-rank.",
    )
    parser.add_argument(
        "--stop-after",
        choices=STOP_AFTER_CHOICES,
        default="inserter_preinsertion",
        help="Last target to check, in MOTION_SEQUENCE order (default: the full sequence).",
    )
    parser.add_argument(
        "--ik-strategy",
        choices=IK_STRATEGIES,
        default="direct",
        help=(
            "Applied identically to the coupled and isolated passes, so a "
            "target that still diverges between them is not explained by "
            "this choice. 'cartesian_waypoints' walks each target through "
            "--cartesian-waypoint-count interpolated poses instead of one "
            "direct compute_ik call."
        ),
    )
    parser.add_argument("--cartesian-waypoint-count", type=int, default=10)
    parser.add_argument("--wait-for-moveit-timeout-s", type=float, default=20.0)
    parser.add_argument("--ik-timeout-s", type=float, default=2.0)
    parser.add_argument("--planning-time-s", type=float, default=8.0)
    parser.add_argument("--planning-attempts", type=int, default=8)
    return parser.parse_args()


def _select_candidate(
    candidates: tuple[dict[str, object], ...],
    *,
    candidate_rank: int,
    pair_id: str,
) -> tuple[dict[str, object], int]:
    if pair_id:
        for index, candidate in enumerate(candidates):
            candidate_id = str(candidate.get("execution_candidate_id", candidate.get("pair_id", "")))
            if candidate_id == pair_id or str(candidate.get("pair_id", "")) == pair_id:
                return candidate, index + 1
        raise ValueError(f"No ranked candidate has pair_id/execution_candidate_id={pair_id!r}.")
    if not 1 <= candidate_rank <= len(candidates):
        raise ValueError(f"--candidate-rank must be between 1 and {len(candidates)}; got {candidate_rank}.")
    return candidates[candidate_rank - 1], candidate_rank


def _print_results(results: tuple[TargetIsolationResult, ...]) -> None:
    header = f"{'role':<10}{'target':<28}{'coupled':<10}{'isolated':<10}{'':<10}"
    print(header)
    print("-" * len(header))
    for result in results:
        flag = "DIVERGES" if result.diverges else ""
        print(
            f"{result.role:<10}{result.target_name:<28}"
            f"{'ok' if result.coupled_ok else 'FAIL':<10}"
            f"{'ok' if result.isolated_ok else 'FAIL':<10}"
            f"{flag:<10}"
        )
        if not result.coupled_ok:
            print(f"    coupled:  {result.coupled_message}")
        if not result.isolated_ok:
            print(f"    isolated: {result.isolated_message}")

    diverging = [result for result in results if result.diverges and not result.coupled_ok]
    still_failing = [result for result in results if not result.coupled_ok and not result.isolated_ok]
    print()
    if diverging:
        names = ", ".join(result.target_name for result in diverging)
        print(
            f"[VERDICT] {len(diverging)} target(s) only fail with the other arm "
            f"present: {names}. Arm-arm interference is a real contributor here."
        )
    if still_failing:
        names = ", ".join(result.target_name for result in still_failing)
        print(
            f"[VERDICT] {len(still_failing)} target(s) fail even with the other "
            f"arm removed from collision consideration: {names}. That failure "
            "is not explained by the other arm's presence; check that arm's "
            "own reachability, the target pose, and the frame conversion."
        )
    if not diverging and not still_failing:
        print("[VERDICT] Every checked target passed in both modes.")


def main() -> int:
    args = _parse_args()
    if rclpy is None:
        raise RuntimeError(
            "ROS2 MoveIt dependencies are unavailable. Source ROS2, lbr-stack, "
            "and this repository's ros2_ws overlay first."
        )

    plan_json = args.plan_json.expanduser().resolve()
    source_plan = load_and_validate_dual_plan(plan_json)
    candidates = _ranked_candidate_plans(source_plan)
    candidate, rank = _select_candidate(
        candidates,
        candidate_rank=int(args.candidate_rank),
        pair_id=str(args.pair_id),
    )
    pair_id = str(candidate.get("pair_id", ""))
    print(
        f"[ISOLATION-CHECK] plan={plan_json} candidate_rank={rank}/{len(candidates)} "
        f"pair_id={pair_id} stop_after={args.stop_after} ik_strategy={args.ik_strategy}",
        flush=True,
    )

    config = DualRealExecutionConfig(
        moveit_namespace=str(args.moveit_namespace),
        wait_for_moveit_timeout_s=float(args.wait_for_moveit_timeout_s),
        ik_timeout_s=float(args.ik_timeout_s),
        planning_time_s=float(args.planning_time_s),
        num_planning_attempts=int(args.planning_attempts),
    )

    commanders: dict[str, object] = {}
    initialized_here = False
    try:
        if not rclpy.ok():
            rclpy.init()
            initialized_here = True
        commanders = {role: _make_commander(role=role, config=config) for role in ROLE_SPECS}
        for commander in commanders.values():
            commander.wait_for_moveit(require_execute=False)

        obstacle = _work_surface_obstacle(source_plan)
        ok, message = commanders["holder"].apply_planning_scene_obstacles(
            [obstacle],
            default_frame_id=str(config.frame_id),
        )
        print(f"[ISOLATION-CHECK] apply_work_surface: {'ok' if ok else 'failed'} {message}", flush=True)
        if not ok:
            raise RuntimeError(message)

        results = compare_candidate_arm_isolation(
            candidate=candidate,
            commanders=commanders,
            frame_id=str(config.frame_id),
            stop_after=str(args.stop_after),
            ik_strategy=str(args.ik_strategy),
            cartesian_waypoint_count=int(args.cartesian_waypoint_count),
        )
    finally:
        for commander in commanders.values():
            commander.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()

    _print_results(results)

    output_path = args.attempt_artifact.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "dual_robot_isolated_arm_preflight_attempt",
                "input_plan_json": str(plan_json),
                "candidate_rank": rank,
                "candidate_count": len(candidates),
                "pair_id": pair_id,
                "stop_after": str(args.stop_after),
                "ik_strategy": str(args.ik_strategy),
                "cartesian_waypoint_count": int(args.cartesian_waypoint_count),
                "results": [
                    {
                        "role": result.role,
                        "target_name": result.target_name,
                        "coupled_ok": result.coupled_ok,
                        "coupled_message": result.coupled_message,
                        "isolated_ok": result.isolated_ok,
                        "isolated_message": result.isolated_message,
                        "diverges": result.diverges,
                    }
                    for result in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ISOLATION-CHECK] wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
