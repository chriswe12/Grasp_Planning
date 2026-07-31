#!/usr/bin/env python3
"""Preflight or execute the simple holder/inserter sequence on two real KUKAs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.ros2.dual_real_grasp_executor import (  # noqa: E402
    STOP_AFTER_CHOICES,
    DualRealExecutionConfig,
    execute_dual_real_plan,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=Path("artifacts/dual_grasp_planning/plumbers_block/simple_dual_robot_sim_plan.json"),
    )
    parser.add_argument(
        "--attempt-artifact",
        type=Path,
        default=Path("artifacts/dual_grasp_planning/plumbers_block/simple_dual_robot_real_attempt.json"),
    )
    parser.add_argument("--moveit-namespace", default="/lbr_dual_arm")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-objectless-planning",
        action="store_true",
        help=(
            "Acknowledge that MoveIt checks the table and both robots but does "
            "not contain the two Fabrica object meshes."
        ),
    )
    parser.add_argument(
        "--stop-after",
        choices=STOP_AFTER_CHOICES,
        default="holder_pregrasp",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the final typed confirmation.")
    parser.add_argument(
        "--skip-grippers",
        action="store_true",
        help="Only valid for an executed holder_pregrasp motion test.",
    )
    parser.add_argument("--velocity-scale", type=float, default=0.05)
    parser.add_argument("--acceleration-scale", type=float, default=0.05)
    parser.add_argument("--planning-time-s", type=float, default=8.0)
    parser.add_argument("--planning-attempts", type=int, default=8)
    parser.add_argument("--execute-timeout-s", type=float, default=120.0)
    parser.add_argument("--gripper-timeout-s", type=float, default=10.0)
    parser.add_argument("--grasp-settle-time-s", type=float, default=0.5)
    for role, robot in (("holder", "lbr_one"), ("inserter", "lbr_two")):
        parser.add_argument(
            f"--{role}-gripper-open-service",
            default=f"/{robot}/gripper_controller/open",
        )
        parser.add_argument(
            f"--{role}-gripper-close-service",
            default=f"/{robot}/gripper_controller/close",
        )
        parser.add_argument(
            f"--{role}-gripper-stop-service",
            default=f"/{robot}/gripper_controller/stop",
        )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.velocity_scale <= 0.20:
        raise ValueError("--velocity-scale must be in (0, 0.20] for this hardware runner.")
    if not 0.0 < args.acceleration_scale <= 0.20:
        raise ValueError("--acceleration-scale must be in (0, 0.20] for this hardware runner.")
    config = DualRealExecutionConfig(
        moveit_namespace=str(args.moveit_namespace),
        planning_time_s=float(args.planning_time_s),
        num_planning_attempts=int(args.planning_attempts),
        velocity_scale=float(args.velocity_scale),
        acceleration_scale=float(args.acceleration_scale),
        execute_timeout_s=float(args.execute_timeout_s),
        execute=bool(args.execute),
        require_confirmation=not bool(args.yes),
        allow_objectless_planning=bool(args.allow_objectless_planning),
        stop_after=str(args.stop_after),
        grippers_enabled=not bool(args.skip_grippers),
        gripper_timeout_s=float(args.gripper_timeout_s),
        grasp_settle_time_s=float(args.grasp_settle_time_s),
        holder_gripper_open_service=str(args.holder_gripper_open_service),
        holder_gripper_close_service=str(args.holder_gripper_close_service),
        holder_gripper_stop_service=str(args.holder_gripper_stop_service),
        inserter_gripper_open_service=str(args.inserter_gripper_open_service),
        inserter_gripper_close_service=str(args.inserter_gripper_close_service),
        inserter_gripper_stop_service=str(args.inserter_gripper_stop_service),
    )
    result = execute_dual_real_plan(
        plan_json=args.plan_json,
        attempt_artifact_path=args.attempt_artifact,
        config=config,
    )
    print(
        f"[DUAL-REAL] success={result.success} status={result.status} "
        f"last_completed={result.last_completed_phase or 'none'} "
        f"artifact={result.attempt_artifact_path}",
        flush=True,
    )
    if result.message:
        print(f"[DUAL-REAL] {result.message}", flush=True)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
