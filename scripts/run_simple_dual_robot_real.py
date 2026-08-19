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
            "Permit execution of a legacy task that lacks phase-aware part "
            "AABBs and an attached incoming-part collision body."
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
    parser.add_argument("--ik-timeout-s", type=float, default=0.35)
    parser.add_argument("--exact-ik-candidates", type=int, default=7)
    parser.add_argument("--exact-ik-beam-width", type=int, default=4)
    parser.add_argument("--exact-ik-seed-perturbation-rad", type=float, default=0.60)
    parser.add_argument("--cartesian-max-step-m", type=float, default=0.005)
    parser.add_argument("--cartesian-revolute-jump-threshold-rad", type=float, default=0.35)
    parser.add_argument("--execution-start-tolerance-rad", type=float, default=0.05)
    parser.add_argument("--execute-timeout-s", type=float, default=120.0)
    parser.add_argument("--gripper-timeout-s", type=float, default=10.0)
    parser.add_argument("--grasp-settle-time-s", type=float, default=0.5)
    parser.add_argument("--gripper-position-feedback-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--debug-gui",
        action="store_true",
        help="Open the live dual-arm candidate and execution browser view.",
    )
    parser.add_argument("--debug-gui-port", type=int, default=0)
    parser.add_argument(
        "--no-debug-gui-open-browser",
        action="store_false",
        dest="debug_gui_open_browser",
        help="Reuse an existing debug browser tab instead of opening another one.",
    )
    parser.set_defaults(debug_gui_open_browser=True)
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
        parser.add_argument(
            f"--{role}-gripper-position-command-topic",
            default=f"/{robot}/gripper_controller/position_command",
        )
        parser.add_argument(
            f"--{role}-gripper-position-feedback-topic",
            default=f"/{robot}/gripper_controller/position",
        )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not 0.0 < args.velocity_scale <= 0.20:
        raise ValueError("--velocity-scale must be in (0, 0.20] for this hardware runner.")
    if not 0.0 < args.acceleration_scale <= 0.20:
        raise ValueError("--acceleration-scale must be in (0, 0.20] for this hardware runner.")
    if args.cartesian_max_step_m <= 0.0:
        raise ValueError("--cartesian-max-step-m must be positive.")
    if args.cartesian_revolute_jump_threshold_rad <= 0.0:
        raise ValueError("--cartesian-revolute-jump-threshold-rad must be positive.")
    if args.execution_start_tolerance_rad <= 0.0:
        raise ValueError("--execution-start-tolerance-rad must be positive.")
    if args.exact_ik_candidates <= 0:
        raise ValueError("--exact-ik-candidates must be positive.")
    if args.ik_timeout_s <= 0.0:
        raise ValueError("--ik-timeout-s must be positive.")
    if args.exact_ik_beam_width <= 0:
        raise ValueError("--exact-ik-beam-width must be positive.")
    config = DualRealExecutionConfig(
        moveit_namespace=str(args.moveit_namespace),
        ik_timeout_s=float(args.ik_timeout_s),
        planning_time_s=float(args.planning_time_s),
        num_planning_attempts=int(args.planning_attempts),
        ik_candidate_count=int(args.exact_ik_candidates),
        ik_beam_width=int(args.exact_ik_beam_width),
        ik_seed_perturbation_rad=float(args.exact_ik_seed_perturbation_rad),
        velocity_scale=float(args.velocity_scale),
        acceleration_scale=float(args.acceleration_scale),
        cartesian_max_step_m=float(args.cartesian_max_step_m),
        cartesian_revolute_jump_threshold_rad=float(args.cartesian_revolute_jump_threshold_rad),
        execution_start_tolerance_rad=float(args.execution_start_tolerance_rad),
        execute_timeout_s=float(args.execute_timeout_s),
        execute=bool(args.execute),
        require_confirmation=not bool(args.yes),
        allow_objectless_planning=bool(args.allow_objectless_planning),
        stop_after=str(args.stop_after),
        grippers_enabled=not bool(args.skip_grippers),
        gripper_timeout_s=float(args.gripper_timeout_s),
        grasp_settle_time_s=float(args.grasp_settle_time_s),
        gripper_position_feedback_tolerance=float(args.gripper_position_feedback_tolerance),
        holder_gripper_open_service=str(args.holder_gripper_open_service),
        holder_gripper_close_service=str(args.holder_gripper_close_service),
        holder_gripper_stop_service=str(args.holder_gripper_stop_service),
        holder_gripper_position_command_topic=str(args.holder_gripper_position_command_topic),
        holder_gripper_position_feedback_topic=str(args.holder_gripper_position_feedback_topic),
        inserter_gripper_open_service=str(args.inserter_gripper_open_service),
        inserter_gripper_close_service=str(args.inserter_gripper_close_service),
        inserter_gripper_stop_service=str(args.inserter_gripper_stop_service),
        inserter_gripper_position_command_topic=str(args.inserter_gripper_position_command_topic),
        inserter_gripper_position_feedback_topic=str(args.inserter_gripper_position_feedback_topic),
        debug_gui=bool(args.debug_gui),
        debug_gui_port=int(args.debug_gui_port),
        debug_gui_open_browser=bool(args.debug_gui_open_browser),
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
