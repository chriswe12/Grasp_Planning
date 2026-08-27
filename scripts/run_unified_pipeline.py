#!/usr/bin/env python3
"""One public dispatcher for bringup, execution, actions, and benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_SCRIPTS = {
    "grasp-generation": "scripts/run_grasp_generation_benchmark.py",
    "grasp-execution": "scripts/run_grasp_execution_benchmark.py",
    "dual-assembly": "scripts/run_dual_assembly_benchmark.py",
    "solo-pickup-ik": "scripts/run_solo_pickup_ik_ab_benchmark.py",
}
ROBOTS = ("left", "right", "both")
WORKFLOWS = ("dual", "single-object")


@dataclass(frozen=True)
class UnifiedInvocation:
    command: tuple[str, ...]
    description: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified Fabrica entrypoint. Dual holder/inserter execution is the "
            "default; select --workflow single-object only for the legacy "
            "single-object stage-1/stage-2 path."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./run_pipeline.sh --mode real --robots both --execute
  ./run_pipeline.sh --mode real --robots left --grasp-only --execute
  ./run_pipeline.sh --mode real --robots both --policy velocity-rotation --execute
  ./run_pipeline.sh --mode real --robots left --bringup-only --rviz
  ./run_pipeline.sh --mode pitl --robots both --serve-action --headless
  ./run_pipeline.sh --benchmark dual-assembly --limit-cases 4
  ./run_pipeline.sh --workflow single-object --mode sim --backend isaac --headless
""",
    )
    parser.add_argument("--workflow", choices=WORKFLOWS, default="dual")
    parser.add_argument("--mode", choices=("sim", "pitl", "real"), default=None)
    parser.add_argument("--robots", choices=ROBOTS, default="both")
    parser.add_argument(
        "--role",
        choices=("holder", "inserter"),
        default="inserter",
        help="Task role when one physical robot is selected. Default: inserter.",
    )
    parser.add_argument(
        "--grasp-only",
        "--grasp_only",
        action="store_true",
        help="Grasp and lift the incoming part, then stop before transport.",
    )
    parser.add_argument(
        "--policy",
        default="",
        help="Real-only policy registry name or checkpoint path used for every active grasp approach.",
    )
    parser.add_argument("--camera", choices=("realsense_1", "realsense_2"), default=None)
    parser.add_argument("--left-camera", choices=("realsense_1", "realsense_2"), default="realsense_1")
    parser.add_argument("--right-camera", choices=("realsense_1", "realsense_2"), default="realsense_2")
    parser.add_argument("--bringup-only", action="store_true")
    parser.add_argument("--serve-action", action="store_true")
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_SCRIPTS), default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--backend", choices=("config", "mujoco", "isaac", "both", "none"), default="config")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rviz", action="store_true")
    parser.add_argument("--servo", action="store_true")
    moveit_ownership = parser.add_mutually_exclusive_group()
    moveit_ownership.add_argument(
        "--start-moveit",
        action="store_true",
        help="Start and own a temporary MoveIt stack; persistent-stack reuse is the default.",
    )
    moveit_ownership.add_argument(
        "--reuse-moveit",
        action="store_true",
        help="Explicit compatibility spelling for the default persistent-stack reuse.",
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--gripper-model", choices=("pdz_gripper", "y_gripper"), default="pdz_gripper")
    parser.add_argument("--list-policies", action="store_true")
    parser.add_argument("--generate-config-only", action="store_true")
    parser.add_argument(
        "--stage2-bundle",
        type=Path,
        default=None,
        help="Execute an existing stage-2 bundle through the selected simulation backend.",
    )
    parser.add_argument("--grasp-id", default="")
    parser.add_argument("--attempt-artifact", type=Path, default=None)
    parser.add_argument(
        "--backend-python",
        default="",
        help="Backend interpreter command; primarily the Isaac Lab launcher.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved child command without starting it.",
    )
    return parser


def _python() -> str:
    return os.environ.get("PIPELINE_PYTHON", "").strip() or sys.executable or "python3"


def _append_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def _single_object_invocation(args: argparse.Namespace, extra: Sequence[str]) -> UnifiedInvocation:
    if args.mode is None:
        raise ValueError("--mode is required for single-object execution.")
    config = args.config or REPO_ROOT / "configs" / f"grasp_pipeline_{args.mode}.yaml"
    command = [
        _python(),
        str(REPO_ROOT / "scripts/run_grasp_pipeline.py"),
        "--mode",
        args.mode,
        "--config",
        str(config),
        "--backend",
        args.backend,
    ]
    _append_flag(command, args.headless, "--headless")
    command.extend(extra)
    return UnifiedInvocation(tuple(command), "single-object pipeline")


def _saved_bundle_invocation(args: argparse.Namespace, extra: Sequence[str]) -> UnifiedInvocation:
    if args.mode != "sim":
        raise ValueError("--stage2-bundle currently requires --mode sim.")
    if args.backend not in {"mujoco", "isaac"}:
        raise ValueError("Saved stage-2 execution requires --backend mujoco or --backend isaac.")
    if not args.grasp_id:
        raise ValueError("--grasp-id is required with --stage2-bundle.")
    if args.attempt_artifact is None:
        raise ValueError("--attempt-artifact is required with --stage2-bundle.")
    python_command = (
        shlex.split(args.backend_python)
        if args.backend_python.strip()
        else [_python()]
    )
    script = (
        "scripts/run_fabrica_grasp_in_mujoco.py"
        if args.backend == "mujoco"
        else "scripts/run_fabrica_grasp_in_isaac.py"
    )
    command = [
        *python_command,
        str(REPO_ROOT / script),
        "--input-json",
        str(args.stage2_bundle.expanduser().resolve()),
        "--grasp-id",
        args.grasp_id,
        "--attempt-artifact",
        str(args.attempt_artifact.expanduser().resolve()),
    ]
    _append_flag(command, args.headless, "--headless")
    command.extend(extra)
    return UnifiedInvocation(tuple(command), f"saved stage-2 {args.backend} execution")


def _policy_single_invocation(args: argparse.Namespace, extra: Sequence[str]) -> UnifiedInvocation:
    if args.mode != "real":
        raise ValueError("--policy is supported only with --mode real.")
    if args.robots == "both":
        raise ValueError("Internal error: dual policy selection reached the single-arm adapter.")
    camera = args.camera or (args.left_camera if args.robots == "left" else args.right_camera)
    command = [
        _python(),
        str(REPO_ROOT / "scripts/run_single_arm_policy_pickup.py"),
        "--policy",
        args.policy,
        "--robot",
        args.robots,
        "--camera",
        camera,
    ]
    _append_flag(command, args.generate_config_only, "--generate-config-only")
    command.extend(extra)
    return UnifiedInvocation(tuple(command), f"{args.robots} single-arm policy pickup")


def _bringup_invocation(args: argparse.Namespace, extra: Sequence[str]) -> UnifiedInvocation:
    if args.mode is None:
        raise ValueError("--mode is required with --bringup-only.")
    launch_mode = "hardware" if args.mode == "real" else "mock"
    servo = bool(args.servo or args.policy)
    if args.workflow == "dual":
        command = [
            str(REPO_ROOT / "start_dual_lbr_moveit.sh"),
            "--mode",
            launch_mode,
            "--robots",
            args.robots,
            "--gripper-model",
            args.gripper_model,
        ]
        _append_flag(command, servo, "--servo")
    else:
        if args.robots == "both":
            raise ValueError("The single-object bringup requires --robots left or right.")
        command = [
            str(REPO_ROOT / "start_lbr_moveit.sh"),
            "--mode",
            launch_mode,
            "--arm",
            "lbr-one" if args.robots == "left" else "lbr-two",
            "--gripper-model",
            args.gripper_model,
        ]
        _append_flag(command, servo, "--servo")
    _append_flag(command, args.rviz, "--rviz")
    command.extend(extra)
    return UnifiedInvocation(tuple(command), f"{args.robots} MoveIt bringup")


def _action_invocation(args: argparse.Namespace, extra: Sequence[str]) -> UnifiedInvocation:
    if args.mode not in {"pitl", "real"}:
        raise ValueError("--serve-action requires --mode pitl or --mode real.")
    command = [
        "ros2",
        "run",
        "robot_integration_ros",
        "grasp_assembly_action_server",
        "--repo-root",
        str(REPO_ROOT),
        "--dual-mode",
        args.mode,
        "--robots",
        args.robots,
        "--single-role",
        args.role,
    ]
    if args.config is not None:
        command.extend(("--config", str(args.config)))
    _append_flag(command, args.execute, "--execute")
    _append_flag(command, args.headless, "--headless")
    stop_after = "inserter_preinsertion"
    if args.robots != "both" and args.role == "holder":
        stop_after = "holder_grasp"
    if args.grasp_only:
        stop_after = "inserter_pickup_lift" if args.role == "inserter" else "holder_grasp"
    command.extend(("--stop-after", stop_after))
    if args.policy:
        command.extend(("--policy", args.policy, "--left-camera", args.left_camera, "--right-camera", args.right_camera))
    command.extend(extra)
    return UnifiedInvocation(tuple(command), f"{args.robots} {args.mode} GraspAssembly action server")


def _dual_invocation(args: argparse.Namespace, extra: Sequence[str]) -> UnifiedInvocation:
    if args.mode is None:
        raise ValueError("--mode is required for execution.")
    if args.policy and args.mode != "real":
        raise ValueError("--policy is supported only with --mode real.")
    if args.robots != "both" and args.mode != "real":
        raise ValueError("Single-active-robot task execution is currently supported only with --mode real.")
    command = [
        str(REPO_ROOT / "scripts/run_dual_pipeline.sh"),
        "--mode",
        "sim" if args.mode == "pitl" else args.mode,
        "--robots",
        args.robots,
        "--single-role",
        args.role,
        "--gripper-model",
        args.gripper_model,
    ]
    stop_after = "inserter_preinsertion"
    if args.robots != "both" and args.role == "holder":
        stop_after = "holder_grasp"
    if args.grasp_only:
        stop_after = "inserter_pickup_lift" if args.role == "inserter" else "holder_grasp"
    if args.mode == "real":
        command.extend(("--stop-after", stop_after))
    _append_flag(command, args.execute, "--execute")
    _append_flag(command, args.yes, "--yes")
    _append_flag(command, args.headless, "--headless")
    _append_flag(command, args.rviz, "--rviz")
    _append_flag(command, args.start_moveit, "--start-moveit")
    _append_flag(command, args.reuse_moveit, "--reuse-moveit")
    if args.policy:
        command.extend(
            (
                "--policy",
                args.policy,
                "--left-camera",
                args.left_camera,
                "--right-camera",
                args.right_camera,
            )
        )
    command.extend(extra)
    return UnifiedInvocation(tuple(command), f"{args.robots} dual-task pipeline")


def _benchmark_invocation(args: argparse.Namespace, extra: Sequence[str]) -> UnifiedInvocation:
    assert args.benchmark is not None
    command = [_python(), str(REPO_ROOT / BENCHMARK_SCRIPTS[args.benchmark])]
    if args.config is not None:
        command.extend(("--config", str(args.config)))
    if args.benchmark == "grasp-execution" and args.backend != "config":
        command.extend(("--backend", args.backend))
    command.extend(extra)
    return UnifiedInvocation(tuple(command), f"{args.benchmark} benchmark")


def _command_option(command: Sequence[str], name: str) -> str:
    try:
        index = list(command).index(name)
    except ValueError as exc:
        raise ValueError(f"Resolved backend command is missing required option {name}.") from exc
    if index + 1 >= len(command):
        raise ValueError(f"Resolved backend command has no value after {name}.")
    return str(command[index + 1])


def _prepare_saved_isaac_invocation(invocation: UnifiedInvocation) -> UnifiedInvocation:
    """Own host-side MoveIt preparation before entering Isaac's Python runtime."""

    if invocation.description != "saved stage-2 isaac execution":
        return invocation
    if "--moveit-plan-json" in invocation.command:
        return invocation

    from grasp_planning.ros2.saved_stage2_moveit import (
        config_from_backend_command,
        preplan_saved_stage2_for_isaac,
    )

    stage2_bundle = Path(_command_option(invocation.command, "--input-json"))
    grasp_id = _command_option(invocation.command, "--grasp-id")
    attempt_artifact = Path(_command_option(invocation.command, "--attempt-artifact"))
    plan_path = attempt_artifact.with_name(f"{attempt_artifact.stem}_moveit_plan.json")
    config = config_from_backend_command(invocation.command)
    try:
        preplan_saved_stage2_for_isaac(
            stage2_bundle=stage2_bundle,
            grasp_id=grasp_id,
            output_path=plan_path,
            config=config,
        )
    except Exception as exc:
        attempt_artifact.parent.mkdir(parents=True, exist_ok=True)
        attempt_artifact.write_text(
            json.dumps(
                {
                    "attempt": {
                        "stage2_bundle": str(stage2_bundle),
                        "grasp_id": grasp_id,
                        "backend": "isaac",
                    },
                    "execution": {
                        "controller": "moveit",
                        "success": False,
                        "status": "moveit_preplan_failed",
                        "message": str(exc),
                    },
                    "moveit": {
                        "plan_json": str(plan_path),
                        "planning_group": config.planning_group,
                        "pose_link": config.pose_link,
                        "namespace": config.namespace,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(f"Saved-stage2 Isaac MoveIt preplanning failed: {exc}") from exc
    return UnifiedInvocation(
        (*invocation.command, "--moveit-plan-json", str(plan_path)),
        invocation.description,
    )


def resolve_invocation(argv: Sequence[str]) -> tuple[UnifiedInvocation, bool]:
    args, extra = _parser().parse_known_args(list(argv))
    if args.list_policies and not args.policy:
        command = [_python(), str(REPO_ROOT / "scripts/run_single_arm_policy_pickup.py"), "--list-policies"]
        return UnifiedInvocation(tuple(command), "policy registry listing"), bool(args.dry_run)
    if args.benchmark is not None:
        invocation = _benchmark_invocation(args, extra)
    elif args.bringup_only:
        invocation = _bringup_invocation(args, extra)
    elif args.serve_action:
        invocation = _action_invocation(args, extra)
    elif args.stage2_bundle is not None:
        invocation = _saved_bundle_invocation(args, extra)
    elif args.workflow == "single-object":
        if args.policy:
            if args.robots == "both":
                raise ValueError("The single-object policy workflow requires --robots left or right.")
            invocation = _policy_single_invocation(args, extra)
        else:
            invocation = _single_object_invocation(args, extra)
    else:
        invocation = _dual_invocation(args, extra)
    return invocation, bool(args.dry_run)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        invocation, dry_run = resolve_invocation(sys.argv[1:] if argv is None else argv)
    except ValueError as exc:
        _parser().error(str(exc))
    print(f"[PIPELINE] route={invocation.description}", flush=True)
    print(f"[PIPELINE] command={shlex.join(invocation.command)}", flush=True)
    if dry_run:
        return 0
    try:
        invocation = _prepare_saved_isaac_invocation(invocation)
    except (ValueError, RuntimeError) as exc:
        print(f"[PIPELINE] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    if "--moveit-plan-json" in invocation.command:
        print(f"[PIPELINE] prepared_command={shlex.join(invocation.command)}", flush=True)
    return subprocess.run(list(invocation.command), cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
