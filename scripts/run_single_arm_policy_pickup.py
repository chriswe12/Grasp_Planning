#!/usr/bin/env python3
"""Run one live-planned single-arm pickup with a selected D405 PPO policy."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.gripper_profiles import (
    SERVO_GRIPPER_CLOSED_WIDTH_M,
    SERVO_GRIPPER_OPEN_WIDTH_M,
)
from grasp_planning.rl.d405_deployment_config import (  # noqa: E402
    camera_driver_root,
    write_visual_servo_config,
)
from grasp_planning.rl.policy_registry import (  # noqa: E402
    load_policy_registry,
    resolve_policy_reference,
)
from grasp_planning.rl.policy_registry import (
    load_yaml_mapping as _load_yaml_mapping,
)
from grasp_planning.rl.policy_registry import (
    resolve_policy_assets as resolve_policy_assets,
)
from grasp_planning.subprocess_lifecycle import run_process_group

DEFAULT_REGISTRY = REPO_ROOT / "configs/d405_policy_registry.yaml"
DEFAULT_PIPELINE = REPO_ROOT / "configs/grasp_pipeline_real_lbr_iiwa7.yaml"
DEFAULT_VISUAL_SERVO = REPO_ROOT / "configs/visual_servo_real_d405.yaml"


def _validate_part_id(part_id: str) -> Path:
    requested = str(part_id).strip()
    if not requested or not requested.isdigit():
        raise ValueError("--part-id must be a non-negative integer.")
    mesh = REPO_ROOT / f"assets/obj/fabrica/plumbers_block/{requested}.obj"
    if not mesh.is_file():
        mesh = REPO_ROOT / f"obj/fabrica/plumbers_block/{requested}.obj"
    if not mesh.is_file():
        raise ValueError(f"No plumbers_block mesh exists for part '{requested}'.")
    return mesh.resolve()


def write_resolved_configs(
    *,
    policy_name: str,
    part_id: str,
    part_mesh: Path,
    assets: Mapping[str, object],
    pipeline_template: Path,
    visual_servo_template: Path,
    output_dir: Path,
    model_device: str,
    open_debug_html: bool,
    camera_name: str = "realsense_1",
    robot_side: str = "left",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    selected_camera = camera_driver_root(camera_name)
    selected_side = str(robot_side).strip()
    if selected_side not in {"left", "right"}:
        raise ValueError("robot_side must be 'left' or 'right'.")
    gripper_model = str(assets["gripper_model"])
    if gripper_model == "pdz_gripper":
        tcp_frame = "pdz_gripper_tcp"
        moveit_gripper_joint = "pdz_gripper_left_finger_joint"
        collision_model = "pdz_gripper"
        planning_min_width = 0.008
        planning_max_width = 0.062
    elif gripper_model == "y_gripper":
        tcp_frame = "gripper_tcp"
        moveit_gripper_joint = "left_finger_joint"
        collision_model = "kuka_y_gripper"
        planning_min_width = SERVO_GRIPPER_CLOSED_WIDTH_M
        planning_max_width = SERVO_GRIPPER_OPEN_WIDTH_M
    else:
        raise ValueError(f"Unsupported policy gripper_model '{gripper_model}'.")
    visual_path = output_dir / "visual_servo.yaml"
    write_visual_servo_config(
        policy_name=policy_name,
        assets=assets,
        template_path=visual_servo_template,
        output_path=visual_path,
        output_root=output_dir / "policy_runs",
        model_device=model_device,
        camera_name=selected_camera,
    )

    pipeline = _load_yaml_mapping(pipeline_template)
    geometry = dict(pipeline.get("geometry", {}))
    geometry["target_mesh_path"] = str(part_mesh)
    pipeline["geometry"] = geometry
    ros2_config = dict(pipeline.get("ros2", {}))
    ros2_config.update({"assembly_name": "plumbers_block", "part_id": int(part_id)})
    pipeline["ros2"] = ros2_config
    artifacts = dict(pipeline.get("artifacts", {}))
    artifacts.update(
        {
            "stage1_json": str(output_dir / "stage1.json"),
            "stage1_html": str(output_dir / "stage1.html"),
            "stage2_json": str(output_dir / "stage2.json"),
            "stage2_html": str(output_dir / "stage2.html"),
            "part_frame_html": str(output_dir / "part_frame.html"),
            "execution_debug_html": str(output_dir / "policy_execution_debug.html"),
            "open_debug_html": bool(open_debug_html),
        }
    )
    pipeline["artifacts"] = artifacts
    real = dict(pipeline.get("real_execution", {}))
    real.update(
        {
            "enabled": True,
            "grasp_id": "",
            "attempt_artifact": str(output_dir / "real_pick_attempt.json"),
            "grasp_approach_controller": "d405_policy",
            "visual_servo_config": str(visual_path),
            "gripper_enabled": True,
            "gripper_client": "normalized_position",
            "gripper_trigger_open_service": "/left/gripper_controller/open",
            "gripper_trigger_close_service": "/left/gripper_controller/close",
            "gripper_trigger_stop_service": "/left/gripper_controller/stop",
            "gripper_position_command_topic": "/left/gripper_controller/position_command",
            "gripper_position_feedback_topic": "/left/gripper_controller/position",
            "gripper_position_feedback_tolerance": 0.02,
            "moveit_gripper_joint_name": moveit_gripper_joint,
            "pose_link": tcp_frame,
            "gripper_closed_width": SERVO_GRIPPER_CLOSED_WIDTH_M,
            "gripper_open_width": SERVO_GRIPPER_OPEN_WIDTH_M,
        }
    )
    pipeline["real_execution"] = real
    ros2_config = dict(pipeline.get("ros2", {}))
    # Perception publishes in shared dual-cell base_link. A standalone arm uses
    # its own link_0 frame: lbr_one is at y=-0.420, lbr_two at y=+0.420.
    ros2_config["position_offset_m"] = [
        0.0,
        0.420 if selected_side == "left" else -0.420,
        0.0,
    ]
    pipeline["ros2"] = ros2_config
    gripper_prefix = f"/{selected_side}/gripper_controller"
    real.update(
        {
            "gripper_trigger_open_service": f"{gripper_prefix}/open",
            "gripper_trigger_close_service": f"{gripper_prefix}/close",
            "gripper_trigger_stop_service": f"{gripper_prefix}/stop",
            "gripper_position_command_topic": f"{gripper_prefix}/position_command",
            "gripper_position_feedback_topic": f"{gripper_prefix}/position",
        }
    )
    pipeline["real_execution"] = real
    planning = dict(pipeline.get("planning", {}))
    maximum_contact_width = min(float(real["gripper_open_width"]), planning_max_width)
    minimum_contact_width = max(float(real["gripper_closed_width"]), planning_min_width)
    if maximum_contact_width <= minimum_contact_width:
        raise ValueError(
            "The configured gripper opening leaves no valid jaw-width range."
        )
    planning["max_jaw_width"] = maximum_contact_width
    planning["min_jaw_width"] = minimum_contact_width
    planning["gripper_collision_model"] = collision_model
    pipeline["planning"] = planning
    pipeline_path = output_dir / "pipeline.yaml"
    pipeline_path.write_text(yaml.safe_dump(pipeline, sort_keys=False), encoding="utf-8")
    return pipeline_path, visual_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", help="Policy registry name or explicit checkpoint path.")
    parser.add_argument("--part-id", default=None, help="Part to pick; defaults to registry part 0.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--pipeline-template", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--visual-servo-template", type=Path, default=DEFAULT_VISUAL_SERVO)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument(
        "--robot",
        choices=("left", "right"),
        default="left",
        help="Physical standalone arm/gripper side. Default: left (lbr_one).",
    )
    parser.add_argument(
        "--camera",
        default="realsense_1",
        help=(
            "RGB-D camera namespace or complete camera-node namespace. "
            "Default: realsense_1."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts/single_arm_policy_pickup",
    )
    parser.add_argument(
        "--no-planning-debug-gui",
        action="store_true",
        help="Do not open/refocus the live execution debug HTML.",
    )
    parser.add_argument("--list-policies", action="store_true")
    parser.add_argument(
        "--generate-config-only",
        action="store_true",
        help="Write resolved configs without perception, MoveIt planning, or robot execution.",
    )
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    registry_path = args.registry.expanduser().resolve()
    registry, asset_root = load_policy_registry(registry_path)
    policies = registry["policies"]
    assert isinstance(policies, Mapping)
    if args.list_policies:
        for name, raw in policies.items():
            record = dict(raw) if isinstance(raw, Mapping) else {}
            metrics = dict(record.get("validation_percent", {}))
            if metrics:
                result = (
                    f"nominal={float(metrics.get('nominal', 0.0)):4.1f}% "
                    f"clutter={float(metrics.get('clutter', 0.0)):4.1f}% "
                    f"depth={float(metrics.get('depth_errors', 0.0)):4.1f}%"
                )
            else:
                result = "held-out-validation=pending"
            print(
                f"{name:22s} context={str(record.get('policy_context', 'action')):21s} {result}"
            )
        return
    if not args.policy:
        parser.error("--policy is required unless --list-policies is used")

    policy_reference = str(args.policy).strip()
    policy_name, assets = resolve_policy_reference(
        policy_reference,
        registry_path=registry_path,
    )
    part_id = (
        str(args.part_id).strip()
        if args.part_id is not None
        else str(registry.get("default_part_id", "0")).strip()
    )
    part_mesh = _validate_part_id(part_id)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = args.output_root.expanduser().resolve() / f"{timestamp}-{policy_name}-part_{part_id}"
    pipeline_path, visual_path = write_resolved_configs(
        policy_name=policy_name,
        part_id=part_id,
        part_mesh=part_mesh,
        assets=assets,
        pipeline_template=args.pipeline_template.expanduser().resolve(),
        visual_servo_template=args.visual_servo_template.expanduser().resolve(),
        output_dir=output_dir,
        model_device=str(args.model_device),
        open_debug_html=not bool(args.no_planning_debug_gui),
        camera_name=str(args.camera),
        robot_side=str(args.robot),
    )
    robot_name = "lbr_one" if args.robot == "left" else "lbr_two"
    fri_peer = "192.170.10.2" if args.robot == "left" else "192.170.20.2"
    print(f"[SINGLE-PICK] arm={robot_name} FRI-peer={fri_peer} namespace=/lbr")
    print(f"[SINGLE-PICK] assembly=plumbers_block part={part_id} policy={policy_name}")
    print("[SINGLE-PICK] grasp=live stage-2 score order; no stored target or grasp whitelist")
    print(
        "[SINGLE-PICK] goal_rgbd=MuJoCo Filament on demand after "
        "collision-aware MoveIt grasp selection"
    )
    print(
        "[SINGLE-PICK] live_rgbd=compressed JPEG color + lossless compressedDepth; "
        "publisher_header_age=warning-only; local_receipt_timeout=0.50s"
    )
    print(
        f"[SINGLE-PICK] camera={camera_driver_root(str(args.camera))}; "
        "serial/intrinsics=diagnostic-only"
    )
    print("[SINGLE-PICK] TEST-ONLY robot_feedback_timeout=0.50s")
    print("[SINGLE-PICK] ros_executor=4-thread; GPU inference and feedback callbacks overlap")
    print(
        f"[SINGLE-PICK] embodiment={assets['gripper_model']}; "
        "tcp_feedback=TF lbr_link_0<-"
        f"{'pdz_gripper_tcp' if assets['gripper_model'] == 'pdz_gripper' else 'gripper_tcp'}"
    )
    print(
        f"[SINGLE-PICK] policy_context={assets['policy_context_mode']} "
        f"rate={float(assets['policy_rate_hz']):g}Hz; "
        f"camera_rotation={'live PDZ TF' if assets['gripper_model'] == 'pdz_gripper' else 'legacy calibrated link7 fallback'}"
    )
    print(
        f"[SINGLE-PICK] gripper=/{args.robot}/gripper_controller; stroke=7-74mm; "
        "position=closure_fraction"
    )
    print(
        f"[SINGLE-PICK] required_moveit=./run_pipeline.sh --mode real --robots {args.robot} "
        f"--bringup-only --servo --gripper-model {assets['gripper_model']}"
    )
    print(f"[SINGLE-PICK] pipeline_config={pipeline_path}")
    print(f"[SINGLE-PICK] visual_servo_config={visual_path}")
    print(f"[SINGLE-PICK] execution_debug_html={output_dir / 'policy_execution_debug.html'}")
    if args.generate_config_only:
        print("[SINGLE-PICK] Config generation only; no perception, planning, or robot command was started.")
        return
    print("[SINGLE-PICK] Starting real pickup; typed confirmation is still required before motion.")
    returncode = run_process_group(
        [
            str(REPO_ROOT / "run_pipeline.sh"),
            "--workflow",
            "single-object",
            "--mode",
            "real",
            "--robots",
            str(args.robot),
            "--config",
            str(pipeline_path),
        ],
        cwd=REPO_ROOT,
    )
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
