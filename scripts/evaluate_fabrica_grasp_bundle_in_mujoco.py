"""Batch-evaluate saved Fabrica grasps in MuJoCo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning import (
    accepted_grasps,
    build_pickup_pose_world,
    evaluate_saved_grasps_against_pickup_pose,
    load_grasp_bundle,
    sample_pickup_placement_spec,
    saved_grasp_to_world_grasp,
    score_grasps,
)
from grasp_planning.mujoco import (
    MujocoExecutionConfig,
    build_bundle_local_mesh,
    load_robot_config,
    run_world_grasp_in_mujoco,
    write_temporary_triangle_mesh_stl,
)


def _parse_vec2(raw: str) -> tuple[float, float]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 2:
        raise ValueError(f"Expected exactly 2 comma-separated values, got '{raw}'.")
    return values


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one comma-separated float, got '{raw}'.")
    return values


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one comma-separated token, got '{raw}'.")
    return values


def _resolve_placement_spec(args_cli):
    raise RuntimeError("bundle-aware placement resolution requires the input bundle.")


def _resolve_placement_spec_from_bundle(args_cli, bundle):
    from grasp_planning.grasping.fabrica_grasp_debug import PickupPlacementSpec

    metadata = dict(bundle.metadata)
    support_face = str(metadata.get("pickup_support_face", "")).strip()
    yaw_deg = metadata.get("pickup_yaw_deg")
    xy_world = metadata.get("pickup_xy_world", (0.0, 0.0))

    if support_face and yaw_deg is not None:
        base_spec = PickupPlacementSpec(
            support_face=support_face,
            yaw_deg=float(yaw_deg),
            xy_world=tuple(float(v) for v in xy_world),
        )
    else:
        rng = np.random.default_rng(args_cli.seed)
        base_spec = sample_pickup_placement_spec(
            rng=rng,
            allowed_support_faces=_parse_str_tuple(args_cli.allowed_support_faces),
            allowed_yaw_deg=_parse_float_tuple(args_cli.allowed_yaw_deg),
            xy_min_world=_parse_vec2(args_cli.xy_min_world),
            xy_max_world=_parse_vec2(args_cli.xy_max_world),
        )

    return PickupPlacementSpec(
        support_face=args_cli.support_face or base_spec.support_face,
        yaw_deg=float(args_cli.yaw_deg) if args_cli.yaw_deg is not None else float(base_spec.yaw_deg),
        xy_world=_parse_vec2(args_cli.xy_world) if args_cli.xy_world else tuple(float(v) for v in base_spec.xy_world),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True, help="Input grasp bundle from stage 1.")
    parser.add_argument("--robot-config", type=Path, required=True, help="MuJoCo robot binding JSON.")
    parser.add_argument("--support-face", type=str, default="", help="Optional explicit support face.")
    parser.add_argument("--yaw-deg", type=float, default=None, help="Optional explicit pickup yaw in degrees.")
    parser.add_argument("--xy-world", type=str, default="", help="Optional explicit world XY as x,y.")
    parser.add_argument(
        "--allowed-support-faces",
        type=str,
        default="pos_x,neg_x,pos_y,neg_y,neg_z",
        help="Comma-separated support faces used by the random sampler.",
    )
    parser.add_argument(
        "--allowed-yaw-deg",
        type=str,
        default="0,90,180,270",
        help="Comma-separated yaw values used by the random sampler.",
    )
    parser.add_argument("--xy-min-world", type=str, default="-0.45,-0.05", help="Random placement XY lower bound.")
    parser.add_argument("--xy-max-world", type=str, default="-0.35,0.05", help="Random placement XY upper bound.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--pregrasp-offset", type=float, default=0.20, help="Pregrasp offset in meters.")
    parser.add_argument(
        "--gripper-width-clearance",
        type=float,
        default=0.01,
        help="Clearance added to the saved grasp jaw width for the open approach width.",
    )
    parser.add_argument(
        "--contact-gap-m",
        type=float,
        default=0.002,
        help="Detailed Franka finger contact gap used during the ground recheck.",
    )
    parser.add_argument("--object-mass-kg", type=float, default=0.15, help="Target object mass in kg.")
    parser.add_argument("--object-scale", type=float, default=1.0, help="Uniform object mesh scale.")
    parser.add_argument("--lift-height-m", type=float, default=0.12, help="Vertical lift height after grasp.")
    parser.add_argument(
        "--success-height-margin-m",
        type=float,
        default=0.05,
        help="Required object Z gain for the attempt to count as success.",
    )
    parser.add_argument("--max-grasps", type=int, default=10, help="Maximum feasible grasps to evaluate.")
    parser.add_argument("--stop-on-success", action="store_true", help="Stop once one grasp succeeds.")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("artifacts/mujoco_grasp_evaluation_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--keep-generated-scene",
        action="store_true",
        help="Keep generated MuJoCo scene XML files for inspection.",
    )
    args_cli = parser.parse_args()

    bundle = load_grasp_bundle(args_cli.input_json)
    mesh_local = build_bundle_local_mesh(bundle)
    object_mesh_path = write_temporary_triangle_mesh_stl(mesh_local, prefix=f"{args_cli.input_json.stem}_bundle_local_")
    placement_spec = _resolve_placement_spec_from_bundle(args_cli, bundle)
    object_pose_world = build_pickup_pose_world(
        mesh_local,
        support_face=placement_spec.support_face,
        yaw_deg=placement_spec.yaw_deg,
        xy_world=placement_spec.xy_world,
    )
    statuses = evaluate_saved_grasps_against_pickup_pose(
        bundle.candidates,
        object_pose_world=object_pose_world,
        contact_gap_m=args_cli.contact_gap_m,
    )
    feasible = accepted_grasps(statuses)
    if not feasible:
        raise RuntimeError("No ground-feasible grasps remain for the requested pickup pose.")
    score_grasps(feasible, mesh_local=mesh_local)
    feasible_sorted = sorted(
        feasible, key=lambda grasp: float("-inf") if grasp.score is None else float(grasp.score), reverse=True
    )
    selected_grasps = feasible_sorted[: max(1, int(args_cli.max_grasps))]

    robot_cfg = load_robot_config(args_cli.robot_config)
    execution_cfg = MujocoExecutionConfig(
        object_mass_kg=args_cli.object_mass_kg,
        object_scale=float(args_cli.object_scale),
        lift_height_m=args_cli.lift_height_m,
        success_height_margin_m=args_cli.success_height_margin_m,
    )

    attempt_rows: list[dict[str, object]] = []
    try:
        for grasp in selected_grasps:
            world_grasp = saved_grasp_to_world_grasp(
                grasp,
                object_pose_world,
                pregrasp_offset=args_cli.pregrasp_offset,
                gripper_width_clearance=args_cli.gripper_width_clearance,
                frame_convention="mesh_grasp",
            )
            result = run_world_grasp_in_mujoco(
                robot_cfg=robot_cfg,
                execution_cfg=execution_cfg,
                object_mesh_path=object_mesh_path,
                object_pose_world=object_pose_world,
                world_grasp=world_grasp,
                keep_generated_scene=args_cli.keep_generated_scene,
            )
            row = {
                "grasp_id": grasp.grasp_id,
                "score": grasp.score,
                "success": result.success,
                "status": result.status,
                "message": result.message,
                "lift_height_m": result.lift_height_m,
                "target_lift_height_m": result.target_lift_height_m,
                "pregrasp_reached": result.pregrasp_reached,
                "grasp_reached": result.grasp_reached,
                "generated_scene_xml": result.generated_scene_xml,
            }
            attempt_rows.append(row)
            print(
                f"[INFO]: grasp_id={grasp.grasp_id} success={result.success} status={result.status} "
                f"lift_height_m={result.lift_height_m:.4f}",
                flush=True,
            )
            if args_cli.stop_on_success and result.success:
                break
    finally:
        if not args_cli.keep_generated_scene:
            try:
                object_mesh_path.unlink()
            except FileNotFoundError:
                pass

    payload = {
        "input_json": str(args_cli.input_json),
        "robot_config": str(args_cli.robot_config),
        "placement": {
            "support_face": placement_spec.support_face,
            "yaw_deg": placement_spec.yaw_deg,
            "xy_world": list(placement_spec.xy_world),
            "object_position_world": list(object_pose_world.position_world),
            "object_orientation_xyzw_world": list(object_pose_world.orientation_xyzw_world),
        },
        "num_saved_grasps": len(bundle.candidates),
        "num_ground_feasible_grasps": len(feasible),
        "attempts": attempt_rows,
    }
    args_cli.report_json.parent.mkdir(parents=True, exist_ok=True)
    args_cli.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO]: Wrote MuJoCo evaluation report to {args_cli.report_json}", flush=True)


if __name__ == "__main__":
    main()
