#!/usr/bin/env python3
"""Resolve ranked holder/inserter pairs into a real preflight task artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.pipeline.dual_robot_pair_scoring import (  # noqa: E402
    MovableFrame,
)
from grasp_planning.pipeline.dual_robot_simple_sim import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_FLOOR_Z_WORLD_M,
    DEFAULT_RUNTIME_PAIR_CANDIDATE_LIMIT,
    load_simple_dual_robot_pair_tasks,
    resolve_dual_robot_step_selection,
    simple_dual_robot_pregrasp_aabb_obstacles,
    simple_dual_robot_pregrasp_aabb_schedule,
)


def _include_nonretained_identity_fallbacks(pair_id: str) -> bool:
    """Fill the default queue with collision-validated Stage-3 fallbacks."""

    return not bool(pair_id)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--assembly", default=None)
    parser.add_argument("--incoming-part-id", default=None)
    parser.add_argument("--step-id", default=None)
    parser.add_argument("--pair-id", default="")
    parser.add_argument(
        "--max-pair-candidates",
        type=int,
        default=DEFAULT_RUNTIME_PAIR_CANDIDATE_LIMIT,
        help=(
            "Maximum pair/transition candidates saved for pre-motion real IK "
            "fallback. Fully transition-validated candidates come first, then "
            "other explicitly validated transitions and canonical identity-only "
            "pairs fill the queue after the actual pickup-pose floor check. A "
            "fixed --pair-id keeps all of that pair's transitions up to this limit."
        ),
    )
    parser.add_argument("--assembly-x", type=float, default=0.55)
    parser.add_argument("--assembly-y", type=float, default=0.0)
    parser.add_argument(
        "--assembly-z",
        type=float,
        default=None,
        help="Defaults to --floor-z when no perceived assembly Z is supplied.",
    )
    parser.add_argument("--assembly-yaw-deg", type=float, default=0.0)
    parser.add_argument("--pickup-x", type=float, default=0.55)
    parser.add_argument("--pickup-y", type=float, default=0.28)
    parser.add_argument("--pickup-roll-deg", type=float, default=0.0)
    parser.add_argument("--pickup-pitch-deg", type=float, default=0.0)
    parser.add_argument("--pickup-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--floor-z",
        type=float,
        default=DEFAULT_FLOOR_Z_WORLD_M,
    )
    parser.add_argument("--transport-clearance-m", type=float, default=0.08)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_pair_candidates < 1:
        raise ValueError("--max-pair-candidates must be at least 1.")
    assembly_z = float(args.floor_z) if args.assembly_z is None else float(args.assembly_z)
    selection = resolve_dual_robot_step_selection(
        assembly=args.assembly,
        incoming_part_id=args.incoming_part_id,
        artifact_root=args.artifact_root,
        artifact_dir=args.artifact_dir,
        step_id=args.step_id,
    )
    tasks = list(
        load_simple_dual_robot_pair_tasks(
            artifact_dir=selection.artifact_dir,
            step_id=selection.step_id,
            assembly_world=MovableFrame(
                (
                    float(args.assembly_x),
                    float(args.assembly_y),
                    assembly_z,
                ),
                float(args.assembly_yaw_deg),
            ),
            pickup_source_world_xy=(
                float(args.pickup_x),
                float(args.pickup_y),
            ),
            pickup_orientation_rpy_deg=(
                float(args.pickup_roll_deg),
                float(args.pickup_pitch_deg),
                float(args.pickup_yaw_deg),
            ),
            pickup_floor_z_world_m=float(args.floor_z),
            transport_clearance_m=float(args.transport_clearance_m),
            retained_only=False,
            include_nonretained_identity_fallbacks=(_include_nonretained_identity_fallbacks(str(args.pair_id))),
        )
    )
    if args.pair_id:
        tasks = [task for task in tasks if task.pair_id == str(args.pair_id)]
    tasks = tasks[: int(args.max_pair_candidates)]
    if not tasks:
        requested = str(args.pair_id) or "the ranked compatible set"
        raise RuntimeError(f"No accepted dual task was found for {requested} at {selection.step_id}.")

    candidate_payloads: list[dict[str, object]] = []
    for rank, task in enumerate(tasks, start=1):
        pregrasp_aabb_obstacles = simple_dual_robot_pregrasp_aabb_obstacles(task)
        moveit_payload = {
            "namespace": "/lbr_dual_arm",
            "frame_id": "base_link",
            "object_collision_geometry_in_scene": False,
            "pregrasp_aabb_collision_geometry": {
                "representation": ("object_world_aabb_minus_selected_gripper_sweep"),
                "obstacles": pregrasp_aabb_obstacles,
                "active_by_target": (simple_dual_robot_pregrasp_aabb_schedule(pregrasp_aabb_obstacles)),
                "removed_before_grasp_approach": True,
            },
        }
        candidate_payload = task.to_payload()
        candidate_payload["candidate_rank"] = rank
        candidate_payload["generated_by"] = "scripts/build_simple_dual_robot_task.py"
        candidate_payload["target_only"] = True
        candidate_payload["moveit"] = moveit_payload
        candidate_payloads.append(candidate_payload)

    payload = dict(candidate_payloads[0])
    payload["ranked_pair_candidates"] = candidate_payloads
    payload["real_pair_selection"] = {
        "policy": "strict_score_order_collision_aware_ik_before_motion",
        "candidate_count": len(candidate_payloads),
        "maximum_candidate_count": int(args.max_pair_candidates),
        "fixed_pair_requested": bool(args.pair_id),
        "default_candidate_scope": ("retained_transition_validated_then_canonical_identity_fallbacks"),
        "pickup_floor_check_scope": "actual_supplied_pickup_pose_and_orientation",
        "candidate_identity": "execution_candidate_id",
        "fallback_scope": ("ranked grasp-pair and symmetry-transition targets before motion"),
    }
    output = (
        selection.artifact_dir / f"simple_dual_robot_real_task_{selection.step_id}.json"
        if args.output is None
        else args.output.expanduser().resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[DUAL-TASK] pair={payload['pair_id']} score={payload['pair_score']:.4f} "
        f"candidates={len(candidate_payloads)} wrote={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
