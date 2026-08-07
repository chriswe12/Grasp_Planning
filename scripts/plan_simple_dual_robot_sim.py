#!/usr/bin/env python3
"""Plan a simple holder plus pickup-to-preinsertion sequence with dual MoveIt."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.pipeline.dual_robot_pair_scoring import (  # noqa: E402
    MovableFrame,
)
from grasp_planning.pipeline.dual_robot_planning_debug import (  # noqa: E402
    DualRobotPlanningDebugServer,
)
from grasp_planning.pipeline.dual_robot_simple_sim import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_FLOOR_Z_WORLD_M,
    DEFAULT_HOLDER_BASE_WORLD,
    DEFAULT_INSERTER_BASE_WORLD,
    DEFAULT_RUNTIME_PAIR_CANDIDATE_LIMIT,
    load_simple_dual_robot_pair_tasks,
    resolve_dual_robot_step_selection,
    simple_dual_robot_pregrasp_aabb_obstacles,
    simple_dual_robot_pregrasp_aabb_schedule,
)
from grasp_planning.ros2.moveit_pose_commander import (  # noqa: E402
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    PoseTarget,
    rclpy,
)
from grasp_planning.ros2.multi_ik_planner import (  # noqa: E402
    MultiIkPlanningConfig,
    plan_pose_sequence_multi_ik,
)
from grasp_planning.start_poses import (  # noqa: E402
    KUKA_MOVEIT_ARM_START_JOINT_VALUES,
)

MOVEIT_START_JOINT_POSITIONS = KUKA_MOVEIT_ARM_START_JOINT_VALUES
ARM_SPEC_BY_ROBOT = {
    "lbr_one": {
        "planning_group": "arm_one",
        "pose_link": "lbr_one_gripper_tcp",
        "joint_names": tuple(f"lbr_one_A{index}" for index in range(1, 8)),
    },
    "lbr_two": {
        "planning_group": "arm_two",
        "pose_link": "lbr_two_gripper_tcp",
        "joint_names": tuple(f"lbr_two_A{index}" for index in range(1, 8)),
    },
}
ARM_SPECS = {
    "holder": dict(ARM_SPEC_BY_ROBOT["lbr_one"]),
    "inserter": dict(ARM_SPEC_BY_ROBOT["lbr_two"]),
}
TARGET_SEQUENCE = (
    ("holder", "holder_pregrasp"),
    ("holder", "holder_grasp"),
    ("inserter", "inserter_pickup_pregrasp"),
    ("inserter", "inserter_pickup_grasp"),
    ("inserter", "inserter_pickup_lift"),
    ("inserter", "inserter_above_preinsertion"),
    ("inserter", "inserter_preinsertion"),
)
WORK_SURFACE_SIZE_M = (1.20, 1.40, 0.05)
WORK_SURFACE_CENTER_XY_M = (0.75, 0.0)
INSERTER_PICKUP_SEQUENCE = (
    "inserter_pickup_pregrasp",
    "inserter_pickup_grasp",
    "inserter_pickup_lift",
)
INSERTER_TRANSITION_SEQUENCE = (
    "inserter_above_preinsertion",
    "inserter_preinsertion",
)
# Mirrors ros2_ws/.../dual_iiwa7_y_gripper_moveit.urdf.xacro.
KUKA_MOVEIT_JOINT_LOWER_LIMITS = (-2.97, -2.09, -2.97, -2.09, -2.97, -2.09, -3.05)
KUKA_MOVEIT_JOINT_UPPER_LIMITS = (2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05)
KUKA_MOVEIT_JOINT_MAX_VELOCITIES = (1.71, 1.71, 1.75, 2.27, 2.44, 3.14, 3.14)
KUKA_MOVEIT_TRANSITION_JOINT_WEIGHTS = tuple(1.0 / velocity for velocity in KUKA_MOVEIT_JOINT_MAX_VELOCITIES)
KUKA_A7_HALF_TURN_SEED_OFFSETS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.pi),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -math.pi),
)
KUKA_A7_NEAR_LIMIT_BRANCH_RAD = 3.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Root containing per-assembly dual planning artifacts.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Explicit per-assembly artifact directory compatibility override.",
    )
    parser.add_argument(
        "--assembly",
        default=None,
        help="Assembly name under --artifact-root (for example plumbers_block).",
    )
    parser.add_argument(
        "--incoming-part-id",
        default=None,
        help="Resolve the selected-order step that inserts this part.",
    )
    parser.add_argument(
        "--step-id",
        default=None,
        help="Explicit step compatibility override.",
    )
    parser.add_argument("--pair-id", default="")
    parser.add_argument(
        "--holder-grasp-id",
        default="",
        help="Restrict planning to pairs using this holder grasp.",
    )
    parser.add_argument(
        "--holder-only",
        action="store_true",
        help="Plan only holder pregrasp and grasp trajectories.",
    )
    parser.add_argument(
        "--holder-start-joint-positions",
        type=float,
        nargs=7,
        default=None,
        metavar=("A1", "A2", "A3", "A4", "A5", "A6", "A7"),
        help=(
            "MoveIt holder start state. Holder-only stress sequences use the "
            "previous case terminal state instead of the nominal reset pose."
        ),
    )
    parser.add_argument(
        "--max-pair-attempts",
        type=int,
        default=DEFAULT_RUNTIME_PAIR_CANDIDATE_LIMIT,
    )
    parser.add_argument("--assembly-x", type=float, default=0.55)
    parser.add_argument("--assembly-y", type=float, default=0.0)
    parser.add_argument(
        "--assembly-z",
        type=float,
        default=None,
        help=(
            "Assembly support-plane Z. Defaults to --floor-z so the assembled "
            "prefix rests on the same surface used by MoveIt and Isaac."
        ),
    )
    parser.add_argument("--assembly-yaw-deg", type=float, default=0.0)
    parser.add_argument("--pickup-x", type=float, default=0.55)
    parser.add_argument("--pickup-y", type=float, default=0.28)
    parser.add_argument("--pickup-roll-deg", type=float, default=0.0)
    parser.add_argument("--pickup-pitch-deg", type=float, default=0.0)
    parser.add_argument("--pickup-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--inserter-arm",
        choices=("auto", "lbr_one", "lbr_two"),
        default="lbr_two",
        help=(
            "Physical arm assigned to incoming-part pickup. 'auto' selects "
            "lbr_one for pickup Y below assembly Y and lbr_two otherwise."
        ),
    )
    parser.add_argument(
        "--floor-z",
        type=float,
        default=DEFAULT_FLOOR_Z_WORLD_M,
    )
    parser.add_argument(
        "--transport-clearance-m",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--pickup-floor-clearance-margin-m",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--pickup-top-down-score-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--all-compatible",
        action="store_true",
        help="Compatibility flag; all checked-compatible Stage-3 pairs are now searched by default.",
    )
    parser.add_argument(
        "--retained-only",
        action="store_true",
        help="Restrict online fallback to the small Stage-3 retained subset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plan JSON (default: selected assembly artifact directory).",
    )
    parser.add_argument("--moveit-namespace", default="/lbr_dual_arm")
    parser.add_argument("--planning-time-s", type=float, default=5.0)
    parser.add_argument("--planning-attempts", type=int, default=8)
    parser.add_argument("--velocity-scale", type=float, default=0.35)
    parser.add_argument("--acceleration-scale", type=float, default=0.35)
    parser.add_argument(
        "--skip-ik-preflight",
        action="store_true",
        help="Skip the fast per-grasp IK screen before full trajectory attempts.",
    )
    parser.add_argument(
        "--skip-joint-space-ranking",
        action="store_true",
        help="Keep the retained Stage-3 order without MoveIt joint-path pre-ranking.",
    )
    parser.add_argument(
        "--joint-rank-candidates",
        type=int,
        default=8,
        help="Number of retained execution candidates to pre-plan and rank in joint space.",
    )
    parser.add_argument(
        "--joint-rank-ik-candidates",
        type=int,
        default=4,
        help="Seeded IK solutions considered per target during joint-space ranking.",
    )
    parser.add_argument(
        "--joint-rank-beam-width",
        type=int,
        default=1,
        help="Partial joint paths retained per transition target during ranking.",
    )
    parser.add_argument(
        "--debug-gui",
        action="store_true",
        help=("Open a localhost browser GUI showing the active grasp pair, world-frame parts, and planning phase."),
    )
    parser.add_argument(
        "--debug-gui-port",
        type=int,
        default=0,
        help="Local debug server port; 0 selects an available port.",
    )
    return parser.parse_args()


def _work_surface_obstacle(*, floor_z_world_m: float) -> dict[str, object]:
    return {
        "id": "dual_sim_work_surface",
        "type": "box",
        "frame_id": "base_link",
        "size_m": list(WORK_SURFACE_SIZE_M),
        "xyz": [
            WORK_SURFACE_CENTER_XY_M[0],
            WORK_SURFACE_CENTER_XY_M[1],
            float(floor_z_world_m) - 0.5 * WORK_SURFACE_SIZE_M[2],
        ],
        "rpy": [0.0, 0.0, 0.0],
    }


def _pregrasp_aabb_obstacles_for_target(
    obstacles: dict[str, dict[str, object]],
    *,
    target_name: str,
) -> list[dict[str, object]]:
    schedule = simple_dual_robot_pregrasp_aabb_schedule(obstacles)
    return [obstacles[key] for key in schedule.get(target_name, ())]


def _pose_target(raw: dict[str, object]) -> PoseTarget:
    position = tuple(
        float(value)
        for value in raw["position_world_m"]  # type: ignore[index]
    )
    orientation = tuple(
        float(value)
        for value in raw["orientation_xyzw_world"]  # type: ignore[index]
    )
    return PoseTarget.from_quaternion(
        x=position[0],
        y=position[1],
        z=position[2],
        quaternion_xyzw=orientation,
        frame_id="base_link",
    )


def _trajectory_payload(
    trajectory,
    *,
    expected_joint_names: tuple[str, ...],
) -> dict[str, object]:
    trajectory_message = trajectory.joint_trajectory
    source_names = tuple(str(name) for name in trajectory_message.joint_names)
    missing = [joint_name for joint_name in expected_joint_names if joint_name not in source_names]
    if missing:
        raise RuntimeError(f"MoveIt trajectory is missing expected joints: {missing}")
    indices = tuple(source_names.index(name) for name in expected_joint_names)
    points = []
    for point in trajectory_message.points:
        points.append([float(point.positions[index]) for index in indices])
    if not points:
        raise RuntimeError("MoveIt returned a trajectory without points.")
    return {
        "joint_names": list(expected_joint_names),
        "waypoints": points,
    }


def _multi_ik_joint_targets(plan) -> dict[str, tuple[float, ...]]:
    targets: dict[str, tuple[float, ...]] = {}
    for label, waypoints in plan.trajectories.items():
        if not waypoints:
            raise RuntimeError(f"Joint-space ranking returned no waypoints for '{label}'.")
        targets[str(label)] = tuple(float(value) for value in waypoints[-1])
    return targets


def _pickup_target_signature(task) -> tuple[object, ...]:
    targets = dict(task.to_payload()["targets"])
    return (
        task.inserter_candidate.grasp_id,
        *(
            round(float(value), 9)
            for label in INSERTER_PICKUP_SEQUENCE
            for field in ("position_world_m", "orientation_xyzw_world")
            for value in dict(targets[label])[field]
        ),
    )


def _transition_corridor_key(task) -> str:
    raw_transition = getattr(task, "transition_symmetry", {})
    transition = dict(raw_transition) if isinstance(raw_transition, dict) else {}
    raw_vector = transition.get("pre_to_final_translation_assembly_m")
    try:
        vector = np.asarray(raw_vector, dtype=float)
    except (TypeError, ValueError):
        return "corridor_unknown"
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return "corridor_unknown"
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        direction = np.zeros(3, dtype=float)
    else:
        direction = vector / norm
    return "corridor_" + "_".join(f"{float(value):+.6f}" for value in direction)


def _transition_crosses_holder_corridor(task) -> bool:
    components = getattr(task, "layout_proxy_components", {})
    if not isinstance(components, dict):
        return False
    return bool(components.get("transition_segments_cross_xy", False))


def _corridor_diverse_joint_rank_pool(tasks: list, *, limit: int) -> list:
    """Select the cheap preplan pool round-robin across insertion corridors."""

    bounded_limit = max(0, min(int(limit), len(tasks)))
    if bounded_limit == 0:
        return []
    by_corridor: dict[str, list] = {}
    for task in tasks:
        by_corridor.setdefault(_transition_corridor_key(task), []).append(task)
    corridor_order = tuple(by_corridor)
    selected: list = []
    depth = 0
    while len(selected) < bounded_limit:
        added = False
        for corridor_key in corridor_order:
            corridor_tasks = by_corridor[corridor_key]
            if depth >= len(corridor_tasks):
                continue
            selected.append(corridor_tasks[depth])
            added = True
            if len(selected) >= bounded_limit:
                break
        if not added:
            break
        depth += 1
    return selected


def _a7_transition_seed_offsets(
    start_joint_positions: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    """Add valid near-limit A7 branches when a literal half-turn is bounded."""

    if len(start_joint_positions) != 7:
        raise ValueError(f"Expected seven iiwa start joints, got {len(start_joint_positions)}.")
    current_a7 = float(start_joint_positions[-1])
    branch_offsets = []
    for target_a7 in (KUKA_A7_NEAR_LIMIT_BRANCH_RAD, -KUKA_A7_NEAR_LIMIT_BRANCH_RAD):
        branch_offsets.append(
            (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                target_a7 - current_a7,
            )
        )
    return (*KUKA_A7_HALF_TURN_SEED_OFFSETS, *branch_offsets)


def _rank_tasks_by_inserter_joint_path(
    tasks: list,
    *,
    commander,
    candidate_limit: int,
    ik_candidate_count: int,
    beam_width: int,
    update_debug=None,
) -> tuple[list, dict[str, object], dict[str, dict[str, tuple[float, ...]]]]:
    """Pre-plan pickup/transition IK chains and prefer the cheapest transition."""

    checked_tasks = _corridor_diverse_joint_rank_pool(
        tasks,
        limit=candidate_limit,
    )
    if not checked_tasks:
        return (
            tasks,
            {
                "skipped": True,
                "reason": "candidate_limit_is_zero",
                "candidate_count_before": len(tasks),
            },
            {},
        )

    common_config = {
        "candidate_count": max(1, int(ik_candidate_count)),
        "beam_width": max(1, int(beam_width)),
        "seed_perturbation_rad": 0.35,
        "dedup_tolerance_rad": 0.05,
        "joint_weights": KUKA_MOVEIT_TRANSITION_JOINT_WEIGHTS,
        "joint_lower_limits_rad": KUKA_MOVEIT_JOINT_LOWER_LIMITS,
        "joint_upper_limits_rad": KUKA_MOVEIT_JOINT_UPPER_LIMITS,
        "continuous_joints": (False,) * 7,
    }
    pickup_config = MultiIkPlanningConfig(**common_config)
    joint_names = tuple(str(value) for value in ARM_SPECS["inserter"]["joint_names"])
    pickup_cache: dict[tuple[object, ...], tuple[object | None, str]] = {}
    preferred_joint_targets: dict[str, dict[str, tuple[float, ...]]] = {}
    planned: list[tuple[bool, float, float, int, object]] = []
    failed: list[tuple[int, object]] = []
    records: list[dict[str, object]] = []

    for original_rank, task in enumerate(checked_tasks, start=1):
        if update_debug is not None:
            update_debug(
                task=task,
                attempt_index=original_rank,
                phase="joint_space_ranking",
                status="planning",
                message=(
                    "Pre-planning pickup-to-pre-insertion joint paths with "
                    "bounded A7 half-turn and near-limit branch seeds."
                ),
            )
        payload = task.to_payload()
        raw_targets = dict(payload["targets"])
        pose_targets = {
            label: _pose_target(dict(raw_targets[label]))
            for label in (*INSERTER_PICKUP_SEQUENCE, *INSERTER_TRANSITION_SEQUENCE)
        }
        pickup_key = _pickup_target_signature(task)
        pickup_cache_hit = pickup_key in pickup_cache
        if pickup_cache_hit:
            pickup_plan, pickup_failure = pickup_cache[pickup_key]
        else:
            try:
                pickup_plan = plan_pose_sequence_multi_ik(
                    commander,
                    targets=pose_targets,
                    labels=INSERTER_PICKUP_SEQUENCE,
                    start_joint_positions=MOVEIT_START_JOINT_POSITIONS,
                    joint_names=joint_names,
                    config=pickup_config,
                    label_prefix=f"joint_rank_{task.inserter_candidate.grasp_id}_pickup",
                )
                pickup_failure = ""
            except (RuntimeError, ValueError) as exc:
                pickup_plan = None
                pickup_failure = str(exc)
            pickup_cache[pickup_key] = (pickup_plan, pickup_failure)

        record: dict[str, object] = {
            "original_rank": original_rank,
            "pair_id": task.pair_id,
            "transition_id": task.transition_id,
            "execution_candidate_id": task.execution_candidate_id,
            "corridor_key": _transition_corridor_key(task),
            "transition_segments_cross_xy": _transition_crosses_holder_corridor(task),
            "selection_score": float(task.selection_score),
            "pickup_cache_hit": pickup_cache_hit,
        }
        if pickup_plan is None:
            record.update(
                {
                    "status": "failed",
                    "failure": pickup_failure,
                    "failed_phase": "pickup",
                }
            )
            records.append(record)
            failed.append((original_rank, task))
            if update_debug is not None:
                update_debug(
                    task=task,
                    attempt_index=original_rank,
                    phase="joint_space_ranking",
                    status="failed",
                    message=pickup_failure,
                )
            continue

        transition_seed_offsets = _a7_transition_seed_offsets(
            pickup_plan.terminal_joint_positions,
        )
        transition_config = MultiIkPlanningConfig(
            **common_config,
            seed_offsets_rad=transition_seed_offsets,
        )
        try:
            transition_plan = plan_pose_sequence_multi_ik(
                commander,
                targets=pose_targets,
                labels=INSERTER_TRANSITION_SEQUENCE,
                start_joint_positions=pickup_plan.terminal_joint_positions,
                joint_names=joint_names,
                config=transition_config,
                label_prefix=f"joint_rank_{task.execution_candidate_id}",
            )
        except (RuntimeError, ValueError) as exc:
            failure = str(exc)
            record.update(
                {
                    "status": "failed",
                    "failure": failure,
                    "failed_phase": "transition",
                    "pickup_joint_path_cost": float(pickup_plan.joint_path_cost),
                }
            )
            records.append(record)
            failed.append((original_rank, task))
            if update_debug is not None:
                update_debug(
                    task=task,
                    attempt_index=original_rank,
                    phase="joint_space_ranking",
                    status="failed",
                    message=failure,
                )
            continue

        transition_cost = float(transition_plan.joint_path_cost)
        preferred_joint_targets[task.execution_candidate_id] = {
            **_multi_ik_joint_targets(pickup_plan),
            **_multi_ik_joint_targets(transition_plan),
        }
        record.update(
            {
                "status": "planned",
                "failure": "",
                "pickup_joint_path_cost": float(pickup_plan.joint_path_cost),
                "transition_joint_path_cost": transition_cost,
                "a7_seed_offsets_rad": [list(offset) for offset in transition_seed_offsets],
                "pickup_diagnostics": list(pickup_plan.diagnostics),
                "transition_diagnostics": list(transition_plan.diagnostics),
            }
        )
        records.append(record)
        planned.append(
            (
                _transition_crosses_holder_corridor(task),
                transition_cost,
                -float(task.selection_score),
                original_rank,
                task,
            )
        )
        if update_debug is not None:
            update_debug(
                task=task,
                attempt_index=original_rank,
                phase="joint_space_ranking",
                status="succeeded",
                message=f"Transition joint-path cost {transition_cost:.4f}.",
            )

    planned.sort(key=lambda item: item[:4])
    checked_object_ids = {id(task) for task in checked_tasks}
    unranked = [task for task in tasks if id(task) not in checked_object_ids]
    planned_tasks = [item[4] for item in planned]
    failed_tasks = [item[1] for item in failed]

    def crossing_phase(crosses: bool) -> list:
        return [
            *(task for task in planned_tasks if _transition_crosses_holder_corridor(task) is crosses),
            *(task for task in unranked if _transition_crosses_holder_corridor(task) is crosses),
            *(task for task in failed_tasks if _transition_crosses_holder_corridor(task) is crosses),
        ]

    # A successful cheap pre-plan is useful evidence, but it must never move a
    # crossed corridor ahead of an unchecked or cheap-preplan-failed clear
    # corridor. Exact shared-scene planning remains the authority inside each
    # phase; crossing is entered only after the bounded clear phase is spent.
    ordered_tasks = crossing_phase(False) + crossing_phase(True)
    final_rank_by_id = {task.execution_candidate_id: rank for rank, task in enumerate(ordered_tasks, start=1)}
    for record in records:
        record["final_rank"] = final_rank_by_id[str(record["execution_candidate_id"])]
    diagnostics = {
        "skipped": False,
        "mode": "seeded_multi_ik_transition_joint_path",
        "candidate_count_before": len(tasks),
        "candidate_count_checked": len(checked_tasks),
        "candidate_count_planned": len(planned),
        "candidate_count_failed": len(failed),
        "unranked_fallback_count": len(unranked),
        "pool_selection": "round_robin_insertion_corridor",
        "corridors_checked": list(dict.fromkeys(_transition_corridor_key(task) for task in checked_tasks)),
        "primary_sort": "strict_noncrossing_phase_then_preplan_status_then_transition_joint_path_cost",
        "tie_breaker": "selection_score_within_preplan_status",
        "joint_weights_inverse_max_velocity": list(KUKA_MOVEIT_TRANSITION_JOINT_WEIGHTS),
        "bounded_joint_limits": {
            "lower_rad": list(KUKA_MOVEIT_JOINT_LOWER_LIMITS),
            "upper_rad": list(KUKA_MOVEIT_JOINT_UPPER_LIMITS),
            "continuous": [False] * 7,
        },
        "a7_half_turn_seed_offsets_rad": [list(offset) for offset in KUKA_A7_HALF_TURN_SEED_OFFSETS],
        "a7_near_limit_branch_targets_rad": [
            KUKA_A7_NEAR_LIMIT_BRANCH_RAD,
            -KUKA_A7_NEAR_LIMIT_BRANCH_RAD,
        ],
        "records": records,
    }
    return ordered_tasks, diagnostics, preferred_joint_targets


def _commander(
    *,
    role: str,
    moveit_namespace: str,
    args: argparse.Namespace,
) -> MoveItPoseCommander:
    spec = ARM_SPECS[role]
    return MoveItPoseCommander(
        MoveItPoseCommanderConfig(
            planning_group=str(spec["planning_group"]),
            pose_link=str(spec["pose_link"]),
            joint_names=tuple(spec["joint_names"]),
            moveit_namespace=moveit_namespace,
            planning_time_s=float(args.planning_time_s),
            num_planning_attempts=int(args.planning_attempts),
            velocity_scale=float(args.velocity_scale),
            acceleration_scale=float(args.acceleration_scale),
            execute_timeout_s=60.0,
        ),
        node_name=f"simple_dual_sim_{role}",
    )


def _plan_and_execute(
    commander: MoveItPoseCommander,
    *,
    target: PoseTarget,
    label: str,
    expected_joint_names: tuple[str, ...],
    preferred_joint_positions: tuple[float, ...] | None = None,
) -> tuple[dict[str, object] | None, str]:
    if preferred_joint_positions is None:
        trajectory, message = commander.plan_to_pose(target, label=label)
    else:
        trajectory, message = commander.plan_to_joint_positions(
            preferred_joint_positions,
            label=f"{label}_joint_ranked",
        )
        if trajectory is None:
            pose_trajectory, pose_message = commander.plan_to_pose(
                target,
                label=f"{label}_pose_fallback",
            )
            trajectory = pose_trajectory
            message = f"preferred joint target failed ({message}); pose fallback: {pose_message}"
    if trajectory is None and message.startswith("IK failed"):
        fallback_joints, fallback_message = commander.compute_ik(
            target,
            seed_joint_positions=MOVEIT_START_JOINT_POSITIONS,
        )
        if fallback_joints is not None:
            trajectory, fallback_plan_message = commander.plan_to_joint_positions(
                fallback_joints,
                label=f"{label}_alternate_ik",
            )
            message = (
                f"current-state IK failed; alternate start-seeded IK "
                f"{'planned' if trajectory is not None else 'could not plan'}: "
                f"{fallback_plan_message}"
            )
        else:
            message = f"{message}; alternate start-seeded IK also failed: {fallback_message}"
    if trajectory is None:
        return None, message
    payload = _trajectory_payload(
        trajectory,
        expected_joint_names=expected_joint_names,
    )
    ok, execution_message = commander.execute_trajectory(
        trajectory,
        label=label,
    )
    if not ok:
        return None, execution_message
    return payload, execution_message


def _reset_arm(
    commander: MoveItPoseCommander,
    *,
    role: str,
    joint_positions: tuple[float, ...] = MOVEIT_START_JOINT_POSITIONS,
) -> tuple[bool, str]:
    trajectory, message = commander.plan_to_joint_positions(
        joint_positions,
        label=f"{role}_reset",
    )
    if trajectory is None:
        return False, message
    return commander.execute_trajectory(
        trajectory,
        label=f"{role}_reset",
    )


def _reset_active_roles(
    commanders: dict[str, MoveItPoseCommander],
    *,
    active_roles: tuple[str, ...],
    holder_start_joint_positions: tuple[float, ...] | None = None,
    recovering_from_candidate: bool,
) -> tuple[bool, dict[str, str]]:
    """Return both arms to known starts in a collision-aware order.

    After a partially executed candidate, the inserter is the arm deep in the
    shared workspace. Retracting it first clears the holder's route home. The
    initial reset retains normal holder/inserter order because neither arm has
    entered a candidate trajectory yet.
    """

    reset_order = tuple(reversed(active_roles)) if recovering_from_candidate else active_roles
    messages: dict[str, str] = {}
    all_ok = True
    for role in reset_order:
        reset_positions = (
            holder_start_joint_positions
            if role == "holder" and holder_start_joint_positions is not None
            else MOVEIT_START_JOINT_POSITIONS
        )
        ok, message = _reset_arm(
            commanders[role],
            role=role,
            joint_positions=reset_positions,
        )
        messages[role] = message
        all_ok = all_ok and ok
    return all_ok, messages


IK_PREFLIGHT_TARGETS = {
    "holder": ("holder_pregrasp", "holder_grasp"),
    "inserter": (
        "inserter_pickup_pregrasp",
        "inserter_pickup_grasp",
        "inserter_pickup_lift",
        "inserter_above_preinsertion",
        "inserter_preinsertion",
    ),
}


def _new_ik_preflight_state(*, pair_task_count: int) -> dict[str, object]:
    return {
        "skipped": False,
        "mode": "lazy_strict_score_order",
        "scope": (
            "Each ranked pair is collision-aware IK screened from the shared "
            "mock start state immediately before its full sequential plan. "
            "Results are cached by role, grasp ID, and exact target poses."
        ),
        "pair_tasks_before": int(pair_task_count),
        "pair_tasks_checked": 0,
        "pair_tasks_after": 0,
        "holder_grasps_checked": 0,
        "holder_grasps_feasible": 0,
        "inserter_grasps_checked": 0,
        "inserter_grasps_feasible": 0,
        "records": {role: [] for role in IK_PREFLIGHT_TARGETS},
        "pair_records": [],
    }


def _ik_preflight_pair(
    task,
    *,
    commanders: dict[str, MoveItPoseCommander],
    feasible_cache: dict[
        str,
        dict[tuple[str, tuple[float, ...]], bool],
    ],
    state: dict[str, object],
    rank: int,
    roles: tuple[str, ...] = ("holder", "inserter"),
    preferred_joint_targets: dict[str, tuple[float, ...]] | None = None,
) -> tuple[bool, str]:
    """Screen one ranked pair and stop immediately on its first failed role."""

    task_payload = task.to_payload()
    targets = dict(task_payload["targets"])
    grasp_ids = {
        "holder": task.holder_candidate.grasp_id,
        "inserter": task.inserter_candidate.grasp_id,
    }
    pair_role_records: dict[str, object] = {}
    preferred_joint_targets = preferred_joint_targets or {}
    failure = ""
    for role in roles:
        target_names = IK_PREFLIGHT_TARGETS[role]
        grasp_id = grasp_ids[role]
        target_signature = tuple(
            round(float(value), 9)
            for target_name in target_names
            for field_name in (
                "position_world_m",
                "orientation_xyzw_world",
            )
            for value in dict(targets[target_name])[field_name]  # type: ignore[index]
        ) + tuple(
            round(float(value), 9)
            for target_name in target_names
            for value in preferred_joint_targets.get(target_name, ())
        )
        cache_key = (grasp_id, target_signature)
        cache_hit = cache_key in feasible_cache[role]
        if cache_hit:
            grasp_feasible = feasible_cache[role][cache_key]
        else:
            target_records = []
            grasp_feasible = True
            previous_joints: tuple[float, ...] | None = None
            for target_name in target_names:
                seed = preferred_joint_targets.get(target_name, previous_joints)
                if seed is None:
                    joints, message = commanders[role].compute_ik(_pose_target(dict(targets[target_name])))
                else:
                    joints, message = commanders[role].compute_ik(
                        _pose_target(dict(targets[target_name])),
                        seed_joint_positions=seed,
                    )
                ok = joints is not None
                target_records.append(
                    {
                        "target": target_name,
                        "ok": ok,
                        "message": message,
                    }
                )
                if not ok:
                    grasp_feasible = False
                    failure = f"{role} grasp {grasp_id} failed {target_name}: {message}"
                    break
                previous_joints = tuple(float(value) for value in joints)
            feasible_cache[role][cache_key] = grasp_feasible
            records = state["records"]
            assert isinstance(records, dict)
            role_records = records[role]
            assert isinstance(role_records, list)
            role_records.append(
                {
                    "grasp_id": grasp_id,
                    "feasible": grasp_feasible,
                    "targets": target_records,
                }
            )
            checked_key = f"{role}_grasps_checked"
            feasible_key = f"{role}_grasps_feasible"
            state[checked_key] = int(state[checked_key]) + 1
            if grasp_feasible:
                state[feasible_key] = int(state[feasible_key]) + 1
        if cache_hit and not grasp_feasible:
            failure = f"{role} grasp {grasp_id} reused a cached IK failure"
        pair_role_records[role] = {
            "grasp_id": grasp_id,
            "cache_hit": cache_hit,
            "feasible": grasp_feasible,
        }
        if not grasp_feasible:
            break

    pair_feasible = not failure
    state["pair_tasks_checked"] = int(state["pair_tasks_checked"]) + 1
    if pair_feasible:
        state["pair_tasks_after"] = int(state["pair_tasks_after"]) + 1
    pair_records = state["pair_records"]
    assert isinstance(pair_records, list)
    pair_records.append(
        {
            "rank": int(rank),
            "pair_id": task.pair_id,
            "transition_id": str(getattr(task, "transition_id", "")),
            "execution_candidate_id": str(getattr(task, "execution_candidate_id", task.pair_id)),
            "selection_score": float(task.selection_score),
            "feasible": pair_feasible,
            "failure": failure,
            "roles": pair_role_records,
        }
    )
    return pair_feasible, failure


def _configure_role_assignment(
    *,
    requested_inserter_arm: str,
    pickup_y: float,
    assembly_y: float,
) -> tuple[str, str, MovableFrame, MovableFrame]:
    global ARM_SPECS

    inserter_robot = str(requested_inserter_arm)
    if inserter_robot == "auto":
        inserter_robot = "lbr_one" if float(pickup_y) < float(assembly_y) else "lbr_two"
    if inserter_robot not in ARM_SPEC_BY_ROBOT:
        raise ValueError(f"Unsupported inserter arm '{requested_inserter_arm}'.")
    holder_robot = "lbr_two" if inserter_robot == "lbr_one" else "lbr_one"
    robot_bases = {
        "lbr_one": DEFAULT_HOLDER_BASE_WORLD,
        "lbr_two": DEFAULT_INSERTER_BASE_WORLD,
    }
    ARM_SPECS = {
        "holder": dict(ARM_SPEC_BY_ROBOT[holder_robot]),
        "inserter": dict(ARM_SPEC_BY_ROBOT[inserter_robot]),
    }
    return (
        holder_robot,
        inserter_robot,
        robot_bases[holder_robot],
        robot_bases[inserter_robot],
    )


def main() -> int:
    args = _parse_args()
    if rclpy is None:
        raise RuntimeError(
            "ROS2 MoveIt dependencies are unavailable. Source ROS2 and both workspaces before running this planner."
        )
    if args.max_pair_attempts < 1:
        raise ValueError("--max-pair-attempts must be at least 1.")
    if args.joint_rank_candidates < 0:
        raise ValueError("--joint-rank-candidates must be non-negative.")
    if args.joint_rank_ik_candidates < 1:
        raise ValueError("--joint-rank-ik-candidates must be at least 1.")
    if args.joint_rank_beam_width < 1:
        raise ValueError("--joint-rank-beam-width must be at least 1.")
    (
        holder_robot,
        inserter_robot,
        holder_robot_base_world,
        inserter_robot_base_world,
    ) = _configure_role_assignment(
        requested_inserter_arm=str(args.inserter_arm),
        pickup_y=float(args.pickup_y),
        assembly_y=float(args.assembly_y),
    )
    print(
        "[DUAL-SIM-PLAN] Role assignment "
        f"holder={holder_robot} inserter={inserter_robot} "
        f"pickup_y={float(args.pickup_y):.3f}",
        flush=True,
    )
    assembly_z = float(args.floor_z) if args.assembly_z is None else float(args.assembly_z)
    selection = resolve_dual_robot_step_selection(
        assembly=args.assembly,
        incoming_part_id=args.incoming_part_id,
        artifact_root=args.artifact_root,
        artifact_dir=args.artifact_dir,
        step_id=args.step_id,
    )
    print(
        "[DUAL-SIM-PLAN] Resolved "
        f"assembly={selection.assembly} base={selection.base_part_id} "
        f"incoming={selection.incoming_part_id} step={selection.step_id} "
        f"prefix={list(selection.assembled_part_ids_before)}",
        flush=True,
    )

    strict_retained_only = bool(args.retained_only) and not bool(args.all_compatible) and not bool(args.pair_id)
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
            pickup_floor_clearance_margin_m=float(args.pickup_floor_clearance_margin_m),
            transport_clearance_m=float(args.transport_clearance_m),
            pickup_top_down_score_weight=float(args.pickup_top_down_score_weight),
            holder_robot_name=holder_robot,
            inserter_robot_name=inserter_robot,
            holder_robot_base_world=holder_robot_base_world,
            inserter_robot_base_world=inserter_robot_base_world,
            retained_only=strict_retained_only,
            include_nonretained_identity_fallbacks=(not strict_retained_only and not bool(args.pair_id)),
        )
    )
    if args.pair_id:
        tasks = [task for task in tasks if task.pair_id == str(args.pair_id)]
        if not tasks:
            raise ValueError(f"Requested pair '{args.pair_id}' is not an accepted pair for {selection.step_id}.")
    if args.holder_grasp_id:
        tasks = [task for task in tasks if task.holder_candidate.grasp_id == str(args.holder_grasp_id)]
        if not tasks:
            raise ValueError(
                f"No accepted pair uses holder grasp '{args.holder_grasp_id}' "
                f"for {selection.step_id} at this placement."
            )
    if args.holder_only:
        unique_holder_tasks = {}
        for task in tasks:
            unique_holder_tasks.setdefault(
                task.holder_candidate.grasp_id,
                task,
            )
        tasks = list(unique_holder_tasks.values())
    tasks = tasks[: int(args.max_pair_attempts)]
    if not tasks:
        raise RuntimeError("No ranked compatible pair is available to plan.")
    debug_candidate_counts = dict(getattr(tasks[0], "candidate_filter_diagnostics", {}))
    debug_candidate_counts.update(
        {
            "planner_queue_execution_candidates": len(tasks),
            "planner_queue_noncrossing_execution_candidates": sum(
                not _transition_crosses_holder_corridor(task) for task in tasks
            ),
            "planner_queue_crossed_execution_candidates": sum(
                _transition_crosses_holder_corridor(task) for task in tasks
            ),
            "planner_queue_unique_holder_grasps": len({task.holder_candidate.grasp_id for task in tasks}),
            "planner_queue_unique_inserter_grasps": len({task.inserter_candidate.grasp_id for task in tasks}),
            "joint_rank_candidates_checked": 0,
            "joint_rank_candidates_planned": 0,
            "joint_rank_candidates_failed": 0,
            "exact_ik_pair_tasks_checked": 0,
            "exact_ik_holder_grasps_checked": 0,
            "exact_ik_inserter_grasps_checked": 0,
        }
    )
    debug_server: DualRobotPlanningDebugServer | None = None
    if bool(args.debug_gui):
        try:
            debug_server = DualRobotPlanningDebugServer(port=int(args.debug_gui_port))
            debug_url = debug_server.start(open_browser=True)
            print(
                f"[DUAL-SIM-PLAN] Live planning debugger: {debug_url}",
                flush=True,
            )
        except OSError as exc:
            print(
                f"[DUAL-SIM-PLAN] Could not start live planning debugger: {exc}",
                flush=True,
            )
            debug_server = None

    def update_debug(
        *,
        task=None,
        attempt_index: int | None = None,
        phase: str,
        status: str,
        message: str = "",
        record_event: bool = True,
    ) -> None:
        if debug_server is None:
            return
        try:
            debug_server.update(
                task=task,
                attempt_index=attempt_index,
                attempt_total=len(tasks),
                phase=phase,
                status=status,
                message=message,
                candidate_counts=debug_candidate_counts,
                record_event=record_event,
            )
        except Exception as exc:  # pragma: no cover - display must not stop planning
            print(
                f"[DUAL-SIM-PLAN] Live debugger update failed: {exc}",
                flush=True,
            )

    rclpy.init()
    commanders: dict[str, MoveItPoseCommander] = {}
    attempt_records: list[dict[str, object]] = []
    fatal_failure = ""
    joint_space_ranking: dict[str, object] = {
        "skipped": True,
        "reason": "not_started",
        "candidate_count_before": len(tasks),
    }
    preferred_joint_targets_by_candidate: dict[
        str,
        dict[str, tuple[float, ...]],
    ] = {}
    ik_preflight = (
        {
            "skipped": True,
            "mode": "disabled",
            "pair_tasks_before": len(tasks),
        }
        if bool(args.skip_ik_preflight)
        else _new_ik_preflight_state(pair_task_count=len(tasks))
    )
    ik_feasible_cache: dict[
        str,
        dict[tuple[str, tuple[float, ...]], bool],
    ] = {role: {} for role in IK_PREFLIGHT_TARGETS}
    try:
        active_roles = ("holder",) if args.holder_only else tuple(ARM_SPECS)
        target_sequence = (
            (("holder", "holder_pregrasp"), ("holder", "holder_grasp")) if args.holder_only else TARGET_SEQUENCE
        )
        commanders = {
            role: _commander(
                role=role,
                moveit_namespace=str(args.moveit_namespace),
                args=args,
            )
            for role in active_roles
        }
        for commander in commanders.values():
            commander.wait_for_moveit(require_execute=True)
        work_surface = _work_surface_obstacle(floor_z_world_m=float(args.floor_z))
        surface_ok, surface_message = commanders["holder"].apply_planning_scene_obstacles(
            [work_surface],
            default_frame_id="base_link",
        )
        if not surface_ok:
            raise RuntimeError(f"Could not add the work surface to MoveIt: {surface_message}")
        print(
            f"[DUAL-SIM-PLAN] {surface_message}",
            flush=True,
        )

        update_debug(
            task=tasks[0],
            attempt_index=0,
            phase="reset",
            status="planning",
            message="Returning both mock arms to the nominal start state.",
        )
        reset_ok, reset_messages = _reset_active_roles(
            commanders,
            active_roles=active_roles,
            holder_start_joint_positions=(
                None
                if args.holder_start_joint_positions is None
                else tuple(float(value) for value in args.holder_start_joint_positions)
            ),
            recovering_from_candidate=False,
        )
        if not reset_ok:
            update_debug(
                task=tasks[0],
                attempt_index=0,
                phase="reset",
                status="fatal",
                message=f"Initial dual-arm reset failed: {reset_messages}",
            )
            time.sleep(0.35)
            raise RuntimeError(f"Could not reset dual MoveIt mock state: {reset_messages}")

        if args.holder_only:
            joint_space_ranking = {
                "skipped": True,
                "reason": "holder_only",
                "candidate_count_before": len(tasks),
            }
        elif bool(args.skip_joint_space_ranking):
            joint_space_ranking = {
                "skipped": True,
                "reason": "disabled_by_cli",
                "candidate_count_before": len(tasks),
            }
        else:
            tasks, joint_space_ranking, preferred_joint_targets_by_candidate = _rank_tasks_by_inserter_joint_path(
                tasks,
                commander=commanders["inserter"],
                candidate_limit=int(args.joint_rank_candidates),
                ik_candidate_count=int(args.joint_rank_ik_candidates),
                beam_width=int(args.joint_rank_beam_width),
                update_debug=update_debug,
            )
            print(
                "[DUAL-SIM-PLAN] Joint-space transition ranking: "
                f"checked={joint_space_ranking['candidate_count_checked']} "
                f"planned={joint_space_ranking['candidate_count_planned']} "
                f"failed={joint_space_ranking['candidate_count_failed']}",
                flush=True,
            )
            debug_candidate_counts.update(
                {
                    "joint_rank_candidates_checked": int(joint_space_ranking["candidate_count_checked"]),
                    "joint_rank_candidates_planned": int(joint_space_ranking["candidate_count_planned"]),
                    "joint_rank_candidates_failed": int(joint_space_ranking["candidate_count_failed"]),
                }
            )

        last_task = tasks[0]
        for attempt_index, task in enumerate(tasks, start=1):
            last_task = task
            update_debug(
                task=task,
                attempt_index=attempt_index,
                phase="ik_preflight",
                status="planning",
                message=(
                    "Checking exact holder and inserter target IK before "
                    "executing this candidate in the mock MoveIt state."
                ),
            )
            print(
                f"[DUAL-SIM-PLAN] Attempt {attempt_index}/{len(tasks)} "
                f"pair={task.pair_id} transition={task.transition_id} "
                f"pair_score={task.pair_score:.4f} "
                f"selection_score={task.selection_score:.4f} "
                f"layout_proxy={task.layout_proxy_score:.4f}",
                flush=True,
            )

            task_payload = task.to_payload()
            targets = dict(task_payload["targets"])
            pregrasp_aabb_obstacles = simple_dual_robot_pregrasp_aabb_obstacles(task)
            pregrasp_aabb_schedule = simple_dual_robot_pregrasp_aabb_schedule(pregrasp_aabb_obstacles)
            if not bool(args.skip_ik_preflight):
                pair_ik_ok, pair_ik_failure = _ik_preflight_pair(
                    task,
                    commanders=commanders,
                    feasible_cache=ik_feasible_cache,
                    state=ik_preflight,
                    rank=attempt_index,
                    roles=active_roles,
                    preferred_joint_targets=(
                        preferred_joint_targets_by_candidate.get(
                            task.execution_candidate_id,
                            {},
                        )
                    ),
                )
                checked = int(ik_preflight["pair_tasks_checked"])
                holder_checked = int(ik_preflight["holder_grasps_checked"])
                inserter_checked = int(ik_preflight["inserter_grasps_checked"])
                debug_candidate_counts.update(
                    {
                        "exact_ik_pair_tasks_checked": checked,
                        "exact_ik_holder_grasps_checked": holder_checked,
                        "exact_ik_inserter_grasps_checked": inserter_checked,
                    }
                )
                if not pair_ik_ok:
                    update_debug(
                        task=task,
                        attempt_index=attempt_index,
                        phase="ik_preflight",
                        status="failed",
                        message=pair_ik_failure,
                    )
                    print(
                        "[DUAL-SIM-PLAN] IK preflight "
                        f"rank {attempt_index}/{len(tasks)} failed: "
                        f"{pair_ik_failure}. Checked {checked} pair(s), "
                        f"{holder_checked} holder and {inserter_checked} "
                        "inserter grasp(s); trying the next score.",
                        flush=True,
                    )
                    attempt_records.append(
                        {
                            "attempt_index": attempt_index,
                            "pair_id": task.pair_id,
                            "transition_id": task.transition_id,
                            "execution_candidate_id": task.execution_candidate_id,
                            "score": task.pair_score,
                            "selection_score": task.selection_score,
                            "pickup_top_down_score": (task.pickup_top_down_score),
                            "layout_proxy_score": (task.layout_proxy_score),
                            "holder_reachability_proxy_score": (task.holder_reachability_proxy_score),
                            "inserter_reachability_proxy_score": (task.inserter_reachability_proxy_score),
                            "success": False,
                            "failure": (f"ik_preflight: {pair_ik_failure}"),
                            "steps": [],
                        }
                    )
                    continue
                print(
                    "[DUAL-SIM-PLAN] IK preflight "
                    f"rank {attempt_index}/{len(tasks)} passed; "
                    "starting its full trajectory plan immediately.",
                    flush=True,
                )
                update_debug(
                    task=task,
                    attempt_index=attempt_index,
                    phase="ik_preflight",
                    status="succeeded",
                    message="Exact target IK preflight passed.",
                )
            trajectories: dict[str, object] = {}
            steps: list[dict[str, object]] = []
            failure = ""
            for role, target_name in target_sequence:
                update_debug(
                    task=task,
                    attempt_index=attempt_index,
                    phase=target_name,
                    status="planning",
                    message=(f"Planning and executing {role} target '{target_name}' in the shared MoveIt scene."),
                )
                target = _pose_target(dict(targets[target_name]))
                joint_names = tuple(str(value) for value in ARM_SPECS[role]["joint_names"])
                active_aabbs = _pregrasp_aabb_obstacles_for_target(
                    pregrasp_aabb_obstacles,
                    target_name=target_name,
                )
                if active_aabbs:
                    scene_ok, scene_message = commanders["holder"].apply_planning_scene_obstacles(
                        active_aabbs,
                        default_frame_id="base_link",
                    )
                    if not scene_ok:
                        raise RuntimeError(f"Could not add object AABBs before {target_name}: {scene_message}")
                    print(
                        f"[DUAL-SIM-PLAN] {target_name}: {scene_message}",
                        flush=True,
                    )
                try:
                    trajectory_payload, message = _plan_and_execute(
                        commanders[role],
                        target=target,
                        label=f"{task.pair_id}_{target_name}",
                        expected_joint_names=joint_names,
                        preferred_joint_positions=(
                            preferred_joint_targets_by_candidate.get(
                                task.execution_candidate_id,
                                {},
                            ).get(target_name)
                            if role == "inserter"
                            else None
                        ),
                    )
                finally:
                    if active_aabbs:
                        remove_ok, remove_message = commanders["holder"].remove_planning_scene_obstacles(
                            [str(obstacle["id"]) for obstacle in active_aabbs],
                            default_frame_id="base_link",
                        )
                        if not remove_ok:
                            raise RuntimeError(f"Could not remove object AABBs after {target_name}: {remove_message}")
                        print(
                            f"[DUAL-SIM-PLAN] {target_name}: {remove_message}",
                            flush=True,
                        )
                ok = trajectory_payload is not None
                steps.append(
                    {
                        "role": role,
                        "target": target_name,
                        "ok": ok,
                        "message": message,
                    }
                )
                print(
                    f"[DUAL-SIM-PLAN] {target_name}: {'ok' if ok else 'failed'} {message}",
                    flush=True,
                )
                if not ok:
                    failure = f"{target_name}: {message}"
                    update_debug(
                        task=task,
                        attempt_index=attempt_index,
                        phase=target_name,
                        status="failed",
                        message=message,
                    )
                    break
                update_debug(
                    task=task,
                    attempt_index=attempt_index,
                    phase=target_name,
                    status="succeeded",
                    message=message,
                )
                trajectories[target_name] = trajectory_payload

            attempt_records.append(
                {
                    "attempt_index": attempt_index,
                    "pair_id": task.pair_id,
                    "transition_id": task.transition_id,
                    "execution_candidate_id": task.execution_candidate_id,
                    "score": task.pair_score,
                    "selection_score": task.selection_score,
                    "pickup_top_down_score": (task.pickup_top_down_score),
                    "layout_proxy_score": task.layout_proxy_score,
                    "holder_reachability_proxy_score": (task.holder_reachability_proxy_score),
                    "inserter_reachability_proxy_score": (task.inserter_reachability_proxy_score),
                    "success": not failure,
                    "failure": failure,
                    "steps": steps,
                }
            )
            if failure:
                update_debug(
                    task=task,
                    attempt_index=attempt_index,
                    phase="reset",
                    status="planning",
                    message=(
                        "Candidate failed after partial mock execution; retracting inserter before resetting holder."
                    ),
                )
                recovery_ok, recovery_messages = _reset_active_roles(
                    commanders,
                    active_roles=active_roles,
                    holder_start_joint_positions=(
                        None
                        if args.holder_start_joint_positions is None
                        else tuple(float(value) for value in args.holder_start_joint_positions)
                    ),
                    recovering_from_candidate=True,
                )
                attempt_records[-1]["reset"] = {
                    "ok": recovery_ok,
                    "order": list(reversed(active_roles)),
                    "messages": recovery_messages,
                }
                if not recovery_ok:
                    fatal_failure = (
                        f"Could not safely recover the mock state after candidate {attempt_index}: {recovery_messages}"
                    )
                    update_debug(
                        task=task,
                        attempt_index=attempt_index,
                        phase="reset",
                        status="fatal",
                        message=fatal_failure,
                    )
                    break
                update_debug(
                    task=task,
                    attempt_index=attempt_index,
                    phase="reset",
                    status="succeeded",
                    message=f"Mock-state recovery complete: {recovery_messages}",
                )
                continue

            task_payload["generated_by"] = "scripts/plan_simple_dual_robot_sim.py"
            task_payload["moveit"] = {
                "namespace": str(args.moveit_namespace),
                "frame_id": "base_link",
                "object_collision_geometry_in_scene": False,
                "work_surface_collision_geometry_in_scene": True,
                "work_surface": work_surface,
                "pregrasp_aabb_collision_geometry": {
                    "representation": ("object_world_aabb_minus_selected_gripper_sweep"),
                    "obstacles": pregrasp_aabb_obstacles,
                    "active_by_target": pregrasp_aabb_schedule,
                    "removed_before_grasp_approach": True,
                },
                "arm_arm_collision_checking": True,
                "start_joint_positions": list(MOVEIT_START_JOINT_POSITIONS),
                "ik_preflight": ik_preflight,
                "joint_space_ranking": joint_space_ranking,
                "attempts": attempt_records,
            }
            task_payload["trajectories"] = trajectories
            task_payload["holder_only"] = bool(args.holder_only)
            output = (
                selection.artifact_dir / f"simple_dual_robot_sim_plan_{selection.step_id}.json"
                if args.output is None
                else args.output.expanduser().resolve()
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(task_payload, indent=2),
                encoding="utf-8",
            )
            print(
                f"[DUAL-SIM-PLAN] Selected pair {task.pair_id}; wrote {output}",
                flush=True,
            )
            update_debug(
                task=task,
                attempt_index=attempt_index,
                phase="complete",
                status="complete",
                message=f"Selected {task.execution_candidate_id}; plan written to {output}",
            )
            time.sleep(0.35)
            return 0
        update_debug(
            task=last_task,
            attempt_index=len(attempt_records),
            phase="complete",
            status="fatal" if fatal_failure else "failed",
            message=(fatal_failure or f"No complete plan among {len(tasks)} ranked candidates."),
        )
        time.sleep(0.35)
    finally:
        for commander in commanders.values():
            commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if debug_server is not None:
            debug_server.close()

    output = (
        selection.artifact_dir / f"simple_dual_robot_sim_plan_{selection.step_id}.json"
        if args.output is None
        else args.output.expanduser().resolve()
    )
    failure_payload = {
        "schema_version": 1,
        "kind": "dual_robot_simple_sim_plan_failure",
        "assembly": selection.assembly,
        "incoming_part_id": selection.incoming_part_id,
        "step_id": selection.step_id,
        "fatal_failure": fatal_failure,
        "joint_space_ranking": joint_space_ranking,
        "ik_preflight": ik_preflight,
        "attempts": attempt_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(failure_payload, indent=2),
        encoding="utf-8",
    )
    if fatal_failure:
        raise RuntimeError(f"{fatal_failure}; diagnostics written to {output}.")
    raise RuntimeError(f"MoveIt could not plan any of {len(tasks)} ranked pairs; diagnostics written to {output}.")


if __name__ == "__main__":
    raise SystemExit(main())
