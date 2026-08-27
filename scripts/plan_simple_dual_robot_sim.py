#!/usr/bin/env python3
"""Plan a simple holder plus pickup-to-preinsertion sequence with dual MoveIt."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping

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
    simple_dual_robot_attached_collision_objects,
    simple_dual_robot_pregrasp_aabb_obstacles,
    simple_dual_robot_pregrasp_aabb_schedule,
    with_inserter_pickup_pregrasp_offset,
)
from grasp_planning.ros2.kuka_ik_seeds import kuka_iiwa_ik_seed_candidates  # noqa: E402
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
    gripper_clamp_width,
    kuka_moveit_gripper_state,
)

MOVEIT_START_JOINT_POSITIONS = KUKA_MOVEIT_ARM_START_JOINT_VALUES
ARM_SPEC_BY_ROBOT = {
    "lbr_one": {
        "robot": "lbr_one",
        "planning_group": "arm_one",
        "pose_link": "lbr_one_gripper_tcp",
        "joint_names": tuple(f"lbr_one_A{index}" for index in range(1, 8)),
    },
    "lbr_two": {
        "robot": "lbr_two",
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
        "--gripper-model",
        choices=("y_gripper", "pdz_gripper"),
        default="y_gripper",
        help="MoveIt end-effector model carried by both arms.",
    )
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
        help=(
            "Maximum exact-IK-feasible candidates admitted to path planning "
            "and execution. This does not truncate the exact-IK input pool."
        ),
    )
    parser.add_argument(
        "--max-ik-screen-candidates",
        type=int,
        default=0,
        help=(
            "Optional cap on producer-ranked candidates screened by exact IK; "
            "0 checks the finite pose-feasible pool until enough path candidates survive."
        ),
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
    parser.add_argument(
        "--ik-timeout-s",
        type=float,
        default=0.35,
        help="MoveIt kinematic IK timeout for each distinct active-arm seed.",
    )
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
        "--exact-ik-candidates",
        type=int,
        default=7,
        help="Distinct complete-state seeds tried per target during exact IK preflight.",
    )
    parser.add_argument(
        "--exact-ik-beam-width",
        type=int,
        default=4,
        help="Complete holder/inserter IK branches retained between exact targets.",
    )
    parser.add_argument(
        "--exact-ik-seed-perturbation-rad",
        type=float,
        default=0.60,
        help="Deterministic shoulder/elbow/wrist seed perturbation used by exact IK.",
    )
    parser.add_argument(
        "--pickup-approach-ik-steps",
        type=int,
        default=5,
        help=(
            "Full-state-valid continuation steps from incoming pickup pregrasp "
            "to grasp. Each waypoint is seeded by the preceding solution."
        ),
    )
    parser.add_argument(
        "--pickup-pregrasp-offsets-m",
        type=float,
        nargs="+",
        default=(0.10, 0.075, 0.05, 0.025),
        metavar="METERS",
        help=(
            "Ordered pickup-only pregrasp distances tried by exact IK. A shorter "
            "distance is tried only after the preceding distance has pure "
            "kinematic no-IK in the pickup pregrasp/approach chain."
        ),
    )
    parser.add_argument(
        "--ik-collision-diagnostics",
        action="store_true",
        help=(
            "Record exact colliding bodies and per-target validity totals from the normal "
            "kinematic-IK plus complete-state-validity preflight."
        ),
    )
    parser.add_argument(
        "--ik-only",
        action="store_true",
        help="Stop after an exact IK-feasible pair is selected; do not run OMPL or mock execution.",
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


def _apply_attached_collision_objects(
    commander,
    obstacles: Mapping[str, Mapping[str, object]],
) -> tuple[bool, str]:
    apply_attached = getattr(commander, "apply_planning_scene_attached_obstacles", None)
    if not callable(apply_attached):
        return True, "Attached-collision API unavailable in lightweight test adapter."
    return apply_attached(
        [dict(value) for value in obstacles.values()],
        default_frame_id="base_link",
    )


def _apply_phase_collision_obstacles(
    commander,
    obstacles: list[dict[str, object]],
) -> tuple[bool, str]:
    apply_obstacles = getattr(commander, "apply_planning_scene_obstacles", None)
    if not callable(apply_obstacles):
        return True, "World-collision API unavailable in lightweight test adapter."
    return apply_obstacles(obstacles, default_frame_id="base_link")


def _remove_phase_collision_obstacles(
    commander,
    obstacles: list[dict[str, object]],
) -> tuple[bool, str]:
    remove_obstacles = getattr(commander, "remove_planning_scene_obstacles", None)
    if not callable(remove_obstacles):
        return True, "World-collision API unavailable in lightweight test adapter."
    return remove_obstacles(
        [str(obstacle["id"]) for obstacle in obstacles],
        default_frame_id="base_link",
    )


def _remove_attached_collision_objects(
    commander,
    obstacles: Mapping[str, Mapping[str, object]],
) -> tuple[bool, str]:
    remove_attached = getattr(commander, "remove_planning_scene_attached_obstacles", None)
    if not callable(remove_attached):
        return True, "Attached-collision API unavailable in lightweight test adapter."
    return remove_attached(
        [dict(value) for value in obstacles.values()],
        default_frame_id="base_link",
    )


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


def _inserter_diverse_task_prefix(tasks: list, *, limit: int) -> list:
    """Keep score order while visiting unique pickup targets before repeats.

    Pair/transition expansion can place hundreds of holder variants for a few
    high-scoring inserter grasps at the head of the queue. Exact IK then spends
    its whole budget proving the same pickup poses unreachable. Round-robin is
    applied separately to noncrossing and crossed phases so arm-crossing policy
    remains stricter than score or pickup diversity.
    """

    bounded_limit = max(0, min(int(limit), len(tasks)))
    if bounded_limit == 0:
        return []
    selected: list = []
    for crossed in (False, True):
        phase_tasks = [task for task in tasks if _transition_crosses_holder_corridor(task) is crossed]
        by_pickup: dict[tuple[object, ...], list] = {}
        for task in phase_tasks:
            by_pickup.setdefault(_pickup_target_signature(task), []).append(task)
        pickup_order = tuple(by_pickup)
        depth = 0
        while len(selected) < bounded_limit:
            added = False
            for pickup_key in pickup_order:
                pickup_tasks = by_pickup[pickup_key]
                if depth >= len(pickup_tasks):
                    continue
                selected.append(pickup_tasks[depth])
                added = True
                if len(selected) >= bounded_limit:
                    break
            if not added:
                break
            depth += 1
        if len(selected) >= bounded_limit:
            break
    return selected


def _runtime_ik_screen_queue(tasks: list, *, holder_only: bool) -> list:
    """Return the complete finite pool in a diversity-preserving screen order."""

    if holder_only:
        unique_holder_tasks = {}
        for task in tasks:
            unique_holder_tasks.setdefault(task.holder_candidate.grasp_id, task)
        return list(unique_holder_tasks.values())
    return _inserter_diverse_task_prefix(tasks, limit=len(tasks))


@dataclass(frozen=True)
class _ExactIkFeasibleCandidate:
    """One producer-ranked task admitted to bounded path planning."""

    task: object
    screen_rank: int
    candidate_rank: int
    joint_targets: dict[str, tuple[float, ...]]


def _iter_exact_ik_feasible_candidates(
    tasks: list,
    *,
    path_candidate_limit: int,
    ik_screen_candidate_limit: int,
    evaluate: Callable[
        [object, int],
        tuple[bool, str, dict[str, tuple[float, ...]]]
        | tuple[bool, str, dict[str, tuple[float, ...]], object],
    ],
) -> Iterator[_ExactIkFeasibleCandidate]:
    """Lazily screen a broad queue before each bounded path attempt.

    ``path_candidate_limit`` deliberately caps only candidates that survive
    exact complete-state IK.  It must not truncate the producer-ranked input
    queue: doing that prevents a later pose-feasible grasp from ever reaching
    the authoritative robot check.  ``ik_screen_candidate_limit == 0`` means
    the finite input queue is the bound; a positive value is an explicit
    operational cap for unusually large artifacts.  Yielding immediately is
    important: a caller can try the first IK-feasible candidate without first
    paying to fill the entire path-attempt budget.
    """

    if int(path_candidate_limit) < 1:
        raise ValueError("path_candidate_limit must be at least 1.")
    if int(ik_screen_candidate_limit) < 0:
        raise ValueError("ik_screen_candidate_limit must be non-negative.")

    source_count = len(tasks)
    screen_count = (
        source_count
        if int(ik_screen_candidate_limit) == 0
        else min(source_count, int(ik_screen_candidate_limit))
    )
    admitted = 0
    for screen_rank, task in enumerate(tasks[:screen_count], start=1):
        evaluation = evaluate(task, screen_rank)
        feasible, _failure, joint_targets = evaluation[:3]
        selected_task = task if len(evaluation) == 3 else evaluation[3]
        if not feasible:
            continue
        producer_rank = int(getattr(selected_task, "candidate_rank", 0))
        admitted += 1
        yield _ExactIkFeasibleCandidate(
            task=selected_task,
            screen_rank=screen_rank,
            candidate_rank=(producer_rank if producer_rank > 0 else screen_rank),
            joint_targets=dict(joint_targets),
        )
        if admitted >= int(path_candidate_limit):
            return


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
    planned_tasks = [item[4] for item in planned]
    planned_object_ids = {id(task) for task in planned_tasks}
    # A bounded pre-rank success is positive evidence and may promote a task.
    # A failure from its deliberately smaller IK/beam search is not proof that
    # the exact shared-scene gate will fail.  Keep every non-planned task in
    # producer order so a producer-top grasp cannot be demoted behind the whole
    # unchecked queue merely because the weaker pre-ranker missed its branch.
    fallback_tasks = [task for task in tasks if id(task) not in planned_object_ids]
    checked_object_ids = {id(task) for task in checked_tasks}
    unranked = [task for task in tasks if id(task) not in checked_object_ids]

    def crossing_phase(crosses: bool) -> list:
        return [
            *(task for task in planned_tasks if _transition_crosses_holder_corridor(task) is crosses),
            *(task for task in fallback_tasks if _transition_crosses_holder_corridor(task) is crosses),
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
        "primary_sort": "strict_noncrossing_phase_then_successful_preplans_then_stable_producer_fallback",
        "tie_breaker": "transition_cost_for_successful_preplans; producer_order_for_all_other_tasks",
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
            ik_timeout_s=float(args.ik_timeout_s),
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
    preferred_joint_sequence: tuple[tuple[str, tuple[float, ...]], ...] | None = None,
    gripper_robot_state: Mapping[str, float] | None = None,
) -> tuple[dict[str, object] | None, str]:
    if preferred_joint_positions is not None and preferred_joint_sequence is not None:
        raise ValueError("Provide either one preferred joint target or a preferred joint sequence, not both.")

    if preferred_joint_sequence is not None:
        if not preferred_joint_sequence:
            raise ValueError("A preferred joint sequence must contain at least one target.")
        combined_waypoints: list[list[float]] = []
        segment_records: list[dict[str, object]] = []
        final_execution_message = ""
        segment_count = len(preferred_joint_sequence)
        for segment_index, (target_name, joint_positions) in enumerate(preferred_joint_sequence, start=1):
            segment_label = (
                label
                if segment_count == 1 or segment_index == segment_count
                else f"{label}__validated_segment_{segment_index:02d}_of_{segment_count:02d}"
            )
            trajectory, planning_message = commander.plan_to_joint_positions(
                joint_positions,
                label=f"{segment_label}_joint_ranked",
            )
            if trajectory is None:
                return None, f"validated joint segment '{target_name}' path planning failed: {planning_message}"
            segment_payload = _trajectory_payload(
                trajectory,
                expected_joint_names=expected_joint_names,
            )
            waypoint_start = len(combined_waypoints)
            combined_waypoints.extend(segment_payload["waypoints"])  # type: ignore[arg-type]
            waypoint_end = len(combined_waypoints)
            ok, execution_message = commander.execute_trajectory(
                trajectory,
                label=segment_label,
            )
            if not ok:
                return None, f"validated joint segment '{target_name}' execution failed: {execution_message}"
            final_execution_message = execution_message
            segment_records.append(
                {
                    "target": str(target_name),
                    "joint_target": [float(value) for value in joint_positions],
                    "waypoint_range": [waypoint_start, waypoint_end],
                }
            )
        payload: dict[str, object] = {
            "joint_names": list(expected_joint_names),
            "waypoints": combined_waypoints,
        }
        if segment_count > 1:
            payload["validated_joint_segments"] = segment_records
            return payload, f"executed {segment_count} validated joint segments; {final_execution_message}"
        return payload, final_execution_message

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
        fallback_seed_state = dict(gripper_robot_state or {})
        fallback_seed_state.update(
            (str(name), float(value))
            for name, value in zip(expected_joint_names, MOVEIT_START_JOINT_POSITIONS)
        )
        fallback_joints, fallback_message = commander.compute_ik(
            target,
            seed_robot_state=fallback_seed_state,
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


def _validated_joint_target_sequence(
    joint_targets: Mapping[str, tuple[float, ...]],
    *,
    target_name: str,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Return the ordered exact-IK chain associated with one runtime target.

    The canonical target key remains the endpoint for compatibility with older
    artifacts. Newer results place stable approach keys immediately before it.
    """

    endpoint = joint_targets.get(target_name)
    if endpoint is None:
        return ()
    if target_name not in {"holder_grasp", "inserter_pickup_grasp"}:
        return ((target_name, tuple(float(value) for value in endpoint)),)

    approach_prefix = f"{target_name}__approach_"
    approach_targets = sorted(
        (
            (str(name), tuple(float(value) for value in values))
            for name, values in joint_targets.items()
            if str(name).startswith(approach_prefix)
        ),
        key=lambda item: int(item[0][len(approach_prefix) :].split("_of_", 1)[0]),
    )
    return (*approach_targets, (target_name, tuple(float(value) for value in endpoint)))


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


@dataclass(frozen=True)
class _IkPreflightCacheEntry:
    feasible: bool
    failure: str
    branches: tuple["_IkPreflightBranch", ...]
    failure_kind: str = ""


@dataclass(frozen=True)
class _KinematicIkCacheEntry:
    """Collision-independent IK result for one active-arm target and seed."""

    joints: tuple[float, ...] | None
    message: str


@dataclass(frozen=True)
class _IkPreflightBranch:
    target_joint_positions: tuple[tuple[str, tuple[float, ...]], ...]
    terminal_robot_state: tuple[tuple[str, float], ...]
    joint_path_cost: float


@dataclass(frozen=True)
class _IkRoleSearchResult:
    branches: tuple[_IkPreflightBranch, ...]
    failure: str
    target_records: tuple[dict[str, object], ...]
    failure_kind: str = ""


@dataclass(frozen=True)
class _IkSearchTarget:
    label: str
    pose: PoseTarget
    result_target_name: str | None
    scene_target_name: str


def _complete_dual_arm_start_state(
    *,
    holder_start_joint_positions: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Return an explicit state for both arm groups after role assignment."""

    state: dict[str, float] = {}
    for role in ("holder", "inserter"):
        positions = (
            holder_start_joint_positions
            if role == "holder" and holder_start_joint_positions is not None
            else MOVEIT_START_JOINT_POSITIONS
        )
        joint_names = tuple(str(value) for value in ARM_SPECS[role]["joint_names"])
        if len(positions) != len(joint_names):
            raise ValueError(f"Expected {len(joint_names)} {role} start joints, got {len(positions)}.")
        state.update((name, float(value)) for name, value in zip(joint_names, positions))
    return state


def _task_approach_gripper_state(task) -> dict[str, float]:
    """Return candidate-specific partially-closed MoveIt finger joints."""

    holder_grasp = getattr(task, "holder_world_grasp", None)
    inserter_grasp = getattr(task, "inserter_pickup_world_grasp", None)
    holder_robot = str(
        getattr(task, "holder_robot_name", ARM_SPECS["holder"]["robot"])
    )
    inserter_robot = str(
        getattr(task, "inserter_robot_name", ARM_SPECS["inserter"]["robot"])
    )
    gripper_model = str(getattr(task, "pickup_gripper_collision_model", "kuka_y_gripper"))
    state: dict[str, float] = {}
    state.update(
        kuka_moveit_gripper_state(
            holder_robot,
            gripper_clamp_width(
                getattr(holder_grasp, "gripper_width", 0.05),
                gripper_model=gripper_model,
            ),
            gripper_model=gripper_model,
        )
    )
    state.update(
        kuka_moveit_gripper_state(
            inserter_robot,
            gripper_clamp_width(
                getattr(inserter_grasp, "gripper_width", 0.05),
                gripper_model=gripper_model,
            ),
            gripper_model=gripper_model,
        )
    )
    return state


def _task_post_grasp_state_updates(task) -> dict[str, dict[str, float]]:
    """Return finger-state changes that occur immediately after contact."""

    holder_grasp = getattr(task, "holder_world_grasp", None)
    inserter_grasp = getattr(task, "inserter_pickup_world_grasp", None)
    holder_robot = str(
        getattr(task, "holder_robot_name", ARM_SPECS["holder"]["robot"])
    )
    inserter_robot = str(
        getattr(task, "inserter_robot_name", ARM_SPECS["inserter"]["robot"])
    )
    gripper_model = str(getattr(task, "pickup_gripper_collision_model", "kuka_y_gripper"))
    return {
        "holder_grasp": kuka_moveit_gripper_state(
            holder_robot,
            getattr(holder_grasp, "jaw_width", 0.04),
            gripper_model=gripper_model,
        ),
        "inserter_pickup_grasp": kuka_moveit_gripper_state(
            inserter_robot,
            getattr(inserter_grasp, "jaw_width", 0.04),
            gripper_model=gripper_model,
        ),
    }


def _robot_state_signature(state: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(name), round(float(value), 9)) for name, value in state.items()))


def _new_ik_preflight_state(
    *,
    pair_task_count: int,
    ik_candidate_count: int = 1,
    ik_beam_width: int = 1,
    ik_seed_perturbation_rad: float = 0.60,
    pickup_approach_ik_steps: int = 1,
    collision_diagnostics: bool = False,
) -> dict[str, object]:
    state = {
        "skipped": False,
        "mode": "lazy_cached_kinematics_complete_state_multi_seed_beam",
        "scope": (
            "Each ranked pair is screened by active-arm kinematic IK followed by "
            "MoveIt validity of the complete hypothetical dual-arm state. Active-arm "
            "solutions are cached across passive-arm variants, but every reused "
            "solution is revalidated. Multiple holder branches are retained "
            "from pregrasp through grasp; every retained holder state then seeds "
            "the inserter pickup and transition search. Pickup pregrasp-to-grasp "
            "IK uses short continuation waypoints. Complete role results are cached by "
            "role, grasp ID, exact targets, search settings, preferred branches, "
            "and the complete input robot state."
        ),
        "ik_candidates_per_target": int(ik_candidate_count),
        "ik_beam_width": int(ik_beam_width),
        "ik_seed_perturbation_rad": float(ik_seed_perturbation_rad),
        "pickup_approach_ik_steps": int(pickup_approach_ik_steps),
        "collision_diagnostics_enabled": bool(collision_diagnostics),
        "ik_seed_calls": 0,
        "ik_request_duration_s": 0.0,
        "ik_kinematic_cache_hits": 0,
        "ik_kinematic_cache_misses": 0,
        "ik_kinematic_solutions_returned": 0,
        "ik_kinematic_failures": 0,
        "ik_state_validity_requests": 0,
        "ik_valid_states": 0,
        "ik_invalid_states": 0,
        "ik_invalid_states_without_contacts": 0,
        "ik_solutions_found": 0,
        "ik_distinct_solutions_retained": 0,
        "post_grasp_state_validity_requests": 0,
        "post_grasp_valid_states": 0,
        "post_grasp_invalid_states": 0,
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
    if collision_diagnostics:
        state["collision_diagnostics"] = {
            "mode": "collision_disabled_ik_then_full_state_validity",
            "scene_scope": (
                "Complete dual-arm robot state plus the MoveIt work surface. "
                "Target-specific part AABBs are applied on their configured "
                "pregrasp phases, and attached-object geometry is applied after pickup."
            ),
            "ik_requests": 0,
            "kinematic_cache_hits": 0,
            "kinematic_cache_misses": 0,
            "ik_request_duration_s": 0.0,
            "collision_disabled_ik_solutions": 0,
            "kinematic_or_numerical_failures": 0,
            "state_validity_requests": 0,
            "valid_states": 0,
            "invalid_states": 0,
            "invalid_states_without_contacts": 0,
            "contact_class_counts": {},
            "contact_pair_counts": {},
            "contact_examples": [],
        }
    return state


def _collision_body_arm(body: str) -> str:
    value = str(body)
    if value.startswith("lbr_one_"):
        return "lbr_one"
    if value.startswith("lbr_two_"):
        return "lbr_two"
    return ""


def _collision_contact_class(contact: Mapping[str, object]) -> str:
    body_1 = str(contact.get("body_1", ""))
    body_2 = str(contact.get("body_2", ""))
    bodies = {body_1, body_2}
    if "dual_sim_work_surface" in bodies:
        robot_body = body_2 if body_1 == "dual_sim_work_surface" else body_1
        if "finger" in robot_body:
            return "finger_floor"
        if "gripper" in robot_body:
            return "gripper_floor"
        return "arm_floor"
    arm_1 = _collision_body_arm(body_1)
    arm_2 = _collision_body_arm(body_2)
    if arm_1 and arm_2:
        return "inter_arm" if arm_1 != arm_2 else "self_collision"
    if int(contact.get("body_type_1", -1)) == 1 or int(contact.get("body_type_2", -1)) == 1:
        return "robot_world_object"
    return "other"


def _canonical_contact_pair(contact: Mapping[str, object]) -> str:
    bodies = sorted((str(contact.get("body_1", "")), str(contact.get("body_2", ""))))
    return f"{bodies[0]} <-> {bodies[1]}"


def _record_collision_diagnostic(
    state: dict[str, object],
    *,
    role: str,
    target: str,
    parent_index: int,
    seed_index: int,
    contacts: list[dict[str, object]],
) -> None:
    diagnostics = state.get("collision_diagnostics")
    if not isinstance(diagnostics, dict):
        return
    class_counts = diagnostics["contact_class_counts"]
    pair_counts = diagnostics["contact_pair_counts"]
    examples = diagnostics["contact_examples"]
    assert isinstance(class_counts, dict)
    assert isinstance(pair_counts, dict)
    assert isinstance(examples, list)
    for contact in contacts:
        contact_class = _collision_contact_class(contact)
        contact_pair = _canonical_contact_pair(contact)
        class_counts[contact_class] = int(class_counts.get(contact_class, 0)) + 1
        pair_counts[contact_pair] = int(pair_counts.get(contact_pair, 0)) + 1
        if len(examples) < 100:
            examples.append(
                {
                    "role": str(role),
                    "target": str(target),
                    "parent_index": int(parent_index),
                    "seed_index": int(seed_index),
                    "class": contact_class,
                    **contact,
                }
            )


def _exact_ik_seed_candidates(
    start_joint_positions: tuple[float, ...],
    *,
    preferred_joint_positions: tuple[float, ...] | None,
    candidate_count: int,
    perturbation_rad: float,
) -> tuple[tuple[float, ...], ...]:
    """Build deterministic bounded seeds spanning useful iiwa IK branches."""

    return kuka_iiwa_ik_seed_candidates(
        start_joint_positions,
        preferred_joint_positions=preferred_joint_positions,
        candidate_count=candidate_count,
        perturbation_rad=perturbation_rad,
    )

def _is_distinct_ik_solution(
    solution: tuple[float, ...],
    accepted: list[tuple[float, ...]],
    *,
    tolerance_rad: float = 0.05,
) -> bool:
    return all(
        float(np.max(np.abs(np.asarray(solution, dtype=float) - np.asarray(other, dtype=float))))
        >= float(tolerance_rad)
        for other in accepted
    )


def _ik_branch_sort_key(branch: _IkPreflightBranch) -> tuple[object, ...]:
    return (
        round(float(branch.joint_path_cost), 12),
        tuple(round(float(value), 9) for _name, joints in branch.target_joint_positions for value in joints),
    )


def _ik_search_targets(
    *,
    targets: Mapping[str, object],
    target_names: tuple[str, ...],
    pickup_approach_ik_steps: int,
) -> tuple[_IkSearchTarget, ...]:
    """Expand object-contact approaches into numerical continuation targets."""

    steps = max(1, int(pickup_approach_ik_steps))
    search_targets: list[_IkSearchTarget] = []
    for target_name in target_names:
        pregrasp_target_name = {
            "holder_grasp": "holder_pregrasp",
            "inserter_pickup_grasp": "inserter_pickup_pregrasp",
        }.get(target_name)
        if pregrasp_target_name is None or steps == 1:
            search_targets.append(
                _IkSearchTarget(
                    label=target_name,
                    pose=_pose_target(dict(targets[target_name])),
                    result_target_name=target_name,
                    scene_target_name=target_name,
                )
            )
            continue

        start_payload = dict(targets[pregrasp_target_name])
        final_payload = dict(targets[target_name])
        start_position = np.asarray(start_payload["position_world_m"], dtype=float)
        final_position = np.asarray(final_payload["position_world_m"], dtype=float)
        final_orientation = tuple(float(value) for value in final_payload["orientation_xyzw_world"])
        if start_position.shape != (3,) or final_position.shape != (3,) or len(final_orientation) != 4:
            raise ValueError("Contact pregrasp/grasp targets must contain 3D positions and a quaternion.")
        for step_index in range(1, steps + 1):
            fraction = float(step_index) / float(steps)
            position = start_position + fraction * (final_position - start_position)
            label = f"{target_name}__approach_{step_index:02d}_of_{steps:02d}"
            search_targets.append(
                _IkSearchTarget(
                    label=label,
                    pose=PoseTarget.from_quaternion(
                        x=float(position[0]),
                        y=float(position[1]),
                        z=float(position[2]),
                        quaternion_xyzw=final_orientation,
                        frame_id="base_link",
                    ),
                    # Preserve every accepted continuation state for runtime
                    # path planning. The final step retains the canonical key
                    # expected by existing artifact consumers.
                    result_target_name=(target_name if step_index == steps else label),
                    scene_target_name=target_name,
                )
            )
    return tuple(search_targets)


def _solve_role_ik_branches(
    *,
    commander: MoveItPoseCommander,
    role: str,
    targets: Mapping[str, object],
    target_names: tuple[str, ...],
    initial_robot_state: Mapping[str, float],
    preferred_joint_targets: Mapping[str, tuple[float, ...]],
    candidate_count: int,
    beam_width: int,
    seed_perturbation_rad: float,
    pickup_approach_ik_steps: int,
    post_target_state_updates: Mapping[str, Mapping[str, float]],
    collision_diagnostics: bool,
    phase_obstacles: dict[str, dict[str, object]],
    attached_collision_objects: dict[str, dict[str, object]],
    attachment_state: dict[str, bool],
    kinematic_cache: dict[tuple[object, ...], _KinematicIkCacheEntry] | None,
    state: dict[str, object],
) -> _IkRoleSearchResult:
    """Retain distinct full-state-valid IK branches through one role sequence.

    KDL solves only the active arm's kinematics.  Those solutions are reusable
    when the passive arm changes, but collision validity is not: every cached
    active-arm solution is therefore inserted into the current complete robot
    state and checked again with MoveIt.
    """

    role_joint_names = tuple(str(value) for value in ARM_SPECS[role]["joint_names"])
    initial_branch = _IkPreflightBranch(
        target_joint_positions=(),
        terminal_robot_state=tuple((str(name), float(value)) for name, value in initial_robot_state.items()),
        joint_path_cost=0.0,
    )
    beam = [initial_branch]
    target_records: list[dict[str, object]] = []
    search_targets = _ik_search_targets(
        targets=targets,
        target_names=target_names,
        pickup_approach_ik_steps=pickup_approach_ik_steps,
    )
    for search_target in search_targets:
        target_name = search_target.label
        active_aabbs = _pregrasp_aabb_obstacles_for_target(
            phase_obstacles,
            target_name=search_target.scene_target_name,
        )
        if active_aabbs:
            scene_ok, scene_message = _apply_phase_collision_obstacles(commander, active_aabbs)
            if not scene_ok:
                return _IkRoleSearchResult(
                    branches=(),
                    failure=f"{target_name}: could not apply part collision geometry: {scene_message}",
                    target_records=tuple(target_records),
                    failure_kind="scene_error",
                )
        expanded: list[_IkPreflightBranch] = []
        failure_messages: list[str] = []
        attempted_seed_count = 0
        solution_count = 0
        collision_disabled_solution_count = 0
        kinematic_solution_observed_count = 0
        collision_invalid_state_count = 0
        valid_state_count = 0
        kinematic_failure_count = 0
        invalid_without_contacts_count = 0
        kinematic_cache_hit_count = 0
        kinematic_cache_miss_count = 0
        post_grasp_valid_state_count = 0
        post_grasp_invalid_state_count = 0
        contact_class_counts: dict[str, int] = {}
        contact_pair_counts: dict[str, int] = {}
        diagnostic_examples: list[dict[str, object]] = []
        for parent_index, parent in enumerate(beam):
            parent_state = dict(parent.terminal_robot_state)
            active_start = tuple(float(parent_state[name]) for name in role_joint_names)
            seeds = _exact_ik_seed_candidates(
                active_start,
                preferred_joint_positions=(
                    preferred_joint_targets.get(search_target.result_target_name)
                    if search_target.result_target_name is not None
                    else None
                ),
                candidate_count=candidate_count,
                perturbation_rad=seed_perturbation_rad,
            )
            distinct_solutions: list[tuple[float, ...]] = []
            # The first target creates the requested branch diversity. Once a
            # beam exists, preserve one continuation from every parent instead
            # of creating beam_width**2 children and immediately pruning them.
            solution_limit = int(beam_width) if len(beam) == 1 else 1
            for seed_index, active_seed in enumerate(seeds):
                attempted_seed_count += 1
                seed_robot_state = dict(parent_state)
                seed_robot_state.update((name, value) for name, value in zip(role_joint_names, active_seed))
                kinematic_cache_hit = False
                pose = search_target.pose
                kinematic_cache_key = (
                    str(role),
                    str(pose.frame_id),
                    *(round(float(value), 9) for value in (*pose.position_xyz, *pose.orientation_xyzw)),
                    *(round(float(value), 9) for value in active_seed),
                )
                cached_ik = None if kinematic_cache is None else kinematic_cache.get(kinematic_cache_key)
                if cached_ik is None:
                    ik_started_at = time.monotonic()
                    raw_joints, message = commander.compute_ik(
                        search_target.pose,
                        seed_robot_state=seed_robot_state,
                        avoid_collisions=False,
                    )
                    ik_duration_s = time.monotonic() - ik_started_at
                    cached_ik = _KinematicIkCacheEntry(
                        joints=(None if raw_joints is None else tuple(float(value) for value in raw_joints)),
                        message=str(message),
                    )
                    if kinematic_cache is not None:
                        kinematic_cache[kinematic_cache_key] = cached_ik
                    kinematic_cache_miss_count += 1
                    state["ik_kinematic_cache_misses"] = int(state["ik_kinematic_cache_misses"]) + 1
                    state["ik_seed_calls"] = int(state["ik_seed_calls"]) + 1
                    state["ik_request_duration_s"] = float(state["ik_request_duration_s"]) + float(ik_duration_s)
                    diagnostics = state.get("collision_diagnostics")
                    if isinstance(diagnostics, dict):
                        diagnostics["kinematic_cache_misses"] = int(diagnostics["kinematic_cache_misses"]) + 1
                        diagnostics["ik_requests"] = int(diagnostics["ik_requests"]) + 1
                        diagnostics["ik_request_duration_s"] = (
                            float(diagnostics["ik_request_duration_s"]) + float(ik_duration_s)
                        )
                else:
                    kinematic_cache_hit = True
                    ik_duration_s = 0.0
                    kinematic_cache_hit_count += 1
                    state["ik_kinematic_cache_hits"] = int(state["ik_kinematic_cache_hits"]) + 1
                    diagnostics = state.get("collision_diagnostics")
                    if isinstance(diagnostics, dict):
                        diagnostics["kinematic_cache_hits"] = int(diagnostics["kinematic_cache_hits"]) + 1
                joints = None if cached_ik.joints is None else list(cached_ik.joints)
                message = cached_ik.message
                diagnostics = state.get("collision_diagnostics")
                if joints is None:
                    if not kinematic_cache_hit:
                        kinematic_failure_count += 1
                        state["ik_kinematic_failures"] = int(state["ik_kinematic_failures"]) + 1
                        if isinstance(diagnostics, dict):
                            diagnostics["kinematic_or_numerical_failures"] = (
                                int(diagnostics["kinematic_or_numerical_failures"]) + 1
                            )
                    cache_note = " (cached)" if kinematic_cache_hit else ""
                    failure_messages.append(f"parent={parent_index} seed={seed_index}: {message}{cache_note}")
                    continue
                kinematic_solution_observed_count += 1
                if not kinematic_cache_hit:
                    collision_disabled_solution_count += 1
                    state["ik_kinematic_solutions_returned"] = int(state["ik_kinematic_solutions_returned"]) + 1
                    if isinstance(diagnostics, dict):
                        diagnostics["collision_disabled_ik_solutions"] = (
                            int(diagnostics["collision_disabled_ik_solutions"]) + 1
                        )
                candidate_robot_state = dict(parent_state)
                candidate_robot_state.update(
                    (name, float(value)) for name, value in zip(role_joint_names, joints)
                )
                validity, validity_message = commander.check_state_validity(candidate_robot_state, group_name="")
                state["ik_state_validity_requests"] = int(state["ik_state_validity_requests"]) + 1
                if isinstance(diagnostics, dict):
                    diagnostics["state_validity_requests"] = int(diagnostics["state_validity_requests"]) + 1
                if validity is None:
                    failure_messages.append(
                        f"parent={parent_index} seed={seed_index}: kinematic IK found a state but "
                        f"state validity failed: {validity_message}"
                    )
                    continue
                if not bool(validity["valid"]):
                    collision_invalid_state_count += 1
                    state["ik_invalid_states"] = int(state["ik_invalid_states"]) + 1
                    contacts = [dict(value) for value in validity.get("contacts", [])]
                    if not contacts:
                        invalid_without_contacts_count += 1
                        state["ik_invalid_states_without_contacts"] = (
                            int(state["ik_invalid_states_without_contacts"]) + 1
                        )
                    if isinstance(diagnostics, dict):
                        diagnostics["invalid_states"] = int(diagnostics["invalid_states"]) + 1
                        if not contacts:
                            diagnostics["invalid_states_without_contacts"] = (
                                int(diagnostics["invalid_states_without_contacts"]) + 1
                            )
                        for contact in contacts:
                            contact_class = _collision_contact_class(contact)
                            contact_pair = _canonical_contact_pair(contact)
                            contact_class_counts[contact_class] = int(contact_class_counts.get(contact_class, 0)) + 1
                            contact_pair_counts[contact_pair] = int(contact_pair_counts.get(contact_pair, 0)) + 1
                    if collision_diagnostics and len(diagnostic_examples) < 8:
                        diagnostic_examples.append(
                            {
                                "parent_index": int(parent_index),
                                "seed_index": int(seed_index),
                                "ik_duration_s": float(ik_duration_s),
                                "contacts": contacts,
                            }
                        )
                    if collision_diagnostics:
                        _record_collision_diagnostic(
                            state,
                            role=role,
                            target=target_name,
                            parent_index=parent_index,
                            seed_index=seed_index,
                            contacts=contacts,
                        )
                    contact_summary = ", ".join(sorted({_canonical_contact_pair(value) for value in contacts}))
                    failure_messages.append(
                        f"parent={parent_index} seed={seed_index}: kinematic IK state is invalid"
                        + (f" ({contact_summary})" if contact_summary else " (no contacts returned)")
                    )
                    continue
                state["ik_valid_states"] = int(state["ik_valid_states"]) + 1
                if isinstance(diagnostics, dict):
                    diagnostics["valid_states"] = int(diagnostics["valid_states"]) + 1
                valid_state_count += 1
                post_state_update = (
                    post_target_state_updates.get(search_target.result_target_name, {})
                    if search_target.result_target_name is not None
                    else {}
                )
                if post_state_update:
                    post_grasp_state = dict(candidate_robot_state)
                    post_grasp_state.update(
                        (str(name), float(value)) for name, value in post_state_update.items()
                    )
                    validity_checker = getattr(commander, "check_state_validity", None)
                    if callable(validity_checker):
                        post_validity, post_validity_message = validity_checker(
                            post_grasp_state,
                            group_name="",
                        )
                    else:
                        post_validity, post_validity_message = ({"valid": True, "contacts": []}, "not available")
                    state["post_grasp_state_validity_requests"] = (
                        int(state["post_grasp_state_validity_requests"]) + 1
                    )
                    if post_validity is None:
                        failure_messages.append(
                            f"parent={parent_index} seed={seed_index}: post-grasp state validity failed: "
                            f"{post_validity_message}"
                        )
                        continue
                    if not bool(post_validity["valid"]):
                        post_grasp_invalid_state_count += 1
                        state["post_grasp_invalid_states"] = int(state["post_grasp_invalid_states"]) + 1
                        contacts = [dict(value) for value in post_validity.get("contacts", [])]
                        contact_summary = ", ".join(sorted({_canonical_contact_pair(value) for value in contacts}))
                        failure_messages.append(
                            f"parent={parent_index} seed={seed_index}: post-grasp closed state is invalid"
                            + (f" ({contact_summary})" if contact_summary else " (no contacts returned)")
                        )
                        continue
                    post_grasp_valid_state_count += 1
                    state["post_grasp_valid_states"] = int(state["post_grasp_valid_states"]) + 1
                state["ik_solutions_found"] = int(state["ik_solutions_found"]) + 1
                solution = tuple(float(value) for value in joints)
                if not _is_distinct_ik_solution(solution, distinct_solutions):
                    continue
                distinct_solutions.append(solution)
                solution_count += 1
                state["ik_distinct_solutions_retained"] = int(state["ik_distinct_solutions_retained"]) + 1
                if len(distinct_solutions) >= solution_limit:
                    break

            for solution in distinct_solutions:
                terminal_state = dict(parent_state)
                terminal_state.update((name, value) for name, value in zip(role_joint_names, solution))
                if search_target.result_target_name is not None:
                    terminal_state.update(
                        (str(name), float(value))
                        for name, value in post_target_state_updates.get(
                            search_target.result_target_name,
                            {},
                        ).items()
                    )
                active_delta = np.asarray(solution, dtype=float) - np.asarray(active_start, dtype=float)
                edge_cost = float(
                    np.linalg.norm(active_delta * np.asarray(KUKA_MOVEIT_TRANSITION_JOINT_WEIGHTS, dtype=float))
                )
                target_joint_positions = dict(parent.target_joint_positions)
                if search_target.result_target_name is not None:
                    target_joint_positions[search_target.result_target_name] = solution
                expanded.append(
                    _IkPreflightBranch(
                        target_joint_positions=tuple(target_joint_positions.items()),
                        terminal_robot_state=tuple(terminal_state.items()),
                        joint_path_cost=float(parent.joint_path_cost) + edge_cost,
                    )
                )

        target_record = {
            "target": target_name,
            "result_target": search_target.result_target_name,
            "ok": bool(expanded),
            "input_branches": len(beam),
            "seed_attempts": attempted_seed_count,
            "ik_requests": int(kinematic_cache_miss_count),
            "kinematic_cache_hits": int(kinematic_cache_hit_count),
            "kinematic_cache_misses": int(kinematic_cache_miss_count),
            "distinct_solutions": solution_count,
            "output_branches_before_prune": len(expanded),
            "output_branches_retained": min(len(expanded), int(beam_width)),
            "seed_mode": "complete_dual_arm_multi_seed",
            "last_failure": failure_messages[-1] if failure_messages else "",
            "post_grasp_valid_states": int(post_grasp_valid_state_count),
            "post_grasp_invalid_states": int(post_grasp_invalid_state_count),
        }
        if collision_diagnostics:
            target_record["collision_diagnostics"] = {
                "collision_disabled_ik_solutions": int(collision_disabled_solution_count),
                "kinematic_or_numerical_failures": int(kinematic_failure_count),
                "valid_states": int(valid_state_count),
                "invalid_states": int(collision_invalid_state_count),
                "invalid_states_without_contacts": int(invalid_without_contacts_count),
                "contact_class_counts": contact_class_counts,
                "contact_pair_counts": contact_pair_counts,
                "examples": diagnostic_examples,
            }
        target_records.append(target_record)
        if active_aabbs:
            remove_ok, remove_message = _remove_phase_collision_obstacles(commander, active_aabbs)
            if not remove_ok:
                return _IkRoleSearchResult(
                    branches=(),
                    failure=f"{target_name}: could not remove part collision geometry: {remove_message}",
                    target_records=tuple(target_records),
                    failure_kind="scene_error",
                )
        if not expanded:
            detail = failure_messages[-1] if failure_messages else "no distinct IK solution"
            if kinematic_solution_observed_count == 0:
                failure_kind = "kinematic_no_ik"
            elif collision_invalid_state_count or post_grasp_invalid_state_count:
                failure_kind = "state_collision"
            else:
                failure_kind = "state_validity_error"
            return _IkRoleSearchResult(
                branches=(),
                failure=(
                    f"{target_name}: {detail} across {attempted_seed_count} seed evaluation(s), "
                    f"{kinematic_cache_miss_count} IK request(s)"
                ),
                target_records=tuple(target_records),
                failure_kind=failure_kind,
            )
        beam = sorted(expanded, key=_ik_branch_sort_key)[: max(1, int(beam_width))]
        if search_target.result_target_name == "inserter_pickup_grasp" and attached_collision_objects:
            attach_ok, attach_message = _apply_attached_collision_objects(
                commander,
                attached_collision_objects,
            )
            if not attach_ok:
                return _IkRoleSearchResult(
                    branches=(),
                    failure=f"{target_name}: could not attach incoming collision geometry: {attach_message}",
                    target_records=tuple(target_records),
                    failure_kind="scene_error",
                )
            attachment_state["active"] = True

    return _IkRoleSearchResult(
        branches=tuple(beam),
        failure="",
        target_records=tuple(target_records),
    )


def _ik_preflight_pair(
    task,
    *,
    commanders: dict[str, MoveItPoseCommander],
    feasible_cache: dict[str, dict[tuple[object, ...], _IkPreflightCacheEntry]],
    kinematic_cache: dict[tuple[object, ...], _KinematicIkCacheEntry] | None = None,
    state: dict[str, object],
    rank: int,
    roles: tuple[str, ...] = ("holder", "inserter"),
    preferred_joint_targets: dict[str, tuple[float, ...]] | None = None,
    initial_robot_state: Mapping[str, float] | None = None,
    ik_candidate_count: int = 1,
    ik_beam_width: int = 1,
    ik_seed_perturbation_rad: float = 0.60,
    pickup_approach_ik_steps: int = 1,
    collision_diagnostics: bool = False,
) -> tuple[bool, str, dict[str, tuple[float, ...]]]:
    """Screen one pair using coordinated complete-state multi-branch IK."""

    task_payload = task.to_payload()
    targets = dict(task_payload["targets"])
    grasp_ids = {
        "holder": task.holder_candidate.grasp_id,
        "inserter": task.inserter_candidate.grasp_id,
    }
    pair_role_records: dict[str, object] = {}
    preferred_joint_targets = preferred_joint_targets or {}
    task_initial_robot_state = dict(initial_robot_state or _complete_dual_arm_start_state())
    task_initial_robot_state.update(_task_approach_gripper_state(task))
    post_target_state_updates = _task_post_grasp_state_updates(task)
    if isinstance(task_payload.get("objects"), dict):
        phase_obstacles = simple_dual_robot_pregrasp_aabb_obstacles(task)
        attached_collision_objects = simple_dual_robot_attached_collision_objects(task)
    else:
        # Unit-level synthetic tasks exercise the IK search without mesh data.
        phase_obstacles = {}
        attached_collision_objects = {}
    pair_beam = [
        _IkPreflightBranch(
            target_joint_positions=(),
            terminal_robot_state=tuple(
                (str(name), float(value))
                for name, value in task_initial_robot_state.items()
            ),
            joint_path_cost=0.0,
        )
    ]
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
        expanded_pair_beam: list[_IkPreflightBranch] = []
        cache_hits = 0
        cache_misses = 0
        role_failures: list[str] = []
        role_failure_kinds: list[str] = []
        for parent in pair_beam:
            input_robot_state = dict(parent.terminal_robot_state)
            cache_key = (
                grasp_id,
                target_signature,
                _robot_state_signature(input_robot_state),
                int(ik_candidate_count),
                int(ik_beam_width),
                round(float(ik_seed_perturbation_rad), 9),
                int(pickup_approach_ik_steps),
                bool(collision_diagnostics),
            )
            cache_hit = cache_key in feasible_cache[role]
            if cache_hit:
                cache_hits += 1
                cache_entry = feasible_cache[role][cache_key]
            else:
                cache_misses += 1
                attachment_state = {"active": False}
                try:
                    search_result = _solve_role_ik_branches(
                        commander=commanders[role],
                        role=role,
                        targets=targets,
                        target_names=target_names,
                        initial_robot_state=input_robot_state,
                        preferred_joint_targets=preferred_joint_targets,
                        candidate_count=ik_candidate_count,
                        beam_width=ik_beam_width,
                        seed_perturbation_rad=ik_seed_perturbation_rad,
                        pickup_approach_ik_steps=pickup_approach_ik_steps,
                        post_target_state_updates=post_target_state_updates,
                        collision_diagnostics=collision_diagnostics,
                        phase_obstacles=phase_obstacles,
                        attached_collision_objects=(attached_collision_objects if role == "inserter" else {}),
                        attachment_state=attachment_state,
                        kinematic_cache=kinematic_cache,
                        state=state,
                    )
                finally:
                    if (
                        role == "inserter"
                        and attached_collision_objects
                        and attachment_state["active"]
                    ):
                        detach_ok, detach_message = _remove_attached_collision_objects(
                            commanders[role],
                            attached_collision_objects,
                        )
                        if not detach_ok:
                            raise RuntimeError(
                                f"Could not clean up incoming attached collision geometry: {detach_message}"
                            )
                        attachment_state["active"] = False
                cache_entry = _IkPreflightCacheEntry(
                    feasible=bool(search_result.branches),
                    failure=search_result.failure,
                    branches=search_result.branches,
                    failure_kind=search_result.failure_kind,
                )
                feasible_cache[role][cache_key] = cache_entry
                records = state["records"]
                assert isinstance(records, dict)
                role_records = records[role]
                assert isinstance(role_records, list)
                role_records.append(
                    {
                        "grasp_id": grasp_id,
                        "feasible": cache_entry.feasible,
                        "input_robot_state": list(_robot_state_signature(input_robot_state)),
                        "branches_retained": len(cache_entry.branches),
                        "targets": list(search_result.target_records),
                    }
                )
                checked_key = f"{role}_grasps_checked"
                feasible_key = f"{role}_grasps_feasible"
                state[checked_key] = int(state[checked_key]) + 1
                if cache_entry.feasible:
                    state[feasible_key] = int(state[feasible_key]) + 1

            if not cache_entry.feasible:
                role_failures.append(cache_entry.failure)
                role_failure_kinds.append(cache_entry.failure_kind)
                continue
            for role_branch in cache_entry.branches:
                combined_targets = dict(parent.target_joint_positions)
                combined_targets.update(dict(role_branch.target_joint_positions))
                expanded_pair_beam.append(
                    _IkPreflightBranch(
                        target_joint_positions=tuple(combined_targets.items()),
                        terminal_robot_state=role_branch.terminal_robot_state,
                        joint_path_cost=float(parent.joint_path_cost) + float(role_branch.joint_path_cost),
                    )
                )

        grasp_feasible = bool(expanded_pair_beam)
        if grasp_feasible:
            pair_beam = sorted(expanded_pair_beam, key=_ik_branch_sort_key)[: max(1, int(ik_beam_width))]
        elif cache_misses == 0 and cache_hits:
            failure = f"{role} grasp {grasp_id} reused a cached IK failure: {role_failures[-1]}"
        else:
            detail = role_failures[-1] if role_failures else "no complete-state branch survived"
            failure = f"{role} grasp {grasp_id} failed {detail}"
        failure_kind = role_failure_kinds[-1] if role_failure_kinds else ""
        pair_role_records[role] = {
            "grasp_id": grasp_id,
            "cache_hit": cache_misses == 0,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "input_branches": cache_hits + cache_misses,
            "output_branches": len(pair_beam) if grasp_feasible else 0,
            "feasible": grasp_feasible,
            "failure_kind": failure_kind,
        }
        if not grasp_feasible:
            break

    pair_feasible = not failure
    state["pair_tasks_checked"] = int(state["pair_tasks_checked"]) + 1
    if pair_feasible:
        state["pair_tasks_after"] = int(state["pair_tasks_after"]) + 1
    pair_records = state["pair_records"]
    assert isinstance(pair_records, list)
    resolved_joint_targets = dict(pair_beam[0].target_joint_positions) if pair_feasible else {}
    pair_record: dict[str, object] = {
        "rank": int(rank),
        "candidate_rank": int(getattr(task, "candidate_rank", rank)),
        "pair_id": task.pair_id,
        "transition_id": str(getattr(task, "transition_id", "")),
        "execution_candidate_id": str(getattr(task, "execution_candidate_id", task.pair_id)),
        "selection_score": float(task.selection_score),
        "feasible": pair_feasible,
        "failure": failure,
        "failure_kind": failure_kind if failure else "",
        "roles": pair_role_records,
    }
    if resolved_joint_targets:
        pair_record["validated_joint_target_order"] = list(resolved_joint_targets)
        pair_record["validated_joint_targets"] = {
            str(name): [float(value) for value in joints]
            for name, joints in resolved_joint_targets.items()
        }
    pair_records.append(pair_record)
    return pair_feasible, failure, resolved_joint_targets


def _is_retryable_pickup_kinematic_failure(pair_record: Mapping[str, object]) -> bool:
    """Whether shortening only the pickup approach can address this failure."""

    failure = str(pair_record.get("failure", ""))
    return (
        str(pair_record.get("failure_kind", "")) == "kinematic_no_ik"
        and "inserter grasp " in failure
        and (
            "inserter_pickup_pregrasp:" in failure
            or "inserter_pickup_grasp__approach_" in failure
        )
    )


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


def _configure_gripper_model(gripper_model: str) -> None:
    """Select the dual MoveIt TCP links before commanders are constructed."""

    global ARM_SPEC_BY_ROBOT, ARM_SPECS
    model = str(gripper_model)
    tcp_suffix = "pdz_gripper_tcp" if model == "pdz_gripper" else "gripper_tcp"
    ARM_SPEC_BY_ROBOT = {
        robot: {
            "robot": robot,
            "planning_group": "arm_one" if robot == "lbr_one" else "arm_two",
            "pose_link": f"{robot}_{tcp_suffix}",
            "joint_names": tuple(f"{robot}_A{index}" for index in range(1, 8)),
        }
        for robot in ("lbr_one", "lbr_two")
    }
    ARM_SPECS = {
        "holder": dict(ARM_SPEC_BY_ROBOT["lbr_one"]),
        "inserter": dict(ARM_SPEC_BY_ROBOT["lbr_two"]),
    }


def main() -> int:
    args = _parse_args()
    _configure_gripper_model(str(args.gripper_model))
    if rclpy is None:
        raise RuntimeError(
            "ROS2 MoveIt dependencies are unavailable. Source ROS2 and both workspaces before running this planner."
        )
    if args.max_pair_attempts < 1:
        raise ValueError("--max-pair-attempts must be at least 1.")
    if args.max_ik_screen_candidates < 0:
        raise ValueError("--max-ik-screen-candidates must be non-negative.")
    if args.joint_rank_candidates < 0:
        raise ValueError("--joint-rank-candidates must be non-negative.")
    if args.joint_rank_ik_candidates < 1:
        raise ValueError("--joint-rank-ik-candidates must be at least 1.")
    if args.joint_rank_beam_width < 1:
        raise ValueError("--joint-rank-beam-width must be at least 1.")
    if args.exact_ik_candidates < 1:
        raise ValueError("--exact-ik-candidates must be at least 1.")
    if args.exact_ik_beam_width < 1:
        raise ValueError("--exact-ik-beam-width must be at least 1.")
    if args.exact_ik_seed_perturbation_rad < 0.0:
        raise ValueError("--exact-ik-seed-perturbation-rad must be non-negative.")
    if args.pickup_approach_ik_steps < 1:
        raise ValueError("--pickup-approach-ik-steps must be at least 1.")
    pickup_pregrasp_offsets_m = tuple(float(value) for value in args.pickup_pregrasp_offsets_m)
    if not pickup_pregrasp_offsets_m or any(
        not math.isfinite(value) or value <= 0.0 for value in pickup_pregrasp_offsets_m
    ):
        raise ValueError("--pickup-pregrasp-offsets-m must contain finite positive distances.")
    if any(
        next_offset >= offset
        for offset, next_offset in zip(
            pickup_pregrasp_offsets_m,
            pickup_pregrasp_offsets_m[1:],
        )
    ):
        raise ValueError("--pickup-pregrasp-offsets-m must be strictly decreasing.")
    if args.ik_timeout_s <= 0.0:
        raise ValueError("--ik-timeout-s must be positive.")
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
    artifact_models = {
        str(getattr(task, "pickup_gripper_collision_model", "kuka_y_gripper"))
        for task in tasks
    }
    requested_artifact_model = "pdz_gripper" if str(args.gripper_model) == "pdz_gripper" else "kuka_y_gripper"
    if artifact_models != {requested_artifact_model}:
        raise ValueError(
            "The selected MoveIt gripper model does not match the Stage-3 artifacts: "
            f"requested={requested_artifact_model} artifacts={sorted(artifact_models)}."
        )
    tasks = [
        with_inserter_pickup_pregrasp_offset(task, pickup_pregrasp_offsets_m[0])
        for task in tasks
    ]
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
    task_count_before_runtime_selection = len(tasks)
    tasks = _runtime_ik_screen_queue(tasks, holder_only=bool(args.holder_only))
    if not tasks:
        raise RuntimeError("No ranked compatible pair is available to plan.")
    debug_candidate_counts = dict(getattr(tasks[0], "candidate_filter_diagnostics", {}))
    debug_candidate_counts.update(
        {
            "planner_queue_execution_candidates": 0,
            "planner_queue_source_execution_candidates": int(task_count_before_runtime_selection),
            "planner_queue_ik_screen_candidates": len(tasks),
            "planner_queue_path_candidate_limit": int(args.max_pair_attempts),
            "planner_queue_selection": (
                "broad_noncrossing_then_round_robin_unique_pickup_exact_ik_"
                "then_bounded_path_candidates"
            ),
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
            "exact_ik_seed_calls": 0,
            "exact_ik_kinematic_cache_hits": 0,
            "exact_ik_state_validity_requests": 0,
            "exact_ik_solutions_found": 0,
            "exact_ik_distinct_solutions_retained": 0,
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
        else _new_ik_preflight_state(
            pair_task_count=len(tasks),
            ik_candidate_count=int(args.exact_ik_candidates),
            ik_beam_width=int(args.exact_ik_beam_width),
            ik_seed_perturbation_rad=float(args.exact_ik_seed_perturbation_rad),
            pickup_approach_ik_steps=int(args.pickup_approach_ik_steps),
            collision_diagnostics=bool(args.ik_collision_diagnostics),
        )
    )
    ik_preflight["pickup_pregrasp_offset_candidates_m"] = list(
        pickup_pregrasp_offsets_m
    )
    ik_preflight["pickup_pregrasp_offset_attempts"] = []
    ik_feasible_cache: dict[str, dict[tuple[object, ...], _IkPreflightCacheEntry]] = {
        role: {} for role in IK_PREFLIGHT_TARGETS
    }
    # Separate active-arm kinematics from complete-state validity. Keep the
    # former across pair/holder variants; every reuse is still revalidated with
    # the current passive arm and finger state inside _solve_role_ik_branches.
    ik_kinematic_cache: dict[tuple[object, ...], _KinematicIkCacheEntry] = {}
    try:
        active_roles = ("holder",) if args.holder_only else tuple(ARM_SPECS)
        initial_robot_state = _complete_dual_arm_start_state(
            holder_start_joint_positions=(
                None
                if args.holder_start_joint_positions is None
                else tuple(float(value) for value in args.holder_start_joint_positions)
            )
        )
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
            commander.wait_for_moveit(require_execute=not bool(args.ik_only))
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
            message=(
                "IK-only mode uses the explicit nominal complete robot state without motion."
                if args.ik_only
                else "Returning both mock arms to the nominal start state."
            ),
        )
        if args.ik_only:
            reset_ok, reset_messages = True, {"mode": "explicit_state_no_motion"}
        else:
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

        if args.ik_only:
            joint_space_ranking = {
                "skipped": True,
                "reason": "ik_only",
                "candidate_count_before": len(tasks),
            }
        elif args.holder_only:
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
        path_candidate_limit = 1 if bool(args.ik_only) else int(args.max_pair_attempts)
        ik_screen_candidate_limit = int(args.max_ik_screen_candidates)
        ik_screen_bound = (
            len(tasks)
            if ik_screen_candidate_limit == 0
            else min(len(tasks), ik_screen_candidate_limit)
        )
        ik_screened_candidate_count = 0
        ik_feasible_candidate_count = 0
        selected_screen_ranks: list[int] = []
        selected_candidate_ranks: list[int] = []
        exact_ik_selection: dict[str, object] = {}
        pickup_offset_attempts_by_screen_rank: dict[int, list[dict[str, object]]] = {}

        def update_exact_ik_selection(stop_reason: str) -> None:
            exact_ik_selection.clear()
            exact_ik_selection.update(
                {
                    "source_candidate_count": len(tasks),
                    "ik_screen_candidate_limit": (
                        None if ik_screen_candidate_limit == 0 else ik_screen_candidate_limit
                    ),
                    "candidates_screened": ik_screened_candidate_count,
                    "ik_feasible_candidates": ik_feasible_candidate_count,
                    "path_candidate_limit": path_candidate_limit,
                    "stop_reason": stop_reason,
                    "selected_screen_ranks": list(selected_screen_ranks),
                    "selected_candidate_ranks": list(selected_candidate_ranks),
                }
            )
            debug_candidate_counts.update(
                {
                    "planner_queue_execution_candidates": ik_feasible_candidate_count,
                    "planner_queue_ik_candidates_screened": ik_screened_candidate_count,
                    "planner_queue_exact_ik_feasible_candidates": ik_feasible_candidate_count,
                    "planner_queue_selected_candidate_ranks": list(selected_candidate_ranks),
                    "planner_queue_selected_screen_ranks": list(selected_screen_ranks),
                    "planner_queue_stop_reason": stop_reason,
                }
            )
            ik_preflight["candidate_selection"] = exact_ik_selection

        update_exact_ik_selection("screening_not_started")

        def evaluate_exact_ik(task, screen_rank: int):
            nonlocal last_task, ik_screened_candidate_count, ik_feasible_candidate_count
            last_task = task
            ik_screened_candidate_count += 1
            update_debug(
                task=task,
                attempt_index=screen_rank,
                phase="ik_preflight",
                status="planning",
                message=(
                    "Checking cached kinematic IK plus complete dual-arm state validity through the sequence."
                ),
            )
            print(
                f"[DUAL-SIM-PLAN] IK screen {screen_rank}/{len(tasks)} "
                f"candidate_rank={int(getattr(task, 'candidate_rank', screen_rank))} "
                f"pair={task.pair_id} transition={task.transition_id} "
                f"pair_score={task.pair_score:.4f} "
                f"selection_score={task.selection_score:.4f} "
                f"layout_proxy={task.layout_proxy_score:.4f}",
                flush=True,
            )
            if bool(args.skip_ik_preflight):
                ik_feasible_candidate_count += 1
                selected_screen_ranks.append(screen_rank)
                producer_rank = int(getattr(task, "candidate_rank", 0))
                selected_candidate_ranks.append(
                    producer_rank if producer_rank > 0 else screen_rank
                )
                update_exact_ik_selection("path_candidate_admitted")
                return (
                    True,
                    "",
                    dict(preferred_joint_targets_by_candidate.get(task.execution_candidate_id, {})),
                )

            selected_task = task
            pair_ik_ok = False
            pair_ik_failure = ""
            preflight_joint_targets: dict[str, tuple[float, ...]] = {}
            offset_attempts: list[dict[str, object]] = []
            pair_tasks_checked_before_offsets = int(
                ik_preflight["pair_tasks_checked"]
            )
            pair_tasks_after_before_offsets = int(ik_preflight["pair_tasks_after"])
            for offset_attempt_index, pickup_pregrasp_offset_m in enumerate(
                pickup_pregrasp_offsets_m,
                start=1,
            ):
                selected_task = with_inserter_pickup_pregrasp_offset(
                    task,
                    pickup_pregrasp_offset_m,
                )
                pair_ik_ok, pair_ik_failure, preflight_joint_targets = _ik_preflight_pair(
                    selected_task,
                    commanders=commanders,
                    feasible_cache=ik_feasible_cache,
                    kinematic_cache=ik_kinematic_cache,
                    state=ik_preflight,
                    rank=screen_rank,
                    roles=active_roles,
                    preferred_joint_targets=(
                        preferred_joint_targets_by_candidate.get(
                            task.execution_candidate_id,
                            {},
                        )
                    ),
                    initial_robot_state=initial_robot_state,
                    ik_candidate_count=int(args.exact_ik_candidates),
                    ik_beam_width=int(args.exact_ik_beam_width),
                    ik_seed_perturbation_rad=float(args.exact_ik_seed_perturbation_rad),
                    pickup_approach_ik_steps=int(args.pickup_approach_ik_steps),
                    collision_diagnostics=bool(args.ik_collision_diagnostics),
                )
                pair_records = ik_preflight["pair_records"]
                assert isinstance(pair_records, list)
                pair_record = pair_records[-1]
                assert isinstance(pair_record, dict)
                retryable = (
                    not pair_ik_ok
                    and not bool(args.holder_only)
                    and _is_retryable_pickup_kinematic_failure(pair_record)
                )
                offset_record = {
                    "screen_rank": screen_rank,
                    "candidate_rank": int(
                        getattr(task, "candidate_rank", screen_rank)
                    ),
                    "pair_id": task.pair_id,
                    "execution_candidate_id": task.execution_candidate_id,
                    "offset_attempt_index": offset_attempt_index,
                    "pickup_pregrasp_offset_m": pickup_pregrasp_offset_m,
                    "success": bool(pair_ik_ok),
                    "failure": pair_ik_failure,
                    "failure_kind": str(pair_record.get("failure_kind", "")),
                    "retryable_with_shorter_offset": bool(retryable),
                }
                pair_record.update(offset_record)
                offset_attempts.append(offset_record)
                recorded_offset_attempts = ik_preflight[
                    "pickup_pregrasp_offset_attempts"
                ]
                assert isinstance(recorded_offset_attempts, list)
                recorded_offset_attempts.append(offset_record)
                if pair_ik_ok or not retryable:
                    break
                print(
                    "[DUAL-SIM-PLAN] Pickup pregrasp has pure kinematic "
                    f"no-IK at {pickup_pregrasp_offset_m:.3f} m; trying the "
                    "next shorter collision-checked approach.",
                    flush=True,
                )
            # These counters describe producer-ranked candidates, not the
            # number of adaptive offset probes. The latter are recorded in
            # pickup_pregrasp_offset_attempts and on each pair record.
            ik_preflight["pair_tasks_checked"] = pair_tasks_checked_before_offsets + 1
            ik_preflight["pair_tasks_after"] = (
                pair_tasks_after_before_offsets + int(pair_ik_ok)
            )
            pickup_offset_attempts_by_screen_rank[screen_rank] = offset_attempts
            last_task = selected_task
            checked = int(ik_preflight["pair_tasks_checked"])
            holder_checked = int(ik_preflight["holder_grasps_checked"])
            inserter_checked = int(ik_preflight["inserter_grasps_checked"])
            debug_candidate_counts.update(
                {
                    "exact_ik_pair_tasks_checked": checked,
                    "exact_ik_holder_grasps_checked": holder_checked,
                    "exact_ik_inserter_grasps_checked": inserter_checked,
                    "exact_ik_seed_calls": int(ik_preflight["ik_seed_calls"]),
                    "exact_ik_kinematic_cache_hits": int(ik_preflight["ik_kinematic_cache_hits"]),
                    "exact_ik_state_validity_requests": int(ik_preflight["ik_state_validity_requests"]),
                    "exact_ik_solutions_found": int(ik_preflight["ik_solutions_found"]),
                    "exact_ik_distinct_solutions_retained": int(ik_preflight["ik_distinct_solutions_retained"]),
                }
            )
            if not pair_ik_ok:
                update_debug(
                    task=task,
                    attempt_index=screen_rank,
                    phase="ik_preflight",
                    status="failed",
                    message=pair_ik_failure,
                )
                print(
                    "[DUAL-SIM-PLAN] IK preflight "
                    f"screen rank {screen_rank}/{len(tasks)} failed: "
                    f"{pair_ik_failure}. Checked {checked} pair(s), "
                    f"{holder_checked} holder and {inserter_checked} "
                    "inserter grasp(s); screening the next candidate.",
                    flush=True,
                )
                attempt_records.append(
                    {
                        "attempt_index": screen_rank,
                        "candidate_rank": int(getattr(task, "candidate_rank", screen_rank)),
                        "screen_rank": screen_rank,
                        "path_attempt_index": None,
                        "phase": "exact_ik_preflight",
                        "pair_id": task.pair_id,
                        "transition_id": task.transition_id,
                        "execution_candidate_id": task.execution_candidate_id,
                        "score": task.pair_score,
                        "selection_score": task.selection_score,
                        "pickup_top_down_score": task.pickup_top_down_score,
                        "layout_proxy_score": task.layout_proxy_score,
                        "holder_reachability_proxy_score": task.holder_reachability_proxy_score,
                        "inserter_reachability_proxy_score": task.inserter_reachability_proxy_score,
                        "success": False,
                        "failure": f"ik_preflight: {pair_ik_failure}",
                        "pickup_pregrasp_offset_attempts": offset_attempts,
                        "steps": [],
                    }
                )
                update_exact_ik_selection("screening")
                return False, pair_ik_failure, {}, selected_task

            print(
                "[DUAL-SIM-PLAN] IK preflight "
                f"screen rank {screen_rank}/{len(tasks)} passed; "
                "admitting it to the bounded path-planning pool.",
                flush=True,
            )
            update_debug(
                task=selected_task,
                attempt_index=screen_rank,
                phase="ik_preflight",
                status="succeeded",
                message=("Complete-state target IK passed; its full validated joint chain is retained."),
            )
            ik_feasible_candidate_count += 1
            selected_screen_ranks.append(screen_rank)
            producer_rank = int(getattr(task, "candidate_rank", 0))
            selected_candidate_ranks.append(producer_rank if producer_rank > 0 else screen_rank)
            update_exact_ik_selection("path_candidate_admitted")
            return True, "", preflight_joint_targets, selected_task

        exact_ik_candidates = _iter_exact_ik_feasible_candidates(
            tasks,
            path_candidate_limit=path_candidate_limit,
            ik_screen_candidate_limit=ik_screen_candidate_limit,
            evaluate=evaluate_exact_ik,
        )
        path_attempt_count = 0
        for path_attempt_index, selected_candidate in enumerate(exact_ik_candidates, start=1):
            path_attempt_count = path_attempt_index
            task = selected_candidate.task
            last_task = task
            attempt_index = selected_candidate.screen_rank
            preferred_joint_targets_by_candidate[
                task.execution_candidate_id
            ] = selected_candidate.joint_targets
            task_payload = task.to_payload()
            targets = dict(task_payload["targets"])
            pregrasp_aabb_obstacles = simple_dual_robot_pregrasp_aabb_obstacles(task)
            pregrasp_aabb_schedule = simple_dual_robot_pregrasp_aabb_schedule(pregrasp_aabb_obstacles)
            attached_collision_objects = simple_dual_robot_attached_collision_objects(task)
            print(
                f"[DUAL-SIM-PLAN] Path attempt {path_attempt_index} "
                f"(limit {path_candidate_limit}) "
                f"from screen_rank={selected_candidate.screen_rank} "
                f"candidate_rank={selected_candidate.candidate_rank} "
                f"pair={task.pair_id} transition={task.transition_id}",
                flush=True,
            )
            if args.ik_only:
                update_exact_ik_selection("candidate_succeeded")
                attempt_records.append(
                    {
                        "attempt_index": attempt_index,
                        "candidate_rank": selected_candidate.candidate_rank,
                        "screen_rank": selected_candidate.screen_rank,
                        "path_attempt_index": path_attempt_index,
                        "phase": "exact_ik_preflight",
                        "pair_id": task.pair_id,
                        "transition_id": task.transition_id,
                        "execution_candidate_id": task.execution_candidate_id,
                        "score": task.pair_score,
                        "selection_score": task.selection_score,
                        "pickup_top_down_score": task.pickup_top_down_score,
                        "layout_proxy_score": task.layout_proxy_score,
                        "holder_reachability_proxy_score": task.holder_reachability_proxy_score,
                        "inserter_reachability_proxy_score": task.inserter_reachability_proxy_score,
                        "success": True,
                        "failure": "",
                        "steps": [],
                        "mode": "ik_only",
                        "selected_pickup_pregrasp_offset_m": float(
                            task.inserter_pickup_world_grasp.pregrasp_offset
                        ),
                        "pickup_pregrasp_offset_attempts": pickup_offset_attempts_by_screen_rank.get(
                            selected_candidate.screen_rank,
                            [],
                        ),
                    }
                )
                task_payload["generated_by"] = "scripts/plan_simple_dual_robot_sim.py"
                task_payload["moveit"] = {
                    "namespace": str(args.moveit_namespace),
                    "frame_id": "base_link",
                    "object_collision_geometry_in_scene": True,
                    "work_surface_collision_geometry_in_scene": True,
                    "work_surface": work_surface,
                    "arm_arm_collision_checking": True,
                    "start_joint_positions": list(MOVEIT_START_JOINT_POSITIONS),
                    "ik_only": True,
                    "pregrasp_aabb_collision_geometry": {
                        "representation": "phase_aware_world_aabb_minus_intended_contact_sweeps",
                        "obstacles": pregrasp_aabb_obstacles,
                        "active_by_target": pregrasp_aabb_schedule,
                        "removed_after_each_target": True,
                    },
                    "attached_collision_geometry": {
                        "representation": "pickup_world_aabb_in_grasp_tcp_frame",
                        "objects": attached_collision_objects,
                        "attach_after_target": {
                            str(value["attach_after_target"]): key
                            for key, value in attached_collision_objects.items()
                        },
                    },
                    "ik_preflight": ik_preflight,
                    "pickup_pregrasp_offset_selection": {
                        "selected_m": float(
                            task.inserter_pickup_world_grasp.pregrasp_offset
                        ),
                        "attempts": pickup_offset_attempts_by_screen_rank.get(
                            selected_candidate.screen_rank,
                            [],
                        ),
                    },
                    "joint_space_ranking": joint_space_ranking,
                    "attempts": attempt_records,
                }
                task_payload["trajectories"] = {}
                task_payload["holder_only"] = bool(args.holder_only)
                output = (
                    selection.artifact_dir / f"simple_dual_robot_sim_plan_{selection.step_id}.json"
                    if args.output is None
                    else args.output.expanduser().resolve()
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(task_payload, indent=2), encoding="utf-8")
                print(
                    f"[DUAL-SIM-PLAN] IK-only selected pair {task.pair_id}; wrote {output}",
                    flush=True,
                )
                update_debug(
                    task=task,
                    attempt_index=attempt_index,
                    phase="complete",
                    status="complete",
                    message=f"IK-only selected {task.execution_candidate_id}; diagnostics written to {output}",
                )
                time.sleep(0.10)
                return 0
            gripper_scene_state = _task_approach_gripper_state(task)
            scene_ok, scene_message = commanders["holder"].apply_planning_scene_robot_state(
                gripper_scene_state
            )
            if not scene_ok:
                raise RuntimeError(
                    f"Could not set candidate-specific MoveIt approach gripper widths: {scene_message}"
                )
            print(f"[DUAL-SIM-PLAN] approach grippers: {scene_message}", flush=True)
            post_grasp_state_updates = _task_post_grasp_state_updates(task)
            trajectories: dict[str, object] = {}
            steps: list[dict[str, object]] = []
            failure = ""
            incoming_collision_attached = False
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
                    candidate_joint_targets = preferred_joint_targets_by_candidate.get(
                        task.execution_candidate_id,
                        {},
                    )
                    validated_joint_sequence = _validated_joint_target_sequence(
                        candidate_joint_targets,
                        target_name=target_name,
                    ) if not bool(args.skip_ik_preflight) else ()
                    trajectory_payload, message = _plan_and_execute(
                        commanders[role],
                        target=target,
                        label=f"{task.pair_id}_{target_name}",
                        expected_joint_names=joint_names,
                        preferred_joint_positions=(
                            candidate_joint_targets.get(target_name)
                            if bool(args.skip_ik_preflight)
                            else None
                        ),
                        preferred_joint_sequence=(validated_joint_sequence or None),
                        gripper_robot_state=gripper_scene_state,
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
                        "validated_joint_targets": [
                            name for name, _joints in validated_joint_sequence
                        ],
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
                closed_state_update = post_grasp_state_updates.get(target_name)
                if closed_state_update:
                    gripper_scene_state.update(closed_state_update)
                    state_ok, state_message = commanders["holder"].apply_planning_scene_robot_state(
                        closed_state_update
                    )
                    steps.append(
                        {
                            "role": role,
                            "target": f"{target_name}_gripper_state",
                            "ok": state_ok,
                            "message": state_message,
                        }
                    )
                    if not state_ok:
                        failure = f"{target_name}: could not apply closed gripper state: {state_message}"
                        break
                    print(
                        f"[DUAL-SIM-PLAN] {target_name} closed gripper: {state_message}",
                        flush=True,
                    )
                    if target_name == "inserter_pickup_grasp" and attached_collision_objects:
                        attach_ok, attach_message = _apply_attached_collision_objects(
                            commanders["holder"],
                            attached_collision_objects,
                        )
                        steps.append(
                            {
                                "role": "shared",
                                "target": f"{target_name}_attach_incoming_collision",
                                "ok": attach_ok,
                                "message": attach_message,
                            }
                        )
                        if not attach_ok:
                            failure = f"{target_name}: {attach_message}"
                            break
                        incoming_collision_attached = True

            if incoming_collision_attached:
                detach_ok, detach_message = _remove_attached_collision_objects(
                    commanders["holder"],
                    attached_collision_objects,
                )
                steps.append(
                    {
                        "role": "shared",
                        "target": "cleanup_attached_incoming_collision",
                        "ok": detach_ok,
                        "message": detach_message,
                    }
                )
                if not detach_ok:
                    cleanup_failure = f"attached collision cleanup: {detach_message}"
                    failure = f"{failure}; {cleanup_failure}" if failure else cleanup_failure
                    # Continuing would make every subsequent collision result
                    # and cache entry depend on an unknown planning scene.
                    fatal_failure = cleanup_failure

            attempt_records.append(
                {
                    "attempt_index": attempt_index,
                    "candidate_rank": selected_candidate.candidate_rank,
                    "screen_rank": selected_candidate.screen_rank,
                    "path_attempt_index": path_attempt_index,
                    "phase": "path_planning_execution",
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
                    "selected_pickup_pregrasp_offset_m": float(
                        task.inserter_pickup_world_grasp.pregrasp_offset
                    ),
                    "pickup_pregrasp_offset_attempts": pickup_offset_attempts_by_screen_rank.get(
                        selected_candidate.screen_rank,
                        [],
                    ),
                    "steps": steps,
                }
            )
            if fatal_failure:
                update_debug(
                    task=task,
                    attempt_index=attempt_index,
                    phase="cleanup_attached_incoming_collision",
                    status="fatal",
                    message=fatal_failure,
                )
                break
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
                update_exact_ik_selection("screening_after_path_failure")
                update_debug(
                    task=task,
                    attempt_index=attempt_index,
                    phase="reset",
                    status="succeeded",
                    message=f"Mock-state recovery complete: {recovery_messages}",
                )
                continue

            update_exact_ik_selection("candidate_succeeded")
            task_payload["generated_by"] = "scripts/plan_simple_dual_robot_sim.py"
            task_payload["moveit"] = {
                "namespace": str(args.moveit_namespace),
                "frame_id": "base_link",
                "object_collision_geometry_in_scene": True,
                "work_surface_collision_geometry_in_scene": True,
                "work_surface": work_surface,
                "pregrasp_aabb_collision_geometry": {
                    "representation": ("phase_aware_world_aabb_minus_intended_contact_sweeps"),
                    "obstacles": pregrasp_aabb_obstacles,
                    "active_by_target": pregrasp_aabb_schedule,
                    "removed_after_each_target": True,
                },
                "attached_collision_geometry": {
                    "representation": "pickup_world_aabb_in_grasp_tcp_frame",
                    "objects": attached_collision_objects,
                    "attach_after_target": {
                        str(value["attach_after_target"]): key
                        for key, value in attached_collision_objects.items()
                    },
                },
                "arm_arm_collision_checking": True,
                "start_joint_positions": list(MOVEIT_START_JOINT_POSITIONS),
                "ik_preflight": ik_preflight,
                "pickup_pregrasp_offset_selection": {
                    "selected_m": float(
                        task.inserter_pickup_world_grasp.pregrasp_offset
                    ),
                    "attempts": pickup_offset_attempts_by_screen_rank.get(
                        selected_candidate.screen_rank,
                        [],
                    ),
                },
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
        if fatal_failure:
            exact_ik_stop_reason = "fatal_candidate_cleanup_or_recovery_failure"
        elif path_attempt_count >= path_candidate_limit:
            exact_ik_stop_reason = "path_candidate_limit_reached"
        elif ik_screened_candidate_count >= len(tasks):
            exact_ik_stop_reason = "source_pool_exhausted"
        elif ik_screened_candidate_count >= ik_screen_bound:
            exact_ik_stop_reason = "ik_screen_candidate_limit_reached"
        else:
            exact_ik_stop_reason = "screening_stopped"
        update_exact_ik_selection(exact_ik_stop_reason)
        print(
            "[DUAL-SIM-PLAN] Exact-IK/path selection exhausted: "
            f"screened={ik_screened_candidate_count}/{len(tasks)} "
            f"path_attempts={path_attempt_count}/{path_candidate_limit} "
            f"stop={exact_ik_stop_reason}",
            flush=True,
        )
        update_debug(
            task=last_task,
            attempt_index=len(attempt_records),
            phase="complete",
            status="fatal" if fatal_failure else "failed",
            message=(
                fatal_failure
                or (
                    "No complete plan after screening "
                    f"{exact_ik_selection['candidates_screened']} ranked candidates and attempting "
                    f"{path_attempt_count} exact-IK-feasible candidate(s)."
                )
            ),
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
        "candidate_filter_diagnostics": debug_candidate_counts,
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
    raise RuntimeError(
        "MoveIt could not complete any exact-IK-feasible candidate after "
        f"screening {exact_ik_selection['candidates_screened']} ranked pair(s) and admitting "
        f"{path_attempt_count} to bounded path planning; diagnostics written to {output}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
