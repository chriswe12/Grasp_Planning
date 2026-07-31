#!/usr/bin/env python3
"""Plan a simple holder plus pickup-to-preinsertion sequence with dual MoveIt."""

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
from grasp_planning.start_poses import (  # noqa: E402
    KUKA_MOVEIT_ARM_START_JOINT_VALUES,
)

MOVEIT_START_JOINT_POSITIONS = KUKA_MOVEIT_ARM_START_JOINT_VALUES
ARM_SPECS = {
    "holder": {
        "planning_group": "arm_one",
        "pose_link": "lbr_one_gripper_tcp",
        "joint_names": tuple(f"lbr_one_A{index}" for index in range(1, 8)),
    },
    "inserter": {
        "planning_group": "arm_two",
        "pose_link": "lbr_two_gripper_tcp",
        "joint_names": tuple(f"lbr_two_A{index}" for index in range(1, 8)),
    },
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
    parser.add_argument("--max-pair-attempts", type=int, default=48)
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
) -> tuple[dict[str, object] | None, str]:
    trajectory, message = commander.plan_to_pose(target, label=label)
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
) -> tuple[bool, str]:
    """Screen one ranked pair and stop immediately on its first failed role."""

    task_payload = task.to_payload()
    targets = dict(task_payload["targets"])
    grasp_ids = {
        "holder": task.holder_candidate.grasp_id,
        "inserter": task.inserter_candidate.grasp_id,
    }
    pair_role_records: dict[str, object] = {}
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
        )
        cache_key = (grasp_id, target_signature)
        cache_hit = cache_key in feasible_cache[role]
        if cache_hit:
            grasp_feasible = feasible_cache[role][cache_key]
        else:
            target_records = []
            grasp_feasible = True
            for target_name in target_names:
                joints, message = commanders[role].compute_ik(_pose_target(dict(targets[target_name])))
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


def main() -> int:
    args = _parse_args()
    if rclpy is None:
        raise RuntimeError(
            "ROS2 MoveIt dependencies are unavailable. Source ROS2 and both workspaces before running this planner."
        )
    if args.max_pair_attempts < 1:
        raise ValueError("--max-pair-attempts must be at least 1.")
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
            retained_only=bool(args.retained_only) and not (bool(args.all_compatible) or bool(args.pair_id)),
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
    rclpy.init()
    commanders: dict[str, MoveItPoseCommander] = {}
    attempt_records: list[dict[str, object]] = []
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

        needs_reset = True
        for attempt_index, task in enumerate(tasks, start=1):
            print(
                f"[DUAL-SIM-PLAN] Attempt {attempt_index}/{len(tasks)} "
                f"pair={task.pair_id} pair_score={task.pair_score:.4f} "
                f"selection_score={task.selection_score:.4f} "
                f"layout_proxy={task.layout_proxy_score:.4f}",
                flush=True,
            )
            if needs_reset:
                reset_ok = True
                reset_messages = {}
                for role in active_roles:
                    reset_positions = (
                        tuple(float(value) for value in args.holder_start_joint_positions)
                        if role == "holder" and args.holder_start_joint_positions is not None
                        else MOVEIT_START_JOINT_POSITIONS
                    )
                    ok, message = _reset_arm(
                        commanders[role],
                        role=role,
                        joint_positions=reset_positions,
                    )
                    reset_messages[role] = message
                    reset_ok = reset_ok and ok
                if not reset_ok:
                    raise RuntimeError(f"Could not reset dual MoveIt mock state: {reset_messages}")
                needs_reset = False

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
                )
                checked = int(ik_preflight["pair_tasks_checked"])
                holder_checked = int(ik_preflight["holder_grasps_checked"])
                inserter_checked = int(ik_preflight["inserter_grasps_checked"])
                if not pair_ik_ok:
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
            trajectories: dict[str, object] = {}
            steps: list[dict[str, object]] = []
            failure = ""
            for role, target_name in target_sequence:
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
                    break
                trajectories[target_name] = trajectory_payload

            attempt_records.append(
                {
                    "attempt_index": attempt_index,
                    "pair_id": task.pair_id,
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
                needs_reset = True
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
            return 0
    finally:
        for commander in commanders.values():
            commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

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
        "ik_preflight": ik_preflight,
        "attempts": attempt_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(failure_payload, indent=2),
        encoding="utf-8",
    )
    raise RuntimeError(f"MoveIt could not plan any of {len(tasks)} ranked pairs; diagnostics written to {output}.")


if __name__ == "__main__":
    raise SystemExit(main())
