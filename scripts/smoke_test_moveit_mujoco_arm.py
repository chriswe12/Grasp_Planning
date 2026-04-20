#!/usr/bin/env python3
"""Headless smoke test for MoveIt2 joint planning with MuJoCo FR3 execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.moveit import MoveItHeadlessFr3Server, MoveItJointPlanner, MoveItPlannerConfig
from grasp_planning.mujoco import (
    MujocoExecutionConfig,
    MujocoPickupRuntime,
    MujocoTrajectoryPoint,
    load_robot_config,
)


def _parse_float_tuple(raw: str, *, expected_len: int) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated floats, got '{raw}'.")
    return values


def _to_mujoco_points(plan) -> list[MujocoTrajectoryPoint]:
    return [
        MujocoTrajectoryPoint(positions=tuple(point), time_from_start_s=time_s)
        for point, time_s in zip(plan.points, plan.time_from_start_s)
    ]


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("configs/mujoco_fr3_with_hand.json"),
        help="MuJoCo robot binding JSON.",
    )
    parser.add_argument(
        "--object-mesh",
        type=Path,
        default=Path("assets/stl/Pencil_Organizer_Wide_Remix.stl"),
        help="Mesh to include in the scene while testing the arm path.",
    )
    parser.add_argument(
        "--object-position",
        type=str,
        default="-0.90,0.60,0.20",
        help="Dummy object world position x,y,z. Keep it away from the robot.",
    )
    parser.add_argument(
        "--object-orientation-xyzw",
        type=str,
        default="0.0,0.0,0.0,1.0",
        help="Dummy object world orientation quaternion x,y,z,w.",
    )
    parser.add_argument(
        "--goal-joints",
        type=str,
        default="0.10,-0.10,0.00,-1.70,0.00,1.60,-0.70",
        help="Target arm joint positions q1..q7.",
    )
    parser.add_argument("--return-home", action="store_true", help="Plan and execute a return trajectory to home.")
    parser.add_argument("--moveit-group", type=str, default="fr3_arm")
    parser.add_argument("--moveit-pipeline-id", type=str, default="move_group")
    parser.add_argument("--moveit-planner-id", type=str, default="")
    parser.add_argument("--moveit-planning-time", type=float, default=5.0)
    parser.add_argument("--moveit-velocity-scale", type=float, default=0.2)
    parser.add_argument("--moveit-acceleration-scale", type=float, default=0.2)
    parser.add_argument("--goal-joint-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--skip-start-move-group", action="store_true")
    parser.add_argument("--move-group-log", type=Path, default=Path("artifacts/move_group_smoke_arm.log"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/mujoco_moveit_smoke.json"))
    parser.add_argument("--keep-generated-scene", action="store_true")
    args = parser.parse_args()

    robot_cfg = load_robot_config(args.robot_config)
    goal_joints = _parse_float_tuple(args.goal_joints, expected_len=len(robot_cfg.arm_joint_names))
    object_pose = ObjectWorldPose(
        position_world=_parse_float_tuple(args.object_position, expected_len=3),
        orientation_xyzw_world=_parse_float_tuple(args.object_orientation_xyzw, expected_len=4),
    )
    execution_cfg = MujocoExecutionConfig()
    planner_cfg = MoveItPlannerConfig(
        group_name=args.moveit_group,
        pipeline_id=args.moveit_pipeline_id,
        planner_id=args.moveit_planner_id,
        allowed_planning_time=args.moveit_planning_time,
        max_velocity_scaling_factor=args.moveit_velocity_scale,
        max_acceleration_scaling_factor=args.moveit_acceleration_scale,
        goal_tolerance=args.goal_joint_tolerance,
        arm_joint_names=tuple(robot_cfg.arm_joint_names),
    )

    server_cm = None
    if not args.skip_start_move_group:
        server_cm = MoveItHeadlessFr3Server(planner_cfg, log_path=args.move_group_log)

    with server_cm if server_cm is not None else _NullContext():
        planner = MoveItJointPlanner()
        try:
            if not planner.wait_until_ready(timeout_sec=30.0):
                raise RuntimeError("MoveIt `/plan_kinematic_path` service did not become ready.")

            runtime = MujocoPickupRuntime(
                robot_cfg=robot_cfg,
                execution_cfg=execution_cfg,
                object_mesh_path=args.object_mesh,
                object_pose_world=object_pose,
                keep_generated_scene=args.keep_generated_scene,
            )
            try:
                runtime.settle_home()
                home_q = tuple(float(v) for v in runtime.get_arm_qpos())
                ee_start_pos, ee_start_quat = runtime.site_pose()

                goal_plan = planner.plan_joint_path(
                    start_positions=home_q,
                    goal_positions=goal_joints,
                    cfg=planner_cfg,
                )
                goal_ok = runtime.execute_arm_trajectory(
                    joint_names=goal_plan.joint_names,
                    points=_to_mujoco_points(goal_plan),
                    gripper_ctrl=robot_cfg.open_gripper_ctrl,
                )
                q_after_goal = runtime.get_arm_qpos()
                ee_goal_pos, ee_goal_quat = runtime.site_pose()
                goal_max_joint_error = float(np.max(np.abs(q_after_goal - np.asarray(goal_joints, dtype=float))))

                return_ok = None
                return_plan_points = 0
                return_planning_time_s = None
                q_after_return = None
                ee_return_pos = None
                ee_return_quat = None
                return_home_max_joint_error = None
                if args.return_home:
                    return_plan = planner.plan_joint_path(
                        start_positions=tuple(float(v) for v in q_after_goal),
                        goal_positions=home_q,
                        cfg=planner_cfg,
                    )
                    return_plan_points = len(return_plan.points)
                    return_planning_time_s = float(return_plan.planning_time_s)
                    return_ok = runtime.execute_arm_trajectory(
                        joint_names=return_plan.joint_names,
                        points=_to_mujoco_points(return_plan),
                        gripper_ctrl=robot_cfg.open_gripper_ctrl,
                    )
                    q_after_return = runtime.get_arm_qpos()
                    ee_return_pos, ee_return_quat = runtime.site_pose()
                    return_home_max_joint_error = float(
                        np.max(np.abs(q_after_return - np.asarray(home_q, dtype=float)))
                    )

                artifact = {
                    "success": bool(goal_ok and (True if return_ok is None else return_ok)),
                    "move_group_log": None if args.skip_start_move_group else str(args.move_group_log),
                    "generated_scene_xml": runtime.generated_scene_xml_path if args.keep_generated_scene else None,
                    "robot_config": str(args.robot_config),
                    "object_mesh": str(args.object_mesh),
                    "start_home_q": list(home_q),
                    "goal_joints": list(goal_joints),
                    "planning": {
                        "goal_points": len(goal_plan.points),
                        "goal_planning_time_s": float(goal_plan.planning_time_s),
                        "return_points": return_plan_points,
                        "return_planning_time_s": return_planning_time_s,
                    },
                    "execution": {
                        "goal_ok": bool(goal_ok),
                        "goal_max_joint_error": goal_max_joint_error,
                        "q_after_goal": [float(v) for v in q_after_goal],
                        "ee_start_position_world": [float(v) for v in ee_start_pos],
                        "ee_start_orientation_xyzw": [float(v) for v in ee_start_quat],
                        "ee_goal_position_world": [float(v) for v in ee_goal_pos],
                        "ee_goal_orientation_xyzw": [float(v) for v in ee_goal_quat],
                        "return_ok": None if return_ok is None else bool(return_ok),
                        "return_home_max_joint_error": return_home_max_joint_error,
                        "q_after_return": None if q_after_return is None else [float(v) for v in q_after_return],
                        "ee_return_position_world": None
                        if ee_return_pos is None
                        else [float(v) for v in ee_return_pos],
                        "ee_return_orientation_xyzw": None
                        if ee_return_quat is None
                        else [float(v) for v in ee_return_quat],
                    },
                }
            finally:
                runtime.close()
        finally:
            planner.close()

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        f"[INFO]: MoveIt+MuJoCo smoke test success={artifact['success']} "
        f"goal_ok={artifact['execution']['goal_ok']} "
        f"goal_max_joint_error={artifact['execution']['goal_max_joint_error']:.4f}",
        flush=True,
    )
    print(f"[INFO]: Wrote artifact to {args.artifact}", flush=True)


if __name__ == "__main__":
    main()
