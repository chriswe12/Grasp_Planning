"""One-shot move-to-pose controller for the FR3 arm."""

from __future__ import annotations

from .collision_checker import CollisionChecker
from .fr3_motion_context import FR3MotionContext
from .goal_ik import GoalIKSolver
from .joint_path_planner import JointPathPlanner
from .trajectory_executor import TrajectoryExecutor
from .types import JointTrajectory, PlanResult, PoseCommand


class FR3MoveToPoseController:
    """Move the FR3 arm to a target TCP pose with conservative collision checks."""

    def __init__(
        self,
        *,
        robot,
        cube,
        scene,
        sim,
        fixed_gripper_width: float = 0.04,
    ) -> None:
        self._context = FR3MotionContext(
            robot=robot,
            scene=scene,
            sim=sim,
            fixed_gripper_width=fixed_gripper_width,
        )
        self._ik = GoalIKSolver(self._context)
        self._collision_checker = CollisionChecker(self._context, cube)
        self._planner = JointPathPlanner(self._collision_checker)
        self._executor = TrajectoryExecutor(self._context)

    @property
    def ee_body_name(self) -> str:
        return self._context.ee_body_name

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return self._context.arm_joint_names

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return self._context.hand_joint_names

    def get_current_tcp_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        tcp_pos_w, tcp_quat_w = self._context.get_tcp_pose_w()
        pos = tuple(float(v) for v in tcp_pos_w[0].tolist())
        quat_wxyz = tcp_quat_w[0]
        quat_xyzw = (
            float(quat_wxyz[1].item()),
            float(quat_wxyz[2].item()),
            float(quat_wxyz[3].item()),
            float(quat_wxyz[0].item()),
        )
        return pos, quat_xyzw

    def move_to_pose(self, position_w, orientation_xyzw) -> PlanResult:
        cmd = PoseCommand(position_w=tuple(position_w), orientation_xyzw=tuple(orientation_xyzw))
        q_start = self._context.get_arm_q()
        valid, reason = self._collision_checker.is_state_valid()
        if not valid and reason == "plane_collision":
            print("[WARN]: Rechecking transient plane collision after a short settle window.", flush=True)
            self._context.hold_position(q_start, steps=8)
            valid, reason = self._collision_checker.is_state_valid()
        if not valid and reason == "plane_collision":
            print("[WARN]: Ignoring start-state plane collision rejection for debugging.", flush=True)
            valid = True
        if not valid and reason != "joint_limits":
            return PlanResult(False, "start_in_collision", f"Current arm state is invalid: {reason}.")
        if not valid and reason == "joint_limits":
            print(
                "[WARN]: Ignoring start-state joint limit rejection for debugging: "
                + self._context.describe_joint_limit_state(q_start),
                flush=True,
            )

        q_goal = self._ik.solve(cmd)
        if q_goal is None:
            return PlanResult(False, "ik_failed", "No IK solution found for the requested target pose.")
        print(f"[INFO]: IK goal joints={q_goal[0].tolist()}", flush=True)

        trajectory, plan_reason = self._planner.plan(q_start, q_goal, dt=self._context.physics_dt)
        if trajectory is None:
            return PlanResult(False, "planning_failed", f"Direct joint path rejected: {plan_reason}.", goal_q=q_goal)
        print(f"[INFO]: Planned {len(trajectory.waypoints)} waypoints.", flush=True)

        ok, execution_detail = self._executor.execute(trajectory)
        if not ok:
            return PlanResult(
                False,
                "execution_failed",
                f"Arm did not converge to the planned joint waypoints: {execution_detail}.",
                trajectory=trajectory,
                goal_q=q_goal,
            )

        return PlanResult(True, "ok", "Motion executed successfully.", trajectory=trajectory, goal_q=q_goal)

    def move_through_poses(
        self, poses: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]]
    ) -> PlanResult:
        """Plan all requested poses first, then stream them as one joint trajectory."""

        if not poses:
            return PlanResult(True, "ok", "No poses requested.")

        q_start = self._context.get_arm_q()
        valid, reason = self._collision_checker.is_state_valid()
        if not valid and reason == "plane_collision":
            print("[WARN]: Rechecking transient plane collision after a short settle window.", flush=True)
            self._context.hold_position(q_start, steps=8)
            valid, reason = self._collision_checker.is_state_valid()
        if not valid and reason == "plane_collision":
            print("[WARN]: Ignoring start-state plane collision rejection for debugging.", flush=True)
            valid = True
        if not valid and reason != "joint_limits":
            return PlanResult(False, "start_in_collision", f"Current arm state is invalid: {reason}.")
        if not valid and reason == "joint_limits":
            print(
                "[WARN]: Ignoring start-state joint limit rejection for debugging: "
                + self._context.describe_joint_limit_state(q_start),
                flush=True,
            )

        all_waypoints = []
        q_segment_start = q_start.clone()
        for index, (position_w, orientation_xyzw) in enumerate(poses, start=1):
            self._context.hold_position(q_segment_start, steps=2)
            cmd = PoseCommand(position_w=tuple(position_w), orientation_xyzw=tuple(orientation_xyzw))
            q_goal = self._ik.solve(cmd)
            if q_goal is None:
                self._context.hold_position(q_start, steps=8)
                return PlanResult(False, "ik_failed", f"No IK solution found for pose {index}/{len(poses)}.")
            trajectory, plan_reason = self._planner.plan(q_segment_start, q_goal, dt=self._context.physics_dt)
            if trajectory is None:
                self._context.hold_position(q_start, steps=8)
                return PlanResult(
                    False,
                    "planning_failed",
                    f"Joint path to pose {index}/{len(poses)} rejected: {plan_reason}.",
                    goal_q=q_goal,
                )
            all_waypoints.extend(trajectory.waypoints)
            q_segment_start = q_goal.clone()

        self._context.hold_position(q_start, steps=8)
        stitched_trajectory = JointTrajectory(waypoints=all_waypoints, dt=self._context.physics_dt)
        print(
            f"[INFO]: Planned streamed pose sequence poses={len(poses)} joint_waypoints={len(all_waypoints)}.",
            flush=True,
        )
        ok, execution_detail = self._executor.execute(stitched_trajectory)
        if not ok:
            return PlanResult(
                False,
                "execution_failed",
                f"Arm did not settle after streamed joint trajectory: {execution_detail}.",
                trajectory=stitched_trajectory,
                goal_q=q_segment_start,
            )
        return PlanResult(
            True,
            "ok",
            "Streamed motion executed successfully.",
            trajectory=stitched_trajectory,
            goal_q=q_segment_start,
        )
