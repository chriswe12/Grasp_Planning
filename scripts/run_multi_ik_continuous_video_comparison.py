#!/usr/bin/env python3
"""Plan and record continuous single-IK versus multi-IK motion sequences."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.ros2.moveit_pose_commander import MoveItPoseCommander, MoveItPoseCommanderConfig, PoseTarget, rclpy
from grasp_planning.ros2.multi_ik_planner import MultiIkPlanningConfig, plan_pose_sequence_multi_ik

OUTPUT_ROOT = REPO_ROOT / "artifacts/multi_ik_video_comparison/continuous"
ISAAC_PYTHON = Path("/media/pdz/Elements1/IsaacLab/_isaac_sim/python.sh")
JOINT_NAMES = tuple(f"lbr_A{index}" for index in range(1, 8))
LOCATIONS = (
    ("center", 0.50, 0.00),
    ("left", 0.48, 0.12),
    ("right", 0.48, -0.12),
    ("far_left", 0.60, 0.30),
    ("far_right", 0.60, -0.30),
    ("wide_left", 0.42, 0.34),
    ("wide_right", 0.42, -0.34),
    ("far_center", 0.68, 0.00),
)


def _waypoints(trajectory) -> tuple[tuple[float, ...], ...]:
    names = tuple(str(value) for value in trajectory.joint_trajectory.joint_names)
    indices = {name: index for index, name in enumerate(names)}
    return tuple(
        tuple(float(point.positions[indices[name]]) for name in JOINT_NAMES)
        for point in trajectory.joint_trajectory.points
    )


def _targets(reference_plan: dict[str, object]) -> tuple[dict[str, PoseTarget], tuple[str, ...]]:
    world_grasp = dict(reference_plan["selected_world_grasp"])
    reference = tuple(float(value) for value in world_grasp["pregrasp_position_w"])
    orientation = tuple(float(value) for value in world_grasp["orientation_xyzw"])
    targets = {}
    labels = []
    for name, x, y in LOCATIONS:
        label = f"move_{name}"
        labels.append(label)
        targets[label] = PoseTarget.from_quaternion(
            x=reference[0] + x - 0.50,
            y=reference[1] + y,
            z=reference[2],
            quaternion_xyzw=orientation,
            frame_id="lbr_link_0",
        )
    return targets, tuple(labels)


def _plan(strategy: str, reference_plan: dict[str, object]) -> Path:
    targets, labels = _targets(reference_plan)
    start = tuple(float(value) for value in reference_plan["start_joint_positions"])
    commander = MoveItPoseCommander(
        MoveItPoseCommanderConfig(
            planning_group="arm",
            pose_link="gripper_tcp",
            joint_names=JOINT_NAMES,
            moveit_namespace="/lbr",
            planning_time_s=5.0,
            num_planning_attempts=5,
        ),
        node_name=f"continuous_{strategy}_planner",
    )
    try:
        commander.wait_for_moveit(require_execute=False)
        if strategy == "multi_ik":
            result = plan_pose_sequence_multi_ik(
                commander,
                targets=targets,
                labels=labels,
                start_joint_positions=start,
                joint_names=JOINT_NAMES,
                config=MultiIkPlanningConfig(
                    candidate_count=8,
                    beam_width=3,
                    seed_perturbation_rad=0.7,
                    dedup_tolerance_rad=0.05,
                    joint_weights=(0.5, 2.0, 2.0, 2.5, 1.5, 1.5, 0.5),
                ),
                label_prefix="continuous",
            )
            trajectories = dict(result.trajectories)
        else:
            trajectories = {}
            current = start
            for label in labels:
                trajectory, message = commander.plan_to_pose(
                    targets[label],
                    label=f"continuous_{label}",
                    start_joint_positions=current,
                )
                if trajectory is None:
                    raise RuntimeError(f"Single-IK continuous planning failed at {label}: {message}")
                trajectories[label] = _waypoints(trajectory)
                current = trajectories[label][-1]
    finally:
        commander.destroy_node()
    path = OUTPUT_ROOT / strategy / "sequence_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "labels": list(labels),
                "locations": [{"name": name, "x": x, "y": y} for name, x, y in LOCATIONS],
                "joint_names": list(JOINT_NAMES),
                "start_joint_positions": list(start),
                "trajectories": {
                    label: [list(waypoint) for waypoint in waypoints] for label, waypoints in trajectories.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _record(strategy: str, plan_path: Path) -> Path:
    output = OUTPUT_ROOT / strategy
    video_path = output / "raw.mp4"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) if not env.get("PYTHONPATH") else f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}"
    subprocess.run(
        [
            str(ISAAC_PYTHON),
            str(REPO_ROOT / "scripts/run_fabrica_grasp_in_isaac.py"),
            "--input-json",
            str(REPO_ROOT / "artifacts/multi_ik_video_comparison/center/single_ik/stage2.json"),
            "--fr3-usd",
            "assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda",
            "--controller",
            "moveit",
            "--grasp-id",
            "g3058",
            "--moveit-motion-sequence-json",
            str(plan_path),
            "--moveit-joint-names",
            ",".join(JOINT_NAMES),
            "--moveit-start-joint-positions",
            "0.0,0.5,0.0,-1.3962634015954636,0.0,1.1,0.0",
            "--attempt-artifact",
            str(output / "attempt.json"),
            "--record-video",
            str(video_path),
            "--video-width",
            "640",
            "--video-height",
            "480",
            "--video-fps",
            "30",
            "--headless",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Isaac did not produce the expected continuous video: {video_path}")
    return video_path


def main() -> None:
    reference_plan = json.loads(
        (REPO_ROOT / "artifacts/multi_ik_video_comparison/center/single_ik/attempt_moveit_plan.json").read_text(
            encoding="utf-8"
        )
    )
    initialized_here = False
    if not rclpy.ok():
        rclpy.init()
        initialized_here = True
    try:
        single_plan = _plan("single_ik", reference_plan)
        multi_plan = _plan("multi_ik", reference_plan)
    finally:
        if initialized_here and rclpy.ok():
            rclpy.shutdown()
    single_video = _record("single_ik", single_plan)
    multi_video = _record("multi_ik", multi_plan)
    subprocess.run(
        [
            str(ISAAC_PYTHON),
            str(REPO_ROOT / "scripts/compose_multi_ik_comparison_videos.py"),
            "--single",
            str(single_video),
            "--multi",
            str(multi_video),
            "--case-label",
            "continuous 8-location sequence (no arm reset)",
            "--output",
            str(OUTPUT_ROOT / "continuous_side_by_side.mp4"),
        ],
        check=True,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
