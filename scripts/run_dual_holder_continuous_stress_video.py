#!/usr/bin/env python3
"""Plan and record holder grasps continuously across widely spaced base poses."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
from pathlib import Path

from run_dual_holder_stress_video import (
    CASES,
    ISAACLAB_SH,
    REPO_ROOT,
    _wait_for_moveit,
)


def _ros_command(arguments: list[str]) -> int:
    setup = (
        "source /opt/ros/humble/setup.bash; "
        "source /home/pdz/lbr-stack/install/setup.bash; "
        f"source {REPO_ROOT}/ros2_ws/install/setup.bash; "
        "export ROS_LOG_DIR=/tmp/ros-log ROS_DOMAIN_ID=0 "
        "ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp "
        "FASTDDS_BUILTIN_TRANSPORTS=UDPv4; "
        "unset ROS_DISCOVERY_SERVER ROS_STATIC_PEERS ROS_AUTOMATIC_DISCOVERY_RANGE; "
        f"exec {shlex.join(arguments)}"
    )
    return subprocess.run(["bash", "-lc", setup], cwd=REPO_ROOT).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/dual_holder_stress/continuous"),
    )
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prior_manifest_path = REPO_ROOT / "artifacts/dual_holder_stress/video/manifest.json"
    prior_ids = {}
    if prior_manifest_path.is_file():
        prior = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        prior_ids = {case["name"]: case["holder_grasp_id"] for case in prior["cases"] if case.get("holder_grasp_id")}

    moveit_log = (output_root / "moveit.log").open("w", encoding="utf-8")
    environment = dict(os.environ)
    environment.update({"ROS_LOG_DIR": "/tmp/ros-log", "ROS_DOMAIN_ID": "0"})
    moveit = subprocess.Popen(
        [
            str(REPO_ROOT / "start_dual_lbr_moveit.sh"),
            "--mode",
            "mock",
            "--ros-domain-id",
            "0",
        ],
        cwd=REPO_ROOT,
        env=environment,
        stdout=moveit_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    sequence_cases: list[dict[str, object]] = []
    previous_joints = [0.0, 0.5, 0.0, -1.3962634, 0.0, 1.1, 0.0]
    try:
        _wait_for_moveit()
        for index, (name, x, y, preferred) in enumerate(CASES, start=1):
            case_dir = output_root / f"{index:02d}_{name}"
            case_dir.mkdir(parents=True, exist_ok=True)
            plan_path = case_dir / "plan.json"
            candidate_ids = []
            for candidate in (prior_ids.get(name), *preferred, ""):
                if candidate is not None and candidate not in candidate_ids:
                    candidate_ids.append(candidate)
            planned = False
            for holder_id in candidate_ids:
                command = [
                    "python3",
                    str(REPO_ROOT / "scripts/plan_simple_dual_robot_sim.py"),
                    "--artifact-root",
                    "artifacts/dual_grasp_planning",
                    "--assembly",
                    "plumbers_block",
                    "--incoming-part-id",
                    "0",
                    "--holder-only",
                    "--assembly-x",
                    str(x),
                    "--assembly-y",
                    str(y),
                    "--assembly-z",
                    "-0.03",
                    "--pickup-x",
                    "0.55",
                    "--pickup-y",
                    "0.30",
                    "--floor-z",
                    "-0.03",
                    "--max-pair-attempts",
                    "48",
                    "--holder-start-joint-positions",
                    *(str(value) for value in previous_joints),
                    "--output",
                    str(plan_path),
                ]
                if holder_id:
                    command.extend(("--holder-grasp-id", holder_id))
                print(
                    f"[CONTINUOUS-HOLDER] {index}/{len(CASES)} {name}: "
                    f"holder={holder_id or 'ranked fallback'} start={previous_joints}",
                    flush=True,
                )
                if _ros_command(command) == 0:
                    planned = True
                    break
            if not planned:
                print(
                    f"[CONTINUOUS-HOLDER] Skipping unreachable case {name}.",
                    flush=True,
                )
                continue
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            selected_id = plan["grasps"]["holder"]["grasp_id"]
            previous_joints = [float(value) for value in plan["trajectories"]["holder_grasp"]["waypoints"][-1]]
            sequence_cases.append(
                {
                    "name": name,
                    "x": x,
                    "y": y,
                    "holder_grasp_id": selected_id,
                    "plan": str(plan_path),
                    "terminal_joint_positions": previous_joints,
                }
            )
        if len(sequence_cases) < 2:
            raise RuntimeError("Fewer than two continuous holder cases planned.")
        sequence_path = output_root / "sequence.json"
        sequence_path.write_text(
            json.dumps(
                {
                    "kind": "dual_holder_continuous_stress_sequence",
                    "simulation_only": True,
                    "cases": sequence_cases,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raw_video = output_root / "raw.mp4"
        attempt = output_root / "attempt.json"
        first_plan = str(sequence_cases[0]["plan"])
        subprocess.run(
            [
                str(ISAACLAB_SH),
                "-p",
                str(REPO_ROOT / "scripts/run_simple_dual_robot_sim_in_isaac.py"),
                "--plan-json",
                first_plan,
                "--holder-only",
                "--holder-sequence-json",
                str(sequence_path),
                "--attempt-artifact",
                str(attempt),
                "--record-video",
                str(raw_video),
                "--headless",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        output_video = output_root / "continuous_holder_stress.mp4"
        subprocess.run(
            [
                str(ISAACLAB_SH),
                "-p",
                str(REPO_ROOT / "scripts/compose_continuous_holder_stress_video.py"),
                "--raw-video",
                str(raw_video),
                "--sequence",
                str(sequence_path),
                "--attempt",
                str(attempt),
                "--output",
                str(output_video),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        print(f"[CONTINUOUS-HOLDER] Video: {output_video}", flush=True)
    finally:
        if moveit.poll() is None:
            os.killpg(moveit.pid, signal.SIGINT)
            try:
                moveit.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                os.killpg(moveit.pid, signal.SIGTERM)
                moveit.wait(timeout=10.0)
        moveit_log.close()


if __name__ == "__main__":
    main()
