#!/usr/bin/env python3
"""Record a multi-location, multi-grasp holder-only dual-KUKA Isaac stress video."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_SH = Path("/media/pdz/Elements1/IsaacLab/isaaclab.sh")

CASES = (
    ("center_low_side_a", 0.55, 0.00, ("h0510", "h0511", "h0450")),
    ("center_low_side_b", 0.55, 0.00, ("h0512", "h0451", "h0449")),
    ("holder_side", 0.55, -0.18, ("h0450", "h0451", "h0510")),
    ("far_holder_side", 0.62, -0.24, ("h0450", "h0451")),
    ("holder_front_diagonal", 0.70, -0.18, ("h0450", "h0451", "h0510")),
    ("far_forward", 0.72, 0.00, ("h0510", "h0450", "h0451")),
    ("inserter_side", 0.55, 0.18, ("h0512", "h0511", "h0451")),
    ("inserter_front_diagonal", 0.68, 0.18, ("h0512", "h0511", "h0451")),
    ("near_rear", 0.42, -0.12, ("h0450", "h0510", "h0451")),
    ("wide_cross_side", 0.62, 0.25, ("h0512", "h0511", "h0451")),
)


def _ros_shell(command: str, *, check: bool = False) -> subprocess.CompletedProcess:
    setup = (
        "source /opt/ros/humble/setup.bash; "
        "source /home/pdz/lbr-stack/install/setup.bash; "
        f"source {REPO_ROOT}/ros2_ws/install/setup.bash; "
        "export ROS_LOG_DIR=/tmp/ros-log ROS_DOMAIN_ID=0 "
        "ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp "
        "FASTDDS_BUILTIN_TRANSPORTS=UDPv4; "
        "unset ROS_DISCOVERY_SERVER ROS_STATIC_PEERS ROS_AUTOMATIC_DISCOVERY_RANGE; "
    )
    return subprocess.run(
        ["bash", "-lc", setup + command],
        cwd=REPO_ROOT,
        check=check,
    )


def _wait_for_moveit(timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = _ros_shell("ros2 service type /lbr_dual_arm/compute_ik >/dev/null 2>&1")
        if result.returncode == 0:
            return
        time.sleep(1.0)
    raise RuntimeError("Dual mock MoveIt did not become ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/dual_holder_stress/video"),
    )
    parser.add_argument("--case-limit", type=int, default=len(CASES))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    moveit_log = (output_root / "moveit.log").open("w", encoding="utf-8")
    environment = dict(os.environ)
    environment["ROS_LOG_DIR"] = "/tmp/ros-log"
    environment["ROS_DOMAIN_ID"] = "0"
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
    records: list[dict[str, object]] = []
    try:
        _wait_for_moveit()
        for name, x, y, preferred_ids in CASES[: max(1, args.case_limit)]:
            case_dir = output_root / name
            case_dir.mkdir(parents=True, exist_ok=True)
            plan = case_dir / "plan.json"
            attempt = case_dir / "attempt.json"
            video = case_dir / "raw.mp4"
            if args.skip_existing and attempt.is_file() and video.is_file() and video.stat().st_size > 0:
                payload = json.loads(attempt.read_text(encoding="utf-8"))
                selected = json.loads(plan.read_text(encoding="utf-8"))
                records.append(
                    {
                        "name": name,
                        "x": x,
                        "y": y,
                        "holder_grasp_id": selected["grasps"]["holder"]["grasp_id"],
                        "success": bool(payload["result"]["success"]),
                        "recorded": True,
                        "plan": str(plan),
                        "attempt": str(attempt),
                        "video": str(video),
                    }
                )
                continue
            selected_id = ""
            execution_success = False
            errors = []
            for holder_id in (*preferred_ids, ""):
                command = [
                    str(REPO_ROOT / "run_simple_dual_robot.sh"),
                    "--mode",
                    "sim",
                    "--reuse-moveit",
                    "--headless",
                    "--holder-only",
                    "--assembly",
                    "plumbers_block",
                    "--incoming-part-id",
                    "0",
                    "--assembly-x",
                    str(x),
                    "--assembly-y",
                    str(y),
                    "--pickup-x",
                    "0.55",
                    "--pickup-y",
                    "0.30",
                    "--max-pair-attempts",
                    "48",
                    "--plan-output",
                    str(plan),
                    "--attempt-output",
                    str(attempt),
                    "--record-video",
                    str(video),
                ]
                if holder_id:
                    command.extend(("--holder-grasp-id", holder_id))
                print(
                    f"[HOLDER-STRESS] {name}: trying holder={holder_id or 'ranked fallback'} at ({x:.2f}, {y:.2f})",
                    flush=True,
                )
                result = subprocess.run(command, cwd=REPO_ROOT)
                if plan.is_file() and video.is_file() and video.stat().st_size > 0:
                    selected = json.loads(plan.read_text(encoding="utf-8"))
                    selected_id = str(selected["grasps"]["holder"]["grasp_id"])
                    if attempt.is_file():
                        attempt_payload = json.loads(attempt.read_text(encoding="utf-8"))
                        execution_success = bool(attempt_payload["result"]["success"])
                    break
                errors.append({"holder_grasp_id": holder_id, "returncode": result.returncode})
            recorded = bool(selected_id)
            records.append(
                {
                    "name": name,
                    "x": x,
                    "y": y,
                    "holder_grasp_id": selected_id,
                    "success": execution_success,
                    "recorded": recorded,
                    "errors": errors,
                    "plan": str(plan),
                    "attempt": str(attempt),
                    "video": str(video),
                }
            )
        manifest = output_root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "kind": "dual_holder_stress_video",
                    "simulation_only": True,
                    "cases": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        recorded = [record for record in records if record.get("recorded")]
        if not recorded:
            raise RuntimeError(f"No holder stress case was recorded; see {manifest}")
        subprocess.run(
            [
                str(ISAACLAB_SH),
                "-p",
                str(REPO_ROOT / "scripts/compose_holder_stress_videos.py"),
                "--manifest",
                str(manifest),
                "--output",
                str(output_root / "all_holder_stress.mp4"),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        print(
            f"[HOLDER-STRESS] Recorded {len(recorded)}/{len(records)} cases. "
            f"Video: {output_root / 'all_holder_stress.mp4'}",
            flush=True,
        )
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
