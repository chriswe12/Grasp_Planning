#!/usr/bin/env python3
"""Run one D405 PPO visual-servo trial, dry-run by checked-in default."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.ros2.d405_visual_servo import run_d405_policy_visual_servo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/visual_servo_real_d405.yaml",
        help="Deployment YAML. The checked-in file uses the dry-run command sink.",
    )
    parser.add_argument(
        "--expected-grasp-id",
        default="",
        help="Optional stage-2 grasp ID that the selected goal target must match.",
    )
    parser.add_argument(
        "--expected-part-id",
        default="",
        help="Optional stage-2 part ID that the selected goal target must match.",
    )
    parser.add_argument(
        "--confirm-real-motion",
        action="store_true",
        help="Required in addition to real_motion_approved=true and command_sink=moveit_servo.",
    )
    args = parser.parse_args()
    result = run_d405_policy_visual_servo(
        config_path=args.config,
        expected_grasp_id=str(args.expected_grasp_id),
        expected_part_id=str(args.expected_part_id),
        allow_real_motion=bool(args.confirm_real_motion),
    )
    print(
        f"[D405-POLICY] completed={result.completed} state={result.state} "
        f"motion_applied={result.motion_applied} target={result.target_id} message={result.message}",
        flush=True,
    )
    print(f"[D405-POLICY] artifacts={result.run_directory}", flush=True)
    if not result.completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
