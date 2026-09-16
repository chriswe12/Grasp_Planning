#!/usr/bin/env python3
"""Write the mandatory compatibility sidecar for one reviewed PPO checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.rl.d405_policy_runtime import write_checkpoint_metadata_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--completion-probability-threshold", type=float, default=0.95)
    parser.add_argument("--completion-required-consecutive-steps", type=int, default=4)
    parser.add_argument(
        "--policy-context",
        choices=("action", "action_twist", "action_twist_rotation"),
        default="action",
        help="Actor context contract used when training this checkpoint.",
    )
    parser.add_argument("--policy-rate-hz", type=float, default=15.0)
    parser.add_argument("--action-delta-limit", type=float, default=0.25)
    args = parser.parse_args()
    output = write_checkpoint_metadata_template(
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        source_commit=args.source_commit,
        completion_probability_threshold=args.completion_probability_threshold,
        completion_required_consecutive_steps=args.completion_required_consecutive_steps,
        policy_context_mode=args.policy_context,
        policy_rate_hz=args.policy_rate_hz,
        action_delta_limit=args.action_delta_limit,
    )
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
