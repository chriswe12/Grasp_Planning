#!/usr/bin/env python3
"""Export one curriculum episode as a synchronized diagnostic video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.rl.dataset_visualizer import (
    load_episode_visualization,
    load_overview_frames,
    prepend_overview_frame,
    render_episode_frame,
)
from grasp_planning.video import OpenCvVideoWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_npz", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hold-final-seconds", type=float, default=1.0)
    parser.add_argument(
        "--overview-npz",
        type=Path,
        default=None,
        help="Optional synchronized overview sidecar from render_visual_servo_episode_overview.py.",
    )
    args = parser.parse_args()

    episode = load_episode_visualization(args.episode_npz)
    overview_frames = (
        load_overview_frames(args.overview_npz, expected_steps=episode.step_count)
        if args.overview_npz is not None
        else None
    )
    output = args.output or args.episode_npz.with_name(f"{args.episode_npz.stem}_visualization.mp4")

    def render(step_index: int):
        frame = render_episode_frame(episode, step_index)
        if overview_frames is not None:
            frame = prepend_overview_frame(frame, overview_frames[step_index])
        return frame

    first_frame = render(0)
    with OpenCvVideoWriter(
        output,
        fps=args.fps,
        width=first_frame.shape[1],
        height=first_frame.shape[0],
    ) as writer:
        for step_index in range(episode.step_count):
            writer.append_rgb(render(step_index))
        final_frame = render(episode.step_count - 1)
        for _ in range(max(0, int(round(args.hold_final_seconds * args.fps)))):
            writer.append_rgb(final_frame)
        frame_count = writer.frame_count
    print(f"Wrote {frame_count} frames at {args.fps:.3f} FPS to {output}.")


if __name__ == "__main__":
    main()
