#!/usr/bin/env python3
"""Print the latest RL-Games epoch and FPS from a TensorBoard event file."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _latest_scalar(accumulator: EventAccumulator, tag: str):
    values = accumulator.Scalars(tag)
    return max(values, key=lambda value: (value.step, value.wall_time)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_file", type=Path)
    args = parser.parse_args()

    accumulator = EventAccumulator(str(args.event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", ()))

    epoch_value = None
    for tag in ("rewards/iter", "episode_lengths/iter", "Episode/success_rate"):
        if tag in scalar_tags:
            epoch_value = _latest_scalar(accumulator, tag)
            if epoch_value is not None:
                break
    if epoch_value is None:
        raise SystemExit(1)

    fps = "unknown"
    fps_tag = "performance/step_inference_rl_update_fps"
    if fps_tag in scalar_tags:
        fps_value = _latest_scalar(accumulator, fps_tag)
        if fps_value is not None:
            fps = str(round(fps_value.value))

    print(f"{epoch_value.step}|{fps}")


if __name__ == "__main__":
    main()
