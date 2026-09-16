#!/usr/bin/env python3
"""Label a continuous holder-stress recording from its attempt artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-video", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sequence = json.loads(args.sequence.read_text(encoding="utf-8"))
    attempt = json.loads(args.attempt.read_text(encoding="utf-8"))
    metadata = {str(Path(case["plan"]).resolve()): case for case in sequence["cases"]}
    ranges = []
    for execution in attempt["result"]["holder_sequence"]:
        case = metadata[str(Path(execution["plan"]).resolve())]
        ranges.append((execution, case))

    capture = cv2.VideoCapture(str(args.raw_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.raw_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        active = None
        for execution, case in ranges:
            if int(execution["video_start_frame"]) <= frame_index <= int(execution["video_end_frame"]):
                active = (execution, case)
                break
        if active is not None:
            execution, case = active
            label = (
                f"{int(execution['index'])}/{len(ranges)} {case['name']} | "
                f"base=({case['x']:.2f}, {case['y']:.2f}) | "
                f"holder={execution['holder_grasp_id']} | "
                f"{'PASS' if execution['success'] else 'FAIL'}"
            )
        else:
            label = "Continuous holder stress | transition without articulation reset"
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 54), (15, 15, 15), -1)
        frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0.0)
        cv2.putText(
            frame,
            label,
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_index += 1
    capture.release()
    writer.release()
    print(f"Wrote {frame_index} frames to {args.output}.")


if __name__ == "__main__":
    main()
