#!/usr/bin/env python3
"""Add case labels and concatenate dual-holder Isaac stress recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = [case for case in manifest["cases"] if case.get("video") and Path(case["video"]).is_file()]
    if not cases:
        raise RuntimeError("No successful holder stress recordings to compose.")

    first = cv2.VideoCapture(str(cases[0]["video"]))
    fps = float(first.get(cv2.CAP_PROP_FPS))
    width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {args.output}")
    frames = 0
    for index, case in enumerate(cases, start=1):
        capture = cv2.VideoCapture(str(case["video"]))
        label = (
            f"{index}/{len(cases)} {case['name']} | base=({case['x']:.2f},"
            f" {case['y']:.2f}) | holder={case['holder_grasp_id']} | "
            f"{'PASS' if case.get('success') else 'FAIL'}"
        )
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, (width, height))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 54), (15, 15, 15), -1)
            frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0.0)
            cv2.putText(
                frame,
                label,
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(np.asarray(frame))
            frames += 1
        capture.release()
    writer.release()
    manifest["combined_video"] = str(args.output.resolve())
    manifest["combined_frame_count"] = frames
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {frames} frames to {args.output}.")


if __name__ == "__main__":
    main()
