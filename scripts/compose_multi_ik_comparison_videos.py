#!/usr/bin/env python3
"""Compose labeled single-IK versus multi-IK videos with OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _open(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return capture


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 52), (20, 20, 20), thickness=-1)
    cv2.putText(output, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def compose_pair(*, single_path: Path, multi_path: Path, output_path: Path, case_label: str) -> int:
    single = _open(single_path)
    multi = _open(multi_path)
    fps = min(value for value in (single.get(cv2.CAP_PROP_FPS), multi.get(cv2.CAP_PROP_FPS)) if value > 0.0)
    width = int(single.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(single.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (2 * width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output_path}")
    last_single = np.zeros((height, width, 3), dtype=np.uint8)
    last_multi = np.zeros((height, width, 3), dtype=np.uint8)
    frame_count = 0
    while True:
        ok_single, frame_single = single.read()
        ok_multi, frame_multi = multi.read()
        if not ok_single and not ok_multi:
            break
        if ok_single:
            last_single = cv2.resize(frame_single, (width, height))
        if ok_multi:
            last_multi = cv2.resize(frame_multi, (width, height))
        left = _label(last_single, f"Single IK | {case_label}")
        right = _label(last_multi, f"Multi IK beam search | {case_label}")
        writer.write(np.hstack((left, right)))
        frame_count += 1
    single.release()
    multi.release()
    writer.release()
    return frame_count


def concatenate(*, inputs: tuple[Path, ...], output_path: Path) -> int:
    if not inputs:
        raise ValueError("At least one comparison video is required.")
    first = _open(inputs[0])
    fps = first.get(cv2.CAP_PROP_FPS)
    width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output_path}")
    frame_count = 0
    for path in inputs:
        capture = _open(path)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(cv2.resize(frame, (width, height)))
            frame_count += 1
        capture.release()
    writer.release()
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path)
    parser.add_argument("--multi", type=Path)
    parser.add_argument("--case-label", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concatenate", type=Path, nargs="*", default=())
    args = parser.parse_args()
    if args.concatenate:
        frames = concatenate(inputs=tuple(args.concatenate), output_path=args.output)
    else:
        if args.single is None or args.multi is None:
            parser.error("--single and --multi are required unless --concatenate is used.")
        frames = compose_pair(
            single_path=args.single,
            multi_path=args.multi,
            output_path=args.output,
            case_label=str(args.case_label),
        )
    print(f"Wrote {frames} frames to {args.output}.")


if __name__ == "__main__":
    main()
