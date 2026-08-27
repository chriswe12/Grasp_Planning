#!/usr/bin/env python3
"""Record offline policy predictions beside one saved curriculum episode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.rl.dataset_visualizer import (  # noqa: E402
    load_episode_visualization,
    load_overview_frames,
    prepend_overview_frame,
    render_episode_frame,
)
from grasp_planning.rl.visual_servo_dataset import (  # noqa: E402
    ANGULAR_ACTION_SCALE_RAD_S,
    LINEAR_ACTION_SCALE_M_S,
    VisualServoFrameDataset,
    normalize_twist,
    world_twist_to_camera,
)
from grasp_planning.rl.visual_servo_policy import ResidualVisualServoPolicy  # noqa: E402
from grasp_planning.video import OpenCvVideoWriter  # noqa: E402


def _prediction_panel(
    *,
    prediction: np.ndarray,
    expert: np.ndarray,
    height: int,
    step_index: int,
    step_count: int,
) -> np.ndarray:
    panel = Image.new("RGB", (430, height), (20, 23, 28))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), "CAMERA-FRAME RESIDUAL POLICY", fill=(245, 245, 245))
    draw.text((12, 30), f"step {step_index + 1}/{step_count}", fill=(180, 190, 205))
    labels = ("vx", "vy", "vz", "wx", "wy", "wz")
    units = ("m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s")
    y = 62
    for axis, (label, unit) in enumerate(zip(labels, units, strict=True)):
        draw.text(
            (12, y),
            f"{label} expert={expert[axis]:+8.4f}  predicted={prediction[axis]:+8.4f} {unit}",
            fill=(230, 230, 230),
        )
        center_x = 215
        scale = 150.0 / (
            LINEAR_ACTION_SCALE_M_S if axis < 3 else ANGULAR_ACTION_SCALE_RAD_S
        )
        draw.line((center_x, y + 15, center_x, y + 27), fill=(120, 120, 120), width=1)
        for value, color, offset in (
            (expert[axis], (50, 210, 100), 18),
            (prediction[axis], (255, 170, 40), 24),
        ):
            endpoint = int(np.clip(center_x + value * scale, 15, 415))
            draw.line((center_x, y + offset, endpoint, y + offset), fill=color, width=4)
        y += 34
    linear_error = np.mean(np.abs(prediction[:3] - expert[:3])) * 1000.0
    angular_error = (
        np.mean(np.abs(prediction[3:] - expert[3:])) * 180.0 / np.pi
    )
    draw.text((12, height - 42), f"linear MAE: {linear_error:.3f} mm/s", fill=(120, 210, 255))
    draw.text((12, height - 24), f"angular MAE: {angular_error:.3f} deg/s", fill=(120, 210, 255))
    return np.asarray(panel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("episode_npz", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hold-final-seconds", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    arrays = episode.arrays
    required = {"tcp_orientation_xyzw_w", "joint_positions"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(
            f"{args.episode_npz} lacks {missing}; use an episode from the batched collector."
        )
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = ResidualVisualServoPolicy().to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    action_scale = np.array(
        [LINEAR_ACTION_SCALE_M_S] * 3 + [ANGULAR_ACTION_SCALE_RAD_S] * 3,
        dtype=np.float32,
    )
    output = args.output or args.episode_npz.with_name(
        f"{args.episode_npz.stem}_{args.checkpoint.stem}_policy.mp4"
    )
    frames = []
    linear_errors = []
    angular_errors = []
    with torch.inference_mode():
        for step_index in range(episode.step_count):
            tcp_orientation = arrays["tcp_orientation_xyzw_w"][step_index]
            nominal_camera = world_twist_to_camera(
                arrays["nominal_twist"][step_index], tcp_orientation
            )
            expert_camera = world_twist_to_camera(
                arrays["expert_residual_twist"][step_index], tcp_orientation
            ).astype(np.float32)
            inputs = {
                "live_rgbd": VisualServoFrameDataset._rgbd(
                    arrays["rgb_live"][step_index], arrays["depth_live"][step_index]
                )
                .unsqueeze(0)
                .to(args.device),
                "goal_rgbd": VisualServoFrameDataset._rgbd(
                    arrays["rgb_goal"], arrays["depth_goal"]
                )
                .unsqueeze(0)
                .to(args.device),
                "joint_positions": torch.from_numpy(
                    arrays["joint_positions"][step_index].astype(np.float32)
                )
                .unsqueeze(0)
                .to(args.device),
                "progress": torch.tensor(
                    [[arrays["trajectory_progress"][step_index]]],
                    dtype=torch.float32,
                    device=args.device,
                ),
                "nominal_twist_camera": torch.from_numpy(
                    normalize_twist(nominal_camera)
                )
                .unsqueeze(0)
                .to(args.device),
            }
            prediction_camera = model(**inputs)[0].cpu().numpy() * action_scale
            linear_errors.append(
                float(np.mean(np.abs(prediction_camera[:3] - expert_camera[:3])))
            )
            angular_errors.append(
                float(np.mean(np.abs(prediction_camera[3:] - expert_camera[3:])))
            )
            diagnostic = render_episode_frame(episode, step_index)
            if overview_frames is not None:
                diagnostic = prepend_overview_frame(
                    diagnostic,
                    overview_frames[step_index],
                )
            panel = _prediction_panel(
                prediction=prediction_camera,
                expert=expert_camera,
                height=diagnostic.shape[0],
                step_index=step_index,
                step_count=episode.step_count,
            )
            frames.append(np.concatenate((diagnostic, panel), axis=1))

    with OpenCvVideoWriter(
        output,
        fps=args.fps,
        width=frames[0].shape[1],
        height=frames[0].shape[0],
    ) as writer:
        for frame in frames:
            writer.append_rgb(frame)
        for _ in range(max(0, int(round(args.hold_final_seconds * args.fps)))):
            writer.append_rgb(frames[-1])
        frame_count = writer.frame_count
    print(
        f"Wrote {frame_count} frames to {output}; "
        f"linear_mae={np.mean(linear_errors) * 1000.0:.3f} mm/s "
        f"angular_mae={np.mean(angular_errors) * 180.0 / np.pi:.3f} deg/s."
    )


if __name__ == "__main__":
    main()
