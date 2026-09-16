#!/usr/bin/env python3
"""Render a deterministic preview grid of the provisional D405 sensor model."""

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

from grasp_planning.rl.d405_observation import (  # noqa: E402
    D405ObservationPreprocessCfg,
    resize_aligned_rgbd_torch,
)
from grasp_planning.rl.live_observation_randomization import (  # noqa: E402
    LiveObservationRandomizationCfg,
    LiveObservationRandomizer,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("isaac_rl/data/plumbers_block/goal_catalog.npz"),
    )
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/d405_randomization/provisional_sensor_grid.png"),
    )
    return parser.parse_args()


def _depth_image(depth_m: np.ndarray, *, minimum: float = 0.07, maximum: float = 0.50) -> np.ndarray:
    normalized = np.clip((depth_m - minimum) / (maximum - minimum), 0.0, 1.0)
    invalid = (~np.isfinite(depth_m)) | (depth_m < minimum) | (depth_m >= maximum)
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    image = np.stack((red, green, blue), axis=-1)
    image[invalid] = 0.0
    return np.round(image * 255.0).astype(np.uint8)


def main() -> None:
    args = _parse_args()
    catalog = args.catalog.expanduser().resolve()
    with np.load(catalog, allow_pickle=False) as source:
        target_count = len(source["target_ids"])
        if not 0 <= args.target_index < target_count:
            raise ValueError(f"--target-index must be in [0, {target_count - 1}].")
        target_id = str(source["target_ids"][args.target_index])
        rgb = source["goal_rgb"][args.target_index]
        depth = source["goal_depth"][args.target_index]

    torch.manual_seed(args.seed)
    rgb_tensor = torch.as_tensor(rgb, dtype=torch.float32).div(255.0).unsqueeze(0)
    depth_tensor = torch.as_tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    preprocess_cfg = D405ObservationPreprocessCfg()
    rgb_tensor, depth_tensor, _ = resize_aligned_rgbd_torch(
        rgb_tensor, depth_tensor, cfg=preprocess_cfg
    )
    randomizer = LiveObservationRandomizer(
        LiveObservationRandomizationCfg(clean_episode_fraction=0.0),
        num_envs=1,
        device="cpu",
    )

    columns: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, strength in (("canonical", 0.0), ("weak", 0.33), ("medium", 0.66), ("full", 1.0)):
        randomizer.sample(torch.tensor([0]), strength=strength)
        varied_rgb, varied_depth = randomizer.apply(rgb_tensor, depth_tensor)
        columns.append(
            (
                label,
                np.round(varied_rgb[0].clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8),
                varied_depth[0, ..., 0].numpy(),
            )
        )

    scale = 3
    panel_width = preprocess_cfg.output_width * scale
    panel_height = preprocess_cfg.output_height * scale
    label_height = 32
    canvas = Image.new("RGB", (panel_width * len(columns), label_height + 2 * panel_height), "black")
    draw = ImageDraw.Draw(canvas)
    for column_index, (label, rgb_image, depth_image) in enumerate(columns):
        x = column_index * panel_width
        draw.text((x + 6, 8), f"{label}  strength={column_index / 3:.2f}", fill="white")
        rgb_panel = Image.fromarray(rgb_image).resize((panel_width, panel_height), Image.Resampling.NEAREST)
        depth_panel = Image.fromarray(_depth_image(depth_image)).resize(
            (panel_width, panel_height), Image.Resampling.NEAREST
        )
        canvas.paste(rgb_panel, (x, label_height))
        canvas.paste(depth_panel, (x, label_height + panel_height))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"[DONE] target={target_id} seed={args.seed} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
