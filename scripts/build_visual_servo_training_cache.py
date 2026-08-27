#!/usr/bin/env python3
"""Build a sequential memory-mapped cache for visual-servo BC training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.d405_wrist_camera import (  # noqa: E402
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
)
from grasp_planning.rl.visual_servo_dataset import (  # noqa: E402
    normalize_twist,
    world_twist_to_camera,
)


def _area_resize(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    source = np.asarray(image)
    source_height, source_width = source.shape[:2]
    if source_height % height or source_width % width:
        raise ValueError(
            f"Area resize requires integer factors, got {source.shape[:2]} -> "
            f"{(height, width)}."
        )
    factor_y = source_height // height
    factor_x = source_width // width
    if source.ndim == 2:
        resized = source.reshape(height, factor_y, width, factor_x).mean(
            axis=(1, 3)
        )
    elif source.ndim == 3:
        resized = source.reshape(
            height, factor_y, width, factor_x, source.shape[2]
        ).mean(axis=(1, 3))
    else:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {source.shape}.")
    if np.issubdtype(source.dtype, np.integer):
        resized = np.rint(resized)
    return resized.astype(source.dtype, copy=False)


def _area_resize_batch(
    images: np.ndarray, *, height: int, width: int
) -> np.ndarray:
    return np.stack(
        [_area_resize(image, height=height, width=width) for image in images],
        axis=0,
    )


def _selected_episodes(dataset_dir: Path, split: str) -> list[Path]:
    selected = []
    for metadata_path in sorted(dataset_dir.glob("episode_*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("split") != split or not metadata.get("success", False):
            continue
        selected.append(metadata_path.with_suffix(".npz"))
    return selected


def _open_memmap(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.memmap:
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument(
        "--maximum-frames-per-episode",
        type=int,
        default=256,
        help="Sparse allocation capacity; files are truncated to the actual frame count.",
    )
    args = parser.parse_args()
    if args.downsample < 1 or args.maximum_frames_per_episode < 1:
        parser.error("--downsample and --maximum-frames-per-episode must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            int(existing_manifest.get("version", -1)) != 2
            or existing_manifest.get("resampling") != "area"
            or existing_manifest.get("observation_profile")
            != D405_VISUAL_SERVO_OBSERVATION_PROFILE
        ):
            raise ValueError(
                f"Existing cache at {args.output_dir} uses an obsolete visual "
                "observation profile. Rebuild it in a new output directory."
            )
        print(f"[CACHE] Already complete: {manifest_path}", flush=True)
        return

    selections = {
        split: _selected_episodes(args.dataset_dir, split)
        for split in ("train", "validation")
    }
    first_path = selections["train"][0]
    with np.load(first_path) as first:
        source_height, source_width = first["rgb_live"].shape[1:3]
        height = source_height // args.downsample
        width = source_width // args.downsample
        if (height * args.downsample, width * args.downsample) != (
            source_height,
            source_width,
        ):
            raise ValueError(
                f"Source image shape {(source_height, source_width)} must be divisible "
                f"by --downsample={args.downsample}."
            )
        goal_rgb = _area_resize(first["rgb_goal"], height=height, width=width)
        goal_depth = _area_resize(first["depth_goal"], height=height, width=width)
    np.save(args.output_dir / "goal_rgb.npy", goal_rgb)
    np.save(args.output_dir / "goal_depth.npy", goal_depth.astype(np.float16))

    split_manifests = {}
    started_at = time.monotonic()
    processed_frames = 0
    for split, episodes in selections.items():
        capacity = len(episodes) * args.maximum_frames_per_episode
        prefix = args.output_dir / split
        output_specs = {
            "rgb": (
                Path(f"{prefix}_live_rgb.uint8"),
                np.uint8,
                (capacity, height, width, 3),
            ),
            "depth": (
                Path(f"{prefix}_live_depth.float16"),
                np.float16,
                (capacity, height, width),
            ),
            "joint": (
                Path(f"{prefix}_joint_positions.float32"),
                np.float32,
                (capacity, 7),
            ),
            "progress": (
                Path(f"{prefix}_progress.float32"),
                np.float32,
                (capacity, 1),
            ),
            "nominal": (
                Path(f"{prefix}_nominal_twist_camera.float32"),
                np.float32,
                (capacity, 6),
            ),
            "residual": (
                Path(f"{prefix}_residual_twist_camera.float32"),
                np.float32,
                (capacity, 6),
            ),
        }
        outputs = {
            name: _open_memmap(path, dtype, shape)
            for name, (path, dtype, shape) in output_specs.items()
        }
        cursor = 0
        for episode_index, npz_path in enumerate(episodes, start=1):
            with np.load(npz_path) as archive:
                count = int(archive["trajectory_progress"].shape[0])
                if count > args.maximum_frames_per_episode:
                    raise ValueError(
                        f"{npz_path} has {count} frames, exceeding "
                        f"--maximum-frames-per-episode={args.maximum_frames_per_episode}."
                    )
                destination = slice(cursor, cursor + count)
                orientations = archive["tcp_orientation_xyzw_w"]
                outputs["rgb"][destination] = _area_resize_batch(
                    archive["rgb_live"], height=height, width=width
                )
                outputs["depth"][destination] = _area_resize_batch(
                    archive["depth_live"], height=height, width=width
                )
                outputs["joint"][destination] = archive["joint_positions"]
                outputs["progress"][destination, 0] = archive["trajectory_progress"]
                outputs["nominal"][destination] = normalize_twist(
                    world_twist_to_camera(archive["nominal_twist"], orientations)
                )
                outputs["residual"][destination] = normalize_twist(
                    world_twist_to_camera(
                        archive["expert_residual_twist"], orientations
                    )
                )
            cursor += count
            processed_frames += count
            if episode_index == 1 or episode_index % 100 == 0:
                elapsed = max(time.monotonic() - started_at, 1.0e-6)
                print(
                    f"[CACHE] split={split} episode={episode_index}/{len(episodes)} "
                    f"frames={processed_frames} rate={processed_frames / elapsed:.1f} frames/s",
                    flush=True,
                )
        for output in outputs.values():
            output.flush()
        del output
        del outputs
        for path, dtype, shape in output_specs.values():
            elements_per_frame = int(np.prod(shape[1:], dtype=np.int64))
            os.truncate(path, cursor * elements_per_frame * np.dtype(dtype).itemsize)
        split_manifests[split] = {
            "episode_count": len(episodes),
            "frame_count": cursor,
        }

    manifest = {
        "version": 2,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "downsample": args.downsample,
        "resampling": "area",
        "observation_profile": D405_VISUAL_SERVO_OBSERVATION_PROFILE,
        "source_image_shape": [source_height, source_width],
        "image_shape": [height, width],
        "splits": split_manifests,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(f"[CACHE] Complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
