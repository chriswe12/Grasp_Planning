"""PyTorch dataset for camera-frame residual visual-servo behavior cloning."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    D405WristCameraConfig,
)

LINEAR_ACTION_SCALE_M_S = 0.05
ANGULAR_ACTION_SCALE_RAD_S = 0.30
DEPTH_MIN_M = 0.04
DEPTH_MAX_M = 0.50


def quaternion_xyzw_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized XYZW quaternions to [..., 3, 3] rotation matrices."""

    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def world_twist_to_camera(
    twist_world: np.ndarray,
    tcp_orientation_xyzw_world: np.ndarray,
    *,
    rotation_camera_in_tcp: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate world-frame linear/angular velocity into the calibrated camera frame."""

    twist_world = np.asarray(twist_world, dtype=np.float64)
    rotation_world_from_tcp = quaternion_xyzw_to_rotation_matrix(tcp_orientation_xyzw_world)
    if rotation_camera_in_tcp is None:
        rotation_camera_in_tcp = np.asarray(
            D405WristCameraConfig().rotation_camera_in_calibration_parent,
            dtype=np.float64,
        ).reshape(3, 3)
    rotation_world_from_camera = rotation_world_from_tcp @ np.asarray(
        rotation_camera_in_tcp, dtype=np.float64
    )
    rotation_camera_from_world = np.swapaxes(rotation_world_from_camera, -1, -2)
    linear = np.einsum("...ij,...j->...i", rotation_camera_from_world, twist_world[..., :3])
    angular = np.einsum("...ij,...j->...i", rotation_camera_from_world, twist_world[..., 3:])
    return np.concatenate((linear, angular), axis=-1)


def camera_twist_to_world(
    twist_camera: np.ndarray,
    tcp_orientation_xyzw_world: np.ndarray,
    *,
    rotation_camera_in_tcp: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate camera-frame linear/angular velocity into the world frame."""

    twist_camera = np.asarray(twist_camera, dtype=np.float64)
    rotation_world_from_tcp = quaternion_xyzw_to_rotation_matrix(
        tcp_orientation_xyzw_world
    )
    if rotation_camera_in_tcp is None:
        rotation_camera_in_tcp = np.asarray(
            D405WristCameraConfig().rotation_camera_in_calibration_parent,
            dtype=np.float64,
        ).reshape(3, 3)
    rotation_world_from_camera = rotation_world_from_tcp @ np.asarray(
        rotation_camera_in_tcp, dtype=np.float64
    )
    linear = np.einsum(
        "...ij,...j->...i", rotation_world_from_camera, twist_camera[..., :3]
    )
    angular = np.einsum(
        "...ij,...j->...i", rotation_world_from_camera, twist_camera[..., 3:]
    )
    return np.concatenate((linear, angular), axis=-1)


def normalize_twist(twist: np.ndarray) -> np.ndarray:
    scale = np.array(
        [LINEAR_ACTION_SCALE_M_S] * 3 + [ANGULAR_ACTION_SCALE_RAD_S] * 3,
        dtype=np.float32,
    )
    return np.asarray(twist, dtype=np.float32) / scale


@dataclass(frozen=True)
class LoadedEpisode:
    episode_index: int
    npz_path: Path
    frame_count: int


class VisualServoFrameDataset(Dataset):
    """In-memory frame dataset restricted to successful expert episodes."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: str,
        max_episodes: int = 0,
        successful_only: bool = True,
        cache_episodes: int = 2,
        raw_images: bool = False,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")
        self.dataset_dir = Path(dataset_dir)
        if cache_episodes < 1:
            raise ValueError("cache_episodes must be >= 1.")
        self.cache_episodes = int(cache_episodes)
        self.raw_images = bool(raw_images)
        self._episode_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        metadata_paths = sorted(self.dataset_dir.glob("episode_*.json"))
        selected: list[tuple[Path, dict[str, object]]] = []
        for metadata_path in metadata_paths:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if str(metadata.get("split")) != split:
                continue
            if successful_only and not bool(metadata.get("success", False)):
                continue
            selected.append((metadata_path, metadata))
            if max_episodes > 0 and len(selected) >= max_episodes:
                break
        if not selected:
            raise ValueError(f"No {split} episodes selected under {self.dataset_dir}.")

        self.episodes: list[LoadedEpisode] = []
        self.frame_index: list[tuple[int, int]] = []
        self.episode_frame_indices: list[range] = []
        required = {
            "rgb_live",
            "depth_live",
            "rgb_goal",
            "depth_goal",
            "joint_positions",
            "tcp_orientation_xyzw_w",
            "nominal_twist",
            "expert_residual_twist",
            "trajectory_progress",
        }
        for metadata_path, metadata in selected:
            npz_path = metadata_path.with_suffix(".npz")
            with np.load(npz_path) as episode:
                missing = sorted(required.difference(episode.files))
                if missing:
                    raise ValueError(
                        f"{npz_path} lacks {missing}. Regenerate it with the batched collector "
                        "so measured TCP poses are stored."
                    )
                frame_count = int(episode["trajectory_progress"].shape[0])
            episode_slot = len(self.episodes)
            episode_index = int(metadata["episode_index"])
            self.episodes.append(
                LoadedEpisode(
                    episode_index=episode_index,
                    npz_path=npz_path,
                    frame_count=frame_count,
                )
            )
            first_frame_index = len(self.frame_index)
            self.frame_index.extend(
                (episode_slot, step_index)
                for step_index in range(frame_count)
            )
            self.episode_frame_indices.append(
                range(first_frame_index, first_frame_index + frame_count)
            )
        self.required_arrays = frozenset(required)

    def __len__(self) -> int:
        return len(self.frame_index)

    def _load_episode_arrays(self, episode_slot: int) -> dict[str, np.ndarray]:
        cached = self._episode_cache.pop(episode_slot, None)
        if cached is not None:
            self._episode_cache[episode_slot] = cached
            return cached
        episode = self.episodes[episode_slot]
        with np.load(episode.npz_path) as archive:
            arrays = {
                name: archive[name].copy() for name in self.required_arrays
            }
        self._episode_cache[episode_slot] = arrays
        while len(self._episode_cache) > self.cache_episodes:
            self._episode_cache.popitem(last=False)
        return arrays

    def clear_cache(self) -> None:
        """Release decompressed episode arrays held by this dataset instance."""

        self._episode_cache.clear()

    @staticmethod
    def _rgbd(rgb: np.ndarray, depth: np.ndarray) -> torch.Tensor:
        rgb_float = np.asarray(rgb, dtype=np.float32) / 255.0
        depth_float = np.clip(
            (np.asarray(depth, dtype=np.float32) - DEPTH_MIN_M)
            / (DEPTH_MAX_M - DEPTH_MIN_M),
            0.0,
            1.0,
        )
        rgbd = np.concatenate((rgb_float, depth_float[..., None]), axis=-1)
        return torch.from_numpy(np.moveaxis(rgbd, -1, 0).copy())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_slot, step_index = self.frame_index[index]
        loaded = self.episodes[episode_slot]
        arrays = self._load_episode_arrays(episode_slot)
        tcp_orientation = arrays["tcp_orientation_xyzw_w"][step_index]
        nominal_camera = world_twist_to_camera(
            arrays["nominal_twist"][step_index], tcp_orientation
        )
        residual_camera = world_twist_to_camera(
            arrays["expert_residual_twist"][step_index], tcp_orientation
        )
        sample = {
            "joint_positions": torch.from_numpy(
                arrays["joint_positions"][step_index].astype(np.float32)
            ),
            "progress": torch.tensor(
                [arrays["trajectory_progress"][step_index]], dtype=torch.float32
            ),
            "nominal_twist_camera": torch.from_numpy(
                normalize_twist(nominal_camera)
            ),
            "residual_twist_camera": torch.from_numpy(
                normalize_twist(residual_camera)
            ),
            "episode_index": torch.tensor(loaded.episode_index, dtype=torch.int64),
            "step_index": torch.tensor(step_index, dtype=torch.int64),
        }
        if self.raw_images:
            sample.update(
                {
                    "live_rgb": torch.from_numpy(
                        arrays["rgb_live"][step_index]
                    ),
                    "live_depth": torch.from_numpy(
                        arrays["depth_live"][step_index]
                    ),
                    "goal_rgb": torch.from_numpy(arrays["rgb_goal"]),
                    "goal_depth": torch.from_numpy(arrays["depth_goal"]),
                }
            )
        else:
            sample.update(
                {
                    "live_rgbd": self._rgbd(
                        arrays["rgb_live"][step_index],
                        arrays["depth_live"][step_index],
                    ),
                    "goal_rgbd": self._rgbd(
                        arrays["rgb_goal"],
                        arrays["depth_goal"],
                    ),
                }
            )
        return sample


class MmapVisualServoFrameDataset(Dataset):
    """Frame dataset backed by a contiguous, preprocessed memory-mapped cache."""

    def __init__(self, cache_dir: str | Path, *, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")
        self.cache_dir = Path(cache_dir)
        manifest = json.loads(
            (self.cache_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if int(manifest.get("version", -1)) != 2:
            raise ValueError(
                "Visual-servo cache must use schema 2 area filtering; rebuild it "
                "with scripts/build_visual_servo_training_cache.py."
            )
        if manifest.get("resampling") != "area":
            raise ValueError("Visual-servo cache resampling must be 'area'.")
        if (
            manifest.get("observation_profile")
            != D405_VISUAL_SERVO_OBSERVATION_PROFILE
        ):
            raise ValueError(
                "Visual-servo cache observation profile does not match the current "
                f"pipeline ({D405_VISUAL_SERVO_OBSERVATION_PROFILE})."
            )
        split_manifest = manifest["splits"][split]
        self.frame_count = int(split_manifest["frame_count"])
        self.image_shape = tuple(int(value) for value in manifest["image_shape"])
        prefix = self.cache_dir / split
        height, width = self.image_shape
        self.live_rgb = np.memmap(
            f"{prefix}_live_rgb.uint8",
            mode="r",
            dtype=np.uint8,
            shape=(self.frame_count, height, width, 3),
        )
        self.live_depth = np.memmap(
            f"{prefix}_live_depth.float16",
            mode="r",
            dtype=np.float16,
            shape=(self.frame_count, height, width),
        )
        self.joint_positions = np.memmap(
            f"{prefix}_joint_positions.float32",
            mode="r",
            dtype=np.float32,
            shape=(self.frame_count, 7),
        )
        self.progress = np.memmap(
            f"{prefix}_progress.float32",
            mode="r",
            dtype=np.float32,
            shape=(self.frame_count, 1),
        )
        self.nominal_twist_camera = np.memmap(
            f"{prefix}_nominal_twist_camera.float32",
            mode="r",
            dtype=np.float32,
            shape=(self.frame_count, 6),
        )
        self.residual_twist_camera = np.memmap(
            f"{prefix}_residual_twist_camera.float32",
            mode="r",
            dtype=np.float32,
            shape=(self.frame_count, 6),
        )
        self.goal_rgb = np.load(self.cache_dir / "goal_rgb.npy", mmap_mode="r")
        self.goal_depth = np.load(self.cache_dir / "goal_depth.npy", mmap_mode="r")

    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # torch cannot safely wrap read-only memmaps, so copy only the selected frame.
        return {
            "live_rgb": torch.from_numpy(np.array(self.live_rgb[index], copy=True)),
            "live_depth": torch.from_numpy(
                np.array(self.live_depth[index], dtype=np.float32, copy=True)
            ),
            "goal_rgb": torch.from_numpy(np.array(self.goal_rgb, copy=True)),
            "goal_depth": torch.from_numpy(
                np.array(self.goal_depth, dtype=np.float32, copy=True)
            ),
            "joint_positions": torch.from_numpy(
                np.array(self.joint_positions[index], copy=True)
            ),
            "progress": torch.from_numpy(np.array(self.progress[index], copy=True)),
            "nominal_twist_camera": torch.from_numpy(
                np.array(self.nominal_twist_camera[index], copy=True)
            ),
            "residual_twist_camera": torch.from_numpy(
                np.array(self.residual_twist_camera[index], copy=True)
            ),
        }


class LocalityBlockBatchSampler(Sampler[list[int]]):
    """Shuffle mmap frames inside shuffled contiguous blocks for fast disk access."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        block_size: int = 8192,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        if batch_size < 1 or block_size < batch_size:
            raise ValueError("block_size must be at least batch_size, and both positive.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        blocks = [
            np.arange(start, min(start + self.block_size, len(self.dataset)))
            for start in range(0, len(self.dataset), self.block_size)
        ]
        if self.shuffle:
            rng.shuffle(blocks)
            for block in blocks:
                rng.shuffle(block)
        carry: list[int] = []
        for block in blocks:
            indices = carry + [int(index) for index in block]
            full_count = len(indices) // self.batch_size * self.batch_size
            for start in range(0, full_count, self.batch_size):
                yield indices[start : start + self.batch_size]
            carry = indices[full_count:]
        if carry:
            yield carry


class EpisodeGroupedBatchSampler(Sampler[list[int]]):
    """Shuffle episodes and frames while keeping NPZ access cache-friendly."""

    def __init__(
        self,
        dataset: VisualServoFrameDataset,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        drop_last: bool = False,
        episode_shuffle_block_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if episode_shuffle_block_size < 1:
            raise ValueError("episode_shuffle_block_size must be positive.")
        self.episode_shuffle_block_size = int(episode_shuffle_block_size)
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        episode_slots = np.arange(len(self.dataset.episodes))
        if self.shuffle:
            episode_blocks = [
                episode_slots[start : start + self.episode_shuffle_block_size].copy()
                for start in range(
                    0,
                    len(episode_slots),
                    self.episode_shuffle_block_size,
                )
            ]
            rng.shuffle(episode_blocks)
            for block in episode_blocks:
                rng.shuffle(block)
            episode_slots = np.concatenate(episode_blocks)
        ordered_indices: list[int] = []
        for episode_slot in episode_slots:
            frame_indices = np.fromiter(
                self.dataset.episode_frame_indices[int(episode_slot)],
                dtype=np.int64,
            )
            if self.shuffle:
                rng.shuffle(frame_indices)
            ordered_indices.extend(int(index) for index in frame_indices)
        for start in range(0, len(ordered_indices), self.batch_size):
            batch = ordered_indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
