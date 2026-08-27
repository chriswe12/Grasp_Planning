from __future__ import annotations

import json

import numpy as np
import torch

from grasp_planning.rl.visual_servo_dataset import (
    EpisodeGroupedBatchSampler,
    LocalityBlockBatchSampler,
    VisualServoFrameDataset,
    camera_twist_to_world,
    world_twist_to_camera,
)
from grasp_planning.rl.visual_servo_policy import ResidualVisualServoPolicy


def test_world_twist_to_camera_uses_tcp_and_camera_rotations() -> None:
    identity_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    camera_in_tcp = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    twist_camera = world_twist_to_camera(
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        identity_xyzw,
        rotation_camera_in_tcp=camera_in_tcp,
    )
    assert np.allclose(twist_camera, [0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(
        camera_twist_to_world(
            twist_camera,
            identity_xyzw,
            rotation_camera_in_tcp=camera_in_tcp,
        ),
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )


def _write_episode(root, *, episode_index: int, split: str) -> None:
    steps, height, width = 2, 16, 24
    stem = f"episode_{episode_index:06d}"
    np.savez_compressed(
        root / f"{stem}.npz",
        rgb_live=np.full((steps, height, width, 3), 64, dtype=np.uint8),
        depth_live=np.full((steps, height, width), 0.2, dtype=np.float32),
        rgb_goal=np.full((height, width, 3), 96, dtype=np.uint8),
        depth_goal=np.full((height, width), 0.15, dtype=np.float32),
        joint_positions=np.zeros((steps, 7), dtype=np.float32),
        tcp_orientation_xyzw_w=np.tile([0.0, 0.0, 0.0, 1.0], (steps, 1)).astype(np.float32),
        nominal_twist=np.zeros((steps, 6), dtype=np.float32),
        expert_residual_twist=np.zeros((steps, 6), dtype=np.float32),
        trajectory_progress=np.linspace(0.0, 1.0, steps, dtype=np.float32),
    )
    (root / f"{stem}.json").write_text(
        json.dumps(
            {
                "episode_index": episode_index,
                "split": split,
                "success": True,
            }
        ),
        encoding="utf-8",
    )


def test_dataset_and_policy_shapes(tmp_path) -> None:
    _write_episode(tmp_path, episode_index=1, split="train")
    dataset = VisualServoFrameDataset(tmp_path, split="train")
    sample = dataset[0]
    assert sample["live_rgbd"].shape == (4, 16, 24)
    assert sample["goal_rgbd"].shape == (4, 16, 24)
    assert sample["nominal_twist_camera"].shape == (6,)

    model = ResidualVisualServoPolicy(feature_channels=32)
    prediction = model(
        live_rgbd=sample["live_rgbd"].unsqueeze(0),
        goal_rgbd=sample["goal_rgbd"].unsqueeze(0),
        joint_positions=sample["joint_positions"].unsqueeze(0),
        progress=sample["progress"].unsqueeze(0),
        nominal_twist_camera=sample["nominal_twist_camera"].unsqueeze(0),
    )
    assert prediction.shape == (1, 6)
    assert torch.all(torch.abs(prediction) <= 1.0)


def test_lazy_dataset_cache_and_grouped_batches(tmp_path) -> None:
    _write_episode(tmp_path, episode_index=1, split="train")
    _write_episode(tmp_path, episode_index=2, split="train")
    dataset = VisualServoFrameDataset(
        tmp_path,
        split="train",
        cache_episodes=1,
    )
    assert len(dataset) == 4
    assert dataset._episode_cache == {}
    assert dataset[0]["episode_index"].item() == 1
    assert list(dataset._episode_cache) == [0]
    assert dataset[2]["episode_index"].item() == 2
    assert list(dataset._episode_cache) == [1]

    sampler = EpisodeGroupedBatchSampler(
        dataset,
        batch_size=3,
        shuffle=False,
    )
    batches = list(sampler)
    assert batches == [[0, 1, 2], [3]]
    assert sorted(index for batch in batches for index in batch) == list(range(4))


def test_policy_broadcasts_one_shared_goal_across_live_batch() -> None:
    model = ResidualVisualServoPolicy(feature_channels=32)
    prediction = model(
        live_rgbd=torch.zeros((3, 4, 16, 24)),
        goal_rgbd=torch.zeros((1, 4, 16, 24)),
        joint_positions=torch.zeros((3, 7)),
        progress=torch.zeros((3, 1)),
        nominal_twist_camera=torch.zeros((3, 6)),
    )
    assert prediction.shape == (3, 6)


def test_locality_block_sampler_keeps_every_frame() -> None:
    dataset = list(range(103))
    sampler = LocalityBlockBatchSampler(
        dataset,
        batch_size=16,
        block_size=32,
        shuffle=True,
        seed=4,
    )
    batches = list(sampler)
    assert all(len(batch) == 16 for batch in batches[:-1])
    assert len(batches[-1]) == 7
    assert sorted(index for batch in batches for index in batch) == list(range(103))
