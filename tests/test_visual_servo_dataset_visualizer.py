from __future__ import annotations

import json

import numpy as np

from grasp_planning.rl.dataset_visualizer import (
    load_episode_visualization,
    load_overview_frames,
    prepend_overview_frame,
    render_episode_frame,
)


def test_load_and_render_episode(tmp_path) -> None:
    steps, height, width = 3, 12, 20
    npz_path = tmp_path / "episode_000082.npz"
    np.savez_compressed(
        npz_path,
        rgb_live=np.full((steps, height, width, 3), 80, dtype=np.uint8),
        depth_live=np.linspace(0.1, 0.5, steps * height * width, dtype=np.float32).reshape(steps, height, width),
        object_mask=np.zeros((steps, height, width), dtype=np.uint8),
        rgb_goal=np.full((height, width, 3), 120, dtype=np.uint8),
        nominal_twist=np.zeros((steps, 6), dtype=np.float32),
        expert_twist=np.zeros((steps, 6), dtype=np.float32),
        expert_residual_twist=np.zeros((steps, 6), dtype=np.float32),
        pose_error=np.zeros((steps, 6), dtype=np.float32),
        trajectory_progress=np.linspace(0.0, 1.0, steps, dtype=np.float32),
    )
    npz_path.with_suffix(".json").write_text(
        json.dumps({"episode_index": 82, "split": "train", "success": False}),
        encoding="utf-8",
    )

    episode = load_episode_visualization(npz_path)
    frame = render_episode_frame(episode, 1)

    assert episode.step_count == steps
    assert frame.shape == (2 * height, 2 * width + 384, 3)
    assert frame.dtype == np.uint8


def test_load_and_prepend_overview_frames(tmp_path) -> None:
    overview_path = tmp_path / "overview.npz"
    overview = np.full((3, 60, 80, 3), 127, dtype=np.uint8)
    np.savez_compressed(overview_path, overview_rgb=overview)

    loaded = load_overview_frames(overview_path, expected_steps=3)
    diagnostic = np.zeros((120, 200, 3), dtype=np.uint8)
    combined = prepend_overview_frame(diagnostic, loaded[0])

    assert loaded.shape == (3, 60, 80, 3)
    assert combined.shape == (120, 360, 3)
    assert combined.dtype == np.uint8
