import subprocess

import numpy as np
import pytest
import torch

from grasp_planning.real_franka.recording import RunRecording, ffmpeg_binary


@pytest.mark.parametrize("duration", [1, None])
def test_video_roundtrip_and_exact_rgbd(tmp_path, duration):
    try:
        ffmpeg_binary()
    except RuntimeError:
        pytest.skip("No local encoder")
    goal = np.zeros((72, 128, 4), np.float32)
    goal[..., 2] = 1
    recorder = RunRecording(tmp_path / "run.json", goal, duration)
    raw_depth = np.full((376, 672), 0.25, np.float32)
    raw_depth[0, 0] = np.nan
    for i in range(3):
        live = torch.full((1, 72, 128, 4), 0.1 * i)
        step = dict(
            time_s=i / 15,
            frame_seq=i,
            completion_probability=1e-6 if i == 0 else None,
            phase=["policy", "lifting", "post_lift"][i],
            base_tcp_twist=[0] * 6,
            valid_depth_fraction=0.8,
        )
        recorder.append((0, i, np.full((376, 672, 3), 40 * i, np.uint8), raw_depth), live, step)
        live[:] = 0.9  # Recorded model input must not alias the mutable caller.
    result = recorder.finish()
    with np.load(result["rgbd"]) as archive:
        np.testing.assert_array_equal(archive["goal_rgbd"], goal)
        np.testing.assert_array_equal(archive["live_rgbd"][1], np.full_like(goal, 0.1))
        np.testing.assert_array_equal(archive["frame_seq"], [0, 1, 2])
        np.testing.assert_array_equal(archive["raw_depth_m"][0], raw_depth)
    decoded = subprocess.run(
        [recorder.ffmpeg, "-loglevel", "error", "-i", result["video"], "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    assert len(decoded) == 3 * 1344 * 620 * 3
    assert result["frames"] == 3 and result["encoded_after_robot_stop"]


def test_recording_is_bounded_and_empty_run_has_no_video(tmp_path):
    recorder = RunRecording(tmp_path / "empty.json", np.zeros((72, 128, 4)), 0)
    assert recorder.finish() == {"frames": 0}
    frame = (0, 0, np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1)))
    for _ in range(recorder.limit):
        recorder.append(frame, torch.zeros((1, 72, 128, 4)), {})
    with pytest.raises(RuntimeError, match="frame budget"):
        recorder.append(frame, torch.zeros((1, 72, 128, 4)), {})


def test_pickup_extends_recording_without_unbounding_it(tmp_path):
    recorder = RunRecording(tmp_path / "combined.json", np.zeros((72, 128, 4)), 25)
    assert recorder.limit == 378
    recorder.allow_pickup()
    assert recorder.limit == 1278  # Policy + bounded closing, lift and final second.
    recorder.allow_pickup()
    assert recorder.limit == 1353


def test_unlimited_recording_past_old_caps_is_disk_backed(tmp_path):
    from grasp_planning.real_franka.frame_store import FrameStore, save_rgbd_archive

    r = RunRecording(tmp_path / "long.json", np.zeros((2, 3, 4), np.float32), None)
    assert r.limit is None and isinstance(r.frames, FrameStore)
    r.allow_pickup()
    assert r.limit is None
    for i in range(1500):
        r.append(
            (0, i, np.zeros((2, 3, 3), np.uint8), np.full((2, 3), i, np.float32)),
            torch.full((1, 2, 3, 4), float(i)),
            {"time_s": i / 15, "frame_seq": i},
        )
        if i % 16 == 0:
            r.frames.queue.join()
    r.frames.seal()
    assert len(r.frames) == 1500 and r.frames.queue.maxsize == 32
    assert r.frames.queue.empty() and not r.frames.worker.is_alive()
    save_rgbd_archive(tmp_path / "long.npz", r.frames, r.goal)
    with np.load(tmp_path / "long.npz") as data:
        assert data["live_rgbd"].shape == (1500, 2, 3, 4)
        np.testing.assert_array_equal(data["frame_seq"], np.arange(1500))
        assert data["raw_depth_m"][-1, 0, 0] == 1499
    r.frames.cleanup()
    assert not r.frames.path.exists()


def test_spool_failure_stops_capture_without_blocking(tmp_path):
    import queue

    from grasp_planning.real_franka.frame_store import FrameStore

    store = FrameStore.__new__(FrameStore)
    store.error = OSError("disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        store.append(None)
    store.error = None
    store.sealed = False
    store.queue = queue.Queue(maxsize=1)
    store.queue.put("occupied")
    with pytest.raises(RuntimeError, match="cannot keep up"):
        store.append("next frame")
