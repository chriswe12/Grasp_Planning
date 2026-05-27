"""Small video-writing helpers used by simulation execution scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OpenCvVideoWriter:
    """Write RGB frames to an MP4/AVI file using OpenCV."""

    def __init__(self, path: str | Path, *, fps: float, width: int, height: int) -> None:
        if fps <= 0.0:
            raise ValueError("Video FPS must be positive.")
        if width <= 0 or height <= 0:
            raise ValueError("Video dimensions must be positive.")
        self.path = Path(path)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.frame_count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("OpenCV (`cv2`) is required for video recording.") from exc
        self._cv2 = cv2
        suffix = self.path.suffix.lower()
        codec = "XVID" if suffix == ".avi" else "mp4v"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (self.width, self.height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for '{self.path}'.")

    def append_rgb(self, frame_rgb: np.ndarray) -> None:
        frame = np.asarray(frame_rgb)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Expected an RGB/RGBA frame, got shape {frame.shape}.")
        frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                frame = np.clip(frame, 0.0, 1.0) * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = self._cv2.resize(frame, (self.width, self.height), interpolation=self._cv2.INTER_AREA)
        frame_bgr = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
        self._writer.write(frame_bgr)
        self.frame_count += 1

    def close(self) -> None:
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.release()
            self._writer = None

    def __enter__(self) -> "OpenCvVideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
