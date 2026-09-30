"""Bounded capture of exact policy frames; all encoding happens after control closes."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .frame_store import FrameStore, save_rgbd_archive


def ffmpeg_binary():
    candidates = [os.environ.get("FRANKA_FFMPEG"), shutil.which("ffmpeg")]
    candidates += sorted(
        str(p)
        for p in (Path.home() / "miniconda3/lib").glob("python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux*")
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("Recording needs ffmpeg; set FRANKA_FFMPEG to an installed binary")


class RunRecording:
    def __init__(self, log_path, goal, max_duration_s):
        self.path = Path(log_path)
        self.goal = np.asarray(goal, dtype=np.float32).copy()
        self.limit = (
            None
            if max_duration_s is None or float(max_duration_s) > 30
            else int(np.ceil(float(max_duration_s) * 15)) + 3
        )
        self.frames = []
        self.ffmpeg = ffmpeg_binary()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.limit is None:
            # Keep real-time spooling off the external project drive. Final
            # artifacts are exported there only after control has stopped.
            self.frames = FrameStore(Path(tempfile.gettempdir()))

    def allow_pickup(self):
        # Reserve another bounded 60 seconds; storage is allocated per frame.
        if self.limit is not None:
            self.limit = min(1353, self.limit + 900)

    def append(self, frame, live, step):
        if self.limit is not None and len(self.frames) >= self.limit:
            raise RuntimeError("Recording frame budget exceeded")
        # Camera frames are immutable, separately allocated by ZedCamera. Keep
        # their RGB references; copy the small model input. No compression or I/O.
        self.frames.append((frame[2], live[0].detach().cpu().numpy().copy(), dict(step), frame[3]))

    def finish(self):
        if isinstance(self.frames, FrameStore):
            self.frames.seal()
        if not self.frames:
            if isinstance(self.frames, FrameStore):
                self.frames.cleanup()
            return dict(frames=0)
        archive = self.path.with_suffix(".rgbd.npz")
        video = self.path.with_suffix(".mp4")
        save_rgbd_archive(archive, self.frames, self.goal)
        partial = video.with_suffix(".partial.mp4")
        log = video.with_suffix(".encoding.log")
        command = [
            self.ffmpeg,
            "-y",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "1344x620",
            "-framerate",
            "15",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(partial),
        ]
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        with log.open("w") as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            try:
                maximum = 0.0
                for raw, live, step, _raw_depth in self.frames:
                    canvas = Image.new("RGB", (1344, 620), "#101820")
                    draw = ImageDraw.Draw(canvas)
                    for x, name, rgb in [
                        (0, "Real ZED left", raw),
                        (448, "Live policy RGB", live[..., :3]),
                        (896, "Selected goal RGB", self.goal[..., :3]),
                    ]:
                        if rgb.dtype != np.uint8:
                            rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
                        draw.text((x + 10, 7), name, font=font, fill="white")
                        canvas.paste(Image.fromarray(rgb).resize((448, 252)), (x, 32))
                    for x, name, depth in [
                        (0, "Live policy depth", live[..., 3]),
                        (448, "Goal depth", self.goal[..., 3]),
                    ]:
                        gray = (255 * (1 - depth.clip(0, 1))).astype(np.uint8)
                        draw.text((x + 10, 292), name + " (white=near)", font=font, fill="white")
                        canvas.paste(Image.fromarray(gray).convert("RGB").resize((448, 252)), (x, 318))
                    p = step["completion_probability"]
                    if p is not None:
                        maximum = max(maximum, p)
                    feedback = step.get("measured_feedback") or {}
                    twist = np.asarray(step["base_tcp_twist"])
                    lines = [
                        f"t = {step['time_s']:.3f} s | frame {step['frame_seq']}",
                        f"p(done) = {p:.6g}" if p is not None else "p(done) = n/a (manual pickup)",
                        f"Maximum p(done) = {maximum:.6g}",
                        f"Stable: {step.get('completion_stable')} | streak: {step.get('completion_streak', 0)}",
                        f"Valid depth: {step['valid_depth_fraction']:.1%}",
                        f"TCP displacement: {feedback.get('displacement_from_start_m', 0) * 1000:.1f} mm",
                        f"Command speed: {np.linalg.norm(twist[:3]) * 1000:.1f} mm/s",
                        f"Command rotation: {np.linalg.norm(twist[3:]):.3f} rad/s",
                    ]
                    for j, line in enumerate(lines):
                        draw.text((906, 320 + j * 29), line, font=font, fill="white")
                    draw.text(
                        (10, 590),
                        self.path.stem + " | " + step.get("phase", "policy") + " | 15 Hz | grasp success unverified",
                        font=font,
                        fill="#a6bdcc",
                    )
                    process.stdin.write(canvas.tobytes())
                process.stdin.close()
                if process.wait(timeout=60):
                    raise RuntimeError(f"Video encoding failed; see {log}")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
        partial.replace(video)
        if isinstance(self.frames, FrameStore):
            self.frames.cleanup()
        return dict(frames=len(self.frames), video=str(video), rgbd=str(archive), fps=15, encoded_after_robot_stop=True)
