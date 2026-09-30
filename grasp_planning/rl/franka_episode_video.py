"""Stream actual Isaac episode frames to H.264 without retaining frame folders."""

import math
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


class EpisodeVideo:
    def __init__(self, path, part, progress, delta, goal):
        self.path = path
        self.part = part
        self.progress = progress
        self.delta = delta
        self.goal = Image.fromarray(goal).resize((304, 171))
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        self.log = path.with_suffix(".ffmpeg.log").open("wb")
        self.process = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "960x540",
                "-r",
                "15",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "21",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.log,
        )
        self.frames = 0
        self.last = None

    def frame(self, world, wrist, step, position_mm, rotation_deg, completion):
        canvas = Image.new("RGB", (960, 540), "#101a25")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 8), f"Epoch 5000 | {self.part} | start {self.progress:g} | RANDOMIZED", font=self.font, fill="white"
        )
        dx, dy, yaw = self.delta
        draw.text(
            (12, 34),
            f"15 Hz / 1x | dx {dx * 100:+.1f} cm  dy {dy * 100:+.1f} cm  yaw {math.degrees(yaw):+.1f} deg",
            font=self.font,
            fill="#bcd5e8",
        )
        canvas.paste(Image.fromarray(world), (0, 68))
        draw.text((650, 74), "Live policy RGB", font=self.font, fill="white")
        canvas.paste(Image.fromarray(wrist).resize((304, 171)), (650, 96))
        draw.text((650, 272), "Unchanged canonical goal", font=self.font, fill="white")
        canvas.paste(self.goal, (650, 294))
        draw.text(
            (12, 483),
            f"t={step / 15:.2f}s | error {position_mm:.2f} mm / {rotation_deg:.2f} deg | p(done)={completion:.3f}",
            font=self.font,
            fill="white",
        )
        draw.text(
            (12, 510),
            "Open-finger alignment. Actual policy rollout; automatic resets excluded.",
            font=self.font,
            fill="#bcd5e8",
        )
        self.last = canvas
        self.process.stdin.write(canvas.tobytes())
        self.frames += 1

    def finish(self, outcome, terminal):
        card = self.last.copy()
        draw = ImageDraw.Draw(card)
        draw.rectangle((0, 0, 960, 66), fill="#101a25")
        draw.text(
            (12, 8),
            f"{outcome.upper()} | final error {terminal['position_error_m'] * 1000:.2f} mm / {math.degrees(terminal['rotation_error_rad']):.2f} deg",
            font=self.font,
            fill="white",
        )
        draw.text(
            (12, 34),
            "Last pre-action image frozen for 2 seconds; terminal metrics above.",
            font=self.font,
            fill="#bcd5e8",
        )
        for _ in range(30):
            self.process.stdin.write(card.tobytes())
        self.process.stdin.close()
        code = self.process.wait(timeout=30)
        self.log.close()
        if code:
            raise RuntimeError(f"Video encoding failed: {self.path}")
        self.last.save(self.path.with_suffix(".jpg"), quality=90)
        return dict(
            path=self.path.name,
            poster=self.path.with_suffix(".jpg").name,
            frames=self.frames + 30,
            fps=15,
            outcome=outcome,
            part=self.part,
            progress=self.progress,
        )
