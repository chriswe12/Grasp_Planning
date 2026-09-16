#!/usr/bin/env python3
"""Encode Isaac inspection frames and create a local gallery (requires FFmpeg and Pillow)."""

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--recording-dir", type=Path, required=True)
parser.add_argument("--goal-images", type=Path, required=True)
parser.add_argument("--title", default="Franka scene inspection")
parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
args = parser.parse_args()
if not args.ffmpeg:
    try:
        import imageio_ffmpeg

        args.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        parser.error("Supply --ffmpeg or install imageio-ffmpeg")
out = args.recording_dir.resolve()
report = json.loads((out / "recording.json").read_text())
cards = []
for episode in report["episodes"]:
    name = episode["name"]
    frames = sorted((out / name).glob("*.png"))
    assert len(frames) == episode["frames"], f"Stale or missing frames: {name}"
    video = out / (name + ".mp4")
    subprocess.run(
        [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(report["fps"]),
            "-i",
            str(out / name / "%04d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    subprocess.run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video), "-f", "null", "-"], check=True)
    terminal = episode["terminal"]
    status = "Success" if terminal["success"] else ("Timeout" if terminal["timeout"] else "Terminated")
    title = "Zero-action hold" if episode["mode"] == "zero" else "Scripted approach"
    cards.append(
        f"<article><h3>{title} · {html.escape(episode['target_id'])}</h3>"
        f'<video controls playsinline preload="metadata" poster="{name}/0000.png" src="{name}.mp4"></video>'
        f"<p>{episode['frames'] / report['fps']:.1f} s · {status} · "
        f"Final position error {terminal['position_error_m'] * 1000:.1f} mm</p>"
        f'<a href="{name}.mp4" download>Download video</a></article>'
    )
    print(f"[VIDEO] {video.name}: {len(frames)} frames, decoded successfully")

goals = report["goal_views"]
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
sheet = Image.new("RGB", (1056, 48 + 232 * ((len(goals) + 2) // 3)), "#15202b")
draw = ImageDraw.Draw(sheet)
draw.text((16, 12), args.title + " | saved wrist-camera goals", font=font, fill="white")
for i, goal in enumerate(goals):
    target = goal["target_id"]
    x, y = 16 + (i % 3) * 346, 48 + (i // 3) * 232
    sheet.paste(Image.open(args.goal_images / (target + ".png")).convert("RGB").resize((330, 186)), (x, y))
    draw.text((x, y + 192), target, font=font, fill="white")
sheet.save(out / "saved_grasps.jpg", quality=94)
gallery = "".join(
    f'<a href="{g["image"]}"><img loading="lazy" src="{g["image"]}" alt="{html.escape(g["target_id"])}"></a>'
    for g in goals
)
variations = report.get("variations", [])
variation_gallery = ""
if variations:
    variation_gallery = '<h2>Per-episode appearance samples</h2><p>Same grasp pose with different live colors and lighting; canonical goal stays on the right.</p><div class="grid">'
    sweep = out / "appearance_sweep.mp4"
    subprocess.run(
        [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "0.5",
            "-i",
            str(out / "variation_%02d.png"),
            "-frames:v",
            str(len(variations) * 30),
            "-r",
            "15",
            "-c:v",
            "libx264",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(sweep),
        ],
        check=True,
    )
    subprocess.run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(sweep), "-f", "null", "-"], check=True)
    variation_gallery += '<article><h3>Appearance sweep · 2 seconds per sample</h3><video controls playsinline preload="metadata" poster="variation_00.png" src="appearance_sweep.mp4"></video></article>'
    variation_gallery += "".join(
        f'<a href="{v["image"]}"><img src="{v["image"]}" loading="lazy" alt="Appearance sample"></a>'
        for v in variations
    )
    variation_gallery += "</div>"
    collage = Image.new("RGB", (1280, 360 * ((len(variations) + 1) // 2)), "#15202b")
    for i, variation in enumerate(variations):
        collage.paste(Image.open(out / variation["image"]).resize((640, 360)), ((i % 2) * 640, (i // 2) * 360))
    collage.save(out / "appearance_variations.jpg", quality=92)
title = html.escape(args.title)
notes = " ".join(html.escape(n) for n in report["notes"])
(out / "index.html").write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{margin:0;background:#101923;color:#edf3f8;font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1440px;margin:auto;padding:28px}}h2{{margin-top:36px}}h3{{font-size:17px}}
p{{color:#bdcbd9}}a{{color:#86cfff}}img,video{{width:100%;border-radius:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:24px}}
article{{background:#1b2938;padding:16px;border-radius:12px}}</style><main>
<h1>{title}</h1><p>Actual Isaac GPU episodes with the Panda arm, Panda hand and project ZED Mini camera profile.</p>
<p>{notes}</p><h2>Episodes</h2><p>Full scene on the left; live wrist image and saved goal on the right. Playback uses simulation time.</p>
<div class="grid">{"".join(cards)}</div>{variation_gallery}<h2>Grasp target views</h2><div class="grid">{gallery}</div>
<h2>Saved wrist-camera goals</h2><img src="saved_grasps.jpg" alt="Rendered grasp goals">
<p><a href="recording.json">Recording metadata and measurements</a></p></main></html>""")
print(f"[GALLERY] {out / 'index.html'}")
