#!/usr/bin/env python3
"""Inspect saved ZED calibration/depth and compare regenerated optics; no hardware access."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp_planning.real_franka.recording import ffmpeg_binary
from grasp_planning.rl.zed_mini import load_zed_profile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--recording", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--old-gallery", type=Path, required=True)
p.add_argument("--new-gallery", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
record = json.loads(a.recording.read_text())
old = load_zed_profile()
new = load_zed_profile(ROOT / "configs/franka_zed_mini_sn13829658.json")


def fov(c):
    return [
        math.degrees(math.atan(c[k] / c[f]) + math.atan((c[s] - c[k]) / c[f]))
        for k, f, s in [("cx", "fx", "source_width"), ("cy", "fy", "source_height")]
    ]


with np.load(a.recording.with_suffix(".rgbd.npz"), allow_pickle=False) as z:
    times = z["time_s"]
    depth = z["raw_depth_m"]
    finite = np.isfinite(depth) & (depth > 0)
    within = finite & (depth >= new["depth_min_m"]) & (depth < new["depth_max_m"])
    fractions = within.mean(axis=(1, 2))
    report = dict(
        recording=str(a.recording),
        frames=len(times),
        recorded_duration_s=float(times[-1]),
        camera=record["camera"],
        old_fov_deg=fov(old),
        measured_fov_deg=fov(new),
        raw_depth_in_policy_range_fraction=dict(
            min=float(fractions.min()), median=float(np.median(fractions)), max=float(fractions.max())
        ),
        raw_depth_finite_positive_fraction_median=float(np.median(finite.mean(axis=(1, 2)))),
        outcome=record["outcome"],
        absolute_depth_bias="not identifiable without measured reference geometry",
        video_rgb="decoded lossy video panel; exact preprocessed old RGB-D and raw optical-Z depth are in the NPZ",
        training_goal_comparison="same target and appearance seed; old/new optics; real pose is NOT matched to simulation",
    )
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    for i in [0, len(times) // 2, len(times) - 1]:
        frame_path = a.output / f"real_{i:03d}.png"
        subprocess.run(
            [
                ffmpeg_binary(),
                "-y",
                "-loglevel",
                "error",
                "-ss",
                str(i / 15),
                "-i",
                str(a.recording.with_suffix(".mp4")),
                "-frames:v",
                "1",
                str(frame_path),
            ],
            check=True,
        )
    valid_map = (within.mean(axis=0) * 255).astype("uint8")
    Image.fromarray(valid_map).save(a.output / "depth_coverage.png")

old_records = json.loads((a.old_gallery / "renders.json").read_text())["records"]
new_records = json.loads((a.new_gallery / "renders.json").read_text())["records"]
row = new_records[0]
match = next(x for x in old_records if x["target_id"] == row["target_id"])
canvas = Image.new("RGB", (1344, 650), "#101820")
draw = ImageDraw.Draw(canvas)
frame = Image.open(a.output / "real_000.png")
for x, label, img in [
    (0, "Real ZED view: 89.6 x 58.1 deg", frame.crop((0, 32, 448, 284))),
    (448, "Old policy input: narrower crop", frame.crop((448, 32, 896, 284))),
    (896, "Real depth coverage across 169 frames", Image.open(a.output / "depth_coverage.png").convert("RGB")),
]:
    draw.text((x + 8, 8), label, font=font, fill="white")
    canvas.paste(img.resize((448, 252)), (x, 38))
for x, label, path in [
    (0, "Old Isaac reference: 76.2 x 47.4 deg", a.old_gallery / (match["stem"] + "_canonical.png")),
    (448, "New Isaac reference: measured FOV", a.new_gallery / (row["stem"] + "_canonical.png")),
    (896, "New MuJoCo: same measured FOV", a.new_gallery / (row["stem"] + "_goal_2.png")),
]:
    draw.text((x + 8, 315), label, font=font, fill="white")
    canvas.paste(Image.open(path).resize((448, 252)), (x, 345))
draw.text(
    (8, 616),
    "Bottom row: same catalog grasp. Real scene above is not a pose-matched reconstruction.",
    font=font,
    fill="#a6bdcc",
)
canvas.save(a.output / "comparison.jpg")
(a.output / "recording_audit.json").write_text(json.dumps(report, indent=2) + "\n")
video = a.output / "real_run.mp4"
if not video.exists():
    video.symlink_to(a.recording.with_suffix(".mp4").resolve())
(
    a.output / "index.html"
).write_text("""<!doctype html><meta charset="utf-8"><title>ZED Mini calibration correction</title>
<style>body{color:#e7eef5;background:#101820;max-width:1440px;margin:35px auto;font:18px system-ui;padding:0 22px}p{line-height:1.6;color:#bbcbd7}img,video{width:100%;border-radius:12px}a{color:#87d6f1}section{padding:22px;background:#172432;border-radius:16px;margin:24px 0}</style>
<h1>The measured ZED Mini view</h1><p>Recorded serial 13829658: <b>89.6° horizontal × 58.1° vertical</b>. Previous training: 76.2° × 47.4°. Both new goal renderers and the live Isaac camera use the measured rectified intrinsics and principal point. Mount and TCP settings are retained.</p>
<img src="comparison.jpg"><p>Real images and simulation are not the same pose. Bottom-row Isaac references use the same grasp and appearance seed, isolating the projection change; the MuJoCo example also varies color.</p>
<p><a href="../preview/index.html">Inspect new mixed-renderer goals and live Isaac scenes</a> · <a href="recording_audit.json">Recorded calibration and depth audit</a></p>
<section><h2>Your recorded deployment</h2><video src="real_run.mp4" controls preload="metadata"></video><p>This is the previous policy and camera preprocessing. Its stop was the recorded external-torque limit, not a success declaration. No new hardware execution was performed.</p></section>
<p>Raw depth reveals coverage and structured holes. Absolute depth bias requires measured reference geometry; it cannot be inferred from this video alone.</p>""")
print(json.dumps(report, indent=2))
