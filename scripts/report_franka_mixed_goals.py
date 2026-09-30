#!/usr/bin/env python3
"""Inspection gallery for actual mixed-renderer references and live Isaac resets."""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("directory", type=Path)
a = p.parse_args()
r = json.loads((a.directory / "renders.json").read_text())
h = [
    """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Franka · mixed goal inspection</title><style>
:root{color-scheme:dark}body{background:#0c1420;color:#e6edf6;font:16px system-ui;max-width:1500px;margin:auto;padding:30px}h1{font-size:38px}p{line-height:1.6;color:#bbcede}a{color:#8dcaff}section{margin:36px 0;padding:22px;background:#121e2d;border:1px solid #2c3c4c;border-radius:16px}.scene{display:grid;grid-template-columns:2fr 1fr;gap:20px;align-items:start}.goals{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}img{width:100%;border-radius:10px}figure{margin:0}figcaption{margin:8px 0;color:#bacfe3}button{padding:10px 18px;border:1px solid #4b627a;background:#24374d;color:white;border-radius:9px;cursor:pointer}.depth{display:none}body.show-depth .rgb{display:none}body.show-depth .depth{display:block}.tag{color:#83e3c3;font-size:14px}small{color:#aebfd0}@media(max-width:850px){.scene{grid-template-columns:1fr}.goals{grid-template-columns:1fr 1fr}}select{padding:10px;background:#22354a;color:white;border-radius:8px}</style>
<h1>Same grasp. Different renderers.</h1><p>Preview for the next training run. Actual Isaac RTX and MuJoCo OpenGL renders, using the same object, robot visual meshes, grasp, camera and aperture. Live object placement and color are independent of the goal. <b>No training has started.</b></p>
<p>Proposed sampling: 50% Isaac / 50% MuJoCo. Refreshed blue Isaac goals make up 20% of all episodes; the other 80% use independently colored references. Goal appearance stays fixed for the episode. MuJoCo simplifies lab textures and thin decals while retaining material base colors. The live Isaac table/floor setup is unchanged.</p>
<button onclick="document.body.classList.toggle('show-depth')">Toggle RGB / depth</button> <select id="filter" onchange="document.querySelectorAll('section[data-part]').forEach(x=>x.hidden=this.value!=='all'&&x.dataset.part!==this.value)"><option value="all">All parts</option>"""
]
for part in sorted({x["part"] for x in r["records"]}):
    h.append(f"<option>{html.escape(part)}</option>")
h.append(
    "</select><p><small>Depth uses identical limits in every panel: white = 0.1 m; black = 1 m / invalid. Renderer brightness differences are appearance variation, not a different grasp. Click any image for native pixels.</small></p>"
)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
sheets = []
for row in r["records"]:
    stem = row["stem"]
    part = html.escape(row["part"])
    delta = row["placement_delta"]
    h.append(
        f'<section data-part="{part}"><div class="tag">{html.escape(row["target_id"])}</div><h2>{part}</h2><p>Live: {row["live_color"]} · independent placement ΔX {delta[0] * 100:+.1f} cm, ΔY {delta[1] * 100:+.1f} cm, yaw {delta[2] * 180 / 3.14159265:+.1f}°</p><div class="scene"><figure><a href="{stem}_scene.png"><img loading="lazy" src="{stem}_scene.png"></a><figcaption>Live Isaac scene · accepted middle-distance reset</figcaption></figure><figure>'
    )
    for cls, suffix in [("rgb", "live"), ("depth", "live_depth")]:
        h.append(f'<a class="{cls}" href="{stem}_{suffix}.png"><img loading="lazy" src="{stem}_{suffix}.png"></a>')
    h.append(
        '<figcaption>Clean live wrist render · 128 × 72; sensor augmentation follows in training</figcaption></figure></div><h3>Alternative goals for this exact grasp</h3><div class="goals">'
    )
    for v in range(4):
        h.append("<figure>")
        for cls, suffix in [("rgb", f"goal_{v}"), ("depth", f"goal_{v}_depth")]:
            h.append(f'<a class="{cls}" href="{stem}_{suffix}.png"><img loading="lazy" src="{stem}_{suffix}.png"></a>')
        h.append(f"<figcaption>{'Isaac RTX' if v < 2 else 'MuJoCo'} · {row['goal_colors'][v]}</figcaption></figure>")
    h.append(f'</div><p><a href="{stem}_canonical.png">Refreshed blue reference</a></p></section>')
    if len(sheets) < 3:
        canvas = Image.new("RGB", (1152, 700), "#0c1420")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), row["part"] + " | live scene and independent goals", font=font, fill="white")
        canvas.paste(Image.open(a.directory / f"{stem}_scene.png").resize((640, 400)), (0, 45))
        canvas.paste(Image.open(a.directory / f"{stem}_live.png").resize((480, 270)), (660, 82))
        draw.text((660, 50), "Live wrist | " + row["live_color"], font=font, fill="white")
        for v in range(4):
            canvas.paste(Image.open(a.directory / f"{stem}_goal_{v}.png").resize((276, 155)), (v * 288, 495))
            draw.text(
                (v * 288 + 5, 462),
                ("Isaac" if v < 2 else "MuJoCo") + " | " + row["goal_colors"][v],
                font=font,
                fill="white",
            )
        canvas.save(a.directory / f"{stem}_comparison.jpg")
        sheets.append(canvas)
h.append(
    '<section><h2>Depth preprocessing correction</h2><p>The multi-environment GPU packing path had a singleton-channel stride bug: separate 0.2 / 0.5 / 0.8 m inputs could become 0.2 / 0.2 / 0.2 m. The new catalog uses corrected packing and explicit optical-Z conversion. This was isolated in our preprocessing; it is not evidence of a ZED depth bias or an Isaac sensor defect.</p><p><a href="../depth_layout_regression.json">GPU before/after regression</a></p></section>'
)
audit_path = a.directory.parent / "deployment/deployment_audit.json"
if audit_path.exists():
    audit = json.loads(audit_path.read_text())
    h.append(
        f'<section><h2>What your real tests recorded</h2><p>{len(audit["runs"])} sessions contain timing and depth-validity summaries, with motion feedback in some runs. Raw RGB-D sequences were not saved. One older snapshot is available below. These data help diagnose coverage and tracking, but cannot establish absolute depth error or temporal noise without known geometry and repeated frames.</p><div class="goals"><figure><img src="../deployment/real_rgb.png"><figcaption>Saved rectified ZED RGB</figcaption></figure><figure><img src="../deployment/real_depth.png"><figcaption>Saved optical-Z depth</figcaption></figure><figure><img src="../deployment/real_depth_validity.png"><figcaption>Green: valid policy range · red: missing · yellow: out of range</figcaption></figure></div></section>'
    )
h.append(
    '<p><a href="renders.json">Renderer provenance and geometry checks</a> · <a href="../deployment/deployment_audit.json">Deployment-record audit</a></p></html>'
)
(a.directory / "index.html").write_text("\n".join(h))
if sheets:
    sheets[0].save(a.directory / "comparison.jpg")
print(a.directory / "index.html")
