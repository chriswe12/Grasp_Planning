#!/usr/bin/env python3
"""Make an inspectable paired-placement report from the frozen-policy test."""

import argparse
import collections
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
args = p.parse_args()
root = args.directory
result = json.loads((root / "results.json").read_text())
if result.get("controller", "frozen_policy") != "frozen_policy":
    raise ValueError("The privileged controller check is not a policy benchmark; keep its results separate")
rows = result["records"]


def summarize(rs):
    return dict(
        episodes=len(rs),
        outcomes=dict(collections.Counter(r["outcome"] for r in rs)),
        success_rate=sum(r["outcome"] == "success" for r in rs) / len(rs),
        ever_geometric_ready_rate=sum(r["ever_geometric_ready"] for r in rs) / len(rs),
        position_mm_percentiles=np.percentile(
            [r["terminal"]["position_error_m"] * 1000 for r in rs], [25, 50, 75, 90]
        ).tolist(),
        rotation_deg_percentiles=np.percentile(
            [np.rad2deg(r["terminal"]["rotation_error_rad"]) for r in rs], [25, 50, 75, 90]
        ).tolist(),
        median_duration_s=float(np.median([r["duration_s"] for r in rs])),
    )


summary = {c: summarize([r for r in rows if r["condition"] == c]) for c in ["canonical", "randomized"]}
by_key = {}
for r in rows:
    key = (r["seed"], r["progress"], r["target_id"], r["reset_bank"])
    by_key.setdefault(key, {})[r["condition"]] = r
assert all(set(pair) == {"canonical", "randomized"} for pair in by_key.values()), "Incomplete paired evaluation"
paired_by_part = collections.defaultdict(list)
for pair in by_key.values():
    a, b = pair["canonical"], pair["randomized"]
    paired_by_part[a["part_key"]].append(int(b["outcome"] == "success") - int(a["outcome"] == "success"))
parts = sorted(paired_by_part)
rng = np.random.default_rng(20260921)
means = np.array([np.mean(paired_by_part[p]) for p in parts])
bootstrap = np.mean(rng.choice(means, size=(10000, len(parts)), replace=True), axis=1)
summary["paired"] = dict(
    parts=len(parts),
    pairs=len(by_key),
    success_difference_pp=float(means.mean() * 100),
    part_bootstrap_95ci_pp=(np.percentile(bootstrap, [2.5, 97.5]) * 100).tolist(),
    lost_success=sum(
        p["canonical"]["outcome"] == "success" and p["randomized"]["outcome"] != "success" for p in by_key.values()
    ),
    gained_success=sum(
        p["canonical"]["outcome"] != "success" and p["randomized"]["outcome"] == "success" for p in by_key.values()
    ),
)
summary["pairing_error"] = {
    field: np.percentile(
        [abs(pair["canonical"][field] - pair["randomized"][field]) for pair in by_key.values()], [50, 95, 100]
    ).tolist()
    for field in ["initial_position_mm", "initial_rotation_deg"]
}
summary["randomized_yaw_bins"] = {}
for low, high in [(0, 30), (30, 60), (60, 90.001)]:
    cohort = [
        r for r in rows if r["condition"] == "randomized" and low <= abs(np.rad2deg(r["placement_delta"][2])) < high
    ]
    if cohort:
        summary["randomized_yaw_bins"][f"{low}-{min(high, 90):g} deg"] = summarize(cohort)
summary["by_progress"] = {
    str(progress): {
        condition: summarize([r for r in rows if r["condition"] == condition and r["progress"] == progress])
        for condition in ["canonical", "randomized"]
    }
    for progress in sorted(set(r["progress"] for r in rows))
}
summary["by_part"] = {
    part: {
        condition: summarize([r for r in rows if r["condition"] == condition and r["part_key"] == part])
        for condition in ["canonical", "randomized"]
    }
    for part in parts
}
summary["by_split"] = {
    split: {
        condition: summarize([r for r in rows if r["condition"] == condition and r["split"] == split])
        for condition in ["canonical", "randomized"]
    }
    for split in sorted({r["split"] for r in rows})
}
for split, detail in summary["by_split"].items():
    grouped_differences = collections.defaultdict(list)
    for pair in by_key.values():
        a, b = pair["canonical"], pair["randomized"]
        if a["split"] == split:
            grouped_differences[a["part_key"]].append(int(b["outcome"] == "success") - int(a["outcome"] == "success"))
    values = np.array([np.mean(v) for v in grouped_differences.values()])
    samples = np.mean(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
    detail["paired"] = dict(
        parts=len(values),
        success_difference_pp=float(values.mean() * 100),
        part_bootstrap_95ci_pp=(np.percentile(samples, [2.5, 97.5]) * 100).tolist() if len(values) > 1 else None,
    )
(root / "summary.json").write_text(json.dumps(summary, indent=2))

header = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Franka · independent placement test</title><style>body{font:16px system-ui;background:#0c1420;color:#e6edf6;max-width:1500px;margin:auto;padding:32px}h1{font-size:34px}p{line-height:1.6;max-width:1000px;color:#bbcbe0}table{border-collapse:collapse;margin:24px 0}td,th{padding:12px 22px;text-align:left;border-bottom:1px solid #344155}section{margin:42px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%;border-radius:10px}a{color:#8ac7ff}small{color:#b8c7d9}button{background:#243650;color:white;border:0;padding:10px 18px;border-radius:7px;margin:4px;cursor:pointer}.muted{color:#a7b7ce}@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>"""
body = [
    header,
    "<h1>Same goal. Different object placement.</h1>",
    f"<p>Frozen checkpoint: <b>{html.escape(Path(result['checkpoint']).name)}</b>. Original placement is compared with independently sampled object X/Y/yaw. Robot base and table stay fixed. Both conditions use identical canonical goal pixels, source perturbations, and appearance seeds. The selected grasps/orientations remain known; true object location is used only for simulation reset and scoring.</p>",
    f"<p><b>{len(by_key)} paired trials across {len(parts)} parts.</b> Split: {html.escape(result['split'])}. This is alignment with open fingers, not a lift test. Results are conditional on placements that pass IK/contact checks. A new randomization setting does not retrain this checkpoint.</p>",
    "<table><tr><th>Condition</th><th>Success</th><th>Median final error</th><th>Failures</th></tr>",
]
for condition in ["canonical", "randomized"]:
    s = summary[condition]
    body.append(
        f"<tr><td>{condition}</td><td>{s['outcomes'].get('success', 0)}/{s['episodes']} ({s['success_rate']:.1%})</td><td>{s['position_mm_percentiles'][1]:.2f} mm / {s['rotation_deg_percentiles'][1]:.2f}°</td><td>{html.escape(str({k: v for k, v in s['outcomes'].items() if k != 'success'}))}</td></tr>"
    )
body.append("</table>")
if (root / "coverage.json").is_file():
    coverage = json.loads((root / "coverage.json").read_text())
    body.append(
        f'<p>New bank: {coverage["valid_states"]:,} accepted reset states; {coverage["retained"]:,}/{coverage["total"]:,} targets retained. Excluded targets are listed in the <a href="coverage.json">coverage report</a>. <a href="catalog_audit.json">Canonical-image / pose audit</a>.</p>'
    )
if (root / "oracle_matched.json").is_file():
    oracle = json.loads((root / "oracle_matched.json").read_text())
    assert oracle["controller"] == "privileged_oracle"
    check = [r for r in oracle["records"] if r["condition"] == "randomized"]
    body.append(
        f'<p><b>Controllability check:</b> a separate ground-truth Cartesian controller succeeded on {sum(r["outcome"] == "success" for r in check)}/{len(check)} exact randomized near-start cases from this test. This is a simulator sanity check, not policy performance. <a href="oracle_matched.json">Raw check</a>.</p>'
    )
if len(summary["by_split"]) > 1:
    body.append(
        "<h2>Separate evaluation splits</h2><table><tr><th>Split</th><th>Original placement</th><th>Randomized placement</th><th>Parts</th></tr>"
    )
    for split, detail in summary["by_split"].items():
        a, b = detail["canonical"], detail["randomized"]
        body.append(
            f"<tr><td>{html.escape(split)}</td><td>{a['outcomes'].get('success', 0)}/{a['episodes']} ({a['success_rate']:.1%})</td><td>{b['outcomes'].get('success', 0)}/{b['episodes']} ({b['success_rate']:.1%})</td><td>{detail['paired']['parts']}</td></tr>"
        )
    body.append(
        "</table><p>The supplemental plumbers-block validation examples are reported separately from held-out test results.</p>"
    )
pair = summary["paired"]
body.append(
    f"<p>Part-averaged success change: <b>{pair['success_difference_pp']:+.1f} percentage points</b>; paired part-bootstrap 95% interval: {pair['part_bootstrap_95ci_pp'][0]:+.1f} to {pair['part_bootstrap_95ci_pp'][1]:+.1f}. Lost successes: {pair['lost_success']}; gained successes: {pair['gained_success']}.</p>"
)
video_results = root / "videos/results.json"
if video_results.is_file():
    clips = json.loads(video_results.read_text()).get("videos", [])
    if clips:
        body.insert(2, '<p><a href="#videos">Watch randomized episodes ↓</a></p>')
        body.append(
            f'<section id="videos"><h2>{len(clips)} randomized episode videos</h2><p>Actual epoch-5000 policy rollouts, recorded six at a time. Playback is 15 fps at real simulated speed. Each video shows the world, the live RGB actually passed to the policy, and the unchanged goal. The final two seconds freeze the last pre-action view and show terminal metrics. Automatic resets are excluded.</p><div class="pair">'
        )
        for clip in sorted(clips, key=lambda c: (-c["progress"], c["part"] != "plumbers_block__part_3", c["part"])):
            src = html.escape("videos/" + clip["path"])
            poster = html.escape("videos/" + clip["poster"])
            title = html.escape(f"{clip['part']} · start {clip['progress']:g} · {clip['outcome']}")
            body.append(
                f'<article><h3>{title}</h3><video controls playsinline preload="none" poster="{poster}" style="width:100%;border-radius:10px" aria-label="{title}"><source src="{src}" type="video/mp4"></video><p><a href="{src}">Open video</a></p></article>'
            )
        body.append("</div></section>")
body.append(
    "<p>Each image below is a real Isaac render at reset, before the policy acts. Left: original placement. Right: randomized placement. Within each image: world view, live wrist, and unchanged goal. Click to inspect at full resolution.</p>"
)
grouped = {}
for item in result["images"]:
    grouped.setdefault((item["part"], item["progress"]), {})[item["condition"]] = item["path"]
for (part, progress), images in grouped.items():
    body.append(f'<section><h2>{html.escape(part)} · start {progress:g}</h2><div class="pair">')
    for condition in ["canonical", "randomized"]:
        name = html.escape(images[condition])
        sample = next(
            r for r in rows if r["part_key"] == part and r["progress"] == progress and r["condition"] == condition
        )
        dx, dy, yaw = sample["placement_delta"]
        body.append(
            f'<div><p>{condition} · ΔX {dx * 100:+.1f} cm, ΔY {dy * 100:+.1f} cm, yaw {np.rad2deg(yaw):+.1f}°</p><a href="{name}"><img loading="lazy" src="{name}" alt="{condition} scene"></a></div>'
        )
    body.append("</div></section>")
body.append(
    '<p><a href="summary.json">Summary / per-part outcomes</a> · <a href="results.json">Full paired traces and checkpoint hash</a></p></html>'
)
(root / "index.html").write_text("\n".join(body))
selected = [v for (part, progress), v in grouped.items() if progress == 0.94][:3]
if selected:
    sheet = Image.new("RGB", (1152, 320 * len(selected)), "#101a25")
    for row, v in enumerate(selected):
        for col, condition in enumerate(["canonical", "randomized"]):
            sheet.paste(Image.open(root / v[condition]).resize((576, 320)), (col * 576, row * 320))
    sheet.save(root / "comparison.png")
print(json.dumps({k: v for k, v in summary.items() if k in ["canonical", "randomized", "paired"]}, indent=2))
