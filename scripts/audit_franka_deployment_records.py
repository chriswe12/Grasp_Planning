#!/usr/bin/env python3
"""Summarize recorded deployment evidence without accessing camera or robot."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--runs", type=Path, default=Path("artifacts/franka_real_runs"))
p.add_argument("--snapshot", type=Path, default=Path("artifacts/franka_real_20260918"))
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
report = {
    "runs": [],
    "limitations": [
        "No RGB-D sequences in run JSON logs: cannot estimate temporal depth noise or persistence.",
        "Depth error needs known geometry/pose or an independent reference; valid-pixel fraction is not accuracy.",
        "Commanded action is not measured motion; controller versions and speed caps differ across sessions.",
    ],
}
for path in sorted(a.runs.glob("*.json")):
    d = json.loads(path.read_text())
    s = d.get("steps", [])
    row = dict(file=str(path), execute=d.get("execute"), outcome=d.get("outcome"), steps=len(s))
    if s:
        row["valid_depth_fraction_percentiles"] = np.percentile(
            [v["valid_depth_fraction"] for v in s], [0, 50, 100]
        ).tolist()
        row["inference_cycle_ms_percentiles"] = np.percentile(
            [v["inference_cycle_s"] * 1000 for v in s], [50, 95, 100]
        ).tolist()
        if len(s) > 1:
            row["loop_interval_ms_percentiles"] = np.percentile(
                np.diff([v["time_s"] for v in s]) * 1000, [50, 95, 100]
            ).tolist()
        fs = [v for v in s if v.get("measured_feedback")]
        if len(fs) > 1:
            t = np.array([v["time_s"] for v in fs])
            poses = np.array([v["measured_feedback"]["tcp_position_m"] for v in fs])
            twist = np.array([v["base_tcp_twist"][:3] for v in fs])
            delta = poses[-1] - poses[0]
            integral = (twist[:-1] * np.diff(t)[:, None]).sum(0)
            row.update(
                measured_net_displacement_mm=float(np.linalg.norm(delta) * 1000),
                command_integral_mm=(integral * 1000).tolist(),
                measured_delta_mm=(delta * 1000).tolist(),
            )
            ages = [
                v["measured_feedback"]["tcp_transform_age_s"] * 1000
                for v in fs
                if "tcp_transform_age_s" in v["measured_feedback"]
            ]
            row["tcp_age_ms_p95"] = float(np.percentile(ages, 95)) if ages else None
    report["runs"].append(row)
    if d.get("camera"):
        report["camera"] = d["camera"]
profile = json.loads(Path("configs/franka_zed_mini.json").read_text())
report["training_camera"] = {
    k: profile[k] for k in ["fx", "fy", "cx", "cy", "source_width", "source_height", "calibration_status"]
}
report["intrinsics_note"] = (
    "Deployment live_to_training reprojects rectified rays into training intrinsics; focal-length mismatch alone is not evidence of a current projection bug."
)
path = a.snapshot / "real_depth.npy"
if path.exists():
    depth = np.load(path)
    valid = np.isfinite(depth) & (depth > 0)
    policy_valid = valid & (depth >= profile["depth_min_m"]) & (depth < profile["depth_max_m"])
    report["snapshot"] = dict(
        file=str(path),
        shape=list(depth.shape),
        finite_positive_fraction=float(valid.mean()),
        in_policy_depth_range_fraction=float(policy_valid.mean()),
        valid_depth_m_percentiles=np.percentile(depth[valid], [5, 50, 95]).tolist(),
        note="Raw full-resolution frame; run coverage uses cropped, resized policy observations and is not directly comparable.",
    )
    dep = np.where(policy_valid, depth, profile["depth_max_m"])
    dep = (1 - np.clip((dep - 0.1) / 0.9, 0, 1)) * 255
    Image.fromarray(dep.astype("uint8")).save(a.output / "real_depth.png")
    mask = np.zeros((*depth.shape, 3), np.uint8)
    mask[policy_valid] = [52, 170, 133]
    mask[~valid] = [225, 79, 94]
    mask[valid & ~policy_valid] = [212, 161, 54]
    Image.fromarray(mask).save(a.output / "real_depth_validity.png")
    Image.open(a.snapshot / "real_left.png").save(a.output / "real_rgb.png")
(a.output / "deployment_audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
