#!/usr/bin/env python3
"""Create a lower-resolution two-view-only actor catalog without changing source data."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp_planning.rl.franka_goal_variants import validate_variants, variant_digest
from grasp_planning.rl.zed_mini import profile_id, validate_zed_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    if a.source.resolve() == a.output.resolve() or a.output.exists():
        raise ValueError("Use a new output path; source/previous catalogs are immutable")
    torch.set_num_threads(4)
    with np.load(a.source, allow_pickle=False) as src:
        data = {k: src[k].copy() for k in src.files if k not in ("goal_rgbd", "goal_rgbd_variants")}
        contract = json.loads(data["contract_json"].item())
        profile = contract["camera_profile_data"]
        profile.update(observation_width=128, observation_height=72, render_width=256, render_height=144)
        validate_zed_profile(profile)
        contract["camera_profile"] = profile_id(profile)
        network = contract["training_recipe"]["agent"]["params"]["network"]
        network.update(image_width=128, image_height=72, visual_fusion="paired", use_policy_context=False)
        for key in ("goal_rgbd", "goal_rgbd_variants"):
            images = src[key]
            shape = images.shape
            flat = images.reshape(-1, *shape[-3:])
            out = np.empty((len(flat), 72, 128, 4), np.float16)
            for i in range(0, len(flat), 32):
                batch = torch.from_numpy(flat[i : i + 32].astype(np.float32)).permute(0, 3, 1, 2)
                out[i : i + 32] = (
                    F.interpolate(batch, size=(72, 128), mode="area").permute(0, 2, 3, 1).numpy().astype(np.float16)
                )
            data[key] = out.reshape(*shape[:-3], 72, 128, 4)
            del images, flat, out
            print("Resized", key, data[key].shape, flush=True)
    contract["goal_randomization"]["images_sha256"] = variant_digest(data["goal_rgbd_variants"])
    data["contract_json"] = np.asarray(json.dumps(contract, sort_keys=True))
    validate_variants(data, contract["goal_randomization"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(a.output)

    def digest(p):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(b)
        return h.hexdigest()

    report = dict(
        source=str(a.source),
        source_sha256=digest(a.source),
        output=str(a.output),
        sha256=digest(a.output),
        targets=len(data["target_ids"]),
        parts=len(set(data["part_keys"].tolist())),
        image_shape=[72, 128, 4],
        actor_inputs="live and goal RGB-D only; no engineered pair features or previous action",
        training_only="Pose/completion labels and centralized critic unchanged",
        resize="Area-filtered 384x216 references; RGB-D channels retain existing normalized packing",
        preserved="All target IDs, splits, source hashes, grasp poses and reset/placement arrays; mixed renderers, independent colors, soft lighting and pose curriculum",
    )
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
