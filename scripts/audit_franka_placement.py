#!/usr/bin/env python3
"""Audit canonical-image identity and all accepted independent-placement states."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp_planning.rl.franka_placement import (
    PLACEMENT_PROFILE,
    placement_transfer_allowed,
    transform_placement,
    validate_placement_poses,
)


def audit(catalog, source_catalog):
    with np.load(catalog) as data, np.load(source_catalog) as source:
        contract = json.loads(data["contract_json"].item())
        original = json.loads(source["contract_json"].item())
        assert placement_transfer_allowed(original, contract)
        index = {str(v): i for i, v in enumerate(source["target_ids"])}
        rows = np.array([index[str(v)] for v in data["target_ids"]])
        valid = data["pose_reset_valid"]
        t, b = np.where(valid)
        source_banks = data["placement_source_bank"][t, b]
        assert len(t) > 0 and (source_banks >= 0).all()
        assert source["pose_reset_valid"][rows[t], source_banks].all()
        assert np.array_equal(data["pose_reset_kind"][b], source["pose_reset_kind"][source_banks])
        assert np.allclose(data["pose_reset_progress"][b], source["pose_reset_progress"][source_banks])
        assert np.array_equal(data["goal_rgbd"], source["goal_rgbd"][rows]), "Canonical goals changed"
        assert np.array_equal(data["split"], source["split"][rows]), "Split identity changed"
        assert np.array_equal(data["object_poses"], source["object_poses"][rows]), "Canonical metadata changed"
        for key in ["placement_object_poses", "placement_goal_poses"]:
            validate_placement_poses(data[key], valid)
        obj = data["placement_object_poses"][t, b]
        goal = data["placement_goal_poses"][t, b]
        delta = data["placement_delta_xy_yaw"][t, b]
        anchor = torch.tensor(source["object_poses"][rows[t], :3])
        for key, actual in [("object_poses", obj), ("goal_poses", goal)]:
            expected = transform_placement(torch.tensor(source[key][rows[t]]), anchor, torch.tensor(delta)).numpy()
            assert np.allclose(actual, expected, atol=2e-6), f"Incorrect object-local transform: {key}"
        bounds = PLACEMENT_PROFILE["world_xy_range_m"]
        assert ((obj[:, :2] >= np.array(bounds)[:, 0] - 1e-6) & (obj[:, :2] <= np.array(bounds)[:, 1] + 1e-6)).all()
        assert (np.abs(delta[:, 2]) <= np.deg2rad(PLACEMENT_PROFILE["yaw_half_range_deg"]) + 1e-6).all()
        assert (data["pose_reset_contact_n"][valid] < 0.5).all()
        assert np.isfinite(data["pose_reset_joints"]).all()
        coverage = {
            str(p): int(
                (valid & (data["pose_reset_kind"][None, :] == 0) & np.isclose(data["pose_reset_progress"][None, :], p))
                .any(1)
                .sum()
            )
            for p in [0, 0.25, 0.5, 0.75, 0.94]
        }
        return dict(
            passed=True,
            targets=len(rows),
            parts=len(set(data["part_keys"])),
            accepted_states=len(t),
            source_targets=len(source["target_ids"]),
            goal_images_identical=True,
            canonical_goal_sha256=hashlib.sha256(data["goal_rgbd"].tobytes()).hexdigest(),
            world_xy_min=obj[:, :2].min(0).tolist(),
            world_xy_max=obj[:, :2].max(0).tolist(),
            yaw_delta_deg_percentiles=np.percentile(np.rad2deg(delta[:, 2]), [0, 5, 50, 95, 100]).tolist(),
            progress_target_coverage=coverage,
            max_contact_n=float(data["pose_reset_contact_n"][valid].max()),
            splits={str(s): int((data["split"] == s).sum()) for s in np.unique(data["split"])},
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("catalog", type=Path)
    p.add_argument("--source", type=Path, default=ROOT / "isaac_rl/data/franka_clutter_v5_fast_fxaa/catalog.npz")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = audit(args.catalog, args.source)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
