#!/usr/bin/env python3
"""Audit multipart geometry, source identity, split isolation and physical labels."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit(catalog, root=ROOT):
    with np.load(catalog, allow_pickle=False) as source:
        data = {key: source[key].copy() for key in source.files}
    count = len(data["target_ids"])
    assert count >= 3 and len(set(data["target_ids"])) == count, "Missing or duplicate targets"
    contract = json.loads(str(data["contract_json"].item()))
    assets = contract["object_assets"]
    for key in (
        "joint_paths",
        "goal_rgbd",
        "goal_poses",
        "object_poses",
        "open_widths",
        "jaw_widths",
        "target_part_indices",
        "part_keys",
        "split",
        "source_grasp_ids",
        "orientation_ids",
        "validated",
        "lab_approach_validated",
        "lift_validated",
    ):
        assert len(data[key]) == count, f"Target count differs: {key}"
        if data[key].dtype.kind in "fiu":
            assert np.isfinite(data[key]).all(), f"Nonfinite values: {key}"
    for key in ("validated", "lab_approach_validated", "lift_validated"):
        assert data[key].all(), f"Unvalidated targets: {key}"
    if "lift_validation_json" in data:
        assert len(data["lift_clearance_validated"]) == count and data["lift_clearance_validated"].all()
    assert data["joint_paths"].shape[2] == 7
    assert data["goal_poses"].shape == data["object_poses"].shape == (count, 7)
    for key in ("goal_poses", "object_poses"):
        assert np.allclose(np.linalg.norm(data[key][:, 3:], axis=1), 1.0, atol=1e-4), key
    assert (data["open_widths"] <= 0.080001).all()
    assert (data["open_widths"] >= data["jaw_widths"] + 0.0079).all()
    assert data["target_part_indices"].dtype.kind in "iu"
    assert data["target_part_indices"].min() >= 0 and data["target_part_indices"].max() < len(assets)
    assert all(assets[i]["part_key"] == key for i, key in zip(data["target_part_indices"], data["part_keys"]))

    def verified_file(relative, expected):
        path = (root / str(relative)).resolve()
        assert path.is_relative_to(root.resolve()), f"Nonportable source: {relative}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Changed asset/source: {relative}"

    for item in assets:
        verified_file(item["object_usd"], item["object_sha256"])
    sources = list(zip(data["source_bundle_paths"], data["source_bundle_sha256"], strict=True))
    for relative, expected in sources:
        verified_file(relative, expected)
    assert set(data["split"]) == {"train", "validation", "test"}
    groups = defaultdict(set)
    for part, grasp, split in zip(data["part_keys"], data["source_grasp_ids"], data["split"]):
        groups[(str(part), str(grasp))].add(str(split))
        if part.startswith("gamepad__"):
            assert split == "test", "Held-out assembly leaked into training"
        elif int(hashlib.sha256(str(part).encode()).hexdigest()[:8], 16) % 5 == 0:
            assert split == "validation", "Held-out part leaked into training"
    assert all(len(splits) == 1 for splits in groups.values()), "A source grasp crosses splits"
    split_report = {
        split: dict(
            targets=int((data["split"] == split).sum()),
            parts=sorted(set(data["part_keys"][data["split"] == split].tolist())),
        )
        for split in ("train", "validation", "test")
    }
    expected = {f"{p.parent.name}__part_{p.stem}" for p in (root / "assets/obj/fabrica").glob("*/*.obj")}
    observed = set(data["part_keys"].tolist())
    return dict(
        passed=True,
        catalog=str(catalog),
        catalog_sha256=hashlib.sha256(Path(catalog).read_bytes()).hexdigest(),
        target_count=count,
        source_bundles=len(sources),
        usable_parts=len(observed),
        lift_protocol=json.loads(str(data["lift_validation_json"].item()))
        if "lift_validation_json" in data
        else "legacy_root_height_v1",
        expected_parts=len(expected),
        excluded_parts=sorted(expected - observed),
        per_part_targets=dict(sorted(Counter(data["part_keys"].tolist()).items())),
        splits=split_report,
        checks=[
            "finite_arrays",
            "matched_geometry",
            "unique_targets",
            "source_and_asset_sha256",
            "grouped_split_isolation",
            "approach_and_dynamic_lift_labels",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
