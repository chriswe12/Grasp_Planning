#!/usr/bin/env python3
"""Add separately validated missing parts without changing existing target physics."""

import argparse
import hashlib
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle
from grasp_planning.mujoco import build_bundle_local_mesh

TARGET_KEYS = (
    "target_ids",
    "split",
    "validated",
    "joint_paths",
    "goal_rgbd",
    "goal_poses",
    "object_poses",
    "open_widths",
    "jaw_widths",
    "source_grasp_ids",
    "orientation_ids",
    "target_part_indices",
    "part_keys",
    "lab_approach_validated",
    "lift_validated",
)


def read(path, stage):
    with np.load(path, allow_pickle=False) as source:
        result = {key: source[key].copy() for key in source.files}
    for key in (
        ("validated", "lab_approach_validated", "lift_validated")
        if stage == "lift"
        else ("validated", "lab_approach_validated")
    ):
        assert result[key].all(), f"Unvalidated input: {path} {key}"
    return result


def mesh_for(part_key, paths):
    assembly, part = part_key.split("__part_")
    match = next(str(p) for p in paths if f"/parts/{assembly}/{part}/orientations/" in str(p))
    return build_bundle_local_mesh(load_grasp_bundle(ROOT / match))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--additional", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("approach", "lift"), default="lift")
    args = parser.parse_args()
    assert not args.output.exists(), "Do not overwrite a catalog already used for training"
    base = read(args.base, args.stage)
    extra = read(args.additional, args.stage)
    contract = json.loads(str(base["contract_json"].item()))
    extra_contract = json.loads(str(extra["contract_json"].item()))
    left = deepcopy(contract)
    right = deepcopy(extra_contract)
    assets = left.pop("object_assets")
    extra_assets = right.pop("object_assets")
    assert left == right, "Robot/camera/controller/appearance contracts differ"
    assert not set(base["part_keys"]) & set(extra["part_keys"]), "Only add parts with no existing validated targets"
    assert not set(base["target_ids"]) & set(extra["target_ids"]), "Duplicate target IDs"
    mapping = {}
    for index in np.unique(extra["target_part_indices"]):
        item = extra_assets[int(index)]
        part = item["part_key"]
        existing = next(i for i, a in enumerate(assets) if a["part_key"] == part)
        old_mesh = mesh_for(part, base["source_bundle_paths"])
        new_mesh = mesh_for(part, extra["source_bundle_paths"])
        assert np.array_equal(old_mesh.faces, new_mesh.faces), "Mesh topology changed"
        assert np.allclose(old_mesh.vertices_obj, new_mesh.vertices_obj, rtol=0.0, atol=1e-9), (
            "Bundle-local geometry/frame changed"
        )
        assert np.isclose(assets[existing]["object_mass_kg"], item["object_mass_kg"])
        assets[existing] = item
        mapping[int(index)] = existing
    extra["target_part_indices"] = np.asarray([mapping[int(i)] for i in extra["target_part_indices"]], dtype=np.int64)
    keys = [key for key in TARGET_KEYS if args.stage == "lift" or key != "lift_validated"]
    if args.stage == "lift" and ("lift_validation_json" in base or "lift_validation_json" in extra):
        assert str(base.get("lift_validation_json")) == str(extra.get("lift_validation_json")), "Lift protocols differ"
        keys.append("lift_clearance_validated")
    result = {key: np.concatenate((base[key], extra[key]), axis=0) for key in keys}
    if args.stage == "lift" and "lift_validation_json" in base:
        result["lift_validation_json"] = base["lift_validation_json"]

    contract["object_assets"] = assets
    result["contract_json"] = np.asarray(json.dumps(contract, sort_keys=True))
    sources = {}
    for data in (base, extra):
        for path, digest in zip(data["source_bundle_paths"], data["source_bundle_sha256"], strict=True):
            if path in sources:
                assert sources[path] == digest, "Source file identity changed"
            sources[str(path)] = str(digest)
    result["source_bundle_paths"] = np.asarray(list(sources))
    result["source_bundle_sha256"] = np.asarray(list(sources.values()))
    manifests = [json.loads((p.parent / "source_manifest.json").read_text()) for p in (args.base, args.additional)]
    lookup = {r["target_id"]: r for manifest in manifests for r in manifest["targets"]}
    targets = []
    for target, part_index in zip(result["target_ids"], result["target_part_indices"]):
        row = deepcopy(lookup[str(target)])
        row["part_index"] = int(part_index)
        targets.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, data in ((args.base, base), (args.additional, extra)):
        for target in data["target_ids"]:
            image = path.parent / f"{target}.png"
            if image.is_file():
                shutil.copyfile(image, args.output.parent / image.name)
    manifest = dict(
        schema_version=3,
        targets=targets,
        object_assets=assets,
        sources=[dict(path=path, sha256=digest) for path, digest in sources.items()],
        inputs=[
            dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (args.base, args.additional)
        ],
        validation_stage=args.stage,
        addition_policy=(
            "Previously missing parts only; same bundle-local mesh and physical mass; validated labels retained"
            if args.stage == "lift"
            else "Approach-only merge; physical labels cleared for a new complete lift validation"
        ),
    )
    (args.output.parent / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(args.output)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                targets=len(result["target_ids"]),
                parts=len(set(result["part_keys"])),
                added_parts=sorted(set(extra["part_keys"].tolist())),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
