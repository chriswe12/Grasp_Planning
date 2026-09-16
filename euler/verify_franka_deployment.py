#!/usr/bin/env python3
"""Check portable Franka data/asset identity and inventory a deployment locally."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_planning.rl.franka_offline_asset import verified_robot_asset
from grasp_planning.rl.video_lab_scene import training_asset_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--catalog", type=Path, default=Path("isaac_rl/data/franka_fabrica_pencil_randomized/catalog.npz")
    )
    parser.add_argument(
        "--copy-sources-from", type=Path, help="Stage the exact catalog-referenced grasp bundles before verification"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.catalog.is_absolute() or ".." in args.catalog.parts:
        raise ValueError("Catalog must be a portable project-relative path")
    catalog = root / args.catalog
    with np.load(catalog, allow_pickle=False) as source:
        contract = json.loads(str(source["contract_json"].item()))
        split_counts = {split: int((source["split"] == split).sum()) for split in ("train", "validation", "test")}
        sources = list(
            zip(source["source_bundle_paths"].tolist(), source["source_bundle_sha256"].tolist(), strict=True)
        )
    for relative, expected in sources:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Source bundle is not portable: {relative}")
        if args.copy_sources_from:
            original = (args.copy_sources_from / relative).read_bytes()
            assert hashlib.sha256(original).hexdigest() == expected, f"Source bundle changed: {relative}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Missing or changed bundle: {relative}"
    verified_robot_asset(root / "assets/usd/franka_panda_offline/manifest.json", contract["robot_usd"])
    objects = contract.get("object_assets") or [contract]
    for item in objects:
        part = (root / item["object_usd"]).resolve()
        if not part.is_relative_to(root):
            raise ValueError(f"Object asset is not portable: {item['object_usd']}")
        if args.copy_sources_from and contract.get("object_assets"):
            original = (args.copy_sources_from / item["object_usd"]).resolve()
            if not original.is_relative_to(args.copy_sources_from.resolve()):
                raise ValueError("Object source is outside the source project")
            assert hashlib.sha256(original.read_bytes()).hexdigest() == item["object_sha256"]
            # Each converted object has its own directory. Include adjacent USD
            # dependencies/configuration even when the catalog was merged elsewhere.
            shutil.copytree(original.parent, part.parent, dirs_exist_ok=True)
        assert hashlib.sha256(part.read_bytes()).hexdigest() == item["object_sha256"]
    lab = contract["lab_scene"]
    assert training_asset_digest(root / lab["asset_dir"]) == lab["asset_sha256"]
    sidecar = root / "checkpoints/resume.contract.json"
    if sidecar.exists():
        assert json.loads(sidecar.read_text()) == contract, "Checkpoint contract differs from catalog"
        assert (root / "checkpoints/resume.pth").is_file()
    files = {
        p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    report = dict(
        passed=True,
        root=str(root),
        catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        splits=split_counts,
        source_bundles=len(sources),
        file_count=len(files),
        total_bytes=sum(f["bytes"] for f in files.values()),
        files=files,
        scope="Portable source/data/asset checks only; no simulator or optimizer execution",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
