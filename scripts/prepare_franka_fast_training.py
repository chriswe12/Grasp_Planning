#!/usr/bin/env python3
"""Create an explicit visual-only catalog upgrade and optional frame-preserving resume copy."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp_planning.rl.franka_performance import optimized_contract, resume_epoch_for_frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--renderer",
        choices=("preserve_isaaclab_2_3_balanced", "balanced_fxaa"),
        default="preserve_isaaclab_2_3_balanced",
    )
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint-output", type=Path)
    p.add_argument("--total-envs", type=int)
    a = p.parse_args()
    if a.source.resolve() == a.output.resolve():
        p.error("Preserve the source catalog")
    with np.load(a.source, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    source_contract = json.loads(str(data["contract_json"].item()))
    contract = optimized_contract(source_contract, a.renderer)
    original_hashes = {
        key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in data.items() if key != "contract_json"
    }
    data["contract_json"] = np.asarray(json.dumps(contract, sort_keys=True))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        with np.load(a.output, allow_pickle=False) as existing:
            if json.loads(str(existing["contract_json"].item())) != contract or any(
                hashlib.sha256(existing[k].tobytes()).hexdigest() != digest for k, digest in original_hashes.items()
            ):
                raise ValueError("Refusing to overwrite a different output catalog")
    else:
        np.savez_compressed(a.output, **data)
    report = dict(
        source=str(a.source),
        source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        catalog_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        unchanged_arrays=original_hashes,
        performance_profile=contract["performance_profile"],
        goal_images="Original canonical goal images preserved byte-for-byte",
        color_palette="unchanged",
    )
    if a.checkpoint:
        if int(np.__version__.split(".")[0]) >= 2:
            raise RuntimeError("Promote checkpoints using scripts/franka_isaac_python.sh (Isaac NumPy ABI)")
        if not a.checkpoint_output or not a.total_envs:
            p.error("Checkpoint promotion requires --checkpoint-output and --total-envs")
        if a.checkpoint.resolve() == a.checkpoint_output.resolve() or a.checkpoint_output.exists():
            p.error("Use a new checkpoint destination; preserve the original")
        if json.loads(a.checkpoint.with_suffix(".contract.json").read_text()) != source_contract:
            raise ValueError("Source checkpoint and source catalog must match exactly")
        import torch

        checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        original_epoch = checkpoint["epoch"]
        checkpoint["epoch"] = resume_epoch_for_frames(int(checkpoint["frame"]), a.total_envs)
        a.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, a.checkpoint_output)
        a.checkpoint_output.with_suffix(".contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        provenance = dict(
            source_checkpoint=str(a.checkpoint),
            source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            source_epoch=original_epoch,
            epoch=checkpoint["epoch"],
            frames=checkpoint["frame"],
            total_envs=a.total_envs,
            changed_checkpoint_fields=["epoch"] if original_epoch != checkpoint["epoch"] else [],
            checkpoint_sha256=hashlib.sha256(a.checkpoint_output.read_bytes()).hexdigest(),
            contract_change="Explicit performance_profile only; model and optimizer preserved",
        )
        a.checkpoint_output.with_suffix(".promotion.json").write_text(json.dumps(provenance, indent=2) + "\n")
        report["checkpoint_promotion"] = provenance
    a.output.with_suffix(".performance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "unchanged_arrays"}, indent=2))


if __name__ == "__main__":
    main()
