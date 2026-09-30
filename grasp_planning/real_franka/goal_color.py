"""Cached Isaac goal renders shared by the configurator and policy runner."""

import hashlib
import json
import re
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from .core import ROOT, sha256

RENDER_VERSION = 2
CACHE = ROOT / "artifacts/franka_goal_colors"


def normalize_color(value):
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError("Choose an RGB color in #RRGGBB format")
    return value.lower()


def metadata(cat, i, color):
    return dict(
        version=RENDER_VERSION,
        catalog_sha256=cat.hash,
        target_id=str(cat.data["target_ids"][i]),
        color=normalize_color(color),
    )


def load_goal(cat, i, cfg):
    """Fail closed on a missing, stale, mismatched or corrupted custom goal."""
    index = cfg.get("goal_variant_index", 0)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 4:
        raise ValueError("Goal variant index must be 0..4")
    if index:
        if cfg.get("goal_render"):
            raise ValueError("Choose either a cached goal variant or a custom color render")
        if "goal_rgbd_variants" not in cat.data:
            raise ValueError("This catalog has no mixed renderer goals")
        return cat.data["goal_rgbd_variants"][i, index - 1].astype(np.float32)
    variant = cfg.get("goal_render")
    if not variant:
        return cat.data["goal_rgbd"][i]
    expected = metadata(cat, i, variant["color"])
    path = Path(variant["path"])
    if sha256(path) != variant["sha256"]:
        raise ValueError("Selected color render changed; render and save again")
    with np.load(path, allow_pickle=False) as data:
        if json.loads(str(data["metadata_json"].item())) != expected:
            raise ValueError("Color render belongs to a different target or catalog")
        image = data["goal_rgbd"].astype(np.float32)
    if image.shape != cat.data["goal_rgbd"][i].shape or not np.isfinite(image).all():
        raise ValueError("Invalid color goal RGB-D")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Color goal RGB-D must be normalized")
    if not np.array_equal(image[..., 3], cat.data["goal_rgbd"][i][..., 3]):
        raise ValueError("Color goal must preserve the original training depth")
    return image


def render_goal(cat, i, color, stop, report):
    """One owned, temporary Isaac container. Never touches ROS or robot control."""
    meta = metadata(cat, i, color)
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    folder = CACHE / key
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "goal.npz"

    def result():
        variant = dict(color=meta["color"], path=str(output), sha256=sha256(output))
        load_goal(cat, i, {"goal_render": variant})
        return variant

    if output.exists():
        return result()
    contract = cat.contract
    asset = next(a for a in contract["object_assets"] if a["part_key"] == cat.data["part_keys"][i])
    request = dict(
        metadata=meta,
        asset=asset,
        contract=contract,
        joints=cat.data["joint_paths"][i, -1].tolist(),
        object_pose=cat.data["object_poses"][i].tolist(),
        open_width=float(cat.data["open_widths"][i]),
        goal_depth=cat.data["goal_rgbd"][i, ..., 3].tolist(),
    )
    request_path = folder / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    name = "franka-goal-" + uuid.uuid4().hex[:12]
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--gpus",
        "all",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "PRIVACY_CONSENT=Y",
        "-v",
        f"{ROOT}:/workspace/project",
        "-w",
        "/workspace/project",
        "--entrypoint",
        "/isaac-sim/python.sh",
        "isaac-lab-euler:2.3.2",
        "scripts/render_franka_goal_color.py",
        "--request",
        str(Path("/workspace/project") / request_path.relative_to(ROOT)),
        "--headless",
        "--device",
        "cuda:0",
    ]
    log = folder / "render.log"
    report("Rendering object color in Isaac… First render may take a few minutes. Stop cancels it.")
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while process.poll() is None:
                if stop.wait(0.25):
                    raise RuntimeError("Color render cancelled")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Isaac render timed out. See {log}")
            if process.returncode or not output.exists():
                raise RuntimeError(f"Isaac color render failed. See {log}")
        finally:
            # Remove only this request's container, including on cancellation.
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=20)
    return result()
