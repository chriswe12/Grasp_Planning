"""Operator video ratings and verified copies of selected recordings."""

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .core import ROOT, sha256

RUN_ID = re.compile(r"\d{8}_\d{6}")


def completed_run(path):
    path = Path(path)
    if not RUN_ID.fullmatch(path.stem) or not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
        recording = data.get("recording") or {}
        return bool(
            recording.get("frames")
            and recording.get("video")
            and recording.get("rgbd")
            and path.with_suffix(".mp4").is_file()
            and path.with_suffix(".rgbd.npz").is_file()
        )
    except (ValueError, OSError):
        return False


def latest_recording(folder):
    for path in sorted(Path(folder).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if completed_run(path):
            return path
    return None


def rating(path):
    if path is None:
        return None
    file = Path(path).with_suffix(".review.json")
    return json.loads(file.read_text()) if file.exists() else None


def write_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def review_run(path, good, collection=None, source="user_ui"):
    path = Path(path).resolve()
    if type(good) is not bool:
        raise ValueError("Choose Good video or Not good")
    if not completed_run(path):
        raise ValueError("Wait until the run, video and RGB-D recording have finished saving")
    collection = Path(collection or ROOT / "artifacts/franka_good_runs").resolve()
    destination = collection / path.stem
    review = dict(
        run_id=path.stem,
        good_video=good,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        source=source,
        meaning="Operator video-quality/usefulness rating; not verified grasp success",
    )
    if good:
        data = json.loads(path.read_text())
        cfg = data["config"]
        collection.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            manifest = json.loads((destination / "manifest.json").read_text())
            if manifest["source_log_sha256"] != sha256(path):
                raise ValueError("A different recording already has this run ID")
            for item in manifest["files"].values():
                if sha256(destination / item["path"]) != item["sha256"]:
                    raise ValueError("Archived recording changed; refusing to silently replace it")
        else:
            temporary = collection / ("." + path.stem + "-" + uuid.uuid4().hex)
            temporary.mkdir()
            files = {}

            def copy_checked(src, dst, expected=None):
                digest = sha256(src)
                if expected is not None and digest != expected:
                    raise ValueError(f"Run artifact changed since execution: {src}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                if sha256(dst) != digest:
                    raise ValueError(f"Archive copy verification failed: {dst}")
                return digest

            try:
                for suffix, name in [
                    (".json", "run.json"),
                    (".mp4", "video.mp4"),
                    (".rgbd.npz", "frames.rgbd.npz"),
                    (".encoding.log", "encoding.log"),
                ]:
                    src = path.with_suffix(suffix)
                    if src.exists():
                        files[name] = dict(path=name, sha256=copy_checked(src, temporary / name))
                # Keep one real copy of each large immutable asset in the collection.
                # Relative references remain valid when the entire collection moves.
                assets = [
                    ("checkpoint", Path(cfg["checkpoint"]), cfg["checkpoint_sha256"]),
                    ("catalog", Path(cfg["catalog"]), cfg["catalog_sha256"]),
                    ("contract", Path(cfg["checkpoint"]).with_suffix(".contract.json"), None),
                ]
                if cfg.get("goal_render"):
                    assets.append(("custom_goal", Path(cfg["goal_render"]["path"]), cfg["goal_render"]["sha256"]))
                for key, src, expected in assets:
                    digest = sha256(src)
                    if expected is not None and digest != expected:
                        raise ValueError(f"Run artifact changed since execution: {src}")
                    dst = collection / "_assets" / digest / src.name
                    if not dst.exists():
                        temp_asset = dst.with_name(dst.name + "." + uuid.uuid4().hex + ".tmp")
                        try:
                            copy_checked(src, temp_asset, digest)
                            temp_asset.replace(dst)
                        finally:
                            temp_asset.unlink(missing_ok=True)
                    elif sha256(dst) != digest:
                        raise ValueError(f"Archived asset hash mismatch: {dst}")
                    files[key] = dict(path="../" + str(dst.relative_to(collection)), sha256=digest)
                write_json(temporary / "config.json", cfg)
                write_json(temporary / "camera.json", data["camera"])
                for name in ("config.json", "camera.json"):
                    files[name] = dict(path=name, sha256=sha256(temporary / name))
                manifest = dict(
                    schema_version=1,
                    run_id=path.stem,
                    source_log=str(path),
                    source_log_sha256=sha256(path),
                    files=files,
                    outcome=data["outcome"],
                    execute=data["execute"],
                    steps=len(data["steps"]),
                    archived_at=review["reviewed_at"],
                )
                write_json(temporary / "manifest.json", manifest)
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        review["archive"] = str(destination)
    if destination.exists():
        # A later negative rating keeps existing data, and clearly changes its label.
        review["archive"] = str(destination)
        write_json(destination / "review.json", review)
    write_json(path.with_suffix(".review.json"), review)
    return review
