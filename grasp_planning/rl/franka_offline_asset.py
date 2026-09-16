"""Validate a byte-identical offline mirror without changing robot identity."""

import hashlib
import json
from pathlib import Path


def verified_robot_asset(manifest_path, expected_source):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_url"] != expected_source:
        raise ValueError("Offline robot asset source differs from the task robot")
    root = manifest_path.parent
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Offline robot dependency escapes its mirror")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Offline robot dependency checksum mismatch: {relative}")
    if manifest["entry"] not in manifest["files"]:
        raise ValueError("Robot entry is not checksum protected")
    return str(root / manifest["entry"])
