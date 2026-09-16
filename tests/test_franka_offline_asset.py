import hashlib
import json

import pytest

from grasp_planning.rl.franka_offline_asset import verified_robot_asset


def test_offline_asset_checks_identity_and_dependency_bytes(tmp_path):
    (tmp_path / "robot.usda").write_bytes(b"#usda 1.0\n")
    (tmp_path / "mesh.usda").write_bytes(b"#usda 1.0\n")
    manifest = dict(
        source_url="https://example.test/robot.usda",
        entry="robot.usda",
        files={
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in ("robot.usda", "mesh.usda")
        },
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert verified_robot_asset(path, manifest["source_url"]) == str(tmp_path / "robot.usda")
    with pytest.raises(ValueError, match="source differs"):
        verified_robot_asset(path, "https://example.test/different.usda")
    (tmp_path / "mesh.usda").write_bytes(b"changed geometry")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verified_robot_asset(path, manifest["source_url"])


def test_offline_asset_rejects_path_escape(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(source_url="robot", entry="../robot.usda", files={"../robot.usda": "unused"})))
    with pytest.raises(ValueError, match="escapes"):
        verified_robot_asset(path, "robot")
