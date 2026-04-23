from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PACKAGE_ROOT = REPO_ROOT / "ros2_ws" / "src" / "robot_integration_ros"
ROS2_DEPENDENCY_MANIFEST = REPO_ROOT / "ros2_ws" / "dependencies.repos"
ROS2_DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "download_ros2_dependencies.sh"


def test_workspace_package_layout_exists() -> None:
    assert (REPO_ROOT / "ros2_ws" / "README.md").is_file()
    assert (WORKSPACE_PACKAGE_ROOT / "package.xml").is_file()
    assert (WORKSPACE_PACKAGE_ROOT / "setup.py").is_file()
    assert (WORKSPACE_PACKAGE_ROOT / "robot_integration_ros" / "__init__.py").is_file()


def test_workspace_dependency_bootstrap_files_exist() -> None:
    assert ROS2_DEPENDENCY_MANIFEST.is_file()
    assert ROS2_DOWNLOAD_SCRIPT.is_file()

    payload = yaml.safe_load(ROS2_DEPENDENCY_MANIFEST.read_text(encoding="utf-8"))
    repositories = payload["repositories"]
    fp_debug_msgs = repositories["fp_debug_msgs"]
    assert fp_debug_msgs["type"] == "git"
    assert fp_debug_msgs["url"] == "https://github.com/Moreno-Nautilus/fp_debug_msgs.git"
    assert fp_debug_msgs["version"] == "7cab8c96effad8f3489fa509dfe5cd2795242c37"

    script_text = ROS2_DOWNLOAD_SCRIPT.read_text(encoding="utf-8")
    assert "ros2_ws/dependencies.repos" in script_text
    assert "fp_debug_msgs" in script_text


def test_workspace_cli_module_imports_from_source_tree() -> None:
    sys.path.insert(0, str(WORKSPACE_PACKAGE_ROOT))
    try:
        module = importlib.import_module("robot_integration_ros.move_real_robot_ee")
        assert callable(module.main)
        assert callable(module.pose_target_from_args)
    finally:
        sys.path = [entry for entry in sys.path if entry != str(WORKSPACE_PACKAGE_ROOT)]


def _write_temp_repo_copy(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "ros2_ws").mkdir(parents=True)
    shutil.copy2(ROS2_DOWNLOAD_SCRIPT, repo_root / "scripts" / ROS2_DOWNLOAD_SCRIPT.name)
    shutil.copy2(ROS2_DEPENDENCY_MANIFEST, repo_root / "ros2_ws" / ROS2_DEPENDENCY_MANIFEST.name)
    return repo_root


def _create_fp_debug_msgs_remote(tmp_path: Path) -> tuple[Path, str]:
    seed_root = tmp_path / "seed"
    remote_root = tmp_path / "fp_debug_msgs.git"
    (seed_root / "msg").mkdir(parents=True)
    (seed_root / "package.xml").write_text("<package format='3'></package>\n", encoding="utf-8")
    (seed_root / "msg" / "DebugFrame.msg").write_text("string object_id\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=seed_root, check=True)
    subprocess.run(["git", "add", "package.xml", "msg/DebugFrame.msg"], cwd=seed_root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "seed"],
        cwd=seed_root,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=seed_root, text=True).strip()
    subprocess.run(["git", "init", "--bare", str(remote_root)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(remote_root)], cwd=seed_root, check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=seed_root, check=True)
    return remote_root, commit


def _write_fake_vcs(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    vcs_path = fake_bin / "vcs"
    vcs_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
input=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      input="$2"
      shift 2
      ;;
    --force)
      shift
      ;;
    *)
      shift
      ;;
  esac
done
grep -F -- "$FP_DEBUG_MSGS_REMOTE" "$input" >/dev/null
""",
        encoding="utf-8",
    )
    vcs_path.chmod(0o755)
    return fake_bin


def test_download_script_honors_remote_override_for_vcs_import(tmp_path: Path) -> None:
    repo_root = _write_temp_repo_copy(tmp_path)
    remote_root, commit = _create_fp_debug_msgs_remote(tmp_path)
    fake_bin = _write_fake_vcs(tmp_path)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["FP_DEBUG_MSGS_REMOTE"] = str(remote_root)
    env["FP_DEBUG_MSGS_REF"] = commit

    subprocess.run(
        ["bash", str(repo_root / "scripts" / "download_ros2_dependencies.sh")],
        cwd=repo_root,
        env=env,
        check=True,
    )

    checkout_root = repo_root / "ros2_ws" / "src" / "fp_debug_msgs"
    assert (checkout_root / "package.xml").is_file()
    assert (checkout_root / "msg" / "DebugFrame.msg").is_file()
    assert subprocess.check_output(
        ["git", "-C", str(checkout_root), "remote", "get-url", "origin"],
        text=True,
    ).strip() == str(remote_root)


def test_download_script_skips_fetch_when_checkout_is_already_pinned(tmp_path: Path) -> None:
    repo_root = _write_temp_repo_copy(tmp_path)
    remote_root, commit = _create_fp_debug_msgs_remote(tmp_path)
    fake_bin = _write_fake_vcs(tmp_path)

    first_env = dict(os.environ)
    first_env["PATH"] = f"{fake_bin}:{first_env['PATH']}"
    first_env["FP_DEBUG_MSGS_REMOTE"] = str(remote_root)
    first_env["FP_DEBUG_MSGS_REF"] = commit

    script_path = repo_root / "scripts" / "download_ros2_dependencies.sh"
    subprocess.run(["bash", str(script_path)], cwd=repo_root, env=first_env, check=True)

    second_env = dict(first_env)
    second_env["FP_DEBUG_MSGS_REMOTE"] = str(tmp_path / "offline-remote.git")
    subprocess.run(["bash", str(script_path)], cwd=repo_root, env=second_env, check=True)

    checkout_root = repo_root / "ros2_ws" / "src" / "fp_debug_msgs"
    assert (
        subprocess.check_output(
            ["git", "-C", str(checkout_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        == commit
    )
