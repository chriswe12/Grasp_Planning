from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PACKAGE_ROOT = REPO_ROOT / "ros2_ws" / "src" / "robot_integration_ros"


def test_workspace_package_layout_exists() -> None:
    assert (REPO_ROOT / "ros2_ws" / "README.md").is_file()
    assert (WORKSPACE_PACKAGE_ROOT / "package.xml").is_file()
    assert (WORKSPACE_PACKAGE_ROOT / "setup.py").is_file()
    assert (WORKSPACE_PACKAGE_ROOT / "robot_integration_ros" / "__init__.py").is_file()


def test_workspace_cli_module_imports_from_source_tree() -> None:
    sys.path.insert(0, str(WORKSPACE_PACKAGE_ROOT))
    try:
        module = importlib.import_module("robot_integration_ros.move_real_robot_ee")
        assert callable(module.main)
        assert callable(module.pose_target_from_args)
    finally:
        sys.path = [entry for entry in sys.path if entry != str(WORKSPACE_PACKAGE_ROOT)]
