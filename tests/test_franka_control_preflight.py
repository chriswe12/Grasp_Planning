import hashlib
import json
import os

import pytest

from grasp_planning.real_franka import control_preflight


def test_duplicate_robot_connection_is_blocked():
    key = f"offline-lock-test-{os.getpid()}"
    fd = control_preflight.acquire_robot_lock(key)
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            control_preflight.acquire_robot_lock(key)
    finally:
        os.close(fd)
    os.close(control_preflight.acquire_robot_lock(key))


def test_driver_manifest_rejects_stale_or_changed_binary(tmp_path):
    with pytest.raises(RuntimeError, match="Updated velocity driver"):
        control_preflight.require_driver_manifest(tmp_path)
    overlay = tmp_path / ".cache/franka_velocity_driver"
    library = overlay / "install/franka_hardware/lib/libfranka_hardware.so"
    header = tmp_path / "scripts/franka_policy_velocity_filter.hpp"
    library.parent.mkdir(parents=True)
    header.parent.mkdir(parents=True)
    library.write_bytes(b"new driver")
    header.write_bytes(b"filter")

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    (overlay / "policy_driver_manifest.json").write_text(
        json.dumps(dict(version=2, library_sha256=digest(library), filter_sha256=digest(header)))
    )
    control_preflight.require_driver_manifest(tmp_path)
    library.write_bytes(b"old driver")
    with pytest.raises(RuntimeError, match="Updated velocity driver"):
        control_preflight.require_driver_manifest(tmp_path)


def test_policy_rejects_duplicate_command_publishers():
    from types import SimpleNamespace

    from grasp_planning.real_franka.ros_control import RosControl

    control = RosControl.__new__(RosControl)
    control.node = SimpleNamespace(count_publishers=lambda topic: 1)
    control.require_single_command_sources()
    for count in [0, 2]:
        control.node.count_publishers = lambda topic: count
        with pytest.raises(RuntimeError, match="Expected one command publisher"):
            control.require_single_command_sources()
