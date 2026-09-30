"""Connection ownership and build checks; never connects to the robot."""

import os
from pathlib import Path


def acquire_robot_lock(robot_ip):
    """Keep the returned descriptor alive for the entire launch process lifetime."""
    import fcntl
    import hashlib

    key = hashlib.sha256(robot_ip.encode()).hexdigest()[:16]
    path = Path(f"/tmp/franka-policy-{os.getuid()}-{key}.lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError("A Franka policy connection already owns this robot; stop it before reconnecting.") from None
    return fd


def require_driver_manifest(root):
    """Do not accept a pre-smoothing library merely because its path matches."""
    import hashlib
    import json

    overlay = root / ".cache/franka_velocity_driver"
    try:
        manifest = json.loads((overlay / "policy_driver_manifest.json").read_text())
        library = overlay / "install/franka_hardware/lib/libfranka_hardware.so"
        header = root / "scripts/franka_policy_velocity_filter.hpp"
        valid = (
            manifest["version"] == 2
            and manifest["library_sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
            and manifest["filter_sha256"] == hashlib.sha256(header.read_bytes()).hexdigest()
        )
    except (OSError, ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise RuntimeError("Updated velocity driver required: run bash scripts/build_franka_velocity_driver.sh")
