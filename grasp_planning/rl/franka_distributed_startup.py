"""Serialize native Isaac startup, then rendezvous before PPO/NCCL starts."""

import fcntl
import os
import time
from pathlib import Path


def acquire_startup_lock():
    value = os.environ.get("ISAAC_RL_STARTUP_LOCK")
    if not value:
        return None
    path = Path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    print("[FRANKA STARTUP] Waiting for native startup lock", flush=True)
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def release_startup_lock(handle):
    if handle is not None and not handle.closed:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def environment_barrier(rank, world_size):
    value = os.environ.get("ISAAC_RL_DISTRIBUTED_READY_DIR")
    if world_size <= 1 or not value:
        return
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"rank_{rank}.ready").write_text("ready\n")
    deadline = time.monotonic() + 1800
    while not all((path / f"rank_{i}.ready").is_file() for i in range(world_size)):
        if time.monotonic() > deadline:
            raise TimeoutError(f"Franka environments did not rendezvous at {path}")
        time.sleep(1)
    print(f"[FRANKA STARTUP] All {world_size} environments ready", flush=True)
