"""Bounded subprocess execution that also reaps detached descendants."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Sequence


def _signal_process_group(process_group_id: int, signal_number: int) -> bool:
    try:
        os.killpg(int(process_group_id), signal_number)
    except ProcessLookupError:
        return False
    return True


def terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_period_s: float = 0.5,
) -> None:
    """Terminate the launched process and every descendant in its process group."""

    process_group_id = int(process.pid)
    if _signal_process_group(process_group_id, signal.SIGTERM):
        deadline = time.monotonic() + max(0.0, float(grace_period_s))
        while time.monotonic() < deadline:
            try:
                os.killpg(process_group_id, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        _signal_process_group(process_group_id, signal.SIGKILL)
    if process.poll() is None:
        try:
            process.wait(timeout=max(0.1, float(grace_period_s)))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_process_group(
    command: Sequence[str],
    *,
    cwd: str | Path,
    timeout_s: float | None = None,
) -> int:
    """Run one command and always tear down its complete private process group."""

    process = subprocess.Popen(
        [str(value) for value in command],
        cwd=Path(cwd),
        start_new_session=True,
    )
    try:
        return int(process.wait(timeout=timeout_s))
    finally:
        terminate_process_group(process)


__all__ = ["run_process_group", "terminate_process_group"]
