from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from grasp_planning import subprocess_lifecycle


def test_run_process_group_starts_private_session_and_always_reaps_group(
    tmp_path: Path,
) -> None:
    process = mock.Mock(spec=subprocess.Popen)
    process.pid = 4321
    process.wait.return_value = 7
    with (
        mock.patch.object(subprocess_lifecycle.subprocess, "Popen", return_value=process) as popen,
        mock.patch.object(subprocess_lifecycle, "terminate_process_group") as terminate,
    ):
        result = subprocess_lifecycle.run_process_group(
            ["renderer", "--goal", "g1"], cwd=tmp_path, timeout_s=12.0
        )

    assert result == 7
    popen.assert_called_once_with(
        ["renderer", "--goal", "g1"],
        cwd=tmp_path,
        start_new_session=True,
    )
    process.wait.assert_called_once_with(timeout=12.0)
    terminate.assert_called_once_with(process)


def test_run_process_group_reaps_group_after_timeout(tmp_path: Path) -> None:
    process = mock.Mock(spec=subprocess.Popen)
    process.pid = 4321
    process.wait.side_effect = subprocess.TimeoutExpired("renderer", 1.0)
    with (
        mock.patch.object(subprocess_lifecycle.subprocess, "Popen", return_value=process),
        mock.patch.object(subprocess_lifecycle, "terminate_process_group") as terminate,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        subprocess_lifecycle.run_process_group(
            ["renderer"], cwd=tmp_path, timeout_s=1.0
        )

    terminate.assert_called_once_with(process)
