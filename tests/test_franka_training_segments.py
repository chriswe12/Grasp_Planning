"""Verify continuation milestones and optional evaluation without submitting jobs."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "submit_franka_long", Path(__file__).parents[1] / "euler/submit_franka_long.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("mode,expected", [("each", [True, True]), ("final", [False, True]), ("none", [False, False])])
def test_completed_resume_skips_passed_milestones_and_controls_evaluation(tmp_path, monkeypatch, mode, expected):
    catalog = tmp_path / "catalog.npz"
    catalog.write_bytes(b"audited catalog")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "passed": True,
                "splits": {"train": {"targets": 1}},
                "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
            }
        )
    )
    output = tmp_path / "chain.json"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "submit_franka_long.py",
            "--config",
            str(tmp_path / "euler.env"),
            "--catalog",
            "catalog.npz",
            "--catalog-audit",
            str(audit),
            "--output",
            str(output),
            "--initial-job",
            "123",
            "--segments",
            "3",
            "--epochs-per-segment",
            "1000",
            "--start-after-epoch",
            "1006",
            "--evaluation-mode",
            mode,
        ],
    )
    commands = []

    def submit(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=f"Submitted batch job {200 + len(commands)}\n")

    monkeypatch.setattr(module.subprocess, "run", submit)
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda command, **kw: "COMPLETED|0:0|\n" if command[0] == "ssh" else "test-login",
    )
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=987))
    module.main()
    assert [int(c[c.index("--iterations") + 1]) for c in commands] == [2000, 3000]
    assert ["--evaluate-after" in c for c in commands] == expected
    assert "--evaluate-test-after" not in commands[0]
    assert ("--evaluate-test-after" in commands[-1]) == (mode != "none")
    assert "--afterok" not in commands[0]  # Already completed, possibly outside Slurm MinJobAge.
    assert commands[1][commands[1].index("--afterok") + 1] == "201"
    assert json.loads(output.read_text())["final_epoch"] == 3000
