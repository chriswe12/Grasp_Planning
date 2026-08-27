from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "euler/select_validation_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("select_validation_checkpoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def _write_summary(root: Path, epoch: int, *, success: int, collision: int) -> None:
    output_dir = root / f"epoch_{epoch}"
    output_dir.mkdir(parents=True)
    attempts = 10
    payload = {
        "checkpoint": f"/tmp/checkpoint_{epoch}.pth",
        "catalog_split": "validation",
        "conditions": {
            "far": {
                "attempts": attempts,
                "successes": success,
                "terminations": {
                    "success": success,
                    "unsafe_collision": collision,
                    "timeout": attempts - success - collision,
                },
            }
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )


def test_selector_prefers_safe_validation_success_and_writes_record(tmp_path: Path) -> None:
    _write_summary(tmp_path, 1000, success=8, collision=2)
    _write_summary(tmp_path, 2000, success=8, collision=0)
    _write_summary(tmp_path, 3000, success=7, collision=0)

    records = selector.rank_validation_reports(tmp_path)
    selector.write_selection(tmp_path, records)

    assert [record["epoch"] for record in records] == [2000, 3000, 1000]
    selection = json.loads((tmp_path / "checkpoint_selection.json").read_text(encoding="utf-8"))
    assert selection["best"]["epoch"] == 2000
    assert (tmp_path / "best_checkpoint.txt").read_text(encoding="utf-8").strip().endswith("checkpoint_2000.pth")


def test_selector_rejects_test_split_to_prevent_selection_leakage(tmp_path: Path) -> None:
    _write_summary(tmp_path, 1000, success=8, collision=0)
    summary_path = tmp_path / "epoch_1000/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["catalog_split"] = "test"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    try:
        selector.rank_validation_reports(tmp_path)
    except ValueError as exc:
        assert "non-validation" in str(exc)
    else:
        raise AssertionError("test split must not participate in checkpoint selection")
