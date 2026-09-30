"""Prevent selecting an easy close-start result as the best overall policy."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "franka_validation_summary", Path(__file__).parents[1] / "euler/franka_validation_summary.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_report(root, checkpoint, progress, score):
    path = root / f"{checkpoint}_{progress}" / "evaluation.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            dict(
                catalog_split="validation",
                coverage_complete=True,
                checkpoint=checkpoint,
                pose_evaluation_progress=progress,
                macro_part_success=score,
            )
        )
    )


def test_ranks_complete_means_instead_of_easiest_condition(tmp_path):
    for progress, easy, balanced in zip((0.0, 0.5, 0.94), (0.0, 0.1, 1.0), (0.5, 0.6, 0.7)):
        write_report(tmp_path, "easy", progress, easy)
        write_report(tmp_path, "balanced", progress, balanced)
    write_report(tmp_path, "incomplete", 0.94, 1.0)
    result = module.summarize_validation(tmp_path)
    assert result["best_validation"]["checkpoint"] == "balanced"
    assert len(result["validation_by_checkpoint"]) == 2
    assert len(result["validation"]) == 7


def test_legacy_single_condition_and_empty_results(tmp_path):
    assert module.summarize_validation(tmp_path)["best_validation"] is None
    write_report(tmp_path, "legacy", None, 0.5)
    assert module.summarize_validation(tmp_path)["best_validation"]["checkpoint"] == "legacy"
