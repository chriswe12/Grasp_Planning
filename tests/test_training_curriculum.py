from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/training_curriculum.py"
)
SPEC = importlib.util.spec_from_file_location("training_curriculum", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
training_curriculum = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = training_curriculum
SPEC.loader.exec_module(training_curriculum)


def test_curriculum_warms_up_then_expands_every_difficulty_axis() -> None:
    before = training_curriculum.curriculum_state(
        10,
        enabled=True,
        warmup_steps=100,
        full_steps=300,
        initial_progress_min=0.7,
        final_progress_min=0.0,
        final_failure_replay_fraction=0.25,
    )
    middle = training_curriculum.curriculum_state(
        200,
        enabled=True,
        warmup_steps=100,
        full_steps=300,
        initial_progress_min=0.7,
        final_progress_min=0.0,
        final_failure_replay_fraction=0.25,
    )
    after = training_curriculum.curriculum_state(
        400,
        enabled=True,
        warmup_steps=100,
        full_steps=300,
        initial_progress_min=0.7,
        final_progress_min=0.0,
        final_failure_replay_fraction=0.25,
    )

    assert before.fraction == 0.0
    assert before.progress_min == pytest.approx(0.7)
    assert middle.fraction == pytest.approx(0.5)
    assert middle.progress_min == pytest.approx(0.35)
    assert middle.perturbation_scale == pytest.approx(0.5)
    assert middle.visual_randomization_strength == pytest.approx(0.5)
    assert middle.failure_replay_fraction == pytest.approx(0.125)
    assert after.fraction == 1.0
    assert after.progress_min == 0.0


def test_reset_mixture_assigns_every_explicit_mode() -> None:
    samples = torch.tensor([0.05, 0.20, 0.35, 0.80])

    result = training_curriculum.sample_reset_modes(
        samples,
        no_noise_fraction=0.10,
        ready_fraction=0.20,
        boundary_fraction=0.20,
    )

    assert result.tolist() == [
        training_curriculum.RESET_MODE_NO_NOISE,
        training_curriculum.RESET_MODE_READY,
        training_curriculum.RESET_MODE_BOUNDARY,
        training_curriculum.RESET_MODE_PATH,
    ]


def test_reset_timeout_tracks_distance_and_special_modes() -> None:
    progress = torch.tensor([0.0, 0.5, 1.0, 0.2])
    modes = torch.tensor(
        [
            training_curriculum.RESET_MODE_PATH,
            training_curriculum.RESET_MODE_PATH,
            training_curriculum.RESET_MODE_READY,
            training_curriculum.RESET_MODE_BOUNDARY,
        ]
    )

    result = training_curriculum.reset_timeout_seconds(
        progress,
        modes,
        far_seconds=12.0,
        close_seconds=4.0,
        ready_seconds=1.5,
        boundary_seconds=2.5,
        exponent=1.0,
    )

    assert result.tolist() == pytest.approx([12.0, 8.0, 1.5, 2.5])


def test_failure_replay_stays_part_balanced_and_prefers_hard_targets() -> None:
    torch.manual_seed(5)
    groups = torch.tensor([0, 0, 1, 1])
    targets = torch.tensor([0, 1, 2, 3] * 100)
    failure_scores = torch.tensor([0.0, 10.0, 0.0, 10.0])

    replayed, mask = training_curriculum.apply_failure_replay(
        targets,
        target_group_indices=groups,
        failure_scores=failure_scores,
        replay_fraction=1.0,
        score_floor=0.1,
        score_power=1.5,
    )

    assert mask.all()
    assert torch.equal(groups[replayed], groups[targets])
    assert (replayed == 1).sum() > 180
    assert (replayed == 3).sum() > 180


def test_failure_scores_average_repeated_terminal_targets() -> None:
    scores = torch.zeros(3)

    result = training_curriculum.update_failure_scores(
        scores,
        target_indices=torch.tensor([0, 0, 1, 2]),
        terminal_mask=torch.tensor([True, True, True, False]),
        failure_values=torch.tensor([1.0, 0.0, 1.5, 1.0]),
        decay=0.8,
    )

    assert result.tolist() == pytest.approx([0.1, 0.3, 0.0])
