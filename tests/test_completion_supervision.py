from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/completion.py"
)
SPEC = importlib.util.spec_from_file_location("completion_reward_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
completion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = completion
SPEC.loader.exec_module(completion)


def test_operational_completion_quality_has_plateau_taper_and_reject_regions() -> None:
    position = torch.tensor([0.004, 0.007, 0.0095, 0.012, 0.006])
    rotation = torch.deg2rad(torch.tensor([3.0, 5.0, 7.5, 10.0, 4.0]))
    collision_free = torch.tensor([True, True, True, True, False])

    quality = completion.completion_quality(
        position,
        rotation,
        ready_position_m=0.007,
        ready_rotation_rad=torch.deg2rad(torch.tensor(5.0)).item(),
        negative_position_m=0.012,
        negative_rotation_rad=torch.deg2rad(torch.tensor(10.0)).item(),
        collision_free=collision_free,
    )

    assert torch.allclose(quality, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]), atol=1.0e-6)


def test_graded_terminal_reward_is_continuous_across_operational_band() -> None:
    reward = completion.graded_completion_terminal_reward(
        torch.tensor([True, True, True, False]),
        torch.tensor([1.0, 0.5, 0.0, 1.0]),
        correct_reward=50.0,
        premature_penalty=30.0,
    )

    assert reward.tolist() == [50.0, 10.0, -30.0, 0.0]
