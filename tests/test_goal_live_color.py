from __future__ import annotations

import torch

from grasp_planning.rl.goal_live_color import (
    COLOR_RELATIONSHIP_DIFFERENT,
    COLOR_RELATIONSHIP_MATCH,
    COLOR_RELATIONSHIP_SIMILAR,
    sample_goal_live_color_pairs,
)


def test_color_pair_sampler_matches_requested_mixture_and_semantics() -> None:
    torch.manual_seed(23)
    pairs = sample_goal_live_color_pairs(
        20_000,
        torch.tensor([0, 2, 3, 9, 19, 23]),
        match_fraction=0.25,
        similar_fraction=0.20,
        device="cpu",
    )

    rates = torch.bincount(pairs.relationship_codes, minlength=3).float() / 20_000
    torch.testing.assert_close(rates, torch.tensor([0.25, 0.20, 0.55]), atol=0.015, rtol=0.0)
    matched = pairs.relationship_codes == COLOR_RELATIONSHIP_MATCH
    similar = pairs.relationship_codes == COLOR_RELATIONSHIP_SIMILAR
    different = pairs.relationship_codes == COLOR_RELATIONSHIP_DIFFERENT
    assert torch.equal(pairs.goal_palette_indices[matched], pairs.live_palette_indices[matched])
    assert torch.all(pairs.goal_palette_indices[similar] != pairs.live_palette_indices[similar])
    assert torch.all(pairs.goal_palette_indices[different] != pairs.live_palette_indices[different])
