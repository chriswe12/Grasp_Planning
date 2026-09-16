from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("rl_games.algos_torch.network_builder")

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/agents/resnet_rgbd_network.py"
)
SPEC = importlib.util.spec_from_file_location("resnet_rgbd_network", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
resnet_rgbd_network = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = resnet_rgbd_network
SPEC.loader.exec_module(resnet_rgbd_network)


def _make_network():
    params = {
        "pretrained": False,
        "image_height": 72,
        "image_width": 128,
        "image_channels": 8,
        "policy_context_size": 6,
        "pose_target_size": 6,
        "completion_target_size": 2,
        "motion_action_size": 6,
        "geometry_feature_size": 128,
        "completion_motion_slowdown_start": 0.70,
        "completion_motion_speed_floor": 0.25,
        "space": {"continuous": {"sigma_init": {"val": -1.5}}},
    }
    return resnet_rgbd_network.GraspRgbdResNetNetwork(
        params,
        actions_num=7,
        input_shape=(72 * 128 * 8 + 14,),
        value_size=1,
    ).eval()


def test_privileged_labels_cannot_change_policy_outputs() -> None:
    torch.manual_seed(13)
    network = _make_network()
    observation = torch.rand(1, 72 * 128 * 8 + 14)
    changed_labels = observation.clone()
    changed_labels[:, -8:] = torch.rand_like(changed_labels[:, -8:]) * 20.0 - 10.0

    with torch.inference_mode():
        baseline = network({"obs": observation, "is_train": False})
        changed = network({"obs": changed_labels, "is_train": False})

    for baseline_value, changed_value in zip(baseline[:4], changed[:4], strict=True):
        torch.testing.assert_close(baseline_value, changed_value, rtol=0.0, atol=0.0)


def test_previous_action_context_affects_motion_but_not_visual_completion() -> None:
    torch.manual_seed(17)
    network = _make_network()
    observation = torch.rand(1, 72 * 128 * 8 + 14)
    changed_context = observation.clone()
    context_start = 72 * 128 * 8
    changed_context[:, context_start : context_start + 6] += 1.0

    with torch.inference_mode():
        baseline = network({"obs": observation, "is_train": False})
        changed = network({"obs": changed_context, "is_train": False})

    assert not torch.equal(baseline[0], changed[0])
    torch.testing.assert_close(baseline[2], changed[2], rtol=0.0, atol=0.0)


def test_all_shared_heads_receive_gradients() -> None:
    torch.manual_seed(19)
    network = _make_network().train()
    observation = torch.rand(2, 72 * 128 * 8 + 14)
    observation[:, -2] = torch.tensor([0.0, 1.0])
    observation[:, -1] = 1.0

    motion, _, _, value, _ = network({"obs": observation, "is_train": True})
    losses = network.get_aux_loss()
    total = motion.square().mean() + value.square().mean() + losses["pose_aux_loss"] + losses["completion_aux_loss"]
    total.backward()

    for layer in (
        network.geometry_trunk[0],
        network.motion_head[-1],
        network.pose_head[-1],
        network.completion_head[-1],
    ):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
