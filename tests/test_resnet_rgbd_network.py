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


def _make_network(**overrides):
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
    network_class = overrides.pop("network_class", resnet_rgbd_network.GraspRgbdResNetNetwork)
    params.update(overrides)
    return network_class(
        params,
        actions_num=7,
        input_shape=(72 * 128 * 8 + 6 + params["pose_target_size"] + 2,),
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


def test_pair_only_actor_ignores_all_nonimage_values_and_uses_both_views():
    torch.manual_seed(23)
    network = _make_network(visual_fusion="paired", use_policy_context=False)
    assert network.spatial_fusion[0].in_channels == 2 * (256 + 128)
    assert network.motion_head[0].in_features == 256 + 128
    observation = torch.rand(1, 72 * 128 * 8 + 14)
    changed = observation.clone()
    changed[:, 72 * 128 * 8 :] = float("nan")
    with torch.inference_mode():
        original = network({"obs": observation, "is_train": False})
        ignored = network({"obs": changed, "is_train": False})
        for a, b in zip(original[:4], ignored[:4], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for channels in (slice(0, 4), slice(4, 8)):
            image_changed = observation.clone()
            image_changed[:, : 72 * 128 * 8].view(1, 72, 128, 8)[..., channels] = 0
            output = network({"obs": image_changed, "is_train": False})
            assert not torch.equal(original[0], output[0])


def test_pair_only_actor_has_no_nonimage_gradient():
    network = _make_network(visual_fusion="paired", use_policy_context=False)
    observation = torch.rand(1, 72 * 128 * 8 + 14, requires_grad=True)
    motion, _, completion, _, _ = network({"obs": observation, "is_train": False})
    (motion.sum() + completion.sum()).backward()
    assert torch.count_nonzero(observation.grad[:, -14:]) == 0
    assert torch.count_nonzero(observation.grad[:, : 72 * 128 * 8]) > 0


def test_symmetry_set_labels_do_not_enter_actor_and_auxiliary_backpropagates():
    import numpy as np

    params = dict(
        pose_target_size=24,
        visual_fusion="paired",
        use_policy_context=False,
        symmetry_aux={"orbits": [np.eye(4)[None].tolist()], "continuous": [[]], "rotation_scale": 0.5},
    )
    network = _make_network(**params)
    obs = torch.rand(2, 72 * 128 * 8 + 32)
    target = torch.zeros(2, 24)
    target[:, 3] = 1
    target[:, 10] = 1
    target[:, 14:23] = torch.eye(3).flatten()
    target[:, 7] = 0.02
    obs[:, -26:-2] = target
    obs[:, -2:] = 1
    with torch.no_grad():
        baseline = network({"obs": obs, "is_train": False})
        changed = obs.clone()
        changed[:, -26:] = torch.randn_like(changed[:, -26:]) * 3
        alternative = network({"obs": changed, "is_train": False})
    for x, y in zip(baseline[:4], alternative[:4], strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    network({"obs": obs, "is_train": True})
    loss = network.get_aux_loss()["pose_aux_loss"]
    loss.backward()
    assert torch.isfinite(loss) and network.pose_head[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(network.pose_head[-1].weight.grad).all()


def test_symmetry_inference_matches_deployment_network():
    import numpy as np

    from grasp_planning.rl.deployment_model.resnet_rgbd_network import GraspRgbdResNetNetwork

    kwargs = dict(
        pose_target_size=24,
        visual_fusion="paired",
        use_policy_context=False,
        symmetry_aux={"orbits": [np.eye(4)[None].tolist()], "continuous": [[]], "rotation_scale": 0.5},
    )
    training = _make_network(**kwargs)
    deployment = _make_network(network_class=GraspRgbdResNetNetwork, **kwargs)
    deployment.load_state_dict(training.state_dict(), strict=True)
    obs = torch.rand(1, 72 * 128 * 8 + 32)
    obs[:, -26:] = 0
    with torch.no_grad():
        a = training({"obs": obs, "is_train": False})
        b = deployment({"obs": obs, "is_train": False})
    for x, y in zip(a[:4], b[:4], strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
