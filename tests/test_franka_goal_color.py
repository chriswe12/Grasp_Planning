import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grasp_planning.real_franka.core import Actor, sha256
from grasp_planning.real_franka.goal_color import load_goal, metadata, normalize_color


@pytest.fixture
def goal(tmp_path):
    original = np.zeros((72, 128, 4), dtype=np.float32)
    cat = SimpleNamespace(hash="catalog", data={"target_ids": ["a", "b"], "goal_rgbd": [original, original]})
    image = original.copy()
    image[..., 0] = 0.5
    path = tmp_path / "goal.npz"
    np.savez_compressed(path, goal_rgbd=image, metadata_json=json.dumps(metadata(cat, 0, "#884bb0")))
    cfg = {"goal_render": dict(color="#884bb0", path=str(path), sha256=sha256(path))}
    return cat, image, cfg


def test_custom_goal_is_the_actor_input(goal):
    cat, image, cfg = goal
    actor = Actor.__new__(Actor)
    actor.catalog, actor.device, actor.previous = cat, torch.device("cpu"), np.ones(6)
    actor.set_target(0, cfg)
    np.testing.assert_array_equal(actor.goal[0].numpy(), image)
    assert not actor.previous.any()
    actor.set_target(0)
    np.testing.assert_array_equal(actor.goal[0].numpy(), cat.data["goal_rgbd"][0])


def test_wrong_pose_or_modified_render_cannot_be_used(goal):
    cat, image, cfg = goal
    with pytest.raises(ValueError, match="different target"):
        load_goal(cat, 1, cfg)
    cfg["goal_render"]["color"] = "#ff0000"
    with pytest.raises(ValueError, match="different target"):
        load_goal(cat, 0, cfg)
    cfg["goal_render"]["sha256"] = "invalid"
    with pytest.raises(ValueError, match="changed"):
        load_goal(cat, 0, cfg)


def test_color_cannot_change_canonical_depth(goal):
    cat, image, cfg = goal
    image[..., 3] = 0.2
    path = cfg["goal_render"]["path"]
    np.savez_compressed(path, goal_rgbd=image, metadata_json=json.dumps(metadata(cat, 0, "#884bb0")))
    cfg["goal_render"]["sha256"] = sha256(path)
    with pytest.raises(ValueError, match="original training depth"):
        load_goal(cat, 0, cfg)


@pytest.mark.parametrize("value", ["red", "#fff", "#ff0000; exit", "#00000g", [1, 0, 0]])
def test_invalid_color(value):
    with pytest.raises(ValueError):
        normalize_color(value)


def test_color_web_save_reload_and_selection_reset(tmp_path, monkeypatch):
    from grasp_planning.real_franka import web_app
    from grasp_planning.real_franka.core import DEFAULT_CATALOG, DEFAULT_CHECKPOINT, Catalog

    if not DEFAULT_CATALOG.exists():
        pytest.skip("Local catalog absent")
    args = SimpleNamespace(
        catalog=DEFAULT_CATALOG,
        checkpoint=DEFAULT_CHECKPOINT,
        config=tmp_path / "selection.json",
        execute=False,
        device="cpu",
    )
    camera = SimpleNamespace(latest=None, calibration=None, error=None)
    app = web_app.create_app(args, camera=camera)
    client = app.test_client()
    headers = {"X-Session-Token": app.workbench["token"]}
    cat = Catalog()
    index = next(i for i in cat.indices() if str(cat.data["part_keys"][i]).startswith("plumbers_block__"))
    client.post("/api/select", json={"index": int(index)}, headers=headers)
    client.post("/api/save", json={"mount_confirmed": True}, headers=headers)
    app.workbench["state"]["config"]["max_linear_speed_m_s"] = 0.007
    original = client.get("/goal").data
    entered, release = threading.Event(), threading.Event()

    def render(cat, i, color, stop, report):
        entered.set()
        assert release.wait(5)
        image = cat.data["goal_rgbd"][i].copy()
        image[..., :3] = [0.5, 0, 0.7]
        path = tmp_path / "goal.npz"
        np.savez_compressed(path, goal_rgbd=image, metadata_json=json.dumps(metadata(cat, i, color)))
        return dict(color=color, path=str(path), sha256=sha256(path))

    monkeypatch.setattr(web_app, "render_goal", render)
    assert client.post("/api/color", json={"color": "purple"}, headers=headers).status_code == 400
    assert client.post("/api/color", json={"color": "#884bb0"}, headers=headers).status_code == 200
    assert entered.wait(2)
    for endpoint, payload in [
        ("/api/save", {}),
        ("/api/start", {"execute": False}),
        ("/api/select", {"index": int(index)}),
        ("/api/gripper", {"confirmed": True}),
    ]:
        assert client.post(endpoint, json=payload, headers=headers).status_code == 400
    release.set()
    app.workbench["state"]["worker"].join(5)
    status = client.get("/api/state").json
    assert status["goal_color"] == "#884bb0" and not status["saved"]
    custom = client.get("/goal").data
    assert custom != original
    assert client.post("/api/save", json={"mount_confirmed": True}, headers=headers).status_code == 200
    cfg = json.loads(args.config.read_text())
    assert cfg["max_linear_speed_m_s"] == 0.007
    np.testing.assert_array_equal(load_goal(cat, index, cfg), app.workbench["state"]["goal_image"])
    reloaded = web_app.create_app(args, camera=camera).test_client()
    assert reloaded.get("/api/state").json["goal_color"] == "#884bb0"
    assert reloaded.get("/goal").data == custom
    client.post("/api/color", json={"color": None}, headers=headers)
    assert client.get("/goal").data == original
    client.post("/api/save", json={"mount_confirmed": True}, headers=headers)
    assert "goal_render" not in json.loads(args.config.read_text())
    client.post("/api/select", json={"index": int(index)}, headers=headers)
    assert client.get("/api/state").json["goal_color"] is None
    assert not client.get("/api/state").json["saved"]


def test_cached_mixed_goal_selection_is_explicit_and_fixed(goal):
    cat, _, _ = goal
    cat.data["goal_rgbd_variants"] = np.zeros((2, 4, 72, 128, 4), np.float16)
    cat.data["goal_rgbd_variants"][1, 2] = 0.25
    np.testing.assert_array_equal(load_goal(cat, 1, {"goal_variant_index": 3}), 0.25)
    assert load_goal(cat, 1, {}).max() == 0
    with pytest.raises(ValueError, match="either"):
        load_goal(cat, 1, {"goal_variant_index": 3, "goal_render": {"path": "unused"}})
    with pytest.raises(ValueError, match="0..4"):
        load_goal(cat, 1, {"goal_variant_index": -1})
