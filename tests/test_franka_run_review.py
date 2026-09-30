import json
from pathlib import Path

import pytest

from grasp_planning.real_franka.core import sha256
from grasp_planning.real_franka.run_review import latest_recording, rating, review_run


@pytest.fixture
def recorded(tmp_path):
    logs = tmp_path / "runs"
    logs.mkdir()
    checkpoint, catalog = tmp_path / "policy.pth", tmp_path / "catalog.npz"
    checkpoint.write_bytes(b"checkpoint")
    catalog.write_bytes(b"catalog")
    checkpoint.with_suffix(".contract.json").write_text("{}")
    path = logs / "20260922_114107.json"
    data = dict(
        config=dict(
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint),
            catalog=str(catalog),
            catalog_sha256=sha256(catalog),
        ),
        camera={"serial": 13829658},
        recording=dict(frames=1, video="video", rgbd="frames"),
        outcome="operator_stop",
        execute=True,
        steps=[{"time_s": 0}],
    )
    path.write_text(json.dumps(data))
    path.with_suffix(".mp4").write_bytes(b"video")
    path.with_suffix(".rgbd.npz").write_bytes(b"frames")
    return path, tmp_path / "good"


def test_archive_is_complete_verified_idempotent_and_relabelable(recorded):
    path, collection = recorded
    original = path.read_bytes()
    result = review_run(path, True, collection)
    archive = Path(result["archive"])
    manifest = json.loads((archive / "manifest.json").read_text())
    assert set(manifest["files"]) == {
        "run.json",
        "video.mp4",
        "frames.rgbd.npz",
        "checkpoint",
        "catalog",
        "contract",
        "config.json",
        "camera.json",
    }
    for item in manifest["files"].values():
        assert sha256(archive / item["path"]) == item["sha256"]
    assert (archive / "run.json").read_bytes() == original
    assert latest_recording(path.parent) == path
    assert rating(path)["good_video"] is True
    assert review_run(path, True, collection)["archive"] == str(archive)
    review_run(path, False, collection)
    assert not rating(path)["good_video"] and (archive / "video.mp4").exists()
    assert not json.loads((archive / "review.json").read_text())["good_video"]
    assert path.read_bytes() == original


def test_negative_rating_does_not_copy(recorded):
    path, collection = recorded
    review_run(path, False, collection)
    assert not collection.exists()
    assert not rating(path)["good_video"]


def test_reject_incomplete_or_changed_assets_without_labeling_good(recorded):
    path, collection = recorded
    cfg = json.loads(path.read_text())["config"]
    Path(cfg["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since execution"):
        review_run(path, True, collection)
    assert not (collection / path.stem).exists() and rating(path) is None
    path.with_suffix(".mp4").unlink()
    with pytest.raises(ValueError, match="finished saving"):
        review_run(path, True, collection)
    assert latest_recording(path.parent) is None


def test_shared_assets_are_copied_only_once(recorded):
    path, collection = recorded
    review_run(path, True, collection)
    other = path.with_name("20260922_114200.json")
    for suffix in [".json", ".mp4", ".rgbd.npz"]:
        other.with_suffix(suffix).write_bytes(path.with_suffix(suffix).read_bytes())
    review_run(other, True, collection)
    assert len(list((collection / "_assets").glob("*/*"))) == 3
    assert (collection / other.stem / "video.mp4").exists()


def test_review_api_targets_explicit_run_and_survives_restart(recorded, monkeypatch):
    from types import SimpleNamespace

    from grasp_planning.real_franka import web_app
    from grasp_planning.real_franka.core import DEFAULT_CATALOG, DEFAULT_CHECKPOINT

    if not DEFAULT_CATALOG.exists():
        pytest.skip("Local catalog absent")
    path, collection = recorded
    root = path.parent.parent
    run_folder = root / "artifacts/franka_real_runs"
    run_folder.mkdir(parents=True)
    for suffix in (".json", ".mp4", ".rgbd.npz"):
        (run_folder / path.with_suffix(suffix).name).write_bytes(path.with_suffix(suffix).read_bytes())
    monkeypatch.setattr(web_app, "ROOT", root)
    args = SimpleNamespace(
        catalog=DEFAULT_CATALOG,
        checkpoint=DEFAULT_CHECKPOINT,
        config=root / "selection.json",
        execute=False,
        device="cpu",
    )
    camera = SimpleNamespace(latest=None, calibration=None, error=None)
    app = web_app.create_app(args, camera=camera)
    client = app.test_client()
    headers = {"X-Session-Token": app.workbench["token"]}
    assert client.get("/api/state").json["recording_run_id"] == path.stem
    assert client.get("/recording.mp4").status_code == 200
    for body in [{"run_id": "../../unsafe", "good": True}, {"run_id": path.stem, "good": "yes"}]:
        assert client.post("/api/recording/review", json=body, headers=headers).status_code == 400
    body = {"run_id": path.stem, "good": True}
    assert client.post("/api/recording/review", json=body).status_code == 403
    app.workbench["state"]["worker"] = SimpleNamespace(is_alive=lambda: True)
    assert client.post("/api/recording/review", json=body, headers=headers).status_code == 400
    app.workbench["state"]["worker"] = None
    assert client.post("/api/recording/review", json=body, headers=headers).status_code == 200
    app.workbench["state"]["worker"].join(5)
    assert client.get("/api/state").json["recording_review"]["good_video"]
    restarted = web_app.create_app(args, camera=camera).test_client()
    assert restarted.get("/api/state").json["recording_review"]["good_video"]
