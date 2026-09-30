import numpy as np
import pytest
import torch

from grasp_planning.real_franka.core import Catalog, live_to_training, rotation, validate_execute_config
from grasp_planning.real_franka.ros_control import check_state
from grasp_planning.rl.zed_mini import load_zed_profile


def test_ray_crop_and_optical_depth():
    p = load_zed_profile()
    h, w = 376, 672
    rgb = np.zeros((h, w, 3), dtype="uint8")
    rgb[..., 0] = 255
    depth = np.full((h, w), 0.25, dtype="float32")
    c = dict(width=w, height=h, fx=338.6, fy=338.6, cx=332.0, cy=184.0)
    packed, valid = live_to_training(rgb, depth, c, p)
    assert packed.shape == (1, 72, 128, 4)
    assert torch.all(valid)
    torch.testing.assert_close(packed[..., 0], torch.ones_like(packed[..., 0]))
    torch.testing.assert_close(packed[..., 3], torch.full_like(packed[..., 3], (0.25 - 0.1) / 0.9))


def test_invalid_depth_cannot_become_near_object():
    p = load_zed_profile()
    rgb = np.zeros((376, 672, 3), dtype="uint8")
    depth = np.full((376, 672), np.nan, dtype="float32")
    c = dict(width=672, height=376, fx=338.6, fy=338.6, cx=332.0, cy=184.0)
    packed, valid = live_to_training(rgb, depth, c, p)
    assert not valid.any()
    assert torch.isfinite(packed).all()
    assert (packed[..., 3] == 1).all()


def test_outside_fov_invalid():
    p = load_zed_profile()
    c = dict(width=672, height=376, fx=338.6, fy=338.6, cx=10000.0, cy=184.0)
    packed, valid = live_to_training(
        np.zeros((376, 672, 3), dtype="uint8"), np.full((376, 672), 0.3, dtype="float32"), c, p
    )
    assert not valid.any()
    assert (packed[..., 3] == 1).all()


def test_rotation_is_wxyz():
    np.testing.assert_allclose(rotation([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]) @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-7)
    with pytest.raises(ValueError):
        rotation([0, 0, 0, 0])


def test_policy_rejects_old_effort_or_partial_controller():
    from types import SimpleNamespace

    from grasp_planning.real_franka.ros_control import validate_velocity_controller

    controller = SimpleNamespace(name="fr3_arm_controller", state="active", claimed_interfaces=[])
    for interfaces in ([], [f"fr3_joint{i}/effort" for i in range(1, 8)], ["fr3_joint1/velocity"]):
        controller.claimed_interfaces = interfaces
        with pytest.raises(RuntimeError, match="restart"):
            validate_velocity_controller([controller])
    controller.claimed_interfaces = [f"fr3_joint{i}/velocity" for i in range(1, 8)]
    validate_velocity_controller([controller])
    controller.state = "inactive"
    with pytest.raises(RuntimeError):
        validate_velocity_controller([controller])


def test_startup_waits_for_gripper_after_arm_is_ready(monkeypatch):
    from grasp_planning.real_franka import ros_control

    clock = [100.0]
    monkeypatch.setattr(ros_control.time, "monotonic", lambda: clock[0])
    control = ros_control.RosControl.__new__(ros_control.RosControl)
    control.width = None
    control.width_received = 0
    control.state = lambda: {"sample_time": clock[0]}

    def spin():
        clock[0] += 0.1
        if clock[0] >= 100.15:
            control.width = 0.048
            control.width_received = clock[0]

    control.spin = spin
    result = control.wait(seconds=1)
    assert result["sample_time"] >= 100.15
    control.check_width(0.048)
    with pytest.raises(RuntimeError, match="training aperture"):
        control.check_width(0.060)
    clock[0] += 0.31
    with pytest.raises(RuntimeError, match="No fresh gripper"):
        control.check_width(0.048)


@pytest.mark.parametrize("width", [None, 0.048])
def test_startup_rejects_missing_or_stale_gripper(monkeypatch, width):
    from grasp_planning.real_franka import ros_control

    clock = [100.0]
    monkeypatch.setattr(ros_control.time, "monotonic", lambda: clock[0])
    control = ros_control.RosControl.__new__(ros_control.RosControl)
    control.width = width
    control.width_received = 90.0
    control.state = lambda: {}
    control.spin = lambda: clock.__setitem__(0, clock[0] + 0.1)
    with pytest.raises(RuntimeError, match="No fresh gripper"):
        control.wait(seconds=0.5)


def test_invalid_gripper_sample_does_not_refresh_feedback():
    from types import SimpleNamespace

    from grasp_planning.real_franka.ros_control import RosControl

    control = RosControl.__new__(RosControl)
    control.width, control.width_received = 0.048, 10.0
    control._gripper(SimpleNamespace(position=[np.nan, 0.024]))
    assert control.width == 0.048 and control.width_received == 10.0


def state():
    return dict(
        position=np.array([0.45, 0, 0.2]),
        rotation=np.eye(3),
        force=np.zeros(3),
        torque=np.zeros(3),
        robot_mode=1,
        collision=False,
    )


def cfg():
    return dict(
        robot_model="fr3",
        mount_confirmed=True,
        workspace_min_m=[0.2, -0.5, 0.05],
        workspace_max_m=[0.8, 0.5, 0.8],
        max_displacement_m=0.03,
        max_rotation_rad=0.2,
        max_linear_speed_m_s=0.01,
        max_angular_speed_rad_s=0.06,
        max_duration_s=15,
        external_force_stop_n=10,
        external_torque_stop_nm=3,
        minimum_valid_depth_fraction=0.2,
        maximum_frame_age_s=0.15,
        hand_from_flange_position_m=[0, 0, 0],
        hand_from_flange_wxyz=[1, 0, 0, 0],
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("collision", True),
        ("robot_mode", 3),
        ("force", np.array([11, 0, 0])),
        ("torque", np.array([0, np.nan, 0])),
        ("position", np.array([0.49, 0, 0.2])),
        ("rotation", rotation([np.cos(0.2), 0, 0, np.sin(0.2)])),
    ],
)
def test_motion_gates(key, value):
    start = state()
    current = state()
    current[key] = value
    with pytest.raises(RuntimeError):
        check_state(current, start, cfg())


def test_external_torque_norm_trips_and_reports_actual_value():
    current = state()
    current["torque"] = np.array([2.2, 2.2, 0.0])
    with pytest.raises(RuntimeError, match=r"External torque limit: 3.11 Nm \(limit 3.00 Nm\)"):
        check_state(current, state(), cfg())


def test_config_requires_mount_and_valid_limits():
    c = cfg()
    validate_execute_config(c)
    c["mount_confirmed"] = False
    with pytest.raises(ValueError):
        validate_execute_config(c)
    c = cfg()
    c["max_linear_speed_m_s"] = float("nan")
    with pytest.raises(ValueError):
        validate_execute_config(c)


def test_diversity_rotations_not_just_first_ids():
    c = Catalog.__new__(Catalog)
    poses = np.zeros((20, 7))
    poses[:, 3] = 1
    poses[:, 0] = np.arange(20) * 0.001
    c.data = {"goal_poses": poses, "jaw_widths": np.full(20, 0.03)}
    ids = c.diverse(np.arange(20), 4)
    assert len(set(ids)) == 4 and 19 in ids


@pytest.fixture(scope="module")
def workbench(tmp_path_factory):
    from types import SimpleNamespace

    from grasp_planning.real_franka.core import DEFAULT_CATALOG, DEFAULT_CHECKPOINT
    from grasp_planning.real_franka.web_app import create_app

    if not DEFAULT_CATALOG.exists():
        pytest.skip("Local training artifacts absent")
    args = SimpleNamespace(
        catalog=DEFAULT_CATALOG,
        checkpoint=DEFAULT_CHECKPOINT,
        config=tmp_path_factory.mktemp("franka") / "selection.json",
        execute=False,
        device="cpu",
    )
    camera = SimpleNamespace(latest=None, calibration=None, error=None)
    app = create_app(args, camera=camera)
    return app, app.test_client()


def test_browser_local_only_and_csrf(workbench):
    app, client = workbench
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 403
    assert client.post("/api/stop", json={}).status_code == 403
    assert client.get("/").status_code == 200
    assert b"ArrowRight" in client.get("/").data


def test_depth_snapshot_preserves_metric_values_and_invalid_pixels(workbench, monkeypatch):
    import io
    import json

    app, client = workbench
    camera = app.workbench["camera"]
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.array([[0.123456, np.nan, np.inf], [0.7, 0.8, 0.9]], dtype=np.float32)
    monkeypatch.setattr(camera, "calibration", {"depth": "float32 optical Z metres"})
    monkeypatch.setattr(camera, "frame", lambda max_age: (123.0, 42, rgb, depth), raising=False)
    response = client.get("/camera/snapshot.npz")
    assert response.status_code == 200
    with np.load(io.BytesIO(response.data), allow_pickle=False) as snapshot:
        np.testing.assert_array_equal(snapshot["rgb"], rgb)
        np.testing.assert_array_equal(snapshot["depth_m"], depth)
        assert snapshot["depth_m"].dtype == np.float32
        assert snapshot["frame_seq"].item() == 42
        assert json.loads(snapshot["calibration_json"].item()) == camera.calibration

    def stale(max_age):
        raise RuntimeError("Camera frame is stale")

    monkeypatch.setattr(camera, "frame", stale)
    assert client.get("/camera/snapshot.npz").status_code == 400


def test_browser_selection_and_no_motion_gate(workbench):
    app, client = workbench
    headers = {"X-Session-Token": app.workbench["token"]}
    catalog = client.get("/api/catalog").json
    assert len(catalog["parts"]) == 5
    i = catalog["parts"][0]["orientations"][0]["diverse"][0]
    assert client.post("/api/select", json={"index": i}, headers=headers).status_code == 200
    assert client.post("/api/save", json={"mount_confirmed": False}, headers=headers).status_code == 200
    assert client.get("/image/" + str(i)).mimetype == "image/jpeg"
    assert client.get("/shape/" + str(i)).mimetype == "image/jpeg"
    assert client.post("/api/start", json={"execute": True}, headers=headers).status_code == 400
    assert client.post("/api/gripper", json={"confirmed": True}, headers=headers).status_code == 400
    assert app.workbench["state"]["worker"] is None


def test_torque_stop_disables_further_physical_starts(workbench, monkeypatch):
    from grasp_planning.real_franka import runner

    app, client = workbench
    headers = {"X-Session-Token": app.workbench["token"]}

    def stopped_run(cfg, camera, stop, report, **kwargs):
        report("blocked: External torque limit: 3.20 Nm (limit 3.00 Nm)")

    monkeypatch.setattr(runner, "run", stopped_run)
    index = client.get("/api/catalog").json["parts"][0]["orientations"][0]["diverse"][0]
    assert client.post("/api/select", json={"index": index}, headers=headers).status_code == 200
    assert client.post("/api/mode", json={"confirmed": True, "execute": True}, headers=headers).status_code == 200
    assert client.post("/api/save", json={"mount_confirmed": True}, headers=headers).status_code == 200
    assert client.post("/api/start", json={"execute": True}, headers=headers).status_code == 200
    app.workbench["state"]["worker"].join(timeout=2)
    assert not client.get("/api/state").json["execute_enabled"]
    assert client.post("/api/start", json={"execute": True}, headers=headers).status_code == 400


def test_requested_fifteen_cm_displacement_limit():
    c = cfg()
    c["max_displacement_m"] = 0.15
    validate_execute_config(c)
    start = state()
    current = state()
    current["position"] = start["position"] + np.array([0.149, 0, 0])
    check_state(current, start, c)
    current["position"] = start["position"] + np.array([0.151, 0, 0])
    with pytest.raises(RuntimeError, match="Session displacement limit"):
        check_state(current, start, c)
    c["max_displacement_m"] = 0.151
    with pytest.raises(ValueError, match="max_displacement_m"):
        validate_execute_config(c)


@pytest.mark.parametrize("duration", [None, 120.0])
def test_policy_allows_unlimited_or_explicit_duration(duration):
    c = cfg()
    c["max_duration_s"] = duration
    validate_execute_config(c)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_policy_duration_rejected(duration):
    c = cfg()
    c["max_duration_s"] = duration
    with pytest.raises(ValueError, match="max_duration_s"):
        validate_execute_config(c)


def test_five_nm_guard_keeps_collision_reflex_and_force_stops():
    settings = cfg()
    settings["external_torque_stop_nm"] = 5.0
    validate_execute_config(settings)
    current = state()
    current["torque"] = np.array([-2.7440786, -1.1736495, -0.3968830])
    check_state(current, state(), settings)
    current["torque"] = np.array([5.01, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="External torque limit"):
        check_state(current, state(), settings)
    current["torque"] = np.zeros(3)
    current["collision"] = True
    with pytest.raises(RuntimeError, match="Franka collision flag"):
        check_state(current, state(), settings)
    current["collision"] = False
    current["robot_mode"] = 4
    with pytest.raises(RuntimeError, match="not idle/moving"):
        check_state(current, state(), settings)
    current["robot_mode"] = 1
    current["force"] = np.array([10.01, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="External force limit"):
        check_state(current, state(), settings)
