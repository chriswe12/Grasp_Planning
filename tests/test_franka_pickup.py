import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_planning.real_franka import pickup as module


@pytest.fixture
def rig(monkeypatch, tmp_path):
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    cfg = dict(
        robot_model="fr3",
        mount_confirmed=True,
        max_linear_speed_m_s=0.01,
        max_angular_speed_rad_s=0.06,
        max_displacement_m=0.15,
        max_rotation_rad=0.2,
        max_duration_s=25,
        external_force_stop_n=10,
        external_torque_stop_nm=3,
        minimum_valid_depth_fraction=0.2,
        maximum_frame_age_s=0.15,
        manual_gripper_force_n=10,
        workspace_min_m=[0.15, -0.65, 0.02],
        workspace_max_m=[0.85, 0.65, 0.85],
    )
    state = dict(
        position=np.array([0.4, 0.0, 0.3]),
        rotation=np.eye(3),
        force=np.zeros(3),
        torque=np.zeros(3),
        joint_velocity=np.zeros(7),
        robot_mode=1,
        collision=False,
    )
    events = []

    class Stop:
        stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, dt):
            clock[0] += dt
            if control.active:
                state["position"] += control.twist[:3] * dt
            hook()

    stop = Stop()

    def hook():
        pass

    class Control:
        width = 0.04
        active = False
        twist = np.zeros(6)

        def state(self):
            return copy.deepcopy(state)

        wait = state

        def require_fresh_aperture(self):
            pass

        def require_velocity_controller(self):
            events.append("controller")

        def require_single_command_sources(self):
            events.append("sources")

        def activate(self):
            self.active = True
            events.append("activate")

        def send(self, twist):
            self.twist = twist.copy()
            events.append(twist.copy())

        def stop(self):
            self.active = False
            events.append("stop")

        def close(self):
            self.active = False
            events.append("closed")

    control = Control()
    monkeypatch.setattr(module, "RosControl", lambda: control)

    def grasp(cfg, width, *, stop_event, monitor):
        events.append("grasp")
        monitor()

    monkeypatch.setattr(module, "close_gripper", grasp)
    reports = []
    log = tmp_path / "pickup.json"

    def run():
        return module.pickup(cfg, 0.04, stop, reports.append, log)

    def set_hook(fn):
        nonlocal hook
        hook = fn

    return SimpleNamespace(
        cfg=cfg,
        state=state,
        control=control,
        stop=stop,
        events=events,
        reports=reports,
        run=run,
        log=log,
        set_hook=set_hook,
    )


def commands(rig):
    return [e for e in rig.events if isinstance(e, np.ndarray)]


def test_lift_is_measured_20cm_base_z_and_preserves_policy_limits(rig):
    before = copy.deepcopy(rig.cfg)
    assert rig.run() == "lift_complete_grasp_unverified"
    assert 0.498 <= rig.state["position"][2] <= 0.500
    np.testing.assert_allclose(rig.state["position"][:2], [0.4, 0])
    twists = np.asarray(commands(rig))
    assert np.max(twists[:, 2]) <= 0.01
    assert np.min(twists[:, 2]) >= 0
    assert np.all(twists[:, [0, 1, 3, 4, 5]] == 0)
    assert rig.cfg == before
    assert rig.events.index("grasp") < rig.events.index("activate")
    assert rig.events[-1] == "closed"
    assert json.loads(rig.log.read_text())["outcome"] == "lift_complete_grasp_unverified"


@pytest.mark.parametrize(
    "fault, message",
    [
        ("workspace", "endpoint outside"),
        ("moving", "stationary"),
        ("force", "External force limit"),
        ("torque", "External torque limit"),
        ("collision", "collision flag"),
        ("mode", "not idle/moving"),
    ],
)
def test_preflight_faults_never_close_or_move(rig, fault, message):
    if fault == "workspace":
        rig.state["position"][2] = 0.70
    if fault == "moving":
        rig.state["joint_velocity"][0] = 0.1
    if fault == "force":
        rig.state["force"][0] = 11
    if fault == "torque":
        rig.state["torque"][0] = 4
    if fault == "collision":
        rig.state["collision"] = True
    if fault == "mode":
        rig.state["robot_mode"] = 4
    assert message in rig.run()
    assert "grasp" not in rig.events and not commands(rig)
    assert rig.events[-1] == "closed"


@pytest.mark.parametrize("fault", ["grasp_failed", "stop_during_close", "empty", "stale"])
def test_no_lift_without_confirmed_grasp(rig, monkeypatch, fault):
    def grasp(*args, **kwargs):
        if fault == "grasp_failed":
            raise RuntimeError("grasp failed")
        if fault == "stop_during_close":
            rig.stop.set()
        if fault == "empty":
            rig.control.width = 0.0
        if fault == "stale":
            rig.control.require_fresh_aperture = lambda: (_ for _ in ()).throw(RuntimeError("stale aperture"))

    monkeypatch.setattr(module, "close_gripper", grasp)
    assert rig.run() != "lift_complete_grasp_unverified"
    assert "activate" not in rig.events and not commands(rig)
    assert rig.events[-1] == "closed"


@pytest.mark.parametrize(
    "fault, message",
    [
        ("stop", "operator_stop"),
        ("sideways", "corridor"),
        ("slip", "aperture changed"),
        ("force", "External force limit"),
        ("stall", "stalled"),
        ("stale", "Stale TCP"),
        ("rotation", "rotation limit"),
    ],
)
def test_lift_interruptions_stop_and_never_release(rig, fault, message):
    def fault_hook():
        if not rig.control.active:
            return
        if fault == "stop":
            rig.stop.set()
        if fault == "sideways":
            rig.state["position"][0] += 0.01
        if fault == "slip":
            rig.control.width += 0.01
        if fault == "force":
            rig.state["force"][0] = 11
        if fault == "stall":
            rig.state["position"][2] = 0.3
        if fault == "stale":
            rig.control.state = lambda: (_ for _ in ()).throw(RuntimeError("Stale TCP"))
        if fault == "rotation":
            rig.state["rotation"] = np.diag([1.0, -1.0, -1.0])

    rig.set_hook(fault_hook)
    assert message in rig.run()
    assert rig.events[-1] == "closed"
    assert sum(isinstance(e, str) and e == "grasp" for e in rig.events) == 1
    assert len(commands(rig)) > 0


def test_pickup_api_single_click_requires_execute_saved_mount_and_idle(monkeypatch, tmp_path):
    from grasp_planning.real_franka import web_app
    from grasp_planning.real_franka.core import DEFAULT_CATALOG, DEFAULT_CHECKPOINT

    if not DEFAULT_CATALOG.exists():
        pytest.skip("Local catalog absent")
    args = SimpleNamespace(
        catalog=DEFAULT_CATALOG,
        checkpoint=DEFAULT_CHECKPOINT,
        config=tmp_path / "selection.json",
        execute=False,
        device="cpu",
    )
    app = web_app.create_app(args, camera=SimpleNamespace(latest=None, calibration=None, error=None))
    client = app.test_client()
    headers = {"X-Session-Token": app.workbench["token"]}
    body = {}
    calls = []
    from grasp_planning.real_franka import runner

    monkeypatch.setattr(runner, "run", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(web_app, "ROOT", tmp_path)
    assert client.post("/api/pickup", json=body).status_code == 403
    assert client.post("/api/pickup", json=body, headers=headers).status_code == 400
    index = client.get("/api/catalog").json["targets"][0]["index"]
    assert client.post("/api/select", json={"index": index}, headers=headers).status_code == 200
    assert client.post("/api/save", json={"mount_confirmed": False}, headers=headers).status_code == 200
    args.execute = True
    assert client.post("/api/pickup", json=body, headers=headers).status_code == 400
    assert client.post("/api/save", json={"mount_confirmed": True}, headers=headers).status_code == 200
    app.workbench["state"]["worker"] = SimpleNamespace(is_alive=lambda: True)
    assert client.post("/api/pickup", json=body, headers=headers).status_code == 400
    from grasp_planning.real_franka.handoff import PickupHandoff

    handoff = PickupHandoff()
    app.workbench["state"]["handoff"] = handoff
    handoff.policy_started()
    assert client.get("/api/state").json["pickup_available"]
    assert client.post("/api/pickup", json={}, headers=headers).json["queued"]
    assert not client.get("/api/state").json["pickup_available"]
    assert client.post("/api/pickup", json={}, headers=headers).status_code == 400
    assert handoff.requested()
    handoff.finish_policy()
    app.workbench["state"]["worker"] = None
    assert not calls
    assert client.post("/api/pickup", json=body, headers=headers).status_code == 200
    app.workbench["state"]["worker"].join(2)
    assert len(calls) == 1
    assert calls[0][0][2] is app.workbench["stop"]
    assert "franka_real_runs" in str(calls[0][1]["log_path"])
    assert calls[0][1]["pickup_only"]
    client.post("/api/stop", json={}, headers=headers)
    assert app.workbench["stop"].is_set()


@pytest.mark.parametrize("mode", ["success", "stop", "monitor_error", "aborted"])
def test_gripper_success_and_cancel_path(monkeypatch, mode):
    import sys
    import threading

    from grasp_planning.real_franka.ros_control import close_gripper

    stop = threading.Event()
    events = []
    result = SimpleNamespace(
        done=lambda: mode in ("success", "aborted"),
        result=lambda: SimpleNamespace(status=6 if mode == "aborted" else 4, result=SimpleNamespace(success=True)),
    )

    def cancel():
        events.append("cancel")
        return SimpleNamespace(done=lambda: True)

    handle = SimpleNamespace(accepted=True, get_result_async=lambda: result, cancel_goal_async=cancel)

    def send_goal(goal):
        events.append("goal")
        if mode == "stop":
            stop.set()
        return SimpleNamespace(done=lambda: True, result=lambda: handle)

    client = SimpleNamespace(wait_for_server=lambda **kw: True, send_goal_async=send_goal)
    node = SimpleNamespace(destroy_node=lambda: events.append("destroy"))
    rclpy = SimpleNamespace(
        ok=lambda: True,
        create_node=lambda name: node,
        spin_until_future_complete=lambda *a, **kw: None,
        spin_once=lambda *a, **kw: None,
    )
    action = SimpleNamespace(Goal=lambda: SimpleNamespace(epsilon=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.action", SimpleNamespace(ActionClient=lambda *a: client))
    monkeypatch.setitem(sys.modules, "franka_msgs.action", SimpleNamespace(Grasp=action, Move=action))

    def monitor():
        if mode == "monitor_error" and "goal" in events:
            raise RuntimeError("External force limit")

    if mode == "success":
        close_gripper({"manual_gripper_force_n": 10}, 0.04, stop_event=stop, monitor=monitor)
        assert "cancel" not in events
    else:
        with pytest.raises(RuntimeError):
            close_gripper({"manual_gripper_force_n": 10}, 0.04, stop_event=stop, monitor=monitor)
        assert "cancel" in events
    assert events[-1] == "destroy"


def test_handoff_requests_are_single_use_and_cannot_arrive_after_finish():
    from grasp_planning.real_franka.handoff import PickupHandoff

    handoff = PickupHandoff()
    assert not handoff.request()
    handoff.policy_started()
    assert handoff.available() and handoff.request()
    assert not handoff.available() and not handoff.request()
    assert handoff.finish_policy()
    assert not handoff.request()
    late = PickupHandoff()
    late.policy_started()
    assert not late.finish_policy() and not late.request()


@pytest.mark.parametrize("stop_at_handoff", [False, True])
@pytest.mark.parametrize("policy_seconds", [0, 35])
def test_policy_handoff_uses_one_control_and_one_recording_with_tail(rig, monkeypatch, stop_at_handoff, policy_seconds):
    import torch

    from grasp_planning.real_franka import recording, runner
    from grasp_planning.real_franka.handoff import PickupHandoff

    rig.cfg.update(
        catalog="test",
        checkpoint="test",
        catalog_sha256="hash",
        checkpoint_sha256="hash",
        target_id="part",
        camera_serial=123,
    )
    rig.state.update(
        tcp_transform_age_s=0.0,
        joints=[0.0] * 7,
        desired_joint_velocity=[0.0] * 7,
        desired_joint_acceleration=[0.0] * 7,
        control_command_success_rate=1.0,
        controller_feedback=None,
        controller_feedback_age_s=None,
    )
    rig.control.check_width = lambda width: None
    rig.control.now = lambda: 0.0
    rig.control.sink = SimpleNamespace(health=lambda **kw: SimpleNamespace(status_code=0, status_text="ok"))
    created = []

    def control_factory():
        created.append(rig.control)
        return rig.control

    monkeypatch.setattr(runner, "RosControl", control_factory)
    cat = SimpleNamespace(
        index=lambda target: 0,
        hash="hash",
        profile={"quaternion_wxyz": [1, 0, 0, 0]},
        data={"open_widths": [0.04], "jaw_widths": [0.04]},
        contract={
            "training_recipe": {
                "source_settings": {
                    "completion_required_consecutive_steps": 4,
                    "completion_probability_threshold": 0.9,
                    "completion_max_linear_speed_m_s": 0.001,
                    "completion_max_angular_speed_rad_s": 0.01,
                    "action_delta_limit": 0.5,
                }
            }
        },
    )
    monkeypatch.setattr(runner, "Catalog", lambda path: cat)
    monkeypatch.setattr(runner, "sha256", lambda path: "hash")
    actor = SimpleNamespace(
        goal=torch.zeros((1, 4, 6, 4)),
        set_target=lambda *a: None,
        infer=lambda live: np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
    )
    monkeypatch.setattr(runner, "Actor", lambda *a, **kw: actor)
    monkeypatch.setattr(runner, "live_to_training", lambda *a: (torch.zeros((1, 4, 6, 4)), torch.ones((4, 6))))
    frame_sequence = [0]

    def frame(*args):
        frame_sequence[0] += 1
        return (module.time.monotonic(), frame_sequence[0], np.zeros((4, 6, 3), np.uint8), np.ones((4, 6)))

    camera = SimpleNamespace(latest=True, error=None, calibration={"serial": 123}, frame=frame)
    captures = []

    class Recorder:
        def __init__(self, *args):
            self.frames = []
            self.finished = 0
            self.expanded = False
            captures.append(self)

        def allow_pickup(self):
            self.expanded = True

        def append(self, frame, live, step):
            self.frames.append(step)

        def finish(self):
            assert not rig.control.active
            self.finished += 1
            return {"frames": len(self.frames), "video": "combined.mp4"}

    monkeypatch.setattr(recording, "RunRecording", Recorder)
    handoff = PickupHandoff()
    requested = []
    began = module.time.monotonic()
    if policy_seconds:
        rig.cfg["max_duration_s"] = None

    def click():
        if not requested and module.time.monotonic() - began >= policy_seconds:
            assert handoff.request()
            requested.append(True)
            if stop_at_handoff:
                rig.stop.set()

    rig.set_hook(click)
    outcome = runner.run(
        rig.cfg, camera, rig.stop, lambda *a: None, execute=True, device="cpu", log_path=rig.log, handoff=handoff
    )
    assert requested
    if policy_seconds:
        assert captures[0].frames[-1]["time_s"] >= policy_seconds - 1 / 15
    assert len(created) == len(captures) == captures[0].finished == 1
    assert not handoff.available() and not handoff.request()
    if stop_at_handoff:
        assert outcome == "operator_stop"
        assert not captures[0].expanded
        assert not any(isinstance(e, str) and e == "grasp" for e in rig.events)
    else:
        assert outcome == "lift_complete_grasp_unverified"
        steps = captures[0].frames
        phases = [s["phase"] for s in steps]
        assert phases[0] == "policy"
        assert all(phase in phases for phase in ["settling", "closing", "lifting", "post_lift"])
        tail = [s for s in steps if s["phase"] == "post_lift"]
        assert len(tail) >= 14
        assert 0.85 <= tail[-1]["time_s"] - tail[0]["time_s"] <= 1.01
        assert all(s["base_tcp_twist"] == [0.0] * 6 for s in tail)
        assert all(a["time_s"] < b["time_s"] for a, b in zip(steps, steps[1:]))
        saved = json.loads(rig.log.read_text())
        assert saved["pickup"]["outcome"] == outcome and saved["recording"]["video"]
