"""Explicit operator-triggered close and base-Z lift, separate from the policy."""

import json
import time
from pathlib import Path

import numpy as np

from .core import validate_execute_config
from .ros_control import RosControl, check_state, close_gripper

LIFT_M = 0.20
TOLERANCE_M = 0.002


def pickup(cfg, width, stop, report, log_path=None, *, control=None, sample=None, result_data=None):
    records = []
    reason = "operator_stop"
    phase = "preflight"
    start = None
    last_checked_state = None
    try:
        validate_execute_config(cfg)
        if stop.is_set():
            raise RuntimeError("Pickup cancelled")
        control = control or RosControl()
        control.stop()
        start = control.wait()
        last_checked_state = start
        check_state(start, start, cfg)
        phase = "settling"
        report("PICKUP — stopping policy motion…")
        deadline = time.monotonic() + 3
        stationary = 0
        while stationary < 3:
            if stop.is_set():
                raise RuntimeError("Pickup cancelled")
            state = control.state()
            last_checked_state = state
            check_state(state, start, cfg)
            stationary = stationary + 1 if np.linalg.norm(state["joint_velocity"]) <= 0.03 else 0
            if sample:
                sample(state, phase, np.zeros(6))
            if time.monotonic() >= deadline:
                raise RuntimeError("Arm must be stationary at pickup start")
            stop.wait(1 / 15)
        start = state
        target = start["position"] + np.array([0.0, 0.0, LIFT_M])
        if not ((target >= cfg["workspace_min_m"]) & (target <= cfg["workspace_max_m"])).all():
            raise RuntimeError("20 cm lift endpoint outside configured base-frame workspace")
        control.require_velocity_controller()
        control.require_single_command_sources()

        def monitor_closing():
            nonlocal last_checked_state
            state = control.state()
            last_checked_state = state
            check_state(state, start, cfg)
            if np.linalg.norm(state["position"] - start["position"]) > 0.003:
                raise RuntimeError("Arm moved while closing gripper")
            if np.linalg.norm(state["joint_velocity"]) > 0.03:
                raise RuntimeError("Arm must remain stationary while closing gripper")
            if sample:
                sample(state, phase, np.zeros(6))

        phase = "closing"
        report("PICKUP — closing gripper; arm stationary…")
        close_gripper(cfg, width, stop_event=stop, monitor=monitor_closing)
        if stop.is_set():
            raise RuntimeError("Pickup cancelled")
        monitor_closing()
        control.require_fresh_aperture()
        closed_width = control.width
        if closed_width < 0.001 or abs(closed_width - width) > 0.003:
            raise RuntimeError("Gripper aperture does not confirm the selected grasp")

        # Only this explicit operator action gets a 20 cm travel allowance.
        # The policy's displacement bound and all force/torque limits stay intact.
        lift_cfg = dict(cfg, max_displacement_m=LIFT_M + 0.005, max_rotation_rad=min(cfg["max_rotation_rad"], 0.05))
        phase = "lifting"
        control.activate()
        began = time.monotonic()
        previous_time = began
        progress_time = began
        progress_z = float(start["position"][2])
        previous_speed = 0.0
        speed_limit = min(cfg["max_linear_speed_m_s"], 0.01)
        deadline = began + min(40.0, LIFT_M / speed_limit + 10.0)
        while not stop.is_set():
            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError("Pickup lift timed out")
            state = control.state()
            last_checked_state = state
            check_state(state, start, lift_cfg)
            control.require_fresh_aperture()
            if abs(control.width - closed_width) > 0.003:
                raise RuntimeError("Gripper aperture changed during pickup")
            delta = state["position"] - start["position"]
            if np.linalg.norm(delta[:2]) > 0.005 or delta[2] < -0.003 or delta[2] > LIFT_M + 0.003:
                raise RuntimeError("Pickup deviated from the vertical lift corridor")
            remaining = float(target[2] - state["position"][2])
            if remaining <= TOLERANCE_M:
                reason = "lift_complete_grasp_unverified"
                break
            if state["position"][2] >= progress_z + 0.001:
                progress_z = float(state["position"][2])
                progress_time = now
            elif now - progress_time > 3.0:
                raise RuntimeError("Pickup stalled; no upward progress")
            # Ramp up at 20 mm/s² and slow proportionally near the endpoint.
            speed = min(speed_limit, remaining, previous_speed + 0.02 * max(0.0, now - previous_time))
            twist = np.array([0.0, 0.0, speed, 0.0, 0.0, 0.0])
            if stop.is_set():
                break
            control.send(twist)
            records.append(
                dict(
                    time_s=now - began,
                    position_m=state["position"].tolist(),
                    force_n=state["force"].tolist(),
                    torque_nm=state["torque"].tolist(),
                    gripper_width_m=control.width,
                    base_tcp_twist=twist.tolist(),
                )
            )
            if sample:
                sample(state, phase, twist)
            report(f"PICKUP — lifting {delta[2] * 100:.1f} / 20.0 cm along base +Z")
            previous_speed, previous_time = speed, now
            stop.wait(max(0.0, 1.0 / 15.0 - (time.monotonic() - now)))
        if reason == "lift_complete_grasp_unverified":
            control.stop()
            phase = "post_lift"
            report("PICKUP — lift complete; recording one more second…")
            tail_end = time.monotonic() + 1.0
            while time.monotonic() < tail_end and not stop.is_set():
                state = control.state()
                last_checked_state = state
                check_state(state, start, lift_cfg)
                control.require_fresh_aperture()
                if abs(control.width - closed_width) > 0.003:
                    raise RuntimeError("Gripper aperture changed after pickup")
                if sample:
                    sample(state, phase, np.zeros(6))
                stop.wait(min(1 / 15, max(0.0, tail_end - time.monotonic())))
            if stop.is_set():
                reason = "operator_stop"
    except Exception as exc:
        reason = "operator_stop" if stop.is_set() else f"blocked: {exc}"
        report(reason)
    finally:
        if control:
            try:
                control.close()
            except Exception as exc:
                reason += f"; stop service error: {exc}"
        data = dict(
            operation="manual_close_and_lift",
            lift_m=LIFT_M,
            config=cfg,
            jaw_width_m=width,
            outcome=reason,
            phase=phase,
            start=start,
            last_checked_state=last_checked_state,
            steps=records,
        )
        if result_data is not None:
            result_data.update(data)
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    data,
                    indent=2,
                    default=lambda value: value.tolist(),
                )
            )
        report(f"STOPPED — {reason}. No automatic release; support the object before opening.")
    return reason
