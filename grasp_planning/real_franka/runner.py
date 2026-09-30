"""15 Hz manual-start policy loop; observe-only by default."""

import json
import math
import time
from pathlib import Path

import numpy as np

from .core import Actor, Catalog, live_to_training, rotation, sha256, validate_execute_config
from .ros_control import RosControl, check_state


def run(
    cfg, camera, stop_event, report, execute=False, device="cuda:0", log_path=None, handoff=None, pickup_only=False
):
    control = None
    records = []
    reason = "stopped"
    last_checked_state = None
    recording = None
    recording_result = None
    pickup_result = None
    duration = cfg.get("max_duration_s")
    try:
        if pickup_only and not execute:
            raise ValueError("Pickup requires physical execution mode")
        cat = Catalog(cfg["catalog"])
        i = cat.index(cfg["target_id"])
        if cat.hash != cfg["catalog_sha256"] or sha256(cfg["checkpoint"]) != cfg["checkpoint_sha256"]:
            raise ValueError("Configured artifacts changed; configure again")
        if execute:
            validate_execute_config(cfg)
        report("Loading checkpoint…")
        actor = Actor(cat, cfg["checkpoint"], device=device)
        actor.set_target(i, cfg)
        if log_path and cfg.get("record_video", True):
            from .recording import RunRecording

            recording = RunRecording(log_path, actor.goal[0].detach().cpu().numpy(), duration)
        deadline = time.monotonic() + 20
        while camera.latest is None and time.monotonic() < deadline and not camera.error:
            time.sleep(0.05)
        frame = camera.frame()
        live, _ = live_to_training(frame[2], frame[3], camera.calibration, cat.profile, device)
        actor.infer(live)  # GPU warm-up before Servo activation.
        if camera.calibration["serial"] != cfg["camera_serial"]:
            raise ValueError("Camera serial differs from selection")
        if stop_event.is_set():
            raise RuntimeError("Start cancelled")
        if execute:
            report("Waiting for fresh arm, TCP and gripper feedback…")
            control = RosControl()
            start = control.wait()
            last_checked_state = start
            check_state(start, start, cfg)
            if np.linalg.norm(start["joint_velocity"]) > 0.03:
                raise RuntimeError("Arm must be stationary at start")
            if not pickup_only:
                control.check_width(float(cat.data["open_widths"][i]))
                control.activate()
        report("RUNNING: robot motion enabled" if execute else "OBSERVE ONLY: live inference; no robot commands")
        start_time = time.monotonic()
        next_tick = start_time
        seq = -1
        streak = 0
        previous = np.zeros(6)
        prev_state = None
        settings = cat.contract["training_recipe"]["source_settings"]
        count = math.ceil(settings["completion_required_consecutive_steps"] / 2)
        if handoff and execute and not pickup_only:
            handoff.policy_started()
        while (
            not pickup_only
            and not stop_event.is_set()
            and (duration is None or time.monotonic() - start_time < duration)
        ):
            if handoff and handoff.requested():
                reason = "operator_pickup"
                break
            tic = time.monotonic()
            frame = camera.frame(cfg["maximum_frame_age_s"])
            if frame[1] == seq:
                raise RuntimeError("Repeated camera frame")
            seq = frame[1]
            live, valid = live_to_training(frame[2], frame[3], camera.calibration, cat.profile, device)
            coverage = float(valid.float().mean())
            if coverage < cfg["minimum_valid_depth_fraction"]:
                raise RuntimeError(f"Insufficient valid depth: {coverage:.1%}")
            a = actor.infer(live)
            motion = previous + np.clip(
                a[:6] - previous, -settings["action_delta_limit"] * 2, settings["action_delta_limit"] * 2
            )
            stable = False
            twist = np.zeros(6)
            feedback = None
            measured_distance = 0.0
            if execute:
                state = control.state()
                last_checked_state = state
                check_state(state, start, cfg)
                control.check_width(float(cat.data["open_widths"][i]))
                R = state["rotation"] @ rotation(cat.profile["quaternion_wxyz"])
                twist = np.r_[R @ (motion[:3] * 0.04), R @ (motion[3:] * 0.24)]
                scale = min(
                    1.0,
                    cfg["max_linear_speed_m_s"] / max(np.linalg.norm(twist[:3]), 1e-9),
                    cfg["max_angular_speed_rad_s"] / max(np.linalg.norm(twist[3:]), 1e-9),
                )
                twist *= scale
                motion *= scale
                if prev_state is not None:
                    dt = tic - prev_state[0]
                    v = np.linalg.norm(state["position"] - prev_state[1]["position"]) / dt
                    omega = (
                        np.arccos(np.clip((np.trace(prev_state[1]["rotation"].T @ state["rotation"]) - 1) / 2, -1, 1))
                        / dt
                    )
                    stable = (
                        v <= settings["completion_max_linear_speed_m_s"]
                        and omega <= settings["completion_max_angular_speed_rad_s"]
                    )
                prev_state = (tic, state)
                # Never send a command based on a frame that expired during inference.
                if time.monotonic() - frame[0] > cfg["maximum_frame_age_s"]:
                    raise RuntimeError("Frame expired during inference")
                # A pickup/stop click during inference must not dispatch another policy command.
                if stop_event.is_set() or (handoff and handoff.requested()):
                    reason = "operator_stop" if stop_event.is_set() else "operator_pickup"
                    break
                control.send(twist)
                measured_distance = float(np.linalg.norm(state["position"] - start["position"]))
                health = control.sink.health(now_s=control.now())
                feedback = dict(
                    tcp_position_m=state["position"].tolist(),
                    tcp_transform_age_s=state["tcp_transform_age_s"],
                    tcp_rotation=state["rotation"].tolist(),
                    displacement_from_start_m=measured_distance,
                    joints=state["joints"],
                    joint_velocity=state["joint_velocity"].tolist(),
                    desired_joint_velocity=state["desired_joint_velocity"],
                    desired_joint_acceleration=state["desired_joint_acceleration"],
                    control_command_success_rate=state["control_command_success_rate"],
                    external_force_n=state["force"].tolist(),
                    external_torque_nm=state["torque"].tolist(),
                    gripper_width_m=control.width,
                    servo_status_code=health.status_code,
                    servo_status_text=health.status_text,
                    controller=state["controller_feedback"],
                    controller_age_s=state["controller_feedback_age_s"],
                )
            else:
                # A stationary observe-only robot receives no action: previous applied stays zero.
                motion[:] = 0
            previous = motion.copy()
            actor.previous = previous.astype(np.float32)
            streak = streak + 1 if a[6] >= settings["completion_probability_threshold"] and stable else 0
            records.append(
                dict(
                    time_s=tic - start_time,
                    phase="policy",
                    frame_seq=seq,
                    completion_probability=float(a[6]),
                    completion_stable=bool(stable),
                    completion_streak=streak,
                    requested_action=a.tolist(),
                    applied_action=previous.tolist(),
                    commanded_action=previous.tolist(),
                    base_tcp_twist=twist.tolist(),
                    measured_feedback=feedback,
                    valid_depth_fraction=coverage,
                    inference_cycle_s=time.monotonic() - tic,
                )
            )
            if recording:
                recording.append(frame, live, records[-1])
            activity = f"COMMANDING | TCP displaced {measured_distance * 1000:.1f} mm" if execute else "OBSERVE"
            report(f"{activity} | p(done) {a[6]:.3g} | depth {coverage:.0%}", a)
            if streak >= count:
                reason = "policy_declared_completion_unverified"
                break
            next_tick += 1 / 15
            remaining = next_tick - time.monotonic()
            if remaining < -0.10:
                raise RuntimeError("Policy loop missed timing deadline")
            stop_event.wait(max(0.0, remaining))
        else:
            reason = "operator_stop" if stop_event.is_set() else "time_limit"
        requested = handoff.finish_policy() if handoff else False
        if execute and not stop_event.is_set() and (pickup_only or requested):
            from .pickup import pickup

            if recording:
                recording.allow_pickup()
            pickup_result = {}
            pickup_control = control
            sample_time = -float("inf")

            def sample(state, phase, twist):
                nonlocal sample_time, last_checked_state, seq
                last_checked_state = state
                now = time.monotonic()
                if now - sample_time < 1 / 15 - 0.001:
                    return
                frame = camera.frame(cfg["maximum_frame_age_s"])
                if frame[1] == seq:
                    return
                seq = frame[1]
                live, valid = live_to_training(frame[2], frame[3], camera.calibration, cat.profile, device)
                sample_time = now
                step = dict(
                    time_s=now - start_time,
                    phase=phase,
                    frame_seq=seq,
                    completion_probability=None,
                    completion_stable=False,
                    completion_streak=0,
                    requested_action=[0.0] * 7,
                    applied_action=[0.0] * 6,
                    commanded_action=[0.0] * 6,
                    base_tcp_twist=twist.tolist(),
                    valid_depth_fraction=float(valid.float().mean()),
                    measured_feedback=dict(
                        tcp_position_m=state["position"].tolist(),
                        displacement_from_start_m=float(np.linalg.norm(state["position"] - start["position"])),
                        external_force_n=state["force"].tolist(),
                        external_torque_nm=state["torque"].tolist(),
                        gripper_width_m=pickup_control.width,
                    ),
                )
                records.append(step)
                if recording:
                    recording.append(frame, live, step)

            # Transfer this same controller to pickup; never create competing command publishers.
            control = None
            reason = pickup(
                cfg,
                float(cat.data["jaw_widths"][i]),
                stop_event,
                report,
                control=pickup_control,
                sample=sample,
                result_data=pickup_result,
            )
    except Exception as e:
        reason = f"blocked: {e}"
        report(reason)
    finally:
        if handoff:
            handoff.finish_policy()
        if control:
            try:
                control.close()
            except Exception as e:
                reason += f"; stop service error: {e}"
        if log_path:
            if recording:
                report(f"STOPPED — {reason}. Saving video…")
                try:
                    recording_result = recording.finish()
                except Exception as exc:
                    recording_result = dict(error=str(exc))
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(
                    dict(
                        execute=execute,
                        outcome=reason,
                        config=cfg,
                        camera=camera.calibration,
                        steps=records,
                        recording=recording_result,
                        pickup=pickup_result,
                        # Keep the rejected sample too: check_state can throw
                        # before the next successful command/step is recorded.
                        last_checked_state=last_checked_state,
                        action_semantics="applied_action and commanded_action are limited commands sent to Servo, not measured motion",
                    ),
                    indent=2,
                    default=lambda value: value.tolist(),
                )
            )
        suffix = ""
        if recording_result:
            suffix = " | Video saved" if recording_result.get("video") else ""
            if recording_result.get("error"):
                suffix = " | Recording failed: " + recording_result["error"]
        report(f"STOPPED — {reason}" + suffix)
    return reason
