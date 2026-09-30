"""Loopback-only browser workbench for selection, camera and deliberate policy runs."""

import io
import json
import secrets
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image, ImageDraw, ImageFont

from .camera import ZedCamera
from .core import ROOT, Catalog, live_to_training
from .goal_color import load_goal, normalize_color, render_goal
from .handoff import PickupHandoff
from .preview import orientation_preview
from .run_review import RUN_ID, completed_run, latest_recording, rating, review_run


def create_app(args, camera=None):
    app = Flask(__name__)
    token = secrets.token_hex(24)
    cat = Catalog(args.catalog)
    camera = camera or ZedCamera().start()
    mutex = threading.RLock()
    stop = threading.Event()
    state = dict(
        selection=None,
        config=None,
        draft=None,
        goal_render=None,
        goal_image=None,
        goal_revision=0,
        worker=None,
        handoff=None,
        status="Choose your part, orientation and grasp.",
        action=None,
        log=latest_recording(ROOT / "artifacts/franka_real_runs"),
        preview=None,
        preview_seq=-1,
    )
    if args.config.exists():
        cfg = json.loads(args.config.read_text())
        state["selection"] = cat.index(cfg["target_id"])
        state["config"] = cfg
        state["goal_render"] = cfg.get("goal_render")
        state["goal_image"] = load_goal(cat, state["selection"], cfg)

    def busy():
        return state["worker"] is not None and state["worker"].is_alive()

    def report(message, action=None):
        state["status"] = message
        if message.startswith("blocked:") and any(
            reason in message for reason in ("External force limit", "External torque limit", "Franka collision flag")
        ):
            # A contact/load-related stop requires deliberate re-enabling,
            # rather than another click silently resuming physical execution.
            args.execute = False
        if action is not None:
            state["action"] = np.asarray(action).tolist()

    def jpeg(image):
        b = io.BytesIO()
        image.save(b, format="JPEG", quality=88)
        return Response(b.getvalue(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    def selected():
        i = state["selection"]
        if i is None:
            raise ValueError("Choose a grasp first")
        return i

    def save(confirmed=None):
        i = selected()
        cfg = dict(state["config"] or state["draft"] or cat.select(i, args.checkpoint))
        if state["goal_render"]:
            cfg["goal_render"] = state["goal_render"]
            load_goal(cat, i, cfg)
        else:
            cfg.pop("goal_render", None)
        if confirmed is not None:
            cfg["mount_confirmed"] = bool(confirmed)
        state["config"] = cfg
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(json.dumps(cfg, indent=2) + "\n")
        return cfg

    def preview():
        f = camera.frame(0.5)
        with mutex:
            if state["preview_seq"] != f[1]:
                packed, valid = live_to_training(f[2], f[3], camera.calibration, cat.profile)
                state["preview"] = (packed[0].numpy(), float(valid.float().mean()))
                state["preview_seq"] = f[1]
            return f, state["preview"]

    @app.before_request
    def protect():
        if request.host.split(":")[0] not in ("127.0.0.1", "localhost"):
            return "Local access only", 403
        if request.method == "POST" and not secrets.compare_digest(request.headers.get("X-Session-Token", ""), token):
            return "Invalid session", 403

    @app.errorhandler(Exception)
    def error(e):
        return jsonify(error=str(e)), 400

    @app.get("/")
    def index():
        html = (Path(__file__).parent / "web/index.html").read_text(encoding="utf-8").replace("__TOKEN__", token)
        return Response(
            html,
            mimetype="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
            },
        )

    @app.get("/api/catalog")
    def catalog():
        parts = []
        for part in sorted(p for p in set(cat.data["part_keys"]) if p.startswith("plumbers_block__")):
            orientations = []
            for orient in sorted(set(cat.data["orientation_ids"][cat.indices(part)])):
                ids = cat.indices(part, orient)
                diverse = cat.diverse(ids)
                orientations.append(
                    dict(id=str(orient), example=int(ids[0]), grasps=[int(i) for i in ids], diverse=diverse)
                )
            parts.append(dict(id=str(part), example=int(cat.indices(part)[0]), orientations=orientations))
        targets = [
            dict(
                index=int(i),
                id=str(cat.data["target_ids"][i]),
                grasp=str(cat.data["source_grasp_ids"][i]),
                jaw_mm=float(cat.data["jaw_widths"][i] * 1000),
                open_mm=float(cat.data["open_widths"][i] * 1000),
                split=str(cat.data["split"][i]),
            )
            for i in cat.indices()
            if str(cat.data["part_keys"][i]).startswith("plumbers_block__")
        ]
        return jsonify(
            parts=parts,
            targets=targets,
            selected=state["selection"],
            execute_enabled=args.execute,
            checkpoint_name=Path(args.checkpoint).name,
            policy_hz=cat.contract["training_recipe"]["policy_hz"],
            training_camera=cat.profile,
        )

    @app.get("/api/state")
    def status():
        f = camera.latest
        return jsonify(
            status=state["status"],
            running=busy(),
            pickup_available=bool(
                args.execute
                and state["config"] is not None
                and (not busy() or (not stop.is_set() and state["handoff"] and state["handoff"].available()))
            ),
            selected=state["selection"],
            action=state["action"],
            camera=camera.calibration,
            camera_error=camera.error,
            frame_age_s=None if f is None else time.monotonic() - f[0],
            valid_depth=None if state["preview"] is None else state["preview"][1],
            saved=state["config"] is not None,
            mount_confirmed=bool(state["config"] and state["config"].get("mount_confirmed")),
            execute_enabled=args.execute,
            goal_color=(state["goal_render"] or {}).get("color"),
            goal_revision=state["goal_revision"],
            recording_ready=bool(state["log"] and state["log"].with_suffix(".mp4").is_file()),
            recording_run_id=state["log"].stem if state["log"] else None,
            recording_review=rating(state["log"]),
        )

    @app.post("/api/recording/review")
    def review_recording():
        with mutex:
            if busy():
                raise ValueError("Wait until the run and video saving finish before rating")
            run_id = request.json.get("run_id", "")
            good = request.json.get("good")
            if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id) or type(good) is not bool:
                raise ValueError("Select a completed run and choose Good video or Not good")
            path = ROOT / "artifacts/franka_real_runs" / (run_id + ".json")
            if not completed_run(path):
                raise ValueError("This run does not have a completed recording")

            def work():
                try:
                    report("Copying approved video and run data…" if good else "Saving video rating…")
                    result = review_run(path, good, ROOT / "artifacts/franka_good_runs")
                    report(
                        "Good video saved: " + result["archive"]
                        if good
                        else "Marked Not good. Original files retained."
                    )
                except Exception as exc:
                    report("Video rating failed: " + str(exc))

            state["worker"] = threading.Thread(target=work, daemon=True)
            state["worker"].start()
        return jsonify(ok=True)

    @app.get("/recording.mp4")
    def recording_video():
        path = state["log"]
        if path is None or not path.with_suffix(".mp4").is_file():
            raise ValueError("No video yet. It is saved after the next run stops.")
        return send_file(path.with_suffix(".mp4"), mimetype="video/mp4", conditional=True)

    def selected_goal():
        return state["goal_image"] if state["goal_image"] is not None else cat.data["goal_rgbd"][selected()]

    @app.get("/goal")
    def selected_goal_image():
        return jpeg(Image.fromarray((selected_goal()[..., :3] * 255).astype("uint8")).resize((512, 288)))

    @app.post("/api/color")
    def color():
        with mutex:
            if busy():
                raise ValueError("Stop before changing the goal color")
            i = selected()
            value = request.json.get("color")
            if value is not None:
                value = normalize_color(value)
            state["draft"] = dict(state["config"] or state["draft"] or cat.select(i, args.checkpoint))
            state["config"] = None
            if value is None:
                state["goal_render"] = None
                state["goal_image"] = None
                state["goal_revision"] += 1
                report("Original training goal restored. Save when ready.")
            else:
                stop.clear()

                def work():
                    try:
                        variant = render_goal(cat, i, value, stop, report)
                        image = load_goal(cat, i, {"goal_render": variant})
                        with mutex:
                            if stop.is_set():
                                raise RuntimeError("Color render cancelled")
                            state["goal_render"] = variant
                            state["goal_image"] = image
                            state["goal_revision"] += 1
                            report("Object color rendered. Preview below; save when ready.")
                    except Exception as exc:
                        report(str(exc))

                state["worker"] = threading.Thread(target=work, daemon=True)
                state["worker"].start()
        return jsonify(ok=True)

    @app.get("/image/<int:i>")
    def goal(i):
        if i not in cat.indices():
            raise ValueError("Invalid target")
        return jpeg(Image.fromarray((cat.data["goal_rgbd"][i, :, :, :3] * 255).astype("uint8")).resize((512, 288)))

    @app.get("/shape/<int:i>")
    def shape(i):
        if i not in cat.indices():
            raise ValueError("Invalid target")
        return jpeg(orientation_preview(cat, i, (420, 260)))

    @app.get("/camera/snapshot.npz")
    def camera_snapshot():
        # Capture both arrays from the same immutable camera frame, before any
        # policy resizing, depth packing, JPEG conversion or visualization.
        f = camera.frame(0.15)
        b = io.BytesIO()
        np.savez_compressed(
            b,
            rgb=f[2],
            depth_m=f[3],
            frame_seq=np.asarray(f[1]),
            capture_monotonic_s=np.asarray(f[0]),
            calibration_json=np.asarray(json.dumps(camera.calibration)),
        )
        return Response(b.getvalue(), mimetype="application/octet-stream", headers={"Cache-Control": "no-store"})

    @app.get("/camera/<kind>")
    def live(kind):
        f, (packed, coverage) = preview()
        if kind == "raw":
            im = Image.fromarray(f[2])
        elif kind == "policy":
            im = Image.fromarray((packed[..., :3] * 255).clip(0, 255).astype("uint8")).resize((512, 288))
        elif kind == "depth":
            gray = (255 * (1 - packed[..., 3])).clip(0, 255).astype("uint8")
            im = Image.fromarray(gray).convert("RGB").resize((512, 288))
        else:
            raise ValueError("Unknown image")
        return jpeg(im)

    @app.post("/api/select")
    def select():
        with mutex:
            if busy():
                raise ValueError("Stop the policy before changing selection")
            i = int(request.json["index"])
            if i not in cat.indices() or not str(cat.data["part_keys"][i]).startswith("plumbers_block__"):
                raise ValueError("Invalid plumbers-block target")
            state["selection"] = i
            state["config"] = None
            state["draft"] = None
            state["goal_render"] = None
            state["goal_image"] = None
            state["goal_revision"] += 1
            state["status"] = "Selection changed. Choose a goal color or keep the original, then save."
        return jsonify(ok=True)

    @app.post("/api/save")
    def save_selection():
        with mutex:
            if busy():
                raise ValueError("Stop before saving")
            cfg = save(request.json.get("mount_confirmed", False))
            state["status"] = "Configuration saved. Ready to observe."
        return jsonify(ok=True, path=str(args.config), target=cfg["target_id"])

    @app.get("/api/info")
    def info():
        return jsonify(application="franka_visual_policy", config=str(args.config.resolve()))

    @app.post("/api/mode")
    def mode():
        with mutex:
            if busy():
                raise ValueError("Stop the active run before changing modes")
            if not request.json.get("confirmed"):
                raise ValueError("CLI mode confirmation required")
            args.execute = bool(request.json.get("execute"))
        return jsonify(ok=True)

    @app.post("/api/start")
    def start():
        with mutex:
            if busy():
                raise ValueError("A run is already active")
            execute = bool(request.json.get("execute"))
            if execute and not args.execute:
                raise ValueError("Restart with run --execute to enable physical motion")
            if state["config"] is None:
                raise ValueError("Save the selection before running")
            cfg = dict(state["config"])
            if execute:
                from .core import validate_execute_config

                validate_execute_config(cfg)
            from .runner import run

            stop.clear()
            state["handoff"] = PickupHandoff()
            state["log"] = ROOT / "artifacts/franka_real_runs" / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
            state["worker"] = threading.Thread(
                target=run,
                args=(cfg, camera, stop, report),
                kwargs=dict(execute=execute, device=args.device, log_path=state["log"], handoff=state["handoff"]),
                daemon=True,
            )
            state["worker"].start()
        return jsonify(ok=True)

    @app.post("/api/stop")
    def stop_run():
        stop.set()
        return jsonify(ok=True)

    @app.post("/api/gripper")
    def gripper():
        with mutex:
            if busy():
                raise ValueError("Stop before operating the gripper")
            if not args.execute or state["config"] is None:
                raise ValueError("Use run --execute and save a selection")
            if not request.json.get("confirmed"):
                raise ValueError("Manual confirmation required")
            opening = bool(request.json.get("opening"))
            cfg = dict(state["config"])
            i = selected()
            stop.clear()

            def work():
                from .ros_control import close_gripper

                try:
                    report(
                        close_gripper(
                            cfg,
                            float(cat.data["open_widths" if opening else "jaw_widths"][i]),
                            opening=opening,
                            stop_event=stop,
                        )
                    )
                except Exception as e:
                    report(str(e))

            state["worker"] = threading.Thread(target=work, daemon=True)
            state["worker"].start()
        return jsonify(ok=True)

    @app.post("/api/pickup")
    def pickup_run():
        with mutex:
            if not args.execute or state["config"] is None:
                raise ValueError("Use run --execute and save a selection")
            from .core import validate_execute_config
            from .runner import run

            cfg = dict(state["config"])
            validate_execute_config(cfg)
            if busy():
                if stop.is_set() or not state["handoff"] or not state["handoff"].request():
                    raise ValueError("Pickup is available during robot policy motion, or after this operation finishes")
                return jsonify(ok=True, queued=True)
            stop.clear()
            state["handoff"] = None
            state["log"] = ROOT / "artifacts/franka_real_runs" / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
            state["worker"] = threading.Thread(
                target=run,
                args=(cfg, camera, stop, report),
                kwargs=dict(execute=True, device=args.device, log_path=state["log"], pickup_only=True),
                daemon=True,
            )
            state["worker"].start()
        return jsonify(ok=True)

    @app.get("/comparison.png")
    def comparison():
        i = selected()
        f, (packed, _) = preview()
        out = Image.new("RGB", (1536, 370), "#101a23")
        d = ImageDraw.Draw(out)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
        for j, (title, im) in enumerate(
            zip(
                ["Real ZED left", "Real at training intrinsics", "Selected Isaac goal (policy input)"],
                [
                    f[2],
                    (packed[..., :3] * 255).astype("uint8"),
                    (selected_goal()[..., :3] * 255).astype("uint8"),
                ],
            )
        ):
            d.text((j * 512 + 10, 10), title, fill="white", font=font)
            out.paste(Image.fromarray(im).resize((512, 288)), (j * 512, 40))
        d.text(
            (10, 339),
            "Goal: " + str(cat.data["target_ids"][i]) + " | Goal view is not a current-pose reconstruction.",
            fill="white",
            font=font,
        )
        b = io.BytesIO()
        out.save(b, "PNG")
        return Response(b.getvalue(), mimetype="image/png")

    app.workbench = dict(state=state, camera=camera, stop=stop, token=token)
    return app


def serve(args):
    from werkzeug.serving import make_server

    app = create_app(args)
    server = make_server("127.0.0.1", args.port, app, threaded=True)
    print(f"Franka workbench: http://127.0.0.1:{args.port}", flush=True)
    if not args.no_browser:
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    finally:
        app.workbench["stop"].set()
        worker = app.workbench["state"]["worker"]
        if worker:
            worker.join(timeout=25)
        app.workbench["camera"].close()
        server.server_close()
