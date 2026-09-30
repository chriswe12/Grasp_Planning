#!/usr/bin/env python3
"""Configure a training-catalog grasp, stream ZED RGB-D, then run a manual-start policy."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_planning.real_franka.core import DEFAULT_CATALOG, DEFAULT_CHECKPOINT, ROOT


def main():
    import torch

    torch.set_num_threads(2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["configure", "stream", "run", "capture", "check"])
    p.add_argument("--config", type=Path, default=ROOT / "configs/franka_real_selection.local.json")
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--execute", action="store_true", help="Enable explicit GUI start and manual-close buttons; does not auto-start"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/franka_real_20260918")
    args = p.parse_args()
    if args.execute and args.mode != "run":
        p.error("--execute is only valid with run")
    cfg = {}
    if args.config.exists():
        cfg = json.loads(args.config.read_text())
        args.catalog = Path(cfg["catalog"])
        args.checkpoint = Path(cfg["checkpoint"])
    if args.mode == "check":
        from grasp_planning.real_franka.core import Actor, Catalog

        cat = Catalog(args.catalog)
        actor = Actor(cat, args.checkpoint, device=args.device)
        target = cat.index(cfg["target_id"]) if cfg else int(cat.indices()[0])
        actor.set_target(target, cfg)
        print(
            json.dumps(
                dict(
                    epoch=actor.epoch,
                    checkpoint=str(args.checkpoint),
                    target_id=str(cat.data["target_ids"][target]),
                    camera_profile=cat.contract["camera_profile"],
                    validated_targets=len(cat.indices()),
                    parts=len(set(cat.data["part_keys"])),
                    example_action=actor.infer(actor.goal).tolist(),
                ),
                indent=2,
            )
        )
        return
    if args.mode == "capture":
        import time

        import numpy as np
        from PIL import Image

        from grasp_planning.real_franka.camera import ZedCamera

        cam = ZedCamera().start()
        try:
            deadline = time.monotonic() + 45
            while cam.latest is None and not cam.error and time.monotonic() < deadline:
                time.sleep(0.1)
            time.sleep(1)
            f = cam.frame(0.5)
            args.output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(f[2]).save(args.output / "real_left.png")
            np.save(args.output / "real_depth.npy", f[3])
            (args.output / "camera.json").write_text(json.dumps(cam.calibration, indent=2))
            print(args.output)
        finally:
            cam.close()
        return
    # Reuse the camera-owning server across configure/stream/run commands.
    import re
    import urllib.error
    import urllib.request
    import webbrowser

    address = f"http://127.0.0.1:{args.port}"
    try:
        with urllib.request.urlopen(address + "/api/info", timeout=1) as response:
            info = json.load(response)
    except (urllib.error.URLError, TimeoutError):
        info = None
    if info:
        if info.get("application") != "franka_visual_policy" or info["config"] != str(args.config.resolve()):
            raise RuntimeError("Port belongs to another workbench/config; use --port")
        if args.mode == "run":
            with urllib.request.urlopen(address, timeout=2) as response:
                page = response.read().decode("utf-8")
            token = re.search(r"const TOKEN='([a-f0-9]+)'", page).group(1)
            req = urllib.request.Request(
                address + "/api/mode",
                data=json.dumps(dict(execute=args.execute, confirmed=True)).encode(),
                headers={"Content-Type": "application/json", "X-Session-Token": token},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                json.load(response)
        print(address)
        if not args.no_browser:
            webbrowser.open(address)
        return

    from grasp_planning.real_franka.web_app import serve

    serve(args)


if __name__ == "__main__":
    main()
