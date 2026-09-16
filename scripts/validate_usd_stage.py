"""Open a USD stage inside Isaac and print whether selected prims exist."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("usd_path", type=Path)
parser.add_argument("--prim", action="append", default=[])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd  # noqa: E402


def main() -> None:
    stage = Usd.Stage.Open(str(args.usd_path.expanduser().resolve()))
    print(f"stage_open={bool(stage)}", flush=True)
    if stage is None:
        os._exit(1)
    for path in args.prim:
        prim = stage.GetPrimAtPath(path)
        print(f"{path} valid={prim.IsValid()} type={prim.GetTypeName() if prim else ''}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
