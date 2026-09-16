#!/usr/bin/env python3
"""Add runtime-selectable finger/pad materials to the generated PDZ robot USD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "usd",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "assets/usd/kuka_iiwa7_pdz_gripper/configuration/kuka_iiwa7_pdz_gripper_base.usd",
    )
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


ARGS = _parse_args()
APP = SimulationApp({"headless": ARGS.headless})

from pxr import Usd  # noqa: E402

from grasp_planning.isaac_visual_materials import (  # noqa: E402
    author_pdz_gripper_material_variants,
)


def main() -> None:
    usd_path = ARGS.usd.expanduser().resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open generated PDZ base USD: {usd_path}")
    metadata = author_pdz_gripper_material_variants(stage)
    stage.GetRootLayer().Save()
    print(
        f"[DONE] usd={usd_path} roots={metadata['visual_roots']} variants={metadata['variants']}",
        flush=True,
    )


try:
    main()
finally:
    APP.close()
