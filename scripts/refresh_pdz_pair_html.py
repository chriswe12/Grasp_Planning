#!/usr/bin/env python3
"""Update existing Stage-3 pair HTML files to render PDZ collision hulls.

Use this after a long offline build that completed its JSON artifacts before a
renderer-only failure. It does not rerun grasp generation or collision checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.pipeline.holder_state_debug_html import _gripper_payload

OLD_DRAW = '''function drawGripper(c,color,translation=[0,0,0],alpha=.68){const h=c.jaw_width/2,items=[[data.gripper.base,0],[data.gripper.left_finger,-h-data.gripper.left_fingertip_inner_y],[data.gripper.right_finger,h-data.gripper.right_fingertip_inner_y]],records=[];items.forEach(([comp,shift])=>records.push(...faceRecords(componentWorld(comp,c,shift,translation),comp.faces,color)));drawFaces(records,alpha)}'''
PDZ_DRAW = '''function drawGripper(c,color,translation=[0,0,0],alpha=.68){const h=c.jaw_width/2,items=data.gripper.model==="pdz_gripper"?[[data.gripper.base,0],[data.gripper.left_finger,-Math.max(0,(c.jaw_width-.012)/2)],[data.gripper.right_finger,Math.max(0,(c.jaw_width-.012)/2)]]:[[data.gripper.base,0],[data.gripper.left_finger,-h-data.gripper.left_fingertip_inner_y],[data.gripper.right_finger,h-data.gripper.right_fingertip_inner_y]],records=[];items.forEach(([comp,shift])=>records.push(...faceRecords(componentWorld(comp,c,shift,translation),comp.faces,color)));drawFaces(records,alpha)}'''


def _refresh(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.index("const data=") + len("const data=")
    end = text.index(",S=data.sequence", start)
    payload = json.loads(text[start:end])
    payload["gripper"] = _gripper_payload("pdz_gripper")
    if OLD_DRAW not in text:
        raise ValueError(f"'{path}' does not contain the expected legacy gripper renderer.")
    text = text[:start] + json.dumps(payload, separators=(",", ":")) + text[end:]
    path.write_text(text.replace(OLD_DRAW, PDZ_DRAW), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.output_dir.expanduser().resolve().glob("dual_grasp_pairs_step_*.html"))
    if not paths:
        raise FileNotFoundError("No Stage-3 pair HTML files found.")
    for path in paths:
        _refresh(path)
        print(path)


if __name__ == "__main__":
    main()
