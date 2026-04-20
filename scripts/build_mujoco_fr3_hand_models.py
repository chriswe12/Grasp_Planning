#!/usr/bin/env python3
"""Generate the local MuJoCo FR3 model augmented with the Menagerie Panda hand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_repo_root()

from grasp_planning.mujoco.model_builder import build_fr3_with_panda_hand_xml  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--menagerie-root",
        type=Path,
        default=Path(".cache/robot_descriptions/mujoco_menagerie"),
        help="Path to the local MuJoCo Menagerie checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/generated_mujoco_models"),
        help="Directory to write merged FR3-with-hand XML files into.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    menagerie_root = args.menagerie_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    panda_hand_xml = menagerie_root / "franka_emika_panda" / "hand.xml"
    output_path = output_dir / "fr3_with_panda_hand.xml"
    build_fr3_with_panda_hand_xml(
        arm_xml_path=menagerie_root / "franka_fr3" / "fr3.xml",
        panda_hand_xml_path=panda_hand_xml,
        output_xml_path=output_path,
    )
    print(output_path)


if __name__ == "__main__":
    main()
