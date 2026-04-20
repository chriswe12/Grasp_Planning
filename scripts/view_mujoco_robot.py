#!/usr/bin/env python3
"""Open a local MuJoCo robot XML or repo robot config in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_repo_root()

from grasp_planning.mujoco.runner import load_robot_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help="Path to a MuJoCo robot config JSON. If omitted, --xml-path must be provided.",
    )
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=None,
        help="Path to a MuJoCo XML/MJCF file to open directly.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print joint, actuator, and EE binding information before opening the viewer.",
    )
    return parser.parse_args()


def _resolve_xml_path(args: argparse.Namespace) -> tuple[Path, object | None]:
    if args.robot_config is not None:
        cfg = load_robot_config(args.robot_config)
        return Path(cfg.robot_xml_path).resolve(), cfg
    if args.xml_path is not None:
        return args.xml_path.expanduser().resolve(), None
    raise SystemExit("Provide either --robot-config or --xml-path.")


def main() -> None:
    args = _parse_args()
    xml_path, cfg = _resolve_xml_path(args)
    if not xml_path.is_file():
        raise SystemExit(f"MuJoCo XML not found at '{xml_path}'.")

    import mujoco  # type: ignore
    import mujoco.viewer  # type: ignore

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    if cfg is not None or args.print_summary:
        print(f"xml: {xml_path}")
        print(f"nq={model.nq} nv={model.nv} nu={model.nu}")
        if cfg is not None:
            print(f"ee_site_name={cfg.ee_site_name}")
            print(f"ee_body_name={cfg.ee_body_name}")
            print(f"arm_joint_names={list(cfg.arm_joint_names)}")
            print(f"arm_actuator_names={list(cfg.arm_actuator_names)}")
            print(f"gripper_actuator_names={list(cfg.gripper_actuator_names)}")

    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
