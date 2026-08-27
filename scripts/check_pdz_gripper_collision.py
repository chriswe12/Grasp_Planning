#!/usr/bin/env python3
"""Validate PDZ gripper collision meshes against a representative grasp object.

This is deliberately independent of the grasp planner: it verifies that the
collision STL files, their URDF offsets, and the prismatic finger motion agree
before those meshes are used as a planning collision model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from trimesh.collision import CollisionManager


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLISION_DIR = REPO_ROOT / "assets" / "urdf" / "kuka_iiwa7_pdz_gripper" / "meshes" / "collision"


def _mesh(name: str, *, scale: tuple[float, float, float] = (0.001, 0.001, 0.001), offset=(0.0, 0.0, 0.0)):
    mesh = trimesh.load(COLLISION_DIR / name, force="mesh")
    transform = np.eye(4)
    transform[:3, :3] = np.diag(scale)
    transform[:3, 3] = np.asarray(offset, dtype=float)
    mesh.apply_transform(transform)
    return mesh


def _pdz_components(finger_position_m: float) -> dict[str, trimesh.Trimesh]:
    """Return collision meshes in the PDZ base-link frame at one jaw position."""

    position = float(finger_position_m)
    if not 0.0 <= position <= 0.032:
        raise ValueError("finger_position_m must be within the URDF range [0, 0.032].")
    return {
        "base": _mesh("base.stl"),
        # URDF: left joint axis -X, right joint axis +X.
        "left_finger": _mesh("left_finger.stl", offset=(-position, 0.0, 0.0)),
        "right_finger": _mesh("right_finger.stl", offset=(position, 0.0, 0.0)),
        # Pad origins/scales are copied from the URDF collision elements.
        "left_pad": _mesh(
            "left_pad_8mm.stl",
            offset=(-position, 0.0, 0.0),
        ),
        "right_pad": _mesh(
            "right_pad_8mm.stl",
            offset=(position, 0.0, 0.0),
        ),
    }


def _colliding_components(components: dict[str, trimesh.Trimesh], object_mesh: trimesh.Trimesh) -> list[str]:
    manager = CollisionManager()
    manager.add_object("object", object_mesh)
    return [name for name, mesh in components.items() if manager.in_collision_single(mesh)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finger-position-m", type=float, default=0.010)
    parser.add_argument("--cube-size-m", type=float, default=0.012)
    parser.add_argument("--cube-center-m", type=float, nargs=3, default=(0.0, 0.0, 0.136))
    args = parser.parse_args()

    components = _pdz_components(args.finger_position_m)
    cube = trimesh.creation.box(extents=(args.cube_size_m,) * 3)
    cube.apply_translation(np.asarray(args.cube_center_m, dtype=float))
    collisions = _colliding_components(components, cube)
    print(f"finger_position_m={args.finger_position_m:.6f}")
    print(f"cube_center_m={list(args.cube_center_m)} cube_size_m={args.cube_size_m:.6f}")
    print(f"colliding_components={','.join(collisions) if collisions else 'none'}")
    print(f"collision_free={not collisions}")


if __name__ == "__main__":
    main()
