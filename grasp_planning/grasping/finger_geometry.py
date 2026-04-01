"""Shared finger collision geometry helpers."""

from __future__ import annotations

import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def finger_boxes_from_grasp(
    grasp_rotmat: np.ndarray,
    contact_point_a: np.ndarray,
    contact_point_b: np.ndarray,
    *,
    finger_extent_lateral: float,
    finger_extent_closing: float,
    finger_extent_approach: float,
    finger_clearance: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build coarse finger boxes from a rolled grasp pose.

    Grasp-frame convention for the mesh antipodal path:
    - local x: lateral axis orthogonal to closing and approach
    - local y: closing axis between the two contacts
    - local z: approach axis

    The current coarse finger model places each box on its respective side of
    the object along the closing axis, with extents ordered as (x, y, z).
    """

    grasp_y = _normalize(grasp_rotmat[:, 1])
    # Extents are expressed in grasp-frame axis order: (lateral x, closing y, approach z).
    half_extents = 0.5 * np.array(
        [
            finger_extent_lateral,
            finger_extent_closing,
            finger_extent_approach,
        ],
        dtype=float,
    )
    offset = grasp_y * (0.5 * finger_clearance + half_extents[1])
    box_a = (contact_point_a - offset, grasp_rotmat, half_extents)
    box_b = (contact_point_b + offset, grasp_rotmat, half_extents)
    return box_a, box_b


def finger_box_corners(center: np.ndarray, rotation: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    local_corners = signs * half_extents[None, :]
    return center[None, :] + local_corners @ rotation.T
