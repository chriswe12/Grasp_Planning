from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from scripts import build_kuka_iiwa7_gripper_assets as asset_builder

SOURCE_INERTIA = np.array(
    [
        [4.0, 0.2, -0.3],
        [0.2, 5.0, 0.4],
        [-0.3, 0.4, 6.0],
    ],
    dtype=float,
)


def _write_source_urdf(path: Path) -> None:
    robot = ET.Element("robot", {"name": "inertia_test"})
    for link_name in asset_builder.ARM_LINKS:
        link = ET.SubElement(robot, "link", {"name": link_name})
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0.1 -0.2 0.3", "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": "2.5"})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": "4",
                "ixy": "0.2",
                "ixz": "-0.3",
                "iyy": "5",
                "iyz": "0.4",
                "izz": "6",
            },
        )
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def _tensor_from_urdf(inertia: ET.Element) -> np.ndarray:
    ixx = float(inertia.get("ixx"))
    ixy = float(inertia.get("ixy"))
    ixz = float(inertia.get("ixz"))
    iyy = float(inertia.get("iyy"))
    iyz = float(inertia.get("iyz"))
    izz = float(inertia.get("izz"))
    return np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ],
        dtype=float,
    )


def _rotation_from_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def test_arm_inertia_frame_conversion_preserves_and_serializes_full_tensor(tmp_path: Path) -> None:
    source_urdf = tmp_path / "source.urdf"
    _write_source_urdf(source_urdf)

    links, source_inertials, _ = asset_builder._load_arm_model(source_urdf)
    assert source_inertials["link2"].off_diagonal_inertia == (0.2, -0.3, 0.4)

    aligned = asset_builder._hardware_aligned_arm_inertials(source_inertials)["link2"]
    _, rpy = asset_builder.ARM_SOURCE_TO_LBR_LINK_FRAME["link2"]
    rotation = asset_builder._rpy_to_rotmat(*rpy)
    expected_tensor = rotation @ SOURCE_INERTIA @ rotation.T
    np.testing.assert_allclose(asset_builder._inertia_tensor(aligned), expected_tensor, atol=1.0e-12)

    local_xyz, local_rpy = asset_builder.ARM_SOURCE_TO_LBR_LINK_FRAME["link2"]
    spec = asset_builder.MeshSpec(
        link_name="link2",
        source_path=tmp_path / "link2.STL",
        output_name="link2.STL",
        scale=1.0,
        color_rgb=(0.65, 0.62, 0.59),
        local_xyz=local_xyz,
        local_rpy=local_rpy,
    )
    asset_builder._rewrite_link_for_output(links["link2"], spec=spec, inertial=aligned)
    output_inertia = links["link2"].find("inertial/inertia")
    assert output_inertia is not None
    np.testing.assert_allclose(_tensor_from_urdf(output_inertia), expected_tensor, atol=1.0e-8)


def test_usd_principal_axes_reconstruct_full_inertia_tensor(tmp_path: Path, monkeypatch) -> None:
    inertial = asset_builder.LinkInertial(
        mass_kg=2.5,
        center=(0.1, -0.2, 0.3),
        diagonal_inertia=(4.0, 5.0, 6.0),
        off_diagonal_inertia=(0.2, -0.3, 0.4),
    )
    spec = asset_builder.MeshSpec(
        link_name="link2",
        source_path=tmp_path / "link2.STL",
        output_name="link2.STL",
        scale=1.0,
        color_rgb=(0.65, 0.62, 0.59),
    )
    monkeypatch.setattr(
        asset_builder,
        "_mesh_payload",
        lambda _spec, _path: (np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)),
    )
    monkeypatch.setattr(asset_builder, "_write_visual_mesh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(asset_builder, "_write_collision_mesh", lambda *_args, **_kwargs: None)

    output = io.StringIO()
    asset_builder._write_link(
        output,
        spec=spec,
        mesh_path=spec.source_path,
        inertial=inertial,
        transform=np.eye(4, dtype=float),
        indent="",
    )
    usd_text = output.getvalue()

    moments_match = re.search(r"physics:diagonalInertia = \(([^)]+)\)", usd_text)
    axes_match = re.search(r"physics:principalAxes = \(([^)]+)\)", usd_text)
    assert moments_match is not None
    assert axes_match is not None
    moments = np.asarray([float(value) for value in moments_match.group(1).split(",")], dtype=float)
    axes_quaternion = np.asarray([float(value) for value in axes_match.group(1).split(",")], dtype=float)
    principal_axes = _rotation_from_quat_wxyz(axes_quaternion)

    reconstructed = principal_axes @ np.diag(moments) @ principal_axes.T
    np.testing.assert_allclose(reconstructed, SOURCE_INERTIA, atol=2.0e-8)
