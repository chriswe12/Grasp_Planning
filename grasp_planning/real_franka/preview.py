"""Local configure/stream/run GUI. Goals are the exact saved Isaac training pixels."""

import numpy as np
from PIL import Image, ImageDraw

from .core import ROOT, rotation


def orientation_preview(catalog, i, size=(320, 210)):
    """Illustrative mesh/grasp diagram, explicitly separate from the Isaac policy image."""
    part = str(catalog.data["part_keys"][i])
    assembly, number = part.split("__part_")
    path = ROOT / "assets/obj/fabrica" / assembly / (number + ".obj")
    v = []
    faces = []
    for line in path.read_text().splitlines():
        s = line.split()
        if s and s[0] == "v":
            v.append([float(x) for x in s[1:4]])
        if s and s[0] == "f":
            faces.append([int(x.split("/")[0]) - 1 for x in s[1:]])
    v = np.asarray(v) * 0.01
    v -= np.mean(v, axis=0)
    pose = catalog.data["object_poses"][i]
    v = v @ rotation(pose[3:]).T
    basis = np.array([[0.80, -0.60, 0], [0.33, 0.44, 0.84], [0.50, 0.67, -0.55]])
    xyz = v @ basis.T
    scale = min((size[0] - 45) / max(np.ptp(xyz[:, 0]), 0.03), (size[1] - 55) / max(np.ptp(xyz[:, 1]), 0.03))
    uv = xyz[:, :2] * [scale, -scale] + np.array(size) / 2
    image = Image.new("RGB", size, "#e8e3d7")
    draw = ImageDraw.Draw(image)
    for f in sorted(faces, key=lambda f: np.mean(xyz[f, 2])):
        draw.polygon([tuple(x) for x in uv[f]], fill="#5578a8")
    draw.text((8, 5), part, fill="black")
    draw.text((8, size[1] - 20), str(catalog.data["orientation_ids"][i]) + " | orientation diagram", fill="black")
    return image
