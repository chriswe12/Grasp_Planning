"""Verify batched overrides preserve referenced assets and issue one USD notice."""

import pytest

pytest.importorskip("pxr")
from pxr import Gf, Sdf, Tf, Usd

from grasp_planning.rl.franka_appearance import FrankaAppearanceRandomizer


def test_cached_sdf_overrides_batch_without_editing_source():
    source = Sdf.Layer.CreateAnonymous("source.usda")
    source.ImportFromString('#usda 1.0\ndef Xform "Room" {\n color3f color = (1, 1, 1)\n float roughness = 0.5\n}\n')
    before = source.ExportToString()
    root = Sdf.Layer.CreateAnonymous("root.usda")
    root.subLayerPaths = [source.identifier]
    stage = Usd.Stage.Open(root)
    randomizer = FrankaAppearanceRandomizer.__new__(FrankaAppearanceRandomizer)
    randomizer.stage = stage
    randomizer._specs = {}
    color = stage.GetAttributeAtPath("/Room.color")
    roughness = stage.GetAttributeAtPath("/Room.roughness")
    color_spec, roughness_spec = randomizer._spec(color), randomizer._spec(roughness)
    notices = []
    registration = Tf.Notice.Register(Usd.Notice.ObjectsChanged, lambda *args: notices.append(1), stage)
    with Sdf.ChangeBlock():
        color_spec.default = Gf.Vec3f(0.1, 0.2, 0.3)
        roughness_spec.default = 0.8
    assert len(notices) == 1
    assert color.Get() == Gf.Vec3f(0.1, 0.2, 0.3)
    assert roughness.Get() == pytest.approx(0.8)
    assert source.ExportToString() == before
    assert randomizer._spec(color) is color_spec
    registration.Revoke()
