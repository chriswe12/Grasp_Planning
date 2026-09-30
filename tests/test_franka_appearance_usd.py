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


def test_fixed_wear_keeps_visibility_but_randomizes_materials():
    import json
    from pathlib import Path

    from pxr import UsdGeom, UsdLux, UsdShade

    root = Path(__file__).resolve().parents[1]
    stage = Usd.Stage.CreateInMemory()
    env = "/World/env_0"
    stage.DefinePrim(env + "/Lab", "Xform").GetReferences().AddReference(
        str(root / "assets/scenes/video_lab_pencil/environment.usdc")
    )
    shader = UsdShade.Shader.Define(stage, env + "/Part/material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.2, 0.4))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    light = UsdLux.SphereLight.Define(stage, env + "/RandomKey")
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
    profile = json.loads((root / "configs/franka_pencil_randomization.json").read_text())
    randomizer = FrankaAppearanceRandomizer(stage, [env], [[0, 0, 0]], [], profile, fixed_wear=True)
    first = randomizer.apply(0, 43)
    visibility = [attr.Get() for attr in randomizer.handles[0]["wear"]]
    changed = []
    registration = Tf.Notice.Register(
        Usd.Notice.ObjectsChanged,
        lambda notice, sender: changed.extend(map(str, notice.GetChangedInfoOnlyPaths())),
        stage,
    )
    second = randomizer.apply(0, 1234)
    assert first["wear_visible"] == second["wear_visible"]
    assert visibility == [attr.Get() for attr in randomizer.handles[0]["wear"]]
    assert first["part_color"] != second["part_color"]
    assert not any(path.endswith(".visibility") for path in changed)
    assert any(path.endswith("inputs:diffuseColor") for path in changed)
    registration.Revoke()


def test_background_simplification_preserves_table_floor_and_materials():
    from pathlib import Path

    from pxr import UsdGeom

    from grasp_planning.rl.franka_performance import simplify_background

    root = Path(__file__).resolve().parents[1]
    stage = Usd.Stage.CreateInMemory()
    env = "/World/env_0"
    stage.DefinePrim(env + "/Lab", "Xform").GetReferences().AddReference(
        str(root / "assets/scenes/video_lab_pencil/environment.usdc")
    )

    def snapshot(prefix):
        return {
            str(attr.GetPath()): str(attr.Get())
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prefix))
            for attr in prim.GetAttributes()
        }

    protected = [
        env + "/Lab/" + suffix
        for suffix in ("Table", "Room/Room_floor", "_materials", "Table_Support_Collision", "Floor_Collision")
    ]
    before = [snapshot(path) for path in protected]
    hidden = simplify_background(stage, [env])
    assert len(hidden) > 60
    assert before == [snapshot(path) for path in protected]
    assert all(UsdGeom.Imageable(stage.GetPrimAtPath(path)).ComputeVisibility() == "invisible" for path in hidden)
