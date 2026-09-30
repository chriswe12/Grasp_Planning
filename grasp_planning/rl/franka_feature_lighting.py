"""Catalog-bound soft room and wrist lights, with bounded seeded variation."""

import numpy as np

FEATURE_LIGHTING_PROFILE = {
    "version": "soft_features_v1",
    "room_lights": [
        ["key", [0.05, -0.5, 0.7], [0.65, 0.5], 850.0],
        ["fill", [0.95, 0.35, 0.5], [0.65, 0.5], 600.0],
        ["top", [0.45, 0.25, 1.0], [0.8, 0.6], 400.0],
    ],
    "camera_fill_intensity": 2500.0,
    "dome_intensity": 400.0,
    "intensity_scale_range": [0.8, 1.2],
}


def install(env):
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    profile = env.cfg.feature_lighting
    if profile != FEATURE_LIGHTING_PROFILE:
        raise ValueError("Unsupported feature lighting contract")
    if env.appearance is None:
        raise ValueError("Feature lighting requires per-environment appearance handles")
    groups = []

    def make(path, root, pos, size, intensity, quat, group):
        light = UsdLux.RectLight.Define(env.sim.stage, path)
        light.CreateWidthAttr(size[0])
        light.CreateHeightAttr(size[1])
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(1, 1, 1))
        light.CreateNormalizeAttr(False)
        x = UsdGeom.Xformable(light.GetPrim())
        x.AddTranslateOp().Set(Gf.Vec3d(*pos))
        x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat)
        for collection in [
            UsdLux.LightAPI(light.GetPrim()).GetLightLinkCollectionAPI(),
            UsdLux.LightAPI(light.GetPrim()).GetShadowLinkCollectionAPI(),
        ]:
            collection.CreateIncludeRootAttr(False)
            collection.CreateIncludesRel().SetTargets([Sdf.Path(root)])
        group.append((light.GetIntensityAttr(), intensity))

    q = env.camera_profile["quaternion_wxyz"]
    mount = Gf.Quatd(q[0], Gf.Vec3d(*q[1:]))
    rotation = Gf.Rotation(mount)
    for root in env.scene.env_prim_paths:
        group = []
        for name, pos, size, power in profile["room_lights"]:
            aim = (
                Gf.Matrix4d()
                .SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(0.45, 0, 0.08), Gf.Vec3d(0, 0, 1))
                .GetInverse()
                .ExtractRotationQuat()
            )
            make(root + "/FeatureLight_" + name, root, pos, size, power, aim, group)
        hand = next(
            p
            for p in env.sim.stage.Traverse()
            if str(p.GetPath()).startswith(root + "/Robot/") and p.GetName() == "panda_hand"
        )
        for side, x in [("left", -0.1), ("right", 0.1)]:
            pos = Gf.Vec3d(*env.camera_profile["position_m"]) + rotation.TransformDir(Gf.Vec3d(x, -0.05, 0.02))
            make(
                str(hand.GetPath()) + "/FeatureLight_camera_" + side,
                root,
                pos,
                (0.16, 0.12),
                profile["camera_fill_intensity"],
                mount * Gf.Quatd(0, Gf.Vec3d(1, 0, 0)),
                group,
            )
        groups.append(group)
    originals = []
    for prim in env.sim.stage.Traverse():
        if prim.HasAPI(UsdLux.LightAPI) and "FeatureLight" not in str(prim.GetPath()):
            attr = UsdLux.LightAPI(prim).GetIntensityAttr()
            if attr.Get() is not None:
                originals.append((attr, profile["dome_intensity"] if "DomeLight" in str(prim.GetPath()) else 0.0))
    # Match the existing appearance path's batched Sdf edits. Avoid a separate
    # USD change notification for every light in every resetting environment.
    group_specs = [[(env.appearance._spec(attr), value) for attr, value in group] for group in groups]
    original_specs = [(env.appearance._spec(attr), value) for attr, value in originals]
    original = env.appearance.apply_many

    def apply(env_ids, seeds):
        result = original(env_ids, seeds)
        edits = list(original_specs)
        for index, seed in zip(env_ids, seeds):
            rng = np.random.default_rng(int(seed) + 17371)
            for spec, value in group_specs[index]:
                edits.append((spec, value * float(rng.uniform(*profile["intensity_scale_range"]))))
        with Sdf.ChangeBlock():
            for spec, value in edits:
                spec.default = value
        return result

    env.appearance.apply_many = apply
    for attr, value in originals:
        attr.Set(value)
