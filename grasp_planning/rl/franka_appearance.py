"""Seeded per-episode USD material/light variation, independent of grasp geometry."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path


def validate_profile(profile):
    if (
        profile["profile"] != "franka_pencil_episode_appearance_v1"
        or profile["goal_appearance"] != "canonical_saved_reference"
    ):
        raise ValueError("Unsupported Franka appearance/goal relationship")
    for name, bounds in profile.items():
        if isinstance(bounds, list):
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"Invalid range: {name}")
    for key in ("canonical_part_fraction", "wear_visible_fraction"):
        if not 0 <= profile[key] <= 1:
            raise ValueError(key)
    for color in profile["palette"].values():
        if len(color) != 3 or not all(math.isfinite(x) and 0 <= x <= 1 for x in color):
            raise ValueError("Invalid palette color")
    return profile


def load_profile(path):
    return validate_profile(json.loads(Path(path).read_text()))


def sample_appearance(profile, seed):
    rng = random.Random(int(seed))

    def draw(name):
        return rng.uniform(*profile[name])

    names = list(profile["palette"])
    name = "blue" if rng.random() < profile["canonical_part_fraction"] else rng.choice(names)
    gain = draw("part_color_gain")
    table = draw("table_gain")
    sample = dict(
        seed=int(seed),
        part_color_name=name,
        part_color=[min(0.95, x * gain) for x in profile["palette"][name]],
        part_roughness=draw("part_roughness"),
        table_scale=[table * draw("table_channel_gain") for _ in range(3)],
        table_roughness=draw("table_roughness"),
        wall_gain=draw("wall_gain"),
        wear_visible=rng.random() < profile["wear_visible_fraction"],
        light_intensity=draw("light_intensity"),
        light_radius_m=draw("light_radius_m"),
        light_position=[draw("light_x_m"), draw("light_y_m"), draw("light_z_m")],
        light_temperature_k=draw("light_temperature_k"),
        exposure_ev=draw("exposure_ev"),
        white_balance=[draw("white_balance_gain") for _ in range(3)],
        prop_colors=[list(profile["palette"][rng.choice(names)]) for _ in range(5)],
    )
    sample["rgb_gain"] = [2 ** sample["exposure_ev"] * gain for gain in sample["white_balance"]]
    return sample


class FrankaAppearanceRandomizer:
    """Cache USD handles once; update only the environments being reset.

    Key lights are linked to their own room, so resetting one environment does
    not relight another. Background geometry and collision state are untouched.
    """

    def __init__(self, stage, roots, origins, prop_names, profile):
        from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade

        self.profile = validate_profile(profile)
        self.stage = stage
        self._specs = {}
        self.origins = origins
        self.samples = [None] * len(roots)
        self.handles = []
        for root in roots:

            def shaders(path):
                return [UsdShade.Shader(p) for p in Usd.PrimRange(stage.GetPrimAtPath(path)) if p.IsA(UsdShade.Shader)]

            part = [s for s in shaders(root + "/Part") if s.GetIdAttr().Get() == "UsdPreviewSurface"]
            # Bind override created by make_franka_part_material_cfg, not the mesh's unused material.
            part = [s for s in part if str(s.GetPath()).startswith(root + "/Part/material/")]
            if len(part) != 1:
                raise RuntimeError(f"Expected one bound part shader at {root}: {part}")
            lab = shaders(root + "/Lab")
            table = [s for s in lab if "/Table_Laminate/" in str(s.GetPath())]
            albedo = [s for s in table if s.GetPrim().GetName() == "Laminate_Albedo"]
            surface = [s for s in table if s.GetIdAttr().Get() == "UsdPreviewSurface"]
            if len(albedo) != 1 or len(surface) != 1:
                raise RuntimeError("Missing laminate shader inputs")
            surface[0].GetInput("roughness").DisconnectSource()
            walls = [
                (s.GetInput("diffuseColor"), tuple(s.GetInput("diffuseColor").Get()))
                for s in lab
                if "/Warm_White_Partition/" in str(s.GetPath()) and s.GetIdAttr().Get() == "UsdPreviewSurface"
            ]
            wear = [
                UsdGeom.Imageable(p).CreateVisibilityAttr()
                for p in Usd.PrimRange(stage.GetPrimAtPath(root + "/Lab"))
                if p.GetName().startswith(("Tape_Remnant", "Handling_Scuff", "Faded_fixture_outline"))
            ]
            props = [
                [s for s in shaders(root + "/" + name) if s.GetIdAttr().Get() == "UsdPreviewSurface"]
                for name in prop_names
            ]
            key_prim = stage.GetPrimAtPath(root + "/RandomKey")
            light = UsdLux.LightAPI(key_prim)
            for collection in (light.GetLightLinkCollectionAPI(), light.GetShadowLinkCollectionAPI()):
                collection.CreateIncludeRootAttr(False)
                collection.CreateIncludesRel().SetTargets([Sdf.Path(root)])
            light.CreateEnableColorTemperatureAttr(True)
            move = next(
                op
                for op in UsdGeom.Xformable(key_prim).GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            )
            self.handles.append(
                dict(
                    part=part[0],
                    table=surface[0],
                    albedo=albedo[0].CreateInput("scale", Sdf.ValueTypeNames.Float4),
                    walls=walls,
                    wear=wear,
                    props=props,
                    light=light,
                    sphere=UsdLux.SphereLight(key_prim),
                    move=move,
                )
            )

    def _spec(self, attribute):
        """Resolve authoring handles outside any Sdf.ChangeBlock."""
        from pxr import Sdf

        path = attribute.GetPath()
        if path not in self._specs:
            target = self.stage.GetEditTarget()
            path_in_layer = target.MapToSpecPath(path)
            layer = target.GetLayer()
            spec = layer.GetAttributeAtPath(path_in_layer)
            if spec is None:
                owner = Sdf.CreatePrimInLayer(layer, path_in_layer.GetPrimPath())
                spec = Sdf.AttributeSpec(
                    owner,
                    attribute.GetName(),
                    attribute.GetTypeName(),
                    attribute.GetVariability(),
                    attribute.IsCustom(),
                )
            self._specs[path] = spec
        return self._specs[path]

    def _edits(self, index, sample):
        from pxr import Gf, UsdGeom

        h = self.handles[index]
        edits = []

        def put(attribute, value):
            edits.append((self._spec(attribute), value))

        put(h["part"].GetInput("diffuseColor").GetAttr(), Gf.Vec3f(*sample["part_color"]))
        put(h["part"].GetInput("roughness").GetAttr(), sample["part_roughness"])
        put(h["albedo"].GetAttr(), Gf.Vec4f(*sample["table_scale"], 1.0))
        put(h["table"].GetInput("roughness").GetAttr(), sample["table_roughness"])
        for attr, original in h["walls"]:
            put(attr.GetAttr(), Gf.Vec3f(*[min(0.98, c * sample["wall_gain"]) for c in original]))
        for attr in h["wear"]:
            put(attr, UsdGeom.Tokens.inherited if sample["wear_visible"] else UsdGeom.Tokens.invisible)
        for shaders, color in zip(h["props"], sample["prop_colors"]):
            for shader in shaders:
                put(shader.GetInput("diffuseColor").GetAttr(), Gf.Vec3f(*color))
                put(shader.GetInput("roughness").GetAttr(), sample["part_roughness"])
        put(h["light"].GetIntensityAttr(), sample["light_intensity"])
        put(h["light"].GetColorTemperatureAttr(), sample["light_temperature_k"])
        put(h["sphere"].GetRadiusAttr(), sample["light_radius_m"])
        # Light translation is local to env_N; the clone parent owns env origin.
        put(h["move"].GetAttr(), Gf.Vec3d(*sample["light_position"]))
        return edits

    def apply_many(self, indices, seeds):
        from pxr import Sdf

        samples = [sample_appearance(self.profile, seed) for seed in seeds]
        edits = [edit for index, sample in zip(indices, samples) for edit in self._edits(index, sample)]
        # Only cached Sdf specs are touched here: no Usd queries/mutations while
        # notices are deferred. Commit all resetting rooms in one notification.
        with Sdf.ChangeBlock():
            for spec, value in edits:
                spec.default = value
        for index, sample in zip(indices, samples):
            self.samples[index] = sample
        return samples

    def apply(self, index, seed):
        return self.apply_many([index], [seed])[0]
