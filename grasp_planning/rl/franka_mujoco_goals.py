"""MuJoCo RGB-D from the exact Isaac visual meshes and measured body poses.

Rendering only: no MuJoCo controller, IK, contact model or physics stepping.
Lab textures/decals are simplified to their material palette; object and robot
surfaces retain USD triangles. Geometry is never rebuilt from another robot URDF.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np


def _numbers(a):
    return " ".join(f"{v:.9g}" for v in np.asarray(a).ravel())


def _material_color(prim, fallback):
    from pxr import Usd, UsdShade

    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if material:
        # Preserve the palette of textured table/floor materials when simplifying
        # textures. Do not fall back to neutral gray for a connected albedo.
        from PIL import Image

        for shader_prim in Usd.PrimRange(material.GetPrim()):
            if shader_prim.IsA(UsdShade.Shader):
                shader = UsdShade.Shader(shader_prim)
                if shader.GetIdAttr().Get() == "UsdUVTexture" and "Albedo" in shader_prim.GetName():
                    asset = shader.GetInput("file").Get()
                    if asset and asset.resolvedPath:
                        rgb = np.asarray(Image.open(asset.resolvedPath).convert("RGB").resize((64, 64))) / 255.0
                        return rgb.mean(axis=(0, 1))
        for p in Usd.PrimRange(material.GetPrim()):
            if p.IsA(UsdShade.Shader):
                shader = UsdShade.Shader(p)
                for key in ("diffuseColor", "diffuse_color_constant"):
                    value = shader.GetInput(key).Get()
                    if value is not None:
                        return np.clip(np.asarray(value, dtype=float)[:3], 0, 1)
    return np.array(fallback)


class IsaacMeshMujocoRenderer:
    def __init__(self, env, slot):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco
        from pxr import Usd, UsdGeom

        self.mj = mujoco
        self.env, self.slot = env, slot
        self.origin = env.scene.env_origins[slot].cpu().numpy()
        self.profile = env.camera_profile
        root = env.scene.env_prim_paths[slot]
        cache = UsdGeom.XformCache()
        xml = ET.Element("mujoco", model="isaac_visual_snapshot")
        ET.SubElement(xml, "compiler", angle="radian", fusestatic="false")
        ET.SubElement(xml, "statistic", extent="1", center="0 0 0")
        visual = ET.SubElement(xml, "visual")
        ET.SubElement(
            visual, "global", offwidth=str(self.profile["render_width"]), offheight=str(self.profile["render_height"])
        )
        ET.SubElement(visual, "map", znear="0.01", zfar="1")
        ET.SubElement(visual, "quality", shadowsize="1024", offsamples="4")
        ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.25 0.25 0.25", specular="0 0 0")
        default = ET.SubElement(xml, "default")
        ET.SubElement(default, "geom", contype="0", conaffinity="0", mass="0", friction="0 0 0")
        assets = ET.SubElement(xml, "asset")
        world = ET.SubElement(xml, "worldbody")
        ET.SubElement(world, "light", name="key", pos="0.4 -0.2 1.5", dir="0 0 -1", diffuse="0.7 0.7 0.7")
        for name in ["fill", "rim", "camera_left", "camera_right"]:
            ET.SubElement(world, "light", name=name, pos="0 0 1", dir="0 0 -1", diffuse="0 0 0", specular="0 0 0")
        h, w = self.profile["render_height"], self.profile["render_width"]
        self.fy = self.profile["fy"] * h / self.profile["source_height"]
        ET.SubElement(world, "camera", name="wrist", fovy=str(np.rad2deg(2 * np.arctan(h / (2 * self.fy)))))
        self.body_names = [*env.robot.body_names, "part"]
        bodies = {name: ET.SubElement(world, "body", name=name) for name in self.body_names}
        self.part_geoms = []
        self.mesh_count = 0
        self.exported_paths = []
        for prim in Usd.PrimRange(env.sim.stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            path = str(prim.GetPath())
            if mesh.ComputeVisibility() == "invisible" or mesh.ComputePurpose() == "guide":
                continue
            if "/Lab/" in path and any(v in path for v in ("Handling_Scuff", "Tape_Remnant", "Faded_fixture")):
                continue  # Appearance-only thin decals are deliberately omitted.
            if not any(v in path for v in ("/Robot/", "/Part/", "/Lab/")):
                continue
            parent = world
            matrix = np.array(cache.GetLocalToWorldTransform(prim)).T
            fallback = [0.75, 0.75, 0.75]
            if "/Robot/" in path:
                name = path.split("/Robot/")[1].split("/")[0]
                if name not in bodies:
                    raise ValueError(f"Unmapped robot visual {path}")
                anchor = env.sim.stage.GetPrimAtPath(root + "/Robot/" + name)
                matrix = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(anchor)).T) @ matrix
                parent = bodies[name]
                fallback = [0.07] * 3 if "finger" in name else [0.75] * 3
            elif "/Part/" in path:
                anchor = env.sim.stage.GetPrimAtPath(root + "/Part")
                matrix = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(anchor)).T) @ matrix
                parent = bodies["part"]
            else:
                matrix[:3, 3] -= self.origin
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
            # Triangulate USD polygons without convexifying the visible surface.
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            faces, face_sources = [], []
            offset = 0
            for fi, count in enumerate(counts):
                poly = indices[offset : offset + count]
                for k in range(1, count - 1):
                    faces.append([poly[0], poly[k], poly[k + 1]])
                    face_sources.append(fi)
                offset += count
            faces = np.asarray(faces)
            if mesh.GetOrientationAttr().Get() == "leftHanded" or np.linalg.det(matrix[:3, :3]) < 0:
                faces = faces[:, ::-1]
            groups = []
            assigned = np.zeros(len(counts), dtype=bool)
            for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
                chosen = np.asarray(subset.GetIndicesAttr().Get(), dtype=int)
                assigned[chosen] = True
                groups.append((np.isin(face_sources, chosen), _material_color(subset.GetPrim(), fallback)))
            if not assigned.all():
                groups.append((~assigned[np.asarray(face_sources)], _material_color(prim, fallback)))
            for mask, color in groups:
                if not mask.any():
                    continue
                name = f"mesh{self.mesh_count}"
                ET.SubElement(
                    assets, "mesh", name=name, vertex=_numbers(vertices), face=_numbers(faces[mask]), inertia="shell"
                )
                ET.SubElement(parent, "geom", name=name, type="mesh", mesh=name, rgba=_numbers([*color, 1]))
                if "/Part/" in path:
                    self.part_geoms.append(self.mesh_count)
                self.exported_paths.append(path)
                self.mesh_count += 1
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        self.body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, v) for v in self.body_names]
        self.part_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"mesh{i}") for i in self.part_geoms
        ]

    def render(self, color, seed):
        import torch
        from isaaclab.utils.math import quat_apply, quat_mul

        from grasp_planning.rl.zed_mini import pack_zed_rgbd, reproject_intrinsics, scaled_intrinsics

        env, slot = self.env, self.slot
        pos = env.robot.data.body_pos_w[slot].cpu().numpy() - self.origin
        quat = env.robot.data.body_quat_w[slot].cpu().numpy()
        for i, body in enumerate(self.body_ids[:-1]):
            self.model.body_pos[body] = pos[i]
            self.model.body_quat[body] = quat[i]
        self.model.body_pos[self.body_ids[-1]] = env.part.data.root_pos_w[slot].cpu().numpy() - self.origin
        self.model.body_quat[self.body_ids[-1]] = env.part.data.root_quat_w[slot].cpu().numpy()
        # TiledCamera pose metadata can remain at initialization while RTX follows
        # the articulated prim. Derive the current optical frame from PhysX.
        hand_q = env.robot.data.body_quat_w[slot : slot + 1, env.hand_id]
        offset = torch.tensor([self.profile["position_m"]], device=env.device)
        camera_pos = env.robot.data.body_pos_w[slot, env.hand_id] + quat_apply(hand_q, offset)[0]
        ros_q = quat_mul(hand_q, env.mount_quat[slot : slot + 1])
        gl_q = quat_mul(ros_q, torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.device))
        self.model.cam_pos[0] = camera_pos.cpu().numpy() - self.origin
        self.model.cam_quat[0] = gl_q[0].cpu().numpy()
        self.model.geom_rgba[self.part_geom_ids, :3] = color
        rng = np.random.default_rng(seed)
        self.model.light_pos[0] = [rng.uniform(-0.1, 1), rng.uniform(-0.6, 0.3), rng.uniform(1.1, 1.9)]
        self.model.light_diffuse[0] = rng.uniform(0.45, 0.9, 3)
        self.model.light_diffuse[1:] = 0
        self.model.vis.headlight.ambient[:] = 0.35
        self.model.vis.headlight.diffuse[:] = 0.25
        if getattr(self, "lighting_mode", "original") != "original":
            self.model.vis.headlight.ambient[:] = 0.24
            self.model.vis.headlight.diffuse[:] = 0.24
            self.model.light_pos[:3] = [[0.05, -0.5, 0.7], [0.95, 0.35, 0.5], [0.45, 0.25, 1.0]]
            self.model.light_dir[:] = np.array([0.45, 0, 0.08]) - self.model.light_pos
            self.model.light_diffuse[:3] = np.array([0.18, 0.14, 0.08])[:, None]
            self.model.vis.headlight.ambient[:] = 0.40
            self.model.vis.headlight.diffuse[:] = 0.65
            from isaaclab.utils.math import matrix_from_quat

            rot = matrix_from_quat(ros_q)[0].cpu().numpy()
            for light_id, x in [(3, -0.10), (4, 0.10)]:
                self.model.light_pos[light_id] = self.model.cam_pos[0] + rot @ np.array([x, -0.05, 0.02])
                self.model.light_dir[light_id] = rot @ np.array([0.0, 0.0, 1.0])
                self.model.light_attenuation[light_id] = [0.0, 0.0, 1.0]
                self.model.light_diffuse[light_id] = 0.015 if self.lighting_mode == "soft" else 0.025
            self.model.light_specular[:] = 0.08
            if getattr(self.env.cfg, "feature_lighting", None):
                self.model.light_diffuse[:] *= rng.uniform(0.8, 1.2, (5, 1))

        self.mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="wrist")
        self.renderer.scene.flags[self.mj.mjtRndFlag.mjRND_SHADOW] = False
        rgb = self.renderer.render().copy()
        self.last_rgb = rgb
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        p = self.profile
        w, h = p["render_width"], p["render_height"]
        matrix = [self.fy, 0, w / 2, 0, self.fy, h / 2, 0, 0, 1]
        color, metric = reproject_intrinsics(
            torch.from_numpy(rgb[None]), torch.from_numpy(depth[None]), matrix, scaled_intrinsics(p, w, h)
        )
        return pack_zed_rgbd(color, metric, p)[0][0].numpy()

    def close(self):
        self.renderer.close()
