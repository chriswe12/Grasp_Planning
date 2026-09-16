"""Build the video-derived modular lab. Run with Blender's Python, no addons.

blender -b --python scripts/blender/build_video_lab.py -- --output assets/scenes/video_lab --render
Dimensions are estimates; z=0 is the table support plane, not the room floor.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))


def linear(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def material(name, rgb, rough=0.5, metal=0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*[linear(c) for c in rgb], 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    m.diffuse_color = (*[linear(c) for c in rgb], 1)
    m["canonical_srgb"] = list(rgb)
    return m


def collection(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def move(obj, col, parent=None):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    if parent:
        obj.parent = parent
    return obj


def empty(name, col, pos=(0, 0, 0)):
    o = bpy.data.objects.new(name, None)
    col.objects.link(o)
    o.location = pos
    o.empty_display_size = 0.10
    return o


def cube(name, loc, size, mat, col, bevel=0.001, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    move(o, col, parent)
    if mat:
        o.data.materials.append(mat)
    if bevel:
        b = o.modifiers.new("Small manufactured edge radii", "BEVEL")
        b.width = bevel
        b.segments = 3
        b = o.modifiers.new("Face normals", "WEIGHTED_NORMAL")
    return o


def cylinder(name, loc, radius, depth, mat, col, parent=None, vertices=48):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc)
    o = bpy.context.object
    o.name = name
    move(o, col, parent)
    if mat:
        o.data.materials.append(mat)
    b = o.modifiers.new("Edge radius", "BEVEL")
    b.width = min(0.001, radius * 0.12, depth * 0.15)
    b.segments = 3
    o.modifiers.new("Face normals", "WEIGHTED_NORMAL")
    return o


def line(name, points, radius, mat, col, parent=None):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.resolution_u = 16
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    s = cu.splines.new("BEZIER")
    s.bezier_points.add(len(points) - 1)
    for p, co in zip(s.bezier_points, points):
        p.co = co
        p.handle_left_type = "AUTO"
        p.handle_right_type = "AUTO"
    o = bpy.data.objects.new(name, cu)
    col.objects.link(o)
    if parent:
        o.parent = parent
    cu.materials.append(mat)
    return o


def point_at(o, target):
    o.rotation_euler = (Vector(target) - o.location).to_track_quat("-Z", "Y").to_euler()


def texture_image(name, array, folder, data=False):
    h, w = array.shape[:2]
    im = bpy.data.images.new(name, width=w, height=h, alpha=True)
    if data:
        im.colorspace_settings.name = "Non-Color"
    if not data:
        array = array.copy()
        rgb = array[:, :, :3]
        array[:, :, :3] = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)
    im.pixels.foreach_set(array.astype(np.float32).ravel())
    im.filepath_raw = str(folder / (name + ".png"))
    im.file_format = "PNG"
    im.save()
    return im


def laminate_texture(mat, folder, roughness):
    """Authored texture, not a photograph: avoids baking robot/shadows into the table."""
    n = 1536
    rng = np.random.default_rng(732)
    yy, xx = np.mgrid[0:n, 0:n] / n
    # Microscopic mottling and sparse, directional handling scratches.
    field = rng.normal(0, 0.003, (n, n))
    field += 0.005 * np.sin(xx * 27 + np.sin(yy * 18)) + 0.003 * np.cos(yy * 35)
    for _ in range(90):
        cx, cy = rng.uniform(0, 1, 2)
        sx = rng.uniform(0.008, 0.07)
        sy = rng.uniform(0.00018, 0.0006)
        field -= rng.uniform(0.005, 0.025) * np.exp(-(((xx - cx) / sx) ** 2) - ((yy - cy) / sy) ** 2)
    base = mat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value[:3]
    rgba = np.ones((n, n, 4))
    rgba[:, :, :3] = np.clip(np.array(base)[None, None, :] + field[:, :, None], 0, 1)
    color = texture_image("laminate_albedo", rgba, folder)
    rgba[:, :, :3] = np.clip(roughness + field[:, :, None] * 2, 0.18, 0.8)
    rough = texture_image("laminate_roughness", rgba, folder, True)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    p = nodes.get("Principled BSDF")
    t = nodes.new("ShaderNodeTexImage")
    t.name = "Laminate_Albedo"
    t.image = color
    links.new(t.outputs["Color"], p.inputs["Base Color"])
    # Multiply is exposed for Blender randomization. USD adapter separately authors
    # the equivalent UsdUVTexture.scale so exported albedo stays randomizable.
    mult = nodes.new("ShaderNodeMixRGB")
    mult.name = "Table_Tint"
    mult.blend_type = "MULTIPLY"
    mult.inputs[0].default_value = 1
    mult.inputs[2].default_value = (1, 1, 1, 1)
    links.new(t.outputs["Color"], mult.inputs[1])
    links.new(mult.outputs[0], p.inputs["Base Color"])
    t = nodes.new("ShaderNodeTexImage")
    t.name = "Laminate_Roughness"
    t.image = rough
    links.new(t.outputs["Color"], p.inputs["Roughness"])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1600
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.075
    bump.inputs["Distance"].default_value = 0.000045
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], p.inputs["Normal"])


def tabletop_uv(o, w, d):
    uv = o.data.uv_layers.active
    for poly in o.data.polygons:
        for li in poly.loop_indices:
            p = o.data.vertices[o.data.loops[li].vertex_index].co + o.location
            uv.data[li].uv = (p.x / w + 0.5, p.y / d + 0.5)


def hole(obj, loc, r, depth, col):
    cutter = cylinder("Boolean_cutter", loc, r, depth, None, col)
    # Apply only boolean; leave bevel until after cut.
    b = obj.modifiers.new("Machined hole", "BOOLEAN")
    b.operation = "DIFFERENCE"
    b.object = cutter
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=b.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def create_props(col, mats):
    roots = []
    root = empty("Blue_FDM_Flange", col)
    roots.append(root)
    base = cube("Blue_flange_base", (0, 0, 0.005), (0.105, 0.086, 0.010), mats["blue"], col, 0.002, root)
    for x in [-0.043, 0.043]:
        for y in [-0.034, 0.034]:
            hole(base, (x, y, 0.005), 0.0032, 0.04, col)
    cube("Blue_flange_raised_center", (0, 0, 0.014), (0.077, 0.058, 0.010), mats["blue"], col, 0.0015, root)
    for x in [-0.038, 0.038]:
        cube("Blue_raised_edge", (x, 0, 0.018), (0.006, 0.062, 0.018), mats["blue"], col, 0.001, root)
    for x in [-0.021, 0.021]:
        cylinder("Blue_recess_dark", (x, 0, 0.0192), 0.0037, 0.00035, mats["dark"], col, root)
    # Slightly visible FDM perimeter layer ridges, all stay with their movable part.
    for z in np.arange(0.011, 0.02, 0.001):
        cube("FDM_layer", (0, -0.0291, float(z)), (0.070, 0.00022, 0.00018), mats["blue"], col, 0.00006, root)
    root["mass_kg"] = 0.06
    root["footprint_radius_m"] = 0.074
    root = empty("Brown_FDM_Bracket", col)
    roots.append(root)
    base = cube("Brown_base", (0, 0, 0.005), (0.085, 0.068, 0.010), mats["brown"], col, 0.001, root)
    plate = cube("Brown_upright", (0, 0.021, 0.032), (0.082, 0.012, 0.054), mats["brown"], col, 0.001, root)
    cut = cylinder("Cut", (0, 0.021, 0.037), 0.015, 0.05, None, col)
    cut.rotation_euler.x = math.pi / 2
    b = plate.modifiers.new("Central bore", "BOOLEAN")
    b.object = cut
    b.operation = "DIFFERENCE"
    bpy.context.view_layer.objects.active = plate
    bpy.ops.object.modifier_apply(modifier=b.name)
    bpy.data.objects.remove(cut, do_unlink=True)
    for x in [-0.03, 0.03]:
        hole(base, (x, -0.013, 0.005), 0.003, 0.04, col)
    root["mass_kg"] = 0.07
    root["footprint_radius_m"] = 0.062
    root = empty("Green_Plastic_Cap", col)
    roots.append(root)
    cylinder("Green_cap", (0, 0, 0.027), 0.024, 0.054, mats["green"], col, root)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=40, ring_count=20, radius=1, location=(0, 0, 0.052))
    o = bpy.context.object
    o.name = "Green_round_top"
    o.scale = (0.024, 0.024, 0.009)
    move(o, col, root)
    o.data.materials.append(mats["green"])
    for f in o.data.polygons:
        f.use_smooth = True
    root["mass_kg"] = 0.025
    root["footprint_radius_m"] = 0.032
    root = empty("Yellow_Tape_Roll", col)
    roots.append(root)
    roll = cylinder("Yellow_tape", (0, 0, 0.022), 0.043, 0.044, mats["yellow"], col, root)
    hole(roll, (0, 0, 0.022), 0.033, 0.08, col)
    inner = cylinder("Cardboard_core", (0, 0, 0.0215), 0.0331, 0.043, mats["cardboard"], col, root)
    hole(inner, (0, 0, 0.022), 0.031, 0.08, col)
    root["mass_kg"] = 0.09
    root["footprint_radius_m"] = 0.048
    root = empty("Red_Pencil", col)
    roots.append(root)
    o = cylinder("Red_pencil_shaft", (0, 0, 0.004), 0.0037, 0.155, mats["red"], col, root, vertices=6)
    o.rotation_euler.y = math.pi / 2
    bpy.ops.mesh.primitive_cone_add(
        vertices=16, radius1=0.0037, radius2=0.0004, depth=0.018, location=(0.0865, 0, 0.004)
    )
    o = bpy.context.object
    o.name = "Pencil_wood_tip"
    o.rotation_euler.y = math.pi / 2
    move(o, col, root)
    o.data.materials.append(mats["cardboard"])
    root["mass_kg"] = 0.006
    root["footprint_radius_m"] = 0.10
    poses = [(-0.32, -0.16, 0.10), (-0.60, -0.08, -0.20), (-0.70, -0.18, 0), (0.52, -0.15, 0), (0.32, -0.24, 0.60)]
    for root, (x, y, rz) in zip(roots, poses):
        root.location = (x, y, 0.0004)
        root.rotation_euler.z = rz
    return roots


def build(cfg, out):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1
    cols = {
        n: collection(n)
        for n in [
            "Table",
            "Surface_Wear",
            "Room",
            "Background_Furniture",
            "Optional_Props",
            "Lighting",
            "Cameras",
            "Anchors",
            "Collision",
        ]
    }
    m = {
        "table": material("Table_Laminate", cfg["table"]["color_srgb"], cfg["table"]["roughness"]),
        "edge": material("ABS_Edge_Banding", (0.67, 0.665, 0.63), 0.45),
        "wall": material("Warm_White_Partition", cfg["wall"]["color_srgb"], 0.82),
        "trim": material("Satin_Aluminum_Extrusion", (0.68, 0.69, 0.70), 0.32, 0.75),
        "frame": material("Powder_Coated_Table_Frame", (0.63, 0.65, 0.65), 0.52, 0.18),
        "floor": material("Charcoal_Speckled_Vinyl", (0.19, 0.205, 0.205), 0.84),
        "dark": material("Black_Rubber", (0.045, 0.048, 0.05), 0.59),
        "white": material("Socket_Plastic", (0.87, 0.87, 0.85), 0.32),
        "tape": material("Old_Masking_Tape", (0.76, 0.74, 0.67), 0.73),
        "residue": material("Adhesive_Remnant", (0.59, 0.57, 0.51), 0.44),
        "scuff": material("Graphite_Transfer", (0.50, 0.49, 0.445), 0.73),
        "blue": material("Muted_Blue_FDM", (0.018, 0.34, 0.50), 0.46),
        "brown": material("Brown_FDM", (0.29, 0.19, 0.145), 0.58),
        "green": material("Lime_Green_Plastic", (0.36, 0.66, 0.045), 0.42),
        "yellow": material("Ochre_Yellow_Tape", (0.79, 0.61, 0.055), 0.36),
        "cardboard": material("Cardboard", (0.58, 0.43, 0.29), 0.82),
        "red": material("Red_Pencil_Lacquer", (0.55, 0.035, 0.045), 0.36),
        "screen": material("Monitor_Dark_Glass", (0.025, 0.032, 0.037), 0.21),
    }
    texdir = out / "textures"
    texdir.mkdir(exist_ok=True, parents=True)
    laminate_texture(m["table"], texdir, cfg["table"]["roughness"])
    # Physical microstructure on background surfaces (Blender); portable base PBR in USD.
    for key, scale, strength, dist in [
        ("floor", 800, 0.20, 0.0005),
        ("wall", 380, 0.12, 0.00018),
        ("blue", 900, 0.11, 0.00008),
        ("brown", 900, 0.11, 0.00008),
    ]:
        nt = m[key].node_tree
        p = nt.nodes.get("Principled BSDF")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = scale
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = strength
        bump.inputs["Distance"].default_value = dist
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    rng_floor = np.random.default_rng(401)
    rgba = np.ones((1024, 1024, 4), dtype=np.float32)
    base = np.array([linear(c) for c in (0.19, 0.205, 0.205)])
    grit = rng_floor.normal(0, 0.005, (1024, 1024, 1))
    rgba[:, :, :3] = np.clip(base + grit, 0.003, 0.12)
    im = texture_image("floor_vinyl_albedo", rgba, texdir)
    nt = m["floor"].node_tree
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = im
    nt.links.new(node.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    w = cfg["table"]["width"]
    d = cfg["table"]["depth"]
    h = cfg["table"]["height"]
    t = cfg["table"]["thickness"]
    gap = cfg["table"]["seam_width"]
    table = empty("Table", cols["Table"])
    table["support_plane_z_m"] = 0.0
    table["estimated_dimensions_m"] = [w, d, h]
    for sign in [-1, 1]:
        o = cube(
            "Laminate_Left" if sign < 0 else "Laminate_Right",
            (sign * (w / 4 + gap / 4), 0, -t / 2),
            (w / 2 - gap / 2, d, t),
            m["table"],
            cols["Table"],
            0.00065,
            table,
        )
        tabletop_uv(o, w, d)
    # Front edge banding is narrow and slightly cooler than the warm laminate.
    cube(
        "Front_ABS_edge", (0, -d / 2 - 0.00035, -t / 2), (w, 0.0014, t - 0.001), m["edge"], cols["Table"], 0.0004, table
    )
    cube("Rear_ABS_edge", (0, d / 2 + 0.00035, -t / 2), (w, 0.0014, t - 0.001), m["edge"], cols["Table"], 0.0004, table)
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            x = sx * (w / 2 - 0.095)
            y = sy * (d / 2 - 0.075)
            cube(
                "Steel_square_leg",
                (x, y, -h / 2 - 0.02),
                (0.05, 0.05, h - 0.06),
                m["frame"],
                cols["Table"],
                0.002,
                table,
            )
            cylinder("Adjustable_rubber_foot", (x, y, -h + 0.012), 0.032, 0.024, m["dark"], cols["Table"], table)
        cube(
            "End_apron",
            (sx * (w / 2 - 0.095), 0, -0.063),
            (0.048, d - 0.11, 0.076),
            m["frame"],
            cols["Table"],
            0.001,
            table,
        )
    for sy in [-1, 1]:
        cube(
            "Long_apron",
            (0, sy * (d / 2 - 0.075), -0.063),
            (w - 0.16, 0.048, 0.076),
            m["frame"],
            cols["Table"],
            0.001,
            table,
        )
    # Black inset caps seen along the rear mounting edge; kept detachable.
    for x in [-0.69, -0.53, 0.48, 0.65]:
        o = cylinder("Cable_grommet", (x, d / 2 - 0.095, 0.00045), 0.027, 0.0009, m["dark"], cols["Surface_Wear"])
        line(
            "Grommet_slot",
            [(x - 0.015, d / 2 - 0.095, 0.001), (x + 0.015, d / 2 - 0.095, 0.001)],
            0.00065,
            m["frame"],
            cols["Surface_Wear"],
        )
    # Scuffs and adhesive patches inferred from the moving close-ups.
    rng = random.Random(27)
    for i, (x, y, sx, sy) in enumerate(
        [
            (0.37, -0.06, 0.065, 0.03),
            (0.15, 0.12, 0.022, 0.011),
            (-0.05, 0.03, 0.034, 0.018),
            (0.57, 0.06, 0.018, 0.034),
            (-0.44, 0.22, 0.015, 0.031),
        ]
    ):
        o = cube(f"Tape_Remnant_{i}", (x, y, 0.00008), (sx, sy, 0.00009), m["tape"], cols["Surface_Wear"], 0.00004)
        o.rotation_euler.z = rng.uniform(-0.16, 0.16)
    for i in range(46):
        x = rng.gauss(0.35, 0.12)
        y = rng.gauss(0.14, 0.085)
        if abs(x) > w / 2 - 0.04 or abs(y) > d / 2 - 0.04:
            continue
        o = cube(
            f"Handling_Scuff_{i}",
            (x, y, 0.000055),
            (rng.uniform(0.004, 0.022), rng.uniform(0.0004, 0.001), 0.000035),
            m["residue"] if i % 3 else m["scuff"],
            cols["Surface_Wear"],
            0.000015,
        )
        o.rotation_euler.z = rng.uniform(-0.12, 0.12)
    for i in range(7):
        cube(
            "Faded_fixture_outline",
            (0.30 + i * 0.016, 0.16, 0.00006),
            (0.00035, 0.070, 0.00003),
            m["residue"],
            cols["Surface_Wear"],
            0,
        )
    rear = cfg["wall"]["rear_y"]
    wh = cfg["wall"]["height"]
    room = empty("Room", cols["Room"])
    cube("Room_floor", (0, -0.8, -h - 0.04), (4.6, 4.4, 0.08), m["floor"], cols["Room"], 0.001, room)
    cube("Rear_partition", (0.57, rear + 0.04, wh / 2 - h), (2.82, 0.08, wh), m["wall"], cols["Room"], 0.002, room)
    # Open doorway at rear left, darker hall beyond.
    for x in [-1.45, -0.86, 1.05]:
        cube(
            "Partition_vertical_frame",
            (x, rear - 0.013, wh / 2 - h),
            (0.034, 0.033, wh),
            m["trim"],
            cols["Room"],
            0.001,
            room,
        )
    cube("Door_header", (-1.155, rear - 0.013, 1.38), (0.62, 0.05, 0.04), m["trim"], cols["Room"], 0.001, room)
    cube("Hall_wall", (-1.55, 1.22, 0.50), (1.1, 0.08, 2.5), m["wall"], cols["Room"], 0.001, room)
    cube("Left_partition", (-1.50, -0.35, 0.53), (0.075, 1.75, 2.55), m["wall"], cols["Room"], 0.001, room)
    cube("Rear_skirting", (0.57, rear - 0.008, -h + 0.045), (2.82, 0.016, 0.09), m["frame"], cols["Room"], 0.001, room)
    cube("Door_threshold", (-1.15, rear - 0.10, -h + 0.004), (0.58, 0.17, 0.008), m["trim"], cols["Room"], 0.001, room)
    # Swiss-style outlet/switch panel visible above the table.
    cube("Outlet_faceplate", (-0.61, rear - 0.050, 0.31), (0.082, 0.013, 0.155), m["white"], cols["Room"], 0.004, room)
    for z in [0.285, 0.345]:
        o = cylinder("Outlet_inset", (-0.61, rear - 0.058, z), 0.022, 0.002, m["edge"], cols["Room"], room)
        o.rotation_euler.x = math.pi / 2
        for dx, dz in [(-0.010, 0), (0.010, 0), (0, -0.008)]:
            o = cylinder(
                "Outlet_pin_hole", (-0.61 + dx, rear - 0.060, z + dz), 0.002, 0.001, m["dark"], cols["Room"], room
            )
            o.rotation_euler.x = math.pi / 2
    line(
        "Wall_power_lead",
        [(-0.61, rear - 0.061, 0.27), (-0.65, rear - 0.08, 0.13), (-0.65, rear - 0.08, 0.03), (-0.56, 0.34, 0.004)],
        0.003,
        m["white"],
        cols["Room"],
        room,
    )
    # Background workbench from the video orbit. Separate collection, freely removable.
    desk = empty("Background_Workbench", cols["Background_Furniture"], (1.48, 0.15, 0))
    cube("Desk_top", (0, 0, -0.06), (0.90, 0.60, 0.025), m["edge"], cols["Background_Furniture"], 0.002, desk)
    for x in [-0.34, 0.34]:
        cube("Desk_leg", (x, 0.04, -0.42), (0.045, 0.055, 0.69), m["dark"], cols["Background_Furniture"], 0.002, desk)
        cube(
            "Desk_foot",
            (x, -0.03, -h + 0.015),
            (0.065, 0.53, 0.03),
            m["dark"],
            cols["Background_Furniture"],
            0.003,
            desk,
        )
    cube("Monitor_foot", (0, 0.12, -0.043), (0.24, 0.19, 0.016), m["dark"], cols["Background_Furniture"], 0.004, desk)
    cube("Monitor_stand", (0, 0.18, 0.065), (0.045, 0.045, 0.23), m["dark"], cols["Background_Furniture"], 0.003, desk)
    cube("Monitor_bezel", (0, 0.19, 0.27), (0.49, 0.036, 0.295), m["dark"], cols["Background_Furniture"], 0.007, desk)
    cube(
        "Monitor_screen",
        (0, 0.169, 0.272),
        (0.466, 0.003, 0.264),
        m["screen"],
        cols["Background_Furniture"],
        0.002,
        desk,
    )
    cube("Keyboard", (0, -0.17, -0.036), (0.36, 0.13, 0.024), m["dark"], cols["Background_Furniture"], 0.004, desk)
    for i in range(14):
        for j in range(4):
            cube(
                "Keycap",
                (-0.156 + i * 0.023, -0.214 + j * 0.024, -0.020),
                (0.017, 0.018, 0.004),
                m["frame"],
                cols["Background_Furniture"],
                0.001,
                desk,
            )
    for i, x in enumerate([1.32, 1.52]):
        line(
            "Loose_floor_cable",
            [
                (x, 0.2, -0.1),
                (x, 0.08, -0.70),
                (x + 0.12, -0.2, -h + 0.004),
                (x - 0.24, -0.48, -h + 0.004),
                (x - 0.15, -0.7, -h + 0.004),
                (x + 0.22, -0.65, -h + 0.004),
            ],
            0.003,
            m["dark"] if i == 0 else m["white"],
            cols["Background_Furniture"],
        )
    props = create_props(cols["Optional_Props"], m)
    for name, loc in [
        ("Robot_Mount_Left", (-w / 2 + 0.20, d / 2 - 0.17, 0)),
        ("Robot_Mount_Right", (w / 2 - 0.20, d / 2 - 0.17, 0)),
        ("Task_Origin", (0, -0.10, 0)),
    ]:
        o = empty(name, cols["Anchors"], loc)
        o["purpose"] = "Placement guide only; estimated, not calibrated"
    # Exact support slab, independent of decals and seam. Collision USD purpose=guide.
    o = cube("Table_Support_Collision", (0, 0, -t / 2), (w, d, t), None, cols["Collision"], 0)
    o.hide_render = True
    o.display_type = "WIRE"
    o["collision_role"] = "table"
    o = cube("Floor_Collision", (0, -0.8, -h - 0.04), (4.6, 4.4, 0.08), None, cols["Collision"], 0)
    o.hide_render = True
    o.display_type = "WIRE"
    o["collision_role"] = "floor"
    for name, loc, power, size, color, target in [
        ("Ceiling_Softbox", (-0.2, -0.4, 1.85), 77, 2.1, (1, 0.95, 0.87), (0, 0, 0)),
        ("Window_Fill", (-1.25, -0.8, 0.9), 36, 1.5, (0.85, 0.92, 1), (0, 0, 0.1)),
        ("Room_Bounce", (1.2, 0.1, 1.6), 34, 1.6, (1, 1, 0.98), (0, 0, 0)),
    ]:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = power
        data.shape = "RECTANGLE"
        data.size = size
        data.size_y = size * 0.6
        data.color = color
        o = bpy.data.objects.new(name, data)
        cols["Lighting"].objects.link(o)
        o.location = loc
        point_at(o, target)
        o["nominal_energy"] = power
    world = bpy.data.worlds.new("Neutral_Room_Ambient")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.65, 0.70, 0.78, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.09
    scene.world = world
    for name, loc, target, lens in [
        ("Camera_Overview", (1.65, -2.15, 1.35), (0, 0.02, -0.05), 36),
        ("Camera_Table_Detail", (0.63, -1.04, 0.85), (0.15, 0.08, 0), 40),
        ("Camera_Reference", (-1.00, -1.15, 0.91), (-0.05, 0.13, 0.02), 34),
    ]:
        data = bpy.data.cameras.new(name)
        o = bpy.data.objects.new(name, data)
        cols["Cameras"].objects.link(o)
        o.location = loc
        point_at(o, target)
        data.lens = lens
        data.clip_start = 0.01
        data.clip_end = 100
        o["nominal_location"] = list(loc)
        o["nominal_rotation"] = list(o.rotation_euler)
        o["nominal_lens"] = lens
    scene.camera = bpy.data.objects["Camera_Overview"]
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene["source_video"] = cfg["reference_video"]
    scene["dimension_status"] = cfg["dimension_status"]
    scene["table_top_z"] = 0.0
    scene["room_floor_z"] = -h
    scene["randomization_seed"] = 0
    scene["randomization_config"] = json.dumps(cfg["randomization"])
    scene["asset_notes"] = (
        "Environment excludes robot and props. Optional_Props can be hidden. Use randomize_video_lab.py. All dimensions estimated."
    )
    # Open straight into the useful camera view, with unobtrusive guides hidden.
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
                area.spaces.active.overlay.show_extras = False
                area.spaces.active.shading.type = "MATERIAL"
    return props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "assets/scenes/video_lab")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--resolution", type=int, default=1200)
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.config or ROOT / "assets/scenes/video_lab/scene_config.json").read_text())
    props = build(cfg, out)
    bpy.context.scene.cycles.samples = args.samples
    bpy.context.scene.render.resolution_x = args.resolution
    bpy.context.scene.render.resolution_y = round(args.resolution * 0.75)
    # Export clean environment, separate props, and authored physics schemas.
    from video_lab_usd import export_package

    export_package(out, props, cfg)
    for im in bpy.data.images:
        if im.source == "FILE" and im.filepath:
            im.pack()
    for filename in ["randomize_video_lab.py"]:
        txt = bpy.data.texts.load(str(Path(__file__).with_name(filename)))
        txt.use_module = False
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "video_lab.blend"))
    if args.render:
        for camera, name in [("Camera_Overview", "preview_overview"), ("Camera_Table_Detail", "preview_table_detail")]:
            bpy.context.scene.camera = bpy.data.objects[camera]
            bpy.context.scene.render.filepath = str(out / (name + ".png"))
            bpy.ops.render.render(write_still=True)
    print("VIDEO_LAB_BUILD_OK", out)


if __name__ == "__main__":
    main()
