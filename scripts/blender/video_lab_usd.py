"""Portable USD export with explicit static support and movable prop contracts."""
import json
from pathlib import Path
import bpy
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf


def export_selection(path, objects, root):
    bpy.ops.object.select_all(action='DESELECT')
    hidden=[]
    for o in objects:
        if o.hide_render: hidden.append(o); o.hide_render=False
        o.select_set(True)
    bpy.ops.wm.usd_export(filepath=str(path), selected_objects_only=True,
        root_prim_path=root, export_materials=True, generate_preview_surface=True,
        generate_materialx_network=False, export_textures_mode='NEW',
        relative_paths=True, overwrite_textures=True, export_custom_properties=True,
        export_cameras=True, export_lights=True, convert_world_material=False,
        merge_parent_xform=False, evaluation_mode='RENDER')
    for o in hidden: o.hide_render=True
    stage=Usd.Stage.Open(str(path)); stage.SetDefaultPrim(stage.GetPrimAtPath(root))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1.0)
    stage.GetRootLayer().Save()
    return stage


def physics_material(stage):
    p=UsdShade.Material.Define(stage,'/Lab/Physics/TableContact')
    api=UsdPhysics.MaterialAPI.Apply(p.GetPrim())
    api.CreateStaticFrictionAttr(.8); api.CreateDynamicFrictionAttr(.6); api.CreateRestitutionAttr(0.)
    return p


def export_package(out, props, cfg):
    # Exporter understands direct PBR image inputs. Restore editable Blender tint
    # immediately afterwards; USD uses the texture's scale for its tint instead.
    nt=bpy.data.materials['Table_Laminate'].node_tree; p=nt.nodes.get('Principled BSDF')
    nt.links.new(nt.nodes['Laminate_Albedo'].outputs['Color'],p.inputs['Base Color'])
    names=['Table','Surface_Wear','Room','Background_Furniture','Collision','Anchors']
    objects=[o for n in names for o in bpy.data.collections[n].objects]
    stage=export_selection(out/'environment.usdc',objects,'/Lab')
    material=physics_material(stage); collision_paths=[]
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and ('Table_Support_Collision' in str(prim.GetPath()) or 'Floor_Collision' in str(prim.GetPath())):
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
            collision_paths.append(str(prim.GetPath()))
    stage.GetRootLayer().Save()
    export_selection(out/'lighting.usdc',list(bpy.data.collections['Lighting'].objects),'/Lighting')
    export_selection(out/'cameras.usdc',list(bpy.data.collections['Cameras'].objects),'/Cameras')
    prop_info=[]; folder=out/'props'; folder.mkdir(exist_ok=True)
    for root in props:
        loc=root.location.copy(); rot=root.rotation_euler.copy()
        root.location=(0,0,0); root.rotation_euler=(0,0,0)
        asset=folder/(root.name+'.usdc')
        ps=export_selection(asset,[root]+list(root.children_recursive),'/Prop')
        prim=ps.GetDefaultPrim(); UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(root['mass_kg'])
        for child in ps.Traverse():
            if child.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr('convexHull')
        ps.GetRootLayer().Save(); root.location=loc; root.rotation_euler=rot
        prop_info.append(dict(name=root.name,asset=str(asset.relative_to(out)),mass_kg=root['mass_kg'],footprint_radius_m=root['footprint_radius_m'],preview_position=list(loc),preview_yaw_rad=rot.z))
    nt.links.new(nt.nodes['Table_Tint'].outputs[0],p.inputs['Base Color'])
    # A composition for inspection, environment.usdc remains clean and standalone.
    stage=Usd.Stage.CreateNew(str(out/'preview.usda')); root=UsdGeom.Xform.Define(stage,'/World'); stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.DefinePrim('/World/Lab').GetReferences().AddReference('environment.usdc')
    stage.DefinePrim('/World/Lighting').GetReferences().AddReference('lighting.usdc')
    stage.DefinePrim('/World/Cameras').GetReferences().AddReference('cameras.usdc')
    for info in prop_info:
        obj=UsdGeom.Xform.Define(stage,'/World/Props/'+info['name']); obj.GetPrim().GetReferences().AddReference(info['asset'])
        obj.AddTranslateOp().Set(Gf.Vec3d(*info['preview_position'])); obj.AddRotateZOp().Set(info['preview_yaw_rad']*180/3.141592653589793)
    stage.GetRootLayer().Save()
    manifest=dict(schema_version=1,environment='environment.usdc',blender='video_lab.blend',root_prim='/Lab',up_axis='Z',meters_per_unit=1.,table_top_z_m=0.,room_floor_z_m=-cfg['table']['height'],dimensions=cfg['table'],dimension_status=cfg['dimension_status'],collision_paths=collision_paths,props=prop_info,randomization=cfg['randomization'],limitations=['Video-derived approximation, no metric calibration','USD uses portable PreviewSurface PBR; procedural Blender micro-bump is not exported','Prop collisions use per-mesh convex hulls; bores are visual, not insertion-accurate','No robot or fixed training task is embedded'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
