"""Run CPU structural/portability checks using Blender's bundled USD Python."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import bpy
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).parent))
spec=importlib.util.spec_from_file_location('video_lab_scene',ROOT/'grasp_planning/rl/video_lab_scene.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
add_environment=module.add_environment; randomize_environment=module.randomize_environment
add_mock_props=module.add_mock_props; sample_prop_layout=module.sample_prop_layout
from randomize_video_lab import randomize


def main():
    folder=ROOT/'assets/scenes/video_lab'; manifest=json.loads((folder/'manifest.json').read_text())
    stage=Usd.Stage.Open(str(folder/'environment.usdc'))
    assert UsdGeom.GetStageMetersPerUnit(stage)==1 and UsdGeom.GetStageUpAxis(stage)=='Z'
    assert stage.GetDefaultPrim().GetPath().pathString=='/Lab'
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
    assert not any('Blue_FDM' in str(p.GetPath()) or 'panda' in str(p.GetPath()).lower() for p in stage.Traverse())
    collision=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(collision)==2
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','guide'])
    support=next(p for p in collision if 'Table_Support' in str(p.GetPath()))
    bounds=cache.ComputeWorldBound(support).ComputeAlignedRange()
    assert abs(bounds.GetMax()[2])<1e-7, bounds
    assert abs((bounds.GetMax()[0]-bounds.GetMin()[0])-manifest['dimensions']['width'])<1e-6
    for p in collision: assert UsdGeom.Imageable(p).GetPurposeAttr().Get()=='guide'
    files,assets,unresolved=UsdUtils.ComputeAllDependencies(str(folder/'preview.usda'))
    assert not unresolved, unresolved
    for info in manifest['props']:
        ps=Usd.Stage.Open(str(folder/info['asset'])); root=ps.GetDefaultPrim()
        assert root.HasAPI(UsdPhysics.RigidBodyAPI)
        assert abs(UsdPhysics.MassAPI(root).GetMassAttr().Get()-info['mass_kg'])<1e-6
        for mesh in ps.Traverse():
            if mesh.IsA(UsdGeom.Mesh): assert mesh.HasAPI(UsdPhysics.CollisionAPI)
    # Bounds/exclusions over enough seeds to encounter empty and full layouts.
    counts=set()
    cfg=manifest['randomization']; w=manifest['dimensions']['width']; d=manifest['dimensions']['depth']
    for seed in range(200):
        layout=sample_prop_layout(seed,manifest); assert layout==sample_prop_layout(seed,manifest); counts.add(len(layout))
        for i,p in enumerate(layout):
            x,y,_=p['position']; r=p['radius']
            assert abs(x)+r<w/2 and abs(y)+r<d/2
            exclusions=cfg['reserved_robot_mounts_xy_radius']+[cfg['reserved_task_center_xy_radius']]
            exclusions += [(q['position'][0],q['position'][1],q['radius']) for q in layout[:i]]
            assert all(math.hypot(x-a,y-b)>=r+c+.02-1e-9 for a,b,c in exclusions)
    assert 0 in counts and max(counts)>=4
    digest=hashlib.sha256((folder/'environment.usdc').read_bytes()).hexdigest()
    # Two differently tinted references must stay independent; source unchanged.
    test=Usd.Stage.CreateInMemory()
    add_environment(test,'/World/Env0/Lab',folder); add_environment(test,'/World/Env1/Lab',folder)
    a=randomize_environment(test,'/World/Env0/Lab',7,folder)
    before=test.GetRootLayer().ExportToString()
    assert a==randomize_environment(test,'/World/Env0/Lab',7,folder)
    assert before==test.GetRootLayer().ExportToString()
    b=randomize_environment(test,'/World/Env1/Lab',23,folder); assert a!=b
    assert a==json.loads(test.GetPrimAtPath('/World/Env0/Lab').GetAttribute('lab:appearanceSample').Get())
    for index,sample in [(0,a),(1,b)]:
        shader=UsdShade.Shader(test.GetPrimAtPath(f'/World/Env{index}/Lab/_materials/Table_Laminate/Laminate_Albedo'))
        assert abs(shader.GetInput('scale').Get()[0]-sample['table_color_gain'])<1e-6
        local_support=test.GetPrimAtPath(f'/World/Env{index}/Lab'+str(support.GetPath())[4:])
        assert UsdGeom.BBoxCache(Usd.TimeCode.Default(),['guide']).ComputeWorldBound(local_support).ComputeAlignedRange()==bounds
    add_mock_props(test,'/World/Env0/Props',23,folder)
    assert digest==hashlib.sha256((folder/'environment.usdc').read_bytes()).hexdigest()
    assert cache.ComputeWorldBound(support).ComputeAlignedRange()==bounds
    test.GetRootLayer().Export('/tmp/video_lab/validated_composition.usda')
    # Blender: repeated seeds reproduce material/light/object values.
    bpy.ops.wm.open_mainfile(filepath=str(folder/'video_lab.blend'))
    a=randomize(7); first=bpy.data.materials['Warm_White_Partition'].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value[:]
    randomize(23); assert randomize(7)==a
    assert bpy.data.materials['Warm_White_Partition'].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value[:]==first
    assert abs(bpy.data.objects['Table_Support_Collision'].location.z+manifest['dimensions']['thickness']/2)<1e-7
    for seed in range(200):
        poses=randomize(seed)['props']
        for i,p in enumerate(poses):
            assert abs(p['x'])+p['radius']<w/2 and abs(p['y'])+p['radius']<d/2
            exclusions=cfg['reserved_robot_mounts_xy_radius']+[cfg['reserved_task_center_xy_radius']]
            exclusions += [(q['x'],q['y'],q['radius']) for q in poses[:i]]
            assert all(math.hypot(p['x']-x,p['y']-y)>=p['radius']+r+.02-1e-9 for x,y,r in exclusions)
    result=dict(status='passed',blender_layout_seeds_checked=200,usd_dependencies_resolved=True,static_colliders=2,props_checked=len(manifest['props']),layout_seeds_checked=200,observed_prop_counts=sorted(counts),table_support_z_m=bounds.GetMax()[2],randomization_repeatable=True,per_environment_materials_independent=True,source_asset_unchanged=True,blender_version=bpy.app.version_string,isaac_simulation='Not run: NVIDIA driver unavailable in this session')
    (folder/'validation.json').write_text(json.dumps(result,indent=2)+'\n'); print('VIDEO_LAB_VALIDATION_OK',json.dumps(result))

if __name__=='__main__': main()
