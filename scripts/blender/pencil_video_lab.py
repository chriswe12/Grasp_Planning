"""Add clearly visible, removable graphite strokes to a separate lab variant."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import bpy
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from build_video_lab import material
from video_lab_usd import export_selection


def hashes(folder):
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def make_pencil_marks(seed=42,density=1.,contrast=1.):
    rng=random.Random(seed)
    scene=bpy.context.scene
    col=bpy.data.collections.new('Pencil_Marks')
    bpy.data.collections['Surface_Wear'].children.link(col)
    parent=bpy.data.objects.new('Handling_Scuff_PencilMarks',None)
    col.objects.link(parent)
    mats=[]
    for i,value in enumerate([.29,.34,.39,.45,.50]):
        rgb=tuple(a*(1-contrast)+b*contrast for a,b in zip((.68,.65,.59),(value,value*.99,value*.97)))
        mats.append(material(f'Pencil_Graphite_{i}',rgb,.84))
    vertices=[]; faces=[]; mat_ids=[]; strokes=0
    # Exclude tape remnants, so pencil does not appear to float above masking tape.
    tapes=[(.37,-.06,.065,.03),(.15,.12,.022,.011),(-.05,.03,.034,.018),(.57,.06,.018,.034),(-.44,.22,.015,.031)]
    def blocked(x,y):
        return any(abs(x-a)<w/2+.007 and abs(y-b)<h/2+.007 for a,b,w,h in tapes)

    def stroke(points,width=.0007,shade=None):
        nonlocal strokes
        if shade is None: shade=rng.choices(range(5),weights=[1,3,4,3,2])[0]
        strokes+=1
        for i,((x1,y1),(x2,y2)) in enumerate(zip(points,points[1:])):
            if blocked((x1+x2)/2,(y1+y2)/2): continue
            if max(abs(x1),abs(x2))>.76 or max(abs(y1),abs(y2))>.36: continue
            dx=x2-x1; dy=y2-y1; length=math.hypot(dx,dy)
            if length<1e-7: continue
            # Pressure varies along each stroke; short skipped segments look rubbed off.
            if rng.random()<.055: continue
            taper=max(.38,math.sin(math.pi*(i+.5)/max(1,len(points)-1)))
            half=width*.5*rng.uniform(.68,1.25)*taper
            nx=-dy/length*half; ny=dx/length*half
            k=len(vertices)
            vertices.extend([(x1+nx,y1+ny,.00013),(x1-nx,y1-ny,.00013),(x2-nx,y2-ny,.00013),(x2+nx,y2+ny,.00013)])
            faces.append((k,k+1,k+2,k+3)); mat_ids.append(shade)

    def line(cx,cy,length,angle,width=.0007,shade=None):
        pts=[]
        for i in range(9):
            t=(i/8-.5)*length
            jitter=rng.uniform(-.00035,.00035)
            pts.append((cx+t*math.cos(angle)-jitter*math.sin(angle),cy+t*math.sin(angle)+jitter*math.cos(angle)))
        stroke(pts,width,shade)

    # More of the short, parallel graphite drags highlighted in the screenshot.
    for _ in range(round(140*density)):
        cx=rng.gauss(.34,.15); cy=rng.gauss(.12,.115)
        angle=rng.gauss(-.12,.34)
        for j in range(rng.choices([1,2,3],[5,3,1])[0]):
            line(cx,cy+j*rng.uniform(.0018,.0035),rng.uniform(.008,.040),angle,rng.uniform(.00042,.00105))
    # Hand-sketched fixture outline and offset retraces, slightly irregular.
    for offset,shade in [(0.,2),(.0018,4)]:
        x0,x1=.17+offset,.43+offset; y0,y1=.075+offset,.25+offset
        for cx,cy,length,angle in [((x0+x1)/2,y0,x1-x0,0),((x0+x1)/2,y1,x1-x0,0),(x0,(y0+y1)/2,y1-y0,math.pi/2),(x1,(y0+y1)/2,y1-y0,math.pi/2)]:
            line(cx,cy,length,angle,.00065,shade)
    for i in range(round(9*density)):
        x=.18+i*.027
        line(x,.16,rng.uniform(.10,.175),math.pi/2+rng.uniform(-.025,.025),rng.uniform(.0004,.0007),3)
    # Small cross-hatching / scrubbing clusters rather than a uniform noisy coating.
    for _ in range(round(11*density)):
        cx=rng.uniform(.06,.62); cy=rng.uniform(-.16,.27)
        for j in range(rng.randint(4,8)):
            line(cx+j*.002,cy+j*.0007,rng.uniform(.006,.019),rng.uniform(.5,.8),rng.uniform(.00045,.0009),rng.choice([1,2,3]))
    # A few curved scribble fragments: pencil slips, not printed text.
    for _ in range(round(8*density)):
        cx=rng.uniform(.10,.58); cy=rng.uniform(-.15,.23)
        sx=rng.uniform(.008,.018); sy=rng.uniform(.002,.005)
        pts=[(cx+(i/27-.5)*sx*3,cy+sy*math.sin(i/27*math.pi*4)+rng.uniform(-.0003,.0003)) for i in range(28)]
        stroke(pts,.00065,3)
    # Sparse marks beyond the primary right-hand work area.
    for _ in range(round(20*density)):
        line(rng.uniform(-.55,-.1),rng.uniform(-.24,.24),rng.uniform(.008,.029),rng.uniform(-.4,.4),.00055,4)
    mesh=bpy.data.meshes.new('Graphite_surface_ribbons')
    mesh.from_pydata(vertices,[],faces); mesh.update()
    obj=bpy.data.objects.new('Handling_Scuff_PencilStrokes',mesh); col.objects.link(obj); obj.parent=parent
    for m in mats: mesh.materials.append(m)
    for face,index in zip(mesh.polygons,mat_ids): face.material_index=index
    obj['role']='Render-only graphite marks; no physical collision or rigid body'
    parent['seed']=seed; parent['density']=density; parent['contrast']=contrast
    parent['stroke_count']=strokes
    # Pencil visibility follows the same existing wear randomization.
    text=bpy.data.texts.get('randomize_video_lab.py')
    if text:
        updated=text.as_string().replace("bpy.data.collections['Surface_Wear'].objects", "list(bpy.data.collections['Surface_Wear'].all_objects)")
        text.clear(); text.write(updated)
    scene['pencil_marks_collection']='Pencil_Marks'
    return parent,obj,strokes


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT/'assets/scenes/video_lab_weathered')
    ap.add_argument('--output',type=Path,default=ROOT/'assets/scenes/video_lab_pencil')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--density',type=float,default=1.)
    ap.add_argument('--contrast',type=float,default=1.)
    ap.add_argument('--render',action='store_true')
    ap.add_argument('--samples',type=int,default=64)
    a=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    src=a.source.resolve(); out=a.output.resolve()
    if src==out or src in out.parents or out in src.parents: raise ValueError('Use a separate output folder.')
    if not .1<=a.density<=3 or not 0<=a.contrast<=1: raise ValueError('density: 0.1..3; contrast: 0..1')
    originals={str(p):hashes(p) for p in [ROOT/'assets/scenes/video_lab',src]}
    out.mkdir(parents=True,exist_ok=True)
    for name in ['props','textures']: shutil.copytree(src/name,out/name,dirs_exist_ok=True)
    for name in ['environment.usdc','lighting.usdc','cameras.usdc','preview.usda','manifest.json','scene_config.json']:
        shutil.copy2(src/name,out/name)
    bpy.ops.wm.open_mainfile(filepath=str(src/'video_lab.blend'))
    parent,obj,count=make_pencil_marks(a.seed,a.density,a.contrast)
    export_selection(out/'pencil_marks.usdc',[parent,obj],'/PencilMarks')
    stage=Usd.Stage.Open(str(out/'environment.usdc'))
    stage.DefinePrim('/Lab/Handling_Scuff_PencilMarks').GetReferences().AddReference('./pencil_marks.usdc')
    stage.GetRootLayer().Save()
    scene=bpy.context.scene
    for im in bpy.data.images:
        if im.source=='FILE' and im.filepath:
            target=out/'textures'/Path(im.filepath).name
            if target.exists(): im.filepath=str(target)
            im.pack()
    scene.camera=bpy.data.objects['Camera_Overview']
    bpy.context.preferences.filepaths.save_version=0
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'video_lab.blend'))
    config=dict(seed=a.seed,density=a.density,contrast=a.contrast,stroke_count=count,collection='Pencil_Marks',usd_root='/Lab/Handling_Scuff_PencilMarks')
    (out/'pencil_config.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest=json.loads((out/'manifest.json').read_text()); manifest['pencil_marks']=config
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert not UsdUtils.ComputeAllDependencies(str(out/'preview.usda'))[2]
    assert all(hashes(Path(p))==h for p,h in originals.items())
    collision=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    old=Usd.Stage.Open(str(src/'environment.usdc'))
    assert collision==[str(p.GetPath()) for p in old.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert all(not p.HasAPI(UsdPhysics.CollisionAPI) and not p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(stage.GetPrimAtPath('/Lab/Handling_Scuff_PencilMarks')))
    (out/'fallback_hashes.json').write_text(json.dumps(originals,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps(dict(status='passed',earlier_packages_unchanged=True,collision_paths_unchanged=True,pencil_marks_render_only=True,usd_dependencies_resolved=True,stroke_count=count),indent=2)+'\n')
    if a.render:
        scene.cycles.samples=a.samples
        for camera,name in [('Camera_Overview','preview_overview'),('Camera_Table_Detail','preview_table_detail')]:
            scene.camera=bpy.data.objects[camera]; scene.render.filepath=str(out/(name+'.png')); bpy.ops.render.render(write_still=True)
        obj.hide_render=True
        scene.render.filepath=str(out/'comparison_before_detail.png'); bpy.ops.render.render(write_still=True)
    print('PENCIL_LAB_OK',count,'strokes',out)

if __name__=='__main__': main()
