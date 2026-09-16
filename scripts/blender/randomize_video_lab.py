"""Seeded, non-accumulating Blender scene randomization.

Blender UI: load this text, then call randomize(seed=17), or use command below.
blender -b assets/scenes/video_lab/video_lab.blend --python scripts/blender/randomize_video_lab.py -- --seed 17 --output /tmp/lab17.blend --render /tmp/lab17.png
"""
import argparse
import colorsys
import json
import math
from pathlib import Path
import random
import sys
import bpy
from mathutils import Vector


def sample_layout(rng, cfg, specs, width, depth):
    count=rng.randint(*cfg['prop_count']); chosen=rng.sample(specs,min(count,len(specs)))
    reserved=cfg['reserved_robot_mounts_xy_radius']+[cfg['reserved_task_center_xy_radius']]
    placed=[]
    for name,radius in chosen:
        for _ in range(300):
            x=rng.uniform(-width/2+radius+.025,width/2-radius-.025)
            y=rng.uniform(-depth/2+radius+.025,depth/2-radius-.025)
            if any(math.hypot(x-a,y-b)<radius+r+.02 for a,b,r in reserved+[(p['x'],p['y'],p['radius']) for p in placed]): continue
            placed.append(dict(name=name,x=x,y=y,yaw=rng.uniform(-math.pi,math.pi),radius=radius)); break
    return placed


def randomize(seed=0, include_props=True):
    scene=bpy.context.scene; scene['randomization_seed']=seed; cfg=json.loads(scene['randomization_config']); rng=random.Random(seed)
    def u(key): return rng.uniform(*cfg[key])
    table=bpy.data.materials['Table_Laminate']; nt=table.node_tree; p=nt.nodes.get('Principled BSDF')
    gain=u('table_color_gain'); nt.nodes['Table_Tint'].inputs[2].default_value=(gain,gain,gain,1)
    for link in list(p.inputs['Roughness'].links): nt.links.remove(link)
    rough=u('table_roughness'); p.inputs['Roughness'].default_value=rough
    wall=bpy.data.materials['Warm_White_Partition']; wp=wall.node_tree.nodes.get('Principled BSDF')
    if 'nominal_linear' not in wall: wall['nominal_linear']=list(wp.inputs['Base Color'].default_value)
    wg=u('wall_color_gain'); wp.inputs['Base Color'].default_value=(*[min(1,c*wg) for c in wall['nominal_linear'][:3]],1)
    kelvin=u('light_temperature_kelvin'); temp=(kelvin-5300)/2300
    light_gain=u('light_power_gain')
    for o in bpy.data.collections['Lighting'].objects:
        o.data.energy=o['nominal_energy']*light_gain
        o.data.color=(min(1,1-.12*temp),.97,min(1,.94+.15*temp))
    scene.view_settings.exposure=u('exposure_ev')
    for o in bpy.data.collections['Cameras'].objects:
        o.location=Vector(o['nominal_location'])+Vector([u('camera_position_delta_m') for _ in range(3)])
        o.rotation_euler=o['nominal_rotation']; o.data.lens=u('camera_focal_length_mm')
    wear=rng.random()<cfg['wear_visibility_probability']
    for o in bpy.data.collections['Surface_Wear'].objects:
        if 'Grommet' not in o.name and 'grommet' not in o.name: o.hide_render=not wear; o.hide_viewport=not wear
    col=bpy.data.collections['Optional_Props']; roots=[o for o in col.objects if not o.parent]
    specs=[(o.name,o['footprint_radius_m']) for o in roots]
    dims=bpy.data.objects['Table']['estimated_dimensions_m']
    layout=sample_layout(rng,cfg,specs,dims[0],dims[1]) if include_props else []
    for o in col.objects: o.hide_render=True; o.hide_viewport=True
    for pose in layout:
        root=bpy.data.objects[pose['name']]; root.location=(pose['x'],pose['y'],.0004); root.rotation_euler=(0,0,pose['yaw'])
        for o in [root]+list(root.children_recursive): o.hide_render=False; o.hide_viewport=False
    for name in ['Muted_Blue_FDM','Brown_FDM','Lime_Green_Plastic','Ochre_Yellow_Tape','Red_Pencil_Lacquer']:
        m=bpy.data.materials[name]; p=m.node_tree.nodes.get('Principled BSDF')
        if 'nominal_linear' not in m: m['nominal_linear']=list(p.inputs['Base Color'].default_value)
        h,s,v=colorsys.rgb_to_hsv(*m['nominal_linear'][:3]); rgb=colorsys.hsv_to_rgb((h+u('prop_hue_delta'))%1,s,min(1,v*u('prop_color_gain')))
        p.inputs['Base Color'].default_value=(*rgb,1); p.inputs['Roughness'].default_value=u('prop_roughness')
    result=dict(seed=seed,table_color_gain=gain,table_roughness=rough,light_gain=light_gain,temperature_kelvin=kelvin,wear_visible=wear,props=layout)
    scene['last_randomization']=json.dumps(result); return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=bpy.context.scene.get('randomization_seed',0)); ap.add_argument('--output',type=Path); ap.add_argument('--render',type=Path); ap.add_argument('--no-props',action='store_true'); ap.add_argument('--samples',type=int,default=32)
    a=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    result=randomize(a.seed,not a.no_props); print(json.dumps(result,indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(a.output.resolve()))
        a.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    if a.render:
        a.render.parent.mkdir(parents=True,exist_ok=True)
        bpy.context.scene.cycles.samples=a.samples; bpy.context.scene.render.filepath=str(a.render.resolve()); bpy.ops.render.render(write_still=True)

if __name__=='__main__': main()
