"""Make a separate, reversible tabletop appearance study from the approved lab.

blender -b --python-exit-code 1 --python scripts/blender/weather_video_lab.py -- --render
Original assets are read-only to this script. New textures are authored in UV space.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import bpy
import numpy as np
from pxr import Usd, UsdShade, Sdf, UsdUtils

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from build_video_lab import texture_image


def hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def linear_rgb(image):
    w, h = image.size
    pixels = np.empty(w*h*4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    rgb = pixels.reshape(h,w,4)[:,:,:3].copy()
    if image.colorspace_settings.name == 'sRGB':
        rgb = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
    return rgb


def author_weathering(base, seed, amount):
    """Localized contact grime, warm aging, edge dirt and directional wipe traces.

    Masks use metric table coordinates. No geometry, shading or object silhouettes
    are baked into the material; visual dirt is independent of physics.
    """
    rng = np.random.default_rng(seed)
    h,w = base.shape[:2]
    y,x = np.mgrid[0:h,0:w].astype(np.float32)
    x = (x/(w-1)-.5)*1.6
    y = (y/(h-1)-.5)*.8
    # Band-limited irregularity, avoiding uniformly sprayed dirt or circular blobs.
    cloud = np.zeros((h,w),dtype=np.float32)
    for frequency in [7,15,31,67,139,283]:
        for _ in range(3):
            angle,phase=rng.uniform(0,2*np.pi,2)
            cloud += np.sin((x*np.cos(angle)+y*np.sin(angle))*frequency+phase)/np.sqrt(frequency)
    cloud=(cloud-cloud.min())/(cloud.max()-cloud.min())
    fleck=np.clip((cloud-.25)/.65,0,1)

    def patch(cx,cy,sx,sy,angle=0):
        dx=x-cx; dy=y-cy
        u=dx*np.cos(angle)+dy*np.sin(angle)
        v=-dx*np.sin(angle)+dy*np.cos(angle)
        return np.exp(-((u/sx)**2+(v/sy)**2)*1.7)*(.24+.76*fleck)

    # Warm old adhesive discoloration, slight cool-gray grime from handling.
    warm=np.zeros((h,w),dtype=np.float32)
    dirt=np.zeros_like(warm)
    for cx,cy,sx,sy,a,k in [(.34,.12,.23,.12,.2,.33),(-.46,.25,.18,.065,-.1,.24),(.53,-.19,.12,.05,.3,.18),(-.15,-.23,.20,.055,-.3,.12)]:
        warm+=k*patch(cx,cy,sx,sy,a)
    for cx,cy,sx,sy,a,k in [(.35,.14,.13,.075,.1,.26),(-.57,.24,.11,.06,.3,.22),(.60,.26,.10,.05,-.2,.20),(.19,-.28,.18,.025,-.15,.11),(-.37,-.02,.12,.028,.6,.11)]:
        dirt+=k*patch(cx,cy,sx,sy,a)
    # Finger/cloth drag marks, clustered around work rather than equally spaced.
    for _ in range(34):
        cx=rng.normal(.32,.13); cy=rng.normal(.12,.08)
        dirt+=rng.uniform(.025,.075)*patch(cx,cy,rng.uniform(.01,.055),rng.uniform(.0007,.0035),rng.uniform(-.3,.3))
    # Edge and center-joint dirt is low amplitude and discontinuous.
    edge=np.exp(-((y+.394)/.010)**2)+np.exp(-((y-.393)/.009)**2)
    seam=np.exp(-(x/.0028)**2)
    dirt+=(edge*.038+seam*.025)*fleck
    # Sparse tiny spots: not modeled debris and not baked object shadows.
    for _ in range(95):
        cx=rng.uniform(-.75,.75); cy=rng.uniform(-.36,.36)
        dirt+=rng.uniform(.025,.12)*patch(cx,cy,rng.uniform(.0005,.0016),rng.uniform(.0004,.0013))
    # Soft interrupted ring of handling/adhesive residue near the rear work area.
    r=np.sqrt(((x-.38)/1.05)**2+(y-.215)**2)
    ring=np.exp(-((r-.032)/.0013)**2)*np.clip((cloud-.35)*2,0,1)
    warm+=ring*.075
    dirt=np.clip(dirt,0,.50)*amount
    warm=np.clip(warm,0,.45)*amount
    # Keep overall laminate recognizable; add chromatic variation locally.
    aged=base*np.array([1.014,1.001,.974],dtype=np.float32)
    color=aged*(1-dirt[:,:,None])
    amber=np.array([.36,.265,.145],dtype=np.float32)
    color=color*(1-warm[:,:,None])+amber*warm[:,:,None]
    rough=np.clip(.38+dirt*.55+warm*.28+(cloud-.5)*.035,.28,.65)
    return np.clip(color,0,1),rough


def add_blend_control(nt, clean, dirty, destination, scene):
    mix=nt.nodes.new('ShaderNodeMixRGB')
    mix.name='Weathering_'+destination.identifier
    mix.label='Table dirt: Table_Weathering_Strength (0 = original)'
    nt.links.new(clean.outputs['Color'],mix.inputs[1])
    nt.links.new(dirty.outputs['Color'],mix.inputs[2])
    strength=nt.nodes.get('Table_Weathering_Strength')
    if strength is None:
        strength=nt.nodes.new('ShaderNodeValue')
        strength.name='Table_Weathering_Strength'
        strength.label='DIRT STRENGTH: 0 = original, 1 = weathered'
        strength.outputs[0].default_value=1.
    nt.links.new(strength.outputs[0],mix.inputs[0])
    nt.links.new(mix.outputs[0],destination)
    return mix


def layout_material(nt):
    positions = {
        'Table_Weathering_Strength': (-450, 850),
        'Original_Laminate_Albedo': (-450, 620),
        'Laminate_Albedo': (-450, 300),
        'Weathering_Color1': (0, 580),
        'Table_Tint': (250, 580),
        'Original_Laminate_Roughness': (-450, -80),
        'Laminate_Roughness': (-450, -390),
        'Weathering_Roughness': (0, -80),
        'Noise Texture': (0, -400),
        'Bump': (280, -200),
        'Principled BSDF': (530, 480),
        'Material Output': (870, 480),
    }
    for name, position in positions.items():
        nt.nodes[name].location = position
    nt.nodes['Table_Weathering_Strength'].use_custom_color = True
    nt.nodes['Table_Weathering_Strength'].color = (.32, .19, .06)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT/'assets/scenes/video_lab')
    ap.add_argument('--output',type=Path,default=ROOT/'assets/scenes/video_lab_weathered')
    ap.add_argument('--seed',type=int,default=194)
    ap.add_argument('--amount',type=float,default=1.3)
    ap.add_argument('--render',action='store_true')
    ap.add_argument('--samples',type=int,default=96)
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    src=args.source.resolve(); out=args.output.resolve()
    if src==out or src in out.parents or out in src.parents:
        raise ValueError('Source and output must be separate sibling directories.')
    if not 0<=args.amount<=2: raise ValueError('--amount must be between 0 and 2')
    before=hashes(src)
    out.mkdir(parents=True,exist_ok=True)
    # Copy supporting assets, never the original validation or its preview images.
    for name in ['props','textures']:
        shutil.copytree(src/name,out/name,dirs_exist_ok=True)
    for name in ['environment.usdc','lighting.usdc','cameras.usdc','preview.usda','manifest.json','scene_config.json']:
        shutil.copy2(src/name,out/name)
    bpy.ops.wm.open_mainfile(filepath=str(src/'video_lab.blend'))
    scene=bpy.context.scene
    mat=bpy.data.materials['Table_Laminate']; nt=mat.node_tree
    albedo=nt.nodes['Laminate_Albedo']; clean_image=albedo.image
    rough=nt.nodes['Laminate_Roughness']; clean_rough=rough.image
    color,roughness=author_weathering(linear_rgb(clean_image),args.seed,args.amount)
    rgba=np.ones((*color.shape[:2],4),dtype=np.float32); rgba[:,:,:3]=color
    dirty_image=texture_image('laminate_weathered_albedo',rgba,out/'textures')
    rgba[:,:,:3]=roughness[:,:,None]
    rough_image=texture_image('laminate_weathered_roughness',rgba,out/'textures',data=True)
    albedo.image=dirty_image; rough.image=rough_image
    original=nt.nodes.new('ShaderNodeTexImage'); original.name='Original_Laminate_Albedo'; original.image=clean_image
    original_rough=nt.nodes.new('ShaderNodeTexImage'); original_rough.name='Original_Laminate_Roughness'; original_rough.image=clean_rough
    scene['table_weathering_seed']=args.seed
    scene['approved_fallback_blend']=str(src/'video_lab.blend')
    add_blend_control(nt,original,albedo,nt.nodes['Table_Tint'].inputs[1],scene)
    add_blend_control(nt,original_rough,rough,nt.nodes['Principled BSDF'].inputs['Roughness'],scene)
    # Keep other material images packed; move path hints into this package too.
    for im in bpy.data.images:
        if im.source=='FILE' and im.filepath:
            target=out/'textures'/Path(im.filepath).name
            if target.exists(): im.filepath=str(target)
            im.pack()
    layout_material(nt)
    bpy.context.preferences.filepaths.save_version=0
    scene.camera=bpy.data.objects['Camera_Overview']
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'video_lab.blend'))
    # USD geometry and physics stay byte-for-byte semantically identical: only
    # the two tabletop texture asset paths are changed in the copied stage.
    stage=Usd.Stage.Open(str(out/'environment.usdc'))
    for node,filename in [('Laminate_Albedo','laminate_weathered_albedo.png'),('Laminate_Roughness','laminate_weathered_roughness.png')]:
        shader=UsdShade.Shader(stage.GetPrimAtPath('/Lab/_materials/Table_Laminate/'+node))
        shader.GetInput('file').Set(Sdf.AssetPath('./textures/'+filename))
    stage.GetRootLayer().Save()
    manifest=json.loads((out/'manifest.json').read_text())
    manifest['appearance_variant']=dict(name='warm_localized_table_weathering_v1',seed=args.seed,amount=args.amount,fallback='../video_lab/video_lab.blend',blender_strength_node='Table_Weathering_Strength')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    _,_,unresolved=UsdUtils.ComputeAllDependencies(str(out/'preview.usda'))
    assert not unresolved,unresolved
    assert before==hashes(src),'Original package changed!'
    (out/'fallback_hashes.json').write_text(json.dumps(before,indent=2)+'\n')
    scene.cycles.samples=args.samples
    if args.render:
        for camera,name in [('Camera_Overview','preview_overview'),('Camera_Table_Detail','preview_table_detail')]:
            scene.camera=bpy.data.objects[camera]
            scene.render.filepath=str(out/(name+'.png')); bpy.ops.render.render(write_still=True)
        # Same camera/lights/props make this a controlled visual comparison.
        nt.nodes['Table_Weathering_Strength'].outputs[0].default_value=0.; nt.update_tag(); bpy.context.view_layer.update()
        scene.render.filepath=str(out/'comparison_original_detail.png'); bpy.ops.render.render(write_still=True)
    print('WEATHERED_LAB_OK',out)

if __name__=='__main__': main()
