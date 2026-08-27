#!/usr/bin/env python3
"""Render a scalable, reshufflable 3D selector for expert trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    load_asset_mesh,
    load_grasp_bundle,
)
from grasp_planning.grasping.grasp_transforms import saved_grasp_to_world_grasp  # noqa: E402
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.pipeline.fabrica_pipeline import _mesh_in_source_frame  # noqa: E402
from scripts.visualize_visual_servo_dataset_3d import _episode_geometry  # noqa: E402


def _rounded(values: np.ndarray) -> list:
    return np.round(np.asarray(values, dtype=np.float64), 6).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/pipeline_stage2_ground_feasible.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/visual_servo_dataset_selector.html"),
    )
    parser.add_argument("--pregrasp-offset", type=float, default=0.10)
    parser.add_argument("--initial-percent", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.initial_percent <= 100:
        parser.error("--initial-percent must lie between 1 and 100.")

    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise SystemExit("Plotly is required to render the selector.") from exc

    bundle = load_grasp_bundle(args.bundle)
    pose_raw = bundle.metadata.get("execution_world_pose")
    if not isinstance(pose_raw, dict):
        raise ValueError(f"{args.bundle} has no execution_world_pose metadata.")
    nominal_object_pose = ObjectWorldPose(
        position_world=tuple(float(value) for value in pose_raw["position_world"]),
        orientation_xyzw_world=tuple(
            float(value) for value in pose_raw["orientation_xyzw_world"]
        ),
    )
    metadata_paths = sorted(args.dataset_dir.glob("episode_*.json"))
    if not metadata_paths:
        raise ValueError(f"No episode metadata found under {args.dataset_dir}.")
    first_metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    grasp_id = str(first_metadata["grasp_id"])
    saved_grasp = next(
        (candidate for candidate in bundle.candidates if candidate.grasp_id == grasp_id),
        None,
    )
    if saved_grasp is None:
        raise ValueError(f"Grasp {grasp_id!r} is absent from {args.bundle}.")
    world_grasp = saved_grasp_to_world_grasp(
        saved_grasp,
        nominal_object_pose,
        pregrasp_offset=float(args.pregrasp_offset),
        gripper_width_clearance=0.01,
    )
    nominal_pregrasp = np.asarray(world_grasp.pregrasp_position_w)
    nominal_grasp = np.asarray(world_grasp.position_w)
    mesh_world = load_asset_mesh(bundle.target_mesh_path, scale=bundle.mesh_scale)
    source_pose = ObjectWorldPose(
        position_world=bundle.source_frame_origin_obj_world,
        orientation_xyzw_world=bundle.source_frame_orientation_xyzw_obj_world,
    )
    mesh_local = _mesh_in_source_frame(mesh_world, source_pose)
    vertices = np.asarray(mesh_local.vertices_obj, dtype=np.float64)
    faces = np.asarray(mesh_local.faces, dtype=np.int64)

    episodes = []
    for item_index, metadata_path in enumerate(metadata_paths, start=1):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        geometry = _episode_geometry(
            npz_path=metadata_path.with_suffix(".npz"),
            metadata=metadata,
            nominal_object_pose=nominal_object_pose,
            nominal_pregrasp=nominal_pregrasp,
            nominal_grasp=nominal_grasp,
        )
        stress = metadata.get("initial_ee_stress", {})
        episodes.append(
            {
                "episode": geometry["episode"],
                "file": str(metadata_path.with_suffix(".npz")),
                "path": _rounded(geometry["path"]),
                "initial_rotation": _rounded(geometry["initial_rotation"]),
                "final_rotation": _rounded(geometry["final_rotation"]),
                "final_position_error_mm": round(
                    geometry["final_position_error_m"] * 1000.0, 4
                ),
                "final_rotation_error_deg": round(
                    geometry["final_rotation_error_deg"], 5
                ),
                "offset_grasp_mm": _rounded(
                    np.asarray(stress.get("offset_grasp_m", (0.0, 0.0, 0.0)))
                    * 1000.0
                ),
                "rotation_xyz_deg": _rounded(
                    stress.get("rotation_xyz_deg", (0.0, 0.0, 0.0))
                ),
            }
        )
        if item_index % 256 == 0 or item_index == len(metadata_paths):
            print(
                f"[SELECTOR] Loaded {item_index}/{len(metadata_paths)} episodes.",
                flush=True,
            )

    payload = {
        "episodes": episodes,
        "mesh": {
            "vertices": _rounded(vertices),
            "faces": faces.tolist(),
        },
        "pregrasp": _rounded(
            nominal_object_pose.rotation_world_from_object.T
            @ (nominal_pregrasp - nominal_object_pose.translation_world)
        ),
        "grasp": _rounded(
            nominal_object_pose.rotation_world_from_object.T
            @ (nominal_grasp - nominal_object_pose.translation_world)
        ),
        "initial_percent": args.initial_percent,
    }
    template = """<!doctype html>
<html><head><meta charset="utf-8"><title>Visual-servo expert paths</title>
<script>__PLOTLY__</script>
<style>
body{margin:0;font:14px system-ui;background:#f7f8fa;color:#18202a}
#controls{display:flex;align-items:center;gap:14px;padding:10px 16px;background:white;border-bottom:1px solid #ddd}
#percentage{width:min(520px,45vw)} button{padding:7px 13px;cursor:pointer}
#plot{height:calc(100vh - 55px)} .value{font-variant-numeric:tabular-nums;font-weight:650}
</style></head><body>
<div id="controls">
  <label>Visible <input id="percentage" type="range" min="1" max="100" step="1"></label>
  <span class="value"><span id="percentValue"></span>% · <span id="countValue"></span> / <span id="totalValue"></span></span>
  <button id="shuffle">Reshuffle selection</button>
  <label><input id="axes" type="checkbox"> Show initial/final orientation axes</label>
  <span id="shuffleValue"></span>
</div><div id="plot"></div>
<script>
const data=__DATA__, episodes=data.episodes, total=episodes.length;
let order=Array.from({length:total},(_,i)=>i), shuffleNumber=0;
const slider=document.getElementById("percentage"), axesBox=document.getElementById("axes");
slider.value=data.initial_percent; document.getElementById("totalValue").textContent=total;
function shuffled(a){for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function lineSegments(selected,key,axis,length){
  const x=[],y=[],z=[];
  selected.forEach(e=>{const p=key==="initial_rotation"?e.path[0]:e.path[e.path.length-1], r=e[key];
    x.push(p[0],p[0]+r[0][axis]*length,null); y.push(p[1],p[1]+r[1][axis]*length,null); z.push(p[2],p[2]+r[2][axis]*length,null);});
  return {x,y,z};
}
function redraw(){
  const percent=Number(slider.value), count=Math.max(1,Math.round(total*percent/100)), selected=order.slice(0,count).map(i=>episodes[i]);
  const px=[],py=[],pz=[], ix=[],iy=[],iz=[], fx=[],fy=[],fz=[], ih=[],fh=[];
  selected.forEach(e=>{
    e.path.forEach(p=>{px.push(p[0]);py.push(p[1]);pz.push(p[2]);}); px.push(null);py.push(null);pz.push(null);
    const a=e.path[0],b=e.path[e.path.length-1]; ix.push(a[0]);iy.push(a[1]);iz.push(a[2]);fx.push(b[0]);fy.push(b[1]);fz.push(b[2]);
    ih.push(`episode ${e.episode}<br>${e.file}<br>initial xyz: ${a.map(v=>v.toFixed(4)).join(", ")} m<br>offset: ${e.offset_grasp_mm.join(", ")} mm<br>rotation: ${e.rotation_xyz_deg.join(", ")} deg`);
    fh.push(`episode ${e.episode}<br>${e.file}<br>final xyz: ${b.map(v=>v.toFixed(4)).join(", ")} m<br>error: ${e.final_position_error_mm} mm, ${e.final_rotation_error_deg} deg`);
  });
  const v=data.mesh.vertices,f=data.mesh.faces, traces=[
    {type:"mesh3d",x:v.map(p=>p[0]),y:v.map(p=>p[1]),z:v.map(p=>p[2]),i:f.map(q=>q[0]),j:f.map(q=>q[1]),k:f.map(q=>q[2]),color:"#a16207",opacity:.45,name:"part",hoverinfo:"skip"},
    {type:"scatter3d",mode:"lines+markers",x:[data.pregrasp[0],data.grasp[0]],y:[data.pregrasp[1],data.grasp[1]],z:[data.pregrasp[2],data.grasp[2]],line:{color:"#111827",width:5,dash:"dash"},marker:{size:3},name:"target path"},
    {type:"scatter3d",mode:"lines",x:px,y:py,z:pz,line:{color:"#2563eb",width:2},opacity:.24,name:"actual paths",hoverinfo:"skip"},
    {type:"scatter3d",mode:"markers",x:ix,y:iy,z:iz,marker:{size:4,color:"#f59e0b"},text:ih,hovertemplate:"%{text}<extra></extra>",name:"initial pose"},
    {type:"scatter3d",mode:"markers",x:fx,y:fy,z:fz,marker:{size:4,color:"#7c3aed",symbol:"square"},text:fh,hovertemplate:"%{text}<extra></extra>",name:"final pose"}
  ];
  if(axesBox.checked){["#ef4444","#22c55e","#3b82f6"].forEach((color,axis)=>{
    [["initial_rotation","initial axes"],["final_rotation","final axes"]].forEach(([key,name])=>{const s=lineSegments(selected,key,axis,.012);traces.push({type:"scatter3d",mode:"lines",...s,line:{color,width:3},name:`${name} ${"xyz"[axis]}`,showlegend:false,hoverinfo:"skip"});});
  });}
  Plotly.react("plot",traces,{title:`Expert paths · selection ${shuffleNumber} · ${count} of ${total}`,template:"plotly_white",scene:{xaxis:{title:"part X [m]"},yaxis:{title:"part Y [m]"},zaxis:{title:"part Z [m]"},aspectmode:"data",camera:{eye:{x:1.4,y:-1.5,z:1.1}}},margin:{l:0,r:0,t:55,b:0},legend:{x:.82,y:.98}},{responsive:true});
  document.getElementById("percentValue").textContent=percent; document.getElementById("countValue").textContent=count;
  document.getElementById("shuffleValue").textContent=`selection ${shuffleNumber}`;
}
slider.addEventListener("input",redraw); axesBox.addEventListener("change",redraw);
document.getElementById("shuffle").addEventListener("click",()=>{order=shuffled(order.slice());shuffleNumber++;redraw();});
redraw();
</script></body></html>"""
    html = template.replace("__PLOTLY__", get_plotlyjs()).replace(
        "__DATA__", json.dumps(payload, separators=(",", ":"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(
        f"Wrote {len(episodes)} selectable episodes to {args.output} "
        f"({args.output.stat().st_size / 1.0e6:.1f} MB)."
    )


if __name__ == "__main__":
    main()
