"""Compact interactive HTML for large KUKA holder-grasp libraries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grasp_planning.grasping.collision import (
    _PDZ_GRIPPER_BASE_TO_GRASP_CENTER_M,
    _PDZ_GRIPPER_BODY_ROTATION_TCP,
    GRIPPER_COLLISION_MODEL_PDZ,
    _load_pdz_gripper_collision_hull,
    normalize_gripper_collision_model_name,
)
from grasp_planning.grasping.fabrica_grasp_debug import (
    KUKA_Y_GRIPPER_TCP_TO_GRASP_CENTER_M,
    SavedGraspCandidate,
    _load_kuka_y_gripper_visual_mesh_tcp,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh


def _rounded(values: np.ndarray, digits: int = 6) -> list[list[float]]:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def _component_payload(name: str) -> dict[str, object]:
    vertices, faces = _load_kuka_y_gripper_visual_mesh_tcp(name)
    return {
        "vertices": _rounded(vertices),
        "faces": np.asarray(faces, dtype=np.int64).tolist(),
    }


def _pdz_component_payload(name: str) -> dict[str, object]:
    """Express a PDZ URDF component in the planner TCP frame for the HTML."""

    vertices, faces = _load_pdz_gripper_collision_hull(name)
    vertices_tcp = vertices @ _PDZ_GRIPPER_BODY_ROTATION_TCP.T - (
        _PDZ_GRIPPER_BODY_ROTATION_TCP @ _PDZ_GRIPPER_BASE_TO_GRASP_CENTER_M
    )[None, :]
    return {"vertices": _rounded(vertices_tcp), "faces": np.asarray(faces, dtype=np.int64).tolist()}


def _candidate_payload(candidate: SavedGraspCandidate, rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "grasp_id": candidate.grasp_id,
        "position": list(candidate.grasp_position_obj),
        "orientation_xyzw": list(candidate.grasp_orientation_xyzw_obj),
        "contact_a": list(candidate.contact_point_a_obj),
        "contact_b": list(candidate.contact_point_b_obj),
        "normal_a": list(candidate.contact_normal_a_obj),
        "normal_b": list(candidate.contact_normal_b_obj),
        "jaw_width": candidate.jaw_width,
        "roll_angle_rad": candidate.roll_angle_rad,
        "contact_patch_lateral_offset_m": candidate.contact_patch_lateral_offset_m,
        "contact_patch_approach_offset_m": candidate.contact_patch_approach_offset_m,
        "score": candidate.score,
        "score_components": candidate.score_components or {},
        "metadata": candidate.metadata or {},
    }


def write_holder_grasp_debug_html(
    *,
    title: str,
    subtitle: str,
    mesh_local: TriangleMesh,
    candidates: tuple[SavedGraspCandidate, ...],
    output_html: str | Path,
    metadata_lines: list[str],
    table_plane_local: list[list[float]] | None = None,
    gripper_collision_model: str = "kuka_y_gripper",
) -> None:
    """Write one shared KUKA mesh plus lightweight data for every candidate."""

    is_pdz = normalize_gripper_collision_model_name(gripper_collision_model) == GRIPPER_COLLISION_MODEL_PDZ
    if is_pdz:
        gripper_payload = {
            "model": GRIPPER_COLLISION_MODEL_PDZ,
            "tcp_to_grasp_center_m": [0.0, 0.0, 0.0],
            "robot_tcp_from_grasp_center_m": [0.0, 0.0, 0.0],
            "base": _pdz_component_payload("base"),
            "left_finger": _pdz_component_payload("left_finger"),
            "right_finger": _pdz_component_payload("right_finger"),
        }
    else:
        left_vertices, _ = _load_kuka_y_gripper_visual_mesh_tcp("left_finger")
        right_vertices, _ = _load_kuka_y_gripper_visual_mesh_tcp("right_finger")
        left_tip = left_vertices[left_vertices[:, 2] >= 0.08]
        right_tip = right_vertices[right_vertices[:, 2] >= 0.08]
        if not len(left_tip):
            left_tip = left_vertices
        if not len(right_tip):
            right_tip = right_vertices
        gripper_payload = {
            "model": "kuka_y_gripper",
            "tcp_to_grasp_center_m": np.asarray(KUKA_Y_GRIPPER_TCP_TO_GRASP_CENTER_M, dtype=float).tolist(),
            "base": _component_payload("base"),
            "left_finger": _component_payload("left_finger"),
            "right_finger": _component_payload("right_finger"),
            "left_fingertip_inner_y": float(np.max(left_tip[:, 1])),
            "right_fingertip_inner_y": float(np.min(right_tip[:, 1])),
        }

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -(float(candidate.score) if candidate.score is not None else float("-inf")),
            candidate.grasp_id,
        ),
    )
    data = {
        "title": title,
        "subtitle": subtitle,
        "mesh": {
            "vertices": _rounded(mesh_local.vertices_obj),
            "faces": np.asarray(mesh_local.faces, dtype=np.int64).tolist(),
        },
        "gripper": gripper_payload,
        "metadata_lines": metadata_lines,
        "table_plane_local": table_plane_local,
        "candidates": [
            _candidate_payload(candidate, rank) for rank, candidate in enumerate(sorted_candidates, start=1)
        ],
    }
    data_json = json.dumps(data, separators=(",", ":"))
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Holder Grasp Library</title>
  <style>
    :root {
      --paper: #f2ede1; --panel: #fffaf0; --ink: #22211d; --muted: #716b60;
      --line: #d8cdb8; --accent: #b4462e; --green: #28705b; --blue: #2777a8;
      --orange: #d48224; --base: #846020; --contact-a: #d04732; --contact-b: #178166;
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); font-family: Inter, "Segoe UI", sans-serif;
      background: radial-gradient(circle at 10% 0%, #fff8e9, transparent 30%), var(--paper); }
    .layout { min-height: 100vh; display: grid; grid-template-columns: 390px minmax(0, 1fr); }
    aside { padding: 20px; border-right: 1px solid var(--line); background: rgba(255,250,240,.94); overflow: auto; }
    main { padding: 18px; min-width: 0; }
    h1 { margin: 0 0 8px; font-size: 27px; line-height: 1.1; }
    .subtitle { margin: 0 0 16px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }
    input, button { width: 100%; border: 1px solid var(--line); border-radius: 10px; background: white;
      padding: 9px 10px; color: var(--ink); font: inherit; }
    button { cursor: pointer; }
    button:hover, input:focus { border-color: var(--accent); outline: none; }
    .wide { grid-column: 1 / -1; }
    .filter-label { color: var(--muted); font-size: 12px; display: flex; justify-content: space-between; }
    #resultCount { color: var(--muted); font: 12px ui-monospace, monospace; margin: 8px 0; }
    #candidateList { display: grid; gap: 7px; }
    .candidate { text-align: left; display: grid; grid-template-columns: 70px 1fr auto; gap: 8px; align-items: center; }
    .candidate.active { border-color: var(--accent); background: #fff3ea; box-shadow: 0 5px 14px #7d3c1b1c; }
    .id { font: 700 13px ui-monospace, monospace; }
    .rank { color: var(--muted); font: 11px ui-monospace, monospace; }
    .score { color: var(--green); font: 12px ui-monospace, monospace; }
    .card { background: rgba(255,250,240,.92); border: 1px solid var(--line); border-radius: 18px;
      padding: 14px; box-shadow: 0 12px 30px #57432612; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(340px, .55fr); gap: 16px; }
    canvas { width: 100%; aspect-ratio: 1.3 / 1; display: block; border-radius: 13px;
      background: radial-gradient(circle at 25% 15%, #fff, #ece5d6); cursor: grab; touch-action: none; }
    canvas.dragging { cursor: grabbing; }
    h2 { margin: 2px 0 10px; font-size: 14px; text-transform: uppercase; letter-spacing: .08em; }
    pre { white-space: pre-wrap; margin: 0; font: 12px/1.48 ui-monospace, "SFMono-Regular", monospace; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; color: var(--muted); font-size: 12px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }
    .hint { color: var(--muted); font-size: 12px; margin: 9px 0 0; }
    @media (max-width: 1050px) {
      .layout { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<div class="layout">
  <aside>
    <h1 id="title"></h1>
    <p id="subtitle" class="subtitle"></p>
    <div class="controls">
      <input id="search" class="wide" placeholder="Find holder ID, e.g. h1094">
      <button id="prev">Previous</button><button id="next">Next</button>
      <label class="wide filter-label"><span>Minimum score</span><span id="scoreValue"></span></label>
      <input id="scoreMin" class="wide" type="range" min="0" max="1" step="0.001" value="0">
      <button id="overlay">All candidates: On</button><button id="meshMode">Mesh: Solid</button>
    </div>
    <div id="resultCount"></div>
    <div id="candidateList"></div>
  </aside>
  <main>
    <div class="grid">
      <section class="card">
        <h2>Base source frame</h2>
        <canvas id="scene" width="1100" height="840"></canvas>
        <div class="legend">
          <span><i class="dot" style="background:var(--green)"></i>base mesh</span>
          <span><i class="dot" style="background:var(--base)"></i>KUKA collision hulls</span>
          <span><i class="dot" style="background:var(--contact-a)"></i>contact A</span>
          <span><i class="dot" style="background:var(--contact-b)"></i>contact B</span>
          <span><i class="dot" style="background:var(--blue)"></i>approach / pregrasp</span>
        </div>
        <p class="hint">Left drag rotates, middle/Shift-drag pans, wheel zooms, arrow keys change grasp.</p>
      </section>
      <section class="card"><h2>Selection</h2><pre id="details"></pre></section>
    </div>
  </main>
</div>
<script>
const data=__DATA__;
const $=id=>document.getElementById(id);
const canvas=$("scene"),ctx=canvas.getContext("2d");
const state={selected:0,yaw:-.8,pitch:.48,zoom:1,panX:0,panY:0,drag:false,mode:"rotate",
  lastX:0,lastY:0,overlay:true,solid:true,search:"",minScore:0};
$("title").textContent=data.title;$("subtitle").textContent=data.subtitle;

function add(a,b){return a.map((v,i)=>v+b[i])}
function sub(a,b){return a.map((v,i)=>v-b[i])}
function mul(a,s){return a.map(v=>v*s)}
function qrot(v,q){const [x,y,z,w]=q;const tx=2*(y*v[2]-z*v[1]),ty=2*(z*v[0]-x*v[2]),tz=2*(x*v[1]-y*v[0]);
  return [v[0]+w*tx+(y*tz-z*ty),v[1]+w*ty+(z*tx-x*tz),v[2]+w*tz+(x*ty-y*tx)]}
function camera(v){const cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);
  const x=cy*v[0]+sy*v[1],y=-sy*v[0]+cy*v[1],z=v[2];return [x,cp*y+sp*z,-sp*y+cp*z]}
const objectPoints=[...data.mesh.vertices,...(data.table_plane_local||[]),...data.candidates.flatMap(c=>[c.position,c.contact_a,c.contact_b])];
const lo=[0,1,2].map(i=>Math.min(...objectPoints.map(p=>p[i]))),hi=[0,1,2].map(i=>Math.max(...objectPoints.map(p=>p[i])));
const center=lo.map((v,i)=>(v+hi[i])/2);const extent=Math.max(...lo.map((v,i)=>hi[i]-v),.18);
function project(v){const p=camera(sub(v,center)),s=560/extent*state.zoom;return [canvas.width/2+state.panX+p[0]*s,canvas.height/2+state.panY-p[1]*s,p[2]]}
function filtered(){const query=state.search.trim().toLowerCase();return data.candidates.filter(c=>
  (c.score===null||c.score>=state.minScore)&&(!query||c.grasp_id.toLowerCase().includes(query)))}
function selectedCandidate(){const list=filtered();if(!list.length)return null;state.selected=Math.min(state.selected,list.length-1);return list[state.selected]}
function line(a,b,color,width=2,dash=[]){const p=project(a),q=project(b);ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);
  ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();ctx.setLineDash([])}
function point(v,color,r=5){const p=project(v);ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle="#fff";ctx.lineWidth=1.4;ctx.stroke()}
function arrow(a,b,color,width=2){line(a,b,color,width);const p=project(a),q=project(b),ang=Math.atan2(q[1]-p[1],q[0]-p[0]);
  ctx.beginPath();ctx.moveTo(q[0],q[1]);ctx.lineTo(q[0]-10*Math.cos(ang-.45),q[1]-10*Math.sin(ang-.45));
  ctx.lineTo(q[0]-10*Math.cos(ang+.45),q[1]-10*Math.sin(ang+.45));ctx.closePath();ctx.fillStyle=color;ctx.fill()}
function faceRecords(vertices,faces,fill){return faces.map(face=>{const pts=face.map(i=>project(vertices[i]));return{pts,depth:pts.reduce((s,p)=>s+p[2],0)/pts.length,fill}})}
function drawFaces(records){records.sort((a,b)=>a.depth-b.depth).forEach(r=>{ctx.beginPath();ctx.moveTo(r.pts[0][0],r.pts[0][1]);
  r.pts.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));ctx.closePath();ctx.fillStyle=r.fill;ctx.fill();ctx.strokeStyle="#40372b55";ctx.lineWidth=.45;ctx.stroke()})}
function edges(faces){const seen=new Set(),out=[];faces.forEach(f=>[[f[0],f[1]],[f[1],f[2]],[f[2],f[0]]].forEach(([a,b])=>{const k=a<b?`${a}:${b}`:`${b}:${a}`;if(!seen.has(k)){seen.add(k);out.push([a,b])}}));return out}
const targetEdges=edges(data.mesh.faces);
function componentWorld(component,c,shiftY=0){const patch=qrot([c.contact_patch_lateral_offset_m||0,0,c.contact_patch_approach_offset_m||0],c.orientation_xyzw),origin=sub(sub(c.position,patch),qrot(data.gripper.tcp_to_grasp_center_m,c.orientation_xyzw));
  return component.vertices.map(v=>add(origin,qrot([v[0],v[1]+shiftY,v[2]],c.orientation_xyzw)))}
function drawTarget(){if(state.solid)drawFaces(faceRecords(data.mesh.vertices,data.mesh.faces,"#5f8275cc"));
  targetEdges.forEach(e=>line(data.mesh.vertices[e[0]],data.mesh.vertices[e[1]],"#3f6256",state.solid?.55:1.4))}
function drawTable(){if(!data.table_plane_local)return;const pts=data.table_plane_local.map(project);ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);
  pts.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));ctx.closePath();ctx.fillStyle="#2777a822";ctx.fill();ctx.strokeStyle="#2777a899";ctx.lineWidth=1.4;ctx.stroke()}
function drawGripper(c){const half=c.jaw_width/2;
  const items=data.gripper.model==="pdz_gripper"?
    // The PDZ hand is deliberately a single convex collision hull.  Keep it
    // translucent so its conservative envelope cannot hide the pad contacts.
    [[data.gripper.base,0,"#82612638"],[data.gripper.left_finger,-Math.max(0,(c.jaw_width-.012)/2),"#d18a2bc8"],[data.gripper.right_finger,Math.max(0,(c.jaw_width-.012)/2),"#d18a2bc8"]]:[
    [data.gripper.base,0,"#826126b8"],
    [data.gripper.left_finger,-half-data.gripper.left_fingertip_inner_y,"#d18a2bc8"],
    [data.gripper.right_finger,half-data.gripper.right_fingertip_inner_y,"#d18a2bc8"]];
  const records=[];items.forEach(([component,shift,fill])=>records.push(...faceRecords(componentWorld(component,c,shift),component.faces,fill)));drawFaces(records)}
function drawCandidateOverlay(list,selected){if(!state.overlay)return;list.forEach(c=>{if(c===selected)return;const a=project(c.contact_a),b=project(c.contact_b);
  ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle="#b4462e24";ctx.lineWidth=.65;ctx.stroke()})}
function drawAxes(){const o=[0,0,0],len=.025;arrow(o,[len,0,0],"#c23b32",1.5);arrow(o,[0,len,0],"#2f8b56",1.5);arrow(o,[0,0,len],"#2877b4",1.5)}
function renderScene(){ctx.clearRect(0,0,canvas.width,canvas.height);const list=filtered(),c=selectedCandidate();drawTable();drawCandidateOverlay(list,c);drawTarget();drawAxes();if(!c)return;
  drawGripper(c);line(c.contact_a,c.contact_b,"#2d2b27",2.4);point(c.contact_a,"#d04732",6);point(c.contact_b,"#178166",6);point(c.position,"#b4462e",5);
  if(data.gripper.model==="pdz_gripper"){const tcp=add(c.position,qrot(data.gripper.robot_tcp_from_grasp_center_m,c.orientation_xyzw));line(c.position,tcp,"#175eb0",1.7,[3,3]);point(tcp,"#175eb0",4)}
  const z=qrot([0,0,1],c.orientation_xyzw),x=qrot([1,0,0],c.orientation_xyzw),y=qrot([0,1,0],c.orientation_xyzw);
  const pre=sub(c.position,mul(z,.05));line(pre,c.position,"#2777a8",2,[7,5]);point(pre,"#2777a8",4);arrow(c.position,add(c.position,mul(z,.035)),"#2777a8",2.2);
  arrow(c.position,add(c.position,mul(x,.025)),"#c23b32",1.5);arrow(c.position,add(c.position,mul(y,.025)),"#2f8b56",1.5);
  arrow(c.contact_a,add(c.contact_a,mul(c.normal_a,.018)),"#d04732",1.5);arrow(c.contact_b,add(c.contact_b,mul(c.normal_b,.018)),"#178166",1.5)}
function fmt(v){return v.map(n=>(n>=0?"+":"")+n.toFixed(6)).join(", ")}
function renderDetails(){const c=selectedCandidate();if(!c){$("details").textContent=[...data.metadata_lines,"","No candidates match the filter."].join("\\n");return}
  const scores=Object.entries(c.score_components).map(([k,v])=>`  ${k}: ${typeof v==="number"?v.toFixed(6):v}`);
  $("details").textContent=[...data.metadata_lines,"",`rank:              ${c.rank} / ${data.candidates.length}`,`grasp_id:          ${c.grasp_id}`,
    `score:             ${c.score===null?"n/a":c.score.toFixed(6)}`,`jaw_width_m:       ${c.jaw_width.toFixed(6)}`,
    `roll_angle_rad:    ${c.roll_angle_rad.toFixed(6)}`,data.gripper.model==="pdz_gripper"?`robot_tcp_offset_z:${data.gripper.robot_tcp_from_grasp_center_m[2].toFixed(6)}`:"",`contact_offset_x:  ${c.contact_patch_lateral_offset_m.toFixed(6)}`,
    `contact_offset_z:  ${c.contact_patch_approach_offset_m.toFixed(6)}`,`position:          ${fmt(c.position)}`,
    `contact_a:         ${fmt(c.contact_a)}`,`contact_b:         ${fmt(c.contact_b)}`,`normal_a:          ${fmt(c.normal_a)}`,
    `normal_b:          ${fmt(c.normal_b)}`,"score_components:",...scores].join("\\n")}
function renderList(){const list=filtered(),shown=list.slice(0,200);$("resultCount").textContent=`${list.length} / ${data.candidates.length} candidates${list.length>200?" (first 200 listed)":""}`;
  const selected=selectedCandidate();$("candidateList").replaceChildren(...shown.map((c,i)=>{const b=document.createElement("button");b.className=`candidate${c===selected?" active":""}`;
    b.innerHTML=`<span><span class="id">${c.grasp_id}</span><br><span class="rank">rank ${c.rank}</span></span><span class="rank">jaw ${c.jaw_width.toFixed(4)} m</span><span class="score">${c.score===null?"n/a":c.score.toFixed(3)}</span>`;
    b.onclick=()=>{state.selected=i;renderAll()};return b}))}
function renderAll(){renderList();renderDetails();renderScene()}
function resetSelection(){state.selected=0;renderAll()}
$("search").oninput=e=>{state.search=e.target.value;resetSelection()};
$("scoreMin").oninput=e=>{state.minScore=Number(e.target.value);$("scoreValue").textContent=state.minScore.toFixed(3);resetSelection()};
$("scoreValue").textContent=state.minScore.toFixed(3);
$("prev").onclick=()=>{const n=filtered().length;if(n){state.selected=(state.selected-1+n)%n;renderAll()}};
$("next").onclick=()=>{const n=filtered().length;if(n){state.selected=(state.selected+1)%n;renderAll()}};
$("overlay").onclick=()=>{state.overlay=!state.overlay;$("overlay").textContent=`All candidates: ${state.overlay?"On":"Off"}`;renderScene()};
$("meshMode").onclick=()=>{state.solid=!state.solid;$("meshMode").textContent=`Mesh: ${state.solid?"Solid":"Wire"}`;renderScene()};
window.onkeydown=e=>{if(e.key==="ArrowLeft")$("prev").click();if(e.key==="ArrowRight")$("next").click()};
canvas.onpointerdown=e=>{if(e.button!==0&&e.button!==1)return;state.drag=true;state.mode=e.button===1||e.shiftKey?"pan":"rotate";state.lastX=e.clientX;state.lastY=e.clientY;canvas.setPointerCapture(e.pointerId);canvas.classList.add("dragging")};
canvas.onpointermove=e=>{if(!state.drag)return;const dx=e.clientX-state.lastX,dy=e.clientY-state.lastY;state.lastX=e.clientX;state.lastY=e.clientY;
  if(state.mode==="pan"){state.panX+=dx;state.panY+=dy}else{state.yaw+=dx*.01;state.pitch-=dy*.01}renderScene()};
canvas.onpointerup=canvas.onpointercancel=()=>{state.drag=false;canvas.classList.remove("dragging")};
canvas.onwheel=e=>{e.preventDefault();state.zoom=Math.max(.3,Math.min(5,state.zoom*(e.deltaY<0?1.08:1/1.08)));renderScene()};
canvas.oncontextmenu=e=>e.preventDefault();
renderAll();
</script>
</body></html>
""".replace("__DATA__", data_json)
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
