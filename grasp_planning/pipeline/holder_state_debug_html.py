"""Interactive state/candidate debugger for holder feasibility results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grasp_planning.grasping.collision import GRIPPER_COLLISION_MODEL_PDZ, normalize_gripper_collision_model_name
from grasp_planning.grasping.fabrica_grasp_debug import (
    KUKA_Y_GRIPPER_TCP_TO_GRASP_CENTER_M,
    _load_kuka_y_gripper_visual_mesh_tcp,
    quat_to_rotmat_xyzw,
    rotmat_to_quat_xyzw,
)

from .assembly_sequence import AssemblySequence
from .assembly_sequence_debug_html import assembly_sequence_visual_payload
from .holder_grasp_debug_html import _pdz_component_payload
from .holder_state_feasibility import HolderStateFeasibilityResult


def _component_payload(name: str) -> dict[str, object]:
    vertices, faces = _load_kuka_y_gripper_visual_mesh_tcp(name)
    return {
        "vertices": np.round(vertices, 6).tolist(),
        "faces": np.asarray(faces, dtype=np.int64).tolist(),
    }


def _gripper_payload(gripper_collision_model: str = "kuka_y_gripper") -> dict[str, object]:
    if normalize_gripper_collision_model_name(gripper_collision_model) == GRIPPER_COLLISION_MODEL_PDZ:
        return {
            "model": GRIPPER_COLLISION_MODEL_PDZ,
            "tcp_to_grasp_center_m": [0.0, 0.0, 0.0],
            # Candidate poses are expressed directly at the calibrated TCP.
            "robot_tcp_from_grasp_center_m": [0.0, 0.0, 0.0],
            "base": _pdz_component_payload("base"),
            "left_finger": _pdz_component_payload("left_finger"),
            "right_finger": _pdz_component_payload("right_finger"),
        }
    left_vertices, _ = _load_kuka_y_gripper_visual_mesh_tcp("left_finger")
    right_vertices, _ = _load_kuka_y_gripper_visual_mesh_tcp("right_finger")
    left_tip = left_vertices[left_vertices[:, 2] >= 0.08]
    right_tip = right_vertices[right_vertices[:, 2] >= 0.08]
    return {
        "model": "kuka_y_gripper",
        "tcp_to_grasp_center_m": np.asarray(
            KUKA_Y_GRIPPER_TCP_TO_GRASP_CENTER_M,
            dtype=float,
        ).tolist(),
        "base": _component_payload("base"),
        "left_finger": _component_payload("left_finger"),
        "right_finger": _component_payload("right_finger"),
        "left_fingertip_inner_y": float(np.max(left_tip[:, 1])),
        "right_fingertip_inner_y": float(np.min(right_tip[:, 1])),
    }


def _candidate_payloads(result: HolderStateFeasibilityResult) -> list[dict[str, object]]:
    rotation_assembly_from_source = result.source_frame_pose_assembly.rotation_world_from_object
    translation_assembly_from_source = result.source_frame_pose_assembly.translation_world
    payloads = []
    for rank, candidate in enumerate(result.candidates, start=1):
        rotation_assembly = rotation_assembly_from_source @ quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj)

        def point_assembly(point_source: tuple[float, float, float]) -> list[float]:
            point = (
                rotation_assembly_from_source @ np.asarray(point_source, dtype=float) + translation_assembly_from_source
            )
            return np.round(point, 8).tolist()

        payloads.append(
            {
                "rank": rank,
                "grasp_id": candidate.grasp_id,
                "position": point_assembly(candidate.grasp_position_obj),
                "orientation_xyzw": list(rotmat_to_quat_xyzw(rotation_assembly)),
                "contact_a": point_assembly(candidate.contact_point_a_obj),
                "contact_b": point_assembly(candidate.contact_point_b_obj),
                "normal_a": np.round(
                    rotation_assembly_from_source @ np.asarray(candidate.contact_normal_a_obj, dtype=float),
                    8,
                ).tolist(),
                "normal_b": np.round(
                    rotation_assembly_from_source @ np.asarray(candidate.contact_normal_b_obj, dtype=float),
                    8,
                ).tolist(),
                "jaw_width": candidate.jaw_width,
                "roll_angle_rad": candidate.roll_angle_rad,
                "contact_patch_lateral_offset_m": candidate.contact_patch_lateral_offset_m,
                "contact_patch_approach_offset_m": candidate.contact_patch_approach_offset_m,
                "score": candidate.score,
                "score_components": candidate.score_components or {},
            }
        )
    return payloads


def _debug_payload(
    result: HolderStateFeasibilityResult,
    sequence: AssemblySequence,
    *,
    initial_step_index: int,
) -> dict[str, object]:
    sequence_payload = assembly_sequence_visual_payload(
        sequence,
        max_edges_per_part=250,
        max_faces_per_part=1200,
    )
    return {
        "assembly": result.assembly,
        "base_part_id": result.base_part_id,
        "base_part_source": result.base_part_source,
        "selected_order": list(result.selected_order),
        "initial_step_index": int(initial_step_index),
        "configuration": result.to_payload()["configuration"],
        "table": result.to_payload()["table"],
        "sequence": sequence_payload,
        "candidates": _candidate_payloads(result),
        "states": [
            {
                "step_id": state.step_id,
                "step_index": state.step_index,
                "incoming_part_id": state.incoming_part_id,
                "holder_base_available": state.holder_base_available,
                "assembled_part_ids_before": list(state.assembled_part_ids_before),
                "static_obstacle_part_ids": list(state.static_obstacle_part_ids),
                "incoming_final_to_pre_translation_m": list(state.incoming_final_to_pre_translation_m),
                "reason_counts": state.reason_counts,
                "results": {
                    candidate_result.grasp_id: candidate_result.to_payload()
                    for candidate_result in state.candidate_results
                },
            }
            for state in result.states
        ],
        "gripper": _gripper_payload(),
    }


def write_holder_state_debug_html(
    result: HolderStateFeasibilityResult,
    sequence: AssemblySequence,
    output_path: str | Path,
    *,
    initial_step_index: int,
) -> None:
    data_json = json.dumps(
        _debug_payload(result, sequence, initial_step_index=initial_step_index),
        separators=(",", ":"),
    ).replace("</", "<\\/")
    html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Holder State Feasibility</title>
<style>
:root{--bg:#f2eee5;--panel:#fffaf1;--ink:#24231f;--muted:#706a5f;--line:#d8cdb9;
--base:#218267;--assembled:#69746f;--incoming:#da7a20;--accepted:#198754;--rejected:#c43e32;
--table:#2d79bd;--gripper:#9a6a24;--pregrasp:#3b82b6;--na:#db4a37;--nb:#16866a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Segoe UI",sans-serif}
.layout{display:grid;grid-template-columns:410px minmax(0,1fr);min-height:100vh}
aside{padding:18px;background:var(--panel);border-right:1px solid var(--line);overflow:auto}
main{padding:16px;min-width:0}.scene-grid{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:12px}
h1{font-size:25px;margin:0 0 6px}h2{font-size:13px;text-transform:uppercase;letter-spacing:.07em;margin:16px 0 8px}
.subtitle,.count{font-size:12px;color:var(--muted);line-height:1.45}.controls{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button,input,select{border:1px solid var(--line);border-radius:9px;background:#fff;padding:8px 9px;font:inherit;width:100%}
button{cursor:pointer}.wide{grid-column:1/-1}.range{display:flex;gap:8px;align-items:center;margin:7px 0}
.range label{min-width:68px;color:var(--muted);font-size:12px}.range input{padding:0}
#candidateList{display:grid;gap:6px;margin-top:8px}.candidate{text-align:left;display:grid;grid-template-columns:68px 1fr auto;gap:8px}
.candidate.active{border-color:var(--rejected);background:#fff1e9}.id{font:700 12px ui-monospace,monospace}
.meta{font:11px ui-monospace,monospace;color:var(--muted)}.ok{color:var(--accepted)}.bad{color:var(--rejected)}
canvas{width:100%;display:block;border-radius:11px;background:linear-gradient(#fff,#ebe5d9);cursor:grab;touch-action:none}
#scene{aspect-ratio:1.3/1}#matrix{height:260px;image-rendering:pixelated;cursor:crosshair}
pre{white-space:pre-wrap;margin:0;font:11px/1.45 ui-monospace,"SFMono-Regular",monospace}
.reasons{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.pill{font:11px ui-monospace,monospace;padding:5px 7px;border-radius:999px;background:#eee7da}
.legend{display:flex;flex-wrap:wrap;gap:10px;color:var(--muted);font-size:11px;margin-top:8px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}
@media(max-width:1150px){.layout{grid-template-columns:1fr}.scene-grid{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body><div class="layout"><aside>
<h1>Holder State Feasibility</h1><div id="subtitle" class="subtitle"></div>
<h2>State</h2><div class="range"><label>Step</label><input id="step" type="range" min="0" step="1"><output id="stepOut"></output></div>
<div class="range"><label>Insertion</label><input id="progress" type="range" min="0" max="1" step=".01" value="0"><output id="progressOut">0%</output></div>
<div id="reasons" class="reasons"></div>
<h2>Candidate</h2><div class="controls">
<input id="search" class="wide" placeholder="Find holder ID">
<select id="reason" class="wide"><option value="">All reasons</option></select>
<button id="prev">Previous</button><button id="next">Next</button>
</div><div id="count" class="count"></div><div id="candidateList"></div>
</aside><main><div class="scene-grid"><section class="card">
<canvas id="scene" width="1200" height="920"></canvas>
<div class="legend"><span><i class="dot" style="background:var(--base)"></i>base</span>
<span><i class="dot" style="background:var(--assembled)"></i>assembled</span>
<span><i class="dot" style="background:var(--incoming)"></i>incoming/sweep</span>
<span><i class="dot" style="background:var(--gripper)"></i>holder</span>
<span><i class="dot" style="background:var(--pregrasp)"></i>pregrasp path</span>
<span><i class="dot" style="background:var(--rejected)"></i>failing geometry</span></div>
</section><section class="card"><h2>Selected result</h2><pre id="details"></pre></section></div>
<section class="card" style="margin-top:14px"><h2>Candidate × step validity matrix</h2>
<canvas id="matrix" width="1200" height="500"></canvas>
<div class="subtitle">Columns are assembly steps; rows follow candidate score rank. Click to select a cell.</div></section>
</main></div><script>
const data=__DATA__,S=data.sequence.visualization,$=id=>document.getElementById(id);
const canvas=$("scene"),ctx=canvas.getContext("2d"),matrix=$("matrix"),mctx=matrix.getContext("2d");
const state={step:data.initial_step_index,progress:0,selected:0,search:"",reason:"",yaw:-.72,pitch:.52,zoom:1,panX:0,panY:0,drag:false,lastX:0,lastY:0};
$("subtitle").textContent=`${data.assembly}: ${data.selected_order.join(" → ")} · base ${data.base_part_id} (${data.base_part_source})`;
$("step").max=String(data.states.length-1);$("step").value=String(state.step);
const reasons=[...new Set(data.states.flatMap(s=>Object.values(s.results).map(r=>r.reason)))].sort();
reasons.forEach(r=>{const o=document.createElement("option");o.value=r;o.textContent=r;$("reason").appendChild(o)});
function preferAccepted(){const st=data.states[state.step];state.reason=(st.reason_counts.accepted||0)>0?"accepted":"";$("reason").value=state.reason}
preferAccepted();
function add(a,b){return a.map((v,i)=>v+b[i])}function sub(a,b){return a.map((v,i)=>v-b[i])}function mul(a,s){return a.map(v=>v*s)}
function qrot(v,q){const[x,y,z,w]=q,tx=2*(y*v[2]-z*v[1]),ty=2*(z*v[0]-x*v[2]),tz=2*(x*v[1]-y*v[0]);return[v[0]+w*tx+y*tz-z*ty,v[1]+w*ty+z*tx-x*tz,v[2]+w*tz+x*ty-y*tx]}
const bounds=S.scene_bounds_assembly_m,center=bounds.center,extent=Math.max(bounds.extent,.2);
function camera(v){const p=sub(v,center),cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);
const x=cy*p[0]-sy*p[1],y=sy*p[0]+cy*p[1],z=p[2];return[x,cp*y-sp*z,sp*y+cp*z]}
function project(v){const p=camera(v),s=.72*Math.min(canvas.width,canvas.height)/extent*state.zoom;return[canvas.width/2+state.panX+p[0]*s,canvas.height*.52+state.panY-p[2]*s,p[1]]}
function line(a,b,color,w=2,dash=[]){const p=project(a),q=project(b);ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.strokeStyle=color;ctx.lineWidth=w;ctx.stroke();ctx.setLineDash([])}
function point(v,color,r=5){const p=project(v);ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle="#fff";ctx.stroke()}
function polygon(points,fill,stroke,alpha=.3){const ps=points.map(project);ctx.beginPath();ctx.moveTo(ps[0][0],ps[0][1]);ps.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));ctx.closePath();ctx.globalAlpha=alpha;ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.stroke();ctx.globalAlpha=1}
function drawPart(id,color,alpha=1,offset=[0,0,0]){const m=S.parts[id],verts=m.vertices_assembly_m.map(p=>add(p,offset));
const records=m.faces.map(f=>{const ps=f.map(i=>project(verts[i]));return{ps,d:ps.reduce((s,p)=>s+p[2],0)/3}}).sort((a,b)=>a.d-b.d);
ctx.globalAlpha=.34*alpha;ctx.fillStyle=color;ctx.strokeStyle=color;ctx.lineWidth=.35;records.forEach(r=>{ctx.beginPath();ctx.moveTo(r.ps[0][0],r.ps[0][1]);ctx.lineTo(r.ps[1][0],r.ps[1][1]);ctx.lineTo(r.ps[2][0],r.ps[2][1]);ctx.closePath();ctx.fill();ctx.stroke()});ctx.globalAlpha=1;
m.edges.forEach(e=>line(verts[e[0]],verts[e[1]],color,.7*alpha))}
function faceRecords(vertices,faces,fill){return faces.map(f=>{const ps=f.map(i=>project(vertices[i]));return{ps,d:ps.reduce((s,p)=>s+p[2],0)/ps.length,fill}})}
function drawFaces(records,alpha=.68){records.sort((a,b)=>a.d-b.d);ctx.globalAlpha=alpha;records.forEach(r=>{ctx.beginPath();ctx.moveTo(r.ps[0][0],r.ps[0][1]);r.ps.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));ctx.closePath();ctx.fillStyle=r.fill;ctx.fill();ctx.strokeStyle="#3c302955";ctx.lineWidth=.35;ctx.stroke()});ctx.globalAlpha=1}
function componentWorld(comp,c,shift=0,translation=[0,0,0]){const patch=qrot([c.contact_patch_lateral_offset_m||0,0,c.contact_patch_approach_offset_m||0],c.orientation_xyzw),origin=add(sub(sub(c.position,patch),qrot(data.gripper.tcp_to_grasp_center_m,c.orientation_xyzw)),translation);
return comp.vertices.map(v=>add(origin,qrot([v[0],v[1]+shift,v[2]],c.orientation_xyzw)))}
function drawGripper(c,color,translation=[0,0,0],alpha=.68){const h=c.jaw_width/2,items=data.gripper.model==="pdz_gripper"?[[data.gripper.base,0],[data.gripper.left_finger,-Math.max(0,(c.jaw_width-.012)/2)],[data.gripper.right_finger,Math.max(0,(c.jaw_width-.012)/2)]]:[[data.gripper.base,0],[data.gripper.left_finger,-h-data.gripper.left_fingertip_inner_y],[data.gripper.right_finger,h-data.gripper.right_fingertip_inner_y]],records=[];
items.forEach(([comp,shift])=>records.push(...faceRecords(componentWorld(comp,c,shift,translation),comp.faces,color)));drawFaces(records,alpha)}
function currentState(){return data.states[state.step]}function filtered(){const st=currentState(),q=state.search.toLowerCase();return data.candidates.filter(c=>{const r=st.results[c.grasp_id];return(!q||c.grasp_id.toLowerCase().includes(q))&&(!state.reason||r.reason===state.reason)})}
function selected(){const a=filtered();if(!a.length)return null;state.selected=Math.min(state.selected,a.length-1);return a[state.selected]}
function resultFor(c){return c?currentState().results[c.grasp_id]:null}
function failureColor(r){return r&&r.status==="accepted"?"#198754":"#c43e32"}
function renderScene(){ctx.clearRect(0,0,canvas.width,canvas.height);const st=currentState(),c=selected(),r=resultFor(c);
polygon(S.table_vertices_assembly_m,r&&["table_collision","clearance_margin_failed"].includes(r.reason)&&r.details.obstacle_type?.startsWith("table")?"#c43e32":"#2d79bd","#2d79bd",.15);
for(const id of st.assembled_part_ids_before){const fail=r&&((r.details.obstacle_part_ids||[]).includes(id)||r.details.obstacle_type==="base_part"&&id===data.base_part_id);drawPart(id,fail?"#c43e32":id===data.base_part_id?"#218267":"#69746f",1)}
const step=data.sequence.steps[state.step],pre=st.incoming_final_to_pre_translation_m,offset=mul(pre,1-state.progress);
drawPart(st.incoming_part_id,r&&r.details.obstacle_type==="incoming_part_sweep"?"#c43e32":"#da7a20",1,offset);drawPart(st.incoming_part_id,"#da7a20",.18,pre);drawPart(st.incoming_part_id,"#da7a20",.18,[0,0,0]);
if(!c||!st.holder_base_available)return;const color=failureColor(r),z=qrot([0,0,1],c.orientation_xyzw),pregrasp=mul(z,-data.configuration.pregrasp_offset_m);
drawGripper(c,color);drawGripper(c,"#3b82b6",pregrasp,.18);line(add(c.position,pregrasp),c.position,"#3b82b6",2,[7,5]);
point(c.contact_a,"#db4a37",5);point(c.contact_b,"#16866a",5);point(c.position,color,4)}
function fmt(v){return v===null||v===undefined?"n/a":typeof v==="number"?v.toFixed(6):JSON.stringify(v)}
function renderDetails(){const st=currentState(),c=selected(),r=resultFor(c);$("stepOut").textContent=`${state.step+1}/${data.states.length}`;$("progressOut").textContent=`${Math.round(state.progress*100)}%`;
$("reasons").replaceChildren(...Object.entries(st.reason_counts).map(([k,v])=>{const s=document.createElement("span");s.className="pill";s.textContent=`${k}: ${v}`;return s}));
if(!c){$("details").textContent="No candidate matches the filters.";return}$("details").textContent=[
`step:              ${st.step_id}`,`incoming_part:     ${st.incoming_part_id}`,`assembled_before:  ${JSON.stringify(st.assembled_part_ids_before)}`,
`holder_available:  ${st.holder_base_available}`,`grasp_id:          ${c.grasp_id}`,`rank:              ${c.rank}/${data.candidates.length}`,
`status:            ${r.status}`,`reason:            ${r.reason}`,`score:             ${fmt(c.score)}`,`jaw_width_m:       ${fmt(c.jaw_width)}`,
`min_clearance_m:   ${fmt(r.minimum_clearance_m)}`,`details:           ${JSON.stringify(r.details,null,2)}`].join("\\n")}
function renderList(){const a=filtered(),sel=selected();$("count").textContent=`${a.length}/${data.candidates.length} candidates (first 160 listed)`;
$("candidateList").replaceChildren(...a.slice(0,160).map((c,i)=>{const r=resultFor(c),b=document.createElement("button");b.className=`candidate${c===sel?" active":""}`;
b.innerHTML=`<span><span class="id">${c.grasp_id}</span><br><span class="meta">rank ${c.rank}</span></span><span class="meta">${r.reason}</span><span class="${r.status==="accepted"?"ok":"bad"}">${c.score===null?"n/a":c.score.toFixed(3)}</span>`;b.onclick=()=>{state.selected=i;renderAll()};return b}))}
const reasonColors={accepted:"#198754",base_not_available:"#aaa39a",table_collision:"#2563aa",base_collision:"#8c5b24",assembled_part_collision:"#7c3ea3",holder_pregrasp_collision:"#d56a1f",holder_approach_sweep_collision:"#d19b20",incoming_part_sweep_collision:"#c43e32",clearance_margin_failed:"#d64f86"};
function renderMatrix(){const w=matrix.width,h=matrix.height,cols=data.states.length,rows=data.candidates.length,cw=w/cols,rh=h/rows;mctx.clearRect(0,0,w,h);
data.candidates.forEach((c,y)=>data.states.forEach((st,x)=>{mctx.fillStyle=reasonColors[st.results[c.grasp_id].reason]||"#555";mctx.fillRect(x*cw,y*rh,Math.ceil(cw),Math.max(1,Math.ceil(rh)))}));
mctx.strokeStyle="#fff";mctx.lineWidth=3;mctx.strokeRect(state.step*cw,0,cw,h);const c=selected();if(c){mctx.strokeStyle="#111";mctx.lineWidth=1.5;mctx.strokeRect(0,(c.rank-1)*rh,w,Math.max(2,rh))}}
function renderAll(){renderList();renderDetails();renderScene();renderMatrix()}
function reset(){state.selected=0;renderAll()}$("step").oninput=e=>{state.step=Number(e.target.value);state.progress=0;$("progress").value="0";preferAccepted();reset()};
$("progress").oninput=e=>{state.progress=Number(e.target.value);renderScene();renderDetails()};$("search").oninput=e=>{state.search=e.target.value;reset()};$("reason").onchange=e=>{state.reason=e.target.value;reset()};
$("prev").onclick=()=>{const n=filtered().length;if(n){state.selected=(state.selected-1+n)%n;renderAll()}};$("next").onclick=()=>{const n=filtered().length;if(n){state.selected=(state.selected+1)%n;renderAll()}};
matrix.onclick=e=>{const rect=matrix.getBoundingClientRect(),x=(e.clientX-rect.left)/rect.width,y=(e.clientY-rect.top)/rect.height;state.step=Math.min(data.states.length-1,Math.floor(x*data.states.length));$("step").value=String(state.step);
const rank=Math.min(data.candidates.length-1,Math.floor(y*data.candidates.length));state.search=data.candidates[rank].grasp_id;$("search").value=state.search;state.reason="";$("reason").value="";reset()};
canvas.onpointerdown=e=>{state.drag=true;state.lastX=e.clientX;state.lastY=e.clientY;canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!state.drag)return;state.yaw+=(e.clientX-state.lastX)*.01;state.pitch-=(e.clientY-state.lastY)*.01;state.lastX=e.clientX;state.lastY=e.clientY;renderScene()};
canvas.onpointerup=canvas.onpointercancel=()=>state.drag=false;canvas.onwheel=e=>{e.preventDefault();state.zoom=Math.max(.3,Math.min(4,state.zoom*(e.deltaY<0?1.08:1/1.08)));renderScene()};canvas.oncontextmenu=e=>e.preventDefault();
renderAll();
</script></body></html>""".replace("__DATA__", data_json)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def write_holder_state_debug_artifacts(
    result: HolderStateFeasibilityResult,
    sequence: AssemblySequence,
    output_dir: str | Path,
) -> tuple[Path, tuple[Path, ...]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    initial_step = sequence.first_holder_step_index or 0
    matrix_path = output / "holder_validity_matrix.html"
    write_holder_state_debug_html(
        result,
        sequence,
        matrix_path,
        initial_step_index=initial_step,
    )
    state_paths = []
    for state in result.states:
        path = output / f"holder_state_{state.step_id}.html"
        write_holder_state_debug_html(
            result,
            sequence,
            path,
            initial_step_index=state.step_index,
        )
        state_paths.append(path)
    return matrix_path, tuple(state_paths)


__all__ = [
    "write_holder_state_debug_artifacts",
    "write_holder_state_debug_html",
]
