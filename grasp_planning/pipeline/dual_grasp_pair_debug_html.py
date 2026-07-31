"""Interactive compatibility-matrix debugger for Stage-3 grasp pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspCandidate,
    quat_to_rotmat_xyzw,
    rotmat_to_quat_xyzw,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose

from .assembly_sequence import AssemblySequence
from .assembly_sequence_debug_html import assembly_sequence_visual_payload
from .dual_grasp_pair_planner import (
    DualGraspPairPlanningResult,
    DualGraspPairStepResult,
    pair_html_name,
)
from .holder_state_debug_html import _gripper_payload


def _candidate_world_payload(
    candidate: SavedGraspCandidate,
    *,
    source_pose_assembly: ObjectWorldPose,
) -> dict[str, object]:
    rotation_source = source_pose_assembly.rotation_world_from_object
    translation_source = source_pose_assembly.translation_world
    rotation_assembly = rotation_source @ quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj)

    def point(source_point: tuple[float, float, float]) -> list[float]:
        return np.round(
            rotation_source @ np.asarray(source_point, dtype=float) + translation_source,
            8,
        ).tolist()

    return {
        "grasp_id": candidate.grasp_id,
        "position": point(candidate.grasp_position_obj),
        "orientation_xyzw": list(rotmat_to_quat_xyzw(rotation_assembly)),
        "contact_a": point(candidate.contact_point_a_obj),
        "contact_b": point(candidate.contact_point_b_obj),
        "jaw_width": candidate.jaw_width,
        "score": candidate.score,
    }


def _step_debug_payload(
    result: DualGraspPairPlanningResult,
    step_result: DualGraspPairStepResult,
    sequence: AssemblySequence,
) -> dict[str, object]:
    holder_by_id = {candidate.grasp_id: candidate for candidate in result.holder_feasibility.candidates}
    library = result.inserter_libraries_by_step[step_result.step_id]
    inserter_by_id = {status.grasp_id: status.candidate for status in library.candidate_statuses}
    holder_references = {candidate.grasp_id: candidate.to_payload() for candidate in step_result.holder_candidates}
    inserter_references = {candidate.grasp_id: candidate.to_payload() for candidate in step_result.inserter_candidates}
    return {
        "assembly": result.assembly,
        "base_part_id": result.base_part_id,
        "selected_order": list(result.selected_order),
        "configuration": result.config.to_payload(),
        "step": {
            "step_id": step_result.step_id,
            "step_index": step_result.step_index,
            "incoming_part_id": step_result.incoming_part_id,
            "assembled_part_ids_before": list(step_result.assembled_part_ids_before),
            "final_to_pre_translation_m": list(step_result.final_to_pre_translation_assembly_m),
            "retreat_translation_m": list(step_result.retreat_translation_assembly_m),
            "metadata": step_result.metadata,
            "reason_counts": step_result.reason_counts,
        },
        "sequence": assembly_sequence_visual_payload(
            sequence,
            max_edges_per_part=250,
            max_faces_per_part=1200,
        ),
        "holder_ids": list(step_result.matrix_holder_ids),
        "inserter_ids": list(step_result.matrix_inserter_ids),
        "holders": {
            grasp_id: {
                **_candidate_world_payload(
                    holder_by_id[grasp_id],
                    source_pose_assembly=(result.holder_feasibility.source_frame_pose_assembly),
                ),
                "unary": holder_references[grasp_id],
            }
            for grasp_id in step_result.matrix_holder_ids
        },
        "inserters": {
            grasp_id: {
                **_candidate_world_payload(
                    inserter_by_id[grasp_id],
                    source_pose_assembly=library.source_frame_pose_assembly,
                ),
                "unary": inserter_references[grasp_id],
            }
            for grasp_id in step_result.matrix_inserter_ids
        },
        "evaluations": {
            f"{evaluation.holder_grasp_id}|{evaluation.inserter_grasp_id}": (evaluation.to_payload())
            for evaluation in step_result.evaluations
        },
        "retained_pair_ids": list(step_result.retained_pair_ids),
        "detailed_rejected_pair_ids": list(step_result.detailed_rejected_pair_ids),
        "gripper": _gripper_payload(),
    }


def write_dual_grasp_pair_step_html(
    result: DualGraspPairPlanningResult,
    step_result: DualGraspPairStepResult,
    sequence: AssemblySequence,
    output_path: str | Path,
) -> None:
    data_json = json.dumps(
        _step_debug_payload(result, step_result, sequence),
        separators=(",", ":"),
    ).replace("</", "<\\/")
    html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dual-Grasp Pair Compatibility</title>
<style>
:root{--bg:#eeeae1;--panel:#fffaf1;--ink:#24231f;--muted:#716b61;--line:#d8cdb9;
--base:#23836b;--assembled:#68736f;--incoming:#dc7b20;--holder:#9a6722;--inserter:#7052b5;
--accepted:#178650;--rejected:#ca4036;--unchecked:#326eaa;--unary:#aaa49b;--selected:#f1b728}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Segoe UI",sans-serif}
.layout{display:grid;grid-template-columns:390px minmax(0,1fr);min-height:100vh}aside{padding:18px;background:var(--panel);border-right:1px solid var(--line);overflow:auto}
main{padding:15px;min-width:0}.grid{display:grid;grid-template-columns:minmax(0,1fr) 355px;gap:14px}.card{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:12px}
h1{font-size:25px;margin:0 0 6px}h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;margin:16px 0 8px}.subtitle,.count{font-size:12px;color:var(--muted);line-height:1.45}
select,input,button{width:100%;border:1px solid var(--line);border-radius:9px;background:white;padding:8px;font:inherit}.controls{display:grid;grid-template-columns:1fr 1fr;gap:7px}.wide{grid-column:1/-1}
.range{display:flex;align-items:center;gap:8px;margin:8px 0}.range label{font-size:12px;color:var(--muted);min-width:58px}.range input{padding:0}
#pairs{display:grid;gap:6px;margin-top:8px}.pair{display:grid;grid-template-columns:1fr auto;gap:8px;text-align:left}.pair.active{border-color:var(--selected);background:#fff4d0}.pid{font:700 11px ui-monospace,monospace}.meta{font:11px ui-monospace,monospace;color:var(--muted)}
canvas{width:100%;display:block;border-radius:11px;background:linear-gradient(#fff,#e8e2d8);touch-action:none}#scene{aspect-ratio:1.35/1;cursor:grab}#matrix{height:470px;image-rendering:pixelated;cursor:crosshair}
pre{white-space:pre-wrap;margin:0;font:11px/1.44 ui-monospace,"SFMono-Regular",monospace}.legend{display:flex;gap:10px;flex-wrap:wrap;color:var(--muted);font-size:11px;margin-top:8px}.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
.pills{display:flex;flex-wrap:wrap;gap:5px}.pill{padding:5px 7px;border-radius:999px;background:#eee8dc;font:11px ui-monospace,monospace}
@media(max-width:1120px){.layout{grid-template-columns:1fr}.grid{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body><div class="layout"><aside>
<h1>Dual-Grasp Pairs</h1><div id="subtitle" class="subtitle"></div>
<h2>Motion</h2><select id="phase"><option value="insertion">Pre-insertion → final</option><option value="retreat">Final → retreat</option></select>
<div class="range"><label>Progress</label><input id="progress" type="range" min="0" max="1" step=".01" value="0"><output id="progressOut">0%</output></div>
<h2>Pair selection</h2><div class="controls"><select id="filter" class="wide"><option value="retained">Retained pairs</option><option value="accepted">All checked compatible</option><option value="rejected">Checked collisions</option><option value="all">All detailed</option></select>
<button id="prev">Previous</button><button id="next">Next</button></div><div id="count" class="count"></div><div id="pairs"></div>
<h2>Counts</h2><div id="pills" class="pills"></div>
</aside><main><div class="grid"><section class="card"><canvas id="scene" width="1200" height="880"></canvas>
<div class="legend"><span><i class="dot" style="background:var(--base)"></i>base</span><span><i class="dot" style="background:var(--assembled)"></i>assembled</span>
<span><i class="dot" style="background:var(--incoming)"></i>incoming</span><span><i class="dot" style="background:var(--holder)"></i>holder</span>
<span><i class="dot" style="background:var(--inserter)"></i>inserter</span><span><i class="dot" style="background:var(--rejected)"></i>collision</span></div></section>
<section class="card"><h2>Selected cell</h2><pre id="details"></pre></section></div>
<section class="card" style="margin-top:14px"><h2>Holder × inserter compatibility matrix</h2><canvas id="matrix" width="1400" height="700"></canvas>
<div class="legend"><span><i class="dot" style="background:var(--accepted)"></i>compatible</span><span><i class="dot" style="background:var(--rejected)"></i>exact collision/margin</span>
<span><i class="dot" style="background:var(--unary)"></i>unary rejected</span><span><i class="dot" style="background:var(--unchecked)"></i>not checked by limit</span>
<span><i class="dot" style="background:var(--selected)"></i>retained/selected</span></div></section></main></div>
<script>
const data=__DATA__,S=data.sequence.visualization,$=id=>document.getElementById(id),canvas=$("scene"),ctx=canvas.getContext("2d"),matrix=$("matrix"),mctx=matrix.getContext("2d");
const retained=new Set(data.retained_pair_ids),state={row:0,col:0,phase:"insertion",progress:0,filter:"retained",pairIndex:0,yaw:-.72,pitch:.5,zoom:1,panX:0,panY:0,drag:false,lastX:0,lastY:0};
$("subtitle").textContent=`${data.assembly} · step ${data.step.step_index} · incoming ${data.step.incoming_part_id} · base ${data.base_part_id}`;
function add(a,b){return a.map((v,i)=>v+b[i])}function sub(a,b){return a.map((v,i)=>v-b[i])}function mul(a,s){return a.map(v=>v*s)}
function qrot(v,q){const[x,y,z,w]=q,tx=2*(y*v[2]-z*v[1]),ty=2*(z*v[0]-x*v[2]),tz=2*(x*v[1]-y*v[0]);return[v[0]+w*tx+y*tz-z*ty,v[1]+w*ty+z*tx-x*tz,v[2]+w*tz+x*ty-y*tx]}
const bounds=S.scene_bounds_assembly_m,center=bounds.center,extent=Math.max(bounds.extent,.2);
function camera(v){const p=sub(v,center),cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch),x=cy*p[0]-sy*p[1],y=sy*p[0]+cy*p[1],z=p[2];return[x,cp*y-sp*z,sp*y+cp*z]}
function project(v){const p=camera(v),s=.72*Math.min(canvas.width,canvas.height)/extent*state.zoom;return[canvas.width/2+state.panX+p[0]*s,canvas.height*.52+state.panY-p[2]*s,p[1]]}
function line(a,b,color,w=2,dash=[]){const p=project(a),q=project(b);ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.strokeStyle=color;ctx.lineWidth=w;ctx.stroke();ctx.setLineDash([])}
function polygon(points,fill,alpha=.2){const ps=points.map(project);ctx.beginPath();ctx.moveTo(ps[0][0],ps[0][1]);ps.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));ctx.closePath();ctx.globalAlpha=alpha;ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=fill;ctx.stroke();ctx.globalAlpha=1}
function drawPart(id,color,alpha=1,offset=[0,0,0]){const m=S.parts[id],verts=m.vertices_assembly_m.map(p=>add(p,offset)),records=m.faces.map(f=>{const ps=f.map(i=>project(verts[i]));return{ps,d:ps.reduce((s,p)=>s+p[2],0)/3}}).sort((a,b)=>a.d-b.d);ctx.globalAlpha=.34*alpha;ctx.fillStyle=color;ctx.strokeStyle=color;ctx.lineWidth=.35;records.forEach(r=>{ctx.beginPath();ctx.moveTo(r.ps[0][0],r.ps[0][1]);ctx.lineTo(r.ps[1][0],r.ps[1][1]);ctx.lineTo(r.ps[2][0],r.ps[2][1]);ctx.closePath();ctx.fill();ctx.stroke()});ctx.globalAlpha=1;m.edges.forEach(e=>line(verts[e[0]],verts[e[1]],color,.65*alpha))}
function faceRecords(vertices,faces,fill){return faces.map(f=>{const ps=f.map(i=>project(vertices[i]));return{ps,d:ps.reduce((s,p)=>s+p[2],0)/ps.length,fill}})}
function drawFaces(records,alpha=.64){records.sort((a,b)=>a.d-b.d);ctx.globalAlpha=alpha;records.forEach(r=>{ctx.beginPath();ctx.moveTo(r.ps[0][0],r.ps[0][1]);r.ps.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));ctx.closePath();ctx.fillStyle=r.fill;ctx.fill();ctx.strokeStyle="#342c2444";ctx.lineWidth=.3;ctx.stroke()});ctx.globalAlpha=1}
function componentWorld(comp,c,shift=0,translation=[0,0,0]){const origin=add(sub(c.position,qrot(data.gripper.tcp_to_grasp_center_m,c.orientation_xyzw)),translation);return comp.vertices.map(v=>add(origin,qrot([v[0],v[1]+shift,v[2]],c.orientation_xyzw)))}
function drawGripper(c,color,translation=[0,0,0],alpha=.68){const h=c.jaw_width/2,items=[[data.gripper.base,0],[data.gripper.left_finger,-h-data.gripper.left_fingertip_inner_y],[data.gripper.right_finger,h-data.gripper.right_fingertip_inner_y]],records=[];items.forEach(([comp,shift])=>records.push(...faceRecords(componentWorld(comp,c,shift,translation),comp.faces,color)));drawFaces(records,alpha)}
function key(h,i){return `${h}|${i}`}function selected(){const h=data.holder_ids[state.row],i=data.inserter_ids[state.col];return{h,i,holder:data.holders[h],inserter:data.inserters[i],evaluation:data.evaluations[key(h,i)]||null}}
function cell(h,i){const H=data.holders[h],I=data.inserters[i];if(H.unary.status!=="accepted"||I.unary.status!=="accepted")return{status:"unary",reason:H.unary.status!=="accepted"?H.unary.reason:I.unary.reason};return data.evaluations[key(h,i)]||{status:"unchecked",reason:"not_checked_limit"}}
function motionTranslation(){const t=state.phase==="insertion"?data.step.final_to_pre_translation_m:data.step.retreat_translation_m;return state.phase==="insertion"?mul(t,1-state.progress):mul(t,state.progress)}
function renderScene(){ctx.clearRect(0,0,canvas.width,canvas.height);const s=selected(),r=s.evaluation,fail=r&&r.status==="rejected";polygon(S.table_vertices_assembly_m,"#2d79bd",.13);for(const id of data.step.assembled_part_ids_before)drawPart(id,id===data.base_part_id?"#23836b":"#68736f");const tr=motionTranslation(),partTr=state.phase==="insertion"?tr:[0,0,0];drawPart(data.step.incoming_part_id,"#dc7b20",1,partTr);drawPart(data.step.incoming_part_id,"#dc7b20",.12,data.step.final_to_pre_translation_m);drawPart(data.step.incoming_part_id,"#dc7b20",.12);drawGripper(s.holder,fail?"#ca4036":"#9a6722");drawGripper(s.inserter,fail?"#ca4036":"#7052b5",tr);drawGripper(s.inserter,"#7052b5",data.step.final_to_pre_translation_m,.12);drawGripper(s.inserter,"#7052b5",data.step.retreat_translation_m,.12);line(add(s.inserter.position,data.step.final_to_pre_translation_m),s.inserter.position,"#7052b5",2,[7,5]);line(s.inserter.position,add(s.inserter.position,data.step.retreat_translation_m),"#7052b5",2,[7,5])}
function renderDetails(){const s=selected(),c=cell(s.h,s.i),r=s.evaluation;const lines=[`holder:            ${s.h}`,`holder unary:      ${s.holder.unary.status} / ${s.holder.unary.reason}`,`holder score:      ${s.holder.score??"n/a"}`,`inserter:          ${s.i}`,`inserter unary:    ${s.inserter.unary.status} / ${s.inserter.unary.reason}`,`inserter score:    ${s.inserter.score??"n/a"}`,`cell:              ${c.status} / ${c.reason}`,r?`pair_id:           ${r.pair_id}`:"",r?`retained:          ${retained.has(r.pair_id)}`:"",r?`pair score:        ${r.score.toFixed(6)}`:"",r?`minimum clearance: ${r.minimum_clearance_m}`:"",r?`check:             ${r.collision_check}`:"",r?`details:           ${JSON.stringify(r.details,null,2)}`:""]; $("details").textContent=lines.filter(Boolean).join("\\n")}
function renderMatrix(){const rows=data.holder_ids.length,cols=data.inserter_ids.length,cw=matrix.width/cols,rh=matrix.height/rows;mctx.clearRect(0,0,matrix.width,matrix.height);data.holder_ids.forEach((h,y)=>data.inserter_ids.forEach((i,x)=>{const c=cell(h,i),r=data.evaluations[key(h,i)],color=c.status==="accepted"?"#178650":c.status==="rejected"?"#ca4036":c.status==="unary"?"#aaa49b":"#326eaa";mctx.fillStyle=color;mctx.fillRect(x*cw,y*rh,Math.ceil(cw),Math.ceil(rh));if(r&&retained.has(r.pair_id)){mctx.fillStyle="#f1b728";mctx.fillRect(x*cw,y*rh,Math.max(1,cw*.2),Math.ceil(rh))}}));mctx.strokeStyle="#111";mctx.lineWidth=2;mctx.strokeRect(state.col*cw,state.row*rh,Math.max(2,cw),Math.max(2,rh))}
function pairList(){const all=Object.values(data.evaluations),detailed=new Set([...data.retained_pair_ids,...data.detailed_rejected_pair_ids]);return all.filter(r=>state.filter==="retained"?retained.has(r.pair_id):state.filter==="accepted"?r.status==="accepted":state.filter==="rejected"?r.status==="rejected":detailed.has(r.pair_id)).sort((a,b)=>b.score-a.score||a.pair_id.localeCompare(b.pair_id))}
function selectEvaluation(r){state.row=Math.max(0,data.holder_ids.indexOf(r.holder_grasp_id));state.col=Math.max(0,data.inserter_ids.indexOf(r.inserter_grasp_id));if(r.status==="rejected"&&r.details.first_failing_phase){state.phase=r.details.first_failing_phase;state.progress=r.details.first_failing_progress??0;$("phase").value=state.phase;$("progress").value=state.progress;$("progressOut").textContent=`${Math.round(state.progress*100)}%`}renderAll()}
function renderPairs(){const a=pairList(),s=selected();state.pairIndex=Math.min(state.pairIndex,Math.max(0,a.length-1));$("count").textContent=`${a.length} pair records`;$("pairs").replaceChildren(...a.slice(0,120).map((r,i)=>{const b=document.createElement("button");b.className=`pair${r.holder_grasp_id===s.h&&r.inserter_grasp_id===s.i?" active":""}`;b.innerHTML=`<span><span class="pid">${r.pair_id}</span><br><span class="meta">${r.reason} · ${r.collision_check}</span></span><span class="meta">${r.score.toFixed(3)}</span>`;b.onclick=()=>{state.pairIndex=i;selectEvaluation(r)};return b}))}
function renderAll(){renderPairs();renderDetails();renderScene();renderMatrix()}const first=data.retained_pair_ids.length?Object.values(data.evaluations).find(r=>r.pair_id===data.retained_pair_ids[0]):Object.values(data.evaluations)[0];if(first){state.row=Math.max(0,data.holder_ids.indexOf(first.holder_grasp_id));state.col=Math.max(0,data.inserter_ids.indexOf(first.inserter_grasp_id))}
$("pills").replaceChildren(...Object.entries({...data.step.metadata,...data.step.reason_counts}).filter(([,v])=>typeof v==="number").map(([k,v])=>{const s=document.createElement("span");s.className="pill";s.textContent=`${k}: ${v}`;return s}));
$("phase").onchange=e=>{state.phase=e.target.value;renderScene()};$("progress").oninput=e=>{state.progress=Number(e.target.value);$("progressOut").textContent=`${Math.round(state.progress*100)}%`;renderScene()};$("filter").onchange=e=>{state.filter=e.target.value;state.pairIndex=0;const a=pairList();if(a.length)selectEvaluation(a[0]);else renderAll()};
$("prev").onclick=()=>{const a=pairList();if(a.length){state.pairIndex=(state.pairIndex-1+a.length)%a.length;selectEvaluation(a[state.pairIndex])}};$("next").onclick=()=>{const a=pairList();if(a.length){state.pairIndex=(state.pairIndex+1)%a.length;selectEvaluation(a[state.pairIndex])}};
matrix.onclick=e=>{const r=matrix.getBoundingClientRect();state.col=Math.min(data.inserter_ids.length-1,Math.floor((e.clientX-r.left)/r.width*data.inserter_ids.length));state.row=Math.min(data.holder_ids.length-1,Math.floor((e.clientY-r.top)/r.height*data.holder_ids.length));renderAll()};
canvas.onpointerdown=e=>{state.drag=true;state.lastX=e.clientX;state.lastY=e.clientY;canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!state.drag)return;state.yaw+=(e.clientX-state.lastX)*.01;state.pitch-=(e.clientY-state.lastY)*.01;state.lastX=e.clientX;state.lastY=e.clientY;renderScene()};canvas.onpointerup=canvas.onpointercancel=()=>state.drag=false;canvas.onwheel=e=>{e.preventDefault();state.zoom=Math.max(.3,Math.min(4,state.zoom*(e.deltaY<0?1.08:1/1.08)));renderScene()};
renderAll();
</script></body></html>""".replace("__DATA__", data_json)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def write_dual_grasp_pair_summary_html(
    result: DualGraspPairPlanningResult,
    output_path: str | Path,
) -> None:
    rows = []
    for step in result.steps:
        selected = "none" if not step.retained_pair_ids else step.retained_pair_ids[0]
        rows.append(
            f"""<a class="step" href="{pair_html_name(step)}">
<div><strong>Step {step.step_index}: incoming part {step.incoming_part_id}</strong>
<span>{step.step_id}</span></div>
<div class="counts"><b>{step.metadata["retained_pair_count"]}</b> retained
<b>{step.metadata["compatible_pair_count"]}</b> compatible
<b>{step.metadata["rejected_pair_count"]}</b> rejected
<b>{step.metadata["checked_pair_count"]}</b> checked</div>
<code>{selected}</code></a>"""
        )
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dual-Grasp Pair Summary</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#eeeae1;color:#24231f;font-family:Inter,"Segoe UI",sans-serif}}
main{{max-width:1000px;margin:auto;padding:28px}}h1{{margin:0 0 5px}}p{{color:#716b61}}.steps{{display:grid;gap:11px;margin-top:22px}}
.step{{display:grid;grid-template-columns:1.2fr 1.3fr 1fr;gap:18px;align-items:center;padding:17px;border:1px solid #d8cdb9;border-radius:14px;background:#fffaf1;color:inherit;text-decoration:none}}
.step:hover{{border-color:#23836b;transform:translateY(-1px)}}span{{display:block;color:#716b61;font-size:12px;margin-top:4px}}.counts{{display:flex;gap:9px;flex-wrap:wrap;font-size:12px}}b{{color:#178650}}code{{font-size:11px;overflow-wrap:anywhere}}
@media(max-width:750px){{.step{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>Dual-Grasp Pair Summary</h1><p>{result.assembly}: {" → ".join(result.selected_order)} · base {result.base_part_id}</p>
<p>Stage 3 checks KUKA end-effector geometry through insertion and retreat. Robot IK, links, and trajectories remain deferred.</p>
<div class="steps">{"".join(rows)}</div></main></body></html>"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def write_dual_grasp_pair_debug_artifacts(
    result: DualGraspPairPlanningResult,
    sequence: AssemblySequence,
    output_dir: str | Path,
) -> tuple[Path, tuple[Path, ...]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    step_paths = []
    for step in result.steps:
        path = output / pair_html_name(step)
        write_dual_grasp_pair_step_html(result, step, sequence, path)
        step_paths.append(path)
    summary_path = output / "dual_grasp_pair_summary.html"
    write_dual_grasp_pair_summary_html(result, summary_path)
    return summary_path, tuple(step_paths)


__all__ = [
    "write_dual_grasp_pair_debug_artifacts",
    "write_dual_grasp_pair_step_html",
    "write_dual_grasp_pair_summary_html",
]
