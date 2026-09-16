"""Interactive Stage-3 unary inserter grasp debugger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import quat_to_rotmat_xyzw, rotmat_to_quat_xyzw

from .assembly_sequence import AssemblySequence
from .assembly_sequence_debug_html import assembly_sequence_visual_payload
from .holder_state_debug_html import _gripper_payload

_TABLE_REASONS = {"inserter_table_collision"}
_CLEARANCE_REASONS = {"inserter_clearance_margin_failed"}
_ASSEMBLY_REASONS = {"assembly_insertion_sweep_collision", "inserter_retreat_collision"}


def _stage_for_status(status: object) -> str:
    reason = str(getattr(status, "reason"))
    if str(getattr(status, "status")) == "accepted":
        return "accepted"
    if reason in _TABLE_REASONS:
        return "table"
    if reason in _CLEARANCE_REASONS:
        return "clearance"
    if reason in _ASSEMBLY_REASONS:
        return "assembly"
    return "not_evaluated"


def _candidate_payloads(library: object) -> list[dict[str, object]]:
    source = getattr(library, "source_frame_pose_assembly")
    rotation = source.rotation_world_from_object
    translation = source.translation_world
    payloads = []
    for rank, status in enumerate(getattr(library, "candidate_statuses"), start=1):
        candidate = status.candidate

        def point(value: object) -> list[float]:
            return np.round(rotation @ np.asarray(value, dtype=float) + translation, 8).tolist()

        orientation = rotmat_to_quat_xyzw(rotation @ quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj))
        payloads.append(
            {
                "rank": rank,
                "grasp_id": candidate.grasp_id,
                "position": point(candidate.grasp_position_obj),
                "orientation_xyzw": np.round(orientation, 8).tolist(),
                "contact_a": point(candidate.contact_point_a_obj),
                "contact_b": point(candidate.contact_point_b_obj),
                "jaw_width": float(candidate.jaw_width),
                "score": candidate.score,
                "status": status.status,
                "reason": status.reason,
                "constraint_stage": _stage_for_status(status),
                "minimum_clearance_m": status.minimum_clearance_m,
                "details": status.details,
                "contact_patch_lateral_offset_m": float(candidate.contact_patch_lateral_offset_m),
                "contact_patch_approach_offset_m": float(candidate.contact_patch_approach_offset_m),
            }
        )
    return payloads


def write_inserter_unary_debug_html(
    *,
    library: object,
    sequence: AssemblySequence,
    planning: object,
    pair_config: object,
    output_path: str | Path,
) -> None:
    """Write a filterable overview plus a complete selected-grasp scene."""

    step = next(item for item in sequence.steps if item.step_id == library.step_id)
    sequence_payload = assembly_sequence_visual_payload(
        sequence, max_edges_per_part=260, max_faces_per_part=900
    )
    data = {
        "assembly": sequence.assembly,
        "step_id": library.step_id,
        "incoming_part_id": library.incoming_part_id,
        "assembled_part_ids_before": list(step.assembled_part_ids_before),
        "table_z_m": sequence.table_z_assembly_m,
        "insertion_translation_m": list(step.final_to_pre_insertion_translation_m),
        "retreat_translation_m": list(library.retreat_translation_assembly_m),
        "table_clearance_margin_m": float(pair_config.table_clearance_margin_m),
        "sequence": sequence_payload,
        "gripper": _gripper_payload(planning.gripper_collision_model),
        "candidates": _candidate_payloads(library),
        "reason_counts": library.reason_counts,
    }
    data_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    document = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inserter unary constraints</title><style>
:root{--bg:#f2eee5;--paper:#fffaf1;--ink:#24231f;--line:#d8cdb8;--muted:#706a5f;--table:#2777a8;--part:#24775e;--incoming:#d58024;--bad:#c43e32;--clear:#c55b96;--ok:#198754;--asm:#8154a3}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,sans-serif}.layout{display:grid;grid-template-columns:355px minmax(0,1fr);min-height:100vh}aside{padding:18px;background:var(--paper);border-right:1px solid var(--line);overflow:auto}main{padding:16px;min-width:0}h1{font-size:23px;margin:0 0 8px}h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;margin:0 0 8px}.note{color:var(--muted);font-size:12px;line-height:1.45}.controls{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:10px 0}.wide{grid-column:1/-1}button,input{font:inherit;border:1px solid var(--line);border-radius:8px;padding:8px;background:#fff;color:var(--ink)}button{cursor:pointer}.check{display:flex;align-items:center;gap:8px;padding:5px 0;color:var(--muted);font-size:13px}.check input{width:16px;height:16px;padding:0}.card{background:var(--paper);border:1px solid var(--line);border-radius:13px;padding:12px}.scenes{display:grid;grid-template-columns:1fr 1fr;gap:14px}canvas{width:100%;display:block;border-radius:9px;background:linear-gradient(#fff,#ebe5d9);cursor:grab;touch-action:none}#overview{aspect-ratio:1.25/1}#detail{aspect-ratio:1.25/1}pre{white-space:pre-wrap;margin:0;font:11px/1.48 ui-monospace,monospace}.legend{display:flex;flex-wrap:wrap;gap:10px;color:var(--muted);font-size:11px;margin-top:7px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}#candidateList{display:grid;gap:5px;max-height:44vh;overflow:auto}.candidate{text-align:left;display:grid;grid-template-columns:70px 1fr auto;gap:6px}.candidate.active{border-color:var(--incoming);background:#fff2e4}.mono{font:11px ui-monospace,monospace}.range{display:flex;gap:7px;align-items:center;color:var(--muted);font-size:12px}.range input{flex:1;padding:0}.trace{margin-top:12px}@media(max-width:1100px){.layout{grid-template-columns:1fr}.scenes{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body><div class="layout"><aside><h1>Stage 3 inserter debugger</h1><p class="note" id="subtitle"></p><h2>Constraint filters</h2><label class="check"><input type="checkbox" data-stage="table" checked>Table intersection</label><label class="check"><input type="checkbox" data-stage="clearance" checked>Table / geometry clearance</label><label class="check"><input type="checkbox" data-stage="assembly" checked>Assembly insertion / retreat sweep</label><label class="check"><input type="checkbox" data-stage="accepted" checked>Passed all unary checks</label><label class="check"><input type="checkbox" data-stage="not_evaluated">Not evaluated</label><div class="range"><span>Shown</span><input id="shown" type="range" min="1" max="100" value="100"><output id="shownOut">100%</output></div><div class="controls"><input id="search" class="wide" placeholder="Find grasp ID"><button id="prev">Previous</button><button id="next">Next</button></div><div id="count" class="note"></div><div id="candidateList"></div></aside><main><div class="scenes"><section class="card"><h2>All selected grasps — simplified</h2><canvas id="overview" width="1050" height="840"></canvas><div class="legend"><span><i class="dot" style="background:var(--bad)"></i>table</span><span><i class="dot" style="background:var(--clear)"></i>clearance</span><span><i class="dot" style="background:var(--asm)"></i>assembly sweep</span><span><i class="dot" style="background:var(--ok)"></i>passed</span></div></section><section class="card"><h2>Selected grasp — full collision scene</h2><canvas id="detail" width="1050" height="840"></canvas><div class="legend"><span><i class="dot" style="background:var(--part)"></i>assembled</span><span><i class="dot" style="background:var(--incoming)"></i>incoming target</span><span><i class="dot" style="background:var(--bad)"></i>selected PDZ hulls</span><span><i class="dot" style="background:var(--table)"></i>motion ghosts</span></div></section></div><section class="card trace"><h2>Selected constraint trace</h2><pre id="details"></pre></section></main></div><script>
const data=__DATA__,S=data.sequence.visualization,$=id=>document.getElementById(id);$("subtitle").textContent=`${data.assembly} / ${data.step_id}: ${data.incoming_part_id}. Click an overview marker or a list row for the detailed scene.`;
const colors={table:"#c43e32",clearance:"#c55b96",assembly:"#8154a3",accepted:"#198754",not_evaluated:"#9b9488"};const state={selected:0,shown:100,search:"",stages:new Set(["table","clearance","assembly","accepted"]),yaw:-.72,pitch:.5,zoom:1,panX:0,panY:0,drag:false,lastX:0,lastY:0};
function add(a,b){return a.map((v,i)=>v+b[i])}function sub(a,b){return a.map((v,i)=>v-b[i])}function mul(a,s){return a.map(v=>v*s)}function qrot(v,q){const[x,y,z,w]=q,tx=2*(y*v[2]-z*v[1]),ty=2*(z*v[0]-x*v[2]),tz=2*(x*v[1]-y*v[0]);return[v[0]+w*tx+y*tz-z*ty,v[1]+w*ty+z*tx-x*tz,v[2]+w*tz+x*ty-y*tx]}
function bounds(){return S.scene_bounds_assembly_m}function camera(v){const c=bounds().center,p=sub(v,c),cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch),x=cy*p[0]-sy*p[1],y=sy*p[0]+cy*p[1],z=p[2];return[x,cp*y-sp*z,sp*y+cp*z]};function project(v,canvas){const p=camera(v),s=.7*Math.min(canvas.width,canvas.height)/Math.max(bounds().extent,.15)*state.zoom;return[canvas.width/2+state.panX+p[0]*s,canvas.height*.53+state.panY-p[2]*s,p[1]]}
function filtered(){let a=data.candidates.filter(c=>state.stages.has(c.constraint_stage)&&(!state.search||c.grasp_id.toLowerCase().includes(state.search)));const n=Math.max(1,Math.ceil(a.length*state.shown/100));return a.filter((_,i)=>i<n)}function selected(){const a=filtered();if(!a.length)return null;state.selected=Math.min(state.selected,a.length-1);return a[state.selected]}
function line(ctx,a,b,color,w=1,dash=[]){const p=project(a,ctx.canvas),q=project(b,ctx.canvas);ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.strokeStyle=color;ctx.lineWidth=w;ctx.stroke();ctx.setLineDash([])}function point(ctx,v,color,r=4){const p=project(v,ctx.canvas);ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle="#fff";ctx.stroke()}
function polygon(ctx,vs,fill){const ps=vs.map(v=>project(v,ctx.canvas));ctx.beginPath();ctx.moveTo(ps[0][0],ps[0][1]);ps.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=fill;ctx.stroke()}
function drawPart(ctx,id,color,alpha=1,offset=[0,0,0]){const m=S.parts[id],v=m.vertices_assembly_m.map(p=>add(p,offset));ctx.globalAlpha=.22*alpha;m.faces.forEach(f=>polygon(ctx,f.map(i=>v[i]),color));ctx.globalAlpha=1;m.edges.forEach(e=>line(ctx,v[e[0]],v[e[1]],color,.6))}
function drawTable(ctx){ctx.globalAlpha=.17;polygon(ctx,S.table_vertices_assembly_m,"#2777a8");ctx.globalAlpha=1}
function compWorld(comp,c,shift,translation){const patch=qrot([-(c.contact_patch_lateral_offset_m||0),0,-(c.contact_patch_approach_offset_m||0)],c.orientation_xyzw),tcp=add(c.position,patch),o=add(sub(tcp,qrot(data.gripper.tcp_to_grasp_center_m,c.orientation_xyzw)),translation);return comp.vertices.map(v=>add(o,qrot([v[0],v[1]+shift,v[2]],c.orientation_xyzw)))}function drawGripper(ctx,c,color,translation=[0,0,0],alpha=.65){let items;if(data.gripper.model==="pdz_gripper"){const d=Math.max(0,(c.jaw_width-.012)/2);items=[[data.gripper.base,0],[data.gripper.left_finger,-d],[data.gripper.right_finger,d]]}else{const h=c.jaw_width/2;items=[[data.gripper.base,0],[data.gripper.left_finger,-h-data.gripper.left_fingertip_inner_y],[data.gripper.right_finger,h-data.gripper.right_fingertip_inner_y]]}ctx.globalAlpha=alpha;for(const [comp,shift] of items){const v=compWorld(comp,c,shift,translation);comp.faces.forEach(f=>polygon(ctx,f.map(i=>v[i]),color))}ctx.globalAlpha=1}
function clear(ctx){ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height)}function overview(){const ctx=$("overview").getContext("2d");clear(ctx);drawTable(ctx);for(const id of data.assembled_part_ids_before)drawPart(ctx,id,"#24775e",.7);drawPart(ctx,data.incoming_part_id,"#d58024",.75);const a=filtered(),sel=selected();a.forEach(c=>{line(ctx,c.contact_a,c.contact_b,colors[c.constraint_stage],c===sel?2.5:.7);if(c===sel){point(ctx,c.position,"#111",6);point(ctx,c.contact_a,colors[c.constraint_stage],5);point(ctx,c.contact_b,colors[c.constraint_stage],5)}})}
function detail(){const ctx=$("detail").getContext("2d"),c=selected();clear(ctx);drawTable(ctx);for(const id of data.assembled_part_ids_before)drawPart(ctx,id,"#24775e");drawPart(ctx,data.incoming_part_id,"#d58024");if(!c)return;const bad=colors[c.constraint_stage];drawGripper(ctx,c,bad);drawGripper(ctx,c,"#2777a8",data.insertion_translation_m,.18);drawGripper(ctx,c,"#2777a8",data.retreat_translation_m,.18);line(ctx,c.position,add(c.position,data.insertion_translation_m),"#2777a8",2,[7,5]);line(ctx,c.position,add(c.position,data.retreat_translation_m),"#2777a8",2,[3,4]);point(ctx,c.contact_a,"#d04732",5);point(ctx,c.contact_b,"#178166",5);point(ctx,c.position,bad,5)}
function trace(c){if(!c){$("details").textContent="No grasp matches the current filters.";return}const stage=c.constraint_stage,pass=x=>stage===x?"FAIL":(["table","clearance","assembly"].indexOf(stage)>["table","clearance","assembly"].indexOf(x)||stage==="accepted")?"PASS":"NOT REACHED";$("details").textContent=[`grasp_id: ${c.grasp_id}`,`status: ${c.status}`,`first reported reason: ${c.reason}`,`selected patch lateral/approach (m): ${c.contact_patch_lateral_offset_m.toFixed(4)}, ${c.contact_patch_approach_offset_m.toFixed(4)}`,`minimum_clearance_m: ${c.minimum_clearance_m??"n/a"}`,"",`1 table intersection (final, pre-insertion, retreat): ${pass("table")}`,`2 required table clearance (${data.table_clearance_margin_m} m): ${pass("clearance")}`,`3 exact FCL gripper sweep vs assembled parts: ${pass("assembly")}`,"",`details: ${JSON.stringify(c.details,null,2)}`,"",`shown hulls: final solid; final→pre-insertion blue ghost; final→retreat blue ghost.`].join("\n")}
function list(){const a=filtered(),c=selected();$("count").textContent=`${a.length} shown of ${data.candidates.length} status records`;$("candidateList").replaceChildren(...a.slice(0,220).map((x,i)=>{const b=document.createElement("button");b.className="candidate"+(x===c?" active":"");b.innerHTML=`<span class="mono">${x.grasp_id}</span><span class="mono">${x.reason}</span><span style="color:${colors[x.constraint_stage]}">${x.score===null?"n/a":x.score.toFixed(3)}</span>`;b.onclick=()=>{state.selected=i;render()};return b}))}
function render(){list();overview();detail();trace(selected());$("shownOut").textContent=state.shown+"%"}function reset(){state.selected=0;render()}document.querySelectorAll("[data-stage]").forEach(e=>e.onchange=()=>{e.checked?state.stages.add(e.dataset.stage):state.stages.delete(e.dataset.stage);reset()});$("shown").oninput=e=>{state.shown=Number(e.target.value);reset()};$("search").oninput=e=>{state.search=e.target.value.toLowerCase();reset()};$("prev").onclick=()=>{const n=filtered().length;if(n){state.selected=(state.selected-1+n)%n;render()}};$("next").onclick=()=>{const n=filtered().length;if(n){state.selected=(state.selected+1)%n;render()}};
$("overview").onclick=e=>{const r=e.currentTarget.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*e.currentTarget.width,y=(e.clientY-r.top)/r.height*e.currentTarget.height;let best=1e9,idx=0;filtered().forEach((c,i)=>{const p=project(c.position,e.currentTarget),d=(p[0]-x)**2+(p[1]-y)**2;if(d<best){best=d;idx=i}});if(best<900){state.selected=idx;render()}};for(const id of ["overview","detail"]){const c=$(id);c.onpointerdown=e=>{state.drag=true;state.lastX=e.clientX;state.lastY=e.clientY;c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(!state.drag)return;state.yaw+=(e.clientX-state.lastX)*.01;state.pitch-=(e.clientY-state.lastY)*.01;state.lastX=e.clientX;state.lastY=e.clientY;overview();detail()};c.onpointerup=c.onpointercancel=()=>state.drag=false;c.onwheel=e=>{e.preventDefault();state.zoom=Math.max(.3,Math.min(4,state.zoom*(e.deltaY<0?1.1:1/1.1)));overview();detail()}}render();
</script></body></html>'''.replace("__DATA__", data_json)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
