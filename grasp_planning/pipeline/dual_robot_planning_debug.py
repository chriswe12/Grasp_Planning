"""Live browser debugger for dual-arm MoveIt candidate planning."""

from __future__ import annotations

import json
import math
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import quat_to_rotmat_xyzw
from grasp_planning.grasping.mesh_io import load_triangle_mesh
from grasp_planning.grasping.world_constraints import ObjectWorldPose

from .holder_state_debug_html import _gripper_payload


def _pose_payload(pose: Any) -> dict[str, list[float]]:
    return {
        "position_world_m": [float(value) for value in pose.position_world],
        "orientation_xyzw_world": [float(value) for value in pose.orientation_xyzw_world],
    }


def _source_vertices(
    vertices_assembly_m: np.ndarray,
    *,
    source_pose_assembly: Any,
) -> np.ndarray:
    """Express final assembly-coordinate vertices in a part source frame."""

    return (
        vertices_assembly_m - source_pose_assembly.translation_world[None, :]
    ) @ source_pose_assembly.rotation_world_from_object


def _world_vertices(vertices_source_m: np.ndarray, *, source_pose_world: Any) -> np.ndarray:
    return (
        vertices_source_m @ source_pose_world.rotation_world_from_object.T
        + source_pose_world.translation_world[None, :]
    )


def _compact_mesh_payload(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    vertex_field: str,
    max_faces: int = 1600,
) -> dict[str, object]:
    """Return deterministic, bounded geometry suitable for a live browser."""

    vertices_array = np.asarray(vertices, dtype=float)
    faces_array = np.asarray(faces, dtype=np.int64)
    if len(faces_array) > max_faces:
        selected = np.linspace(
            0,
            len(faces_array) - 1,
            num=max_faces,
            dtype=np.int64,
        )
        faces_array = faces_array[selected]
    used = np.unique(faces_array.reshape(-1))
    remap = np.full(len(vertices_array), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    compact_faces = remap[faces_array]
    edges = {
        tuple(sorted((int(face[index]), int(face[(index + 1) % 3])))) for face in compact_faces for index in range(3)
    }
    return {
        vertex_field: np.round(vertices_array[used], 6).tolist(),
        "faces": compact_faces.tolist(),
        "edges": [list(edge) for edge in sorted(edges)],
    }


def _visual_gripper_floor_clearance_m(
    *,
    gripper: dict[str, object],
    target: dict[str, object],
    jaw_width_m: float,
    floor_z_world_m: float,
) -> float:
    """Return clearance of the exact gripper mesh rendered in the browser."""

    rotation = quat_to_rotmat_xyzw(target["orientation_xyzw_world"])
    grasp_center = np.asarray(target["position_world_m"], dtype=float)
    tcp_to_grasp_center = np.asarray(gripper["tcp_to_grasp_center_m"], dtype=float)
    gripper_origin = grasp_center - rotation @ tcp_to_grasp_center
    half_width = 0.5 * float(jaw_width_m)
    components = (
        (dict(gripper["base"]), 0.0),
        (
            dict(gripper["left_finger"]),
            -half_width - float(gripper["left_fingertip_inner_y"]),
        ),
        (
            dict(gripper["right_finger"]),
            half_width - float(gripper["right_fingertip_inner_y"]),
        ),
    )
    minimum_z = math.inf
    for component, lateral_shift in components:
        vertices = np.asarray(component["vertices"], dtype=float).copy()
        vertices[:, 1] += lateral_shift
        vertices_world = vertices @ rotation.T + gripper_origin[None, :]
        minimum_z = min(minimum_z, float(np.min(vertices_world[:, 2])))
    return minimum_z - float(floor_z_world_m)


def dual_robot_planning_scene_payload(task: Any) -> dict[str, object]:
    """Build a bounded world-frame scene for one runtime pair/transition."""

    subassembly_parts = []
    for part in task.subassembly_parts:
        mesh = load_triangle_mesh(part.mesh_path, scale=float(task.mesh_scale))
        vertices_source = _source_vertices(
            mesh.vertices_obj,
            source_pose_assembly=task.holder_source_pose_assembly,
        )
        vertices_world = _world_vertices(
            vertices_source,
            source_pose_world=task.holder_source_pose_world,
        )
        subassembly_parts.append(
            {
                "part_id": str(part.part_id),
                "is_base": str(part.part_id) == str(task.base_part_id),
                **_compact_mesh_payload(
                    vertices_world,
                    mesh.faces,
                    vertex_field="vertices_world_m",
                ),
            }
        )

    incoming_mesh = load_triangle_mesh(
        task.incoming_mesh_path,
        scale=float(task.mesh_scale),
    )
    incoming_source_vertices = _source_vertices(
        incoming_mesh.vertices_obj,
        source_pose_assembly=task.incoming_source_pose_assembly,
    )
    incoming = {
        "part_id": str(task.incoming_part_id),
        **_compact_mesh_payload(
            incoming_source_vertices,
            incoming_mesh.faces,
            vertex_field="vertices_source_m",
        ),
    }
    pickup_pose = _pose_payload(task.incoming_pickup_source_pose_world)
    preinsertion_pose = _pose_payload(task.incoming_preinsertion_source_pose_world)
    final_pose = _pose_payload(task.incoming_final_source_pose_world)
    lift = float(task.transport_clearance_m)

    def lifted(raw_pose: dict[str, list[float]]) -> dict[str, list[float]]:
        result = {key: list(value) for key, value in raw_pose.items()}
        result["position_world_m"][2] += lift
        return result

    task_payload = task.to_payload()
    targets = dict(task_payload["targets"])
    holder_grasp = dict(dict(task_payload["grasps"])["holder"])
    inserter_grasp = dict(dict(task_payload["grasps"])["inserter_pickup"])
    gripper = _gripper_payload()
    floor_z_world_m = float(task.pickup_floor_z_world_m)
    gripper_floor_clearance_m = {
        name: _visual_gripper_floor_clearance_m(
            gripper=gripper,
            target=dict(target_payload),
            jaw_width_m=(
                float(holder_grasp["jaw_width_m"])
                if name.startswith("holder_")
                else float(inserter_grasp["jaw_width_m"])
            ),
            floor_z_world_m=floor_z_world_m,
        )
        for name, target_payload in targets.items()
        if name.startswith(("holder_", "inserter_"))
    }
    return {
        "schema_version": 1,
        "frame_id": "base_link",
        "assembly": str(task.assembly),
        "step_id": str(task.step_id),
        "incoming_part_id": str(task.incoming_part_id),
        "base_part_id": str(task.base_part_id),
        "pair_id": str(task.pair_id),
        "transition_id": str(task.transition_id),
        "execution_candidate_id": str(task.execution_candidate_id),
        "selection_score": float(task.selection_score),
        "transition_motion_score": float(task.transition_motion_score),
        "layout_proxy_components": dict(getattr(task, "layout_proxy_components", {})),
        "subassembly_parts": subassembly_parts,
        "incoming": incoming,
        "incoming_poses": {
            "pickup": pickup_pose,
            "pickup_lift": lifted(pickup_pose),
            "above_preinsertion": lifted(preinsertion_pose),
            "preinsertion": preinsertion_pose,
            "final": final_pose,
        },
        "targets": targets,
        "jaw_widths_m": {
            "holder": float(holder_grasp["jaw_width_m"]),
            "inserter": float(inserter_grasp["jaw_width_m"]),
        },
        "gripper": gripper,
        "gripper_floor_clearance_m": gripper_floor_clearance_m,
        "robot_bases_world_m": {
            "holder": list(task_payload["layout"]["holder_base_world_m"]),
            "inserter": list(task_payload["layout"]["inserter_base_world_m"]),
        },
        "floor_z_world_m": floor_z_world_m,
        "candidate_filter_diagnostics": dict(getattr(task, "candidate_filter_diagnostics", {})),
        "transition_symmetry": dict(task.transition_symmetry),
    }


def _object_pose_from_plan_payload(raw: Any, *, field_name: str) -> ObjectWorldPose:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be an object.")
    position = raw.get("position_world_m")
    orientation = raw.get("orientation_xyzw_world")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError(f"{field_name}.position_world_m must contain three values.")
    if not isinstance(orientation, (list, tuple)) or len(orientation) != 4:
        raise ValueError(f"{field_name}.orientation_xyzw_world must contain four values.")
    return ObjectWorldPose(
        position_world=tuple(float(value) for value in position),
        orientation_xyzw_world=tuple(float(value) for value in orientation),
    )


def dual_robot_planning_scene_payload_from_plan(
    plan: dict[str, object],
) -> dict[str, object]:
    """Build the existing live scene directly from a serialized real task."""

    objects = plan.get("objects")
    layout = plan.get("layout")
    targets = plan.get("targets")
    if not isinstance(objects, dict) or not isinstance(layout, dict) or not isinstance(targets, dict):
        raise ValueError("Dual task must contain objects, layout, and targets objects for live debugging.")
    subassembly = objects.get("subassembly")
    incoming = objects.get("incoming")
    if not isinstance(subassembly, dict) or not isinstance(incoming, dict):
        raise ValueError("Dual task must contain subassembly and incoming object descriptions.")
    raw_parts = subassembly.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError("Dual task subassembly must contain at least one part mesh.")
    subassembly_parts = []
    for index, raw_part in enumerate(raw_parts):
        if not isinstance(raw_part, dict) or not raw_part.get("mesh_path"):
            raise ValueError(f"Dual task subassembly part {index} has no mesh_path.")
        subassembly_parts.append(
            SimpleNamespace(
                part_id=str(raw_part.get("part_id", index)),
                mesh_path=Path(str(raw_part["mesh_path"])),
            )
        )
    if not incoming.get("mesh_path"):
        raise ValueError("Dual task incoming object has no mesh_path.")

    pickup_grasp = targets.get("inserter_pickup_grasp")
    pickup_lift = targets.get("inserter_pickup_lift")
    if not isinstance(pickup_grasp, dict) or not isinstance(pickup_lift, dict):
        raise ValueError("Dual task is missing incoming pickup grasp/lift targets.")
    pickup_grasp_position = pickup_grasp.get("position_world_m")
    pickup_lift_position = pickup_lift.get("position_world_m")
    if not isinstance(pickup_grasp_position, (list, tuple)) or not isinstance(
        pickup_lift_position,
        (list, tuple),
    ):
        raise ValueError("Dual task pickup grasp/lift targets have invalid positions.")
    transport_clearance_m = float(pickup_lift_position[2]) - float(pickup_grasp_position[2])

    task = SimpleNamespace(
        assembly=str(plan.get("assembly", "")),
        step_id=str(plan.get("step_id", "")),
        incoming_part_id=str(plan.get("incoming_part_id", incoming.get("part_id", ""))),
        base_part_id=str(plan.get("base_part_id", subassembly.get("base_part_id", ""))),
        pair_id=str(plan.get("pair_id", "")),
        transition_id=str(plan.get("transition_id", "")),
        execution_candidate_id=str(plan.get("execution_candidate_id", plan.get("pair_id", ""))),
        selection_score=float(plan.get("selection_score", plan.get("pair_score", 0.0))),
        transition_motion_score=float(plan.get("transition_motion_score", 0.0)),
        layout_proxy_components=dict(plan.get("layout_proxy_components", {})),
        subassembly_parts=tuple(subassembly_parts),
        mesh_scale=float(subassembly.get("mesh_scale", incoming.get("mesh_scale", 1.0))),
        holder_source_pose_assembly=_object_pose_from_plan_payload(
            subassembly.get("source_pose_assembly"),
            field_name="objects.subassembly.source_pose_assembly",
        ),
        holder_source_pose_world=_object_pose_from_plan_payload(
            subassembly.get("source_pose_world"),
            field_name="objects.subassembly.source_pose_world",
        ),
        incoming_mesh_path=Path(str(incoming["mesh_path"])),
        incoming_source_pose_assembly=_object_pose_from_plan_payload(
            incoming.get("source_pose_assembly"),
            field_name="objects.incoming.source_pose_assembly",
        ),
        incoming_pickup_source_pose_world=_object_pose_from_plan_payload(
            incoming.get("pickup_source_pose_world"),
            field_name="objects.incoming.pickup_source_pose_world",
        ),
        incoming_preinsertion_source_pose_world=_object_pose_from_plan_payload(
            incoming.get("preinsertion_source_pose_world"),
            field_name="objects.incoming.preinsertion_source_pose_world",
        ),
        incoming_final_source_pose_world=_object_pose_from_plan_payload(
            incoming.get("final_source_pose_world"),
            field_name="objects.incoming.final_source_pose_world",
        ),
        transport_clearance_m=transport_clearance_m,
        pickup_floor_z_world_m=float(layout.get("pickup_floor_z_world_m", 0.0)),
        transition_symmetry=dict(plan.get("transition_symmetry", {})),
        candidate_filter_diagnostics=dict(plan.get("candidate_filter_diagnostics", {})),
        to_payload=lambda: plan,
    )
    return dual_robot_planning_scene_payload(task)


_LIVE_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dual-Arm Live Planner</title>
<style>
:root{--bg:#171a1f;--panel:#22272e;--line:#39414b;--ink:#f2f5f7;--muted:#9da7b3;--holder:#efb64d;--inserter:#a98bfa;--base:#31c49a;--part:#85969c;--incoming:#ff842c;--ok:#39c47b;--bad:#ff5964;--active:#54a8ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Segoe UI",sans-serif}.layout{display:grid;grid-template-columns:360px minmax(0,1fr);min-height:100vh}.panel{background:var(--panel);border-right:1px solid var(--line);padding:18px;overflow:auto}main{padding:14px;min-width:0}.card{background:#1d2228;border:1px solid var(--line);border-radius:14px;padding:12px;margin-bottom:12px}h1{font-size:23px;margin:0 0 5px}h2{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:16px 0 8px}.sub,.small{font-size:12px;color:var(--muted);line-height:1.45}.phase{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.phase div{border:1px solid var(--line);border-radius:9px;padding:8px 5px;text-align:center;font-size:11px;color:var(--muted)}.phase div.active{border-color:var(--active);background:#17324c;color:#d8edff}.phase div.done{border-color:var(--ok);color:#9be5bd}.badge{display:inline-block;border-radius:999px;padding:4px 8px;background:#343b44;font:700 11px ui-monospace,monospace}.badge.ok{background:#163d2b;color:#75e2a7}.badge.bad{background:#4a2025;color:#ff9aa1}.kv{display:grid;grid-template-columns:105px 1fr;gap:6px;font:11px/1.4 ui-monospace,monospace}.kv span:nth-child(odd){color:var(--muted)}#message{white-space:pre-wrap;font:11px/1.45 ui-monospace,monospace;color:#d6dce2;max-height:150px;overflow:auto}.history{display:grid;gap:5px}.event{border-left:3px solid var(--line);padding:4px 7px;font:10px/1.35 ui-monospace,monospace;color:var(--muted)}.event.failed{border-color:var(--bad);color:#ffadb2}.event.succeeded{border-color:var(--ok);color:#9be5bd}canvas{width:100%;height:calc(100vh - 30px);min-height:620px;display:block;border-radius:12px;background:linear-gradient(#242b33,#13171b);cursor:grab;touch-action:none}.legend{position:absolute;left:28px;bottom:28px;background:#171b20dd;border:1px solid var(--line);border-radius:10px;padding:8px 10px;font-size:11px;color:var(--muted)}.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin:0 4px 0 9px}.scene-wrap{position:relative}@media(max-width:950px){.layout{grid-template-columns:1fr}.panel{border-right:0;border-bottom:1px solid var(--line)}canvas{height:70vh}}
</style></head><body><div class="layout"><aside class="panel">
<h1>Dual-Arm Live Planner</h1><div class="sub">World frame: <code>base_link</code> · live MoveIt candidate state</div>
<h2>Planning stage</h2><div class="phase"><div id="stage-holder">1 · holder grasp</div><div id="stage-inserter">2 · incoming grasp</div><div id="stage-transition">3 · transition</div></div>
<div class="card" style="margin-top:12px"><span id="status" class="badge">waiting</span><h2>Current candidate</h2><div class="kv">
<span>attempt</span><span id="attempt">-</span><span>pair</span><span id="pair">-</span><span>transition</span><span id="transition">-</span><span>phase</span><span id="target">-</span><span>score</span><span id="score">-</span><span>arm crossing</span><span id="crossing">-</span><span>holder floor</span><span id="holder-floor">-</span></div></div>
<h2>Candidate checks</h2><div class="card"><div class="kv">
<span>floor plane</span><span id="floor-plane">-</span><span>pickup floor</span><span id="pickup-counts">-</span><span>Stage-3 pool</span><span id="stage3-counts">-</span><span>pose fallback</span><span id="fallback-counts">-</span><span>pose queue</span><span id="queue-counts">-</span><span>joint pre-rank</span><span id="joint-counts">-</span><span>exact IK</span><span id="ik-counts">-</span></div></div>
<h2>Planner message</h2><div id="message">Waiting for planner state…</div><h2>Recent events</h2><div id="history" class="history"></div>
</aside><main><div class="scene-wrap"><canvas id="scene" width="1500" height="980"></canvas><div class="legend"><i class="dot" style="background:var(--base)"></i>base <i class="dot" style="background:var(--part)"></i>assembled <i class="dot" style="background:var(--incoming)"></i>incoming <i class="dot" style="background:var(--holder)"></i>holder <i class="dot" style="background:var(--inserter)"></i>inserter</div></div></main></div>
<script>
const $=id=>document.getElementById(id),canvas=$("scene"),ctx=canvas.getContext("2d");let live=null,scene=null,lastServerId="",lastRevision=-1,lastStateRevision=-1,lastVisualKey="",renderQueued=false,frameBounds={c:[.55,0,.18],e:1};
const view={yaw:-.78,pitch:.58,zoom:1,panX:0,panY:0,drag:false,x:0,y:0};
function add(a,b){return a.map((v,i)=>v+b[i])}function sub(a,b){return a.map((v,i)=>v-b[i])}function qrot(v,q){const[x,y,z,w]=q,tx=2*(y*v[2]-z*v[1]),ty=2*(z*v[0]-x*v[2]),tz=2*(x*v[1]-y*v[0]);return[v[0]+w*tx+y*tz-z*ty,v[1]+w*ty+z*tx-x*tz,v[2]+w*tz+x*ty-y*tx]}
function incomingPose(){if(!scene)return null;const p=live?.phase||"";if(p==="inserter_pickup_lift")return scene.incoming_poses.pickup_lift;if(p==="inserter_above_preinsertion")return scene.incoming_poses.above_preinsertion;if(p==="ik_preflight"||p==="joint_space_ranking"||p==="inserter_preinsertion"||live?.status==="complete")return scene.incoming_poses.preinsertion;return scene.incoming_poses.pickup}
function transformedIncoming(){if(!scene)return[];const pose=incomingPose(),q=pose.orientation_xyzw_world,t=pose.position_world_m;return scene.incoming.vertices_source_m.map(v=>add(qrot(v,q),t))}
function bounds(incomingVertices){let pts=[];if(scene){scene.subassembly_parts.forEach(p=>pts.push(...p.vertices_world_m));pts.push(...incomingVertices);pts.push(...Object.values(scene.robot_bases_world_m));Object.values(scene.targets).forEach(t=>pts.push(t.position_world_m))}if(!pts.length)return{c:[.55,0,.18],e:1};const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];pts.forEach(p=>{for(let i=0;i<3;i++){if(p[i]<lo[i])lo[i]=p[i];if(p[i]>hi[i])hi[i]=p[i]}});return{c:lo.map((v,i)=>(v+hi[i])/2),e:Math.max(.5,...lo.map((v,i)=>hi[i]-v))}}
function camera(v){const b=frameBounds,p=sub(v,b.c),cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch),x=cy*p[0]-sy*p[1],y=sy*p[0]+cy*p[1],z=p[2];return[x,cp*y-sp*z,sp*y+cp*z,b.e]}
function project(v){const p=camera(v),s=.78*Math.min(canvas.width,canvas.height)/p[3]*view.zoom;return[canvas.width/2+view.panX+p[0]*s,canvas.height*.54+view.panY-p[2]*s,p[1]]}
function line(a,b,color,w=2,dash=[]){const p=project(a),q=project(b);ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.strokeStyle=color;ctx.lineWidth=w;ctx.stroke();ctx.setLineDash([])}
function label(point,text,color){const p=project(point);ctx.font="700 16px ui-monospace,monospace";ctx.fillStyle=color;ctx.fillText(text,p[0]+8,p[1]-8)}
function drawMesh(vertices,faces,color,alpha=.55){const projected=vertices.map(project),records=faces.map(f=>{const p=[projected[f[0]],projected[f[1]],projected[f[2]]];return{p,d:(p[0][2]+p[1][2]+p[2][2])/3}}).sort((a,b)=>a.d-b.d);ctx.globalAlpha=alpha;records.forEach(r=>{ctx.beginPath();ctx.moveTo(r.p[0][0],r.p[0][1]);ctx.lineTo(r.p[1][0],r.p[1][1]);ctx.lineTo(r.p[2][0],r.p[2][1]);ctx.closePath();ctx.fillStyle=color;ctx.fill();ctx.strokeStyle="#0d101388";ctx.lineWidth=.35;ctx.stroke()});ctx.globalAlpha=1}
function compWorld(comp,target,shift=0){const q=target.orientation_xyzw,center=target.position,jaw=target.jaw_width/2,offset=qrot(scene.gripper.tcp_to_grasp_center_m,q),origin=sub(center,offset);return comp.vertices.map(v=>add(origin,qrot([v[0],v[1]+shift,v[2]],q)))}
function drawGripper(target,color,alpha=.75){if(!target)return;const clearance=scene.gripper_floor_clearance_m?.[target.label],actualColor=Number.isFinite(clearance)&&clearance<0?"#ff5964":color,g=scene.gripper,h=target.jaw_width/2,items=[[g.base,0],[g.left_finger,-h-g.left_fingertip_inner_y],[g.right_finger,h-g.right_fingertip_inner_y]];items.forEach(([comp,shift])=>drawMesh(compWorld(comp,target,shift),comp.faces,actualColor,alpha));label(target.position,target.label,actualColor)}
function target(name,role){const t=scene.targets[name];if(!t)return null;return{position:t.position_world_m,orientation_xyzw:t.orientation_xyzw_world,jaw_width:scene.jaw_widths_m[role],label:name}}
function stage(){const p=live?.phase||"";if(p.startsWith("holder_"))return"holder";if(p==="pickup_floor_check"||p==="inserter_pickup_pregrasp"||p==="inserter_pickup_grasp")return"inserter";if(p==="joint_space_ranking"||p.startsWith("inserter_")||p==="transition")return"transition";return p==="complete"?"transition":""}
function render(){if(!scene)return;const incomingVertices=transformedIncoming();frameBounds=bounds(incomingVertices);ctx.clearRect(0,0,canvas.width,canvas.height);const f=scene.floor_z_world_m;const table=[[.05,-.72,f],[1.45,-.72,f],[1.45,.72,f],[.05,.72,f]];ctx.globalAlpha=.18;ctx.fillStyle="#4d94d8";ctx.beginPath();table.map(project).forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fill();ctx.globalAlpha=1;line([0,0,f],[.18,0,f],"#f05a62",3);line([0,0,f],[0,.18,f],"#49cf86",3);line([0,0,f],[0,0,f+.18],"#559cff",3);scene.subassembly_parts.forEach(p=>drawMesh(p.vertices_world_m,p.faces,p.is_base?"#31c49a":"#85969c",.64));drawMesh(incomingVertices,scene.incoming.faces,"#ff842c",.75);const bases=scene.robot_bases_world_m;Object.entries(bases).forEach(([role,p])=>{const q=[p[0],p[1],f];ctx.fillStyle=role==="holder"?"#efb64d":"#a98bfa";const s=project(q);ctx.beginPath();ctx.arc(s[0],s[1],13,0,Math.PI*2);ctx.fill();label(q,role+" base",ctx.fillStyle)});const phase=live?.phase||"";const holderName=phase==="holder_pregrasp"?"holder_pregrasp":"holder_grasp";let inserterName="inserter_pickup_grasp";if(phase==="ik_preflight"||phase==="joint_space_ranking")inserterName="inserter_preinsertion";else if(phase.startsWith("inserter_"))inserterName=phase;drawGripper(target(holderName,"holder"),"#efb64d",stage()==="holder"?.95:.62);drawGripper(target(inserterName,"inserter"),"#a98bfa",stage()==="inserter"||stage()==="transition"?.95:.45);const holderTarget=scene.targets.holder_grasp.position_world_m,inserterTarget=scene.targets.inserter_preinsertion.position_world_m;line(bases.holder,holderTarget,"#efb64d",2,[5,6]);line(bases.inserter,inserterTarget,"#a98bfa",2,[5,6]);const path=["inserter_pickup_lift","inserter_above_preinsertion","inserter_preinsertion"].map(n=>scene.targets[n].position_world_m);line(path[0],path[1],"#a98bfa",2,[9,6]);line(path[1],path[2],"#a98bfa",2,[9,6])}
function scheduleRender(){if(renderQueued)return;renderQueued=true;requestAnimationFrame(()=>{renderQueued=false;render()})}
function count(c,key){return Number.isFinite(Number(c[key]))?Number(c[key]):0}
function updatePanel(){if(!live)return;$("attempt").textContent=`${live.attempt_index||0} / ${live.attempt_total||0}`;$("pair").textContent=live.pair_id||"-";$("transition").textContent=live.transition_id||"-";$("target").textContent=live.phase||"-";$("score").textContent=scene?scene.selection_score.toFixed(5):"-";const cross=scene?.layout_proxy_components||{};$("crossing").textContent=scene?(cross.transition_segments_cross_xy?"INSERT CROSSED":cross.pickup_segments_cross_xy?"pickup crossed":"clear"):"-";$("crossing").style.color=(cross.transition_segments_cross_xy||cross.pickup_segments_cross_xy)?"var(--bad)":"var(--ok)";const holderClearance=scene?.gripper_floor_clearance_m?.holder_grasp,$holderFloor=$("holder-floor");$holderFloor.textContent=Number.isFinite(holderClearance)?`${(1000*holderClearance).toFixed(1)} mm`:"-";$holderFloor.style.color=Number.isFinite(holderClearance)&&holderClearance<0?"var(--bad)":"var(--ok)";const c=live.candidate_counts||{},floorZ=Number(c.pickup_floor_z_world_m);$("floor-plane").textContent=Number.isFinite(floorZ)?`z = ${floorZ.toFixed(3)} m`:"-";$("pickup-counts").textContent=`${count(c,"pickup_grasps_accepted")} / ${count(c,"pickup_grasps_checked")} accepted`;$("stage3-counts").textContent=`${count(c,"stage3_retained_execution_candidates")} executions · ${count(c,"stage3_retained_pairs")} pairs`;$("fallback-counts").textContent=`${count(c,"pose_feasible_retained_execution_candidates")} retained · ${count(c,"pose_feasible_validated_transition_fallback_candidates")} sym · ${count(c,"pose_feasible_identity_fallback_candidates")} identity`;$("queue-counts").textContent=`${count(c,"planner_queue_execution_candidates")} executions · ${count(c,"planner_queue_noncrossing_execution_candidates")} clear · ${count(c,"planner_queue_crossed_execution_candidates")} crossed · ${count(c,"planner_queue_unique_holder_grasps")} H · ${count(c,"planner_queue_unique_inserter_grasps")} I`;$("joint-counts").textContent=`${count(c,"joint_rank_candidates_planned")} planned / ${count(c,"joint_rank_candidates_checked")} checked`;$("ik-counts").textContent=`${count(c,"exact_ik_pair_tasks_checked")} pairs · ${count(c,"exact_ik_holder_grasps_checked")} H · ${count(c,"exact_ik_inserter_grasps_checked")} I`;$("message").textContent=live.message||"";const badge=$("status");badge.textContent=live.status||"waiting";badge.className=`badge ${live.status==="failed"||live.status==="fatal"?"bad":live.status==="succeeded"||live.status==="complete"?"ok":""}`;const active=stage(),order=["holder","inserter","transition"];order.forEach((name,i)=>{const el=$("stage-"+name);el.className=name===active?"active":active&&order.indexOf(active)>i?"done":""});$("history").replaceChildren(...(live.history||[]).slice().reverse().map(e=>{const d=document.createElement("div");d.className=`event ${e.status||""}`;d.textContent=`${e.attempt_index||0} · ${e.phase||"-"} · ${e.status||""}${e.message?" · "+e.message:""}`;return d}))}
async function poll(){try{const r=await fetch(`/state.json?t=${Date.now()}`,{cache:"no-store"});if(!r.ok)throw Error(r.status);const nextLive=await r.json(),serverChanged=nextLive.server_id!==lastServerId;if(serverChanged){lastServerId=nextLive.server_id;lastRevision=-1;lastStateRevision=-1}const sceneChanged=nextLive.scene_revision!==lastRevision;if(sceneChanged){if(nextLive.scene_revision>0){const s=await fetch(`/scene.json?t=${Date.now()}`,{cache:"no-store"});if(!s.ok)throw Error(s.status);scene=await s.json()}else scene=null;lastRevision=nextLive.scene_revision}const stateChanged=nextLive.state_revision!==lastStateRevision;live=nextLive;if(stateChanged){lastStateRevision=live.state_revision;updatePanel()}const visualKey=`${live.server_id}|${live.scene_revision}|${live.phase}|${live.status}`;if(sceneChanged||visualKey!==lastVisualKey){lastVisualKey=visualKey;scheduleRender()}}catch(e){if(!live||!["complete","failed","fatal"].includes(live.status)){const b=$("status");b.textContent="planner disconnected";b.className="badge bad"}}setTimeout(poll,100)}
canvas.onpointerdown=e=>{view.drag=true;view.x=e.clientX;view.y=e.clientY;canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!view.drag)return;view.yaw+=(e.clientX-view.x)*.006;view.pitch=Math.max(-1.35,Math.min(1.35,view.pitch+(e.clientY-view.y)*.006));view.x=e.clientX;view.y=e.clientY;scheduleRender()};canvas.onpointerup=canvas.onpointercancel=()=>view.drag=false;canvas.onwheel=e=>{e.preventDefault();view.zoom=Math.max(.35,Math.min(4,view.zoom*Math.exp(-e.deltaY*.001)));scheduleRender()};poll();
</script></body></html>"""


class DualRobotPlanningDebugServer:
    """Serve a localhost-only live view updated by the planner thread."""

    def __init__(self, *, port: int = 0) -> None:
        self._lock = threading.Lock()
        self._scene: dict[str, object] = {}
        self._state: dict[str, object] = {
            "schema_version": 1,
            "server_id": uuid.uuid4().hex,
            "state_revision": 0,
            "scene_revision": 0,
            "attempt_index": 0,
            "attempt_total": 0,
            "pair_id": "",
            "transition_id": "",
            "phase": "startup",
            "status": "waiting",
            "message": "Waiting for the first candidate.",
            "candidate_counts": {},
            "history": [],
        }
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path in {"/", "/index.html"}:
                    body = _LIVE_HTML.encode("utf-8")
                    content_type = "text/html; charset=utf-8"
                elif path == "/state.json":
                    with owner._lock:
                        payload = dict(owner._state)
                    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                    content_type = "application/json"
                elif path == "/scene.json":
                    with owner._lock:
                        payload = dict(owner._scene)
                    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                    content_type = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        ThreadingHTTPServer.allow_reuse_address = True
        self._server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="dual-planning-debug-server",
            daemon=True,
        )

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self, *, open_browser: bool = True) -> str:
        self._thread.start()
        if open_browser:
            threading.Thread(
                target=webbrowser.open,
                args=(self.url,),
                kwargs={"new": 2},
                daemon=True,
            ).start()
        return self.url

    def update(
        self,
        *,
        task: Any | None = None,
        scene_payload: dict[str, object] | None = None,
        attempt_index: int | None = None,
        attempt_total: int | None = None,
        phase: str,
        status: str,
        message: str = "",
        candidate_counts: dict[str, object] | None = None,
        record_event: bool = True,
    ) -> None:
        if task is not None and scene_payload is not None:
            raise ValueError("Provide either task or scene_payload, not both.")
        scene = None
        if task is not None or scene_payload is not None:
            next_candidate = str(
                task.execution_candidate_id if task is not None else scene_payload.get("execution_candidate_id", "")
            )
            with self._lock:
                current_candidate = str(self._scene.get("execution_candidate_id", ""))
            if current_candidate != next_candidate:
                scene = dual_robot_planning_scene_payload(task) if task is not None else dict(scene_payload)
        with self._lock:
            if scene is not None:
                self._scene = scene
                self._state["scene_revision"] = int(self._state["scene_revision"]) + 1
                self._state["pair_id"] = str(scene["pair_id"])
                self._state["transition_id"] = str(scene["transition_id"])
                self._state["candidate_counts"] = dict(scene.get("candidate_filter_diagnostics", {}))
            if candidate_counts is not None:
                merged_counts = dict(self._state["candidate_counts"])
                merged_counts.update(candidate_counts)
                self._state["candidate_counts"] = merged_counts
            if attempt_index is not None:
                self._state["attempt_index"] = int(attempt_index)
            if attempt_total is not None:
                self._state["attempt_total"] = int(attempt_total)
            self._state["phase"] = str(phase)
            self._state["status"] = str(status)
            self._state["message"] = str(message)
            if record_event:
                history = list(self._state["history"])
                history.append(
                    {
                        "attempt_index": self._state["attempt_index"],
                        "phase": str(phase),
                        "status": str(status),
                        "message": str(message),
                    }
                )
                self._state["history"] = history[-24:]
            self._state["state_revision"] = int(self._state["state_revision"]) + 1

    def snapshot(self) -> tuple[dict[str, object], dict[str, object]]:
        """Return copies of the live status and scene for tests/diagnostics."""

        with self._lock:
            return dict(self._state), dict(self._scene)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


__all__ = [
    "DualRobotPlanningDebugServer",
    "dual_robot_planning_scene_payload",
    "dual_robot_planning_scene_payload_from_plan",
]
