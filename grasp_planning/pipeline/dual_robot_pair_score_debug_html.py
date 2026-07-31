"""Movable-cell HTML debugger for frame-aware dual-grasp pair ranking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspCandidate,
    load_grasp_bundle,
    quat_to_rotmat_xyzw,
    rotmat_to_quat_xyzw,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose

from .assembly_sequence import AssemblySequence
from .assembly_sequence_debug_html import assembly_sequence_visual_payload
from .dual_robot_pair_scoring import (
    MovableFrame,
    ReachabilityProxyConfig,
)
from .holder_state_debug_html import _gripper_payload


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Stage-3 artifact does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in '{path}'.")
    return payload


def _source_pose(raw: dict[str, object]) -> ObjectWorldPose:
    return ObjectWorldPose(
        position_world=tuple(
            float(value)
            for value in raw["position"]  # type: ignore[arg-type]
        ),
        orientation_xyzw_world=tuple(
            float(value)
            for value in raw["orientation_xyzw"]  # type: ignore[arg-type]
        ),
    )


def _pose_payload(
    *,
    position_source: tuple[float, float, float],
    orientation_source: tuple[float, float, float, float],
    source_pose_assembly: ObjectWorldPose,
) -> dict[str, object]:
    rotation = source_pose_assembly.rotation_world_from_object @ quat_to_rotmat_xyzw(orientation_source)
    position = (
        source_pose_assembly.rotation_world_from_object @ np.asarray(position_source, dtype=float)
        + source_pose_assembly.translation_world
    )
    return {
        "position_assembly_m": np.round(position, 9).tolist(),
        "orientation_xyzw_assembly": list(rotmat_to_quat_xyzw(rotation)),
    }


def _saved_candidate_payload(
    candidate: SavedGraspCandidate,
    *,
    source_pose_assembly: ObjectWorldPose,
) -> dict[str, object]:
    return {
        **_pose_payload(
            position_source=candidate.grasp_position_obj,
            orientation_source=candidate.grasp_orientation_xyzw_obj,
            source_pose_assembly=source_pose_assembly,
        ),
        "position_source_m": list(candidate.grasp_position_obj),
        "orientation_xyzw_source": list(candidate.grasp_orientation_xyzw_obj),
        "jaw_width_m": candidate.jaw_width,
        "grasp_score": candidate.score,
    }


def _holder_candidate_payload(
    raw: dict[str, object],
    *,
    source_pose_assembly: ObjectWorldPose,
) -> dict[str, object]:
    grasp_pose = dict(raw["grasp_pose_obj"])  # type: ignore[arg-type]
    position = tuple(
        float(value)
        for value in grasp_pose["position"]  # type: ignore[arg-type]
    )
    orientation = tuple(
        float(value)
        for value in grasp_pose["orientation_xyzw"]  # type: ignore[arg-type]
    )
    return {
        **_pose_payload(
            position_source=position,
            orientation_source=orientation,
            source_pose_assembly=source_pose_assembly,
        ),
        "jaw_width_m": float(raw["jaw_width"]),
        "grasp_score": raw.get("score"),
    }


def build_dual_robot_pair_score_debug_payload(
    *,
    sequence: AssemblySequence,
    artifact_dir: str | Path,
    holder_base_world: MovableFrame,
    inserter_base_world: MovableFrame,
    assembly_world: MovableFrame,
    robot_separation_y_m: float,
    lock_robot_separation: bool,
    scoring: ReachabilityProxyConfig,
) -> dict[str, object]:
    """Load Stage-3 references and produce a compact interactive payload."""

    artifact_root = Path(artifact_dir)
    summary = _read_json(artifact_root / "dual_grasp_pair_summary.json")
    if str(summary.get("assembly")) != sequence.assembly:
        raise ValueError("Pair summary assembly does not match the compiled sequence.")
    holder_payload = _read_json(artifact_root / "holder_state_feasibility.json")
    holder_source = _source_pose(
        dict(holder_payload["source_frame_pose_assembly"])  # type: ignore[arg-type]
    )
    holder_candidates_raw = dict(holder_payload["candidates"])  # type: ignore[arg-type]
    holder_pregrasp_offset_m = float(
        dict(holder_payload["configuration"])["pregrasp_offset_m"]  # type: ignore[arg-type]
    )
    sequence_steps = {step.step_id: step for step in sequence.steps}
    visual = assembly_sequence_visual_payload(
        sequence,
        max_edges_per_part=220,
        max_faces_per_part=1000,
    )
    steps = []
    all_holder_ids: set[str] = set()
    first_pickup_position_assembly: list[float] | None = None
    for summary_step_raw in summary["steps"]:  # type: ignore[index]
        summary_step = dict(summary_step_raw)
        step_id = str(summary_step["step_id"])
        step = sequence_steps[step_id]
        pair_payload = _read_json(artifact_root / str(summary_step["pair_artifact"]))
        inserter_source_raw = dict(summary_step["inserter_source"])
        inserter_bundle = load_grasp_bundle(artifact_root / str(inserter_source_raw["artifact"]))
        inserter_source_pose = ObjectWorldPose(
            position_world=inserter_bundle.source_frame_origin_obj_world,
            orientation_xyzw_world=(inserter_bundle.source_frame_orientation_xyzw_obj_world),
        )
        if first_pickup_position_assembly is None:
            first_pickup_position_assembly = list(inserter_source_pose.position_world)
        inserter_by_id = {candidate.grasp_id: candidate for candidate in inserter_bundle.candidates}
        retained_ids = set(pair_payload["retained_pair_ids"])  # type: ignore[arg-type]
        compatible = [
            dict(evaluation)
            for evaluation in pair_payload["evaluations"]  # type: ignore[index]
            if dict(evaluation).get("status") == "accepted"
        ]
        holder_ids = {str(evaluation["holder_grasp_id"]) for evaluation in compatible}
        inserter_ids = {str(evaluation["inserter_grasp_id"]) for evaluation in compatible}
        all_holder_ids.update(holder_ids)
        missing_inserters = inserter_ids - set(inserter_by_id)
        if missing_inserters:
            raise ValueError(
                f"Pair artifact '{step_id}' references missing inserter IDs: {sorted(missing_inserters)[:5]}."
            )
        incoming_visual = visual["visualization"]["parts"][  # type: ignore[index]
            step.incoming_part_id
        ]
        source_rotation = inserter_source_pose.rotation_world_from_object
        source_translation = inserter_source_pose.translation_world
        vertices_assembly = np.asarray(
            incoming_visual["vertices_assembly_m"],
            dtype=float,
        )
        vertices_source = (vertices_assembly - source_translation[None, :]) @ source_rotation
        steps.append(
            {
                "step_id": step_id,
                "step_index": step.step_index,
                "incoming_part_id": step.incoming_part_id,
                "assembled_part_ids_before": list(step.assembled_part_ids_before),
                "final_to_pre_translation_assembly_m": list(step.final_to_pre_insertion_translation_m),
                "retreat_translation_assembly_m": list(
                    dict(pair_payload["motion"])["retreat_translation_end_m"]  # type: ignore[arg-type]
                ),
                "inserter_source_pose_assembly": {
                    "position": list(inserter_source_pose.position_world),
                    "orientation_xyzw": list(inserter_source_pose.orientation_xyzw_world),
                },
                "incoming_mesh_source": {
                    "vertices_m": np.round(vertices_source, 6).tolist(),
                    "faces": incoming_visual["faces"],
                    "edges": incoming_visual["edges"],
                },
                "pairs": [
                    {
                        "pair_id": str(evaluation["pair_id"]),
                        "holder_grasp_id": str(evaluation["holder_grasp_id"]),
                        "inserter_grasp_id": str(evaluation["inserter_grasp_id"]),
                        "offline_score": float(evaluation["score"]),
                        "minimum_clearance_m": evaluation.get("minimum_clearance_m"),
                        "collision_check": evaluation.get("collision_check"),
                        "retained": (str(evaluation["pair_id"]) in retained_ids),
                    }
                    for evaluation in compatible
                ],
                "inserters": {
                    grasp_id: _saved_candidate_payload(
                        inserter_by_id[grasp_id],
                        source_pose_assembly=inserter_source_pose,
                    )
                    for grasp_id in sorted(inserter_ids)
                },
            }
        )

    missing_holders = all_holder_ids - set(holder_candidates_raw)
    if missing_holders:
        raise ValueError(f"Pair artifacts reference missing holder IDs: {sorted(missing_holders)[:5]}.")
    pickup_position = [0.0, 0.0, 0.0] if first_pickup_position_assembly is None else first_pickup_position_assembly
    assembly_yaw_rad = np.radians(assembly_world.yaw_deg)
    cosine = float(np.cos(assembly_yaw_rad))
    sine = float(np.sin(assembly_yaw_rad))
    pickup_position_world = np.asarray(assembly_world.position_world_m, dtype=float) + np.asarray(
        [
            cosine * pickup_position[0] - sine * pickup_position[1],
            sine * pickup_position[0] + cosine * pickup_position[1],
            pickup_position[2],
        ],
        dtype=float,
    )
    return {
        "schema_version": 1,
        "kind": "dual_robot_pair_score_debug",
        "assembly": sequence.assembly,
        "base_part_id": sequence.base_part_id,
        "selected_order": list(sequence.selected_order),
        "scope_warning": (
            "Reachability proxy for ranking only; not IK, arm collision, trajectory, or execution feasibility."
        ),
        "robot_layout_assumption": (
            "KUKA base +X is treated as forward. Defaults place the robots "
            "side by side, 0.840 m apart in world Y, both facing world +X."
        ),
        "initial_layout": {
            "holder_base": {
                **holder_base_world.to_payload(),
                "robot_id": "robot_1",
                "initial_side": "negative_y",
            },
            "inserter_base": {
                **inserter_base_world.to_payload(),
                "robot_id": "robot_2",
                "initial_side": "positive_y",
            },
            "assembly": assembly_world.to_payload(),
            "pickup_source": MovableFrame(
                position_world_m=tuple(float(value) for value in pickup_position_world),
                yaw_deg=assembly_world.yaw_deg,
            ).to_payload(),
            "robot_separation_y_m": float(robot_separation_y_m),
            "lock_robot_separation": bool(lock_robot_separation),
            "include_pickup_in_score": False,
            "retained_pairs_only": True,
            "auto_select_top_pair": True,
        },
        "scoring": scoring.to_payload(),
        "holder_pregrasp_offset_m": holder_pregrasp_offset_m,
        "visualization": visual["visualization"],
        "gripper": _gripper_payload(),
        "holders": {
            grasp_id: _holder_candidate_payload(
                dict(holder_candidates_raw[grasp_id]),  # type: ignore[arg-type]
                source_pose_assembly=holder_source,
            )
            for grasp_id in sorted(all_holder_ids)
        },
        "steps": steps,
    }


def write_dual_robot_pair_score_debug_html(
    payload: dict[str, object],
    output_path: str | Path,
) -> None:
    data_json = json.dumps(payload, separators=(",", ":")).replace(
        "</",
        "<\\/",
    )
    html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dual-Robot Pair Reachability Ranking</title>
<style>
:root{--bg:#ede9e0;--panel:#fffaf1;--ink:#25231f;--muted:#706a60;--line:#d7cbb7;--holder:#9b6622;--inserter:#684bb0;--assembly:#25816b;--pickup:#dc7a20;--good:#168653;--bad:#c84135;--accent:#2879aa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Segoe UI",sans-serif}.layout{display:grid;grid-template-columns:430px minmax(0,1fr);min-height:100vh}
aside{padding:17px;background:var(--panel);border-right:1px solid var(--line);overflow:auto}main{padding:15px;min-width:0}h1{font-size:24px;margin:0 0 5px}h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;margin:15px 0 7px}
.subtitle,.hint{font-size:12px;line-height:1.4;color:var(--muted)}select,input,button,textarea{border:1px solid var(--line);border-radius:8px;background:#fff;padding:7px;font:inherit;width:100%}
.frame{display:grid;grid-template-columns:58px repeat(4,1fr);gap:5px;align-items:center;margin-bottom:6px}.frame label{font-size:11px;font-weight:700}.frame span{font-size:10px;color:var(--muted);text-align:center}
.checks{display:grid;gap:6px}.check{display:flex;align-items:center;gap:7px;font-size:12px}.check input{width:auto}.range{display:grid;grid-template-columns:84px 1fr 38px;gap:7px;align-items:center;margin:6px 0}.range label{font-size:11px;color:var(--muted)}.range input{padding:0}
.buttons{display:grid;grid-template-columns:1fr 1fr;gap:7px}.card{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:12px}.top{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(390px,.75fr);gap:14px}
canvas{width:100%;display:block;border-radius:11px;background:linear-gradient(#fff,#e9e3d8)}#cell{aspect-ratio:1.35/1;cursor:grab;touch-action:none}#graspScene{aspect-ratio:1.35/1;cursor:grab;touch-action:none}
table{border-collapse:collapse;width:100%;font-size:11px}th,td{text-align:right;padding:7px 6px;border-bottom:1px solid #e7dfd2}th:first-child,td:first-child{text-align:left}tbody tr{cursor:pointer}tbody tr:hover,tbody tr.active{background:#fff0cf}.score{font-weight:800;color:var(--good)}
pre{white-space:pre-wrap;margin:0;font:11px/1.43 ui-monospace,"SFMono-Regular",monospace}.legend{display:flex;flex-wrap:wrap;gap:9px;color:var(--muted);font-size:11px;margin-top:8px}.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:4px}
textarea{height:96px;font:10px/1.3 ui-monospace,monospace;margin-top:7px}.warning{background:#fff1dc;border:1px solid #e9b779;border-radius:9px;padding:8px;font-size:11px;margin-top:9px}.roles{display:grid;grid-template-columns:1fr 1fr;gap:7px}.role{border:1px solid var(--line);border-radius:9px;padding:8px;font-size:11px}.role b,.role span{display:block}.role span{color:var(--muted);margin-top:3px}.role.holder{border-left:6px solid var(--holder)}.role.inserter{border-left:6px solid var(--inserter)}
@media(max-width:1120px){.layout{grid-template-columns:1fr}.top{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body><div class="layout"><aside>
<h1>Pair Reachability Ranking</h1><div id="subtitle" class="subtitle"></div><div class="warning" id="warning"></div>
<h2>Assembly step</h2><select id="step"></select>
<h2>Robot role assignment</h2><select id="holderRobot"><option value="robot_1">Robot 1 (initial −Y) is holder</option><option value="robot_2">Robot 2 (initial +Y) is holder</option></select>
<div class="hint">The other physical robot automatically becomes the inserter.</div>
<h2>Movable frames (m / deg)</h2>
<div class="frame"><span></span><span>X</span><span>Y</span><span>Z</span><span>Yaw</span></div>
<div id="frames"></div><div class="hint">Drag H, I, A, or P in the top-down canvas. Yaws remain numeric controls.</div>
<h2>Options</h2><div class="checks">
<label class="check"><input id="lock" type="checkbox"> Lock robot offset to +<span id="separation"></span> m in Y</label>
<label class="check"><input id="pickupScore" type="checkbox"> Include movable pickup pose in inserter score</label>
<label class="check"><input id="retainedOnly" type="checkbox"> Rank Stage-3 retained pairs only</label>
<label class="check"><input id="autoTop" type="checkbox"> Auto-follow the highest-scoring grasp pair</label></div>
<h2>Combined score weights</h2>
<div class="range"><label>Stage 3</label><input id="wOffline" type="range" min="0" max="1" step=".01"><output id="wOfflineOut"></output></div>
<div class="range"><label>Reachability</label><input id="wReach" type="range" min="0" max="1" step=".01"><output id="wReachOut"></output></div>
<div class="range"><label>Layout</label><input id="wLayout" type="range" min="0" max="1" step=".01"><output id="wLayoutOut"></output></div>
<h2>View</h2><div class="range"><label>Half-width</label><input id="view" type="range" min=".5" max="2" step=".05" value="1.05"><output id="viewOut">1.05m</output></div>
<div class="buttons"><button id="reset">Reset</button><button id="copy">Copy layout JSON</button></div><textarea id="layoutJson" readonly></textarea>
</aside><main><div class="top"><section class="card"><h2>Cell layout · side by side · both face +X</h2><canvas id="cell" width="1200" height="880"></canvas>
<div class="legend"><span><i class="dot" style="background:var(--holder)"></i>holder H</span><span><i class="dot" style="background:var(--inserter)"></i>inserter I</span><span><i class="dot" style="background:var(--assembly)"></i>assembly A</span><span><i class="dot" style="background:var(--pickup)"></i>pickup P</span></div></section>
<section class="card"><h2>Robot roles</h2><div class="roles"><div class="role holder"><b>H · HOLDER ROBOT</b><span id="holderRole"></span></div><div class="role inserter"><b>I · INSERTER ROBOT</b><span id="inserterRole"></span></div></div>
<h2>Selected grasp geometry</h2><canvas id="graspScene" width="760" height="560"></canvas>
<div class="legend"><span><i class="dot" style="background:var(--holder)"></i>H holder gripper</span><span><i class="dot" style="background:var(--inserter)"></i>I inserter gripper</span><span><i class="dot" style="background:var(--pickup)"></i>incoming part</span></div>
<div class="hint">Drag to orbit; use the mouse wheel to zoom. Solid hands are final grasp poses; faint hands are approach/retreat poses.</div>
<div class="buttons" style="margin-top:8px"><button id="prevTop">← Previous top pair</button><button id="nextTop">Next top pair →</button></div><div class="hint" id="rankOut"></div>
<h2>Selected pair scores</h2><pre id="details"></pre></section></div>
<section class="card" style="margin-top:14px"><h2>Current ranking</h2><div class="hint" id="count"></div>
<table><thead><tr><th>Pair</th><th>Total</th><th>H holder</th><th>I inserter</th><th>Layout</th><th>Stage 3</th></tr></thead><tbody id="ranking"></tbody></table></section></main></div>
<script>
const data=__DATA__,$=id=>document.getElementById(id),canvas=$("cell"),ctx=canvas.getContext("2d"),graspCanvas=$("graspScene"),gctx=graspCanvas.getContext("2d"),initial=structuredClone(data.initial_layout),initialScoring=structuredClone(data.scoring);
const state={layout:structuredClone(initial),step:0,selected:null,ranked:[],view:1.05,drag:null,graspYaw:-.72,graspPitch:.5,graspZoom:1,graspDrag:false,graspLastX:0,graspLastY:0};
const frameKeys=["holder_base","inserter_base","assembly","pickup_source"],frameLabels={holder_base:"H holder",inserter_base:"I inserter",assembly:"A assembly",pickup_source:"P pickup"};
function robotName(frame){return frame.robot_id==="robot_1"?"Robot 1 (initial −Y)":"Robot 2 (initial +Y)"}
function clamp(v,a=0,b=1){return Math.max(a,Math.min(b,v))}function add(a,b){return a.map((v,i)=>v+b[i])}function sub(a,b){return a.map((v,i)=>v-b[i])}function mul(a,s){return a.map(v=>v*s)}
function rz(v,d){const a=d*Math.PI/180,c=Math.cos(a),s=Math.sin(a);return[c*v[0]-s*v[1],s*v[0]+c*v[1],v[2]]}function qmul(a,b){const[ax,ay,az,aw]=a,[bx,by,bz,bw]=b;return[aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw,aw*bw-ax*bx-ay*by-az*bz]}
function qyaw(d){const a=d*Math.PI/360;return[0,0,Math.sin(a),Math.cos(a)]}function qrot(v,q){const[x,y,z,w]=q,tx=2*(y*v[2]-z*v[1]),ty=2*(z*v[0]-x*v[2]),tz=2*(x*v[1]-y*v[0]);return[v[0]+w*tx+y*tz-z*ty,v[1]+w*ty+z*tx-x*tz,v[2]+w*tz+x*ty-y*tx]}
function worldPose(p,f){return{position:add(rz(p.position_assembly_m,f.yaw_deg),f.position_world_m),orientation:qmul(qyaw(f.yaw_deg),p.orientation_xyzw_assembly)}}
function sourceWorldPose(p,f){return{position:add(rz(p.position_source_m,f.yaw_deg),f.position_world_m),orientation:qmul(qyaw(f.yaw_deg),p.orientation_xyzw_source)}}
function shoulder(robot){return add(robot.position_world_m,rz(data.scoring.shoulder_offset_base_m,robot.yaw_deg))}
function triangle(v,lo,mid,hi){if(v<=lo||v>=hi)return 0;return v<=mid?(v-lo)/(mid-lo):(hi-v)/(hi-mid)}
function poseScore(target,robot){const C=data.scoring,b=robot.position_world_m,s=shoulder(robot),dv=sub(target.position,s),d=Math.hypot(...dv),rel=rz(sub(target.position,b),-robot.yaw_deg),dir=mul(dv,1/Math.max(d,1e-12)),approach=qrot([0,0,1],target.orientation),align=approach.reduce((z,v,i)=>z+v*dir[i],0),ds=triangle(d,C.minimum_reach_m,C.comfort_reach_m,C.maximum_reach_m),hs=triangle(rel[2],C.minimum_height_base_m,C.comfort_height_base_m,C.maximum_height_base_m),fs=clamp((rel[0]-C.front_zero_m)/(C.front_full_m-C.front_zero_m)),as=clamp((align+.2)/1.2),w=C.distance_weight+C.height_weight+C.front_weight+C.approach_weight,insideReach=d>C.minimum_reach_m&&d<C.maximum_reach_m,insideHeight=rel[2]>C.minimum_height_base_m&&rel[2]<C.maximum_height_base_m,raw=(C.distance_weight*ds+C.height_weight*hs+C.front_weight*fs+C.approach_weight*as)/w;return{score:insideReach&&insideHeight?raw:0,distance_score:ds,height_score:hs,front_score:fs,approach_score:as,distance_m:d,height_base_m:rel[2],front_base_m:rel[0],approach_alignment:align,inside_reach_shell:insideReach,inside_height_band:insideHeight}}
function armScore(targets,robot){const C=data.scoring,items=targets.map(t=>({name:t.name,...poseScore(t,robot)})),scores=items.map(x=>x.score),mn=Math.min(...scores),mean=scores.reduce((a,b)=>a+b,0)/scores.length;return{score:(C.target_min_weight*mn+C.target_mean_weight*mean)/(C.target_min_weight+C.target_mean_weight),minimum_target_score:mn,mean_target_score:mean,targets:items}}
function cross2(a,b,c,d){const o=(p,q,r)=>(q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]);return o(a,b,c)*o(a,b,d)<0&&o(c,d,a)*o(c,d,b)<0}
function targets(pair){const step=data.steps[state.step],A=state.layout.assembly,P=state.layout.pickup_source,hc=data.holders[pair.holder_grasp_id],ic=step.inserters[pair.inserter_grasp_id],hp={...worldPose(hc,A),jaw_width:hc.jaw_width_m},ip={...worldPose(ic,A),jaw_width:ic.jaw_width_m},ha=qrot([0,0,1],hp.orientation),hpre={name:"holder_pregrasp",position:add(hp.position,mul(ha,-data.holder_pregrasp_offset_m)),orientation:hp.orientation},holder=[hpre,{name:"holder_grasp",...hp}],pre={name:"inserter_pre",position:add(ip.position,rz(step.final_to_pre_translation_assembly_m,A.yaw_deg)),orientation:ip.orientation},ret={name:"inserter_retreat",position:add(ip.position,rz(step.retreat_translation_assembly_m,A.yaw_deg)),orientation:ip.orientation},ins=[pre,{name:"inserter_final",...ip},ret];if(state.layout.include_pickup_in_score){const pick=sourceWorldPose(ic,P);ins.unshift({name:"inserter_pickup",...pick})}return{holder,ins,hp,ip,hpre,pre,ret}}
function pairScore(pair){const C=data.scoring,T=targets(pair),H=armScore(T.holder,state.layout.holder_base),I=armScore(T.ins,state.layout.inserter_base),reach=(C.target_min_weight*Math.min(H.score,I.score)+C.target_mean_weight*.5*(H.score+I.score))/(C.target_min_weight+C.target_mean_weight),hs=shoulder(state.layout.holder_base),is=shoulder(state.layout.inserter_base),hd=Math.hypot(...sub(T.hp.position,hs)),ho=Math.hypot(...sub(T.hp.position,is)),id=Math.hypot(...sub(T.ip.position,is)),io=Math.hypot(...sub(T.ip.position,hs)),ownH=clamp(.5+(ho-hd)/(2*C.ownership_margin_m)),ownI=clamp(.5+(io-id)/(2*C.ownership_margin_m)),own=(C.target_min_weight*Math.min(ownH,ownI)+C.target_mean_weight*.5*(ownH+ownI))/(C.target_min_weight+C.target_mean_weight),cross=cross2(hs,T.hp.position,is,T.ip.position),non=cross?0:1,layout=(C.ownership_weight*own+C.noncrossing_weight*non)/(C.ownership_weight+C.noncrossing_weight),tw=C.offline_pair_weight+C.reachability_weight+C.layout_weight,total=tw>0?(C.offline_pair_weight*clamp(pair.offline_score)+C.reachability_weight*reach+C.layout_weight*layout)/tw:0;return{...pair,score:total,holder:H,inserter:I,reachability_score:reach,layout_score:layout,ownership_score:own,holder_ownership_score:ownH,inserter_ownership_score:ownI,segments_cross_xy:cross,targets:T}}
function rank(){const step=data.steps[state.step],pairs=state.layout.retained_pairs_only?step.pairs.filter(p=>p.retained):step.pairs;state.ranked=pairs.map(pairScore).sort((a,b)=>b.score-a.score||a.pair_id.localeCompare(b.pair_id));if(state.layout.auto_select_top_pair||!state.selected||!state.ranked.some(p=>p.pair_id===state.selected))state.selected=state.ranked[0]?.pair_id||null}
function selected(){return state.ranked.find(p=>p.pair_id===state.selected)||state.ranked[0]}
function manualSelect(pairId){state.layout.auto_select_top_pair=false;state.selected=pairId;syncInputs();render()}
function cycleTop(delta){const top=state.ranked.slice(0,30);if(!top.length)return;let index=top.findIndex(p=>p.pair_id===state.selected);if(index<0)index=0;index=(index+delta+top.length)%top.length;manualSelect(top[index].pair_id)}
function enforceRobotLock(changedKey){if(!state.layout.lock_robot_separation)return;const changed=state.layout[changedKey],otherKey=changedKey==="holder_base"?"inserter_base":"holder_base",other=state.layout[otherKey],s=state.layout.robot_separation_y_m;other.position_world_m[0]=changed.position_world_m[0];other.position_world_m[1]=changed.position_world_m[1]+(changed.robot_id==="robot_1"?s:-s)}
function assignHolderRobot(robotId){if(state.layout.holder_base.robot_id!==robotId){const previous=state.layout.holder_base;state.layout.holder_base=state.layout.inserter_base;state.layout.inserter_base=previous}syncInputs();render()}
function screen(p){const s=Math.min(canvas.width,canvas.height)/(2*state.view);return[canvas.width/2+p[0]*s,canvas.height/2-p[1]*s]}function world(p){const s=Math.min(canvas.width,canvas.height)/(2*state.view);return[(p[0]-canvas.width/2)/s,(canvas.height/2-p[1])/s]}
function line(a,b,color,w=2,dash=[]){const p=screen(a),q=screen(b);ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(...p);ctx.lineTo(...q);ctx.strokeStyle=color;ctx.lineWidth=w;ctx.stroke();ctx.setLineDash([])}
function circle(p,r,color,alpha=.15,w=1){const q=screen(p),s=Math.min(canvas.width,canvas.height)/(2*state.view);ctx.beginPath();ctx.arc(q[0],q[1],r*s,0,Math.PI*2);ctx.globalAlpha=alpha;ctx.fillStyle=color;ctx.fill();ctx.globalAlpha=1;ctx.strokeStyle=color;ctx.lineWidth=w;ctx.stroke()}
function drawMesh(vertices,faces,frame,color,alpha=.22){const v=vertices.map(p=>add(rz(p,frame.yaw_deg),frame.position_world_m)).map(screen);ctx.globalAlpha=alpha;ctx.fillStyle=color;ctx.strokeStyle=color;for(const f of faces){ctx.beginPath();ctx.moveTo(...v[f[0]]);f.slice(1).forEach(i=>ctx.lineTo(...v[i]));ctx.closePath();ctx.fill();ctx.stroke()}ctx.globalAlpha=1}
function marker(key,label,color){const f=state.layout[key],p=screen(f.position_world_m);ctx.beginPath();ctx.arc(p[0],p[1],13,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.stroke();ctx.fillStyle="#fff";ctx.font="700 12px sans-serif";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(label,p[0],p[1]);const d=rz([.10,0,0],f.yaw_deg);line(f.position_world_m,add(f.position_world_m,d),"#222",2)}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);const grid=.1;for(let x=-state.view;x<=state.view+.001;x+=grid)line([x,-state.view,0],[x,state.view,0],"#cfc8bb",x.toFixed(3)==="0.000"?2:.6);for(let y=-state.view;y<=state.view+.001;y+=grid)line([-state.view,y,0],[state.view,y,0],"#cfc8bb",y.toFixed(3)==="0.000"?2:.6);const H=state.layout.holder_base,I=state.layout.inserter_base,C=data.scoring;circle(shoulder(H),C.maximum_reach_m,"#9b6622",.04);circle(shoulder(H),C.comfort_reach_m,"#9b6622",.025);circle(shoulder(I),C.maximum_reach_m,"#684bb0",.04);circle(shoulder(I),C.comfort_reach_m,"#684bb0",.025);const step=data.steps[state.step],A=state.layout.assembly;for(const id of step.assembled_part_ids_before){const m=data.visualization.parts[id];drawMesh(m.vertices_assembly_m,m.faces,A,id===data.base_part_id?"#25816b":"#69746f")}{const m=data.visualization.parts[step.incoming_part_id];drawMesh(m.vertices_assembly_m,m.faces,A,"#dc7a20",.18)}drawMesh(step.incoming_mesh_source.vertices_m,step.incoming_mesh_source.faces,state.layout.pickup_source,"#dc7a20",.12);const p=selected();if(p){const T=p.targets,hs=shoulder(H),is=shoulder(I);line(hs,T.hp.position,p.holder.score>.5?"#168653":"#c84135",4);line(is,T.ip.position,p.inserter.score>.5?"#168653":"#c84135",4);T.holder.forEach(t=>circle(t.position,.012,"#9b6622",.8,2));T.ins.forEach(t=>circle(t.position,.012,"#684bb0",.8,2))}marker("holder_base","H","#9b6622");marker("inserter_base","I","#684bb0");marker("assembly","A","#25816b");marker("pickup_source","P","#dc7a20")}
function graspCamera(v){const center=add(state.layout.assembly.position_world_m,[0,0,.05]),p=sub(v,center),cy=Math.cos(state.graspYaw),sy=Math.sin(state.graspYaw),cp=Math.cos(state.graspPitch),sp=Math.sin(state.graspPitch),x=cy*p[0]-sy*p[1],y=sy*p[0]+cy*p[1],z=p[2];return[x,cp*y-sp*z,sp*y+cp*z]}
function graspProject(v){const p=graspCamera(v),extent=Math.max(data.visualization.scene_bounds_assembly_m.extent,.30),s=.76*Math.min(graspCanvas.width,graspCanvas.height)/extent*state.graspZoom;return[graspCanvas.width/2+p[0]*s,graspCanvas.height*.53-p[2]*s,p[1]]}
function graspLine(a,b,color,w=2,dash=[]){const p=graspProject(a),q=graspProject(b);gctx.beginPath();gctx.setLineDash(dash);gctx.moveTo(p[0],p[1]);gctx.lineTo(q[0],q[1]);gctx.strokeStyle=color;gctx.lineWidth=w;gctx.stroke();gctx.setLineDash([])}
function graspFaceRecords(vertices,faces,fill){return faces.map(f=>{const ps=f.map(i=>graspProject(vertices[i]));return{ps,d:ps.reduce((s,p)=>s+p[2],0)/ps.length,fill}})}
function graspDrawFaces(records,alpha=.64){records.sort((a,b)=>a.d-b.d);gctx.globalAlpha=alpha;records.forEach(r=>{gctx.beginPath();gctx.moveTo(r.ps[0][0],r.ps[0][1]);r.ps.slice(1).forEach(p=>gctx.lineTo(p[0],p[1]));gctx.closePath();gctx.fillStyle=r.fill;gctx.fill();gctx.strokeStyle="#342c2444";gctx.lineWidth=.3;gctx.stroke()});gctx.globalAlpha=1}
function graspPartVertices(id){const A=state.layout.assembly,m=data.visualization.parts[id];return m.vertices_assembly_m.map(p=>add(rz(p,A.yaw_deg),A.position_world_m))}
function graspDrawPart(id,color,alpha=.36){const m=data.visualization.parts[id],v=graspPartVertices(id);graspDrawFaces(graspFaceRecords(v,m.faces,color),alpha)}
function graspComponentWorld(comp,c,shift=0){const origin=sub(c.position,qrot(data.gripper.tcp_to_grasp_center_m,c.orientation));return comp.vertices.map(v=>add(origin,qrot([v[0],v[1]+shift,v[2]],c.orientation)))}
function graspDrawGripper(c,color,alpha=.72){const h=c.jaw_width/2,items=[[data.gripper.base,0],[data.gripper.left_finger,-h-data.gripper.left_fingertip_inner_y],[data.gripper.right_finger,h-data.gripper.right_fingertip_inner_y]],records=[];items.forEach(([comp,shift])=>records.push(...graspFaceRecords(graspComponentWorld(comp,c,shift),comp.faces,color)));graspDrawFaces(records,alpha)}
function graspLabel(position,textValue,color,dy){const p=graspProject(position);gctx.font="700 12px ui-monospace,monospace";const w=gctx.measureText(textValue).width+12,x=p[0]-w/2,y=p[1]+dy;gctx.globalAlpha=.88;gctx.fillStyle="#fffaf1";gctx.fillRect(x,y-13,w,18);gctx.globalAlpha=1;gctx.strokeStyle=color;gctx.strokeRect(x,y-13,w,18);gctx.fillStyle=color;gctx.textAlign="center";gctx.fillText(textValue,p[0],y)}
function renderGraspScene(){gctx.clearRect(0,0,graspCanvas.width,graspCanvas.height);const step=data.steps[state.step],A=state.layout.assembly,table=data.visualization.table_vertices_assembly_m.map(p=>add(rz(p,A.yaw_deg),A.position_world_m)),tableRecords=graspFaceRecords(table,[[0,1,2],[0,2,3]],"#2879aa");graspDrawFaces(tableRecords,.10);for(const id of step.assembled_part_ids_before)graspDrawPart(id,id===data.base_part_id?"#25816b":"#69746f",.42);graspDrawPart(step.incoming_part_id,"#dc7a20",.46);const p=selected();if(!p)return;const T=p.targets,hpre={...T.hpre,jaw_width:T.hp.jaw_width},ipre={...T.pre,jaw_width:T.ip.jaw_width},iret={...T.ret,jaw_width:T.ip.jaw_width};graspDrawGripper(hpre,"#9b6622",.13);graspDrawGripper(ipre,"#684bb0",.13);graspDrawGripper(iret,"#684bb0",.10);graspDrawGripper(T.hp,"#9b6622",.78);graspDrawGripper(T.ip,"#684bb0",.78);graspLine(T.hpre.position,T.hp.position,"#9b6622",2,[7,5]);graspLine(T.pre.position,T.ip.position,"#684bb0",2,[7,5]);graspLine(T.ip.position,T.ret.position,"#684bb0",2,[7,5]);graspLabel(T.hp.position,`H HOLDER · ${p.holder_grasp_id}`,"#7a4d17",-20);graspLabel(T.ip.position,`I INSERTER · ${p.inserter_grasp_id}`,"#553593",24)}
function renderRanking(){const top=state.ranked.slice(0,30),sel=selected(),step=data.steps[state.step],H=state.layout.holder_base,I=state.layout.inserter_base,selectedRank=sel?state.ranked.findIndex(p=>p.pair_id===sel.pair_id)+1:0;$("holderRole").textContent=`${robotName(H)} · base (${H.position_world_m.map(v=>v.toFixed(3)).join(", ")}) · yaw ${H.yaw_deg.toFixed(1)}° · holds base part ${data.base_part_id}`;$("inserterRole").textContent=`${robotName(I)} · base (${I.position_world_m.map(v=>v.toFixed(3)).join(", ")}) · yaw ${I.yaw_deg.toFixed(1)}° · inserts incoming part ${step.incoming_part_id}`;$("rankOut").textContent=sel?`Selected rank ${selectedRank}/${state.ranked.length}. Buttons cycle through the top ${top.length}; manual selection disables auto-follow.`:"No ranked pair selected.";$("count").textContent=`${state.ranked.length} compatible Stage-3 pairs ranked · showing top ${top.length}`;$("ranking").replaceChildren(...top.map((p,i)=>{const tr=document.createElement("tr");if(p.pair_id===state.selected)tr.className="active";tr.innerHTML=`<td>${i+1}. ${p.pair_id}${p.retained?" ★":""}</td><td class="score">${p.score.toFixed(3)}</td><td>${p.holder.score.toFixed(3)}</td><td>${p.inserter.score.toFixed(3)}</td><td>${p.layout_score.toFixed(3)}</td><td>${p.offline_score.toFixed(3)}</td>`;tr.onclick=()=>manualSelect(p.pair_id);return tr}));if(!sel){$("details").textContent="No pairs match the current filter.";return}$("details").textContent=[`pair:               ${sel.pair_id}`,"",`H HOLDER ROBOT`,`  physical robot:    ${robotName(H)}`,`  grasp ID:          ${sel.holder_grasp_id}`,`  role:              hold base/current assembly`,`  arm score:         ${sel.holder.score.toFixed(6)}`,`  min / mean:        ${sel.holder.minimum_target_score.toFixed(4)} / ${sel.holder.mean_target_score.toFixed(4)}`,"",`I INSERTER ROBOT`,`  physical robot:    ${robotName(I)}`,`  grasp ID:          ${sel.inserter_grasp_id}`,`  role:              insert incoming part ${step.incoming_part_id}`,`  arm score:         ${sel.inserter.score.toFixed(6)}`,`  min / mean:        ${sel.inserter.minimum_target_score.toFixed(4)} / ${sel.inserter.mean_target_score.toFixed(4)}`,"",`online score:       ${sel.score.toFixed(6)}`,`Stage-3 score:      ${sel.offline_score.toFixed(6)}`,`reachability:       ${sel.reachability_score.toFixed(6)}`,`layout:             ${sel.layout_score.toFixed(6)}`,`ownership:          ${sel.ownership_score.toFixed(6)}`,`segments cross XY:  ${sel.segments_cross_xy}`,`clearance Stage 3:  ${sel.minimum_clearance_m}`,`retained Stage 3:   ${sel.retained}`,"",...sel.holder.targets.map(t=>`H ${t.name.padEnd(18)} ${t.score.toFixed(3)}  d=${t.distance_m.toFixed(3)} front=${t.front_base_m.toFixed(3)} z=${t.height_base_m.toFixed(3)}`),...sel.inserter.targets.map(t=>`I ${t.name.padEnd(18)} ${t.score.toFixed(3)}  d=${t.distance_m.toFixed(3)} front=${t.front_base_m.toFixed(3)} z=${t.height_base_m.toFixed(3)}`)].join("\\n")}
function layoutExport(){return JSON.stringify({assembly:data.assembly,step_id:data.steps[state.step].step_id,holder_robot_id:state.layout.holder_base.robot_id,inserter_robot_id:state.layout.inserter_base.robot_id,holder_base:state.layout.holder_base,inserter_base:state.layout.inserter_base,assembly_frame:state.layout.assembly,pickup_source:state.layout.pickup_source,robot_separation_y_m:state.layout.robot_separation_y_m,lock_robot_separation:state.layout.lock_robot_separation,include_pickup_in_score:state.layout.include_pickup_in_score,auto_select_top_pair:state.layout.auto_select_top_pair,weights:{offline:data.scoring.offline_pair_weight,reachability:data.scoring.reachability_weight,layout:data.scoring.layout_weight}},null,2)}
function render(){rank();renderRanking();draw();renderGraspScene();$("layoutJson").value=layoutExport()}
function syncInputs(){for(const key of frameKeys){const f=state.layout[key],p=f.position_world_m;["x","y","z"].forEach((axis,i)=>$(`${key}_${axis}`).value=p[i].toFixed(3));$(`${key}_yaw`).value=f.yaw_deg.toFixed(1);$(`${key}_label`).textContent=key==="holder_base"?`H holder ${f.robot_id==="robot_1"?"R1":"R2"}`:key==="inserter_base"?`I inserter ${f.robot_id==="robot_1"?"R1":"R2"}`:frameLabels[key]}$("holderRobot").value=state.layout.holder_base.robot_id;$("lock").checked=state.layout.lock_robot_separation;$("pickupScore").checked=state.layout.include_pickup_in_score;$("retainedOnly").checked=state.layout.retained_pairs_only;$("autoTop").checked=state.layout.auto_select_top_pair}
function updateFrame(key,field,value){const f=state.layout[key];if(field==="yaw")f.yaw_deg=value;else f.position_world_m["xyz".indexOf(field)]=value;if(key==="holder_base"||key==="inserter_base")enforceRobotLock(key);syncInputs();render()}
for(const key of frameKeys){const row=document.createElement("div");row.className="frame";row.innerHTML=`<label id="${key}_label">${frameLabels[key]}</label>${["x","y","z","yaw"].map(a=>`<input id="${key}_${a}" type="number" step="${a==="yaw"?"1":"0.01"}">`).join("")}`;$("frames").appendChild(row);["x","y","z","yaw"].forEach(field=>$(`${key}_${field}`).onchange=e=>updateFrame(key,field,Number(e.target.value)))}
data.steps.forEach((s,i)=>{const o=document.createElement("option");o.value=i;o.textContent=`Step ${s.step_index}: incoming ${s.incoming_part_id}`;$("step").appendChild(o)});
$("step").onchange=e=>{state.step=Number(e.target.value);state.selected=null;render()};$("holderRobot").onchange=e=>assignHolderRobot(e.target.value);$("lock").onchange=e=>{state.layout.lock_robot_separation=e.target.checked;if(e.target.checked){enforceRobotLock("holder_base");syncInputs()}render()};$("pickupScore").onchange=e=>{state.layout.include_pickup_in_score=e.target.checked;render()};$("retainedOnly").onchange=e=>{state.layout.retained_pairs_only=e.target.checked;state.selected=null;render()};$("autoTop").onchange=e=>{state.layout.auto_select_top_pair=e.target.checked;if(e.target.checked)state.selected=null;render()};$("prevTop").onclick=()=>cycleTop(-1);$("nextTop").onclick=()=>cycleTop(1);
for(const [id,key] of [["wOffline","offline_pair_weight"],["wReach","reachability_weight"],["wLayout","layout_weight"]]){$(id).value=data.scoring[key];$(`${id}Out`).textContent=Number(data.scoring[key]).toFixed(2);$(id).oninput=e=>{data.scoring[key]=Number(e.target.value);$(`${id}Out`).textContent=Number(e.target.value).toFixed(2);render()}}
$("view").oninput=e=>{state.view=Number(e.target.value);$("viewOut").textContent=`${state.view.toFixed(2)}m`;draw()};$("reset").onclick=()=>{state.layout=structuredClone(initial);Object.assign(data.scoring,structuredClone(initialScoring));for(const [id,key] of [["wOffline","offline_pair_weight"],["wReach","reachability_weight"],["wLayout","layout_weight"]]){$(id).value=data.scoring[key];$(`${id}Out`).textContent=Number(data.scoring[key]).toFixed(2)}state.selected=null;syncInputs();render()};$("copy").onclick=async()=>{const value=layoutExport();$("layoutJson").value=value;try{await navigator.clipboard.writeText(value);$("copy").textContent="Copied";setTimeout(()=>$("copy").textContent="Copy layout JSON",1200)}catch{ $("layoutJson").select()}};
canvas.onpointerdown=e=>{const r=canvas.getBoundingClientRect(),p=[(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height],items=frameKeys.map(k=>[k,screen(state.layout[k].position_world_m)]).map(([k,q])=>[k,Math.hypot(p[0]-q[0],p[1]-q[1])]).sort((a,b)=>a[1]-b[1]);if(items[0][1]<28){state.drag=items[0][0];canvas.setPointerCapture(e.pointerId)}};canvas.onpointermove=e=>{if(!state.drag)return;const r=canvas.getBoundingClientRect(),p=[(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height],w=world(p),f=state.layout[state.drag];f.position_world_m[0]=w[0];f.position_world_m[1]=w[1];if(state.drag==="holder_base"||state.drag==="inserter_base")enforceRobotLock(state.drag);syncInputs();render()};canvas.onpointerup=canvas.onpointercancel=()=>state.drag=null;
graspCanvas.onpointerdown=e=>{state.graspDrag=true;state.graspLastX=e.clientX;state.graspLastY=e.clientY;graspCanvas.setPointerCapture(e.pointerId)};graspCanvas.onpointermove=e=>{if(!state.graspDrag)return;state.graspYaw+=(e.clientX-state.graspLastX)*.01;state.graspPitch=clamp(state.graspPitch-(e.clientY-state.graspLastY)*.01,-1.45,1.45);state.graspLastX=e.clientX;state.graspLastY=e.clientY;renderGraspScene()};graspCanvas.onpointerup=graspCanvas.onpointercancel=()=>state.graspDrag=false;graspCanvas.onwheel=e=>{e.preventDefault();state.graspZoom=clamp(state.graspZoom*(e.deltaY<0?1.08:1/1.08),.35,4);renderGraspScene()};graspCanvas.oncontextmenu=e=>e.preventDefault();
$("subtitle").textContent=`${data.assembly}: ${data.selected_order.join(" → ")} · base ${data.base_part_id}`;$("warning").textContent=data.scope_warning+" "+data.robot_layout_assumption;$("separation").textContent=Number(data.initial_layout.robot_separation_y_m).toFixed(3);syncInputs();render();
</script></body></html>""".replace("__DATA__", data_json)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


__all__ = [
    "build_dual_robot_pair_score_debug_payload",
    "write_dual_robot_pair_score_debug_html",
]
