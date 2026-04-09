"""Run a saved Fabrica grasp bundle through an Isaac pickup attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Load saved grasps, sample a part placement, and execute a pickup in Isaac."
)
parser.add_argument("--input-json", type=Path, required=True, help="Input grasp bundle from stage 1.")
parser.add_argument("--part-usd", type=str, required=True, help="USD asset used to spawn the target part in Isaac.")
parser.add_argument("--fr3-usd", type=str, default="", help="Optional override for the Franka Research 3 USD path.")
parser.add_argument(
    "--controller",
    type=str,
    default="admittance",
    choices=("planner", "admittance"),
    help="Execution controller: conservative joint-space planner or Isaac-side admittance controller.",
)
parser.add_argument("--pregrasp-offset", type=float, default=0.20, help="Pregrasp offset in meters.")
parser.add_argument(
    "--gripper-width-clearance",
    type=float,
    default=0.01,
    help="Clearance added to the saved grasp jaw width for the open approach width.",
)
parser.add_argument("--close-width", type=float, default=0.0, help="Finger joint target width for close.")
parser.add_argument(
    "--tcp-to-grasp-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, -0.045),
    metavar=("X", "Y", "Z"),
    help="Override the fixed TCP-to-grasp-center offset used for pose conversion and pickup execution.",
)
parser.add_argument("--support-face", type=str, default="", help="Optional explicit support face.")
parser.add_argument("--yaw-deg", type=float, default=None, help="Optional explicit pickup yaw in degrees.")
parser.add_argument("--xy-world", type=str, default="", help="Optional explicit world XY as x,y.")
parser.add_argument("--random-support-face", action="store_true", help="Sample support face from allowed set.")
parser.add_argument("--random-yaw", action="store_true", help="Sample yaw from allowed set.")
parser.add_argument(
    "--allowed-support-faces",
    type=str,
    default="pos_x,neg_x,pos_y,neg_y,neg_z",
    help="Comma-separated support faces used by the random sampler.",
)
parser.add_argument(
    "--allowed-yaw-deg",
    type=str,
    default="0,90,180,270",
    help="Comma-separated yaw values used by the random sampler.",
)
parser.add_argument("--xy-min-world", type=str, default="-0.45,-0.05", help="Random placement XY lower bound.")
parser.add_argument("--xy-max-world", type=str, default="-0.35,0.05", help="Random placement XY upper bound.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument(
    "--detailed-finger-contact-gap-m",
    type=float,
    default=0.002,
    help="Detailed Franka finger contact gap used during the ground recheck.",
)
parser.add_argument("--pregrasp-only", action="store_true", help="Stop after reaching pregrasp.")
parser.add_argument(
    "--run-seconds",
    type=float,
    default=0.0,
    help="Optional wall-clock duration to keep the simulation alive. Use 0 for until interrupted.",
)
parser.add_argument(
    "--attempt-artifact",
    type=Path,
    default=Path("artifacts/isaac_pick_attempt.json"),
    help="Optional JSON artifact for the selected attempt.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

from grasp_planning import (  # noqa: E402
    accepted_grasps,
    build_pickup_pose_world,
    evaluate_saved_grasps_against_pickup_pose,
    load_grasp_bundle,
    sample_pickup_placement_spec,
    saved_grasp_to_world_grasp,
    select_first_feasible_grasp,
)
from grasp_planning.controllers.fr3_pick_controller import FR3PickController  # noqa: E402
from grasp_planning.envs import make_fr3_part_scene_cfg  # noqa: E402
from grasp_planning.grasping.fabrica_grasp_debug import load_stl_mesh  # noqa: E402
from grasp_planning.planning.fr3_motion_context import FR3MotionContext  # noqa: E402
from grasp_planning.planning.pick_execution import (  # noqa: E402
    drive_robot_to_start_pose,
    execute_pick_from_world_grasp,
)
from grasp_planning.scene_defaults import ROBOT_BASE_ORIENTATION_XYZW, ROBOT_BASE_POSITION  # noqa: E402


def _parse_vec2(raw: str) -> tuple[float, float]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 2:
        raise ValueError(f"Expected exactly 2 comma-separated values, got '{raw}'.")
    return values


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one comma-separated float, got '{raw}'.")
    return values


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one comma-separated token, got '{raw}'.")
    return values


def resolve_fr3_usd_path() -> str:
    if args_cli.fr3_usd:
        return args_cli.fr3_usd
    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Unable to resolve Isaac asset root for the built-in FR3 asset.")
    return assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd"


def configure_grasp_tcp_calibration() -> None:
    tcp_to_grasp_offset = tuple(float(value) for value in args_cli.tcp_to_grasp_offset)
    FR3MotionContext._TCP_TO_GRASP_CENTER_OFFSET = tcp_to_grasp_offset
    FR3PickController._TCP_TO_GRASP_CENTER_OFFSET = tcp_to_grasp_offset


def _mesh_in_bundle_local_frame(bundle) -> object:
    mesh_global = load_stl_mesh(bundle.target_stl_path, scale=bundle.stl_scale)
    rotation = bundle.local_frame_orientation_xyzw_world
    object_pose_world = type("BundlePose", (), {})()
    object_pose_world.rotation_world_from_object = None
    # Use the same row-vector convention as ObjectWorldPose.transform_points_to_world.
    from grasp_planning.grasping.fabrica_grasp_debug import TriangleMesh, quat_to_rotmat_xyzw

    rot = quat_to_rotmat_xyzw(rotation)
    translation = np.asarray(bundle.local_frame_origin_world, dtype=float)
    vertices_local = (np.asarray(mesh_global.vertices_obj, dtype=float) - translation[None, :]) @ rot
    return TriangleMesh(vertices_obj=vertices_local, faces=np.asarray(mesh_global.faces, dtype=np.int64))


def _resolve_placement_spec():
    explicit_xy = _parse_vec2(args_cli.xy_world) if args_cli.xy_world else None
    if args_cli.support_face and args_cli.yaw_deg is not None and explicit_xy is not None:
        from grasp_planning.grasping.fabrica_grasp_debug import PickupPlacementSpec

        return PickupPlacementSpec(
            support_face=args_cli.support_face,
            yaw_deg=float(args_cli.yaw_deg),
            xy_world=explicit_xy,
        )

    rng = np.random.default_rng(args_cli.seed)
    return sample_pickup_placement_spec(
        rng=rng,
        allowed_support_faces=_parse_str_tuple(args_cli.allowed_support_faces),
        allowed_yaw_deg=_parse_float_tuple(args_cli.allowed_yaw_deg),
        xy_min_world=_parse_vec2(args_cli.xy_min_world),
        xy_max_world=_parse_vec2(args_cli.xy_max_world),
    )


def build_scene(*, object_pose_world) -> tuple[sim_utils.SimulationContext, InteractiveScene]:
    print("[INFO]: Creating simulation context...", flush=True)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim._app_control_on_stop_handle = None
    sim._disable_app_control_on_stop_handle = True
    sim.set_camera_view([1.6, -1.2, 1.0], [0.35, 0.0, 0.3])

    print("[INFO]: Building FR3 + part scene config...", flush=True)
    scene_cfg = make_fr3_part_scene_cfg(
        fr3_asset_path=resolve_fr3_usd_path(),
        part_usd_path=args_cli.part_usd,
        part_position=object_pose_world.position_world,
        part_orientation_xyzw=object_pose_world.orientation_xyzw_world,
        robot_base_position=ROBOT_BASE_POSITION,
        robot_base_orientation_xyzw=ROBOT_BASE_ORIENTATION_XYZW,
    )
    print("[INFO]: Creating interactive scene...", flush=True)
    scene = InteractiveScene(scene_cfg)
    print("[INFO]: Waiting for stage assets to finish loading...", flush=True)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    print("[INFO]: Resetting simulator...", flush=True)
    sim.reset()
    print("[INFO]: Resetting scene buffers...", flush=True)
    scene.reset()
    print("[INFO]: Scene ready.", flush=True)
    return sim, scene


def _write_attempt_artifact(
    *,
    bundle,
    placement_spec,
    object_pose_world,
    statuses,
    selected_world_grasp,
    execution_result,
) -> None:
    artifact = {
        "input_json": str(args_cli.input_json),
        "target_stl_path": bundle.target_stl_path,
        "part_usd": args_cli.part_usd,
        "placement": {
            "support_face": placement_spec.support_face,
            "yaw_deg": placement_spec.yaw_deg,
            "xy_world": list(placement_spec.xy_world),
            "object_position_world": list(object_pose_world.position_world),
            "object_orientation_xyzw_world": list(object_pose_world.orientation_xyzw_world),
        },
        "counts": {
            "saved": len(bundle.candidates),
            "ground_feasible": len(accepted_grasps(statuses)),
        },
        "selected_grasp_id": None if selected_world_grasp is None else selected_world_grasp.grasp_id,
        "selected_world_grasp": None
        if selected_world_grasp is None
        else {
            "position_w": list(selected_world_grasp.position_w),
            "orientation_xyzw": list(selected_world_grasp.orientation_xyzw),
            "normal_w": list(selected_world_grasp.normal_w),
            "pregrasp_position_w": list(selected_world_grasp.pregrasp_position_w),
            "jaw_width": selected_world_grasp.jaw_width,
            "gripper_width": selected_world_grasp.gripper_width,
        },
        "execution": {
            "success": execution_result.success,
            "status": execution_result.status,
            "message": execution_result.message,
        },
    }
    output = args_cli.attempt_artifact
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def run() -> None:
    configure_grasp_tcp_calibration()
    print("[INFO]: Loading grasp bundle...", flush=True)
    bundle = load_grasp_bundle(args_cli.input_json)
    print(f"[INFO]: Loaded {len(bundle.candidates)} saved grasps from '{args_cli.input_json}'.", flush=True)
    print("[INFO]: Loading local STL mesh for placement and filtering...", flush=True)
    mesh_local = _mesh_in_bundle_local_frame(bundle)
    print("[INFO]: Resolving pickup placement...", flush=True)
    placement_spec = _resolve_placement_spec()
    object_pose_world = build_pickup_pose_world(
        mesh_local,
        support_face=placement_spec.support_face,
        yaw_deg=placement_spec.yaw_deg,
        xy_world=placement_spec.xy_world,
    )
    print(
        "[INFO]: Pickup placement "
        f"support_face={placement_spec.support_face} yaw_deg={placement_spec.yaw_deg:.1f} "
        f"xy_world={placement_spec.xy_world} object_pose_world={object_pose_world.position_world} "
        f"orientation_xyzw={object_pose_world.orientation_xyzw_world}",
        flush=True,
    )

    print("[INFO]: Rechecking saved grasps against the selected pickup pose...", flush=True)
    statuses = evaluate_saved_grasps_against_pickup_pose(
        bundle.candidates,
        object_pose_world=object_pose_world,
        contact_gap_m=args_cli.detailed_finger_contact_gap_m,
    )
    feasible_grasps = accepted_grasps(statuses)
    selected_grasp = select_first_feasible_grasp(statuses)
    print(
        f"[INFO]: Ground recheck complete. feasible={len(feasible_grasps)} / saved={len(bundle.candidates)}",
        flush=True,
    )
    if selected_grasp is None:
        result = type(
            "ExecutionResult",
            (),
            {
                "success": False,
                "status": "no_feasible_grasp",
                "message": "No saved grasp survives the pickup-ground recheck for the sampled placement.",
            },
        )()
        _write_attempt_artifact(
            bundle=bundle,
            placement_spec=placement_spec,
            object_pose_world=object_pose_world,
            statuses=statuses,
            selected_world_grasp=None,
            execution_result=result,
        )
        raise RuntimeError(result.message)
    selected_world_grasp = saved_grasp_to_world_grasp(
        selected_grasp,
        object_pose_world,
        pregrasp_offset=args_cli.pregrasp_offset,
        gripper_width_clearance=args_cli.gripper_width_clearance,
    )
    print(
        "[INFO]: Selected grasp "
        f"id={selected_grasp.grasp_id} grasp_w={selected_world_grasp.position_w} "
        f"pregrasp_w={selected_world_grasp.pregrasp_position_w} "
        f"orientation_xyzw={selected_world_grasp.orientation_xyzw} "
        f"gripper_width={selected_world_grasp.gripper_width:.4f}",
        flush=True,
    )

    sim, scene = build_scene(object_pose_world=object_pose_world)
    physics_dt = sim.get_physics_dt()
    print("[INFO]: Warming up simulation...", flush=True)
    for _ in range(max(1, int(0.1 / physics_dt))):
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)
    print("[INFO]: Driving robot to start pose...", flush=True)
    drive_robot_to_start_pose(sim, scene)

    robot = scene["robot"]
    part = scene["part"]
    fixed_gripper_width = selected_world_grasp.gripper_width / 2.0
    print("[INFO]: Executing pick attempt...", flush=True)
    execution_result = execute_pick_from_world_grasp(
        sim=sim,
        scene=scene,
        robot=robot,
        object_asset=part,
        world_grasp=selected_world_grasp,
        controller_type=args_cli.controller,
        fixed_gripper_width=fixed_gripper_width,
        closed_gripper_width=float(args_cli.close_width),
        pregrasp_only=bool(args_cli.pregrasp_only),
    )
    _write_attempt_artifact(
        bundle=bundle,
        placement_spec=placement_spec,
        object_pose_world=object_pose_world,
        statuses=statuses,
        selected_world_grasp=selected_world_grasp,
        execution_result=execution_result,
    )

    print(
        "[INFO]: Fabrica Isaac pickup attempt "
        f"support_face={placement_spec.support_face} yaw_deg={placement_spec.yaw_deg:.1f} "
        f"saved={len(bundle.candidates)} feasible={len(feasible_grasps)} "
        f"selected={selected_grasp.grasp_id} status={execution_result.status} success={execution_result.success}",
        flush=True,
    )

    elapsed_s = 0.0
    while args_cli.run_seconds <= 0.0 or elapsed_s < args_cli.run_seconds:
        try:
            if sim.is_stopped():
                break
            if not sim.is_playing():
                sim.step()
                continue
            scene.write_data_to_sim()
            sim.step()
            scene.update(physics_dt)
            elapsed_s += physics_dt
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
