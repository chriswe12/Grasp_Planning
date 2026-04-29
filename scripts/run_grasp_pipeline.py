"""Run the shared planning pipeline in sim, pitl, or real modes."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    DEFAULT_CONTACT_APPROACH_OFFSETS_M,
    DEFAULT_CONTACT_LATERAL_OFFSETS_M,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.pipeline import (  # noqa: E402
    ExecutionWorldPoseConfig,
    GeometryConfig,
    IsaacPipelineConfig,
    MujocoPipelineConfig,
    PickupPoseConfig,
    PlanningConfig,
    RealExecutionConfig,
    Ros2Config,
    generate_stage1_result,
    recheck_stage2_result,
    write_stage1_artifacts,
    write_stage2_artifacts,
)
from grasp_planning.ros2 import (  # noqa: E402
    execute_real_grasp_from_bundle,
    wait_for_debug_frame_pose_message,
)
from scripts.write_part_frame_debug_html import write_part_frame_debug_html  # noqa: E402

DEBUG_FRAME_MESSAGE_TYPE = "fp_debug_msgs/msg/DebugFrame"
BACKEND_CHOICES = ("config", "mujoco", "isaac", "both", "none")


def _tuple_floats(values: object, *, expected_len: int | None = None) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a list/tuple of floats, got {values!r}.")
    result = tuple(float(value) for value in values)
    if expected_len is not None and len(result) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(result)}.")
    return result


def _roll_angles_from_planning(raw: dict[str, object]) -> tuple[float, ...]:
    if raw.get("roll_angle_step_deg") not in ("", None):
        step_deg = float(raw["roll_angle_step_deg"])
        if step_deg <= 0.0 or step_deg > 360.0:
            raise ValueError("planning.roll_angle_step_deg must be > 0 and <= 360.")
        count = max(1, int(math.ceil(360.0 / step_deg)))
        return tuple(float(math.radians(index * step_deg)) for index in range(count) if index * step_deg < 360.0)
    return _tuple_floats(raw.get("roll_angles_rad", [0.0]))


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in '{path}'.")
    return payload


def _optional_float(raw: dict[str, object], key: str) -> float | None:
    value = raw.get(key)
    if value in ("", None):
        return None
    return float(value)


def _geometry_config(payload: dict[str, object]) -> GeometryConfig:
    raw = dict(payload.get("geometry", {}))
    return GeometryConfig(
        target_mesh_path=str(raw["target_mesh_path"]),
        mesh_scale=float(raw.get("mesh_scale", 1.0)),
        assembly_glob=None if raw.get("assembly_glob") in ("", None) else str(raw["assembly_glob"]),
    )


def _planning_config(payload: dict[str, object]) -> PlanningConfig:
    raw = dict(payload.get("planning", {}))
    return PlanningConfig(
        num_surface_samples=int(raw.get("num_surface_samples", 1024)),
        min_jaw_width=float(raw.get("min_jaw_width", 0.002)),
        max_jaw_width=float(raw.get("max_jaw_width", 0.09)),
        antipodal_cosine_threshold=float(raw.get("antipodal_cosine_threshold", 0.984807753012208)),
        roll_angles_rad=_roll_angles_from_planning(raw),
        max_pair_checks=int(raw.get("max_pair_checks", 40960)),
        detailed_finger_contact_gap_m=float(raw.get("detailed_finger_contact_gap_m", 0.002)),
        floor_clearance_margin_m=float(raw.get("floor_clearance_margin_m", 0.0)),
        skip_stage1_collision_checks=bool(raw.get("skip_stage1_collision_checks", False)),
        top_grasp_score_weight=float(raw.get("top_grasp_score_weight", 0.35)),
        contact_lateral_offsets_m=_tuple_floats(raw.get("contact_lateral_offsets_m", []))
        or DEFAULT_CONTACT_LATERAL_OFFSETS_M,
        contact_approach_offsets_m=_tuple_floats(raw.get("contact_approach_offsets_m", []))
        or DEFAULT_CONTACT_APPROACH_OFFSETS_M,
        rng_seed=int(raw.get("rng_seed", 0)),
    )


def _execution_pose_config(payload: dict[str, object]) -> ExecutionWorldPoseConfig:
    raw = dict(payload.get("execution_world_pose", {}))
    return ExecutionWorldPoseConfig(
        position_world=_tuple_floats(raw["position_world"], expected_len=3),
        orientation_xyzw_world=_tuple_floats(raw["orientation_xyzw_world"], expected_len=4),
    )


def _pickup_pose_config(payload: dict[str, object]) -> PickupPoseConfig | None:
    raw = payload.get("pickup_pose")
    if not isinstance(raw, dict):
        return None
    return PickupPoseConfig(
        support_face=str(raw["support_face"]),
        yaw_deg=float(raw["yaw_deg"]),
        xy_world=_tuple_floats(raw["xy_world"], expected_len=2),
    )


def _mujoco_execution_config(payload: dict[str, object]) -> MujocoPipelineConfig:
    raw = dict(payload.get("mujoco_execution", {}))
    controller = str(raw.get("controller", "native")).strip().lower()
    if controller not in {"native", "moveit"}:
        raise ValueError(f"Unsupported mujoco_execution.controller value '{controller}'.")
    return MujocoPipelineConfig(
        enabled=bool(raw.get("enabled", False)),
        python_executable=str(raw.get("python_executable", "")),
        robot_config=str(raw.get("robot_config", "")),
        simulation_config=str(raw.get("simulation_config", "")),
        controller=controller,
        grasp_id=str(raw.get("grasp_id", "")),
        pregrasp_offset=_optional_float(raw, "pregrasp_offset"),
        gripper_width_clearance=_optional_float(raw, "gripper_width_clearance"),
        contact_gap_m=_optional_float(raw, "contact_gap_m"),
        object_mass_kg=_optional_float(raw, "object_mass_kg"),
        object_scale=_optional_float(raw, "object_scale"),
        lift_height_m=_optional_float(raw, "lift_height_m"),
        success_height_margin_m=_optional_float(raw, "success_height_margin_m"),
        attempt_artifact=str(raw.get("attempt_artifact", "artifacts/mujoco_pick_attempt.json")),
        viewer=bool(raw.get("viewer", True)),
        viewer_left_ui=bool(raw.get("viewer_left_ui", False)),
        viewer_right_ui=bool(raw.get("viewer_right_ui", False)),
        viewer_no_realtime=bool(raw.get("viewer_no_realtime", False)),
        viewer_hold_seconds=float(raw.get("viewer_hold_seconds", 8.0)),
        viewer_block_at_end=bool(raw.get("viewer_block_at_end", False)),
        keep_generated_scene=bool(raw.get("keep_generated_scene", False)),
        moveit_frame_id=str(raw.get("moveit_frame_id", "base")),
        moveit_planning_group=str(raw.get("moveit_planning_group", "fr3_arm")),
        moveit_pose_link=str(raw.get("moveit_pose_link", "fr3_hand_tcp")),
        moveit_planner_id=str(raw.get("moveit_planner_id", "")),
        moveit_wait_for_moveit_timeout_s=float(raw.get("moveit_wait_for_moveit_timeout_s", 15.0)),
        moveit_ik_timeout_s=float(raw.get("moveit_ik_timeout_s", 2.0)),
        moveit_planning_time_s=float(raw.get("moveit_planning_time_s", 5.0)),
        moveit_num_planning_attempts=int(raw.get("moveit_num_planning_attempts", 5)),
        moveit_velocity_scale=float(raw.get("moveit_velocity_scale", 0.05)),
        moveit_acceleration_scale=float(raw.get("moveit_acceleration_scale", 0.05)),
        moveit_execute_timeout_s=float(raw.get("moveit_execute_timeout_s", 120.0)),
        moveit_allow_collisions=bool(raw.get("moveit_allow_collisions", False)),
    )


def _isaac_execution_config(payload: dict[str, object]) -> IsaacPipelineConfig:
    raw = dict(payload.get("isaac_execution", {}))
    tcp_to_grasp_offset = None
    if raw.get("tcp_to_grasp_offset") not in ("", None):
        tcp_to_grasp_offset = _tuple_floats(raw["tcp_to_grasp_offset"], expected_len=3)
    controller = str(raw.get("controller", "admittance")).strip().lower()
    if controller not in {"planner", "admittance"}:
        raise ValueError(f"Unsupported isaac_execution.controller value '{controller}'.")
    return IsaacPipelineConfig(
        enabled=bool(raw.get("enabled", False)),
        python_executable=str(raw.get("python_executable", "")),
        part_usd=str(raw.get("part_usd", "")),
        fr3_usd=str(raw.get("fr3_usd", "")),
        controller=controller,
        grasp_id=str(raw.get("grasp_id", "")),
        pregrasp_offset=_optional_float(raw, "pregrasp_offset"),
        gripper_width_clearance=_optional_float(raw, "gripper_width_clearance"),
        contact_gap_m=_optional_float(raw, "contact_gap_m"),
        close_width=float(raw.get("close_width", 0.0)),
        tcp_to_grasp_offset=tcp_to_grasp_offset,
        attempt_artifact=str(raw.get("attempt_artifact", "artifacts/isaac_pick_attempt.json")),
        pregrasp_only=bool(raw.get("pregrasp_only", False)),
        run_seconds=float(raw.get("run_seconds", 0.0)),
        headless=bool(raw.get("headless", False)),
    )


def _ros2_config(payload: dict[str, object]) -> Ros2Config:
    raw = dict(payload.get("ros2", {}))
    return Ros2Config(
        debug_frame_topic="" if raw.get("debug_frame_topic") in ("", None) else str(raw["debug_frame_topic"]),
        frame_id=str(raw.get("frame_id", "world")),
        timeout_s=float(raw.get("timeout_s", 10.0)),
        object_id=str(raw.get("object_id", "")),
    )


def _artifacts(payload: dict[str, object]) -> dict[str, Path]:
    raw = dict(payload.get("artifacts", {}))
    stage2_html = Path(str(raw["stage2_html"]))
    return {
        "stage1_json": Path(str(raw["stage1_json"])),
        "stage1_html": Path(str(raw["stage1_html"])),
        "stage2_json": Path(str(raw["stage2_json"])),
        "stage2_html": stage2_html,
        "part_frame_html": Path(
            str(raw.get("part_frame_html", stage2_html.with_name(f"{stage2_html.stem}_part_frame.html")))
        ),
    }


def _real_execution_config(payload: dict[str, object]) -> RealExecutionConfig:
    raw = dict(payload.get("real_execution", {}))
    stop_after = str(raw.get("stop_after", "pregrasp")).strip().lower()
    if stop_after not in {"pregrasp", "grasp", "lift", "full"}:
        raise ValueError(f"Unsupported real_execution.stop_after value '{stop_after}'.")
    return RealExecutionConfig(
        enabled=bool(raw.get("enabled", False)),
        grasp_id=str(raw.get("grasp_id", "")),
        attempt_artifact=str(raw.get("attempt_artifact", "artifacts/real_robot_pick_attempt.json")),
        planning_group=str(raw.get("planning_group", "fr3_arm")),
        pose_link=str(raw.get("pose_link", "fr3_hand_tcp")),
        frame_id=str(raw.get("frame_id", "base")),
        wait_for_moveit_timeout_s=float(raw.get("wait_for_moveit_timeout_s", 15.0)),
        ik_timeout_s=float(raw.get("ik_timeout_s", 2.0)),
        planning_time_s=float(raw.get("planning_time_s", 5.0)),
        num_planning_attempts=int(raw.get("num_planning_attempts", 5)),
        velocity_scale=float(raw.get("velocity_scale", 0.05)),
        acceleration_scale=float(raw.get("acceleration_scale", 0.05)),
        execute_timeout_s=float(raw.get("execute_timeout_s", 120.0)),
        post_execute_sleep_s=float(raw.get("post_execute_sleep_s", 0.5)),
        pregrasp_offset_m=float(raw.get("pregrasp_offset_m", 0.10)),
        gripper_width_clearance_m=float(raw.get("gripper_width_clearance_m", 0.01)),
        lift_height_m=float(raw.get("lift_height_m", 0.08)),
        require_confirmation=bool(raw.get("require_confirmation", True)),
        stop_after=stop_after,
        allow_collisions=bool(raw.get("allow_collisions", False)),
        gripper_enabled=bool(raw.get("gripper_enabled", False)),
        gripper_grasp_action=str(raw.get("gripper_grasp_action", "/fr3_gripper/grasp")),
        gripper_move_action=str(raw.get("gripper_move_action", "/fr3_gripper/move")),
        gripper_open_width=float(raw.get("gripper_open_width", 0.08)),
        gripper_grasp_speed=float(raw.get("gripper_grasp_speed", 0.03)),
        gripper_grasp_force=float(raw.get("gripper_grasp_force", 30.0)),
        gripper_epsilon_inner=float(raw.get("gripper_epsilon_inner", 0.002)),
        gripper_epsilon_outer=float(raw.get("gripper_epsilon_outer", 0.08)),
        gripper_timeout_s=float(raw.get("gripper_timeout_s", 10.0)),
        grasp_settle_time_s=float(raw.get("grasp_settle_time_s", 0.5)),
    )


def _format_topic(topic_template: str, *, object_id: str) -> str:
    if "{object_id}" in topic_template:
        if not object_id:
            raise ValueError(f"object_id is required to resolve ROS topic template '{topic_template}'.")
        return topic_template.format(object_id=object_id)
    return topic_template


def _resolve_object_pose_world(ros2: Ros2Config) -> ObjectWorldPose:
    if not ros2.debug_frame_topic:
        raise ValueError("ros2.debug_frame_topic must be non-empty for pitl and real modes.")
    if not ros2.object_id:
        raise ValueError("ros2.object_id must be non-empty for pitl and real modes.")

    debug_frame_topic = _format_topic(ros2.debug_frame_topic, object_id=ros2.object_id)
    print("[PIPELINE] Waiting for object pose on DebugFrame topic.", flush=True)
    return wait_for_debug_frame_pose_message(
        topic_name=debug_frame_topic,
        message_type=DEBUG_FRAME_MESSAGE_TYPE,
        object_id=ros2.object_id,
        timeout_s=ros2.timeout_s,
    )


def _normalize_mode(raw_mode: str) -> str:
    normalized = str(raw_mode).strip().lower()
    aliases = {
        "perception_in_the_loop": "pitl",
        "perception-in-the-loop": "pitl",
        "simulation": "sim",
    }
    return aliases.get(normalized, normalized)


def _effective_python_executable(raw_value: str) -> str:
    value = str(raw_value).strip()
    if value:
        return value
    if sys.executable:
        return sys.executable
    raise RuntimeError("Could not determine a Python executable for simulation execution.")


def _run_mujoco_execution(
    mujoco_execution: MujocoPipelineConfig,
    *,
    input_json: Path,
    headless: bool,
) -> None:
    if not mujoco_execution.enabled:
        return
    if not mujoco_execution.robot_config:
        raise ValueError("mujoco_execution.robot_config is required when MuJoCo execution is enabled.")
    command = [
        _effective_python_executable(mujoco_execution.python_executable),
        "scripts/run_fabrica_grasp_in_mujoco.py",
        "--input-json",
        str(input_json),
        "--robot-config",
        mujoco_execution.robot_config,
        "--attempt-artifact",
        mujoco_execution.attempt_artifact,
        "--controller",
        mujoco_execution.controller,
    ]
    if mujoco_execution.simulation_config:
        command.extend(["--simulation-config", mujoco_execution.simulation_config])
    if mujoco_execution.grasp_id:
        command.extend(["--grasp-id", mujoco_execution.grasp_id])
    if mujoco_execution.pregrasp_offset is not None:
        command.extend(["--pregrasp-offset", str(mujoco_execution.pregrasp_offset)])
    if mujoco_execution.gripper_width_clearance is not None:
        command.extend(["--gripper-width-clearance", str(mujoco_execution.gripper_width_clearance)])
    if mujoco_execution.contact_gap_m is not None:
        command.extend(["--contact-gap-m", str(mujoco_execution.contact_gap_m)])
    if mujoco_execution.object_mass_kg is not None:
        command.extend(["--object-mass-kg", str(mujoco_execution.object_mass_kg)])
    if mujoco_execution.object_scale is not None:
        command.extend(["--object-scale", str(mujoco_execution.object_scale)])
    if mujoco_execution.lift_height_m is not None:
        command.extend(["--lift-height-m", str(mujoco_execution.lift_height_m)])
    if mujoco_execution.success_height_margin_m is not None:
        command.extend(["--success-height-margin-m", str(mujoco_execution.success_height_margin_m)])
    if mujoco_execution.viewer and not headless:
        command.append("--viewer")
    if mujoco_execution.viewer_left_ui:
        command.append("--viewer-left-ui")
    if mujoco_execution.viewer_right_ui:
        command.append("--viewer-right-ui")
    if mujoco_execution.viewer_no_realtime:
        command.append("--viewer-no-realtime")
    if mujoco_execution.viewer_block_at_end:
        command.append("--viewer-block-at-end")
    if mujoco_execution.keep_generated_scene:
        command.append("--keep-generated-scene")
    if mujoco_execution.viewer_hold_seconds != 8.0:
        command.extend(["--viewer-hold-seconds", str(mujoco_execution.viewer_hold_seconds)])
    if mujoco_execution.controller == "moveit":
        command.extend(
            [
                "--moveit-frame-id",
                mujoco_execution.moveit_frame_id,
                "--moveit-planning-group",
                mujoco_execution.moveit_planning_group,
                "--moveit-pose-link",
                mujoco_execution.moveit_pose_link,
                "--moveit-planner-id",
                mujoco_execution.moveit_planner_id,
                "--moveit-wait-for-moveit-timeout-s",
                str(mujoco_execution.moveit_wait_for_moveit_timeout_s),
                "--moveit-ik-timeout-s",
                str(mujoco_execution.moveit_ik_timeout_s),
                "--moveit-planning-time-s",
                str(mujoco_execution.moveit_planning_time_s),
                "--moveit-num-planning-attempts",
                str(mujoco_execution.moveit_num_planning_attempts),
                "--moveit-velocity-scale",
                str(mujoco_execution.moveit_velocity_scale),
                "--moveit-acceleration-scale",
                str(mujoco_execution.moveit_acceleration_scale),
                "--moveit-execute-timeout-s",
                str(mujoco_execution.moveit_execute_timeout_s),
            ]
        )
        if mujoco_execution.moveit_allow_collisions:
            command.append("--moveit-allow-collisions")
    print("[PIPELINE] Starting MuJoCo execution.", flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def _run_isaac_execution(
    isaac_execution: IsaacPipelineConfig,
    *,
    input_json: Path,
    headless: bool,
) -> None:
    if not isaac_execution.enabled:
        return
    command = [
        _effective_python_executable(isaac_execution.python_executable),
        "scripts/run_fabrica_grasp_in_isaac.py",
        "--input-json",
        str(input_json),
        "--controller",
        isaac_execution.controller,
        "--attempt-artifact",
        isaac_execution.attempt_artifact,
        "--close-width",
        str(isaac_execution.close_width),
        "--run-seconds",
        str(isaac_execution.run_seconds),
    ]
    if isaac_execution.part_usd:
        command.extend(["--part-usd", isaac_execution.part_usd])
    if isaac_execution.fr3_usd:
        command.extend(["--fr3-usd", isaac_execution.fr3_usd])
    if isaac_execution.grasp_id:
        command.extend(["--grasp-id", isaac_execution.grasp_id])
    if isaac_execution.pregrasp_offset is not None:
        command.extend(["--pregrasp-offset", str(isaac_execution.pregrasp_offset)])
    if isaac_execution.gripper_width_clearance is not None:
        command.extend(["--gripper-width-clearance", str(isaac_execution.gripper_width_clearance)])
    if isaac_execution.contact_gap_m is not None:
        command.extend(["--detailed-finger-contact-gap-m", str(isaac_execution.contact_gap_m)])
    if isaac_execution.tcp_to_grasp_offset is not None:
        command.extend(["--tcp-to-grasp-offset", *(str(value) for value in isaac_execution.tcp_to_grasp_offset)])
    if isaac_execution.pregrasp_only:
        command.append("--pregrasp-only")
    if headless or isaac_execution.headless:
        command.append("--headless")
    print("[PIPELINE] Starting Isaac execution.", flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def _write_part_frame_debug_artifact(*, input_json: Path, output_html: Path) -> None:
    write_part_frame_debug_html(input_json=input_json, output_html=output_html)
    print(f"[PIPELINE] Wrote part frame debug HTML to {output_html}.", flush=True)


def _settle_object_pose_on_floor(
    object_pose_world: ObjectWorldPose | None,
    mesh_local: object,
    *,
    floor_z: float = 0.0,
) -> ObjectWorldPose | None:
    if object_pose_world is None:
        return None
    vertices_world = object_pose_world.transform_points_to_world(np.asarray(mesh_local.vertices_obj, dtype=float))
    min_z = float(vertices_world[:, 2].min())
    dz = float(floor_z) - min_z
    if abs(dz) <= 1.0e-8:
        return object_pose_world
    settled_pose = ObjectWorldPose(
        position_world=(
            float(object_pose_world.position_world[0]),
            float(object_pose_world.position_world[1]),
            float(object_pose_world.position_world[2] + dz),
        ),
        orientation_xyzw_world=object_pose_world.orientation_xyzw_world,
    )
    print(
        f"[PIPELINE] Settled object pose onto floor: mesh_min_z {min_z:.6f} -> {float(floor_z):.6f}, dz={dz:+.6f} m.",
        flush=True,
    )
    return settled_pose


def _execution_backend_configs(
    *,
    mujoco_execution: MujocoPipelineConfig,
    isaac_execution: IsaacPipelineConfig,
    backend: str,
) -> tuple[MujocoPipelineConfig, IsaacPipelineConfig]:
    normalized = str(backend).strip().lower()
    if normalized == "config":
        return mujoco_execution, isaac_execution
    if normalized == "mujoco":
        return replace(mujoco_execution, enabled=True), replace(isaac_execution, enabled=False)
    if normalized == "isaac":
        return replace(mujoco_execution, enabled=False), replace(isaac_execution, enabled=True)
    if normalized == "both":
        return replace(mujoco_execution, enabled=True), replace(isaac_execution, enabled=True)
    if normalized == "none":
        return replace(mujoco_execution, enabled=False), replace(isaac_execution, enabled=False)
    raise ValueError(f"Unsupported execution backend '{backend}'.")


def run_sim(payload: dict[str, object], *, headless: bool, backend: str = "config") -> None:
    geometry = _geometry_config(payload)
    planning = _planning_config(payload)
    artifacts = _artifacts(payload)
    mujoco_execution = _mujoco_execution_config(payload)
    isaac_execution = _isaac_execution_config(payload)
    mujoco_execution, isaac_execution = _execution_backend_configs(
        mujoco_execution=mujoco_execution,
        isaac_execution=isaac_execution,
        backend=backend,
    )
    pickup_pose = _pickup_pose_config(payload)
    execution_world_pose = None if pickup_pose is not None else _execution_pose_config(payload).to_object_pose_world()

    print("[PIPELINE] Loading geometry and generating raw grasps.", flush=True)
    stage1 = generate_stage1_result(geometry=geometry, planning=planning)
    print(
        f"[PIPELINE] Stage 1 complete: kept {len(stage1.bundle.candidates)} / {stage1.raw_candidate_count}.", flush=True
    )
    write_stage1_artifacts(
        stage1,
        geometry=geometry,
        planning=planning,
        output_json=artifacts["stage1_json"],
        output_html=artifacts["stage1_html"],
    )
    execution_world_pose = _settle_object_pose_on_floor(execution_world_pose, stage1.target_mesh_local)

    print("[PIPELINE] Rechecking grasps against the execution-world floor.", flush=True)
    stage2 = recheck_stage2_result(
        bundle=stage1.bundle,
        pickup_spec=pickup_pose.to_spec() if pickup_pose is not None else None,
        planning=planning,
        object_pose_world=execution_world_pose,
    )
    print(
        f"[PIPELINE] Stage 2 complete: feasible {len(stage2.accepted)} / {len(stage2.source_bundle.candidates)}.",
        flush=True,
    )
    write_stage2_artifacts(
        stage2,
        planning=planning,
        output_json=artifacts["stage2_json"],
        output_html=artifacts["stage2_html"],
    )
    _write_part_frame_debug_artifact(input_json=artifacts["stage2_json"], output_html=artifacts["part_frame_html"])
    _run_mujoco_execution(mujoco_execution, input_json=artifacts["stage2_json"], headless=headless)
    _run_isaac_execution(isaac_execution, input_json=artifacts["stage2_json"], headless=headless)


def run_pitl(payload: dict[str, object], *, headless: bool, backend: str = "config") -> None:
    geometry = _geometry_config(payload)
    planning = _planning_config(payload)
    artifacts = _artifacts(payload)
    ros2 = _ros2_config(payload)
    mujoco_execution = _mujoco_execution_config(payload)
    isaac_execution = _isaac_execution_config(payload)
    mujoco_execution, isaac_execution = _execution_backend_configs(
        mujoco_execution=mujoco_execution,
        isaac_execution=isaac_execution,
        backend=backend,
    )

    print("[PIPELINE] Starting repo-local ROS2 planning nodes.", flush=True)
    object_pose_world = _resolve_object_pose_world(ros2)
    print("[PIPELINE] Generating and filtering grasps.", flush=True)
    stage1 = generate_stage1_result(geometry=geometry, planning=planning)
    print(
        f"[PIPELINE] Stage 1 complete: kept {len(stage1.bundle.candidates)} / {stage1.raw_candidate_count}.", flush=True
    )
    write_stage1_artifacts(
        stage1,
        geometry=geometry,
        planning=planning,
        output_json=artifacts["stage1_json"],
        output_html=artifacts["stage1_html"],
    )
    object_pose_world = _settle_object_pose_on_floor(object_pose_world, stage1.target_mesh_local)

    print("[PIPELINE] Rechecking grasps against the real-world floor pose.", flush=True)
    stage2 = recheck_stage2_result(
        bundle=stage1.bundle,
        pickup_spec=None,
        planning=planning,
        object_pose_world=object_pose_world,
    )
    print(
        f"[PIPELINE] Planning complete: feasible {len(stage2.accepted)} / {len(stage2.source_bundle.candidates)}.",
        flush=True,
    )
    write_stage2_artifacts(
        stage2,
        planning=planning,
        output_json=artifacts["stage2_json"],
        output_html=artifacts["stage2_html"],
    )
    _write_part_frame_debug_artifact(input_json=artifacts["stage2_json"], output_html=artifacts["part_frame_html"])
    _run_mujoco_execution(mujoco_execution, input_json=artifacts["stage2_json"], headless=headless)
    _run_isaac_execution(isaac_execution, input_json=artifacts["stage2_json"], headless=headless)


def run_real(payload: dict[str, object]) -> None:
    geometry = _geometry_config(payload)
    planning = _planning_config(payload)
    artifacts = _artifacts(payload)
    ros2 = _ros2_config(payload)
    real_execution = _real_execution_config(payload)

    print("[PIPELINE] Starting repo-local ROS2 planning nodes.", flush=True)
    object_pose_world = _resolve_object_pose_world(ros2)
    print("[PIPELINE] Generating and filtering grasps.", flush=True)
    stage1 = generate_stage1_result(geometry=geometry, planning=planning)
    print(
        f"[PIPELINE] Stage 1 complete: kept {len(stage1.bundle.candidates)} / {stage1.raw_candidate_count}.", flush=True
    )
    write_stage1_artifacts(
        stage1,
        geometry=geometry,
        planning=planning,
        output_json=artifacts["stage1_json"],
        output_html=artifacts["stage1_html"],
    )
    object_pose_world = _settle_object_pose_on_floor(object_pose_world, stage1.target_mesh_local)

    print("[PIPELINE] Rechecking grasps against the real-world floor pose.", flush=True)
    stage2 = recheck_stage2_result(
        bundle=stage1.bundle,
        pickup_spec=None,
        planning=planning,
        object_pose_world=object_pose_world,
    )
    print(
        f"[PIPELINE] Planning complete: feasible {len(stage2.accepted)} / {len(stage2.source_bundle.candidates)}.",
        flush=True,
    )
    write_stage2_artifacts(
        stage2,
        planning=planning,
        output_json=artifacts["stage2_json"],
        output_html=artifacts["stage2_html"],
    )
    _write_part_frame_debug_artifact(input_json=artifacts["stage2_json"], output_html=artifacts["part_frame_html"])
    if real_execution.enabled:
        print("[PIPELINE] Starting real-robot execution from the stage-2 bundle.", flush=True)
        result = execute_real_grasp_from_bundle(input_json=artifacts["stage2_json"], config=real_execution)
        print(
            f"[PIPELINE] Real execution finished success={result.success} status={result.status} "
            f"grasp_id={result.grasp_id} message={result.message}",
            flush=True,
        )
        print(f"[PIPELINE] Wrote real execution artifact to {result.attempt_artifact_path}", flush=True)
        if not result.success:
            raise RuntimeError(result.message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shared grasp-planning pipeline.")
    parser.add_argument(
        "--mode",
        choices=("sim", "pitl", "real", "perception_in_the_loop", "perception-in-the-loop", "simulation"),
        required=True,
        help="Pipeline mode.",
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config path.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the MuJoCo viewer for executing modes.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="config",
        help="Override sim/pitl execution backend. Use 'config' to honor YAML.",
    )
    parser.add_argument(
        "--skip-stage1-collision-checks",
        action="store_true",
        help="Keep all stage-1 generated grasps and skip offline assembly collision filtering.",
    )
    args = parser.parse_args()
    mode = _normalize_mode(args.mode)

    if args.config is None:
        default_names = {
            "sim": "grasp_pipeline_sim.yaml",
            "pitl": "grasp_pipeline_pitl.yaml",
            "real": "grasp_pipeline_real.yaml",
        }
        default_name = default_names[mode]
        config_path = REPO_ROOT / "configs" / default_name
    else:
        config_path = args.config
    payload = _load_yaml(config_path)
    if args.skip_stage1_collision_checks:
        planning_payload = dict(payload.get("planning", {}))
        planning_payload["skip_stage1_collision_checks"] = True
        payload["planning"] = planning_payload
    if mode == "sim":
        run_sim(payload, headless=bool(args.headless), backend=args.backend)
        return
    if mode == "pitl":
        run_pitl(payload, headless=bool(args.headless), backend=args.backend)
        return
    if args.backend != "config":
        raise ValueError("--backend is only supported for sim and pitl modes.")
    run_real(payload)


if __name__ == "__main__":
    main()
