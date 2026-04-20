"""Run the shared planning pipeline in sim, pitl, or real modes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    DEFAULT_CONTACT_APPROACH_OFFSETS_M,
    DEFAULT_CONTACT_LATERAL_OFFSETS_M,
    rotmat_to_quat_xyzw,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.pipeline import (  # noqa: E402
    ExecutionWorldPoseConfig,
    GeometryConfig,
    MujocoPipelineConfig,
    PickupPoseConfig,
    PlanningConfig,
    Ros2Config,
    generate_stage1_result,
    recheck_stage2_result,
    write_stage1_artifacts,
    write_stage2_artifacts,
)
from grasp_planning.ros2 import (  # noqa: E402
    wait_for_object_pose_message,
    wait_for_real_frame_pair_messages,
)


def _tuple_floats(values: object, *, expected_len: int | None = None) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a list/tuple of floats, got {values!r}.")
    result = tuple(float(value) for value in values)
    if expected_len is not None and len(result) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(result)}.")
    return result


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
        roll_angles_rad=_tuple_floats(raw.get("roll_angles_rad", [0.0])),
        max_pair_checks=int(raw.get("max_pair_checks", 40960)),
        detailed_finger_contact_gap_m=float(raw.get("detailed_finger_contact_gap_m", 0.002)),
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
    return MujocoPipelineConfig(
        enabled=bool(raw.get("enabled", False)),
        python_executable=str(raw.get("python_executable", "")),
        robot_config=str(raw.get("robot_config", "")),
        simulation_config=str(raw.get("simulation_config", "")),
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
    )


def _ros2_config(payload: dict[str, object]) -> Ros2Config:
    raw = dict(payload.get("ros2", {}))
    return Ros2Config(
        object_pose_topic=str(raw.get("object_pose_topic", "/grasp_planning/object_pose")),
        pose_message_type=str(raw.get("pose_message_type", "geometry_msgs/msg/Pose")),
        frame_id=str(raw.get("frame_id", "world")),
        timeout_s=float(raw.get("timeout_s", 10.0)),
        object_id=str(raw.get("object_id", "")),
        local_frame_offset_topic=str(raw.get("local_frame_offset_topic", "")),
        local_frame_offset_message_type=str(
            raw.get("local_frame_offset_message_type", "geometry_msgs/msg/Vector3Stamped")
        ),
        execution_frame_topic=str(raw.get("execution_frame_topic", "")),
        execution_frame_message_type=str(raw.get("execution_frame_message_type", "fp_debug_msgs/msg/DebugFrame")),
    )


def _artifacts(payload: dict[str, object]) -> dict[str, Path]:
    raw = dict(payload.get("artifacts", {}))
    return {
        "stage1_json": Path(str(raw["stage1_json"])),
        "stage1_html": Path(str(raw["stage1_html"])),
        "stage2_json": Path(str(raw["stage2_json"])),
        "stage2_html": Path(str(raw["stage2_html"])),
    }


def _format_topic(topic_template: str, *, object_id: str) -> str:
    if "{object_id}" in topic_template:
        if not object_id:
            raise ValueError(f"object_id is required to resolve ROS topic template '{topic_template}'.")
        return topic_template.format(object_id=object_id)
    return topic_template


def _compose_object_world_poses(
    parent_pose_world: ObjectWorldPose,
    child_pose_parent: ObjectWorldPose,
) -> ObjectWorldPose:
    parent_rotation = parent_pose_world.rotation_world_from_object
    child_rotation = child_pose_parent.rotation_world_from_object
    composed_rotation = parent_rotation @ child_rotation
    composed_translation = parent_rotation @ child_pose_parent.translation_world + parent_pose_world.translation_world
    return ObjectWorldPose(
        position_world=tuple(float(v) for v in composed_translation.tolist()),
        orientation_xyzw_world=rotmat_to_quat_xyzw(composed_rotation),
    )


def _resolve_real_world_frames(ros2: Ros2Config):
    if ros2.local_frame_offset_topic and ros2.execution_frame_topic and ros2.object_id:
        local_frame_topic = _format_topic(ros2.local_frame_offset_topic, object_id=ros2.object_id)
        execution_frame_topic = _format_topic(ros2.execution_frame_topic, object_id=ros2.object_id)
        print("[PIPELINE] Waiting for source-frame offset and execution-frame pose.", flush=True)
        frame_pair = wait_for_real_frame_pair_messages(
            source_topic_name=local_frame_topic,
            source_message_type=ros2.local_frame_offset_message_type,
            execution_topic_name=execution_frame_topic,
            execution_message_type=ros2.execution_frame_message_type,
            object_id=ros2.object_id,
            timeout_s=ros2.timeout_s,
        )
        source_frame_pose_world = _compose_object_world_poses(
            frame_pair.execution_pose_world,
            frame_pair.source_frame_pose_obj_world,
        )
        return frame_pair.source_frame_pose_obj_world, source_frame_pose_world

    print("[PIPELINE] Waiting for legacy object pose topic.", flush=True)
    object_pose_world = wait_for_object_pose_message(
        topic_name=ros2.object_pose_topic,
        message_type=ros2.pose_message_type,
        timeout_s=ros2.timeout_s,
    )
    return None, object_pose_world


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
    raise RuntimeError("Could not determine a Python executable for MuJoCo execution.")


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
    print("[PIPELINE] Starting MuJoCo execution.", flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def run_sim(payload: dict[str, object], *, headless: bool) -> None:
    geometry = _geometry_config(payload)
    planning = _planning_config(payload)
    execution_world_pose = _execution_pose_config(payload).to_object_pose_world()
    artifacts = _artifacts(payload)
    mujoco_execution = _mujoco_execution_config(payload)
    pickup_pose = _pickup_pose_config(payload)

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
    _run_mujoco_execution(mujoco_execution, input_json=artifacts["stage2_json"], headless=headless)


def run_pitl(payload: dict[str, object], *, headless: bool) -> None:
    geometry = _geometry_config(payload)
    planning = _planning_config(payload)
    artifacts = _artifacts(payload)
    ros2 = _ros2_config(payload)
    mujoco_execution = _mujoco_execution_config(payload)

    print("[PIPELINE] Starting repo-local ROS2 planning nodes.", flush=True)
    source_frame_pose_obj_world, object_pose_world = _resolve_real_world_frames(ros2)
    print("[PIPELINE] Generating and filtering grasps.", flush=True)
    stage1 = generate_stage1_result(
        geometry=geometry,
        planning=planning,
        source_frame_pose_obj_world=source_frame_pose_obj_world,
    )
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
    _run_mujoco_execution(mujoco_execution, input_json=artifacts["stage2_json"], headless=headless)


def run_real(payload: dict[str, object]) -> None:
    geometry = _geometry_config(payload)
    planning = _planning_config(payload)
    artifacts = _artifacts(payload)
    ros2 = _ros2_config(payload)

    print("[PIPELINE] Starting repo-local ROS2 planning nodes.", flush=True)
    source_frame_pose_obj_world, object_pose_world = _resolve_real_world_frames(ros2)
    print("[PIPELINE] Generating and filtering grasps.", flush=True)
    stage1 = generate_stage1_result(
        geometry=geometry,
        planning=planning,
        source_frame_pose_obj_world=source_frame_pose_obj_world,
    )
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
    if mode == "sim":
        run_sim(payload, headless=bool(args.headless))
        return
    if mode == "pitl":
        run_pitl(payload, headless=bool(args.headless))
        return
    run_real(payload)


if __name__ == "__main__":
    main()
