"""Generate per-arm compressed RGB-D policy deployment configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from grasp_planning.rl.policy_registry import load_yaml_mapping, resolve_from


def camera_driver_root(camera_name: str) -> str:
    """Return the ROS camera-node root for any selected namespace.

    ``realsense_2`` resolves to ``/realsense_2/camera`` while callers that
    already pass ``/cell/wrist/camera`` keep that complete node namespace.
    Camera identity is deliberately topic-driven; no serial or calibration
    allow-list is part of deployment routing.
    """

    parts = tuple(part for part in str(camera_name).strip().split("/") if part)
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError("camera_name must be a non-empty ROS namespace.")
    root = "/" + "/".join(parts)
    return root if parts[-1] == "camera" else f"{root}/camera"


def write_visual_servo_config(
    *,
    policy_name: str,
    assets: Mapping[str, object],
    template_path: Path,
    output_path: Path,
    output_root: Path,
    model_device: str,
    camera_name: str,
    robot_name: str | None = None,
) -> Path:
    """Write one standalone or shared-dual-cell policy route.

    ``robot_name`` is omitted for the standalone ``/lbr`` stack. In the dual
    stack it must be ``lbr_one`` or ``lbr_two`` and all TF, joint, force, and
    Servo endpoints are resolved into the shared ``/lbr_dual_arm`` graph.
    """

    visual = load_yaml_mapping(template_path)
    visual_block = dict(visual.get("visual_servo", {}))
    for path_key in (
        "goal_renderer_launcher",
        "goal_renderer_script",
        "goal_renderer_robot_urdf",
    ):
        visual_block[path_key] = str(
            resolve_from(visual_block.get(path_key, ""), base=template_path.parent)
        )
    camera_root = camera_driver_root(camera_name)
    if robot_name not in {None, "lbr_one", "lbr_two"}:
        raise ValueError("robot_name must be omitted, lbr_one, or lbr_two.")

    gripper_model = str(assets["gripper_model"])
    if gripper_model == "pdz_gripper":
        tcp_suffix = "pdz_gripper_tcp"
        renderer_urdf = (
            Path(__file__).resolve().parents[2]
            / "assets/urdf/kuka_iiwa7_pdz_gripper/urdf/kuka_iiwa7_pdz_gripper.urdf"
        )
    elif gripper_model == "y_gripper":
        tcp_suffix = "gripper_tcp"
        renderer_urdf = (
            Path(__file__).resolve().parents[2]
            / "assets/urdf/kuka_iiwa7_y_gripper/urdf/kuka_iiwa7_y_gripper.urdf"
        )
    else:
        raise ValueError(f"Unsupported policy gripper_model '{gripper_model}'.")

    visual_block.update(
        {
            "policy_name": str(policy_name),
            "checkpoint_path": str(assets["checkpoint"]),
            "checkpoint_metadata_path": str(assets["metadata"]),
            "agent_config_path": str(assets["agent_config"]),
            "goal_observation_path": "",
            "model_device": str(model_device),
            "command_sink": "moveit_servo",
            "real_motion_approved": True,
            "allow_gripper_close_on_completion": True,
            "color_topic": f"{camera_root}/color/image_rect/compressed",
            "depth_topic": f"{camera_root}/aligned_depth_to_color/image_rect/compressedDepth",
            "color_camera_info_topic": f"{camera_root}/color/camera_info",
            "depth_camera_info_topic": f"{camera_root}/aligned_depth_to_color/camera_info",
            "policy_rate_hz": float(assets["policy_rate_hz"]),
            "action_delta_limit": float(assets["action_delta_limit"]),
            "expected_camera_profile": str(assets["camera_profile"]),
            "goal_renderer_robot_urdf": str(renderer_urdf.resolve()),
            "require_deadman": False,
            "deadman_topic": "",
            "emergency_stop_topic": "",
            "max_image_age_s": 0.50,
            "enforce_source_image_age": False,
            "max_pose_age_s": 0.50,
            "max_tf_age_s": 0.50,
            "max_servo_status_age_s": 0.50,
            "max_joint_state_age_s": 0.50,
            "max_force_age_s": 0.50,
            "output_root": str(output_root),
        }
    )
    if robot_name is None:
        visual_block.update(
            {
                "camera_optical_frame": "camera_color_optical_frame",
                "command_frame": "lbr_link_0",
                "tcp_frame": tcp_suffix,
                "joint_state_topic": "/lbr/joint_states",
                "force_topic": "/lbr/force_torque_broadcaster/wrench",
                "moveit_servo_twist_topic": "/lbr/servo_node/delta_twist_cmds",
                "moveit_servo_status_topic": "/lbr/servo_node/status",
                "moveit_servo_start_service": "/lbr/servo_node/start_servo",
                "moveit_servo_stop_service": "/lbr/servo_node/stop_servo",
                "expected_joint_names": [f"lbr_A{index}" for index in range(1, 8)],
            }
        )
    else:
        servo_root = f"/lbr_dual_arm/{robot_name}_servo_node"
        visual_block.update(
            {
                "camera_optical_frame": f"{robot_name}_camera_color_optical_frame",
                # Topic namespace + verified camera serial select the physical
                # stream. The deployed RealSense drivers still stamp the
                # generic camera_color_optical_frame name, while the shared
                # robot_description must keep per-arm TF frame names unique.
                "allow_camera_topic_frame_alias": True,
                "command_frame": "base_link",
                "tcp_frame": f"{robot_name}_{tcp_suffix}",
                "joint_state_topic": "/lbr_dual_arm/joint_states",
                "force_topic": (
                    f"/lbr_dual_arm/{robot_name}_control/"
                    f"{robot_name}_force_torque_broadcaster/wrench"
                ),
                "moveit_servo_twist_topic": f"{servo_root}/delta_twist_cmds",
                "moveit_servo_status_topic": f"{servo_root}/status",
                "moveit_servo_start_service": f"{servo_root}/start_servo",
                "moveit_servo_stop_service": f"{servo_root}/stop_servo",
                "expected_joint_names": [f"{robot_name}_A{index}" for index in range(1, 8)],
            }
        )
    visual_block["allow_pdz_camera_rotation_fallback"] = gripper_model == "y_gripper"
    visual["visual_servo"] = visual_block
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(visual, sort_keys=False), encoding="utf-8")
    return output_path


__all__ = ["camera_driver_root", "write_visual_servo_config"]
