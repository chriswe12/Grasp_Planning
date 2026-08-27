from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts import run_single_arm_policy_pickup as pickup

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checked_in_registry_names_policies_without_stored_targets() -> None:
    registry, _asset_root = pickup.load_policy_registry(
        REPO_ROOT / "configs/d405_policy_registry.yaml"
    )

    assert set(registry["policies"]) == {
        "combined-v4",
        "clutter-v5",
        "depth-v5-a",
        "depth-v5-b",
        "baseline",
        "background",
        "velocity",
        "velocity-rotation",
    }
    assert registry["default_part_id"] == "0"
    assert "goal_catalog" not in registry
    assert "planned_manifest" not in registry
    assert "grasp_source_bundles" not in registry


def test_policy_assets_validate_checkpoint_without_goal_catalogue(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    checkpoint = assets / "policy.pth"
    checkpoint.write_bytes(b"checkpoint")
    metadata = assets / "policy.json"
    metadata.write_text(
        json.dumps({"checkpoint_sha256": _sha256(checkpoint)}),
        encoding="utf-8",
    )
    agent_config = tmp_path / "agent.yaml"
    agent_config.write_text("params: {}\n", encoding="utf-8")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "asset_root": "assets",
                "agent_config": "agent.yaml",
                "policies": {
                    "test": {"checkpoint": "policy.pth", "metadata": "policy.json"}
                },
            }
        ),
        encoding="utf-8",
    )

    registry, asset_root = pickup.load_policy_registry(registry_path)
    resolved = pickup.resolve_policy_assets(
        registry,
        registry_path=registry_path,
        asset_root=asset_root,
        policy_name="test",
    )

    assert resolved["checkpoint"] == checkpoint.resolve()
    assert resolved["metadata"] == metadata.resolve()
    assert resolved["agent_config"] == agent_config.resolve()
    assert resolved["policy_context_mode"] == "action"
    assert resolved["policy_rate_hz"] == 30.0
    assert resolved["gripper_model"] == "y_gripper"


def test_resolved_config_uses_any_live_grasp_and_runtime_goal_renderer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pth"
    metadata = tmp_path / "policy.json"
    agent = tmp_path / "agent.yaml"
    for path in (checkpoint, metadata, agent):
        path.write_bytes(b"fixture")
    output = tmp_path / "run"

    pipeline_path, visual_path = pickup.write_resolved_configs(
        policy_name="clutter-v5",
        part_id="2",
        part_mesh=pickup._validate_part_id("2"),
        assets={
            "checkpoint": checkpoint,
            "metadata": metadata,
            "agent_config": agent,
            "policy_context_mode": "action_twist_rotation",
            "policy_rate_hz": 15.0,
            "action_delta_limit": 0.5,
            "camera_profile": (
                "d405_color_848x480_intrinsics_260322275185_render_256x144_"
                "pdz_named_frame_link7_mount35mm_rpy30_0_180_v1"
            ),
            "gripper_model": "pdz_gripper",
        },
        pipeline_template=REPO_ROOT / "configs/grasp_pipeline_real_lbr_iiwa7.yaml",
        visual_servo_template=REPO_ROOT / "configs/visual_servo_real_d405.yaml",
        output_dir=output,
        model_device="cuda:0",
        open_debug_html=True,
    )

    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    visual = yaml.safe_load(visual_path.read_text(encoding="utf-8"))["visual_servo"]
    real = pipeline["real_execution"]
    assert pipeline["ros2"]["part_id"] == 2
    assert real["grasp_id"] == ""
    assert real["grasp_approach_controller"] == "d405_policy"
    assert real["gripper_client"] == "normalized_position"
    assert real["gripper_trigger_open_service"] == "/left/gripper_controller/open"
    assert real["gripper_trigger_close_service"] == "/left/gripper_controller/close"
    assert real["gripper_trigger_stop_service"] == "/left/gripper_controller/stop"
    assert real["gripper_position_command_topic"] == "/left/gripper_controller/position_command"
    assert real["gripper_position_feedback_topic"] == "/left/gripper_controller/position"
    assert real["gripper_closed_width"] == 0.007
    assert real["gripper_open_width"] == 0.074
    assert pipeline["planning"]["min_jaw_width"] == 0.008
    assert pipeline["planning"]["max_jaw_width"] == 0.062
    assert pipeline["planning"]["gripper_collision_model"] == "pdz_gripper"
    assert real["pose_link"] == "pdz_gripper_tcp"
    assert real["moveit_gripper_joint_name"] == "pdz_gripper_left_finger_joint"
    assert "policy_target_id" not in real
    assert "policy_target_candidates" not in real
    assert "policy_grasp_source_bundle" not in real
    assert visual["goal_observation_path"] == ""
    assert visual["image_transport"] == "compressed"
    assert visual["color_topic"].endswith("/image_rect/compressed")
    assert visual["depth_topic"].endswith("/image_rect/compressedDepth")
    assert visual["color_topic"].startswith("/realsense_1/camera/")
    assert visual["color_camera_info_topic"] == "/realsense_1/camera/color/camera_info"
    assert visual["depth_camera_info_topic"] == (
        "/realsense_1/camera/aligned_depth_to_color/camera_info"
    )
    assert visual["camera_parameter_node"] == "/realsense_1/camera"
    assert visual["expected_camera_serial"] == "260522275434"
    assert visual["allow_pdz_camera_rotation_fallback"] is False
    assert visual["tcp_frame"] == "pdz_gripper_tcp"
    assert visual["policy_rate_hz"] == 15.0
    assert visual["action_delta_limit"] == 0.5
    assert visual["require_deadman"] is False
    assert visual["deadman_topic"] == ""
    assert visual["emergency_stop_topic"] == ""
    assert visual["max_image_age_s"] == 0.5
    assert visual["enforce_source_image_age"] is False
    assert visual["max_pose_age_s"] == 0.5
    assert visual["max_tf_age_s"] == 0.5
    assert visual["max_servo_status_age_s"] == 0.5
    assert visual["max_joint_state_age_s"] == 0.5
    assert visual["max_force_age_s"] == 0.5
    assert visual["goal_renderer_backend"] == "filament"
    assert Path(visual["goal_renderer_launcher"]).is_absolute()
    assert Path(visual["goal_renderer_script"]).is_absolute()
    assert Path(visual["goal_renderer_robot_urdf"]).is_absolute()
    assert "IsaacLab" not in visual["goal_renderer_python_command"]
    assert pipeline["artifacts"]["open_debug_html"] is True


def test_part_validation_accepts_all_fabrica_parts() -> None:
    for part_id in range(5):
        assert pickup._validate_part_id(str(part_id)).is_file()


def test_resolved_config_can_route_complete_rgbd_source_to_realsense_2(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.write_bytes(b"fixture")
    _, visual_path = pickup.write_resolved_configs(
        policy_name="velocity",
        part_id="0",
        part_mesh=pickup._validate_part_id("0"),
        assets={
            "checkpoint": fixture,
            "metadata": fixture,
            "agent_config": fixture,
            "policy_context_mode": "action_twist",
            "policy_rate_hz": 15.0,
            "action_delta_limit": 0.5,
            "camera_profile": "pdz-test-profile",
            "gripper_model": "pdz_gripper",
        },
        pipeline_template=REPO_ROOT / "configs/grasp_pipeline_real_lbr_iiwa7.yaml",
        visual_servo_template=REPO_ROOT / "configs/visual_servo_real_d405.yaml",
        output_dir=tmp_path / "run",
        model_device="cpu",
        open_debug_html=False,
        camera_name="realsense_2",
    )

    visual = yaml.safe_load(visual_path.read_text(encoding="utf-8"))["visual_servo"]
    assert visual["color_topic"] == "/realsense_2/camera/color/image_rect/compressed"
    assert visual["depth_topic"] == (
        "/realsense_2/camera/aligned_depth_to_color/image_rect/compressedDepth"
    )
    assert visual["color_camera_info_topic"] == "/realsense_2/camera/color/camera_info"
    assert visual["depth_camera_info_topic"] == (
        "/realsense_2/camera/aligned_depth_to_color/camera_info"
    )
    assert visual["camera_parameter_node"] == "/realsense_2/camera"
    assert visual["expected_camera_serial"] == "260322275185"
