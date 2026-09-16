"""Lightweight, Isaac-free runtime for the trained D405 RGB-D PPO actor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml

from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_OBSERVATION_HEIGHT,
    VISUAL_SERVO_OBSERVATION_WIDTH,
    D405WristCameraConfig,
)
from grasp_planning.rl.d405_observation import (
    D405ObservationPreprocessCfg,
    preprocess_aligned_rgbd_torch,
)
from grasp_planning.rl.policy_context import (
    POLICY_CONTEXT_ACTION,
    assemble_policy_context_torch,
    policy_observation_size,
    resolve_policy_context,
)
from grasp_planning.rl.policy_timing import POLICY_RATE_HZ
from grasp_planning.ros2.visual_servo_safety import slew_limit_normalized_action

POLICY_MOTION_SIZE = 6
POLICY_OUTPUT_SIZE = 7
POLICY_IMAGE_CHANNELS = 8
POLICY_IMAGE_VALUE_COUNT = VISUAL_SERVO_OBSERVATION_HEIGHT * VISUAL_SERVO_OBSERVATION_WIDTH * POLICY_IMAGE_CHANNELS
POLICY_DEPLOYMENT_INPUT_SIZE = POLICY_IMAGE_VALUE_COUNT + POLICY_MOTION_SIZE
POLICY_PRIVILEGED_PLACEHOLDER_SIZE = 8
POLICY_FULL_INPUT_SIZE = POLICY_DEPLOYMENT_INPUT_SIZE + POLICY_PRIVILEGED_PLACEHOLDER_SIZE
# Current physical camera published under the left-arm ``realsense_1`` tree.
# Checkpoint compatibility is governed by the camera/observation profiles, not
# by this live USB identity.
EXPECTED_D405_SERIAL = "260522275434"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ros_depth_z16_to_metres(depth_z16: np.ndarray) -> np.ndarray:
    """Convert RealSense ROS ``16UC1`` millimetres to metric float depth once."""

    depth = np.asarray(depth_z16)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Expected a uint16 HW depth image, got shape={depth.shape} dtype={depth.dtype}.")
    return depth.astype(np.float32) * 0.001


def assemble_policy_observation(
    live_rgbd: torch.Tensor,
    goal_rgbd: torch.Tensor,
    previous_applied_action: Sequence[float],
    *,
    policy_context_mode: str = POLICY_CONTEXT_ACTION,
    normalized_tcp_twist_camera: Sequence[float] | None = None,
    rotation_base_from_camera: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> torch.Tensor:
    expected_image_shape = (1, VISUAL_SERVO_OBSERVATION_HEIGHT, VISUAL_SERVO_OBSERVATION_WIDTH, 4)
    if tuple(live_rgbd.shape) != expected_image_shape or tuple(goal_rgbd.shape) != expected_image_shape:
        raise ValueError(
            f"Live and goal RGB-D tensors must both have shape {expected_image_shape}; "
            f"got live={tuple(live_rgbd.shape)} goal={tuple(goal_rgbd.shape)}."
        )
    previous = torch.as_tensor(previous_applied_action, dtype=torch.float32, device=live_rgbd.device)
    if tuple(previous.shape) != (POLICY_MOTION_SIZE,) or not torch.isfinite(previous).all():
        raise ValueError("Previous applied action must contain six finite normalized values.")
    normalized_twist = None
    if normalized_tcp_twist_camera is not None:
        normalized_twist = torch.as_tensor(
            normalized_tcp_twist_camera,
            dtype=torch.float32,
            device=live_rgbd.device,
        ).unsqueeze(0)
    rotation = None
    if rotation_base_from_camera is not None:
        rotation = torch.as_tensor(
            rotation_base_from_camera,
            dtype=torch.float32,
            device=live_rgbd.device,
        ).unsqueeze(0)
    context = assemble_policy_context_torch(
        policy_context_mode,
        previous.unsqueeze(0),
        normalized_tcp_twist_camera=normalized_twist,
        rotation_base_from_camera=rotation,
    )
    image = torch.cat((live_rgbd, goal_rgbd.to(device=live_rgbd.device)), dim=-1).flatten(start_dim=1)
    observation = torch.cat(
        (
            image,
            context,
            torch.zeros((1, POLICY_PRIVILEGED_PLACEHOLDER_SIZE), dtype=torch.float32, device=live_rgbd.device),
        ),
        dim=1,
    )
    expected_size = policy_observation_size(
        policy_context_mode,
        image_value_count=POLICY_IMAGE_VALUE_COUNT,
        privileged_label_size=POLICY_PRIVILEGED_PLACEHOLDER_SIZE,
    )
    if tuple(observation.shape) != (1, expected_size):
        raise AssertionError(f"Unexpected inference observation shape {tuple(observation.shape)}.")
    if not torch.isfinite(observation).all():
        raise ValueError("Inference observation contains non-finite values.")
    return observation


@dataclass(frozen=True)
class GoalTarget:
    goal_id: str
    part_id: str
    grasp_id: str
    goal_rgbd: torch.Tensor
    jaw_width_m: float


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create a Python module spec for '{path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class D405RuntimeGoal:
    """One goal RGB-D observation rendered for the selected live grasp."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_camera_profile: str = D405_VISUAL_SERVO_CAMERA_PROFILE,
        expected_observation_profile: str = D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with np.load(self.path, allow_pickle=False) as source:
            required = (
                "goal_id",
                "part_id",
                "grasp_id",
                "jaw_width_m",
                "goal_rgb",
                "goal_depth",
                "goal_camera_profile",
                "goal_observation_profile",
                "render_validation_passed",
            )
            missing = tuple(name for name in required if name not in source.files)
            if missing:
                raise ValueError(f"Runtime goal observation is missing required arrays: {missing}.")
            self.arrays = {name: source[name].copy() for name in source.files}
        rgb = np.asarray(self.arrays["goal_rgb"])
        depth = np.asarray(self.arrays["goal_depth"])
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Runtime goal_rgb must be uint8 [height, width, 3].")
        if depth.shape != rgb.shape[:2] or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError("Runtime goal_depth must be floating [height, width].")
        if not np.isfinite(depth).all():
            raise ValueError("Runtime goal depth contains non-finite values.")
        if not bool(np.asarray(self.arrays["render_validation_passed"]).item()):
            raise ValueError("Runtime goal renderer did not validate the selected grasp observation.")
        camera_profile = str(np.asarray(self.arrays.get("goal_camera_profile", "")).item())
        observation_profile = str(np.asarray(self.arrays.get("goal_observation_profile", "")).item())
        if camera_profile != expected_camera_profile:
            raise ValueError(
                f"Runtime goal camera profile '{camera_profile or 'unlabeled'}' does not match "
                f"'{expected_camera_profile}'."
            )
        if observation_profile != expected_observation_profile:
            raise ValueError(
                f"Runtime goal observation profile '{observation_profile or 'unlabeled'}' does not match "
                f"'{expected_observation_profile}'."
            )
        self.camera_profile = camera_profile
        self.observation_profile = observation_profile
        self.sha256 = sha256_file(self.path)

    def load(
        self,
        *,
        expected_grasp_id: str = "",
        expected_part_id: str = "",
        device: str | torch.device = "cpu",
    ) -> GoalTarget:
        goal_id = str(np.asarray(self.arrays["goal_id"]).item()).strip()
        part_id = str(np.asarray(self.arrays["part_id"]).item()).strip()
        grasp_id = str(np.asarray(self.arrays["grasp_id"]).item()).strip()
        if not goal_id or not part_id or not grasp_id:
            raise ValueError("Runtime goal identity fields must be non-empty.")
        if expected_grasp_id and grasp_id != str(expected_grasp_id):
            raise ValueError(
                f"Runtime goal '{goal_id}' uses grasp '{grasp_id}', but MoveIt selected "
                f"'{expected_grasp_id}'."
            )
        if expected_part_id and part_id != str(expected_part_id):
            raise ValueError(
                f"Runtime goal '{goal_id}' uses part '{part_id}', but the stage-2 bundle is for "
                f"part '{expected_part_id}'."
            )
        rgb = torch.from_numpy(np.asarray(self.arrays["goal_rgb"]).copy()).to(device=device, dtype=torch.float32)
        rgb = rgb.div(255.0).unsqueeze(0)
        depth_m = torch.from_numpy(np.asarray(self.arrays["goal_depth"]).copy()).to(device=device, dtype=torch.float32)
        depth_m = depth_m.unsqueeze(0)
        cfg = D405ObservationPreprocessCfg.from_camera(D405WristCameraConfig())
        goal_rgbd, _valid = preprocess_aligned_rgbd_torch(rgb, depth_m, cfg=cfg)
        return GoalTarget(
            goal_id=goal_id,
            part_id=part_id,
            grasp_id=grasp_id,
            goal_rgbd=goal_rgbd,
            jaw_width_m=float(np.asarray(self.arrays["jaw_width_m"]).item()),
        )


@dataclass(frozen=True)
class PolicyInference:
    requested_normalized_action: tuple[float, float, float, float, float, float]
    filtered_normalized_action: tuple[float, float, float, float, float, float]
    completion_probability: float
    valid_depth_fraction: float


class D405PolicyRuntime:
    """Preprocess live RGB-D and execute deterministic checkpoint inference."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        checkpoint_metadata_path: str | Path,
        agent_config_path: str | Path,
        goal_observation_path: str | Path,
        expected_grasp_id: str,
        expected_part_id: str,
        device: str = "cuda:0",
        expected_camera_profile: str = D405_VISUAL_SERVO_CAMERA_PROFILE,
        expected_observation_profile: str = D405_VISUAL_SERVO_OBSERVATION_PROFILE,
        linear_action_scale_m_s: float = 0.04,
        angular_action_scale_rad_s: float = 0.24,
        action_delta_limit: float = 0.25,
        policy_rate_hz: float = POLICY_RATE_HZ,
        completion_probability_threshold: float = 0.95,
        completion_required_consecutive_steps: int = 4,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.checkpoint_metadata_path = Path(checkpoint_metadata_path).expanduser().resolve()
        self.agent_config_path = Path(agent_config_path).expanduser().resolve()
        self.device = torch.device(device)
        self.linear_action_scale_m_s = float(linear_action_scale_m_s)
        self.angular_action_scale_rad_s = float(angular_action_scale_rad_s)
        self.action_delta_limit = float(action_delta_limit)
        self.policy_rate_hz = float(policy_rate_hz)
        if not math.isfinite(self.policy_rate_hz) or self.policy_rate_hz <= 0.0:
            raise ValueError("policy_rate_hz must be finite and positive.")
        self.completion_probability_threshold = float(completion_probability_threshold)
        self.completion_required_consecutive_steps = int(completion_required_consecutive_steps)
        self.preprocess_cfg = D405ObservationPreprocessCfg.from_camera(D405WristCameraConfig())
        self.goal_observation = D405RuntimeGoal(
            goal_observation_path,
            expected_camera_profile=expected_camera_profile,
            expected_observation_profile=expected_observation_profile,
        )
        self.goal = self.goal_observation.load(
            expected_grasp_id=expected_grasp_id,
            expected_part_id=expected_part_id,
            device=self.device,
        )
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.policy_context_mode = POLICY_CONTEXT_ACTION
        self.policy_context_size = resolve_policy_context(POLICY_CONTEXT_ACTION).size
        self.network_input_size = POLICY_FULL_INPUT_SIZE
        self._validate_checkpoint_metadata(
            expected_camera_profile=expected_camera_profile,
            expected_observation_profile=expected_observation_profile,
        )
        self.model = self._load_model()
        self.previous_applied_action = np.zeros(POLICY_MOTION_SIZE, dtype=np.float32)

    def _validate_checkpoint_metadata(
        self,
        *,
        expected_camera_profile: str,
        expected_observation_profile: str,
    ) -> None:
        if not self.checkpoint_metadata_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint compatibility metadata is required for deployment: {self.checkpoint_metadata_path}"
            )
        raw = yaml.safe_load(self.checkpoint_metadata_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("Checkpoint metadata must be a JSON/YAML mapping.")
        context_spec = resolve_policy_context(
            str(raw.get("policy_context_mode", POLICY_CONTEXT_ACTION))
        )
        self.policy_context_mode = context_spec.name
        self.policy_context_size = context_spec.size
        self.network_input_size = policy_observation_size(
            context_spec.name,
            image_value_count=POLICY_IMAGE_VALUE_COUNT,
            privileged_label_size=POLICY_PRIVILEGED_PLACEHOLDER_SIZE,
        )
        expected = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "camera_profile": expected_camera_profile,
            "observation_profile": expected_observation_profile,
            "network_input_size": self.network_input_size,
            "network_output_size": POLICY_OUTPUT_SIZE,
            "linear_action_scale_m_s": self.linear_action_scale_m_s,
            "angular_action_scale_rad_s": self.angular_action_scale_rad_s,
            "completion_probability_threshold": self.completion_probability_threshold,
            "completion_required_consecutive_steps": self.completion_required_consecutive_steps,
        }
        for key, expected_value in expected.items():
            actual = raw.get(key)
            if isinstance(expected_value, float):
                matches = actual is not None and math.isclose(float(actual), expected_value, abs_tol=1.0e-9)
            else:
                matches = actual == expected_value
            if not matches:
                raise ValueError(
                    f"Checkpoint metadata mismatch for '{key}': expected {expected_value!r}, got {actual!r}."
                )
        optional_expected = {
            "policy_context_size": self.policy_context_size,
            "action_delta_limit": self.action_delta_limit,
            "policy_rate_hz": self.policy_rate_hz,
        }
        for key, expected_value in optional_expected.items():
            if key not in raw:
                continue
            actual = raw[key]
            matches = (
                math.isclose(float(actual), float(expected_value), abs_tol=1.0e-9)
                if isinstance(expected_value, float)
                else actual == expected_value
            )
            if not matches:
                raise ValueError(
                    f"Checkpoint metadata mismatch for '{key}': expected {expected_value!r}, got {actual!r}."
                )

    def _load_model(self):
        if not self.agent_config_path.is_file():
            raise FileNotFoundError(self.agent_config_path)
        config = yaml.safe_load(self.agent_config_path.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping) or not isinstance(config.get("params"), Mapping):
            raise ValueError("RL-Games agent configuration is missing its params mapping.")
        params = dict(config["params"])
        network_params = dict(params.get("network", {}))
        network_params["pretrained"] = False
        network_params["policy_context_size"] = self.policy_context_size
        model_config = dict(params.get("config", {}))
        agents_root = Path(__file__).resolve().parent / "deployment_model"
        network_module = _load_source_module(
            "_d405_deployment_resnet_rgbd_network",
            agents_root / "resnet_rgbd_network.py",
        )
        model_module = _load_source_module(
            "_d405_deployment_completion_model",
            agents_root / "completion_model.py",
        )
        builder = network_module.GraspRgbdResNetBuilder()
        builder.load(network_params)
        model_factory = model_module.GraspCompletionModel(builder)
        model = model_factory.build(
            {
                "actions_num": POLICY_OUTPUT_SIZE,
                "input_shape": (self.network_input_size,),
                "num_seqs": 1,
                "value_size": 1,
                "normalize_value": bool(model_config.get("normalize_value", True)),
                "normalize_input": bool(model_config.get("normalize_input", False)),
            }
        )
        numpy_core = getattr(np, "_core", np.core)
        scalar_type = numpy_core.multiarray.scalar
        scalar_safe_global: object = scalar_type
        if scalar_type.__module__ != "numpy.core.multiarray":
            scalar_safe_global = (scalar_type, "numpy.core.multiarray.scalar")
        numpy_safe_globals = [
            scalar_safe_global,
            (np.dtype, "numpy.dtype"),
            *{
                type(np.dtype(dtype))
                for dtype in (
                    np.bool_,
                    np.float32,
                    np.float64,
                    np.int32,
                    np.int64,
                    np.uint32,
                    np.uint64,
                )
            },
        ]
        try:
            with torch.serialization.safe_globals(numpy_safe_globals):
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        except (AttributeError, TypeError):  # PyTorch before safe weights-only loading support
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model"), Mapping):
            raise ValueError("RL-Games checkpoint does not contain a model state dictionary.")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)
        return model

    def infer(
        self,
        rgb_uint8: np.ndarray,
        depth_z16: np.ndarray,
        *,
        tcp_twist_camera: Sequence[float] | None = None,
        rotation_base_from_camera: np.ndarray | Sequence[Sequence[float]] | None = None,
    ) -> PolicyInference:
        rgb = np.asarray(rgb_uint8)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB uint8 HWC input, got shape={rgb.shape} dtype={rgb.dtype}.")
        depth_m = ros_depth_z16_to_metres(depth_z16)
        if rgb.shape[:2] != depth_m.shape:
            raise ValueError("Live RGB and aligned depth dimensions do not match.")
        rgb_t = torch.from_numpy(rgb.copy()).to(self.device, dtype=torch.float32).div(255.0).unsqueeze(0)
        depth_t = torch.from_numpy(depth_m.copy()).to(self.device, dtype=torch.float32).unsqueeze(0)
        live_rgbd, valid = preprocess_aligned_rgbd_torch(rgb_t, depth_t, cfg=self.preprocess_cfg)
        normalized_twist = None
        if resolve_policy_context(self.policy_context_mode).uses_tcp_twist:
            twist = np.asarray(tcp_twist_camera, dtype=np.float32)
            if twist.shape != (6,) or not np.isfinite(twist).all():
                raise ValueError("This checkpoint requires one finite six-value camera-frame TCP twist.")
            normalized_twist = twist.copy()
            normalized_twist[:3] /= self.linear_action_scale_m_s
            normalized_twist[3:] /= self.angular_action_scale_rad_s
            normalized_twist = np.clip(normalized_twist, -5.0, 5.0)
        observation = assemble_policy_observation(
            live_rgbd,
            self.goal.goal_rgbd,
            self.previous_applied_action,
            policy_context_mode=self.policy_context_mode,
            normalized_tcp_twist_camera=normalized_twist,
            rotation_base_from_camera=rotation_base_from_camera,
        )
        with torch.inference_mode():
            output = self.model(
                {
                    "obs": observation,
                    "is_train": False,
                    "prev_actions": None,
                    "rnn_states": None,
                }
            )
        mus = output.get("mus") if isinstance(output, Mapping) else None
        if not isinstance(mus, torch.Tensor) or tuple(mus.shape) != (1, POLICY_OUTPUT_SIZE):
            raise RuntimeError(f"Policy returned an invalid deterministic output shape: {getattr(mus, 'shape', None)}")
        values = mus[0].detach().float().cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("Policy returned a non-finite deterministic output.")
        requested = tuple(float(value) for value in np.clip(values[:POLICY_MOTION_SIZE], -1.0, 1.0))
        filtered = slew_limit_normalized_action(
            requested,
            self.previous_applied_action,
            delta_limit=self.action_delta_limit,
        )
        return PolicyInference(
            requested_normalized_action=requested,  # type: ignore[arg-type]
            filtered_normalized_action=filtered,
            completion_probability=float(np.clip(values[POLICY_MOTION_SIZE], 0.0, 1.0)),
            valid_depth_fraction=float(valid.float().mean().item()),
        )

    def commit_applied_action(self, action: Sequence[float]) -> None:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (POLICY_MOTION_SIZE,) or not np.isfinite(values).all():
            raise ValueError("Applied action context must contain six finite values.")
        self.previous_applied_action = np.clip(values, -1.0, 1.0)

    def reset_action_context(self) -> None:
        self.previous_applied_action.fill(0.0)


def write_checkpoint_metadata_template(
    *,
    output_path: str | Path,
    checkpoint_path: str | Path,
    source_commit: str = "",
    completion_probability_threshold: float = 0.95,
    completion_required_consecutive_steps: int = 4,
    policy_context_mode: str = POLICY_CONTEXT_ACTION,
    policy_rate_hz: float = POLICY_RATE_HZ,
    action_delta_limit: float = 0.25,
) -> Path:
    output = Path(output_path)
    if not math.isfinite(float(policy_rate_hz)) or float(policy_rate_hz) <= 0.0:
        raise ValueError("policy_rate_hz must be finite and positive.")
    context_spec = resolve_policy_context(policy_context_mode)
    payload = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "camera_profile": D405_VISUAL_SERVO_CAMERA_PROFILE,
        "observation_profile": D405_VISUAL_SERVO_OBSERVATION_PROFILE,
        "network_input_size": policy_observation_size(
            context_spec.name,
            image_value_count=POLICY_IMAGE_VALUE_COUNT,
            privileged_label_size=POLICY_PRIVILEGED_PLACEHOLDER_SIZE,
        ),
        "network_output_size": POLICY_OUTPUT_SIZE,
        "policy_context_mode": context_spec.name,
        "policy_context_size": context_spec.size,
        "linear_action_scale_m_s": 0.04,
        "angular_action_scale_rad_s": 0.24,
        "action_delta_limit": float(action_delta_limit),
        "policy_rate_hz": float(policy_rate_hz),
        "completion_probability_threshold": float(completion_probability_threshold),
        "completion_required_consecutive_steps": int(completion_required_consecutive_steps),
        "source_commit": str(source_commit),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


__all__ = [
    "D405PolicyRuntime",
    "D405RuntimeGoal",
    "EXPECTED_D405_SERIAL",
    "GoalTarget",
    "POLICY_DEPLOYMENT_INPUT_SIZE",
    "POLICY_FULL_INPUT_SIZE",
    "POLICY_IMAGE_VALUE_COUNT",
    "POLICY_OUTPUT_SIZE",
    "PolicyInference",
    "assemble_policy_observation",
    "ros_depth_z16_to_metres",
    "sha256_file",
    "write_checkpoint_metadata_template",
]
