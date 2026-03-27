"""State-machine controller for a single cube pick attempt."""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from grasp_planning.grasping import GraspCandidate
from grasp_planning.planning.fr3_motion_context import tcp_pose_to_grasp_pose


def quat_xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternions from xyzw to wxyz convention."""

    return torch.cat((quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]), dim=-1)


def quat_slerp_wxyz(q1: torch.Tensor, q2: torch.Tensor, tau: float) -> torch.Tensor:
    """Spherical linear interpolation for quaternions in wxyz convention."""

    q1 = q1 / torch.linalg.norm(q1, dim=-1, keepdim=True).clamp_min(1e-8)
    q2 = q2 / torch.linalg.norm(q2, dim=-1, keepdim=True).clamp_min(1e-8)
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
    q2 = torch.where(dot < 0.0, -q2, q2)
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True).clamp(-1.0, 1.0)

    if float(torch.max(torch.abs(dot))) > 0.9995:
        result = q1 + tau * (q2 - q1)
        return result / torch.linalg.norm(result, dim=-1, keepdim=True).clamp_min(1e-8)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp_min(1e-8)
    theta = theta_0 * tau
    s1 = torch.sin(theta_0 - theta) / sin_theta_0
    s2 = torch.sin(theta) / sin_theta_0
    result = s1 * q1 + s2 * q2
    return result / torch.linalg.norm(result, dim=-1, keepdim=True).clamp_min(1e-8)


@dataclass(frozen=True)
class _PhaseDurations:
    settle_s: float = 0.25
    pregrasp_s: float = 2.5
    approach_s: float = 4.0
    close_s: float = 0.6
    hold_s: float = 0.25
    retreat_s: float = 4.0


class FR3PickController:
    """Execute a grasp using differential IK and simple gripper commands."""

    _EE_PATTERNS = (r"fr3_hand_tcp", r".*tcp.*", r".*hand.*", r".*gripper.*", r".*tool.*")
    _GRASP_TO_TCP_QUAT_WXYZ = (0.70710678, 0.0, 0.70710678, 0.0)
    _TCP_TO_GRASP_CENTER_OFFSET = (0.0, 0.0, -0.045)

    def __init__(
        self,
        *,
        robot,
        grasp: GraspCandidate,
        physics_dt: float,
        position_tolerance_m: float = 0.01,
        close_width: float = 0.0,
        max_joint_delta_rad: float = 0.02,
        start_phase: str = "settle",
    ) -> None:
        self._robot = robot
        self._grasp = grasp
        self._physics_dt = float(physics_dt)
        self._position_tolerance_m = float(position_tolerance_m)
        self._close_width = float(close_width)
        self._max_joint_delta_rad = float(max_joint_delta_rad)
        self._durations = _PhaseDurations()
        if start_phase not in {"settle", "pregrasp", "approach"}:
            raise ValueError(f"Unsupported start_phase '{start_phase}'.")
        self._phase = start_phase
        self._phase_elapsed_s = 0.0
        self._status = "running"
        self._ee_body_name, self._ee_body_idx = self._resolve_ee_body()
        self._ee_jacobi_body_idx = self._resolve_jacobi_body_idx(self._ee_body_idx)
        self._arm_joint_names, self._arm_joint_ids = self._resolve_joint_ids(r"fr3_joint[1-7]")
        self._hand_joint_names, self._hand_joint_ids = self._resolve_joint_ids(r"fr3_finger_joint[12]")
        self._ik_controller = self._build_ik_controller()
        self._phase_start_pos_w, self._phase_start_quat_w = self._current_grasp_pose_w()

    @property
    def status(self) -> str:
        return self._status

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def ee_body_name(self) -> str:
        return self._ee_body_name

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return self._arm_joint_names

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return self._hand_joint_names

    def step(self) -> str:
        """Advance the controller by one physics step."""

        self._phase_elapsed_s += self._physics_dt
        self._apply_gripper_command()

        if self._phase == "settle":
            if self._phase_elapsed_s >= self._durations.settle_s:
                self._transition("pregrasp")
        elif self._phase == "pregrasp":
            self._track_phase_pose(
                target_position_w=self._grasp.pregrasp_position_w,
                target_orientation_xyzw=self._grasp.orientation_xyzw,
                duration_s=self._durations.pregrasp_s,
            )
            if self._within_position_tolerance(self._grasp.pregrasp_position_w):
                self._transition("approach")
        elif self._phase == "approach":
            self._track_straight_line_phase(
                start_position_w=self._grasp.pregrasp_position_w,
                target_position_w=self._grasp.position_w,
                target_orientation_xyzw=self._grasp.orientation_xyzw,
                duration_s=self._durations.approach_s,
            )
            if self._within_position_tolerance(self._grasp.position_w):
                self._transition("close")
        elif self._phase == "close":
            self._track_pose(self._grasp.position_w, self._grasp.orientation_xyzw)
            if self._phase_elapsed_s >= self._durations.close_s:
                self._transition("hold")
        elif self._phase == "hold":
            self._track_pose(self._grasp.position_w, self._grasp.orientation_xyzw)
            if self._phase_elapsed_s >= self._durations.hold_s:
                self._transition("retreat")
        elif self._phase == "retreat":
            retreat_target = self._grasp.pregrasp_position_w
            self._track_straight_line_phase(
                start_position_w=self._grasp.position_w,
                target_position_w=retreat_target,
                target_orientation_xyzw=self._grasp.orientation_xyzw,
                duration_s=self._durations.retreat_s,
            )
            if self._within_position_tolerance(retreat_target):
                self._transition("done")
        elif self._phase == "done":
            self._track_pose(self._grasp.pregrasp_position_w, self._grasp.orientation_xyzw)
            self._status = "done"
        else:
            raise RuntimeError(f"Unknown controller phase '{self._phase}'.")

        return self._status

    def _transition(self, next_phase: str) -> None:
        self._phase = next_phase
        self._phase_elapsed_s = 0.0
        self._phase_start_pos_w, self._phase_start_quat_w = self._current_grasp_pose_w()

    def _apply_gripper_command(self) -> None:
        if self._hand_joint_ids.numel() == 0:
            return

        if self._phase in {"pregrasp", "approach"}:
            joint_target = self._grasp.gripper_width / 2.0
        else:
            joint_target = self._close_width

        hand_targets = torch.full(
            (1, int(self._hand_joint_ids.numel())),
            float(joint_target),
            device=self._device,
            dtype=torch.float32,
        )
        self._robot.set_joint_position_target(hand_targets, joint_ids=self._hand_joint_ids)

    def _track_pose(
        self,
        position_w: tuple[float, float, float],
        orientation_xyzw: tuple[float, float, float, float],
    ) -> None:
        from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms
        from isaaclab.utils.math import quat_apply, quat_mul

        desired_grasp_position_w = torch.tensor([position_w], dtype=torch.float32, device=self._device)
        desired_grasp_quat_w = quat_xyzw_to_wxyz(
            torch.tensor([orientation_xyzw], dtype=torch.float32, device=self._device)
        )
        grasp_to_tcp_quat_w = torch.tensor(
            [self._GRASP_TO_TCP_QUAT_WXYZ], dtype=torch.float32, device=self._device
        )
        tcp_to_grasp_center_b = torch.tensor(
            [self._TCP_TO_GRASP_CENTER_OFFSET], dtype=torch.float32, device=self._device
        )
        desired_tcp_quat_w = quat_mul(desired_grasp_quat_w, grasp_to_tcp_quat_w)
        desired_tcp_position_w = desired_grasp_position_w - quat_apply(desired_tcp_quat_w, tcp_to_grasp_center_b)

        ee_pose_w = self._robot.data.body_pose_w[:, self._ee_body_idx]
        ee_pos_w = ee_pose_w[:, :3]
        ee_quat_w = ee_pose_w[:, 3:7]
        root_pose_w = self._robot.data.root_pose_w
        root_pos_w = root_pose_w[:, :3]
        root_quat_w = root_pose_w[:, 3:7]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        desired_pos_b, desired_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, desired_tcp_position_w, desired_tcp_quat_w
        )
        desired_pose_b = torch.cat((desired_pos_b, desired_quat_b), dim=1)
        self._ik_controller.set_command(desired_pose_b)

        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_body_idx, :, self._arm_joint_ids]
        base_rot = matrix_from_quat(quat_inv(root_quat_w))
        jacobian[:, :3, :] = torch.bmm(base_rot, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot, jacobian[:, 3:, :])
        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        # Keep the Cartesian path smooth by limiting per-step joint changes.
        # This reduces branch-jumping where multiple joint solutions satisfy the same EE pose.
        joint_pos_des = self._limit_joint_delta(joint_pos, joint_pos_des, self._max_joint_delta_rad)
        self._robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_joint_ids)

    @staticmethod
    def _limit_joint_delta(
        joint_pos: torch.Tensor,
        joint_pos_des: torch.Tensor,
        max_joint_delta_rad: float,
    ) -> torch.Tensor:
        if max_joint_delta_rad <= 0.0:
            return joint_pos_des
        joint_delta = joint_pos_des - joint_pos
        joint_delta = torch.clamp(joint_delta, -max_joint_delta_rad, max_joint_delta_rad)
        return joint_pos + joint_delta

    def _track_phase_pose(
        self,
        *,
        target_position_w: tuple[float, float, float],
        target_orientation_xyzw: tuple[float, float, float, float],
        duration_s: float,
    ) -> None:
        interp_position_w, interp_orientation_xyzw = self._interpolate_phase_pose(
            target_position_w=target_position_w,
            target_orientation_xyzw=target_orientation_xyzw,
            duration_s=duration_s,
        )
        self._track_pose(interp_position_w, interp_orientation_xyzw)

    def _track_straight_line_phase(
        self,
        *,
        start_position_w: tuple[float, float, float],
        target_position_w: tuple[float, float, float],
        target_orientation_xyzw: tuple[float, float, float, float],
        duration_s: float,
    ) -> None:
        tau = 1.0 if duration_s <= 0.0 else min(max(self._phase_elapsed_s / duration_s, 0.0), 1.0)
        start_pos = torch.tensor([start_position_w], dtype=torch.float32, device=self._device)
        target_pos = torch.tensor([target_position_w], dtype=torch.float32, device=self._device)
        interp_pos = start_pos + tau * (target_pos - start_pos)
        self._track_pose(
            tuple(float(v) for v in interp_pos[0].tolist()),
            target_orientation_xyzw,
        )

    def _interpolate_phase_pose(
        self,
        *,
        target_position_w: tuple[float, float, float],
        target_orientation_xyzw: tuple[float, float, float, float],
        duration_s: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        tau = 1.0 if duration_s <= 0.0 else min(max(self._phase_elapsed_s / duration_s, 0.0), 1.0)
        start_pos = torch.tensor([self._phase_start_pos_w], dtype=torch.float32, device=self._device)
        target_pos = torch.tensor([target_position_w], dtype=torch.float32, device=self._device)
        interp_pos = start_pos + tau * (target_pos - start_pos)

        target_quat_w = quat_xyzw_to_wxyz(
            torch.tensor([target_orientation_xyzw], dtype=torch.float32, device=self._device)
        )
        interp_quat_w = quat_slerp_wxyz(self._phase_start_quat_w, target_quat_w, tau)
        interp_quat_xyzw = torch.cat((interp_quat_w[:, 1:4], interp_quat_w[:, 0:1]), dim=-1)
        return (
            tuple(float(v) for v in interp_pos[0].tolist()),
            tuple(float(v) for v in interp_quat_xyzw[0].tolist()),
        )

    def _current_tcp_pose_w(self) -> tuple[tuple[float, float, float], torch.Tensor]:
        ee_pose_w = self._robot.data.body_pose_w[:, self._ee_body_idx]
        ee_pos_w = tuple(float(v) for v in ee_pose_w[0, :3].tolist())
        ee_quat_w = ee_pose_w[:, 3:7].clone()
        return ee_pos_w, ee_quat_w

    def _current_grasp_pose_w(self) -> tuple[tuple[float, float, float], torch.Tensor]:
        tcp_position_w, tcp_quat_w = self._current_tcp_pose_w()
        tcp_orientation_xyzw = (
            float(tcp_quat_w[0, 1].item()),
            float(tcp_quat_w[0, 2].item()),
            float(tcp_quat_w[0, 3].item()),
            float(tcp_quat_w[0, 0].item()),
        )
        grasp_position_w, grasp_orientation_xyzw = tcp_pose_to_grasp_pose(
            position_w=tcp_position_w,
            orientation_xyzw=tcp_orientation_xyzw,
            grasp_to_tcp_quat_wxyz=self._GRASP_TO_TCP_QUAT_WXYZ,
            tcp_to_grasp_center_offset=self._TCP_TO_GRASP_CENTER_OFFSET,
        )
        grasp_quat_w = quat_xyzw_to_wxyz(
            torch.tensor([grasp_orientation_xyzw], dtype=torch.float32, device=self._device)
        )
        return grasp_position_w, grasp_quat_w

    def _within_position_tolerance(self, target_position_w: tuple[float, float, float]) -> bool:
        current_position_w, _current_quat_w = self._current_grasp_pose_w()
        current_position = torch.tensor(current_position_w, dtype=torch.float32, device=self._device)
        target = torch.tensor(target_position_w, dtype=torch.float32, device=self._device)
        error = torch.linalg.norm(current_position - target)
        return bool(error <= self._position_tolerance_m)

    def _build_ik_controller(self):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        return DifferentialIKController(cfg=cfg, num_envs=1, device=self._device)

    def _resolve_ee_body(self) -> tuple[str, int]:
        body_names = list(getattr(self._robot, "body_names", []))
        if not body_names:
            raise RuntimeError("Robot articulation does not expose body_names; cannot select an end-effector body.")

        for pattern in self._EE_PATTERNS:
            compiled = re.compile(pattern)
            for idx, name in enumerate(body_names):
                if compiled.fullmatch(name):
                    return name, idx

        raise RuntimeError(
            "Could not find an end-effector body. Available bodies: " + ", ".join(body_names)
        )

    def _resolve_jacobi_body_idx(self, body_idx: int) -> int:
        if getattr(self._robot, "is_fixed_base", False):
            return body_idx - 1
        return body_idx

    def _resolve_joint_ids(self, joint_pattern: str) -> tuple[tuple[str, ...], torch.Tensor]:
        joint_names = list(getattr(self._robot, "joint_names", []))
        matched_names = tuple(name for name in joint_names if re.fullmatch(joint_pattern, name))
        ids = [idx for idx, name in enumerate(joint_names) if re.fullmatch(joint_pattern, name)]
        if not ids:
            raise RuntimeError(f"Could not resolve joints matching '{joint_pattern}' on the articulation.")
        return matched_names, torch.tensor(ids, dtype=torch.long, device=self._device)

    @property
    def _device(self) -> str:
        return str(self._robot.device)
