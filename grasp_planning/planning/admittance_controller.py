"""Isaac-side FR3 admittance controller inspired by the upstream ROS2 torque controller.

This version is intentionally adapted to Isaac Sim / Isaac Lab:
- it tracks a virtual Cartesian mass-spring-damper state in grasp coordinates,
- it uses Isaac Lab differential IK plus joint position targets as the inner loop,
- it estimates external wrench opportunistically from articulation data when available,
- it avoids ROS2, libfranka, and effort-control-only assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .fr3_motion_context import FR3MotionContext, quat_wxyz_to_xyzw
from .types import PlanResult, PoseCommand


def _clamp_unit_interval(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _normalize_quat_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    return quat_xyzw / torch.linalg.norm(quat_xyzw, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quat_conjugate_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    result = quat_xyzw.clone()
    result[..., :3] = -result[..., :3]
    return result


def _quat_mul_xyzw(q1_xyzw: torch.Tensor, q2_xyzw: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1_xyzw.unbind(dim=-1)
    x2, y2, z2, w2 = q2_xyzw.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _quat_slerp_xyzw(q1_xyzw: torch.Tensor, q2_xyzw: torch.Tensor, tau: float) -> torch.Tensor:
    q1 = _normalize_quat_xyzw(q1_xyzw)
    q2 = _normalize_quat_xyzw(q2_xyzw)
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
    q2 = torch.where(dot < 0.0, -q2, q2)
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True).clamp(-1.0, 1.0)

    if float(torch.max(torch.abs(dot))) > 0.9995:
        result = q1 + float(tau) * (q2 - q1)
        return _normalize_quat_xyzw(result)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp_min(1.0e-8)
    theta = theta_0 * float(tau)
    s1 = torch.sin(theta_0 - theta) / sin_theta_0
    s2 = torch.sin(theta) / sin_theta_0
    return _normalize_quat_xyzw(s1 * q1 + s2 * q2)


def _quat_to_rotation_matrix_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat_xyzw(quat_xyzw)
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = torch.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1)
    row1 = torch.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)), dim=-1)
    row2 = torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _rotation_error_vector_xyzw(current_xyzw: torch.Tensor, target_xyzw: torch.Tensor) -> torch.Tensor:
    current = _normalize_quat_xyzw(current_xyzw)
    target = _normalize_quat_xyzw(target_xyzw)
    if float(torch.sum(current * target).item()) < 0.0:
        target = -target
    error_quat = _quat_mul_xyzw(_quat_conjugate_xyzw(current), target)
    error_vec = error_quat[..., :3]
    current_rot = _quat_to_rotation_matrix_xyzw(current)
    return -torch.bmm(current_rot, error_vec.unsqueeze(-1)).squeeze(-1)


def _integrate_orientation_xyzw(quat_xyzw: torch.Tensor, angular_velocity: torch.Tensor, dt: float) -> torch.Tensor:
    angle = torch.linalg.norm(angular_velocity, dim=-1, keepdim=True) * float(dt)
    axis = angular_velocity / torch.linalg.norm(angular_velocity, dim=-1, keepdim=True).clamp_min(1.0e-8)
    half_angle = 0.5 * angle
    delta = torch.cat((axis * torch.sin(half_angle), torch.cos(half_angle)), dim=-1)
    delta = torch.where((angle > 1.0e-8).expand_as(delta), delta, torch.tensor([0.0, 0.0, 0.0, 1.0], device=quat_xyzw.device))
    return _normalize_quat_xyzw(_quat_mul_xyzw(delta, quat_xyzw))


def _diag_tensor(values: tuple[float, ...], device: str) -> torch.Tensor:
    return torch.diag(torch.tensor(values, dtype=torch.float32, device=device))


@dataclass(frozen=True)
class AdmittanceControllerCfg:
    """Controller gains and execution parameters."""

    stiffness_diag: tuple[float, float, float, float, float, float] = (320.0, 320.0, 320.0, 10.0, 10.0, 8.0)
    inertia_diag: tuple[float, float, float, float, float, float] = (3.0, 1.8, 2.4, 0.075, 0.2, 0.001)
    reference_filter: float = 0.05
    wrench_filter: float = 0.15
    use_estimated_external_wrench: bool = False
    max_linear_speed_mps: float = 0.35
    max_angular_speed_radps: float = 1.8
    max_duration_s: float = 12.0
    inner_settle_steps: int = 6
    settle_steps: int = 10
    position_tolerance_m: float = 0.015
    orientation_tolerance_rad: float = 0.10


@dataclass(frozen=True)
class AdmittanceStepState:
    """Pure virtual-state snapshot used for tests and debug."""

    position_w: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    twist: tuple[float, float, float, float, float, float]


def integrate_admittance_step(
    *,
    state: AdmittanceStepState,
    target_position_w: tuple[float, float, float],
    target_orientation_xyzw: tuple[float, float, float, float],
    external_wrench: tuple[float, float, float, float, float, float],
    stiffness_diag: tuple[float, float, float, float, float, float],
    inertia_diag: tuple[float, float, float, float, float, float],
    dt: float,
) -> AdmittanceStepState:
    """Advance the virtual admittance state by one step.

    This mirrors the upstream outer-loop mass-spring-damper law, but stays independent
    of Isaac-specific APIs so it can be unit-tested in this repository.
    """

    device = "cpu"
    stiffness = _diag_tensor(stiffness_diag, device)
    inertia = _diag_tensor(inertia_diag, device)
    damping = 2.05 * torch.diag(torch.sqrt(torch.diag(stiffness) * torch.diag(inertia)))

    position = torch.tensor([state.position_w], dtype=torch.float32, device=device)
    orientation = torch.tensor([state.orientation_xyzw], dtype=torch.float32, device=device)
    twist = torch.tensor([state.twist], dtype=torch.float32, device=device)
    target_position = torch.tensor([target_position_w], dtype=torch.float32, device=device)
    target_orientation = torch.tensor([target_orientation_xyzw], dtype=torch.float32, device=device)
    wrench = torch.tensor([external_wrench], dtype=torch.float32, device=device)

    linear_error = position - target_position
    linear_twist = twist[:, :3]
    linear_wrench = wrench[:, :3]
    linear_stiffness = stiffness[:3, :3]
    linear_inertia = inertia[:3, :3]
    linear_damping = damping[:3, :3]

    linear_accel = torch.linalg.solve(
        linear_inertia,
        (linear_wrench - linear_twist @ linear_damping.T - linear_error @ linear_stiffness.T).T,
    ).T
    twist_next = twist.clone()
    twist_next[:, 3:] = 0.0
    twist_next[:, :3] = linear_twist + linear_accel * float(dt)
    position_next = position + twist_next[:, :3] * float(dt)
    orientation_next = _quat_slerp_xyzw(orientation, target_orientation, min(1.0, 6.0 * float(dt)))
    return AdmittanceStepState(
        position_w=tuple(float(v) for v in position_next[0].tolist()),
        orientation_xyzw=tuple(float(v) for v in orientation_next[0].tolist()),
        twist=tuple(float(v) for v in twist_next[0].tolist()),
    )


class FR3AdmittanceController:
    """Isaac Sim admittance controller for FR3 based on virtual Cartesian compliance."""

    def __init__(
        self,
        *,
        robot,
        scene,
        sim,
        fixed_gripper_width: float = 0.04,
        cfg: AdmittanceControllerCfg | None = None,
    ) -> None:
        self._context = FR3MotionContext(
            robot=robot,
            scene=scene,
            sim=sim,
            fixed_gripper_width=fixed_gripper_width,
        )
        self._cfg = cfg or AdmittanceControllerCfg()
        self._stiffness = _diag_tensor(self._cfg.stiffness_diag, self._context.device)
        self._inertia = _diag_tensor(self._cfg.inertia_diag, self._context.device)
        self._damping = 2.05 * torch.diag(torch.sqrt(torch.diag(self._stiffness) * torch.diag(self._inertia)))
        self._twist = torch.zeros((1, 6), dtype=torch.float32, device=self._context.device)
        self._filtered_wrench = torch.zeros((1, 6), dtype=torch.float32, device=self._context.device)
        self._target_position_w, self._target_orientation_xyzw = self._get_current_grasp_pose()
        self._reference_position = torch.tensor(
            [self._target_position_w], dtype=torch.float32, device=self._context.device
        )
        self._reference_orientation = torch.tensor(
            [self._target_orientation_xyzw], dtype=torch.float32, device=self._context.device
        )
        self._virtual_position = self._reference_position.clone()
        self._virtual_orientation = self._reference_orientation.clone()
        self._ik_controller = self._build_ik_controller()

    @property
    def ee_body_name(self) -> str:
        return self._context.ee_body_name

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return self._context.arm_joint_names

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return self._context.hand_joint_names

    def get_current_tcp_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        tcp_pos_w, tcp_quat_w = self._context.get_tcp_pose_w()
        pos = tuple(float(v) for v in tcp_pos_w[0].tolist())
        quat_wxyz = tcp_quat_w[0]
        quat_xyzw = (
            float(quat_wxyz[1].item()),
            float(quat_wxyz[2].item()),
            float(quat_wxyz[3].item()),
            float(quat_wxyz[0].item()),
        )
        return pos, quat_xyzw

    def move_to_pose(
        self,
        *,
        position_w: tuple[float, float, float],
        orientation_xyzw: tuple[float, float, float, float],
    ) -> PlanResult:
        self.set_target_pose(position_w=position_w, orientation_xyzw=orientation_xyzw)
        steps = max(1, int(math.ceil(self._cfg.max_duration_s / self._context.physics_dt)))
        stable_steps = 0
        target_position = torch.tensor([position_w], dtype=torch.float32, device=self._context.device)
        target_orientation = torch.tensor([orientation_xyzw], dtype=torch.float32, device=self._context.device)

        for step_idx in range(1, steps + 1):
            self.step()
            pos_error, rot_error = self._context.compute_pose_error(target_position, target_orientation)
            pos_norm = float(torch.linalg.norm(pos_error).item())
            rot_norm = float(torch.linalg.norm(rot_error).item())

            if step_idx == 1 or step_idx % 50 == 0:
                print(
                    "[INFO]: Admittance step "
                    f"{step_idx}/{steps} position_error={pos_norm:.4f} orientation_error={rot_norm:.4f}",
                    flush=True,
                )

            if pos_norm <= self._cfg.position_tolerance_m and rot_norm <= self._cfg.orientation_tolerance_rad:
                stable_steps += 1
                if stable_steps >= self._cfg.settle_steps:
                    return PlanResult(True, "ok", "Admittance controller reached the requested pose.")
            else:
                stable_steps = 0

        actual_tcp_position_w, actual_tcp_orientation_xyzw = self.get_current_tcp_pose()
        target_tcp_position_w, target_tcp_orientation_xyzw = self._context.grasp_pose_to_tcp_pose(
            position_w,
            orientation_xyzw,
        )
        print(
            "[WARN]: Admittance final TCP pose "
            f"target_position={target_tcp_position_w} target_orientation_xyzw={target_tcp_orientation_xyzw} "
            f"actual_position={actual_tcp_position_w} actual_orientation_xyzw={actual_tcp_orientation_xyzw} "
            f"position_error_xyz={tuple(float(v) for v in pos_error[0].tolist())} "
            f"orientation_error_xyz={tuple(float(v) for v in rot_error[0].tolist())}",
            flush=True,
        )

        return PlanResult(
            False,
            "execution_failed",
            (
                "Admittance controller did not settle within the configured timeout. "
                f"last_position_error={pos_norm:.4f} last_orientation_error={rot_norm:.4f}"
            ),
        )

    def set_target_pose(
        self,
        *,
        position_w: tuple[float, float, float],
        orientation_xyzw: tuple[float, float, float, float],
    ) -> None:
        self._target_position_w = tuple(float(v) for v in position_w)
        self._target_orientation_xyzw = tuple(float(v) for v in orientation_xyzw)

    def step(self, external_wrench: torch.Tensor | None = None) -> None:
        self._update_references()
        if external_wrench is not None:
            wrench = external_wrench
        elif self._cfg.use_estimated_external_wrench:
            wrench = self._estimate_external_wrench()
        else:
            wrench = torch.zeros((1, 6), dtype=torch.float32, device=self._context.device)
        if wrench.ndim == 1:
            wrench = wrench.unsqueeze(0)
        self._filtered_wrench = (
            (1.0 - self._cfg.wrench_filter) * self._filtered_wrench + self._cfg.wrench_filter * wrench.to(self._context.device)
        )

        position_error = self._virtual_position - self._reference_position
        linear_accel = torch.linalg.solve(
            self._inertia[:3, :3],
            (self._filtered_wrench[:, :3] - self._twist[:, :3] @ self._damping[:3, :3].T - position_error @ self._stiffness[:3, :3].T).T,
        ).T
        self._twist[:, :3] = self._twist[:, :3] + linear_accel * self._context.physics_dt
        self._twist[:, 3:] = 0.0
        self._twist[:, :3] = self._twist[:, :3].clamp(-self._cfg.max_linear_speed_mps, self._cfg.max_linear_speed_mps)
        self._virtual_position = self._virtual_position + self._twist[:, :3] * self._context.physics_dt
        self._virtual_orientation = _quat_slerp_xyzw(
            self._virtual_orientation,
            self._reference_orientation,
            min(1.0, self._cfg.max_angular_speed_radps * self._context.physics_dt),
        )

        cmd = PoseCommand(
            position_w=tuple(float(v) for v in self._virtual_position[0].tolist()),
            orientation_xyzw=tuple(float(v) for v in self._virtual_orientation[0].tolist()),
        )
        q_des = self._context.command_pose_via_differential_ik(self._ik_controller, cmd)
        lower, upper = self._context.get_joint_limits()
        if self._context.joint_limits_are_usable(lower, upper):
            q_des = torch.max(torch.min(q_des, upper), lower)
        self._context.hold_position(q_des, steps=self._cfg.inner_settle_steps)

    def _build_ik_controller(self):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        return DifferentialIKController(cfg=cfg, num_envs=1, device=self._context.device)

    def _get_current_grasp_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        tcp_position_w, tcp_quat_wxyz = self._context.get_tcp_pose_w()
        tcp_orientation_xyzw = quat_wxyz_to_xyzw(tcp_quat_wxyz)[0]
        return self._context.tcp_pose_to_grasp_pose(
            tuple(float(v) for v in tcp_position_w[0].tolist()),
            tuple(float(v) for v in tcp_orientation_xyzw.tolist()),
        )

    def _update_references(self) -> None:
        tau = _clamp_unit_interval(self._cfg.reference_filter)
        target_position = torch.tensor([self._target_position_w], dtype=torch.float32, device=self._context.device)
        target_orientation = torch.tensor(
            [self._target_orientation_xyzw], dtype=torch.float32, device=self._context.device
        )
        self._reference_position = self._reference_position + tau * (target_position - self._reference_position)
        self._reference_orientation = _quat_slerp_xyzw(self._reference_orientation, target_orientation, tau)

    def _estimate_external_wrench(self) -> torch.Tensor:
        force = self._extract_body_vector(("body_net_forces_w", "body_force_w", "body_external_force_w"), expected_width=3)
        torque = self._extract_body_vector(("body_net_torques_w", "body_torque_w", "body_external_torque_w"), expected_width=3)
        if force is None and torque is None:
            return torch.zeros((1, 6), dtype=torch.float32, device=self._context.device)
        if force is None:
            force = torch.zeros((1, 3), dtype=torch.float32, device=self._context.device)
        if torque is None:
            torque = torch.zeros((1, 3), dtype=torch.float32, device=self._context.device)
        return torch.cat((force, torque), dim=-1)

    def _extract_body_vector(self, attr_names: tuple[str, ...], expected_width: int) -> torch.Tensor | None:
        for attr_name in attr_names:
            value = getattr(self._context.robot.data, attr_name, None)
            if not isinstance(value, torch.Tensor):
                continue
            tensor = value.to(device=self._context.device, dtype=torch.float32)
            if tensor.ndim == 3 and tensor.shape[1] > self._context.ee_body_idx and tensor.shape[2] >= expected_width:
                return tensor[:, self._context.ee_body_idx, :expected_width]
            if tensor.ndim == 2 and tensor.shape[0] > self._context.ee_body_idx and tensor.shape[1] >= expected_width:
                return tensor[self._context.ee_body_idx : self._context.ee_body_idx + 1, :expected_width]
        return None
