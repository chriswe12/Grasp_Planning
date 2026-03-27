"""Shared FR3 articulation helpers for planning and execution."""

from __future__ import annotations

import re

import torch

from .types import PoseCommand


def quat_xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternions from xyzw to wxyz convention."""

    return torch.cat((quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]), dim=-1)


def quat_wxyz_to_xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert quaternions from wxyz to xyzw convention."""

    return torch.cat((quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]), dim=-1)


def _quat_mul_wxyz(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Multiply two quaternions stored in wxyz order."""

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_conjugate_wxyz(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Return the conjugate of a quaternion in wxyz order."""

    w, x, y, z = quat_wxyz
    return (w, -x, -y, -z)


def _quat_apply_wxyz(
    quat_wxyz: tuple[float, float, float, float],
    vec_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a 3D vector by a quaternion in wxyz order."""

    qx, qy, qz = quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    qw = quat_wxyz[0]
    vx, vy, vz = vec_xyz

    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def tcp_pose_to_grasp_pose(
    position_w: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
    *,
    grasp_to_tcp_quat_wxyz: tuple[float, float, float, float],
    tcp_to_grasp_center_offset: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert a TCP-frame pose into the grasp-frame pose consumed by the planner."""

    tcp_quat_wxyz = (
        float(orientation_xyzw[3]),
        float(orientation_xyzw[0]),
        float(orientation_xyzw[1]),
        float(orientation_xyzw[2]),
    )
    grasp_quat_wxyz = _quat_mul_wxyz(tcp_quat_wxyz, _quat_conjugate_wxyz(grasp_to_tcp_quat_wxyz))
    rotated_offset = _quat_apply_wxyz(tcp_quat_wxyz, tcp_to_grasp_center_offset)
    grasp_position_w = tuple(float(position_w[i]) + rotated_offset[i] for i in range(3))
    grasp_orientation_xyzw = (
        float(grasp_quat_wxyz[1]),
        float(grasp_quat_wxyz[2]),
        float(grasp_quat_wxyz[3]),
        float(grasp_quat_wxyz[0]),
    )
    return grasp_position_w, grasp_orientation_xyzw


def grasp_pose_to_tcp_pose(
    position_w: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
    *,
    grasp_to_tcp_quat_wxyz: tuple[float, float, float, float],
    tcp_to_grasp_center_offset: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert a grasp-frame pose into the TCP-frame pose commanded in the simulator."""

    grasp_quat_wxyz = (
        float(orientation_xyzw[3]),
        float(orientation_xyzw[0]),
        float(orientation_xyzw[1]),
        float(orientation_xyzw[2]),
    )
    tcp_quat_wxyz = _quat_mul_wxyz(grasp_quat_wxyz, grasp_to_tcp_quat_wxyz)
    rotated_offset = _quat_apply_wxyz(tcp_quat_wxyz, tcp_to_grasp_center_offset)
    tcp_position_w = tuple(float(position_w[i]) - rotated_offset[i] for i in range(3))
    tcp_orientation_xyzw = (
        float(tcp_quat_wxyz[1]),
        float(tcp_quat_wxyz[2]),
        float(tcp_quat_wxyz[3]),
        float(tcp_quat_wxyz[0]),
    )
    return tcp_position_w, tcp_orientation_xyzw


class FR3MotionContext:
    """Isaac-specific articulation accessors shared by motion components."""

    _EE_PATTERNS = (r"fr3_hand_tcp", r".*tcp.*", r".*hand.*", r".*gripper.*", r".*tool.*")
    _GRASP_TO_TCP_QUAT_WXYZ = (0.70710678, 0.0, -0.70710678, 0.0)
    _TCP_TO_GRASP_CENTER_OFFSET = (0.0, 0.0, -0.045)

    def __init__(self, *, robot, scene, sim, fixed_gripper_width: float = 0.04) -> None:
        self.robot = robot
        self.scene = scene
        self.sim = sim
        self.fixed_gripper_width = float(fixed_gripper_width)
        self.ee_body_name, self.ee_body_idx = self._resolve_ee_body()
        self.ee_jacobi_body_idx = self._resolve_jacobi_body_idx(self.ee_body_idx)
        self.arm_joint_names, self.arm_joint_ids = self._resolve_joint_ids(r"fr3_joint[1-7]")
        self.hand_joint_names, self.hand_joint_ids = self._resolve_joint_ids(r"fr3_finger_joint[12]")

    @property
    def device(self) -> str:
        return str(self.robot.device)

    @property
    def physics_dt(self) -> float:
        return float(self.sim.get_physics_dt())

    def get_arm_q(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.arm_joint_ids].clone()

    def get_hand_q(self) -> torch.Tensor:
        if self.hand_joint_ids.numel() == 0:
            return torch.zeros((1, 0), dtype=torch.float32, device=self.device)
        return self.robot.data.joint_pos[:, self.hand_joint_ids].clone()

    def get_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        joint_limits = getattr(self.robot.data, "joint_pos_limits", None)
        if joint_limits is None:
            joint_limits = getattr(self.robot.data, "joint_limits", None)
        if joint_limits is None:
            lower = torch.full((1, int(self.arm_joint_ids.numel())), -3.0, dtype=torch.float32, device=self.device)
            upper = torch.full((1, int(self.arm_joint_ids.numel())), 3.0, dtype=torch.float32, device=self.device)
            return lower, upper

        lower = joint_limits[:, self.arm_joint_ids, 0].clone().to(dtype=torch.float32)
        upper = joint_limits[:, self.arm_joint_ids, 1].clone().to(dtype=torch.float32)
        return lower, upper

    def joint_limits_are_usable(self, lower: torch.Tensor, upper: torch.Tensor) -> bool:
        span = upper - lower
        return bool(torch.all(torch.isfinite(lower)) and torch.all(torch.isfinite(upper)) and torch.all(span > 0.05))

    def joint_state_within_limits(self, q: torch.Tensor, tolerance: float = 5.0e-2) -> bool:
        lower, upper = self.get_joint_limits()
        if not self.joint_limits_are_usable(lower, upper):
            return True
        tol = float(tolerance)
        return bool(torch.all(q >= (lower - tol)) and torch.all(q <= (upper + tol)))

    def describe_joint_limit_state(self, q: torch.Tensor) -> str:
        lower, upper = self.get_joint_limits()
        span = upper - lower
        return (
            f"q={q[0].tolist()} lower={lower[0].tolist()} upper={upper[0].tolist()} span={span[0].tolist()} "
            f"usable_limits={self.joint_limits_are_usable(lower, upper)}"
        )

    def get_tcp_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_idx]
        return ee_pose_w[:, :3].clone(), ee_pose_w[:, 3:7].clone()

    def get_body_positions_w(self, body_names: tuple[str, ...]) -> dict[str, torch.Tensor]:
        positions: dict[str, torch.Tensor] = {}
        for name in body_names:
            body_idx = self.robot.body_names.index(name)
            positions[name] = self.robot.data.body_pose_w[:, body_idx, :3].clone()
        return positions

    @classmethod
    def tcp_pose_to_grasp_pose(
        cls,
        position_w: tuple[float, float, float],
        orientation_xyzw: tuple[float, float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Convert a TCP-frame pose into the corresponding grasp-frame target pose."""

        return tcp_pose_to_grasp_pose(
            position_w,
            orientation_xyzw,
            grasp_to_tcp_quat_wxyz=cls._GRASP_TO_TCP_QUAT_WXYZ,
            tcp_to_grasp_center_offset=cls._TCP_TO_GRASP_CENTER_OFFSET,
        )

    @classmethod
    def grasp_pose_to_tcp_pose(
        cls,
        position_w: tuple[float, float, float],
        orientation_xyzw: tuple[float, float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Convert a grasp-frame pose into the corresponding TCP-frame target pose."""

        return grasp_pose_to_tcp_pose(
            position_w,
            orientation_xyzw,
            grasp_to_tcp_quat_wxyz=cls._GRASP_TO_TCP_QUAT_WXYZ,
            tcp_to_grasp_center_offset=cls._TCP_TO_GRASP_CENTER_OFFSET,
        )

    def command_arm(self, q: torch.Tensor) -> None:
        self.robot.set_joint_position_target(q, joint_ids=self.arm_joint_ids)

    def command_fixed_gripper(self) -> None:
        if self.hand_joint_ids.numel() == 0:
            return
        targets = torch.full(
            (1, int(self.hand_joint_ids.numel())),
            self.fixed_gripper_width,
            dtype=torch.float32,
            device=self.device,
        )
        self.robot.set_joint_position_target(targets, joint_ids=self.hand_joint_ids)

    def step_sim(self, steps: int = 1) -> None:
        for _ in range(max(1, int(steps))):
            self.command_fixed_gripper()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.physics_dt)

    def hold_position(self, q: torch.Tensor, steps: int = 1) -> None:
        for _ in range(max(1, int(steps))):
            self.command_arm(q)
            self.command_fixed_gripper()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.physics_dt)

    def compute_pose_error(
        self,
        target_position_w: torch.Tensor,
        target_orientation_xyzw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from isaaclab.utils.math import quat_conjugate, quat_mul

        current_pos_w, current_quat_w = self.get_tcp_pose_w()
        desired_grasp_quat_w = quat_xyzw_to_wxyz(target_orientation_xyzw)
        grasp_to_tcp_quat_w = torch.tensor(
            [self._GRASP_TO_TCP_QUAT_WXYZ], dtype=torch.float32, device=self.device
        )
        tcp_to_grasp_center_b = torch.tensor(
            [self._TCP_TO_GRASP_CENTER_OFFSET], dtype=torch.float32, device=self.device
        )
        target_quat_w = quat_mul(desired_grasp_quat_w, grasp_to_tcp_quat_w)
        from isaaclab.utils.math import quat_apply
        target_tcp_position_w = target_position_w - quat_apply(target_quat_w, tcp_to_grasp_center_b)
        position_error = target_tcp_position_w - current_pos_w
        quat_error = quat_mul(target_quat_w, quat_conjugate(current_quat_w))
        orientation_error = 2.0 * quat_error[:, 1:4]
        return position_error, orientation_error

    def command_pose_via_differential_ik(self, ik_controller, cmd: PoseCommand) -> torch.Tensor:
        from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul, subtract_frame_transforms

        desired_grasp_position_w = torch.tensor([cmd.position_w], dtype=torch.float32, device=self.device)
        desired_grasp_quat_w = quat_xyzw_to_wxyz(
            torch.tensor([cmd.orientation_xyzw], dtype=torch.float32, device=self.device)
        )
        grasp_to_tcp_quat_w = torch.tensor(
            [self._GRASP_TO_TCP_QUAT_WXYZ], dtype=torch.float32, device=self.device
        )
        tcp_to_grasp_center_b = torch.tensor(
            [self._TCP_TO_GRASP_CENTER_OFFSET], dtype=torch.float32, device=self.device
        )
        desired_tcp_quat_w = quat_mul(desired_grasp_quat_w, grasp_to_tcp_quat_w)
        desired_tcp_position_w = desired_grasp_position_w - quat_apply(desired_tcp_quat_w, tcp_to_grasp_center_b)

        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_idx]
        ee_pos_w = ee_pose_w[:, :3]
        ee_quat_w = ee_pose_w[:, 3:7]
        root_pose_w = self.robot.data.root_pose_w
        root_pos_w = root_pose_w[:, :3]
        root_quat_w = root_pose_w[:, 3:7]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        desired_pos_b, desired_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, desired_tcp_position_w, desired_tcp_quat_w
        )
        desired_pose_b = torch.cat((desired_pos_b, desired_quat_b), dim=1)
        ik_controller.set_command(desired_pose_b)

        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_body_idx, :, self.arm_joint_ids]
        base_rot = matrix_from_quat(quat_inv(root_quat_w))
        jacobian[:, :3, :] = torch.bmm(base_rot, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot, jacobian[:, 3:, :])
        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.command_arm(joint_pos_des)
        return joint_pos_des

    def _resolve_ee_body(self) -> tuple[str, int]:
        body_names = list(getattr(self.robot, "body_names", []))
        if not body_names:
            raise RuntimeError("Robot articulation does not expose body_names; cannot select an end-effector body.")
        for pattern in self._EE_PATTERNS:
            compiled = re.compile(pattern)
            for idx, name in enumerate(body_names):
                if compiled.fullmatch(name):
                    return name, idx
        raise RuntimeError("Could not find an end-effector body. Available bodies: " + ", ".join(body_names))

    def _resolve_jacobi_body_idx(self, body_idx: int) -> int:
        if getattr(self.robot, "is_fixed_base", False):
            return body_idx - 1
        return body_idx

    def _resolve_joint_ids(self, joint_pattern: str) -> tuple[tuple[str, ...], torch.Tensor]:
        joint_names = list(getattr(self.robot, "joint_names", []))
        matched_names = tuple(name for name in joint_names if re.fullmatch(joint_pattern, name))
        ids = [idx for idx, name in enumerate(joint_names) if re.fullmatch(joint_pattern, name)]
        if not ids:
            raise RuntimeError(f"Could not resolve joints matching '{joint_pattern}' on the articulation.")
        return matched_names, torch.tensor(ids, dtype=torch.long, device=self.device)
