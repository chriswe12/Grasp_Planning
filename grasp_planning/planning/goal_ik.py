"""Goal-state IK for pose commands."""

from __future__ import annotations

import torch

from .fr3_motion_context import FR3MotionContext
from .types import PoseCommand


class GoalIKSolver:
    """Iteratively solve a target TCP pose into a joint target."""

    def __init__(
        self,
        context: FR3MotionContext,
        position_tolerance_m: float = 0.025,
        orientation_tolerance_rad: float = 0.08,
        fallback_position_tolerance_m: float = 0.045,
        fallback_orientation_tolerance_rad: float = 0.14,
        settle_steps_per_iter: int = 6,
    ) -> None:
        self._context = context
        self._position_tolerance_m = float(position_tolerance_m)
        self._orientation_tolerance_rad = float(orientation_tolerance_rad)
        self._fallback_position_tolerance_m = float(fallback_position_tolerance_m)
        self._fallback_orientation_tolerance_rad = float(fallback_orientation_tolerance_rad)
        self._settle_steps_per_iter = int(settle_steps_per_iter)

    def solve(self, cmd: PoseCommand, max_iters: int = 220, restore_start_state: bool = True) -> torch.Tensor | None:
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        q_start = self._context.get_arm_q()
        lower, upper = self._context.get_joint_limits()
        cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        ik_controller = DifferentialIKController(cfg=cfg, num_envs=1, device=self._context.device)

        best_q = q_start.clone()
        best_position_error = float("inf")
        best_orientation_error = float("inf")
        target_position_w = torch.tensor([cmd.position_w], dtype=torch.float32, device=self._context.device)
        target_orientation_xyzw = torch.tensor(
            [cmd.orientation_xyzw], dtype=torch.float32, device=self._context.device
        )

        for _ in range(max_iters):
            q_des = self._context.command_pose_via_differential_ik(ik_controller, cmd)
            if self._context.joint_limits_are_usable(lower, upper):
                q_des = torch.max(torch.min(q_des, upper), lower)
            self._context.hold_position(q_des, steps=self._settle_steps_per_iter)
            pos_error, rot_error = self._context.compute_pose_error(target_position_w, target_orientation_xyzw)
            pos_norm = float(torch.linalg.norm(pos_error).item())
            rot_norm = float(torch.linalg.norm(rot_error).item())
            if pos_norm + rot_norm < best_position_error + best_orientation_error:
                best_position_error = pos_norm
                best_orientation_error = rot_norm
                best_q = self._context.get_arm_q()
            if pos_norm <= self._position_tolerance_m and rot_norm <= self._orientation_tolerance_rad:
                goal_q = self._context.get_arm_q()
                if restore_start_state:
                    self._context.hold_position(q_start, steps=8)
                print(
                    f"[INFO]: IK converged with position_error={pos_norm:.4f} orientation_error={rot_norm:.4f}",
                    flush=True,
                )
                return goal_q

        if (
            best_position_error <= self._fallback_position_tolerance_m
            and best_orientation_error <= self._fallback_orientation_tolerance_rad
        ):
            if restore_start_state:
                self._context.hold_position(q_start, steps=8)
            print(
                "[WARN]: IK accepted approximate solution. "
                f"best_position_error={best_position_error:.4f} best_orientation_error={best_orientation_error:.4f}",
                flush=True,
            )
            return best_q

        if restore_start_state:
            self._context.hold_position(q_start, steps=8)
        print(
            "[WARN]: IK did not converge. "
            f"best_position_error={best_position_error:.4f} best_orientation_error={best_orientation_error:.4f}",
            flush=True,
        )
        return None
