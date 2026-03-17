"""State-machine controller for a single cube pick attempt."""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from grasp_planning.grasping import GraspCandidate


@dataclass(frozen=True)
class _PhaseDurations:
    settle_s: float = 0.25
    close_s: float = 0.6
    hold_s: float = 0.25


class FR3PickController:
    """Execute a grasp using differential IK and simple gripper commands."""

    _EE_PATTERNS = (r".*hand.*", r".*gripper.*", r".*tool.*")

    def __init__(
        self,
        *,
        robot,
        grasp: GraspCandidate,
        physics_dt: float,
        lift_height: float = 0.12,
        position_tolerance_m: float = 0.01,
        close_width: float = 0.0,
    ) -> None:
        self._robot = robot
        self._grasp = grasp
        self._physics_dt = float(physics_dt)
        self._lift_height = float(lift_height)
        self._position_tolerance_m = float(position_tolerance_m)
        self._close_width = float(close_width)
        self._durations = _PhaseDurations()
        self._phase = "pregrasp"
        self._phase_elapsed_s = 0.0
        self._status = "running"
        self._ee_body_name, self._ee_body_idx = self._resolve_ee_body()
        self._arm_joint_ids = self._resolve_joint_ids(r".*joint[1-7].*")
        self._hand_joint_ids = self._resolve_joint_ids(r".*finger.*")
        self._ik_controller = self._build_ik_controller()

    @property
    def status(self) -> str:
        return self._status

    @property
    def phase(self) -> str:
        return self._phase

    def step(self) -> str:
        """Advance the controller by one physics step."""

        self._phase_elapsed_s += self._physics_dt
        self._apply_gripper_command()

        if self._phase == "pregrasp":
            self._track_pose(self._grasp.pregrasp_position_w, self._grasp.orientation_xyzw)
            if self._within_position_tolerance(self._grasp.pregrasp_position_w):
                self._transition("approach")
        elif self._phase == "approach":
            self._track_pose(self._grasp.position_w, self._grasp.orientation_xyzw)
            if self._within_position_tolerance(self._grasp.position_w):
                self._transition("close")
        elif self._phase == "close":
            if self._phase_elapsed_s >= self._durations.close_s:
                self._transition("hold")
        elif self._phase == "hold":
            if self._phase_elapsed_s >= self._durations.hold_s:
                self._transition("lift")
        elif self._phase == "lift":
            lift_target = (
                self._grasp.position_w[0],
                self._grasp.position_w[1],
                self._grasp.position_w[2] + self._lift_height,
            )
            self._track_pose(lift_target, self._grasp.orientation_xyzw)
            if self._within_position_tolerance(lift_target):
                self._transition("done")
        elif self._phase == "done":
            self._status = "done"
        else:
            raise RuntimeError(f"Unknown controller phase '{self._phase}'.")

        return self._status

    def _transition(self, next_phase: str) -> None:
        self._phase = next_phase
        self._phase_elapsed_s = 0.0

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
        desired_pose = torch.tensor(
            [[*position_w, *orientation_xyzw]],
            dtype=torch.float32,
            device=self._device,
        )
        self._ik_controller.set_command(desired_pose)

        ee_pose = self._robot.data.body_state_w[:, self._ee_body_idx, :7]
        ee_pos = ee_pose[:, :3]
        ee_quat = ee_pose[:, 3:7]
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._ee_body_idx - 1, :, self._arm_joint_ids]
        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        joint_pos_des = self._ik_controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
        self._robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_joint_ids)

    def _within_position_tolerance(self, target_position_w: tuple[float, float, float]) -> bool:
        current_position = self._robot.data.body_state_w[0, self._ee_body_idx, :3]
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

    def _resolve_joint_ids(self, joint_pattern: str) -> torch.Tensor:
        joint_names = list(getattr(self._robot, "joint_names", []))
        ids = [idx for idx, name in enumerate(joint_names) if re.fullmatch(joint_pattern, name)]
        if not ids and joint_pattern == r".*joint[1-7].*":
            raise RuntimeError("Could not find FR3 arm joints on the articulation.")
        return torch.tensor(ids, dtype=torch.long, device=self._device)

    @property
    def _device(self) -> str:
        return str(self._robot.device)
