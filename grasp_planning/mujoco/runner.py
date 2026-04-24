"""Minimal MuJoCo grasp-validation runtime for saved FR3 grasps."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspBundle,
    TriangleMesh,
    load_stl_mesh,
    quat_to_rotmat_xyzw,
    rotmat_to_quat_xyzw,
)
from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate
from grasp_planning.grasping.world_constraints import ObjectWorldPose

from .scene_builder import MujocoObjectSceneConfig, write_temporary_scene_xml


def _import_mujoco():
    try:
        import mujoco  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "MuJoCo is not installed in this environment. Install the `mujoco` Python package before running "
            "the MuJoCo grasp-validation scripts."
        ) from exc
    return mujoco


@dataclass(frozen=True)
class MujocoRobotConfig:
    """Robot-specific MuJoCo model bindings required for grasp execution."""

    robot_xml_path: str
    arm_joint_names: tuple[str, ...]
    arm_actuator_names: tuple[str, ...]
    gripper_actuator_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...] = ()
    ee_site_name: str | None = None
    ee_body_name: str | None = None
    home_joint_positions: dict[str, float] = field(default_factory=dict)
    open_gripper_ctrl: tuple[float, ...] = (0.04, 0.04)
    closed_gripper_ctrl: tuple[float, ...] = (0.0, 0.0)
    timestep: float = 0.002
    control_substeps: int = 8


@dataclass(frozen=True)
class MujocoExecutionConfig:
    """Execution and scoring parameters for one pickup attempt."""

    ik_max_iters: int = 200
    ik_damping: float = 1.0e-3
    ik_step_size: float = 0.7
    ik_position_tolerance_m: float = 0.003
    ik_orientation_tolerance_rad: float = 0.04
    settle_steps: int = 120
    close_steps: int = 240
    lift_height_m: float = 0.12
    hold_steps: int = 240
    trajectory_waypoints: int = 25
    waypoint_settle_steps: int = 45
    arm_speed_scale: float = 1.0
    success_height_margin_m: float = 0.05
    object_mass_kg: float = 0.15
    object_scale: float = 1.0
    object_friction: tuple[float, float, float] = (2.2, 0.08, 0.01)
    object_condim: int = 6
    object_solref: tuple[float, float] = (0.003, 1.0)
    object_solimp: tuple[float, float, float] = (0.95, 0.99, 0.001)
    object_margin: float = 0.001
    object_gap: float = 0.0005
    ground_friction: tuple[float, float, float] = (1.5, 0.04, 0.002)
    gripper_settle_position_delta_m: float = 1.0e-4
    gripper_settle_velocity_mps: float = 1.0e-3
    gripper_settle_consecutive_steps: int = 12


@dataclass(frozen=True)
class MujocoAttemptResult:
    """Summary of one MuJoCo pickup attempt."""

    success: bool
    status: str
    message: str
    pregrasp_reached: bool
    grasp_reached: bool
    final_object_position_world: tuple[float, float, float]
    initial_object_position_world: tuple[float, float, float]
    lift_height_m: float
    target_lift_height_m: float
    position_error_m: float | None = None
    orientation_error_rad: float | None = None
    generated_scene_xml: str | None = None


def load_robot_config(path: str | Path) -> MujocoRobotConfig:
    """Load a MuJoCo robot binding config from JSON."""

    cfg_path = Path(path).expanduser().resolve()
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    robot_xml_path = Path(str(raw["robot_xml_path"])).expanduser()
    if not robot_xml_path.is_absolute():
        robot_xml_path = (cfg_path.parent / robot_xml_path).resolve()
    return MujocoRobotConfig(
        robot_xml_path=str(robot_xml_path),
        ee_site_name=str(raw["ee_site_name"]) if raw.get("ee_site_name") is not None else None,
        ee_body_name=str(raw["ee_body_name"]) if raw.get("ee_body_name") is not None else None,
        arm_joint_names=tuple(str(name) for name in raw["arm_joint_names"]),
        arm_actuator_names=tuple(str(name) for name in raw.get("arm_actuator_names", raw["arm_joint_names"])),
        gripper_actuator_names=tuple(str(name) for name in raw.get("gripper_actuator_names", ())),
        gripper_joint_names=tuple(str(name) for name in raw.get("gripper_joint_names", ())),
        home_joint_positions={str(name): float(value) for name, value in raw.get("home_joint_positions", {}).items()},
        open_gripper_ctrl=tuple(float(v) for v in raw.get("open_gripper_ctrl", ())),
        closed_gripper_ctrl=tuple(float(v) for v in raw.get("closed_gripper_ctrl", ())),
        timestep=float(raw.get("timestep", 0.002)),
        control_substeps=int(raw.get("control_substeps", 8)),
    )


def build_bundle_local_mesh(bundle: SavedGraspBundle) -> TriangleMesh:
    """Return the bundle mesh in the same local frame used by saved grasps."""

    mesh_global = load_stl_mesh(bundle.target_stl_path, scale=bundle.stl_scale)
    rot = quat_to_rotmat_xyzw(bundle.local_frame_orientation_xyzw_world)
    translation = np.asarray(bundle.local_frame_origin_world, dtype=float)
    vertices_local = (np.asarray(mesh_global.vertices_obj, dtype=float) - translation[None, :]) @ rot
    return TriangleMesh(vertices_obj=vertices_local, faces=np.asarray(mesh_global.faces, dtype=np.int64))


def _rotation_error_rad(
    current_xyzw: tuple[float, float, float, float], target_xyzw: tuple[float, float, float, float]
) -> float:
    current_rot = quat_to_rotmat_xyzw(current_xyzw)
    target_rot = quat_to_rotmat_xyzw(target_xyzw)
    error_rot = target_rot @ current_rot.T
    trace = float(np.trace(error_rot))
    cosine = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    return float(np.arccos(cosine))


def _rotation_error_vector(
    current_xyzw: tuple[float, float, float, float], target_xyzw: tuple[float, float, float, float]
) -> np.ndarray:
    current_rot = quat_to_rotmat_xyzw(current_xyzw)
    target_rot = quat_to_rotmat_xyzw(target_xyzw)
    error_rot = target_rot @ current_rot.T
    skew = np.asarray(
        (
            error_rot[2, 1] - error_rot[1, 2],
            error_rot[0, 2] - error_rot[2, 0],
            error_rot[1, 0] - error_rot[0, 1],
        ),
        dtype=float,
    )
    sine = 0.5 * float(np.linalg.norm(skew))
    cosine = min(1.0, max(-1.0, 0.5 * (float(np.trace(error_rot)) - 1.0)))
    angle = float(np.arctan2(sine, cosine))
    if angle <= 1.0e-8:
        return 0.5 * skew
    axis = skew / max(2.0 * sine, 1.0e-8)
    return axis * angle


class MujocoPickupRuntime:
    """Internal helper that owns one MuJoCo scene and executes one world-frame grasp."""

    def __init__(
        self,
        *,
        robot_cfg: MujocoRobotConfig,
        execution_cfg: MujocoExecutionConfig,
        object_mesh_path: str | Path,
        object_pose_world: ObjectWorldPose,
        keep_generated_scene: bool,
    ) -> None:
        self._mujoco = _import_mujoco()
        self._robot_cfg = robot_cfg
        self._execution_cfg = execution_cfg
        self._keep_generated_scene = bool(keep_generated_scene)
        self._scene_xml_path = write_temporary_scene_xml(
            robot_xml_path=robot_cfg.robot_xml_path,
            object_mesh_path=object_mesh_path,
            object_pose_world=object_pose_world,
            object_scale=execution_cfg.object_scale,
            scene_cfg=MujocoObjectSceneConfig(
                object_mass_kg=execution_cfg.object_mass_kg,
                object_friction=execution_cfg.object_friction,
                object_condim=execution_cfg.object_condim,
                object_solref=execution_cfg.object_solref,
                object_solimp=execution_cfg.object_solimp,
                object_margin=execution_cfg.object_margin,
                object_gap=execution_cfg.object_gap,
                ground_friction=execution_cfg.ground_friction,
            ),
        )
        self._model = self._mujoco.MjModel.from_xml_path(str(self._scene_xml_path))
        self._model.opt.timestep = float(robot_cfg.timestep)
        self._data = self._mujoco.MjData(self._model)
        self._viewer = None
        self._viewer_realtime = True
        self._viewer_wall_start_s = 0.0
        self._viewer_sim_start_s = 0.0
        if robot_cfg.ee_site_name:
            self._site_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_SITE, robot_cfg.ee_site_name)
            if self._site_id < 0:
                raise RuntimeError(f"Could not resolve MuJoCo end-effector site '{robot_cfg.ee_site_name}'.")
        else:
            self._site_id = -1
        if robot_cfg.ee_body_name:
            self._ee_body_id = self._mujoco.mj_name2id(
                self._model, self._mujoco.mjtObj.mjOBJ_BODY, robot_cfg.ee_body_name
            )
            if self._ee_body_id < 0:
                raise RuntimeError(f"Could not resolve MuJoCo end-effector body '{robot_cfg.ee_body_name}'.")
        else:
            self._ee_body_id = -1
        if self._site_id < 0 and self._ee_body_id < 0:
            raise RuntimeError("MuJoCo robot config must define either `ee_site_name` or `ee_body_name`.")
        self._arm_qpos_indices = []
        self._arm_dof_indices = []
        self._arm_joint_lower = []
        self._arm_joint_upper = []
        for joint_name in robot_cfg.arm_joint_names:
            joint_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Could not resolve MuJoCo arm joint '{joint_name}'.")
            self._arm_qpos_indices.append(int(self._model.jnt_qposadr[joint_id]))
            self._arm_dof_indices.append(int(self._model.jnt_dofadr[joint_id]))
            if int(self._model.jnt_limited[joint_id]):
                self._arm_joint_lower.append(float(self._model.jnt_range[joint_id, 0]))
                self._arm_joint_upper.append(float(self._model.jnt_range[joint_id, 1]))
            else:
                self._arm_joint_lower.append(float("-inf"))
                self._arm_joint_upper.append(float("inf"))
        self._arm_qpos_indices = np.asarray(self._arm_qpos_indices, dtype=np.int32)
        self._arm_dof_indices = np.asarray(self._arm_dof_indices, dtype=np.int32)
        self._arm_joint_lower = np.asarray(self._arm_joint_lower, dtype=float)
        self._arm_joint_upper = np.asarray(self._arm_joint_upper, dtype=float)
        gripper_joint_names = tuple(robot_cfg.gripper_joint_names)
        if not gripper_joint_names:
            arm_joint_name_set = set(robot_cfg.arm_joint_names)
            gripper_joint_names = tuple(
                joint_name
                for joint_name in robot_cfg.home_joint_positions.keys()
                if joint_name not in arm_joint_name_set
            )
        self._gripper_joint_names = gripper_joint_names
        self._gripper_qpos_indices = []
        self._gripper_dof_indices = []
        for joint_name in self._gripper_joint_names:
            joint_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Could not resolve MuJoCo gripper joint '{joint_name}'.")
            self._gripper_qpos_indices.append(int(self._model.jnt_qposadr[joint_id]))
            self._gripper_dof_indices.append(int(self._model.jnt_dofadr[joint_id]))
        self._gripper_qpos_indices = np.asarray(self._gripper_qpos_indices, dtype=np.int32)
        self._gripper_dof_indices = np.asarray(self._gripper_dof_indices, dtype=np.int32)
        self._arm_actuator_ids = np.asarray(
            [
                int(self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, name))
                for name in robot_cfg.arm_actuator_names
            ],
            dtype=np.int32,
        )
        if np.any(self._arm_actuator_ids < 0):
            raise RuntimeError(f"Could not resolve all MuJoCo arm actuators: {robot_cfg.arm_actuator_names}")
        self._gripper_actuator_ids = np.asarray([], dtype=np.int32)
        if robot_cfg.gripper_actuator_names:
            self._gripper_actuator_ids = np.asarray(
                [
                    int(self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, name))
                    for name in robot_cfg.gripper_actuator_names
                ],
                dtype=np.int32,
            )
        if np.any(self._gripper_actuator_ids < 0):
            raise RuntimeError(f"Could not resolve all MuJoCo gripper actuators: {robot_cfg.gripper_actuator_names}")
        object_body_name = "target_object"
        self._object_body_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_BODY, object_body_name)
        if self._object_body_id < 0:
            raise RuntimeError(f"Could not resolve MuJoCo object body '{object_body_name}'.")

    def close(self) -> None:
        if not self._keep_generated_scene:
            try:
                os.unlink(self._scene_xml_path)
            except FileNotFoundError:
                pass

    @property
    def generated_scene_xml_path(self) -> str:
        return str(self._scene_xml_path)

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    def attach_viewer(self, viewer, *, realtime: bool = True) -> None:
        self._viewer = viewer
        self._viewer_realtime = bool(realtime)
        self._viewer_wall_start_s = time.perf_counter()
        self._viewer_sim_start_s = float(self._data.time)
        self._sync_viewer(force=True)

    def detach_viewer(self) -> None:
        self._viewer = None

    def hold_viewer_open(self, *, seconds: float | None = None) -> None:
        """Keep the passive viewer alive after execution for inspection."""

        if self._viewer is None:
            return
        if seconds is None:
            while self._viewer is not None:
                self._sync_viewer(force=True)
                time.sleep(0.02)
            return
        deadline_s = time.perf_counter() + max(0.0, float(seconds))
        while self._viewer is not None and time.perf_counter() < deadline_s:
            self._sync_viewer(force=True)
            time.sleep(0.02)

    def _sync_viewer(self, *, force: bool = False) -> None:
        if self._viewer is None:
            return
        try:
            is_running = getattr(self._viewer, "is_running", None)
            if callable(is_running) and not is_running():
                self.detach_viewer()
                return
            if self._viewer_realtime and not force:
                sim_elapsed_s = float(self._data.time) - float(self._viewer_sim_start_s)
                wall_elapsed_s = time.perf_counter() - float(self._viewer_wall_start_s)
                remaining_s = sim_elapsed_s - wall_elapsed_s
                if remaining_s > 0.0:
                    time.sleep(min(remaining_s, 0.02))
            self._viewer.sync()
        except Exception:
            self.detach_viewer()

    def _forward(self) -> None:
        self._mujoco.mj_forward(self._model, self._data)
        self._sync_viewer(force=True)

    def _step(self, steps: int) -> None:
        for _ in range(max(1, int(steps))):
            self._mujoco.mj_step(self._model, self._data)
            self._sync_viewer()

    def _scaled_arm_steps(self, base_steps: int) -> int:
        speed_scale = max(1.0e-6, float(self._execution_cfg.arm_speed_scale))
        return max(1, int(np.ceil(float(base_steps) / speed_scale)))

    def get_arm_qpos(self) -> np.ndarray:
        return np.asarray(self._data.qpos[self._arm_qpos_indices], dtype=float).copy()

    def get_gripper_qpos(self) -> np.ndarray:
        if self._gripper_qpos_indices.size == 0:
            return np.asarray([], dtype=float)
        return np.asarray(self._data.qpos[self._gripper_qpos_indices], dtype=float).copy()

    def get_gripper_qvel(self) -> np.ndarray:
        if self._gripper_dof_indices.size == 0:
            return np.asarray([], dtype=float)
        return np.asarray(self._data.qvel[self._gripper_dof_indices], dtype=float).copy()

    def gripper_state_summary(self) -> dict[str, object]:
        qpos = self.get_gripper_qpos()
        qvel = self.get_gripper_qvel()
        return {
            "joint_names": list(self._gripper_joint_names),
            "qpos": [float(v) for v in qpos],
            "qvel": [float(v) for v in qvel],
            "opening": float(np.sum(qpos)) if qpos.size else 0.0,
            "max_abs_velocity": float(np.max(np.abs(qvel))) if qvel.size else 0.0,
        }

    def _set_arm_ctrl(self, q_target: np.ndarray) -> None:
        self._data.ctrl[self._arm_actuator_ids] = np.asarray(q_target, dtype=float)

    def _set_gripper_ctrl(self, ctrl_values: tuple[float, ...]) -> None:
        if self._gripper_actuator_ids.size == 0:
            return
        values = np.asarray(ctrl_values, dtype=float)
        if values.size == 1 and self._gripper_actuator_ids.size > 1:
            values = np.repeat(values, self._gripper_actuator_ids.size)
        if values.size != self._gripper_actuator_ids.size:
            raise ValueError(f"Expected {self._gripper_actuator_ids.size} gripper control values, got {values.size}.")
        self._data.ctrl[self._gripper_actuator_ids] = values

    def _site_pose_from_data(self, data) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        if self._site_id >= 0:
            pos = np.asarray(data.site_xpos[self._site_id], dtype=float).copy()
            rot = np.asarray(data.site_xmat[self._site_id], dtype=float).reshape(3, 3)
        else:
            pos = np.asarray(data.xpos[self._ee_body_id], dtype=float).copy()
            rot = np.asarray(data.xmat[self._ee_body_id], dtype=float).reshape(3, 3)
        quat_xyzw = rotmat_to_quat_xyzw(rot)
        return pos, quat_xyzw

    def site_pose(self) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        return self._site_pose_from_data(self._data)

    def object_position_world(self) -> np.ndarray:
        return np.asarray(self._data.xpos[self._object_body_id], dtype=float).copy()

    def _apply_home_configuration(self) -> None:
        if not self._robot_cfg.home_joint_positions:
            return
        for joint_name, value in self._robot_cfg.home_joint_positions.items():
            joint_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Home configuration references unknown MuJoCo joint '{joint_name}'.")
            qpos_index = int(self._model.jnt_qposadr[joint_id])
            self._data.qpos[qpos_index] = float(value)
        self._forward()

    def settle_home(self) -> None:
        self._apply_home_configuration()
        if self._robot_cfg.home_joint_positions:
            home_targets = np.asarray(
                [self._robot_cfg.home_joint_positions[name] for name in self._robot_cfg.arm_joint_names],
                dtype=float,
            )
        else:
            home_targets = self.get_arm_qpos()
        self._set_gripper_ctrl(self._robot_cfg.open_gripper_ctrl)
        for _ in range(self._scaled_arm_steps(self._execution_cfg.settle_steps)):
            self._set_arm_ctrl(home_targets)
            self._set_gripper_ctrl(self._robot_cfg.open_gripper_ctrl)
            self._step(self._robot_cfg.control_substeps)

    def solve_ik(
        self,
        target_position_w: tuple[float, float, float],
        target_orientation_xyzw: tuple[float, float, float, float],
        *,
        gripper_ctrl: tuple[float, ...],
    ) -> tuple[np.ndarray | None, float, float]:
        # Solve IK on a scratch MuJoCo state so the visible robot does not "snap"
        # through the solver iterations before the slower planned trajectory starts.
        scratch = self._mujoco.MjData(self._model)
        scratch.qpos[:] = self._data.qpos
        scratch.qvel[:] = self._data.qvel
        if (
            hasattr(self._data, "act")
            and hasattr(scratch, "act")
            and self._data.act is not None
            and scratch.act is not None
        ):
            scratch.act[:] = self._data.act
        scratch.ctrl[:] = self._data.ctrl
        self._mujoco.mj_forward(self._model, scratch)
        jacp = np.zeros((3, self._model.nv), dtype=float)
        jacr = np.zeros((3, self._model.nv), dtype=float)
        damping = float(self._execution_cfg.ik_damping)
        for _ in range(max(1, int(self._execution_cfg.ik_max_iters))):
            current_pos, current_quat_xyzw = self._site_pose_from_data(scratch)
            pos_error = np.asarray(target_position_w, dtype=float) - current_pos
            rot_error = _rotation_error_vector(current_quat_xyzw, target_orientation_xyzw)
            pos_norm = float(np.linalg.norm(pos_error))
            rot_norm = float(np.linalg.norm(rot_error))
            if pos_norm <= float(self._execution_cfg.ik_position_tolerance_m) and rot_norm <= float(
                self._execution_cfg.ik_orientation_tolerance_rad
            ):
                return np.asarray(scratch.qpos[self._arm_qpos_indices], dtype=float).copy(), pos_norm, rot_norm

            self._mujoco.mj_jacSite(self._model, scratch, jacp, jacr, self._site_id)
            full_jac = np.vstack((jacp[:, self._arm_dof_indices], jacr[:, self._arm_dof_indices]))
            error = np.concatenate((pos_error, rot_error), axis=0)
            lhs = full_jac @ full_jac.T + (damping**2) * np.eye(6, dtype=float)
            dq = full_jac.T @ np.linalg.solve(lhs, error)
            q_current = np.asarray(scratch.qpos[self._arm_qpos_indices], dtype=float)
            q_target = q_current + float(self._execution_cfg.ik_step_size) * dq
            q_target = np.clip(q_target, self._arm_joint_lower, self._arm_joint_upper)
            scratch.qpos[self._arm_qpos_indices] = q_target
            if self._gripper_qpos_indices.size and len(gripper_ctrl) == self._gripper_qpos_indices.size:
                scratch.qpos[self._gripper_qpos_indices] = np.asarray(gripper_ctrl, dtype=float)
            self._mujoco.mj_forward(self._model, scratch)

        current_pos, current_quat_xyzw = self._site_pose_from_data(scratch)
        pos_norm = float(np.linalg.norm(np.asarray(target_position_w, dtype=float) - current_pos))
        rot_norm = _rotation_error_rad(current_quat_xyzw, target_orientation_xyzw)
        return None, pos_norm, rot_norm

    def execute_joint_target(self, q_goal: np.ndarray, *, gripper_ctrl: tuple[float, ...]) -> bool:
        q_start = self.get_arm_qpos()
        num_waypoints = max(2, int(self._execution_cfg.trajectory_waypoints))
        for step_idx in range(1, num_waypoints + 1):
            alpha = float(step_idx) / float(num_waypoints)
            q_target = (1.0 - alpha) * q_start + alpha * q_goal
            for _ in range(self._scaled_arm_steps(self._execution_cfg.waypoint_settle_steps)):
                self._set_arm_ctrl(q_target)
                self._set_gripper_ctrl(gripper_ctrl)
                self._step(self._robot_cfg.control_substeps)
        final_error = float(np.max(np.abs(self.get_arm_qpos() - q_goal)))
        return final_error <= 0.02

    def close_gripper(self) -> dict[str, object]:
        q_hold = self.get_arm_qpos()
        previous_qpos = None
        stable_steps = 0
        settled = False
        for _ in range(max(1, int(self._execution_cfg.close_steps))):
            self._set_arm_ctrl(q_hold)
            self._set_gripper_ctrl(self._robot_cfg.closed_gripper_ctrl)
            self._step(self._robot_cfg.control_substeps)
            qpos = self.get_gripper_qpos()
            qvel = self.get_gripper_qvel()
            if qpos.size:
                max_delta = 0.0 if previous_qpos is None else float(np.max(np.abs(qpos - previous_qpos)))
                max_velocity = float(np.max(np.abs(qvel))) if qvel.size else 0.0
                if max_delta <= float(self._execution_cfg.gripper_settle_position_delta_m) and max_velocity <= float(
                    self._execution_cfg.gripper_settle_velocity_mps
                ):
                    stable_steps += 1
                    if stable_steps >= max(1, int(self._execution_cfg.gripper_settle_consecutive_steps)):
                        settled = True
                        break
                else:
                    stable_steps = 0
                previous_qpos = qpos.copy()
        summary = self.gripper_state_summary()
        summary["settled"] = bool(settled or self._gripper_qpos_indices.size == 0)
        summary["stable_steps"] = int(stable_steps)
        return summary

    def hold_closed(self) -> None:
        q_hold = self.get_arm_qpos()
        for _ in range(max(1, int(self._execution_cfg.hold_steps))):
            self._set_arm_ctrl(q_hold)
            self._set_gripper_ctrl(self._robot_cfg.closed_gripper_ctrl)
            self._step(self._robot_cfg.control_substeps)

    def run(self, world_grasp: WorldFrameGraspCandidate) -> MujocoAttemptResult:
        self._forward()
        self.settle_home()
        initial_object_position_world = self.object_position_world()

        if self._gripper_actuator_ids.size == 0:
            return MujocoAttemptResult(
                success=False,
                status="no_gripper_configured",
                message=(
                    "The selected MuJoCo robot model has no gripper actuators configured. "
                    "This Menagerie FR3 asset supports reach validation, not pickup execution."
                ),
                pregrasp_reached=False,
                grasp_reached=False,
                initial_object_position_world=tuple(float(v) for v in initial_object_position_world),
                final_object_position_world=tuple(float(v) for v in initial_object_position_world),
                lift_height_m=0.0,
                target_lift_height_m=float(self._execution_cfg.success_height_margin_m),
                generated_scene_xml=self.generated_scene_xml_path if self._keep_generated_scene else None,
            )

        pregrasp_q, pregrasp_pos_err, pregrasp_rot_err = self.solve_ik(
            world_grasp.pregrasp_position_w,
            world_grasp.orientation_xyzw,
            gripper_ctrl=self._robot_cfg.open_gripper_ctrl,
        )
        if pregrasp_q is None or not self.execute_joint_target(
            pregrasp_q, gripper_ctrl=self._robot_cfg.open_gripper_ctrl
        ):
            return MujocoAttemptResult(
                success=False,
                status="pregrasp_failed",
                message=(
                    "Failed to reach pregrasp in MuJoCo. "
                    f"position_error={pregrasp_pos_err:.4f} orientation_error={pregrasp_rot_err:.4f}"
                ),
                pregrasp_reached=False,
                grasp_reached=False,
                initial_object_position_world=tuple(float(v) for v in initial_object_position_world),
                final_object_position_world=tuple(float(v) for v in self.object_position_world()),
                lift_height_m=0.0,
                target_lift_height_m=float(self._execution_cfg.success_height_margin_m),
                position_error_m=pregrasp_pos_err,
                orientation_error_rad=pregrasp_rot_err,
                generated_scene_xml=self.generated_scene_xml_path if self._keep_generated_scene else None,
            )

        grasp_q, grasp_pos_err, grasp_rot_err = self.solve_ik(
            world_grasp.position_w,
            world_grasp.orientation_xyzw,
            gripper_ctrl=self._robot_cfg.open_gripper_ctrl,
        )
        if grasp_q is None or not self.execute_joint_target(grasp_q, gripper_ctrl=self._robot_cfg.open_gripper_ctrl):
            return MujocoAttemptResult(
                success=False,
                status="grasp_failed",
                message=(
                    "Failed to reach grasp in MuJoCo. "
                    f"position_error={grasp_pos_err:.4f} orientation_error={grasp_rot_err:.4f}"
                ),
                pregrasp_reached=True,
                grasp_reached=False,
                initial_object_position_world=tuple(float(v) for v in initial_object_position_world),
                final_object_position_world=tuple(float(v) for v in self.object_position_world()),
                lift_height_m=0.0,
                target_lift_height_m=float(self._execution_cfg.success_height_margin_m),
                position_error_m=grasp_pos_err,
                orientation_error_rad=grasp_rot_err,
                generated_scene_xml=self.generated_scene_xml_path if self._keep_generated_scene else None,
            )

        self.close_gripper()
        lift_target = (
            float(world_grasp.position_w[0]),
            float(world_grasp.position_w[1]),
            float(world_grasp.position_w[2] + self._execution_cfg.lift_height_m),
        )
        lift_q, lift_pos_err, lift_rot_err = self.solve_ik(
            lift_target,
            world_grasp.orientation_xyzw,
            gripper_ctrl=self._robot_cfg.closed_gripper_ctrl,
        )
        if lift_q is not None:
            self.execute_joint_target(lift_q, gripper_ctrl=self._robot_cfg.closed_gripper_ctrl)
        self.hold_closed()

        final_object_position_world = self.object_position_world()
        lift_height_m = float(final_object_position_world[2] - initial_object_position_world[2])
        target_lift_height_m = float(self._execution_cfg.success_height_margin_m)
        success = lift_height_m >= target_lift_height_m
        status = "ok" if success else "lift_failed"
        message = (
            f"Object lifted by {lift_height_m:.4f} m (required {target_lift_height_m:.4f} m)."
            if success
            else f"Object only lifted by {lift_height_m:.4f} m (required {target_lift_height_m:.4f} m)."
        )
        return MujocoAttemptResult(
            success=success,
            status=status,
            message=message,
            pregrasp_reached=True,
            grasp_reached=True,
            initial_object_position_world=tuple(float(v) for v in initial_object_position_world),
            final_object_position_world=tuple(float(v) for v in final_object_position_world),
            lift_height_m=lift_height_m,
            target_lift_height_m=target_lift_height_m,
            position_error_m=lift_pos_err if lift_q is None else None,
            orientation_error_rad=lift_rot_err if lift_q is None else None,
            generated_scene_xml=self.generated_scene_xml_path if self._keep_generated_scene else None,
        )


def run_world_grasp_in_mujoco(
    *,
    robot_cfg: MujocoRobotConfig,
    execution_cfg: MujocoExecutionConfig,
    object_mesh_path: str | Path,
    object_pose_world: ObjectWorldPose,
    world_grasp: WorldFrameGraspCandidate,
    keep_generated_scene: bool = False,
    show_viewer: bool = False,
    viewer_left_ui: bool = False,
    viewer_right_ui: bool = False,
    viewer_realtime: bool = True,
    viewer_hold_seconds: float = 8.0,
    viewer_block_at_end: bool = False,
) -> MujocoAttemptResult:
    """Execute one world-frame grasp in MuJoCo and return a pickup result."""

    runtime = MujocoPickupRuntime(
        robot_cfg=robot_cfg,
        execution_cfg=execution_cfg,
        object_mesh_path=object_mesh_path,
        object_pose_world=object_pose_world,
        keep_generated_scene=keep_generated_scene,
    )
    try:
        if show_viewer:
            import mujoco.viewer  # type: ignore

            with mujoco.viewer.launch_passive(
                runtime.model,
                runtime.data,
                show_left_ui=viewer_left_ui,
                show_right_ui=viewer_right_ui,
            ) as viewer:
                runtime.attach_viewer(viewer, realtime=viewer_realtime)
                result = runtime.run(world_grasp)
                runtime.hold_viewer_open(seconds=None if viewer_block_at_end else viewer_hold_seconds)
                return result
        return runtime.run(world_grasp)
    finally:
        runtime.close()
