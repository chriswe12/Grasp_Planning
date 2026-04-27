"""Geometry-based collision checks for the Franka scene."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .fr3_motion_context import FR3MotionContext


@dataclass(frozen=True)
class _ColliderPrim:
    """Collision-enabled geometry prim tracked for scene queries."""

    collider_path: str
    rigid_body_path: str


class CollisionChecker:
    """Geometry-based collision checker backed by PhysX scene queries."""

    def __init__(
        self,
        context: FR3MotionContext,
        cube,
        safety_margin_m: float = 0.0,
        plane_height_m: float = 0.0,
    ) -> None:
        self._context = context
        self._cube = cube
        self._safety_margin_m = float(safety_margin_m)
        self._plane_height_m = float(plane_height_m)
        self._query_interface = None
        self._robot_root_path: str | None = None
        self._cube_root_path: str | None = None
        self._robot_colliders: tuple[_ColliderPrim, ...] = ()
        self._robot_collision_queries_disabled = False

    def is_state_valid(self) -> tuple[bool, str]:
        q = self._context.get_arm_q()
        if not self._context.joint_state_within_limits(q):
            return False, "joint_limits"

        if self._safety_margin_m > 0.0:
            valid, reason = self._plane_margin_check()
            if not valid:
                return valid, reason

        self._ensure_scene_query_state()
        if self._robot_collision_queries_disabled:
            return True, "ok"
        for collider in self._robot_colliders:
            valid, reason = self._collider_is_valid(collider)
            if not valid:
                return False, reason
        return True, "ok"

    def is_edge_valid(self, q_start: torch.Tensor, q_goal: torch.Tensor, num_checks: int = 20) -> tuple[bool, str]:
        q_restore = self._context.get_arm_q()
        hand_restore = self._context.get_hand_q()
        total_checks = max(2, int(num_checks))
        for step_idx in range(total_checks + 1):
            alpha = float(step_idx) / float(total_checks)
            q_interp = (1.0 - alpha) * q_start + alpha * q_goal
            self._context.hold_position(q_interp, steps=2)
            valid, reason = self.is_state_valid()
            if not valid and reason == "plane_collision":
                print(
                    "[WARN]: Rechecking transient plane collision during edge validation "
                    f"at step={step_idx}/{total_checks}.",
                    flush=True,
                )
                self._context.hold_position(q_interp, steps=8)
                valid, reason = self.is_state_valid()
                if not valid and reason == "plane_collision":
                    print(
                        "[WARN]: Ignoring plane collision rejection during edge validation for debugging.",
                        flush=True,
                    )
                    valid, reason = True, "ok"
            if not valid:
                self._restore(q_restore, hand_restore)
                return False, reason
        self._restore(q_restore, hand_restore)
        return True, "ok"

    def _plane_margin_check(self) -> tuple[bool, str]:
        tcp_pos_w, _tcp_quat_w = self._context.get_tcp_pose_w()
        if float(torch.min(tcp_pos_w[:, 2]).item()) <= self._plane_height_m + self._safety_margin_m:
            return False, "plane_collision"
        return True, "ok"

    def _collider_is_valid(self, collider: _ColliderPrim) -> tuple[bool, str]:
        hits = self._query_overlap_hits(collider.collider_path)
        for hit in hits:
            collision_path = str(getattr(hit, "collision", "") or "")
            rigid_body_path = str(getattr(hit, "rigid_body", "") or "")
            reason = self._classify_hit(collider, collision_path, rigid_body_path)
            if reason is not None:
                return False, reason
        return True, "ok"

    def _classify_hit(self, collider: _ColliderPrim, collision_path: str, rigid_body_path: str) -> str | None:
        if not collision_path and not rigid_body_path:
            return None
        if collision_path == collider.collider_path:
            return None
        if rigid_body_path == collider.rigid_body_path:
            return None
        if self._robot_root_path and rigid_body_path.startswith(self._robot_root_path):
            return "self_collision"
        if self._robot_root_path and collision_path.startswith(self._robot_root_path):
            other_body = self._find_rigid_body_ancestor(collision_path)
            if other_body and other_body != collider.rigid_body_path:
                return "self_collision"
            return None
        if self._cube_root_path and (
            rigid_body_path.startswith(self._cube_root_path) or collision_path.startswith(self._cube_root_path)
        ):
            return "cube_collision"
        if "GroundPlane" in collision_path or "GroundPlane" in rigid_body_path:
            return "plane_collision"
        return "scene_collision"

    def _query_overlap_hits(self, collider_path: str) -> list[object]:
        import omni.usd
        from omni.physx import get_physx_scene_query_interface
        from pxr import PhysicsSchemaTools, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(collider_path)
        if not prim.IsValid():
            raise RuntimeError(f"Collider prim '{collider_path}' is not valid on the current USD stage.")
        if not prim.IsA(UsdGeom.Gprim):
            raise RuntimeError(f"Collider prim '{collider_path}' is not a UsdGeom.Gprim.")

        if self._query_interface is None:
            self._query_interface = get_physx_scene_query_interface()

        hits: list[object] = []

        def _report(hit) -> bool:
            hits.append(hit)
            return True

        path0, path1 = PhysicsSchemaTools.encodeSdfPath(collider_path)
        self._query_interface.overlap_shape(path0, path1, _report, False)
        return hits

    def _ensure_scene_query_state(self) -> None:
        if self._robot_colliders:
            return
        if self._robot_collision_queries_disabled:
            return

        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        self._robot_root_path = self._resolve_root_prim_path(self._context.robot)
        self._cube_root_path = self._resolve_root_prim_path(self._cube)
        robot_root_prim = stage.GetPrimAtPath(self._robot_root_path)
        if not robot_root_prim.IsValid():
            raise RuntimeError(f"Robot root prim '{self._robot_root_path}' is not valid on the current USD stage.")

        colliders: list[_ColliderPrim] = []
        for prim in Usd.PrimRange(robot_root_prim):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            rigid_body_ancestor = self._find_rigid_body_ancestor(str(prim.GetPath()))
            if rigid_body_ancestor is None:
                continue
            colliders.append(_ColliderPrim(collider_path=str(prim.GetPath()), rigid_body_path=rigid_body_ancestor))

        if not colliders:
            print(
                "[WARN]: Disabling robot collision queries because no collision-enabled GPrims were found under "
                f"robot root '{self._robot_root_path}'. Joint limits and optional plane-margin checks remain active.",
                flush=True,
            )
            self._robot_collision_queries_disabled = True
            return
        self._robot_colliders = tuple(colliders)

    def _resolve_root_prim_path(self, asset) -> str:
        prim_path = getattr(asset, "prim_path", None)
        if prim_path:
            return self._normalize_prim_path(str(prim_path))
        cfg = getattr(asset, "cfg", None)
        cfg_prim_path = getattr(cfg, "prim_path", None)
        if cfg_prim_path:
            return self._normalize_prim_path(str(cfg_prim_path))
        raise RuntimeError("Asset does not expose a resolved prim_path or cfg.prim_path.")

    def _normalize_prim_path(self, prim_path: str) -> str:
        normalized = prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        normalized = normalized.replace("env_.*", "env_0")
        normalized = normalized.replace("env_.*", "env_0")
        return normalized

    def _find_rigid_body_ancestor(self, prim_path: str) -> str | None:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        while prim and prim.IsValid():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(prim.GetPath())
            prim = prim.GetParent()
        return None

    def _restore(self, q_arm: torch.Tensor, q_hand: torch.Tensor) -> None:
        self._context.command_arm(q_arm)
        if self._context.hand_joint_ids.numel() > 0 and q_hand.numel() > 0:
            self._context.robot.set_joint_position_target(q_hand, joint_ids=self._context.hand_joint_ids)
        self._context.step_sim(steps=4)
