"""Object-local placement transforms and exact-USD batched Panda kinematics.

Ground truth is used only by offline reset construction, never by actor inputs or
canonical goal rendering. No simulator stepping occurs inside the IK solver.
"""

from copy import deepcopy

import torch

PLACEMENT_PROFILE = {
    "version": "independent_object_xy_yaw_v1",
    "world_xy_range_m": [[0.33, 0.58], [-0.185, 0.185]],
    "yaw_half_range_deg": 90.0,
    "goal_image": "unchanged_canonical_source",
    "robot_base_and_table": "fixed",
    "sampling": "shared_absolute_xy_region_for_every_target",
    "validation": "ik_and_12_physics_steps_start_and_goal",
}


def placement_transfer_allowed(source, target):
    """Explicit evaluation transfer only; never weaken normal checkpoint checks."""
    a, b = deepcopy(source), deepcopy(target)
    profile = b.pop("placement_randomization", None)
    return profile == PLACEMENT_PROFILE and a == b


def sample_placement_deltas(rng, object_xy):
    """Sample a shared world XY region, independent of the source target's XY."""
    import math

    import numpy as np

    bounds = PLACEMENT_PROFILE["world_xy_range_m"]
    yaw = math.radians(PLACEMENT_PROFILE["yaw_half_range_deg"])
    delta = rng.uniform(
        [bounds[0][0], bounds[1][0], -yaw],
        [bounds[0][1], bounds[1][1], yaw],
        (len(object_xy), 3),
    ).astype(np.float32)
    delta[:, :2] -= np.asarray(object_xy, dtype=np.float32)
    return delta


def transform_placement(pose, anchor, delta):
    """Apply yaw about the object's origin plus independent XY translation (WXYZ)."""
    c, s = torch.cos(delta[:, 2]), torch.sin(delta[:, 2])
    local = pose[:, :3] - anchor
    result = pose.clone()
    result[:, 0] = anchor[:, 0] + c * local[:, 0] - s * local[:, 1] + delta[:, 0]
    result[:, 1] = anchor[:, 1] + s * local[:, 0] + c * local[:, 1] + delta[:, 1]
    ch, sh = torch.cos(delta[:, 2] / 2), torch.sin(delta[:, 2] / 2)
    w, x, y, z = pose[:, 3:].unbind(-1)
    result[:, 3:] = torch.stack((ch * w - sh * z, ch * x - sh * y, ch * y + sh * x, ch * z + sh * w), -1)
    return result


def validate_placement_poses(value, valid):
    """Invalid bank slots may be zero; every selectable pose must be normalized."""
    import numpy as np

    if value.shape != (*valid.shape, 7) or not np.isfinite(value).all():
        raise ValueError("Invalid placement reset poses")
    if not np.allclose(np.linalg.norm(value[valid, 3:], axis=-1), 1, atol=1e-4):
        raise ValueError("Selectable placement quaternion is not normalized")


class PandaUsdKinematics:
    def __init__(self, env):
        from isaaclab.utils.math import matrix_from_quat
        from pxr import Usd, UsdPhysics

        self.device = env.device

        def tensor(x):
            return torch.tensor(x, device=self.device, dtype=torch.float32)

        def matrix(pos, rot):
            q = tensor([rot.GetReal(), *rot.GetImaginary()])
            m = torch.eye(4, device=self.device)
            m[:3, :3] = matrix_from_quat(q)
            m[:3, 3] = tensor(list(pos))
            return m

        joints = {}
        for prim in Usd.PrimRange(env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot")):
            if prim.IsA(UsdPhysics.Joint):
                j = UsdPhysics.Joint(prim)
                body = j.GetBody1Rel().GetTargets()
                if body:
                    joints[body[0].name] = (prim, j)
        chain = []
        body = "panda_hand"
        while body != "panda_link0":
            prim, j = joints[body]
            chain.append((prim, j))
            body = j.GetBody0Rel().GetTargets()[0].name
        self.chain = []
        for prim, j in reversed(chain):
            idx = int(prim.GetName().removeprefix("panda_joint")) - 1 if prim.IsA(UsdPhysics.RevoluteJoint) else None
            axis = None
            if idx is not None:
                axis = tensor(
                    [float(k == "XYZ".index(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())) for k in range(3)]
                )
            self.chain.append(
                (
                    matrix(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
                    torch.linalg.inv(matrix(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())),
                    idx,
                    axis,
                )
            )
        assert sorted(i for _, _, i, _ in self.chain if i is not None) == list(range(7))
        link = env.robot.find_bodies("panda_link0")[0][0]
        self.base = torch.eye(4, device=self.device)
        self.base[:3, :3] = matrix_from_quat(env.robot.data.body_quat_w[0, link])
        self.base[:3, 3] = env.robot.data.body_pos_w[0, link] - env.scene.env_origins[0]
        self.offset = env.tcp_offset[0].clone()

    def forward(self, q):
        from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

        n = len(q)
        T = self.base.expand(n, 4, 4).clone()
        points = []
        axes = []
        for a, b, idx, axis in self.chain:
            T = T @ a
            if idx is not None:
                points.append(T[:, :3, 3].clone())
                axes.append((T[:, :3, :3] @ axis).clone())
                quat = torch.cat((torch.cos(q[:, idx : idx + 1] / 2), torch.sin(q[:, idx : idx + 1] / 2) * axis), dim=1)
                R = torch.eye(4, device=q.device).repeat(n, 1, 1)
                R[:, :3, :3] = matrix_from_quat(quat)
                T = T @ R
            T = T @ b
        pos = T[:, :3, 3] + T[:, :3, :3] @ self.offset
        jac = torch.stack(
            [torch.cat((torch.cross(a, pos - p, dim=-1), a), dim=1) for p, a in zip(points, axes)], dim=-1
        )
        return torch.cat((pos, quat_from_matrix(T[:, :3, :3])), dim=-1), jac

    def solve(self, q, desired, limits, iterations=50):
        from isaaclab.utils.math import compute_pose_error

        from grasp_planning.rl.zed_mini import damped_joint_velocity

        q = q.clone()
        for _ in range(iterations):
            pose, jac = self.forward(q)
            p, r = compute_pose_error(
                pose[:, :3], pose[:, 3:], desired[:, :3], desired[:, 3:], rot_error_type="axis_angle"
            )
            if bool(((p.norm(dim=-1) < 0.0002) & (r.norm(dim=-1) < 0.002)).all()):
                break
            dq = damped_joint_velocity(jac, torch.cat((p, r), dim=1), 0.02).clamp(-0.12, 0.12)
            q = (q + dq).clamp(limits[..., 0] + 0.003, limits[..., 1] - 0.003)
        return q
