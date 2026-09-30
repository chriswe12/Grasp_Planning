"""Object-symmetry-aware grasp scoring, independent of policy observations.

Finite rotations and surface-validated continuous axial rotations are admitted.
Equivalent TCPs form O S O^-1 G, not G S: an off-centre symmetry changes the
TCP position as well as its orientation. Position and angle must pass for the
same representative. This module does not infer symmetry from an image.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "franka_object_symmetry_v2"
MAX_GEOMETRY_ERROR_M = 0.0001
MAX_AXIAL_SURFACE_ERROR_M = 0.00015


def source_axial_symmetries(target, records, root=ROOT):
    """Validate provenance and express audited axes/centres in the bundle frame."""
    origin = target.get("source_frame_origin_obj_world", target.get("local_frame_origin_world"))
    q = target.get("source_frame_orientation_xyzw_obj_world", target.get("local_frame_orientation_xyzw_world"))
    frame = pose_matrix([*origin, q[3], *q[:3]])
    result = []
    for record in records:
        scale = float(target["mesh_scale"]) / record["mesh_scale"]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid symmetry mesh scale")
        tolerance = MAX_AXIAL_SURFACE_ERROR_M
        error = float(record["max_surface_error_m"]) * scale
        if not record["accepted"] or not np.isfinite(error) or error > tolerance:
            continue
        path = Path(root) / record["mesh_path"]
        if path.parts[-2:] != Path(target["mesh_path"]).parts[-2:]:
            raise ValueError("Axial symmetry mesh does not match target")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["mesh_sha256"]:
            raise ValueError(f"Axial symmetry mesh changed: {path}")
        axis = frame[:3, :3].T @ np.asarray(record["axis_obj"])
        center = frame[:3, :3].T @ (np.asarray(record["center_obj_m"]) * scale - frame[:3, 3])
        result.append(dict(**record, axis_source=axis.tolist(), center_source=center.tolist()))
    return result


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
        raise ValueError("Expected finite XYZ + WXYZ pose")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def rigid_matrix(value):
    value = np.asarray(value, dtype=float)
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Symmetry must be a finite proper rigid transform")
    return value


def source_symmetries(target, asset):
    """Scale asset translations and conjugate into the saved bundle frame."""
    part_id = Path(target["mesh_path"]).stem
    part = asset.get("parts", {}).get(part_id)
    if asset.get("frame") != "object" or part is None:
        raise ValueError("Missing part or unsupported symmetry frame")
    scale = float(target["mesh_scale"]) / float(part.get("mesh_scale", asset["mesh_scale"]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid symmetry mesh scale")
    origin = target.get("source_frame_origin_obj_world", target.get("local_frame_origin_world"))
    q = target.get("source_frame_orientation_xyzw_obj_world", target.get("local_frame_orientation_xyzw_world"))
    obj_from_source = pose_matrix([*origin, q[3], *q[:3]])
    matrices, names, excluded = [np.eye(4)], ["identity"], []
    for record in part.get("symmetries", []):
        matrix = rigid_matrix(record["matrix_obj"]).copy()
        if np.allclose(matrix, np.eye(4), atol=1e-8, rtol=0):
            continue
        validation = record.get("validation", {})
        # Existing geometric assets can accept millimetre-scale near-symmetries.
        # Those must not silently enlarge a 4 mm success region.
        error = float(validation.get("vertex_max_m", float("inf"))) * scale
        if (
            record.get("type") != "finite_rotation"
            or not validation.get("accepted")
            or not np.isfinite(error)
            or error > MAX_GEOMETRY_ERROR_M
        ):
            excluded.append(record["name"])
            continue
        matrix[:3, 3] *= scale
        matrix = np.linalg.inv(obj_from_source) @ matrix @ obj_from_source
        if not any(np.allclose(matrix, prior, atol=1e-7, rtol=0) for prior in matrices):
            matrices.append(matrix)
            names.append(record["name"])
    return matrices, dict(
        names=names, excluded=excluded, continuous_hints_not_enabled=len(part.get("continuous_symmetries", []))
    )


def goal_relative_symmetries(object_pose, goal_pose, symmetries):
    obj, goal = pose_matrix(object_pose), pose_matrix(goal_pose)
    return np.stack([np.linalg.inv(goal) @ obj @ s @ np.linalg.inv(obj) @ goal for s in symmetries])


def catalog_symmetries(data, selected, root=ROOT):
    """Build target-specific TCP orbits with source hashes and coverage evidence."""
    root = Path(root)
    axial_path = root / "assets/obj/fabrica/continuous_symmetries.json"
    axial_bytes = axial_path.read_bytes()
    axial_asset = json.loads(axial_bytes)
    sources = {}
    for value, digest in zip(data["source_bundle_paths"], data["source_bundle_sha256"]):
        path = Path(str(value))
        # Canonical stage-2 layout: parts/<assembly>/<part>/orientations/<orientation>/stage2.json.
        key = (path.parents[3].name + "__part_" + path.parents[2].name, path.parent.name)
        resolved = path if path.is_absolute() else root / path
        sources.setdefault(key, []).append((resolved, str(digest)))
    cache, asset_cache, orbits, reports = {}, {}, [], []
    for row in selected:
        key = (str(data["part_keys"][row]), str(data["orientation_ids"][row]))
        if key not in cache:
            target = None
            for path, digest in sources[key]:
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError(f"Source bundle changed: {path}")
                current = json.loads(content)["target"]
                if target is not None and target != current:
                    raise ValueError(f"Ambiguous source mesh/frame for {key}")
                target = current
            assembly = key[0].split("__part_")[0]
            asset_path = root / "assets/obj/fabrica" / assembly / "symmetries.json"
            if asset_path not in asset_cache:
                content = asset_path.read_bytes()
                asset_cache[asset_path] = json.loads(content), hashlib.sha256(content).hexdigest()
            asset, asset_hash = asset_cache[asset_path]
            if Path(target["mesh_path"]).parent.name != assembly:
                raise ValueError("Source mesh and symmetry assembly disagree")
            transforms, info = source_symmetries(target, asset)
            axial = source_axial_symmetries(target, axial_asset["parts"].get(key[0], []), root)
            finite = source_axial_symmetries(target, axial_asset.get("finite_rotations", {}).get(key[0], []), root)
            for record in finite:
                r = Rotation.from_rotvec(
                    np.asarray(record["axis_source"]) * np.deg2rad(record["angle_deg"])
                ).as_matrix()
                c = np.asarray(record["center_source"])
                t = np.eye(4)
                t[:3, :3], t[:3, 3] = r, c - r @ c
                if not any(np.allclose(t, prior, atol=1e-6, rtol=0) for prior in transforms):
                    transforms.append(t)
                    info["names"].append(record["name"])
            info["continuous_source"] = axial
            info["surface_finite_names"] = [r["name"] for r in finite]
            info["continuous_hints_not_enabled"] -= len(axial)
            info.update(
                part=key[0],
                sources=[dict(path=str(p), sha256=h) for p, h in sources[key]],
                symmetry_asset=str(asset_path),
                symmetry_sha256=asset_hash,
            )
            cache[key] = transforms, info
        transforms, info = cache[key]
        orbits.append(goal_relative_symmetries(data["object_poses"][row], data["goal_poses"][row], transforms))
        goal_from_source = np.linalg.inv(pose_matrix(data["goal_poses"][row])) @ pose_matrix(data["object_poses"][row])
        continuous = [
            dict(
                axis=(goal_from_source[:3, :3] @ r["axis_source"]).tolist(),
                center=(goal_from_source[:3, :3] @ r["center_source"] + goal_from_source[:3, 3]).tolist(),
            )
            for r in info["continuous_source"]
        ]
        reports.append(dict(target_id=str(data["target_ids"][row]), continuous=continuous, **info))
    return orbits, dict(
        profile=PROFILE,
        max_geometry_error_m=MAX_GEOMETRY_ERROR_M,
        max_axial_surface_error_m=MAX_AXIAL_SURFACE_ERROR_M,
        axial_asset=str(axial_path),
        axial_sha256=hashlib.sha256(axial_bytes).hexdigest(),
        gripper_flip_enabled=False,
        targets=reports,
    )


def _mul(a, b):
    return torch.cat(
        (
            a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True),
            a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + torch.linalg.cross(a[..., 1:], b[..., 1:]),
        ),
        -1,
    )


def _rotate(q, v):
    uv = torch.linalg.cross(q[..., 1:], v)
    return v + 2 * (q[..., :1] * uv + torch.linalg.cross(q[..., 1:], uv))


class SymmetryEvaluator:
    """GPU-compatible finite and continuous axial pose-orbit distances."""

    def __init__(self, orbits, device="cpu", continuous=None):
        n, k = len(orbits), max(map(len, orbits))
        positions = np.zeros((n, k, 3))
        quaternions = np.zeros((n, k, 4))
        quaternions[..., 0] = 1
        valid = np.zeros((n, k), dtype=bool)
        for i, transforms in enumerate(orbits):
            transforms = np.asarray([rigid_matrix(t) for t in transforms])
            if not np.allclose(transforms[0], np.eye(4), atol=1e-6, rtol=0):
                raise ValueError("First orbit representative must be identity")
            positions[i, : len(transforms)] = transforms[:, :3, 3]
            quaternions[i, : len(transforms)] = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
            valid[i, : len(transforms)] = True
        self.positions = torch.tensor(positions, dtype=torch.float32, device=device)
        self.quaternions = torch.tensor(quaternions, dtype=torch.float32, device=device)
        self.valid = torch.tensor(valid, device=device)
        self.axial = None
        if continuous is not None and len(continuous) != n:
            raise ValueError("Continuous records must match the orbit target count")
        if continuous and any(continuous):
            # S(theta) D: include finite cosets, e.g. a cylinder's end-for-end flip.
            size = max(len(a) * len(o) for a, o in zip(continuous, orbits))
            axes, centers, bases, mask = (
                np.zeros((n, size, 3)),
                np.zeros((n, size, 3)),
                np.zeros((n, size), int),
                np.zeros((n, size), bool),
            )
            for i, records in enumerate(continuous):
                for j, (a, base) in enumerate((a, b) for a in records for b in range(len(orbits[i]))):
                    axis, center = np.asarray(a["axis"]), np.asarray(a["center"])
                    if not np.isfinite(axis).all() or not np.isfinite(center).all() or np.linalg.norm(axis) < 1e-8:
                        raise ValueError("Invalid continuous axis/centre")
                    # Conjugate the axis into each finite representative's frame.
                    t = orbits[i][base]
                    axes[i, j] = t[:3, :3].T @ (axis / np.linalg.norm(axis))
                    centers[i, j] = t[:3, :3].T @ (center - t[:3, 3])
                    bases[i, j], mask[i, j] = base, True
            self.axial = tuple(torch.tensor(x, device=device) for x in (axes, centers, bases, mask))

    def errors(self, tcp_pose, goal_pose, target_index):
        goal_q = torch.nn.functional.normalize(goal_pose[:, 3:], dim=-1)[:, None, :]
        local_p, local_q = (
            self.positions[target_index].to(goal_pose.dtype),
            self.quaternions[target_index].to(goal_pose.dtype),
        )
        goal_q = goal_q.expand_as(local_q)
        p = goal_pose[:, None, :3] + _rotate(goal_q, local_p)
        q = _mul(goal_q, local_q)
        tcp_q = torch.nn.functional.normalize(tcp_pose[:, 3:], dim=-1)[:, None, :].expand_as(q)
        inverse = torch.cat((tcp_q[..., :1], -tcp_q[..., 1:]), -1)
        delta = _mul(q, inverse)
        pe = (p - tcp_pose[:, None, :3]).norm(dim=-1)
        re = 2 * torch.atan2(delta[..., 1:].norm(dim=-1), delta[..., 0].abs())
        valid_pose = (tcp_pose[:, 3:].norm(dim=-1) > 1e-8) & (goal_pose[:, 3:].norm(dim=-1) > 1e-8)
        valid = self.valid[target_index] & torch.isfinite(pe) & torch.isfinite(re) & valid_pose[:, None]
        return pe.masked_fill(~valid, torch.inf), re.masked_fill(~valid, torch.inf)

    @staticmethod
    def select(pe, re, position_tolerance, rotation_tolerance):
        """Select one representative; never combine independent minima."""
        if position_tolerance <= 0 or rotation_tolerance <= 0:
            raise ValueError("Pose tolerances must be positive")
        score = torch.maximum(pe / position_tolerance, re / rotation_tolerance)
        index = score.argmin(-1)
        p = pe.gather(1, index[:, None]).squeeze(1)
        r = re.gather(1, index[:, None]).squeeze(1)
        return p, r, index, (p <= position_tolerance) & (r <= rotation_tolerance)

    def evaluate(self, tcp_pose, goal_pose, target_index, position_tolerance, rotation_tolerance, *, return_goal=False):
        """Paired minimax distance over finite representatives and entire circles.

        Each tolerance defines an allowed angular arc on the symmetry circle.
        Bisection finds their first intersection, avoiding an angular sample grid
        and never pairing position at one angle with orientation at another.
        """
        pe, re = self.errors(tcp_pose, goal_pose, target_index)

        def finish(pe, re, extra_pose=None):
            result = self.select(pe, re, position_tolerance, rotation_tolerance)
            if not return_goal:
                return result
            lq = self.quaternions[target_index].to(goal_pose.dtype)
            gq = torch.nn.functional.normalize(goal_pose[:, 3:], dim=-1)[:, None].expand_as(lq)
            candidates = torch.cat(
                (
                    goal_pose[:, None, :3] + _rotate(gq, self.positions[target_index].to(goal_pose.dtype)),
                    _mul(gq, lq),
                ),
                -1,
            )
            if extra_pose is not None:
                candidates = torch.cat((candidates, extra_pose.to(goal_pose.dtype)), 1)
            selected = candidates.gather(1, result[2][:, None, None].expand(-1, 1, 7)).squeeze(1)
            return (*result, selected)

        if self.axial is None:
            return finish(pe, re)
        if position_tolerance <= 0 or rotation_tolerance <= 0:
            raise ValueError("Pose tolerances must be positive")
        a, c, base, valid = (t[target_index] for t in self.axial)
        lp = self.positions[target_index].double().gather(1, base[..., None].expand(-1, -1, 3))
        lq = self.quaternions[target_index].double().gather(1, base[..., None].expand(-1, -1, 4))
        g = torch.nn.functional.normalize(goal_pose[:, 3:].double(), dim=-1)[:, None].expand_as(lq)
        bp = goal_pose[:, None, :3].double() + _rotate(g, lp)
        bq = _mul(g, lq)
        inv = torch.cat((bq[..., :1], -bq[..., 1:]), -1)
        x = _rotate(inv, tcp_pose[:, None, :3].double().expand_as(bp) - bp)
        tq = torch.nn.functional.normalize(tcp_pose[:, 3:].double(), dim=-1)[:, None].expand_as(bq)
        q = _mul(inv, tq)
        c = c - (c * a).sum(-1, keepdim=True) * a
        u, v, d = -c, -torch.linalg.cross(a, c), c - x
        ap = (d * d).sum(-1) + (c * c).sum(-1)
        bp_coeff, cp = 2 * (d * u).sum(-1), 2 * (d * v).sum(-1)
        w, b = q[..., 0], (q[..., 1:] * a).sum(-1)
        ar, br, cr = (w * w + b * b) / 2, (w * w - b * b) / 2, w * b

        def arc(bc, cc, rhs):
            amplitude = torch.hypot(bc, cc)
            ratio = rhs / amplitude.clamp_min(1e-30)
            full = (amplitude < 1e-20) & (rhs <= 0)
            half = torch.where(full, torch.pi, torch.acos(ratio.clamp(-1, 1)))
            return torch.atan2(cc, bc), half, (rhs <= amplitude + 1e-18)

        def intersection(s):
            pp, hp, vp = arc(-bp_coeff, -cp, ap - (s * position_tolerance).square())
            rp, hr, vr = arc(br, cr, torch.cos((s * rotation_tolerance).clamp(max=torch.pi) / 2).square() - ar)
            delta = torch.atan2(torch.sin(rp - pp), torch.cos(rp - pp))
            lower, upper = torch.maximum(-hp, delta - hr), torch.minimum(hp, delta + hr)
            return vp & vr & (lower <= upper), pp + (lower + upper) / 2

        lo = torch.zeros_like(ap)
        hi = (
            torch.maximum(
                x.norm(dim=-1) / position_tolerance,
                2 * torch.atan2(q[..., 1:].norm(dim=-1), q[..., 0].abs()) / rotation_tolerance,
            )
            + 1e-6
        )
        for _ in range(32):
            mid = (lo + hi) / 2
            ok, _ = intersection(mid)
            lo, hi = torch.where(ok, lo, mid), torch.where(ok, mid, hi)
        _, theta = intersection(hi)
        p = c + u * theta.cos()[..., None] + v * theta.sin()[..., None]
        sq = torch.cat((torch.cos(theta / 2)[..., None], a * torch.sin(theta / 2)[..., None]), -1)
        dq = _mul(torch.cat((sq[..., :1], -sq[..., 1:]), -1), q)
        ep, er = (p - x).norm(dim=-1), 2 * torch.atan2(dq[..., 1:].norm(dim=-1), dq[..., 0].abs())
        good_pose = (tcp_pose[:, 3:].norm(dim=-1) > 1e-8) & (goal_pose[:, 3:].norm(dim=-1) > 1e-8)
        valid = valid & good_pose[:, None] & torch.isfinite(ep) & torch.isfinite(er)
        return finish(
            torch.cat((pe.double(), ep.masked_fill(~valid, torch.inf)), -1),
            torch.cat((re.double(), er.masked_fill(~valid, torch.inf)), -1),
            torch.cat((bp + _rotate(bq, p), _mul(bq, sq)), -1) if return_goal else None,
        )
