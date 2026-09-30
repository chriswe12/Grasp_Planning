"""Consistent symmetry potential and set-valued auxiliary supervision.

Axial extrema are optimized numerically over every interval of the circle,
not by reusing the readiness minimax representative. No pose is averaged.
"""

import torch
import torch.nn.functional as F

from .franka_symmetry import _mul, _rotate

OBJECTIVE = "orbit_potential_pose_set_v2"
AXIAL_INTERVALS = 64
AXIAL_ITERATIONS = 24


def potential(position, rotation, recipe):
    result = -recipe["position_progress_weight"] * position - recipe["rotation_progress_weight"] * rotation
    for error, prefix, unit in [(position, "position", "m"), (rotation, "rotation", "rad")]:
        result = result + recipe[prefix + "_precision_weight"] * torch.exp(
            -error / recipe[prefix + "_precision_scale_" + unit]
        )
    return result


def pose_distances(actual, goals):
    p = (actual[:, None, :3] - goals[..., :3]).norm(dim=-1)
    aq = F.normalize(actual[:, 3:], dim=-1)[:, None].expand_as(goals[..., 3:])
    dq = _mul(goals[..., 3:], torch.cat((aq[..., :1], -aq[..., 1:]), -1))
    r = 2 * torch.atan2(dq[..., 1:].norm(dim=-1), dq[..., 0].abs())
    return p, r


def minimize_orbit(evaluator, goal, index, cost):
    """Return detached argmin goal; caller evaluates its differentiable loss.

    Golden-section search on ALL 64 circular intervals (24 refinements), plus
    all grid endpoints and finite representatives. Approximate continuous
    optimization; tested against dense angular oracles. Fixed bounded work.
    """
    with torch.no_grad():
        lp = evaluator.positions[index].to(goal)
        lq = evaluator.quaternions[index].to(goal)
        gq = F.normalize(goal[:, 3:], dim=-1)[:, None].expand_as(lq)
        candidates = torch.cat((goal[:, None, :3] + _rotate(gq, lp), _mul(gq, lq)), -1)
        scores = cost(candidates).masked_fill(~evaluator.valid[index], torch.inf)
        ix = scores.argmin(-1)
        best = candidates[torch.arange(len(goal), device=goal.device), ix]
        best_cost = scores.min(-1).values
        if evaluator.axial is None:
            return best
        axes, centers, bases, valid = (x[index] for x in evaluator.axial)
        active = valid.any(-1)
        if not active.any():
            return best
        # Retain the batch shape so cost closures can broadcast actual/predicted poses.
        axes, centers = axes.to(goal), centers.to(goal)
        bp = candidates.gather(1, bases[..., None].expand(-1, -1, 7))
        n, m = valid.shape

        def poses(theta):
            axis = axes[:, :, None].expand(-1, -1, theta.shape[-1], -1)
            center = centers[:, :, None].expand_as(axis)
            q = torch.cat((torch.cos(theta / 2)[..., None], axis * torch.sin(theta / 2)[..., None]), -1)
            offset = center - _rotate(q, center)
            bq = bp[:, :, None, 3:].expand_as(q)
            return torch.cat((bp[:, :, None, :3] + _rotate(bq, offset), _mul(bq, q)), -1).reshape(n, -1, 7)

        def values(theta):
            return cost(poses(theta)).reshape(n, m, -1).masked_fill(~valid[..., None], torch.inf)

        grid = torch.arange(AXIAL_INTERVALS, device=goal.device, dtype=goal.dtype) * (2 * torch.pi / AXIAL_INTERVALS)
        lo = grid.expand(n, m, -1).clone()
        hi = lo + 2 * torch.pi / AXIAL_INTERVALS
        ratio = (5**0.5 - 1) / 2
        x = hi - ratio * (hi - lo)
        y = lo + ratio * (hi - lo)
        fx, fy = values(x), values(y)
        for _ in range(AXIAL_ITERATIONS):
            left = fx <= fy
            hi, lo = torch.where(left, y, hi), torch.where(left, lo, x)
            x, y = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
            fx, fy = values(x), values(y)
        theta = torch.cat((grid.expand(n, m, -1), (lo + hi) / 2), -1)
        scores = values(theta).reshape(n, -1)
        ix = scores.argmin(-1)
        candidate = poses(theta)[torch.arange(n, device=goal.device), ix]
        improve = scores.min(-1).values < best_cost
        return torch.where(improve[:, None], candidate, best)


def orbit_potential(evaluator, actual, goal, index, recipe):
    def cost(poses):
        return -potential(*pose_distances(actual, poses), recipe)

    selected = minimize_orbit(evaluator, goal, index, cost)
    return potential(*pose_distances(actual, selected[:, None]), recipe).squeeze(-1)


def pose_set_loss(prediction, target, evaluator, rotation_scale):
    """24 privileged values: TCP7, goal7, camera rotation9, normalized orbit ID.

    Head predicts camera-frame translation/axis-angle errors (six outputs).
    Compare its reconstructed goal pose with the closest valid goal using
    SmoothL1 translation and geodesic rotation. Set membership, not an arbitrary
    representative, supervises the head. Search is detached; loss retains grad.
    """
    actual, goal = target[:, :7], target[:, 7:14]
    camera = target[:, 14:23].reshape(-1, 3, 3)
    index = (target[:, 23] * len(evaluator.valid)).round().long().clamp(0, len(evaluator.valid) - 1)
    p = actual[:, :3] + (camera @ (prediction[:, :3] * 0.10)[..., None]).squeeze(-1)
    rv = (camera @ (prediction[:, 3:] * rotation_scale)[..., None]).squeeze(-1)
    angle = rv.norm(dim=-1, keepdim=True)
    dq = torch.cat((torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * torch.pi)) * rv), -1)
    predicted = torch.cat((p, _mul(dq, F.normalize(actual[:, 3:], dim=-1))), -1)

    def cost(poses):
        position = F.smooth_l1_loss(
            (predicted[:, None, :3] - poses[..., :3]) / 0.10, torch.zeros_like(poses[..., :3]), reduction="none"
        ).mean(-1)
        _, angle = pose_distances(predicted, poses)
        # Match the old three-component loss's small-angle curvature.
        rotation = F.smooth_l1_loss(angle / rotation_scale, torch.zeros_like(angle), reduction="none") / 3
        return position + rotation

    selected = minimize_orbit(evaluator, goal, index, cost)
    return cost(selected[:, None]).squeeze(-1)
