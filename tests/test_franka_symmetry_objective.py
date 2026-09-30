import numpy as np
import torch
from scipy.spatial.transform import Rotation

from grasp_planning.rl.franka_symmetry import SymmetryEvaluator
from grasp_planning.rl.franka_symmetry_objective import orbit_potential, pose_distances, pose_set_loss, potential

R = dict(
    position_progress_weight=20.0,
    rotation_progress_weight=2.0,
    position_precision_weight=3.0,
    rotation_precision_weight=1.0,
    position_precision_scale_m=0.005,
    rotation_precision_scale_rad=0.08,
)


def transform(deg, center=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("z", deg, degrees=True).as_matrix()
    c = np.asarray(center)
    t[:3, 3] = c - t[:3, :3] @ c
    return t


def pose(deg=0, xyz=(0, 0, 0)):
    q = Rotation.from_euler("z", deg, degrees=True).as_quat()
    return torch.tensor([[*xyz, q[3], *q[:3]]], dtype=torch.float64)


def test_identity_is_original_dense_potential_and_progress_telescopes():
    e = SymmetryEvaluator([[np.eye(4)]])
    goal = pose()
    actual = pose(13, (0.02, 0.01, 0))
    ix = torch.tensor([0])
    value = orbit_potential(e, actual, goal, ix, R)
    assert torch.allclose(value, potential(*pose_distances(actual, goal[:, None]), R).squeeze(1))
    states = [pose(20, (0.03, 0, 0)), actual, goal]
    vals = [orbit_potential(e, s, goal, ix, R) for s in states]
    assert torch.allclose(sum(vals[i + 1] - vals[i] for i in range(2)), vals[-1] - vals[0])


def test_finite_switch_potential_continuous_and_equivalent_reward_equal():
    e = SymmetryEvaluator([[np.eye(4), transform(180)]])
    ix = torch.tensor([0])
    goal = pose()
    vals = [orbit_potential(e, pose(d), goal, ix, R) for d in [89.9999, 90, 90.0001]]
    assert abs(vals[0] - vals[1]) < 1e-5 and abs(vals[1] - vals[2]) < 1e-5
    assert torch.allclose(orbit_potential(e, pose(0), goal, ix, R), orbit_potential(e, pose(180), goal, ix, R))


def axial(center):
    return SymmetryEvaluator([[np.eye(4)]], continuous=[[{"axis": [0, 0, 1], "center": list(center)}]])


def test_continuous_offcenter_potential_matches_dense_oracle():
    e = axial((0.04, 0, 0))
    ix = torch.tensor([0])
    goal = pose()
    actual = pose(117, (0.028, -0.014, 0.006))
    value = orbit_potential(e, actual, goal, ix, R)
    angles = torch.linspace(0, 2 * torch.pi, 100001, dtype=torch.float64)
    candidates = torch.stack(
        (
            0.04 - 0.04 * angles.cos(),
            -0.04 * angles.sin(),
            torch.zeros_like(angles),
            (angles / 2).cos(),
            torch.zeros_like(angles),
            torch.zeros_like(angles),
            (angles / 2).sin(),
        ),
        -1,
    )[None]
    oracle = potential(*pose_distances(actual, candidates), R).max()
    assert abs(value - oracle) < 2e-5


def descriptor(actual, goal):
    return torch.cat(
        (actual, goal, torch.eye(3, dtype=actual.dtype).reshape(1, 9), torch.zeros(1, 1, dtype=actual.dtype)), 1
    )


def test_pose_set_loss_accepts_finite_equivalent_and_has_finite_gradient():
    e = SymmetryEvaluator([[np.eye(4), transform(180)]])
    target = descriptor(pose(), pose())
    pred = torch.tensor([[0.0, 0, 0, 0, 0, np.pi / 0.5]], dtype=torch.float64, requires_grad=True)
    loss = pose_set_loss(pred, target, e, 0.5)
    assert loss < 1e-12
    loss.sum().backward()
    assert torch.isfinite(pred.grad).all()


def test_continuous_aux_loss_accepts_non_grid_angle_and_offcenter_translation():
    e = axial((0.04, 0, 0))
    target = descriptor(pose(), pose())
    theta = 0.7312
    pred = torch.tensor(
        [[(0.04 - 0.04 * np.cos(theta)) / 0.1, -0.04 * np.sin(theta) / 0.1, 0, 0, 0, theta / 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = pose_set_loss(pred, target, e, 0.5)
    assert loss < 1e-8
    loss.sum().backward()
    assert torch.isfinite(pred.grad).all()


def test_set_aux_loss_continuous_at_switch_and_does_not_average_goals():
    e = SymmetryEvaluator([[np.eye(4), transform(180)]])
    target = descriptor(pose(), pose())
    losses = [
        pose_set_loss(torch.tensor([[0.0, 0, 0, 0, 0, np.deg2rad(d) / 0.5]], dtype=torch.float64), target, e, 0.5)
        for d in [89.9999, 90, 90.0001]
    ]
    assert losses[1] > 0.5 and abs(losses[0] - losses[2]) < 1e-8
    assert abs(losses[0] - losses[1]) < 1e-5


def test_continuous_potential_random_states_match_dense_oracle():
    torch.manual_seed(711)
    n = 8
    e = axial((0.033, -0.017, 0))
    goal = pose().expand(n, -1)
    actual = torch.cat(
        (
            torch.randn(n, 3, dtype=torch.float64) * 0.04,
            torch.nn.functional.normalize(torch.randn(n, 4, dtype=torch.float64), dim=-1),
        ),
        1,
    )
    ix = torch.zeros(n, dtype=torch.long)
    result = orbit_potential(e, actual, goal, ix, R)
    theta = torch.linspace(0, 2 * torch.pi, 65537, dtype=torch.float64)
    x = 0.033 - 0.033 * theta.cos() - 0.017 * theta.sin()
    y = -0.017 - 0.033 * theta.sin() + 0.017 * theta.cos()
    candidates = torch.stack(
        (
            x,
            y,
            torch.zeros_like(theta),
            (theta / 2).cos(),
            torch.zeros_like(theta),
            torch.zeros_like(theta),
            (theta / 2).sin(),
        ),
        -1,
    )[None].expand(n, -1, -1)
    oracle = potential(*pose_distances(actual, candidates), R).max(-1).values
    assert torch.allclose(result, oracle, atol=2e-5, rtol=0)
