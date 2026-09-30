import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from grasp_planning.rl.franka_symmetry import (
    SymmetryEvaluator,
    goal_relative_symmetries,
    pose_matrix,
    rigid_matrix,
    source_symmetries,
)


def pose(matrix):
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.r_[matrix[:3, 3], q[3], q[:3]]


def rotation(degrees, centre=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
    result[:3, 3] = np.asarray(centre) - result[:3, :3] @ centre
    return result


def errors(evaluator, actual, goal):
    return evaluator.errors(
        torch.tensor(np.array([pose(actual)]), dtype=torch.float32),
        torch.tensor(np.array([pose(goal)]), dtype=torch.float32),
        torch.tensor([0]),
    )


def test_off_centre_symmetry_moves_tcp_and_is_world_frame_invariant():
    obj = rotation(37)
    obj[:3, 3] = [0.3, -0.2, 0.1]
    goal = obj.copy()
    goal[:3, 3] += obj[:3, :3] @ [0.04, 0, 0.02]
    s = rotation(180)
    orbit = goal_relative_symmetries(pose(obj), pose(goal), [np.eye(4), s])
    equivalent = obj @ s @ np.linalg.inv(obj) @ goal
    evaluator = SymmetryEvaluator([orbit])
    placement = rotation(-71)
    placement[:3, 3] = [3, 4, 0.5]
    p, r = errors(evaluator, placement @ equivalent, placement @ goal)
    pp, rr, index, ready = evaluator.select(p, r, 0.004, np.deg2rad(3))
    assert p[0, 0] == pytest.approx(0.08, abs=1e-6)
    assert ready.item() and index.item() == 1
    assert pp.item() < 1e-6 and rr.item() < 1e-6


def test_position_and_rotation_must_match_same_representative():
    p, r = torch.tensor([[0.0, 0.1]]), torch.tensor([[np.pi, 0.0]])
    assert not SymmetryEvaluator.select(p, r, 0.004, np.deg2rad(3))[-1].item()


def test_small_residual_is_not_erased_by_180_degree_symmetry():
    evaluator = SymmetryEvaluator([[np.eye(4), rotation(180)]])
    p, r = errors(evaluator, rotation(3.78), np.eye(4))
    assert not evaluator.select(p, r, 0.004, np.deg2rad(3))[-1].item()
    assert evaluator.select(p, r, 0.005, np.deg2rad(5))[-1].item()


def test_quaternion_sign_and_padded_identity_do_not_create_false_alternatives():
    evaluator = SymmetryEvaluator([[np.eye(4)], [np.eye(4), rotation(180)]])
    tcp = torch.tensor([[0, 0, 0, -1, 0, 0, 0]], dtype=torch.float32)
    goal = -tcp
    goal[:, :3] = 0
    p, r = evaluator.errors(tcp, goal, torch.tensor([0]))
    assert p[0, 0] == r[0, 0] == 0
    assert torch.isinf(p[0, 1]) and torch.isinf(r[0, 1])


def test_bad_poses_cannot_pass():
    evaluator = SymmetryEvaluator([[np.eye(4)]])
    p, r = errors(evaluator, np.eye(4), np.eye(4))
    p[:] = torch.inf
    assert not evaluator.select(p, r, 0.004, 0.05)[-1].any()
    with pytest.raises(ValueError):
        pose_matrix([0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        rigid_matrix(np.diag([-1, 1, 1, 1]))
    p, r = evaluator.errors(torch.zeros(1, 7), torch.zeros(1, 7), torch.tensor([0]))
    assert not evaluator.select(p, r, 0.004, 0.05)[-1].any()


def test_asset_scale_and_rotated_translated_bundle_frame():
    frame = rotation(30)
    frame[:3, 3] = [0.3, -0.1, 0.2]
    q = Rotation.from_matrix(frame[:3, :3]).as_quat()
    target = dict(
        mesh_path="obj/fabrica/test/0.obj",
        mesh_scale=0.02,
        source_frame_origin_obj_world=frame[:3, 3].tolist(),
        source_frame_orientation_xyzw_obj_world=q.tolist(),
    )
    s = rotation(180, [0.1, 0.2, 0])
    record = dict(
        name="halfturn",
        type="finite_rotation",
        matrix_obj=s.tolist(),
        validation=dict(accepted=True, vertex_max_m=1e-6),
    )
    asset = dict(frame="object", mesh_scale=0.01, parts={"0": dict(symmetries=[record])})
    transforms, report = source_symmetries(target, asset)
    s[:3, 3] *= 2
    assert np.allclose(transforms[1], np.linalg.inv(frame) @ s @ frame)
    assert report["names"] == ["identity", "halfturn"]
    approximate = copy.deepcopy(asset)
    approximate["parts"]["0"]["symmetries"][0]["validation"]["vertex_max_m"] = 0.0005
    transforms, report = source_symmetries(target, approximate)
    assert len(transforms) == 1 and report["excluded"] == ["halfturn"]


def test_tolerance_changes_reselect_paired_representative():
    p, r = torch.tensor([[0.001, 0.006]]), torch.tensor([[0.07, 0.01]])
    assert SymmetryEvaluator.select(p, r, 0.004, 0.05)[2].item() == 0
    result = SymmetryEvaluator.select(p, r, 0.01, 0.03)
    assert result[2].item() == 1 and result[3].item()


@pytest.mark.parametrize(
    "contact,declared,enabled,expected",
    [
        (0.0, True, True, True),
        (4.0, True, True, False),
        (0.0, False, True, False),
        (0.0, True, False, False),
    ],
)
def test_simulator_completion_preserves_stop_and_collision_gates(contact, declared, enabled, expected):
    # Execute the actual environment method without importing the Isaac runtime.
    path = (
        Path(__file__).resolve().parents[1]
        / "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/franka_zed_env.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FrankaZedEnv")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_dones")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    cfg = SimpleNamespace(
        training_recipe=None,
        unsafe_contact_force_n=3.0,
        symmetry_evaluation=enabled,
        symmetry_training=False,
        pose_evaluation=True,
        ready_position_m=0.004,
        ready_rotation_rad=np.deg2rad(3),
    )
    env = SimpleNamespace(
        cfg=cfg,
        actions=torch.tensor([[0.0, 0, 0, 0, 0, 0, float(declared)]]),
        target_index=torch.tensor([0]),
        goal_pose=torch.tensor([[0.0, 0, 0, 1, 0, 0, 0]]),
        episode_length_buf=torch.tensor([0]),
        max_episode_length=20,
        symmetry_evaluator=SymmetryEvaluator([[np.eye(4), rotation(180)]]),
    )
    env.pose_errors = lambda: (torch.zeros(1, 3), torch.tensor([[0.0, 0, np.pi]]))
    env.tcp_pose = lambda: (torch.zeros(1, 3), torch.tensor([[0.0, 0, 0, 1.0]]))
    env.contact_force = lambda: torch.tensor([contact])
    env._labels = lambda p, r: SimpleNamespace(ready=(p <= 0.004) & (r <= np.deg2rad(3)) & (contact < 3.0))
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.tensor([[[contact, 0.0, 0.0]]])))
    env.scene = {k: sensor for k in ["arm_contact", "hand_contact", "left_finger_contact", "right_finger_contact"]}
    namespace["_get_dones"](env)
    display = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "evaluation_error_norms")
    exec(compile(ast.Module(body=[display], type_ignores=[]), str(path), "exec"), namespace)
    dp, dr = namespace["evaluation_error_norms"](env)
    assert torch.allclose(dp, env.last_transition["position_error_m"])
    assert torch.allclose(dr, env.last_transition["rotation_error_rad"])
    assert env.terminal_success.item() == expected
    assert env.terminal_collision.item() == (contact >= 3.0)
    if enabled:
        assert env.last_transition["nominal_rotation_error_rad"].item() == pytest.approx(np.pi)
        assert env.last_transition["rotation_error_rad"].item() < 1e-6


def continuous_evaluate(actual, center=(0.04, 0, 0), orbit=None, goal=None):
    evaluator = SymmetryEvaluator(
        [orbit if orbit is not None else np.array([np.eye(4)])], continuous=[[dict(axis=[0, 0, 1], center=center)]]
    )
    return evaluator.evaluate(
        torch.tensor(np.array([pose(actual)]), dtype=torch.float64),
        torch.tensor(np.array([pose(np.eye(4) if goal is None else goal)]), dtype=torch.float64),
        torch.tensor([0]),
        0.004,
        np.deg2rad(3),
    )


@pytest.mark.parametrize("angle", [7.3, 43.17, 179.9, 223.2, 359.7])
def test_continuous_arbitrary_angle_offcenter(angle):
    p, r, _, ready = continuous_evaluate(rotation(angle, (0.04, 0, 0)))
    assert ready.item() and p.item() < 1e-7 and r.item() < 1e-7


def test_continuous_coupled_errors_and_axis_tilt():
    # Orientation matches theta=180, position matches theta=0: no shared solution.
    assert not continuous_evaluate(rotation(180))[-1].item()
    tilted = np.eye(4)
    tilted[:3, :3] = Rotation.from_euler("x", 10, degrees=True).as_matrix()
    assert not continuous_evaluate(tilted, center=(0, 0, 0))[-1].item()


def test_continuous_coset_and_world_placement():
    flip = np.eye(4)
    flip[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("xyz", [23, 51, 77], degrees=True).as_matrix()
    world[:3, 3] = [0.5, -0.3, 0.8]
    result = continuous_evaluate(
        world @ rotation(41.37, (0.04, 0, 0)) @ flip, orbit=np.array([np.eye(4), flip]), goal=world
    )
    assert result[-1].item() and result[0].item() < 1e-7 and result[1].item() < 1e-7


def test_continuous_minimax_matches_dense_independent_search():
    rng = np.random.default_rng(53)
    angles = np.linspace(0, 360, 36001)
    transforms = np.array([rotation(a, (0.04, 0, 0)) for a in angles])
    candidates = Rotation.from_matrix(transforms[:, :3, :3])
    for _ in range(12):
        actual = rotation(rng.uniform(0, 360), (0.04, 0, 0))
        actual[:3, 3] += rng.normal(0, 0.004, 3)
        actual[:3, :3] = actual[:3, :3] @ Rotation.from_rotvec(rng.normal(0, 0.03, 3)).as_matrix()
        p, r, _, _ = continuous_evaluate(actual)
        pe = np.linalg.norm(transforms[:, :3, 3] - actual[:3, 3], axis=-1)
        re = (candidates.inv() * Rotation.from_matrix(actual[:3, :3])).magnitude()
        expected = np.maximum(pe / 0.004, re / np.deg2rad(3)).min()
        assert max(p.item() / 0.004, r.item() / np.deg2rad(3)) == pytest.approx(expected, abs=0.002)


def test_continuous_invalid_poses_and_axis():
    with pytest.raises(ValueError):
        SymmetryEvaluator([[np.eye(4)]], continuous=[[dict(axis=[0, 0, 0], center=[0, 0, 0])]])
    e = SymmetryEvaluator([[np.eye(4)]], continuous=[[dict(axis=[0, 0, 1], center=[0, 0, 0])]])
    for bad in [torch.zeros(1, 7), torch.full((1, 7), float("nan"))]:
        assert not e.evaluate(bad, bad, torch.tensor([0]), 0.004, 0.05)[-1].any()


def test_continuous_asset_provenance_scale_and_frame(tmp_path):
    import hashlib

    from grasp_planning.rl.franka_symmetry import source_axial_symmetries

    mesh = tmp_path / "shape" / "0.obj"
    mesh.parent.mkdir()
    mesh.write_text("verified mesh bytes")
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_euler("x", 45, degrees=True).as_matrix()
    frame[:3, 3] = [0.3, -0.2, 0.1]
    target = dict(
        mesh_path="shape/0.obj",
        mesh_scale=0.02,
        source_frame_origin_obj_world=frame[:3, 3].tolist(),
        source_frame_orientation_xyzw_obj_world=Rotation.from_matrix(frame[:3, :3]).as_quat().tolist(),
    )
    record = dict(
        accepted=True,
        type="continuous_axial",
        mesh_path="shape/0.obj",
        mesh_scale=0.01,
        mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
        max_surface_error_m=0.00005,
        axis_obj=[0, 0, 1],
        center_obj_m=[0.1, 0.2, 0.3],
    )
    result = source_axial_symmetries(target, [record], tmp_path)[0]
    assert np.allclose(result["axis_source"], frame[:3, :3].T @ [0, 0, 1])
    assert np.allclose(result["center_source"], frame[:3, :3].T @ (np.array([0.1, 0.2, 0.3]) * 2 - frame[:3, 3]))
    assert not source_axial_symmetries(target, [dict(record, max_surface_error_m=0.0001)], tmp_path)
    assert not source_axial_symmetries(target, [dict(record, max_surface_error_m=float("nan"))], tmp_path)
    mesh.write_text("modified mesh")
    with pytest.raises(ValueError, match="mesh changed"):
        source_axial_symmetries(target, [record], tmp_path)


@pytest.mark.parametrize("continuous", [False, True])
def test_selected_training_goal_matches_joint_position_and_rotation_metric(continuous):
    orbit = [np.eye(4), rotation(180, (0.035, -0.01, 0))]
    evaluator = SymmetryEvaluator(
        [orbit], continuous=[[dict(axis=[0, 0, 1], center=[0.035, -0.01, 0])]] if continuous else None
    )
    actual = rotation(73 if continuous else 180, (0.035, -0.01, 0))
    actual[:3, 3] += [0.001, -0.0005, 0.0003]
    tcp = torch.tensor(np.array([pose(actual)]), dtype=torch.float64)
    goal = torch.tensor([[0.0, 0, 0, 1, 0, 0, 0]], dtype=torch.float64)
    pn, rn, _, ready, chosen = evaluator.evaluate(tcp, goal, torch.tensor([0]), 0.004, np.deg2rad(3), return_goal=True)
    assert ready.item()
    assert torch.allclose((chosen[:, :3] - tcp[:, :3]).norm(dim=-1), pn, atol=1e-7)
    measured = (
        Rotation.from_quat(chosen[0, [4, 5, 6, 3]].numpy()) * Rotation.from_quat(tcp[0, [4, 5, 6, 3]].numpy()).inv()
    )
    assert measured.magnitude() == pytest.approx(rn.item(), abs=1e-7)
    # The equivalent goal includes object-centre displacement, not just an EE spin.
    assert chosen[0, :3].norm() > 0.01


def test_training_pose_errors_use_selected_equivalent_goal_and_legacy_is_unchanged():
    path = (
        Path(__file__).resolve().parents[1]
        / "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/franka_zed_env.py"
    )
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "FrankaZedEnv")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "pose_errors")

    def compute(pos, quat, gp, gq, **kwargs):
        q = Rotation.from_quat(gq[:, [1, 2, 3, 0]].numpy()) * Rotation.from_quat(quat[:, [1, 2, 3, 0]].numpy()).inv()
        return gp - pos, torch.tensor(q.as_rotvec(), dtype=pos.dtype)

    ns = {"torch": torch, "compute_pose_error": compute}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    t = torch.tensor(np.array([pose(rotation(180, (0.04, 0, 0)))]), dtype=torch.float32)
    env = SimpleNamespace(
        cfg=SimpleNamespace(symmetry_training=True, ready_position_m=0.004, ready_rotation_rad=np.deg2rad(3)),
        goal_pose=torch.tensor([[0.0, 0, 0, 1, 0, 0, 0]]),
        target_index=torch.tensor([0]),
        symmetry_evaluator=SymmetryEvaluator([[np.eye(4), rotation(180, (0.04, 0, 0))]]),
        tcp_pose=lambda: (t[:, :3], t[:, 3:]),
    )
    p, r = ns["pose_errors"](env)
    assert p.norm() < 1e-6 and r.norm() < 1e-6
    env.cfg.symmetry_training = False
    p, r = ns["pose_errors"](env)
    assert p.norm() > 0.07 and r.norm() > 3
