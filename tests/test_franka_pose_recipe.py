"""Behavioral checks for the historical reward and15Hz completion adaptation."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "franka_pose_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    PACKAGE + ".franka_pose_recipe", Path(package.__path__[0]) / "franka_pose_recipe.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_completion_requires_stable_hold_and_resets_on_motion():
    env = module.FrankaPoseRecipeMixin()
    env.num_envs = 2
    env.device = "cpu"
    env.arm_ids = list(range(7))
    env.recipe = {
        "action_delta_limit": 0.5,
        "completion_probability_threshold": 0.95,
        "completion_required_consecutive_steps": 2,
        "completion_max_linear_speed_m_s": 0.005,
        "completion_max_angular_speed_rad_s": 0.03,
    }
    env.motion_history = torch.zeros(3, 2, 6)
    env.motion_delay = torch.zeros(2, dtype=torch.long)
    env.motion_scale = torch.ones(2, 1)
    env.motion_alpha = torch.ones(2, 1)
    env.motion_bias = torch.zeros(2, 6)
    env.motion_filter = torch.zeros(2, 6)
    env.previous_actions = torch.zeros(2, 6)
    env.completion_streak = torch.zeros(2, dtype=torch.long)
    env.robot = SimpleNamespace(data=SimpleNamespace(joint_vel=torch.zeros(2, 7)))
    jac = torch.zeros(2, 6, 7)
    jac[:, 0, 0] = 1
    env.tcp_jacobian = lambda: jac
    actions = torch.zeros(2, 7)
    actions[:, 6] = 0.99
    env._pose_actions(actions)
    assert not env.completion_declaration.any()
    env.robot.data.joint_vel[1, 0] = 0.01
    env._pose_actions(actions)
    assert env.completion_declaration.tolist() == [True, False]
    assert env.completion_streak.tolist() == [2, 0]


def test_fifteen_hz_discount_preserves_source_time_horizon():
    cfg = json.loads((ROOT / "configs/franka_clutter_v5_replica.json").read_text())
    agent = cfg["agent"]["params"]["config"]
    assert cfg["policy_hz"] == 15 and cfg["physics_hz"] == 120
    assert abs(agent["gamma"] ** 15 - 0.99**30) < 1e-12
    assert abs(agent["tau"] ** 15 - 0.95**30) < 1e-12


def test_boundary_mixture_separates_position_and_rotation_and_never_falls_back_to_ready():
    valid = torch.tensor([[True, True, True, True], [True, True, True, False]])
    kinds = torch.tensor([0, 1, 2, 3])
    progress = torch.tensor([0.94, 1.0, 0.985, 1.0])
    orientation = module.boundary_pool(valid, kinds, progress, torch.tensor([True, True]))
    assert orientation.tolist() == [[False, False, False, True], [True, False, False, False]]
    position = module.boundary_pool(valid, kinds, progress, torch.tensor([False, False]))
    assert position.tolist() == [[False, False, True, False], [False, False, True, False]]


def test_ready_exact_requires_physics_validation_and_position_boundary_fallback_stays_negative():
    kinds = torch.tensor([0, 1, 2, 3, 4])
    valid = torch.tensor([[True, True, False, True, False], [True, True, True, True, True]])
    ready = module.ready_pool(valid, kinds, torch.tensor([True, True]))
    assert ready.tolist() == [[False, True, False, False, False], [False, False, False, False, True]]
    boundary = module.boundary_pool(
        valid, kinds, torch.tensor([0.94, 1.0, 0.985, 1.0, 1.0]), torch.tensor([False, False])
    )
    assert boundary.tolist() == [[True, False, False, False, False], [False, False, True, False, False]]


def test_nominal_curriculum_uses_only_validated_non_goal_states():
    valid = torch.tensor([[True, True, True, True], [True, False, True, True]])
    kinds = torch.tensor([5, 5, 5, 1])
    progress = torch.tensor([0.0, 0.9, 1.0, 1.0])
    assert module.nominal_pool(valid, kinds, progress, 0.7).tolist() == [
        [False, True, False, False],
        [True, False, False, False],
    ]


def test_curriculum_is_matched_by_total_experience_and_survives_resume(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    recipe = json.loads((ROOT / "configs/franka_clutter_v5_replica.json").read_text())["source_settings"]
    env = module.FrankaPoseRecipeMixin()
    env.recipe = recipe
    env.num_envs = 64
    env.cfg = SimpleNamespace(pose_evaluation=False)
    env.common_step_counter = 64000
    env.pose_curriculum_offset = 0
    before = env._pose_curriculum()
    env.common_step_counter = 0
    env.pose_curriculum_offset = 64000
    assert env._pose_curriculum() == before
    assert 0 < before.fraction < 1
    env.pose_curriculum_offset = 384000
    assert env._pose_curriculum().fraction == 1


def test_dense_and_terminal_reward_agree_for_equivalent_goals_and_keep_collision_penalty():
    import numpy as np
    from scipy.spatial.transform import Rotation

    from grasp_planning.rl.franka_symmetry import SymmetryEvaluator

    symmetry = np.eye(4)
    symmetry[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    evaluator = SymmetryEvaluator([[np.eye(4), symmetry]])
    goal = torch.tensor([[0.0, 0, 0, 1, 0, 0, 0]])
    recipe = json.loads((ROOT / "configs/franka_clutter_v5_replica.json").read_text())["source_settings"]

    def reward(quaternion, contact=0.0):
        env = module.FrankaPoseRecipeMixin()
        env.recipe = recipe
        env.time_ratio = 2.0
        env.cfg = SimpleNamespace(symmetry_training=False)
        env.previous_position_error = torch.tensor([0.01])
        env.previous_rotation_error = torch.tensor([0.1])
        env.completion_declaration = torch.tensor([True])
        env.reset_mode = torch.tensor([0])
        env.episode_length_buf = torch.tensor([2])
        env.pose_timeout_steps = torch.tensor([224])
        env.terminal_collision = torch.tensor([contact >= 3])
        env.terminal_divergence = torch.tensor([False])
        env.terminal_success = ~env.terminal_collision
        env.actions = torch.zeros(1, 7)
        env.target_failure_scores = torch.zeros(1)
        env.target_index = torch.tensor([0])
        env.contact_force = lambda: torch.tensor([contact])
        env._labels = lambda p, r: SimpleNamespace(ready=(p <= 0.004) & (r <= 0.05236) & ~env.terminal_collision)
        p, r, *_ = evaluator.evaluate(torch.tensor([[0.0, 0, 0, *quaternion]]), goal, env.target_index, 0.004, 0.05236)
        return env._pose_reward(p, r)

    nominal = reward([1, 0, 0, 0])
    equivalent = reward([0, 0, 0, 1])
    assert torch.allclose(nominal, equivalent, atol=1e-6)
    assert equivalent.item() > 0
    assert reward([0, 0, 0, 1], 4).item() < equivalent.item() - recipe["unsafe_collision_penalty"]
