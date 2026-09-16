from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from grasp_planning.rl.training_profiles import (
    TRAINING_PROFILE_NAMES,
    apply_training_environment_profile,
    apply_training_profile,
    get_training_profile,
)


def _agent_cfg() -> dict:
    return {
        "params": {
            "network": {
                "completion_loss_weight": 0.2,
                "completion_positive_weight": 3.0,
                "completion_motion_slowdown_start": 0.7,
                "completion_motion_speed_floor": 0.25,
            },
            "config": {
                "learning_rate": 5.0e-5,
                "lr_schedule": "linear",
                "central_value_config": {
                    "learning_rate": 5.0e-5,
                    "lr_schedule": "linear",
                    "network": {"mlp": {"units": [256, 128, 64]}},
                },
            },
        }
    }


def _env_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        reset_no_noise_fraction=0.15,
        reset_ready_fraction=0.15,
        reset_boundary_fraction=0.15,
        reset_ready_exact_fraction=0.25,
        live_calibration_shift_x_px=(-1.5, 1.5),
        live_calibration_shift_y_px=(-1.0, 1.0),
        live_calibration_scale=(0.99, 1.01),
        live_calibration_roll_deg=(-1.0, 1.0),
        live_depth_patch_dropout_probability=0.04,
        live_depth_structured_dropout_probability=0.08,
        live_depth_structured_dropout_seed_probability=(0.001, 0.006),
        live_patch_area_fraction=(0.005, 0.03),
        live_clean_episode_fraction=0.15,
        goal_live_color_relationship_enabled=False,
        goal_live_color_relationship_required=False,
        goal_live_color_match_fraction=0.25,
        goal_live_color_similar_fraction=0.20,
        near_goal_action_penalty_weight=0.0,
        action_delta_penalty_weight=0.0,
        near_goal_regression_penalty_weight=0.0,
        near_goal_excess_speed_penalty_weight=0.0,
        completion_premature_penalty=50.0,
        lift_reward_enabled=False,
        lift_reward_profile="none",
        lift_completion_negative_supervision_enabled=True,
        state_space=26,
        position_progress_weight=300.0,
        rotation_progress_weight=30.0,
        position_precision_weight=4.0,
        rotation_precision_weight=2.0,
        lift_commit_geometric_reward=0.0,
        lift_quality_reward=0.0,
        lift_geometric_lift_bonus=0.0,
        lift_neither_penalty=0.0,
    )


def test_baseline_profile_is_an_exact_no_op() -> None:
    agent_cfg = _agent_cfg()
    original = deepcopy(agent_cfg)
    env_cfg = _env_cfg()

    profile = apply_training_profile(env_cfg, agent_cfg, "baseline")

    assert profile.name == "baseline"
    assert agent_cfg == original
    assert env_cfg.reset_ready_fraction == 0.15
    assert env_cfg.reset_ready_exact_fraction == 0.25


def test_long_run_profile_applies_only_documented_overrides() -> None:
    agent_cfg = _agent_cfg()
    env_cfg = _env_cfg()

    profile = apply_training_profile(env_cfg, agent_cfg, "long_run_improved")

    train_cfg = agent_cfg["params"]["config"]
    assert profile.identifier.endswith(":long_run_improved")
    assert train_cfg["learning_rate"] == 3.0e-5
    assert train_cfg["lr_schedule"] is None
    assert train_cfg["central_value_config"]["learning_rate"] == 3.0e-5
    assert train_cfg["central_value_config"]["lr_schedule"] is None
    assert train_cfg["central_value_config"]["network"]["mlp"]["units"] == [512, 256, 128]
    assert env_cfg.reset_ready_fraction == 0.20
    assert env_cfg.reset_ready_exact_fraction == 0.35
    assert profile.metadata()["environment_overrides"]["reset_ready_fraction"] == 0.20


def test_profile_lookup_and_application_fail_closed() -> None:
    assert TRAINING_PROFILE_NAMES == (
        "baseline",
        "long_run_improved",
        "robust_no_reward_change",
        "robust_reward_change",
        "lift_conservative",
        "lift_primary",
    )
    assert get_training_profile("baseline").agent_overrides == {}
    with pytest.raises(ValueError, match="Unknown training profile"):
        get_training_profile("unknown")
    with pytest.raises(AttributeError, match="reset_ready_fraction"):
        apply_training_profile(SimpleNamespace(), _agent_cfg(), "long_run_improved")


def test_robust_profiles_differ_only_in_declared_reward_fields() -> None:
    no_reward_agent = _agent_cfg()
    reward_agent = _agent_cfg()
    no_reward_env = _env_cfg()
    reward_env = _env_cfg()

    no_reward = apply_training_profile(no_reward_env, no_reward_agent, "robust_no_reward_change")
    reward = apply_training_profile(reward_env, reward_agent, "robust_reward_change")

    assert not no_reward.changes_reward
    assert reward.changes_reward
    assert no_reward_agent == reward_agent
    no_reward_values = vars(no_reward_env)
    reward_values = vars(reward_env)
    changed = {key for key in no_reward_values if no_reward_values[key] != reward_values[key]}
    assert changed == {"near_goal_excess_speed_penalty_weight"}
    assert reward_env.near_goal_excess_speed_penalty_weight == 0.010
    assert reward_agent["params"]["network"]["completion_positive_weight"] == 1.5
    assert no_reward.metadata()["changes_reward"] is False
    assert reward.metadata()["changes_reward"] is True


def test_lift_reward_profiles_preserve_guidance_but_change_priority() -> None:
    conservative_agent = _agent_cfg()
    primary_agent = _agent_cfg()
    conservative_env = _env_cfg()
    primary_env = _env_cfg()

    conservative = apply_training_profile(conservative_env, conservative_agent, "lift_conservative")
    primary = apply_training_profile(primary_env, primary_agent, "lift_primary")

    assert conservative.changes_reward and primary.changes_reward
    assert conservative_env.lift_reward_enabled is True
    assert primary_env.lift_reward_enabled is True
    assert conservative_env.state_space == primary_env.state_space == 32
    assert conservative_env.position_progress_weight == 300.0
    assert primary_env.position_progress_weight == 100.0
    assert primary_env.rotation_progress_weight == 10.0
    assert primary_env.position_precision_weight == 1.5
    assert primary_env.rotation_precision_weight == 0.75
    assert conservative_env.lift_quality_reward == 25.0
    assert primary_env.lift_quality_reward == 55.0
    assert conservative_env.lift_commit_geometric_reward == 40.0
    assert primary_env.lift_commit_geometric_reward == 15.0


def test_environment_only_profile_application_supports_physical_smoke() -> None:
    env_cfg = _env_cfg()

    profile = apply_training_environment_profile(env_cfg, "lift_conservative")

    assert profile.name == "lift_conservative"
    assert env_cfg.lift_reward_enabled is True
    assert env_cfg.lift_reward_profile == "conservative"
