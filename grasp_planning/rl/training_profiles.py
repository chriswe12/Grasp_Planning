"""Named, reproducible optimization profiles for visual-servo training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

TRAINING_PROFILE_VERSION = "fabrica_training_v4"
TRAINING_PROFILE_NAMES = (
    "baseline",
    "long_run_improved",
    "robust_no_reward_change",
    "robust_reward_change",
    "lift_conservative",
    "lift_primary",
)

REWARD_ENVIRONMENT_FIELDS = frozenset(
    {
        "near_goal_action_penalty_weight",
        "action_delta_penalty_weight",
        "near_goal_regression_penalty_weight",
        "near_goal_excess_speed_penalty_weight",
        "completion_premature_penalty",
        "lift_reward_enabled",
        "lift_commit_geometric_reward",
        "lift_quality_reward",
        "lift_geometric_lift_bonus",
        "lift_neither_penalty",
        "position_progress_weight",
        "rotation_progress_weight",
        "position_precision_weight",
        "rotation_precision_weight",
    }
)


@dataclass(frozen=True)
class TrainingProfile:
    """One immutable set of agent and environment training overrides."""

    name: str
    agent_overrides: dict[str, Any]
    environment_overrides: dict[str, Any]
    description: str

    @property
    def changes_reward(self) -> bool:
        return bool(REWARD_ENVIRONMENT_FIELDS.intersection(self.environment_overrides))

    @property
    def identifier(self) -> str:
        return f"{TRAINING_PROFILE_VERSION}:{self.name}"

    def metadata(self) -> dict[str, Any]:
        """Return a serialization-safe description of the applied profile."""

        return {
            "name": self.name,
            "profile_id": self.identifier,
            "description": self.description,
            "agent_overrides": deepcopy(self.agent_overrides),
            "environment_overrides": deepcopy(self.environment_overrides),
            "changes_reward": self.changes_reward,
        }


_ROBUST_AGENT_OVERRIDES = {
    "params.config.learning_rate": 3.0e-5,
    "params.config.lr_schedule": None,
    "params.config.central_value_config.learning_rate": 3.0e-5,
    "params.config.central_value_config.lr_schedule": None,
    "params.config.central_value_config.network.mlp.units": [512, 256, 128],
    # These alter auxiliary supervision and action scaling, not environment reward.
    "params.network.completion_loss_weight": 0.30,
    # Reset sampling already supplies explicit positive and boundary examples.
    # A mild class weight avoids turning this jointly PPO-trained score into an
    # unnecessarily overconfident pseudo-probability.
    "params.network.completion_positive_weight": 1.5,
    "params.network.completion_motion_slowdown_start": 0.60,
    "params.network.completion_motion_speed_floor": 0.10,
}

_ROBUST_ENVIRONMENT_OVERRIDES = {
    # More exact positives and hard near-threshold negatives without starving
    # continuous path resets (35% remain ordinary path resets).
    "reset_no_noise_fraction": 0.10,
    "reset_ready_fraction": 0.25,
    "reset_boundary_fraction": 0.30,
    "reset_ready_exact_fraction": 0.40,
    # Cover the observed difference between the two physical D405 intrinsics,
    # with modest margin rather than matching either camera exactly.
    "live_calibration_shift_x_px": (-3.5, 3.5),
    "live_calibration_shift_y_px": (-2.0, 2.0),
    "live_calibration_scale": (0.98, 1.02),
    "live_calibration_roll_deg": (-1.5, 1.5),
    # Raise the existing episode-persistent structured patch frequency while
    # retaining clean episodes and the correlated disparity model.
    "live_depth_patch_dropout_probability": 0.10,
    "live_depth_structured_dropout_probability": 0.12,
    "live_depth_structured_dropout_seed_probability": (0.001, 0.008),
    "live_patch_area_fraction": (0.003, 0.04),
    "live_clean_episode_fraction": 0.15,
    "goal_live_color_relationship_enabled": True,
    # Never silently run the named robust profiles without the true rendered
    # goal variants that define the intended color-pair distribution.
    "goal_live_color_relationship_required": True,
    "goal_live_color_match_fraction": 0.25,
    "goal_live_color_similar_fraction": 0.20,
}

_LIFT_ENVIRONMENT_OVERRIDES = {
    **_ROBUST_ENVIRONMENT_OVERRIDES,
    "lift_reward_enabled": True,
    "lift_completion_negative_supervision_enabled": False,
    # The centralized critic receives four phase indicators, stored geometric
    # success, and normalized object lift in addition to the legacy 26 values.
    "state_space": 32,
    "near_goal_excess_speed_penalty_weight": 0.010,
}

_PROFILES = {
    "baseline": TrainingProfile(
        name="baseline",
        agent_overrides={},
        environment_overrides={},
        description="Unmodified repository PPO, central critic, and reset-mixture settings.",
    ),
    "long_run_improved": TrainingProfile(
        name="long_run_improved",
        agent_overrides={
            # The previous 5k runs decayed 5e-5 linearly toward zero while
            # diagnostics were still improving. A lower constant rate keeps
            # useful updates alive throughout a long run without increasing
            # the early-step size.
            "params.config.learning_rate": 3.0e-5,
            "params.config.lr_schedule": None,
            "params.config.central_value_config.learning_rate": 3.0e-5,
            "params.config.central_value_config.lr_schedule": None,
            # Only the privileged critic is enlarged. The visual actor and
            # deployment-time policy interface remain exactly baseline-sized.
            "params.config.central_value_config.network.mlp.units": [512, 256, 128],
        },
        environment_overrides={
            # Add five percentage points of completion-ready starts and make
            # a larger subset exact positives. Path resets still remain 50%
            # of the mix; this deliberately avoids a completion-only regime.
            "reset_ready_fraction": 0.20,
            "reset_ready_exact_fraction": 0.35,
        },
        description=(
            "Constant 3e-5 actor/critic learning rates, a 512-256-128 privileged "
            "critic, and modestly stronger completion-ready reset coverage."
        ),
    ),
    "robust_no_reward_change": TrainingProfile(
        name="robust_no_reward_change",
        agent_overrides=_ROBUST_AGENT_OVERRIDES,
        environment_overrides=_ROBUST_ENVIRONMENT_OVERRIDES,
        description=(
            "Robust completion supervision, hard reset coverage, dual-D405 camera uncertainty, "
            "structured depth dropout, constant long-run optimization, and no reward changes."
        ),
    ),
    "robust_reward_change": TrainingProfile(
        name="robust_reward_change",
        agent_overrides=_ROBUST_AGENT_OVERRIDES,
        environment_overrides={
            **_ROBUST_ENVIRONMENT_OVERRIDES,
            # Penalize only speed above the safe-stop gate near the goal. Slow
            # final corrections remain free; progress already penalizes pose
            # regression and the controller already limits action slew.
            "near_goal_excess_speed_penalty_weight": 0.010,
        },
        description=(
            "The robust profile plus a mild excess-speed cost near the operational completion region; "
            "slow corrections remain unpenalized."
        ),
    ),
    "lift_conservative": TrainingProfile(
        name="lift_conservative",
        agent_overrides=_ROBUST_AGENT_OVERRIDES,
        environment_overrides={
            **_LIFT_ENVIRONMENT_OVERRIDES,
            "lift_reward_profile": "conservative",
            "lift_commit_geometric_reward": 40.0,
            "lift_quality_reward": 25.0,
            "lift_geometric_lift_bonus": 10.0,
            "lift_neither_penalty": 30.0,
        },
        description=(
            "Robust visual training with the original dense pose shaping, a positive operational-pose "
            "commit reward, and a conservative retained-lift bonus."
        ),
    ),
    "lift_primary": TrainingProfile(
        name="lift_primary",
        agent_overrides=_ROBUST_AGENT_OVERRIDES,
        environment_overrides={
            **_LIFT_ENVIRONMENT_OVERRIDES,
            "lift_reward_profile": "lift_primary",
            # Keep about one third of the pose objective so the policy still
            # knows where to explore before it has discovered physical pickup.
            "position_progress_weight": 100.0,
            "rotation_progress_weight": 10.0,
            "position_precision_weight": 1.5,
            "rotation_precision_weight": 0.75,
            "lift_commit_geometric_reward": 15.0,
            "lift_quality_reward": 55.0,
            "lift_geometric_lift_bonus": 15.0,
            "lift_neither_penalty": 30.0,
        },
        description=(
            "Retained pickup is primary, with one-third-strength dense pose shaping and an additional "
            "bonus for lifting from the requested operational grasp pose."
        ),
    ),
}


def get_training_profile(name: str) -> TrainingProfile:
    """Resolve a named profile or fail with the supported names."""

    try:
        return _PROFILES[name]
    except KeyError as exc:
        supported = ", ".join(TRAINING_PROFILE_NAMES)
        raise ValueError(f"Unknown training profile {name!r}; choose one of: {supported}") from exc


def _set_nested(mapping: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".")
    target: Any = mapping
    for key in keys[:-1]:
        if not isinstance(target, dict) or key not in target:
            raise AttributeError(f"Training agent configuration has no field {dotted_path!r}")
        target = target[key]
    leaf = keys[-1]
    if not isinstance(target, dict) or leaf not in target:
        raise AttributeError(f"Training agent configuration has no field {dotted_path!r}")
    target[leaf] = deepcopy(value)


def apply_training_environment_profile(env_cfg: Any, name: str) -> TrainingProfile:
    """Apply only environment overrides, for simulator smoke/evaluation tools."""

    profile = get_training_profile(name)
    for field, value in profile.environment_overrides.items():
        if not hasattr(env_cfg, field):
            raise AttributeError(f"Training environment configuration has no field {field!r}")
        setattr(env_cfg, field, deepcopy(value))
    return profile


def apply_training_profile(env_cfg: Any, agent_cfg: dict[str, Any], name: str) -> TrainingProfile:
    """Apply one profile to the Isaac environment and RL-Games configuration."""

    profile = get_training_profile(name)
    for dotted_path, value in profile.agent_overrides.items():
        _set_nested(agent_cfg, dotted_path, value)
    apply_training_environment_profile(env_cfg, name)
    return profile
