"""Self-contained checkpoint-compatible Gaussian-motion policy wrapper."""

from __future__ import annotations

import torch
from torch import nn


def hybrid_neglogp(
    actions: torch.Tensor,
    motion_mu: torch.Tensor,
    motion_sigma: torch.Tensor,
    motion_logstd: torch.Tensor,
    completion_logits: torch.Tensor,
) -> torch.Tensor:
    """Joint negative log likelihood of six Gaussian and one Bernoulli action."""

    motion_actions = actions[..., :-1]
    completion_actions = actions[..., -1].clamp(0.0, 1.0)
    gaussian = (
        0.5 * (((motion_actions - motion_mu) / motion_sigma) ** 2).sum(dim=-1)
        + 0.5 * torch.log(motion_mu.new_tensor(2.0 * torch.pi)) * motion_actions.shape[-1]
        + motion_logstd.sum(dim=-1)
    )
    bernoulli = torch.nn.functional.binary_cross_entropy_with_logits(
        completion_logits.squeeze(-1), completion_actions, reduction="none"
    )
    return gaussian + bernoulli


def hybrid_entropy(
    motion_sigma: torch.Tensor,
    motion_logstd: torch.Tensor,
    completion_logits: torch.Tensor,
) -> torch.Tensor:
    gaussian = 0.5 + 0.5 * torch.log(motion_sigma.new_tensor(2.0 * torch.pi))
    gaussian = gaussian * motion_sigma.shape[-1] + motion_logstd.sum(dim=-1)
    bernoulli = torch.distributions.Bernoulli(logits=completion_logits).entropy().squeeze(-1)
    return gaussian + bernoulli


def bernoulli_kl_from_probabilities(old_probability: torch.Tensor, new_probability: torch.Tensor) -> torch.Tensor:
    """Per-sample Bernoulli KL with numerically bounded probabilities."""

    eps = torch.finfo(old_probability.dtype).eps
    old = old_probability.clamp(eps, 1.0 - eps)
    new = new_probability.clamp(eps, 1.0 - eps)
    return old * torch.log(old / new) + (1.0 - old) * torch.log((1.0 - old) / (1.0 - new))


class _RunningMeanStdState(nn.Module):
    """Checkpoint-compatible value statistics needed for strict state loading."""

    def __init__(self, value_size: int) -> None:
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(value_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(value_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        bounded = torch.clamp(value, min=-5.0, max=5.0)
        return torch.sqrt(self.running_var.float() + 1.0e-5) * bounded + self.running_mean.float()


class GraspCompletionModel:
    """Build the checkpoint-compatible hybrid policy without RL-Games.

    RL-Games supplied only the outer module names, optional value statistics,
    and action-distribution bookkeeping. Deterministic deployment consumes
    ``mus``; reproducing that small wrapper locally avoids installing training
    dependencies such as Gym, Ray, and Weights & Biases on the ROS machine.
    """

    def __init__(self, network):
        self.network_builder = network

    def build(self, config: dict) -> "GraspCompletionModel.Network":
        return self.Network(
            self.network_builder.build("a2c", **config),
            normalize_value=bool(config.get("normalize_value", False)),
            normalize_input=bool(config.get("normalize_input", False)),
            value_size=int(config.get("value_size", 1)),
        )

    class Network(nn.Module):
        def __init__(
            self,
            a2c_network: nn.Module,
            *,
            normalize_value: bool,
            normalize_input: bool,
            value_size: int,
        ) -> None:
            super().__init__()
            if normalize_input:
                raise ValueError("The self-contained deployment model requires normalize_input=false.")
            self.normalize_value = bool(normalize_value)
            if self.normalize_value:
                self.value_mean_std = _RunningMeanStdState(value_size)
            self.a2c_network = a2c_network

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def forward(self, input_dict: dict) -> dict[str, torch.Tensor | None]:
            is_train = input_dict.get("is_train", True)
            previous_actions = input_dict.get("prev_actions")
            motion_mu, motion_logstd, completion_logits, value, states = self.a2c_network(input_dict)
            motion_sigma = torch.exp(motion_logstd)
            completion_probability = torch.sigmoid(completion_logits)
            if is_train:
                if previous_actions is None:
                    raise ValueError("Training the hybrid model requires previous actions.")
                return {
                    "prev_neglogp": hybrid_neglogp(
                        previous_actions,
                        motion_mu,
                        motion_sigma,
                        motion_logstd,
                        completion_logits,
                    ),
                    "values": value,
                    "entropy": hybrid_entropy(motion_sigma, motion_logstd, completion_logits),
                    "rnn_states": states,
                    "mus": torch.cat((motion_mu, completion_probability), dim=-1),
                    "sigmas": torch.cat(
                        (motion_sigma, torch.ones_like(completion_probability)),
                        dim=-1,
                    ),
                    "completion_logits": completion_logits,
                    "completion_probability": completion_probability,
                }

            motion_action = torch.distributions.Normal(motion_mu, motion_sigma, validate_args=False).sample()
            completion_action = torch.distributions.Bernoulli(logits=completion_logits).sample()
            actions = torch.cat((motion_action, completion_action), dim=-1)
            return {
                "neglogpacs": hybrid_neglogp(
                    actions,
                    motion_mu,
                    motion_sigma,
                    motion_logstd,
                    completion_logits,
                ),
                "values": self.value_mean_std.denormalize(value) if self.normalize_value else value,
                "actions": actions,
                "rnn_states": states,
                # Deterministic players use mus. The final value is therefore
                # the actual visual completion probability, not a raw logit.
                "mus": torch.cat((motion_mu, completion_probability), dim=-1),
                "sigmas": torch.cat((motion_sigma, torch.ones_like(completion_probability)), dim=-1),
                "completion_logits": completion_logits,
                "completion_probability": completion_probability,
            }


__all__ = [
    "GraspCompletionModel",
    "bernoulli_kl_from_probabilities",
    "hybrid_entropy",
    "hybrid_neglogp",
]
