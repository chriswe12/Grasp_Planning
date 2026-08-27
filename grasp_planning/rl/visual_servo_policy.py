"""Spatial goal-conditioned residual visual-servo policy."""

from __future__ import annotations

import torch
from torch import nn


class SharedRgbdEncoder(nn.Module):
    """Encode RGB-D while retaining a spatial feature map."""

    def __init__(self, feature_channels: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, feature_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(),
        )

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        return self.network(rgbd)


class ResidualVisualServoPolicy(nn.Module):
    """Predict normalized camera-frame residual twist from goal and live RGB-D."""

    def __init__(self, *, joint_count: int = 7, feature_channels: int = 128) -> None:
        super().__init__()
        self.encoder = SharedRgbdEncoder(feature_channels=feature_channels)
        # goal_map plus signed (live_map - goal_map): no redundant live concatenation.
        self.comparison = nn.Sequential(
            nn.Conv2d(2 * feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        proprioception_size = joint_count + 1 + 6
        self.proprioception = nn.Sequential(
            nn.Linear(proprioception_size, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feature_channels + 64, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 6),
            nn.Tanh(),
        )

    def forward(
        self,
        *,
        live_rgbd: torch.Tensor,
        goal_rgbd: torch.Tensor,
        joint_positions: torch.Tensor,
        progress: torch.Tensor,
        nominal_twist_camera: torch.Tensor,
    ) -> torch.Tensor:
        live_map = self.encoder(live_rgbd)
        goal_map = self.encoder(goal_rgbd)
        if goal_map.shape[0] == 1 and live_map.shape[0] > 1:
            goal_map = goal_map.expand(live_map.shape[0], -1, -1, -1)
        elif goal_map.shape[0] != live_map.shape[0]:
            raise ValueError(
                "goal_rgbd batch size must be one or match live_rgbd batch size."
            )
        comparison = self.comparison(torch.cat((goal_map, live_map - goal_map), dim=1))
        proprioception = self.proprioception(
            torch.cat((joint_positions, progress, nominal_twist_camera), dim=1)
        )
        return self.head(torch.cat((comparison, proprioception), dim=1))
