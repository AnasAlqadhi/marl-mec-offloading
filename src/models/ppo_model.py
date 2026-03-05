"""
models/ppo_model.py – Actor-Critic Networks for PPO
====================================================
Two-layer MLP actor (Gaussian policy) and value-function critic
used by the PPO agent.
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Stochastic Gaussian policy network.

    Outputs mean and log-std for a continuous action distribution.
    log_std is a learnable parameter (not state-dependent).
    """

    def __init__(self, s_dim: int, a_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(s_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),    nn.ReLU(),
            nn.Linear(64, a_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(a_dim))

    def forward(self, x: torch.Tensor):
        """Returns (mean, std) of the action distribution."""
        mean = self.fc(x)
        std  = self.log_std.exp().expand_as(mean)
        return mean, std


class Critic(nn.Module):
    """State-value function V(s)."""

    def __init__(self, s_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(s_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),    nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
