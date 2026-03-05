"""
models/maddpg_model.py – Actor-Critic Networks for MADDPG
==========================================================
Deterministic actor (Tanh output in [-1, 1]) and a centralised
Q-critic that takes the concatenated global obs+actions.
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Deterministic policy network with Tanh output.

    Actions are bounded to [-1, 1].
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),      nn.ReLU(),
            nn.Linear(64, act_dim), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    """Centralised Q-network Q(s_all, a_all).

    Input dimension = (obs_dim + act_dim) * num_agents.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),        nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
