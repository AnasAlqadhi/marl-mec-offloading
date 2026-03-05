"""
agents/maddpg_agent.py – Multi-Agent Deep Deterministic Policy Gradient
========================================================================
Centralised training / decentralised execution (CTDE) paradigm.

Each agent i has:
  • A local deterministic actor  πi(oi) → ai
  • A centralised Q-critic       Qi(o_all, a_all) → scalar

Target networks are soft-updated with parameter τ.

References
----------
Lowe et al., "Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments", NeurIPS 2017.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.maddpg_model import Actor, Critic
from models.replay_buffer import ReplayBuffer


class MADDPG:
    """MADDPG controller for N homogeneous agents.

    Parameters
    ----------
    N       : number of agents
    obs_dim : per-agent observation dimension
    act_dim : per-agent action dimension
    gamma   : discount factor
    tau     : soft-update coefficient for target networks
    batch   : replay-buffer mini-batch size
    lr      : learning rate (Adam) for all actors and critics
    """

    def __init__(
        self,
        N:       int,
        obs_dim: int,
        act_dim: int,
        gamma:   float = 0.99,
        tau:     float = 0.005,
        batch:   int   = 256,
        lr:      float = 1e-4,
    ):
        self.N       = N
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma   = gamma
        self.tau     = tau
        self.batch   = batch

        # Per-agent actors, critics, and their target copies
        critic_in = (obs_dim + act_dim) * N
        self.actors  = [Actor(obs_dim, act_dim) for _ in range(N)]
        self.critics = [Critic(critic_in)        for _ in range(N)]
        self.tgt_actors  = [Actor(obs_dim, act_dim) for _ in range(N)]
        self.tgt_critics = [Critic(critic_in)        for _ in range(N)]

        # Initialise target weights identical to online weights
        for i in range(N):
            self.tgt_actors[i].load_state_dict(self.actors[i].state_dict())
            self.tgt_critics[i].load_state_dict(self.critics[i].state_dict())

        self.opt_actors  = [optim.Adam(a.parameters(), lr=lr) for a in self.actors]
        self.opt_critics = [optim.Adam(c.parameters(), lr=lr) for c in self.critics]

        self.buffer = ReplayBuffer()

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Decentralised action selection (no gradients).

        Parameters
        ----------
        obs : ndarray, shape (N, obs_dim)

        Returns
        -------
        actions : ndarray, shape (N, act_dim)
        """
        return np.stack([
            self.actors[i](torch.FloatTensor(obs[i])).detach().numpy()
            for i in range(self.N)
        ])

    # Backward-compatibility alias
    select = select_action

    def store(self, transition) -> None:
        """Store a (s, a, r, s') transition in the replay buffer."""
        self.buffer.add(transition)

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Sample a mini-batch and update all actor-critic pairs."""
        if len(self.buffer) < self.batch:
            return

        batch = self.buffer.sample(self.batch)
        S, A, R, S2 = map(np.array, zip(*batch))

        S_flat  = torch.FloatTensor(S.reshape(self.batch, -1))
        A_flat  = torch.FloatTensor(A.reshape(self.batch, -1))
        S2_flat = torch.FloatTensor(S2.reshape(self.batch, -1))
        R_t     = torch.FloatTensor(R)           # shape (batch, N)

        for i in range(self.N):
            # ── Critic update ─────────────────────────────────────────
            with torch.no_grad():
                tgt_actions = torch.cat([
                    self.tgt_actors[j](torch.FloatTensor(S2[:, j, :]))
                    for j in range(self.N)
                ], dim=1)
                y = (
                    R_t[:, i]
                    + self.gamma
                    * self.tgt_critics[i](
                        torch.cat([S2_flat, tgt_actions], dim=1)
                    ).squeeze()
                )

            q = self.critics[i](torch.cat([S_flat, A_flat], dim=1)).squeeze()
            critic_loss = nn.MSELoss()(q, y)
            self.opt_critics[i].zero_grad()
            critic_loss.backward()
            self.opt_critics[i].step()

            # ── Actor update ──────────────────────────────────────────
            curr_actions = torch.cat([
                self.actors[j](torch.FloatTensor(S[:, j, :]))
                for j in range(self.N)
            ], dim=1)
            actor_loss = -self.critics[i](
                torch.cat([S_flat, curr_actions], dim=1)
            ).mean()
            self.opt_actors[i].zero_grad()
            actor_loss.backward()
            self.opt_actors[i].step()

            # ── Soft target update ────────────────────────────────────
            self._soft_update(self.tgt_critics[i], self.critics[i])
            self._soft_update(self.tgt_actors[i],  self.actors[i])

    # ------------------------------------------------------------------
    def _soft_update(self, target: nn.Module, online: nn.Module) -> None:
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
