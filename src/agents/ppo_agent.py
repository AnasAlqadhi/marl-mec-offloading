"""
agents/ppo_agent.py – Proximal Policy Optimisation (PPO) Agent
===============================================================
On-policy actor-critic with clipped surrogate objective.

Key hyper-parameters
--------------------
gamma : discount factor
clip  : PPO clipping epsilon (ε_clip)
lr    : learning rate for both actor and critic (Adam)
ent   : entropy bonus coefficient
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.ppo_model import Actor, Critic


class PPOAgent:
    """Single shared PPO agent controlling all N agents' joint action.

    Parameters
    ----------
    s_dim : int   – flattened state dimension  (obs_len × N)
    a_dim : int   – flattened action dimension (act_len × N)
    """

    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        gamma: float = 0.99,
        clip:  float = 0.1,
        lr:    float = 1e-4,
        ent:   float = 1e-3,
    ):
        self.actor  = Actor(s_dim, a_dim)
        self.critic = Critic(s_dim)
        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip  = clip
        self.ent   = ent

    # ------------------------------------------------------------------
    def select_action(self, s_flat: np.ndarray):
        """Sample action and return (action, log_prob)."""
        s = torch.tensor(s_flat, dtype=torch.float32)
        mu, sd = self.actor(s)
        dist   = torch.distributions.Normal(mu, sd)
        action = dist.sample()
        log_p  = dist.log_prob(action)
        return action.detach().numpy(), log_p.detach().numpy()

    # Backward-compatibility alias
    select = select_action

    # ------------------------------------------------------------------
    def _evaluate(self, S, A):
        mu, sd = self.actor(S)
        dist   = torch.distributions.Normal(mu, sd)
        return dist.log_prob(A), dist.entropy(), self.critic(S).squeeze()

    # ------------------------------------------------------------------
    def update(self, states, actions, old_log_probs, returns, n_epochs: int = 5):
        """Run PPO update for *n_epochs* mini-epochs on the collected batch.

        Parameters
        ----------
        states       : list[ndarray]
        actions      : list[ndarray]
        old_log_probs: list[ndarray]
        returns      : list[float]   – Monte-Carlo returns (discounted rewards)
        """
        S      = torch.tensor(np.array(states),        dtype=torch.float32)
        A      = torch.tensor(np.array(actions),       dtype=torch.float32)
        old_lp = torch.tensor(np.array(old_log_probs), dtype=torch.float32)
        R      = torch.tensor(np.array(returns),       dtype=torch.float32)

        for _ in range(n_epochs):
            log_p, entropy, V = self._evaluate(S, A)

            advantage = R - V.detach()
            ratio     = torch.exp(log_p.sum(-1) - old_lp.sum(-1))

            # Clipped surrogate loss
            surr_loss = torch.min(
                ratio * advantage,
                torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage,
            ).mean()

            actor_loss  = -surr_loss - self.ent * entropy.mean()
            critic_loss = nn.MSELoss()(V, R)

            self.opt_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.opt_actor.step()

            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()
