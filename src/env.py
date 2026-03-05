"""
env.py – Mobile Edge Computing (MEC) Simulation Environment
============================================================
A lightweight gym-style environment that models N mobile devices
each deciding whether to process a task locally or offload it to
a nearby edge server.

State  (per agent): [task_size, cpu_load, channel_gain, battery, deadline]
Action (per agent): continuous scalar in [-1, 1]  (maps to offloading ratio)

Metrics tracked per step
------------------------
energy  : total energy consumed across all agents  [Joules]
latency : average task completion delay            [seconds]
phi     : average video/task quality (0–1)
"""

import numpy as np


class MECEnv:
    """Toy MEC environment for multi-agent RL experiments.

    Parameters
    ----------
    num_agents : int
        Number of mobile devices (agents) in the system.
    """

    def __init__(self, num_agents: int = 4):
        self.n = num_agents
        self.obs_len = 5          # observation dimension per agent
        self.act_len = 1          # action dimension per agent

        # Per-run configurable constraints
        self.latency_thresholds  = [0.5] * self.n   # τ  deadline (seconds)
        self.quality_constraints = [0.9] * self.n   # ε  minimum QoE
        self.num_channels        = 2                 # available wireless channels

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset environment; returns initial state (n × obs_len)."""
        self.state = np.random.rand(self.n, self.obs_len)
        return self.state

    # ------------------------------------------------------------------
    def step(self, actions: np.ndarray):
        """Advance one time step.

        Parameters
        ----------
        actions : ndarray, shape (n, act_len)
            Offloading decision for each agent.

        Returns
        -------
        next_state, reward, done, info
        """
        # Toy reward: penalise large actions (high offloading cost)
        reward = -float(np.sum(np.square(actions)))

        next_state = np.random.rand(self.n, self.obs_len)

        info = {
            "energy":  float(np.sum(np.abs(actions)) * 0.1),
            "latency": float(np.random.rand() * 0.3 + 0.4),
            "phi":     float(0.95 - np.mean(np.abs(actions)) * 0.1),
        }

        self.state = next_state
        done = False
        return next_state, reward, done, info
