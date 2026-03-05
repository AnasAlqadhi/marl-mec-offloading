"""
configs/config.py – Experiment Configuration
=============================================
Edit the values below and pass a Config object to the training scripts.
Each run is identified by a unique `run_id` that determines where
results are saved under  output/<run_id>/.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Run identity ────────────────────────────────────────────────
    run_id: str = "run1_lat05_q09_chan2"

    # ── Environment ─────────────────────────────────────────────────
    num_agents:          int        = 4
    num_channels:        int        = 2
    latency_threshold:   float      = 0.5   # τ – deadline in seconds
    quality_constraint:  float      = 0.9   # ε – minimum QoE score
    episodes:            int        = 50_000

    # ── PPO hyper-parameters ─────────────────────────────────────────
    ppo_lr:    float = 1e-4
    ppo_clip:  float = 0.1
    ppo_gamma: float = 0.99
    ppo_ent:   float = 1e-3
    ppo_batch: int   = 256    # update every N steps (on-policy rollout)

    # ── MADDPG hyper-parameters ───────────────────────────────────────
    maddpg_lr:      float = 1e-4
    maddpg_gamma:   float = 0.99
    maddpg_tau:     float = 0.005
    maddpg_batch:   int   = 256
    maddpg_buf_size:int   = 200_000

    # ── Logging ──────────────────────────────────────────────────────
    log_every:    int = 1_000   # print progress every N episodes
    smooth_window:int = 500     # moving-average window for plots


# ── Pre-defined experiment configurations (Table 1 in the report) ────
EXPERIMENTS: List[Config] = [
    Config("run1_lat05_q09_chan2",   num_channels=2, latency_threshold=0.50, quality_constraint=0.90),
    Config("run2_lat04_q09_chan2",   num_channels=2, latency_threshold=0.40, quality_constraint=0.90),
    Config("run3_lat03_q095_chan3",  num_channels=3, latency_threshold=0.30, quality_constraint=0.95),
    Config("run4_lat02_q10_chan1",   num_channels=1, latency_threshold=0.20, quality_constraint=1.00),
    Config("run5_lat025_q092_chan2", num_channels=2, latency_threshold=0.25, quality_constraint=0.92),
]
