"""
scripts/run_ppo.py – PPO Training Script
=========================================
Run from the repository root:

    python scripts/run_ppo.py                       # uses default Config
    python scripts/run_ppo.py --run_id run2 --latency 0.4 --quality 0.9

Results are written to  output/<run_id>/
    ppo_rewards.csv   – per-episode metrics
    reward_curve.png  – smoothed reward plot
"""

import argparse
import pathlib
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.env import MECEnv
from src.agents.ppo_agent import PPOAgent


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on MEC environment")
    p.add_argument("--run_id",   default="run1_lat05_q09_chan2")
    p.add_argument("--episodes", type=int,   default=50_000)
    p.add_argument("--agents",   type=int,   default=4)
    p.add_argument("--latency",  type=float, default=0.5)
    p.add_argument("--quality",  type=float, default=0.9)
    p.add_argument("--channels", type=int,   default=2)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--clip",     type=float, default=0.1)
    p.add_argument("--batch",    type=int,   default=256)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    outdir = pathlib.Path("output") / args.run_id
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[PPO] Saving results to → {outdir}")

    # Environment
    env = MECEnv(args.agents)
    env.latency_thresholds  = [args.latency]  * args.agents
    env.quality_constraints = [args.quality]  * args.agents
    env.num_channels        = args.channels

    state_dim  = env.obs_len * args.agents
    action_dim = env.act_len * args.agents
    agent = PPOAgent(state_dim, action_dim, lr=args.lr, clip=args.clip)

    # Logging
    rewards, energies, latencies, phis = [], [], [], []
    S_buf, A_buf, LP_buf, R_buf = [], [], [], []

    for ep in range(1, args.episodes + 1):
        s    = env.reset()
        flat = s.reshape(-1)
        a, lp = agent.select_action(flat)
        ns, r, _, info = env.step(a.reshape(args.agents, -1))

        S_buf.append(flat);         A_buf.append(a.reshape(-1))
        LP_buf.append(lp.reshape(-1)); R_buf.append(r)

        rewards.append(r);    energies.append(info["energy"])
        latencies.append(info["latency"]); phis.append(info["phi"])

        # On-policy update every `batch` steps
        if ep % args.batch == 0:
            G, returns = 0, []
            for rew in reversed(R_buf):
                G = rew + agent.gamma * G
                returns.insert(0, G)
            agent.update(S_buf, A_buf, LP_buf, returns)
            S_buf, A_buf, LP_buf, R_buf = [], [], [], []

        if ep % 1_000 == 0:
            print(f"[PPO {args.run_id}]  ep {ep:>6}/{args.episodes}  "
                  f"reward={r:.3f}  energy={info['energy']:.3f}  "
                  f"latency={info['latency']:.3f}  phi={info['phi']:.3f}")

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = outdir / "ppo_rewards.csv"
    pd.DataFrame({
        "Episode": range(1, args.episodes + 1),
        "Reward":  rewards,
        "Energy":  energies,
        "Latency": latencies,
        "Phi":     phis,
    }).to_csv(csv_path, index=False)
    print(f"[PPO] CSV saved → {csv_path}")

    # ── Quick plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"PPO – {args.run_id}", fontsize=14)
    eps = range(1, args.episodes + 1)
    for ax, col, title in zip(
        axes.flat,
        [rewards, energies, latencies, phis],
        ["Reward", "Energy (J)", "Latency (s)", "Quality (Φ)"],
    ):
        ax.plot(eps, pd.Series(col).rolling(500, min_periods=1).mean())
        ax.set_title(title); ax.set_xlabel("Episode"); ax.grid(True)

    plt.tight_layout()
    plt.savefig(outdir / "reward_curve.png", dpi=120)
    print(f"[PPO] Plot saved → {outdir / 'reward_curve.png'}")
    plt.show()


if __name__ == "__main__":
    main()
