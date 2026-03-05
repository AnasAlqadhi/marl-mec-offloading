"""
scripts/run_maddpg.py – MADDPG Training Script
================================================
Run from the repository root:

    python scripts/run_maddpg.py                        # default config
    python scripts/run_maddpg.py --run_id run2 --latency 0.4

Results are written to  output/<run_id>/
    ddpg_rewards.csv  – per-episode metrics
    ddpg_curve.png    – smoothed reward plot
"""

import argparse
import pathlib
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.env import MECEnv
from src.agents.maddpg_agent import MADDPG


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train MADDPG on MEC environment")
    p.add_argument("--run_id",   default="run1_lat05_q09_chan2")
    p.add_argument("--episodes", type=int,   default=50_000)
    p.add_argument("--agents",   type=int,   default=4)
    p.add_argument("--latency",  type=float, default=0.5)
    p.add_argument("--quality",  type=float, default=0.9)
    p.add_argument("--channels", type=int,   default=2)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--tau",      type=float, default=0.005)
    p.add_argument("--batch",    type=int,   default=256)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    outdir = pathlib.Path("output") / args.run_id
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[MADDPG] Saving results to → {outdir}")

    # Environment
    env = MECEnv(args.agents)
    env.latency_thresholds  = [args.latency]  * args.agents
    env.quality_constraints = [args.quality]  * args.agents
    env.num_channels        = args.channels

    agent = MADDPG(args.agents, env.obs_len, env.act_len,
                   lr=args.lr, tau=args.tau, batch=args.batch)

    rewards, energies, latencies, phis = [], [], [], []

    for ep in range(1, args.episodes + 1):
        s  = env.reset()
        a  = agent.select_action(s)
        ns, r, _, info = env.step(a)

        agent.store((s, a, [r] * args.agents, ns))
        agent.train()

        rewards.append(r);    energies.append(info["energy"])
        latencies.append(info["latency"]); phis.append(info["phi"])

        if ep % 1_000 == 0:
            print(f"[MADDPG {args.run_id}]  ep {ep:>6}/{args.episodes}  "
                  f"reward={r:.3f}  energy={info['energy']:.3f}  "
                  f"latency={info['latency']:.3f}  phi={info['phi']:.3f}")

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = outdir / "ddpg_rewards.csv"
    pd.DataFrame({
        "Episode": range(1, args.episodes + 1),
        "Reward":  rewards,
        "Energy":  energies,
        "Latency": latencies,
        "Phi":     phis,
    }).to_csv(csv_path, index=False)
    print(f"[MADDPG] CSV saved → {csv_path}")

    # ── Quick plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"MADDPG – {args.run_id}", fontsize=14)
    eps = range(1, args.episodes + 1)
    for ax, col, title in zip(
        axes.flat,
        [rewards, energies, latencies, phis],
        ["Reward", "Energy (J)", "Latency (s)", "Quality (Φ)"],
    ):
        ax.plot(eps, pd.Series(col).rolling(500, min_periods=1).mean())
        ax.set_title(title); ax.set_xlabel("Episode"); ax.grid(True)

    plt.tight_layout()
    plt.savefig(outdir / "ddpg_curve.png", dpi=120)
    print(f"[MADDPG] Plot saved → {outdir / 'ddpg_curve.png'}")
    plt.show()


if __name__ == "__main__":
    main()
