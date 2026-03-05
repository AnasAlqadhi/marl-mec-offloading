"""
scripts/plot_all_results.py – Cross-Run Comparison Plots
=========================================================
Reads all output/<run_id>/ppo_rewards.csv and ddpg_rewards.csv files
and produces four comparison figures (Reward / Energy / Latency / Phi).

Usage
-----
    python scripts/plot_all_results.py
    python scripts/plot_all_results.py --metric Reward   # single metric
    python scripts/plot_all_results.py --smooth 1000
"""

import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


METRICS = ["Reward", "Energy", "Latency", "Phi"]
YLABELS = {
    "Reward":  "Episode Reward",
    "Energy":  "Energy (J)",
    "Latency": "Latency (s)",
    "Phi":     "Quality Score (Φ)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="output")
    p.add_argument("--metric",  default=None,
                   help="Single metric to plot (default: all four)")
    p.add_argument("--smooth",  type=int, default=500,
                   help="Rolling-average window size")
    return p.parse_args()


def load_all_runs(output_dir: pathlib.Path):
    """Scan output/ and return dict  run_id → {ppo: df, maddpg: df}."""
    runs = {}
    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        entry = {}
        ppo_csv  = run_dir / "ppo_rewards.csv"
        ddpg_csv = run_dir / "ddpg_rewards.csv"
        if ppo_csv.exists():
            entry["PPO"]    = pd.read_csv(ppo_csv)
        if ddpg_csv.exists():
            entry["MADDPG"] = pd.read_csv(ddpg_csv)
        if entry:
            runs[run_dir.name] = entry
    return runs


def plot_metric(runs, metric: str, smooth: int, save_path: pathlib.Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    colors  = cm.tab10(np.linspace(0, 1, len(runs) * 2))
    c_idx   = 0

    for run_id, algos in runs.items():
        for algo_name, df in algos.items():
            if metric not in df.columns:
                continue
            y = df[metric].rolling(smooth, min_periods=1).mean()
            label = f"{run_id} – {algo_name}"
            ax.plot(df["Episode"], y, label=label,
                    color=colors[c_idx], linewidth=1.2)
            c_idx += 1

    ax.set_title(f"{metric} Comparison – All Runs (MA{smooth})", fontsize=13)
    ax.set_xlabel("Episode")
    ax.set_ylabel(YLABELS.get(metric, metric))
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.close()


def main():
    args       = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    runs       = load_all_runs(output_dir)

    if not runs:
        print(f"[plot] No run directories found in '{output_dir}'. "
              "Run training scripts first.")
        return

    metrics = [args.metric] if args.metric else METRICS
    for metric in metrics:
        save_path = output_dir / f"compare_{metric.lower()}_{len(runs)}runs.png"
        print(f"Plotting {metric} …")
        plot_metric(runs, metric, args.smooth, save_path)

    print("Done.")


if __name__ == "__main__":
    main()
