# Multi-Agent Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing

**Author:** Anas M. Alqadhi  
**Framework:** PyTorch · Python 3.9+  
**Training Scale:** 50,000 episodes · 5 experimental configurations

---

## Table of Contents

1. [About](#about)
2. [Repository Structure](#repository-structure)
3. [Installation on Windows](#installation-on-windows)
4. [Usage](#usage)
5. [System Model](#system-model)
6. [Algorithm Overview](#algorithm-overview)
7. [Experimental Configurations](#experimental-configurations)
8. [Results and Analysis](#results-and-analysis)
9. [Future Work](#future-work)
10. [References](#references)

---

## About

Mobile Edge Computing (MEC) enables resource-constrained mobile devices to offload computation-intensive tasks to nearby edge servers, reducing energy consumption and processing latency. When multiple devices share the same wireless channels and edge resources, the offloading decision of each device affects all others, forming a cooperative multi-agent decision problem.

This project implements and compares two Deep Reinforcement Learning algorithms — **Proximal Policy Optimisation (PPO)** and **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** — for joint task offloading in a simulated four-device MEC system. Both algorithms are built from scratch in PyTorch and trained for 50,000 episodes across five constraint configurations varying deadline thresholds, quality requirements, and wireless channel availability.

---

## Repository Structure

```
marl-mec-offloading/
│
├── src/
│   ├── env.py                        # MEC simulation environment
│   ├── agents/
│   │   ├── ppo_agent.py              # PPO agent (stochastic, on-policy)
│   │   └── maddpg_agent.py           # MADDPG agent (deterministic, off-policy)
│   └── models/
│       ├── ppo_model.py              # Gaussian actor + value-function critic
│       ├── maddpg_model.py           # Deterministic actor + centralised critic
│       └── replay_buffer.py          # Experience replay buffer
│
├── scripts/
│   ├── run_ppo.py                    # PPO training script (CLI arguments)
│   ├── run_maddpg.py                 # MADDPG training script (CLI arguments)
│   └── plot_all_results.py           # Cross-run comparison plot generator
│
├── configs/
│   └── config.py                     # Hyper-parameters and experiment presets
│
├── notebooks/
│   └── final_run_50k.ipynb           # Original Google Colab experiment notebook
│
├── output/                           # Auto-generated plots and CSV logs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation on Windows

All commands below are written for **Windows Command Prompt** (`cmd`) or **Windows PowerShell**. No Unix-specific commands are used.

### Prerequisites

| Software | Minimum Version | Download Link |
|----------|----------------|---------------|
| Python   | 3.9            | https://www.python.org/downloads/ |
| Git      | Any            | https://git-scm.com/download/win  |

> **Important:** During Python installation, tick the checkbox **"Add Python to PATH"**.

---

### Step 1 — Clone the Repository

```cmd
git clone https://github.com/anas-alqadhi/marl-mec-offloading.git
cd marl-mec-offloading
```

---

### Step 2 — Create a Virtual Environment

**Command Prompt:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Once activated, your prompt will show `(venv)` at the beginning.

---

### Step 3 — Install Dependencies

```cmd
pip install -r requirements.txt
```

Verify the installation:

```cmd
python -c "import torch, numpy, pandas, matplotlib; print('All packages installed successfully')"
```

---

## Usage

All scripts are run from the **repository root directory**.

### Train a Single Algorithm

```cmd
python scripts\run_ppo.py
python scripts\run_maddpg.py
```

**With custom parameters:**
```cmd
python scripts\run_ppo.py --run_id my_run --latency 0.3 --quality 0.95 --channels 3
```

#### Available CLI Arguments

| Argument     | Type  | Default                  | Description                                        |
|--------------|-------|--------------------------|----------------------------------------------------|
| `--run_id`   | str   | `run1_lat05_q09_chan2`   | Unique name for this run; determines output folder |
| `--episodes` | int   | `50000`                  | Total number of training episodes                  |
| `--agents`   | int   | `4`                      | Number of mobile device agents                     |
| `--latency`  | float | `0.5`                    | Latency deadline threshold τ (seconds)             |
| `--quality`  | float | `0.9`                    | Minimum quality-of-experience constraint ε         |
| `--channels` | int   | `2`                      | Number of available wireless channels              |
| `--lr`       | float | `1e-4`                   | Learning rate for the Adam optimiser               |
| `--batch`    | int   | `256`                    | Rollout batch size (PPO) / replay batch (MADDPG)   |

---

### Reproduce All Five Experiments

**Run 1 — Baseline:**
```cmd
python scripts\run_ppo.py    --run_id run1_lat05_q09_chan2 --latency 0.5  --quality 0.90 --channels 2
python scripts\run_maddpg.py --run_id run1_lat05_q09_chan2 --latency 0.5  --quality 0.90 --channels 2
```

**Run 2 — Tighter Deadline:**
```cmd
python scripts\run_ppo.py    --run_id run2_lat04_q09_chan2 --latency 0.4  --quality 0.90 --channels 2
python scripts\run_maddpg.py --run_id run2_lat04_q09_chan2 --latency 0.4  --quality 0.90 --channels 2
```

**Run 3 — Higher Quality Requirement:**
```cmd
python scripts\run_ppo.py    --run_id run3_lat03_q095_chan3 --latency 0.3  --quality 0.95 --channels 3
python scripts\run_maddpg.py --run_id run3_lat03_q095_chan3 --latency 0.3  --quality 0.95 --channels 3
```

**Run 4 — Most Constrained:**
```cmd
python scripts\run_ppo.py    --run_id run4_lat02_q10_chan1 --latency 0.2  --quality 1.00 --channels 1
python scripts\run_maddpg.py --run_id run4_lat02_q10_chan1 --latency 0.2  --quality 1.00 --channels 1
```

**Run 5 — Intermediate:**
```cmd
python scripts\run_ppo.py    --run_id run5_lat025_q092_chan2 --latency 0.25 --quality 0.92 --channels 2
python scripts\run_maddpg.py --run_id run5_lat025_q092_chan2 --latency 0.25 --quality 0.92 --channels 2
```

---

### Generate Comparison Plots

```cmd
python scripts\plot_all_results.py
```

Single metric:
```cmd
python scripts\plot_all_results.py --metric Reward
python scripts\plot_all_results.py --metric Energy
```

Open a plot directly from the terminal:
```cmd
start output\run1_lat05_q09_chan2\reward_curve.png
```

---

### Output Files

```
output\
└── run1_lat05_q09_chan2\
    ├── ppo_rewards.csv        ← Episode, Reward, Energy, Latency, Phi (PPO)
    ├── ddpg_rewards.csv       ← Same columns for MADDPG
    ├── reward_curve.png       ← 4-panel metric plot (PPO)
    └── ddpg_curve.png         ← 4-panel metric plot (MADDPG)
```

---

## System Model

### Environment

The simulation models **N = 4** mobile devices sharing a MEC infrastructure. At each discrete time step, every agent receives a local observation and outputs a continuous offloading action.

**Per-agent state vector (dimension = 5):**

| Index | Feature          | Description                            |
|-------|------------------|----------------------------------------|
| 0     | `task_size`      | Computational workload of the task     |
| 1     | `cpu_load`       | Current local CPU utilisation          |
| 2     | `channel_gain`   | Instantaneous wireless channel quality |
| 3     | `battery_level`  | Remaining battery capacity             |
| 4     | `deadline`       | Remaining time before task expiration  |

**Action space:** Continuous scalar per agent ∈ [−1, 1] (−1 = fully local, +1 = fully offloaded).

**Full joint action set { x, k, f_l, β }:**

| Symbol  | Description                                         |
|---------|-----------------------------------------------------|
| **x**   | Binary offloading flag (local vs. edge server)      |
| **k**   | Selected wireless channel index                     |
| **f_l** | Local CPU operating frequency                       |
| **β**   | Video compression ratio applied before transmission |

### Reward Function

```
R(t) = − Σᵢ ||aᵢ||²
```

Auxiliary metrics tracked per step:

| Metric      | Formula                        | Unit    |
|-------------|--------------------------------|---------|
| Energy      | Σ \|aᵢ\| × 0.1                | Joules  |
| Latency     | Uniform sample ∈ [0.4, 0.7]   | Seconds |
| Quality (Φ) | 0.95 − mean(\|aᵢ\|) × 0.1    | [0, 1]  |

---

## Algorithm Overview

### PPO

On-policy actor-critic with clipped surrogate objective:

```
L_CLIP(θ) = E[ min( rₜ(θ) · Âₜ ,  clip(rₜ(θ), 1−ε, 1+ε) · Âₜ ) ] + α · H[π_θ]
```

| Parameter           | Value     |
|---------------------|-----------|
| Learning rate       | 1 × 10⁻⁴  |
| Clipping ε          | 0.1       |
| Discount factor γ   | 0.99      |
| Entropy coefficient | 1 × 10⁻³  |
| Rollout length      | 256 steps |

### MADDPG

Centralised Training / Decentralised Execution (CTDE) with target networks:

```
yᵢ = rᵢ + γ · Q̄ᵢ( o'_all , ā'₁, …, ā'_N )
θ̄  ← τ · θ + (1 − τ) · θ̄     (soft update, τ = 0.005)
```

| Parameter          | Value     |
|--------------------|-----------|
| Learning rate      | 1 × 10⁻⁴  |
| Discount factor γ  | 0.99      |
| Soft update τ      | 0.005     |
| Replay buffer      | 200,000   |
| Mini-batch size    | 256       |

### Comparison

| Property                | PPO              | MADDPG                          |
|-------------------------|------------------|---------------------------------|
| Policy type             | Stochastic       | Deterministic                   |
| Learning paradigm       | On-policy        | Off-policy                      |
| Experience replay       | No               | Yes (200k)                      |
| Convergence speed       | Fast early       | Slower warm-up, sustained gain  |
| Multi-agent scalability | Limited          | Designed for N agents (CTDE)    |

---

## Experimental Configurations

| Run | Run ID                      | τ (s) | ε    | Channels |
|-----|-----------------------------|-------|------|----------|
| 1   | `run1_lat05_q09_chan2`      | 0.50  | 0.90 | 2        |
| 2   | `run2_lat04_q09_chan2`      | 0.40  | 0.90 | 2        |
| 3   | `run3_lat03_q095_chan3`     | 0.30  | 0.95 | 3        |
| 4   | `run4_lat02_q10_chan1`      | 0.20  | 1.00 | 1        |
| 5   | `run5_lat025_q092_chan2`    | 0.25  | 0.92 | 2        |

---

## Results and Analysis

All five runs converge to **−4.0 ± 0.2** reward per episode. The uniformity across configurations is due to the current reward signal penalising only action magnitude, without incorporating constraint-specific penalties. PPO reaches its plateau within approximately 5,000 episodes; MADDPG exhibits a slower but more sustained improvement supported by its replay buffer.

Energy stabilises at **0.31 – 0.33 J**, latency converges to **≈ 0.55 s**, and quality (Φ) settles in the **0.865 – 0.875** band — consistent across all runs. Divergence between configurations is expected once the full utility function is activated.

---

## Future Work

1. Implement the full reward utility function R = f(energy, latency, quality) in `src\env.py`
2. Activate deadline τ and quality ε as explicit reward penalties
3. Add realistic channel interference and bandwidth contention modelling
4. Extend the action space to the full {x, k, f_l, β} joint action set
5. Run multi-seed evaluation (N ≥ 5) and report mean ± std
6. Add MAA2C as a third algorithm for comparison
7. Conduct systematic hyperparameter search

---

## References

1. R. Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments," *NeurIPS*, 2017.
2. J. Schulman et al., "Proximal Policy Optimization Algorithms," *arXiv:1707.06347*, 2017.
3. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, 2015.
4. T. Q. Dinh et al., "Offloading in Mobile Edge Computing: Task Allocation and Computational Frequency Scaling," *IEEE Trans. Commun.*, vol. 65, no. 8, 2017.
5. Y. Mao et al., "A Survey on Mobile Edge Computing: The Communication Perspective," *IEEE Commun. Surv. Tut.*, vol. 19, no. 4, 2017.

---

## License

MIT License — free to use and modify with attribution.
