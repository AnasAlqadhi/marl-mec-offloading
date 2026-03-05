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
├── assets/                           # Result plots
├── output/                           # Auto-generated plots and CSV logs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation on Windows

### Prerequisites

| Software | Minimum Version | Download Link |
|----------|----------------|---------------|
| Python   | 3.9            | https://www.python.org/downloads/ |
| Git      | Any            | https://git-scm.com/download/win  |

> **Important:** During Python installation, tick the checkbox **"Add Python to PATH"**.

### Step 1 — Clone the Repository

```cmd
git clone https://github.com/AnasAlqadhi/marl-mec-offloading.git
cd marl-mec-offloading
```

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

### Step 3 — Install Dependencies

```cmd
pip install -r requirements.txt
```

Verify:
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

### Generate Comparison Plots

```cmd
python scripts\plot_all_results.py
```

Single metric:
```cmd
python scripts\plot_all_results.py --metric Reward
python scripts\plot_all_results.py --metric Energy
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

### PPO vs. MADDPG — Reward Comparison (Run 5)

![PPO vs MADDPG Reward](assets/algo_compare_run5.png)

PPO converges rapidly within the first ~2,000 episodes and then plateaus. MADDPG starts with high variance during replay buffer warm-up but stabilises at a comparable reward level. Both algorithms settle around **−4.0** under the current energy-proxy reward.

---

### Reward — All 5 Runs

![Reward Comparison All Runs](assets/compare_reward_50k.png)

All five configurations converge to the same narrow reward band of **−4.0 ± 0.2**, confirming that the current reward signal is driven primarily by action magnitude rather than the constraint parameters (τ, ε, channels). Divergence is expected once the full utility function is activated.

---

### Energy Consumption — All 5 Runs

![Energy Comparison All Runs](assets/compare_energy_50k.png)

Energy stabilises at **0.31 – 0.33 J** across all agents and all runs. Both algorithms converge to the same low-power offloading policy within the first few hundred episodes.

---

### Latency — All 5 Runs

![Latency Comparison All Runs](assets/compare_latency_50k.png)

Latency converges uniformly to **≈ 0.55 s** across all configurations. No differentiation between runs is observable at this stage because channel contention is not yet modelled in the environment.

---

### Quality Score (Φ) — All 5 Runs

![Quality Comparison All Runs](assets/compare_phi_50k.png)

Quality stabilises in the **0.865 – 0.875** band for all runs. The narrow spread confirms that quality-constraint enforcement is not yet active in the current reward formulation.

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
