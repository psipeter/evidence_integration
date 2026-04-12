# Evidence Integration

This repository contains code for modeling and analyzing individual variability in evidence integration across cognitive tasks in humans. It accompanies a manuscript currently under revision.

---

## Research Overview

To navigate uncertain environments, the brain must continuously integrate new information while weighing recent observations against longer-horizon outcomes. This project investigates the mechanisms underlying individual variability in this process using:

- **Mathematical models** — a prediction-error update rule scaled by the number of observations, capturing a spectrum of integration strategies
- **Behavioral analysis** — applied to three cognitive tasks, capturing trial-by-trial estimation variability, action switching under incongruent social information, and decay in update magnitude
- **Biophysical neural network models** — linking cognitive parameters to observable behavioral heterogeneity, implemented in Nengo (Neural Engineering Framework)

---

## Tasks

| Name | Reference | n | Response type | Key measure |
|---|---|---|---|---|
| `carrabin` | Prat-Carrabin & Woodford (2024) | 21 | Continuous slider | Response distribution variability |
| `jiang` | Jiang et al. (2023) | 224 | Binary choice | Switch probability vs. social conflict |
| `yoo` | Yoo et al. (2025) | 38 | Continuous slider | Power-law decay of update magnitude |

---

## Repository Structure

```
evidence_integration/
├── data/
│   ├── carrabin.pkl          # human behavioral data (tracked)
│   ├── jiang.pkl             # human behavioral data (tracked)
│   ├── yoo.pkl               # human behavioral data (tracked)
│   ├── jiang_networks.npy    # social network adjacency matrices (tracked)
│   └── runs/                 # model fitting outputs (not tracked by git)
│       └── Apr12_632pm/      # example run folder (rename manually)
│           ├── run_config.json
│           ├── RL_carrabin_1_params.pkl
│           └── ...
├── models/
│   ├── math_models.py        # all mathematical models
│   ├── synaptic.py           # NEF synaptic model (TODO)
│   └── recurrent.py          # NEF recurrent model (TODO)
├── fitting/
│   ├── losses.py             # MSE, NLL, Wasserstein loss functions
│   ├── fit.py                # Optuna fitting with k-fold CV
│   └── param_ranges.py       # parameter search spaces
├── jobs/
│   └── run.py                # single entry point for all job management
├── utils/
│   ├── paths.py              # central path config
│   ├── plot_style.py         # shared matplotlib/seaborn style
│   └── uniform_encoders.py   # quasi-Monte Carlo encoders for Nengo
└── notebooks/
    └── performance_mse_nll.ipynb
```

---

## Data Schema

All three task dataframes share a common column schema:

| Column | Description |
|---|---|
| `pid` | Participant ID |
| `trial` | Trial index |
| `observation` | Sequential observation index within trial (carrabin, yoo) |
| `stage` | Social round index — jiang only |
| `value` | Stimulus input |
| `response` | Participant or model output |

Jiang-specific: `network`, `who`, `degree`, `rd`, `stage`
Carrabin-specific: `qid`

### Jiang network data

Network adjacency matrices are stored in `data/jiang_networks.npy`, shape
`(7, 7, 43)`. The `network` column in `jiang.pkl` maps directly to the third
dimension (`graphs[:, :, network_id - 1]`, 1-indexed). Networks 0-3 exist
in the array but were not used in the experiment. Each network has 7 agents
(indexed 0-6); focal participant index = `who - 1` from the stage-0 row.

---

## Models

All models expose a unified interface:

```python
from models.math_models import run
responses = run(params)  # params dict with model_type, dataset, pid, ...
```

| Dataset | Model | Role | Parameters |
|---|---|---|---|
| carrabin | `Bayes` | optimal | sigma only |
| carrabin | `NoisyCounting` | human-matching | mu, sigma_c, nu |
| carrabin | `RL` | naive | alpha |
| jiang | `Bayes` | optimal | beta only |
| jiang | `DeGroot` | human-matching | omega, beta |
| jiang | `RL` | naive | alpha, beta |
| yoo | `Mean` | optimal | sigma only |
| yoo | `ADM` | human-matching | phi, rho, nu |
| yoo | `RL` | naive | alpha |

Parameter-free models (`Bayes` for carrabin/jiang, `Mean` for yoo) skip
the Optuna loop and fit only the noise parameter (sigma or beta).

---

## Fitting

Fitting uses Optuna with k-fold cross-validation (k=5). The loss function
is task-aware by default: MSE for carrabin and yoo, NLL for jiang.
All outputs are saved to a timestamped run folder under `data/runs/`.

```python
# fit.py entry point
python -m fitting.fit {dataset} {model_type} {pid} [n_trials] [loss_type] [n_runs] [run_folder]
```

Loss type is stored inside `params.pkl` — not in the filename. Rename run
folders manually to track experiment type (e.g. `Apr12_carrabin_mse`).

---

## Job Management

All job operations go through a single entry point:

```bash
# Submit a new run (SLURM)
python -m jobs.run all --n_trials 500
python -m jobs.run carrabin RL --n_trials 500
python -m jobs.run carrabin NoisyCounting --n_trials 500 --n_runs 50 --loss_type wasserstein

# Run locally (no SLURM)
python -m jobs.run carrabin RL --n_trials 10 --local

# Resubmit missing jobs from an existing run
python -m jobs.run --resubmit Apr12_632pm

# Collect results into combined files
python -m jobs.run --collect Apr12_632pm

# Dry run
python -m jobs.run all --dry_run
```

Run folders are created automatically with timestamp names (`Apr12_632pm`).
Rename them manually to reflect the experiment. To clear a run, delete the
folder: `rm -rf data/runs/Apr12_632pm`.

---

## Plotting Conventions

All notebooks use `from utils.plot_style import apply_style` for consistent
aesthetics. Colors are assigned by model role using the `colorblind` palette:
- Optimal (Bayes, Mean): `palette[0]`
- Naive (RL): `palette[1]`
- Human-matching (NoisyCounting, DeGroot, ADM): `palette[2]`

Figures are saved as both PNG (300 dpi) and PDF to `figures/`.

---

## Environment

```bash
conda activate PY311
source venv/bin/activate
```

Both environments are kept in sync via `requirements.txt`.

---

## Cursor Prompt Format

When drafting Cursor prompts, Claude should always format them as follows:

- Title: `Cursor Prompt 00X — Brief description`
- Body: single fenced code block with language `markdown`
- Numbering: sequential across entire project, never reset
- One prompt per response; combine multi-file changes into labeled sections
- File paths relative to project root
- Self-contained — Cursor should execute without reading conversation history

---

## Status

- [x] Port core utility code (`uniform_encoders.py`, `paths.py`, `plot_style.py`)
- [x] Standardize data schema across all three task dataframes
- [x] Port and refactor mathematical models (`math_models.py`)
- [x] Implement MSE, NLL, Wasserstein loss functions (`losses.py`)
- [x] Implement Optuna fitting loop with k-fold CV (`fit.py`)
- [x] Implement unified job management (`jobs/run.py`)
- [x] Run population-level math model fits on cluster
- [ ] Port NEF models (`synaptic.py`, `recurrent.py`)
- [ ] Validate refactored code reproduces original results
- [ ] Implement task-specific Experiment 2 losses (switch, decay)
- [ ] Design and implement new experiments
- [ ] Final analysis and figure generation