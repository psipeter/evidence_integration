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
│       ├── MSE/              # all math models, default losses, 500 trials
│       ├── wasserstein/      # carrabin Bayes/RL/NoisyCounting, wasserstein loss
│       └── switch_probability/ # jiang models with beta-sampled responses
├── models/
│   ├── math_models.py        # all mathematical models
│   ├── counting.py           # NEF counting circuit testbed (integrator, LMU)
│   ├── synaptic.py           # NEF synaptic model (stub)
│   └── recurrent.py          # NEF recurrent model (stub)
├── fitting/
│   ├── losses.py             # MSE, NLL, Wasserstein loss functions
│   ├── fit.py                # Optuna fitting with k-fold CV
│   └── param_ranges.py       # parameter search spaces
├── jobs/
│   └── run.py                # single entry point for all job management
├── scripts/
│   ├── model_performance.py
│   ├── response_variability_carrabin.py
│   ├── switch_probability_jiang.py
│   └── response_change_yoo.py
├── utils/
│   ├── paths.py              # central path config
│   ├── plot_style.py         # shared matplotlib/seaborn style
│   └── uniform_encoders.py   # quasi-Monte Carlo encoders for Nengo
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

Colors are assigned by model role using the `colorblind` palette via
`get_palette()` in `utils/plot_style.py`:
- Optimal (Bayes, Mean): `palette[0]`
- Naive (RL): `palette[1]`
- Human-matching (NoisyCounting, DeGroot, ADM): `palette[2]`
- NEF models: `palette[3]`

All figure scripts import from `utils/plot_style.py`:

```python
from utils.plot_style import apply_style, get_palette, FIGURE_SIZE, SAMPLE_MARKERS
from utils.plot_style import annotate_violins
```

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

## Figures

All finalized figures are standalone Python scripts in `scripts/`:

```bash
python scripts/model_performance.py [run_folder]   # default: MSE
python scripts/response_variability_carrabin.py
python scripts/switch_probability_jiang.py
python scripts/response_change_yoo.py
```

Scripts save both PNG (300 dpi) and PDF to `figures/`.

### Conventions for figure scripts
- No `plt.show()` — save to file and inspect via file browser
- Use `annotate_violins()` from `utils/plot_style` for violin plot significance brackets
- Configuration variables (run folders, sample pids, etc.) live at the top
  of each script — edit there, not in the plotting logic

### Editing sample participants
Run once with `SAMPLE_PIDS = None` to print the pid/parameter table, then
set `SAMPLE_PIDS` at the top of the script and rerun. Example:

    SAMPLE_PIDS = {'narrow': 15, 'medium': 7, 'broad': 4}

---

## Run Folders

Fitted model data lives in `data/runs/` (not tracked by git). Key folders:

| Folder | Contents |
|---|---|
| `MSE` | All math models, MSE/NLL loss, 500 trials |
| `wasserstein` | Carrabin models (Bayes, RL, NoisyCounting), Wasserstein loss, 500 trials, n_runs=50 for NoisyCounting |
| `switch_probability` | Jiang models with beta-sampled binary responses for switch probability analysis |

Rename run folders manually to reflect experiment type. Inspect
`run_config.json` inside any folder to see exact hyperparameters used.

---

## NEF Model Development

The NEF model implements the prediction-error update rule in spiking neurons
using the Neural Engineering Framework (Nengo). Two architectures are planned:

- `models/synaptic.py` — synaptic weight accumulation (primary, in development)
- `models/recurrent.py` — recurrent line-attractor (secondary, stub)

Both will expose the same `run(params)` interface as the math models.

### Counting Circuit (`models/counting.py`)

A standalone testbed for neural counting mechanisms, used to develop and
compare methods for tracking observation count n(t) before integrating into
the full model. Run with:

```bash
python models/counting.py --mechanism {mechanism} [--n_obs 30] [--n_neurons 200] [--seed 0] [--n_seeds 5]
```

**Mechanisms:**

| Mechanism | Description | RMSE (n=30) | Notes |
|---|---|---|---|
| `integrator` | Recurrent line-attractor | ~0.67 | Requires radius=n_obs; drifts at high n |
| `lmu_math` | LMU via Nengo Nodes (no neurons) | ~0.17 | Best current accuracy; no radius problem |
| `lmu_neural` | LMU via spiking EnsembleArray | ~0.33 | Generalizes across scales: ~0.19 (n=5), ~0.19 (n=4) |

**LMU parameters (tuned):**
- `LMU_ORDER = 32` — number of Legendre polynomials
- `LMU_THETA_MULT = 1.1` — theta = n_obs * T_STEP * 1.1
- ZOH discretization via `nengo.utils.filter_design.cont2discrete`
- Pretraining readout uses `LMU_N_OBS_MAX = 30` and generalizes across tasks (carrabin n=5, jiang n<=6, yoo n=30)

**Key findings:**
- Integrator drifts at high n due to boundary effects at `radius=n_obs`
- LMU avoids radius problem entirely; state stays bounded regardless of n_obs
- LMU requires theta ≥ total trial duration to retain all pulse history
- theta=1.1x trial duration and order=32 give best accuracy
- `lmu_neural` pretraining on `LMU_N_OBS_MAX=30` generalizes well across task scales

### Planned Full Model Architecture

The full synaptic NEF model will consist of:

1. **Observation population** — represents current observation o(t)
2. **Delta/differentiator circuit** — detects new observations, outputs pulse
3. **Counting circuit** — tracks n(t) using best mechanism from testbed
4. **Weight population** — computes α(n, λ) from n(t)
5. **Error population** — computes weighted prediction error e(t) = α(n) * (o(t) - v(t))
6. **Context population** — provides stable basis for learning (constant input)
7. **Value population** — represents current estimate v(t)
8. **Estimate synapses** — learnable connections from context to value (PES rule)

The model will be fitted to behavioral data by optimizing λ (decay exponent)
and any noise parameters per participant, using the same Optuna/CV pipeline
as the math models.

---

## Status

- [x] Port core utility code (`uniform_encoders.py`, `paths.py`, `plot_style.py`)
- [x] Standardize data schema across all three task dataframes
- [x] Port and refactor mathematical models (`math_models.py`)
- [x] Implement MSE, NLL, Wasserstein loss functions (`losses.py`)
- [x] Implement Optuna fitting loop with k-fold CV (`fit.py`)
- [x] Implement unified job management (`jobs/run.py`)
- [x] Run population-level math model fits on cluster (folder: MSE)
- [x] Run Wasserstein fits for carrabin (folder: wasserstein)
- [x] Create model performance figure (`scripts/model_performance.py`)
- [x] Create response variability figure (`scripts/response_variability_carrabin.py`)
- [x] Create switch probability figure (`scripts/switch_probability_jiang.py`)
- [x] Create response change figure (`scripts/response_change_yoo.py`)
- [x] Implement counting circuit testbed (`models/counting.py`)
- [x] Implement and tune integrator counting mechanism
- [x] Implement and tune LMU math counting mechanism (RMSE~0.17)
- [ ] Implement LMU neural counting mechanism (`lmu_neural`)
- [ ] Compare integrator vs LMU neural accuracy with multiple seeds
- [ ] Implement full synaptic NEF model (`models/synaptic.py`)
- [ ] Fit NEF model to behavioral data (carrabin, jiang, yoo)
- [ ] Create NEF model performance figures
- [ ] Design and implement new experiments
- [ ] Final analysis and figure generation