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
│   ├── carrabin.pkl
│   ├── jiang.pkl
│   ├── yoo.pkl
│   ├── jiang_networks.npy
│   └── runs/                  # model fitting outputs (not tracked)
│       ├── MSE/               # math models, response loss (MSE or jiang NLL)
│       └── mse_wass/          # math + NEF models, joint loss
├── models/
│   ├── math_models.py         # all mathematical models
│   ├── counting_integrator.py # integrator counting circuit testbed
│   ├── counting_lmu.py        # LMU counting circuit testbed
│   └── NEF.py                 # NEF recurrent and synaptic models
├── fitting/
│   ├── losses.py              # response, shape, joint losses
│   ├── fit.py                 # Optuna fitting with k-fold CV
│   ├── param_ranges.py        # parameter search spaces
│   ├── submit.py              # job submission and rerun
│   └── collect.py             # result aggregation
├── experiments/
│   └── template.py            # template for experiment scripts
├── scripts/
│   ├── model_performance.py
│   ├── response_variability_carrabin.py
│   ├── switch_probability_jiang.py
│   ├── response_change_yoo.py
│   ├── counting_accuracy.py   # counting circuit sweep
│   └── NEF_plots.py           # NEF population dynamics
├── utils/
│   ├── paths.py
│   ├── plot_style.py
│   ├── slurm.py               # SLURM job utilities
│   └── uniform_encoders.py
└── logs/                      # SLURM job logs (not tracked)
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
| all | `NEF_recurrent` | neural | lambda_, alpha_0 (+ omega, beta for jiang) |
| all | `NEF_synaptic` | neural | lambda_, alpha_0 (+ omega, beta for jiang) |

Parameter-free models (`Bayes` for carrabin/jiang, `Mean` for yoo) skip
the Optuna loop and fit only the noise parameter (sigma or beta).

---

## Fitting

Fitting uses Optuna with k-fold cross-validation (k=5). The loss function
is task-aware by default: ``response`` loss for all datasets (MSE on carrabin/yoo,
NLL on jiang).
All outputs are saved to a timestamped run folder under `data/runs/`.

```python
# fit.py entry point
python -m fitting.fit {dataset} {model_type} {pid} [n_trials] [loss_type] [n_runs] [run_folder]
```

Loss type is stored inside `params.pkl` — not in the filename. Rename run
folders manually to track experiment type (e.g. `Apr12_carrabin_response`).

Loss functions in `fitting/losses.py`:
- `response` — response accuracy for all datasets (MSE on carrabin/yoo; total NLL on jiang, requires `beta`)
- `shape` — Wasserstein on response distribution (carrabin), smoothed mean |Δresponse| curve (yoo), or switch-vs-conflict aggregates (jiang; requires `beta`)
- `joint` — combined response + shape; default blend `w` in `JOINT_LOSS_W`: carrabin 0.2, yoo 0.5, jiang 0.3 (override with `wasserstein_w` in params)

---

## Job Management

Job operations are split across submit and collect entry points:

```bash
# Submit a new run (SLURM)
python -m fitting.submit all --n_trials 500
python -m fitting.submit carrabin RL --n_trials 500
python -m fitting.submit carrabin NoisyCounting --n_trials 500 --n_runs 50 --loss_type shape

# Run locally (no SLURM)
python -m fitting.submit carrabin RL --n_trials 10 --local

# Resubmit missing jobs from an existing run
python -m fitting.submit --resubmit Apr12_632pm

# Collect results into combined files
python -m fitting.collect Apr12_632pm

# Dry run
python -m fitting.submit all --dry_run
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
| `MSE` | All math models + NEF, default `response` loss, 300-500 trials |
| `mse_wass` | All math models + NEF, joint loss, 300 trials |

Rename run folders manually to reflect experiment type. Inspect
`run_config.json` inside any folder to see exact hyperparameters used.

---

## NEF Model Development

`models/NEF.py` implements both `NEF_recurrent` and `NEF_synaptic` with the
same `run(params)` interface as the math models.

Counting testbeds are maintained separately:
- `models/counting_integrator.py` — integrator counting circuit
- `models/counting_lmu.py` — LMU counting circuit

Pretraining is dispatched via `_pretrain()` and supports shared counting
subnetwork defaults (`counting="integrator"`, `n_neurons_counting=1000`).

---

## Status

- [x] Port core utility code (`uniform_encoders.py`, `paths.py`, `plot_style.py`)
- [x] Standardize data schema across all three task dataframes
- [x] Port and refactor mathematical models (`math_models.py`)
- [x] Implement response, shape, and joint loss functions (`losses.py`)
- [x] Implement Optuna fitting loop with k-fold CV (`fit.py`)
- [x] Implement counting circuit testbeds (`counting_integrator.py`, `counting_lmu.py`)
- [x] Implement NEF recurrent and synaptic models (`models/NEF.py`)
- [x] Fit NEF models to carrabin and yoo (response and joint loss)
- [x] Implement joint loss for carrabin, yoo, and jiang

- [x] Restructure job management (`fitting/submit.py`, `fitting/collect.py`)
- [x] Create experiments/ framework with template
- [x] Run population-level math model fits on cluster (folder: MSE)
- [x] Create model performance figure (`scripts/model_performance.py`)
- [x] Create response variability figure (`scripts/response_variability_carrabin.py`)
- [x] Create switch probability figure (`scripts/switch_probability_jiang.py`)
- [x] Create response change figure (`scripts/response_change_yoo.py`)
- [ ] Fit NEF models to jiang
- [ ] Collect and analyze full joint-loss fits
- [ ] Design and implement experiment scripts
- [ ] Final analysis and figure generation