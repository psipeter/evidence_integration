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
│   ├── runs/                  # model fitting outputs (not tracked)
│   │   ├── response_loss/     # math models + NEF, response loss, 500 trials
│   │   └── joint_loss/        # math models + NEF, joint loss, 100-500 trials
│   └── experiments/           # experiment outputs (not tracked)
│       └── experiment_01/     # error population activity data
├── models/
│   ├── math_models.py         # all mathematical models
│   ├── counting_integrator.py # integrator counting circuit testbed
│   ├── counting_lmu.py        # LMU counting circuit testbed
│   └── NEF.py                 # NEF recurrent and synaptic models
├── fitting/
│   ├── losses.py              # response, shape, joint losses
│   ├── fit.py                 # Optuna fitting with k-fold CV
│   ├── model_params.py         # model parameter search spaces & NEF fixed params
│   ├── submit.py              # job submission and rerun
│   └── collect.py             # result aggregation
├── experiments/
│   └── experiment_01_error_activity.py
├── scripts/
│   ├── model_performance.py
│   ├── response_variability_carrabin.py
│   ├── switch_probability_jiang.py
│   ├── response_change_yoo.py
│   ├── plot_experiment_01.py
│   ├── check_jobs.py          # monitor and cancel finished SLURM jobs
│   └── NEF_plots.py
├── utils/
│   ├── paths.py
│   ├── plot_style.py
│   ├── slurm.py
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
| carrabin | `Bayes` | optimal | — |
| carrabin | `NoisyCounting` | human-matching | mu, sigma_c, nu |
| carrabin | `RL` | naive | alpha |
| jiang | `Bayes` | optimal | beta |
| jiang | `DeGroot` | human-matching | omega, beta |
| jiang | `RL` | naive | alpha, beta |
| yoo | `Mean` | optimal | — |
| yoo | `ADM` | human-matching | phi, rho, nu |
| yoo | `RL` | naive | alpha |
| all | `NEF_recurrent` | neural | lambda_, alpha_0 (+ omega, beta for jiang) |
| all | `NEF_synaptic` | neural | lambda_, alpha_0 (+ omega, beta for jiang) |

---

## Fitting

Fitting uses Optuna with k-fold cross-validation (k=5). Default loss is `response` for all datasets.

```bash
python -m fitting.fit {dataset} {model_type} {pid} [n_trials] [loss_type] [n_runs] [k] [run_folder]
```

Loss functions in `fitting/losses.py`:
- `response` — MSE on carrabin/yoo; total NLL on jiang (requires `beta`)
- `shape` — Wasserstein on response distribution (carrabin), smoothed mean |Δresponse| curve (yoo), switch-vs-conflict aggregates (jiang)
- `joint` — combined response + shape; `JOINT_LOSS_W`: carrabin=0.2, yoo=0.5, jiang=0.95

**Two-loss strategy:** `response_loss` fits are used for model performance figures (best trial-by-trial accuracy). `joint_loss` fits are used for task-specific shape violin plots (best distributional match). Optimizing joint loss compresses response loss differences between models, so the two are kept separate.

---

## Job Management

```bash
# Submit fitting jobs
python -m fitting.submit carrabin RL --n_trials 500 --loss_type response --run_folder response_loss
python -m fitting.submit all --n_trials 500 --loss_type response --run_folder response_loss

# Run locally
python -m fitting.submit carrabin RL --n_trials 10 --local

# Resubmit missing jobs
python -m fitting.submit --resubmit response_loss

# Collect results
python -m fitting.collect response_loss

# Monitor cluster jobs (cancel finished ones)
python scripts/check_jobs.py --cancel
```

Jobs print `JOB_COMPLETE` when finished; `check_jobs.py` uses this to identify done-but-still-running jobs.

---

## Plotting

All figure scripts accept `--run_folder`:

```bash
python scripts/model_performance.py --run_folder response_loss
python scripts/response_variability_carrabin.py --run_folder response_loss
python scripts/switch_probability_jiang.py --run_folder joint_loss
python scripts/response_change_yoo.py --run_folder response_loss
```

Scripts skip missing models gracefully and save PNG (300 dpi) + PDF to `figures/`.

### Conventions
- No `plt.show()` — save to file
- Use `annotate_violins()` for significance brackets
- Run with `SAMPLE_PIDS = None` first to get pid table, then set sample pids

---

## Experiments

Each experiment script handles submission, local running, and collection:

```bash
# Run locally for one pid
python experiments/experiment_01_error_activity.py --pid 1 --dataset carrabin --local

# Submit all pids to cluster
python experiments/experiment_01_error_activity.py --dataset carrabin

# Collect results
python experiments/experiment_01_error_activity.py --collect --dataset carrabin

# Plot
python scripts/plot_experiment_01.py --dataset carrabin
```

### experiment_01_error_activity
Measures mean error population activity (on/off neurons split by encoder threshold=0.5)
and prediction error (raw: obs − prev_response; decoded: error probe) at 100ms into
each observation. Output: `data/experiments/experiment_01/experiment_01_{dataset}.pkl`.

`plot_experiment_01.py` produces a 3-panel figure:
1. On/off neuron activity vs signed PE (obs 3–5)
2. Per-group response std vs firing rate std
3. Per-pid residual firing rate std (after regressing out PE and observation) vs response std

---

## Environment

```bash
conda activate PY311
source venv/bin/activate
```

---

## Cursor Prompt Format

- Title: `Cursor Prompt 00X — Brief description`
- Numbering: sequential, never reset — **next prompt is 218**
- One prompt per response; combine multi-file changes into labeled sections
- File paths relative to project root
- Self-contained — Cursor should execute without reading conversation history

---

## Status

**Complete:**
- All math models (carrabin, jiang, yoo)
- Response, shape, joint loss functions
- Optuna fitting with k-fold CV
- NEF recurrent and synaptic models
- Job management (submit.py, collect.py, check_jobs.py)
- All four main figure scripts
- experiment_01_error_activity.py (carrabin)

**In progress:**
- response_loss fits: math models 500 trials (running), NEF n=1 placeholder
- joint_loss fits: math models 500 trials (running), NEF carrabin 100 trials (overnight)
- experiment_01 correlation analysis: weak with n=1 NEF fits, pending 100-trial rerun

**Pending:**
- NEF fits with 300+ trials for all datasets
- experiment_01 for jiang and yoo
- Additional experiment scripts
- Final analysis and figure generation