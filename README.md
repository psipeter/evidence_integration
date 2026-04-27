# Evidence Integration

This repository contains code for modeling and analyzing individual variability in evidence integration across cognitive tasks in humans. It accompanies a manuscript currently under revision.

---

## Research Overview

To navigate uncertain environments, the brain must continuously integrate new information while weighing recent observations against longer-horizon outcomes. This project investigates the mechanisms underlying individual variability in this process using:

- **Mathematical models** — a prediction-error update rule scaled by the number of observations
- **Behavioral analysis** — applied to three cognitive tasks
- **Biophysical neural network models** — linking cognitive parameters to behavioral heterogeneity, implemented in Nengo (NEF)

---

## Tasks

| Name | Reference | n | Response type | Key measure |
|---|---|---|---|---|
| `carrabin` | Prat-Carrabin & Woodford (2024) | 21 | Continuous slider | Response distribution variability |
| `jiang` | Jiang et al. (2023) | 209 | Binary choice | Switch probability vs. social conflict |
| `yoo` | Yoo et al. (2025) | 38 | Continuous slider | Power-law decay of update magnitude |

---

## Repository Structure

```
evidence_integration/
├── data/
│   ├── carrabin.pkl / jiang.pkl / yoo.pkl / jiang_networks.npy
│   └── runs/                  # model fitting outputs (not tracked)
├── models/
│   ├── math_models.py         # all mathematical models
│   ├── counting_integrator.py # integrator counting circuit
│   ├── counting_lmu.py        # LMU counting circuit
│   └── NEF.py                 # NEF recurrent and synaptic models
├── fitting/
│   ├── losses.py              # response, shape, joint losses
│   ├── fit.py                 # Optuna fitting with k-fold CV
│   ├── model_params.py        # parameter search spaces and NEF fixed params
│   ├── submit.py              # job submission, resubmit, local run
│   ├── collect.py             # result aggregation (params, responses, activities)
│   └── save_responses.py      # (moved to utils/)
├── experiments/
│   └── experiment_01_error_activity.py
├── utils/
│   ├── paths.py
│   ├── plot_style.py
│   ├── slurm.py
│   ├── save_responses.py      # util: regenerate NEF responses from params
│   └── save_activities.py     # util: save per-neuron activities and encoders
├── scripts/
│   ├── model_performance.py
│   ├── response_variability_carrabin.py
│   ├── switch_probability_jiang.py
│   ├── response_change_yoo.py
│   ├── plot_experiment_01.py
│   ├── plot_activities.py
│   ├── NEF_plots.py
│   └── check_jobs.py
└── logs/                      # SLURM job logs (not tracked)
```

---

## Models

| Dataset | Model | Role | Parameters |
|---|---|---|---|
| carrabin | `Bayes` | optimal | — |
| carrabin | `NoisyCounting` | human-matching | mu, sigma_c, nu |
| carrabin | `RL` | naive | alpha |
| jiang | `Bayes` | optimal | beta |
| jiang | `DeGroot` | human-matching | beta (weights = 1 + rd) |
| jiang | `RL` | naive | alpha, beta |
| yoo | `Mean` | optimal | — |
| yoo | `ADM` | human-matching | phi, rho (nu fixed=0.01) |
| yoo | `RL` | naive | alpha |
| all | `NEF_recurrent` | neural | lambda_, alpha_0 (+ beta for jiang) |
| all | `NEF_synaptic` | neural | lambda_, alpha_0 (+ beta for jiang) |

---

## Fitting

```bash
python -m fitting.submit carrabin NEF_recurrent --n_trials 200 --loss_type joint --run_folder NEF200
python -m fitting.submit --resubmit params --run_folder NEF200
python -m fitting.collect NEF200 --type params
python -m fitting.collect NEF200 --type responses
python -m fitting.collect NEF200 --type activities --ensembles error value counting
```

Loss functions:
- `response` — MSE (carrabin/yoo); NLL (jiang)
- `shape` — Wasserstein distribution distance
- `joint` — combined; weights: carrabin=0.2, yoo=0.5, jiang=0.95

---

## Plotting

```bash
python scripts/model_performance.py --run_folder NEF200
python scripts/plot_activities.py --run_folder NEF200
python scripts/plot_experiment_01.py
```

---

## Environment

```bash
conda activate PY311
source venv/bin/activate
```

---

## Status

**Complete:**
- All math models (carrabin, jiang, yoo)
- Response, shape, joint loss functions
- Optuna fitting with NEF-optimized k-fold CV (single simulation per Optuna trial)
- NEF recurrent and synaptic models with calibrated T_error, tau_error, radius_e
- Job management (submit.py, collect.py, check_jobs.py)
- Activity saving pipeline (save_activities, once_per_obs and once_per_dt)
- All figure scripts including plot_activities.py (5-panel neural activity figure)
- experiment_01 (carrabin + jiang)
- omega removed from all models; DeGroot uses weights = 1 + rd

**In progress:**
- NEF200_v2 fits: carrabin + yoo, 100 trials, new architecture (T_error=0.4, tau_error=0.1, radius_e=2)
- Jiang NEF fits pending architecture validation

**Pending:**
- Jiang NEF fits with updated architecture
- Response loss fits for model performance comparison
- Final figure generation