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
| `usher` | Rosenbaum et al. (2021) | 97 | Continuous slider (final obs only) | Trial-wise RMSE |

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
│   ├── response_change_vs_weight_activity.py  # yoo: brain–behavior weights vs activity
│   ├── noisy_representations.py             # carrabin: neural vs response noise
│   ├── iti_perturbation.py                  # ITI noise injection experiments
│   ├── plot_iti_perturbation.py             # ITI perturbation figure
│   ├── plot_experiment_01.py
│   ├── plot_activities.py
│   ├── NEF_plots.py                         # NEF dynamics visualization
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
| carrabin | `RL_lambda_offset` | decaying learning (offset parametrization) | alpha_0, lambda_ |
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

Run outputs are written under `data/runs/<run_folder>/`. Common folder names include `response`, `noisy_representations`, and `iti_perturbation` (in addition to older experiment-specific names).

```bash
# New fits
python -m fitting.submit carrabin NEF_recurrent --n_trials 200 --loss_type response --run_folder response
python -m fitting.submit carrabin NEF_recurrent --run_folder response --pid 1 --local
python -m fitting.submit jiang DeGroot --n_trials 500 --loss_type response --run_folder response

# Resubmit missing jobs (same interface as save/regenerate responses)
python -m fitting.submit --resubmit params --run_folder response
python -m fitting.submit --resubmit responses --run_folder response
python -m fitting.submit --resubmit responses --run_folder response --model_type NEF_recurrent
python -m fitting.submit carrabin --resubmit responses --run_folder response --pid 3
python -m fitting.submit --resubmit activities --run_folder response --ensembles error value counting
python -m fitting.submit --resubmit activities --run_folder response --ensembles error --timing once_per_dt

python -m fitting.collect response --type params
python -m fitting.collect response --type responses
python -m fitting.collect response --type activities --ensembles error value counting
python -m fitting.collect response --type activities --ensembles error --timing once_per_dt
```

With `--resubmit`, jobs listed in `run_config.json` can be filtered by **dataset** (first positional argument, default `all`), optional **model type** (second positional), and **`--pid`**.

Per-model SLURM time limits are set in `utils/slurm.py` (`DEFAULT_TIME_LIMITS`), including `RL_lambda`, `RL_lambda_rd`, and `RL_lambda_offset` (2:0:0), `NEF_recurrent` / `NEF_synaptic` (72:0:0), and lighter limits for analytic models.

Loss functions (`fitting/losses.py`):
- **`response`** (default fitting objective) — root mean squared error vs human sliders on **carrabin** and **yoo**; **mean negative log-likelihood** of human binary choices on **jiang** under `sigmoid(beta · model_prediction)` with fitted `beta`.
- **`shape`** — diagnostics / alternate objective: **jiang** uses mean |Δcoefficient| between human and model for **network-degree–weighted OLS** coefficients predicting stage response sign (averaged over stages 1–2); carrabin and yoo use shape metrics described in the module docstring (e.g., per-qid variability and Wasserstein on smoothed delta curves).
- **`joint`** — weighted mix of response + shape (`JOINT_LOSS_W`); available for experiments but **not** the default workflow (`--loss_type response` is standard).

---

## Plotting

```bash
python scripts/model_performance.py --run_folder response
python scripts/plot_activities.py --run_folder response
python scripts/plot_experiment_01.py
python scripts/response_change_vs_weight_activity.py --run_folder response
python scripts/noisy_representations.py --experiment probe_pids --run_folder response --out_folder noisy_representations
python scripts/iti_perturbation.py --experiment probe_conditions --run_simulation --pid 14 --out_folder iti_perturbation
python scripts/plot_iti_perturbation.py --out_folder iti_perturbation --pid 14
python scripts/NEF_plots.py --dataset carrabin --model_type NEF_recurrent --pid 14
```

---

## Environment

```bash
conda activate PY311
source venv/bin/activate
```

---

## Status

**Complete (May 2026):**
- All math models (carrabin, jiang, yoo), including **`RL_lambda_offset`** on carrabin where applicable
- **Response** loss as default optimizer target; shape / joint objectives for diagnostics or specialized fits
- Optuna fitting with NEF k-fold CV (single simulation per Optuna trial); warm-start from **`RL_lambda`** checkpoints for NEF (`fitting/fit.py`)
- NEF recurrent and synaptic models with calibrated `T_error`, `tau_error`, `radius_e`; **base_seed** vs per-trial **trial_seed** split for counting versus remainder of network
- ITI perturbation tooling (`scripts/iti_perturbation.py`, hook in **`NEF._simulate_trial`**); probe pickles store **all trials** when saved
- Job management (`submit.py`, `collect.py`, `check_jobs.py`); resubmission regenerates responses via **`--resubmit responses`** (replacing legacy “rerun responses” wording)
- Activity saving pipeline (`save_activities`, `once_per_obs` and `once_per_dt`); readout timing uses **`READOUT_OFFSET = 0.5`** s in `utils/save_activities.py`
- Figure and analysis scripts, including **`noisy_representations.py`**, **`plot_iti_perturbation.py`**, **`NEF_plots.py`**, and **`response_change_vs_weight_activity.py`**
- experiment_01 (carrabin + jiang); DeGroot uses weights = 1 + rd

**Ongoing:**
- Fits under `data/runs/response/` and related run folders; jiang / NEF coverage may grow with new hypotheses and scans