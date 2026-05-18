# Evidence integration

## Research overview

This project studies **individual differences in how people integrate sequential noisy evidence**. Updates follow a **power-law learning rate**:

\[
\alpha(t) = \alpha_0 \,/\, t^{\lambda}
\]

where \(t\) is observation index within a trial. Participants vary in how much weight they put on early versus recent observations (**primacy vs. recency**). The codebase combines **mathematical cognitive models** with a **biophysical spiking network** built in **Nengo** using the **Neural Engineering Framework (NEF)** ([Eliasmith & Anderson, 2003](https://mitpress.mit.edu/9780262250430/neural-engineering/)) so fitted mechanisms can be expressed both as equations and as recurrent spiking dynamics.

---

## Tasks

| Name | Reference | N | Response structure | Key behavioral measure |
|------|-----------|---|--------------------|-------------------------|
| **carrabin** | Prat-Carrabin & Woodford (2024) | 21 | Continuous slider after each observation | Per-question (**qid**) response variability |
| **yoo** | Yoo et al. (2025) | 38 | Continuous slider after each observation (30 obs/trial × 30 trials/participant) | Power-law decay of update magnitude across observations |

Behavioral pickles live under `data/` (e.g. `carrabin.pkl`, `yoo.pkl`). Columns always include at least `pid`, `trial`, `observation`, `value`, and `response`; **carrabin** additionally uses `qid`.

---

## Models

| Dataset | Model | Role | Free parameters |
|---------|--------|------|-----------------|
| **carrabin** | Bayes | Optimal Bayesian integration | *(none)* |
| **carrabin** | NoisyCounting | Human-matching process model (Prat-Carrabin & Woodford 2024) | `mu`, `sigma_c`, `nu` |
| **carrabin** | RL | Rescorla–Wagner / delta rule | `alpha` |
| **carrabin** | RL_lambda | Delta rule with power-law \(\alpha(t)\) | `alpha_0`, `lambda_` |
| **carrabin** | NEF_recurrent | Spiking NEF evidence integrator | `alpha_0`, `lambda_` |
| **carrabin** | NEF_synaptic | Spiking NEF (synaptic-learning variant) | `alpha_0`, `lambda_` |
| **yoo** | Mean | Optimal running mean | *(none)* |
| **yoo** | RL | Delta rule | `alpha` |
| **yoo** | RL_lambda | Delta rule with power-law \(\alpha(t)\) | `alpha_0`, `lambda_` |
| **yoo** | ADM | Adaptive decision-making (Yoo et al. 2025) | `phi`, `rho` |
| **yoo** | NEF_recurrent | Spiking NEF evidence integrator | `alpha_0`, `lambda_` |
| **yoo** | NEF_synaptic | Spiking NEF (synaptic-learning variant) | `alpha_0`, `lambda_` |

**NEF (recurrent / synaptic):** A recurrent spiking network implements a running estimate (**value** ensemble), prediction-error-driven updates (**error** ensemble), and observation counting so effective learning rate tracks \(\alpha(t)\) (**counting** subnetwork—integrator or LMU). Per-participant **`alpha_0`** and **`lambda_`** are fit with Optuna; architecture and timing live in **`_NEF_FIXED`** / **`PARAM_DEFAULTS`** (see `fitting/model_params.py` and `models/NEF.py`).

---

## Repository structure

```
evidence_integration/
  data/
    carrabin.pkl           # behavioral data (active)
    yoo.pkl                # behavioral data (active)
    runs/                  # fit outputs (gitignored / not version-controlled)
  archive/                 # archived code & data — see archive/archive_readme.md
  models/
    math_models.py         # mathematical models (carrabin, yoo)
    NEF.py                 # NEF recurrent & synaptic spiking models
    counting_integrator.py
    counting_lmu.py
  fitting/
    losses.py              # RMSE loss + figure/diagnostic helpers
    fit.py                 # Optuna + k-fold CV
    model_params.py        # MODEL_PARAMS, _NEF_FIXED
    submit.py              # SLURM submission, resubmit, local runs
    collect.py             # aggregate per-participant pickles
  utils/
    paths.py               # PROJECT_ROOT, DATA_DIR, RUNS_DIR, resolve_run_folder, …
    plot_style.py          # matplotlib/seaborn defaults, palettes
    slurm.py               # job scripts, default time/mem tables
    run_params.py          # load_run_params, trial_seed
    save_responses.py      # regenerate NEF responses from best params
    plot_spikes.py         # spike raster helpers (used where needed)
    save_activities.py     # per-neuron activities & encoders (NEF)
  scripts/
    figure_carrabin.py     # main carrabin figure
    figure_yoo.py          # main yoo figure
    dynamics_NEF.py        # single-trial NEF dynamics figure
    iti_perturbation.py    # ITI noise injection experiments
    check_jobs.py          # SLURM job cleanup / status helper
    build_diederen.py      # auxiliary / exploratory (not in main fitting pipeline)
    counting_accuracy.py   # auxiliary counting diagnostics
```

---

## Fitting pipeline

Typical loop: **submit jobs → each job runs `fitting.fit` → collect aggregates figures**.

1. **`fitting.submit`** enumerates `(dataset, model_type, pid)` from **`MODEL_PARAMS`** (optionally filtered), writes **`run_config.json`** under `data/runs/<run_folder>/`, and either submits SLURM scripts or runs **`fitting.fit`** locally (`--local`).
2. **`fitting.fit`** runs Optuna with **k-fold cross-validation**. The objective is **RMSE** between model and human responses (`fitting.losses.response_loss`). Math models are re-simulated per trial; **NEF** runs one full simulation per Optuna trial and CV is evaluated on cached responses.
3. **`fitting.collect`** reads **`run_config.json`** and concatenates per-participant **`_params.pkl`**, **`_performance.pkl`**, **`_folds.pkl`**, **`_responses.pkl`**, or activity files into run-level aggregates.
4. **Figure scripts** read from `data/runs/<run_folder>/` (and optional side folders such as noise experiments for yoo).

### Commands (cluster)

```bash
python -m fitting.submit carrabin NEF_recurrent --n_trials 200 --run_folder response
python -m fitting.submit yoo NEF_recurrent --n_trials 200 --run_folder response
```

### Local single-participant example

```bash
python -m fitting.submit carrabin RL_lambda --n_trials 500 --run_folder response --pid 1 --local
```

### Collect

```bash
python -m fitting.collect response --type params
python -m fitting.collect response --type responses
```

### Resubmit missing artifacts

```bash
python -m fitting.submit --resubmit params --run_folder response
python -m fitting.submit --resubmit responses --run_folder response
python -m fitting.submit --resubmit activities --run_folder response --ensembles error
```

### Direct `fitting.fit` entrypoint (positional arguments)

There is **no** `--n_trials` flag on the module CLI. Order:

```bash
python -m fitting.fit <dataset> <model_type> <pid> <n_trials> <k> <run_folder> [optuna_seed]
```

Examples:

```bash
python -m fitting.fit carrabin RL_lambda 1 500 5 response 42
python -m fitting.fit yoo NEF_recurrent 14 200 5 response 42
```

If you pass **only five** tokens after the script (`dataset`, `model_type`, `pid`, `n_trials`, `run_folder`), **`k`** defaults to **5**. Passing **`k`** explicitly requires **seven** argv tokens total including the script name (`… pid n_trials k run_folder`).

### Warm-start for NEF

If **`RL_lambda_<dataset>_<pid>_params.pkl`** already exists in the same run folder, **`fitting.fit`** enqueues those **`alpha_0`** / **`lambda_`** values as the first Optuna trial for **NEF** models. Fit **RL_lambda** (or copy equivalent pickles) **before** large **NEF** searches when you want that seed.

### Run folder naming

Prefer a **short folder name** (e.g. `response`). **`RUNS_DIR / run_folder`** is `data/runs/<name>`. The codebase also normalizes mistaken relative paths such as `data/runs/foo` via **`utils.paths.resolve_run_folder`**—short names remain the clearest convention.

### Local SLURM helpers

Anything run with **`--local`** must print **`JOB_COMPLETE`** as its **last** line so **`scripts/check_jobs.py`** can detect completion.

---

## Activities (NEF)

Save or resubmit ensemble traces after fits:

```bash
python -m fitting.submit --resubmit activities --run_folder response --ensembles error --timing once_per_obs
python -m fitting.collect response --type activities --ensembles error
```

Single-participant CLI:

```bash
python -m utils.save_activities carrabin NEF_recurrent 1 response error once_per_obs
```

---

## Figures

Bottom panels of the carrabin figure need probe data from `scripts/extras_carrabin.py`
(see that module docstring and `jobs/submit_probe_pids.sh` / `jobs/submit_neurons_scan.sh`).

```bash
python scripts/figure_carrabin.py --run_folder refit
python scripts/figure_yoo.py --run_folder response --noise_folder yoo_response_noise
python scripts/dynamics_NEF.py --dataset carrabin --pid 1 --run_folder response
```

Outputs go to **`figures/`** (PNG/PDF; some scripts also write SVG).

---

## Environment

```bash
conda activate PY311    # or your Python 3.11 scientific env
source venv/bin/activate   # project venv on top (recommended)
```

Dependencies include **numpy**, **pandas**, **matplotlib**, **seaborn**, **optuna**, **nengo**, **scipy**, etc. (see `requirements.txt` / env docs if present).

---

## Archive

Older **jiang** / **usher** task code, models, losses, and data live under **`archive/`**. Do not rely on those paths for active analyses. See **`archive/archive_readme.md`** for layout and how to restore material if needed.
