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
| **diederen** | Diederen & Schultz (2015, 2017) | 85 | Continuous slider after each observation (interleaved distributions, 6 trials/participant) | Learning-rate decay across observations; context-switch carryover bias |

Behavioral pickles live under `data/` (e.g. `carrabin.pkl`, `yoo.pkl`, `diederen.pkl`). Columns always include at least `pid`, `trial`, `observation`, `value`, and `response`; **carrabin** additionally uses `qid`.

**Diederen data:** built from raw MATLAB files with:

```bash
python scripts/build_diederen.py --data_dir data/Diederen
```

Groups: **CTRL** (n=28), **PCB** / **SUL** / **BRO** (n=19 each).

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
| **diederen** | Mean | Optimal running mean (flat prior) | *(none)* |
| **diederen** | RL | Delta rule | `alpha` |
| **diederen** | RL_lambda | Delta rule with power-law \(\alpha(t)\) | `alpha_0`, `lambda_` |
| **diederen** | PearceHall | Surprise-driven adaptive \(\alpha\) (Pearce & Hall 1980; Diederen & Schultz 2015) | `alpha_0`, `eta` |
| **diederen** | NEF_recurrent | Spiking NEF evidence integrator | `alpha_0`, `lambda_` |
| **diederen** | NEF_synaptic | Spiking NEF (synaptic-learning variant) | `alpha_0`, `lambda_` |

**NEF (recurrent / synaptic):** A recurrent spiking network implements a running estimate (**value** ensemble), prediction-error-driven updates (**error** ensemble), and observation counting so effective learning rate tracks \(\alpha(t)\) (**counting** subnetwork—integrator or LMU). Per-participant **`alpha_0`** and **`lambda_`** are fit with Optuna; architecture and timing live in **`_NEF_FIXED`** / **`PARAM_DEFAULTS`** (see `fitting/model_params.py` and `models/NEF.py`).

**PearceHall (diederen):** \(\alpha(t+1) = \eta \cdot |\delta(t)| + (1 - \eta) \cdot \alpha(t)\), with \(\alpha\) clipped to \([0, 2]\). **`eta=0`** recovers fixed-alpha RL.

**Diederen catch trials:** catch trials are included in the value sequence for simulation (reward is shown) but **excluded from RMSE loss**. Only **`missed`** rows are excluded from simulation.

---

## Repository structure

```
evidence_integration/
  data/
    carrabin.pkl           # behavioral data
    yoo.pkl                # behavioral data
    diederen.pkl           # behavioral data (built by build_diederen.py)
    runs/                  # fit outputs (gitignored / not version-controlled)
      refit/               # primary run folder for all final fits
      nef200/              # NEF fits (200 Optuna trials); copied to refit after collection
  archive/                 # archived code & data — see archive/archive_readme.md
  models/
    math_models.py         # mathematical models (carrabin, yoo, diederen)
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
    figure_carrabin.py     # carrabin figure (2×4, panels A–H)
    figure_yoo.py          # yoo figure (2×4, panels A–H)
    figure_diederen.py     # diederen figure (2×4, panels A–H)
    extras_carrabin.py     # NEF probe data for figure_carrabin bottom panels
    extras_yoo.py          # NEF response-noise simulations for figure_yoo panel H
    build_diederen.py      # build data/diederen.pkl from raw MATLAB files
    dynamics_NEF.py        # single-trial NEF dynamics figure
    iti_perturbation.py    # ITI noise injection experiments
    check_jobs.py          # SLURM job cleanup / status helper
    counting_accuracy.py   # auxiliary counting diagnostics
  jobs/
    submit_probe_pids.sh       # submit extras_carrabin probe_pids experiment
    submit_neurons_scan.sh     # submit extras_carrabin n_neurons_scan experiment
    submit_yoo_noise.sh        # submit extras_yoo response-noise simulations
```

---

## Fitting pipeline

Typical loop: **submit jobs → each job runs `fitting.fit` → collect aggregates → figures**.

1. **`fitting.submit`** enumerates `(dataset, model_type, pid)` from **`MODEL_PARAMS`** (optionally filtered), writes **`run_config.json`** under `data/runs/<run_folder>/`, and either submits SLURM scripts or runs **`fitting.fit`** locally (`--local`).
2. **`fitting.fit`** runs Optuna with **k-fold cross-validation**. The objective is **RMSE** between model and human responses (`fitting.losses.response_loss`). Math models are re-simulated per trial; **NEF** runs one full simulation per Optuna trial and CV is evaluated on cached responses.
3. **`fitting.collect`** reads **`run_config.json`** and concatenates per-participant **`_params.pkl`**, **`_performance.pkl`**, **`_folds.pkl`**, **`_responses.pkl`**, or activity files into run-level aggregates.
4. **Figure scripts** read from `data/runs/<run_folder>/` (and optional side folders such as **`yoo_response_noise/`** for yoo panel H).

### Run folder conventions

- **`refit`**: primary run folder for all final math model and NEF fits. All figure scripts default to **`--run_folder refit`**.
- **`nef200`**: intermediate folder for NEF fits run with **200** Optuna trials. After collection, copy combined NEF pickles from **`nef200/`** to **`refit/`** (see regeneration guide below).

Prefer a **short folder name** (e.g. `refit`, `nef200`). **`RUNS_DIR / run_folder`** is `data/runs/<name>`. The codebase also normalizes mistaken relative paths such as `data/runs/foo` via **`utils.paths.resolve_run_folder`**—short names remain the clearest convention.

### Commands (cluster)

```bash
python -m fitting.submit carrabin NEF_recurrent --n_trials 200 --run_folder nef200
python -m fitting.submit yoo NEF_recurrent --n_trials 200 --run_folder nef200
python -m fitting.submit diederen NEF_recurrent --n_trials 200 --run_folder nef200
```

### Local single-participant example

```bash
python -m fitting.submit carrabin RL_lambda --n_trials 500 --run_folder refit --pid 1 --local
```

### Collect

```bash
python -m fitting.collect refit --type params
python -m fitting.collect refit --type responses
```

### Resubmit missing artifacts

```bash
python -m fitting.submit --resubmit params --run_folder refit
python -m fitting.submit --resubmit responses --run_folder refit
python -m fitting.submit --resubmit activities --run_folder refit --ensembles error
```

### Direct `fitting.fit` entrypoint (positional arguments)

There is **no** `--n_trials` flag on the module CLI. Order:

```bash
python -m fitting.fit <dataset> <model_type> <pid> <n_trials> <k> <run_folder> [optuna_seed]
```

Examples:

```bash
python -m fitting.fit carrabin RL_lambda 1 500 5 refit 42
python -m fitting.fit yoo NEF_recurrent 14 200 5 refit 42
python -m fitting.fit diederen PearceHall 1 500 5 refit 42
```

If you pass **only five** tokens after the script (`dataset`, `model_type`, `pid`, `n_trials`, `run_folder`), **`k`** defaults to **5**. Passing **`k`** explicitly requires **seven** argv tokens total including the script name (`… pid n_trials k run_folder`).

### Warm-start for NEF

If **`RL_lambda_<dataset>_<pid>_params.pkl`** already exists in the same run folder, **`fitting.fit`** enqueues those **`alpha_0`** / **`lambda_`** values as the first Optuna trial for **NEF** models. Fit **RL_lambda** (or copy equivalent pickles) **before** large **NEF** searches when you want that seed.

### Local SLURM helpers

Anything run with **`--local`** must print **`JOB_COMPLETE`** as its **last** line so **`scripts/check_jobs.py`** can detect completion.

---

## Complete figure data regeneration

Commands to regenerate all data required by the three figures, in order.
Run each block after the previous one completes. All commands assume the
working directory is the project root.

---

### 1. Build diederen dataset (if not already built)

```bash
python scripts/build_diederen.py --data_dir data/Diederen
```

---

### 2. Fit math models (all datasets)

```bash
# carrabin
python -m fitting.submit carrabin Bayes         --n_trials 200 --run_folder refit
python -m fitting.submit carrabin RL            --n_trials 500 --run_folder refit
python -m fitting.submit carrabin RL_lambda     --n_trials 500 --run_folder refit
python -m fitting.submit carrabin NoisyCounting --n_trials 500 --run_folder refit

# yoo
python -m fitting.submit yoo Mean       --n_trials 200 --run_folder refit
python -m fitting.submit yoo RL         --n_trials 500 --run_folder refit
python -m fitting.submit yoo RL_lambda  --n_trials 500 --run_folder refit
python -m fitting.submit yoo ADM        --n_trials 500 --run_folder refit

# diederen
python -m fitting.submit diederen Mean       --n_trials 200 --run_folder refit
python -m fitting.submit diederen RL         --n_trials 500 --run_folder refit
python -m fitting.submit diederen RL_lambda  --n_trials 500 --run_folder refit
python -m fitting.submit diederen PearceHall --n_trials 500 --run_folder refit
```

Collect after all jobs complete:

```bash
python -m fitting.collect refit --type params
python -m fitting.collect refit --type responses
```

---

### 3. Fit NEF models

NEF fits are saved to **`nef200/`** (200 Optuna trials) then copied to **`refit/`**.
Fit **RL_lambda** first to provide warm-start seeds for NEF.

```bash
# Warm-start: ensure RL_lambda fits exist in refit/ (step 2 above)

# Submit NEF fits to nef200/
python -m fitting.submit carrabin NEF_recurrent --n_trials 200 --run_folder nef200
python -m fitting.submit carrabin NEF_synaptic  --n_trials 200 --run_folder nef200
python -m fitting.submit yoo      NEF_recurrent --n_trials 200 --run_folder nef200
python -m fitting.submit yoo      NEF_synaptic  --n_trials 200 --run_folder nef200
python -m fitting.submit diederen NEF_recurrent --n_trials 200 --run_folder nef200
python -m fitting.submit diederen NEF_synaptic  --n_trials 200 --run_folder nef200
```

Collect from **nef200**, then copy to **refit**:

```bash
python -m fitting.collect nef200 --type params
python -m fitting.collect nef200 --type responses

cp data/runs/nef200/NEF_recurrent_carrabin_*.pkl data/runs/refit/
cp data/runs/nef200/NEF_synaptic_carrabin_*.pkl  data/runs/refit/
cp data/runs/nef200/NEF_recurrent_yoo_*.pkl      data/runs/refit/
cp data/runs/nef200/NEF_synaptic_yoo_*.pkl       data/runs/refit/
cp data/runs/nef200/NEF_recurrent_diederen_*.pkl data/runs/refit/
cp data/runs/nef200/NEF_synaptic_diederen_*.pkl  data/runs/refit/
```

---

### 4. Save NEF activities (for figure_carrabin and figure_yoo neural panels)

```bash
python -m fitting.submit --resubmit activities --run_folder refit \
    --ensembles error --timing once_per_obs
python -m fitting.collect refit --type activities --ensembles error
```

---

### 5. Generate extras_carrabin probe data (for figure_carrabin bottom panels)

```bash
# Submit (one job per pid):
bash jobs/submit_probe_pids.sh

# Collect after all jobs complete:
python scripts/extras_carrabin.py --experiment probe_pids \
    --mode collect --out_folder refit

# Submit n_neurons scan:
bash jobs/submit_neurons_scan.sh

# Collect after all jobs complete:
python scripts/extras_carrabin.py --experiment n_neurons_scan \
    --mode collect --out_folder refit
```

---

### 6. Generate extras_yoo response-noise data (for figure_yoo panel H)

```bash
# Submit (one job per pid × seed):
bash jobs/submit_yoo_noise.sh

# Collect after all jobs complete:
python scripts/extras_yoo.py --mode collect --n_seeds 10 \
    --run_folder yoo_response_noise
```

---

### 7. Generate figures

```bash
python scripts/figure_carrabin.py --run_folder refit
python scripts/figure_yoo.py --run_folder refit \
    --noise_folder yoo_response_noise
python scripts/figure_diederen.py --run_folder refit
```

To include **RL_lambda** in top-row model panels:

```bash
python scripts/figure_carrabin.py --run_folder refit --include_rl_lambda
python scripts/figure_yoo.py      --run_folder refit --include_rl_lambda \
    --noise_folder yoo_response_noise
python scripts/figure_diederen.py --run_folder refit --include_rl_lambda
```

---

## Activities (NEF)

Save or resubmit ensemble traces after fits:

```bash
python -m fitting.submit --resubmit activities --run_folder refit \
    --ensembles error --timing once_per_obs
python -m fitting.collect refit --type activities --ensembles error
```

Single-participant CLI:

```bash
python -m utils.save_activities carrabin NEF_recurrent 1 refit error once_per_obs
```

---

## Figures

Main summary figures (2×4 layouts). **Carrabin** bottom row (panels E–H) requires
**extras_carrabin** probe/scan data (see regeneration guide). **Yoo** panel H
requires **extras_yoo** response-noise data in **`yoo_response_noise/`**.

```bash
python scripts/figure_carrabin.py --run_folder refit
python scripts/figure_yoo.py      --run_folder refit --noise_folder yoo_response_noise
python scripts/figure_diederen.py --run_folder refit
```

Other figure scripts:

```bash
python scripts/dynamics_NEF.py --dataset carrabin --pid 1 --run_folder refit
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
