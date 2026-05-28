# Evidence Integration

## Scientific overview

This project studies **individual differences in how people integrate sequential
noisy evidence**, using a combination of mathematical cognitive models and a
biophysical spiking neural network to identify the computational and neural
mechanisms underlying that process.

The central scientific argument proceeds in four steps:

1. **Realistic comparative benchmarking.** The NEF model is evaluated against
   a full spectrum — optimal baselines, simple RL, task-specific cognitive
   models, and (for the new task) a black-box RNN/LLM ceiling. The NEF is not
   expected to win on trial-wise RMSE, but must be competitive with cognitive
   models and consistently better than simple baselines.

2. **Cross-task generalisability.** The same NEF architecture is applied to
   multiple tasks (carrabin, yoo, new task). If it achieves reasonable
   performance across tasks where task-specific models cannot generalise, this
   supports the claim that the NEF captures a *general* cognitive strategy
   rather than a task-specific fit.

3. **Breadth of predictions.** Beyond RMSE, the NEF must capture secondary
   behavioural signatures it was not trained to reproduce — power-law update
   decay, response noise, individual differences in λ — and generate neural
   predictions (ensemble activity, spiking noise) that can be compared to data.

4. **Novel testable predictions.** The NEF generates predictions about how
   response noise scales with learning rate, how neural populations correspond
   to anatomical areas, and how λ correlates across task conditions — all
   testable in future experiments.

### Central model: power-law learning rate

    α(t) = α₀ / t^λ

`t` is observation index within a trial. High λ → primacy bias; low λ →
recency bias. This is a free individual-difference parameter. In the NEF it
emerges from a counting subnetwork modulating the error ensemble rather than
being hardcoded — making α(t) an emergent property of the spiking dynamics.

---

## Tasks

| Name | Reference | N | Response structure | Key measure |
|------|-----------|---|--------------------|-------------|
| **carrabin** | Prat-Carrabin & Woodford (2024) | 21 | Slider after each obs; sequences repeat (qid) | Per-qid response variability (noise) |
| **yoo** | Yoo et al. (2025) | 38 | Slider after each obs (30 obs × 30 trials) | Power-law decay of update magnitude |
| **new task** | (planned) | 60–80 | Slider; binary + Gaussian conditions | Cross-task λ correlation; λ reliability |

Behavioral pickles: `data/carrabin.pkl`, `data/yoo.pkl`.
Columns: `pid`, `trial`, `observation`, `value`, `response`; carrabin adds `qid`.

**Archived** (diederen, jiang, usher): see `archive/`. Diederen was archived
due to insufficient trials per participant (only ~6) and context-switch
carryover contaminating the learning rate signal. Usher/jiang were archived
because responses were only collected at trial end, providing no within-trial
learning trajectory.

---

## Models

| Dataset | Model | Role | Free params |
|---------|-------|------|-------------|
| carrabin | Bayes | Optimal Bayesian | — |
| carrabin | NoisyCounting | Task-specific (Prat-Carrabin) | `mu`, `sigma_c`, `nu` |
| carrabin | RL | Simple baseline | `alpha` |
| carrabin | RL_lambda | Power-law baseline | `alpha_0`, `lambda_` |
| carrabin | NEF | Spiking NEF integrator | `alpha_0`, `lambda_` |
| carrabin | NEF | Spiking NEF (PES variant) | `alpha_0`, `lambda_` |
| yoo | Mean | Optimal running mean | — |
| yoo | RL | Simple baseline | `alpha` |
| yoo | RL_lambda | Power-law baseline | `alpha_0`, `lambda_` |
| yoo | ADM | Task-specific (Yoo et al.) | `phi`, `rho` |
| yoo | NEF | Spiking NEF integrator | `alpha_0`, `lambda_` |
| yoo | NEF | Spiking NEF (PES variant) | `alpha_0`, `lambda_` |

**NEF architecture:** A recurrent spiking network implements a running
estimate (**value** ensemble), prediction-error-driven updates (**error**
ensemble), and observation counting so the effective learning rate tracks
α(t) (**counting** subnetwork — integrator or LMU). Per-participant `alpha_0`
and `lambda_` are fit with Optuna; architecture and timing live in `_NEF_FIXED`
/ `PARAM_DEFAULTS` in `fitting/model_params.py` and `models/NEF.py`.

---

## Repository structure

```
evidence_integration/
  data/
    carrabin.pkl
    yoo.pkl
    runs/                    # fit outputs (gitignored)
      refit/                 # primary run folder for all final fits
      nef200/                # NEF fits; copy to refit after collecting
  archive/                   # do not import from here
  models/
    math_models.py           # mathematical models (carrabin, yoo)
    NEF.py                   # NEF recurrent & synaptic spiking models
    counting_integrator.py
    counting_lmu.py
  fitting/
    losses.py                # RMSE loss
    fit.py                   # Optuna + k-fold CV
    model_params.py          # MODEL_PARAMS, _NEF_FIXED, _NEF_RANGES
    submit.py                # SLURM submission and --local runner
    collect.py               # aggregate per-participant pickles
  utils/
    paths.py
    plot_style.py
    slurm.py
    run_params.py
    save_responses.py
    save_activities.py
    plot_spikes.py
  scripts/                   # ALL analysis and figure scripts
    figure_carrabin.py       # 2×4 main figure
    figure_yoo.py            # 2×4 main figure
    extras_carrabin.py       # NEF probe data for figure_carrabin bottom panels
    extras_yoo.py            # NEF response-noise simulations
    dynamics_NEF.py          # single-trial NEF dynamics
    iti_perturbation.py      # ITI noise injection experiments
    noise_reliability.py     # response noise estimation reliability (carrabin)
    noise_metric_comparison.py  # model-residual vs qid noise correlation
    trial_obs_reliability_figure.py  # λ reliability vs trials/obs (yoo)
    check_jobs.py
    counting_accuracy.py
  jobs/
    submit_probe_pids.sh
    submit_neurons_scan.sh
    submit_yoo_noise.sh
```

---

## Fitting pipeline

```
submit jobs → fit.py (Optuna k-fold CV) → collect → figures
```

1. **`fitting.submit`** enumerates `(dataset, model_type, pid)` from
   `MODEL_PARAMS`, writes `run_config.json`, and submits SLURM scripts or
   runs locally with `--local`.

2. **`fitting.fit`** runs Optuna TPE with k-fold cross-validation. Objective
   is RMSE between model and human responses. NEF runs one full simulation per
   Optuna trial; CV is evaluated on cached responses.

3. **`fitting.collect`** concatenates per-participant pickles into run-level
   aggregates.

4. **Figure scripts** read from `data/runs/<run_folder>/`.

### Run folder conventions

- **`refit`**: primary folder for all final math model and NEF fits.
- **`nef200`**: intermediate folder for NEF fits (200 Optuna trials). Copy to
  `refit/` after collecting.

### Commands (cluster)

```bash
python -m fitting.submit carrabin NEF --n_trials 200 --run_folder nef200
python -m fitting.submit yoo NEF --n_trials 200 --run_folder nef200
```

### Local single-participant

```bash
python -m fitting.submit carrabin RL_lambda --n_trials 500 --run_folder refit --pid 1 --local
```

### Direct fit.py entrypoint

```bash
python -m fitting.fit <dataset> <model_type> <pid> <n_trials> <k> <run_folder> [seed]
python -m fitting.fit carrabin RL_lambda 1 500 5 refit 42
```

### Collect

```bash
python -m fitting.collect refit --type params
python -m fitting.collect refit --type responses
```

### Resubmit missing

```bash
python -m fitting.submit --resubmit params --run_folder refit
python -m fitting.submit --resubmit responses --run_folder refit
python -m fitting.submit --resubmit activities --run_folder refit --ensembles error
```

---

## Planned new task

**Platform:** jsPsych + MindProbe + Prolific
**Design:** binary (Bernoulli) + continuous (Gaussian) conditions, within-subject,
counterbalanced order. ~30 trials × 15 observations × 4s ≈ 30 min per condition,
targeting ~60 min total session.

**Scientific goals:**
- Cross-task λ correlation (primary): show λ is a stable trait across input modalities
- λ reliability: bootstrap on yoo confirms 30 trials × 15 obs → Spearman r ≈ 0.94
- Response noise: binary condition supports natural prefix collision analysis
- RNN/LLM benchmarking: 60–80 participants × 30 trials enables group-level fits

**Why this design:**
- 15 obs/trial: captures full visible decay without short-window λ bias
  (power-law fit quality flattens after obs 10–12; bias drops sharply at 12+)
- 30 trials: reliability r ≈ 0.94 vs ground truth; additional trials give
  diminishing returns
- Binary + continuous within-subject: tests whether λ is a stable individual
  trait independent of input type, directly extending carrabin to a new population
  and extending the cross-task generalisability argument

---

## Analysis scripts

Key analysis scripts in `scripts/`:

```bash
# λ reliability scan (yoo data)
venv/bin/python scripts/trial_obs_reliability_figure.py

# Response noise reliability analysis (carrabin data)
venv/bin/python scripts/noise_reliability.py

# Model-residual vs qid noise comparison (carrabin)
venv/bin/python scripts/noise_metric_comparison.py

# Main figures
python scripts/figure_carrabin.py --run_folder refit
python scripts/figure_yoo.py --run_folder refit --noise_folder yoo_response_noise
```

---

## Environment

```bash
# Always use project venv
/home/psipeter/evidence_integration/venv/bin/python

# Fallback only if venv unavailable
/home/psipeter/miniconda3/envs/PY311/bin/python
```

Dependencies: numpy, pandas, matplotlib, seaborn, optuna, nengo, scipy.

---

## Archive

Older models and data for diederen, jiang, and usher live under `archive/`.
See `archive/archive_readme.md`. Do not rely on those paths for active analyses.
