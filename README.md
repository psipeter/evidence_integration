# Evidence Integration

## Scientific overview

This project studies **individual differences in how people integrate sequential
noisy evidence**, using a combination of mathematical cognitive models and a
biophysical spiking neural network to identify the computational and neural
mechanisms underlying that process.

The central scientific argument proceeds in four steps:

1. **Realistic comparative benchmarking.** The NEF model is evaluated against
   a full spectrum — optimal baselines, simple RL, task-specific cognitive
   models, and a black-box RNN ceiling. The NEF is not expected to win on
   trial-wise RMSE, but must be competitive with cognitive models and
   consistently better than simple baselines.

2. **Cross-task generalisability.** The same NEF architecture is applied to
   multiple tasks (carrabin, yoo). If it achieves reasonable performance across
   tasks where task-specific models cannot generalise, this supports the claim
   that the NEF captures a general cognitive strategy rather than a
   task-specific fit.

3. **Breadth of predictions.** Beyond RMSE, the NEF must capture secondary
   behavioural signatures it was not trained to reproduce — temporal update
   patterns, response noise, state persistence, individual differences in
   lambda and alpha_0 — and generate neural predictions (ensemble activity,
   spiking noise) comparable to data.

4. **Novel testable predictions.** Spiking noise produces state-persistent
   variability distinguishable from response noise; response variability scales
   with n_neurons and alpha_0; neural activity profiles match ensemble dynamics.

### Central model: power-law learning rate

    alpha(t) = alpha_0 / t^lambda

t is observation index within a trial. High lambda: primacy bias; low lambda:
recency bias. In the NEF this emerges from a counting subnetwork modulating
the error ensemble rather than being hardcoded — making alpha(t) an emergent
property of the spiking dynamics.

---

## Tasks

| Name | Reference | N | Response structure | Key features |
|------|-----------|---|--------------------|--------------|
| carrabin | Prat-Carrabin & Woodford (2024) | 21 | Slider after each of 5 obs; binary inputs; sequences repeat (qid) | Response variability, state persistence |
| yoo | Yoo et al. (2025) | 38 | Slider after each of 30 obs; continuous inputs; no sequence repetition | Primacy/recency weight profiles |

Behavioral pickles: data/carrabin.pkl, data/yoo.pkl.
Columns: pid, trial, observation, value, response; carrabin adds qid.

**Proposed new task**: combines repeated sequences (carrabin) with long
sequences and continuous values (yoo). This unlocks all PVTBN metrics
simultaneously and is the intended platform for testing all analyses below.

**Archived** (diederen, jiang, usher): see archive/.

---

## Metric taxonomy (PVTBN framework)

Analyses are organised into five groups. One figure per group per task is the
target; figures may be combined if journal limits require it.

### P — Performance
| Code | Metric |
|------|--------|
| P1 | Overall task performance: mean response vs Bayesian optimal per observation |
| P2 | Trial-wise RMSE: model vs human, boxplots across participants |

### V — Variance
| Code | Metric |
|------|--------|
| V1 | Distributional fit (future MLE-based): model captures full response distribution |
| V2 | Response variability for identical inputs: std(response|obs,qid) per pid; KDE and regplot |
| V3 | Variance broken down by data features (obs position, sequence type, etc.) |
| V4 | State vs response noise decomposition: persistent state noise vs independent response noise |

V2 requires repeated sequences — applicable to carrabin and new task, not yoo.
V4 is the key novel contribution: NEF spiking noise is state-persistent; NoisyCounting
as RMSE-fitted has near-zero state noise (response-noise artefact of RMSE fitting).

### T — Temporal
| Code | Metric |
|------|--------|
| T1 | Task performance vs observation: RMSE as function of observation position |
| T2 | Response change vs observation: mean |delta_response| per observation |
| T3 | Residual variance growth: std(resid|obs,qid) growing across obs (state noise accumulation) |
| T4 | Within-trial residual autocorrelation decay: lag-k correlation, k=1..n_obs (state persistence) |

T3/T4 require repeated sequences — applicable to carrabin and new task, not yoo.

### B — Bias
| Code | Metric |
|------|--------|
| B1 | Weight profile across observations: fitted temporal weighting (flat/primacy/recency/U-shaped) |
| B2 | Surprise sensitivity and confirmation bias: update size by surprise magnitude and direction |

### N — Neural (NEF predictions, no empirical neural data)
| Code | Metric |
|------|--------|
| N1 | Decoded PE timecourse: PE signal decoded from NEF activity within observation window |
| N2 | Response and PE variability vs n_neurons: both decrease with n_neurons; human reference lines |
| N3 | State persistence from spiking noise: NEF prediction driving T3/T4 patterns |
| N4 | (Future) Neural population geometry |

---

## Models

| Dataset | Model | Role | Free params |
|---------|-------|------|-------------|
| carrabin | Mean | Optimal running mean | none |
| carrabin | LeakyIntegrator | Leaky integrator baseline | gamma |
| carrabin | PrimacyRecency | Temporal weighting function | eps_p, eps_r |
| carrabin | NoisyCounting | Task-specific (Prat-Carrabin) | mu, sigma_c, nu |
| carrabin | RL_lambda | Power-law delta rule | alpha_0, lambda_ |
| carrabin | NEF | Spiking NEF integrator | alpha_0, lambda_ |
| yoo | Mean | Optimal running mean | none |
| yoo | LeakyIntegrator | Leaky integrator baseline | gamma |
| yoo | PrimacyRecency | Temporal weighting function | eps_p, eps_r |
| yoo | RL_lambda | Power-law delta rule | alpha_0, lambda_ |
| yoo | NEF | Spiking NEF integrator | alpha_0, lambda_ |

RNN (models/RNN.py): TinyGRU noise estimator — not a cognitive model. Retained
for reference; no longer used in active figure panels. Response variability is
now computed directly as qid-grouped response std.

### NEF architecture

- value ensemble: running estimate (n_neurons=100)
- error ensemble: prediction-error-driven updates (n_neurons=100)
- counting subnetwork: decodes count and weight W=alpha_0/t^lambda
  (n_neurons_counting=100, radius_c=5 carrabin / 30 yoo)

Seed: params["seed"] = int(trial). Trial-to-trial tuning curve variability is
the primary spiking noise source.

Fast counting decoder: Gram matrices precomputed once, saved to
data/counting_activities_n{n}_nc{nc}_{dataset}.pkl. Per Optuna trial,
W_weight recomputed analytically (~300x faster than re-running Nengo).

---

## Repository structure

```
evidence_integration/
  data/
    carrabin.pkl
    yoo.pkl
    counting_activities_n{n}_nc{nc}_{dataset}.pkl
    runs/
      carrabin/
      yoo/
    sim_db/          # future MLE simulation database
  papers/
    carrabin_paper.txt
  archive/
  models/
    math_models.py
    NEF.py
    RNN.py
    counting_integrator.py
  fitting/
    losses.py
    fit.py
    fit_sim_db.py    # future MLE fitting
    model_params.py
    submit.py
    collect.py
  utils/
    paths.py
    plot_style.py
    slurm.py
    carrabin_transform.py
    save_responses.py
  scripts/
    figure_carrabin.py
    figure_yoo.py
    extras_carrabin.py
    extras_yoo.py
    build_sim_db.py
    dynamics_NEF.py
    check_jobs.py
  jobs/
    submit_pe_readout.sh
    submit_n_neurons_scan.sh
    submit_probe_pids.sh
    submit_yoo_noise.sh
  venv/
```

---

## Current figure panel inventory

### figure_carrabin.py (2x4, panels A-H)

| Panel | Category | Content | Status |
|-------|----------|---------|--------|
| A | - | Task schematic | done |
| B | P2 | RMSE boxplots | done |
| C | V2 | KDE of response variability for identical inputs | done |
| D | P2/V2 | Model RMSE vs human response variability regplot | done |
| E | N2/N3 | NEF response and PE variability vs n_neurons | done |
| F | T4 | Within-trial residual autocorrelation decay (human, NEF, NoisyCounting) | done |
| G | T3 | Residual variance growth across observations (human, NEF, NoisyCounting) | done |
| H | - | pending | - |

### figure_yoo.py (2x4, panels A-H)

| Panel | Category | Content | Status |
|-------|----------|---------|--------|
| A | - | Task schematic | done |
| B | P2 | RMSE boxplots | done |
| C | T2 | Response change vs observation | done |
| D | B1 | Decay rate error boxplots (weight profile) | done |
| E-F | - | pending | - |
| G | T1 | Task error curves by obs | done |
| H | - | pending | - |

---

## Fitting pipeline

```
submit -> fit.py (Optuna k-fold CV, RMSE) -> collect -> figures
```

```bash
python -m fitting.submit carrabin NEF --n_trials 100 --run_folder carrabin
python -m fitting.submit carrabin RL_lambda --n_trials 500 --run_folder carrabin
python -m fitting.submit carrabin NoisyCounting --n_trials 500 --run_folder carrabin
python -m fitting.submit carrabin NEF --n_trials 1 --run_folder test --pid 1 --local
python -m fitting.collect carrabin --type params
python -m fitting.collect carrabin --type responses
python scripts/figure_carrabin.py --run_folder carrabin --extra_models NoisyCounting RNN
python scripts/figure_yoo.py --run_folder yoo
```

---

## Environment

Always use: /home/psipeter/evidence_integration/venv/bin/python

Cluster home: /dartfs-hpc/rc/home/n/f007qzn/
All SLURM scripts: use pwd -P and export EVIDENCE_INTEGRATION_ROOT=${ROOT}.

---

## Archive

Older models and data for diederen, jiang, and usher live under archive/.
Do not rely on those paths for active analyses.
