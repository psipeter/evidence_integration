# CLAUDE.md — evidence_integration

This file is the source of truth for Claude when working on this project.
Read it fully before making any changes or suggestions. Prefer this file over
README.md when they conflict.

---

## Scientific goals and hypotheses

This project studies **how people integrate sequential noisy evidence**, with
the goal of identifying the cognitive algorithms and neural mechanisms
underlying that process. The central scientific argument is that a biophysical
spiking neural network model — built using the **Neural Engineering Framework
(NEF)** — captures those mechanisms in a principled, generalisable, and
mechanistically transparent way.

There are four interlocking goals:

### Goal 1 — Realistic comparative model performance
The NEF model must be benchmarked honestly against a spectrum of alternative
models: optimal baselines, simple RL, task-specific cognitive models, and a
black-box RNN ceiling. The expected ordering on trial-wise RMSE is:

    RNN (best) > task-specific model ≈ NEF > simple RL ≥ optimal (worst)

The NEF is not expected to beat task-specific models on trial-wise RMSE
because: (a) it has intrinsic spiking noise that inflates RMSE; (b) its
mechanisms are task-agnostic. Comparable (not superior) RMSE relative to
cognitive models is sufficient, as long as NEF is consistently better than
simple baselines.

### Goal 2 — Cross-task generalisability
The same NEF architecture is applied across multiple tasks. If it achieves
reasonable performance across tasks where task-specific models cannot
generalise, this establishes that the NEF implements a general cognitive
strategy rather than a task-specific fit.

### Goal 3 — Breadth of behavioural and neural predictions
Beyond RMSE, the NEF must also capture secondary behavioural signatures that
it was not trained to reproduce: temporal update patterns, response noise,
state persistence, individual differences in lambda and alpha_0. For neural
predictions, the NEF produces ensemble-level activity traces and spiking noise
magnitude as a function of n_neurons and alpha_0.

### Goal 4 — Novel testable predictions
Spiking noise produces state-persistent variability that differs qualitatively
from response noise; this prediction distinguishes the NEF from NoisyCounting
even when both achieve similar RMSE. Response variability scales with n_neurons
and alpha_0; neural activity profiles match known ensemble dynamics.

---

## Metric taxonomy (PVTBN framework)

All analyses and figure panels are organised under five groups.
Target layout: one figure per group per task; figures may be combined later.

### P — Performance
Metrics measuring how well participants and models do on the task itself.

| Code | Metric | Carrabin | Yoo | New task |
|------|--------|----------|-----|----------|
| P1 | Overall task performance: mean response vs Bayesian optimal, per obs | Y | Y | Y |
| P2 | Trial-wise RMSE: model vs human, boxplots across pids | Y | Y | Y |

### V — Variance
Metrics measuring how noisy participants and models are, including
distributional fit quality.

| Code | Metric | Carrabin | Yoo | New task |
|------|--------|----------|-----|----------|
| V1 | Distributional fit (future MLE-based): model captures full response distribution | future | future | Y |
| V2 | Response variability for identical inputs: std(response|obs,qid) per pid; KDE across sources; regplot vs RMSE | Y | N | Y |
| V3 | Variance broken down by data features: any decomposition showing variance has interesting structure | Y | partial | Y |
| V4 | State vs response noise decomposition: analyses distinguishing persistent state noise from independent response noise | Y | partial | Y |

V4 is a key differentiator for Goal 4. NoisyCounting fitted by RMSE collapses
sigma_c to zero, producing response-noise behaviour (flat autocorrelation, flat
variance growth). The NEF's spiking noise is state-persistent by construction.
V4 analyses expose this distinction and are the primary novel contribution of
the variance group.

Retired metric: RNN-based sigma (cv_rmse from TinyGRU) replaced throughout by
direct qid-grouped response std = mean of std(response | pid, obs, qid) per pid.
RNN files remain in data/runs/carrabin/ but are no longer used in figure panels.

### T — Temporal
Metrics that look at patterns within the observation sequence.

| Code | Metric | Carrabin | Yoo | New task |
|------|--------|----------|-----|----------|
| T1 | Task performance vs observation: RMSE as function of obs position | Y | Y | Y |
| T2 | Response change vs observation: mean |Δresponse| per obs position | Y | Y | Y |
| T3 | Residual variance growth: std(resid|obs,qid) growing across obs | Y | N | Y |
| T4 | Within-trial residual autocorrelation decay: lag-k correlation of residuals, k=1..n_obs | Y | N | Y |

Residuals for T3/T4: response - mean(response | pid, obs, qid). No qid grouping
possible for yoo (no repeated sequences). These become applicable in the new
task by design, which is a key argument for that task's design.

### B — Bias
Metrics investigating suboptimal or structured patterns in human responses.

| Code | Metric | Carrabin | Yoo | New task |
|------|--------|----------|-----|----------|
| B1 | Weight profile across observations: fitted temporal weighting (flat/primacy/recency/U-shaped) | Y | Y | Y |
| B2 | Surprise sensitivity and confirmation bias: update size as function of surprise magnitude and direction; measured while controlling for each other | partial | Y | Y |

Excluded: Bayesian calibration tests, consistent-updates property,
quasi-Bayesian model comparison. These test whether humans are Bayesian, which
is not our claim. Informational insensitivity (flat updates vs obs, Carrabin
Figure 9) is already captured by T2.

### N — Neural
Predictions from NEF neural dynamics. No empirical neural data exists for
these tasks; N panels are testable predictions for future experiments.

| Code | Metric | Carrabin | Yoo | New task |
|------|--------|----------|-----|----------|
| N1 | Decoded PE timecourse: PE signal decoded from NEF activity within obs window | Y | Y | Y |
| N2 | Response and PE variability vs n_neurons: both decrease with n_neurons; human reference lines | Y | future | Y |
| N3 | State persistence from spiking noise: NEF-specific prediction driving T3/T4 patterns | Y | N | Y |
| N4 | (Future) Neural population geometry: dimensionality and structure of counting network manifold | future | future | future |

---

## Central cognitive model

Updates follow a power-law decaying learning rate:

    alpha(t) = alpha_0 / t^lambda

High lambda: primacy bias. Low lambda: recency bias. In the NEF, alpha(t) is
an emergent property of the spiking dynamics (counting subnetwork modulates
the error ensemble) rather than a hardcoded equation.

---

## Active datasets

| Name | N | Task |
|------|---|------|
| carrabin | 21 | Binary inputs; slider after each of 5 obs; sequences repeat (qid) |
| yoo | 38 | Continuous inputs; slider; 30 obs x 30 trials |

Pickles: data/carrabin.pkl, data/yoo.pkl.
Required columns: pid, trial, observation, value, response.
Carrabin adds qid.

Proposed new task: combines repeated sequences (carrabin) with long sequences
and continuous values (yoo). Unlocks all PVTBN metrics simultaneously,
including V2, V4, T3, T4, B2 — currently inapplicable to yoo.

Archived (do not reactivate): diederen, jiang, usher.

---

## Active models

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

RNN (models/RNN.py): TinyGRU noise estimator. Retained for reference; no
longer used in active figure panels.

NoisyCounting note: fitted by RMSE, which does not recover the state-noise
parameter (sigma_c collapses to ~0). This is a known limitation documented in
V4. MLE via Kalman filter would recover correct sigma_c but is not yet
implemented. NoisyCounting as fitted behaves as a response-noise model in
V4/T3/T4 panels, which is informative for Goal 4.

---

## Response variability metric (primary noise measure)

The primary noise metric is:

    qid_response_std(pid) = mean over (obs, qid) groups of std(response | pid, obs, qid)

where each group requires at least 3 trials. Controls for observation position
and input sequence, isolating pure trial-to-trial response noise.
Implemented in figure_carrabin.py via _qid_response_std(). Used in panels
C (KDE), D (regplot vs RMSE), E (n_neurons scan).

---

## Repository structure

```
evidence_integration/
  data/
    carrabin.pkl
    yoo.pkl
    counting_activities_n{n}_nc{nc}_{dataset}.pkl  # precomputed Gram matrices
    runs/
      carrabin/      # primary: carrabin fits + extras + n_neurons_scan
      yoo/           # primary: yoo fits
    sim_db/          # simulation database for future MLE fitting
  papers/
    carrabin_paper.txt   # extracted text of Prat-Carrabin & Woodford 2024
  archive/           # do not import from here
  models/
    math_models.py
    NEF.py
    RNN.py           # retained but not used in active figures
    counting_integrator.py
  fitting/
    losses.py        # RMSE + compute_sim_db_loss (future MLE)
    fit.py           # Optuna k-fold CV, RMSE
    fit_sim_db.py    # future: Optuna + simulation database
    model_params.py
    submit.py
    collect.py
  utils/
    paths.py         # PROJECT_ROOT override via EVIDENCE_INTEGRATION_ROOT env var
    plot_style.py
    slurm.py
    carrabin_transform.py
    save_responses.py
  scripts/
    figure_carrabin.py   # 2x4 main carrabin figure (panels A-H)
    figure_yoo.py        # 2x4 main yoo figure
    extras_carrabin.py   # NEF probe data: pe_readout, probe_timeseries, n_neurons_scan
    extras_yoo.py
    build_sim_db.py      # simulation database builder (future MLE)
    dynamics_NEF.py
    check_jobs.py
    counting_accuracy_scan.py
  jobs/
    submit_pe_readout.sh
    submit_n_neurons_scan.sh   # 21 pids x 5 n_neurons = 105 jobs
    submit_probe_pids.sh
    submit_yoo_noise.sh
  venv/              # always use this Python
```

All new scripts go in scripts/. Never create scripts at the project root.

---

## Current figure panel inventory

### figure_carrabin.py (2x4, panels A-H)

| Panel | Category | Metric | Status |
|-------|----------|--------|--------|
| A | - | Task schematic | done |
| B | P2 | RMSE boxplots, models vs human | done |
| C | V2 | KDE of response variability for identical inputs | done |
| D | P2/V2 | Model RMSE vs human response variability regplot | done |
| E | N2/N3 | NEF response and PE variability vs n_neurons | done |
| F | T4 | Within-trial residual autocorrelation decay (human, NEF, NoisyCounting) | done |
| G | T3 | Residual variance growth across obs (human, NEF, NoisyCounting) | done |
| H | - | pending | - |

### figure_yoo.py (2x4, panels A-H)

| Panel | Category | Metric | Status |
|-------|----------|--------|--------|
| A | - | Task schematic | done |
| B | P2 | RMSE boxplots | done |
| C | T2 | Response change vs observation | done |
| D | B1 | Decay rate error boxplots (primacy/recency weight profile) | done |
| E-F | - | pending | - |
| G | T1 | Task error curves by obs (decaying vs U-shaped groups) | done |
| H | - | pending | - |

---

## Environment

Always use:
    /home/psipeter/evidence_integration/venv/bin/python

Fall back to PY311 conda env only if venv unavailable. Never use base Python.

The cluster resolves ~ to /dartfs-hpc/rc/home/n/f007qzn/.
All SLURM submit scripts use pwd -P for ROOT and export
EVIDENCE_INTEGRATION_ROOT=${ROOT} so Python resolves paths consistently.

---

## NEF implementation

### Architecture
- value ensemble: running estimate (n_neurons=100)
- error ensemble: prediction-error-driven updates (n_neurons=100)
- counting subnetwork: integrator decoding count and weight W=alpha_0/t^lambda
  (n_neurons_counting=100, radius_c=5 for carrabin / 30 for yoo)

### Seed
params["seed"] = int(trial) set directly before each network build.
No base_seed, no pid-level hashing, no trial_seed utility function.
Trial-to-trial variability in tuning curves is the primary noise source.

### Fast counting decoder
Activity files: data/counting_activities_n{n}_nc{nc}_{dataset}.pkl
(new compact format: Mty_basis + mem_readout, ~25-100 MB vs old 1.5 GB).
Per Optuna trial, W_weight recomputed analytically via fast_decode().
Falls back to full _pretrain if file not found.

Generate activity files:
    venv/bin/python models/counting_integrator.py \
        --precompute_activities \
        --n_neurons 100 --n_neurons_counting 100 \
        --dataset carrabin

### Key findings
- qid_response_std: human ~0.10, NEF ~0.07 — NEF in lower half of human range
- NEF response variability scales with n_neurons; converges to human range at n~100-200
- Within-trial residual autocorrelation: human r~0.62, NEF r~0.78 at lag=1 (both decay to ~0.22 at lag=4)
- NoisyCounting (RMSE-fitted): near-zero autocorrelation, flat variance growth — response noise artefact

### Model params source of truth
fitting/model_params.py: _NEF_FIXED, _NEF_RANGES, MODEL_PARAMS.
radius_c is set per-dataset in the "fixed" dict (carrabin=5, yoo=30).

---

## Carrabin response transform

All carrabin models except NoisyCounting apply:

    response = response_raw * t / (t + 2)

Implemented in utils/carrabin_transform.py. Applied inside NEF.run(),
math_models.run(), and utils/save_responses.py. Never apply it twice.

---

## n_neurons scan (carrabin)

scripts/extras_carrabin.py --experiment n_neurons_scan

Scans n_neurons in [25, 50, 100, 200, 400]. Saves raw responses and PE
per (pid, n_neurons) as n_neurons_scan_{pid}_{n_neurons}.pkl. Metrics
(resp_std, pe_std) are computed at plot time in figure_carrabin.py panel E.

Output format after collect: {n_neurons: {"responses": df, "pe_readout": df}}

    bash jobs/submit_n_neurons_scan.sh
    venv/bin/python scripts/extras_carrabin.py \
        --experiment n_neurons_scan --mode collect --out_folder carrabin

---

## Fitting pipeline

    submit -> fit.py (Optuna k-fold CV, RMSE) -> collect -> figures

    python -m fitting.submit carrabin NEF           --n_trials 100 --run_folder carrabin
    python -m fitting.submit carrabin RL_lambda     --n_trials 500 --run_folder carrabin
    python -m fitting.submit carrabin NoisyCounting --n_trials 500 --run_folder carrabin
    python -m fitting.submit carrabin NEF --n_trials 1 --run_folder test --pid 1 --local
    python -m fitting.fit carrabin RL_lambda 1 500 5 carrabin 42
    python -m fitting.collect carrabin --type params
    python -m fitting.collect carrabin --type responses
    python scripts/figure_carrabin.py --run_folder carrabin
    python scripts/figure_carrabin.py --run_folder carrabin --extra_models NoisyCounting RNN
    python scripts/figure_yoo.py --run_folder yoo

---

## Future: simulation database and MLE fitting

scripts/build_sim_db.py and fitting/fit_sim_db.py implement a shared
simulation database for PMMH-style likelihood fitting (V1/V4). Architecture:
- One file per (model, params_hash): stores {seq_tuple: (n_seeds, n_obs) array}
- Simulations shared across all pids (carrabin only — yoo sequences are unique)
- Loss: negative mean log-likelihood from per-observation Gaussian

Currently tabled. Key design decisions documented in conversation history.

---

## Code conventions

- alpha_0, lambda_ (trailing underscore), gamma, eps_p, eps_r
- Merge order: PARAM_DEFAULTS < _NEF_FIXED < fitted Optuna params
- Read loss with _get_loss(perf_df) — never hardcode cv_loss_mean
- Run folder: always pass short name (e.g. carrabin) — resolve_run_folder
  prepends RUNS_DIR. Full path causes double-path bug.
- --local runs must print JOB_COMPLETE as the final stdout line
- Python 3.11; pathlib via utils.paths; figures to figures/ as PNG+PDF
- New figure panels go inside existing figure_*.py scripts
- Do not compute metrics in extras scripts — save raw data, compute in figure scripts

---

## Workflow guidelines

### Before making changes
1. Read the relevant files fully first.
2. Check fitting/model_params.py before touching models or fitting.
3. Propose a plan for structural changes before executing.

### NEF simulations via MCP
Never run NEF simulations directly through MCP tool calls — they take 3-8
minutes per pid and will time out. Write a script and give the run command.
Exceptions: fast operations (loading pkls, one-line analyses) are fine.

### Inspecting data files
Use one-liners:
    venv/bin/python -c "import pandas as pd; df = pd.read_pickle('...'); print(df.head())"

### SLURM job scripts
Prefer --wrap for simple per-pid jobs. Always use pwd -P for ROOT and
export EVIDENCE_INTEGRATION_ROOT=${ROOT}.

### Figure iteration
After any figure change, always regenerate and display the figure image.

### Git
Generate a commit message and wait for confirmation before any git operations.
Never push to git unless explicitly asked.

---

## What NOT to do

- Do not add diederen, jiang, or usher back without explicit plan
- Do not add loss_type, shape_loss, joint_loss, beta hooks
- Do not use trial_seed / base_seed for NEF — seed = int(trial) directly
- Do not read cv_loss_mean directly — use _get_loss
- Do not create scripts outside scripts/
- Do not add NEF_synaptic, LMU counting variant, or ADM model name
- Do not double-apply the carrabin transform
- Do not pass a full path as run_folder — always use a short name
- Do not commit or push without being asked
- Do not run NEF simulations through MCP tool calls (will time out)
- Do not use RNN-based sigma as the noise metric — use qid-grouped response std
- Do not compute metrics in extras scripts — save raw data, compute in figure scripts
