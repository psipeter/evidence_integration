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
Figures save PDF only (no PNG/SVG).

### P — Performance
Metrics measuring how well participants and models do on the task itself.

| Code | Metric | Carrabin | Yoo | New task |
|------|--------|----------|-----|----------|
| P1 | Overall task performance: RMSE to hidden probability (true_p), boxplot per source | Y | Y | Y |
| P2 | Trial-wise RMSE to human responses: model vs human, boxplots across pids | Y | Y | Y |

P1 uses true_p from carrabin_original.csv, merged into carrabin.pkl as the
true_p column. Responses on [-1,1]; true_p converted as true_p*2-1.
Key result: humans have higher task error than all models, establishing that
human noise is a major source of performance limitation.

### V — Variance
Metrics measuring how noisy participants and models are, including
distributional fit quality.

| Code | Metric | Carrabin | Yoo | New task |
|------|--------|----------|-----|----------|
| V1 | Distributional fit (MLE-based): model captures full response distribution | active | future | Y |
| V2 | Response variability for identical inputs: std(response|obs,qid) per pid; KDE + regplot | Y | N | Y |
| V3 | Test-retest reliability: response variability stable across first/second half of session | Y | N | Y |
| V4 | State vs response noise: T3/T4 patterns distinguish NEF spiking noise from NoisyCounting response noise | Y | partial | Y |

V4 is the key novel contribution: NoisyCounting fitted by RMSE collapses
sigma_c to ~0 (response-noise artefact). MLE fitting recovers larger sigma_c
(~0.03-0.08) and nu (~0.08-0.21), confirming state noise is present. The
T3/T4 temporal patterns expose this distinction.

Retired: RNN-based sigma (cv_rmse from TinyGRU). Replaced by qid-grouped
response std = mean of std(response | pid, obs, qid) per pid.

Validated: inter-trial carryover check confirms qid-grouped std is not
contaminated by preceding trial (all r < 0.04 ns). Safe to use as noise metric.

### T — Temporal
Metrics that look at patterns within the observation sequence.

| Code | Metric | Carrabin | Yoo | New task |
|------|--------|----------|-----|----------|
| T1 | Task performance vs observation: RMSE as function of obs position | Y | Y | Y |
| T2 | Response change vs observation: mean |Δresponse| per obs position | Y | Y | Y |
| T3 | Residual variance growth: std(resid|obs,qid) growing across obs | Y | N | Y |
| T4 | Within-trial residual autocorrelation decay: lag-k correlation of residuals, k=1..n_obs | Y | N | Y |

Residuals for T3/T4: response - mean(response | pid, obs, qid).
No qid grouping possible for yoo (no repeated sequences).

### B — Bias
Metrics investigating suboptimal or structured patterns in human responses.

| Code | Metric | Carrabin | Yoo | New task |
|------|--------|----------|-----|----------|
| B1 | Weight profile across observations: fitted temporal weighting (flat/primacy/recency/U-shaped) | Y | Y | Y |
| B2 | Surprise sensitivity and confirmation bias: update size by surprise magnitude and direction | partial | Y | Y |

### N — Neural
Predictions from NEF neural dynamics. No empirical neural data exists for
these tasks; N panels are testable predictions for future experiments.

| Code | Metric | Carrabin | Yoo | New task |
|------|--------|----------|-----|----------|
| N1 | Decoded PE timecourse: PE signal decoded from NEF activity within obs window | Y | Y | Y |
| N2 | Response and PE variability vs n_neurons: both decrease with n_neurons | Y | future | Y |
| N3 | State persistence from spiking noise: NEF prediction driving T3/T4 patterns | Y | N | Y |
| N4 | (Future) Neural population geometry | future | future | future |

---

## Central cognitive model

Updates follow a power-law decaying learning rate:

    alpha(t) = alpha_0 / t^lambda

High lambda: primacy bias. Low lambda: recency bias. In the NEF, alpha(t) is
an emergent property of the spiking dynamics rather than a hardcoded equation.

---

## Active datasets

| Name | N | Task |
|------|---|------|
| carrabin | 21 | Binary inputs; slider after each of 5 obs; sequences repeat (qid); true_p known |
| yoo | 38 | Continuous inputs; slider; 30 obs x 30 trials |

Pickles: data/carrabin.pkl, data/yoo.pkl.
Required columns: pid, trial, observation, value, response.
Carrabin adds: qid, true_p (generating probability, from carrabin_original.csv).

Proposed new task: combines repeated sequences (carrabin) with long sequences
and continuous values (yoo). Unlocks all PVTBN metrics simultaneously.

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

RNN (models/RNN.py): retained for reference; not used in active figures.

NoisyCounting note: RMSE fitting collapses sigma_c to ~0 (response-noise
artefact). MLE fitting (fit_mle.py) correctly recovers sigma_c ~0.03-0.08.
Both fitted versions are scientifically meaningful — RMSE version demonstrates
the limitation of RMSE as a noise metric; MLE version shows what the model
actually predicts when properly calibrated.

---

## NoisyCounting generative model (build_sim_db.py)

The simulation in _simulate_noisy_counting must match math_models.py exactly:

    r = 0.0; p_hat = 0.0
    for each observation x:
        xi    = N(0, sigma_c)
        r     = r + x * mu + xi
        eps   = N(0, nu)
        p_hat = p_hat + (r - p_hat) * exp(eps)
        p_hat = clip(p_hat, -1, 1)
        response[obs] = p_hat

No carrabin shrinkage transform applied (NoisyCounting is excluded from it).
Seeds are 0..n_sims-1 (not trial-based), so each simulation is independent.

---

## MLE fitting pipeline (fit_mle.py)

### Architecture
Shared simulation database + per-pid Optuna studies sharing one SQLite file.

Loop per process (n_fits iterations per pid):
  1. Scan database -> inject new (params, loss_for_this_pid) into this pid's study
  2. Ask TPE for next params
  3. If params already in database: evaluate loss, report, continue (free reuse)
  4. Simulate n_sims times -> save to shared database (atomic rename, NFS-safe)
  5. Evaluate loss for this pid -> report to study

Other pids pick up new simulations at their next step 1 (cross-pid sharing).
No explicit cross-reporting needed.

### Loss function (compute_sim_db_loss)
For each (sequence, obs_idx) cell, collect all observed responses from this pid.
Fit Gaussian from simulated responses: mu_sim = mean(sim), sigma_sim = std(sim).
Group log-likelihood: sum_i log N(r_i | mu_sim, sigma_sim).
Normalise by total n_obs. Returns negative mean log-likelihood (lower = better).
This naturally penalises both mean mismatch and variance mismatch simultaneously.

### MLE_PARAMS ranges (carrabin NoisyCounting)
mu: 0.05-0.40 step 0.002
sigma_c: 0.001-0.30 step 0.002  (RMSE collapses to ~0; MLE recovers ~0.03-0.08)
nu: 0.001-0.35 step 0.002       (expanded from 0.25; pid1 nu~0.21)

### Run on cluster
    bash jobs/submit_mle_fit.sh NoisyCounting carrabin 500 100
    # -> 21 jobs, ~8 min wall time, ~10500 trials per study after cross-sharing

### Check results
    venv/bin/python -c "
    import optuna
    for pid in range(1,22):
        s = optuna.load_study(
            study_name=f'NoisyCounting_carrabin_pid{pid}',
            storage='sqlite:///data/optuna/NoisyCounting_carrabin.db')
        b = s.best_trial
        print(f'pid={pid}: loss={b.value:.4f} params={b.params}')
    "

---

## Response variability metric (primary noise measure)

    qid_response_std(pid) = mean over (obs, qid) groups of std(response | pid, obs, qid)

Requires >= 3 trials per group. Implemented in figure_carrabin.py via
_qid_response_std(). Used in V2, V3, N2 panels.

Validated: inter-trial carryover is negligible (r < 0.04, ns for obs=1
residuals vs preceding trial features). Safe to interpret as within-condition
noise rather than carryover artefact (suitable for footnote/SI).

Test-retest reliability (V3): r = 0.88 (****) across session halves,
confirming noise is a stable individual trait.

---

## Repository structure

```
evidence_integration/
  data/
    carrabin.pkl              # includes true_p column from carrabin_original.csv
    carrabin_original.csv     # raw data with probability column
    yoo.pkl
    counting_activities_n{n}_nc{nc}_{dataset}.pkl
    runs/
      carrabin/
      yoo/
    sim_db/                   # MLE simulation database (one pkl per params hash)
  papers/
    carrabin_paper.txt
  archive/
  models/
    math_models.py
    NEF.py
    RNN.py
    counting_integrator.py
  fitting/
    losses.py          # RMSE + compute_sim_db_loss (group-level MLE)
    fit.py             # Optuna k-fold CV, RMSE
    fit_mle.py         # MLE fitting via shared simulation database
    model_params.py    # MODEL_PARAMS + MLE_PARAMS
    submit.py
    collect.py
  utils/
    paths.py
    plot_style.py
    slurm.py
    carrabin_transform.py
    save_responses.py
  scripts/
    figure_carrabin.py              # 2x4 main carrabin figure (A-H)
    figure_carrabin_performance.py  # P group: schematic + P1 + P2
    figure_carrabin_variability.py  # V group: V2 KDE + regplot + V3 test-retest
    figure_yoo.py                   # 2x4 main yoo figure
    extras_carrabin.py
    extras_yoo.py
    build_sim_db.py    # simulate (model, params) x 32 sequences x n_sims
    dynamics_NEF.py
    check_jobs.py
  jobs/
    submit_pe_readout.sh
    submit_n_neurons_scan.sh
    submit_probe_pids.sh
    submit_yoo_noise.sh
    submit_mle_fit.sh  # one job per pid, shared sim_db + optuna SQLite
  venv/
```

All new scripts go in scripts/. Never create scripts at the project root.
Figures save PDF only (no PNG/SVG).

---

## Current figure panel inventory

### figure_carrabin.py (2x4, panels A-H)

| Panel | Category | Content | Status |
|-------|----------|---------|--------|
| A | - | Task schematic | done |
| B | P2 | RMSE boxplots (models vs human) | done |
| C | V2 | KDE of response variability for identical inputs | done |
| D | P2/V2 | Model RMSE vs human response variability regplot | done |
| E | N2/N3 | NEF resp and PE variability vs n_neurons | done |
| F | T4 | Within-trial residual autocorrelation decay | done |
| G | T3 | Residual variance growth across obs | done |
| H | - | pending | - |

### figure_carrabin_performance.py (1x3)

| Panel | Category | Content | Status |
|-------|----------|---------|--------|
| A | - | Task schematic | done |
| B | P1 | Estimation error (RMSE to hidden probability), human + models | done |
| C | P2 | Model fit (RMSE to human responses) | done |

### figure_carrabin_variability.py (1x3)

| Panel | Category | Content | Status |
|-------|----------|---------|--------|
| A | V2 | KDE of response variability, per-pid lines for human | done |
| B | V2 | Model fit vs human response variability regplot | done |
| C | V3 | Test-retest reliability scatter (first vs second half) | done |

### figure_yoo.py (2x4, panels A-H)

| Panel | Category | Content | Status |
|-------|----------|---------|--------|
| A | - | Task schematic | done |
| B | P2 | RMSE boxplots | done |
| C | T2 | Response change vs observation | done |
| D | B1 | Decay rate error boxplots (primacy/recency weight profile) | done |
| E-F | - | pending | - |
| G | T1 | Task error curves by obs | done |
| H | - | pending | - |

---

## Environment

Always use: /home/psipeter/evidence_integration/venv/bin/python

Cluster home: /dartfs-hpc/rc/home/n/f007qzn/
All SLURM scripts: use pwd -P and export EVIDENCE_INTEGRATION_ROOT=${ROOT}.
NFS mount uses local_lock=none (safe for fcntl/SQLite). Atomic rename used
for simulation database writes (NFS-safe, no stale locks if job dies).

---

## NEF implementation

### Architecture
- value ensemble: running estimate (n_neurons=100)
- error ensemble: prediction-error-driven updates (n_neurons=100)
- counting subnetwork: decodes count and weight W=alpha_0/t^lambda
  (n_neurons_counting=100, radius_c=5 carrabin / 30 yoo)

### Seed
params["seed"] = int(trial). Trial-to-trial tuning curve variability is the
primary spiking noise source.

### Fast counting decoder
Activity files: data/counting_activities_n{n}_nc{nc}_{dataset}.pkl
(compact format: Mty_basis + mem_readout, ~25-100 MB).

### Key findings
- qid_response_std: human ~0.10, NEF ~0.07
- NEF response variability scales with n_neurons; converges to human range at n~100-200
- Within-trial residual autocorrelation: human r~0.62, NEF r~0.78 at lag=1
- NoisyCounting (RMSE-fitted): near-zero autocorrelation, flat variance growth

---

## Carrabin response transform

All carrabin models EXCEPT NoisyCounting apply: response = raw * t/(t+2)
Implemented in utils/carrabin_transform.py. Never apply it twice.

---

## Fitting pipelines

### RMSE fitting (existing)
    python -m fitting.submit carrabin NEF --n_trials 100 --run_folder carrabin
    python -m fitting.collect carrabin --type params
    python scripts/figure_carrabin.py --run_folder carrabin --extra_models NoisyCounting RNN

### MLE fitting (new, NoisyCounting first)
    bash jobs/submit_mle_fit.sh NoisyCounting carrabin 500 100
    # args: model, dataset, n_fits, n_sims
    # output: data/runs/carrabin/NoisyCounting_carrabin_{pid}_params_mle.pkl

---

## Code conventions

- alpha_0, lambda_ (trailing underscore), gamma, eps_p, eps_r
- Merge order: PARAM_DEFAULTS < _NEF_FIXED < fitted Optuna params
- Read loss with _get_loss(perf_df) — never hardcode cv_loss_mean
- Run folder: always pass short name (e.g. carrabin) — resolve_run_folder prepends RUNS_DIR
- --local runs must print JOB_COMPLETE as the final stdout line
- Python 3.11; pathlib via utils.paths; figures save PDF only
- New figure panels go inside existing figure_*.py scripts
- Do not compute metrics in extras scripts — save raw data, compute in figure scripts

---

## Workflow guidelines

### Before making changes
1. Read the relevant files fully first.
2. Check fitting/model_params.py before touching models or fitting.
3. Propose a plan for structural changes before executing.

### NEF simulations via MCP
Never run NEF simulations via MCP — will time out. Write a script, give command.

### Figure iteration
After any figure change, regenerate and display using pdftoppm to convert PDF
to a temporary PNG for viewing, then delete the PNG.

### Git
Generate a commit message and wait for confirmation. Never push without being asked.

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
- Do not save figures as PNG or SVG — PDF only
