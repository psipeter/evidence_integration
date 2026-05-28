# CLAUDE.md — evidence_integration

This file is the source of truth for Claude when working on this project.
Read it fully before making any changes or suggestions. Prefer this file over
README.md when they conflict (README may contain stale references).

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
models: optimal baselines, simple RL baselines, task-specific hand-tuned
cognitive models, and (for the new task) a high-fidelity black-box model
(RNN or finetuned LLM). The expected ordering on trial-wise RMSE is:

    RNN/LLM (best) > task-specific model ≈ NEF > simple RL ≥ optimal (worst)

The NEF is not expected to beat task-specific models on trial-wise RMSE
because: (a) it has intrinsic spiking noise that inflates RMSE; (b) its
mechanisms are task-agnostic; (c) it omits task-specific biases included in
other models. Comparable (not superior) RMSE relative to cognitive models is
sufficient, as long as NEF is consistently better than simple baselines.

### Goal 2 — Cross-task generalisability
The same NEF architecture is applied across multiple tasks. If it achieves
reasonable performance across tasks where task-specific models cannot
generalise (because they were hand-tuned), this establishes that the NEF
implements a **general cognitive strategy** rather than a task-specific fit.

Current tasks: carrabin (binary inputs), yoo (continuous inputs).
Planned: new task with both conditions within-subject.

### Goal 3 — Breadth of behavioural and neural predictions
Trial-wise RMSE is one metric among many. The NEF must also capture
**secondary behavioural signatures** that it was not trained to reproduce:

- Power-law decay of update magnitude: α(t) = α₀/t^λ
- Response noise (trial-to-trial variability for identical sequences)
- Individual differences in λ and α₀, correlated across tasks
- Model-residual noise profile as a function of observation index

For neural predictions, the NEF produces:
- Ensemble-level activity traces (value, error, counting subnetwork)
- Spiking noise magnitude as a function of n_neurons and α₀
- Predictions about which brain areas correspond to which ensembles

These secondary predictions are important because they are **not designed in**
— they emerge from the spiking dynamics. Showing they match human data
strengthens the mechanistic argument considerably.

### Goal 4 — Novel testable predictions
The NEF should generate predictions that can be tested in future experiments:
- Response noise scales with learning rate (α₀)
- ITI noise injection perturbs updating in specific ways
- Neural activity profiles in fMRI/EEG match ensemble dynamics
- λ trait correlates across binary vs. continuous task conditions

---

## Central cognitive model: power-law learning rate

Updates follow a **power-law decaying learning rate**:

    α(t) = α₀ / t^λ

where `t` is observation index within a trial. High λ → primacy bias (early
observations dominate); low λ → recency bias (recent observations dominate).
Participants vary along this spectrum. The NEF implements this via a counting
subnetwork that modulates the error ensemble, so α(t) is an *emergent property*
of the spiking dynamics rather than a hardcoded equation.

---

## Active datasets

| Name | Reference | N | Task |
|------|-----------|---|------|
| **carrabin** | Prat-Carrabin & Woodford (2024) | 21 | Binary inputs; slider after each obs; sequences repeat (qid) enabling response noise measurement |
| **yoo** | Yoo et al. (2025) | 38 | Continuous inputs; slider; 30 obs × 30 trials; power-law decay of update magnitude |
| **new task** | (planned) | 60–80 | Binary + Gaussian conditions; within-subject; designed for λ reliability, cross-task correlation, and RNN/LLM benchmarking |

**Archived** (do not reactivate without explicit plan): diederen, jiang, usher.
Behavioral pickles: `data/carrabin.pkl`, `data/yoo.pkl`.
Required columns: `pid`, `trial`, `observation`, `value`, `response`.
Carrabin additionally uses `qid`.

---

## Active models

| Dataset | Model | Role | Free params |
|---------|-------|------|-------------|
| carrabin | Bayes | Optimal Bayesian | — |
| carrabin | NoisyCounting | Task-specific (Prat-Carrabin) | `mu`, `sigma_c`, `nu` |
| carrabin | RL | Simple baseline | `alpha` |
| carrabin | RL_lambda | Power-law baseline | `alpha_0`, `lambda_` |
| carrabin | NEF_recurrent | Spiking NEF integrator | `alpha_0`, `lambda_` |
| carrabin | NEF_synaptic | Spiking NEF (PES variant) | `alpha_0`, `lambda_` |
| yoo | Mean | Optimal running mean | — |
| yoo | RL | Simple baseline | `alpha` |
| yoo | RL_lambda | Power-law baseline | `alpha_0`, `lambda_` |
| yoo | ADM | Task-specific (Yoo et al.) | `phi`, `rho` |
| yoo | NEF_recurrent | Spiking NEF integrator | `alpha_0`, `lambda_` |
| yoo | NEF_synaptic | Spiking NEF (PES variant) | `alpha_0`, `lambda_` |
| new task | all above + RNN/LLM | Full benchmark spectrum | TBD |

---

## Planned new task

**Platform:** jsPsych + MindProbe (free JATOS hosting) + Prolific
**Design:** 2 conditions (binary Bernoulli, continuous Gaussian), within-subject,
  counterbalanced order, ~30 trials × 15 observations × 4s ≈ 30 min per condition.
**Primary goal:** establish cross-task λ correlation (binary vs continuous),
  showing λ is a stable cognitive trait across input modalities.
**Secondary goals:** response noise estimation (binary condition supports
  natural prefix collisions); RNN/LLM upper-bound benchmarking at 60–80
  participants × 30 trials.
**λ reliability:** bootstrap analysis on yoo confirms 25–30 trials × 15 obs
  achieves Spearman r ≈ 0.94 vs ground-truth λ. 15 obs captures full visible
  decay without short-window upward bias (verified in scripts/trial_obs_reliability_figure.py).

---

## Repository structure

```
evidence_integration/
  data/
    carrabin.pkl
    yoo.pkl
    runs/
      refit/          # primary run folder for all final fits
      nef200/         # NEF fits (200 Optuna trials); copy to refit after collecting
  archive/            # archived code — do not import from here
  models/
    math_models.py
    NEF.py
    counting_integrator.py
    counting_lmu.py
  fitting/
    losses.py
    fit.py
    model_params.py
    submit.py
    collect.py
  utils/
    paths.py
    plot_style.py
    slurm.py
    run_params.py
    save_responses.py
    save_activities.py
    plot_spikes.py
  scripts/            # ALL analysis and figure scripts go here
    figure_carrabin.py
    figure_yoo.py
    extras_carrabin.py
    extras_yoo.py
    dynamics_NEF.py
    iti_perturbation.py
    check_jobs.py
    counting_accuracy.py
    noise_reliability.py
    noise_metric_comparison.py
    trial_obs_reliability_figure.py
  jobs/
  venv/               # always use this Python
```

All new scripts go in `scripts/`. Never create one-off scripts at the project root.

---

## Environment

Always use the project venv:

```bash
/home/psipeter/evidence_integration/venv/bin/python
```

Fall back to PY311 conda env only if venv unavailable. Never use base conda Python.

---

## Workflow

Claude can read files, write files, and run code directly via MCP tools.

### Before making changes
- Read the relevant file(s) first.
- Check `fitting/model_params.py` before touching models or fitting.
- Check `fitting/losses.py` before touching the objective or dataset routing.
- Propose a plan for structural changes before executing.

---

## Code conventions

### Parameter naming
- Power-law: `alpha_0`, `lambda_` (trailing underscore — `lambda` is a keyword)
- Plain RL: `alpha`
- NoisyCounting: `mu`, `sigma_c`, `nu`
- ADM: `phi`, `rho`
- Fixed NEF architecture: in `_NEF_FIXED` in `fitting/model_params.py`

### Merge order for simulation dicts
`PARAM_DEFAULTS` < `_NEF_FIXED` < fitted Optuna params

### Seeds
Use `trial_seed(base_seed, trial_number)` from `utils.run_params` everywhere
a per-trial RNG seed is needed. Do not redefine locally.

### Unified model API
```python
run(params: dict, save: bool = False, trials: list | None = None) -> pd.DataFrame
```
Output columns: `model_type`, `pid`, `trial`, `observation`, `response`.

### Performance pickles
Read loss via `_get_loss(perf_df)` in figure scripts. Never hardcode `cv_loss_mean`.

### Run folder conventions
Pass short names (e.g. `refit`, `nef200`). `utils.paths.resolve_run_folder`
resolves them under `RUNS_DIR`.

### Local completion marker
`--local` runs must print `JOB_COMPLETE` as the final stdout line.

### Style
- Python 3.11; format with `black`; use `pathlib` via `utils.paths`
- Figures: save PNG (300 dpi) + PDF to `figures/`; no `plt.show()` in batch scripts
- New figure panels go inside existing `figure_*.py` scripts

---

## Fitting pipeline

```
submit → fit.py (Optuna k-fold CV, RMSE) → collect → figures
```

Key commands:
```bash
python -m fitting.submit carrabin RL_lambda --n_trials 500 --run_folder refit
python -m fitting.submit carrabin RL_lambda --n_trials 500 --run_folder refit --pid 1 --local
python -m fitting.fit carrabin RL_lambda 1 500 5 refit 42
python -m fitting.collect refit --type params
python -m fitting.collect refit --type responses
python -m fitting.submit --resubmit params --run_folder refit
python -m fitting.submit --resubmit activities --run_folder refit --ensembles error --timing once_per_obs
python -m fitting.collect refit --type activities --ensembles error
python scripts/figure_carrabin.py --run_folder refit
python scripts/figure_yoo.py --run_folder refit --noise_folder yoo_response_noise
```

---

## NEF implementation notes

- Build: `build_network(obs_values, params, decoders)` after `_pretrain(params)`
- `counting` in `_NEF_FIXED`: `"integrator"` (default) or `"lmu"`
- `base_seed` stabilises counting pretrain; `trial_seed` separates per-trial variability
- Default: `n_neurons=200`, `n_neurons_counting=2000`, `lmu_n_obs_max=30`
- `models/NEF.py` imports only from `fitting.model_params`. Keep this dependency shallow.

---

## What NOT to do

- Do not add diederen, jiang, or usher back to active modules without explicit plan
- Do not add `loss_type`, `shape_loss`, `joint_loss`, `beta`, `alpha_bias_array`, or `rd` hooks
- Do not redefine `_trial_seed` locally
- Do not read `cv_loss_mean` directly — use `_get_loss`
- Do not create scripts outside `scripts/`
- Do not commit secrets or edit `archive/` unless explicitly asked
- Do not push to git without being asked
