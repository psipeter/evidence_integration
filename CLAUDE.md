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
it was not trained to reproduce: power-law decay of update magnitude, response
noise, individual differences in λ and α₀. For neural predictions, the NEF
produces ensemble-level activity traces and spiking noise magnitude as a
function of n_neurons and α₀.

### Goal 4 — Novel testable predictions
Response noise scales with learning rate (α₀) and network size (n_neurons);
neural activity profiles match ensemble dynamics; λ correlates across task
conditions.

---

## Central cognitive model

Updates follow a **power-law decaying learning rate**:

    α(t) = α₀ / t^λ

High λ → primacy bias; low λ → recency bias. In the NEF, α(t) is an
*emergent property* of the spiking dynamics (counting subnetwork modulates
the error ensemble) rather than a hardcoded equation.

---

## Active datasets

| Name | N | Task |
|------|---|------|
| **carrabin** | 21 | Binary inputs; slider after each of 5 obs; sequences repeat (qid) |
| **yoo** | 38 | Continuous inputs; slider; 30 obs × 30 trials |

Pickles: `data/carrabin.pkl`, `data/yoo.pkl`.
Required columns: `pid`, `trial`, `observation`, `value`, `response`.
Carrabin adds `qid`.

**Archived** (do not reactivate): diederen, jiang, usher.

---

## Active models

| Dataset | Model | Role | Free params |
|---------|-------|------|-------------|
| carrabin | Mean | Optimal running mean | — |
| carrabin | LeakyIntegrator | Leaky integrator baseline | `gamma` |
| carrabin | PrimacyRecency | Temporal weighting function | `eps_p`, `eps_r` |
| carrabin | NoisyCounting | Task-specific (Prat-Carrabin) | `mu`, `sigma_c`, `nu` |
| carrabin | RL_lambda | Power-law delta rule | `alpha_0`, `lambda_` |
| carrabin | NEF | Spiking NEF integrator | `alpha_0`, `lambda_` |
| yoo | Mean | Optimal running mean | — |
| yoo | LeakyIntegrator | Leaky integrator baseline | `gamma` |
| yoo | PrimacyRecency | Temporal weighting function | `eps_p`, `eps_r` |
| yoo | RL_lambda | Power-law delta rule | `alpha_0`, `lambda_` |
| yoo | NEF | Spiking NEF integrator | `alpha_0`, `lambda_` |

**RNN** (`models/RNN.py`): TinyGRU noise estimator — not a cognitive model.
Fitted per-participant per-source to estimate response noise sigma. See
RNN noise analysis section.

---

## Repository structure

```
evidence_integration/
  data/
    carrabin.pkl
    yoo.pkl
    counting_activities_n{n}_nc{nc}.pkl   # precomputed counting Gram matrices
    runs/
      carrabin/      # primary run folder: carrabin fits + RNN noise + extras
      yoo/           # primary run folder: yoo fits
  archive/           # do not import from here
  models/
    math_models.py
    NEF.py
    RNN.py
    counting_integrator.py   # counting subnetwork + precompute_activities + fast_decode
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
    carrabin_transform.py
    save_responses.py
  scripts/           # ALL analysis and figure scripts
    figure_carrabin.py
    figure_yoo.py
    extras_carrabin.py   # NEF probe data for figure_carrabin bottom panels
    extras_yoo.py
    dynamics_NEF.py
    iti_perturbation.py
    check_jobs.py
    counting_accuracy_scan.py
    nef_noise_experiment.py
    nef_sigma_test.py
    update_correlation.py
    noise_reliability.py
    noise_metric_comparison.py
  jobs/
    submit_pe_readout.sh
    submit_n_neurons_scan.sh
    submit_probe_pids.sh
    submit_yoo_noise.sh
  venv/              # always use this Python
```

All new scripts go in `scripts/`. Never create scripts at the project root.

---

## Environment

**Always use:**
```bash
/home/psipeter/evidence_integration/venv/bin/python
```

Fall back to PY311 conda env only if venv unavailable. Never use base Python.

---

## NEF implementation

### Architecture
- **value** ensemble: running estimate (n_neurons=100)
- **error** ensemble: prediction-error-driven updates (n_neurons=100)
- **counting** subnetwork: integrator decoding count and weight W=α₀/t^λ
  (n_neurons_counting=100, radius_c=5 for carrabin / 30 for yoo)

### Seed
`params["seed"] = int(trial)` set directly before each network build.
No base_seed, no pid-level hashing, no trial_seed utility function.
Trial-to-trial variability in tuning curves is the primary noise source.

### Fast counting decoder
Counting network activities (Gram matrices) are precomputed once per
(n_neurons, n_neurons_counting, radius_c) configuration and saved to
`data/counting_activities_n{n}_nc{nc}.pkl`. Per Optuna trial, W_weight
is recomputed analytically via `fast_decode(activity, alpha_0, lambda_)` —
~300× faster than re-running the Nengo simulation. Falls back to full
`_pretrain` if activity file not found.

Generate activity files:
```bash
venv/bin/python models/counting_integrator.py \
    --precompute_activities --n_trials 200 \
    --n_neurons 100 --n_neurons_counting 100 \
    --dataset carrabin
```

### Key findings
- sigma_NEF ≈ 0.057 (mean); human sigma ≈ 0.118 — NEF accounts for ~60%
- sigma scales ~1/√n_neurons; matches median human noise at n≈50
- alpha_0 explains R²=0.80 of sigma_NEF variance; lambda_ adds nothing
  independent (partial r=-0.02 after controlling for alpha_0)
- std(PE at readout) mediates the alpha_0 → sigma relationship (r=0.94)

### Model params source of truth
`fitting/model_params.py`: `_NEF_FIXED`, `_NEF_RANGES`, `MODEL_PARAMS`.
`radius_c` is set per-dataset in the `"fixed"` dict (carrabin=5, yoo=30).

---

## Carrabin response transform

All carrabin models except NoisyCounting apply:

    response = response_raw × t / (t + 2)

Implemented in `utils/carrabin_transform.py`. Applied inside `NEF.run()`,
`math_models.run()`, and `utils/save_responses.py`. Never apply it twice.

---

## RNN noise analysis

`models/RNN.py` — TinyGRU (n_hidden=4) fitted per-participant per-source.
sigma = std(source_response - RNN_prediction) across trials.

Output files in `data/runs/carrabin/`:
- `RNN_carrabin_performance.pkl` — human-only RNN fit (21 pids)
- `RNN_sigma_carrabin_sigma.pkl` — sigma for all sources × pids
- `RNN_sigma_{source}_carrabin_{pid}.pkl` — per-(source, pid) intermediates

Sources: `human`, `Mean`, `LeakyIntegrator`, `PrimacyRecency`, `RL_lambda`,
`NoisyCounting`, `NEF`.

```bash
venv/bin/python models/RNN.py --source NEF --all_pids --run_folder carrabin
venv/bin/python models/RNN.py --collect --run_folder carrabin
```

`collect()` also rebuilds human-only `RNN_carrabin_{performance,params,responses}.pkl`.
The `RNN_sigma_` prefix distinguishes noise-estimation files from human-fit files.

---

## Carrabin extras (figure bottom panels)

`scripts/extras_carrabin.py` — supplementary NEF data for figure_carrabin E–H.

| Experiment | Panel | Output | CLI |
|------------|-------|--------|-----|
| `pe_readout` | F, G | `pe_readout_NEF_carrabin_{pid}.pkl` | `--pid N` or `--mode collect` |
| `probe_timeseries` | E | `probe_timeseries_NEF_carrabin_{pid}.pkl` | `--pid N` (no collect step) |
| `n_neurons_scan` | H | `n_neurons_scan_{pid}.pkl` → `n_neurons_scan.pkl` | `--scan_pid N` or `--mode collect` |

```bash
# pe_readout: one job per pid on cluster
bash jobs/submit_pe_readout.sh
venv/bin/python scripts/extras_carrabin.py --experiment pe_readout \
    --mode collect --out_folder carrabin

# n_neurons_scan: one job per pid on cluster
bash jobs/submit_n_neurons_scan.sh
venv/bin/python scripts/extras_carrabin.py --experiment n_neurons_scan \
    --mode collect --out_folder carrabin

# probe_timeseries: run locally per pid (no collect needed)
venv/bin/python scripts/extras_carrabin.py --experiment probe_timeseries \
    --pid 6 --run_folder carrabin --out_folder carrabin
```

---

## Fitting pipeline

```
submit → fit.py (Optuna k-fold CV, RMSE) → collect → figures
```

```bash
# Fit models
python -m fitting.submit carrabin NEF         --n_trials 100 --run_folder carrabin
python -m fitting.submit carrabin RL_lambda   --n_trials 500 --run_folder carrabin
python -m fitting.submit carrabin NoisyCounting --n_trials 500 --run_folder carrabin

# Local test
python -m fitting.submit carrabin NEF --n_trials 1 --run_folder test --pid 1 --local

# Direct entrypoint
python -m fitting.fit carrabin RL_lambda 1 500 5 carrabin 42

# Collect
python -m fitting.collect carrabin --type params
python -m fitting.collect carrabin --type responses

# Resubmit missing
python -m fitting.submit --resubmit params --run_folder carrabin

# Figures
python scripts/figure_carrabin.py --run_folder carrabin
python scripts/figure_carrabin.py --run_folder carrabin --extra_models RNN NoisyCounting
python scripts/figure_yoo.py --run_folder yoo
```

---

## Code conventions

- `alpha_0`, `lambda_` (trailing underscore), `gamma`, `eps_p`, `eps_r`
- Merge order: `PARAM_DEFAULTS` < `_NEF_FIXED` < fitted Optuna params
- Read loss with `_get_loss(perf_df)` — never hardcode `cv_loss_mean`
- Run folder: always pass short name (e.g. `carrabin`) — `resolve_run_folder`
  prepends `RUNS_DIR`. Passing a full path causes the double-path bug
  (`data/runs/data/runs/carrabin`).
- `--local` runs must print `JOB_COMPLETE` as the final stdout line
- Python 3.11; `pathlib` via `utils.paths`; figures to `figures/` as PNG+PDF
- New figure panels go inside existing `figure_*.py` scripts

---

## Workflow guidelines

### Before making changes
1. Read the relevant file(s) fully first — including checking that patterns
   to be replaced actually exist before attempting substitution.
2. Check `fitting/model_params.py` before touching models or fitting.
3. Propose a plan for structural changes before executing.

### NEF simulations via MCP
**Never run NEF simulations directly through MCP tool calls** — they take
3–8 minutes per pid and will time out. Instead: write a script and give the
run command. Exceptions: very fast operations (e.g. loading a pkl, running
a 1-line venv/bin/python -c analysis) are fine.

### Inspecting data files
Use one-liners rather than creating a separate script:
```bash
venv/bin/python -c "import pandas as pd; df = pd.read_pickle('...'); print(df.head())"
```
Only create a script file if the analysis is multi-step or will be reused.

### SLURM job scripts
Prefer `--wrap` for simple per-pid jobs over generating per-pid `.sh` files:
```bash
sbatch --mem=16G --time=0:30:0 --output=logs/%j.out \
    --wrap="cd ~/evidence_integration && source venv/bin/activate && python ..."
```
Use heredoc-generated per-pid scripts only when the job needs non-trivial setup.

### Figure iteration
After any figure change, always regenerate and display the figure image
(resize if needed for the context window), not just the save confirmation.

### Git
Generate a commit message and wait for confirmation before any git operations.
Never push to git unless explicitly asked.

---

## What NOT to do

- Do not add diederen, jiang, or usher back without explicit plan
- Do not add `loss_type`, `shape_loss`, `joint_loss`, `beta` hooks
- Do not use `trial_seed` / `base_seed` for NEF — seed is set to `int(trial)` directly
- Do not read `cv_loss_mean` directly — use `_get_loss`
- Do not create scripts outside `scripts/`
- Do not add NEF_synaptic — removed; only NEF (recurrent) exists
- Do not add LMU counting variant — removed; only integrator counting exists
- Do not use ADM name — replaced by PrimacyRecency throughout
- Do not double-apply the carrabin transform
- Do not pass a full path as `run_folder` — always use a short name
- Do not commit or push without being asked
- Do not run NEF simulations through MCP tool calls (will time out)
