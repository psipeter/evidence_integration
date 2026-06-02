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
   that the NEF captures a *general* cognitive strategy rather than a task-specific fit.

3. **Breadth of predictions.** Beyond RMSE, the NEF must capture secondary
   behavioural signatures it was not trained to reproduce — power-law update
   decay, response noise, individual differences in λ — and generate neural
   predictions (ensemble activity, spiking noise) that can be compared to data.

4. **Novel testable predictions.** The NEF generates predictions about how
   response noise scales with learning rate and network size, how neural
   populations correspond to anatomical areas, and how λ correlates across
   task conditions — all testable in future experiments.

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
| **carrabin** | Prat-Carrabin & Woodford (2024) | 21 | Slider after each of 5 obs; sequences repeat (qid) | Per-qid response variability (noise) |
| **yoo** | Yoo et al. (2025) | 38 | Slider after each obs (30 obs × 30 trials) | Power-law decay of update magnitude |

Behavioral pickles: `data/carrabin.pkl`, `data/yoo.pkl`.
Columns: `pid`, `trial`, `observation`, `value`, `response`; carrabin adds `qid`.

**Archived** (diederen, jiang, usher): see `archive/`.

---

## Models

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

**RNN** (noise estimation only, not a cognitive model): a TinyGRU fitted
per-participant per-source to estimate response noise sigma. See
`models/RNN.py` and the RNN noise analysis section below.

### NEF architecture

A recurrent spiking network implements:

- **value** ensemble: running estimate of the hidden probability
- **error** ensemble: prediction-error-driven updates, modulated by α(t)
- **counting** subnetwork: integrator that decodes observation count and
  learning-rate weight W = α₀/t^λ, used to scale error→value updates

Key architectural parameters (`_NEF_FIXED` in `fitting/model_params.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_neurons` | 100 | Neurons in value and error ensembles |
| `n_neurons_counting` | 100 | Neurons in counting memory ensemble |
| `radius_c` | 5 (carrabin) / 30 (yoo) | Counting ensemble representational radius |
| `dt` | 0.001 s | Simulation timestep |
| `t_obs` | 1.5 s | Observation window duration |
| `t_iti` | 0.5 s | Inter-trial interval |

**Seed**: `params["seed"] = int(trial)` set directly before each network
build. Trial-to-trial variability in tuning curves is the primary source of
spiking noise.

**Fast counting decoder**: counting network activities (Gram matrices) are
precomputed once and saved to `data/counting_activities_n{n}_nc{nc}.pkl`.
Per Optuna trial, W_weight is recomputed analytically from these matrices for
the current (α₀, λ) — ~300× faster than re-running the Nengo simulation.
Falls back to full pretrain if activity file not found.

Generate activity files:
```bash
venv/bin/python models/counting_integrator.py \
    --precompute_activities --n_trials 200 \
    --n_neurons 100 --n_neurons_counting 100 \
    --dataset carrabin   # sets radius_c=5
```

---

## Repository structure

```
evidence_integration/
  data/
    carrabin.pkl
    yoo.pkl
    counting_activities_n{n}_nc{nc}.pkl   # precomputed counting Gram matrices
    runs/                    # fit outputs (gitignored)
      carrabin/              # primary run folder for carrabin fits + extras
      yoo/                   # primary run folder for yoo fits
  archive/                   # do not import from here
  models/
    math_models.py           # mathematical models (carrabin, yoo)
    NEF.py                   # NEF spiking model
    RNN.py                   # TinyGRU noise estimator
    counting_integrator.py   # counting subnetwork + precompute_activities + fast_decode
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
    carrabin_transform.py    # Laplace shrinkage transform for carrabin responses
    save_responses.py
  scripts/
    figure_carrabin.py       # 2×4 main carrabin figure (panels A–H)
    figure_yoo.py            # 2×4 main yoo figure
    extras_carrabin.py       # supplementary NEF data for figure_carrabin bottom panels
    extras_yoo.py            # supplementary yoo simulations
    dynamics_NEF.py          # single-trial NEF dynamics figure
    iti_perturbation.py      # ITI noise injection experiments
    check_jobs.py            # SLURM job status helper
    counting_accuracy_scan.py  # counting circuit accuracy diagnostics
    nef_noise_experiment.py    # reference: per-trial pretrain noise experiment
    nef_sigma_test.py          # reference: sigma test for new architecture
    update_correlation.py
    noise_reliability.py
    noise_metric_comparison.py
  jobs/
    submit_probe_pids.sh
    submit_pe_readout.sh        # one SLURM job per pid for pe_readout experiment
    submit_n_neurons_scan.sh    # one SLURM job per pid for n_neurons_scan
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
   Optuna trial using fast_decode; CV is evaluated on cached responses.

3. **`fitting.collect`** concatenates per-participant pickles into run-level
   aggregates.

4. **Figure scripts** read from `data/runs/<run_folder>/`.

### Run folder conventions

- **`carrabin/`**: primary folder for all carrabin fits, RNN noise files, and
  extras (pe_readout, probe_timeseries, n_neurons_scan).
- **`yoo/`**: primary folder for all yoo fits.

### Commands (cluster)

```bash
# Fit math models
python -m fitting.submit carrabin Mean          --n_trials 200 --run_folder carrabin
python -m fitting.submit carrabin LeakyIntegrator --n_trials 500 --run_folder carrabin
python -m fitting.submit carrabin PrimacyRecency  --n_trials 500 --run_folder carrabin
python -m fitting.submit carrabin NoisyCounting   --n_trials 500 --run_folder carrabin
python -m fitting.submit carrabin RL_lambda       --n_trials 500 --run_folder carrabin

# Fit NEF
python -m fitting.submit carrabin NEF --n_trials 100 --run_folder carrabin
```

### Local single-participant

```bash
python -m fitting.submit carrabin NEF --n_trials 1 --run_folder test --pid 1 --local
```

### Direct fit.py entrypoint

```bash
python -m fitting.fit <dataset> <model_type> <pid> <n_trials> <k> <run_folder> [seed]
python -m fitting.fit carrabin RL_lambda 1 500 5 carrabin 42
```

### Collect

```bash
python -m fitting.collect carrabin --type params
python -m fitting.collect carrabin --type responses
```

### Carrabin response transform

All carrabin models (except NoisyCounting) apply a Laplace shrinkage transform:

    response = response_raw × t / (t + 2)

This converts the raw weighted mean into a task-appropriate probability
estimate. Applied in `utils/carrabin_transform.py`, called from `NEF.run()`,
`math_models.run()`, and `utils/save_responses.py`.

---

## RNN noise analysis

`models/RNN.py` implements a TinyGRU (n_hidden=4) fitted per-participant to
estimate per-participant response noise sigma, defined as the standard deviation
of residuals between a model's responses and the RNN's prediction of those
responses. This is fitted separately for each source (human data, and each
cognitive model's responses).

**Output files** in `data/runs/carrabin/`:

| File | Contents |
|------|----------|
| `RNN_carrabin_performance.pkl` | Human-only RNN fit quality (21 pids) |
| `RNN_carrabin_responses.pkl` | Human-only RNN predicted responses |
| `RNN_sigma_carrabin_sigma.pkl` | Per-pid sigma for all sources (147 rows) |
| `RNN_sigma_carrabin_params.pkl` | Per-pid RNN params for all sources |
| `RNN_sigma_{source}_carrabin_{pid}.pkl` | Per-(source, pid) intermediate files |

**Sources**: `human`, `Mean`, `LeakyIntegrator`, `PrimacyRecency`, `RL_lambda`,
`NoisyCounting`, `NEF`.

**Key finding**: NEF sigma ≈ 0.057 (mean across pids), human sigma ≈ 0.118 —
the NEF accounts for ~60% of human response variability. Deterministic models
show sigma ≈ 0.005 (near-zero).

### Commands

```bash
# Fit RNN for one source (all pids)
venv/bin/python models/RNN.py --source NEF --all_pids --run_folder carrabin

# Collect all sources into combined files
venv/bin/python models/RNN.py --collect --run_folder carrabin
```

---

## Carrabin extras (figure bottom panels)

`scripts/extras_carrabin.py` generates supplementary NEF data for
`figure_carrabin.py` bottom panels (E–H). Each experiment is run with
`--experiment <name>`.

### pe_readout (panels F, G)

Runs the standard NEF simulation for one pid and saves the decoded prediction
error at the readout moment for each (trial, observation).

Output: `pe_readout_NEF_carrabin_{pid}.pkl` — columns: `pid`, `trial`,
`observation`, `qid`, `pe_at_readout`.

```bash
# Run per pid (or via SLURM: bash jobs/submit_pe_readout.sh)
venv/bin/python scripts/extras_carrabin.py \
    --experiment pe_readout --pid 1 \
    --run_folder carrabin --out_folder carrabin

# Collect after all pids complete
venv/bin/python scripts/extras_carrabin.py \
    --experiment pe_readout --mode collect --out_folder carrabin
```

### probe_timeseries (panel E)

Runs the full NEF simulation for one pid with once-per-dt probe saving.
Output: `probe_timeseries_NEF_carrabin_{pid}.pkl` — columns: `pid`, `trial`,
`observation`, `qid`, `t_within_obs`, `decoded_pe`, `decoded_value`.

```bash
venv/bin/python scripts/extras_carrabin.py \
    --experiment probe_timeseries --pid 6 \
    --run_folder carrabin --out_folder carrabin
```

### n_neurons_scan (panel H)

Scans n_neurons ∈ [25, 50, 100, 200, 400] with n_neurons_counting set to the
same value. For each value: precomputes counting activities if needed,
simulates 200 trials, computes std(PE at readout), fits an RNN for sigma.
Output: `n_neurons_scan_{pid}.pkl` per pid; collected into `n_neurons_scan.pkl`.

```bash
# Run per pid (or via SLURM: bash jobs/submit_n_neurons_scan.sh)
venv/bin/python scripts/extras_carrabin.py \
    --experiment n_neurons_scan --scan_pid 1 \
    --run_folder carrabin --out_folder carrabin

# Collect after all pids complete
venv/bin/python scripts/extras_carrabin.py \
    --experiment n_neurons_scan --mode collect --out_folder carrabin
```

---

## Figures

### figure_carrabin.py

2×4 layout. Run folder: `carrabin`. Default model order:
`Mean`, `LeakyIntegrator`, `PrimacyRecency`, `NEF`.
Extra models (e.g. `RNN`, `NoisyCounting`) can be added with `--extra_models`.

```bash
python scripts/figure_carrabin.py --run_folder carrabin
python scripts/figure_carrabin.py --run_folder carrabin --extra_models RNN NoisyCounting
```

| Panel | Content | Data source |
|-------|---------|-------------|
| A | Task schematic | `figures/carrabin_task.pdf` |
| B | RMSE boxplots per model, with significance bars (NEF vs Mean/LI/PR) | `{model}_carrabin_performance.pkl` |
| C | Response noise schematic | `figures/response_noise_schematic.pdf` |
| D | Normalised KDE of sigma per source (human, NEF, NoisyCounting; deterministic as vertical lines) | `RNN_sigma_carrabin_sigma.pkl` |
| E | Decoded PE timecourse for obs=1, qid starting with "1" (pids 6 and 7) | `probe_timeseries_NEF_carrabin_{pid}.pkl` |
| F | Scatter/regression: fitted α₀ vs sigma_NEF across pids | `RNN_sigma_carrabin_sigma.pkl`, `NEF_carrabin_params.pkl` |
| G | Scatter/regression: std(PE at readout) vs sigma_NEF | `pe_readout_NEF_carrabin.pkl` |
| H | sigma_NEF and std(PE) vs n_neurons, with individual human sigma reference lines | `n_neurons_scan.pkl`, `RNN_sigma_carrabin_sigma.pkl` |

### figure_yoo.py

```bash
python scripts/figure_yoo.py --run_folder yoo
```

---

## Key analysis findings

**Response noise and learning rate (carrabin)**:
- alpha_0 and lambda_ are highly collinear in NEF fits (r=0.93)
- alpha_0 explains R²=0.80 of sigma_NEF variance; lambda_ adds nothing
  independent (partial r=-0.02 after controlling for alpha_0)
- Higher alpha_0 → lower sigma_NEF (high learning rate → rapid convergence →
  predictable responses → low RNN residual)
- std(PE at readout) is the mechanistic mediator: r=0.94 with sigma_NEF

**Neural noise scaling**:
- sigma_NEF scales approximately as 1/√n_neurons
- At n=100 (default): sigma_NEF ≈ 0.057, human sigma ≈ 0.118 (ratio ≈ 0.60)
- Network would match median human sigma at approximately n≈50 neurons

---

## Environment

```bash
# Always use project venv
/home/psipeter/evidence_integration/venv/bin/python
```

Dependencies: numpy, pandas, matplotlib, seaborn, optuna, nengo, scipy, torch.

---

## Archive

Older models and data for diederen, jiang, and usher live under `archive/`.
See `archive/archive_readme.md`. Do not rely on those paths for active analyses.
