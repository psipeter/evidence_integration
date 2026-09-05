# docs/SCIENCE.md — scientific goals, current thread, and results

This is the scientific record: what this project is trying to show, where
that argument currently stands, and what the figures have found so far.
For code/architecture conventions, see CLAUDE.md. For the reasoning
behind past methodology choices, see docs/DECISIONS.md.

---

## Scientific goals

This project studies **how people integrate sequential noisy evidence**,
using cognitive models and a biophysical spiking neural network (NEF) to
identify the computational and neural mechanisms underlying that process.

**Central model:** updates follow a power-law decaying learning rate,
`alpha(t) = alpha_0 / t^lambda`. High lambda = steep discounting
(primacy-like); low lambda = slow discounting (recency-like). In the NEF,
`alpha(t)` is an emergent property of spiking dynamics rather than a
hardcoded equation — a counting subnetwork tracks the observation index
and decodes the appropriate weight, gating the error signal that drives
the value ensemble. `RL_lambda` implements the same equation explicitly —
it is the mathematical theory the NEF realises biophysically, not a point
of direct comparison.

### Goal 1 — Cross-task generalisation of cognitive mechanisms
The NEF must capture human behaviour across multiple tasks (carrabin and
yoo) without task-specific modification, benchmarked against an optimal
Bayesian integrator (Mean), a leaky integrator (LeakyIntegrator), and
primacy/recency weighting models (PrimacyRecency). Expected RMSE ordering:

    task-specific model ≈ NEF > LeakyIntegrator ≥ PrimacyRecency ≥ Mean (optimal)

The NEF need not outperform task-specific models; comparable RMSE combined
with cross-task generalisability is the target.

### Goal 2 — Emergent higher-order behavioural signatures
Beyond RMSE, the NEF must reproduce secondary behavioural phenomena it was
not explicitly trained to capture: temporal update patterns, decay of
response change across the sequence, individual differences in
discounting rate (λ), test-retest reliability of noise/decay-rate
metrics, and state-persistent response variability. That these emerge
without being directly optimised is the key contribution.

### Goal 3 — Joint behavioural and neural predictions
The NEF generates behavioural and neural predictions simultaneously from
the same mechanism. Behavioural: response trajectories, update
magnitudes, individual λ and α₀. Neural: error-ensemble activity,
prediction-error dynamics, and how both scale with architectural
parameters (n_neurons, α₀, λ). Together these form a mechanistically
coherent account testable at multiple levels of analysis.

### Goal 4 — Novel testable predictions
Spiking noise produces state-persistent variability that differs
qualitatively from response noise — distinguishing the NEF from
NoisyCounting even at similar RMSE. Response and PE variability scale
with n_neurons and α₀. These are quantitative predictions for future
empirical work.

### Response noise mechanism

The only noise mechanism in active use is i.i.d. response noise, wrapping
any deterministic base model (Mean, LeakyIntegrator, PrimacyRecency,
RL_lambda):

```
<model>_resp_noise (models.math_models.add_noise):
  mu = run(base_params).response          -- ONE deterministic call
  response = clip(mu + eta, -1, 1)         eta ~ N(0, sigma_resp), i.i.d. per row
```

A compounding-state-noise alternative (`NoisyRL_lambda`) was tried and
compared against this at equal parameter count, then retired from active
analysis -- see `docs/DECISIONS.md` for the comparison and why.

---

## Current thread

**Active:** `neural_main` (`scripts/make_paper_figures.py`'s
`make_neural_main()`) — the sole, authoritative figure for the causal
impact of neural parameters (α₀, λ, n_neurons) on behaviour and activity.
Isolates each parameter's own contribution one row at a time, rather than
reading off a correlation across randomly-covarying parameters (its
predecessor, `neural_giant`, did the latter — retired; see
docs/DECISIONS.md).

- **Row 1 (α₀) — `oddball` experiment.** Built, stable.
- **Row 2 (λ) — `param_scan` on real/synthetic trials.** Built, stable.
- **Row 3 (n_neurons) — a different underlying experiment, settled after
  extensive exploration** (a convergence hypothesis tested and not
  supported cleanly; a Fano-factor purely-neural SNR measure tried and
  abandoned; split-half population reliability tried and kept). Column 1
  (toy trace demo) is built and iterated to final form; the remaining
  columns are the current frontier.

**Recently settled, feeding into the current thread:** NEF network sizes
bumped to `n_neurons=500` across all datasets (see docs/DECISIONS.md); a
shared-seed bug in `add_noise` fixed (independent response noise per
pid/model, not one shared draw); `model_performance`/`lambda`/`sigma`
giants retired alongside `neural_giant` in favour of the row-based
`neural_main` design.

**Retired from active analysis (this session):** state-noise models
(`NoisyRL_lambda`) and the task-specific `NoisyCounting` model, along
with the MLE fitting pipeline built for it, the RNN conditional-mean
estimator, and NEF's own NLL/multi-seed-ensemble branch (too expensive to
run at the scale this project needs). None of this affects the RMSE-fit
models the figures above are built on (Mean, LeakyIntegrator,
PrimacyRecency, RL_lambda, NEF, and the still-active `_resp_noise`
i.i.d.-noise wrapper). Code archived under `archive/models/`,
`archive/fitting/`; full reasoning in `docs/DECISIONS.md`.

**Not yet started:** the "Future extensions" below (ablation/statistical
validation of `neural_main`'s parameter-vs-outcome relationships; a
synaptic-vs-working-memory implementation comparison). Model fitting
against real `task_backend` (soltani) data — the human-only pilot figures
exist, but NEF/math-model fits to that data haven't been run yet.

---

## NEF architecture

Three interacting neural populations:

1. **Value ensemble** — maintains a running estimate of the current
   evidence mean, decoded after each observation.
2. **Error ensemble** — computes prediction error and gates it by the
   current observation weight α(t). Weight-tuned neurons here are the
   key neural readout; their activity directly tracks α(t).
3. **Counting subnetwork** — tracks observation count and decodes
   α(t) = α₀/t^λ, the same discounting RL_lambda implements explicitly
   but via spiking dynamics. Requires a precomputed activity file
   (`counting_activities_n{n}_nc{nc}_{dataset}.pkl`, generated by
   `counting_integrator.py`).

Trial-to-trial variability in neural tuning curves — keyed by
`counting_integrator.activity_key_for_trial(dataset, trial)` (`int(trial)`
for carrabin/yoo, `trial+1` for 0-indexed soltani) — is the primary
spiking noise source, producing state-persistent response variability
across observations within a trial (the mechanism behind Goal 4's novel
testable predictions). That value keys the activity file; the two must
never diverge (see CLAUDE.md's session-start checklist).

Activity files load at fit time for speed (`fast_decode` mode). Current
sizes: `n_neurons=500` for all four datasets; `n_neurons_counting=2000`
for yoo/soltani, `500` for carrabin (why: docs/DECISIONS.md's NEF-sizing
entry).

NEF fits under RMSE only now — the NLL/multi-seed-ensemble branch was
retired (too expensive to run at scale; see docs/DECISIONS.md and
"Current thread" above).

---

## Metric taxonomy

All analyses and figure panels are organised around the central model's
own free parameters — `alpha_0`/`lambda_` (the decay-rate construct) and
`sigma_resp` (the noise construct) — bookended by overall model
performance and by the neural mechanism that realises both constructs
biophysically. Figures save PDF only.

### 1. Model performance (RMSE)
Establishes the models — including NEF — as credible fits to human
behaviour, supporting Goal 1's cross-task generalisation claim:
`task-specific model ≈ NEF > LeakyIntegrator ≥ PrimacyRecency ≥ Mean`.
`model_performance` is the main 4-task comparison (Mean/LeakyIntegrator/
PrimacyRecency/RL_lambda/NEF); `model_best_fit` and
`model_performance_nll` give the same comparison as best-fit-fraction and
NLL views; `temporal_performance` shows the human error trajectory
(RMSE-to-ground-truth vs observation) across all four tasks.

### 2. Lambda exploration
The discounting/recency-bias signature of power-law integration (Goal 2).

**2.1 Response change decay across tasks.** Update magnitude
(`|Δresponse|`) shrinks with observation count, at a task-appropriate
rate — `response_change` (main 4-task figure, human + 5 models),
`lambda_metric` (illustration of the power-law fitting procedure),
`lambda_main` (composite pairing response-change with fitted-λ
distributions).

**2.2 Individual differences for humans.** A single fitted λ per person
locates them on the primacy↔recency spectrum — `lambda_human` (per-task λ
distributions across participants), `lambda_overview` (adds cross-task
reliability underneath).

### 3. Sigma exploration
Goal 4's novel prediction: spiking noise is qualitatively different from
i.i.d. response noise, and NEF — not the deterministic-plus-`_resp_noise`
models — is the one that reproduces it.

**3.1 Individual differences in response variability for humans.** How
noisy is each person's response to a repeated, identical stimulus —
`variability_human` (per-task KDE, human-only) and `variability_models`
(same, with model overlay).

**3.2 Growth of variability during the early sequence.** Response noise
should accumulate over the course of a trial if it's state-persistent
rather than i.i.d. — `sigma_main` row 2 (normalised residual-variance
growth vs observation, human + models + NEF, all three tasks). NEF tracks
the human growth pattern; the `_resp_noise` models do not.

**3.3 Autocorrelation of deviation from average behaviour at time t vs
t+k.** The more direct signature of state persistence: genuine
state-persistent noise produces decaying positive autocorrelation of the
residual; pure i.i.d. response noise looks like scatter around zero at
every lag — `variance_autocorr_human`/`variance_autocorr_models`
(dedicated 4-panel figures) and `sigma_main` row 3 (folded into the
composite). As with growth, NEF reproduces the human autocorrelation
pattern; the `_resp_noise` models don't.

Together, 3.2 and 3.3 are the empirical core of Goal 4: two independent
metrics, both distinguishing NEF's spiking-noise mechanism from ordinary
response noise on the same behavioural data the models were fit to.

**Supplementary:**
1. **Lambda for balls task (no decay)** — `lambda_balls`, kept separate
   since balls doesn't show the expected decay the other three tasks do.
2. **Lambda and sigma reliability within and across tasks** —
   `lambda_reliability`/`sigma_reliability` (odd/even split-half, within
   task), `lambda_sigma_crosstask` (colors-vs-numbers, paired),
   `lambda_sanity_human`/`sigma_sanity_human` (combined reliability +
   cross-task).
3. **Lambda and sigma, human vs model** — `lambda_model_correlation`/
   `lambda_humanvmodel` and `sigma_model_correlation`: how well each
   model's own fitted λ/σ tracks the same participant's.

### 4. Neural predictions
Goal 3's joint behavioural-and-neural account, realised in one figure,
`neural_main`, run on `soltani_numbers` (the one task with both a real
fitted λ and a real fitted σ). Each row isolates one architectural
parameter causally via a controlled sweep, and each row shares the same
internal structure: observe the qualitative phenomenon, show it scales
with the parameter, then predict an individual-difference signature
testable with future spike-resolved recordings.

- **Row 1 — oddball experiment (α₀).** A run of consistent inputs
  followed by an oddball: the error population's decoded PE rises sharply
  then declines as the value estimate updates in real time. Magnitude and
  decline rate both depend on the synaptic learning rate α₀; individuals
  with larger oddball PE responses are predicted to adapt fastest.
- **Row 2 — error activity decline across the block (λ).** Error-sensitive
  neurons' activity decays over the course of a block, producing
  progressively smaller (more conservative) value updates later on. Both
  the activity decay and the shrinking update size depend on the synaptic
  modulation λ; individuals with larger activity attenuation are predicted
  to show the most stable late-block behaviour.
- **Row 3 — oddball SNR / response variability (n_neurons).** The decoded
  value signal drifts over time, producing different responses to
  identical repeated sequences. Both response variability and
  within-trial error-population variability depend on the number of
  simulated neurons; individuals with the most inconsistent post-oddball
  error-population readouts are predicted to show the most response
  variability.

---

## Current figure panel inventory

| Figure | Section | Layout |
|---|---|---|
| `temporal_performance` | 1. Model performance | 1×1 |
| `model_performance` | 1. Model performance | 1×4 |
| `model_best_fit` | 1. Model performance | 2×4 |
| `model_performance_nll` | 1. Model performance | 1×4 |
| `response_change` | 2.1 Response change decay | 1×4 |
| `lambda_metric` | 2.1 Response change decay | 1×1 |
| `lambda_main` | 2.1 + 2.2 composite | 2×3 |
| `lambda_human` | 2.2 Individual differences | 1×4 |
| `lambda_overview` | 2.2 composite | 2×4 |
| `lambda_balls` | Supplementary | 1×1 |
| `lambda_reliability` | Supplementary | 1×3 |
| `sigma_reliability` | Supplementary | 1×3 |
| `lambda_sigma_crosstask` | Supplementary | 1×2 |
| `lambda_sanity_human` | Supplementary | 1×4 |
| `sigma_sanity_human` | Supplementary | 1×4 |
| `lambda_model_correlation` | Supplementary | 1×3 |
| `lambda_humanvmodel` | Supplementary | 1×3 |
| `sigma_model_correlation` | Supplementary | 1×3 |
| `variability_human` | 3.1 Individual differences | 1×4 |
| `variability_models` | 3.1 Individual differences | 1×4 |
| `sigma_overview` | 3.1 composite | 2×4 |
| `sigma_main` | 3.1 + 3.2 + 3.3 composite | 3×3 |
| `variance_autocorr_human` | 3.3 Autocorrelation | 1×4 |
| `variance_autocorr_models` | 3.3 Autocorrelation | 1×4 |
| `neural_main` | 4. Neural predictions | 3×3 |

---

## Future extensions (soft todos, not tied to any current figure's structure)

- **Make row 2 (error activity decline / λ) more compelling** by relating
  it back to behavioural error rates later in the task, not just update
  magnitude.
- **Validation via ablation/statistical control** (not yet built) — a
  perturbation experiment as `neural_main`'s row 4. For each
  parameter-vs-outcome relationship rows 1-3 show: a partial correlation
  controlling for the other parameters, and, where feasible, a mechanistic
  ablation (forcing a parameter to null and showing the correlation
  collapses) — direct causal validation of the current rows.
- **Synaptic vs. working-memory implementation comparison** (not
  started, separate downstream scope) — `neural_main`'s row 5. Different
  predictions under an ITI manipulation depending on which implementation
  of the learning rule is assumed. Deliberately out of scope for now.
