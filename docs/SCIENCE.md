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
models the P/T/N figures are built on (Mean, LeakyIntegrator,
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

## Metric taxonomy (PTN framework)

All analyses and figure panels are organised under three groups. Figures
save PDF only.

### P — Performance
| Code | Metric | Carrabin | Yoo |
|------|--------|----------|-----|
| P1 | Estimation error: RMSE to hidden probability / true mean, per pid | Y | Y |
| P2 | Model fit: RMSE to human responses, per pid | Y | Y |

### T — Temporal
| Code | Metric | Carrabin | Yoo |
|------|--------|----------|-----|
| T1 | Task performance vs observation | Y | Y |
| T2 | Response change vs observation (mean \|Δresponse\|) | Y | Y |
| T3 | Split-half reliability of λ | N | Y |
| T4 | λ_model vs λ_human (individual differences) | N | Y |
| T5 | Residual variance growth across obs (state noise accumulation) | Y | N |
| T6 | Within-trial residual autocorrelation decay (state persistence) | Y | N |

### N — Neural (NEF predictions; testable in future empirical experiments)
| Code | Metric | Carrabin | Yoo |
|------|--------|----------|-----|
| N1 | Decoded PE timecourse within observation window | Y | Y |
| N2 | PE variability vs response variability (partial-r control for α₀) | Y | N |
| N3 | Response/PE variability vs fitted α₀ | Y | N |
| N4 | Response/PE variability vs n_neurons scan | Y | N |
| N5-N8 | Weight-neuron activity/λ relationships | N | Y |

N1-N8 above is the older per-task taxonomy (`figure_carrabin_neural.py`/
`figure_yoo_neural.py`); the current consolidated argument lives in
`neural_main` (above), run on `soltani_numbers` specifically — the one
task with both a real fitted σ and a real fitted λ.

---

## Scientific narrative per figure group

### P figures — establishing the model as a credible fit
NEF is competitive with or better than Mean/LI/PR on RMSE, both tasks.
NoisyCounting performs best on carrabin (task-specific, expected). On
yoo, Mean has near-zero estimation error (it computes the exact running
mean) but humans diverge from it — motivating the temporal analyses.
Cross-task consistency of this pattern is the contribution.

### V figures — response variability structure (carrabin only)
NEF produces the right level and temporal structure of response
variability, which deterministic models (Mean, LI, PR) structurally
cannot, since they give identical responses to identical inputs.

### T figures — temporal dynamics of evidence integration
NEF captures within-sequence update dynamics: decay of update magnitude
(recency bias), individual λ differences, and accumulation/persistence of
response variability.

### N figures — neural predictions
The error ensemble generates quantitative neural predictions — PE
dynamics, variability scaling with α₀/n_neurons, weight-neuron activity
profiles — internally consistent with the behavioural fit and testable in
future neural recording studies. `neural_main` is the current, sole
authoritative version of this argument (see "Current thread" above).

---

## Current figure panel inventory

| Script | Group | Layout | Status |
|---|---|---|---|
| figure_carrabin_performance.py | P | 1×3 | Built |
| figure_carrabin_variability.py | V | 1×4 | Built |
| figure_carrabin_temporal.py | T | 1×4 | Built |
| figure_carrabin_neural.py | N | 1×4 | Built (older per-task taxonomy) |
| figure_yoo_performance.py | P | 1×3 | Built |
| figure_yoo_temporal.py | T | 1×4 | Built |
| figure_yoo_neural.py | N | 1×4 | Built (older per-task taxonomy) |
| figure_soltani_performance.py | P | 2×3 | Built, human-only (real pilot data) |
| figure_soltani_temporal.py | T | 2×6 | Built, human-only |
| figure_soltani_variability.py | V | 2×3 | Built, human-only |
| make_paper_figures.py: neural_main | N | 3×3 | Row 1-2 built; row 3 col 1 built, cols 2-3 in progress |

`figure_carrabin.py`/`figure_yoo.py` are legacy combined figures,
superseded by the split P/V/T/N scripts above.

---

## Future extensions (soft todos, not tied to any current figure's structure)

- **Validation via ablation/statistical control** (not yet built). For
  each parameter-vs-outcome relationship `neural_main` shows: a partial
  correlation controlling for the other parameters, and, where feasible,
  a mechanistic ablation (forcing a parameter to null and showing the
  correlation collapses) — matching yoo's existing λ=0 ablation
  precedent.
- **Synaptic vs. working-memory implementation comparison** (not
  started, separate downstream scope). Different predictions under an
  ITI manipulation depending on which implementation of the learning
  rule is assumed. Deliberately out of scope for now.
