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

**Settled this session:** investigated why LeakyIntegrator/RL_lambda
underperform PrimacyRecency on colors/numbers specifically, using the
running-mean-ground-truth reframing below (Metric taxonomy 2.3). Two
separable causes, resolved differently: colors/numbers pids' very first
response anchors almost exactly on that trial's raw first observation
(fixed via an `init_from_obs1` fit flag, kept for LeakyIntegrator, dropped
for RL_lambda — see `docs/DECISIONS.md`), and RL_lambda's `lambda_` search
bound was silently capped at 1.0 with ~80% of colors pids pinned there
(widened to 2.0). Neither fully closes RL_lambda's residual gap to
PrimacyRecency on colors — a structural, not fitting-range, limitation
(Metric taxonomy 2.3). Also found and fixed, unrelated: a carrabin-only
obs-indexing bug in the Laplace-smoothing response transform.

**Also settled this session:** the `NoisyRL_lambda` figure-level stand-in
— left in place when the model itself was retired (above), since
colors/numbers didn't have real NEF fits yet at the time — is now fully
closed out. NEF fills that role everywhere it's needed
(`VARIABILITY_STOCHASTIC_MODEL`, `SIGMA_CORR_MODELS`), matching balls' own
existing convention. Two figures that depended on the old stand-in and
had become redundant were archived: `variability_models` (its
model-overlay branch had been sitting disabled since before this session;
the same per-pid comparison it would show is already covered by
`sigma_model_correlation`) and `variance_autocorr_models` (fully
superseded by `sigma_main` row 3, which already shows the identical
metric/roster with NEF — not `NoisyRL_lambda` — in the 4th slot). See
`docs/DECISIONS.md`.

**Also settled this session (new, separate thread): NEF_synaptic
reimplemented.** Retired in May (`e5281f6`) when the project consolidated
to a single active NEF variant; revived per this doc's own "Future
extensions" entry below. Same architecture as NEF (recurrent) except the
value-dynamics block: a `background` population (fires from bias only)
connects to `value` via a PES-learned connection, driven by `error`'s
product signal through the learning rule (`transform=-T_error`) instead of
a direct multiplicative connection plus a `value` self-recurrence.
Branches on `nef_type` (derived from `model_type`, `"synaptic" in
model_type`) inside `models.NEF.build_network` itself, not a separate
model file — the counting-subnetwork/activity-loading/dataset-dispatch
machinery that's historically been the real source of bugs here (see
CLAUDE.md's session-start checklist) stays single-sourced.
`scripts/check_NEF_pipeline.py` gained `--model_type`, `--compare_synaptic`,
and `--pes_learning_rate_sweep` for exactly this kind of diagnostic.

Baseline check (reusing NEF (recurrent)'s own already-fitted alpha_0/
lambda_, not an independent fit): NEF_synaptic tracks the same qualitative
integration direction as NEF (recurrent) on every sampled trial, visually
confirmed across all four datasets, but consistently lags/underweights
sudden evidence swings — smoother, slower-responding dynamics, as expected
for a PES-learned ("activity-silent"/synaptic-trace) mechanism versus a
persistent-recurrent-activity one. `pes_learning_rate` (fixed,
`_NEF_FIXED`) was swept and set to `4e-4` (was a leftover `1e-4` from the
pre-May implementation — see `docs/DECISIONS.md` for the tuning). At that
value, NEF_synaptic vs. NEF (recurrent) RMSE runs 0.03–0.09 depending on
dataset, and NEF_synaptic vs. RL_lambda RMSE is comparable in magnitude to
NEF (recurrent)'s own RL_lambda RMSE (~0.05) — i.e. NEF_synaptic is a
credible fit to the same underlying computation, not a broken
reimplementation, with the expected mechanistic difference in dynamics.

**Not yet a real quantitative comparison:** NEF_synaptic has no Optuna fit
of its own (no `MODEL_PARAMS` entry, and `fitting/fit.py`'s objective
dispatch is still `model_type == "NEF"` exact-match — would need
`.startswith("NEF")`, matching the convention `fitting/submit.py`/
`utils/run_params.py` already use). Before any real synaptic-vs-recurrent
behavioural claim, NEF_synaptic needs its own per-pid RMSE fit (own
alpha_0/lambda_, not NEF (recurrent)'s borrowed values) — see "Future
extensions" below.

Refitting the `_resp_noise` models under the corrected setup (this
session's carrabin fix, `lambda_` bound, and `init_from_obs1`) also
surfaced a known artifact in two new places: colors' LeakyIntegrator/
RL_lambda now hit the same `add_noise` boundary-clipping issue
Mean/PrimacyRecency's colors fits already needed a correction for (see
`docs/DECISIONS.md`'s "Colors' LeakyIntegrator/RL_lambda added to the
sigma-growth boundary-clipping correction") — `sigma_main` row 2 now
applies that same correction to all four. Checked whether Human/NEF
needed the same treatment: NEF doesn't (its own decoded value essentially
never approaches the boundary, any task, any observation). Human's colors
responses DO show the same near-boundary pattern, and a correction is
feasible (using each qid group's own mean response in place of a model's
mu), but is deliberately not applied — unlike the math models' exactly-known
noise process, this would rest on an unvalidated Gaussian-noise assumption
about human behavior, and it doesn't just flatten the curve, it reverses
its apparent direction (see `docs/DECISIONS.md` for the actual numbers).

**Also settled this session (new, separate thread): `neural_main`'s row
1/row 2 dependent variables converted to relative (%) framing.** Row 1's
"PE decrease" and row 2's "Activity decay"/"ΔR decay" were absolute
differences, confounded (row 1) or just less interpretable (row 2) than
a percentage of each quantity's own starting value — see
`docs/DECISIONS.md` for the full reasoning and the correlation numbers
that motivated it. Now "PE decay (%)", "Activity decay (%)", "ΔR decay
(%)" throughout. Row 3's "PE noise" gets an analogous
coefficient-of-variation treatment (`pe_cv_pct`) — designed and verified
locally (well-scaled, decreases with `n_neurons` as expected), but NOT
yet wired into the figure: `_n_neurons_snr_worker` now saves the
`pe_mean_mean` the ratio needs, but the existing 50-cell grid predates
that field and needs a full resubmit/recollect before `neural_main`'s
row 3 can actually switch over.

**Also settled this session: `gate_error_feedback`, an alternative
ITI-silencing mechanism in `models.NEF.build_network` (default off).**
The ITI-perturbation experiment (`scripts/neural_experiments.py`'s
`iti_perturbation`) found NEF_synaptic essentially immune to ITI noise
injected on `value`'s neurons, unlike NEF (recurrent) — largely an
artifact of how the ITI silences the value→error feedback: by default,
`node_input[1]` inhibits `error.neurons` directly, blocking ALL of
error's activity (including any noise-driven learning) during the ITI.
`gate_error_feedback=True` instead routes `value→error[dim 1]` through an
intermediate `gate` population and moves the inhibition onto `gate`'s
neurons instead of `error`'s — `error` itself is never silenced, so
during the ITI its dim-1 input is `obs(~0) - gate(~0)` (small,
noise-driven) rather than fully blocked, letting injected noise leak a
little into downstream learning/feedback instead of being fully gated
out. Applies identically to `nef_type="recurrent"` (there the leak feeds
`value`'s dynamics directly and continuously, not just a learning rule,
so baseline drift is expected to rise somewhat even without injected
noise) and `"synaptic"`.

Tuned via local dynamics inspection (`scripts/check_NEF_pipeline.py
--gate_error_feedback`, `--plot_trials`): the first pass (`tau_ff` on
both new hops, `-10.0`/`tau_error`-filtered inhibition on `gate`, mirroring
the default path's own inhibition exactly) produced a startup transient —
`error` briefly spiked toward the raw input before `value`'s feedback
through the extra hop caught up — and value traces diverged more than
"fairly similar" between gate on/off on some trials even with zero
injected perturbation noise (per-trial correlation as low as 0.59, max
abs diff up to 0.92 on a radius-1.0 ensemble at `n_neurons=50`), likely
compounded by that small network size. Fixed by halving the synapse on
each of the two new hops (`tau_ff/2` each, so total lag is comparable to
the single-`tau_ff`-hop paths already feeding `error`, not double), and
weakening/defiltering `gate`'s inhibition (`-3.0`, `synapse=0` instead of
`-10.0`, `synapse=tau_error`) so `gate`'s neurons recover quickly right
when the ITI ends rather than lagging on a filtered rebound. Re-checked
at carrabin's production size (`n_neurons=500`, `n_neurons_counting=500`)
via manual inspection: value traces now track closely between gate
on/off.

**Also settled this session: `synaptic_main`'s ITI-perturbation experiments
tried, then reverted from, per-model-type fitted params.** At the
original shared `(alpha_0=0.7, lambda_=0.7)`, NEF_synaptic's baseline
(strength=0, `gate_error_feedback=True`) RMSE ran ~45% higher than NEF's
— the shared value was itself a confound. Tried replacing it with pid
33's own independently-fitted `(alpha_0, lambda_)` per model_type (NEF:
0.999, 0.194; NEF_synaptic: 0.880, 0.226) — the pid where the two
models' real-data RMSE/sigma are closest together. Reverted (see
docs/DECISIONS.md for both entries): pid 33's fitted `lambda_` decays far
more slowly than the old shared value, and within this experiment's
4-observation window a high alpha(t) at every step lets each new
observation erase whatever the ITI noise just did before the readout —
confirmed directly, pilot data (20 sessions, gate on) came back with
NEF's own RMSE completely flat across strength. Also, on reflection,
using one real pid's fit on synthetic trials that pid never saw, and
breaking with `neural_main`'s own all-artificial-params convention, were
both worth avoiding independent of the masking effect.
`iti_perturbation`/`iti_perturbation_dynamics` are back to a single
shared `--alpha_0`/`--lambda_` float; `_parse_param_map`/the `KEY=VALUE`
CLI syntax removed as dead code.

**Not yet started:** the "Future extensions" below (ablation/statistical
validation of `neural_main`'s parameter-vs-outcome relationships). Model
fitting against real `task_backend` (soltani) data — the human-only pilot
figures exist, but NEF/math-model fits to that data haven't been run yet.
The synaptic-vs-working-memory implementation comparison is underway (see
above) — NEF_synaptic reimplemented and baseline-checked, with an initial
Optuna RMSE fit now collected (`data/runs/nef_synaptic/`, 100 trials/pid,
all 4 datasets, not yet promoted to the canonical `rmse` folder pending
review). Finding a new shared `(alpha_0, lambda_)` with comparable
baseline RMSE/sigma between model_types (a small local grid search, not
yet run) is the remaining prerequisite before `synaptic_main`'s
dynamics/dose-response panels can be regenerated for real — the current
on-disk data for both is stale (old shared-value dose-response data, and
dynamics data clobbered by an unrelated debug test).

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

`gate_error_feedback` (params flag, default `False`) swaps how the ITI
silences the value→error feedback: off, `node_input[1]` inhibits
`error.neurons` directly (blocks all of error's activity during the ITI);
on, an intermediate `gate` population carries value→error instead, and
the inhibition targets `gate.neurons` — `error` keeps firing on small
residual/noise-driven activity through the ITI rather than being fully
silenced. Applies identically to both `nef_type` branches. See "Current
thread" and docs/DECISIONS.md for why and the tuning.

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
locates them on the primacy↔recency spectrum — `lambda_overview` (per-task
λ distributions, row 1; cross-task/split-half reliability, row 2). A
standalone human-only breakout (`lambda_human`/`lambda_sanity_human`) was
tried and deleted as a pure duplicate of this composite's own two rows.

**2.3 Running-mean ground truth and the colors/numbers weighting kernel.**
Redefining ground truth as the running mean of raw observations (rather
than the fixed generative parameter) surfaces a distinct signature: human
colors/numbers error relative to this running mean *increases* over a
trial (recency-biased divergence) for the large majority of individual
pids, not just as a population-median artifact. Snacks looks different —
its apparent population-level U-shape is mostly a joystick-catch-up
artifact at trial onset (the slider always resets to center, confirmed in
`task_backend` source, not a UI bug — a genuine motor-control confound
specific to that task's response device) mixed with a real
majority/minority split in individual trajectories, not a per-pid U-shape.

Colors' empirical weighting kernel (pooled regression of final response
on all 15 raw observations, non-anchor-mismatch pids) is genuinely
double-humped: highest at the first observation, a trough mid-trial, and
a real recency uptick at the end — the same primacy-and-recency pattern
Yoo et al. (2025) report for their own (different) task. PrimacyRecency's
two independent exponentials (`eps_p`, `eps_r`) can represent this shape;
RL_lambda's single power-law decay rate cannot, for *any* parameter
value — its implied per-observation weight is mathematically
non-increasing over the course of a trial, so it can approximate a steep
early decay but never a late re-elevation. This is a structural
limitation of the functional form, not something a wider search range can
fix (see `docs/DECISIONS.md`'s fitting-config entry for the resulting
decision on `init_from_obs1`/the `lambda_` bound).

Separately: colors/numbers pids' very first response tracks that trial's
raw first observation almost exactly, rather than a partial update from a
neutral prior — true for the large majority, but a genuine minority (most
clearly 2 colors pids, 100% consistent across all 32 of their trials, and
otherwise indistinguishable from typical end-of-trial performance) instead
show a stable, low-variance pattern of ignoring a single binary sample
entirely — plausibly a deliberate skepticism-of-weak-evidence strategy
rather than a mistake. One further pid shows a genuine within-session
learning transition (near-total non-anchoring in the first half of their
session, near-perfect anchoring in the second).

### 3. Sigma exploration
Goal 4's novel prediction: spiking noise is qualitatively different from
i.i.d. response noise, and NEF — not the deterministic-plus-`_resp_noise`
models — is the one that reproduces it.

**3.1 Individual differences in response variability for humans.** How
noisy is each person's response to a repeated, identical stimulus —
`sigma_overview` (per-task KDE, row 1; split-half reliability, row 2).
Standalone human-only breakouts (`variability_human`/`sigma_sanity_human`)
and a model-overlay variant (`variability_models`) were all tried and
removed as redundant: the human KDE/reliability panels duplicate
`sigma_overview`'s own two rows, and the per-pid model-vs-human comparison
is shown, more informatively, by `sigma_model_correlation` below.

**3.2 Growth of variability during the early sequence.** Response noise
should accumulate over the course of a trial if it's state-persistent
rather than i.i.d. — `sigma_main` row 2 (normalised residual-variance
growth vs observation, human + models + NEF, all three tasks). NEF tracks
the human growth pattern; the `_resp_noise` models do not.

**3.3 Autocorrelation of deviation from average behaviour at time t vs
t+k.** The more direct signature of state persistence: genuine
state-persistent noise produces decaying positive autocorrelation of the
residual; pure i.i.d. response noise looks like scatter around zero at
every lag — `sigma_main` row 3 (human + models + NEF, all three tasks).
Two dedicated standalone figures that used to carry this metric were tried
and deleted as fully redundant with this row: `variance_autocorr_models`
(overlaid the same models on top of the human panels) and
`variance_autocorr_human` (the human-only panels alone, plus a schematic
panel with no other current home — see `docs/DECISIONS.md`). As with
growth, NEF reproduces the human autocorrelation pattern; the
`_resp_noise` models don't.

Together, 3.2 and 3.3 are the empirical core of Goal 4: two independent
metrics, both distinguishing NEF's spiking-noise mechanism from ordinary
response noise on the same behavioural data the models were fit to.

**Supplementary:**
1. **Lambda for balls task (no decay)** — `lambda_balls`, kept separate
   since balls doesn't show the expected decay the other three tasks do.
2. **Lambda and sigma reliability within and across tasks** —
   `lambda_reliability`/`sigma_reliability` (odd/even split-half, within
   task), `lambda_sigma_crosstask` (colors-vs-numbers, paired) — the same
   reliability/cross-task pair also appears combined in
   `lambda_overview`/`sigma_overview` row 2.
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
| `lambda_overview` | 2.2 composite | 2×4 |
| `lambda_balls` | Supplementary | 1×1 |
| `lambda_reliability` | Supplementary | 1×3 |
| `sigma_reliability` | Supplementary | 1×3 |
| `lambda_sigma_crosstask` | Supplementary | 1×2 |
| `lambda_model_correlation` | Supplementary | 1×3 |
| `lambda_humanvmodel` | Supplementary | 1×3 |
| `sigma_model_correlation` | Supplementary | 1×3 |
| `sigma_overview` | 3.1 composite | 2×4 |
| `sigma_main` | 3.1 + 3.2 + 3.3 composite | 3×3 |
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
- **Synaptic vs. working-memory implementation comparison** (in
  progress) — `neural_main`'s row 5. NEF_synaptic reimplemented and
  baseline-checked against NEF (recurrent) this session (see "Current
  thread" above): qualitatively consistent integration direction, same
  order-of-magnitude RL_lambda agreement, but smoother/lagged dynamics, as
  expected. Remaining before the actual ITI-manipulation experiment: (1)
  NEF_synaptic needs its own Optuna RMSE fit (own alpha_0/lambda_, not NEF
  (recurrent)'s borrowed values) — requires a `MODEL_PARAMS` entry and
  fixing `fitting/fit.py`'s `model_type == "NEF"` exact-match dispatch
  (should be `.startswith("NEF")`, matching `fitting/submit.py`'s own
  convention); (2) `archive/scripts/iti_perturbation.py`'s gated ITI-noise
  injection (already built, currently archived) would need reviving to
  actually run the manipulation.
