# Evidence Integration

## Scientific overview

This project studies **how people integrate sequential noisy evidence**, using
cognitive models and a biophysical spiking neural network (NEF) to identify
the computational and neural mechanisms underlying that process.

Three goals:

**1. Cross-task generalisation of cognitive mechanisms.** The same NEF
architecture is applied across multiple tasks (carrabin, yoo) without
task-specific modification, demonstrating that the model captures a general
cognitive mechanism rather than a task-specific fit. The NEF is benchmarked
against established models from the evidence-integration literature: an optimal
Bayesian integrator (Mean), a leaky integrator (LeakyIntegrator), and primacy/
recency weighting models (PrimacyRecency). The NEF matches or exceeds these
models across both tasks.

**2. Emergent higher-order behavioural signatures.** Beyond response-level fit
(RMSE), the NEF naturally reproduces secondary behavioural phenomena without
being trained to do so: temporal update patterns, the decay of response-change
magnitude across the sequence, individual differences in discounting rate (λ),
and state-persistent response variability. These phenomena emerge from the
model's spiking dynamics.

**3. Joint behavioural and neural predictions.** The NEF produces both
behavioural and neural predictions from the same mechanism. Neural: error-
ensemble activity tracks the per-observation weight α(t) and decays with λ;
this signal correlates strongly with human behavioural updating (r=0.92).
Behavioural: λ mediates both neural activity change and mean response update
magnitude across participants. These are framed as testable predictions for
future empirical work (e.g. EEG/fMRI); we do not have empirical neural data.

Central model: **α(t) = α₀ / t^λ** (power-law decaying learning rate).
In the NEF this emerges from spiking dynamics rather than being hardcoded.
RL_lambda implements the same equation explicitly — it is the mathematical
theory that the NEF realises biophysically, not a point of direct comparison.

---

## Tasks

| Name | N | Key features |
|------|---|-------------|
| carrabin | 21 | Binary inputs; 5 obs/trial; sequences repeat (qid); true_p known |
| yoo | 38 | Continuous inputs; 30 obs/trial; no sequence repetition |

Proposed new task: repeated sequences (carrabin) + long sequences + continuous
values (yoo), completed by the same participants. Unlocks all PTN metrics
simultaneously and enables cross-task individual-differences analysis (same
pid's λ across both task types). Target ≥20 obs/trial for reliable per-pid
U-shape and λ detection. May include a working memory manipulation to test
NEF-specific predictions.

---

## Scientific narrative per figure group

This section captures the intended argument for each figure group so that the
scientific logic is preserved across work sessions.

### P figures — Establishing the model as a credible fit

**Intent:** Show that NEF fits human responses at least as well as other models
across both tasks, establishing it as a viable model before making stronger claims.
This is not the star of the paper — it is the prerequisite.

**Carrabin:** NEF competitive with or better than Mean/LI/PR on RMSE. NoisyCounting
performs best (task-specific), expected.
**Yoo:** Same story. Mean has near-zero estimation error (it computes the exact
running mean), but humans diverge — motivating the temporal analyses.
**Key point:** Cross-task consistency of the fit pattern is the P-figure contribution.
It cannot be said that NEF fits one task by overfitting — the same parameters work on both.

### V figures — Capturing the structure of response variability (carrabin only)

**Intent:** Show that NEF produces the right level and temporal structure of
response variability, which purely deterministic models (Mean, LI, PR) cannot
do because they produce identical responses to identical inputs.

**Key results:**
- V2: Human response variability for identical inputs is substantial and stable
  across individuals. NEF naturally produces non-zero variability; other models
  predict zero. This is not a parameter — it is an emergent property of spiking.
- V3: Variability is a stable individual trait (r=0.88 split-half), not noise.
  NEF matches this reliability; NoisyCounting (MLE) also does.
- V1 (NLL): NEF captures the full response distribution, not just the mean.
- NoisyCounting's role: demonstrates that RMSE fitting misses state noise
  (sigma_c → 0 under RMSE), motivating MLE. But NoisyCounting is task-specific
  and lacks neural interpretation — the main V story is about NEF.

**Not possible for yoo:** No repeated sequences → cannot compute V2/V3.
This directly motivates the new task design.

### T figures — Logical elimination: only NEF reproduces all temporal signatures

**Intent:** The four panels together form an elimination argument. Each model
fails on at least one panel; only NEF passes all four.

**Carrabin T (T5/T6 — state persistence):**
- T5 (residual variance growth): Human response variance grows across obs within
  a trial — state noise accumulates. NEF + NoisyCounting (MLE) reproduce this;
  deterministic models produce flat variance. Mean/LI/PR fail.
- T6 (autocorrelation): Human residuals are autocorrelated within trial (lag-1
  r~0.62) — a state-persistence signature. NEF reproduces (r~0.78); uncorrelated
  noise models (RMSE-fitted NoisyCounting) produce near-zero.
- These panels cannot be run on yoo (no qid), directly motivating new task.

**Yoo T (T1–T4 — power-law decay and individual differences):**
- T1 (estimation error vs obs): Mean produces monotonically decreasing error;
  humans show U-shaped error curves (error decreases early then rises). Mean fails.
- T2 (|Δresponse| vs obs): LeakyIntegrator produces rapid decay to near-zero
  updating; humans maintain substantial updating throughout. LI fails here.
- T3 (split-half λ reliability): LeakyIntegrator's λ has *negative* split-half
  reliability (r=-0.56) — its apparent power-law decay is not a stable individual
  property. PrimacyRecency (r=0.94) and NEF (r=0.83) are reliable. LI fails here.
- T4 (λ_model vs λ_human regplot): Mean (r=0.14 ns, flat), LI (r=-0.32*,
  inverted), PR (r=0.69****), NEF (r=0.61****). Mean and LI fail. PR and NEF
  both pass — but PR lacks neural interpretation (see N figures).

**Conclusion:** Only NEF passes all panels across both tasks. This is the core
behavioural argument.

### N figures — Testable neural predictions from the same mechanism

**Intent:** Show that the NEF's internal dynamics generate specific, measurable
neural predictions that no other model can make. These are predictions for future
empirical work, not empirical findings. The framing is: "if the NEF is the right
model, here is what you should see in neural recordings."

**Carrabin N:**
- N1: PE timecourse shows the prediction-error signal decoded from the error
  ensemble rises at observation onset and decays as the estimate stabilises.
  Different α₀/n_neurons combinations produce characteristically different
  timecourses — testable in EEG/single-unit data.
- N2/N3: Both response variability and PE variability decrease with n_neurons
  (more neurons = less spiking noise = less variability). The two converge to
  human levels at n~100-200, suggesting a plausible biological parameter range.

**Yoo N:**
- N4: Weight-neuron activity in the error ensemble decays more steeply for
  high-λ pids — the neural signature of stronger discounting.
- N5: Mean weight-neuron activity correlates strongly with mean |Δresponse|
  across observation positions (r=0.92). The model's α(t) signal directly drives
  the magnitude of behavioural updating — this is the key mechanistic link.
- N6: λ mediates both the neural activity change (activity decay across sequence)
  and the behavioural update magnitude (mean |Δresponse|). Both relationships
  are negative and significant, showing λ as the shared parameter.
- N7: Late |Δresponse| vs late estimation error (obs 21-30) — pids who keep
  updating late also have higher late error. NEF reproduces this tightly
  (r=0.89****); humans show the same trend but noisier (r=0.40*), consistent
  with the model capturing the mechanistic relationship.

**Connecting N to T:** The same λ that differentiates models in T3/T4 is also
the parameter that drives the neural predictions in N4-N6. This is the punchline:
λ is not just a behavioural parameter — it has a specific neural implementation
in the counting subnetwork whose activity is directly observable.

---

## Metric taxonomy (PTN)

One figure per group per task. Figures save PDF only.

### P — Performance
| Code | Metric |
|------|--------|
| P1 | Estimation error: RMSE to hidden probability / true mean; human + models |
| P2 | Model fit: RMSE to human responses; model comparison |

### T — Temporal
| Code | Metric |
|------|--------|
| T1 | Task performance vs observation position |
| T2 | Response change (|Δresponse|) vs observation |
| T3 | Split-half reliability of λ (first vs second half of trials) |
| T4 | Dynamical model fit: λ_model vs λ_human regplot |
| T5 | Residual variance growth across obs (state noise accumulation; carrabin) |
| T6 | Within-trial residual autocorrelation decay (state persistence; carrabin) |

λ fitted via curve_fit A·n^(-λ), bounds [0,2], obs ≥ 2.

### N — Neural (NEF predictions; testable in future experiments)
| Code | Metric |
|------|--------|
| N1 | Decoded PE timecourse within observation window |
| N2 | Response/PE variability vs n_neurons (carrabin) |
| N3 | PE variability vs response variability, matched probe simulations (carrabin) |
| N4 | Weight-neuron activity vs observation, split by λ group (yoo) |
| N5 | Mean weight-neuron activity vs mean |Δresponse| per observation (yoo) |
| N6 | λ mediates activity change and mean |Δresponse| (yoo) |
| N7 | Late |Δresponse| vs late estimation error, obs 21–30 (yoo) |

---

## Models

| Model | Role | Free params |
|-------|------|-------------|
| Mean | Optimal Bayesian baseline (running mean) | none |
| LeakyIntegrator | Exponential forgetting | gamma |
| PrimacyRecency | Explicit primacy + recency weighting | eps_p, eps_r |
| NoisyCounting | Task-specific (Prat-Carrabin 2024); carrabin only | mu, sigma_c, nu |
| RL_lambda | Mathematical theory underlying NEF (power-law delta rule) | alpha_0, lambda_ |
| NEF | Spiking neural network (emergent power-law dynamics) | alpha_0, lambda_ |

NoisyCounting: RMSE-fitted sigma_c collapses to ~0; MLE-fitted recovers
sigma_c ~0.03–0.08. Both versions are scientifically meaningful.

---

## Current figure inventory

### figure_carrabin_performance.py (P group, 1×3)
A: schematic | B: P1 estimation error | C: P2 model fit

### figure_carrabin_variability.py (V group, 1×4)
A: V2 KDE | B: V2 RMSE regplot | C: V3 test-retest | D: V1 NLL boxplots

### figure_carrabin_temporal.py (T group, 1×4)
A: T1 RMSE vs obs | B: T2 |Δresponse| vs obs | C: T6 autocorrelation | D: T5 variance growth

### figure_carrabin_neural.py (N group, 1×4)
A: N1 PE dynamics | B: N3 PE vs response variability | C: N2 variability vs n_neurons | D: N2 slope_c vs n_neurons

### figure_yoo_performance.py (P group, 1×3)
A: schematic | B: P1 estimation error | C: P2 model fit
Run: python scripts/figure_yoo_performance.py --run_folder yoo --nef_folder refit

### figure_yoo_temporal.py (T group, 1×4)
A: T1 + U-shape bands | B: T2 |Δresponse| | C: T3 split-half λ | D: T4 λ_model vs λ_human
Run: python scripts/figure_yoo_temporal.py --run_folder yoo --nef_folder refit

### figure_yoo_neural.py (N group, 1×4)
A: N4 weight activity by λ group | B: N5 activity vs |Δresponse| | C: N6 λ twin-axis | D: N7 late error vs late delta
Run: python scripts/figure_yoo_neural.py --nef_folder refit

---

## Fitting pipeline

### RMSE fitting (cluster)
    venv/bin/python -m fitting.submit carrabin NEF --n_trials 100 --run_folder carrabin --k 5
    venv/bin/python -m fitting.submit yoo NEF --run_folder yoo --n_trials 100 --k 5
    venv/bin/python -m fitting.collect carrabin --type params
    venv/bin/python -m fitting.collect carrabin --type responses
    venv/bin/python -m fitting.collect yoo --type params
    venv/bin/python -m fitting.collect yoo --type responses
    venv/bin/python -m fitting.collect yoo --type activities --ensembles error --timing once_per_obs

### MLE fitting (NoisyCounting, carrabin only)
    bash jobs/submit_mle_fit.sh NoisyCounting carrabin 500 100

### Counting activity files (must exist on cluster before NEF fitting)
    venv/bin/python models/counting_integrator.py --precompute_activities \
        --n_neurons 200 --n_neurons_counting 1000 --dataset yoo --n_trials 30
    scp data/counting_activities_n200_nc1000_yoo.pkl \
        f007qzn@discovery.dartmouth.edu:~/evidence_integration/data/

---

## Environment

Always use: /home/psipeter/evidence_integration/venv/bin/python
Cluster: /dartfs-hpc/rc/home/n/f007qzn/

---

## Archive

Older models/data (diederen, jiang, usher) in archive/. Do not reactivate.
Legacy combined figures (figure_carrabin.py, figure_yoo.py) retained for reference.
