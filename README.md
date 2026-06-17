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

| Name | N | Key features | Status |
|------|---|-------------|--------|
| carrabin | 21 | Binary inputs; 5 obs/trial; sequences repeat (qid); true_p known | Active |
| yoo | 38 | Continuous inputs; 30 obs/trial; no sequence repetition | Active |
| task-continuous | TBD | Continuous inputs; 15 obs/trial; Normal(mean, std_fixed=20); 40 trials; prefix_length=4 | **Under development** |
| task-binary | TBD | Binary inputs (blue/red); 15 obs/trial; Bernoulli(p); 40 trials; prefix_length=4 | **Under development** |

task-continuous and task-binary are designed to be completed within-subject
(same participants recruited via Prolific allowlist). Together they unlock all
PTN metrics simultaneously and enable cross-task individual-differences analysis
(same pid's λ across both task types). See task/ section below for details.

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
- N2 (panel B): PE variability and response variability covary tightly across
  participants (r=0.97****), using probe simulations run at each pid's fitted
  α₀ and λ. Partial correlation after regressing out α₀ remains r=0.97****,
  showing the covariation is driven by spiking noise propagating from the error
  ensemble to the value ensemble — not by individual differences in learning
  rate. α₀ acts as a shared scaling factor on both metrics, not an independent
  cause of their covariation. This is confirmed by the fact that the
  deterministic model (RL_lambda) produces zero within-qid variability
  regardless of α₀.
- N3 (panel C): Both response variability and PE variability decrease with
  fitted α₀ (r≈-0.56** and r≈-0.66**), confirming that α₀ scales the
  amplitude of spiking noise in the same direction for both metrics.
- N4 (panel D): Both variability metrics decrease with n_neurons for fixed
  α₀/λ (n_neurons scan). They converge to human levels at n~100-200,
  suggesting a plausible biological parameter range.

**Yoo N:**
- N5 (panel A): Weight-neuron activity in the error ensemble decays more steeply
  for high-λ pids — the neural signature of stronger temporal discounting.

- N6 (panel B): Activity decay (obs 1 − obs 30) correlates with |Δresponse| decay
  (early − late) per pid (r=0.68****). This coupling is mediated by λ: removing the
  temporal discounting mechanism (λ=0 ablation) eliminates the correlation
  (r=0.25 ns), confirming that the activity↔behaviour coupling arises from the
  counting dynamics that implement α(t), not from spiking noise alone.

- N7 (panel C): λ mediates both activity decay and |Δresponse| decay simultaneously.
  Human per-pid |Δresponse| decay values (grey reference lines) fall within the
  range of NEF model values, showing that the fitted NEF models span the observed
  human distribution. Higher λ → steeper decay in both neural activity and
  behavioural updating.

- N8 (panel D): Pids with higher late |Δresponse| (obs 21-30) also have higher
  late performance error — the U-shaped error curve seen in T1 is related to
  continued large updates late in the sequence. NEF reproduces this relationship
  tightly (r=0.89****); humans show the same trend (r=0.40*). This links the
  U-shape phenomenon in T to the neural updating mechanism in N.

**Connecting N to T:** λ drives both the temporal signatures in T (split-half
reliability, λ_model vs λ_human) and the neural dynamics in N (activity decay,
activity↔behaviour coupling). The U-shaped performance pattern in T1 is explained
mechanistically by N8: pids who keep updating late (high λ) accumulate more error
in the second half of the sequence because they over-weight recent noisy evidence.

**Potential panel E (planned):** λ vs late performance error per pid — showing
that both NEF and human late-error is predicted by fitted λ. This would close
the loop from T1 (group-level U-shape) → N8 (late update→late error) → N-E
(λ→late error), providing a complete mechanistic account.

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
| N2 | PE variability vs response variability, probe sims; partial-r control for α₀ (carrabin) |
| N3 | Response and PE variability vs fitted α₀ — shared scaling factor (carrabin) |
| N4 | Response and PE variability vs n_neurons scan; converge to human levels (carrabin) |
| N5 | Weight-neuron activity vs observation, split by λ group (yoo) |
| N6 | Mean weight-neuron activity vs mean |Δresponse| per observation (yoo) |
| N7 | λ mediates activity change and mean |Δresponse| (yoo) |
| N8 | Late |Δresponse| vs late estimation error, obs 21–30 (yoo) |

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
A: N1 PE dynamics | B: N2 PE vs response variability (r=0.97, partial r excl. α₀=0.97) | C: N3 variability vs fitted α₀ | D: N4 variability vs n_neurons

### figure_yoo_performance.py (P group, 1×3)
A: schematic | B: P1 estimation error | C: P2 model fit
Run: python scripts/figure_yoo_performance.py --run_folder yoo --nef_folder refit

### figure_yoo_temporal.py (T group, 1×4)
A: T1 + U-shape bands | B: T2 |Δresponse| | C: T3 split-half λ | D: T4 λ_model vs λ_human
Run: python scripts/figure_yoo_temporal.py --run_folder yoo --nef_folder refit

### figure_yoo_neural.py (N group, 1×4)
A: N5 weight activity by λ group | B: N6 activity decay vs |Δresponse| decay (fitted vs λ=0 ablation) | C: N7 λ mediates both decays | D: N8 late |Δresponse| vs late error
Run: python scripts/figure_yoo_neural.py --nef_folder refit --ablation_folder yoo_ablation

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

---

## New task (task/)

Two online experiments sharing all infrastructure, deployed separately on
Prolific via MindProbe/JATOS. Both use jsPsych 8 + Vite 6.

### Tasks

| Task | Stimulus | Response | Generative model |
|------|----------|----------|-----------------|
| Continuous | Integer -100..100 | Slider -100..100 | Normal(mean, std_fixed=20) |
| Binary | Blue/red circle | Slider 0–100% | Bernoulli(p) |

Both: 40 trials × 15 observations (u8_r5, prefix_length=4), response deadline per obs,
ITI clock, 5-observation interactive tutorial, post-trial summary plot.

### Design goals
- Continuous: repeated sequences + long sequences + continuous values — unlocks
  all PTN metrics simultaneously; cross-task λ correlation with binary task
- Binary: same participants (Prolific allowlist), Bernoulli generative model;
  enables cross-task individual-differences analysis

### Directory structure

```
task/
  src/
    shared/                        — all reusable code (both tasks)
      timeline-builder.js          — full timeline, parameterised by config
      jatos-shim.js                — dev no-op shim (POSTs to dev-server.js)
      plugin-observation.js        — continuous obs (number + slider + timeout)
      plugin-observation-binary.js — binary obs (circle + gradient slider + timeout)
      plugin-iti-clock.js          — inter-obs ITI clock
      plugin-tutorial-intro-continuous.js   — continuous interactive intro (3-stage reveal)
      plugin-tutorial-intro-binary.js       — binary interactive intro (3-stage reveal)
      plugin-practice-observation.js        — continuous tutorial obs 2–5
      plugin-practice-observation-binary.js — binary tutorial obs 2–5
      plugin-practice-summary.js            — continuous tutorial summary
      plugin-practice-summary-binary.js     — binary tutorial summary (bar chart)
      plugin-trial-summary.js               — continuous trial summary
      plugin-trial-summary-binary.js        — binary trial summary (bar chart)
      draw-performance.js          — SVG distribution + ticks (continuous summary)
      bar-chart.js                 — SVG bar chart (binary summary)
      style.css                    — all shared styles
    continuous/
      config.js                    — continuous task parameters (N_TRIALS_TO_RUN=2 dev; SET TO 40)
    binary/
      config.js                    — binary task parameters (N_TRIALS_TO_RUN=2 dev; SET TO 40)
    experiment-continuous.js       — entry point
    experiment-binary.js           — entry point
  sequences/
    continuous_sequences.json      — master stimulus sequences (imported by config.js)
    continuous_sequences.pkl       — analysis version
    binary_sequences.json          — master stimulus sequences (imported by config.js)
    binary_sequences.pkl           — analysis version
  generate_sequences.py            — sequence generation + seed search (--n_tries N)
  parse_results.py                 — JATOS JSON → tidy DataFrame → .pkl
  dev-server.js                    — local result capture server (port 3099)
  index-continuous.html            — Vite entry
  index-binary.html                — Vite entry
  index-dev.html                   — dev launcher (both tasks)
  package.json
  vite.config.js                   — mode-based multi-entry build
```

### Key parameters (src/continuous/config.js and src/binary/config.js)

```js
const N_TRIALS_TO_RUN        = 2;      // ← SET TO 40 BEFORE DEPLOYMENT
const SHOW_SLIDER_VALUE      = true;   // numeric label above slider thumb
const SLIDER_DEFAULT         = 'none'; // 'none' | 'last' | 'value'
const ITI_MS                 = 1000;
const T_OBS_MS               = 5000;
const SHOW_TRIAL_PERFORMANCE = true;   // post-trial summary plot
// Practice: 5 fixed observations hardcoded in config.js
```

### Commands

```bash
cd task
npm install               # first time only
npm run dev               # ← USE THIS for local testing (serves both tasks)
npm run dev:server        # local result capture (port 3099) — separate terminal
npm run build:continuous  # production build → dist-continuous/
npm run build:binary      # production build → dist-binary/
# Note: npm run dev:continuous / dev:binary use a different Vite config
# and will NOT work correctly for local testing
```

### Local testing pipeline

```bash
# Terminal 1: local result server
npm run dev:server

# Terminal 2: task
npm run dev
# Complete a task in browser → data saved to task/dev-results/

# Parse results
python task/parse_results.py --input_dir task/dev-results/ \
    --output data/task_results.pkl --verbose
```

### Deploying to MindProbe

1. `npm run build:continuous` → `dist-continuous/`
2. Zip `dist-continuous/` contents; upload to MindProbe as a JATOS study
3. Repeat for binary: `npm run build:binary` → `dist-binary/`
4. Set Prolific URL: `https://mindprobe.eu/publix/...?PROLIFIC_PID={{%PROLIFIC_PID%}}`
5. Set completion redirect: `https://app.prolific.com/submissions/complete?cc=<CODE>`
6. After continuous study completes, export completers' PIDs and create a
   Prolific allowlist study for the binary task

### Data format (downloaded from MindProbe / saved by dev-server)

`parse_results.py` filters to `screen='observation'` rows and outputs:

| Column | Type | Description |
|--------|------|-------------|
| `prolific_pid` | str | Prolific participant ID |
| `task` | str | `'continuous'` or `'binary'` |
| `trial` | int | 0-indexed trial number |
| `observation` | int | 0-indexed observation within trial |
| `value` | int | Stimulus (-100..100 continuous; -1/1 binary) |
| `true_mean` | float | Generative mean (continuous); NaN for binary |
| `true_std` | float | Generative std (continuous); NaN for binary |
| `true_p` | float | True Bernoulli probability (binary); NaN for continuous |
| `qid` | int | Unique sequence ID (structured trials); NaN for random |
| `trial_type` | str | `'structured'` or `'random'` |
| `prefix_length` | int | Number of fixed prefix observations |
| `std_condition` | float | Observation std (continuous); NaN for binary |
| `response` | float | Participant estimate (NaN if timed out) |
| `timed_out` | bool | True if response deadline elapsed |
| `rt` | float | Response time in ms (NaN if timed out) |
| `time_elapsed` | int | ms since experiment start |

Both tasks share the same output file — use `df[df.task == 'continuous']` to split.
