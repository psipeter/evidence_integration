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

### P figures — Establishing the model as a credible fit

**Intent:** Show that NEF fits human responses at least as well as other models
across both tasks, establishing it as a viable model before making stronger claims.

**Carrabin:** NEF competitive with or better than Mean/LI/PR on RMSE. NoisyCounting
performs best (task-specific), expected.
**Yoo:** Same story. Mean has near-zero estimation error (it computes the exact
running mean), but humans diverge — motivating the temporal analyses.
**Key point:** Cross-task consistency of the fit pattern is the P-figure contribution.

### V figures — Capturing the structure of response variability (carrabin only)

**Intent:** Show that NEF produces the right level and temporal structure of
response variability, which purely deterministic models (Mean, LI, PR) cannot
do because they produce identical responses to identical inputs.

### T figures — Temporal dynamics of evidence integration

**Intent:** Show that the NEF captures the within-sequence dynamics of human
updating behaviour: how update magnitudes decay across observations (recency
bias), individual differences in λ, and the accumulation and persistence of
response variability across the sequence.

### N figures — Neural predictions

**Intent:** Demonstrate that the error ensemble in the NEF generates specific,
quantitative neural predictions — PE dynamics, variability scaling with α₀ and
n_neurons, and weight-neuron activity profiles — that are internally consistent
with the model's behavioural fit and testable in future neural recording studies.

---

## Repository structure

```
evidence_integration/
  data/
    carrabin.pkl
    carrabin_original.csv
    yoo.pkl
    counting_activities_n{n}_nc{nc}_{dataset}.pkl
    runs/
      carrabin/      — RMSE fits + MLE fits + extras
      yoo/           — RMSE fits for non-NEF models
      refit/         — NEF responses/params/activities (yoo + carrabin)
    sim_db/          — MLE simulation database
    optuna/          — MLE Optuna SQLite databases
  models/
    math_models.py
    NEF.py
    counting_integrator.py
    RNN.py
  fitting/
    fit.py           — Optuna k-fold CV RMSE
    fit_mle.py       — MLE via shared simulation database
    model_params.py  — MODEL_PARAMS, MLE_PARAMS, NEF_N_NEURONS_VALUES
    submit.py
    collect.py
    losses.py
  utils/
    paths.py
    plot_style.py
    slurm.py
    carrabin_transform.py
    save_responses.py
  scripts/
    figure_carrabin_performance.py
    figure_carrabin_variability.py
    figure_carrabin_temporal.py
    figure_carrabin_neural.py
    figure_yoo_performance.py
    figure_yoo_temporal.py
    figure_yoo_neural.py
    extras_carrabin.py
    extras_yoo.py
  jobs/
  task/              — online experiment (see task/ section below)
  venv/
```

---

## task/ — Online Experiment

Two online experiments deployed on Prolific via MindProbe/JATOS:
- **Continuous task**: Normal(mean, std=20) stimulus; slider response [0–100]; 40 trials × 15 obs
- **Binary task**: Bernoulli(p) stimulus (blue/red circle); slider response [0–100%]; 40 trials × 15 obs

Both tasks share all infrastructure (jsPsych 8, Vite 6, shared plugins/CSS).
Data pipeline: JATOS JSON → `task/parse_results.py` → `data/task_results.pkl`.
Target: ~50–80 participants per task, within-subject.

### Design

- **Sequences**: 10 unique sequences × 4 repeats = 40 trials; prefix_length=4; std_fixed=20
- **ITI**: all trials use 1000ms (ITI manipulation removed — no effect observed in pilot)
- **BTI**: 3s between-trial reset screen ("Trial X / 40 — generating new distribution…")
- **Distractor**: `iti_condition` per trial ('control'/'distract', 2-of-4 per qid);
  `DISTRACTOR_TYPE` in config: 'none' | 'iti_length' | 'popup' (default: 'none')
- **Timeout**: 7s response deadline per observation
  - Per-trial timeout budget: 3 timeouts before session terminates
  - Timeout flow: "Too slow" screen (fade in/out/in, 3.2s) → replay ITI → same observation
  - On 3rd timeout: "Too slow / 0 timeouts remaining" → "Session terminated" screen with button
- **Tutorial**: 3-box progressive reveal → 4 practice observations → practice summary →
  timeout demonstration (3 screens) → BTI → trial 1
- **Summary slides**: binary — per-obs bar chart (blue/red split at estimate, obs circle left);
  continuous — per-obs number line (red obs thumb, black circle at estimate)
- **Consent form**: verbatim IRB-approved text from task/consent_form.txt

### Sequence generation

```bash
# Regenerate with best known seeds
python task/generate_sequences.py --task continuous --seed 934
python task/generate_sequences.py --task binary --seed 1229
```

Best seeds: continuous=934, binary=1229 (prefix_length=4, k_std=0.7).
All trials use ITI_MS=1000ms.

### Directory structure

```
task/
  src/
    shared/
      build-trial-timeline.js      — pure-JS trial loop (importable by test harness)
      timeline-builder.js          — full timeline, parameterised by config
      jatos-shim.js                — dev no-op shim
      plugin-inter-trial.js        — BTI reset screen
      plugin-observation.js        — continuous obs (number + slider + timeout clock)
      plugin-observation-binary.js — binary obs (circle + gradient slider + timeout clock)
      plugin-iti-clock.js          — ITI countdown clock; shows "Too slow" if timed_out=true
      plugin-timeout-demo.js       — 3-screen timeout explanation (tutorial)
      plugin-tutorial-intro-continuous.js   — continuous 3-stage interactive intro
      plugin-tutorial-intro-binary.js       — binary 3-stage interactive intro
      plugin-practice-observation.js        — continuous tutorial obs 2–5
      plugin-practice-observation-binary.js — binary tutorial obs 2–5
      plugin-practice-summary.js            — continuous tutorial summary
      plugin-practice-summary-binary.js     — binary tutorial summary
      plugin-trial-summary.js               — continuous trial summary
      plugin-trial-summary-binary.js        — binary trial summary
      draw-performance.js          — SVG distribution + dot plots (continuous)
      bar-chart.js                 — SVG bar + dot plots (binary)
      urn-binary.js                — shared binary urn SVG
      slider.js                    — continuous slider
      slider-binary.js             — binary slider (blue/red gradient, split % labels)
      style.css                    — all shared styles
    continuous/
      config.js
    binary/
      config.js
    experiment-continuous.js
    experiment-binary.js
  sequences/
    continuous_sequences.{pkl,json}   — seed=934
    binary_sequences.{pkl,json}       — seed=1229
  generate_sequences.py
  parse_results.py
  test_browser.mjs         — Playwright end-to-end tests (true timings)
  index-continuous.html
  index-binary.html
  index-dev.html
  package.json
  vite.config.js
```

### Key parameters

```js
// src/{task}/config.js
const N_TRIALS_TO_RUN        = 40;
const N_OBS_TO_RUN           = 15;
const SHOW_SLIDER_VALUE      = true;
const SLIDER_DEFAULT         = 'none';
const BTI_MS                 = 5000;
const ITI_SHORT_MS           = 1000;
const T_OBS_MS               = 7000;
const SHOW_TRIAL_PERFORMANCE = true;
const MAX_TIMEOUTS_PER_TRIAL = 3;      // defined in timeline-builder.js
const EARLY_EXIT_CODE        = 'EARLYEXIT'; // TODO: replace before publishing
```

### Commands

```bash
cd task
npm install                    # first time only
npm run dev                    # local dev server; open index-dev.html for dev setup page
npm run build:continuous       # production build → dist-continuous/
npm run build:binary           # production build → dist-binary/
node test_browser.mjs          # full Playwright E2E tests (~3 min, true timings)
```

### Testing

Two complementary test systems:

**`test_browser.mjs`** — spins up a static HTTP server, loads the full built app
in headless Chromium via Playwright, and automates consent → tutorial → main
experiment. Tests normal submit, timeout replay, "1 timeout remaining" text,
3-timeout termination screen, and multi-timeout-then-submit. Uses true timings
(7s obs clock) so takes ~3 minutes.

Run both before any deployment.

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

**Pre-deployment checklist:**
- Set `TEST_MODE = false` in both configs
- Fill IRB Protocol Number in consent form (`[Protocol Number]` in `timeline-builder.js`)
- Replace `EARLY_EXIT_CODE = 'EARLYEXIT'` with real Prolific partial-payment code
- Obtain binary task completion code from Prolific (continuous: `C3W3TF1O`)
- Fund Prolific wallet; confirm payment rate with PI

```bash
npm run build:continuous && npm run build:binary
python task/generate_jzip.py   # generates evidence-integration-{task}.jzip
```

Import each `.jzip` into MindProbe: Studies → **+** → **Import Study**.

### Prolific rollout plan

- Publish both studies **simultaneously**; no inter-study screening filter
- **Free task ordering** — natural counterbalancing via Prolific dashboard
- Set both to **auto-approve** so second task appears immediately after first
- End screen: *"one half of a two-part study — look for the other on your dashboard"*
- **Payment:** ≥$12/hr; set estimated time conservatively
- **Non-completions:** request return (not rejection) — slot reopens
- **Partial compensation:** participants who reach 3 timeouts in one trial receive
  partial payment via the `EARLY_EXIT_CODE` Prolific completion path
- **Academic discount:** use Dartmouth institutional email for 33.3% platform fee discount
- **Device restriction:** desktop-only in Prolific study settings

### Data format

`parse_results.py` filters to `screen='observation'` rows and outputs:

| Column | Type | Description |
|--------|------|-------------|
| `prolific_pid` | str | Prolific participant ID |
| `task` | str | `'continuous'` or `'binary'` |
| `trial` | int | 0-indexed trial number |
| `observation` | int | 0-indexed observation within trial |
| `value` | int | Stimulus value |
| `true_mean` | float | Generative mean (continuous); NaN for binary |
| `true_p` | float | True Bernoulli probability (binary); NaN for continuous |
| `qid` | int | Unique sequence ID |
| `iti_ms` | int | ITI for this trial (always 1000ms) |
| `response` | float | Participant estimate (NaN if timed out) |
| `timed_out` | bool | True if response deadline elapsed |
| `rt` | float | Response time in ms (NaN if timed out) |
| `time_elapsed` | int | ms since experiment start |

Both tasks share the same output file — use `df[df.task == 'continuous']` to split.
