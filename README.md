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
| task-continuous | TBD | Continuous inputs; 15 obs/trial; Normal(mean, std); 8x4=32 trials (hybrid method, per-participant pool of 200) | **Piloting** |
| task-binary | TBD | Binary inputs (blue/red); 15 obs/trial; Bernoulli(p); 8x4=32 trials (hybrid method, per-participant pool of 200) | **Piloting** |

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
- **Continuous task**: Normal(mean, std) stimulus; slider response [0–100]; 8x4=32 trials × 15 obs
- **Binary task**: Bernoulli(p) stimulus (blue/red circle); slider response [0–100%]; 8x4=32 trials × 15 obs

Each participant is assigned ONE of 200 independently-generated sequence sets
per task (a per-participant pool, not one shared file) -- see "Design" below
and CLAUDE.md's "Per-participant sequence pool" section for the full
mechanism.

Both tasks share all infrastructure (jsPsych 8, Vite 6, shared plugins/CSS).
Data pipeline: JATOS JSON → `task/parse_results.py` → `data/task_results.pkl`.
Target: ~50–80 participants per task, within-subject.

### Design

- **Sequences**: hybrid method -- binary via quota/momentmatch construction
  (no seed search), continuous via an unrescaled i.i.d.-suffix construction
  (no seed search either); 8 distinct prefixes × 4 repeats = 32 trials;
  prefix_length=4; std_fixed=15 (continuous). Each participant gets ONE of
  200 independently-generated sequence sets per task, assigned via a
  deterministic hash of their participant ID (same index for both tasks) --
  not one shared file. See CLAUDE.md's "Per-participant sequence pool" and
  "Sequence design" sections for the full mechanism and rationale.
- **ITI**: all trials use 1000ms (ITI manipulation removed — no effect observed in pilot)
- **BTI**: 3s between-trial reset screen ("Trial X / N — generating new sequence…") —
  same wording for both tasks (continuous used to say "generating new distribution…";
  "sequence" is the more defensible word since the underlying hidden parameter can
  repeat across trials — 6 unique parameter sets × 4 repeats — so "new
  distribution" would sometimes overclaim novelty)
- **Distractor**: `iti_condition` per trial ('control'/'distract', 2-of-4 per qid);
  `DISTRACTOR_TYPE` in config: 'none' | 'iti_length' | 'popup' (default: 'none')
- **Timeout**: 7s response deadline per observation
  - Per-trial timeout budget: 3 timeouts before session terminates
  - Timeout flow: "Too slow" screen (fade in/out/in, 3.2s) → replay ITI → same observation
  - On 3rd timeout: "Too slow / 0 timeouts remaining" → "Session terminated" screen with button
- **Welcome screen**: title/branding page shown first, before consent — "Evidence
  Integration", "Numbers" (continuous) or "Colors" (binary), inside a bordered box,
  matching the study names given on Prolific; "Begin" button leads to consent.
- **Tutorial**: box 1 (text) → image box (separate click-to-reveal step, showing a
  bubbling generative animation — binary: bubbles rise inside the blue/red bar;
  continuous: bubbles fall from under the Gaussian curve to the x-axis, weighted by
  density) → box 2 (goal text, alongside a yellow tutorial-only-visualization warning
  box) → box 3 (slider instructions) → slider, across 5 tutorial observations →
  tutorial summary → timeout demonstration (3 screens, same fade-in as real
  observations) → BTI → trial 1. The main obs circle/number, and the tutorial's own
  observation marker, fade in (1000ms) rather than appearing instantly, for a
  consistent feel between tutorial and real trials. Tutorial's illustrative sequence
  is derived from a real trial in the sequences data (config-base.js's
  pickTutorialExample), not hand-picked, so it can't drift out of sync with the
  actual generation parameters.
- **Summary slides**: binary — per-obs bar chart (blue/red split at estimate, obs circle left);
  continuous — per-obs number line (red obs thumb, black circle at estimate)
- **Consent form**: verbatim IRB-approved text from task/consent_form.txt, followed by
  3 boxes with ordered disclosure (each stays locked until the one before it is
  revealed): a blue payment-motivation box first ("You will be paid $8.00 -
  10.00 based on your performance"), then 2 red warning boxes (data-loss /
  response-deadline). The proceed button doesn't use the native `disabled`
  attribute (disabled buttons never dispatch `click`, so a premature click got
  silently swallowed with zero feedback) — a capturing-phase click listener
  gates it instead.

### Sequence generation

Three scripts, each with a ROLE note at the top of its own module docstring
(see CLAUDE.md's "Sequence generation methods" section for the full
rationale and tradeoffs):

```bash
# Pure i.i.d. (no smoothing, no seed search — single draw) -- one of two
# candidates still under PI consideration for the 10x4 design, not current
# production:
python task/generate_sequences_iid.py --task both --seed 0 \
    --n_unique_sequences 10 --n_repeats 4 --mean_range 20 80 --std_fixed 15 --p_range 0.2 0.8

# Moment-matched / quota (isotonic-residual seed search, default score_mode) --
# this is what generated the CURRENT PRODUCTION 6x4 pilot. Prefix identity
# and target level are independent axes (see CLAUDE.md for the full
# mechanism and the collision bug this fixes):
python task/generate_sequences_momentmatch.py --task both --n_tries 1000 \
    --n_prefix 6 --n_repeats 4 --rl_alpha_0 1.0 --rl_lambda 0.5
# Defaults used above (all overridable): --n_prefix 6, --mean_range 15 85
# (continuous), --blue_range 2 13 (binary, blue-ball count out of
# --seq_length -- NOT a p fraction).

# generate_sequences.py has no CLI of its own anymore -- shared-utilities
# module only, imported by both scripts above, not run directly.
```

**Current production** (8x4 hybrid, per-participant pool): binary uses
unchanged quota/momentmatch construction (prefix/target independence,
optimal matching, exact-quota suffix) but with NO seed search; continuous
uses the same prefix/target construction but a genuinely unrescaled
i.i.d.-suffix (also no seed search) -- see CLAUDE.md's "Sequence generation
methods" and "Per-participant sequence pool" sections for the full
rationale and the empirical findings behind this per-task split. 8 distinct
prefixes, 4 repeats each = 32 trials PER POOL MEMBER; 200 independent pool
members per task (task/generate_sequences_pool.py), each participant
assigned one via a deterministic hash of their ID. Continuous mean_range=
[15,85]; binary blue_range=[2,13] out of 15. Verified across the whole
pool (not just one member): 200/200 members pass prefix uniqueness both
tasks, zero binary quota mismatches. scripts/inspect_iid_sequences.py
--sequence_type pool and scripts/inspect_sequences.py --pool_dir both
support inspecting the real pool directly -- see CLAUDE.md for usage.

The single reference file (task/sequences/{continuous,binary}_sequences.
{pkl,json}) remains the promotion/verification target when changing
generation parameters, but is NOT what real participants are served --
see "Per-participant sequence pool" in CLAUDE.md.

**10x4 full experiment**: NOT yet finalized. The previously-found candidate
seeds (moment-matched, isotonic score_mode, mean_range=[20,80],
p_range=[0.2,0.8], std_fixed=15: continuous seed=245, binary seed=68) now
predate BOTH the evenly-spaced/no-mirroring redesign AND the prefix/target-
independence redesign above -- they'd need regenerating from scratch under
the current script to be current, not just re-checked. Choice between the
i.i.d. and moment-matched branches, AND what --n_prefix/range to use at
this larger scale, are both pending PI consultation (moment-matching
introduces a real, literature-documented behavioral tradeoff — see
CLAUDE.md — it is not a free smoothness win).

**See `docs/sequence_design_open_questions.md`** for the full write-up of
this tradeoff: a real, quantified behavioral confound from quota
construction (not simple gambler's-fallacy reasoning — the actual
mechanism is subtler and confirmed with real simulation numbers), a
three-way design trilemma between statistical cleanliness, target
diversity, and the repeat-based reliability metrics (T5/T6/V-group) this
project's scientific goals depend on, and what it would actually take to
serve genuinely unique sequences to every participant if that path is
ever chosen. Nothing there blocks the current 6x4 pilot or the promoted
10x4 production sequences from shipping — it's there so the trade-off is
fully informed rather than re-discovered from scratch later.

### Directory structure

```
task/
  src/
    shared/
      timeline-builder.js          — orchestrator: builds full timeline from config
      build-trial-timeline.js      — pure-JS trial loop (importable by test harness)
      build-tutorial-timeline.js   — tutorial sub-timeline (intro → obs → summary → timeout demo)
      build-welcome-screen.js      — title/branding screen shown first, before consent
      build-consent-screen.js      — informed-consent screen
      build-end-screen.js          — final "Thank you" screen
      create-early-exit.js         — session-terminated flow (3-timeout exhaustion)
      finish-session.js             — shared "how does a session actually end"
                                      implementation (single call, data passed
                                      as the argument; Prolific redirects
                                      externally, everyone else gets a DOM
                                      update + no redirect at all -- see
                                      "Exit/redirect and data-saving
                                      architecture" in CLAUDE.md for why a
                                      same-origin redirect after ending the
                                      session failed on a real MindProbe pilot)
      config-base.js               — shared config factory for continuous/binary configs
      jatos-shim.js                — dev no-op shim
      plugin-inter-trial.js        — BTI reset screen
      plugin-observation-continuous.js — continuous obs (number + slider + timeout clock)
      plugin-observation-binary.js     — binary obs (circle + gradient slider + timeout clock)
      observation-timeout-clock.js — shared countdown-clock renderer (used by both
                                      observation plugins AND plugin-timeout-demo.js)
      plugin-iti-clock.js          — ITI countdown clock; shows "Too slow" if timed_out=true
      plugin-timeout-demo.js       — 3-screen timeout explanation (tutorial)
      plugin-tutorial-intro-continuous.js       — continuous 3-stage interactive intro
      plugin-tutorial-intro-binary.js           — binary 3-stage interactive intro
      plugin-tutorial-observation-continuous.js — continuous tutorial obs 2–5 (no timeout clock)
      plugin-tutorial-observation-binary.js     — binary tutorial obs 2–5 (no timeout clock)
      plugin-tutorial-summary-continuous.js     — continuous tutorial summary
      plugin-tutorial-summary-binary.js         — binary tutorial summary
      plugin-trial-summary-continuous.js        — continuous trial summary
      plugin-trial-summary-binary.js            — binary trial summary
      distribution-continuous.js   — shared continuous distribution SVG (revealed flag)
      draw-performance-continuous.js — SVG distribution + dot plots (continuous)
      draw-performance-binary.js     — SVG bar + dot plots (binary)
      urn-binary.js                 — shared binary urn SVG (revealed flag)
      continuous-draw-animation.js  — tutorial bubbling animation (continuous):
                                      bubbles fall from under the curve to the x-axis
      binary-draw-animation.js      — tutorial bubbling animation (binary):
                                      bubbles rise inside the blue/red bar
      slider-continuous.js         — continuous slider
      slider-binary.js             — binary slider (blue/red gradient after first
                                      interaction; 2-row axis ruler below)
      style.css                    — all shared styles
    continuous/
      config.js
    binary/
      config.js
    experiment-continuous.js
    experiment-binary.js
    test-harness.js         — test-ONLY entry (index-test.html); never bundled into
                              any production build, never linked from production code
  sequences/
    continuous_sequences.{pkl,json}   — single reference/promotion-target copy
                                        (NOT what real participants are served
                                        -- see sequences_pool/ below)
    binary_sequences.{pkl,json}       — same, binary side
    continuous_momentmatch_sequences.{pkl,json} — currently an IDENTICAL COPY of
                                        the production file above (this is the
                                        generation script's own output location,
                                        always overwritten by whatever was most
                                        recently searched here -- the OLD 10x4
                                        candidate that used to live at this same
                                        path (seed=245, mean_range=[20,80]) was
                                        overwritten by the 6x4 pilot search and
                                        is no longer in the working tree, but IS
                                        still recoverable via git (commit 274b598
                                        -- confirmed by directly checking out and
                                        inspecting that version: 10 qids, 40
                                        trials, means spanning ~22-78)
    binary_momentmatch_sequences.{pkl,json}     — same situation, binary side
                                        (old 10x4 candidate seed=68 likewise
                                        overwritten in the working tree, same
                                        git commit for recovery)
    continuous_iid_sequences.{pkl,json}         — 10x4 candidate (i.i.d. branch)
    binary_iid_sequences.{pkl,json}             — 10x4 candidate (i.i.d. branch)
  sequences_pool/                     — the REAL per-participant pool served to
                                        real participants -- 200 members/task,
                                        {task}_{0000..0199}_sequences.{pkl,json},
                                        generated by generate_sequences_pool.py.
                                        Gitignored (800 small files, fully
                                        reproducible via that script's own CLI
                                        defaults) -- see CLAUDE.md's
                                        "Per-participant sequence pool" section.
  generate_sequences.py             — original: k-constrained rejection sampling
  generate_sequences_iid.py         — pure i.i.d., no smoothing, no seed search
  generate_sequences_momentmatch.py — quota/moment-matching, isotonic seed search
  generate_sequences_hybrid.py      — CURRENT PRODUCTION method (per-task split)
  generate_sequences_pool.py        — wraps generate_sequences_hybrid.py, writes
                                        the 200-member pool above
  parse_results.py
  test_browser.mjs         — Playwright E2E tests (Chromium/Firefox/WebKit, both tasks)
  index-continuous.html
  index-binary.html
  index-test.html           — test-ONLY entry point, drives test-harness.js; not
                              a build input, real participants can't reach it
  package.json
  vite.config.js
```

Naming convention: every file/class with a continuous/binary pair uses an
explicit `-continuous`/`-binary` suffix on both sides — never leave one side
as an implicit unsuffixed default. See CLAUDE.md for the fuller rationale.

### Key parameters

```js
// src/shared/config-base.js DEFAULTS (shared by both task configs)
const N_OBS_TO_RUN           = 15;
const SHOW_SLIDER_VALUE      = true;
const SLIDER_DEFAULT         = 'none';
const DEFAULT_VALUE          = 50;
const BTI_MS                 = 3000;
const ITI_SHORT_MS           = 1000;   // tutorial between-observation ITI
const T_OBS_MS                = 7000;
const SHOW_TRIAL_PERFORMANCE = true;
const DISTRACTOR_TYPE        = 'none';
const MAX_TIMEOUTS_PER_TRIAL = 3;      // defined in timeline-builder.js
```

Prolific completion/early-exit codes (4 total: {continuous,binary} ×
{completion,earlyExit}) live in timeline-builder.js's `PROLIFIC_CODES`, not
config-base.js — see "Deploying to MindProbe" below.

Trial count is NOT a config constant anymore — it's fully implicit from
however many trials task/sequences/{task}_sequences.json contains.

### Commands

```bash
cd task
npm install                    # first time only
npm run dev:continuous         # local dev server on :5173, opens index-continuous.html
npm run dev:binary             # local dev server on :5174, opens index-binary.html
                                # (both can run simultaneously, in two terminals)
npm run build:continuous       # production build → dist-continuous/
npm run build:binary           # production build → dist-binary/
node test_browser.mjs          # Playwright E2E tests (~2-3 min); see Testing below
```

### Testing

**`test_browser.mjs`** — spawns the real Vite dev server (not a patched build) and
drives a test-ONLY entry point (`index-test.html` / `src/test-harness.js`) via
Playwright across Chromium, Firefox, and WebKit, for both tasks (30 scenarios
total: normal submit, timeout replay, "N timeouts remaining", 3-timeout
termination screen, submit-then-continue — tutorial included in full for every
scenario). That harness is never linked from production code and never
included in any build (vite.config.js's build inputs are only
index-continuous.html/index-binary.html) — real participants can never reach
it. It calls the exact same `buildAndRun()` production uses; it only adjusts
plain config fields (trial count via array slicing, tObsMs/btiMs/itiMs via
direct assignment) before handing the config to the same production code
path — no override logic lives inside buildAndRun/timeline-builder.js itself.
URL params: `?task=&trials=&tObsMs=&btiMs=&itiMs=`. Screen transitions are
detected via `body[data-screen="..."]` rather than guessed sleep durations.

Takes ~2–3 minutes for the full matrix (longer than before the tutorial was
included) — run it after big `task/` changes or when asked, not after every
small edit.

```bash
node test_browser.mjs                                   # full matrix
node test_browser.mjs --task=binary --browser=chromium  # a subset
```

Firefox/WebKit need their browser binaries installed once:
`npx playwright install firefox webkit` (plus `sudo npx playwright install-deps`
for system libraries if missing).

### Local testing pipeline

```bash
# Terminal 1: local result server
npm run dev:server

# Terminal 2: task (either task works the same way)
npm run dev:continuous
# Complete a task in browser → data saved to task/dev-results/

# Parse results
python task/parse_results.py --input_dir task/dev-results/ \
    --output data/task_results.pkl --verbose
```

### Deploying to MindProbe

**Pre-deployment checklist:** (see CLAUDE.md's own checklist for full detail
and current per-item status)
- [DONE] Sequence pool generated and verified (200/task, hybrid method)
- [DONE] PILOT ONLY name field removed
- [DONE] All 4 Prolific code placeholders filled in timeline-builder.js's `PROLIFIC_CODES`
- [DONE] Prolific wallet funded; payment rate confirmed ($10 completion, $3 early-exit)
- [DONE] Jzips rebuilt against the final per-participant-pool state
- [DONE] Full 6-way browser/task E2E matrix -- 48/48 passing (8/8 each)
- [PENDING] A genuinely full completion run via real Prolific preview (not
  just early-exit) -- the one real-platform path not yet exercised
- [DONE] Incremental per-trial saving, save-then-end-then-redirect gating,
  and the GeneralSingle-only worker-type switch, all confirmed against real
  MindProbe/JATOS via six manual test scenarios this session (hand-edited
  `?PROLIFIC_PID=` params, no real Prolific involved) -- see CLAUDE.md's
  "REAL-TEST FINDINGS" note for what was confirmed and two corrections
  (`jatos.log` isn't visible anywhere in the JATOS UI; GeneralSingle's block
  is keyed on the browser's cookie, not the `PROLIFIC_PID` value).

```bash
npm run build:continuous && npm run build:binary
python task/generate_jzip.py --max-workers 30   # generates evidence-integration-{task}.jzip
```

`--max-workers` sets a hard JATOS-side cap on total GeneralSingle workers
for the batch (a backstop independent of Prolific's own participant-slot
count) -- pass your intended sample size plus a small margin, not omitted
(omitting it leaves the batch unlimited; a warning prints but the build
still proceeds).

Import each `.jzip` into MindProbe: Studies → **+** → **Import Study**. The
batch now only accepts **GeneralSingle** workers (previously all five JATOS
worker types) -- grab the General Single link from MindProbe's Worker &
Batch Manager and use that as the Prolific Study URL (same
`?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID=...&SESSION_ID=...` suffix as
before).

### Prolific rollout plan

- Publish both studies **simultaneously**; no inter-study screening filter
- **Free task ordering** — natural counterbalancing via Prolific dashboard
- Set both to **MANUAL approve** (changed this session from auto-approve) --
  every submission is reviewed before payment, which is also what makes the
  "NO JATOS DATA AT ALL" / "STUCK" cases from `reconcile_prolific_jatos.py`
  (see "Data format" below) actionable rather than already-paid by the time
  they're noticed
- End screen: *"one half of a two-part study — look for the other on your dashboard"*
- **Payment:** $10 for normal completion, $3 for the screen-out/early-exit path
- **Non-completions:** request return (not rejection) — slot reopens; Prolific
  also supports rejecting for "gave no study data" specifically, if caught
  before its 21-day auto-approval window
- **Partial compensation:** participants who reach 3 timeouts in one trial receive
  partial payment via the per-task `earlyExit` code in `PROLIFIC_CODES`
  (timeline-builder.js) -- appears in JATOS results as `progress: 'terminated'`
- **Academic discount:** use Dartmouth institutional email for 33.3% platform fee discount
- **Device restriction:** desktop-only in Prolific study settings

### Data format

`parse_results.py` filters to `screen='observation'` rows and extracts every
column DIRECTLY from the raw export -- no lookup/join against a sequence
file (see CLAUDE.md's "Participant-data columns" section for why this
changed: a join against a single shared file stopped being valid once
different participants can be assigned different pool members):

| Column | Type | Description |
|--------|------|-------------|
| `prolific_pid` | str | Prolific participant ID |
| `task` | str | `'continuous'` or `'binary'` |
| `pool_index` | int | Which of the 200 pool members this participant was assigned |
| `trial` | int | 0-indexed trial number |
| `observation` | int | 0-indexed observation within trial |
| `qid` | int | Which of that pool member's distinct prefixes this trial used |
| `value` | int | Stimulus value |
| `true_mean` | float | Continuous only; NaN for binary rows |
| `true_std` | float | Continuous only; NaN for binary rows |
| `true_p` | float | Binary only; NaN for continuous rows |
| `response` | float | Participant estimate (NaN if timed out) |
| `timed_out` | bool | True if response deadline elapsed |
| `rt` | float | Response time in ms (NaN if timed out) |
| `time_elapsed` | int | ms since experiment start |

Older (pre-pool) export files remain parseable -- `pool_index` comes back
missing/NaN with a printed warning rather than crashing.

Both tasks share the same output file — use `df[df.task == 'continuous']` to split.

**Every row now also carries a `progress` field** (e.g. `"welcome"`,
`"tutorial 2/4"`, `"trial 7/24"`, `"finished"`, `"terminated"`) and has
`stimulus`/`button_html` (rendered HTML/CSS) stripped -- see CLAUDE.md's
"CURRENT ARCHITECTURE" note. `parse_results.py` itself is unaffected (still
filters to `screen='observation'`), but the raw JATOS export is now
scannable by eye without downloading anything.

**Reconciling Prolific vs. JATOS**: `task/reconcile_prolific_jatos.py`
cross-references a Prolific submissions-export CSV against a JATOS results
export and flags every participant as OK / TERMINATED / STUCK / NO JATOS
DATA AT ALL / pilot-ignore, for manual review under the manual-approve
workflow above:

```bash
python task/reconcile_prolific_jatos.py \
    --jatos_dir <path_to_jatos_export> \
    --prolific_csv <path_to_prolific_export.csv> \
    --output reconciliation_report.csv
```

Verified against synthetic fixtures only so far -- not yet run against a
real Prolific export (its column-name detection is a best guess at
Prolific's current CSV headers; override with `--prolific-id-col` etc. if
it picks the wrong one).
