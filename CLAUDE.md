# CLAUDE.md — evidence_integration

This file is the source of truth for Claude when working on this project.
Read it fully before making any changes or suggestions. Prefer this file over
README.md when they conflict. For the full design history and rationale
behind decisions inherited from earlier systems (the retired `task/`
online-task pipeline, the sequence-generation-method debate, etc.), see
`docs/HISTORY.md` -- this file covers current state only.

**After any conversation compaction**: re-read this file in full before doing
anything else. Compaction summaries omit conventions. Key ones to remember:
- **Tool routing (critical)**: `str_replace`, `create_file`, and `view` operate
  on Claude's own local sandbox container, NOT this remote host. Only
  `filesystem:read_text_file` / `filesystem:write_file` / `filesystem:edit_file`
  and `shell:run_command` touch actual project files here. Using the wrong
  tool silently "succeeds" (writes to a local copy that looks identical) with
  no error — this caused a real bug where several edits to
  generate_sequences_momentmatch.py appeared to work but never reached the
  remote file, and the mismatch wasn't caught until a CLI flag failed at
  runtime. Always use the filesystem/shell MCP tools for anything under
  /home/psipeter/evidence_integration/; if unsure whether an edit landed,
  grep the remote file for the new content before trusting it.
- Figures save as PDF only — never convert to PNG/SVG or upload images to chat.
  This applies to task_backend/task UI screenshots too: don't upload Playwright
  screenshots to context to verify UI work — use DOM/computed-style assertions
  instead (textContent, getComputedStyle, attribute checks) and only pull an
  actual image when a genuine visual judgment call is needed.
- **Testing — the person now runs tests themselves.** Tell them what you'd
  like to run and give the exact command(s); don't run tests directly unless
  explicitly asked to. This applies to task_backend's own suite
  (`cd task_backend && npx playwright test`, 4 spec files, ~4-5 min against
  the real deployed backend) and to anything in `task/` (now retired, but
  its own `node test_browser.mjs` suite still exists on disk).
- **A single full-suite `npx playwright test` call via shell:run_command can
  exceed the tool's own response window** — confirmed directly this session
  (a `timeout 300 npx playwright test` call in task_backend returned no
  result after 4 minutes even though the suite was running fine remotely,
  the same failure class task/'s old `test_browser.mjs` had). If Claude ever
  does run a test itself (only when asked), use a background+poll pattern
  (`setsid nohup ... > logfile &`, then poll with `sleep`/`tail`) rather than
  one blocking call, and check `lsof -ti:5183 -ti:5184` are empty before
  starting (stale servers from a prior timed-out attempt can squat the test
  ports). task/'s own `test_browser.mjs` has an analogous, more severe
  version of this (6 separate calls required, one per browser × task
  combination — see docs/HISTORY.md for the full mechanics if task/ is ever
  touched again).
- All NEF simulation data → data/runs/; figures → figures/

---

## Scientific goals

This project studies **how people integrate sequential noisy evidence**, using
cognitive models and a biophysical spiking neural network (NEF) to identify the
computational and neural mechanisms underlying that process.

### Goal 1 — Cross-task generalisation of cognitive mechanisms
The NEF model must capture human behaviour across multiple tasks (carrabin and
yoo), demonstrating that the same underlying cognitive algorithm generalises
across task domains without task-specific modification. The NEF is benchmarked
against a spectrum of established models from the evidence-integration
literature: an optimal Bayesian integrator (Mean), a leaky integrator
(LeakyIntegrator), and primacy/recency weighting models (PrimacyRecency). On
trial-wise RMSE the expected ordering is:

    task-specific model ≈ NEF > LeakyIntegrator ≥ PrimacyRecency ≥ Mean (optimal)

The NEF need not outperform task-specific models; comparable RMSE combined with
cross-task generalisability is the target.

### Goal 2 — Emergent higher-order behavioural signatures
Beyond RMSE, the NEF must reproduce secondary behavioural phenomena that it was
not explicitly trained to capture. These signatures should emerge naturally from
the model's spiking dynamics: temporal update patterns, the decay of response
change across the observation sequence, individual differences in discounting
rate (λ), test-retest reliability of noise and decay-rate metrics, and
state-persistent response variability. The fact that these emerge without being
directly optimised is the key scientific contribution.

### Goal 3 — Joint behavioural and neural predictions
The NEF generates both behavioural and neural predictions simultaneously from
the same underlying mechanism. Behavioural: response trajectories, update
magnitudes, individual λ and α₀. Neural: error-ensemble activity, prediction-
error dynamics, and how both scale with architectural parameters (n_neurons,
α₀, λ). Together these constitute a mechanistically coherent account that is
testable at multiple levels of analysis.

The current active plan for making this argument concretely, as one
consolidated figure (the "neural giant" -- see `## Neural predictions figure
(Acts 1-5)` below for the full motivation/structure), reuses the SAME
underlying error population across all of its panels rather than the older
per-task carrabin/yoo neural figures (N1-N8 below), which each only ever
showed part of the story.

### Goal 4 — Novel testable predictions
Spiking noise produces state-persistent variability that differs qualitatively
from response noise; this prediction distinguishes the NEF from NoisyCounting
even when both achieve similar RMSE. Response variability and PE variability
scale with n_neurons and α₀; neural activity profiles match known ensemble
dynamics. These are quantitative predictions for future empirical work.

---

## Metric taxonomy (PTN framework)

All analyses and figure panels are organised under three groups.
Figures save PDF only (no PNG/SVG).

### P — Performance
Metrics measuring how well participants and models do on the task.

| Code | Metric | Carrabin | Yoo |
|------|--------|----------|-----|
| P1 | Estimation error: RMSE to hidden probability / true mean, per pid boxplot | Y | Y |
| P2 | Model fit: RMSE to human responses, per pid boxplot | Y | Y |

Carrabin: true_p converted as true_p*2-1 to match response scale [-1,1].
Yoo: true_mean = cumulative mean of value stream (expanding mean per trial).

### T — Temporal
Metrics capturing within-sequence dynamics.

| Code | Metric | Carrabin | Yoo |
|------|--------|----------|-----|
| T1 | Task performance vs observation: RMSE per obs position | Y | Y |
| T2 | Response change vs observation: mean |Δresponse| per obs | Y | Y |
| T3 | Split-half reliability of λ: first vs second half of trials | N | Y |
| T4 | Dynamical model fit: λ_model vs λ_human regplot (individual differences) | N | Y |
| T5 | Residual variance growth across obs (state noise accumulation) | Y | N |
| T6 | Within-trial residual autocorrelation decay (state persistence) | Y | N |

T3–T4 require enough trials and observations to fit λ reliably.
λ is fit via curve_fit: A·n^(-λ), bounds [0,2], on mean |Δresponse| curve.
T5–T6 use residuals = response − mean(response | pid, obs, qid); require qid.

### N — Neural
NEF predictions; testable in future empirical experiments.

| Code | Metric | Carrabin | Yoo |
|------|--------|----------|-----|
| N1 | Decoded PE timecourse within observation window | Y | Y |
| N2 | PE variability vs response variability, probe sims; partial-r control for α₀ | Y | N |
| N3 | Response and PE variability vs fitted α₀ (shared scaling factor) | Y | N |
| N4 | Response and PE variability vs n_neurons scan | Y | N |
| N5 | Error-ensemble weight-neuron activity vs observation, split by λ group | N | Y |
| N6 | Mean weight-neuron activity vs mean |Δresponse| across observations | N | Y |
| N7 | Fitted λ mediates activity change and mean |Δresponse| (twin-axis) | N | Y |
| N8 | Late |Δresponse| vs late estimation error (last 10 obs) | N | Y |

N1-N8 above is the OLDER per-task taxonomy (figure_carrabin_neural.py/
figure_yoo_neural.py, each only covering part of the story on its own task).
The CURRENT active plan consolidates this into one figure -- see below.

---

## Neural predictions figure (Acts 1-5)

### Motivation
The theory's central mechanism is a WEIGHTED prediction error (PE) updating
an internal estimate. The hypothesis this figure exists to support: many of
the behavioural phenomena already shown in the P/T figures are driven by the
dynamics of the neural population that represents this PE. We test this by
looking at the simulated error population inside the NEF model itself,
hypothesised to resemble a real neural population somewhere in PFC (or
possibly striatum/VTA) -- i.e. every claim in this figure is framed as a
concrete, testable prediction for a real neuroimaging experiment, not just a
model-internal description.

### Structure -- 5 acts, building in strength of claim

1. **Toy/illustrative population dynamics** (no fitting, no behavioural
   data -- pure model mechanism, arbitrary parameter values). Shows: a
   single-trial spike raster of the error population with decoded PE
   overlaid; error-neuron activity vs observation-within-trial across a few
   λ values; decoded PE vs time-within-observation across an α₀ x n_neurons
   grid. Claims: α₀ controls the upswing of decoded PE; n_neurons controls
   its noise level; λ controls the rate at which error neurons become
   quiescent (stop responding to new input). These are the measurable-in-
   the-lab quantities (neural activity, decoded representations) with no
   model-fitting involved.
2. **Behaviour <-> PE representation, both axes measurable.** Ties Act 1's
   predictions back to σ and λ: σ (response variability) vs PE variability;
   ΔR(early-late) vs ΔA(early-late) [response-change decay vs activity
   decay]. Both axes on each panel are things a real neuroimaging study
   could measure directly -- no parametric model comparison needed -- so
   each panel is its own standalone empirical prediction.
3. **Both X and Y jointly controlled by the same underlying parameter.**
   σ AND PE variability vs fitted α₀ (both depend on the same scaling
   factor); fitted λ vs ΔR-decay AND ΔA-decay (twin axes). Same
   underlying data as Act 2, replotted against the parameter that drives
   both measured quantities together.
4. **Validation via ablation/statistical control** (NOT YET BUILT). For
   each Act 2/3 relationship: a partial correlation controlling for the
   other parameter, and, where feasible, a mechanistic ablation (forcing a
   parameter to a null value and showing the correlation collapses) --
   matching yoo's own existing λ=0 ablation precedent.
5. **Optional -- synaptic vs working-memory implementation comparison**
   (NOT STARTED, separate downstream scope). Different predictions under
   an ITI manipulation, depending on which implementation of the learning
   rule is assumed. Deliberately out of scope for now.

### Task choice: soltani_numbers throughout
Unlike the old N1-N8 table (split across carrabin/yoo, each missing half the
metrics), Acts 1-3 all run on `soltani_numbers` specifically -- the ONE task
with both a real fitted σ and a real fitted λ (carrabin has σ but not λ;
yoo has λ but not σ; both soltani tasks have both, numbers picked
arbitrarily over colors). This lets one task carry the whole argument
rather than splitting it across two tasks that each only show half.

### Implementation
- `scripts/neural_experiments.py` -- NEW script, generalises extras_
  carrabin.py's pattern (param-grid sweeps, probe simulations) to an
  arbitrary `--task`, since none of it is actually carrabin-specific under
  the hood. Three experiments: `raster_demo` (Act 1, one trial, arbitrary
  params, full per-timestep error-population output for a spike raster),
  `sweep` (Act 1, one OR two swept parameters -- a cross product if two --
  full per-timestep resolution), `probe` (Act 2/3's expensive half -- full
  per-timestep simulation at a pid's own fitted params across their real
  trials; has a `--mode run/submit/collect` lifecycle since this is
  cluster-bound). Output: `data/runs/neural_experiments/`.
- `scripts/make_paper_figures.py`'s `make_neural_giant()` builds the figure
  itself. Currently 1x3 (Act 1's three panels); more rows/panels will be
  added as Act 2/3 data comes in, same incremental approach the
  lambda_giant/sigma_giant combined figures used.

### Status
- **Act 1: DONE.** All 3 panels built, data generated locally (cheap --
  see docs/HISTORY.md for the exact parameter values used).
- **Act 2/3: PLANNED, not yet run.** Two data sources:
  1. Probe simulation -- `neural_experiments.py`'s own `probe` command,
     needs `--mode submit` then `--mode collect` for numbers' 46 pids on
     the cluster. Already writes to its own `data/runs/neural_experiments/`
     folder, not rmse/nll.
  2. Per-observation ensemble activities/encoders -- NOT something
     `neural_experiments.py` needs to implement; `utils/save_activities.py`
     already handles any dataset generically (the same mechanism that
     produced yoo's own `activities_error_yoo.pkl`), invoked via
     `fitting.submit --resubmit activities`. That mechanism DID need one
     small generalization this session: it used to read fitted params AND
     write activity/encoder output to the SAME `--run_folder`, which would
     have written neural output straight into `data/runs/rmse/`, mixing it
     with pure behavioural fitting results. Both `fitting/submit.py` and
     `utils/save_activities.py` now take an optional `--out_folder`
     (defaults to `--run_folder`, so every existing caller's behavior is
     unchanged) so activities can land in `data/runs/neural_experiments/`
     instead, reserving rmse/nll for behavioural results only.
     `NEF_soltani_numbers_responses.pkl` (needed for the ΔR-decay half)
     already exists from the original RMSE fit.
- **Act 4/5: NOT STARTED.**

## Central cognitive model

Updates follow a power-law decaying learning rate:

    alpha(t) = alpha_0 / t^lambda

High lambda: steep discounting (primacy-like). Low lambda: slow discounting
(recency-like). In the NEF, alpha(t) is an emergent property of the spiking
dynamics rather than a hardcoded equation — a counting subnetwork tracks the
observation index and decodes the appropriate weight, which then gates the
error signal driving the value ensemble. This is analogous to RL_lambda (which
implements the same equation explicitly) but arises from biophysical dynamics.

---

## Active datasets

| Name | N | Task |
|------|---|------|
| carrabin | 21 | Binary inputs; slider after each of 5 obs; sequences repeat (qid); true_p known |
| yoo | 38 | Continuous inputs; slider; 30 obs × 30 trials; no sequence repetition |
| soltani_numbers | live | Our own task_backend `numbers` task; 32 trials × 15 obs |
| soltani_colors | live | Our own task_backend `colors` task; 32 trials × 15 obs |

Pickles: data/carrabin.pkl, data/yoo.pkl, data/soltani_{numbers,colors}[_<datafile>].pkl.
Required columns: pid, trial, observation, value, response.
Carrabin adds: qid, true_p (from carrabin_original.csv).
soltani_* add: qid, plus true_mean (numbers) / true_p (colors).

**soltani naming (renamed; do not reintroduce the old names)**: these two
datasets were `task_continuous`/`task_binary` until they were renamed to
`soltani_numbers`/`soltani_colors` — matching task_backend's own
numbers/colors terminology (see "Terminology" above) and the
`figure_soltani_*` scripts. `continuous`/`binary` is the RETIRED `task/`
pipeline's naming and must not be used for these datasets. Note the one
deliberate exception: `utils/binary_transform.py`'s module name,
`apply_binary_transform`, and `normalise_binary`/`normalise_continuous`
keep their names — that module's "binary" refers to binary-valued
observations generally and it is applied to **carrabin**, not only to
soltani_colors.

**soltani_* are 0-INDEXED on both trial (0-31) and observation (0-14)**,
unlike carrabin/yoo (both 1-indexed). Anything that assumes a 1-indexed
observation (a log(observation) fit, a `t/(t+2)` count, an activity map
keyed 1..n) needs an explicit guard. The known cases are already handled:
`counting_integrator.activity_key_for_trial` for the activity map/seed,
`_fit_lambda_curve_fit`'s `n = observation + 1` for the power-law λ fit, and
`apply_binary_transform`'s `t = observation + 1`.

`scripts/pull_soltani_data.py` flags worth knowing:
- `--complete_pairs` derives the pid set with a `finished` row in BOTH tasks live
  from Supabase, instead of passing `--numbers_pids`/`--colors_pids` by hand. No
  cohort pid lists are recorded in this repo, so this is the only reproducible
  way to re-select the same people.
- `--exclusion_method {contingency,performance,baseline}` selects WHICH criterion
  set decides exclusion, and `--per_task_exclusion` opts out of the default
  subject-level policy. See "Participant exclusion criteria" for all three and
  the measured effect of each.
- `--no_filter` skips `utils/participant_filters` entirely (also
  `build_from_df(apply_filters=False)`). For diagnosing how much the exclusion
  criteria change a result -- NOT for published output; it prints a warning.
  On complete_pairs, 60 participants finished both tasks; `contingency` keeps 27
  (55% excluded), `integration`+require_both keeps 24, `performance` keeps 44,
  `non_integrator` keeps ~24 per task. The
  excluded group roughly doubles both the mean and the SD of performance error
  (numbers 0.107 -> 0.201 mean, 0.074 -> 0.141 SD), and the MEDIAN moves as much
  as the mean, i.e. they are systematically worse rather than a noisy tail.
- Integer pids come from the PERSISTENT registry (`utils/pid_registry.py`),
  keyed on prolific_pid identity alone -- so a filtered and an unfiltered
  build now assign the SAME integer pid to the same real person (this
  used to be false: an earlier from-scratch `sorted(...).unique()` mapping
  depended on who else was in that specific call's batch; see
  utils/pid_registry.py's own docstring for the real bug this fixed --
  growing the pool from 35 to 45 pids silently reassigned most existing
  participants' pids, breaking every model fit's join against current
  human data).

Archived (do not reactivate): diederen, jiang, usher.

---

## Online task: task_backend (current implementation)

Two online experiments -- **numbers** (continuous, Normal(mean,std)
stimulus) and **colors** (binary, Bernoulli(p) blue/red stimulus) --
deployed as a single jsPsych 8 + Vite 6 web app, backed by Supabase
(Postgres + Edge Functions), hosted on GitHub Pages. 8x4=32 trials x 15
observations per task, per-participant pool of 200 independently-
generated sequence sets.

**Why task_backend exists, in one paragraph**: it replaced a JATOS/
MindProbe pipeline (`task/`, now retired -- see "Legacy: task/" below)
after two real Prolific participants hit genuine JATOS-level data-loss
failures during pilot #3, and a follow-up empirical test confirmed a
third, worse gap: per-trial saves could fail silently for an entire
session with zero participant-visible symptom. Gorilla, Cognition.run,
and Labvanced were all evaluated as hosted alternatives and rejected.
**Full incident investigation and platform evaluation, and the entire
build-out/pilot history since: `docs/HISTORY.md`** ("Pilot #3
real-participant incidents...", "Own-backend decision (Supabase)", and
its own "task_backend: build history and settled decisions" section,
folded in from the now-retired `task_backend/TODO.md` once the initial
build-out and first two real pilot rounds settled). This section only
documents what's true right now.

### Terminology

**numbers**/**colors** throughout (file names, directory names, the
`task` column/parameter, class names, CSS classes) -- NOT continuous/
binary, which is `task/`'s own retired naming. The two codebases don't
share terminology. One naming exception worth remembering: the shared
scoring module is `scoring.js` (task-neutral), not `bonus-numbers.js`,
since it's used by both tasks.

### Backend schema (`events` table, Supabase)

One append-only table. Every row is either a checkpoint (tutorial/trial
observation) or a bookkeeping/terminal marker (welcome/consent/finished/
terminated).

| Column | Notes |
|---|---|
| `id` | `bigserial primary key` -- `ORDER BY id DESC LIMIT 1` finds "latest state," never `created_at` (which isn't reliable under retries) |
| `prolific_pid` | real Prolific ID, or `dev_${Date.now()}` fallback locally |
| `task` | `'numbers'` or `'colors'` |
| `pool_index` | deterministic hash of `prolific_pid` (see "Sequences" below) |
| `phase` | `'welcome'|'consent'|'tutorial'|'trial'|'finished'|'terminated'` |
| `trial_index` | 0-31 for `phase='trial'`; `-1` sentinel otherwise (never `null` -- Postgres treats two `NULL`s as distinct, which would silently break the idempotency constraint below) |
| `observation_index` | 0-14 for `tutorial`/`trial`; `-1` sentinel otherwise |
| `attempt` | increments only on a timeout-triggered replay of the same `(trial_index, observation_index)` |
| `response`, `timed_out`, `rt`, `value`, `true_mean`, `true_std`, `true_p`, `qid`, `error`, `reward` | nullable, self-explanatory |

**Idempotency**: `unique (prolific_pid, task, phase, trial_index, observation_index, attempt)`.
**Access model**: RLS enabled with **zero policies** for `anon`/
`authenticated` (deny-all, including read) -- the three Edge Functions
below are the *only* way in or out, using the service-role key
server-side. The browser never talks to the database directly.

### Edge Functions

- **`progress-check`** -- called before building the jsPsych timeline.
  Queries the latest row for `(prolific_pid, task)` and branches:
  `finished`/`terminated` -> returns that status + the participant's
  Prolific code, client skips the timeline entirely; `trial` -> resume at
  the start of the current (or next, if the last observation is already
  logged) trial; `tutorial`/`consent` -> resume at the start of the
  tutorial; `welcome`-only or no rows -> full run from welcome.
- **`progress-append`** -- replaces the old JATOS `appendResultData` call.
  Fire-and-forget is NOT repeated here on purpose: the client tracks
  consecutive failures and surfaces a visible (non-blocking) warning
  banner after 2 in a row, rather than JATOS's confirmed silent-failure
  gap.
- **`progress-finish`** -- sanity-checks the expected number of trial rows
  actually exist before accepting a "finished" claim, then hands back the
  completion code as visible text (not just embedded in a redirect URL --
  a failed redirect can never lose the code).

Resume granularity is **trial-boundary, not exact-observation** --
reloading mid-trial restarts that trial (cheap: 15 observations), not the
exact observation; every observation is still logged individually for
analysis, only the *resume point* is coarser than the *log*.

### Sequences

**One consolidated script**: `task_backend/generate_sequences.py` --
replaced `task/`'s five-script iid/momentmatch/hybrid debate (see
`docs/HISTORY.md` for that now-closed methodological question) with a
single method per task, plus its own `verify_numbers_trials`/
`verify_colors_trials` asserts run at generation time (prefix uniqueness,
exact-quota correctness) -- downstream tools (`scripts/plot_sequences.py`)
deliberately trust these rather than re-auditing.

**`NUMBERS_STD_FIXED = 10`** (current value, reverted from 15 -- see the
constant's own comment for the full 10 -> 15 -> 10 history: the first
change fixed a tutorial-example flat-suffix bug, the second reverted it
after a real pilot at std=15 showed weak |delta response| decay signal,
testing whether std=15 was simply too noisy a task). If a real trial's
achieved std still misses the tolerance band even after
`build_numbers_suffix`'s own retries, `generate_numbers_trials` now
regenerates that qid's WHOLE PREFIX from scratch and rebuilds all 4
repeats against their already-assigned targets (up to 30 attempts) --
this replaced an earlier, more complex two-mechanism repair (a pairwise
target-swap step immediately after Hungarian matching, PLUS this same
prefix regeneration), removed once confirmed empirically redundant
(6/6400 vs 5/6400 outliers on a real 200-member pool testing prefix
regeneration alone). One direct question asked repeatedly (does the
achieved std land in tolerance?) rather than two separate repair
strategies reasoning about the problem from different ends.

**Fixed tutorial sequences**: `choose_tutorial_sequences` (also in
`generate_sequences.py`, run via `--tutorial`) selects ONE trial per task
from the real production pool to serve as every participant's tutorial
example -- the same trial for everyone, not derived dynamically per-load.
Numbers: a two-stage selection (percentile-band on early prefix-response
variability, then best suffix-response variability within that band) plus
a hard filter excluding any candidate with a repeated raw value anywhere
in its 15 observations. Colors: since colors' own literal `qid` never
repeats at all (every trial gets its own, by design -- confirmed
empirically against real data), an empirically-derived "quasi-qid"
repeat structure (`utils/colors_quasi_qids.py`) is reused for the SAME
selection logic.
Written to `tutorial_sequence_{numbers,colors}.json` at the repo root,
imported directly by each task's own `config.js` -- superseded an earlier
dynamic `pickTutorialExample()` (removed entirely, see git history to
restore).

**Files**: `sequences_numbers.json`/`sequences_colors.json`, each a plain
JSON array of 200 independent pool members (no `.pkl` at all). Each member
is a list of 32 trial dicts: `qid, true_mean, true_std, true_p, values,
prefix_length, iti_ms, iti_condition, trial`. Every real participant is
assigned ONE member via `poolIndexForParticipant` -- a deterministic
DJB2-style hash of their `prolific_pid` -- so there's no single shared
"production" file distinct from the pool at all (unlike `task/`'s old
design). Same hash formula for both tasks, so one participant gets the
same pool index in numbers and colors.

**Test variants**: `generate_sequences.py --name <suffix>` builds a small
variant (e.g. `test2trial`, 2 trials/member) using its OWN separate
constants, gated behind `--name` so it can never accidentally produce a
real-shaped production file. Client-side, `VITE_SEQUENCES_VARIANT`
(env var) selects which file to load via `import.meta.glob` -- unset in
every real build (confirmed: the deploy workflow never sets it, and the
production files are git-tracked while only `_*`-suffixed variants are
gitignored, so a fresh CI checkout can never bundle test data by
accident).

**`iti_condition`/distractor system**: REMOVED entirely (chat history) --
this study has no distractor manipulation, so `config-base.js`'s
`DISTRACTOR_TYPE`, `plugin-iti-clock.js`'s popup-spawning logic, and every
prop/param that only existed to support them (`distractor_type`,
`is_colors` inside that plugin specifically) were all dead weight once
that mode could never actually fire, and were deleted rather than left
inert. `iti_condition` (`'control'`/`'distract'`) is still generated into
every sequence (`generate_sequences.py`'s own balanced-repeat design is
unchanged, untouched by this) but is no longer consumed anywhere on the
client at all -- it's inert DATA now, not inert CODE.

### Scoring

`scoring.js` (shared by both tasks): `normError = rawError / MAX_POSSIBLE_ERROR;
reward = max(0, MAX_REWARD * (1 - bonusDecay * normError))`, computed per
observation and summed for the trial/tutorial total. Current parameters:
`MAX_REWARD = 2` cents, `MAX_POSSIBLE_ERROR = 100`, and -- split per task
(chat history: numbers' std_fixed changed 15 -> 10, which made a single
shared decay value give inflated rewards on numbers relative to colors
for the same relative precision) -- `NUMBERS_BONUS_DECAY = 25`,
`COLORS_BONUS_DECAY = 15` (colors unchanged). `bonusDecay` is a REQUIRED
parameter to `computeResponseReward`/`computeTrialReward` (no default),
so a call site can never silently fall back to the wrong task's value.
`ERROR_MODE` (config-base.js) is `'running_mean'` for numbers / `'running_p'`
for colors -- scores against the running statistic of observed values, not
the fixed generative parameter (a deliberate methodological choice, not a
placeholder -- see `docs/HISTORY.md` for the full running-mean-vs-true-
mean discussion).

### Tutorial

A "Correct answer" panel (`correct-answer-numbers.js`/`correct-answer-
colors.js`) replaces an earlier KDE-curve/urn-bar design (motivated by a
pilot #3 comprehension finding -- see `docs/HISTORY.md`): numbers shows a
thumb sliding to the running mean's position on a 0-100 track with a tick
per observation; colors shows a blue/red bar split at the running blue
proportion, with dots accumulating above it. Both are plain HTML/CSS, no
bubbling animation or artificial delay. Intro plugin uses a 3-click
progressive reveal (box0 -> box1+correct-answer-panel -> box2+slider);
tutorial observations use a 5-phase (A-E) top-right hint system keyed on
observation number.

### Client code conventions (verified consistent, not just assumed)

- **`build-*.js`** -- plain builder functions, no jsPsych-plugin shape.
- **`plugin-*.js`** -- real jsPsych plugins (`info` + `trial()`).
- **`create-*.js`** -- hand-rolled, non-jsPsych-trial DOM/orchestration
  code (e.g. `create-terminate-session.js`), matching the same pattern
  `task/`'s retired `create-early-exit.js` used.
- **jsPsych 8 plugin conventions**: `trial(display_el, trial, on_load)`
  must NEVER be declared `async` (jsPsych 8.2.3 advances the timeline on
  Promise resolution, not `finishTrial()`, causing overlapping trial
  instances if declared async -- a real, previously-shipped bug in
  `task/`'s own history). Pattern A (no timeout clock: consent/tutorial/
  summary screens) vs Pattern B (has a timeout clock: real observation
  plugins) are the only two shapes. Parameters the app always supplies
  explicitly (`true_mean`, `true_std`, `true_p`) must have NO `default`
  key in `info.parameters` -- omitting `default` makes jsPsych treat it as
  required and fail loudly if ever missing, rather than silently rendering
  wrong data.

### Testing

4 Playwright spec files (`task_backend/tests/`), run against the real
deployed Supabase backend, not mocked:
- **`happy-path.spec.mjs`** -- one canonical full-session traversal per
  task (numbers, colors), two-phase design: a pure UI-level test (screens
  reached, console errors, checkpoint HTTP statuses) followed by a
  database-only test that only runs if the first passed.
- **`resume.spec.mjs`** -- reload mid-session resumes at the correct
  trial boundary.
- **`timeout-retry.spec.mjs`** -- three real observation timeouts in a
  row: `attempt` increments, session terminates on the third.
- **`completion-screens.spec.mjs`** -- all three session-ending paths
  (finish/terminate/returning-participant) show the visible completion
  code.

`playwright.config.mjs` uses dedicated ports (5183/5184, distinct from
this project's own default dev ports, which have historically been
occupied by long-running `task/` dev servers) and a small `test2trial`
sequence variant (generated on demand if missing) so a full session
completes in seconds rather than the ~15-30 minutes a real 32-trial
session would take.

### Hosting / deployment

Single combined `dist/` (both `index-numbers.html`/`index-colors.html`
built together via plain `npm run build`), deployed to GitHub Pages via
`.github/workflows/deploy-task-backend.yml` (path-filtered to
`task_backend/**`, so unrelated commits don't trigger a rebuild). Live at
`https://psipeter.github.io/evidence_integration/index-{numbers,colors}.html`.

**Prolific cutover: DONE.** Two real pilot rounds have run against
task_backend directly (not JATOS) since the state described in an
earlier version of this section: pilot 4 (5 real participants, both
tasks, `NUMBERS_STD_FIXED=15`) and pilot 5 (numbers only, std=10,
ongoing as of the last check -- see `docs/HISTORY.md`'s task_backend
section for the count as of when that history was folded in, or query
Supabase directly via `pull_soltani_data.py --list_candidates`
for the current live count). Both used Prolific's own Study URL field pointed
directly at `https://psipeter.github.io/evidence_integration/
index-{numbers,colors}.html?PROLIFIC_PID={{%PROLIFIC_PID%}}`, exactly
the mechanism this section used to describe as a not-yet-taken step.
`supabase/functions/_shared/prolific-codes.ts` mirroring the old JATOS
completion/early-exit codes (`C1CNSEMJ`/`C1ARJ6LO` numbers,
`C12FEFJU`/`C1L1GGHT` colors) has been confirmed working end-to-end
against real Prolific submissions, not just in tests.

### Data pipeline: Supabase -> analysis (built, explicit-pid-list based)

`scripts/pull_soltani_data.py` pulls real, finished participant
data directly from Supabase's `events` table for an EXPLICIT list of
`prolific_pid`s (not "everyone finished so far" -- different pilot
rounds are different people with different generative parameters, e.g.
numbers' `std_fixed` changing between pilot 4 and pilot 5, so silently
merging them would make cross-pilot comparison impossible). `--pilot
<name> --numbers_pids ... --colors_pids ...` builds that pilot's own
`data/soltani_numbers_<name>.pkl`/`soltani_colors_<name>.pkl` via
`build_model_inputs.py`'s shared `build_from_df()` (the same filter/
rescale/anonymize/save pipeline carrabin/yoo already use -- refactored
out of the old JATOS-pilot-file-only `build()` so both sources share one
implementation). `--list_candidates <task>` probes current real-
participant status (finished/terminated/in-progress, plus `true_std`)
directly from Supabase without building anything, for constructing an
accurate pid list rather than guessing from memory.

`figure_soltani_{performance,temporal,variability}.py` all take a
general `--datafile <name>` argument (a plain filename suffix, not a
pilot-specific concept -- works the same way for a future non-pilot
experiment dataset) pointing at these files; omit it for the canonical
unsuffixed `data/soltani_numbers.pkl`/`soltani_colors.pkl`. Each figure
degrades to an explicit placeholder (not a crash) when a task has no
file for a given datafile (e.g. pilot 5 has no colors data at all).

Currently human-data-only in `figure_soltani_temporal.py`/`figure_soltani_variability.py`'s
own default (`--models` opt-in, same as always) -- but a real fitting pass
DID run against the canonical, corrected 46-pid data since this section was
written: Mean/LeakyIntegrator/PrimacyRecency/RL_lambda under RMSE
(`data/runs/rmse/`) and Mean_resp_noise/LeakyIntegrator_resp_noise/
PrimacyRecency_resp_noise/NoisyRL_lambda under NLL (`data/runs/nll/`) --
see "soltani math-model fits" below for the exact commands, and
`presentations/make_figures.py`'s own model-correlation/sigma figures for
confirmed real results built on these fits. **NEF itself is still not fit
for any dataset as of this write-up** -- a submit is planned for this
weekend (all 4 datasets, large `n_neurons`; see "soltani math-model fits"
below, under "NEF RMSE fits", for exact status/commands/open risks), but
has not run yet, so RL_lambda (RMSE figures) and NoisyRL_lambda (NLL
figures) still stand in for it in the presentation deck's own "our model"
slot for now.

Anonymization: `build_from_df()` maps `prolific_pid` (string) -> a small
sequential int `pid` via the PERSISTENT registry `utils/pid_registry.py`
-- NOT computed fresh per call. An existing participant keeps the SAME
pid forever, across every future pull/pilot/build, no matter how the pool
grows; new prolific_pids only ever get new integers appended after the
current max. The registry file itself (`data/pid_registry.json`) contains
real prolific_pids and must NEVER be committed or pushed through GitHub
-- see that module's own docstring for the full rationale (including the
real bug this replaced) and for how to keep it in sync with the cluster
(copy the one file directly, scp/rsync, never git). The real `prolific_pid`
never appears in the saved canonical pkl.

`data/soltani_numbers.pkl`/`data/soltani_colors.pkl` are now TRACKED in
git (an `!`-override in `.gitignore`, matching carrabin.pkl/yoo.pkl) --
GitHub is the sync channel for these two files specifically between this
machine and the cluster. `data/pid_registry.json` is the one exception:
always gitignored, always moved by hand.

### Participant exclusion criteria (`utils/participant_filters.py`)

THREE CANDIDATE CRITERION SETS, selected by `--exclusion_method`. All three are
always COMPUTED and appear in the report; only the `excluded` column differs, so
a report always carries the diagnostics for the methods you did not pick. None is
settled -- they are candidates being compared.

| method | basis | numbers | colors | notes |
|--------|-------|---------|--------|-------|
| `contingency` (default) | three Cohen's f² tests | 25/60 (42%) | 19/60 (32%) | model-BASED; `recency_only` tests the same construct as `non_integrator` |
| `performance` | carrabin's gross-outlier rule, `--max_error_sd` (default 2.0) | 9/60 (15%) | 9/60 (15%) | model-free, accuracy-based |
| `integration` | did not beat "report the most recent observation", `--min_skill` (default 0.10) | 36/61 (59%) | ~1 | model-free; **discards inaccurate integrators -- see below** |
| **`non_integrator`** | **prior observations make no RELIABLE contribution to predicting the response** | **19/61 (31%)** | **17/61 (28%)** | **model-free, no magnitude threshold; the recommended criterion** |

#### `non_integrator`: the definition-first criterion

> A **non-integrator** is a participant for whom observations BEFORE the most
> recent one make no reliable contribution to predicting their responses.

Stated as a property of INFORMATION rather than of accuracy or of weighting,
which is what makes it right for this project:

- It does **not** require accuracy, so a participant who integrates all history
  but inaccurately -- strong recency bias, or large response noise -- is RETAINED.
- It does **not** require a particular weight on history, only that the weight be
  distinguishable from zero. There is no magnitude threshold to choose.
- It catches BOTH observed failure modes with one test: a copier gets no
  predictive benefit from history, a random/drifting responder gets none from
  anything.

Operationalised as `response_t ~ 1 + value_t + mean(value_0..value_{t-1})` per
participant, with a **trial-level cluster bootstrap** (resample whole trials) --
necessary because responses within a trial are strongly dependent (the slider
persists, the running mean moves slowly), so OLS standard errors are far too
small. Retained if the prior-mean coefficient's CI excludes zero.
`flag_non_integrator` in `utils/participant_filters.py`.

**Validation it does not just carve the distribution arbitrarily**: the excluded
group is far worse on an accuracy measure the criterion never sees -- median
|error vs running mean| 8.33 vs 4.90 (numbers) and 27.37 vs 6.94 (colors).

**Why the other three lose.** `performance` is accuracy-based, so it cannot
distinguish an inaccurate integrator from a non-integrator. `integration` (the
skill score) is **not monotone in integration depth**: measured on synthetic leaky
integrators it PEAKS at alpha=0.20 (+0.745) and is lower for a near-optimal
alpha=0.10 agent (+0.603), because with 15 observations mild recency
overweighting tracks the running mean better than a sluggish filter does -- and a
genuine alpha=0.70 integrator with realistic noise scores +0.115, a hair above
its own 0.10 threshold. `contingency`'s `recency_only` is the closest of the
three and tests the same construct; it differs in using an f²=0.02 effect-size
threshold and in-sample variance without clustering. `non_integrator` is that
idea done properly.

**Tested and rejected**, all recorded in the code so they are not re-attempted:
- Thresholding the serial-position weight on the latest observation (`g_lag0`).
  It recovers alpha almost exactly (0.100/0.200/0.350/0.494/0.687/0.959 for true
  alpha 0.10-1.00) and is nearly immune to response noise, so it is the right
  MEASURE of integration depth -- but it is continuous with no natural cutoff
  (largest gap 0.076 across a 0.03-1.00 range) and **cannot catch random
  responders**, who produce diffuse weights scoring ~0.12, indistinguishable from
  optimal. Any weight-based test is blind to the "nothing predicts them" mode.
  Report `g_lag0` descriptively; do not filter on it.
- A one-sided version (requiring a POSITIVE contribution, to catch scale
  inversion). 1 of 61 numbers and 0 of 61 colors are reliably negative, and that
  one is marginal (b=-0.074, CI [-0.171,-0.013]). Not worth the assumption.
- Requiring stability across session HALVES. 26% of retained numbers participants
  pass pooled but not both halves -- and the asymmetry runs the WRONG way for a
  fatigue story: 12 integrate only in the second half against 4 only in the
  first. The instability is mostly LATE LEARNING (error falls 19% from the first 8
  to the last 8 trials), so requiring both halves penalises a slow start, which
  is a consequence of the tutorial having no comprehension gate.
- A trials 8-31 burn-in. Moves retention by exactly ONE participant per task
  (numbers 42->41, colors 43->44), and the two retained sets are
  indistinguishable in accuracy measured on the same late trials (4.90 vs 4.79
  numbers; 6.94 vs 7.03 colors). Use all 32 trials.

**IT IS NOT THRESHOLD-FREE.** It removes the arbitrary MAGNITUDE threshold, which
was the main objection to the other three, and replaces it with a conventional
significance level. Measured sensitivity, which should be reported as a range:

| varied | numbers | colors |
|--------|---------|--------|
| `ci` = 90 / 95 / 99 | 16 / 17 / 24 flagged | 17 / 17 / 20 |
| bootstrap `seed` at `n_boot=2000` | 17 / 18 / 19 | stable |
| predictor set: `prior_mean` vs last-3-lags + older mean | 17 -> 23 (churn +10/-4) | 17 -> 15 (+3/-5) |

`ci=99` adds 7 for numbers (+41%). Seed noise moved 2-3 participants at
`n_boot=2000`, which is why the default is now **`n_boot=20000`** -- verified
stable, seeds 0/1/2 give identical membership, ~10 s per task. `n_boot`, `ci` and
`seed` are returned in the report so any exclusion set is reproducible.

The **predictor set is the largest source of variation**, and `prior_mean` is
right on principle, not merely convenient: it asks the definitional question as
ONE test, whereas the full-lag version splits the signal across four correlated
predictors, so every CI widens (power loss -> more flagged, numbers) while giving
four uncorrected chances at significance (multiplicity -> fewer flagged, colors).
Those errors moving in OPPOSITE directions across tasks is the signature of an
ill-posed test. Do not add lag predictors without a multiplicity correction.

**Known gaps, deliberately not engineered around.** The definition retains anyone
whose responses reliably use history, so it does NOT catch: integrating the WRONG
STATISTIC (running sum, max, a hand-picked subset); SCALE COMPRESSION (correct
direction, only using 40-60 of the slider -- fine for temporal panels, bad for
accuracy panels); or ANCHORED-WITH-A-NUDGE (parked near 50, shifting slightly).
The first two are arguably correct to retain; the third is a real miss.
Accuracy-sensitive analyses may want `performance` in ADDITION -- one filter
serving both jobs was probably never the right goal. And being a significance
test it is POWER-DEPENDENT: the ~30% rate is tied to this design's 32 trials and
does not transfer to a shorter one.

**`require_both_tasks` is the DEFAULT** (opt out with `--per_task_exclusion`):
a participant failing in either task is dropped from BOTH, so every task keeps
the same people. Per-task exclusion silently degrades every WITHIN-SUBJECT
cross-task panel (temporal col 6, variability col 3) whenever the criterion is
not equally strict in both tasks. Measured directly under `integration` with
per-task exclusion: numbers retained 29 and colors 36, and the 26-pid
intersection was a differently-selected group from either sample -- the
cross-task λ correlation fell to r=0.331 (p=0.099) from r=0.587 (p=0.0013).
That was NOT power or reliability: λ's split-half reliability was if anything
HIGHER (colors 0.836 vs 0.796), the attenuation ceilings were indistinguishable
(0.791 vs 0.780), and λ's SD and range were unchanged. Purely the composition of
the intersection. Requiring both tasks makes the intersection the sample by
construction, and restored r=0.409 (p=0.028) at n=29.

**Published precedent, for calibration.** carrabin (Prat-Carrabin 2024) excluded
4/25 (16%) on ONE model-free quantity -- mean absolute error against the true
generative parameter, with the excluded group at .263 (SD .0298) against .176
(SD .0132), a >6 SD separation. yoo excluded 8/46 (17%), of which SEVEN were
fMRI-technical (1 structural abnormality, 6 head motion >3mm) and exactly ONE was
behavioural, via a post-experiment questionnaire in which the subject said they
tracked pairwise differences rather than the average. Neither used a model-based
contingency test. Our rates are higher, but both were supervised lab studies (yoo
paying a $35 base) against an unsupervised 32-trial × 15-observation Prolific
session. NOTE carrabin's literal >6 SD threshold excludes ZERO participants here,
because our error distribution is CONTINUOUS where theirs had a 6-SD gap.

**Why the high rate is probably real, not an artefact of an aggressive filter.**
The `integration` criterion is model-free, shares no quantity with the temporal
panels, and has an untuned threshold -- and it independently reproduces 23 of the
25 and 18 of the 19 participants `contingency` excludes. Roughly half of numbers
participants perform WORSE than reporting only the latest observation, which is
non-compliance on any reading. Three criteria converging on the same people is a
stronger argument than any one of them alone, and is how this should be reported.

**Effect on the results, measured** (numbers unless noted; see the figure
sections for what each panel is):

| build | n both | cross-task λ r | col 3 p | col 2 decay |
|-------|--------|----------------|---------|-------------|
| contingency | 27 | 0.587 (p=.0013) | 0.00043 | 3.09x |
| non_integrator | not yet built | -- | -- | -- |
| baseline + require_both | 29 | 0.409 (p=.028) | 0.00000 | 2.54x |
| performance | 44 | 0.572 (p<.0001) | 0.033 | 1.23x |
| no filter | 60 | **0.656** (p<.0001) | 0.011 | 1.13x |

Two things to carry from that table. The DECAY results (cols 2-3) need a filter
and weaken monotonically as it loosens. The CROSS-TASK λ correlation does not --
it is STRONGEST with everyone included (r=0.656), so that finding does not depend
on any exclusion. Different panels have different sensitivity to exclusion, and
that asymmetry should be reported rather than smoothed over.

**Threshold-choice honesty.** These criteria were compared partly on whether the
results survive, which is the same hazard as tuning a threshold. What makes
`integration` defensible independently is that its threshold is PRINCIPLED, not
chosen: skill < 0 means "did not beat a strategy the instructions rule out".
`performance`'s 2.0 SD, by contrast, is a conventional outlier bound, and
`contingency`'s f²=0.02 is Cohen's convention -- both citable, neither derived
from this data.

**A consequence of `integration` worth stating explicitly in any write-up**: for
colors this criterion is near-vacuous on its own, because "report the last binary
draw" means slamming to 0%/100% every trial (error 39.35 vs optimal 11.75) and
almost anything beats it. So under `integration`, COLORS IS EFFECTIVELY FILTERED BY
NUMBERS-TASK BEHAVIOUR, via `require_both_tasks`. Defensible -- numbers is where
the trivial strategy is plausible -- but not something to leave implicit.

#### The `contingency` criteria in detail

Three criteria, combined with OR, identify participants who show no
evidence of genuinely attempting the task (as opposed to attempting it
but updating sub-optimally, which is explicitly NOT excluded): recency-
only updating, non-contingent-sign updating, non-contingent-magnitude
updating. All three ask the SAME kind of question, the same way: does
adding one specific piece of task-relevant information to a regression
explain a non-trivial amount of additional variance, measured by Cohen's
f² (the standard effect size for an added predictor's incremental
contribution) against Cohen's own conventional f²=0.02 "small effect"
boundary -- one established, citable convention, applied consistently,
rather than a different ad hoc statistical construction per criterion.

- **recency_only**: f² of adding `prior_mean` (mean of all strictly-
  prior values in the trial) to a regression that already has
  `current_value`, predicting `response`. Someone whose response is
  explained by the single most recent observation, with prior history
  adding nothing beyond that, is flagged; someone genuinely trying to
  integrate but doing so poorly or noisily (a real, if weak, incremental
  contribution from history) is not.
- **noncontingent_sign**: f² of adding `sign(discrepancy)` to an
  intercept-only model, predicting `update` (the participant's own
  response change). Near-zero means the DIRECTION of their updates
  carries no information about the stimulus.
- **noncontingent_magnitude**: f² of adding `|discrepancy|` to an
  intercept-only model, predicting `|update|`. Near-zero means the SIZE
  of their updates carries no information about how surprising the
  evidence was, even if direction sometimes happens to line up.

**Deliberately an effect-size threshold, not a significance test**:
checked directly against two real batches (~448 updates/pid each) before
settling on this -- at that sample size, a plain significance test
declares virtually any nonzero effect significant, making it uselessly
lenient, AND makes the flagging decision depend on how much data a
participant happens to have (more timeouts -> less power -> easier to
"pass" with the same underlying behavior) rather than on how large the
effect actually is.

**Superseded an earlier three-different-statistical-tools version**
(a tolerance-based literal-copy check, a binomial test, a raw Pearson
correlation, and a partial-correlation-with-a-hand-picked-r=0.10-cutoff)
that was defensible criterion-by-criterion but read as an ad hoc
patchwork as a COLLECTION -- hard to answer "why these specific tests,
with these specific cutoffs, and how do you know you've covered every
way someone could be inattentive?" The current recency_only criterion
SUBSUMES the old literal-copy check empirically (every case the old
check flagged shows an even more extreme version of the same signature)
and, as a side effect of using Cohen's regression-specific f²=0.02
convention (~r=0.14 equivalent) rather than his bivariate-correlation
r=0.10 convention, catches 2 real participants the r=0.10 version had
missed. Archived at `archive/utils/archive_participant_filters_legacy.py`
-- not deleted, since it's genuine prior methodology worth pointing to,
just no longer live.

---

## Legacy: task/ (retired)

The original JATOS/MindProbe-hosted online task (continuous/binary
naming). Superseded by task_backend above and no longer under active
development -- remains on disk for historical reference and as a
fallback that was never actually cut over from. Full design history,
every real bug found and fixed, the sequence-generation-method debate,
and the JATOS incident investigation that motivated building
task_backend at all: **`docs/HISTORY.md`**.

---

## Active models

| Model | Role | Free params |
|-------|------|-------------|
| Mean | Optimal running mean (Bayesian baseline) | none |
| LeakyIntegrator | Exponential forgetting baseline | gamma |
| PrimacyRecency | Temporal weighting (primacy + recency terms) | eps_p, eps_r |
| NoisyCounting | Task-specific model (Prat-Carrabin 2024); carrabin only | mu, sigma_c, nu |
| RL_lambda | Power-law delta rule (explicit equation) | alpha_0, lambda_ |
| NEF | Spiking NEF integrator (emergent power-law dynamics) | alpha_0, lambda_ |

| NoisyRL_lambda | RL_lambda + STATE noise only; all 4 datasets | alpha_0, lambda_, sigma_state |
| {Mean,LeakyIntegrator,PrimacyRecency,RL_lambda}_resp_noise | base model + i.i.d. RESPONSE noise via `add_noise()`; all 4 datasets | base params + sigma_resp |

### Two response-noise mechanisms, kept deliberately separate

    NoisyRL_lambda:
      alpha(n) = alpha_0 / n^lambda
      e_n = clip(e_{n-1} + alpha(n)(x_n - e_{n-1}) + xi_n, -1, 1)   xi ~ N(0, sigma_state)
      response = e_n

    <model>_resp_noise (models.math_models.add_noise):
      mu = run(base_params).response          -- ONE deterministic call
      response = clip(mu + eta, -1, 1)         eta ~ N(0, sigma_resp), i.i.d. per row

**Originally NoisyRL_lambda had BOTH sigma_state and sigma_resp** (see
docs/HISTORY.md for that version's results). It was split so the two noise
MECHANISMS -- compounding vs i.i.d. -- can be compared at EQUAL parameter count (one
extra param each: NoisyRL_lambda's sigma_state vs
e.g. RL_lambda_resp_noise's sigma_resp), isolating which mechanism captures human
data better rather than confounding it with parameter count. This is the
comparison the NLL infrastructure below exists to run.

- `sigma_state` enters the ESTIMATE, so it persists and compounds -- a perturbation
  at step j reaches step n with weight `prod_{k=j+1..n}(1-alpha(k))`. Produces
  residual variance GROWTH and within-trial residual AUTOCORRELATION (temporal
  cols 3-4 for soltani).
- `sigma_resp` (now via `add_noise`, not a NoisyRL_lambda parameter) enters only
  the REPORTED value, i.i.d. per row, no persistence. Raises |Δresponse| by
  ~`1.128*sigma_resp` uniformly -- a PLATEAU rather than growth, zero
  autocorrelation. `add_noise` needs no per-model branch and no trial-replay
  logic (i.i.d. noise has no sequential dependency to preserve), so it wraps ANY
  of Mean/LeakyIntegrator/PrimacyRecency/RL_lambda generically, without depending
  on a prior RMSE fit -- mu comes fresh from run() every call.

Both registered for **all four datasets** now (carrabin, yoo, soltani_numbers,
soltani_colors), not soltani-only as originally built -- extending required fixing
two real bugs, see "PITFALLS" below.

### Fitting noise: RMSE cannot, NLL can (`--loss {rmse,nll}` on fitting.fit)

**RMSE cannot identify a noise parameter, ever.** Squared error is minimised by
the conditional mean, so noise only adds cost. Measured directly: an unbounded
RMSE fit of the original two-sigma NoisyRL_lambda gave `sigma_state` median 0.0000
(24/35 exactly zero) and `sigma_resp` median 0.0000 (25/35 zero) for soltani
numbers -- the same collapse documented below for NoisyCounting's `sigma_c` under
RMSE. A floor forces a nonzero value but does not mean the DATA chose it.

**`fitting.losses.compute_nll`** (Gaussian NLL of the observed response under the
model's simulated predictive distribution) is a proper scoring rule -- it
penalises a wrong mean AND a wrong variance, so it CAN find a genuine interior
optimum. Verified directly, unconstrained (floor 0.001, effectively no floor): on
soltani_numbers pid 1, NLL fell from 389 at sigma_resp=0.001 to -2.46 at the
optimum (~0.04-0.05) and rose again beyond it -- a real U-shape, not a monotone
pull toward zero.

`fitting.fit(..., loss_fn="nll", n_sims=100)`. Dispatches on `model_type`:
genuinely stochastic math models (`_STOCHASTIC_ENSEMBLE_MODELS`, currently
`{NoisyRL_lambda}`) go through `math_models.simulate_ensemble`; `<model>_resp_noise`
names go through `math_models.add_noise`; `NEF` goes through its OWN
`NEF.simulate_ensemble` (added for NEF's NLL branch -- see "NEF architecture"
below and docs/HISTORY.md), a real Nengo ensemble rather than a closed-form
formula, requiring a counting-activity file with n_trials*n_sims precomputed
seeds. Checked BEFORE the Optuna study is created, so a bad combination (e.g.
`--loss nll` on plain `Mean`) fails immediately with the valid alternatives
listed, not on the first trial.

**For NEF specifically, pass `--n_sims 50` explicitly** rather than relying on
this flag's own default (100, validated for `sigma_resp`/NoisyRL_lambda, NOT
for NEF). 50 is a ballpark from cheap-model calibration
(`scripts/calibrate_nll_nsims.py`, using NoisyRL_lambda as a structural proxy
for NEF's own compounding noise), not a direct NEF measurement -- see
`models.NEF.NEF_DEFAULT_N_SIMS`'s own comment and docs/HISTORY.md. Raise it
later once a real NEF measurement exists.

`n_sims=100` is verified stable: 5 reseeded reps of a sigma_resp sweep all picked
the identical argmin (n_sims=25 already agreed). Cost ~0.45s/eval, so a 300-trial
fit is ~2.3 min/pid.

**NLL output files get a `_nll` suffix** inserted before `{pid}` --
`{model_type}_{stem}_nll_{pid}_*.pkl` -- so an NLL fit can NEVER silently overwrite
an RMSE fit of the same model_type in the same run_folder (their loss scales
differ; NLL can be negative, RMSE cannot).

### Default NLL fitting method: noise-only override (this session)

**NLL fits now default to fixing the base model's own free parameters at
their RMSE-fitted values, and searching ONLY the noise/architecture
parameter** (`sigma_resp` for `<model>_resp_noise`; `n_neurons` for a future
NEF variant -- not yet built, see "NEF architecture" below), rather than a
full joint search over everything. `fitting.fit`'s own `override_from_folder`
parameter (CLI: `--override_from_folder`) implements this: it reads that
pid's own RMSE fit (`{base_model}_{stem}_{pid}_params.pkl`, falling back to
the combined `{base_model}_{stem}_params.pkl` filtered by pid if no per-pid
file exists -- carrabin/yoo's own RMSE folders only have the combined one),
and pins every one of the base model's free parameters to those values --
Optuna's search space then contains only whatever parameter(s) aren't
listed in that override.

**Why**: verified directly across LeakyIntegrator/PrimacyRecency/RL_lambda x
all 4 datasets (12 combos) that a full joint NLL search barely improves on
this restricted one, on EITHER performance or behaviour:
- Loss: full-joint IS statistically significantly better in 11/12 combos
  (Wilcoxon, large n makes even tiny shifts detectable) but the actual
  magnitude is negligible in all but one -- 0.001-0.028 NLL units against
  medians of -0.5 to -2.2 (under 2% of the loss scale).
- Behaviour: response correlation between the two fits' simulated
  trajectories is r=0.995-1.000 everywhere, RMSE 0.009-0.058 on the [-1,1]
  response scale.

**One real exception**: RL_lambda on carrabin -- diff=0.116 NLL units and
r=0.995 (still high, but the weakest of all 12 cells), 4-6x every other
cell's own diff. Consistent with an earlier, separate finding: RL_lambda's
own alpha_0/lambda_ show the LARGEST RMSE-vs-full-joint-NLL drift on
carrabin (26%/25%) and yoo (17%/22%), vs 0-5% on both soltani tasks --
fixing those parameters costs the most precisely where they most want to
move. Worth re-checking before relying on this simplification for RL_lambda
on carrabin specifically; the soltani tasks (where the neural work actually
runs) show no such issue.

**Recipe** (one dataset x model at a time, matching every other multi-
dataset loop this project uses -- note carrabin/yoo's own RMSE fits live in
their own folders, not a shared one):

    for m in LeakyIntegrator PrimacyRecency RL_lambda; do
      venv/bin/python -m fitting.submit carrabin ${m}_resp_noise --loss nll \
          --n_trials 100 --run_folder nll_noise_only --override_from_folder carrabin
    done
    # yoo: --override_from_folder yoo; soltani_colors/soltani_numbers: --override_from_folder rmse

    venv/bin/python -m fitting.collect nll_noise_only --type params
    venv/bin/python -m fitting.collect nll_noise_only --type responses

`data/runs/nll_noise_only/` is now the CANONICAL location for NEW NLL fits
of these 3 models going forward. `data/runs/nll/` (the old full-joint
search) is kept as the verification baseline the comparison above was run
against -- not deleted, not being actively added to.

### PITFALLS when extending a model to a new dataset, learned the hard way

Two real bugs surfaced extending NoisyRL_lambda from soltani-only to all four
datasets, both invisible to `py_compile` and to exercising branches in isolation --
caught only by comparing `simulate_ensemble`/`add_noise` against `run()` directly:

1. An unguarded string-replace on an anchor that exists once per dataset
   (`_run_carrabin`, `_run_yoo`, `_run_soltani_common`) silently TRIPLICATED a
   model's branch into all three instead of the one intended. Dormant until the
   model was actually registered for the other datasets.
2. `simulate_ensemble` labelled ensemble columns with a synthetic `range(n_obs)`
   instead of the dataset's REAL observation values. Harmless for soltani
   (0-indexed, coincidentally identical) but WRONG for carrabin (1-indexed),
   silently feeding the wrong exponent into the Laplace-shrinkage formula and
   biasing the ensemble by up to 0.167.

**Run `scripts/verify_ensemble_invariant.py` after touching `simulate_ensemble`,
`add_noise`, any `_run_*` dispatcher, or `_validate_model_dataset`'s allowlists,
and before trusting any `--loss nll` fit on a dataset/model combination it has not
been checked against before.** It checks, per dataset x model: `simulate_ensemble`
matches `run(seed=i)` exactly; `add_noise` reduces exactly to `run()` at sigma=0;
its empirical mean/SD track the requested values away from the +-1 clipping
boundary (clipping bias there is CORRECT behaviour, not a bug -- confirmed
directly on soltani_colors' Mean model, which legitimately outputs exactly +-1 on
15.6% of rows); and the bare model name and the `_resp_noise`-suffixed name
produce IDENTICAL output (the exact seam that broke silently once already).
There is no pytest suite in this project; do not add a docstring claiming
otherwise.

**This script does NOT cover `models.NEF.simulate_ensemble`** -- it only imports
`models.math_models`. NEF's own ensemble now HAS a real check, added this
session to `scripts/check_NEF_pipeline.py` (`--n_sims_ensemble N`,
`check_ensemble_invariant()`) rather than to `verify_ensemble_invariant.py`
itself (kept there since that script is math-models-only by design; NEF's
check needs a real Nengo run and an activity file, which doesn't fit that
script's cheap/no-Nengo nature). Two genuine invariants, both confirmed
against REAL Nengo (carrabin, n_neurons=100/n_neurons_counting=100,
n_sims=5, pid 5, 3 trials): (1) `simulate_ensemble`'s sim=1 row matches
`run()`'s point estimate to machine precision (1.11e-16), since both
resolve to the identical seed by construction of the key formula -- any
real disagreement would mean a bug in one of the two independently-written
code paths; (2) different sims give genuinely different responses (std
0.017-0.10 across the tested trials), confirming the multi-seed mechanism
produces real independence rather than accidentally-duplicated seeds. Both
failure modes were also confirmed to actually fire when deliberately broken
(a seed-mismatch stub, a degenerate-sim stub), not just pass trivially.

### Earlier result (pre-split model; superseded, see docs/HISTORY.md for full detail)

With BOTH sigma_state and sigma_resp fit by RMSE with hand-calibrated floors
(soltani only), adding calibrated response noise to RL_lambda's fitted output
moved the temporal decay ratio from 7.24 (RL_lambda alone, ~3x too steep) to 2.50
against a human 2.46, and the descriptive-lambda gap from +0.382 (p<0.0001) to
+0.035 (p=0.62, indistinguishable). That finding used add_noise-style
post-hoc noise on RL_lambda's OUTPUT, not a NoisyRL_lambda parameter, so it is
UNAFFECTED by the sigma_resp removal and still reproduces. Known limitation,
still real: identical noise for every pid gives human-SCALE variability but not
human INDIVIDUAL DIFFERENCES in it (numbers lambda correlation dropped 0.644 ->
0.524 even as the level matched) -- resolving this needs either the NLL fits
above (which let sigma vary per participant) or per-participant fixed sigma.

NoisyCounting applies to carrabin only. Two fitted versions:
- RMSE-fitted: sigma_c collapses to ~0 (response-noise artefact; methodologically revealing)
- MLE-fitted (fit_mle.py): recovers sigma_c ~0.03-0.08, nu ~0.08-0.21

### RNN as a conditional-mean estimator (models/RNN.py)

A TinyGRU (`n_hidden` units, k-fold over trials, early stopping) fit per
participant, after Ger, Shahar & Shahar (2024, eLife). Intended as a
"best-possible" predictor of a participant's responses, so that
`sigma = std(source - RNN)` estimates irreducible response variability and the RNN
prediction can serve as a denoised conditional-mean target.

**Whether that premise holds is DATASET-DEPENDENT, and it was measured on both.**
The test is whether the RNN beats simple models on HELD-OUT data -- if a
2-parameter model out-predicts it, it is not a best-possible conditional mean and
its residual is contaminated with its own prediction error.

| | trials/pid | obs/trial | sequences | held-out RMSE |
|---|---|---|---|---|
| carrabin | 200 | 5 | repeating pool | RNN **0.1225** beats NoisyCounting 0.1324 (15/21 pids) and every other model 21/21 |
| soltani | 32 | 15 | mostly unique | RNN **0.0626** LOSES to RL_lambda 0.0526 (0/4 pids) and to the parameter-free running mean 0.0545 |

Carrabin gives the GRU 6x more trials AND repeating sequences, so a held-out trial
has often been seen -- interpolation. Soltani's 32 sequences are unique, so a
held-out trial is genuinely novel -- extrapolation, which is where a 101-parameter
model loses to a 2-parameter delta rule. **So use the RNN for carrabin; for soltani
use qid-grouped response std.**

`n_hidden` matters but does not rescue it. Sweep on soltani_numbers (4 pids, k=8,
28 of 32 trials per fit), held-out RMSE: n_hidden=1 0.1751, 2 0.0911, **3 0.0626**,
4 0.0701, 5 0.0722. A clean U-shape with an interior optimum -- 1 underfits, 4-5
overfit -- and the default of 4 was mistuned by ~19 percentage points. But even at
the optimum RL_lambda wins on 4/4 pids.

Consequences for the two applications considered:
- `sigma_RNN` cannot replace prefix (qid-grouped) variability for soltani. At the
  best setting it is 0.0626 against the qid estimate of ~0.055 -- only 14%
  inflated, tempting, but the inflation is the GRU's OWN prediction error, and
  RL_lambda's residual on the same rows would give a lower estimate still.
- The RNN prediction cannot serve as a denoised target for a distributional loss on
  soltani, because it is LESS accurate than the models being evaluated. Scoring
  NoisyRL_lambda against a target that RL_lambda predicts better would be perverse.
  A distributional (NLL) loss needs no conditional-mean estimator anyway -- score
  the observed y under the model's simulated predictive distribution.

TWO BUGS FIXED while investigating, both of which made the module unusable on
soltani rather than merely inaccurate:
- `build_trial_tensors` derived observations per trial from `max(observation)`,
  silently assuming 1-INDEXED data. On soltani (0-indexed, 0..14) it computed
  n_obs=14 while every trial has 15 rows, so the `len(td) != n_obs` guard dropped
  EVERY trial and the function failed on an empty stack. Now uses the modal row
  count, which is index-agnostic, and raises with a clear message if no trial
  matches.
- `generate_rnn_responses` emitted `observation = oi + 1` over `range(n_obs)`,
  hardcoding 1-indexing: on soltani that mislabelled every row and dropped
  observation 0. Now uses each trial's own observation labels.

Also added `cross_validated_predictions()`, which stitches OUT-OF-FOLD predictions
covering every observation, so sigma is not deflated by the fit absorbing noise
(in-sample sigma on soltani is ~0.046-0.056 against 0.18 out-of-fold at k=5 --
in-sample matches the qid estimate only because it has memorised the trials). Note
it still uses the held-out fold for early stopping, so its predictions are mildly
optimistic; a nested split was judged not worth a third partition of 32 trials.

RNN fits are retained for reference and are not used in active figures.

---

## NEF architecture

**Planned, NOT YET BUILT**: applying the same noise-only override approach
(above) to NEF -- fixing alpha_0/lambda_ at their RMSE-fitted values and
letting Optuna search ONLY n_neurons under NLL, instead of the current
joint alpha_0/lambda_ search at a fixed n_neurons. Blocked on a real
prerequisite, checked directly rather than assumed: n_neurons can't be an
arbitrary Optuna range -- each candidate value needs its OWN precomputed
counting-activity file (`data/counting_activities_n{N}_nc{N}_{dataset}.pkl`),
and currently only the single production value (n_neurons=500) has one, for
ANY of the 4 datasets. Generating a real candidate set (e.g. matching the
existing `NEF_N_NEURONS_VALUES` list used by the old MLE n_neurons scan) is
a genuine disk/compute cost (the existing n=500 files already run ~1.3GB
each) that hasn't been scoped or approved yet.

The NEF model implements sequential evidence integration via three interacting
neural populations:

1. **Value ensemble**: maintains a running estimate of the current evidence
   mean. Receives weighted prediction-error input; its activity decodes the
   current estimate after each observation.

2. **Error ensemble**: computes the prediction error (new value − current
   estimate) and gates it by the current observation weight α(t). This
   ensemble's weight-tuned neurons are the key neural readout — their activity
   directly tracks α(t) across the sequence.

3. **Counting subnetwork**: tracks the observation count and decodes the
   power-law weight α(t) = α₀ / t^λ. This is the mechanism that produces
   the same per-observation discounting as RL_lambda but via spiking dynamics
   rather than an explicit equation. The subnetwork requires a precomputed
   activity file (counting_activities_n{n}_nc{nc}_{dataset}.pkl) generated by
   counting_integrator.py.

Trial-to-trial variability in neural tuning curves (controlled by the seed,
`counting_integrator.activity_key_for_trial(dataset, trial)` — `int(trial)` for
carrabin/yoo, `trial+1` for the 0-indexed soltani datasets) is the primary
spiking noise source, producing state-persistent response variability across
observations within a trial. That same value keys the activity file, and the two
must never diverge — see "what NOT to do".

Verified for soltani (2026-08-13): the memory ensemble's encoders/gain/bias are
bit-identical between the train-time build (`precompute_activities`,
`train=True`) and the test-time build (`NEF.run`, `train=False`) at the same
seed, and stored `mem_readout` matches a fresh test-time run exactly
(corr=1.000000). The existing `counting_activities_n200_nc1000_soltani_*.pkl`
files are therefore valid for the current fixed params and need no
regeneration; they are also byte-identical to each other, since the content
depends only on (n_neurons, n_neurons_counting, radius_c, timing, seed) and both
soltani tasks share those.

Activity files are loaded at fit time for speed (fast_decode mode). Generate
locally with counting_integrator.py then scp to the cluster before submitting
fitting jobs (see Simulation pipeline below).

### Multi-seed activity files, for NEF's NLL branch (`NEF.simulate_ensemble`)

`NEF.run()`'s single canonical seed per trial (`activity_key_for_trial(dataset,
trial)`) gives ONE deterministic point-estimate response -- fine for RMSE, but a
distributional (NLL) loss needs a genuine ENSEMBLE of responses per trial, which
means genuinely DIFFERENT seeds (different neural tuning curves) simulating the
SAME trial's stimulus, not n_sims copies of one seed. Reusing one seed across
trials would silently CORRELATE supposedly-independent ensemble members --
that seed's idiosyncratic tuning-curve bias would show up identically in every
trial that reused it.

`activity_key_for_trial(dataset, trial, sim=1)` now takes a `sim` argument
(default 1, exactly backward compatible): for `sim>1` the key is offset by a
full dataset-sized BLOCK per sim -- `(sim-1)*_DATASET_N_TRIALS[dataset] + base`.
So a genuine `n_sims`-member ensemble needs an activity file with
`n_trials*n_sims` entries, not just `n_trials` -- generate with
`counting_integrator.py --precompute_activities --n_sims N` (RESUMABLE: growing
an existing file to a larger `n_sims` only simulates the newly-needed keys, not
the ones already on disk -- these files are expensive enough at large
`n_neurons_counting` that re-paying for existing keys is a real cost, not a
convenience question).

`models.NEF.simulate_ensemble(params, n_sims, return_index=False)` is the NEF
analogue of `math_models.simulate_ensemble` -- same `(n_sims, n_rows)` return
shape, same row order (sorted by trial, then observation), so both slot into
`fitting.fit`'s NLL dispatch identically. It applies `run()`'s own
post-processing (`nef_response_to_model_scale`, then `apply_binary_transform`)
ONCE on the full stacked (sim, trial, observation) frame rather than
re-deriving carrabin's Laplace-shrinkage formula by hand, the way
`math_models.simulate_ensemble` has to (see that function's own docstring for
why re-deriving it is a real drift risk, not just extra code).

`models.NEF.NEF_DEFAULT_N_SIMS = 50` is a BALLPARK, not a validated number --
it comes from cheap-model calibration (`scripts/calibrate_nll_nsims.py`,
sweeping n_sims against NoisyRL_lambda/RL_lambda_resp_noise as structural
proxies for NEF's own noise, since a real NEF measurement is far more
expensive). Two findings from that calibration worth remembering:
- The noise MECHANISM matters for how many sims are needed. On soltani_numbers
  pid 13, RL_lambda_resp_noise's i.i.d. response noise gave a stable argmin at
  n_sims=10 already; NoisyRL_lambda's compounding state noise needed n_sims=40
  to stabilise. NEF's recurrent value ensemble means a sim's tuning-curve
  idiosyncrasies persist and compound through a trial, structurally closer to
  NoisyRL_lambda than to i.i.d. response noise -- so NoisyRL_lambda's number,
  not the cheaper i.i.d. one, is the honest stand-in.
- The required n_sims is NOT uniform across pids -- it scales with how much
  noise that pid's fixed (alpha_0, lambda_) needs to explain its residuals.
  Across 4 soltani_numbers pids, 3 stabilised at n_sims=10; one needed n_sims=40
  (and would need more to reach the same tightness the other three had for
  free). Raise n_sims if a real NEF NLL fit shows argmin instability across
  reseeded reps, using the same check (`fitting.fit`'s own historical n_sims=100
  validation for sigma_resp is the template: several reseeded reps, same
  argmin) -- there is no NEF-specific version of that check yet.

### Disk-cost model for multi-seed activity files (measured, not estimated)

Activity-file size depends almost entirely on `n_neurons_counting`^2 times the
number of trial-seeds precomputed -- NOT on `n_neurons` at all, since the file
only stores the counting subnetwork's `memory` ensemble (sized by
`n_neurons_counting`), never the value/error ensembles. Verified against every
file on disk before relying on it. Consequence for `n_sims>1`: total keys become
`n_trials*n_sims`, so cost scales LINEARLY in n_sims, and carrabin (200
trial-seeds) is ~5-7x more expensive per unit of `n_neurons_counting` than yoo
(30) or soltani (40) at the same n_sims. At `n_neurons_counting=2000`: yoo/
soltani are ~1-1.3GB per n_sims=1 (already generated, see below); carrabin at
the same size would be ~6.4GB for n_sims=1 ALONE, before any n_sims>1
multiplication -- worth choosing a smaller `n_neurons_counting` for carrabin's
own NLL activity files rather than reusing whatever was chosen for the RMSE
pass, if that pass ends up wanting a large `n_neurons_counting` there.

`data/counting_activities_n500_nc2000_{yoo,soltani_numbers,soltani_colors}.pkl`
(n_sims=1, generated this session as the large-`n_neurons` candidate for the
RMSE pass) are ~958MB / ~1.3GB / ~1.3GB respectively -- confirmed matching
this cost model, and soltani_numbers/soltani_colors confirmed byte-identical
via checksum. Carrabin instead uses `data/counting_activities_n500_nc500_
carrabin.pkl` (500/500, NOT 2000) -- exactly the cost reasoning above: at
carrabin's 200 trial-seeds, nc=2000 would be ~6.4GB, so nc=500 was chosen
instead. That file already existed on disk from an earlier session's n==nc
scan (406MB, verified 200/200 keys present) -- no new generation was needed
for carrabin at all.

---

## Carrabin response transform

All carrabin models EXCEPT NoisyCounting apply: response = raw * t/(t+2)
Implemented in utils/carrabin_transform.py. Never apply it twice.

---

## Fitting pipeline (RMSE)

Submit and collect per-dataset per-model:

    # Submit (cluster)
    venv/bin/python -m fitting.submit carrabin NEF --n_trials 100 --run_folder carrabin --k 5
    venv/bin/python -m fitting.submit yoo NEF --run_folder yoo --n_trials 100 --k 5

    # Collect params and responses
    venv/bin/python -m fitting.collect carrabin --type params
    venv/bin/python -m fitting.collect carrabin --type responses
    venv/bin/python -m fitting.collect yoo --type params
    venv/bin/python -m fitting.collect yoo --type responses

    # Collect activities (after responses; needed for neural figures)
    venv/bin/python -m fitting.collect yoo --type activities --ensembles error --timing once_per_obs

Run folders: data/runs/carrabin/, data/runs/yoo/, data/runs/refit/,
data/runs/soltani/
The --nef_folder flag in figure scripts redirects NEF data to a separate folder
(e.g. --run_folder yoo --nef_folder refit uses yoo for other models, refit for NEF).

### dataset vs --datafile (the decoupling)

`dataset` is the model-FAMILY key: it indexes MODEL_PARAMS, selects the
branch in math_models, keys the transforms in binary_transform, and names
the NEF counting-activity files. `--datafile` is a plain filename suffix
selecting WHICH BUILD of that family's human data to fit:

    data/{dataset}_{datafile}.pkl        # input
    {model_type}_{dataset}_{datafile}_{pid}_*.pkl   # every output

`utils.paths.dataset_stem(dataset, datafile)` is the single source of truth
for that combination — always use it rather than formatting the name
locally. Consequence: a new round of data needs NO new model plumbing (no
MODEL_PARAMS entry, no math_models branch, no 340 MB activity-file
regeneration), just a new pkl. `--datafile` matches the figure scripts'
existing flag of the same name, deliberately.

Why this exists: `data/runs/soltani_math_v1` (now in `_trash/`) held fits
made against JATOS-era data under the same unsuffixed dataset name as a
later, different pkl, with non-corresponding pids — and the figures merged
them on `pid`, plotting one set of people's fits against another's data.
The suffix in the filename makes that class of mismatch impossible.

### soltani math-model fits

Both tasks share ONE run folder; each filename carries its own dataset
stem, so they cannot collide. The RMSE and NLL fits now use TWO SEPARATE
run folders (`data/runs/rmse/` and `data/runs/nll/`), not one shared
`data/runs/soltani/` -- that older folder holds STALE fits made against
pre-fix (contaminated / smaller-pid-count) data and is no longer read by
any current figure. See docs/HISTORY.md's pid-registry/pilot-4 section for
why the switch happened.

`--datafile complete_pairs` was the canonical production data at an
EARLIER, smaller stage (21 pids); it is now the CANONICAL UNSUFFIXED
`data/soltani_{numbers,colors}.pkl` at 46 pids (contamination-free,
registry-stable -- see "Data pipeline" above). Omit `--datafile` for any
new fit.

`all` expands to every model INCLUDING NEF, and there is no skip flag — to
fit only the math models, submit one model at a time:

    for m in Mean LeakyIntegrator PrimacyRecency RL_lambda; do
      venv/bin/python -m fitting.submit soltani_numbers $m \
          --n_trials 300 --k 5 --run_folder rmse
    done
    venv/bin/python -m fitting.submit soltani_colors $m \
        --n_trials 300 --k 5 --run_folder rmse   # (loop over $m again)
    venv/bin/python -m fitting.collect rmse --type params
    venv/bin/python -m fitting.collect rmse --type responses

NLL fits (adds `--loss nll`; Mean/LeakyIntegrator/PrimacyRecency need their
own `_resp_noise` suffix -- see "Active models"/"Fitting noise" above --
NoisyRL_lambda does not):

    for m in Mean_resp_noise LeakyIntegrator_resp_noise PrimacyRecency_resp_noise NoisyRL_lambda; do
      venv/bin/python -m fitting.submit soltani_numbers $m \
          --n_trials 300 --k 5 --run_folder nll --loss nll
    done
    # (repeat for soltani_colors)
    venv/bin/python -m fitting.collect nll --type params
    venv/bin/python -m fitting.collect nll --type responses

**NEF RMSE fits -- all four datasets, this weekend's planned run.** Not
soltani-only any more: per this session's decision, carrabin and yoo's NEF
fits are being redone fresh too (their existing `data/runs/carrabin/`,
`data/runs/yoo/`, `data/runs/refit/` fits are the OLD, smaller-`n_neurons`
versions -- ignore them as a baseline going forward, and note every current
figure script still DEFAULTS to reading those old folders until someone
deliberately repoints `--run_folder`/`--nef_folder` at the new ones).

`fitting/model_params.py`'s NEF `fixed` dict was bumped this session:
yoo/soltani_numbers/soltani_colors to `n_neurons=500, n_neurons_counting=2000`
(previously 200/1000), carrabin to `n_neurons=500, n_neurons_counting=500`
(previously 100/100, and NOT 2000 like the others -- carrabin precomputes
200 trial-seeds vs yoo's 30/soltani's 40, so nc=2000 there would cost
~6.4GB against ~1-1.3GB for the other three; nc=500 keeps it cheap and
reuses a file already on disk). See that file's own module docstring for
why this is the ONLY place that controls submit-time size (no CLI override
exists). Matching counting-activity files were confirmed this session:
`counting_activities_n500_nc2000_{yoo,soltani_numbers,soltani_colors}.pkl`
generated LOCALLY and verified (correct key counts/ranges for each
dataset's own `_DATASET_N_TRIALS`; soltani_numbers/soltani_colors confirmed
byte-identical via checksum); `counting_activities_n500_nc500_carrabin.pkl`
already existed from an earlier session and was verified valid (200/200
keys, correct MtM shape) -- no new generation needed for carrabin at all.

**One thing NOT yet confirmed before submitting a weekend-long run --
worth checking rather than assuming:**

~~scp to the cluster~~ -- DONE (confirmed this session): the local files
(`counting_activities_n500_nc500_carrabin.pkl`,
`counting_activities_n500_nc2000_{yoo,soltani_numbers,soltani_colors}.pkl`,
now including the `--n_sims 2` entries) were generated locally then copied
to the cluster.

~~Real per-trial timing at the new sizes~~ -- DECIDED AGAINST measuring
first (informed decision, this session): proceeding straight to a 200-trial
submit on the strength of the priors already in hand (carrabin's real
~2s/trial at the OLD 100/100 size; the June session's finding that NEF's
per-point cost is dominated by fixed Nengo overhead rather than scaling
much with `n_neurons`), rather than spending the few minutes per dataset to
confirm directly. Worth being explicit about what this accepts: `fitting.
fit`'s `study.optimize()` runs with NO persistent Optuna storage (the CLI
never passes `--storage`), and every output file (`params`/`performance`/
`folds`/`responses`) is written ONLY after `study.optimize()` returns, i.e.
only once all `n_trials` trials complete. If SLURM kills a job at the 72h
wall-clock limit before that, NOTHING is written -- not a partial result,
the entire 72 hours for that pid produces zero usable output. The 72h
limit bounds wasted TIME, not wasted OUTPUT. `fitting.submit`'s job count
for this run: 21 (carrabin) + 38 (yoo) + 46 (soltani_numbers) + 46
(soltani_colors) = 151 separate SLURM jobs, one per pid, each independently
subject to this risk.

**Submit** -- one SLURM job per pid, per dataset (151 total this run),
run_folder `rmse` (shared with the existing math-model fits there --
filenames never collide across datasets, via `dataset_stem`). `--dry_run`
first is cheap insurance for something this size (writes the job scripts
and prints what would be submitted, without calling `sbatch`):

    for ds in carrabin yoo soltani_numbers soltani_colors; do
      venv/bin/python -m fitting.submit $ds NEF --n_trials 200 --k 5 --run_folder rmse --dry_run
    done
    # inspect jobs/*.sh, then drop --dry_run to actually submit:
    for ds in carrabin yoo soltani_numbers soltani_colors; do
      venv/bin/python -m fitting.submit $ds NEF --n_trials 200 --k 5 --run_folder rmse
    done
    venv/bin/python -m fitting.collect rmse --type params
    venv/bin/python -m fitting.collect rmse --type responses

NEF's SLURM limits are 72h/32G (utils/slurm.py). CAVEAT, still unresolved:
`make_job_script` requests `--ntasks-per-node=1` with NO `--cpus-per-task`
and sets no OMP/MKL thread vars, so a cluster job may get 1 core and run
several times slower than a local multi-core estimate -- worth checking
the cluster's actual core allocation before committing to the full submit,
not just after a job is already running.

`fitting.fit`'s CLI is argparse-based (positional `dataset model_type pid`,
then `--n_trials/--k/--run_folder/--optuna_seed/--datafile`) — it used to
take up to 7 POSITIONAL args, so any old job script or muscle-memory
invocation of that form will now fail.

---

## Fitting pipeline (MLE — NoisyCounting only, carrabin)

    bash jobs/submit_mle_fit.sh NoisyCounting carrabin 500 100
    # args: model, dataset, n_fits, n_sims
    # output: data/runs/carrabin/NoisyCounting_carrabin_{pid}_params_mle.pkl

---

## Simulation pipeline (extra data for figure scripts)

Some figure panels require data generated outside the main fitting pipeline.
Always generate locally (or via cluster if slow), then scp to the cluster.
Never run NEF simulations through MCP tool calls (will time out).

### Counting activity files (required before NEF fitting)

    # Generate locally
    venv/bin/python models/counting_integrator.py --precompute_activities \
        --n_neurons 200 --n_neurons_counting 1000 --dataset yoo --n_trials 30

    # Copy to cluster
    scp data/counting_activities_n200_nc1000_yoo.pkl \
        f007qzn@discovery.dartmouth.edu:~/evidence_integration/data/

### PE dynamics (carrabin neural panel A — figure_carrabin_neural.py)

    # Run locally or via cluster (slow for large n_neurons)
    python scripts/extras_carrabin.py --experiment pe_dynamics --mode simulate \
        --alpha_0_list 0.1 0.3 --n_neurons_list 50 150 --run_folder carrabin
    # Output: data/runs/carrabin/pe_dynamics_NEF_carrabin_a{...}_n{...}.pkl

### Probe simulations (carrabin neural panels B–D — figure_carrabin_neural.py)

    # Submit to cluster
    bash jobs/submit_probe_pids.sh

    # Collect
    python scripts/extras_carrabin.py --experiment probe_pids --mode collect \
        --out_folder refit
    # Output: data/runs/refit/probe_pids_carrabin.pkl

### n_neurons scan (carrabin neural panel D — figure_carrabin_neural.py)

    # Run locally (moderate compute)
    python scripts/extras_carrabin.py --experiment n_neurons_scan \
        --n_neurons_list 50 100 150 200 250 300 --run_folder carrabin
    # Output: data/runs/carrabin/n_neurons_scan.pkl, n_neurons_scan_metrics.pkl

### Lambda=0 ablation (yoo neural panel B control — figure_yoo_neural.py)

    # Run all pids (on cluster or locally — ~3-6 min per pid)
    for pid in $(venv/bin/python -c "import pandas as pd; from utils.paths import data_path; print(' '.join(str(p) for p in sorted(pd.read_pickle(data_path('yoo.pkl'))['pid'].unique())))"); do
      venv/bin/python scripts/extras_yoo.py --experiment lambda0 \
          --mode run --pid $pid --source_folder refit --run_folder yoo_lambda0
    done

    # Collect (combines responses + activities, copies encoders + params)
    venv/bin/python scripts/extras_yoo.py --experiment lambda0 \
        --mode collect --run_folder yoo_lambda0 --source_folder refit

    # Use in figures: --nef_folder yoo_lambda0
    # Output: data/runs/yoo_lambda0/
    #   NEF_yoo_lambda0_responses.pkl, NEF_yoo_responses.pkl (alias)
    #   activities_error_yoo.pkl, encoders_error_yoo.pkl
    #   NEF_yoo_{pid}_params.pkl (copied from refit — lambda_ values for reference)

### Error ensemble activities (yoo neural panels A–C — figure_yoo_neural.py)

    # Collect after NEF yoo fitting is complete
    venv/bin/python -m fitting.collect yoo --type activities \
        --ensembles error --timing once_per_obs
    # Output: data/runs/{folder}/activities_error_yoo.pkl, encoders_error_yoo.pkl

### Neural predictions figure (Acts 1-3 — scripts/neural_experiments.py, make_neural_giant())

See `## Neural predictions figure (Acts 1-5)` above for the full
motivation/structure. All commands below target `soltani_numbers`.

    # Act 1.1 -- single-trial spike raster + decoded PE demo (arbitrary params)
    python scripts/neural_experiments.py raster_demo --task soltani_numbers \
        --alpha_0 0.8 --n_neurons 100 --lambda_ 0.7 --n_obs 15

    # Act 1.2 -- lambda sweep (error-neuron activity vs observation)
    python scripts/neural_experiments.py sweep --task soltani_numbers \
        --sweep_param lambda_ --sweep_values 0.1 0.6 \
        --base_alpha_0 1.0 --base_n_neurons 100 --base_lambda_ 0.5 \
        --n_obs 15 --n_seeds 10

    # Act 1.3 -- alpha_0 x n_neurons cross product (decoded PE vs time-within-obs)
    python scripts/neural_experiments.py sweep --task soltani_numbers \
        --sweep_param alpha_0 --sweep_values 0.1 0.3 \
        --sweep_param2 n_neurons --sweep_values2 30 300 \
        --base_alpha_0 0.1 --base_n_neurons 30 --base_lambda_ 0.5 \
        --n_obs 15 --n_seeds 10

    # Act 2/3 data source A -- probe simulation (cluster, per-pid; NOT YET RUN)
    python scripts/neural_experiments.py probe --task soltani_numbers \
        --mode submit --run_folder rmse
    python scripts/neural_experiments.py probe --task soltani_numbers --mode collect

    # Act 2/3 data source B -- per-observation activities (cluster, per-pid; NOT YET RUN)
    # Uses the existing generic activity-saving mechanism (utils/save_
    # activities.py), same one that produced yoo's own files above --
    # extended this session with an --out_folder flag (defaults to
    # --run_folder, so every EXISTING caller is unaffected) so activity/
    # encoder output can land somewhere other than the fitted-params
    # folder itself, keeping data/runs/rmse/ and data/runs/nll/ reserved
    # for pure behavioural fitting output.
    python -m fitting.submit soltani_numbers NEF --resubmit activities \
        --run_folder rmse --out_folder neural_experiments \
        --ensembles error --timing once_per_obs

    # Output: data/runs/neural_experiments/
    #   raster_demo_soltani_numbers.pkl
    #   sweep_soltani_numbers_lambda_.pkl
    #   sweep_soltani_numbers_alpha_0_n_neurons.pkl
    #   probe_soltani_numbers_pid{pid}.pkl (per-pid), probe_soltani_numbers.pkl (collected)

    # Build the figure (reads whatever's currently in data/runs/neural_experiments/):
    python scripts/make_paper_figures.py neural_giant

---

## Sequence-generation diagnostics (scripts/plot_sequences.py)

Plots candidate math/NEF models against task_backend's real, deployed
sequence pool (sequences_numbers.json/sequences_colors.json) -- NOT
task/'s old sequence files at all. Two branches, matching the two output
PDFs' own names (see the module's own docstring for the full rationale):

    # across_models: fix the sequences (aggregated across all 200 real
    # pool members by default), vary the MODEL (Bayes/RL_lambda/
    # LeakyIntegrator/PrimacyRecency, optionally NEF).
    venv/bin/python scripts/plot_sequences.py across_models --alpha_0 1.0 --rl_lambda 0.5
    venv/bin/python scripts/plot_sequences.py across_models --skip_nef   # math models only, fast
    # Output: figures/inspect_pool_sequences_across_models.pdf

    # across_pids: fix the MODEL (Mean by default, or RL_lambda), vary the
    # PID (each of the 200 real pool members individually) -- shows the
    # per-pid spread, not just the aggregate.
    venv/bin/python scripts/plot_sequences.py across_pids
    venv/bin/python scripts/plot_sequences.py across_pids --agent RL_lambda --alpha_0 1.0 --rl_lambda 0.3
    # Output: figures/inspect_pool_sequences_across_pids.pdf

Both default to `--pool_root task_backend` (assumes task_backend/ is a
sibling of this repo's root); both take `--n_pool N` to cap the pool for a
fast smoke test instead of the full 200 members. `--gt_mode {true,
running_mean}` (across_models only, default running_mean) controls the
RMSE panels' ground truth -- see the module's own `_metrics_from_responses`
docstring for the mechanics (unchanged from the pre-consolidation script).

Trusts `task_backend/generate_sequences.py`'s own `verify_numbers_trials`/
`verify_colors_trials` asserts to have already caught any prefix-
collision/quota-mismatch problem at generation time -- this script only
plots, it does not re-audit the pool's own correctness (an earlier version
bundled a human-readable audit CSV into across_models; dropped
deliberately -- see archive/archive_readme.md if that's ever needed again).

**Superseded/archived** (see archive/archive_readme.md's own "Sequence-
generation diagnostic scripts" section for the full account): the old
`scripts/inspect_sequences.py` / `scripts/inspect_iid_sequences.py` (task/-
pointed predecessors of this file), plus `scripts/test_sequences.py` and
its SLURM job trio (`run_nef_sequences.py`/`submit_nef_sequences.py`/
`collect_nef_sequences.py`) -- that whole pipeline read a stale task/ pkl
schema (`trial_type`/`std_condition` columns absent from every current
schema) and produced `figures/test_sequences.pdf`, a cross-task RL_lambda/
NEF-vs-Human quartile-split comparison now genuinely dead relative to
current production sequences, not merely redundant. The three still-useful
utility functions it exported (`fit_lambda_mid`/`split_half_lambda`/
`compute_abs_delta`) were inlined into `plot_sequences.py` before archiving,
not lost. `task/`'s own sequence-generation scripts (`generate_sequences_
iid.py`/`generate_sequences_momentmatch.py`/`generate_sequences_hybrid.py`)
are untouched by this and remain on disk for historical reference -- see
docs/HISTORY.md's "Sequence generation methods (task/)" section for that
now-closed methodological debate.

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
    aggregate.py     — SHARED aggregation for all three temporal figures'
                       error and |Δresponse| curves; see "Cols 1-2 aggregation"
    paths.py
    pid_registry.py  — persistent, append-only prolific_pid -> anonymized
                       pid registry (data/pid_registry.json, gitignored,
                       never through GitHub); replaces build_from_df's old
                       from-scratch sorted(...).unique() mapping
    plot_style.py    — apply_style, get_palette, pvalue_to_stars, fit_power_law_params
    slurm.py
    carrabin_transform.py
    save_responses.py
    participant_filters.py  — exclusion criteria (recency_only/noncontingent_sign/
                              noncontingent_magnitude), all via nested-regression
                              + Cohen's f² -- see "Participant exclusion criteria"
                              under "Online task: task_backend" below.
                              build_from_df's one deliberate departure from
                              carrabin/yoo's own pipeline
    colors_quasi_qids.py     — empirically-derived repeat structure for colors (real `qid`
                              never repeats there); shared by figure_soltani_temporal.py
                              (cols 3-4) and figure_soltani_variability.py
  scripts/
    figure_carrabin_performance.py   — P group (1×3)
    figure_carrabin_variability.py   — V group (1×4)
    figure_carrabin_temporal.py      — T group (1×4)
    figure_carrabin_neural.py        — N group (1×4)
    figure_yoo_performance.py        — P group (1×3)
    figure_yoo_temporal.py           — T group (1×4)
    figure_yoo_neural.py             — N group (1×4)
    figure_carrabin.py               — legacy combined figure
    figure_yoo.py                    — legacy combined figure
    figure_soltani_performance.py    — task_backend real pilot data (human only for now); P group
    figure_soltani_temporal.py       — task_backend real pilot data (human only for now); T group
    figure_soltani_variability.py    — task_backend real pilot data (human only for now); V group
    build_model_inputs.py            — shared build_from_df() filter/rescale/anonymize/save pipeline;
                                        build() wraps it for the old JATOS-pilot-file path
    pull_soltani_data.py     — Supabase -> build_from_df(), explicit pid lists per pilot
                                        round (--pilot/--numbers_pids/--colors_pids/--list_candidates)
    inspect_participant.py           — one real finished participant's raw responses vs 2 untuned
                                        reference agents (Bayes/RL), pulled directly from Supabase
    inspect_participant_temporal.py  — figure_soltani_temporal.py's own 5-panel layout, scoped to
                                        one real participant
    pilot_overview.py                — real pilot data vs. fixed-param models; likely
                                        superseded by figure_soltani_*.py's properly-
                                        fitted equivalent (see archive/archive_readme.md)
    plot_sequences.py                — sequence-generation diagnostics against
                                        task_backend's real pool (two branches:
                                        across_models, across_pids)
    extras_carrabin.py               — PE dynamics, probe_pids, n_neurons_scan
    extras_yoo.py                    — NEF response noise simulations
  jobs/
    submit_probe_pids.sh
    submit_n_neurons_scan.sh
    submit_yoo_noise.sh
    submit_mle_fit.sh
  venv/
```

All new scripts go in scripts/. Never create scripts at the project root.
Figures save PDF only (no PNG/SVG).

---

## Current figure panel inventory

### figure_carrabin_performance.py (P group, 1×3)
| Panel | Code | Content |
|-------|------|---------|
| A | — | Task schematic |
| B | P1 | Estimation error (RMSE to hidden probability) |
| C | P2 | Model fit (RMSE to human responses); sig bars from NEF outward |

### figure_carrabin_variability.py (V group, 1×4)
| Panel | Code | Content |
|-------|------|---------|
| A | V2 | KDE of response variability (human only); per-pid lines |
| B | V2 | Model RMSE regplot vs human response variability |
| C | V3 | Test-retest reliability scatter (first vs second half); Human, NEF, NC |
| D | V1 | NLL boxplots; sig bars from NEF outward |

### figure_carrabin_temporal.py (T group, 1×4)
| Panel | Code | Content |
|-------|------|---------|
| A | T1 | RMSE to true_p vs observation |
| B | T2 | Mean |Δresponse| vs observation |
| C | T6 | Within-trial residual autocorrelation, lag 1–3 |
| D | T5 | Residual std growth across observations |

### figure_carrabin_neural.py (N group, 1×4)
| Panel | Code | Content |
|-------|------|---------|
| A | N1 | Decoded PE dynamics, 4 param combinations (α₀ × n_neurons) |
| B | N2 | PE vs response variability (r=0.97****); partial r excl. α₀=0.97**** |
| C | N3 | Response and PE variability vs fitted α₀ |
| D | N4 | Response and PE variability vs n_neurons scan |

### figure_yoo_performance.py (P group, 1×3)
| Panel | Code | Content |
|-------|------|---------|
| A | — | Task schematic (yoo_task.pdf) |
| B | P1 | Estimation error (RMSE to cumulative true mean) |
| C | P2 | Model fit (RMSE to human responses); sig bars from NEF outward |

Default: --run_folder yoo --nef_folder refit

### figure_yoo_temporal.py (T group, 1×4)
| Panel | Code | Content |
|-------|------|---------|
| A | T1 | Estimation error vs obs; shaded bands show weak/strong U-shape range (N_GROUP=10) |
| B | T2 | Mean |Δresponse| vs obs (sns.lineplot, CI across trials, obs ≥ 2) |
| C | T3 | Split-half λ reliability: regplot per source (Human + all models) |
| D | T4 | λ_model vs λ_human regplot; identity line; one line per model |

λ fitted via curve_fit A·n^(-λ), bounds [0,2], no smoothing, obs ≥ 2.
U-strength = mean(task_error[obs≥26]) − min(smoothed error curve, w=5).
Default: --run_folder yoo --nef_folder refit

### figure_yoo_neural.py (N group, 1×4)
| Panel | Code | Content |
|-------|------|---------|
| A | N5 | Weight-neuron activity vs obs, split by high/low λ group (N=10 each) |
| B | N6 | Mean weight-neuron activity vs mean |Δresponse| per obs (regplot); r=0.92**** |
| C | N7 | Fitted λ mediates activity change (right axis) and mean |Δresponse| (left axis) |
| D | N8 | Late |Δresponse| vs late estimation error (obs 21–30); Human + NEF |

Weight-on neurons: enc_dim_0 > 0.5 in error ensemble encoders.
Default: --nef_folder refit

### figure_soltani_performance.py (P group, 2x3)
| Panel | Content |
|-------|---------|
| Col 1 | Task schematic (placeholder -- no schematic PDF exists yet) |
| Col 2 (P1) | Estimation error vs the RUNNING MEAN (not the fixed generative
         parameter), per pid, as a BOXPLOT -- Human AND each fitted model.
         Same `sns.boxplot` call and 45-degree tick rotation as
         figure_carrabin_performance.py / figure_yoo_performance.py |
| Col 3 (P2) | Model fit: fitted RMSE to HUMAN responses (IN-SAMPLE -- see below), per pid,
         one boxplot per model; significance bars from SIG_REFERENCE outward |

Row 1 = task-colors, row 2 = task-numbers. `--datafile <name>` selects which
round's data to load AND which fits to pair with it (see "dataset vs
--datafile" above); `--run_folder` defaults to `soltani`.

**Mean scores exactly 0 in P1 by construction** -- Mean *is* the running mean
and the ground truth *is* the running mean. This is a settled, deliberate
choice, not an oversight: it doubles as a live check that math_models' Mean
and `_add_running_mean_ground_truth` still agree, so a non-zero Mean box
means one of them has drifted. Do not "fix" it by dropping Mean from the panel
or changing the panel's ground truth. Differs from
figure_carrabin_performance.py's P1, whose ground truth is the fixed true_p,
where Mean is NOT degenerate.

P2 reads each model's fitted loss from `{model}_{stem}_performance.pkl` via
`_get_loss` (never hardcode a column name), because that is what the fit
minimised.

**That loss is IN-SAMPLE, not held-out.** `fitting.fit._cross_validate` computes
model responses ONCE per parameter set and then partitions trials into k disjoint
folds, so every fold contributes to the objective Optuna minimises and no fold is
excluded from parameter selection. Verified: for RL_lambda on soltani_numbers
pid 1, mean-of-folds 0.06295 against the all-trials loss 0.06308 -- a 1.2e-4
difference, purely Jensen's inequality. The per-fold spread in `_folds.pkl` is
real and useful as a stability check, but it is not validation. This is SHARED
code, so carrabin and yoo are identical in this respect.

ACCEPTED, NOT FIXED. Expected optimism is small (0-2 free parameters against ~480
observations per participant) but has not been measured. Consequence to respect:
comparing models with the SAME parameter count (PrimacyRecency vs RL_lambda) is
fine, but comparing parameter-free Mean against them is biased toward the richer
models. Fix with real nested CV -- fit on k-1 folds, evaluate on the held-out
fold, ~k times the cost -- before any claim rests on cross-complexity comparison.

Related performance note found alongside this: `math_models.run` issues a pandas
`query` per observation (~480 per parameter set), so a 300-trial Optuna fit does
~144k queries per participant. Precomputing each trial's value array once would
speed the math-model pipeline up substantially.

**RESPONSE SCALE: [-1,1] EVERYWHERE, in all three soltani figures.** No percent
conversion anywhere -- so RMSE, mean |Δresponse| and response variability are
numerically comparable with the carrabin and yoo figures (an RMSE of 0.12 means
the same thing in all of them), and any estimator borrowed from those figures
drops in unmodified. An earlier version converted to [0,100] percent for
readability, which required round-trip compensation factors (`LOSS_TO_PCT`,
`PCT_PER_UNIT`, `_to_pct`) purely to undo itself -- and one of them caused the λ
bounds bug above. Do not reintroduce a percent conversion. The ONE remaining
conversion is colors' `true_p`, which `build_from_df` leaves on its native [0,1]
while colors' `response` is on [-1,1]; `_add_ground_truth` maps it with `2p-1`.
For reference, RL_lambda CV RMSE across datasets now reads directly: soltani
numbers 0.085, carrabin 0.149, soltani colors 0.204, yoo 0.248.

`SIG_REFERENCE = "NEF"`, matching the carrabin/yoo figures.
`annotate_nef_comparisons` takes the reference as a parameter despite its name,
so this is a one-line change either way. NOTE: if the reference model has no
fits in the run folder, the significance bars are SUPPRESSED (not an error) --
so a run folder with only math fits shows P2 without bars. Set it back to
`RL_lambda` if you want bars in a math-only figure.

### figure_soltani_temporal.py (T group, 2x6)
| Panel | Content |
|-------|---------|
| Col 1 | Performance error vs observation, against the ground truth chosen by
         `--gt_mode` (default `true`) and aggregated per `--aggregate` |
| Col 2 | \|Δresponse\| vs observation, aggregated per `--aggregate`. Colors
         starts at observation 2, numbers at 1 (`DELTA_MIN_OBS`) |
| Col 3 | Residual variance growth (shared-prefix window only) -- Human +
         STOCHASTIC models only; see below |
| Col 4 | Within-trial residual autocorrelation (shared-prefix window) --
         Human + STOCHASTIC models only; see below |
| Col 5 | Split-half reliability of fitted λ -- ODD/EVEN trial split, not
         first/second half (see `_fit_lambda_split_half`'s own docstring);
         Human AND each fitted model, one regplot per source |
| Col 6 | λ_model vs λ_human per pid, one regplot per model + identity line
         (~yoo T4). Mean kept as a null control; see below |

### Cols 1-2 aggregation (`--aggregate`, `--errorbar`)

Lives in **`utils/aggregate.py`** and is SHARED by all three temporal figures
(soltani cols 1-2, yoo panel B, carrabin panel B). Both columns share one
aggregation choice, applied identically to Human and every model. Default
**`hier_mean_median`** with **`--errorbar ci`**; flags come from
`add_aggregate_args(parser)`.

It was extracted because the three figures had been using three DIFFERENT
schemes, with nothing making that visible anywhere: carrabin took the mean over
each pid's trials then mean ± SEM across pids (`hier_mean_sem`); yoo called
`sns.lineplot` straight on the long per-trial frame with `errorbar="ci"`, i.e. a
pooled mean with a ROW bootstrap (`flat_mean`, and a pseudo-replicated interval);
soltani had its own copy. Any estimator borrowed between them silently inherited
a different aggregation — the same class of problem as the λ bounds bug. Do not
reintroduce a per-figure implementation.

| mode | what it does |
|------|--------------|
| `hier_mean_median` | mean over each pid's trials, then MEDIAN across pids (default) |
| `hier_mean_sem` | mean over trials, then mean ± SEM across pids (the old behaviour) |
| `flat_median` | pool all trials and pids, take the median |
| `flat_mean` | pool all trials and pids, take the mean |

`--errorbar {ci,se,iqr,pi80}`. `ci`/`se` are INFERENTIAL (how precisely the
central tendency is pinned down); `iqr`/`pi80` are DESCRIPTIVE percentile spreads
of the underlying values and **must not be called confidence intervals**.

Why the default is a two-stage mean-then-median: per-pid |Δresponse| LEVEL varies
3-4x across participants (yoo 0.026-0.238, soltani numbers 0.025-0.251) and the
high-amplitude participants tend to be FLAT -- their movement is response noise
that does not decay. Under a mean they contribute in proportion to amplitude and
so dominate the late observations, roughly HALVING the visible decay: soltani
numbers decays 1.69x under a mean vs 3.09x under a median (yoo 1.66x vs 2.38x).
Taking the mean WITHIN a participant first (trials are exchangeable replicates)
and the median ACROSS participants (who differ in kind) is the standard two-stage
pattern; the reverse ordering would be the odd one.

Three findings worth not rediscovering:
- **A pooled (`flat_*`) median quantises to the response grid.** Responses come
  from a 101-value slider, so a median over raw per-trial deltas lands on a grid
  value and STAYS there: numbers' flat_median curve reads 0.06, 0.06, 0.06, 0.06,
  0.06, 0.04, 0.04 -- 4 distinct values across the whole curve, vs 14 for
  hier_mean_median. Averaging within pid first restores continuity.
- **`flat_mean` == `hier_mean_sem`'s point estimate for col 2**, exactly (verified
  max|diff| 1.4e-17), because trial counts are balanced (32/32 soltani, 30/30
  yoo) and the operation is linear. The hierarchy was never doing anything for a
  mean there. NOT true for col 1, where mean-of-sqrt differs from sqrt-of-mean by
  Jensen's inequality.
- **The `ci` band's step-like edge is inherent, not Monte Carlo noise.** With ~27
  pids the bootstrap median can only land on a small set of order statistics (19
  distinct values across 20000 resamples), so raising `n_boot` changes nothing
  (upper-edge mean |step| 0.0188 / 0.0183 / 0.0183 at n_boot 1e3 / 1e4 / 1e5).
  The same effect makes the median POINT estimate mildly non-monotone -- numbers'
  col 1 dips at observation 5 and rebounds at 6 purely because a distribution gap
  sits at the median rank there (values 0.0945, 0.0945, 0.0957, then 0.1138), and
  the median pid's identity changes each step. The mean is flat across that
  region. Do not read it as structure at a particular observation.

Effect of switching each dataset to `hier_mean_median`, measured before doing it:

| dataset | n_pid | n_obs | per-pid \|Δ\| spread | decay, mean → median |
|---------|-------|-------|--------------------|---------------------|
| carrabin | 21 | 5 | 1.7x | 1.00x → 1.01x (no change) |
| yoo | 38 | 30 | 3.1x | 1.66x → 2.38x |
| soltani numbers | 27 | 15 | 4.3x | 1.69x → 3.09x |

Carrabin genuinely does not change — with 5 observations and 200 trials per pid
there is no decay for a mean to understate, and the two estimators agree within
1%. It was switched anyway so the three figures cannot drift apart again. Note
carrabin keeps its own first-observation convention, where `delta` at the first
observation is `|response|` rather than NaN (the initial response treated as a
change from 0); that is applied inside its own `abs_delta()` BEFORE any
aggregation, and does shift its first point under a median.

Col 1's `flat_median` changes the METRIC, not just the estimator: RMSE already
contains an averaging step, so a "median of RMSEs" is ill-defined. The
aggregation is composed with the sqrt over raw per-trial squared errors, so
`sqrt(median(sq_err))` IS the median absolute error and `sqrt(mean(sq_err))` is
the pooled RMSE. The y-axis label follows the mode (`ERROR_METRIC_LABEL`) so it
never claims an estimator that wasn't used.

Colors' col 2 starts at observation 2 (`DELTA_MIN_OBS`). With binary evidence the
first delta is near-degenerate -- the running mean either doesn't move or jumps
the whole way -- so its per-trial distribution is BIMODAL on essentially two
values, and a median lands in the zero mode. Measured: the colors Mean model has
2 distinct delta values there with 58% exactly 0 (median 0.000 vs mean 0.424);
colors humans 46% zeros (median 0.060 vs mean 0.394). Numbers has no such problem
(7-9% zeros, ~87 distinct values) and starts at 1.

Investigated and REJECTED: dropping observation 0 from cols 3-4 to avoid colors'
slider-ceiling effect (78.7% of colors prefix responses are pinned at an extreme
vs 0.3% for numbers). Col 4's correlations do improve, but col 3 gets WORSE for
both tasks (numbers p=0.0004 -> 0.141) because losing a fitting point costs more
than the contamination. The binding constraint was prefix length, not the ceiling.

Cols 3-4 use colors' empirically-derived quasi-qid repeat structure
(`utils/colors_quasi_qids.py`); numbers uses its real, designed qid
repeats. `--plot_models` (off by default) overlays fitted
Mean/LeakyIntegrator/PrimacyRecency/RL_lambda/NEF in cols 1-2 and 5-6
(cols 3-4: stochastic only). `--datafile <name>` as above.

`--gt_mode` defaults to **`true`** (the fixed generative true_mean/true_p), not
`running_mean`. Against a fixed target error starts high and DECAYS as evidence
accumulates; against the running mean it is flat-to-rising because the target
itself moves. The two converge late in a trial as the running mean approaches the
true mean. Measured on complete_pairs numbers: running_mean 2.90 -> 5.79 across
observations, true 10.07 -> 6.40 (pre-scale-change units).

**λ is fitted by BOUNDED NONLINEAR LEAST SQUARES** (`scipy.optimize.curve_fit`
on `A*n^(-λ)`, `p0=[0.1,0.5]`, `bounds=([0,0],[2,2])`) -- byte-for-byte the same
estimator `figure_yoo_temporal.py` uses, so soltani and yoo λ are comparable.
**Do NOT use a log-log linear regression for λ anywhere in this project.** An
earlier version of this figure did, on the belief that the nonlinear fit
"degenerated to a ~0 floor on noisy human data" -- that was WRONG, and the
misdiagnosis is instructive: the degeneracy was a BOUNDS artefact. yoo fits on
the canonical [-1,1] scale where the |Δresponse| curve starts ~0.11 and `A<=2` is
ample; this figure had converted responses to [0,100] percent, where the same
curve starts ~6.0, so `A` saturated at 2, the fit could not reach the curve, and
λ collapsed to 0 for 22/27 pids. Log-log also weights the small late-observation
deltas far more heavily and depends on an arbitrary floor for `delta == 0`. The
two methods correlate r=0.91 (human) / r=0.95 (RL_lambda) but differ in level.

Two things λ depends on, both of which were wrong before:
- **n means "observations seen", not the raw `observation` value.** soltani is
  0-indexed, so `n = observation + 1`; yoo is 1-indexed, so `n = observation`.
  Getting this wrong understated n by one, which has huge leverage at the low end
  of a power law (log(1)=0 vs log(2)=0.693): human numbers λ 0.279 raw vs 0.433
  corrected.
- **the curve must be on [-1,1] before fitting** -- see the RESPONSE SCALE note
  below.

Sanity check that the estimator is working: `Mean` must return λ≈1 (a running
mean has α(t)=1/t) and `LeakyIntegrator` λ≈0 (fixed γ). On complete_pairs they
return 1.174/0.999 and 0.168/0.068 for numbers/colors. Neither was recoverable
under the log-log version.

**The shared-prefix window is PER TASK, and tunable for colors.**
`NUMBERS_PREFIX_LENGTH = 4` is fixed by DESIGN -- verified directly: within
`(pid, qid)`, numbers' `value` is identical across trials for observations 0-3 in
216/216 groups and identical in 0/216 at observation 4, so widening it would
admit non-shared stimuli and turn genuine stimulus differences into apparent
"response variability". Colors has no designed prefix at all; its groups are
CONSTRUCTED by `utils/colors_quasi_qids.py`, so its window is free, and its
default is now **5** (`PREFIX_LENGTH`), matching carrabin's own 5-observation
repeat window. `MIN_REPEATS = 3` for both -- three points is the floor for a
meaningful "typical response" per repeat, and col 3's within-group SD needs more
than 1 dof. Exposed as `--colors_prefix_length` / `--colors_min_repeats` on both
figure_soltani_temporal.py and figure_soltani_variability.py (colors only;
numbers is unaffected by either flag).

Raising colors from 4 to 5 is what made its col 3 trend detectable: per-pid
variance-growth slope went +0.0044 (20/27 pids positive, Wilcoxon p=0.086) to
+0.0135 (19/27, p=0.0047), retaining all 27 pids. Because colors draws are
binary there are 2^n possible prefixes against ~32 trials, so group size falls
off geometrically in n -- n=6 gives a steeper slope but drops to 23 pids, and
`min_repeats>=4` collapses the sample. Note these defaults were chosen from a
16-cell grid, so they are justified by matching carrabin's window and by SD
validity, NOT by having won the search.

Caveat for the variability figure's cross-task panel: its two axes are now
computed over DIFFERENT windows (numbers 4, colors 5). The correlation is the
interpretable quantity; absolute positions and the slope's distance from 1 are
not.

**Cols 3-4 include only Human + STOCHASTIC models**, where eligibility is
decided by `STOCHASTIC_MODELS` (NEF, NoisyCounting, NoisyRL_lambda) rather than
MODEL_ORDER.

### `--models`, and utils/soltani_models.py

All THREE soltani figures take `--models MODEL [MODEL ...]`, defaulting to
**Mean LeakyIntegrator PrimacyRecency**. RL_lambda, NoisyRL_lambda and NEF are
opt-in, so a default figure stays readable.

`MODEL_ORDER`, `DEFAULT_MODELS`, `STOCHASTIC_MODELS` and the
`add_model_args`/`resolve_models`/`stochastic_only` helpers live in ONE place,
`utils/soltani_models.py`. Each figure previously carried its own MODEL_ORDER copy
-- the same failure mode that let the three temporal figures end up with three
different aggregation schemes (see utils/aggregate.py). Do not reintroduce a
per-figure copy.

`MODEL_ORDER` is also the COLOUR order, since `get_palette` returns the first n of
a fixed list: a model's colour is its index. So **append, never insert** --
inserting shifts every later model's colour and silently makes new figures
incomparable with old ones (verified `get_palette(5) == get_palette(6)[:5]`, which
is why NoisyRL_lambda went last). Palettes are always built over the FULL
MODEL_ORDER, never a requested subset, so a model keeps its colour in any subset.

The variability figure gained model support in the same change (it previously
loaded none) plus a `--run_folder` flag. Only STOCHASTIC models are eligible there
-- every panel is built on within-qid residuals, which are exactly zero for a
deterministic model -- so with the all-deterministic default it prints an
explanatory line and stays human-only.
Both panels use residuals against a qid-conditional mean, and since a qid's
repeats share an identical prefix by design, a DETERMINISTIC model gives the
identical response every repeat and its residual is EXACTLY zero (verified on
pilot 5: max|resid| = 0.000e+00 for Mean/LeakyIntegrator/PrimacyRecency/
RL_lambda across 1152 prefix rows, vs 0.68 for Human). Including them draws
four flat lines at zero. These are carrabin's T5/T6 -- metrics whose whole
purpose is state-persistent response variability, which only a noisy generative
process has. Do NOT "fix" the missing deterministic curves by widening
`_STOCHASTIC_MODELS`.

Model response files have no `qid` column, so `_attach_qid` merges it from
`human_for_repeats` -- i.e. AFTER `add_quasi_qids` for colors -- so models are
grouped by the same repeat structure as Human.

Cols 5-6 have no such problem and include ALL models, deterministic included:
λ is fitted to each source's own |Δresponse| curve, which differs between odd
and even trials because the STIMULUS sequences differ, so no response noise is
needed for the split to be informative. Models are expected to be MORE reliable
than Human in col 5 (pilot 5: Human r=0.96, RL_lambda 0.98, PrimacyRecency 0.95,
LeakyIntegrator 0.87, Mean 0.80) -- that gap is a result, not an artefact. In
col 6, Mean is retained as a NULL CONTROL (no free parameters, so it cannot
track individual differences; pilot 5 r=0.30 ns vs 0.65-0.77 for the
parameterised models).

### figure_soltani_variability.py (V group, 2x3)
| Panel | Content |
|-------|---------|
| Col 1 | KDE of prefix response variability, human only |
| Col 2 | Split-half reliability (odd/even trials, not first/second half) |
| Col 3 | Cross-task comparison (pids who did both tasks) |

Same row convention and colors quasi-qid usage as the temporal figure
above. `--datafile <name>` as above.

---

## Environment

Always use: /home/psipeter/evidence_integration/venv/bin/python

Cluster: /dartfs-hpc/rc/home/n/f007qzn/
SLURM scripts: use pwd -P and export EVIDENCE_INTEGRATION_ROOT=${ROOT}.
NFS mount uses local_lock=none. Atomic rename used for simulation DB writes.

---

## Code conventions

- alpha_0, lambda_ (trailing underscore), gamma, eps_p, eps_r
- Merge order: PARAM_DEFAULTS < _NEF_FIXED < fitted Optuna params
- Read loss with _get_loss(perf_df) — never hardcode cv_loss_mean
- Run folder: always pass short name (e.g. yoo) — resolve_run_folder prepends RUNS_DIR
- --local runs must print JOB_COMPLETE as the final stdout line
- Python 3.11; pathlib via utils.paths; figures save PDF only
- New figure panels go inside existing figure_*.py scripts
- Do not compute metrics in extras scripts — save raw data, compute in figure scripts
- pvalue_to_stars, fit_power_law_params, smooth_curve, POWER_LAW_SMOOTH_WINDOW are in utils/plot_style.py
- Aggregation for the temporal figures' error/|Δresponse| curves lives in
  utils/aggregate.py and is SHARED by soltani, yoo and carrabin. Do not
  reimplement it per figure, and do not aggregate inline in a panel — the three
  figures previously used three different schemes with nothing making that
  visible (see "Cols 1-2 aggregation"). Add flags via add_aggregate_args(parser)
  so all three document the choice identically
- Temporal curves are LINES ONLY — no markers/scatterpoints on the aggregate
  curves (cols 1-4 of soltani, 1-4 of carrabin, 2 of yoo). Cols 5-6's regplots
  keep their scatter, since there the per-pid points ARE the data

---

## Workflow guidelines

### Before making changes
1. Read the relevant files fully first.
2. Check fitting/model_params.py before touching models or fitting.
3. Propose a plan for structural changes before executing.

### NEF simulations
Never run NEF simulations via MCP tool calls — will time out.
Write a script and give the command to run on cluster.

### Figure iteration
After any figure change, render using pdftoppm and inspect with
filesystem:read_media_file. **Use this sparingly** — each image upload
consumes significant context. Prefer:
- Running analysis in shell_run_command to check numerical results first
- Only uploading an image when visual layout/style review is actually needed
- Deleting the temporary PNG immediately after inspection

Render command:
    pdftoppm -png -singlefile -r 150 figures/figure_X.pdf figures/_prev
    # then filesystem:read_media_file figures/_prev.png
    # then git clean -f figures/_prev.png

### Temporary analysis scripts
For exploratory analysis (scanning parameters, computing correlations):
- Write to scripts/_tmp_*.py
- Run via shell:run_command
- Delete immediately after: venv/bin/python -c "import pathlib; pathlib.Path('scripts/_tmp_X.py').unlink()"
- Never commit _tmp files

### Git
Generate a commit message and wait for confirmation. **The person handles all
`git commit`/`git push` themselves** (explicit instruction) -- never run
either yourself, even after generating a message and getting a verbal
"looks good."

### Context efficiency
- Prefer shell:run_command with Python -c for short computations over writing tmp files
- Use filesystem tools (read_text_file with head/tail) rather than loading full files
  when only a portion is needed
- When scanning over parameters, print a compact table rather than per-pid details
  unless the per-pid breakdown is specifically needed
- Avoid loading the full transcript for routine tasks; use conversation_search instead

---

## Workflow rules

### Suggesting vs implementing changes
When a question or observation implies that a code change *might* be warranted,
Claude must **describe the proposed change and ask for approval before writing
any code**. This applies especially to:
- Changes to figure aesthetics or panel logic (sorting, colouring, metrics)
- Changes to analysis methodology (metrics, transforms, thresholds)
- Any change that was not explicitly requested

Only implement immediately when the user has explicitly asked for a specific
change (e.g. "change X to Y", "add Z", "remove W").

---

## What NOT to do

- Do not use str_replace/create_file/view for anything under
  /home/psipeter/evidence_integration/ — they write to Claude's local sandbox,
  not this remote host, and fail silently (see compaction-reminder note at top)
- Do not add diederen, jiang, or usher back without explicit plan
- Do not reintroduce `task_continuous`/`task_binary` as dataset names, or
  `continuous`/`binary` as soltani task labels — the datasets are
  `soltani_numbers`/`soltani_colors` and the task labels are
  `numbers`/`colors` (see "Active datasets"). The one intended exception is
  `utils/binary_transform.py`'s own module/function names, which describe
  binary-valued observations generally and serve carrabin
- Do not build a `{dataset}_{datafile}` name by hand — call
  `utils.paths.dataset_stem()`. Formatting it locally is how the input pkl
  and the output filenames drift apart, which is the exact failure the
  suffix exists to prevent
- Do not add a `*`-globbed pid to a filename pattern that could see two
  dataset stems — `{model}_soltani_numbers_*_responses.pkl` also matches
  `{model}_soltani_numbers_pilot5_3_responses.pkl`. Both
  `fitting.collect._collect_responses` and `_collect_activities` drive off the
  explicit pid list from run_config.json for this reason
- Do not re-add a response transform or rescale for the soltani datasets.
  BOTH tasks ask for the MEAN of all observations, so no Laplace shrinkage
  applies (`_TRANSFORM_DATASETS` is carrabin-only) and `value`/`response` are
  already on [-1,1] in the built pkls, so `nef_obs_values` and
  `nef_response_to_model_scale` are identity. soltani_colors is untransformed
  DESPITE having {-1,+1} observations like carrabin: the observation alphabet
  is not what decides it, the quantity being reported is. Note the raw
  pre-build scale IS 0-100 (data/task_results_*.pkl, task_backend's Supabase
  output, the sequence pool) — `nef_obs_values` raises if it sees |value| >
  1.5 so that mistake fails loudly instead of saturating the ensembles
- Do not pair a counting-activity key with a different simulation seed.
  Activity entry `k` was precomputed from a network built with `seed=k`, so its
  decoders are valid ONLY for that seed's tuning curves.
  `counting_integrator.activity_key_for_trial()` returns both and is the single
  source of truth; use it for the map lookup AND for `params["seed"]`. It is
  identity for carrabin/yoo and `trial+1` for the 0-indexed soltani datasets
  (`_ZERO_INDEXED_DATASETS`). Same rule extends to the `sim` argument (added
  for NEF's NLL branch): NEVER hand-derive the `(sim-1)*n_trials + base` offset
  inline -- always call `activity_key_for_trial(dataset, trial, sim=sim)` for
  both halves of the pairing, exactly as for the trial-only case. A second,
  hand-rolled copy of that arithmetic is the same class of risk that already
  bit this exact function once (a bare `.get(trial)` silently missing
  0-indexed trial 0).
- Do not read soltani human data from `task/sequences/` — that branch was
  removed from `models/NEF.py` and the only source of human data is
  `data/soltani_*[_datafile].pkl`, built by
  `scripts/pull_soltani_data.py`
- Do not add loss_type, shape_loss, joint_loss, beta hooks
- Do not use trial_seed / base_seed for NEF — seed = int(trial) directly
- Do not read cv_loss_mean directly — use _get_loss
- Do not create scripts outside scripts/
- Do not add NEF_synaptic, LMU counting variant, or ADM model name
- Do not double-apply the carrabin transform (NoisyCounting excluded)
- Do not pass a full path as run_folder — always use a short name
- Do not commit or push without being asked
- Do not run NEF simulations through MCP tool calls (will time out)
- Do not use RNN-based sigma as the noise metric FOR SOLTANI — use qid-grouped
  response std. The rule is dataset-specific, not a general one; it holds for
  soltani and NOT for carrabin. See "RNN as a conditional-mean estimator" below
- Do not compute metrics in extras scripts — save raw data, compute in figure scripts
- Do not save figures as PNG or SVG — PDF only
- Do not upload figure images unnecessarily — use numerical checks first
- Do not promote generate_sequences_iid.py or generate_sequences_momentmatch.py
  output directly to the production {task}_sequences.{pkl,json} filenames
  without explicit go-ahead -- production is generate_sequences_hybrid.py's
  output (a deliberate per-task combination of both, chosen after PI
  discussion; see docs/HISTORY.md's "Sequence generation methods (task/)"
  section), not either pure method on its own
- Do not add a seed search / best-of-N ranking to generate_sequences_iid.py
  or generate_sequences_hybrid.py -- deliberately absent from both; any
  outcome-dependent seed selection reintroduces the conditioning/confound
  this project spent real effort establishing and then avoiding (see
  docs/HISTORY.md's "Sequence design: open questions" section)
- Do not reintroduce dev-only override knobs (testMode, nTrialsDefault,
  trialItiMs, showTutorial) into buildAndRun/timeline-builder.js/config-base.js
  — these were deliberately removed along with index-dev.html; any test-only
  need for different config values belongs in src/test-harness.js building a
  modified config object, never inside the production code path itself
- Do not redirect non-Prolific participants to a same-origin file (e.g. via
  jatos.endStudyAndRedirect) after ending a JATOS study session — confirmed
  broken on a real MindProbe pilot run ("you have no access rights" trying
  to serve public/exit-complete.html post-session-end; see docs/HISTORY.md's
  "Exit/redirect and data-saving architecture" section). Only redirect to
  an EXTERNAL domain (Prolific); for everyone else, use finish-session.js's
  DOM-update-in-place approach instead
- Do not delete generate_sequences.py or remove any of the specific
  functions it exports — it has no CLI/generation logic of its own anymore,
  but it is a genuine, live shared-utilities dependency of BOTH
  generate_sequences_iid.py and generate_sequences_momentmatch.py (check
  both scripts' imports before touching anything in this file)
- Do not reintroduce a design where one qid means one fixed (prefix, target)
  pair in generate_sequences_momentmatch.py — this was the actual mechanism
  behind a real, confirmed bug (two different qids ending up with an
  identical realized prefix by chance; see docs/HISTORY.md's "Tutorial
  redesign, bonus/error system, and binary no-prefix sequences" section,
  its "Sequence design" subsection). Prefix
  identity and target level must stay independent axes, matched via
  optimal_matching, not tied 1:1 or paired via a greedy heuristic (greedy
  was tried and rejected — see that function's docstring for the measured
  failure mode)
- Do not assume prefix_length is always > 0 when writing a NEW prefix-
  uniqueness/collision check against sequences.json data — binary's
  no-prefix branch (generate_binary_sequences_no_prefix) legitimately
  writes prefix_length=0, and values[:0] is the same empty tuple for every
  trial regardless of qid, which will false-positive as a collision
  against any check that doesn't explicitly skip this case (this bit BOTH
  generate_sequences_pool.py's verify_pool and the retired
  inspect_sequences.py's build_inspection_csv independently -- see
  docs/HISTORY.md's "Tutorial redesign..." section for both fixes; a
  third implementation of this same check would need the same guard).
- Do not treat BONUS_DECAY as a simple 0-100-scale constant without
  checking what it's actually being multiplied against — totalError is a
  SUM across N_OBS_TO_RUN observations, not a single observation's error;
  see docs/HISTORY.md's "Tutorial redesign..." section for the real bug
  this caused (reward silently 0 for nearly every real response) and why
  BONUS_DECAY is now `1 / DEFAULTS.N_OBS_TO_RUN`, not a flat literal.
