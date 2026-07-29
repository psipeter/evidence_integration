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

---

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

Pickles: data/carrabin.pkl, data/yoo.pkl.
Required columns: pid, trial, observation, value, response.
Carrabin adds: qid, true_p (from carrabin_original.csv).

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
**Full incident investigation and platform evaluation: `docs/HISTORY.md`**
("Pilot #3 real-participant incidents..." and "Own-backend decision
(Supabase)" sections). **Current build status and open items:
`task_backend/TODO.md`** -- that file is the actively-maintained build
log; this section only documents what's true right now.

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

**`iti_condition`/distractor system**: still generated (`'control'`/
`'distract'`) and still present in the plugin code
(`plugin-iti-clock.js`'s popup-spawning logic), but currently **inert**
-- `config-base.js`'s `DISTRACTOR_TYPE` defaults to `'none'` in both
tasks' configs.

### Scoring

`scoring.js` (shared by both tasks): `normError = rawError / MAX_POSSIBLE_ERROR;
reward = max(0, MAX_REWARD * (1 - BONUS_DECAY * normError))`, computed per
observation and summed for the trial/tutorial total. Current parameters:
`MAX_REWARD = 2` cents, `BONUS_DECAY = 15`, `MAX_POSSIBLE_ERROR = 100`.
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

**Prolific cutover status: NOT done as of the last check.** `supabase/
functions/_shared/prolific-codes.ts` deliberately mirrors the OLD JATOS
pipeline's exact completion/early-exit codes, meaning the cutover reduces
to a single manual step per task in Prolific's own dashboard (updating
the Study URL field to `https://psipeter.github.io/evidence_integration/
index-{numbers,colors}.html?PROLIFIC_PID={{%PROLIFIC_PID%}}` -- no
`STUDY_ID`/`SESSION_ID` needed, unlike the old JATOS URL) -- no new
Prolific studies required. Confirmed via the live database that zero real
participant traffic has reached task_backend yet.

### Data pipeline (deferred on purpose)

No path yet from the `events` table back into the `fitting/`/`models/`
analysis pipeline -- deliberately deferred until real data actually
exists (even test data) to build against, at which point participant-
data anonymization should be built as part of that same export step. See
`task_backend/TODO.md`'s "Post-buildout review" section.

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

NoisyCounting applies to carrabin only. Two fitted versions:
- RMSE-fitted: sigma_c collapses to ~0 (response-noise artefact; methodologically revealing)
- MLE-fitted (fit_mle.py): recovers sigma_c ~0.03-0.08, nu ~0.08-0.21

RNN (models/RNN.py): retained for reference; not used in active figures.

---

## NEF architecture

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

Trial-to-trial variability in neural tuning curves (controlled by `seed =
int(trial)`) is the primary spiking noise source, producing state-persistent
response variability across observations within a trial.

Activity files are loaded at fit time for speed (fast_decode mode). Generate
locally with counting_integrator.py then scp to the cluster before submitting
fitting jobs (see Simulation pipeline below).

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

Run folders: data/runs/carrabin/, data/runs/yoo/, data/runs/refit/
The --nef_folder flag in figure scripts redirects NEF data to a separate folder
(e.g. --run_folder yoo --nef_folder refit uses yoo for other models, refit for NEF).

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
    paths.py
    plot_style.py    — apply_style, get_palette, pvalue_to_stars, fit_power_law_params
    slurm.py
    carrabin_transform.py
    save_responses.py
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
    figure_soltani_performance.py    — task_backend pilot data vs. models (P group)
    figure_soltani_temporal.py       — task_backend pilot data vs. models (T group)
    figure_soltani_variability.py    — task_backend pilot data vs. models (V group)
    build_model_inputs.py            — builds data/task_{continuous,binary}.pkl for figure_soltani_*.py
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
- Do not add loss_type, shape_loss, joint_loss, beta hooks
- Do not use trial_seed / base_seed for NEF — seed = int(trial) directly
- Do not read cv_loss_mean directly — use _get_loss
- Do not create scripts outside scripts/
- Do not add NEF_synaptic, LMU counting variant, or ADM model name
- Do not double-apply the carrabin transform (NoisyCounting excluded)
- Do not pass a full path as run_folder — always use a short name
- Do not commit or push without being asked
- Do not run NEF simulations through MCP tool calls (will time out)
- Do not use RNN-based sigma as the noise metric — use qid-grouped response std
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
