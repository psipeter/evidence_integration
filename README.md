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
| numbers | TBD | Continuous inputs; 15 obs/trial; Normal(mean, std); 8x4=32 trials, per-participant pool of 200 | **Piloting** (task_backend, cut over from JATOS -- two real Prolific pilot rounds run so far; see "task_backend" section below) |
| colors | TBD | Binary inputs (blue/red); 15 obs/trial; Bernoulli(p); 32 trials/participant, per-participant pool of 200 | **Piloting** (task_backend, cut over from JATOS -- one real Prolific pilot round run so far; see "task_backend" section below) |

numbers and colors are designed to be completed within-subject
(same participants recruited via Prolific allowlist). Together they unlock all
PTN metrics simultaneously and enable cross-task individual-differences analysis
(same pid's λ across both task types). See "task_backend" section below for details.

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

## task_backend — Online Experiment

Two online experiments -- **numbers** (continuous, Normal(mean,std)
stimulus, slider response [0-100]) and **colors** (binary, Bernoulli(p)
blue/red stimulus, slider response [0-100%]) -- 8x4=32 trials x 15
observations per task, deployed as a single jsPsych 8 + Vite 6 web app,
backed by Supabase (Postgres + Edge Functions), hosted on GitHub Pages.
Each participant is assigned ONE of 200 independently-generated sequence
sets per task (a per-participant pool, not one shared file), via a
deterministic hash of their Prolific ID -- same pool index in both tasks.

Live at `https://psipeter.github.io/evidence_integration/index-{numbers,colors}.html`.
Full design rationale, the pilot #3 JATOS incident investigation that
motivated building this at all, and the entire build-out/pilot history
since: **`docs/HISTORY.md`**'s own "task_backend: build history and
settled decisions" section (folded in from the now-retired
`task_backend/TODO.md` once the initial build-out settled). This section
only covers current, stable facts.

### Backend

- **`events` table** (Supabase Postgres): one append-only table, every
  row a checkpoint (tutorial/trial observation) or a bookkeeping/terminal
  marker (welcome/consent/finished/terminated). Idempotency via
  `unique (prolific_pid, task, phase, trial_index, observation_index, attempt)`.
  RLS enabled with zero policies for `anon`/`authenticated` -- the three
  Edge Functions below (using the service-role key server-side) are the
  only way in or out.
- **`progress-check`** / **`progress-append`** / **`progress-finish`**
  (Edge Functions) -- resume-check before building the timeline, per-
  checkpoint append (with a client-side consecutive-failure warning
  banner, not silent fire-and-forget), and session-finish (hands back the
  completion code as visible text, not just a redirect param).
- Resume granularity is **trial-boundary**, not exact-observation --
  reloading mid-trial restarts that trial (cheap, 15 observations), not
  the exact spot; every observation is still logged individually.

### Sequences

`task_backend/generate_sequences.py` -- one consolidated script (replaced
`task/`'s five-script iid/momentmatch/hybrid debate, see
`docs/HISTORY.md`), with its own `verify_numbers_trials`/
`verify_colors_trials` asserts run at generation time. Output:
`sequences_numbers.json`/`sequences_colors.json`, each a plain JSON array
of 200 pool members (no `.pkl`). Small test variants via `--name
<suffix>` + `VITE_SEQUENCES_VARIANT` client-side (see `docs/HISTORY.md`'s
task_backend section, "Small-sequence test variants").

### Scoring

`scoring.js` (shared by both tasks): per-observation reward =
`max(0, MAX_REWARD * (1 - bonusDecay * normError))`, summed per trial.
Current: `MAX_REWARD = 2` cents; `bonusDecay` split per task since a
single shared value stopped being comparable once numbers' own
`std_fixed` changed pilot-to-pilot -- `NUMBERS_BONUS_DECAY = 25`,
`COLORS_BONUS_DECAY = 15` (colors unchanged). `ERROR_MODE` scores
against the RUNNING mean/ratio of observed values (`'running_mean'`
numbers / `'running_p'` colors), not the fixed generative parameter -- a
confirmed, deliberate methodological choice for production, not a
leftover test setting (see `docs/HISTORY.md`).

### Commands

```bash
cd task_backend
npm install                    # first time only
npm run dev:numbers            # local dev server, opens index-numbers.html
npm run dev:colors             # local dev server, opens index-colors.html
npm run build                  # production build -> dist/ (BOTH tasks, one combined build)
npm test                       # Playwright suite (see Testing below)
```

### Testing

4 spec files (`task_backend/tests/`), run against the real deployed
Supabase backend, not mocked -- `happy-path.spec.mjs` (one canonical
full-session traversal per task), `resume.spec.mjs`, `timeout-retry.spec.mjs`,
`completion-screens.spec.mjs`. **The person running this project now runs
tests themselves** -- Claude should give exact commands rather than
running them directly. A single `npx playwright test` call can exceed a
tool's own response-window timeout even though the suite runs fine on the
actual host (confirmed directly) -- if ever run programmatically, use a
background+poll pattern, not one blocking call.

```bash
cd task_backend
lsof -ti:5183 -ti:5184 | xargs -r kill -9   # clear any stale test servers first
npx playwright test
```

### Deployment / Prolific cutover

Deployed via `.github/workflows/deploy-task-backend.yml` (GitHub Actions,
path-filtered to `task_backend/**`). **Prolific cutover: done.** Real
Prolific traffic has run against task_backend directly (not JATOS) --
two pilot rounds for numbers (5 participants at `std_fixed=15`, then 8+
at `std_fixed=10`) and one for colors (the same 5 participants from
numbers' first round, both tasks). The backend's completion/early-exit
codes deliberately mirror the OLD JATOS pipeline's exact codes, so the
cutover itself was just updating each existing Prolific study's Study URL
field to point at the GitHub Pages URL above.

### Data pipeline

`scripts/build_task_backend_inputs.py` pulls real, finished participants
directly from Supabase for an explicit pid list per pilot round (not
"everyone finished so far" -- different pilots are different people with
different generative parameters) into `data/task_continuous_<name>.pkl`/
`task_binary_<name>.pkl`, via the same shared filter/rescale/anonymize/
save pipeline (`build_model_inputs.py`'s `build_from_df()`) carrabin/yoo
already use. `figure_soltani_*.py` take a `--datafile <name>` argument
pointing at these. Human-data-only for now -- model fitting against real
task_backend data hasn't been run yet. See `CLAUDE.md`'s own "Data
pipeline" section for the full detail.

---

## Legacy: task/ (retired)

The original JATOS/MindProbe-hosted online task (continuous/binary
naming). Superseded by task_backend above; remains on disk for
historical reference. Full design history, every directory listing,
exact command, deployment checklist, and Prolific rollout detail that
used to live in this section: **`docs/HISTORY.md`**.
