# CLAUDE.md — evidence_integration

Source of truth for conventions and workflow. For scientific goals,
current work, and results (including NEF architecture and the neural
predictions figure), see `docs/SCIENCE.md`. For why past methodology
choices were made (including retired models), see `docs/DECISIONS.md`.
For task_backend specifics (including participant exclusion and
sequence-generation diagnostics), see `task_backend/CLAUDE.md` (loads
automatically when working there). For exact CLI recipes (data pulls,
fitting jobs), see `.claude/skills/` — these auto-surface via Claude
Code's Skill tool based on task match; invoke by name or let them load
implicitly. Read this file fully before making changes.

---

## Session-start checklist

- **Cluster reach:** Claude's tools (Bash included) only reach `hydra`,
  not `discovery-01` (or whichever node cluster jobs actually run
  on/from, under a different username there). Claude cannot verify
  cluster-side file/job state directly — give the person commands to run
  themselves on `discovery-01` and take their terminal output as ground
  truth.
- **Figures save PDF only** — never PNG/SVG, never upload images to chat
  unless a genuine visual judgment call is needed (see "Figure
  iteration" below). Don't upload Playwright screenshots either — use
  DOM/computed-style assertions instead.
- **NEF simulation runtime varies from minutes to hours.** Never run one
  directly — write the script, then give the person the exact command
  so they can judge expected runtime themselves before running it (on
  cluster or locally).
- All NEF simulation data → `data/runs/`; figures → `figures/`.
- **Never let a NEF/counting-integrator simulation silently fall back to
  a live `_pretrain()` training run** when a precomputed counting-activity
  file (or a seed's key within it) is missing. Always load via
  `load_activities()`/`fast_decode()` and RAISE with the exact regenerate
  command if missing — `models.NEF`'s own `_require_activity_map`
  convention, used by `scripts/neural_experiments.py`'s
  `_require_activities()`/`_decoders_for_seed()`. This has bitten the
  project before (see `docs/DECISIONS.md` and `archive/HISTORY_modeling_2026.md`).
- **Never pair a counting-activity key with a different simulation
  seed.** `counting_integrator.activity_key_for_trial(dataset, trial,
  sim=sim)` is the single source of truth for BOTH the activity-map
  lookup and the simulation seed — never hand-derive the offset for
  either half separately.
- **Never let `models.math_models.add_noise` fall back to a bare
  `params.get("seed", 0)` default** — every pid/model would silently
  share the identical noise draw. Resolve through
  `_resp_noise_seed(pid, model_type)` (now `add_noise`'s own built-in
  behavior; don't add a new call site that reimplements a seed default).

---

## Active datasets

| Name | N | Task |
|------|---|------|
| carrabin | 21 | Binary inputs; slider after each of 5 obs; sequences repeat (qid); true_p known |
| yoo | 38 | Continuous inputs; slider; 30 obs × 30 trials; no sequence repetition |
| soltani_numbers | live | Our own task_backend `numbers` task; 32 trials × 15 obs |
| soltani_colors | live | Our own task_backend `colors` task; 32 trials × 15 obs |

Pickles: `data/carrabin.pkl`, `data/yoo.pkl`,
`data/soltani_{numbers,colors}[_<datafile>].pkl`. Required columns: pid,
trial, observation, value, response. Carrabin adds: qid, true_p. soltani_*
add: qid, plus true_mean (numbers) / true_p (colors).

**soltani naming**: `soltani_numbers`/`soltani_colors` — NOT
`task_continuous`/`task_binary` (the retired `task/` pipeline's naming).
One deliberate exception: `utils/binary_transform.py`'s own
module/function names describe binary-valued observations generally and
serve **carrabin**, not only soltani_colors.

**soltani_\* are 0-INDEXED** on both trial (0-31) and observation (0-14),
unlike carrabin/yoo (1-indexed). Anything assuming 1-indexed observation
needs an explicit guard — already handled in
`counting_integrator.activity_key_for_trial`, `_fit_lambda_curve_fit`'s
`n = observation + 1`, and `apply_binary_transform`'s `t = observation + 1`.

`scripts/pull_soltani_data.py` flags (which exclusion method, how pids
are kept stable across pulls): see `.claude/skills/data-pipeline/SKILL.md`.

Archived (do not reactivate): diederen, jiang, usher.

---

## Central cognitive model

Updates follow a power-law decaying learning rate:
`alpha(t) = alpha_0 / t^lambda`. See `docs/SCIENCE.md` for the full
scientific framing (goals, current thread, results, NEF architecture).

---

## Active models

| Model | Role | Free params |
|-------|------|-------------|
| Mean | Optimal running mean (Bayesian baseline) | none |
| LeakyIntegrator | Exponential forgetting baseline | gamma |
| PrimacyRecency | Temporal weighting (primacy + recency terms) | eps_p, eps_r |
| RL_lambda | Power-law delta rule (explicit equation) | alpha_0, lambda_ |
| NEF | Spiking NEF integrator (emergent power-law dynamics); RMSE-fit only | alpha_0, lambda_ |
| `{Mean,LeakyIntegrator,PrimacyRecency,RL_lambda}_resp_noise` | base model + i.i.d. RESPONSE noise via `add_noise()`; all 4 datasets | base params + sigma_resp |

**Retired from active analysis** (code archived, restorable): see
`docs/DECISIONS.md`. Doesn't affect the models in the table above or any
currently-published figure.

### Response noise and NLL fitting

The only active noise mechanism is `add_noise()`/`_resp_noise` — see
`docs/SCIENCE.md`'s "Response noise mechanism" for the formula and
`docs/DECISIONS.md` for why the compounding-state-noise alternative was
retired. `--loss nll` on `fitting.fit` applies only to `<model>_resp_noise`
names now (NEF's and the retired state-noise model's own NLL branches are
gone) — for exact CLI mechanics (filename conventions, the noise-only
override default, when to run `verify_ensemble_invariant.py`), see
`.claude/skills/fitting-pipeline/SKILL.md`.

---

## NEF architecture

See `docs/SCIENCE.md`'s "NEF architecture" section for the three-population
description and current network sizes. Operationally: activity files load
at fit time via `fast_decode`; generate locally with
`counting_integrator.py`, then scp to the cluster before submitting fitting
jobs (see "Simulation pipeline" below). Sizes are set via
`fitting/model_params.py`'s NEF `fixed` dicts — the only mechanism
controlling submit-time network size (no CLI override exists).

---

## Carrabin response transform

All active carrabin models apply: `response = raw * t/(t+2)`. Implemented
in `utils/carrabin_transform.py`. Never apply it twice.

---

## Fitting pipeline

For exact submit/collect commands (RMSE and NLL, any dataset, run-folder
conventions), see `.claude/skills/fitting-pipeline/SKILL.md`.

### `dataset` vs `--datafile` (the decoupling)

`dataset` is the model-FAMILY key (indexes `MODEL_PARAMS`, selects the
`math_models` branch, keys `binary_transform`, names NEF activity files).
`--datafile` selects WHICH BUILD of that family's human data:

```
data/{dataset}_{datafile}.pkl                     # input
{model_type}_{dataset}_{datafile}_{pid}_*.pkl     # every output
```

`utils.paths.dataset_stem(dataset, datafile)` is the single source of
truth for this combination — always use it, never format the name
locally (this is how an input pkl and its output filenames drift apart).
A new round of data needs no new model plumbing, just a new pkl.

---

## Simulation pipeline (extra data for figure scripts)

For generating counting activity files or neural predictions figure
(`neural_main`) data, see `.claude/skills/neural-simulation-pipeline/SKILL.md`.
Always generate locally (or via cluster if slow), then scp to the
cluster. Runtime varies minutes-to-hours — write the script, give the
person the exact command, let them run it themselves.

---

## Repository structure

```
evidence_integration/
  data/
    carrabin.pkl, carrabin_original.csv, yoo.pkl
    counting_activities_n{n}_nc{nc}_{dataset}.pkl
    soltani_{numbers,colors}[_<datafile>].pkl
    runs/
      carrabin/, yoo/, refit/, rmse/, nll/, nll_noise_only/
  models/
    math_models.py, NEF.py, counting_integrator.py
  fitting/
    fit.py           — Optuna k-fold CV RMSE/NLL
    model_params.py   — MODEL_PARAMS
    submit.py, collect.py, losses.py
  utils/
    aggregate.py       — SHARED aggregation for all temporal figures
    paths.py, plot_style.py, slurm.py, carrabin_transform.py, save_responses.py
    pid_registry.py     — persistent prolific_pid -> anonymized pid registry
    participant_filters.py  — exclusion criteria (see task_backend/CLAUDE.md)
    colors_quasi_qids.py    — empirically-derived repeat structure for colors
  scripts/
    figure_{carrabin,yoo,soltani}_{performance,variability,temporal,neural}.py
    make_paper_figures.py     — composite/presentation figures (incl. neural_main)
    build_model_inputs.py, pull_soltani_data.py
    inspect_participant.py, inspect_participant_temporal.py
    plot_sequences.py         — see .claude/skills/task-backend-sequences/SKILL.md
    extras_carrabin.py, extras_yoo.py, neural_experiments.py
    verify_ensemble_invariant.py, check_NEF_pipeline.py
  jobs/
    submit_probe_pids.sh, submit_n_neurons_scan.sh, submit_yoo_noise.sh
  task_backend/        — online task (see task_backend/CLAUDE.md)
  archive/              — retired code + frozen history (incl. archive/task/,
                          the fully-retired legacy JATOS/MindProbe pipeline;
                          see archive/HISTORY_task_legacy.md)
  docs/
    SCIENCE.md          — scientific goals, current thread, results, NEF architecture
    DECISIONS.md        — non-diff-shaped methodology/platform decisions
  venv/
```

All new scripts go in `scripts/`. Never create scripts at the project
root. Figures save PDF only.

---

## Environment

Always use: `/home/psipeter/evidence_integration/venv/bin/python`

Cluster: `/dartfs-hpc/rc/home/n/f007qzn/`
SLURM scripts: use `pwd -P` and export `EVIDENCE_INTEGRATION_ROOT=${ROOT}`.
NFS mount uses `local_lock=none`. Atomic rename used for simulation DB writes.

---

## Code conventions

- `alpha_0`, `lambda_` (trailing underscore), `gamma`, `eps_p`, `eps_r`
- Merge order: `PARAM_DEFAULTS < _NEF_FIXED < fitted Optuna params`
- Read loss with `_get_loss(perf_df)` — never hardcode `cv_loss_mean`
- Run folder: always pass a short name (e.g. `yoo`) — `resolve_run_folder`
  prepends `RUNS_DIR`
- `--local` runs must print `JOB_COMPLETE` as the final stdout line
- Python 3.11; pathlib via `utils.paths`; figures save PDF only
- New figure panels go inside existing `figure_*.py` scripts
- Do not compute metrics in extras scripts — save raw data, compute in
  figure scripts
- `pvalue_to_stars`, `fit_power_law_params`, `smooth_curve`,
  `POWER_LAW_SMOOTH_WINDOW` are in `utils/plot_style.py`
- Aggregation for temporal figures' error/|Δresponse| curves lives in
  `utils/aggregate.py`, shared by soltani/yoo/carrabin — do not
  reimplement per figure or aggregate inline in a panel. Add flags via
  `add_aggregate_args(parser)` so all three document the choice identically
- Temporal curves are LINES ONLY — no markers on aggregate curves. Regplot
  columns (per-pid individual-differences panels) keep their scatter,
  since there the points ARE the data

---

## Workflow guidelines

### Before making changes
1. Read the relevant files fully first.
2. Check `fitting/model_params.py` before touching models or fitting.
3. Propose a plan for structural changes before executing.

### Suggesting vs. implementing changes
When a question or observation implies a code change *might* be
warranted, describe the proposed change and ask for approval before
writing code. This applies especially to figure aesthetics/panel logic,
analysis methodology (metrics, transforms, thresholds), and any change
not explicitly requested. Only implement immediately when explicitly
asked for a specific change ("change X to Y", "add Z", "remove W").

### Figure iteration
After any figure change, render via `pdftoppm` and inspect with the
`Read` tool — sparingly, since each image read costs context. Prefer
running analysis via Bash to check numerical results first; only read
an image when visual layout/style review is genuinely needed; delete
the temporary PNG immediately after.

```bash
pdftoppm -png -singlefile -r 150 figures/figure_X.pdf figures/_prev
# then Read figures/_prev.png
# then git clean -f figures/_prev.png
```

### Temporary analysis scripts
For exploratory analysis: write to `scripts/_tmp_*.py`, run via Bash,
delete immediately after (`rm scripts/_tmp_X.py`). Never commit `_tmp`
files.

### Git and commit messages
The person handles all `git commit`/`git push` themselves — never run
either, even after generating a message and getting a verbal "looks
good." Generate a commit message and wait for confirmation.

**Write thorough commit messages for anything diffable** (a bug found and
fixed, a design change to code) — the commit body is the primary home
for that narrative going forward, findable later via `git log --grep
<term>` or `git log --follow <file>`, not a standing doc. Reserve
`docs/DECISIONS.md` for decisions git can't hold: rejected alternatives,
platform evaluations, methodology choices made before any code existed.

### Context efficiency
- Prefer Bash with `python -c` for short computations over writing tmp
  files
- Use `Read`'s `offset`/`limit` rather than loading full files when only
  a portion is needed — conserves context tokens, not working around a
  hard size limit
- When scanning over parameters, print a compact table rather than
  per-pid details unless specifically needed

---

## What NOT to do

- Do not add diederen, jiang, or usher back without an explicit plan.
- Do not reintroduce `NoisyCounting`, `NoisyRL_lambda`, the MLE fitting
  pipeline, `models/RNN.py`, or NEF's NLL/multi-seed-ensemble branch
  without an explicit plan — all retired, see `docs/DECISIONS.md`. Code
  is archived (restorable), not deleted.
- Do not resurrect the `task/` (JATOS/MindProbe) online-task pipeline
  without an explicit plan — fully superseded by `task_backend/`
  (Supabase-backed, in production with real published data) and now
  fully retired: its code lives at `archive/task/` (restorable), but its
  raw participant data (`dev-results/`, `pilot1-3/`) and reproducible
  build artifacts (`node_modules/`, `dist*/`, `sequences_pool*/`) were
  deleted, not archived — see `archive/HISTORY_task_legacy.md`.
- Do not reintroduce `task_continuous`/`task_binary` as dataset names, or
  `continuous`/`binary` as soltani task labels (see "Active datasets").
- Do not build a `{dataset}_{datafile}` name by hand — call
  `utils.paths.dataset_stem()`.
- Do not add a `*`-globbed pid to a filename pattern that could see two
  dataset stems — drive filenames off the explicit pid list in
  `run_config.json` instead.
- Do not re-add a response transform or rescale for the soltani
  datasets — both tasks report the MEAN of all observations (no Laplace
  shrinkage), and `value`/`response` are already on [-1,1] in the built
  pkls. The raw pre-build scale IS 0-100; `nef_obs_values` raises if it
  sees `|value| > 1.5` so that mistake fails loudly.
- Do not pair a counting-activity key with a different simulation seed
  (see checklist above).
- Do not read soltani human data from `task/sequences/` (the retired
  JATOS/MindProbe pipeline, now fully archived under `archive/task/`) —
  the only source is `data/soltani_*[_datafile].pkl`, built by
  `scripts/pull_soltani_data.py`.
- Do not add `loss_type`, `shape_loss`, `joint_loss`, `beta` hooks.
- Do not let `models.math_models.add_noise` fall back to a bare seed
  default (see checklist above).
- Do not use `trial_seed`/`base_seed` for NEF — `seed = int(trial)` directly.
- Do not read `cv_loss_mean` directly — use `_get_loss`.
- Do not create scripts outside `scripts/`.
- Do not add `NEF_synaptic`, an LMU counting variant, or an `ADM` model name.
- Do not double-apply the carrabin transform.
- Do not pass a full path as `run_folder` — always a short name.
- Do not commit or push without being asked.
- Do not run NEF simulations directly — runtime varies minutes-to-hours;
  hand the person the exact command and let them judge when to run it.
- Do not use RNN-based sigma as a noise metric — use qid-grouped response
  std for soltani (the RNN estimator itself is retired; see
  `docs/DECISIONS.md`).
- Do not compute metrics in extras scripts — save raw data, compute in
  figure scripts.
- Do not save figures as PNG or SVG — PDF only.
- Do not upload figure images unnecessarily — use numerical checks first.
- Do not promote `generate_sequences_iid.py`/`generate_sequences_momentmatch.py`
  output directly to production filenames without explicit go-ahead —
  production is `generate_sequences_hybrid.py`'s output (why:
  `docs/DECISIONS.md`).
- Do not add a seed search/best-of-N ranking to sequence generation —
  reintroduces a conditioning confound (why: `docs/DECISIONS.md`).
- Do not reintroduce dev-only override knobs (`testMode`,
  `nTrialsDefault`, `trialItiMs`, `showTutorial`) into
  `buildAndRun`/`timeline-builder.js`/`config-base.js` — any test-only
  need belongs in `src/test-harness.js` building a modified config
  object.
- Do not redirect non-Prolific participants to a same-origin file after
  ending a session — only redirect to an external domain (Prolific); use
  `finish-session.js`'s DOM-update-in-place approach otherwise.
- **When extending any model to a new dataset**, watch for two failure
  classes that have bitten this project before, invisible to
  `py_compile` and to exercising branches in isolation:
  1. An unguarded string-replace on a per-dataset anchor
     (`_run_carrabin`/`_run_yoo`/`_run_soltani_common`-style dispatch)
     can silently duplicate a branch into datasets it was never intended
     for.
  2. Code that labels per-observation arrays must use the dataset's REAL
     observation values, not a synthetic `range(n_obs)` — harmless for
     0-indexed soltani, wrong for 1-indexed carrabin (this is the same
     class of bug as the 0-indexing bugs that made `models/RNN.py`
     silently unusable on soltani before it was retired).
