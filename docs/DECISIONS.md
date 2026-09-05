# docs/DECISIONS.md — non-diff-shaped project decisions

This file records decisions that git history can't hold: rejected
alternatives, platform/methodology evaluations, and conclusions reached
before any code existed to attach a commit to. Bug fixes and diffable
changes belong in commit messages instead (`git log --grep <term>`,
`git log --follow <file>`), not here — see CLAUDE.md's workflow rules for
that convention.

Compacted (conclusion + why, not the full diagnostic trail). Full detail
for anything below, where it existed, is preserved in `archive/HISTORY_*.md`.

---

## Online task platform: own backend (Supabase) over JATOS/Gorilla/Cognition.run/Labvanced

**Decision:** built a small custom backend (`task_backend/`, Supabase
Postgres + Edge Functions) rather than staying on JATOS or migrating to a
hosted alternative.

**Why:** two real Prolific participants hit genuine JATOS-level failures
during pilot #3 (session death mid-tutorial; a `GeneralSingle` cookie
pre-consumed by a link-prefetch scanner before the participant's first
click) — both root-caused with hard evidence, not inferred. A follow-up
Playwright simulation confirmed a third, worse gap: per-trial saves can
fail silently for an entire session with zero participant-visible symptom
until the very last click. Both jsPsych's and JATOS's own maintainers
confirm this is a documented, acknowledged limitation of the underlying
tools (jsPsych/JATOS#811), not a misconfiguration.

**Alternatives evaluated and rejected:** Pavlovia (worse — no recovery,
batch-at-end saving, costs for private data); Cognition.run (no
documented resumability advantage over JATOS); Gorilla (native
Prolific-ID-keyed resumability, real candidate, ~£170-220 for ~200
participants — but the PI pushed back on reliability grounds based on
outside reports, prompting the deeper comparison that led here instead).

**Full investigation:** `archive/HISTORY_task_legacy.md`.

---

## Sequence generation: hybrid method, not pure i.i.d. or moment-matched

**Decision:** production sequences (`generate_sequences_hybrid.py`) are a
deliberate per-task combination of i.i.d. and moment-matched generation,
chosen after PI discussion — not either pure method alone. Neither
`generate_sequences_iid.py` nor `generate_sequences_hybrid.py` gets a
seed-search/best-of-N ranking added: any outcome-dependent seed selection
reintroduces the exact conditioning/confound this project spent real
effort establishing and then avoiding.

**Why:** pure i.i.d. sequences don't give reproducible prefix/target control
for the tutorial and quota structure; pure moment-matching alone risked a
real confound (confirmed, not just suspected) between prefix identity and
target level. Prefix identity and target level are matched via
`optimal_matching` as independent axes — a greedy heuristic was tried and
rejected (measured failure mode documented in that function's own
docstring).

**Full investigation, including the empirical i.i.d.-sequence results
that triggered this:** `archive/HISTORY_task_legacy.md`.

---

## Participant exclusion: `non_integrator` criterion, `require_both_tasks` default

**Decision:** exclusion uses `non_integrator` (prior observations make no
*reliable* contribution, via nested-regression + Cohen's f² test) as the
default criterion, applied at the *subject* level — a participant failing
in either task (numbers/colors) is dropped from both.

**Why:** started from the hypothesis that the existing filter was too
aggressive (55% excluded). That hypothesis was disconfirmed — a
model-free, independent criterion (`integration`, skill vs. "copy the
latest observation") reproduced 23/25 and 18/19 of the same exclusions,
and roughly half of `numbers` participants genuinely score worse than
just reporting the latest observation. There are two distinct failure
modes (literal copying; unrelated slider-drifting), which is why no
single-axis weighting measure can catch both. `require_both_tasks`
became the default after per-task exclusion was found to collapse the
cross-task λ correlation (r=0.587→0.331) purely through a change in
*which* participants ended up in the intersection — not power, not
reliability, not range restriction.

**Rejected alternatives:** `performance` (carrabin's SD-based rule — too
lenient, retains copiers), `integration`/skill-score (not monotone in
integration depth — penalizes accurate mild-recency integrators), a
trials-8-31 burn-in (moves almost nobody), stability-across-session-halves
(penalizes genuine late learning, not fatigue — there is no fatigue in
this data; error *decreases* with trial index).

**Known limitation, stated honestly:** the criterion retains anyone using
history reliably, so it does not catch the wrong-statistic, scale-compression,
or anchored-with-a-nudge failure modes. The last of these is a real miss.

**Full investigation:** `archive/HISTORY_modeling_2026.md`.

---

## Response noise: split into two mechanisms (`sigma_state` vs `sigma_resp`)

**Decision:** `NoisyRL_lambda` models response noise as two separable
components — `sigma_state` (perturbs the integrated estimate itself,
compounds across observations → variance growth + autocorrelation) and
`sigma_resp` (perturbs only the reported value, i.i.d. → a flat plateau,
no autocorrelation) — rather than one undifferentiated noise term, at the
user's suggestion.

**Why:** this resolved an apparent RL_lambda misfit (fitted `lambda_`
mostly >0.75, several pinned at 1.0, against a descriptive lambda <0.5
measured from |Δresponse| decay). With both mechanisms present and
floored at human-calibrated minimums, NoisyRL_lambda matches the human
first/last update ratio almost exactly (2.50 vs. human 2.46) with
`alpha_0`/`lambda_` barely moved from the original fit. RMSE alone cannot
identify either sigma (both collapse toward zero), so an NLL loss was
needed to make the fit informative at all — see NLL fitting decision below.

**Known limitation:** identical noise magnitude across all pids gives
human-*scale* variability but not human individual *differences* in that
variability; per-pid `sigma_resp` is the natural next step but isn't wired
up yet (`MODEL_PARAMS` supports a fixed dict, not per-pid values).

**Update:** `NoisyRL_lambda` (the `sigma_state` side of this comparison)
was subsequently retired from active analysis entirely -- see "State-noise
models, NoisyCounting, and their MLE/NLL pipelines retired" below. The
`sigma_resp`/`add_noise` side remains active and is the only noise
mechanism currently in use.

**Full investigation:** `archive/HISTORY_modeling_2026.md`.

---

## RNN as a noise-ceiling estimator: rejected for soltani, kept for carrabin

**Decision:** `models/RNN.py`'s conditional-mean estimator is used as a
response-noise ceiling for carrabin but NOT for soltani — for soltani, use
qid-grouped response std instead. This is dataset-specific, not general.

**Why:** on soltani, the RNN's own prediction error (14-300%+ inflated
depending on settings) contaminates the noise estimate, and — more
fundamentally — the RNN is *less* accurate than the models it would be
used to evaluate, making it an inappropriate denoised target. An NLL loss
against the model's own predictive distribution makes the RNN's role here
unnecessary anyway (penalizes mean and variance mismatch jointly as a
proper scoring rule). Caveat: tested on 4 pids only; a firm decision would
want ~10, though the ordering held consistently across all four at every
setting tested.

**Full investigation:** `archive/HISTORY_modeling_2026.md`.

**Update:** `models/RNN.py` was subsequently retired entirely (moved to
`archive/models/archive_RNN.py`) -- it had fallen out of active use even
for carrabin, and the qid-grouped-std approach fully covers soltani. If
ever needed again, restore from archive.

---

## NLL fitting adopted (noise-only), shared cross-pid simulation database stays tabled

**Decision:** NEF/math-model fitting supports an NLL loss (`--loss nll`)
for models with a nonzero noise parameter, as a complement to RMSE (RMSE
can't identify a noise magnitude that collapses toward zero; NLL can, and
is a proper scoring rule for mean+variance jointly). A shared, cross-pid
Optuna simulation database (evaluated twice, independently re-derived the
same conclusion both times) is NOT adopted for NEF or the production
RMSE/NLL pipeline — it stays carrabin/NoisyCounting-only.

**Why not shared for NEF:** the caching benefit depends on sequences
*repeating* across pids/trials — true for carrabin's small repeating
pool, false for yoo/soltani's mostly-unique per-participant sequences.

**n_sims=50** was settled as NEF's NLL working default — a ballpark
informed by cheap math-model-proxy calibration (`calibrate_nll_nsims.py`,
tested against real human data, checking whether independent Monte Carlo
reps agree rather than recovering a known synthetic truth), not a
directly-measured NEF number. The state-noise mechanism (`NoisyRL_lambda`)
needed up to n_sims=40-320 to stabilize depending on pid, against a
response-noise-only model's stable n_sims=10 — NEF's own recurrent
dynamics are structurally closer to the state-noise case, so this is the
conservative choice.

**Full investigation:** `archive/HISTORY_modeling_2026.md`.

**Update:** NEF's own NLL/multi-seed-ensemble branch (`NEF.simulate_ensemble`,
`NEF_DEFAULT_N_SIMS`) was subsequently retired -- too expensive to run at
the scale this project needs. `scripts/calibrate_nll_nsims.py` (the
calibration tool referenced above) is archived alongside it. NLL fitting
remains active, but only for the `<model>_resp_noise` wrapper models --
see "State-noise models, NoisyCounting, and their MLE/NLL pipelines
retired" below.

---

## NEF network size: n_neurons=500 for all datasets, n_neurons_counting split by dataset

**Decision:** all four datasets (carrabin/yoo/soltani_numbers/soltani_colors)
fit at `n_neurons=500`, explicitly erring toward more neurons than a
precisely-justified minimum. `n_neurons_counting` is NOT uniform: 2000 for
yoo/soltani, but 500 (not 2000) for carrabin.

**Why:** counting-activity file size scales with `n_neurons_counting² ×
precomputed trial-seeds`. Carrabin precomputes 200 trial-seeds against
yoo/soltani's 30-40, so `nc=2000` for carrabin would cost ~6.4GB against
~1-1.3GB for the others at the same setting. `nc=500` for carrabin also
reused a file already on disk. This is set via `fitting/model_params.py`'s
NEF `fixed` dicts — the only mechanism controlling submit-time network
size (`fitting.fit`/`fitting.submit` have no CLI override for it).

**Full investigation:** `archive/HISTORY_modeling_2026.md`.

---

## soltani math-model fits: separate `rmse/`/`nll/` folders, not the older shared `soltani/`

**Decision:** RMSE and NLL fits for soltani_numbers/soltani_colors write
to `data/runs/rmse/` and `data/runs/nll/` respectively -- not the older
`data/runs/soltani/`, which is now read by no current figure.

**Why:** `data/runs/soltani/` held fits made against an earlier,
contaminated/smaller-pid-count build of the data (from before the pid
registry existed and before a stale pilot-4 contamination was found and
removed -- see the pid-registry section of
`archive/HISTORY_modeling_2026.md`). Keeping old fits in a
differently-named folder rather than overwriting them in place meant the
contamination fix couldn't silently corrupt a folder that figures might
still be pointed at, and made the stale folder trivially avoidable going
forward rather than requiring every caller to somehow know which fits
within one shared folder predate the fix.

**Full investigation:** `archive/HISTORY_modeling_2026.md`.

---

## State-noise models, NoisyCounting, and their MLE/NLL pipelines retired from active analysis

**Decision:** `NoisyRL_lambda` (compounding state noise) and
`NoisyCounting` (carrabin's task-specific model, Prat-Carrabin 2024) are
retired from active analysis. Along with them: the MLE fitting pipeline
built for `NoisyCounting` (`fitting/fit_mle.py`, `jobs/submit_mle_fit.sh`,
`MLE_PARAMS`/`NEF_N_NEURONS_VALUES`), and NEF's own NLL/multi-seed-ensemble
branch (`NEF.simulate_ensemble`, too expensive to run at scale).

**Why:** the project's active analysis narrows to the RMSE-fit model set
(Mean, LeakyIntegrator, PrimacyRecency, RL_lambda, NEF) plus the still-
active i.i.d.-response-noise wrapper (`<model>_resp_noise` via
`add_noise()`). The state-noise/task-specific-noise line of work (see the
"Response noise" and "RNN" entries above) had already been superseded in
practice; this makes that explicit and removes the code so it can't drift
out of sync with a codebase that no longer exercises it.

**What's unaffected:** every currently-published figure reads pre-computed
`.pkl` files under `data/runs/` and continues to work unchanged --
figure scripts read cached output by path, they don't import the retired
model classes directly. This retirement only removes the ability to
generate *new* fits of these models.

**Where the code went:** `archive/models/archive_math_models_noise.py`
(NoisyCounting, NoisyRL_lambda, their shared `simulate_ensemble`),
`archive/models/archive_NEF_simulate_ensemble.py` (NEF's NLL branch),
`archive/fitting/archive_fit_mle.py`, `archive/fitting/
archive_model_params_retired.py`, `archive/scripts/
archive_calibrate_nll_nsims.py`, `archive/models/archive_RNN.py`. All
restorable by merging back in -- each archive file's own header comment
says exactly what to reconnect.

**Cleanup completed 2026-09-05:** the initial pass above left
`scripts/build_sim_db.py` and three MLE-only collection functions in the
still-active `fitting/collect.py`. Both are now archived too --
`archive/scripts/build_sim_db.py` (whole-file move; a distinct, older
prototype already occupying that path was preserved alongside it as
`archive/scripts/build_sim_db_early_draft.py`) and `archive/fitting/
archive_collect_mle.py` (extracted from `fitting/collect.py`, which
remains active for its RMSE/NLL `params`/`responses`/`activities`
branches). Full narrative: `archive/HISTORY_modeling_2026.md`'s
"MLE-pipeline retirement completed" entry.

**Fully done as of 2026-09-05:** the one function left out of scope by
that pass, `fitting/losses.py`'s `compute_sim_db_loss` (zero active
callers, only the two archived MLE files), is now archived too --
`archive/fitting/archive_losses_mle.py`. See `archive/HISTORY_modeling_2026.md`'s
"MLE-pipeline retirement, final loose end" entry.

---

## neural_main replaces neural_giant as the sole neural-parameter-impact figure

**Decision:** `neural_giant` (a 3×4 figure covering α₀/λ/n_neurons impact
via random-virtual-pid covariation) was retired in favor of `neural_main`,
which isolates each parameter's own causal contribution one row at a time
by sweeping it while holding the others fixed.

**Why:** covariation-based random draws can't distinguish a parameter's
own causal effect from confounded co-variation with the others that a
one-parameter-at-a-time sweep design isolates directly. See
`docs/SCIENCE.md`'s neural predictions section for the current figure's
full structure and status — this entry exists only to record that the
switch happened and why.

**Full investigation:** `archive/HISTORY_modeling_2026.md`.

---

## Legacy per-dataset figure scripts retired in favor of make_paper_figures.py

**Decision:** the split per-dataset P/V/T/N figure scripts
(`figure_{carrabin,yoo,soltani}_{performance,variability,temporal,neural}.py`),
the two older legacy combined figures (`figure_carrabin.py`,
`figure_yoo.py`), and the neural data-generation scripts that fed the old
N1-N8 taxonomy (`extras_carrabin.py`, `extras_yoo.py`, plus the job
scripts that only ever invoked them) are retired from active use, in
favor of `scripts/make_paper_figures.py`'s consolidated `make_*`
functions.

**Why:** a thorough review compared every panel/function in the retired
scripts against `make_paper_figures.py`'s current `make_*` functions and
found each one either already has a newer equivalent there, or computed a
metric deliberately dropped from the current figure set. The N1-N8
per-task neural taxonomy specifically (`figure_carrabin_neural.py`/
`figure_yoo_neural.py`) only ever covered HALF the neural story per task
(carrabin has a real fitted sigma but no fitted lambda; yoo the reverse)
-- exactly the gap closed by `neural_main`'s soltani-only design, which
runs on one task with both real fits in hand. The project owner confirmed
this comparison against the rendered figures themselves before
archiving.

**Full investigation:** `archive/HISTORY_modeling_2026.md`.

