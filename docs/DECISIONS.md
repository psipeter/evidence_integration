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

**Last loose end closed 2026-09-06:** `NoisyRL_lambda`'s own model code was
retired above, but it was explicitly left in place as a figure-level
stand-in (colors/numbers hadn't had NEF fit yet). NEF has since been fit
for those two tasks; see "NoisyRL_lambda retired as the colors/numbers
stochastic stand-in, NEF takes its place" below for that final cleanup.

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

## `init_from_obs1` kept for LeakyIntegrator, dropped for RL_lambda; RL_lambda's `lambda_` bound widened to 2.0

**Decision:** `init_from_obs1` (initialize LeakyIntegrator's/RL_lambda's
persistent state at that trial's own raw first observation instead of
0.0, for soltani_colors/soltani_numbers only) is a `fitting.fit`/
`fitting.submit` CLI flag (`--init_from_obs1 true`), not a `MODEL_PARAMS`
default — so it's just a choice of which fits get the flag, not a code
branch. That choice: **on for LeakyIntegrator, off for RL_lambda**
(RL_lambda reverts to the original 0-initialized behavior everywhere).
Separately, RL_lambda's/RL_lambda_resp_noise's own `lambda_` search bound
was widened from `(0.01, 1.0)` to `(0.01, 2.0)` for all four datasets.

**Why colors/numbers pids' very first response needed a look at all:**
the overwhelming majority set it to (or within noise of) that trial's raw
first observation, not a partial step from a neutral prior — which
Mean/PrimacyRecency already reproduce for free (both reduce to exactly
`value[0]` at n=1) but LeakyIntegrator/RL_lambda's recursive updates from
0.0 do not.

**Why the two models benefit so differently:** LeakyIntegrator improves
substantially and robustly (colors: -13.5% across all pids, -23% once a
6-pid anchor-mismatch minority is excluded — the effect gets *bigger*,
not smaller, once removed; numbers: -24%, no anchor-mismatch minority at
all) — its single `gamma` had to compromise between fitting the first
observation and fitting the rest of the trial's decay, and the fix
removes that tension entirely. RL_lambda barely benefits (numbers: ~1%,
statistically significant but not practically meaningful; colors: net
*negative* even with the widened `lambda_` bound, driven by the same
anchor-mismatch pids) because its typically-high fitted `alpha_0`
(~0.8-0.9 on colors) already lands close to `value[0]` on the first
update under the old scheme — there wasn't much gap left to close. Judged
against the standard that the change needed to be substantial to justify
inclusion, LeakyIntegrator clears that bar and RL_lambda doesn't.

**Why the `lambda_` bound needed widening, separately:** ~80% of colors
pids' RMSE fits were pinned exactly at the old upper bound of 1.0 (numbers
~10-30% depending on `init_from_obs1`, carrabin ~14%, yoo 0% — confirming
this wasn't a uniform optimizer artifact). Widening it recovers real fit
quality, but does not fully close RL_lambda's residual gap to
PrimacyRecency on colors — see `docs/SCIENCE.md`'s "Running-mean ground
truth and the colors/numbers weighting kernel" entry for the structural
(not fitting-range) reason why.

**Also found and fixed in the same investigation, unrelated to the above:**
a carrabin-only bug in `utils/binary_transform.py`'s Laplace-smoothing
transform — `t` was computed as `observation + 1`, silently assuming a
0-indexed observation column, but carrabin's is already 1-indexed. Live
since a prior session's commit that generalized this transform from
carrabin-only code; confirmed by comparing an old carrabin fit's response
against a fresh refit under the buggy code (identical pre-transform
values, different post-transform ones, exactly matching the predicted
off-by-one). Fixed to `t = observation` directly.

**Full investigation:** this session's chat; no `archive/HISTORY_*.md`
entry (nothing here was retired).

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

---

## NoisyRL_lambda retired as the colors/numbers stochastic stand-in, NEF takes its place

**Decision:** `NoisyRL_lambda` no longer appears anywhere in
`scripts/make_paper_figures.py`. NEF now fills its former role as the
colors/numbers "genuinely stochastic model" stand-in in every figure that
used it (`VARIABILITY_STOCHASTIC_MODEL`, `SIGMA_CORR_MODELS`), matching
`balls`' own already-established convention of reading NEF there. Two
figure functions that depended on the old stand-in, and had become
redundant, were archived alongside this: `make_variability_models` (its
own model-overlay branch had been fully disabled -- `include_models=False`
unconditionally -- since before this session, pending exactly this fix;
even with NEF wired in and confirmed non-degenerate, the per-pid
model-vs-human comparison it would show is already covered, more
informatively, by `make_sigma_model_correlation`'s own paired scatter) and
`make_variance_autocorr_models` (fully superseded by `make_sigma_main`'s
row 3, which already shows the same autocorrelation metric against the
same `_resp_noise` models, with NEF -- not NoisyRL_lambda -- in the 4th
slot).

**Why:** `NoisyRL_lambda` (RL_lambda plus a compounding `sigma_state`
noise term) was originally used as this stand-in specifically because NEF
hadn't been fit for colors/numbers yet at the time these figures were
built. `NoisyRL_lambda`'s own model code and fitting pipeline were already
retired from active analysis in an earlier session (see "State-noise
models, NoisyCounting..." above), which explicitly left its use as a
figure-level stand-in untouched at the time, since every figure read
pre-computed `.pkl` files and needed no code change to keep working. NEF
has since been fit (RMSE) for colors/numbers too, and independently
reproduces the same state-persistent noise signature (variance growth +
decaying autocorrelation matching human patterns, which the `_resp_noise`
models do not) that NoisyRL_lambda was standing in for -- this is
`sigma_main` rows 2-3's own established finding. So the stand-in is no
longer needed anywhere in this file.

**A pre-existing drift this closed:** `make_model_performance_nll`'s own
reference-model roster had ALREADY been superseded, in an earlier session,
by a different roster (`NLL_RESP_NOISE_MODELS`, reference
`RL_lambda_resp_noise`) -- confirmed by that earlier refactor's own
comment ("Deliberately NOT touching NLL_MODEL_ORDER/... above -- those
still serve make_variance_autocorr_human/models exactly as before"). The
OLD roster (`NLL_MODEL_ORDER`/`NLL_REFERENCE`/`NLL_LABELS`/
`NLL_MODEL_COLORS`, built around NoisyRL_lambda) had therefore silently
kept living on, unused by its apparent namesake figure, serving only
`make_variance_autocorr_models` (now archived) and, via a stale default
argument, `make_variance_autocorr_human`'s own shared-y-limit probe pass
-- fixed in this same session to pass the current roster
(`NLL_RESP_NOISE_MODELS` + NEF) explicitly instead, matching
`make_sigma_main`'s own row 3. A fully dead function found along the way,
`_nll_perf_path` (zero callers even before this cleanup, confirmed by
grep), was archived alongside the rest of the old roster.

**Alternatives evaluated and rejected:** re-enabling
`make_variability_models`'s dormant model-overlay branch with NEF instead
of retiring the function outright -- rejected because the resulting
KDE-overlay comparison, even genuinely non-degenerate (NEF's colors/
numbers per-pid qid-residual variability confirmed nonzero and varying
across all 46 pids -- not collapsed to a floor), would still be strictly
less informative than the existing paired per-pid correlation in
`make_sigma_model_correlation`.

**Full investigation:** `archive/HISTORY_modeling_2026.md`'s "NoisyRL_lambda
retired as the colors/numbers stochastic stand-in in
scripts/make_paper_figures.py" entry.

---

## Four standalone human-only figures deleted (not archived) as pure duplicates

**Decision:** `make_lambda_human`, `make_lambda_sanity_human`,
`make_variability_human`, and `make_sigma_sanity_human` were deleted
outright from `scripts/make_paper_figures.py` -- not moved to `archive/`,
unlike every other retirement in this file. `git log --follow` recovers
them if ever needed; no separate archive copy was made.

**Why deleted rather than archived:** each one is a 100%-content subset of
a composite figure that already exists -- `lambda_overview`'s own
docstring says its row 1 is "identical to make_lambda_human" and its row 2
is the same panels as `make_lambda_sanity_human`; `sigma_overview`'s
docstring says the same for `make_variability_human` (row 1) and
`make_sigma_sanity_human` (row 2). Both composites were built by literally
combining these exact pairs into one 2-row figure. There was no unique
content left in any of the four standalone versions to preserve, unlike
the `NoisyRL_lambda` retirement above (real judgment calls, partial
restoration nuance) -- so the lighter-weight "just delete it" treatment
fit, rather than archiving code with nothing distinct to restore.

**What was deliberately NOT touched (at the time):** `lambda_metric` -- a
differently-styled, standalone version of the fitting-demo panel, built
specifically as an Inkscape-inset source for `lambda_main`, not a
duplicate. `lambda_overview`/`sigma_overview` themselves -- these are the
composites the four deleted figures were redundant with, obviously kept.

**Update:** `make_variance_autocorr_human` was subsequently deleted too
(same treatment as the four above, not archived), on the person's
explicit instruction after being reminded of the one real difference: its
panel A held `AUTOCORR_SCHEMATIC`, a hand-made diagram with no other
current home (`sigma_main` dropped every schematic panel when it was
built). Everything else in the figure -- the 3 human-only autocorrelation
panels -- was already a pure duplicate of `sigma_main` row 3's own human
curves, same reasoning as the four above. `AUTOCORR_SCHEMATIC`'s own
constant definition was removed alongside it (orphaned, zero other
callers); the underlying `.svg` source asset itself was not touched, only
its use inside this one now-deleted figure.

---

## Colors' LeakyIntegrator/RL_lambda added to the sigma-growth boundary-clipping correction

**Decision:** `SIGMA_GROWTH_BOUNDARY_CORRECTED` (`scripts/make_paper_figures.py`,
controlling `sigma_main` row 2's residual-variance-growth panel) now also
covers `("colors", "LeakyIntegrator")` and `("colors", "RL_lambda")`,
alongside the pre-existing `("colors", "Mean")`/`("colors",
"PrimacyRecency")` entries.

**Why:** the original correction (see
`archive/HISTORY_modeling_2026.md`'s "Boundary-clipping correction for
sigma-growth negative control") was scoped to Mean/PrimacyRecency because,
at the time, LeakyIntegrator/RL_lambda's colors fits were "already close
to flat under the RAW metric." This session's fit changes broke that:
LeakyIntegrator's `init_from_obs1` now forces its colors `mu` to exactly
`value[0]` (±1) at observation 0 for 100% of trials (previously a free
0-init), and RL_lambda's widened `lambda_` bound pushes its own colors
`mu` within 0.9 of ±1 at observation 0 for 78% of trials -- both now hit
the same `add_noise` boundary-clip artifact the original correction
targets. Confirmed directly, not just by inspection: raw growth ratios
were 1.457 (LeakyIntegrator) / 1.363 (RL_lambda) -- the same range as
Mean/PrimacyRecency's own original uncorrected 1.54/1.37 -- and applying
the existing `_resid_variance_growth_corrected` machinery unchanged
flattens both to 0.987/1.110, matching Mean/PrimacyRecency's own corrected
0.98-1.15 range. Same mechanism, not a coincidence.

**What did NOT need touching:** the correction machinery itself
(`_clipped_normal_var`, `_implied_sigma`, `_resid_variance_growth_corrected`)
-- fully general over (task, model), already worked correctly for the two
new pairs once added to the set. Every other (task, model) combination
remains uncorrected -- confirmed still close to flat under the raw
metric.

**Also checked, for consistency, whether Human/NEF need the same
treatment -- Human: not applied; NEF: not needed.**

*Human* -- 85.7% of colors responses at observation 0 are within 0.9 of
the boundary, same order as the models, so the same concern transfers.
Feasibility check (not wired in): the correction needs a known `mu` per
(pid, observation, qid) to invert against -- for the math models that's
the deterministic model's own re-run output; for Human, the qid-group
MEAN response (already computed by `_resid_frame` to build the residuals
in the first place) is the natural empirical stand-in, and re-using
`_implied_sigma`/`_clipped_normal_var` unchanged with that substitution
produces a real, computable result. But it doesn't just flatten the
curve the way it did for the math models -- it changes the SHAPE (raw
human variance looks like it grows to ~1.3x by the last observation;
corrected, it looks like it SHRINKS to ~0.88x, dipping to ~0.66x in the
middle), because human `mu` sits closest to the boundary at observation 0
specifically, inflating the corrected BASELINE the most. Not applied,
deliberately: unlike `add_noise`'s known exactly-Gaussian-then-clip
generative process, assuming Gaussian noise around a "true" human
response is an unvalidated assumption, and `mu` itself would be estimated
from only 2 repeats per qid group (`RESID_MIN_REPEATS["colors"]=2`) --
noisy on its own, unlike the math models' exact, zero-estimation-error
mu. Applying it would risk revising `docs/SCIENCE.md`'s own reading of
colors' human variance-growth panel on a shakier methodological footing
than the math-model correction has.

*NEF* -- checked directly at a stricter >0.98 threshold, all
observations, all three tasks (not just colors, not just observation 0):
colors overall 0.02% (1 row out of ~1472 at a handful of observations),
numbers 0.00% at every single observation, balls (MLE variant) 0.00%.
Nowhere near significant. NEF's own decoded value essentially never
approaches the boundary at all -- no analogue correction needed or
applicable; its row-2 curves stay as the uncorrected, genuine signal.

**Full investigation:** this session's chat; no separate
`archive/HISTORY_*.md` entry (nothing here was retired, just a set
extended).

---

## NEF_synaptic's `pes_learning_rate`: single fixed `4e-4` default, not per-dataset

**Decision:** `pes_learning_rate` (`_NEF_FIXED`, `fitting/model_params.py`)
set to `4e-4` for all four datasets, replacing a leftover `1e-4` inherited
unchanged from the pre-May implementation (see NEF_synaptic's
reimplementation in `docs/SCIENCE.md`'s "Current thread"). Not split into
per-dataset overrides despite each dataset having its own RMSE-minimizing
value.

**Why 4e-4, not per-dataset values:** a coordinate sweep of
`pes_learning_rate` against NEF (recurrent) on identical trials/params
(`scripts/check_NEF_pipeline.py --compare_synaptic
--pes_learning_rate_sweep`) found a clean, unimodal RMSE minimum at `5e-4`
for carrabin (`n_neurons=500`, `n_neurons_counting=500`) and `3e-4` for
soltani_numbers (`n_neurons=500`, `n_neurons_counting=2000`) -- `4e-4`
splits the two. The gap was deliberately checked against the
`n_neurons`/`n_neurons_counting` question this reimplementation raised:
Nengo's own `PES` builder (`nengo.builder.learning_rules.SimPES`)
normalizes its per-step decoder update by the *presynaptic* population's
`n_neurons` (`alpha = -learning_rate * dt / n_neurons`), so the learned
readout's effective speed should already be architecturally invariant to
population size. Both sweep points held `n_neurons=500` fixed and only
varied `n_neurons_counting` (500 vs. 2000, a 4x difference) alongside
dataset -- the optimum moved by <2x, far smaller than the 3x+ RMSE swings
seen scanning `pes_learning_rate` itself over a decade in either sweep.
That rules out population size as the driver and points instead at
dataset-specific factors not yet disentangled (most likely trial length --
soltani_numbers trials run 15 observations vs. carrabin's 5, and
`models.NEF._simulate_trial` rebuilds the network, resetting PES's learned
weights, fresh every trial -- giving the synaptic connection 3x longer to
learn before reset; observation-value statistics also differ between
carrabin's binary-probability scale and soltani's continuous [-1,1] scale
and weren't independently ruled out).

**Why not chase this further with per-dataset overrides now:** the shared
`fixed` dict already supports per-dataset overrides for exactly this kind
of architecture constant (carrabin already overrides `n_neurons`/
`n_neurons_counting`/`radius_c` relative to `_NEF_FIXED`'s defaults), so
adding dataset-specific `pes_learning_rate` values later is a trivial,
idiomatic extension if it turns out to matter -- no dynamic-scaling
formula needed. It doesn't matter yet either way: `NEF` (recurrent) never
reads `pes_learning_rate`, and `NEF_synaptic` has no `MODEL_PARAMS` entry
or Optuna fit of its own yet (deliberately deferred -- see
`docs/SCIENCE.md`). A single shared default that gets both datasets' RMSE
within ~2x of their own individual optimum is more than adequate for the
current qualitative-baseline-consistency check; revisit once
`NEF_synaptic` gets a real per-pid fit.

**Full investigation:** this session's chat; no separate
`archive/HISTORY_*.md` entry (NEF_synaptic was reimplemented, not
retired).

---

## `neural_main`'s row 1/row 2 dependent variables converted to relative (%) metrics; row 3's PE noise CV designed, pending cluster collection

**Decision:** Row 1's "PE decrease" (`max_pe - end_pe`, oddball window)
and row 2's "Activity decay"/"ΔR decay" (`first - last`, `early - late`)
are now expressed as percentages of their own starting value --
`(max-end)/max`, `(first-last)/first`, `(early-late)/early` -- relabeled
"PE decay (%)", "Activity decay (%)", "ΔR decay (%)" throughout
`neural_main` (`scripts/make_paper_figures.py`:
`_plot_oddball_param_effect`, `_plot_oddball_dv_scatter`,
`_param_scan_decay_metrics`, `_plot_neural_main_decay_vs_param`,
`_plot_param_scan_dv_scatter`). Row 3's "PE noise" (`pe_variance_mean`,
`n_neurons` sweep) gets an analogous coefficient-of-variation metric,
`pe_cv_pct = sqrt(pe_variance_mean) / pe_mean_mean * 100` --
`_n_neurons_snr_worker` (`scripts/neural_experiments.py`) now also saves
`pe_mean_mean` (mean `|pe_product|` in the same window), but the figure
itself isn't repointed at it yet -- the existing 50-cell grid predates
this field and needs a full resubmit/recollect first (see chat for the
exact commands).

**Why row 1:** a critic's objection, not a numerical audit -- a bigger
peak PE has more room to fall before settling near zero regardless of
whether the network is genuinely "resolving" it faster, so the absolute
decrease is confounded with `max_pe` itself, which is exactly what `α₀`
is being swept to test. Verified on real data (`alpha_0` sweep,
`soltani_numbers`, 90 grid cells): `max_pe` and the correction both stay
well-behaved (ratio always in [0,1], no cell with `end_pe > max_pe`),
and the relative version shows proportionally tighter SEMs at every
`alpha_0` value than the absolute one.

**Why row 2 too, despite a different mechanism:** checked directly (155
real pids, `lambda_` sweep) whether the SAME confound applies here --
it doesn't, in either direction expected. Activity decay's own
denominator (`act_first`) is uncorrelated with `lambda_` (r=-0.11,
n.s.); ΔR decay's own denominator (`resp_early`) is mildly
*anti*-correlated (r=-0.35), the opposite direction from a confound (a
smaller starting value if anything works against a bigger absolute
decrease, not for it). So neither absolute metric was actually biased.
Converted anyway for interpretability (a bounded [0,1] "fraction
resolved" reads more directly than a raw activity/response-magnitude
difference) and so all three rows share one consistent percentage
framing. Both relative versions came out with meaningfully *tighter*
correlations to `lambda_` than their absolute counterparts (activity:
r=0.86→0.89; ΔR: r=0.61→0.92) -- a real clarity gain, not just a
relabeling.

**Why row 3's CV over the alternative (normalize by the smallest-
`n_neurons` condition instead):** that alternative needs no new
simulation output at all (pure recompute from the existing
`pe_variance_mean` grid) and was seriously considered -- it produces a
clean, monotonic 100%→14% curve on the existing data. Went with CV
instead because it's relative to the signal itself (the actual textbook
SNR framing implied by this row's own name), not to an arbitrary
reference condition inside the current sweep's own range -- the
baseline-relative version's absolute numbers would silently shift if
the sweep's own smallest `n_neurons` value ever changed. Confirmed
locally before committing to a real cluster resubmit (`nef-debug-runner`,
`n_neurons` ∈ {50, 250}, `n_seeds=3`, no files written): `pe_cv_pct`
computes without a near-zero-denominator blowup (32.7% → 13.9%), and
decreases with `n_neurons` matching every other metric in this row.

**What did NOT change:** row 1's oddball design, row 2's `param_scan`
design, and row 3's `n_neurons_snr` window/worker structure are all
unchanged -- only how the same underlying quantities get combined into
a displayed number.

**Full investigation:** this session's chat; no separate
`archive/HISTORY_*.md` entry (metric refinement, nothing retired).

