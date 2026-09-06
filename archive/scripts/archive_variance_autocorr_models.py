"""
scripts/archive_variance_autocorr_models.py

Archived: make_variance_autocorr_models() from scripts/make_paper_figures.py,
plus the model-roster constants/helper it (and it alone, once
make_model_performance_nll was separately refactored -- see below) still
depended on: NLL_MODEL_ORDER, NLL_REFERENCE, NLL_LABELS, NLL_MODEL_COLORS,
and _nll_perf_path.

## What this figure was

Same 1x4 layout as make_variance_autocorr_human (still active): panel A a
hand-made autocorrelation-metric schematic, panels B-D within-trial lag-k
residual autocorrelation of the response, one per task (balls, colors,
numbers). This "_models" variant additionally overlaid Mean/
LeakyIntegrator/PrimacyRecency's own "_resp_noise" NLL fits PLUS
NoisyRL_lambda (via NLL_MODEL_ORDER) as the colors/numbers "our model" 4th
slot, recolored to RL_lambda's own red-orange and labeled "RL_lambda*" in
the legend (NLL_LABELS) -- RL_lambda's own bare/deterministic fit was
deliberately NOT shown, since NoisyRL_lambda already played its
conceptual role there.

## Why archived

Superseded by make_sigma_main's own row 3 (still active), which shows the
exact same metric (autocorrelation of the qid-conditional residual) for
the exact same task roster (RESID_TASK_PANELS), using the SAME
"_resp_noise" models (NLL_RESP_NOISE_MODELS) -- but with NEF, not
NoisyRL_lambda, in the 4th slot (via `_load_variance_autocorr_data(
models=NLL_RESP_NOISE_MODELS, ..., include_nef=True)`). NoisyRL_lambda
was retired as the colors/numbers stochastic stand-in this same session
(see docs/DECISIONS.md's "NoisyRL_lambda retired as the colors/numbers
stochastic stand-in, NEF takes its place"), since NEF has since been fit
for colors/numbers too and independently confirmed to reproduce the same
decaying-autocorrelation signature NoisyRL_lambda used to show (this
figure's OWN docstring, reproduced below, is itself the historical record
of that original NoisyRL_lambda-vs-"_resp_noise" finding). With NEF
already covering this exact comparison in sigma_main's row 3, a standalone
"_models" sibling to make_variance_autocorr_human would be pure
duplication -- it was retired rather than updated in place to read NEF,
since sigma_main already IS that update.

## A real, pre-existing drift this closes

This figure's own docstring claimed its roster "match[ed] model_performance_
nll's own roster exactly" -- true when written, but no longer true by the
time this session started: `make_model_performance_nll` (still active) had
since been refactored, in an earlier session, to read a completely
different, newer roster (NLL_RESP_NOISE_MODELS: Mean/LeakyIntegrator/
PrimacyRecency/RL_lambda_resp_noise, reference RL_lambda_resp_noise, NOT
NoisyRL_lambda) -- that earlier refactor's own comment says so explicitly
("Deliberately NOT touching NLL_MODEL_ORDER/NLL_REFERENCE/NLL_MODEL_COLORS/
NLL_LABELS/_nll_perf_path above -- those still serve make_variance_
autocorr_human/models exactly as before"). So NLL_MODEL_ORDER and friends
had ALREADY been orphaned from make_model_performance_nll for some time,
serving only this now-archived figure (plus, indirectly, make_variance_
autocorr_human's own probe pass, which called `_load_variance_autocorr_
data()` with no override and so inherited NLL_MODEL_ORDER as its default --
fixed in the same session as this archival to instead pass
`models=NLL_RESP_NOISE_MODELS, responses_path_fn=_nll_resp_noise_responses_
path, include_nef=True` explicitly, matching sigma_main's own row 3; see
that function's own current docstring).

`_nll_perf_path` specifically was found to have NO live callers at all
(only comment/docstring mentions) even before this archival -- a fully
dead function, presumably left over from an even earlier version of
make_model_performance_nll before ITS OWN refactor to
`_nll_resp_noise_perf_path`. Archived here anyway since it's part of the
same now-fully-orphaned NLL_MODEL_ORDER-era roster and its own docstring
explicitly referenced that roster.

## What moved

- `make_variance_autocorr_models` (whole function).
- `NLL_MODEL_ORDER`, `NLL_REFERENCE` (also already dead -- confirmed zero
  callers even for the archived function itself, which never referenced
  it directly), `NLL_LABELS`, `NLL_MODEL_COLORS`, `_nll_perf_path`, plus
  their own explanatory module-level comment block.
- `NLL_TASK_PANELS` was NOT moved -- confirmed it is still used by
  `make_model_best_fit` and `make_model_performance_nll`, both active.
- `_load_variance_autocorr_data`/`_draw_variance_autocorr_panel` (the
  shared loader/drawing helpers this figure used) were NOT moved -- both
  remain active, still used by `make_variance_autocorr_human` and
  `make_sigma_main`. Their own default arguments were updated (in the live
  file) from `NLL_MODEL_ORDER`/`_nll_responses_path`/`NLL_MODEL_COLORS` to
  `NLL_RESP_NOISE_MODELS`/`_nll_resp_noise_responses_path`/`MODEL_COLORS`,
  so a future no-argument call resolves to the CURRENT roster rather than
  a NameError against a now-archived constant.
- `_nll_responses_path` (used by `_sigma_model_source_path` for the three
  "_resp_noise" models, unrelated to NoisyRL_lambda) was NOT moved -- only
  its own dead `model == "NoisyRL_lambda"` special case was stripped, since
  no remaining caller ever passes that model name to it anymore.

## How to restore

`git mv` this file back to `scripts/`, re-inline `make_variance_
autocorr_models`/`NLL_MODEL_ORDER`/`NLL_REFERENCE`/`NLL_LABELS`/
`NLL_MODEL_COLORS`/`_nll_perf_path` into `scripts/make_paper_figures.py`,
re-add a `"variance_autocorr_models": make_variance_autocorr_models` entry
to that file's own FIGURES dispatch dict, and restore `_nll_responses_path`'s
`model == "NoisyRL_lambda"` special case if NoisyRL_lambda itself is ever
un-retired too (see docs/DECISIONS.md and
archive/models/archive_math_models_noise.py). NOT standalone-runnable as
archived: it references module-level state (FIGURE_SIZE, AUTOCORR_SCHEMATIC,
RESID_TASK_PANELS, MODEL_COLORS, RUNS_DIR, HUMAN_COLOR, _apply_slide_style,
_rasterize_svg, _load_variance_autocorr_data, _draw_variance_autocorr_panel,
_save_fig) that still lives in scripts/make_paper_figures.py and was not
duplicated here.
"""

# ── Original imports needed by this code ───────────────────────────────────
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Orphaned roster constants (see module docstring for why) ───────────────

# Real, fresh fitting output found on disk (see chat) after searching the
# codebase directly rather than a since-unlocatable "EI#16" chat: a genuine
# per-pid NLL fit now exists for every task, replacing RMSE's inability to
# identify a noise term at all (RMSE is minimised by the conditional mean,
# so any noise parameter collapses to its lower bound -- confirmed earlier
# this session for NoisyRL_lambda's old RMSE fit: sigma_state pinned at
# EXACTLY 0.02 for all 35 pids). Verified this NEW fit is genuinely
# per-pid, not another floor-collapse: NoisyRL_lambda's sigma_state now has
# 37 distinct values across 45 pids (numbers), ranging 0.017-0.292, not one
# repeated constant.
#
# NO NEF FIT EXISTS UNDER THIS METRIC AT ALL (checked directly: no
# "*NEF*nll*" file anywhere in data/runs) -- full NEF simulation at
# n_sims=100 per Optuna trial was presumably judged too expensive to run
# yet. NoisyRL_lambda is therefore the uniform "best/reference" model for
# EVERY task here, not split NEF-for-balls/snacks vs RL_lambda-for-colors/
# numbers the way make_model_performance's own reference was.
#
# A REAL DATA-VINTAGE DIFFERENCE FROM make_model_performance, WORTH FLAGGING
# EXPLICITLY: this fit's own human data has 45 pids for colors/numbers
# (confirmed against the CURRENT canonical data/soltani_{colors,numbers}.pkl
# -- both now 45, not the 35 every earlier figure in this deck, including
# make_model_performance itself, was built against). carrabin (21) and yoo
# (38) are unchanged. This figure is therefore NOT a strict apples-to-apples
# re-run of make_model_performance on a different loss -- its colors/
# numbers panels reflect 10 more participants than the RMSE figure's do.
#
# RL_lambda's OWN box is dropped here, per instruction -- NoisyRL_lambda
# already plays RL_lambda's role at this position (the "our model" entry),
# so showing both would be redundant. NoisyRL_lambda is recolored to
# RL_lambda's OWN established color (#d55e00, red-orange) rather than its
# usual tan (#ca9161, still used elsewhere in this deck, e.g. the
# variability figures) -- same convention as NEF/RL_lambda sharing one
# color in make_model_performance: same color signals "playing the same
# conceptual role", here explicitly replacing RL_lambda's own slot. Legend
# TEXT stays "Noisy RL" (MODEL_DISPLAY, unchanged) -- only the color, not
# the label, is borrowed.
NLL_MODEL_ORDER = ["Mean", "LeakyIntegrator", "PrimacyRecency", "NoisyRL_lambda"]
NLL_REFERENCE = "NoisyRL_lambda"  # uniform across all four tasks -- see above

# Legend labels for THIS figure specifically -- full (non-abbreviated) names
# now that the legend only has 4 entries and comfortably fits them, unlike
# MODEL_DISPLAY's compact "LI"/"PR" abbreviations built for the 5-entry
# legends elsewhere in this deck. NoisyRL_lambda is labeled "RL_lambda*" per
# explicit instruction -- the asterisk is a live-talk footnote (explained
# verbally, not spelled out in the figure itself).
NLL_LABELS = {
    "Mean": "Mean",
    "LeakyIntegrator": "LeakyIntegrator",
    "PrimacyRecency": "PrimacyRecency",
    "NoisyRL_lambda": "RL_lambda*",
}

# Local color override for THIS figure only -- MODEL_COLORS itself is left
# untouched (NoisyRL_lambda's own tan still applies everywhere else, e.g.
# the now-also-archived make_variability_models). NOTE: MODEL_COLORS is
# module-level state from scripts/make_paper_figures.py, not duplicated
# here -- see this file's own module docstring.
NLL_MODEL_COLORS = {m: MODEL_COLORS[m] for m in NLL_MODEL_ORDER}
NLL_MODEL_COLORS["NoisyRL_lambda"] = MODEL_COLORS["RL_lambda"]


def _nll_perf_path(task_key: str, model: str) -> Path:
    """Path to one (task, model)'s *_nll_performance.pkl. Every model except
    NoisyRL_lambda is fit as its own name PLUS "_resp_noise" on disk (see
    this section's own module-level comment for why) -- that suffix is
    purely a file-naming/fitting-pipeline detail, so it's added here rather
    than exposed to any caller; every other function in this file refers to
    these models by their plain names (Mean, LeakyIntegrator, ...).

    NOTE: had ZERO live callers even before this archival -- confirmed by
    grep, only comment/docstring mentions referenced it. Kept here anyway
    since it's part of the same orphaned roster. RUNS_DIR is module-level
    state from scripts/make_paper_figures.py, not duplicated here.
    """
    file_model = model if model == "NoisyRL_lambda" else f"{model}_resp_noise"
    if task_key == "balls":
        return RUNS_DIR / "carrabin" / f"{file_model}_carrabin_nll_performance.pkl"
    if task_key == "snacks":
        return RUNS_DIR / "yoo" / f"{file_model}_yoo_nll_performance.pkl"
    dataset = "soltani_colors" if task_key == "colors" else "soltani_numbers"
    return RUNS_DIR / "nll" / f"{file_model}_{dataset}_nll_performance.pkl"


def make_variance_autocorr_models() -> Path:
    """Same 1x4 layout as make_variance_autocorr_human (panel A unchanged) --
    IDENTICAL except for the added model data and legend, per instruction:
    panels B-D now also overlay Mean/LeakyIntegrator/PrimacyRecency's own
    "_resp_noise" NLL fits plus NoisyRL_lambda (NLL_MODEL_ORDER -- RL_lambda's
    own bare/deterministic fit is NOT shown, matching model_performance_nll's
    own roster exactly, per instruction; it was in an earlier version of
    this figure and has been dropped). NoisyRL_lambda is recolored to
    RL_lambda's own established red-orange (NLL_MODEL_COLORS) and labeled
    "RL_lambda*" in the legend (NLL_LABELS) -- same convention as
    make_model_performance_nll's own legend, reused here rather than
    reinvented, per instruction that the last label be "RL_lambda*".
    Legend uses FULL model names throughout (NLL_LABELS), not
    MODEL_DISPLAY's abbreviated "LI"/"PR" -- per instruction.

    Confirms directly (not just visually -- an earlier visual read was
    wrong, see chat) that only NoisyRL_lambda shows genuine decaying
    autocorrelation resembling Human's own pattern; the three
    "_resp_noise" models stay within noise of zero at every lag in every
    task, matching what their own math predicts (i.i.d. response noise,
    added AFTER the clean deterministic trajectory per
    models/math_models.py's own add_noise(), has no mechanism to produce
    lag correlation).

    NOTE (added on archival): its claim above of matching
    model_performance_nll's own roster "exactly" was true when written but
    stale by the time this was archived -- see this file's own module
    docstring, "A real, pre-existing drift this closes".
    """
    _apply_slide_style()
    data = _load_variance_autocorr_data()

    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, sharey=True,
                             constrained_layout=True)
    axes[0].axis("off")
    axes[0].set_title("Metric Definition", color="0.3")
    schematic = _rasterize_svg(AUTOCORR_SCHEMATIC)
    if schematic is not None:
        axes[0].imshow(schematic, aspect="auto")

    for i, (ax_ac, (task_key, title)) in enumerate(zip(axes[1:], RESID_TASK_PANELS)):
        human_res, model_results, lags = data[task_key]
        _draw_variance_autocorr_panel(ax_ac, task_key, title, human_res,
                                      model_results, lags, include_models=True,
                                      show_ylabel=(i == 0))
    # sharey autoscales to human+models here -- this IS the range
    # make_variance_autocorr_human's own probe pass independently
    # reconstructs and reuses (see that function's own docstring).

    legend_handles = [Line2D([0], [0], color=HUMAN_COLOR, lw=2.2, label="Human")]
    for m in NLL_MODEL_ORDER:
        legend_handles.append(Line2D([0], [0], color=NLL_MODEL_COLORS[m], lw=2.2,
                                     label=NLL_LABELS.get(m, m)))
    fig.get_layout_engine().set(h_pad=0.25)
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=5,
               frameon=True, framealpha=0.9)

    out_path, _ = _save_fig(fig, "variance_autocorr_models")
    plt.close(fig)
    return out_path
