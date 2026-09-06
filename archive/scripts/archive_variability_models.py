"""
scripts/archive_variability_models.py

Archived: make_variability_models() from scripts/make_paper_figures.py.

This was the model-overlay counterpart to make_variability_human (still
active): SAME 1x4 layout (panel A = a hand-made schematic explaining the
response-variability metric, panels B-D = per-task KDEs of response
variability under identical repeated inputs, for balls/colors/numbers),
but with an `include_models=True` branch intended to add each task's ONE
genuinely-stochastic model (VARIABILITY_STOCHASTIC_MODEL) as a proper KDE
line, plus every other (deterministic) model as a jittered cluster of
points at x=0.

That model-overlay branch was NEVER ACTUALLY ENABLED: this function's own
body always called `_plot_variability_panel(..., include_models=False,
...)`, identically to make_variability_human -- its own docstring said so
explicitly ("model plotting is DISABLED here for now ... rather than
showing that stale result"), pending a real (non-degenerate) colors/numbers
stochastic fit. Before this archival, the two functions were therefore
BYTE-FOR-BYTE IDENTICAL in output, differing only in which filename
(`variability_models` vs `variability_human`) they saved to.

Retired this session, when NoisyRL_lambda (the colors/numbers stochastic
stand-in this function's own docstring was waiting on) was itself retired
in favor of NEF -- see docs/DECISIONS.md's "NoisyRL_lambda retired as the
colors/numbers stochastic stand-in, NEF takes its place" entry. Checked
directly, not assumed, whether NEF being wired in made re-enabling
`include_models=True` worth it: NEF's own colors/numbers RMSE-fitted
response files (`data/runs/rmse/NEF_soltani_{colors,numbers}_responses.pkl`)
exist and produce genuine, non-degenerate per-pid qid-residual variability
(std ~0.007-0.05 colors, ~0.009-0.02 numbers across 46 pids -- not
collapsed to a zero floor). But even with a real, working model overlay
available, this figure would still be fully redundant for its actual
purpose: the per-pid model-vs-human comparison it would show (aggregate
KDE overlap) is already covered, MORE informatively (a paired per-pid
scatter with a correlation coefficient, not just distributional overlap),
by make_sigma_model_correlation -- which was ALSO updated this session to
read NEF instead of NoisyRL_lambda for colors/numbers. So "fix" (flip
include_models to True) was rejected in favor of "retire": there was no
remaining reason to keep a second, more expensive KDE-based figure around
once a strictly more informative comparison of the same underlying
quantity already exists elsewhere in this file.

Kept here for reference/history, not deleted -- see
archive/HISTORY_modeling_2026.md's "NoisyRL_lambda retired as the
colors/numbers stochastic stand-in..." entry for the full session
narrative. NOT standalone-runnable as archived: it references module-level
state (FIGURE_SIZE, VARIABILITY_SCHEMATIC, VARIABILITY_TASK_PANELS,
_apply_slide_style, _rasterize_svg, _plot_variability_panel, _save_fig)
that still lives in scripts/make_paper_figures.py and was not duplicated
here.

How to restore: `git mv` this file back to `scripts/`, rename the function
back to a top-level definition inside make_paper_figures.py (or just
re-inline its body there), and re-add a `"variability_models":
make_variability_models` entry to that file's own FIGURES dispatch dict.
If restoring WITH the model overlay actually enabled this time
(include_models=True), also re-verify VARIABILITY_STOCHASTIC_MODEL/
VARIABILITY_DETERMINISTIC_MODELS' colors/numbers entries still point at
real, non-degenerate response files before trusting the output -- do not
assume the check performed this session is still valid if the underlying
fits have changed since.
"""

# ── Original imports needed by this function ──────────────────────────────
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt


def make_variability_models() -> Path:
    """Same layout as make_variability_human (including the panel-A
    schematic), but panels B-D now add each task's ONE genuinely-stochastic
    model as a proper KDE, plus every other (deterministic) model as a
    jittered cluster of points at x=0 -- see _plot_variability_panel's own
    docstring for why.

    NOTE: NoisyRL_lambda's fit was found to be from an old, pre-quasi-MLE
    RMSE run (sigma_state/sigma_resp pinned at their manually-chosen floor,
    not genuinely fit per pid -- see chat); the actual NLL/quasi-MLE fitting
    work for colors/numbers has been offloaded to another session, so model
    plotting is DISABLED here for now (include_models=False, same as the
    human-only figure) rather than showing that stale result. Once a real
    fit lands, flip include_models back to True below -- the plotting
    machinery itself is untouched and ready.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, constrained_layout=True)

    axes[0].axis("off")
    axes[0].set_title("Metric Definition", color="0.3")
    schematic = _rasterize_svg(VARIABILITY_SCHEMATIC)
    if schematic is not None:
        axes[0].imshow(schematic, aspect="auto")

    for i, (ax, (task_key, title)) in enumerate(zip(axes[1:], VARIABILITY_TASK_PANELS)):
        _plot_variability_panel(ax, task_key, title, include_models=False,
                                show_ylabel=(i == 0))

    out_path, _ = _save_fig(fig, "variability_models")
    plt.close(fig)
    return out_path
