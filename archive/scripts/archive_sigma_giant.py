"""
scripts/archive_sigma_giant.py

Archived: make_sigma_giant() from scripts/make_paper_figures.py.

This was a 3-row, 4-column MEGA figure: rows 1-2 were make_sigma_overview's
own 2x4 content (col 1 = schematic/crosstask special panel, cols 2-4 =
balls/colors/numbers task panels), row 3 was the autocorrelation panels
(NLL_RESP_NOISE_MODELS roster). Retired this session and split into two
figures, per instruction:

  - make_sigma_main        -- 3x3 grid, column 1 (the "sigma definition"
                               schematic, the crosstask panel, and the "rho
                               definition" schematic -- previously col 1 of
                               rows 1/2/3) removed entirely. Row 1
                               (variability KDEs) and row 3 (autocorrelation)
                               keep their content, shifted one column left;
                               row 2 is left BLANK (its own former content --
                               the splithalf panels -- now lives in
                               make_sigma_reliability). Row 3's titles are
                               cleared (redundant with row 1's, two rows up).
  - make_sigma_reliability -- 1x3 supplementary figure: row 2 cols 2-4 of
                               the giant (splithalf reliability panels for
                               balls/colors/numbers), titles RESTORED since
                               it's now standalone (matching
                               make_lambda_reliability's own treatment).

The crosstask panel that used to sit in row 2 col 1 of this giant now lives
in make_lambda_sigma_crosstask instead, paired with its lambda analogue --
it was never re-homed into either of the two figures above.

Kept here for reference/history, not deleted -- see docs/HISTORY.md if the
full rationale for the original 3x4 stack is ever needed again. NOT
standalone-runnable as archived: it references module-level state
(FIGURE_SIZE, VARIABILITY_SCHEMATIC, AUTOCORR_SCHEMATIC, VARIABILITY_
TASK_PANELS, RESID_TASK_PANELS, NLL_RESP_NOISE_MODELS, MODEL_COLORS,
MODEL_LABEL, HUMAN_COLOR, _rasterize_svg, _plot_variability_panel,
_plot_sigma_crosstask_panel, _plot_sigma_splithalf_panel,
_load_variance_autocorr_data, _draw_variance_autocorr_panel,
_nll_resp_noise_responses_path, _apply_slide_style, _save_fig) that still
lives in scripts/make_paper_figures.py and was not duplicated here.
"""

# ── Original imports needed by this function ──────────────────────────────
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def make_sigma_giant() -> Path:
    """3-row, 4-column MEGA figure: rows 1-2 are make_sigma_overview's own
    2x4 content (schematic + 3 KDEs, then crosstask + 3 splithalf, no
    titles), row 3 is the autocorrelation panels ("noise is
    autocorrelated") -- per instruction.

    UNLIKE make_lambda_giant, all three rows here already share the SAME
    3-task column mapping (col 1 = non-task schematic/special panel,
    col 2 = balls, col 3 = colors, col 4 = numbers) -- VARIABILITY_
    TASK_PANELS and RESID_TASK_PANELS are the identical list, so there's
    no task-set mismatch across rows to reconcile the way response_change's
    own balls-inclusive roster needed for make_lambda_giant's row 1.

    ROW 3 MODEL ROSTER: NLL_RESP_NOISE_MODELS (Mean/LeakyIntegrator/
    PrimacyRecency/RL_lambda, all "_resp_noise", all from the unified
    data/runs/nll/ folder) -- per instruction, NOT make_variance_autocorr_
    models' own default NLL_MODEL_ORDER (which uses NoisyRL_lambda
    instead of RL_lambda_resp_noise, and whose own file no longer exists
    in data/runs/nll/ after this session's folder cleanup+refit anyway).
    NEF is NOT included -- no NLL fit exists for it yet; add it to
    NLL_RESP_NOISE_MODELS once one does, no other change needed here,
    per instruction ("we'll include NEF when it's available").
    Passed in via _load_variance_autocorr_data/_draw_variance_autocorr_
    panel's own new `models`/`model_colors`/`responses_path_fn`
    parameters (added this session, defaulting to each function's
    EXISTING behavior so make_variance_autocorr_human/models are
    completely unaffected -- verified directly, both still run and
    produce their own separate outputs).

    Every panel reuses the EXACT SAME helper functions the source figures
    already call -- no panel-drawing logic duplicated here, only the
    layout (and, for row 3, the model-roster override) differs.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(3, 4, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 2.1 * 0.75),
                             constrained_layout=True)

    # Row 1 -- make_sigma_overview's own row 1 (== make_variability_human's
    # own panels), unchanged.
    axes[0, 0].axis("off")
    axes[0, 0].set_title("\u03c3 definition", color="0.3")
    schematic = _rasterize_svg(VARIABILITY_SCHEMATIC)
    if schematic is not None:
        axes[0, 0].imshow(schematic, aspect="auto")
    for i, (task_key, title) in enumerate(VARIABILITY_TASK_PANELS):
        ax = axes[0, i + 1]
        _plot_variability_panel(ax, task_key, title, include_models=False,
                                show_ylabel=(i == 0))
        ax.set_ylabel("Density" if i == 0 else "")

    # Row 2 -- make_sigma_overview's own row 2, unchanged (no titles).
    _plot_sigma_crosstask_panel(axes[1, 0])
    axes[1, 0].set_title("")
    for i, (task_key, title) in enumerate(VARIABILITY_TASK_PANELS):
        _plot_sigma_splithalf_panel(axes[1, i + 1], task_key, "", show_ylabel=(i == 0))

    # Row 3 -- autocorrelation, with NLL_RESP_NOISE_MODELS instead of the
    # NoisyRL_lambda-based default.
    data = _load_variance_autocorr_data(models=NLL_RESP_NOISE_MODELS,
                                        responses_path_fn=_nll_resp_noise_responses_path)
    axes[2, 0].axis("off")
    axes[2, 0].set_title("\u03c1 definition", color="0.3")
    schematic = _rasterize_svg(AUTOCORR_SCHEMATIC)
    if schematic is not None:
        axes[2, 0].imshow(schematic, aspect="auto")
    for i, (task_key, title) in enumerate(RESID_TASK_PANELS):
        ax = axes[2, i + 1]
        human_res, model_results, lags = data[task_key]
        _draw_variance_autocorr_panel(ax, task_key, title, human_res, model_results, lags,
                                      include_models=True, show_ylabel=(i == 0),
                                      models=NLL_RESP_NOISE_MODELS, model_colors=MODEL_COLORS)
        ax.set_xlabel("k")
        ax.set_ylabel("\u03c1" if i == 0 else "")

    legend_handles = [Line2D([0], [0], color=HUMAN_COLOR, lw=2.2, label="Human")]
    for m in NLL_RESP_NOISE_MODELS:
        legend_handles.append(Line2D([0], [0], color=MODEL_COLORS[m], lw=2.2,
                                     label=MODEL_LABEL.get(m, m)))
    axes[2, 3].legend(handles=legend_handles, fontsize=7, loc="upper right",
                      frameon=True, framealpha=0.9, ncol=1)

    out_path, _ = _save_fig(fig, "sigma_giant")
    plt.close(fig)
    return out_path
