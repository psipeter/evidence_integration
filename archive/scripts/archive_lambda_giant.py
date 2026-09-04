"""
scripts/archive_lambda_giant.py

Archived: make_lambda_giant() from scripts/make_paper_figures.py.

This was a 4-row, 4-column MEGA figure stacking four existing figures' own
panels unchanged (row 1: make_response_change's 4 task panels; rows 2-3:
make_lambda_overview's 2x4 content; row 4: make_lambda_model_correlation's
3 panels in cols 2-4, col 1 empty). Retired this session and split into six
separate figures, per instruction:

  - make_lambda_main       -- rows 1-2, cols 2-4 (snacks/colors/numbers only,
                               dropping the balls/demo column) -- 2x3 grid.
  - make_lambda_metric     -- row 2 col 1 (the inlined "lambda definition"
                               demo panel), as its own standalone figure, to
                               be composited into lambda_main as an Inkscape
                               inset by hand from the saved SVG.
  - make_lambda_balls      -- row 1 col 1 (balls-task response-change panel)
                               -- doesn't show the expected trend, moved to
                               supplementary on its own.
  - make_lambda_reliability -- row 3 cols 2-4 (splithalf panels for snacks/
                               colors/numbers) -- supplementary.
  - make_lambda_humanvmodel -- row 4 cols 2-4 (model-correlation panels for
                               snacks/colors/numbers) -- supplementary. NOTE:
                               this content is identical to the pre-existing
                               make_lambda_model_correlation() (same panels,
                               same helper); the person may want to retire
                               that older name once this is confirmed.
  - make_lambda_sigma_crosstask -- NEW combination, not a piece of the
                               original giant: 2x1 grid, r1c1 = row 3 col 1
                               (lambda crosstask panel, _plot_lambda_crosstask_
                               panel) and r2c1 = the analogous sigma crosstask
                               panel from make_sigma_giant
                               (_plot_sigma_crosstask_panel).

Kept here for reference/history, not deleted -- see docs/HISTORY.md if the
full rationale for the original 4x4 stack is ever needed again. NOT
standalone-runnable as archived: it references module-level state
(TASK_PANELS, LAMBDA_TASK_PANELS, TASK_COLORS, HUMAN_COLOR, MODEL_COLORS,
MODEL_LABEL, _load_response_change_data, _load_lambda_delta,
_fit_lambda_series, LAMBDA_N_OFFSET, _plot_lambda_distribution,
_plot_lambda_crosstask_panel, _plot_lambda_splithalf_panel,
_plot_lambda_model_corr_panel, _power_law, _human_data_path, _save_fig,
FIGURES_DIR) that still lives in scripts/make_paper_figures.py and was not
duplicated here.
"""

# ── Original imports needed by this function ──────────────────────────────
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import seaborn as sns


def make_lambda_giant() -> Path:
    """4-row, 4-column MEGA figure stacking four existing figures' own
    panels, UNCHANGED, into one combined figure, per instruction:
      Row 1: make_response_change's own 4 panels (balls/snacks/colors/
        numbers -- TASK_PANELS).
      Rows 2-3: make_lambda_overview's own 2x4 content, unchanged (row 2 =
        lambda_overview's row 1: demo + 3 KDEs; row 3 = lambda_overview's
        row 2: crosstask + 3 splithalf, no titles).
      Row 4: make_lambda_model_correlation's own 3 panels (snacks/colors/
        numbers -- LAMBDA_TASK_PANELS), in columns 2-4 (matching rows 2-3's
        own snacks/colors/numbers column positions); column 1 is empty
        (turned off) since that figure has no 4th/non-task panel of its
        own to put there.

    Row 1's own columns are [balls, snacks, colors, numbers] (TASK_PANELS'
    own order) -- NOT task-aligned with rows 2-4's own column 1 (a
    non-task demo/crosstask panel there, or empty in row 4). This is a
    direct, unmodified stack of each source figure's own existing column
    layout, not a reshuffle -- unlike make_lambda_overview's own
    combination of two SAME-roster figures, response_change's own roster
    includes balls (no lambda fit exists for it), so there's no shared
    task set across all 4 rows to align columns by in the first place.

    Row 1's legend (Human + 5 models) is drawn INSIDE the last panel
    (numbers, upper right, small font) rather than as a figure-spanning
    legend below the whole figure -- make_response_change's own
    fig.legend(loc="outside lower center") would otherwise land at the
    very bottom of this 4-row figure, nowhere near the row it belongs to.

    Every panel reuses the EXACT SAME helper functions the four source
    figures already call -- no panel-drawing logic is duplicated here,
    only the layout differs. All four source figures are left completely
    unchanged and still produce their own separate outputs.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(4, 4, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 1.8),
                             constrained_layout=True)

    # Row 1 -- make_response_change's own panels, unchanged.
    data = _load_response_change_data()
    obs_max_by_task = {}
    for task_key in data:
        human_delta, models, _ = data[task_key]
        obs_vals = [human_delta["observation"].max()] + [
            df["observation"].max() for df in models.values() if len(df)]
        obs_max_by_task[task_key] = max(obs_vals)
    for i, (task_key, title, models_list) in enumerate(TASK_PANELS):
        human_delta, models, _ = data[task_key]
        ax = axes[0, i]
        ylabel = "Median \u0394R" if i == 0 else ""
        _draw_response_change_panel(ax, human_delta, models, include_models=True,
                                    ylabel=ylabel, obs_max=obs_max_by_task[task_key])
        ax.set_title(title, color=TASK_COLORS[task_key])
        ax.tick_params(axis="y", labelleft=(i == 0))
    axes[0, 0].set_ylim(bottom=0)
    legend_handles = [Line2D([0], [0], color=HUMAN_COLOR, lw=2, label="Human")]
    legend_handles += [Line2D([0], [0], color=MODEL_COLORS[m], lw=2, label=MODEL_LABEL.get(m, m))
                       for m in ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"]]
    axes[0, 3].legend(handles=legend_handles, fontsize=7, loc="upper right",
                      frameon=True, framealpha=0.9, ncol=1)

    # Row 2, col 1 -- INLINED rather than calling _plot_lambda_demo, since
    # the changes here (regplot-with-binning instead of a bare scatter,
    # plus the shortened/mean-dropped ylabel) are specific to this one
    # combined figure and shouldn't alter that shared helper's behavior
    # for make_lambda_human/make_lambda_overview, which still call it
    # as-is. Fit itself is UNCHANGED (same curve_fit call, same per-
    # observation mean curve it's fit to) -- only the VISUAL replacement
    # of individual scatter points with sns.regplot's own binned mean+CI
    # markers changes, giving a within-observation variance indication
    # the single mean point never showed. x_bins is passed the actual
    # sorted unique observation values (not a bin COUNT), so every bin is
    # exactly one observation -- the "binsize=1" the person asked for --
    # rather than seaborn's own evenly-SPACED (not necessarily one-per-
    # observation) default binning.
    demo_task_key = "numbers"
    ax = axes[1, 0]
    human_delta_demo = _load_lambda_delta(demo_task_key, _human_data_path(demo_task_key))
    lam_demo = _fit_lambda_series(human_delta_demo, LAMBDA_N_OFFSET[demo_task_key])
    demo_pid = (lam_demo - lam_demo.median()).abs().idxmin()
    g = human_delta_demo[human_delta_demo["pid"] == demo_pid].copy()
    g["n"] = g["observation"].astype(float) + LAMBDA_N_OFFSET[demo_task_key]
    curve = g.groupby("observation")["delta"].mean().dropna().sort_index()
    n = curve.index.values.astype(float) + LAMBDA_N_OFFSET[demo_task_key]
    y = curve.values.astype(float)
    popt, _ = curve_fit(_power_law, n, y, p0=[0.1, 0.5],
                        bounds=([0.0, 0.0], [2.0, 2.0]), maxfev=2000)
    A_fit, lam_fit = popt

    bin_centers = sorted(g["n"].unique())
    sns.regplot(data=g, x="n", y="delta", x_bins=bin_centers, fit_reg=False,
               color=HUMAN_COLOR, ax=ax)
    n_smooth = np.linspace(n.min(), n.max(), 200)
    fit_line, = ax.plot(n_smooth, _power_law(n_smooth, A_fit, lam_fit),
                        color=HUMAN_COLOR, lw=2.5, zorder=4)

    ax.set_xlabel("Observation")
    ax.set_ylabel("\u0394R")
    ax.set_title("\u03bb definition", color="0.3")
    ax.set_xlim(left=0)
    ax.set_xticks(sorted(set([0.0] + list(ax.get_xticks()))))
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, top=True, right=True)

    blank = Line2D([], [], linestyle="none")
    leg = ax.legend(
        [fit_line, blank, blank],
        [r"$A n^{-\lambda}$, $\lambda=%.2f$" % lam_fit,
         f"Participant #{demo_pid}",
         TASK_LABELS[demo_task_key]],
        loc="lower right", frameon=True, framealpha=0.9,
        handlelength=1.2, fontsize=8,
    )
    leg.get_texts()[2].set_color(TASK_COLORS[demo_task_key])

    # Row 2, cols 2-4 -- _plot_lambda_distribution unchanged, but the
    # task-colored title is dropped (row 1 already shows it -- "using
    # just the top row for those titles") and the one visible ylabel is
    # shortened to "Density".
    for i, (task_key, title) in enumerate(LAMBDA_TASK_PANELS):
        ax = axes[1, i + 1]
        human_delta = _load_lambda_delta(task_key, _human_data_path(task_key))
        lam = _fit_lambda_series(human_delta, LAMBDA_N_OFFSET[task_key])
        _plot_lambda_distribution(ax, lam, task_key)
        ax.set_ylabel("Density" if i == 0 else "")
        ax.tick_params(axis="y", labelleft=(i == 0))

    _plot_lambda_crosstask_panel(axes[2, 0])
    axes[2, 0].set_title("")
    for i, (task_key, title) in enumerate(LAMBDA_TASK_PANELS):
        _plot_lambda_splithalf_panel(axes[2, i + 1], task_key, "", show_ylabel=(i == 0))

    # Row 4 -- make_lambda_model_correlation's own panels; task-colored
    # titles cleared the same way as row 2 (row 1 already shows them).
    # _plot_lambda_model_corr_panel has no title parameter of its own, so
    # it's cleared with an explicit ax.set_title("") right after the call.
    axes[3, 0].axis("off")
    legend_locs = {"snacks": "lower right"}
    for i, (task_key, title) in enumerate(LAMBDA_TASK_PANELS):
        _plot_lambda_model_corr_panel(axes[3, i + 1], task_key, title, show_ylabel=(i == 0),
                                      legend_loc=legend_locs.get(task_key, "best"))
        axes[3, i + 1].set_title("")

    out_path, _ = _save_fig(fig, "lambda_giant")
    plt.close(fig)
    return out_path
