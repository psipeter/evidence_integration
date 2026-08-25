#!/usr/bin/env python3
"""presentations/make_figures.py

Presentation-specific figures for the Dartmouth postdoc talk, tailored
variants of the analogous panels in scripts/figure_soltani_*.py. This is a
DELIBERATE exception to CLAUDE.md's "all new scripts go in scripts/" rule --
discussed and approved for this presentation-only file, kept out of scripts/
precisely so it is never mistaken for, or silently diverges from, the real
analysis pipeline. It copies and trims the RELEVANT PARTS of those working
scripts rather than editing them, so the working figures are never touched
by presentation-formatting decisions.

Each figure here is ONE PANEL sized to fill a single reveal.js slide
(SLIDE_FIGSIZE -- see its own comment for why it's ~1.91:1 rather than the
slide's nominal 3:2), saved as SVG (not PDF -- these need to embed directly
in HTML slides via Quarto, unlike scripts/'s PDF-only publication pipeline;
vector rather than PNG so line plots stay crisp at any projector/display
resolution) into presentations/figures/.

Run:
    venv/bin/python presentations/make_figures.py temporal_performance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import data_path, RUNS_DIR
from utils.aggregate import plot_error_aggregate
from utils.plot_style import draw_sig_line, pvalue_to_stars

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
# 1050x700 is the FULL slide canvas (3:2), but a title bar + footer eat a
# fixed chunk of that height regardless of slide content, so the actual
# USABLE area inside an .image-card is closer to ~1050x550 -- a wider ratio
# than the nominal slide. Sized empirically against this deck's own
# title+footer chrome (measured overflow at figsize=(9,6)/3:2, width=85%)
# rather than the raw slide dimensions, so a figure at this figsize clears
# the footer bar with margin at typical display widths (75-90%).
SLIDE_FIGSIZE = (9, 4.7)  # ~1.91:1

# new_tasks.svg's own "Numbers Task"/"Colors Task" labels use seaborn
# colorblind indices 0/1 (#0173b2/#de8f05) -- old_tasks.svg's "Balls
# Task"/"Snacks Task" reuse those SAME two indices, which is fine on their
# own separate slide but collides once all four tasks share one panel here.
# Balls keeps index 2 (teal). Colors keeps its own slide's orange
# (#de8f05, index 1) since numbers already has index 0 and nothing else
# in this figure uses orange. Snacks -- which would otherwise collide with
# balls/colors' own slide colors -- gets pink (#cc78bc, the NEF color in
# model_palette.svg) instead of its own slide's red-orange.
TASK_COLORS = {
    "numbers": "#0173b2",
    "colors": "#de8f05",
    "balls": "#029e73",
    "snacks": "#cc78bc",
}
TASK_LABELS = {
    "numbers": "Numbers task",
    "colors": "Colors task",
    "balls": "Balls task",
    "snacks": "Snacks task",
}
TASK_ORDER = ["balls", "snacks", "numbers", "colors"]  # old tasks, then new
DATASET_FOR_TASK = {"numbers": "soltani_numbers", "colors": "soltani_colors"}


def _apply_slide_style() -> None:
    """Presentation sizing -- bigger than utils.plot_style.apply_style()'s
    publication defaults (font.size 9), since this renders as ONE full-slide
    panel viewed on a projector, not a multi-panel PDF read up close."""
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.size": 15,
        "axes.labelsize": 17,
        "axes.titlesize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 14,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "lines.linewidth": 2.5,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    })


def _load_human_true_gt(task: str) -> pd.DataFrame:
    """Human data for one soltani task, with `ground_truth` set to the FIXED
    generative true_mean (numbers) / true_p (colors, mapped 2p-1 onto the
    [-1,1] response scale) -- i.e. gt_mode='true' from
    scripts/figure_soltani_temporal.py's own _add_ground_truth, trimmed down
    to just this one mode since it's the only one this figure needs: this
    talk is about how people integrate evidence toward a fixed target, not
    against the (also-moving) running mean of what they've seen so far.
    Canonical unsuffixed data/soltani_{numbers,colors}.pkl -- data collection
    for both tasks is finished, so no --datafile pilot suffix is needed."""
    dataset = DATASET_FOR_TASK[task]
    df = pd.read_pickle(data_path(f"{dataset}.pkl"))
    out = df[["pid", "trial", "observation", "response"]].copy()
    if task == "colors":
        out["ground_truth"] = df["true_p"] * 2.0 - 1.0
    else:
        out["ground_truth"] = df["true_mean"]
    return out


def _load_carrabin_true_gt() -> pd.DataFrame:
    """Human data for the balls task (carrabin), with `ground_truth` set to
    the FIXED generative true_p, mapped 2p-1 onto the [-1,1] response scale --
    the same transform figure_carrabin_temporal.py's own panel A uses
    (`true_p_resp = true_p*2-1`). observation is 1-indexed, 1-5."""
    df = pd.read_pickle(data_path("carrabin.pkl"))
    out = df[["pid", "trial", "observation", "response"]].copy()
    out["ground_truth"] = df["true_p"] * 2.0 - 1.0
    return out


def _load_yoo_true_gt() -> pd.DataFrame:
    """Human data for the snacks task (yoo), with `ground_truth` set to the
    expanding (cumulative) mean of the raw stimulus stream `value` per
    (pid, trial) -- yoo has no separate fixed generative parameter column at
    all, so this expanding mean IS its 'true value': the same convention
    figure_yoo_temporal.py's own panel A uses (`true_mean` there). observation
    is 1-indexed, 1-30."""
    df = pd.read_pickle(data_path("yoo.pkl")).sort_values(
        ["pid", "trial", "observation"]).copy()
    df["ground_truth"] = (df.groupby(["pid", "trial"])["value"]
                           .expanding().mean().values)
    return df[["pid", "trial", "observation", "response", "ground_truth"]]


# One loader per task, all returning the same [pid, trial, observation,
# response, ground_truth] shape regardless of each source dataset's own
# column names/scales/indexing -- so make_temporal_performance can treat
# every task identically.
TASK_LOADERS = {
    "numbers": lambda: _load_human_true_gt("numbers"),
    "colors": lambda: _load_human_true_gt("colors"),
    "balls": _load_carrabin_true_gt,
    "snacks": _load_yoo_true_gt,
}


def _sq_err_long(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(pid, trial, observation) squared error against `ground_truth`.
    Left un-aggregated on purpose -- RMSE's averaging step must be composed
    with the aggregation exactly once; see utils.aggregate.plot_error_aggregate's
    own docstring for why a "median of RMSEs" isn't well-defined otherwise."""
    return df.assign(sq_err=(df["response"] - df["ground_truth"]) ** 2)


def make_temporal_performance() -> Path:
    """One panel: human RMSE-to-true-value vs observation, one line per task
    (balls, snacks, numbers, colors) -- the "learning"/evidence-integration
    curve for this talk. Tailored variant of the analogous panel A/col-1 in
    figure_carrabin_temporal.py / figure_yoo_temporal.py /
    figure_soltani_temporal.py: human-only (no model overlays), one combined
    panel instead of each dataset's own separate figure, ground truth fixed
    to each task's own "true value" convention (see each TASK_LOADERS entry).

    Uses the SAME aggregation convention as the working pipeline
    (hier_mean_median + its default 'ci' band -- see utils/aggregate.py)
    via the shared plot_error_aggregate helper, so each curve's shape is
    directly comparable to the analysis figures rather than a bespoke
    presentation-only statistic.

    Observation is NOT normalized across tasks -- each task's own raw
    (1-indexed for balls/snacks, 0-indexed for numbers/colors) observation
    count is plotted as-is, so the four curves' different x-extents (balls
    1-5, snacks 1-30, numbers/colors 0-14) directly reflect how many
    observations each task's own design actually asks participants to
    integrate over.
    """
    _apply_slide_style()
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE, constrained_layout=True)

    handles, labels = [], []
    for task in TASK_ORDER:
        human = TASK_LOADERS[task]()
        sq_err_df = _sq_err_long(human)
        color = TASK_COLORS[task]
        plot_error_aggregate(ax, sq_err_df, color, mode="hier_mean_median",
                              zorder_line=3, zorder_fill=1, errorbar_kind=None)
        handles.append(Line2D([0], [0], color=color, lw=3))
        labels.append(TASK_LABELS[task])

    ax.set_xlabel("Observation")
    ax.set_ylabel("Human Task Error \n (median rmse vs ground truth)")
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, frameon=True, framealpha=0.9, loc="upper right")
    sns.despine(ax=ax, top=True, right=True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "temporal_performance.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


# ── Model performance across all four tasks ─────────────────────────────────

# Model colors, matching presentations/images/model_palette.svg's own
# seaborn-colorblind indices EXCEPT RL_lambda/NEF, which are both pinned to
# #d55e00 (red-orange) per explicit request -- rather than RL_lambda's own
# index-3 color and NEF's own index-4 pink -- because for colors/numbers (see
# below) RL_lambda's fitted loss is being shown IN PLACE OF NEF (not yet fit
# for those two datasets), so giving both the same color signals "these are
# playing the same conceptual role" rather than implying two different,
# independently-fit models.
MODEL_COLORS = {
    "Mean": "#0173b2",
    "LeakyIntegrator": "#de8f05",
    "PrimacyRecency": "#029e73",
    "RL_lambda": "#d55e00",
    "NEF": "#d55e00",
}

# (task_key, panel title, model list). Model lists intentionally differ in
# their last entry: carrabin/yoo have real fitted NEF; colors/numbers do not
# (task_backend model fitting against real data hasn't been run yet -- see
# CLAUDE.md), so RL_lambda's own fit stands in for it there -- the same
# substitution figure_soltani_performance.py's own SIG_REFERENCE historically
# used while NEF was unwired for these two datasets (see that script's module
# docstring).
TASK_PANELS = [
    ("balls", "Balls task", ["Mean", "LeakyIntegrator", "PrimacyRecency", "NEF"]),
    ("snacks", "Snacks task", ["Mean", "LeakyIntegrator", "PrimacyRecency", "NEF"]),
    ("colors", "Colors task", ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda"]),
    ("numbers", "Numbers task", ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda"]),
]


def _model_fit_path(task_key: str, model: str) -> Path:
    """Path to one (task, model)'s *_performance.pkl -- the fitted CV loss
    (RMSE to human responses), read the same way every working figure does
    (via _get_loss, never a hardcoded column name). Each task family keeps
    its own run-folder/dataset-stem convention:
      balls (carrabin)  -> data/runs/carrabin/{model}_carrabin_performance.pkl
      snacks (yoo)       -> data/runs/yoo/{model}_yoo_performance.pkl, except
                            NEF which lives in data/runs/refit/ (matching this
                            project's own default --nef_folder refit)
      colors/numbers     -> data/runs/soltani/{model}_soltani_{task}_performance.pkl
                            (one shared run folder for both soltani tasks)
    """
    if task_key == "balls":
        return RUNS_DIR / "carrabin" / f"{model}_carrabin_performance.pkl"
    if task_key == "snacks":
        run_dir = RUNS_DIR / ("refit" if model == "NEF" else "yoo")
        return run_dir / f"{model}_yoo_performance.pkl"
    dataset = "soltani_colors" if task_key == "colors" else "soltani_numbers"
    return RUNS_DIR / "soltani" / f"{model}_{dataset}_performance.pkl"


def _get_loss(perf: pd.DataFrame) -> pd.Series:
    """Never hardcode cv_loss_mean -- project convention (utils.plot_style /
    every figure_*.py script's own _get_loss)."""
    return perf["loss"] if "loss" in perf.columns else perf["cv_loss_mean"]


def _compute_sig_lines(plot_df: pd.DataFrame, x_col: str, y_col: str,
                        order: list[str], ref_label: str) -> list[tuple[int, int, str]]:
    """Paired-Wilcoxon comparisons from `ref_label` to every other model in
    `order`, returning ONLY the ones where `ref_label` is significantly
    BETTER (lower RMSE) -- a one-sided variant of
    utils.plot_style.annotate_nef_comparisons, which flags any significant
    two-sided difference regardless of direction. That direction-blindness
    is the right default for the working figures ("does NEF differ from
    this model"), but this talk's point is narrower ("is our model
    better"), so a comparison NEF/RL_lambda significantly LOSES is left out
    rather than flagged the same way a win would be.

    Returns (x1, x2, stars) tuples, sorted nearest-first, with no drawing or
    ylim side effects -- kept separate from the actual line-drawing so a
    caller can compute every panel's bars BEFORE deciding how much headroom
    the shared y-axis needs (see make_model_performance's own sharey note).
    """
    x_positions = {m: i for i, m in enumerate(order)}
    if ref_label not in x_positions:
        return []

    candidates = [m for m in order if m != ref_label]
    pairs = sorted(candidates, key=lambda m: abs(x_positions[m] - x_positions[ref_label]))

    sig_lines = []
    for m_other in pairs:
        p1 = plot_df.loc[plot_df[x_col] == m_other, ["pid", y_col]]
        p2 = plot_df.loc[plot_df[x_col] == ref_label, ["pid", y_col]]
        merged = p1.merge(p2, on="pid", suffixes=("_other", "_ref"))
        if len(merged) < 4:
            continue
        other_vals = merged[f"{y_col}_other"].to_numpy(dtype=float)
        ref_vals = merged[f"{y_col}_ref"].to_numpy(dtype=float)
        diff = other_vals - ref_vals  # positive => reference has LOWER (better) RMSE
        if np.all(diff == 0) or np.nanstd(diff) == 0:
            continue
        try:
            res = wilcoxon(other_vals, ref_vals)
        except ValueError:
            continue
        p = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
        stars = pvalue_to_stars(p)
        if stars == "ns":
            continue
        if np.median(diff) <= 0:
            # Reference is not actually better on this comparison -- a
            # significant difference in the WRONG direction is not a win,
            # so it's excluded rather than drawn.
            continue
        sig_lines.append((x_positions[m_other], x_positions[ref_label], stars))

    sig_lines.sort(key=lambda t: abs(t[1] - t[0]))
    return sig_lines


def make_model_performance() -> Path:
    """1x4 panel: model fit (RMSE to human responses), one panel per task
    (balls/carrabin, snacks/yoo, colors, numbers) -- how well each
    cognitive/spiking model reproduces what people actually did. Tailored
    combination of figure_carrabin_performance.py's rendered panel C,
    figure_yoo_performance.py's rendered panel C (its 1x3 layout has no
    panel D; C is the analogous "model fit" panel there -- see chat), and
    figure_soltani_performance.py's rendered panels C/F (colors/numbers'
    own col-3 model-fit panel in its 2x3 grid).

    All three sources already share the same metric -- fitted CV loss on the
    canonical [-1,1] RMSE scale, read via _get_loss -- confirmed directly
    against the saved data/runs/soltani/*_performance.pkl files (loss range
    ~0.03-0.5, matching carrabin/yoo's own documented RMSE ranges) before
    writing this, despite that module's own docstring mentioning an earlier,
    now-reverted percent-scale version. No rescaling applied.

    NEF has not been fit yet for colors/numbers (task_backend data -- see
    CLAUDE.md's "Data pipeline" section) -- RL_lambda's own fitted loss is
    shown in that slot instead, colored identically to NEF (MODEL_COLORS)
    rather than mislabeled as NEF: the x-axis stays honest about which model
    actually produced each box.

    Significance bars (paired Wilcoxon, via _compute_sig_lines) are drawn
    from NEF (balls/snacks) or RL_lambda (colors/numbers) outward to each
    other model, ONLY where the reference is significantly better -- a
    one-sided version of utils.plot_style.annotate_nef_comparisons, since
    this talk's point is narrower than the working figures' ("is our model
    better", not "does it differ").

    X-axis ticks/labels are dropped per panel (repeating the same four model
    names four times ate a lot of vertical space for no new information) in
    favor of ONE shared legend below all four panels -- Mean/LeakyIntegrator/
    PrimacyRecency plus a single combined "NEF / RL_lambda" swatch, since
    those two are already the same color for the reason given above.

    Y-AXIS IS SHARED ACROSS ALL FOUR PANELS (sharey=True), so absolute RMSE
    is now directly comparable between tasks, not just between models within
    one task -- worth it despite compressing the smaller-scale tasks
    (numbers' boxes end up visually short next to snacks') because the whole
    POINT of this figure is a cross-task comparison.

    THIS IS WHY SIGNIFICANCE-BAR HEADROOM CAN'T BE ADDED PER PANEL, THE WAY
    _annotate_reference_better used to (single-panel figures never had this
    problem). With sharey, all four Axes share one underlying y-limits
    object -- calling ax.set_ylim(top=...) on panel 1 to make room for its
    bars would silently push panel 2/3/4's baseline too, and since each
    panel's own headroom calculation reads ax.get_ylim() AFTER the previous
    panel already changed it, the required space would compound across
    panels instead of just reflecting each panel's own bars. Fixed by
    computing every panel's sig_lines FIRST (pure data, no axis side
    effects), figuring out the max extra headroom any single panel needs
    for its own tallest stack of bars, then extending the shared ylim ONCE
    (via axes[0]; since sharey, this applies to every panel) before actually
    drawing any line -- so no panel's calculation can see, or be thrown off
    by, another panel's bars.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 4, figsize=(11, 4.4), sharey=True,
                             constrained_layout=True)

    panel_data = []  # (ax, task_key, title, order, plot_df) for panels with data
    for i, (ax, (task_key, title, models)) in enumerate(zip(axes, TASK_PANELS)):
        rows = []
        for model in models:
            path = _model_fit_path(task_key, model)
            if not path.exists():
                print(f"  (missing {path.name} -- skipping {model} for {task_key})")
                continue
            perf = pd.read_pickle(path)
            rows.append(pd.DataFrame({
                "pid": perf["pid"],
                "rmse": _get_loss(perf),
                "model": model,
            }))

        if not rows:
            ax.text(0.5, 0.5, "No fitted models\nfor this task",
                    ha="center", va="center", transform=ax.transAxes,
                    color="0.5", style="italic")
            ax.set_title(title)
            continue

        plot_df = pd.concat(rows, ignore_index=True)
        order = [m for m in models if m in plot_df["model"].unique()]
        pal = {m: MODEL_COLORS[m] for m in order}

        sns.boxplot(data=plot_df, x="model", y="rmse", order=order,
                    hue="model", palette=pal, legend=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Model fit (RMSE to\nhuman responses)" if i == 0 else "")
        ax.tick_params(axis="y", labelleft=(i == 0))
        ax.set_xticks([])
        sns.despine(ax=ax, top=True, right=True)
        panel_data.append((ax, task_key, title, order, plot_df))

    axes[0].set_ylim(bottom=0)  # shared -- applies to every panel

    # Compute every panel's significance bars BEFORE touching the (shared)
    # ylim -- see the sharey note above for why order matters here.
    y_lo, y_hi = axes[0].get_ylim()
    dy_step = (y_hi - y_lo) * 0.07
    per_panel_sig_lines = []
    max_bars = 0
    for ax, task_key, title, order, plot_df in panel_data:
        ref_label = "NEF" if task_key in ("balls", "snacks") else "RL_lambda"
        sig_lines = (_compute_sig_lines(plot_df, "model", "rmse", order, ref_label)
                    if ref_label in order else [])
        per_panel_sig_lines.append((ax, sig_lines))
        max_bars = max(max_bars, len(sig_lines))

    if max_bars:
        axes[0].set_ylim(top=y_hi + dy_step * 0.5 + max_bars * dy_step * 2.0 + dy_step)

    for ax, sig_lines in per_panel_sig_lines:
        y_current = y_hi + dy_step * 0.5
        for x1, x2, stars in sig_lines:
            draw_sig_line(ax, x1, x2, y_current, stars)
            y_current += dy_step * 2.0

    legend_handles = [Patch(facecolor=MODEL_COLORS[m], label=m)
                      for m in ["Mean", "LeakyIntegrator", "PrimacyRecency"]]
    legend_handles.append(Patch(facecolor=MODEL_COLORS["NEF"], label="NEF / RL_lambda"))
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=4,
               frameon=True, framealpha=0.9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "model_performance.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


FIGURES = {
    "temporal_performance": make_temporal_performance,
    "model_performance": make_model_performance,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=sorted(FIGURES),
                        help="Which presentation figure to (re)generate.")
    args = parser.parse_args()
    FIGURES[args.figure]()


if __name__ == "__main__":
    main()
