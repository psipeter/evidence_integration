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
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, pearsonr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import data_path, RUNS_DIR
from utils.aggregate import plot_error_aggregate, plot_delta_aggregate
from utils.plot_style import draw_sig_line, pvalue_to_stars

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
# ONE shared figsize for EVERY figure this script produces, regardless of
# panel count -- so every generated figure fills the same visual footprint
# at the same display width% in the deck, rather than each figure picking
# its own canvas size. Midway between what "Responses improve with data"
# (a single panel, (9, 4.7)) and "Model Performance"/"Response Change Decay"
# (four panels, (11, 6.2)) each used before being unified -- one panel and
# four panels don't need the same canvas size to look equally full, but a
# consistent canvas across every figure is what was asked for here.
FIGURE_SIZE = (10.6, 5.45)

# 4-color TASK palette (distinct from MODEL_COLORS/other colors used
# throughout this deck -- Dartmouth green #00693e, the tab10 red/purple used
# for equation highlighting, and every MODEL_COLORS hue). Chosen by searching
# seaborn's husl_palette (evenly-hue-spaced, seaborn-native) over starting
# hue for the offset that maximizes the WORST-CASE CIELAB deltaE both (a)
# between these 4 colors and (b) against every color already in use in this
# presentation -- not just picked by eye. Candidates considered and
# rejected: seaborn's own remaining colorblind indices 6-9 (pink/gray/
# yellow/sky-blue -- the gray and sky-blue collide with Human-curve gray and
# Mean's blue), Set2/Dark2/tab10's own remaining colors, Accent (a pale
# ffff99 yellow projects poorly). This palette's worst-case separation
# (deltaE ~22-23) comfortably beats all of those (deltaE as low as 0.4-17).
TASK_COLORS = {
    "balls": "#c86741",
    "snacks": "#638e3f",
    "colors": "#458d95",
    "numbers": "#bd54de",
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
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)

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
    # Index 5 of the same seaborn-colorblind MODEL_ORDER palette (see
    # utils/soltani_models.py's own module docstring: "NoisyRL_lambda is last
    # so that adding it left every existing model's colour untouched") --
    # NOT reusing #d55e00: unlike RL_lambda (a deterministic stand-in for
    # NEF elsewhere in this deck), NoisyRL_lambda is a genuinely DIFFERENT
    # model (adds two real noise terms) playing a DIFFERENT role in the
    # variability figure below -- the one soltani model with actual
    # nonzero within-qid-repeat variance, standing in for NEF (not yet fit
    # for colors/numbers) in that specific sense only.
    "NoisyRL_lambda": "#ca9161",
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
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, sharey=True,
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
            ax.set_title(title, color=TASK_COLORS[task_key])
            continue

        plot_df = pd.concat(rows, ignore_index=True)
        order = [m for m in models if m in plot_df["model"].unique()]
        pal = {m: MODEL_COLORS[m] for m in order}

        sns.boxplot(data=plot_df, x="model", y="rmse", order=order,
                    hue="model", palette=pal, legend=False, ax=ax)
        ax.set_title(title, color=TASK_COLORS[task_key])
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
    fig.get_layout_engine().set(h_pad=0.25)
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=4,
               frameon=True, framealpha=0.9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "model_performance.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


# ── Response change decay across all four tasks ───────────────────

# Same 4-model list per task as make_model_performance (Mean/LeakyIntegrator/
# PrimacyRecency + NEF or RL_lambda), reused here for the identical reason.
DELTA_TASK_PANELS = [
    ("balls", "Balls task", "NEF"),
    ("snacks", "Snacks task", "NEF"),
    ("colors", "Colors task", "RL_lambda"),
    ("numbers", "Numbers task", "RL_lambda"),
]

# First observation whose |delta response| is included, PER TASK -- copied
# from each source figure's own established convention, not reinvented:
#   balls (carrabin): 1 -- every observation has a defined delta (see
#     FIRST_OBS_IS_RESPONSE below; there is nothing to drop).
#   snacks (yoo): 2 -- figure_yoo_temporal.py's own _abs_delta_long drops the
#     first defined delta (at its 1-indexed observation=1) unconditionally.
#   colors: 2, numbers: 1 -- figure_soltani_temporal.py's own DELTA_MIN_OBS;
#     colors' first delta is near-degenerate/bimodal with binary evidence
#     (see that script's own module-level comment above its DELTA_MIN_OBS).
DELTA_MIN_OBS = {"balls": 1, "snacks": 2, "colors": 2, "numbers": 1}

# carrabin's own panel B treats the FIRST observation's delta as |response|
# (a "change" from a neutral zero starting point) rather than leaving it
# undefined/NaN like every other task -- applied identically to Human AND
# every model's response file in that script, so it's applied the same way
# here, not just to Human.
FIRST_OBS_IS_RESPONSE = {"balls": True, "snacks": False, "colors": False, "numbers": False}

HUMAN_COLOR = "0.3"  # matches every figure_*.py script's own HUMAN_COLOR


def _human_data_path(task_key: str) -> Path:
    if task_key == "balls":
        return data_path("carrabin.pkl")
    if task_key == "snacks":
        return data_path("yoo.pkl")
    return data_path("soltani_colors.pkl" if task_key == "colors" else "soltani_numbers.pkl")


def _delta_responses_path(task_key: str, model: str) -> Path:
    """Path to one (task, model)'s *_responses.pkl -- the actual per-
    observation response SEQUENCE (needed to compute a delta), NOT the
    scalar *_performance.pkl loss make_model_performance reads. Two
    quirks, both copied from each source figure's own established choice
    rather than the plainer default:
      balls (carrabin): NEF specifically comes from
        NEF_carrabin_responses_mle.pkl (the MLE-fitted variant), matching
        figure_carrabin_temporal.py's own panel B -- NOT the RMSE-fitted
        NEF_carrabin_responses.pkl make_model_performance uses for ITS
        (different) panel C. Different panels of the same working script
        can and do use different fit variants; this one follows panel B's.
      snacks (yoo): NEF comes from data/runs/refit/, matching this
        project's own default --nef_folder refit for yoo figures.
    """
    if task_key == "balls":
        if model == "NEF":
            return RUNS_DIR / "carrabin" / "NEF_carrabin_responses_mle.pkl"
        return RUNS_DIR / "carrabin" / f"{model}_carrabin_responses.pkl"
    if task_key == "snacks":
        run_dir = RUNS_DIR / ("refit" if model == "NEF" else "yoo")
        return run_dir / f"{model}_yoo_responses.pkl"
    dataset = "soltani_colors" if task_key == "colors" else "soltani_numbers"
    return RUNS_DIR / "soltani" / f"{model}_{dataset}_responses.pkl"


def _abs_delta_long(df: pd.DataFrame, min_obs: int,
                     first_obs_is_response: bool) -> pd.DataFrame:
    """Per-(pid, trial, observation) |delta response|, one row per defined
    delta. Applied IDENTICALLY to Human and every model's response file --
    same function, same flags per task -- so a task's own quirks
    (first_obs_is_response, min_obs) never accidentally apply to one
    source but not another."""
    pieces = []
    for (_, _), g in df.groupby(["pid", "trial"], sort=False):
        g = g.sort_values("observation").copy()
        g["delta"] = g["response"].diff().abs()
        if first_obs_is_response:
            first_idx = g["observation"].idxmin()
            g.loc[first_idx, "delta"] = abs(g.loc[first_idx, "response"])
        pieces.append(g)
    if not pieces:
        return pd.DataFrame(columns=["pid", "trial", "observation", "delta"])
    out = pd.concat(pieces, ignore_index=True)
    return out[out["observation"] >= min_obs].dropna(subset=["delta"])


def _load_response_change_data() -> dict:
    """task_key -> (human_delta_df, {model: delta_df}, title, ref_label).
    Loaded ONCE and shared between the human-only and human+models figure
    passes below, so both read the exact same underlying data."""
    out = {}
    for task_key, title, ref_label in DELTA_TASK_PANELS:
        min_obs = DELTA_MIN_OBS[task_key]
        first_resp = FIRST_OBS_IS_RESPONSE[task_key]

        human_df = pd.read_pickle(_human_data_path(task_key))
        human_delta = _abs_delta_long(
            human_df[["pid", "trial", "observation", "response"]], min_obs, first_resp)

        models = {}
        for model in ["Mean", "LeakyIntegrator", "PrimacyRecency", ref_label]:
            path = _delta_responses_path(task_key, model)
            if not path.exists():
                print(f"  (missing {path.name} -- skipping {model} for {task_key})")
                continue
            mdf = pd.read_pickle(path)[["pid", "trial", "observation", "response"]]
            models[model] = _abs_delta_long(mdf, min_obs, first_resp)

        out[task_key] = (human_delta, models, title, ref_label)
    return out


def _four_xticks(obs_max: float) -> list[int]:
    """[0, ..., obs_max] with exactly 2 evenly-spaced intermediate ticks (4
    total), rounded to whole observations since that's what they are."""
    raw = np.linspace(0, obs_max, 4)
    return sorted(set(int(round(v)) for v in raw))


def _draw_response_change_panel(ax, human_delta: pd.DataFrame, models: dict,
                                ref_label: str, include_models: bool,
                                ylabel: str, obs_max: float) -> None:
    plot_delta_aggregate(ax, human_delta, HUMAN_COLOR, "hier_mean_median",
                         zorder_line=3, zorder_fill=1, errorbar_kind=None)
    if include_models:
        for i, model in enumerate(["Mean", "LeakyIntegrator", "PrimacyRecency", ref_label]):
            if model in models:
                plot_delta_aggregate(ax, models[model], MODEL_COLORS[model],
                                     "hier_mean_median", zorder_line=4 + i,
                                     zorder_fill=1, errorbar_kind=None)
    ax.set_xlabel("Observation")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0, right=obs_max)
    ax.set_xticks(_four_xticks(obs_max))
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, top=True, right=True)


def make_response_change() -> list[Path]:
    """Two 1x4-panel figures (same layout as make_model_performance: one
    panel per task, sharey=True, one shared legend below instead of a
    per-panel one) for a two-stage reveal.js build: response_change_human.svg
    (Human only) advances to response_change_full.svg (+ all 4 models) via
    an .r-stack fragment, the same pattern presentations/images/
    complexity_*.svg already uses for the "Theories and Models" slide.

    Metric: median (across pids) of each pid's own mean |delta response|
    vs observation -- the established "response change" panel from
    figure_carrabin_temporal.py's panel B / figure_yoo_temporal.py's panel B
    / figure_soltani_temporal.py's col 2, via the shared
    utils.aggregate.plot_delta_aggregate (hier_mean_median), so this curve's
    shape is directly comparable to those working figures. Per-task
    first-observation/minimum-observation conventions are copied from each
    source rather than reinvented -- see DELTA_MIN_OBS/FIRST_OBS_IS_RESPONSE
    and _delta_responses_path's own docstrings for exactly which, and why
    carrabin's NEF comes from a DIFFERENT fit variant (MLE) here than in
    make_model_performance (RMSE) -- that mismatch is inherited from the two
    source panels this figure and that one are each modeled on, not
    introduced by combining them.

    BOTH STAGES SHARE IDENTICAL Y-LIMITS PER PANEL, computed from the FULL
    (human+models) pass and then applied explicitly to the human-only pass
    -- not left to each figure's own autoscale. Two separately-saved SVGs
    have no shared Axes object the way sharey=True panels within ONE figure
    do, so without this the human-only curve would render at a different
    vertical scale than it does once models are added, and the r-stack
    overlay would visibly jump/rescale on that fragment's build step instead
    of the models cleanly drawing in on top of an unchanged human curve.
    """
    _apply_slide_style()
    data = _load_response_change_data()

    # Max observation actually plotted per task (human UNION every model
    # that loaded), used for the shared x-axis range/ticks below -- computed
    # once so both stages use the identical range regardless of whether a
    # given model's file happened to be missing in one stage's data.
    obs_max_by_task = {}
    for task_key in data:
        human_delta, models, _, _ = data[task_key]
        obs_vals = [human_delta["observation"].max()] + [
            df["observation"].max() for df in models.values() if len(df)]
        obs_max_by_task[task_key] = max(obs_vals)

    # Pass 1: human + models, to establish canonical per-panel y-limits AND
    # produce the "full" stage-2 image.
    fig_full, axes_full = plt.subplots(1, 4, figsize=FIGURE_SIZE, sharey=True,
                                       constrained_layout=True)
    for i, (task_key, title, ref_label) in enumerate(DELTA_TASK_PANELS):
        human_delta, models, _, _ = data[task_key]
        ax = axes_full[i]
        ylabel = "Median |\u0394response|" if i == 0 else ""
        _draw_response_change_panel(ax, human_delta, models, ref_label,
                                    include_models=True, ylabel=ylabel,
                                    obs_max=obs_max_by_task[task_key])
        ax.set_title(title, color=TASK_COLORS[task_key])
        ax.tick_params(axis="y", labelleft=(i == 0))

    axes_full[0].set_ylim(bottom=0)  # shared -- applies to every panel
    shared_top = axes_full[0].get_ylim()[1]

    legend_handles = [Line2D([0], [0], color=HUMAN_COLOR, lw=3, label="Human")]
    legend_handles += [Line2D([0], [0], color=MODEL_COLORS[m], lw=3, label=m)
                       for m in ["Mean", "LeakyIntegrator", "PrimacyRecency"]]
    legend_handles.append(Line2D([0], [0], color=MODEL_COLORS["NEF"], lw=3,
                                 label="NEF / RL_lambda"))
    fig_full.get_layout_engine().set(h_pad=0.25)
    fig_full.legend(handles=legend_handles, loc="outside lower center", ncol=5,
                    frameon=True, framealpha=0.9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    full_path = FIGURES_DIR / "response_change_full.svg"
    fig_full.savefig(full_path)
    plt.close(fig_full)
    print(f"Saved {full_path}")

    # Pass 2: human only, reusing pass 1's shared_top so the two images
    # align exactly when overlaid as an r-stack fragment build.
    fig_human, axes_human = plt.subplots(1, 4, figsize=FIGURE_SIZE, sharey=True,
                                         constrained_layout=True)
    for i, (task_key, title, ref_label) in enumerate(DELTA_TASK_PANELS):
        human_delta, models, _, _ = data[task_key]
        ax = axes_human[i]
        ylabel = "Median |\u0394response|" if i == 0 else ""
        _draw_response_change_panel(ax, human_delta, models, ref_label,
                                    include_models=False, ylabel=ylabel,
                                    obs_max=obs_max_by_task[task_key])
        ax.set_title(title, color=TASK_COLORS[task_key])
        ax.tick_params(axis="y", labelleft=(i == 0))

    axes_human[0].set_ylim(0, shared_top)  # identical to pass 1, not autoscaled

    # Same legend SLOT reserved in both images (so constrained_layout gives
    # both passes the same amount of bottom margin and the axes don't shift
    # position between stages) -- but only "Human" actually shown yet.
    fig_human.get_layout_engine().set(h_pad=0.25)
    fig_human.legend(handles=[Line2D([0], [0], color=HUMAN_COLOR, lw=3, label="Human")],
                     loc="outside lower center", ncol=1,
                     frameon=True, framealpha=0.9)

    human_path = FIGURES_DIR / "response_change_human.svg"
    fig_human.savefig(human_path)
    plt.close(fig_human)
    print(f"Saved {human_path}")

    return [human_path, full_path]



# ── Lambda fitting, individual differences, and recency bias ─────────────

# Copied EXACTLY from figure_soltani_temporal.py's/figure_yoo_temporal.py's
# own _fit_lambda_curve_fit (same functional form, same p0, same bounds --
# see _fit_lambda_series's own docstring for the full comparison).
def _power_law(n, A, lam):
    return A * np.power(np.asarray(n, dtype=float), -lam)


# (task_key, panel title) -- balls/carrabin excluded: neither
# figure_carrabin_temporal.py nor any other working script fits a decay-rate
# lambda for that task at all (no lambda panel exists there), so there is
# nothing to reproduce.
LAMBDA_TASK_PANELS = [
    ("snacks", "Snacks task"),
    ("colors", "Colors task"),
    ("numbers", "Numbers task"),
]

# Minimum-observation threshold and n-offset for the lambda fit itself, PER
# TASK -- copied from each source script's ACTUAL _fit_lambda_curve_fit code
# (not from DELTA_MIN_OBS above, which is a DIFFERENT threshold used only in
# the visual |delta response| panel elsewhere in figure_soltani_temporal.py).
#   snacks (yoo, 1-indexed observation): curve.index >= 2, n = index (no
#     offset) -- figure_yoo_temporal.py's own _fit_lambda_curve_fit.
#   colors/numbers (0-indexed observation): curve.index >= 1, n = index + 1
#     -- figure_soltani_temporal.py's own _fit_lambda_curve_fit, used
#     IDENTICALLY for both tasks in that file.
#
# ONE REAL INCONSISTENCY WAS FOUND (see chat) and is deliberately NOT fixed
# here, per instruction to default to the colors/numbers method as coded:
# colors has its own SEPARATE DELTA_MIN_OBS=2 threshold (used only in the
# visual delta panel) to exclude a documented bimodal/degenerate first delta
# unique to binary evidence, but figure_soltani_temporal.py's shared lambda-
# fit function does NOT apply that exclusion -- it uses >=1 for colors and
# numbers alike. So colors' fitted lambda here may include that same
# degenerate point its own visual panel elsewhere deliberately drops. This
# reproduces the ACTUAL lambda-fit code (>=1), not the stricter threshold.
LAMBDA_MIN_OBS = {"snacks": 2, "colors": 1, "numbers": 1}
LAMBDA_N_OFFSET = {"snacks": 0, "colors": 1, "numbers": 1}

# lambda is bounded to [0,1.5] here (the fit's own bounds= argument allows
# up to 2, but no fitted pid/model exceeds ~1.2 in this data, and capping
# the display range at 1.5 makes better use of the axis) -- shared across
# every histogram/KDE panel below for comparability across tasks and
# between the human-only and human+models figures.
LAMBDA_XLIM = (0.0, 1.5)


def _fit_lambda_series(delta_df: pd.DataFrame, n_offset: int) -> pd.Series:
    """Per-pid decay exponent lambda, fitting A*n^(-lambda) to that pid's own
    mean |delta response| vs n curve by bounded nonlinear least squares --
    IDENTICAL estimator (p0=[0.1, 0.5], bounds=([0,0],[2,2]), maxfev=2000)
    to figure_yoo_temporal.py's/figure_soltani_temporal.py's own
    _fit_lambda_curve_fit. `delta_df` must already be filtered to this
    task's own min_obs (see LAMBDA_MIN_OBS) via _abs_delta_long -- this
    function only adds the n_offset and does the per-pid aggregation/fit.
    Pids whose fit doesn't converge, or with fewer than 3 defined points,
    are omitted (matching the working scripts' own behavior) rather than
    substituted with a degenerate value."""
    out = {}
    for pid, g in delta_df.groupby("pid"):
        curve = g.groupby("observation")["delta"].mean().dropna()
        if len(curve) < 3:
            continue
        n = curve.index.values.astype(float) + n_offset
        y = curve.values.astype(float)
        try:
            popt, _ = curve_fit(_power_law, n, y, p0=[0.1, 0.5],
                                bounds=([0.0, 0.0], [2.0, 2.0]), maxfev=2000)
            out[int(pid)] = float(popt[1])
        except Exception:
            pass
    return pd.Series(out, name="lambda_")


def _load_lambda_delta(task_key: str, path) -> pd.DataFrame:
    """Raw response file (human or model) -> filtered |delta response| long
    frame, ready for _fit_lambda_series. Reuses _abs_delta_long (defined
    above for make_response_change) with this task's LAMBDA-specific
    min_obs -- NOT its DELTA_MIN_OBS -- and first_obs_is_response=False
    (that carrabin-only quirk doesn't apply to any of these three tasks)."""
    df = pd.read_pickle(path)[["pid", "trial", "observation", "response"]]
    return _abs_delta_long(df, LAMBDA_MIN_OBS[task_key], False)


def _plot_lambda_demo(ax, task_key: str = "numbers") -> None:
    """Panel 1: illustrates the fitting PROCEDURE itself -- one representative
    pid's own mean |delta response| vs n curve (points) with its fitted
    power law overlaid (line), rather than summarizing across pids like the
    other three panels. The demo pid is whoever's fitted lambda is closest
    to that task's OWN median -- representative of a typical fit, not
    cherry-picked for a clean-looking curve."""
    human_delta = _load_lambda_delta(task_key, _human_data_path(task_key))
    lam = _fit_lambda_series(human_delta, LAMBDA_N_OFFSET[task_key])
    if lam.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    demo_pid = (lam - lam.median()).abs().idxmin()
    g = human_delta[human_delta["pid"] == demo_pid]
    curve = g.groupby("observation")["delta"].mean().dropna().sort_index()
    n = curve.index.values.astype(float) + LAMBDA_N_OFFSET[task_key]
    y = curve.values.astype(float)
    popt, _ = curve_fit(_power_law, n, y, p0=[0.1, 0.5],
                        bounds=([0.0, 0.0], [2.0, 2.0]), maxfev=2000)
    A_fit, lam_fit = popt

    ax.scatter(n, y, color=HUMAN_COLOR, s=45, zorder=3)
    n_smooth = np.linspace(n.min(), n.max(), 200)
    ax.plot(n_smooth, _power_law(n_smooth, A_fit, lam_fit),
            color=HUMAN_COLOR, lw=2.5, zorder=4)

    ax.text(0.95, 0.95, "$A n^{-\\lambda}$\n$\\lambda=%.2f$" % lam_fit,
            transform=ax.transAxes, ha="right", va="top", fontsize=14)
    ax.set_xlabel("Observations seen (n)")
    ax.set_ylabel("Mean |\u0394response|")
    ax.set_title("Fitting example", color="0.3")
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, top=True, right=True)


def _plot_lambda_distribution(ax, human_lam: pd.Series, task_key: str,
                              model_lams=None) -> None:
    """Panels 2-4: normalized KDE of fitted lambda across pids -- SAME
    convention as figure_soltani_variability.py's own _plot_panel_kde
    (peak-normalized density + individual rug ticks along the baseline),
    reused here rather than a bar histogram so multiple overlaid
    distributions (human + up to 4 models) stay legible. `model_lams`, if
    given, overlays each model's own lambda distribution the same way
    make_model_performance/make_response_change overlay models on human
    data -- omit for the human-only figure. Human is plain gray (HUMAN_COLOR)
    here, matching make_response_change's own human/model color scheme
    exactly -- task color lives on the panel TITLE only (see the two
    make_lambda_* callers), not on the human curve/fill itself."""
    vals = human_lam.dropna()
    if len(vals) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    x = np.linspace(LAMBDA_XLIM[0], LAMBDA_XLIM[1], 400)
    color = HUMAN_COLOR

    def _norm_kde(v):
        kde = gaussian_kde(v, bw_method="scott")
        d = kde(x)
        d = d / d.max()
        d[x < float(v.min())] = 0
        d[x > float(v.max())] = 0
        return kde, d

    kde, density = _norm_kde(vals)
    ax.fill_between(x, density, alpha=0.15, color=color, zorder=1)
    ax.plot(x, density, lw=2.2, color=color, zorder=3)
    kpeak = float(kde(vals.values).max())
    for v in vals.values:
        top = float(kde([v])[0]) / kpeak
        ax.vlines(v, 0, top, color=color, lw=0.6, alpha=0.5, zorder=2)

    for model, mlam in (model_lams or {}).items():
        mv = mlam.dropna()
        if len(mv) < 2:
            continue
        _, mdensity = _norm_kde(mv)
        ax.plot(x, mdensity, lw=2.0, color=MODEL_COLORS[model], zorder=4)

    ax.set_xlabel("Fitted \u03bb")
    ax.set_ylabel("Normalized density")
    ax.set_xlim(*LAMBDA_XLIM)
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, top=True, right=True)


def make_lambda_human() -> Path:
    """1x4 panel: panel 1 demos the power-law fitting PROCEDURE on one
    representative human pid (see _plot_lambda_demo); panels 2-4 are KDEs of
    the fitted decay exponent lambda across pids, one per task (snacks,
    colors, numbers -- balls/carrabin excluded, see LAMBDA_TASK_PANELS).
    Lambda near 0 means little decay (roughly equal weight to every
    observation, primacy-leaning); lambda near 1 is close to an optimal/
    running-mean-like weighting; lambda above 1 over-weights early
    observations even more steeply -- so the SPREAD of a task's histogram
    is a direct picture of individual differences in integration strategy,
    and its rug ticks are literally one mark per real participant.

    Fitting procedure (per _fit_lambda_series/LAMBDA_MIN_OBS/
    LAMBDA_N_OFFSET): the SAME estimator (A*n^(-lambda), bounded nonlinear
    least squares, identical p0/bounds/maxfev) as
    figure_yoo_temporal.py's/figure_soltani_temporal.py's own
    _fit_lambda_curve_fit, with each task's own min_obs/n_offset copied
    from that task's own source script rather than reinvented. One genuine
    inconsistency was found between colors' visual delta panel and its
    lambda fit (see LAMBDA_MIN_OBS's own comment) and is deliberately left
    as-is, matching this project's own actual code rather than the
    stricter threshold used elsewhere.

    Carries a "Human"-only legend in the same reserved slot make_lambda_
    models uses for its full Human+model legend, so the figure's overall
    size/layout doesn't shift between this slide and the follow-up one --
    same reasoning as make_response_change's own human-only stage.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, constrained_layout=True)

    _plot_lambda_demo(axes[0], task_key="numbers")

    for i, (task_key, title) in enumerate(LAMBDA_TASK_PANELS):
        ax = axes[i + 1]
        human_delta = _load_lambda_delta(task_key, _human_data_path(task_key))
        lam = _fit_lambda_series(human_delta, LAMBDA_N_OFFSET[task_key])
        _plot_lambda_distribution(ax, lam, task_key)
        ax.set_title(title, color=TASK_COLORS[task_key])
        ax.set_ylabel("Normalized density" if i == 0 else "")
        ax.tick_params(axis="y", labelleft=(i == 0))

    # Same legend SLOT reserved as make_lambda_models (same h_pad, same
    # "outside lower center" placement) -- so the figure DOESN'T resize or
    # shift when models get added on the follow-up slide; only "Human" is
    # actually shown yet, matching make_response_change's own human-only
    # stage convention exactly.
    fig.get_layout_engine().set(h_pad=0.25)
    fig.legend(handles=[Line2D([0], [0], color=HUMAN_COLOR, lw=3, label="Human")],
               loc="outside lower center", ncol=1,
               frameon=True, framealpha=0.9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "lambda_human.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def make_lambda_models() -> Path:
    """Same 1x4 layout as make_lambda_human (panel 1 unchanged), but panels
    2-4 now overlay each task's 4 fitted models' own lambda distributions
    on top of the human one -- same Mean/LeakyIntegrator/PrimacyRecency +
    NEF-or-RL_lambda roster and MODEL_COLORS as make_model_performance/
    make_response_change, and the same reasoning for why NEF and RL_lambda
    share one color (RL_lambda stands in for NEF on colors/numbers, which
    haven't been fit yet).

    A model's own lambda is fit the SAME way a human pid's is (identical
    _fit_lambda_series call) against that model's own *_responses.pkl
    sequence -- not read from any pre-computed model-comparison file.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, constrained_layout=True)

    _plot_lambda_demo(axes[0], task_key="numbers")

    for i, (task_key, title) in enumerate(LAMBDA_TASK_PANELS):
        ax = axes[i + 1]
        human_delta = _load_lambda_delta(task_key, _human_data_path(task_key))
        lam = _fit_lambda_series(human_delta, LAMBDA_N_OFFSET[task_key])

        ref_label = "RL_lambda" if task_key in ("colors", "numbers") else "NEF"
        model_lams = {}
        for model in ["Mean", "LeakyIntegrator", "PrimacyRecency", ref_label]:
            path = _delta_responses_path(task_key, model)
            if not path.exists():
                print(f"  (missing {path.name} -- skipping {model} for {task_key})")
                continue
            model_delta = _load_lambda_delta(task_key, path)
            model_lams[model] = _fit_lambda_series(model_delta, LAMBDA_N_OFFSET[task_key])

        _plot_lambda_distribution(ax, lam, task_key, model_lams=model_lams)
        ax.set_title(title, color=TASK_COLORS[task_key])
        ax.set_ylabel("Normalized density" if i == 0 else "")
        ax.tick_params(axis="y", labelleft=(i == 0))

    legend_handles = [Line2D([0], [0], color=HUMAN_COLOR, lw=3, label="Human")]
    legend_handles += [Line2D([0], [0], color=MODEL_COLORS[m], lw=3, label=m)
                       for m in ["Mean", "LeakyIntegrator", "PrimacyRecency"]]
    legend_handles.append(Line2D([0], [0], color=MODEL_COLORS["NEF"], lw=3,
                                 label="NEF / RL_lambda"))
    fig.get_layout_engine().set(h_pad=0.25)
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=5,
               frameon=True, framealpha=0.9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "lambda_models.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path



# ── Sanity check: split-half reliability of lambda + cross-task comparison ──

# Abbreviated legend labels for the sanity-check panels below, which pack up
# to 5 sources into one legend per panel -- "Mean"/"NEF" stay full-length
# (already short), LeakyIntegrator/PrimacyRecency/RL_lambda get shortened so
# a 5-entry legend stays compact. "Human" is never in this dict (looked up
# via .get(label, label), so it passes through unchanged).
MODEL_DISPLAY = {
    "Mean": "Mean",
    "LeakyIntegrator": "LI",
    "PrimacyRecency": "PR",
    "NEF": "NEF",
    "RL_lambda": "RL_l",
    "NoisyRL_lambda": "Noisy RL",
}


def _fit_lambda_split_half(task_key: str, path: Path) -> pd.DataFrame:
    """Per-pid lambda fit separately on ODD vs EVEN trial-index halves --
    the SAME convention figure_soltani_temporal.py's own _fit_lambda_split_half
    uses for its panels E/K (colors/numbers), NOT figure_yoo_temporal.py's own
    _fit_lambda_split_half for ITS OWN panel C, which splits trials
    CHRONOLOGICALLY (first half vs second half) instead.

    THIS IS A REAL METHODOLOGY DIFFERENCE BETWEEN THE TWO SOURCE SCRIPTS (see
    chat) -- soltani's own docstring explains why odd/even is preferred:
    interleaving samples both "halves" from the same span of session-time,
    isolating genuine estimation noise from any systematic drift (learning,
    fatigue, boredom) that a strict chronological split would confound with
    unreliability. Per instruction to match panels E/K specifically, this
    applies the ODD/EVEN split uniformly to all three tasks here, including
    snacks -- overriding yoo's own chronological convention for this figure
    rather than importing it.

    Requires >=3 trials in EACH half to attempt a fit (matching both source
    scripts' own threshold). Returns columns [pid, odd, even] -- renamed from
    soltani's own [pid, first, second] for clarity, since the split itself is
    odd/even, not first/second."""
    raw = pd.read_pickle(path)[["pid", "trial", "observation", "response"]]
    rows = []
    for pid, grp in raw.groupby("pid"):
        trials = sorted(grp["trial"].unique())
        odd_trials, even_trials = trials[0::2], trials[1::2]
        if min(len(odd_trials), len(even_trials)) < 3:
            continue
        for half_label, trial_set in [("odd", odd_trials), ("even", even_trials)]:
            sub = grp[grp["trial"].isin(trial_set)]
            delta_sub = _abs_delta_long(sub, LAMBDA_MIN_OBS[task_key], False)
            lam = _fit_lambda_series(delta_sub, LAMBDA_N_OFFSET[task_key])
            if pid in lam.index:
                rows.append({"pid": pid, "half": half_label, "lambda_": float(lam[pid])})
    if not rows:
        return pd.DataFrame(columns=["pid", "odd", "even"])
    wide = pd.DataFrame(rows).pivot(index="pid", columns="half", values="lambda_").dropna()
    wide.columns.name = None
    return wide.reset_index()


def _plot_lambda_splithalf_panel(ax, legend_ax, task_key: str, title: str,
                                 include_models: bool, show_ylabel: bool) -> None:
    """Panels 1-3: odd-vs-even split-half reliability of fitted lambda, one
    regplot per source -- matching figure_soltani_temporal.py's own panels
    E/K exactly (scatter=True for EVERY source, not just Human; that panel is
    meaningful even for deterministic models, since the split's two curves
    differ due to different STIMULUS sequences across odd/even trials, no
    response noise required -- see that panel's own docstring). Human is
    plain gray; task color lives on the panel title only, same convention as
    every other lambda/response-change figure in this deck.

    Panels 1-3 share BOTH axes at a fixed [0, 1.5] range (matches LAMBDA_XLIM)
    -- not autoscaled per task -- so the three tasks' reliability scatter is
    directly visually comparable rather than each panel silently rescaling to
    its own data range. show_ylabel controls whether this panel draws its own
    y-axis label/ticks (only the leftmost of the three needs to, since the
    scale is now identical across all three).

    LEGEND GOES IN A SEPARATE, DEDICATED `legend_ax` (a second GridSpec row
    turned off via axis('off'), passed in by the caller), not a bbox_to_anchor
    placement on `ax` itself. This replaced an earlier bbox_to_anchor approach
    (`ax.legend(loc='upper center', bbox_to_anchor=(0.5, <negative>))`) that
    worked for a 1-row legend but became unreliable once FIGURE_SIZE was
    fixed (see that constant's own comment) and a 5-row Human+4-models
    legend needed more room than the fixed canvas could grow to provide --
    constrained_layout shrinks an Axes to make room for a bbox_to_anchor
    legend placed outside it, but only up to a point, and past that point
    the legend silently clipped rather than erroring (confirmed directly by
    rendering and inspecting the actual file, not assumed). A dedicated
    legend_ax with an explicit height_ratios allocation is deterministic
    instead: its size is fixed by the GridSpec, not inferred from content."""
    ref_label = "RL_lambda" if task_key in ("colors", "numbers") else "NEF"
    sources = [("Human", _human_data_path(task_key), HUMAN_COLOR)]
    if include_models:
        for model in ["Mean", "LeakyIntegrator", "PrimacyRecency", ref_label]:
            path = _delta_responses_path(task_key, model)
            if path.exists():
                sources.append((model, path, MODEL_COLORS[model]))
            else:
                print(f"  (missing {path.name} -- skipping {model} for {task_key})")

    handles, labels = [], []
    for label, path, color in sources:
        wide = _fit_lambda_split_half(task_key, path)
        if len(wide) < 2:
            continue
        sns.regplot(data=wide, x="odd", y="even", ax=ax, color=color,
                    ci=95 if len(wide) >= 3 else None, scatter=True,
                    line_kws={"lw": 1.5}, scatter_kws={"s": 20, "alpha": 0.6})
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        disp = MODEL_DISPLAY.get(label, label)
        if len(wide) >= 3:
            r, p = pearsonr(wide["odd"], wide["even"])
            labels.append(f"{disp} r={r:.2f}{pvalue_to_stars(p)}")
        else:
            labels.append(f"{disp} n={len(wide)}")

    legend_ax.axis("off")
    ax.set_title(title, color=TASK_COLORS[task_key])
    ax.set_xlim(*LAMBDA_XLIM)
    ax.set_ylim(*LAMBDA_XLIM)
    if not handles:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    legend_ax.legend(handles=handles, labels=labels, fontsize=9, loc="center",
                     ncol=1, labelspacing=0.4, frameon=True, framealpha=0.9)
    ax.set_xlabel("\u03bb (odd trials)")
    ax.set_ylabel("\u03bb (even trials)" if show_ylabel else "")
    ax.tick_params(axis="y", labelleft=show_ylabel)
    sns.despine(ax=ax, top=True, right=True)


def _plot_lambda_crosstask_panel(ax, legend_ax, include_models: bool = False) -> None:
    """Panel 4: cross-task comparison of fitted lambda, colors vs numbers, one
    point per pid who did BOTH -- matching figure_soltani_temporal.py's own
    panel L (_plot_panel_lambda_crosstask). That panel is human-only BY
    DESIGN in the source script (an individual-differences/trait-stability
    check, not a model-fit panel) -- include_models=True here is a
    deliberate departure from that convention, per explicit instruction, not
    an oversight: each model's own colors-lambda and numbers-lambda are
    fit from that model's own *_responses.pkl sequences (same _fit_lambda_
    series call as every other lambda panel), showing whether a model's
    OWN cross-task consistency looks anything like a human's. Legend goes in
    a dedicated legend_ax, same reasoning as _plot_lambda_splithalf_panel's
    own docstring."""
    ref_label = "RL_lambda"  # both colors and numbers use RL_lambda, never NEF
    sources = [("Human", _human_data_path("colors"), _human_data_path("numbers"), HUMAN_COLOR)]
    if include_models:
        for model in ["Mean", "LeakyIntegrator", "PrimacyRecency", ref_label]:
            c_path = _delta_responses_path("colors", model)
            n_path = _delta_responses_path("numbers", model)
            if c_path.exists() and n_path.exists():
                sources.append((model, c_path, n_path, MODEL_COLORS[model]))
            else:
                print(f"  (missing responses -- skipping {model} for crosstask)")

    legend_ax.axis("off")
    ax.set_title("Colors vs Numbers", color="0.3", fontsize=14)
    handles, labels = [], []
    for label, c_path, n_path, color in sources:
        lam_colors = _fit_lambda_series(
            _load_lambda_delta("colors", c_path), LAMBDA_N_OFFSET["colors"])
        lam_numbers = _fit_lambda_series(
            _load_lambda_delta("numbers", n_path), LAMBDA_N_OFFSET["numbers"])
        merged = pd.DataFrame({"colors": lam_colors, "numbers": lam_numbers}).dropna()
        if len(merged) < 2:
            continue

        ax.scatter(merged["colors"], merged["numbers"], color=color, s=30,
                  alpha=0.7, zorder=3)
        disp = MODEL_DISPLAY.get(label, label)
        if len(merged) >= 3:
            sns.regplot(data=merged, x="colors", y="numbers", ax=ax, color=color,
                       ci=95, scatter=False, line_kws={"lw": 1.5})
            r, p = pearsonr(merged["colors"], merged["numbers"])
            handles.append(Line2D([0], [0], color=color, lw=1.5))
            labels.append(f"{disp} r={r:.2f}{pvalue_to_stars(p)}")
        else:
            handles.append(Line2D([0], [0], color=color, lw=1.5))
            labels.append(f"{disp} n={len(merged)}")

    if not handles:
        ax.text(0.5, 0.5, "No pids completed both tasks", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    legend_ax.legend(handles=handles, labels=labels, fontsize=9, loc="center",
                     ncol=1, labelspacing=0.4, frameon=True, framealpha=0.9)
    ax.set_xlabel("\u03bb (colors)")
    ax.set_ylabel("\u03bb (numbers)")
    sns.despine(ax=ax, top=True, right=True)


def make_lambda_sanity_human() -> Path:
    """1x4 panel: panels 1-3 are odd/even split-half reliability of fitted
    lambda for snacks/colors/numbers (matching figure_soltani_temporal.py's
    own panels E/K -- see _fit_lambda_split_half's own docstring for the one
    real methodology difference found and resolved), sharing a fixed [0,1.5]
    x/y range (LAMBDA_XLIM) across all three rather than each autoscaling to
    its own data. Panel 4 is the colors-vs-numbers cross-task comparison
    (matching that same script's panel L), human only in THIS figure -- see
    make_lambda_sanity_models for the model-added version of every panel,
    including panel 4.

    Uses a 2-ROW GridSpec (plots on top, a dedicated legend row underneath)
    rather than one row of Axes with bbox_to_anchor legends -- see
    _plot_lambda_splithalf_panel's own docstring for why that approach
    became unreliable once a 5-source legend needed to fit in this figure's
    now-fixed canvas size.
    """
    _apply_slide_style()
    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[3.2, 1.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    legend_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

    for i, (ax, lax, (task_key, title)) in enumerate(zip(axes[:3], legend_axes[:3], LAMBDA_TASK_PANELS)):
        _plot_lambda_splithalf_panel(ax, lax, task_key, title, include_models=False,
                                     show_ylabel=(i == 0))
    _plot_lambda_crosstask_panel(axes[3], legend_axes[3], include_models=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "lambda_sanity_human.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def make_lambda_sanity_models() -> Path:
    """Same layout as make_lambda_sanity_human, but panels 1-3 now add each
    task's fitted models' own lambda alongside Human's -- ALL 4 models
    (deterministic ones too, matching figure_soltani_temporal.py's own E/K,
    unlike figure_yoo_temporal.py's own panel C which excludes Mean/
    LeakyIntegrator for reasons not inherited here). Panel 4 (cross-task) is
    human-only, matching panel L's own design -- models were tried there
    too but explicitly removed per instruction; UNCHANGED from
    make_lambda_sanity_human. Same 2-row GridSpec as make_lambda_sanity_
    human, so the two figures' plot areas stay the same size regardless of
    legend row count.
    """
    _apply_slide_style()
    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[3.2, 1.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    legend_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

    for i, (ax, lax, (task_key, title)) in enumerate(zip(axes[:3], legend_axes[:3], LAMBDA_TASK_PANELS)):
        _plot_lambda_splithalf_panel(ax, lax, task_key, title, include_models=True,
                                     show_ylabel=(i == 0))
    _plot_lambda_crosstask_panel(axes[3], legend_axes[3], include_models=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "lambda_sanity_models.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path



# ── Response variability for identical inputs (balls, colors, numbers) ─────

# Matches figure_carrabin_variability.py's own panel A metric, but with
# user-facing text (x-axis label, slide titles) renamed "Response Noise" --
# shorter and reads more naturally as a slide title than "variability".
VARIABILITY_NOISE_LABEL = "Response Noise"

# (task_key, panel title) -- snacks/yoo excluded: figure_yoo_temporal.py has
# no variance-growth/within-qid-repeat panel at all for that task (no
# established methodology to reproduce), unlike carrabin (this metric's own
# source) and colors/numbers (via figure_soltani_temporal.py's cols 3-4
# machinery, repurposed here -- see below).
VARIABILITY_TASK_PANELS = [
    ("balls", "Balls task"),
    ("colors", "Colors task"),
    ("numbers", "Numbers task"),
]

# The ONE model per task with a genuine noise term -- everything else is
# exactly deterministic (verified directly: max std 2.3e-16 for Mean/
# LeakyIntegrator/PrimacyRecency on carrabin, i.e. floating-point zero, not
# just "small"). NEF for balls (carrabin's own model roster); RL_lambda is
# deterministic for colors/numbers (see utils/soltani_models.py's own
# STOCHASTIC_MODELS), so NoisyRL_lambda stands in there instead -- checked
# directly that its noise did NOT collapse to the zero floor that module's
# own docstring warns can happen under RMSE fitting (median std 0.33-0.39,
# not 0), so it is a genuine, non-degenerate stand-in, not a coincidental
# flat line.
VARIABILITY_STOCHASTIC_MODEL = {
    "balls": "NEF",
    "colors": "NoisyRL_lambda",
    "numbers": "NoisyRL_lambda",
}

# Every OTHER model for that task -- all exactly deterministic, plotted as
# jittered points at x=0 rather than a degenerate zero-width "KDE" (see
# _plot_variability_panel's own docstring for why).
VARIABILITY_DETERMINISTIC_MODELS = {
    "balls": ["Mean", "LeakyIntegrator", "PrimacyRecency"],
    "colors": ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda"],
    "numbers": ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda"],
}

# colors has no designed qid repeat structure of its own -- utils.colors_
# quasi_qids empirically derives one (see that module's own docstring for
# the sweep that settled these defaults). numbers uses its REAL designed
# prefix (4 -- see figure_soltani_temporal.py's own NUMBERS_PREFIX_LENGTH).
# Both restrict to the prefix window, where a qid's repeats actually share
# an identical stimulus; carrabin needs no such restriction; its qid IS the
# whole 5-observation trial, repeated verbatim.
VARIABILITY_PREFIX_LENGTH = {"colors": None, "numbers": 4}  # colors' own set below


def _qid_response_std(resp_df: pd.DataFrame, qid_map: pd.DataFrame,
                      min_trials: int = 3) -> pd.Series:
    """Mean of std(response | pid, observation, qid) per pid -- copied
    VERBATIM from figure_carrabin_variability.py's own _qid_response_std.
    One number per pid: how noisy is this person's (or model's) response to
    a REPEATED, identical stimulus. resp_df needs no `qid` column of its
    own -- it is merged in from qid_map on (pid, trial, observation), so the
    same repeat-structure labelling can be applied uniformly to Human and
    every model's response file."""
    df = resp_df.drop(columns=["qid"], errors="ignore").merge(
        qid_map, on=["pid", "trial", "observation"])
    grp = (df.groupby(["pid", "observation", "qid"])["response"]
           .apply(lambda x: x.std() if len(x) >= min_trials else np.nan)
           .dropna().reset_index(name="resp_std"))
    return grp.groupby("pid")["resp_std"].mean()


def _variability_qid_map(task_key: str) -> tuple[pd.DataFrame, int | None]:
    """(qid_map, prefix_length) for one task -- qid_map has columns [pid,
    trial, observation, qid], already restricted to the prefix window for
    colors/numbers (see VARIABILITY_PREFIX_LENGTH's own comment). Built
    once per task and reused for Human and every model, so the exact same
    repeat-structure labelling applies to all of them."""
    if task_key == "balls":
        human = pd.read_pickle(data_path("carrabin.pkl"))
        qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()
        return qid_map, None

    from utils.colors_quasi_qids import (
        MIN_REPEATS as QQ_MIN_REPEATS,
        PREFIX_LENGTH as QQ_PREFIX_LENGTH,
        add_quasi_qids,
    )

    if task_key == "colors":
        human = pd.read_pickle(data_path("soltani_colors.pkl"))
        human = add_quasi_qids(human, prefix_length=QQ_PREFIX_LENGTH,
                               min_repeats=QQ_MIN_REPEATS)
        prefix = QQ_PREFIX_LENGTH
    else:
        human = pd.read_pickle(data_path("soltani_numbers.pkl"))
        prefix = VARIABILITY_PREFIX_LENGTH["numbers"]

    qid_map = (human[human["observation"] < prefix]
               [["pid", "trial", "observation", "qid"]].drop_duplicates())
    return qid_map, prefix


def _variability_series(task_key: str, path: Path, qid_map: pd.DataFrame,
                        prefix: int | None) -> pd.Series:
    """Per-pid response-variability series for one (task, source) -- Human
    or one model's *_responses.pkl. `path` is the raw response file;
    restricted to the prefix window first for colors/numbers, matching
    qid_map's own restriction, before computing _qid_response_std."""
    df = pd.read_pickle(path)[["pid", "trial", "observation", "response"]]
    if prefix is not None:
        df = df[df["observation"] < prefix]
    return _qid_response_std(df, qid_map)


def _variability_model_path(task_key: str, model: str) -> Path:
    """Path to one (task, model)'s *_responses.pkl. balls' NEF specifically
    uses the MLE-fitted variant (NEF_carrabin_responses_mle.pkl), matching
    figure_carrabin_variability.py's OWN panels A/C convention -- NOT the
    RMSE-fitted NEF_carrabin_responses.pkl _delta_responses_path uses for
    ITS (different) panel B. Every other (task, model) is the plain
    RMSE-fitted response file."""
    if task_key == "balls":
        if model == "NEF":
            return RUNS_DIR / "carrabin" / "NEF_carrabin_responses_mle.pkl"
        return RUNS_DIR / "carrabin" / f"{model}_carrabin_responses.pkl"
    dataset = "soltani_colors" if task_key == "colors" else "soltani_numbers"
    return RUNS_DIR / "soltani" / f"{model}_{dataset}_responses.pkl"


def _plot_variability_panel(ax, task_key: str, title: str,
                            include_models: bool, show_ylabel: bool) -> None:
    """Panels B-D: normalized KDE of per-pid response variability for
    identical (repeated-stimulus) inputs -- matching
    figure_carrabin_variability.py's own panel A (KDE + peak-normalized +
    individual-pid rug ticks, same convention this deck's other lambda/
    KDE panels already use), extended to colors/numbers via
    figure_soltani_temporal.py's own qid/quasi-qid repeat-structure
    machinery (see _variability_qid_map). Human is plain gray; task color
    lives on the panel title only, same convention as every other figure
    in this deck.

    X-AXIS IS PER-PANEL (autoscaled to that task's own data, with headroom)
    -- NOT shared across B-D. A shared range was tried first, but left a
    lot of empty space in the narrower-scale tasks (balls/numbers) once the
    range had to accommodate colors' wider spread; per-panel autoscaling
    trades cross-task comparability for using each panel's own space fully.
    show_ylabel controls whether this panel draws its own y-axis label/
    ticks (only the leftmost of the three needs to, since the density is
    already peak-normalized to 1 everywhere).

    When include_models=True, only ONE model per task gets drawn as a
    proper KDE line -- whichever one actually has a nonzero noise term
    (VARIABILITY_STOCHASTIC_MODEL). Every OTHER model is deterministic, so
    its own per-pid variability is EXACTLY zero (floating-point zero, not
    "small" -- verified directly, see VARIABILITY_TASK_PANELS's own
    comment): plotting those as a KDE would be a degenerate zero-width
    spike, or as a flat line at x=0 would look like an artifact/error
    rather than a genuine result. Instead each deterministic model is drawn
    as a small cluster of (x-jittered, y-scattered) points anchored at
    x=0 -- one marker per pid, honestly showing "this model's variability
    really is zero for every participant" rather than hiding that fact or
    faking a distribution that doesn't exist.

    No dedicated legend row here (see make_variability_human's own note on
    why the reserved legend row was removed) -- with >1 source, a legend is
    drawn INSIDE the axes itself instead.
    """
    qid_map, prefix = _variability_qid_map(task_key)
    human_path = _human_data_path(task_key)
    human_vals = _variability_series(task_key, human_path, qid_map, prefix).dropna()

    ax.set_title(title, color=TASK_COLORS[task_key])

    if len(human_vals) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    all_vals = list(human_vals.values)
    stochastic_model = VARIABILITY_STOCHASTIC_MODEL[task_key]
    stochastic_vals = None
    if include_models:
        spath = _variability_model_path(task_key, stochastic_model)
        if spath.exists():
            stochastic_vals = _variability_series(task_key, spath, qid_map, prefix).dropna()
            if len(stochastic_vals) >= 2:
                all_vals.extend(stochastic_vals.values)
        else:
            print(f"  (missing {spath.name} -- skipping {stochastic_model} for {task_key})")

    x_max = float(np.quantile(all_vals, 0.99)) * 1.15
    x = np.linspace(0, x_max, 400)

    def _norm_kde(vals):
        kde = gaussian_kde(vals, bw_method="scott")
        d = kde(x)
        d = d / d.max()
        d[x < float(vals.min())] = 0
        d[x > float(vals.max())] = 0
        return kde, d

    handles, labels = [], []

    kde, density = _norm_kde(human_vals.values)
    ax.fill_between(x, density, alpha=0.15, color=HUMAN_COLOR, zorder=1)
    ax.plot(x, density, lw=2.2, color=HUMAN_COLOR, zorder=3)
    kpeak = float(kde(human_vals.values).max())
    for v in human_vals.values:
        top = float(kde([v])[0]) / kpeak
        ax.vlines(v, 0, top, color=HUMAN_COLOR, lw=0.6, alpha=0.5, zorder=2)
    handles.append(Line2D([0], [0], color=HUMAN_COLOR, lw=2.2))
    labels.append("Human")

    if include_models:
        if stochastic_vals is not None and len(stochastic_vals) >= 2:
            _, mdensity = _norm_kde(stochastic_vals.values)
            color = MODEL_COLORS[stochastic_model]
            ax.plot(x, mdensity, lw=2.0, color=color, zorder=4)
            handles.append(Line2D([0], [0], color=color, lw=2.0))
            labels.append(MODEL_DISPLAY.get(stochastic_model, stochastic_model))

        rng = np.random.RandomState(0)  # deterministic jitter, not a new draw per render
        for i, model in enumerate(VARIABILITY_DETERMINISTIC_MODELS[task_key]):
            mpath = _variability_model_path(task_key, model)
            if not mpath.exists():
                print(f"  (missing {mpath.name} -- skipping {model} for {task_key})")
                continue
            mvals = _variability_series(task_key, mpath, qid_map, prefix).dropna()
            if len(mvals) < 1:
                continue
            n = len(mvals)
            x_jit = rng.normal(0, x_max * 0.01, n)
            y_jit = rng.uniform(0.03, 0.16, n)
            color = MODEL_COLORS[model]
            ax.scatter(x_jit, y_jit, color=color, s=14, alpha=0.6, zorder=5,
                      edgecolors="none")
            handles.append(Line2D([0], [0], color=color, lw=0, marker="o",
                                  markersize=5, alpha=0.8))
            labels.append(MODEL_DISPLAY.get(model, model))

    ax.set_xlabel(VARIABILITY_NOISE_LABEL, fontsize=14)
    ax.set_ylabel("Normalized density" if show_ylabel else "")
    ax.tick_params(axis="y", labelleft=show_ylabel)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, top=True, right=True)
    # Skip the legend entirely when there's only one source (just "Human")
    # -- a single-entry legend saying "Human" is redundant clutter once
    # there's nothing else to distinguish it from; the panel title already
    # names the task. Kept ready for when models get re-added: with >1
    # source, drawn INSIDE the axes (no dedicated row anymore).
    if len(handles) > 1:
        ax.legend(handles=handles, labels=labels, fontsize=9, loc="upper right",
                  frameon=True, framealpha=0.9)


def _embed_svg_into_rect(main_path: Path, insert_path: Path,
                         rect: tuple[float, float, float, float],
                         id_prefix: str) -> None:
    """Merge `insert_path`'s SVG content into `main_path`'s own rect (x, y,
    w, h, in main_path's own viewBox units), overwriting main_path in place
    with ONE genuinely self-contained SVG -- NOT a nested <image
    href="sibling.svg">, which silently fails to load further external
    resources once a composite like this is itself embedded via <img> in
    the deck (the same browser-sandboxing quirk discovered building
    presentations/images/behavioral_tasks.svg earlier this project). IDs are
    namespaced with id_prefix to avoid collisions with main_path's own
    (matplotlib-generated) IDs -- same fix as that composite."""
    import re

    def extract_svg_body(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        start = text.index("<svg")
        end = text.rindex("</svg>") + len("</svg>")
        body = text[start:end]
        return re.sub(r'(width|height)="([0-9.]+)mm"', r'\1="\2"', body, count=2)

    def namespace_ids(svg_text: str, prefix: str) -> str:
        ids = set(re.findall(r'\bid="([^"]+)"', svg_text))
        svg_text = re.sub(r'\bid="([^"]+)"',
                         lambda m: f'id="{prefix}{m.group(1)}"', svg_text)
        for old_id in ids:
            esc = re.escape(old_id)
            svg_text = re.sub(rf'((?:xlink:)?href="#){esc}(")',
                             rf'\g<1>{prefix}{old_id}\g<2>', svg_text)
            svg_text = re.sub(rf'(url\(#){esc}(\))',
                             rf'\g<1>{prefix}{old_id}\g<2>', svg_text)
        return svg_text

    main_body = extract_svg_body(main_path)
    x, y, w, h = rect
    insert_body = namespace_ids(extract_svg_body(insert_path), id_prefix)

    # Replace the insert's OWN width/height (and any x/y) on its root <svg>
    # tag rather than appending new ones alongside them -- appending caused
    # an invalid "attribute redefined" SVG (confirmed by rendering: Chromium
    # refused to parse past that point). Only the FIRST <svg ...> opening
    # tag is touched (re.sub count=1), matching how extract_svg_body already
    # isolates just the root element.
    svg_open_end = insert_body.index(">", insert_body.index("<svg"))
    svg_open_tag = insert_body[:svg_open_end]
    svg_open_tag = re.sub(r'\s(width|height|x|y)="[^"]*"', "", svg_open_tag)
    svg_open_tag += f' x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet"'
    insert_body = svg_open_tag + insert_body[svg_open_end:]

    insert_at = main_body.rindex("</svg>")
    merged = main_body[:insert_at] + insert_body + main_body[insert_at:]
    main_path.write_text(merged, encoding="utf-8")


def _panel_a_rect(fig, ax) -> tuple[float, float, float, float]:
    """axes[0]'s own plotting rectangle, in the SAME point-space as the
    saved SVG's viewBox (FIGURE_SIZE inches * 72 pt/inch -- matplotlib's
    SVG backend default, confirmed against an actual saved file: figsize
    (10.6, 5.45) -> viewBox "0 0 763.2 392.4", i.e. exactly *72). Forces a
    draw first, since constrained_layout only finalizes axes positions at
    draw/save time, not immediately after subplot creation. matplotlib's
    Bbox is bottom-up (y0=bottom); SVG is top-down, so y is flipped."""
    fig.canvas.draw()
    bbox = ax.get_position()
    w_pt = FIGURE_SIZE[0] * 72
    h_pt = FIGURE_SIZE[1] * 72
    x = bbox.x0 * w_pt
    y = (1 - bbox.y1) * h_pt
    w = (bbox.x1 - bbox.x0) * w_pt
    h = (bbox.y1 - bbox.y0) * h_pt
    return (x, y, w, h)


# Hand-made schematic explaining the response-variability metric itself
# (repeated identical trials -> spread of responses = "response noise") --
# goes in panel A of both variability figures, the same conceptual role
# _plot_lambda_demo's "Fitting example" panel plays for the lambda figures.
VARIABILITY_SCHEMATIC = Path(__file__).resolve().parent / "images" / "response_noise_schematic.svg"


def make_variability_human() -> Path:
    """1x4 panel: panel A holds VARIABILITY_SCHEMATIC (a hand-made diagram of
    the metric itself -- repeated identical trials, response spread =
    "response noise"), composited in via _embed_svg_into_rect rather than
    plotted; panels B-D are Human-only KDEs of response variability for
    identical inputs, one per task (balls, colors, numbers -- snacks
    excluded, see VARIABILITY_TASK_PANELS), each autoscaled to its OWN data
    range (see _plot_variability_panel's own docstring for why the shared
    range this used to have was dropped).

    PLAIN 1x4 plt.subplots -- NOT the 2-row GridSpec (plots + a dedicated
    legend row) the lambda sanity-check figures use. That reserved legend
    row left a lot of empty space at the bottom once the legend itself
    stopped being drawn (single-source panels skip it -- see
    _plot_variability_panel), so it was removed; a legend, if one ever
    becomes necessary again, is drawn INSIDE each axes instead (no
    reserved row).
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, constrained_layout=True)

    axes[0].axis("off")
    axes[0].set_title("Metric Definition", color="0.3")

    for i, (ax, (task_key, title)) in enumerate(zip(axes[1:], VARIABILITY_TASK_PANELS)):
        _plot_variability_panel(ax, task_key, title, include_models=False,
                                show_ylabel=(i == 0))

    panel_a_rect = _panel_a_rect(fig, axes[0])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "variability_human.svg"
    fig.savefig(out_path)
    plt.close(fig)

    if VARIABILITY_SCHEMATIC.exists():
        _embed_svg_into_rect(out_path, VARIABILITY_SCHEMATIC, panel_a_rect, "panelA_")
    else:
        print(f"  (missing {VARIABILITY_SCHEMATIC.name} -- panel A left blank)")

    print(f"Saved {out_path}")
    return out_path


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

    for i, (ax, (task_key, title)) in enumerate(zip(axes[1:], VARIABILITY_TASK_PANELS)):
        _plot_variability_panel(ax, task_key, title, include_models=False,
                                show_ylabel=(i == 0))

    panel_a_rect = _panel_a_rect(fig, axes[0])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "variability_models.svg"
    fig.savefig(out_path)
    plt.close(fig)

    if VARIABILITY_SCHEMATIC.exists():
        _embed_svg_into_rect(out_path, VARIABILITY_SCHEMATIC, panel_a_rect, "panelA_")
    else:
        print(f"  (missing {VARIABILITY_SCHEMATIC.name} -- panel A left blank)")

    print(f"Saved {out_path}")
    return out_path



# -- Model performance under the new NLL/quasi-MLE metric (all four tasks) --

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
# THE METRIC ITSELF (read from fitting/losses.py's own compute_nll/
# nll_from_ensemble, models/math_models.py's simulate_ensemble): Gaussian
# NLL of each observed human response under the model's own SIMULATED
# predictive distribution -- mean and SD taken across an ensemble of
# n_sims=100 independent stochastic simulations at that pid's fitted
# parameters, per (pid, trial, observation) row. This is a proper scoring
# rule: a wrong conditional mean is punished by the quadratic term, an
# UNDERSTATED sigma is punished by the same quadratic term blowing up as
# sigma shrinks, and an OVERSTATED sigma is punished by log(sigma) -- unlike
# RMSE, which cannot see variance at all. Lower (more negative -- NLL can be
# negative, unlike RMSE) is better, same "lower = better" direction RMSE
# used, so _compute_sig_lines below needs no change.
#
# WHY EVERY MODEL NOW HAS AN "_resp_noise" FILE VARIANT: NLL is undefined
# for a deterministic model (its ensemble SD is exactly 0, an infinite
# NLL) -- simulate_ensemble refuses to silently paper over this the way an
# ad hoc sigma floor would. So Mean/LeakyIntegrator/PrimacyRecency/RL_lambda
# were each given one ADDED free parameter, sigma_resp (i.i.d. Gaussian
# response noise on top of that model's own deterministic prediction),
# making every model's NLL well-defined and put on the SAME scale for the
# first time. NoisyRL_lambda already had its own genuine noise mechanism
# (sigma_state), so it keeps its own name -- no "_resp_noise" suffix, and no
# separate sigma_resp column in its own fit (verified directly: its saved
# nll_params.pkl has only alpha_0/lambda_/sigma_state, three parameters).
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
# make_variability_models).
NLL_MODEL_COLORS = {m: MODEL_COLORS[m] for m in NLL_MODEL_ORDER}
NLL_MODEL_COLORS["NoisyRL_lambda"] = MODEL_COLORS["RL_lambda"]

NLL_TASK_PANELS = [
    ("balls", "Balls task"),
    ("snacks", "Snacks task"),
    ("colors", "Colors task"),
    ("numbers", "Numbers task"),
]


def _nll_perf_path(task_key: str, model: str) -> Path:
    """Path to one (task, model)'s *_nll_performance.pkl. Every model except
    NoisyRL_lambda is fit as its own name PLUS "_resp_noise" on disk (see
    this section's own module-level comment for why) -- that suffix is
    purely a file-naming/fitting-pipeline detail, so it's added here rather
    than exposed to any caller; every other function in this file refers to
    these models by their plain names (Mean, LeakyIntegrator, ...)."""
    file_model = model if model == "NoisyRL_lambda" else f"{model}_resp_noise"
    if task_key == "balls":
        return RUNS_DIR / "carrabin" / f"{file_model}_carrabin_nll_performance.pkl"
    if task_key == "snacks":
        return RUNS_DIR / "yoo" / f"{file_model}_yoo_nll_performance.pkl"
    dataset = "soltani_colors" if task_key == "colors" else "soltani_numbers"
    return RUNS_DIR / "soltani" / f"{file_model}_{dataset}_nll_performance.pkl"


def make_model_performance_nll() -> Path:
    """1x4 panel: model fit under the NEW NLL/quasi-MLE metric, one panel
    per task (balls, snacks, colors, numbers) -- same layout, significance-
    bar logic, and shared-legend convention as make_model_performance, but
    a genuinely different loss (see this section's own module-level
    comment for the full metric description, the "_resp_noise" file-naming
    quirk, why NoisyRL_lambda is the uniform reference model here, and the
    45-vs-35-pid data-vintage caveat for colors/numbers).

    Y-AXIS IS SHARED (sharey=True) but NOT forced to start at 0 -- unlike
    make_model_performance's RMSE (which is non-negative by construction),
    NLL can be, and often is, negative (confirmed directly: loss ranges as
    low as -3.03 for numbers/NoisyRL_lambda), so clamping the bottom would
    misrepresent the actual data.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, sharey=True,
                             constrained_layout=True)

    panel_data = []
    for i, (ax, (task_key, title)) in enumerate(zip(axes, NLL_TASK_PANELS)):
        rows = []
        for model in NLL_MODEL_ORDER:
            path = _nll_perf_path(task_key, model)
            if not path.exists():
                print(f"  (missing {path.name} -- skipping {model} for {task_key})")
                continue
            perf = pd.read_pickle(path)
            rows.append(pd.DataFrame({
                "pid": perf["pid"],
                "nll": _get_loss(perf),
                "model": model,
            }))

        if not rows:
            ax.text(0.5, 0.5, "No fitted models\nfor this task",
                    ha="center", va="center", transform=ax.transAxes,
                    color="0.5", style="italic")
            ax.set_title(title, color=TASK_COLORS[task_key])
            continue

        plot_df = pd.concat(rows, ignore_index=True)
        order = [m for m in NLL_MODEL_ORDER if m in plot_df["model"].unique()]
        pal = {m: NLL_MODEL_COLORS[m] for m in order}

        sns.boxplot(data=plot_df, x="model", y="nll", order=order,
                    hue="model", palette=pal, legend=False, ax=ax)
        ax.set_title(title, color=TASK_COLORS[task_key])
        ax.set_xlabel("")
        ax.set_ylabel("Model fit (NLL to\nhuman responses)" if i == 0 else "")
        ax.tick_params(axis="y", labelleft=(i == 0))
        ax.set_xticks([])
        sns.despine(ax=ax, top=True, right=True)
        panel_data.append((ax, task_key, title, order, plot_df))

    # No ax.set_ylim(bottom=0) here -- see this function's own docstring.
    y_lo, y_hi = axes[0].get_ylim()
    dy_step = (y_hi - y_lo) * 0.07
    per_panel_sig_lines = []
    max_bars = 0
    for ax, task_key, title, order, plot_df in panel_data:
        sig_lines = (_compute_sig_lines(plot_df, "model", "nll", order, NLL_REFERENCE)
                    if NLL_REFERENCE in order else [])
        per_panel_sig_lines.append((ax, sig_lines))
        max_bars = max(max_bars, len(sig_lines))

    if max_bars:
        axes[0].set_ylim(top=y_hi + dy_step * 0.5 + max_bars * dy_step * 2.0 + dy_step)

    for ax, sig_lines in per_panel_sig_lines:
        y_current = y_hi + dy_step * 0.5
        for x1, x2, stars in sig_lines:
            draw_sig_line(ax, x1, x2, y_current, stars)
            y_current += dy_step * 2.0

    legend_handles = [Patch(facecolor=NLL_MODEL_COLORS[m], label=NLL_LABELS.get(m, m))
                      for m in NLL_MODEL_ORDER]
    fig.get_layout_engine().set(h_pad=0.25)
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=5,
               frameon=True, framealpha=0.9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "model_performance_nll.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path



# -- Response variability growth + autocorrelation (balls, colors, numbers) --

# Matches figure_carrabin_temporal.py's rendered panels D (variance growth)
# and C (autocorrelation) for balls, and figure_soltani_temporal.py's cols
# 3-4 for colors/numbers -- both built on residuals against a qid-
# conditional mean (resid = response - mean(response | pid, observation,
# qid)), restricted to the window where a qid's repeats genuinely share an
# identical stimulus (see _variability_qid_map, reused directly here --
# same qid/quasi-qid/prefix machinery already established for the
# response-noise figures, not reinvented).

RESID_TASK_PANELS = [
    ("balls", "Balls task"),
    ("colors", "Colors task"),
    ("numbers", "Numbers task"),
]

# Minimum repeats required within one (pid, observation, qid) cell before
# its std is trusted, PER TASK -- copied from each task's OWN script AS
# CODED, not reconciled into one value. figure_carrabin_temporal.py's own
# panel D hardcodes MIN=3; figure_soltani_temporal.py's own
# _variance_growth_stats hardcodes MIN=2. This is a genuine, likely
# unintentional inconsistency between the two source scripts (neither this
# session's own instructions nor the scripts' own comments resolve it), so
# each task keeps its own established value rather than picking one to
# apply everywhere.
RESID_MIN_REPEATS = {"balls": 3, "colors": 2, "numbers": 2}

# Autocorrelation lag range, PER TASK -- again copied from each task's own
# script. carrabin explicitly excludes lag 4 despite having 5 observations
# ("lag-4 is degenerate: (obs1, obs5) residuals near-zero for some pids" --
# that script's own comment); colors' prefix is 5 (lags 1-4); numbers'
# prefix is 4 (lags 1-3).
RESID_LAGS = {"balls": [1, 2, 3], "colors": [1, 2, 3, 4], "numbers": [1, 2, 3]}


def _resid_frame(task_key: str, path: Path, qid_map: pd.DataFrame,
                 prefix: int | None) -> pd.DataFrame:
    """One (task, source)'s raw response file -> residuals against the
    qid-conditional mean, restricted to the shared-prefix window (or the
    whole trial for balls, prefix=None). Columns: [pid, trial, observation,
    qid, resid]."""
    df = pd.read_pickle(path)[["pid", "trial", "observation", "response"]]
    if prefix is not None:
        df = df[df["observation"] < prefix]
    df = df.merge(qid_map, on=["pid", "trial", "observation"])
    means = (df.groupby(["pid", "observation", "qid"])["response"]
             .mean().reset_index().rename(columns={"response": "qid_mean"}))
    df = df.merge(means, on=["pid", "observation", "qid"])
    df["resid"] = df["response"] - df["qid_mean"]
    return df


def _resid_variance_growth(resid_df: pd.DataFrame, min_repeats: int) -> pd.DataFrame | None:
    """Per-observation mean/SE (across pids) of within-qid residual SD --
    matching figure_carrabin_temporal.py's own panel D / figure_soltani_
    temporal.py's own _variance_growth_stats exactly (std within (pid, obs,
    qid) -> mean over qid within (pid, obs) -> mean/SEM across pids).
    Individual-pid overlay lines are DELIBERATELY omitted here (unlike both
    source scripts, which include them by default) -- this presentation
    deck hasn't shown per-pid overlays in any other figure, and the
    mean+SEM band alone matches this deck's own established convention."""
    grp = (resid_df.groupby(["pid", "observation", "qid"])["resid"]
           .apply(lambda x: x.std() if len(x) >= min_repeats else np.nan)
           .dropna().reset_index(name="std"))
    if grp.empty:
        return None
    by_pid_obs = grp.groupby(["pid", "observation"])["std"].mean().reset_index()
    stats = by_pid_obs.groupby("observation")["std"].agg(["mean", "std"]).reset_index()
    n_pid = by_pid_obs["pid"].nunique()
    stats["se"] = stats["std"] / np.sqrt(n_pid)
    return stats


def _resid_autocorr(resid_df: pd.DataFrame, lags: list[int]):
    """Cross-pid mean/SEM of within-trial lag-k residual autocorrelation --
    matching figure_soltani_temporal.py's own _autocorr_stats (pairs by
    ACTUAL observation index, not array position, so a missing checkpoint
    can't silently mispair two observations that aren't really `lag` apart;
    provably equivalent to carrabin's own position-based pairing for balls
    specifically, since every carrabin trial always has all 5 observations
    with no gaps -- so this one shared implementation is faithful to both
    source scripts, not a compromise between them). Returns (lags, means,
    sems) or "no_repeats"/"insufficient" (see figure_soltani_temporal.py's
    own degenerate-case guards, reproduced identically: a qid with only one
    repeat gives a trivially-zero residual, not a genuine signal)."""
    repeat_counts = resid_df.groupby(["pid", "observation", "qid"]).size()
    if not (repeat_counts >= 2).any():
        return "no_repeats"

    pid_rs: dict[int, list[float]] = {lag: [] for lag in lags}
    for _, pid_df in resid_df.groupby("pid"):
        for lag in lags:
            pairs = []
            for (_, _), g in pid_df.groupby(["pid", "trial"]):
                obs_to_resid = dict(zip(g["observation"], g["resid"]))
                for o, resid_o in obs_to_resid.items():
                    if (o + lag) in obs_to_resid:
                        pairs.append((resid_o, obs_to_resid[o + lag]))
            if len(pairs) < 3:
                continue
            arr = np.array(pairs)
            if arr[:, 0].std() <= 1e-9 or arr[:, 1].std() <= 1e-9:
                continue
            rv, _ = pearsonr(arr[:, 0], arr[:, 1])
            pid_rs[lag].append(rv)

    if all(len(v) == 0 for v in pid_rs.values()):
        return "insufficient"

    means = np.array([np.mean(pid_rs[lag]) if pid_rs[lag] else np.nan for lag in lags])
    sems = np.array([np.std(pid_rs[lag]) / np.sqrt(len(pid_rs[lag]))
                    if len(pid_rs[lag]) > 1 else np.nan for lag in lags])
    return lags, means, sems


# Hand-made schematic explaining the autocorrelation metric itself -- goes
# in panel A, same conceptual role VARIABILITY_SCHEMATIC plays for the
# response-noise figures. Not yet created (per instruction, left blank
# until it exists); the missing-file guard below prints a note and leaves
# the panel blank rather than failing.
AUTOCORR_SCHEMATIC = Path(__file__).resolve().parent / "images" / "autocorr_schematic.svg"


def make_variance_autocorr_human() -> Path:
    """1x4 panel: panel A holds AUTOCORR_SCHEMATIC (a hand-made diagram of
    the metric itself), composited in via _embed_svg_into_rect -- same
    convention as make_variability_human's own panel A; panels B-D are
    Human-only within-trial lag-k residual autocorrelation, one per task
    [balls, colors, numbers] -- snacks excluded, same reasoning as
    VARIABILITY_TASK_PANELS. Y-AXIS IS SHARED across B-D (sharey=True), so
    the three tasks' decay is directly comparable rather than each panel
    autoscaling to its own range.

    VARIANCE GROWTH (the OLD top row) WAS DROPPED, per instruction, after
    checking the actual numbers directly rather than relying on the
    earlier visual read (which was wrong -- see chat): only Human and
    NoisyRL_lambda show a genuine, substantial DECAYING autocorrelation
    (starting well above zero, decaying toward/past it); every
    "_resp_noise" model (Mean/LeakyIntegrator/PrimacyRecency/RL_lambda)
    stays within about +-0.09 of zero at EVERY lag in EVERY task -- noise
    scatter around zero, not a real signal. Autocorrelation alone is the
    metric that actually distinguishes state-persistent noise from pure
    i.i.d. response noise; variance growth did not (it grew inconsistently
    for the resp_noise models too, likely just sampling noise around a
    flat truth).
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, sharey=True,
                             constrained_layout=True)

    axes[0].axis("off")
    axes[0].set_title("Metric Definition", color="0.3")

    for i, (ax_ac, (task_key, title)) in enumerate(zip(axes[1:], RESID_TASK_PANELS)):
        qid_map, prefix = _variability_qid_map(task_key)
        human_path = _human_data_path(task_key)
        resid_df = _resid_frame(task_key, human_path, qid_map, prefix)

        lags = RESID_LAGS[task_key]
        res = _resid_autocorr(resid_df, lags)
        ax_ac.set_title(title, color=TASK_COLORS[task_key])
        if isinstance(res, str):
            msg = ("Insufficient data\n(no qid repeats for this task)"
                  if res == "no_repeats" else "Insufficient data")
            ax_ac.text(0.5, 0.5, msg, ha="center", va="center",
                      transform=ax_ac.transAxes, color="0.5", style="italic")
        else:
            _, means, sems = res
            ax_ac.axhline(0, color="0.7", lw=0.8, ls="--", zorder=1)
            ax_ac.plot(lags, means, "-", color=HUMAN_COLOR, lw=2.2, zorder=3)
            ax_ac.fill_between(lags, means - sems, means + sems, color=HUMAN_COLOR,
                              alpha=0.2, zorder=1)
        ax_ac.set_xlabel("Lag (observations)")
        ax_ac.set_ylabel("Autocorrelation" if i == 0 else "")
        ax_ac.tick_params(axis="y", labelleft=(i == 0))
        ax_ac.set_xticks(lags)
        ax_ac.margins(x=0.15)  # keeps the rightmost tick label from
        # clipping against the canvas edge in the last column -- there's
        # no neighboring panel there to absorb the overflow.
        sns.despine(ax=ax_ac, top=True, right=True)

    panel_a_rect = _panel_a_rect(fig, axes[0])

    # Same legend SLOT reserved as make_variance_autocorr_models (same
    # h_pad, same "outside lower center" placement) -- so the figure
    # doesn't resize/shift when models get added on the follow-up slide;
    # only "Human" is actually shown yet, matching make_lambda_human's own
    # human-only stage convention exactly.
    fig.get_layout_engine().set(h_pad=0.25)
    fig.legend(handles=[Line2D([0], [0], color=HUMAN_COLOR, lw=2.2, label="Human")],
               loc="outside lower center", ncol=1, frameon=True, framealpha=0.9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "variance_autocorr_human.svg"
    fig.savefig(out_path)
    plt.close(fig)

    if AUTOCORR_SCHEMATIC.exists():
        _embed_svg_into_rect(out_path, AUTOCORR_SCHEMATIC, panel_a_rect, "panelA_")
    else:
        print(f"  (missing {AUTOCORR_SCHEMATIC.name} -- panel A left blank)")

    print(f"Saved {out_path}")
    return out_path


# The full 5-model roster for the models figure -- unlike
# make_model_performance_nll's boxplot (which dropped RL_lambda's own box
# as redundant with NoisyRL_lambda at that single x-position), both
# RL_lambda and NoisyRL_lambda are kept here with their own established
# MODEL_COLORS: their CURVES can look genuinely different (RL_lambda's own
# added sigma_resp vs NoisyRL_lambda's own sigma_state mechanism), so
# showing both is informative rather than redundant.
RESID_MODEL_ORDER = ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NoisyRL_lambda"]


def _nll_responses_path(task_key: str, model: str) -> Path:
    """Path to one (task, model)'s *_nll_responses.pkl -- same "_resp_noise"
    file-naming quirk as _nll_perf_path (see that function's own
    docstring), just pointing at the actual per-observation response
    SEQUENCE (needed to compute a residual) rather than the scalar
    performance loss."""
    file_model = model if model == "NoisyRL_lambda" else f"{model}_resp_noise"
    if task_key == "balls":
        return RUNS_DIR / "carrabin" / f"{file_model}_carrabin_nll_responses.pkl"
    dataset = "soltani_colors" if task_key == "colors" else "soltani_numbers"
    return RUNS_DIR / "soltani" / f"{file_model}_{dataset}_nll_responses.pkl"


def make_variance_autocorr_models() -> Path:
    """Same 1x4 layout as make_variance_autocorr_human (panel A unchanged),
    now overlaying each of the 5 NLL-fitted models on panels B-D's
    autocorrelation. Confirms directly (not just visually -- the earlier
    visual read of the 2x3 version was wrong, see chat) that only
    NoisyRL_lambda shows genuine decaying autocorrelation resembling
    Human's own pattern; the four "_resp_noise" models stay within noise
    of zero at every lag in every task, matching what their own math
    predicts (i.i.d. response noise, added AFTER the clean deterministic
    trajectory per models/math_models.py's own add_noise(), has no
    mechanism to produce lag correlation).
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, sharey=True,
                             constrained_layout=True)

    axes[0].axis("off")
    axes[0].set_title("Metric Definition", color="0.3")

    legend_handles = [Line2D([0], [0], color=HUMAN_COLOR, lw=2.2)]
    legend_labels = ["Human"]
    for m in RESID_MODEL_ORDER:
        legend_handles.append(Line2D([0], [0], color=MODEL_COLORS[m], lw=2.2))
        legend_labels.append(MODEL_DISPLAY.get(m, m))

    for i, (ax_ac, (task_key, title)) in enumerate(zip(axes[1:], RESID_TASK_PANELS)):
        qid_map, prefix = _variability_qid_map(task_key)
        human_path = _human_data_path(task_key)
        human_resid = _resid_frame(task_key, human_path, qid_map, prefix)

        model_resids = {}
        for m in RESID_MODEL_ORDER:
            mpath = _nll_responses_path(task_key, m)
            if not mpath.exists():
                print(f"  (missing {mpath.name} -- skipping {m} for {task_key})")
                continue
            model_resids[m] = _resid_frame(task_key, mpath, qid_map, prefix)

        lags = RESID_LAGS[task_key]
        ax_ac.set_title(title, color=TASK_COLORS[task_key])
        ax_ac.axhline(0, color="0.7", lw=0.8, ls="--", zorder=1)
        res = _resid_autocorr(human_resid, lags)
        if isinstance(res, str):
            msg = ("Insufficient data\n(no qid repeats for this task)"
                  if res == "no_repeats" else "Insufficient data")
            ax_ac.text(0.5, 0.5, msg, ha="center", va="center",
                      transform=ax_ac.transAxes, color="0.5", style="italic")
        else:
            _, means, sems = res
            ax_ac.plot(lags, means, "-", color=HUMAN_COLOR, lw=2.2, zorder=6)
            ax_ac.fill_between(lags, means - sems, means + sems, color=HUMAN_COLOR,
                              alpha=0.2, zorder=1)
        for j, m in enumerate(RESID_MODEL_ORDER):
            if m not in model_resids:
                continue
            mres = _resid_autocorr(model_resids[m], lags)
            if isinstance(mres, str):
                continue
            _, mmeans, msems = mres
            color = MODEL_COLORS[m]
            ax_ac.plot(lags, mmeans, "-", color=color, lw=2.0, zorder=5 - j)
            ax_ac.fill_between(lags, mmeans - msems, mmeans + msems, color=color,
                              alpha=0.15, zorder=1)
        ax_ac.set_xlabel("Lag (observations)")
        ax_ac.set_ylabel("Autocorrelation" if i == 0 else "")
        ax_ac.tick_params(axis="y", labelleft=(i == 0))
        ax_ac.set_xticks(lags)
        ax_ac.margins(x=0.15)
        sns.despine(ax=ax_ac, top=True, right=True)

    panel_a_rect = _panel_a_rect(fig, axes[0])

    fig.get_layout_engine().set(h_pad=0.25)
    fig.legend(handles=legend_handles, labels=legend_labels,
               loc="outside lower center", ncol=6, frameon=True, framealpha=0.9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "variance_autocorr_models.svg"
    fig.savefig(out_path)
    plt.close(fig)

    if AUTOCORR_SCHEMATIC.exists():
        _embed_svg_into_rect(out_path, AUTOCORR_SCHEMATIC, panel_a_rect, "panelA_")
    else:
        print(f"  (missing {AUTOCORR_SCHEMATIC.name} -- panel A left blank)")

    print(f"Saved {out_path}")
    return out_path


FIGURES = {
    "temporal_performance": make_temporal_performance,
    "model_performance": make_model_performance,
    "model_performance_nll": make_model_performance_nll,
    "response_change": make_response_change,
    "lambda_human": make_lambda_human,
    "lambda_models": make_lambda_models,
    "lambda_sanity_human": make_lambda_sanity_human,
    "lambda_sanity_models": make_lambda_sanity_models,
    "variability_human": make_variability_human,
    "variability_models": make_variability_models,
    "variance_autocorr_human": make_variance_autocorr_human,
    "variance_autocorr_models": make_variance_autocorr_models,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=sorted(FIGURES),
                        help="Which presentation figure to (re)generate.")
    args = parser.parse_args()
    FIGURES[args.figure]()


if __name__ == "__main__":
    main()
