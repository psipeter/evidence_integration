#!/usr/bin/env python3
"""scripts/make_paper_figures.py

Main-paper figures, extended from presentations/make_figures.py (copied on
2026-08-31, NOT moved -- the presentation script stays exactly as it was,
under its own DELIBERATE exception to the "all new scripts go in scripts/"
rule, generating slide SVGs for the talk deck; this copy is the normal-rules
version, living in scripts/ like every other analysis script, producing the
paper's own figures under the project's real figures/ directory (PDF, per
convention) rather than presentations/figures/ (SVG, for Quarto/reveal.js
embedding).

Inherited AS-IS from the presentation script for now, not yet re-verified or
updated for the paper: every figure function below except
make_model_performance (remade this session -- see its own docstring for
what changed). Many still reference presentation-specific conventions (SVG
compositing helpers, slide-only styling, stand-in-model comments that
predate this session's real NEF RMSE fits) that will need revisiting one
figure at a time as this script is extended, not assumed correct just
because they were correct for the deck.

Run:
    venv/bin/python scripts/make_paper_figures.py model_performance
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

from utils.paths import data_path, RUNS_DIR, FIGURES_DIR
from utils.aggregate import plot_error_aggregate, plot_delta_aggregate
from utils.plot_style import draw_sig_line, pvalue_to_stars, get_palette
from utils.plot_spikes import plot_spikes, preprocess_spikes, sample_by_variance, cluster

# NOT presentations/make_figures.py's own local
# `Path(__file__).resolve().parent / "figures"` (that pointed at
# presentations/figures/, for slide SVGs) -- this is the project's real,
# shared figures/ directory (utils.paths.FIGURES_DIR), same convention
# every other scripts/figure_*.py file uses.
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


def _save_fig(fig, stem: str) -> tuple[Path, Path]:
    """Save `fig` as BOTH {stem}.pdf and {stem}.svg under FIGURES_DIR -- the
    house convention for every figure in THIS script (unlike
    presentations/make_figures.py, which is SVG-only, for slide embedding).
    One shared helper so every figure function saves both formats
    identically rather than each hand-rolling its own pair of fig.savefig()
    calls -- exactly the kind of per-function repetition that let several
    of this file's figures drift to PDF-missing (inherited from the
    presentation script, SVG-only) before this helper existed.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    svg_path = FIGURES_DIR / f"{stem}.svg"
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")
    return pdf_path, svg_path


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
    out_path, _ = _save_fig(fig, "temporal_performance")
    plt.close(fig)
    return out_path


# ── Model performance across all four tasks ─────────────────────────────────

# Model colors, matching presentations/images/model_palette.svg's own
# seaborn-colorblind indices for Mean/LeakyIntegrator/PrimacyRecency/
# RL_lambda/NEF (0-4) and NoisyRL_lambda (5). Presentations/make_figures.py
# pins RL_lambda and NEF to the SAME color there (#d55e00) because NEF
# hadn't been fit for 3 of 4 tasks when that file was written, and RL_lambda
# stood in for it -- same color signaled "playing the same conceptual
# role". That reason no longer applies here: real NEF RMSE fits now exist
# for all 4 datasets (this session's weekend submit, run_folder rmse), so
# NEF gets its own genuine slot back -- seaborn colorblind index 4
# (#cc78bc, pink), the color it already had everywhere in this palette
# except where it was being deliberately overridden.
MODEL_COLORS = {
    "Mean": "#0173b2",
    "LeakyIntegrator": "#de8f05",
    "PrimacyRecency": "#029e73",
    "RL_lambda": "#d55e00",
    "NEF": "#cc78bc",
    # Index 5 of the same seaborn-colorblind MODEL_ORDER palette (see
    # utils/soltani_models.py's own module docstring: "NoisyRL_lambda is last
    # so that adding it left every existing model's colour untouched").
    "NoisyRL_lambda": "#ca9161",
}

# RL_lambda's internal Python name (a valid identifier -- used as a dict
# key and as a component of every filename it touches) was never meant to
# be read literally as legend text. Defined ONCE, here, so every legend
# convention below renders it identically rather than drifting: MODEL_LABEL
# (this dict -- full model names, just this one reformatted, e.g.
# make_model_performance's own 5-entry legend) and MODEL_DISPLAY (defined
# further down, for the abbreviated multi-source legends elsewhere in this
# file -- LI/PR/etc) both use it.
_RL_LAMBDA_PRETTY = r"RL-$\lambda$"
MODEL_LABEL = {"RL_lambda": _RL_LAMBDA_PRETTY}

# (task_key, panel title, model list) -- ALL FOUR tasks now get the SAME
# 5-model roster (Mean/LeakyIntegrator/PrimacyRecency/RL_lambda/NEF), unlike
# presentations/make_figures.py's own TASK_PANELS, which used a 4-model list
# with RL_lambda standing in for NEF on 3 of 4 tasks. That stand-in is gone:
# this session's weekend submit landed real NEF RMSE fits for every dataset
# (data/runs/rmse/, run_folder rmse -- see _model_fit_path below for exactly
# which file each model/task reads). Carrabin's own earlier NEF fit
# (data/runs/carrabin/, INCOMPLETE at 16/21 pids) and yoo's
# (data/runs/refit/, the old smaller-n_neurons version) are NOT what this
# reads -- both are superseded by the fresh run_folder rmse fit for their
# task, at n_neurons=500 (n_neurons_counting=500 for carrabin, 2000 for the
# other three -- see fitting/model_params.py's own module docstring for why
# carrabin differs).
TASK_PANELS = [
    ("balls", "Balls task", ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"]),
    ("snacks", "Snacks task", ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"]),
    ("colors", "Colors task", ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"]),
    ("numbers", "Numbers task", ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"]),
]


def _model_fit_path(task_key: str, model: str) -> Path:
    """Path to one (task, model)'s *_performance.pkl -- the fitted CV loss
    (RMSE to human responses), read the same way every working figure does
    (via _get_loss, never a hardcoded column name).

    NEF, all 4 tasks: ALWAYS data/runs/rmse/NEF_{stem}_performance.pkl --
    this session's fresh weekend fit, run_folder rmse. NOT carrabin's own
    data/runs/carrabin/ (that fit is INCOMPLETE, 16/21 pids) or yoo's
    data/runs/refit/ (the old, smaller-n_neurons fit) -- both superseded.

    Every OTHER model keeps its EXISTING location, unchanged by this
    session's NEF-only submit:
      balls (carrabin)  -> data/runs/carrabin/{model}_carrabin_performance.pkl
      snacks (yoo)       -> data/runs/yoo/{model}_yoo_performance.pkl
      colors/numbers     -> data/runs/rmse/{model}_soltani_{task}_performance.pkl
                            -- fit against the CORRECTED, contamination-free
                            46-pid canonical data (see chat: the pilot-4/
                            pid-registry fixes). NOT data/runs/soltani/,
                            which holds the earlier, stale fits against the
                            pre-fix data and is no longer read by any
                            function in this file.
    """
    dataset = {"balls": "carrabin", "snacks": "yoo",
               "colors": "soltani_colors", "numbers": "soltani_numbers"}[task_key]
    if model == "NEF":
        return RUNS_DIR / "rmse" / f"NEF_{dataset}_performance.pkl"
    if task_key == "balls":
        return RUNS_DIR / "carrabin" / f"{model}_carrabin_performance.pkl"
    if task_key == "snacks":
        return RUNS_DIR / "yoo" / f"{model}_yoo_performance.pkl"
    return RUNS_DIR / "rmse" / f"{model}_{dataset}_performance.pkl"


def _get_loss(perf: pd.DataFrame) -> pd.Series:
    """Never hardcode cv_loss_mean -- project convention (utils.plot_style /
    every figure_*.py script's own _get_loss)."""
    return perf["loss"] if "loss" in perf.columns else perf["cv_loss_mean"]


def _compute_sig_lines(plot_df: pd.DataFrame, x_col: str, y_col: str,
                        order: list[str], ref_label: str,
                        exclude: frozenset = frozenset()) -> list[tuple[int, int, str]]:
    """Paired-Wilcoxon comparisons from `ref_label` to every other model in
    `order` EXCEPT those in `exclude`, returning ONLY the ones where
    `ref_label` is significantly BETTER (lower RMSE) -- a one-sided variant
    of utils.plot_style.annotate_nef_comparisons, which flags any significant
    two-sided difference regardless of direction. That direction-blindness
    is the right default for the working figures ("does NEF differ from
    this model"), but this talk's point is narrower ("is our model
    better"), so a comparison NEF/RL_lambda significantly LOSES is left out
    rather than flagged the same way a win would be.

    `exclude` keeps a model IN `order` (so x-axis positions/box placement
    stay correct) while removing it from the comparison candidates -- e.g.
    make_model_performance excludes NEF when ref_label="RL_lambda", since
    NEF is expected to rarely beat RL_lambda by design (added noise on top
    of the same underlying behavior it's built to reproduce), so that
    specific comparison isn't a meaningful "is RL_lambda better" question
    the way RL_lambda-vs-the-math-models is.

    Returns (x1, x2, stars) tuples, sorted nearest-first, with no drawing or
    ylim side effects -- kept separate from the actual line-drawing so a
    caller can compute every panel's bars BEFORE deciding how much headroom
    the shared y-axis needs (see make_model_performance's own sharey note).
    """
    x_positions = {m: i for i, m in enumerate(order)}
    if ref_label not in x_positions:
        return []

    candidates = [m for m in order if m != ref_label and m not in exclude]
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


def _best_fit_counts(plot_df: pd.DataFrame, order: list[str],
                     value_col: str = "rmse") -> tuple[pd.Series, int]:
    """Win-count (lowest value per pid) for each model in `order`, restricted
    to pids with EVERY model in `order` present -- a fair head-to-head. A
    pid missing one model (e.g. yoo's pid 30, missing NEF) can't say which
    model would have won for it, so it's excluded rather than guessed by
    comparing across a smaller, inconsistent subset of models per pid.
    `value_col` is generic over the metric ('rmse' or 'nll' -- both are
    lower-is-better, see make_model_performance_nll's own module comment).
    Returns (counts indexed by model in `order`'s own order, n pids counted).
    """
    wide = plot_df.pivot(index="pid", columns="model", values=value_col)
    wide = wide.dropna(subset=order)
    if wide.empty:
        return pd.Series(0, index=order), 0
    best = wide[order].idxmin(axis=1)
    counts = best.value_counts().reindex(order, fill_value=0)
    return counts, len(wide)


def _print_best_fit_counts(panel_data: list, value_col: str = "rmse",
                           metric_label: str = "RMSE") -> None:
    """For each task, how many pids are best-fit (lowest RMSE or NLL) by
    each model -- a simple win-count ranking, distinct from the boxplot's
    own median comparison (a model can have a lower median while still
    losing the per-pid count, if its wins are concentrated on a few pids
    with unusually large gaps -- this print surfaces that possibility
    rather than leaving it implicit in the boxplot alone).

    `panel_data` is a (ax, task_key, title, order, plot_df) list -- reused
    directly rather than reloading anything, so these counts are
    guaranteed to reflect the EXACT same data the figure itself just
    plotted, and the exact same counts _plot_best_fit_bar draws.
    """
    print(f"\nBest-fit model counts (lowest {metric_label} per pid), by task:")
    totals: dict[str, int] = {}
    for ax, task_key, title, order, plot_df in panel_data:
        counts, n = _best_fit_counts(plot_df, order, value_col)
        if n == 0:
            print(f"  {title}: no pids with all {len(order)} models fit")
            continue
        print(f"  {title} (n={n} pids with all models fit):")
        for model, count in counts.items():
            print(f"    {model:18s} {int(count):3d}/{n}  ({100*count/n:.0f}%)")
            totals[model] = totals.get(model, 0) + int(count)

    if totals:
        print("\nOverall ordering (total best-fit count summed across all 4 tasks):")
        for model, count in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"  {model:18s} {count:3d}")


def _plot_best_fit_bar(ax, plot_df: pd.DataFrame, order: list[str],
                       show_ylabel: bool, value_col: str = "rmse") -> None:
    """Fraction of pids best-fit (lowest value) by each model, as a plain
    bar chart -- the plotted counterpart of _print_best_fit_counts's own
    stdout table (same _best_fit_counts call, so the two can never
    disagree). Deliberately minimal -- no axis labels/ticks beyond the
    y-axis, no title, no legend of its own -- model identity is carried by
    color (MODEL_COLORS) and by a shared legend elsewhere.
    """
    counts, n = _best_fit_counts(plot_df, order, value_col)
    if n == 0:
        ax.text(0.5, 0.5, "No pids with\nall models fit", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    fracs = (counts / n).to_numpy()
    colors = [MODEL_COLORS[m] for m in order]
    ax.bar(range(len(order)), fracs, color=colors, width=0.7)
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction\nbest fit" if show_ylabel else "")
    ax.tick_params(axis="y", labelleft=show_ylabel, labelsize=7)
    if show_ylabel:
        ax.set_yticks([0, 0.5, 1])
    sns.despine(ax=ax, top=True, right=True)


def _gather_metric_data(task_key: str, models: list[str], path_fn, get_loss_fn,
                        value_col: str):
    """Load one task's per-pid loss for every model in `models` (skipping
    any whose file is missing, with a printed note), returning
    (order, plot_df) -- order is `models` filtered to those that actually
    had data, plot_df has columns [pid, value_col, model]. Returns None if
    NO model had data for this task. Generic over the metric via `path_fn`
    (task_key, model) -> Path and `get_loss_fn` (perf_df) -> Series (always
    _get_loss in practice, but not hardcoded here), so the exact same
    function drives both RMSE and NLL panels without duplicating this
    loading loop per metric.
    """
    rows = []
    for model in models:
        path = path_fn(task_key, model)
        if not path.exists():
            print(f"  (missing {path.name} -- skipping {model} for {task_key})")
            continue
        perf = pd.read_pickle(path)
        rows.append(pd.DataFrame({
            "pid": perf["pid"],
            value_col: get_loss_fn(perf),
            "model": model,
        }))
    if not rows:
        return None
    plot_df = pd.concat(rows, ignore_index=True)
    order = [m for m in models if m in plot_df["model"].unique()]
    return order, plot_df


def _draw_metric_boxplot(ax, task_key: str, title: str, order: list[str],
                         plot_df: pd.DataFrame, value_col: str, show_ylabel: bool,
                         ylabel: str) -> None:
    """Draw one boxplot panel from already-gathered (order, plot_df) data
    (see _gather_metric_data) -- the plotting half split out from that
    loading half so make_model_best_fit can reuse the DATA without
    redrawing a boxplot from it."""
    pal = {m: MODEL_COLORS[m] for m in order}
    sns.boxplot(data=plot_df, x="model", y=value_col, order=order,
                hue="model", palette=pal, legend=False, ax=ax)
    ax.set_title(title, color=TASK_COLORS[task_key])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel if show_ylabel else "")
    ax.tick_params(axis="y", labelleft=show_ylabel)
    ax.set_xticks([])
    sns.despine(ax=ax, top=True, right=True)


def make_model_performance() -> Path:
    """2-row, 4-column figure: TOP row is model fit under RMSE (to human
    responses), BOTTOM row is model fit under NLL/quasi-MLE -- one
    consolidated figure covering both metrics, per instruction, replacing
    this figure's own earlier second row (fraction of pids each model
    best-fits). That best-fit content has NOT been dropped -- it now lives
    in its own figure, make_model_best_fit, covering BOTH metrics there
    (see that function's own docstring) -- just no longer crammed into a
    height_ratios=[4,1] strip under each boxplot here. The person plans to
    add those fraction-best-fit panels back in as small Inkscape insets
    onto each panel of THIS figure by hand, so this file no longer needs
    to lay the two out together itself.

    TOP ROW (RMSE): 5 models per task -- Mean, LeakyIntegrator,
    PrimacyRecency, RL_lambda, NEF (TASK_PANELS, _model_fit_path) -- same
    roster/fits this figure always used. Y-axis shared ACROSS the row
    (sharey='row') and forced to start at 0 (RMSE is non-negative by
    construction).

    BOTTOM ROW (NLL): 4 models per task -- Mean, LeakyIntegrator,
    PrimacyRecency, RL_lambda, all "_resp_noise" variants, all read from
    the unified data/runs/nll/ folder (NLL_RESP_NOISE_MODELS,
    _nll_resp_noise_perf_path -- see that constant's own comment for the
    session's refit this depends on). NEF is not included -- no NLL fit
    exists for it yet. Y-axis shared ACROSS the row separately from the
    top row (sharey='row', NOT a single sharey=True for the whole figure)
    and NOT forced to start at 0 -- NLL can be, and often is, negative.

    SIGNIFICANCE BARS in BOTH rows: RL_lambda vs the 3 math models, drawn
    in RL_lambda's own color (MODEL_COLORS["RL_lambda"]) -- top row
    excludes NEF from the comparison candidates (NEF is built to reproduce
    RL_lambda's own behavior plus noise, so it's expected to rarely beat
    it by design, not a meaningful comparison -- see this function's
    inherited reasoning below); bottom row has no NEF to exclude at all.
    Each row's headroom is computed independently (its own y_lo/y_hi/
    dy_step), matching how the old single-metric figures each did this on
    their own shared axis.

    ONE shared 5-entry legend at the bottom (Mean/LeakyIntegrator/
    PrimacyRecency/RL_lambda/NEF) -- NEF only actually appears in the top
    row's boxes, but including it costs nothing and keeps one legend
    covering every color used anywhere in the figure, rather than two
    separate row-specific legends.

    Saved as model_performance.pdf/.svg (renamed from
    model_performance_rmse, since the figure is no longer RMSE-only) --
    if anything outside this script (LaTeX source, etc.) still references
    the old "model_performance_rmse" filename, that reference will need
    updating too.

    Still PRINTS best-fit counts for BOTH metrics (via
    _print_best_fit_counts, parameterized by value_col/metric_label) --
    kept here since these numbers are directly about what this exact
    figure just plotted, even though the corresponding BAR panels now
    live in make_model_best_fit instead.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(2, 4, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 1.9 * 0.75),
                             sharey="row", constrained_layout=True)

    rmse_panel_data = []
    for i, (ax, (task_key, title, models)) in enumerate(zip(axes[0], TASK_PANELS)):
        gathered = _gather_metric_data(task_key, models, _model_fit_path, _get_loss, "rmse")
        if gathered is None:
            ax.text(0.5, 0.5, "No fitted models\nfor this task",
                    ha="center", va="center", transform=ax.transAxes,
                    color="0.5", style="italic")
            ax.set_title(title, color=TASK_COLORS[task_key])
            continue
        order, plot_df = gathered
        _draw_metric_boxplot(ax, task_key, title, order, plot_df, "rmse",
                             show_ylabel=(i == 0), ylabel="Model fit (RMSE)")
        rmse_panel_data.append((ax, task_key, title, order, plot_df))
    axes[0, 0].set_ylim(bottom=0)  # shared across the row via sharey='row'

    nll_panel_data = []
    for i, (ax, (task_key, title)) in enumerate(zip(axes[1], NLL_TASK_PANELS)):
        gathered = _gather_metric_data(task_key, NLL_RESP_NOISE_MODELS,
                                       _nll_resp_noise_perf_path, _get_loss, "nll")
        if gathered is None:
            ax.text(0.5, 0.5, "No fitted models\nfor this task",
                    ha="center", va="center", transform=ax.transAxes,
                    color="0.5", style="italic")
            ax.set_title(title, color=TASK_COLORS[task_key])
            continue
        order, plot_df = gathered
        _draw_metric_boxplot(ax, task_key, title, order, plot_df, "nll",
                             show_ylabel=(i == 0), ylabel="Model fit (NLL)")
        nll_panel_data.append((ax, task_key, title, order, plot_df))
    # No axes[1, 0].set_ylim(bottom=0) -- NLL can be negative.

    sig_color = MODEL_COLORS["RL_lambda"]

    def _draw_row_sig_bars(row_axis_0, panel_data, ref_label, exclude, value_col):
        y_lo, y_hi = row_axis_0.get_ylim()
        dy_step = (y_hi - y_lo) * 0.07
        per_panel_sig_lines = []
        max_bars = 0
        for ax, task_key, title, order, plot_df in panel_data:
            sig_lines = (_compute_sig_lines(plot_df, "model", value_col, order, ref_label,
                                            exclude=exclude)
                        if ref_label in order else [])
            per_panel_sig_lines.append((ax, sig_lines))
            max_bars = max(max_bars, len(sig_lines))
        if max_bars:
            row_axis_0.set_ylim(top=y_hi + dy_step * 0.5 + max_bars * dy_step * 2.0 + dy_step)
        for ax, sig_lines in per_panel_sig_lines:
            y_current = y_hi + dy_step * 0.5
            for x1, x2, stars in sig_lines:
                draw_sig_line(ax, x1, x2, y_current, stars, color=sig_color)
                y_current += dy_step * 2.0

    _draw_row_sig_bars(axes[0, 0], rmse_panel_data, "RL_lambda", frozenset({"NEF"}), "rmse")
    _draw_row_sig_bars(axes[1, 0], nll_panel_data, NLL_RESP_NOISE_REFERENCE, frozenset(), "nll")

    legend_handles = [Patch(facecolor=MODEL_COLORS[m], label=MODEL_LABEL.get(m, m))
                      for m in ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"]]
    fig.get_layout_engine().set(h_pad=0.25)
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=5,
               frameon=True, framealpha=0.9)

    _print_best_fit_counts(rmse_panel_data, value_col="rmse", metric_label="RMSE")
    _print_best_fit_counts(nll_panel_data, value_col="nll", metric_label="NLL")

    out_path, _ = _save_fig(fig, "model_performance")
    plt.close(fig)
    return out_path


def make_model_best_fit() -> Path:
    """2-row, 4-column figure: for each task, the fraction of pids best-fit
    (lowest loss) by each model -- TOP row under RMSE, BOTTOM row under
    NLL, mirroring make_model_performance's own row/column layout exactly
    (same tasks in the same column order, same metric in the same row) so
    each small panel here can be dropped in as an Inkscape inset directly
    onto the corresponding panel of that figure, per instruction. Reuses
    the EXACT SAME data-loading (_gather_metric_data) and bar-drawing
    (_plot_best_fit_bar) helpers make_model_performance's own predecessor
    used for this content -- only the layout (now its own standalone
    figure, not a reserved sub-row) is new.

    Deliberately minimal -- no titles, no per-panel axis labels beyond a
    shared y-axis per row, no legend -- these are meant to be small,
    croppable insets, not a standalone readable figure; task/model
    identity is carried entirely by position (matching
    make_model_performance's own layout) and by color (MODEL_COLORS,
    matching that figure's boxes and legend exactly).

    Also prints the same best-fit counts make_model_performance itself
    prints (same _print_best_fit_counts calls, same underlying data), so
    the two are guaranteed to agree without needing to run both.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(2, 4, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 0.75),
                             sharey="row", constrained_layout=True)

    rmse_panel_data = []
    for i, (ax, (task_key, title, models)) in enumerate(zip(axes[0], TASK_PANELS)):
        gathered = _gather_metric_data(task_key, models, _model_fit_path, _get_loss, "rmse")
        if gathered is None:
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        order, plot_df = gathered
        _plot_best_fit_bar(ax, plot_df, order, show_ylabel=(i == 0), value_col="rmse")
        rmse_panel_data.append((ax, task_key, title, order, plot_df))

    nll_panel_data = []
    for i, (ax, (task_key, title)) in enumerate(zip(axes[1], NLL_TASK_PANELS)):
        gathered = _gather_metric_data(task_key, NLL_RESP_NOISE_MODELS,
                                       _nll_resp_noise_perf_path, _get_loss, "nll")
        if gathered is None:
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        order, plot_df = gathered
        _plot_best_fit_bar(ax, plot_df, order, show_ylabel=(i == 0), value_col="nll")
        nll_panel_data.append((ax, task_key, title, order, plot_df))

    _print_best_fit_counts(rmse_panel_data, value_col="rmse", metric_label="RMSE")
    _print_best_fit_counts(nll_panel_data, value_col="nll", metric_label="NLL")

    out_path, _ = _save_fig(fig, "model_best_fit")
    plt.close(fig)
    return out_path


# ── Response change decay across all four tasks ───────────────────

# Reuses TASK_PANELS directly (defined above, near MODEL_COLORS) -- no
# separate DELTA_TASK_PANELS any more. presentations/make_figures.py's own
# version needed a per-task 4th-slot choice (NEF for balls/snacks,
# RL_lambda standing in for colors/numbers); that's gone now that real NEF
# RMSE fits exist for all 4 datasets, so this figure's model roster is
# identical to make_model_performance's: Mean/LeakyIntegrator/
# PrimacyRecency/RL_lambda/NEF, every task.

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
    scalar *_performance.pkl loss make_model_performance reads.

    NEF, all 4 tasks: ALWAYS data/runs/rmse/NEF_{stem}_responses.pkl -- the
    same fresh weekend RMSE fit _model_fit_path reads for
    make_model_performance. This deliberately DROPS two presentations/
    make_figures.py quirks: balls' NEF used to come from the MLE-fitted
    variant (NEF_carrabin_responses_mle.pkl, matching that deck's
    figure_carrabin_temporal.py panel B), and snacks' NEF used to come from
    data/runs/refit/ (the old, smaller-n_neurons fit). Both are gone so
    that "NEF" means the SAME fit, everywhere, in this figure AND in
    make_model_performance -- a reader shouldn't have to wonder whether
    two panels showing "NEF" in the same paper are secretly two different
    fits of it.

    Every other model keeps its existing location, unchanged:
      balls (carrabin)  -> data/runs/carrabin/{model}_carrabin_responses.pkl
      snacks (yoo)       -> data/runs/yoo/{model}_yoo_responses.pkl
      colors/numbers     -> data/runs/rmse/{model}_soltani_{task}_responses.pkl
    """
    dataset = {"balls": "carrabin", "snacks": "yoo",
               "colors": "soltani_colors", "numbers": "soltani_numbers"}[task_key]
    if model == "NEF":
        return RUNS_DIR / "rmse" / f"NEF_{dataset}_responses.pkl"
    if task_key == "balls":
        return RUNS_DIR / "carrabin" / f"{model}_carrabin_responses.pkl"
    if task_key == "snacks":
        return RUNS_DIR / "yoo" / f"{model}_yoo_responses.pkl"
    return RUNS_DIR / "rmse" / f"{model}_{dataset}_responses.pkl"


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
    """task_key -> (human_delta_df, {model: delta_df}, title). Loaded ONCE
    and shared between the human-only and human+models figure passes below,
    so both read the exact same underlying data. Model roster is
    TASK_PANELS' own 5-model list (Mean/LeakyIntegrator/PrimacyRecency/
    RL_lambda/NEF), identical for every task now that real NEF RMSE fits
    exist everywhere.
    """
    out = {}
    for task_key, title, models_list in TASK_PANELS:
        min_obs = DELTA_MIN_OBS[task_key]
        first_resp = FIRST_OBS_IS_RESPONSE[task_key]

        human_df = pd.read_pickle(_human_data_path(task_key))
        human_delta = _abs_delta_long(
            human_df[["pid", "trial", "observation", "response"]], min_obs, first_resp)

        model_deltas = {}
        for model in models_list:
            path = _delta_responses_path(task_key, model)
            if not path.exists():
                print(f"  (missing {path.name} -- skipping {model} for {task_key})")
                continue
            mdf = pd.read_pickle(path)[["pid", "trial", "observation", "response"]]
            model_deltas[model] = _abs_delta_long(mdf, min_obs, first_resp)

        out[task_key] = (human_delta, model_deltas, title)
    return out


def _four_xticks(obs_max: float) -> list[int]:
    """[0, ..., obs_max] with exactly 2 evenly-spaced intermediate ticks (4
    total), rounded to whole observations since that's what they are."""
    raw = np.linspace(0, obs_max, 4)
    return sorted(set(int(round(v)) for v in raw))


def _draw_response_change_panel(ax, human_delta: pd.DataFrame, models: dict,
                                include_models: bool,
                                ylabel: str, obs_max: float) -> None:
    plot_delta_aggregate(ax, human_delta, HUMAN_COLOR, "hier_mean_median",
                         zorder_line=3, zorder_fill=1, errorbar_kind=None)
    if include_models:
        for i, model in enumerate(["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"]):
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


def make_response_change() -> Path:
    """1x4 panel (one per task: balls/carrabin, snacks/yoo, colors,
    numbers): median |delta response| vs observation, Human plus all 5
    models (Mean/LeakyIntegrator/PrimacyRecency/RL_lambda/NEF) overlaid --
    the "response change decay" figure.

    REMADE this session, matching make_model_performance's own update:
    NEF is now a genuine 5th model everywhere (TASK_PANELS' own roster),
    reading from the SAME fresh weekend RMSE fit (data/runs/rmse/) --
    see _delta_responses_path's own docstring for the two presentation-
    only fit-variant quirks (balls' MLE variant, snacks' refit/ folder)
    this drops, so "NEF" means the same fit in this figure as in
    make_model_performance.

    ONE combined figure now, not presentations/make_figures.py's own TWO
    separate SVGs (response_change_human.svg advancing to
    response_change_full.svg via an r-stack reveal.js fragment). That
    split existed purely to support a slide-build animation -- irrelevant
    for a static paper figure, where the human+models version is the one
    actually worth publishing. If a separate human-only panel turns out to
    be wanted for some other part of the paper, it's a small change (the
    include_models=False path _draw_response_change_panel already supports
    is kept, just not called as its own top-level figure here).

    Metric: median (across pids) of each pid's own mean |delta response|
    vs observation -- the established "response change" panel from
    figure_carrabin_temporal.py's panel B / figure_yoo_temporal.py's panel B
    / figure_soltani_temporal.py's col 2, via the shared
    utils.aggregate.plot_delta_aggregate (hier_mean_median). Per-task
    first-observation/minimum-observation conventions are copied from each
    source rather than reinvented -- see DELTA_MIN_OBS/FIRST_OBS_IS_RESPONSE.

    Y-AXIS SHARED ACROSS ALL FOUR PANELS (sharey=True via plt.subplots),
    starting at 0 (a delta magnitude is non-negative by construction).

    Legend: Human (gray) + the same 5 model colors/labels as
    make_model_performance, including RL_lambda's pretty math-formatted
    display text (MODEL_LABEL) instead of the raw "RL_lambda" string.

    Saved as response_change_decay.pdf (PDF only, per this project's usual
    convention -- unlike make_model_performance, this one wasn't asked to
    also save .svg).
    """
    _apply_slide_style()
    data = _load_response_change_data()

    obs_max_by_task = {}
    for task_key in data:
        human_delta, models, _ = data[task_key]
        obs_vals = [human_delta["observation"].max()] + [
            df["observation"].max() for df in models.values() if len(df)]
        obs_max_by_task[task_key] = max(obs_vals)

    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, sharey=True,
                             constrained_layout=True)
    for i, (task_key, title, models_list) in enumerate(TASK_PANELS):
        human_delta, models, _ = data[task_key]
        ax = axes[i]
        ylabel = "Median |\u0394response|" if i == 0 else ""
        _draw_response_change_panel(ax, human_delta, models,
                                    include_models=True, ylabel=ylabel,
                                    obs_max=obs_max_by_task[task_key])
        ax.set_title(title, color=TASK_COLORS[task_key])
        ax.tick_params(axis="y", labelleft=(i == 0))

    axes[0].set_ylim(bottom=0)  # shared -- applies to every panel

    legend_handles = [Line2D([0], [0], color=HUMAN_COLOR, lw=3, label="Human")]
    legend_handles += [Line2D([0], [0], color=MODEL_COLORS[m], lw=3, label=MODEL_LABEL.get(m, m))
                       for m in ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"]]
    fig.get_layout_engine().set(h_pad=0.25)
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=6,
               frameon=True, framealpha=0.9)

    out_path, _ = _save_fig(fig, "response_change_decay")
    plt.close(fig)
    return out_path



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
    cherry-picked for a clean-looking curve.

    X-axis relabeled "Observation" (was "Observations seen (n)") and
    extended to show an explicit 0 tick -- purely cosmetic, the underlying
    n/lambda fit is unchanged; the curve itself still only spans
    [n.min(), n.max()], so the axis now shows blank space between 0 and
    where the actual data starts rather than cropping right up to it.

    Y-axis label is back to plain "Mean |\u0394response|" -- participant/
    task identity moved into a LEGEND instead (lower right, where the
    panel has empty space), rather than living in the y-axis label. The
    legend has 3 rows: the fitted equation (tied to the actual fit-curve
    handle, so its color swatch matches the plotted line), "Participant
    #X", and the task name -- the task row is colored to match
    TASK_COLORS, the SAME color the panel titles elsewhere in this file
    use for their own task name. The last two rows use a blank (invisible)
    handle, since they're not identifying a second plotted series -- just
    metadata riding along in the legend box for convenient, easy-to-
    position stacked text.
    """
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
    fit_line, = ax.plot(n_smooth, _power_law(n_smooth, A_fit, lam_fit),
                        color=HUMAN_COLOR, lw=2.5, zorder=4)

    ax.set_xlabel("Observation")
    ax.set_ylabel("Mean |\u0394response|")
    ax.set_title("Fitting example", color="0.3")
    ax.set_xlim(left=0)
    ax.set_xticks(sorted(set([0.0] + list(ax.get_xticks()))))
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, top=True, right=True)

    blank = Line2D([], [], linestyle="none")
    leg = ax.legend(
        [fit_line, blank, blank],
        [r"$A n^{-\lambda}$, $\lambda=%.2f$" % lam_fit,
         f"Participant #{demo_pid}",
         TASK_LABELS[task_key]],
        loc="lower right", frameon=True, framealpha=0.9,
        handlelength=1.2, fontsize=11,
    )
    leg.get_texts()[2].set_color(TASK_COLORS[task_key])


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

    Y-AXIS LEGEND: none -- with only "Human" plotted (models were removed
    from this deck per instruction), a single-entry legend saying "Human"
    is redundant clutter, same reasoning as make_variability_human's own
    dropped legend; the panel titles already name each task. No reserved
    legend slot either, so there's no leftover empty margin at the bottom.
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

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path, _ = _save_fig(fig, "lambda_human")
    plt.close(fig)
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
    "RL_lambda": _RL_LAMBDA_PRETTY,
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


def _plot_lambda_splithalf_panel(ax, task_key: str, title: str,
                                 show_ylabel: bool) -> None:
    """Panels 1-3: odd-vs-even split-half reliability of fitted lambda,
    Human only -- matching figure_soltani_temporal.py's own panels E/K
    (scatter + regression line), reduced to a single source per instruction
    (models were tried in an earlier version of this figure and removed).
    Human is plain gray; task color lives on the panel title only, same
    convention as every other lambda/response-change figure in this deck.

    Panels 1-3 share BOTH axes at a fixed [0, 1.5] range (matches LAMBDA_XLIM)
    -- not autoscaled per task -- so the three tasks' reliability scatter is
    directly visually comparable rather than each panel silently rescaling to
    its own data range. show_ylabel controls whether this panel draws its own
    y-axis label/ticks (only the leftmost of the three needs to, since the
    scale is now identical across all three).

    LEGEND IS DRAWN INSIDE `ax` ITSELF, not a dedicated legend_ax row --
    with only one source, "r=0.xx*" is short enough to fit directly in the
    corner without needing a reserved row (the multi-source GridSpec+legend_
    ax approach this panel used before is no longer needed once there's
    only one line to label)."""
    path = _human_data_path(task_key)
    wide = _fit_lambda_split_half(task_key, path)

    ax.set_title(title, color=TASK_COLORS[task_key])
    ax.set_xlim(*LAMBDA_XLIM)
    ax.set_ylim(*LAMBDA_XLIM)
    if len(wide) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    sns.regplot(data=wide, x="odd", y="even", ax=ax, color=HUMAN_COLOR,
                ci=95 if len(wide) >= 3 else None, scatter=True,
                line_kws={"lw": 1.5}, scatter_kws={"s": 20, "alpha": 0.6})
    if len(wide) >= 3:
        r, p = pearsonr(wide["odd"], wide["even"])
        label = f"r={r:.2f}{pvalue_to_stars(p)}"
    else:
        label = f"n={len(wide)}"
    ax.legend(handles=[Line2D([0], [0], color=HUMAN_COLOR, lw=1.5, label=label)],
              fontsize=10, loc="upper right", frameon=True, framealpha=0.9)

    ax.set_xlabel("\u03bb (odd trials)")
    ax.set_ylabel("\u03bb (even trials)" if show_ylabel else "")
    ax.tick_params(axis="y", labelleft=show_ylabel)
    sns.despine(ax=ax, top=True, right=True)


def _plot_lambda_crosstask_panel(ax) -> None:
    """Panel 4: cross-task comparison of fitted lambda, colors vs numbers,
    one point per pid who did BOTH -- matching figure_soltani_temporal.py's
    own panel L (_plot_panel_lambda_crosstask), human-only by design in that
    source script (an individual-differences/trait-stability check, not a
    model-fit panel) -- and reduced to human-only here too, matching the
    single-figure simplification applied to panels 1-3.

    LEGEND IS DRAWN INSIDE `ax` ITSELF, same reasoning as
    _plot_lambda_splithalf_panel's own docstring."""
    lam_colors = _fit_lambda_series(
        _load_lambda_delta("colors", _human_data_path("colors")), LAMBDA_N_OFFSET["colors"])
    lam_numbers = _fit_lambda_series(
        _load_lambda_delta("numbers", _human_data_path("numbers")), LAMBDA_N_OFFSET["numbers"])
    merged = pd.DataFrame({"colors": lam_colors, "numbers": lam_numbers}).dropna()

    ax.set_title("Colors vs Numbers", color="0.3", fontsize=14)
    if len(merged) < 2:
        ax.text(0.5, 0.5, "No pids completed both tasks", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    ax.scatter(merged["colors"], merged["numbers"], color=HUMAN_COLOR, s=30,
              alpha=0.7, zorder=3)
    if len(merged) >= 3:
        sns.regplot(data=merged, x="colors", y="numbers", ax=ax, color=HUMAN_COLOR,
                   ci=95, scatter=False, line_kws={"lw": 1.5})
        r, p = pearsonr(merged["colors"], merged["numbers"])
        label = f"r={r:.2f}{pvalue_to_stars(p)}"
    else:
        label = f"n={len(merged)}"
    ax.legend(handles=[Line2D([0], [0], color=HUMAN_COLOR, lw=1.5, label=label)],
              fontsize=10, loc="upper right", frameon=True, framealpha=0.9)

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
    (matching that same script's panel L). Human only throughout -- a
    models version was tried and then removed per instruction, so this is
    now the only lambda-reliability figure/slide in the deck.

    PLAIN 1x4 plt.subplots -- NOT the 2-row GridSpec (plots + a dedicated
    legend row) this figure used before. With only one source per panel,
    each legend is now just "r=0.xx*" or "n=N", short enough to sit inside
    its own axes (see _plot_lambda_splithalf_panel/_plot_lambda_crosstask_
    panel's own docstrings) -- the reserved legend row is no longer needed
    and left a lot of empty space at the bottom once removed.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, constrained_layout=True)

    for i, (ax, (task_key, title)) in enumerate(zip(axes[:3], LAMBDA_TASK_PANELS)):
        _plot_lambda_splithalf_panel(ax, task_key, title, show_ylabel=(i == 0))
    _plot_lambda_crosstask_panel(axes[3])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path, _ = _save_fig(fig, "lambda_sanity_human")
    plt.close(fig)
    return out_path


def make_lambda_overview() -> Path:
    """2-row, 4-column figure trying make_lambda_human and
    make_lambda_sanity_human combined into one, rather than two separate
    1x4 figures:
      Row 1 (identical to make_lambda_human): Fitting-example demo,
        then Snacks/Colors/Numbers KDE distributions of fitted lambda.
      Row 2: Colors-vs-Numbers cross-task regression in col 1 (taking the
        slot row 1 uses for the demo panel, since col 1 has no task title
        to share), then Snacks/Colors/Numbers odd-even split-half
        reliability in cols 2-4.

    Columns 2-4 have NO title in row 2 -- removed entirely, per
    instruction, since task identity there will be conveyed by manually
    recoloring the row-2 axes afterward instead of by a repeated text
    title. Column 1's row-2 title ("Colors vs Numbers", normally drawn by
    _plot_lambda_crosstask_panel itself) is cleared the same way, since
    the instruction was ALL of row 2, not just the task-name columns --
    that function has no title parameter of its own, so it's cleared with
    an explicit ax.set_title("") right after the call rather than changing
    that shared function's signature (also used, unchanged, by
    make_lambda_sanity_human). Row 1's col-1 title is overridden to
    "Metric Definition" the same way (call _plot_lambda_demo, then
    ax.set_title(...) again) -- for consistency with make_sigma_overview's
    own panel A, WITHOUT touching _plot_lambda_demo itself, which
    make_lambda_human still relies on for its own real "Fitting example"
    title.

    Every panel reuses the EXACT SAME helper functions make_lambda_human/
    make_lambda_sanity_human already call (_plot_lambda_demo,
    _plot_lambda_distribution, _plot_lambda_splithalf_panel,
    _plot_lambda_crosstask_panel) -- no panel-drawing logic is duplicated
    here, only the layout differs. Figure height is 1.9x the shared 1-row
    FIGURE_SIZE, then shrunk to 75% of that (some squeezing accepted, per
    instruction) so two rows of real panels aren't as tall as an unscaled
    doubling would make them.

    An experiment, per instruction ("I want to try combining...") -- both
    source figures (make_lambda_human, make_lambda_sanity_human) are left
    completely unchanged and still produce their own separate outputs.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(2, 4, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 1.9 * 0.75),
                             constrained_layout=True)

    _plot_lambda_demo(axes[0, 0], task_key="numbers")
    axes[0, 0].set_title("Metric Definition", color="0.3")
    for i, (task_key, title) in enumerate(LAMBDA_TASK_PANELS):
        ax = axes[0, i + 1]
        human_delta = _load_lambda_delta(task_key, _human_data_path(task_key))
        lam = _fit_lambda_series(human_delta, LAMBDA_N_OFFSET[task_key])
        _plot_lambda_distribution(ax, lam, task_key)
        ax.set_title(title, color=TASK_COLORS[task_key])
        ax.set_ylabel("Normalized density" if i == 0 else "")
        ax.tick_params(axis="y", labelleft=(i == 0))

    _plot_lambda_crosstask_panel(axes[1, 0])
    axes[1, 0].set_title("")
    for i, (task_key, title) in enumerate(LAMBDA_TASK_PANELS):
        _plot_lambda_splithalf_panel(axes[1, i + 1], task_key, "", show_ylabel=(i == 0))

    out_path, _ = _save_fig(fig, "lambda_overview")
    plt.close(fig)
    return out_path


# ── How well does each model's own fitted lambda track a person's? ───────

# Model roster per task -- ALL THREE tasks now get the SAME 5 models
# (Mean/LeakyIntegrator/PrimacyRecency/RL_lambda/NEF), matching TASK_PANELS'
# own current roster. Previously this was a 4-model stand-in split (NEF
# only for snacks, RL_lambda only for colors/numbers) inherited from before
# real NEF RMSE fits existed for colors/numbers -- that's gone now that
# both models are real everywhere, so both get shown, every task.
LAMBDA_CORR_MODELS = {
    "snacks": ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"],
    "colors": ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"],
    "numbers": ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"],
}


def _plot_lambda_model_corr_panel(ax, task_key: str, title: str,
                                  show_ylabel: bool, legend_loc: str = "best") -> None:
    """One panel: EACH model's own fitted lambda (y) vs that SAME pid's
    fitted human lambda (x), for one task -- a genuinely different question
    from every other lambda panel in this deck (those ask "how is lambda
    distributed" or "is a pid's own lambda stable"; this one asks "does a
    model that fits pid X's OVERALL responses well also correctly track
    THAT PARTICULAR PERSON'S decay rate, relative to everyone else's"). A
    model whose points hug the y=x line, or at least trend upward with
    Human's own lambda, is capturing genuine individual differences in
    integration strategy -- not just an average curve shape.

    Both lambdas are fit the SAME way (identical _fit_lambda_series call,
    same LAMBDA_MIN_OBS/LAMBDA_N_OFFSET) -- Human's from _human_data_path,
    each model's from that model's own *_responses.pkl (_delta_responses_
    path, the same file every other model-lambda panel in this deck reads).
    Merged on pid (inner join) before fitting the regression, so only pids
    with a DEFINED fit for both sources contribute to that model's r.

    DASHED GREY DIAGONAL (y=x) marks perfect model-human alignment -- drawn
    first, so every regplot layers visibly on top of it, and deliberately
    kept OUT of the legend: it's never added to the explicit handles/
    labels list passed to ax.legend() below, so it can't appear there
    regardless of whether it has its own label.

    LEGEND NOW DRAWN INSIDE `ax` ITSELF (loc="best", matplotlib's own
    least-overlap heuristic -- picks whichever corner has the most actual
    empty space, per panel, rather than one fixed corner guessed by eye),
    not a separate legend_ax row below the panel -- the same
    simplification make_lambda_sanity_human's own splithalf/crosstask
    panels already made, applied here too.

    X/Y LIMITS AND TICKS ARE SET AFTER ALL PLOTTING, not before -- setting
    them first (this function's own earlier version) let each regplot's
    own autoscaling -- its confidence-interval band, in particular --
    silently re-expand the y-axis past the intended range: confirmed
    directly by rendering that earlier version, where y went up to
    ~1.4-1.5 with 8 ticks against the x-axis's own 4. Both axes now
    genuinely end at LAMBDA_XLIM (0, 1.5) with the SAME explicit ticks.
    """
    human_delta = _load_lambda_delta(task_key, _human_data_path(task_key))
    lam_human = _fit_lambda_series(human_delta, LAMBDA_N_OFFSET[task_key])

    ax.set_title(title, color=TASK_COLORS[task_key])
    ax.plot(LAMBDA_XLIM, LAMBDA_XLIM, linestyle="--", color="0.6", lw=1.2, zorder=1)

    handles, labels = [], []
    for model in LAMBDA_CORR_MODELS[task_key]:
        path = _delta_responses_path(task_key, model)
        if not path.exists():
            print(f"  (missing {path.name} -- skipping {model} for {task_key})")
            continue
        model_delta = _load_lambda_delta(task_key, path)
        lam_model = _fit_lambda_series(model_delta, LAMBDA_N_OFFSET[task_key])

        merged = pd.DataFrame({"human": lam_human, "model": lam_model}).dropna()
        if len(merged) < 2:
            continue
        color = MODEL_COLORS[model]
        disp = MODEL_DISPLAY.get(model, model)
        sns.regplot(data=merged, x="human", y="model", ax=ax, color=color,
                    ci=95 if len(merged) >= 3 else None, scatter=True,
                    line_kws={"lw": 1.5}, scatter_kws={"s": 20, "alpha": 0.6})
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        if len(merged) >= 3:
            r, p = pearsonr(merged["human"], merged["model"])
            labels.append(f"{disp} r={r:.2f}{pvalue_to_stars(p)}")
        else:
            labels.append(f"{disp} n={len(merged)}")

    if not handles:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    ticks = [0.0, 0.5, 1.0, 1.5]
    ax.set_xlim(*LAMBDA_XLIM)
    ax.set_ylim(*LAMBDA_XLIM)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=9,
              frameon=True, framealpha=0.9)
    ax.set_xlabel("\u03bb (human)")
    ax.set_ylabel("\u03bb (model)" if show_ylabel else "")
    ax.tick_params(axis="y", labelleft=show_ylabel)
    sns.despine(ax=ax, top=True, right=True)


def make_lambda_model_correlation() -> Path:
    """1x3 panel (snacks, colors, numbers -- balls excluded, no lambda fit
    exists for it, see LAMBDA_TASK_PANELS's own comment): for each task,
    each fitted model's own lambda (y) plotted against that SAME pid's
    human lambda (x), one regplot per model, sharing a fixed [0,1.5] x/y
    range with matching ticks (LAMBDA_XLIM) for direct comparability
    across panels, plus a dashed grey y=x reference line (perfect
    model-human alignment) excluded from the legend -- see
    _plot_lambda_model_corr_panel's own docstring for both. The point of
    this figure is to visualize how well each model captures each
    participant's OWN decay-rate/recency-bias, not just the population
    average -- see that function's own docstring.

    Only the leftmost panel draws its own y-axis label/ticks (per
    instruction) -- the other two share the identical LAMBDA_XLIM scale,
    so repeating the label added no information.

    PLAIN 1x3 plt.subplots -- NOT the 2-row GridSpec (plots + a dedicated
    legend row) this figure used before. Each panel's legend (up to 5
    "r=0.xx*" lines) is now drawn INSIDE the axes itself, matching the
    simplification make_lambda_sanity_human's own panels already made --
    the reserved legend row is no longer needed. loc="best" for
    colors/numbers (matplotlib's own least-overlap heuristic, which
    happens to land both in a lower-right-ish spot); snacks is pinned
    explicitly to "lower right" instead, since its own "best" pick landed
    at the top -- for visual consistency across all three panels, not
    because "best" was wrong for that panel's own data.

    Figure height is 60% of the shared FIGURE_SIZE height, per instruction
    (some squeezing accepted) -- width is unchanged.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 0.6),
                             constrained_layout=True)

    legend_locs = {"snacks": "lower right"}  # others default to "best"
    for i, (ax, (task_key, title)) in enumerate(zip(axes, LAMBDA_TASK_PANELS)):
        _plot_lambda_model_corr_panel(ax, task_key, title, show_ylabel=(i == 0),
                                      legend_loc=legend_locs.get(task_key, "best"))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path, _ = _save_fig(fig, "lambda_model_correlation")
    plt.close(fig)
    return out_path


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
    return RUNS_DIR / "rmse" / f"{model}_{dataset}_responses.pkl"


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


def _rasterize_svg(svg_path: Path, dpi: int = 300):
    """Rasterize an SVG (via inkscape) to an RGBA array, for embedding as a
    normal image ARTIST in a matplotlib Axes (ax.imshow) -- unlike the old
    _embed_svg_into_rect/_panel_a_rect approach (splicing raw SVG XML into
    a SAVED .svg file in place), this works identically for BOTH .pdf and
    .svg output, since it's baked into the figure itself before ANY
    fig.savefig() call, not a post-hoc edit of one specific saved format.
    It also needs no rect-computation the old approach did (matching a
    hand-computed pixel rect to the panel's own position): ax.imshow just
    fills whichever Axes it's called on, and that Axes is already
    positioned correctly by the figure's own layout.

    Returns None (caller should leave the panel blank, matching the old
    behaviour for a missing schematic) if the file doesn't exist or
    inkscape isn't available/fails -- a missing or unrasterizable schematic
    should never crash the whole figure.
    """
    import subprocess
    import tempfile

    if not svg_path.exists():
        print(f"  (missing {svg_path.name} -- panel A left blank)")
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        subprocess.run(
            ["inkscape", "--export-type=png", f"--export-dpi={dpi}",
             "-o", str(tmp_path), str(svg_path)],
            check=True, capture_output=True, timeout=30,
        )
        return plt.imread(tmp_path)
    except Exception as e:
        print(f"  (could not rasterize {svg_path.name}: {e} -- panel A left blank)")
        return None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


# Hand-made schematic explaining the response-variability metric itself
# (repeated identical trials -> spread of responses = "response noise") --
# goes in panel A of both variability figures, the same conceptual role
# _plot_lambda_demo's "Fitting example" panel plays for the lambda figures.
# Copied into FIGURES_DIR/schematics/ (from presentations/images/, where
# the original still lives for the slide deck's own separate use) so this
# script -- and everything it produces under figures/ -- is fully
# self-contained and doesn't reach into presentations/ at runtime.
VARIABILITY_SCHEMATIC = FIGURES_DIR / "schematics" / "response_noise_schematic.svg"


def make_variability_human() -> Path:
    """1x4 panel: panel A holds VARIABILITY_SCHEMATIC (a hand-made diagram of
    the metric itself -- repeated identical trials, response spread =
    "response noise"), rasterized and embedded via ax.imshow (see
    _rasterize_svg's own docstring for why this replaced the presentation
    script's SVG-XML-splicing approach: that could only ever produce a
    correct .svg, never a .pdf); panels B-D are Human-only KDEs of response
    variability for identical inputs, one per task (balls, colors, numbers
    -- snacks excluded, see VARIABILITY_TASK_PANELS), each autoscaled to
    its OWN data range (see _plot_variability_panel's own docstring for
    why the shared range this used to have was dropped).

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
    schematic = _rasterize_svg(VARIABILITY_SCHEMATIC)
    if schematic is not None:
        axes[0].imshow(schematic, aspect="auto")

    for i, (ax, (task_key, title)) in enumerate(zip(axes[1:], VARIABILITY_TASK_PANELS)):
        _plot_variability_panel(ax, task_key, title, include_models=False,
                                show_ylabel=(i == 0))

    out_path, _ = _save_fig(fig, "variability_human")
    plt.close(fig)
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
    schematic = _rasterize_svg(VARIABILITY_SCHEMATIC)
    if schematic is not None:
        axes[0].imshow(schematic, aspect="auto")

    for i, (ax, (task_key, title)) in enumerate(zip(axes[1:], VARIABILITY_TASK_PANELS)):
        _plot_variability_panel(ax, task_key, title, include_models=False,
                                show_ylabel=(i == 0))

    out_path, _ = _save_fig(fig, "variability_models")
    plt.close(fig)
    return out_path


# ── Consistent Across Trials & Tasks, for response noise (σ) ───────────────

# Axis-label text per task -- now uniform (odd/even) for all three tasks,
# since the split itself is uniform after this session's fix (see
# _fit_sigma_split_half's own docstring: balls used to use a median
# first/second-half split; changed to odd/even, matching colors/numbers).
# Kept as a per-task dict (rather than one shared tuple) purely so a
# future task-specific override stays easy to add, not because the text
# actually differs right now.
SIGMA_SPLIT_LABELS = {
    "balls": ("odd trials", "even trials"),
    "colors": ("odd trials", "even trials"),
    "numbers": ("odd trials", "even trials"),
}


def _fit_sigma_split_half(task_key: str, path: Path, qid_map: pd.DataFrame,
                          prefix: int | None) -> pd.DataFrame:
    """Per-pid response noise (sigma), computed separately on ODD vs EVEN
    trial-index halves -- UNIFORM across all three tasks now (balls,
    colors, numbers), matching _fit_lambda_split_half's own established
    odd/even convention, for the identical reason: interleaved sampling
    isolates genuine estimation noise from any systematic drift across a
    session (learning, fatigue, boredom), which a chronological/median
    split would confound with unreliability.

    CHANGED this session, per instruction: balls used to split by trial
    MEDIAN instead (matching figure_carrabin_variability.py's own panel C,
    `half_split_std`, with a min_trials=3 threshold per (observation, qid)
    cell) -- colors/numbers already used odd/even, and there was no
    remaining reason for balls to differ once that was noticed. No
    min_trials threshold survives this unification either, beyond
    pandas' own std() (needs n>=2) -- the SAME reasoning this function
    already established for colors specifically (a stricter threshold
    destroyed the sample there: verified directly, colors dropped from 46
    pids to just 14 under min_trials=3, versus 45/46 with no threshold),
    now applied uniformly rather than kept as a balls-only special case.

    Returns columns [pid, odd, even] for every task.
    """
    raw = pd.read_pickle(path)[["pid", "trial", "observation", "response"]]
    if prefix is not None:
        raw = raw[raw["observation"] < prefix]
    df = raw.merge(qid_map, on=["pid", "trial", "observation"])

    rows = []
    for pid, g in df.groupby("pid"):
        trials = sorted(g["trial"].unique())
        halves = {"odd": trials[0::2], "even": trials[1::2]}
        for half_label, tset in halves.items():
            gg = g[g["trial"].isin(tset)]
            pv = gg.groupby(["qid", "observation"])["response"].std().dropna()
            if len(pv) > 0:
                rows.append({"pid": pid, "half": half_label, "sigma": float(pv.mean())})

    if not rows:
        return pd.DataFrame(columns=["pid", "odd", "even"])
    wide = pd.DataFrame(rows).pivot(index="pid", columns="half", values="sigma").dropna()
    wide.columns.name = None
    return wide.reset_index()


def _plot_sigma_splithalf_panel(ax, task_key: str, title: str,
                                show_ylabel: bool) -> None:
    """Panels 1-3: odd-vs-even split-half reliability of response noise
    (sigma), Human only -- the SAME structure as
    _plot_lambda_splithalf_panel (mirrored per instruction), for the
    response-noise metric instead of fitted lambda. Tasks are
    VARIABILITY_TASK_PANELS (balls, colors, numbers) -- snacks excluded,
    same reasoning as the main response-noise figure
    (figure_yoo_temporal.py has no qid-repeat structure to compute sigma
    from at all, so there is no established methodology to reproduce).

    UNLIKE lambda's own splithalf panel, this one does NOT share one
    fixed x/y range across all three tasks -- sigma's natural scale
    varies far more across balls/colors/numbers than lambda's own bounded
    [0,1.5] fit range does (same per-panel-autoscale decision
    _plot_variability_panel's own docstring already made for this exact
    metric, and for the identical reason: a shared range left a lot of
    empty space in the narrower-scale tasks). Each panel's x and y axes DO
    share the same range as EACH OTHER, computed from that task's own
    odd+even values combined, so the diagonal comparison within one panel
    stays meaningful -- just not comparable in absolute terms ACROSS
    panels the way lambda's shared-range panels are.

    LEGEND IS DRAWN INSIDE `ax` ITSELF, matching the (now-simplified)
    lambda splithalf panel's own convention -- a single-source "r=0.xx*"
    fits directly in the corner with no dedicated legend row needed.
    """
    qid_map, prefix = _variability_qid_map(task_key)
    path = _human_data_path(task_key)
    wide = _fit_sigma_split_half(task_key, path, qid_map, prefix)

    ax.set_title(title, color=TASK_COLORS[task_key])
    if len(wide) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    hi = float(max(wide["odd"].max(), wide["even"].max())) * 1.1
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)

    sns.regplot(data=wide, x="odd", y="even", ax=ax, color=HUMAN_COLOR,
                ci=95 if len(wide) >= 3 else None, scatter=True,
                line_kws={"lw": 1.5}, scatter_kws={"s": 20, "alpha": 0.6})
    if len(wide) >= 3:
        r, p = pearsonr(wide["odd"], wide["even"])
        label = f"r={r:.2f}{pvalue_to_stars(p)}"
    else:
        label = f"n={len(wide)}"
    ax.legend(handles=[Line2D([0], [0], color=HUMAN_COLOR, lw=1.5, label=label)],
              fontsize=10, loc="upper right", frameon=True, framealpha=0.9)

    odd_label, even_label = SIGMA_SPLIT_LABELS[task_key]
    ax.set_xlabel(f"\u03c3 ({odd_label})")
    ax.set_ylabel(f"\u03c3 ({even_label})" if show_ylabel else "")
    ax.tick_params(axis="y", labelleft=show_ylabel)
    sns.despine(ax=ax, top=True, right=True)


def _plot_sigma_crosstask_panel(ax) -> None:
    """Panel 4: cross-task comparison of response noise (sigma), colors vs
    numbers, one point per pid who did BOTH -- same convention as
    _plot_lambda_crosstask_panel, for sigma instead of lambda. Balls/
    carrabin has NO overlapping population with colors/numbers at all (a
    completely separate study with different participants, not just a
    different task within the same pilot), so it cannot participate in a
    cross-task comparison the way it can in the split-half panels above --
    this panel is colors-vs-numbers only, matching the lambda figure's own
    equivalent panel exactly.

    LEGEND IS DRAWN INSIDE `ax` ITSELF, same reasoning as
    _plot_sigma_splithalf_panel's own docstring."""
    qid_map_c, prefix_c = _variability_qid_map("colors")
    qid_map_n, prefix_n = _variability_qid_map("numbers")
    sigma_colors = _variability_series("colors", _human_data_path("colors"), qid_map_c, prefix_c)
    sigma_numbers = _variability_series("numbers", _human_data_path("numbers"), qid_map_n, prefix_n)
    merged = pd.DataFrame({"colors": sigma_colors, "numbers": sigma_numbers}).dropna()

    ax.set_title("Colors vs Numbers", color="0.3", fontsize=14)
    if len(merged) < 2:
        ax.text(0.5, 0.5, "No pids completed both tasks", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    ax.scatter(merged["colors"], merged["numbers"], color=HUMAN_COLOR, s=30,
              alpha=0.7, zorder=3)
    if len(merged) >= 3:
        sns.regplot(data=merged, x="colors", y="numbers", ax=ax, color=HUMAN_COLOR,
                   ci=95, scatter=False, line_kws={"lw": 1.5})
        r, p = pearsonr(merged["colors"], merged["numbers"])
        label = f"r={r:.2f}{pvalue_to_stars(p)}"
    else:
        label = f"n={len(merged)}"
    ax.legend(handles=[Line2D([0], [0], color=HUMAN_COLOR, lw=1.5, label=label)],
              fontsize=10, loc="upper right", frameon=True, framealpha=0.9)

    ax.set_xlabel("\u03c3 (colors)")
    ax.set_ylabel("\u03c3 (numbers)")
    sns.despine(ax=ax, top=True, right=True)


def make_sigma_sanity_human() -> Path:
    """1x4 panel: mirrors make_lambda_sanity_human exactly, for response
    noise (sigma) instead of fitted lambda -- per instruction, using
    "sigma"/"\u03c3" as the shorthand throughout this figure's own labels.
    Panels 1-3 are odd/even split-half reliability of sigma for balls/
    colors/numbers (VARIABILITY_TASK_PANELS -- snacks excluded, same
    reasoning as the main response-noise figure). Panel 4 is the
    colors-vs-numbers cross-task comparison. Human only, matching the
    lambda figure's own human-only convention (a models version was tried
    for lambda and then removed per instruction; not attempted here at
    all for that same reason).

    UNLIKE lambda's own splithalf panels, sigma's panels 1-3 do NOT share
    one fixed range across all three tasks -- see
    _plot_sigma_splithalf_panel's own docstring for why (sigma's natural
    scale varies far more across tasks than lambda's bounded fit range
    does; each panel is autoscaled to its OWN combined odd+even data, same
    convention already established for the main response-noise figure).

    PLAIN 1x4 plt.subplots -- NOT a 2-row GridSpec -- matching
    make_lambda_sanity_human's own current (simplified) layout: a single-
    source "r=0.xx*"/"n=N" legend fits inside each panel with no reserved
    row needed.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, constrained_layout=True)

    for i, (ax, (task_key, title)) in enumerate(zip(axes[:3], VARIABILITY_TASK_PANELS)):
        _plot_sigma_splithalf_panel(ax, task_key, title, show_ylabel=(i == 0))
    _plot_sigma_crosstask_panel(axes[3])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path, _ = _save_fig(fig, "sigma_sanity_human")
    plt.close(fig)
    return out_path


def make_sigma_overview() -> Path:
    """2-row, 4-column figure trying make_variability_human and
    make_sigma_sanity_human combined into one, mirroring
    make_lambda_overview's own reshuffle-and-resize of make_lambda_human/
    make_lambda_sanity_human -- sigma's own true structural analogs of
    those two (same 4-panel shape each: one non-task panel + 3 per-task
    panels), NOT make_sigma_model_correlation (a 3-panel figure with no
    schematic/cross-task panel and a different task roster otherwise):
      Row 1 (identical to make_variability_human): schematic in col 1,
        then Balls/Colors/Numbers KDE distributions of response noise
        (sigma) for identical inputs.
      Row 2: Colors-vs-Numbers cross-task sigma regression in col 1 (the
        slot row 1 uses for the schematic), then Balls/Colors/Numbers
        odd-even split-half sigma reliability in cols 2-4.

    Both source figures already share the SAME 3-task roster
    (VARIABILITY_TASK_PANELS: balls, colors, numbers) for every one of
    their own panels -- unlike lambda's own family (which mixes
    LAMBDA_TASK_PANELS' snacks/colors/numbers with balls-only elsewhere)
    -- so, unlike make_lambda_overview, there's no task-set mismatch to
    reconcile between the two rows here.

    Row 2 has NO title at all -- neither the task-colored titles in cols
    2-4 nor col 1's own "Colors vs Numbers" -- matching
    make_lambda_overview's own FINAL state (task identity there gets
    added back by manually recoloring the row-2 axes afterward instead of
    a repeated text title). Row 1's own titles ("Metric Definition" in
    col 1, task-colored titles in cols 2-4) are untouched, matching row 1
    of make_lambda_overview too.

    Every panel reuses the EXACT SAME helper functions make_variability_
    human/make_sigma_sanity_human already call (_rasterize_svg,
    _plot_variability_panel, _plot_sigma_crosstask_panel,
    _plot_sigma_splithalf_panel) -- no panel-drawing logic is duplicated
    here, only the layout differs. Figure height uses the SAME scaling
    make_lambda_overview settled on (1.9x the shared 1-row FIGURE_SIZE,
    then shrunk to 75% of that), per instruction to resize "in the same
    way".

    An experiment, per instruction -- both source figures
    (make_variability_human, make_sigma_sanity_human) are left completely
    unchanged and still produce their own separate outputs.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(2, 4, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 1.9 * 0.75),
                             constrained_layout=True)

    axes[0, 0].axis("off")
    axes[0, 0].set_title("Metric Definition", color="0.3")
    schematic = _rasterize_svg(VARIABILITY_SCHEMATIC)
    if schematic is not None:
        axes[0, 0].imshow(schematic, aspect="auto")

    for i, (task_key, title) in enumerate(VARIABILITY_TASK_PANELS):
        ax = axes[0, i + 1]
        _plot_variability_panel(ax, task_key, title, include_models=False,
                                show_ylabel=(i == 0))

    _plot_sigma_crosstask_panel(axes[1, 0])
    axes[1, 0].set_title("")
    for i, (task_key, title) in enumerate(VARIABILITY_TASK_PANELS):
        _plot_sigma_splithalf_panel(axes[1, i + 1], task_key, "", show_ylabel=(i == 0))

    out_path, _ = _save_fig(fig, "sigma_overview")
    plt.close(fig)
    return out_path


# ── How well does each model's own fitted sigma track a person's? ──────

# Per-task model roster -- Mean/LeakyIntegrator/PrimacyRecency now included
# (per instruction), since their OWN "_resp_noise" NLL fit -- unlike their
# bare-name RMSE fit, which is exactly deterministic (sigma=0 for every
# pid) -- adds a genuine, per-pid-fitted sigma_resp term. Confirmed
# directly before wiring this in: all three vary meaningfully across pids
# in every task (e.g. numbers' Mean_resp_noise ranges 0.028-0.459 across
# 46 pids), not pinned at a shared floor. NEF (balls) / NoisyRL_lambda
# (colors/numbers) fill the 4th slot, matching SIGMA_CORR's own
# VARIABILITY_STOCHASTIC_MODEL choice elsewhere in this deck.
SIGMA_CORR_MODELS = {
    "balls": ["Mean", "LeakyIntegrator", "PrimacyRecency", "NEF"],
    "colors": ["Mean", "LeakyIntegrator", "PrimacyRecency", "NoisyRL_lambda"],
    "numbers": ["Mean", "LeakyIntegrator", "PrimacyRecency", "NoisyRL_lambda"],
}

# Fixed, SHARED x/y range across ALL THREE panels (not per-task anymore) --
# set explicitly per instruction, matching lambda_model_correlation's own
# shared LAMBDA_XLIM convention. With one shared 0.6 cap, nothing is
# actually cropped: confirmed directly, the max plotted (human + model)
# value in any task is 0.523 (colors) -- well under 0.6 -- so unlike the
# earlier per-task version of this constant, this range comfortably
# contains every point in every panel.
SIGMA_CORR_XLIM = 0.6

# Local color override for THIS figure only -- MODEL_COLORS itself is left
# untouched (NoisyRL_lambda's own tan still applies everywhere else, e.g.
# make_variability_models). Same reasoning as make_model_performance_nll's
# own NLL_MODEL_COLORS override, per instruction: NoisyRL_lambda plays
# RL_lambda's own conceptual role here (the "our model" 4th slot), so it
# takes RL_lambda's established red-orange rather than its usual tan.
SIGMA_CORR_COLORS = {m: MODEL_COLORS[m] for m in
                    ["Mean", "LeakyIntegrator", "PrimacyRecency", "NEF", "NoisyRL_lambda"]}
SIGMA_CORR_COLORS["NoisyRL_lambda"] = MODEL_COLORS["RL_lambda"]


def _sigma_model_source_path(task_key: str, model: str) -> Path:
    """Where to read one model's own response file for THIS figure.

    Mean/LeakyIntegrator/PrimacyRecency: their "_resp_noise" NLL fit, via
    _nll_responses_path -- which already adds that suffix internally and
    already routes balls to data/runs/carrabin/ vs colors/numbers to
    data/runs/nll/, so no extra branching is needed here. NOT their
    bare-name RMSE fit (exactly deterministic, sigma=0 for every pid).

    NEF (balls only): the MLE-fitted variant
    (NEF_carrabin_responses_mle.pkl, via _variability_model_path) --
    matching figure_carrabin_variability.py's own panels A/C convention.
    No NLL fit exists for NEF at all (see make_model_performance_nll's own
    module comment), so this is the only real option for balls' 4th slot.

    NoisyRL_lambda (colors/numbers only): the fresh NLL fit, also via
    _nll_responses_path -- deliberately NOT _variability_model_path's own
    path for these two tasks, which points at NoisyRL_lambda's OLDER,
    stale pre-quasi-MLE RMSE fit (see make_variability_models' own
    docstring for why that fit was never trustworthy).
    """
    if model == "NEF":
        return _variability_model_path(task_key, model)
    return _nll_responses_path(task_key, model)


def _plot_sigma_model_corr_panel(ax, task_key: str, title: str,
                                 show_ylabel: bool, legend_loc: str = "best") -> None:
    """One panel: EACH model's own response noise (sigma, y) vs that SAME
    pid's human sigma (x), for one task -- the sigma analogue of
    _plot_lambda_model_corr_panel, matching that function's own final
    structure (same session, same set of changes):

    DASHED GREY DIAGONAL (y=x) marks perfect model-human alignment -- drawn
    first, so every regplot layers visibly on top of it, and deliberately
    kept OUT of the legend: it's never added to the explicit handles/
    labels list passed to ax.legend() below.

    LEGEND NOW DRAWN INSIDE `ax` ITSELF (loc="best" by default, matplotlib's
    own least-overlap heuristic; a caller can override per panel via
    `legend_loc` for visual consistency, the same escape hatch
    make_lambda_model_correlation uses for its own snacks panel), not a
    separate legend_ax row below the panel -- the dedicated-legend-row
    GridSpec this panel used before is gone.

    AXIS LIMITS/TICKS ARE SET AFTER ALL PLOTTING, not before -- matching
    _plot_lambda_model_corr_panel's own fix (there, setting them first let
    a regplot's own CI-band autoscaling silently re-expand the range).
    Confirmed directly this particular bug wasn't actually manifesting
    here (SIGMA_CORR_XLIM's 0.6 cap already had enough margin over the
    real data), but the ordering is unified anyway, defensively, rather
    than leaving two near-identical panel functions with two different
    (one fragile) conventions for the same kind of plot.

    Model roster (SIGMA_CORR_MODELS) is UNCHANGED here -- unlike lambda's
    own model-correlation figure, this one was never a "stand-in because a
    model hadn't been fit yet" situation: NEF (balls) / NoisyRL_lambda
    (colors/numbers) are deliberately the ONE model per task with a
    genuine, non-deterministic sigma term (see SIGMA_CORR_MODELS' own
    comment), so forcing every task onto an identical 5-model list the way
    lambda's fix did would put models with EXACTLY ZERO sigma (RL_lambda's
    own bare RMSE fit, for instance) into a sigma-correlation panel, which
    isn't a meaningful comparison to draw.
    """
    qid_map, prefix = _variability_qid_map(task_key)
    human_sigma = _variability_series(task_key, _human_data_path(task_key), qid_map, prefix)

    ax.set_title(title, color=TASK_COLORS[task_key])
    ax.plot([0, SIGMA_CORR_XLIM], [0, SIGMA_CORR_XLIM], linestyle="--",
            color="0.6", lw=1.2, zorder=1)

    handles, labels = [], []
    plotted = []  # (model, merged) pairs
    for model in SIGMA_CORR_MODELS[task_key]:
        path = _sigma_model_source_path(task_key, model)
        if not path.exists():
            print(f"  (missing {path.name} -- skipping {model} for {task_key})")
            continue
        model_sigma = _variability_series(task_key, path, qid_map, prefix)
        merged = pd.DataFrame({"human": human_sigma, "model": model_sigma}).dropna()
        if len(merged) < 2:
            continue
        plotted.append((model, merged))

    if not plotted:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    for model, merged in plotted:
        color = SIGMA_CORR_COLORS[model]
        disp = MODEL_DISPLAY.get(model, model)
        sns.regplot(data=merged, x="human", y="model", ax=ax, color=color,
                    ci=95 if len(merged) >= 3 else None, scatter=True,
                    line_kws={"lw": 1.5}, scatter_kws={"s": 20, "alpha": 0.6})
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        if len(merged) >= 3:
            r, p = pearsonr(merged["human"], merged["model"])
            labels.append(f"{disp} r={r:.2f}{pvalue_to_stars(p)}")
        else:
            labels.append(f"{disp} n={len(merged)}")

    ticks = [0.0, 0.2, 0.4, 0.6]
    ax.set_xlim(0, SIGMA_CORR_XLIM)
    ax.set_ylim(0, SIGMA_CORR_XLIM)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=9,
              frameon=True, framealpha=0.9)
    ax.set_xlabel("\u03c3 (human)")
    ax.set_ylabel("\u03c3 (model)" if show_ylabel else "")
    ax.tick_params(axis="y", labelleft=show_ylabel)
    sns.despine(ax=ax, top=True, right=True)


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


def make_sigma_model_correlation() -> Path:
    """1x3 panel (balls, colors, numbers -- VARIABILITY_TASK_PANELS; snacks
    excluded, same reasoning as every other sigma figure in this deck, no
    qid-repeat structure to compute sigma from at all): for each task, up
    to 4 models' own response noise plotted against that SAME pid's human
    response noise (SIGMA_CORR_MODELS) -- the sigma analogue of
    make_lambda_model_correlation, matching that figure's own final
    structure (dashed y=x reference line excluded from the legend,
    in-panel legend rather than a dedicated row, 60%-height figure) -- see
    _plot_sigma_model_corr_panel's own docstring for exactly what changed
    and why the model roster itself was deliberately left alone.

    Y-AXIS (AND X-AXIS) IS SHARED ACROSS ALL THREE PANELS, fixed at
    [0, 0.6] with matching ticks (SIGMA_CORR_XLIM), matching
    make_lambda_model_correlation's own shared-range convention. Only the
    leftmost panel draws its own y-axis label/ticks.

    PLAIN 1x3 plt.subplots -- NOT the 2-row GridSpec (plots + a dedicated
    legend row) this figure used before. loc="best" per panel by default;
    override individual panels via `legend_locs` below if any one panel's
    own "best" pick ends up visually inconsistent with the others (same
    escape hatch make_lambda_model_correlation uses for its own snacks
    panel).

    Figure height is 60% of the shared FIGURE_SIZE height, per instruction
    (some squeezing accepted) -- width is unchanged.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 0.6),
                             constrained_layout=True)

    legend_locs = {}  # no per-task override needed yet -- see docstring
    for i, (ax, (task_key, title)) in enumerate(zip(axes, VARIABILITY_TASK_PANELS)):
        _plot_sigma_model_corr_panel(ax, task_key, title, show_ylabel=(i == 0),
                                     legend_loc=legend_locs.get(task_key, "best"))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path, _ = _save_fig(fig, "sigma_model_correlation")
    plt.close(fig)
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
    return RUNS_DIR / "nll" / f"{file_model}_{dataset}_nll_performance.pkl"


# NEW this session: a genuinely UNIFORM roster/location, replacing the
# split above for THIS figure specifically. All 4 tasks were refit with
# all 4 "_resp_noise" models (Mean/LeakyIntegrator/PrimacyRecency/
# RL_lambda) landing in ONE common folder, data/runs/nll/ -- balls/snacks
# used to live in their own carrabin/yoo folders, and colors/numbers used
# NoisyRL_lambda (a DIFFERENT, native state-noise mechanism) as their 4th
# model instead of RL_lambda's own "_resp_noise" fit, which didn't exist
# for them until this session's refit. RL_lambda_resp_noise now exists for
# every task, so it plays this figure's "our model" role directly --
# MODEL_COLORS/MODEL_LABEL already give it a proper color and pretty
# "RL-lambda" formatting, so no NLL_MODEL_COLORS/NLL_LABELS-style override
# is needed here the way NoisyRL_lambda required one.
#
# Deliberately NOT touching NLL_MODEL_ORDER/NLL_REFERENCE/
# NLL_MODEL_COLORS/NLL_LABELS/_nll_perf_path above -- those still serve
# make_variance_autocorr_human/models exactly as before, unchanged by this
# session's refit, until/unless that figure is updated too.
NLL_RESP_NOISE_MODELS = ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda"]
NLL_RESP_NOISE_REFERENCE = "RL_lambda"


def _nll_resp_noise_perf_path(task_key: str, model: str) -> Path:
    """Path to one (task, model)'s *_nll_performance.pkl under the unified
    data/runs/nll/ folder -- see NLL_RESP_NOISE_MODELS' own comment for
    why this exists alongside (not instead of) _nll_perf_path."""
    dataset = {"balls": "carrabin", "snacks": "yoo",
               "colors": "soltani_colors", "numbers": "soltani_numbers"}[task_key]
    return RUNS_DIR / "nll" / f"{model}_resp_noise_{dataset}_nll_performance.pkl"


def make_model_performance_nll() -> Path:
    """1x4 panel: model fit under the NLL/quasi-MLE metric, one panel per
    task (balls, snacks, colors, numbers), all four now reading from the
    SAME data/runs/nll/ folder with the SAME 4-model roster
    (NLL_RESP_NOISE_MODELS: Mean/LeakyIntegrator/PrimacyRecency/RL_lambda,
    every one of them the "_resp_noise" variant) -- see that constant's
    own comment for what changed and why (this session's fresh refit of
    all 4 tasks x 4 models into one folder, replacing the old carrabin/
    yoo/nll three-way split and the NoisyRL_lambda stand-in for colors/
    numbers). NEF is not included -- no NLL fit exists for it yet; add it
    to NLL_RESP_NOISE_MODELS once one does, no other change needed here.

    SIGNIFICANCE BARS: RL_lambda vs the 3 math models (Mean/
    LeakyIntegrator/PrimacyRecency), drawn in RL_lambda's OWN color
    (MODEL_COLORS["RL_lambda"]) -- matching make_model_performance's own
    convention exactly (that figure's docstring explains why: visually
    unambiguous which model the comparison is FROM). No exclude= needed
    here (unlike make_model_performance's own NEF exclusion) since NEF
    isn't in this roster at all yet.

    Y-AXIS IS SHARED (sharey=True) but NOT forced to start at 0 -- unlike
    make_model_performance's RMSE (non-negative by construction), NLL can
    be, and often is, negative, so clamping the bottom would misrepresent
    the actual data.

    Legend uses plain MODEL_COLORS/MODEL_LABEL directly (RL_lambda's own
    pretty "RL-lambda" formatting) -- no NLL_LABELS/NLL_MODEL_COLORS-style
    override needed, since RL_lambda is playing its own real role here,
    not standing in for a differently-colored model.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, sharey=True,
                             constrained_layout=True)

    panel_data = []
    for i, (ax, (task_key, title)) in enumerate(zip(axes, NLL_TASK_PANELS)):
        rows = []
        for model in NLL_RESP_NOISE_MODELS:
            path = _nll_resp_noise_perf_path(task_key, model)
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
        order = [m for m in NLL_RESP_NOISE_MODELS if m in plot_df["model"].unique()]
        pal = {m: MODEL_COLORS[m] for m in order}

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
        sig_lines = (_compute_sig_lines(plot_df, "model", "nll", order,
                                        NLL_RESP_NOISE_REFERENCE)
                    if NLL_RESP_NOISE_REFERENCE in order else [])
        per_panel_sig_lines.append((ax, sig_lines))
        max_bars = max(max_bars, len(sig_lines))

    if max_bars:
        axes[0].set_ylim(top=y_hi + dy_step * 0.5 + max_bars * dy_step * 2.0 + dy_step)

    sig_color = MODEL_COLORS["RL_lambda"]
    for ax, sig_lines in per_panel_sig_lines:
        y_current = y_hi + dy_step * 0.5
        for x1, x2, stars in sig_lines:
            draw_sig_line(ax, x1, x2, y_current, stars, color=sig_color)
            y_current += dy_step * 2.0

    legend_handles = [Patch(facecolor=MODEL_COLORS[m], label=MODEL_LABEL.get(m, m))
                      for m in NLL_RESP_NOISE_MODELS]
    fig.get_layout_engine().set(h_pad=0.25)
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=5,
               frameon=True, framealpha=0.9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path, _ = _save_fig(fig, "model_performance_nll")
    plt.close(fig)
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
# response-noise figures. Copied into FIGURES_DIR/schematics/ too, same
# reasoning as VARIABILITY_SCHEMATIC's own comment above.
AUTOCORR_SCHEMATIC = FIGURES_DIR / "schematics" / "autocorr_schematic.svg"


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
    return RUNS_DIR / "nll" / f"{file_model}_{dataset}_nll_responses.pkl"


def _nll_resp_noise_responses_path(task_key: str, model: str) -> Path:
    """Responses counterpart of _nll_resp_noise_perf_path -- see that
    function's own comment for why this exists alongside (not instead of)
    _nll_responses_path."""
    dataset = {"balls": "carrabin", "snacks": "yoo",
               "colors": "soltani_colors", "numbers": "soltani_numbers"}[task_key]
    return RUNS_DIR / "nll" / f"{model}_resp_noise_{dataset}_nll_responses.pkl"


def _load_variance_autocorr_data(models: list[str] | None = None,
                                 responses_path_fn=None) -> dict:
    """task_key -> (human_res, model_results, lags). Loaded ONCE and shared
    between the human-only and human+models figure functions below -- same
    established pattern as _load_response_change_data -- so both read the
    EXACT same underlying numbers.

    `models`/`responses_path_fn` let a caller override the roster/file
    location (added for make_sigma_giant, which uses NLL_RESP_NOISE_MODELS/
    _nll_resp_noise_responses_path instead) -- both default to None,
    resolving to NLL_MODEL_ORDER/_nll_responses_path, so
    make_variance_autocorr_human/models (which call this with no
    arguments) are completely unaffected.
    """
    models = models if models is not None else NLL_MODEL_ORDER
    responses_path_fn = responses_path_fn or _nll_responses_path
    out = {}
    for task_key, title in RESID_TASK_PANELS:
        qid_map, prefix = _variability_qid_map(task_key)
        human_path = _human_data_path(task_key)
        human_resid = _resid_frame(task_key, human_path, qid_map, prefix)
        lags = RESID_LAGS[task_key]
        human_res = _resid_autocorr(human_resid, lags)

        model_results = {}
        for m in models:
            mpath = responses_path_fn(task_key, m)
            if not mpath.exists():
                print(f"  (missing {mpath.name} -- skipping {m} for {task_key})")
                continue
            mresid = _resid_frame(task_key, mpath, qid_map, prefix)
            mres = _resid_autocorr(mresid, lags)
            if not isinstance(mres, str):
                model_results[m] = mres

        out[task_key] = (human_res, model_results, lags)
    return out


def _draw_variance_autocorr_panel(ax_ac, task_key: str, title: str, human_res,
                                  model_results: dict, lags: list[int],
                                  include_models: bool, show_ylabel: bool,
                                  models: list[str] | None = None,
                                  model_colors: dict | None = None) -> None:
    """Draws ONE autocorrelation panel -- the SAME function, called
    identically by both make_variance_autocorr_human and
    make_variance_autocorr_models, so the two figures differ ONLY in
    include_models (whether model curves get drawn) and in the
    figure-level legend each caller builds afterward -- per instruction.
    Human is always drawn; models only when include_models=True.

    `models`/`model_colors` default to None, resolving to NLL_MODEL_ORDER/
    NLL_MODEL_COLORS -- same override mechanism as
    _load_variance_autocorr_data, added for make_sigma_giant; the two
    standalone callers pass neither, so their behavior is unchanged.
    """
    models = models if models is not None else NLL_MODEL_ORDER
    model_colors = model_colors or NLL_MODEL_COLORS
    ax_ac.set_title(title, color=TASK_COLORS[task_key])
    ax_ac.axhline(0, color="0.7", lw=0.8, ls="--", zorder=1)
    if isinstance(human_res, str):
        msg = ("Insufficient data\n(no qid repeats for this task)"
              if human_res == "no_repeats" else "Insufficient data")
        ax_ac.text(0.5, 0.5, msg, ha="center", va="center",
                  transform=ax_ac.transAxes, color="0.5", style="italic")
    else:
        _, means, sems = human_res
        ax_ac.plot(lags, means, "-", color=HUMAN_COLOR, lw=2.2, zorder=6)
        ax_ac.fill_between(lags, means - sems, means + sems, color=HUMAN_COLOR,
                          alpha=0.2, zorder=1)
    if include_models:
        for j, m in enumerate(models):
            if m not in model_results:
                continue
            _, mmeans, msems = model_results[m]
            color = model_colors[m]
            ax_ac.plot(lags, mmeans, "-", color=color, lw=2.0, zorder=5 - j)
            ax_ac.fill_between(lags, mmeans - msems, mmeans + msems, color=color,
                              alpha=0.15, zorder=1)
    ax_ac.set_xlabel("Lag (observations)")
    ax_ac.set_ylabel("Autocorrelation" if show_ylabel else "")
    ax_ac.tick_params(axis="y", labelleft=show_ylabel)
    ax_ac.set_xticks(lags)
    ax_ac.margins(x=0.15)
    sns.despine(ax=ax_ac, top=True, right=True)


def make_variance_autocorr_human() -> Path:
    """1x4 panel: panel A holds AUTOCORR_SCHEMATIC (a hand-made diagram of
    the metric itself), rasterized and embedded via ax.imshow (see
    _rasterize_svg's own docstring for why this replaced the presentation
    script's SVG-XML-splicing approach) -- same convention as
    make_variability_human's own panel A; panels B-D are Human-only
    within-trial lag-k residual autocorrelation, one per task [balls,
    colors, numbers] -- snacks excluded, same reasoning as
    VARIABILITY_TASK_PANELS.

    IDENTICAL TO make_variance_autocorr_models EXCEPT FOR THE ADDED MODEL
    DATA AND LEGEND, per instruction -- both call the exact same
    _draw_variance_autocorr_panel for every panel, with include_models the
    only thing that differs (False here). Y-AXIS RANGE IS ALSO SHARED
    ACROSS BOTH FIGURES, not just within this one's own 4 panels: a
    throwaway PROBE pass (drawn with include_models=True on a scratch
    figure, never saved) establishes the SAME y-limits
    make_variance_autocorr_models' own real run would produce, then that
    range is applied here explicitly -- mirroring make_response_change's
    own two-pass shared-ylim mechanism, adapted for two independent
    top-level functions (callable in either order) rather than one
    combined function returning both paths.

    VARIANCE GROWTH (an earlier top row) WAS DROPPED, per instruction,
    after checking the actual numbers directly rather than relying on the
    earlier visual read (which was wrong -- see chat): only Human and
    NoisyRL_lambda show a genuine, substantial DECAYING autocorrelation
    (starting well above zero, decaying toward/past it); every
    "_resp_noise" model stays within about +-0.09 of zero at EVERY lag in
    EVERY task -- noise scatter around zero, not a real signal.
    Autocorrelation alone is the metric that actually distinguishes
    state-persistent noise from pure i.i.d. response noise; variance
    growth did not.
    """
    _apply_slide_style()
    data = _load_variance_autocorr_data()

    fig_probe, axes_probe = plt.subplots(1, 4, figsize=FIGURE_SIZE, sharey=True,
                                         constrained_layout=True)
    for i, (ax_ac, (task_key, title)) in enumerate(zip(axes_probe[1:], RESID_TASK_PANELS)):
        human_res, model_results, lags = data[task_key]
        _draw_variance_autocorr_panel(ax_ac, task_key, title, human_res,
                                      model_results, lags, include_models=True,
                                      show_ylabel=(i == 0))
    shared_ylim = axes_probe[0].get_ylim()
    plt.close(fig_probe)

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
                                      model_results, lags, include_models=False,
                                      show_ylabel=(i == 0))
    axes[0].set_ylim(*shared_ylim)  # identical to the models figure, not autoscaled

    # Same legend SLOT reserved as make_variance_autocorr_models (same
    # h_pad, same "outside lower center" placement) -- so the figure
    # doesn't resize/shift when models get added on the follow-up slide;
    # only "Human" is actually shown yet, matching make_lambda_human's own
    # human-only stage convention exactly.
    fig.get_layout_engine().set(h_pad=0.25)
    fig.legend(handles=[Line2D([0], [0], color=HUMAN_COLOR, lw=2.2, label="Human")],
               loc="outside lower center", ncol=1, frameon=True, framealpha=0.9)

    out_path, _ = _save_fig(fig, "variance_autocorr_human")
    plt.close(fig)
    return out_path


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


# ── Neural giant figure (Acts 1-3; neural_experiments.py's own outputs) ─────
# Task: soltani_numbers throughout -- the one task with BOTH a real sigma
# fit and a real lambda fit, per instruction, so it alone can carry the
# whole Acts-1-3 narrative rather than splitting it across two tasks the
# way carrabin/yoo's own old neural figures did.

NEURAL_EXP_DIR = RUNS_DIR / "neural_experiments"
NEURAL_READOUT_OFFSET = 0.5  # seconds into the observation window -- matches
                             # neural_experiments.py's own READOUT_OFFSET.


def _fold_observation_time(t: np.ndarray, t_iti: float, t_obs: float):
    """Split raw absolute simulation time into (observation_number
    [1-indexed], t_within_obs [seconds since THAT observation's own onset,
    NaN during the ITI]). neural_experiments.py's sweep/probe simulations
    run several observations back-to-back in ONE continuous trial (unlike
    the old extras_carrabin.py convention of one observation per simulated
    trial), so this folding has to happen here at plot time rather than
    already being built into the saved 't' column.
    """
    t_step = t_obs + t_iti
    observation_number = np.floor(t / t_step).astype(int) + 1
    t_in_step = t - (observation_number - 1) * t_step
    t_within_obs = np.where(t_in_step >= t_iti, t_in_step - t_iti, np.nan)
    return observation_number, t_within_obs


def _plot_neural_raster_demo(ax) -> None:
    """Panel 1 (Act 1.1): spike raster of the error population's raw neuron
    output for one representative trial (neural_experiments.py's
    raster_demo experiment), with the decoded PE trace overlaid on a twin
    axis.

    Uses sample_by_variance + cluster DIRECTLY (not the preprocess_spikes
    convenience wrapper) -- confirmed by checking check_NEF_pipeline.py and
    its archived predecessor, the only other NEF-dynamics spike-raster code
    in this repo, that neither actually does anything more than call
    preprocess_spikes(t, arr, num=50) as-is. That wrapper's own default
    sample_size=200 exceeds our n_neurons=100, so sample_by_variance's
    'select the highest-variance (truly active) neurons' step was a no-op
    (nothing to filter out of only 100 available), and its final merge
    step then block-averages neurons into synthetic composites -- fine
    when sampling genuinely thins a large pool, but here it just blurred
    real individual spike trains together without ever having filtered
    anything. Calling sample_by_variance with num=50 (well under 100) and
    skipping merge keeps real, individual, genuinely-active neurons.

    X-axis zoomed to the first 5 observations (10s of the full 30s trial),
    per instruction. Raster's own y-axis has no text label (per
    instruction, neuron index isn't inherently meaningful to a general
    reader) and its ticks are moved to the RIGHT side, since the decoded
    PE axis (the more informative one) takes the LEFT side instead --
    physical spines are unaffected by this (sns.despine's own left/right
    already matched this after the swap: ax2's default top+right removal
    keeps its left spine where its now-left ticks sit; ax's explicit
    right=False keeps its right spine where its now-right ticks sit).
    Decoded-PE line stays the palette green, but its axis label/ticks no
    longer use that color (per instruction -- color removed from the
    label specifically, not the line).
    """
    path = NEURAL_EXP_DIR / "raster_demo_soltani_numbers.pkl"
    if not path.exists():
        ax.text(0.5, 0.5, "No raster demo data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    d = pd.read_pickle(path)
    t_active, spikes_active = sample_by_variance(d["t"], d["error_neurons"],
                                                 num=50, filter_width=0.02)
    t_sorted, spikes_sorted = cluster(t_active, spikes_active, filter_width=0.002)
    plot_spikes(t_sorted, spikes_sorted, ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, 5 * 2.0)  # 5 observations x (t_obs=1.5 + t_iti=0.5)

    pe_color = get_palette(6)[2]  # palette green -- kept on the LINE only
    ax2 = ax.twinx()
    ax2.plot(d["t"], d["pe_product"], color=pe_color, lw=1.0)
    ax2.set_ylabel("Decoded Prediction Error")
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()
    ax2.set_ylim(0.0, 0.8)
    # ax's own tick-right must be set AFTER twinx() -- twinx() resets it
    # back to the left otherwise (confirmed directly by rendering: setting
    # this before twinx() left both axes' tick numbers stacked on the
    # left, overlapping).
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_yticks([])  # no explicit neuron count needed, per instruction
    ax.set_ylim(0, 50)  # raster fills the full panel height (50 neurons)
    sns.despine(ax=ax2, top=True)
    sns.despine(ax=ax, top=True, right=False)


def _plot_neural_lambda_activity(ax) -> None:
    """Panel 2 (Act 1.2): raw error-neuron activity vs observation-within-
    trial, one line per arbitrary lambda_ value (neural_experiments.py's
    sweep experiment, sweep_param='lambda_'). Style matches the reference
    lambda_drives_discounting figure's own leftmost panel, generalized from
    a 2-group (high/low median split of real fitted lambdas) comparison to
    N explicit, arbitrary swept values -- there's no real per-pid lambda
    here at all, by design (see chat).
    """
    path = NEURAL_EXP_DIR / "sweep_soltani_numbers_lambda_.pkl"
    if not path.exists():
        ax.text(0.5, 0.5, "No lambda sweep data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    from fitting.model_params import _NEF_FIXED

    d = pd.read_pickle(path)
    df = d["df"].copy()
    t_iti, t_obs = _NEF_FIXED["t_iti"], _NEF_FIXED["t_obs"]
    obs_num, t_within = _fold_observation_time(df["t"].values, t_iti, t_obs)
    df["observation"] = obs_num
    df["t_within_obs"] = t_within
    active = df[~np.isnan(df["t_within_obs"])]

    # Mean activity within each observation's own active window, per
    # (sweep_value, seed, observation) -- then averaged across seeds.
    per_obs = (active.groupby(["sweep_value", "seed", "observation"])["mean_error_activity"]
              .mean().reset_index())
    stats = (per_obs.groupby(["sweep_value", "observation"])["mean_error_activity"]
            .agg(["mean", "sem"]).reset_index())

    pal = get_palette(6)
    for i, val in enumerate(sorted(stats["sweep_value"].unique())):
        sub = stats[stats["sweep_value"] == val].sort_values("observation")
        ax.plot(sub["observation"], sub["mean"], color=pal[i], lw=1.8,
                label=f"\u03bb={val:g}")
        ax.fill_between(sub["observation"], sub["mean"] - sub["sem"],
                        sub["mean"] + sub["sem"], color=pal[i], alpha=0.18)

    ax.set_xlabel("Observation")
    ax.set_xlim(0, 15)
    ax.set_xticks(range(0, 16, 5))
    ax.set_ylabel("Error neuron activity (Hz)")
    ax.set_ylim(62, 82)
    ax.set_yticks(range(62, 83, 2))
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper right")
    sns.despine(ax=ax, top=True, right=True)


def _plot_neural_pe_dynamics(ax, show_markers: bool = False) -> None:
    """Panel 2 (Act 1.3): decoded PE vs time-within-observation, for the
    cross product of arbitrary alpha_0 x n_neurons values (matching the
    original reference PE_dynamics figure's own two-parameter convention
    -- reverted from a single-parameter sweep after reflection; see chat).
    Reads neural_experiments.py's sweep experiment run with BOTH
    --sweep_param2/--sweep_values2 set (sweep_soltani_numbers_alpha_0_
    n_neurons.pkl).

    show_markers=False (the default, per instruction) hides the dashed
    "PE/Response measured at" vertical lines and their labels entirely --
    set True to bring them back (matching the reference figure's own
    convention).

    Uses ONLY the first observation window, per instruction -- not
    averaged across all 15 -- for a clean single-transient read, matching
    the reference figure's own one-observation-per-trial convention.

    Reads sweep_param/sweep_param2 from the saved file's own metadata
    (rather than hardcoding "alpha_0"/"n_neurons" here) so this still works
    unchanged if the two swept parameters are ever reassigned.
    """
    path = NEURAL_EXP_DIR / "sweep_soltani_numbers_alpha_0_n_neurons.pkl"
    if not path.exists():
        ax.text(0.5, 0.5, "No alpha_0 x n_neurons sweep data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    from fitting.model_params import _NEF_FIXED

    d = pd.read_pickle(path)
    df = d["df"].copy()
    p1, p2 = d["sweep_param"], d["sweep_param2"]
    t_iti, t_obs = _NEF_FIXED["t_iti"], _NEF_FIXED["t_obs"]
    obs_num, t_within = _fold_observation_time(df["t"].values, t_iti, t_obs)
    df["observation"] = obs_num
    df["t_within_obs"] = t_within
    first_obs = df[(df["observation"] == 1) & (~np.isnan(df["t_within_obs"]))].copy()

    # Downsample for a cleaner line: dt=0.001s -> every 5ms.
    first_obs["t_bin"] = (first_obs["t_within_obs"] * 200).round() / 200
    stats = (first_obs.groupby(["sweep_value", "sweep_value2", "t_bin"])["pe_product"]
            .agg(["mean", "sem"]).reset_index())

    combos = sorted({(row.sweep_value, row.sweep_value2)
                     for row in stats.itertuples()})
    pal = get_palette(max(6, len(combos)))
    label_sym = {"alpha_0": "\u03b1\u2080", "n_neurons": "n", "lambda_": "\u03bb"}
    for i, (v1, v2) in enumerate(combos):
        sub = stats[(stats["sweep_value"] == v1) & (stats["sweep_value2"] == v2)].sort_values("t_bin")
        label = f"{label_sym[p1]}={v1:g}, {label_sym[p2]}={v2:g}"
        ax.plot(sub["t_bin"], sub["mean"], color=pal[i], lw=1.8, label=label)
        ax.fill_between(sub["t_bin"], sub["mean"] - sub["sem"],
                        sub["mean"] + sub["sem"], color=pal[i], alpha=0.18)

    if show_markers:
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for x, lbl in [(NEURAL_READOUT_OFFSET, "PE\nmeasured at"), (t_obs, "Response\nmeasured at")]:
            ax.axvline(x, color="0.4", lw=1.0, ls="--", zorder=0)
            ax.text(x, 1.02, lbl, transform=trans, ha="center", va="bottom",
                    clip_on=False, fontsize=7, color="0.4")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decoded Prediction Error")
    ax.set_xlim(0, t_obs + 0.05)
    ax.set_ylim(0.0, 0.3)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, ncol=1, loc="upper right")
    sns.despine(ax=ax, top=True, right=True)


NEURAL_ENCODER_THRESHOLD = 0.5  # matches figure_yoo_neural.py's own ENCODER_THRESHOLD


def _neural_weight_on_cols(pid_enc: pd.DataFrame, neuron_cols: list[str]) -> list[str]:
    """Which of the error ensemble's neurons are tuned to the WEIGHT
    dimension (enc_dim_0 -- net.error[0] in build_network, fed from the
    counting memory via W_weight) rather than the PE dimension (enc_dim_1).
    Direct port of figure_yoo_neural.py's own _weight_on_cols -- same
    encoders file layout, same threshold.
    """
    on_idx = pid_enc[pid_enc["enc_dim_0"] > NEURAL_ENCODER_THRESHOLD]["neuron_idx"].values
    return [f"n{i}" for i in on_idx if f"n{i}" in neuron_cols]


def _load_neural_probe_variability(min_trials: int = 3) -> pd.DataFrame | None:
    """Per-virtual-pid response variability (sigma) and PE variability,
    from neural_experiments.py's `synthetic` experiment (Acts 2/3's actual
    data source -- see CLAUDE.md's own "Neural predictions figure" Status
    section for why this replaced the original fitted-pid `probe` data) --
    mean std across repeated qid presentations. Uses the SAME min_trials=3
    gate as _qid_response_std (the canonical sigma computation every other
    figure in this file uses): a (virtual_pid, qid, observation) cell with
    fewer than min_trials repeated presentations has its std discarded
    (NaN) rather than trusted, before averaging per virtual pid.

    alpha_0/lambda_/n_neurons here are the RANDOM draw for that virtual
    pid (see neural_experiments.py's own _synthetic_params), not a fitted
    value -- these are qualitative covariation predictions for future
    empirical studies, not fits to existing behavioural data, so this is
    by design, not a limitation to work around.
    """
    probe_path = NEURAL_EXP_DIR / "synthetic_soltani_numbers_probe.pkl"
    params_path = NEURAL_EXP_DIR / "synthetic_soltani_numbers_params.pkl"
    if not (probe_path.exists() and params_path.exists()):
        return None
    df = pd.read_pickle(probe_path)
    params = pd.read_pickle(params_path)
    agg = (df.groupby(["virtual_pid", "qid", "observation"])[["pe", "response"]]
          .agg(lambda x: x.std() if len(x) >= min_trials else np.nan)
          .dropna())
    if agg.empty:
        return None
    per_pid = agg.groupby("virtual_pid")[["pe", "response"]].mean().reset_index()
    per_pid = per_pid.rename(columns={"pe": "pe_std", "response": "resp_std"})
    return per_pid.merge(params[["virtual_pid", "alpha_0", "lambda_", "n_neurons"]], on="virtual_pid")


def _load_neural_decay_metrics() -> pd.DataFrame | None:
    """Per-virtual-pid activity decay (mean weight-tuned-neuron activity,
    first observation minus last) and response-change decay (mean
    |Delta response|, first 2 observations minus last 2), from
    neural_experiments.py's `synthetic` experiment.

    Weight-tuned-neuron identification happens PER (virtual_pid, trial),
    not per virtual_pid -- confirmed directly (see CLAUDE.md/docs/
    HISTORY.md) that a trial's own error-ensemble encoders depend on that
    trial's own seed, so a single pid-level encoders set (the convention
    the OLD fitted-pid loader used, inherited from utils/save_activities.py)
    would silently misidentify weight-tuned neurons for every trial but
    the one its encoders happened to come from.
    """
    probe_path = NEURAL_EXP_DIR / "synthetic_soltani_numbers_probe.pkl"
    act_path = NEURAL_EXP_DIR / "synthetic_soltani_numbers_activity.pkl"
    enc_path = NEURAL_EXP_DIR / "synthetic_soltani_numbers_encoders.pkl"
    params_path = NEURAL_EXP_DIR / "synthetic_soltani_numbers_params.pkl"
    if not all(p.exists() for p in [probe_path, act_path, enc_path, params_path]):
        return None

    probe = pd.read_pickle(probe_path)
    act = pd.read_pickle(act_path)
    enc = pd.read_pickle(enc_path)
    params = pd.read_pickle(params_path)

    # Weight-tuned neuron indices, per (virtual_pid, trial) -- NOT per
    # virtual_pid alone, since encoders genuinely differ by trial.
    weight_tuned = (enc[enc["enc_dim_0"] > NEURAL_ENCODER_THRESHOLD]
                    .groupby(["virtual_pid", "trial"])["neuron_idx"]
                    .apply(list))

    act_indexed = act.set_index(["virtual_pid", "trial"]).sort_index()
    mean_act_rows = []
    for (vp, trial), idxs in weight_tuned.items():
        if not idxs:
            continue
        cols = [f"n{i}" for i in idxs]
        try:
            sub = act_indexed.loc[(vp, trial)]
        except KeyError:
            continue
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        mean_vals = sub[cols].mean(axis=1)
        for obs, val in zip(sub["observation"], mean_vals):
            mean_act_rows.append({"virtual_pid": vp, "trial": trial,
                                  "observation": obs, "mean_act": val})
    if not mean_act_rows:
        return None
    mean_act_df = pd.DataFrame(mean_act_rows)

    rows = []
    for vp in params["virtual_pid"].unique():
        pid_act = mean_act_df[mean_act_df["virtual_pid"] == vp]
        if pid_act.empty:
            continue
        act_by_obs = pid_act.groupby("observation")["mean_act"].mean()
        obs_sorted = sorted(act_by_obs.index)
        if len(obs_sorted) < 2:
            continue
        act_decay = float(act_by_obs[obs_sorted[0]]) - float(act_by_obs[obs_sorted[-1]])

        pid_resp = probe[probe["virtual_pid"] == vp].sort_values(["trial", "observation"]).copy()
        obs_sorted_r = sorted(pid_resp["observation"].unique())
        if len(obs_sorted_r) < 4:
            continue
        pid_resp["delta"] = pid_resp.groupby("trial")["response"].diff().abs()
        first_obs = obs_sorted_r[0]
        pid_resp.loc[pid_resp["observation"] == first_obs, "delta"] = (
            pid_resp.loc[pid_resp["observation"] == first_obs, "response"].abs())
        early = pid_resp[pid_resp["observation"].isin(obs_sorted_r[:2])]["delta"].mean()
        late = pid_resp[pid_resp["observation"].isin(obs_sorted_r[-2:])]["delta"].mean()
        resp_decay = float(early) - float(late)

        rows.append({"virtual_pid": int(vp), "act_decay": act_decay, "resp_decay": resp_decay})

    if not rows:
        return None
    return pd.DataFrame(rows).merge(params[["virtual_pid", "alpha_0", "lambda_", "n_neurons"]], on="virtual_pid")


def _plot_neural_sigma_vs_pe_variability(ax) -> None:
    """Panel: response variability (sigma) vs PE variability, one point per
    virtual pid -- both measurable in a real neuroimaging study with no
    model fitting on either axis. Points small and low-alpha, regression
    line thick with its CI band -- the fit is the point of this panel, not
    any individual point.
    """
    df = _load_neural_probe_variability()
    if df is None or len(df) < 3:
        ax.text(0.5, 0.5, "No probe variability data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    color = get_palette(6)[0]
    r, p = pearsonr(df["resp_std"], df["pe_std"])
    ax.scatter(df["pe_std"], df["resp_std"], color=color, s=8, alpha=0.35, zorder=2)
    sns.regplot(data=df, x="pe_std", y="resp_std", ax=ax, color=color, ci=95,
               scatter=False, line_kws={"lw": 2.2, "zorder": 3},
               label=f"r={r:.2f}{pvalue_to_stars(p)}")
    ax.set_xlabel("\u03c3PE")
    ax.set_ylabel("\u03c3R")
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper left")
    sns.despine(ax=ax, top=True, right=True)


def _plot_neural_resp_vs_act_decay(ax) -> None:
    """Panel: NEF's own |Delta response| decay vs activity decay, one
    point per virtual pid -- both measurable, no model fitting on either
    axis. Points small and low-alpha, regression line thick with its CI
    band -- the fit is the point of this panel, not any individual point.
    """
    df = _load_neural_decay_metrics()
    if df is None or len(df) < 3:
        ax.text(0.5, 0.5, "No activity/response decay data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    color = get_palette(6)[0]
    r, p = pearsonr(df["act_decay"], df["resp_decay"])
    ax.scatter(df["act_decay"], df["resp_decay"], color=color, s=8, alpha=0.35, zorder=2)
    sns.regplot(data=df, x="act_decay", y="resp_decay", ax=ax, color=color, ci=95,
               scatter=False, line_kws={"lw": 2.2, "zorder": 3},
               label=f"r={r:.2f}{pvalue_to_stars(p)}")
    ax.set_xlabel("\u0394A (Hz)")
    ax.set_ylabel("\u0394R decay")
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper left")
    sns.despine(ax=ax, top=True, right=True)


def _plot_neural_dual_vs_param(
    ax, df: pd.DataFrame, param_col: str, param_label: str,
    y1_col: str, y2_col: str, y1_label: str, y2_label: str,
    include_x_zero: bool = False,
):
    """Generic panel: two dependent measures (y1_col, y2_col), twin y-axes,
    both plotted against ONE parameter (param_col: alpha_0, lambda_, or
    n_neurons). Reused across all three parameters for both the
    sigma_R/sigma_PE row and the DeltaR/DeltaA-decay row, so each row
    shows how much each of the three parameters individually contributes
    to that row's pair of dependent measures -- a breakdown, not a
    replacement for the actual (still-pending) multivariate regression.
    Points small and low-alpha, regression lines thick with their CI
    bands -- the fits are the point of these panels, not any individual
    point.

    Returns ax2 (the twin axis) so a caller can apply shared y-limits
    across a row afterward -- twin axes aren't reachable via plt.subplots'
    own sharey, since they're created per-panel, not at subplot-creation
    time.

    include_x_zero=True extends the x-axis to include 0 as a tick/limit
    (appropriate for a bounded [0,1]-style parameter like alpha_0/lambda_,
    where 0 is a meaningful reference point) -- left False for n_neurons,
    where the real data starts at 500 and forcing the axis down to 0 would
    waste half the panel on empty space.
    """
    pal = get_palette(6)
    c1, c2 = pal[0], pal[1]
    r1, p1 = pearsonr(df[param_col], df[y1_col])
    r2, p2 = pearsonr(df[param_col], df[y2_col])

    ax.scatter(df[param_col], df[y1_col], color=c1, s=8, alpha=0.35, zorder=2)
    sns.regplot(data=df, x=param_col, y=y1_col, ax=ax, color=c1, ci=95,
               scatter=False, line_kws={"lw": 2.2, "zorder": 3},
               label=f"r={r1:.2f}{pvalue_to_stars(p1)}")
    ax.set_xlabel(param_label)
    ax.set_ylabel(y1_label, color=c1)
    ax.tick_params(axis="y", labelcolor=c1)
    if include_x_zero:
        ax.set_xlim(left=0)

    ax2 = ax.twinx()
    ax2.scatter(df[param_col], df[y2_col], color=c2, s=8, alpha=0.35, zorder=2)
    sns.regplot(data=df, x=param_col, y=y2_col, ax=ax2, color=c2, ci=95,
               scatter=False, line_kws={"lw": 2.2, "zorder": 3},
               label=f"r={r2:.2f}{pvalue_to_stars(p2)}")
    ax2.set_ylabel(y2_label, color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    sns.despine(ax=ax2, top=True)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=7,
             frameon=True, framealpha=0.9, loc="best")
    sns.despine(ax=ax, top=True, right=True)
    return ax2


def _oddball_primary_context(grid: pd.DataFrame) -> tuple[float, float]:
    """Pick one representative (cluster_center, oddball_deviation) for
    panels that need a single context rather than the full grid.

    Center: 50 (the task's own raw-scale midpoint) if present in the grid
    -- per instruction, since it's the cleanest single context to show a
    representative PE trace at (deviation sign is symmetric there, unlike
    off-center positions where sign and position are confounded -- see
    chat). Falls back to the middle of whatever centers ARE present if 50
    isn't one of them, rather than erroring, so this still degrades
    gracefully for a grid that doesn't include 50.

    Deviation: smallest-magnitude POSITIVE deviation available (falling
    back to whatever deviation exists if none are positive).
    """
    centers = sorted(grid["cluster_center"].unique())
    deviations = sorted(grid["oddball_deviation"].unique())
    center = 50.0 if 50.0 in centers else centers[len(centers) // 2]
    positive = [d for d in deviations if d > 0]
    deviation = min(positive) if positive else deviations[-1]
    return center, deviation


def _oddball_base_value(d: dict, sweep_param: str) -> float:
    return {"alpha_0": d["base_alpha_0"], "lambda_": d["base_lambda_"],
           "n_neurons": d["base_n_neurons"]}[sweep_param]


def _plot_oddball_pe_trace(ax, sweep_param: str, task: str = "soltani_numbers") -> None:
    """Panel: |decoded PE| vs time WITHIN THE ODDBALL (4TH) OBSERVATION
    ONLY -- the trace neural_experiments.py's _oddball_worker now returns
    is already windowed to that single observation (t=0 at that
    observation's own onset, excluding its preceding ITI), so this panel
    shows the model's response to the surprise itself rather than the
    whole 4-observation trial. ONE representative cluster_center
    (_oddball_primary_context's own pick, currently 50), up to 3
    representative values of sweep_param (low/mid/high of whatever was
    actually run, per instruction) -- BOTH deviation signs present in the
    grid are folded into ONE long-format frame per sweep_param value and
    handed to sns.lineplot, which aggregates them itself (mean + its own
    default CI band across the two deviation-sign traces at each
    timepoint) rather than being drawn as separate lines/linestyles --
    per instruction, simpler than manually distinguishing sign. No
    title -- the legend already identifies every line.
    """
    path = NEURAL_EXP_DIR / f"oddball_{sweep_param}_{task}.pkl"
    if not path.exists():
        ax.text(0.5, 0.5, f"No oddball {sweep_param} data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    d = pd.read_pickle(path)
    grid, traces = d["grid"], d["traces"]
    center, _ = _oddball_primary_context(grid)
    deviations = sorted(grid["oddball_deviation"].unique())

    values = sorted(grid[sweep_param].unique())
    picks = [values[0], values[len(values) // 2], values[-1]] if len(values) >= 3 else values

    rows = []
    for val in picks:
        for deviation in deviations:
            tr = traces.get((center, deviation, val))
            if tr is None:
                continue
            rows.append(pd.DataFrame({"t": tr["t"], "pe": tr["pe"], "sweep_value": val}))
    if not rows:
        ax.text(0.5, 0.5, f"No oddball {sweep_param} data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    df = pd.concat(rows, ignore_index=True)

    label_sym = {"alpha_0": "\u03b1\u2080", "lambda_": "\u03bb", "n_neurons": "n"}
    sym = label_sym.get(sweep_param, sweep_param)
    pal = get_palette(max(6, len(picks)))
    color_map = {val: pal[i] for i, val in enumerate(picks)}

    sns.lineplot(data=df, x="t", y="pe", hue="sweep_value", hue_order=picks,
                palette=color_map, ax=ax, lw=1.8)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [f"{sym}={float(l):g}" for l in labels],
              fontsize=7, frameon=True, framealpha=0.9, loc="upper right")

    ax.set_xlim(0, float(df["t"].max()))
    ax.set_xlabel("Time since oddball onset (s)")
    ax.set_ylabel("|Decoded PE|")
    sns.despine(ax=ax, top=True, right=True)


def _plot_oddball_param_effect(ax, sweep_param: str, task: str = "soltani_numbers") -> None:
    """Panel: sweep_param (x) vs max |decoded PE| (peak error response
    within the oddball observation's own window) AND its absolute
    decrease by the end of that SAME window (max minus the window's own
    end value -- how much learning/value-updating has shrunk the error
    WITHIN the oddball observation, not across the whole trial), twin
    y-axes -- mean +- SEM AGGREGATED across every (cluster_center,
    oddball_deviation) cell in the grid for that sweep_param value.

    AGGREGATION JUSTIFICATION (revised after real data, not the original
    a-priori assumption -- see chat): centers are NOT actually
    interchangeable in absolute magnitude -- both max_pe and decrease vary
    non-monotonically with the oddball's own distance from the task's raw
    midpoint (50), confirmed directly on soltani_numbers (e.g. |rescaled
    oddball position| 0.20 gave a LARGER alpha_0-driven swing than 0.40,
    breaking simple edge-distance monotonicity). What IS robust across
    every position tested is the SIGN of the sweep_param's effect --
    every tested cell showed higher alpha_0 -> higher max_pe AND higher
    decrease, with only the magnitude of that effect varying by position.
    Averaging across the grid here is therefore a summary of that
    direction-robust effect, not a claim that position doesn't matter --
    the per-position magnitude variation is a real, separate finding
    that belongs in its own analysis/panel, not smoothed over silently by
    this aggregate.
    """
    path = NEURAL_EXP_DIR / f"oddball_{sweep_param}_{task}.pkl"
    if not path.exists():
        ax.text(0.5, 0.5, f"No oddball {sweep_param} data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    d = pd.read_pickle(path)
    grid = d["grid"]
    agg = (grid.groupby(sweep_param)[["max_pe", "decrease"]]
          .agg(["mean", "sem"]).reset_index())
    agg.columns = [sweep_param, "max_pe_mean", "max_pe_sem",
                  "decrease_mean", "decrease_sem"]
    agg = agg.sort_values(sweep_param)

    label_sym = {"alpha_0": "\u03b1\u2080", "lambda_": "\u03bb", "n_neurons": "n"}
    sym = label_sym.get(sweep_param, sweep_param)

    pal = get_palette(6)
    c1, c2 = pal[0], pal[1]
    ax.errorbar(agg[sweep_param], agg["max_pe_mean"], yerr=agg["max_pe_sem"],
               fmt="o-", color=c1, lw=1.8, ms=5, capsize=3)
    ax.set_xlabel(sym)
    ax.set_ylabel("Max |decoded PE|", color=c1)
    ax.tick_params(axis="y", labelcolor=c1)
    if sweep_param in ("alpha_0", "lambda_"):
        ax.set_xlim(left=0)

    ax2 = ax.twinx()
    ax2.errorbar(agg[sweep_param], agg["decrease_mean"], yerr=agg["decrease_sem"],
                fmt="o-", color=c2, lw=1.8, ms=5, capsize=3)
    ax2.set_ylabel("PE decrease (max\u2212end)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    sns.despine(ax=ax2, top=True)
    sns.despine(ax=ax, top=True, right=True)


def _plot_oddball_dv_scatter(ax, sweep_param: str, task: str = "soltani_numbers") -> None:
    """Row 1, col 3: max |decoded PE| (x) vs its own decrease by the
    oddball window's end (y), one point per (cluster_center,
    oddball_deviation, sweep_param) grid cell -- the SAME full grid
    _plot_oddball_param_effect aggregates into twin-axis means vs
    sweep_param, here shown instead as a direct scatter of panel 2's own
    two dependent variables against each other. Matches the ORIGINAL
    neural_giant figure's own DV-vs-DV convention exactly
    (_plot_neural_sigma_vs_pe_variability / _plot_neural_resp_vs_act_decay:
    single flat color, small low-alpha points, sns.regplot fit line with
    CI band, pearsonr r + significance stars in the legend) rather than
    color-coding by sweep_param -- this directly visualizes the row's own
    claim (higher alpha_0 -> both higher max_pe AND higher decrease) as a
    single positive correlation across the whole grid.
    """
    path = NEURAL_EXP_DIR / f"oddball_{sweep_param}_{task}.pkl"
    if not path.exists():
        ax.text(0.5, 0.5, f"No oddball {sweep_param} data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    d = pd.read_pickle(path)
    grid = d["grid"]

    color = get_palette(6)[0]
    r, p = pearsonr(grid["max_pe"], grid["decrease"])
    ax.scatter(grid["max_pe"], grid["decrease"], color=color, s=8, alpha=0.35, zorder=2)
    sns.regplot(data=grid, x="max_pe", y="decrease", ax=ax, color=color, ci=95,
               scatter=False, line_kws={"lw": 2.2, "zorder": 3},
               label=f"r={r:.2f}{pvalue_to_stars(p)}")
    ax.set_xlabel("Max |decoded PE|")
    ax.set_ylabel("PE decrease (max−end)")
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper left")
    sns.despine(ax=ax, top=True, right=True)


def _plot_oddball_center_invariance(ax, sweep_param: str, task: str = "soltani_numbers") -> None:
    """Panel: |decoded PE| vs time WITHIN THE ODDBALL (4TH) OBSERVATION
    ONLY (see _plot_oddball_pe_trace's own note -- the saved trace is
    already windowed to this one observation), one line per
    cluster_center, all at ONE representative oddball_deviation and
    sweep_param held at its own --base_* value (whichever sweep value is
    closest to it) -- a direct visual test of the person's own
    prediction that a fixed-magnitude surprise produces the same
    response regardless of where the cluster sits. If these collapse
    onto each other, that confirms _plot_oddball_param_effect's own
    cross-center aggregation is legitimate; if they don't, that's a
    real, informative finding about where the model's behaviour depends
    on absolute position (e.g. saturation near the edges of the rescaled
    range), not just relative deviation -- and this panel should stay in
    the figure rather than be removed as trivial.
    """
    path = NEURAL_EXP_DIR / f"oddball_{sweep_param}_{task}.pkl"
    if not path.exists():
        ax.text(0.5, 0.5, f"No oddball {sweep_param} data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    d = pd.read_pickle(path)
    grid, traces = d["grid"], d["traces"]
    centers = sorted(grid["cluster_center"].unique())
    _, deviation = _oddball_primary_context(grid)

    base_val = _oddball_base_value(d, sweep_param)
    values = sorted(grid[sweep_param].unique())
    closest_val = min(values, key=lambda v: abs(v - base_val))

    pal = get_palette(max(6, len(centers)))
    t_max = None
    for i, c in enumerate(centers):
        tr = traces.get((c, deviation, closest_val))
        if tr is None:
            continue
        ax.plot(tr["t"], tr["pe"], color=pal[i], lw=1.5, label=f"center={c:g}")
        t_max = tr["t"][-1] if t_max is None else max(t_max, tr["t"][-1])

    if t_max is not None:
        ax.set_xlim(0, t_max)

    label_sym = {"alpha_0": "\u03b1\u2080", "lambda_": "\u03bb", "n_neurons": "n"}
    sym = label_sym.get(sweep_param, sweep_param)
    ax.set_xlabel("Time since oddball onset (s)")
    ax.set_ylabel("|Decoded PE|")
    ax.set_title(f"deviation={deviation:+g}, {sym}={closest_val:g}", fontsize=8)
    ax.legend(fontsize=7, frameon=True, framealpha=0.9, loc="upper right")
    sns.despine(ax=ax, top=True, right=True)


def _plot_neural_giant2_activity_vs_obs(ax, sweep_param: str, task: str = "soltani_numbers",
                                        low_thresh: float = 0.2, high_thresh: float = 0.7) -> None:
    """Row 2/3, col 1: weight-tuned error-neuron activity (Hz) vs
    observation, for two GROUPS of replicates -- every pid/virtual_pid
    with sweep_param <= low_thresh (line 1) and every one with
    sweep_param >= high_thresh (line 2) -- each pid gets its OWN single
    randomly-drawn value now, not a shared explicit grid (see param_
    scan's own docstring for why), so this groups BY VALUE RANGE rather
    than picking one representative value/pid. Each matching pid's own
    trials are pre-folded to ONE row per (pid, observation) -- raw
    per-timestep samples within a trial aren't independent, so they must
    be collapsed before handing off -- then that per-pid frame is passed
    to sns.lineplot with hue=group, which computes mean + its own default
    CI band ACROSS PIDS itself, matching _plot_oddball_pe_trace's own
    established convention (hand seaborn a long-format frame at the
    correct unit-of-independence, let it aggregate) rather than manually
    computing mean/SEM + fill_between. N per group = the pid count shown
    in its own legend entry. Direct structural port of the ORIGINAL
    neural_giant's own panel 3 (_plot_neural_lambda_activity), reading
    neural_experiments.py's own `param_scan` experiment instead of
    `sweep` (a genuinely different design from row 1's `oddball` -- every
    replicate's own full 32-trial sequence, not an arbitrary constant-
    input toy trial and not a windowed surprise response). Weight-tuned-
    neuron reduction is ALREADY applied there, per trial (encoders
    genuinely differ by trial/seed).
    """
    path = NEURAL_EXP_DIR / f"param_scan_{sweep_param}_{task}.pkl"
    if not path.exists():
        ax.text(0.5, 0.5, f"No param_scan {sweep_param} data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    from fitting.model_params import _NEF_FIXED

    df = pd.read_pickle(path).copy()
    t_iti, t_obs = _NEF_FIXED["t_iti"], _NEF_FIXED["t_obs"]
    obs_num, t_within = _fold_observation_time(df["t"].values, t_iti, t_obs)
    df["observation"] = obs_num
    df["t_within_obs"] = t_within
    active = df[~np.isnan(df["t_within_obs"])]

    # Since each pid now has its OWN single sweep_value (a random draw,
    # not a shared explicit grid -- see param_scan's own docstring),
    # group by VALUE RANGE (<=low_thresh, >=high_thresh) rather than
    # picking one representative pid -- lets every matching pid
    # contribute (particularly useful with --trial_source synthetic,
    # where N=200 makes multi-pid bins routine rather than the exception).
    pid_values = active.groupby("pid")["sweep_value"].first()
    low_pids = pid_values[pid_values <= low_thresh].index
    high_pids = pid_values[pid_values >= high_thresh].index

    label_sym = {"alpha_0": "\u03b1\u2080", "lambda_": "\u03bb", "n_neurons": "n"}
    sym = label_sym.get(sweep_param, sweep_param)
    groups = [("\u2264", low_thresh, low_pids), ("\u2265", high_thresh, high_pids)]

    rows = []
    hue_order = []
    for rel, thresh, pids_sel in groups:
        if len(pids_sel) == 0:
            print(f"  Warning: no pids with {sweep_param}{rel}{thresh:g} in "
                  f"param_scan_{sweep_param}_{task}.pkl -- line skipped")
            continue
        label = f"{sym}{rel}{thresh:g} (n={len(pids_sel)})"
        hue_order.append(label)
        sub = active[active["pid"].isin(pids_sel)]
        # Pre-fold to ONE row per (pid, observation) -- raw per-timestep
        # samples within a trial aren't independent draws, so they must
        # be collapsed to a single trial-mean, then averaged across that
        # pid's own trials, BEFORE handing off to sns.lineplot; otherwise
        # seaborn would treat every raw timestep as its own independent
        # sample and compute a falsely tight CI.
        per_trial = (sub.groupby(["pid", "trial", "observation"])["weight_tuned_activity"]
                    .mean().reset_index())
        per_pid = (per_trial.groupby(["pid", "observation"])["weight_tuned_activity"]
                  .mean().reset_index())
        per_pid["group"] = label
        rows.append(per_pid)

    if not rows:
        ax.text(0.5, 0.5, f"No pids in either threshold group for {sweep_param}",
               ha="center", va="center", transform=ax.transAxes, color="0.5", style="italic")
        return
    combined = pd.concat(rows, ignore_index=True)

    pal = get_palette(6)
    color_map = {label: pal[i] for i, label in enumerate(hue_order)}
    sns.lineplot(data=combined, x="observation", y="weight_tuned_activity", hue="group",
                hue_order=hue_order, palette=color_map, ax=ax, lw=1.8)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper right", title=None)

    n_obs_max = int(df["observation"].max())
    ax.set_xlabel("Observation")
    ax.set_xlim(0, n_obs_max)
    ax.set_ylabel("Weight-tuned error\nneuron activity (Hz)")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper right")
    sns.despine(ax=ax, top=True, right=True)


def _param_scan_decay_metrics(sweep_param: str, task: str = "soltani_numbers") -> pd.DataFrame | None:
    """Per (sweep_value, seed): response-change decay (mean |Delta
    response| first 2 obs minus last 2) and weight-tuned activity decay
    (activity at first obs minus last), extracted from
    neural_experiments.py's param_scan raw per-timestep output at each
    observation's own readout time -- matching models.NEF._extract_
    responses' own small-window-average convention for response (|t -
    t_resp| < dt*3) and neural_experiments.py's own READOUT_OFFSET (0.5s
    into the observation) for activity, the SAME conventions the
    synthetic/oddball pipelines already use elsewhere in this file.

    Output column is named EXACTLY `sweep_param` (not a generic
    "sweep_value") so this drops directly into _plot_neural_dual_vs_param,
    the SAME shared helper the original neural_giant's own row-3 panels
    use, with no adapter needed.
    """
    path = NEURAL_EXP_DIR / f"param_scan_{sweep_param}_{task}.pkl"
    if not path.exists():
        return None
    from fitting.model_params import _NEF_FIXED

    df = pd.read_pickle(path)
    dt = float(_NEF_FIXED["dt"])
    t_iti, t_obs = float(_NEF_FIXED["t_iti"]), float(_NEF_FIXED["t_obs"])
    t_step = t_obs + t_iti
    n_obs = int(np.floor(df["t"].max() / t_step))

    # One point per (sweep_value, pid) -- pooling ACROSS that pid's own 32
    # real trials before computing decay, exactly mirroring
    # _load_neural_decay_metrics's own convention (pool-then-decay on the
    # pid-level per-observation curve, not decay-per-trial-then-averaged).
    # This project's data is now real (pid, trial) simulations rather than
    # one row per arbitrary seed -- see neural_experiments.py's
    # _param_scan_worker docstring for why the earlier constant-input
    # toy-trial design was replaced.
    rows = []
    for (sweep_value, pid), g in df.groupby(["sweep_value", "pid"]):
        resp_rows, act_rows = [], []
        for trial, gt in g.groupby("trial"):
            gt = gt.sort_values("t")
            t_arr = gt["t"].to_numpy()
            value_arr = gt["value_decoded"].to_numpy()
            act_arr = gt["weight_tuned_activity"].to_numpy()
            for i in range(n_obs):
                t_resp = t_iti + i * t_step + t_obs
                resp_mask = np.abs(t_arr - t_resp) < dt * 3
                resp_val = float(np.mean(value_arr[resp_mask])) if resp_mask.any() else np.nan
                resp_rows.append({"trial": trial, "observation": i + 1, "response": resp_val})

                t_act = t_iti + i * t_step + NEURAL_READOUT_OFFSET
                idx_act = int(np.argmin(np.abs(t_arr - t_act)))
                act_rows.append({"trial": trial, "observation": i + 1,
                                 "activity": float(act_arr[idx_act])})

        resp_df = pd.DataFrame(resp_rows)
        act_df = pd.DataFrame(act_rows)
        obs_sorted = sorted(act_df["observation"].unique())

        # Activity: pool across ALL of this pid's trials per observation,
        # THEN take first-minus-last on that pooled curve.
        act_by_obs = act_df.groupby("observation")["activity"].mean()
        act_decay = float(act_by_obs[obs_sorted[0]]) - float(act_by_obs[obs_sorted[-1]])

        # Response: per-trial delta (first observation's own delta
        # convention is |response|, matching this file's other decay
        # metrics), THEN pool deltas across trials per observation
        # position, early(first 2) minus late(last 2).
        resp_df = resp_df.sort_values(["trial", "observation"])
        resp_df["delta"] = resp_df.groupby("trial")["response"].diff().abs()
        first_obs = obs_sorted[0]
        resp_df.loc[resp_df["observation"] == first_obs, "delta"] = (
            resp_df.loc[resp_df["observation"] == first_obs, "response"].abs())
        early = resp_df[resp_df["observation"].isin(obs_sorted[:2])]["delta"].mean()
        late = resp_df[resp_df["observation"].isin(obs_sorted[-2:])]["delta"].mean()
        resp_decay = float(early) - float(late)

        rows.append({sweep_param: sweep_value, "pid": pid,
                    "resp_decay": resp_decay, "act_decay": act_decay})

    return pd.DataFrame(rows)


def _plot_neural_giant2_decay_vs_param(ax, sweep_param: str, task: str = "soltani_numbers") -> None:
    """Row 2/3, col 2: decay(deltaR) AND decay(deltaA) vs sweep_param,
    twin axes, one point per (sweep_value, seed) -- reuses
    _plot_neural_dual_vs_param DIRECTLY, the SAME helper the ORIGINAL
    neural_giant's own row-3 panels use, so the visual convention is
    identical. Seeds play the role virtual pids played in that figure's
    own large-N regression (many independent draws), rather than a
    separate large-N synthetic campaign -- this figure's own design scans
    explicit parameter values instead of random draws, so seeds are the
    only source of independent replication available per value.
    """
    df = _param_scan_decay_metrics(sweep_param, task)
    if df is None or len(df) < 3:
        ax.text(0.5, 0.5, f"No param_scan {sweep_param} decay data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    label_sym = {"alpha_0": "\u03b1\u2080", "lambda_": "\u03bb", "n_neurons": "n"}
    param_label = label_sym.get(sweep_param, sweep_param)
    _plot_neural_dual_vs_param(
        ax, df, sweep_param, param_label,
        "resp_decay", "act_decay", "decay (\u0394R)", "decay (\u0394A)",
        include_x_zero=(sweep_param in ("alpha_0", "lambda_")))


def _plot_param_scan_dv_scatter(ax, sweep_param: str, task: str = "soltani_numbers") -> None:
    """Row 2, col 3: NEF's own |Delta response| decay (y) vs weight-tuned
    activity decay (x), one point per (sweep_param value, real pid) --
    the SAME per-pid decay metrics _plot_neural_giant2_decay_vs_param
    twin-axis plots vs sweep_param, here shown instead as a direct
    scatter of THAT panel's own two dependent variables against each
    other. This is the row-2 analogue of row 1's _plot_oddball_dv_scatter
    and of the ORIGINAL neural_giant's own DV-vs-DV panels (panel 5:
    sigma_R vs sigma_PE; panel 9: DeltaR-decay vs DeltaA-decay) --
    matching that exact convention (single flat color, small low-alpha
    points, sns.regplot fit line with CI band, pearsonr r + significance
    stars in the legend) rather than color-coding by sweep_param.
    """
    df = _param_scan_decay_metrics(sweep_param, task)
    if df is None or len(df) < 3:
        ax.text(0.5, 0.5, f"No param_scan {sweep_param} decay data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    color = get_palette(6)[0]
    r, p = pearsonr(df["act_decay"], df["resp_decay"])
    ax.scatter(df["act_decay"], df["resp_decay"], color=color, s=8, alpha=0.35, zorder=2)
    sns.regplot(data=df, x="act_decay", y="resp_decay", ax=ax, color=color, ci=95,
               scatter=False, line_kws={"lw": 2.2, "zorder": 3},
               label=f"r={r:.2f}{pvalue_to_stars(p)}")
    ax.set_xlabel("ΔA (Hz)")
    ax.set_ylabel("ΔR decay")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper left")
    sns.despine(ax=ax, top=True, right=True)


def make_neural_giant2() -> Path:
    """3x2 figure: a second neural-predictions figure, one row per
    parameter (alpha_0, lambda_, n_neurons), each investigated via its own
    fresh grid of oddball simulations (3 observations clustered around a
    center, then one surprising observation deviating from it, across
    several centers x deviations x parameter values) rather than the
    giant's own random-virtual-pid design.

      Row 1 (alpha_0, lambda_=0.7, n_neurons=500/nc=2000 -- the RMSE
        production default):
        Col 1: |decoded PE| vs time, one representative center, BOTH
          deviation signs, 3 representative alpha_0 values.
        Col 2: alpha_0 (x) vs max |decoded PE| AND absolute decrease by
          the end of the oddball's own window, twin axes, mean +- SEM
          aggregated across the whole grid -- the neural prediction this
          row tests: higher alpha_0 produces a bigger initial response
          AND more dramatic attenuation from learning/value-updating.
        Col 3: max |decoded PE| (x) vs decrease (y) plotted directly
          against each other, one point per full grid cell -- the same
          two DVs col 2 twin-axis plots vs alpha_0, here as a direct
          scatter, matching the ORIGINAL neural_giant's own DV-vs-DV
          panels (5, 9).
      Row 2 (lambda_ swept 0.1-1.0, alpha_0=0.7, n_neurons=500/nc=2000 --
        neural_experiments.py's own `param_scan` experiment, NOT
        `oddball` -- a different design: every REAL soltani_numbers
        pid's own full 32-trial sequence (CORRECTED this session -- an
        earlier version used a degenerate constant-input arbitrary toy
        trial, Act 1.2-style; see param_scan's own docstring for why that
        was replaced), not a windowed surprise response):
        Col 1: weight-tuned error-neuron activity (Hz) vs observation, 2
          representative lambda_ values (low/high of whatever grid was
          run), mean +- SEM across REAL PIDS -- direct structural port of
          the ORIGINAL neural_giant's own panel 3
          (_plot_neural_lambda_activity), just at this figure's own
          explicit lambda_ values instead of that figure's own base
          value, and folded hierarchically (trial -> pid) since each
          replicate is now a real multi-trial participant, not a single
          arbitrary seed.
        Col 2: decay(deltaR) AND decay(deltaA) vs lambda_, twin axes, one
          point per (lambda_, real pid) -- direct reuse of
          _plot_neural_dual_vs_param (the SAME helper the ORIGINAL
          neural_giant's own row-3 panels use), with real pids playing
          the "many independent draws" role that `synthetic`'s virtual
          pids played in that figure, since this figure's own design
          scans explicit parameter values rather than random draws.
        Col 3: decay(deltaA) (x) vs decay(deltaR) (y) plotted directly
          against each other, one point per (lambda_, real pid) -- the
          row-2 analogue of row 1's own col-3 panel, and of the ORIGINAL
          neural_giant's own panel 9 (DeltaR-decay vs DeltaA-decay).
      Row 3 (n_neurons): NOT YET BUILT -- same param_scan structure, own
        fresh scan, once row 2 is confirmed.

    The center-invariance check (col 3 in an earlier version of this
    figure) was REMOVED per instruction -- it had already served its
    purpose: the analysis it was checking for (whether magnitude is
    center-independent) came back genuinely non-trivial (magnitude varies
    non-monotonically with the oddball's own distance from the task's raw
    midpoint -- see chat and _plot_oddball_param_effect's own docstring),
    so keeping a dedicated panel around to re-confirm that on every render
    was no longer the point; the finding itself is now documented in
    prose instead. _plot_oddball_center_invariance itself is left defined
    (not deleted) in case a future row wants to re-run this check on its
    own grid, just no longer wired into this figure's layout.

    Row 1 reads neural_experiments.py's own `oddball` experiment; row 2
    reads its `param_scan` experiment -- both cluster-bound (--mode
    run/submit/collect; a real timing check found even a modest
    single-context oddball run exceeds a reasonable single local call,
    and param_scan's own per-real-pid 32-trial jobs cost similarly).
    """
    _apply_slide_style()
    fig, axes = plt.subplots(3, 3, figsize=(FIGURE_SIZE[0] * 0.8, FIGURE_SIZE[1] * 2.1 * 0.75),
                             constrained_layout=True)

    _plot_oddball_pe_trace(axes[0, 0], "alpha_0")
    _plot_oddball_param_effect(axes[0, 1], "alpha_0")
    _plot_oddball_dv_scatter(axes[0, 2], "alpha_0")

    _plot_neural_giant2_activity_vs_obs(axes[1, 0], "lambda_")
    _plot_neural_giant2_decay_vs_param(axes[1, 1], "lambda_")
    _plot_param_scan_dv_scatter(axes[1, 2], "lambda_")

    for col in (0, 1, 2):
        axes[2, col].text(0.5, 0.5, "n_neurons row not yet built", ha="center",
                          va="center", transform=axes[2, col].transAxes,
                          color="0.5", style="italic")

    out_path, _ = _save_fig(fig, "neural_giant2")
    plt.close(fig)
    return out_path


def make_neural_giant() -> Path:
    """3x4 figure: Acts 1-3 of the neural predictions narrative (see chat
    for the full 5-act plan):
      Row 1 (Act 1, toy/illustrative, arbitrary params):
        Panel 1: spike raster + decoded-PE demo.
        Panel 2: alpha_0 x n_neurons cross product, decoded PE vs
          time-within-observation, first observation only.
        Panel 3: lambda sweep, error-neuron activity vs observation.
        Panel 4: (empty -- row 1 only has 3 panels).
      Row 2 (sigma_R and sigma_PE, both measurable, no fitting on either):
        Panel 5: sigma_R vs sigma_PE.
        Panels 6-8: sigma_R AND sigma_PE, twin axes, each vs ONE of
          alpha_0/lambda_/n_neurons -- a breakdown of how much each
          parameter individually contributes, not a substitute for the
          still-pending multivariate regression (see chat).
      Row 3 (DeltaR-decay and DeltaA-decay, same structure):
        Panel 9: DeltaR-decay vs DeltaA-decay.
        Panels 10-12: DeltaR-decay AND DeltaA-decay, twin axes, each vs
          ONE of alpha_0/lambda_/n_neurons.

    Rows 2/3 read neural_experiments.py's own `synthetic` experiment --
    N randomly-parameterized virtual pids (NOT fitted params), per
    instruction; see CLAUDE.md's own "Neural predictions figure" Status
    section and docs/HISTORY.md for the full design rationale, including
    two real bugs found and fixed along the way (a response-readout
    averaging mismatch, and a raw-vs-canonical-scale mismatch that
    saturated NEF's ensembles) and the sampling-bounds narrowing that
    followed (alpha_0 in [0.5,1], lambda_ in [0.1,1], n_neurons in
    [500,1500] -- avoiding a genuine floor effect in alpha(t)=alpha_0/
    t^lambda at low alpha_0, and the extra measurement noise at low
    n_neurons that diluted the sigma-related relationships specifically).

    Act 4/5 are not included yet.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(3, 4, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 2.85 * 0.75),
                             constrained_layout=True)

    _plot_neural_raster_demo(axes[0, 0])
    _plot_neural_pe_dynamics(axes[0, 1])
    _plot_neural_lambda_activity(axes[0, 2])
    axes[0, 3].axis("off")

    sigma_df = _load_neural_probe_variability()
    _plot_neural_sigma_vs_pe_variability(axes[1, 0])
    if sigma_df is not None and len(sigma_df) >= 3:
        row2_twins = []
        row2_twins.append(_plot_neural_dual_vs_param(
            axes[1, 1], sigma_df, "alpha_0", "\u03b1\u2080",
            "resp_std", "pe_std", "\u03c3 (response)", "\u03c3 (prediction error)",
            include_x_zero=True))
        row2_twins.append(_plot_neural_dual_vs_param(
            axes[1, 2], sigma_df, "lambda_", "\u03bb",
            "resp_std", "pe_std", "\u03c3 (response)", "\u03c3 (prediction error)",
            include_x_zero=True))
        row2_twins.append(_plot_neural_dual_vs_param(
            axes[1, 3], sigma_df, "n_neurons", "neurons",
            "resp_std", "pe_std", "\u03c3 (response)", "\u03c3 (prediction error)"))

        # Shared y-axes across the whole row: axes[1,0]'s own y (resp_std)
        # and every panel's left axis (also resp_std) get one common range;
        # every panel's twin (right) axis (pe_std) gets another. Twin axes
        # aren't reachable via plt.subplots' own sharey (they're created
        # per-panel, not at subplot-creation time), hence doing this
        # manually here rather than at fig, axes = plt.subplots(...).
        resp_pad = 0.05 * (sigma_df["resp_std"].max() - sigma_df["resp_std"].min())
        pe_pad = 0.05 * (sigma_df["pe_std"].max() - sigma_df["pe_std"].min())
        resp_lim = (sigma_df["resp_std"].min() - resp_pad, sigma_df["resp_std"].max() + resp_pad)
        pe_lim = (sigma_df["pe_std"].min() - pe_pad, sigma_df["pe_std"].max() + pe_pad)
        for col in (0, 1, 2, 3):
            axes[1, col].set_ylim(resp_lim)
        for ax2 in row2_twins:
            ax2.set_ylim(pe_lim)
    else:
        for col in (1, 2, 3):
            axes[1, col].text(0.5, 0.5, "No probe variability data", ha="center",
                              va="center", transform=axes[1, col].transAxes,
                              color="0.5", style="italic")

    decay_df = _load_neural_decay_metrics()
    _plot_neural_resp_vs_act_decay(axes[2, 0])
    if decay_df is not None and len(decay_df) >= 3:
        row3_twins = []
        row3_twins.append(_plot_neural_dual_vs_param(
            axes[2, 1], decay_df, "alpha_0", "\u03b1\u2080",
            "resp_decay", "act_decay", "decay (\u0394R)", "decay (\u0394A)",
            include_x_zero=True))
        row3_twins.append(_plot_neural_dual_vs_param(
            axes[2, 2], decay_df, "lambda_", "\u03bb",
            "resp_decay", "act_decay", "decay (\u0394R)", "decay (\u0394A)",
            include_x_zero=True))
        row3_twins.append(_plot_neural_dual_vs_param(
            axes[2, 3], decay_df, "n_neurons", "neurons",
            "resp_decay", "act_decay", "decay (\u0394R)", "decay (\u0394A)"))

        resp_pad = 0.05 * (decay_df["resp_decay"].max() - decay_df["resp_decay"].min())
        act_pad = 0.05 * (decay_df["act_decay"].max() - decay_df["act_decay"].min())
        resp_lim = (decay_df["resp_decay"].min() - resp_pad, decay_df["resp_decay"].max() + resp_pad)
        act_lim = (decay_df["act_decay"].min() - act_pad, decay_df["act_decay"].max() + act_pad)
        for col in (0, 1, 2, 3):
            axes[2, col].set_ylim(resp_lim)
        for ax2 in row3_twins:
            ax2.set_ylim(act_lim)
    else:
        for col in (1, 2, 3):
            axes[2, col].text(0.5, 0.5, "No activity/response decay data", ha="center",
                              va="center", transform=axes[2, col].transAxes,
                              color="0.5", style="italic")

    out_path, _ = _save_fig(fig, "neural_giant")
    plt.close(fig)
    return out_path


FIGURES = {
    "temporal_performance": make_temporal_performance,
    "model_performance": make_model_performance,
    "model_best_fit": make_model_best_fit,
    "model_performance_nll": make_model_performance_nll,
    "response_change": make_response_change,
    "lambda_human": make_lambda_human,
    "lambda_sanity_human": make_lambda_sanity_human,
    "lambda_overview": make_lambda_overview,
    "lambda_model_correlation": make_lambda_model_correlation,
    "lambda_giant": make_lambda_giant,
    "variability_human": make_variability_human,
    "variability_models": make_variability_models,
    "sigma_sanity_human": make_sigma_sanity_human,
    "sigma_overview": make_sigma_overview,
    "sigma_giant": make_sigma_giant,
    "neural_giant": make_neural_giant,
    "neural_giant2": make_neural_giant2,
    "sigma_model_correlation": make_sigma_model_correlation,
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
