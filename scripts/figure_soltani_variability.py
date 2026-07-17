#!/usr/bin/env python3
"""figure_soltani_variability.py — V group figure for the soltani task/ pilot
(task-continuous + task-binary, Prolific pilot 1).

Layout: 2x3
  Row 1 = task-binary, Row 2 = task-continuous (standing row-order
  convention for soltani figures — see figure_soltani_performance.py)
  Col 1 (~carrabin V-fig panel A): group-level (within-task) KDE of
    prefix response variability, human only (no fitted models yet)
  Col 2 (~carrabin V-fig panel C): test-retest reliability — one point per
    prolific_pid, x/y = prefix response variability in the 1st/2nd half of
    trials. Regplot with scatter=True (raw points shown, not just the fit).
  Col 3: blank in row 1 (binary); row 2 (continuous) shows the cross-task
    comparison for prolific_pids who completed BOTH tasks — binary prefix
    variability on x, continuous on y (~test_sequences.py's plot_L /
    "cross-task prefix variability" panel).

WHY "PREFIX" VARIABILITY, NOT "QID" VARIABILITY (important, dataset-specific)
------------------------------------------------------------------------------
carrabin's qid repeats show an IDENTICAL sequence every time, so response
variability can be computed at any observation index. This task's qid
repeats do NOT: prefix identity and target level are independent axes (see
CLAUDE.md's "Sequence design" section) — a qid's 4 repeats share the same
first `prefix_length` (=4) observations, but the SUFFIX differs on every
repeat because each repeat is steered toward a different target. Verified
directly against this pilot's own data (task/pilot1): for a fixed
(prolific_pid, qid), observations 0-3 are byte-identical across all 4
trials, and observation 4 onward differ on every one. So "response
variability for identical inputs" is only a valid concept when restricted
to `observation < prefix_length` — exactly the same restriction
scripts/test_sequences.py's `prefix_var_per_mid` already applies for the
simulated-model cross-task panel this figure's column 3 is modeled on.
PREFIX_LENGTH is hardcoded to 4 below (not stored in this pilot's parsed
columns) since it's a fixed generation parameter for both tasks in the
current 8x4 production design — confirmed against a real pool sequence
file's own prefix_length field before hardcoding it here.

SAMPLE-SIZE CAVEATS (read before interpreting)
------------------------------------------------
- No minimum-trials-per-cell threshold is enforced (matching
  test_sequences.py's prefix_var_per_mid, which doesn't either) — a
  (qid, observation) cell's std is computed from however many of the 4
  repeats survived (pandas .std() already returns NaN, dropped, below n=2).
  With only 4 repeats per qid total, these per-cell stds are individually
  noisy; only the pid-level average (over many cells) is at all stable, and
  even that is based on very few pids (3 binary, 7 continuous) in this
  pilot.
- The split-half panel (col 2) splits a pid's OWN trial list in half by
  trial index; a qid's repeats are not guaranteed to distribute evenly
  across the split, so a half's prefix-variability estimate may rest on as
  few as 1-2 (qid, obs) cells.
- The cross-task panel (col 3, row 2) has only 2 pids who did both tasks in
  this pilot — no meaningful regression line is drawn below n=3 (matching
  test_sequences.py's own `if len(mids) < 3: continue` convention); the two
  points are shown with a note instead.

PLACEHOLDER NOTE
----------------
Human only — no models have been fit to this pilot yet. Once fits exist,
extend column 1 with model KDE overlays and column 2/3 with per-model
regplots, mirroring figure_carrabin_variability.py's panels A/C exactly.

Run:
    python scripts/figure_soltani_variability.py
    python scripts/figure_soltani_variability.py --results_file task_results_pilot1.pkl
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
from scipy.stats import gaussian_kde, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style, pvalue_to_stars

TASK_ROWS     = ["binary", "continuous"]  # standing row-order convention
PREFIX_LENGTH = 4
HUMAN_COLOR   = "0.3"
MIN_CORR_N    = 3  # matches test_sequences.py's cross-task correlation threshold


# ── metric helpers ───────────────────────────────────────────────────────────

def _prefix_response_std(df_task: pd.DataFrame) -> pd.DataFrame:
    """Mean std(response | qid, observation) within the prefix region,
    per prolific_pid. One row per pid; columns [prolific_pid, resp_std]."""
    sub = (df_task[(df_task["timed_out"] == False) &
                    (df_task["observation"] < PREFIX_LENGTH)]
           .drop_duplicates(subset=["prolific_pid", "trial", "observation"]))
    grp = (sub.groupby(["prolific_pid", "qid", "observation"])["response"]
           .std().dropna().reset_index(name="resp_std"))
    return grp.groupby("prolific_pid")["resp_std"].mean().reset_index()


def _prefix_response_std_split(df_task: pd.DataFrame) -> pd.DataFrame:
    """Per-pid prefix response std computed separately on the first vs
    second half of that pid's own trial list. Columns
    [prolific_pid, first, second]; NaN half dropped by the caller."""
    sub_all = (df_task[df_task["timed_out"] == False]
               .drop_duplicates(subset=["prolific_pid", "trial", "observation"]))
    rows = []
    for pid, g in sub_all.groupby("prolific_pid"):
        trials = sorted(g["trial"].unique())
        mid = len(trials) // 2
        halves = {"first": trials[:mid], "second": trials[mid:]}
        vals = {}
        for half, tset in halves.items():
            gg = g[(g["trial"].isin(tset)) & (g["observation"] < PREFIX_LENGTH)]
            pv = gg.groupby(["qid", "observation"])["response"].std().dropna()
            vals[half] = float(pv.mean()) if len(pv) > 0 else np.nan
        rows.append({"prolific_pid": pid, **vals})
    return pd.DataFrame(rows)


# ── Col 1 — KDE of prefix response variability ──────────────────────────────

def _plot_panel_kde(ax, prefix_std: pd.DataFrame) -> None:
    vals = prefix_std["resp_std"].dropna()
    if len(vals) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    x_max = float(vals.max()) * 1.1
    x = np.linspace(0, x_max, 400)
    kde = gaussian_kde(vals, bw_method="scott")
    density = kde(x)
    density = density / density.max()
    density[x < float(vals.min())] = 0
    density[x > float(vals.max())] = 0

    ax.fill_between(x, density, alpha=0.15, color=HUMAN_COLOR)
    ax.plot(x, density, lw=1.8, color=HUMAN_COLOR, label="Human")

    kde_peak = float(kde(vals.values).max())
    for v in vals.values:
        top = float(kde([v])[0]) / kde_peak
        ax.vlines(v, 0, top, color=HUMAN_COLOR, lw=0.6, alpha=0.5, zorder=2)

    ax.set_xlabel("Prefix response variability")
    ax.set_ylabel("Normalised density")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper right")
    sns.despine(ax=ax, top=True, right=True)


# ── Col 2 — Split-half reliability ──────────────────────────────────────────

def _plot_panel_splithalf(ax, split_df: pd.DataFrame) -> None:
    wide = split_df.dropna(subset=["first", "second"])
    if len(wide) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    sns.regplot(data=wide, x="first", y="second", ax=ax,
                color=HUMAN_COLOR, ci=95 if len(wide) >= MIN_CORR_N else None,
                scatter=True, line_kws={"lw": 1.5},
                scatter_kws={"s": 20, "alpha": 0.7})

    if len(wide) >= MIN_CORR_N:
        r, p = pearsonr(wide["first"], wide["second"])
        ax.legend(handles=[Line2D([0], [0], color=HUMAN_COLOR, lw=1.5)],
                  labels=[f"Human r={r:.2f}{pvalue_to_stars(p)}"],
                  fontsize=8, frameon=True, framealpha=0.9)
    else:
        ax.text(0.02, 0.98, f"n={len(wide)} (too few for r)",
                ha="left", va="top", transform=ax.transAxes,
                fontsize=7, style="italic", color="0.5")

    ax.set_xlabel("Prefix response variability\n(first half of trials)")
    ax.set_ylabel("Prefix response variability\n(second half of trials)")
    sns.despine(ax=ax, top=True, right=True)


# ── Col 3, row 2 — Cross-task comparison ────────────────────────────────────

def _plot_panel_crosstask(ax, bin_std: pd.DataFrame, cont_std: pd.DataFrame) -> None:
    b = bin_std.set_index("prolific_pid")["resp_std"]
    c = cont_std.set_index("prolific_pid")["resp_std"]
    both = b.index.intersection(c.index)
    wide = pd.DataFrame({"binary": b[both], "continuous": c[both]})

    if len(wide) < 2:
        ax.text(0.5, 0.5, "No pids completed both tasks",
                ha="center", va="center", transform=ax.transAxes,
                color="0.5", style="italic")
        return

    ax.scatter(wide["binary"], wide["continuous"],
              color=HUMAN_COLOR, s=30, alpha=0.8, zorder=3)

    if len(wide) >= MIN_CORR_N:
        sns.regplot(data=wide, x="binary", y="continuous", ax=ax,
                    color=HUMAN_COLOR, ci=95, scatter=False,
                    line_kws={"lw": 1.5})
        r, p = pearsonr(wide["binary"], wide["continuous"])
        ax.legend(handles=[Line2D([0], [0], color=HUMAN_COLOR, lw=1.5)],
                  labels=[f"Human r={r:.2f}{pvalue_to_stars(p)}"],
                  fontsize=8, frameon=True, framealpha=0.9)
    else:
        ax.text(0.02, 0.98, f"n={len(wide)} (too few for r)",
                ha="left", va="top", transform=ax.transAxes,
                fontsize=7, style="italic", color="0.5")

    ax.set_xlabel("Prefix response variability (binary)")
    ax.set_ylabel("Prefix response variability (continuous)")
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="task_results_pilot1.pkl",
                        help="Filename under data/ produced by task/parse_results.py")
    args = parser.parse_args()

    df = pd.read_pickle(data_path(args.results_file))
    apply_style()

    fig, axes = plt.subplots(
        2, 3,
        figsize=(FIGURE_SIZE[0] * 0.75, FIGURE_SIZE[1]),
        constrained_layout=True,
    )

    prefix_std: dict[str, pd.DataFrame] = {}

    for row, task in enumerate(TASK_ROWS):
        sub = df[df["task"] == task]
        prefix_std[task] = _prefix_response_std(sub)
        split_df = _prefix_response_std_split(sub)

        _plot_panel_kde(axes[row, 0], prefix_std[task])
        _plot_panel_splithalf(axes[row, 1], split_df)
        axes[row, 0].set_title(f"task-{task}", loc="left", fontsize=9, style="italic")

    # Col 3: blank for row 1 (binary), cross-task comparison for row 2 (continuous)
    axes[0, 2].axis("off")
    _plot_panel_crosstask(axes[1, 2], prefix_std["binary"], prefix_std["continuous"])

    from utils.plot_style import label_panels
    label_panels(axes)

    fig.text(0.5, -0.02,
              "PLACEHOLDER: human only (no models fit yet). Variability restricted to "
              "observation < prefix_length=4, the only region guaranteed identical "
              "across a qid's repeats in this task's design (unlike carrabin).",
              ha="center", va="top", fontsize=7, style="italic", color="0.4")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_soltani_variability"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
