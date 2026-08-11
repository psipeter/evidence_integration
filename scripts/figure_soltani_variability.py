#!/usr/bin/env python3
"""figure_soltani_variability.py — V group figure for the soltani task/
pilot (task-continuous + task-binary).

Layout: 2x3
  Row 1 = task-binary, Row 2 = task-continuous (standing row-order
  convention for soltani figures -- see figure_soltani_performance.py)
  Col 1 (~carrabin V-fig panel A): group-level (within-task) KDE of
    prefix response variability, human only
  Col 2 (~carrabin V-fig panel C): test-retest reliability -- one point
    per pid, x/y = prefix response variability in the 1st/2nd half of
    trials.
  Col 3: cross-task comparison (pids who did both tasks) -- binary prefix
    variability on x, continuous on y.

ROW 1 (binary/colors) AND COL 3 NOW USE QUASI-QIDS
------------------------------------------------------------------------
"Prefix response variability" is inherently about repeated exposure to an
IDENTICAL prefix across a qid's repeats -- colors' own literal `qid`
column never repeats (confirmed directly this session), so a DIFFERENT,
empirically-derived repeat structure is used instead: see
utils/colors_quasi_qids.py's own module docstring for the full
definition (group a participant's trials by their own literal first-4
raw stimulus values, keep only groups with >=3 repeats) and the
empirical sweep that settled its defaults. This is the same mechanism
figure_soltani_temporal.py's columns 3-4 now use for colors -- not a
separate, independently-invented one.

WHY "PREFIX" VARIABILITY, NOT "QID" VARIABILITY (numbers/continuous)
------------------------------------------------------------------------
carrabin's qid repeats show an IDENTICAL sequence every time, so response
variability can be computed at any observation index. Numbers' qid
repeats do NOT: prefix identity and target level are independent axes (see
docs/HISTORY.md's "Sequence generation methods" section) -- a qid's 4
repeats share the same first `prefix_length` (=4) observations, but the
SUFFIX differs on every repeat because each repeat is steered toward a
different target. So "response variability for identical inputs" is only
a valid concept when restricted to `observation < prefix_length`.
PREFIX_LENGTH is hardcoded to 4 below (matching task_backend/
generate_sequences.py's own NUMBERS_N_PREFIX*-derived design, confirmed
against a real pool sequence file's own prefix_length field before
hardcoding it here, same as this file's own earlier pass already did).

DATA SOURCE
-----------
data/task_continuous.pkl -- built by scripts/build_task_backend_inputs.py
(pulls real, finished participants directly from task_backend's Supabase
`events` table) via scripts/build_model_inputs.py's own build_from_df().
Participant filtering and the prolific_pid -> int pid mapping already
happened when that file was built, so this script does not re-apply
utils.participant_filters or see any real prolific_pid at all -- this is
a change from this file's own earlier version, which read a raw
task_results_pilot*.pkl and applied filter_participants itself.

SAMPLE-SIZE CAVEATS (read before interpreting)
------------------------------------------------
- No minimum-trials-per-cell threshold is enforced -- a (qid, observation)
  cell's std is computed from however many of the 4 repeats survived
  (pandas .std() already returns NaN, dropped, below n=2). With only 4
  repeats per qid total, these per-cell stds are individually noisy; only
  the pid-level average (over many cells) is at all stable, and even that
  is based on very few pids so far.
- The split-half panel (col 2) splits a pid's OWN trial list in half by
  trial index; a qid's repeats are not guaranteed to distribute evenly
  across the split, so a half's prefix-variability estimate may rest on
  as few as 1-2 (qid, obs) cells.

Run:
    python scripts/figure_soltani_variability.py
"""
from __future__ import annotations

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
from utils.plot_style import FIGURE_SIZE, apply_style, label_panels, pvalue_to_stars
from utils.colors_quasi_qids import add_quasi_qids

PREFIX_LENGTH = 4
HUMAN_COLOR   = "0.3"
MIN_CORR_N    = 3  # matches figure_soltani_temporal.py's cross-task correlation threshold


# ── metric helpers (both tasks -- colors' df must be pre-processed via
# utils.colors_quasi_qids.add_quasi_qids first, see module docstring) ────

def _prefix_response_std(df: pd.DataFrame) -> pd.DataFrame:
    """Mean std(response | qid, observation) within the prefix region,
    per pid. One row per pid; columns [pid, resp_std]. No timed_out/dedup
    filtering here -- both data/task_continuous.pkl and data/task_binary.pkl
    are already deduped to successful attempts only (see
    build_model_inputs.py's build_from_df), unlike this file's own earlier
    version, which read a raw,
    not-yet-deduped pilot file."""
    sub = df[df["observation"] < PREFIX_LENGTH]
    grp = (sub.groupby(["pid", "qid", "observation"])["response"]
           .std().dropna().reset_index(name="resp_std"))
    return grp.groupby("pid")["resp_std"].mean().reset_index()


def _prefix_response_std_split(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pid prefix response std computed separately on ODD vs EVEN
    trial indices, not first-half/second-half -- a strict chronological
    split confounds genuine estimation noise (what split-half reliability
    is meant to measure) with any systematic drift in behavior over the
    session (learning, fatigue, boredom); interleaving odd/even trials
    samples both halves from the same span of session-time, isolating
    noise from drift (see chat history). Columns [pid, first, second]
    (labels kept for column-naming consistency with the rest of this
    file/figure_soltani_temporal.py -- 'first'=odd-indexed, 'second'=
    even-indexed trials)."""
    rows = []
    for pid, g in df.groupby("pid"):
        trials = sorted(g["trial"].unique())
        halves = {"first": trials[0::2], "second": trials[1::2]}
        vals = {}
        for half, tset in halves.items():
            gg = g[(g["trial"].isin(tset)) & (g["observation"] < PREFIX_LENGTH)]
            pv = gg.groupby(["qid", "observation"])["response"].std().dropna()
            vals[half] = float(pv.mean()) if len(pv) > 0 else np.nan
        rows.append({"pid": pid, **vals})
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

    ax.set_xlabel("Prefix response variability\n(odd-indexed trials)")
    ax.set_ylabel("Prefix response variability\n(even-indexed trials)")
    sns.despine(ax=ax, top=True, right=True)


# ── Col 3 — Cross-task comparison ──────────────────────────

def _plot_panel_crosstask(ax, bin_std: pd.DataFrame, cont_std: pd.DataFrame) -> None:
    """bin_std/cont_std: each [pid, resp_std] (bin_std computed on
    colors' quasi-qid-restricted data, cont_std on numbers' real qid --
    see module docstring). Merges on the real integer `pid`, which is
    valid either way -- the quasi-qid relabeling only ever touches which
    ROWS/trials qualify and what to call the derived group, never the
    underlying participant identity itself."""
    b = bin_std.set_index("pid")["resp_std"]
    c = cont_std.set_index("pid")["resp_std"]
    both = b.index.intersection(c.index)
    wide = pd.DataFrame({"binary": b[both], "continuous": c[both]})

    if len(wide) < 2:
        msg = ("No pids completed both tasks" if len(wide) == 0
              else f"Only {len(wide)} pid completed both tasks (need >=2 to plot)")
        ax.text(0.5, 0.5, msg,
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
    apply_style()

    fig, axes = plt.subplots(
        2, 3,
        figsize=(FIGURE_SIZE[0] * 0.75, FIGURE_SIZE[1]),
        constrained_layout=True,
    )

    prefix_std: dict[str, pd.DataFrame] = {}

    # Row 0 = binary/colors -- via quasi-qids (see module docstring).
    df_binary = pd.read_pickle(data_path("task_binary.pkl"))
    df_binary_qq = add_quasi_qids(df_binary)
    print(f"task-binary: {len(df_binary)} rows, {df_binary['pid'].nunique()} pids "
          f"-> {len(df_binary_qq)} rows in a qualifying quasi-qid group")
    prefix_std["binary"] = _prefix_response_std(df_binary_qq)
    split_df_binary = _prefix_response_std_split(df_binary_qq)

    _plot_panel_kde(axes[0, 0], prefix_std["binary"])
    _plot_panel_splithalf(axes[0, 1], split_df_binary)
    axes[0, 0].set_title("task-binary", loc="left", fontsize=9, style="italic")

    # Row 1 = continuous/numbers -- real qid, unchanged.
    df_continuous = pd.read_pickle(data_path("task_continuous.pkl"))
    print(f"task-continuous: {len(df_continuous)} rows, {df_continuous['pid'].nunique()} pids")
    prefix_std["continuous"] = _prefix_response_std(df_continuous)
    split_df_continuous = _prefix_response_std_split(df_continuous)

    _plot_panel_kde(axes[1, 0], prefix_std["continuous"])
    _plot_panel_splithalf(axes[1, 1], split_df_continuous)
    axes[1, 0].set_title("task-continuous", loc="left", fontsize=9, style="italic")

    # Col 3: cross-task comparison, now possible for both rows since both
    # tasks have a valid prefix-variability metric -- but colors' pid
    # values are quasi-qid-restricted-trial-derived, computed on the SAME
    # underlying pid identifiers either way, so the merge in
    # _plot_panel_crosstask (on real integer pid) is still valid.
    axes[0, 2].axis("off")
    _plot_panel_crosstask(axes[1, 2], prefix_std["binary"], prefix_std["continuous"])

    label_panels(axes)

    fig.text(0.5, -0.02,
              "Human only (no models fit yet). task-binary uses an empirically-derived "
              "quasi-qid repeat structure (see this script's own module docstring); "
              "task-continuous uses its real, designed qid repeats. Both restricted to "
              "observation < prefix_length=4.",
              ha="center", va="top", fontsize=7, style="italic", color="0.4")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_soltani_variability"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
