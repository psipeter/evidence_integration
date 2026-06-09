"""Archive: Panel E (N5) — Lag-1 autocorrelation vs n_neurons.

Archived from figure_carrabin_neural.py.
Shows NEF lag-1 autocorrelation mean ± SD across pids vs n_neurons,
with individual human pid values as thin grey reference lines.
Findings: NEF autocorrelation consistently above all human pids at every
n_neurons level, even at n=400 (~0.68 vs human max ~0.84).
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import pearsonr


def _compute_lag1_r(resp_df: pd.DataFrame) -> pd.DataFrame:
    """Per-pid lag-1 autocorrelation of within-sequence response residuals."""
    df = resp_df.copy()
    df["resid"] = df["response"] - df.groupby(
        ["pid", "observation", "qid"])["response"].transform("mean")
    rows = []
    for pid, g in df.groupby("pid"):
        pairs = []
        for trial, tg in g.groupby("trial"):
            r = tg.sort_values("observation")["resid"].values
            if len(r) > 1:
                pairs.extend(zip(r[:-1], r[1:]))
        if len(pairs) >= 3:
            arr = np.array(pairs)
            rv, _ = pearsonr(arr[:, 0], arr[:, 1])
            rows.append({"pid": int(pid), "lag1_r": float(rv)})
    return pd.DataFrame(rows)


def _plot_panel_e(ax, run_folder: str, data_path, RUNS_DIR,
                  get_palette, Line2D) -> None:
    """Panel E (N5): Lag-1 autocorrelation of residuals vs n_neurons.

    NEF: mean ± SD across pids per n_neurons from n_neurons_scan_metrics.pkl.
    Human: each pid's lag-1 r as a thin grey horizontal reference line.
    y-axis starts at 0.4.
    """
    from pathlib import Path
    run_dir      = data_path("runs") / run_folder
    metrics_path = run_dir / "n_neurons_scan_metrics.pkl"
    if not metrics_path.exists():
        ax.text(0.5, 0.5, "No n_neurons scan metrics",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="0.5", style="italic")
        return

    scan_m = pd.read_pickle(metrics_path)
    n_vals = sorted(scan_m["n_neurons"].unique())

    human      = pd.read_pickle(data_path("carrabin.pkl"))
    human_lag1 = _compute_lag1_r(human)

    pal   = get_palette(6)
    color = pal[0]

    for _, row in human_lag1.iterrows():
        ax.axhline(row["lag1_r"], color="0.78", lw=0.3, zorder=0)

    stats = (scan_m.groupby("n_neurons")["lag1_r"]
               .agg(["mean", "std"]).reset_index())
    ax.plot(stats["n_neurons"], stats["mean"],
            color=color, lw=1.8, marker="o", ms=5, label="NEF")
    ax.fill_between(stats["n_neurons"],
                    stats["mean"] - stats["std"],
                    stats["mean"] + stats["std"],
                    color=color, alpha=0.18)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="0.78", lw=0.8))
    labels.append("Human (individual)")

    ax.set_ylim(bottom=0.4)
    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Lag-1 autocorrelation")
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(n) for n in n_vals])
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)
