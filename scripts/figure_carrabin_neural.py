#!/usr/bin/env python3
"""figure_carrabin_neural.py — N group figure for carrabin task.

Layout: 1×4
  Panel A (N1): Decoded PE dynamics for 4 param combinations
  Panel B (N2): Std PE vs std response — per-pid, both from same probe simulations
  Panel C (N3): Response variability and PE variability vs n_neurons (n_neurons_scan)
  Panel D (N4): Response variability growth (slope_c) vs n_neurons — NEF + human refs

Run:
    python scripts/figure_carrabin_neural.py --run_folder carrabin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, linregress

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
    pvalue_to_stars,
)

DOWNSAMPLE     = 5
MIN_REPEATS    = 5
READOUT_OFFSET = 0.5
OBS_VALS       = [1, 2, 3, 4, 5]
MIN_STD        = 3


def _placeholder(ax, text: str) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, text, ha="center", va="center",
            transform=ax.transAxes, color="0.5", style="italic", fontsize=8)


def _qid_response_std(resp_df: pd.DataFrame, qid_map: pd.DataFrame,
                      min_trials: int = 3) -> pd.DataFrame:
    """Mean of std(response | obs, qid) per pid."""
    df = resp_df.drop(columns=["qid"], errors="ignore").merge(
        qid_map, on=["pid", "trial", "observation"])
    grp = (
        df.groupby(["pid", "observation", "qid"])["response"]
        .apply(lambda x: x.std() if len(x) >= min_trials else np.nan)
        .dropna()
        .reset_index(name="resp_std")
    )
    return grp.groupby("pid")["resp_std"].mean().reset_index()


def _compute_slope_c(resp_df: pd.DataFrame, min_cells: int = MIN_STD) -> pd.DataFrame:
    """Per-pid slope of mean residual-std vs observation position."""
    df = resp_df.copy()
    df["resid"] = df["response"] - df.groupby(
        ["pid", "observation", "qid"])["response"].transform("mean")
    rows = []
    for pid, g in df.groupby("pid"):
        obs_std = []
        for obs in OBS_VALS:
            cell_stds = (g[g["observation"] == obs]
                          .groupby("qid")["resid"]
                          .apply(lambda x: x.std() if len(x) >= min_cells else np.nan)
                          .dropna())
            obs_std.append(float(cell_stds.mean()) if len(cell_stds) > 0 else np.nan)
        obs_std = np.array(obs_std)
        valid   = ~np.isnan(obs_std)
        if valid.sum() >= 3:
            slope, *_ = linregress(np.array(OBS_VALS)[valid], obs_std[valid])
            rows.append({"pid": int(pid), "slope_c": float(slope)})
    return pd.DataFrame(rows)


# ── Panel A (N1) — Decoded PE dynamics ───────────────────────────────────────

def _plot_panel_a(ax, run_folder: str) -> None:
    """Panel A (N1): Decoded PE timeseries — 4 alpha_0 × n_neurons combinations."""
    run_dir = data_path("runs") / run_folder
    files   = sorted(run_dir.glob("pe_dynamics_NEF_carrabin_a*.pkl"))
    if not files:
        _placeholder(ax, "No PE dynamics data")
        return

    combined = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    combined = combined[
        combined["t"].apply(lambda t: round(t * 1000) % DOWNSAMPLE == 0)
    ].copy()

    pal         = get_palette(6)
    alpha_vals  = sorted(combined["alpha_0"].unique())
    n_vals      = sorted(combined["n_neurons"].unique())
    combos      = [(a, n) for a in alpha_vals for n in n_vals]
    combo_color = {c: pal[i] for i, c in enumerate(combos)}

    handles, labels = [], []
    for alpha_0, n_neurons in combos:
        sub   = combined[(combined["alpha_0"] == alpha_0) &
                          (combined["n_neurons"] == n_neurons)]
        if sub.empty: continue
        stats = sub.groupby("t")["pe_product"].agg(["mean", "sem"]).reset_index()
        color = combo_color[(alpha_0, n_neurons)]
        ax.plot(stats["t"], stats["mean"], color=color, lw=1.8)
        ax.fill_between(stats["t"],
                        stats["mean"] - stats["sem"],
                        stats["mean"] + stats["sem"],
                        color=color, alpha=0.18, linewidth=0)
        handles.append(Line2D([0], [0], color=color, lw=2))
        labels.append(f"\u03b1\u2080={alpha_0}, n={n_neurons}")

    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for x, lbl in [(0.5, "PE\nmeasured at"), (1.5, "Response\nmeasured at")]:
        ax.axvline(x, color="0.4", lw=1.0, ls="--", zorder=0)
        ax.text(x, 1.02, lbl, transform=trans,
                ha="center", va="bottom", clip_on=False,
                fontsize=7, color="0.4")

    ax.set_xlabel("Time within observation (s)")
    ax.set_ylabel("Decoded PE  \u03b1(t) \u00d7 (obs \u2212 value)")
    ax.set_xlim(0, 1.55)
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9,
              ncol=2, loc="upper right")
    sns.despine(ax=ax, top=True, right=True)


# ── Panel B (N2) — Std PE vs std response ────────────────────────────────────

def _load_panel_b_metrics(run_folder: str) -> pd.DataFrame:
    refit_dir  = RUNS_DIR / "refit"
    probe_path = refit_dir / "probe_pids_carrabin.pkl"
    if not probe_path.exists():
        return pd.DataFrame()

    human      = pd.read_pickle(data_path("carrabin.pkl"))
    all_probes = pd.read_pickle(probe_path)
    qid_lookup = human.set_index(["pid", "trial", "observation"])["qid"]

    rows = []
    for probe in all_probes:
        pid    = int(probe["pid"])
        trial  = int(probe["trial"])
        t      = np.array(probe["t"])
        error  = probe["error"]
        value  = np.array(probe["value"]).ravel()
        params = probe["params"]
        t_iti  = float(params["t_iti"])
        t_obs_ = float(params["t_obs"])
        t_step = t_obs_ + t_iti
        n_obs  = int(round((t[-1] + t[1] - t[0]) / t_step))

        for n in range(1, n_obs + 1):
            idx_pe   = int(np.argmin(np.abs(t - (t_iti + (n-1)*t_step + READOUT_OFFSET))))
            idx_resp = int(np.argmin(np.abs(t - (t_iti + (n-1)*t_step + t_obs_))))
            pe_val       = abs(float(error[idx_pe, 1]))
            response_val = float(value[idx_resp]) * n / (n + 2)
            try:
                qid = qid_lookup.loc[(pid, trial, n)]
            except KeyError:
                continue
            rows.append({"pid": pid, "trial": trial, "obs": n,
                          "qid": qid, "pe": pe_val, "response": response_val})

    flat = pd.DataFrame(rows)
    agg_rows = []
    for (pid, qid, obs), g in flat.groupby(["pid", "qid", "obs"]):
        if len(g) < MIN_REPEATS:
            continue
        agg_rows.append({"pid": int(pid),
                          "pe_std":   float(g["pe"].std()),
                          "resp_std": float(g["response"].std())})

    agg = pd.DataFrame(agg_rows)
    if agg.empty:
        return pd.DataFrame()
    return agg.groupby("pid")[["pe_std", "resp_std"]].mean().reset_index()


def _plot_panel_b(ax, run_folder: str) -> None:
    df = _load_panel_b_metrics(run_folder)
    if df.empty:
        _placeholder(ax, "No probe data")
        return

    r, p  = pearsonr(df["pe_std"], df["resp_std"])
    color = get_palette(6)[0]

    ax.scatter(df["pe_std"], df["resp_std"],
               color=color, s=35, alpha=0.85, zorder=3,
               label="Mean across sequences for one participant")
    sns.regplot(data=df, x="pe_std", y="resp_std", ax=ax,
                color=color, ci=95, scatter=False,
                line_kws={"lw": 1.8},
                label=f"Group-level correlation, r={r:.2f}{pvalue_to_stars(p)}")

    ax.set_xlabel("PE variability (within sequence)")
    ax.set_ylabel("Response variability (within sequence)")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel C (N3) — Response and PE variability vs n_neurons ──────────────────

def _plot_panel_c(ax, run_folder: str) -> None:
    """Panel C (N3): Response variability and PE variability vs n_neurons."""
    run_dir   = data_path("runs") / run_folder
    scan_path = run_dir / "n_neurons_scan.pkl"
    if not scan_path.exists():
        _placeholder(ax, "No n_neurons scan data")
        return

    scan_raw = pd.read_pickle(scan_path)
    if not isinstance(scan_raw, dict) or not all(
        isinstance(v, dict) and "responses" in v for v in scan_raw.values()
    ):
        _placeholder(ax, "Rerun n_neurons scan")
        return

    human      = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map    = human[["pid", "trial", "observation", "qid"]].drop_duplicates()
    human_vals = _qid_response_std(human, qid_map)["resp_std"].values

    pal       = get_palette(6)
    color_var = pal[0]
    color_pe  = pal[1]

    rows = []
    for n_neurons, nd in scan_raw.items():
        resp_df = nd["responses"]
        pe_df   = nd["pe_readout"]
        for pid, rg in resp_df.groupby("pid"):
            resp_std = float(
                rg.groupby(["observation", "qid"])["response"]
                .apply(lambda x: x.std() if len(x) >= 3 else np.nan)
                .dropna().mean()
            )
            pe_std = float(
                pe_df[pe_df["pid"] == pid]
                .groupby(["observation", "qid"])["pe"]
                .apply(lambda x: x.std() if len(x) >= 3 else np.nan)
                .dropna().mean()
            )
            rows.append({"n_neurons": n_neurons, "pid": pid,
                         "resp_std": resp_std, "pe_std": pe_std})

    if not rows:
        _placeholder(ax, "No scan data"); return

    scan_df = pd.DataFrame(rows)
    n_vals  = sorted(scan_df["n_neurons"].unique())

    for hv in human_vals:
        ax.axhline(hv, color="0.78", lw=0.3, zorder=0)

    sns.lineplot(data=scan_df, x="n_neurons", y="resp_std",
                 color=color_var, lw=1.8, errorbar="sd", err_style="band",
                 label="NEF response variability", ax=ax)
    sns.lineplot(data=scan_df, x="n_neurons", y="pe_std",
                 color=color_pe, lw=1.8, errorbar="sd", err_style="band",
                 label="NEF PE variability", ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="0.78", lw=0.8))
    labels.append("Human response variability")

    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Variability")
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(n) for n in n_vals])
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9,
              loc="upper right")
    sns.despine(ax=ax, top=True, right=True)


# ── Panel D (N4) — Response variability growth (slope_c) vs n_neurons ────────

def _plot_panel_d(ax, run_folder: str) -> None:
    """Panel D (N4): Slope of response variability growth vs n_neurons.

    NEF: mean ± SD across pids at each n_neurons level from n_neurons_scan.
    Human: each pid's slope_c drawn as a thin grey horizontal reference line.
    """
    run_dir      = data_path("runs") / run_folder
    metrics_path = run_dir / "n_neurons_scan_metrics.pkl"

    if not metrics_path.exists():
        _placeholder(ax, "No n_neurons scan metrics"); return

    scan_m = pd.read_pickle(metrics_path)
    n_vals = sorted(scan_m["n_neurons"].unique())

    human        = pd.read_pickle(data_path("carrabin.pkl"))
    human_slopes = _compute_slope_c(human)

    pal   = get_palette(6)
    color = pal[0]

    for _, row in human_slopes.iterrows():
        ax.axhline(row["slope_c"], color="0.78", lw=0.3, zorder=0)

    stats = (scan_m.groupby("n_neurons")["slope_c"]
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

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Response variability growth")
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(n) for n in n_vals])
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="carrabin")
    args = parser.parse_args()

    apply_style()

    fig, axes = plt.subplots(
        1, 4,
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] / 2),
        constrained_layout=True,
    )

    _plot_panel_a(axes[0], args.run_folder)
    _plot_panel_b(axes[1], args.run_folder)
    _plot_panel_c(axes[2], args.run_folder)
    _plot_panel_d(axes[3], args.run_folder)

    label_panels(axes.reshape(1, -1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_carrabin_neural"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
