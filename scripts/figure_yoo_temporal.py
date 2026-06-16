#!/usr/bin/env python3
"""figure_yoo_temporal.py — T group figure for yoo task.

Layout: 1×4
  Panel A (T1): Estimation error vs observation; shaded bands show weak/strong U-shape range
  Panel B (T2): Mean |Δresponse| vs observation (obs ≥ 2)
  Panel C (T3): Split-half reliability of λ (first vs second half of trials)
  Panel D (T4): λ_model vs λ_human regplot (dynamical model fit)

Run:
    python scripts/figure_yoo_temporal.py
    python scripts/figure_yoo_temporal.py --run_folder yoo --nef_folder refit
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
from scipy.optimize import curve_fit as scipy_curve_fit
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
    annotate_nef_comparisons,
    fit_power_law_params,
    pvalue_to_stars,
)

MODEL_ORDER = ["Mean", "PrimacyRecency", "LeakyIntegrator", "NEF"]
HUMAN_COLOR = "0.3"
OBS_TICKS   = [5, 10, 15, 20, 25, 30]
N_GROUP     = 10
SMOOTH_WIN  = 5


def _display(model_type: str) -> str:
    return "NEF" if model_type.startswith("NEF") else model_type


def _resp_path(mt: str, run_dir: Path, nef_dir: Path) -> Path:
    d = nef_dir if mt == "NEF" else run_dir
    return d / f"{mt}_yoo_responses.pkl"


def _abs_delta_long(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for (pid, trial), g in df.groupby(["pid", "trial"], sort=False):
        g = g.sort_values("observation").copy()
        g["delta"] = g["response"].diff().abs()
        pieces.append(g)
    if not pieces:
        return pd.DataFrame(columns=["pid", "trial", "observation", "delta"])
    out = pd.concat(pieces, ignore_index=True)
    return out[out["observation"] >= 2].dropna(subset=["delta"])


def _fit_lambda_curve_fit(df: pd.DataFrame) -> pd.Series:
    def power_law(n, A, lam):
        return A * np.power(np.asarray(n, dtype=float), -lam)
    out: dict = {}
    for pid, grp in df.groupby("pid"):
        pieces = []
        for _, tg in grp.groupby("trial"):
            g = tg.sort_values("observation").copy()
            g["delta"] = g["response"].diff().abs()
            pieces.append(g)
        delta = pd.concat(pieces, ignore_index=True)
        curve = delta.groupby("observation")["delta"].mean().dropna()
        curve = curve[curve.index >= 2]
        if len(curve) < 3: continue
        n = curve.index.values.astype(float)
        y = curve.values.astype(float)
        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(y))): continue
        try:
            popt, _ = scipy_curve_fit(power_law, n, y, p0=[0.1, 0.5],
                                      bounds=([0.0, 0.0], [2.0, 2.0]), maxfev=2000)
            out[int(pid)] = float(popt[1])
        except Exception:
            pass
    return pd.Series(out, name="lambda_")


def _fit_lambda_split_half(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pid λ fitted separately on first and second half of trials."""
    rows = []
    for pid, grp in df.groupby("pid"):
        trials = sorted(grp["trial"].unique())
        mid    = len(trials) // 2
        if mid < 3:
            continue
        for half_label, trial_set in [("first", trials[:mid]), ("second", trials[mid:])]:
            sub = grp[grp["trial"].isin(trial_set)].copy()
            lam = _fit_lambda_curve_fit(sub.assign(pid=pid))
            if int(pid) in lam.index:
                rows.append({"pid": int(pid), "half": half_label,
                             "lambda_": float(lam[int(pid)])})
    if not rows:
        return pd.DataFrame(columns=["pid", "first", "second"])
    wide = (pd.DataFrame(rows)
            .pivot(index="pid", columns="half", values="lambda_")
            .dropna())
    wide.columns.name = None
    return wide.reset_index()


def _task_error_per_pid_obs(df: pd.DataFrame, value_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = df.drop(columns=["value"], errors="ignore").merge(
        value_map, on=["pid", "trial", "observation"], how="left")
    for (pid, trial), tdf in merged.groupby(["pid", "trial"]):
        tdf = tdf.sort_values("observation").copy()
        tdf = tdf[tdf["value"].notna()]
        if tdf.empty: continue
        rm = tdf["value"].expanding().mean()
        te = (tdf["response"] - rm).abs()
        for obs, t in zip(tdf["observation"], te):
            if np.isfinite(t):
                rows.append({"pid": int(pid), "observation": int(obs), "task_error": float(t)})
    if not rows:
        return pd.DataFrame(columns=["pid", "observation", "task_error"])
    return pd.DataFrame(rows).groupby(["pid","observation"], as_index=False)["task_error"].mean()


def _u_strength(te_df: pd.DataFrame) -> pd.Series:
    out = {}
    for pid, g in te_df.groupby("pid"):
        g   = g.sort_values("observation")
        obs = g["observation"].to_numpy(int)
        y   = g["task_error"].to_numpy(float)
        sm  = (pd.Series(y).rolling(SMOOTH_WIN, min_periods=1, center=True)
               .mean().to_numpy())
        late = float(np.nanmean(y[obs >= 26])) if np.any(obs >= 26) else float(np.nanmean(y))
        s    = late - float(np.nanmin(sm))
        if np.isfinite(s):
            out[int(pid)] = s
    return pd.Series(out, dtype=float)


# ── Panel A ───────────────────────────────────────────────────────────────────

def _plot_panel_a(ax, run_folder, palette, model_order, nef_folder):
    """Panel A (T1): Estimation error vs observation.

    For each source, pids are split into Q1 (bottom 25% late delta) and
    Q4 (top 25% late delta) using the same quartile split as neural panel D.
    Q1 shown as dashed line, Q4 as solid line, fill between.
    No legend entry for the band itself.
    """
    run_dir = RUNS_DIR / run_folder
    nef_dir = RUNS_DIR / nef_folder if nef_folder else run_dir
    yoo     = pd.read_pickle(data_path("yoo.pkl"))
    yoo_s   = yoo.sort_values(["pid","trial","observation"]).copy()
    yoo_s["true_mean"] = (yoo_s.groupby(["pid","trial"])["value"]
                               .expanding().mean().values)
    true_map = yoo_s[["pid","trial","observation","true_mean"]].drop_duplicates()

    LATE_OBS = range(21, 31)

    def task_rmse_per_pid_obs(df):
        m = df.drop(columns=["true_mean"], errors="ignore").merge(
            true_map, on=["pid","trial","observation"], how="left")
        return (m.assign(sq_err=(m["response"] - m["true_mean"]) ** 2)
                 .groupby(["pid","observation"])["sq_err"].mean()
                 .apply(np.sqrt).reset_index(name="rmse"))

    def quartile_split(df):
        """Return (q1_pids, q4_pids) based on mean |delta response| in obs 21-30."""
        delta_rows = []
        for (pid, trial), g in df.groupby(["pid","trial"]):
            g = g.sort_values("observation").copy()
            g["delta"] = g["response"].diff().abs()
            delta_rows.append(g[g["observation"].isin(LATE_OBS)][["pid","delta"]])
        d = pd.concat(delta_rows).dropna()
        mean_delta = d.groupby("pid")["delta"].mean().sort_values()
        q1_cut = mean_delta.quantile(0.25)
        q3_cut = mean_delta.quantile(0.75)
        q1 = set(mean_delta[mean_delta <= q1_cut].index.astype(int).tolist())
        q4 = set(mean_delta[mean_delta >= q3_cut].index.astype(int).tolist())
        return q1, q4

    handles, labels = [], []

    all_sources = [("Human", yoo, HUMAN_COLOR)] + [
        (_display(mt), pd.read_pickle(_resp_path(mt, run_dir, nef_dir)),
         palette.get(_display(mt), "0.5"))
        for mt in model_order
        if _resp_path(mt, run_dir, nef_dir).exists()
    ]

    for source_name, df, color in all_sources:
        q1_pids, q4_pids = quartile_split(df)
        if not q1_pids or not q4_pids:
            continue
        rm  = task_rmse_per_pid_obs(df)
        obs = sorted(rm["observation"].unique())
        q1_mean = rm[rm["pid"].isin(q1_pids)].groupby("observation")["rmse"].mean().reindex(obs)
        q4_mean = rm[rm["pid"].isin(q4_pids)].groupby("observation")["rmse"].mean().reindex(obs)
        ax.fill_between(obs, q1_mean.values, q4_mean.values,
                        color=color, alpha=0.18, zorder=1, linewidth=0)
        ax.plot(obs, q1_mean.values, color=color, lw=1.8, ls="--", zorder=2)
        ax.plot(obs, q4_mean.values, color=color, lw=1.8, ls="-",  zorder=2)
        handles.append(Line2D([0],[0], color=color, lw=1.8))
        labels.append(source_name)

    # Linestyle key
    handles += [Line2D([0],[0], color="0.4", lw=1.4, ls="--"),
                Line2D([0],[0], color="0.4", lw=1.4, ls="-")]
    labels  += ["Q1 (low late Δresponse)", "Q4 (high late Δresponse)"]

    ax.set_xlabel("Observation"); ax.set_ylabel("Performance error vs ground truth (RMSE)")
    ax.set_xticks(OBS_TICKS); ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel B ───────────────────────────────────────────────────────────────────

def _plot_panel_b(ax, run_folder, palette, model_order, nef_folder):
    run_dir = RUNS_DIR / run_folder
    nef_dir = RUNS_DIR / nef_folder if nef_folder else run_dir
    yoo     = pd.read_pickle(data_path("yoo.pkl"))
    handles, labels = [], []

    sns.lineplot(data=_abs_delta_long(yoo), x="observation", y="delta",
                 color=HUMAN_COLOR, lw=1.8, errorbar="ci", ax=ax, label="_nolegend_")
    handles.append(Line2D([0],[0], color=HUMAN_COLOR, lw=1.8, alpha=0.65)); labels.append("Human")

    for mt in model_order:
        rp = _resp_path(mt, run_dir, nef_dir)
        if not rp.exists(): continue
        color = palette.get(_display(mt), "0.5")
        sns.lineplot(data=_abs_delta_long(pd.read_pickle(rp)),
                     x="observation", y="delta", color=color, lw=1.8,
                     errorbar="ci", ax=ax, label="_nolegend_")
        handles.append(Line2D([0],[0], color=color, lw=1.8, alpha=0.65))
        labels.append(_display(mt))

    ax.set_xlabel("Observation"); ax.set_ylabel("Mean |Δresponse|")
    ax.set_xticks(OBS_TICKS); ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel C (T3) — Split-half reliability of λ ───────────────────────────────

def _plot_panel_c(ax, run_folder: str, nef_folder: str | None,
                  palette: dict, model_order: list) -> None:
    """Panel C (T3): Split-half reliability of the decay-rate metric λ."""
    run_dir = RUNS_DIR / run_folder
    nef_dir = RUNS_DIR / nef_folder if nef_folder else run_dir
    yoo     = pd.read_pickle(data_path("yoo.pkl"))

    EXCLUDE_C = {"Mean", "LeakyIntegrator"}
    sources = [("Human", yoo, HUMAN_COLOR)]
    for mt in model_order:
        if _display(mt) in EXCLUDE_C: continue
        rp = _resp_path(mt, run_dir, nef_dir)
        if rp.exists():
            sources.append((_display(mt), pd.read_pickle(rp),
                            palette.get(_display(mt), "0.5")))

    handles, labels = [], []
    for source_name, df, color in sources:
        wide = _fit_lambda_split_half(df)
        if len(wide) < 3:
            continue
        r, p = pearsonr(wide["first"], wide["second"])
        sns.regplot(data=wide, x="first", y="second", ax=ax,
                    color=color, ci=95, scatter=False,
                    line_kws={"lw": 1.5})
        handles.append(Line2D([0],[0], color=color, lw=1.5))
        labels.append(f"{source_name}, r={r:.2f}{pvalue_to_stars(p)}")

    ax.set_xlabel("λ (first half of trials)")
    ax.set_ylabel("λ (second half of trials)")
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel D (T4) — Lambda model vs lambda human regplot ──────────────────────

def _plot_panel_d(ax, run_folder, palette, model_order, nef_folder):
    """Panel D (T4): λ_model vs λ_human per pid, one regplot line per model."""
    run_dir = RUNS_DIR / run_folder
    nef_dir = RUNS_DIR / nef_folder if nef_folder else run_dir
    yoo     = pd.read_pickle(data_path("yoo.pkl"))
    lam_h   = _fit_lambda_curve_fit(yoo)

    EXCLUDE_D = {"Mean", "LeakyIntegrator"}
    handles, labels = [], []
    for mt in model_order:
        if _display(mt) in EXCLUDE_D: continue
        rp = _resp_path(mt, run_dir, nef_dir)
        if not rp.exists(): continue
        lam_m  = _fit_lambda_curve_fit(pd.read_pickle(rp))
        merged = pd.DataFrame({"human": lam_h, "model": lam_m}).dropna()
        if len(merged) < 3: continue
        r, p   = pearsonr(merged["human"], merged["model"])
        color  = palette.get(_display(mt), "0.5")
        sns.regplot(data=merged, x="human", y="model", ax=ax,
                    color=color, ci=95, scatter=False,
                    line_kws={"lw": 1.5})
        handles.append(Line2D([0],[0], color=color, lw=1.5))
        labels.append(f"{_display(mt)}, r={r:.2f}{pvalue_to_stars(p)}")

    lims = [ax.get_xlim(), ax.get_ylim()]
    lo   = min(lims[0][0], lims[1][0])
    hi   = max(lims[0][1], lims[1][1])
    ax.plot([lo, hi], [lo, hi], color="0.7", lw=0.8, ls="--", zorder=0)

    ax.set_xlabel("λ (human)")
    ax.set_ylabel("λ (model)")
    ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="yoo")
    parser.add_argument("--nef_folder", type=str, default=None)
    parser.add_argument("--extra_models", nargs="*", default=[])
    args = parser.parse_args()

    model_order = MODEL_ORDER + [m for m in args.extra_models if m not in MODEL_ORDER]

    apply_style()
    pal     = get_palette(len(model_order) + 1)
    palette = {m: pal[i] for i, m in enumerate(model_order)}
    for mt in model_order:
        disp = _display(mt)
        if disp not in palette:
            palette[disp] = palette[mt]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] / 2),
        constrained_layout=True,
    )

    _plot_panel_a(axes[0], args.run_folder, palette, model_order, args.nef_folder)
    _plot_panel_b(axes[1], args.run_folder, palette, model_order, args.nef_folder)
    _plot_panel_c(axes[2], args.run_folder, args.nef_folder, palette, model_order)
    _plot_panel_d(axes[3], args.run_folder, palette, model_order, args.nef_folder)

    label_panels(axes.reshape(1, -1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_yoo_temporal"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
