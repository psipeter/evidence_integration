#!/usr/bin/env python3
"""figure_carrabin_temporal.py — T group figure for carrabin task.

Layout: 1x4
  Panel A (T1): Task performance vs observation — RMSE to true_p per obs
  Panel B (T2): Response change vs observation — mean |Δresponse| per obs
  Panel C (T4): Within-trial residual autocorrelation decay (lag 1-4)
  Panel D (T3): Residual variance growth across observations

Stochastic sources for C/D: Human, NEF (RMSE-fitted), NoisyCounting (MLE-fitted)

Run:
    python scripts/figure_carrabin_temporal.py --run_folder carrabin
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
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
)

MODEL_ORDER  = ["Mean", "PrimacyRecency", "LeakyIntegrator", "NEF"]
HUMAN_COLOR  = "0.3"


def _display(model_type: str) -> str:
    return "NEF" if model_type.startswith("NEF") else model_type


def _add_resid(df: pd.DataFrame) -> pd.DataFrame:
    """Add resid column = response - mean(response | pid, obs, qid)."""
    means = (df.groupby(["pid", "observation", "qid"])["response"]
               .mean().reset_index().rename(columns={"response": "qid_mean"}))
    df2 = df.merge(means, on=["pid", "observation", "qid"])
    df2["resid"] = df2["response"] - df2["qid_mean"]
    return df2


def _load_with_qid(path: Path, qid_map: pd.DataFrame) -> pd.DataFrame:
    """Load response file and ensure qid column is present."""
    df = pd.read_pickle(path).drop(columns=["qid"], errors="ignore")
    return df.merge(qid_map, on=["pid", "trial", "observation"])


# ── Panel A (T1) — Task performance vs observation ────────────────────────────

def _plot_panel_a(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel A (T1): RMSE to true_p as a function of observation position."""
    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))
    human["true_p_resp"] = human["true_p"] * 2 - 1
    true_p_map = human[["pid", "trial", "observation", "true_p_resp"]].drop_duplicates()
    obs_vals   = sorted(human["observation"].unique())
    handles, labels = [], []

    h_rmse = (human.assign(sq_err=(human["response"] - human["true_p_resp"]) ** 2)
              .groupby(["pid", "observation"])["sq_err"].mean()
              .apply(np.sqrt).reset_index(name="rmse"))
    stats = h_rmse.groupby("observation")["rmse"].agg(["mean", "sem"]).reset_index()
    ax.plot(stats["observation"], stats["mean"], "o-", color=HUMAN_COLOR, lw=1.8, ms=5)
    ax.fill_between(stats["observation"],
                    stats["mean"] - stats["sem"], stats["mean"] + stats["sem"],
                    color=HUMAN_COLOR, alpha=0.2)
    handles.append(Line2D([0], [0], color=HUMAN_COLOR, lw=1.5)); labels.append("Human")

    for mt in model_order:
        if mt == "NEF":
            resp_path = run_dir / "NEF_carrabin_responses_mle.pkl"
        else:
            resp_path = run_dir / f"{mt}_carrabin_responses.pkl"
        if not resp_path.exists(): continue
        mdf    = pd.read_pickle(resp_path).merge(
            true_p_map, on=["pid", "trial", "observation"], how="left")
        m_rmse = (mdf.assign(sq_err=(mdf["response"] - mdf["true_p_resp"]) ** 2)
                  .groupby(["pid", "observation"])["sq_err"].mean()
                  .apply(np.sqrt).reset_index(name="rmse"))
        stats_m = m_rmse.groupby("observation")["rmse"].agg(["mean", "sem"]).reset_index()
        color = palette.get(_display(mt), "0.5")
        ax.plot(stats_m["observation"], stats_m["mean"], "o-", color=color, lw=1.8, ms=5)
        ax.fill_between(stats_m["observation"],
                        stats_m["mean"] - stats_m["sem"], stats_m["mean"] + stats_m["sem"],
                        color=color, alpha=0.2)
        lbl = "NEF (MLE)" if mt == "NEF" else _display(mt)
        handles.append(Line2D([0], [0], color=color, lw=1.5)); labels.append(lbl)

    # Add NoisyCounting MLE explicitly
    nc_mle_path = run_dir / "NoisyCounting_carrabin_responses_mle.pkl"
    if nc_mle_path.exists():
        mdf = pd.read_pickle(nc_mle_path).merge(
            true_p_map, on=["pid", "trial", "observation"], how="left")
        m_rmse = (mdf.assign(sq_err=(mdf["response"] - mdf["true_p_resp"]) ** 2)
                  .groupby(["pid", "observation"])["sq_err"].mean()
                  .apply(np.sqrt).reset_index(name="rmse"))
        stats_m = m_rmse.groupby("observation")["rmse"].agg(["mean", "sem"]).reset_index()
        color = palette.get("NoisyCounting_mle", palette.get("NoisyCounting (MLE)", "0.5"))
        ax.plot(stats_m["observation"], stats_m["mean"], "o-",
                color=color, lw=1.8, ms=5)
        ax.fill_between(stats_m["observation"],
                        stats_m["mean"] - stats_m["sem"], stats_m["mean"] + stats_m["sem"],
                        color=color, alpha=0.2)
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append("NoisyCounting (MLE)")

    ax.set_xlabel("Observation")
    ax.set_ylabel("Estimation error (RMSE to hidden probability)")
    ax.set_xticks(obs_vals); ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel B (T2) — Response change vs observation ─────────────────────────────

def _plot_panel_b(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel B (T2): Mean |Δresponse| as a function of observation position."""
    run_dir  = data_path("runs") / run_folder
    human    = pd.read_pickle(data_path("carrabin.pkl"))
    obs_vals = sorted(human["observation"].unique())
    handles, labels = [], []

    def abs_delta(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (pid, trial), g in df.groupby(["pid", "trial"]):
            g = g.sort_values("observation").copy()
            g["delta"] = g["response"].diff().abs()
            g.loc[g["observation"] == g["observation"].min(), "delta"] = (
                g.loc[g["observation"] == g["observation"].min(), "response"].abs())
            rows.append(g[["pid", "trial", "observation", "delta"]])
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    h_delta = abs_delta(human).dropna()
    stats_h = (h_delta.groupby(["pid", "observation"])["delta"].mean().reset_index()
               .groupby("observation")["delta"].agg(["mean", "sem"]).reset_index())
    ax.plot(stats_h["observation"], stats_h["mean"], "o-", color=HUMAN_COLOR, lw=1.8, ms=5)
    ax.fill_between(stats_h["observation"],
                    stats_h["mean"] - stats_h["sem"], stats_h["mean"] + stats_h["sem"],
                    color=HUMAN_COLOR, alpha=0.2)
    handles.append(Line2D([0], [0], color=HUMAN_COLOR, lw=1.5)); labels.append("Human")

    for mt in model_order:
        if mt == "NEF":
            resp_path = run_dir / "NEF_carrabin_responses_mle.pkl"
        else:
            resp_path = run_dir / f"{mt}_carrabin_responses.pkl"
        if not resp_path.exists(): continue
        m_delta = abs_delta(pd.read_pickle(resp_path)).dropna()
        stats_m = (m_delta.groupby(["pid", "observation"])["delta"].mean().reset_index()
                   .groupby("observation")["delta"].agg(["mean", "sem"]).reset_index())
        color = palette.get(_display(mt), "0.5")
        ax.plot(stats_m["observation"], stats_m["mean"], "o-", color=color, lw=1.8, ms=5)
        ax.fill_between(stats_m["observation"],
                        stats_m["mean"] - stats_m["sem"], stats_m["mean"] + stats_m["sem"],
                        color=color, alpha=0.2)
        lbl = "NEF (MLE)" if mt == "NEF" else _display(mt)
        handles.append(Line2D([0], [0], color=color, lw=1.5)); labels.append(lbl)

    # Add NoisyCounting MLE explicitly
    nc_mle_path = run_dir / "NoisyCounting_carrabin_responses_mle.pkl"
    if nc_mle_path.exists():
        m_delta = abs_delta(pd.read_pickle(nc_mle_path)).dropna()
        stats_m = (m_delta.groupby(["pid", "observation"])["delta"].mean().reset_index()
                   .groupby("observation")["delta"].agg(["mean", "sem"]).reset_index())
        color = palette.get("NoisyCounting_mle", palette.get("NoisyCounting (MLE)", "0.5"))
        ax.plot(stats_m["observation"], stats_m["mean"], "o-",
                color=color, lw=1.8, ms=5)
        ax.fill_between(stats_m["observation"],
                        stats_m["mean"] - stats_m["sem"], stats_m["mean"] + stats_m["sem"],
                        color=color, alpha=0.2)
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append("NoisyCounting (MLE)")

    ax.set_xlabel("Observation")
    ax.set_ylabel("Mean |Δresponse|")
    ax.set_xticks(obs_vals); ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel C (T4) — Within-trial residual autocorrelation ──────────────────────

def _plot_panel_c(ax, run_folder: str, palette: dict) -> None:
    """Panel C (T4): Within-trial lag-k residual autocorrelation (lag 1-4).

    Shows Human, NEF (RMSE-fitted), and NoisyCounting (MLE-fitted).
    State noise signature: positive autocorrelation decaying with lag.
    NoisyCounting RMSE-fitted has near-zero autocorrelation (response noise).
    NoisyCounting MLE-fitted should show intermediate autocorrelation since
    MLE correctly recovers sigma_c > 0.
    """
    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()

    # Build sources: Human, NEF (RMSE), NoisyCounting (MLE)
    sources = []
    sources.append((_add_resid(human), HUMAN_COLOR, "Human"))

    nef_path = run_dir / "NEF_carrabin_responses_mle.pkl"
    if nef_path.exists():
        nef2 = _add_resid(_load_with_qid(nef_path, qid_map))
        sources.append((nef2, palette.get("NEF", "0.5"), "NEF (MLE)"))

    nc_mle_path = run_dir / "NoisyCounting_carrabin_responses_mle.pkl"
    if nc_mle_path.exists():
        nc_mle2 = _add_resid(_load_with_qid(nc_mle_path, qid_map))
        sources.append((nc_mle2, palette.get("NoisyCounting_mle",
                                              palette.get("NoisyCounting", "0.6")),
                        "NoisyCounting (MLE)"))

    lags    = [1, 2, 3]  # lag-4 is degenerate: (obs1, obs5) residuals near-zero for some pids
    handles, labels = [], []
    for df, color, src in sources:
        # Compute lag-k autocorrelation per pid, then mean ± SEM across pids
        # (consistent with SEM error bars in panels A and B)
        pid_rs = {lag: [] for lag in lags}
        for pid, pid_df in df.groupby("pid"):
            for lag in lags:
                pairs = []
                for (_, trial), g in pid_df.groupby(["pid", "trial"]):
                    r = g.sort_values("observation")["resid"].values
                    if len(r) > lag:
                        pairs.extend(zip(r[:-lag], r[lag:]))
                if len(pairs) < 3:
                    continue
                arr = np.array(pairs)
                rv, _ = pearsonr(arr[:, 0], arr[:, 1])
                pid_rs[lag].append(rv)

        means = [np.mean(pid_rs[lag]) for lag in lags]
        sems  = [np.std(pid_rs[lag]) / np.sqrt(len(pid_rs[lag]))
                 for lag in lags]
        means_arr = np.array(means)
        sems_arr  = np.array(sems)

        ax.plot(lags, means_arr, "o-", color=color, lw=1.8, ms=5)
        ax.fill_between(lags,
                        means_arr - sems_arr,
                        means_arr + sems_arr,
                        color=color, alpha=0.2)
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append(src)

    ax.axhline(0, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("Lag (observations)")
    ax.set_ylabel("Autocorrelation of trial-to-trial deviations")
    ax.set_xticks(lags)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel D (T3) — Residual variance growth ───────────────────────────────────

def _plot_panel_d(ax, run_folder: str, palette: dict) -> None:
    """Panel D (T3): Residual std as a function of observation position.

    State noise accumulation: std(resid | obs, qid) grows with obs if noise
    is injected into the cognitive state at each update.
    Shows Human, NEF (RMSE-fitted), NoisyCounting (MLE-fitted).
    NoisyCounting RMSE-fitted should be flat (pure response noise).
    """
    MIN     = 3
    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()
    obs_vals = sorted(human["observation"].unique())

    sources = []
    sources.append((_add_resid(human), HUMAN_COLOR, "Human"))

    nef_path = run_dir / "NEF_carrabin_responses_mle.pkl"
    if nef_path.exists():
        nef2 = _add_resid(_load_with_qid(nef_path, qid_map))
        sources.append((nef2, palette.get("NEF", "0.5"), "NEF (MLE)"))

    nc_mle_path = run_dir / "NoisyCounting_carrabin_responses_mle.pkl"
    if nc_mle_path.exists():
        nc_mle2 = _add_resid(_load_with_qid(nc_mle_path, qid_map))
        sources.append((nc_mle2, palette.get("NoisyCounting_mle",
                                              palette.get("NoisyCounting", "0.6")),
                        "NoisyCounting (MLE)"))

    handles, labels = [], []
    for df, color, src in sources:
        grp = (df.groupby(["pid", "observation", "qid"])["resid"]
                 .apply(lambda x: x.std() if len(x) >= MIN else np.nan)
                 .dropna().reset_index(name="std"))
        by_pid_obs = grp.groupby(["pid", "observation"])["std"].mean().reset_index()
        stats = by_pid_obs.groupby("observation")["std"].agg(["mean", "std"]).reset_index()
        n_pid = by_pid_obs["pid"].nunique()
        stats["se"] = stats["std"] / np.sqrt(n_pid)

        ax.plot(stats["observation"], stats["mean"], "o-", color=color, lw=1.8, ms=5)
        ax.fill_between(stats["observation"],
                        stats["mean"] - stats["se"],
                        stats["mean"] + stats["se"],
                        color=color, alpha=0.25)
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append(src)

    ax.set_xlabel("Observation")
    ax.set_ylabel("Response variability")
    ax.set_xticks(obs_vals)
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder",   type=str, default="carrabin")
    parser.add_argument(
        "--extra_models", nargs="*", default=[],
        help="Additional models beyond MODEL_ORDER (for panels A and B)",
    )
    args = parser.parse_args()

    model_order = MODEL_ORDER + [
        m for m in args.extra_models if m not in MODEL_ORDER
    ]

    apply_style()
    pal     = get_palette(len(model_order) + 2)
    palette = {m: pal[i] for i, m in enumerate(model_order)}
    for mt in model_order:
        disp = _display(mt)
        if disp not in palette:
            palette[disp] = palette[mt]
    # Extra colours for MLE variants
    nc_idx = len(model_order)
    palette["NoisyCounting_mle"]   = pal[nc_idx]     if nc_idx     < len(pal) else "0.5"
    palette["NoisyCounting (MLE)"] = palette["NoisyCounting_mle"]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] / 2),
        constrained_layout=True,
    )

    _plot_panel_a(axes[0], args.run_folder, palette, model_order)
    _plot_panel_b(axes[1], args.run_folder, palette, model_order)
    _plot_panel_d(axes[2], args.run_folder, palette)
    _plot_panel_c(axes[3], args.run_folder, palette)

    label_panels(axes.reshape(1, -1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_carrabin_temporal"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
