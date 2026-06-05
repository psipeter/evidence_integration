#!/usr/bin/env python3
"""figure_carrabin_variability.py — V group figure for carrabin task.

Layout: 1x4
  Panel A (V2): KDE of response variability for identical inputs
  Panel B (V2): Model RMSE vs human response variability regplot
  Panel C (V3): Test-retest reliability of response variability
  Panel D (V1): Distributional model fit (NLL boxplots)

Run:
    python scripts/figure_carrabin_variability.py --run_folder carrabin
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path, resolve_run_folder
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
    pvalue_to_stars,
)

MODEL_ORDER         = ["Mean", "PrimacyRecency", "LeakyIntegrator", "NEF"]
NOISE_LABEL         = "Response variability for identical inputs"
HUMAN_NEUTRAL_COLOR = "0.3"


def _display(model_type: str) -> str:
    if model_type.startswith("NEF"):
        return "NEF"
    if model_type == "NoisyCounting_mle":
        return "NoisyCounting (MLE)"
    return model_type


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


# ── Panel A (V2) — KDE of response variability ───────────────────────────────

def _plot_panel_a(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel A (V2): Normalised KDE of response variability for identical inputs.

    Shows Human, NoisyCounting (RMSE-fitted), and NoisyCounting (MLE-fitted).
    RMSE fitting collapses sigma_c to zero; MLE recovers the correct
    state-noise distribution that overlaps well with the human distribution.
    Thin vertical grey lines mark individual human participants.
    """
    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()

    # Fixed source list: human + two NoisyCounting variants only
    SOURCES = [
        ("human",             None,
         "Human"),
        ("NoisyCounting",     run_dir / "NoisyCounting_carrabin_responses.pkl",
         "NoisyCounting (RMSE)"),
        ("NoisyCounting_mle", run_dir / "NoisyCounting_carrabin_responses_mle.pkl",
         "NoisyCounting (MLE)"),
    ]

    source_data: dict[str, pd.Series] = {}
    source_label: dict[str, str]      = {}
    all_vals: list[float] = []

    for key, resp_path, label in SOURCES:
        if key == "human":
            rs = _qid_response_std(human, qid_map)
        else:
            if resp_path is None or not resp_path.exists():
                continue
            rs = _qid_response_std(pd.read_pickle(resp_path), qid_map)
        vals = rs["resp_std"].dropna()
        if len(vals) < 2:
            continue
        source_data[key]  = vals
        source_label[key] = label
        all_vals.extend(vals.tolist())

    if not source_data:
        ax.text(0.5, 0.5, "No response data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    x_max = np.quantile(all_vals, 0.99) * 1.1
    x     = np.linspace(0, x_max, 400)

    for src, vals in source_data.items():
        color     = palette.get(src, palette.get(_display(src), "0.5"))
        label     = source_label.get(src, _display(src))
        sigma_std = float(vals.std())
        bw        = 0.002 if sigma_std < 0.003 else "scott"
        alpha_fill = 0.15 if sigma_std < 0.003 else 0.20
        lw         = 1.2  if sigma_std < 0.003 else 1.8
        kde        = gaussian_kde(vals, bw_method=bw)
        density    = kde(x)
        density    = density / density.max()
        ax.fill_between(x, density, alpha=alpha_fill, color=color)
        ax.plot(x, density, lw=lw, color=color, label=label)

    # Per-pid vertical lines for human
    if "human" in source_data:
        hvals    = source_data["human"].values
        hcolor   = palette.get("human", palette.get("Human", HUMAN_NEUTRAL_COLOR))
        hkde     = gaussian_kde(source_data["human"], bw_method="scott")
        kde_peak = float(hkde(hvals).max())
        for hv in hvals:
            top = float(hkde([hv])[0]) / kde_peak
            ax.vlines(hv, 0, top, color=hcolor, lw=0.6, alpha=0.5, zorder=2)

    ax.set_xlabel(NOISE_LABEL)
    ax.set_ylabel("Normalised density")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel B (V2) — RMSE vs response variability ──────────────────────────────

def _plot_panel_b(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel B (V2): Model RMSE vs human response variability.

    Each model shown as a regression line (scatter=False).
    Positive correlations for all models: noise drives RMSE.
    Steeper slope = model more affected by human variability.
    """
    run_dir  = data_path("runs") / run_folder
    human    = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map  = human[["pid", "trial", "observation", "qid"]].drop_duplicates()
    human_rs = _qid_response_std(human, qid_map).rename(
        columns={"resp_std": "human_var"})

    PANEL_B_MODELS = MODEL_ORDER + ["NoisyCounting"]

    rows = []
    for mt in PANEL_B_MODELS:
        f = run_dir / f"{mt}_carrabin_performance.pkl"
        if not f.exists():
            continue
        perf   = pd.read_pickle(f)[["pid", "loss"]].rename(columns={"loss": "rmse"})
        merged = human_rs.merge(perf, on="pid").dropna()
        merged["model"] = _display(mt)
        rows.append(merged)

    if not rows:
        ax.text(0.5, 0.5, "No performance data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    df    = pd.concat(rows, ignore_index=True)
    order = [_display(m) for m in PANEL_B_MODELS
             if _display(m) in df["model"].unique()]
    pal   = {_display(m): palette.get(_display(m), palette.get(m, "0.5"))
             for m in PANEL_B_MODELS}

    for model in order:
        sub   = df[df["model"] == model].copy()
        color = pal.get(model, "0.5")
        r, p  = pearsonr(sub["human_var"], sub["rmse"])
        stars = pvalue_to_stars(p)
        sns.regplot(data=sub, x="human_var", y="rmse", ax=ax,
                    color=color, ci=95, scatter=False,
                    line_kws={"lw": 1.5},
                    label=f"{model} (r={r:.2f}{stars})")

    ax.set_xlabel("Human response variability")
    ax.set_ylabel("Model fit (RMSE to human responses)")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel C (V3) — Test-retest reliability of response variability ────────────

def _plot_panel_c(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel C (V3): Test-retest reliability of response variability.

    Splits trials into first and second half per pid. Computes qid_resp_std
    for each half. Regplot (no scatter) of first vs second half per pid.
    Shows Human, NEF, NoisyCounting (RMSE), NoisyCounting (MLE).
    High correlation = noise is a stable individual trait.
    """
    from scipy.stats import pearsonr
    from matplotlib.lines import Line2D

    MIN     = 3
    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()

    def half_split_std(resp_df):
        """Return DataFrame with columns [pid, first, second]."""
        df = resp_df.drop(columns=["qid"], errors="ignore").merge(
            qid_map, on=["pid", "trial", "observation"])
        rows = []
        for pid, g in df.groupby("pid"):
            mid = g["trial"].median()
            for half, hdf in [(0, g[g["trial"] <= mid]),
                               (1, g[g["trial"] >  mid])]:
                grp = (hdf.groupby(["observation", "qid"])["response"]
                          .apply(lambda x: x.std() if len(x) >= MIN else np.nan)
                          .dropna())
                if len(grp) == 0: continue
                rows.append({"pid": pid, "half": half,
                             "resp_std": float(grp.mean())})
        wide = pd.DataFrame(rows).pivot(
            index="pid", columns="half", values="resp_std").dropna()
        wide.columns = ["first", "second"]
        return wide.reset_index()

    color_h       = "0.3"
    color_n       = palette.get("NEF", list(palette.values())[3])
    color_nc_rmse = palette.get("NoisyCounting",     "0.6")
    color_nc_mle  = palette.get("NoisyCounting_mle", "0.4")

    nef_path     = run_dir / "NEF_carrabin_responses.pkl"
    nc_rmse_path = run_dir / "NoisyCounting_carrabin_responses.pkl"
    nc_mle_path  = run_dir / "NoisyCounting_carrabin_responses_mle.pkl"

    sources = [
        (human,                                                                 color_h,       "Human"),
        (pd.read_pickle(nef_path)     if nef_path.exists()     else pd.DataFrame(), color_n,       "NEF"),
        (pd.read_pickle(nc_rmse_path) if nc_rmse_path.exists() else pd.DataFrame(), color_nc_rmse, "NoisyCounting (RMSE)"),
        (pd.read_pickle(nc_mle_path)  if nc_mle_path.exists()  else pd.DataFrame(), color_nc_mle,  "NoisyCounting (MLE)"),
    ]

    handles, labels = [], []
    for resp_df, color, src_label in sources:
        if resp_df.empty: continue
        wide = half_split_std(resp_df)
        if len(wide) < 3: continue
        r, p = pearsonr(wide["first"], wide["second"])
        stars = ("****" if p<1e-4 else "***" if p<1e-3 else
                 "**"   if p<0.01  else "*"   if p<0.05  else "ns")
        sns.regplot(data=wide, x="first", y="second", ax=ax,
                    color=color, ci=95, scatter=False,
                    line_kws={"lw": 1.5})
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append(f"{src_label} r={r:.2f}{stars}")

    # Identity line
    lo, hi = 0.0, 0.40
    ax.plot([lo, hi], [lo, hi], color="0.75", lw=0.8, ls="--", zorder=1)
    handles.append(Line2D([0], [0], color="0.75", lw=0.8, ls="--"))
    labels.append("Identity")

    ax.set_xlabel("Response variability\n(first half of trials)")
    ax.set_ylabel("Response variability\n(second half of trials)")
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel D (V1) — MLE distributional fit ────────────────────────────────────

def _mle_loss_from_responses(
    model_resp: pd.DataFrame,
    human: pd.DataFrame,
    seq_map: dict,
    sigma_floor: float = 0.10,
) -> pd.DataFrame:
    """Compute group-level MLE loss from a saved response DataFrame."""
    from collections import defaultdict
    from scipy.stats import norm

    rows = []
    for pid, hg in human.groupby("pid"):
        mg = model_resp[model_resp["pid"] == pid]
        cell_obs: dict = defaultdict(list)
        for trial, tdf in hg.groupby("trial"):
            seq = seq_map[(pid, trial)]
            for obs_idx, r in enumerate(
                    tdf.sort_values("observation")["response"].values):
                cell_obs[(seq, obs_idx)].append(float(r))
        cell_model: dict = defaultdict(list)
        for trial, tdf in mg.groupby("trial"):
            seq = seq_map[(pid, trial)]
            for obs_idx, r in enumerate(
                    tdf.sort_values("observation")["response"].values):
                cell_model[(seq, obs_idx)].append(float(r))
        total_ll, n_total = 0.0, 0
        for (seq, obs_idx), r_list in cell_obs.items():
            if (seq, obs_idx) not in cell_model:
                continue
            sim_col  = cell_model[(seq, obs_idx)]
            mu_sim   = float(np.mean(sim_col))
            sig_sim  = max(float(np.std(sim_col)), sigma_floor)
            r_arr    = np.array(r_list)
            total_ll += float(np.sum(norm.logpdf(r_arr, loc=mu_sim, scale=sig_sim)))
            n_total  += len(r_arr)
        if n_total > 0:
            rows.append({"pid": pid, "mle_loss": -total_ll / n_total})
    return pd.DataFrame(rows)


def _plot_panel_d(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel D (V1): Distributional fit — MLE loss per pid, all models.

    For deterministic models: sigma_floor=0.10 represents irreducible noise.
    For NoisyCounting: uses pre-fitted MLE params file.
    For NEF: uses saved responses with natural trial-to-trial variability.
    Ordering (lower = better):
        NoisyCounting MLE > NEF > LeakyIntegrator > PrimacyRecency > Mean
    """
    SIGMA_FLOOR = 0.10
    run_dir     = data_path("runs") / run_folder
    human       = pd.read_pickle(data_path("carrabin.pkl"))

    seq_map = {}
    for (pid, trial), g in human.groupby(["pid", "trial"]):
        seq_map[(pid, trial)] = tuple(
            g.sort_values("observation")["value"].values)

    PANEL_D_MODELS = MODEL_ORDER + ["NoisyCounting"]
    rows = []

    for mt in PANEL_D_MODELS:
        if mt == "NoisyCounting":
            mle_path = run_dir / "NoisyCounting_carrabin_params_mle.pkl"
            if not mle_path.exists():
                continue
            mle_df = pd.read_pickle(mle_path)[["pid", "mle_loss"]].copy()
            mle_df["source"] = "NoisyCounting"
            rows.append(mle_df)
        else:
            rpath = run_dir / f"{mt}_carrabin_responses.pkl"
            if not rpath.exists():
                continue
            df = _mle_loss_from_responses(
                pd.read_pickle(rpath), human, seq_map, SIGMA_FLOOR)
            df["source"] = _display(mt)
            rows.append(df)

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    plot_df = pd.concat(rows, ignore_index=True)
    order   = [_display(m) for m in PANEL_D_MODELS
               if _display(m) in plot_df["source"].unique()]
    pal     = {_display(m): palette.get(_display(m), palette.get(m, "0.5"))
               for m in PANEL_D_MODELS}

    sns.boxplot(data=plot_df, x="source", y="mle_loss", order=order,
                hue="source", palette=pal, legend=False, ax=ax)

    ax.axhline(0, color="0.7", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel("Distributional model fit (NLL)")
    ax.tick_params(axis="x", rotation=45)

    # Significance bars: NEF vs each other model (Wilcoxon signed-rank)
    from scipy.stats import wilcoxon
    nef_vals = plot_df[plot_df["source"] == "NEF"]["mle_loss"]
    if len(nef_vals) >= 3:
        nef_idx = order.index("NEF")
        y_max   = plot_df["mle_loss"].quantile(0.95)
        y_range = plot_df["mle_loss"].quantile(0.95) - plot_df["mle_loss"].quantile(0.05)
        y_step  = y_range * 0.12
        bar_n   = 0
        for other in [m for m in order if m != "NEF"]:
            other_vals = plot_df[plot_df["source"] == other]["mle_loss"]
            if len(other_vals) < 3:
                continue
            merged = (plot_df[plot_df["source"] == "NEF"][["pid", "mle_loss"]]
                      .merge(plot_df[plot_df["source"] == other][["pid", "mle_loss"]],
                             on="pid", suffixes=("_nef", "_other")))
            if len(merged) < 3:
                continue
            _, p = wilcoxon(merged["mle_loss_nef"], merged["mle_loss_other"])
            stars = pvalue_to_stars(p)
            if stars == "ns":
                continue
            x1, x2 = nef_idx, order.index(other)
            y = y_max + y_step * (bar_n + 1)
            ax.plot([x1, x2], [y, y], color="black", lw=0.9, clip_on=False)
            ax.text((x1 + x2) / 2, y, stars, ha="center", va="bottom",
                    fontsize=7)
            bar_n += 1

    sns.despine(ax=ax, top=True, right=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder",   type=str, default="carrabin")
    parser.add_argument(
        "--extra_models", nargs="*", default=[],
        help="Additional models beyond MODEL_ORDER",
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
    palette["Human"] = HUMAN_NEUTRAL_COLOR
    palette["human"] = HUMAN_NEUTRAL_COLOR
    nc_idx = len(model_order)
    palette["NoisyCounting"]       = pal[nc_idx]     if nc_idx     < len(pal) else "0.6"
    palette["NoisyCounting_mle"]   = pal[nc_idx + 1] if nc_idx + 1 < len(pal) else "0.4"
    palette["NoisyCounting (MLE)"] = palette["NoisyCounting_mle"]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] / 2),
        constrained_layout=True,
    )

    _plot_panel_a(axes[0], args.run_folder, palette, model_order)
    _plot_panel_b(axes[1], args.run_folder, palette, model_order)
    _plot_panel_c(axes[2], args.run_folder, palette, model_order)
    _plot_panel_d(axes[3], args.run_folder, palette, model_order)

    label_panels(axes.reshape(1, -1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_carrabin_variability"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
