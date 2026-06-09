#!/usr/bin/env python3
"""figure_carrabin_bias.py — B group figure for carrabin task.

Layout: 1x2
  Panel A (B1): Temporal weight profile — per-pid thin lines + group mean ± SEM
  Panel B (B2): Surprise sensitivity — regplot per source × condition

Run:
    python scripts/figure_carrabin_bias.py --run_folder carrabin
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
from numpy.linalg import lstsq
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
    pvalue_to_stars,
)

HUMAN_COLOR   = "0.3"
N_OBS         = 5
BELIEF_THRESH = 0.05


def _compute_weights(human: pd.DataFrame) -> pd.DataFrame:
    """Per-pid regression weights: regress obs-k inputs against final response."""
    rows = []
    for pid, g in human.groupby("pid"):
        final  = (g[g["observation"] == N_OBS][["trial", "response"]]
                  .rename(columns={"response": "final"}))
        pivot  = g.pivot_table(
            index="trial", columns="observation", values="value").reset_index()
        merged = pivot.merge(final, on="trial").dropna()
        X      = merged[[1, 2, 3, 4, 5]].values
        y      = merged["final"].values
        weights, _, _, _ = lstsq(X, y, rcond=None)
        row = {"pid": pid}
        for i in range(N_OBS):
            row[f"w{i+1}"] = weights[i]
        ws    = weights.copy()
        total = np.abs(ws).sum()
        for i in range(N_OBS):
            row[f"w{i+1}_norm"] = ws[i] / total if total > 0 else 1 / N_OBS
        row["primacy_index"] = (weights[0] + weights[1]) - (weights[3] + weights[4])
        rows.append(row)
    return pd.DataFrame(rows)


def _build_surprise_df(resp_df: pd.DataFrame,
                       input_df: pd.DataFrame) -> pd.DataFrame:
    """Build (pid, trial, obs, surprise, delta, confirming) DataFrame.

    resp_df   — response trajectories (pid, trial, observation, response)
    input_df  — provides input values (pid, trial, observation, value);
                 use human data for both human and model responses

    surprise  = |input(t) - response(t-1)|  — continuous [0, 2]
    confirming = sign(input(t)) == sign(response(t-1)), excluding near-neutral
    delta     = |response(t) - response(t-1)|
    """
    val_map = input_df[["pid", "trial", "observation", "value"]].drop_duplicates()
    df = resp_df.drop(columns=["value"], errors="ignore").merge(
        val_map, on=["pid", "trial", "observation"], how="left")

    rows = []
    for (pid, trial), g in df.groupby(["pid", "trial"]):
        g = g.sort_values("observation").reset_index(drop=True)
        for i in range(1, len(g)):
            response_prev = float(g.loc[i - 1, "response"])
            input_t       = float(g.loc[i,     "value"])
            response_t    = float(g.loc[i,     "response"])
            delta         = abs(response_t - response_prev)
            surprise      = abs(input_t - response_prev)
            if abs(response_prev) < BELIEF_THRESH:
                confirming = None
            else:
                confirming = bool(np.sign(input_t) == np.sign(response_prev))
            rows.append({
                "pid": pid, "trial": trial,
                "obs": int(g.loc[i, "observation"]),
                "surprise": surprise, "delta": delta,
                "confirming": confirming,
            })
    return pd.DataFrame(rows)


# ── Panel A (B1) — Temporal weight profile ────────────────────────────────────

def _plot_panel_a(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel A (B1): Temporal weight profile."""
    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))
    wdf     = _compute_weights(human)
    obs     = np.arange(1, N_OBS + 1)
    w_norm  = wdf[[f"w{i}_norm" for i in range(1, N_OBS + 1)]].values

    pidx      = wdf["primacy_index"].values
    vmax      = np.abs(pidx).max()
    cmap      = plt.cm.RdBu_r
    norm_cmap = plt.Normalize(vmin=-vmax, vmax=vmax)
    for _, row in wdf.iterrows():
        ws    = np.array([row[f"w{j}_norm"] for j in range(1, N_OBS + 1)])
        color = cmap(norm_cmap(row["primacy_index"]))
        ax.plot(obs, ws, lw=0.8, alpha=0.55, color=color, zorder=2)

    mean = w_norm.mean(axis=0)
    sem  = w_norm.std(axis=0) / np.sqrt(len(wdf))
    ax.plot(obs, mean, "o-", color="black", lw=2.2, ms=6, zorder=4, label="Human")
    ax.fill_between(obs, mean - sem, mean + sem, color="black", alpha=0.15, zorder=3)
    ax.axhline(1 / N_OBS, color="0.6", lw=0.9, ls="--", zorder=1)

    for mt in model_order:
        resp_path = run_dir / f"{mt}_carrabin_responses.pkl"
        if not resp_path.exists():
            continue
        resp = pd.read_pickle(resp_path)
        rows = []
        for pid, g_resp in resp.groupby("pid"):
            g_human = human[human["pid"] == pid]
            final   = (g_resp[g_resp["observation"] == N_OBS][["trial", "response"]]
                       .rename(columns={"response": "final"}))
            pivot   = g_human.pivot_table(
                index="trial", columns="observation", values="value").reset_index()
            merged  = pivot.merge(final, on="trial").dropna()
            if len(merged) < 5:
                continue
            X = merged[[1, 2, 3, 4, 5]].values
            y = merged["final"].values
            weights, _, _, _ = lstsq(X, y, rcond=None)
            total  = np.abs(weights).sum()
            w_n    = weights / total if total > 0 else np.full(N_OBS, 1 / N_OBS)
            rows.append({f"w{i+1}_norm": w_n[i] for i in range(N_OBS)})
        if not rows:
            continue
        wdf_m  = pd.DataFrame(rows)
        m_mean = wdf_m.values.mean(axis=0)
        m_sem  = wdf_m.values.std(axis=0) / np.sqrt(len(wdf_m))
        color  = palette.get(mt, "0.5")
        label  = "NEF" if mt.startswith("NEF") else mt
        ax.plot(obs, m_mean, "o-", color=color, lw=1.5, ms=4, alpha=0.85, label=label)
        ax.fill_between(obs, m_mean - m_sem, m_mean + m_sem, color=color, alpha=0.15)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label("Primacy index", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel("Observation")
    ax.set_ylabel("Normalised weight")
    ax.set_xticks(obs)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, frameon=True, framealpha=0.9, ncol=2, loc="upper right")
    sns.despine(ax=ax, top=True, right=True)


# ── Panel B (B2) — Surprise sensitivity ──────────────────────────────────────

def _plot_panel_b(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel B (B2): Surprise sensitivity — regplot per source × condition.

    Colour = source, line style = confirming (solid) vs disconfirming (dashed).
    Human CI shaded; models CI omitted to reduce clutter.
    Human r values annotated. Boundary at surprise=1 marked with dotted line.
    """
    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))

    # ── Build combined dataframe ──────────────────────────────────────────────
    all_dfs = []
    sdf = _build_surprise_df(human, human)
    sdf_s = sdf.dropna(subset=["confirming"]).copy()
    sdf_s["confirming"] = sdf_s["confirming"].astype(bool)
    sdf_s["source"] = "Human"
    all_dfs.append(sdf_s)

    for mt in model_order:
        resp_path = run_dir / f"{mt}_carrabin_responses.pkl"
        if not resp_path.exists():
            continue
        resp    = pd.read_pickle(resp_path)
        sdf_m   = _build_surprise_df(resp, human)
        sdf_m_s = sdf_m.dropna(subset=["confirming"]).copy()
        sdf_m_s["confirming"] = sdf_m_s["confirming"].astype(bool)
        sdf_m_s["source"]     = "NEF" if mt.startswith("NEF") else mt
        all_dfs.append(sdf_m_s)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Source order and colours
    source_order = ["Human"] + [
        ("NEF" if mt.startswith("NEF") else mt)
        for mt in model_order
        if (run_dir / f"{mt}_carrabin_responses.pkl").exists()
    ]
    source_color = {"Human": HUMAN_COLOR}
    for mt in model_order:
        source_color["NEF" if mt.startswith("NEF") else mt] = palette.get(mt, "0.5")

    # ── regplot per source × condition ───────────────────────────────────────
    handles, labels = [], []
    for source in source_order:
        color    = source_color[source]
        is_human = source == "Human"
        sub      = combined[combined["source"] == source]
        conf     = sub[sub["confirming"]]
        disconf  = sub[~sub["confirming"]]

        if len(conf) < 10 or len(disconf) < 10:
            continue

        ci = 95
        lw = 1.8 if is_human else 1.2

        sns.regplot(data=conf, x="surprise", y="delta", ax=ax,
                    color=color, ci=ci, scatter=False,
                    line_kws={"lw": lw, "linestyle": "-"})
        sns.regplot(data=disconf, x="surprise", y="delta", ax=ax,
                    color=color, ci=ci, scatter=False,
                    line_kws={"lw": lw, "linestyle": "--"})

        # One legend entry per source (solid line to represent it)
        handles.append(Line2D([0], [0], color=color, lw=lw))
        labels.append(source)

    # Style legend entries
    handles += [Line2D([0], [0], color="0.5", lw=1.2, ls="-"),
                Line2D([0], [0], color="0.5", lw=1.2, ls="--")]
    labels  += ["Confirming", "Disconfirming"]

    # Annotate human r values
    h_conf    = combined[(combined["source"] == "Human") &  combined["confirming"]]
    h_disconf = combined[(combined["source"] == "Human") & ~combined["confirming"]]
    r_c, p_c  = pearsonr(h_conf["surprise"],    h_conf["delta"])
    r_d, p_d  = pearsonr(h_disconf["surprise"], h_disconf["delta"])
    ax.text(0.02, 0.88,
            f"Human: conf r={r_c:.2f}{pvalue_to_stars(p_c)},  "
            f"disconf r={r_d:.2f}{pvalue_to_stars(p_d)}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7, color=HUMAN_COLOR)

    # Boundary and region labels
    ax.axvline(1.0, color="0.75", lw=0.8, ls=":", zorder=0)
    ax.text(0.35, 0.295, "Confirming",    ha="center", va="top",
            fontsize=7, color="0.4")
    ax.text(1.65, 0.295, "Disconfirming", ha="center", va="top",
            fontsize=7, color="0.4")

    ax.set_ylim(0.05, 0.30)
    ax.set_xlabel("Surprise  |obs(t) \u2212 response(t\u22121)|")
    ax.set_ylabel("Mean |\u0394response|")
    ax.legend(handles, labels, fontsize=7, frameon=True,
              framealpha=0.9, ncol=2, loc="lower right")
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder",   type=str, default="carrabin")
    parser.add_argument("--extra_models", nargs="*", default=[])
    args = parser.parse_args()

    MODEL_ORDER = ["Mean", "PrimacyRecency", "LeakyIntegrator", "NEF",
                   "NoisyCounting"]
    model_order = MODEL_ORDER + [
        m for m in args.extra_models if m not in MODEL_ORDER]

    apply_style()
    pal     = get_palette(len(model_order) + 1)
    palette = {m: pal[i] for i, m in enumerate(model_order)}

    fig, axes = plt.subplots(
        1, 2,
        figsize=(FIGURE_SIZE[0] * 0.6, FIGURE_SIZE[1] / 2),
        constrained_layout=True,
    )

    _plot_panel_a(axes[0], args.run_folder, palette, model_order)
    _plot_panel_b(axes[1], args.run_folder, palette, model_order)

    label_panels(axes.reshape(1, -1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_carrabin_bias"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
