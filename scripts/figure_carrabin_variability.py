#!/usr/bin/env python3
"""figure_carrabin_variability.py — V group figure for carrabin task.

Layout: 1×3
  Panel A (schematic): task schematic
  Panel B (V2): KDE of response variability for identical inputs across sources
  Panel C (V2): Model RMSE vs human response variability regplot

Message: humans are highly variable even for identical inputs, and this
variability drives model RMSE — underfitting is partially irreducible noise.

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
    return "NEF" if model_type.startswith("NEF") else model_type


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


# ── Panel A — schematic ───────────────────────────────────────────────────────

def _plot_schematic(ax) -> None:
    pdf_path = FIGURES_DIR / "carrabin_task.pdf"
    if not pdf_path.exists():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.text(0.5, 0.5, "carrabin_task.pdf\nnot found",
                ha="center", va="center", transform=ax.transAxes,
                color="0.5", style="italic")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "carrabin_task"
        cmd = ["pdftoppm", "-png", "-singlefile", str(pdf_path), str(out_prefix)]
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            return
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            return
        img = mpimg.imread(img_path)
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_aspect("equal"); ax.set_anchor("C")


# ── Panel B (V2) — KDE of response variability ───────────────────────────────

def _plot_panel_b(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel B (V2): Normalised KDE of response variability for identical inputs.

    Shows that NEF captures the human variability distribution best.
    Thin vertical grey lines mark each human participant.
    Deterministic models produce near-delta spikes; stochastic models spread.
    """
    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()

    # Build source list: human first, then models in order
    sources_in_order = ["human"] + [
        m for m in model_order
        if (run_dir / f"{m}_carrabin_responses.pkl").exists()
    ]
    # Add any stochastic extras not already in list
    for extra in ["NoisyCounting"]:
        if extra not in sources_in_order and \
           (run_dir / f"{extra}_carrabin_responses.pkl").exists():
            sources_in_order.append(extra)

    source_data: dict[str, pd.Series] = {}
    all_vals: list[float] = []

    for src in sources_in_order:
        if src == "human":
            rs = _qid_response_std(human, qid_map)
        else:
            resp_path = run_dir / f"{src}_carrabin_responses.pkl"
            if not resp_path.exists():
                continue
            rs = _qid_response_std(pd.read_pickle(resp_path), qid_map)
        vals = rs["resp_std"].dropna()
        if len(vals) < 2:
            continue
        source_data[src] = vals
        all_vals.extend(vals.tolist())

    if not source_data:
        ax.text(0.5, 0.5, "No response data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    x_max = np.quantile(all_vals, 0.99) * 1.1
    x     = np.linspace(0, x_max, 400)

    for src, vals in source_data.items():
        color      = palette.get(src, palette.get(_display(src), "0.5"))
        label      = "Human" if src == "human" else _display(src)
        sigma_std  = float(vals.std())
        bw         = 0.002 if sigma_std < 0.003 else "scott"
        alpha_fill = 0.15  if sigma_std < 0.003 else 0.20
        lw         = 1.2   if sigma_std < 0.003 else 1.8
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


# ── Panel C (V2) — RMSE vs response variability ──────────────────────────────

def _plot_panel_c(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel C (V2): Model RMSE vs human response variability.

    Each model shown as a regression line (scatter=False).
    Positive correlations for all models: noise drives RMSE.
    Steeper slope = model more affected by human variability.
    """
    run_dir  = data_path("runs") / run_folder
    human    = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map  = human[["pid", "trial", "observation", "qid"]].drop_duplicates()
    human_rs = _qid_response_std(human, qid_map).rename(
        columns={"resp_std": "human_var"})

    PANEL_C_MODELS = MODEL_ORDER + ["NoisyCounting"]

    rows = []
    for mt in PANEL_C_MODELS:
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
    order = [_display(m) for m in PANEL_C_MODELS
             if _display(m) in df["model"].unique()]
    pal   = {_display(m): palette.get(_display(m), palette.get(m, "0.5"))
             for m in PANEL_C_MODELS}

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


# ── Panel D (V3) — Test-retest reliability of response variability ────────────

def _plot_panel_d(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel D (V3): Test-retest reliability of response variability.

    Splits trials into first and second half per pid. Computes qid_resp_std
    for each half. Scatter of first-half vs second-half noise per pid,
    for human and NEF. High correlation = noise is a stable individual trait.

    Human: r~0.73 (***) — genuinely stable individual differences.
    NEF: should be near r=1 since noise is determined by fitted params.
    """
    from scipy.stats import pearsonr
    from matplotlib.lines import Line2D

    MIN = 3
    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()

    def half_split_std(resp_df):
        """Return (first_half_std, second_half_std) per pid."""
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

    pal     = list(palette.values())
    color_h = "0.3"
    color_n = pal[3] if len(pal) > 3 else pal[0]

    handles, labels = [], []
    nef_resp_path = run_dir / "NEF_carrabin_responses.pkl"

    for resp_df, color, src in [
        (human,                                       color_h, "Human"),
        (pd.read_pickle(nef_resp_path) if nef_resp_path.exists()
         else pd.DataFrame(),                         color_n, "NEF"),
    ]:
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
        labels.append(f"{src} r={r:.2f}{stars}")

    # Identity line
    lo, hi = 0.0, 0.20
    ax.plot([lo, hi], [lo, hi], color="0.75", lw=0.8,
            ls="--", zorder=1)
    handles.append(Line2D([0], [0], color="0.75", lw=0.8, ls="--"))
    labels.append("Identity")

    ax.set_xlabel("Response variability\n(first half of trials)")
    ax.set_ylabel("Response variability\n(second half of trials)")
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

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
    # NoisyCounting gets next colour slot
    nc_idx = len(model_order)
    palette["NoisyCounting"] = pal[nc_idx] if nc_idx < len(pal) else "0.6"

    fig, axes = plt.subplots(
        1, 3,
        figsize=(FIGURE_SIZE[0] * 0.75, FIGURE_SIZE[1] / 2),
        constrained_layout=True,
    )

    _plot_panel_b(axes[0], args.run_folder, palette, model_order)
    _plot_panel_c(axes[1], args.run_folder, palette, model_order)
    _plot_panel_d(axes[2], args.run_folder, palette, model_order)

    label_panels(axes.reshape(1, -1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_carrabin_variability"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
