#!/usr/bin/env python3
"""Carrabin summary figure: 2×4 layout, panels A–H."""

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
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path, resolve_run_folder
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette, label_panels, annotate_nef_comparisons

MODEL_ORDER = ["Mean", "PrimacyRecency", "LeakyIntegrator", "NEF"]
HUMAN_NEUTRAL_COLOR = "0.3"

# --- bottom row (E–H) ---
NOISE_LABEL  = "Response variability for identical inputs"
N_NEURONS_LIST = [25, 50, 100, 200, 400]


def _display(model_type: str) -> str:
    return "NEF" if model_type.startswith("NEF") else model_type


def _placeholder(ax, text: str) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, text, ha="center", va="center",
            transform=ax.transAxes, color="0.5", style="italic")


def _empty_pdf_panel(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_aspect("equal"); ax.set_anchor("C")


def _get_loss(perf_df: pd.DataFrame) -> pd.Series:
    if "loss" in perf_df.columns:
        return perf_df["loss"]
    return perf_df["cv_loss_mean"]


def _qid_response_std(resp_df: pd.DataFrame, qid_map: pd.DataFrame,
                      min_trials: int = 3) -> pd.DataFrame:
    """Mean of std(response | obs, qid) per pid.

    Controls for observation position and input sequence, isolating
    pure trial-to-trial response variability.
    Returns DataFrame with columns [pid, resp_std].
    """
    df = resp_df.drop(columns=["qid"], errors="ignore").merge(
        qid_map, on=["pid", "trial", "observation"])
    grp = (
        df.groupby(["pid", "observation", "qid"])["response"]
        .apply(lambda x: x.std() if len(x) >= min_trials else np.nan)
        .dropna()
        .reset_index(name="resp_std")
    )
    return grp.groupby("pid")["resp_std"].mean().reset_index()


# ── Panel A ───────────────────────────────────────────────────────────────────

def _plot_panel_a(ax) -> None:
    """Render carrabin_task.pdf into panel A."""
    pdf_path = FIGURES_DIR / "carrabin_task.pdf"
    if not pdf_path.exists():
        _empty_pdf_panel(ax); return
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "carrabin_task"
        cmd = ["pdftoppm", "-png", "-singlefile", str(pdf_path), str(out_prefix)]
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            _empty_pdf_panel(ax); return
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            _empty_pdf_panel(ax); return
        img = mpimg.imread(img_path)
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_aspect("equal"); ax.set_anchor("C")


# ── Panel B ───────────────────────────────────────────────────────────────────

def _plot_panel_b(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel B: RMSE boxplots."""
    run_dir = data_path("runs") / run_folder
    rows = []
    for mt in model_order:
        f = run_dir / f"{mt}_carrabin_performance.pkl"
        if not f.exists():
            continue
        perf = pd.read_pickle(f).copy()
        perf["plot_loss"] = _get_loss(perf)
        perf["model_disp"] = _display(mt)
        rows.append(perf[["pid", "model_disp", "plot_loss"]])
    if not rows:
        _placeholder(ax, "No performance data"); return
    df = pd.concat(rows, ignore_index=True)
    order = [_display(m) for m in model_order]
    available = [m for m in order if m in set(df["model_disp"])]
    pal = {m: palette.get(m, "0.5") for m in available}
    sns.boxplot(data=df, x="model_disp", y="plot_loss", order=available,
                hue="model_disp", palette=pal, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Response error (trial-wise RMSE)")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax, top=True, right=True)
    nef_disp = _display("NEF")
    if nef_disp in available:
        annotate_nef_comparisons(
            ax, df, "model_disp", "plot_loss", available,
            nef_label=nef_disp,
            compare_only=["Mean", "LeakyIntegrator", "PrimacyRecency"])





# ── Panel D — KDE of response variability ────────────────────────────────────

def _plot_panel_c(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel D: Normalised KDE of response variability (qid_resp_std) per source.

    Shows that NEF captures the human variability distribution best.
    """
    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()

    STOCHASTIC = {"human", "NoisyCounting", "NEF"}
    sources_in_order = ["human"] + [m for m in model_order
                                    if (run_dir / f"{m}_carrabin_responses.pkl").exists()]
    for extra in STOCHASTIC:
        if extra not in sources_in_order and \
           (run_dir / f"{extra}_carrabin_responses.pkl").exists():
            sources_in_order.append(extra)

    all_vals = []
    source_data: dict[str, pd.Series] = {}

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
        _placeholder(ax, "No response data"); return

    x_max = np.quantile(all_vals, 0.99) * 1.1
    x     = np.linspace(0, x_max, 400)

    for src, vals in source_data.items():
        color = palette.get(src, palette.get(_display(src), "0.5"))
        label = "Human" if src == "human" else _display(src)
        sigma_std = float(vals.std())
        bw         = 0.002 if sigma_std < 0.003 else "scott"
        alpha_fill = 0.15  if sigma_std < 0.003 else 0.20
        lw         = 1.2   if sigma_std < 0.003 else 1.8
        kde     = gaussian_kde(vals, bw_method=bw)
        density = kde(x)
        density = density / density.max()
        ax.fill_between(x, density, alpha=alpha_fill, color=color)
        ax.plot(x, density, lw=lw, color=color, label=label)

    # Thin vertical lines for each human pid, from x-axis to their KDE height
    if "human" in source_data:
        human_vals  = source_data["human"].values
        human_color = palette.get("human", palette.get("Human", "0.3"))
        human_kde   = gaussian_kde(source_data["human"], bw_method="scott")
        kde_peak    = float(human_kde(human_vals).max())
        for hv in human_vals:
            top = float(human_kde([hv])[0]) / kde_peak
            ax.vlines(hv, 0, top, color=human_color, lw=0.6, alpha=0.5, zorder=2)

    ax.set_xlabel(NOISE_LABEL)
    ax.set_ylabel("Normalised density")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel E — RMSE vs response variability ───────────────────────────────────

def _plot_panel_d(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel E: Model RMSE vs human response variability (qid_resp_std).

    Each model gets its own regplot; shows that human variability predicts
    model difficulty universally — noise is a major contributor to RMSE.
    """
    from scipy.stats import pearsonr
    from matplotlib.lines import Line2D

    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()

    human_rs = _qid_response_std(human, qid_map).rename(
        columns={"resp_std": "human_var"})

    PANEL_D_MODELS = MODEL_ORDER + ["NoisyCounting"]

    rows = []
    for mt in PANEL_D_MODELS:
        f = run_dir / f"{mt}_carrabin_performance.pkl"
        if not f.exists():
            continue
        perf = pd.read_pickle(f)[["pid", "loss"]].rename(columns={"loss": "rmse"})
        merged = human_rs.merge(perf, on="pid").dropna()
        merged["model"] = _display(mt)
        rows.append(merged)

    if not rows:
        _placeholder(ax, "No performance data"); return

    df    = pd.concat(rows, ignore_index=True)
    order = [_display(m) for m in PANEL_D_MODELS if _display(m) in df["model"].unique()]
    pal   = {_display(m): palette.get(_display(m), palette.get(m, "0.5"))
             for m in PANEL_D_MODELS}

    for model in order:
        sub   = df[df["model"] == model].copy()
        color = pal.get(model, "0.5")
        r, p  = pearsonr(sub["human_var"], sub["rmse"])
        stars = ("****" if p<1e-4 else "***" if p<1e-3 else
                 "**"   if p<0.01  else "*"   if p<0.05  else "ns")
        sns.regplot(data=sub, x="human_var", y="rmse", ax=ax,
                    color=color, ci=95, scatter=False,
                    line_kws={"lw": 1.5},
                    label=f"{model} (r={r:.2f}{stars})")

    ax.set_xlabel("Human response variability")
    ax.set_ylabel("Model RMSE")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel F — n_neurons scan ──────────────────────────────────────────────────

def _plot_panel_e(ax, run_folder: str, palette: dict) -> None:
    """Panel E: response variability and prediction error variability vs n_neurons.

    Both metrics are computed at plot time from raw responses and PE values
    saved by extras_carrabin.py --n_neurons_scan.
    palette[0]: resp_std  — response variability for identical inputs
    palette[1]: pe_std    — prediction error variability (std PE at readout)
    Human response variability shown as thin grey reference lines.
    """
    run_dir   = data_path("runs") / run_folder
    scan_path = run_dir / "n_neurons_scan.pkl"
    if not scan_path.exists():
        _placeholder(ax, "No n_neurons scan data\n(run extras_carrabin.py)")
        return

    scan_raw = pd.read_pickle(scan_path)
    human    = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map  = human[["pid", "trial", "observation", "qid"]].drop_duplicates()
    human_rs = _qid_response_std(human, qid_map)
    human_vals = human_rs["resp_std"].values

    color_var = list(palette.values())[0]   # resp_std
    color_pe  = list(palette.values())[1]   # pe_std

    # ── Compute metrics from raw scan data ────────────────────────────────────
    # New format: {n_neurons: {"responses": df, "pe_readout": df}}
    if not isinstance(scan_raw, dict) or not all(
        isinstance(v, dict) and "responses" in v for v in scan_raw.values()
    ):
        _placeholder(ax, "Rerun n_neurons scan\n(extras_carrabin.py --n_neurons_scan)")
        return

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

    # Thin grey lines for each human participant's response variability
    for hv in human_vals:
        ax.axhline(hv, color="0.78", lw=0.3, zorder=0)

    sns.lineplot(data=scan_df, x="n_neurons", y="resp_std",
                 color=color_var, lw=1.8, errorbar="sd", err_style="band",
                 label="NEF response variability", ax=ax)
    sns.lineplot(data=scan_df, x="n_neurons", y="pe_std",
                 color=color_pe, lw=1.8, errorbar="sd", err_style="band",
                 label="NEF prediction error variability", ax=ax)

    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Variability")
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(n) for n in n_vals])
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="0.78", lw=0.8))
    labels.append("Human response variability")
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9,
              loc="upper right")
    sns.despine(ax=ax, top=True, right=True)



# ── Panels G, H — blank ──────────────────────────────────────────────────────

def _add_resid(df: pd.DataFrame) -> pd.DataFrame:
    """Add resid column = response - mean(response | pid, obs, qid)."""
    means = (df.groupby(["pid", "observation", "qid"])["response"]
               .mean().reset_index().rename(columns={"response": "qid_mean"}))
    df2 = df.merge(means, on=["pid", "observation", "qid"])
    df2["resid"] = df2["response"] - df2["qid_mean"]
    return df2


def _plot_panel_f(ax, run_folder: str, palette: dict) -> None:
    """Panel F: Within-trial lag-k residual autocorrelation (lag 1-4).

    Residual = response - mean(response | pid, obs, qid), removing the
    systematic trajectory. Autocorrelation of residuals across consecutive
    observations within a trial is the signature of state noise: noise
    injected at obs t persists into obs t+1, t+2, etc.

    Replicates Prat-Carrabin & Woodford (2024) Figure 5B but adds NEF.
    Human r≈0.62 at lag=1, decaying to ≈0.22 at lag=4 (matches paper).
    NEF shows same pattern, slightly stronger (r≈0.78 at lag=1).
    """
    from scipy.stats import pearsonr
    from matplotlib.lines import Line2D

    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()
    nef_resp = pd.read_pickle(run_dir / "NEF_carrabin_responses.pkl").drop(
        columns=["qid"], errors="ignore").merge(
        qid_map, on=["pid", "trial", "observation"])

    human2 = _add_resid(human)
    nef2   = _add_resid(nef_resp)

    nc_resp_path = run_dir / "NoisyCounting_carrabin_responses.pkl"
    if nc_resp_path.exists():
        nc_resp = pd.read_pickle(nc_resp_path).drop(
            columns=["qid"], errors="ignore").merge(
            qid_map, on=["pid", "trial", "observation"])
        nc2 = _add_resid(nc_resp)
    else:
        nc2 = pd.DataFrame()

    pal     = list(palette.values())
    color_h = "0.3"
    color_n = pal[3] if len(pal) > 3 else pal[0]
    color_nc = palette.get("NoisyCounting", pal[4] if len(pal) > 4 else "0.6")
    lags    = [1, 2, 3, 4]

    sources = [(human2, color_h, "Human"), (nef2, color_n, "NEF")]
    if not nc2.empty:
        sources.append((nc2, color_nc, "NoisyCounting"))

    handles, labels = [], []
    for df, color, src in sources:
        rs = []
        for lag in lags:
            pairs = []
            for (pid, trial), g in df.groupby(["pid", "trial"]):
                r = g.sort_values("observation")["resid"].values
                if len(r) > lag:
                    pairs.extend(zip(r[:-lag], r[lag:]))
            arr = np.array(pairs)
            rv, _ = pearsonr(arr[:, 0], arr[:, 1])
            rs.append(rv)
        ax.plot(lags, rs, "o-", color=color, lw=1.8, ms=5)
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append(src)

    ax.axhline(0, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("Lag (observations)")
    ax.set_ylabel("Residual autocorrelation")
    ax.set_xticks(lags)
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_g(ax, run_folder: str, palette: dict) -> None:
    """Panel G: Response variability growth across observations.

    Computes mean std(resid | pid, obs, qid) per observation position,
    averaged across pids with SD error bands.

    State noise predicts monotonic growth (noise accumulates with each
    update). Observation noise predicts a flat line. Leak slows growth
    but still produces a rising pattern.

    NEF grows steeply (pure accumulated spiking noise), human grows
    modestly (same qualitative shape, additional noise sources at obs=1).
    """
    MIN = 3
    run_dir = data_path("runs") / run_folder
    human   = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()
    nef_resp = pd.read_pickle(run_dir / "NEF_carrabin_responses.pkl").drop(
        columns=["qid"], errors="ignore").merge(
        qid_map, on=["pid", "trial", "observation"])

    human2 = _add_resid(human)
    nef2   = _add_resid(nef_resp)

    nc_resp_path = run_dir / "NoisyCounting_carrabin_responses.pkl"
    if nc_resp_path.exists():
        nc_resp_g = pd.read_pickle(nc_resp_path).drop(
            columns=["qid"], errors="ignore").merge(
            qid_map, on=["pid", "trial", "observation"])
        nc2_g = _add_resid(nc_resp_g)
    else:
        nc2_g = pd.DataFrame()

    from matplotlib.lines import Line2D
    pal      = list(palette.values())
    color_h  = "0.3"
    color_n  = pal[3] if len(pal) > 3 else pal[0]
    color_nc = palette.get("NoisyCounting", pal[4] if len(pal) > 4 else "0.6")
    obs_vals = sorted(human["observation"].unique())

    sources = [(human2, color_h, "Human"), (nef2, color_n, "NEF")]
    if not nc2_g.empty:
        sources.append((nc2_g, color_nc, "NoisyCounting"))

    handles, labels = [], []
    for df, color, src in sources:
        grp = (df.groupby(["pid", "observation", "qid"])["resid"]
                 .apply(lambda x: x.std() if len(x) >= MIN else np.nan)
                 .dropna().reset_index(name="std"))
        by_pid_obs = grp.groupby(["pid", "observation"])["std"].mean().reset_index()
        stats = by_pid_obs.groupby("observation")["std"].agg(["mean", "std"]).reset_index()
        n_pid = by_pid_obs["pid"].nunique()
        stats["se"] = stats["std"] / np.sqrt(n_pid)

        ax.plot(stats["observation"], stats["mean"], "o-",
                color=color, lw=1.8, ms=5)
        ax.fill_between(stats["observation"],
                        stats["mean"] - stats["se"],
                        stats["mean"] + stats["se"],
                        color=color, alpha=0.25)
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append(src)

    ax.set_xlabel("Observation")
    ax.set_ylabel("Response variability (residual std)")
    ax.set_xticks(obs_vals)
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_h(ax) -> None:
    _placeholder(ax, "(pending)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="carrabin")
    parser.add_argument("--out_folder", type=str, default="carrabin")
    parser.add_argument(
        "--extra_models", nargs="*", default=["NoisyCounting", "RNN"],
        help="Additional models to include in top-row panels",
    )
    args = parser.parse_args()

    model_order = MODEL_ORDER + [m for m in args.extra_models
                                 if m not in MODEL_ORDER]

    apply_style()
    _pal     = get_palette(len(model_order))
    palette  = {m: _pal[i] for i, m in enumerate(model_order)}
    for mt in model_order:
        disp = _display(mt)
        if disp not in palette:
            palette[disp] = palette[mt]
    palette["Human"] = HUMAN_NEUTRAL_COLOR

    fig, axes = plt.subplots(2, 4, figsize=FIGURE_SIZE, constrained_layout=True)
    row0, row1 = axes[0], axes[1]

    _plot_panel_a(row0[0])
    _plot_panel_b(row0[1], args.run_folder, palette, model_order)
    _plot_panel_c(row0[2], args.run_folder, palette, model_order)
    _plot_panel_d(row0[3], args.run_folder, palette, model_order)

    _plot_panel_e(row1[0], args.run_folder, palette)
    _plot_panel_f(row1[1], args.run_folder, palette)
    _plot_panel_g(row1[2], args.run_folder, palette)
    _plot_panel_h(row1[3])

    label_panels(axes)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "figure_carrabin.png", dpi=300)
    plt.savefig(FIGURES_DIR / "figure_carrabin.pdf")
    plt.savefig(FIGURES_DIR / "figure_carrabin.svg")
    print("Saved figures/figure_carrabin.{png,pdf,svg}")


if __name__ == "__main__":
    main()
