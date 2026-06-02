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
MODEL_ORDER_B = MODEL_ORDER
MODEL_ORDER_D = ["Human", "Mean", "LeakyIntegrator", "PrimacyRecency", "NEF"]

HUMAN_NEUTRAL_COLOR = "0.3"

# --- bottom row (E–H): data from scripts/extras_carrabin.py ---
SAMPLE_PIDS = [6, 7]  # high/low alpha_0 example pids for panel 1
MIN_REPEATS = 10  # minimum trial repeats per qid for analysis
READOUT_OFFSET = 0.5  # seconds into obs window for readout
N_NEURONS_LIST = [50, 75, 100, 150, 200, 300, 500]


def _display(model_type: str) -> str:
    return "NEF" if model_type.startswith("NEF") else model_type


def _placeholder(ax, text: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="0.5",
        style="italic",
    )


def _empty_pdf_panel(ax) -> None:
    """Panels A/C: bare axes matching embedded-PDF styling, no decorative text."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _plot_panel_a(ax) -> None:
    """Render first page of figures/carrabin_task.pdf into panel A."""
    pdf_path = FIGURES_DIR / "carrabin_task.pdf"
    if not pdf_path.exists():
        _empty_pdf_panel(ax)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "carrabin_task"
        cmd = [
            "pdftoppm",
            "-png",
            "-singlefile",
            str(pdf_path),
            str(out_prefix),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            _empty_pdf_panel(ax)
            return
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            _empty_pdf_panel(ax)
            return
        img = mpimg.imread(img_path)

    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Preserve aspect ratio while maximizing panel coverage.
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _get_loss(perf_df: pd.DataFrame) -> pd.Series:
    # "loss" is the current column name; fall back to "cv_loss_mean"
    # for performance pickles produced before the column rename.
    if "loss" in perf_df.columns:
        return perf_df["loss"]
    return perf_df["cv_loss_mean"]


def _plot_panel_b(ax, run_folder: str, palette: dict, model_order: list[str]) -> None:
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
        _placeholder(ax, "No performance data")
        return

    df = pd.concat(rows, ignore_index=True)
    order = [_display(m) for m in model_order]
    available = [m for m in order if m in set(df["model_disp"])]
    pal = {m: palette.get(m, "0.5") for m in available}
    sns.boxplot(
        data=df,
        x="model_disp",
        y="plot_loss",
        order=available,
        hue="model_disp",
        palette=pal,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Response error (trial-wise RMSE)")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax, top=True, right=True)
    nef_disp = _display("NEF")
    if nef_disp in available:
        annotate_nef_comparisons(ax, df, "model_disp", "plot_loss", available, nef_label=nef_disp,
                          compare_only=["Mean", "LeakyIntegrator", "PrimacyRecency"])


def _plot_panel_c(ax, run_folder: str, palette: dict, model_order: list[str]) -> None:
    """Panel C: Normalised KDE of per-participant sigma (RNN residual noise).

    Human, NEF, and NoisyCounting shown as filled KDEs normalised to peak=1.
    Deterministic models shown as vertical lines at their mean sigma.
    """
    run_dir = data_path("runs") / run_folder
    noise_f = run_dir / "RNN_sigma_carrabin_sigma.pkl"
    if not noise_f.exists():
        _placeholder(ax, "No RNN noise data (run models/RNN.py --all_sources)")
        return

    sigma_df = pd.read_pickle(noise_f)
    STOCHASTIC = {"human", "NoisyCounting", "NEF"}

    sources_in_order = ["human"] + [
        m for m in model_order if m in sigma_df["source"].unique()
    ]
    for extra in sigma_df["source"].unique():
        if extra in STOCHASTIC and extra not in sources_in_order:
            sources_in_order.append(extra)

    x_max = sigma_df["sigma"].quantile(0.99) * 1.1
    x     = np.linspace(0, x_max, 400)

    for src in sources_in_order:
        sub = sigma_df[sigma_df["source"] == src]["sigma"].dropna()
        if len(sub) == 0:
            continue
        color = palette.get(src, palette.get(_display(src), "0.5"))
        label = "Human" if src == "human" else _display(src)
        if src in STOCHASTIC and len(sub) >= 4:
            kde     = gaussian_kde(sub, bw_method="scott")
            density = kde(x)
            density = density / density.max()
            ax.fill_between(x, density, alpha=0.20, color=color)
            ax.plot(x, density, lw=1.8, color=color, label=label)
        else:
            mean_sigma = float(sub.mean())
            ax.axvline(mean_sigma, color=color, lw=1.5,
                       linestyle=":", label=f"{label}")

    ax.set_xlabel("Response noise")
    ax.set_ylabel("Normalised density")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def _load_loss_long(
    run_dir: Path,
    model_order: list[str],
    dataset: str,
) -> pd.DataFrame:
    """
    Load per-pid qid-std shape metric for each model.
    Returns DataFrame with columns: pid, model_type, loss.
    """
    from utils.plot_style import mean_qid_std

    rows = []
    human_full = pd.read_pickle(data_path(f"{dataset}.pkl"))

    for mt in model_order:
        resp_path = run_dir / f"{mt}_{dataset}_responses.pkl"
        if not resp_path.exists():
            print(f"Warning: missing {resp_path.name}, cannot compute loss for {mt}")
            continue
        responses = pd.read_pickle(resp_path)
        for pid, model_pid in responses.groupby("pid"):
            human_pid = human_full[human_full["pid"] == pid]
            qid_map = human_pid[["pid", "trial", "observation", "qid"]].drop_duplicates()
            model_with_qid = model_pid.merge(
                qid_map, on=["pid", "trial", "observation"], how="left"
            )
            loss = abs(mean_qid_std(human_pid) - mean_qid_std(model_with_qid))
            rows.append({"pid": int(pid), "model_type": mt, "loss": loss})

    return pd.DataFrame(rows)


PANEL_D_MODELS = MODEL_ORDER + ["NoisyCounting"]


def _plot_panel_d(ax, run_folder: str, palette: dict, model_order: list[str], base_model_order: list[str] = None) -> None:
    """Panel D: Human sigma vs model RMSE for each model.

    Shows that response noise (human sigma) is a strong predictor of model
    difficulty across all model types — evidence that irreducible noise is
    a primary driver of performance differences between participants.
    Each model gets its own regression line; all share the same x-axis.
    """
    from scipy.stats import pearsonr

    if base_model_order is None:
        base_model_order = PANEL_D_MODELS
    run_dir  = data_path("runs") / run_folder
    sigma_df = pd.read_pickle(run_dir / "RNN_sigma_carrabin_sigma.pkl")
    human_s  = sigma_df[sigma_df["source"]=="human"][["pid","sigma"]].rename(
        columns={"sigma":"human_sigma"})

    rows = []
    for mt in base_model_order:  # always use base MODEL_ORDER, not extras
        f = run_dir / f"{mt}_carrabin_performance.pkl"
        if not f.exists():
            continue
        perf = pd.read_pickle(f)[["pid","loss"]].rename(columns={"loss":"rmse"})
        merged = human_s.merge(perf, on="pid").dropna()
        merged["model"] = _display(mt)
        rows.append(merged)

    if not rows:
        _placeholder(ax, "No performance data")
        return

    df = pd.concat(rows, ignore_index=True)
    order = [_display(m) for m in base_model_order
             if _display(m) in df["model"].unique()]
    pal = {_display(m): palette.get(_display(m), palette.get(m, "0.5"))
           for m in base_model_order}

    for model in order:
        sub   = df[df["model"]==model].copy()
        color = pal.get(model, "0.5")
        r, p  = pearsonr(sub["human_sigma"], sub["rmse"])
        stars = ("****" if p<1e-4 else "***" if p<1e-3 else "**" if p<0.01
                 else "*" if p<0.05 else "ns")
        sns.regplot(
            data=sub,
            x="human_sigma",
            y="rmse",
            ax=ax,
            color=color,
            ci=95,
            scatter=False,
            line_kws={"lw": 1.5},
            label=f"{model} (r={r:.2f}{stars})",
        )

    ax.set_xlabel("Human response noise (σ)")
    ax.set_ylabel("Model RMSE")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)



def _plot_panel_e(ax, run_folder: str, palette: dict) -> None:
    """Panel E: decoded PE timecourse — obs=1, first value +1, mean ± 95% CI.

    Uses sns.lineplot with errorbar=("ci", 95) across trials.
    Filtered to observation=1, qid starting with "1" (first obs was +1).
    Downsampled to every 20ms for a clean smooth trace.
    """
    run_dir    = data_path("runs") / run_folder
    PIDS       = [6, 7]
    READOUT_T  = 0.5
    DOWNSAMPLE = 20    # keep every Nth row (dt=0.001 -> every 20ms)
    MIN_TRIALS = 5

    params_all = pd.read_pickle(run_dir / "NEF_carrabin_params.pkl")
    pal_list   = list(palette.values())
    colors     = [pal_list[3], pal_list[0]] if len(pal_list) >= 4 else ["C0", "C1"]

    dfs = []
    for pid, color in zip(PIDS, colors):
        path = run_dir / f"probe_timeseries_NEF_carrabin_{pid}.pkl"
        if not path.exists():
            continue
        df = pd.read_pickle(path)
        df = df[
            (df["observation"] == 1) &
            (df["qid"].astype(str).str.startswith("1"))
        ].copy()
        if df["trial"].nunique() < MIN_TRIALS:
            continue
        # Downsample: keep every DOWNSAMPLE-th timestep
        df = df[df["t_within_obs"].apply(
            lambda t: int(round(t * 1000)) % DOWNSAMPLE == 0
        )].copy()
        df["abs_pe"] = np.abs(df["decoded_pe"])
        row = params_all[params_all["pid"] == pid].iloc[0]
        df["pid_label"] = f"pid={pid}  α₀={row['alpha_0']:.2f}  λ={row['lambda_']:.2f}"
        dfs.append((df, color))

    if not dfs:
        _placeholder(ax, "No probe timeseries data\n(run extras_carrabin.py)")
        return

    combined = pd.concat([d for d, _ in dfs], ignore_index=True)
    order = combined[["pid","pid_label"]].drop_duplicates().sort_values("pid")["pid_label"].tolist()
    pal = {label: color for (d, color), label in zip(dfs, order)}

    sns.lineplot(
        data=combined,
        x="t_within_obs",
        y="abs_pe",
        hue="pid_label",
        hue_order=order,
        palette=pal,
        errorbar="sd",
        err_style="band",
        ax=ax,
    )
    ax.set_xlabel("Time within observation (s)")
    ax.set_ylabel("Decoded prediction error")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, title=None)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_f(ax, run_folder: str, palette: dict) -> None:
    """Panel F: alpha_0 vs response noise (sigma_NEF) across participants.

    Scatter + regression showing that higher learning rate -> more response noise.
    """
    from scipy.stats import pearsonr

    run_dir = data_path("runs") / run_folder
    sigma_df = pd.read_pickle(run_dir / "RNN_sigma_carrabin_sigma.pkl")
    nef_sigma = sigma_df[sigma_df["source"]=="NEF"][["pid","sigma"]].rename(
        columns={"sigma": "response_noise"})
    params = pd.read_pickle(run_dir / "NEF_carrabin_params.pkl")[["pid","alpha_0","lambda_"]]

    df = nef_sigma.merge(params, on="pid").dropna()
    if df.empty:
        _placeholder(ax, "No NEF sigma data")
        return

    color = list(palette.values())[0]
    r, p  = pearsonr(df["alpha_0"], df["response_noise"])
    stars = "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    sns.regplot(
        data=df,
        x="alpha_0",
        y="response_noise",
        scatter_kws={"color": color, "s": 25, "alpha": 0.8},
        line_kws={"color": color, "lw": 1.5},
        ci=95,
        ax=ax,
        label=f"r={r:.2f} {stars}",
    )
    ax.set_xlabel("Fitted α₀")
    ax.set_ylabel("Response noise")
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    # Replace default scatter handle with a solid line
    handles = [Line2D([0],[0], color=color, lw=1.5)]
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_g(ax, run_folder: str, palette: dict) -> None:
    """Panel G: decoded PE std at readout vs response noise (sigma_NEF).

    Requires probe_timeseries data for all pids. Falls back to lambda_ vs
    sigma_NEF if only a subset of pids have probe data.
    """
    from scipy.stats import pearsonr

    run_dir = data_path("runs") / run_folder
    sigma_df = pd.read_pickle(run_dir / "RNN_sigma_carrabin_sigma.pkl")
    nef_sigma = sigma_df[sigma_df["source"]=="NEF"][["pid","sigma"]].rename(
        columns={"sigma": "response_noise"})
    params = pd.read_pickle(run_dir / "NEF_carrabin_params.pkl")[["pid","alpha_0","lambda_"]]
    color = list(palette.values())[0]

    # Load pe_readout data if collected; fall back to lambda_ otherwise
    pe_path = run_dir / "pe_readout_NEF_carrabin.pkl"
    if pe_path.exists():
        pe_df = pd.read_pickle(pe_path)
        # Compute std within (obs, qid) groups, then average across groups.
        # This controls for observation number and stimulus sequence,
        # isolating pure trial-to-trial noise (same method as old panel G).
        grp_std = (
            pe_df.groupby(["pid", "observation", "qid"])["pe_at_readout"]
            .std()
            .reset_index()
            .rename(columns={"pe_at_readout": "pe_std"})
        )
        # Require at least 3 trials per group to get a reliable std estimate
        grp_counts = (
            pe_df.groupby(["pid", "observation", "qid"])["pe_at_readout"]
            .count()
            .reset_index()
            .rename(columns={"pe_at_readout": "n"})
        )
        grp_std = grp_std.merge(grp_counts, on=["pid", "observation", "qid"])
        grp_std = grp_std[grp_std["n"] >= 3]
        pe_std = (
            grp_std.groupby("pid")["pe_std"]
            .mean()
            .reset_index()
        )
        df = pe_std.merge(nef_sigma, on="pid").dropna()
        x_col, x_label = "pe_std", "Mean std of decoded PE at readout"
    else:
        # Fall back to lambda_ vs response_noise
        df = nef_sigma.merge(params, on="pid").dropna()
        x_col, x_label = "lambda_", "Fitted λ"

    if df.empty:
        _placeholder(ax, "No data for panel G")
        return

    r, p  = pearsonr(df[x_col], df["response_noise"])
    stars = "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    sns.regplot(
        data=df,
        x=x_col,
        y="response_noise",
        scatter_kws={"color": color, "s": 25, "alpha": 0.8},
        line_kws={"color": color, "lw": 1.5},
        ci=95,
        ax=ax,
        label=f"r={r:.2f} {stars}",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Response noise")
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles = [Line2D([0],[0], color=color, lw=1.5)]
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_h(ax, run_folder: str, palette: dict) -> None:
    """Panel H: sigma_NEF and std(PE) vs n_neurons.

    Shows that both response noise and PE noise scale with network size,
    with the human sigma reference line for comparison.
    """
    from scipy.stats import pearsonr

    run_dir  = data_path("runs") / run_folder
    scan_path = run_dir / "n_neurons_scan.pkl"
    if not scan_path.exists():
        _placeholder(ax, "No n_neurons scan data\n(run extras_carrabin.py)")
        return

    df = pd.read_pickle(scan_path)
    sigma_df    = pd.read_pickle(run_dir / "RNN_sigma_carrabin_sigma.pkl")
    human_sigma = float(
        sigma_df[sigma_df["source"]=="human"]["sigma"].mean()
    )

    color_sigma = list(palette.values())[0]
    color_pe    = list(palette.values())[1]
    n_vals      = sorted(df["n_neurons"].unique())

    # Thin grey lines — drawn first so they sit behind model curves
    human_sigmas = sigma_df[sigma_df["source"]=="human"]["sigma"].values
    for hs in human_sigmas:
        ax.axhline(hs, color="0.78", lw=0.3, zorder=0)

    if "pid" in df.columns and df["pid"].nunique() > 1:
        # Multiple pids — use lineplot with SD
        sns.lineplot(data=df, x="n_neurons", y="sigma",
                     color=color_sigma, lw=1.8,
                     errorbar="sd", err_style="band",
                     label="Response noise (σ)", ax=ax)
        sns.lineplot(data=df, x="n_neurons", y="pe_std",
                     color=color_pe, lw=1.8,
                     linestyle="--", errorbar="sd", err_style="band",
                     label="Std PE at readout", ax=ax)
    else:
        ax.plot(n_vals, df.set_index("n_neurons")["sigma"][n_vals].values,
                "-", color=color_sigma, lw=1.8, label="Response noise (σ)")
        ax.plot(n_vals, df.set_index("n_neurons")["pe_std"][n_vals].values,
                "--", color=color_pe, lw=1.8, label="Std PE at readout")



    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Noise")
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(n) for n in n_vals])
    # Append human reference line as last legend entry
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="0.78", lw=0.8))
    labels.append("Human response noise")
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9, loc="upper right")
    sns.despine(ax=ax, top=True, right=True)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_folder",
        type=str,
        default="carrabin",
        help="Source folder for fitted NEF params",
    )
    parser.add_argument("--out_folder", type=str, default="carrabin")
    parser.add_argument(
        "--scan_pid",
        type=int,
        default=None,
        help="Single PID alias for --scan_pids",
    )
    parser.add_argument(
        "--scan_pids",
        type=int,
        nargs="+",
        default=[14],
        help="PIDs to use for n_neurons_scan (default: [14])",
    )
    parser.add_argument(
        "--n_neurons_list",
        type=int,
        nargs="+",
        default=list(N_NEURONS_LIST),
    )
    parser.add_argument(
        "--extra_models",
        nargs="*",
        default=["NoisyCounting", "RNN"],
        help="Additional models to include in top-row panels (default: ['RNN'])",
    )
    args = parser.parse_args()

    if args.scan_pid is not None:
        args.scan_pids = [args.scan_pid]

    model_order = MODEL_ORDER + [m for m in args.extra_models if m not in MODEL_ORDER]

    apply_style()
    _pal = get_palette(len(model_order))
    palette = {m: _pal[i] for i, m in enumerate(model_order)}
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

    # ── bottom row: E–H archived, pending new noise analysis ─────────────────
    _plot_panel_e(row1[0], run_folder=args.run_folder, palette=palette)
    _plot_panel_f(row1[1], run_folder=args.run_folder, palette=palette)
    _plot_panel_g(row1[2], run_folder=args.run_folder, palette=palette)
    _plot_panel_h(row1[3], run_folder=args.run_folder, palette=palette)

    label_panels(axes)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "figure_carrabin.png", dpi=300)
    plt.savefig(FIGURES_DIR / "figure_carrabin.pdf")
    plt.savefig(FIGURES_DIR / "figure_carrabin.svg")
    print("Saved figures/figure_carrabin.{png,pdf,svg}")


if __name__ == "__main__":
    main()
