#!/usr/bin/env python3
"""Usher summary figure: row 1 (panels A–D), row 2 reserved (E–H)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    annotate_violins,
    apply_style,
    get_palette,
    label_panels,
)

DATASET = "usher"


def _model_order(include_rl_lambda: bool) -> list[str]:
    order = ["Mean", "RL"]
    if include_rl_lambda:
        order.append("RL_lambda")
    order.append("PopulationCoding")
    return order


def _display(model_type: str) -> str:
    if model_type.startswith("NEF"):
        return "NEF"
    if model_type == "PopulationCoding":
        return "PopCode"
    if model_type == "RL_lambda":
        return "RL_λ"
    return model_type


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
    """Bare axes when PDF is missing or conversion fails."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _blank_panel(ax) -> None:
    """Reserved / placeholder: no ticks, spines, or text."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")


def _plot_panel_a(ax) -> None:
    """Render first page of figures/usher_task.pdf into panel A."""
    pdf_path = FIGURES_DIR / "usher_task.pdf"
    if not pdf_path.exists():
        _empty_pdf_panel(ax)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "usher_task"
        cmd = [
            "pdftoppm",
            "-png",
            "-singlefile",
            str(pdf_path),
            str(out_prefix),
        ]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
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
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _usher_seq_std_task_error() -> pd.DataFrame:
    """
    One row per (pid, trial): seq_std over 10 values, task_error from final response.
    """
    human = pd.read_pickle(data_path("usher.pkl"))
    trial_stats = (
        human.groupby(["pid", "trial"], sort=False)["value"]
        .agg(seq_std="std", true_mean="mean")
        .reset_index()
    )
    final = human.loc[human["observation"] == 10, ["pid", "trial", "response"]]
    df = final.merge(trial_stats, on=["pid", "trial"], how="inner")
    df["task_error"] = (df["response"].astype(float) - df["true_mean"].astype(float)).abs()
    return df


def _model_per_trial_seq_std_task_error(model_df: pd.DataFrame) -> pd.DataFrame:
    """Per (pid, trial): seq_std / true_mean from human usher; task_error from model obs 10."""
    human_src = pd.read_pickle(data_path("usher.pkl"))
    trial_stats = (
        human_src.groupby(["pid", "trial"], sort=False)["value"]
        .agg(seq_std="std", true_mean="mean")
        .reset_index()
    )
    final = model_df.loc[model_df["observation"] == 10, ["pid", "trial", "response"]]
    out = final.merge(trial_stats, on=["pid", "trial"], how="inner")
    out["task_error"] = (
        out["response"].astype(float) - out["true_mean"].astype(float)
    ).abs()
    return out


def _plot_panel_c(
    ax, palette: dict, run_folder: str, include_rl_lambda: bool
) -> None:
    """Sequence std vs task error: per-pid regressions then population."""
    df = _usher_seq_std_task_error()
    if df.empty or len(df) < 2:
        _placeholder(ax, "No usher data")
        return

    pop_color = palette.get("Mean", sns.color_palette("colorblind")[0])
    line_kw_pid = {"linewidth": 0.6, "alpha": 0.4}
    line_kw_pop = {"linewidth": 2.0}

    for pid in sorted(df["pid"].unique()):
        sub = df[df["pid"] == pid]
        if len(sub) < 2:
            continue
        sns.regplot(
            data=sub,
            x="seq_std",
            y="task_error",
            ax=ax,
            scatter=False,
            truncate=True,
            ci=None,
            color="0.75",
            line_kws=line_kw_pid,
        )

    if include_rl_lambda:
        run_dir = data_path("runs") / run_folder
        rlp = run_dir / f"RL_lambda_{DATASET}_responses.pkl"
        if rlp.exists():
            rll = pd.read_pickle(rlp)
            trial_df = _model_per_trial_seq_std_task_error(rll)
            rl_lambda_color = sns.color_palette("colorblind")[2]
            line_kw_rll = {"linewidth": 0.6, "alpha": 0.55}
            for pid in sorted(trial_df["pid"].unique()):
                sub = trial_df[trial_df["pid"] == pid]
                if len(sub) < 2:
                    continue
                sns.regplot(
                    data=sub,
                    x="seq_std",
                    y="task_error",
                    ax=ax,
                    scatter=False,
                    truncate=True,
                    ci=None,
                    color=rl_lambda_color,
                    line_kws=line_kw_rll,
                )

    sns.regplot(
        data=df,
        x="seq_std",
        y="task_error",
        ax=ax,
        scatter=False,
        truncate=True,
        ci=95,
        color=pop_color,
        line_kws=line_kw_pop,
    )
    ax.set_xlabel("Sequence std")
    ax.set_ylabel("Task error")
    sns.despine(ax=ax, top=True, right=True)


def _seq_std_slope(df: pd.DataFrame) -> pd.Series:
    """
    Per-pid OLS slope of task_error ~ seq_std (linregress), matching panel C
    construction. ``df`` is full human usher data, or model responses (uses
    ``usher.pkl`` for ``value`` / ``true_mean`` when ``value`` is absent).
    """
    from scipy.stats import linregress

    if "value" in df.columns:
        human_src = df
    else:
        human_src = pd.read_pickle(data_path("usher.pkl"))
    trial_stats = (
        human_src.groupby(["pid", "trial"], sort=False)["value"]
        .agg(seq_std="std", true_mean="mean")
        .reset_index()
    )
    final = df.loc[df["observation"] == 10, ["pid", "trial", "response"]]
    per_trial = final.merge(trial_stats, on=["pid", "trial"], how="inner")
    per_trial["task_error"] = (
        per_trial["response"].astype(float) - per_trial["true_mean"].astype(float)
    ).abs()

    slopes: dict[int, float] = {}
    for pid, grp in per_trial.groupby("pid"):
        if len(grp) < 2:
            continue
        slope, _, _, _, _ = linregress(
            grp["seq_std"].to_numpy(dtype=float),
            grp["task_error"].to_numpy(dtype=float),
        )
        slopes[int(pid)] = float(slope)
    return pd.Series(slopes, name="slope")


def _get_loss(perf_df: pd.DataFrame) -> pd.Series:
    """Return response_component if available, else cv_loss_mean (model_performance)."""
    if "response_component" in perf_df.columns:
        rc = perf_df["response_component"]
        if rc.notna().all():
            return rc
    return perf_df["cv_loss_mean"]


def _plot_panel_b(ax, run_folder: str, palette: dict, include_rl_lambda: bool) -> None:
    """Per-pid RMSE distribution per model (logic aligned with scripts/model_performance.py)."""
    run_dir = data_path("runs") / run_folder
    model_order = _model_order(include_rl_lambda)
    rows = []
    for mt in model_order:
        f = run_dir / f"{mt}_{DATASET}_performance.pkl"
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
    pal = {}
    for m in available:
        if m == "PopCode":
            # TODO: add PopulationCoding to get_palette() in plot_style.py; model_performance
            #  uses get_palette() keys only — no PopulationCoding entry yet
            pal[m] = palette.get(
                "PopulationCoding",
                sns.color_palette("colorblind")[4],
            )
        elif m == "RL_λ":
            pal[m] = palette.get("RL_lambda", sns.color_palette("colorblind")[5])
        else:
            pal[m] = palette.get(m, "0.5")
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
    sns.despine(ax=ax, top=True, right=True)

    if len(available) >= 2:
        annotate_violins(ax, df, "model_disp", "plot_loss", available)


def _plot_panel_d(ax, run_folder: str, palette: dict, include_rl_lambda: bool) -> None:
    """|model slope − human slope| on task_error ~ seq_std (panel C), per pid."""
    human = pd.read_pickle(data_path("usher.pkl"))
    human_slope = _seq_std_slope(human)
    if human_slope.empty:
        _placeholder(ax, "No human slopes")
        return

    run_dir = data_path("runs") / run_folder
    model_order = _model_order(include_rl_lambda)
    rows: list[dict] = []
    for mt in model_order:
        resp_path = run_dir / f"{mt}_{DATASET}_responses.pkl"
        if not resp_path.exists():
            continue
        model_df = pd.read_pickle(resp_path)
        m_slope = _seq_std_slope(model_df)
        common = human_slope.index.intersection(m_slope.index)
        for pid in common:
            err = abs(float(m_slope.loc[pid]) - float(human_slope.loc[pid]))
            rows.append(
                {"pid": int(pid), "model_disp": _display(mt), "slope_err": err}
            )

    if not rows:
        _placeholder(ax, "No model responses")
        return

    plot_df = pd.DataFrame(rows)
    order = [_display(m) for m in model_order]
    available = [m for m in order if m in set(plot_df["model_disp"])]
    if not available:
        _placeholder(ax, "No slope data")
        return

    pal = {}
    for m in available:
        if m == "PopCode":
            pal[m] = palette.get(
                "PopulationCoding",
                sns.color_palette("colorblind")[4],
            )
        elif m == "RL_λ":
            pal[m] = palette.get("RL_lambda", sns.color_palette("colorblind")[5])
        else:
            pal[m] = palette.get(m, "0.5")
    sns.boxplot(
        data=plot_df,
        x="model_disp",
        y="slope_err",
        order=available,
        hue="model_disp",
        palette=pal,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Slope error (|model − human|)")
    sns.despine(ax=ax, top=True, right=True)
    if len(available) >= 2:
        annotate_violins(ax, plot_df, "model_disp", "slope_err", available)


def main() -> None:
    parser = argparse.ArgumentParser(description="Usher summary figure (panels A–H).")
    parser.add_argument(
        "--run_folder",
        type=str,
        default="response",
        help="Run folder under data/runs/ for performance pickles",
    )
    parser.add_argument(
        "--include_rl_lambda",
        action="store_true",
        default=False,
        help="Include RL_lambda (display RL_λ) in panels B, C, and D",
    )
    args = parser.parse_args()

    apply_style()
    palette = get_palette()

    fig, axes = plt.subplots(2, 4, figsize=FIGURE_SIZE, constrained_layout=True)
    row0, row1 = axes[0], axes[1]

    _plot_panel_a(row0[0])
    _plot_panel_b(row0[1], args.run_folder, palette, args.include_rl_lambda)
    _plot_panel_c(row0[2], palette, args.run_folder, args.include_rl_lambda)
    _plot_panel_d(row0[3], args.run_folder, palette, args.include_rl_lambda)
    for ax in row1:
        _blank_panel(ax)

    label_panels(axes)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "figure_usher.png", dpi=300)
    plt.savefig(FIGURES_DIR / "figure_usher.pdf")
    plt.savefig(FIGURES_DIR / "figure_usher.svg")
    print("Saved figures/figure_usher.{png,pdf,svg}")


if __name__ == "__main__":
    main()
