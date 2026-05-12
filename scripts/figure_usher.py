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
import numpy as np
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

_COLORBLIND = sns.color_palette("colorblind")
DATASET = "usher"


def _model_order(include_rl_lambda: bool) -> list[str]:
    order = ["Mean", "EmpiricalWeights", "RL"]
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
    if model_type == "EmpiricalWeights":
        return "EmpWeights"
    return model_type


def _model_color(mt: str, palette: dict) -> str:
    """Color for math model type ``mt``; PopulationCoding / RL_lambda use shared fallbacks."""
    if mt == "PopulationCoding":
        return palette.get("PopulationCoding", _COLORBLIND[4])
    if mt == "RL_lambda":
        return palette.get("RL_lambda", _COLORBLIND[5])
    if mt == "EmpiricalWeights":
        # TODO: add EmpiricalWeights to get_palette() in utils/plot_style.py
        return palette.get("EmpiricalWeights", _COLORBLIND[6])
    d = _display(mt)
    if d == "Mean":
        return palette.get("Mean", _COLORBLIND[0])
    if d == "RL":
        return palette.get("RL", _COLORBLIND[1])
    if d == "PopCode":
        return palette.get("PopulationCoding", _COLORBLIND[4])
    if d == "RL_λ":
        return palette.get("RL_lambda", _COLORBLIND[5])
    return palette.get(d, "0.5")


def _placeholder(ax, text: str) -> None:
    """Axes with centered italic status text (no ticks or spines)."""
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


def _blank_panel(ax) -> None:
    """Blank axes: no ticks, spines, or axis labels."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")


def _plot_panel_a(ax) -> None:
    """Panel A: embed ``figures/usher_task.pdf`` (or blank axes if missing)."""
    pdf_path = FIGURES_DIR / "usher_task.pdf"
    if not pdf_path.exists():
        _blank_panel(ax)
        ax.set_aspect("equal")
        ax.set_anchor("C")
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
            _blank_panel(ax)
            ax.set_aspect("equal")
            ax.set_anchor("C")
            return
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            _blank_panel(ax)
            ax.set_aspect("equal")
            ax.set_anchor("C")
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


def _serial_weights(df: pd.DataFrame, use_human_values: bool = False) -> pd.DataFrame:
    """Per-pid OLS coefficients of obs-10 response on v1..v10 (wide item values); columns obs_1..obs_10."""
    work = df.copy()
    if use_human_values:
        work = work.drop(columns=["value"], errors="ignore")
        human_vals = pd.read_pickle(data_path("usher.pkl"))[
            ["pid", "trial", "observation", "value"]
        ].drop_duplicates(["pid", "trial", "observation"])
        work = work.merge(
            human_vals,
            on=["pid", "trial", "observation"],
            how="inner",
        )
    if "value" not in work.columns:
        return pd.DataFrame(columns=[f"obs_{i}" for i in range(1, 11)])

    wide = work.pivot_table(
        index=["pid", "trial"],
        columns="observation",
        values="value",
        aggfunc="first",
    )
    for obs in range(1, 11):
        if obs not in wide.columns:
            wide[obs] = np.nan
    wide = wide.reindex(columns=list(range(1, 11))).astype(float)
    wide.columns = [f"v{i}" for i in range(1, 11)]

    resp10 = work.loc[work["observation"] == 10, ["pid", "trial", "response"]].drop_duplicates(
        ["pid", "trial"]
    )
    merged = wide.reset_index().merge(resp10, on=["pid", "trial"], how="inner")
    vcols = [f"v{i}" for i in range(1, 11)]
    merged = merged.dropna(subset=vcols + ["response"])

    out_rows: list[dict] = []
    for pid, grp in merged.groupby("pid"):
        if grp.empty:
            continue
        X = np.column_stack([np.ones(len(grp)), grp[vcols].to_numpy(dtype=float)])
        y = grp["response"].to_numpy(dtype=float)
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        if coef.size < 11:
            continue
        w = coef[1:11]
        row = {"pid": int(pid)}
        for i in range(10):
            row[f"obs_{i + 1}"] = float(w[i])
        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame(columns=[f"obs_{i}" for i in range(1, 11)])
    out = pd.DataFrame(out_rows).set_index("pid")
    return out


def _plot_panel_c(
    ax,
    palette: dict,
    run_folder: str,
    include_rl_lambda: bool,
    panel_c_show_models: bool,
) -> None:
    """Panel C: task error vs sequence std (human per-pid gray lines; optional pooled model lines)."""
    df = _usher_seq_std_task_error()
    if df.empty or len(df) < 2:
        _placeholder(ax, "No usher data")
        return

    line_kw_pid = {"linewidth": 0.6, "alpha": 0.4}
    line_kw_pop: dict = {"linewidth": 2.0}

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
        ax.lines[-1].set_label("_pid")

    sns.regplot(
        data=df,
        x="seq_std",
        y="task_error",
        ax=ax,
        scatter=False,
        truncate=True,
        ci=95,
        color="black",
        line_kws=line_kw_pop,
    )
    if panel_c_show_models:
        ax.lines[-1].set_label("Human")

    if panel_c_show_models:
        run_dir = data_path("runs") / run_folder
        line_kw_mod = {"linewidth": 2.0}
        active = _model_order(include_rl_lambda)
        for mt in active:
            mp = run_dir / f"{mt}_{DATASET}_responses.pkl"
            if not mp.exists():
                continue
            trial_pool = _model_per_trial_seq_std_task_error(pd.read_pickle(mp))
            if len(trial_pool) < 2:
                continue
            disp = _display(mt)
            sns.regplot(
                data=trial_pool,
                x="seq_std",
                y="task_error",
                ax=ax,
                scatter=False,
                truncate=True,
                ci=95,
                color=_model_color(mt, palette),
                line_kws=line_kw_mod,
            )
            ax.lines[-1].set_label(disp)
        ax.legend(frameon=False)
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
    """Panel B: cross-participant RMSE by model (boxplot)."""
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
    pal = {
        m: _model_color(next(mt for mt in model_order if _display(mt) == m), palette)
        for m in available
    }
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

    if len(available) >= 2 and not df.empty:
        annotate_violins(ax, df, "model_disp", "plot_loss", available)


def _plot_panel_d(ax, run_folder: str, palette: dict, include_rl_lambda: bool) -> None:
    """Panel D: absolute slope mismatch between each model and human (seq_std vs task_error)."""
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

    pal = {
        m: _model_color(next(mt for mt in model_order if _display(mt) == m), palette)
        for m in available
    }
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
    if len(available) >= 2 and not plot_df.empty:
        annotate_violins(ax, plot_df, "model_disp", "slope_err", available)


def _weights_long(weights: pd.DataFrame, obs_cols: list[str]) -> pd.DataFrame:
    """Long format: pid, observation (1–10), weight (for lineplot + errorbar across pids)."""
    long = weights.reset_index().melt(
        id_vars="pid", value_vars=obs_cols, var_name="_c", value_name="weight"
    )
    long["observation"] = long["_c"].str.replace("obs_", "", regex=False).astype(int)
    return long.drop(columns=["_c"])


def _plot_panel_e(ax, run_folder: str, palette: dict, include_rl_lambda: bool) -> None:
    """Panel E: serial-position regression weights (human + models; SE across pids)."""
    obs_cols = [f"obs_{i}" for i in range(1, 11)]

    human = pd.read_pickle(data_path("usher.pkl"))
    human_w = _serial_weights(human, use_human_values=False)
    if human_w.empty:
        _placeholder(ax, "No human weight data")
        return

    human_long = _weights_long(human_w, obs_cols)
    sns.lineplot(
        data=human_long,
        x="observation",
        y="weight",
        ax=ax,
        errorbar="se",
        color=palette.get("Human", "black"),
        linewidth=2.0,
        label="Human",
    )

    run_dir = data_path("runs") / run_folder
    model_order = _model_order(include_rl_lambda)
    for mt in model_order:
        resp_path = run_dir / f"{mt}_{DATASET}_responses.pkl"
        if not resp_path.exists():
            continue
        model_df = pd.read_pickle(resp_path)
        mw = _serial_weights(model_df, use_human_values=True)
        if mw.empty:
            continue
        model_long = _weights_long(mw, obs_cols)
        sns.lineplot(
            data=model_long,
            x="observation",
            y="weight",
            ax=ax,
            errorbar="se",
            color=_model_color(mt, palette),
            linewidth=2.0,
            label=_display(mt),
        )

    ax.set_xticks(list(range(1, 11)))
    ax.set_xlabel("Observation")
    ax.set_ylabel("Regression weight")
    ax.legend(frameon=False)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_f(ax, run_folder: str, palette: dict, include_rl_lambda: bool) -> None:
    """Panel F: RMSE between each model's per-pid weight profile and human (boxplot)."""
    obs_cols = [f"obs_{i}" for i in range(1, 11)]

    human = pd.read_pickle(data_path("usher.pkl"))
    human_w = _serial_weights(human, use_human_values=False)
    if human_w.empty:
        _placeholder(ax, "No human weight data")
        return

    run_dir = data_path("runs") / run_folder
    model_order = _model_order(include_rl_lambda)
    rows: list[dict] = []
    for mt in model_order:
        resp_path = run_dir / f"{mt}_{DATASET}_responses.pkl"
        if not resp_path.exists():
            continue
        model_df = pd.read_pickle(resp_path)
        mw = _serial_weights(model_df, use_human_values=True)
        common = human_w.index.intersection(mw.index)
        for pid in common:
            diff = mw.loc[pid, obs_cols].to_numpy(dtype=float) - human_w.loc[
                pid, obs_cols
            ].to_numpy(dtype=float)
            rmse = float(np.sqrt(np.mean(diff**2)))
            rows.append({"pid": int(pid), "model_disp": _display(mt), "rmse": rmse})

    if not rows:
        _placeholder(ax, "No model weight RMSE")
        return

    plot_df = pd.DataFrame(rows)
    order = [_display(m) for m in model_order]
    available = [m for m in order if m in set(plot_df["model_disp"])]
    if not available:
        _placeholder(ax, "No RMSE data")
        return

    pal = {
        m: _model_color(next(mt for mt in model_order if _display(mt) == m), palette)
        for m in available
    }
    sns.boxplot(
        data=plot_df,
        x="model_disp",
        y="rmse",
        order=available,
        hue="model_disp",
        palette=pal,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Weight profile RMSE")
    sns.despine(ax=ax, top=True, right=True)
    if len(available) >= 2 and not plot_df.empty:
        annotate_violins(ax, plot_df, "model_disp", "rmse", available)


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
        help="Include RL_lambda (display RL_λ) in panels B–F",
    )
    parser.add_argument(
        "--panel_c_show_models",
        action="store_true",
        default=False,
        help="Panel C: overlay population regplots per model + legend",
    )
    args = parser.parse_args()

    apply_style()
    palette = get_palette()

    fig, axes = plt.subplots(2, 4, figsize=FIGURE_SIZE, constrained_layout=True)
    row0, row1 = axes[0], axes[1]

    _plot_panel_a(row0[0])
    _plot_panel_b(row0[1], args.run_folder, palette, args.include_rl_lambda)
    _plot_panel_c(
        row0[2],
        palette,
        args.run_folder,
        args.include_rl_lambda,
        args.panel_c_show_models,
    )
    _plot_panel_d(row0[3], args.run_folder, palette, args.include_rl_lambda)
    _plot_panel_e(row1[0], args.run_folder, palette, args.include_rl_lambda)
    _plot_panel_f(row1[1], args.run_folder, palette, args.include_rl_lambda)
    for ax in row1[2:]:
        _blank_panel(ax)

    label_panels(axes)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "figure_usher.png", dpi=300)
    plt.savefig(FIGURES_DIR / "figure_usher.pdf")
    plt.savefig(FIGURES_DIR / "figure_usher.svg")
    print("Saved figures/figure_usher.{png,pdf,svg}")


if __name__ == "__main__":
    main()
