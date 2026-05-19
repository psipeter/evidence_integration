"""
Diederen dataset figure: 2×4 layout, panels A–E.

Panel A: task diagram (figures/diederen_task.pdf)
Panel B: model RMSE boxplot (requires model fits in --run_folder)
Panel C: response change vs observation (raw vs pre-switch, post-switch)
Panel D: carryover bias vs observations since context switch
Panel E: response change by SD condition and drug condition
Panel F: response change diff (post-switch − pre-switch) vs absence_length
Panels G–H: reserved for future panels
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
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
    pvalue_to_stars,
)

# OBS_MAX: panel A raw/clean x-axis and global long_df observation filter.
OBS_MAX = 10
OBS_MIN = 2  # obs=1 has no previous response (diff is NaN)

PID_LINE_COLOR = "0.75"
MIN_LAMBDA_PLOT = 0.02  # skip degenerate per-pid power-law fits
MIN_PIDS_PER_OBS = 15  # minimum pids required to plot a group mean

MODEL_ORDER = ["Mean", "RL", "RL_lambda", "PearceHall", "NEF_recurrent", "NEF_synaptic"]


def _display(model_type: str) -> str:
    return "NEF" if model_type.startswith("NEF") else model_type


def _abs_delta_long(human: pd.DataFrame) -> pd.DataFrame:
    """Long-format per-trial |Δresponse| with row metadata carried through."""
    pieces = []
    for (pid, trial), tgrp in human.groupby(["pid", "trial"], sort=False):
        g = tgrp.sort_values("observation").copy()
        g["delta"] = g["response"].diff().abs()
        pieces.append(g)
    if not pieces:
        return pd.DataFrame(
            columns=[
                "pid",
                "trial",
                "trial_in_session",
                "observation",
                "delta",
                "group",
                "sd_value",
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def _add_oss_to_long_df(long_df: pd.DataFrame) -> pd.DataFrame:
    """Add oss column to long_df. oss=1 is first obs after a switch; NaN if
    no switch has yet occurred in this trial."""
    out = long_df.copy().sort_values(["pid", "trial", "trial_in_session"])
    tis_gap = out.groupby(["pid", "trial"])["trial_in_session"].diff()
    switched = (tis_gap > 1).fillna(False)
    oss_arr = np.full(len(out), np.nan)
    for (_pid, _trial), grp in out.groupby(["pid", "trial"], sort=False):
        grp_sorted = grp.sort_values("trial_in_session")
        idx = grp_sorted.index.tolist()
        sw = switched.loc[idx].to_numpy()
        count = np.nan
        for i, s in enumerate(sw):
            if i == 0:
                count = np.nan
            elif s:
                count = 1.0
            else:
                count = (count + 1.0) if not np.isnan(count) else np.nan
            oss_arr[out.index.get_loc(idx[i])] = count
    out["oss"] = oss_arr
    return out


def _pre_switch_rows(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the rows from each (pid, trial) that precede the first
    context switch. The switch row itself (switched==True) is excluded.

    Requires long_df to have 'tis_gap' and 'switched' columns.
    If these are not present, compute them first:
        long_df['tis_gap'] = long_df.groupby(['pid','trial'])['trial_in_session'].diff()
        long_df['switched'] = long_df['tis_gap'] > 1
    """
    pieces = []
    for (_pid, _trial), grp in long_df.groupby(["pid", "trial"], sort=False):
        grp_sorted = grp.sort_values("trial_in_session")
        sw = grp_sorted["switched"].fillna(False).to_numpy()
        switch_positions = np.flatnonzero(sw)
        if len(switch_positions) == 0:
            pieces.append(grp_sorted)
        else:
            pieces.append(grp_sorted.iloc[: switch_positions[0]])
    if not pieces:
        return long_df.iloc[0:0]
    return pd.concat(pieces, ignore_index=True)


def _compute_carryover_metrics(valid: pd.DataFrame) -> pd.DataFrame:
    """
    Add carryover-analysis columns to the filtered behavioral dataframe.

    oss: observations since last context switch within (pid, trial).
         1 = first obs after returning; NaN for first obs of a trial.
         Computed via explicit loop to avoid groupby.apply issues with Int64.

    running_mean_this: expanding mean of prior rewards in this distribution
         (shift(1)), i.e. the optimal running estimate.

    running_mean_other: most recent running_mean_this of the co-active
         distribution (other distrib_index in same pid+session) at
         trial_in_session strictly before the current row.

    pull = running_mean_other - running_mean_this
    bias = response - running_mean_this
    """
    out = valid.copy().sort_values(["pid", "session", "trial_in_session"])
    out["running_mean_this"] = out.groupby(["pid", "trial"])["value"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    out["tis_gap"] = out.groupby(["pid", "trial"])["trial_in_session"].diff()
    out["switched"] = out["tis_gap"] > 1

    # oss via explicit loop (avoids Int64 / groupby.apply issues)
    oss_arr = np.full(len(out), np.nan)
    for (_pid, _trial), grp in out.groupby(["pid", "trial"], sort=False):
        grp_sorted = grp.sort_values("trial_in_session")
        idx = grp_sorted.index.tolist()
        switched = grp_sorted["switched"].fillna(False).to_numpy()
        count = np.nan
        for i, _s in enumerate(switched):
            if i == 0:
                count = np.nan
            elif _s:
                count = 1.0
            else:
                count = (count + 1.0) if not np.isnan(count) else np.nan
            oss_arr[out.index.get_loc(idx[i])] = count
    out["oss"] = oss_arr

    # running_mean_other via lookup within (pid, session)
    out["running_mean_other"] = np.nan
    for (_pid, sess), grp in out.groupby(["pid", "session"], sort=False):
        distribs = grp["distrib_index"].unique()
        if len(distribs) != 2:
            continue
        d1, d2 = sorted(distribs)
        g1 = grp[grp["distrib_index"] == d1].sort_values("trial_in_session")
        g2 = grp[grp["distrib_index"] == d2].sort_values("trial_in_session")
        for g_focal, g_other in [(g1, g2), (g2, g1)]:
            for _, row in g_focal.iterrows():
                prev = g_other[g_other["trial_in_session"] < row["trial_in_session"]]
                if len(prev) > 0:
                    out.loc[row.name, "running_mean_other"] = prev.iloc[
                        -1
                    ]["running_mean_this"]

    out["pull"] = out["running_mean_other"] - out["running_mean_this"]
    out["bias"] = out["response"] - out["running_mean_this"]
    return out


def _fit_power_law_per_pid(pid_obs: pd.DataFrame) -> pd.DataFrame:
    """Per-participant ``A · n^{-λ}`` fit on columns [pid, observation, delta]."""

    def power_law(n, A, lam):
        return A * np.power(np.asarray(n, dtype=float), -lam)

    rows: list[dict] = []
    for pid, grp in pid_obs.groupby("pid", sort=False):
        gg = grp.sort_values("observation")
        n_obs = gg["observation"].to_numpy(dtype=float)
        y = gg["delta"].to_numpy(dtype=float)
        if len(n_obs) < 3 or not (
            np.all(np.isfinite(n_obs)) and np.all(np.isfinite(y))
        ):
            rows.append({"pid": int(pid), "A": np.nan, "lambda_": np.nan})
            continue
        try:
            popt, _ = curve_fit(
                power_law,
                n_obs,
                y,
                p0=[0.1, 0.5],
                bounds=([0.0, 0.0], [2.0, 2.0]),
                maxfev=2000,
            )
            rows.append(
                {"pid": int(pid), "A": float(popt[0]), "lambda_": float(popt[1])}
            )
        except (RuntimeError, ValueError, TypeError):
            rows.append({"pid": int(pid), "A": np.nan, "lambda_": np.nan})
    return pd.DataFrame(rows)


def _filter_pid_means_by_n(pid_means: pd.DataFrame) -> pd.DataFrame:
    obs_counts = pid_means.groupby("observation")["pid"].count()
    valid_obs = obs_counts[obs_counts >= MIN_PIDS_PER_OBS].index
    return pid_means[pid_means["observation"].isin(valid_obs)]


def _placeholder(ax, text: str) -> None:
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=8,
        color="0.5",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


def _plot_panel_a(ax) -> None:
    """Render first page of figures/diederen_task.pdf into panel A."""
    pdf_path = FIGURES_DIR / "diederen_task.pdf"
    if not pdf_path.exists():
        _placeholder(ax, "diederen_task.pdf not found")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "diederen_task"
        cmd = ["pdftoppm", "-png", "-singlefile", str(pdf_path), str(out_prefix)]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            _placeholder(ax, "pdftoppm failed")
            return
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            _placeholder(ax, "diederen_task.pdf render failed")
            return
        img = mpimg.imread(img_path)
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _plot_panel_b(
    ax, run_folder: str, palette: dict, model_order: list[str]
) -> None:
    """Per-pid RMSE boxplot for all fitted models."""
    run_dir = data_path("runs") / run_folder
    rows = []
    for mt in model_order:
        f = run_dir / f"{mt}_diederen_performance.pkl"
        if not f.exists():
            continue
        perf = pd.read_pickle(f).copy()
        loss_col = "loss" if "loss" in perf.columns else "cv_loss_mean"
        perf["plot_loss"] = perf[loss_col]
        perf["model_disp"] = mt
        rows.append(perf[["pid", "model_disp", "plot_loss"]])
    if not rows:
        _placeholder(ax, "No performance data\n(run model fits first)")
        return
    df = pd.concat(rows, ignore_index=True)
    available = [m for m in model_order if m in set(df["model_disp"])]
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
    ax.set_ylabel("Response error (RMSE)")
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_c(ax, long_df: pd.DataFrame) -> None:
    palette_colors = get_palette()
    raw_color = palette_colors[0]
    clean_color = palette_colors[1]
    post_color = palette_colors[2]

    df = long_df.copy()
    if "switched" not in df.columns:
        df["tis_gap"] = df.groupby(["pid", "trial"])["trial_in_session"].diff()
        df["switched"] = df["tis_gap"] > 1

    pre_df = _pre_switch_rows(df)
    pre_df = pre_df[pre_df["delta"].notna() & (pre_df["observation"] >= OBS_MIN)]

    pre_pid_means = (
        pre_df.groupby(["pid", "observation"])["delta"].mean().reset_index()
    )
    pre_pid_means = _filter_pid_means_by_n(pre_pid_means)

    per_pid_fits = _fit_power_law_per_pid(pre_pid_means)
    n_grid = np.arange(OBS_MIN, OBS_MAX + 1, dtype=float)
    excluded_count = 0
    for _, row in per_pid_fits.iterrows():
        A, lam = float(row["A"]), float(row["lambda_"])
        if not (
            np.isfinite(A)
            and np.isfinite(lam)
            and A > 0
            and lam >= MIN_LAMBDA_PLOT
        ):
            excluded_count += 1
            continue
        ax.plot(
            n_grid,
            A * n_grid ** (-lam),
            color=PID_LINE_COLOR,
            alpha=0.28,
            linewidth=0.9,
            zorder=1,
        )

    n_total = len(per_pid_fits)
    if excluded_count > 0 and n_total > 0:
        print(
            f"Panel C: {excluded_count}/{n_total} pids "
            f"({100 * excluded_count / n_total:.0f}%) excluded from per-pid "
            f"power-law lines (lambda_ < {MIN_LAMBDA_PLOT} or NaN)."
        )

    raw_pid_means = (
        df[df["delta"].notna() & df["observation"].between(OBS_MIN, OBS_MAX)]
        .groupby(["pid", "observation"])["delta"]
        .mean()
        .reset_index()
    )
    raw_pid_means = _filter_pid_means_by_n(raw_pid_means)

    post_switch = df[df["oss"].notna() & df["delta"].notna()].copy()
    post_switch["oss"] = post_switch["oss"].astype(float)
    pid_oss = (
        post_switch.groupby(["pid", "oss"])["delta"].mean().reset_index()
    )
    pid_oss["observation"] = pid_oss["oss"] + 1  # align: oss=1 -> x=2, oss=2 -> x=3, ...
    pid_oss = pid_oss.drop(columns=["oss"])
    pid_oss = _filter_pid_means_by_n(pid_oss)

    if not raw_pid_means.empty:
        sns.lineplot(
            data=raw_pid_means,
            x="observation",
            y="delta",
            color=raw_color,
            linewidth=1.8,
            errorbar="se",
            label="All observations",
            zorder=2,
            ax=ax,
        )
    if not pid_oss.empty:
        sns.lineplot(
            data=pid_oss,
            x="observation",
            y="delta",
            color=post_color,
            linewidth=1.8,
            errorbar="se",
            label="Post-switch",
            zorder=2,
            ax=ax,
        )
    if not pre_pid_means.empty:
        sns.lineplot(
            data=pre_pid_means,
            x="observation",
            y="delta",
            color=clean_color,
            linewidth=2.0,
            errorbar="se",
            label="Pre-first-switch",
            zorder=3,
            ax=ax,
        )

    ax.set_xlim(1.5, 10.5)
    ax.set_xticks(range(2, 11))
    ax.set_ylim(0.0, 0.5)
    ax.set_xlabel("Observation  /  obs since switch")
    ax.set_ylabel("Response change")
    ax.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_d(ax, carry: pd.DataFrame) -> None:
    pal = get_palette(2)
    d = carry.dropna(subset=["pull", "bias", "oss"]).copy()
    d = d[(d["oss"] >= 1) & (d["oss"] <= 5) & (d["pull"] != 0)].copy()
    d["pull_direction"] = np.where(d["pull"] > 0, "Other higher", "Other lower")

    pid_means = (
        d.groupby(["pid", "oss", "pull_direction"])["bias"].mean().reset_index()
    )
    sns.lineplot(
        data=pid_means,
        x="oss",
        y="bias",
        hue="pull_direction",
        errorbar="se",
        linewidth=1.8,
        markers=True,
        palette=[pal[0], pal[1]],
        hue_order=["Other higher", "Other lower"],
        ax=ax,
    )
    ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")

    for oss_val in range(1, 6):
        sub = d[d["oss"] == oss_val].dropna(subset=["pull", "bias"])
        if len(sub) < 10:
            continue
        _, p = spearmanr(sub["pull"], sub["bias"])
        stars = pvalue_to_stars(p)
        if stars:
            ax.text(
                oss_val,
                ax.get_ylim()[1] * 0.95,
                stars,
                ha="center",
                va="top",
                fontsize=7,
            )

    ax.set_xlabel("Observations since context switch")
    ax.set_ylabel("Response bias\n(response − running mean)")
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks(range(1, 6))
    ax.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_e(ax, long_df: pd.DataFrame) -> None:
    pal = get_palette()
    pre_color = pal[1]
    post_color = pal[2]

    df = long_df.copy()
    if "switched" not in df.columns:
        df["tis_gap"] = df.groupby(["pid", "trial"])["trial_in_session"].diff()
        df["switched"] = df["tis_gap"] > 1

    pre_df = _pre_switch_rows(df)
    pre_df = pre_df[pre_df["delta"].notna() & (pre_df["observation"] >= OBS_MIN)]

    for sd, ls, label in [
        (0.1, "-", "Pre-switch, SD=5"),
        (0.3, "--", "Pre-switch, SD=15"),
    ]:
        subset = pre_df[pre_df["sd_value"] == sd]
        if subset.empty:
            continue
        pid_mean = (
            subset.groupby(["pid", "observation"])["delta"].mean().reset_index()
        )
        pid_mean = _filter_pid_means_by_n(pid_mean)
        if pid_mean.empty:
            continue
        sns.lineplot(
            data=pid_mean,
            x="observation",
            y="delta",
            color=pre_color,
            linestyle=ls,
            linewidth=1.8,
            errorbar="se",
            label=label,
            ax=ax,
        )

    post_df = df[df["oss"].notna() & df["delta"].notna()].copy()
    post_df["x"] = post_df["oss"].astype(float) + 1
    post_df["cond"] = post_df["group"].map(
        {
            "CTRL": "Ctrl/PCB",
            "PCB": "Ctrl/PCB",
            "SUL": "DA-mod",
            "BRO": "DA-mod",
        }
    )

    for cond, ls, label in [
        ("Ctrl/PCB", "-", "Post-switch, Ctrl/PCB"),
        ("DA-mod", "--", "Post-switch, DA-mod"),
    ]:
        subset = post_df[post_df["cond"] == cond]
        if subset.empty:
            continue
        pid_mean = (
            subset.groupby(["pid", "x"])["delta"]
            .mean()
            .reset_index()
            .rename(columns={"x": "observation"})
        )
        pid_mean = _filter_pid_means_by_n(pid_mean)
        if pid_mean.empty:
            continue
        sns.lineplot(
            data=pid_mean,
            x="observation",
            y="delta",
            color=post_color,
            linestyle=ls,
            linewidth=1.8,
            errorbar="se",
            label=label,
            ax=ax,
        )

    ax.set_xlim(1.5, 10.5)
    ax.set_xticks(range(2, 11))
    ax.set_ylim(0.0, 0.5)
    ax.set_xlabel("Observation  /  obs since switch")
    ax.set_ylabel("Response change")
    ax.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax, top=True, right=True)


def _compute_switch_alpha_diff(valid: pd.DataFrame) -> pd.DataFrame:
    """
    For each context switch within a (pid, session), compute:
      absence_length: number of consecutive other-distribution observations
                      made while this distribution was inactive
      diff: delta_post - delta_pre
        delta_post: |response change| at the first obs after returning
        delta_pre:  |response change| at the last obs before leaving

    Switch detection uses distrib_index changes within (pid, session).
    Delta values come from valid (free-response, non-missed rows only).
    """
    valid = valid.copy().sort_values(["pid", "session", "trial_in_session"])
    valid["delta"] = valid.groupby(["pid", "trial"])["response"].diff().abs()

    records = []
    for (pid, session), grp in valid.groupby(["pid", "session"], sort=False):
        grp = grp.sort_values("trial_in_session").reset_index(drop=True)
        distribs = sorted(grp["distrib_index"].dropna().unique().tolist())
        if len(distribs) != 2:
            continue
        n = len(grp)
        for i in range(1, n):
            if grp.loc[i, "distrib_index"] == grp.loc[i - 1, "distrib_index"]:
                continue
            absence_length = 0
            j = i - 1
            while j >= 0 and grp.loc[j, "distrib_index"] != grp.loc[i, "distrib_index"]:
                absence_length += 1
                j -= 1
            delta_post = grp.loc[i, "delta"]
            if pd.isna(delta_post):
                continue
            delta_pre = np.nan
            k = j
            while k >= 0 and grp.loc[k, "distrib_index"] == grp.loc[i, "distrib_index"]:
                if not pd.isna(grp.loc[k, "delta"]):
                    delta_pre = float(grp.loc[k, "delta"])
                    break
                k -= 1
            if pd.isna(delta_pre):
                continue
            records.append(
                {
                    "pid": int(pid),
                    "absence_length": int(absence_length),
                    "diff": float(delta_post) - float(delta_pre),
                }
            )
    return pd.DataFrame(records)


def _plot_panel_f(ax, valid: pd.DataFrame) -> None:
    from scipy.stats import linregress, wilcoxon

    sw_df = _compute_switch_alpha_diff(valid)

    sw_clean = sw_df[sw_df["absence_length"].between(1, 6)].copy()

    pal = get_palette()
    color = pal[0]

    sns.regplot(
        data=sw_clean,
        x="absence_length",
        y="diff",
        ax=ax,
        scatter=False,
        line_kws={"color": color, "linewidth": 2.0},
        ci=95,
    )

    ax.set_xlabel("Observations in other distribution (absence length)")
    ax.set_ylabel("Response change diff\n(post-switch − pre-switch)")
    ax.set_xticks(sorted(sw_clean["absence_length"].unique()))
    sns.despine(ax=ax, top=True, right=True)

    n_events = len(sw_clean)
    n_pids = sw_clean["pid"].nunique()

    slope, intercept, r_val, p_ols, se_slope = linregress(
        sw_clean["absence_length"], sw_clean["diff"]
    )

    pid_slopes = []
    for pid, grp in sw_clean.groupby("pid"):
        if grp["absence_length"].nunique() < 2:
            continue
        s, *_ = linregress(grp["absence_length"], grp["diff"])
        pid_slopes.append(float(s))
    pid_slopes = np.array(pid_slopes)
    n_pos = int((pid_slopes > 0).sum())
    n_pids_fit = len(pid_slopes)
    try:
        _, p_wil = wilcoxon(pid_slopes)
    except Exception:
        p_wil = float("nan")

    print(
        f"Panel F — response change diff vs absence length:\n"
        f"  N events={n_events}, N pids={n_pids}\n"
        f"  OLS: slope={slope:.4f} (SE={se_slope:.4f}), r={r_val:.3f}, p={p_ols:.4f}\n"
        f"  Per-pid slopes: mean={pid_slopes.mean():.4f} "
        f"(SE={pid_slopes.std() / np.sqrt(n_pids_fit):.4f}), "
        f"positive={n_pos}/{n_pids_fit}, Wilcoxon p={p_wil:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral decay figure for Diederen dataset."
    )
    parser.add_argument("--out_folder", type=str, default=None)
    parser.add_argument(
        "--run_folder",
        type=str,
        default="refit",
        help="Run folder for model performance files.",
    )
    parser.add_argument(
        "--include_rl_lambda",
        action="store_true",
        default=False,
        help="Include RL_lambda in panel B (excluded by default).",
    )
    args = parser.parse_args()

    model_order = [
        m for m in MODEL_ORDER if args.include_rl_lambda or m != "RL_lambda"
    ]

    apply_style()

    _pal = get_palette(len(model_order))
    palette = {m: _pal[i] for i, m in enumerate(model_order)}
    for mt in model_order:
        disp = _display(mt)
        if disp not in palette:
            palette[disp] = palette[mt]

    human = pd.read_pickle(data_path("diederen.pkl"))
    valid = human[
        ~human["missed"]
        & human["response"].notna()
        & (human["catch_trial"] == False)
    ].copy()

    long_df = _abs_delta_long(valid)
    long_df = long_df[
        long_df["delta"].notna()
        & (long_df["observation"] >= OBS_MIN)
        & (long_df["observation"] <= OBS_MAX)
    ].copy()

    long_df = _add_oss_to_long_df(long_df)
    carry = _compute_carryover_metrics(valid)

    fig, axes = plt.subplots(2, 4, figsize=FIGURE_SIZE, constrained_layout=True)

    _plot_panel_a(axes[0, 0])
    _plot_panel_b(axes[0, 1], args.run_folder, palette, model_order)
    _plot_panel_c(axes[0, 2], long_df)
    _plot_panel_d(axes[0, 3], carry)
    _plot_panel_e(axes[1, 0], long_df)
    axes[1, 1].set_visible(True)
    _plot_panel_f(axes[1, 1], valid)
    for col in range(2, 4):
        axes[1, col].set_visible(False)

    label_panels(
        [
            axes[0, 0],
            axes[0, 1],
            axes[0, 2],
            axes[0, 3],
            axes[1, 0],
            axes[1, 1],
        ]
    )

    out_dir = Path(args.out_folder) if args.out_folder else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "figure_diederen.png", dpi=300)
    plt.savefig(out_dir / "figure_diederen.pdf")
    plt.close(fig)
    print(f"Saved {out_dir}/figure_diederen.{{png,pdf}}")


if __name__ == "__main__":
    main()
