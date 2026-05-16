"""
Behavioral analysis figure for the Diederen dataset (pure data, no model fits).

Panel A uses OBS_MAX=10 on the x-axis; long_df is truncated to observations
≤ OBS_MAX before plotting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _plot_panel_a(ax, long_df: pd.DataFrame) -> None:
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
            f"Panel A: {excluded_count}/{n_total} pids "
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


def _plot_panel_b(ax, carry: pd.DataFrame) -> None:
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


def _plot_panel_c(ax, long_df: pd.DataFrame) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral decay figure for Diederen dataset."
    )
    parser.add_argument("--out_folder", type=str, default=None)
    args = parser.parse_args()

    apply_style()

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

    fig, axes = plt.subplots(
        1, 4, figsize=FIGURE_SIZE, constrained_layout=True
    )
    _plot_panel_a(axes[0], long_df)
    _plot_panel_b(axes[1], carry)
    _plot_panel_c(axes[2], long_df)
    axes[3].set_visible(False)

    label_panels([axes[0], axes[1], axes[2]])

    out_dir = Path(args.out_folder) if args.out_folder else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "figure_diederen.png", dpi=300)
    plt.savefig(out_dir / "figure_diederen.pdf")
    plt.close(fig)
    print(f"Saved {out_dir}/figure_diederen.{{png,pdf}}")


if __name__ == "__main__":
    main()
