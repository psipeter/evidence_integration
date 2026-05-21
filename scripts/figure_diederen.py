"""
Diederen dataset figure: 2×4 layout, panels A–E.

Panel A: task diagram (figures/diederen_task.pdf)
Panel B: model RMSE boxplot (requires model fits in --run_folder)
Panel C: mean |Δresponse| vs observation (human + models)
Panel D: oss=1 bias by pull direction (Human + models)
Panel E: response change by SD condition and drug condition
Panel F: response change diff vs absence length (oss=1)
Panels G–H: reserved (hidden)
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
from scipy.stats import linregress, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
)

# OBS_MAX: panel A raw/clean x-axis and global long_df observation filter.
OBS_MAX = 15
OBS_MIN = 2  # obs=1 has no previous response (diff is NaN)
OSS_MAX = 4  # max observations-since-switch in _compute_switch_bias

MIN_PIDS_PER_OBS = 15  # minimum pids required to plot a group mean

MODEL_ORDER = ["Mean", "RL", "RL_lambda", "PearceHall", "NEF2d"]
CARRYOVER_MODELS = ["Mean", "RL", "PearceHall", "NEF2d"]


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


def _build_long_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute delta, switched, oss columns from a responses dataframe."""
    df = df.copy().sort_values(["pid", "session", "trial_in_session"])
    df["delta"] = df.groupby(["pid", "trial"])["response"].diff().abs()
    df["tis_gap"] = df.groupby(["pid", "trial"])["trial_in_session"].diff()
    df["switched"] = df["tis_gap"] > 1
    df = _add_oss_to_long_df(df)
    return df


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


def _compute_switch_bias(resp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute switch-relative response bias for human or model data.

    resp_df must have columns:
        pid, session, trial, trial_in_session, observation, response, distrib_index

    For each return event (oss=1..OSS_MAX), computes:
        bias     = response(oss) - response_pre_this
        pull_dir = "Other higher" / "Other lower" based on
                   sign of (response_pre_other - response_pre_this)

    Returns long-format DataFrame with columns:
        [pid, oss, bias, pull_dir]
    """
    df = resp_df.copy().sort_values(["pid", "session", "trial_in_session"])
    df["tis_gap"] = df.groupby(["pid", "trial"])["trial_in_session"].diff()
    df["switched"] = df["tis_gap"] > 1

    records = []
    for (pid, session), grp in df.groupby(["pid", "session"], sort=False):
        grp = grp.sort_values("trial_in_session").reset_index(drop=True)
        distribs = sorted(grp["distrib_index"].dropna().unique().tolist())
        if len(distribs) != 2:
            continue
        n = len(grp)

        for i in range(1, n):
            if grp.loc[i, "distrib_index"] == grp.loc[i - 1, "distrib_index"]:
                continue

            this_distrib = grp.loc[i, "distrib_index"]

            j = i - 1
            response_pre_other = np.nan
            while j >= 0 and grp.loc[j, "distrib_index"] != this_distrib:
                if not pd.isna(grp.loc[j, "response"]):
                    response_pre_other = float(grp.loc[j, "response"])
                    break
                j -= 1

            k = i - 1
            while k >= 0 and grp.loc[k, "distrib_index"] != this_distrib:
                k -= 1
            response_pre_this = np.nan
            while k >= 0 and grp.loc[k, "distrib_index"] == this_distrib:
                if not pd.isna(grp.loc[k, "response"]):
                    response_pre_this = float(grp.loc[k, "response"])
                    break
                k -= 1

            if pd.isna(response_pre_this) or pd.isna(response_pre_other):
                continue

            pull_dir = (
                "Other higher"
                if response_pre_other > response_pre_this
                else "Other lower"
            )

            same_count = 0
            for m in range(i, min(i + 20, n)):
                if grp.loc[m, "distrib_index"] != this_distrib:
                    break
                if not pd.isna(grp.loc[m, "response"]):
                    same_count += 1
                    if same_count <= OSS_MAX:
                        records.append(
                            {
                                "pid": int(pid),
                                "oss": int(same_count),
                                "bias": float(grp.loc[m, "response"])
                                - response_pre_this,
                                "pull_dir": pull_dir,
                            }
                        )

    return pd.DataFrame(records)


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


def _plot_panel_c(
    ax,
    long_df: pd.DataFrame,
    run_folder: str,
    palette: dict,
    model_order: list[str],
) -> None:
    run_dir = data_path("runs") / run_folder

    for mt in model_order:
        resp_path = run_dir / f"{mt}_diederen_responses.pkl"
        if not resp_path.exists():
            continue
        resp = pd.read_pickle(resp_path)
        resp = resp.sort_values(["pid", "trial", "observation"])
        resp["delta"] = resp.groupby(["pid", "trial"])["response"].diff().abs()
        d = resp[
            resp["delta"].notna() & resp["observation"].between(OBS_MIN, OBS_MAX)
        ]
        pid_means = d.groupby(["pid", "observation"])["delta"].mean().reset_index()
        pid_means = _filter_pid_means_by_n(pid_means)
        if pid_means.empty:
            continue
        col = palette.get(_display(mt), palette.get(mt, "0.5"))
        sns.lineplot(
            data=pid_means,
            x="observation",
            y="delta",
            color=col,
            linewidth=1.8,
            errorbar="se",
            label=_display(mt),
            zorder=2,
            ax=ax,
        )

    d_h = long_df[
        long_df["delta"].notna() & long_df["observation"].between(OBS_MIN, OBS_MAX)
    ]
    pid_means_h = d_h.groupby(["pid", "observation"])["delta"].mean().reset_index()
    pid_means_h = _filter_pid_means_by_n(pid_means_h)
    if not pid_means_h.empty:
        sns.lineplot(
            data=pid_means_h,
            x="observation",
            y="delta",
            color="black",
            linewidth=2.2,
            errorbar="se",
            label="Human",
            zorder=3,
            ax=ax,
        )

    ax.set_xlim(OBS_MIN - 0.5, OBS_MAX + 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Response change")
    ax.legend(frameon=False, fontsize=6)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_d(
    ax,
    carry: pd.DataFrame,
    model_sw: dict[str, pd.DataFrame] | None = None,
) -> None:
    pal = get_palette()
    all_sources = ["Human"] + list((model_sw or {}).keys())
    source_colors = {"Human": "black"}
    source_colors.update(
        {mt: pal[i] for i, mt in enumerate((model_sw or {}).keys())}
    )

    def _get_oss1(df: pd.DataFrame, source: str) -> pd.DataFrame:
        sub = df[df["oss"] == 1].copy()
        pm = sub.groupby(["pid", "pull_dir"])["bias"].mean().reset_index()
        pm["source"] = source
        return pm

    pieces = [_get_oss1(carry, "Human")]
    for mt, mdf in (model_sw or {}).items():
        pieces.append(_get_oss1(mdf, mt))
    combined = pd.concat(pieces, ignore_index=True)

    sns.pointplot(
        data=combined,
        x="pull_dir",
        y="bias",
        hue="source",
        hue_order=all_sources,
        palette=source_colors,
        linewidth=2.0,
        markersize=3,
        ax=ax,
    )

    ax.set_xlabel("Estimate about the other distribution")
    ax.set_ylabel("Response bias after return")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Other higher than current",
                        "Other lower than current"])
    ax.set_xlim(-0.1, 1.1)
    ax.legend(frameon=False)
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
    For each context switch, compute diff = delta_oss_N - delta_pre
    for N = 1, 2, 3, where delta_pre is the last obs before leaving.
    Returns long-format DataFrame with columns:
        [pid, absence_length, oss, diff]
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
            if not (1 <= absence_length <= 6):
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

            post_deltas = {}
            same_count = 0
            for m in range(i, min(i + 12, n)):
                if grp.loc[m, "distrib_index"] != grp.loc[i, "distrib_index"]:
                    break
                if not pd.isna(grp.loc[m, "delta"]):
                    same_count += 1
                    if same_count <= 3:
                        post_deltas[same_count] = float(grp.loc[m, "delta"])

            for oss_val, d_post in post_deltas.items():
                records.append(
                    {
                        "pid": int(pid),
                        "absence_length": int(absence_length),
                        "oss": int(oss_val),
                        "diff": d_post - float(delta_pre),
                    }
                )

    return pd.DataFrame(records)


def _plot_panel_f(
    ax,
    valid: pd.DataFrame,
    model_sw: dict[str, pd.DataFrame] | None = None,
) -> None:
    sw_df = _compute_switch_alpha_diff(valid)

    pal = get_palette()
    source_colors = {mt: pal[i] for i, mt in enumerate((model_sw or {}).keys())}

    sub = sw_df[sw_df["oss"] == 1].copy()
    sns.regplot(
        data=sub,
        x="absence_length",
        y="diff",
        ax=ax,
        scatter=False,
        line_kws={"color": "black", "linewidth": 2.0},
        ci=95,
        color="black",
        label="Human",
    )

    for mt, mdf in (model_sw or {}).items():
        sub = mdf[mdf["oss"] == 1].copy()
        if sub.empty:
            continue
        sns.regplot(
            data=sub,
            x="absence_length",
            y="diff",
            ax=ax,
            scatter=False,
            line_kws={"color": source_colors[mt], "linewidth": 1.6},
            ci=95,
            color=source_colors[mt],
            label=mt,
        )

    ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Observations in other distribution (absence length)")
    ax.set_ylabel("Response change increase\n(Δr post-switch − Δr pre-switch)")
    ax.set_xticks(sorted(sw_df["absence_length"].unique()))
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    sns.despine(ax=ax, top=True, right=True)

    for source_label, source_df in [("Human", sw_df)] + list(
        (model_sw or {}).items()
    ):
        sub = source_df[source_df["oss"] == 1].copy()
        if sub.empty:
            continue
        slope, _, r_val, p_ols, se_slope = linregress(
            sub["absence_length"].values, sub["diff"].values
        )
        pid_slopes = []
        for pid, g in sub.groupby("pid"):
            if g["absence_length"].nunique() < 2:
                continue
            s, *_ = linregress(g["absence_length"].values, g["diff"].values)
            pid_slopes.append(s)
        pid_slopes = np.array(pid_slopes)
        n_pos = int((pid_slopes > 0).sum())
        try:
            _, p_wil = wilcoxon(pid_slopes)
        except Exception:
            p_wil = float("nan")
        pid_means = sub.groupby("pid")["diff"].mean().values
        try:
            _, p_elev = wilcoxon(pid_means)
        except Exception:
            p_elev = float("nan")
        print(
            f"Panel F {source_label} oss=1: n={len(sub)}, "
            f"slope={slope:.4f} (p={p_ols:.4f}), "
            f"per-pid +ve={n_pos}/{len(pid_slopes)} (p={p_wil:.4f}), "
            f"elevation p={p_elev:.4f}"
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

    long_df = _build_long_df(valid)
    long_df = long_df[
        long_df["delta"].notna()
        & (long_df["observation"] >= OBS_MIN)
        & (long_df["observation"] <= OBS_MAX)
    ].copy()

    carry = _compute_switch_bias(valid)

    meta = valid[
        ["pid", "trial", "observation", "session", "trial_in_session", "distrib_index"]
    ].drop_duplicates(subset=["pid", "trial", "observation"])

    model_sw: dict[str, pd.DataFrame] = {}
    model_alpha: dict[str, pd.DataFrame] = {}
    run_dir = data_path("runs") / args.run_folder
    for mt in CARRYOVER_MODELS:
        resp_path = run_dir / f"{mt}_diederen_responses.pkl"
        if not resp_path.exists():
            continue
        resp = pd.read_pickle(resp_path)
        resp_full = resp[["pid", "trial", "observation", "response"]].merge(
            meta, on=["pid", "trial", "observation"], how="left"
        )
        model_sw[mt] = _compute_switch_bias(resp_full)
        model_alpha[mt] = _compute_switch_alpha_diff(resp_full)

    fig, axes = plt.subplots(2, 4, figsize=FIGURE_SIZE, constrained_layout=True)

    _plot_panel_a(axes[0, 0])
    _plot_panel_b(axes[0, 1], args.run_folder, palette, model_order)
    _plot_panel_c(axes[0, 2], long_df, args.run_folder, palette, model_order)
    _plot_panel_d(axes[0, 3], carry, model_sw)
    _plot_panel_e(axes[1, 0], long_df)
    axes[1, 1].set_visible(True)
    _plot_panel_f(axes[1, 1], valid, model_alpha)
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
