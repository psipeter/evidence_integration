"""
Diederen dataset figure: 2×4 layout, panels A–E active.

Panel A: task diagram (figures/diederen_task.pdf)
Panel B: model RMSE boxplot (requires model fits in --run_folder)
Panel C: mean |Δresponse| vs observation (filter via PANEL_C_FILTER; human + models)
Panel D: forward carryover bias A→B (Human + NEF2d)
Panel E: response change around first A→B switch (Human + NEF2d)
Columns 1–3 on row 2: hidden
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.losses import _filter_first_blocks
from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
)

MIN_DATAPOINTS_PANEL_C = 30  # min pid-observations to plot a point in panel C

# Participants excluded: RMSE vs EV >= RMSE vs 0 (no better than prior mean)
EXCLUDE_PIDS: list[int] = [
    1011,
    1023,
    1027,
    1028,
    1032,
    2001,
    2029,
    2036,
    2038,
    2047,
    2048,
    2064,
    2083,
    2092,
    2099,
]

MODEL_ORDER = ["Mean", "RL", "RL_lambda", "PearceHall", "NEF2d"]
CARRYOVER_MODELS = ["Mean", "RL", "PearceHall", "NEF2d"]

# --- Global data filters (applied to all panels) ---
DATA_FILTERS: dict[str, bool] = {
    "ctrl_pcb_only": True,  # restrict to CTRL and PCB groups
    "exclude_bad_pids": True,  # remove EXCLUDE_PIDS (poor performers)
}

# --- Panel C data filter (controls which observations are included) ---
# Exactly one of these should be True; if none, all observations are shown.
PANEL_C_FILTER: dict[str, bool] = {
    "all_obs": False,  # no filtering — show all observations
    "two_plus_two": True,  # first 2 blocks per distribution per session
    "pre_first_return": False,  # first block of A and first block of B per session
    "pre_switch": False,  # distribution A only, pre-first-switch
}


def _display(model_type: str) -> str:
    return "NEF" if model_type.startswith("NEF") else model_type


def _add_oss(df: pd.DataFrame) -> pd.DataFrame:
    """Add oss column. oss=1 is first obs after a switch; NaN if
    no switch has yet occurred in this trial."""
    out = df.copy().sort_values(["pid", "trial", "trial_in_session"])
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


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta, tis_gap, switched, and oss columns to a responses dataframe."""
    df = df.copy().sort_values(["pid", "session", "trial_in_session"])
    df["delta"] = df.groupby(["pid", "trial"])["response"].diff().abs()
    df["tis_gap"] = df.groupby(["pid", "trial"])["trial_in_session"].diff()
    df["switched"] = df["tis_gap"] > 1
    df = _add_oss(df)
    return df


def _apply_panel_c_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the PANEL_C_FILTER to a responses dataframe.
    df must have columns: pid, session, trial_in_session, distrib_index.
    Returns the filtered dataframe.
    """
    active = [k for k, v in PANEL_C_FILTER.items() if v]
    if len(active) != 1:
        if len(active) == 0:
            return df
        raise ValueError(f"Exactly one PANEL_C_FILTER must be True, got: {active}")

    mode = active[0]

    if mode == "all_obs":
        return df

    if mode == "two_plus_two":
        return _filter_first_blocks(df, n_blocks=2)

    if mode == "pre_first_return":
        out = []
        for (pid, session), grp in df.groupby(["pid", "session"], sort=False):
            g = grp.sort_values("trial_in_session").reset_index(drop=True)
            distribs = sorted(g["distrib_index"].dropna().unique().tolist())
            if len(distribs) != 2:
                continue
            bc = {d: 0 for d in distribs}
            prev = None
            keep = []
            for i in range(len(g)):
                curr = int(g.at[i, "distrib_index"])
                if prev is not None and curr != prev:
                    bc[prev] += 1
                if bc[curr] < 1:
                    keep.append(i)
                prev = curr
            if keep:
                out.append(g.iloc[keep])
        return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]

    if mode == "pre_switch":
        out = []
        for (pid, session), grp in df.groupby(["pid", "session"], sort=False):
            g = grp.sort_values("trial_in_session").reset_index(drop=True)
            distrib_A = int(g.at[0, "distrib_index"])
            keep = []
            for i in range(len(g)):
                if int(g.at[i, "distrib_index"]) != distrib_A:
                    break
                keep.append(i)
            if keep:
                out.append(g.iloc[keep])
        return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]

    return df


def _compute_forward_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward carryover at first A→B switch per (pid, session).

    Returns one row per (pid, session) with:
      bias     = first_B_response - EV_B
      condition = "Prior higher" if last_A_response > EV_B else "Prior lower"
    """
    df = df.copy().sort_values(["pid", "session", "trial_in_session"])
    records = []
    for (pid, session), grp in df.groupby(["pid", "session"], sort=False):
        g = grp.sort_values("trial_in_session").reset_index(drop=True)
        distribs = sorted(g["distrib_index"].dropna().unique().tolist())
        if len(distribs) != 2:
            continue
        dA = int(g.at[0, "distrib_index"])
        dB = [d for d in distribs if d != dA][0]
        ev_B = float(g[g["distrib_index"] == dB]["ev"].iloc[0])

        last_A_resp = np.nan
        first_B_resp = np.nan
        for i in range(len(g)):
            if int(g.at[i, "distrib_index"]) == dA and not pd.isna(g.at[i, "response"]):
                last_A_resp = float(g.at[i, "response"])
            if int(g.at[i, "distrib_index"]) == dB:
                if not pd.isna(g.at[i, "response"]):
                    first_B_resp = float(g.at[i, "response"])
                break
        if pd.isna(last_A_resp) or pd.isna(first_B_resp):
            continue

        records.append(
            {
                "pid": int(pid),
                "bias": first_B_resp - ev_B,
                "condition": "Prior higher" if last_A_resp > ev_B else "Prior lower",
            }
        )
    return pd.DataFrame(records)


def _pvalue_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


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
        perf = perf[~perf["pid"].isin(EXCLUDE_PIDS)].copy()
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
    valid: pd.DataFrame,
    run_folder: str,
    palette: dict,
    model_order: list[str],
    valid_2b: pd.DataFrame,
    human_valid_raw: pd.DataFrame,
) -> None:
    run_dir = data_path("runs") / run_folder
    meta_full = human_valid_raw[
        ["pid", "trial", "observation", "session", "trial_in_session", "distrib_index"]
    ].drop_duplicates(subset=["pid", "trial", "observation"])
    tis_keep = valid_2b[["pid", "session", "trial_in_session"]].drop_duplicates()

    for mt in model_order:
        resp_path = run_dir / f"{mt}_diederen_responses.pkl"
        if not resp_path.exists():
            continue
        resp = pd.read_pickle(resp_path)
        resp = resp[~resp["pid"].isin(EXCLUDE_PIDS)].copy()
        resp = resp.merge(meta_full, on=["pid", "trial", "observation"], how="inner")
        resp = resp.sort_values(["pid", "trial", "observation"])
        resp["delta"] = resp.groupby(["pid", "trial"])["response"].diff().abs()
        resp = resp.merge(tis_keep, on=["pid", "session", "trial_in_session"], how="inner")
        d = resp[resp["delta"].notna()].copy()
        pid_means = d.groupby(["pid", "observation"])["delta"].mean().reset_index()
        obs_counts = pid_means.groupby("observation")["pid"].count()
        valid_obs = obs_counts[obs_counts >= MIN_DATAPOINTS_PANEL_C].index
        pid_means = pid_means[pid_means["observation"].isin(valid_obs)]
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

    d_h = valid[valid["delta"].notna()]
    pid_means_h = d_h.groupby(["pid", "observation"])["delta"].mean().reset_index()
    obs_counts = pid_means_h.groupby("observation")["pid"].count()
    valid_obs = obs_counts[obs_counts >= MIN_DATAPOINTS_PANEL_C].index
    pid_means_h = pid_means_h[pid_means_h["observation"].isin(valid_obs)]
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

    # ax.set_xlim(1.5, 11.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Response change")
    ax.legend(frameon=False, fontsize=6)
    sns.despine(ax=ax, top=True, right=True)


def _plot_bias_panel(
    ax,
    human_df: pd.DataFrame,
    model_dfs: dict[str, pd.DataFrame],
    source_colors: dict[str, str],
    condition_order: list[str],
    ylabel: str,
) -> None:
    """Grouped barplot of per-pid mean bias by condition with SE and Wilcoxon stars."""
    all_sources = ["Human"] + list(model_dfs.keys())

    rows = []
    for src, df in [("Human", human_df)] + list(model_dfs.items()):
        pm = df.groupby(["pid", "condition"])["bias"].mean().reset_index()
        pm["source"] = src
        rows.append(pm)
    plot_df = pd.concat(rows, ignore_index=True)

    x_positions = {}
    n_sources = len(all_sources)
    group_width = 0.7
    bar_width = group_width / n_sources
    for ci, cond in enumerate(condition_order):
        for si, src in enumerate(all_sources):
            x = ci + (si - (n_sources - 1) / 2) * bar_width
            x_positions[(cond, src)] = x
            sub = plot_df[
                (plot_df["condition"] == cond) & (plot_df["source"] == src)
            ]["bias"]
            if sub.empty:
                continue
            mean_val = sub.mean()
            se_val = sub.sem()
            color = source_colors.get(src, "0.5")
            ax.bar(
                x,
                mean_val,
                width=bar_width * 0.9,
                color="white",
                edgecolor=color,
                linewidth=1.5,
                zorder=2,
            )
            ax.errorbar(
                x,
                mean_val,
                yerr=se_val,
                fmt="none",
                color="black",
                linewidth=1.2,
                capsize=3,
                zorder=3,
            )

    y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo
    star_row_height = y_range * 0.06
    y_top = y_hi - y_range * 0.02

    for si, src in enumerate(all_sources):
        y_star = y_top - si * star_row_height
        for ci, cond in enumerate(condition_order):
            sub_df = plot_df[
                (plot_df["condition"] == cond) & (plot_df["source"] == src)
            ]
            per_pid = sub_df.groupby("pid")["bias"].mean().values
            if len(per_pid) < 5:
                continue
            try:
                _, p = wilcoxon(per_pid)
            except Exception:
                p = 1.0
            stars = _pvalue_to_stars(p)
            if not stars:
                continue
            x = x_positions[(cond, src)]
            star_color = source_colors.get(src, "black")
            ax.text(
                x,
                y_star,
                stars,
                ha="center",
                va="top",
                fontsize=9,
                color=star_color,
            )

    ax.set_xticks(range(len(condition_order)))
    ax.set_xticklabels(condition_order, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

    handles = []
    for s in all_sources:
        if s not in source_colors:
            continue
        handles.append(
            Patch(
                facecolor="white",
                edgecolor=source_colors[s],
                linewidth=1.5,
                label=s,
            )
        )
    ax.legend(handles=handles, frameon=False, loc="lower left")
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_d(
    ax,
    valid_all: pd.DataFrame,
    run_folder: str,
    palette: dict,
) -> None:
    """Forward carryover bias (A→B): first B response relative to EV_B."""
    run_dir = data_path("runs") / run_folder
    nef_color = get_palette()[3]

    human_fwd = _compute_forward_bias(valid_all)

    model_dfs = {}
    resp_path = run_dir / "NEF2d_diederen_responses.pkl"
    if resp_path.exists():
        meta = valid_all[
            [
                "pid",
                "session",
                "trial",
                "trial_in_session",
                "observation",
                "distrib_index",
                "ev",
            ]
        ].drop_duplicates(subset=["pid", "trial", "observation"])
        nef_resp = pd.read_pickle(resp_path)
        # Do NOT filter by EXCLUDE_PIDS — exclusion criterion is behavioral,
        # not applicable to model responses
        nef_full = nef_resp.merge(
            meta, on=["pid", "trial", "observation"], how="left"
        )
        nef_fwd = _compute_forward_bias(nef_full)
        if not nef_fwd.empty:
            model_dfs["NEF2d"] = nef_fwd

    source_colors = {"Human": "black", "NEF2d": nef_color}

    _plot_bias_panel(
        ax=ax,
        human_df=human_fwd,
        model_dfs=model_dfs,
        source_colors=source_colors,
        condition_order=["Prior higher", "Prior lower"],
        ylabel="Response bias after first switch",
    )


def _compute_switch_aligned_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (pid, session), find the first A->B switch and compute
    within-distribution |Δresponse| at positions relative to the switch:
      rel_pos = -4,-3,-2,-1 : first 4 within-A deltas (|a[n]-a[n-1]|)
      rel_pos = +2,+3,+4    : obs 2,3,4 of B block   (|b[n]-b[n-1]|)
    rel_pos=+1 is excluded — |b1-a_last| crosses distributions.

    Returns DataFrame with columns: [pid, rel_pos, delta, block]
    where block is 'A' or 'B'.
    """
    df = df.copy().sort_values(["pid", "session", "trial_in_session"])
    records = []
    for (pid, session), grp in df.groupby(["pid", "session"], sort=False):
        g = grp.sort_values("trial_in_session").reset_index(drop=True)
        distribs = sorted(g["distrib_index"].dropna().unique().tolist())
        if len(distribs) != 2:
            continue
        n = len(g)
        dA = int(g.at[0, "distrib_index"])
        dB = [d for d in distribs if d != dA][0]

        switch_i = None
        for i in range(1, n):
            if (
                int(g.at[i, "distrib_index"]) == dB
                and int(g.at[i - 1, "distrib_index"]) == dA
            ):
                switch_i = i
                break
        if switch_i is None:
            continue

        a_rows = []
        k = switch_i - 1
        while k >= 0 and g.loc[k, "distrib_index"] == dA:
            a_rows.append(k)
            k -= 1
        a_rows = a_rows[::-1]

        b_rows = []
        k = switch_i
        while k < n and g.loc[k, "distrib_index"] == dB:
            b_rows.append(k)
            k += 1

        for pos_idx in range(1, min(5, len(a_rows))):
            row = a_rows[pos_idx]
            row_prev = a_rows[pos_idx - 1]
            resp = g.at[row, "response"]
            resp_prev = g.at[row_prev, "response"]
            if pd.isna(resp) or pd.isna(resp_prev):
                continue
            records.append(
                {
                    "pid": int(pid),
                    "rel_pos": int(-(5 - pos_idx)),
                    "delta": abs(float(resp) - float(resp_prev)),
                    "block": "A",
                }
            )

        for pos_idx in range(1, min(4, len(b_rows))):
            row = b_rows[pos_idx]
            row_prev = b_rows[pos_idx - 1]
            resp = g.at[row, "response"]
            resp_prev = g.at[row_prev, "response"]
            if pd.isna(resp) or pd.isna(resp_prev):
                continue
            records.append(
                {
                    "pid": int(pid),
                    "rel_pos": int(pos_idx + 1),
                    "delta": abs(float(resp) - float(resp_prev)),
                    "block": "B",
                }
            )

    return pd.DataFrame(records)


def _plot_panel_e(
    ax,
    valid_all: pd.DataFrame,
    run_folder: str,
    palette: dict,
) -> None:
    """
    Panel E: mean |Δresponse| aligned to first A→B switch.
    rel_pos -4...-1: first observations of A (within-A deltas).
    rel_pos +2...+4: first observations of B (within-B deltas).
    rel_pos +1 excluded (cross-distribution delta).
    Human=black, NEF2d=palette[3].
    """
    nef_color = get_palette()[3]
    run_dir = data_path("runs") / run_folder

    human_df = _compute_switch_aligned_delta(valid_all)

    meta = valid_all[
        [
            "pid",
            "session",
            "trial",
            "trial_in_session",
            "observation",
            "distrib_index",
        ]
    ].drop_duplicates(subset=["pid", "trial", "observation"])
    nef_df = None
    resp_path = run_dir / "NEF2d_diederen_responses.pkl"
    if resp_path.exists():
        nef_resp = pd.read_pickle(resp_path)
        nef_full = nef_resp.merge(
            meta, on=["pid", "trial", "observation"], how="left"
        )
        nef_full = nef_full.dropna(
            subset=["distrib_index", "session", "trial_in_session"]
        ).copy()
        nef_full = nef_full.sort_values(
            ["pid", "session", "trial_in_session"]
        ).reset_index(drop=True)
        nef_df = _compute_switch_aligned_delta(nef_full)

    x_order = [-4, -3, -2, -1, 2, 3, 4]
    x_labels = [str(p) if p < 0 else f"+{p}" for p in x_order]
    x_plot = [-4, -3, -2, -1, 0.5, 1.5, 2.5]

    for df, color, label, lw in [
        (human_df, "black", "Human", 2.2),
        (nef_df, nef_color, "NEF2d", 1.8),
    ]:
        if df is None or df.empty:
            continue
        xs, ys, errs = [], [], []
        for xi, pos in zip(x_plot, x_order):
            sub = df[df["rel_pos"] == pos]
            pm = sub.groupby("pid")["delta"].mean()
            if len(pm) < 5:
                continue
            xs.append(xi)
            ys.append(pm.mean())
            errs.append(pm.sem())
        ax.errorbar(
            xs,
            ys,
            yerr=errs,
            color=color,
            linewidth=lw,
            marker="o",
            markersize=4,
            capsize=3,
            label=label,
        )

    ax.axvline(x=-0.25, color="0.6", linewidth=1.0, linestyle="--")

    ax.set_xticks(x_plot)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_xlabel("Observation relative to first A→B switch")
    ax.set_ylabel("Response change")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=6)
    sns.despine(ax=ax, top=True, right=True)


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
    valid_all = human[
        ~human["missed"]
        & human["response"].notna()
        & (human["catch_trial"] == False)
    ].copy()
    valid_all = _add_derived_columns(valid_all)

    human_valid_raw = valid_all.copy()
    if DATA_FILTERS.get("ctrl_pcb_only"):
        human_valid_raw = human_valid_raw[
            human_valid_raw["group"].isin(["CTRL", "PCB"])
        ].copy()
    if DATA_FILTERS.get("exclude_bad_pids"):
        human_valid_raw = human_valid_raw[
            ~human_valid_raw["pid"].isin(EXCLUDE_PIDS)
        ].copy()

    valid_2b = _add_derived_columns(_apply_panel_c_filter(human_valid_raw.copy()))
    valid_2b = valid_2b[valid_2b["delta"].notna()].copy()

    fig, axes = plt.subplots(2, 4, figsize=FIGURE_SIZE, constrained_layout=True)

    _plot_panel_a(axes[0, 0])
    _plot_panel_b(axes[0, 1], args.run_folder, palette, model_order)
    _plot_panel_c(
        axes[0, 2],
        valid_2b,
        args.run_folder,
        palette,
        model_order,
        valid_2b,
        human_valid_raw,
    )
    _plot_panel_d(axes[0, 3], valid_all, args.run_folder, palette)
    _plot_panel_e(axes[1, 0], valid_all, args.run_folder, palette)
    for col in range(1, 4):
        axes[1, col].set_visible(False)

    label_panels(
        [
            axes[0, 0],
            axes[0, 1],
            axes[0, 2],
            axes[0, 3],
            axes[1, 0],
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
