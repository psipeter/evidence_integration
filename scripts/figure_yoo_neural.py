#!/usr/bin/env python3
"""figure_yoo_neural.py — N group figure for yoo task.

Layout: 1×4
  Panel A (N1): Error-ensemble weight-neuron activity vs observation,
                split by high vs low lambda group
  Panel B (N2): Mean weight-neuron activity vs mean |delta response| across obs
                (regplot, population means per observation)
  Panel C (N3): Fitted lambda vs mean |delta response| — model scatter/line
                with per-pid human reference lines
  Panel D (N4): Mean |Δresponse| vs estimation error in last 5 obs (Human and NEF)

Run:
    python scripts/figure_yoo_neural.py
    python scripts/figure_yoo_neural.py --run_folder yoo --nef_folder refit
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
from scipy.stats import linregress, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
    pvalue_to_stars,
)

ENCODER_THRESHOLD = 0.5
LAMBDA_N          = 10
OBS_RANGE         = (2, 30)
ERROR_STYLE       = "ci"


def _placeholder(ax, text: str) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, text, ha="center", va="center",
            transform=ax.transAxes, color="0.5", style="italic", fontsize=8)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _weight_on_cols(pid_enc: pd.DataFrame, neuron_cols: list[str]) -> list[str]:
    on_idx = pid_enc[pid_enc["enc_dim_0"] > ENCODER_THRESHOLD]["neuron_idx"].values
    return [f"n{i}" for i in on_idx if f"n{i}" in neuron_cols]


def _per_pid_metrics(nef_dir: Path) -> pd.DataFrame | None:
    """Per-pid lambda, activity_decay, NEF delta_decay, and human delta_decay.

    act_decay   = mean(act[obs=1]) - mean(act[obs=30])  [positive = decay]
    delta_decay = mean(|NEF delta|[obs 1-2]) - mean(|NEF delta|[obs 29-30])
    hum_delta_decay = same metric computed from human responses (for reference lines)
    """
    acts_path   = nef_dir / "activities_error_yoo.pkl"
    encs_path   = nef_dir / "encoders_error_yoo.pkl"
    params_path = nef_dir / "NEF_yoo_params.pkl"
    resp_path   = nef_dir / "NEF_yoo_responses.pkl"
    for p in [params_path]:
        if not p.exists(): return None

    params   = pd.read_pickle(params_path)[["pid", "lambda_"]].drop_duplicates()
    yoo      = pd.read_pickle(data_path("yoo.pkl"))
    nef_resp = pd.read_pickle(resp_path) if resp_path.exists() else None

    def compute_delta_decay(df):
        rows = {}
        for pid, g in df.groupby("pid"):
            pieces = []
            for trial, tg in g.groupby("trial"):
                tg = tg.sort_values("observation").copy()
                tg["delta"] = tg["response"].diff().abs()
                tg.loc[tg["observation"] == 1, "delta"] = (
                    tg.loc[tg["observation"] == 1, "response"].abs())
                pieces.append(tg[["observation", "delta"]])
            d = pd.concat(pieces).dropna(subset=["delta"])
            early = float(d[d["observation"].isin([1, 2])]["delta"].mean())
            late  = float(d[d["observation"].isin([29, 30])]["delta"].mean())
            rows[int(pid)] = early - late
        return pd.Series(rows)

    act_decay_map: dict[int, float] = {}
    if acts_path.exists() and encs_path.exists():
        acts = pd.read_pickle(acts_path)
        encs = pd.read_pickle(encs_path)
        neuron_cols = [c for c in acts.columns if c.startswith("n") and c[1:].isdigit()]
        for pid, pid_enc in encs.groupby("pid"):
            on_cols = _weight_on_cols(pid_enc, neuron_cols)
            if not on_cols: continue
            acts_pid = acts[acts["pid"] == pid].copy()
            acts_pid["mean_act"] = acts_pid[on_cols].mean(axis=1)
            obs_mean  = acts_pid.groupby("observation")["mean_act"].mean()
            act_early = float(obs_mean.get(1,  np.nan))
            act_late  = float(obs_mean.get(30, np.nan))
            act_decay_map[int(pid)] = act_early - act_late

    nef_delta_s = compute_delta_decay(nef_resp) if nef_resp is not None else pd.Series(dtype=float)
    hum_delta_s = compute_delta_decay(yoo)

    rows = []
    for pid in params["pid"].unique():
        rows.append({
            "pid":            int(pid),
            "delta_decay":    float(nef_delta_s.get(int(pid), np.nan)),
            "hum_delta_decay":float(hum_delta_s.get(int(pid), np.nan)),
            "act_decay":      act_decay_map.get(int(pid), np.nan),
        })

    df = pd.DataFrame(rows).merge(params, on="pid")
    return df if not df.empty else None


def _prepare_activity_delta(nef_dir: Path):
    """Per-obs population-mean activity and human delta for panel B."""
    acts_path = nef_dir / "activities_error_yoo.pkl"
    encs_path = nef_dir / "encoders_error_yoo.pkl"
    for p in [acts_path, encs_path]:
        if not p.exists(): return None

    acts   = pd.read_pickle(acts_path)
    encs   = pd.read_pickle(encs_path)
    params_path = nef_dir / "NEF_yoo_params.pkl"
    params = pd.read_pickle(params_path).set_index("pid") if params_path.exists() else None
    human  = pd.read_pickle(data_path("yoo.pkl"))

    neuron_cols = [c for c in acts.columns if c.startswith("n") and c[1:].isdigit()]
    obs_range   = np.arange(OBS_RANGE[0], OBS_RANGE[1] + 1, dtype=int)

    pid_results, activity_rows, delta_rows = [], [], []
    for pid in sorted(human["pid"].unique()):
        enc_pid = encs[encs["pid"] == pid]
        on_cols = _weight_on_cols(enc_pid, neuron_cols)
        if not on_cols: continue
        acts_pid = acts[acts["pid"] == pid].copy()
        acts_pid["mean_weight_on"] = acts_pid[on_cols].mean(axis=1)
        hum_pid = human[human["pid"] == pid].sort_values(["trial","observation"]).copy()
        hum_pid["prev_response"] = hum_pid.groupby("trial")["response"].shift(1)
        hum_pid["delta_abs"] = (hum_pid["response"] - hum_pid["prev_response"]).abs()
        merged = acts_pid.merge(hum_pid[["trial","observation","delta_abs"]],
                                on=["trial","observation"], how="inner")
        g_act = merged.groupby("observation")["mean_weight_on"].mean()
        g_del = merged.groupby("observation")["delta_abs"].mean()
        activity = np.array([float(g_act[o]) if o in g_act.index else np.nan for o in obs_range])
        delta    = np.array([float(g_del[o]) if o in g_del.index else np.nan for o in obs_range])
        mask = np.isfinite(activity) & np.isfinite(delta)
        if mask.sum() < 3: continue
        slope, _, r_val, pval, _ = linregress(activity[mask], delta[mask])
        lam = float(params.loc[pid, "lambda_"]) if params is not None and pid in params.index else np.nan
        pid_results.append({"pid": int(pid), "delta": delta, "activity": activity,
                             "slope": float(slope), "r": float(r_val), "pval": float(pval),
                             "lambda_": lam})
        activity_rows.append(activity)
        delta_rows.append(delta)
    if not pid_results: return None
    return pid_results, np.nanmean(activity_rows, axis=0), np.nanmean(delta_rows, axis=0)


def _load_panel_a_data(nef_dir: Path):
    acts_path   = nef_dir / "activities_error_yoo.pkl"
    encs_path   = nef_dir / "encoders_error_yoo.pkl"
    resp_path   = nef_dir / "NEF_yoo_responses.pkl"
    params_path = nef_dir / "NEF_yoo_params.pkl"
    for p in [acts_path, encs_path, resp_path, params_path]:
        if not p.exists(): return None, None, None
    acts   = pd.read_pickle(acts_path)
    encs   = pd.read_pickle(encs_path)
    resp   = pd.read_pickle(resp_path)
    params = pd.read_pickle(params_path)[["pid","lambda_"]].drop_duplicates()
    yoo    = pd.read_pickle(data_path("yoo.pkl"))
    merged = resp.merge(yoo[["pid","trial","observation","value"]],
                        on=["pid","trial","observation"], how="left")
    merged = merged.sort_values(["pid","trial","observation"])
    merged["prev_response"] = merged.groupby(["pid","trial"])["response"].shift(1).fillna(0.0)
    merged["pe"] = merged["value"] - merged["prev_response"]
    acts = acts.merge(merged[["pid","trial","observation","pe"]],
                      on=["pid","trial","observation"], how="left")
    neuron_cols = [c for c in acts.columns if c.startswith("n") and c[1:].isdigit()]
    for pid, pid_enc in encs.groupby("pid"):
        on_cols = _weight_on_cols(pid_enc, neuron_cols)
        mask = acts["pid"] == pid
        if on_cols:
            acts.loc[mask, "mean_activity_weight_on"] = acts.loc[mask, on_cols].mean(axis=1)
    if "mean_activity_weight_on" not in acts.columns: return None, None, None
    acts = acts.merge(params, on="pid", how="left")
    plot_df = acts[(acts["observation"] >= OBS_RANGE[0]) &
                   (acts["observation"] <= OBS_RANGE[1])].copy()
    if plot_df.empty: return None, None, None
    lam_by_pid = plot_df.groupby("pid")["lambda_"].first().sort_values()
    if len(lam_by_pid) < LAMBDA_N: return None, None, None
    low_thr  = float(lam_by_pid.iloc[LAMBDA_N - 1])
    high_thr = float(lam_by_pid.iloc[-LAMBDA_N])
    low_df  = plot_df[plot_df["pid"].isin(lam_by_pid.index[:LAMBDA_N])].copy()
    high_df = plot_df[plot_df["pid"].isin(lam_by_pid.index[-LAMBDA_N:])].copy()
    low_df["lambda_group"] = "low"; high_df["lambda_group"] = "high"
    return pd.concat([low_df, high_df], ignore_index=True), low_thr, high_thr


# ── Panel A (N1) ──────────────────────────────────────────────────────────────

def _plot_panel_a(ax, nef_dir: Path) -> None:
    plot_df, low_thr, high_thr = _load_panel_a_data(nef_dir)
    if plot_df is None:
        _placeholder(ax, "No activity data\n(run fitting.collect --type activities)"); return
    cb = get_palette(2)
    sns.lineplot(data=plot_df[plot_df["lambda_group"] == "high"],
                 x="observation", y="mean_activity_weight_on",
                 color=cb[1], errorbar=ERROR_STYLE, ax=ax, legend=False)
    sns.lineplot(data=plot_df[plot_df["lambda_group"] == "low"],
                 x="observation", y="mean_activity_weight_on",
                 color=cb[0], errorbar=ERROR_STYLE, ax=ax, legend=False)
    handles = [
        Line2D([0],[0], color=cb[1], lw=2,
               label=f"High discounting (λ > {high_thr:.2f}, n={LAMBDA_N})"),
        Line2D([0],[0], color=cb[0], lw=2,
               label=f"Low discounting (λ < {low_thr:.2f}, n={LAMBDA_N})"),
    ]
    ax.set_xticks(range(0, 31, 5))
    ax.set_xlabel("Observation"); ax.set_ylabel("Error neuron activity (Hz)")
    ax.legend(handles=handles, frameon=True, framealpha=0.9, loc="upper left", fontsize=8)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel B (N2) ──────────────────────────────────────────────────────────────

def _prepare_per_pid_changes(nef_dir: Path) -> pd.DataFrame | None:
    """Per-pid: NEF activity decay and NEF delta_decay for panel B."""
    acts_path   = nef_dir / "activities_error_yoo.pkl"
    encs_path   = nef_dir / "encoders_error_yoo.pkl"
    params_path = nef_dir / "NEF_yoo_params.pkl"
    resp_path   = nef_dir / "NEF_yoo_responses.pkl"
    for p in [acts_path, encs_path, params_path, resp_path]:
        if not p.exists(): return None

    acts     = pd.read_pickle(acts_path)
    encs     = pd.read_pickle(encs_path)
    params   = pd.read_pickle(params_path)[["pid","lambda_"]].drop_duplicates()
    nef_resp = pd.read_pickle(resp_path)
    yoo      = pd.read_pickle(data_path("yoo.pkl"))
    neuron_cols = [c for c in acts.columns if c.startswith("n") and c[1:].isdigit()]

    rows = []
    for pid, pid_enc in encs.groupby("pid"):
        on_cols = _weight_on_cols(pid_enc, neuron_cols)
        if not on_cols: continue

        # Activity decay (NEF)
        acts_pid = acts[acts["pid"]==pid].copy()
        acts_pid["mean_act"] = acts_pid[on_cols].mean(axis=1)
        act_by_obs = acts_pid.groupby("observation")["mean_act"].mean()
        act_early  = act_by_obs.get(1,  np.nan)
        act_late   = act_by_obs.get(30, np.nan)
        if not (np.isfinite(act_early) and np.isfinite(act_late)): continue
        act_decay = float(act_early) - float(act_late)  # positive = decay

        # Delta decay from NEF responses
        nef_pid = nef_resp[nef_resp["pid"]==pid]
        pieces  = []
        for trial, tg in nef_pid.groupby("trial"):
            tg = tg.sort_values("observation").copy()
            tg["delta"] = tg["response"].diff().abs()
            tg.loc[tg["observation"]==1, "delta"] = tg.loc[tg["observation"]==1, "response"].abs()
            pieces.append(tg[["observation","delta"]])
        d = pd.concat(pieces).dropna(subset=["delta"])
        nef_early = float(d[d["observation"].isin([1,2])]["delta"].mean())
        nef_late  = float(d[d["observation"].isin([29,30])]["delta"].mean())
        nef_decay = nef_early - nef_late

        # Human delta decay for reference
        hum_pid = yoo[yoo["pid"]==pid]
        pieces_h = []
        for trial, tg in hum_pid.groupby("trial"):
            tg = tg.sort_values("observation").copy()
            tg["delta"] = tg["response"].diff().abs()
            tg.loc[tg["observation"]==1,"delta"] = tg.loc[tg["observation"]==1,"response"].abs()
            pieces_h.append(tg[["observation","delta"]])
        dh = pd.concat(pieces_h).dropna(subset=["delta"])
        hum_early = float(dh[dh["observation"].isin([1,2])]["delta"].mean())
        hum_late  = float(dh[dh["observation"].isin([29,30])]["delta"].mean())
        hum_decay = hum_early - hum_late

        rows.append({"pid": int(pid), "act_decay": act_decay,
                     "nef_decay": nef_decay, "hum_decay": hum_decay})

    df = pd.DataFrame(rows).dropna(subset=["act_decay","nef_decay"]).merge(params, on="pid")
    return df if not df.empty else None


def _plot_panel_b(ax, nef_dir: Path) -> None:
    """Panel B (N6): NEF activity decay vs NEF |Δresponse| decay, per pid.

    X: activity decay = mean(act[obs=1]) - mean(act[obs=30]) [Hz, positive = decay]
    Y: NEF |Δresponse| decay = mean(|delta|[obs 1-2]) - mean(|delta|[obs 29-30])
    Each point is one pid. Grey reference lines show per-pid human delta decay.
    Partial r after removing lambda reported in legend.
    """
    from numpy.linalg import lstsq as nplstsq

    df = _prepare_per_pid_changes(nef_dir)
    if df is None or len(df) < 5:
        _placeholder(ax, "No activity data"); return

    pal   = get_palette(2)
    color = pal[0]

    r_raw, p_raw = pearsonr(df["act_decay"], df["nef_decay"])

    def resid_of(y_col):
        X = np.column_stack([df["lambda_"].values, np.ones(len(df))])
        c, _, _, _ = nplstsq(X, df[y_col].values, rcond=None)
        return df[y_col].values - X @ c

    r_part, p_part = pearsonr(resid_of("act_decay"), resid_of("nef_decay"))

    # Grey lines: human delta decay per pid
    for _, row in df.iterrows():
        ax.axhline(row["hum_decay"], color="0.78", lw=0.3, zorder=0)

    ax.scatter(df["act_decay"], df["nef_decay"],
               color=color, s=35, alpha=0.85, zorder=3,
               label="Mean across trials for one participant")
    sns.regplot(data=df, x="act_decay", y="nef_decay", scatter=False,
                color=color, ci=95, ax=ax, line_kws={"lw": 1.8},
                label=(f"r={r_raw:.2f}{pvalue_to_stars(p_raw)}"
                       f" (partial r excl. \u03bb: {r_part:.2f}{pvalue_to_stars(p_part)})"))

    handles, labels_leg = ax.get_legend_handles_labels()
    handles.append(Line2D([0],[0], color="0.78", lw=0.8))
    labels_leg.append("Human (individual)")

    ax.set_xlabel("Activity decay (obs 1 \u2212 obs 30, Hz)")
    ax.set_ylabel("|\u0394response| decay (early \u2212 late)")
    ax.legend(handles, labels_leg, fontsize=7, frameon=True, framealpha=0.9, loc="upper left")
    sns.despine(ax=ax, top=True, right=True)

# ── Panel C (N3) — Lambda mediates activity decay and mean delta ──────────────

def _plot_panel_c(ax, nef_dir: Path) -> None:
    """Panel C (N7): Fitted lambda mediates both activity decay and |Deltaresponse| decay.

    X-axis: fitted lambda per pid.
    Left y-axis (blue): delta_decay = mean(|delta|[obs 1-2]) - mean(|delta|[obs 29-30])
      — positive: updating decays; higher lambda -> steeper decay.
    Right y-axis (orange): act_decay = mean(act[obs=1]) - mean(act[obs=30])
      — positive: activity decays; higher lambda -> steeper decay.
    Thin grey horizontal lines: per-pid delta_decay as reference.
    """
    df = _per_pid_metrics(nef_dir)
    if df is None:
        _placeholder(ax, "No params data"); return

    pal     = get_palette(2)
    c_del   = pal[0]
    c_act   = pal[1]
    c_human = "0.78"

    df_act = df.dropna(subset=["act_decay"])
    r_act, p_act = pearsonr(df_act["lambda_"], df_act["act_decay"])
    r_del, p_del = pearsonr(df["lambda_"],     df["delta_decay"])

    # Left axis: NEF delta decay; grey lines = human reference
    for _, row in df.iterrows():
        ax.axhline(row["hum_delta_decay"], color=c_human, lw=0.3, zorder=0)

    ax.scatter(df["lambda_"], df["delta_decay"],
               color=c_del, s=35, alpha=0.85, zorder=3,
               label=f"|\u0394response| decay, r={r_del:.2f}{pvalue_to_stars(p_del)}")
    sns.regplot(data=df, x="lambda_", y="delta_decay", scatter=False,
                color=c_del, line_kws={"lw": 1.8}, ci=95, ax=ax, label="_nolegend_")

    ax.set_xlabel("Fitted \u03bb (discounting rate)")
    ax.set_ylabel("|\u0394response| decay (early \u2212 late)", color=c_del)
    ax.tick_params(axis="y", labelcolor=c_del)
    ax.set_ylim(bottom=0)

    # Right axis: activity decay
    ax2 = ax.twinx()

    ax2.scatter(df_act["lambda_"], df_act["act_decay"],
                color=c_act, s=35, alpha=0.85, zorder=3,
                label=f"Activity decay, r={r_act:.2f}{pvalue_to_stars(p_act)}")
    sns.regplot(data=df_act, x="lambda_", y="act_decay", scatter=False,
                color=c_act, line_kws={"lw": 1.8}, ci=95, ax=ax2, label="_nolegend_")

    ax2.set_ylabel("Activity decay (early \u2212 late, Hz)", color=c_act)
    ax2.tick_params(axis="y", labelcolor=c_act)
    ax2.set_ylim(bottom=0)
    sns.despine(ax=ax2, top=True)

    # Combined figure-level legend
    handles, labels = [], []
    for a in [ax, ax2]:
        h, l = a.get_legend_handles_labels()
        handles += [x for x, y in zip(h, l) if y != "_nolegend_"]
        labels  += [y for y in l if y != "_nolegend_"]
    handles.append(Line2D([0],[0], color=c_human, lw=0.8))
    labels.append("Human (individual)")
    ax.figure.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.95,
                     loc="upper right",
                     bbox_to_anchor=(ax.get_position().x1 - 0.01,
                                     ax.get_position().y1 - 0.01),
                     bbox_transform=ax.figure.transFigure)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel D (N4) — Late delta response vs late estimation error ───────────────

def _plot_panel_d(ax, nef_dir: Path, run_folder: str) -> None:
    """Panel D (N4): Mean |Δresponse| vs estimation error in the last 5 obs.

    X-axis: mean |Δresponse| (obs 26-30) per pid — how much a pid is still
    updating their estimate near the end of the sequence.
    Y-axis: mean estimation error (RMSE to true mean, obs 26-30) per pid —
    how accurate they are at the end.

    Pids that keep updating late (high x) have higher late error (positive r
    for NEF: r=0.87****). Humans show the same trend but noisier (r=0.31 ns).
    One point per pid; separate scatter+regline for Human (grey) and NEF (colour).
    """
    LATE_OBS = range(21, 31)
    yoo      = pd.read_pickle(data_path("yoo.pkl"))
    yoo_s    = yoo.sort_values(["pid","trial","observation"]).copy()
    yoo_s["true_mean"] = yoo_s.groupby(["pid","trial"])["value"].expanding().mean().values
    true_map = yoo_s[["pid","trial","observation","true_mean"]].drop_duplicates()

    pal        = get_palette(2)
    nef_color  = pal[0]
    hum_color  = "0.4"

    def late_per_pid(df):
        df = df.sort_values(["pid","trial","observation"]).copy()
        # delta
        delta_rows = []
        for (pid, trial), g in df.groupby(["pid","trial"]):
            g = g.sort_values("observation").copy()
            g["delta"] = g["response"].diff().abs()
            delta_rows.append(g[g["observation"].isin(LATE_OBS)][["pid","observation","delta"]])
        delta_df   = pd.concat(delta_rows).dropna()
        mean_delta = delta_df.groupby("pid")["delta"].mean()
        # rmse
        m    = df.drop(columns=["true_mean"], errors="ignore").merge(
            true_map, on=["pid","trial","observation"], how="left")
        m    = m[m["observation"].isin(LATE_OBS)]
        rmse = (m.assign(sq=(m["response"] - m["true_mean"])**2)
                 .groupby("pid")["sq"].mean().apply(np.sqrt))
        return pd.DataFrame({"delta": mean_delta, "rmse": rmse}).dropna()

    # Human
    h = late_per_pid(yoo)
    r_h, p_h = pearsonr(h["delta"], h["rmse"])
    ax.scatter(h["delta"], h["rmse"], color=hum_color, s=30, alpha=0.7, zorder=3,
               label=f"Human, r={r_h:.2f}{pvalue_to_stars(p_h)}")
    sns.regplot(data=h, x="delta", y="rmse", scatter=False,
                color=hum_color, line_kws={"lw": 1.5}, ci=95, ax=ax, label="_nolegend_")

    # NEF
    nef_resp_path = nef_dir / "NEF_yoo_responses.pkl"
    if nef_resp_path.exists():
        n = late_per_pid(pd.read_pickle(nef_resp_path))
        r_n, p_n = pearsonr(n["delta"], n["rmse"])
        ax.scatter(n["delta"], n["rmse"], color=nef_color, s=30, alpha=0.7, zorder=3,
                   label=f"NEF, r={r_n:.2f}{pvalue_to_stars(p_n)}")
        sns.regplot(data=n, x="delta", y="rmse", scatter=False,
                    color=nef_color, line_kws={"lw": 1.5}, ci=95, ax=ax, label="_nolegend_")

    ax.set_xlabel("Mean |Δresponse| (obs 21-30)")
    ax.set_ylabel("Estimation error (obs 21-30)")
    ax.set_ylim(bottom=0); ax.set_xlim(left=0)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="yoo")
    parser.add_argument("--nef_folder", type=str, default="refit")
    args = parser.parse_args()

    nef_dir = RUNS_DIR / args.nef_folder

    apply_style()

    fig, axes = plt.subplots(
        1, 4,
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] / 2),
        constrained_layout=True,
    )

    _plot_panel_a(axes[0], nef_dir)
    _plot_panel_b(axes[1], nef_dir)
    _plot_panel_c(axes[2], nef_dir)
    _plot_panel_d(axes[3], nef_dir, args.run_folder)

    label_panels(axes.reshape(1, -1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_yoo_neural"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
