"""
scripts/archive_neural_giant.py

Archived: make_neural_giant() from scripts/make_paper_figures.py, plus
every helper function/constant used EXCLUSIVELY by it (confirmed via
direct grep across the whole repo before archiving -- nothing else
calls any of these).

make_neural_giant was the original "Acts 1-3" neural-predictions figure
(3x4: row 1 = toy/illustrative population dynamics at arbitrary
parameter values -- spike raster, alpha_0 x n_neurons PE-dynamics grid,
lambda-swept activity; rows 2-3 = sigma_R/sigma_PE and DeltaR/DeltaA-decay,
each vs alpha_0/lambda_/n_neurons via random-virtual-pid covariation,
drawn from neural_experiments.py's own `synthetic` experiment). Retired
this session, per instruction: **make_neural_main (scripts/
make_paper_figures.py's own make_neural_main()) is now the
authoritative figure for presenting the impact of neural parameters on
behavior and activity.** neural_main isolates each parameter's own
causal contribution one row/column at a time (oddball for alpha_0,
param_scan -- real or synthetic trials -- for lambda_/n_neurons),
rather than this figure's random-covariation design across all three
parameters at once. See CLAUDE.md's own current "Neural predictions
figure" section for neural_main's full up-to-date structure, and
docs/HISTORY.md for the retirement rationale and the "Acts 4/5" soft
todos (validation via ablation/partial correlation; a synaptic vs
working-memory implementation comparison) that carried forward as
standalone future-work notes rather than being tied to this figure's
own "5 acts" framing.

Removed from the `FIGURES` dict; `neural_giant` as a CLI argument to
`python -m scripts.make_paper_figures` now fails with a clear "invalid
choice" listing every remaining valid name.

Kept here for reference/history, not deleted -- see docs/HISTORY.md if
the full "Acts 1-3" narrative or its synthetic-virtual-pid data pipeline
(N=200 virtual pids, final sampling bounds alpha_0~U(0.5,1),
lambda_~U(0.1,1), n_neurons~{500,...,1500}, and the two real bugs found
building it -- a probe-worker activity-key/seed mismatch, and a raw-
vs-canonical observation-scale mismatch that saturated NEF's ensembles)
is ever needed again. NOT standalone-runnable as archived: it references
module-level state (FIGURE_SIZE, NEURAL_EXP_DIR, _apply_slide_style,
_save_fig, _fold_observation_time, get_palette, pvalue_to_stars,
_plot_neural_dual_vs_param -- the last one is SHARED with
make_neural_main and was NOT archived; it still lives in
make_paper_figures.py) that still lives in scripts/make_paper_figures.py
and was not duplicated here.

`NEURAL_ENCODER_THRESHOLD` (below) was this file's own local copy of the
same-valued constant `scripts/neural_experiments.py` independently
defines for its own weight-tuned-neuron identification -- NOT a shared
import; removing this copy does not affect that file at all.

`_neural_weight_on_cols` (below) had ZERO callers even before this
archiving -- confirmed by grep -- `_load_neural_decay_metrics` duplicates
its logic inline instead of calling it. Archived alongside the rest
since it's exclusively neural_giant-era code, not because it was ever
load-bearing.
"""

# ── Original imports needed by this code (see also the module-level
# state noted above, still in make_paper_figures.py, not duplicated) ──
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from utils.plot_spikes import plot_spikes, sample_by_variance, cluster

def _plot_neural_raster_demo(ax) -> None:
    """Panel 1 (Act 1.1): spike raster of the error population's raw neuron
    output for one representative trial (neural_experiments.py's
    raster_demo experiment), with the decoded PE trace overlaid on a twin
    axis.

    Uses sample_by_variance + cluster DIRECTLY (not the preprocess_spikes
    convenience wrapper) -- confirmed by checking check_NEF_pipeline.py and
    its archived predecessor, the only other NEF-dynamics spike-raster code
    in this repo, that neither actually does anything more than call
    preprocess_spikes(t, arr, num=50) as-is. That wrapper's own default
    sample_size=200 exceeds our n_neurons=100, so sample_by_variance's
    'select the highest-variance (truly active) neurons' step was a no-op
    (nothing to filter out of only 100 available), and its final merge
    step then block-averages neurons into synthetic composites -- fine
    when sampling genuinely thins a large pool, but here it just blurred
    real individual spike trains together without ever having filtered
    anything. Calling sample_by_variance with num=50 (well under 100) and
    skipping merge keeps real, individual, genuinely-active neurons.

    X-axis zoomed to the first 5 observations (10s of the full 30s trial),
    per instruction. Raster's own y-axis has no text label (per
    instruction, neuron index isn't inherently meaningful to a general
    reader) and its ticks are moved to the RIGHT side, since the decoded
    PE axis (the more informative one) takes the LEFT side instead --
    physical spines are unaffected by this (sns.despine's own left/right
    already matched this after the swap: ax2's default top+right removal
    keeps its left spine where its now-left ticks sit; ax's explicit
    right=False keeps its right spine where its now-right ticks sit).
    Decoded-PE line stays the palette green, but its axis label/ticks no
    longer use that color (per instruction -- color removed from the
    label specifically, not the line).
    """
    path = NEURAL_EXP_DIR / "raster_demo_soltani_numbers.pkl"
    if not path.exists():
        ax.text(0.5, 0.5, "No raster demo data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    d = pd.read_pickle(path)
    t_active, spikes_active = sample_by_variance(d["t"], d["error_neurons"],
                                                 num=50, filter_width=0.02)
    t_sorted, spikes_sorted = cluster(t_active, spikes_active, filter_width=0.002)
    plot_spikes(t_sorted, spikes_sorted, ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, 5 * 2.0)  # 5 observations x (t_obs=1.5 + t_iti=0.5)

    pe_color = get_palette(6)[2]  # palette green -- kept on the LINE only
    ax2 = ax.twinx()
    ax2.plot(d["t"], d["pe_product"], color=pe_color, lw=1.0)
    ax2.set_ylabel("Decoded Prediction Error")
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()
    ax2.set_ylim(0.0, 0.8)
    # ax's own tick-right must be set AFTER twinx() -- twinx() resets it
    # back to the left otherwise (confirmed directly by rendering: setting
    # this before twinx() left both axes' tick numbers stacked on the
    # left, overlapping).
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_yticks([])  # no explicit neuron count needed, per instruction
    ax.set_ylim(0, 50)  # raster fills the full panel height (50 neurons)
    sns.despine(ax=ax2, top=True)
    sns.despine(ax=ax, top=True, right=False)


def _plot_neural_lambda_activity(ax) -> None:
    """Panel 2 (Act 1.2): raw error-neuron activity vs observation-within-
    trial, one line per arbitrary lambda_ value (neural_experiments.py's
    sweep experiment, sweep_param='lambda_'). Style matches the reference
    lambda_drives_discounting figure's own leftmost panel, generalized from
    a 2-group (high/low median split of real fitted lambdas) comparison to
    N explicit, arbitrary swept values -- there's no real per-pid lambda
    here at all, by design (see chat).
    """
    path = NEURAL_EXP_DIR / "sweep_soltani_numbers_lambda_.pkl"
    if not path.exists():
        ax.text(0.5, 0.5, "No lambda sweep data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    from fitting.model_params import _NEF_FIXED

    d = pd.read_pickle(path)
    df = d["df"].copy()
    t_iti, t_obs = _NEF_FIXED["t_iti"], _NEF_FIXED["t_obs"]
    obs_num, t_within = _fold_observation_time(df["t"].values, t_iti, t_obs)
    df["observation"] = obs_num
    df["t_within_obs"] = t_within
    active = df[~np.isnan(df["t_within_obs"])]

    # Mean activity within each observation's own active window, per
    # (sweep_value, seed, observation) -- then averaged across seeds.
    per_obs = (active.groupby(["sweep_value", "seed", "observation"])["mean_error_activity"]
              .mean().reset_index())
    stats = (per_obs.groupby(["sweep_value", "observation"])["mean_error_activity"]
            .agg(["mean", "sem"]).reset_index())

    pal = get_palette(6)
    for i, val in enumerate(sorted(stats["sweep_value"].unique())):
        sub = stats[stats["sweep_value"] == val].sort_values("observation")
        ax.plot(sub["observation"], sub["mean"], color=pal[i], lw=1.8,
                label=f"\u03bb={val:g}")
        ax.fill_between(sub["observation"], sub["mean"] - sub["sem"],
                        sub["mean"] + sub["sem"], color=pal[i], alpha=0.18)

    ax.set_xlabel("Observation")
    ax.set_xlim(0, 15)
    ax.set_xticks(range(0, 16, 5))
    ax.set_ylabel("Error neuron activity (Hz)")
    ax.set_ylim(62, 82)
    ax.set_yticks(range(62, 83, 2))
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper right")
    sns.despine(ax=ax, top=True, right=True)


def _plot_neural_pe_dynamics(ax, show_markers: bool = False) -> None:
    """Panel 2 (Act 1.3): decoded PE vs time-within-observation, for the
    cross product of arbitrary alpha_0 x n_neurons values (matching the
    original reference PE_dynamics figure's own two-parameter convention
    -- reverted from a single-parameter sweep after reflection; see chat).
    Reads neural_experiments.py's sweep experiment run with BOTH
    --sweep_param2/--sweep_values2 set (sweep_soltani_numbers_alpha_0_
    n_neurons.pkl).

    show_markers=False (the default, per instruction) hides the dashed
    "PE/Response measured at" vertical lines and their labels entirely --
    set True to bring them back (matching the reference figure's own
    convention).

    Uses ONLY the first observation window, per instruction -- not
    averaged across all 15 -- for a clean single-transient read, matching
    the reference figure's own one-observation-per-trial convention.

    Reads sweep_param/sweep_param2 from the saved file's own metadata
    (rather than hardcoding "alpha_0"/"n_neurons" here) so this still works
    unchanged if the two swept parameters are ever reassigned.
    """
    path = NEURAL_EXP_DIR / "sweep_soltani_numbers_alpha_0_n_neurons.pkl"
    if not path.exists():
        ax.text(0.5, 0.5, "No alpha_0 x n_neurons sweep data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    from fitting.model_params import _NEF_FIXED

    d = pd.read_pickle(path)
    df = d["df"].copy()
    p1, p2 = d["sweep_param"], d["sweep_param2"]
    t_iti, t_obs = _NEF_FIXED["t_iti"], _NEF_FIXED["t_obs"]
    obs_num, t_within = _fold_observation_time(df["t"].values, t_iti, t_obs)
    df["observation"] = obs_num
    df["t_within_obs"] = t_within
    first_obs = df[(df["observation"] == 1) & (~np.isnan(df["t_within_obs"]))].copy()

    # Downsample for a cleaner line: dt=0.001s -> every 5ms.
    first_obs["t_bin"] = (first_obs["t_within_obs"] * 200).round() / 200
    stats = (first_obs.groupby(["sweep_value", "sweep_value2", "t_bin"])["pe_product"]
            .agg(["mean", "sem"]).reset_index())

    combos = sorted({(row.sweep_value, row.sweep_value2)
                     for row in stats.itertuples()})
    pal = get_palette(max(6, len(combos)))
    label_sym = {"alpha_0": "\u03b1\u2080", "n_neurons": "n", "lambda_": "\u03bb"}
    for i, (v1, v2) in enumerate(combos):
        sub = stats[(stats["sweep_value"] == v1) & (stats["sweep_value2"] == v2)].sort_values("t_bin")
        label = f"{label_sym[p1]}={v1:g}, {label_sym[p2]}={v2:g}"
        ax.plot(sub["t_bin"], sub["mean"], color=pal[i], lw=1.8, label=label)
        ax.fill_between(sub["t_bin"], sub["mean"] - sub["sem"],
                        sub["mean"] + sub["sem"], color=pal[i], alpha=0.18)

    if show_markers:
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for x, lbl in [(NEURAL_READOUT_OFFSET, "PE\nmeasured at"), (t_obs, "Response\nmeasured at")]:
            ax.axvline(x, color="0.4", lw=1.0, ls="--", zorder=0)
            ax.text(x, 1.02, lbl, transform=trans, ha="center", va="bottom",
                    clip_on=False, fontsize=7, color="0.4")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decoded Prediction Error")
    ax.set_xlim(0, t_obs + 0.05)
    ax.set_ylim(0.0, 0.3)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, ncol=1, loc="upper right")
    sns.despine(ax=ax, top=True, right=True)


NEURAL_ENCODER_THRESHOLD = 0.5  # matches figure_yoo_neural.py's own ENCODER_THRESHOLD


def _neural_weight_on_cols(pid_enc: pd.DataFrame, neuron_cols: list[str]) -> list[str]:
    """Which of the error ensemble's neurons are tuned to the WEIGHT
    dimension (enc_dim_0 -- net.error[0] in build_network, fed from the
    counting memory via W_weight) rather than the PE dimension (enc_dim_1).
    Direct port of figure_yoo_neural.py's own _weight_on_cols -- same
    encoders file layout, same threshold.
    """
    on_idx = pid_enc[pid_enc["enc_dim_0"] > NEURAL_ENCODER_THRESHOLD]["neuron_idx"].values
    return [f"n{i}" for i in on_idx if f"n{i}" in neuron_cols]


def _load_neural_probe_variability(min_trials: int = 3) -> pd.DataFrame | None:
    """Per-virtual-pid response variability (sigma) and PE variability,
    from neural_experiments.py's `synthetic` experiment (Acts 2/3's actual
    data source -- see CLAUDE.md's own "Neural predictions figure" Status
    section for why this replaced the original fitted-pid `probe` data) --
    mean std across repeated qid presentations. Uses the SAME min_trials=3
    gate as _qid_response_std (the canonical sigma computation every other
    figure in this file uses): a (virtual_pid, qid, observation) cell with
    fewer than min_trials repeated presentations has its std discarded
    (NaN) rather than trusted, before averaging per virtual pid.

    alpha_0/lambda_/n_neurons here are the RANDOM draw for that virtual
    pid (see neural_experiments.py's own _synthetic_params), not a fitted
    value -- these are qualitative covariation predictions for future
    empirical studies, not fits to existing behavioural data, so this is
    by design, not a limitation to work around.
    """
    probe_path = NEURAL_EXP_DIR / "synthetic_soltani_numbers_probe.pkl"
    params_path = NEURAL_EXP_DIR / "synthetic_soltani_numbers_params.pkl"
    if not (probe_path.exists() and params_path.exists()):
        return None
    df = pd.read_pickle(probe_path)
    params = pd.read_pickle(params_path)
    agg = (df.groupby(["virtual_pid", "qid", "observation"])[["pe", "response"]]
          .agg(lambda x: x.std() if len(x) >= min_trials else np.nan)
          .dropna())
    if agg.empty:
        return None
    per_pid = agg.groupby("virtual_pid")[["pe", "response"]].mean().reset_index()
    per_pid = per_pid.rename(columns={"pe": "pe_std", "response": "resp_std"})
    return per_pid.merge(params[["virtual_pid", "alpha_0", "lambda_", "n_neurons"]], on="virtual_pid")


def _load_neural_decay_metrics() -> pd.DataFrame | None:
    """Per-virtual-pid activity decay (mean weight-tuned-neuron activity,
    first observation minus last) and response-change decay (mean
    |Delta response|, first 2 observations minus last 2), from
    neural_experiments.py's `synthetic` experiment.

    Weight-tuned-neuron identification happens PER (virtual_pid, trial),
    not per virtual_pid -- confirmed directly (see CLAUDE.md/docs/
    HISTORY.md) that a trial's own error-ensemble encoders depend on that
    trial's own seed, so a single pid-level encoders set (the convention
    the OLD fitted-pid loader used, inherited from utils/save_activities.py)
    would silently misidentify weight-tuned neurons for every trial but
    the one its encoders happened to come from.
    """
    probe_path = NEURAL_EXP_DIR / "synthetic_soltani_numbers_probe.pkl"
    act_path = NEURAL_EXP_DIR / "synthetic_soltani_numbers_activity.pkl"
    enc_path = NEURAL_EXP_DIR / "synthetic_soltani_numbers_encoders.pkl"
    params_path = NEURAL_EXP_DIR / "synthetic_soltani_numbers_params.pkl"
    if not all(p.exists() for p in [probe_path, act_path, enc_path, params_path]):
        return None

    probe = pd.read_pickle(probe_path)
    act = pd.read_pickle(act_path)
    enc = pd.read_pickle(enc_path)
    params = pd.read_pickle(params_path)

    # Weight-tuned neuron indices, per (virtual_pid, trial) -- NOT per
    # virtual_pid alone, since encoders genuinely differ by trial.
    weight_tuned = (enc[enc["enc_dim_0"] > NEURAL_ENCODER_THRESHOLD]
                    .groupby(["virtual_pid", "trial"])["neuron_idx"]
                    .apply(list))

    act_indexed = act.set_index(["virtual_pid", "trial"]).sort_index()
    mean_act_rows = []
    for (vp, trial), idxs in weight_tuned.items():
        if not idxs:
            continue
        cols = [f"n{i}" for i in idxs]
        try:
            sub = act_indexed.loc[(vp, trial)]
        except KeyError:
            continue
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        mean_vals = sub[cols].mean(axis=1)
        for obs, val in zip(sub["observation"], mean_vals):
            mean_act_rows.append({"virtual_pid": vp, "trial": trial,
                                  "observation": obs, "mean_act": val})
    if not mean_act_rows:
        return None
    mean_act_df = pd.DataFrame(mean_act_rows)

    rows = []
    for vp in params["virtual_pid"].unique():
        pid_act = mean_act_df[mean_act_df["virtual_pid"] == vp]
        if pid_act.empty:
            continue
        act_by_obs = pid_act.groupby("observation")["mean_act"].mean()
        obs_sorted = sorted(act_by_obs.index)
        if len(obs_sorted) < 2:
            continue
        act_decay = float(act_by_obs[obs_sorted[0]]) - float(act_by_obs[obs_sorted[-1]])

        pid_resp = probe[probe["virtual_pid"] == vp].sort_values(["trial", "observation"]).copy()
        obs_sorted_r = sorted(pid_resp["observation"].unique())
        if len(obs_sorted_r) < 4:
            continue
        pid_resp["delta"] = pid_resp.groupby("trial")["response"].diff().abs()
        first_obs = obs_sorted_r[0]
        pid_resp.loc[pid_resp["observation"] == first_obs, "delta"] = (
            pid_resp.loc[pid_resp["observation"] == first_obs, "response"].abs())
        early = pid_resp[pid_resp["observation"].isin(obs_sorted_r[:2])]["delta"].mean()
        late = pid_resp[pid_resp["observation"].isin(obs_sorted_r[-2:])]["delta"].mean()
        resp_decay = float(early) - float(late)

        rows.append({"virtual_pid": int(vp), "act_decay": act_decay, "resp_decay": resp_decay})

    if not rows:
        return None
    return pd.DataFrame(rows).merge(params[["virtual_pid", "alpha_0", "lambda_", "n_neurons"]], on="virtual_pid")


def _plot_neural_sigma_vs_pe_variability(ax) -> None:
    """Panel: response variability (sigma) vs PE variability, one point per
    virtual pid -- both measurable in a real neuroimaging study with no
    model fitting on either axis. Points small and low-alpha, regression
    line thick with its CI band -- the fit is the point of this panel, not
    any individual point.
    """
    df = _load_neural_probe_variability()
    if df is None or len(df) < 3:
        ax.text(0.5, 0.5, "No probe variability data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    color = get_palette(6)[0]
    r, p = pearsonr(df["resp_std"], df["pe_std"])
    ax.scatter(df["pe_std"], df["resp_std"], color=color, s=8, alpha=0.35, zorder=2)
    sns.regplot(data=df, x="pe_std", y="resp_std", ax=ax, color=color, ci=95,
               scatter=False, line_kws={"lw": 2.2, "zorder": 3},
               label=f"r={r:.2f}{pvalue_to_stars(p)}")
    ax.set_xlabel("\u03c3PE")
    ax.set_ylabel("\u03c3R")
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper left")
    sns.despine(ax=ax, top=True, right=True)


def _plot_neural_resp_vs_act_decay(ax) -> None:
    """Panel: NEF's own |Delta response| decay vs activity decay, one
    point per virtual pid -- both measurable, no model fitting on either
    axis. Points small and low-alpha, regression line thick with its CI
    band -- the fit is the point of this panel, not any individual point.
    """
    df = _load_neural_decay_metrics()
    if df is None or len(df) < 3:
        ax.text(0.5, 0.5, "No activity/response decay data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    color = get_palette(6)[0]
    r, p = pearsonr(df["act_decay"], df["resp_decay"])
    ax.scatter(df["act_decay"], df["resp_decay"], color=color, s=8, alpha=0.35, zorder=2)
    sns.regplot(data=df, x="act_decay", y="resp_decay", ax=ax, color=color, ci=95,
               scatter=False, line_kws={"lw": 2.2, "zorder": 3},
               label=f"r={r:.2f}{pvalue_to_stars(p)}")
    ax.set_xlabel("\u0394A (Hz)")
    ax.set_ylabel("\u0394R decay")
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper left")
    sns.despine(ax=ax, top=True, right=True)


def make_neural_giant() -> Path:
    """3x4 figure: Acts 1-3 of the neural predictions narrative (see chat
    for the full 5-act plan):
      Row 1 (Act 1, toy/illustrative, arbitrary params):
        Panel 1: spike raster + decoded-PE demo.
        Panel 2: alpha_0 x n_neurons cross product, decoded PE vs
          time-within-observation, first observation only.
        Panel 3: lambda sweep, error-neuron activity vs observation.
        Panel 4: (empty -- row 1 only has 3 panels).
      Row 2 (sigma_R and sigma_PE, both measurable, no fitting on either):
        Panel 5: sigma_R vs sigma_PE.
        Panels 6-8: sigma_R AND sigma_PE, twin axes, each vs ONE of
          alpha_0/lambda_/n_neurons -- a breakdown of how much each
          parameter individually contributes, not a substitute for the
          still-pending multivariate regression (see chat).
      Row 3 (DeltaR-decay and DeltaA-decay, same structure):
        Panel 9: DeltaR-decay vs DeltaA-decay.
        Panels 10-12: DeltaR-decay AND DeltaA-decay, twin axes, each vs
          ONE of alpha_0/lambda_/n_neurons.

    Rows 2/3 read neural_experiments.py's own `synthetic` experiment --
    N randomly-parameterized virtual pids (NOT fitted params), per
    instruction; see CLAUDE.md's own "Neural predictions figure" Status
    section and docs/HISTORY.md for the full design rationale, including
    two real bugs found and fixed along the way (a response-readout
    averaging mismatch, and a raw-vs-canonical-scale mismatch that
    saturated NEF's ensembles) and the sampling-bounds narrowing that
    followed (alpha_0 in [0.5,1], lambda_ in [0.1,1], n_neurons in
    [500,1500] -- avoiding a genuine floor effect in alpha(t)=alpha_0/
    t^lambda at low alpha_0, and the extra measurement noise at low
    n_neurons that diluted the sigma-related relationships specifically).

    Act 4/5 are not included yet.
    """
    _apply_slide_style()
    fig, axes = plt.subplots(3, 4, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 2.85 * 0.75),
                             constrained_layout=True)

    _plot_neural_raster_demo(axes[0, 0])
    _plot_neural_pe_dynamics(axes[0, 1])
    _plot_neural_lambda_activity(axes[0, 2])
    axes[0, 3].axis("off")

    sigma_df = _load_neural_probe_variability()
    _plot_neural_sigma_vs_pe_variability(axes[1, 0])
    if sigma_df is not None and len(sigma_df) >= 3:
        row2_twins = []
        row2_twins.append(_plot_neural_dual_vs_param(
            axes[1, 1], sigma_df, "alpha_0", "\u03b1\u2080",
            "resp_std", "pe_std", "\u03c3 (response)", "\u03c3 (prediction error)",
            include_x_zero=True))
        row2_twins.append(_plot_neural_dual_vs_param(
            axes[1, 2], sigma_df, "lambda_", "\u03bb",
            "resp_std", "pe_std", "\u03c3 (response)", "\u03c3 (prediction error)",
            include_x_zero=True))
        row2_twins.append(_plot_neural_dual_vs_param(
            axes[1, 3], sigma_df, "n_neurons", "neurons",
            "resp_std", "pe_std", "\u03c3 (response)", "\u03c3 (prediction error)"))

        # Shared y-axes across the whole row: axes[1,0]'s own y (resp_std)
        # and every panel's left axis (also resp_std) get one common range;
        # every panel's twin (right) axis (pe_std) gets another. Twin axes
        # aren't reachable via plt.subplots' own sharey (they're created
        # per-panel, not at subplot-creation time), hence doing this
        # manually here rather than at fig, axes = plt.subplots(...).
        resp_pad = 0.05 * (sigma_df["resp_std"].max() - sigma_df["resp_std"].min())
        pe_pad = 0.05 * (sigma_df["pe_std"].max() - sigma_df["pe_std"].min())
        resp_lim = (sigma_df["resp_std"].min() - resp_pad, sigma_df["resp_std"].max() + resp_pad)
        pe_lim = (sigma_df["pe_std"].min() - pe_pad, sigma_df["pe_std"].max() + pe_pad)
        for col in (0, 1, 2, 3):
            axes[1, col].set_ylim(resp_lim)
        for ax2 in row2_twins:
            ax2.set_ylim(pe_lim)
    else:
        for col in (1, 2, 3):
            axes[1, col].text(0.5, 0.5, "No probe variability data", ha="center",
                              va="center", transform=axes[1, col].transAxes,
                              color="0.5", style="italic")

    decay_df = _load_neural_decay_metrics()
    _plot_neural_resp_vs_act_decay(axes[2, 0])
    if decay_df is not None and len(decay_df) >= 3:
        row3_twins = []
        row3_twins.append(_plot_neural_dual_vs_param(
            axes[2, 1], decay_df, "alpha_0", "\u03b1\u2080",
            "resp_decay", "act_decay", "decay (\u0394R)", "decay (\u0394A)",
            include_x_zero=True))
        row3_twins.append(_plot_neural_dual_vs_param(
            axes[2, 2], decay_df, "lambda_", "\u03bb",
            "resp_decay", "act_decay", "decay (\u0394R)", "decay (\u0394A)",
            include_x_zero=True))
        row3_twins.append(_plot_neural_dual_vs_param(
            axes[2, 3], decay_df, "n_neurons", "neurons",
            "resp_decay", "act_decay", "decay (\u0394R)", "decay (\u0394A)"))

        resp_pad = 0.05 * (decay_df["resp_decay"].max() - decay_df["resp_decay"].min())
        act_pad = 0.05 * (decay_df["act_decay"].max() - decay_df["act_decay"].min())
        resp_lim = (decay_df["resp_decay"].min() - resp_pad, decay_df["resp_decay"].max() + resp_pad)
        act_lim = (decay_df["act_decay"].min() - act_pad, decay_df["act_decay"].max() + act_pad)
        for col in (0, 1, 2, 3):
            axes[2, col].set_ylim(resp_lim)
        for ax2 in row3_twins:
            ax2.set_ylim(act_lim)
    else:
        for col in (1, 2, 3):
            axes[2, col].text(0.5, 0.5, "No activity/response decay data", ha="center",
                              va="center", transform=axes[2, col].transAxes,
                              color="0.5", style="italic")

    out_path, _ = _save_fig(fig, "neural_giant")
    plt.close(fig)
    return out_path


