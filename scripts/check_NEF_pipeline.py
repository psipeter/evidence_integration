#!/usr/bin/env python3
"""
Generic NEF pipeline check: simulate several real trials for any
dataset/pid at an arbitrary set of NEF params, exercising the required-
activity-file loading, fast_decode, and Nengo simulation end to end, and
compare the result against RL_lambda at the same (alpha_0, lambda_).
Produces both a numerical report (RMSE, NEF vs RL_lambda) and dynamics
plots for a small subset of those trials. The RL_lambda comparison is one
part of this, not the whole point -- the other half is exercising the same
activity-loading/fast_decode/build_network machinery a real fit uses, so a
broken activity file, a seed mismatch, or a bad decoder shows up here
before it shows up in an expensive cluster fit.

Always ad hoc -- there is no --run_folder path and no completed fit is ever
read. The point of this tool is checking NEF's general behaviour across
parameter combinations, not validating one specific prior fit.

The RMSE-vs-RL_lambda comparison itself (score_vs_rl_lambda below) is NEW --
it does not exist anywhere else in the codebase. What IS reused, rather
than reimplemented, is the underlying SIMULATION that produces each side of
the comparison: NEF's response values come from the real models.NEF.run(),
and RL_lambda's come from the real models.math_models.run() -- the same
functions a real fit would call -- so the comparison is against actual
production output, not a second, possibly-drifted reimplementation of
either model's response.

REQUIRES a precomputed counting-activity file
(data/counting_activities_n{n}_nc{nc}_{dataset}.pkl) for whatever
(n_neurons, n_neurons_counting, dataset) you ask for. This script never
falls back to _pretrain() -- that fallback used to (in models/NEF.py) reuse
ONE base-seed decoder set across every trial, which is a genuine seed
mismatch against each trial's own seed (see models/NEF.py's own comment),
not just a slow path. If the file is missing, this script raises with the
exact command to generate it -- regenerate deliberately, don't let a script
decide to do it for you.

--n_sims_ensemble N (optional) also runs a REAL Nengo equivalence check of
NEF.simulate_ensemble against NEF.run() -- see check_ensemble_invariant
below. This is the check CLAUDE.md's "NEF architecture" section flags as
missing: scripts/verify_ensemble_invariant.py covers the math-model ensemble
paths but never touches models.NEF, so this is the closest thing to a real
Nengo-level correctness check the multi-seed/NLL mechanism has. Needs an
activity file with n_trials*n_sims entries for the sampled trials --
regenerate with --precompute_activities --n_sims N first if it raises.

Usage:
    python scripts/check_NEF_pipeline.py --dataset soltani_numbers --pid 3 \
        --alpha_0 0.6 --lambda_ 0.4 --n_neurons 200 --n_neurons_counting 1000 \
        --n_trials 8 --plot_trials 3 --n_sims_ensemble 5
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import nengo
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.model_params import MODEL_PARAMS
from models import math_models
from models.NEF import PARAM_DEFAULTS, build_network
from models.NEF import run as nef_run
from models.NEF import simulate_ensemble as nef_simulate_ensemble
from models.counting_integrator import (
    activity_key_for_trial,
    fast_decode as fast_decode_counting,
    load_activities as load_counting_activities,
)
from utils.binary_transform import nef_obs_values
from utils.paths import FIGURES_DIR, data_path, dataset_stem
from utils.plot_spikes import cm_gray_r_a, plot_spikes, preprocess_spikes
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette

DATASETS = ("carrabin", "yoo", "soltani_numbers", "soltani_colors")


# ── Panel-drawing helpers (dataset-agnostic; unchanged in spirit from the
#    original single-trial version) ─────────────────────────────────────────

def _iti_shading(ax, n_obs: int, t_iti: float, t_step: float) -> None:
    for i in range(n_obs):
        t_iti_start = i * t_step
        t_iti_end = t_iti_start + t_iti
        ax.axvspan(
            t_iti_start,
            t_iti_end,
            alpha=0.08,
            color="gray",
            linewidth=0,
            zorder=0,
        )


def _despine_main_raster(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _decoded_y_extent(
    *arrays: np.ndarray | None,
    extras: tuple[float, ...] = (),
    pad_ratio: float = 0.05,
) -> tuple[float, float]:
    chunks: list[np.ndarray] = []
    for a in arrays:
        if a is None:
            continue
        a = np.asarray(a).astype(float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            chunks.append(a)
    for e in extras:
        chunks.append(np.array([float(e)]))
    if not chunks:
        return (-1.0, 1.0)
    allv = np.concatenate(chunks)
    lo, hi = float(np.min(allv)), float(np.max(allv))
    if hi <= lo:
        lo, hi = lo - 0.5 * (abs(lo) + 1.0), hi + 0.5 * (abs(hi) + 1.0)
    span = hi - lo
    pad = pad_ratio * span
    return lo - pad, hi + pad


def _finalize_main_axes(ax, panel_index: int, *, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    if panel_index == 0:
        ax.set_ylabel("Decoded value")
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)


def _strip_export_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def _strip_individual_panel_figure(fig: plt.Figure) -> None:
    """Strip decorations on all axes (main and any twins)."""
    for a in fig.axes:
        _strip_export_axes(a)


def _counting_decoded_probe_targets(net):
    """Resolve weight / count ensembles for optional decoded probes (integrator vs LMU)."""
    w = (
        getattr(net.counting, "weight", None)
        or getattr(net.counting, "weight_out", None)
        or getattr(net.counting, "lmu_neural_weight_out", None)
    )
    c = (
        getattr(net.counting, "count", None)
        or getattr(net.counting, "count_out", None)
        or getattr(net.counting, "lmu_neural_count_out", None)
    )
    return w, c


def _maybe_add_iti_noise(net, params: dict, n_obs: int) -> None:
    if float(params.get("iti_noise_amplitude", 0.0)) > 0:
        try:
            from scripts.iti_perturbation import _add_iti_noise

            _add_iti_noise(net, params, n_obs)
        except ImportError:
            pass


def _simulate_trial_rich(
    obs_values: np.ndarray,
    params: dict,
    decoders: dict,
) -> dict:
    """Simulate one trial with full spike-level probing, for dynamics plots.

    This is intentionally separate from models.NEF's own _simulate_trial
    (used by the RMSE-vs-RL_lambda scoring path via nef_run below) -- that
    one only probes what a real fit needs; this one probes spikes on every
    population for visual sanity-checking, which is far more data than any
    real fitting run should ever carry.
    """
    n_obs = len(obs_values)
    t_total = n_obs * (float(params["t_obs"]) + float(params["t_iti"]))
    dt = float(params["dt"])
    tau_probe = float(params["tau_probe"])
    spike_sample_every = 0.01

    net = build_network(obs_values, params, decoders)
    _maybe_add_iti_noise(net, params, n_obs)

    p_input = p_error = p_value = None
    p_count_weight = p_count_count = None

    with net:
        p_input = nengo.Probe(net.node_input, synapse=tau_probe)
        p_error = nengo.Probe(net.error, synapse=tau_probe)
        p_value = nengo.Probe(net.value, synapse=tau_probe)
        w_tgt, c_tgt = _counting_decoded_probe_targets(net)
        if w_tgt is not None:
            p_count_weight = nengo.Probe(w_tgt, synapse=tau_probe)
        if c_tgt is not None:
            p_count_count = nengo.Probe(c_tgt, synapse=tau_probe)
        p_spk_error = nengo.Probe(
            net.error.neurons, synapse=None, sample_every=spike_sample_every
        )
        p_spk_value = nengo.Probe(
            net.value.neurons, synapse=None, sample_every=spike_sample_every
        )
        p_spk_count = nengo.Probe(
            net.counting.memory.neurons,
            synapse=None,
            sample_every=spike_sample_every,
        )
        inpt = nengo.Ensemble(
            n_neurons=int(params["n_neurons"]),
            dimensions=1,
            seed=int(params["seed"]),
            label="input",
        )
        nengo.Connection(
            net.node_input, inpt, transform=[[1.0, 0.0]], synapse=None
        )
        p_inpt_val = nengo.Probe(inpt, synapse=tau_probe)
        p_spk_inpt = nengo.Probe(
            inpt.neurons, synapse=None, sample_every=spike_sample_every
        )

    with nengo.Simulator(
        net, dt=dt, seed=int(params["seed"]), progress_bar=False
    ) as sim:
        sim.run(t_total)
        t_axis = np.asarray(sim.trange(), dtype=float)
        inp = np.asarray(sim.data[p_input])
        if inp.ndim == 2 and inp.shape[1] >= 1:
            obs = inp[:, 0].squeeze()
        else:
            obs = inp.squeeze()
        error = np.asarray(sim.data[p_error])
        value = np.asarray(sim.data[p_value]).squeeze()
        spk_e = np.asarray(sim.data[p_spk_error])
        spk_v = np.asarray(sim.data[p_spk_value])
        spk_c = np.asarray(sim.data[p_spk_count])
        cw = (
            np.asarray(sim.data[p_count_weight]).squeeze()
            if p_count_weight is not None
            else None
        )
        cc = (
            np.asarray(sim.data[p_count_count]).squeeze()
            if p_count_count is not None
            else None
        )
        inpt_decoded = np.asarray(sim.data[p_inpt_val]).squeeze()
        spk_i = np.asarray(sim.data[p_spk_inpt])

    t_spike_nominal = np.arange(0.0, t_total, spike_sample_every, dtype=float)
    spike_rows = [
        np.asarray(spk_e),
        np.asarray(spk_v),
        np.asarray(spk_i),
        np.asarray(spk_c),
    ]
    n_align = min(len(t_spike_nominal), *(m.shape[0] for m in spike_rows))
    t_spike = t_spike_nominal[:n_align]
    spk_e = spike_rows[0][:n_align]
    spk_v = spike_rows[1][:n_align]
    spk_i = spike_rows[2][:n_align]
    spk_c = spike_rows[3][:n_align]

    return {
        "t": t_axis,
        "obs": obs,
        "inpt_decoded": inpt_decoded,
        "error": error,
        "value": value,
        "params": dict(params),
        "t_spike": t_spike,
        "spikes_error": spk_e,
        "spikes_value": spk_v,
        "spikes_inpt": spk_i,
        "spikes_count": spk_c,
        "counting_weight": cw,
        "counting_count": cc,
        "n_obs_trial": int(n_obs),
    }


def _warn_spike_missing(warn_cache: set[str] | None, key: str, msg: str) -> None:
    if warn_cache is None:
        print(msg)
        return
    if key in warn_cache:
        return
    warn_cache.add(key)
    print(msg)


def _try_preprocess_spikes(
    t,
    raw,
    label: str,
    warn_cache: set[str] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    arr = np.asarray(raw)
    if arr.ndim != 2:
        _warn_spike_missing(
            warn_cache,
            f"shape_{label}",
            f"Warning: probe_data['{label}'] has shape {arr.shape}; "
            "expected 2D (time, neurons). Skipping spike raster for this panel.",
        )
        return None, None
    if arr.shape[0] != len(t):
        _warn_spike_missing(
            warn_cache,
            f"len_{label}",
            f"Warning: probe_data['{label}'] length {arr.shape[0]} != len(t)={len(t)}. "
            "Skipping spike raster for this panel.",
        )
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_pp, sp_pp = preprocess_spikes(t, arr, num=50)
    return t_pp, sp_pp


def _panel_node_input(
    ax_main,
    t,
    probe_data: dict,
    n_obs: int,
    t_iti: float,
    t_step: float,
    *,
    panel_index: int,
    warn_cache: set[str] | None,
    color_decoded,
    decoded_linewidth: float = 0.8,
) -> None:
    _iti_shading(ax_main, n_obs, t_iti, t_step)
    t_spike = np.asarray(probe_data["t_spike"], dtype=float)
    inpt_line = np.asarray(probe_data["inpt_decoded"]).squeeze()
    key = "spikes_inpt"
    raw = probe_data.get(key)
    has_raster = raw is not None
    t_pp, sp_pp = (None, None)
    if has_raster:
        t_pp, sp_pp = _try_preprocess_spikes(t_spike, raw, key, warn_cache)
        has_raster = sp_pp is not None
    y_lo, y_hi = _decoded_y_extent(inpt_line)

    if not has_raster:
        if raw is None:
            _warn_spike_missing(
                warn_cache,
                f"missing_{key}",
                f"Warning: probe_data has no '{key}'; "
                "skipping spike raster for node_input panel (line-only fallback).",
            )
        ax_main.plot(
            t,
            inpt_line,
            color=color_decoded,
            linewidth=decoded_linewidth,
            label="Input",
            zorder=2,
        )
    else:
        plot_spikes(
            t_pp,
            sp_pp,
            ax=ax_main,
            cmap=cm_gray_r_a,
            zorder=1,
            extent=(float(t_pp[0]), float(t_pp[-1]), y_lo, y_hi),
            contrast_scale=0.8,
            interpolation="none",
        )
        ax_main.plot(
            t,
            inpt_line,
            color=color_decoded,
            linewidth=decoded_linewidth,
            label="Input",
            zorder=3,
        )

    ax_main.set_ylim(y_lo, y_hi)
    _despine_main_raster(ax_main)
    _finalize_main_axes(ax_main, panel_index, title="Input")


def _panel_error(
    ax_main,
    t,
    error,
    probe_data: dict,
    color_decoded,
    n_obs: int,
    t_iti: float,
    t_step: float,
    *,
    panel_index: int,
    warn_cache: set[str] | None,
    decoded_linewidth: float = 0.8,
) -> None:
    _iti_shading(ax_main, n_obs, t_iti, t_step)
    t_spike = np.asarray(probe_data["t_spike"], dtype=float)
    gated = error[:, 0] * error[:, 1]
    key = "spikes_error"
    raw = probe_data.get(key)
    has_raster = raw is not None
    t_pp, sp_pp = (None, None)
    if has_raster:
        t_pp, sp_pp = _try_preprocess_spikes(t_spike, raw, key, warn_cache)
        has_raster = sp_pp is not None

    y_lo, y_hi = _decoded_y_extent(gated)

    if not has_raster:
        if raw is None:
            _warn_spike_missing(
                warn_cache,
                f"missing_{key}",
                f"Warning: probe_data has no '{key}'; "
                "skipping spike raster for error panel (line-only fallback).",
            )
        ax_main.plot(
            t,
            gated,
            color=color_decoded,
            linewidth=decoded_linewidth,
            label="α(n)·(o−v)",
            zorder=2,
        )
    else:
        plot_spikes(
            t_pp,
            sp_pp,
            ax=ax_main,
            cmap=cm_gray_r_a,
            zorder=1,
            extent=(float(t_pp[0]), float(t_pp[-1]), y_lo, y_hi),
            contrast_scale=0.6,
            interpolation="none",
        )
        ax_main.plot(
            t,
            gated,
            color=color_decoded,
            linewidth=decoded_linewidth,
            label="α(n)·(o−v)",
            zorder=3,
        )

    ax_main.set_ylim(y_lo, y_hi)
    _despine_main_raster(ax_main)
    _finalize_main_axes(ax_main, panel_index, title="Error")


def _panel_value(
    ax_main,
    t,
    value,
    probe_data: dict,
    color_decoded,
    n_obs: int,
    t_iti: float,
    t_step: float,
    *,
    panel_index: int,
    warn_cache: set[str] | None,
    decoded_linewidth: float = 0.8,
) -> None:
    _iti_shading(ax_main, n_obs, t_iti, t_step)
    t_spike = np.asarray(probe_data["t_spike"], dtype=float)
    key = "spikes_value"
    raw = probe_data.get(key)
    has_raster = raw is not None
    t_pp, sp_pp = (None, None)
    if has_raster:
        t_pp, sp_pp = _try_preprocess_spikes(t_spike, raw, key, warn_cache)
        has_raster = sp_pp is not None

    y_lo, y_hi = _decoded_y_extent(value, extras=(0.0,))

    if not has_raster:
        if raw is None:
            _warn_spike_missing(
                warn_cache,
                f"missing_{key}",
                f"Warning: probe_data has no '{key}'; "
                "skipping spike raster for value panel (line-only fallback).",
            )
        ax_main.plot(
            t,
            value,
            color=color_decoded,
            linewidth=decoded_linewidth,
            label="v(t)",
            zorder=2,
        )
    else:
        plot_spikes(
            t_pp,
            sp_pp,
            ax=ax_main,
            cmap=cm_gray_r_a,
            zorder=1,
            extent=(float(t_pp[0]), float(t_pp[-1]), y_lo, y_hi),
            contrast_scale=0.6,
            interpolation="none",
        )
        ax_main.plot(
            t,
            value,
            color=color_decoded,
            linewidth=decoded_linewidth,
            label="v(t)",
            zorder=3,
        )
    ax_main.axhline(0, color="0.7", linewidth=0.5, linestyle="--", zorder=2)

    ax_main.set_ylim(y_lo, y_hi)
    _despine_main_raster(ax_main)
    _finalize_main_axes(ax_main, panel_index, title="Value")


def _panel_count(
    ax_main,
    t,
    probe_data: dict,
    color_weight_decoded,
    color_count_decoded,
    n_obs: int,
    t_iti: float,
    t_step: float,
    *,
    panel_index: int,
    warn_cache: set[str] | None,
    decoded_linewidth: float = 0.8,
) -> None:
    _iti_shading(ax_main, n_obs, t_iti, t_step)
    t_spike = np.asarray(probe_data["t_spike"], dtype=float)
    key = "spikes_count"
    raw = probe_data.get(key)
    has_raster = raw is not None
    t_pp, sp_pp = (None, None)
    if has_raster:
        t_pp, sp_pp = _try_preprocess_spikes(t_spike, raw, key, warn_cache)
        has_raster = sp_pp is not None
    if not has_raster and raw is None:
        _warn_spike_missing(
            warn_cache,
            f"missing_{key}",
            f"Warning: probe_data has no '{key}'; "
            "skipping spike raster for count panel (line-only fallback).",
        )

    cw = probe_data.get("counting_weight")
    cc = probe_data.get("counting_count")
    n_obs_f = float(max(n_obs, 1))

    cw_arr = np.asarray(cw).squeeze() if cw is not None else None
    cc_arr = np.asarray(cc).squeeze() if cc is not None else None

    w_max_dec = float(np.max(np.abs(cw_arr))) if cw_arr is not None and cw_arr.size else 0.0
    w_max_dec = max(w_max_dec, 1e-12)

    y_lo, y_hi = 0.0, 1.0

    if has_raster:
        plot_spikes(
            t_pp,
            sp_pp,
            ax=ax_main,
            cmap=cm_gray_r_a,
            zorder=1,
            extent=(float(t_pp[0]), float(t_pp[-1]), y_lo, y_hi),
            contrast_scale=0.6,
            interpolation="none",
        )

    if cw_arr is not None:
        ax_main.plot(
            t,
            cw_arr / w_max_dec,
            color=color_weight_decoded,
            linewidth=decoded_linewidth,
            label="α(n) decoded",
            zorder=3,
        )
    else:
        _warn_spike_missing(
            warn_cache,
            "no_counting_weight",
            "Warning: probe_data has no 'counting_weight'; omitting decoded α line.",
        )

    if cc_arr is not None:
        ax_main.plot(
            t,
            cc_arr / n_obs_f,
            color=color_count_decoded,
            linewidth=decoded_linewidth,
            label="n decoded",
            zorder=3,
        )
    else:
        _warn_spike_missing(
            warn_cache,
            "no_counting_count",
            "Warning: probe_data has no 'counting_count'; omitting decoded n line.",
        )

    ax_main.set_ylim(y_lo, y_hi)
    leg = ax_main.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        loc="best",
    )
    leg.set_zorder(20)
    _despine_main_raster(ax_main)
    _finalize_main_axes(ax_main, panel_index, title="Count")


# ── Param loading (always ad hoc -- no completed fit is read) ──────────────

MODEL_TYPE = "NEF"  # this script only ever tests NEF; RL_lambda is the comparison target, not an alternative subject


def _load_params(
    dataset: str,
    pid: int,
    alpha_0: float,
    lambda_: float,
    n_neurons: int,
    n_neurons_counting: int,
    datafile: str | None,
) -> dict:
    """Build a full NEF params dict from explicit CLI values. No completed
    fit is ever read -- merge order is PARAM_DEFAULTS < dataset's fixed
    params < the CLI values (the same convention utils.run_params.
    load_run_params uses, with the CLI values playing the role a fitted
    Optuna trial's params would play)."""
    fixed = MODEL_PARAMS.get(dataset, {}).get(MODEL_TYPE, {}).get("fixed", {})
    merged = {**PARAM_DEFAULTS, **fixed}
    merged.update(
        {
            "alpha_0": float(alpha_0),
            "lambda_": float(lambda_),
            "n_neurons": int(n_neurons),
            "n_neurons_counting": int(n_neurons_counting),
        }
    )
    merged["dataset"] = dataset
    merged["datafile"] = datafile
    merged["model_type"] = MODEL_TYPE
    merged["pid"] = int(pid)
    merged["nef_type"] = "recurrent"
    return merged


def _require_activity_map(n_neurons: int, n_neurons_counting: int, dataset: str) -> dict:
    """Load precomputed counting activities, or fail with the exact command
    to generate them. Mirrors models/NEF.py's own required-file check --
    this script must never fall back to _pretrain() either (see that file's
    comment for why the old fallback was a genuine seed-mismatch bug, not
    just a slow path)."""
    try:
        return load_counting_activities(
            n_neurons=n_neurons, n_neurons_counting=n_neurons_counting, dataset=dataset
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No precomputed counting-activity file for "
            f"(n_neurons={n_neurons}, n_neurons_counting={n_neurons_counting}, "
            f"dataset={dataset!r}). Required -- this script does not fall back "
            f"to _pretrain(). Generate it first:\n"
            f"  venv/bin/python models/counting_integrator.py "
            f"--precompute_activities --n_neurons {n_neurons} "
            f"--n_neurons_counting {n_neurons_counting} --dataset {dataset}"
        ) from e


# ── Scoring: NEF vs RL_lambda over the sampled trials ───────────────────────

def score_vs_rl_lambda(params: dict, trials: list[int]) -> dict:
    """Run the REAL NEF.run() and math_models.run() pipelines (not a local
    reimplementation) over the sampled trials, so the reported numbers reflect
    exactly what a fit would produce -- not a second, possibly-drifted
    reimplementation of NEF's response computation."""
    nef_df = nef_run(params, trials=trials)
    rl_params = {**params, "model_type": "RL_lambda"}
    rl_df = math_models.run(rl_params, trials=trials)

    merged = nef_df.merge(
        rl_df, on=["trial", "observation"], suffixes=("_nef", "_rl")
    )
    merged["residual"] = merged["response_nef"] - merged["response_rl"]

    overall_rmse = float(np.sqrt(np.mean(merged["residual"] ** 2)))
    per_trial_rmse = (
        merged.groupby("trial")["residual"]
        .apply(lambda r: float(np.sqrt(np.mean(r ** 2))))
        .to_dict()
    )
    per_obs_rmse = (
        merged.groupby("observation")["residual"]
        .apply(lambda r: float(np.sqrt(np.mean(r ** 2))))
        .to_dict()
    )
    return {
        "merged": merged,
        "overall_rmse": overall_rmse,
        "per_trial_rmse": per_trial_rmse,
        "per_obs_rmse": per_obs_rmse,
    }


def print_report(params: dict, trials: list[int], result: dict) -> None:
    print(
        f"\nNEF vs RL_lambda  |  dataset={params['dataset']} pid={params['pid']} "
        f"n_neurons={int(params['n_neurons'])} "
        f"n_neurons_counting={int(params['n_neurons_counting'])} "
        f"alpha_0={float(params['alpha_0']):.3f} lambda_={float(params['lambda_']):.3f}"
    )
    print(f"  trials sampled ({len(trials)}): {trials}")
    print(f"  overall RMSE (NEF vs RL_lambda): {result['overall_rmse']:.4f}")
    print("  per-trial RMSE:")
    for t, v in sorted(result["per_trial_rmse"].items()):
        print(f"    trial {t}: {v:.4f}")
    print("  per-observation RMSE:")
    for o, v in sorted(result["per_obs_rmse"].items()):
        print(f"    obs {o}: {v:.4f}")


def check_ensemble_invariant(params: dict, trials: list[int], n_sims: int) -> None:
    """REAL Nengo-level check of NEF.simulate_ensemble against NEF.run() --
    the check CLAUDE.md's "NEF architecture" section flags as missing
    (scripts/verify_ensemble_invariant.py only covers the math-model
    ensemble paths, never models.NEF). Two things, both genuine invariants
    rather than tautologies, since simulate_ensemble and run() are two
    independently-written code paths:

    1. sim=1's ensemble row must EXACTLY match run()'s point-estimate
       response, because activity_key_for_trial(dataset, trial, sim=1) --
       what simulate_ensemble uses for sim 1 -- IS activity_key_for_trial
       (dataset, trial) -- what run() uses. If these ever disagree, either
       the seed formula or one of the two simulation paths has a real bug.
    2. Different sims must give genuinely DIFFERENT responses for the same
       row. If they don't, the multi-seed mechanism isn't doing what it
       claims (e.g. a key-lookup bug silently resolving every sim to the
       same seed).

    REQUIRES an activity file with n_trials*n_sims entries for the sampled
    trials -- raises (from within NEF.simulate_ensemble) with the exact
    regenerate command if it doesn't have them.
    """
    print(f"\nEnsemble invariant check (n_sims={n_sims}, trials={trials}):")
    ens, idx = nef_simulate_ensemble(params, n_sims, trials=trials, return_index=True)
    run_df = nef_run(params, trials=trials)

    sim1_df = idx.copy()
    sim1_df["response_ensemble"] = ens[0, :]
    merged = sim1_df.merge(
        run_df[["trial", "observation", "response"]], on=["trial", "observation"]
    )
    if len(merged) != len(sim1_df):
        raise ValueError(
            f"Row mismatch aligning simulate_ensemble against run(): "
            f"{len(sim1_df)} ensemble rows, {len(merged)} matched. Both should "
            f"cover exactly the same (trial, observation) pairs."
        )
    diff = (merged["response_ensemble"] - merged["response"]).abs()
    max_diff = float(diff.max())
    verdict1 = "PASS" if max_diff < 1e-9 else "FAIL -- see models/NEF.py's simulate_ensemble/run()"
    print(f"  sim=1 vs run() max abs diff: {max_diff:.2e}  [{verdict1}]")

    per_row_std = ens.std(axis=0)
    verdict2 = (
        "PASS -- genuine per-sim spread"
        if per_row_std.min() > 1e-9
        else "FAIL -- sims are not independent; check activity_key_for_trial(sim=...) usage"
    )
    print(
        f"  per-row std across sims: mean={per_row_std.mean():.4f} "
        f"min={per_row_std.min():.4f} max={per_row_std.max():.4f}  [{verdict2}]"
    )


def plot_comparison(params: dict, result: dict, out_stem: str) -> None:
    apply_style()
    merged = result["merged"].sort_values(["trial", "observation"])
    trials = sorted(merged["trial"].unique())
    pal = get_palette(max(len(trials), 3))

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(FIGURE_SIZE[0] * 1.8, FIGURE_SIZE[1]), constrained_layout=True
    )

    for i, trial in enumerate(trials):
        g = merged[merged["trial"] == trial]
        color = pal[i % len(pal)]
        ax0.plot(g["observation"], g["response_nef"], color=color, linewidth=1.2)
        ax0.plot(
            g["observation"], g["response_rl"], color=color, linewidth=1.0,
            linestyle="--",
        )
    ax0.plot([], [], color="0.3", linewidth=1.2, label="NEF")
    ax0.plot([], [], color="0.3", linewidth=1.0, linestyle="--", label="RL_lambda")
    ax0.set_xlabel("Observation")
    ax0.set_ylabel("Response")
    ax0.set_title(f"{len(trials)} sampled trials (colour = trial)")
    ax0.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax0)

    ax1.axline((0, 0), slope=1, color="0.7", linestyle="--", linewidth=1, zorder=0)
    sc = ax1.scatter(
        merged["response_rl"], merged["response_nef"],
        c=merged["observation"], cmap="viridis", s=10, alpha=0.7,
    )
    ax1.set_xlabel("RL_lambda response")
    ax1.set_ylabel("NEF response")
    ax1.set_title(f"RMSE = {result['overall_rmse']:.4f}")
    cbar = fig.colorbar(sc, ax=ax1)
    cbar.set_label("Observation")
    sns.despine(ax=ax1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{out_stem}.pdf")
    plt.close(fig)
    merged.to_pickle(FIGURES_DIR / f"{out_stem}.pkl")
    print(f"Saved figures/{out_stem}.pdf and .pkl")


# ── Per-trial dynamics figure (subset of sampled trials only) ──────────────

def plot_dynamics(probe_data: dict, trial: int) -> None:
    apply_style()
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    cb = sns.color_palette("colorblind")
    warn_cache: set[str] = set()

    t = probe_data["t"]
    error = probe_data["error"]
    value = probe_data["value"]
    params = probe_data["params"]
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    n_obs = int(probe_data.get("n_obs_trial", round((t[-1] + float(params["dt"])) / t_step)))

    fig, axes = plt.subplots(
        1, 4,
        figsize=(FIGURE_SIZE[0] * 2.0, FIGURE_SIZE[1] * 0.7),
        sharex=True,
        constrained_layout=True,
    )

    _panel_node_input(
        axes[0], t, probe_data, n_obs, t_iti, t_step,
        panel_index=0, warn_cache=warn_cache, color_decoded=cb[1],
    )
    _panel_error(
        axes[1], t, error, probe_data, cb[2], n_obs, t_iti, t_step,
        panel_index=1, warn_cache=warn_cache,
    )
    _panel_value(
        axes[2], t, value, probe_data, cb[0], n_obs, t_iti, t_step,
        panel_index=2, warn_cache=warn_cache,
    )
    _panel_count(
        axes[3], t, probe_data, cb[3], cb[4], n_obs, t_iti, t_step,
        panel_index=3, warn_cache=warn_cache,
    )

    for ax in axes:
        ax.margins(x=0)

    fig.suptitle(
        f"{params['dataset']} pid={params['pid']} trial={trial}  "
        f"n_neurons={int(params['n_neurons'])} "
        f"n_neurons_counting={int(params['n_neurons_counting'])}  "
        f"alpha_0={float(params['alpha_0']):.3f} lambda_={float(params['lambda_']):.3f}",
        fontsize=9,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = (
        f"NEF_dynamics_{params['model_type']}_{params['dataset']}_{params['pid']}"
        f"_n{int(params['n_neurons'])}_nc{int(params['n_neurons_counting'])}"
        f"_trial{trial}"
    )
    fig.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")
    plt.close(fig)


def save_individual_panels(probe_data: dict, trial: int) -> None:
    """Optional (--save_panels): per-panel PDFs for slide decks. PDF only."""
    cb = sns.color_palette("colorblind")
    warn_cache: set[str] = set()
    t = probe_data["t"]
    error = probe_data["error"]
    value = probe_data["value"]
    params = probe_data["params"]
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    n_obs = int(probe_data.get("n_obs_trial", round((t[-1] + float(params["dt"])) / t_step)))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"{params['dataset']}_{params['pid']}"
        f"_n{int(params['n_neurons'])}_nc{int(params['n_neurons_counting'])}"
        f"_trial{trial}"
    )
    panel_specs = [
        ("node_input", lambda ax: _panel_node_input(
            ax, t, probe_data, n_obs, t_iti, t_step, panel_index=0,
            warn_cache=warn_cache, color_decoded=cb[1], decoded_linewidth=3.0)),
        ("error", lambda ax: _panel_error(
            ax, t, error, probe_data, cb[2], n_obs, t_iti, t_step, panel_index=1,
            warn_cache=warn_cache, decoded_linewidth=3.0)),
        ("value", lambda ax: _panel_value(
            ax, t, value, probe_data, cb[0], n_obs, t_iti, t_step, panel_index=2,
            warn_cache=warn_cache, decoded_linewidth=3.0)),
        ("count", lambda ax: _panel_count(
            ax, t, probe_data, cb[3], cb[4], n_obs, t_iti, t_step, panel_index=3,
            warn_cache=warn_cache, decoded_linewidth=3.0)),
    ]
    for name, draw in panel_specs:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        draw(ax)
        ax.margins(x=0)
        _strip_individual_panel_figure(fig)
        out = FIGURES_DIR / f"dynamics_{name}_{suffix}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved figures/{out.name}")


# ── Orchestration ───────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--datafile", type=str, default=None)

    p.add_argument("--alpha_0", type=float, required=True)
    p.add_argument("--lambda_", type=float, required=True)
    p.add_argument("--n_neurons", type=int, required=True)
    p.add_argument("--n_neurons_counting", type=int, required=True)

    p.add_argument("--n_trials", type=int, default=5,
                    help="Number of trials to sample (first N sorted trial ids) "
                         "for scoring against RL_lambda")
    p.add_argument("--plot_trials", type=int, default=3,
                    help="How many of the sampled trials also get a full "
                         "4-panel dynamics figure (capped at --n_trials)")
    p.add_argument("--save_panels", action="store_true", default=False,
                    help="Also export individual per-panel PDFs for the "
                         "plotted trials (slide-deck use)")
    p.add_argument("--n_sims_ensemble", type=int, default=0,
                    help="If >0, also run check_ensemble_invariant (a real "
                         "Nengo check of NEF.simulate_ensemble against "
                         "NEF.run(), plus non-degeneracy across sims) with "
                         "this many sims, on the sampled trials. Needs an "
                         "activity file with n_trials*n_sims entries -- "
                         "generate with --precompute_activities --n_sims N "
                         "first.")
    args = p.parse_args()

    params = _load_params(
        args.dataset, args.pid,
        args.alpha_0, args.lambda_, args.n_neurons, args.n_neurons_counting,
        args.datafile,
    )

    n_neurons = int(params["n_neurons"])
    n_neurons_counting = int(params["n_neurons_counting"])
    activity_map = _require_activity_map(n_neurons, n_neurons_counting, args.dataset)

    stem = dataset_stem(args.dataset, params.get("datafile"))
    human = pd.read_pickle(data_path(f"{stem}.pkl"))
    human_pid = human[human["pid"] == args.pid]
    if human_pid.empty:
        raise ValueError(f"No rows for pid={args.pid} in data/{stem}.pkl")
    trial_ids = sorted(int(t) for t in human_pid["trial"].unique())

    n_trials = min(args.n_trials, len(trial_ids))
    if n_trials < args.n_trials:
        print(f"Note: only {len(trial_ids)} trials available for pid={args.pid}; "
              f"using all of them.")
    sampled = trial_ids[:n_trials]

    plot_n = min(args.plot_trials, len(sampled))
    plotted = sampled[:plot_n]

    # ── Score NEF vs RL_lambda across all sampled trials (real pipeline) ────
    result = score_vs_rl_lambda(params, sampled)
    print_report(params, sampled, result)
    out_stem = (
        f"nef_vs_rl_lambda_{args.dataset}_{args.pid}_n{n_neurons}_nc{n_neurons_counting}"
    )
    plot_comparison(params, result, out_stem)

    # ── Full dynamics figures for a small subset ────────────────────────────
    for trial in plotted:
        trial_data = human_pid[human_pid["trial"] == trial].sort_values("observation")
        obs_values = nef_obs_values(
            trial_data["value"].to_numpy(dtype=float), args.dataset
        )
        akey = activity_key_for_trial(args.dataset, trial)
        sim_params = {**params, "seed": akey}
        activity = activity_map.get(akey)
        if activity is None:
            raise KeyError(
                f"No precomputed counting activity for key {akey} "
                f"(dataset={args.dataset!r}, trial={trial}). The activity "
                f"file has keys 1..n_trials; check _DATASET_N_TRIALS and "
                f"_ZERO_INDEXED_DATASETS in models/counting_integrator.py, "
                f"or regenerate with --precompute_activities."
            )
        decoders = fast_decode_counting(
            activity,
            alpha_0=float(params["alpha_0"]),
            lambda_=float(params["lambda_"]),
        )
        probe_data = _simulate_trial_rich(obs_values, sim_params, decoders)
        plot_dynamics(probe_data, trial)
        if args.save_panels:
            save_individual_panels(probe_data, trial)

    # ── Ensemble invariant check (optional, real Nengo) ─────────────────────
    if args.n_sims_ensemble > 0:
        check_ensemble_invariant(params, sampled, args.n_sims_ensemble)


if __name__ == "__main__":
    main()
