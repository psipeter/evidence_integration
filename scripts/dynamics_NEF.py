#!/usr/bin/env python3
"""
Plot population dynamics from a single NEF.py probe run.

Usage:
    python scripts/dynamics_NEF.py --dataset carrabin --pid 1 --model_type NEF_recurrent
    python scripts/dynamics_NEF.py ... --spikes_run_folder response
    python scripts/dynamics_NEF.py ... --run_folder myrun  # default spikes folder = myrun
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import DATA_DIR, FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_spikes import cm_gray_r_a, plot_spikes, preprocess_spikes
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette


def _ideal_alpha_steps(params: dict, n_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_steps, alpha_steps) as a step function for ideal alpha(n)."""
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    alpha_0 = float(params.get("alpha_0", 1.0))
    lambda_ = float(params.get("lambda_", 0.0))
    ts, alphas = [], []
    for i in range(n_obs):
        t_start = t_iti + i * t_step
        t_end = t_start + t_obs
        alpha = alpha_0 / ((i + 1) ** lambda_)
        ts.extend([t_start, t_end])
        alphas.extend([alpha, alpha])
    return np.array(ts), np.array(alphas)


def _ideal_count_steps(params: dict, n_obs: int) -> tuple[np.ndarray, np.ndarray]:
    t_obs = params["t_obs"]
    t_iti = params["t_iti"]
    t_step = t_obs + t_iti
    ts, counts = [], []
    for i in range(n_obs):
        t_start = t_iti + i * t_step
        t_end = t_start + t_obs
        ts.extend([t_start, t_end])
        counts.extend([i + 1, i + 1])
    return np.array(ts), np.array(counts)


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


def _style_twin_decoded(ax_twin) -> None:
    ax_twin.spines["top"].set_visible(False)


def _panel_node_input(
    ax, t, obs, n_obs: int, t_iti: float, t_step: float, *, show_xlabel: bool
) -> None:
    _iti_shading(ax, n_obs, t_iti, t_step)
    ax.plot(t, obs, color="0.3", label="Input", linewidth=0.8, zorder=2)
    ax.set_ylabel("o(t)")
    if show_xlabel:
        ax.set_xlabel("Time (s)")
    _despine_main_raster(ax)


def _warn_spike_missing(warn_cache: set[str] | None, key: str, msg: str) -> None:
    if warn_cache is None:
        print(msg)
        return
    if key in warn_cache:
        return
    warn_cache.add(key)
    print(msg)


def _inject_spikes_from_activities_npz(
    probe_data: dict,
    spikes_run_folder: str,
    trial_idx: int,
    warn_cache: set[str] | None,
    *,
    spike_file_pid: int,
) -> None:
    params = probe_data["params"]
    dataset = params["dataset"]
    base = RUNS_DIR / spikes_run_folder
    mapping = (
        ("error", "spikes_error"),
        ("value", "spikes_value"),
        ("counting", "spikes_count"),
    )
    probe_data.pop("_spike_t_by_ens", None)
    by_ens: dict[str, np.ndarray] = {}
    for ens, pkey in mapping:
        probe_data.pop(pkey, None)
        path = base / f"activities_full_{ens}_{dataset}_{spike_file_pid}.npz"
        if not path.exists():
            _warn_spike_missing(
                warn_cache,
                f"missing_{pkey}",
                f"Warning: no {path.name} under {base}; "
                f"line-only raster fallback for {ens} ensemble.",
            )
            continue
        z = np.load(path)
        act = z["activities"]
        ti = int(trial_idx)
        if ti < 0 or ti >= act.shape[0]:
            _warn_spike_missing(
                warn_cache,
                f"missing_{pkey}",
                f"Warning: trial index {ti} out of range for {path.name} "
                f"(n_trials={act.shape[0]}); line-only for {ens} ensemble.",
            )
            continue
        raw = np.asarray(act[ti], dtype=float)
        t_ax = np.asarray(z["t"], dtype=float)
        if raw.shape[0] != len(t_ax):
            _warn_spike_missing(
                warn_cache,
                f"missing_{pkey}",
                f"Warning: activities time length {raw.shape[0]} != len(t)={len(t_ax)} "
                f"in {path.name}; line-only for {ens} ensemble.",
            )
            continue
        probe_data[pkey] = raw
        by_ens[ens] = t_ax
    if by_ens:
        probe_data["_spike_t_by_ens"] = by_ens


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
        t_pp, sp_pp = preprocess_spikes(t, arr, num=100)
    return t_pp, sp_pp


def _panel_error(
    ax_main,
    t,
    error,
    probe_data: dict,
    nef_color: str,
    n_obs: int,
    t_iti: float,
    t_step: float,
    *,
    show_xlabel: bool,
    warn_cache: set[str] | None,
) -> None:
    _iti_shading(ax_main, n_obs, t_iti, t_step)
    gated = error[:, 0] * error[:, 1]
    key = "spikes_error"
    raw = probe_data.get(key)
    has_raster = raw is not None
    t_pp, sp_pp = (None, None)
    t_raster = probe_data.get("_spike_t_by_ens", {}).get("error", t)
    if has_raster:
        t_pp, sp_pp = _try_preprocess_spikes(t_raster, raw, key, warn_cache)
        has_raster = sp_pp is not None
    if not has_raster:
        if raw is None:
            _warn_spike_missing(
                warn_cache,
                f"missing_{key}",
                f"Warning: probe_data has no '{key}'; "
                "skipping spike raster for error panel (line-only fallback).",
            )
        ax_main.plot(t, gated, color=nef_color, linewidth=0.8, label="α(n)·(o−v)", zorder=2)
        ax_main.set_ylabel("α(n)·(o−v)")
        if show_xlabel:
            ax_main.set_xlabel("Time (s)")
        _despine_main_raster(ax_main)
        return

    plot_spikes(t_pp, sp_pp, ax=ax_main, cmap=cm_gray_r_a, zorder=1)
    n_neurons = sp_pp.shape[1]
    ax_main.set_ylim(0, n_neurons)
    ax_main.set_ylabel("Neuron")
    if show_xlabel:
        ax_main.set_xlabel("Time (s)")
    _despine_main_raster(ax_main)

    ax_twin = ax_main.twinx()
    ax_twin.plot(
        t,
        gated,
        color=nef_color,
        linewidth=0.8,
        label="α(n)·(o−v)",
        zorder=3,
    )
    ax_twin.set_ylabel("α(n)·(o−v)")
    _style_twin_decoded(ax_twin)


def _panel_value(
    ax_main,
    t,
    value,
    probe_data: dict,
    nef_color: str,
    n_obs: int,
    t_iti: float,
    t_step: float,
    *,
    show_xlabel: bool,
    warn_cache: set[str] | None,
) -> None:
    _iti_shading(ax_main, n_obs, t_iti, t_step)
    key = "spikes_value"
    raw = probe_data.get(key)
    has_raster = raw is not None
    t_pp, sp_pp = (None, None)
    t_raster = probe_data.get("_spike_t_by_ens", {}).get("value", t)
    if has_raster:
        t_pp, sp_pp = _try_preprocess_spikes(t_raster, raw, key, warn_cache)
        has_raster = sp_pp is not None
    if not has_raster:
        if raw is None:
            _warn_spike_missing(
                warn_cache,
                f"missing_{key}",
                f"Warning: probe_data has no '{key}'; "
                "skipping spike raster for value panel (line-only fallback).",
            )
        ax_main.plot(t, value, color=nef_color, linewidth=0.8, label="v(t)", zorder=2)
        ax_main.axhline(0, color="0.7", linewidth=0.5, linestyle="--", zorder=2)
        ax_main.set_ylabel("v(t)")
        if show_xlabel:
            ax_main.set_xlabel("Time (s)")
        _despine_main_raster(ax_main)
        return

    plot_spikes(t_pp, sp_pp, ax=ax_main, cmap=cm_gray_r_a, zorder=1)
    n_neurons = sp_pp.shape[1]
    ax_main.set_ylim(0, n_neurons)
    ax_main.set_ylabel("Neuron")
    if show_xlabel:
        ax_main.set_xlabel("Time (s)")
    _despine_main_raster(ax_main)

    ax_twin = ax_main.twinx()
    ax_twin.plot(t, value, color=nef_color, linewidth=0.8, label="v(t)", zorder=3)
    ax_twin.axhline(0, color="0.7", linewidth=0.5, linestyle="--", zorder=3)
    ax_twin.set_ylabel("v(t)")
    _style_twin_decoded(ax_twin)


def _panel_count(
    ax_main,
    t,
    probe_data: dict,
    params: dict,
    palette: dict,
    nef_color: str,
    n_obs: int,
    t_iti: float,
    t_step: float,
    *,
    show_xlabel: bool,
    warn_cache: set[str] | None,
) -> None:
    _iti_shading(ax_main, n_obs, t_iti, t_step)
    key = "spikes_count"
    raw = probe_data.get(key)
    has_raster = raw is not None
    t_pp, sp_pp = (None, None)
    t_raster = probe_data.get("_spike_t_by_ens", {}).get("counting", t)
    if has_raster:
        t_pp, sp_pp = _try_preprocess_spikes(t_raster, raw, key, warn_cache)
        has_raster = sp_pp is not None
    if has_raster:
        plot_spikes(t_pp, sp_pp, ax=ax_main, cmap=cm_gray_r_a, zorder=1)
        n_neurons = sp_pp.shape[1]
        ax_main.set_ylim(0, n_neurons)
        ax_main.set_ylabel("Neuron")
        _despine_main_raster(ax_main)
    elif raw is None:
        _warn_spike_missing(
            warn_cache,
            f"missing_{key}",
            f"Warning: probe_data has no '{key}'; "
            "skipping spike raster for count panel (line-only fallback).",
        )

    ax_twin = ax_main.twinx() if has_raster else ax_main

    cw = probe_data.get("counting_weight")
    cc = probe_data.get("counting_count")
    t_steps_a, alpha_steps = _ideal_alpha_steps(params, n_obs)
    t_steps_c, count_steps = _ideal_count_steps(params, n_obs)

    if cw is not None:
        ax_twin.plot(
            t,
            np.asarray(cw).squeeze(),
            color=palette["Bayes"],
            linewidth=0.8,
            label="α(n) decoded",
            zorder=3,
        )
    else:
        _warn_spike_missing(
            warn_cache,
            "no_counting_weight",
            "Warning: probe_data has no 'counting_weight'; omitting decoded α line.",
        )

    ax_twin.plot(
        t_steps_a,
        alpha_steps,
        color="0.4",
        linewidth=1.0,
        linestyle="--",
        label="α(n) ideal",
        zorder=3,
    )

    if cc is not None:
        ax_twin.plot(
            t,
            np.asarray(cc).squeeze(),
            color=nef_color,
            linewidth=0.8,
            label="n decoded",
            zorder=3,
        )
    else:
        _warn_spike_missing(
            warn_cache,
            "no_counting_count",
            "Warning: probe_data has no 'counting_count'; omitting decoded n line.",
        )

    ax_twin.plot(
        t_steps_c,
        count_steps,
        color="0.6",
        linewidth=1.0,
        linestyle="--",
        label="n ideal",
        zorder=3,
    )

    ax_twin.set_ylabel("α(n) / n")
    ax_twin.legend(frameon=False, fontsize=7, loc="best")
    if show_xlabel:
        ax_main.set_xlabel("Time (s)")
    if has_raster:
        _style_twin_decoded(ax_twin)
    else:
        _despine_main_raster(ax_main)


def plot_dynamics(
    probe_data: dict,
    *,
    spikes_run_folder: str = "response",
    trial_idx: int = 0,
    spike_file_pid: int | None = None,
) -> None:
    # TODO: [decision needed] Twin y-axis shares one scale for α(n) and n; if
    # decoded ranges differ strongly, consider dual-axis or normalized overlay.
    apply_style()
    palette = get_palette()
    nef_color = palette["NEF"]
    warn_cache: set[str] = set()
    if spike_file_pid is None:
        spike_file_pid = int(probe_data["params"]["pid"])
    _inject_spikes_from_activities_npz(
        probe_data,
        spikes_run_folder,
        trial_idx,
        warn_cache,
        spike_file_pid=spike_file_pid,
    )

    t = probe_data["t"]
    obs = probe_data["obs"]
    error = probe_data["error"]
    value = probe_data["value"]
    params = probe_data["params"]
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    n_obs = int(round((t[-1] + float(params["dt"])) / t_step))

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FIGURE_SIZE[0] * 2.0, FIGURE_SIZE[1] * 0.7),
        sharex=True,
        constrained_layout=True,
    )

    _panel_node_input(axes[0], t, obs, n_obs, t_iti, t_step, show_xlabel=False)
    _panel_error(
        axes[1],
        t,
        error,
        probe_data,
        nef_color,
        n_obs,
        t_iti,
        t_step,
        show_xlabel=False,
        warn_cache=warn_cache,
    )
    _panel_value(
        axes[2],
        t,
        value,
        probe_data,
        nef_color,
        n_obs,
        t_iti,
        t_step,
        show_xlabel=False,
        warn_cache=warn_cache,
    )
    _panel_count(
        axes[3],
        t,
        probe_data,
        params,
        palette,
        nef_color,
        n_obs,
        t_iti,
        t_step,
        show_xlabel=True,
        warn_cache=warn_cache,
    )

    for ax in axes:
        ax.margins(x=0)

    fig.suptitle(
        f"{params['model_type']} | {params['dataset']} pid={params['pid']} "
        f"seed={params['seed']}",
        fontsize=9,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"NEF_dynamics_{params['model_type']}_{params['dataset']}_{params['pid']}"
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.{{png,pdf}}")

    _save_individual_panels(
        t,
        obs,
        error,
        value,
        probe_data,
        params,
        palette,
        nef_color,
        n_obs,
        t_iti,
        t_step,
        warn_cache,
    )

    plt.close(fig)


def _save_individual_panels(
    t,
    obs,
    error,
    value,
    probe_data: dict,
    params: dict,
    palette: dict,
    nef_color: str,
    n_obs: int,
    t_iti: float,
    t_step: float,
    warn_cache: set[str],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panel_specs = [
        (
            "node_input",
            lambda ax: _panel_node_input(
                ax, t, obs, n_obs, t_iti, t_step, show_xlabel=True
            ),
        ),
        (
            "error",
            lambda ax: _panel_error(
                ax,
                t,
                error,
                probe_data,
                nef_color,
                n_obs,
                t_iti,
                t_step,
                show_xlabel=True,
                warn_cache=warn_cache,
            ),
        ),
        (
            "value",
            lambda ax: _panel_value(
                ax,
                t,
                value,
                probe_data,
                nef_color,
                n_obs,
                t_iti,
                t_step,
                show_xlabel=True,
                warn_cache=warn_cache,
            ),
        ),
        (
            "count",
            lambda ax: _panel_count(
                ax,
                t,
                probe_data,
                params,
                palette,
                nef_color,
                n_obs,
                t_iti,
                t_step,
                show_xlabel=True,
                warn_cache=warn_cache,
            ),
        ),
    ]
    for name, draw in panel_specs:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        draw(ax)
        ax.margins(x=0)
        base = FIGURES_DIR / f"dynamics_{name}"
        fig.savefig(f"{base}.pdf")
        fig.savefig(f"{base}.svg")
        plt.close(fig)
        print(f"Saved figures/dynamics_{name}.{{pdf,svg}}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot NEF probe dynamics")
    p.add_argument("--dataset", type=str, default="carrabin")
    p.add_argument("--pid", type=int, default=None)
    p.add_argument(
        "--trial",
        type=int,
        default=None,
        help="Trial index to plot (default: first trial)",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Required, e.g. NEF_recurrent or NEF_synaptic",
    )
    p.add_argument(
        "--run_folder",
        type=str,
        default=None,
        help="Optional; default parent for spike activity npz when --spikes_run_folder omitted",
    )
    p.add_argument(
        "--spikes_run_folder",
        type=str,
        default=None,
        help="Run folder under data/runs/ for activities_full_*.npz (default: "
        "--run_folder if set, else response)",
    )
    args = p.parse_args()
    if not args.model_type:
        p.error("--model_type is required (e.g. NEF_recurrent, NEF_synaptic)")

    spikes_run_folder = (
        args.spikes_run_folder
        if args.spikes_run_folder is not None
        else (args.run_folder if args.run_folder is not None else "response")
    )

    probe_path = None
    if args.pid is not None:
        fname = f"probe_{args.model_type}_{args.dataset}_{args.pid}.pkl"
        candidate = data_path(fname)
        if candidate.exists():
            probe_path = candidate
    if probe_path is None:
        pattern = f"probe_{args.model_type}_{args.dataset}_*.pkl"
        candidates = sorted(DATA_DIR.glob(pattern))
        if not candidates:
            raise FileNotFoundError(
                f"No probe file found matching {pattern} in {DATA_DIR}"
            )
        probe_path = candidates[0]
    probe_data_all = pd.read_pickle(probe_path)
    if isinstance(probe_data_all, list):
        trial_idx = 0 if args.trial is None else args.trial
        probe_data = probe_data_all[trial_idx]
    else:
        probe_data = probe_data_all
        trial_idx = 0 if args.trial is None else args.trial
    spike_file_pid = (
        int(args.pid)
        if args.pid is not None
        else int(probe_data["params"]["pid"])
    )
    plot_dynamics(
        probe_data,
        spikes_run_folder=spikes_run_folder,
        trial_idx=trial_idx,
        spike_file_pid=spike_file_pid,
    )


if __name__ == "__main__":
    main()
