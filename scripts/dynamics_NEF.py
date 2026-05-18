#!/usr/bin/env python3
"""
Plot NEF population dynamics by simulating one trial in memory.

Usage:
    python scripts/dynamics_NEF.py --dataset carrabin --pid 1 --model_type NEF_recurrent
    python scripts/dynamics_NEF.py --dataset carrabin --pid 1 --trial 2 --run_folder response
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

from models.NEF import _pretrain, build_network
from utils.run_params import load_run_params, trial_seed as _trial_seed
from utils.paths import FIGURES_DIR, data_path
from utils.plot_spikes import cm_gray_r_a, plot_spikes, preprocess_spikes
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette


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


def _simulate_trial(
    obs_values: np.ndarray,
    params: dict,
    decoders: dict,
) -> dict:
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
        # TODO: [decision needed] Single-step Connection(net.node_input, inpt, synapse=None)
        # is invalid (2→1); use observation component via transform.
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

    probe_data: dict = {
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
    return probe_data


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


def _load_params(
    dataset: str, pid: int, model_type: str, run_folder: str
) -> dict:
    params = load_run_params(pid, dataset, model_type, run_folder)
    params["base_seed"] = int(params.get("seed", params.get("base_seed", 0)))
    return params


def plot_dynamics(probe_data: dict) -> None:
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
    _ = get_palette()
    cb = sns.color_palette("colorblind")
    warn_cache: set[str] = set()

    t = probe_data["t"]
    obs = probe_data["obs"]
    error = probe_data["error"]
    value = probe_data["value"]
    params = probe_data["params"]
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    n_obs = int(probe_data.get("n_obs_trial", round((t[-1] + float(params["dt"])) / t_step)))

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FIGURE_SIZE[0] * 2.0, FIGURE_SIZE[1] * 0.7),
        sharex=True,
        constrained_layout=True,
    )

    _panel_node_input(
        axes[0],
        t,
        probe_data,
        n_obs,
        t_iti,
        t_step,
        panel_index=0,
        warn_cache=warn_cache,
        color_decoded=cb[1],
    )
    _panel_error(
        axes[1],
        t,
        error,
        probe_data,
        cb[2],
        n_obs,
        t_iti,
        t_step,
        panel_index=1,
        warn_cache=warn_cache,
    )
    _panel_value(
        axes[2],
        t,
        value,
        probe_data,
        cb[0],
        n_obs,
        t_iti,
        t_step,
        panel_index=2,
        warn_cache=warn_cache,
    )
    _panel_count(
        axes[3],
        t,
        probe_data,
        cb[3],
        cb[4],
        n_obs,
        t_iti,
        t_step,
        panel_index=3,
        warn_cache=warn_cache,
    )

    for ax in axes:
        ax.margins(x=0)

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
        cb,
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
    cb,
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
                ax,
                t,
                probe_data,
                n_obs,
                t_iti,
                t_step,
                panel_index=0,
                warn_cache=warn_cache,
                color_decoded=cb[1],
                decoded_linewidth=3.0,
            ),
        ),
        (
            "error",
            lambda ax: _panel_error(
                ax,
                t,
                error,
                probe_data,
                cb[2],
                n_obs,
                t_iti,
                t_step,
                panel_index=1,
                warn_cache=warn_cache,
                decoded_linewidth=3.0,
            ),
        ),
        (
            "value",
            lambda ax: _panel_value(
                ax,
                t,
                value,
                probe_data,
                cb[0],
                n_obs,
                t_iti,
                t_step,
                panel_index=2,
                warn_cache=warn_cache,
                decoded_linewidth=3.0,
            ),
        ),
        (
            "count",
            lambda ax: _panel_count(
                ax,
                t,
                probe_data,
                cb[3],
                cb[4],
                n_obs,
                t_iti,
                t_step,
                panel_index=3,
                warn_cache=warn_cache,
                decoded_linewidth=3.0,
            ),
        ),
    ]
    for name, draw in panel_specs:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        draw(ax)
        ax.margins(x=0)
        _strip_individual_panel_figure(fig)
        base = FIGURES_DIR / f"dynamics_{name}"
        fig.savefig(f"{base}.pdf")
        fig.savefig(f"{base}.svg")
        plt.close(fig)
        print(f"Saved figures/dynamics_{name}.{{pdf,svg}}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot NEF dynamics (simulate one trial)")
    p.add_argument("--dataset", type=str, default="carrabin")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument(
        "--trial",
        type=int,
        default=0,
        help="Trial index into sorted unique trial ids for this pid (default: 0)",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default="NEF_recurrent",
        help="e.g. NEF_recurrent or NEF_synaptic",
    )
    p.add_argument(
        "--run_folder",
        type=str,
        default="refit",
        help="Under data/runs/, for params pickle and context",
    )
    args = p.parse_args()

    base_params = _load_params(
        args.dataset, args.pid, args.model_type, args.run_folder
    )
    human_pid = pd.read_pickle(data_path(f"{args.dataset}.pkl"))
    human_pid = human_pid[human_pid["pid"] == args.pid]
    if len(human_pid) == 0:
        raise ValueError(f"No rows for dataset={args.dataset!r} pid={args.pid}")
    trial_ids = sorted(human_pid["trial"].unique())
    if args.trial < 0 or args.trial >= len(trial_ids):
        raise ValueError(
            f"trial index {args.trial} out of range "
            f"(n_trials={len(trial_ids)} for pid={args.pid})"
        )
    trial_db_id = int(trial_ids[args.trial])
    trial_data = human_pid[human_pid["trial"] == trial_db_id].sort_values(
        "observation"
    )
    obs_values = trial_data["value"].to_numpy(dtype=float)
    trial_seed = _trial_seed(int(base_params["seed"]), trial_db_id)
    sim_params = {
        **base_params,
        "seed": trial_seed,
    }

    decoders = _pretrain(sim_params)
    probe_data = _simulate_trial(obs_values, sim_params, decoders)
    plot_dynamics(probe_data)


if __name__ == "__main__":
    main()
