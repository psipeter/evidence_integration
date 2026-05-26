#!/usr/bin/env python3
"""
Plot NEF2d session dynamics: decoded values of all ensembles.

Simulates one full diederen session and plots:
  - Input: 4D input signal (value_A, value_B, ctx_A, ctx_B)
  - Count: decoded alpha and count for distributions A and B
  - Value: decoded value_A and value_B (2D value ensemble)
  - Error: decoded error_A and error_B (each 2D)

Usage:
    python scripts/dynamics_NEF2d.py --pid 1005 --session 1
    python scripts/dynamics_NEF2d.py --pid 1005 --session 1 --alpha_0 0.3 --lambda_ 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nengo
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.NEF2d import (
    PARAM_DEFAULTS,
    _build_main_network,
    _pretrain_counting_1d,
    _session_duration,
    _session_input_timeseries,
)
from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import apply_style


def _simulate_session_with_probes(
    session_df: pd.DataFrame,
    params: dict,
    decoders: dict,
) -> dict:
    """
    Simulate one NEF2d session and return decoded probe data for all ensembles.
    """
    rows, distrib_a, distrib_b, input_fn = _session_input_timeseries(
        session_df, params
    )
    net = _build_main_network(params, decoders, input_fn)

    dt = float(params["dt"])
    tau_probe = float(params["tau_probe"])
    t_total = _session_duration(len(rows), params)

    with net:
        p_input = nengo.Probe(net.input_node, synapse=tau_probe)
        p_value = nengo.Probe(net.value, synapse=tau_probe, sample_every=dt)
        p_weight_A = nengo.Probe(net.alpha_A_node, synapse=tau_probe)
        p_count_A = nengo.Probe(net.count_A, synapse=tau_probe)
        p_weight_B = nengo.Probe(net.alpha_B_node, synapse=tau_probe)
        p_count_B = nengo.Probe(net.count_B, synapse=tau_probe)
        p_switch = nengo.Probe(net.switch_out, synapse=tau_probe)
        p_error = nengo.Probe(net.error, synapse=tau_probe)

    with nengo.Simulator(
        net, dt=dt, seed=int(params["seed"]), progress_bar=False
    ) as sim:
        sim.run(t_total)
        t = np.asarray(sim.trange())
        inp = np.asarray(sim.data[p_input])
        value = np.asarray(sim.data[p_value])
        weight_A = np.asarray(sim.data[p_weight_A]).squeeze()
        count_A = np.asarray(sim.data[p_count_A]).squeeze()
        weight_B = np.asarray(sim.data[p_weight_B]).squeeze()
        count_B = np.asarray(sim.data[p_count_B]).squeeze()
        switch = np.asarray(sim.data[p_switch])
        error = np.asarray(sim.data[p_error])

    return {
        "t": t,
        "input": inp,
        "value": value,
        "weight_A": weight_A,
        "count_A": count_A,
        "weight_B": weight_B,
        "count_B": count_B,
        "switch": switch,
        "error": error,
        "params": dict(params),
        "distrib_a": int(distrib_a),
        "distrib_b": int(distrib_b),
        "n_obs": len(rows),
    }


def plot_dynamics(data: dict) -> None:
    apply_style()
    cb = sns.color_palette("colorblind")
    params = data["params"]
    t = data["t"]
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    n_obs = data["n_obs"]

    def _shade(ax):
        for i in range(n_obs):
            ax.axvspan(
                i * t_step,
                i * t_step + t_iti,
                alpha=0.08,
                color="gray",
                linewidth=0,
            )

    def _despine(ax):
        sns.despine(ax=ax, top=True, right=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex=True,
        constrained_layout=True,
    )

    ax = axes[0, 0]
    _shade(ax)
    ax.plot(t, data["input"][:, 0], color=cb[0], linewidth=0.8, label="value_A")
    ax.plot(t, data["input"][:, 1], color=cb[1], linewidth=0.8, label="value_B")
    ax.plot(
        t,
        data["input"][:, 2],
        color=cb[0],
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
        label="ctx_A",
    )
    ax.plot(
        t,
        data["input"][:, 3],
        color=cb[1],
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
        label="ctx_B",
    )
    ax.set_ylabel("Input")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _despine(ax)

    ax = axes[0, 1]
    _shade(ax)
    ax.plot(
        t, data["weight_A"], color=cb[0], linewidth=0.8, label="alpha_A decoded"
    )
    ax.plot(
        t, data["weight_B"], color=cb[1], linewidth=0.8, label="alpha_B decoded"
    )
    ax.plot(
        t,
        data["count_A"],
        color=cb[0],
        linewidth=0.8,
        linestyle="--",
        alpha=0.7,
        label="count_A decoded",
    )
    ax.plot(
        t,
        data["count_B"],
        color=cb[1],
        linewidth=0.8,
        linestyle="--",
        alpha=0.7,
        label="count_B decoded",
    )
    ax.set_ylabel("Count / Alpha")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _despine(ax)

    ax = axes[1, 0]
    _shade(ax)
    ax.plot(t, data["value"][:, 0], color=cb[0], linewidth=1.2, label="value_A")
    ax.plot(t, data["value"][:, 1], color=cb[1], linewidth=1.2, label="value_B")
    ax.axhline(0, color="0.7", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.legend(frameon=False, fontsize=7)
    _despine(ax)

    ax = axes[1, 1]
    _shade(ax)
    gated_A = data["error"][:, 0] * data["error"][:, 2]
    gated_B = data["error"][:, 1] * data["error"][:, 3]
    ax.plot(t, gated_A, color=cb[0], linewidth=0.8, label="α_A·PE_A")
    ax.plot(t, gated_B, color=cb[1], linewidth=0.8, label="α_B·PE_B")
    ax.axhline(0, color="0.7", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gated error")
    ax.legend(frameon=False, fontsize=7)
    _despine(ax)

    pid = int(params.get("pid", 0))
    sess = int(params.get("_session", 0))
    fig.suptitle(
        f"NEF2d dynamics — pid={pid} session={sess} "
        f"(distrib_a={data['distrib_a']}, distrib_b={data['distrib_b']})",
        fontsize=10,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"NEF2d_dynamics_pid{pid}_sess{sess}"
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=150)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.{{png,pdf}}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot NEF2d session dynamics")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--session", type=int, default=1)
    p.add_argument("--alpha_0", type=float, default=0.3)
    p.add_argument("--lambda_", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    params = {
        **PARAM_DEFAULTS,
        "model_type": "NEF2d",
        "dataset": "diederen",
        "pid": args.pid,
        "alpha_0": args.alpha_0,
        "lambda_": args.lambda_,
        "seed": args.seed,
        "base_seed": args.seed,
        "_session": args.session,
    }

    human = pd.read_pickle(data_path("diederen.pkl"))
    human_pid = human[(human["pid"] == args.pid) & ~human["missed"]].copy()
    sess_df = human_pid[human_pid["session"] == args.session].copy()
    if sess_df.empty:
        raise ValueError(f"No data for pid={args.pid} session={args.session}")

    decoders = _pretrain_counting_1d(params)
    data = _simulate_session_with_probes(sess_df, params, decoders)
    plot_dynamics(data)


if __name__ == "__main__":
    main()
