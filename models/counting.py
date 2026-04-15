#!/usr/bin/env python3
"""
Neural counting circuit testbed.

Tests different NEF-based mechanisms for counting discrete observation events.
Each mechanism receives ideal pulses (one per observation, simulating the
output of a delta/differentiator circuit) and attempts to maintain an
accurate count n(t).

Usage:
    python models/counting.py --mechanism integrator [--n_obs 30] [--n_neurons 200] [--seed 0]

Mechanisms:
    integrator  -- recurrent line-attractor working memory (baseline)

Saves figures to figures/counting_{mechanism}.pdf and figures/counting_{mechanism}.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nengo
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR
from utils.plot_style import apply_style, FIGURE_SIZE, get_palette

# -- defaults ------------------------------------------------------------------
DEFAULT_N_OBS = 30  # number of observations to count
DEFAULT_N_NEURONS = 200  # neurons per population
DEFAULT_SEED = 0  # RNG seed for neuron tuning curves

# timing (seconds)
T_OBS = 1.0  # duration of each observation pulse
T_ITI = 1.0  # inter-stimulus interval between pulses
T_STEP = T_OBS + T_ITI  # total time per observation
PROBE_DT = 0.010   # probe sample interval (s) — 10ms

# integrator parameters
TAU_FB = 0.200   # feedback synapse time constant (s)
RECUR_W = 1.0     # recurrent weight for line attractor

# LMU parameters
LMU_ORDER = 32     # number of Legendre polynomials
LMU_THETA_MULT = 1.1    # theta = n_obs * T_STEP * LMU_THETA_MULT


def make_pulse_input(n_obs: int, amplitude: float = 1.0) -> callable:
    """
    Return a Nengo input function that produces a square pulse of `amplitude`
    during the observation window of each step, and 0 during the ITI.
    These pulses simulate the output of the delta/differentiator circuit.
    """

    def pulse(t: float) -> float:
        step = int(t / T_STEP)
        phase = t - step * T_STEP
        if step < n_obs and phase < T_OBS:
            return amplitude
        return 0.0

    return pulse


def true_count(t: float, n_obs: int) -> float:
    """
    Ground-truth count at time t.
    Count is 1 during and after the first observation, 2 after the second, etc.
    Returns 0 before the first pulse onset.
    """
    phase = t - int(t / T_STEP) * T_STEP
    step = int(t / T_STEP)
    # count increments as soon as the pulse begins
    if step >= n_obs:
        return float(n_obs)
    if phase < T_OBS:
        return float(step + 1)
    return float(step + 1)


# -- mechanisms ----------------------------------------------------------------


def build_integrator(n_obs: int, n_neurons: int, seed: int) -> nengo.Network:
    """
    Line-attractor integrator counting circuit.

    Architecture:
        input (node) -[TAU_FB]-> memory
        memory -[recurrent, TAU_FB, RECUR_W]-> memory

    The memory population represents n(t) as a scalar in [0, n_obs].
    Each observation pulse increments the memory by ~1 via the feedforward
    connection. Between pulses the recurrent connection maintains the count.
    """
    pulse_fn = make_pulse_input(n_obs, amplitude=1.0)

    with nengo.Network(label="integrator", seed=seed) as net:
        # input node: ideal observation pulses (delta circuit output)
        net.input = nengo.Node(pulse_fn, label="input")

        # memory population: line attractor representing n(t)
        net.memory = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=1,
            radius=n_obs,
            label="memory",
        )

        # feedforward: each pulse increments the count
        nengo.Connection(
            net.input,
            net.memory,
            synapse=None,
            transform=TAU_FB,
        )

        # recurrent: maintains current count between pulses
        nengo.Connection(
            net.memory,
            net.memory,
            synapse=TAU_FB,
            transform=RECUR_W,
        )

        # probes
        net.probe_input = nengo.Probe(net.input, synapse=None,
                                      sample_every=PROBE_DT, label="probe_input")
        net.probe_memory = nengo.Probe(net.memory, synapse=0.01,
                                       sample_every=PROBE_DT, label="probe_memory")

    return net


def _compute_lmu_matrices(n_obs: int) -> tuple:
    """
    Compute LMU A, B matrices and offline readout weights W.
    Returns (A, B, W_readout) where:
        A: (order, order) discretized state transition matrix
        B: (order, 1) discretized input matrix
        W_readout: (order,) least-squares readout weights for decoding n(t)
    """
    from nengo.utils.filter_design import cont2discrete as nengo_c2d

    theta = n_obs * T_STEP * LMU_THETA_MULT
    order = LMU_ORDER
    dt_lmu = 0.001

    Q = np.arange(order, dtype=np.float64)
    R = (2 * Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)
    A_cont = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
    B_cont = ((-1.0) ** Q)[:, None] * R
    C = np.ones((1, order))
    D = np.zeros((1,))
    A, B, _, _, _ = nengo_c2d((A_cont, B_cont, C, D), dt=dt_lmu, method="zoh")

    # solve readout weights offline
    pulse_fn = make_pulse_input(n_obs, amplitude=1.0)
    t_total = n_obs * T_STEP + T_ITI
    n_steps = int(t_total / dt_lmu)
    m = np.zeros(order)
    states, targets = [], []
    for k in range(n_steps):
        t_k = k * dt_lmu
        u = pulse_fn(t_k)
        m = A @ m + B.flatten() * u
        states.append(m.copy())
        targets.append(true_count(t_k, n_obs))
    states = np.array(states)
    targets = np.array(targets)
    W_readout, _, _, _ = np.linalg.lstsq(states, targets, rcond=None)

    pred = states @ W_readout
    rmse = float(np.sqrt(np.mean((pred - targets) ** 2)))
    print(f"  Offline readout RMSE: {rmse:.4f}, "
          f"max error: {np.abs(pred - targets).max():.4f}")

    return A, B, W_readout


def build_lmu_math(n_obs: int, n_neurons: int, seed: int) -> nengo.Network:
    """
    LMU counting circuit — pure math implementation (no neurons).

    The LMU state is represented exactly using nengo.Node objects.
    n_neurons is unused but kept for API consistency with other mechanisms.
    Use this to verify the LMU math before introducing neural noise.
    """
    A, B, W_readout = _compute_lmu_matrices(n_obs)
    pulse_fn = make_pulse_input(n_obs, amplitude=1.0)

    with nengo.Network(label="lmu_math", seed=seed) as net:
        # input node
        net.input = nengo.Node(pulse_fn, label="input")

        net.lmu_node = nengo.Node(size_in=LMU_ORDER, label="lmu_node")
        nengo.Connection(net.input,    net.lmu_node, transform=B,
                         synapse=None)
        nengo.Connection(net.lmu_node, net.lmu_node, transform=A,
                         synapse=0)

        net.readout = nengo.Node(size_in=1, label="readout")
        nengo.Connection(
            net.lmu_node,
            net.readout,
            transform=W_readout[np.newaxis, :],
            synapse=TAU_FB,
        )

        # probes
        net.probe_input  = nengo.Probe(net.input,   synapse=None,
                                       sample_every=PROBE_DT,
                                       label="probe_input")
        net.probe_memory = nengo.Probe(net.readout, synapse=0.01,
                                       sample_every=PROBE_DT,
                                       label="probe_memory")

    return net


def build_lmu_neural(n_obs: int, n_neurons: int, seed: int) -> nengo.Network:
    """
    LMU counting circuit — spiking neural implementation (stub).

    TODO: replace lmu_node with a spiking EnsembleArray to represent the
    LMU state in neurons. Currently identical to build_lmu_math.
    n_neurons will control the size of the EnsembleArray.
    """
    # stub: identical to math implementation for now
    return build_lmu_math(n_obs, n_neurons, seed)


# -- simulation & analysis -----------------------------------------------------


def run_and_analyse(
    mechanism: str,
    n_obs: int,
    n_neurons: int,
    seed: int,
    n_seeds: int,
) -> None:
    apply_style()

    if mechanism == "integrator":
        net = build_integrator(n_obs, n_neurons, seed)
    elif mechanism == "lmu_math":
        net = build_lmu_math(n_obs, n_neurons, seed)
    elif mechanism == "lmu_neural":
        net = build_lmu_neural(n_obs, n_neurons, seed)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism!r}")

    t_total = n_obs * T_STEP + T_ITI
    dt = 0.001
    all_decoded = []

    for s in range(seed, seed + n_seeds):
        print(f"  Running seed {s} / {seed + n_seeds - 1} ...")
        if mechanism == "integrator":
            net_s = build_integrator(n_obs, n_neurons, seed=s)
        elif mechanism == "lmu_math":
            net_s = build_lmu_math(n_obs, n_neurons, seed=s)
        elif mechanism == "lmu_neural":
            net_s = build_lmu_neural(n_obs, n_neurons, seed=s)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism!r}")
        with nengo.Simulator(net_s, dt=dt, progress_bar=False) as sim:
            sim.run(t_total)
        all_decoded.append(sim.data[net_s.probe_memory].squeeze())

    t = sim.trange(dt=PROBE_DT)
    input_ = sim.data[net_s.probe_input].squeeze()
    true = np.array([true_count(ti, n_obs) for ti in t])

    decoded_all = np.stack(all_decoded, axis=0)  # (n_seeds, T)
    decoded_mean = decoded_all.mean(axis=0)
    decoded_std = decoded_all.std(axis=0)

    # evaluate accuracy during the ITI: sample midpoint of each ITI window
    # i.e. T_OBS + T_ITI/2 after each pulse onset, where count should be stable
    eval_times = np.array([i * T_STEP + T_OBS + T_ITI / 2.0 for i in range(n_obs)])
    eval_idx = np.array([np.argmin(np.abs(t - te)) for te in eval_times])
    n_true_eval = true[eval_idx]
    n_decoded_eval = decoded_mean[eval_idx]
    rmse = float(np.sqrt(np.mean((n_decoded_eval - n_true_eval) ** 2)))
    print(f"Mechanism : {mechanism}")
    print(f"  n_obs={n_obs}, n_neurons={n_neurons}, seed={seed}")
    print(f"  RMSE: {rmse:.4f}")

    PALETTE = get_palette()
    nef_color = PALETTE["NEF"]

    # reshape decoded_all into long format for seaborn
    # shape: (n_seeds, T) -> long df with columns [time, decoded, seed]
    records = []
    for s_idx, dec in enumerate(all_decoded):
        for ti, val in zip(t, dec):
            records.append({"time": ti, "decoded": val, "seed": s_idx})
    df_long = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, constrained_layout=True)
    ax_dyn, ax_err = axes

    # -- dynamics panel ------------------------------------------------------------
    ax_dyn.plot(t, true, color="0.5", linewidth=1.0, linestyle="--",
                label="true n", zorder=2)
    sns.lineplot(
        data=df_long,
        x="time",
        y="decoded",
        color=nef_color,
        errorbar="sd",
        err_style="band",
        linewidth=1.5,
        label="decoded n (mean ± sd)",
        ax=ax_dyn,
    )
    ax_dyn.set_xlabel("Time (s)")
    ax_dyn.set_ylabel("Count")
    ax_dyn.set_title("Count dynamics")
    ax_dyn.set_xlim(0, t_total)
    ax_dyn.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax_dyn, top=True, right=True)

    # -- error vs n panel ----------------------------------------------------------
    ax_err.axhline(0, color="0.7", linewidth=1.0, linestyle="--")
    ax_err.scatter(
        n_true_eval,
        n_decoded_eval - n_true_eval,
        color=nef_color,
        s=20,
        zorder=3,
    )
    ax_err.set_xlabel("True n")
    ax_err.set_ylabel("Decoded − True")
    ax_err.set_title(f"Error vs. n  (RMSE={rmse:.3f})")
    sns.despine(ax=ax_err, top=True, right=True)

    fig.suptitle(
        f"Counting: {mechanism}  |  n_neurons={n_neurons}  "
        f"seeds={seed}–{seed + n_seeds - 1}",
        fontsize=9,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"counting_{mechanism}"
    plt.savefig(FIGURES_DIR / f"{stem}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.{{png,pdf}}")


# -- entry point ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neural counting circuit testbed")
    p.add_argument(
        "--mechanism",
        type=str,
        default="integrator",
        choices=["integrator", "lmu_math", "lmu_neural"],
        help="Counting mechanism to test",
    )
    p.add_argument("--n_obs", type=int, default=DEFAULT_N_OBS)
    p.add_argument("--n_neurons", type=int, default=DEFAULT_N_NEURONS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n_seeds", type=int, default=5,
                   help="Number of random seeds to average over")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_and_analyse(
        mechanism=args.mechanism,
        n_obs=args.n_obs,
        n_neurons=args.n_neurons,
        seed=args.seed,
        n_seeds=args.n_seeds,
    )
