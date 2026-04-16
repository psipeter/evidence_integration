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
    integrator   -- recurrent line-attractor working memory (baseline)
    lmu_math     -- LMU exact math node
    lmu_neural   -- LMU spiking EnsembleArray with pretrained count readout
    lmu_weight   -- Direct vs spiking LMU readouts for n(t) and 1/n^lambda (2x2 figure)

Saves figures to figures/counting_{mechanism}.pdf and figures/counting_{mechanism}.png
"""

from __future__ import annotations

import argparse
import sys
import time
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

# integrator parameters
TAU_FB = 0.2   # feedback synapse time constant (s)
RECUR_W = 1.0     # recurrent weight for line attractor

# LMU parameters
LMU_ORDER = 24     # number of Legendre polynomials
LMU_THETA_MULT = 1.1    # theta = n_obs * T_STEP * LMU_THETA_MULT
LMU_TAU = 0.2   # synaptic filter time constant for LMU neural connections
                # matches official Nengo-Loihi LMU example (Voelker et al.)
LMU_N_OBS_MAX = 30  # pretraining uses this n_obs for generalization across tasks
LMU_LAMBDA = 0.5  # default decay exponent for weight decoding test
TAU_PROBE = 0.1   # synapse on probe_memory / readout probes; pretrain targets match this


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

        # probes (default: sample every simulator dt)
        net.probe_input = nengo.Probe(net.input, synapse=None, label="probe_input")
        net.probe_memory = nengo.Probe(net.memory, synapse=TAU_PROBE, label="probe_memory")

    return net


def _compute_lmu_matrices(
    n_obs: int,
    lmu_order: int = LMU_ORDER,
    lmu_tau: float = LMU_TAU,
    verbose: bool = True,
) -> tuple:
    """
    Compute LMU A, B matrices and offline readout weights W.
    Returns (A, B, W_readout) where:
        A: (order, order) discretized state transition matrix
        B: (order, 1) discretized input matrix
        W_readout: (order,) least-squares readout weights for decoding n(t)
    """
    from nengo.utils.filter_design import cont2discrete as nengo_c2d

    theta = n_obs * T_STEP * LMU_THETA_MULT
    order = lmu_order
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
    # Normalize targets by n_obs to keep readout weights small (~2 vs ~60).
    targets_norm = np.array(targets) / n_obs
    W_readout, _, _, _ = np.linalg.lstsq(states, targets_norm, rcond=None)

    pred = states @ W_readout * n_obs
    rmse = float(np.sqrt(np.mean((pred - np.array(targets)) ** 2)))
    if verbose:
        print(f"  Offline readout RMSE: {rmse:.4f}, "
              f"max error: {np.abs(pred - np.array(targets)).max():.4f}")

    # Solve readout weights from tau-filtered trajectory for lmu_neural.
    # The synaptic filter smooths the ideal state; W_readout_filt matches
    # what the spiking EnsembleArray actually represents.
    m_filt = np.zeros(order)
    states_filt = []
    for k in range(n_steps):
        m_filt = m_filt + (dt_lmu / lmu_tau) * (states[k] - m_filt)
        states_filt.append(m_filt.copy())
    states_filt = np.array(states_filt)
    W_readout_filt, _, _, _ = np.linalg.lstsq(
        states_filt, targets_norm, rcond=None
    )

    return A, B, W_readout, W_readout_filt, A_cont, B_cont, theta, states_filt


def _pretrain_lmu_readout(
    n_obs_max: int,
    n_neurons: int,
    seed: int,
    A: np.ndarray,
    B: np.ndarray,
    radius: float,
    lmu_order: int = LMU_ORDER,
    lmu_tau: float = LMU_TAU,
    lambda_: float = LMU_LAMBDA,
    math: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a pretraining simulation to collect actual EnsembleArray activities
    and solve readout weights W_count and W_weight such that
    activities @ W_count ≈ low-passed n(t) (probe-consistent) and
    activities @ W_weight ≈ alpha from that same smoothed count,
    ``1 / n_filt^lambda`` (0 when ``n_filt <= 0``), avoiding sharp jumps from
    applying ``1/n^lambda`` to the raw staircase count.

    Always uses n_obs_max observations for pretraining. The resulting
    weights generalize to smaller n_obs at inference time, so a
    single pretrained network can serve all tasks.

    Uses the same seed as the main network build to ensure identical
    neuron tuning curves. Returns (W_count, W_weight), each shape (lmu_order,).

    If ``math`` is True, pretraining uses a Direct EnsembleArray so weights
    match the ideal ``build_lmu_weight(..., math=True)`` variant.
    """
    order = lmu_order
    tau = lmu_tau
    pulse_fn = make_pulse_input(n_obs_max, amplitude=1.0)
    t_total = n_obs_max * T_STEP + T_ITI
    dt = 0.001

    neuron_type = nengo.Direct() if math else nengo.SpikingRectifiedLinear()

    with nengo.Network(label="lmu_pretrain", seed=seed) as pre_net:
        pre_input = nengo.Node(pulse_fn, label="input")
        pre_ea = nengo.networks.EnsembleArray(
            n_neurons=n_neurons,
            n_ensembles=order,
            neuron_type=neuron_type,
            radius=radius,
            label="lmu_ea",
            seed=seed,
        )
        nengo.Connection(pre_input, pre_ea.input,
                         transform=B, synapse=tau)
        nengo.Connection(pre_ea.output, pre_ea.input,
                         transform=A, synapse=tau)
        probe_output = nengo.Probe(pre_ea.output, synapse=tau)

    with nengo.Simulator(pre_net, dt=dt, seed=seed, progress_bar=False) as sim:
        sim.run(t_total)

    t = sim.trange()
    activities = sim.data[probe_output]   # (T, order)
    targets = np.array([true_count(ti, n_obs_max) for ti in t])
    # Match probe_memory: first-order low-pass at simulator dt with same tau as probes.
    targets_filt = np.zeros_like(targets, dtype=float)
    for k in range(1, len(targets)):
        targets_filt[k] = targets_filt[k - 1] + (dt / TAU_PROBE) * (
            targets[k] - targets_filt[k - 1]
        )

    # solve W_count: activities @ W_count ≈ filtered n(t) (as seen on probe_memory)
    W_count, _, _, _ = np.linalg.lstsq(activities, targets_filt, rcond=None)

    # solve W_weight: map spikes to alpha from *smoothed* count (no sharp 1/n^lambda jumps)
    alpha_smooth = np.where(
        targets_filt > 0, 1.0 / targets_filt ** lambda_, 0.0
    )
    W_weight, _, _, _ = np.linalg.lstsq(activities, alpha_smooth, rcond=None)

    return W_count, W_weight


def build_lmu_math(
    n_obs: int,
    n_neurons: int,
    seed: int,
    lmu_order: int = LMU_ORDER,
    lmu_tau: float = LMU_TAU,
) -> nengo.Network:
    """
    LMU counting circuit — pure math implementation (no neurons).

    The LMU state is represented exactly using nengo.Node objects.
    n_neurons is unused but kept for API consistency with other mechanisms.
    Use this to verify the LMU math before introducing neural noise.
    """
    A, B, W_readout, _, _, _, _, _ = _compute_lmu_matrices(
        n_obs,
        lmu_order=lmu_order,
        lmu_tau=lmu_tau,
        verbose=False,
    )
    pulse_fn = make_pulse_input(n_obs, amplitude=1.0)

    with nengo.Network(label="lmu_math", seed=seed) as net:
        # input node
        net.input = nengo.Node(pulse_fn, label="input")

        net.lmu_node = nengo.Node(size_in=lmu_order, label="lmu_node")
        nengo.Connection(net.input,    net.lmu_node, transform=B,
                         synapse=None)
        nengo.Connection(net.lmu_node, net.lmu_node, transform=A,
                         synapse=0)

        net.readout = nengo.Node(size_in=1, label="readout")
        nengo.Connection(
            net.lmu_node,
            net.readout,
            transform=W_readout[np.newaxis, :] * n_obs,
            synapse=lmu_tau,
        )

        # probes (default: sample every simulator dt)
        net.probe_input = nengo.Probe(net.input, synapse=None, label="probe_input")
        net.probe_memory = nengo.Probe(
            net.readout, synapse=TAU_PROBE, label="probe_memory"
        )

    return net


def build_lmu_neural(
    n_obs: int,
    n_neurons: int,
    seed: int,
    lmu_order: int = LMU_ORDER,
    lmu_tau: float = LMU_TAU,
) -> nengo.Network:
    """
    LMU counting circuit — spiking neural implementation.

    Uses Euler discretization at the Nengo simulation dt level (Voelker 2019):
        Ā = dt * A_cont + I
        B̄ = dt * B_cont
    where A_cont, B_cont are the continuous-time matrices (pre-scaled by 1/θ).
    Spike filtering uses LMU_TAU as the synaptic time constant.

    Parameters
    ----------
    n_neurons : int
        Neurons per sub-ensemble. Total = n_neurons * LMU_ORDER.
    """
    A, B, W_readout, _, _, _, _, _ = _compute_lmu_matrices(
        n_obs,
        lmu_order=lmu_order,
        lmu_tau=lmu_tau,
        verbose=False,
    )
    pulse_fn = make_pulse_input(n_obs, amplitude=1.0)
    order = lmu_order
    # Use ZOH-discretized A and B directly as transforms.
    # These are already stable (eigenvalues ~0.999).
    # synapse=tau filters spikes — it does not alter the dynamics.
    tau = lmu_tau
    A_H = A     # ZOH discretized, shape (order, order)
    B_H = B     # ZOH discretized, shape (order, 1)
    radius = 2.0
    # Pretrain on LMU_N_OBS_MAX observations — generalizes to all task scales.
    # A, B, radius are computed from the actual n_obs for task-appropriate dynamics.
    W_count, _W_weight = _pretrain_lmu_readout(
        n_obs_max=LMU_N_OBS_MAX,
        n_neurons=n_neurons,
        seed=seed,
        A=A,
        B=B,
        radius=radius,
        lmu_order=lmu_order,
        lmu_tau=lmu_tau,
    )

    with nengo.Network(label="lmu_neural", seed=seed) as net:
        net.input = nengo.Node(pulse_fn, label="input")

        # spiking EnsembleArray matching official Nengo-Loihi example
        net.lmu_ea = nengo.networks.EnsembleArray(
            n_neurons=n_neurons,
            n_ensembles=order,
            neuron_type=nengo.SpikingRectifiedLinear(),
            radius=radius,
            label="lmu_ea",
            seed=seed,
        )

        # input -> LMU
        nengo.Connection(
            net.input,
            net.lmu_ea.input,
            transform=B_H,
            synapse=tau,
        )

        # recurrent
        nengo.Connection(
            net.lmu_ea.output,
            net.lmu_ea.input,
            transform=A_H,
            synapse=tau,
        )

        # lmu_ea.output is a passthrough Node — cannot use function= on it.
        # Pass state through a Direct Ensemble first, then decode from that.
        net.state_ens = nengo.Ensemble(
            n_neurons=1,
            dimensions=order,
            neuron_type=nengo.Direct(),
            label="state_ens",
        )
        nengo.Connection(
            net.lmu_ea.output,
            net.state_ens,
            synapse=tau,
        )

        net.readout = nengo.Ensemble(
            n_neurons=1,
            dimensions=1,
            neuron_type=nengo.Direct(),
            label="readout",
        )
        nengo.Connection(
            net.state_ens,
            net.readout,
            function=lambda x, W=W_count: np.dot(W, x),
            synapse=tau,
            solver=nengo.solvers.LstsqL2(reg=1e-3),
            transform=1.0,
        )

        net.probe_input = nengo.Probe(net.input, synapse=None, label="probe_input")
        net.probe_memory = nengo.Probe(
            net.readout, synapse=TAU_PROBE, label="probe_memory"
        )

    return net


def build_lmu_weight(
    n_obs: int,
    n_neurons: int,
    seed: int,
    lmu_order: int = LMU_ORDER,
    lmu_tau: float = LMU_TAU,
    lambda_: float = LMU_LAMBDA,
    math: bool = False,
) -> nengo.Network:
    """
    LMU circuit decoding both n(t) and alpha(n) = 1/n^lambda from ensemble state.

    ``math=True`` uses ``Direct`` sub-ensembles (ideal); ``math=False`` uses
    spiking ``SpikingRectifiedLinear``. Two Direct readout heads decode count
    and weight from the same filtered state passthrough.
    """
    A, B, _, _, _, _, _, states_filt = _compute_lmu_matrices(
        n_obs,
        lmu_order=lmu_order,
        lmu_tau=lmu_tau,
        verbose=False,
    )
    pulse_fn = make_pulse_input(n_obs, amplitude=1.0)
    order = lmu_order
    tau = lmu_tau
    A_H = A
    B_H = B

    radius = float(np.abs(states_filt).max()) * 1.5

    W_count, W_weight = _pretrain_lmu_readout(
        LMU_N_OBS_MAX,
        n_neurons,
        seed,
        A,
        B,
        radius,
        lmu_order=lmu_order,
        lmu_tau=lmu_tau,
        lambda_=lambda_,
        math=math,
    )

    neuron_type = nengo.Direct() if math else nengo.SpikingRectifiedLinear()
    label = "lmu_weight_math" if math else "lmu_weight_neural"

    with nengo.Network(label=label, seed=seed) as net:
        net.input = nengo.Node(pulse_fn, label="input")

        net.lmu_ea = nengo.networks.EnsembleArray(
            n_neurons=n_neurons,
            n_ensembles=order,
            neuron_type=neuron_type,
            radius=radius,
            seed=seed,
            label="lmu_ea",
        )
        nengo.Connection(net.input, net.lmu_ea.input, transform=B_H, synapse=tau)
        nengo.Connection(
            net.lmu_ea.output, net.lmu_ea.input, transform=A_H, synapse=tau
        )

        net.lmu_state = nengo.Ensemble(
            1,
            order,
            neuron_type=nengo.Direct(),
            label="lmu_state",
        )
        nengo.Connection(net.lmu_ea.output, net.lmu_state, synapse=tau)

        net.readout_count = nengo.Ensemble(
            1, 1, neuron_type=nengo.Direct(), label="readout_count"
        )
        nengo.Connection(
            net.lmu_state,
            net.readout_count,
            function=lambda x, W=W_count: np.dot(W, x),
            synapse=tau,
        )

        net.readout_weight = nengo.Ensemble(
            1, 1, neuron_type=nengo.Direct(), label="readout_weight"
        )
        nengo.Connection(
            net.lmu_state,
            net.readout_weight,
            function=lambda x, W=W_weight: np.dot(W, x),
            synapse=tau,
        )

        net.probe_input = nengo.Probe(net.input, synapse=None, label="probe_input")
        net.probe_count = nengo.Probe(
            net.readout_count, synapse=TAU_PROBE, label="probe_count"
        )
        net.probe_weight = nengo.Probe(
            net.readout_weight, synapse=TAU_PROBE, label="probe_weight"
        )

    return net


# -- simulation & analysis -----------------------------------------------------


def run_and_analyse(
    mechanism: str,
    n_obs: int,
    n_neurons: int,
    seed: int,
    n_seeds: int,
    lmu_order: int = LMU_ORDER,
    lmu_tau: float = LMU_TAU,
    lambda_: float = LMU_LAMBDA,
) -> None:
    apply_style()

    if mechanism == "integrator":
        net = build_integrator(n_obs, n_neurons, seed)
    elif mechanism == "lmu_math":
        net = build_lmu_math(
            n_obs, n_neurons, seed, lmu_order=lmu_order, lmu_tau=lmu_tau
        )
    elif mechanism == "lmu_neural":
        net = build_lmu_neural(
            n_obs, n_neurons, seed, lmu_order=lmu_order, lmu_tau=lmu_tau
        )
    elif mechanism == "lmu_weight":
        # Built and simulated in dedicated branch (math vs neural).
        pass
    else:
        raise ValueError(f"Unknown mechanism: {mechanism!r}")

    t_total = n_obs * T_STEP + T_ITI
    dt = 0.001

    if mechanism == "lmu_weight":
        all_count_math: list[np.ndarray] = []
        all_weight_math: list[np.ndarray] = []
        all_count_neural: list[np.ndarray] = []
        all_weight_neural: list[np.ndarray] = []

        t0 = time.time()
        for s in range(seed, seed + n_seeds):
            net_m = build_lmu_weight(
                n_obs,
                n_neurons,
                seed=s,
                lmu_order=lmu_order,
                lmu_tau=lmu_tau,
                lambda_=lambda_,
                math=True,
            )
            with nengo.Simulator(net_m, dt=dt, progress_bar=False) as sim:
                sim.run(t_total)
            all_count_math.append(sim.data[net_m.probe_count].squeeze())
            all_weight_math.append(sim.data[net_m.probe_weight].squeeze())

            net_n = build_lmu_weight(
                n_obs,
                n_neurons,
                seed=s,
                lmu_order=lmu_order,
                lmu_tau=lmu_tau,
                lambda_=lambda_,
                math=False,
            )
            with nengo.Simulator(net_n, dt=dt, progress_bar=False) as sim:
                sim.run(t_total)
            all_count_neural.append(sim.data[net_n.probe_count].squeeze())
            all_weight_neural.append(sim.data[net_n.probe_weight].squeeze())

        t = sim.trange()
        true = np.array([true_count(ti, n_obs) for ti in t])
        true_alpha = np.where(true > 0, 1.0 / true ** lambda_, 0.0)
        # Same low-pass n(t) as pretrain / count target, then alpha = 1 / n_filt^lambda
        true_n_filt = np.zeros_like(true, dtype=float)
        for k in range(1, len(true)):
            true_n_filt[k] = true_n_filt[k - 1] + (dt / TAU_PROBE) * (
                true[k] - true_n_filt[k - 1]
            )
        true_alpha_smooth = np.where(
            true_n_filt > 0, 1.0 / true_n_filt ** lambda_, 0.0
        )

        count_math = np.stack(all_count_math, axis=0)
        count_neural = np.stack(all_count_neural, axis=0)
        weight_math = np.stack(all_weight_math, axis=0)
        weight_neural = np.stack(all_weight_neural, axis=0)

        count_math_mean = count_math.mean(axis=0)
        count_neural_mean = count_neural.mean(axis=0)
        count_neural_std = count_neural.std(axis=0)
        weight_math_mean = weight_math.mean(axis=0)
        weight_neural_mean = weight_neural.mean(axis=0)
        weight_neural_std = weight_neural.std(axis=0)

        eval_times = np.array(
            [i * T_STEP + T_OBS + T_ITI / 2.0 for i in range(n_obs)]
        )
        eval_idx = np.array([np.argmin(np.abs(t - te)) for te in eval_times])
        n_true_eval = true[eval_idx]
        alpha_true_eval = true_alpha[eval_idx]

        def _rmse_stack(stack: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
            per = [
                float(np.sqrt(np.mean((row[eval_idx] - ref) ** 2)))
                for row in stack
            ]
            return float(np.mean(per)), float(np.std(per)), float(
                np.sqrt(np.mean((stack.mean(axis=0)[eval_idx] - ref) ** 2))
            )

        rmse_c_m, rmse_c_m_std, _ = _rmse_stack(count_math, n_true_eval)
        rmse_c_n, rmse_c_n_std, _ = _rmse_stack(count_neural, n_true_eval)
        rmse_w_m, rmse_w_m_std, _ = _rmse_stack(weight_math, alpha_true_eval)
        rmse_w_n, rmse_w_n_std, _ = _rmse_stack(weight_neural, alpha_true_eval)

        elapsed = time.time() - t0
        print(
            f"Mechanism : lmu_weight  |  order={lmu_order}  tau={lmu_tau}  "
            f"neurons={n_neurons}  lambda={lambda_}"
        )
        print(
            f"  RMSE count  — math: {rmse_c_m:.3f} +/- {rmse_c_m_std:.3f}  "
            f"neural: {rmse_c_n:.3f} +/- {rmse_c_n_std:.3f}"
        )
        print(
            f"  RMSE weight — math: {rmse_w_m:.3f} +/- {rmse_w_m_std:.3f}  "
            f"neural: {rmse_w_n:.3f} +/- {rmse_w_n_std:.3f}"
        )
        print(
            f"  ({n_seeds} seeds x 2 variants, {elapsed:.1f}s total, "
            f"{elapsed/max(n_seeds,1):.1f}s/seed)"
        )

        PALETTE = get_palette()
        nef_color = PALETTE["NEF"]
        grey = "0.45"

        fig_w = float(FIGURE_SIZE[0]) * 1.35
        fig_h = float(FIGURE_SIZE[1]) * 1.35
        fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), constrained_layout=True)
        ax00, ax01 = axes[0, 0], axes[0, 1]
        ax10, ax11 = axes[1, 0], axes[1, 1]

        # [0,0] count dynamics
        ax00.plot(t, true, color="0.5", linewidth=1.0, linestyle="--",
                  label="true n", zorder=2)
        ax00.plot(t, count_math_mean, color=grey, linewidth=1.5,
                  label="math (mean)", zorder=3)
        ax00.fill_between(
            t,
            count_neural_mean - count_neural_std,
            count_neural_mean + count_neural_std,
            color=nef_color,
            alpha=0.25,
            linewidth=0,
        )
        ax00.plot(
            t,
            count_neural_mean,
            color=nef_color,
            linewidth=1.5,
            label="neural (mean +/- sd)",
            zorder=4,
        )
        ax00.set_xlabel("Time (s)")
        ax00.set_ylabel("Count")
        ax00.set_title("Count dynamics")
        ax00.set_xlim(0, t_total)
        ax00.legend(fontsize=7, frameon=False)
        sns.despine(ax=ax00, top=True, right=True)

        # [0,1] count error vs n
        ax01.axhline(0, color="0.7", linewidth=1.0, linestyle="--")
        err_c_m = count_math_mean[eval_idx] - n_true_eval
        err_c_n = count_neural_mean[eval_idx] - n_true_eval
        rmse_c_m_pt = float(np.sqrt(np.mean(err_c_m**2)))
        rmse_c_n_pt = float(np.sqrt(np.mean(err_c_n**2)))
        ax01.scatter(
            n_true_eval,
            err_c_m,
            color=grey,
            s=20,
            zorder=3,
            label=f"math (RMSE={rmse_c_m_pt:.3f})",
        )
        ax01.scatter(
            n_true_eval,
            err_c_n,
            color=nef_color,
            s=20,
            zorder=4,
            label=f"neural (RMSE={rmse_c_n_pt:.3f})",
        )
        ax01.set_xlabel("True n")
        ax01.set_ylabel("Decoded - True")
        ax01.set_title("Count error vs. n")
        ax01.legend(fontsize=7, frameon=False)
        sns.despine(ax=ax01, top=True, right=True)

        # [1,0] weight dynamics
        ax10.plot(
            t,
            true_alpha_smooth,
            color="0.5",
            linewidth=1.0,
            linestyle="--",
            label=f"target alpha = 1 / n_filt^{lambda_} (n low-pass)",
            zorder=2,
        )
        ax10.plot(t, weight_math_mean, color=grey, linewidth=1.5,
                  label="math (mean)", zorder=3)
        ax10.fill_between(
            t,
            weight_neural_mean - weight_neural_std,
            weight_neural_mean + weight_neural_std,
            color=nef_color,
            alpha=0.25,
            linewidth=0,
        )
        ax10.plot(
            t,
            weight_neural_mean,
            color=nef_color,
            linewidth=1.5,
            label="neural (mean +/- sd)",
            zorder=4,
        )
        ax10.set_xlabel("Time (s)")
        ax10.set_ylabel("alpha(n)")
        ax10.set_title("Weight dynamics")
        ax10.set_xlim(0, t_total)
        ax10.legend(fontsize=7, frameon=False)
        sns.despine(ax=ax10, top=True, right=True)

        # [1,1] weight error vs n (at ITI midpoints)
        ax11.axhline(0, color="0.7", linewidth=1.0, linestyle="--")
        err_w_m = weight_math_mean[eval_idx] - alpha_true_eval
        err_w_n = weight_neural_mean[eval_idx] - alpha_true_eval
        rmse_w_m_pt = float(np.sqrt(np.mean(err_w_m**2)))
        rmse_w_n_pt = float(np.sqrt(np.mean(err_w_n**2)))
        ax11.scatter(
            n_true_eval,
            err_w_m,
            color=grey,
            s=20,
            zorder=3,
            label=f"math (RMSE={rmse_w_m_pt:.3f})",
        )
        ax11.scatter(
            n_true_eval,
            err_w_n,
            color=nef_color,
            s=20,
            zorder=4,
            label=f"neural (RMSE={rmse_w_n_pt:.3f})",
        )
        ax11.set_xlabel("True n")
        ax11.set_ylabel("Decoded - True")
        ax11.set_title("Weight error vs. n")
        ax11.legend(fontsize=7, frameon=False)
        sns.despine(ax=ax11, top=True, right=True)

        fig.suptitle(
            f"Counting: lmu_weight (math vs neural)  |  n_neurons={n_neurons}  "
            f"seeds={seed}-{seed + n_seeds - 1}  lambda={lambda_}",
            fontsize=9,
        )

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        stem = "counting_lmu_weight"
        plt.savefig(FIGURES_DIR / f"{stem}.png", dpi=300)
        plt.savefig(FIGURES_DIR / f"{stem}.pdf")
        print(f"Saved figures/{stem}.{{png,pdf}}")
        return

    all_decoded = []

    t0 = time.time()
    for s in range(seed, seed + n_seeds):
        if mechanism == "integrator":
            net_s = build_integrator(n_obs, n_neurons, seed=s)
        elif mechanism == "lmu_math":
            net_s = build_lmu_math(
                n_obs, n_neurons, seed=s, lmu_order=lmu_order, lmu_tau=lmu_tau
            )
        elif mechanism == "lmu_neural":
            net_s = build_lmu_neural(
                n_obs, n_neurons, seed=s, lmu_order=lmu_order, lmu_tau=lmu_tau
            )
        else:
            raise ValueError(f"Unknown mechanism: {mechanism!r}")
        with nengo.Simulator(net_s, dt=dt, progress_bar=False) as sim:
            sim.run(t_total)
        all_decoded.append(sim.data[net_s.probe_memory].squeeze())

    t = sim.trange()
    true = np.array([true_count(ti, n_obs) for ti in t])

    decoded_all = np.stack(all_decoded, axis=0)  # (n_seeds, T)
    elapsed = time.time() - t0
    decoded_mean = decoded_all.mean(axis=0)
    decoded_std = decoded_all.std(axis=0)

    # evaluate accuracy during the ITI: sample midpoint of each ITI window
    # i.e. T_OBS + T_ITI/2 after each pulse onset, where count should be stable
    eval_times = np.array([i * T_STEP + T_OBS + T_ITI / 2.0 for i in range(n_obs)])
    eval_idx = np.array([np.argmin(np.abs(t - te)) for te in eval_times])
    n_true_eval = true[eval_idx]
    n_decoded_eval = decoded_mean[eval_idx]
    rmse = float(np.sqrt(np.mean((n_decoded_eval - n_true_eval) ** 2)))
    # compute per-seed RMSE for mean/std
    per_seed_rmse = []
    for dec in all_decoded:
        r = float(np.sqrt(np.mean((dec[eval_idx] - n_true_eval) ** 2)))
        per_seed_rmse.append(r)
    rmse_mean = float(np.mean(per_seed_rmse))
    rmse_std = float(np.std(per_seed_rmse))

    mech_line = (
        f"Mechanism : {mechanism}  |  order={lmu_order}  tau={lmu_tau}  "
        f"neurons={n_neurons}"
    )
    print(mech_line)
    print(f"  RMSE: {rmse_mean:.3f} ± {rmse_std:.3f}  "
          f"({n_seeds} seeds, {elapsed:.1f}s total, {elapsed/n_seeds:.1f}s/seed)")

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

    # -- dynamics panel (decoded count vs true n) ---------------------------------
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

    # -- error vs n panel (count decode at ITI midpoints) -------------------------
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

    supt = (
        f"Counting: {mechanism}  |  n_neurons={n_neurons}  "
        f"seeds={seed}–{seed + n_seeds - 1}"
    )
    fig.suptitle(supt, fontsize=9)

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
        choices=["integrator", "lmu_math", "lmu_neural", "lmu_weight"],
        help="Counting mechanism to test",
    )
    p.add_argument("--n_obs", type=int, default=DEFAULT_N_OBS)
    p.add_argument("--n_neurons", type=int, default=DEFAULT_N_NEURONS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n_seeds", type=int, default=5,
                   help="Number of random seeds to average over")
    p.add_argument("--lmu_order", type=int, default=LMU_ORDER,
                   help="Number of Legendre polynomials for LMU")
    p.add_argument("--lmu_tau",   type=float, default=LMU_TAU,
                   help="Synaptic time constant for LMU neural connections")
    p.add_argument(
        "--lambda_",
        type=float,
        default=LMU_LAMBDA,
        help="Decay exponent λ for α(n)=1/n^λ in lmu_weight pretrain/decode",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_and_analyse(
        mechanism=args.mechanism,
        n_obs=args.n_obs,
        n_neurons=args.n_neurons,
        seed=args.seed,
        n_seeds=args.n_seeds,
        lmu_order=args.lmu_order,
        lmu_tau=args.lmu_tau,
        lambda_=args.lambda_,
    )
