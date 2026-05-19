#!/usr/bin/env python3
"""
2D NEF evidence integrator for the Diederen interleaved task.

Tracks two simultaneous reward distributions with independent 1D counting
circuits (count_A, count_B) and context-gated learning rates. One simulation
per session over the full ``trial_in_session`` sequence.

Usage:
    from models.NEF2d import run
    responses = run(params)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import nengo
import numpy as np
import pandas as pd

nengo.rc.set("decoder_cache", "enabled", "False")

for _logger_name in (
    "nengo",
    "nengo.simulator",
    "nengo.builder",
    "nengo.builder.network",
    "nengo.builder.optimizer",
    "nengo.builder.connection",
):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.model_params import MODEL_PARAMS, _NEF_FIXED
from models.counting_integrator import (
    _ideal_stream as _ideal_stream_1d,
    decode_outputs as _decode_outputs_1d,
)
from utils.paths import RUNS_DIR, data_path
from utils.run_params import trial_seed as _trial_seed

READOUT_FRACTION = 0.8
SWITCH_THRESHOLD = 0.5

_NEF2D_EXTRA: dict[str, object] = {
    "n_neurons_counting": 2000,
    "radius_c": 50.0,
    "onset_detector_amp": 0.5,
    "count_leak": 0.005,
}
PARAM_DEFAULTS: dict = {**_NEF_FIXED, **_NEF2D_EXTRA}


def _session_distrib_map(session_df: pd.DataFrame) -> tuple[int, int]:
    """Map lower/higher ``distrib_index`` in session to A/B dimensions."""
    distribs = sorted(int(d) for d in session_df["distrib_index"].dropna().unique())
    if len(distribs) != 2:
        raise ValueError(
            f"Expected exactly 2 distrib_index values per session, got {distribs}"
        )
    return distribs[0], distribs[1]


def _eval_input_at_t(
    t: float,
    session_rows: list[dict],
    distrib_a: int,
    distrib_b: int,
    params: dict,
) -> np.ndarray:
    """4D input vector at simulation time ``t`` for one session."""
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    out = np.zeros(4, dtype=float)

    if t < t_iti:
        return out

    rel = t - t_iti
    step = int(rel / t_step)
    phase = rel - step * t_step
    if step >= len(session_rows) or phase >= t_obs:
        return out

    row = session_rows[step]
    val = float(row["value"])
    d = int(row["distrib_index"])
    if d == distrib_a:
        out[0] = val
        out[2] = 1.0
    elif d == distrib_b:
        out[1] = val
        out[3] = 1.0
    return out


def _make_session_input(
    session_rows: list[dict],
    distrib_a: int,
    distrib_b: int,
    params: dict,
) -> callable:
    """Nengo Node function: 4D input for one diederen session."""

    def fn(t: float) -> list[float]:
        return _eval_input_at_t(t, session_rows, distrib_a, distrib_b, params).tolist()

    return fn


def _session_input_timeseries(
    session_df: pd.DataFrame,
    params: dict,
) -> tuple[list[dict], int, int, callable]:
    """
    Build session row list, distrib A/B ids, and input Node callable.

    ``session_df`` must be one session, sorted by ``trial_in_session``.
    """
    sess = session_df.sort_values("trial_in_session").reset_index(drop=True)
    rows = sess.to_dict("records")
    distrib_a, distrib_b = _session_distrib_map(sess)
    input_fn = _make_session_input(rows, distrib_a, distrib_b, params)
    return rows, distrib_a, distrib_b, input_fn


def _ideal_targets_at_times(
    t_arr: np.ndarray,
    session_rows: list[dict],
    distrib_a: int,
    distrib_b: int,
    params: dict,
) -> np.ndarray:
    """Ideal ``[count_A, count_B, alpha_A, alpha_B]`` at each simulation time."""
    alpha_0 = float(params["alpha_0"])
    lambda_ = float(params["lambda_"])
    count_a, count_b = 0, 0
    prev_ctx_a, prev_ctx_b = 0.0, 0.0
    targets = np.zeros((len(t_arr), 4), dtype=float)

    for i, t in enumerate(t_arr):
        inp = _eval_input_at_t(float(t), session_rows, distrib_a, distrib_b, params)
        ctx_a, ctx_b = float(inp[2]), float(inp[3])
        if ctx_a > 0.5 and prev_ctx_a <= 0.5:
            count_a += 1
        if ctx_b > 0.5 and prev_ctx_b <= 0.5:
            count_b += 1
        prev_ctx_a, prev_ctx_b = ctx_a, ctx_b
        targets[i, 0] = count_a
        targets[i, 1] = count_b
        targets[i, 2] = alpha_0 / max(count_a, 1) ** lambda_
        targets[i, 3] = alpha_0 / max(count_b, 1) ** lambda_

    return targets


def _session_duration(n_steps: int, params: dict) -> float:
    t_step = float(params["t_obs"]) + float(params["t_iti"])
    return float(params["t_iti"]) + n_steps * t_step


def _validate_1d_decoders(decoders: dict, n_neurons_counting: int) -> None:
    """Decoder weights must match 1D count ensemble size."""
    expected = (1, n_neurons_counting)
    for key in ("W_count", "W_weight"):
        w = decoders[key]
        if tuple(w.shape) != expected:
            raise ValueError(
                f"Decoder {key!r} shape {w.shape} != {expected} "
                f"(count uses n_neurons_counting={n_neurons_counting})"
            )


def _make_pulse_input_signs(params: dict, signs: np.ndarray) -> callable:
    """1D pulse input with explicit per-observation signs (for pretraining)."""
    n_obs = len(signs)
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti

    def pulse(t: float) -> float:
        if t < t_iti:
            return 0.0
        step = int((t - t_iti) / t_step)
        phase = (t - t_iti) - step * t_step
        if step < n_obs and phase < t_obs:
            return float(signs[step])
        return 0.0

    return pulse


def _build_1d_counting_train(
    params: dict, pulse_fn: callable
) -> nengo.Network:
    """1D counting train network (no leak), matching counting_integrator.py."""
    n_neurons = int(params["n_neurons"])
    n_neurons_counting = int(params["n_neurons_counting"])
    n_obs = int(params["n_obs"])
    seed = int(params.get("base_seed", params["seed"]))
    tau_fb = float(params["tau_fb"])
    tau_probe = float(params["tau_probe"])
    amp = float(params["onset_detector_amp"])

    with nengo.Network(label="counting_1d_train", seed=seed) as net:
        net.node_input = nengo.Node(pulse_fn, label="node_input")
        net.ideal = nengo.Node(
            _ideal_stream_1d(params), size_in=1, size_out=2, label="ideal"
        )
        nengo.Connection(net.node_input, net.ideal, synapse=None, seed=seed)
        net.probe_ideal_raw = nengo.Probe(net.ideal, synapse=None)

        net.memory = nengo.Ensemble(
            n_neurons=n_neurons_counting,
            dimensions=1,
            radius=n_obs,
            label="memory",
            seed=seed,
        )
        net.onset_detector = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=1,
            encoders=nengo.dists.Choice([[1]]),
            intercepts=nengo.dists.Uniform(0, 1),
            seed=seed,
            label="onset_detector",
        )
        nengo.Connection(
            net.node_input,
            net.onset_detector,
            synapse=float(params["tau_fast"]),
            function=lambda x: np.abs(x),
            seed=seed,
        )
        nengo.Connection(
            net.node_input,
            net.onset_detector,
            synapse=float(params["tau_slow"]),
            function=lambda x: -np.abs(x),
            seed=seed,
        )
        nengo.Connection(
            net.onset_detector,
            net.memory,
            synapse=tau_fb,
            function=lambda x, amp=amp: amp,
            seed=seed,
        )
        nengo.Connection(net.memory, net.memory, transform=1.0, synapse=tau_fb, seed=seed)
        net.probe_memory_raw = nengo.Probe(net.memory.neurons, synapse=None)

    return net


def _simulate_1d_counting_train(net: nengo.Network, params: dict) -> dict:
    """Run 1D train network; return ideal and memory activities."""
    dt = float(params["dt"])
    n_obs = int(params["n_obs"])
    t_step = float(params["t_obs"]) + float(params["t_iti"])
    t_total = n_obs * t_step
    with nengo.Simulator(
        net, dt=dt, seed=int(params["seed"]), progress_bar=False
    ) as sim:
        sim.run(t_total)
    return {
        "ideal": sim.data[net.probe_ideal_raw],
        "memory": sim.data[net.probe_memory_raw],
    }


def _pretrain_counting_1d(params: dict) -> dict:
    """
    Train 1D decoders (W_count, W_weight) on a single monotonic pulse sequence.

    A single run from count=0 to count=radius_c covers the full representable
    range for a 1D integrator. count_leak=0 during training preserves decoder
    validity. The same decoders are reused for both count_A and count_B.
    """
    n_obs = int(params["radius_c"])
    train_params = {
        **params,
        "n_obs": n_obs,
        "count_leak": 0.0,
        "seed": int(params.get("base_seed", params["seed"])),
    }
    signs = np.ones(n_obs)
    pulse_fn = _make_pulse_input_signs(train_params, signs)
    net = _build_1d_counting_train(train_params, pulse_fn)
    raw = _simulate_1d_counting_train(net, train_params)
    return _decode_outputs_1d(raw, train_params)


def _build_1d_counting_circuit(
    net: nengo.Network,
    name: str,
    input_dim: int,
    params: dict,
    seed: int,
    decoders: dict,
) -> None:
    """
    Build one complete 1D counting circuit into an existing network.
    Attaches onset detector and integrator for one distribution.
    Sets net.onset_{name} and net.count_{name} as attributes.
    """
    n_neurons = int(params["n_neurons"])
    n_neurons_counting = int(params["n_neurons_counting"])
    radius_c = float(params["radius_c"])
    tau_fb = float(params["tau_fb"])
    tau_fast = float(params["tau_fast"])
    tau_slow = float(params["tau_slow"])
    tau_probe = float(params["tau_probe"])
    amp = float(params["onset_detector_amp"])
    count_leak = float(params.get("count_leak", 0.0))

    onset = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=1,
        radius=1.5,
        encoders=nengo.dists.Choice([[1]]),
        intercepts=nengo.dists.Uniform(0.0, 1.0),
        label=f"onset_{name}",
        seed=seed,
    )
    setattr(net, f"onset_{name}", onset)

    nengo.Connection(
        net.input_node[input_dim], onset, synapse=tau_fast, seed=seed
    )
    nengo.Connection(
        net.input_node[input_dim],
        onset,
        synapse=tau_slow,
        function=lambda x: -x,
        seed=seed,
    )

    count = nengo.Ensemble(
        n_neurons=n_neurons_counting,
        dimensions=1,
        radius=radius_c,
        label=f"count_{name}",
        seed=seed,
    )
    setattr(net, f"count_{name}", count)

    nengo.Connection(
        onset,
        count,
        synapse=tau_fb,
        function=lambda x, amp=amp: [amp] if x > 0 else [0.0],
        seed=seed,
    )
    nengo.Connection(count, count, transform=1.0, synapse=tau_fb, seed=seed)
    if count_leak > 0.0:
        nengo.Connection(count, count, transform=-count_leak, synapse=tau_fb, seed=seed)

    alpha_node = nengo.Ensemble(
        1,
        1,
        neuron_type=nengo.Direct(),
        label=f"alpha_{name}_node",
        seed=seed,
    )
    setattr(net, f"alpha_{name}_node", alpha_node)

    nengo.Connection(
        count.neurons,
        alpha_node,
        transform=decoders["W_weight"],
        synapse=tau_probe,
        seed=seed,
    )

    setattr(
        net,
        f"probe_count_{name}_neurons",
        nengo.Probe(count.neurons, synapse=None, sample_every=float(params["dt"])),
    )
    setattr(net, f"probe_alpha_{name}", nengo.Probe(alpha_node, synapse=None))


def _switch_fn(t: float, x: np.ndarray) -> np.ndarray:
    alpha_A, alpha_B, ctx_A, ctx_B = x
    gated_A = alpha_A * (1.0 if ctx_A > SWITCH_THRESHOLD else 0.0)
    gated_B = alpha_B * (1.0 if ctx_B > SWITCH_THRESHOLD else 0.0)
    return np.array([gated_A, gated_B], dtype=float)


def _error_to_value(x: np.ndarray) -> np.ndarray:
    delta_A, delta_B, alpha_A, alpha_B = x
    return np.array([delta_A * alpha_A, delta_B * alpha_B], dtype=float)


def _build_main_network(
    params: dict,
    decoders: dict,
    input_fn: callable,
) -> nengo.Network:
    """Full 2D NEF network for one diederen session."""
    seed = int(params.get("base_seed", params["seed"]))
    tau_fb = float(params["tau_fb"])
    tau_ff = float(params["tau_ff"])
    tau_error = float(params["tau_error"])
    tau_probe = float(params["tau_probe"])
    T_error = float(params["T_error"])
    n_neurons = int(params["n_neurons"])
    n_neurons_counting = int(params["n_neurons_counting"])
    _validate_1d_decoders(decoders, n_neurons_counting)

    with nengo.Network(label=str(params.get("model_type", "NEF2d")), seed=seed) as net:
        net.input_node = nengo.Node(input_fn, size_out=4, label="input_4d")

        _build_1d_counting_circuit(net, "A", 2, params, seed, decoders)
        _build_1d_counting_circuit(net, "B", 3, params, seed + 1, decoders)

        # --- switch: decoded alphas + context indicators → gated alphas ---
        net.switch_in = nengo.Ensemble(
            4,
            4,
            neuron_type=nengo.Direct(),
            label="switch_in",
            seed=seed,
        )
        nengo.Connection(
            net.count_A.neurons,
            net.switch_in[0],
            transform=decoders["W_weight"],
            synapse=tau_probe,
            seed=seed,
        )
        nengo.Connection(
            net.count_B.neurons,
            net.switch_in[1],
            transform=decoders["W_weight"],
            synapse=tau_probe,
            seed=seed,
        )
        nengo.Connection(net.input_node[2], net.switch_in[2], synapse=None, seed=seed)
        nengo.Connection(net.input_node[3], net.switch_in[3], synapse=None, seed=seed)

        net.switch_out = nengo.Node(_switch_fn, size_in=4, size_out=2, label="switch_out")
        nengo.Connection(net.switch_in, net.switch_out, synapse=None, seed=seed)

        # --- value (2D) ---
        net.value = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=2,
            radius=float(params["radius_v"]),
            label="value_2d",
            seed=seed,
        )
        nengo.Connection(
            net.value,
            net.value,
            transform=np.eye(2),
            synapse=tau_fb,
            seed=seed,
        )

        # --- error (4D) ---
        net.error = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=4,
            radius=float(params["radius_e"]),
            label="error_4d",
            seed=seed,
        )
        nengo.Connection(
            net.input_node[0:2],
            net.error[0:2],
            transform=np.eye(2),
            synapse=tau_ff,
            seed=seed,
        )
        nengo.Connection(
            net.value,
            net.error[0:2],
            transform=-np.eye(2),
            synapse=tau_ff,
            seed=seed,
        )
        nengo.Connection(
            net.switch_out,
            net.error[2:4],
            transform=np.eye(2),
            synapse=tau_ff,
            seed=seed,
        )
        nengo.Connection(
            net.error,
            net.value,
            function=_error_to_value,
            transform=T_error,
            synapse=tau_error,
            seed=seed,
        )

        net.probe_value = nengo.Probe(
            net.value,
            synapse=tau_probe,
            sample_every=float(params["dt"]),
        )

    return net


def _readout_index(t_readout: float, dt: float, n_samples: int) -> int:
    idx = int(np.round(t_readout / dt))
    return int(np.clip(idx, 0, n_samples - 1))


def _simulate_session(
    session_df: pd.DataFrame,
    params: dict,
    decoders: dict,
) -> list[dict]:
    """Simulate one session; return one output dict per observation row."""
    rows, distrib_a, distrib_b, input_fn = _session_input_timeseries(session_df, params)
    net = _build_main_network(params, decoders, input_fn)

    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    dt = float(params["dt"])
    t_total = _session_duration(len(rows), params)

    with nengo.Simulator(
        net, dt=dt, seed=int(params["seed"]), progress_bar=False
    ) as sim:
        sim.run(t_total)

    value_traj = sim.data[net.probe_value]
    n_samples = len(value_traj)
    out_rows: list[dict] = []

    sess_sorted = session_df.sort_values("trial_in_session")
    for _, row in sess_sorted.iterrows():
        tis = int(row["trial_in_session"])
        t_readout = t_iti + (tis - 1) * t_step + t_obs * READOUT_FRACTION
        idx = _readout_index(t_readout, dt, n_samples)
        response_a = float(value_traj[idx, 0])
        response_b = float(value_traj[idx, 1])
        d = int(row["distrib_index"])
        if d == distrib_a:
            response = response_a
        elif d == distrib_b:
            response = response_b
        else:
            response = float("nan")

        out_rows.append(
            {
                "model_type": params["model_type"],
                "pid": int(params["pid"]),
                "trial": int(row["trial"]),
                "observation": int(row["observation"]),
                "response": response,
                "response_A": response_a,
                "response_B": response_b,
            }
        )

    return out_rows


def run(params: dict, save: bool = False) -> pd.DataFrame:
    """
    Run NEF2d for one diederen participant.

    Simulates each session as one continuous interleaved sequence (sorted by
    ``trial_in_session``).
    """
    nef2d_fixed = dict(MODEL_PARAMS["diederen"]["NEF2d"].get("fixed", {}))
    pfull = {**PARAM_DEFAULTS, **nef2d_fixed, **params}
    pfull.setdefault("model_type", "NEF2d")
    pfull.setdefault("dataset", "diederen")
    pfull["base_seed"] = int(pfull["seed"])

    required = ("model_type", "dataset", "pid", "alpha_0", "lambda_", "seed")
    for key in required:
        if key not in pfull:
            raise KeyError(f"params must include {key!r}")

    pid = int(pfull["pid"])
    human = pd.read_pickle(data_path("diederen.pkl"))
    human_pid = human[(human["pid"] == pid) & ~human["missed"]].copy()
    if human_pid.empty:
        raise ValueError(f"No non-missed rows for pid={pid}")

    decoders = _pretrain_counting_1d(pfull)
    all_rows: list[dict] = []

    sessions = sorted(human_pid["session"].unique())
    for sess in sessions:
        sess_df = human_pid[human_pid["session"] == sess].copy()
        try:
            _session_distrib_map(sess_df)
        except ValueError as exc:
            logging.warning("Skipping session %s pid %s: %s", sess, pid, exc)
            continue

        t0 = time.time()
        sess_seed = _trial_seed(int(pfull["seed"]), int(sess))
        p_sess = {**pfull, "seed": sess_seed}
        rows = _simulate_session(sess_df, p_sess, decoders)
        all_rows.extend(rows)
        elapsed = time.time() - t0
        print(
            f"  pid={pid} session {int(sess)}: {elapsed:.1f}s "
            f"({len(rows)} observations)",
            flush=True,
        )

    out = pd.DataFrame(all_rows)
    if save:
        run_folder = pfull.get("run_folder", "response")
        out_dir = RUNS_DIR / run_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"NEF2d_diederen_{pid}_responses.pkl"
        out.to_pickle(out_path)
        print(f"Saved {out_path}")
    return out


def _params_from_args(args: argparse.Namespace) -> dict:
    fixed = dict(MODEL_PARAMS["diederen"]["NEF2d"]["fixed"])
    return {
        **fixed,
        "model_type": "NEF2d",
        "dataset": "diederen",
        "pid": int(args.pid),
        "alpha_0": float(args.alpha_0),
        "lambda_": float(args.lambda_),
        "seed": int(args.seed),
        "run_folder": str(args.run_folder),
        "radius_c": float(args.radius_c),
        "count_leak": float(args.count_leak),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEF2d diederen interleaved integrator")
    p.add_argument("--pid", type=int, default=2097)
    p.add_argument("--alpha_0", type=float, default=0.3)
    p.add_argument("--lambda_", type=float, default=0.5)
    p.add_argument("--n_seeds", type=int, default=1, help="unused; reserved")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_folder", type=str, default="test")
    p.add_argument("--radius_c", type=float, default=PARAM_DEFAULTS["radius_c"])
    p.add_argument(
        "--count_leak",
        type=float,
        default=PARAM_DEFAULTS.get("count_leak", 0.0),
    )
    p.add_argument("--save", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    params = _params_from_args(args)
    df = run(params, save=args.save)
    print(df.head(20).to_string())
    print("\n--- per-session summary ---")
    if not df.empty:
        human = pd.read_pickle(data_path("diederen.pkl"))
        human = human[~human["missed"]].query("pid == @params['pid']")
        merged = df.merge(
            human[["trial", "observation", "session"]].drop_duplicates(),
            on=["trial", "observation"],
            how="left",
        )
        for sess, grp in merged.groupby("session", sort=True):
            print(
                f"session {int(sess)}: "
                f"response_A mean={grp['response_A'].mean():.4f}, "
                f"response_B mean={grp['response_B'].mean():.4f}, "
                f"response mean={grp['response'].mean():.4f} "
                f"(n={len(grp)})"
            )
