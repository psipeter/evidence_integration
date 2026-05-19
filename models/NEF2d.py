#!/usr/bin/env python3
"""
2D NEF evidence integrator for the Diederen interleaved task.

Tracks two simultaneous reward distributions (dimensions 0 and 1) with a
shared 2D counting integrator and context-gated learning rates. One simulation
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
from utils.paths import RUNS_DIR, data_path
from utils.run_params import trial_seed as _trial_seed

READOUT_FRACTION = 0.8
SWITCH_THRESHOLD = 0.5

_NEF2D_EXTRA: dict[str, object] = {
    "n_train_trials": 10,
    "radius_c": 60.0,
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


def _validate_count_decoders(decoders: dict, n_neurons_counting: int) -> None:
    """Decoder weights must match ``count.neurons`` (not onset_detector)."""
    expected = (1, n_neurons_counting)
    for key in ("W_count_A", "W_count_B", "W_alpha_A", "W_alpha_B"):
        w = decoders[key]
        if tuple(w.shape) != expected:
            raise ValueError(
                f"Decoder {key!r} shape {w.shape} != {expected} "
                f"(count uses n_neurons_counting={n_neurons_counting})"
            )


def _add_onset_detector_2d(
    net: nengo.Network,
    input_node: nengo.Node,
    params: dict,
    seed: int,
) -> None:
    """
    Two independent 1D onset detectors, one per distribution.
    net.onset_A fires on rising edges of input_node[2] (context A).
    net.onset_B fires on rising edges of input_node[3] (context B).
    """
    n_neurons = int(params["n_neurons"])
    tau_fast = float(params["tau_fast"])
    tau_slow = float(params["tau_slow"])

    for name, dim in [("onset_A", 2), ("onset_B", 3)]:
        ens = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=1,
            radius=1.5,
            encoders=nengo.dists.Choice([[1]]),
            intercepts=nengo.dists.Uniform(0.0, 1.0),
            label=name,
            seed=seed,
        )
        setattr(net, name, ens)
        nengo.Connection(
            input_node[dim],
            ens,
            synapse=tau_fast,
            seed=seed,
        )
        nengo.Connection(
            input_node[dim],
            ens,
            synapse=tau_slow,
            function=lambda x: -x,
            seed=seed,
        )


def _build_counting_network_2d(
    params: dict,
    train: bool,
    input_fn: callable,
    decoders: dict | None = None,
) -> nengo.Network:
    """2D counting subnetwork (onset detector + shared 2D integrator)."""
    seed = int(params["seed"])
    tau_fb = float(params["tau_fb"])
    tau_probe = float(params["tau_probe"])
    amp = float(params["onset_detector_amp"])
    n_neurons_counting = int(params["n_neurons_counting"])
    radius_count = float(params["radius_c"])

    with nengo.Network(label="counting_2d", seed=seed) as net:
        net.input_node = nengo.Node(input_fn, size_out=4, label="input_4d")

        _add_onset_detector_2d(net, net.input_node, params, seed)

        net.count = nengo.Ensemble(
            n_neurons=n_neurons_counting,
            dimensions=2,
            radius=float(radius_count),
            label="count_2d",
            seed=seed,
        )
        nengo.Connection(
            net.onset_A,
            net.count[0],
            synapse=tau_fb,
            function=lambda x, amp=amp: [amp] if x > 0 else [0.0],
            seed=seed,
        )
        nengo.Connection(
            net.onset_B,
            net.count[1],
            synapse=tau_fb,
            function=lambda x, amp=amp: [amp] if x > 0 else [0.0],
            seed=seed,
        )
        nengo.Connection(
            net.count,
            net.count,
            transform=np.eye(2),
            synapse=tau_fb,
            seed=seed,
        )

        # Pretrain activities: count.neurons (n_neurons_counting), not onset_detector.
        net.probe_count_neurons = nengo.Probe(
            net.count.neurons,
            synapse=None,
            sample_every=float(params["dt"]),
        )

        if not train:
            if decoders is None:
                raise ValueError("decoders required when train=False")
            _validate_count_decoders(decoders, n_neurons_counting)
            net.alpha_a_node = nengo.Ensemble(
                1, 1, neuron_type=nengo.Direct(), label="alpha_a_node", seed=seed
            )
            net.alpha_b_node = nengo.Ensemble(
                1, 1, neuron_type=nengo.Direct(), label="alpha_b_node", seed=seed
            )
            nengo.Connection(
                net.count.neurons,
                net.alpha_a_node,
                transform=decoders["W_alpha_A"],
                synapse=tau_probe,
                seed=seed,
            )
            nengo.Connection(
                net.count.neurons,
                net.alpha_b_node,
                transform=decoders["W_alpha_B"],
                synapse=tau_probe,
                seed=seed,
            )
            net.probe_alpha_a = nengo.Probe(net.alpha_a_node, synapse=None)
            net.probe_alpha_b = nengo.Probe(net.alpha_b_node, synapse=None)

    return net


def _decode_counting_2d(activities: np.ndarray, targets: np.ndarray) -> dict:
    """Solve joint decoders for 2D count and alpha streams."""
    solver = nengo.solvers.LstsqL2(reg=1e-3)
    W_count_A, _ = solver(activities, targets[:, 0:1])
    W_count_B, _ = solver(activities, targets[:, 1:2])
    W_alpha_A, _ = solver(activities, targets[:, 2:3])
    W_alpha_B, _ = solver(activities, targets[:, 3:4])
    return {
        "W_count_A": W_count_A.T,
        "W_count_B": W_count_B.T,
        "W_alpha_A": W_alpha_A.T,
        "W_alpha_B": W_alpha_B.T,
    }


def _simulate_counting_session(
    session_rows: list[dict],
    distrib_a: int,
    distrib_b: int,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Run counting-only network on one session; return activities and targets."""
    input_fn = _make_session_input(session_rows, distrib_a, distrib_b, params)
    net = _build_counting_network_2d(params, train=True, input_fn=input_fn)
    t_total = _session_duration(len(session_rows), params)
    dt = float(params["dt"])

    with nengo.Simulator(
        net, dt=dt, seed=int(params["seed"]), progress_bar=False
    ) as sim:
        sim.run(t_total)

    activities = sim.data[net.probe_count_neurons]
    n_steps = len(activities)
    t_arr = np.arange(n_steps) * dt
    targets = _ideal_targets_at_times(t_arr, session_rows, distrib_a, distrib_b, params)
    return activities, targets


def _pretrain_counting_2d(params: dict, human: pd.DataFrame | None = None) -> dict:
    """
    Sample real interleaved sessions and train 2D counting decoders.

    Parameters
    ----------
    human
        Optional pre-loaded diederen dataframe (missed rows already removed).
    """
    if human is None:
        human = pd.read_pickle(data_path("diederen.pkl"))
        human = human[~human["missed"]].copy()

    sessions: list[tuple[int, int]] = []
    for (pid, session), _ in human.groupby(["pid", "session"], sort=False):
        sess = human[(human["pid"] == pid) & (human["session"] == session)]
        try:
            _session_distrib_map(sess)
        except ValueError:
            continue
        sessions.append((int(pid), int(session)))

    if not sessions:
        raise ValueError("No valid (pid, session) pairs for 2D counting pretrain")

    n_train = int(params.get("n_train_trials", 10))
    rng = np.random.default_rng(int(params.get("base_seed", params["seed"])))
    n_pick = min(n_train, len(sessions))
    chosen = [sessions[i] for i in rng.choice(len(sessions), size=n_pick, replace=False)]

    act_pieces: list[np.ndarray] = []
    tgt_pieces: list[np.ndarray] = []

    for pid, session in chosen:
        sess = human[(human["pid"] == pid) & (human["session"] == session)].copy()
        rows, distrib_a, distrib_b, _ = _session_input_timeseries(sess, params)
        activities, targets = _simulate_counting_session(
            rows, distrib_a, distrib_b, params
        )
        act_pieces.append(activities)
        tgt_pieces.append(targets)

    activities_all = np.vstack(act_pieces)
    targets_all = np.vstack(tgt_pieces)
    return _decode_counting_2d(activities_all, targets_all)


def _switch_fn(x: np.ndarray) -> np.ndarray:
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
    seed = int(params["seed"])
    tau_fb = float(params["tau_fb"])
    tau_ff = float(params["tau_ff"])
    tau_error = float(params["tau_error"])
    tau_probe = float(params["tau_probe"])
    T_error = float(params["T_error"])
    n_neurons = int(params["n_neurons"])
    n_neurons_counting = int(params["n_neurons_counting"])
    radius_count = float(params["radius_c"])
    amp = float(params["onset_detector_amp"])
    _validate_count_decoders(decoders, n_neurons_counting)

    with nengo.Network(label=str(params.get("model_type", "NEF2d")), seed=seed) as net:
        net.input_node = nengo.Node(input_fn, size_out=4, label="input_4d")

        # --- 2D counting (shared with pretrain architecture) ---
        _add_onset_detector_2d(net, net.input_node, params, seed)

        net.count = nengo.Ensemble(
            n_neurons=n_neurons_counting,
            dimensions=2,
            radius=float(radius_count),
            label="count_2d",
            seed=seed,
        )
        nengo.Connection(
            net.onset_A,
            net.count[0],
            synapse=tau_fb,
            function=lambda x, amp=amp: [amp] if x > 0 else [0.0],
            seed=seed,
        )
        nengo.Connection(
            net.onset_B,
            net.count[1],
            synapse=tau_fb,
            function=lambda x, amp=amp: [amp] if x > 0 else [0.0],
            seed=seed,
        )
        nengo.Connection(
            net.count,
            net.count,
            transform=np.eye(2),
            synapse=tau_fb,
            seed=seed,
        )

        # --- switch: decoded alphas + context indicators → gated alphas ---
        net.switch_in = nengo.Ensemble(
            4,
            4,
            neuron_type=nengo.Direct(),
            label="switch_in",
            seed=seed,
        )
        nengo.Connection(
            net.count.neurons,
            net.switch_in[0],
            transform=decoders["W_alpha_A"],
            synapse=tau_probe,
            seed=seed,
        )
        nengo.Connection(
            net.count.neurons,
            net.switch_in[1],
            transform=decoders["W_alpha_B"],
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

    decoders = _pretrain_counting_2d(pfull, human=human)
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
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEF2d diederen interleaved integrator")
    p.add_argument("--pid", type=int, default=2097)
    p.add_argument("--alpha_0", type=float, default=0.3)
    p.add_argument("--lambda_", type=float, default=0.5)
    p.add_argument("--n_seeds", type=int, default=1, help="unused; reserved")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_folder", type=str, default="response")
    p.add_argument("--radius_c", type=float, default=PARAM_DEFAULTS["radius_c"])
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
