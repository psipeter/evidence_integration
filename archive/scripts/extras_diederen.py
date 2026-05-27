#!/usr/bin/env python3
"""
Generate supplementary NEF2d probe data for diederen.

This script runs probe-enabled NEF2d session simulations for one or more pids
and writes per-pid probe pickle files, then collects them into a combined file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nengo
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.model_params import MODEL_PARAMS
from models.NEF2d import (
    PARAM_DEFAULTS,
    _build_main_network,
    _pretrain_counting_1d,
    _session_distrib_map,
    _session_duration,
    _session_input_timeseries,
)
from utils.paths import RUNS_DIR, data_path
from utils.run_params import trial_seed as _trial_seed


def _simulate_session_probes(session_df: pd.DataFrame, params: dict, decoders: dict) -> dict:
    rows, distrib_a, distrib_b, input_fn = _session_input_timeseries(session_df, params)
    net = _build_main_network(params, decoders, input_fn)

    tau_probe = float(params["tau_probe"])
    dt = float(params["dt"])
    dt_sample = 0.01
    t_total = _session_duration(len(rows), params)
    session = int(session_df["session"].iloc[0])

    with net:
        p_input = nengo.Probe(net.input_node, synapse=tau_probe, sample_every=dt_sample)
        p_value = nengo.Probe(net.value, synapse=tau_probe, sample_every=dt_sample)
        p_count_A = nengo.Probe(net.alpha_A_node, synapse=tau_probe, sample_every=dt_sample)
        p_count_B = nengo.Probe(net.alpha_B_node, synapse=tau_probe, sample_every=dt_sample)
        p_count_A_raw = nengo.Probe(net.count_A, synapse=tau_probe, sample_every=dt_sample)
        p_count_B_raw = nengo.Probe(net.count_B, synapse=tau_probe, sample_every=dt_sample)
        p_switch = nengo.Probe(net.switch_out, synapse=tau_probe, sample_every=dt_sample)
        p_error = nengo.Probe(net.error, synapse=tau_probe, sample_every=dt_sample)

    with nengo.Simulator(net, dt=dt, seed=int(params["seed"]), progress_bar=False) as sim:
        sim.run(t_total)
        t = np.asarray(sim.trange(dt=dt_sample))
        inp = np.asarray(sim.data[p_input])
        value = np.asarray(sim.data[p_value])
        alpha_A = np.asarray(sim.data[p_count_A]).squeeze()
        alpha_B = np.asarray(sim.data[p_count_B]).squeeze()
        count_A = np.asarray(sim.data[p_count_A_raw]).squeeze()
        count_B = np.asarray(sim.data[p_count_B_raw]).squeeze()
        switch = np.asarray(sim.data[p_switch])
        error = np.asarray(sim.data[p_error])

    alpha = np.column_stack([alpha_A, alpha_B])
    count = np.column_stack([count_A, count_B])
    error_alpha = np.asarray(error)[:, 2:4]

    return {
        "pid": int(params["pid"]),
        "session": session,
        "t": t,
        "input": inp,
        "value": value,
        "count": count,
        "alpha": alpha,
        "error_alpha": error_alpha,
        "switch": switch,
        "params": dict(params),
        "distrib_a": int(distrib_a),
        "distrib_b": int(distrib_b),
        "n_obs": len(rows),
    }


def _run_probe_pids_simulate(pids: list[int], run_folder: str, out_folder: str) -> None:
    run_dir = RUNS_DIR / run_folder
    out_dir = RUNS_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    params_path = run_dir / "NEF2d_diederen_params.pkl"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params file: {params_path}")
    params_df = pd.read_pickle(params_path)

    human = pd.read_pickle(data_path("diederen.pkl"))

    for pid in pids:
        pid_row = params_df[params_df["pid"] == int(pid)]
        if pid_row.empty:
            print(f"Skipping pid={pid}: not found in {params_path.name}")
            continue

        fitted = pid_row.iloc[0].to_dict()
        fixed = MODEL_PARAMS.get("diederen", {}).get("NEF2d", {}).get("fixed", {})
        base_params = {**PARAM_DEFAULTS, **fixed, **fitted}
        base_params["model_type"] = "NEF2d"
        base_params["dataset"] = "diederen"
        base_params["pid"] = int(pid)
        base_params["base_seed"] = int(base_params.get("seed", 0))

        human_pid = human[(human["pid"] == int(pid)) & ~human["missed"]].copy()
        if human_pid.empty:
            print(f"Skipping pid={pid}: no non-missed rows")
            continue

        decoders = _pretrain_counting_1d(base_params)
        session_payloads: list[dict] = []
        sessions = sorted(human_pid["session"].unique())
        for sess in sessions:
            sess_df = human_pid[human_pid["session"] == sess].copy()
            try:
                _session_distrib_map(sess_df)
            except ValueError as exc:
                print(f"Skipping pid={pid} session={int(sess)}: {exc}")
                continue

            sess_seed = _trial_seed(int(base_params["base_seed"]), int(sess))
            p_sess = {**base_params, "seed": int(sess_seed)}
            payload = _simulate_session_probes(sess_df, p_sess, decoders)
            session_payloads.append(payload)
            print(f"  pid={pid} session={int(sess)} done")

        out_path = out_dir / f"probe_NEF2d_diederen_{int(pid)}.pkl"
        pd.to_pickle(session_payloads, out_path)
        print(f"Saved {len(session_payloads)} sessions -> {out_path}")


def _collect_probe_pids(out_dir: Path) -> None:
    probe_files = sorted(out_dir.glob("probe_NEF2d_diederen_*.pkl"))
    combined: list[dict] = []
    for path in probe_files:
        data = pd.read_pickle(path)
        combined.extend(data)
    out_path = out_dir / "probe_pids_diederen.pkl"
    pd.to_pickle(combined, out_path)
    print(f"Collected {len(probe_files)} files, {len(combined)} sessions -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="probe_pids", choices=["probe_pids"])
    parser.add_argument("--mode", choices=["run", "collect"], required=True)
    parser.add_argument("--run_folder", default="refit")
    parser.add_argument("--out_folder", default="refit")
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Single PID to simulate (cluster mode)",
    )
    args = parser.parse_args()

    out_dir = RUNS_DIR / args.out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "collect":
        _collect_probe_pids(out_dir)
        return

    run_dir = RUNS_DIR / args.run_folder
    params_path = run_dir / "NEF2d_diederen_params.pkl"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params file: {params_path}")
    params_df = pd.read_pickle(params_path)
    all_pids = sorted(int(p) for p in params_df["pid"].unique())
    pids = [int(args.pid)] if args.pid is not None else all_pids

    _run_probe_pids_simulate(pids, args.run_folder, args.out_folder)


if __name__ == "__main__":
    main()
