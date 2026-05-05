#!/usr/bin/env python3
"""ITI perturbation experiments: white-noise injected into value ensemble during ITI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import RUNS_DIR, data_path


def _add_iti_noise(net, params: dict, n_obs: int) -> None:
    """Add gated white noise to value ensemble during ITI periods."""
    import nengo
    import nengo.processes

    amplitude = float(params.get("iti_noise_amplitude", 0.0))
    freq = float(params.get("iti_noise_freq", 10.0))
    if amplitude == 0.0:
        return
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti

    noise_process = nengo.processes.WhiteSignal(
        period=float(n_obs * t_step),
        high=freq,
        rms=amplitude,
    )
    with net:
        noise_node = nengo.Node(noise_process, size_out=1)

        def _iti_gate(_t):
            return 1.0 if (_t % t_step) < t_iti else 0.0

        gate_node = nengo.Node(_iti_gate)

        def _gated(_t, x):
            return x[0] * x[1]

        gated_node = nengo.Node(_gated, size_in=2)
        nengo.Connection(noise_node, gated_node[0], synapse=None)
        nengo.Connection(gate_node, gated_node[1], synapse=None)
        nengo.Connection(
            gated_node,
            net.value,
            transform=np.ones((1, 1)),
            synapse=float(params.get("tau_ff", 0.01)),
        )


def _nef_run_with_iti_noise(params: dict) -> pd.DataFrame:
    """Run NEF with optional ITI noise (no edits to ``models/NEF.py``)."""
    import models.NEF as nef_module

    build_orig = nef_module.build_network

    def build_wrapped(obs_values: np.ndarray, p: dict, decoders: dict):
        net = build_orig(obs_values, p, decoders)
        _add_iti_noise(net, p, len(obs_values))
        return net

    nef_module.build_network = build_wrapped
    try:
        return nef_module.run(params)
    finally:
        nef_module.build_network = build_orig


def _load_base_params(pid: int, run_folder: Path) -> dict:
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS

    params = pd.read_pickle(
        run_folder / f"NEF_recurrent_carrabin_{pid}_params.pkl"
    ).iloc[0].to_dict()
    fixed = MODEL_PARAMS["carrabin"]["NEF_recurrent"].get("fixed", {})
    params = {**PARAM_DEFAULTS, **fixed, **params}
    params.update(
        {
            "nef_type": "recurrent",
            "dataset": "carrabin",
            "model_type": "NEF_recurrent",
        }
    )
    return params


def _compute_metrics(resp: pd.DataFrame, qid_map: pd.DataFrame) -> dict:
    """Compute response_noise and rmse for a responses DataFrame."""
    from fitting.losses import _mean_qid_std

    grp = resp.merge(qid_map, on=["pid", "trial", "observation"], how="left")
    noise = _mean_qid_std(grp)
    qid_mean = qid_map.groupby("qid")["value"].mean()
    grp = grp.assign(true_mean=grp["qid"].map(qid_mean))
    rmse = float(np.sqrt(((grp["response"] - grp["true_mean"]) ** 2).mean()))
    return {"response_noise": noise, "rmse": rmse}


def _run_noise_scan(
    pids: list[int], args: argparse.Namespace, run_folder: Path, out_dir: Path
) -> None:
    for pid in pids:
        base = _load_base_params(pid, run_folder)
        for amp in args.noise_amplitudes:
            freqs = [0.0] if amp == 0.0 else args.noise_freqs
            for freq in freqs:
                out_path = (
                    out_dir / f"noise_scan_{pid}_amp{amp}_freq{freq}.pkl"
                )
                if out_path.exists():
                    print(
                        f"  Skipping pid={pid} amp={amp} freq={freq} (exists)"
                    )
                    continue
                p = {**base, "iti_noise_amplitude": amp, "iti_noise_freq": freq}
                responses = _nef_run_with_iti_noise(p)
                responses["iti_noise_amplitude"] = amp
                responses["iti_noise_freq"] = freq
                responses.to_pickle(out_path)
                print(f"  pid={pid} amp={amp} freq={freq}: saved")


def _run_iti_scan(
    pids: list[int], args: argparse.Namespace, run_folder: Path, out_dir: Path
) -> None:
    for pid in pids:
        base = _load_base_params(pid, run_folder)
        for t_iti in args.t_iti_values:
            out_path = out_dir / f"iti_scan_{pid}_titi{t_iti}.pkl"
            if out_path.exists():
                print(f"  Skipping pid={pid} t_iti={t_iti} (exists)")
                continue
            p = {
                **base,
                "t_iti": t_iti,
                "iti_noise_amplitude": args.fixed_noise_amplitude,
                "iti_noise_freq": args.fixed_noise_freq,
            }
            responses = _nef_run_with_iti_noise(p)
            responses["t_iti"] = t_iti
            responses["iti_noise_amplitude"] = args.fixed_noise_amplitude
            responses["iti_noise_freq"] = args.fixed_noise_freq
            responses.to_pickle(out_path)
            print(f"  pid={pid} t_iti={t_iti}: saved")


def _analyze(args: argparse.Namespace, out_dir: Path, human: pd.DataFrame, qid_map: pd.DataFrame) -> None:
    rows = []
    pattern = "noise_scan" if args.sub_experiment == "noise_scan" else "iti_scan"
    for f in sorted(out_dir.glob(f"{pattern}_*.pkl")):
        resp = pd.read_pickle(f)
        metrics = _compute_metrics(resp, qid_map)
        row = {"pid": int(resp["pid"].iloc[0]), **metrics}
        for col in ("iti_noise_amplitude", "iti_noise_freq", "t_iti"):
            if col in resp.columns:
                row[col] = float(resp[col].iloc[0])
        rows.append(row)
    if not rows:
        print("No data found.")
        return
    df = pd.DataFrame(rows)
    if args.sub_experiment == "noise_scan":
        print("\nnoise_scan results (mean over pids):")
        print(
            df.groupby(["iti_noise_amplitude", "iti_noise_freq"])[
                ["response_noise", "rmse"]
            ]
            .mean()
            .round(4)
        )
    else:
        print("\niti_scan results (mean over pids):")
        print(
            df.groupby("t_iti")[["response_noise", "rmse"]].mean().round(4)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub_experiment",
        type=str,
        default="noise_scan",
        choices=["noise_scan", "iti_scan"],
    )
    parser.add_argument("--run_simulation", action="store_true", default=False)
    parser.add_argument(
        "--run_folder",
        type=str,
        default="response",
        help="Source folder for fitted NEF params",
    )
    parser.add_argument("--out_folder", type=str, default="iti_perturbation")
    parser.add_argument(
        "--pid", type=int, default=14, help="Single pid to simulate"
    )
    parser.add_argument("--run_all_pids", action="store_true", default=False)

    parser.add_argument(
        "--noise_amplitudes",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.5],
    )
    parser.add_argument(
        "--noise_freqs",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 50.0],
    )
    parser.add_argument("--fixed_noise_freq", type=float, default=10.0)

    parser.add_argument(
        "--t_iti_values",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    )
    parser.add_argument(
        "--fixed_noise_amplitude", type=float, default=0.05
    )
    args = parser.parse_args()

    run_folder = RUNS_DIR / args.run_folder
    out_dir = RUNS_DIR / args.out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    human = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[
        ["pid", "trial", "observation", "qid", "value"]
    ].drop_duplicates()

    pids = list(range(1, 22)) if args.run_all_pids else [args.pid]

    if args.run_simulation:
        if args.sub_experiment == "noise_scan":
            _run_noise_scan(pids, args, run_folder, out_dir)
        else:
            _run_iti_scan(pids, args, run_folder, out_dir)

    _analyze(args, out_dir, human, qid_map)


if __name__ == "__main__":
    main()
