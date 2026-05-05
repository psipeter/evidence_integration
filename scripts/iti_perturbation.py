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

# WhiteSignal upper frequency bound (Hz); single setting for probes + noise_scan.
DEFAULT_ITI_NOISE_FREQ_HZ = 10.0


def _add_iti_noise(net, params: dict, n_obs: int) -> None:
    """Add gated white noise to value ensemble during ITI periods."""
    import nengo
    import nengo.processes

    amplitude = float(params.get("iti_noise_amplitude", 0.0))
    freq = float(params.get("iti_noise_freq", DEFAULT_ITI_NOISE_FREQ_HZ))
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
            """0 during first ITI and all t_obs periods; 1 during ITIs after obs 1."""
            if _t < t_iti:
                return 0.0
            t_in_step = (_t - t_iti) % t_step
            return 1.0 if t_in_step >= t_obs else 0.0

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
        net.probe_iti_noise = nengo.Probe(gated_node, synapse=None)


def _nef_run_with_iti_noise(
    params: dict, *, save_probes: bool = False
) -> pd.DataFrame:
    """Run NEF; ITI noise is injected inside ``models.NEF._simulate_trial``."""
    from models.NEF import run as nef_run

    return nef_run(params, save_probes=save_probes)


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
    freq = DEFAULT_ITI_NOISE_FREQ_HZ
    for pid in pids:
        base = _load_base_params(pid, run_folder)
        for amp in args.noise_amplitudes:
            out_path = out_dir / f"noise_scan_{pid}_amp{amp}.pkl"
            if out_path.exists():
                print(f"  Skipping pid={pid} amp={amp} (exists)")
                continue
            p = {**base, "iti_noise_amplitude": amp}
            if amp > 0.0:
                p["iti_noise_freq"] = freq
            responses = _nef_run_with_iti_noise(p)
            responses["iti_noise_amplitude"] = amp
            responses["iti_noise_freq"] = freq if amp > 0.0 else 0.0
            responses.to_pickle(out_path)
            print(f"  pid={pid} amp={amp} (@ {freq:g} Hz): saved")


# Panel-1 probes (``plot_iti_perturbation.py``): basename -> ITI noise RMS amplitude (fitted t_iti)
PROBE_PANEL1_CONDITIONS: tuple[tuple[str, float], ...] = (
    ("no_noise", 0.0),
    ("amp0p1", 0.1),
    ("amp0p05", 0.05),
)


def _run_probe_conditions(
    pid: int, args: argparse.Namespace, run_folder: Path, out_dir: Path
) -> None:
    """Save probe data for ``plot_iti_perturbation`` panel 1 (three noise amplitudes, default ITI)."""
    base = _load_base_params(pid, run_folder)
    freq = DEFAULT_ITI_NOISE_FREQ_HZ

    for label, amp in PROBE_PANEL1_CONDITIONS:
        out_path = out_dir / f"probe_panel1_{pid}_{label}.pkl"
        if out_path.exists():
            print(f"  Skipping probe {label} (exists)")
            continue
        p = {**base, "iti_noise_amplitude": amp}
        if amp > 0.0:
            p["iti_noise_freq"] = freq
        print(f"  Running probe {label} (amp={amp}, t_iti={float(base['t_iti'])})...")
        _nef_run_with_iti_noise(p, save_probes=True)
        src = data_path(f"probe_NEF_recurrent_carrabin_{pid}.pkl")
        dst = out_path
        if src.exists():
            src.rename(dst)
            print(f"  Saved {dst.name}")


def _analyze(args: argparse.Namespace, out_dir: Path, human: pd.DataFrame, qid_map: pd.DataFrame) -> None:
    if args.experiment == "probe_conditions":
        for label, _ in PROBE_PANEL1_CONDITIONS:
            path = out_dir / f"probe_panel1_{args.pid}_{label}.pkl"
            print(f"  {label}: {'exists' if path.exists() else 'missing'}")
        return

    rows = []
    for f in sorted(out_dir.glob("noise_scan_*.pkl")):
        resp = pd.read_pickle(f)
        metrics = _compute_metrics(resp, qid_map)
        row = {"pid": int(resp["pid"].iloc[0]), **metrics}
        for col in ("iti_noise_amplitude", "t_iti"):
            if col in resp.columns:
                row[col] = float(resp[col].iloc[0])
        rows.append(row)
    if not rows:
        print("No data found.")
        return
    df = pd.DataFrame(rows)
    print("\nnoise_scan results (mean over pids):")
    print(
        df.groupby("iti_noise_amplitude")[["response_noise", "rmse"]]
        .mean()
        .round(4)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="probe_conditions",
        choices=["probe_conditions", "noise_scan"],
        help=(
            "probe_conditions: save probe pickles for panel 1 (three noise amplitudes); "
            "noise_scan: sweep noise amplitude at 10 Hz for panel 2"
        ),
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
    args = parser.parse_args()

    run_folder = RUNS_DIR / args.run_folder
    out_dir = RUNS_DIR / args.out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    human = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[
        ["pid", "trial", "observation", "qid", "value"]
    ].drop_duplicates()

    if args.run_simulation:
        if args.experiment == "probe_conditions":
            _run_probe_conditions(args.pid, args, run_folder, out_dir)
        elif args.experiment == "noise_scan":
            pids = list(range(1, 22)) if args.run_all_pids else [args.pid]
            _run_noise_scan(pids, args, run_folder, out_dir)

    _analyze(args, out_dir, human, qid_map)


if __name__ == "__main__":
    main()
