#!/usr/bin/env python3
"""
Supplementary NEF simulations for yoo task figure panels.

=============================================================================
EXPERIMENT: noise
    Multi-seed NEF simulations to estimate response variability.
    Output: NEF_yoo_all_responses.pkl in run_folder.

  Run (cluster, one job per pid×seed):
      bash jobs/submit_yoo_noise.sh

  Collect:
      python scripts/extras_yoo.py --experiment noise --mode collect \\
          --run_folder nef200 --n_seeds 10

=============================================================================
EXPERIMENT: lambda0
    Ablation: simulate all trials for all pids using fitted alpha_0 but
    lambda_=0 (no temporal discounting). Saves responses and error-ensemble
    activities in the same format as the main fitting pipeline, so
    figure_yoo_neural.py can use them directly via --nef_folder yoo_lambda0.

    This tests whether the activity↔delta relationship in panel B depends
    on the lambda discounting mechanism: if the relationship disappears
    with lambda=0, it is driven by temporal discounting; if it persists,
    spiking dynamics alone are sufficient.

  Run one pid locally:
      venv/bin/python scripts/extras_yoo.py --experiment lambda0 \\
          --mode run --pid 1 --source_folder refit --run_folder yoo_lambda0

  Run all pids (loop):
      for pid in $(venv/bin/python -c "
          import pandas as pd
          from utils.paths import data_path
          print(' '.join(str(p) for p in
              sorted(pd.read_pickle(data_path('yoo.pkl'))['pid'].unique())))"); do
        venv/bin/python scripts/extras_yoo.py --experiment lambda0 \\
            --mode run --pid $pid --source_folder refit --run_folder yoo_lambda0
      done

  Collect (combine per-pid files, copy encoders):
      venv/bin/python scripts/extras_yoo.py --experiment lambda0 \\
          --mode collect --run_folder yoo_lambda0 --source_folder refit

  Output files in data/runs/yoo_lambda0/:
      NEF_yoo_lambda0_{pid}_responses.pkl  — per-pid responses
      NEF_yoo_lambda0_responses.pkl        — combined responses
      activities_error_yoo.pkl             — error ensemble activities
      encoders_error_yoo.pkl               — copied from source_folder
      (encoders are tuning-curve based; unchanged by lambda=0)
=============================================================================
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.model_params import MODEL_PARAMS
from models import NEF
from utils.paths import RUNS_DIR, data_path

MODEL_TYPE = "NEF"
DATASET    = "yoo"
RESP_RE    = re.compile(rf"^{MODEL_TYPE}_{DATASET}_(\d+)_seed(\d+)_responses\.pkl$")
L0_RE      = re.compile(rf"^{MODEL_TYPE}_{DATASET}_lambda0_(\d+)_responses\.pkl$")


def _yoo_pids() -> list[int]:
    return sorted(int(p) for p in pd.read_pickle(data_path(f"{DATASET}.pkl"))["pid"].unique())


def _source_params(source_folder: str, pid: int) -> tuple[float, float, int]:
    """Load fitted lambda_, alpha_0, and base_seed for a pid.

    Reads from the combined params file (NEF_yoo_params.pkl) since
    per-pid param files may not exist.
    """
    combined = RUNS_DIR / source_folder / f"{MODEL_TYPE}_{DATASET}_params.pkl"
    if not combined.exists():
        raise FileNotFoundError(f"Missing combined params: {combined}")
    df  = pd.read_pickle(combined)
    row = df[df["pid"] == int(pid)]
    if row.empty:
        raise KeyError(f"pid {pid} not found in {combined}")
    row = row.iloc[0]
    for key in ("lambda_", "alpha_0"):
        if key not in row:
            raise KeyError(f"Expected {key!r} in {combined}")
    base_seed = int(row["seed"]) if "seed" in row and pd.notna(row["seed"]) else 0
    return float(row["lambda_"]), float(row["alpha_0"]), base_seed


def _total_seed(pid: int, seed_index: int, base_seed: int) -> int:
    return int(pid) * 1_000_000 + int(seed_index) * 1_000 + (int(base_seed) % 1000)


# ── Noise experiment ──────────────────────────────────────────────────────────

def _run_one_noise(pid: int, seed_index: int, run_folder: str, source_folder: str) -> Path:
    lambda_, alpha_0, base_seed = _source_params(source_folder, pid)
    seed   = _total_seed(pid, seed_index, base_seed)
    fixed  = dict(MODEL_PARAMS[DATASET][MODEL_TYPE].get("fixed", {}))
    params = {
        "model_type": MODEL_TYPE,
        "dataset":    DATASET,
        "pid":        int(pid),
        **fixed,
        "lambda_":    float(lambda_),
        "alpha_0":    float(alpha_0),
        "seed":       int(seed),
    }
    df = NEF.run(params, save=False)
    df = df[["model_type", "pid", "trial", "observation", "response"]].copy()
    df["seed"] = int(seed)
    df = df[["model_type", "pid", "seed", "trial", "observation", "response"]]

    out_dir  = RUNS_DIR / run_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{MODEL_TYPE}_{DATASET}_{pid}_seed{seed_index}_responses.pkl"
    df.to_pickle(out_path)
    print(f"Saved {out_path}")
    print("JOB_COMPLETE")
    return out_path


def _run_noise_mode(args: argparse.Namespace) -> None:
    if args.pid is None or args.seed_index is None:
        raise ValueError("--experiment noise --mode run requires --pid and --seed_index")
    _run_one_noise(args.pid, args.seed_index, args.run_folder, args.source_folder)


def _collect_noise_mode(args: argparse.Namespace) -> None:
    run_dir = RUNS_DIR / args.run_folder
    run_dir.mkdir(parents=True, exist_ok=True)
    files   = sorted(run_dir.glob(f"{MODEL_TYPE}_{DATASET}_*_seed*_responses.pkl"))
    if not files:
        print(f"No response files found in {run_dir}")
        return

    df  = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    out = run_dir / f"{MODEL_TYPE}_{DATASET}_all_responses.pkl"
    df.to_pickle(out)

    expected = {(int(pid), int(si)) for pid in _yoo_pids()
                for si in range(args.n_seeds)}
    found: set[tuple[int, int]] = set()
    for f in files:
        m = RESP_RE.match(f.name)
        if m:
            found.add((int(m.group(1)), int(m.group(2))))
    missing = sorted(expected - found)
    if missing:
        print(f"Warning: missing {len(missing)} pid×seed files")
        for pid, si in missing[:20]:
            print(f"  pid={pid} seed_index={si}")

    print(f"Collected {len(files)} → {out}  shape={df.shape}")


# ── Lambda=0 ablation ─────────────────────────────────────────────────────────

def _run_one_simulate(pid: int, run_folder: str, source_folder: str,
                      set_lambda: float | None = None) -> None:
    """Simulate one pid saving responses and activities in a single pass.

    Runs each trial exactly once through Nengo, capturing both responses
    (via probe_value) and error-neuron activities (via probe_error_neurons).
    This avoids the double-pass that would occur if NEF.run and
    simulate_and_save were called separately.

    set_lambda: if given, override lambda_ with this value (e.g. 0.0 for
                ablation). If None, use the fitted lambda_ from source_folder.
    """
    import time
    import nengo
    import numpy as np
    from models.NEF import (PARAM_DEFAULTS, _pretrain, build_network,
                             load_counting_activities, fast_decode_counting)
    from utils.carrabin_transform import apply_carrabin_transform

    lambda_fitted, alpha_0, _ = _source_params(source_folder, pid)
    lambda_use = set_lambda if set_lambda is not None else lambda_fitted
    fixed = dict(MODEL_PARAMS[DATASET][MODEL_TYPE].get("fixed", {}))

    params = {
        **PARAM_DEFAULTS,
        "model_type": MODEL_TYPE,
        "dataset":    DATASET,
        "pid":        int(pid),
        **fixed,
        "lambda_":    float(lambda_use),
        "alpha_0":    float(alpha_0),
        "seed":       int(pid),
    }

    out_dir = RUNS_DIR / run_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix  = f"lambda{set_lambda:.0f}_" if set_lambda is not None else ""

    yoo      = pd.read_pickle(data_path(DATASET + ".pkl"))
    pid_data = yoo[yoo["pid"] == pid].sort_values(["trial", "observation"])
    trials   = sorted(pid_data["trial"].unique())

    # Load precomputed counting activities for fast weight decoding
    _activity_map = None
    try:
        _activity_map = load_counting_activities(
            n_neurons          = int(params["n_neurons"]),
            n_neurons_counting = int(params["n_neurons_counting"]),
            dataset            = DATASET,
        )
    except FileNotFoundError:
        pass  # will fall back to _pretrain inside the loop

    t_obs      = float(params["t_obs"])
    t_iti      = float(params["t_iti"])
    t_step     = t_obs + t_iti
    dt         = float(params["dt"])
    tau_probe  = float(params["tau_probe"])
    READOUT_OFFSET = 0.5

    resp_rows    = []
    act_rows     = []
    encoders_enc = None   # captured once from first trial

    for trial in trials:
        t0         = time.time()
        trial_data = pid_data[pid_data["trial"] == trial].sort_values("observation")
        obs_values = trial_data["value"].to_numpy(dtype=float)
        n_obs      = len(obs_values)
        t_total    = n_obs * t_step

        p = {**params, "seed": int(trial)}

        # Decode counting weights (fast path if activity map available)
        if _activity_map is not None:
            activity = _activity_map.get(int(trial))
            if activity is not None:
                decoders = fast_decode_counting(
                    activity,
                    alpha_0  = float(params["alpha_0"]),
                    lambda_  = float(params["lambda_"]),
                )
            else:
                decoders = _pretrain({**p, "base_seed": int(trial)})
        else:
            decoders = _pretrain({**p, "base_seed": int(trial)})

        # Build network and add probes
        net = build_network(obs_values, p, decoders)
        with net:
            probe_value   = nengo.Probe(net.value,         synapse=tau_probe)
            probe_neurons = nengo.Probe(net.error.neurons, synapse=tau_probe)

        # Single Nengo run
        with nengo.Simulator(net, dt=dt, seed=int(trial), progress_bar=False) as sim:
            sim.run(t_total)

        value_decoded = sim.data[probe_value].squeeze()
        neuron_data   = sim.data[probe_neurons]

        # Extract response and activity at once-per-obs readout times
        for n_idx, (_, row) in enumerate(trial_data.iterrows()):
            t_readout = t_iti + n_idx * t_step + READOUT_OFFSET
            idx = int(np.clip(np.round(t_readout / dt), 0, len(value_decoded) - 1))

            # Response
            resp_rows.append({
                "model_type":  MODEL_TYPE,
                "pid":         int(pid),
                "trial":       int(trial),
                "observation": int(row["observation"]),
                "response":    float(value_decoded[idx]),
            })

            # Activity
            act_row = {
                "pid":         int(pid),
                "trial":       int(trial),
                "observation": int(row["observation"]),
            }
            for j, val in enumerate(neuron_data[idx]):
                act_row[f"n{j}"] = float(val)
            act_rows.append(act_row)

        # Capture encoders once
        if encoders_enc is None:
            encoders_enc = np.array(sim.data[net.error].encoders, copy=True)

        print(f"  pid={pid} trial {int(trial)}: {time.time()-t0:.1f}s", flush=True)

    # Save responses
    resp_df   = apply_carrabin_transform(pd.DataFrame(resp_rows), DATASET)
    resp_path = out_dir / f"{MODEL_TYPE}_{DATASET}_{prefix}{pid}_responses.pkl"
    resp_df.to_pickle(resp_path)
    print(f"Saved responses: {resp_path}")

    # Save activities
    act_path = out_dir / f"activities_error_{DATASET}_{prefix}{pid}.pkl"
    pd.DataFrame(act_rows).to_pickle(act_path)
    print(f"Saved activities: {act_path}")

    # Save encoders
    enc_rows = []
    for neuron_idx in range(encoders_enc.shape[0]):
        row = {"pid": int(pid), "ensemble": "error", "neuron_idx": int(neuron_idx)}
        for d in range(encoders_enc.shape[1]):
            row[f"enc_dim_{d}"] = float(encoders_enc[neuron_idx, d])
        enc_rows.append(row)
    enc_path = out_dir / f"encoders_error_{DATASET}_{prefix}{pid}.pkl"
    pd.DataFrame(enc_rows).to_pickle(enc_path)
    print(f"Saved encoders: {enc_path}")

    print("JOB_COMPLETE")


def _collect_lambda0_mode(args: argparse.Namespace) -> None:
    """Combine per-pid files into per-configuration combined files.

    Detects two configurations by filename prefix:
      - "lambda0_" prefix  → lambda=0 ablation
      - no prefix (plain)  → fitted lambda

    For each configuration found, produces:
      NEF_yoo_{config}_responses.pkl       — combined responses
      activities_error_yoo_{config}.pkl    — combined activities
      encoders_error_yoo.pkl               — shared (tuning curves unchanged)
      NEF_yoo_params.pkl                   — copied from source

    Also writes NEF_yoo_responses.pkl / activities_error_yoo.pkl without a
    config suffix for each configuration into its own subfolder, so that
    --nef_folder pointing at a subfolder works directly with figure scripts.
    """
    run_dir = RUNS_DIR / args.run_folder
    src_dir = RUNS_DIR / args.source_folder
    run_dir.mkdir(parents=True, exist_ok=True)

    # Regex patterns for each configuration
    configs = {
        "lambda0": re.compile(
            rf"^{MODEL_TYPE}_{DATASET}_lambda0_(\d+)_responses\.pkl$"),
        "fitted":  re.compile(
            rf"^{MODEL_TYPE}_{DATASET}_(\d+)_responses\.pkl$"),
    }
    act_configs = {
        "lambda0": re.compile(
            rf"^activities_error_{DATASET}_lambda0_(\d+)\.pkl$"),
        "fitted":  re.compile(
            rf"^activities_error_{DATASET}_(\d+)\.pkl$"),
    }
    enc_configs = {
        "lambda0": re.compile(
            rf"^encoders_error_{DATASET}_lambda0_(\d+)\.pkl$"),
        "fitted":  re.compile(
            rf"^encoders_error_{DATASET}_(\d+)\.pkl$"),
    }

    all_resp = sorted(run_dir.glob(f"{MODEL_TYPE}_{DATASET}_*_responses.pkl"))
    all_act  = sorted(run_dir.glob(f"activities_error_{DATASET}_*.pkl"))
    all_enc  = sorted(run_dir.glob(f"encoders_error_{DATASET}_*.pkl"))

    for config, resp_re in configs.items():
        resp_files = [f for f in all_resp if resp_re.match(f.name)]
        act_files  = [f for f in all_act  if act_configs[config].match(f.name)]
        enc_files  = [f for f in all_enc  if enc_configs[config].match(f.name)]

        if not resp_files:
            print(f"[{config}] No response files found — skipping.")
            continue

        print(f"\n=== Collecting config: {config} ({len(resp_files)} pids) ===")

        # Responses
        resp_df  = pd.concat([pd.read_pickle(f) for f in resp_files], ignore_index=True)
        resp_out = run_dir / f"{MODEL_TYPE}_{DATASET}_{config}_responses.pkl"
        resp_df.to_pickle(resp_out)
        print(f"  Responses → {resp_out}  shape={resp_df.shape}")

        # Activities
        if act_files:
            act_df  = pd.concat([pd.read_pickle(f) for f in act_files], ignore_index=True)
            act_out = run_dir / f"activities_error_{DATASET}_{config}.pkl"
            act_df.to_pickle(act_out)
            print(f"  Activities → {act_out}  shape={act_df.shape}")
        else:
            print(f"  [warning] No activity files found for {config}")

        # Encoders (per-pid or fall back to source copy)
        if enc_files:
            enc_df  = pd.concat([pd.read_pickle(f) for f in enc_files], ignore_index=True)
            enc_out = run_dir / f"encoders_error_{DATASET}_{config}.pkl"
            enc_df.to_pickle(enc_out)
            print(f"  Encoders  → {enc_out}")
        else:
            print(f"  [info] No per-pid encoder files for {config}; "
                  "encoders are config-independent (tuning curves use seed=trial)")

        # Completeness check
        expected = set(_yoo_pids())
        found    = {int(resp_re.match(f.name).group(1)) for f in resp_files}
        missing  = sorted(expected - found)
        if missing:
            print(f"  [warning] Missing {len(missing)} pids: {missing}")
        else:
            print(f"  All {len(expected)} pids present.")

    # --- Shared files copied from source (config-independent) ---
    # Encoders: tuning curves depend only on seed=trial, not lambda
    src_enc = src_dir / "encoders_error_yoo.pkl"
    dst_enc = run_dir / "encoders_error_yoo.pkl"
    if src_enc.exists() and not dst_enc.exists():
        shutil.copy2(src_enc, dst_enc)
        print(f"\nCopied shared encoders: {src_enc} → {dst_enc}")

    # Params
    src_params = src_dir / f"{MODEL_TYPE}_{DATASET}_params.pkl"
    dst_params = run_dir / f"{MODEL_TYPE}_{DATASET}_params.pkl"
    if src_params.exists():
        shutil.copy2(src_params, dst_params)
        print(f"Copied params: {src_params} → {dst_params}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="scripts.extras_yoo")
    parser.add_argument("--experiment",   choices=["noise", "lambda0"], default="noise")
    parser.add_argument("--mode",         choices=["run", "collect"], required=True)
    parser.add_argument("--n_seeds",      type=int, default=10)
    parser.add_argument("--run_folder",   type=str, default="nef200")
    parser.add_argument("--source_folder",type=str, default="refit")
    parser.add_argument("--pid",          type=int, default=None)
    parser.add_argument("--seed_index",   type=int, default=None)
    parser.add_argument("--set_lambda",   type=float, default=None,
                        help="Override lambda_ with this value (e.g. 0.0). "                             "None = use fitted value.")
    args = parser.parse_args()

    if args.experiment == "noise":
        if args.mode == "run":
            _run_noise_mode(args)
        elif args.mode == "collect":
            _collect_noise_mode(args)

    elif args.experiment == "lambda0":
        if args.mode == "run":
            _run_one_simulate(args.pid, args.run_folder, args.source_folder,
                              set_lambda=args.set_lambda)
        elif args.mode == "collect":
            _collect_lambda0_mode(args)


if __name__ == "__main__":
    main()
