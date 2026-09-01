#!/usr/bin/env python3
"""neural_experiments.py — supplementary NEF simulations for the neural
results giant figure (Acts 1-3; see chat for the full narrative plan).

Generalizes extras_carrabin.py's pattern (param-grid sweeps, probe
simulations) to an arbitrary --task, since none of this is actually
carrabin-specific under the hood -- models/NEF.py's build_network/_pretrain
are always built on counting_integrator regardless of dataset.

Three experiments:

  raster_demo  — ONE trial, arbitrary (alpha_0, n_neurons, lambda_), full
                 per-timestep trace of the error population's raw neuron
                 output (for a spike raster) plus its decoded value/error.
                 Local, cheap. (Act 1.1)

  sweep        — vary ONE or TWO of {alpha_0, n_neurons, lambda_} across
                 several arbitrary values (a cross product if two), the
                 remaining one held at a fixed base value, over several
                 seeds. Always simulated at full per-timestep resolution
                 (Act 1.3's own need) since Act 1.2's per-trial aggregates
                 are trivially derived from that at plot time -- no need to
                 run two separate simulations. Local, cheap. Two-parameter
                 mode is for the PE-dynamics panel specifically (alpha_0 x
                 n_neurons, matching the original reference figure's own
                 convention); lambda_'s own panel stays single-parameter.

  probe        — full per-timestep probe simulation at a given pid's own
                 fitted params, repeated across that pid's real trials, for
                 the within-repeat variability numbers Acts 2/3 need. Has a
                 --mode run/submit/collect lifecycle since this is the
                 expensive piece (~cluster-bound, matching carrabin's own
                 2.36GB probe_pids_carrabin.pkl at only 21 pids).

Run examples:
    python scripts/neural_experiments.py raster_demo --task soltani_numbers \\
        --alpha_0 0.5 --n_neurons 200 --lambda_ 0.5 --n_obs 5

    python scripts/neural_experiments.py sweep --task soltani_numbers \\
        --sweep_param alpha_0 --sweep_values 0.1 0.5 1.0 \\
        --base_alpha_0 0.5 --base_n_neurons 200 --base_lambda_ 0.5 --n_seeds 10

    python scripts/neural_experiments.py probe --task soltani_numbers \\
        --mode run --pid 3 --run_folder rmse
    python scripts/neural_experiments.py probe --task soltani_numbers \\
        --mode submit --run_folder rmse --dry_run
    python scripts/neural_experiments.py probe --task soltani_numbers \\
        --mode collect

Output: data/runs/neural_experiments/
    raster_demo_{task}.pkl
    sweep_{task}_{sweep_param}.pkl
    probe_{task}_pid{pid}.pkl        (per-pid, --mode run)
    probe_{task}.pkl                 (combined, --mode collect)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import RUNS_DIR, data_path
from utils.slurm import make_job_script, submit_script

OUT_DIR = RUNS_DIR / "neural_experiments"

# task -> human data pickle stem -- only used by `probe` to pull a real
# pid's own trial/observation structure. raster_demo and sweep never touch
# human data at all -- everything about them is an arbitrary, self-contained
# toy trial, per instruction.
TASK_DATAFILE = {
    "carrabin": "carrabin.pkl",
    "yoo": "yoo.pkl",
    "soltani_colors": "soltani_colors.pkl",
    "soltani_numbers": "soltani_numbers.pkl",
}

READOUT_OFFSET = 0.5  # seconds into observation window for readout, matching
                      # every other NEF-adjacent script's own convention.


def _base_params(task: str, alpha_0: float, n_neurons: float, lambda_: float,
                  **overrides) -> dict:
    """Full NEF params dict for one arbitrary (alpha_0, n_neurons, lambda_)
    setting -- PARAM_DEFAULTS + this task's own MODEL_PARAMS[...]['NEF']
    ['fixed'] (radius_c/n_neurons_counting differ per task), then the three
    swept/demo values on top, then any explicit overrides.
    """
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS

    fixed = MODEL_PARAMS[task]["NEF"].get("fixed", {})
    params = {
        **PARAM_DEFAULTS,
        **fixed,
        "dataset": task,
        "model_type": "NEF",
        "pid": 0,
        "base_seed": 0,
        "seed": 0,
        "alpha_0": float(alpha_0),
        "n_neurons": int(n_neurons),
        "lambda_": float(lambda_),
        **overrides,
    }
    return params


def _simulate_full(params: dict, obs_values: np.ndarray, decoders: dict | None = None,
                   seed: int = 0) -> dict:
    """Build and run one trial, returning EVERY per-timestep probe as a
    plain array -- unlike models.NEF._simulate_trial, which only exposes
    net.probe_error_neurons at readout-moment snapshots (one row per
    observation). raster_demo and sweep both need the FULL, unsubsampled
    trace (raster_demo for the spike raster itself; sweep because deriving
    Act 1.2's per-observation aggregate from a full per-timestep trace is
    trivial, while the reverse isn't possible), so this bypasses
    _simulate_trial and reads nengo.Simulator's own probe data directly --
    same pattern extras_carrabin.py's own _run_pe_dynamics already uses for
    the identical reason.
    """
    import nengo
    from models.NEF import _pretrain, build_network

    p = {**params, "seed": int(seed)}
    if decoders is None:
        decoders = _pretrain({**p, "base_seed": int(seed)})

    net = build_network(obs_values, p, decoders)
    dt = float(p["dt"])
    n_obs = len(obs_values)
    t_total = n_obs * (float(p["t_obs"]) + float(p["t_iti"]))

    with nengo.Simulator(net, dt=dt, seed=int(seed), progress_bar=False) as sim:
        sim.run(t_total)

    t_arr = np.arange(len(sim.data[net.probe_value])) * dt
    error_dec = sim.data[net.probe_error]          # (T, 2): [:,0]=weight/alpha(t), [:,1]=raw PE
    return {
        "t": t_arr,
        "value": sim.data[net.probe_value].squeeze(),
        "weight": error_dec[:, 0],
        "pe_raw": error_dec[:, 1],
        "pe_product": error_dec[:, 0] * error_dec[:, 1],
        "error_neurons": sim.data[net.probe_error_neurons],   # (T, n_neurons) -- raw, for the raster
        "obs": sim.data[net.probe_obs].squeeze(),
    }


# ── raster_demo (Act 1.1) ────────────────────────────────────────────────────

def run_raster_demo(args) -> None:
    """One trial, one arbitrary parameter setting -- full per-timestep raw
    error-population output (for utils.plot_spikes.plot_spikes) plus the
    decoded value/PE trace to overlay alongside it.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"raster_demo_{args.task}.pkl"

    obs_values = np.array(args.obs_values, dtype=float) if args.obs_values \
        else np.ones(args.n_obs)
    params = _base_params(args.task, args.alpha_0, args.n_neurons, args.lambda_)

    print(f"raster_demo: task={args.task} alpha_0={args.alpha_0} "
          f"n_neurons={args.n_neurons} lambda_={args.lambda_} "
          f"n_obs={len(obs_values)}")
    result = _simulate_full(params, obs_values, seed=args.seed)
    result["params"] = params
    result["obs_values"] = obs_values

    pd.to_pickle(result, out_path)
    print(f"Saved -> {out_path}")


# ── sweep (Act 1.2 + 1.3) ─────────────────────────────────────────────────────

def run_sweep(args) -> None:
    """Vary ONE or TWO of {alpha_0, n_neurons, lambda_} across
    --sweep_values (and --sweep_values2, if --sweep_param2 is given), the
    remaining one held at its --base_* value, over --n_seeds seeds. One
    output row per (sweep_value[, sweep_value2], seed, timestep).

    Two-parameter mode produces the cross product of --sweep_values x
    --sweep_values2 -- e.g. 2x2=4 settings for 2 alpha_0 values x 2
    n_neurons values. Never a three-way cross product (sweep_param2 is
    optional and limited to exactly one partner) -- that's what made the
    old carrabin Panel A hard to read beyond two dimensions.

    Output filename disambiguates by BOTH swept parameter names when two
    are given (sweep_{task}_{sweep_param}_{sweep_param2}.pkl), since the
    single-parameter filename alone doesn't encode which base value the
    OTHER parameter was held at -- two single-parameter sweep calls at
    different base values for the same sweep_param would otherwise
    silently overwrite each other's output.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.sweep_param2:
        out_path = OUT_DIR / f"sweep_{args.task}_{args.sweep_param}_{args.sweep_param2}.pkl"
    else:
        out_path = OUT_DIR / f"sweep_{args.task}_{args.sweep_param}.pkl"

    base = {"alpha_0": args.base_alpha_0, "n_neurons": args.base_n_neurons,
            "lambda_": args.base_lambda_}
    obs_values = np.array(args.obs_values, dtype=float) if args.obs_values \
        else np.ones(args.n_obs)
    values2 = args.sweep_values2 if args.sweep_param2 else [None]

    print(f"sweep: task={args.task} sweep_param={args.sweep_param} "
          f"values={args.sweep_values}"
          + (f" sweep_param2={args.sweep_param2} values2={args.sweep_values2}"
             if args.sweep_param2 else "")
          + f" base={base} n_seeds={args.n_seeds}")

    rows = []
    for value in args.sweep_values:
        for value2 in values2:
            setting = {**base, args.sweep_param: value}
            if args.sweep_param2:
                setting[args.sweep_param2] = value2
            params = _base_params(args.task, setting["alpha_0"], setting["n_neurons"],
                                  setting["lambda_"])
            for seed in range(args.n_seeds):
                result = _simulate_full(params, obs_values, seed=seed)
                row_data = {
                    "sweep_value": value,
                    "seed": seed,
                    "t": result["t"],
                    "obs": result["obs"],
                    "pe_product": result["pe_product"],
                    "pe_raw": result["pe_raw"],
                    "weight": result["weight"],
                    "value_decoded": result["value"],
                    "mean_error_activity": result["error_neurons"].mean(axis=1),
                }
                if args.sweep_param2:
                    row_data["sweep_value2"] = value2
                rows.append(pd.DataFrame(row_data))
                label = f"{args.sweep_param}={value}"
                if args.sweep_param2:
                    label += f" {args.sweep_param2}={value2}"
                print(f"  {label}  seed {seed + 1}/{args.n_seeds}", end="\r", flush=True)
            print()

    df = pd.concat(rows, ignore_index=True)
    meta = {"df": df, "sweep_param": args.sweep_param, "base": base, "task": args.task}
    if args.sweep_param2:
        meta["sweep_param2"] = args.sweep_param2
    pd.to_pickle(meta, out_path)
    print(f"Saved {len(df):,} rows -> {out_path}")


# ── probe (Act 2/3's expensive half) ─────────────────────────────────────────

def _probe_worker(task: str, pid: int, run_folder: str) -> pd.DataFrame:
    """Full per-timestep probe simulation for one pid's own fitted params,
    across ALL of that pid's real trials -- the within-repeat variability
    data Acts 2/3 need (sigma/lambda vs PE-variability, decay metrics).
    Direct generalization of extras_carrabin.py's probe_timeseries.
    """
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS, _pretrain, _simulate_trial
    from models.counting_integrator import fast_decode, load_activities

    dataset_stem = TASK_DATAFILE[task]
    human = pd.read_pickle(data_path(dataset_stem))
    human_pid = human[human["pid"] == pid]
    if human_pid.empty:
        raise ValueError(f"No human data for pid={pid} in {dataset_stem}")
    has_qid = "qid" in human_pid.columns
    qid_map = (
        human_pid[["trial", "observation", "qid"]].drop_duplicates()
        .set_index(["trial", "observation"])["qid"]
        if has_qid else None
    )

    params_path = RUNS_DIR / run_folder / f"NEF_{task}_params.pkl"
    if not params_path.exists():
        raise FileNotFoundError(f"No fitted params at {params_path}")
    params_df = pd.read_pickle(params_path)
    row = params_df[params_df["pid"] == pid]
    if row.empty:
        raise ValueError(f"No fitted params for pid={pid} in {params_path}")

    fixed = MODEL_PARAMS[task]["NEF"].get("fixed", {})
    params = {**PARAM_DEFAULTS, **fixed, **row.iloc[0].to_dict()}
    params["dataset"] = task
    params["model_type"] = "NEF"
    params["pid"] = int(pid)

    try:
        activity_map = load_activities(
            n_neurons=int(params["n_neurons"]),
            n_neurons_counting=int(params["n_neurons_counting"]),
            dataset=task,
        )
    except FileNotFoundError:
        activity_map = None

    t_obs_ = float(params["t_obs"])
    t_iti_ = float(params["t_iti"])
    t_step = t_obs_ + t_iti_
    dt = float(params["dt"])
    n_obs = int(human_pid["observation"].max())
    n_trials = human_pid["trial"].nunique()

    rows = []
    for ti, (trial, trial_data) in enumerate(human_pid.groupby("trial"), 1):
        trial_data = trial_data.sort_values("observation")
        obs_values = trial_data["value"].to_numpy(dtype=float)
        p = {**params, "seed": int(trial)}

        if activity_map is not None:
            activity = activity_map.get(int(trial))
            decoders = (
                fast_decode(activity, alpha_0=float(params["alpha_0"]),
                            lambda_=float(params["lambda_"]))
                if activity is not None else _pretrain({**p, "base_seed": int(trial)})
            )
        else:
            decoders = _pretrain({**p, "base_seed": int(trial)})

        try:
            responses, probe = _simulate_trial(obs_values, p, decoders, return_probes=True)
        except Exception as e:
            print(f"\n  Warning: trial {trial} failed ({e}), skipping")
            continue

        t_arr = probe["t"]
        pe_trace = probe["error"][:, 1]
        for obs in range(1, n_obs + 1):
            idx_pe = int(np.argmin(np.abs(t_arr - (t_iti_ + (obs - 1) * t_step + READOUT_OFFSET))))
            idx_resp = int(np.argmin(np.abs(t_arr - (t_iti_ + (obs - 1) * t_step + t_obs_))))
            qid = qid_map.get((trial, obs), np.nan) if qid_map is not None else np.nan
            rows.append({
                "pid": pid, "trial": int(trial), "observation": obs, "qid": qid,
                "pe": float(abs(pe_trace[idx_pe])),
                "response": float(responses[obs - 1]),
            })
        print(f"\r  pid={pid}  trial {ti:3d}/{n_trials}", end="", flush=True)
    print()

    df = pd.DataFrame(rows)
    df["alpha_0"] = float(params["alpha_0"])
    df["lambda_"] = float(params["lambda_"])
    return df


def run_probe(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "run":
        if args.pid is None:
            raise SystemExit("--pid required for --mode run")
        out_path = OUT_DIR / f"probe_{args.task}_pid{args.pid}.pkl"
        if out_path.exists():
            print(f"Already exists: {out_path.name} -- skipping (delete to rerun)")
            return
        df = _probe_worker(args.task, args.pid, args.run_folder)
        df.to_pickle(out_path)
        print(f"Saved {len(df)} rows -> {out_path}")

    elif args.mode == "submit":
        params_path = RUNS_DIR / args.run_folder / f"NEF_{args.task}_params.pkl"
        pids = sorted(pd.read_pickle(params_path)["pid"].unique().tolist())
        root = str(Path(__file__).resolve().parent.parent)
        print(f"Submitting {len(pids)} probe jobs for task={args.task} "
              f"(run_folder={args.run_folder})")
        for pid in pids:
            out_path = OUT_DIR / f"probe_{args.task}_pid{pid}.pkl"
            if out_path.exists():
                print(f"  pid={pid}: already exists -- skipping")
                continue
            cmd = (f"venv/bin/python scripts/neural_experiments.py probe "
                  f"--task {args.task} --mode run --pid {pid} "
                  f"--run_folder {args.run_folder}")
            script = make_job_script(root, [cmd], time_limit="24:0:0", mem="16G")
            script_path = OUT_DIR / f"_job_probe_{args.task}_pid{pid}.sh"
            script_path.write_text(script)
            submit_script(script_path, dry_run=args.dry_run)

    elif args.mode == "collect":
        files = sorted(OUT_DIR.glob(f"probe_{args.task}_pid*.pkl"))
        if not files:
            print(f"No probe_{args.task}_pid*.pkl files found in {OUT_DIR}")
            return
        df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
        out_path = OUT_DIR / f"probe_{args.task}.pkl"
        df.to_pickle(out_path)
        n_pids = df["pid"].nunique()
        print(f"Collected {len(files)} file(s), {n_pids} pids -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_raster = sub.add_parser("raster_demo")
    p_raster.add_argument("--task", required=True)
    p_raster.add_argument("--alpha_0", type=float, required=True)
    p_raster.add_argument("--n_neurons", type=int, required=True)
    p_raster.add_argument("--lambda_", type=float, required=True)
    p_raster.add_argument("--n_obs", type=int, default=5,
                          help="Number of observations, if --obs_values not given")
    p_raster.add_argument("--obs_values", type=float, nargs="+", default=None,
                          help="Explicit observation sequence for the demo trial")
    p_raster.add_argument("--seed", type=int, default=0)
    p_raster.set_defaults(func=run_raster_demo)

    p_sweep = sub.add_parser("sweep")
    p_sweep.add_argument("--task", required=True)
    p_sweep.add_argument("--sweep_param", required=True,
                         choices=["alpha_0", "n_neurons", "lambda_"])
    p_sweep.add_argument("--sweep_values", type=float, nargs="+", required=True)
    p_sweep.add_argument("--sweep_param2", default=None,
                         choices=["alpha_0", "n_neurons", "lambda_"],
                         help="Optional second swept parameter (cross product)")
    p_sweep.add_argument("--sweep_values2", type=float, nargs="+", default=None)
    p_sweep.add_argument("--base_alpha_0", type=float, required=True)
    p_sweep.add_argument("--base_n_neurons", type=int, required=True)
    p_sweep.add_argument("--base_lambda_", type=float, required=True)
    p_sweep.add_argument("--n_obs", type=int, default=5)
    p_sweep.add_argument("--obs_values", type=float, nargs="+", default=None)
    p_sweep.add_argument("--n_seeds", type=int, default=10)
    p_sweep.set_defaults(func=run_sweep)

    p_probe = sub.add_parser("probe")
    p_probe.add_argument("--task", required=True)
    p_probe.add_argument("--mode", required=True, choices=["run", "submit", "collect"])
    p_probe.add_argument("--pid", type=int, default=None)
    p_probe.add_argument("--run_folder", type=str, default="rmse")
    p_probe.add_argument("--dry_run", action="store_true")
    p_probe.set_defaults(func=run_probe)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
