#!/usr/bin/env python3
"""neural_experiments.py — supplementary NEF simulations for the neural
results giant figure (Acts 1-3; see chat for the full narrative plan).

Generalizes extras_carrabin.py's pattern (param-grid sweeps, probe
simulations) to an arbitrary --task, since none of this is actually
carrabin-specific under the hood -- models/NEF.py's build_network/_pretrain
are always built on counting_integrator regardless of dataset.

Four experiments:

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
                 FITTED params, repeated across that pid's real trials, for
                 the within-repeat variability numbers Acts 2/3 originally
                 used. Has a --mode run/submit/collect lifecycle. Superseded
                 for the neural giant's row 2 by `synthetic` below (per
                 instruction: artificial data is just as good as fitted-pid
                 data for a qualitative covariation prediction, and much
                 cheaper/more robust to generate) -- kept here, not removed,
                 since it's still a real, independently-useful experiment.

  synthetic    — Acts 2/3's replacement data source. N "virtual pids", each
                 an independently-generated real trial sequence (via
                 task_backend/generate_sequences.py's own pool mechanism --
                 same generative design as real participants, including the
                 repeated-prefix/qid structure a real sigma estimate needs)
                 paired with ONE randomly-drawn (alpha_0, lambda_, n_neurons)
                 -- NOT a fitted pid's own params. One simulation pass per
                 virtual pid saves response, decoded PE, per-neuron error-
                 population activity, AND that trial's own encoders (which
                 genuinely differ per seed -- see chat) all at once, so no
                 further commands are needed afterward. Has the same
                 --mode run/submit/collect lifecycle as `probe`.

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

    python scripts/neural_experiments.py synthetic --task soltani_numbers \\
        --mode run --pid 1
    python scripts/neural_experiments.py synthetic --task soltani_numbers \\
        --mode submit --n_pids 200 --dry_run
    python scripts/neural_experiments.py synthetic --task soltani_numbers \\
        --mode collect

Output: data/runs/neural_experiments/
    raster_demo_{task}.pkl
    sweep_{task}_{sweep_param}.pkl
    probe_{task}_pid{pid}.pkl        (per-pid, --mode run)
    probe_{task}.pkl                 (combined, --mode collect)
    synthetic_{task}_{probe,activity,encoders,params}_pid{pid}.pkl  (per-pid)
    synthetic_{task}_{probe,activity,encoders,params}.pkl           (combined)
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

    Uses models.counting_integrator.activity_key_for_trial for BOTH the
    activity-map lookup AND the simulation seed -- NOT the raw trial number
    directly. That function exists specifically because soltani trials are
    0-indexed while activity keys start at 1; using the raw trial number for
    either one (as an earlier version of this function did) either misses
    the activity map for trial 0 entirely, or -- worse -- silently pairs a
    trial's simulation with a DIFFERENT trial's own tuning curves for every
    other trial, since key k's decoders are only valid for a network built
    with seed=k. See that function's own docstring.
    """
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS, _pretrain, _simulate_trial
    from models.counting_integrator import activity_key_for_trial, fast_decode, load_activities

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
        akey = activity_key_for_trial(task, int(trial))
        p = {**params, "seed": akey}

        if activity_map is not None:
            activity = activity_map.get(akey)
            decoders = (
                fast_decode(activity, alpha_0=float(params["alpha_0"]),
                            lambda_=float(params["lambda_"]))
                if activity is not None else _pretrain({**p, "base_seed": akey})
            )
        else:
            decoders = _pretrain({**p, "base_seed": akey})

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


# ── synthetic (Acts 2/3's replacement data source) ──────────────────────────────────────

SYNTHETIC_POOL_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_pool"
SYNTHETIC_N_NEURONS_CHOICES = list(range(100, 1001, 100))


def _synthetic_pool_path(task: str) -> Path:
    """task_backend/generate_sequences.py's own file naming uses the bare
    task name ("numbers"/"colors"), not this project's "soltani_"-prefixed
    dataset keys.
    """
    bare = task.removeprefix("soltani_")
    return SYNTHETIC_POOL_DIR / f"sequences_{bare}.json"


def _synthetic_params(virtual_pid: int) -> dict:
    """Deterministic random draw for one virtual pid -- alpha_0/lambda_
    Uniform(0,1), n_neurons uniform over the precomputed grid
    {100,...,1000}, per instruction. Seeded by virtual_pid so re-running
    --mode run for the same pid always reproduces the same draw, matching
    this project's own established convention elsewhere (e.g.
    fitting.fit._cross_validate's RandomState(seed=pid)).
    """
    rng = np.random.RandomState(seed=int(virtual_pid))
    return {
        "alpha_0": float(rng.uniform(0.0, 1.0)),
        "lambda_": float(rng.uniform(0.0, 1.0)),
        "n_neurons": int(rng.choice(SYNTHETIC_N_NEURONS_CHOICES)),
    }


def _load_synthetic_trials(task: str, virtual_pid: int) -> list[dict]:
    """Pool member (virtual_pid - 1) from the pre-generated pool -- a list
    of trial dicts, each with 'trial', 'qid', and 'values' (the observation
    sequence), matching the SAME repeated-qid structure real participants
    see (task_backend/generate_sequences.py's own generative design).
    """
    import json

    path = _synthetic_pool_path(task)
    if not path.exists():
        raise FileNotFoundError(
            f"No synthetic pool at {path}. Generate it first:\n"
            f"  venv/bin/python task_backend/generate_sequences.py "
            f"--task {task.removeprefix('soltani_')} --n_pool 200 "
            f"--pool_dir {SYNTHETIC_POOL_DIR}"
        )
    with open(path) as f:
        pool = json.load(f)
    if virtual_pid < 1 or virtual_pid > len(pool):
        raise ValueError(f"virtual_pid must be in [1, {len(pool)}], got {virtual_pid}")
    return pool[virtual_pid - 1]


def _simulate_trial_full(obs_values: np.ndarray, params: dict, decoders: dict,
                         seed: int) -> tuple[list[dict], np.ndarray]:
    """One trial: build+run once, extract (per observation) response and
    decoded PE at their usual readout points, AND per-neuron error-
    population activity at the SAME points -- tau_probe-filtered, matching
    utils/save_activities.py's own convention for this exact quantity (NOT
    the raw synapse=None probe _simulate_full uses for the spike raster,
    a different, deliberately unfiltered use case). Also returns this
    trial's own error-ensemble encoders, which genuinely differ by seed
    (see chat: net.error is built with seed=seed directly) -- so a caller
    needing them for weight-tuned-neuron identification must keep them
    paired with THIS trial, not reuse another trial's.
    """
    import nengo
    from models.NEF import build_network

    p = {**params, "seed": int(seed)}
    net = build_network(obs_values, p, decoders)
    tau_probe = float(p["tau_probe"])
    with net:
        probe_activity = nengo.Probe(net.error.neurons, synapse=tau_probe)

    dt = float(p["dt"])
    n_obs = len(obs_values)
    t_obs = float(p["t_obs"])
    t_iti = float(p["t_iti"])
    t_step = t_obs + t_iti
    t_total = n_obs * t_step

    with nengo.Simulator(net, dt=dt, seed=int(seed), progress_bar=False) as sim:
        sim.run(t_total)
        encoders = np.array(sim.data[net.error].encoders, copy=True)
        value_trace = sim.data[net.probe_value].squeeze()
        error_trace = sim.data[net.probe_error]        # (T, 2): weight, pe_raw
        activity_trace = sim.data[probe_activity]       # (T, n_neurons)

    n_timesteps = len(value_trace)
    rows = []
    for n_idx in range(n_obs):
        t_pe = t_iti + n_idx * t_step + READOUT_OFFSET
        t_resp = t_iti + n_idx * t_step + t_obs
        idx_pe = int(np.clip(np.round(t_pe / dt), 0, n_timesteps - 1))
        idx_resp = int(np.clip(np.round(t_resp / dt), 0, n_timesteps - 1))
        rows.append({
            "observation": n_idx + 1,
            "pe": float(abs(error_trace[idx_pe, 0] * error_trace[idx_pe, 1])),
            "response": float(value_trace[idx_resp]),
            "activity": activity_trace[idx_pe].copy(),
        })
    return rows, encoders


def _synthetic_worker(task: str, virtual_pid: int) -> dict:
    """Full simulation for one virtual pid across ALL of its (synthetic)
    trials -- one randomly-drawn (alpha_0, lambda_, n_neurons), one
    generated trial sequence. Returns everything needed for Acts 2/3's
    panels from this ONE call: probe rows (pe/response per observation,
    for sigma), activity rows (per-neuron, for activity decay), encoder
    rows (per trial, for weight-tuned-neuron identification), and the
    drawn params themselves -- no further commands needed afterward, per
    instruction.
    """
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS, _pretrain
    from models.counting_integrator import fast_decode, load_activities

    draw = _synthetic_params(virtual_pid)
    fixed = MODEL_PARAMS[task]["NEF"].get("fixed", {})
    params = {
        **PARAM_DEFAULTS, **fixed,
        "dataset": task, "model_type": "NEF", "pid": int(virtual_pid),
        "alpha_0": draw["alpha_0"], "lambda_": draw["lambda_"],
        "n_neurons": draw["n_neurons"], "n_neurons_counting": draw["n_neurons"],
    }

    try:
        activity_map = load_activities(
            n_neurons=draw["n_neurons"], n_neurons_counting=draw["n_neurons"], dataset=task)
    except FileNotFoundError:
        activity_map = None

    trials = _load_synthetic_trials(task, virtual_pid)

    probe_rows, activity_rows, encoder_rows = [], [], []
    for ti, trial_data in enumerate(trials, 1):
        trial_idx = int(trial_data["trial"])
        qid = trial_data["qid"]
        obs_values = np.array(trial_data["values"], dtype=float)
        seed = trial_idx + 1  # 1-indexed, matching activity_key_for_trial's own
                              # +1 convention for 0-indexed (soltani-style) trials

        if activity_map is not None:
            activity = activity_map.get(seed)
            decoders = (
                fast_decode(activity, alpha_0=draw["alpha_0"], lambda_=draw["lambda_"])
                if activity is not None
                else _pretrain({**params, "seed": seed, "base_seed": seed})
            )
        else:
            decoders = _pretrain({**params, "seed": seed, "base_seed": seed})

        try:
            obs_rows, encoders = _simulate_trial_full(obs_values, params, decoders, seed)
        except Exception as e:
            print(f"\n  Warning: virtual_pid={virtual_pid} trial {trial_idx} failed ({e}), skipping")
            continue

        for r in obs_rows:
            probe_rows.append({
                "virtual_pid": virtual_pid, "trial": trial_idx, "qid": qid,
                "observation": r["observation"], "pe": r["pe"], "response": r["response"],
            })
            act_row = {"virtual_pid": virtual_pid, "trial": trial_idx,
                      "observation": r["observation"]}
            for j, v in enumerate(r["activity"]):
                act_row[f"n{j}"] = float(v)
            activity_rows.append(act_row)
        for neuron_idx in range(encoders.shape[0]):
            enc_row = {"virtual_pid": virtual_pid, "trial": trial_idx, "neuron_idx": neuron_idx}
            for d in range(encoders.shape[1]):
                enc_row[f"enc_dim_{d}"] = float(encoders[neuron_idx, d])
            encoder_rows.append(enc_row)
        print(f"\r  virtual_pid={virtual_pid}  trial {ti:3d}/{len(trials)}", end="", flush=True)
    print()

    return {
        "params": pd.DataFrame([{"virtual_pid": virtual_pid, **draw}]),
        "probe": pd.DataFrame(probe_rows),
        "activity": pd.DataFrame(activity_rows),
        "encoders": pd.DataFrame(encoder_rows),
    }


def run_synthetic(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kinds = ["probe", "activity", "encoders", "params"]

    if args.mode == "run":
        if args.pid is None:
            raise SystemExit("--pid required for --mode run")
        out_paths = {k: OUT_DIR / f"synthetic_{args.task}_{k}_pid{args.pid}.pkl" for k in kinds}
        if all(p.exists() for p in out_paths.values()):
            print(f"Already exists: virtual_pid={args.pid} -- skipping (delete to rerun)")
            return
        result = _synthetic_worker(args.task, args.pid)
        for k in kinds:
            result[k].to_pickle(out_paths[k])
        print(f"Saved virtual_pid={args.pid}: "
              f"{len(result['probe'])} probe rows, {len(result['activity'])} activity rows, "
              f"{len(result['encoders'])} encoder rows -> {OUT_DIR}")

    elif args.mode == "submit":
        root = str(Path(__file__).resolve().parent.parent)
        print(f"Submitting {args.n_pids} synthetic jobs for task={args.task}")
        for virtual_pid in range(1, args.n_pids + 1):
            out_paths = [OUT_DIR / f"synthetic_{args.task}_{k}_pid{virtual_pid}.pkl" for k in kinds]
            if all(p.exists() for p in out_paths):
                print(f"  virtual_pid={virtual_pid}: already exists -- skipping")
                continue
            cmd = (f"venv/bin/python scripts/neural_experiments.py synthetic "
                  f"--task {args.task} --mode run --pid {virtual_pid}")
            script = make_job_script(root, [cmd], time_limit="2:0:0", mem="16G")
            script_path = OUT_DIR / f"_job_synthetic_{args.task}_pid{virtual_pid}.sh"
            script_path.write_text(script)
            submit_script(script_path, dry_run=args.dry_run)

    elif args.mode == "collect":
        for k in kinds:
            files = sorted(OUT_DIR.glob(f"synthetic_{args.task}_{k}_pid*.pkl"))
            if not files:
                print(f"No synthetic_{args.task}_{k}_pid*.pkl files found in {OUT_DIR}")
                continue
            df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
            out_path = OUT_DIR / f"synthetic_{args.task}_{k}.pkl"
            df.to_pickle(out_path)
            n_pids = df["virtual_pid"].nunique()
            print(f"Collected {len(files)} file(s), {n_pids} virtual pids -> {out_path}")


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

    p_synth = sub.add_parser("synthetic")
    p_synth.add_argument("--task", required=True)
    p_synth.add_argument("--mode", required=True, choices=["run", "submit", "collect"])
    p_synth.add_argument("--pid", type=int, default=None,
                        help="Virtual pid index, 1-based (--mode run)")
    p_synth.add_argument("--n_pids", type=int, default=200,
                        help="Total virtual pids to submit (--mode submit)")
    p_synth.add_argument("--dry_run", action="store_true")
    p_synth.set_defaults(func=run_synthetic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
