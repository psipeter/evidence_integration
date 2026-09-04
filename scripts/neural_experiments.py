#!/usr/bin/env python3
"""neural_experiments.py — supplementary NEF simulations for the neural
results giant figure (Acts 1-3; see chat for the full narrative plan).

Generalizes extras_carrabin.py's pattern (param-grid sweeps, probe
simulations) to an arbitrary --task, since none of this is actually
carrabin-specific under the hood -- models/NEF.py's build_network/_pretrain
are always built on counting_integrator regardless of dataset.

*** CONVENTION, RE-STATED AFTER A REAL VIOLATION (see docs/HISTORY.md) ***
NEVER let any simulation in this file fall back to a live _pretrain()
training run when a precomputed counting-activity file (or a specific
seed's key within it) is missing. Always load it via
models.counting_integrator.load_activities()/fast_decode() (this file's
own _require_activities()/_decoders_for_seed() wrap that pattern) and
RAISE with the exact regenerate command if it's missing -- matching
models.NEF's own _require_activity_map convention exactly. This was an
EXPLICIT prior instruction that got silently reintroduced once already
(_simulate_full, used by raster_demo/sweep/oddball, had a `decoders is
None -> _pretrain(...)` fallback baked in) and only surfaced because a
person noticed an unexplained asymmetry in an oddball result and asked
whether the real activity file was actually being used. A silent
_pretrain() fallback is NOT just slower (a full from-scratch Nengo
training run per call, vs. an analytic decode from a cached Gram matrix)
-- it is a genuinely DIFFERENT, unverified code path that can silently
diverge from the file-based path with no error, no warning, and no visual
cue in the output. If you are adding a new experiment to this file, or
any other simulation calling models.NEF.build_network anywhere in this
codebase, and it needs decoders: call _require_activities() +
_decoders_for_seed() (or the equivalent in models.NEF/counting_integrator)
and let a missing file/key raise. Do not add a new _pretrain() fallback,
here or anywhere else that touches NEF simulation.

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
                 Uses models.NEF._require_activity_map directly (its own
                 REQUIRED, not optional, no-_pretrain()-fallback
                 convention) -- fixed this session alongside `synthetic`
                 below; see this module's own top-of-file convention note.

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

# Cache for _require_activities -- see that function's own docstring.
_ACTIVITY_MAP_CACHE: dict[tuple[int, int, str], dict] = {}


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


def _require_activities(task: str, n_neurons: int, n_neurons_counting: int) -> dict:
    """Load precomputed counting-activity Gram matrices for (n_neurons,
    n_neurons_counting, task), or fail loudly with the exact regenerate
    command. REQUIRED, not optional -- raster_demo/sweep/oddball (every
    caller of _simulate_full) must NEVER fall back to a live _pretrain()
    training run when the file is missing. This mirrors models.NEF's own
    _require_activity_map convention exactly, and exists because that
    convention was previously violated in this exact file (a silent
    _pretrain() fallback inside _simulate_full) despite an explicit prior
    instruction never to do this -- see this module's own top-of-file
    convention note and docs/HISTORY.md.

    Cached per (n_neurons, n_neurons_counting, task) for the lifetime of
    one script invocation, since a sweep over many (alpha_0, lambda_)
    values at a fixed n_neurons would otherwise reload the same
    (possibly large, at n_neurons_counting=2000) file repeatedly.
    """
    key = (int(n_neurons), int(n_neurons_counting), task)
    if key not in _ACTIVITY_MAP_CACHE:
        from models.counting_integrator import load_activities
        try:
            _ACTIVITY_MAP_CACHE[key] = load_activities(
                n_neurons=int(n_neurons), n_neurons_counting=int(n_neurons_counting),
                dataset=task,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No precomputed counting-activity file for (n_neurons={n_neurons}, "
                f"n_neurons_counting={n_neurons_counting}, dataset={task!r}). This file is "
                f"REQUIRED -- neural_experiments.py never falls back to a live _pretrain() "
                f"training run (see this module's own top-of-file convention note). "
                f"Generate it first:\n"
                f"  venv/bin/python models/counting_integrator.py --precompute_activities "
                f"--n_neurons {n_neurons} --n_neurons_counting {n_neurons_counting} "
                f"--dataset {task}\n"
                f"then scp data/counting_activities_n{n_neurons}_nc{n_neurons_counting}_"
                f"{task}.pkl to the cluster if running remotely."
            ) from e
    return _ACTIVITY_MAP_CACHE[key]


def _toy_activity_key(seed: int) -> int:
    """Maps an arbitrary toy seed (0-indexed, as used by raster_demo/
    sweep/oddball/param_scan's own --seed/range(n_seeds) loops, which have
    no real trial to key off of) to the activity file's own 1-indexed key
    space. THE SINGLE SOURCE OF TRUTH for this mapping -- every caller
    must use this SAME returned value for BOTH the activity-map lookup
    (via _decoders_for_seed) AND the actual simulation seed passed to
    _simulate_full/_simulate_param_scan_trial.

    An EARLIER version of this file's own _decoders_for_seed applied this
    +1 offset ONLY to the lookup key, while every caller still passed the
    raw (un-offset) seed as the actual simulation seed -- silently pairing
    a network's own seed-dependent tuning curves with a DIFFERENT seed's
    decoders. This is exactly the "activity key vs simulation seed"
    mismatch this project's own activity_key_for_trial (models/
    counting_integrator.py) exists to prevent for real human trials (see
    that function's own docstring and this file's "What NOT to do" list),
    reintroduced here for these toy experiments and only caught because a
    person noticed neural_giant2's row-2 activity panel didn't match the
    ORIGINAL neural_giant's own (correctly seed-matched) equivalent panel.
    See docs/HISTORY.md for the full incident. NEVER hand-derive this
    offset inline a second time -- always call this function for both
    halves of the pairing.
    """
    return int(seed) + 1


def _decoders_for_seed(activity_map: dict, key: int, alpha_0: float, lambda_: float) -> dict:
    """fast_decode for one specific activity KEY -- callers must obtain
    `key` via _toy_activity_key(seed) and pass that SAME value both here
    AND as the `seed` argument to _simulate_full/_simulate_param_scan_
    trial. REQUIRED to be present in activity_map -- raises KeyError,
    never falls back to _pretrain(), matching _require_activities' own
    convention.
    """
    from models.counting_integrator import fast_decode
    activity = activity_map.get(key)
    if activity is None:
        raise KeyError(
            f"No precomputed counting activity for key={key}. "
            f"This dataset's activity file only has keys 1..n_trials -- lower "
            f"--n_seeds/--seed, or regenerate with a larger --n_trials via "
            f"models/counting_integrator.py --precompute_activities."
        )
    return fast_decode(activity, alpha_0=float(alpha_0), lambda_=float(lambda_))


def _simulate_full(params: dict, obs_values: np.ndarray, decoders: dict,
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

    `decoders` is REQUIRED -- this function must NEVER fall back to a live
    _pretrain() training run. Every caller must obtain decoders via
    _require_activities()/_decoders_for_seed() (fast_decode against a
    precomputed counting-activity file) BEFORE calling this, so a missing
    file or seed key fails loudly there rather than silently retraining
    here. See this module's own top-of-file convention note.
    """
    import nengo
    from models.NEF import build_network

    p = {**params, "seed": int(seed)}
    net = build_network(obs_values, p, decoders)
    dt = float(p["dt"])
    n_obs = len(obs_values)
    t_total = n_obs * (float(p["t_obs"]) + float(p["t_iti"]))

    with nengo.Simulator(net, dt=dt, seed=int(seed), progress_bar=False) as sim:
        sim.run(t_total)
        encoders = np.array(sim.data[net.error].encoders, copy=True)

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
        "encoders": encoders,   # (n_neurons, 2) -- ADDED this session, for weight-tuned filtering
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

    activity_map = _require_activities(args.task, args.n_neurons, params["n_neurons_counting"])
    key = _toy_activity_key(args.seed)
    decoders = _decoders_for_seed(activity_map, key, args.alpha_0, args.lambda_)

    print(f"raster_demo: task={args.task} alpha_0={args.alpha_0} "
          f"n_neurons={args.n_neurons} lambda_={args.lambda_} "
          f"n_obs={len(obs_values)}")
    result = _simulate_full(params, obs_values, decoders, seed=key)
    result["params"] = params
    result["obs_values"] = obs_values

    pd.to_pickle(result, out_path)
    print(f"Saved -> {out_path}")


def run_n_neurons_demo(args) -> None:
    """Local, cheap demo (NOT cluster-submitted, matching raster_demo/
    sweep's own convention -- a handful of single-seed toy simulations,
    not a real/synthetic-pool sweep): simulates the SAME single 5-
    observation toy sequence across several (n_neurons, n_neurons_
    counting) PAIRS (n_neurons_counting = 4x n_neurons, per instruction --
    a DIFFERENT ratio convention from both `synthetic`'s own n_neurons=
    n_neurons_counting and param_scan's own n_neurons_counting-fixed-at-
    2000; see this experiment's own required activity files below), at
    ONE shared (alpha_0, lambda_), --n_seeds independent toy seeds PER
    PAIR (via _toy_activity_key -- the SAME arbitrary-demo-seed
    convention raster_demo/sweep already use, NOT a real trial or
    synthetic-pool member). Built for neural_giant2's row 3 (n_neurons)
    col 1 panel, to visually illustrate how the decoded value
    population's own tracking gets noisier/more-drifting at fewer
    neurons, before row 3's own col 2/3 quantify this systematically.

    Saves RAW per-seed traces, NOT pre-averaged -- a SINGLE seed risked
    an unrepresentative too-good/too-bad draw making the panel
    confusing (per instruction), but full averaging across many seeds
    would smooth away the very spike-driven noise this panel exists to
    show; --n_seeds (default a SMALL number) is deliberately a middle
    ground, and the actual mean+CI aggregation happens in the figure
    script via sns.lineplot (matching this project's established
    convention of handing seaborn a long-format frame and letting it
    aggregate, e.g. _plot_oddball_pe_trace), not here.

    Each pair needs its OWN precomputed activity file (file identity is
    the exact (n_neurons, n_neurons_counting) pair, not just n_neurons --
    see _require_activities' own docstring) -- REQUIRED, no _pretrain()
    fallback, same as every other experiment in this module.

    Saves ALL pairs'/seeds' own full per-timestep decoded-VALUE trace,
    plus the obs_values/alpha_0/lambda_ actually used -- needed
    downstream so the figure script can recompute the RL_lambda ideal-
    target overlay reproducibly from the saved file alone, rather than
    hardcoding these values again in two places.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"n_neurons_demo_{args.task}.pkl"

    obs_values_raw = np.array(args.obs_values, dtype=float)
    # RAW 0-100 pool scale -- rescale for soltani_numbers, matching
    # _synthetic_worker's/_param_scan_worker's own convention exactly.
    obs_values = (obs_values_raw / 50.0 - 1.0) if args.task == "soltani_numbers" \
        else obs_values_raw

    pairs = []
    for spec in args.n_neurons_pairs:
        n_str, nc_str = spec.split(":")
        pairs.append((int(n_str), int(nc_str)))

    traces = {}
    for n_neurons, n_neurons_counting in pairs:
        params = _base_params(args.task, args.alpha_0, n_neurons, args.lambda_,
                              n_neurons_counting=n_neurons_counting)
        activity_map = _require_activities(args.task, n_neurons, n_neurons_counting)
        seed_traces = []
        for seed in range(args.n_seeds):
            key = _toy_activity_key(seed)
            decoders = _decoders_for_seed(activity_map, key, args.alpha_0, args.lambda_)
            print(f"n_neurons_demo: n_neurons={n_neurons} n_neurons_counting={n_neurons_counting} "
                  f"seed {seed + 1}/{args.n_seeds}")
            result = _simulate_full(params, obs_values, decoders, seed=key)
            seed_traces.append({"seed": seed, "t": result["t"], "value": result["value"]})
        traces[(n_neurons, n_neurons_counting)] = seed_traces

    out = {
        "traces": traces,
        "obs_values_raw": obs_values_raw,
        "alpha_0": args.alpha_0,
        "lambda_": args.lambda_,
        "task": args.task,
    }
    pd.to_pickle(out, out_path)
    print(f"Saved {len(pairs)} pair(s) x {args.n_seeds} seed(s) -> {out_path}")


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
            activity_map = _require_activities(
                args.task, setting["n_neurons"], params["n_neurons_counting"])
            for seed in range(args.n_seeds):
                key = _toy_activity_key(seed)
                decoders = _decoders_for_seed(
                    activity_map, key, setting["alpha_0"], setting["lambda_"])
                result = _simulate_full(params, obs_values, decoders, seed=key)
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
    from models.NEF import PARAM_DEFAULTS, _require_activity_map, _simulate_trial
    from models.counting_integrator import activity_key_for_trial, fast_decode

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

    # REQUIRED, not optional -- matches models.NEF.run()'s own convention
    # (_require_activity_map) exactly. NEVER fall back to _pretrain() when
    # this file is missing -- see this module's own top-of-file convention
    # note and docs/HISTORY.md for the incident this re-states.
    activity_map = _require_activity_map(
        int(params["n_neurons"]), int(params["n_neurons_counting"]), task)

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

        # REQUIRED, not optional -- a missing key means this activity file
        # doesn't cover this trial at all (e.g. too few trial-seeds
        # precomputed); regenerate rather than silently retraining with
        # mismatched tuning curves. See this module's own top-of-file
        # convention note.
        activity = activity_map.get(akey)
        if activity is None:
            raise KeyError(
                f"No precomputed counting activity for key={akey} "
                f"(dataset={task!r}, trial={int(trial)}). The activity file "
                f"has keys 1..n_trials -- check _DATASET_N_TRIALS in "
                f"models/counting_integrator.py, or regenerate with "
                f"--precompute_activities."
            )
        decoders = fast_decode(activity, alpha_0=float(params["alpha_0"]),
                                lambda_=float(params["lambda_"]))

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
SYNTHETIC_N_NEURONS_CHOICES = list(range(500, 1501, 100))


def _synthetic_pool_path(task: str) -> Path:
    """task_backend/generate_sequences.py's own file naming uses the bare
    task name ("numbers"/"colors"), not this project's "soltani_"-prefixed
    dataset keys.
    """
    bare = task.removeprefix("soltani_")
    return SYNTHETIC_POOL_DIR / f"sequences_{bare}.json"


def _synthetic_params(virtual_pid: int) -> dict:
    """Deterministic random draw for one virtual pid -- alpha_0 ~
    Uniform(0.5,1), lambda_ ~ Uniform(0.1,1), n_neurons uniform over the
    precomputed grid {500,...,1500}. Narrowed from the original
    Uniform(0,1)/{100,...,1000} after checking real soltani_numbers RMSE
    fits directly: alpha_0 never falls below 0.384 there (5th percentile
    0.481), and alpha_0 below ~0.2-0.4 produces a genuine floor effect in
    alpha(t)=alpha_0/t^lambda -- lambda has essentially nothing to
    modulate when alpha_0 is that small, which is why the lambda-vs-decay
    relationship was floor-limited at the old bounds (confirmed directly:
    restricting to alpha_0>0.2 alone recovered r=0.36-0.54 from r=0.16-0.38
    on the unrestricted draw). n_neurons raised to 500-1500 for the same
    reason sigma-related panels needed n_neurons>=500 to show a clean
    signal -- see docs/HISTORY.md for the full investigation. Seeded by
    virtual_pid so re-running --mode run for the same pid always
    reproduces the same draw, matching this project's own established
    convention elsewhere (e.g. fitting.fit._cross_validate's
    RandomState(seed=pid)).
    """
    rng = np.random.RandomState(seed=int(virtual_pid))
    return {
        "alpha_0": float(rng.uniform(0.5, 1.0)),
        "lambda_": float(rng.uniform(0.1, 1.0)),
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
    t_arr = np.arange(n_timesteps) * dt
    rows = []
    for n_idx in range(n_obs):
        t_pe = t_iti + n_idx * t_step + READOUT_OFFSET
        t_resp = t_iti + n_idx * t_step + t_obs
        idx_pe = int(np.clip(np.round(t_pe / dt), 0, n_timesteps - 1))
        # Matches models.NEF._extract_responses EXACTLY -- a small averaging
        # window around the readout time, not a single-point lookup. Found
        # by direct comparison against the canonical _simulate_trial after
        # the decay-metric correlations collapsed on real data: max diff
        # ~0.016 per observation, small individually but large enough
        # relative to |Delta response| decay's own magnitude to matter.
        response = float(np.mean(value_trace[np.abs(t_arr - t_resp) < dt * 3]))
        rows.append({
            "observation": n_idx + 1,
            "pe": float(abs(error_trace[idx_pe, 0] * error_trace[idx_pe, 1])),
            "response": response,
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
    from models.NEF import PARAM_DEFAULTS, _require_activity_map
    from models.counting_integrator import fast_decode

    draw = _synthetic_params(virtual_pid)
    fixed = MODEL_PARAMS[task]["NEF"].get("fixed", {})
    params = {
        **PARAM_DEFAULTS, **fixed,
        "dataset": task, "model_type": "NEF", "pid": int(virtual_pid),
        "alpha_0": draw["alpha_0"], "lambda_": draw["lambda_"],
        "n_neurons": draw["n_neurons"], "n_neurons_counting": draw["n_neurons"],
    }

    # REQUIRED, not optional -- matches models.NEF.run()'s own convention
    # (_require_activity_map) exactly. NEVER fall back to _pretrain() when
    # this file is missing -- see this module's own top-of-file convention
    # note and docs/HISTORY.md for the incident this re-states.
    activity_map = _require_activity_map(draw["n_neurons"], draw["n_neurons"], task)

    trials = _load_synthetic_trials(task, virtual_pid)

    probe_rows, activity_rows, encoder_rows = [], [], []
    for ti, trial_data in enumerate(trials, 1):
        trial_idx = int(trial_data["trial"])
        qid = trial_data["qid"]
        obs_values = np.array(trial_data["values"], dtype=float)
        # Pool JSON values are on the RAW 0-100 scale (task_backend's own
        # native scale) -- NOT the canonical [-1,1] scale NEF expects and
        # data/soltani_numbers.pkl's own "value" column already has
        # (scripts/build_model_inputs.py's build_from_df() applies this
        # exact rescale to real human data upstream, before it ever reaches
        # NEF). Confirmed directly this was missing here: raw pool values
        # (e.g. 1-28) fed straight into NEF saturate its ensembles
        # (radius_e=1.5, radius_v=1.0) almost immediately, producing
        # plausible-looking but meaningless responses -- exactly what
        # nef_obs_values()'s own docstring warns about. Colors' own pool
        # values are already +-1 (blue/red) and must NOT be rescaled again.
        if task == "soltani_numbers":
            obs_values = obs_values / 50.0 - 1.0
        seed = trial_idx + 1  # 1-indexed, matching activity_key_for_trial's own
                              # +1 convention for 0-indexed (soltani-style) trials

        # REQUIRED, not optional -- a missing key means this activity file
        # doesn't cover this virtual pid's trial at all; regenerate rather
        # than silently retraining with mismatched tuning curves. See this
        # module's own top-of-file convention note.
        activity = activity_map.get(seed)
        if activity is None:
            raise KeyError(
                f"No precomputed counting activity for key={seed} "
                f"(dataset={task!r}, virtual_pid={virtual_pid}, trial={trial_idx}). "
                f"The activity file has keys 1..n_trials -- regenerate with a "
                f"larger --n_trials via models/counting_integrator.py "
                f"--precompute_activities."
            )
        decoders = fast_decode(activity, alpha_0=draw["alpha_0"], lambda_=draw["lambda_"])

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


# ── oddball (neural_giant2's per-parameter rows) ──────────────────────────────────────

def _oddball_value_tag(val: float) -> str:
    """Filesystem-safe tag for one numeric value, e.g. 0.2 -> '0p2', -15 -> 'm15'."""
    return f"{val:g}".replace(".", "p").replace("-", "m")


def _oddball_worker(args, cluster_center: float, oddball_deviation: float,
                    sweep_value: float) -> dict:
    """One (cluster_center, oddball_deviation, sweep_value) cell of the
    grid: 3 observations clustered around cluster_center (+-
    --cluster_spread), then one oddball observation at cluster_center +
    oddball_deviation. Returns the per-timestep abs(decoded PE) trace
    (mean over --n_seeds seeds), WINDOWED TO THE 4TH (ODDBALL)
    OBSERVATION ONLY -- excludes its own preceding ITI, since PE during
    that ITI still reflects the 3rd (clustered) observation's tail, not
    the oddball's own response -- plus summary stats (max within that
    window, end-of-window value, and their absolute difference) --
    abs() throughout per instruction.
    """
    cluster_vals = [cluster_center - args.cluster_spread, cluster_center,
                    cluster_center + args.cluster_spread]
    oddball_val = cluster_center + oddball_deviation
    obs_values_raw = np.array(cluster_vals + [oddball_val], dtype=float)
    obs_values = obs_values_raw / 50.0 - 1.0 if args.task == "soltani_numbers" else obs_values_raw

    base_kwargs = dict(alpha_0=args.base_alpha_0, n_neurons=args.base_n_neurons,
                       lambda_=args.base_lambda_)
    base_kwargs[args.sweep_param] = sweep_value
    params = _base_params(args.task, **base_kwargs)

    activity_map = _require_activities(args.task, params["n_neurons"], params["n_neurons_counting"])

    pe_traces = []
    t = None
    for seed in range(args.n_seeds):
        key = _toy_activity_key(seed)
        decoders = _decoders_for_seed(activity_map, key, params["alpha_0"], params["lambda_"])
        result = _simulate_full(params, obs_values, decoders, seed=key)
        pe_traces.append(np.abs(result["pe_product"]))
        t = result["t"]
    pe_mean = np.mean(pe_traces, axis=0)

    # Window to the 4th (oddball) observation's own stimulus window only:
    # [3*t_step + t_iti, 4*t_step] -- i.e. from the end of its own ITI
    # (excluding it) to the end of the trial. n_obs is always exactly 4
    # for this experiment (3 clustered + 1 oddball).
    t_obs_ = float(params["t_obs"])
    t_iti_ = float(params["t_iti"])
    t_step = t_obs_ + t_iti_
    window_start = 3 * t_step + t_iti_
    window_end = 4 * t_step
    mask = (t >= window_start) & (t <= window_end)
    t_window = t[mask] - window_start
    pe_window = pe_mean[mask]

    max_pe = float(np.max(pe_window))
    end_pe = float(pe_window[-1])
    decrease = max_pe - end_pe

    return {
        "cluster_center": cluster_center,
        "oddball_deviation": oddball_deviation,
        "sweep_value": sweep_value,
        "t": t_window,
        "pe": pe_window,
        "max_pe": max_pe,
        "end_pe": end_pe,
        "decrease": decrease,
        "obs_values_raw": obs_values_raw,
        "obs_values": obs_values,
    }


def _n_neurons_snr_worker(args, cluster_center: float, oddball_deviation: float,
                          n_neurons: int, n_neurons_counting: int) -> dict:
    """One (cluster_center, oddball_deviation, n_neurons_pair) cell: the
    two settled SNR DVs (decoded PE within-seed variance, split-half
    spike-population reliability on non-weight-tuned neurons), both
    restricted to the SAME 400-600ms window within the oddball's own
    presentation, averaged across --n_seeds. See run_n_neurons_snr's own
    docstring for the full definitions.
    """
    cluster_vals = [cluster_center - args.cluster_spread, cluster_center,
                    cluster_center + args.cluster_spread]
    oddball_val = cluster_center + oddball_deviation
    obs_values_raw = np.array(cluster_vals + [oddball_val], dtype=float)
    obs_values = obs_values_raw / 50.0 - 1.0 if args.task == "soltani_numbers" else obs_values_raw

    def _bin_counts(spikes: np.ndarray, dt: float, bin_ms: float) -> np.ndarray:
        bin_size = int(round((bin_ms / 1000.0) / dt))
        n_timesteps, n_neurons_ = spikes.shape
        n_bins = n_timesteps // bin_size
        return spikes[:n_bins * bin_size].reshape(n_bins, bin_size, n_neurons_).sum(axis=1) * dt

    def _split_half_corr(counts: np.ndarray, idx: np.ndarray, n_splits: int,
                         rng: np.random.Generator) -> float:
        sub = counts[:, idx]
        n = sub.shape[1]
        if n < 4:
            return float("nan")
        corrs = []
        for _ in range(n_splits):
            perm = rng.permutation(n)
            half = n // 2
            h1 = sub[:, perm[:half]].sum(axis=1)
            h2 = sub[:, perm[half:2 * half]].sum(axis=1)
            if h1.std() > 0 and h2.std() > 0:
                corrs.append(np.corrcoef(h1, h2)[0, 1])
        return float(np.mean(corrs)) if corrs else float("nan")

    rng = np.random.default_rng(0)
    params = _base_params(args.task, args.alpha_0, n_neurons, args.lambda_,
                          n_neurons_counting=n_neurons_counting)
    activity_map = _require_activities(args.task, n_neurons, n_neurons_counting)

    t_obs_ = float(params["t_obs"]); t_iti_ = float(params["t_iti"]); dt = float(params["dt"])
    t_step = t_obs_ + t_iti_

    pe_variances = []
    split_half_rs = []
    for seed in range(args.n_seeds):
        key = _toy_activity_key(seed)
        decoders = _decoders_for_seed(activity_map, key, params["alpha_0"], params["lambda_"])
        result = _simulate_full(params, obs_values, decoders, seed=key)
        t_arr = result["t"]
        pe_trace = np.abs(result["pe_product"])

        m = (t_arr >= 3 * t_step + t_iti_ + 0.4) & (t_arr < 3 * t_step + t_iti_ + 0.6)
        pe_variances.append(float(np.var(pe_trace[m])))

        non_weight_idx = np.where(result["encoders"][:, 0] <= NEURAL_ENCODER_THRESHOLD)[0]
        counts = _bin_counts(result["error_neurons"][m], dt, args.splithalf_bin_ms)
        split_half_rs.append(_split_half_corr(counts, non_weight_idx,
                                               args.splithalf_n_splits, rng))

    split_half_valid = [r for r in split_half_rs if not np.isnan(r)]
    return {
        "cluster_center": cluster_center,
        "oddball_deviation": oddball_deviation,
        "n_neurons": n_neurons,
        "n_neurons_counting": n_neurons_counting,
        "pe_variance_mean": float(np.mean(pe_variances)),
        "pe_variance_per_seed": pe_variances,
        "split_half_r_mean": float(np.mean(split_half_valid)) if split_half_valid else float("nan"),
        "split_half_r_sd": float(np.std(split_half_valid)) if split_half_valid else float("nan"),
        "split_half_r_per_seed": split_half_rs,
        "obs_values_raw": obs_values_raw,
    }


def run_n_neurons_snr(args) -> None:
    """SETTLED design (renamed from the exploratory run_n_neurons_
    convergence -- see archive/scripts/archive_n_neurons_convergence_
    exploration.py and docs/HISTORY.md's own "n_neurons SNR measure
    exploration" entry for the full narrative of how this was arrived
    at): measures TWO SNR DVs for neural_giant2's row 3 (n_neurons),
    both restricted to the SAME 200ms window (t_iti+400ms to
    t_iti+600ms, the established ~0.5s peak-response latency) within
    the oddball's own presentation, for the SAME oddball trial structure
    `oddball` already uses, now over a FULL GRID of --cluster_centers x
    --oddball_deviations (matching `oddball`'s own grid convention,
    since aggregation across this grid isn't decided yet -- every cell
    gets its own point, not pre-averaged) x --n_neurons_pairs:

      1. Decoded PE within-seed variance -- abs(pe_product) trace across
         timesteps within the window, for ONE seed; variance computed
         per seed, reported as the mean across --n_seeds. Already
         established (this session) to decrease monotonically with
         n_neurons, same sign as response variability (sigma_response),
         though with a shallower slope -- response variability reflects
         DRIFT accumulated/amplified from this same noise source across
         every prior observation, while this measures the noise source
         itself at one instant.

      2. Split-half spike-population reliability -- a purely-neural,
         decoder-free complement: bin the window's raw error-population
         spike-impulse array (--splithalf_bin_ms bins), restrict to
         NON-weight-tuned neurons (enc_dim_0 <= NEURAL_ENCODER_THRESHOLD
         -- i.e. the PE-dimension-tuned neurons, per instruction: this
         subpopulation is easier to explain/replicate empirically than
         "weight-tuned," a model-internal concept), randomly split those
         neurons into two halves --splithalf_n_splits times, pool
         (sum) each half's own spike counts per bin, and correlate the
         two halves' pooled time series -- all WITHIN one seed/trial,
         never mixing across seeds. Reported as the mean across --
         n_seeds. Confirmed (this session, via a temporary raw-spike-
         saving pass now retired) to increase monotonically with
         n_neurons on this subpopulation, with shrinking across-seed SD
         too -- both a Fano-factor-based alternative (single-neuron
         statistic, no mechanism to capture the population-averaging
         benefit decoding gets from many independent noisy units) and
         restricting to weight-tuned-only or using ALL neurons instead
         were tried and are documented in docs/HISTORY.md, not reused
         here.

    NOTE: this experiment's own `_simulate_full` calls confirmed these
    spikes come from the ERROR population (net.error.neurons, 2D: weight
    dimension + raw-PE dimension) -- NOT the value population (net.value,
    a separate ensemble holding the running decoded estimate), which this
    experiment does not touch at all.

    Computed ENTIRELY IN MEMORY -- no raw spike arrays are persisted.
    The earlier exploratory version saved full per-seed spike arrays to a
    temporary folder specifically to support trying several different
    candidate measures without rerunning simulations; now that the
    measure is settled, that flexibility is no longer needed, and
    persisting raw spikes at real-panel scale (many more n_neurons
    values x more seeds) would be needlessly large. Matches this
    project's normal convention (raw traces saved only when a script
    needs to support genuinely open-ended downstream analysis).

    Values are on the RAW 0-100 numbers-task scale, matching `oddball`'s
    own convention -- keep --cluster_centers comfortably away from 0 and
    100 (and --oddball_deviations small enough not to push the oddball
    itself close to those edges), same saturation concern `oddball`'s
    own docstring flags.

    Has the SAME --mode run/submit/collect lifecycle `oddball` uses --
    one job per (cluster_center, oddball_deviation, n_neurons_pair) cell,
    matching `oddball`'s own per-cell granularity exactly (a real timing
    check on that experiment found even a modest single-context run
    exceeds a reasonable single local call, and this grid multiplies
    that the same way: n_centers x n_deviations x n_pairs).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _tag3(c, d, pair_str):
        return f"c{_oddball_value_tag(c)}_d{_oddball_value_tag(d)}_p{pair_str.replace(':', '-')}"

    if args.mode == "run":
        if args.cluster_center is None or args.oddball_deviation is None \
                or args.n_neurons_pair is None:
            raise SystemExit("--cluster_center, --oddball_deviation, and --n_neurons_pair "
                             "all required for --mode run")
        n_str, nc_str = args.n_neurons_pair.split(":")
        n_neurons, n_neurons_counting = int(n_str), int(nc_str)
        tag = _tag3(args.cluster_center, args.oddball_deviation, args.n_neurons_pair)
        out_path = OUT_DIR / f"n_neurons_snr_{args.task}_{tag}.pkl"
        if out_path.exists():
            print(f"Already exists: {out_path.name} -- skipping (delete to rerun)")
            return
        result = _n_neurons_snr_worker(args, args.cluster_center, args.oddball_deviation,
                                       n_neurons, n_neurons_counting)
        pd.to_pickle(result, out_path)
        print(f"Saved center={args.cluster_center} deviation={args.oddball_deviation} "
              f"n_neurons={n_neurons} n_neurons_counting={n_neurons_counting}: "
              f"pe_variance={result['pe_variance_mean']:.6f} "
              f"split_half_r={result['split_half_r_mean']:.4f} -> {out_path}")

    elif args.mode == "submit":
        root = str(Path(__file__).resolve().parent.parent)
        combos = [(c, d, p) for c in args.cluster_centers
                 for d in args.oddball_deviations for p in args.n_neurons_pairs]
        print(f"Submitting {len(combos)} n_neurons_snr jobs "
              f"({len(args.cluster_centers)} centers x {len(args.oddball_deviations)} "
              f"deviations x {len(args.n_neurons_pairs)} n_neurons pairs) for task={args.task}")
        for c, d, p in combos:
            tag = _tag3(c, d, p)
            out_path = OUT_DIR / f"n_neurons_snr_{args.task}_{tag}.pkl"
            if out_path.exists():
                print(f"  center={c} deviation={d} pair={p}: already exists -- skipping")
                continue
            cmd = (
                f"venv/bin/python scripts/neural_experiments.py n_neurons_snr "
                f"--task {args.task} --mode run --n_neurons_pair {p} "
                f"--cluster_center {c} --oddball_deviation {d} "
                f"--cluster_spread {args.cluster_spread} "
                f"--alpha_0 {args.alpha_0} --lambda_ {args.lambda_} --n_seeds {args.n_seeds} "
                f"--splithalf_bin_ms {args.splithalf_bin_ms} "
                f"--splithalf_n_splits {args.splithalf_n_splits}"
            )
            script = make_job_script(root, [cmd], time_limit="1:0:0", mem="16G")
            script_path = OUT_DIR / f"_job_n_neurons_snr_{args.task}_{tag}.sh"
            script_path.write_text(script)
            submit_script(script_path, dry_run=args.dry_run)

    elif args.mode == "collect":
        files = sorted(OUT_DIR.glob(f"n_neurons_snr_{args.task}_c*_d*_p*.pkl"))
        if not files:
            print(f"No n_neurons_snr_{args.task}_c*_d*_p*.pkl files found in {OUT_DIR}")
            return
        results = [pd.read_pickle(f) for f in files]
        grid = pd.DataFrame([
            {"cluster_center": r["cluster_center"], "oddball_deviation": r["oddball_deviation"],
             "n_neurons": r["n_neurons"], "n_neurons_counting": r["n_neurons_counting"],
             "pe_variance_mean": r["pe_variance_mean"], "split_half_r_mean": r["split_half_r_mean"],
             "split_half_r_sd": r["split_half_r_sd"]}
            for r in results
        ]).sort_values(["cluster_center", "oddball_deviation", "n_neurons"]).reset_index(drop=True)
        out_path = OUT_DIR / f"n_neurons_snr_{args.task}.pkl"
        pd.to_pickle({"grid": grid, "results": results, "task": args.task}, out_path)
        print(f"Collected {len(files)} cell(s) -> {out_path}")
        print(grid)


def run_oddball(args) -> None:
    """Toy demo: 3 observations clustered around a center, then one
    "oddball" observation deviating from it -- across a full grid of
    (--cluster_centers x --oddball_deviations x --sweep_values), the other
    two of {alpha_0, lambda_, n_neurons} held fixed at --base_alpha_0/
    --base_lambda_/--base_n_neurons. Averaged over --n_seeds seeds per
    cell. Tests directly whether the response to a fixed-magnitude
    surprise is independent of where the cluster sits (the person's own
    prediction), rather than assuming it.

    Values are on the RAW 0-100 numbers-task scale -- rescaled via the
    exact same x/50-1 transform scripts/build_model_inputs.py applies to
    real human data before NEF ever sees it. Colors' own values are
    already +-1 and would NOT need this. Keep --cluster_centers
    comfortably away from 0 and 100 (and --oddball_deviations small
    enough not to push the oddball itself close to those edges either) --
    values near the edges of the rescaled [-1,1] range risk exactly the
    saturation this grid is partly designed to detect as a confound, not
    a genuine center effect.

    Has the same --mode run/submit/collect lifecycle as probe/synthetic --
    a real timing check found even a modest single-context run exceeds a
    reasonable single local call, and the full grid multiplies that by
    n_centers x n_deviations. One job per (cluster_center,
    oddball_deviation, sweep_value) triple; --n_seeds worth of
    simulations run serially within that one job.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _tag3(c, d, v):
        return f"c{_oddball_value_tag(c)}_d{_oddball_value_tag(d)}_v{_oddball_value_tag(v)}"

    if args.mode == "run":
        if args.cluster_center is None or args.oddball_deviation is None or args.sweep_value is None:
            raise SystemExit("--cluster_center, --oddball_deviation, and --sweep_value "
                             "all required for --mode run")
        tag = _tag3(args.cluster_center, args.oddball_deviation, args.sweep_value)
        out_path = OUT_DIR / f"oddball_{args.sweep_param}_{args.task}_{tag}.pkl"
        if out_path.exists():
            print(f"Already exists: {out_path.name} -- skipping (delete to rerun)")
            return
        result = _oddball_worker(args, args.cluster_center, args.oddball_deviation, args.sweep_value)
        pd.to_pickle(result, out_path)
        print(f"Saved center={args.cluster_center} deviation={args.oddball_deviation} "
              f"{args.sweep_param}={args.sweep_value}: max_pe={result['max_pe']:.4f} "
              f"end_pe={result['end_pe']:.4f} decrease={result['decrease']:.4f} "
              f"-> {out_path}")

    elif args.mode == "submit":
        root = str(Path(__file__).resolve().parent.parent)
        combos = [(c, d, v) for c in args.cluster_centers
                 for d in args.oddball_deviations for v in args.sweep_values]
        print(f"Submitting {len(combos)} oddball jobs "
              f"({len(args.cluster_centers)} centers x {len(args.oddball_deviations)} "
              f"deviations x {len(args.sweep_values)} {args.sweep_param} values) "
              f"for task={args.task}")
        for c, d, v in combos:
            tag = _tag3(c, d, v)
            out_path = OUT_DIR / f"oddball_{args.sweep_param}_{args.task}_{tag}.pkl"
            if out_path.exists():
                print(f"  center={c} deviation={d} {args.sweep_param}={v}: already exists -- skipping")
                continue
            cmd = (
                f"venv/bin/python scripts/neural_experiments.py oddball "
                f"--task {args.task} --mode run --sweep_param {args.sweep_param} "
                f"--sweep_value {v} --cluster_center {c} --oddball_deviation {d} "
                f"--cluster_spread {args.cluster_spread} "
                f"--base_alpha_0 {args.base_alpha_0} --base_lambda_ {args.base_lambda_} "
                f"--base_n_neurons {args.base_n_neurons} --n_seeds {args.n_seeds}"
            )
            script = make_job_script(root, [cmd], time_limit="1:0:0", mem="16G")
            script_path = OUT_DIR / f"_job_oddball_{args.sweep_param}_{args.task}_{tag}.sh"
            script_path.write_text(script)
            submit_script(script_path, dry_run=args.dry_run)

    elif args.mode == "collect":
        files = sorted(OUT_DIR.glob(f"oddball_{args.sweep_param}_{args.task}_c*_d*_v*.pkl"))
        if not files:
            print(f"No oddball_{args.sweep_param}_{args.task}_c*_d*_v*.pkl files found in {OUT_DIR}")
            return
        results = [pd.read_pickle(f) for f in files]
        grid = pd.DataFrame([
            {"cluster_center": r["cluster_center"], "oddball_deviation": r["oddball_deviation"],
             args.sweep_param: r["sweep_value"], "max_pe": r["max_pe"], "end_pe": r["end_pe"],
             "decrease": r["decrease"]}
            for r in results
        ]).sort_values(["cluster_center", "oddball_deviation", args.sweep_param]).reset_index(drop=True)
        traces = {
            (r["cluster_center"], r["oddball_deviation"], r["sweep_value"]): {"t": r["t"], "pe": r["pe"]}
            for r in results
        }
        result_all = {
            "grid": grid,
            "traces": traces,
            "sweep_param": args.sweep_param,
            "base_alpha_0": args.base_alpha_0,
            "base_lambda_": args.base_lambda_,
            "base_n_neurons": args.base_n_neurons,
            "cluster_spread": args.cluster_spread,
        }
        out_path = OUT_DIR / f"oddball_{args.sweep_param}_{args.task}.pkl"
        pd.to_pickle(result_all, out_path)
        print(f"Collected {len(files)} cell(s) -> {out_path}")
        print(grid)


# ── param_scan (neural_giant2 rows 2/3 -- lambda_/n_neurons vs activity+decay) ──

NEURAL_ENCODER_THRESHOLD = 0.5  # matches figure_yoo_neural.py's/make_paper_figures.py's
                                # own ENCODER_THRESHOLD -- same weight-tuned-neuron
                                # convention as the original neural_giant figure.


def _draw_sweep_value(sweep_param: str, trial_source: str, pid: int, low: float, high: float) -> float:
    """Deterministic per-replicate uniform draw in [low, high] for
    sweep_param, seeded by (trial_source, sweep_param, pid) -- trial_source
    included so a real pid and a virtual_pid that happen to share the same
    integer never accidentally draw the same value by coincidence; those
    are two different id namespaces ('real' pids from soltani_numbers.pkl,
    'synthetic' virtual_pids 1..len(pool)), never mixed within one
    collected file, but kept seed-distinct regardless. Each replicate
    (real pid OR synthetic virtual_pid) gets its OWN single value, rather
    than every replicate seeing every value from a shared explicit grid --
    the earlier (sweep_value, pid) full-cross-product design produced an
    unnatural-looking discrete-strip spacing in the decay-vs-param
    regression panels, since every replicate landed on the same handful
    of x-positions; a per-replicate random draw gives a genuinely
    continuous x-axis instead, matching how `synthetic`'s own virtual-pid
    parameter draws already work.

    Uses zlib.crc32 on an explicit byte string, NOT Python's built-in
    hash() -- hash() on str is randomized per-process (PYTHONHASHSEED),
    so seeding from it would silently draw a DIFFERENT value for the
    same replicate on every separate cluster job/rerun, defeating the
    whole point of a deterministic draw.
    """
    import zlib
    key = f"{trial_source}:{sweep_param}:{int(pid)}".encode()
    rng = np.random.default_rng(zlib.crc32(key))
    return float(rng.uniform(low, high))


def _simulate_param_scan_trial(params: dict, obs_values: np.ndarray, decoders: dict,
                               seed: int = 0) -> dict:
    """Build and run one arbitrary trial (same shape as _simulate_full),
    but ALSO captures this trial's own error-ensemble encoders --
    needed to identify weight-tuned neurons (enc_dim_0 > threshold) per
    (sweep_value, seed), since net.error is built with seed=seed directly
    and its encoders genuinely differ per seed (the same fact that made
    per-trial encoders necessary for the `synthetic` experiment's own
    _simulate_trial_full applies here too -- see that function's own
    docstring). Returns per-timestep t/value/error_neurons (raw,
    synapse=None, matching _simulate_full's own convention) plus the
    (n_neurons, 2) encoders array.
    """
    import nengo
    from models.NEF import build_network

    p = {**params, "seed": int(seed)}
    net = build_network(obs_values, p, decoders)
    dt = float(p["dt"])
    n_obs = len(obs_values)
    t_total = n_obs * (float(p["t_obs"]) + float(p["t_iti"]))

    with nengo.Simulator(net, dt=dt, seed=int(seed), progress_bar=False) as sim:
        sim.run(t_total)
        encoders = np.array(sim.data[net.error].encoders, copy=True)

    t_arr = np.arange(len(sim.data[net.probe_value])) * dt
    return {
        "t": t_arr,
        "value": sim.data[net.probe_value].squeeze(),
        "error_neurons": sim.data[net.probe_error_neurons],  # (T, n_neurons), raw
        "encoders": encoders,
    }


def _param_scan_worker(args, sweep_value: float, pid: int) -> pd.DataFrame:
    """One sweep_value's worth of simulations for ONE replicate's own
    trials, the other two of {alpha_0, lambda_, n_neurons} held fixed at
    --base_*. --trial_source selects where those trials come from:

      'real' (default): pid indexes a REAL participant in
        data/soltani_numbers.pkl (TASK_DATAFILE[args.task] generally);
        uses that pid's own 32 real trials AS-IS (soltani_numbers.pkl's
        own 'value' column is already on NEF's canonical [-1,1] scale --
        see build_model_inputs.py's build_from_df(), matching
        _probe_worker's own convention exactly, no rescale here).

      'synthetic': pid indexes a virtual_pid (1..len(pool)) in the
        pre-generated 200-member pool (data/synthetic_pool/, matching
        `synthetic`'s own trial-count-per-member structure); uses that
        virtual_pid's own 32 generated trials. Pool JSON values are on
        the RAW 0-100 scale, NOT canonical [-1,1] -- MUST be rescaled
        (/50-1 for soltani_numbers) before reaching NEF, exactly matching
        _synthetic_worker's own convention (see that function's own
        docstring for the saturation bug this avoids). Lets N grow past
        the ~46 real pids available, per instruction.

    Both sources use models.counting_integrator.activity_key_for_trial
    (dataset, trial) for BOTH the activity-map lookup and the simulation
    seed -- the exact convention _probe_worker/_synthetic_worker already
    use; both trial-index spaces are 0..31 (0-indexed, +1 offset), well
    within the activity file's own 40 precomputed keys for soltani_numbers
    (_DATASET_N_TRIALS), so no extra precompute is needed for either.

    CORRECTED earlier this session: an original version of this function
    simulated a single arbitrary constant-input trial (obs_values=
    ones(n_obs)) across --n_seeds independent seeds, per Act 1.2's own
    toy-demo convention -- appropriate for the ORIGINAL neural_giant's
    illustrative lambda panel, but never actually reconsidered for THIS
    use (a quantitative decay-vs-parameter regression), where a flat,
    unchanging input gives the value ensemble nothing to integrate and
    produces an activity trend that is an artefact of the degenerate
    input rather than a real lambda-driven signature.

    For each trial, reduces the error population's raw per-timestep
    activity to the WEIGHT-TUNED neurons' own mean (per THAT TRIAL's own
    error-ensemble encoders, enc_dim_0 > NEURAL_ENCODER_THRESHOLD --
    encoders genuinely differ per seed/trial, so must be captured per
    trial, not once per replicate) -- matching the ORIGINAL neural_giant
    figure's own weight-tuned-neuron convention.

    Job granularity is (sweep_value, pid) -- one job per replicate per
    sweep_value, each running that replicate's 32 trials serially,
    matching _probe_worker's own per-pid job granularity.

    REQUIRED activity file, no _pretrain() fallback -- see this module's
    own top-of-file convention note.
    """
    from models.counting_integrator import activity_key_for_trial, fast_decode

    base_kwargs = dict(alpha_0=args.base_alpha_0, n_neurons=args.base_n_neurons,
                       lambda_=args.base_lambda_)
    base_kwargs[args.sweep_param] = sweep_value
    params = _base_params(args.task, **base_kwargs)

    activity_map = _require_activities(args.task, params["n_neurons"], params["n_neurons_counting"])

    if args.trial_source == "real":
        dataset_stem = TASK_DATAFILE[args.task]
        human = pd.read_pickle(data_path(dataset_stem))
        human_pid = human[human["pid"] == pid]
        if human_pid.empty:
            raise ValueError(f"No human data for pid={pid} in {dataset_stem}")
        trial_iter = [
            (int(trial), trial_data.sort_values("observation")["value"].to_numpy(dtype=float))
            for trial, trial_data in human_pid.groupby("trial")
        ]
    else:  # synthetic
        pool_trials = _load_synthetic_trials(args.task, pid)
        trial_iter = []
        for trial_data in pool_trials:
            obs_values = np.array(trial_data["values"], dtype=float)
            # RAW 0-100 pool scale -- rescale for soltani_numbers, matching
            # _synthetic_worker's own convention exactly. Colors' own pool
            # values are already +-1 and must NOT be rescaled again.
            if args.task == "soltani_numbers":
                obs_values = obs_values / 50.0 - 1.0
            trial_iter.append((int(trial_data["trial"]), obs_values))

    rows = []
    n_trials = len(trial_iter)
    for ti, (trial, obs_values) in enumerate(trial_iter, 1):
        akey = activity_key_for_trial(args.task, trial)

        # REQUIRED, not optional -- a missing key means this activity file
        # doesn't cover this trial at all; regenerate rather than
        # silently retraining with mismatched tuning curves. See this
        # module's own top-of-file convention note.
        activity = activity_map.get(akey)
        if activity is None:
            raise KeyError(
                f"No precomputed counting activity for key={akey} "
                f"(dataset={args.task!r}, trial_source={args.trial_source!r}, "
                f"pid={pid}, trial={trial}). The activity file has keys "
                f"1..n_trials -- regenerate with --precompute_activities if "
                f"this dataset's real/synthetic trial count has grown."
            )
        decoders = fast_decode(activity, alpha_0=params["alpha_0"], lambda_=params["lambda_"])

        result = _simulate_param_scan_trial(params, obs_values, decoders, seed=akey)

        weight_idx = np.where(result["encoders"][:, 0] > NEURAL_ENCODER_THRESHOLD)[0]
        if len(weight_idx) == 0:
            print(f"\n  Warning: pid={pid} trial={trial} {args.sweep_param}={sweep_value}: "
                  f"no weight-tuned neurons found (enc_dim_0 > {NEURAL_ENCODER_THRESHOLD}), "
                  f"skipping this trial")
            continue
        weight_tuned_activity = result["error_neurons"][:, weight_idx].mean(axis=1)

        rows.append(pd.DataFrame({
            "sweep_value": sweep_value,
            "pid": pid,
            "trial": trial,
            "t": result["t"],
            "value_decoded": result["value"],
            "weight_tuned_activity": weight_tuned_activity,
            "n_weight_tuned": len(weight_idx),
        }))
        print(f"\r  pid={pid} ({args.trial_source}) {args.sweep_param}={sweep_value}  "
              f"trial {ti:3d}/{n_trials}", end="", flush=True)
    print()

    if not rows:
        raise RuntimeError(
            f"pid={pid}, {args.sweep_param}={sweep_value}: every trial hit the "
            f"enc_dim_0 > {NEURAL_ENCODER_THRESHOLD} threshold with zero matching "
            f"weight-tuned neurons; investigate before trusting this cell."
        )
    return pd.concat(rows, ignore_index=True)


def run_param_scan(args) -> None:
    """Scan ONE of {alpha_0, lambda_, n_neurons} over trials from EITHER
    --trial_source: real soltani_numbers pids (default) or the synthetic
    200-member pool (see _param_scan_worker's own docstring for both
    conventions) -- each replicate gets its OWN single value of
    sweep_param, drawn uniformly from [--sweep_low, --sweep_high]
    (deterministic per replicate, see _draw_sweep_value's own docstring),
    rather than every replicate seeing every value from a shared explicit
    grid. The other two of {alpha_0, lambda_, n_neurons} stay fixed at
    --base_*.

    Job granularity is (pid) alone -- --mode run simulates ONE replicate's
    trials at that replicate's own drawn value; --mode submit loops over
    every replicate in the chosen --trial_source (one job per replicate).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "run":
        if args.pid is None:
            raise SystemExit("--pid required for --mode run")
        if args.sweep_value is not None:
            sweep_value = args.sweep_value  # explicit override, e.g. for spot-checking
        else:
            if args.sweep_low is None or args.sweep_high is None:
                raise SystemExit("--sweep_low/--sweep_high required for --mode run "
                                 "(unless --sweep_value is given explicitly)")
            sweep_value = _draw_sweep_value(args.sweep_param, args.trial_source, args.pid,
                                            args.sweep_low, args.sweep_high)
        tag = _oddball_value_tag(sweep_value)
        out_path = OUT_DIR / (f"param_scan_{args.sweep_param}_{args.task}_"
                              f"{args.trial_source}_v{tag}_pid{args.pid}.pkl")
        if out_path.exists():
            print(f"Already exists: {out_path.name} -- skipping (delete to rerun)")
            return
        df = _param_scan_worker(args, sweep_value, args.pid)
        df.to_pickle(out_path)
        print(f"Saved {len(df):,} rows ({df['trial'].nunique()} trials, "
              f"{args.sweep_param}={sweep_value:.4f}) -> {out_path}")

    elif args.mode == "submit":
        if args.sweep_low is None or args.sweep_high is None:
            raise SystemExit("--sweep_low/--sweep_high required for --mode submit")
        if args.trial_source == "real":
            dataset_stem = TASK_DATAFILE[args.task]
            human = pd.read_pickle(data_path(dataset_stem))
            pids = sorted(human["pid"].unique().tolist())
        else:
            import json
            with open(_synthetic_pool_path(args.task)) as f:
                pool = json.load(f)
            pids = list(range(1, len(pool) + 1))
        root = str(Path(__file__).resolve().parent.parent)
        print(f"Submitting {len(pids)} param_scan jobs (one per {args.trial_source} "
              f"replicate, {args.sweep_param} ~ Uniform({args.sweep_low}, {args.sweep_high})) "
              f"for task={args.task}")
        for pid in pids:
            existing = list(OUT_DIR.glob(
                f"param_scan_{args.sweep_param}_{args.task}_{args.trial_source}_v*_pid{pid}.pkl"))
            if existing:
                print(f"  pid={pid}: already exists ({existing[0].name}) -- skipping")
                continue
            cmd = (
                f"venv/bin/python scripts/neural_experiments.py param_scan "
                f"--task {args.task} --mode run --sweep_param {args.sweep_param} "
                f"--trial_source {args.trial_source} "
                f"--pid {pid} --sweep_low {args.sweep_low} --sweep_high {args.sweep_high} "
                f"--base_alpha_0 {args.base_alpha_0} --base_lambda_ {args.base_lambda_} "
                f"--base_n_neurons {args.base_n_neurons}"
            )
            script = make_job_script(root, [cmd], time_limit="2:0:0", mem="16G")
            script_path = OUT_DIR / f"_job_param_scan_{args.sweep_param}_{args.task}_{args.trial_source}_pid{pid}.sh"
            script_path.write_text(script)
            submit_script(script_path, dry_run=args.dry_run)

    elif args.mode == "collect":
        files = sorted(OUT_DIR.glob(
            f"param_scan_{args.sweep_param}_{args.task}_{args.trial_source}_v*_pid*.pkl"))
        if not files:
            print(f"No param_scan_{args.sweep_param}_{args.task}_{args.trial_source}_v*_pid*.pkl "
                  f"files found in {OUT_DIR}")
            return
        df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
        out_path = OUT_DIR / f"param_scan_{args.sweep_param}_{args.task}.pkl"
        df.to_pickle(out_path)
        print(f"Collected {len(files)} pid file(s) ({args.trial_source}) -> {out_path}")
        print(df.groupby("pid")["sweep_value"].first().describe())


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

    p_ndemo = sub.add_parser("n_neurons_demo")
    p_ndemo.add_argument("--task", required=True)
    p_ndemo.add_argument("--n_neurons_pairs", type=str, nargs="+", required=True,
                        help="'n_neurons:n_neurons_counting' pairs, e.g. 50:200 200:800 500:2000")
    p_ndemo.add_argument("--alpha_0", type=float, default=0.7)
    p_ndemo.add_argument("--lambda_", type=float, default=0.7)
    p_ndemo.add_argument("--obs_values", type=float, nargs="+", default=[30, 70, 40, 80],
                        help="Raw 0-100 scale toy sequence, rescaled internally for soltani_numbers.")
    p_ndemo.add_argument("--n_seeds", type=int, default=10,
                        help="Independent toy seeds PER PAIR -- a SMALL number, deliberately "
                             "not enough to fully average away spike noise (see this "
                             "experiment's own docstring).")
    p_ndemo.set_defaults(func=run_n_neurons_demo)

    p_conv = sub.add_parser("n_neurons_snr")
    p_conv.add_argument("--task", required=True)
    p_conv.add_argument("--mode", required=True, choices=["run", "submit", "collect"])
    p_conv.add_argument("--n_neurons_pair", type=str, default=None,
                       help="Single 'n_neurons:n_neurons_counting' pair for --mode run.")
    p_conv.add_argument("--n_neurons_pairs", type=str, nargs="+", default=None,
                       help="Full list of pairs for --mode submit, e.g. 50:200 100:400 200:800")
    p_conv.add_argument("--cluster_center", type=float, default=None,
                       help="Single center for --mode run.")
    p_conv.add_argument("--cluster_centers", type=float, nargs="+", default=None,
                       help="Full grid of centers for --mode submit, e.g. 20 35 50 65 80")
    p_conv.add_argument("--cluster_spread", type=float, required=True)
    p_conv.add_argument("--oddball_deviation", type=float, default=None,
                       help="Single deviation for --mode run.")
    p_conv.add_argument("--oddball_deviations", type=float, nargs="+", default=None,
                       help="Full grid of deviations for --mode submit, e.g. -10 10")
    p_conv.add_argument("--alpha_0", type=float, default=0.7)
    p_conv.add_argument("--lambda_", type=float, default=0.7)
    p_conv.add_argument("--n_seeds", type=int, default=10)
    p_conv.add_argument("--splithalf_bin_ms", type=float, default=20.0,
                       help="Bin width (ms) for the split-half spike-count bins.")
    p_conv.add_argument("--splithalf_n_splits", type=int, default=50,
                       help="Random half-splits averaged per seed for a stable estimate.")
    p_conv.add_argument("--dry_run", action="store_true")
    p_conv.set_defaults(func=run_n_neurons_snr)

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

    p_odd = sub.add_parser("oddball")
    p_odd.add_argument("--task", required=True)
    p_odd.add_argument("--mode", required=True, choices=["run", "submit", "collect"])
    p_odd.add_argument("--sweep_param", required=True, choices=["alpha_0", "lambda_", "n_neurons"])
    p_odd.add_argument("--sweep_value", type=float, default=None,
                       help="Single value for --mode run (one cluster job per grid cell).")
    p_odd.add_argument("--sweep_values", type=float, nargs="+", default=None,
                       help="Full list of values for --mode submit.")
    p_odd.add_argument("--cluster_center", type=float, default=None,
                       help="Single center for --mode run (one cluster job per grid cell).")
    p_odd.add_argument("--cluster_centers", type=float, nargs="+", default=None,
                       help="Raw (0-100 scale) cluster centers to test, e.g. 20 40 60 80. "
                            "Keep comfortably away from 0/100 -- see run_oddball's own docstring.")
    p_odd.add_argument("--cluster_spread", type=float, required=True,
                       help="+- offset for the 3 clustered observations around each center, "
                            "e.g. 1 gives (center-1, center, center+1).")
    p_odd.add_argument("--oddball_deviation", type=float, default=None,
                       help="Single deviation for --mode run (one cluster job per grid cell).")
    p_odd.add_argument("--oddball_deviations", type=float, nargs="+", default=None,
                       help="Signed deviations from each center to test, e.g. -15 -10 10 15.")
    p_odd.add_argument("--base_alpha_0", type=float, required=True)
    p_odd.add_argument("--base_lambda_", type=float, required=True)
    p_odd.add_argument("--base_n_neurons", type=int, required=True)
    p_odd.add_argument("--n_seeds", type=int, default=20)
    p_odd.add_argument("--dry_run", action="store_true")
    p_odd.set_defaults(func=run_oddball)

    p_scan = sub.add_parser("param_scan")
    p_scan.add_argument("--task", required=True)
    p_scan.add_argument("--mode", required=True, choices=["run", "submit", "collect"])
    p_scan.add_argument("--sweep_param", required=True, choices=["alpha_0", "lambda_", "n_neurons"])
    p_scan.add_argument("--trial_source", default="real", choices=["real", "synthetic"],
                       help="'real': soltani_numbers pids (~46 replicates). 'synthetic': "
                            "the pre-generated 200-member pool (more replicates, same "
                            "per-member trial-count structure).")
    p_scan.add_argument("--sweep_value", type=float, default=None,
                       help="Explicit override for --mode run (skips the random draw).")
    p_scan.add_argument("--sweep_low", type=float, default=None,
                       help="Lower bound for the per-replicate uniform draw (--mode run/submit).")
    p_scan.add_argument("--sweep_high", type=float, default=None,
                       help="Upper bound for the per-replicate uniform draw (--mode run/submit).")
    p_scan.add_argument("--pid", type=int, default=None,
                       help="Real pid, or synthetic virtual_pid (1..len(pool)) -- --mode run.")
    p_scan.add_argument("--base_alpha_0", type=float, required=True)
    p_scan.add_argument("--base_lambda_", type=float, required=True)
    p_scan.add_argument("--base_n_neurons", type=int, required=True)
    p_scan.add_argument("--dry_run", action="store_true")
    p_scan.set_defaults(func=run_param_scan)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
