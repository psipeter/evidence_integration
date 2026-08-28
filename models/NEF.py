#!/usr/bin/env python3
"""
NEF model of evidence integration.

Supports **carrabin** and **yoo**: sequential scalar ``value`` inputs per
observation.

Architecture (per trial):
    counting subnetwork (LMU or integrator, pretrained decoders)
    counting → error[dim 0]   (alpha(n) via W_weight decoder)
    node_input[0] → error[dim 1]   (observation o(t))
    node_input[1] → error.neurons  (ITI inhibition)
    value → error[dim 1]      (transform=-1, subtracts v)

Recurrent value dynamics: multiplicative error→value connection and recurrent
self-connection on ``value``.

Usage:
    from models.NEF import run
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

from models.counting_integrator import (
    activity_key_for_trial,
    build_network as build_counting_integrator,
    decode_outputs as decode_counting_integrator,
    fast_decode as fast_decode_counting,
    load_activities as load_counting_activities,
    load_decoders as load_counting_decoders,
    simulate_network as simulate_counting_integrator,
)
from utils.paths import data_path, dataset_stem

from fitting.model_params import _NEF_FIXED
from utils.binary_transform import (
    apply_binary_transform,
    nef_obs_values,
    nef_response_to_model_scale,
)

PARAM_DEFAULTS: dict = {
    **_NEF_FIXED,
    "n_obs": 30,
    "lambda_": 0.5,
    "alpha_0": 1,
    "T_error": 0.5,
    "tau_error": 0.1,
}


def _make_input(obs_values: np.ndarray, params: dict) -> callable:
    """Returns [obs(t), inh(t)] where inh=1 during ITI, 0 during observation."""
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    n_obs = len(obs_values)

    def fn(t: float) -> list[float]:
        if t < t_iti:
            return [0.0, 1.0]
        step = int((t - t_iti) / t_step)
        phase = (t - t_iti) - step * t_step
        if step < n_obs and phase < t_obs:
            return [float(obs_values[step]), 0.0]
        return [0.0, 1.0]

    return fn


def _extract_responses(
    t_arr: np.ndarray,
    value_decoded: np.ndarray,
    n_obs: int,
    params: dict,
) -> np.ndarray:
    """Compute readout times and return averaged responses."""
    t_obs = params["t_obs"]
    t_iti = params["t_iti"]
    t_step = t_obs + t_iti
    readout_times = np.array([t_iti + i * t_step + t_obs for i in range(n_obs)])
    return np.array(
        [
            float(np.mean(value_decoded[np.abs(t_arr - rt) < params["dt"] * 3]))
            for rt in readout_times
        ]
    )


def _pretrain(params: dict) -> dict:
    """Run counting pretraining for the integrator subnetwork."""
    seed = int(params.get("seed", 0))
    p = {**params, "n_obs": params["radius_c"], "seed": seed}
    net = build_counting_integrator(p, train=True)
    raw = simulate_counting_integrator(net, p, train=True)
    return decode_counting_integrator(raw, p)


# Ballpark default for NEF's NLL ensemble size (see docs/HISTORY.md), from
# CHEAP-MODEL calibration (NoisyRL_lambda proxy, scripts/calibrate_nll_nsims.py)
# rather than a direct NEF measurement -- NEF's own noise magnitude/mechanism
# hasn't been checked against this number yet, so treat it as a starting
# point to raise later for a more exact estimate, not a validated final
# answer the way fitting/fit.py's own n_sims=100 is documented to be for
# NoisyRL_lambda's sigma_resp. Deliberately NOT wired into fitting.fit's own
# shared --n_sims CLI default (that flag applies to every --loss nll model
# type and stays at its own validated default for NoisyRL_lambda/
# _resp_noise) -- pass --n_sims 50 explicitly when fitting NEF under NLL.
NEF_DEFAULT_N_SIMS = 50


def _require_activity_map(
    n_neurons: int, n_neurons_counting: int, dataset: str, n_sims: int = 1,
) -> dict:
    """Load precomputed counting activities, or fail with the exact command
    to generate them. REQUIRED, not optional -- see run()'s own inline
    comment (and docs/HISTORY.md) for why falling back to _pretrain() here
    is a genuine seed-mismatch bug, not just a slow path. Shared by run()
    (n_sims=1, one seed per trial -- a single point-estimate response) and
    simulate_ensemble() (n_sims>1, n_trials*n_sims seeds -- a genuine
    ensemble for a distributional loss) so the two paths' requirements can
    never drift apart.
    """
    try:
        return load_counting_activities(
            n_neurons=n_neurons, n_neurons_counting=n_neurons_counting, dataset=dataset,
        )
    except FileNotFoundError as e:
        cmd_sims = f" --n_sims {n_sims}" if n_sims > 1 else ""
        raise FileNotFoundError(
            f"No precomputed counting-activity file for "
            f"(n_neurons={n_neurons}, n_neurons_counting={n_neurons_counting}, "
            f"dataset={dataset!r}, n_sims={n_sims}). This file is REQUIRED -- "
            f"NEF never falls back to _pretrain(). Generate it first:\n"
            f"  venv/bin/python models/counting_integrator.py "
            f"--precompute_activities --n_neurons {n_neurons} "
            f"--n_neurons_counting {n_neurons_counting} --dataset {dataset}"
            f"{cmd_sims}\n"
            f"then scp data/counting_activities_n{n_neurons}_nc"
            f"{n_neurons_counting}_{dataset}.pkl to the cluster if fitting remotely."
        ) from e


def build_network(
    obs_values: np.ndarray,
    params: dict,
    decoders: dict,
) -> nengo.Network:
    seed = int(params["seed"])
    tau_fb = float(params["tau_fb"])
    T_error = float(params["T_error"])

    _build_c = build_counting_integrator

    with nengo.Network(label=str(params["model_type"]), seed=seed) as net:
        net.node_input = nengo.Node(
            _make_input(obs_values, params), size_out=2, label="node_input"
        )

        # Counting uses n_neurons_counting for memory and n_neurons for
        # onset_detector (error and value use this n_neurons only).
        c_params = {
            **params,
            "n_obs": int(params["radius_c"]),
        }
        net.counting = _build_c(c_params, train=False, decoders=decoders)
        # probe counting weight and count decoded outputs
        net.probe_counting_weight = nengo.Probe(
            net.counting.weight_out,
            synapse=float(params["tau_probe"]),
            sample_every=float(params["dt"]),
        )
        net.probe_counting_count = nengo.Probe(
            net.counting.count_out,
            synapse=float(params["tau_probe"]),
            sample_every=float(params["dt"]),
        )

        net.error = nengo.Ensemble(
            n_neurons=int(params["n_neurons"]),
            dimensions=2,
            radius=float(params["radius_e"]),
            seed=seed,
            label="error",
        )
        nengo.Connection(
                net.counting.memory.neurons,
                net.error[0],
                transform=decoders["W_weight"],
                synapse=float(params["tau_ff"]),
                seed=seed,
            )

        nengo.Connection(net.node_input[0], net.error[1], synapse=None, seed=seed)
        w_inh = -10.0 * np.ones((net.error.n_neurons, 1))
        nengo.Connection(
            net.node_input[1],
            net.error.neurons,
            transform=w_inh,
            synapse=float(params["tau_error"]),
            seed=seed,
        )

        net.value = nengo.Ensemble(
            n_neurons=int(params["n_neurons"]),
            dimensions=1,
            radius=float(params["radius_v"]),
            seed=seed,
            label="value",
        )

        nengo.Connection(
            net.value,
            net.error[1],
            transform=-1,
            synapse=float(params["tau_ff"]),
            seed=seed,
        )

        nengo.Connection(
            net.error,
            net.value,
            function=lambda x: x[0] * x[1],
            transform=T_error,
            synapse=tau_fb,
            seed=seed,
        )
        nengo.Connection(
            net.value,
            net.value,
            synapse=tau_fb,
            seed=seed,
        )

        net.probe_value = nengo.Probe(
            net.value,
            synapse=float(params["tau_probe"]),
            sample_every=float(params["dt"]),
        )
        net.probe_error = nengo.Probe(
            net.error,
            synapse=float(params["tau_probe"]),
            sample_every=float(params["dt"]),
        )
        net.probe_error_neurons = nengo.Probe(net.error.neurons, synapse=None)
        net.probe_obs = nengo.Probe(
            net.node_input[0],
            synapse=None,
            sample_every=float(params["dt"]),
        )

    return net


def _simulate_trial(
    obs_values: np.ndarray,
    params: dict,
    decoders: dict,
    *,
    return_probes: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Simulate one trial, return model responses (one per observation).

    If ``return_probes`` is True, return ``(responses, probe_data)`` instead.
    """
    n_obs = len(obs_values)
    t_total = n_obs * (float(params["t_obs"]) + float(params["t_iti"]))

    net = build_network(obs_values, params, decoders)
    # optional ITI noise injection (used by iti_perturbation.py)
    if float(params.get("iti_noise_amplitude", 0.0)) > 0:
        try:
            from scripts.iti_perturbation import _add_iti_noise

            _add_iti_noise(net, params, len(obs_values))
        except ImportError:
            pass
    with nengo.Simulator(
        net,
        dt=float(params["dt"]),
        seed=int(params["seed"]),
        progress_bar=False,
    ) as sim:
        sim.run(t_total)

    t_arr = np.arange(len(sim.data[net.probe_value])) * float(params["dt"])
    value_decoded = sim.data[net.probe_value].squeeze()
    responses = _extract_responses(t_arr, value_decoded, n_obs, params)
    if not return_probes:
        return responses
    probe_data = {
        "obs": sim.data[net.probe_obs].squeeze(),
        "error": sim.data[net.probe_error],
        "value": sim.data[net.probe_value].squeeze(),
        "counting_weight": sim.data[net.probe_counting_weight].squeeze(),
        "counting_count": sim.data[net.probe_counting_count].squeeze(),
        "t": np.arange(len(sim.data[net.probe_value])) * float(params["dt"]),
    }
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    dt = float(params["dt"])
    readout_offset = 0.5
    error_neuron_data = sim.data[net.probe_error_neurons]
    readout_indices = []
    for n in range(n_obs):
        t_readout = t_iti + n * t_step + readout_offset
        idx = int(np.round(t_readout / dt))
        idx = int(np.clip(idx, 0, len(error_neuron_data) - 1))
        readout_indices.append(idx)
    probe_data["error_neurons"] = error_neuron_data[readout_indices]
    if return_probes and hasattr(net, "probe_iti_noise"):
        probe_data["iti_noise"] = sim.data[net.probe_iti_noise].squeeze()
    return responses, probe_data


def run(
    params: dict,
    save: bool = False,
    trials: list | None = None,
    save_probes: bool = False,
) -> pd.DataFrame:
    """Run the NEF model for a single participant."""
    pfull = {**PARAM_DEFAULTS, **params}
    pfull["nef_type"] = "recurrent"

    required = (
        "model_type",
        "dataset",
        "pid",
        "t_obs",
        "t_iti",
        "dt",
        "tau_probe",
        "seed",
    )
    for key in required:
        if key not in pfull:
            raise KeyError(f"params must include {key!r}")

    dataset = pfull["dataset"]
    pid = int(pfull["pid"])

    # Every dataset -- carrabin, yoo, soltani_numbers, soltani_colors -- loads
    # real per-participant human data from its own pkl. An earlier version of
    # this function special-cased the soltani datasets to read the RETIRED
    # task/sequences/{continuous,binary}_sequences.pkl files with a dummy pid,
    # from back when no real human data existed for them; that branch is gone.
    # It silently discarded the `pid` argument, so an NEF fit would have
    # simulated old task/ sequences while fitting.losses scored the result
    # against real participant responses.
    #
    # The soltani datasets are now fully supported. Two issues that used to be
    # listed here are resolved:
    #
    # 1. 0-INDEXING. soltani trials/observations are 0-indexed (trial 0-31,
    #    obs 0-14) unlike carrabin/yoo (1-indexed), and the counting-activity
    #    map is keyed 1..n_trials, so a bare .get(trial) left trial 0 missing.
    #    activity_key_for_trial() now supplies the key AND the seed together
    #    (they must never diverge -- see its docstring), and a miss now raises
    #    instead of silently falling through to the ~300x slower _pretrain
    #    path with a different decoder-solve procedure.
    #
    # 2. RESPONSE SCALE. Both
    # soltani tasks ask for the MEAN of all observations, so neither the
    # Laplace transform nor any [0,100]<->[-1,1] rescale applies; value and
    # response are already on the canonical [-1,1] scale in the pkl, verified
    # against data/soltani_{numbers,colors}_complete_pairs.pkl. nef_obs_values
    # and nef_response_to_model_scale are now identity, and
    # apply_binary_transform is carrabin-only. See utils/binary_transform.py's
    # own module docstring for the audited scales.
    # dataset is the model-FAMILY key; datafile selects which build of that
    # family's human data to read (see utils.paths.dataset_stem).
    stem = dataset_stem(dataset, pfull.get("datafile"))
    human_pid = pd.read_pickle(data_path(f"{stem}.pkl")).query("pid == @pid")
    if trials is not None:
        human_pid = human_pid[human_pid["trial"].isin(trials)]

    # Load precomputed counting network activities (Gram matrices).
    # W_weight is recomputed per-trial via fast_decode using the current
    # (alpha_0, lambda_) -- 300x faster than re-running Nengo, AND the only
    # way every trial's decoders come from the SAME seeded tuning curves that
    # _simulate_trial actually builds the network with for that trial (see
    # activity_key_for_trial's own docstring).
    #
    # REQUIRED, not optional. This used to fall back to a single
    # _pretrain(pfull) call (pfull's own base seed) reused across EVERY trial
    # in the run when the activity file was missing entirely -- ~300x slower
    # per trial, AND a genuine seed mismatch: trial N's network is built and
    # simulated with seed=activity_key_for_trial(dataset, N), but the
    # fallback's decoders came from a network trained at pfull's base seed
    # instead. Silent, plausible-looking, wrong for every trial that doesn't
    # happen to share the base seed. Never re-add this fallback -- regenerate
    # the activity file instead. See _require_activity_map, shared with
    # simulate_ensemble() below so both paths' requirements stay in sync.
    _activity_map = _require_activity_map(
        int(pfull["n_neurons"]), int(pfull["n_neurons_counting"]),
        str(pfull.get("dataset", "carrabin")),
    )

    rows = []
    all_probe_data: list[dict] = []

    for trial, trial_data in human_pid.groupby("trial"):
        t_trial = time.time()
        trial_data = trial_data.sort_values("observation")
        obs_values = trial_data["value"].to_numpy(dtype=float)
        obs_values = nef_obs_values(obs_values, dataset)
        # The activity key and the simulation seed MUST be the same number:
        # activity entry k was precomputed from a network built with seed=k, so
        # decoders solved from it are only valid for a network with those tuning
        # curves. activity_key_for_trial() supplies both, handling the fact that
        # soltani trials are 0-indexed while activity keys start at 1 (a bare
        # _activity_map.get(trial) left trial 0 to miss the map entirely).
        akey = activity_key_for_trial(dataset, trial)
        p = {**pfull, "seed": akey}
        # _activity_map is guaranteed non-None here (the load above raises if
        # missing), but an individual KEY can still miss -- e.g. a trial count
        # beyond what was precomputed. That must still fail loudly, not fall
        # back to a mismatched-seed _pretrain() call.
        activity = _activity_map.get(akey)
        if activity is not None:
            decoders = fast_decode_counting(
                activity,
                alpha_0=float(pfull["alpha_0"]),
                lambda_=float(pfull["lambda_"]),
            )
        else:
            raise KeyError(
                f"No precomputed counting activity for key {akey} "
                f"(dataset={dataset!r}, trial={int(trial)}). The activity "
                f"file has keys 1..n_trials; check _DATASET_N_TRIALS and "
                f"_ZERO_INDEXED_DATASETS in models/counting_integrator.py, "
                f"or regenerate with --precompute_activities."
            )
        if save_probes:
            responses, probe_data = _simulate_trial(
                obs_values, p, decoders, return_probes=True
            )
            probe_data["trial"] = int(trial)
            probe_data["params"] = dict(p)
            all_probe_data.append(probe_data)
        else:
            responses = _simulate_trial(obs_values, p, decoders)
        elapsed_trial = time.time() - t_trial
        print(f"  pid={pid} trial {int(trial)}: {elapsed_trial:.1f}s", flush=True)
        for i, (_, row) in enumerate(trial_data.iterrows()):
            entry = {
                "model_type": pfull["model_type"],
                "pid": pid,
                "trial": int(trial),
                "observation": int(row["observation"]),
                "response": nef_response_to_model_scale(float(responses[i]), dataset),
            }
            rows.append(entry)

    out = apply_binary_transform(pd.DataFrame(rows), dataset)
    if save_probes and all_probe_data:
        fname = f"probe_{pfull['model_type']}_{stem}_{pid}.pkl"
        pd.to_pickle(all_probe_data, data_path(fname))
        print(f"  Saved probe data ({len(all_probe_data)} trials) to data/{fname}")
    if save:
        out.to_pickle(data_path(f"{pfull['model_type']}_{stem}_{pid}.pkl"))
    return out


def simulate_ensemble(
    params: dict, n_sims: int, return_index: bool = False, trials: list | None = None,
):
    """n_sims independent realisations of NEF, one per (trial, sim), for a
    distributional (NLL) loss -- the NEF analogue of
    models.math_models.simulate_ensemble, added to support NEF's --loss nll
    path (see docs/HISTORY.md and fitting/fit.py's dispatch).

    UNLIKE run()'s single canonical seed per trial (activity_key_for_trial's
    trial-tied seed, giving one deterministic point-estimate response), each
    sim here uses a GENUINELY DIFFERENT seed -- and therefore genuinely
    different neural tuning curves -- for the SAME trial's stimulus.
    activity_key_for_trial(dataset, trial, sim) supplies that seed; see its
    own docstring for why reusing seeds ACROSS TRIALS (rather than across
    sims within a trial) would silently correlate supposedly-independent
    ensemble members instead of giving genuine per-sim independence.

    REQUIRES an activity file with n_trials * n_sims entries, not just
    n_trials -- see counting_integrator.precompute_activities' own n_sims
    parameter. Generate with --precompute_activities --n_sims N (resumable:
    growing an existing file to a larger n_sims does not re-simulate the
    keys it already has).

    `trials` (optional, unlike math_models.simulate_ensemble which has no
    such option -- added because a real Nengo ensemble costs n_trials*n_sims
    real simulations, unlike a cheap closed-form math model, so restricting
    to a subset matters in practice, e.g. for scripts/check_NEF_pipeline.py's
    ensemble-invariant check): if given, only those trials are simulated,
    same filtering convention as run()'s own `trials` argument.

    Returns (n_sims, n_rows), rows ordered exactly as run() returns them
    (sorted by trial, then observation) -- the same convention
    models.math_models.simulate_ensemble uses, so both can be sliced/scored
    identically by fitting.losses.nll_from_ensemble.

    Applies the SAME post-processing run() applies
    (nef_response_to_model_scale, then apply_binary_transform) rather than a
    second, hand-rolled copy of it -- math_models.simulate_ensemble's own
    docstring flags exactly this as a real risk (it has to inline carrabin's
    Laplace-shrinkage formula itself, with a comment warning that re-deriving
    it must stay in sync with utils/binary_transform.py). Here that risk is
    avoided entirely: apply_binary_transform is called directly, once, on
    the full stacked (sim, trial, observation) frame.
    """
    pfull = {**PARAM_DEFAULTS, **params}
    pfull["nef_type"] = "recurrent"
    dataset = pfull["dataset"]
    pid = int(pfull["pid"])

    stem = dataset_stem(dataset, pfull.get("datafile"))
    human_pid = pd.read_pickle(data_path(f"{stem}.pkl")).query("pid == @pid")
    if trials is not None:
        human_pid = human_pid[human_pid["trial"].isin(trials)]

    _activity_map = _require_activity_map(
        int(pfull["n_neurons"]), int(pfull["n_neurons_counting"]),
        str(dataset), n_sims=n_sims,
    )

    per_trial = []
    index_rows = []
    for trial, trial_data in human_pid.groupby("trial"):
        trial_data = trial_data.sort_values("observation")
        obs_values = nef_obs_values(
            trial_data["value"].to_numpy(dtype=float), dataset
        )
        n_obs = len(obs_values)
        out = np.empty((n_sims, n_obs))
        for sim in range(1, n_sims + 1):
            akey = activity_key_for_trial(dataset, trial, sim=sim)
            p = {**pfull, "seed": akey}
            activity = _activity_map.get(akey)
            if activity is None:
                raise KeyError(
                    f"No precomputed counting activity for key {akey} "
                    f"(dataset={dataset!r}, trial={int(trial)}, sim={sim}). "
                    f"The activity file needs n_trials*n_sims entries -- "
                    f"regenerate with --precompute_activities "
                    f"--n_sims {n_sims}."
                )
            decoders = fast_decode_counting(
                activity,
                alpha_0=float(pfull["alpha_0"]),
                lambda_=float(pfull["lambda_"]),
            )
            out[sim - 1, :] = _simulate_trial(obs_values, p, decoders)
        per_trial.append(out)
        obs_labels = trial_data["observation"].to_numpy()
        index_rows.append(pd.DataFrame(
            {"trial": [int(trial)] * len(obs_labels), "observation": obs_labels}))

    ens_raw = np.concatenate(per_trial, axis=1)  # (n_sims, n_rows), pre-transform
    index_df = pd.concat(index_rows, ignore_index=True)
    n_rows = ens_raw.shape[1]

    # Apply run()'s own post-processing ONCE on the full stacked frame,
    # rather than per-sim or via a re-derived formula.
    long_df = pd.DataFrame({
        "model_type": pfull["model_type"],
        "pid": pid,
        "trial": np.tile(index_df["trial"].to_numpy(), n_sims),
        "observation": np.tile(index_df["observation"].to_numpy(), n_sims),
        "response": [
            nef_response_to_model_scale(float(v), dataset) for v in ens_raw.ravel()
        ],
    })
    long_df = apply_binary_transform(long_df, dataset)
    ens = long_df["response"].to_numpy().reshape(n_sims, n_rows)

    if return_index:
        return ens, index_df
    return ens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEF evidence integration")
    p.add_argument(
        "--dataset",
        type=str,
        default="carrabin",
        choices=("carrabin", "yoo", "soltani_numbers", "soltani_colors"),
    )
    p.add_argument("--pid", type=int, default=1)
    p.add_argument("--model_type", type=str, default="NEF")
    p.add_argument("--n_obs", type=int, default=PARAM_DEFAULTS["n_obs"])
    p.add_argument("--n_neurons", type=int, default=PARAM_DEFAULTS["n_neurons"])
    p.add_argument(
        "--n_neurons_counting",
        type=int,
        default=PARAM_DEFAULTS["n_neurons_counting"],
    )
    p.add_argument("--lambda_", type=float, default=PARAM_DEFAULTS["lambda_"])
    p.add_argument("--tau_ff", type=float, default=PARAM_DEFAULTS["tau_ff"])
    p.add_argument("--tau_fb", type=float, default=PARAM_DEFAULTS["tau_fb"])
    p.add_argument("--T_error", type=float, default=PARAM_DEFAULTS["T_error"])
    p.add_argument("--tau_error", type=float, default=PARAM_DEFAULTS["tau_error"])
    p.add_argument("--onset_detector_amp", type=float, default=PARAM_DEFAULTS["onset_detector_amp"])
    p.add_argument("--tau_fast", type=float, default=PARAM_DEFAULTS["tau_fast"])
    p.add_argument("--tau_slow", type=float, default=PARAM_DEFAULTS["tau_slow"])
    p.add_argument("--tau_probe", type=float, default=PARAM_DEFAULTS["tau_probe"])
    p.add_argument("--radius_e", type=float, default=PARAM_DEFAULTS["radius_e"])
    p.add_argument("--radius_v", type=float, default=PARAM_DEFAULTS["radius_v"])
    p.add_argument("--pes_learning_rate", type=float, default=PARAM_DEFAULTS["pes_learning_rate"])
    p.add_argument("--dt", type=float, default=PARAM_DEFAULTS["dt"])
    p.add_argument("--t_obs", type=float, default=PARAM_DEFAULTS["t_obs"])
    p.add_argument("--t_iti", type=float, default=PARAM_DEFAULTS["t_iti"])
    p.add_argument("--seed", type=int, default=PARAM_DEFAULTS["seed"])
    p.add_argument("--alpha_0", type=float, default=PARAM_DEFAULTS["alpha_0"])
    p.add_argument("--save_probes", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = run(vars(args), save_probes=args.save_probes)
    print(df.head())
