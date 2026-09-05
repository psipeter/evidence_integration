# archive/models/archive_NEF_simulate_ensemble.py — retired NEF NLL/multi-seed ensemble branch
#
# Retired (this session): too expensive to feasibly run at scale. See
# docs/DECISIONS.md. This was models/NEF.py's `simulate_ensemble()` and
# `NEF_DEFAULT_N_SIMS`, used only by fitting.fit's now-removed NEF NLL
# dispatch branch and scripts/check_NEF_pipeline.py's now-removed
# check_ensemble_invariant(). NEF still fits under RMSE via NEF.run() --
# unaffected by this retirement.
#
# To restore: paste this function body back into models/NEF.py (it uses
# that module's own PARAM_DEFAULTS, _require_activity_map, and
# _simulate_trial, all still present there), reinstate the NLL dispatch
# branch in fitting/fit.py (see git history for the pre-retirement
# version), and regenerate multi-seed activity files
# (counting_integrator.py --precompute_activities --n_sims N -- that
# generic sim>1 support was NOT removed, since it's shared, harmless
# infrastructure used by run()'s own sim=1 case too).

# Ballpark default for NEF's NLL ensemble size, from CHEAP-MODEL calibration
# (NoisyRL_lambda proxy, scripts/calibrate_nll_nsims.py -- also archived)
# rather than a direct NEF measurement.
NEF_DEFAULT_N_SIMS = 50


def simulate_ensemble(
    params: dict, n_sims: int, return_index: bool = False, trials: list | None = None,
):
    """n_sims independent realisations of NEF, one per (trial, sim), for a
    distributional (NLL) loss -- the NEF analogue of
    models.math_models.simulate_ensemble (also retired, see
    archive_math_models_noise.py).

    UNLIKE run()'s single canonical seed per trial (activity_key_for_trial's
    trial-tied seed, giving one deterministic point-estimate response), each
    sim here uses a GENUINELY DIFFERENT seed -- and therefore genuinely
    different neural tuning curves -- for the SAME trial's stimulus.
    activity_key_for_trial(dataset, trial, sim) supplies that seed.

    REQUIRES an activity file with n_trials * n_sims entries, not just
    n_trials. Generate with --precompute_activities --n_sims N (resumable).

    Requires this module's own PARAM_DEFAULTS, _require_activity_map, and
    _simulate_trial to be in scope -- paste back into models/NEF.py rather
    than importing this file directly.
    """
    import numpy as np
    import pandas as pd
    from models.counting_integrator import (
        activity_key_for_trial, fast_decode as fast_decode_counting,
    )
    from utils.paths import data_path, dataset_stem
    from utils.binary_transform import (
        apply_binary_transform, nef_obs_values, nef_response_to_model_scale,
    )
    # The three names below (PARAM_DEFAULTS, _require_activity_map,
    # _simulate_trial) are NEF.py module-level objects -- present when this
    # function is pasted back into that file, undefined here.
    pfull = {**PARAM_DEFAULTS, **params}                       # noqa: F821
    pfull["nef_type"] = "recurrent"
    dataset = pfull["dataset"]
    pid = int(pfull["pid"])

    stem = dataset_stem(dataset, pfull.get("datafile"))
    human_pid = pd.read_pickle(data_path(f"{stem}.pkl")).query("pid == @pid")
    if trials is not None:
        human_pid = human_pid[human_pid["trial"].isin(trials)]

    _activity_map = _require_activity_map(                     # noqa: F821
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
            out[sim - 1, :] = _simulate_trial(obs_values, p, decoders)  # noqa: F821
        per_trial.append(out)
        obs_labels = trial_data["observation"].to_numpy()
        index_rows.append(pd.DataFrame(
            {"trial": [int(trial)] * len(obs_labels), "observation": obs_labels}))

    ens_raw = np.concatenate(per_trial, axis=1)
    index_df = pd.concat(index_rows, ignore_index=True)
    n_rows = ens_raw.shape[1]

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
