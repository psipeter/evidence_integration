# archive/models/archive_math_models_noise.py — retired noise/task-specific models
#
# Retired from active use (this session): NoisyCounting (carrabin-specific
# task model, MLE-fitted) and NoisyRL_lambda (RL_lambda + compounding STATE
# noise). See docs/DECISIONS.md for why. Existing fits/figures that already
# read cached `NoisyCounting_*`/`NoisyRL_lambda_*` .pkl files under
# data/runs/ are UNAFFECTED -- this only retires the ability to generate NEW
# fits of these models. To restore, merge this back into models/math_models.py
# (re-add the two allowlist entries, the two `_run_*` branches, and
# `simulate_ensemble`/`_STOCHASTIC_ENSEMBLE_MODELS`/`_noisy_rl_lambda_response`
# below) and reinstate `NoisyCounting`/`NoisyRL_lambda` in
# fitting/model_params.py, utils/soltani_models.py, and utils/slurm.py.
#
# ---------------------------------------------------------------------------
# Allowlist entries that were removed from math_models.py's dataset->model
# sets (add back if restoring):
#   _CARRABIN_MODELS: "NoisyCounting", "NoisyRL_lambda"
#   _YOO_MODELS: "NoisyRL_lambda"
#   _SOLTANI_COLORS_MODELS / _SOLTANI_NUMBERS_MODELS: "NoisyRL_lambda"
#   _STOCHASTIC_ENSEMBLE_MODELS = frozenset({"NoisyRL_lambda"})  # deleted entirely
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd

from utils.paths import data_path, dataset_stem
from utils.run_params import trial_seed as _trial_seed


def simulate_ensemble(params: dict, n_sims: int, return_index: bool = False):
    """`n_sims` independent realisations of a STOCHASTIC model (only
    NoisyRL_lambda used this), for a distributional loss. See the retired
    docstring in git history (models/math_models.py, pre-retirement) for the
    full seeding/shrinkage rationale -- unchanged below, just relocated.
    """
    model_type = params["model_type"]
    if model_type != "NoisyRL_lambda":
        raise ValueError(
            f"simulate_ensemble (archived) only ever supported NoisyRL_lambda; "
            f"got {model_type!r}.")

    dataset = params["dataset"]
    stem = dataset_stem(dataset, params.get("datafile"))
    human = pd.read_pickle(data_path(f"{stem}.pkl"))
    pid = int(params["pid"])
    hp = human[human["pid"] == pid].sort_values(["trial", "observation"])

    alpha_0 = float(params["alpha_0"])
    lambda_ = float(params["lambda_"])
    sigma_state = float(params["sigma_state"])

    per_trial = []
    index_rows = []
    for trial, g in hp.groupby("trial", sort=True):
        vals = g["value"].to_numpy(float)
        n_obs = len(vals)
        out = np.empty((n_sims, n_obs))
        for sim in range(n_sims):
            rng = np.random.RandomState(_trial_seed(sim, int(trial)))
            e = 0.0
            for n, x in enumerate(vals, start=1):
                e = float(np.clip(e + (alpha_0 / n ** lambda_) * (x - e)
                                  + rng.normal(0.0, sigma_state), -1, 1))
                out[sim, n - 1] = e
        per_trial.append(out)
        obs_labels = g["observation"].to_numpy()
        index_rows.append(pd.DataFrame(
            {"trial": [int(trial)] * len(obs_labels), "observation": obs_labels}))

    ens = np.concatenate(per_trial, axis=1)
    index_df = pd.concat(index_rows, ignore_index=True)

    if dataset == "carrabin":
        t = index_df["observation"].to_numpy(float) + 1.0
        shrink = t / (t + 2.0)
        ens = ens * shrink[np.newaxis, :]

    if return_index:
        return ens, index_df
    return ens


def _noisy_rl_lambda_response(params: dict, values: np.ndarray, trial: int) -> float:
    """RL_lambda + compounding sigma_state only (sigma_resp was split out into
    the still-active add_noise()/`_resp_noise` wrapper -- see docs/DECISIONS.md
    for the "two response-noise mechanisms" rationale, which is UNAFFECTED by
    this retirement; only the STATE-noise side is retired).
    """
    alpha_0 = float(params["alpha_0"])
    lambda_ = float(params["lambda_"])
    sigma_state = float(params["sigma_state"])
    if len(values) == 0:
        return 0.0
    seed = _trial_seed(int(params.get("seed", 0)), int(trial))
    rng = np.random.RandomState(seed)
    expectation = 0.0
    for n, value in enumerate(values, start=1):
        alpha = alpha_0 / (n ** lambda_)
        error = value - expectation
        expectation += alpha * error + rng.normal(0.0, sigma_state)
        expectation = float(np.clip(expectation, -1, 1))
    return expectation


def _noisy_counting_response(params: dict, values: np.ndarray, trial: int) -> float:
    """Prat-Carrabin & Woodford (2024), Table 5 Line 12: Eq. 31 (cognitive
    state) and Eq. 34 (response), on [-1, 1]. Carrabin-only, task-specific.
    RMSE-fitted sigma_c collapses to ~0 (response-noise artefact); the
    MLE-fitted version (retired fitting/fit_mle.py) recovered sigma_c
    ~0.03-0.08, nu ~0.08-0.21.
    """
    mu = float(params["mu"])
    sigma_c = float(params["sigma_c"])
    nu = float(params["nu"])
    if len(values) == 0:
        return 0.0
    seed = _trial_seed(int(params.get("seed", 0)), int(trial))
    rng = np.random.RandomState(seed)
    r = 0.0
    p_hat = 0.0
    for x in values:
        xi = rng.normal(0.0, sigma_c)
        r = r + float(x) * mu + xi
        epsilon = rng.normal(0.0, nu)
        p_hat = p_hat + (r - p_hat) * float(np.exp(epsilon))
        p_hat = float(np.clip(p_hat, -1.0, 1.0))
    return float(p_hat)
