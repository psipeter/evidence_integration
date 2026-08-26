# NOTE: jiang/usher model classes archived in archive/models/archive_math_models.py
"""
Mathematical (non-NEF) models of evidence integration.

Expectations are computed from empirical sequences in per-dataset pickle files
and collected into a single tabular format with model ``response`` values.

**Datasets and model types**

- **carrabin:** ``Mean`` (optimal), ``NoisyCounting`` (human-matching), ``RL`` (naive), ``PrimacyRecency`` (flexible temporal weighting)
- **soltani:** ``Mean``, ``LeakyIntegrator``, ``PrimacyRecency``, ``RL_lambda``, and
  ``NoisyRL_lambda`` -- RL_lambda plus tunable state and response noise, the only
  math model here with a stochastic component
- **yoo:** ``Mean`` (optimal), ``PrimacyRecency`` (flexible temporal weighting), ``RL`` (naive)
- **soltani_colors, soltani_numbers:** ``Mean``, ``LeakyIntegrator``, ``PrimacyRecency``, ``RL_lambda`` --
  together intended to capture recency-biased (non-shrinking-learning-rate)
  behavior. Human data for these two comes from
  ``scripts/build_model_inputs.py``, which rescales this task's native
  [0,100] response/value scale to the [-1,1] scale carrabin/yoo already use,
  so the SAME model implementations below (just under new dataset names)
  apply with no scale-specific changes.
- diederen models archived in ``archive/misc/math_models_diederen.py``

**Unified interface**

Every model is run via ``run(params, save=False, trials=None)``. Required keys in
``params`` for all models:

- ``"model_type"`` (``str``): one of the strings above for the chosen dataset
- ``"dataset"`` (``str``): ``"carrabin"``, ``"yoo"``, ``"soltani_colors"``, or ``"soltani_numbers"``
- ``"pid"`` (``int``): participant id

Additional keys are model-specific (learning rates, noise scales, etc.). The
optional ``trials`` argument restricts execution to a subset of trial ids.
"""

import numpy as np
import pandas as pd

from utils.paths import data_path, dataset_stem
from utils.carrabin_transform import apply_carrabin_transform
from utils.run_params import trial_seed as _trial_seed


_CARRABIN_MODELS = frozenset(
    {"Mean", "NoisyCounting", "RL", "RL_lambda", "LeakyIntegrator", "PrimacyRecency", "NoisyRL_lambda"}
)
_YOO_MODELS = frozenset({"Mean", "LeakyIntegrator", "PrimacyRecency", "RL", "RL_lambda", "NoisyRL_lambda"})
# Deliberately narrower than carrabin/yoo: these four together are meant to
# capture recency-biased (non-shrinking-learning-rate) behavior, which is
# what this task's human data actually looks like (see the conversation that
# motivated this integration) -- no NoisyCounting (carrabin-specific) or
# plain fixed-alpha RL (superseded by RL_lambda, which subsumes it at
# lambda_->0) for either task dataset.
# Models whose ensemble is non-degenerate, i.e. usable with a distributional
# loss. Deterministic models are excluded by construction, not by policy.
_STOCHASTIC_ENSEMBLE_MODELS = frozenset({"NoisyRL_lambda"})

_SOLTANI_COLORS_MODELS = frozenset(
    {"Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NoisyRL_lambda"})
_SOLTANI_NUMBERS_MODELS = frozenset(
    {"Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NoisyRL_lambda"})


# Base models that add_noise() knows how to wrap. Any model whose run() output
# is a deterministic function of (dataset, pid, its own params) qualifies --
# add_noise never touches per-model branches, it only calls run() once and adds
# i.i.d. noise on top.
_NOISE_WRAPPABLE_BASE_MODELS = frozenset(
    {"Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda"})

# model_type suffix -> base model name, and back. "_resp_noise" is the ONLY
# mechanism add_noise implements (i.i.d. per observation); it is not a stand-in
# for arbitrary noise types, so do not overload this suffix for anything else.
_RESP_NOISE_SUFFIX = "_resp_noise"


def base_model_of(model_type: str) -> str:
    """Strip the _resp_noise suffix, if present. 'RL_lambda' -> 'RL_lambda';
    'RL_lambda_resp_noise' -> 'RL_lambda'."""
    if model_type.endswith(_RESP_NOISE_SUFFIX):
        return model_type[: -len(_RESP_NOISE_SUFFIX)]
    return model_type


def is_resp_noise_model(model_type: str) -> bool:
    return model_type.endswith(_RESP_NOISE_SUFFIX)


def add_noise(params: dict, n_sims: int, sigma_resp: float,
             return_index: bool = False):
    """Ensemble for a DETERMINISTIC base model plus i.i.d. response noise.

    `params["model_type"]` must be one of _NOISE_WRAPPABLE_BASE_MODELS (the bare
    name, e.g. "RL_lambda" -- NOT "RL_lambda_resp_noise"; the suffix is a
    fitting-time label, this function only needs the base model to run). Calls
    run() ONCE to get the deterministic mean trajectory mu (length n_rows), then

        ens = clip(mu + N(0, sigma_resp), -1, 1)     for n_sims draws

    No trial-structure or replay logic is needed, unlike simulate_ensemble's
    state-noise loop: i.i.d. noise has no sequential dependency to preserve, so
    this is a single vectorised draw rather than a per-observation Python loop --
    cheaper than simulating a real state-noise model.

    This is what lets a deterministic model enter an NLL comparison on equal
    footing with a genuinely stochastic one (e.g. NoisyRL_lambda's remaining
    sigma_state) for exactly ONE extra parameter, without any dependency on a
    prior RMSE fit -- mu comes fresh from run(), not from a saved _responses.pkl.
    `params["model_type"]` may be either the bare base name ("RL_lambda") or the
    fitting-time suffixed name ("RL_lambda_resp_noise") -- base_model_of() strips
    the suffix if present, so both work identically. The stripped name is what
    gets passed to run() and to _validate_model_dataset; sigma_resp is always
    removed from base_params before that call, since run() has no such parameter
    on a deterministic model.
    """
    model_type = base_model_of(params["model_type"])
    if model_type not in _NOISE_WRAPPABLE_BASE_MODELS:
        raise ValueError(
            f"add_noise wraps a deterministic base model; {params['model_type']!r} "
            f"is not in {sorted(_NOISE_WRAPPABLE_BASE_MODELS)} (after stripping any "
            f"_resp_noise suffix). If it is already stochastic (e.g. "
            f"NoisyRL_lambda), use simulate_ensemble instead.")

    base_params = {k: v for k, v in params.items() if k != "sigma_resp"}
    base_params["model_type"] = model_type
    mu_df = run(base_params).sort_values(["trial", "observation"])
    mu = mu_df["response"].to_numpy(float)

    rng = np.random.RandomState(int(params.get("seed", 0)))
    ens = np.clip(mu[np.newaxis, :] + rng.normal(0.0, sigma_resp, (n_sims, len(mu))),
                 -1.0, 1.0)
    if return_index:
        return ens, mu_df[["trial", "observation"]].reset_index(drop=True)
    return ens


def simulate_ensemble(params: dict, n_sims: int,
                      return_index: bool = False):
    """`n_sims` independent realisations of a STOCHASTIC model, for a
    distributional loss. Returns (n_sims, n_rows) with rows ordered exactly as
    `run(params)` returns them, i.e. sorted by (trial, observation).

    WHY THIS EXISTS RATHER THAN CALLING run() n_sims TIMES. run() re-simulates
    from scratch for every (trial, observation) -- 480 pandas .query() calls per
    parameter point on soltani -- so an ensemble of 100 would be ~48k queries per
    Optuna trial, which is not viable. Here each (trial, sim) is ONE forward pass
    of 15 steps, so a 100-sim ensemble is ~3200 passes and takes well under a
    second.

    SEEDING MATCHES run(): sim i uses _trial_seed(i, trial), so
    `simulate_ensemble(params, n)[i]` is identical to
    `run({**params, "seed": i}).response`. VERIFY WITH
    scripts/verify_ensemble_invariant.py before trusting this function on a new
    dataset or after editing it -- there is no automated test (this project has
    no test suite; do not add a comment claiming otherwise). This check is not
    optional: extending NoisyRL_lambda to carrabin surfaced two real bugs that
    only this invariant caught -- (1) an editing accident that silently deleted
    _run_carrabin's RL_lambda/LeakyIntegrator/PrimacyRecency branches, and (2)
    this function labelling ensemble columns with a synthetic range(n_obs)
    instead of the dataset's REAL (possibly 1-indexed) observation values, which
    silently fed the wrong `t` into the carrabin shrinkage formula below. Neither
    was caught by py_compile or by exercising individual model branches in
    isolation -- only by comparing simulate_ensemble against run() directly.

    Only implemented for models whose response is a stochastic function of the
    stimulus. Deterministic models raise: their ensemble is a delta function, so a
    Gaussian NLL over it is undefined (see fitting.losses.compute_nll).

    `return_index=True` additionally returns a DataFrame with `trial` and
    `observation` columns, one row per ENSEMBLE COLUMN in the same order --
    letting a caller (e.g. fitting.fit's cross-validation) partition columns by
    trial without re-deriving the (trial, observation) sort order itself.
    """
    model_type = params["model_type"]
    if model_type not in _STOCHASTIC_ENSEMBLE_MODELS:
        raise ValueError(
            f"simulate_ensemble is for stochastic models only; {model_type!r} is "
            f"deterministic, so its ensemble is a delta function and a Gaussian "
            f"NLL over it is undefined. Use RMSE for that model.")

    dataset = params["dataset"]
    stem = dataset_stem(dataset, params.get("datafile"))
    human = pd.read_pickle(data_path(f"{stem}.pkl"))
    pid = int(params["pid"])
    hp = human[human["pid"] == pid].sort_values(["trial", "observation"])

    alpha_0 = float(params["alpha_0"])
    lambda_ = float(params["lambda_"])
    sigma_state = float(params["sigma_state"])
    # sigma_resp REMOVED from NoisyRL_lambda -- it is now state-noise-only. The
    # i.i.d.-noise comparison lives in add_noise()'s generic "<model>_resp_noise"
    # wrapper instead, applied to a plain (deterministic) RL_lambda, so the two
    # noise MECHANISMS can be compared at equal parameter count (1 extra param
    # each) rather than NoisyRL_lambda alone carrying both. See CLAUDE.md /
    # docs/HISTORY.md for why this split was made.

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
        # Use the REAL observation labels from the data, not a synthetic
        # range(n_obs). For soltani these coincide (0-indexed, contiguous), which
        # is why this was not caught immediately -- but carrabin's observations
        # are 1-indexed (1..5), and the shrinkage formula below needs the ACTUAL
        # value, not a renumbering that happens to have the same length.
        obs_labels = g["observation"].to_numpy()
        index_rows.append(pd.DataFrame(
            {"trial": [int(trial)] * len(obs_labels), "observation": obs_labels}))

    ens = np.concatenate(per_trial, axis=1)
    index_df = pd.concat(index_rows, ignore_index=True)

    # CARRABIN'S LAPLACE SHRINKAGE, applied here because run() applies it via
    # apply_binary_transform AFTER _run() returns, and this function bypasses
    # _run()/run() entirely for speed (see the module-level docstring above).
    # Skipping it would silently break two things: the `simulate_ensemble(...)[i]
    # == run({**params, "seed": i}).response` invariant (verified only for
    # soltani so far, where the transform is identity), and -- more seriously --
    # it would bias every downstream NLL for carrabin, since the ensemble's
    # (mu, sigma) would be on the UNSHRUNK scale while human responses are
    # compared against the shrunk convention every other carrabin model uses.
    #
    # NoisyRL_lambda is NOT in binary_transform._EXEMPT_MODELS (matching
    # RL_lambda, which also is not), so it gets shrunk like every other carrabin
    # model except NoisyCounting. The formula is inlined rather than calling
    # apply_binary_transform, because that function expects a long DataFrame with
    # one row per observation, not an (n_sims, n_rows) array -- but it MUST stay
    # in sync with utils/binary_transform.py's formula. There is an equivalence
    # test against run() covering exactly this.
    if dataset == "carrabin":
        t = index_df["observation"].to_numpy(float) + 1.0   # 1-indexed count
        shrink = t / (t + 2.0)                               # (n_rows,)
        ens = ens * shrink[np.newaxis, :]

    if return_index:
        return ens, index_df
    return ens


def run(params: dict, save: bool = False, trials: list | None = None) -> pd.DataFrame:
    for key in ("model_type", "dataset", "pid"):
        if key not in params:
            raise KeyError(f"params must include {key!r}")

    model_type: str = params["model_type"]
    dataset: str = params["dataset"]
    pid: int = int(params["pid"])

    _validate_model_dataset(model_type, dataset)

    # `dataset` is the model-family key; the optional `datafile` selects which
    # build of that family's human data to read (see utils.paths.dataset_stem).
    stem = dataset_stem(dataset, params.get("datafile"))
    human = pd.read_pickle(data_path(f"{stem}.pkl"))
    human_pid = human.query("pid == @pid")
    if human_pid.empty:
        raise ValueError(f"No rows for pid={pid} in dataset {dataset!r}")
    if trials is not None:
        human_pid = human_pid[human_pid["trial"].isin(trials)]

    human_pid = human_pid[human_pid["response"].notna()]

    rows: list[dict] = []
    pairs = (
        human_pid[["trial", "observation"]]
        .drop_duplicates()
        .sort_values(["trial", "observation"])
    )
    for _, pr in pairs.iterrows():
        trial = int(pr["trial"])
        observation = int(pr["observation"])
        estimate = _run(params, human_pid, trial, observation)
        rows.append(
            {
                "model_type": model_type,
                "pid": pid,
                "trial": trial,
                "observation": observation,
                "response": estimate,
            }
        )

    out = apply_carrabin_transform(pd.DataFrame(rows), dataset)
    if save:
        fname = f"{model_type}_{stem}_{pid}_responses.pkl"
        out.to_pickle(data_path(fname))
    return out


def _validate_model_dataset(model_type: str, dataset: str) -> None:
    if dataset == "carrabin":
        allowed = _CARRABIN_MODELS
    elif dataset == "yoo":
        allowed = _YOO_MODELS
    elif dataset == "soltani_colors":
        allowed = _SOLTANI_COLORS_MODELS
    elif dataset == "soltani_numbers":
        allowed = _SOLTANI_NUMBERS_MODELS
    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected one of "
            f"'carrabin', 'yoo', 'soltani_colors', 'soltani_numbers'"
        )
    if model_type not in allowed:
        raise ValueError(
            f"Model {model_type!r} is not valid for dataset {dataset!r}; "
            f"expected one of {sorted(allowed)}"
        )


def _run(params: dict, human_pid: pd.DataFrame, trial: int, step: int) -> float:
    dataset = params["dataset"]

    if dataset == "carrabin":
        return _run_carrabin(params, human_pid, trial, step)
    if dataset == "yoo":
        return _run_yoo(params, human_pid, trial, step)
    if dataset == "soltani_colors":
        return _run_soltani_colors(params, human_pid, trial, step)
    if dataset == "soltani_numbers":
        return _run_soltani_numbers(params, human_pid, trial, step)
    raise AssertionError("unreachable")


def _run_primacy_recency(
    params: dict, values: np.ndarray, observation: int, trial: int
) -> float:
    """Pooley et al. (2011) / Galdo et al. (2022) temporal weighting function.

    w(o, t) = [1 - (1 - eps_p^o)(1 - eps_r^(t-o+1))] * (1-eta) + eta
    where o is 1-indexed position, t = current observation count.

    Free parameters: eps_p (primacy), eps_r (recency).
    eta fixed at 0.01 (Yoo et al. 2025).
    sigma_w from the original ADM paper is not used here: it captured
    continuous 30 Hz joystick trajectory noise under likelihood fitting,
    which has no equivalent in our single-response-per-observation RMSE setup.
    """
    eps_p = float(params["eps_p"])
    eps_r = float(params["eps_r"])
    eta   = float(params.get("eta", 0.01))
    n = len(values)
    weights = np.array(
        [
            (1.0 - (1.0 - eps_p ** (o + 1)) * (1.0 - eps_r ** (n - o)))
            * (1.0 - eta)
            + eta
            for o in range(n)
        ],
        dtype=float,
    )
    return float(np.dot(weights, values) / np.sum(weights))


def _noisy_rl_lambda_response(params: dict, values: np.ndarray, trial: int) -> float:
    """RL_lambda plus two SEPARATE noise sources. The distinction is the point of
    the model, not a detail:

      sigma_state  perturbs the internal ESTIMATE, so it PERSISTS and compounds
                   into every later update. This is what produces state-
                   persistent response variability -- residual variance growth
                   across observations, and within-trial residual autocorrelation
                   (temporal cols 3-4 for soltani). A deterministic model has
                   none, which is why those panels exclude the deterministic math
                   models by construction.

    sigma_resp (i.i.d. response noise) is DELIBERATELY NOT a parameter of this
    model any more -- it was removed so the two noise MECHANISMS (compounding
    state noise vs i.i.d. response noise) can be compared at EQUAL parameter
    count. The i.i.d. comparison now lives in add_noise()'s generic
    "<model>_resp_noise" wrapper, applied to a plain deterministic RL_lambda:
    that gives RL_lambda_resp_noise one extra parameter (sigma_resp) and this
    model one extra parameter (sigma_state), so an NLL comparison between them
    isolates the MECHANISM rather than one model simply having more parameters
    than the other. See CLAUDE.md / docs/HISTORY.md for the fuller rationale.

    (The earlier two-noise version measured a real |delta response| plateau,
    reconciling RL_lambda's fitted lambda with its descriptive lambda on soltani
    numbers -- deterministic 0.921 vs human 0.294; with added response noise
    0.369, paired gap +0.008, p=0.668. That finding is unaffected: it used
    add_noise on RL_lambda's OUTPUT after the fact, not this model's own
    sigma_resp, so it still reproduces under the new split.)

    ONE DEFINITION, called identically from _run_carrabin, _run_yoo and
    _run_soltani_common -- this used to be triplicated by an accidental
    find-and-replace across all three (see the note above _run_carrabin), which
    is exactly the drift risk a shared helper avoids.

    NOTE: the carrabin Laplace-shrinkage post-processing (apply_binary_transform)
    is applied to run()'s output AFTER this function returns, by run() itself --
    this function must stay UNSHRUNK, matching RL_lambda's own convention (it is
    not in binary_transform._EXEMPT_MODELS). simulate_ensemble applies the same
    shrinkage vectorised, separately -- see its own docstring.
    """
    alpha_0 = float(params["alpha_0"])
    lambda_ = float(params["lambda_"])
    sigma_state = float(params["sigma_state"])
    if len(values) == 0:
        return 0.0
    # Seeded by (seed, trial) exactly as NoisyCounting is. This matters for
    # coherence, not just reproducibility: run() re-simulates from scratch for
    # every (trial, observation), so a trial-scoped seed makes observation k+1's
    # simulation REPLAY the same draws as observation k's for its first k steps.
    # Without it the "trajectory" would be a set of unrelated noise realisations
    # and the state noise would not persist in any meaningful sense.
    seed = _trial_seed(int(params.get("seed", 0)), int(trial))
    rng = np.random.RandomState(seed)
    expectation = 0.0
    for n, value in enumerate(values, start=1):
        alpha = alpha_0 / (n ** lambda_)
        error = value - expectation
        expectation += alpha * error + rng.normal(0.0, sigma_state)
        expectation = float(np.clip(expectation, -1, 1))
    return expectation


def _run_carrabin(
    params: dict, human_pid: pd.DataFrame, trial: int, observation: int
) -> float:
    model_type = params["model_type"]
    subdata = human_pid.query("trial == @trial & observation <= @observation")
    values = subdata["value"].to_numpy()
    t = len(values)
    n_R = np.sum((values + 1) / 2)

    if model_type == "Mean":
        return float(np.mean(values))
    if model_type == "NoisyCounting":
        # Prat-Carrabin & Woodford (2024), Table 5 Line 12: Eq. 31 (cognitive
        # state) and Eq. 34 (response), on [-1, 1].
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
    if model_type == "RL":
        expectation = 0.0
        for value in values:
            error = value - expectation
            expectation += params["alpha"] * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    if model_type == "NoisyRL_lambda":
        return _noisy_rl_lambda_response(params, values, trial)
    if model_type == "RL_lambda":
        alpha_0 = float(params["alpha_0"])
        lambda_ = float(params["lambda_"])
        expectation = 0.0
        for n, value in enumerate(values, start=1):
            alpha = alpha_0 / (n ** lambda_)
            error = value - expectation
            expectation += alpha * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    if model_type == "LeakyIntegrator":
        gamma = float(params["gamma"])
        v = 0.0
        for x in values:
            v = gamma * v + (1.0 - gamma) * float(x)
        return float(np.clip(v, -1.0, 1.0))
    if model_type == "PrimacyRecency":
        return _run_primacy_recency(params, values, observation, trial)
    raise AssertionError("unreachable")


def _run_yoo(
    params: dict,
    human_pid: pd.DataFrame,
    trial: int,
    observation: int,
) -> float:
    model_type = params["model_type"]
    subdata = human_pid.query("trial == @trial & observation <= @observation")
    values = subdata["value"].to_numpy()

    if model_type == "Mean":
        return float(np.mean(values))
    if model_type == "RL":
        expectation = 0.0
        for value in values:
            error = value - expectation
            expectation += params["alpha"] * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    if model_type == "NoisyRL_lambda":
        return _noisy_rl_lambda_response(params, values, trial)
    if model_type == "RL_lambda":
        alpha_0 = float(params["alpha_0"])
        lambda_ = float(params["lambda_"])
        expectation = 0.0
        for n, value in enumerate(values, start=1):
            alpha = alpha_0 / (n ** lambda_)
            error = value - expectation
            expectation += alpha * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    if model_type == "LeakyIntegrator":
        gamma = float(params["gamma"])
        v = 0.0
        for x in values:
            v = gamma * v + (1.0 - gamma) * float(x)
        return float(np.clip(v, -1.0, 1.0))
    if model_type == "PrimacyRecency":
        return _run_primacy_recency(params, values, observation, trial)
    raise AssertionError("unreachable")


def _run_soltani_common(
    params: dict, human_pid: pd.DataFrame, trial: int, observation: int
) -> float:
    """Shared implementation for soltani_colors and soltani_numbers.

    Both datasets are rescaled to the same [-1, 1] scale carrabin/yoo use
    (see scripts/build_model_inputs.py) and both only support the same
    four model types (_SOLTANI_COLORS_MODELS == _SOLTANI_NUMBERS_MODELS), so
    there is no dataset-specific branching needed here -- unlike
    _run_carrabin vs _run_yoo (which differ in which extra models they
    support: NoisyCounting for carrabin, plain RL for both but not these
    two), soltani_colors and soltani_numbers are identical at this level.
    _run_soltani_colors/_run_soltani_numbers below are kept as separate named
    entry points (rather than calling this directly from _run()) so the
    two datasets can diverge later without disturbing the dispatch in
    _run() -- e.g. if a soltani_colors-specific or soltani_numbers-specific
    model is ever added.
    """
    model_type = params["model_type"]
    subdata = human_pid.query("trial == @trial & observation <= @observation")
    values = subdata["value"].to_numpy()

    if model_type == "Mean":
        return float(np.mean(values))
    if model_type == "NoisyRL_lambda":
        return _noisy_rl_lambda_response(params, values, trial)
    if model_type == "RL_lambda":
        alpha_0 = float(params["alpha_0"])
        lambda_ = float(params["lambda_"])
        expectation = 0.0
        for n, value in enumerate(values, start=1):
            alpha = alpha_0 / (n ** lambda_)
            error = value - expectation
            expectation += alpha * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    if model_type == "LeakyIntegrator":
        gamma = float(params["gamma"])
        v = 0.0
        for x in values:
            v = gamma * v + (1.0 - gamma) * float(x)
        return float(np.clip(v, -1.0, 1.0))
    if model_type == "PrimacyRecency":
        return _run_primacy_recency(params, values, observation, trial)
    raise AssertionError("unreachable")


def _run_soltani_colors(
    params: dict, human_pid: pd.DataFrame, trial: int, observation: int
) -> float:
    return _run_soltani_common(params, human_pid, trial, observation)


def _run_soltani_numbers(
    params: dict, human_pid: pd.DataFrame, trial: int, observation: int
) -> float:
    return _run_soltani_common(params, human_pid, trial, observation)


