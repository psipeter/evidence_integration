# NOTE: jiang/usher model classes archived in archive/models/archive_math_models.py
# NOTE: NoisyCounting and NoisyRL_lambda retired (state-noise models phased
# out of active analysis) -- archived in
# archive/models/archive_math_models_noise.py. See docs/DECISIONS.md.
"""
Mathematical (non-NEF) models of evidence integration.

Expectations are computed from empirical sequences in per-dataset pickle files
and collected into a single tabular format with model ``response`` values.

**Datasets and model types**

- **carrabin:** ``Mean`` (optimal), ``RL`` (naive), ``PrimacyRecency`` (flexible temporal weighting)
- **soltani:** ``Mean``, ``LeakyIntegrator``, ``PrimacyRecency``, ``RL_lambda``
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
    {"Mean", "RL", "RL_lambda", "LeakyIntegrator", "PrimacyRecency"}
)
_YOO_MODELS = frozenset({"Mean", "LeakyIntegrator", "PrimacyRecency", "RL", "RL_lambda"})
# Deliberately narrower than carrabin/yoo: these four together are meant to
# capture recency-biased (non-shrinking-learning-rate) behavior, which is
# what this task's human data actually looks like (see the conversation that
# motivated this integration) -- no plain fixed-alpha RL (superseded by
# RL_lambda, which subsumes it at lambda_->0) for either task dataset.

_SOLTANI_COLORS_MODELS = frozenset(
    {"Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda"})
_SOLTANI_NUMBERS_MODELS = frozenset(
    {"Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda"})


# Base models that add_noise() knows how to wrap. Any model whose run() output
# is a deterministic function of (dataset, pid, its own params) qualifies --
# add_noise never touches per-model branches, it only calls run() once and adds
# i.i.d. noise on top. This is the ACTIVE noise mechanism (see docs/DECISIONS.md
# for why the alternative, compounding state noise via NoisyRL_lambda, was
# retired instead).
_NOISE_WRAPPABLE_BASE_MODELS = frozenset(
    {"Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda"})

# model_type suffix -> base model name, and back. "_resp_noise" is the ONLY
# mechanism add_noise implements (i.i.d. per observation); it is not a stand-in
# for arbitrary noise types, so do not overload this suffix for anything else.
_RESP_NOISE_SUFFIX = "_resp_noise"

# Fixed small-int id per wrapped base model, for _resp_noise_seed below --
# NOT a hash of the model_type STRING (Python's str hashing is randomized
# per-process via PYTHONHASHSEED, unlike int/tuple-of-int hashing, which
# utils.run_params.trial_seed relies on being stable). Every model in
# _NOISE_WRAPPABLE_BASE_MODELS needs an entry.
_RESP_NOISE_MODEL_SEED_ID = {"Mean": 0, "LeakyIntegrator": 1, "PrimacyRecency": 2, "RL_lambda": 3}


def _resp_noise_seed(pid: int, model_type: str) -> int:
    """Deterministic seed for add_noise's own i.i.d. noise draw, unique per
    (pid, base model) -- fixes a real bug: add_noise previously defaulted to
    RandomState(0) whenever params had no explicit "seed" key, which was
    ALWAYS the case for every _resp_noise fit (fitting.fit never sets one).
    Since add_noise draws a full (n_sims, len(mu)) array from ONE seeded
    RandomState call, this meant every pid AND every wrapped model reused
    the EXACT SAME underlying z-draw sequence, just rescaled by that pid's
    own fitted sigma_resp -- not independent noise at all.

    Uses utils.run_params.trial_seed (int/tuple-of-int hashing only, stable
    across process invocations) with model identity folded in via
    _RESP_NOISE_MODEL_SEED_ID, so a fresh Python process per (dataset,
    model_type, pid) fit -- the normal fitting.fit entry point -- still
    gets a reproducible, but genuinely distinct, seed for each.
    """
    from utils.run_params import trial_seed
    model = base_model_of(model_type)
    return trial_seed(int(pid), _RESP_NOISE_MODEL_SEED_ID[model])


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

    No trial-structure or replay logic is needed: i.i.d. noise has no
    sequential dependency to preserve, so this is a single vectorised draw.

    This is what lets a deterministic model enter an NLL comparison on equal
    footing with a stochastic one, for exactly ONE extra parameter, without
    any dependency on a prior RMSE fit -- mu comes fresh from run(), not from
    a saved _responses.pkl. `params["model_type"]` may be either the bare
    base name ("RL_lambda") or the fitting-time suffixed name
    ("RL_lambda_resp_noise") -- base_model_of() strips the suffix if present.

    SEEDING: if `params` has no explicit "seed" key (the normal case --
    fitting.fit never sets one), the draw is seeded via
    _resp_noise_seed(pid, model_type) -- unique per (pid, base model) --
    rather than silently defaulting to a fixed RandomState(0) shared by
    every pid and every wrapped model. An explicit "seed" in `params` still
    overrides this.
    """
    model_type = base_model_of(params["model_type"])
    if model_type not in _NOISE_WRAPPABLE_BASE_MODELS:
        raise ValueError(
            f"add_noise wraps a deterministic base model; {params['model_type']!r} "
            f"is not in {sorted(_NOISE_WRAPPABLE_BASE_MODELS)} (after stripping any "
            f"_resp_noise suffix).")

    base_params = {k: v for k, v in params.items() if k != "sigma_resp"}
    base_params["model_type"] = model_type
    mu_df = run(base_params).sort_values(["trial", "observation"])
    mu = mu_df["response"].to_numpy(float)

    if "seed" in params:
        seed = int(params["seed"])
    else:
        seed = _resp_noise_seed(int(params["pid"]), params["model_type"])
    rng = np.random.RandomState(seed)
    ens = np.clip(mu[np.newaxis, :] + rng.normal(0.0, sigma_resp, (n_sims, len(mu))),
                 -1.0, 1.0)
    if return_index:
        return ens, mu_df[["trial", "observation"]].reset_index(drop=True)
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
    if model_type == "RL":
        expectation = 0.0
        for value in values:
            error = value - expectation
            expectation += params["alpha"] * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
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
    raise AssertionError(
        f"unreachable for model_type={model_type!r} -- if this is "
        f"NoisyCounting/NoisyRL_lambda, that model is retired; see "
        f"archive/models/archive_math_models_noise.py")


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
    raise AssertionError(
        f"unreachable for model_type={model_type!r} -- if this is "
        f"NoisyRL_lambda, that model is retired; see "
        f"archive/models/archive_math_models_noise.py")


def _run_soltani_common(
    params: dict, human_pid: pd.DataFrame, trial: int, observation: int
) -> float:
    """Shared implementation for soltani_colors and soltani_numbers.

    Both datasets are rescaled to the same [-1, 1] scale carrabin/yoo use
    (see scripts/build_model_inputs.py) and both only support the same
    four model types (_SOLTANI_COLORS_MODELS == _SOLTANI_NUMBERS_MODELS), so
    there is no dataset-specific branching needed here. _run_soltani_colors/
    _run_soltani_numbers below are kept as separate named entry points
    (rather than calling this directly from _run()) so the two datasets
    can diverge later without disturbing the dispatch in _run().
    """
    model_type = params["model_type"]
    subdata = human_pid.query("trial == @trial & observation <= @observation")
    values = subdata["value"].to_numpy()

    if model_type == "Mean":
        return float(np.mean(values))
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
    raise AssertionError(
        f"unreachable for model_type={model_type!r} -- if this is "
        f"NoisyRL_lambda, that model is retired; see "
        f"archive/models/archive_math_models_noise.py")


def _run_soltani_colors(
    params: dict, human_pid: pd.DataFrame, trial: int, observation: int
) -> float:
    return _run_soltani_common(params, human_pid, trial, observation)


def _run_soltani_numbers(
    params: dict, human_pid: pd.DataFrame, trial: int, observation: int
) -> float:
    return _run_soltani_common(params, human_pid, trial, observation)
