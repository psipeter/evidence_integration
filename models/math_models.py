# NOTE: jiang/usher model classes archived in archive/models/archive_math_models.py
"""
Mathematical (non-NEF) models of evidence integration.

Expectations are computed from empirical sequences in per-dataset pickle files
and collected into a single tabular format with model ``response`` values.

**Datasets and model types**

- **carrabin:** ``Mean`` (optimal), ``NoisyCounting`` (human-matching), ``RL`` (naive), ``PrimacyRecency`` (flexible temporal weighting)
- **yoo:** ``Mean`` (optimal), ``PrimacyRecency`` (flexible temporal weighting), ``RL`` (naive)
- diederen models archived in ``archive/misc/math_models_diederen.py``

**Unified interface**

Every model is run via ``run(params, save=False, trials=None)``. Required keys in
``params`` for all models:

- ``"model_type"`` (``str``): one of the strings above for the chosen dataset
- ``"dataset"`` (``str``): ``"carrabin"`` or ``"yoo"``
- ``"pid"`` (``int``): participant id

Additional keys are model-specific (learning rates, noise scales, etc.). The
optional ``trials`` argument restricts execution to a subset of trial ids.
"""

import numpy as np
import pandas as pd

from utils.paths import data_path
from utils.carrabin_transform import apply_carrabin_transform
from utils.run_params import trial_seed as _trial_seed


_CARRABIN_MODELS = frozenset(
    {"Mean", "NoisyCounting", "RL", "RL_lambda", "LeakyIntegrator", "PrimacyRecency"}
)
_YOO_MODELS = frozenset({"Mean", "LeakyIntegrator", "PrimacyRecency", "RL", "RL_lambda"})


def run(params: dict, save: bool = False, trials: list | None = None) -> pd.DataFrame:
    for key in ("model_type", "dataset", "pid"):
        if key not in params:
            raise KeyError(f"params must include {key!r}")

    model_type: str = params["model_type"]
    dataset: str = params["dataset"]
    pid: int = int(params["pid"])

    _validate_model_dataset(model_type, dataset)

    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
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
        fname = f"{model_type}_{dataset}_{pid}_responses.pkl"
        out.to_pickle(data_path(fname))
    return out


def _validate_model_dataset(model_type: str, dataset: str) -> None:
    if dataset == "carrabin":
        allowed = _CARRABIN_MODELS
    elif dataset == "yoo":
        allowed = _YOO_MODELS
    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected 'carrabin' or 'yoo'"
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


