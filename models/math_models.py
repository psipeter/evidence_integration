# NOTE: jiang/usher model classes archived in archive/models/archive_math_models.py
"""
Mathematical (non-NEF) models of evidence integration.

Expectations are computed from empirical sequences in per-dataset pickle files
and collected into a single tabular format with model ``response`` values.

**Datasets and model types**

- **carrabin:** ``Bayes`` (optimal), ``NoisyCounting`` (human-matching), ``RL`` (naive)
- **yoo:** ``Mean`` (optimal), ``ADM`` (human-matching), ``RL`` (naive)
- **diederen:** ``Mean`` (optimal), ``RL`` (naive), ``RL_lambda``, ``PearceHall``

**Unified interface**

Every model is run via ``run(params, save=False, trials=None)``. Required keys in
``params`` for all models:

- ``"model_type"`` (``str``): one of the strings above for the chosen dataset
- ``"dataset"`` (``str``): ``"carrabin"``, ``"yoo"``, or ``"diederen"``
- ``"pid"`` (``int``): participant id

Additional keys are model-specific (learning rates, noise scales, etc.). The
optional ``trials`` argument restricts execution to a subset of trial ids.
"""

import numpy as np
import pandas as pd

from utils.paths import data_path
from utils.run_params import trial_seed as _trial_seed


_CARRABIN_MODELS = frozenset(
    {"Bayes", "NoisyCounting", "RL", "RL_lambda"}
)
_YOO_MODELS = frozenset({"Mean", "ADM", "RL", "RL_lambda"})
_DIEDEREN_MODELS = frozenset(
    {"Mean", "RL", "RL_lambda", "PearceHall"}
)


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

    if dataset in ("diederen", "diederen_group"):
        # Include catch trials in simulation: reward is shown on catch trials
        # and updates the running estimate. Filter only missed rows.
        human_pid = human_pid[~human_pid["missed"]]
    else:
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

    out = pd.DataFrame(rows)
    if save:
        fname = f"{model_type}_{dataset}_{pid}_responses.pkl"
        out.to_pickle(data_path(fname))
    return out


def _validate_model_dataset(model_type: str, dataset: str) -> None:
    if dataset == "carrabin":
        allowed = _CARRABIN_MODELS
    elif dataset == "yoo":
        allowed = _YOO_MODELS
    elif dataset in ("diederen", "diederen_group"):
        allowed = _DIEDEREN_MODELS
    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected 'carrabin', 'yoo', or 'diederen'"
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
    if dataset in ("diederen", "diederen_group"):
        return _run_diederen(params, human_pid, trial, step)
    raise AssertionError("unreachable")


def _run_carrabin(
    params: dict, human_pid: pd.DataFrame, trial: int, observation: int
) -> float:
    model_type = params["model_type"]
    subdata = human_pid.query("trial == @trial & observation <= @observation")
    values = subdata["value"].to_numpy()
    t = len(values)
    n_R = np.sum((values + 1) / 2)

    if model_type == "Bayes":
        p_star = (n_R + 1) / (t + 2)
        expectation = 2 * p_star - 1
        return float(expectation)
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
    if model_type == "ADM":
        phi = params["phi"]
        rho = params["rho"]
        nu = params.get("nu", 0.01)  # fixed per Yoo et al.; not a free parameter
        n = len(values)
        weights = np.array(
            [
                (1.0 - (1.0 - phi ** (o + 1)) * (1.0 - rho ** (observation - o)))
                * (1.0 - nu)
                + nu
                for o in range(n)
            ],
            dtype=float,
        )
        return float(np.dot(weights, values) / np.sum(weights))
    raise AssertionError("unreachable")


def _run_diederen(
    params: dict,
    human_pid: pd.DataFrame,
    trial: int,
    observation: int,
) -> float:
    model_type = params["model_type"]
    subdata = human_pid.query("trial == @trial & observation <= @observation")
    values = subdata["value"].to_numpy()

    if model_type == "Mean":
        if len(values) == 0:
            return 0.0
        return float(np.mean(values))

    if model_type == "RL":
        alpha = float(params["alpha"])
        expectation = 0.0
        for value in values:
            expectation += alpha * (value - expectation)
            expectation = float(np.clip(expectation, -1.0, 1.0))
        return expectation

    if model_type == "RL_lambda":
        alpha_0 = float(params["alpha_0"])
        lambda_ = float(params["lambda_"])
        expectation = 0.0
        for n, value in enumerate(values, start=1):
            alpha = alpha_0 / (n ** lambda_)
            expectation += alpha * (value - expectation)
            expectation = float(np.clip(expectation, -1.0, 1.0))
        return expectation

    if model_type == "PearceHall":
        alpha_0 = float(params["alpha_0"])
        eta = float(params["eta"])
        expectation = 0.0
        alpha = alpha_0
        for value in values:
            delta = value - expectation
            expectation += alpha * delta
            expectation = float(np.clip(expectation, -1.0, 1.0))
            alpha = float(np.clip(eta * abs(delta) + (1.0 - eta) * alpha, 0.0, 2.0))
        return expectation

    raise AssertionError(f"unreachable model_type={model_type!r}")
