"""
Mathematical (non-NEF) models of evidence integration.

Expectations are computed from empirical sequences in per-dataset pickle files
and collected into a single tabular format with model ``response`` values.
Ported and redesigned from
``get_expectations_carrabin``, ``get_expectations_jiang``, and
``get_expectations_yoo`` in ``observational-learning-social-networks/fit.py``.

**Datasets and model types**

- **carrabin:** ``Bayes`` (optimal), ``NoisyCounting`` (human-matching), ``RL`` (naive)
- **jiang:** ``Bayes`` (optimal), ``DeGroot`` (human-matching), ``RL`` (naive)
- **yoo:** ``Mean`` (optimal), ``ADM`` (human-matching), ``RL`` (naive)

**Unified interface**

Every model is run via ``run(params, save=False, trials=None)``. Required keys in
``params`` for all models:

- ``"model_type"`` (``str``): one of the strings above for the chosen dataset
- ``"dataset"`` (``str``): ``"carrabin"``, ``"jiang"``, or ``"yoo"``
- ``"pid"`` (``int``): participant id

Additional keys are model-specific (learning rates, noise scales, etc.). The
optional ``trials`` argument restricts execution to a subset of trial ids.
"""

import numpy as np
import pandas as pd

from utils.paths import data_path

_CARRABIN_MODELS = frozenset({"Bayes", "NoisyCounting", "RL"})
_JIANG_MODELS = frozenset({"Bayes", "DeGroot", "RL"})
_YOO_MODELS = frozenset({"Mean", "ADM", "RL"})


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

    rows: list[dict] = []
    if dataset in ("carrabin", "yoo"):
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
    else:
        pairs = (
            human_pid[["trial", "stage"]]
            .drop_duplicates()
            .sort_values(["trial", "stage"])
        )
        for _, pr in pairs.iterrows():
            trial = int(pr["trial"])
            stage = int(pr["stage"])
            estimate = _run(params, human_pid, trial, stage)
            rows.append(
                {
                    "model_type": model_type,
                    "pid": pid,
                    "trial": trial,
                    "stage": stage,
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
    elif dataset == "jiang":
        allowed = _JIANG_MODELS
    elif dataset == "yoo":
        allowed = _YOO_MODELS
    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected 'carrabin', 'jiang', or 'yoo'"
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
    if dataset == "jiang":
        return _run_jiang(params, human_pid, trial, step)
    if dataset == "yoo":
        return _run_yoo(params, human_pid, trial, step)
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
        # state) and Eq. 34 (response), on [-1, 1]. Default n_runs=1 is fast for
        # local testing; use n_runs>=20 when fitting on the cluster; set
        # params["n_runs"].
        mu = float(params["mu"])
        sigma_c = float(params["sigma_c"])
        nu = float(params["nu"])
        n_runs = int(params.get("n_runs", 1))
        if len(values) == 0:
            return 0.0
        run_responses: list[float] = []
        for run_idx in range(n_runs):
            seed = 100 * int(params["pid"]) + 1000 * trial + run_idx
            rng = np.random.RandomState(seed)
            r = 0.0
            p_hat = 0.0
            for x in values:
                xi = rng.normal(0.0, sigma_c)
                r = r + float(x) * mu + xi
                epsilon = rng.normal(0.0, nu)
                p_hat = p_hat + (r - p_hat) * float(np.exp(epsilon))
                p_hat = float(np.clip(p_hat, -1.0, 1.0))
            run_responses.append(p_hat)
        return float(np.mean(run_responses))
    if model_type == "RL":
        expectation = 0.0
        for value in values:
            error = value - expectation
            expectation += params["alpha"] * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    raise AssertionError("unreachable")


def _run_jiang(
    params: dict, human_pid: pd.DataFrame, trial: int, stage: int
) -> float:
    model_type = params["model_type"]
    subdata = human_pid.query("trial == @trial & stage <= @stage")
    values = subdata["value"].to_numpy(dtype=float)
    rds = subdata["rd"].to_numpy(dtype=float)

    if model_type == "Bayes":
        # TODO: implement full Bayesian model from Jiang & Zhu appendix
        # Requires network adjacency data (jiang_networks.pkl, not yet available)
        raise NotImplementedError("Bayes model for jiang not yet implemented")
    if model_type == "DeGroot":
        omega = params["omega"]
        weights = 1.0 + omega * rds
        wsum = np.sum(weights)
        if wsum == 0:
            return 0.0
        expectation = float(np.dot(weights, values) / wsum)
        return float(np.clip(expectation, -1, 1))
    if model_type == "RL":
        alpha = params["alpha"]
        weight = alpha
        expectation = 0.0
        for value in values:
            error = value - expectation
            expectation += weight * error
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
    if model_type == "ADM":
        phi = params["phi"]
        rho = params["rho"]
        nu = params["nu"]
        n = len(values)
        # REVIEW: Exponent uses outer ``observation`` (target index) and loop index ``o``; matches fit.py ADM loop with stage renamed.
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
