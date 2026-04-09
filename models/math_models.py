"""
Mathematical (non-NEF) models of evidence integration.

Expectations are computed from empirical sequences in per-dataset pickle files
and collected into a single tabular format with model ``response`` values.
Ported and redesigned from
``get_expectations_carrabin``, ``get_expectations_jiang``, and
``get_expectations_yoo`` in ``observational-learning-social-networks/fit.py``.

**Datasets and model types**

- **carrabin:** ``RL_n``, ``B_n``, ``DG_n``
- **jiang:** ``DG_z``, ``RL_z``
- **yoo:** ``DG``, ``RL_l``, ``ADM``

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

_CARRABIN_MODELS = frozenset({"RL_n", "B_n", "DG_n"})
_JIANG_MODELS = frozenset({"DG_z", "RL_z"})
_YOO_MODELS = frozenset({"DG", "RL_l", "ADM"})


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
        fname = f"{model_type}_{dataset}_{pid}_estimates.pkl"
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
    pid = int(params["pid"])

    if dataset == "carrabin":
        return _run_carrabin(params, human_pid, trial, step, pid)
    if dataset == "jiang":
        return _run_jiang(params, human_pid, trial, step)
    if dataset == "yoo":
        return _run_yoo(params, human_pid, trial, step)
    raise AssertionError("unreachable")


def _run_carrabin(
    params: dict, human_pid: pd.DataFrame, trial: int, observation: int, pid: int
) -> float:
    model_type = params["model_type"]
    rng = np.random.RandomState(seed=100 * pid + 1000 * trial)

    subdata = human_pid.query("trial == @trial & observation <= @observation")
    values = subdata["value"].to_numpy()
    expectation = 0.0
    for c, value in enumerate(values):
        error = value - expectation
        if model_type == "B_n":
            weight = 1.0 / (c + 3)
            eps = rng.normal(0, params["sigma"])
        elif model_type == "DG_n":
            weight = 1.0 / (c + 1)
            eps = rng.normal(0, params["sigma"])
        elif model_type == "RL_n":
            weight = params["alpha"]
            eps = rng.normal(0, params["sigma"])
        else:
            raise AssertionError("unreachable")
        expectation += weight * error + eps
        expectation = float(np.clip(expectation, -1, 1))
    return expectation


def _run_jiang(
    params: dict, human_pid: pd.DataFrame, trial: int, stage: int
) -> float:
    model_type = params["model_type"]
    subdata = human_pid.query("trial == @trial & stage <= @stage")
    values = subdata["value"].to_numpy()
    rds = subdata["rd"].to_numpy()
    expectation = 0.0
    for c, value in enumerate(values):
        stg = int(subdata.iloc[c]["stage"])
        error = value - expectation
        # REVIEW: rd forced to 0 for early stages matches fit.py; confirm this matches task design.
        rd = 0.0 if stg in (0, 1) else float(rds[c])
        if model_type == "DG_z":
            weight = 1.0 / (c + 1) + params["z"] * rd
        elif model_type == "RL_z":
            weight = 1.0 if stg == 0 else params["alpha"] + params["z"] * rd
        else:
            raise AssertionError("unreachable")
        weight = float(np.clip(weight, 0, 1))
        expectation += weight * error
        expectation = float(np.clip(expectation, -1, 1))
    return expectation


def _run_yoo(
    params: dict,
    human_pid: pd.DataFrame,
    trial: int,
    observation: int,
) -> float:
    model_type = params["model_type"]
    subdata = human_pid.query("trial == @trial & observation <= @observation")
    values = subdata["value"].to_numpy()

    if model_type == "DG":
        return float(np.mean(values))
    if model_type == "RL_l":
        expectation = 0.0
        for o, value in enumerate(values):
            error = value - expectation
            weight = params["alpha"] * np.power(o + 1, -params["lambda"])
            expectation += weight * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    if model_type == "ADM":
        primacy = params["primacy"]
        recency = params["recency"]
        nu = params["nu"]
        n = len(values)
        # REVIEW: Exponent uses outer ``observation`` (target index) and loop index ``o``; matches fit.py ADM loop with stage renamed.
        weights = np.array(
            [
                (1.0 - (1.0 - primacy ** (o + 1)) * (1.0 - recency ** (observation - o)))
                * (1.0 - nu)
                + nu
                for o in range(n)
            ],
            dtype=float,
        )
        return float(np.dot(weights, values) / np.sum(weights))
    raise AssertionError("unreachable")
