"""
Model-fitting orchestration layer for participant-level parameter estimation.

This module coordinates parameter search with Optuna and cross-validation,
writing participant-specific outputs to ``data/`` for downstream aggregation.
In practice this is a two-stage workflow: first, cross-validation identifies a
good parameter region; second, the selected parameters are saved for downstream
full-dataset analyses managed by ``rerun.py`` and ``collect.py``.
Cluster parallelism is supported via shared MySQL Optuna storage; complete
``make_storage()`` before cluster use.

Entry point:
``python -m fitting.fit {dataset} {model_type} {pid}``
"""

import logging
import sys
import time

import numpy as np
import optuna
import pandas as pd

import fitting.losses as losses
import models.math_models as math_models
from utils.paths import data_path

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_storage(host: str, user: str, password: str, study_name: str) -> str:
    """
    Construct a MySQL storage URL for use with Optuna on the cluster.
    Credentials should be passed via environment variables, not hardcoded.
    Returns a connection string suitable for the `storage` argument of fit().

    # TODO: fill in MySQL connection details for Dartmouth Discovery HPC
    """
    raise NotImplementedError


def _suggest_params(
    trial: optuna.trial.Trial, model_type: str, dataset: str, pid: int
) -> dict:
    """Sample model parameters for one Optuna trial."""
    params = {"model_type": model_type, "dataset": dataset, "pid": int(pid)}

    if dataset == "carrabin":
        params["sigma"] = trial.suggest_float("sigma", 0.001, 1.0, step=0.001)
        if model_type == "RL_n":
            params["alpha"] = trial.suggest_float("alpha", 0.001, 1.0, step=0.001)
        elif model_type in ("B_n", "DG_n"):
            pass
        else:
            raise ValueError(f"Unsupported carrabin model_type: {model_type!r}")

    elif dataset == "jiang":
        params["beta"] = trial.suggest_float("beta", 0.01, 10.0, step=0.01)
        if model_type == "DG_z":
            params["z"] = trial.suggest_float("z", 0.01, 1.0, step=0.01)
        elif model_type == "RL_z":
            params["alpha"] = trial.suggest_float("alpha", 0.01, 1.5, step=0.01)
            params["z"] = trial.suggest_float("z", 0.01, 1.0, step=0.01)
        else:
            raise ValueError(f"Unsupported jiang model_type: {model_type!r}")

    elif dataset == "yoo":
        params["sigma"] = trial.suggest_float("sigma", 0.001, 1.0, step=0.001)
        if model_type == "DG":
            pass
        elif model_type == "RL_l":
            params["alpha"] = trial.suggest_float("alpha", 0.001, 3.0, step=0.001)
            params["lambda"] = trial.suggest_float("lambda", 0.001, 3.0, step=0.001)
        elif model_type == "ADM":
            params["primacy"] = trial.suggest_float("primacy", 0.001, 1.0, step=0.001)
            params["recency"] = trial.suggest_float("recency", 0.001, 1.0, step=0.001)
            params["nu"] = trial.suggest_float("nu", 0.001, 0.5, step=0.001)
        else:
            raise ValueError(f"Unsupported yoo model_type: {model_type!r}")

    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    return params


def _cross_validate(params: dict, human: pd.DataFrame, k: int = 5) -> tuple[float, list[float]]:
    """Run k-fold CV over trials and return mean NLL and fold NLLs."""
    trials = np.asarray(sorted(human["trial"].unique()))
    if trials.size == 0:
        raise ValueError("Human dataframe has no trials for cross-validation")

    rng = np.random.RandomState(seed=int(params["pid"]))
    shuffled = trials.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, k)

    fold_nlls: list[float] = []
    for fold_trials in folds:
        holdout_trials = [int(t) for t in fold_trials.tolist()]
        if not holdout_trials:
            continue

        model_fold = math_models.run(params, trials=holdout_trials)
        human_fold = human[human["trial"].isin(holdout_trials)]

        fold_nll = losses.nll(params, model_fold, human_fold)
        fold_nlls.append(float(fold_nll))

    if not fold_nlls:
        raise ValueError("No non-empty CV folds were generated")

    mean_nll = float(np.mean(fold_nlls))
    return mean_nll, fold_nlls


def fit(
    dataset: str,
    model_type: str,
    pid: int,
    n_trials: int = 200,
    k: int = 5,
    storage: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one participant/model combination and persist outputs."""
    start = time.time()
    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
    human = human.query("pid == @pid")
    if human.empty:
        raise ValueError(f"No rows for pid={pid} in dataset={dataset!r}")

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model_type}_{dataset}_{pid}",
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_params(trial, model_type, dataset, pid)
        mean_nll, fold_nlls = _cross_validate(params, human, k=k)
        trial.set_user_attr("cv_nll_folds", fold_nlls)
        return mean_nll

    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params.update({"model_type": model_type, "dataset": dataset, "pid": int(pid)})
    best_folds = list(best_trial.user_attrs.get("cv_nll_folds", []))

    runtime = (time.time() - start) / 60.0
    params_df = pd.DataFrame([best_params])
    performance_df = pd.DataFrame(
        [
            {
                "model_type": model_type,
                "dataset": dataset,
                "pid": int(pid),
                "cv_nll_mean": float(best_trial.value),
                "runtime": float(runtime),
            }
        ]
    )
    cv_folds_df = pd.DataFrame(
        [
            {
                "model_type": model_type,
                "dataset": dataset,
                "pid": int(pid),
                "fold": int(i + 1),
                "nll": float(nll_val),
            }
            for i, nll_val in enumerate(best_folds)
        ]
    )

    params_df.to_pickle(data_path(f"{model_type}_{dataset}_{pid}_params.pkl"))
    performance_df.to_pickle(data_path(f"{model_type}_{dataset}_{pid}_performance.pkl"))
    cv_folds_df.to_pickle(data_path(f"{model_type}_{dataset}_{pid}_cv_folds.pkl"))

    return params_df, performance_df


if __name__ == "__main__":
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    pid = int(sys.argv[3])

    logging.basicConfig(level=logging.INFO)
    params_df, performance_df = fit(dataset, model_type, pid)
    elapsed = float(performance_df.loc[0, "runtime"])
    logging.info(f"Completed in {elapsed:.2f} min")
    logging.info(performance_df.to_string())
    logging.info(params_df.to_string())
