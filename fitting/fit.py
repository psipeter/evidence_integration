"""
Model-fitting orchestration layer for participant-level parameter estimation.

Coordinates Optuna search with k-fold cross-validation over a chosen loss
(``loss_type``), writing participant-specific outputs under a run folder
(see ``run_folder``) for downstream aggregation. Filenames are
``{model_type}_{dataset}_{pid}_params.pkl``, ``_performance.pkl``, and
``_cv_folds.pkl`` (loss type is stored in the params row, not in the name).
When ``loss_type`` is omitted (``None``), the default is task-aware:
``mse`` for carrabin and yoo, ``nll`` for jiang; pass a string explicitly to
override.

Two experiment regimes:

- **Experiment 1 (``loss_type="mse"``):** Mean squared error between model and
  human responses. Carrabin/yoo compare continuous values directly; jiang maps
  the model expectation through ``sigmoid(beta * x)`` and thresholds at 0.5 to
  a binary ``±1`` prediction before squaring error vs. human ``response``. All
  jiang fits suggest ``beta`` (inverse temperature for that map).
- **Experiment 2:** Task-specific losses wired through ``losses.compute_loss``.
  For jiang, ``loss_type="nll"`` uses standard binary NLL with a sigmoid decision
  rule on model expectation (see ``fitting.losses.nll``). Other codes
  (``wasserstein``, ``switch``, ``decay``); ``wasserstein`` is implemented for
  carrabin; ``switch`` and ``decay`` remain stubs until implemented.

``fit_noise_only()`` is reserved for NLL-style noise fitting when those losses
are implemented; it is not used for ``mse``.

Optional ``n_runs`` (default ``1``) is forwarded into model ``params`` when
``> 1`` to control Monte Carlo averaging for stochastic models (e.g. carrabin
``NoisyCounting``, future NEF models). Use ``1`` for fast local runs;
``n_runs >= 20`` is a reasonable choice for cluster fits.

Entry point:
``python -m fitting.fit {dataset} {model_type} {pid} [n_trials] [loss_type] [n_runs] [run_folder]``

Optional 4th token ``n_trials`` (default 100); optional 5th ``loss_type`` (omit
for task-aware default); optional 6th ``n_runs`` (default 1); optional 7th
``run_folder`` (omit for ``data/runs/default``).

**Carrabin:** ``Bayes`` / ``NoisyCounting`` — no fitted params. ``RL`` — ``alpha``.

**Jiang:** ``beta`` is always suggested (threshold sharpness for MSE binarization
and for other losses as needed). ``DeGroot`` — ``omega`` (0.01–10.0); ``RL`` —
``alpha`` (naive update, ignores ``rd``). ``Bayes`` — no structural params beyond
``beta``.

**Yoo:** ``Mean`` — no params. ``RL`` — ``alpha``. ``ADM`` — ``phi``, ``rho``, ``nu``.

Cluster parallelism: shared MySQL Optuna storage; complete ``make_storage()``
before cluster use.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

import fitting.losses as losses
import models.math_models as math_models
import models.recurrent as recurrent
from fitting.param_ranges import MODEL_PARAMS
from utils.paths import RUNS_DIR, data_path

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_LOSS: dict[str, str] = {
    "carrabin": "mse",
    "jiang": "nll",
    "yoo": "mse",
}


def make_storage(host: str, user: str, password: str, study_name: str) -> str:
    """
    Construct a MySQL storage URL for use with Optuna on the cluster.
    Credentials should be passed via environment variables, not hardcoded.
    Returns a connection string suitable for the `storage` argument of fit().

    # TODO: fill in MySQL connection details for Dartmouth Discovery HPC
    """
    raise NotImplementedError


def _log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Log progress every 10 trials."""
    if trial.number % 10 == 0:
        logging.info(
            f"Trial {trial.number}: loss={trial.value:.4f} | "
            f"best={study.best_value:.4f} | params={trial.params}"
        )


def _suggest_params(
    trial: optuna.trial.Trial,
    model_type: str,
    dataset: str,
    pid: int,
    loss_type: str,
    n_runs: int = 1,
) -> dict:
    """Sample model parameters for one Optuna trial."""
    params = {"model_type": model_type, "dataset": dataset, "pid": int(pid)}
    if model_type == "NEF_recurrent":
        params["seed"] = int(pid)
    if n_runs > 1:
        params["n_runs"] = n_runs
    if dataset not in MODEL_PARAMS:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    if model_type not in MODEL_PARAMS[dataset]:
        raise ValueError(
            f"Unsupported model_type {model_type!r} for dataset {dataset!r}"
        )
    for param, (low, high, step) in MODEL_PARAMS[dataset][model_type].items():
        params[param] = trial.suggest_float(param, low, high, step=step)
    return params


def _suggest_noise_only(
    trial: optuna.trial.Trial,
    dataset: str,
    model_type: str,
    pid: int,
    loss_type: str,
) -> dict:
    """Sample only noise parameters for precomputed responses (NLL-style fits)."""
    if loss_type == "mse":
        raise NotImplementedError(
            "fit_noise_only() is only meaningful for NLL-based (or other) loss "
            "functions that include a noise parameter; MSE fits use fit() instead."
        )
    params = {"model_type": model_type, "dataset": dataset, "pid": int(pid)}
    if dataset in ("carrabin", "yoo"):
        params["sigma"] = trial.suggest_float("sigma", 0.001, 1.0, step=0.001)
    elif dataset == "jiang":
        params["beta"] = trial.suggest_float("beta", 0.01, 10.0, step=0.01)
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    return params


def _cross_validate(
    params: dict,
    human: pd.DataFrame,
    k: int = 5,
    loss_type: str = "mse",
) -> tuple[float, list[float]]:
    """Run k-fold CV over trials and return mean loss and per-fold losses."""
    trials = np.asarray(sorted(human["trial"].unique()))
    if trials.size == 0:
        raise ValueError("Human dataframe has no trials for cross-validation")

    rng = np.random.RandomState(seed=int(params["pid"]))
    shuffled = trials.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, k)

    fold_losses: list[float] = []
    for fold_trials in folds:
        holdout_trials = [int(t) for t in fold_trials.tolist()]
        if not holdout_trials:
            continue

        if params["model_type"] == "NEF_recurrent":
            model_fold = recurrent.run(params, trials=holdout_trials)
        else:
            model_fold = math_models.run(params, trials=holdout_trials)
        human_fold = human[human["trial"].isin(holdout_trials)]

        fold_loss = losses.compute_loss(loss_type, params, model_fold, human_fold)
        fold_losses.append(float(fold_loss))

    if not fold_losses:
        raise ValueError("No non-empty CV folds were generated")

    mean_loss = float(np.mean(fold_losses))
    return mean_loss, fold_losses


def _cross_validate_precomputed(
    params: dict,
    human: pd.DataFrame,
    model_responses: pd.DataFrame,
    k: int = 5,
    loss_type: str = "mse",
) -> tuple[float, list[float]]:
    """K-fold CV using fixed model responses (no ``math_models.run`` per fold)."""
    trials = np.asarray(sorted(human["trial"].unique()))
    if trials.size == 0:
        raise ValueError("Human dataframe has no trials for cross-validation")

    rng = np.random.RandomState(seed=int(params["pid"]))
    shuffled = trials.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, k)

    fold_losses: list[float] = []
    for fold_trials in folds:
        holdout_trials = [int(t) for t in fold_trials.tolist()]
        if not holdout_trials:
            continue

        model_fold = model_responses[model_responses["trial"].isin(holdout_trials)]
        human_fold = human[human["trial"].isin(holdout_trials)]

        fold_loss = losses.compute_loss(loss_type, params, model_fold, human_fold)
        fold_losses.append(float(fold_loss))

    if not fold_losses:
        raise ValueError("No non-empty CV folds were generated")

    mean_loss = float(np.mean(fold_losses))
    return mean_loss, fold_losses


def fit(
    dataset: str,
    model_type: str,
    pid: int,
    n_trials: int = 100,
    k: int = 5,
    storage: str | None = None,
    loss_type: str | None = None,
    n_runs: int = 1,
    run_folder: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one participant/model combination and persist outputs."""
    if run_folder is None:
        run_folder = RUNS_DIR / "default"
    run_folder = Path(run_folder)
    if loss_type is None:
        loss_type = DEFAULT_LOSS.get(dataset, "mse")
    if model_type == "NEF_recurrent" and n_runs > 1:
        raise ValueError("NEF_recurrent does not support n_runs > 1")
    start = time.time()
    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
    human = human.query("pid == @pid")
    if human.empty:
        raise ValueError(f"No rows for pid={pid} in dataset={dataset!r}")

    if not MODEL_PARAMS[dataset][model_type]:
        n_trials = 1
        logging.info(
            f"{model_type} has no free parameters; running single evaluation."
        )

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model_type}_{dataset}_{pid}_{loss_type}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_params(
            trial, model_type, dataset, pid, loss_type, n_runs
        )
        mean_loss, fold_losses = _cross_validate(params, human, k=k, loss_type=loss_type)
        trial.set_user_attr("cv_loss_folds", fold_losses)
        return mean_loss

    study.optimize(objective, n_trials=n_trials, callbacks=[_log_callback])

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params.update(
        {
            "model_type": model_type,
            "dataset": dataset,
            "pid": int(pid),
            "loss_type": loss_type,
        }
    )
    if model_type == "NEF_recurrent":
        best_params["seed"] = int(pid)
    best_folds = list(best_trial.user_attrs.get("cv_loss_folds", []))

    runtime = (time.time() - start) / 60.0
    params_df = pd.DataFrame([best_params])
    performance_df = pd.DataFrame(
        [
            {
                "model_type": model_type,
                "dataset": dataset,
                "pid": int(pid),
                "cv_loss_mean": float(best_trial.value),
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
                "loss": float(loss_val),
            }
            for i, loss_val in enumerate(best_folds)
        ]
    )

    params_df.to_pickle(run_folder / f"{model_type}_{dataset}_{pid}_params.pkl")
    performance_df.to_pickle(
        run_folder / f"{model_type}_{dataset}_{pid}_performance.pkl"
    )
    cv_folds_df.to_pickle(
        run_folder / f"{model_type}_{dataset}_{pid}_cv_folds.pkl"
    )

    return params_df, performance_df


def fit_noise_only(
    dataset: str,
    model_type: str,
    pid: int,
    n_trials: int = 100,
    k: int = 5,
    storage: str | None = None,
    loss_type: str | None = None,
    run_folder: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit only observation/decision noise (sigma or beta) given pre-saved model
    responses at ``{run_folder}/{model_type}_{dataset}_{pid}_responses.pkl``.
    """
    if run_folder is None:
        run_folder = RUNS_DIR / "default"
    run_folder = Path(run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    if loss_type is None:
        loss_type = DEFAULT_LOSS.get(dataset, "mse")
    start = time.time()
    responses_path = run_folder / f"{model_type}_{dataset}_{pid}_responses.pkl"
    if not responses_path.exists():
        raise FileNotFoundError(
            f"Pre-saved responses not found: {responses_path}. "
            "Generate responses in the run folder (e.g. python -m jobs.run --rerun ...) "
            "after fitting structural parameters."
        )
    model_responses = pd.read_pickle(responses_path)

    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
    human = human.query("pid == @pid")
    if human.empty:
        raise ValueError(f"No rows for pid={pid} in dataset={dataset!r}")

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model_type}_{dataset}_{pid}_{loss_type}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_noise_only(trial, dataset, model_type, pid, loss_type)
        mean_loss, fold_losses = _cross_validate_precomputed(
            params, human, model_responses, k=k, loss_type=loss_type
        )
        trial.set_user_attr("cv_loss_folds", fold_losses)
        return mean_loss

    study.optimize(objective, n_trials=n_trials, callbacks=[_log_callback])

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params.update(
        {
            "model_type": model_type,
            "dataset": dataset,
            "pid": int(pid),
            "loss_type": loss_type,
        }
    )
    best_folds = list(best_trial.user_attrs.get("cv_loss_folds", []))

    runtime = (time.time() - start) / 60.0
    params_df = pd.DataFrame([best_params])
    performance_df = pd.DataFrame(
        [
            {
                "model_type": model_type,
                "dataset": dataset,
                "pid": int(pid),
                "cv_loss_mean": float(best_trial.value),
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
                "loss": float(loss_val),
            }
            for i, loss_val in enumerate(best_folds)
        ]
    )

    params_df.to_pickle(run_folder / f"{model_type}_{dataset}_{pid}_params.pkl")
    performance_df.to_pickle(
        run_folder / f"{model_type}_{dataset}_{pid}_performance.pkl"
    )
    cv_folds_df.to_pickle(
        run_folder / f"{model_type}_{dataset}_{pid}_cv_folds.pkl"
    )

    return params_df, performance_df


if __name__ == "__main__":
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    pid = int(sys.argv[3])
    n_trials = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    loss_type = sys.argv[5] if len(sys.argv) > 5 else None
    n_runs = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    run_folder = sys.argv[7] if len(sys.argv) > 7 else None

    logging.basicConfig(level=logging.INFO)
    params_df, performance_df = fit(
        dataset,
        model_type,
        pid,
        n_trials=n_trials,
        loss_type=loss_type,
        n_runs=n_runs,
        run_folder=run_folder,
    )
    elapsed = float(performance_df.loc[0, "runtime"])
    logging.info(f"Completed in {elapsed:.2f} min")
    logging.info(performance_df.to_string())
    logging.info(params_df.to_string())
