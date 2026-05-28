"""
Participant-level model fitting via Optuna (TPE) and k-fold CV.

Objective: RMSE from ``fitting.losses.compute_loss``.

Entry point::

    python -m fitting.fit {dataset} {model_type} {pid} [n_trials] [k] [run_folder] [optuna_seed]

Writes ``{model_type}_{dataset}_{pid}_params.pkl``, ``_performance.pkl``, and
``_folds.pkl`` under ``run_folder``. SLURM jobs are submitted via ``fitting.submit``.
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
from models import NEF
from fitting.model_params import MODEL_PARAMS
from utils.paths import RUNS_DIR, data_path, resolve_run_folder
from utils.save_responses import save as save_responses

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Log progress every 10 trials."""
    trial_loss = f"{trial.value:.4f}" if trial.value is not None else "failed"
    try:
        best = f"{study.best_value:.4f}"
    except ValueError:
        best = "n/a"
    logging.info(
        f"Trial {trial.number}: loss={trial_loss} | "
        f"best={best} | params={trial.params}"
    )


def _suggest_params(
    trial: optuna.trial.Trial,
    model_type: str,
    dataset: str,
    pid: int,
) -> dict:
    """Sample model parameters for one Optuna trial."""
    params = {"model_type": model_type, "dataset": dataset, "pid": int(pid)}
    if dataset not in MODEL_PARAMS:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    if model_type not in MODEL_PARAMS[dataset]:
        raise ValueError(
            f"Unsupported model_type {model_type!r} for dataset {dataset!r}"
        )
    model_spec = MODEL_PARAMS[dataset][model_type]
    fixed_params = model_spec.get("fixed", {})
    if fixed_params:
        params.update(fixed_params)

    for param, spec in model_spec.items():
        if param == "fixed":
            continue
        low, high, step = spec
        # Keep integer-valued hyperparameters discrete in Optuna.
        if float(step).is_integer() and float(low).is_integer() and float(high).is_integer():
            params[param] = trial.suggest_int(param, int(low), int(high), step=int(step))
        else:
            params[param] = trial.suggest_float(param, low, high, step=step)
    return params


def _cross_validate(
    params: dict,
    model_responses: pd.DataFrame,
    human: pd.DataFrame,
    k: int = 5,
) -> tuple[float, list[float]]:
    """K-fold CV using pre-computed responses. Works for NEF and math models."""
    trials = np.asarray(sorted(human["trial"].unique()))
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
        fold_loss = losses.compute_loss(params, model_fold, human_fold)
        fold_losses.append(float(fold_loss))

    if not fold_losses:
        raise ValueError("No non-empty CV folds were generated")
    return float(np.mean(fold_losses)), fold_losses


def fit(
    dataset: str,
    model_type: str,
    pid: int,
    n_trials: int = 100,
    k: int = 5,
    storage: str | None = None,
    run_folder: Path | str | None = None,
    optuna_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one participant/model combination and persist outputs."""
    if run_folder is None:
        run_folder = RUNS_DIR / "default"
    run_folder = resolve_run_folder(run_folder)
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
        study_name=f"{model_type}_{dataset}_{pid}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=optuna_seed),
    )

    trial_records: list[dict] = []

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_params(trial, model_type, dataset, pid)
        params["seed"] = abs(hash((int(params["pid"]), trial.number))) % (2**31)
        params["base_seed"] = params["seed"]

        trial_wall_start = time.time()
        if model_type == "NEF":
            model_responses_full = NEF.run(params)
        else:
            model_responses_full = math_models.run(params)

        mean_loss, fold_losses = _cross_validate(
            params, model_responses_full, human, k=k
        )

        trial.set_user_attr(
            "runtime_minutes",
            (time.time() - trial_wall_start) / 60.0,
        )

        for i, fold_loss in enumerate(fold_losses):
            record = {
                "model_type": model_type,
                "dataset": dataset,
                "pid": int(pid),
                "trial_number": trial.number,
                "fold": int(i + 1),
                "loss": float(fold_loss),
            }
            for param_name, param_val in params.items():
                if param_name not in (
                    "model_type",
                    "dataset",
                    "pid",
                    "seed",
                    "base_seed",
                ):
                    if param_name not in record:
                        record[param_name] = (
                            float(param_val)
                            if isinstance(param_val, (int, float))
                            else param_val
                        )
            trial_records.append(record)
        return mean_loss

    study.optimize(objective, n_trials=n_trials, callbacks=[_log_callback])
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params.update(
        {
            "model_type": model_type,
            "dataset": dataset,
            "pid": int(pid),
        }
    )
    best_params["seed"] = abs(hash((int(pid), best_trial.number))) % (2**31)
    best_params["base_seed"] = best_params["seed"]

    params_df = pd.DataFrame([best_params])
    performance_df = pd.DataFrame(
        [
            {
                "model_type": model_type,
                "dataset": dataset,
                "pid": int(pid),
                "loss": float(best_trial.value),
                "runtime": float(
                    best_trial.user_attrs.get("runtime_minutes", float("nan"))
                ),
            }
        ]
    )
    folds_df = pd.DataFrame(trial_records)
    folds_df.to_pickle(run_folder / f"{model_type}_{dataset}_{pid}_folds.pkl")

    params_df.to_pickle(run_folder / f"{model_type}_{dataset}_{pid}_params.pkl")
    performance_df.to_pickle(
        run_folder / f"{model_type}_{dataset}_{pid}_performance.pkl"
    )

    if model_type == "NEF":
        save_responses(pid, dataset, run_folder, model_type)
    else:
        best_params_full = {**best_params}
        best_params_full["seed"] = best_params["seed"]
        df = math_models.run(best_params_full)
        df.to_pickle(run_folder / f"{model_type}_{dataset}_{pid}_responses.pkl")

    return params_df, performance_df


if __name__ == "__main__":
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    pid = int(sys.argv[3])
    n_trials = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    if len(sys.argv) >= 8:
        k = int(sys.argv[5])
        run_folder = sys.argv[6]
        optuna_seed = int(sys.argv[7])
    elif len(sys.argv) == 7:
        k = int(sys.argv[5])
        run_folder = sys.argv[6]
        optuna_seed = 42
    else:
        k = 5
        run_folder = sys.argv[5] if len(sys.argv) > 5 else None
        optuna_seed = int(sys.argv[6]) if len(sys.argv) > 6 else 42

    logging.basicConfig(level=logging.INFO)
    params_df, performance_df = fit(
        dataset,
        model_type,
        pid,
        n_trials=n_trials,
        k=k,
        run_folder=run_folder,
        optuna_seed=optuna_seed,
    )
    elapsed = float(performance_df.loc[0, "runtime"])
    logging.info(f"Completed in {elapsed:.2f} min")
    logging.info(performance_df.to_string())
    logging.info(params_df.to_string())
    print("JOB_COMPLETE")
