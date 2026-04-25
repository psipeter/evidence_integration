"""
Model-fitting orchestration layer for participant-level parameter estimation.

Coordinates Optuna search with k-fold cross-validation over a chosen loss
(``loss_type``), writing participant-specific outputs under a run folder
(see ``run_folder``) for downstream aggregation. Filenames are
``{model_type}_{dataset}_{pid}_params.pkl``, ``_performance.pkl``, and
``_cv_folds.pkl`` (loss type is stored in the params row, not in the name).
When ``loss_type`` is omitted (``None``), the default is task-aware:
``response`` for all datasets; pass a string explicitly to override.

Loss codes (see ``fitting.losses.compute_loss``):

- **``response``** — response-accuracy loss for every dataset: mean squared error
  on carrabin/yoo; for jiang, total negative log-likelihood of human binary
  responses under ``sigmoid(beta * model_expectation)`` (requires ``beta``).
- **``shape``** — Wasserstein shape distance (carrabin full distribution; yoo
  smoothed mean ``|Δresponse|`` curve).
- **``joint``** — combined response + shape loss (carrabin, yoo, jiang).

Optional ``n_runs`` (default ``1``) is forwarded into model ``params`` when
``> 1`` to control Monte Carlo averaging for stochastic models (e.g. carrabin
``NoisyCounting``, future NEF models). Use ``1`` for fast local runs;
``n_runs >= 20`` is a reasonable choice for cluster fits.

Entry point:
``python -m fitting.fit {dataset} {model_type} {pid} [n_trials] [loss_type] [n_runs] [k] [run_folder]``

Optional 4th token ``n_trials`` (default 100); optional 5th ``loss_type`` (omit
for task-aware default); optional 6th ``n_runs`` (default 1). With exactly eight
tokens after the program name, the 8th is ``run_folder`` and ``k`` defaults to
5. With nine or more, the 8th is ``k`` (CV folds) and the 9th is ``run_folder``.
Omit trailing tokens for defaults (``run_folder`` defaults to
``data/runs/default``).

**Carrabin:** ``Bayes`` / ``NoisyCounting`` — no fitted params. ``RL`` — ``alpha``.

**Jiang:** ``beta`` is always suggested (inverse temperature in the sigmoid used
inside ``response_loss`` / NLL). ``DeGroot`` — ``omega`` (0.01–10.0); ``RL`` —
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
from models import NEF
from fitting.model_params import MODEL_PARAMS
from utils.paths import RUNS_DIR, data_path

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_LOSS: dict[str, str] = {
    "carrabin": "response",
    "jiang": "response",
    "yoo": "response",
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
    # if trial.number % 10 == 0:
    trial_loss = f"{trial.value:.4f}" if trial.value is not None else "failed"
    best = f"{study.best_value:.4f}" if study.best_value is not None else "n/a"
    logging.info(
        f"Trial {trial.number}: loss={trial_loss} | "
        f"best={best} | params={trial.params}"
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
    if n_runs > 1:
        params["n_runs"] = n_runs
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
    human: pd.DataFrame,
    k: int = 5,
    loss_type: str = "response",
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

        if params["model_type"] in ("NEF_recurrent", "NEF_synaptic"):
            model_fold = NEF.run(params, trials=holdout_trials)
        else:
            model_fold = math_models.run(params, trials=holdout_trials)
        human_fold = human[human["trial"].isin(holdout_trials)]

        fold_loss = losses.compute_loss(loss_type, params, model_fold, human_fold)
        fold_losses.append(float(fold_loss))

    if not fold_losses:
        raise ValueError("No non-empty CV folds were generated")

    mean_loss = float(np.mean(fold_losses))
    return mean_loss, fold_losses


def _cross_validate_components(
    params: dict,
    human: pd.DataFrame,
    k: int = 5,
) -> tuple[float, float]:
    """Return (mean response_loss, mean shape_loss) for joint loss decomposition."""
    trials = np.asarray(sorted(human["trial"].unique()))
    rng = np.random.RandomState(seed=int(params["pid"]))
    shuffled = trials.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, k)

    response_losses: list[float] = []
    shape_losses: list[float] = []
    for fold_trials in folds:
        holdout_trials = [int(t) for t in fold_trials.tolist()]
        if not holdout_trials:
            continue
        if params["model_type"] in ("NEF_recurrent", "NEF_synaptic"):
            model_fold = NEF.run(params, trials=holdout_trials)
        else:
            model_fold = math_models.run(params, trials=holdout_trials)
        human_fold = human[human["trial"].isin(holdout_trials)]
        response_losses.append(
            float(losses.response_loss(params, model_fold, human_fold))
        )
        shape_losses.append(float(losses.shape_loss(params, model_fold, human_fold)))

    if not response_losses:
        raise ValueError("No non-empty CV folds were generated")
    return float(np.mean(response_losses)), float(np.mean(shape_losses))


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
    optuna_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one participant/model combination and persist outputs."""
    if run_folder is None:
        run_folder = RUNS_DIR / "default"
    run_folder = Path(run_folder)
    if loss_type is None:
        loss_type = DEFAULT_LOSS.get(dataset, "response")
    if model_type in ("NEF_recurrent", "NEF_synaptic") and n_runs > 1:
        raise ValueError("NEF_recurrent and NEF_synaptic do not support n_runs > 1")
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
        sampler=optuna.samplers.TPESampler(seed=optuna_seed),
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_params(
            trial, model_type, dataset, pid, loss_type, n_runs
        )
        params["seed"] = abs(hash((int(params["pid"]), trial.number))) % (2**31)
        mean_loss, fold_losses = _cross_validate(params, human, k=k, loss_type=loss_type)
        trial.set_user_attr("cv_loss_folds", fold_losses)
        if loss_type == "joint":
            response_mean, shape_mean = _cross_validate_components(params, human, k=k)
            trial.set_user_attr("response_component", response_mean)
            trial.set_user_attr("shape_component", shape_mean)
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
    best_params["seed"] = abs(hash((int(pid), best_trial.number))) % (2**31)
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
                "response_component": float(
                    best_trial.user_attrs.get("response_component", float("nan"))
                ),
                "shape_component": float(
                    best_trial.user_attrs.get("shape_component", float("nan"))
                ),
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
    if len(sys.argv) >= 9:
        k = int(sys.argv[7])
        run_folder = sys.argv[8]
    else:
        k = 5
        run_folder = sys.argv[7] if len(sys.argv) > 7 else None
    optuna_seed = int(sys.argv[9]) if len(sys.argv) > 9 else 42

    logging.basicConfig(level=logging.INFO)
    params_df, performance_df = fit(
        dataset,
        model_type,
        pid,
        n_trials=n_trials,
        k=k,
        loss_type=loss_type,
        n_runs=n_runs,
        run_folder=run_folder,
        optuna_seed=optuna_seed,
    )
    elapsed = float(performance_df.loc[0, "runtime"])
    logging.info(f"Completed in {elapsed:.2f} min")
    logging.info(performance_df.to_string())
    logging.info(params_df.to_string())
    print("JOB_COMPLETE")
