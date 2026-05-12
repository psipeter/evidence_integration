"""
Model-fitting orchestration layer for participant-level parameter estimation.

Coordinates Optuna search with k-fold cross-validation over a chosen loss
(``loss_type``), writing participant-specific outputs under a run folder
(see ``run_folder``) for downstream aggregation. Filenames are
``{model_type}_{dataset}_{pid}_params.pkl``, ``_performance.pkl``, and
``_folds.pkl`` (loss type is stored in the params row, not in the name).
When ``loss_type`` is omitted (``None``), the default is task-aware:
``response`` for all datasets; pass a string explicitly to override.

Loss codes (see ``fitting.losses.compute_loss``):

- **``response``** — response-accuracy loss for every dataset: mean squared error
  on carrabin/yoo; for jiang, total negative log-likelihood of human binary
  responses under ``sigmoid(beta * model_expectation)`` (requires ``beta``).
- **``shape``** — Wasserstein shape distance (carrabin full distribution; yoo
  smoothed mean ``|Δresponse|`` curve).
- **``joint``** — combined response + shape loss (carrabin, yoo, jiang).

Entry point:
``python -m fitting.fit {dataset} {model_type} {pid} [n_trials] [loss_type] [k] [run_folder]``

Optional 4th token ``n_trials`` (default 100); optional 5th ``loss_type`` (omit
for task-aware default). With eight or more tokens after the program name, the
6th is ``k`` (CV folds), the 7th is ``run_folder``, and an optional 8th is
``optuna_seed``. With exactly seven tokens, the 6th is ``run_folder`` and
``k`` defaults to 5. Omit trailing tokens for defaults (``run_folder`` defaults
to ``data/runs/default``).

**Carrabin:** ``Bayes`` / ``NoisyCounting`` — no fitted params. ``RL`` — ``alpha``.

**Jiang:** ``beta`` is always suggested (inverse temperature in the sigmoid used
inside ``response_loss`` / NLL). ``DeGroot`` — ``beta``; ``RL`` —
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
from utils.save_responses import save as save_responses

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_LOSS: dict[str, str] = {
    "carrabin": "response",
    "jiang": "response",
    "yoo": "response",
    "usher": "response",
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
    loss_type: str,
    beta_outside_optuna: bool = False,
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
        if beta_outside_optuna and param == "beta":
            continue  # beta will be fitted separately
        low, high, step = spec
        # Keep integer-valued hyperparameters discrete in Optuna.
        if float(step).is_integer() and float(low).is_integer() and float(high).is_integer():
            params[param] = trial.suggest_int(param, int(low), int(high), step=int(step))
        else:
            params[param] = trial.suggest_float(param, low, high, step=step)
    return params


def _run_nef_all_trials(params: dict, human: pd.DataFrame) -> pd.DataFrame:
    """Run NEF simulation once for all trials, return responses dataframe."""
    return NEF.run(params)


def _cross_validate(
    params: dict,
    model_responses: pd.DataFrame,
    human: pd.DataFrame,
    k: int = 5,
    loss_type: str = "response",
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
        fold_loss = losses.compute_loss(loss_type, params, model_fold, human_fold)
        fold_losses.append(float(fold_loss))

    if not fold_losses:
        raise ValueError("No non-empty CV folds were generated")
    return float(np.mean(fold_losses)), fold_losses


def _joint_cv_response_and_full_shape(
    params: dict,
    model_responses_full: pd.DataFrame,
    human: pd.DataFrame,
    k: int,
) -> tuple[float, list[float], float, float]:
    """
    Joint objective: shape_loss once on all trials; response_loss k-fold CV on
    held-out trials. Returns (total_loss, per_fold_response_losses,
    cv_response_loss, shape).
    """
    shape = float(losses.shape_loss(params, model_responses_full, human))
    trials_arr = np.asarray(sorted(human["trial"].unique()))
    rng = np.random.RandomState(seed=int(params["pid"]))
    shuffled = trials_arr.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, k)

    fold_losses: list[float] = []
    for fold_trials in folds:
        holdout_trials = [int(t) for t in fold_trials.tolist()]
        if not holdout_trials:
            continue
        model_val = model_responses_full[
            model_responses_full["trial"].isin(holdout_trials)
        ]
        human_val = human[human["trial"].isin(holdout_trials)]
        fold_losses.append(float(losses.response_loss(params, model_val, human_val)))

    if not fold_losses:
        raise ValueError("No non-empty CV folds were generated")
    cv_response_loss = float(np.mean(fold_losses))
    dataset = params["dataset"]
    w = float(params.get("wasserstein_w", losses.JOINT_LOSS_W[dataset]))
    total_loss = (1.0 - w) * cv_response_loss + w * shape
    return total_loss, fold_losses, cv_response_loss, shape


def _fit_beta(
    params: dict,
    model_responses: pd.DataFrame,
    human: pd.DataFrame,
    beta_bounds: tuple[float, float] = (0.01, 15.0),
) -> float:
    """Fit beta via 1D optimization on all trials to minimize response NLL."""
    from scipy.optimize import minimize_scalar

    def neg_nll(beta: float) -> float:
        p = {**params, "beta": float(beta)}
        try:
            return losses.response_loss(p, model_responses, human)
        except Exception:
            return float("inf")

    result = minimize_scalar(neg_nll, bounds=beta_bounds, method="bounded")
    return float(result.x)


def _enqueue_warm_start(
    study: optuna.Study,
    model_type: str,
    dataset: str,
    pid: int,
    run_folder: Path,
) -> bool:
    """
    If RL_lambda params exist for this pid/dataset, enqueue them as the first
    Optuna trial. Returns True if warm start was enqueued, False otherwise.
    Only applies to NEF models (carrabin, yoo, jiang, usher).
    """
    # TODO: [usher] Consider warm-start from RL_lambda_boost if RL_lambda pickle is absent
    if not model_type.startswith("NEF"):
        return False
    warm_path = run_folder / f"RL_lambda_{dataset}_{pid}_params.pkl"
    if not warm_path.exists():
        return False
    try:
        warm_params = pd.read_pickle(warm_path).iloc[0].to_dict()
        enqueue_params = {}
        for param in ("alpha_0", "lambda_"):
            if param in warm_params:
                enqueue_params[param] = float(warm_params[param])
        if not enqueue_params:
            return False
        study.enqueue_trial(enqueue_params)
        logging.info(f"Warm-starting NEF from RL_lambda params: {enqueue_params}")
        return True
    except Exception as e:
        logging.warning(f"Warm-start failed: {e}")
        return False


def fit(
    dataset: str,
    model_type: str,
    pid: int,
    n_trials: int = 100,
    k: int = 5,
    storage: str | None = None,
    loss_type: str | None = None,
    run_folder: Path | str | None = None,
    optuna_seed: int = 42,
    beta_outside_optuna: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one participant/model combination and persist outputs."""
    if run_folder is None:
        run_folder = RUNS_DIR / "default"
    run_folder = Path(run_folder)
    if loss_type is None:
        loss_type = DEFAULT_LOSS.get(dataset, "response")
    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
    human = human.query("pid == @pid")
    if human.empty:
        raise ValueError(f"No rows for pid={pid} in dataset={dataset!r}")

    if not MODEL_PARAMS[dataset][model_type]:
        n_trials = 1
        logging.info(
            f"{model_type} has no free parameters; running single evaluation."
        )

    if beta_outside_optuna:
        free_params = [
            p for p in MODEL_PARAMS[dataset][model_type]
            if p not in ("fixed", "beta")
        ]
        if len(free_params) == 0:
            if n_trials > 1:
                print(
                    f"Warning: {model_type} has no free params besides beta. "
                    f"Setting n_trials=1."
                )
            n_trials = 1

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model_type}_{dataset}_{pid}_{loss_type}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=optuna_seed),
    )
    _enqueue_warm_start(study, model_type, dataset, pid, run_folder)

    trial_records: list[dict] = []

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_params(
            trial, model_type, dataset, pid, loss_type, beta_outside_optuna
        )
        params["seed"] = abs(hash((int(params["pid"]), trial.number))) % (2**31)

        trial_wall_start = time.time()
        # run full simulation once
        if model_type in ("NEF_recurrent", "NEF_synaptic"):
            model_responses_full = _run_nef_all_trials(params, human)
        else:
            model_responses_full = math_models.run(params)

        if beta_outside_optuna and "beta" in MODEL_PARAMS[dataset][model_type]:
            params["beta"] = _fit_beta(params, model_responses_full, human)
            trial.set_user_attr("beta", params["beta"])

        # compute loss
        if loss_type == "shape":
            shape = float(losses.shape_loss(params, model_responses_full, human))
            trial.set_user_attr("shape_component", shape)
            # no CV — use full-data shape loss directly
            mean_loss = shape
            fold_losses = [shape] * k  # repeat for folds logging consistency
        elif loss_type == "joint":
            mean_loss, fold_losses, resp_c, shape_c = _joint_cv_response_and_full_shape(
                params, model_responses_full, human, k
            )
            trial.set_user_attr("response_component", resp_c)
            trial.set_user_attr("shape_component", shape_c)
        else:
            mean_loss, fold_losses = _cross_validate(
                params, model_responses_full, human, k=k, loss_type=loss_type
            )

        trial.set_user_attr(
            "runtime_minutes",
            (time.time() - trial_wall_start) / 60.0,
        )

        resp_c = trial.user_attrs.get("response_component", float("nan"))
        shape_c = trial.user_attrs.get("shape_component", float("nan"))
        for i, fold_loss in enumerate(fold_losses):
            record = {
                "model_type": model_type,
                "dataset": dataset,
                "pid": int(pid),
                "trial_number": trial.number,
                "fold": int(i + 1),
                "loss": float(fold_loss),
                "response_component": float(fold_loss),
                "shape_component": float(shape_c),
                "beta": float(trial.user_attrs.get("beta", float("nan"))),
            }
            # add all suggested params (excludes fixed params, model_type, dataset, pid, seed)
            for param_name, param_val in params.items():
                if param_name not in (
                    "model_type",
                    "dataset",
                    "pid",
                    "seed",
                    "base_seed",
                    "alpha_bias_array",
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
            "loss_type": loss_type,
        }
    )
    best_params["seed"] = abs(hash((int(pid), best_trial.number))) % (2**31)
    best_params["base_seed"] = best_params["seed"]
    if beta_outside_optuna:
        best_params["beta"] = float(
            best_trial.user_attrs.get("beta", float("nan"))
        )

    # TODO: runtime now represents total wall time per Optuna trial across all folds
    # (one model simulation + loss/CV through the last fold); performance.pkl stores
    # this for the best trial only, not elapsed time for the entire study.
    runtime = float(best_trial.user_attrs.get("runtime_minutes", float("nan")))
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
                "runtime": runtime,
            }
        ]
    )
    folds_df = pd.DataFrame(trial_records)
    folds_df.to_pickle(run_folder / f"{model_type}_{dataset}_{pid}_folds.pkl")

    params_df.to_pickle(run_folder / f"{model_type}_{dataset}_{pid}_params.pkl")
    performance_df.to_pickle(
        run_folder / f"{model_type}_{dataset}_{pid}_performance.pkl"
    )

    if model_type in ("NEF_recurrent", "NEF_synaptic"):
        save_responses(pid, dataset, run_folder, model_type)
    else:
        # math models: run full simulation with best params and save responses
        best_params_full = {**best_params}
        best_params_full["seed"] = best_params["seed"]
        df = math_models.run(best_params_full)
        df.to_pickle(run_folder / f"{model_type}_{dataset}_{pid}_responses.pkl")

    return params_df, performance_df


if __name__ == "__main__":
    beta_outside_optuna = "--beta_outside_optuna" in sys.argv
    argv = [arg for arg in sys.argv if arg != "--beta_outside_optuna"]
    dataset = argv[1]
    model_type = argv[2]
    pid = int(argv[3])
    n_trials = int(argv[4]) if len(argv) > 4 else 100
    loss_type = argv[5] if len(argv) > 5 else None
    if len(argv) >= 9:
        k = int(argv[6])
        run_folder = argv[7]
        optuna_seed = int(argv[8])
    elif len(argv) >= 8:
        k = int(argv[6])
        run_folder = argv[7]
        optuna_seed = 42
    else:
        k = 5
        run_folder = argv[6] if len(argv) > 6 else None
        optuna_seed = int(argv[7]) if len(argv) > 7 else 42

    logging.basicConfig(level=logging.INFO)
    params_df, performance_df = fit(
        dataset,
        model_type,
        pid,
        n_trials=n_trials,
        k=k,
        loss_type=loss_type,
        run_folder=run_folder,
        optuna_seed=optuna_seed,
        beta_outside_optuna=beta_outside_optuna,
    )
    elapsed = float(performance_df.loc[0, "runtime"])
    logging.info(f"Completed in {elapsed:.2f} min")
    logging.info(performance_df.to_string())
    logging.info(params_df.to_string())
    print("JOB_COMPLETE")
