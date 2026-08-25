"""
Participant-level model fitting via Optuna (TPE) and k-fold CV.

Objective: RMSE from ``fitting.losses.compute_loss`` by default, or Gaussian
NLL from ``fitting.losses.compute_nll``/``nll_from_ensemble`` via ``--loss nll``.
NLL applies to STOCHASTIC models only (currently NoisyRL_lambda) -- a
deterministic model's ensemble is a delta function and its NLL is undefined; see
``models.math_models.simulate_ensemble``'s docstring. RMSE cannot identify a
noise parameter at all (it collapses to zero, since squared error only sees the
mean); NLL was verified on soltani_numbers pid 1 to have a genuine INTERIOR
optimum (sigma_resp ~0.04-0.05, NLL falling from 389 at sigma_resp=0.001 to
-2.46 at the optimum and rising again beyond it) -- see docs/HISTORY.md.

Entry point::

    python -m fitting.fit {dataset} {model_type} {pid} [--n_trials N] [--k K]
        [--run_folder F] [--optuna_seed S] [--datafile SUFFIX]

Writes ``{model_type}_{stem}_{pid}_params.pkl``, ``_performance.pkl``, and
``_folds.pkl`` under ``run_folder``, where ``stem`` is
``utils.paths.dataset_stem(dataset, datafile)`` -- i.e. the dataset family name
plus the optional data-version suffix, so fits against different builds of the
same dataset can coexist in one run folder without overwriting or being
silently mistaken for one another.

SLURM jobs are submitted via ``fitting.submit``.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

import fitting.losses as losses
import models.math_models as math_models
from models.math_models import _STOCHASTIC_ENSEMBLE_MODELS
from models import NEF
from fitting.model_params import MODEL_PARAMS
from utils.paths import RUNS_DIR, data_path, dataset_stem, resolve_run_folder
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
    datafile: str | None = None,
) -> dict:
    """Sample model parameters for one Optuna trial."""
    params = {
        "model_type": model_type,
        "dataset": dataset,
        "pid": int(pid),
        "datafile": datafile,
    }
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
    """Partition trials into k disjoint folds and return (mean fold loss, folds).

    NOT held-out validation, despite the name -- worth being precise about, since
    the outputs get labelled "cross-validated" downstream. Model responses are
    computed ONCE from one parameter set and then partitioned, so every fold
    contributes to the objective Optuna minimises; no fold is excluded from
    parameter selection. The returned mean is therefore an IN-SAMPLE loss,
    computed in a partitioned way. Verified: for RL_lambda on soltani_numbers
    pid 1, mean-of-folds 0.06295 vs the all-trials loss 0.06308 -- a difference of
    1.2e-4, which is just Jensen's inequality (mean of sqrt vs sqrt of mean).

    What the folds DO give is a real spread across trial subsets (the per-fold
    losses are disjoint by trial and land in _folds.pkl), which is useful as a
    stability check even though it is not validation.

    ACCEPTED FOR NOW, not fixed. Expected optimism is small -- these models have
    0-2 free parameters against ~480 observations per participant -- but that has
    NOT been measured. Real nested CV would fit on k-1 folds and evaluate on the
    held-out fold, at ~k times the fitting cost (trivial for math models,
    significant for NEF). If model COMPARISON across differing parameter counts
    ever matters (e.g. parameter-free Mean against the 2-parameter models), this
    should be fixed first, because in-sample loss favours the richer model.

    Applies to carrabin and yoo identically -- this is shared code, not a soltani
    quirk.
    """
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


def _cross_validate_nll(
    params: dict,
    ens: np.ndarray,
    row_index: pd.DataFrame,
    human: pd.DataFrame,
    k: int = 5,
    sigma_floor: float = losses.NLL_SIGMA_FLOOR,
) -> tuple[float, list[float]]:
    """NLL analogue of _cross_validate: partitions the SAME ensemble (simulated
    once per Optuna trial, not once per fold -- re-simulating per fold would cost
    k times as much for no benefit, since the ensemble already covers every row)
    into folds by trial, using the identical seeding as _cross_validate so a given
    pid gets the SAME trial/fold partition regardless of --loss. Not held-out
    validation, for the same reason _cross_validate is not -- every fold
    contributes to the mean Optuna minimises.
    """
    trials = np.asarray(sorted(human["trial"].unique()))
    rng = np.random.RandomState(seed=int(params["pid"]))
    shuffled = trials.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, k)

    human_sorted = human.sort_values(["trial", "observation"])
    fold_losses: list[float] = []
    for fold_trials in folds:
        holdout_trials = [int(t) for t in fold_trials.tolist()]
        if not holdout_trials:
            continue
        mask = row_index["trial"].isin(holdout_trials).to_numpy()
        y = human_sorted.loc[human_sorted["trial"].isin(holdout_trials),
                             "response"].to_numpy(float)
        fold_losses.append(losses.nll_from_ensemble(ens[:, mask], y, sigma_floor))

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
    datafile: str | None = None,
    loss_fn: str = "rmse",
    n_sims: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one participant/model combination and persist outputs.

    loss_fn='nll' requires model_type in models.math_models._STOCHASTIC_ENSEMBLE_MODELS
    (currently {"NoisyRL_lambda"}); checked up front so a bad combination fails
    before an Optuna study is created, not on the first trial.
    """
    if loss_fn not in ("rmse", "nll"):
        raise ValueError(f"loss_fn must be 'rmse' or 'nll', got {loss_fn!r}")
    if loss_fn == "nll" and model_type not in _STOCHASTIC_ENSEMBLE_MODELS:
        raise ValueError(
            f"--loss nll needs a stochastic model; {model_type!r} is "
            f"deterministic, so its ensemble is a delta function and NLL is "
            f"undefined. Use --loss rmse for this model, or fit one of "
            f"{sorted(_STOCHASTIC_ENSEMBLE_MODELS)}.")
    if run_folder is None:
        run_folder = RUNS_DIR / "default"
    run_folder = resolve_run_folder(run_folder)
    stem = dataset_stem(dataset, datafile)
    human = pd.read_pickle(data_path(f"{stem}.pkl"))
    human = human.query("pid == @pid")
    if human.empty:
        raise ValueError(f"No rows for pid={pid} in data/{stem}.pkl")

    if not MODEL_PARAMS[dataset][model_type]:
        n_trials = 1
        logging.info(
            f"{model_type} has no free parameters; running single evaluation."
        )

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model_type}_{stem}_{pid}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=optuna_seed),
    )

    trial_records: list[dict] = []

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_params(trial, model_type, dataset, pid, datafile)
        trial_wall_start = time.time()
        if loss_fn == "nll":
            # Simulated ONCE per Optuna trial; _cross_validate_nll partitions the
            # resulting ensemble by trial rather than re-simulating per fold.
            ens, row_index = math_models.simulate_ensemble(
                params, n_sims, return_index=True)
            mean_loss, fold_losses = _cross_validate_nll(
                params, ens, row_index, human, k=k)
        else:
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
                "loss_fn": loss_fn,
            }
            for param_name, param_val in params.items():
                if param_name not in (
                    "model_type",
                    "dataset",
                    "pid",
                    # a str data-version tag, not a fitted parameter -- would
                    # otherwise become a spurious object column in _folds.pkl
                    "datafile",
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
            "datafile": datafile,
        }
    )

    params_df = pd.DataFrame([best_params])
    performance_df = pd.DataFrame(
        [
            {
                "model_type": model_type,
                "dataset": dataset,
                "pid": int(pid),
                "loss": float(best_trial.value),
                # RMSE and NLL are NOT on the same scale (NLL can be negative) --
                # any reader of {model}_{stem}_performance.pkl must check this
                # before comparing `loss` across a mix of the two.
                "loss_fn": loss_fn,
                "runtime": float(
                    best_trial.user_attrs.get("runtime_minutes", float("nan"))
                ),
            }
        ]
    )
    folds_df = pd.DataFrame(trial_records)
    folds_df.to_pickle(run_folder / f"{model_type}_{stem}_{pid}_folds.pkl")

    params_df.to_pickle(run_folder / f"{model_type}_{stem}_{pid}_params.pkl")
    performance_df.to_pickle(
        run_folder / f"{model_type}_{stem}_{pid}_performance.pkl"
    )

    if model_type == "NEF":
        save_responses(pid, dataset, run_folder, model_type, datafile)
    else:
        best_params_full = {**best_params}
        df = math_models.run(best_params_full)
        df.to_pickle(run_folder / f"{model_type}_{stem}_{pid}_responses.pkl")

    return params_df, performance_df


if __name__ == "__main__":
    # argparse rather than positional parsing: this used to read up to 7
    # positional args via a three-branch len(sys.argv) check, which had no room
    # for another optional arg without becoming genuinely ambiguous. Commands
    # are generated by fitting.submit, and jobs/*.sh is gitignored/regenerated,
    # so nothing tracked depends on the old form.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("model_type")
    parser.add_argument("pid", type=int)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--run_folder", default=None)
    parser.add_argument("--optuna_seed", type=int, default=42)
    parser.add_argument(
        "--datafile",
        default=None,
        help="Data-version suffix selecting data/{dataset}_{datafile}.pkl and "
             "appearing in every output filename. Omit for the canonical "
             "data/{dataset}.pkl.",
    )
    parser.add_argument(
        "--loss", dest="loss_fn", choices=("rmse", "nll"), default="rmse",
        help="'rmse' (default): fitting.losses.compute_loss, works for any model "
             "but cannot identify a noise parameter (collapses to 0). 'nll': "
             "Gaussian NLL of observed responses under the model's simulated "
             "predictive distribution -- a proper scoring rule that penalises "
             "both a wrong mean AND a wrong variance, so it CAN find a genuine "
             "noise level. Only for models in "
             "models.math_models._STOCHASTIC_ENSEMBLE_MODELS (a deterministic "
             "model's ensemble is a delta function, so NLL is undefined).",
    )
    parser.add_argument(
        "--n_sims", type=int, default=100,
        help="Ensemble size per NLL evaluation (ignored for --loss rmse). "
             "Verified stable at n_sims=100 -- 5 reseeded reps all picked the "
             "same argmin on a sigma_resp sweep -- with a smaller ensemble "
             "(n_sims=25) already agreeing. Cost is roughly linear: n_sims=100 "
             "is ~0.45s/eval, i.e. ~2.3 min for a 300-trial fit.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    params_df, performance_df = fit(
        args.dataset,
        args.model_type,
        args.pid,
        n_trials=args.n_trials,
        k=args.k,
        run_folder=args.run_folder,
        optuna_seed=args.optuna_seed,
        datafile=args.datafile,
        loss_fn=args.loss_fn,
        n_sims=args.n_sims,
    )
    elapsed = float(performance_df.loc[0, "runtime"])
    logging.info(f"Completed in {elapsed:.2f} min")
    logging.info(performance_df.to_string())
    logging.info(params_df.to_string())
    print("JOB_COMPLETE")
