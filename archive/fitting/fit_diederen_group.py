"""
Group-level model fitting for the Diederen dataset.

## Why this script exists

The standard fitting pipeline (fitting/fit.py) fits each model separately
to each participant's full trial data using k-fold cross-validation. For
the Diederen dataset, per-participant parameter estimates are poorly
constrained for the key analysis of interest: the power-law decay of
learning rate across the first few observations of a new distribution.

The reason: the learning rate decay is most visible in the first 4-5
observations of each distribution (before repeated context switches
introduce carryover noise). With only 4-5 clean data points per
participant, per-participant power-law fits are unreliable.

## What this script does differently

1. DATA SELECTION: Only uses the cleanest subset of the data:
   - Distribution A only (the first distribution seen in each session),
     before the first context switch to distribution B. This gives
     approximately 4-6 observations per (pid, session) with no
     contamination from the other distribution's carryover effects.
   - Control and placebo groups only (CTRL, PCB).
   - Excludes poor performers (EXCLUDE_PIDS): participants whose RMSE
     vs the true EV was >= their RMSE vs 0.
   This filtered data is saved as data/diederen_group.pkl and used as
   the dataset for model simulation and loss computation.

2. GROUP-LEVEL FITTING: Fits a single set of parameters to the
   group-level mean response trajectory. The objective is:
     RMSE(group_mean_model[obs], group_mean_human[obs])
   averaged across observation numbers (unweighted).

   With ~120 (pid, session) sequences (40 pids × 3 sessions) each
   contributing 4-6 observations, the group mean is well-constrained.

3. SIMULATION: Models are run using the existing math_models.run()
   infrastructure on the filtered diederen_group.pkl dataset — no
   reimplementation needed. The same approach will work for NEF2d
   later by swapping the dataset pkl.

4. OUTPUT: A single master file per model:
     data/runs/diederen_group/{model_type}_diederen_group.pkl
   containing a dict with keys:
     - "model_type": str
     - "params": dict of fitted parameter values
     - "loss": float, group-level RMSE at best params
     - "responses": pd.DataFrame with columns
         [pid, session, trial, observation, response]
       Per-sequence responses for all sequences used in fitting.
       The figure script recomputes group means and variance metrics
       from this DataFrame as needed.

## Relationship to existing pipeline

- Uses the same model implementations (models/math_models.py) unchanged
- Uses MODEL_PARAMS from fitting/model_params.py for parameter bounds
- Does NOT use fitting/fit.py, fitting/losses.py, or fitting/collect.py
- Does NOT require the cluster — runs locally in minutes for math models
- NEF2d excluded for now; will work by pointing to diederen_group.pkl

## Usage

    python scripts/fit_diederen_group.py --model RL --n_trials 500
    python scripts/fit_diederen_group.py --model RL_lambda --n_trials 500
    python scripts/fit_diederen_group.py --model PearceHall --n_trials 500
    python scripts/fit_diederen_group.py --model Mean --n_trials 1

Output written to data/runs/diederen_group/.
The filtered dataset is saved to data/diederen_group.pkl on first run.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models.math_models as math_models
from fitting.model_params import MODEL_PARAMS
from utils.paths import RUNS_DIR, data_path

EXCLUDE_PIDS: list[int] = [
    1011,
    1023,
    1027,
    1028,
    1032,
    2001,
    2029,
    2036,
    2038,
    2047,
    2048,
    2064,
    2083,
    2092,
    2099,
]

MATH_MODELS = ["Mean", "RL", "RL_lambda", "PearceHall"]
GROUP_DATASET = "diederen_group"
MIN_SEQUENCES = 10  # min sequences required to include an observation in the loss


def _build_group_dataset() -> pd.DataFrame:
    """
    Build the group-level dataset and save as data/diederen_group.pkl.

    Filters the full diederen dataset to:
    - CTRL and PCB groups only
    - EXCLUDE_PIDS removed
    - Non-missed, non-catch-trial rows only
    - Distribution A only (first distribution seen per session),
      before the first context switch to distribution B

    The resulting pkl has the same column structure as diederen.pkl
    so it can be used as a drop-in replacement for model simulation.

    Saves to data/diederen_group.pkl and returns the DataFrame.
    """
    df = pd.read_pickle(data_path("diederen.pkl"))
    df = df[
        ~df["pid"].isin(EXCLUDE_PIDS)
        & df["group"].isin(["CTRL", "PCB"])
        & ~df["missed"]
        & df["response"].notna()
    ].copy()

    out = []
    for (pid, session), grp in df.groupby(["pid", "session"], sort=False):
        g = grp.sort_values("trial_in_session").reset_index(drop=True)
        distrib_A = int(g.at[0, "distrib_index"])
        keep = []
        for i in range(len(g)):
            if int(g.at[i, "distrib_index"]) != distrib_A:
                break
            keep.append(i)
        if keep:
            out.append(g.iloc[keep])

    filtered = pd.concat(out, ignore_index=True)
    out_path = data_path(GROUP_DATASET + ".pkl")
    filtered.to_pickle(out_path)
    print(
        f"Saved {GROUP_DATASET}.pkl: {len(filtered)} rows, "
        f"{filtered['pid'].nunique()} pids, "
        f"{filtered.groupby(['pid', 'session']).ngroups} sequences"
    )
    return filtered


def _run_group_model(params: dict, pids: list[int]) -> pd.DataFrame:
    """Simulate model on all pids in diederen_group.pkl."""
    parts = []
    for pid in pids:
        run_params = {**params, "dataset": GROUP_DATASET, "pid": int(pid)}
        parts.append(math_models.run(run_params))
    return pd.concat(parts, ignore_index=True)


def _group_loss(
    params: dict,
    human_means: dict[str, pd.Series],
    ev_map: pd.DataFrame,
    pids: list[int],
) -> float:
    """
    Group-level loss split by EV sign to avoid cancellation.

    Computes RMSE separately for positive-EV and negative-EV sequences,
    then returns the average. This prevents +EV and -EV responses from
    cancelling each other in the group mean.

    human_means: dict with keys "pos" and "neg", each a pd.Series
                 indexed by observation with mean response values.
    """
    model_resp = _run_group_model(params, pids)
    model_resp = model_resp.merge(ev_map, on=["pid", "trial"], how="left")

    losses = []
    for sign, key in [(1, "pos"), (-1, "neg")]:
        human_mean = human_means[key]
        model_sub = model_resp[model_resp["ev"] * sign > 0]
        if model_sub.empty:
            continue
        model_mean = model_sub.groupby("observation")["response"].mean()
        common = human_mean.index.intersection(model_mean.index)
        if len(common) == 0:
            continue
        errors = (model_mean[common] - human_mean[common]) ** 2
        losses.append(float(np.sqrt(errors.mean())))

    return float(np.mean(losses)) if losses else float("inf")


def _log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    trial_loss = f"{trial.value:.4f}" if trial.value is not None else "failed"
    try:
        best = f"{study.best_value:.4f}"
    except ValueError:
        best = "n/a"
    print(
        f"  Trial {trial.number:3d}: loss={trial_loss}  best={best}"
        f"  params={trial.params}"
    )


def fit_model(
    model_type: str,
    n_trials: int,
    run_folder: Path,
    seed: int = 42,
) -> None:
    """Fit one model at group level and save master output file."""
    group_pkl = data_path(GROUP_DATASET + ".pkl")
    if not group_pkl.exists():
        human = _build_group_dataset()
    else:
        human = pd.read_pickle(group_pkl)
        print(f"Loaded {GROUP_DATASET}.pkl: {len(human)} rows")

    pids = sorted(human["pid"].unique().tolist())

    obs_counts = human.groupby("observation")["pid"].count()
    valid_obs = obs_counts[obs_counts >= MIN_SEQUENCES].index

    human_means = {
        "pos": human[human["ev"] > 0].groupby("observation")["response"].mean(),
        "neg": human[human["ev"] < 0].groupby("observation")["response"].mean(),
    }
    human_means["pos"] = human_means["pos"][
        human_means["pos"].index.isin(valid_obs)
    ]
    human_means["neg"] = human_means["neg"][
        human_means["neg"].index.isin(valid_obs)
    ]
    ev_map = human[["pid", "trial", "ev"]].drop_duplicates(subset=["pid", "trial"])

    fixed = MODEL_PARAMS["diederen"][model_type].get("fixed", {})
    base_params = {
        "model_type": model_type,
        "dataset": GROUP_DATASET,
        **fixed,
    }
    model_spec = MODEL_PARAMS["diederen"][model_type]

    def objective(trial: optuna.trial.Trial) -> float:
        params = {**base_params}
        for param, spec in model_spec.items():
            if param == "fixed":
                continue
            low, high, step = spec
            if (
                float(step).is_integer()
                and float(low).is_integer()
                and float(high).is_integer()
            ):
                params[param] = trial.suggest_int(
                    param, int(low), int(high), step=int(step)
                )
            else:
                params[param] = trial.suggest_float(param, low, high, step=step)
        params["seed"] = seed
        return _group_loss(params, human_means, ev_map, pids)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[_log_callback])

    best = study.best_trial
    best_params = {**base_params, **best.params, "seed": seed}

    model_resp = _run_group_model(best_params, pids)
    final_loss = _group_loss(best_params, human_means, ev_map, pids)

    print(f"\n{model_type} group fit complete:")
    print(f"  params: {best.params}")
    print(f"  loss:   {final_loss:.4f}")
    print("  obs-by-obs (human_pos | human_neg | model_pos | model_neg):")
    model_resp_best = model_resp.merge(ev_map, on=["pid", "trial"], how="left")
    model_mean_pos = model_resp_best[model_resp_best["ev"] > 0].groupby(
        "observation"
    )["response"].mean()
    model_mean_neg = model_resp_best[model_resp_best["ev"] < 0].groupby(
        "observation"
    )["response"].mean()
    all_obs = sorted(set(human_means["pos"].index) | set(human_means["neg"].index))
    for obs in all_obs:
        hp = human_means["pos"].get(obs, float("nan"))
        hn = human_means["neg"].get(obs, float("nan"))
        mp = model_mean_pos.get(obs, float("nan"))
        mn = model_mean_neg.get(obs, float("nan"))
        print(f"    obs={obs}: h+={hp:.3f} h-={hn:.3f} | m+={mp:.3f} m-={mn:.3f}")

    meta = human[
        ["pid", "session", "trial", "observation", "ev"]
    ].drop_duplicates(subset=["pid", "trial", "observation"])
    resp_with_meta = model_resp.merge(
        meta, on=["pid", "trial", "observation"], how="left"
    )

    result = {
        "model_type": model_type,
        "params": best.params,
        "loss": final_loss,
        "human_means": human_means,
        "responses": resp_with_meta,
    }
    out_path = run_folder / f"{model_type}_diederen_group.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group-level model fitting for Diederen dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MATH_MODELS,
        help="Model to fit.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=500,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--run_folder",
        type=str,
        default="diederen_group",
        help="Output subfolder under data/runs/.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild_dataset",
        action="store_true",
        default=False,
        help="Rebuild diederen_group.pkl even if it exists.",
    )
    args = parser.parse_args()

    run_folder = RUNS_DIR / args.run_folder
    run_folder.mkdir(parents=True, exist_ok=True)

    if args.rebuild_dataset or not data_path(GROUP_DATASET + ".pkl").exists():
        _build_group_dataset()

    fit_model(args.model, args.n_trials, run_folder, seed=args.seed)


if __name__ == "__main__":
    main()
