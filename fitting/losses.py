"""
Loss computation for model fitting across experiments.

Supports multiple objectives:

- **Experiment 1:** ``mse`` — mean squared error between model and human
  responses; task-agnostic baseline for all three datasets.
- **Experiment 2:** ``nll`` — jiang only; negative log-likelihood with a sigmoid
  decision rule on model expectation (requires ``beta`` in ``params``). Task-specific
  losses: ``wasserstein`` (carrabin: Wasserstein distance between response
  distributions), ``mse_wasserstein`` (carrabin / yoo: MSE on responses plus
  Wasserstein on response distribution or smoothed mean ``|Δresponse|`` curves),
  ``switch`` (jiang: switch probability vs.
  conflict), ``decay`` (yoo: power-law decay of update magnitude).

This module does not depend on the model implementation layer.
"""

import numpy as np
import pandas as pd
import scipy.special
from scipy.stats import wasserstein_distance

DELTA_SMOOTH_WINDOW = 3  # rolling window for smoothing delta curves in mse_wasserstein
MSE_WASSERSTEIN_W = {
    "carrabin": 0.2,
    "yoo":      0.5,
}


def _smooth_curve(arr: np.ndarray, window: int) -> np.ndarray:
    """Apply centered rolling average of given window size to 1D array."""
    if window <= 1:
        return arr
    result = arr.astype(float).copy()
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        result[i] = float(arr[lo:hi].mean())
    return result


def mse(params: dict, model: pd.DataFrame, human: pd.DataFrame) -> float:
    """Mean squared error between model and human responses for one participant."""
    dataset = params["dataset"]
    sq_errors: list[float] = []

    if dataset in ("carrabin", "yoo"):
        pairs = (
            human[["trial", "observation"]]
            .drop_duplicates()
            .sort_values(["trial", "observation"])
        )
        for _, pair in pairs.iterrows():
            trial = int(pair["trial"])
            observation = int(pair["observation"])
            h = human.query("trial == @trial & observation == @observation")[
                "response"
            ]
            m = model.query("trial == @trial & observation == @observation")[
                "response"
            ]
            if h.empty or m.empty:
                raise ValueError(
                    f"Missing response for (trial={trial}, observation={observation})"
                )
            human_response = float(h.iloc[0])
            model_response = float(m.iloc[0])
            err = human_response - model_response
            sq_errors.append(err**2)

    elif dataset == "jiang":
        if "beta" not in params:
            raise ValueError("params must include 'beta' for jiang MSE computation")
        beta = float(params["beta"])
        pairs = (
            human[["trial", "stage"]].drop_duplicates().sort_values(["trial", "stage"])
        )
        for _, pair in pairs.iterrows():
            trial = int(pair["trial"])
            stage = int(pair["stage"])
            h = human.query("trial == @trial & stage == @stage")["response"]
            m = model.query("trial == @trial & stage == @stage")["response"]
            if h.empty or m.empty:
                raise ValueError(f"Missing response for (trial={trial}, stage={stage})")
            if h.nunique() != 1:
                raise ValueError(
                    f"Non-unique human response at (trial={trial}, stage={stage})"
                )
            human_response = float(h.iloc[0])
            model_response = float(m.iloc[0])
            p = float(scipy.special.expit(beta * model_response))
            model_binary = 1.0 if p > 0.5 else -1.0
            err = human_response - model_binary
            sq_errors.append(err**2)

    else:
        raise ValueError("params['dataset'] must be one of 'carrabin', 'jiang', 'yoo'")

    out = float(np.mean(sq_errors))
    if not np.isfinite(out):
        raise ValueError(f"MSE is not finite: {out}")
    return out


def nll(params: dict, model: pd.DataFrame, human: pd.DataFrame) -> float:
    """
    Negative log-likelihood for jiang binary choice data.

    Maps model expectation through sigmoid(beta * expectation) to get
    predicted probability of response==1, then computes NLL against
    the human binary response at each (trial, stage).

    Only valid for dataset=="jiang".
    """
    dataset = params["dataset"]
    if dataset != "jiang":
        raise ValueError(
            f"nll() is only implemented for jiang; got dataset={dataset!r}"
        )
    if "beta" not in params:
        raise ValueError("params must include 'beta' for nll computation")

    beta = float(params["beta"])
    total_logp = 0.0

    pairs = (
        human[["trial", "stage"]]
        .drop_duplicates()
        .sort_values(["trial", "stage"])
    )
    for _, pair in pairs.iterrows():
        trial = int(pair["trial"])
        stage = int(pair["stage"])
        h = human.query("trial == @trial & stage == @stage")["response"]
        m = model.query("trial == @trial & stage == @stage")["response"]
        if h.empty or m.empty:
            raise ValueError(
                f"Missing response for (trial={trial}, stage={stage})"
            )
        if h.nunique() != 1:
            raise ValueError(
                f"Non-unique human response at (trial={trial}, stage={stage})"
            )
        human_response = float(h.iloc[0])
        model_response = float(m.iloc[0])
        p = float(
            np.clip(
                scipy.special.expit(beta * model_response), 1e-10, 1 - 1e-10
            )
        )
        total_logp += np.log(p) if human_response == 1 else np.log(1.0 - p)

    total_nll = float(-total_logp)
    if not np.isfinite(total_nll):
        raise ValueError(f"NLL is not finite: {total_nll}")
    return total_nll


def wasserstein_loss(
    params: dict, model: pd.DataFrame, human: pd.DataFrame
) -> float:
    """
    Wasserstein distance between human and model response distributions
    across all trials for one participant (carrabin only).

    Measures how well the model captures the full shape of the
    participant's response distribution, including trial-to-trial
    variability. Lower is better; 0 means identical distributions.

    Used in Experiment 2 for carrabin.
    """
    dataset = params["dataset"]
    if dataset != "carrabin":
        raise ValueError(
            f"wasserstein_loss() is only implemented for carrabin; "
            f"got dataset={dataset!r}"
        )
    human_responses = human["response"].to_numpy(dtype=float)
    model_responses = model["response"].to_numpy(dtype=float)
    if len(human_responses) == 0 or len(model_responses) == 0:
        raise ValueError("Empty response arrays in wasserstein_loss")
    result = float(wasserstein_distance(human_responses, model_responses))
    if not np.isfinite(result):
        raise ValueError(f"wasserstein_loss is not finite: {result}")
    return result


def mse_wasserstein(
    params: dict,
    model: pd.DataFrame,
    human: pd.DataFrame,
) -> float:
    """
    MSE on responses + Wasserstein term.
    - carrabin: Wasserstein on full response distribution, w=0.2
    - yoo: Wasserstein on smoothed mean |delta response| curve, w=0.5
    """
    dataset = params["dataset"]
    if dataset not in ("carrabin", "yoo"):
        raise ValueError(
            f"mse_wasserstein() is only implemented for carrabin and yoo; "
            f"got dataset={dataset!r}"
        )

    w = float(params.get("wasserstein_w", MSE_WASSERSTEIN_W[dataset]))

    # MSE term (shared)
    merged = model.merge(
        human[["pid", "trial", "observation", "response"]],
        on=["pid", "trial", "observation"],
        suffixes=("_model", "_human"),
    )
    if merged.empty:
        return float("inf")
    mse_val = float(
        np.mean((merged["response_model"] - merged["response_human"]) ** 2)
    )

    if dataset == "carrabin":
        human_responses = human["response"].to_numpy(dtype=float)
        model_responses = model["response"].to_numpy(dtype=float)
        wass_val = float(wasserstein_distance(human_responses, model_responses))

    else:  # yoo
        def mean_delta(df: pd.DataFrame) -> np.ndarray:
            pieces = []
            for (pid, trial), grp in df.groupby(["pid", "trial"], sort=False):
                g = grp.sort_values("observation").copy()
                g["delta"] = g["response"].diff().abs()
                pieces.append(g)
            delta = pd.concat(pieces, ignore_index=True)
            curve = delta.groupby("observation")["delta"].mean()
            curve = curve[curve.index >= 2].sort_index().to_numpy(dtype=float)
            return _smooth_curve(curve, DELTA_SMOOTH_WINDOW)

        h_curve = mean_delta(human)
        m_curve = mean_delta(model)
        if len(h_curve) == 0 or len(m_curve) == 0:
            return mse_val
        n = min(len(h_curve), len(m_curve))
        wass_val = float(wasserstein_distance(h_curve[:n], m_curve[:n]))

    return (1.0 - w) * mse_val + w * wass_val


def switch_loss(params: dict, model: pd.DataFrame, human: pd.DataFrame) -> float:
    # TODO: loss on switch probability as function of conflict (RD)
    # Used in Experiment 2 for jiang. Requires beta parameter.
    raise NotImplementedError


def decay_loss(params: dict, model: pd.DataFrame, human: pd.DataFrame) -> float:
    # TODO: loss on power-law decay of response change magnitude
    # Used in Experiment 2 for yoo
    raise NotImplementedError


def compute_loss(
    loss_type: str, params: dict, model: pd.DataFrame, human: pd.DataFrame
) -> float:
    if loss_type == "mse":
        return mse(params, model, human)
    if loss_type == "nll":
        return nll(params, model, human)
    if loss_type == "wasserstein":
        return wasserstein_loss(params, model, human)
    if loss_type == "mse_wasserstein":
        return mse_wasserstein(params, model, human)
    if loss_type == "switch":
        return switch_loss(params, model, human)
    if loss_type == "decay":
        return decay_loss(params, model, human)
    raise ValueError(f"Unknown loss_type: {loss_type!r}")
