# NOTE: shape_loss, joint_loss archived in archive/fitting/archive_losses.py
"""
Loss computation for model fitting across experiments.

Response-accuracy loss: root mean squared error (RMSE) on carrabin, yoo, and diederen.

This module does not depend on the model implementation layer.
"""

import numpy as np
import pandas as pd

POWER_LAW_SMOOTH_WINDOW = 5  # smoothing window for power law fitting in yoo figures / diagnostics
QID_MIN_TRIALS = 10  # minimum trials per qid to include in carrabin qid-std diagnostic (figures)


def _mean_qid_std(df: pd.DataFrame, qid_min_trials: int = QID_MIN_TRIALS) -> float:
    """
    Compute mean per-qid response std for carrabin, using only qids with
    at least qid_min_trials trials. Returns nan if no valid qids.
    """
    counts = df.groupby("qid")["trial"].nunique()
    valid_qids = counts[counts >= qid_min_trials].index
    if len(valid_qids) == 0:
        return float("nan")
    stds = df[df["qid"].isin(valid_qids)].groupby("qid")["response"].std()
    return float(stds.mean())


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


def _fit_power_law_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a power law A * n^(-lambda) to each pid's smoothed mean |delta response|
    curve. Returns DataFrame with columns: pid, A, lambda_.
    """
    from scipy.stats import linregress

    rows = []
    for pid, grp in df.groupby("pid"):
        pieces = []
        for _, tgrp in grp.groupby("trial"):
            g = tgrp.sort_values("observation").copy()
            g["delta"] = g["response"].diff().abs()
            pieces.append(g)
        delta = pd.concat(pieces, ignore_index=True)
        curve = delta.groupby("observation")["delta"].mean().dropna()
        curve = curve[curve.index >= 2]
        if len(curve) < 3:
            continue
        d = _smooth_curve(curve.values, POWER_LAW_SMOOTH_WINDOW)
        if np.any(d <= 0):
            continue
        n = curve.index.values.astype(float)
        slope, intercept, _, _, _ = linregress(np.log(n), np.log(d))
        rows.append({"pid": pid, "A": float(np.exp(intercept)), "lambda_": float(-slope)})
    return pd.DataFrame(rows)


def response_loss(
    params: dict,
    model: pd.DataFrame,
    human: pd.DataFrame,
) -> float:
    """
    Response-accuracy loss for carrabin, yoo, and diederen: root mean squared error
    (RMSE) between model and human responses.
    """
    dataset = params["dataset"]
    if dataset not in ("carrabin", "yoo", "diederen"):
        raise ValueError(
            "params['dataset'] must be one of 'carrabin', 'yoo', 'diederen'"
        )

    if dataset == "diederen":
        human = human[~human["catch_trial"].astype(bool)]

    sq_errors: list[float] = []
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

    out = float(np.sqrt(np.mean(sq_errors)))
    if not np.isfinite(out):
        raise ValueError(f"response_loss is not finite: {out}")
    return out


def compute_loss(params: dict, model: pd.DataFrame, human: pd.DataFrame) -> float:
    return response_loss(params, model, human)
