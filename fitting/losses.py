"""
Loss computation for model fitting across experiments.

Supports multiple objectives:

- **``response``** — single response-accuracy objective for all datasets:
  mean squared error on carrabin and yoo; for jiang, total negative log-likelihood
  of human binary choices under ``sigmoid(beta * model_expectation)`` (requires
  ``beta`` in ``params``).
- **``shape``** — distribution / curve distance (Wasserstein): full response
  distribution for carrabin, smoothed mean ``|Δresponse|`` curve for yoo,
  mean per-pid Wasserstein between human and model switch-vs-conflict
  aggregates for jiang (requires ``beta`` for model sampling). The human-side
  target is built from the full task pickle for all ``pid`` in the fold slice;
  the model side uses only fold-filtered ``model`` rows.
- **``joint``** — combined ``(1 - w) * response_loss + w * shape_loss`` with
  dataset-specific ``w`` (see ``JOINT_LOSS_W``).

This module does not depend on the model implementation layer.
"""

import numpy as np
import pandas as pd
import scipy.special
from scipy.stats import wasserstein_distance

DELTA_SMOOTH_WINDOW = 3  # rolling window for smoothing delta curves in shape_loss
POWER_LAW_SMOOTH_WINDOW = 5  # smoothing window for power law fitting in yoo shape loss
JOINT_LOSS_W = {
    "carrabin": 0.2,
    "yoo":      0.5,
    "jiang":    0.95,
}

QID_MIN_TRIALS = 10  # minimum trials per qid to include in carrabin shape loss


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


def _observations_switch_conflict(
    human_df: pd.DataFrame,
    model_responses: pd.Series,
) -> pd.DataFrame:
    """
    Compute (pid, trial, stage, switch, conflict) for all stages > 0.

    ``human_df`` must include jiang columns ``pid``, ``trial``, ``stage``,
    ``value``, ``response`` (human ``response`` is ignored when
    ``model_responses`` is provided).

    ``model_responses`` is indexed by ``(pid, trial, stage)`` with binary
    ``±1`` responses. Conflict is the fraction of neighbor ``value`` entries
    at the current stage that disagree with the model's sign at the previous
    stage.
    """
    rows: list[dict] = []
    for (pid, trial), grp in human_df.groupby(["pid", "trial"], sort=False):
        for st in sorted(grp["stage"].unique()):
            if st <= 0:
                continue
            prev = grp[grp["stage"] == st - 1]
            curr = grp[grp["stage"] == st]
            if len(prev) == 0 or len(curr) == 0:
                continue
            key_p = (int(pid), int(trial), int(st - 1))
            key_c = (int(pid), int(trial), int(st))
            try:
                prev_resp = float(model_responses.loc[key_p])
                curr_resp = float(model_responses.loc[key_c])
            except KeyError:
                continue
            switch = int(
                not np.isclose(prev_resp, curr_resp, rtol=0.0, atol=1e-5)
            )
            prev_dir = float(np.sign(prev_resp))
            if prev_dir == 0.0:
                prev_dir = 1.0
            disagree = (curr["value"].astype(float) != prev_dir).sum()
            n = len(curr)
            conflict = float(disagree / n)
            rows.append(
                {
                    "pid": int(pid),
                    "trial": int(trial),
                    "stage": int(st),
                    "switch": switch,
                    "conflict": conflict,
                }
            )
    return pd.DataFrame(rows)


def _apply_beta_sampling(
    model: pd.DataFrame,
    params: dict,
    seed: int | None = None,
) -> pd.Series:
    """
    Convert continuous model responses to binary ±1 via sigmoid(beta * response),
    then return as Series indexed by (pid, trial, stage).
    """
    if seed is None:
        seed = int(params.get("seed", 42))
    rng = np.random.RandomState(seed)
    beta = float(params["beta"])
    df = model.copy()
    p_pos = scipy.special.expit(beta * df["response"].to_numpy(dtype=float))
    samples = rng.binomial(1, p_pos)
    df["response"] = np.where(samples == 1, 1.0, -1.0)
    s = df.set_index(["pid", "trial", "stage"])["response"]
    if s.index.duplicated().any():
        s = s[~s.index.duplicated(keep="first")]
    return s


def response_loss(
    params: dict,
    model: pd.DataFrame,
    human: pd.DataFrame,
) -> float:
    """
    Response-accuracy loss for all datasets.

    Carrabin and yoo: mean squared error between model and human responses.
    Jiang: total NLL (same summation as former ``nll()``), not averaged per
    observation.
    """
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

        out = float(np.mean(sq_errors))
        if not np.isfinite(out):
            raise ValueError(f"response_loss is not finite: {out}")
        return out

    if dataset == "jiang":
        if "beta" not in params:
            raise ValueError(
                "params must include 'beta' for jiang response_loss computation"
            )
        beta = float(params["beta"])
        pairs = (
            human[["trial", "stage"]]
            .drop_duplicates()
            .sort_values(["trial", "stage"])
        )
        total_logp = 0.0
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
            p = float(
                np.clip(
                    scipy.special.expit(beta * model_response), 1e-10, 1 - 1e-10
                )
            )
            total_logp += np.log(p) if human_response == 1 else np.log(1.0 - p)
        out = float(-total_logp)
        if not np.isfinite(out):
            raise ValueError(f"response_loss (NLL) is not finite: {out}")
        return out

    raise ValueError("params['dataset'] must be one of 'carrabin', 'jiang', 'yoo'")


def shape_loss(
    params: dict,
    model: pd.DataFrame,
    human: pd.DataFrame,
) -> float:
    """
    Distance between human and model response shape:
    - carrabin: |mean_per_qid_std(human) - mean_per_qid_std(model)| (qids with >= QID_MIN_TRIALS trials)
    - yoo: Wasserstein on smoothed mean |delta response| curve
    - jiang: mean per-pid Wasserstein between human and model
      switch-probability-weighted conflict distributions.

    Human-side targets use the full task pickle for all ``pid`` values present
    in ``human``; the model side uses only the fold-filtered ``model`` frame.
    """
    dataset = params["dataset"]
    if dataset not in ("carrabin", "yoo", "jiang"):
        raise ValueError(
            f"shape_loss() is not implemented for dataset={dataset!r}"
        )

    from utils.paths import data_path

    pids = human["pid"].unique()
    human_full = pd.read_pickle(data_path(f"{dataset}.pkl"))
    human_full = human_full[human_full["pid"].isin(pids)]

    if dataset == "carrabin":
        # merge qid into model responses using human trial/observation index
        qid_map = human_full[["pid", "trial", "observation", "qid"]].drop_duplicates()
        model_with_qid = model.merge(
            qid_map, on=["pid", "trial", "observation"], how="left"
        )
        h_std = _mean_qid_std(human_full)
        m_std = _mean_qid_std(model_with_qid)
        if not np.isfinite(h_std) or not np.isfinite(m_std):
            raise ValueError(
                f"shape_loss: non-finite qid std (human={h_std}, model={m_std})"
            )
        return float(abs(h_std - m_std))
    if dataset == "yoo":
        h_params = _fit_power_law_params(human_full)
        m_params = _fit_power_law_params(model)
        if h_params.empty or m_params.empty:
            return float("nan")
        merged = h_params.merge(m_params, on="pid", suffixes=("_h", "_m"))
        if merged.empty:
            return float("nan")
        loss_A = (merged["A_h"] - merged["A_m"]).abs().mean()
        loss_lambda = (merged["lambda__h"] - merged["lambda__m"]).abs().mean()
        return float(loss_A + loss_lambda)
    if dataset == "jiang":
        if "beta" not in params:
            raise ValueError("params must include 'beta' for jiang shape_loss")
        model_series = _apply_beta_sampling(model, params)
        pid_losses: list[float] = []
        for _pid, h_pid_full in human_full.groupby("pid"):
            human_s = (
                h_pid_full.set_index(["pid", "trial", "stage"])["response"]
                .astype(float)
                .apply(lambda x: 1.0 if float(x) > 0 else -1.0)
            )
            if human_s.index.duplicated().any():
                human_s = human_s[~human_s.index.duplicated(keep="first")]
            obs_human = _observations_switch_conflict(h_pid_full, human_s)
            obs_model = _observations_switch_conflict(h_pid_full, model_series)
            if obs_human.empty or obs_model.empty:
                continue
            h_human_agg = obs_human.groupby("conflict")["switch"].mean().reset_index()
            h_model_agg = obs_model.groupby("conflict")["switch"].mean().reset_index()
            if h_human_agg.empty or h_model_agg.empty:
                continue
            if h_human_agg["switch"].sum() == 0:
                continue
            if h_model_agg["switch"].sum() == 0:
                pid_losses.append(1.0)  # maximum penalty: model never switches
                continue
            loss = float(
                wasserstein_distance(
                    h_human_agg["conflict"].to_numpy(dtype=float),
                    h_model_agg["conflict"].to_numpy(dtype=float),
                    u_weights=h_human_agg["switch"].to_numpy(dtype=float),
                    v_weights=h_model_agg["switch"].to_numpy(dtype=float),
                )
            )
            pid_losses.append(loss)
        if not pid_losses:
            return float("nan")
        return float(np.mean(pid_losses))


def joint_loss(
    params: dict,
    model: pd.DataFrame,
    human: pd.DataFrame,
) -> float:
    """
    Combined response and shape loss: (1-w) * response_loss + w * shape_loss.
    - carrabin: w=0.2
    - yoo: w=0.5
    - jiang: w=0.95 (default; tune via ``wasserstein_w`` / ``JOINT_LOSS_W``)
    """
    dataset = params["dataset"]
    if dataset not in ("carrabin", "yoo", "jiang"):
        raise ValueError(
            f"joint_loss() is only implemented for carrabin, yoo, and jiang; "
            f"got dataset={dataset!r}"
        )
    w = float(params.get("wasserstein_w", JOINT_LOSS_W[dataset]))
    return (1.0 - w) * response_loss(params, model, human) + w * shape_loss(
        params, model, human
    )


def compute_loss(
    loss_type: str, params: dict, model: pd.DataFrame, human: pd.DataFrame
) -> float:
    if loss_type == "response":
        return response_loss(params, model, human)
    if loss_type == "shape":
        return shape_loss(params, model, human)
    if loss_type == "joint":
        return joint_loss(params, model, human)
    raise ValueError(f"Unknown loss_type: {loss_type!r}")
