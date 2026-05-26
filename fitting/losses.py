"""
Loss computation for model fitting.

compute_loss(params, model, human): RMSE between model and human responses.
Supports datasets: carrabin, yoo, diederen.
Diederen: catch trials excluded; optionally restricted to first 2 blocks
per distribution per session (DIEDEREN_FIRST_BLOCKS_ONLY flag).
"""

import numpy as np
import pandas as pd

# Diederen: restrict loss to first 2 blocks per distribution per session.
# Set to False to use all observations.
DIEDEREN_FIRST_BLOCKS_ONLY: bool = False


def _filter_first_blocks(human: pd.DataFrame, n_blocks: int = 2) -> pd.DataFrame:
    """
    Keep only the first `n_blocks` consecutive blocks per distribution
    within each (pid, session). A block is a consecutive run of observations
    from one distribution (distrib_index).
    """
    out = []
    for (pid, session), grp in human.groupby(["pid", "session"], sort=False):
        g = grp.sort_values("trial_in_session").reset_index(drop=True)
        distribs = sorted(g["distrib_index"].dropna().unique().tolist())
        if len(distribs) != 2:
            out.append(g)
            continue
        block_count = {d: 0 for d in distribs}
        prev = None
        keep = []
        for i in range(len(g)):
            curr = int(g.at[i, "distrib_index"])
            if prev is not None and curr != prev:
                block_count[prev] += 1
            if block_count[curr] < n_blocks:
                keep.append(i)
            prev = curr
        if keep:
            out.append(g.iloc[keep])
    return pd.concat(out, ignore_index=True) if out else human.iloc[0:0]


def compute_loss(
    params: dict, model: pd.DataFrame, human: pd.DataFrame
) -> float:
    """RMSE between model and human responses (carrabin, yoo, diederen)."""
    dataset = params["dataset"]
    if dataset not in ("carrabin", "yoo", "diederen"):
        raise ValueError(
            "params['dataset'] must be one of 'carrabin', 'yoo', 'diederen'"
        )

    if dataset == "diederen":
        human = human[~human["catch_trial"].astype(bool)]
        if DIEDEREN_FIRST_BLOCKS_ONLY:
            human = _filter_first_blocks(human)
            model = model[
                model.set_index(["pid", "trial", "observation"]).index.isin(
                    human.set_index(["pid", "trial", "observation"]).index
                )
            ]

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
        raise ValueError(f"compute_loss is not finite: {out}")
    return out
