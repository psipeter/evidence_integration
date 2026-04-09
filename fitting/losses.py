"""
Loss computation for model fitting across experiments.

Supports multiple objectives:

- **Experiment 1:** ``mse`` — mean squared error between model and human
  responses; task-agnostic baseline for all three datasets.
- **Experiment 2:** Task-specific losses (stubs): ``excursion`` (carrabin:
  distributional / sequence variance), ``switch`` (jiang: switch probability
  vs. conflict), ``decay`` (yoo: power-law decay of update magnitude).

This module does not depend on the model implementation layer.
"""

import numpy as np
import pandas as pd
import scipy.special


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
            err = human_response - model_response
            sq_errors.append(err**2)

    else:
        raise ValueError("params['dataset'] must be one of 'carrabin', 'jiang', 'yoo'")

    out = float(np.mean(sq_errors))
    if not np.isfinite(out):
        raise ValueError(f"MSE is not finite: {out}")
    return out


def excursion_loss(params: dict, model: pd.DataFrame, human: pd.DataFrame) -> float:
    # TODO: distributional loss over response variance per qid sequence
    # Used in Experiment 2 for carrabin
    raise NotImplementedError


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
    if loss_type == "excursion":
        return excursion_loss(params, model, human)
    if loss_type == "switch":
        return switch_loss(params, model, human)
    if loss_type == "decay":
        return decay_loss(params, model, human)
    raise ValueError(f"Unknown loss_type: {loss_type!r}")
