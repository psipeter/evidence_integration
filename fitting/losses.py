"""
Universal negative log-likelihood (NLL) computation across all tasks.

This module is the unified loss layer for model fitting and has no dependency
on the model implementation layer. It consumes precomputed model responses and
empirical participant data, then evaluates task-specific likelihoods.

The unified approach is:
- Gaussian observation noise for continuous-response tasks (carrabin, yoo)
- Sigmoid decision rule for binary-response jiang

Conceptually, ``sigma`` (carrabin/yoo) and ``beta`` (jiang) are both inverse
noise controls: they govern how deterministically internal model responses map
to observed behavior.
"""

import numpy as np
import pandas as pd
import scipy.special


def nll(params: dict, model: pd.DataFrame, human: pd.DataFrame) -> float:
    """Compute total negative log-likelihood for one participant."""
    dataset = params["dataset"]

    if dataset in ("carrabin", "yoo"):
        sigma = float(params["sigma"])
        if sigma <= 0:
            raise ValueError("params['sigma'] must be positive for carrabin/yoo")

        const = -0.5 * np.log(2.0 * np.pi * sigma**2)
        total_logp = 0.0

        pairs = (
            human[["trial", "observation"]]
            .drop_duplicates()
            .sort_values(["trial", "observation"])
        )
        for _, pair in pairs.iterrows():
            trial = int(pair["trial"])
            observation = int(pair["observation"])

            h = human.query("trial == @trial & observation == @observation")["response"]
            m = model.query("trial == @trial & observation == @observation")["response"]
            if h.empty or m.empty:
                raise ValueError(
                    f"Missing response for (trial={trial}, observation={observation})"
                )

            # REVIEW: Using first row mirrors historical code assumptions of one row per pair.
            human_response = float(h.iloc[0])
            model_response = float(m.iloc[0])
            total_logp += const - (human_response - model_response) ** 2 / (
                2.0 * sigma**2
            )

    elif dataset == "jiang":
        beta = float(params["beta"])
        total_logp = 0.0

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

            # REVIEW: Require unique empirical outcome by task definition.
            if h.nunique() != 1:
                raise ValueError(
                    f"Non-unique human response at (trial={trial}, stage={stage})"
                )

            human_response = float(h.iloc[0])
            model_response = float(m.iloc[0])
            p = float(np.clip(scipy.special.expit(beta * model_response), 1e-10, 1 - 1e-10))
            total_logp += np.log(p) if human_response == 1 else np.log(1.0 - p)

    else:
        raise ValueError("params['dataset'] must be one of 'carrabin', 'jiang', 'yoo'")

    total_nll = float(-total_logp)
    if not np.isfinite(total_nll):
        raise ValueError(f"NLL is not finite: {total_nll}")
    return total_nll
