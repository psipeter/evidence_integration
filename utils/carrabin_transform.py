"""
utils/carrabin_transform.py

Task-appropriate response transformation for the carrabin dataset.

Carrabin asks participants to estimate the proportion of red balls from
binary {-1, +1} observations. A raw weighted mean of these inputs gives
extreme estimates from small samples. Applying Laplace (add-one) smoothing:

    response = response_raw * t / (t + 2)

converts any raw mean estimate into a regularised proportion estimate,
reflecting that a single binary observation provides weak evidence about
the underlying proportion. When applied to the Mean model this recovers
the optimal Bayesian (Bayes) estimate exactly.

Models that already implement internal response calibration (NoisyCounting,
and in future any model with an explicit noise/temperature parameter) are
exempt from the transform: their response_raw is used as response directly.
Both columns are always saved for downstream analysis.
"""

import pandas as pd

# Models whose internal mechanisms already handle response calibration.
# The t/(t+2) transform is NOT applied to these — doing so would
# double-penalise their fitted noise/temperature parameters.
_EXEMPT_MODELS = frozenset({"NoisyCounting"})


def apply_carrabin_transform(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add `response_raw` column and apply Laplace smoothing to `response`.

    For carrabin (non-exempt models):
        response_raw = original model output
        response     = response_raw * t / (t + 2)

    For carrabin exempt models (NoisyCounting):
        response_raw = response (copy; no transform applied)

    For all other datasets:
        response_raw = response (identity; schema consistency)

    Parameters
    ----------
    df      : DataFrame with columns including `response`, `observation`,
              and `model_type`
    dataset : dataset name string

    Returns
    -------
    DataFrame with `response_raw` added and `response` transformed where
    appropriate.
    """
    df = df.copy()
    df["response_raw"] = df["response"]

    if dataset != "carrabin":
        return df

    # determine exemption per row (handles mixed-model DataFrames safely)
    if "model_type" in df.columns:
        exempt_mask = df["model_type"].isin(_EXEMPT_MODELS)
    else:
        exempt_mask = pd.Series(False, index=df.index)

    t = df["observation"]
    df.loc[~exempt_mask, "response"] = (
        df.loc[~exempt_mask, "response_raw"] * t[~exempt_mask] / (t[~exempt_mask] + 2)
    )
    return df
