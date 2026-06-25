"""
utils/binary_transform.py

Laplace-smoothed response transform for binary and continuous tasks.

Both the binary task ({-1,+1} observations) and the continuous task
(observations normalised from [-100,100] to [-1,1]) share the same
uninformative prior: 0. Laplace smoothing shrinks raw estimates toward
this prior:

    response = response_raw * t / (t + 2)

This is equivalent to adding one pseudo-observation of each sign before
computing the mean, and is the optimal Bayesian estimate under a uniform
prior on [-1, 1].

For the carrabin dataset (legacy) the same transform applies — carrabin
observations are also {-1, +1}.

Models that already implement internal response calibration (NoisyCounting)
are exempt: their output is used directly.

Normalisation conventions
-------------------------
  continuous task : x_norm = 2*x/100 - 1  (maps stimulus [0, 100] → NEF [-1, 1])
                    r_model = (r_norm + 1) / 2  (NEF output back to [0, 1] mean scale)
                    r_slider = r_model * 100  (back to [0, 100] slider)
  binary task     : x_norm = x  (already {-1,+1}; identity)
                    r_model = r_norm on [-1, 1] after Laplace smoothing
                    r_prob = (r_norm + 1) / 2  (probability [0, 1] for display)
  carrabin        : observations already {-1,+1}; same transform applies
  yoo             : observations already ~[-1, 1]; no rescaling
"""

import numpy as np
import pandas as pd

# Models whose internal mechanisms already handle response calibration.
_EXEMPT_MODELS = frozenset({"NoisyCounting"})


def apply_binary_transform(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add `response_raw` column and apply Laplace smoothing to `response`.

    For binary/continuous/carrabin (non-exempt models):
        response_raw = original model output (on [-1,1] normalised scale)
        response     = response_raw * t / (t + 2)

    For exempt models (NoisyCounting):
        response_raw = response (copy; no transform applied)

    For datasets not requiring the transform:
        response_raw = response (identity; schema consistency)

    Parameters
    ----------
    df      : DataFrame with columns including `response`, `observation`,
              and `model_type`
    dataset : dataset name string — transform applied for
              'carrabin', 'task_continuous', 'task_binary'

    Returns
    -------
    DataFrame with `response_raw` added and `response` transformed where
    appropriate.
    """
    _TRANSFORM_DATASETS = frozenset({'carrabin', 'task_binary'})  # task_continuous uses raw responses like yoo

    df = df.copy()
    df["response_raw"] = df["response"]

    if dataset not in _TRANSFORM_DATASETS:
        return df

    if "model_type" in df.columns:
        exempt_mask = df["model_type"].isin(_EXEMPT_MODELS)
    else:
        exempt_mask = pd.Series(False, index=df.index)

    # Use (observation+1) as the 1-indexed count so the first obs (0-indexed)
    # gets factor 1/3 rather than 0/2=0. Matches carrabin convention where
    # observations index from 1.
    t = df["observation"] + 1
    df.loc[~exempt_mask, "response"] = (
        df.loc[~exempt_mask, "response_raw"] * t[~exempt_mask] / (t[~exempt_mask] + 2)
    )
    return df


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalise_continuous(x):
    """Map continuous-task stimulus [0, 100] → NEF input [-1, 1]."""
    return 2.0 * float(x) / 100.0 - 1.0


def denormalise_continuous_response(r_norm):
    """Map NEF response [-1, 1] → normalized mean estimate [0, 1]."""
    return (float(r_norm) + 1.0) / 2.0


def denormalise_continuous_slider(r_norm):
    """Map NEF response [-1, 1] → slider [0, 100]."""
    return denormalise_continuous_response(r_norm) * 100.0


def denormalise_continuous(r_norm):
    """Alias for slider-scale denormalisation (backwards compatibility)."""
    return denormalise_continuous_slider(r_norm)


def nef_obs_values(values: np.ndarray, dataset: str) -> np.ndarray:
    """Observation values in the scale expected by the NEF network."""
    vals = np.asarray(values, dtype=float)
    if dataset == "task_continuous":
        return np.array([normalise_continuous(v) for v in vals], dtype=float)
    return vals


def nef_response_to_model_scale(response: float, dataset: str) -> float:
    """Map raw NEF readout to the response scale used by other task models."""
    if dataset == "task_continuous":
        return denormalise_continuous_response(response)
    return float(response)


def normalise_binary(x):
    """Binary task observations are already {-1, +1}. Identity for consistency."""
    return float(x)


def denormalise_binary_response(r_norm):
    """Map model responses [-1, 1] → probability [0, 1]."""
    return (r_norm + 1.0) / 2.0
