"""
utils/binary_transform.py

Laplace-smoothed response transform. APPLIES TO CARRABIN ONLY.

Laplace smoothing shrinks a raw estimate toward an uninformative prior of 0:

    response = response_raw * t / (t + 2)

equivalent to adding one pseudo-observation of each sign before taking the
mean, and the optimal Bayesian estimate of a PROBABILITY under a uniform prior.
Carrabin's task is exactly that: estimate the hidden probability p generating
{-1,+1} draws.

WHY THE SOLTANI DATASETS ARE NOT TRANSFORMED
---------------------------------------------
soltani_numbers and soltani_colors both ask the participant to report the MEAN
OF ALL OBSERVATIONS SO FAR -- not a probability with a prior. Shrinking a
reported mean toward 0 is simply the wrong operation for that task, so neither
dataset appears in _TRANSFORM_DATASETS. This holds for BOTH tasks despite
colors having {-1,+1} observations like carrabin: the observation alphabet is
not what decides it, the QUANTITY BEING REPORTED is.

(Historical note: soltani_colors was previously in the transform set and
soltani_numbers had a [0,100]<->[-1,1] rescale applied in nef_obs_values /
nef_response_to_model_scale below. Both were written for the retired task/
pipeline, where observations arrived on their native scales and no real human
data existed. They are wrong under the current pkl convention -- see the scale
audit below -- and have been removed. Do not reintroduce either.)

SCALE CONVENTION FOR THE SOLTANI DATASETS (audited against real data)
----------------------------------------------------------------------
scripts/build_model_inputs.build_from_df() already puts everything on the
canonical [-1,1] scale that carrabin/yoo use, so NO rescaling is needed in
either direction. Verified directly against
data/soltani_{numbers,colors}_complete_pairs.pkl (21 pids, 10080 rows each):

  soltani_numbers : value    continuous on [-0.98, 0.98]  (99 distinct)
                    response continuous on [-1, 1]        (101 distinct)
  soltani_colors  : value    exactly {-1, +1}             (2 distinct)
                    response continuous on [-1, 1]        (101 distinct)

`response` is on [-1,1] for BOTH tasks (101 distinct values = a 0-100 slider
mapped linearly). Therefore any model -- NEF or math -- must emit responses on
[-1,1] to be comparable with the human data that fitting.losses scores it
against, which means identity in both nef_obs_values and
nef_response_to_model_scale.

Note that colors' `true_p` is deliberately left on its native [0,1] probability
scale by build_from_df while colors' `response` is on [-1,1]; do not assume
ground-truth and response columns share a scale. The soltani figures avoid this
entirely by using the running mean of `value` (hence [-1,1], matching
`response`) as ground truth rather than true_p.

Models that already implement internal response calibration (NoisyCounting)
are exempt from the carrabin transform: their output is used directly.

Normalisation conventions
-------------------------
  carrabin        : observations already {-1,+1}; Laplace transform applies
  yoo             : observations already ~[-1, 1]; no rescaling, no transform
  soltani_numbers : value/response already [-1,1]; no rescaling, no transform
  soltani_colors  : value {-1,+1}, response [-1,1]; no rescaling, no transform
"""

import numpy as np
import pandas as pd

# Models whose internal mechanisms already handle response calibration.
_EXEMPT_MODELS = frozenset({"NoisyCounting"})

# Largest |observation| nef_obs_values will accept before assuming the caller
# handed it raw pre-build (0-100) data. Set just above 1.0 to allow for float
# noise while still catching a 0-100 stream immediately; the NEF's own
# radius_e=1.5 is the point past which its ensembles start saturating.
_MAX_ABS_OBS = 1.5


def apply_binary_transform(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add `response_raw` column and apply Laplace smoothing to `response`.

    For carrabin (non-exempt models):
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
    dataset : dataset name string — transform applied for 'carrabin' ONLY.
              yoo, soltani_numbers and soltani_colors all report a MEAN, not a
              probability, so they are returned with `response` untouched (see
              the module docstring).

    Returns
    -------
    DataFrame with `response_raw` added and `response` transformed where
    appropriate.
    """
    # carrabin only. Do NOT add the soltani datasets here: both tasks ask for
    # the mean of all observations, and shrinking a mean toward 0 is the wrong
    # operation. See the module docstring.
    _TRANSFORM_DATASETS = frozenset({'carrabin'})

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
    """Observation values in the scale expected by the NEF network ([-1,1]).

    Identity for every current dataset, because the CANONICAL PKL INPUTS ARE
    ALREADY ON [-1,1] -- not because these tasks are natively scaled that way.
    scripts/build_model_inputs.build_from_df() does the conversion upstream:
    numbers `value`/`response`/`true_mean` go through x' = x/50 - 1, while
    colors `value` is left alone because it is already +-1 (blue/red).

    So the two scales genuinely in play are:
      raw/pre-build (data/task_results_*.pkl, task_backend's Supabase output,
        and the sequence pool): numbers value 0-100, response 0-100
      built/canonical (data/soltani_*[_datafile].pkl, what NEF.run reads):
        numbers value [-0.98,0.98], colors value {-1,+1}, response [-1,1]

    A soltani_numbers branch here previously applied normalise_continuous
    (2*x/100 - 1), which is correct for the RAW scale and catastrophic for the
    built one: it maps [-0.98,0.98] to [-1.0196,-0.9804], collapsing the entire
    stimulus range into a 0.04-wide band pinned near -1. Verified against
    data/soltani_numbers_complete_pairs.pkl: 99 distinct values, none with
    |value| > 1. Do not reintroduce it for the pkl path.

    Raises
    ------
    ValueError
        If any |value| > _MAX_ABS_OBS, which almost certainly means raw 0-100
        data reached the NEF instead of a build_from_df output. Failing loudly
        here is deliberate: the NEF's ensembles are built with radius_e=1.5 and
        radius_v=1.0, so out-of-range input saturates them and produces
        plausible-looking but meaningless responses rather than an error.
    """
    vals = np.asarray(values, dtype=float)
    if vals.size:
        peak = float(np.nanmax(np.abs(vals)))
        if peak > _MAX_ABS_OBS:
            raise ValueError(
                f"nef_obs_values got |value| up to {peak:.3g} for dataset "
                f"{dataset!r}, outside the expected [-1,1] NEF input range. "
                "This looks like raw pre-build data (numbers `value` is 0-100 "
                "in data/task_results_*.pkl and in task_backend's sequence "
                "pool). NEF must be fed a build_from_df output "
                "(data/soltani_*[_datafile].pkl), where value is already "
                "rescaled to [-1,1]. Do NOT re-add a /50-1 rescale here -- "
                "that would double-rescale the canonical files."
            )
    return vals


def nef_response_to_model_scale(response: float, dataset: str) -> float:
    """Map raw NEF readout to the response scale used by other task models.

    Identity for every current dataset. Human `response` is on [-1,1] for
    carrabin, yoo, soltani_numbers AND soltani_colors, and fitting.losses
    compares model output to it directly, so the NEF readout is already on the
    right scale. A soltani_numbers branch here previously mapped the readout to
    [0,1] via denormalise_continuous_response, which would have been scored
    against human responses on [-1,1]. Do not reintroduce it.
    """
    return float(response)


def normalise_binary(x):
    """Binary task observations are already {-1, +1}. Identity for consistency."""
    return float(x)


def denormalise_binary_response(r_norm):
    """Map model responses [-1, 1] → probability [0, 1]."""
    return (r_norm + 1.0) / 2.0
