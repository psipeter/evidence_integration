"""
Loss computation for model fitting.

compute_loss(params, model, human): RMSE between model and human responses.
Supports datasets: carrabin, yoo, soltani_colors, soltani_numbers.
"""

import numpy as np
import pandas as pd

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
    """RMSE between model and human responses (carrabin, yoo, soltani_colors,
    soltani_numbers).

    All four datasets store `response` on the canonical [-1,1] scale, and this
    function compares model to human on that scale directly -- so a model must
    emit [-1,1] to be scored correctly.

    For carrabin ONLY, model responses are expected to already have the Laplace
    smoothing transform applied (response = response_raw * t / (t+2)), as
    produced by utils.carrabin_transform.apply_carrabin_transform() (a thin
    wrapper around utils.binary_transform.apply_binary_transform()). The
    `response` column is used directly; `response_raw` is ignored here.

    yoo, soltani_numbers AND soltani_colors use raw (untransformed) responses.
    Note that soltani_colors is untransformed despite having {-1,+1}
    observations like carrabin: both soltani tasks ask for the MEAN of all
    observations rather than a probability, so no shrinkage toward 0 applies.
    See utils/binary_transform.py's own module docstring.
    """
    dataset = params["dataset"]
    _SUPPORTED = ("carrabin", "yoo", "soltani_colors", "soltani_numbers")
    if dataset not in _SUPPORTED:
        raise ValueError(f"params['dataset'] must be one of {_SUPPORTED}")

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



# ── PMMH-style likelihood from simulation database ────────────────────────────

# Floor on the ensemble SD, to keep the Gaussian NLL finite. NOT a free
# parameter and NOT a way to admit deterministic models: a deterministic model's
# ensemble SD is exactly 0, and clamping it turns the NLL into scaled squared
# error with an arbitrary scale -- which is exactly what a naive sigma clamp
# would silently do. compute_nll REFUSES deterministic models instead. This
# floor exists only for the rare cell where a genuinely stochastic model happens
# to produce near-identical responses across sims (e.g. both noise SDs at their
# lower bounds and a clipped response).
NLL_SIGMA_FLOOR = 1e-3


def nll_from_ensemble(ens: np.ndarray, y: np.ndarray,
                      sigma_floor: float = NLL_SIGMA_FLOOR) -> float:
    """Gaussian NLL of `y` (one observed value per column) under the per-column
    (mean, SD) of `ens` (n_sims x n_columns). The arithmetic core of
    compute_nll and _cross_validate_nll, factored out so a fold can subset
    `ens`'s columns and reuse it without re-simulating."""
    if ens.shape[1] != len(y):
        raise ValueError(f"ensemble has {ens.shape[1]} columns, y has {len(y)} rows")
    mu = ens.mean(axis=0)
    sigma = np.maximum(ens.std(axis=0, ddof=1), sigma_floor)
    return float(np.mean(np.log(sigma) + (y - mu) ** 2 / (2.0 * sigma ** 2)))


# compute_nll() (an earlier convenience wrapper calling the now-retired
# math_models.simulate_ensemble directly) was removed when NoisyRL_lambda was
# retired -- see archive/models/archive_math_models_noise.py and
# docs/DECISIONS.md. It was already dead code: fitting.fit's objective()
# builds the ensemble itself (once per Optuna trial, via add_noise for the
# still-active _resp_noise models) and calls nll_from_ensemble() directly,
# so no caller depended on compute_nll. If ever restored for a genuinely
# stochastic model again, its old implementation is in git history
# (fitting/losses.py, pre-retirement) and archive/models/archive_math_models_noise.py.

# compute_sim_db_loss() (group-level log-likelihood from a simulation
# database keyed by params-hash) was removed when the MLE pipeline's last
# loose ends were cleaned up -- see docs/DECISIONS.md's "State-noise
# models, NoisyCounting, and their MLE/NLL pipelines retired from active
# analysis" entry and archive/HISTORY_modeling_2026.md. Its only callers
# were the already-archived archive/fitting/archive_fit_mle.py and
# archive/fitting/archive_collect_mle.py; now archived itself, unchanged,
# at archive/fitting/archive_losses_mle.py.

