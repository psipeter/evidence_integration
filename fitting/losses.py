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

def compute_sim_db_loss(
    model_type: str,
    params: dict,
    human_pid: pd.DataFrame,
    db_dir: "Path | str",
) -> float:
    """Group-level log-likelihood from simulation database.

    For each (sequence, obs) cell, evaluates the likelihood of the full set of
    observed responses under the model's predicted Gaussian. This penalises both
    mean mismatch and variance mismatch — a model with correct mean but wrong
    variance, or correct variance but wrong mean, both score poorly.

    The group log-likelihood of n observed responses under N(mu_sim, sigma_sim^2):
        sum_i log N(r_i | mu_sim, sigma_sim^2)
        = -n/2 log(2*pi*sigma^2) - n/(2*sigma^2) * [var_obs + (mean_obs-mu_sim)^2]
    The sigma^2 term penalises over-dispersion; the squared-mean-error term
    penalises mean mismatch. Both must be small for high likelihood.

    Returns negative mean log-likelihood per observation (lower = better).
    """
    import hashlib, json
    from collections import defaultdict
    from pathlib import Path
    from scipy.stats import norm

    db_dir  = Path(db_dir)
    SKIP    = {"pid", "model_type", "dataset", "seed", "base_seed"}
    free    = {k: v for k, v in params.items() if k not in SKIP}
    key     = json.dumps({"model": model_type, "params": free}, sort_keys=True)
    ph      = hashlib.md5(key.encode()).hexdigest()[:12]
    db_path = db_dir / model_type / f"{model_type}_{ph}.pkl"

    if not db_path.exists():
        raise FileNotFoundError(
            f"Simulation database not found: {db_path}\n"
            f"Run: python scripts/build_sim_db.py --model {model_type} "
            f"--params_json \'{json.dumps(free)}\'"
        )

    db = pd.read_pickle(db_path)["data"]   # {seq_tuple: (n_sims, n_obs)}

    # Group all observed responses by (seq, obs_idx)
    cell_obs: dict[tuple, list] = defaultdict(list)
    for _, tdf in human_pid.groupby("trial"):
        tdf = tdf.sort_values("observation")
        seq = tuple(tdf["value"].values)
        for obs_idx, r in enumerate(tdf["response"].values):
            cell_obs[(seq, obs_idx)].append(float(r))

    total_ll  = 0.0
    n_obs_total = 0

    for (seq, obs_idx), r_list in cell_obs.items():
        if seq not in db:
            continue
        sim_trajs = db[seq]
        if sim_trajs.shape[0] < 1:
            continue  # need at least 1 simulation
        sim_col  = sim_trajs[:, obs_idx]
        mu_sim   = float(sim_col.mean())
        sig_sim  = max(float(sim_col.std()), 1e-3)
        r_arr    = np.array(r_list)
        total_ll += float(np.sum(norm.logpdf(r_arr, loc=mu_sim, scale=sig_sim)))
        n_obs_total += len(r_arr)

    if n_obs_total == 0:
        raise ValueError("No valid (seq, obs) cells found in database for this pid")

    return float(-total_ll / n_obs_total)

