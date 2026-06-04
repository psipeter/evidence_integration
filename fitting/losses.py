"""
Loss computation for model fitting.

compute_loss(params, model, human): RMSE between model and human responses.
Supports datasets: carrabin, yoo.
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
    """RMSE between model and human responses (carrabin, yoo).

    For carrabin, model responses are expected to already have the Laplace
    smoothing transform applied (response = response_raw * t / (t+2)), as
    produced by utils.carrabin_transform.apply_carrabin_transform(). The
    `response` column is used directly; `response_raw` is ignored here.
    """
    dataset = params["dataset"]
    if dataset not in ("carrabin", "yoo"):
        raise ValueError("params['dataset'] must be one of 'carrabin', 'yoo'")

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
    """Marginal log-likelihood from simulation database.

    For each trial a pid ran, looks up the (n_seeds, n_obs) response
    trajectory matrix for that input sequence, fits a Gaussian per
    observation, and evaluates the log-likelihood of the observed response.
    Returns negative mean log-likelihood (lower = better, like RMSE).

    The per-observation Gaussian: μ = mean(sim_responses), σ = std(sim_responses).
    For deterministic models σ is floored at 1e-3 so likelihood is defined.

    Parameters
    ----------
    model_type : str
    params : dict  — model parameters (used only for hashing to find db file)
    human_pid : pd.DataFrame  — one pid's rows from carrabin.pkl
        Must have columns: trial, observation, value, response
    db_dir : Path  — root of simulation database (contains {model_type}/ subdir)

    Returns
    -------
    float : negative mean log-likelihood across all (trial, obs) pairs
    """
    import hashlib, json
    from pathlib import Path
    from scipy.stats import norm

    db_dir = Path(db_dir)
    SKIP   = {"pid", "model_type", "dataset", "seed", "base_seed"}
    free   = {k: v for k, v in params.items() if k not in SKIP}
    key    = json.dumps({"model": model_type, "params": free}, sort_keys=True)
    ph     = hashlib.md5(key.encode()).hexdigest()[:12]
    db_path = db_dir / model_type / f"{model_type}_{ph}.pkl"

    if not db_path.exists():
        raise FileNotFoundError(
            f"Simulation database not found: {db_path}\n"
            f"Run: python scripts/build_sim_db.py --model {model_type} "
            f"--params_json '{json.dumps(free)}'"
        )

    db      = pd.read_pickle(db_path)["data"]   # {seq_tuple: (n_seeds, n_obs)}
    log_liks = []

    for trial, trial_df in human_pid.groupby("trial"):
        trial_df = trial_df.sort_values("observation")
        # Full input sequence for this trial
        seq = tuple(trial_df["value"].values)
        if seq not in db:
            continue
        sim_trajs = db[seq]          # (n_seeds, n_obs)
        if sim_trajs.shape[0] < 2:
            continue

        obs_responses = trial_df["response"].values   # (n_obs,)
        for obs_idx, r_obs in enumerate(obs_responses):
            sim_col = sim_trajs[:, obs_idx]            # n_seeds values
            mu  = float(sim_col.mean())
            sig = float(sim_col.std())
            sig = max(sig, 1e-3)                       # floor for deterministic
            log_liks.append(norm.logpdf(r_obs, loc=mu, scale=sig))

    if not log_liks:
        raise ValueError("No valid (trial, obs) pairs found in database for this pid")

    return float(-np.mean(log_liks))
