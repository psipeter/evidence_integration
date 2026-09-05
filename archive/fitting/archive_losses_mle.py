"""
compute_sim_db_loss, extracted verbatim from fitting/losses.py (2026-09-05),
completing the MLE-pipeline retirement whose main decision and rationale
live in docs/DECISIONS.md ("State-noise models, NoisyCounting, and their
MLE/NLL pipelines retired from active analysis"). By the time this was
extracted, its only callers were already-archived: `archive/fitting/
archive_fit_mle.py` and `archive/fitting/archive_collect_mle.py` (both
`from fitting.losses import compute_sim_db_loss`). A repo-wide grep
(excluding archive/, venv/, node_modules/, .git/) confirmed no other
caller. `fitting/losses.py` itself remains active -- only this one
function, unused by anything still live, was removed from it.

Extracted:
- `compute_sim_db_loss`: group-level Gaussian log-likelihood loss,
  evaluated against a params-hash-keyed simulation database on disk
  (built by `scripts/build_sim_db.py`, now `archive/scripts/
  build_sim_db.py`). Penalises both mean and variance mismatch between
  a pid's observed responses and the model's simulated ensemble at a
  given (sequence, observation-index) cell.

How to restore: copy `compute_sim_db_loss` back into `fitting/losses.py`
(it needs no other change -- its body is unchanged from what lived
there), and re-add `from fitting.losses import compute_sim_db_loss` to
whichever of `archive/fitting/archive_fit_mle.py` /
`archive/fitting/archive_collect_mle.py` is being restored alongside it.
Also requires `scripts/build_sim_db.py` restored from
`archive/scripts/build_sim_db.py` (NOT `archive/scripts/
build_sim_db_early_draft.py` -- see that file's own header for why there
are two archived versions) for `compute_sim_db_loss`'s own
`FileNotFoundError` fix-it message to point at a real, runnable command.
"""

import numpy as np
import pandas as pd


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
            f"build_sim_db.py was retired with the MLE pipeline (see "
            f"docs/DECISIONS.md); restore archive/scripts/build_sim_db.py to "
            f"scripts/ first, then run: python scripts/build_sim_db.py "
            f"--model {model_type} --params_json \'{json.dumps(free)}\'"
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
