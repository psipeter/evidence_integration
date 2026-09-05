"""Verify models.math_models.add_noise against run() directly, for every
dataset/model combination this project fits under --loss nll.

Not a pytest suite -- this project has none. Run this manually after
touching add_noise, _resp_noise_seed, or _validate_model_dataset's
allowlists, and before trusting a fit that used --loss nll on a
dataset/model it has not been run against before.

Checks add_noise, for deterministic base models wrapped with i.i.d.
response noise (_NOISE_WRAPPABLE_BASE_MODELS, e.g. RL_lambda via
"RL_lambda_resp_noise"):
  (a) it reduces EXACTLY to run()'s deterministic output at sigma=0,
  (b) the empirical mean across a large ensemble is close to that same
      deterministic mu (a sanity bound on Monte Carlo error, not an exact
      equality), and (c) the empirical SD is close to the requested
      sigma_resp.
  (d) it accepts BOTH the bare model name ("RL_lambda") and the
      fitting-time suffixed name ("RL_lambda_resp_noise") identically,
      since fit.py passes the suffixed name and a bug here silently broke
      that path once already.

NOTE: this script used to also check models.math_models.simulate_ensemble
against run(seed=i) for genuinely-stochastic models (NoisyRL_lambda). That
model, and simulate_ensemble itself, were retired (state-noise models
phased out of active analysis -- see docs/DECISIONS.md), so that check is
gone too. If ever restored, see
archive/models/archive_math_models_noise.py and git history for this
script's pre-retirement version.
"""
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from models import math_models as mm
from models.math_models import _NOISE_WRAPPABLE_BASE_MODELS
from utils.paths import data_path

DATASETS = ["carrabin", "yoo", "soltani_numbers", "soltani_colors"]
BASE_PARAM_SETS = {
    "Mean": dict(),
    "LeakyIntegrator": dict(gamma=0.5),
    "PrimacyRecency": dict(eps_p=0.3, eps_r=0.5),
    "RL_lambda": dict(alpha_0=0.9, lambda_=0.6),
}
N_SIMS = 10
N_SIMS_LARGE = 4000       # for the mean/SD sanity bounds, not exact equality
SIGMA_RESP_TEST = 0.06
TOL = 1e-9
MEAN_TOL = 0.01           # ~3 SE at N_SIMS_LARGE for typical response ranges
SD_REL_TOL = 0.15         # relative tolerance on the empirical SD vs requested

failures = []


def check(dataset, model_type, worst, label):
    status = "OK" if worst < TOL else "FAIL"
    if worst >= TOL:
        failures.append((dataset, model_type, label, worst))
    print(f"  {dataset:16s} {model_type:22s} {label:26s} {worst:.2e}  {status}")


print("=== add_noise vs run() ===")
for dataset in DATASETS:
    df = pd.read_pickle(data_path(f"{dataset}.pkl"))
    pid = int(df["pid"].iloc[0])
    for base_model in sorted(_NOISE_WRAPPABLE_BASE_MODELS):
        try:
            mm._validate_model_dataset(base_model, dataset)
        except ValueError:
            continue

        base_params = dict(model_type=base_model, dataset=dataset, pid=pid,
                           save=False, **BASE_PARAM_SETS[base_model])
        mu = (mm.run(base_params).sort_values(["trial", "observation"])
              ["response"].to_numpy(float))

        # (a) sigma=0 reduces exactly to run()'s deterministic output.
        zero = mm.add_noise(base_params, 5, sigma_resp=0.0)
        worst = float(np.abs(zero - mu[np.newaxis, :]).max())
        check(dataset, base_model, worst, "sigma=0 == run()")

        # (b)+(c) mean and SD sanity bounds at a fitting-time-plausible sigma.
        # EXCLUDE rows where mu sits near the +-1 boundary: clip(mu + noise, -1, 1)
        # is CORRECTLY biased there (the same clipping run() and every other model
        # in this codebase applies), so a naive mean-vs-mu check fails on those
        # rows for a reason that is not a bug. Confirmed directly on
        # soltani_colors' Mean model, which legitimately reaches the boundary on
        # 15.6% of rows (colors' Mean output can be exactly +-1): the mean gap is
        # 0.0035 away from the boundary and 0.0257 on boundary rows, and the
        # boundary-row gap does not shrink with more sims -- it is bias, not MC
        # noise. So this check is only meaningful, and only checked, away from
        # the boundary.
        ens = mm.add_noise(base_params, N_SIMS_LARGE, sigma_resp=SIGMA_RESP_TEST)
        away = np.abs(mu) <= 0.9
        if away.any():
            mean_gap = float(np.abs(ens.mean(axis=0)[away] - mu[away]).max())
        else:
            mean_gap = 0.0   # every row is boundary-adjacent; nothing to check
        sd_gap = float(np.abs(ens.std(axis=0).mean() - SIGMA_RESP_TEST))
        m_status = "OK" if mean_gap < MEAN_TOL else "FAIL"
        s_status = "OK" if sd_gap < SD_REL_TOL * SIGMA_RESP_TEST else "FAIL"
        if m_status == "FAIL":
            failures.append((dataset, base_model, "mean close to mu", mean_gap))
        if s_status == "FAIL":
            failures.append((dataset, base_model, "SD close to sigma_resp", sd_gap))
        print(f"  {dataset:16s} {base_model:22s} {'mean close to mu':26s} "
              f"{mean_gap:.4f}  {m_status}")
        print(f"  {dataset:16s} {base_model:22s} {'SD close to sigma_resp':26s} "
              f"{sd_gap:.4f}  {s_status}")

        # (d) bare name and "<model>_resp_noise" suffix must behave identically.
        suffixed_params = {**base_params, "model_type": f"{base_model}_resp_noise"}
        rng_check_a = mm.add_noise({**base_params, "seed": 0}, 3, sigma_resp=0.05)
        rng_check_b = mm.add_noise({**suffixed_params, "seed": 0}, 3, sigma_resp=0.05)
        worst = float(np.abs(rng_check_a - rng_check_b).max())
        check(dataset, base_model, worst, "bare name == _resp_noise name")

print()
if failures:
    print(f"{len(failures)} FAILURE(S). Do not trust any --loss nll fit "
          f"until fixed:")
    for dataset, model_type, label, val in failures:
        print(f"  {dataset} / {model_type} / {label}: {val:.4e}")
    sys.exit(1)
print("all combinations verified.")
