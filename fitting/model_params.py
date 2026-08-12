# NOTE: jiang/usher params in archive/fitting/archive_model_params.py
"""
Single source of truth for all model parameters.

``_NEF_FIXED``: fixed architectural and timing parameters for all NEF models.
``MODEL_PARAMS``: per-dataset, per-model parameter search spaces (fitted ranges)
and fixed parameter dicts. Structure:
    dataset → model_type → param_name → (min, max, step)  [for fitted params]
                         → "fixed"    → dict               [for fixed params]

``radius_c``: representational radius of the counting memory ensemble.
Set per-dataset: carrabin=5 (5 obs/trial), yoo=30 (30 obs/trial).
The counting simulation runs for radius_c observations, so neurons are
tuned to the exact count range needed for each task.
"""

from __future__ import annotations

_NEF_FIXED: dict[str, object] = {
    "t_obs": 1.5,
    "t_iti": 0.5,
    "dt": 0.001,
    "tau_probe": 0.01,
    "tau_ff": 0.01,
    "tau_fb": 0.2,
    "tau_error": 0.1,
    "T_error": 0.3,
    "tau_fast": 0.01,
    "tau_slow": 0.2,
    "onset_detector_amp": 0.3,
    "radius_e": 1.5,
    "radius_v": 1.0,
    "radius_c": 30,   # default; overridden per dataset below
    "n_neurons": 100,
    "n_neurons_counting": 100,
    "n_seeds": 1,
    "seed": 0,
    "pes_learning_rate": 1e-4,
}

_NEF_RANGES: dict[str, tuple] = {
    "lambda_": (0.01, 1.0, 0.001),
    "alpha_0": (0.01, 1.0, 0.001),
}

MODEL_PARAMS: dict[str, dict[str, dict[str, object]]] = {
    "carrabin": {
        "Mean": {},
        "NoisyCounting": {
            "mu": (0.001, 2.0, 0.001),
            "sigma_c": (0.001, 2.0, 0.001),
            "nu": (0.001, 2.0, 0.001),
        },
        "LeakyIntegrator": {
            "gamma": (0.001, 0.999, 0.001),
        },
        "PrimacyRecency": {
            "eps_p": (0.001, 1.0, 0.001),
            "eps_r": (0.001, 1.0, 0.001),
        },
        "RL": {
            "alpha": (0.001, 1.0, 0.001),
        },
        "RL_lambda": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
        },
        "NEF": {
            **_NEF_RANGES,
            "fixed": {**_NEF_FIXED, "radius_c": 5},  # 5 obs/trial
        },
    },
    "yoo": {
        "Mean": {},
        "LeakyIntegrator": {
            "gamma": (0.001, 0.999, 0.001),
        },
        "RL": {
            "alpha": (0.001, 1.0, 0.001),
        },
        "RL_lambda": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
        },
        "PrimacyRecency": {
            "eps_p": (0.001, 1.0, 0.001),
            "eps_r": (0.001, 1.0, 0.001),
        },
        "NEF": {
            **_NEF_RANGES,
            "fixed": {**_NEF_FIXED, "radius_c": 30, "n_neurons": 200, "n_neurons_counting": 1000},  # 30 obs/trial
        },
    },
    "soltani_numbers": {
        "Mean": {},
        "LeakyIntegrator": {
            "gamma": (0.001, 0.999, 0.001),
        },
        "PrimacyRecency": {
            "eps_p": (0.001, 1.0, 0.001),
            "eps_r": (0.001, 1.0, 0.001),
        },
        "RL_lambda": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
        },
        "NEF": {
            **_NEF_RANGES,
            "fixed": {**_NEF_FIXED, "radius_c": 15, "n_neurons": 200, "n_neurons_counting": 1000},  # 15 obs/trial
        },
    },
    "soltani_colors": {
        "Mean": {},
        "LeakyIntegrator": {
            "gamma": (0.001, 0.999, 0.001),
        },
        "PrimacyRecency": {
            "eps_p": (0.001, 1.0, 0.001),
            "eps_r": (0.001, 1.0, 0.001),
        },
        "RL_lambda": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
        },
        "NEF": {
            **_NEF_RANGES,
            "fixed": {**_NEF_FIXED, "radius_c": 15, "n_neurons": 200, "n_neurons_counting": 1000},  # 15 obs/trial
        },
    },
}

# Parameter ranges for MLE fitting (sim_db / fit_mle.py).
# Narrower and more realistic than MODEL_PARAMS ranges, since MLE is
# sensitive to extreme params (likelihood blows up for very wide distributions).
# Informed by RMSE-fitted distributions plus expected state-noise range.
MLE_PARAMS: dict[str, dict[str, dict[str, object]]] = {
    "carrabin": {
        "NoisyCounting": {
            "mu":      (0.05,  0.40,  0.002),   # expanded upper; finer step
            "sigma_c": (0.001, 0.30,  0.002),   # RMSE collapses to ~0; MLE recovers ~0.03-0.08
            "nu":      (0.001, 0.35,  0.002),   # expanded upper; pid1 nu=0.21 near old bound
            "fixed": {},
        },
        "NEF": {
            "lambda_": (0.01, 1.0, 0.001),
            "alpha_0": (0.01, 1.0, 0.001),
            # n_neurons is discrete — list signals CategoricalDistribution in fit_mle.py
            # n_neurons_counting is always set equal to n_neurons
            "n_neurons": "categorical",
            "fixed": {
                **{k: v for k, v in _NEF_FIXED.items()
                   if k not in ("n_neurons", "n_neurons_counting")},
                "radius_c": 5,
            },
        },
    },
}

# Discrete n_neurons values for NEF MLE fitting.
# Activity files must exist: data/counting_activities_n{N}_nc{N}_carrabin.pkl
# Generate locally then scp to cluster before submitting MLE jobs.
NEF_N_NEURONS_VALUES: list[int] = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]


# diederen model params archived in archive/misc/
