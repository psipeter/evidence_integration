"""
Single source of truth for all model parameters.

``_NEF_FIXED``: fixed architectural and timing parameters for all NEF models.
``MODEL_PARAMS``: per-dataset, per-model parameter search spaces (fitted ranges)
and fixed parameter dicts. Structure:
    dataset → model_type → param_name → (min, max, step)  [for fitted params]
                         → "fixed"    → dict               [for fixed params]
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
    "lmu_order": 24,
    "lmu_tau": 0.2,
    "lmu_n_obs_max": 30,
    "lmu_theta_mult": 1.1,
    "onset_detector_amp": 0.3,
    "radius_e": 1.5,
    "radius_v": 1.0,
    "counting": "integrator",
    "n_neurons": 200,
    "n_neurons_counting": 1000,
    "n_seeds": 1,
    "seed": 0,
    "pes_learning_rate": 1e-4,
}

_NEF_RANGES: dict[str, tuple] = {
    "lambda_": (0.01, 1.0, 0.001),
    "alpha_0": (0.01, 1.0, 0.001),
}

_NEF_RANGES_JIANG: dict[str, tuple] = {
    **_NEF_RANGES,
    "beta": (0.01, 30.0, 0.01),
}

MODEL_PARAMS: dict[str, dict[str, dict[str, object]]] = {
    "carrabin": {
        "Bayes": {},
        "NoisyCounting": {
            "mu": (0.001, 2.0, 0.001),
            "sigma_c": (0.001, 2.0, 0.001),
            "nu": (0.001, 2.0, 0.001),
        },
        "RL": {
            "alpha": (0.001, 1.0, 0.001),
        },
        "RL_lambda": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
        },
        "RL_lambda_offset": {
            "alpha_0": (0.01, 1.0, 0.01),
            "lambda_": (0.01, 1.0, 0.01),
        },
        "NEF_recurrent": {**_NEF_RANGES, "fixed": _NEF_FIXED},
        "NEF_synaptic": {**_NEF_RANGES, "fixed": _NEF_FIXED},
    },
    "jiang": {
        "Bayes": {"beta": (0.01, 30.0, 0.01)},
        "DeGroot": {
            "w_base": (0.0, 1.0, 0.01),
            "w1":     (0.0, 1.0, 0.01),
            "w2":     (0.0, 1.0, 0.01),
            "w3":     (0.0, 1.0, 0.01),
            "beta":   (0.01, 30.0, 0.01),
        },
        "RL": {
            "beta": (0.01, 30.0, 0.01),
            "alpha": (0.01, 1.0, 0.01),
        },
        "RL_lambda": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
            "beta": (0.01, 30.0, 0.01),
        },
        "RL_lambda_rd": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
            "beta": (0.01, 30.0, 0.01),
        },
        "NEF_recurrent": {**_NEF_RANGES_JIANG, "fixed": _NEF_FIXED},
        "NEF_synaptic": {**_NEF_RANGES_JIANG, "fixed": _NEF_FIXED},
    },
    "yoo": {
        "Mean": {},
        "RL": {
            "alpha": (0.001, 1.0, 0.001),
        },
        "RL_lambda": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
        },
        "ADM": {
            "phi": (0.001, 1.0, 0.001),
            "rho": (0.001, 1.0, 0.001),
        },
        "NEF_recurrent": {**_NEF_RANGES, "fixed": _NEF_FIXED},
        "NEF_synaptic": {**_NEF_RANGES, "fixed": _NEF_FIXED},
    },
    "usher": {
        "Mean": {},
        "RL": {
            "alpha": (0.001, 1.0, 0.001),
        },
        "RL_lambda": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
        },
        "PopulationCoding": {
            "sigma": (0.01, 1.0, 0.001),
            "n_neurons": (10, 500, 1),
        },
    },
}
