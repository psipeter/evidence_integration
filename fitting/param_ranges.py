"""
Single source of truth for all model parameter search spaces used in Optuna.

Structure: ``dataset`` → ``model_type`` → ``param_name`` → ``(min, max, step)``
tuples passed to ``trial.suggest_float(name, low, high, step=step)``.
Parameter-free models map to an empty dict.
"""

from __future__ import annotations

# -- NEF (recurrent + synaptic): shared fixed hyperparameters -----------------

_NEF_FIXED_BASE: dict[str, object] = {
    "t_obs": 0.5,
    "t_iti": 0.5,
    "dt": 0.001,
    "probe_dt": 0.01,
    "tau_probe": 0.1,
    "tau_ff": 0.02,
    "tau_fb": 0.1,
    "tau_fast": 0.01,
    "tau_slow": 0.2,
    "lmu_order": 24,
    "lmu_tau": 0.2,
    "lmu_n_obs_max": 30,
    "lmu_theta_mult": 1.1,
    "onset_detector_amp": 0.3,
    "radius_e": 1.0,
    "radius_v": 1.0,
    "counting": "integrator",
    "n_seeds": 1,
    "seed": 0,
    "n_neurons_counting": 1000,
}

_NEF_FIXED_SYNAPTIC: dict[str, object] = {
    **_NEF_FIXED_BASE,
    "pes_learning_rate": 1e-4,
}

# Carrabin: tighter lambda_/alpha_0, fixed-width n_neurons search.
_NEF_RANGES_CARRABIN: dict[str, tuple] = {
    "lambda_": (0.01, 1.0, 0.001),
    "alpha_0": (0.01, 1.0, 0.001),
    "n_neurons": (200.0, 200.0, 10.0),
}

# Jiang / yoo: wider lambda_, n_neurons range.
_NEF_RANGES_JIANG_YOO: dict[str, tuple] = {
    "lambda_": (0.0, 2.0, 0.001),
    "alpha_0": (0.01, 5.0, 0.001),
    "n_neurons": (50.0, 500.0, 1.0),
}


def _nef_models(ranges: dict[str, tuple]) -> dict[str, dict[str, object]]:
    """Build NEF_recurrent and NEF_synaptic entries with shared fixed blocks."""
    return {
        "NEF_recurrent": {**ranges, "fixed": dict(_NEF_FIXED_BASE)},
        "NEF_synaptic": {**ranges, "fixed": dict(_NEF_FIXED_SYNAPTIC)},
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
        **_nef_models(_NEF_RANGES_CARRABIN),
    },
    "jiang": {
        "Bayes": {
            "beta": (0.01, 15.0, 0.01),
        },
        "DeGroot": {
            "beta": (0.01, 15.0, 0.01),
            "omega": (0.01, 10.0, 0.01),
        },
        "RL": {
            "beta": (0.01, 15.0, 0.01),
            "alpha": (0.01, 1.5, 0.01),
        },
        **_nef_models(_NEF_RANGES_JIANG_YOO),
    },
    "yoo": {
        "Mean": {},
        "RL": {
            "alpha": (0.001, 1.0, 0.001),
        },
        "ADM": {
            "phi": (0.001, 1.0, 0.001),
            "rho": (0.001, 1.0, 0.001),
        },
        **_nef_models(_NEF_RANGES_JIANG_YOO),
    },
}
