"""
Single source of truth for all model parameter search spaces used in Optuna.

Structure: ``dataset`` → ``model_type`` → ``param_name`` → ``(min, max, step)``
tuples passed to ``trial.suggest_float(name, low, high, step=step)``.
Parameter-free models map to an empty dict.
"""

MODEL_PARAMS: dict[str, dict[str, dict[str, tuple[float, float, float]]]] = {
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
        "NEF_recurrent": {
            "alpha_0": (0.001, 2.0, 0.001),
            "lambda_": (0.001, 2.0, 0.001),
            "t_obs": (1.0, 2.0, 0.1),
            "t_iti": (1.0, 2.0, 0.1),
            "dt": (0.001, 0.005, 0.001),
            "probe_syn": (0.001, 0.02, 0.001),
            "probe_dt": (0.005, 0.02, 0.005),
        },
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
        "NEF_recurrent": {
            "alpha_0": (0.001, 2.0, 0.001),
            "lambda_": (0.001, 2.0, 0.001),
            "t_obs": (1.0, 2.0, 0.1),
            "t_iti": (1.0, 2.0, 0.1),
            "dt": (0.001, 0.005, 0.001),
            "probe_syn": (0.001, 0.02, 0.001),
            "probe_dt": (0.005, 0.02, 0.005),
        },
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
        "NEF_recurrent": {
            "alpha_0": (0.001, 2.0, 0.001),
            "lambda_": (0.001, 2.0, 0.001),
            "t_obs": (1.0, 2.0, 0.1),
            "t_iti": (1.0, 2.0, 0.1),
            "dt": (0.001, 0.005, 0.001),
            "probe_syn": (0.001, 0.02, 0.001),
            "probe_dt": (0.005, 0.02, 0.005),
        },
    },
}
