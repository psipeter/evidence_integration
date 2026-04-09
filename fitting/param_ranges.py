"""
Single source of truth for all model parameter search spaces used in Optuna.

Structure: ``dataset`` → ``model_type`` → ``param_name`` → ``(min, max, step)``
tuples passed to ``trial.suggest_float(name, low, high, step=step)``.
Parameter-free models map to an empty dict.
"""

MODEL_PARAMS: dict[str, dict[str, dict[str, tuple[float, float, float]]]] = {
    "carrabin": {
        "Bayes": {},
        "NoisyCounting": {},
        "RL": {
            "alpha": (0.001, 1.0, 0.001),
        },
    },
    "jiang": {
        "Bayes": {
            "beta": (0.01, 10.0, 0.01),
        },
        "DeGroot": {
            "beta": (0.01, 10.0, 0.01),
            "omega": (0.01, 10.0, 0.01),
        },
        "RL": {
            "beta": (0.01, 10.0, 0.01),
            "alpha": (0.01, 1.5, 0.01),
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
            "nu": (0.001, 0.5, 0.001),
        },
    },
}
