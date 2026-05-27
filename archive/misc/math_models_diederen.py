"""
Archived diederen-specific logic from models/math_models.py.
"""

import numpy as np
import pandas as pd

DIEDEREN_MODELS = frozenset({"Mean", "RL", "RL_lambda", "PearceHall"})


def run_diederen(params: dict, human_pid: pd.DataFrame, trial: int, observation: int) -> float:
    model_type = params["model_type"]
    subdata = human_pid.query("trial == @trial & observation <= @observation")
    values = subdata["value"].to_numpy()

    if model_type == "Mean":
        if len(values) == 0:
            return 0.0
        return float(np.mean(values))

    if model_type == "RL":
        alpha = float(params["alpha"])
        expectation = 0.0
        for value in values:
            expectation += alpha * (value - expectation)
            expectation = float(np.clip(expectation, -1.0, 1.0))
        return expectation

    if model_type == "RL_lambda":
        alpha_0 = float(params["alpha_0"])
        lambda_ = float(params["lambda_"])
        expectation = 0.0
        for n, value in enumerate(values, start=1):
            alpha = alpha_0 / (n ** lambda_)
            expectation += alpha * (value - expectation)
            expectation = float(np.clip(expectation, -1.0, 1.0))
        return expectation

    if model_type == "PearceHall":
        alpha_0 = float(params["alpha_0"])
        eta = float(params["eta"])
        expectation = 0.0
        alpha = alpha_0
        for value in values:
            delta = value - expectation
            expectation += alpha * delta
            expectation = float(np.clip(expectation, -1.0, 1.0))
            alpha = float(np.clip(eta * abs(delta) + (1.0 - eta) * alpha, 0.0, 2.0))
        return expectation

    raise AssertionError(f"unreachable model_type={model_type!r}")
