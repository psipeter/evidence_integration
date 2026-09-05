# archive/fitting/archive_model_params_retired.py — retired model param specs
#
# Retired (this session): NoisyCounting, NoisyRL_lambda (state-noise models
# phased out of active analysis -- see docs/DECISIONS.md), and MLE_PARAMS/
# NEF_N_NEURONS_VALUES (the MLE fitting pipeline, also retired). Removed
# from the live fitting/model_params.py's MODEL_PARAMS dicts. If restoring,
# merge these back into MODEL_PARAMS[dataset] for each of carrabin/yoo/
# soltani_numbers/soltani_colors as applicable, and restore
# archive/models/archive_math_models_noise.py into models/math_models.py
# first (these param specs are useless without the model implementations).

# NoisyRL_lambda -- identical spec across all 4 active datasets
# (carrabin/yoo/soltani_numbers/soltani_colors):
NOISY_RL_LAMBDA_SPEC = {
    "alpha_0": (0.01, 1.0, 0.001),
    "lambda_": (0.01, 1.0, 0.001),
    # sigma_resp REMOVED -- this was STATE-NOISE-ONLY by design, compared
    # against RL_lambda_resp_noise (i.i.d. response noise, still active)
    # at equal parameter count. Floor 0.001 is a technical floor only; NLL
    # found a genuine interior optimum on its own (NLL fell from 389 at
    # sigma=0.001 to -2.46 at the optimum and rose again beyond it).
    "sigma_state": (0.001, 2.0, 0.001),
}

# NoisyCounting -- carrabin only (RMSE search space)
NOISY_COUNTING_SPEC = {
    "mu": (0.001, 2.0, 0.001),
    "sigma_c": (0.001, 2.0, 0.001),
    "nu": (0.001, 2.0, 0.001),
}

# MLE_PARAMS -- narrower/more-realistic ranges for MLE fitting
# (sim_db / fitting/fit_mle.py, also retired -- archive/fitting/archive_fit_mle.py)
MLE_PARAMS: dict[str, dict[str, dict[str, object]]] = {
    "carrabin": {
        "NoisyCounting": {
            "mu":      (0.05,  0.40,  0.002),
            "sigma_c": (0.001, 0.30,  0.002),   # RMSE collapses to ~0; MLE recovers ~0.03-0.08
            "nu":      (0.001, 0.35,  0.002),
            "fixed": {},
        },
        "NEF": {
            "lambda_": (0.01, 1.0, 0.001),
            "alpha_0": (0.01, 1.0, 0.001),
            "n_neurons": "categorical",
            "fixed": {
                # was: {**{k: v for k, v in _NEF_FIXED.items() if k not in
                # ("n_neurons", "n_neurons_counting")}, "radius_c": 5}
                # -- rebuild from the live _NEF_FIXED in fitting/model_params.py
                # if restoring, rather than copying a frozen snapshot here.
            },
        },
    },
}

# Discrete n_neurons values for NEF MLE fitting.
NEF_N_NEURONS_VALUES: list[int] = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
