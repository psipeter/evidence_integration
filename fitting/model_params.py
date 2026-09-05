# NOTE: jiang/usher params in archive/fitting/archive_model_params.py
# NOTE: NoisyCounting/NoisyRL_lambda params, and MLE_PARAMS/
# NEF_N_NEURONS_VALUES, retired -- see
# archive/fitting/archive_model_params_retired.py and docs/DECISIONS.md.
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

**NEF's n_neurons/n_neurons_counting bumped this session for the RMSE
pass** -- previously carrabin ran at 100/100 (via _NEF_FIXED's own
defaults, no override) and yoo/soltani_numbers/soltani_colors at 200/1000.
Now: yoo/soltani_numbers/soltani_colors at 500/2000; carrabin at 500/500
(NOT 2000) -- carrabin precomputes 200 trial-seeds vs yoo's 30/soltani's
40, and activity-file size scales with n_neurons_counting^2 * trial-seeds
(NOT with n_neurons at all), so nc=2000 would cost carrabin ~6.4GB against
~1-1.3GB for the other three at the same nc; nc=500 for carrabin reuses a
file already on disk from an earlier session, needing no new generation.
See docs/HISTORY.md for the full reasoning and the still-open gap that
real per-trial timing at these sizes has only been confirmed for carrabin
at the OLD 100/100 size, not at 500/500 or 500/2000, for any dataset.
There is NO CLI override for n_neurons/n_neurons_counting anywhere in
fitting.fit/fitting.submit -- this file's "fixed" dicts are the ONLY place
that controls what size a real submit runs at, so changing it here IS the
mechanism, not a convenience default.
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
        # Generic i.i.d.-response-noise wrapper (models.math_models.add_noise),
        # applied to a plain deterministic RL_lambda. One extra parameter, same
        # as NoisyRL_lambda's sigma_state -- the pairing that isolates whether
        # COMPOUNDING noise beats i.i.d. noise on NLL, rather than one model
        # simply having more parameters. `--loss nll` only; add_noise's ensemble
        # is undefined as a Gaussian likelihood without noise (see
        # fitting.losses.compute_nll's docstring).
        "RL_lambda_resp_noise": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        # Same add_noise() wrapper, applied to the other three deterministic
        # base models. Bounds mirror each base model's own entry exactly.
        "Mean_resp_noise": {
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        "LeakyIntegrator_resp_noise": {
            "gamma": (0.001, 0.999, 0.001),
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        "PrimacyRecency_resp_noise": {
            "eps_p": (0.001, 1.0, 0.001),
            "eps_r": (0.001, 1.0, 0.001),
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        "NEF": {
            **_NEF_RANGES,
            "fixed": {**_NEF_FIXED, "radius_c": 5, "n_neurons": 500, "n_neurons_counting": 500},  # 5 obs/trial; large-n RMSE pass (this session) -- nc=500 not 2000: carrabin precomputes 200 trial-seeds vs yoo's 30/soltani's 40, so nc=2000 would cost ~6.4GB here vs ~1-1.3GB there
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
        # Generic i.i.d.-response-noise wrapper (models.math_models.add_noise),
        # applied to a plain deterministic RL_lambda. One extra parameter, same
        # as NoisyRL_lambda's sigma_state -- the pairing that isolates whether
        # COMPOUNDING noise beats i.i.d. noise on NLL, rather than one model
        # simply having more parameters. `--loss nll` only; add_noise's ensemble
        # is undefined as a Gaussian likelihood without noise (see
        # fitting.losses.compute_nll's docstring).
        "RL_lambda_resp_noise": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        # Same add_noise() wrapper, applied to the other three deterministic
        # base models. Bounds mirror each base model's own entry exactly.
        "Mean_resp_noise": {
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        "LeakyIntegrator_resp_noise": {
            "gamma": (0.001, 0.999, 0.001),
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        "PrimacyRecency_resp_noise": {
            "eps_p": (0.001, 1.0, 0.001),
            "eps_r": (0.001, 1.0, 0.001),
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        "PrimacyRecency": {
            "eps_p": (0.001, 1.0, 0.001),
            "eps_r": (0.001, 1.0, 0.001),
        },
        "NEF": {
            **_NEF_RANGES,
            "fixed": {**_NEF_FIXED, "radius_c": 30, "n_neurons": 500, "n_neurons_counting": 2000},  # 30 obs/trial; large-n RMSE pass (this session)
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
        # Generic i.i.d.-response-noise wrapper (models.math_models.add_noise),
        # applied to a plain deterministic RL_lambda. One extra parameter, same
        # as NoisyRL_lambda's sigma_state -- the pairing that isolates whether
        # COMPOUNDING noise beats i.i.d. noise on NLL, rather than one model
        # simply having more parameters. `--loss nll` only; add_noise's ensemble
        # is undefined as a Gaussian likelihood without noise (see
        # fitting.losses.compute_nll's docstring).
        "RL_lambda_resp_noise": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        # Same add_noise() wrapper, applied to the other three deterministic
        # base models. Bounds mirror each base model's own entry exactly.
        "Mean_resp_noise": {
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        "LeakyIntegrator_resp_noise": {
            "gamma": (0.001, 0.999, 0.001),
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        "PrimacyRecency_resp_noise": {
            "eps_p": (0.001, 1.0, 0.001),
            "eps_r": (0.001, 1.0, 0.001),
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        "NEF": {
            **_NEF_RANGES,
            "fixed": {**_NEF_FIXED, "radius_c": 15, "n_neurons": 500, "n_neurons_counting": 2000},  # 15 obs/trial; large-n RMSE pass (this session)
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
        # Generic i.i.d.-response-noise wrapper (models.math_models.add_noise),
        # applied to a plain deterministic RL_lambda. One extra parameter, same
        # as NoisyRL_lambda's sigma_state -- the pairing that isolates whether
        # COMPOUNDING noise beats i.i.d. noise on NLL, rather than one model
        # simply having more parameters. `--loss nll` only; add_noise's ensemble
        # is undefined as a Gaussian likelihood without noise (see
        # fitting.losses.compute_nll's docstring).
        "RL_lambda_resp_noise": {
            "alpha_0": (0.01, 1.0, 0.001),
            "lambda_": (0.01, 1.0, 0.001),
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        # Same add_noise() wrapper, applied to the other three deterministic
        # base models. Bounds mirror each base model's own entry exactly.
        "Mean_resp_noise": {
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        "LeakyIntegrator_resp_noise": {
            "gamma": (0.001, 0.999, 0.001),
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        "PrimacyRecency_resp_noise": {
            "eps_p": (0.001, 1.0, 0.001),
            "eps_r": (0.001, 1.0, 0.001),
            "sigma_resp": (0.001, 2.0, 0.001),
        },
        "NEF": {
            **_NEF_RANGES,
            "fixed": {**_NEF_FIXED, "radius_c": 15, "n_neurons": 500, "n_neurons_counting": 2000},  # 15 obs/trial; large-n RMSE pass (this session)
        },
    },
}

# MLE_PARAMS and NEF_N_NEURONS_VALUES (MLE fitting via sim_db/fit_mle.py)
# retired along with NoisyCounting and the MLE pipeline itself -- see
# docs/DECISIONS.md and archive/fitting/archive_model_params_retired.py.


# diederen model params archived in archive/misc/
