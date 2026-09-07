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
    "pes_learning_rate": 4e-4,  # NEF_synaptic PES rate -- RMSE-minimizing value split
    # the difference between carrabin's optimum (5e-4, n_neurons=500/nc=500) and
    # soltani_numbers' (3e-4, n_neurons=500/nc=2000); see chat for the sweep. NEF
    # (recurrent) never reads this -- inert for the currently-active model.
}

_NEF_RANGES: dict[str, tuple] = {
    # Widened 0.01-1.0 -> 0.01-2.0 (see chat), matching RL_lambda's own
    # lambda_ range widening above -- NEF's counting subnetwork implements
    # the SAME alpha(t) = alpha_0/t^lambda equation (see docs/SCIENCE.md's
    # "Central model"), so it shows the identical bound-pinning signature:
    # colors 83% of pids' RMSE fits pinned exactly at the old ceiling
    # (median fitted lambda_ was EXACTLY 1.0, even more constrained than
    # RL_lambda's own ~80%), numbers 4%, carrabin ~14%, yoo 0% (max 0.91,
    # never approached the old bound) -- not a uniform artifact, same
    # per-dataset pattern as RL_lambda. Same caveat applies: this cannot
    # let NEF reproduce colors' genuine late-trial recency uptick (still
    # mathematically monotonic in k regardless of the bound) -- it only
    # lets Optuna find a better-fitting early/primacy decay rate. Not yet
    # refit under this wider bound as of this change (see chat) -- a real
    # NEF refit is expensive (spiking simulation, minutes-to-hours per
    # pid), unlike RL_lambda's near-instant math-model refit.
    "lambda_": (0.01, 2.0, 0.001),
    "alpha_0": (0.01, 1.0, 0.001),
}

# init_from_obs1 (initialize RL_lambda's `expectation`/LeakyIntegrator's `v`
# at observation 0's raw value instead of 0.0 -- see
# models.math_models._run_soltani_common's own comment at the init lines) is
# NOT wired into any MODEL_PARAMS entry -- defaults to False everywhere via
# math_models.py's own params.get("init_from_obs1", False). Turn it on
# per-fit via `fitting.fit`/`fitting.submit`'s own --init_from_obs1 true
# flag (deliberately NOT a permanent default here, even for
# soltani_colors/soltani_numbers's LeakyIntegrator/RL_lambda, so the
# canonical `rmse` run_folder stays a uniform baseline across all 4 tasks;
# a run with it on belongs in its own separate run_folder). Relevant only
# for soltani_colors/soltani_numbers -- both tasks' human data shows the
# overwhelming majority of pids set their very first response to (or
# within noise of) that trial's first observation rather than a partial
# step from a neutral prior (see chat), which Mean/PrimacyRecency already
# reproduce for free (both reduce to exactly value[0] at n=1) but
# RL_lambda/LeakyIntegrator do not without this flag. NOT relevant for
# carrabin/yoo -- carrabin's task asks for the underlying probability
# (naturally more conservative than value[0]) and yoo's own first
# observations are confounded by joystick catch-up dynamics that alpha<1
# actually helps capture; see chat for both.

# RL_lambda/RL_lambda_resp_noise's own lambda_ range, all 4 datasets: widened
# 0.01-1.0 -> 0.01-2.0 (see chat). Verified a large fraction of pids' RMSE
# fits pin exactly at the old upper bound of 1.0 (colors ~80%, numbers ~10-30%
# depending on init_from_obs1, carrabin ~14%, yoo 0% -- yoo's own fitted
# lambda_ never even approached 1.0, so this is not a uniform artifact).
# Widening it lets Optuna search past a bound that was silently constraining
# a real fraction of colors/numbers pids toward a slower decay than their own
# data wants. NOTE this does not, and cannot, let RL_lambda reproduce a
# genuine LATE-trial recency uptick (a real feature of colors' empirical
# weighting kernel, see chat) -- alpha_k = alpha_0/k^lambda_ is non-increasing
# in k for ANY lambda_ >= 0, so the implied weight on the most recent
# observation can never exceed an earlier one's, regardless of how high
# lambda_ is allowed to go. This bound widening only lets the model find a
# better-fitting EARLY/primacy decay rate, not a different functional shape.
# NEF's own _NEF_RANGES lambda_ (still 0.01-1.0 above) is a SEPARATE
# architecture-fitting range and is deliberately NOT touched by this change.

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
            "lambda_": (0.01, 2.0, 0.001),
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
            "lambda_": (0.01, 2.0, 0.001),
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
            "lambda_": (0.01, 2.0, 0.001),
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
            "lambda_": (0.01, 2.0, 0.001),
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
            "lambda_": (0.01, 2.0, 0.001),
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
            "lambda_": (0.01, 2.0, 0.001),
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
            "lambda_": (0.01, 2.0, 0.001),
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
            "lambda_": (0.01, 2.0, 0.001),
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
