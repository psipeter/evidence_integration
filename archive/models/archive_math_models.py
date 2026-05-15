"""
Archived model classes for the jiang and usher tasks.
Extracted from models/math_models.py. See archive_readme.md.
"""

import itertools

import numpy as np
import pandas as pd

from utils.paths import data_path

# Cached Jiang supplementary data (Bayes model).
_JIANG_NETWORKS: np.ndarray | None = None
_ALL_SIGNALS: np.ndarray | None = None

# P(d_i matches true state) — 5:2 blue:yellow task ratio
_SIGNAL_ACCURACY = 5.0 / 7.0


def _bayes_load_networks() -> np.ndarray:
    """Load jiang_networks.npy, shape (7, 7, 43). Cached at module level."""
    global _JIANG_NETWORKS
    if _JIANG_NETWORKS is None:
        _JIANG_NETWORKS = np.load(data_path("jiang_networks.npy"))
    return _JIANG_NETWORKS


def _bayes_all_signals() -> np.ndarray:
    """Return all 2^7=128 possible signal vectors, shape (128, 7)."""
    global _ALL_SIGNALS
    if _ALL_SIGNALS is None:
        _ALL_SIGNALS = np.array(
            list(itertools.product([0, 1], repeat=7)), dtype=np.int8
        )
    return _ALL_SIGNALS


def _bayes_p_blue(
    agent: int,
    stage: int,
    d: np.ndarray,
    adj: np.ndarray,
) -> float:
    """
    Soft probability that ``agent`` chooses blue at ``stage`` given
    true signal vector ``d``.

    At stage 0: P(blue) = 5/7 if d[agent]==1 else 2/7.
    At stage s>0: computed via Bayesian update on neighbors'
    stage s-1 choice probabilities, starting from private signal.

    This is a pure function of d and network structure — does not use
    observed actions. Used to compute likelihoods in _bayes_posterior.
    """
    if stage == 0:
        return _SIGNAL_ACCURACY if int(d[agent]) == 1 else (1.0 - _SIGNAL_ACCURACY)

    # start from log-odds of private signal
    log_odds = np.log(_SIGNAL_ACCURACY / (1.0 - _SIGNAL_ACCURACY))
    log_belief = log_odds if int(d[agent]) == 1 else -log_odds

    # update on each neighbor's soft choice probability at stage-1
    neighbors = np.where(adj[agent] != 0)[0]
    for n in neighbors:
        p_n_blue = _bayes_p_blue(int(n), stage - 1, d, adj)
        # treat neighbor's choice as a soft signal
        # P(see blue choice | state=blue) = p_n_blue
        # P(see blue choice | state=yellow) = 1 - p_n_blue (inverted)
        # REVIEW: this assumes neighbor choice accuracy equals their
        # posterior probability, matching Jiang & Zhu Prob2.m logic
        if p_n_blue > 0.5:
            log_belief += np.log(p_n_blue / (1.0 - p_n_blue + 1e-10))
        elif p_n_blue < 0.5:
            log_belief -= np.log((1.0 - p_n_blue) / (p_n_blue + 1e-10))
        # if exactly 0.5, no update

    # convert log-odds to probability
    return float(1.0 / (1.0 + np.exp(-log_belief)))


def _bayes_posterior(
    focal: int,
    stage: int,
    adj: np.ndarray,
    observations: dict,
    prev_belief: float = 0.5,
) -> float:
    """
    Focal agent's P(blue) at end of ``stage`` by marginalizing over
    all 128 signal vectors.

    Prior: P(d) proportional to P(focal private signal | d[focal]).
    Likelihood: soft product over directly observed neighbor actions.
    For each observed (neighbor, s), the likelihood contribution is:
        _bayes_p_blue(neighbor, s-1, d, adj)   if observed action = blue
        1 - _bayes_p_blue(neighbor, s-1, d, adj) if observed action = yellow

    Stage 0 observation (focal private signal) contributes:
        5/7 if d[focal] matches observed, 2/7 otherwise.

    If den==0 (numerical underflow), return prev_belief per SN2.
    """
    all_signals = _bayes_all_signals()
    num = 0.0
    den = 0.0

    for d_row in all_signals:
        d = np.asarray(d_row, dtype=int).reshape(7)

        # prior: proportional to P(focal private signal | d[focal])
        focal_obs = observations.get((focal, 0), None)
        if focal_obs is None:
            prior = 1.0
        else:
            prior = (
                _SIGNAL_ACCURACY
                if int(d[focal]) == focal_obs
                else (1.0 - _SIGNAL_ACCURACY)
            )

        # likelihood: soft product over observed neighbor actions
        lik = 1.0
        for (k, s), obs_action in observations.items():
            if s > stage or k == focal:
                continue  # skip future observations and focal's own action
            if s == 0:
                # stage 0: neighbor's action based on private signal only
                p_blue = (
                    _SIGNAL_ACCURACY if int(d[k]) == 1 else (1.0 - _SIGNAL_ACCURACY)
                )
            else:
                # stage s: neighbor's action based on belief at end of stage s-1
                p_blue = _bayes_p_blue(int(k), s - 1, d, adj)
            if obs_action == 1:
                lik *= p_blue
            else:
                lik *= 1.0 - p_blue

        w = prior * lik
        num += w * float(d[focal])
        den += w

    if den < 1e-300:
        return prev_belief
    return float(num / den)


_JIANG_MODELS = frozenset({"Bayes", "DeGroot", "RL", "RL_lambda", "RL_lambda_rd"})

# Population-mean human serial-position OLS weights (usher); see EmpiricalWeights.
EMPIRICAL_WEIGHTS = np.array(
    [
        0.0638,
        0.0775,
        0.0723,
        0.0704,
        0.0656,
        0.0781,
        0.0754,
        0.0755,
        0.0684,
        0.1503,
    ],
    dtype=float,
)

_USHER_MODELS = frozenset(
    {
        "Mean",
        "EmpiricalWeights",
        "RL",
        "RL_lambda",
        "RL_lambda_boost",
        "PopulationCoding",
        "PoissonCoding",
    }
)


def _run_usher_population_coding(
    params: dict,
    human_pid: pd.DataFrame,
    trial: int,
    observation: int,
) -> float:
    """
    Brezis, Bronfman & Usher (2018)-style population coding: Gaussian tuning
    on ``[0, 1]``, accumulate firing across observations 1..n, decode running
    center of mass at each n.
    """
    # TODO: [usher] Revisit if stimulus scale or column semantics differ from ``value`` on [0, 1]
    subdata = human_pid.query("trial == @trial & observation <= @observation").sort_values(
        "observation"
    )
    values = subdata["value"].to_numpy(dtype=float)
    if len(values) == 0:
        return 0.5

    n_neurons = int(round(float(params["n_neurons"])))
    sigma = float(params["sigma"])
    if sigma <= 0.0:
        sigma = 1e-6

    T = np.linspace(0.0, 1.0, n_neurons)
    total_F = np.zeros(n_neurons, dtype=float)
    eps = 1e-10
    for x_t in values:
        diff = (float(x_t) - T) / sigma
        total_F += np.exp(-0.5 * diff * diff)

    denom = float(np.sum(total_F)) + eps
    com = float(np.sum(total_F * T) / denom)
    return float(np.clip(com, 0.0, 1.0))


def _run_usher_poisson_coding(
    params: dict,
    human_pid: pd.DataFrame,
    trial: int,
    observation: int,
) -> float:
    """
    Like ``PopulationCoding`` (Gaussian tuning on ``[0, 1]``, accumulate and decode
    running center of mass), but each observation's tuning-curve rates are realized
    as Poisson draws scaled by ``gain`` before accumulation.
    """
    subdata = human_pid.query("trial == @trial & observation <= @observation").sort_values(
        "observation"
    )
    if len(subdata) == 0:
        return 0.5

    n_neurons = int(round(float(params["n_neurons"])))
    sigma = float(params["sigma"])
    if sigma <= 0.0:
        sigma = 1e-6
    gain = float(params["gain"])
    if gain <= 0.0:
        gain = 1e-6

    T = np.linspace(0.0, 1.0, n_neurons)
    total_F = np.zeros(n_neurons, dtype=float)
    eps = 1e-10
    base_seed = int(params.get("seed", 0))

    for obs_idx, x_t in zip(
        subdata["observation"].astype(int),
        subdata["value"].astype(float),
    ):
        rng = np.random.default_rng(base_seed + int(trial) * 10000 + int(obs_idx))
        diff = (float(x_t) - T) / sigma
        F_i = np.exp(-0.5 * diff * diff)
        lam = F_i * gain
        F_i_noisy = rng.poisson(lam).astype(float) / gain
        total_F += F_i_noisy

    denom = float(np.sum(total_F)) + eps
    com = float(np.sum(total_F * T) / denom)
    return float(np.clip(com, 0.0, 1.0))


def _run_jiang(
    params: dict, human_pid: pd.DataFrame, trial: int, stage: int
) -> float:
    model_type = params["model_type"]

    if model_type == "Bayes":
        networks = _bayes_load_networks()
        network_id = int(
            human_pid.query("trial == @trial & stage == 0")["network"].iloc[0]
        )
        # network column in jiang.pkl is 1-indexed; subtract 1 for 0-indexed array
        adj = networks[:, :, network_id - 1]

        focal = int(
            human_pid.query("trial == @trial & stage == 0")["who"].iloc[0]
        ) - 1

        subdata = human_pid.query("trial == @trial & stage <= @stage")
        observations = {}
        for _, row in subdata.iterrows():
            agent = int(row["who"]) - 1
            s = int(row["stage"])
            action = 1 if row["value"] == 1 else 0
            observations[(agent, s)] = action

        # Compute posterior stage by stage, maintaining previous belief on
        # zero-posterior fallback (per Jiang & Zhu SN2).
        prev_belief = 0.5
        for s in range(int(stage) + 1):
            obs_s = {k: v for k, v in observations.items() if k[1] <= s}
            p_blue = _bayes_posterior(focal, s, adj, obs_s, prev_belief)
            prev_belief = p_blue

        expectation = 2 * prev_belief - 1
        return float(np.clip(expectation, -1, 1))

    subdata = human_pid.query("trial == @trial & stage <= @stage")
    values = subdata["value"].to_numpy(dtype=float)
    rds = subdata["rd"].to_numpy(dtype=float)

    if model_type == "DeGroot":
        w_base = float(params["w_base"])
        w1 = float(params["w1"])
        w2 = float(params["w2"])
        w3 = float(params["w3"])
        stage_w = {1: w1, 2: w2, 3: w3}

        private_stage = human_pid.query("trial == @trial & stage == 0")
        if private_stage.empty:
            return 0.0
        private_signal = float(private_stage["value"].iloc[0])

        # accumulate weighted observations across all stages
        numerator = w_base * float(private_signal)  # stage 0
        n_obs = 1
        expectation = float(np.clip(numerator / n_obs, -1.0, 1.0))

        for stage_i in sorted(subdata["stage"].unique()):
            if stage_i == 0:
                continue
            stage_data = subdata[subdata["stage"] == stage_i]
            w_rd = stage_w.get(int(stage_i), 0.0)
            for _, row in stage_data.iterrows():
                rd_k = float(row["true_rd"]) if not pd.isna(row["true_rd"]) else 0.0
                obs_k = float(row["value"])
                numerator += (w_base + w_rd * rd_k) * obs_k
                n_obs += 1
            expectation = float(np.clip(numerator / n_obs, -1.0, 1.0))

        return expectation
    if model_type == "RL":
        alpha = params["alpha"]
        weight = alpha
        expectation = 0.0
        for value in values:
            error = value - expectation
            expectation += weight * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    if model_type == "RL_lambda":
        alpha_0 = float(params["alpha_0"])
        lambda_ = float(params["lambda_"])
        expectation = 0.0
        for n, value in enumerate(values, start=1):
            alpha = alpha_0 / (n ** lambda_)
            error = value - expectation
            expectation += alpha * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    if model_type == "RL_lambda_rd":
        alpha_0 = float(params["alpha_0"])
        lambda_ = float(params["lambda_"])
        expectation = 0.0
        for n, (value, rd) in enumerate(zip(values, rds), start=1):
            alpha = alpha_0 / (n ** lambda_) + float(rd)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            error = value - expectation
            expectation += alpha * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    raise AssertionError("unreachable")


def _run_usher_empirical_weights(
    human_pid: pd.DataFrame,
    trial: int,
    observation: int,
) -> float:
    """EmpiricalWeights model (usher only)."""
    sub_sorted = human_pid.query("trial == @trial & observation <= @observation").sort_values(
        "observation"
    )
    vals = sub_sorted["value"].to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        return 0.5
    w = EMPIRICAL_WEIGHTS[:n]
    den = float(np.sum(w))
    if den <= 0.0:
        return 0.5
    num = float(np.sum(w * vals))
    return float(np.clip(num / den, 0.0, 1.0))


def _run_usher_rl_lambda_boost(
    params: dict,
    human_pid: pd.DataFrame,
    trial: int,
    observation: int,
) -> float:
    """RL_lambda_boost model (usher only)."""
    subdata = human_pid.query("trial == @trial & observation <= @observation")
    values = subdata["value"].to_numpy()
    alpha_0 = float(params["alpha_0"])
    lambda_ = float(params["lambda_"])
    beta = float(params["beta"])
    n_total = 10
    expectation = 0.0
    for n, value in enumerate(values, start=1):
        alpha = alpha_0 / (n ** lambda_) + (beta if n == n_total else 0.0)
        error = value - expectation
        expectation += alpha * error
        expectation = float(np.clip(expectation, -1, 1))
    return expectation
