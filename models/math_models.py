"""
Mathematical (non-NEF) models of evidence integration.

Expectations are computed from empirical sequences in per-dataset pickle files
and collected into a single tabular format with model ``response`` values.
Ported and redesigned from
``get_expectations_carrabin``, ``get_expectations_jiang``, and
``get_expectations_yoo`` in ``observational-learning-social-networks/fit.py``.

**Datasets and model types**

- **carrabin:** ``Bayes`` (optimal), ``NoisyCounting`` (human-matching), ``RL`` (naive)
- **jiang:** ``Bayes`` (optimal), ``DeGroot`` (human-matching), ``RL`` (naive)
- **yoo:** ``Mean`` (optimal), ``ADM`` (human-matching), ``RL`` (naive)

**Unified interface**

Every model is run via ``run(params, save=False, trials=None)``. Required keys in
``params`` for all models:

- ``"model_type"`` (``str``): one of the strings above for the chosen dataset
- ``"dataset"`` (``str``): ``"carrabin"``, ``"jiang"``, or ``"yoo"``
- ``"pid"`` (``int``): participant id

Additional keys are model-specific (learning rates, noise scales, etc.). The
optional ``trials`` argument restricts execution to a subset of trial ids.
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


_CARRABIN_MODELS = frozenset({"Bayes", "NoisyCounting", "RL"})
_JIANG_MODELS = frozenset({"Bayes", "DeGroot", "RL"})
_YOO_MODELS = frozenset({"Mean", "ADM", "RL"})


def run(params: dict, save: bool = False, trials: list | None = None) -> pd.DataFrame:
    for key in ("model_type", "dataset", "pid"):
        if key not in params:
            raise KeyError(f"params must include {key!r}")

    model_type: str = params["model_type"]
    dataset: str = params["dataset"]
    pid: int = int(params["pid"])

    _validate_model_dataset(model_type, dataset)

    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
    human_pid = human.query("pid == @pid")
    if human_pid.empty:
        raise ValueError(f"No rows for pid={pid} in dataset {dataset!r}")
    if trials is not None:
        human_pid = human_pid[human_pid["trial"].isin(trials)]

    rows: list[dict] = []
    if dataset in ("carrabin", "yoo"):
        pairs = (
            human_pid[["trial", "observation"]]
            .drop_duplicates()
            .sort_values(["trial", "observation"])
        )
        for _, pr in pairs.iterrows():
            trial = int(pr["trial"])
            observation = int(pr["observation"])
            estimate = _run(params, human_pid, trial, observation)
            rows.append(
                {
                    "model_type": model_type,
                    "pid": pid,
                    "trial": trial,
                    "observation": observation,
                    "response": estimate,
                }
            )
    else:
        pairs = (
            human_pid[["trial", "stage"]]
            .drop_duplicates()
            .sort_values(["trial", "stage"])
        )
        for _, pr in pairs.iterrows():
            trial = int(pr["trial"])
            stage = int(pr["stage"])
            estimate = _run(params, human_pid, trial, stage)
            rows.append(
                {
                    "model_type": model_type,
                    "pid": pid,
                    "trial": trial,
                    "stage": stage,
                    "response": estimate,
                }
            )

    out = pd.DataFrame(rows)
    if save:
        fname = f"{model_type}_{dataset}_{pid}_responses.pkl"
        out.to_pickle(data_path(fname))
    return out


def _validate_model_dataset(model_type: str, dataset: str) -> None:
    if dataset == "carrabin":
        allowed = _CARRABIN_MODELS
    elif dataset == "jiang":
        allowed = _JIANG_MODELS
    elif dataset == "yoo":
        allowed = _YOO_MODELS
    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected 'carrabin', 'jiang', or 'yoo'"
        )
    if model_type not in allowed:
        raise ValueError(
            f"Model {model_type!r} is not valid for dataset {dataset!r}; "
            f"expected one of {sorted(allowed)}"
        )


def _run(params: dict, human_pid: pd.DataFrame, trial: int, step: int) -> float:
    dataset = params["dataset"]

    if dataset == "carrabin":
        return _run_carrabin(params, human_pid, trial, step)
    if dataset == "jiang":
        return _run_jiang(params, human_pid, trial, step)
    if dataset == "yoo":
        return _run_yoo(params, human_pid, trial, step)
    raise AssertionError("unreachable")


def _run_carrabin(
    params: dict, human_pid: pd.DataFrame, trial: int, observation: int
) -> float:
    model_type = params["model_type"]
    subdata = human_pid.query("trial == @trial & observation <= @observation")
    values = subdata["value"].to_numpy()
    t = len(values)
    n_R = np.sum((values + 1) / 2)

    if model_type == "Bayes":
        p_star = (n_R + 1) / (t + 2)
        expectation = 2 * p_star - 1
        return float(expectation)
    if model_type == "NoisyCounting":
        # Prat-Carrabin & Woodford (2024), Table 5 Line 12: Eq. 31 (cognitive
        # state) and Eq. 34 (response), on [-1, 1]. Default n_runs=1 is fast for
        # local testing; use n_runs>=20 when fitting on the cluster; set
        # params["n_runs"].
        mu = float(params["mu"])
        sigma_c = float(params["sigma_c"])
        nu = float(params["nu"])
        n_runs = int(params.get("n_runs", 1))
        if len(values) == 0:
            return 0.0
        run_responses: list[float] = []
        for run_idx in range(n_runs):
            seed = 100 * int(params["pid"]) + 1000 * trial + run_idx
            rng = np.random.RandomState(seed)
            r = 0.0
            p_hat = 0.0
            for x in values:
                xi = rng.normal(0.0, sigma_c)
                r = r + float(x) * mu + xi
                epsilon = rng.normal(0.0, nu)
                p_hat = p_hat + (r - p_hat) * float(np.exp(epsilon))
                p_hat = float(np.clip(p_hat, -1.0, 1.0))
            run_responses.append(p_hat)
        return float(np.mean(run_responses))
    if model_type == "RL":
        expectation = 0.0
        for value in values:
            error = value - expectation
            expectation += params["alpha"] * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    raise AssertionError("unreachable")


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
        omega = params["omega"]
        weights = 1.0 + omega * rds
        wsum = np.sum(weights)
        if wsum == 0:
            return 0.0
        expectation = float(np.dot(weights, values) / wsum)
        return float(np.clip(expectation, -1, 1))
    if model_type == "RL":
        alpha = params["alpha"]
        weight = alpha
        expectation = 0.0
        for value in values:
            error = value - expectation
            expectation += weight * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    raise AssertionError("unreachable")


def _run_yoo(
    params: dict,
    human_pid: pd.DataFrame,
    trial: int,
    observation: int,
) -> float:
    model_type = params["model_type"]
    subdata = human_pid.query("trial == @trial & observation <= @observation")
    values = subdata["value"].to_numpy()

    if model_type == "Mean":
        return float(np.mean(values))
    if model_type == "RL":
        expectation = 0.0
        for value in values:
            error = value - expectation
            expectation += params["alpha"] * error
            expectation = float(np.clip(expectation, -1, 1))
        return expectation
    if model_type == "ADM":
        phi = params["phi"]
        rho = params["rho"]
        nu = params.get("nu", 0.01)  # fixed per Yoo et al.; not a free parameter
        n = len(values)
        weights = np.array(
            [
                (1.0 - (1.0 - phi ** (o + 1)) * (1.0 - rho ** (observation - o)))
                * (1.0 - nu)
                + nu
                for o in range(n)
            ],
            dtype=float,
        )
        return float(np.dot(weights, values) / np.sum(weights))
    raise AssertionError("unreachable")
