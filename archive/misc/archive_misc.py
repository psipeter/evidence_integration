"""
Miscellaneous archived code blocks from jiang/usher task branches.
Each block is labelled with its original source file and approximate
line number.
"""

# --- from models/NEF.py ~lines 426-430 ---
#         if pfull["dataset"] == "jiang":
#             alpha_bias = trial_data["rd"].to_numpy(dtype=float)
#         else:
#             # carrabin, yoo, usher: no rd column; alpha_bias channel unused (zeros)
#             alpha_bias = np.zeros(len(obs_values))

# --- from models/NEF.py ~lines 452-453 ---
#             if pfull["dataset"] == "jiang":
#                 entry["stage"] = int(row["stage"])

# --- from models/NEF.py ~line 472 (parse_args --dataset choices) ---
#         choices=("carrabin", "yoo", "jiang", "usher"),

# --- from fitting/fit.py DEFAULT_LOSS ---
# DEFAULT_LOSS: dict[str, str] = {
#     "carrabin": "response",
#     "jiang": "response",
#     "yoo": "response",
#     "usher": "response",
# }

# --- from fitting/fit.py _enqueue_warm_start docstring ~line 232 ---
#     Only applies to NEF models (carrabin, yoo, jiang, usher).

# --- from fitting/fit.py beta_outside_optuna in _suggest_params ~lines 119-120 ---
#         if beta_outside_optuna and param == "beta":
#             continue  # beta will be fitted separately

# --- from fitting/fit.py objective beta fitting ~lines 322-324 ---
#         if beta_outside_optuna and "beta" in MODEL_PARAMS[dataset][model_type]:
#             params["beta"] = _fit_beta(params, model_responses_full, human)
#             trial.set_user_attr("beta", params["beta"])

# --- from fitting/fit.py best_params beta ~lines 395-398 ---
#     if beta_outside_optuna:
#         best_params["beta"] = float(
#             best_trial.user_attrs.get("beta", float("nan"))
#         )

# --- from fitting/collect.py _collect_responses ~line 68 ---
#     for dataset in ("carrabin", "jiang", "yoo", "usher"):

# --- from fitting/collect.py _collect_activities ~line 81 ---
#     datasets = ("carrabin", "jiang", "yoo", "usher")

# --- from fitting/losses.py response_loss usher branch ~lines 327-334 ---
#     if dataset in ("carrabin", "yoo", "usher"):
#         if dataset == "usher":
#             # TODO: revisit usher masking if fold splits or observation indexing change
#             human_f = human[human["observation"] == 10]
#             model_f = model[model["observation"] == 10]

# --- from models/math_models.py run() stage-based loop ~lines 257-275 ---
#     else:
#         pairs = (
#             human_pid[["trial", "stage"]]
#             .drop_duplicates()
#             .sort_values(["trial", "stage"])
#         )
#         for _, pr in pairs.iterrows():
#             trial = int(pr["trial"])
#             stage = int(pr["stage"])
#             estimate = _run(params, human_pid, trial, stage)
#             rows.append(
#                 {
#                     "model_type": model_type,
#                     "pid": pid,
#                     "trial": trial,
#                     "stage": stage,
#                     "response": estimate,
#                 }
#             )

# Archived from models/NEF.py
# Used to inject per-observation rd (network degree) bias into the error
# ensemble for the jiang social network task. alpha_bias_array was set to
# zeros for all non-jiang datasets, making it a no-op. Removed when jiang
# was archived.
def _make_alpha_bias_input(obs_values: np.ndarray, params: dict) -> callable:
    """
    Outputs alpha_bias_array[step] during each observation, 0 during ITI.
    alpha_bias_array is all zeros so output is always 0.
    """
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    n_obs = len(obs_values)
    bias = np.array(params.get("alpha_bias_array", np.zeros(n_obs)), dtype=float)

    def fn(t: float) -> float:
        if t < t_iti:
            return 0.0
        step = int((t - t_iti) / t_step)
        phase = (t - t_iti) - step * t_step
        if step < n_obs and phase < t_obs:
            return float(bias[step])
        return 0.0

    return fn

# Archived from models/NEF.py build_network()
# Created a nengo.Node outputting alpha_bias_array values and connected
# it to net.error[0]. Was always zero for carrabin/yoo.
#   net.node_alpha_bias = nengo.Node(
#       _make_alpha_bias_input(obs_values, params),
#       label="node_alpha_bias",
#   )
#   nengo.Connection(
#       net.node_alpha_bias,
#       net.error[0],
#       synapse=None,
#       seed=seed,
#   )

# Archived from models/NEF.py run(), inside trial loop
# alpha_bias was always zeros for carrabin/yoo; injected rd values
# for jiang only.
#   alpha_bias = np.zeros(len(obs_values))
#   p = {**pfull, "alpha_bias_array": alpha_bias, "seed": trial_seed}

# Archived from fitting/fit.py objective(), folds record dict
# beta was the jiang softmax temperature parameter; always nan for
# carrabin/yoo.
#   "beta": float(trial.user_attrs.get("beta", float("nan"))),

# Archived from fitting/fit.py objective(), params exclusion list
# alpha_bias_array was excluded from the folds record because it is
# a large array injected at runtime, not a fitted scalar parameter.
#   "alpha_bias_array",

# Archived from scripts/dynamics_NEF.py main() ~lines 847-856
# Loaded per-trial rd (network degree) into alpha_bias_array for jiang;
# zeros for all other datasets. Passed to NEF error-ensemble bias channel.
#   rd_values = (
#       trial_data["rd"].to_numpy(dtype=float)
#       if args.dataset == "jiang"
#       else np.zeros(len(obs_values))
#   )
#   trial_seed = _trial_seed(int(base_params["seed"]), trial_db_id)
#   sim_params = {
#       **base_params,
#       "alpha_bias_array": rd_values,
#       "seed": trial_seed,
#   }

# Archived from utils/save_activities.py
# rd_values / alpha_bias_array injection was used for the jiang task only.

# Block A — rd_values construction and alpha_bias_array injection
#   rd_values = (
#       trial_data["rd"].to_numpy(dtype=float)
#       if params["dataset"] == "jiang"
#       else np.zeros(len(obs_values))
#   )
#   p = {**params, "alpha_bias_array": rd_values}

# Block B — jiang-specific fields added to out_row
#   if params["dataset"] == "jiang":
#       out_row["stage"] = int(row["stage"])
#       out_row["trial_obs_idx"] = n_idx
