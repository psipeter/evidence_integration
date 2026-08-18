"""
models/RNN.py

Tiny GRU (2-5 hidden units) fitted per participant to behavioral data.

Used for two purposes:
  1. Fit to HUMAN data -> best empirical predictor of each participant's
     average behavior (used as the noise reference baseline).
  2. Fit to MODEL responses -> per-participant sigma estimate for each model,
     using the same instrument as for humans, enabling direct comparison.

Following Ger, Shahar & Shahar (2024, eLife).

Architecture
------------
Input at each observation t: [x_t, t]
  - x_t : input value (binary {-1,+1} for carrabin; continuous for yoo)
  - t   : observation index (1-indexed float)
GRU hidden layer (n_hidden units, default 4)
Linear readout -> scalar predicted response

Training
--------
- Loss   : MSE between predicted and source response
- Output : RNN raw prediction clipped to [-1,1];
           for carrabin: also multiplied by t/(t+2) (Laplace shrinkage)
           for yoo: no shrinkage (responses are already on [-1,1])
- Optimiser: Adam (lr=1e-3)
- Early stopping on held-out validation fold (patience=300)
- CV     : k-fold over trials; reports mean held-out RMSE

Outputs
-------
All saved to data/runs/<run_folder>/ with a `source` column identifying
what the RNN was fitted to ("human", "RL_lambda", "NEF", etc.).

Per-pid intermediate files (deleted after collection):
  RNN_sigma_{source}_{dataset}_{pid}.pkl  : dict with keys
      params, performance, sigma, rnn_responses

Combined files (written by --collect):
  RNN_{dataset}_params.pkl       : human-only
  RNN_{dataset}_performance.pkl  : human-only
  RNN_{dataset}_responses.pkl    : human-only
  RNN_sigma_{dataset}_params.pkl     : all sources x all pids
  RNN_sigma_{dataset}_performance.pkl: all sources x all pids
  RNN_sigma_{dataset}_sigma.pkl      : per-pid sigma (std of source - RNN)

Usage
-----
    # Fit to human data
    venv/bin/python models/RNN.py --source human --all_pids \\
        --dataset carrabin --run_folder carrabin

    venv/bin/python models/RNN.py --source human --all_pids \\
        --dataset yoo --run_folder yoo

    # Fit to a specific model's responses
    venv/bin/python models/RNN.py --source NEF --all_pids \\
        --dataset yoo --run_folder yoo

    # Collect per-pid files into combined outputs
    venv/bin/python models/RNN.py --collect --dataset yoo --run_folder yoo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import data_path, resolve_run_folder
from utils.carrabin_transform import apply_carrabin_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets that require Laplace shrinkage on RNN output
SHRINKAGE_DATASETS = {"carrabin"}

# Models whose response files may appear in a run folder.
# "human" is special-cased to read from {dataset}.pkl directly.
ALL_SOURCES = [
    "human",
    "Mean",
    "LeakyIntegrator",
    "RL_lambda",
    "NoisyCounting",
    "PrimacyRecency",
    "NEF",
]


# ── Model ─────────────────────────────────────────────────────────────────────

class TinyGRU(nn.Module):
    """Single-layer GRU with linear readout.
    Input per timestep: [x_t, t]  (2 features)
    Output per timestep: scalar predicted response
    """

    def __init__(self, n_hidden: int = 4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=n_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.readout = nn.Linear(n_hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, h = self.gru(x, h0)
        return self.readout(out).squeeze(-1), h


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pid_data(
    pid: int,
    source: str,
    run_folder: Path,
    dataset: str = "carrabin",
) -> pd.DataFrame:
    """Return DataFrame with columns [pid, trial, observation, value, response].

    For source="human": reads from {dataset}.pkl.
    For source=<model>: reads model responses from run_folder, merges value
    column from {dataset}.pkl (input sequences are the same for all sources).
    """
    human     = pd.read_pickle(data_path(f"{dataset}.pkl"))
    human_pid = human[human["pid"] == pid].copy()

    if source == "human":
        return human_pid

    resp_path = run_folder / f"{source}_{dataset}_responses.pkl"
    if not resp_path.exists():
        raise FileNotFoundError(f"Response file not found: {resp_path}")

    model_resp = pd.read_pickle(resp_path)
    model_resp = model_resp[model_resp["pid"] == pid][
        ["pid", "trial", "observation", "response"]
    ].copy()

    if model_resp.empty:
        raise ValueError(
            f"No responses for pid={pid} in {resp_path.name}"
        )
    merged = model_resp.merge(
        human_pid[["trial", "observation", "value"]],
        on=["trial", "observation"],
    )
    if merged.empty:
        raise ValueError(
            f"Merge produced empty DataFrame for pid={pid} source={source!r}"
        )
    return merged


# ── Data preparation ──────────────────────────────────────────────────────────

def build_trial_tensors(
    pid_data: pd.DataFrame,
    trials: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build input (batch, seq, 2) and target (batch, seq) tensors."""
    pid_data = pid_data.sort_values(["trial", "observation"])
    # Number of observations per trial, taken as the MODAL ROW COUNT rather than
    # max(observation). The old version used max(observation), which silently
    # assumed 1-INDEXED observations: on soltani (0-indexed, 0..14) it computed
    # n_obs=14 while every trial has 15 rows, so the `len(td) != n_obs` guard
    # below dropped EVERY trial and the function failed on an empty stack. Row
    # counts are index-agnostic and work for carrabin (1..5), yoo (1..30) and
    # soltani (0..14) alike.
    counts = pid_data.groupby("trial").size()
    n_obs = int(counts.mode().iloc[0]) if len(counts) else 0
    inputs_list, targets_list, kept = [], [], []

    for trial in trials:
        td = pid_data[pid_data["trial"] == trial].sort_values("observation")
        if len(td) != n_obs:
            continue
        kept.append(int(trial))
        x_vals = td["value"].to_numpy(dtype=np.float32)
        t_vals = td["observation"].to_numpy(dtype=np.float32)
        resp   = td["response"].to_numpy(dtype=np.float32)
        inputs_list.append(np.stack([x_vals, t_vals], axis=1))
        targets_list.append(resp)

    if not inputs_list:
        raise ValueError(
            f"no usable trials: expected {n_obs} observations per trial, got "
            f"{sorted(set(counts.values))}")
    return (
        torch.tensor(np.stack(inputs_list),  dtype=torch.float32),
        torch.tensor(np.stack(targets_list), dtype=torch.float32),
        kept,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    model: TinyGRU,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    lr: float = 1e-3,
    max_epochs: int = 3000,
    patience: int = 300,
    min_delta: float = 1e-4,
    device: torch.device | None = None,
    use_shrinkage: bool = True,
) -> tuple[TinyGRU, int, float]:
    """Train with early stopping; returns model, best_epoch, best_val_loss.

    use_shrinkage: if True, applies Laplace t/(t+2) transform to predictions
    (carrabin only). For yoo, set False.
    """
    if device is None:
        device = DEVICE
    model       = model.to(device)
    inputs      = inputs.to(device)
    targets     = targets.to(device)
    val_inputs  = val_inputs.to(device)
    val_targets = val_targets.to(device)

    if use_shrinkage:
        shrink     = inputs[:, :, 1]     / (inputs[:, :, 1]     + 2.0)
        shrink_val = val_inputs[:, :, 1] / (val_inputs[:, :, 1] + 2.0)
    else:
        shrink     = torch.ones_like(inputs[:, :, 1])
        shrink_val = torch.ones_like(val_inputs[:, :, 1])

    optimiser = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state    = {k: v.clone() for k, v in model.state_dict().items()}
    patience_ctr  = 0
    best_epoch    = 0

    for epoch in range(max_epochs):
        model.train()
        optimiser.zero_grad()
        preds, _ = model(inputs)
        loss = criterion(torch.clamp(preds, -1.0, 1.0) * shrink, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        model.eval()
        with torch.no_grad():
            vp, _ = model(val_inputs)
            val_loss = float(
                criterion(torch.clamp(vp, -1.0, 1.0) * shrink_val, val_targets)
            )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch + 1
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_val_loss


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(
    pid_data: pd.DataFrame,
    n_hidden: int = 4,
    lr: float = 1e-3,
    k: int = 5,
    max_epochs: int = 3000,
    patience: int = 300,
    seed: int = 42,
    use_shrinkage: bool = True,
) -> tuple[float, float, int]:
    """K-fold CV over trials; returns (mean_rmse, std_rmse, mean_best_epoch)."""
    rng    = np.random.RandomState(seed)
    trials = sorted(pid_data["trial"].unique())
    rng.shuffle(trials)
    folds  = np.array_split(trials, k)
    fold_losses, fold_epochs = [], []

    for i, val_trials in enumerate(folds):
        train_trials = [t for j, f in enumerate(folds) for t in f if j != i]
        tr_inp, tr_tgt, _ = build_trial_tensors(pid_data, train_trials)
        va_inp, va_tgt, _ = build_trial_tensors(pid_data, list(val_trials))
        torch.manual_seed(seed + i)
        m, n_ep, val_loss = train_model(
            TinyGRU(n_hidden), tr_inp, tr_tgt, va_inp, va_tgt,
            lr=lr, max_epochs=max_epochs, patience=patience,
            use_shrinkage=use_shrinkage,
        )
        fold_losses.append(np.sqrt(val_loss))
        fold_epochs.append(n_ep)

    return (
        float(np.mean(fold_losses)),
        float(np.std(fold_losses)),
        int(np.mean(fold_epochs)),
    )


# ── Response generation + sigma ───────────────────────────────────────────────

def generate_rnn_responses(
    model: TinyGRU,
    pid_data: pd.DataFrame,
    dataset: str = "carrabin",
) -> pd.DataFrame:
    """Run final model; return DataFrame with RNN predictions.

    Applies Laplace shrinkage for carrabin; plain clip for yoo.
    """
    model.eval()
    model = model.cpu()
    trials = sorted(pid_data["trial"].unique())
    inp, _, trials = build_trial_tensors(pid_data, trials)
    with torch.no_grad():
        preds, _ = model(inp)
    preds_np = preds.numpy()
    pid = int(pid_data["pid"].iloc[0])
    rows = []
    for ti, trial in enumerate(trials):
        # Use that trial's OWN observation labels rather than range(n_obs)+1 --
        # the old version hardcoded 1-indexing and would mislabel every soltani
        # row (and silently drop observation 0) when merged back onto the source.
        obs_labels = (pid_data[pid_data["trial"] == trial]
                      .sort_values("observation")["observation"].to_numpy())
        for oi, obs in enumerate(obs_labels):
            rows.append({
                "pid":         pid,
                "trial":       int(trial),
                "observation": int(obs),
                "response":    float(np.clip(preds_np[ti, oi], -1.0, 1.0)),
            })
    df = pd.DataFrame(rows)
    if dataset in SHRINKAGE_DATASETS:
        df = apply_carrabin_transform(df, dataset)
    return df


def cross_validated_predictions(
    pid_data: pd.DataFrame,
    dataset: str = "carrabin",
    n_hidden: int = 4,
    lr: float = 1e-3,
    k: int = 5,
    max_epochs: int = 3000,
    patience: int = 300,
    seed: int = 42,
    use_shrinkage: bool = True,
) -> pd.DataFrame:
    """OUT-OF-FOLD RNN predictions covering every observation, as the conditional
    mean estimate mu_hat.

    K-fold over TRIALS; each fold's model predicts only the trials it did not
    train on, and the folds are stitched back together. So every row gets a
    prediction from a model that never saw it, and NOTHING is discarded -- there
    is no coverage/validity tradeoff, only a reorganisation of which fit produces
    which prediction.

    WHY NOT the in-sample predictions from generate_rnn_responses (which is what
    fit() saves, and what compute_sigma has historically consumed):

      - For a SIGMA metric, in-sample residuals are systematically too small,
        because the fit absorbs part of the noise. A TinyGRU with n_hidden=4 has
        ~101 parameters against ~480 observations per pid (p/n ~ 0.21, so sigma
        deflated ~11%); n_hidden=2 gives ~39 parameters (p/n ~ 0.08, ~4%). Early
        stopping lowers the effective dof below the raw count, so treat those as
        upper bounds -- but the DIRECTION is guaranteed. Worse, the absorbed
        fraction need not be uniform across pids: a high-noise participant offers
        more noise to absorb, so the spread of sigma ACROSS pids can be
        compressed, which is exactly the measurement individual-differences claims
        rest on.
      - For a DENOISED TARGET in a distributional loss, in-sample predictions
        defeat the purpose entirely. The point of scoring against mu_hat instead
        of the observed y is that mu_hat is noise-free; an in-sample mu_hat has
        absorbed part of that very noise realisation and sits closer to y than the
        true conditional mean does. In the limit of a perfect fit mu_hat == y and
        the denoised target IS the raw target -- the benefit shrinks as the fit
        improves, which is self-defeating.

    KNOWN RESIDUAL LEAKAGE, worth stating rather than hiding: train_model uses the
    held-out fold for EARLY STOPPING as well, so the stopping epoch is chosen on
    the same data the predictions are made for. That makes these predictions
    mildly optimistic. Removing it needs a nested train/stop/test split, which
    costs a third partition of only 32 trials. Judged not worth it here, but it
    means sigma from this function is a slight LOWER bound rather than exact.
    """
    rng = np.random.RandomState(seed)
    trials = sorted(pid_data["trial"].unique())
    shuffled = list(trials)
    rng.shuffle(shuffled)
    folds = [list(map(int, f)) for f in np.array_split(shuffled, k)]

    pieces = []
    for i, val_trials in enumerate(folds):
        train_trials = [t for j, f in enumerate(folds) for t in f if j != i]
        tr_inp, tr_tgt, _ = build_trial_tensors(pid_data, train_trials)
        va_inp, va_tgt, _ = build_trial_tensors(pid_data, val_trials)
        torch.manual_seed(seed + i)
        model, _, _ = train_model(
            TinyGRU(n_hidden), tr_inp, tr_tgt, va_inp, va_tgt,
            lr=lr, max_epochs=max_epochs, patience=patience,
            use_shrinkage=use_shrinkage,
        )
        # predict ONLY this fold's held-out trials
        held = pid_data[pid_data["trial"].isin(val_trials)]
        pieces.append(generate_rnn_responses(model, held, dataset=dataset))

    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(["trial", "observation"]).reset_index(drop=True)


def compute_sigma(
    source_data: pd.DataFrame,
    rnn_responses: pd.DataFrame,
) -> float:
    """Per-participant sigma: std(source_response - rnn_response).

    Pass OUT-OF-FOLD predictions from cross_validated_predictions() -- see that
    function for why in-sample predictions deflate sigma and compress its spread
    across participants.
    """
    merged = source_data.merge(
        rnn_responses[["pid", "trial", "observation", "response"]],
        on=["pid", "trial", "observation"],
        suffixes=("_src", "_rnn"),
    )
    return float((merged["response_src"] - merged["response_rnn"]).std())


# ── Main fit function ─────────────────────────────────────────────────────────

def fit(
    pid: int,
    source: str,
    run_folder: str | Path,
    dataset: str = "carrabin",
    n_hidden: int = 4,
    lr: float = 1e-3,
    k: int = 5,
    max_epochs: int = 5000,
    patience: int = 300,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Fit TinyGRU for one (pid, source) pair.

    Returns dict with keys: params, performance, sigma, rnn_responses.
    Also saves an intermediate pkl to run_folder.
    """
    run_dir       = resolve_run_folder(run_folder)
    pid_data      = load_pid_data(pid, source, run_dir, dataset=dataset)
    use_shrinkage = dataset in SHRINKAGE_DATASETS

    if pid_data.empty:
        raise ValueError(f"No data for pid={pid} source={source!r}")

    if verbose:
        print(f"  source={source:<18} pid={pid}: "
              f"{pid_data['trial'].nunique()} trials  dataset={dataset}")

    # ── CV ────────────────────────────────────────────────────────────────────
    cv_rmse, cv_std, mean_epochs = cross_validate(
        pid_data, n_hidden=n_hidden, lr=lr, k=k,
        max_epochs=max_epochs, patience=patience, seed=seed,
        use_shrinkage=use_shrinkage,
    )

    # ── Final model ───────────────────────────────────────────────────────────
    final_max_epochs = max_epochs
    rng      = np.random.RandomState(seed)
    trials   = sorted(pid_data["trial"].unique())
    shuffled = rng.permutation(trials)
    n_val    = max(1, int(0.1 * len(trials)))
    va_t     = list(shuffled[:n_val])
    tr_t     = list(shuffled[n_val:])

    tr_inp, tr_tgt, _ = build_trial_tensors(pid_data, tr_t)
    va_inp, va_tgt, _ = build_trial_tensors(pid_data, va_t)
    torch.manual_seed(seed)
    final_model, final_epochs, final_val_loss = train_model(
        TinyGRU(n_hidden), tr_inp, tr_tgt, va_inp, va_tgt,
        lr=lr, max_epochs=final_max_epochs, patience=patience,
        use_shrinkage=use_shrinkage,
    )

    # ── Responses + sigma ─────────────────────────────────────────────────────
    rnn_resp = generate_rnn_responses(final_model, pid_data, dataset=dataset)
    sigma    = compute_sigma(pid_data, rnn_resp)

    if verbose:
        print(f"    CV RMSE={cv_rmse:.4f} ± {cv_std:.4f}  "
              f"epochs={mean_epochs}  sigma={sigma:.4f}")

    params_df = pd.DataFrame([{
        "source":         source,
        "pid":            pid,
        "dataset":        dataset,
        "n_hidden":       n_hidden,
        "lr":             lr,
        "k":              k,
        "cv_rmse":        cv_rmse,
        "cv_std":         cv_std,
        "mean_cv_epochs": mean_epochs,
        "final_epochs":   final_epochs,
        "seed":           seed,
    }])
    perf_df = pd.DataFrame([{
        "source":   source,
        "pid":      pid,
        "dataset":  dataset,
        "loss":     cv_rmse,
        "n_hidden": n_hidden,
        "n_epochs": mean_epochs,
    }])
    sigma_df = pd.DataFrame([{
        "source":  source,
        "pid":     pid,
        "dataset": dataset,
        "sigma":   sigma,
    }])

    result = {
        "params":        params_df,
        "performance":   perf_df,
        "sigma":         sigma_df,
        "rnn_responses": rnn_resp,
    }

    out_path = run_dir / f"RNN_sigma_{source}_{dataset}_{pid}.pkl"
    pd.to_pickle(result, out_path)
    return result


# ── Collection ────────────────────────────────────────────────────────────────

def collect(run_folder: str | Path, dataset: str = "carrabin") -> None:
    """Concatenate per-(source,pid) pkl files into combined files."""
    run_dir = resolve_run_folder(run_folder)
    files   = sorted(run_dir.glob(f"RNN_sigma_*_{dataset}_*.pkl"))

    if not files:
        print(f"No RNN_sigma_*_{dataset}_*.pkl files found in {run_dir}")
        return

    params_parts, perf_parts, sigma_parts = [], [], []
    for f in files:
        result = pd.read_pickle(f)
        params_parts.append(result["params"])
        perf_parts.append(result["performance"])
        sigma_parts.append(result["sigma"])

    for name, parts in [
        ("params",      params_parts),
        ("performance", perf_parts),
        ("sigma",       sigma_parts),
    ]:
        combined = pd.concat(parts, ignore_index=True)
        out = run_dir / f"RNN_sigma_{dataset}_{name}.pkl"
        combined.to_pickle(out)
        n_src = combined["source"].nunique()
        n_pid = combined["pid"].nunique()
        print(f"Saved {out.name}  "
              f"({n_src} sources × {n_pid} pids = {len(combined)} rows)")

    # Rebuild human-only files for figure compatibility
    for name, parts in [
        ("params",      params_parts),
        ("performance", perf_parts),
    ]:
        human_parts = [
            p for p in parts if str(p["source"].iloc[0]) == "human"
        ]
        if not human_parts:
            continue
        human_df = pd.concat(human_parts, ignore_index=True).copy()
        human_df["model_type"] = "RNN"
        human_df = human_df.drop(columns=["source"], errors="ignore")
        out = run_dir / f"RNN_{dataset}_{name}.pkl"
        human_df.to_pickle(out)
        print(f"Saved {out.name}  (human-only, {len(human_df)} pids)")

    human_resp_files = sorted(run_dir.glob(f"RNN_sigma_human_{dataset}_*.pkl"))
    if human_resp_files:
        resp_parts = [
            pd.read_pickle(f)["rnn_responses"] for f in human_resp_files
        ]
        resp_combined = pd.concat(resp_parts, ignore_index=True)
        out = run_dir / f"RNN_{dataset}_responses.pkl"
        resp_combined.to_pickle(out)
        print(f"Saved {out.name}  (human-only, {len(resp_combined)} rows)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiny GRU noise estimation for behavioral data"
    )
    parser.add_argument(
        "--source", type=str, default="human",
        help=f"Data source to fit RNN to. Available: {ALL_SOURCES}",
    )
    parser.add_argument(
        "--all_sources", action="store_true",
        help="Fit all sources (human + all models with response files present)",
    )
    parser.add_argument("--pid",        type=int,  default=None)
    parser.add_argument("--all_pids",   action="store_true")
    parser.add_argument("--collect",    action="store_true")
    parser.add_argument("--dataset",    type=str,  default="carrabin",
                        choices=["carrabin", "yoo"],
                        help="Dataset to use (affects shrinkage, file naming)")
    parser.add_argument("--run_folder", type=str,  default="carrabin")
    parser.add_argument("--n_hidden",   type=int,  default=4)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--k",          type=int,  default=5)
    parser.add_argument("--max_epochs", type=int,  default=3000)
    parser.add_argument("--patience",   type=int,  default=300)
    parser.add_argument("--seed",       type=int,  default=42)
    args = parser.parse_args()

    dataset = args.dataset
    run_dir = resolve_run_folder(args.run_folder)

    if args.collect:
        collect(args.run_folder, dataset=dataset)
        return

    # Determine sources
    if args.all_sources:
        sources = []
        for s in ALL_SOURCES:
            if s == "human":
                sources.append(s)
            elif (run_dir / f"{s}_{dataset}_responses.pkl").exists():
                sources.append(s)
            else:
                print(f"  Skipping {s}: no response file found")
    else:
        sources = [args.source]

    # Determine pids
    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
    pids  = sorted(human["pid"].unique()) if args.all_pids else [args.pid]
    if not pids or pids == [None]:
        parser.error("Specify --pid <n> or --all_pids")

    fit_kwargs = dict(
        run_folder=args.run_folder,
        dataset=dataset,
        n_hidden=args.n_hidden,
        lr=args.lr,
        k=args.k,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )

    for source in sources:
        print(f"\n=== source: {source} ===")
        for pid in pids:
            try:
                fit(pid=int(pid), source=source, **fit_kwargs)
            except (ValueError, FileNotFoundError) as e:
                print(f"  Skipping pid={pid}: {e}")

    print("\nJOB_COMPLETE")


if __name__ == "__main__":
    main()
