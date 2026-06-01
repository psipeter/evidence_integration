"""
models/RNN.py

Tiny GRU (2-5 hidden units) fitted per participant to carrabin behavioral data.

Used for two purposes:
  1. Fit to HUMAN data -> best empirical predictor of each participant's
     average behavior (used as the noise reference baseline).
  2. Fit to MODEL responses -> per-participant sigma estimate for each model,
     using the same instrument as for humans, enabling direct comparison.

Following Ger, Shahar & Shahar (2024, eLife).

Architecture
------------
Input at each observation t: [x_t, t]
  - x_t : binary input value {-1, +1}  (from carrabin.pkl, same for all sources)
  - t   : observation index (1-indexed float)
GRU hidden layer (n_hidden units, default 4)
Linear readout -> scalar predicted response

Training
--------
- Loss   : MSE between RNN transformed output and source response
- Output : RNN raw prediction clipped to [-1,1], then * t/(t+2) (Laplace)
- Optimiser: Adam (lr=1e-3)
- Early stopping on held-out validation fold (patience=300)
- CV     : k-fold over trials; reports mean held-out RMSE

Outputs
-------
All saved to data/runs/<run_folder>/ with a `source` column identifying
what the RNN was fitted to ("human", "RL_lambda", "NEF", etc.).

Per-pid intermediate files (deleted after collection):
  RNN_{source}_carrabin_{pid}.pkl  : dict with keys
      params, performance, sigma, rnn_responses

Combined files (written by --collect or fit_all_sources):
  RNN_carrabin_params.pkl       : all sources x all pids
  RNN_carrabin_performance.pkl  : all sources x all pids
  RNN_carrabin_sigma.pkl        : per-pid sigma (std of source - RNN)

Usage
-----
    # Fit to human data
    venv/bin/python models/RNN.py --source human --all_pids --run_folder carrabin

    # Fit to a specific model's responses
    venv/bin/python models/RNN.py --source RL_lambda --all_pids --run_folder carrabin

    # Fit all sources (human + all available models) in one pass
    venv/bin/python models/RNN.py --all_sources --run_folder carrabin

    # Collect per-pid files into combined outputs
    venv/bin/python models/RNN.py --collect --run_folder carrabin
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

DATASET = "carrabin"
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models whose response files are expected in the run folder.
# "human" is special-cased to read from carrabin.pkl directly.
ALL_SOURCES = [
    "human",
    "Mean",
    "LeakyIntegrator",
    "RL_lambda",
    "NoisyCounting",
    "PrimacyRecency",
    "NEF",
]


# ── Model ────────────────────────────────────────────────────────────────────

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
) -> pd.DataFrame:
    """Return a DataFrame with columns [pid, trial, observation, value, response].

    For source="human": reads from carrabin.pkl.
    For source=<model>: reads model responses from run_folder, merges value
    column from carrabin.pkl (input sequences are the same for all sources).
    """
    human = pd.read_pickle(data_path(f"{DATASET}.pkl"))
    human_pid = human[human["pid"] == pid].copy()

    if source == "human":
        return human_pid

    resp_path = run_folder / f"{source}_{DATASET}_responses.pkl"
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
    # Merge value (input) from human data — same sequences for all sources
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
    n_obs = int(pid_data["observation"].max())
    inputs_list, targets_list = [], []

    for trial in trials:
        td = pid_data[pid_data["trial"] == trial].sort_values("observation")
        if len(td) != n_obs:
            continue
        x_vals = td["value"].to_numpy(dtype=np.float32)
        t_vals = td["observation"].to_numpy(dtype=np.float32)
        resp   = td["response"].to_numpy(dtype=np.float32)
        inputs_list.append(np.stack([x_vals, t_vals], axis=1))
        targets_list.append(resp)

    return (
        torch.tensor(np.stack(inputs_list),  dtype=torch.float32),
        torch.tensor(np.stack(targets_list), dtype=torch.float32),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    model: TinyGRU,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    lr: float = 1e-3,
    max_epochs: int = 5000,
    patience: int = 300,
    min_delta: float = 1e-5,
    device: torch.device | None = None,
) -> tuple[TinyGRU, int, float]:
    """Train with early stopping; returns model, best_epoch, best_val_loss."""
    if device is None:
        device = DEVICE
    model       = model.to(device)
    inputs      = inputs.to(device)
    targets     = targets.to(device)
    val_inputs  = val_inputs.to(device)
    val_targets = val_targets.to(device)

    # Laplace shrinkage: clamp(pred) * t/(t+2)
    shrink     = inputs[:, :, 1]     / (inputs[:, :, 1]     + 2.0)
    shrink_val = val_inputs[:, :, 1] / (val_inputs[:, :, 1] + 2.0)

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
    max_epochs: int = 5000,
    patience: int = 300,
    seed: int = 42,
) -> tuple[float, float, int]:
    """K-fold CV over trials; returns (mean_rmse, std_rmse, mean_best_epoch)."""
    rng    = np.random.RandomState(seed)
    trials = sorted(pid_data["trial"].unique())
    rng.shuffle(trials)
    folds  = np.array_split(trials, k)
    fold_losses, fold_epochs = [], []

    for i, val_trials in enumerate(folds):
        train_trials = [t for j, f in enumerate(folds) for t in f if j != i]
        tr_inp, tr_tgt = build_trial_tensors(pid_data, train_trials)
        va_inp, va_tgt = build_trial_tensors(pid_data, list(val_trials))
        torch.manual_seed(seed + i)
        m, n_ep, val_loss = train_model(
            TinyGRU(n_hidden), tr_inp, tr_tgt, va_inp, va_tgt,
            lr=lr, max_epochs=max_epochs, patience=patience,
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
) -> pd.DataFrame:
    """Run final model; return DataFrame with transformed RNN predictions."""
    model.eval()
    model = model.cpu()
    trials = sorted(pid_data["trial"].unique())
    inp, _ = build_trial_tensors(pid_data, trials)
    with torch.no_grad():
        preds, _ = model(inp)
    preds_np = preds.numpy()
    pid  = int(pid_data["pid"].iloc[0])
    n_obs = int(pid_data["observation"].max())
    rows = []
    for ti, trial in enumerate(trials):
        for oi in range(n_obs):
            rows.append({
                "pid":         pid,
                "trial":       int(trial),
                "observation": oi + 1,
                "response":    float(np.clip(preds_np[ti, oi], -1.0, 1.0)),
            })
    df = pd.DataFrame(rows)
    return apply_carrabin_transform(df, DATASET)


def compute_sigma(
    source_data: pd.DataFrame,
    rnn_responses: pd.DataFrame,
) -> float:
    """Per-participant sigma: std(source_response - rnn_response)."""
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
    run_dir  = resolve_run_folder(run_folder)
    pid_data = load_pid_data(pid, source, run_dir)

    if pid_data.empty:
        raise ValueError(f"No data for pid={pid} source={source!r}")

    if verbose:
        print(f"  source={source:<18} pid={pid}: "
              f"{pid_data['trial'].nunique()} trials")

    # ── CV ────────────────────────────────────────────────────────────────────
    cv_rmse, cv_std, mean_epochs = cross_validate(
        pid_data, n_hidden=n_hidden, lr=lr, k=k,
        max_epochs=max_epochs, patience=patience, seed=seed,
    )

    # ── Final model ───────────────────────────────────────────────────────────
    rng      = np.random.RandomState(seed)
    trials   = sorted(pid_data["trial"].unique())
    shuffled = rng.permutation(trials)
    n_val    = max(1, int(0.1 * len(trials)))
    va_t     = list(shuffled[:n_val])
    tr_t     = list(shuffled[n_val:])

    tr_inp, tr_tgt = build_trial_tensors(pid_data, tr_t)
    va_inp, va_tgt = build_trial_tensors(pid_data, va_t)
    torch.manual_seed(seed)
    final_model, final_epochs, final_val_loss = train_model(
        TinyGRU(n_hidden), tr_inp, tr_tgt, va_inp, va_tgt,
        lr=lr, max_epochs=max_epochs, patience=patience,
    )

    # ── Responses + sigma ─────────────────────────────────────────────────────
    rnn_resp = generate_rnn_responses(final_model, pid_data)
    sigma    = compute_sigma(pid_data, rnn_resp)

    if verbose:
        print(f"    CV RMSE={cv_rmse:.4f} ± {cv_std:.4f}  "
              f"epochs={mean_epochs}  sigma={sigma:.4f}")

    params_df = pd.DataFrame([{
        "source":         source,
        "pid":            pid,
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
        "loss":     cv_rmse,
        "n_hidden": n_hidden,
        "n_epochs": mean_epochs,
    }])
    sigma_df = pd.DataFrame([{
        "source": source,
        "pid":    pid,
        "sigma":  sigma,
    }])

    result = {
        "params":       params_df,
        "performance":  perf_df,
        "sigma":        sigma_df,
        "rnn_responses": rnn_resp,
    }

    # Save intermediate per-(source, pid) file
    out_path = run_dir / f"RNN_sigma_{source}_{DATASET}_{pid}.pkl"
    pd.to_pickle(result, out_path)
    return result


# ── Collection ────────────────────────────────────────────────────────────────

def collect(run_folder: str | Path) -> None:
    """Concatenate per-(source,pid) pkl files into three combined files."""
    run_dir = resolve_run_folder(run_folder)
    files   = sorted(run_dir.glob(f"RNN_sigma_*_{DATASET}_*.pkl"))

    if not files:
        print("No RNN_* files found.")
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
        out = run_dir / f"RNN_sigma_{DATASET}_{name}.pkl"
        combined.to_pickle(out)
        n_src = combined["source"].nunique()
        n_pid = combined["pid"].nunique()
        print(f"Saved {out.name}  "
              f"({n_src} sources × {n_pid} pids = {len(combined)} rows)")

    # Also rebuild human-only performance/params/responses for figure compatibility.
    # These files must reflect only the human-source RNN fits (not all sources).
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
        out = run_dir / f"RNN_{DATASET}_{name}.pkl"
        human_df.to_pickle(out)
        print(f"Saved {out.name}  (human-only, {len(human_df)} pids)")

    # Rebuild human-only responses
    human_resp_files = sorted(run_dir.glob(f"RNN_sigma_human_{DATASET}_*.pkl"))
    if human_resp_files:
        resp_parts_human = [
            pd.read_pickle(f)["rnn_responses"] for f in human_resp_files
        ]
        resp_combined = pd.concat(resp_parts_human, ignore_index=True)
        out = run_dir / f"RNN_{DATASET}_responses.pkl"
        resp_combined.to_pickle(out)
        print(f"Saved {out.name}  (human-only, {len(resp_combined)} rows)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiny GRU noise estimation for carrabin (human + models)"
    )
    parser.add_argument(
        "--source", type=str, default="human",
        help=f"Data source to fit RNN to: 'human' or a model name. "
             f"Available: {ALL_SOURCES}",
    )
    parser.add_argument(
        "--all_sources", action="store_true",
        help="Fit all sources (human + all models with response files present)",
    )
    parser.add_argument("--pid",        type=int,  default=None)
    parser.add_argument("--all_pids",   action="store_true")
    parser.add_argument("--collect",    action="store_true")
    parser.add_argument("--run_folder", type=str,  default="carrabin")
    parser.add_argument("--n_hidden",   type=int,  default=4)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--k",          type=int,  default=5)
    parser.add_argument("--max_epochs", type=int,  default=5000)
    parser.add_argument("--patience",   type=int,  default=300)
    parser.add_argument("--seed",       type=int,  default=42)
    args = parser.parse_args()

    run_dir = resolve_run_folder(args.run_folder)

    if args.collect:
        collect(args.run_folder)
        return

    # Determine sources
    if args.all_sources:
        sources = []
        for s in ALL_SOURCES:
            if s == "human":
                sources.append(s)
            elif (run_dir / f"{s}_{DATASET}_responses.pkl").exists():
                sources.append(s)
            else:
                print(f"  Skipping {s}: no response file found")
    else:
        sources = [args.source]

    # Determine pids
    human = pd.read_pickle(data_path(f"{DATASET}.pkl"))
    pids  = sorted(human["pid"].unique()) if args.all_pids else [args.pid]
    if not pids or pids == [None]:
        parser.error("Specify --pid <n> or --all_pids")

    fit_kwargs = dict(
        run_folder=args.run_folder,
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
