#!/usr/bin/env python3
"""Build ``data/diederen.pkl`` from Diederen et al. raw MATLAB session files."""

from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

GROUP_MAP: dict[int, str] = {
    1001: "Pilot",
    1002: "Pilot",
    1003: "Pilot",
    1004: "Pilot",
    1005: "CTRL",
    1006: "CTRL",
    1007: "CTRL",
    1008: "CTRL",
    1009: "CTRL",
    1010: "CTRL",
    1011: "CTRL",
    1012: "CTRL",
    1013: "CTRL",
    1014: "CTRL",
    1015: "CTRL",
    1016: "CTRL",
    1017: "CTRL",
    1018: "CTRL",
    1019: "CTRL",
    1020: "CTRL",
    1021: "CTRL",
    1022: "CTRL",
    1023: "CTRL",
    1024: "CTRL",
    1025: "CTRL",
    1026: "CTRL",
    1027: "CTRL",
    1028: "CTRL",
    1029: "CTRL",
    1030: "CTRL",
    1031: "CTRL",
    1032: "CTRL",
    2001: "SUL",
    2004: "SUL",
    2005: "SUL",
    2006: "SUL",
    2007: "PCB",
    2008: "BRO",
    2009: "PCB",
    2010: "PCB",
    2011: "BRO",
    2013: "BRO",
    2017: "SUL",
    2018: "SUL",
    2019: "BRO",
    2020: "SUL",
    2021: "PCB",
    2024: "BRO",
    2026: "BRO",
    2027: "SUL",
    2029: "SUL",
    2035: "BRO",
    2036: "BRO",
    2038: "SUL",
    2040: "BRO",
    2044: "PCB",
    2046: "PCB",
    2047: "BRO",
    2048: "PCB",
    2049: "PCB",
    2051: "BRO",
    2052: "SUL",
    2054: "PCB",
    2056: "BRO",
    2057: "PCB",
    2059: "PCB",
    2060: "PCB",
    2063: "SUL",
    2064: "SUL",
    2065: "SUL",
    2067: "BRO",
    2068: "PCB",
    2069: "BRO",
    2070: "BRO",
    2072: "PCB",
    2073: "BRO",
    2074: "SUL",
    2077: "SUL",
    2080: "BRO",
    2083: "SUL",
    2085: "SUL",
    2086: "PCB",
    2087: "BRO",
    2092: "SUL",
    2095: "PCB",
    2096: "BRO",
    2097: "PCB",
    2098: "PCB",
    2099: "PCB",
    2062: "BRO",
    2034: "SUL",
    2081: "PCB",
}

# PIDs excluded from the final dataset at build time.
# Sources:
#   Pilots: 1001-1004 (not part of either published paper)
#   DA paper exclusions (Diederen et al. 2017 J. Neurosci.): 2034, 2062, 2081
#   2012: present in data with unknown group; likely renaming of excluded PID 2081
#         per Diederen's note "2081/2012" — excluded for safety
#   2002/2003: listed in exclusions but renumbered to 2018/2038 in the data;
#              those PIDs are retained under their new IDs
EXCLUDED_PIDS = {
    1001,
    1002,
    1003,
    1004,  # pilots
    2034,
    2062,
    2081,  # DA paper exclusions
    2012,  # ambiguous / likely duplicate of 2081
}

NO_RESPONSE_CODE = -6667

COLUMN_ORDER = [
    "pid",
    "group",
    "paper",
    "session",
    "trial",
    "observation",
    "trial_in_session",
    "distrib_index",
    "sd_value",
    "ev",
    "abs_trial_num",
    "catch_trial",
    "timeout",
    "missed",
    "value",
    "response",
    "pe",
    "rt_ms",
    "initial_scale",
]


def _paper_for_pid(pid: int, group: str) -> str:
    if group == "Pilot":
        return "pilot"
    if group == "CTRL":
        return "neuron_2016"
    return "jneurosci_2017"


def _session_mat_path(pid_dir: Path, pid: int, session: int) -> Path:
    return pid_dir / f"{pid}_{session}_PE2SD_fMRI.mat"


def _row_record(pid: int, row: np.ndarray, row_idx: int) -> dict:
    missed = row[0] == 0
    no_response = row[4] == NO_RESPONSE_CODE
    group = GROUP_MAP.get(pid, "unknown")

    if missed:
        return {
            "pid": pid,
            "group": group,
            "paper": _paper_for_pid(pid, group),
            "session": np.nan,
            "trial": np.nan,
            "observation": np.nan,
            "trial_in_session": row_idx + 1,
            "distrib_index": np.nan,
            "sd_value": np.nan,
            "ev": np.nan,
            "abs_trial_num": np.nan,
            "catch_trial": np.nan,
            "timeout": np.nan,
            "missed": True,
            "value": np.nan,
            "response": np.nan,
            "pe": np.nan,
            "rt_ms": np.nan,
            "initial_scale": np.nan,
        }

    response = np.nan if no_response else float(row[4])
    value = float(row[5])
    pe = np.nan if no_response else float(row[4]) - value

    return {
        "pid": pid,
        "group": group,
        "paper": _paper_for_pid(pid, group),
        "session": int(row[0]),
        "trial": np.nan,
        "observation": np.nan,
        "trial_in_session": row_idx + 1,
        "distrib_index": int(row[8]),
        "sd_value": int(row[10]),
        "ev": int(row[11]),
        "abs_trial_num": int(row[1]),
        "catch_trial": bool(row[2] == 1),
        "timeout": bool(row[3] == 2),
        "missed": False,
        "value": value,
        "response": response,
        "pe": pe,
        "rt_ms": float(row[7]),
        "initial_scale": float(row[12]),
    }


def _assign_trial_and_observation(pid_rows: list[dict]) -> None:
    seen: dict[tuple[int, int], int] = {}
    trial_counter = 1
    for r in sorted(
        pid_rows,
        key=lambda x: (x["session"] if not np.isnan(x["session"]) else 999, x["trial_in_session"]),
    ):
        if r["missed"]:
            continue
        key = (int(r["session"]), int(r["distrib_index"]))
        if key not in seen:
            seen[key] = trial_counter
            trial_counter += 1

    for r in pid_rows:
        if r["missed"]:
            continue
        key = (int(r["session"]), int(r["distrib_index"]))
        r["trial"] = seen[key]

    trial_rows: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(pid_rows):
        if not np.isnan(r["trial"]):
            trial_rows[int(r["trial"])].append(i)

    for indices in trial_rows.values():
        for obs_num, idx in enumerate(
            sorted(indices, key=lambda i: pid_rows[i]["trial_in_session"]),
            start=1,
        ):
            pid_rows[idx]["observation"] = obs_num


def _load_pid(pid: int, pid_dir: Path) -> list[dict] | None:
    missing = [
        sess
        for sess in (1, 2, 3)
        if not _session_mat_path(pid_dir, pid, sess).exists()
    ]
    if missing:
        warnings.warn(
            f"PID {pid}: missing session file(s) {missing}, loading available sessions",
            stacklevel=2,
        )
    available = [
        sess for sess in (1, 2, 3) if _session_mat_path(pid_dir, pid, sess).exists()
    ]
    if not available:
        warnings.warn(f"PID {pid}: no session files found, skipping", stacklevel=2)
        return None

    pid_rows: list[dict] = []
    for sess in available:
        mat_path = _session_mat_path(pid_dir, pid, sess)
        mat = sio.loadmat(mat_path)
        if "output" not in mat:
            raise KeyError(f"{mat_path}: expected variable 'output'")
        output = mat["output"].astype(float)
        if output.ndim != 2 or output.shape[1] != 13:
            raise ValueError(
                f"{mat_path}: output must have shape (N, 13), got {output.shape}"
            )
        for row_idx, row in enumerate(output):
            pid_rows.append(_row_record(pid, row, row_idx))

    _assign_trial_and_observation(pid_rows)
    return pid_rows


def _integer_pid_dirs(data_dir: Path) -> list[tuple[int, Path]]:
    pid_dirs: list[tuple[int, Path]] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        pid_dirs.append((int(child.name), child))
    return pid_dirs


# Columns that should be nullable integer (whole-number values or NaN)
_NULLABLE_INT_COLS = [
    "session",
    "trial",
    "observation",
    "trial_in_session",
    "distrib_index",
    "sd_value",
    "ev",
    "abs_trial_num",
]

# Columns that should be nullable boolean (True/False or NaN)
_NULLABLE_BOOL_COLS = ["catch_trial", "timeout"]


def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to appropriate nullable pandas dtypes.

    Uses pd.Int64Dtype() for integer columns that may contain NaN, and
    pd.BooleanDtype() for boolean columns that may contain NaN. This avoids
    float64/object upcasting and ensures integer equality comparisons work
    correctly throughout downstream code.
    """
    for col in _NULLABLE_INT_COLS:
        df[col] = df[col].astype(pd.Int64Dtype())
    for col in _NULLABLE_BOOL_COLS:
        df[col] = df[col].astype(pd.BooleanDtype())
    # TODO: downstream code that passes these columns to numpy directly
    # (e.g. df["session"].values) will get an object array rather than int64.
    # Use df["col"].to_numpy(dtype=int, na_value=-1) or similar at call sites.
    return df


_CANONICAL_EV_VALUES = [35, 65]

# Minimum number of valid trials in a sequence to attempt label inference.
# Sequences shorter than this retain their original labels without checking.
_MIN_TRIALS_FOR_INFERENCE = 10


def _infer_and_correct_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Infer correct ev label from reward mean and correct if needed.

    For each (pid, session, distrib_index) group with at least
    _MIN_TRIALS_FOR_INFERENCE valid trials, assigns:
      ev = argmin over {35, 65} of |reward_mean - candidate_ev|

    SD inference is intentionally omitted: the sd_value label comes from the
    cue presented to the participant on each trial and is independently reliable.
    The 5-point separation between adjacent SD levels is too small relative to
    sampling variation in reward_std (~2-3 points SE for n~25) to support
    reliable inference from reward statistics alone.

    Warns and corrects ev if the inferred value differs from the stored label.
    Sequences shorter than _MIN_TRIALS_FOR_INFERENCE are left unchanged.
    """
    group_cols = ["pid", "session", "distrib_index"]
    valid_mask = ~df["missed"] & df["value"].notna()

    for (pid, session, distrib_index), grp_idx in df[valid_mask].groupby(
        group_cols, sort=False
    ).groups.items():
        grp = df.loc[grp_idx]
        if len(grp) < _MIN_TRIALS_FOR_INFERENCE:
            continue

        reward_mean = float(grp["value"].mean())
        inferred_ev = min(_CANONICAL_EV_VALUES, key=lambda v: abs(reward_mean - v))
        stored_ev = int(grp["ev"].iloc[0])

        if inferred_ev != stored_ev:
            ev_err = abs(reward_mean - stored_ev)
            row_mask = (
                (df["pid"] == pid)
                & (df["session"] == session)
                & (df["distrib_index"] == distrib_index)
            )
            warnings.warn(
                f"PID {pid} session {int(session)} distrib={int(distrib_index)} "
                f"(n={len(grp)}): ev label corrected {stored_ev} → {inferred_ev} "
                f"(reward_mean={reward_mean:.1f}, err={ev_err:.1f})",
                stacklevel=2,
            )
            df.loc[row_mask, "ev"] = inferred_ev

    return df


def _scale_values(df: pd.DataFrame) -> pd.DataFrame:
    """Transform all continuous measurements to [-1, 1] scale.

    The Diederen task uses a 0-100 slider. The transform (x - 50) / 50
    maps this to [-1, 1] with midpoint 0, consistent with carrabin and yoo.

    ev and sd_value are also scaled for use as condition labels in
    SD-dependent analyses:
        ev:       35 → -0.3,  65 → +0.3
        sd_value:  5 →  0.1,  10 →  0.2,  15 →  0.3

    To recover original 0-100 values from the scaled columns:
        value / response / ev / initial_scale:  x_orig = x_scaled * 50 + 50
        pe / sd_value:                          x_orig = x_scaled * 50

    Reference values in original units (for cross-checking against published figures):
        EV=35 (low mean)  → -0.30    EV=65 (high mean) → +0.30
        SD=5  (low noise) →  0.10    SD=10             →  0.20    SD=15 (high noise) →  0.30
        slider midpoint   →  0.00    slider min (0)    → -1.00    slider max (100)   → +1.00
    """
    # Cast ev and sd_value to float64 before scaling (currently Int64)
    df["ev"] = df["ev"].astype("float64")
    df["sd_value"] = df["sd_value"].astype("float64")

    # Centre-and-scale: slider columns and condition mean labels
    for col in ["value", "response", "ev", "initial_scale"]:
        df[col] = (df[col] - 50.0) / 50.0

    # Rescale only (already differences or absolute values, not centred at 50)
    for col in ["pe", "sd_value"]:
        df[col] = df[col] / 50.0

    return df


def build_diederen(data_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for pid, pid_dir in _integer_pid_dirs(data_dir):
        if pid in EXCLUDED_PIDS:
            continue
        pid_rows = _load_pid(pid, pid_dir)
        if pid_rows is not None:
            records.extend(pid_rows)

    if not records:
        raise ValueError(f"No participant data loaded from {data_dir}")

    df = pd.DataFrame(records)[COLUMN_ORDER]
    df = _cast_dtypes(df)
    df = _infer_and_correct_labels(df)
    _validate(df)
    df = _scale_values(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    active = df[~df["missed"]]
    bad_sd = active[~active["sd_value"].isin([5, 10, 15])]
    if not bad_sd.empty:
        raise AssertionError(
            f"Non-missed rows with invalid sd_value: {bad_sd['sd_value'].unique()}"
        )
    bad_ev = active[~active["ev"].isin([35, 65])]
    if not bad_ev.empty:
        raise AssertionError(
            f"Non-missed rows with invalid ev: {bad_ev['ev'].unique()}"
        )

    has_resp = df["response"].notna()
    if has_resp.any():
        pe_err = (df.loc[has_resp, "pe"] - (df.loc[has_resp, "response"] - df.loc[has_resp, "value"])).abs()
        if (pe_err >= 0.01).any():
            bad = df.loc[has_resp].loc[pe_err >= 0.01]
            raise AssertionError(
                f"pe != response - value for {len(bad)} row(s), "
                f"max error {pe_err.max():.4f}"
            )

    pid_to_group = df.groupby("pid")["group"].nunique()
    multi_group = pid_to_group[pid_to_group > 1]
    if not multi_group.empty:
        raise AssertionError(
            f"PIDs with more than one group label: {multi_group.index.tolist()}"
        )

    for (pid, trial), grp in df.groupby(["pid", "trial"], sort=False):
        if pd.isna(trial):
            continue
        obs = grp["observation"].dropna().astype(int)
        if obs.empty:
            continue
        expected = set(range(1, len(obs) + 1))
        if set(obs) != expected:
            raise AssertionError(
                f"PID {pid} trial {int(trial)}: observation not gapless "
                f"1..{len(grp)} (got {sorted(obs.unique())}, n_rows={len(grp)})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build data/diederen.pkl from raw Diederen MATLAB files.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to raw data root (one integer-named subfolder per participant)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output pickle path (default: data/diederen.pkl under project root)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir is not a directory: {data_dir}")

    if args.out is None:
        project_root = Path(__file__).resolve().parent.parent
        out_path = project_root / "data" / "diederen.pkl"
    else:
        out_path = args.out.resolve()

    df = build_diederen(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)
    print(f"Wrote {out_path}: {df['pid'].nunique()} participants, {len(df)} rows")


if __name__ == "__main__":
    main()
