"""
Archived diederen-specific loss logic from fitting/losses.py.
"""

import pandas as pd

DIEDEREN_FIRST_BLOCKS_ONLY: bool = False


def filter_first_blocks(human: pd.DataFrame, n_blocks: int = 2) -> pd.DataFrame:
    out = []
    for (pid, session), grp in human.groupby(["pid", "session"], sort=False):
        g = grp.sort_values("trial_in_session").reset_index(drop=True)
        distribs = sorted(g["distrib_index"].dropna().unique().tolist())
        if len(distribs) != 2:
            out.append(g)
            continue
        block_count = {d: 0 for d in distribs}
        prev = None
        keep = []
        for i in range(len(g)):
            curr = int(g.at[i, "distrib_index"])
            if prev is not None and curr != prev:
                block_count[prev] += 1
            if block_count[curr] < n_blocks:
                keep.append(i)
            prev = curr
        if keep:
            out.append(g.iloc[keep])
    return pd.concat(out, ignore_index=True) if out else human.iloc[0:0]


def apply_diederen_loss_filter(model: pd.DataFrame, human: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    human = human[~human["catch_trial"].astype(bool)]
    if DIEDEREN_FIRST_BLOCKS_ONLY:
        human = filter_first_blocks(human)
        model = model[
            model.set_index(["pid", "trial", "observation"]).index.isin(
                human.set_index(["pid", "trial", "observation"]).index
            )
        ]
    return model, human
