"""utils/colors_quasi_qids.py

Colors' literal `qid` column never repeats -- every trial gets its own,
by design (confirmed empirically against real task_backend data: 0 of
640 (pid, observation, qid) groups have any repeat at all -- see chat
history). This module derives an EMPIRICAL quasi-qid instead, for the
"response variability across a repeated stimulus history" metrics
scripts/figure_soltani_temporal.py (columns 3-4) and
scripts/figure_soltani_variability.py both already compute for numbers
via its real, designed qid repeats.

QUASI-QID DEFINITION
---------------------
Group a participant's own trials by their literal first `prefix_length`
raw stimulus values -- NOT by anything about the eventual target/true_p.
This deliberately mirrors numbers' own qid semantics exactly: numbers'
prefix identity and target level are themselves independent axes (a
qid's repeats can be steered toward different targets), so numbers'
own metrics already only condition on a shared PREFIX, never on a
shared target -- requiring colors to ALSO match on true_p would be a
stricter, non-parallel standard. Confirmed empirically before settling
on this: pooling across true_p levels (rather than stratifying by it)
gave a much richer, evenly-spread sample with no meaningful validity
cost (see chat history for the full comparison).

A quasi-qid only exists for trials whose prefix recurs at least
`min_repeats` times for that participant -- everything else has no real
"repeat" to speak of and is EXCLUDED entirely, not assigned some
singleton group of its own.

DEFAULTS (verified empirically before settling on them, not guessed)
-----------------------------------------------------------------------
PREFIX_LENGTH=4 matches numbers' own prefix length exactly, giving a
genuinely comparable window across both tasks. MIN_REPEATS=3 was chosen
over the more permissive 2 specifically because the yield turned out to
support it: at P=4, R=3 gives 4-5 qualifying quasi-qid groups per
participant, remarkably even across all 5 real participants checked at
the time (not dominated by one or two people) -- see chat history for
the full P=1..6 x R=2..4 sweep that led here. Revisit these constants if
the participant pool grows and either the yield or its evenness changes.
"""
from __future__ import annotations

import pandas as pd

PREFIX_LENGTH = 4
MIN_REPEATS = 3


def add_quasi_qids(df: pd.DataFrame, prefix_length: int = PREFIX_LENGTH,
                    min_repeats: int = MIN_REPEATS) -> pd.DataFrame:
    """df: colors data with at least [pid, trial, observation, value]
    columns -- any existing `qid` column is ignored and then overwritten
    (colors' own qid was never a real repeat identity to begin with, see
    module docstring). Returns a COPY containing only the (all
    observations of) trials that participate in a qualifying quasi-qid
    group, with `qid` replaced by the quasi-qid label -- an arbitrary but
    stable per-(pid, prefix) integer (via groupby.ngroup(), which keys on
    the full (pid, prefix) pair, so two different participants who
    happen to share the same prefix pattern still get distinct quasi-qid
    values). Trials with fewer than `prefix_length` observations logged
    are excluded (can't compute a full-length prefix tuple for them) --
    shouldn't happen for a real, finished participant, but guarded
    rather than assumed.

    Everything downstream (figure_soltani_temporal.py's
    _add_resid_prefix, figure_soltani_variability.py's
    _prefix_response_std/_prefix_response_std_split) already does its
    own `observation < prefix_length` filtering, so this function
    doesn't need to pre-trim observations itself -- it only needs to
    decide WHICH trials qualify and what to call them.
    """
    prefix_obs = df[df["observation"] < prefix_length]

    trial_lengths = prefix_obs.groupby(["pid", "trial"])["observation"].nunique()
    complete_trials = set(trial_lengths[trial_lengths == prefix_length].index)

    prefixes = (prefix_obs.sort_values(["pid", "trial", "observation"])
                .groupby(["pid", "trial"])["value"].apply(tuple)
                .reset_index(name="prefix"))
    prefixes = prefixes[
        prefixes.apply(lambda r: (r["pid"], r["trial"]) in complete_trials, axis=1)
    ]

    prefixes["group_size"] = prefixes.groupby(["pid", "prefix"])["trial"].transform("size")
    qualifying = prefixes[prefixes["group_size"] >= min_repeats].copy()
    qualifying["quasi_qid"] = qualifying.groupby(["pid", "prefix"]).ngroup()

    out = df.merge(qualifying[["pid", "trial", "quasi_qid"]], on=["pid", "trial"], how="inner")
    out = out.drop(columns=["qid"], errors="ignore").rename(columns={"quasi_qid": "qid"})
    return out
