"""utils/pid_registry.py — persistent, append-only prolific_pid -> anonymized
integer pid mapping.

WHY THIS EXISTS
---------------
scripts/build_model_inputs.py's build_from_df() used to compute the integer
`pid` fresh on every call:

    all_pids = sorted(df["prolific_pid"].unique())
    pid_map = {p: i + 1 for i, p in enumerate(all_pids)}

That recomputes the mapping from scratch, by alphabetically sorting
whichever prolific_pids are in THIS call's data, every single time. The
moment the participant pool changes size (a new pilot round, --complete_pairs
picking up newly-finished participants, ...), inserting new prolific_pid
strings into that sort generally shifts the alphabetical rank of most of
the EXISTING participants too -- not just appends new ones at the end.
Confirmed directly this session: data/soltani_numbers.pkl grew from 35 to 45
pids, and the model-fit response files (still only pids 1-35, from the older
build) can no longer be safely joined on `pid` against the current human
data -- pid=5 in one file and pid=5 in the other are very likely different
real people.

This module replaces that from-scratch mapping with a PERSISTENT,
APPEND-ONLY one: once a prolific_pid is assigned an integer, it keeps that
integer forever, across every future build, no matter how the pool grows.
New prolific_pids only ever get NEW integers appended after the current
maximum -- existing assignments are never touched.

PRIVACY
-------
The registry file (REGISTRY_PATH) contains REAL Prolific participant IDs --
unlike every canonical data/*.pkl file, which only ever gets the anonymized
integer. It must NEVER be committed. It lives under data/, which is already
gitignored wholesale (see .gitignore's "Data & Figures" section) with only
carrabin.pkl/yoo.pkl explicitly un-ignored -- this file inherits that
protection with no !-override, and also gets its own explicit .gitignore
line for defense-in-depth and documentation clarity.

Since this file is never pushed through GitHub, keeping it in sync between
this machine and the cluster (or anywhere else a build might run) is a
separate, manual, non-git responsibility -- copy the one file directly
(scp/rsync), not the repo, whenever it changes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "pid_registry.json"


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, int]:
    """Existing prolific_pid -> pid mapping, or {} if the registry doesn't
    exist yet (e.g. the very first build)."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict[str, int], path: Path = REGISTRY_PATH) -> None:
    """Written sorted BY PID (not alphabetically by prolific_pid) so the
    file reads naturally as a 1..N roster when opened by hand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = dict(sorted(registry.items(), key=lambda kv: kv[1]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)
        f.write("\n")


def get_or_assign_pids(prolific_pids: Iterable[str],
                       path: Path = REGISTRY_PATH) -> dict[str, int]:
    """The stable integer pid for every prolific_pid in `prolific_pids`,
    assigning new ones only where needed and leaving every existing
    assignment untouched.

    New prolific_pids (ones not already in the registry) are assigned in
    SORTED order, continuing from max(existing pid) + 1 (or from 1, for an
    empty/nonexistent registry) -- deterministic, so re-running this on the
    exact same new set always produces the exact same new assignments,
    without depending on dict/set iteration order.

    Returns a mapping covering exactly the requested prolific_pids (a
    subset of the full registry, which may contain other people's pids
    from earlier, unrelated builds/pilots too) -- saves the UPDATED FULL
    registry back to disk before returning.
    """
    registry = load_registry(path)
    requested = list(prolific_pids)
    existing_ids = set(registry.values())
    new_pids = sorted({p for p in requested if p not in registry})

    next_id = (max(existing_ids) + 1) if existing_ids else 1
    for p in new_pids:
        registry[p] = next_id
        next_id += 1

    if new_pids:
        save_registry(registry, path)

    print(f"pid registry: {len(requested) - len(new_pids)} already known, "
          f"{len(new_pids)} newly assigned, {len(registry)} total in registry "
          f"({path})")

    return {p: registry[p] for p in requested}
