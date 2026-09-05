#!/usr/bin/env python3
"""
Build production assets and package JATOS study archives (.jzip) for MindProbe.

Usage (from repo root or task/):
    python task/generate_jzip.py
    python task/generate_jzip.py --skip-build   # package existing dist-* only

UUIDs are freshly generated EVERY run (see STUDIES below) -- deliberately,
not an oversight. JATOS matches studies by UUID, not filename or content: importing
a jzip whose UUID matches an already-imported study triggers an "overwrite
this study?" prompt, which replaces that study's served assets in place
while leaving already-collected result data untouched. That's fine for
fixing a typo mid-pilot, but it's the wrong behavior when promoting a
genuinely new version (e.g. 6x4 -> 10x4) while an OLDER pilot's distributed
links are still supposed to be collecting responses on the OLD content --
overwriting would silently swap what those old links serve, with no
visible sign anything changed, potentially mixing two different task
versions under one nominal "pilot" label. Confirmed via JATOS's own docs/
forum: the correct way to keep two versions coexisting side by side is to
give the new one a different UUID, so JATOS imports it as a genuinely
separate study with its own new distribution links, never touching the old
one. Every run of this script now does that automatically -- each jzip you
build is its own new MindProbe study on import, never an overwrite of
whatever's already there. If you ever DO want an in-place overwrite (e.g.
genuinely just fixing a typo on a study that hasn't collected any real data
yet), you'd need to reuse the specific UUID JATOS shows you for that
study's properties -- there's no flag for that here on purpose, since the
safe default (never silently overwrite) is what you want in the overwhelming
majority of cases.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent

# endRedirectUrl (below, per study): JATOS's OWN platform-level redirect
# mechanism, entirely separate from the app's own jatos.endStudy/
# endStudyAndRedirect JS calls in finish-session.js. It only fires if a
# study/component calls a jatos.js function with showEndPage left at its
# default (true) -- finish-session.js now explicitly passes
# showEndPage=false for exactly this reason (see its own CORRECTNESS NOTE).
# Left as an empty string here (matching this same spec's own convention for
# other unset string fields, e.g. "comments": "") rather than any
# plausible-looking URL. A previous version of this script had real-looking
# Prolific completion URLs here (one of them a specific code CLAUDE.md
# documents as deliberately discarded from an old study configuration),
# which meant that if showEndPage's default was ever hit unexpectedly --
# exactly what was happening before the fix above -- a non-Prolific
# participant could get silently redirected to a dead Prolific completion
# page with no visible error. An empty string is the safer inert value: it
# reads unambiguously as "not configured" and won't risk failing any
# URL-format validation JATOS might apply on import, unlike a deliberately
# broken placeholder string would. If this field is ever intentionally
# needed again (e.g. a future component that doesn't route through
# finish-session.js), fill in a real, freshly-confirmed URL at that time --
# don't reuse a stale one found lying around.
UNUSED_END_REDIRECT_URL = ""

# Only GeneralSingle is allowed on the batch now (previously: GeneralMultiple,
# GeneralSingle, Jatos, PersonalMultiple, PersonalSingle -- all five at once).
# This is a deliberate narrowing, not just a default:
#
# - Prolific gives every participant the SAME Study URL (with PROLIFIC_PID
#   etc. appended as query params) -- there's no per-participant JATOS
#   "Personal" link in this workflow, so PersonalSingle/PersonalMultiple were
#   never actually reachable here; Jatos-type workers are for the researcher
#   running the study from JATOS's own GUI, not participants. Neither was
#   ever the real risk -- narrowing to just what's actually used closes the
#   door on ever accidentally handing out the wrong link type from JATOS's
#   admin panel and having the batch silently accept it.
# - GeneralMultiple vs GeneralSingle was the real decision (see chat
#   history): a real JATOS maintainer (Kristian Lange, on the JATOS forum)
#   diagnosed a symptom matching this project's own "Prolific shows
#   started/completed, JATOS shows nothing matching" investigation as
#   GeneralMultiple + a non-reloadable component (this study's components
#   are all reloadable: False) -- a reload/retry for ANY reason ends that
#   run as FAIL, and because the link is GeneralMultiple, nothing stops the
#   participant from just reopening the same link and starting a brand-new,
#   independent, possibly-empty run. GeneralSingle converts that from a
#   SILENT duplicate/empty run into a LOUD "Study can be done only once"
#   error -- worse for that one participant's experience, but far easier to
#   notice and reconcile than an invisible data gap.
# - This does NOT need a matching code change: GeneralSingle still supports
#   the dynamic PROLIFIC_PID query-param workflow (confirmed against JATOS's
#   own "Use Prolific" doc, which lists General Single alongside General
#   Multiple as the two supported options) -- nothing in timeline-builder.js
#   changes because of this.
#
# If a future Prolific study genuinely needs repeat access (e.g. a
# multi-session design), that's a deliberate exception to revisit here, not
# something to default back to broad access for.
ALLOWED_WORKER_TYPES = ["GeneralSingle"]


def assert_show_end_page_disabled() -> None:
    """
    UNUSED_END_REDIRECT_URL above is only safe as an empty string because
    finish-session.js explicitly passes showEndPage=SHOW_END_PAGE (=false)
    to jatos.endStudy -- that's the actual thing making JATOS's own
    end-page/redirect mechanism inert, not this script's value in
    isolation. Those two facts live in two unrelated files with no shared
    code path, so nothing would otherwise catch it if a future edit to
    finish-session.js ever dropped that argument -- this script would keep
    silently packaging a jzip built on a now-false assumption, and nobody
    would find out until a real MindProbe pilot run behaved unexpectedly
    (see CLAUDE.md's "Two independent completion mechanisms" note for the
    full history).

    This imports the REAL finish-session.js module with a throwaway node
    process and reads its actual exported SHOW_END_PAGE value, rather than
    regex-matching the source text (fragile to reformatting) or maintaining
    a second hardcoded copy of the assumption here (which is exactly the
    kind of silent-drift risk this check exists to close). Safe to import
    standalone outside a browser: finish-session.js only references
    `jatos`/`document` inside finishSession()'s function body, never at
    module top level, so importing it alone never executes that code.

    Refuses to build (sys.exit, not a warning) if the value isn't exactly
    `false`, or if the import fails for any reason (missing export, syntax
    error, node not found) -- any of those means the invariant this
    script's endRedirectUrl relies on can no longer be trusted.
    """
    finish_session_path = TASK_DIR / "src" / "shared" / "finish-session.js"
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            f"import('{finish_session_path.as_uri()}')"
            ".then(m => console.log(JSON.stringify(m.SHOW_END_PAGE)))"
            ".catch(e => { console.error(e); process.exit(1); })",
        ],
        cwd=TASK_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != "false":
        sys.exit(
            "generate_jzip.py REFUSING TO BUILD: could not confirm "
            "finish-session.js's exported SHOW_END_PAGE is `false`.\n"
            f"  node exit code: {result.returncode}\n"
            f"  node stdout: {result.stdout!r}\n"
            f"  node stderr: {result.stderr!r}\n"
            "This script's empty endRedirectUrl (see UNUSED_END_REDIRECT_URL "
            "above) is only safe because SHOW_END_PAGE disables JATOS's own "
            "end-page redirect -- if that constant changed intentionally, "
            "update this check AND decide what endRedirectUrl should now be "
            "(there's no automatically-correct value to derive). See "
            "CLAUDE.md's \"Two independent completion mechanisms\" note."
        )

STUDIES = {
    "continuous": {
        "dist_dir": "dist-continuous",
        "output": "evidence-integration-continuous.jzip",
        "jas": {
            "version": "3",
            "data": {
                # Fresh UUID every run, on purpose -- see this file's own
                # top-of-module docstring for why. Never hardcode this back
                # to a fixed literal.
                "uuid": str(uuid.uuid4()),
                "title": "Evidence Integration — Continuous",
                "description": "Sequential evidence integration task (continuous).",
                "groupStudy": False,
                "linearStudy": False,
                "allowPreview": False,
                "dirName": "evidence_integration_continuous",
                "comments": "",
                "jsonData": None,
                "endRedirectUrl": UNUSED_END_REDIRECT_URL,
                "studyEntryMsg": None,
                "componentList": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": "Task",
                        "htmlFilePath": "index-continuous.html",
                        "reloadable": False,
                        "active": True,
                        "comments": "",
                        "jsonData": None,
                    }
                ],
                "batchList": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": "Default",
                        "active": True,
                        "maxActiveMembers": None,
                        "maxTotalMembers": None,
                        # Deliberately left unbounded (None), not a
                        # --max-workers CLI flag as an earlier revision of
                        # this script had. Considered and dropped: Prolific's
                        # own "Places" participant cap is the intended
                        # control for over-recruitment, and a JATOS-side
                        # ceiling here was never confirmed to have caught
                        # anything in this project's actual past incidents --
                        # it would only help against Prolific's own slot
                        # management failing outright, which is a different
                        # (and unconfirmed) risk from the repeat-submission
                        # problem GeneralSingle above actually addresses.
                        # "+/- a few concurrent submissions" is an accepted
                        # risk, not something this script tries to guard
                        # against.
                        "maxTotalWorkers": None,
                        "allowedWorkerTypes": list(ALLOWED_WORKER_TYPES),
                        "comments": None,
                        "jsonData": None,
                    }
                ],
            },
        },
    },
    "binary": {
        "dist_dir": "dist-binary",
        "output": "evidence-integration-binary.jzip",
        "jas": {
            "version": "3",
            "data": {
                "uuid": str(uuid.uuid4()),
                "title": "Evidence Integration — Binary",
                "description": "Sequential evidence integration task (binary).",
                "groupStudy": False,
                "linearStudy": False,
                "allowPreview": False,
                "dirName": "evidence_integration_binary",
                "comments": "",
                "jsonData": None,
                "endRedirectUrl": UNUSED_END_REDIRECT_URL,
                "studyEntryMsg": None,
                "componentList": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": "Task",
                        "htmlFilePath": "index-binary.html",
                        "reloadable": False,
                        "active": True,
                        "comments": "",
                        "jsonData": None,
                    }
                ],
                "batchList": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": "Default",
                        "active": True,
                        "maxActiveMembers": None,
                        "maxTotalMembers": None,
                        "maxTotalWorkers": None,
                        "allowedWorkerTypes": list(ALLOWED_WORKER_TYPES),
                        "comments": None,
                        "jsonData": None,
                    }
                ],
            },
        },
    },
}


def run_build() -> None:
    print("Building production assets …")
    subprocess.run(
        ["npm", "run", "build:continuous"],
        cwd=TASK_DIR,
        check=True,
    )
    subprocess.run(
        ["npm", "run", "build:binary"],
        cwd=TASK_DIR,
        check=True,
    )


def package_study(name: str, spec: dict) -> Path:
    dist_dir = TASK_DIR / spec["dist_dir"]
    if not dist_dir.is_dir():
        raise FileNotFoundError(f"Missing build output: {dist_dir}")

    dir_name = spec["jas"]["data"]["dirName"]
    jas_name = f"{dir_name}.jas"
    out_path = TASK_DIR / spec["output"]

    if out_path.exists():
        out_path.unlink()

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        jas_bytes = json.dumps(spec["jas"], indent=4).encode("utf-8")
        zf.writestr(jas_name, jas_bytes)

        for path in sorted(dist_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = f"{dir_name}/{path.relative_to(dist_dir).as_posix()}"
            zf.write(path, arcname)

    size_kb = out_path.stat().st_size / 1024
    n_files = sum(1 for _ in dist_dir.rglob("*") if _.is_file())
    study_uuid = spec["jas"]["data"]["uuid"]
    batch = spec["jas"]["data"]["batchList"][0]
    print(f"  {name}: {out_path.name} ({size_kb:.0f} KiB, {n_files} assets)")
    print(f"    uuid: {study_uuid}  (fresh this run -- will import as a NEW MindProbe study, not overwrite an existing one)")
    print(f"    allowedWorkerTypes: {batch['allowedWorkerTypes']}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and package MindProbe .jzip files")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip npm build; package existing dist-* directories only",
    )
    parser.add_argument(
        "--task",
        choices=["continuous", "binary", "both"],
        default="both",
        help="Which study archive(s) to generate (default: both)",
    )
    args = parser.parse_args()

    assert_show_end_page_disabled()

    if not shutil.which("npm"):
        sys.exit("npm not found on PATH")

    if not args.skip_build:
        run_build()

    tasks = list(STUDIES) if args.task == "both" else [args.task]
    print("Packaging JATOS archives …")
    for name in tasks:
        package_study(name, STUDIES[name])

    print("Done. Import into MindProbe: Studies → + → Import Study")


if __name__ == "__main__":
    main()
