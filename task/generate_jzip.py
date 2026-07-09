#!/usr/bin/env python3
"""
Build production assets and package JATOS study archives (.jzip) for MindProbe.

Usage (from repo root or task/):
    python task/generate_jzip.py
    python task/generate_jzip.py --skip-build   # package existing dist-* only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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
                "uuid": "6c48d3e3-3b34-4cb8-a840-b7922ec6ff57",
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
                        "uuid": "df03e4d5-d810-447f-bbb1-1442b5e0260f",
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
                        "uuid": "b6c21ad2-f1fb-42a3-9c53-3dc3d702a191",
                        "title": "Default",
                        "active": True,
                        "maxActiveMembers": None,
                        "maxTotalMembers": None,
                        "maxTotalWorkers": None,
                        "allowedWorkerTypes": [
                            "GeneralMultiple",
                            "GeneralSingle",
                            "Jatos",
                            "PersonalMultiple",
                            "PersonalSingle",
                        ],
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
                "uuid": "5327124f-2370-49a8-a3bc-2c0534c1fcf2",
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
                        "uuid": "3a183108-0807-4921-9c3b-2df801a3a607",
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
                        "uuid": "cdb5f9a9-55cb-4d00-8b05-ccc48daf4db5",
                        "title": "Default",
                        "active": True,
                        "maxActiveMembers": None,
                        "maxTotalMembers": None,
                        "maxTotalWorkers": None,
                        "allowedWorkerTypes": [
                            "GeneralMultiple",
                            "GeneralSingle",
                            "Jatos",
                            "PersonalMultiple",
                            "PersonalSingle",
                        ],
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
    print(f"  {name}: {out_path.name} ({size_kb:.0f} KiB, {n_files} assets)")
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
