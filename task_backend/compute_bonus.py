"""
compute_bonus.py
=================
Computes real bonus payments from Supabase, matching the exact convention
established in the original task/compute_bonus_tmp.py (pilot #3, real
payouts): only REAL trial observations count toward bonus -- tutorial
rows also carry a `reward` value but never counted, and never should.
Under the old JATOS schema that meant screen == 'observation'; under this
schema it's phase == 'trial'. Same clip: $5.00 max per participant,
reward values stored in cents, summed then converted to dollars.

Not a "_tmp" scratch script like its predecessor -- this is a permanent
tool, since bonus computation will be needed every time this study runs.

PROLIFIC ID MISMATCH (verified July 2026, since this is exactly the kind
of platform detail worth checking rather than assuming): the dashboard's
bulk-bonus-payment box wants lines of `<submission_id>,<amount>`, and
Submission ID is NOT the same as Participant ID (prolific_pid) -- using
prolific_pid there fails. `events` only ever stored prolific_pid, so this
script needs Prolific's own demographic/submissions export for the study
(which lists Submission ID and Participant ID side by side) to translate
-- pass it via --prolific-export. Without that flag, the script still
works and prints/writes prolific_pid-keyed output, useful for (a) paying
via Prolific's API directly instead of the dashboard (its csv_bonuses
field DOES accept participant IDs, no translation needed there), or (b)
a manual spot-check before you have the export in hand.

Column matching in the Prolific export is keyword-based, not an exact
hardcoded header string -- Prolific's own docs describe the columns in
plain English ("Submission ID", "Participant ID") but the literal CSV
header text wasn't independently confirmed against a real export file,
so matching by keyword (case-insensitive, "submission"+"id",
"participant"+"id") is more robust than guessing exact casing/wording.

Usage
-----
    python compute_bonus.py --task numbers
    python compute_bonus.py --task colors --out bonus_colors.csv
    python compute_bonus.py --task numbers --max-bonus 5.00 --dry-run
    python compute_bonus.py --task numbers --prolific-export prolific_export_numbers.csv
"""

import argparse
import csv
import json
import os
import pathlib
import urllib.request
from collections import defaultdict

SUPABASE_URL = 'https://htzsixtqavzkcqehdmib.supabase.co'
REST_BASE = f'{SUPABASE_URL}/rest/v1'
PAGE_SIZE = 1000  # PostgREST's own default max rows per request


def load_secret_key():
    """Reads SUPABASE_SECRET_KEY from the environment, falling back to
    task_backend/.env.test if it exists (reuses the same local-only,
    gitignored file the test suite's cleanup step already uses -- no
    need to set this up twice)."""
    if os.environ.get('SUPABASE_SECRET_KEY'):
        return os.environ['SUPABASE_SECRET_KEY']
    env_test = pathlib.Path(__file__).parent / '.env.test'
    if env_test.exists():
        for line in env_test.read_text().splitlines():
            if line.startswith('SUPABASE_SECRET_KEY='):
                value = line.split('=', 1)[1].strip()
                if value:
                    return value
    raise SystemExit(
        "SUPABASE_SECRET_KEY not found in the environment or task_backend/.env.test.\n"
        "Get it from the Supabase dashboard: Settings -> API -> Secret keys."
    )


def fetch_all_rows(secret_key, task):
    """Paginates through every (prolific_pid, phase, reward, id) row for
    one task -- id is fetched too so we can determine each participant's
    LATEST phase (their current status) the same way progress-check does
    server-side (max id per group), not just sum rewards."""
    rows = []
    offset = 0
    while True:
        url = (f'{REST_BASE}/events'
               f'?task=eq.{task}&select=id,prolific_pid,phase,reward'
               f'&order=id.asc&limit={PAGE_SIZE}&offset={offset}')
        req = urllib.request.Request(url, headers={
            'apikey': secret_key,
            'Authorization': f'Bearer {secret_key}',
        })
        with urllib.request.urlopen(req) as resp:
            page = json.loads(resp.read())
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def compute_bonuses(rows, max_bonus_dollars):
    """Mirrors compute_bonus_tmp.py's exact convention: only phase=='trial'
    rows count (the direct equivalent of the old schema's
    screen=='observation'); reward is in cents; clipped to max_bonus_dollars."""
    raw_cents = defaultdict(float)
    n_obs = defaultdict(int)
    latest_id = {}
    latest_phase = {}

    for r in rows:
        pid = r['prolific_pid']
        if r['id'] > latest_id.get(pid, -1):
            latest_id[pid] = r['id']
            latest_phase[pid] = r['phase']
        if r['phase'] == 'trial' and r['reward'] is not None:
            raw_cents[pid] += r['reward']
            n_obs[pid] += 1

    pids = sorted(latest_phase.keys())
    results = []
    for pid in pids:
        dollars = raw_cents.get(pid, 0.0) / 100.0
        clipped = min(dollars, max_bonus_dollars)
        results.append({
            'prolific_pid': pid,
            'n_trial_observations': n_obs.get(pid, 0),
            'raw_bonus_dollars': round(dollars, 2),
            'clipped_bonus_dollars': round(clipped, 2),
            'latest_phase': latest_phase[pid],
        })
    return results


def _find_column(header, *keywords):
    """Keyword-based (not exact-string) column matching -- see module
    docstring for why. Returns the column index, or None if no header
    cell contains ALL of the given keywords (case-insensitive)."""
    for i, col in enumerate(header):
        col_lower = col.strip().lower()
        if all(kw in col_lower for kw in keywords):
            return i
    return None


def load_prolific_export(path, submission_col_override=None, participant_col_override=None):
    """Reads Prolific's demographic/submissions export for this study and
    returns {prolific_pid: {'submission_id': ..., 'status': ...}}.
    Raises clearly if the expected columns can't be found, rather than
    silently producing an empty/wrong mapping -- this feeds a real
    payment, worth failing loud. Same fuzzy-match-with-override pattern as
    task/reconcile_prolific_jatos.py's find_col/--prolific-id-col (a
    proven, already-battle-tested approach for this exact "Prolific's
    column names shift across UI versions" problem -- worth reusing
    rather than reinventing)."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)

        def resolve(override, label, *keywords):
            if override:
                if override not in header:
                    raise SystemExit(f"--{label}-col override {override!r} not found in header: {header}")
                return header.index(override)
            return _find_column(header, *keywords)

        submission_col = resolve(submission_col_override, 'submission-id', 'submission', 'id')
        participant_col = resolve(participant_col_override, 'participant-id', 'participant', 'id')
        status_col = _find_column(header, 'status')
        if submission_col is None or participant_col is None:
            raise SystemExit(
                f"Couldn't find both a Submission ID and Participant ID column in {path}.\n"
                f"Header found: {header}\n"
                "Prolific's export column wording may have changed -- re-run with "
                "--submission-id-col / --participant-id-col to specify them explicitly."
            )
        mapping = {}
        for row in reader:
            if not row:
                continue
            pid = row[participant_col].strip()
            mapping[pid] = {
                'submission_id': row[submission_col].strip(),
                'status': row[status_col].strip() if status_col is not None else '?',
            }
    return mapping


def print_table(results, max_bonus_dollars, prolific_map=None):
    header = f"\n{'prolific_pid':<28} {'obs rows':>9} {'raw bonus':>10} {'clipped':>9}  latest_phase"
    if prolific_map is not None:
        header += "  prolific_status  submission_id"
    print(header)
    print('-' * (90 if prolific_map is None else 140))
    total_raw = total_clipped = 0.0
    for r in results:
        total_raw += r['raw_bonus_dollars']
        total_clipped += r['clipped_bonus_dollars']
        line = (f"{r['prolific_pid']:<28} {r['n_trial_observations']:>9} "
                f"${r['raw_bonus_dollars']:>8.2f} ${r['clipped_bonus_dollars']:>7.2f}  {r['latest_phase']}")
        if prolific_map is not None:
            entry = prolific_map.get(r['prolific_pid'])
            if entry is None:
                line += "  NOT FOUND IN PROLIFIC EXPORT"
            else:
                line += f"  {entry['status']:<15}  {entry['submission_id']}"
        print(line)
    print('-' * (90 if prolific_map is None else 140))
    print(f"{'TOTAL':<28} {'':>9} ${total_raw:>8.2f} ${total_clipped:>7.2f}")
    print(f"\n(clipped at ${max_bonus_dollars:.2f} max per participant)")


def build_payment_lines(results, prolific_map=None):
    """Returns the list of ready-to-paste "<id>,<amount>" lines (only
    nonzero bonuses). If prolific_map is given, uses submission_id and
    SKIPS (with a warning) any participant not found in it -- pasting an
    unmatched/blank ID into the bulk-bonus box would fail the whole
    payment, not just that one line, so it's safer to omit and report
    than to guess."""
    lines, skipped = [], []
    for r in results:
        if r['clipped_bonus_dollars'] <= 0:
            continue
        amount = f"{r['clipped_bonus_dollars']:.2f}"
        if prolific_map is None:
            lines.append(f"{r['prolific_pid']},{amount}")
        else:
            entry = prolific_map.get(r['prolific_pid'])
            if entry is None:
                skipped.append(r['prolific_pid'])
                continue
            lines.append(f"{entry['submission_id']},{amount}")
    return lines, skipped


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--task', choices=['numbers', 'colors'], required=True)
    p.add_argument('--max-bonus', type=float, default=5.00)
    p.add_argument('--out', default=None, help='Output path for the ready-to-paste lines (default: bonus_<task>.csv)')
    p.add_argument('--dry-run', action='store_true', help="Print the table only, don't write a file")
    p.add_argument('--prolific-export', default=None,
                   help="Path to Prolific's demographic/submissions export for this study "
                        "(has Submission ID + Participant ID columns) -- translates the output "
                        "to Submission-ID-keyed lines, ready for the dashboard's bulk-bonus box. "
                        "Without this, output is prolific_pid-keyed (only usable via Prolific's API).")
    p.add_argument('--submission-id-col', default=None, help='Override auto-detected Submission ID column')
    p.add_argument('--participant-id-col', default=None, help='Override auto-detected Participant ID column')
    args = p.parse_args()

    secret_key = load_secret_key()
    print(f"Fetching events for task={args.task}...")
    rows = fetch_all_rows(secret_key, args.task)
    print(f"Fetched {len(rows)} rows across {len({r['prolific_pid'] for r in rows})} distinct participants.")

    prolific_map = load_prolific_export(
        args.prolific_export, args.submission_id_col, args.participant_id_col,
    ) if args.prolific_export else None

    results = compute_bonuses(rows, args.max_bonus)
    print_table(results, args.max_bonus, prolific_map)

    lines, skipped = build_payment_lines(results, prolific_map)

    if skipped:
        print(f"\nWARNING: {len(skipped)} participant(s) with a nonzero bonus were NOT found in the "
              f"Prolific export and are EXCLUDED from the output below -- pay them manually:")
        for pid in skipped:
            print(f"  {pid}")

    print(f"\n--- Ready to paste into Prolific's bulk-bonus box ({len(lines)} line(s)) ---")
    for line in lines:
        print(line)
    print("--- end ---")

    if args.dry_run:
        print("\n--dry-run set -- no file written.")
        return

    out_path = args.out or f'bonus_{args.task}.csv'
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    print(f"\nWrote {len(lines)} line(s) to {out_path}")
    if prolific_map is None:
        print(
            "\nNOTE: this file is keyed on prolific_pid, which Prolific's DASHBOARD bulk-bonus\n"
            "box will reject (it wants Submission IDs). Re-run with --prolific-export pointing\n"
            "at this study's Prolific export to get Submission-ID-keyed output, or pay via\n"
            "Prolific's API (/api/v1/bulk-bonus-payments/) directly -- its csv_bonuses field\n"
            "accepts participant IDs (prolific_pid) without any translation needed."
        )


if __name__ == '__main__':
    main()
