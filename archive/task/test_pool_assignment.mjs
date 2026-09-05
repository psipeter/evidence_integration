/**
 * test_pool_assignment.mjs
 * Pure Node.js checks for the per-participant sequence-pool mechanism --
 * no browser, no Vite dev server, no Playwright. Runs in milliseconds.
 *
 * Supersedes the old "Pool assignment" scenario formerly in
 * test_browser.mjs (removed -- see chat history): that scenario ran three
 * full tutorial+session flows through real Chromium just to exercise a
 * plain, DOM-free string hash plus a couple of static-data checks.
 * poolIndexForParticipant is a pure function with zero jsPsych/DOM
 * dependency (confirmed by reading timeline-builder.js's own source); the
 * embedded true_mean/true_p/value fields a real session appends are read
 * straight off the pool's own JSON files by build-trial-timeline.js, so
 * checking the JSON files directly IS checking what a real session would
 * see -- no rendering needed to prove that.
 *
 * Extracts poolIndexForParticipant's ACTUAL source from timeline-builder.js
 * via regex (rather than a hand-copied duplicate, which could silently
 * drift out of sync) -- same pattern test_browser.mjs already uses for
 * PROLIFIC_CODES, and for the same underlying reason: timeline-builder.js
 * can't be imported directly in plain Node (it pulls in jspsych, which
 * pulls in a .css import Node's ESM loader can't resolve without a bundler
 * -- confirmed directly: `node -e "import('./src/shared/timeline-builder.js')"`
 * throws "Unknown file extension '.css'").
 *
 * Run: node test_pool_assignment.mjs
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
let failed = 0;

function check(name, fn) {
  try {
    fn();
    console.log(`  PASS: ${name}`);
  } catch (e) {
    console.log(`  FAIL: ${name}: ${e.message}`);
    failed++;
  }
}

// ── Extract poolIndexForParticipant's real source ──────────────────────────
const timelineBuilderSrc = fs.readFileSync(
  path.join(__dirname, 'src/shared/timeline-builder.js'), 'utf8');
const fnMatch = timelineBuilderSrc.match(
  /export function poolIndexForParticipant\([^)]*\)\s*\{[\s\S]*?\n\}/);
if (!fnMatch) {
  console.log('FAIL: could not find poolIndexForParticipant in timeline-builder.js '
    + '-- source may have changed shape, update this script\'s extraction regex');
  process.exit(1);
}
const poolIndexForParticipant = new Function(
  `${fnMatch[0].replace('export function', 'return function')}`
)();

console.log('### Pool index hashing (poolIndexForParticipant, real source) ###');

check('same ID gives same index across repeated calls (determinism)', () => {
  const a1 = poolIndexForParticipant('participant_a', 200);
  const a2 = poolIndexForParticipant('participant_a', 200);
  if (a1 !== a2) throw new Error(`got ${a1} then ${a2} for the same ID`);
});

check('index is always an integer in [0, poolSize)', () => {
  const ids = ['x', 'participant_a', 'PROLIFIC_1234567890', '', 'a'.repeat(50), '日本語'];
  for (const id of ids) {
    const idx = poolIndexForParticipant(id, 200);
    if (typeof idx !== 'number' || !Number.isInteger(idx) || idx < 0 || idx >= 200) {
      throw new Error(`id=${JSON.stringify(id)} gave out-of-range/non-integer index ${idx}`);
    }
  }
});

check('many distinct IDs are not all forced to the same index (hash spreads)', () => {
  const ids = Array.from({ length: 30 }, (_, i) => `participant_${i}`);
  const indices = new Set(ids.map((id) => poolIndexForParticipant(id, 200)));
  if (indices.size < 10) {
    throw new Error(`30 distinct IDs collapsed to only ${indices.size} distinct indices`);
  }
});

check('a few real rows would carry the SAME pool_index within one session (consistency)', () => {
  // pool_index is assigned once per participant ID and is the same formula
  // call every time -- this is a direct consequence of the function being
  // pure, so "does a live session's rows share one pool_index" reduces to
  // "does calling this pure function with the same ID always return the
  // same number", already checked above. Spot-check a few representative
  // IDs together explicitly, mirroring what the old browser scenario
  // checked empirically via a live session's appended rows.
  const id = 'pool_consistency_check_participant';
  const calls = Array.from({ length: 5 }, () => poolIndexForParticipant(id, 200));
  const distinct = new Set(calls);
  if (distinct.size !== 1) {
    throw new Error(`expected all 5 calls to agree, got: ${[...distinct]}`);
  }
});

// ── Real pool data integrity (spot-check, not exhaustive) ──────────────────
console.log('\n### Sequence pool data integrity (spot-check across the real 200-member pool) ###');

const POOL_DIR = path.join(__dirname, 'sequences_pool');
const SPOT_CHECK_INDICES = [0, 50, 100, 199];  // spread across the pool, not just index 0

for (const task of ['continuous', 'binary']) {
  for (const idx of SPOT_CHECK_INDICES) {
    const idxStr = String(idx).padStart(4, '0');
    const file = path.join(POOL_DIR, `${task}_${idxStr}_sequences.json`);
    check(`${task} pool member ${idxStr}: every trial has non-empty values + correct ground-truth field`, () => {
      if (!fs.existsSync(file)) throw new Error(`file does not exist: ${file}`);
      const trials = JSON.parse(fs.readFileSync(file, 'utf8'));
      if (!Array.isArray(trials) || trials.length === 0) {
        throw new Error('sequences file is empty or not an array of trials');
      }
      for (const trial of trials) {
        if (!Array.isArray(trial.values) || trial.values.length === 0) {
          throw new Error(`trial ${trial.trial} has no values`);
        }
        if (task === 'continuous' && (trial.true_mean === null || trial.true_mean === undefined)) {
          throw new Error(`trial ${trial.trial} missing true_mean`);
        }
        if (task === 'binary' && (trial.true_p === null || trial.true_p === undefined)) {
          throw new Error(`trial ${trial.trial} missing true_p`);
        }
      }
    });
  }
}

console.log('\n' + '='.repeat(40));
if (failed > 0) {
  console.log(`${failed} check(s) FAILED`);
  process.exit(1);
} else {
  console.log('All pool-mechanism checks passed.');
}
