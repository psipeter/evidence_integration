// tests/happy-path.spec.mjs
// The suite's SINGLE canonical full-session traversal per task -- one
// real-UI-driven pass through welcome -> consent -> full tutorial -> both
// real trials -> the genuine end screen, per task (numbers, colors).
// Every check that doesn't need its OWN independent traversal is bundled
// onto this one instead of paying for a fresh multi-minute pass. This
// file replaces basic-flow.spec.mjs, colors-smoke.spec.mjs,
// full-session-bonus.spec.mjs, and a since-deleted color-rendering.spec.mjs
// (dropped per explicit direction -- a dedicated palette/color-
// consolidation check was judged not worth its own file; see TODO.md's
// "Codebase cleanup pass" section if this ever needs re-adding).
//
// TWO-PHASE STRUCTURE (explicit design, per direction from the person
// running this project): test 1 per task is a PURE UI-level pass -- it
// drives the real session end to end and asserts only on what the
// BROWSER can observe directly (screens reached, console errors, HTTP
// statuses on the outgoing checkpoint calls, and a nonzero captured
// reward as a cheap sanity check that the bonus formula isn't silently
// returning zero for every response -- see CLAUDE.md's own history of a
// real BONUS_DECAY miscalibration bug that did exactly this). No database
// access at all in test 1. Tests 2 (and, numbers-only, 3) run AFTER test
// 1 and inspect the DATABASE side of what test 1 just did.
// test.describe.serial + Playwright's own default behavior (a failed
// test in a serial block skips the rest of that block) makes this a real
// pre-test: if the traversal itself is broken, the DB/bonus checks never
// run and never produce a confusing secondary failure stacked on top of
// the real one.
//
// SIMPLIFIED FROM AN EARLIER VERSION: previously drove TWO participants
// per task with different response strategies, purely to prove their
// bonus totals differed from each other -- a sanity check on the test's
// OWN design, not something a real app bug would trip. Dropped per
// explicit direction ("fall back to a single participant, just to
// confirm the bonus calculation is correct") -- one participant is
// enough to verify the pipeline carries a real number through faithfully
// end to end.
//
// A genuine finding from investigating an earlier version of this file,
// worth keeping on record: one run found only 28/30 real trial rows for
// a participant in the database (a real, confirmed short-count, not a
// flaky assertion bug) -- re-run 3 more times immediately after, all 3
// came back with the full 30/30. Treated as an observed-once,
// unreproduced anomaly (most likely a single dropped fire-and-forget
// progress-append call under Playwright's faster-than-human click pace),
// not a confirmed bug. If test 2 below ever comes back short again,
// that's a second data point worth escalating -- don't shrug it off
// reflexively just because it didn't reproduce the first time.
import { execSync } from 'child_process';
import { test, expect } from '@playwright/test';
import {
  NUMBERS_URL, COLORS_URL, testPid, cleanupTestRows,
  poolIndexForParticipant, loadPool, computeRunningMeans,
  completeConsent, completeTutorial, respond, exactRespond,
  fetchEventsForPid,
} from './helpers.mjs';

const VARIANT = 'test2trial';
const N_TRIALS = 2; // must match generate_sequences.py's TEST_*_N_PREFIX x TEST_*_N_REPEATS / TEST_COLORS_N_TRIALS

/** Only numbers targets an EXACT response (the real running mean) via
 * exactRespond -- this lets test 3 (numbers-only) cross-check the
 * reward pipeline against compute_bonus.py byte-for-byte. Colors just
 * nudges the slider via respond() (see below), so it never needs its
 * own pool member's real values -- no equivalent function here for it. */
function loadNumbersSequencesFor(pid) {
  const pool = loadPool('numbers', VARIANT);
  return pool[poolIndexForParticipant(pid, pool.length)];
}

function defineHappyPathTests(task, url, { checkComputeBonusScript = false } = {}) {
  test.describe.serial(`happy path (${task})`, () => {
    const pid = testPid(`happy${task}`);
    const sequences = task === 'numbers' ? loadNumbersSequencesFor(pid) : null;
    let capturedRewardCents = 0;

    test.afterAll(async () => {
      await cleanupTestRows(pid);
    });

    test('drives a full real session with no console errors and all checkpoints returning 200', async ({ page }) => {
      test.setTimeout(120_000);

      const consoleErrors = [];
      page.on('console', (msg) => { if (msg.type() === 'error') consoleErrors.push(msg.text()); });
      page.on('pageerror', (err) => consoleErrors.push(err.message));

      const backendStatuses = [];
      page.on('response', (res) => {
        if (res.url().includes('/functions/v1/')) backendStatuses.push(res.status());
      });
      page.on('request', (req) => {
        if (req.url().includes('/functions/v1/progress-append')) {
          try {
            const body = JSON.parse(req.postData());
            if (body.phase === 'trial' && typeof body.reward === 'number') {
              capturedRewardCents += body.reward;
            }
          } catch { /* ignore -- not every progress-append body is JSON-parseable-as-expected (e.g. non-trial phases), and that's fine, we only care about trial rewards here */ }
        }
      });

      await page.goto(`${url}?PROLIFIC_PID=${pid}`);
      await expect(page.locator('body[data-screen="welcome"]')).toBeAttached({ timeout: 10000 });
      await page.click('#welcome-begin-btn');
      await expect(page.locator('body[data-screen="consent"]')).toBeAttached({ timeout: 10000 });
      await completeConsent(page);
      await completeTutorial(page);

      for (let t = 0; t < N_TRIALS; t++) {
        const runningMeans = task === 'numbers' ? computeRunningMeans(sequences[t].values) : null;
        for (let o = 0; o < 15; o++) {
          await expect(page.locator('body[data-screen="observation"]')).toBeAttached({ timeout: 10000 });
          if (task === 'numbers') {
            await exactRespond(page, runningMeans[o]);
          } else {
            await respond(page);
          }
        }
        await expect(page.locator('body[data-screen="inter_trial"]')).toBeAttached({ timeout: 10000 });
        await page.click('#next-btn');
      }

      await expect(page.locator('body[data-screen="end"]')).toBeAttached({ timeout: 15000 });
      await page.click('button.jspsych-btn');
      await page.waitForTimeout(1500);
      const bodyText = await page.evaluate(() => document.body.innerText);
      expect(bodyText).toMatch(/Session complete/i);

      expect(consoleErrors).toEqual([]);
      expect(backendStatuses.length).toBeGreaterThan(0);
      expect(backendStatuses.every((s) => s === 200)).toBe(true);
      // Cheap regression guard for both tasks -- not a precise numeric
      // check (that's test 3, numbers-only), just confirming the reward
      // formula isn't silently zeroing out every response for THIS task's
      // own error-mode wiring ('running_mean' for numbers, 'running_p' for
      // colors -- see config-base.js's ERROR_MODE, and CLAUDE.md's history
      // of a real BONUS_DECAY bug that did exactly this for every
      // response, not a hypothetical failure mode).
      expect(capturedRewardCents).toBeGreaterThan(0);
    });

    test('database rows for this session are the complete, gapless set', async () => {
      const rows = await fetchEventsForPid(pid, task);
      const trialRows = rows.filter((r) => r.phase === 'trial');
      const tutorialRows = rows.filter((r) => r.phase === 'tutorial');

      expect(rows.filter((r) => r.phase === 'welcome').length).toBeGreaterThanOrEqual(1);
      expect(rows.filter((r) => r.phase === 'consent').length).toBeGreaterThanOrEqual(1);
      expect(tutorialRows.length).toBeGreaterThanOrEqual(15); // intro + 14 observations, at minimum
      expect(trialRows.length).toBe(N_TRIALS * 15);
      expect(new Set(trialRows.map((r) => `${r.trial_index}-${r.observation_index}`)).size).toBe(N_TRIALS * 15);
      expect(trialRows.every((r) => r.attempt === 0)).toBe(true);
      expect(trialRows.every((r) => r.timed_out === false)).toBe(true);
      expect(rows.filter((r) => r.phase === 'finished').length).toBe(1);
      expect(rows.filter((r) => r.phase === 'terminated').length).toBe(0);

      // Pipeline-fidelity check for BOTH tasks -- confirms every reward
      // the app computed client-side (captured from the real outgoing
      // request bodies in test 1) actually landed in the database intact,
      // not just that SOME nonzero total exists.
      const dbTotalCents = trialRows.reduce((sum, r) => sum + r.reward, 0);
      expect(dbTotalCents).toBeCloseTo(capturedRewardCents, 5);
    });

    // Numbers-only: the reward FORMULA itself (computeResponseReward/
    // computeTrialReward in scoring.js) is shared, task-agnostic code --
    // verifying it once here is enough; re-verifying the same formula math
    // a second time on colors would only be re-testing scoring.js, not
    // anything colors-specific (colors' OWN task-specific piece -- its
    // ERROR_MODE='running_p' wiring actually producing a nonzero reward at
    // all -- is already covered by the cheap sanity check in test 1 above).
    if (checkComputeBonusScript) {
      test('compute_bonus.py reports the exact same total as what the app actually computed', () => {
        const output = execSync(`python3 compute_bonus.py --task ${task} --dry-run`, {
          cwd: new URL('..', import.meta.url).pathname,
          encoding: 'utf-8',
        });
        const line = output.split('\n').find((l) => l.startsWith(pid));
        expect(line, `no output line found for ${pid}\n\nFull output:\n${output}`).toBeTruthy();
        const amounts = [...line.matchAll(/\$\s*([\d.]+)/g)].map((m) => parseFloat(m[1]));
        expect(amounts.length).toBeGreaterThanOrEqual(1);
        const [raw] = amounts; // raw and clipped are identical here -- capturedRewardCents can't reach $5.00 at this trial count
        expect(raw).toBeCloseTo(capturedRewardCents / 100, 2);
      });
    }
  });
}

defineHappyPathTests('numbers', NUMBERS_URL, { checkComputeBonusScript: true });
defineHappyPathTests('colors', COLORS_URL, { checkComputeBonusScript: false });
