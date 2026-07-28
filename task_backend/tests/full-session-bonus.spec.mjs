// tests/full-session-bonus.spec.mjs
// The most end-to-end test in the suite: two participants, each driven
// through a COMPLETE, GENUINE session -- welcome, consent, tutorial, both
// real trials, all the way to the real `end` screen -- with ZERO seeding
// or artificial data injection anywhere. Runs against the dev servers'
// small 2-trial sequence variant (see playwright.config.mjs's
// VITE_SEQUENCES_VARIANT / generate_sequences.py's --name flag), which
// exists specifically so a full real session takes seconds, not the
// ~15-30 minutes a real 32-trial production session would.
//
// This is deliberately NOT built by seeding most of the session and only
// driving a few observations for real -- an earlier version of this test
// did exactly that, and it turned out to test "does my seeded-data model
// of the app match the app" more than "does the app work." A bug or
// behavior change in the real trial loop that a seeded shortcut doesn't
// exercise is exactly the kind of thing that approach would miss. Every
// single observation here goes through the real UI, the real backend,
// the real checkpoint calls.
//
// The two participants use different response strategies (respond near
// the real running-mean reference vs. a deliberate offset) purely to get
// two DISTINGUISHABLE bonus totals -- not to hit any exact target value.
// The slider has step="1" (see slider-numbers.js), so the browser rounds
// any value set programmatically to the nearest integer regardless of
// what's requested; rather than fight that to predict an exact reward,
// this test captures the REAL reward the app computes for each real
// observation (via the actual outgoing progress-append request bodies)
// and verifies that value survives correctly all the way through to
// compute_bonus.py's output. That's the property that actually matters --
// whether the pipeline carries the app's own numbers through intact, not
// whether this test can re-derive the reward formula a second time.
//
// NOTE: with only 2 trials (30 observations, max 60 cents raw per
// participant), this can never reach compute_bonus.py's $5.00 clip --
// that branch was already verified manually earlier this session against
// real over-$5 smoke-test data (see TODO.md). This test's job is
// end-to-end pipeline integrity at a genuinely real, fast scale, not
// exhaustive coverage of every reward magnitude.
import { execSync } from 'child_process';
import { test, expect } from '@playwright/test';
import {
  NUMBERS_URL, testPid, cleanupTestRows,
  poolIndexForParticipant, loadPool, computeRunningMeans,
  exactRespond, completeConsent, completeTutorial,
  currentScreenInfo, fetchEventsForPid,
} from './helpers.mjs';

const TASK = 'numbers';
const VARIANT = 'test2trial';
const N_TRIALS = 2; // must match generate_sequences.py's TEST_NUMBERS_N_PREFIX x TEST_NUMBERS_N_REPEATS

test.describe('full real session, no seeding, exact bonus pipeline verification', () => {
  const pool = loadPool(TASK, VARIANT);

  const participants = [
    { label: 'goodResponder', offset: 0 },
    { label: 'poorResponder', offset: 15 },
  ];
  for (const p of participants) {
    p.pid = testPid(p.label);
    p.poolIndex = poolIndexForParticipant(p.pid, pool.length);
    p.sequences = pool[p.poolIndex];
    p.capturedRewardCents = 0;
  }

  test.afterAll(async () => {
    for (const p of participants) await cleanupTestRows(p.pid);
  });

  test('drive a full real session for each participant, reaching the genuine end screen', async ({ browser }) => {
    test.setTimeout(180_000);

    for (const p of participants) {
      const page = await browser.newPage();
      page.on('request', (req) => {
        if (req.url().includes('/functions/v1/progress-append')) {
          try {
            const body = JSON.parse(req.postData());
            if (body.phase === 'trial' && typeof body.reward === 'number') {
              p.capturedRewardCents += body.reward;
            }
          } catch { /* ignore */ }
        }
      });

      await page.goto(`${NUMBERS_URL}?PROLIFIC_PID=${p.pid}`);
      await expect(page.locator('body[data-screen="welcome"]')).toBeAttached({ timeout: 10000 });
      await page.click('#welcome-begin-btn');
      await expect(page.locator('body[data-screen="consent"]')).toBeAttached({ timeout: 10000 });
      await completeConsent(page);
      await completeTutorial(page);

      for (let t = 0; t < N_TRIALS; t++) {
        const seq = p.sequences[t];
        const runningMeans = computeRunningMeans(seq.values);
        for (let o = 0; o < 15; o++) {
          await expect(page.locator('body[data-screen="observation"]')).toBeAttached({ timeout: 10000 });
          const info = await currentScreenInfo(page);
          expect(info.trial).toBe(String(t));
          expect(info.observation).toBe(String(o));
          await exactRespond(page, runningMeans[o] + p.offset);
        }
        await expect(page.locator('body[data-screen="inter_trial"]')).toBeAttached({ timeout: 10000 });
        await page.click('#next-btn');
      }

      await expect(page.locator('body[data-screen="end"]')).toBeAttached({ timeout: 15000 });
      await page.click('button.jspsych-btn');
      await page.waitForTimeout(1500);
      const bodyText = await page.evaluate(() => document.body.innerText);
      expect(bodyText).toMatch(/Session complete/i);
      await page.close();
    }

    // The two strategies should produce genuinely different totals --
    // a sanity check on the test's own design, not the app.
    expect(participants[0].capturedRewardCents).toBeGreaterThan(participants[1].capturedRewardCents);
  });

  test('rows in the database look correct for both participants', async () => {
    for (const p of participants) {
      const rows = await fetchEventsForPid(p.pid, TASK);
      const trialRows = rows.filter((r) => r.phase === 'trial');
      const finishedRows = rows.filter((r) => r.phase === 'finished');
      const terminatedRows = rows.filter((r) => r.phase === 'terminated');
      const welcomeRows = rows.filter((r) => r.phase === 'welcome');
      const consentRows = rows.filter((r) => r.phase === 'consent');
      const tutorialRows = rows.filter((r) => r.phase === 'tutorial');

      expect(trialRows.length).toBe(N_TRIALS * 15);
      expect(new Set(trialRows.map((r) => `${r.trial_index}-${r.observation_index}`)).size).toBe(N_TRIALS * 15);
      expect(trialRows.every((r) => r.attempt === 0)).toBe(true);
      expect(trialRows.every((r) => r.timed_out === false)).toBe(true);
      expect(finishedRows.length).toBe(1);
      expect(terminatedRows.length).toBe(0);
      expect(welcomeRows.length).toBeGreaterThanOrEqual(1);
      expect(consentRows.length).toBeGreaterThanOrEqual(1);
      expect(tutorialRows.length).toBeGreaterThanOrEqual(15); // intro + 14 observations, at minimum

      const dbTotalCents = trialRows.reduce((sum, r) => sum + r.reward, 0);
      expect(dbTotalCents).toBeCloseTo(p.capturedRewardCents, 5);
    }
  });

  test('compute_bonus.py reports the exact same total as what the app actually computed', () => {
    const output = execSync('python3 compute_bonus.py --task numbers --dry-run', {
      cwd: new URL('..', import.meta.url).pathname,
      encoding: 'utf-8',
    });

    for (const p of participants) {
      const line = output.split('\n').find((l) => l.startsWith(p.pid));
      expect(line, `no output line found for ${p.pid}\n\nFull output:\n${output}`).toBeTruthy();
      const amounts = [...line.matchAll(/\$\s*([\d.]+)/g)].map((m) => parseFloat(m[1]));
      expect(amounts.length).toBeGreaterThanOrEqual(1);
      const [raw] = amounts; // raw and clipped are identical here -- capturedRewardCents can't reach $5.00 at this trial count
      expect(raw).toBeCloseTo(p.capturedRewardCents / 100, 2);
    }
  });
});
