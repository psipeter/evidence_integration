// tests/completion-screens.spec.mjs
// All three session-ending paths (finish, terminate, returning
// participant) show the completion code as visible text, not just
// embedded silently in a redirect URL -- the direct fix for a real bug
// found during development (a failed redirect would have left a
// participant with no way to know their own code). Seeds a fully-
// complete trial loop directly via the API rather than driving real UI
// interactions through Playwright -- resume.spec.mjs and
// happy-path.spec.mjs already prove the trial loop itself works
// end to end; this file only cares about what happens after a session
// is over. N_TRIALS=2 matches the dev servers' 2-trial sequence variant
// (see playwright.config.mjs) -- seeding a different trial count than
// what the running client actually sees would trigger a real
// dataComplete=false mismatch in progress-finish's own sanity check.
import { test, expect } from '@playwright/test';
import {
  NUMBERS_URL, testPid, cleanupTestRows,
  seedFullTrialCompletion, callFunction,
} from './helpers.mjs';

const N_TRIALS = 2;

test.describe('completion screens show a visible code', () => {
  const finishPid = testPid('finish');
  const terminatePid = testPid('terminate');

  test.afterAll(async () => {
    await cleanupTestRows(finishPid);
    await cleanupTestRows(terminatePid);
  });

  test('finish path: end screen shows the code + Continue button', async ({ page }) => {
    await seedFullTrialCompletion(finishPid, 'numbers', 0, N_TRIALS);

    await page.goto(`${NUMBERS_URL}?PROLIFIC_PID=${finishPid}`);
    // Fully-complete trial loop -> skips straight to the end screen,
    // never replays any of the N_TRIALS seeded trials.
    await expect(page.locator('body[data-screen="end"]')).toBeAttached({ timeout: 15000 });
    await page.click('button.jspsych-btn');
    await page.waitForTimeout(1500);

    const text = await page.evaluate(() => document.body.innerText);
    expect(text).toMatch(/Session complete/i);
    expect(text).toMatch(/[A-Z0-9]{6,}/);
    expect(text).toMatch(/Continue to Prolific/i);
  });

  test('returning FINISHED participant sees their code, no re-run', async ({ page }) => {
    // Depends on the previous test having already called progress-finish
    // for finishPid -- same describe block, same afterAll cleanup scope.
    await page.goto(`${NUMBERS_URL}?PROLIFIC_PID=${finishPid}`);
    await page.waitForTimeout(2000);
    const text = await page.evaluate(() => document.body.innerText);
    expect(text).toMatch(/already completed/i);
    expect(text).toMatch(/[A-Z0-9]{6,}/);
  });

  test('returning TERMINATED participant sees their (different) code', async ({ page }) => {
    await callFunction('progress-finish', {
      prolific_pid: terminatePid, task: 'numbers', pool_index: 0, phase: 'terminated',
    });
    await page.goto(`${NUMBERS_URL}?PROLIFIC_PID=${terminatePid}`);
    await page.waitForTimeout(2000);
    const text = await page.evaluate(() => document.body.innerText);
    expect(text).toMatch(/already ended/i);
    expect(text).toMatch(/[A-Z0-9]{6,}/);
  });
});
