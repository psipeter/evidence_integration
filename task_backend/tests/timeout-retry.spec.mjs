// tests/timeout-retry.spec.mjs
// Forces three real observation timeouts in a row on the SAME (trial,
// observation) pair, and confirms: the same observation replays each
// time (never silently advancing), the `attempt` counter increments
// 0 -> 1 -> 2 with timed_out=true recorded on the real outgoing request
// each time, and the third timeout correctly triggers termination.
//
// Slow by nature (three real ~7s countdown expirations) -- this is the
// one file in the suite worth running less often if iteration speed ever
// becomes a problem, not a candidate for speeding up by faking the
// timers, since the whole point is observing the REAL countdown/replay
// mechanism end to end.
import { test, expect } from '@playwright/test';
import {
  NUMBERS_URL, testPid, cleanupTestRows,
  completeConsent, completeTutorial, currentScreenInfo,
} from './helpers.mjs';

test.describe('timeout retry + terminate', () => {
  const pid = testPid('timeout');

  test.afterAll(async () => {
    await cleanupTestRows(pid);
  });

  test('three consecutive timeouts on the same observation: attempt increments, then terminates', async ({ page }) => {
    const appendRequests = [];
    page.on('request', (req) => {
      if (req.url().includes('/functions/v1/progress-append')) {
        try { appendRequests.push(JSON.parse(req.postData())); } catch { /* ignore */ }
      }
    });

    await page.goto(`${NUMBERS_URL}?PROLIFIC_PID=${pid}`);
    await expect(page.locator('body[data-screen="welcome"]')).toBeAttached({ timeout: 10000 });
    await page.click('#welcome-begin-btn');
    await expect(page.locator('body[data-screen="consent"]')).toBeAttached({ timeout: 10000 });
    await completeConsent(page);
    await completeTutorial(page);
    await expect(page.locator('body[data-screen="observation"]')).toBeAttached({ timeout: 10000 });

    for (let attempt = 0; attempt < 3; attempt++) {
      const info = await currentScreenInfo(page);
      expect(info.trial).toBe('0');
      expect(info.observation).toBe('0');

      await page.waitForTimeout(7500); // let this observation's own countdown clock expire

      if (attempt < 2) {
        await expect(page.locator('body[data-screen="iti_replay"]')).toBeAttached({ timeout: 5000 });
        // ItiClockPlugin's timed_out=true branch deliberately does NOT
        // auto-advance -- requires a manual "Repeat" click after a ~2.1s
        // fade (anti-tab-visibility-exploit design, not a bug).
        await page.waitForSelector('#repeat-btn:not([disabled])', { timeout: 5000 });
        await page.click('#repeat-btn');
        await expect(page.locator('body[data-screen="observation"]')).toBeAttached({ timeout: 10000 });
      }
    }

    await page.waitForTimeout(4500); // "Too slow" pulse sequence before the terminated screen renders
    await expect(page.locator('body[data-screen="terminated"]')).toBeAttached({ timeout: 10000 });

    const obs00 = appendRequests.filter((c) => c && c.trial_index === 0 && c.observation_index === 0);
    expect(obs00.map((c) => c.attempt)).toEqual([0, 1, 2]);
    expect(obs00.every((c) => c.timed_out === true)).toBe(true);

    await page.click('#early-exit-btn');
    await page.waitForTimeout(1000);
  });
});
