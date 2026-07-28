// tests/resume.spec.mjs
// The core thing this whole backend exists to prove: completing a trial,
// then reloading as the same participant, resumes at the NEXT trial --
// never re-showing a completed trial, never silently losing progress.
// Also confirms the "Trial X/32 -- generating new sequence..." transition
// screen shows the correct trial number on resume, not just on a fresh
// start (a real UX gap found and fixed during development).
import { test, expect } from '@playwright/test';
import {
  NUMBERS_URL, testPid, cleanupTestRows,
  completeConsent, completeTutorial, respond, currentScreenInfo,
} from './helpers.mjs';

test.describe('trial-boundary resume', () => {
  const pid = testPid('resume');

  test.afterAll(async () => {
    await cleanupTestRows(pid);
  });

  test('completing trial 0, then reloading, resumes at trial 1 with the correct transition screen', async ({ page, browser }) => {
    await page.goto(`${NUMBERS_URL}?PROLIFIC_PID=${pid}`);
    await expect(page.locator('body[data-screen="welcome"]')).toBeAttached({ timeout: 10000 });
    await page.click('#welcome-begin-btn');
    await expect(page.locator('body[data-screen="consent"]')).toBeAttached({ timeout: 10000 });
    await completeConsent(page);
    await completeTutorial(page);

    await expect(page.locator('body[data-screen="observation"]')).toBeAttached({ timeout: 10000 });
    let info = await currentScreenInfo(page);
    expect(info.trial).toBe('0');

    for (let o = 0; o < 15; o++) {
      await expect(page.locator('body[data-screen="observation"]')).toBeAttached({ timeout: 10000 });
      info = await currentScreenInfo(page);
      expect(info.trial).toBe('0');
      await respond(page);
    }

    await expect(page.locator('body[data-screen="inter_trial"]')).toBeAttached({ timeout: 10000 });
    await page.click('#next-btn');
    await expect(page.locator('body[data-screen="observation"]')).toBeAttached({ timeout: 10000 });
    info = await currentScreenInfo(page);
    expect(info.trial).toBe('1');

    await page.waitForTimeout(500); // let the last checkpoint land before "closing the tab"
    await page.close();

    // Reload as the SAME participant, fresh browser context.
    const page2 = await browser.newPage();
    await page2.goto(`${NUMBERS_URL}?PROLIFIC_PID=${pid}`);

    await expect(page2.locator('body[data-screen="inter_trial_reset"]')).toBeAttached({ timeout: 10000 });
    const transitionText = await page2.evaluate(() => document.body.textContent);
    expect(transitionText).toContain('Trial 2'); // startTrialIndex=1 -> trial_num=2

    await expect(page2.locator('body[data-screen="observation"]')).toBeAttached({ timeout: 10000 });
    const info2 = await currentScreenInfo(page2);
    expect(info2.trial).toBe('1');
    expect(info2.observation).toBe('0');

    await page2.close();
  });
});
