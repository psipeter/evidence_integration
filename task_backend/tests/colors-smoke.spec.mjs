// tests/colors-smoke.spec.mjs
// Lighter-weight smoke test for the colors task -- confirms its own
// distinct plugin set (different from numbers') at least loads and
// checkpoints correctly through consent. Doesn't drive the colors
// tutorial's own interaction (its box-reveal sequence differs from
// numbers' -- see plugin-tutorial-intro-colors.js) since the orchestration
// logic being verified (checkpointing, resume, timeout-retry) is already
// fully covered by the numbers-task tests; this file only guards against
// a colors-specific plugin regression going unnoticed.
import { test, expect } from '@playwright/test';
import { COLORS_URL, testPid, cleanupTestRows, completeConsent } from './helpers.mjs';

test.describe('colors task smoke test', () => {
  const pid = testPid('colorssmoke');

  test.afterAll(async () => {
    await cleanupTestRows(pid);
  });

  test('welcome -> consent -> tutorial_intro loads with no console errors', async ({ page }) => {
    const consoleErrors = [];
    page.on('console', (msg) => { if (msg.type() === 'error') consoleErrors.push(msg.text()); });
    page.on('pageerror', (err) => consoleErrors.push(err.message));

    await page.goto(`${COLORS_URL}?PROLIFIC_PID=${pid}`);
    await expect(page.locator('body[data-screen="welcome"]')).toBeAttached({ timeout: 10000 });
    await page.click('#welcome-begin-btn');

    await expect(page.locator('body[data-screen="consent"]')).toBeAttached({ timeout: 10000 });
    await completeConsent(page);

    await expect(page.locator('body[data-screen="tutorial_intro"]')).toBeAttached({ timeout: 10000 });
    await page.waitForTimeout(500);

    expect(consoleErrors).toEqual([]);
  });
});
