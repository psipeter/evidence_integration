// tests/basic-flow.spec.mjs
// Welcome -> consent -> tutorial_intro, with real checkpoint calls
// confirmed against the live backend. The smallest meaningful end-to-end
// check -- if this fails, nothing else in the suite is worth trusting.
import { test, expect } from '@playwright/test';
import { NUMBERS_URL, testPid, cleanupTestRows, completeConsent } from './helpers.mjs';

test.describe('basic flow (numbers task)', () => {
  const pid = testPid('basicflow');

  test.afterAll(async () => {
    await cleanupTestRows(pid);
  });

  test('welcome -> consent -> tutorial_intro, checkpoints fire with 200s', async ({ page }) => {
    const backendCalls = [];
    page.on('response', (res) => {
      if (res.url().includes('/functions/v1/')) {
        backendCalls.push({
          name: res.url().split('/functions/v1/')[1].split('?')[0],
          status: res.status(),
        });
      }
    });
    const consoleErrors = [];
    page.on('console', (msg) => { if (msg.type() === 'error') consoleErrors.push(msg.text()); });
    page.on('pageerror', (err) => consoleErrors.push(err.message));

    await page.goto(`${NUMBERS_URL}?PROLIFIC_PID=${pid}`);
    await expect(page.locator('body[data-screen="welcome"]')).toBeAttached({ timeout: 10000 });
    await page.click('#welcome-begin-btn');

    await expect(page.locator('body[data-screen="consent"]')).toBeAttached({ timeout: 10000 });
    await completeConsent(page);

    await expect(page.locator('body[data-screen="tutorial_intro"]')).toBeAttached({ timeout: 10000 });
    await page.waitForTimeout(2000); // let the consent checkpoint's response actually arrive (fire-and-forget, cold Edge Function can be slow)

    expect(consoleErrors).toEqual([]);
    expect(backendCalls.every((c) => c.status === 200)).toBe(true);
    const names = backendCalls.map((c) => c.name);
    expect(names).toContain('progress-check');
    // welcome + consent checkpoints
    expect(names.filter((n) => n === 'progress-append').length).toBeGreaterThanOrEqual(2);
  });
});
