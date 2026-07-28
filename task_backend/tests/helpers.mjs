// tests/helpers.mjs
// Shared test utilities -- browser interactions and API helpers,
// consolidated from the one-off smoke_test_*.mjs scripts written and
// deleted throughout development, now a real reusable, persistent suite.

export const NUMBERS_PORT = 5183;
export const COLORS_PORT = 5184;
export const NUMBERS_URL = `http://localhost:${NUMBERS_PORT}/index-numbers.html`;
export const COLORS_URL = `http://localhost:${COLORS_PORT}/index-colors.html`;

export const SUPABASE_URL = 'https://htzsixtqavzkcqehdmib.supabase.co';
export const PUBLISHABLE_KEY = 'sb_publishable_iTDOvRdvp2-EMJ9hDtXMsw_4y7i7xOK';
const FUNCTIONS_BASE = `${SUPABASE_URL}/functions/v1`;
const REST_BASE = `${SUPABASE_URL}/rest/v1`;

/** Unique-per-run prefix -- repeated runs never collide with each other's
 * rows, and cleanup is trivial (delete everything under one prefix). */
export function testPid(label) {
  return `test_${label}_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
}

export async function callFunction(name, body) {
  const res = await fetch(`${FUNCTIONS_BASE}/${name}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', apikey: PUBLISHABLE_KEY },
    body: JSON.stringify(body),
  });
  return { status: res.status, body: await res.json().catch(() => null) };
}

/**
 * Deletes every row whose prolific_pid starts with `prefix`. Requires
 * SUPABASE_SECRET_KEY in the environment (see .env.test.example) --
 * silently skips (with a console warning) if it's not set, so the suite
 * still runs and verifies real behavior without it; only cleanup is
 * affected.
 */
export async function cleanupTestRows(prefix) {
  const secretKey = process.env.SUPABASE_SECRET_KEY;
  if (!secretKey) {
    console.warn(`[cleanup skipped] SUPABASE_SECRET_KEY not set -- rows starting with "${prefix}" were left in the database. See task_backend/.env.test.example.`);
    return;
  }
  const res = await fetch(`${REST_BASE}/events?prolific_pid=like.${encodeURIComponent(prefix)}*`, {
    method: 'DELETE',
    headers: { apikey: secretKey, Authorization: `Bearer ${secretKey}` },
  });
  if (!res.ok) {
    console.warn(`[cleanup failed] DELETE returned ${res.status} for prefix "${prefix}"`);
  }
}

/**
 * Seeds a fully-complete trial loop (all nTrials x 15 obs) directly via
 * the real API -- avoids driving 480+ real UI interactions through
 * Playwright for tests that only care about what happens AFTER a session
 * is complete (the completion-screen tests). Batched for speed.
 */
export async function seedFullTrialCompletion(pid, task, poolIndex = 0, nTrials = 32) {
  const calls = [];
  for (let t = 0; t < nTrials; t++) for (let o = 0; o < 15; o++) calls.push({ t, o });
  const BATCH = 25;
  for (let i = 0; i < calls.length; i += BATCH) {
    await Promise.all(calls.slice(i, i + BATCH).map(({ t, o }) => callFunction('progress-append', {
      prolific_pid: pid, task, pool_index: poolIndex, phase: 'trial',
      trial_index: t, observation_index: o, attempt: 0,
      response: 50, timed_out: false, rt: 1000, value: 50,
      true_mean: 50, true_std: 10, true_p: null, qid: t, error: 0, reward: 2,
    })));
  }
}

// ── Browser interaction helpers (numbers task) ──────────────────────────
export async function respond(page) {
  await page.focus('#response-slider');
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(50);
  await page.click('#submit-btn');
}

/**
 * tutorial_intro (obs 0, numbers task) gates the slider behind a 4-step
 * click-through reveal (tut-box-0 -> image box -> tut-box-1 -> tut-box-2)
 * -- the slider/submit button aren't interactive at all until all four
 * are clicked (see plugin-tutorial-intro-numbers.js's activateSlider()).
 */
export async function clickThroughTutorialIntro(page) {
  await page.click('#tut-box-0');
  await page.waitForTimeout(1500);
  await page.click('#tut-image-placeholder');
  await page.waitForTimeout(1500);
  await page.click('#tut-box-1');
  await page.waitForTimeout(200);
  await page.click('#tut-box-2');
  await page.waitForTimeout(200);
  await page.waitForSelector('#tut-slider-wrap[style*="visibility: visible"]', { timeout: 5000 });
}

export async function completeConsent(page) {
  for (let i = 0; i < 3; i++) { await page.click(`#reveal-box-${i}`); await page.waitForTimeout(150); }
  await page.check('#consent-checkbox');
  await page.waitForTimeout(150);
  await page.click('#consent-btn');
}

/** Full tutorial (numbers task): intro (obs 0) through all 15 tutorial
 * observations, the summary, and the "proceed to experiment" transition. */
export async function completeTutorial(page) {
  await page.waitForSelector('body[data-screen="tutorial_intro"]', { timeout: 10000 });
  await clickThroughTutorialIntro(page);
  await respond(page);
  for (let i = 1; i < 15; i++) {
    await page.waitForSelector('body[data-screen="tutorial_observation"]', { timeout: 10000 });
    await respond(page);
  }
  await page.waitForSelector('body[data-screen="tutorial_summary"]', { timeout: 10000 });
  await page.click('#proceed-btn');
  await page.waitForSelector('body[data-screen="tutorial_complete"]', { timeout: 10000 });
  await page.click('#tutorial-complete-btn');
}

export async function currentScreenInfo(page) {
  return page.evaluate(() => ({
    screen: document.body.dataset.screen,
    trial: document.body.dataset.trial,
    observation: document.body.dataset.observation,
  }));
}
