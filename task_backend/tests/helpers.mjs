// tests/helpers.mjs
// Shared test utilities -- browser interactions and API helpers,
// consolidated from the one-off smoke_test_*.mjs scripts written and
// deleted throughout development, now a real reusable, persistent suite.

import { readFileSync } from 'fs';

export const NUMBERS_PORT = 5183;
export const COLORS_PORT = 5184;
export const NUMBERS_URL = `http://localhost:${NUMBERS_PORT}/index-numbers.html`;
export const COLORS_URL = `http://localhost:${COLORS_PORT}/index-colors.html`;

export const SUPABASE_URL = 'https://htzsixtqavzkcqehdmib.supabase.co';
export const PUBLISHABLE_KEY = 'sb_publishable_iTDOvRdvp2-EMJ9hDtXMsw_4y7i7xOK';
const FUNCTIONS_BASE = `${SUPABASE_URL}/functions/v1`;
const REST_BASE = `${SUPABASE_URL}/rest/v1`;

/**
 * DUPLICATED from timeline-builder.js (not imported -- that file pulls in
 * jsPsych, CSS, and every plugin at module scope, none of which resolve
 * under plain Node ESM outside a browser/bundler context). Keep these two
 * copies in sync if the hash algorithm ever changes; it's a small, stable,
 * deterministic string hash, low risk of drifting unnoticed.
 */
export function poolIndexForParticipant(participantId, poolSize) {
  let hash = 5381;
  for (let i = 0; i < participantId.length; i++) {
    hash = ((hash * 33) ^ participantId.charCodeAt(i)) >>> 0;
  }
  return hash % poolSize;
}

const _poolCache = {};
/** Loads (and caches) sequences_<task>.json, or sequences_<task>_<variant>.json
 * if a variant name is given (see generate_sequences.py's --name flag) --
 * needed to know a test participant's REAL assigned trial values. */
export function loadPool(task, variant = null) {
  const key = variant ? `${task}:${variant}` : task;
  if (!_poolCache[key]) {
    const filename = variant ? `sequences_${task}_${variant}.json` : `sequences_${task}.json`;
    _poolCache[key] = JSON.parse(readFileSync(new URL(`../${filename}`, import.meta.url)));
  }
  return _poolCache[key];
}

/** Running mean of raw values[0..i] for each i -- mirrors
 * bonus-continuous.js's computeRunningMeans exactly (see its own
 * docstring); duplicated here for the same reason poolIndexForParticipant
 * is: that file isn't importable under plain Node. */
export function computeRunningMeans(values) {
  const out = [];
  let cum = 0;
  for (let i = 0; i < values.length; i++) {
    cum += values[i];
    out.push(cum / (i + 1));
  }
  return out;
}


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

/** Sets the slider to an EXACT value (not just "nudge and submit") --
 * needed for hand-computable-exact-reward tests. Confirmed working
 * against a real (non-gated) observation screen; the earlier failure
 * that led to the keyboard-nudge approach elsewhere was specific to
 * tutorial_intro's visibility-hidden reveal gate, not a general
 * limitation of dispatching a synthetic input event. */
export async function exactRespond(page, value) {
  await page.evaluate((v) => {
    const slider = document.querySelector('#response-slider');
    slider.value = v;
    slider.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
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
  for (let i = 0; i < 2; i++) { await page.click(`#reveal-box-${i}`); await page.waitForTimeout(150); }
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

/** Direct read of every row for one participant/task -- used to assert
 * "the rows themselves look right" (counts, attempt values, phases),
 * not just the app's own derived behavior. Requires SUPABASE_SECRET_KEY
 * (see cleanupTestRows) -- throws clearly if it's not set, since a test
 * that SILENTLY skipped its own row-shape assertions would be worse
 * than no test at all. */
export async function fetchEventsForPid(pid, task) {
  const secretKey = process.env.SUPABASE_SECRET_KEY;
  if (!secretKey) {
    throw new Error('SUPABASE_SECRET_KEY not set -- required to fetch rows directly for assertions. See .env.test.example.');
  }
  const res = await fetch(`${REST_BASE}/events?prolific_pid=eq.${encodeURIComponent(pid)}&task=eq.${task}&order=id.asc&limit=1000`, {
    headers: { apikey: secretKey, Authorization: `Bearer ${secretKey}` },
  });
  if (!res.ok) throw new Error(`fetchEventsForPid failed: ${res.status}`);
  return res.json();
}
