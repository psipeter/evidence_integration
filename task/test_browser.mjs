/**
 * test_browser.mjs
 * Playwright end-to-end tests for both tasks (continuous + binary) across
 * Chromium, Firefox, and WebKit.
 *
 * Non-destructive: no source files are patched. This runs against the real
 * Vite dev server serving the actual task code, via a test-ONLY entry point
 * (index-test.html / src/test-harness.js) that is never linked from
 * production code and never included in any production build (vite.config.js
 * only lists index-continuous.html/index-binary.html as build inputs) — real
 * participants can never reach or discover it. That harness calls the exact
 * same buildAndRun() production uses; it only adjusts plain config fields
 * (trial count via array slicing, tObsMs/btiMs/itiMs via direct assignment)
 * before handing the config to the same production code path — no override
 * logic lives inside buildAndRun/timeline-builder.js itself. See
 * src/test-harness.js's docstring for the full rationale.
 *
 * The tutorial runs in FULL here (no skip option exists anymore) — tutorial
 * screens have no response deadline at all, so skipping was never about
 * avoiding a timer, just extra clicks. Running it for real means these tests
 * actually exercise the tutorial screens across browsers.
 *
 * Screen transitions are detected via `body[data-screen="..."]`, set by a
 * `on_trial_start` hook in timeline-builder.js (harmless in production —
 * just a DOM attribute). This avoids racing against guessed sleep durations,
 * which is what caused early flakiness/hangs while building this harness:
 * a screen with no visible text (e.g. the ITI clock, which is canvas-only)
 * looks identical to "stuck" if you're only checking textContent, so tests
 * must wait for the actual screen to change, not for a fixed amount of time
 * to pass. This matters especially for the tutorial's repeated
 * tutorial_iti <-> tutorial_observation cycle: always wait for the
 * INTERMEDIATE tutorial_iti screen first, not directly for the next
 * tutorial_observation, since the DOM can still show the just-submitted
 * tutorial_observation's stale attribute value for a few ms after
 * submission — tutorial_iti is a genuinely new state that can't be stale.
 *
 * The dev server is spawned in its own process group and killed by group
 * (not just by PID) on exit, so an interrupted run can't leave an orphaned
 * Vite process holding the port — if you ever see "Port 7655 is already in
 * use", run: lsof -ti:7655 | xargs -r kill -9
 *
 * Run:              node test_browser.mjs
 * Run a subset:      node test_browser.mjs --task=binary --browser=chromium
 */
import { chromium, firefox, webkit } from 'playwright';
import { spawn }         from 'child_process';
import path              from 'path';
import { fileURLToPath } from 'url';
import fs                from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT      = 7655;
const BASE_URL  = `http://localhost:${PORT}`;

const ENGINES = { chromium, firefox, webkit };

// ── CLI filters ──────────────────────────────────────────────────────────────
const argTask    = process.argv.find(a => a.startsWith('--task='))?.split('=')[1];
const argBrowser = process.argv.find(a => a.startsWith('--browser='))?.split('=')[1];

const TASKS    = argTask    ? [argTask]    : ['binary', 'continuous'];
const BROWSERS = argBrowser ? [argBrowser] : ['chromium', 'firefox', 'webkit'];

// Fast-test overrides applied via URL params on the test-only harness (see
// src/test-harness.js): 1500ms observation deadline, 400ms BTI/ITI, 3 trials.
const T_OBS_MS = 1500;
const testUrl = (task) =>
  `${BASE_URL}/index-test.html?task=${task}&tObsMs=${T_OBS_MS}&btiMs=400&itiMs=400&trials=3`;

// Same URL, but with a PROLIFIC_PID present -- flips isProlific=true in
// timeline-builder.js, which changes which finishSession() branch runs
// (real Prolific redirect vs local DOM-update-in-place -- see
// finish-session.js). No prior scenario in this suite ever set this param,
// so the Prolific branch had zero automated coverage before.
const testUrlProlific = (task) => `${testUrl(task)}&PROLIFIC_PID=e2e_test_pid`;

// Same idea, but with a caller-chosen PID -- needed by the pool-assignment
// scenario below, which specifically compares indices across DIFFERENT
// participant IDs (testUrlProlific's PID is fixed, so it can't be reused
// for that comparison).
const testUrlWithPid = (task, pid) => `${testUrl(task)}&PROLIFIC_PID=${pid}`;

// Parsed directly out of timeline-builder.js's PROLIFIC_CODES rather than
// hardcoded here -- avoids this test silently going stale if those
// placeholder codes are ever filled in for real Prolific deployment.
// Fails loudly (rather than falling back to a guessed value) if the source
// ever changes shape enough that this regex stops matching, per the
// project's "prefer no default" convention for anything that must always
// be present (see CLAUDE.md's jsPsych plugin conventions section).
const timelineBuilderSrc = fs.readFileSync(
  path.join(__dirname, 'src/shared/timeline-builder.js'), 'utf8');
function expectedProlificCode(task, kind) {
  const re = new RegExp(`${task}:\\s*\\{\\s*completion:\\s*'([^']+)',\\s*earlyExit:\\s*'([^']+)'`);
  const m = timelineBuilderSrc.match(re);
  if (!m) throw new Error(`Could not parse PROLIFIC_CODES for task=${task} from timeline-builder.js`);
  return kind === 'completion' ? m[1] : m[2];
}

// ── Start / stop Vite dev server (own process group, killed by group) ──────
function startDevServer() {
  return new Promise((resolve, reject) => {
    const proc = spawn(
      'npx', ['vite', '--port', String(PORT), '--strictPort'],
      { cwd: __dirname, stdio: ['ignore', 'pipe', 'pipe'], detached: true }
    );
    let resolved = false;
    const onData = (buf) => {
      const s = buf.toString();
      if (!resolved && /ready in|Local:/i.test(s)) {
        resolved = true;
        resolve(proc);
      }
    };
    proc.stdout.on('data', onData);
    proc.stderr.on('data', onData);
    proc.on('error', reject);
    proc.on('exit', (code) => {
      if (!resolved) reject(new Error(`Vite dev server exited early (code ${code})`));
    });
    setTimeout(() => { if (!resolved) reject(new Error('Vite dev server did not start in time')); }, 15000);
  });
}

function stopDevServer(proc) {
  if (!proc || proc.killed) return;
  try {
    process.kill(-proc.pid, 'SIGKILL');  // kill the whole process group
  } catch {
    try { proc.kill('SIGKILL'); } catch {}
  }
}

// ── Start / stop the result server (mimics jatos.submitResultData) ─────────
// jatos-shim.js -- used by index-test.html exactly as it is by real local
// dev, since index-test.html never loads a real jatos.js (see
// test-harness.js) -- POSTs every endStudy/endStudyAndRedirect call to
// dev-server.js, which writes a result_<timestamp>.json file into
// dev-results/. Spinning this up lets the end-screen/early-exit scenarios
// below assert that a save ACTUALLY happened, not just that the right
// screen rendered. This closes the exact coverage gap behind every real
// save/redirect bug in this project's history (see CLAUDE.md's "Exit/
// redirect and data-saving architecture") -- each one was found only by a
// live MindProbe pilot run, because nothing automated used to reach this
// far into the flow.
const RESULT_PORT = 3099;
const RESULTS_DIR = path.join(__dirname, 'dev-results');

function startResultServer() {
  return new Promise((resolve, reject) => {
    const proc = spawn(
      'node', ['dev-server.js'],
      { cwd: __dirname, stdio: ['ignore', 'pipe', 'pipe'], detached: true }
    );
    let resolved = false;
    const onData = (buf) => {
      if (!resolved && /running at/i.test(buf.toString())) {
        resolved = true;
        resolve(proc);
      }
    };
    proc.stdout.on('data', onData);
    proc.stderr.on('data', onData);
    proc.on('error', reject);
    proc.on('exit', (code) => {
      if (!resolved) reject(new Error(`Result server exited early (code ${code})`));
    });
    setTimeout(() => { if (!resolved) reject(new Error('Result server did not start in time')); }, 8000);
  });
}

function stopResultServer(proc) {
  if (!proc || proc.killed) return;
  try {
    process.kill(-proc.pid, 'SIGKILL');
  } catch {
    try { proc.kill('SIGKILL'); } catch {}
  }
}

// Test-created save files use dev-server.js's own `result_<timestamp>.json`
// naming -- distinct from every real pilot/dev filename already sitting in
// dev-results/ (see CLAUDE.md's "Pilot data files"), so snapshotting and
// cleaning up only this pattern can never touch real data.
const isTestResultFile = (name) => /^result_.*\.json$/.test(name);
const snapshotResultFiles = () =>
  new Set(fs.readdirSync(RESULTS_DIR).filter(isTestResultFile));

// Polls for a NEW result_*.json file not present in `before` -- the save
// POST completes asynchronously relative to the button click that triggers
// it, so this can't just check once. Returns the new filename.
async function waitForNewResultFile(before, timeout = 8000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const now = fs.readdirSync(RESULTS_DIR).filter(isTestResultFile);
    const fresh = now.find((f) => !before.has(f));
    if (fresh) return fresh;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('No new result file appeared in dev-results/ within timeout');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const wait   = (p, ms) => p.waitForTimeout(ms);
const hasEl  = (p, sel) => p.$(sel).then(el => el !== null);

// Wait for a specific jsPsych screen (see data.screen in each plugin node)
// rather than guessing how long a transition takes. Also covers
// create-early-exit.js's manual DOM injection ('terminated'), which sets
// the same attribute by hand since it runs outside jsPsych's trial system.
const waitForScreen = (p, screen, timeout = 10000) =>
  p.waitForSelector(`body[data-screen="${screen}"]`, { timeout });

// Wait for the too-slow pulse to show a specific remaining-timeout count,
// via its data attribute rather than matching the countdown phrase's exact
// wording (which is expected to vary/change over time).
const waitForTimeoutsRemaining = (p, n, timeout = 10000) =>
  p.waitForSelector(`#too-slow-pulse[data-timeouts-remaining="${n}"]`, { timeout });

// Sets the slider's value directly and dispatches a genuine 'input' event
// (bubbling, so the app's own listener on #response-slider receives it
// exactly as it would from a real interaction) -- deliberately NOT based on
// computing a pixel position from the element's bounding box. The app's
// slider listeners (slider-continuous.js / slider-binary.js) only need an
// 'input' event to remove the unset/last styling and enable #submit-btn;
// this triggers that same code path without depending on rendered geometry.
const moveSlider = async (p, pct) => {
  await p.evaluate((value) => {
    const el = document.querySelector('#response-slider');
    el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, pct);
};

const submit = async (p) => {
  await p.waitForSelector('#submit-btn:not([disabled])');
  await p.click('#submit-btn');
};

const doConsent = async (p) => {
  // Welcome/title screen is the very first thing shown — click through it
  // before consent. Folded into doConsent (rather than a separate step at
  // every call site) since every scenario needs both, in this order.
  await waitForScreen(p, 'welcome');
  await p.click('#welcome-begin-btn');

  await p.waitForSelector('#reveal-box-0');
  for (const id of ['reveal-box-0', 'reveal-box-1', 'reveal-box-2']) {
    await p.click(`#${id}`);
    await wait(p, 80);
  }
  await p.click('#consent-checkbox');
  await wait(p, 80);
  await p.waitForSelector('#consent-btn:not(.consent-btn-locked)');
  await p.click('#consent-btn');
};

// Walks through the FULL tutorial (no skip option exists anymore): intro
// (progressive box/image reveal, first response) -> remaining tutorial
// observations (each preceded by a fixed 1s tutorial_iti, no click needed)
// -> tutorial summary -> 3-screen timeout demo -> real trial 1.
// Same IDs across both tasks (tut-box-0/1/2, tut-image-placeholder,
// response-slider, submit-btn, proceed-btn) per the project's naming
// convention, so this one implementation covers both without branching.
const doTutorial = async (p) => {
  await waitForScreen(p, 'tutorial_intro');
  await p.click('#tut-box-0');
  await wait(p, 80);
  await p.click('#tut-image-placeholder');
  await wait(p, 80);
  await p.click('#tut-box-1');
  await wait(p, 80);
  await p.click('#tut-box-2');
  await wait(p, 80);
  await p.waitForSelector('#response-slider');
  await moveSlider(p, 55);
  await submit(p);

  // Remaining tutorial observations, if any (n_obs varies but is always
  // >=1 more after the intro). tutorial_iti always intervenes between them
  // (fixed duration, no click) -- waiting for IT first, rather than
  // directly for the next tutorial_observation, avoids a race where the
  // DOM still shows the just-submitted screen's stale attribute value.
  while (true) {
    await p.waitForSelector(
      'body[data-screen="tutorial_iti"], body[data-screen="tutorial_summary"]',
      { timeout: 10000 }
    );
    const screen = await p.getAttribute('body', 'data-screen');
    if (screen === 'tutorial_summary') break;
    await waitForScreen(p, 'tutorial_observation', 5000);
    await p.waitForSelector('#response-slider');
    await moveSlider(p, 45);
    await submit(p);
  }

  // Tutorial summary -> Next
  await p.click('#proceed-btn');

  // Timeout demo: 3 sub-screens inside a SINGLE jsPsych trial (data-screen
  // stays 'timeout_demo' throughout -- these are plain DOM swaps, not new
  // jsPsych trials -- so wait for specific elements, not attribute changes).
  await waitForScreen(p, 'timeout_demo');
  // Screen 1 has a real countdown clock (respects the tObsMs override, so
  // this resolves quickly under test timing) that must run out on its own
  // before screen 2's button exists.
  await p.waitForSelector('#demo-next-btn', { timeout: T_OBS_MS + 8000 });
  await p.click('#demo-next-btn');
  await p.waitForSelector('#demo-proceed-btn', { timeout: 5000 });
  await p.click('#demo-proceed-btn');

  // Real trial 1
  await waitForScreen(p, 'observation', 10000);
  await p.waitForSelector('#response-slider');
};

// Let an observation time out without responding, then land on the
// "too slow" replay screen (data-screen="iti_replay").
const letObsTimeOut = (p) => waitForScreen(p, 'iti_replay', T_OBS_MS + 5000);

// Same, but ALSO clicks through the "too slow" screen's Repeat button --
// use this for any timeout that's NOT the final, budget-exhausting one.
// That screen deliberately no longer auto-advances (see CLAUDE.md's
// "Tab-visibility handling for observation timeouts") -- a real behavioral
// change this suite needs to account for, not a bug. Only the genuinely
// FINAL timeout (which exhausts MAX_TIMEOUTS_PER_TRIAL) skips this screen
// entirely and falls straight through to 'terminated' -- see
// build-trial-timeline.js's conditional_function -- so scenarios calling
// this for every timeout except the last one.
const letObsTimeOutAndRepeat = async (p) => {
  await letObsTimeOut(p);
  await p.waitForSelector('#repeat-btn:not([disabled])');
  await p.click('#repeat-btn');
};

// finishSession's non-Prolific branch (finish-session.js) replaces
// #jspsych-content in place with this exact confirmation copy -- no screen
// change, no navigation. Waiting for it confirms the participant-visible
// half of the save flow didn't silently fail, alongside the result-file
// check that confirms the data-visible half.
const waitForSaveConfirmation = (p, timeout = 5000) =>
  p.waitForSelector('text=Session complete', { timeout });

// ── Test scenarios (task-agnostic — same jsPsych plugin IDs for both tasks) ──
// MAX_TIMEOUTS_PER_TRIAL is 3 (see timeline-builder.js) — these scenarios
// deliberately stay under/at that budget; going one timeout further always
// triggers session termination instead of a "too slow" replay screen.
const SCENARIOS = [
  {
    name: 'Normal submit: no too-slow, ITI appears',
    fn: async (p) => {
      await doConsent(p); await doTutorial(p);
      await moveSlider(p, 65);
      await submit(p);
      // Next screen is either the between-obs ITI or straight to obs 2 —
      // either is fine, but it must NOT be the too-slow replay screen.
      await p.waitForSelector('body[data-screen="iti"], body[data-screen="observation"]', { timeout: 3000 });
      if (await hasEl(p, '#too-slow-pulse')) throw new Error('Too-slow pulse shown after normal submit');
    },
  },
  {
    name: 'Timeout: too-slow shows 2 remaining',
    fn: async (p) => {
      await doConsent(p); await doTutorial(p);
      await letObsTimeOut(p);  // 1st timeout — 2 of 3 remaining
      await waitForTimeoutsRemaining(p, 2);
    },
  },
  {
    name: '1 timeout remaining text correct',
    fn: async (p) => {
      await doConsent(p); await doTutorial(p);
      await letObsTimeOutAndRepeat(p);                     // 1st timeout
      await waitForScreen(p, 'observation', T_OBS_MS + 5000);  // loop_function replays same obs
      await letObsTimeOut(p);                              // 2nd timeout — 1 of 3 remaining
      await waitForTimeoutsRemaining(p, 1);
    },
  },
  {
    name: '3 timeouts: session terminated, no summary',
    fn: async (p) => {
      await doConsent(p); await doTutorial(p);
      await letObsTimeOutAndRepeat(p);                     // 1st timeout
      await waitForScreen(p, 'observation', T_OBS_MS + 5000);
      await letObsTimeOutAndRepeat(p);                     // 2nd timeout
      await waitForScreen(p, 'observation', T_OBS_MS + 5000);
      // 3rd timeout exhausts the budget and triggers earlyExit(), a manual
      // DOM injection (not a jsPsych trial) that sets data-screen='terminated'
      // by hand (see create-early-exit.js) for exactly this reason.
      await waitForScreen(p, 'terminated', T_OBS_MS + 8000);
      if (!await hasEl(p, '#early-exit-btn')) throw new Error('No early-exit button');
      if (await hasEl(p, '#summary-svg'))     throw new Error('Summary shown after termination');

      // Same coverage gap as the "Completes all trials" scenario below, but
      // for the early-exit path: create-early-exit.js wires its button via
      // a raw pointerdown listener straight to finishSession() -- a
      // separate code path from the normal end screen's jsPsych
      // button-response plugin. Reaching 'terminated' only proved the
      // screen renders; it never proved clicking the button actually saves
      // anything, which is the exact class of bug this project has hit
      // before (see CLAUDE.md's "Exit/redirect and data-saving
      // architecture").
      const beforeExit = snapshotResultFiles();
      await p.click('#early-exit-btn');
      await waitForSaveConfirmation(p);
      const savedExitFile = await waitForNewResultFile(beforeExit);
      const savedExit = JSON.parse(fs.readFileSync(path.join(RESULTS_DIR, savedExitFile), 'utf8'));
      if (!Array.isArray(savedExit) || savedExit.length === 0) {
        throw new Error(`Saved result file ${savedExitFile} was empty or not an array`);
      }
      fs.unlinkSync(path.join(RESULTS_DIR, savedExitFile));  // keep dev-results/ clean
    },
  },
  {
    name: 'Submit then continue to next obs',
    fn: async (p) => {
      await doConsent(p); await doTutorial(p);
      await moveSlider(p, 50);
      await submit(p);
      await waitForScreen(p, 'observation', 5000);
      if (await hasEl(p, '#too-slow-pulse')) throw new Error('Unexpected too-slow pulse after submit');
    },
  },
  {
    // Closes a real coverage gap: every other scenario only exercises
    // individual-observation interactions (submit, timeout, etc.) within
    // the first trial or two. None of them ever complete every trial and
    // reach the LAST trial's own summary screen -- which is built by a
    // separate, duplicated code path in build-trial-timeline.js (the
    // "Final summary" block, distinct from the main per-trial loop's own
    // summary block) and DID drift out of sync with it once already (a
    // missing true_p field there crashed the binary task on the very last
    // trial in a real pilot run -- see CLAUDE.md). Completing all
    // `trials` (3, per testUrl's override) x 15 observations is slower
    // than the other scenarios, but it's the only way to actually visit
    // that final-trial-specific code path at all.
    name: 'Completes all trials, reaches final summary without crashing',
    fn: async (p) => {
      await doConsent(p); await doTutorial(p);
      for (let t = 0; t < 3; t++) {
        for (let o = 0; o < 15; o++) {
          await waitForScreen(p, 'observation', T_OBS_MS + 5000);
          await moveSlider(p, 40 + o);
          await submit(p);
          if (o < 14) await waitForScreen(p, 'iti', 5000);
        }
        if (t < 2) {
          await waitForScreen(p, 'inter_trial', 5000);
          await p.click('#next-btn');
          await waitForScreen(p, 'inter_trial_reset', 5000);
        }
      }
      // This is the exact screen that previously threw an uncaught error
      // ("You must specify a value for the 'true_p' parameter...") and left
      // a blank page for binary -- reaching it with real rendered content
      // and no page error (checked below, after fn returns) is the actual
      // regression test.
      await waitForScreen(p, 'inter_trial', 5000);
      const contentLength = await p.evaluate(
        () => document.querySelector('#jspsych-content')?.innerHTML?.length ?? 0);
      if (contentLength === 0) throw new Error('Final summary screen rendered no content (blank page)');

      // Closes a SECOND coverage gap this scenario used to stop short of:
      // reaching the final summary only proved the screen renders, not that
      // a session actually finishes. Clicking through to the "Thank you!"
      // end screen and its button is what triggers finishSession()
      // (finish-session.js) -- the single place that calls
      // jatos.endStudy/endStudyAndRedirect. Every real save/redirect bug in
      // this project's history (the submitResultData/endStudy signature
      // bug, the double-save bug, the same-origin-redirect-after-
      // session-close bug) was found only by a live MindProbe pilot run,
      // never by this suite, precisely because nothing automated ever
      // reached this far -- see CLAUDE.md's "Exit/redirect and
      // data-saving architecture".
      const before = snapshotResultFiles();
      await p.click('#next-btn');
      await waitForScreen(p, 'end', 5000);
      await p.click('body[data-screen="end"] button');
      await waitForSaveConfirmation(p);
      const savedFile = await waitForNewResultFile(before);
      const saved = JSON.parse(fs.readFileSync(path.join(RESULTS_DIR, savedFile), 'utf8'));
      if (!Array.isArray(saved) || saved.length === 0) {
        throw new Error(`Saved result file ${savedFile} was empty or not an array`);
      }
      const hasObservationRows = saved.some((row) => row.screen === 'observation');
      if (!hasObservationRows) {
        throw new Error(`Saved result file ${savedFile} had no observation rows`);
      }
      fs.unlinkSync(path.join(RESULTS_DIR, savedFile));  // keep dev-results/ clean
    },
  },
  {
    // Prolific-branch coverage: finishSession()'s isProlific branch (real
    // redirect to app.prolific.com, see finish-session.js) had ZERO
    // automated coverage before this -- every other scenario runs without
    // ?PROLIFIC_PID, so only the non-Prolific "no redirect, DOM update in
    // place" branch was ever exercised. Reuses the early-exit/terminated
    // path (cheap: 3 timeouts) rather than "Completes all trials" (45
    // interactions) -- both exit paths call the exact same finishSession(),
    // differing only in WHICH prolificCode argument gets passed
    // (completion vs earlyExit), so this cheaper path exercises the same
    // isProlific branch. app.prolific.com is intercepted via page.route so
    // no real external network request is ever made; the intercepted
    // request URL is checked against the expected per-task earlyExit code
    // parsed straight out of timeline-builder.js's PROLIFIC_CODES.
    name: 'Prolific participant: early-exit redirects to app.prolific.com with correct code',
    url: testUrlProlific,
    fn: async (p, task) => {
      let capturedUrl = null;
      await p.route('https://app.prolific.com/**', (route) => {
        capturedUrl = route.request().url();
        route.fulfill({ status: 200, contentType: 'text/plain', body: 'OK' });
      });

      await doConsent(p); await doTutorial(p);
      await letObsTimeOutAndRepeat(p);                     // 1st timeout
      await waitForScreen(p, 'observation', T_OBS_MS + 5000);
      await letObsTimeOutAndRepeat(p);                     // 2nd timeout
      await waitForScreen(p, 'observation', T_OBS_MS + 5000);
      await waitForScreen(p, 'terminated', T_OBS_MS + 8000);

      const before = snapshotResultFiles();
      await p.click('#early-exit-btn');

      // jatos-shim.js's endStudyAndRedirect saves BEFORE navigating (await
      // saveData(data) precedes window.location.href = url), so a result
      // file should still land here even though this branch never shows
      // finishSession's "Session complete" DOM confirmation (that message
      // is non-Prolific only -- see finish-session.js).
      const savedFile = await waitForNewResultFile(before);
      fs.unlinkSync(path.join(RESULTS_DIR, savedFile));  // keep dev-results/ clean

      // Poll rather than page.waitForURL(): the navigation is a plain
      // window.location.href assignment fulfilled by the route above, and
      // capturedUrl (set synchronously inside the route handler) is the
      // more direct signal that the redirect was actually attempted at all.
      const start = Date.now();
      while (!capturedUrl && Date.now() - start < 5000) {
        await new Promise((r) => setTimeout(r, 100));
      }
      if (!capturedUrl) throw new Error('No redirect to app.prolific.com was ever made');

      const expectedCode = expectedProlificCode(task, 'earlyExit');
      if (!capturedUrl.includes(`cc=${expectedCode}`)) {
        throw new Error(`Redirect URL had wrong code: ${capturedUrl} (expected cc=${expectedCode})`);
      }
    },
  },
  {
    name: 'Pool assignment: deterministic, valid range, embedded fields present',
    fn: async (p, task) => {
      const runToEarlyExitAndGetPoolIndex = async (pid) => {
        await p.goto(testUrlWithPid(task, pid));
        await doConsent(p); await doTutorial(p);
        await letObsTimeOutAndRepeat(p);
        await waitForScreen(p, 'observation', T_OBS_MS + 5000);
        await letObsTimeOutAndRepeat(p);
        await waitForScreen(p, 'observation', T_OBS_MS + 5000);
        await waitForScreen(p, 'terminated', T_OBS_MS + 8000);
        const before = snapshotResultFiles();
        await p.click('#early-exit-btn');
        const savedFile = await waitForNewResultFile(before);
        const saved = JSON.parse(fs.readFileSync(path.join(RESULTS_DIR, savedFile), 'utf8'));
        fs.unlinkSync(path.join(RESULTS_DIR, savedFile));

        const poolIndices = new Set(saved.map((row) => row.pool_index));
        if (poolIndices.size !== 1) {
          throw new Error(`Expected one pool_index across all rows, got: ${[...poolIndices]}`);
        }
        const poolIndex = [...poolIndices][0];
        if (typeof poolIndex !== 'number' || poolIndex < 0 || poolIndex >= 200) {
          throw new Error(`pool_index out of expected range [0,200): ${poolIndex}`);
        }

        const obsRows = saved.filter((row) => row.screen === 'observation');
        if (obsRows.length === 0) throw new Error('No observation rows to check embedded fields on');
        for (const row of obsRows) {
          if (row.value === undefined) throw new Error('observation row missing value field');
          if (task === 'continuous' && row.true_mean === undefined) {
            throw new Error('observation row missing true_mean field');
          }
          if (task === 'binary' && row.true_p === undefined) {
            throw new Error('observation row missing true_p field');
          }
        }
        return poolIndex;
      };

      const idxA = await runToEarlyExitAndGetPoolIndex('e2e_pool_test_participant_a');
      const idxB = await runToEarlyExitAndGetPoolIndex('e2e_pool_test_participant_b');
      if (idxA === idxB) {
        console.log(`  NOTE: both test participants hashed to the same pool_index (${idxA}) -- `
                    + `possible by chance (1/200), only concerning if this recurs.`);
      }

      const idxA2 = await runToEarlyExitAndGetPoolIndex('e2e_pool_test_participant_a');
      if (idxA2 !== idxA) {
        throw new Error(`Same participant ID gave different pool_index across runs: ${idxA} vs ${idxA2}`);
      }
    },
  },
];

// ── Runner ────────────────────────────────────────────────────────────────────
let passed = 0, failed = 0;
const failures = [];

async function runSuite(browserName, task) {
  const engine = ENGINES[browserName];
  console.log(`\n### ${browserName} / ${task} ###`);
  for (const { name, fn, url = testUrl } of SCENARIOS) {
    // Fresh browser PER SCENARIO, not just a fresh page within one shared
    // browser for the whole task -- found via a real flake while adding the
    // "completes all trials" scenario below: that scenario (45 real
    // interactions, far heavier than the others) would intermittently hit
    // "Target page, context or browser has been closed" specifically when
    // run as the LAST of several scenarios sharing one browser instance,
    // but never failed running alone in a fresh browser with identical
    // timing/logic -- pointing at accumulated browser-process instability
    // across sequential scenarios, not an app bug (confirmed separately:
    // the same flow via a standalone script completed cleanly with zero
    // errors). Costs a bit more time (one extra browser launch per
    // scenario) for meaningfully better isolation.
    const browser = await engine.launch();
    const page = await browser.newPage();
    page.setDefaultTimeout(15000);
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    console.log('--- ' + name + ' ---');
    try {
      await page.goto(url(task));
      await fn(page, task);
      // Any uncaught page error (e.g. a plugin throwing on a missing
      // required parameter) previously only got logged, never actually
      // failed the test -- a scenario whose fn() doesn't happen to wait on
      // anything past the crash point could report PASS regardless. This
      // was a real gap: it's what let the final-trial true_p bug (see the
      // dedicated scenario above) go undetected even when this suite was
      // passing 30/30. Treat any page error as a failure unconditionally.
      if (errs.length) throw new Error('Uncaught page error(s): ' + errs.join('; '));
      console.log('  PASS');
      passed++;
    } catch (e) {
      console.log('  FAIL: ' + e.message);
      if (errs.length) console.log('  JS errors: ' + errs.join('; '));
      failed++;
      failures.push(`${browserName} / ${task} / ${name}: ${e.message}`);
    } finally {
      await page.close();
      await browser.close();
    }
  }
}

console.log('Starting Vite dev server...');
const devServer = await startDevServer();
console.log(`Dev server ready on :${PORT}`);

console.log('Starting result server (mimics jatos.submitResultData)...');
const resultServer = await startResultServer();
console.log(`Result server ready on :${RESULT_PORT}\n`);

try {
  for (const browserName of BROWSERS) {
    for (const task of TASKS) {
      await runSuite(browserName, task);
    }
  }
} finally {
  stopDevServer(devServer);
  stopResultServer(resultServer);
}

console.log('\n' + '='.repeat(40));
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failures.length) {
  console.log('\nFailures:');
  failures.forEach(f => console.log('  - ' + f));
}
if (failed > 0) process.exit(1);
