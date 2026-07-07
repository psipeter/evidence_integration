/**
 * test_browser.mjs
 * Playwright end-to-end tests for both tasks (continuous + binary) across
 * Chromium, Firefox, and WebKit.
 *
 * Non-destructive: no source files are patched. Fast timings/skip-tutorial
 * are requested via URL query params on index-dev.html (see the "URL param
 * overrides" block on that file), which the dev server reads at runtime.
 * This runs against the real Vite dev server serving the actual task code —
 * nothing is rebuilt or rewritten on disk, so a crash mid-run can't corrupt
 * source files.
 *
 * Screen transitions are detected via `body[data-screen="..."]`, set by a
 * `on_trial_start` hook in timeline-builder.js (harmless in production —
 * just a DOM attribute). This avoids racing against guessed sleep durations,
 * which is what caused early flakiness/hangs while building this harness:
 * a screen with no visible text (e.g. the ITI clock, which is canvas-only)
 * looks identical to "stuck" if you're only checking textContent, so tests
 * must wait for the actual screen to change, not for a fixed amount of time
 * to pass.
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

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT      = 7655;
const BASE_URL  = `http://localhost:${PORT}`;

const ENGINES = { chromium, firefox, webkit };

// ── CLI filters ──────────────────────────────────────────────────────────────
const argTask    = process.argv.find(a => a.startsWith('--task='))?.split('=')[1];
const argBrowser = process.argv.find(a => a.startsWith('--browser='))?.split('=')[1];

const TASKS    = argTask    ? [argTask]    : ['binary', 'continuous'];
const BROWSERS = argBrowser ? [argBrowser] : ['chromium', 'firefox', 'webkit'];

// Fast-test overrides applied via URL params (see index-dev.html).
// obs timeout 1500ms, BTI 400ms, ITI 400ms, no tutorial, 3 trials.
// trials/btiMs/itiMs/tObsMs are free-form overrides (NOT limited to the
// dev-page's button-group presets) — see index-dev.html for details.
const T_OBS_MS = 1500;
const testUrl = (task) =>
  `${BASE_URL}/index-dev.html?task=${task}&tObsMs=${T_OBS_MS}&btiMs=400&itiMs=400` +
  `&tutorial=false&trials=3&autostart=1`;

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

const doConsent = async (p) => {
  await p.waitForSelector('#reveal-box-0');
  for (const id of ['reveal-box-0', 'reveal-box-1']) {
    await p.click(`#${id}`);
    await wait(p, 80);
  }
  await p.fill('#pilot-name', 'TestUser');
  await p.evaluate(() =>
    document.getElementById('pilot-name')
      .dispatchEvent(new Event('input', { bubbles: true })));
  await wait(p, 80);
  await p.click('#consent-checkbox');
  await wait(p, 80);
  await p.waitForSelector('#consent-btn:not(.consent-btn-locked)');
  await p.click('#consent-btn');
};

// Tutorial is skipped via ?tutorial=false — next real screen is the first
// observation ('observation' screen, set by build-trial-timeline.js).
const doTutorial = async (p) => {
  await waitForScreen(p, 'observation', 10000);
  await p.waitForSelector('#response-slider');
};

const moveSlider = async (p, pct) => {
  const box = await p.$eval('#response-slider', el => {
    const r = el.getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width };
  });
  const x = box.x + (pct / 100) * box.width;
  await p.mouse.move(x, box.y + 10);
  await p.mouse.down();
  await wait(p, 50);
  await p.mouse.up();
  await wait(p, 100);
};

const submit = async (p) => {
  await p.waitForSelector('#submit-btn:not([disabled])');
  await p.click('#submit-btn');
};

// Let an observation time out without responding, then land on the
// "too slow" replay screen (data-screen="iti_replay").
const letObsTimeOut = (p) => waitForScreen(p, 'iti_replay', T_OBS_MS + 5000);

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
      await letObsTimeOut(p);                              // 1st timeout
      await waitForScreen(p, 'observation', T_OBS_MS + 5000);  // loop_function replays same obs
      await letObsTimeOut(p);                              // 2nd timeout — 1 of 3 remaining
      await waitForTimeoutsRemaining(p, 1);
    },
  },
  {
    name: '3 timeouts: session terminated, no summary',
    fn: async (p) => {
      await doConsent(p); await doTutorial(p);
      await letObsTimeOut(p);                              // 1st timeout
      await waitForScreen(p, 'observation', T_OBS_MS + 5000);
      await letObsTimeOut(p);                              // 2nd timeout
      await waitForScreen(p, 'observation', T_OBS_MS + 5000);
      // 3rd timeout exhausts the budget and triggers earlyExit(), a manual
      // DOM injection (not a jsPsych trial) that sets data-screen='terminated'
      // by hand (see create-early-exit.js) for exactly this reason.
      await waitForScreen(p, 'terminated', T_OBS_MS + 8000);
      if (!await hasEl(p, '#early-exit-btn')) throw new Error('No early-exit button');
      if (await hasEl(p, '#summary-svg'))     throw new Error('Summary shown after termination');
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
];

// ── Runner ────────────────────────────────────────────────────────────────────
let passed = 0, failed = 0;
const failures = [];

async function runSuite(browserName, task) {
  const engine  = ENGINES[browserName];
  const browser = await engine.launch();
  console.log(`\n### ${browserName} / ${task} ###`);
  for (const { name, fn } of SCENARIOS) {
    const page = await browser.newPage();
    page.setDefaultTimeout(15000);
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    console.log('--- ' + name + ' ---');
    try {
      await page.goto(testUrl(task));
      await fn(page);
      console.log('  PASS');
      passed++;
    } catch (e) {
      console.log('  FAIL: ' + e.message);
      if (errs.length) console.log('  JS errors: ' + errs.join('; '));
      failed++;
      failures.push(`${browserName} / ${task} / ${name}: ${e.message}`);
    } finally {
      await page.close();
    }
  }
  await browser.close();
}

console.log('Starting Vite dev server...');
const devServer = await startDevServer();
console.log(`Dev server ready on :${PORT}\n`);

try {
  for (const browserName of BROWSERS) {
    for (const task of TASKS) {
      await runSuite(browserName, task);
    }
  }
} finally {
  stopDevServer(devServer);
}

console.log('\n' + '='.repeat(40));
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failures.length) {
  console.log('\nFailures:');
  failures.forEach(f => console.log('  - ' + f));
}
if (failed > 0) process.exit(1);
