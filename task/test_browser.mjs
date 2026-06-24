/**
 * test_browser.mjs
 * Playwright end-to-end tests using true timings (7s obs, 5s BTI, 1s ITI).
 * Run: node test_browser.mjs
 */
import { chromium }      from 'playwright';
import http              from 'http';
import fs                from 'fs';
import path              from 'path';
import { execSync }      from 'child_process';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST      = path.join(__dirname, 'dist-binary-test');

// ── Build test version (same as production — no timing changes) ───────────────
console.log('Building test version...');
const viteConf = path.join(__dirname, 'vite.config.js');
const origVite = fs.readFileSync(viteConf, 'utf8');
try {
  fs.writeFileSync(viteConf, origVite.replace("outDir: 'dist-binary'", "outDir: 'dist-binary-test'"));
  execSync('npm run build:binary', { cwd: __dirname, stdio: 'pipe' });
  console.log('Build done.\n');
} finally {
  fs.writeFileSync(viteConf, origVite);
}

// ── Static file server ────────────────────────────────────────────────────────
const server = http.createServer((req, res) => {
  let rel  = req.url.split('?')[0];
  if (rel === '/') rel = '/index-binary.html';
  const file = path.join(DIST, rel);
  if (!fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
  const ext  = path.extname(file);
  const mime = {'.html':'text/html','.js':'application/javascript','.css':'text/css'}[ext]||'application/octet-stream';
  res.writeHead(200, {'Content-Type': mime});
  fs.createReadStream(file).pipe(res);
});
await new Promise(r => server.listen(7654, r));

// ── Test runner ───────────────────────────────────────────────────────────────
const browser = await chromium.launch({ headless: true });
let passed = 0, failed = 0;

async function test(name, fn) {
  const page = await browser.newPage();
  page.setDefaultTimeout(30000);
  const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  console.log('\n--- ' + name + ' ---');
  try {
    await fn(page);
    console.log('  PASS');
    passed++;
  } catch (e) {
    console.log('  FAIL: ' + e.message);
    if (errs.length) console.log('  JS errors: ' + errs.join('; '));
    failed++;
  } finally {
    await page.close();
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const goto  = (p) => p.goto('http://localhost:7654/');
const wait  = (p, ms) => p.waitForTimeout(ms);
const has   = async (p, t) => (await p.textContent('body')).includes(t);

const doConsent = async (p) => {
  await p.waitForSelector('#consent-checkbox');
  await p.click('#consent-checkbox');
  await p.waitForSelector('#consent-btn:not([disabled])');
  await p.click('#consent-btn');
};

const moveSlider = async (p, pct) => {
  await p.waitForSelector('#response-slider');
  const box = await p.$eval('#response-slider', el => {
    const r = el.getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width, height: r.height };
  });
  const x = box.x + (pct / 100) * box.width;
  const y = box.y + box.height / 2;
  await p.mouse.move(x, y);
  await p.mouse.down();
  await wait(p, 50);
  await p.mouse.up();
  await wait(p, 100);
};

const submit = async (p) => {
  await p.waitForSelector('#submit-btn:not([disabled])');
  await p.click('#submit-btn');
};

// Tutorial takes ~7s (obs 0 clock) + 4×1s ITI + 4 obs + demo clock
// With true timings this is slow — skip to main experiment via timeout on obs 0
const doTutorial = async (p) => {
  // Tutorial obs 0: click boxes then respond
  await p.waitForSelector('#tut-box-0');
  await p.click('#tut-box-0');
  await p.click('#tut-box-1');
  await p.click('#tut-box-2');
  await p.waitForSelector('#response-slider');
  await moveSlider(p, 60);
  await submit(p);
  // Practice obs 1-4
  for (let i = 0; i < 4; i++) {
    await p.waitForSelector('#response-slider', { timeout: 10000 });
    await moveSlider(p, 50);
    await submit(p);
  }
  // Practice summary
  await p.waitForSelector('#proceed-btn');
  await p.click('#proceed-btn');
  // Timeout demo — wait for clock (7s) then proceed
  await p.waitForSelector('#demo-next-btn', { timeout: 10000 });
  await p.click('#demo-next-btn');
  await p.waitForSelector('#demo-proceed-btn');
  await p.click('#demo-proceed-btn');
  // BTI (5s)
  await p.waitForSelector('#response-slider', { timeout: 10000 });
};

// ── Tests ─────────────────────────────────────────────────────────────────────

await test('Normal submit: no too-slow, ITI appears', async (p) => {
  await goto(p); await doConsent(p); await doTutorial(p);
  await moveSlider(p, 65);
  await submit(p);
  await wait(p, 500);
  if (await has(p, 'Too slow')) throw new Error('"Too slow" after normal submit');
  await p.waitForSelector('#iti-canvas, #response-slider', { timeout: 3000 });
});

await test('Timeout: too-slow shows 2 remaining', async (p) => {
  await goto(p); await doConsent(p); await doTutorial(p);
  // Let obs 0 time out (7s)
  await p.waitForTimeout(8000);
  await p.waitForFunction(() => document.body.textContent.includes('Too slow'), { timeout: 3000 });
  if (!await has(p, '2 timeouts remaining')) throw new Error('Expected "2 timeouts remaining"');
});

await test('1 timeout remaining: shows "1 timeout remaining" not "last chance"', async (p) => {
  await goto(p); await doConsent(p); await doTutorial(p);
  // Time out twice
  for (let i = 0; i < 2; i++) {
    await p.waitForTimeout(8000);                          // obs clock
    await p.waitForTimeout(3000);                          // too-slow screen
    await p.waitForSelector('#iti-canvas', { timeout: 3000 });  // replay ITI
    await p.waitForTimeout(1500);                          // ITI clock
  }
  // Now at 3rd attempt — should see "1 timeout remaining" on 2nd too-slow
  // Actually after 2 timeouts we're on attempt 3, last chance = 1 remaining
  if (await has(p, 'last chance')) throw new Error('"last chance" should not appear');
  if (!await has(p, '1 timeout remaining')) throw new Error('Expected "1 timeout remaining"');
});

await test('3 timeouts: terminated screen with button', async (p) => {
  await goto(p); await doConsent(p); await doTutorial(p);
  for (let i = 0; i < 3; i++) {
    await p.waitForTimeout(8000);
    if (i < 2) {
      await p.waitForTimeout(3000);  // too-slow
      await p.waitForSelector('#iti-canvas', { timeout: 3000 });
      await p.waitForTimeout(1500);
    }
  }
  await p.waitForFunction(() =>
    document.body.textContent.includes('Session terminated'), { timeout: 5000 });
  if (!await has(p, 'Return to Prolific')) throw new Error('No "Return to Prolific" button');
  if (await has(p, 'Trial summary'))       throw new Error('Summary shown after termination');
});

// ── Results ───────────────────────────────────────────────────────────────────
await browser.close();
server.close();
try { fs.rmSync(DIST, { recursive: true, force: true }); } catch(e) {}
console.log('\n' + '='.repeat(40));
console.log('Results: ' + passed + ' passed, ' + failed + ' failed');
if (failed > 0) process.exit(1);
