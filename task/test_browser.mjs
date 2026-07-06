/**
 * test_browser.mjs
 * Playwright end-to-end tests for the binary task.
 * Patches config for fast obs timeout (1500ms) and builds to dist-binary-test.
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
const PORT      = 7655;

// ── Patch config for fast testing and build ───────────────────────────────────
console.log('Building test version...');
const configPath = path.join(__dirname, 'src/binary/config.js');
const viteConf   = path.join(__dirname, 'vite.config.js');
const tbPath     = path.join(__dirname, 'src/shared/timeline-builder.js');
const origConfig = fs.readFileSync(configPath, 'utf8');
const origVite   = fs.readFileSync(viteConf, 'utf8');
const origTb     = fs.readFileSync(tbPath, 'utf8');
try {
  // Short obs timeout + fast BTI + TEST_MODE
  fs.writeFileSync(configPath, origConfig
    .replace('const T_OBS_MS           = 7000;', 'const T_OBS_MS           = 1500;')
    .replace('const BTI_MS             = TEST_MODE ? 500  : 3000;', 'const BTI_MS             = 500;')
    .replace('const TEST_MODE              = false;', 'const TEST_MODE              = true;')
  );
  // Skip tutorial always in test mode
  fs.writeFileSync(tbPath, origTb.replace(
    'const skipTutorial = !showTutorial;',
    'const skipTutorial = true; // test build: always skip'
  ));
  // Build to separate dir
  fs.writeFileSync(viteConf, origVite.replace(
    "outDir: 'dist-binary',",
    "outDir: 'dist-binary-test',"
  ));
  execSync('npm run build:binary', { cwd: __dirname, stdio: 'pipe' });
  console.log('Build done.\n');
} finally {
  fs.writeFileSync(configPath, origConfig);
  fs.writeFileSync(viteConf, origVite);
  fs.writeFileSync(tbPath, origTb);
}

// Verify dist exists
if (!fs.existsSync(path.join(DIST, 'index-binary.html'))) {
  console.error('Build failed: index-binary.html not found in', DIST);
  process.exit(1);
}

// ── Static file server ────────────────────────────────────────────────────────
const server = http.createServer((req, res) => {
  let rel  = req.url.split('?')[0];
  if (rel === '/') rel = '/index-binary.html';
  const file = path.join(DIST, rel);
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.writeHead(404); res.end(); return;
  }
  const ext  = path.extname(file);
  const mime = { '.html':'text/html', '.js':'application/javascript',
                 '.css':'text/css', '.json':'application/json' }[ext] || 'application/octet-stream';
  res.writeHead(200, { 'Content-Type': mime });
  fs.createReadStream(file).pipe(res);
});
await new Promise(r => server.listen(PORT, r));
console.log(`Server on :${PORT}\n`);

// ── Test runner ───────────────────────────────────────────────────────────────
const browser = await chromium.launch({ headless: true });
let passed = 0, failed = 0;

async function test(name, fn) {
  const page = await browser.newPage();
  page.setDefaultTimeout(15000);
  const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  console.log('--- ' + name + ' ---');
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
const goto = (p) => p.goto(`http://localhost:${PORT}/`);
const wait = (p, ms) => p.waitForTimeout(ms);
const has  = async (p, t) => (await p.textContent('body')).includes(t);

const doConsent = async (p) => {
  await p.waitForSelector('#reveal-box-0');
  for (const id of ['reveal-box-0', 'reveal-box-1', 'reveal-box-2']) {
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
  await p.waitForSelector('#consent-btn:not([disabled])');
  await p.click('#consent-btn');
};

const doTutorial = async (p) => {
  // Tutorial skipped — wait through 500ms BTI then first obs
  await wait(p, 700);
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

// ── Tests ─────────────────────────────────────────────────────────────────────

await test('Normal submit: no too-slow, ITI appears', async (p) => {
  await goto(p); await doConsent(p); await doTutorial(p);
  await moveSlider(p, 65);
  await submit(p);
  await wait(p, 300);
  if (await has(p, 'Too slow')) throw new Error('"Too slow" after normal submit');
  await p.waitForSelector('#iti-canvas, #response-slider', { timeout: 3000 });
});

await test('Timeout: too-slow shows 2 remaining', async (p) => {
  await goto(p); await doConsent(p); await doTutorial(p);
  await wait(p, 2000);  // 1500ms obs clock + buffer
  await p.waitForFunction(() => document.body.textContent.includes('Too slow'), { timeout: 2000 });
  if (!await has(p, '2 timeouts remaining')) throw new Error('Expected "2 timeouts remaining"');
});

await test('1 timeout remaining text correct', async (p) => {
  await goto(p); await doConsent(p); await doTutorial(p);
  for (let i = 0; i < 2; i++) {
    await wait(p, 2000);                                              // obs timeout
    await wait(p, 3500);                                              // too-slow screen
    await p.waitForSelector('#iti-canvas', { timeout: 2000 });       // replay ITI
    await wait(p, 1200);                                              // ITI clock
  }
  if (await has(p, 'last chance'))         throw new Error('"last chance" should not appear');
  if (!await has(p, '1 timeout remaining')) throw new Error('Expected "1 timeout remaining"');
});

await test('3 timeouts: session terminated, no summary', async (p) => {
  await goto(p); await doConsent(p); await doTutorial(p);
  for (let i = 0; i < 3; i++) {
    await wait(p, 2000);
    if (i < 2) {
      await wait(p, 3500);
      await p.waitForSelector('#iti-canvas', { timeout: 2000 });
      await wait(p, 1200);
    }
  }
  await wait(p, 3500);  // final too-slow + terminated
  await p.waitForFunction(() =>
    document.body.textContent.includes('Session terminated'), { timeout: 5000 });
  if (!await has(p, 'Return to Prolific')) throw new Error('No "Return to Prolific" button');
  if (await has(p, 'Trial summary'))       throw new Error('Summary shown after termination');
});

await test('Submit then continue to next obs', async (p) => {
  await goto(p); await doConsent(p); await doTutorial(p);
  await moveSlider(p, 50);
  await submit(p);
  await p.waitForSelector('#iti-canvas', { timeout: 3000 });
  await wait(p, 1200);
  await p.waitForSelector('#response-slider', { timeout: 5000 });
  if (await has(p, 'Too slow')) throw new Error('Unexpected too-slow after submit');
});

// ── Results ───────────────────────────────────────────────────────────────────
await browser.close();
server.close();
try { fs.rmSync(DIST, { recursive: true, force: true }); } catch(e) {}

console.log('\n' + '='.repeat(40));
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
