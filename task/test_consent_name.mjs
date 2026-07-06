/**
 * test_consent_name.mjs
 * Verifies that pilot name entered on consent page is captured correctly.
 */
import { chromium } from 'playwright';
import * as http from 'http';
import * as fs   from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST      = path.join(__dirname, 'dist-binary');

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
await new Promise(r => server.listen(7789, r));

const browser = await chromium.launch({ headless: true });
const page    = await browser.newPage();
const errors  = [];
page.on('pageerror', e => errors.push(e.message));

// Intercept jatos.endStudy to capture final data
await page.addInitScript(() => {
  window._capturedData = null;
  // Stub jatos so the experiment runs without JATOS
  window.jatos = {
    studySessionData: {},
    onLoad: (fn) => fn(),
    startComponentByTitle: () => {},
    endStudy: (json) => { window._capturedData = json; },
    endStudyAndRedirect: (url, json) => { window._capturedData = json; },
  };
});

await page.goto('http://localhost:7789/');
await page.waitForTimeout(1500);

// Reveal both boxes
for (const id of ['reveal-box-0', 'reveal-box-1']) {
  await page.click(`#${id}`).catch(() => {});
  await page.waitForTimeout(150);
}

// Enter name and fire input event
await page.fill('#pilot-name', 'TestPilot');
await page.evaluate(() => {
  document.getElementById('pilot-name')
    .dispatchEvent(new Event('input', { bubbles: true }));
});
await page.waitForTimeout(100);

const nameCapture = await page.evaluate(() => window._pilotNameCapture);
console.log('1. _pilotNameCapture after typing:', nameCapture);

// Tick checkbox
await page.click('#consent-checkbox');
await page.waitForTimeout(100);

const btnEnabled = !(await page.$eval('#consent-btn', b => b.classList.contains('consent-btn-locked')));
console.log('2. Begin button looks ready:', btnEnabled);

// Click Begin — on_finish should fire, read _pilotNameCapture, null it out
await page.click('#consent-btn');
await page.waitForTimeout(500);

// _pilotNameCapture should be null (consumed by on_finish)
const afterFinish = await page.evaluate(() => window._pilotNameCapture);
console.log('3. _pilotNameCapture after on_finish (should be null):', afterFinish);

console.log('   Page errors:', errors.length ? errors : 'none');

const pass = nameCapture === 'TestPilot' && btnEnabled && afterFinish === null;
console.log('\n' + (pass
  ? '✓ PASS: name captured and consumed by on_finish'
  : '✗ FAIL'));

await browser.close();
server.close();
