/**
 * dev-server.js
 * Tiny local server that mimics JATOS's submitResultData endpoint.
 * Saves each submission as a JSON file in dev-results/.
 *
 * Usage: node dev-server.js
 * Runs on port 3099 (separate from Vite's port 5173).
 */

const express = require('express');
const cors    = require('cors');
const fs      = require('fs');
const path    = require('path');

const app     = express();
const PORT    = 3099;
const OUT_DIR = path.join(__dirname, 'dev-results');

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.text({ limit: '10mb' }));

fs.mkdirSync(OUT_DIR, { recursive: true });

app.post('/submit', (req, res) => {
  const data      = typeof req.body === 'string' ? req.body : JSON.stringify(req.body);
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename  = `result_${timestamp}.json`;
  const filepath  = path.join(OUT_DIR, filename);

  fs.writeFileSync(filepath, data, 'utf8');
  console.log(`Saved: ${filename}  (${data.length} bytes)`);
  res.json({ success: true, file: filename });
});

app.get('/health', (req, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`Dev result server running at http://localhost:${PORT}`);
  console.log(`Results saved to: ${OUT_DIR}`);
});
