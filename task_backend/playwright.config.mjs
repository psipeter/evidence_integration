// playwright.config.mjs
// Real, persistent, re-runnable test suite -- replaces the one-off
// smoke_test_*.mjs scripts used throughout development (each written,
// run once, then deleted). Runs against the REAL deployed Supabase
// backend, not a mock -- deliberately, since that's exactly what caught
// the real bugs during development (a missing GRANT, a camelCase/
// snake_case field-name mismatch). A separate test-only Supabase project
// would be the more "correct" isolation, but isn't worth the added
// infrastructure at this project's size; disciplined test-PID naming
// (see tests/helpers.mjs's testPid) is enough.
//
// Dedicated ports (5183/5184), NOT this project's own default dev ports
// (5173/5174) -- those have been persistently occupied throughout
// development by long-running task/ (JATOS pipeline) dev servers on this
// machine. Using the same ports here risked Playwright's webServer
// either colliding or, worse, silently treating task/'s unrelated server
// as if it were this one.
//
// Dev servers run against a SMALL (2-trial) sequence variant, not the
// real 32-trial production pool -- a full genuine session (every
// observation driven through real UI, no artificial seeding) takes
// seconds/minutes instead of ~15-30 minutes this way. See
// generate_sequences.py's --name flag and src/numbers|colors/config.js's
// VITE_SEQUENCES_VARIANT handling. The variant file is gitignored (see
// .gitignore's own comment for why), so it's generated here on demand if
// missing -- the suite is self-sufficient on a fresh checkout, no manual
// setup step required.
import { defineConfig } from '@playwright/test';
import { readFileSync, existsSync } from 'fs';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Local-only secret key for test cleanup (see tests/helpers.mjs's
// cleanupTestRows). NEVER the app's own .env -- this is a separate,
// gitignored file holding the SECRET key (bypasses RLS), not the
// publishable key that's safe to ship in a browser bundle. Tests still
// run and verify real behavior without this file; they just skip
// cleanup and print a warning instead. See .env.test.example.
const envTestPath = resolve(__dirname, '.env.test');
if (existsSync(envTestPath)) {
  for (const line of readFileSync(envTestPath, 'utf-8').split('\n')) {
    const match = line.match(/^([A-Z_]+)=(.*)$/);
    if (match) process.env[match[1]] = match[2];
  }
}

export const NUMBERS_PORT = 5183;
export const COLORS_PORT = 5184;
export const SEQUENCES_VARIANT = 'test2trial';

const numbersVariantPath = resolve(__dirname, `sequences_numbers_${SEQUENCES_VARIANT}.json`);
const colorsVariantPath = resolve(__dirname, `sequences_colors_${SEQUENCES_VARIANT}.json`);
if (!existsSync(numbersVariantPath) || !existsSync(colorsVariantPath)) {
  console.log(`Generating the "${SEQUENCES_VARIANT}" sequence variant (missing -- gitignored, built on demand)...`);
  execSync(`python3 generate_sequences.py --task both --n_pool 200 --name ${SEQUENCES_VARIANT}`, {
    cwd: __dirname, stdio: 'inherit',
  });
}

export default defineConfig({
  testDir: './tests',
  // Tests write real rows to a shared real backend -- running them in
  // parallel would interleave confusingly in failure output for little
  // gain at this suite's size.
  fullyParallel: false,
  workers: 1,
  // No auto-retry: a flaky pass would hide a real bug in exactly the
  // kind of timing-sensitive logic (timeouts, retries, resume) this
  // suite exists to verify.
  retries: 0,
  reporter: 'list',
  timeout: 90_000,
  webServer: [
    {
      command: `npx vite --mode numbers --port ${NUMBERS_PORT} --strictPort`,
      port: NUMBERS_PORT,
      reuseExistingServer: false,
      cwd: __dirname,
      timeout: 30_000,
      env: { VITE_SEQUENCES_VARIANT: SEQUENCES_VARIANT },
    },
    {
      command: `npx vite --mode colors --port ${COLORS_PORT} --strictPort`,
      port: COLORS_PORT,
      reuseExistingServer: false,
      cwd: __dirname,
      timeout: 30_000,
      env: { VITE_SEQUENCES_VARIANT: SEQUENCES_VARIANT },
    },
  ],
});
