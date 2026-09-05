/**
 * test-harness.js
 * Test-ONLY entry point (index-test.html) -- NEVER linked from production
 * code, NEVER included in any build (vite.config.js's rollupOptions.input
 * only lists index-continuous.html/index-binary.html, so this file and
 * index-test.html are automatically excluded from any production build
 * with no extra care needed). Real participants can never reach or
 * discover this.
 *
 * Calls the exact same buildAndRun() production uses -- no override logic
 * lives inside buildAndRun/timeline-builder.js/config-base.js itself; this
 * file just builds a slightly modified CONFIG OBJECT (fewer trials, faster
 * timing) using only field assignment / array slicing that already exists
 * on the config, then hands it to the same production code path. This
 * keeps the thing under test identical to production -- same plugins, same
 * DOM interactions, same real Chromium/Firefox/WebKit rendering via
 * Playwright -- only the numbers differ.
 *
 * Tutorial is always run in full (no skip option) -- tutorial screens have
 * no response deadline at all (see build-tutorial-timeline.js), so skipping
 * it was never about avoiding a timer, just a few extra clicks. Running it
 * for real means automated tests actually exercise those screens across
 * browsers, instead of never touching them at all.
 *
 * URL params (all optional):
 *   task    'continuous' | 'binary'          (default: continuous)
 *   trials  number of trials to run          (default: 3, sliced from the
 *                                              front of the real sequences)
 *   tObsMs  observation response deadline, ms (default: real config value)
 *   btiMs   between-trial interval, ms        (default: real config value)
 *   itiMs   between-observation ITI, ms -- overwrites every sliced trial's
 *           own iti_ms in the MAIN task, and the tutorial's fixed
 *           itiShortMs (both default to their real config values, i.e. no
 *           override, if this param is absent)
 */
import './shared/jatos-shim.js';
import { config as configContinuous } from './continuous/config.js';
import { config as configBinary }     from './binary/config.js';
import { buildAndRun }                from './shared/timeline-builder.js';

const qp = new URLSearchParams(window.location.search);

const task       = qp.get('task') === 'binary' ? 'binary' : 'continuous';
const baseConfig = task === 'binary' ? configBinary : configContinuous;

const nTrials  = qp.has('trials') ? parseInt(qp.get('trials'), 10) : 3;
// Slice EVERY pool member down to nTrials (not just one) and pass the
// full pool through -- lets buildAndRun's real poolIndexForParticipant
// hash pick a member, same as production, per an explicit decision that
// testing should exercise the actual pool-assignment path rather than
// hardcoding a single member. test_browser.mjs's fixed fake PROLIFIC_PID
// (e2e_test_pid) means this always resolves to the same member across runs.
const sequencesPool = baseConfig.sequencesPool.map(pool => pool.slice(0, nTrials));

const itiMsOverride = qp.has('itiMs') ? parseInt(qp.get('itiMs'), 10) : null;
if (itiMsOverride != null) {
  // Safe to mutate: config-base.js's buildConfig() already produced fresh
  // per-trial objects via {...s} for every page load, not references into
  // the raw imported JSON -- this can't leak between test runs or corrupt
  // the module-level config.
  for (const pool of sequencesPool) {
    for (const seq of pool) seq.iti_ms = itiMsOverride;
  }
}

buildAndRun({
  ...baseConfig,
  sequencesPool,
  tObsMs:     qp.has('tObsMs') ? parseInt(qp.get('tObsMs'), 10) : baseConfig.tObsMs,
  btiMs:      qp.has('btiMs')  ? parseInt(qp.get('btiMs'), 10)  : baseConfig.btiMs,
  itiShortMs: itiMsOverride != null ? itiMsOverride : baseConfig.itiShortMs,
});
