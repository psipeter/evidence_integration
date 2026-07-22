# CLAUDE.md — evidence_integration

This file is the source of truth for Claude when working on this project.
Read it fully before making any changes or suggestions. Prefer this file over
README.md when they conflict.

**After any conversation compaction**: re-read this file in full before doing
anything else. Compaction summaries omit conventions. Key ones to remember:
- **Tool routing (critical)**: `str_replace`, `create_file`, and `view` operate
  on Claude's own local sandbox container, NOT this remote host. Only
  `filesystem:read_text_file` / `filesystem:write_file` / `filesystem:edit_file`
  and `shell:run_command` touch actual project files here. Using the wrong
  tool silently "succeeds" (writes to a local copy that looks identical) with
  no error — this caused a real bug where several edits to
  generate_sequences_momentmatch.py appeared to work but never reached the
  remote file, and the mismatch wasn't caught until a CLI flag failed at
  runtime. Always use the filesystem/shell MCP tools for anything under
  /home/psipeter/evidence_integration/; if unsure whether an edit landed,
  grep the remote file for the new content before trusting it.
- Figures save as PDF only — never convert to PNG/SVG or upload images to chat.
  This also applies to task/ UI screenshots: don't upload Playwright screenshots
  to context to verify UI work — use DOM/computed-style assertions instead
  (textContent, getComputedStyle, attribute checks) and only pull an actual
  image when a genuine visual judgment call is needed.
- Run node test_browser.mjs only after big task/ changes or when explicitly asked —
  it's slow (3 browsers × 2 tasks); don't run it reflexively after every small edit
- **Before running test_browser.mjs (or any other slow E2E verification), ask
  Peter first** — don't run it automatically just because a task/ change was
  made. Small/localized changes often don't warrant the full 6-call E2E pass;
  let him decide whether verification is warranted and at what scope (e.g. a
  single browser/task combo vs. the full matrix) before spending that time.
- **When Claude runs test_browser.mjs via shell:run_command**: run it as 6 SEPARATE
  calls, one per browser/task combination (--browser=X --task=Y), never as one
  call for the full matrix. A single full-matrix call (node test_browser.mjs with
  no filters) exceeded the shell tool's own response window and returned a
  timeout error to Claude with no result, even though the test process itself
  was still running fine on the remote host — the tool times out well before
  the full ~2-3+ min suite finishes. Report each of the 6 results (browser,
  task, pass/fail counts) separately as they complete, so the person can track
  how long the run is taking. If two calls are issued back-to-back, a brief
  port-release race on 7655/3099 between them can cause a spurious
  connection-refused failure — check `lsof -ti:7655` / `lsof -ti:3099` are both
  empty and simply retry that one combination rather than assuming an app bug.
- All NEF simulation data → data/runs/; figures → figures/

---

## Scientific goals

This project studies **how people integrate sequential noisy evidence**, using
cognitive models and a biophysical spiking neural network (NEF) to identify the
computational and neural mechanisms underlying that process.

### Goal 1 — Cross-task generalisation of cognitive mechanisms
The NEF model must capture human behaviour across multiple tasks (carrabin and
yoo), demonstrating that the same underlying cognitive algorithm generalises
across task domains without task-specific modification. The NEF is benchmarked
against a spectrum of established models from the evidence-integration
literature: an optimal Bayesian integrator (Mean), a leaky integrator
(LeakyIntegrator), and primacy/recency weighting models (PrimacyRecency). On
trial-wise RMSE the expected ordering is:

    task-specific model ≈ NEF > LeakyIntegrator ≥ PrimacyRecency ≥ Mean (optimal)

The NEF need not outperform task-specific models; comparable RMSE combined with
cross-task generalisability is the target.

### Goal 2 — Emergent higher-order behavioural signatures
Beyond RMSE, the NEF must reproduce secondary behavioural phenomena that it was
not explicitly trained to capture. These signatures should emerge naturally from
the model's spiking dynamics: temporal update patterns, the decay of response
change across the observation sequence, individual differences in discounting
rate (λ), test-retest reliability of noise and decay-rate metrics, and
state-persistent response variability. The fact that these emerge without being
directly optimised is the key scientific contribution.

### Goal 3 — Joint behavioural and neural predictions
The NEF generates both behavioural and neural predictions simultaneously from
the same underlying mechanism. Behavioural: response trajectories, update
magnitudes, individual λ and α₀. Neural: error-ensemble activity, prediction-
error dynamics, and how both scale with architectural parameters (n_neurons,
α₀, λ). Together these constitute a mechanistically coherent account that is
testable at multiple levels of analysis.

### Goal 4 — Novel testable predictions
Spiking noise produces state-persistent variability that differs qualitatively
from response noise; this prediction distinguishes the NEF from NoisyCounting
even when both achieve similar RMSE. Response variability and PE variability
scale with n_neurons and α₀; neural activity profiles match known ensemble
dynamics. These are quantitative predictions for future empirical work.

---

## Metric taxonomy (PTN framework)

All analyses and figure panels are organised under three groups.
Figures save PDF only (no PNG/SVG).

### P — Performance
Metrics measuring how well participants and models do on the task.

| Code | Metric | Carrabin | Yoo |
|------|--------|----------|-----|
| P1 | Estimation error: RMSE to hidden probability / true mean, per pid boxplot | Y | Y |
| P2 | Model fit: RMSE to human responses, per pid boxplot | Y | Y |

Carrabin: true_p converted as true_p*2-1 to match response scale [-1,1].
Yoo: true_mean = cumulative mean of value stream (expanding mean per trial).

### T — Temporal
Metrics capturing within-sequence dynamics.

| Code | Metric | Carrabin | Yoo |
|------|--------|----------|-----|
| T1 | Task performance vs observation: RMSE per obs position | Y | Y |
| T2 | Response change vs observation: mean |Δresponse| per obs | Y | Y |
| T3 | Split-half reliability of λ: first vs second half of trials | N | Y |
| T4 | Dynamical model fit: λ_model vs λ_human regplot (individual differences) | N | Y |
| T5 | Residual variance growth across obs (state noise accumulation) | Y | N |
| T6 | Within-trial residual autocorrelation decay (state persistence) | Y | N |

T3–T4 require enough trials and observations to fit λ reliably.
λ is fit via curve_fit: A·n^(-λ), bounds [0,2], on mean |Δresponse| curve.
T5–T6 use residuals = response − mean(response | pid, obs, qid); require qid.

### N — Neural
NEF predictions; testable in future empirical experiments.

| Code | Metric | Carrabin | Yoo |
|------|--------|----------|-----|
| N1 | Decoded PE timecourse within observation window | Y | Y |
| N2 | PE variability vs response variability, probe sims; partial-r control for α₀ | Y | N |
| N3 | Response and PE variability vs fitted α₀ (shared scaling factor) | Y | N |
| N4 | Response and PE variability vs n_neurons scan | Y | N |
| N5 | Error-ensemble weight-neuron activity vs observation, split by λ group | N | Y |
| N6 | Mean weight-neuron activity vs mean |Δresponse| across observations | N | Y |
| N7 | Fitted λ mediates activity change and mean |Δresponse| (twin-axis) | N | Y |
| N8 | Late |Δresponse| vs late estimation error (last 10 obs) | N | Y |

---

## Central cognitive model

Updates follow a power-law decaying learning rate:

    alpha(t) = alpha_0 / t^lambda

High lambda: steep discounting (primacy-like). Low lambda: slow discounting
(recency-like). In the NEF, alpha(t) is an emergent property of the spiking
dynamics rather than a hardcoded equation — a counting subnetwork tracks the
observation index and decodes the appropriate weight, which then gates the
error signal driving the value ensemble. This is analogous to RL_lambda (which
implements the same equation explicitly) but arises from biophysical dynamics.

---

## Active datasets

| Name | N | Task |
|------|---|------|
| carrabin | 21 | Binary inputs; slider after each of 5 obs; sequences repeat (qid); true_p known |
| yoo | 38 | Continuous inputs; slider; 30 obs × 30 trials; no sequence repetition |

Pickles: data/carrabin.pkl, data/yoo.pkl.
Required columns: pid, trial, observation, value, response.
Carrabin adds: qid, true_p (from carrabin_original.csv).

New task (task/): Two online experiments deployed on Prolific via MindProbe/JATOS.
- Continuous task: Normal(mean, std) stimulus; slider response [0–100]; 8x4=32 trials × 15 obs
- Binary task: Bernoulli(p) stimulus (blue/red circle); slider response [0–100%]; 8x4=32 trials × 15 obs
Both tasks share all infrastructure (jsPsych 8, Vite 6, shared plugins/CSS).
Timeout system: 3 timeouts per trial; timeout → too-slow screen → replay; exhausted → terminated screen.
Naming convention (IMPORTANT): every file/class that has a continuous/binary pair
  uses an explicit -continuous/-binary suffix on BOTH sides (e.g.
  plugin-observation-continuous.js / plugin-observation-binary.js). Never leave one
  side unsuffixed as an implicit default — that drifted into real inconsistency
  once (plugin-observation.js vs -binary.js, draw-performance.js vs bar-chart.js)
  and was cleaned up in a dedicated pass; don't reintroduce it.
Key files: timeline-builder.js (orchestrator), build-trial-timeline.js (pure-JS
  trial loop), build-tutorial-timeline.js, build-welcome-screen.js, build-consent-screen.js,
  build-end-screen.js,
  create-early-exit.js, finish-session.js (shared "how does a session
  actually end" implementation, see "Exit/redirect and data-saving
  architecture" below), plugin-observation-continuous.js, plugin-observation-binary.js,
  observation-timeout-clock.js (shared countdown-clock renderer),
  slider-continuous.js, slider-binary.js, distribution-continuous.js (continuous
  tutorial SVG builder), urn-binary.js (binary tutorial SVG builder),
  continuous-draw-animation.js / binary-draw-animation.js (tutorial bubbling
  animations, see below).
Data pipeline: JATOS JSON → task/parse_results.py → data/task_results.pkl
Consent form: verbatim IRB text from task/consent_form.txt — do not paraphrase or edit
  (note: not literally read from that file at build time -- build-consent-screen.js
  hand-transcribes the same text inline, confirmed matching; consent_form.txt is a
  reference copy, not an active import).
Pilot name field: REMOVED (was PILOT ONLY -- saved a manually-typed name as a
  prolific_pid substitute for distinguishing pilot participants by eye). Removed
  from build-consent-screen.js (HTML + all JS wiring: isReady()'s name check,
  the name-capture listener, on_finish's pilot_name assignment) and
  parse_results.py (the pilot_name capture/fallback in prolific_pid). Safe to
  remove because prolific_pid was already independently set via
  jsPsych.data.addProperties in timeline-builder.js (participantId = prolificPID
  ?? pilot_${jatos.workerId}), applied retroactively to every row including
  observation rows -- the name field was never the actual source of prolific_pid,
  just a manual convenience for a human skimming pilot data. test_consent_name.mjs
  (which exclusively tested this now-removed feature) moved to _trash/.
Target: ~50–80 participants per task, within-subject (both tasks per participant).
See task/ section in README.md for full details.

Current task status (as of latest session):
- No TEST_MODE/N_TRIALS_TO_RUN toggle anymore (removed along with index-dev.html
  -- see "Task testing architecture" below). Trial count is fully implicit from
  however many trials task/sequences/{task}_sequences.json contains (currently
  32, the 8x4 hybrid production set -- see "Sequence design" below). BTI_MS=3000ms,
  DISTRACTOR_TYPE='none' (config-base.js DEFAULTS). SLIDER_DEFAULT is now
  'last' (changed from 'none' this session -- see "Tutorial redesign, bonus/
  error system, and binary no-prefix sequences" below): the slider thumb
  visibly starts at the participant's previous response instead of hidden
  until first touch, to help them remember their running estimate. The
  floating numeric label is decoupled from thumb position and still always
  starts hidden regardless of mode -- see that section for the full
  mechanism and the real submit-enable bug this change surfaced (now fixed
  in both tasks).
- Welcome/title screen (build-welcome-screen.js) is now the first screen shown,
  before consent -- see "Task testing architecture" below.

Stable architecture (established, do not regress):
- jsPsych plugin conventions: see "jsPsych 8 plugin conventions" section below.
  Pattern A (no timeout) vs Pattern B (has timeout) are the only two shapes;
  trial() must never be async (see that section for why — this was a real,
  previously undetected production bug that likely corrupted some pilot data).
- "practice" → "tutorial" rename is complete throughout task/src (files,
  classes, info.name, data.screen, config keys, CSS classes) — zero "practice"
  references remain (verified by grep). Tutorial observation plugins have NO
  timeout clock, deliberately; only plugin-timeout-demo.js shows a countdown
  during the tutorial (it exists specifically to demonstrate the real deadline).
- Naming convention: every continuous/binary file pair has an explicit suffix
  on both sides (see "Naming convention" above) — don't reintroduce an
  unsuffixed default on either side.
- Shared modules to reuse, not reimplement: observation-timeout-clock.js
  (countdown-ring canvas renderer, used by both observation plugins AND
  plugin-timeout-demo.js), distribution-continuous.js / urn-binary.js (SVG
  builders with a `revealed` boolean flag), binary-draw-animation.js /
  continuous-draw-animation.js (tutorial bubbling animations — see below).
- Old/removed files live under _trash/ pending your `git rm` — this directory
  accumulates across sessions and is never committed; clear it out periodically.

Consent screen (build-consent-screen.js):
- 3 boxes: a blue payment-motivation box first ("You will be paid $5.00 for
  finishing and up to $5.00 based on your performance" -- updated this
  session from an earlier flat "$8.00 - 10.00 based on your performance";
  NOT a warning, distinguished via .consent-info-box-blue), followed by the
  two original red warning boxes. The data-loss warning box was trimmed
  this session to just "Do not close, refresh, or navigate away during the
  task." (removed a trailing "-- your data will be lost and you will not be
  paid" clause). Stacked vertically, ordered disclosure (each box locked
  with a "· · ·" placeholder until the one before it is revealed, mirroring
  the tutorial-intro pattern). The checkbox section (no longer a
  name+checkbox section -- see "Pilot name field" above) stays behind its
  own "· · ·" placeholder until all three boxes are done.
- "Begin experiment" does NOT use the native `disabled` attribute — disabled
  buttons never dispatch `click` at all, which silently ate premature clicks
  with zero feedback (a real pilot complaint) and in one case the disabled
  *look* apparently didn't render (likely a browser/extension override
  neutralizing color-based disabled styling — a reminder not to rely on
  color/opacity alone for a state that matters). Instead: `.consent-btn-locked`
  is a plain CSS class for the look, and a capturing-phase click listener on
  an ancestor intercepts the click before jsPsych's own bubble-phase listener
  (attached directly to the button during trial() setup) ever fires — if
  requirements aren't met it's stopped silently (no popup message; an earlier
  version had one but it shifted the layout when shown).
- Layout-shift note: reveal boxes and the checkbox section reserve their
  final height from the start via `visibility:hidden` (not `display:none`) on
  the real content, with the placeholder absolutely-positioned on top — the
  same pattern used in the tutorial-intro plugins. `display:none` removes an
  element from layout, so revealing it later changes the container's height
  and shifts everything below; this bit both the consent screen and both
  tutorial-intro plugins before being fixed.

Binary slider (slider-binary.js):
- Gradient track shows the blue/red split, but ONLY after first interaction —
  the `.slider-unset` state overrides it with flat gray. An earlier revision
  tried showing the gradient from first render (at a default 50/50 split) to
  help participants guess which side is which, but pre-filling any specific
  split — even a neutral one — creates a prior/anchor before the participant
  has made a judgment. Reverted; the value-free cue is the ruler instead.
- Ruler: 5 tick marks, 2 stacked rows below the slider — row 1 shows all 5
  blue values (0,25,...,100), row 2 shows the mirrored red values
  (100,75,...,0), same x-positions, all numbers the same fixed size (1.3rem).
  Earlier versions tried directional text labels ("← more red"/"more blue
  →") and font-size-scaled-by-value pairs on one row — both were reported
  confusing; a two-row axis-style layout is a legend, not an implied answer,
  so it doesn't create a prior the way pre-filling the track did.

Binary + continuous tutorials (plugin-tutorial-intro-*.js /
plugin-tutorial-observation-*.js):
- Both intro plugins now separate the image reveal from box 1's text into its
  own click-to-reveal step (box 1 text → image box → box 2 goal text → box 3
  slider instructions → slider), so participants aren't reading and watching
  an animation at the same moment. NOTE: this restructuring was applied to
  BOTH continuous and binary (originally binary-only, then ported to
  continuous alongside the falling-bubble animation below) — both are now in
  sync structurally.
- Bubbling draw animations (binary-draw-animation.js / continuous-draw-
  animation.js, same overall structure, ~1050ms bubble phase + 1000ms fade):
  binary bubbles rise inside the blue/red bar; continuous bubbles fall
  downward from under the Gaussian curve to the x-axis, x-position weighted
  by the density via rejection sampling (so they cluster near the mean).
  Continuous bubbles have a gray stroke (`#94a3b8`) — plain white had low
  contrast against the curve's pale green fill.
  Binary: the CENTRE circle is the persistent empty ring (white/gray border)
  visible throughout bubbling; the circle above the bar stays fully invisible
  until it pops in and fades to color with the centre circle at resolve —
  this was deliberately swapped from an earlier version where the roles were
  reversed, per explicit design feedback.
  Continuous: the centre NUMBER fades in (opacity 0→1) at resolve, together
  with the #tut-svg-obs marker (which always starts at opacity 0 regardless
  of the shared `revealed` flag — its reveal is owned exclusively by the
  animation, mirroring how urn-binary.js's draw circle is independent of its
  shared `revealed` flag too).
  A critical bug was found and fixed here: resolveDraw() must clear the
  bubble-spawn `setInterval`, not just cancel the aging `requestAnimationFrame`
  loop — otherwise bubbles keep spawning into a dead loop forever after each
  draw resolves (confirmed 30+ orphaned bubbles accumulating within 2s in one
  case). Both animation modules do this correctly now; if a third one is ever
  added, copy this carefully.
- Yellow warning-style caption box (`#fffbeb`/`#fbbf24`, same colors as
  plugin-timeout-demo.js's own warning box) below the image on both tasks,
  appearing alongside box 2 (goal text) — discloses that the
  bar/curve visualization is tutorial-only and won't appear in the real task.
- Box copy: all 3 boxes follow a source → goal → response-mapping structure
  ("you'll see a sequence of X" / "your goal is to estimate Y" / "move the
  slider to Z"). Uses "sequence" (not "series") throughout, matching the BTI
  screen's wording (see below). "probability"/"distribution" are always green
  (DIST_COLOR); "hidden" itself is never colored. "balls"/"numbers" are
  colored to match their task's sample color. Binary intentionally uses
  "balls" language (reintroducing the urn metaphor at the TEXT level, even
  though the dot-grid VISUAL metaphor was removed — a deliberate choice to
  keep "sequence" meaningful, not an oversight).

Main-task fades (plugin-observation-binary.js / plugin-observation-continuous.js
/ plugin-timeout-demo.js): each real observation's circle/number, and the
timeout-demo's illustrative circle/number, fade in on render (1000ms) —
binary circle: white→color; continuous number: opacity 0→1. Purely cosmetic,
mirrors the tutorial's animations for a consistent feel; never gates the
timeout clock or slider, which start immediately regardless.

BTI screen (plugin-inter-trial.js): both tasks now show the same "generating
new sequence…" label (continuous used to say "generating new distribution…").
"Sequence" describes the observable data stream and stays accurate regardless
of whether the hidden generative parameter happens to repeat from an earlier
trial (the task reuses 6 unique parameter sets across 24 trials) — "new
distribution" would overclaim novelty it can't always guarantee.

- Distractor system exists (iti_condition per trial, popup/iti_length/none) but
  currently disabled (DISTRACTOR_TYPE='none'). Ready to reactivate.

### Tutorial redesign, bonus/error system, and binary no-prefix sequences
(this session -- large scope, summarized here; individual files' own
docstrings have the full blow-by-blow if something here needs more detail)

**Tutorial redesign (both tasks, now in sync structurally)**:
- Full N_OBS_TO_RUN-length tutorial (15 observations, not a separate
  hardcoded 5) for both tasks -- matches real-task length exactly now.
- New tutorial-tracker.js: a 15-slot progress row between the distribution
  figure and (formerly) its caption, showing the sequence's accumulating
  history. Continuous renders it as numbers on underlines (settled/
  current/empty states, distinguished by opacity/underline weight, not
  color -- an earlier circle-based design was replaced). Binary renders it
  as colored dots instead (`renderDot:true`, a per-value color FUNCTION
  passed as `color` instead of a fixed string) -- there's no separate
  "number" to show for a blue/red draw, the color IS the content.
- The old static yellow "this graphic won't appear in the real task"
  caption box (below the figure, both tasks) is GONE, replaced by a
  **three-phase system in the top-right box** (BOX0B's slot in continuous;
  a similarly-repurposed slot in binary), driven purely by `obs_num`:
  - Phase A (obs 1-5): default text, white/plain.
  - Phase C (obs 6-10): "...mean of all numbers in this sequence"
    (continuous) / "...ratio over all balls in the sequence" (binary) goal
    reminder -- YELLOW (`.tutorial-notify-yellow`). The tracker below is
    ALSO highlighted in that same yellow during this phase
    (`.tutorial-tracker-highlight`, a dedicated class -- combining
    `.tutorial-info-block` + `.tutorial-notify-yellow` directly broke the
    tracker's own tight spacing/underline-touching at the enlarged number
    size, a real bug not just a hypothetical one).
  - Phase D (obs 11-15): "you will not see these graphics, use your
    memory" warning (`RECAP_TEXT_1`/`RECAP_TEXT_2` in tutorial-text-
    {continuous,binary}.js) -- RED (`.tutorial-notify-red`). The figure and
    tracker are ALSO hidden behind an opaque red overlay
    (`.tutorial-hidden-overlay`) during this phase -- the real elements
    still render underneath, completely unmodified, just visually covered.
  A dedicated post-summary "recap" screen (plugin-tutorial-recap-
  continuous.js) was built, then DELETED once phase D started covering the
  identical ground during the tutorial observations themselves -- a
  separate screen for it became redundant. Continuous's `BOX0` was split
  into `BOX0`/`BOX0B` (second sentence moved to the top-right slot);
  binary's `BOX0` was NOT split the same way (its right-column slot needed
  new default content anyway) -- see tutorial-text-binary.js's own
  docstring for why that's a deliberate difference, not an inconsistency.
- Terminology: continuous's BOX1/BOX2 now use "mean" for the estimation
  target (unchanged). Binary's now deliberately splits "probability"
  (describing the hidden GENERATIVE parameter -- BOX0B, RECAP_TEXT_1) from
  "ratio" (describing what the participant estimates FROM the observed
  balls -- BOX1, BOX2, phase C's reminder, RECAP_TEXT_2's tail) -- not an
  inconsistency, a real terminological distinction (see the extended chat
  discussion on running-mean/running-ratio vs. true-mean/true-probability
  objectives, summarized below).
- Real bug found and fixed in BOTH tasks' tutorial-observation plugins: the
  draw animation's `onComplete` was force-enabling Submit as soon as the
  animation finished whenever `slider_default` was `'last'`, letting a
  participant submit their PREVIOUS response unchanged without ever
  touching the slider on the new observation. Removed -- Submit now only
  enables via genuine slider interaction in every mode, matching how the
  real (non-tutorial) observation plugins already worked. Fixing this
  properly required adding an `onReveal` callback (fired the INSTANT a
  fade BEGINS, not once it's finished) to both continuous-draw-
  animation.js and binary-draw-animation.js, since the tracker's current-
  slot reveal needed to sync with that moment, not `onComplete`'s later one.
- slider-continuous.js / slider-binary.js: the floating numeric label (and
  binary's in-bar %s) are now decoupled from thumb position -- both always
  start hidden regardless of `unset`/`'last'` mode, only appearing on the
  participant's own first interaction with that observation's slider.
  Previously `'last'` mode revealed the exact number immediately alongside
  the thumb position, which defeated some of the point of showing the
  position without also showing the precise prior value. Binary's
  `'last'`-mode gradient fill was also changed from full-strength color to
  real alpha transparency (`rgba(...,0.4)`, not a blended-toward-gray solid
  color, which read as muddy) -- matches the thumb's own dimmed look.

**Per-observation error + per-trial bonus (bonus-continuous.js -- name is a
little dated now, used by both tasks; not renamed, to avoid a wide import-
path change for a pure naming concern)**:
- Error is computed ONCE per observation, at response time, and attached
  directly to that observation's own JATOS row (via jsPsych's
  `on_finish(data)` mutation) -- flows into JATOS via the same per-trial
  `on_trial_finish` append every other trial already uses. The trial-
  summary screen's `total_error` is the SUM of those already-stored
  per-observation errors, never recomputed independently -- one source of
  truth. `reward = max(0, 100 - BONUS_DECAY * totalError)`.
- **Real bug found and fixed**: `BONUS_DECAY` was a flat `1`, which
  implicitly treated `totalError` (a SUM across ~15 observations) as if it
  lived on the same 0-100 scale as a single observation's error -- reward
  hit exactly 0 once AVERAGE per-observation error exceeded ~6.67, a
  genuinely GOOD result, not a bad one. This is why bonus always showed 0c;
  the underlying reward really was 0 for nearly every real response, not a
  display bug. Fixed: `BONUS_DECAY = 1 / DEFAULTS.N_OBS_TO_RUN`, so the
  formula now behaves as `reward ≈ 100 - average_error_per_observation`.
- `ERROR_MODE` (config-base.js DEFAULTS, currently `'running_mean'` for
  continuous / `'running_p'` for binary -- set for TESTING, see below) --
  what a response's error is measured against:
  - `'true_mean'`/`'true_p'`: the trial's fixed generative parameter.
  - `'running_mean'`/`'running_p'`: a per-observation MOVING target -- the
    running mean/ratio of the raw observed values (not responses) up to
    and including that observation. `computeRunningMeans`/
    `computeRunningRatios` in bonus-continuous.js. Also changes the summary
    chart's per-row reference tick and the "true mean"/"running mean" (or
    "true probability"/"running ratio") wording in its legend and the
    tutorial-summary blue banner.
  Binary's own errorMode values (`'true_p'`/`'running_p'`) are DISTINCT
  strings from continuous's (`'true_mean'`/`'running_mean'`) -- each task's
  own config.js sets its own via config-base.js's `overrides` mechanism
  independently.
- **Why this toggle exists -- a real methodological question, not just an
  engineering one** (see chat history for the full literature-grounded
  discussion): asking participants to track the running mean/ratio of
  observed samples is a genuinely different cognitive task than asking them
  to infer the fixed underlying parameter -- the former is closer to a pure
  counting/working-memory task (Prat-Carrabin & Woodford's "imprecise
  counting" account: primacy/recency effects can emerge purely from
  uncertainty about *n*, with no deliberate evidence-weighting at all),
  the latter is a genuine inference problem under irreducible uncertainty.
  `ERROR_MODE` lets both be run and compared. Confirmed via `utils/
  binary_transform.py`: the Laplace-smoothing transform used elsewhere in
  the modeling pipeline (see "Carrabin response transform" section) IS
  exactly the correction that makes a running-mean-style model optimal for
  inferring the FIXED true_p -- it has no justification if the objective is
  the running ratio itself, which needs no prior-shrinkage at all.
  `scripts/inspect_sequences.py`'s `run_agents` now includes a "Running
  ratio (optimal, no Laplace)" agent for binary reflecting this.
- Summary charts (draw-performance-{continuous,binary}.js) redesigned:
  removed the old shared-reference-row-above-everything design (continuous
  had a Gaussian curve at the top; binary had a full-height true-p bar) in
  favor of a per-row reference tick + colored error-distance line, so every
  individual response's accuracy is visible against its own row. Continuous
  uses green for the error line, blue for the reference tick (matching its
  existing color conventions). Binary uses green for the reference tick
  (matching its pre-existing "True probability" legend convention) and a
  new VIOLET for the error line (blue/red were already taken by ball
  colors). Binary's estimate marker changed from a vertical tick to a black
  CIRCLE (matching continuous's own marker) specifically to reduce visual
  merging with the reference tick when they're close together.
  **Real bug found and fixed in binary's chart**: the old blue/red
  split-bar fill was fully opaque and drawn AFTER the (already-implemented)
  error line in the per-row loop -- completely hiding it. Not a missing
  feature, an occlusion bug. Removing the colored fill (replaced with a
  plain gray background matching continuous's own number-line style) fixed
  this as a direct side effect.
  A "Total error: X • Bonus: Y¢" box (green/gold text) now sits above the
  chart on both summary screens for both tasks.

**Binary sequence generation: new no-prefix branch, now in PRODUCTION**
(generate_sequences_hybrid.py, generate_sequences_pool.py):
- Root cause investigated and fixed: `build_binary_prefixes`'s composition
  allocator (`_allocate_binary_composition_counts`) is a pure DETERMINISTIC
  function of `n_prefix`/`prefix_length` (no RNG at all) -- every pool
  member got the EXACT SAME `{0:1,1:2,2:2,3:2,4:1}` blue-count split across
  its 8 prefixes, only the specific arrangements within each level varying.
  This collapsed the `|Δresponse|` curve's between-participant diversity on
  the prefix portion (obs ≤ 4) down to as few as 3 distinct per-member
  averages at obs=4 (confirmed via a real production plot, not a
  hypothetical). A randomized (still capacity-respecting) allocator alone
  raised that to 9; removing the prefix/qid-repeat concept ENTIRELY raised
  it further, to a smooth 17→23→27→33 growth across obs 2-5 with no
  collapse anywhere.
- **New branch**: `generate_binary_sequences_no_prefix` in
  generate_sequences_hybrid.py (`--no_prefix` flag, binary-only, asserts
  loudly if combined with `--task continuous/both`) -- a SEPARATE branch
  alongside the existing prefix/qid-repeat structure, NOT a replacement
  (the old scheme remains fully intact and re-runnable). Every trial gets
  its own independently-drawn `true_p` and its own independent exact-quota
  full-length sequence; `qid` = the trial's own index (never repeated),
  `prefix_length` recorded as 0. Wired through generate_sequences_pool.py
  the same way (`--no_prefix`, requires `--task binary`).
- **PRODUCTION SWITCH**: `task/sequences_pool/binary_*` was replaced with a
  freshly-generated 200-member no-prefix pool (same 8x4=32 trials/member as
  before, exact-quota verified, confirmed the diversity fix holds on the
  real production data). The OLD deterministic-prefix binary pool is
  backed up at `task/sequences_pool_binary_prefix_backup/` (gitignored
  alongside `task/sequences_pool/` itself, fully recoverable by re-running
  `generate_sequences_pool.py` WITHOUT `--no_prefix`). Continuous's pool
  was NOT touched.
- **Real methodological tradeoff, not free**: the old prefix/qid-repeat
  structure existed specifically to give a clean, controlled,
  equal-n comparison for response-variability/reliability metrics (same
  stimulus shown `n_repeats` times, letting you separate response noise
  from genuine stimulus-driven variation). The no-prefix branch removes
  this by construction. A per-pid investigation of the NEW pool found
  substantial NATURAL (accidental, uncontrolled) repetition still exists
  at the 4-observation level purely from the small combinatorial space
  (only 16 possible ±1 patterns of length 4): 198/200 pids have at least
  one 4-observation prefix with ≥4 accidental repeats (mean 2.48 such
  prefixes/pid); coverage at looser thresholds is substantial (86% of
  trials/pid belong to some ≥2-repeat group, 61% at ≥3, 36% at ≥4); these
  natural repeats are NOT clustered in time (mean trial-index spread of
  each pid's best repeat group: 23.1 out of a max possible 39). This makes
  an OPPORTUNISTIC (not by-design) version of the old variability analysis
  possible, just uncontrolled -- not yet decided whether this is
  sufficient, or whether a hybrid (no forced prefix composition, but a
  small number of DELIBERATE full-trial repeats inserted per participant)
  is worth building as a third branch. Full 15-observation trials
  essentially never repeat by chance under either scheme -- if a metric
  needs full-trial repetition specifically, neither current branch
  provides it.
- `scripts/inspect_sequences.py` / `scripts/inspect_iid_sequences.py` got
  several real, unrelated bug fixes while investigating this (see each
  file's own updated docstrings): a multi-agent comparison figure that was
  silently reading only ONE representative pool member instead of
  aggregating over all 200 (`run_agents_pooled`/`_plot_panel_pooled` now
  fix this, `--pool_dir` wired through to the figure itself, not just the
  CSV); TWO independent false-positive prefix-collision checks (one in
  generate_sequences_pool.py's `verify_pool`, one in inspect_sequences.py's
  `build_inspection_csv`) that didn't understand `prefix_length=0`; a
  missing 95% CI band in inspect_iid_sequences.py's per-participant figure.

**Pending / not yet done**: `task/evidence-integration-binary.jzip` is
STALE -- predates all of this session's binary tutorial/bonus/chart work
and needs a rebuild (`npm run build:binary` + `python task/
generate_jzip.py`) before any real pilot testing of these changes.
`evidence-integration-continuous.jzip` was rebuilt during this session and
is current.

Sequence design: **8x4 (32 trials) hybrid production design is CURRENT**,
  superseding the earlier 6x4 pure-momentmatch pilot described below.
  Generated via task/generate_sequences_hybrid.py (see "Sequence generation
  methods" below) -- binary keeps quota/momentmatch construction unchanged
  (no seed search); continuous uses a genuinely i.i.d., unrescaled suffix
  instead. Chosen at 8x4 rather than 10x4 specifically to reduce
  participant time. Promoted to task/sequences/{continuous,binary}_sequences.
  {pkl,json}; the older 6x4 sequences (and the pure _iid_/_momentmatch_
  branch outputs) remain recoverable via git history / their own
  differently-named files, not deleted. **See docs/sequence_design_open_
  questions.md for the full investigation and decision rationale** -- the
  quota-vs-i.i.d. confound this decision was based on, the hybrid design's
  own std-guard tuning, and everything checked before promoting.

  Historical context (6x4, momentmatch-only) -- kept for the mechanism
  detail, since the hybrid design reuses this construction UNCHANGED for
  binary and PARTIALLY for continuous (prefix/target generation + optimal
  matching, just not the suffix rescale):
  6x4 (24 trials) was generated via task/generate_sequences_momentmatch.py
  (1000-try isotonic seed search), std_fixed=15. Superseded the older
  seed=175/198 sequences (fully recoverable via git history).

  **Design (redesigned -- prefix identity and target level are now
  INDEPENDENT axes, not one qid = one fixed (prefix, target) pair)**:
  A real collision bug motivated this -- two DIFFERENT qids in an earlier
  version of the promoted sequences ended up with an IDENTICAL realized
  4-observation prefix purely by chance (binary's prefix_length=4 only
  allows 2**4=16 distinct sequences total, as few as 4 arrangements for a
  given exact quota, so collisions were structurally likely once enough
  qids shared a quota). See generate_sequences_momentmatch.py's module
  docstring ("Prefix generation" section) for the full mechanism; summary:
    - `--n_prefix=6` DISTINCT prefixes are generated first, independent of
      any target -- this is what "qid" and repeat structure track now
      (each repeated `--n_repeats=4` times -> 24 trials). Continuous:
      prefix centers spread evenly across --mean_range (NOT all centered on
      the range midpoint -- an earlier version did that and left extreme
      targets with no genuinely close prefix available, a supply problem
      no amount of matching cleverness could fix). Binary: compositions
      (blue-ball count out of 4) spread across the full range via the same
      linspace pattern, arrangements drawn WITHOUT replacement -- this is
      the actual fix for the collision bug above.
    - 24 TARGET values (true_mean for continuous, true_p for binary) are
      generated SEPARATELY, with NO forced repeat structure -- continuous
      gets 24 DISTINCT evenly-spaced means across --mean_range=[15,85]
      (changed from the old [10,90]); binary gets the FULL native integer
      blue-count granularity across --blue_range=[2,13] (every integer
      level 2..13, not an evenly-spaced subset the old --n_levels design
      used), distributed as evenly as possible (2 repeats each at n=24,
      12 levels -- divides evenly here, but the remainder-handling is
      general for cases where it doesn't).
    - Prefix-slots and target-slots are matched via a GLOBALLY OPTIMAL
      assignment (Hungarian algorithm, scipy.optimize.linear_sum_assignment,
      minimizing total mismatch) -- NOT a greedy heuristic, which was tried
      first and rejected after confirming empirically it can leave an
      arbitrarily bad single-pair mismatch (>40 points) from unlucky
      processing order, even fully greedy. Binary's exact quota-
      reachability constraint (an all-red prefix cannot reach a target near
      the top of blue_range) is enforced as effectively-infinite cost on
      infeasible pairs.
    - Suffix construction targets the algebraic RESIDUAL needed to bring
      the pooled (prefix+suffix) sequence to the trial's actual target, not
      the target directly -- necessary now that the prefix is generic and
      not already near the target. Exact for binary; continuous's pooled
      std runs slightly above std_fixed on average as a result (measured on
      the promoted pilot: mean achieved std ~14.6, max ~20.6 against a
      target of 15 -- see the boundary-bias table below for the separate,
      pre-existing extreme-mean bias this compounds with). **The hybrid
      design (current production) uses this same residual-mean centering
      for continuous but skips the rescale step entirely** -- see
      generate_sequences_hybrid.py's own module docstring for why (dropping
      the rescale, not the seed search, is what actually restores genuine
      trial-ending uncertainty) and its two variance guards (analytical
      bias correction + a loose +/-25% std safety-net rejection).

  **Consequence to know about, not a bug**: a given prefix's 4 repeats
  generally pair with DIFFERENT true_mean/true_p each time now. "qid
  repeats" means "same literal prefix shown multiple times", NOT "same
  hidden parameter shown multiple times" the way carrabin/yoo's qid works.
  config-base.js's `pickTutorialExample` may still assume the latter for
  some of its fallback logic -- flagged in "Sequences.json schema" below,
  not yet verified/fixed.

  Verified directly against the saved files before promoting (not just the
  generation script's own internal assertions): 6 distinct prefixes per
  task (zero collisions), exactly 4 repeats each, zero binary quota
  mismatches, true_mean spans exactly [15,85], true_p spans exactly
  [0.1333, 0.8667] using all 12 integer blue-count levels. Check via
  scripts/inspect_sequences.py, which now ALSO writes a human-readable,
  observation-level CSV (figures/inspect_sequences.csv by default,
  alongside the existing PDF) covering prefix/suffix structure, running
  trajectory, achieved-vs-target mean/std/p, and these same constraint
  checks -- see that script's own module docstring. It reads the literal
  {task}_sequences.* filenames from --seq_dir, so pointing it at a
  differently-named file (e.g. the *_momentmatch_sequences.* search output,
  before promoting) still needs the temp-dir copy trick described in
  "Inspect sequences" further below.

  For the full 10x4 (40 trials) experiment: NOT yet finalized, and the
  previously-found 10x4 candidate seeds (continuous seed=245, binary
  seed=68 -- see "Sequence generation methods" below) now predate BOTH the
  evenly-spaced/no-mirroring redesign AND this prefix/target-independence
  redesign -- they would need regenerating from scratch under the current
  script to be current, not just re-checked. PI decision on i.i.d. vs
  moment-matched is also still open -- see "Open items" below.

Single master copy in task/sequences/{task}_sequences.{pkl,json} --
  **this is the CANONICAL REFERENCE/promotion target, not what's actually
  served to real participants** -- see "Per-participant sequence pool"
  immediately below for what real participants get (one of 200 independent
  pool members, not this single file). This file remains the thing you
  regenerate/verify/promote when changing generation parameters; the pool
  is built FROM the same generator (generate_sequences_hybrid.py) with the
  same promoted parameters, just called 200 times instead of once.
task/src/{task}/config.js imports the POOL (task/sequences_pool/, via
  import.meta.glob) -- NOT task/sequences/{task}_sequences.json directly
  anymore. See "Per-participant sequence pool" below.

### Per-participant sequence pool (unique sequence per participant)

**Decided and built** (see chat history for the full design discussion):
each real participant gets ONE of 200 independently-generated hybrid
sequence sets (not the single shared file above), assigned via a
deterministic hash of their own participant ID -- no server-side
counter/database needed, and the same participant gets the same pool
index in BOTH tasks (same hash formula, not seeded by task name, given
equal pool sizes).

**Generation**: `task/generate_sequences_pool.py` wraps
`generate_task_sequences_hybrid` (generate_sequences_hybrid.py) -- 200
independent calls per task, same promoted parameters as the single
reference file (8 prefixes x 4 repeats, mean_range=[15,85]/blue_range=
[2,13], boundary_margin=1, std_tolerance_frac=0.25). Output:
`task/sequences_pool/{task}_{0000..0199}_sequences.{pkl,json}`. Verified:
200/200 members pass prefix uniqueness both tasks, 0 binary quota
mismatches across the whole pool. NOT committed to git (see .gitignore --
800 files is too much churn to track usefully; fully reproducible via
`python task/generate_sequences_pool.py --n_pool 200 --task both --seed 0`,
those being the script's own CLI defaults).

**Bundling**: `task/src/{continuous,binary}/config.js` uses
`import.meta.glob('../../sequences_pool/{task}_*_sequences.json', {eager:
true})` to statically bundle all 200 members into the build at compile
time -- chosen over a runtime `fetch()` of a pool directory after directly
measuring the cost (~150KB gzip for 200 members, smaller than this app's
own CSS) and weighing it against fetch's unverified reliance on JATOS
serving extra static assets correctly, a risk category this project has
hit real, costly surprises in before (see "Tab-visibility handling" and
"The save mechanism" sections). The tutorial's illustrative example
(`pickTutorialExample`) is derived from a fixed, arbitrary pool member
(index 0) -- the tutorial only needs ONE representative example, not
something tied to any participant's actual assignment.

**Assignment**: `timeline-builder.js`'s `poolIndexForParticipant` -- a
deterministic DJB2-style string hash of the participant's ID (real
PROLIFIC_PID, or the `pilot_${jatos.workerId}` fallback) into [0,
poolSize). `sequences = sequencesPool[poolIndex]`; `pool_index` recorded
via `jsPsych.data.addProperties` alongside the existing `prolific_pid`/
`task`, so it lands on every row. Pilots/local testing draw from the pool
too (decided explicitly), not a special-cased fixed file -- test-harness.js
slices every pool member down to the fast-test trial count and passes the
full pool through, so local/E2E tests exercise the real assignment path.

**Verified working end-to-end, not just locally**: confirmed via real
Prolific "preview as participant" across two different browsers (Chrome,
Firefox incognito) that the SAME underlying preview identity produces the
SAME pool assignment both times; confirmed via local dev with a
manually-supplied `?PROLIFIC_PID=` that DIFFERENT fixed IDs give
DIFFERENT, reload-stable pool assignments. (`npm run dev:continuous`
WITHOUT an explicit `?PROLIFIC_PID=` will look different on every reload --
this is expected, not a bug: with no ID in the URL, the local jatos-shim
falls back to `workerId: 'dev_' + Date.now()`, a fresh timestamp every
page load, which is a genuinely different identity each time -- not
comparable to a real, stable participant ID. Append a fixed
`?PROLIFIC_PID=your_test_id` to get reproducible local pool assignment.)

**Data schema impact**: value/true_mean/true_std/true_p/qid/pool_index are
now embedded DIRECTLY in every observation row (build-trial-timeline.js),
replacing a design that reconstructed them via a (task, trial) join
against the single shared file -- that join assumed every participant saw
identical content at a given trial index, which stopped being true once
pool assignment varies who sees what. See "Participant-data columns"
below (rewritten to match) and parse_results.py's own module docstring.

**Diagnostic tooling, pool-aware**:
  - `scripts/inspect_iid_sequences.py --sequence_type pool` loads every
    real `{task}_{NNNN}_sequences.pkl` under `--pool_dir` (default
    task/sequences_pool) and reuses the SAME multi-participant plotting/
    fitting pipeline the iid path uses -- reads the actual files real
    participants get assigned, not a fresh simulation. Verified: 192,000
    rows (200 members x 2 tasks x 32 trials x 15 obs), fitted lambda
    consistent with earlier single-file/simulated findings.
  - `scripts/inspect_sequences.py`'s `build_inspection_csv` gained an
    optional `pool_dir` param -- reads every pool member instead of the
    single file, adds a `pool_index` column, aggregates the prefix-
    uniqueness/quota checks per member into one summary line per task
    (since 200 separate OK lines wouldn't be readable). Verified: 12,800
    rows, 200/200 members passing all checks both tasks.

**What's NOT yet been directly observed** (an inference, not a gap in the
built mechanism): two genuinely different real Prolific participants (via
real recruitment, not preview) actually receiving different PROLIFIC_PID
values that both correctly flow through to different pool members. Every
check so far exercised either one real ID (preview) or manually-supplied
local IDs -- the remaining link is Prolific's own bedrock guarantee that
real participants get distinct IDs, not anything this app's code could
plausibly interfere with. Closeable only by an actual small pilot with 2+
real participants, comparing pool_index in their exported data afterward.

### Sequences.json schema, tutorial derivation, and participant-data columns

**sequences.json per-trial fields**: trial, qid, true_mean, true_std, true_p,
values, prefix_length, iti_ms, iti_condition (continuous has true_p=null;
binary has true_std=null). true_mean/true_std/true_p are NEEDED here —
they're the canonical generative ground truth, not re-derivable from a
small ~15-observation sample without loss (exactly the distinction between
the "vs true param" and "vs running mean" analysis branches — see
inspect_sequences.py's --gt_mode above). All three generation scripts now
write true_std (it existed internally in every script's templates dict
already, just wasn't serialized) — previously the browser's main-task
summary screens silently fell back to a stale hardcoded default (see plugin
conventions below) because seq.true_std was always undefined.

**Tutorial example** (task/src/shared/config-base.js's `pickTutorialExample`):
derives tutorialValues/tutorialMean/tutorialStd/true_p from a REAL trial in
sequencesData — picks the trial whose true_mean/true_p is closest to the
midpoint (50/0.5), subject to two checks on the first 5 values, falling
through to the next-closest candidate if the top match fails either:
  1. Spread (>=2 above/below the mean for continuous; >=2 of each color for
     binary) — guards against a first-5 slice that looks one-sided by chance
     even in a well-behaved full sequence.
  2. Directional consistency — the shown slice's OWN apparent direction
     (majority color for binary; which side of the midpoint the slice's own
     mean falls on for continuous) must match the true parameter's actual
     direction. Spread alone does NOT guarantee this: a real bug was found
     where the binary tutorial showed 3 blue/2 red in its first 5 balls
     while true_p=0.4 actually favored red — passed the spread check fine
     (>=2 of each color), but visually taught the opposite of the true
     direction, undermining the tutorial's whole point. Fixed by adding
     this check rather than hardcoding a specific example (which would
     reintroduce the exact drift problem pickTutorialExample exists to
     avoid — see below); confirmed the fix picks a genuinely consistent
     example for both tasks against the real production sequences (binary
     falls through past all of the closest qid's repeats, since every one
     of them happened to show a majority-mismatched slice, to the
     next-closest qid, which then shows the correct direction).
Replaced hand-picked literals (`tutorialValues = [48, 75, 38,
82, 57]`, `TUTORIAL_STD = 20`) that silently drifted out of sync the moment
sequences.json's actual std_fixed changed (this happened once already: the
tutorial kept showing std=20 after the pilot moved to std=15). Never
hand-pick a new tutorial example again — if the pedagogical example ever
needs different qualities, change pickTutorialExample's selection logic,
not the values.

**NOT YET VERIFIED against the prefix/target-independence redesign**: the
"falls through past all of the closest qid's repeats" fallback behavior
described above was verified back when a qid's repeats all shared the same
true_mean/true_p (the pre-redesign design). Under the current design a
qid's repeats generally have DIFFERENT targets (see "Sequence design"
above), so falling through a qid's "repeats" as a fallback candidate group
may no longer mean what it used to. This function operates on individual
TRIALS (not qids) for its primary closest-to-midpoint selection, so the
core logic is likely still fine -- but the fallback path hasn't been
re-checked against real production data since the redesign. Verify before
relying on it, don't assume it still holds.

**Participant-data columns** (parse_results.py, build-trial-timeline.js):
**REVISED** -- value/true_mean/true_std/true_p/qid/pool_index are now
recorded DIRECTLY on every observation row (alongside the genuinely
participant-generated fields: prolific_pid, task, trial, observation,
response, timed_out, rt, time_elapsed). This replaced an earlier design
where only the participant-generated fields were saved and everything
else was reconstructed via a join against sequences.json on (task, trial)
alone -- that join relied on every participant sharing exactly one file,
so (task, trial) alone determined `value`. That stopped being true once
per-participant pool assignment was introduced (see "Per-participant
sequence pool" above) -- (task, trial) is no longer enough; you'd also
need to know which of the 200 pool members that participant got. Rather
than doing a three-key join against the right pool member's file,
build-trial-timeline.js now records these fields directly, so every
participant's raw export is fully self-contained: no lookup, no join, no
dependency on the pool files still existing/matching later. `pool_index`
is kept specifically so which pool member a given participant saw is
always traceable even without the pool files on disk.
parse_results.py's `load_values_lookup` and the per-task merge/row-wise
value lookup were removed entirely (not just patched) -- net effect is a
simplification, not just a fix. Old (pre-pool) export files remain
parseable: pool_index comes back missing/NaN with a printed WARNING
rather than crashing.

### Sequence generation methods (task/)

Three separate scripts, kept deliberately distinct rather than one script with
a tunable knob. Each now carries a ROLE note at the top of its own module
docstring stating its current status -- check there first, this section
summarizes but the scripts themselves are the source of truth.

**task/generate_sequences.py** -- SHARED UTILITIES ONLY, not independently
  runnable. Its own generation method (rejection sampling: draw prefix/
  suffix freely, redraw whole blocks until the realized sequence passes a
  plausibility check) was REMOVED in a cleanup pass -- no CLI, no main(),
  nothing calls it directly anymore. It exists purely as an import target:
  generate_sequences_iid.py and generate_sequences_momentmatch.py both pull
  RNG/param-grid/observation-drawing/plausibility-checking/scoring helpers
  from here rather than duplicating them (make_rng, continuous_param_grid,
  binary_param_grid, mirror_sequence, mirror_params, draw_continuous_obs,
  draw_binary_obs, check_sequence_plausibility, _weighted_delta_score,
  _weighted_rmse_score, score_sequences, _bayesian_responses, _rl_responses,
  _save_sequences -- check both other scripts' imports before removing or
  renaming anything in this file). Do not reintroduce the rejection-sampling
  generation logic here; if it's ever needed again it's recoverable from git
  history. The rejection-sampling METHOD's own known problems (for context,
  since the method itself is still referenced in prose elsewhere): the joint
  multi-qid constraint (ALL qids must pass simultaneously in one draw) scaled
  very badly with qid count -- going from 6 to 10 qids collapsed the binary
  structural pass rate from ~12% to ~0% at k=0.5, ~6% at k=0.7 -- and at
  extreme means (e.g. mean=10/90 with std=15) the [0,100] bound truncates the
  achievable std so far below nominal that no amount of resampling passes a
  tight k (a structural mismatch between target and bound, not a sampling
  problem). **Key finding**: k-constrained rejection sampling and exact quota
  sampling are the SAME underlying object at different points on one
  continuum -- i.i.d. sampling conditioned on the final composition falling
  within k x SE of the target. There is no way to tighten k for smoothness
  without buying into finite-population predictability; they are the same
  lever.

**task/generate_sequences_iid.py** -- pure i.i.d. branch, one of two
  candidates still under consideration for the pending 10x4 full-experiment
  design (see "Open items" below) -- NOT current production, NOT dead code.
  Genuinely unconstrained sampling -- no k, no plausibility gate, no
  rejection loop, and deliberately NO seed search or best-of-N ranking
  either (any outcome-dependent seed selection is itself a form of
  conditioning, which would pull back toward the same finite-population
  structure this branch exists to avoid). Single draw, save, done. Matches
  the closest published precedent (Nassar/Behrens/Glaze-style predictive-
  inference tasks draw outcomes directly from the generative distribution
  with no correction). `--report` gives a diagnostic (realized vs target
  moments) that never feeds back into generation.

**task/generate_sequences_momentmatch.py** -- CURRENTLY ACTIVE / PRODUCTION
  method. The promoted 6x4 pilot was generated by this script. Constructs
  each block (prefix or suffix) to hit a target sample mean/std (continuous,
  via iterative rescale+clip) or exact blue/red quota (binary), then
  randomizes order/realization. No rejection loop -- resolves the
  mean=10/90+std=15 truncation problem directly (achieved std within ~0.5 of
  nominal even at the range edges, vs ~4.5 off under rejection sampling).
  Literature check: no support found for exact quota matching as a
  behaviorally-neutral stimulus-generation choice in the probability-
  learning/evidence-integration literature -- every precedent found
  (gambler's-fallacy, probability-matching studies) uses this kind of
  composition constraint as a deliberate, studied manipulation, not a
  neutral background choice. Real methodological tradeoff, not free.

  **Prefix identity and target level are independent axes** (the current
  design, redesigned from an earlier version where each qid meant one fixed
  (prefix, target) pair) -- see "Sequence design" above for the full
  mechanism and the collision bug this fixes, and the script's own module
  docstring ("Prefix generation" section) for the complete rationale. In
  brief: `--n_prefix` (default 6) distinct prefixes, independent of any
  target, each repeated `--n_repeats` (default 4) times; target values
  generated separately with no forced repeats (continuous: N distinct
  evenly-spaced means; binary: full native blue-count granularity,
  distributed as evenly as possible); matched via the Hungarian algorithm
  (scipy.optimize.linear_sum_assignment), not a greedy heuristic (greedy was
  tried and rejected -- see the script's docstring for why). Parameter
  levels/targets are an evenly-spaced grid or full native range, NOT
  random+mirrored -- mirroring's point was compute-saving under rejection
  sampling (no rejection loop here to save on) and guaranteeing symmetry
  (redundant once you specify an already-symmetric grid directly), so it was
  removed entirely; `mirror_sequence`/`mirror_params` still exist in
  generate_sequences.py purely because generate_sequences_iid.py still uses
  them.

  Seed search via `--score_mode {bump, isotonic}` (default: isotonic):
    - 'bump': original approach, penalizes only upward steps in the aggregate
      |Δresponse| curve; includes an RMSE-vs-ground-truth component and a
      bay_score<0.02 gate. Any non-increasing curve scores as "perfect"
      regardless of how irregularly it decreases.
    - 'isotonic' (default): Pool-Adjacent-Violators (PAVA) residual —
      penalizes ANY deviation from the best-fitting smooth non-increasing
      curve, in either direction, with NO assumption about the decay's
      functional form (not power-law-specific, not exponential-specific).
      No RMSE component (curve shape is independent of which ground truth
      downstream analyses use — see gt_mode below), no gate (every seed
      ranked by bay_resid + rl_resid, lowest wins). This is the currently
      preferred method.
    Note: neither score_mode penalizes achieved-vs-target mismatch (mean/
    std/p accuracy) at all -- only curve smoothness. Deliberately left this
    way after discussion: folding in a mismatch term is a genuine dual-
    objective tradeoff (could trade smoothness for accuracy or vice versa)
    with modest expected marginal benefit given the structural fix already
    in place, so it wasn't added. If accuracy ever looks insufficient for a
    winning seed, check via --report / the inspect_sequences.py CSV first;
    revisit adding a score term only if that's not enough.

Current best candidates for the 10x4 full experiment -- STALE, predate BOTH
the evenly-spaced/no-mirroring redesign AND the prefix/target-independence
redesign, would need regenerating from scratch under the current script to
be current (not just re-checked). Recoverable via git (commit 274b598) for
reference only:
  continuous: seed=245 (momentmatch, isotonic, mean_range=[20,80], std_fixed=15)
  binary:     seed=68  (momentmatch, isotonic, p_range=[0.2,0.8])
std_fixed=15 is the default for continuous (down from 20 in the original
rejection-sampling design) -- confirmed to resolve within the achievable
range via moment-matching.

**task/generate_sequences_hybrid.py** -- the CURRENT PRODUCTION method
  (chosen after PI discussion, resolving the i.i.d.-vs-momentmatch decision
  below into a per-task split rather than a single uniform answer). Binary:
  calls the exact same construction as generate_sequences_momentmatch.py
  (prefix/target independence, optimal matching, exact-quota suffix), but
  with NO seed search -- single draw. Continuous: same prefix/target
  construction and optimal matching, but the suffix
  (suffix_for_continuous_target_iid) is a genuinely UNRESCALED i.i.d. draw
  centered on the algebraic residual mean, not forced to hit it via
  momentmatch's iterative rescale -- also no seed search. Confirmed
  empirically (see docs/sequence_design_open_questions.md and this file's
  own module docstring) that dropping ONLY the seed search barely moves
  either the split-half reliability or the "back-half corrects front-half"
  correlation that make quota's terminal accuracy nearly guaranteed
  regardless of genuine evidence integration -- the iterative RESCALE step
  itself, not the search, is the actual source of both effects. Removing
  the rescale (keeping only residual-mean centering) is what actually
  restores genuine trial-ending uncertainty for continuous.
  Two variance guards on the continuous suffix, confirmed not to
  reintroduce the mean-related confound (checked directly, see the
  docstring for the numbers): (1) an analytical bias correction solving for
  the suffix's own variance parameter so E[pooled_std] ~= std_fixed,
  accounting for the prefix's fixed contribution; (2) a loose safety-net
  rejection (`--std_tolerance_frac`, default 0.25 = +/-25%, chosen as a
  cost/benefit bend point after directly comparing +/-33.3%/25%/15% across
  5 seeds) that redraws the whole suffix (fresh i.i.d., never a rescale) if
  the achieved std falls outside tolerance. No seed search anywhere in this
  file, for either task, by design.
  Currently promoted at 8x4 (32 trials, chosen over 10x4 specifically to
  reduce participant time) -- see "Sequence design" above for the exact
  parameters and verification. **`task/generate_sequences_pool.py` is now
  the actual production mechanism for the per-participant pool** -- see the
  new "Per-participant sequence pool" section right after this one for the
  full architecture (not "not wired into anything downstream" anymore --
  that was true when this note was first written, resolved since).

How far can moment-matching push mean_range/p_range toward [0,100]/[0,1]?
Tested empirically (moment_match_continuous/binary directly, 300 draws per
target, std_fixed=15, n=15 obs) -- this table predates the prefix/target-
independence redesign but the underlying per-BLOCK boundary-clipping bias it
measures is a property of moment_match_continuous itself, still accurate:

  target mean | achieved mean (bias)  | achieved std (target 15)
  ----------- | --------------------- | -------------------------
  50          | 49.99 (-0.01)         | 15.00
  30          | 30.00 (+0.00)         | 14.99
  20          | 20.06 (+0.06)         | 14.91
  15          | 15.26 (+0.26)         | 14.72
  10          | 10.95 (+0.95)         | 14.23
  8           | 9.43  (+1.43)         | 13.98
  5           | 7.67  (+2.67)         | 13.34
  2           | 6.14  (+4.14)         | 12.86
  0           | 5.45  (+5.45)         | 12.44

Bias grows smoothly (not a hard cutoff) as the target approaches the [0,100]
bound -- moment-matching pushes the usable range much further than rejection
sampling did, but does NOT eliminate the boundary-clipping problem, just
shrinks it. This is a PER-BLOCK bias (affects a single prefix or suffix
block built via moment_match_continuous); the CURRENT design's pooled-
sequence std inflation (prefix generic/unrelated to target -- see "Sequence
design" above) is a SEPARATE, additional effect layered on top of this one,
not a replacement for it. [15,85] (current production range) sits inside the
region where this per-block bias is small; below ~mean=10 it becomes large
enough to matter on its own. Binary has NO equivalent problem -- quota is
exact for any p in (0,1), the only limitation is 1/n rounding granularity.

### Open items (as of latest session)

- **6x4 pilot regenerated and promoted (prefix/target-independence redesign)**:
  new sequences generated under generate_sequences_momentmatch.py's
  redesigned prefix/target-independent-axes architecture (--n_prefix=6,
  continuous --mean_range=[15,85] (changed from [10,90]), binary
  --blue_range=[2,13] (unchanged), 1000-try isotonic seed search) and
  promoted to task/sequences/{continuous,binary}_sequences.{pkl,json},
  superseding the earlier evenly-spaced/no-mirroring sequences (which had
  the qid-prefix-collision bug -- see "Sequence design" above). Verified
  directly against the saved files before promoting: 6 distinct prefixes
  per task (zero collisions), exactly 4 repeats each, zero binary quota
  mismatches, full target-range coverage -- see "Sequence design" above
  for the exact numbers. NOTE: real pilots already collected participant
  data under BOTH earlier sequence sets (see "Pilot data files" below) --
  going forward, new participants run on yet another different stimulus
  set than those; keep this in mind for any analysis that pools across
  pilot generations.
- **RESOLVED -- hybrid method chosen** (see "Sequence generation methods"
  above for the mechanism, "Sequence design" above for the 8x4 promotion,
  and docs/sequence_design_open_questions.md for the full investigation
  this decision is based on): binary keeps quota/momentmatch construction
  unchanged (no seed search); continuous uses an unrescaled i.i.d. suffix
  with two variance guards. `generate_sequences_iid.py` DID have its own
  prefix-collision bug (independently-drawn per-qid prefixes, no
  uniqueness check -- confirmed empirically, 9/10 seeds collided at
  n_unique_sequences=10) -- this is now FIXED (`_draw_unique_binary_prefix`,
  active dedup with a fresh-redraw fallback for mirror collisions, verified
  10/10 seeds now correct), including a striking case where fixing it
  *lowered* apparent split-half reliability (the old "good" reliability was
  partly an artifact of the collision bug, not genuine signal). Both pure
  branches (generate_sequences_iid.py, generate_sequences_momentmatch.py)
  remain fully intact and re-runnable if either pure method needs
  revisiting later -- the hybrid script is a third, separate option, not a
  replacement for either.
- **Summary screens** (task/src/shared/plugin-trial-summary-{continuous,binary}.js):
  running-mean overlay -- DONE this session (see "Tutorial redesign, bonus/
  error system, and binary no-prefix sequences" above for the full
  mechanism: `ERROR_MODE`, per-observation error, per-trial bonus, redesigned
  charts). Previously listed here as deferred; no longer open.
- **config-base.js's pickTutorialExample** may still assume a qid's repeats
  share a target (see "Sequences.json schema" above) -- not yet verified
  against the prefix/target-independence redesign. Check before relying on
  its fallback path. Also NOT yet re-verified against binary's new
  no-prefix pool (every qid is already unique there, so the specific
  "falls through a qid's repeats" fallback path this note describes may be
  entirely moot for binary now, but hasn't been explicitly re-checked).
- **task/evidence-integration-binary.jzip is stale** -- needs a rebuild
  before any real pilot testing of this session's tutorial/bonus/chart
  work (see "Tutorial redesign..." above).
- **Response-variability/reliability metric tradeoff under binary's new
  no-prefix sequences** -- not yet decided whether the natural,
  opportunistic repetition found (see "Tutorial redesign..." above) is
  sufficient, or whether a deliberate small number of full-trial repeats
  is worth adding as a third generation branch.

Local dev: open http://localhost:5173/index-continuous.html or
  http://localhost:5174/index-binary.html via `npm run dev:continuous` /
  `npm run dev:binary` (each auto-opens its own task on its own fixed port,
  set in vite.config.js -- both can run simultaneously in two terminals).
  These call buildAndRun() with the real production config -- no dev-page,
  no override mechanism, full sequences, real timing.

### Task testing architecture (dev page removed)

The old index-dev.html setup page (task/binary select buttons, tutorial
skip toggle, trial-count/BTI/ITI presets, `?autostart=1`) is gone entirely --
it's not used, and it complicated buildAndRun/config-base.js with dev-only
knobs (`testMode`, `nTrialsDefault`, `trialItiMs`, `showTutorial`) that
nothing else needed. Removing them also closed a real risk: config-base.js's
old N_TRIALS_TO_RUN/TEST_MODE mechanism manually mirrored sequences.json's
trial count rather than reading it, which could have silently truncated
trials if the two ever drifted out of sync. Trial count is now fully
implicit from sequences.json's actual length everywhere.

**Production entry points are unchanged and were already correct**:
index-continuous.html / index-binary.html -> experiment-continuous.js /
experiment-binary.js -> `buildAndRun(config)` directly, full sequences, no
overrides, tutorial always shown. Nothing about production behavior changed.

**Test-only entry point** (index-test.html + src/test-harness.js) --
NEVER linked from production code, NEVER included in any build
(vite.config.js's rollupOptions.input only lists index-continuous.html /
index-binary.html, so this is automatically excluded with no extra care
needed; real participants can never reach or discover it). Calls the exact
same buildAndRun() production uses -- test-harness.js only builds a
slightly modified CONFIG OBJECT (fewer trials via array slicing, faster
timing via direct field assignment) using fields that already exist on the
config; no override logic lives inside buildAndRun/timeline-builder.js
itself. URL params (all optional): `task`, `trials` (default 3), `tObsMs`,
`btiMs`, `itiMs` (overrides both the main task's per-trial iti_ms AND the
tutorial's itiShortMs). No tutorial-skip param exists -- the tutorial
always runs in full, including in tests (see below for why).

**test_browser.mjs** drives this via Playwright across Chromium/Firefox/
WebKit, both tasks (30 scenarios). Since this still spawns the real Vite
dev server and calls the real buildAndRun with real plugins/CSS/DOM, and
Playwright still drives it with real clicks across real browser engines,
this preserves the actual bug-catching value the old dev-page-driven suite
had -- only the config VALUES differ from production, never the code path
or the interaction method.

The tutorial now runs in FULL during tests (previously skipped via
`?tutorial=false`, meaning tutorial screens were NEVER exercised by
automated tests at all). This was a deliberate choice, not just a side
effect of removing the skip option: tutorial screens have no response
deadline at all (see build-tutorial-timeline.js), so skipping was never
about avoiding a timer, just a few extra clicks -- running it for real
costs little and gains real cross-browser coverage of screens that were
previously completely untested. doTutorial() in test_browser.mjs walks the
full progressive-reveal intro -> remaining tutorial observations (each
preceded by a fixed tutorial_iti, no click needed) -> tutorial summary ->
3-screen timeout demo -> real trial 1. Same element ids across both tasks
(tut-box-0/1/2, tut-image-placeholder, response-slider, submit-btn,
proceed-btn) per the project's naming convention, so one implementation
covers both tasks without branching. When waiting for the tutorial's
repeated tutorial_iti <-> tutorial_observation cycle, always wait for the
INTERMEDIATE tutorial_iti screen first, not directly for the next
tutorial_observation -- the DOM can still show the just-submitted
tutorial_observation's stale data-screen value for a few ms after
submission, and tutorial_iti is a genuinely new state that can't be stale.

**Slider interaction in tests** (moveSlider in test_browser.mjs) sets the
slider's `.value` directly and dispatches a real `input` event, rather than
computing a pixel position from the element's bounding box and simulating a
mouse drag. This was a deliberate fix, not a stylistic preference: a
pixel-based approach failed specifically for the binary tutorial slider
under headless Chromium (computed height 16px instead of the CSS-specified
96px, `elementFromPoint` missing the element entirely) while working fine
for continuous in the exact same run -- but this could NOT be confirmed as a
real app-facing bug, since there's no way to launch a headed browser in this
sandbox to check against what a real user actually sees, and the person
running this project manually did not observe any such issue. Treat any
future headless-only slider anomaly the same way: fix the test's interaction
robustness first (event dispatch, not geometry), and don't conclude it's an
app bug without a way to visually verify against real usage.

**itiShortMs bug found and fixed while building this**: the tutorial's
between-observation ITI was hardcoded to `1000` in build-tutorial-timeline.js,
completely ignoring the `itiShortMs` config field that was declared and
threaded through config-base.js/timeline-builder.js the whole time --
dead/disconnected plumbing that had never actually been wired to anything.
Fixed: itiShortMs now flows through and is overridable via the test
harness's `itiMs` param.

Testing:
- node test_browser.mjs      — Playwright E2E tests across Chromium, Firefox, and
                               WebKit, both tasks (30+ scenarios total, full tutorial
                               included, plus end-screen/early-exit save verification
                               — see "Exit/redirect and data-saving architecture" below).
                               SLOW (~2-3+ min for the full matrix) — only run after big
                               task/ changes or when explicitly asked, not after every
                               small edit. Runs fine via shell tool (spawns the real
                               `vite` dev server against index-test.html, drives it via
                               URL params -- see "Task testing architecture" above --
                               detects screen transitions via body[data-screen="..."]
                               set by on_trial_start in timeline-builder.js, kills the
                               dev server by process group on exit).
                               **Claude must always run this as 6 separate shell:run_command
                               calls** (one per browser × task combination, e.g.
                               `node test_browser.mjs --browser=chromium --task=continuous`),
                               never as a single full-matrix call -- the shell tool's own
                               response window is shorter than the full suite takes, so a
                               single unfiltered call returns a tool-level timeout with no
                               result even though the suite is running fine remotely. Report
                               each of the 6 results as they land so the person can track
                               progress/timing. A spurious connection-refused failure right
                               after a prior call can mean a port-release race on 7655/3099
                               between back-to-back calls, not an app bug -- check
                               `lsof -ti:7655` / `lsof -ti:3099` are empty, then retry just
                               that combination.
                               Firefox/WebKit require: npx playwright install firefox webkit
                               (+ system deps via `sudo npx playwright install-deps` if missing —
                               watch for unrelated broken third-party apt repos blocking this)

  Avoid brittle exact-string assertions in E2E tests — copy/wording changes
  frequently (e.g. the early-exit button's text is now conditional on
  isProlific), so a test that greps for literal text breaks on legitimate
  content changes, not just regressions. Prefer structural checks:
    - `body[data-screen="..."]` for screen transitions — set automatically
      by on_trial_start in timeline-builder.js for jsPsych trials;
      create-early-exit.js sets `document.body.dataset.screen = 'terminated'`
      BY HAND for its own screen, since that flow is a manual DOM injection
      outside jsPsych's trial system and never fires on_trial_start on its
      own — if a new non-jsPsych screen is ever added, remember to set this
      attribute manually there too.
    - Element presence/absence via stable ids (#early-exit-btn, #summary-svg,
      #too-slow-pulse) rather than `.textContent.includes(...)`.
    - `data-*` attributes for dynamic values with a semantic meaning
      independent of wording — e.g. `data-timeouts-remaining="N"` on
      #too-slow-pulse (plugin-iti-clock.js), rather than matching the phrase
      "N timeouts remaining".
  Found and fixed a real instance of this: the E2E suite hardcoded 'Return
  to Prolific', which broke the moment the button text became conditional
  on isProlific (a legitimate content change, not a regression) — and two
  dead checks ('last chance', 'Trial summary') were testing for phrases
  that had never existed anywhere in the source at all. All 30 scenarios
  pass cleanly post-fix.

jsPsych 8 plugin conventions (IMPORTANT — do not regress):
- Custom plugins use: trial(display_el, trial, on_load) — NEVER declare this `async`.
  jsPsych 8.2.3 advances the timeline when an async trial()'s Promise resolves, not
  when finishTrial() is called; since these methods never await anything, that
  resolution happens almost immediately, causing overlapping/duplicate trial
  instances (see "Current task status" above — this was a real, previously
  undetected bug in production). Completion must come exclusively from finishTrial().
- Set innerHTML, call on_load(), then wire all interactivity synchronously
- Never use setTimeout or double-rAF to defer listener attachment
- Desktop/mouse only: mousedown for slider drag, click for submit button
- Use jsPsych.pluginAPI.setTimeout() not raw setTimeout for timed events
- Two consistent patterns only — do not introduce a third:
    Pattern A (no timeout clock — consent/tutorial/summary screens): plain
      trial(display_el, trial), wire listeners synchronously right after
      setting innerHTML. No on_load, no async, no rAF/setTimeout deferral.
    Pattern B (has a timeout clock — real observation plugins):
      trial(display_el, trial, on_load), call on_load() after innerHTML is
      set, wire listeners synchronously, then start the countdown via
      observation-timeout-clock.js's startTimeoutClock(canvas, t_obs_ms, onTimeout).
  The tutorial observation plugins (plugin-tutorial-observation-continuous.js,
  plugin-tutorial-observation-binary.js) are Pattern A — no timeout clock,
  deliberately, since participants need unhurried time during the tutorial.
- Parameters that the app ALWAYS supplies explicitly (true_mean, true_std,
  true_p across observation/summary/tutorial-intro plugins) must have NO
  `default` key in info.parameters — omitting `default` makes jsPsych treat
  it as required and throw `You must specify a value for the "X" parameter
  in the "Y" plugin.` if ever missing (confirmed against jsPsych's own
  source, node_modules/jspsych/dist/index.cjs). Several of these plugins
  had stale, mutually inconsistent numeric defaults (true_mean: 20/54,
  true_std: 10/20, true_p: 0.5/0.6/0.7) that were never actually triggered
  (every real call site always supplied a value) but silently masked
  exactly the true_std bug above for a long time — had they been required
  instead of defaulted, the missing-field bug would have thrown immediately
  instead of silently rendering a wrong Gaussian curve width. One file
  (plugin-tutorial-observation-binary.js) also had an EXTRA internal
  fallback beyond the parameter default (`trial.true_p ?? trial.true_mean
  ?? 0.7`) that masked a related hazard (binary sequence objects sometimes
  carrying a stray true_mean field equal to true_p) — removing a stale
  default only helps if you also check for fallback chains like this one.
  Prefer no default (fail loudly) over a plausible-looking fallback for
  anything the app is supposed to always provide.

JATOS/MindProbe deployment:
- .jzip files generated by task/generate_jzip.py (build + package in one step)
- Import each .jzip into MindProbe via Studies → + → Import Study
- Non-completions: request return on Prolific (not rejection); slot reopens
- Abandoned runs stay as DATA_RETRIEVED in MindProbe — filter by FINISHED state
- See "Exit/redirect and data-saving architecture" below for how/where data
  actually gets saved and where participants land afterward.

### Tab-visibility handling for observation timeouts (found and fixed)

**The bug**: `observation-timeout-clock.js`'s countdown (used by both
observation plugins AND `plugin-timeout-demo.js`) and `plugin-iti-clock.js`'s
circular ITI clock are both driven by `requestAnimationFrame` — correct for
smooth per-frame redraw, but rAF callbacks are fully SUSPENDED (not just
throttled) in a hidden/background tab, since rAF is tied to the paint cycle
and there's nothing to paint when hidden. Reported symptom: switching tabs
mid-observation let the clock reach the "X timeouts remaining" screen (that
message is `setTimeout`-driven, which still mostly fires in the background)
but then froze indefinitely on the NEXT observation's fresh rAF-driven
clock, never advancing until the tab regained focus — an unintended,
unbounded pause. This also meant a real response given upon return would
carry a badly inflated `rt` (`performance.now() - trialStart` naively
includes the away-time) — though a genuine timeout itself already records
`rt: null`, so that specific field wasn't corrupted, only a completed
response following a stray tab-switch would have been.

**The fix, two coordinated pieces**:
1. `observation-timeout-clock.js`'s `startTimeoutClock` now also listens for
   `document.visibilitychange` and calls `onTimeout()` IMMEDIATELY if the tab
   becomes hidden while active — treating hidden-while-active as an
   immediate deadline-reached event, same code path as the natural rAF
   completion (guarded by a `done` flag so the two paths can't double-fire).
   Listener is removed in the returned `stop()` too, so a later, unrelated
   tab switch during a subsequent trial has nothing stale left listening.
   Automatically covers both observation plugins AND the tutorial's timeout
   demo, since all three call this same shared function — exactly the kind
   of shared-module payoff this file was extracted for in the first place.
2. `plugin-iti-clock.js`'s `timed_out=true` branch (the "Too slow / X
   remaining" screen) no longer auto-advances via its own rAF-driven clock
   after the fade-message sequence — it now shows a manual "Repeat" button
   (disabled until its own fade-in completes, to prevent an accidental early
   click) that the participant must press to proceed. The normal
   (non-timeout) ITI between trials is untouched, still auto-advances as
   before.

**Why both pieces together, not just #1**: without #2, a stray focus loss
would immediately consume one timeout (#1) and then land on a screen that
still auto-advances on its own timer — which, if the tab is STILL hidden
when that timer would fire, would consume ANOTHER timeout immediately
afterward, and so on, potentially exhausting all `MAX_TIMEOUTS_PER_TRIAL`
and forcing early-exit before the participant ever regains control. With #2,
progression after any timeout (visibility-triggered or a genuine slow
response) requires an explicit click — which cannot happen while the tab is
hidden, since nobody is there to click it — so the WORST case from any
single stray tab-switch is exactly one consumed timeout, followed by an
indefinite, fully inert wait, never an automatic cascade. Confirmed via code
reading that `plugin-timeout-demo.js`'s own "too slow" screen already worked
this way (always required a "Next" click, never auto-advanced) — the real
task's ITI flow is now brought in line with a pattern the tutorial demo
already used, not something novel.

**Not addressed, deliberately out of scope for this fix**: the NORMAL
(non-timeout) ITI clock between trials still auto-advances via rAF and would
still freeze the same way if the tab is hidden during a genuine (not
timeout-triggered) waiting period — lower-stakes than the timeout case
since it can't cascade toward early-exit, but the same underlying rAF-
suspension mechanism applies there too if it's ever worth revisiting.
Browser/tab CLOSURE (as opposed to backgrounding) remains unrecoverable, as
already disclosed in the informed-consent text.

### Exit/redirect and data-saving architecture

**CURRENT ARCHITECTURE (this session) -- supersedes the single-call save
described in "The save mechanism" just below, which is kept as-is for its
historical rationale (the data-as-argument shape it settled on is still
used by every append below).** Prompted by a real discrepancy: more
Prolific completions/starts than JATOS showed matching data for, well past
an intended participant cap. Full investigation is in chat history; summary
of what changed and, importantly, what's still unverified:

- **Incremental per-trial saving, not one call at the end**
  (`timeline-builder.js`): `on_trial_finish` (global jsPsych option) fires
  `jatos.appendResultData` after EVERY trial -- welcome, consent, each
  tutorial step, every real observation, ITI/BTI resets, everything.
  Fire-and-forget (not awaited); failures are logged via `jatos.log`, but
  see the REAL-TEST FINDINGS bullet at the end of this list for why that's
  NOT where you should actually look to confirm a failure happened. A
  `"started"` marker (tagged with `jatos.addJatosIds`, giving a
  JATOS-native `studyResultId`/`workerId` cross-reference independent of
  `prolific_pid`) is appended before `jsPsych.run()` even starts, so a
  participant who closes the tab before clicking past welcome still leaves
  a trace. `jatos.catchAndLogErrors()` is also now wired in early, for the
  same reason -- forwards uncaught JS errors/rejections to the same
  `jatos.log` channel.
- **`finish-session.js` no longer sends the full dataset, and gates
  completion on a confirmed save**: rewritten from a single fire-and-forget
  `jatos.endStudy(data)`/`endStudyAndRedirect(url, data)` call into an
  explicit chain -- `jatos.appendResultData(marker).then(() =>
  jatos.endStudyWithoutRedirect(true)).then(redirect/confirm).catch(log +
  error screen, NO redirect, NOT marked finished)`. `marker` is now a small
  `{prolific_pid, progress, task, pool_index, is_prolific}` object (not
  `jsPsych.data.get().json()`) -- the full dataset would be a pure
  duplicate of everything already appended per-trial, including the final
  screen's own trial, which finishes before this function ever runs.
  `progress` is `'finished'` (normal end) or `'terminated'` (timeout-budget
  early exit) -- passed explicitly by each of the two call sites
  (timeline-builder.js's `on_finish`, create-early-exit.js's button
  handler), since `finishSession` itself has no other way to distinguish
  them. `SHOW_END_PAGE` is still exported (`= false`) purely so
  `generate_jzip.py`'s build check keeps passing -- it's vestigial now,
  since `endStudyWithoutRedirect` never shows a JATOS end-page by
  definition; see "How finish-session.js and generate_jzip.py must stay in
  sync" below, not yet updated to reflect this.
- **Every appended row is now lean and scannable**: `toLeanRow()` promotes
  `prolific_pid`/`progress` to the front of the object and strips
  `stimulus`/`button_html` (rendered HTML/CSS jsPsych's own
  html-button-response trials carry, previously cluttering the raw JATOS
  results view). `progressLabel()` maps each trial's `screen`/`trial`/
  `observation` tags to a human string (`"welcome"`, `"tutorial 2/4"`,
  `"trial 7/24"`, `"end"`) so a row can be read at a glance without
  downloading and parsing the file.
- **Worker type narrowed to GeneralSingle only** (`generate_jzip.py`):
  `allowedWorkerTypes` was all five JATOS worker types; now just
  `["GeneralSingle"]`. Motivated by a real JATOS maintainer
  (Kristian Lange, JATOS forum) diagnosing a symptom matching this
  project's own ("Prolific shows started/completed, JATOS shows nothing
  matching") as GeneralMultiple + a non-reloadable component (this
  project's components are all `reloadable: false`): a reload/retry for ANY
  reason ends that run as FAIL, and GeneralMultiple lets the participant
  just reopen the same link and start a brand-new, possibly-empty run
  silently. GeneralSingle converts that into a LOUD "Study can be done only
  once" error instead. **UNCONFIRMED**: this diagnosis was never actually
  checked against this project's own JATOS results table (would show as
  multiple Result IDs per participant, one empty/FAIL) -- it's the
  best-evidenced hypothesis, not a proven root cause for this project's
  specific past incidents.
- **Deliberately NOT built**: a `jatos.batchSession`-based dedup guard, and
  a global-error-handler-forces-a-terminate-code mechanism. Both were
  scoped and then dropped once the workflow moved to Prolific manual-approve
  -- a duplicate/stuck participant now just shows up for manual review
  (see the reconciliation script below) rather than needing to be blocked
  or force-redirected automatically.
- **`task/reconcile_prolific_jatos.py`** -- cross-references a
  Prolific submissions-export CSV against a JATOS plain-text results
  export (every row, not just observations -- see its own docstring vs.
  `parse_results.py`'s), outer-joined on participant ID, producing a
  `recommendation` per participant (OK / TERMINATED / STUCK / NO JATOS DATA
  AT ALL / pilot-ignore). Originally verified against five synthetic
  scenarios only; now ALSO run successfully against a real 6-participant
  MindProbe export (`dev-results/jatos_test.txt`) after the two parsing
  bugs below were found and fixed against that same real file. **Still NOT
  run against a real Prolific export** -- its Prolific column-name
  detection (fuzzy substring match, with `--prolific-*-col` overrides) is
  still a best guess at Prolific's current CSV headers, unconfirmed
  against an actual downloaded file.
- **`jatos.endStudyWithoutRedirect` CONFIRMED WORKING** (previously listed
  as the biggest open risk -- see the real-test bullet below): fired
  correctly on the real completion test, so MindProbe's deployed JATOS
  version does support it. No longer an open question.
- **REAL-TEST FINDINGS (this session, real MindProbe/JATOS, six manual
  scenarios via hand-edited `?PROLIFIC_PID=` params, no real Prolific
  involved)** -- what got confirmed, and two corrections to claims made
  earlier in this same session before real testing happened:
  - Normal completion, abandon-after-started, stuck-mid-session, and
    timeout-budget termination all produced exactly the expected data
    shape (`progress` values, `finished`/`terminated` markers) when
    inspected directly in a downloaded JATOS export.
  - The save-then-end gating actually works end-to-end: killing network
    (Chrome DevTools Network tab -> Offline) right at the final "Return to
    Prolific" click produced the real "Something went wrong saving your
    data" screen and did NOT redirect -- confirmed by direct observation,
    not just code tracing.
  - **CORRECTION: `jatos.log()`'s output is NOT visible anywhere in the
    JATOS results table/UI** (no "Log column" per result, contrary to what
    an earlier note in this section assumed) -- it most likely goes to
    JATOS's own application-level server log, which isn't exposed on a
    hosted instance like MindProbe at all. Checked directly after the
    network-failure test above: nothing showed up. Practical consequence:
    don't rely on `jatos.log`/`catchAndLogErrors` as a way to SEE failures
    -- the actual signal is the DATA itself. A row that stops mid-session
    with no `finished`/`terminated` marker (or a gap where the next
    expected screen's row should be) IS the failure signal; use the last
    successfully appended row's `progress` value to decide how far someone
    actually got when making a payment call, exactly what
    `reconcile_prolific_jatos.py` already does (it never looks for a
    log message, only for what did or didn't get appended).
  - **GeneralSingle's block is keyed on the BROWSER's cookie, not on the
    `PROLIFIC_PID` value in the URL at all** -- confirmed directly:
    manually editing the URL's `PROLIFIC_PID` to a different value on a
    second visit in an already-used browser still got blocked with "Study
    can be done only once." This is good: a participant can't dodge the
    single-use restriction by relabeling themselves. One practical
    consequence worth knowing before reading a results export: if a
    browser was used for an EARLIER, unrelated visit before the one you
    meant to test, that earlier visit's real (if abandoned) data will
    show up in the export under whatever ID was in the URL for IT, not
    the ID you later edited the URL to -- easy to misread as "data got
    copied" between two attempts when it's actually two independent
    visits. The blocked attempt itself produces zero data, as expected.
  - **TWO REAL PARSING BUGS FOUND AND FIXED**, both only catchable with a
    real multi-append JATOS export (the exact thing this session's
    incremental-saving change made possible for the first time) --
    neither was hypothetical:
    1. `parse_results.py` and `reconcile_prolific_jatos.py` both parsed the
       real 6-participant export as almost entirely EMPTY (1 row out of
       what should have been dozens). Cause: JATOS's plain-text export
       concatenates every `appendResultData` call with ZERO separator
       within one participant's block (`}{`, no comma, no newline) --
       newlines only appear BETWEEN different participants. The old
       newline-split-then-`json.loads()`-per-line approach silently
       dropped everything that wasn't its own clean line. Fixed in BOTH
       files with a shared `iter_json_values()` streaming decoder
       (`json.JSONDecoder().raw_decode` in a loop) that peels off exactly
       one JSON value at a time regardless of what does or doesn't
       separate it from the next -- verified against the same real file
       afterward (86 rows / 5 participants from `parse_results.py`,
       correct per-participant summaries from `reconcile_prolific_jatos.py`).
    2. Separately, `reconcile_prolific_jatos.py`'s "last progress" logic
       was ALSO wrong even once parsing was fixed: it sorted by
       `time_elapsed` to find each participant's last row, but the
       `started`/`finished`/`terminated` MARKER rows (the exact ones this
       whole script exists to detect) carry no `time_elapsed` field at
       all, since they aren't real jsPsych trials -- sorting with
       `na_position='first'` shoved them to the wrong end, so a fully
       finished participant showed their last real trial screen instead of
       `'finished'`. Fixed by using parse/append order (a monotonic
       sequence number) instead of `time_elapsed` for ordering -- JATOS
       writes appends in receipt order, so this is simpler AND correct by
       construction. Verified: the real completion test now correctly
       shows `'finished'`, the real termination test now correctly shows
       `'terminated'`.
  - Not yet tested against real Prolific traffic (all six scenarios used
    hand-edited query params on JATOS's own General Single link, per
    explicit choice to avoid spending real Prolific participants on this).

**The save mechanism** (HISTORICAL -- describes the single-call approach in
place before the incremental-append rework above; kept for why the
data-as-argument shape was chosen, which the new `appendResultData` calls
above still rely on): a single call, `jatos.endStudyAndRedirect(url, data)`
(Prolific) or `jatos.endStudy(data)` (everyone else), with the full
`jsPsych.data.get().json()` results passed directly as the argument. There
is no separate `jatos.submitResultData()` call anywhere in the app code (the
shim still exposes one for API-surface parity, but nothing calls it). This
is a deliberately restored, EMPIRICALLY VERIFIED choice, not an assumption:
a prior revision switched to a two-call `submitResultData(data)` then
`endStudyAndRedirect(url)` (no data arg) pattern, based on an unverified
documentation claim that `endStudy`/`endStudyAndRedirect` don't accept data
at all. That claim was never independently re-confirmed and directly
contradicted real evidence: `task/dev-results/pilot7cont.txt`, a complete
24-trial continuous session downloaded directly from MindProbe, was
produced using exactly the single-call, data-as-argument shape. If this is
ever revisited, get independent confirmation (e.g. a live MindProbe dry-run
checked against the admin panel) before trusting a documentation claim over
a real prior result -- this has now bitten the project twice (see the
redirect bug below, found the same way: real MindProbe testing catching
what no amount of local shim testing could).

**Where participants land, and why non-Prolific gets NO redirect**
(`finish-session.js`, shared by both exit paths below): real Prolific
participants (`PROLIFIC_PID` present in the URL) redirect to a Prolific
completion URL with a per-task, per-exit-reason code (see PROLIFIC_CODES
below) -- an EXTERNAL domain (app.prolific.com), entirely outside anything
JATOS itself controls access to.

Everyone else (local dev/test AND non-Prolific JATOS/MindProbe pilots) gets
NO redirect at all -- confirmation is shown via a DOM update in the
CURRENTLY loaded page instead. This was NOT the original design: an earlier
version redirected non-Prolific participants to a same-origin confirmation
page (`public/exit-complete.html`) the same way the Prolific branch
redirects to an external one. **This failed on a real MindProbe pilot run**:
deliberately timing out 3 times and clicking "Finish and exit" produced a
JATOS error page -- "You tried to access the file
.../exit-complete.html but it seems you have no access rights." Best
explanation: `jatos.endStudyAndRedirect` ends/closes the session as part of
its own execution, and by the time the browser's follow-up navigation to
that file actually fires, JATOS's access-control layer sees the session is
no longer active and rejects the request for that study asset -- a
restriction that Prolific's redirect was never subject to in the first
place, since it's not a JATOS-served file at all. This is exactly the kind
of failure the automated E2E suite (which only exercises the local shim,
never real JATOS access control) cannot catch -- found only because a real
pilot run was actually tested against real JATOS before wider rollout.
`public/exit-complete.html` has been deleted; nothing redirects to it
anymore, locally or otherwise (kept the local shim and production on the
SAME code path deliberately, rather than diverging behavior between them --
see "Task testing architecture" for why that principle matters generally).

**Two screens in sequence, always** -- an in-app screen gates the actual
save behind a deliberate button click, and (for non-Prolific) that SAME
screen is where the confirmation message appears, updated in place:
  - Normal completion: `build-end-screen.js`'s "Thank you!" screen -> click
    -> `on_finish` in timeline-builder.js fires `finishSession`, which
    updates `#jspsych-content` in place for non-Prolific (no navigation).
  - Early exit (3 timeouts in one trial): `create-early-exit.js`'s "Session
    terminated" screen -> click -> its own button handler fires
    `finishSession` the same way. `on_finish` has a guard
    (`if (isExited()) return;`) specifically so it doesn't ALSO fire here --
    jsPsych's own timeline naturally reaches its end (all remaining
    trial-loop nodes become conditionally skipped) well before the
    participant ever sees this screen or clicks anything, which would
    otherwise double-call the exit.

**Early-exit button is one-shot** (`create-early-exit.js`): this screen is
hand-rolled DOM, not a real jsPsych trial (jsPsych's timeline has already
reached its natural end by the time a participant sees it -- see the
module's own docstring), so it never got the double-click protection
jsPsych's own button-response plugin gives every other button in this app
for free (it disables its buttons on click). Fixed to match: the button's
pointerdown listener is registered with `{ once: true }` and sets
`btn.disabled = true` as its first action, reusing the existing
`.jspsych-btn:disabled` CSS (style.css) for identical visual feedback to
every other button. This isn't cosmetic -- calling `finishSession()` (and
therefore `jatos.endStudy`/`endStudyAndRedirect`) twice is the exact bug
class already hit twice in this file's own history: a second call can hit
a DIFFERENT failure on real JATOS (session already closed), not a
harmless no-op.

**PROLIFIC_CODES** (timeline-builder.js): one object grouped by task, each
with a `completion` and `earlyExit` code -- filled in with real codes as of
the latest session: continuous {completion: C1CNSEMJ, earlyExit: C1ARJ6LO},
binary {completion: C12FEFJU, earlyExit: C1L1GGHT}. These belong to fresh
studies created inside a Project in the shared "Human Mixed Task" Prolific
workspace (the PI's workspace) -- NOT the earlier draft studies in the
original personal workspace, which are being abandoned in favor of this
project-based setup. An EARLIER set of codes (`C3W3TF1O` for the shared
normal-completion code, `CHXZJB62`/`C1QXJUFU` for continuous/binary
early-exit) belonged to those now-abandoned drafts -- fully superseded, do
not reuse or confuse with the current set above. (`C3W3TF1O` itself has a
longer history worth knowing if it resurfaces anywhere: it's also the
specific code an earlier session found hardcoded and discarded from
generate_jzip.py's endRedirectUrl field, unrelated to its later brief
reuse as an actual real Prolific code for the old drafts -- see the
"Two independent completion mechanisms" note below for that separate
story. Neither history makes it valid for anything current.) Screen-out
(early-exit) reward is set to a flat $3, decided as a deliberate
simplification over exact time-proportional pay -- see chat history for
the reasoning (informed-consent warnings already cover the risk, and a
meaningfully-smaller flat amount preserves the incentive to stay engaged
rather than diluting it).

**Two independent completion mechanisms exist, and only one was ever
addressed**: app-level JS (`finish-session.js`'s calls to
`jatos.endStudy`/`endStudyAndRedirect`, which is everything documented
above) and a SEPARATE platform-level mechanism -- JATOS's own
`showEndPage` behavior plus the study's "End Redirect URL" property (set
in `generate_jzip.py`'s `.jas` spec). Per JATOS's own jatos.js reference,
`showEndPage` DEFAULTS TO TRUE, meaning `jatos.endStudy(data)` with no
other arguments -- what this file called until a recent review -- tells
JATOS to redirect to an end page (its own default one, or whatever's in
the study's End Redirect URL) AFTER the study finishes, regardless of
what the app's own JS thinks it's doing. `generate_jzip.py` had real-
looking Prolific completion URLs sitting in that property (one of them
the specific `C3W3TF1O` code discarded above), so this could have silently
redirected a non-Prolific pilot participant to a dead Prolific completion
page right after the "Session complete" screen, with no visible error.
`pilot7cont.txt` never ruled this out -- it only proves data saved
correctly, not that no redirect happened afterward; those are independent
facts that got conflated. This is exactly the same class of blind spot as
the redirect-after-session-close bug described above ("Where participants
land..."): a JATOS study/component PROPERTY (evaluated server-side) that
the local dev shim (jatos-shim.js) has zero representation of, since the
shim only mocks jatos.js's client-side API, not server-side property
enforcement -- no amount of local/E2E testing against the shim could ever
have caught this. Fixed:
`finish-session.js` now explicitly passes `showEndPage=false`, and
`generate_jzip.py`'s `endRedirectUrl` is now an empty string (matching
that spec's own convention for other unset string fields) rather than any
plausible-looking URL -- so the JS remains the single source of truth for
redirect behavior in both local dev and real JATOS, and if this field's
inert status is ever accidentally bypassed by a future change, it fails
obviously rather than silently sending someone to a stale code. Other
`.jas` properties (`linearStudy`, each component's `reloadable`) are the
same category of risk -- platform-enforced, invisible to local testing --
though none are currently known to conflict with anything.

**How finish-session.js and generate_jzip.py must stay in sync** (read
this before touching either file -- NOTE: as of this session, `SHOW_END_PAGE`
is vestigial in `finish-session.js` itself, since the rewritten save-then-end
chain calls `jatos.endStudyWithoutRedirect` directly, which never shows a
JATOS end-page by definition; the constant is kept ONLY so the check below
keeps passing unchanged, not because anything still passes it as an
argument -- see "CURRENT ARCHITECTURE" above. The check itself and its
reasoning below are otherwise unchanged): `generate_jzip.py`'s `UNUSED_END_REDIRECT_URL
= ""` is a fixed, hardcoded value -- it is NOT derived from
`finish-session.js` in any general sense, and there is nothing that keeps
these two files in sync automatically beyond one narrow, explicit check.
That check (`assert_show_end_page_disabled()` in `generate_jzip.py`, run
as the very first thing `main()` does, before any build or packaging)
spawns a throwaway `node --input-type=module` process that IMPORTS THE
REAL `finish-session.js` module and reads its exported `SHOW_END_PAGE`
constant -- not a regex/text match against the source, and not a second
hardcoded copy of the assumption, both of which could silently drift.
Safe to import standalone: `finish-session.js` only references
`jatos`/`document` inside `finishSession()`'s function body, never at
module top level, so importing it alone never executes that code. If the
import fails for any reason, or `SHOW_END_PAGE` isn't exactly `false`, the
script refuses to build (`sys.exit`, not a warning) with a specific error
message naming both files.
  - **What this check does NOT do**: it does not make `endRedirectUrl`
    track `finish-session.js`'s behavior automatically, and it would NOT
    catch every possible future divergence between the two files -- only
    this one specific, narrow precondition. There is no general
    auto-derivation here because there is no automatically-correct
    `endRedirectUrl` to compute from the JS; the design intent is simply
    "the platform-level redirect must never fire, ever" (a fixed
    invariant), not "mirror whatever URL the JS happens to use" (there
    isn't one on the non-Prolific path, and the Prolific path's URL is
    dynamic and per-task/per-exit-reason anyway).
  - **If `SHOW_END_PAGE` ever needs to become `true`**: that is a
    deliberate design change, not a one-line toggle. It requires deciding
    what `endRedirectUrl` should be at that point (there's no automatic
    right answer), updating `assert_show_end_page_disabled()`'s expected
    value to match, and re-confirming the whole thing against a real
    MindProbe pilot run before trusting it -- the same standard this
    project already holds every other claim about JATOS's actual runtime
    behavior to (see "The save mechanism" above).
  - **Before Prolific production**: re-run `python task/generate_jzip.py`
    (which re-runs this check) after ANY change to `finish-session.js`,
    not just ones that look redirect-related -- the check is cheap and
    unconditional, so there's no reason to skip it, and no other part of
    the pipeline will catch a regression here (see the "local dev shim has
    zero representation of this" point above).

**generate_jzip.py generates a FRESH study/component/batch UUID on every
run** (not a fixed literal, as it was before a real near-miss surfaced
this): JATOS matches studies by UUID, not filename or content, so
importing a jzip whose UUID matches an already-imported study triggers an
"overwrite this study?" prompt -- which replaces that study's served
assets in place while leaving already-collected result data untouched.
That's fine for a genuine in-place fix, but wrong when promoting a real
new version (e.g. the 6x4 -> 10x4 transition) while an OLDER pilot's
already-distributed links are still meant to be collecting responses on
the OLD content -- overwriting would silently swap what those old links
serve, with no visible sign anything changed, potentially mixing two
different task versions under one nominal "pilot" label. Confirmed via
JATOS's own docs/forum: giving the new build a different UUID is what
makes JATOS import it as a genuinely separate study with its own new
distribution links, never touching the old one. This is now automatic --
every `python task/generate_jzip.py` run prints the UUID it generated, and
importing that jzip will always create a NEW MindProbe study, never
overwrite an existing one. If an in-place overwrite is ever genuinely
wanted (e.g. fixing a typo on a study that hasn't collected any real data
yet), that requires manually reusing the specific UUID JATOS shows for
that study's properties -- there's deliberately no flag for this in the
script, since silently defaulting to "never overwrite" is correct in the
overwhelming majority of cases and the failure mode of getting it wrong
(silently corrupting an in-progress pilot's data) is worse than the
inconvenience of doing a real overwrite manually on the rare occasion it's
actually wanted.

**End-screen/early-exit save verification (test_browser.mjs)**: until a
recent session, the E2E suite never actually exercised this whole section --
every scenario stopped short of the "Thank you!" end screen and the
'terminated' screen's own button, so `finishSession()` (and therefore
`jatos.endStudy`/`endStudyAndRedirect`) was never once called by the
automated suite. This is exactly why every real bug described above (the
submitResultData/endStudy signature bug, the double-save bug, the
same-origin-redirect-after-session-close bug) was found only by a live
MindProbe pilot run, never by this suite. Fixed: test_browser.mjs now also
spins up dev-server.js (the same local shim endpoint jatos-shim.js posts to)
alongside the Vite dev server, and both the "Completes all trials" and "3
timeouts" scenarios click all the way through to their respective buttons,
wait for finishSession's "Session complete" confirmation, and assert a new
`result_<timestamp>.json` file actually appeared in dev-results/ with real
content -- then delete that test-created file afterward (matched by
dev-server.js's own naming, so real pilot/dev files already in that folder
can never be touched). This closes the coverage gap for the LOCAL-SHIM half
of the save flow; it still can't catch a real-JATOS-only failure mode (see
the redirect bug above) -- that class of bug still needs a live MindProbe
dry-run.

Pre-deployment checklist (before Prolific production):
  - [DONE] Confirm MindProbe's actual JATOS server supports
    `jatos.endStudyWithoutRedirect` -- confirmed working directly via a
    real completion test this session (see "REAL-TEST FINDINGS" above).
  - [DONE] Confirm the incremental per-trial `appendResultData` calls
    (`timeline-builder.js`'s `on_trial_finish`) actually land in JATOS's
    results view during a real pilot run, and that `progress`/`prolific_pid`
    show up first as intended -- confirmed via six real manual test
    scenarios this session (normal completion, started-only, stuck
    mid-session, timeout termination, network-failure-at-completion,
    GeneralSingle reuse-block); see "REAL-TEST FINDINGS" above for the two
    corrections that came out of that testing (`jatos.log` visibility,
    GeneralSingle's cookie-based-not-ID-based blocking).
  - [ ] After importing a jzip built with the new `["GeneralSingle"]`-only
    `allowedWorkerTypes`, grab the new General Single link from MindProbe's
    Worker & Batch Manager and update each Prolific study's Study URL to
    point at it (same `?PROLIFIC_PID={{%PROLIFIC_PID%}}&...` suffix as
    before) -- the worker-type change alone doesn't update what's already
    pasted into Prolific.
  - [REMOVED] `--max-workers`/`maxTotalWorkers` was considered as a
    JATOS-side backstop, then deliberately dropped: Prolific's own "Places"
    cap is the intended control for over-recruitment, it was never
    confirmed to have caught anything in this project's actual past
    incidents, and "+/- a few concurrent submissions" was explicitly
    accepted as tolerable risk rather than something worth a code-level
    guard.
  - [ ] Run `task/reconcile_prolific_jatos.py` against a REAL Prolific
    export at least once before relying on it -- its column-name detection
    has only been checked against a synthetic CSV, not Prolific's actual
    current export format. (The JATOS-side half IS now done: run
    successfully against the real `dev-results/jatos_test.txt` export this
    session, after fixing two real parsing bugs found in the process --
    see "REAL-TEST FINDINGS" above. Only the Prolific-CSV half remains
    untested, since no real Prolific export exists yet.)
  - [ ] Before trusting the GeneralSingle worker-type switch as the fix for
    past "leaked through" participants, check the OLD JATOS results table
    (from before this session's changes) for the specific fingerprint that
    motivated it: multiple Result IDs under one participant, one of them
    empty/FAIL -- this was never actually confirmed, only inferred from a
    matching forum report. (This session's real testing confirmed
    GeneralSingle's blocking MECHANISM works correctly -- it did NOT
    confirm this specific historical root-cause attribution, which is a
    separate claim.)
  - [DONE] Confirm task/sequences/{continuous,binary}_sequences.json holds the
    intended final trial count/parameters -- 8x4 hybrid (32 trials), see
    "Sequence design" above.
  - [DONE] Removed PILOT ONLY name field (build-consent-screen.js +
    parse_results.py -- NOT timeline-builder.js, correcting this checklist's
    earlier file reference). test_consent_name.mjs (tested only this feature)
    moved to _trash/.
  - [REMOVED FROM CHECKLIST] "Fill IRB Protocol Number" -- no such placeholder
    exists anywhere in the current codebase (checked timeline-builder.js,
    build-consent-screen.js, consent_form.txt directly -- zero hits for
    "protocol" in any of them). Either resolved outside the on-screen text
    (e.g. communicated to Prolific/IRB directly) or never actually applicable
    to begin with; removed as a checklist item per explicit instruction
    rather than left as a permanently-stale blocker.
  - [DONE] All 4 Prolific code placeholders in timeline-builder.js's
    PROLIFIC_CODES are filled with real codes from the "Human Mixed Task"
    workspace project (see "PROLIFIC_CODES" note above for the exact
    values and which earlier code set they superseded). Still outstanding:
    the actual Study URL field on each Prolific study can't be set until
    the production build is confirmed live on MindProbe and a JATOS link
    is generated -- deliberately last, per explicit decision, not an
    oversight.
  - [DONE] Prolific wallet funded; payment rate confirmed: $10 for normal
    completion, $3 for the screen-out/early-exit path (see "PROLIFIC_CODES"
    note above for the reasoning behind the $3 figure specifically).
  - [DONE] node test_browser.mjs: all 6 browser x task combinations confirmed
    48/48 passing (8/8 each) against the final per-participant-pool state,
    including the new pool-assignment scenario.
  - [DONE] Rebuild jzips (python task/generate_jzip.py) -- rebuilt after
    the per-participant pool work landed; confirmed no source/pool files
    postdate the current jzips, and confirmed directly in the built
    bundles (not just source) that pool_index and urlQueryParameters both
    landed correctly.
  - [ ] **task/evidence-integration-binary.jzip needs a REBUILD** -- it now
    predates all of the latest session's binary tutorial/bonus/chart work
    and the production binary sequence-pool switch (see "Tutorial
    redesign, bonus/error system, and binary no-prefix sequences" above).
    evidence-integration-continuous.jzip WAS rebuilt that same session and
    is current; binary was not.
  - (No manual step needed for the showEndPage/endRedirectUrl invariant --
    generate_jzip.py's assert_show_end_page_disabled() enforces it
    automatically on every run and refuses to build if it's ever violated;
    see "How finish-session.js and generate_jzip.py must stay in sync"
    above. Still worth a live MindProbe dry-run before wide rollout, since
    that check only confirms the JS-side intent, not real JATOS behavior.)
  - Verified via real Prolific "preview as participant" (Chrome + Firefox
    incognito): the tab-visibility redirect fix and the per-participant
    pool assignment both work correctly against the real platform, not
    just the local shim -- see "Per-participant sequence pool" above.
    NOT yet verified against real preview: a genuinely FULL completion run
    (all trials through to the real "Thank you!" end screen and redirect)
    -- every real-platform check so far went through the cheaper early-exit
    path (3 timeouts), which structurally can't exercise the last-trial
    code path this project has had a real crash on before (the missing
    true_p bug). A full preview run is planned next specifically to close
    this.

Pilot data files:
  data/task_results.pkl          — pilot 3 (40 trials, pilot_undefined)
  data/task_results_pilot4.pkl   — pilot 4 (20 trials, pilot_undefined)
  data/task_results_pilot5.pkl   — pilot 5 (24 trials, pilot_undefined, old jzip)
  dev-results/test6bin.txt etc.  — pilot 6 test (2 trials, name='peter' ✓)
  dev-results/pilot7cont.txt,
  dev-results/pilot7bin.txt      — pilot 7 (24 trials each, pilot_undefined) --
                                    downloaded directly from MindProbe; this is
                                    the file that empirically settled the
                                    single-call-vs-two-call data-saving question
                                    above -- don't delete it without good reason.

Archived (do not reactivate): diederen, jiang, usher.

---

## Active models

| Model | Role | Free params |
|-------|------|-------------|
| Mean | Optimal running mean (Bayesian baseline) | none |
| LeakyIntegrator | Exponential forgetting baseline | gamma |
| PrimacyRecency | Temporal weighting (primacy + recency terms) | eps_p, eps_r |
| NoisyCounting | Task-specific model (Prat-Carrabin 2024); carrabin only | mu, sigma_c, nu |
| RL_lambda | Power-law delta rule (explicit equation) | alpha_0, lambda_ |
| NEF | Spiking NEF integrator (emergent power-law dynamics) | alpha_0, lambda_ |

NoisyCounting applies to carrabin only. Two fitted versions:
- RMSE-fitted: sigma_c collapses to ~0 (response-noise artefact; methodologically revealing)
- MLE-fitted (fit_mle.py): recovers sigma_c ~0.03-0.08, nu ~0.08-0.21

RNN (models/RNN.py): retained for reference; not used in active figures.

---

## NEF architecture

The NEF model implements sequential evidence integration via three interacting
neural populations:

1. **Value ensemble**: maintains a running estimate of the current evidence
   mean. Receives weighted prediction-error input; its activity decodes the
   current estimate after each observation.

2. **Error ensemble**: computes the prediction error (new value − current
   estimate) and gates it by the current observation weight α(t). This
   ensemble's weight-tuned neurons are the key neural readout — their activity
   directly tracks α(t) across the sequence.

3. **Counting subnetwork**: tracks the observation count and decodes the
   power-law weight α(t) = α₀ / t^λ. This is the mechanism that produces
   the same per-observation discounting as RL_lambda but via spiking dynamics
   rather than an explicit equation. The subnetwork requires a precomputed
   activity file (counting_activities_n{n}_nc{nc}_{dataset}.pkl) generated by
   counting_integrator.py.

Trial-to-trial variability in neural tuning curves (controlled by `seed =
int(trial)`) is the primary spiking noise source, producing state-persistent
response variability across observations within a trial.

Activity files are loaded at fit time for speed (fast_decode mode). Generate
locally with counting_integrator.py then scp to the cluster before submitting
fitting jobs (see Simulation pipeline below).

---

## Carrabin response transform

All carrabin models EXCEPT NoisyCounting apply: response = raw * t/(t+2)
Implemented in utils/carrabin_transform.py. Never apply it twice.

---

## Fitting pipeline (RMSE)

Submit and collect per-dataset per-model:

    # Submit (cluster)
    venv/bin/python -m fitting.submit carrabin NEF --n_trials 100 --run_folder carrabin --k 5
    venv/bin/python -m fitting.submit yoo NEF --run_folder yoo --n_trials 100 --k 5

    # Collect params and responses
    venv/bin/python -m fitting.collect carrabin --type params
    venv/bin/python -m fitting.collect carrabin --type responses
    venv/bin/python -m fitting.collect yoo --type params
    venv/bin/python -m fitting.collect yoo --type responses

    # Collect activities (after responses; needed for neural figures)
    venv/bin/python -m fitting.collect yoo --type activities --ensembles error --timing once_per_obs

Run folders: data/runs/carrabin/, data/runs/yoo/, data/runs/refit/
The --nef_folder flag in figure scripts redirects NEF data to a separate folder
(e.g. --run_folder yoo --nef_folder refit uses yoo for other models, refit for NEF).

## Fitting pipeline (MLE — NoisyCounting only, carrabin)

    bash jobs/submit_mle_fit.sh NoisyCounting carrabin 500 100
    # args: model, dataset, n_fits, n_sims
    # output: data/runs/carrabin/NoisyCounting_carrabin_{pid}_params_mle.pkl

---

## Simulation pipeline (extra data for figure scripts)

Some figure panels require data generated outside the main fitting pipeline.
Always generate locally (or via cluster if slow), then scp to the cluster.
Never run NEF simulations through MCP tool calls (will time out).

### Counting activity files (required before NEF fitting)

    # Generate locally
    venv/bin/python models/counting_integrator.py --precompute_activities \
        --n_neurons 200 --n_neurons_counting 1000 --dataset yoo --n_trials 30

    # Copy to cluster
    scp data/counting_activities_n200_nc1000_yoo.pkl \
        f007qzn@discovery.dartmouth.edu:~/evidence_integration/data/

### PE dynamics (carrabin neural panel A — figure_carrabin_neural.py)

    # Run locally or via cluster (slow for large n_neurons)
    python scripts/extras_carrabin.py --experiment pe_dynamics --mode simulate \
        --alpha_0_list 0.1 0.3 --n_neurons_list 50 150 --run_folder carrabin
    # Output: data/runs/carrabin/pe_dynamics_NEF_carrabin_a{...}_n{...}.pkl

### Probe simulations (carrabin neural panels B–D — figure_carrabin_neural.py)

    # Submit to cluster
    bash jobs/submit_probe_pids.sh

    # Collect
    python scripts/extras_carrabin.py --experiment probe_pids --mode collect \
        --out_folder refit
    # Output: data/runs/refit/probe_pids_carrabin.pkl

### n_neurons scan (carrabin neural panel D — figure_carrabin_neural.py)

    # Run locally (moderate compute)
    python scripts/extras_carrabin.py --experiment n_neurons_scan \
        --n_neurons_list 50 100 150 200 250 300 --run_folder carrabin
    # Output: data/runs/carrabin/n_neurons_scan.pkl, n_neurons_scan_metrics.pkl

### Lambda=0 ablation (yoo neural panel B control — figure_yoo_neural.py)

    # Run all pids (on cluster or locally — ~3-6 min per pid)
    for pid in $(venv/bin/python -c "import pandas as pd; from utils.paths import data_path; print(' '.join(str(p) for p in sorted(pd.read_pickle(data_path('yoo.pkl'))['pid'].unique())))"); do
      venv/bin/python scripts/extras_yoo.py --experiment lambda0 \
          --mode run --pid $pid --source_folder refit --run_folder yoo_lambda0
    done

    # Collect (combines responses + activities, copies encoders + params)
    venv/bin/python scripts/extras_yoo.py --experiment lambda0 \
        --mode collect --run_folder yoo_lambda0 --source_folder refit

    # Use in figures: --nef_folder yoo_lambda0
    # Output: data/runs/yoo_lambda0/
    #   NEF_yoo_lambda0_responses.pkl, NEF_yoo_responses.pkl (alias)
    #   activities_error_yoo.pkl, encoders_error_yoo.pkl
    #   NEF_yoo_{pid}_params.pkl (copied from refit — lambda_ values for reference)

### Error ensemble activities (yoo neural panels A–C — figure_yoo_neural.py)

    # Collect after NEF yoo fitting is complete
    venv/bin/python -m fitting.collect yoo --type activities \
        --ensembles error --timing once_per_obs
    # Output: data/runs/{folder}/activities_error_yoo.pkl, encoders_error_yoo.pkl

---

## Task simulation pipeline (scripts/test_sequences.py, scripts/inspect_sequences.py)

Simulates RL_lambda and NEF models on the task sequences for validation figures.

### Generate sequences

See "Sequence generation methods" above for the three scripts (and each
script's own module docstring for its current ROLE). Quick reference:

    # Pure i.i.d. (single draw, no search) -- one of two candidates still
    # under PI consideration for the 10x4 design, not current production:
    venv/bin/python task/generate_sequences_iid.py --task both --seed 0 \
        --n_unique_sequences 10 --n_repeats 4 --mean_range 20 80 --std_fixed 15 --p_range 0.2 0.8

    # Moment-matched / quota (isotonic seed search, default score_mode) --
    # this is what generated the CURRENT PRODUCTION 6x4 pilot (prefix
    # identity and target level are independent axes -- see "Sequence
    # generation methods" above):
    venv/bin/python task/generate_sequences_momentmatch.py --task both --n_tries 1000 \
        --n_prefix 6 --n_repeats 4 --rl_alpha_0 1.0 --rl_lambda 0.5
    # Defaults used above (all overridable): --n_prefix 6, --mean_range 15 85
    # (continuous), --blue_range 2 13 (binary, blue-ball count out of
    # --seq_length -- NOT a p fraction). Output goes to
    # task/sequences/{task}_momentmatch_sequences.{pkl,json} by default (NOT
    # the production filenames) -- promote by copying over
    # {task}_sequences.{pkl,json} once verified (see "Sequence design" above
    # for the verification steps actually used for the current pilot).

    # WARNING (both scripts): --task both overwrites BOTH sequence files.
    # After a search, regenerate whichever task you want to keep with --seed N.

    # generate_sequences.py has no CLI of its own anymore -- it's a shared-
    # utilities module only (imported by both scripts above), not something
    # you run directly. See its module docstring / "Sequence generation
    # methods" above.

### Inspect sequences (scripts/inspect_sequences.py)

    venv/bin/python scripts/inspect_sequences.py --alpha_0 1.0 --rl_lambda 0.5 --skip_nef

Builds BOTH a diagnostic figure (figures/inspect_sequences.pdf) AND a
human-readable, observation-level CSV (figures/inspect_sequences.csv by
default, built unless --skip_csv) covering prefix/suffix structure, running
trajectory, achieved-vs-target mean/std/p, and constraint checks (prefix
uniqueness across qids, binary exact-quota correctness, ITI
control/distract balance) -- no NEF/model dependency at all, so the CSV
builds fast regardless of --skip_nef. See build_inspection_csv's own
docstring for the full column list.

Reads exact filenames {task}_sequences.pkl/json from --seq_dir (default
task/sequences/) — NOT the _iid/_momentmatch suffixed branch outputs. To
inspect a branch's output, copy/rename into a temp subfolder first:

    mkdir -p task/sequences/_inspect_tmp
    cp task/sequences/continuous_momentmatch_sequences.pkl  task/sequences/_inspect_tmp/continuous_sequences.pkl
    cp task/sequences/continuous_momentmatch_sequences.json task/sequences/_inspect_tmp/continuous_sequences.json
    cp task/sequences/binary_momentmatch_sequences.pkl       task/sequences/_inspect_tmp/binary_sequences.pkl
    cp task/sequences/binary_momentmatch_sequences.json      task/sequences/_inspect_tmp/binary_sequences.json
    venv/bin/python scripts/inspect_sequences.py --seq_dir task/sequences/_inspect_tmp --skip_nef
    rm -rf task/sequences/_inspect_tmp   # clean up when done

`--gt_mode {true, running_mean}` (default: running_mean) controls the RMSE
panels' ground truth:
  - 'true': constant per trial (true_mean/true_p) — the original behavior.
  - 'running_mean' (default): per-observation moving target = the running
    sample mean of the observed stimulus stream so far. Note: the Bayes
    agent's response IS exactly this running mean by construction (the
    incremental-mean update fully overwrites any prior at n=1), so under
    this mode the Bayes curve should flatline at exactly zero for BOTH
    tasks — for binary this requires passing the running-mean trajectory
    through the same apply_binary_transform (Laplace smoothing) as the
    agent responses, or a spurious nonzero gap appears (this was a real
    bug, now fixed — if a future edit touches gt_traj construction for
    binary, keep the transform applied to both sides).
  - `|Δresponse|` panels never depend on gt_mode (no ground truth involved).
  - NEF cache path is keyed by gt_mode (`_running_mean` suffix) so switching
    modes doesn't silently reuse a stale cache.

### Run RL_lambda simulation (local)

    rm data/runs/test_sequences/test_sequences_responses.pkl
    venv/bin/python scripts/test_sequences.py \
        --run_models --tasks continuous binary \
        --alpha_0 1.0 --n_lambdas 100
    venv/bin/python scripts/test_sequences.py   # plot only

### Run NEF simulation (cluster)

    # After git pull on cluster:
    rm data/runs/test_sequences/nef_runs/nef_*.pkl 2>/dev/null
    # Precompute counting activities if needed:
    venv/bin/python models/counting_integrator.py --precompute_activities \
        --dataset task_continuous --n_neurons 200 --n_neurons_counting 1000
    venv/bin/python models/counting_integrator.py --precompute_activities \
        --dataset task_binary --n_neurons 200 --n_neurons_counting 1000
    python scripts/submit_nef_sequences.py   # submits 100 SLURM jobs
    # After completion:
    python scripts/collect_nef_sequences.py
    # Transfer locally then:
    venv/bin/python scripts/collect_nef_sequences.py
    venv/bin/python scripts/test_sequences.py   # plot

### Figure layout (scripts/test_sequences.py)
- Row 1 (A–G): Binary; Row 2 (H–N): Continuous; Row 3 (O–P): Cross-task
- A/H: RMSE vs obs split by true lambda quartile (Q1–Q4)
- C/J: |Δresponse| vs obs split by true lambda quartile
- F/M: true λ vs late RMSE scatter + regplot
- Splitting variable: true lambda from params_str (not late delta)
- ALPHA_0 = 1.0 hardcoded in run_nef_sequences.py

---

## Repository structure

```
evidence_integration/
  data/
    carrabin.pkl
    carrabin_original.csv
    yoo.pkl
    counting_activities_n{n}_nc{nc}_{dataset}.pkl
    runs/
      carrabin/      — RMSE fits + MLE fits + extras
      yoo/           — RMSE fits for non-NEF models
      refit/         — NEF responses/params/activities (yoo + carrabin)
    sim_db/          — MLE simulation database
    optuna/          — MLE Optuna SQLite databases
  models/
    math_models.py
    NEF.py
    counting_integrator.py
    RNN.py
  fitting/
    fit.py           — Optuna k-fold CV RMSE
    fit_mle.py       — MLE via shared simulation database
    model_params.py  — MODEL_PARAMS, MLE_PARAMS, NEF_N_NEURONS_VALUES
    submit.py
    collect.py
    losses.py
  utils/
    paths.py
    plot_style.py    — apply_style, get_palette, pvalue_to_stars, fit_power_law_params
    slurm.py
    carrabin_transform.py
    save_responses.py
  scripts/
    figure_carrabin_performance.py   — P group (1×3)
    figure_carrabin_variability.py   — V group (1×4)
    figure_carrabin_temporal.py      — T group (1×4)
    figure_carrabin_neural.py        — N group (1×4)
    figure_yoo_performance.py        — P group (1×3)
    figure_yoo_temporal.py           — T group (1×4)
    figure_yoo_neural.py             — N group (1×4)
    figure_carrabin.py               — legacy combined figure
    figure_yoo.py                    — legacy combined figure
    extras_carrabin.py               — PE dynamics, probe_pids, n_neurons_scan
    extras_yoo.py                    — NEF response noise simulations
  jobs/
    submit_probe_pids.sh
    submit_n_neurons_scan.sh
    submit_yoo_noise.sh
    submit_mle_fit.sh
  venv/
```

All new scripts go in scripts/. Never create scripts at the project root.
Figures save PDF only (no PNG/SVG).

---

## Current figure panel inventory

### figure_carrabin_performance.py (P group, 1×3)
| Panel | Code | Content |
|-------|------|---------|
| A | — | Task schematic |
| B | P1 | Estimation error (RMSE to hidden probability) |
| C | P2 | Model fit (RMSE to human responses); sig bars from NEF outward |

### figure_carrabin_variability.py (V group, 1×4)
| Panel | Code | Content |
|-------|------|---------|
| A | V2 | KDE of response variability (human only); per-pid lines |
| B | V2 | Model RMSE regplot vs human response variability |
| C | V3 | Test-retest reliability scatter (first vs second half); Human, NEF, NC |
| D | V1 | NLL boxplots; sig bars from NEF outward |

### figure_carrabin_temporal.py (T group, 1×4)
| Panel | Code | Content |
|-------|------|---------|
| A | T1 | RMSE to true_p vs observation |
| B | T2 | Mean |Δresponse| vs observation |
| C | T6 | Within-trial residual autocorrelation, lag 1–3 |
| D | T5 | Residual std growth across observations |

### figure_carrabin_neural.py (N group, 1×4)
| Panel | Code | Content |
|-------|------|---------|
| A | N1 | Decoded PE dynamics, 4 param combinations (α₀ × n_neurons) |
| B | N2 | PE vs response variability (r=0.97****); partial r excl. α₀=0.97**** |
| C | N3 | Response and PE variability vs fitted α₀ |
| D | N4 | Response and PE variability vs n_neurons scan |

### figure_yoo_performance.py (P group, 1×3)
| Panel | Code | Content |
|-------|------|---------|
| A | — | Task schematic (yoo_task.pdf) |
| B | P1 | Estimation error (RMSE to cumulative true mean) |
| C | P2 | Model fit (RMSE to human responses); sig bars from NEF outward |

Default: --run_folder yoo --nef_folder refit

### figure_yoo_temporal.py (T group, 1×4)
| Panel | Code | Content |
|-------|------|---------|
| A | T1 | Estimation error vs obs; shaded bands show weak/strong U-shape range (N_GROUP=10) |
| B | T2 | Mean |Δresponse| vs obs (sns.lineplot, CI across trials, obs ≥ 2) |
| C | T3 | Split-half λ reliability: regplot per source (Human + all models) |
| D | T4 | λ_model vs λ_human regplot; identity line; one line per model |

λ fitted via curve_fit A·n^(-λ), bounds [0,2], no smoothing, obs ≥ 2.
U-strength = mean(task_error[obs≥26]) − min(smoothed error curve, w=5).
Default: --run_folder yoo --nef_folder refit

### figure_yoo_neural.py (N group, 1×4)
| Panel | Code | Content |
|-------|------|---------|
| A | N5 | Weight-neuron activity vs obs, split by high/low λ group (N=10 each) |
| B | N6 | Mean weight-neuron activity vs mean |Δresponse| per obs (regplot); r=0.92**** |
| C | N7 | Fitted λ mediates activity change (right axis) and mean |Δresponse| (left axis) |
| D | N8 | Late |Δresponse| vs late estimation error (obs 21–30); Human + NEF |

Weight-on neurons: enc_dim_0 > 0.5 in error ensemble encoders.
Default: --nef_folder refit

---

## Environment

Always use: /home/psipeter/evidence_integration/venv/bin/python

Cluster: /dartfs-hpc/rc/home/n/f007qzn/
SLURM scripts: use pwd -P and export EVIDENCE_INTEGRATION_ROOT=${ROOT}.
NFS mount uses local_lock=none. Atomic rename used for simulation DB writes.

---

## Code conventions

- alpha_0, lambda_ (trailing underscore), gamma, eps_p, eps_r
- Merge order: PARAM_DEFAULTS < _NEF_FIXED < fitted Optuna params
- Read loss with _get_loss(perf_df) — never hardcode cv_loss_mean
- Run folder: always pass short name (e.g. yoo) — resolve_run_folder prepends RUNS_DIR
- --local runs must print JOB_COMPLETE as the final stdout line
- Python 3.11; pathlib via utils.paths; figures save PDF only
- New figure panels go inside existing figure_*.py scripts
- Do not compute metrics in extras scripts — save raw data, compute in figure scripts
- pvalue_to_stars, fit_power_law_params, smooth_curve, POWER_LAW_SMOOTH_WINDOW are in utils/plot_style.py

---

## Workflow guidelines

### Before making changes
1. Read the relevant files fully first.
2. Check fitting/model_params.py before touching models or fitting.
3. Propose a plan for structural changes before executing.

### NEF simulations
Never run NEF simulations via MCP tool calls — will time out.
Write a script and give the command to run on cluster.

### Figure iteration
After any figure change, render using pdftoppm and inspect with
filesystem:read_media_file. **Use this sparingly** — each image upload
consumes significant context. Prefer:
- Running analysis in shell_run_command to check numerical results first
- Only uploading an image when visual layout/style review is actually needed
- Deleting the temporary PNG immediately after inspection

Render command:
    pdftoppm -png -singlefile -r 150 figures/figure_X.pdf figures/_prev
    # then filesystem:read_media_file figures/_prev.png
    # then git clean -f figures/_prev.png

### Temporary analysis scripts
For exploratory analysis (scanning parameters, computing correlations):
- Write to scripts/_tmp_*.py
- Run via shell:run_command
- Delete immediately after: venv/bin/python -c "import pathlib; pathlib.Path('scripts/_tmp_X.py').unlink()"
- Never commit _tmp files

### Git
Generate a commit message and wait for confirmation. Never push without being asked.

### Context efficiency
- Prefer shell:run_command with Python -c for short computations over writing tmp files
- Use filesystem tools (read_text_file with head/tail) rather than loading full files
  when only a portion is needed
- When scanning over parameters, print a compact table rather than per-pid details
  unless the per-pid breakdown is specifically needed
- Avoid loading the full transcript for routine tasks; use conversation_search instead

---

## Workflow rules

### Suggesting vs implementing changes
When a question or observation implies that a code change *might* be warranted,
Claude must **describe the proposed change and ask for approval before writing
any code**. This applies especially to:
- Changes to figure aesthetics or panel logic (sorting, colouring, metrics)
- Changes to analysis methodology (metrics, transforms, thresholds)
- Any change that was not explicitly requested

Only implement immediately when the user has explicitly asked for a specific
change (e.g. "change X to Y", "add Z", "remove W").

---

## What NOT to do

- Do not use str_replace/create_file/view for anything under
  /home/psipeter/evidence_integration/ — they write to Claude's local sandbox,
  not this remote host, and fail silently (see compaction-reminder note at top)
- Do not add diederen, jiang, or usher back without explicit plan
- Do not add loss_type, shape_loss, joint_loss, beta hooks
- Do not use trial_seed / base_seed for NEF — seed = int(trial) directly
- Do not read cv_loss_mean directly — use _get_loss
- Do not create scripts outside scripts/
- Do not add NEF_synaptic, LMU counting variant, or ADM model name
- Do not double-apply the carrabin transform (NoisyCounting excluded)
- Do not pass a full path as run_folder — always use a short name
- Do not commit or push without being asked
- Do not run NEF simulations through MCP tool calls (will time out)
- Do not use RNN-based sigma as the noise metric — use qid-grouped response std
- Do not compute metrics in extras scripts — save raw data, compute in figure scripts
- Do not save figures as PNG or SVG — PDF only
- Do not upload figure images unnecessarily — use numerical checks first
- Do not promote generate_sequences_iid.py or generate_sequences_momentmatch.py
  output directly to the production {task}_sequences.{pkl,json} filenames
  without explicit go-ahead -- production is generate_sequences_hybrid.py's
  output (a deliberate per-task combination of both, chosen after PI
  discussion; see "Sequence generation methods"), not either pure method
  on its own
- Do not add a seed search / best-of-N ranking to generate_sequences_iid.py
  or generate_sequences_hybrid.py -- deliberately absent from both; any
  outcome-dependent seed selection reintroduces the conditioning/confound
  this project spent real effort establishing and then avoiding (see
  docs/sequence_design_open_questions.md)
- Do not reintroduce dev-only override knobs (testMode, nTrialsDefault,
  trialItiMs, showTutorial) into buildAndRun/timeline-builder.js/config-base.js
  — these were deliberately removed along with index-dev.html; any test-only
  need for different config values belongs in src/test-harness.js building a
  modified config object, never inside the production code path itself
- Do not redirect non-Prolific participants to a same-origin file (e.g. via
  jatos.endStudyAndRedirect) after ending a JATOS study session — confirmed
  broken on a real MindProbe pilot run ("you have no access rights" trying
  to serve public/exit-complete.html post-session-end; see "Exit/redirect
  and data-saving architecture"). Only redirect to an EXTERNAL domain
  (Prolific); for everyone else, use finish-session.js's DOM-update-in-place
  approach instead
- Do not delete generate_sequences.py or remove any of the specific
  functions it exports — it has no CLI/generation logic of its own anymore,
  but it is a genuine, live shared-utilities dependency of BOTH
  generate_sequences_iid.py and generate_sequences_momentmatch.py (check
  both scripts' imports before touching anything in this file)
- Do not reintroduce a design where one qid means one fixed (prefix, target)
  pair in generate_sequences_momentmatch.py — this was the actual mechanism
  behind a real, confirmed bug (two different qids ending up with an
  identical realized prefix by chance; see "Sequence design"). Prefix
  identity and target level must stay independent axes, matched via
  optimal_matching, not tied 1:1 or paired via a greedy heuristic (greedy
  was tried and rejected — see that function's docstring for the measured
  failure mode)
- Do not reuse the SAME output filename across scripts.inspect_sequences.py
  and scripts/inspect_iid_sequences.py — they produce structurally
  different figures (multi-agent comparison on ONE representative pool
  member's aggregated curves vs. one-agent-only per-pool-member thin lines
  + mean/CI) and a real mistake this session silently overwrote one with
  the other by giving both the same --out_pdf. Use distinct, descriptive
  names (e.g. _running_agents.pdf vs. _running_meanagent.pdf).
- Do not assume prefix_length is always > 0 when writing a NEW prefix-
  uniqueness/collision check against sequences.json data — binary's
  no-prefix branch (generate_binary_sequences_no_prefix) legitimately
  writes prefix_length=0, and values[:0] is the same empty tuple for every
  trial regardless of qid, which will false-positive as a collision
  against any check that doesn't explicitly skip this case (this bit BOTH
  generate_sequences_pool.py's verify_pool and inspect_sequences.py's
  build_inspection_csv independently this session — see "Tutorial
  redesign..." above for both fixes; a third implementation of this same
  check would need the same guard).
- Do not treat BONUS_DECAY as a simple 0-100-scale constant without
  checking what it's actually being multiplied against — totalError is a
  SUM across N_OBS_TO_RUN observations, not a single observation's error;
  see "Tutorial redesign..." above for the real bug this caused (reward
  silently 0 for nearly every real response) and why BONUS_DECAY is now
  `1 / DEFAULTS.N_OBS_TO_RUN`, not a flat literal.
