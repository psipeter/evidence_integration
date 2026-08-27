# docs/HISTORY.md — online task build history (task/, retired; task_backend, active but settled build-out)

**This file exists so CLAUDE.md can stay focused on the CURRENT
architecture without losing the full design rationale, rejected
alternatives, and real-bug narratives that got the project here.** Two
distinct systems' history live here, in chronological order:
`task/` (the JATOS/MindProbe pipeline, fully RETIRED -- nothing in that
first part of this file describes anything still active) and
`task_backend` (the current, ACTIVE system -- its own section, appended
later once its initial build-out and first two real Prolific pilot
rounds settled, holds the SAME kind of settled narrative even though the
system itself keeps evolving). See CLAUDE.md's "Online task: task_backend"
section for what's actually live today -- this file is rationale/history
for both systems, CLAUDE.md is current state for whichever is live.
`task_backend/TODO.md` no longer exists -- its content is the
task_backend section appended at the end of this file, not a separate,
still-growing document.

Read this file when:
- A future session needs to understand WHY task_backend exists at all
  (the pilot #3 JATOS incidents and platform evaluation below).
- Something about task_backend's design references a decision that was
  originally made for task/ and inherited (e.g. the per-participant pool
  hashing scheme, the tutorial redesign, the bonus formula) -- the
  reasoning and rejected alternatives live here, not in CLAUDE.md.
- You're doing archaeology on a specific past bug or design debate (the
  sequence-generation-method comparison, the exit/redirect architecture,
  a task_backend pilot decision, etc.) and want the full account rather
  than a summary.

This is a straight move, not a rewrite -- the content below is exactly
what CLAUDE.md's own "Active datasets" / "New task (task/)" section
contained before this split, preserved verbatim (including its own
internal "this session" framing, which now reads as historical rather
than current -- that framing is left as-is rather than retroactively
edited, per this project's own convention of never rewriting past
narrative to look tidier than it was).

**Companion section**: "Sequence design: open questions (i.i.d. vs.
quota/moment-matching)" (below, right after "Sequence generation methods
(task/)") covers the i.i.d.-vs-quota/moment-matching behavioral-
methodology question in depth -- originally its own standalone file
(`docs/sequence_design_open_questions.md`), merged directly into this
document since it's purely an extension of the section it sits next to.

## Contents of this file (in the order they appear below)

- Online-task overview: continuous/binary tasks, jsPsych/Vite/JATOS
  infrastructure, naming conventions, stable architecture
- Consent screen, binary slider, tutorial designs (pre-redesign and the
  redesign itself)
- Tutorial redesign, bonus/error system, and binary no-prefix sequences
  (the session that introduced the current bonus formula and fixed the
  BONUS_DECAY miscalibration bug)
- Per-participant sequence pool (the hashing/assignment scheme
  task_backend inherited)
- Sequences.json schema, tutorial derivation, participant-data columns
- Sequence generation methods (the iid/momentmatch/hybrid debate)
- **Sequence design: open questions** (i.i.d. vs. quota/moment-matching --
  the deeper behavioral-methodology investigation that extends the
  section directly above it; originally a standalone file, merged in)
- Open items (as of the last task/ session)
- Task testing architecture (the dev-page removal, index-test.html)
- E2E array-bug fix, pool-assignment scenario removal, mini-pool test
  jzips, bonus/std findings
- **Pilot #3 real-participant incidents -> JATOS reliability
  investigation -> decision to prototype a Gorilla migration** (the
  incident investigation that ultimately produced task_backend)
- **Own-backend decision (Supabase)** -- why Gorilla/Cognition.run/
  Labvanced were all rejected in favor of building task_backend
- Tab-visibility handling for observation timeouts
- Exit/redirect and data-saving architecture (the JATOS save-mechanism
  evolution, PROLIFIC_CODES history, pre-deployment checklist, pilot
  data file inventory)
- **task_backend: build history and settled decisions** (folded in from
  task_backend/TODO.md -- schema/RLS design, the Edge Functions, the
  tutorial redesign and "Correct answer" panel, the bonus formula and its
  BONUS_DECAY history, hosting/deployment, the first two real Prolific
  pilot rounds, checkpoint reliability hardening, and the production-
  readiness review)

---

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

**SUPERSEDED (found while investigating a chat session's bonus questions --
see "This session: E2E array-bug fix..." below for what actually changed
in THIS session): the formula narrated in this bullet list
(`reward = max(0, 100 - BONUS_DECAY * totalError)`, one reward per TRIAL
from a pre-summed error) is NOT what the code currently does. At some
point AFTER the session this bullet list describes but BEFORE the most
recent chat session, bonus-continuous.js's real formula changed to a
PER-OBSERVATION reward: `normError = rawError / MAX_POSSIBLE_ERROR;
reward = max(0, MAX_REWARD * (1 - BONUS_DECAY * normError))`, computed
once per observation and SUMMED for the trial/tutorial total (see
`computeResponseReward`/`computeTrialReward` in bonus-continuous.js --
that file's own docstring has the full replacement rationale). Current
parameters: `MAX_REWARD = 2` cents (lowered from 3 this session, see
below), `BONUS_DECAY = 15`, `MAX_POSSIBLE_ERROR = 100`. This whole bullet
list is kept below for the ERROR_MODE/chart/methodology narrative, which
is still accurate -- only the specific reward FORMULA shown is stale.**
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

**Pending / not yet done**: `task/evidence-integration-binary.jzip` was
STALE (predated an earlier session's binary tutorial/bonus/chart work) --
**RESOLVED as of this chat session**: both `evidence-integration-
continuous.jzip` and `evidence-integration-binary.jzip` were rebuilt fresh
(new UUIDs) against the full 200-member production pool and the current
bonus formula (MAX_REWARD=2) -- see "This session: E2E array-bug fix..."
below. Re-verify staleness before trusting this note in a future session;
it's accurate only as of when it was written.

Sequence design: **8x4 (32 trials) hybrid production design is CURRENT**,
  superseding the earlier 6x4 pure-momentmatch pilot described below.
  Generated via task/generate_sequences_hybrid.py (see "Sequence generation
  methods" below) -- binary keeps quota/momentmatch construction unchanged
  (no seed search); continuous uses a genuinely i.i.d., unrescaled suffix
  instead. Chosen at 8x4 rather than 10x4 specifically to reduce
  participant time. Promoted to task/sequences/{continuous,binary}_sequences.
  {pkl,json}; the older 6x4 sequences (and the pure _iid_/_momentmatch_
  branch outputs) remain recoverable via git history / their own
  differently-named files, not deleted. **See "Sequence design: open
  questions" above for the full investigation and decision rationale** --
  the quota-vs-i.i.d. confound this decision was based on, the hybrid
  design's own std-guard tuning, and everything checked before promoting.

  **RESOLVED: production std is 10, not 15 as this section previously
  stated as a target/default.** The real 200-member production pool has
  `true_std=10.0` uniformly across all 200 continuous members (confirmed
  directly), and pilot #3 is now LIVE on this pool -- std=10 is the
  actual, confirmed, currently-deployed value. `generate_sequences_pool.py`'s
  own `--std_fixed` CLI default is still 15 and was NOT changed to match
  (a code change wasn't requested) -- if the pool is ever regenerated
  using that script's bare defaults, it will silently produce std=15
  sequences, not the std=10 currently live. Pass `--std_fixed 10`
  explicitly for any future regeneration intended to match current
  production, and update the script's own default at that point if this
  keeps being a trap.

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
  empirically (see "Sequence design: open questions" below and this file's
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

### Sequence design: open questions (i.i.d. vs. quota/moment-matching)

**Status as of this writing: unresolved, real trade-off, not yet decided by the
PI.** Production currently uses the quota/moment-matched branch
(`generate_sequences_momentmatch.py`) -- the promoted 10x4 sequences already
live at `task/sequences/{task}_sequences.{pkl,json}` and are ready to ship.
Nothing in this memo blocks that from going out. This section exists so a
future session (or the PI) doesn't have to re-derive any of the following
from scratch, and so nobody accidentally "fixes" something here that was a
deliberate, considered trade-off rather than an oversight.

See "Sequence generation methods (task/)" above for the mechanics
(prefix/target independence, the collision bug, optimal matching, etc.) --
this section assumes that context and focuses on the *behavioral/
methodological* questions layered on top of it. (Originally its own
standalone file, `docs/sequence_design_open_questions.md` -- merged into
this document since it's purely an extension of the section immediately
above; nothing lost, just no longer a separate file.)

---

#### 1. The triggering observation

A real pilot participant reported recognizing, partway through a sequence,
approximately where the true mean/probability was -- and using that to
discount later observations as "outliers" rather than genuinely updating on
them. This prompted a full investigation into whether the quota/moment-match
construction method makes this a *rational* strategy (not a bias) for
participants to discover, and whether that would bias derived behavioral
metrics (particularly the power-law decay rate lambda this project's whole
model-comparison pipeline is built around).

#### 2. Is the quota confound real? Yes -- but not for the reason first proposed

**What turned out to be wrong**: the first framing was "seeing a streak
should make you predict a reversal" (classic gambler's-fallacy-is-correct
logic). This is only valid if the participant *knows* the quota/true
parameter in advance. They don't -- it's exactly what they're estimating --
so this specific directional claim doesn't hold up once corrected.

**What actually holds up, worked out rigorously (both analytically and via
direct simulation of the real generation code)**: the *variance* of the
running estimate, conditional on the true hidden parameter, shrinks faster
under quota-construction than under honest i.i.d. sampling -- provably so for
binary via the hypergeometric-vs-binomial finite-population-correction
factor `(N-n)/(N-1)`, and confirmed empirically for continuous via direct
simulation of the real `generate_sequences_momentmatch.py` code:

```
Continuous (target_mean=50, std=15): variance ratio (quota/iid) at each obs
  n=1: 0.96   n=5: 0.83   n=10: 0.38   n=14: 0.08   n=15: 0.00

Binary (target_p=0.6): variance ratio (quota/iid)
  n=1: 1.01   n=5: 1.56*  n=10: 0.44   n=14: 0.07   n=15: 0.00
  (*binary's prefix is INDEPENDENT of target, so early obs are noisier
  than i.i.d., not quieter -- see point 3 below)
```

The mechanism that survives careful scrutiny isn't "predict the opposite of
what you've seen" -- it's closer to **"whatever your running estimate is by
roughly the midpoint of a trial, trust it heavily, because the back half is
constructed to correct toward the true value almost regardless of how
wrong the front half was."** Confirmed directly:

```
Correlation between (how wrong you were at obs 10) and (how much the rest
of the trial corrects it):
  i.i.d.:   r=0.61  (loose -- no guaranteed correction)
  quota:    r=0.998 (near-deterministic -- correction is baked in)

Among the WORST 10% of early (obs-10) guesses:
  i.i.d.:   error 9.63 -> 6.37 by the end (partially rescued)
  quota:    error 5.85 -> 0.08 by the end (almost perfectly rescued)
```

This means terminal accuracy under quota is close to guaranteed by
construction, largely independent of how well a participant is actually
integrating evidence -- which undermines using terminal convergence (or
model fits that reward it) as a signal of genuine integration quality.

#### 3. The prefix/suffix trilemma -- no single design satisfies all three goals

Three separate, real design goals turn out to be in tension, not just two:

1. **Statistical cleanliness** (no discontinuity between prefix and suffix --
   the whole 15-obs sequence looks like one honest distribution) -- needs the
   prefix to be tied to *a* target.
2. **Diversity** (many distinct targets, not capped at however many distinct
   prefixes exist) -- needs the prefix *not* tied to any one target.
3. **Repeat-based noise-isolation metrics (T5/T6, V-group)** -- these compute
   `residuals = response - mean(response | pid, obs, qid)`, which only
   cleanly isolates internal noise from stimulus-driven variance if repeats
   of a qid share **literal, identical stimulus history**, not just a
   shared abstract target. That literal repetition is also exactly what
   creates associative-memorization risk (recognizing "I've seen this exact
   opening before, I remember the answer").

The current momentmatch design (prefix independent of target) resolves (2)
and partially (3)'s memorization risk, at the cost of (1) -- this was a
deliberate, informed trade, not an oversight. Attempting to instead draw the
prefix from the *matched* target distribution (to fix (1)) was tried and
reverted: it reintroduces (2)'s diversity cap and creates a much sharper,
more literal (3) memorization risk (identical 4-observation openings,
repeated verbatim across a participant's own session).

**A real, previously-undetected bug found along the way, since fixed**:
this same literal-repetition mechanism also affected
`generate_sequences_iid.py`'s own prefix generation (drawn from the
matched target distribution, no uniqueness check) -- empirically, **9 of 10
random seeds produced real prefix collisions across different qids** (only
6-8 distinct prefixes out of 10 expected). This was never checked before
this investigation, since all prior collision-bug fixing this project did
was specific to `generate_sequences_momentmatch.py`. **Fixed** via
`_draw_unique_binary_prefix` (active dedup on the base draw, with a
fresh-redraw fallback for the rare case a deterministic mirror collides --
confirmed to actually fire in testing, not just theoretical) plus a
safety-net uniqueness assertion; verified 10/10 seeds now produce the
expected number of distinct prefixes, previously 9/10 failed.

**No design was found that fully satisfies all three goals simultaneously.**
This is presented as a real, acknowledged trade-off for the PI to weigh, not
a solved problem.

#### 4. A model-recovery-based selection scheme was proposed and rejected

Idea considered: instead of quota-matching raw stimulus composition,
generate many i.i.d. sequence sets and keep whichever ones let reference
agents (Bayes/Mean, RL_lambda across various alpha/lambda) most accurately
recover their own known true parameters.

**Rejected** on the grounds that this is very likely the same underlying
mechanism as quota (outcome-conditioned selection from many i.i.d. draws),
just conditioned on a different property (parameter-recovery accuracy
instead of composition-near-target) -- this document's own earlier finding
(see "Sequence generation methods (task/)" above) already established that
k-constrained rejection sampling and quota sampling are "the same
underlying object at different points on one continuum"; this proposal
doesn't escape that continuum, it just picks a new point on it. It also
introduces a *new*, more specific risk: sequences selected because *these
particular reference models* recover cleanly on them would then be used to
test whether real human behavior resembles those same reference models --
a real circularity risk not present in quota's model-agnostic conditioning
on raw composition.

#### 5. Ground truth choice (`true_mean`/`true_p` vs. `running_mean`) interacts
   with all of the above, and doesn't cleanly resolve it

`gt_mode='running_mean'` (in `scripts/inspect_sequences.py`, the retired
predecessor of the current `scripts/plot_sequences.py`) scores a response
against the running mean of *visible* data at that same moment -- never
referencing the hidden true parameter at all. This structurally sidesteps
the quota confound (which is entirely about the relationship between
visible data and the hidden truth) for agents that are themselves close to
a raw running-mean tracker -- confirmed directly: this project's own
"Bayes"/"Mean" agent is *tautologically* perfect under `running_mean`
scoring (near-zero error always, both quota and i.i.d.), making that
gt_mode uninformative for that specific agent. For an agent with genuine
independent dynamics (RL_lambda), `running_mean` scoring not only removes
the quota advantage, it **reverses it** (quota scores slightly *worse* than
i.i.d. under this metric) -- confirmed empirically.

This is a real methodological fork with genuine precedent on both sides:
`true`-referenced scoring is standard in the Bayesian-updating/
volatility-learning literature (Behrens, Glaze, Nassar, Yu & Cohen), which
is explicitly interested in "how well do humans approximate the objectively
optimal observer of the real environment" -- unanswerable without
referencing the real parameter. `running_mean`-referenced scoring resonates
with "decisions from experience" literature, which argues a rational agent
should be judged against the actual finite sample it saw, not an
inaccessible population parameter. Neither is simply "more correct"; they
answer different questions. **Recommendation if this needs to go in a
paper**: report both, explicitly labeled by what each one tests, rather
than picking one silently.

#### 6. Empirical results from `scripts/inspect_iid_sequences.py`

Built this session specifically to investigate all of the above with real
numbers rather than argument alone. Generates N independent i.i.d. sequence
sets (one fully independent draw per simulated participant, via the real
`generate_sequences_iid.py` code -- see that script's own docstring for
exactly what "independent" means here), simulates a chosen agent (Mean or
RL_lambda) on each, and reports fitted-lambda mean/std/range plus
split-half reliability across the simulated population. (`inspect_iid_
sequences.py` was later itself archived once task_backend consolidated
onto one generation method and `scripts/plot_sequences.py` replaced it --
see this document's own note on that consolidation, elsewhere in this
file, and `archive/archive_readme.md`. This investigation's numbers stand
on their own regardless.)

**Headline results, n=50 simulated participants, Mean agent** (true
lambda=1 by construction -- Mean is RL_lambda's own special case
alpha_0=1, lambda=1). **Updated** after fixing the prefix-collision bug
documented in Section 3 (continuous was never affected by that bug and is
unchanged; binary moved both closer to the true value and substantially
tighter once the spurious collision-driven noise was removed):

```
                fitted lambda (mean+/-std)   range          split-half reliability
continuous      1.13 +/- 0.29                [0.60, 1.85]    r=0.82, p<0.0001
binary          0.84 +/- 0.10                [0.60, 1.00]    r=0.41, p=0.003
```

(Binary, pre-fix, for reference on how much the bug mattered: 0.77 +/-
0.17, range [0.38, 1.28], split-half r=0.76 -- std dropped ~40% and the
mean moved closer to the true value of 1 purely from removing the
collision noise. **But reliability dropped**, from r=0.76 to r=0.41 --
the opposite direction from the mean/std improvement, and worth taking
seriously rather than glossing over: this suggests the pre-fix "good"
reliability was likely partly artifactual -- a participant whose random
seed happened to produce many prefix collisions would show a similarly
biased pattern across their WHOLE session (both halves), creating spurious
agreement between first-half and second-half fits that isn't genuine
signal. Once that shared distortion is removed, the true underlying
split-half reliability for binary is real but much more modest than it
first looked.)

Split-half reliability is strong for continuous (r=0.82) despite every
participant seeing completely independently-randomized sequences --
i.i.d. sampling noise alone does *not* wash out the ability to detect a
consistent decay signature there. Binary's reliability is real but far
more modest (r=0.41) once the collision-driven artifact is removed -- see
above. In both cases, **the population mean itself sits meaningfully off
the true value of 1** (especially binary, even post-fix), and the spread
across participants is substantial -- a real cost of not smoothing/
selecting sequences at all.

**Comparison against the real (quota, seed-searched) production
sequences**, same Mean agent, single dataset (not averaged over many
draws). **Updated** with the post-fix i.i.d. mean:

```
                i.i.d. mean (n=50, post-fix)    production (quota, n=1)
continuous      1.133                            1.044   (quota much closer to true=1)
binary          0.842                            0.758   (quota now further from true=1)
```

This flips part of the original conclusion. Pre-fix, binary looked like
"quota does nothing measurable" (0.765 i.i.d. vs 0.758 quota -- essentially
identical). Post-fix, the i.i.d. average (0.842) is actually *closer* to
the true value than this specific quota draw (0.758) -- quota no longer
looks neutral-to-slightly-better for binary, it looks slightly *worse*
than the average i.i.d. participant would do. Continuous still clearly
favors quota. Important caveat unchanged from before: this conflates quota
construction itself with the 1000-try seed search that selected this
specific sequence set for smoothness -- an *unselected* single quota draw
might look different again. Not yet checked; flagged as a natural
follow-up if this matters for a real decision.

**A second, independent bias was found and isolated**: binary's poor
lambda recovery (both i.i.d. and quota alike) is substantially explained by
the Laplace-smoothing transform (`utils/binary_transform.py`) applied
uniformly to every non-exempt model's binary output project-wide -- its
stated justification ("optimal Bayesian estimate under a uniform prior")
is specific to a raw sample-mean estimator and doesn't clearly generalize.
Confirmed by comparing fits with vs. without the transform:

```
                  WITH transform   WITHOUT transform   true lambda
Mean, binary      0.758            0.903                1.0
RL_lambda(1, 0.5) 0.284            0.430                0.5
RL_lambda(0.5,0.3) 0.158           0.323                0.3
```

Removing the transform closes 60-90% of the gap to the true value in every
case checked, at the cost of somewhat higher variance. **This bias is
orthogonal to the i.i.d.-vs-quota question** -- it affects both branches
equally, since it's applied after sequence generation, not as part of it.
`scripts/inspect_iid_sequences.py` (now archived) supported
`--skip_binary_transform` to isolate this. Not resolved/decided whether
production analyses should use the transform or not -- flagged, not
fixed, since it affects more than just this investigation (it's already
used project-wide).

**RL_lambda noise sensitivity** (as predicted going in -- smaller lambda
means alpha decays slower, so individual noisy observations keep mattering
later into a trial): confirmed clearly noisier than the Mean agent,
especially for binary, and the specific bias/noise trade-off shifts in a
non-obvious way with alpha_0 (one config showed large systematic bias with
*low* variance; another showed smaller bias with *higher* variance) --
this isn't simply "smaller lambda -> uniformly more noise," alpha_0 is
doing real, not-yet-fully-characterized work too. **Updated after the
prefix-collision fix** -- same qualitative pattern holds, numbers moved in
the same direction as the Mean agent above (closer to true, tighter std):

```
                        WITH transform          WITHOUT transform
                     (pre-fix -> post-fix)    (pre-fix -> post-fix)      true lambda
RL(1.0, 0.5) binary  0.284+/-.164 -> 0.367+/-.088   0.430+/-.235 -> 0.561+/-.134   0.5
RL(0.5, 0.3) binary  0.158+/-.073 -> 0.189+/-.042   0.323+/-.117 -> 0.381+/-.064   0.3
```

continuous unchanged in both configs (0.574+/-.175 and 0.753+/-.048
respectively) -- confirms, again, that continuous was never touched by
this bug at all; only binary's fits moved.

**Direct quota-vs-i.i.d. split-half reliability comparison, same
methodology on both sides**: `scripts/inspect_sequences.py` (now archived;
its "running_agents"-style aggregation logic lives on in `scripts/
plot_sequences.py`'s `across_models` branch) had a 3rd panel column
(`compute_split_half_reliability`) that swept 50 different RL_lambda
ground-truth values across the production quota sequences and fit/
correlated split-half lambda exactly the way `inspect_iid_sequences.py`
did for i.i.d. -- same `fit_lambda_mid`/`split_half_lambda` calls, not a
reimplementation, so the numbers are genuinely comparable, not just
visually similar:

```
                    quota (production, seed-searched)     i.i.d. (50 sims, post-fix)
binary   split-half   r=0.998, p=8e-59                    r=0.41,  p=0.003
continuous split-half r=1.000, p=7e-75                    r=0.82,  p<0.0001
```

Quota's reliability is essentially perfect, not just better. This is the
sharpest, most concrete evidence in this whole section for the trade-off at
its center: near-perfect reliability isn't a separate nice property of
quota, it's the direct, mechanical consequence of the same seed search
that explicitly optimizes every trial's `|Δresponse|` curve for smoothness
(Section 2) -- if every individual trial is already engineered to be
well-behaved and track its target with minimal noise, splitting trials
into any two halves gives nearly identical aggregate curves almost
regardless of which specific trials land in which half. High reliability
and behavioral realism are pulling in opposite directions here, not
independent properties you get to have both of.

#### 7. What it would take to actually serve unique i.i.d. sequences per
   participant, if that path is ever chosen

Full breakdown was worked through in chat; summary:

- **Sequence delivery**: currently a build-time static import
  (`config.js` imports `{task}_sequences.json` directly, baked into every
  participant's identical bundle). Serving per-participant pools requires a
  real architecture change: bundle a pool of files as static assets, fetch
  the assigned one at runtime, and make the app's bootstrap sequence async
  on that fetch resolving. Needs live confirmation that JATOS actually
  serves additional static files from a study's asset directory via
  relative-path `fetch()` the way this assumes -- not yet verified against
  real JATOS. **(Superseded: task_backend, built later, resolved this
  entirely differently -- one combined JSON array of 200 members, bundled
  eagerly at build time via `import.meta.glob`, no runtime fetch at all --
  see this document's "Per-participant sequence pool" section above and
  CLAUDE.md's current "Online task: task_backend" section.)**
- **Pool generation**: **done** -- `task/generate_sequences_pool.py`
  generates N independent sequence sets per task and writes each to its
  own `{task}_{NNNN}_sequences.{pkl,json}`, with a built-in verification
  pass (member count, per-member prefix uniqueness). Thin wrapper around
  `generate_task_sequences_iid`, same as `inspect_iid_sequences.py`'s own
  `simulate_participants` -- no new generation logic. Not yet wired into
  anything downstream (asset bundling, runtime fetch, assignment,
  provenance recording, `parse_results.py` are all still separate,
  not-yet-started steps).
- **Assignment**: recommend a stateless hash of participant ID mod pool
  size (e.g. Prolific PID) -- avoids needing any server-side coordination or
  counter, which would carry real concurrency risk (two participants
  starting simultaneously racing on a shared counter).
- **Recording what a participant saw -- better news than expected**: the
  raw per-observation JATOS export *already* includes `value` (and, from
  direct inspection of real pilot files, `true_mean`/`true_p`) -- jsPsych
  automatically records trial parameters. The actual gap is in
  `parse_results.py`, which currently *discards* these and instead
  re-derives `value` via a lookup against the single shared sequence file
  (documented explicitly in that script's own docstring), assuming
  `(task, trial)` uniquely determines it -- true only while everyone shares
  one file. The fix is smaller than it first appears: extract these fields
  directly from each participant's own raw row instead of re-deriving them,
  which is *also* a general robustness improvement (removes an existing,
  if currently harmless, fragility where a regenerated sequence file could
  silently produce mismatched historical lookups). Worth confirming exactly
  which mechanism currently puts `true_mean`/`true_p` onto observation rows
  (direct trial parameter vs. a retroactive `jsPsych.data.addProperties()`
  call, which has bitten this project before) before relying on it --
  see this document's own build history for that mechanism. Recording a
  session-level `pool_index` property in addition is cheap and removes any
  ambiguity about provenance. **(Superseded: task_backend's own schema
  records `value`/`true_mean`/`true_std`/`true_p`/`qid`/`pool_index`
  directly on every checkpoint row from the start -- see CLAUDE.md's
  current "Backend schema" subsection.)**
- **Cross-participant analysis**: a genuine methodology change, not just
  plumbing -- once trial index no longer means the same target for
  everyone, anything that aggregates "by trial number" needs to instead
  align by target value or by qid-within-that-participant's-own-pool-member.
- **Diagnostics**: the (now-archived) `inspect_sequences.py` assumed one
  shared file; needed to inspect one representative pool member or
  aggregate across the whole pool (the aggregation machinery for the
  latter existed in `inspect_iid_sequences.py`, also now archived --
  `scripts/plot_sequences.py`'s two branches, `across_models`/
  `across_pids`, are the current equivalent, built against task_backend's
  real deployed pool from the start).
- **Ongoing costs worth naming**: total asset storage scales with pool
  size; debugging a specific participant's data now requires knowing which
  pool member they got; this becomes real standing infrastructure to
  maintain for the life of the study, not a one-time script.

#### 8. Where this leaves things

Nothing here should be read as "quota is wrong, switch to i.i.d." or the
reverse. Every path has real, now-quantified costs:

- **Quota (current production)**: real behavioral confound established with
  numbers, not hand-waving (Section 2), plus a fundamental prefix-design
  trilemma with no fully clean resolution (Section 3), plus near-perfect
  split-half reliability (r=0.998-1.000) that is now confirmed to be the
  direct, mechanical byproduct of the same seed-search smoothing that
  drives the confound in Section 2 -- not a separate advantage (Section 6).
- **Pure i.i.d.**: no behavioral confound from construction, but
  substantially noisier lambda-recovery even averaged over many
  independent draws (Section 6), and would require substantial new
  infrastructure to actually deploy with per-participant uniqueness
  (Section 7) -- or, if deployed with one shared i.i.d. draw (the simpler
  option), inherits the "got lucky/unlucky with the one seed" risk that
  per-participant randomization was meant to avoid. The prefix-collision
  bug (Section 3) is fixed; a pool-generation tool
  (`task/generate_sequences_pool.py`) exists and is verified, though
  nothing downstream (asset bundling, runtime fetch, assignment,
  provenance recording, parse_results.py) has been built yet.
- **Model-recovery-based selection**: rejected (Section 4) as likely just
  quota again, with an added circularity risk.

This is a real trade-off for the PI to make with full information, not a
default to quietly pick. The current 10x4 quota-based production sequences
are already generated, verified, and ready to ship regardless of how this
resolves -- nothing here is blocking deployment.

**Note added when this section was merged into docs/HISTORY.md**: this
whole investigation is about task/'s own retired generation methods
(iid/momentmatch). task_backend resolved the underlying i.i.d.-vs-quota
question with a per-task split rather than picking one side outright
(binary keeps quota/momentmatch construction unchanged; continuous uses a
genuinely unrescaled i.i.d. suffix) -- see "Sequence generation methods
(task/)" above ("task/generate_sequences_hybrid.py") for that resolution,
and CLAUDE.md's current "Sequences" subsection for what's actually live
today. Everything above remains accurate as a record of the investigation
that led there.

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
  and "Sequence design: open questions" above for the full investigation
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
- **std=10 confirmed as the actual production value -- RESOLVED**. The
  real 200-member `sequences_pool/` has `true_std=10.0` uniformly across
  all 200 continuous members; pilot #3 is now LIVE on this pool, which
  settled the question -- std=10 is correct/current, not std=15
  ("Sequence design" section above updated to match). Remaining loose
  end: `generate_sequences_pool.py`'s own `--std_fixed` CLI default is
  still 15 and was deliberately left unchanged (not asked for) -- pass
  `--std_fixed 10` explicitly for any future regeneration meant to match
  current production.
- **dev-results/ test-artifact accumulation** -- scenarios in
  test_browser.mjs that never check saved files never call cleanup, so
  every per-trial append from those runs accumulates indefinitely
  (~2000+ stale files found and cleaned once this session -- see "This
  session..." above). Not fixed at the root; a suite-level teardown step
  would close this properly if it keeps mattering.

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

### This session: E2E array-bug fix, pool-assignment scenario removed, mini-pool test jzips, bonus/std findings

**test_browser.mjs: stale array-assumption bug found and fixed** -- three
scenarios ("3 timeouts: session terminated", "Completes all trials...",
and the Prolific-redirect scenario) asserted `Array.isArray(saved) &&
saved.length > 0` on whatever file a `waitForNewResultFile` helper
returned after a save click. This predates the incremental per-trial-
append architecture (see "Exit/redirect and data-saving architecture"
above) -- the app hasn't sent one full-array dump per session in a long
time; every append (per-trial AND the final completion marker) is now a
single small object, and dev-server.js writes one file per POST. The
array check could never pass anymore, and "grab whichever new file shows
up first" could also grab an unrelated per-trial append instead of the
actual completion marker. Confirmed directly: failing files were valid,
non-empty single objects, exactly as designed -- not data loss, a stale
test assumption. Fixed by replacing `waitForNewResultFile` with
`collectNewResultRows(before, predicate, timeout)`, which polls, parses
EVERY new file (not just the first), and returns once some row satisfies
`predicate` (e.g. `row.progress === 'finished'`/`'terminated'`) --
`rows.find(predicate)` then picks the SPECIFIC row relevant to the
assertion, and `cleanupResultFiles(files)` deletes every file collected,
not just the matching one. "Completes all trials" also had its
`snapshotResultFiles()` call moved to session start (not right before the
final click), since observation rows are appended throughout the whole
session now, not bundled at the end -- the "has observation rows" check
needs the merged full-session set to mean anything. All three affected
scenarios re-verified clean (chromium/continuous, twice).

**dev-results/ had ~2000+ stale result_*.json files accumulated** across
many past runs -- scenarios that never check saved files (e.g. "Normal
submit", the timeout-demo ones) never call cleanup at all, so every
per-trial append from those runs just sits there forever. Cleaned up once
this session (safe: `result_*.json` is exclusively test-output naming,
distinct from every named real pilot file -- see "Pilot data files"
below) but the underlying gap is NOT fixed -- still worth a suite-level
"delete everything created since server start" step if this keeps
mattering.

**"Pool assignment" scenario REMOVED from test_browser.mjs entirely**,
replaced by `task/test_pool_assignment.mjs` (plain Node, no Playwright, no
Vite, runs in well under a second). The old scenario ran THREE full
tutorial+session flows through real Chromium just to check pool-hashing
determinism/range/consistency and embedded-field presence -- all
checkable without a browser at all: `poolIndexForParticipant` (in
timeline-builder.js) is a pure DJB2-style string hash with zero DOM/
jsPsych dependency, and the embedded true_mean/true_p/value fields are
read straight off the pool's own JSON by build-trial-timeline.js, so
checking the real sequences_pool/*.json files directly IS checking what a
real session would see. The old scenario was also the source of a
genuine, never-resolved "Target page, context or browser has been closed"
crash during its own tutorial -- investigated at length (per-iteration
fresh browsers, intercepting the real app.prolific.com redirect its
`testUrlWithPid` unintentionally triggered on early-exit, minimizing to a
single real observation) -- none of it fixed the crash, and an isolated
standalone repro of the SAME code in a fresh process passed cleanly,
pointing at something specific to this execution environment (NOT a
process leak -- checked `ps aux`, no leftover Playwright/ms-playwright
chromium processes, healthy free memory) rather than an app or test-logic
bug. Removing the browser dependency entirely made the whole question
moot. `test_pool_assignment.mjs` checks: `poolIndexForParticipant`'s real
source (extracted via regex from timeline-builder.js, same pattern
already used for PROLIFIC_CODES parsing, since timeline-builder.js can't
be imported directly in plain Node -- it pulls in jspsych, which pulls in
a .css import Node's ESM loader can't resolve) for determinism/range/
spread; and a spot-check across the real 200-member pool (indices 0, 50,
100, 199, both tasks) confirming every trial has non-empty `values` and
the correct non-null ground-truth field (`true_mean` continuous /
`true_p` binary). `test_browser.mjs` now has 7 scenarios per browser/task
combo (was 8) -- the removed one is not replaced 1:1 in the matrix, since
its checks moved entirely to the new standalone script. NOT yet re-run
across the full 6-way browser/task matrix since this change -- only
chromium/continuous has been re-verified so far (twice, clean).

**Mini-pool test jzips (for quick manual JATOS/Prolific completion
checks)**: generating a real jzip a human can click through in minutes
(rather than the full 32-trial production experience) needed DATA, not
CODE, changed -- `config.js`'s `import.meta.glob(
'../../sequences_pool/{task}_*_sequences.json')` path is fixed at build
time, so there's no config-level override (and deliberately no dev-only
knob was added, per this file's own "What NOT to do" convention).
Procedure used (fully code-free, safe to repeat): back up the real
`task/sequences_pool/` (gitignored, no other copy exists) -> generate a
small pool into that same path via `generate_sequences_pool.py`
(`--n_pool 10 --n_prefix 2 --n_repeats 2`, i.e. 4 trials/member -- for
binary, `--no_prefix` too, matching the real production branch) -> run
`generate_jzip.py` (the REAL unmodified build/package pipeline) -> rename
the resulting jzips to `-TEST-Ntrial.jzip` so they're never confused with
production -> restore the real pool -> rebuild the real production jzips
again so nothing is left mismatched. Produced
`evidence-integration-{continuous,binary}-TEST-4trial.jzip` this session
(a `-TEST-2trial` pair from an earlier, unrelated session already existed
and was left untouched).

**Real Prolific pilot validated end-to-end against the 4-trial mini pool**
(`dev-results/pilot3testA.txt` / `pilot3testB.txt`, 2 completions + 2
terminations, real Prolific submissions): verified via a temporary Python
script (streaming JSON decoder, same `iter_json_values` pattern as
parse_results.py/reconcile_prolific_jatos.py, since JATOS's raw export
concatenates appends with no separator within a participant's block --
see "Exit/redirect..." above) -- 68/68 checks passed, covering:
  - `poolIndexForParticipant` recomputed directly from real pids matched
    the recorded `pool_index` in every case; the SAME real pid got the
    SAME pool_index in both tasks (cross-task hash consistency, confirmed
    on real data, not just in isolation).
  - Every real observation row's `value`/`qid`/`true_mean`/`true_p`
    matched the (regenerated, same-seed) mini pool file exactly at
    `[trial][observation]` -- zero mismatches.
  - Completed sessions: every one of the 60 (trial, observation) pairs
    (4 trials x 15 obs) had a resolved response, correct trial-summary
    count, `end` screen present. Terminated sessions: exactly one (trial,
    observation) pair timed out 3 times in a row (`trial_timeouts`
    1->2->3), no `finished` marker, no `end` screen.
  - Per-observation `error`/`reward` recomputed from scratch using the
    REAL formula in bonus-continuous.js (running-mean/running-ratio
    reference, confirmed `ERROR_MODE` is still `running_mean`/`running_p`)
    matched exactly; trial-summary `total_error`/`reward` matched the sum
    of that trial's own observation-level values, for both real trials
    and the tutorial's own summary row.
  - `time_elapsed` strictly increasing throughout, tutorial always exactly
    15 observations 0..14 with no gaps, `is_prolific: true` correctly set.
  Bonus earned (at the MAX_REWARD=3 formula active AT THE TIME of this
  test, since lowered to 2 -- see below): binary completion ~$1.16
  (115.67 cents across 4 trials), continuous completion ~$0.17 (16.86
  cents) -- both well under the existing $5 manual-payment ceiling.

**MAX_REWARD lowered from 3 to 2 cents per observation** (bonus-
continuous.js) -- explicit decision to keep total bonus costs down at
production scale (theoretical per-session max drops from ~$14.40 to
~$9.60 at 32 trials x 15 obs, before the $5 ceiling clip). BONUS_DECAY
(15) unchanged. Docstring's parameter-tuning history and theoretical-max
calculation updated to match. Production jzips rebuilt against this
change and the full 200-member pool (fresh UUIDs, per generate_jzip.py's
own always-fresh-UUID design).

**Discrepancy found and RESOLVED: the real production 200-member pool has
std=10, not std=15** -- confirmed directly (`true_std` uniformly `10.0`
across ALL 200 continuous pool members, file timestamps predate this
session), despite BOTH this file's own "Sequence design" section AND
`generate_sequences_pool.py`'s own `--std_fixed` CLI default stating 15.
**Resolved by real-world fact rather than a documentation debate: pilot
#3 launched live on the std=10 pool**, which settles which value is
actually current production -- std=10. Docs updated to match ("Sequence
design" and "Open items" above); `generate_sequences_pool.py`'s own
`--std_fixed` default was deliberately LEFT at 15 (a code change, not
asked for) -- pass `--std_fixed 10` explicitly for any future
regeneration meant to match what's actually live.

### Pilot #3 real-participant incidents -> JATOS reliability investigation -> decision to prototype a Gorilla migration

Two real Prolific participants during pilot #3 hit genuine, distinct
JATOS-level failures (not app bugs) -- both root-caused with hard evidence
(a downloaded JATOS Results Archive's `metadata.json`, and JATOS's own
official docs/forum, including direct answers from JATOS's own maintainer
Kristian Lange):
1. **`dev-results/requested_return.txt` / `returned_results_archive.zip`**:
   a continuous-task participant's session died mid-tutorial (last real
   append: tutorial observation 15/15, never reached a single real trial).
   The archive's `metadata.json` showed `componentState: FAIL` /
   `studyState: FAIL` with JATOS's own message ("It's not allowed to
   reload this component... Study is finished") and a `lastSeenDate` ~31
   min after `endDate` -- confirmed via a JATOS maintainer's own forum
   answer that a reload attempt on a `reloadable: false` component is
   EXACTLY what produces this FAIL state. The participant's own claim of
   having completed all 32 trials does not match this evidence anywhere
   (no `finished` marker, no real trial data, no button text match --
   checked verbatim: the app's real Prolific-facing strings are "Return to
   Prolific to complete your submission" / "Return to Prolific", never
   "Return to the Prolific site").
2. **A second, binary-task participant** hit "It's not allowed to reload
   this component... Study is finished" on their VERY FIRST click, before
   ever starting -- then "Study can be done only once" on retry. Root
   cause (per JATOS's own "Tips & Tricks" doc): `GeneralSingle`'s
   single-use tracking lives in ONE shared browser cookie
   (`JATOS_GENERALSINGLE_UUIDS`), most likely pre-consumed by an
   email/Prolific link-prefetch scanner before the real click. Resolved
   for that participant via an incognito window (bypasses the cookie).

**A direct, empirical simulation confirmed a third, more serious, and
separate risk**: temporarily patched `jatos-shim.js` (`saveData`
deliberately made to reject after the tutorial, then fully reverted --
see this exact edit's diff in chat history if ever needed again) and ran a
full session through Playwright. Result: every single per-trial
`appendResultData` call after the patch point was silently rejected, with
**zero visible symptom** -- the participant clicked through all 15
observations, the trial summary, and the real "Thank you!" end screen
completely normally. Only the very LAST step (clicking the end-screen
button, triggering `finishSession()`) surfaced anything, and even then
only our own on-screen "Something went wrong saving your data" message --
NOT a blank page. This is a REAL, confirmed gap: `on_trial_finish`'s
fire-and-forget `appendResultData` (timeline-builder.js) has no feedback
loop, so a participant can lose an entire session's data with no warning
until (at best) the very last click.

**Investigated `GeneralSingle` vs `GeneralMultiple` vs Personal links** as
a fix -- concluded NEITHER solves this cleanly: `GeneralSingle` (current)
trades a LOUD, manageable failure (what happened above) for eliminating a
WORSE one `GeneralMultiple` has (a reload-triggered FAIL on `GeneralMultiple`
lets a participant silently reopen the link and start an independent
SECOND run -- a silent duplicate, per the SAME JATOS maintainer's own
diagnosis of an earlier incident in this project's history -- see the
`ALLOWED_WORKER_TYPES` comment in generate_jzip.py for that original
decision). Personal Single/Multiple don't fit Prolific's shared-URL flow
without new JIT-link-generation infrastructure. **Conclusion: worker-type
choice can't fix this -- the underlying gap (jsPsych/JATOS have NO
native crash/reload recovery) has to be fixed architecturally, not by
picking a different link type.** Confirmed directly from BOTH jsPsych's
and JATOS's own core maintainers (2020 GitHub discussion,
jspsych/jsPsych#811): *"it's not possible to [recover] if the experiment
crashes due to an error on the page"* -- this is a documented, acknowledged
gap in the underlying tools, not something we're missing via
misconfiguration.

**Proposed redesign** (not yet built): make the experiment-hosting layer a
"dumb pipe" -- a small backend WE control becomes the source of truth for
progress/data (idempotent, keyed on `prolific_pid`, not a browser cookie),
with client-side checkpointing enabling real resumability (reload/crash
recovery instead of restart-from-scratch or a hard block), and the
Prolific completion code delivered independent of any single save
call's success. Grounded in documented practice: jsPsych's own team built
DataPipe (Sasha & de Leeuw et al., *Behavior Research Methods*, 2023)
precisely on "decouple data persistence from the experiment host"; jsPsych's
docs separately recommend blocking-on-save + client retry over
fire-and-forget.

**Alternative-platform research (this session), given building our own
backend is real, error-prone engineering work**: re-evaluated JATOS
against Pavlovia, Cognition.run, and Gorilla specifically on "native
Prolific-ID-keyed resumability, not cookie-based."
- **Pavlovia**: WORSE, not better -- its own FAQ states outright there's
  "no way to recover your data" on a forced/aborted session; defaults to
  batch-at-end saving (not even per-trial appends); costs real money for
  non-"born-open" (i.e. non-public) data; long tail of community-reported
  major data-loss bugs. Ruled out.
- **Cognition.run**: no documented resumability/reconnect feature found;
  appears to carry similar exposure to raw jsPsych's own save-timing
  risks. No clear improvement over JATOS on the axis that matters. Not
  pursued further.
- **Gorilla Experiment Builder**: the strong candidate. Its own docs state
  directly: "Gorilla will always remember where in the tree a participant
  needs to be placed in order to resume the experiment, based on their
  unique Prolific ID" -- i.e. NATIVE, documented, Prolific-ID-keyed
  resumability, plus bidirectional Prolific<->Gorilla status sync
  (participants who return/timeout on either side get rejected on both).
  Also has a dedicated "Code Editor" import path SPECIFICALLY for existing
  jsPsych code (official tutorial, ~10 min for a basic case; described in
  Gorilla's own peer-reviewed paper -- Anwyl-Irvine et al., PMC7005094 --
  as built to host jsPsych tasks with proper participant/data management)
  -- this is a bounded migration (swap the JATOS integration layer for
  Gorilla's API, keep jsPsych plugin/timeline code largely intact), NOT a
  full rewrite in Gorilla's GUI builder.

**Gorilla pricing** (verified against Gorilla's own pages, though two
different figures turned up across pages -- likely old vs. current, NOT
independently confirmed which is authoritative): free to build/test
(Code Editor included in the free standard toolset, no separate
complexity-based charge); pay-per-participant via tokens, 1
token/participant who starts the study; academic rate either **£0.85** or
**£1.09** per token depending on source. At ~200 real participants: roughly
**£170-£220**. Appears to be a one-time per-study-run cost (no evidence
found of ongoing storage/retention fees), but NEITHER the exact current
rate NOR the storage-fee question has been confirmed directly with
Gorilla -- don't treat either as final without checking their live
pricing page or emailing them.

**STATUS: Gorilla NOT pursued further -- superseded by the own-backend
decision below.** The PI reviewed Gorilla and pushed back: "Gorilla
platform is worse. It aims for people with no coding skills and has many
more glitches based on what I have heard from people." This prompted a
deeper comparison rather than an immediate pivot -- see "Own-backend
decision (Supabase)" below for the full investigation (Cognition.run,
Labvanced) and why it ended on building our own small backend instead of
any hosted platform. A separate chat/session was started to prototype a
Gorilla port (in a `task_gorilla/`-style folder) before this pushback
arrived -- **that direction is abandoned; if any `task_gorilla/` artifacts
exist, they're dead and can be deleted, not a parallel track to maintain.**
The existing JATOS-based production pipeline (task/, generate_jzip.py,
the currently-live pilot #3) was never touched by any of this and remains
the live/fallback system throughout.

### Own-backend decision (Supabase) -- chosen over Gorilla/Cognition.run/Labvanced/JATOS

**STATUS UPDATE (this note added once building actually started, updated
as the build progressed): the plan below ("Architecture" and "Next
steps") is now superseded by `task_backend/TODO.md`, which is the live,
actively-maintained build doc -- read that instead of treating what
follows as current. The Supabase backend and the full `task_backend/`
client port are built, deployed, and verified end-to-end against the real
database, including trial-boundary resume, the timeout-retry/`attempt`
path, and all three session-ending screens. The site itself is now live
on GitHub Pages (repo made public for this), verified against the real
deployed URL, not just localhost. The `task/` JATOS pipeline described
everywhere else in this file remains completely untouched and live as
the fallback throughout. Since this note was first written, the tutorial
was also substantially redesigned there (a pilot #3 comprehension finding
motivated a new "Correct answer" panel replacing the old KDE/urn
figures), followed by a dead-code/consolidation pass and an end-to-end
test-suite rewrite that caught and fixed a real bug the consolidation had
introduced (a stale test helper left over from a tutorial-intro click-flow
change, unrelated to app code itself). `task_backend/TODO.md`'s "Status
note for future sessions" (bottom of that file) has the current, exact
list of what's still open -- deliberately NOT copied here, since specifics
there have already gone stale in this exact spot more than once; read
that section directly rather than trusting a summary of it frozen at
whatever point this note was last edited.**

**Why not the hosted alternatives** (full investigation above; summary
for quick reference):
- **Gorilla**: strong native resumability (Prolific-ID-keyed, not cookie-
  based) and a real jsPsych-import path ("Code Editor") -- but the PI
  specifically rejected it as too no-code-focused / anecdotally glitchy
  for this project's needs (see PI quote above).
- **Cognition.run**: investigated as a code-first alternative (addresses
  the "no-code" half of the PI's concern) but found WORSE on the
  reliability axis that actually matters: a detailed instructor's
  integration writeup states it saves data only ONCE AT THE END, not
  trial-by-trial (worse than even our current JATOS setup); a real GitHub
  jsPsych discussion (#1850) documents the EXACT "data not saved before
  Prolific redirect" failure we're trying to escape, on Cognition.run
  specifically; no documented resumability/duplicate-prevention feature
  anywhere; run by a single person as a free/hobby-scale service ("a
  small group of neuroscientists... working from home since 2020") --
  thinner operational backing than either JATOS or Gorilla, undermining
  any assumption that "simpler" implies "more reliable."
- **Labvanced**: mature, well-resourced, peer-reviewed-validated platform
  with a genuinely relevant native "re-identify subjects across sessions"
  setting keyed on Prolific ID (though this reads as aimed at multi-day/
  longitudinal designs, not confirmed for mid-session crash recovery
  specifically) -- but its own marketing is if anything MORE strongly
  no-code-positioned than Gorilla's ("Experiment Creation without
  Coding" is literally the first highlighted feature), and it has NO
  jsPsych-import equivalent to Gorilla's Code Editor -- porting would mean
  rebuilding the whole experiment inside Labvanced's own proprietary
  frame/event-system GUI, a much larger lift than the Gorilla port would
  have been. Rejected on migration-cost grounds, not reliability grounds
  (no direct evidence of major complaints was found for it, unlike the
  other three platforms investigated this session).

**Conclusion**: no existing hosted platform cleanly satisfies both "code-
  first, not aimed at non-coders" AND "native resumability/reliability"
  AND "low migration cost from our existing jsPsych codebase." Building a
  small backend ourselves, while real new engineering work, gives full
  control over exactly the failure modes this whole investigation
  surfaced, using a well-established, heavily-used managed platform
  (Supabase) rather than self-hosting on a personal workstation (a
  meaningfully more fragile option, considered and set aside -- see chat
  history for the university-firewall/tunneling, single-point-of-failure,
  and "no one else's forum to ask" concerns that ruled it out relative to
  a managed backend-as-a-service).

**Architecture** (not yet built -- this is the plan for the next chat
session, see "Next steps" below):
- **Supabase** (managed Postgres + Edge Functions + auto-generated REST),
  chosen over a raw Firebase-style/rules-DSL approach specifically so all
  logic stays in plain JS functions rather than a separate declarative
  rules language, and over self-hosting for the operational reasons above.
- Two tables:
  - `progress` (one row per participant): `prolific_pid` (PK), `task`,
    `pool_index`, `last_trial`, `last_observation`, `status`
    ('in_progress'/'finished'/'terminated'), `updated_at`.
  - `events` (append-only log, mirrors what `jatos.appendResultData` does
    today): `id`, `prolific_pid`, `trial`, `observation`, `screen`,
    `payload` (jsonb), `created_at` -- upserted keyed on
    `(prolific_pid, trial, observation, screen)` so a retried request
    overwrites itself instead of creating a duplicate (the actual
    mechanism behind "idempotent" from chat history).
- Three Edge Functions / request flow:
  1. `/progress-check` (called BEFORE building the jsPsych timeline, on
     load) -- returns `finished` (skip straight to completion code, don't
     rebuild the timeline at all), `in_progress` + a checkpoint (resume
     the timeline from that trial/observation instead of the tutorial),
     or not-found (normal full run).
  2. `/progress-append` (replaces `on_trial_finish`'s `jatos.
     appendResultData` call) -- fire-and-forget is NOT repeated here on
     purpose: track consecutive failures client-side and surface a
     visible (non-blocking) warning after N in a row, rather than JATOS's
     confirmed-this-session silent-failure-through-an-entire-session gap.
  3. `/progress-finish` -- sanity-checks the expected number of trial
     rows actually exist before accepting a "finished" claim (a
     self-built API is exactly as trusting of whatever the browser sends
     as `jatos.appendResultData` was, so this check matters), then the
     participant is shown the completion code as VISIBLE TEXT **and** the
     redirect is attempted -- closing the "NOCODE" gap Prolific's own
     docs describe (a participant whose redirect fails for any reason
     currently has no way to get their code at all; Prolific explicitly
     supports/expects manual code entry as a fallback).
- Row Level Security (RLS) scopes every request to only read/write its
  own `prolific_pid`'s rows -- the browser calls these endpoints directly
  and unsupervised, same trust model as `jatos.appendResultData` today,
  so this isn't optional.
- The existing JATOS-based `task/` pipeline is NOT touched -- this is
  built in a new folder from scratch, preserving the current live/
  fallback system exactly as-is throughout development.

**Pricing/ops decision**: start on Supabase's FREE tier. Verified against
our actual scale (not just generic tier limits): ~540 rows/participant
(32 trials x 15 obs + tutorial/summary/marker rows) x 200 participants x
~300 bytes/row =~ 30-50MB total, comfortably under the 500MB free limit;
~120,000 total Edge Function calls against a 500,000/month free
allowance; bandwidth similarly well under the 5GB/month free limit. The
real risk on free tier isn't capacity -- it's that free projects have
**zero automated backups** (a Pro-plan-only feature) and **auto-pause
after 7 days of no database activity** (recoverable with one dashboard
click while paused, but only within a 90-day window; past that, or if
the project is fully deleted, recovery is not guaranteed -- confirmed
directly via a real Supabase GitHub support thread where a paused
project's resume failed with a server-side "no backups found" error).
**Decision: stay on free tier, but maintain our OWN scheduled backup**
(a weekly database-backup download, either the manual one-click button in
Supabase Studio's Backups section, or a scripted `pg_dump` -- Supabase is
standard Postgres, no proprietary export format) -- this replaces
reliance on Supabase's own pause/restore mechanism, which is usually fine
but has at least one documented real failure case.

**Next steps** (this is the actual TODO list for the next chat session --
nothing below has been built yet):
1. Create the Supabase project; design/create the `progress` and `events`
   tables and RLS policies exactly as specified above.
2. Write the three Edge Functions (`progress-check`, `progress-append`,
   `progress-finish`), including the consecutive-failure-warning logic
   for `progress-append`.
3. Build the client-side resumability logic: on load, call
   `progress-check` before building the jsPsych timeline; if resuming,
   construct a timeline starting from the right trial/observation instead
   of from the tutorial.
4. Swap `on_trial_finish`'s `jatos.appendResultData` call
   (timeline-builder.js) for the new `progress-append` call; swap
   `finish-session.js`'s save-then-end-then-redirect chain for
   `progress-finish` + visible-completion-code-as-text + redirect.
5. Set up the weekly backup process (manual reminder or scripted).
6. Build all of this in a NEW folder (not `task/`) so the existing JATOS
   pipeline stays completely untouched and deployable throughout.
7. Test the exact three failure modes this whole investigation was about
   before considering this done: (a) reload mid-session actually resumes
   correctly instead of restarting or blocking; (b) a simulated backend
   outage during a session surfaces a visible warning instead of silent
   data loss; (c) a completed participant re-visiting the link gets sent
   straight to their completion code, not a duplicate run.

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
  - [DONE, BUT STALE COUNT] node test_browser.mjs: all 6 browser x task
    combinations confirmed 48/48 passing (8/8 each) against the final
    per-participant-pool state, including the pool-assignment scenario.
    **That scenario was REMOVED this chat session** (replaced by the
    much faster task/test_pool_assignment.mjs -- see "This session: E2E
    array-bug fix..." above), so the matrix is now 7 scenarios/combo (42
    total), not 8/48. Only chromium/continuous has been re-verified
    since (7/7, twice) -- the other 5 combos have NOT been re-run.
  - [DONE] Rebuild jzips (python task/generate_jzip.py) -- rebuilt after
    the per-participant pool work landed; confirmed no source/pool files
    postdate the current jzips, and confirmed directly in the built
    bundles (not just source) that pool_index and urlQueryParameters both
    landed correctly.
  - [DONE as of this chat session] **task/evidence-integration-binary.jzip
    rebuild** -- both continuous and binary jzips rebuilt fresh (new
    UUIDs) against the full 200-member pool and the current MAX_REWARD=2
    bonus formula. This specific checkbox has flipped stale/done more
    than once across sessions now -- check actual file timestamps against
    current source/pool before trusting it blindly in a future session.
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

---

## README.md's own task/ section (moved here verbatim)

The above was CLAUDE.md's own "New task (task/)" narrative. README.md had
a SEPARATE, complementary "task/ — Online Experiment" section (design
summary, directory structure, exact commands, deployment checklist,
Prolific rollout plan, data format) that duplicated some of the same
ground with different emphasis (more "how do I actually run this,"
CLAUDE.md's version more "why does it work this way"). Moved here in full
for the same reason -- README.md now points to task_backend as the
current online-task implementation, same as CLAUDE.md does.

## task/ — Online Experiment

Two online experiments deployed on Prolific via MindProbe/JATOS:
- **Continuous task**: Normal(mean, std) stimulus; slider response [0–100]; 8x4=32 trials × 15 obs
- **Binary task**: Bernoulli(p) stimulus (blue/red circle); slider response [0–100%]; 8x4=32 trials × 15 obs

Each participant is assigned ONE of 200 independently-generated sequence sets
per task (a per-participant pool, not one shared file) -- see "Design" below
and "Per-participant sequence pool" above for the full
mechanism.

Both tasks share all infrastructure (jsPsych 8, Vite 6, shared plugins/CSS).
Data pipeline: JATOS JSON → `task/parse_results.py` → `data/task_results.pkl`.
Target: ~50–80 participants per task, within-subject.

### Design

- **Sequences**: hybrid method -- binary via quota/momentmatch construction
  (no seed search), continuous via an unrescaled i.i.d.-suffix construction
  (no seed search either); 8 distinct prefixes × 4 repeats = 32 trials;
  prefix_length=4; **std=10 (continuous)** -- confirmed as the actual
  current production value (`true_std=10.0` uniformly across all 200
  continuous pool members; pilot #3 is live on this pool). An earlier
  version of this doc, and `generate_sequences_pool.py`'s own
  `--std_fixed` CLI default, both say 15 -- that's the value the script
  defaults to, not what's actually deployed; pass `--std_fixed 10`
  explicitly for any future regeneration meant to match current
  production. Each participant gets ONE of 200 independently-generated
  sequence sets per task, assigned via a
  deterministic hash of their participant ID (same index for both tasks) --
  not one shared file. See "Per-participant sequence pool" above and
  "Sequence generation methods (task/)" above for the full mechanism and rationale.
- **ITI**: all trials use 1000ms (ITI manipulation removed — no effect observed in pilot)
- **BTI**: 3s between-trial reset screen ("Trial X / N — generating new sequence…") —
  same wording for both tasks (continuous used to say "generating new distribution…";
  "sequence" is the more defensible word since the underlying hidden parameter can
  repeat across trials — 6 unique parameter sets × 4 repeats — so "new
  distribution" would sometimes overclaim novelty)
- **Distractor**: `iti_condition` per trial ('control'/'distract', 2-of-4 per qid);
  `DISTRACTOR_TYPE` in config: 'none' | 'iti_length' | 'popup' (default: 'none')
- **Timeout**: 7s response deadline per observation
  - Per-trial timeout budget: 3 timeouts before session terminates
  - Timeout flow: "Too slow" screen (fade in/out/in, 3.2s) → replay ITI → same observation
  - On 3rd timeout: "Too slow / 0 timeouts remaining" → "Session terminated" screen with button
- **Welcome screen**: title/branding page shown first, before consent — "Evidence
  Integration", "Numbers" (continuous) or "Colors" (binary), inside a bordered box,
  matching the study names given on Prolific; "Begin" button leads to consent.
- **Tutorial**: full N_OBS_TO_RUN-length tutorial (15 observations, matching
  real-task length exactly), driven by a three-phase top-right hint system
  keyed on observation number (default text → yellow goal reminder, obs
  6-10 → red "use your memory, these graphics won't repeat" warning with
  the figure/tracker hidden behind an overlay, obs 11-15), plus a 15-slot
  history tracker between the figure and the hint box (numbers on
  underlines for continuous; colored dots for binary, since there's no
  separate "number" for a blue/red draw — the color IS the content). Intro
  screen: box 1 (text) → image box (separate click-to-reveal step, showing
  a bubbling generative animation) → box 2 (goal text) → box 3 (slider
  instructions) → slider. The main obs circle/number, and the tutorial's
  own observation marker, fade in (1000ms) rather than appearing instantly,
  for a consistent feel between tutorial and real trials. Tutorial's
  illustrative sequence is derived from a real trial in the sequences data
  (config-base.js's pickTutorialExample), not hand-picked, so it can't
  drift out of sync with the actual generation parameters. See
  "Tutorial redesign, bonus/error system, and binary no-prefix sequences"
  above
  section for the full phase mechanism and rationale.
- **Bonus payments**: per-observation error (measured against either the
  trial's fixed true mean/probability or a per-observation running mean/
  ratio of the raw observed values, via config-base.js's `ERROR_MODE`) is
  converted to a per-observation reward via `bonus-continuous.js`'s
  `normError = rawError / MAX_POSSIBLE_ERROR; reward = max(0, MAX_REWARD *
  (1 - BONUS_DECAY * normError))` (current parameters: `MAX_REWARD = 2`
  cents, `BONUS_DECAY = 15`, `MAX_POSSIBLE_ERROR = 100`), then summed
  across observations for the trial/tutorial total. Shown on both summary
  screens as a "Total error / Bonus" box above the chart, alongside a
  per-row reference tick + error-distance line on the chart itself. Real
  payment is given manually and clipped to a $5 ceiling regardless of the
  formula's raw sum. See "Tutorial redesign, bonus/error system, and
  binary no-prefix sequences" above for the full mechanism, the formula's
  own tuning history (this specific MAX_REWARD value was most recently
  lowered from 3 to 2 to keep total bonus costs down), and the
  methodological rationale for `ERROR_MODE` (running-mean/ratio tracking
  vs. true-parameter inference are genuinely different cognitive tasks,
  not just different formulas).
- **Summary slides**: binary — per-obs bar chart (gray background, black
  circle at estimate, green reference tick, violet error-distance line, obs
  circle left, still colored by that draw's own value); continuous —
  per-obs number line (red obs thumb, black circle at estimate, blue
  reference tick, green error-distance line)
- **Consent form**: verbatim IRB-approved text from task/consent_form.txt, followed by
  3 boxes with ordered disclosure (each stays locked until the one before it is
  revealed): a blue payment-motivation box first ("You will be paid $5.00 for
  finishing and up to $5.00 based on your performance"), then 2 red warning boxes
  (data-loss / response-deadline — the data-loss box is now just "Do not close,
  refresh, or navigate away during the task."). The proceed button doesn't use the native `disabled`
  attribute (disabled buttons never dispatch `click`, so a premature click got
  silently swallowed with zero feedback) — a capturing-phase click listener
  gates it instead.

### Sequence generation

Three scripts, each with a ROLE note at the top of its own module docstring
(see "Sequence generation methods (task/)" above for the full
rationale and tradeoffs):

```bash
# Pure i.i.d. (no smoothing, no seed search — single draw) -- one of two
# candidates still under PI consideration for the 10x4 design, not current
# production:
python task/generate_sequences_iid.py --task both --seed 0 \
    --n_unique_sequences 10 --n_repeats 4 --mean_range 20 80 --std_fixed 15 --p_range 0.2 0.8

# Moment-matched / quota (isotonic-residual seed search, default score_mode) --
# this is what generated the CURRENT PRODUCTION 6x4 pilot. Prefix identity
# and target level are independent axes (see "Sequence generation
# methods (task/)" above for the full mechanism and the collision bug this fixes):
python task/generate_sequences_momentmatch.py --task both --n_tries 1000 \
    --n_prefix 6 --n_repeats 4 --rl_alpha_0 1.0 --rl_lambda 0.5
# Defaults used above (all overridable): --n_prefix 6, --mean_range 15 85
# (continuous), --blue_range 2 13 (binary, blue-ball count out of
# --seq_length -- NOT a p fraction).

# generate_sequences.py has no CLI of its own anymore -- shared-utilities
# module only, imported by both scripts above, not run directly.
```

**Current production** (8x4 hybrid, per-participant pool): binary uses
unchanged quota/momentmatch construction (prefix/target independence,
optimal matching, exact-quota suffix) but with NO seed search; continuous
uses the same prefix/target construction but a genuinely unrescaled
i.i.d.-suffix (also no seed search) -- see "Sequence generation
methods (task/)" and "Per-participant sequence pool" above for the full
rationale and the empirical findings behind this per-task split. 8 distinct
prefixes, 4 repeats each = 32 trials PER POOL MEMBER; 200 independent pool
members per task (task/generate_sequences_pool.py), each participant
assigned one via a deterministic hash of their ID. Continuous mean_range=
[15,85]; binary blue_range=[2,13] out of 15. Verified across the whole
pool (not just one member): 200/200 members pass prefix uniqueness both
tasks, zero binary quota mismatches. scripts/inspect_iid_sequences.py
--sequence_type pool and scripts/inspect_sequences.py --pool_dir both
supported inspecting the real pool directly (both now archived --
see archive/archive_readme.md; the current equivalent is
scripts/plot_sequences.py, described in CLAUDE.md's own
"Sequence-generation diagnostics" section).

**Binary switched to a NO-PREFIX branch this session** (`--no_prefix` flag
in generate_sequences_hybrid.py / generate_sequences_pool.py, binary only):
a real diversity bug was found in the old prefix scheme -- its composition
allocator gave EVERY pool member the exact same blue-count split across
its 8 prefixes (deterministic, no RNG), which collapsed the `|Δresponse|`
curve's between-participant diversity on the prefix portion (obs ≤ 4) down
to as few as 3 distinct values. The no-prefix branch removes the prefix/
qid-repeat concept for binary entirely -- every trial gets its own
independent `true_p` and its own independent exact-quota full-length
sequence -- confirmed to fix the diversity collapse (smooth 17→23→27→33
growth across obs 2-5, no collapse anywhere). **Production's binary pool
now uses this branch**; the old prefix-based binary pool is backed up at
`task/sequences_pool_binary_prefix_backup/` (gitignored, fully recoverable).
Continuous's pool is unaffected -- still the prefix-based hybrid method
above. This trades away a clean, controlled repeated-stimulus design (used
for response-variability/reliability metrics) for better diversity; a
per-pid analysis found substantial NATURAL but uncontrolled repetition
still exists at the 4-observation level (see this document's own
build history above for the numbers) --
whether that's sufficient is still an open question.

The single reference file (task/sequences/{continuous,binary}_sequences.
{pkl,json}) remains the promotion/verification target when changing
generation parameters, but is NOT what real participants are served --
see "Per-participant sequence pool" above.

**10x4 full experiment**: NOT yet finalized. The previously-found candidate
seeds (moment-matched, isotonic score_mode, mean_range=[20,80],
p_range=[0.2,0.8], std_fixed=15: continuous seed=245, binary seed=68) now
predate BOTH the evenly-spaced/no-mirroring redesign AND the prefix/target-
independence redesign above -- they'd need regenerating from scratch under
the current script to be current, not just re-checked. Choice between the
i.i.d. and moment-matched branches, AND what --n_prefix/range to use at
this larger scale, are both pending PI consultation (moment-matching
introduces a real, literature-documented behavioral tradeoff — see
this file's own "Sequence design: open questions" section above — it is
not a free smoothness win).

**See "Sequence design: open questions" above** for the full write-up of
this tradeoff: a real, quantified behavioral confound from quota
construction (not simple gambler's-fallacy reasoning — the actual
mechanism is subtler and confirmed with real simulation numbers), a
three-way design trilemma between statistical cleanliness, target
diversity, and the repeat-based reliability metrics (T5/T6/V-group) this
project's scientific goals depend on, and what it would actually take to
serve genuinely unique sequences to every participant if that path is
ever chosen. Nothing there blocks the current 6x4 pilot or the promoted
10x4 production sequences from shipping — it's there so the trade-off is
fully informed rather than re-discovered from scratch later.

### Directory structure

```
task/
  src/
    shared/
      timeline-builder.js          — orchestrator: builds full timeline from config
      build-trial-timeline.js      — pure-JS trial loop (importable by test harness)
      build-tutorial-timeline.js   — tutorial sub-timeline (intro → obs → summary → timeout demo)
      build-welcome-screen.js      — title/branding screen shown first, before consent
      build-consent-screen.js      — informed-consent screen
      build-end-screen.js          — final "Thank you" screen
      create-early-exit.js         — session-terminated flow (3-timeout exhaustion)
      finish-session.js             — shared "how does a session actually end"
                                      implementation (single call, data passed
                                      as the argument; Prolific redirects
                                      externally, everyone else gets a DOM
                                      update + no redirect at all -- see
                                      "Exit/redirect and data-saving
                                      architecture" above for why a
                                      same-origin redirect after ending the
                                      session failed on a real MindProbe pilot)
      config-base.js               — shared config factory for continuous/binary configs
      jatos-shim.js                — dev no-op shim
      plugin-inter-trial.js        — BTI reset screen
      plugin-observation-continuous.js — continuous obs (number + slider + timeout clock)
      plugin-observation-binary.js     — binary obs (circle + gradient slider + timeout clock)
      observation-timeout-clock.js — shared countdown-clock renderer (used by both
                                      observation plugins AND plugin-timeout-demo.js)
      plugin-iti-clock.js          — ITI countdown clock; shows "Too slow" if timed_out=true
      plugin-timeout-demo.js       — 3-screen timeout explanation (tutorial)
      plugin-tutorial-intro-continuous.js       — continuous 3-stage interactive intro
      plugin-tutorial-intro-binary.js           — binary 3-stage interactive intro
      plugin-tutorial-observation-continuous.js — continuous tutorial obs 2–5 (no timeout clock)
      plugin-tutorial-observation-binary.js     — binary tutorial obs 2–5 (no timeout clock)
      plugin-tutorial-summary-continuous.js     — continuous tutorial summary
      plugin-tutorial-summary-binary.js         — binary tutorial summary
      plugin-trial-summary-continuous.js        — continuous trial summary
      plugin-trial-summary-binary.js            — binary trial summary
      distribution-continuous.js   — shared continuous distribution SVG (revealed flag)
      draw-performance-continuous.js — SVG distribution + dot plots (continuous)
      draw-performance-binary.js     — SVG bar + dot plots (binary)
      urn-binary.js                 — shared binary urn SVG (revealed flag)
      continuous-draw-animation.js  — tutorial bubbling animation (continuous):
                                      bubbles fall from under the curve to the x-axis
      binary-draw-animation.js      — tutorial bubbling animation (binary):
                                      bubbles rise inside the blue/red bar
      slider-continuous.js         — continuous slider
      slider-binary.js             — binary slider (blue/red gradient after first
                                      interaction; 2-row axis ruler below)
      bonus-continuous.js          — per-observation error / per-trial bonus formula,
                                      shared by both tasks despite the name (not
                                      renamed to avoid a wide import-path change) --
                                      see "Tutorial redesign, bonus/error
                                      system..." above
      tutorial-tracker.js          — 15-slot tutorial history tracker (numbers on
                                      underlines for continuous; colored dots for binary)
      style.css                    — all shared styles
    continuous/
      config.js
    binary/
      config.js
    experiment-continuous.js
    experiment-binary.js
    test-harness.js         — test-ONLY entry (index-test.html); never bundled into
                              any production build, never linked from production code
  sequences/
    continuous_sequences.{pkl,json}   — single reference/promotion-target copy
                                        (NOT what real participants are served
                                        -- see sequences_pool/ below)
    binary_sequences.{pkl,json}       — same, binary side
    continuous_momentmatch_sequences.{pkl,json} — currently an IDENTICAL COPY of
                                        the production file above (this is the
                                        generation script's own output location,
                                        always overwritten by whatever was most
                                        recently searched here -- the OLD 10x4
                                        candidate that used to live at this same
                                        path (seed=245, mean_range=[20,80]) was
                                        overwritten by the 6x4 pilot search and
                                        is no longer in the working tree, but IS
                                        still recoverable via git (commit 274b598
                                        -- confirmed by directly checking out and
                                        inspecting that version: 10 qids, 40
                                        trials, means spanning ~22-78)
    binary_momentmatch_sequences.{pkl,json}     — same situation, binary side
                                        (old 10x4 candidate seed=68 likewise
                                        overwritten in the working tree, same
                                        git commit for recovery)
    continuous_iid_sequences.{pkl,json}         — 10x4 candidate (i.i.d. branch)
    binary_iid_sequences.{pkl,json}             — 10x4 candidate (i.i.d. branch)
  sequences_pool/                     — the REAL per-participant pool served to
                                        real participants -- 200 members/task,
                                        {task}_{0000..0199}_sequences.{pkl,json},
                                        generated by generate_sequences_pool.py.
                                        Gitignored (800 small files, fully
                                        reproducible via that script's own CLI
                                        defaults) -- see "Per-participant
                                        sequence pool" above.
  generate_sequences.py             — original: k-constrained rejection sampling
  generate_sequences_iid.py         — pure i.i.d., no smoothing, no seed search
  generate_sequences_momentmatch.py — quota/moment-matching, isotonic seed search
  generate_sequences_hybrid.py      — CURRENT PRODUCTION method (per-task split);
                                        also has generate_binary_sequences_no_prefix,
                                        a SEPARATE binary-only branch (--no_prefix)
                                        now used in production -- see "Sequence
                                        generation" above and "Sequence generation
                                        methods (task/)" above
  generate_sequences_pool.py        — wraps generate_sequences_hybrid.py, writes
                                        the 200-member pool above
  parse_results.py
  test_browser.mjs         — Playwright E2E tests (Chromium/Firefox/WebKit, both tasks)
  test_pool_assignment.mjs — plain Node, no Playwright/Vite; checks the
                              per-participant pool-hashing mechanism and
                              real pool data integrity in well under a
                              second (see "Testing" above)
  index-continuous.html
  index-binary.html
  index-test.html           — test-ONLY entry point, drives test-harness.js; not
                              a build input, real participants can't reach it
  package.json
  vite.config.js
```

Naming convention: every file/class with a continuous/binary pair uses an
explicit `-continuous`/`-binary` suffix on both sides — never leave one side
as an implicit unsuffixed default.

### Key parameters

```js
// src/shared/config-base.js DEFAULTS (shared by both task configs)
const N_OBS_TO_RUN           = 15;
const SHOW_SLIDER_VALUE      = true;
const SLIDER_DEFAULT         = 'last';  // thumb starts at previous response;
                                         // numeric label stays hidden until
                                         // first interaction)
const DEFAULT_VALUE          = 50;
const BTI_MS                 = 3000;
const ITI_SHORT_MS           = 1000;   // tutorial between-observation ITI
const T_OBS_MS                = 7000;
const SHOW_TRIAL_PERFORMANCE = true;
const DISTRACTOR_TYPE        = 'none';
const ERROR_MODE             = 'running_mean';  // continuous default;
                                                 // binary overrides to
                                                 // 'running_p' -- see "Sequence design: open questions" above (Section 5)
const MAX_TIMEOUTS_PER_TRIAL = 3;      // defined in timeline-builder.js
```

Prolific completion/early-exit codes (4 total: {continuous,binary} ×
{completion,earlyExit}) live in timeline-builder.js's `PROLIFIC_CODES`, not
config-base.js — see "Deploying to MindProbe" below.

Trial count is NOT a config constant anymore — it's fully implicit from
however many trials task/sequences/{task}_sequences.json contains.

### Commands

```bash
cd task
npm install                    # first time only
npm run dev:continuous         # local dev server on :5173, opens index-continuous.html
npm run dev:binary             # local dev server on :5174, opens index-binary.html
                                # (both can run simultaneously, in two terminals)
npm run build:continuous       # production build → dist-continuous/
npm run build:binary           # production build → dist-binary/
node test_browser.mjs          # Playwright E2E tests (~2-3 min); see Testing below
```

### Testing

**`test_browser.mjs`** — spawns the real Vite dev server (not a patched build) and
drives a test-ONLY entry point (`index-test.html` / `src/test-harness.js`) via
Playwright across Chromium, Firefox, and WebKit, for both tasks (7 scenarios
per browser/task combination as of the latest chat session -- was 8, a
"Pool assignment" scenario was removed and replaced with a much faster,
browser-free `test_pool_assignment.mjs`, see below): normal submit, timeout
replay, "N timeouts remaining", 3-timeout termination screen,
submit-then-continue, completes-all-trials, and Prolific-redirect — tutorial
included in full for every scenario. That harness is never linked from
production code and never
included in any build (vite.config.js's build inputs are only
index-continuous.html/index-binary.html) — real participants can never reach
it. It calls the exact same `buildAndRun()` production uses; it only adjusts
plain config fields (trial count via array slicing, tObsMs/btiMs/itiMs via
direct assignment) before handing the config to the same production code
path — no override logic lives inside buildAndRun/timeline-builder.js itself.
URL params: `?task=&trials=&tObsMs=&btiMs=&itiMs=`. Screen transitions are
detected via `body[data-screen="..."]` rather than guessed sleep durations.

Takes ~2–3 minutes for the full matrix (longer than before the tutorial was
included) — run it after big `task/` changes or when asked, not after every
small edit.

```bash
node test_browser.mjs                                   # full matrix
node test_browser.mjs --task=binary --browser=chromium  # a subset
```

Firefox/WebKit need their browser binaries installed once:
`npx playwright install firefox webkit` (plus `sudo npx playwright install-deps`
for system libraries if missing).

**`task/test_pool_assignment.mjs`** — plain Node script, no Playwright, no
Vite dev server, runs in well under a second. Checks the per-participant
sequence-pool mechanism (see "Per-participant sequence pool" above):
`poolIndexForParticipant`'s determinism/range/spread (extracted from
timeline-builder.js's real source via regex, not a hand-copied duplicate),
and a spot-check across the real 200-member pool (both tasks, several
indices) confirming every trial has non-empty values and the correct
non-null ground-truth field. Replaces a much heavier, Playwright-based
"Pool assignment" scenario that used to run three full tutorial+session
flows through real Chromium just to check logic that turned out to have
zero DOM/browser dependency at all — see "This session: E2E array-bug
fix, pool-assignment scenario removed, mini-pool test jzips, bonus/std
findings" above for the full story,
including an unresolved "Target page ... has been closed" instability that
motivated removing the browser dependency entirely rather than continuing
to chase it.

```bash
node task/test_pool_assignment.mjs
```

### Local testing pipeline

```bash
# Terminal 1: local result server
npm run dev:server

# Terminal 2: task (either task works the same way)
npm run dev:continuous
# Complete a task in browser → data saved to task/dev-results/

# Parse results
python task/parse_results.py --input_dir task/dev-results/ \
    --output data/task_results.pkl --verbose
```

### Deploying to MindProbe

**Pre-deployment checklist:**
- [DONE] Sequence pool generated and verified (200/task; continuous:
  prefix-based hybrid method; binary: switched to the no-prefix branch
  this session -- see "Sequence generation" above)
- [DONE] PILOT ONLY name field removed
- [DONE] All 4 Prolific code placeholders filled in timeline-builder.js's `PROLIFIC_CODES`
- [DONE] Prolific wallet funded; payment rate confirmed ($10 completion, $3 early-exit)
- [DONE as of the latest chat session] Both production jzips rebuilt fresh
  (new UUIDs) against the full 200-member pool and the current bonus
  formula (MAX_REWARD lowered from 3 to 2 cents/observation). This item
  has flipped stale/done more than once across sessions -- verify actual
  file timestamps against current source/pool rather than trusting this
  checkbox blindly.
- [DONE] Full 6-way browser/task E2E matrix -- 48/48 passing (8/8 each)
  as of an earlier session; **STALE COUNT as of the latest chat session**
  -- the "Pool assignment" scenario was removed (replaced by the much
  faster, browser-free `task/test_pool_assignment.mjs` -- see "Testing"
  above), so the matrix is now 7 scenarios/combo (42 total). Only
  chromium/continuous has been re-verified since (7/7, twice); the other
  5 combos have not been re-run.
- [DONE] A 4-trial mini-pool test build (generated by temporarily
  swapping `sequences_pool/` for a small 10-member/4-trial pool, building
  jzips, then restoring production -- see "This session: E2E array-bug
  fix, pool-assignment scenario removed, mini-pool test jzips, bonus/std
  findings" above for the exact procedure) was validated end-to-end via 2 real
  Prolific completions + 2 terminations (`dev-results/pilot3testA.txt` /
  `pilot3testB.txt`). 68/68 automated checks passed: pool-hash
  determinism/uniqueness, exact alignment with the served mini pool,
  complete response coverage, correct termination behavior, and bonus
  formula correctness (recomputed from scratch against the real formula).
- [PENDING] A genuinely full completion run via real Prolific preview (not
  just early-exit) against the FULL 32-trial production pool -- the mini-
  pool test above used a 4-trial pool, not full production content.
- [DONE] Incremental per-trial saving, save-then-end-then-redirect gating,
  and the GeneralSingle-only worker-type switch, all confirmed against real
  MindProbe/JATOS via six manual test scenarios this session (hand-edited
  `?PROLIFIC_PID=` params, no real Prolific involved) -- see
  "Exit/redirect and data-saving architecture" above, its "REAL-TEST
  FINDINGS" note, for what was confirmed and two corrections
  (`jatos.log` isn't visible anywhere in the JATOS UI; GeneralSingle's block
  is keyed on the browser's cookie, not the `PROLIFIC_PID` value).
- [RESOLVED] The real production pool's continuous std is 10, not the
  script-default 15 (confirmed directly, all 200 members) -- **pilot #3
  is now live on the std=10 pool, which settles it: std=10 is current
  production.** Docs updated to match. `generate_sequences_pool.py`'s
  own `--std_fixed` default is still 15 -- pass `--std_fixed 10`
  explicitly for any future regeneration meant to match production.

```bash
npm run build:continuous && npm run build:binary
python task/generate_jzip.py   # generates evidence-integration-{task}.jzip
```

Import each `.jzip` into MindProbe: Studies → **+** → **Import Study**. The
batch now only accepts **GeneralSingle** workers (previously all five JATOS
worker types) -- grab the General Single link from MindProbe's Worker &
Batch Manager and use that as the Prolific Study URL (same
`?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID=...&SESSION_ID=...` suffix as
before).

### Future architecture: moving off JATOS's participant-tracking layer

**Built, tested, and live** -- see "Pilot #3 real-
participant incidents..." and "Own-backend decision (Supabase)" above,
and `task_backend/TODO.md` (the actively-maintained build doc, supersedes
this document's own now-superseded "Architecture"/"Next steps" plan
for this), for the full
investigation and current status; this is a pointer, not the detail. Short version: two real pilot #3 participants
hit genuine JATOS-level failures (a reload-triggered session loss, and a
GeneralSingle cookie/link-prefetch collision blocking a legitimate first
attempt), plus a confirmed, empirically-tested gap where per-trial saves
can fail silently for an entire session with zero participant-visible
symptom. Gorilla, Cognition.run, and Labvanced were all evaluated as
hosted alternatives and rejected (see "Own-backend decision (Supabase)
-- chosen over Gorilla/Cognition.run/Labvanced/JATOS" above for why each
one specifically fell short). **A small backend on Supabase** (progress-
check/append/finish, keyed on `prolific_pid` rather than a browser
cookie, enabling real reload/resume instead of restart-or-block) is built
in `task_backend/` alongside -- **not replacing** -- the JATOS pipeline
described above, which remains the live/fallback system throughout.

`task_backend/` is a from-scratch port, not a copy: the two tasks are
called **numbers**/**colors** there (not `continuous`/`binary` -- matches
the participant-facing labels; `task/` keeps its own original naming,
the two codebases don't share terminology), the old JATOS-era blanket
per-screen save hook was dropped entirely in favor of checkpointing only
the 3-4 phases that actually matter for resume, and sequence generation
was consolidated from five scripts down to one
(`task_backend/generate_sequences.py`) with the dead/unused methods
removed. Verified end-to-end against the real deployed backend AND the
real deployed site (not mocked, not just localhost): trial-boundary
resume across a real reload, the timeout-retry/`attempt` mechanism, and
all three session-ending screens showing a visible, redirect-independent
completion code -- confirmed both on the local dev server and on the
live GitHub Pages URL. Still open: a persistent automated test suite
(everything verified so far was one-off scripts) and a weekly database
backup process -- see `task_backend/TODO.md`'s "Status note for future
sessions" for the current, exact list.

### Prolific rollout plan

- Publish both studies **simultaneously**; no inter-study screening filter
- **Free task ordering** — natural counterbalancing via Prolific dashboard
- Set both to **MANUAL approve** (changed this session from auto-approve) --
  every submission is reviewed before payment, which is also what makes the
  "NO JATOS DATA AT ALL" / "STUCK" cases from `reconcile_prolific_jatos.py`
  (see "Data format" below) actionable rather than already-paid by the time
  they're noticed
- End screen: *"one half of a two-part study — look for the other on your dashboard"*
- **Payment:** $10 for normal completion, $3 for the screen-out/early-exit path
- **Non-completions:** request return (not rejection) — slot reopens; Prolific
  also supports rejecting for "gave no study data" specifically, if caught
  before its 21-day auto-approval window
- **Partial compensation:** participants who reach 3 timeouts in one trial receive
  partial payment via the per-task `earlyExit` code in `PROLIFIC_CODES`
  (timeline-builder.js) -- appears in JATOS results as `progress: 'terminated'`
- **Academic discount:** use Dartmouth institutional email for 33.3% platform fee discount
- **Device restriction:** desktop-only in Prolific study settings

### Data format

`parse_results.py` filters to `screen='observation'` rows and extracts every
column DIRECTLY from the raw export -- no lookup/join against a sequence
file (see "Sequences.json schema, tutorial derivation, and
participant-data columns" above, its "Participant-data columns"
subsection, for why this
changed: a join against a single shared file stopped being valid once
different participants can be assigned different pool members):

| Column | Type | Description |
|--------|------|-------------|
| `prolific_pid` | str | Prolific participant ID |
| `task` | str | `'continuous'` or `'binary'` |
| `pool_index` | int | Which of the 200 pool members this participant was assigned |
| `trial` | int | 0-indexed trial number |
| `observation` | int | 0-indexed observation within trial |
| `qid` | int | Which of that pool member's distinct prefixes this trial used |
| `value` | int | Stimulus value |
| `true_mean` | float | Continuous only; NaN for binary rows |
| `true_std` | float | Continuous only; NaN for binary rows |
| `true_p` | float | Binary only; NaN for continuous rows |
| `response` | float | Participant estimate (NaN if timed out) |
| `timed_out` | bool | True if response deadline elapsed |
| `rt` | float | Response time in ms (NaN if timed out) |
| `time_elapsed` | int | ms since experiment start |

Older (pre-pool) export files remain parseable -- `pool_index` comes back
missing/NaN with a printed warning rather than crashing.

Both tasks share the same output file — use `df[df.task == 'continuous']` to split.

**Every row now also carries a `progress` field** (e.g. `"welcome"`,
`"tutorial 2/4"`, `"trial 7/24"`, `"finished"`, `"terminated"`) and has
`stimulus`/`button_html` (rendered HTML/CSS) stripped -- see
"Exit/redirect and data-saving architecture" above, its "CURRENT
ARCHITECTURE" note. `parse_results.py` itself is unaffected (still
filters to `screen='observation'`), but the raw JATOS export is now
scannable by eye without downloading anything.

**Reconciling Prolific vs. JATOS**: `task/reconcile_prolific_jatos.py`
cross-references a Prolific submissions-export CSV against a JATOS results
export and flags every participant as OK / TERMINATED / STUCK / NO JATOS
DATA AT ALL / pilot-ignore, for manual review under the manual-approve
workflow above:

```bash
python task/reconcile_prolific_jatos.py \
    --jatos_dir <path_to_jatos_export> \
    --prolific_csv <path_to_prolific_export.csv> \
    --output reconciliation_report.csv
```

Verified against synthetic fixtures only so far -- not yet run against a
real Prolific export (its column-name detection is a best guess at
Prolific's current CSV headers; override with `--prolific-id-col` etc. if
it picks the wrong one).

---
---

# task_backend — build history and settled decisions (folded in from task_backend/TODO.md)

**Everything below this point was task_backend/TODO.md in full**, folded
into this file once task_backend's initial build-out genuinely settled
(schema, Edge Functions, distractor system, bonus formula, tutorial
redesign, and the Prolific cutover are all done -- see CLAUDE.md's
"Online task: task_backend" section for current state). Unlike task/
above, task_backend is NOT retired -- it's the live, active system, still
being tuned against real pilot data. What moved here is the SETTLED
narrative (decisions made, bugs found and fixed, rationale) accumulated
during its build-out and first two real pilot rounds; CLAUDE.md continues
to hold only current-state facts about it, same relationship it already
has with task/'s own history above. Preserved verbatim, including its own
internal "this session" framing (now historical, not current) -- not
retroactively edited, per this file's own stated convention above.

# task_backend/TODO.md — Supabase backend build plan

**Read this in full before writing any code in task_backend/.** This is the
working plan for the Supabase-backed backend described in CLAUDE.md's
"Pilot #3 real-participant incidents..." and "Own-backend decision
(Supabase)" sections — read those first for the full incident history and
why this exists at all; this document is where the *design* was actually
settled (several details here supersede/refine what's sketched in
CLAUDE.md's "Architecture" and "Next steps" subsections, per a planning
conversation — CLAUDE.md itself should get a pointer to this file once
building starts).

The existing JATOS-based `task/` pipeline is NOT touched by any of this and
remains the live/fallback system throughout. Everything here is built fresh
in `task_backend/`.

---

## Why (one paragraph — see CLAUDE.md for full evidence)

Two real Prolific participants hit genuine JATOS-level failures during pilot
#3 (a reload-triggered session death mid-tutorial; a GeneralSingle
cookie/link-prefetch collision blocking a legitimate first attempt). A
follow-up empirical test confirmed a third, worse gap: per-trial saves can
fail silently for an entire session with zero participant-visible symptom.
Gorilla/Cognition.run/Labvanced were evaluated as hosted fixes and rejected
(see CLAUDE.md). Decision: a small backend on Supabase, keyed on
`prolific_pid` rather than a browser cookie, giving real reload/resume
instead of restart-or-block.

---

## Decisions settled (planning conversation, before any code)

These resolve the open flags raised when the plan in CLAUDE.md was first
reviewed. Treat these as settled unless something concrete surfaces during
implementation that contradicts them — don't silently re-open them.

1. **Access model: locked down, Edge-Functions-only.** There is no login
   system, so `prolific_pid` is never a verified identity claim — per-row
   RLS keyed on it would be security theater (anyone with the anon key
   could claim any `prolific_pid`). Instead: RLS is **enabled** on every
   table with **zero policies** for the `anon`/`authenticated` roles (deny
   all direct access, including read). The three Edge Functions are the
   *only* way in or out, using the service-role key server-side (which
   bypasses RLS by design). The browser never talks to the database
   directly, only to the Edge Function endpoints.

2. **Single table, not two.** Originally planned as separate `progress` +
   `events` tables. Collapsed into one `events` table: "current progress"
   is just the latest row per `(prolific_pid, task)`, queried directly —
   nothing to keep in sync, no dual-write transaction to get right, no way
   for the two to disagree. `progress-check` does one query
   (`ORDER BY id DESC LIMIT 1`), not a join.

3. **Resume granularity: trial-boundary, not exact-observation.** On
   reload, resume at the start of the current incomplete trial (or the
   tutorial, if still there) — not the exact observation. A trial/tutorial
   is only 15 observations, cheap to redo; this avoids reconstructing
   mid-trial UI state (slider position, in-flight animation) from a DB row,
   which is a lot of fragile surface area for very little participant time
   saved. Every observation is still logged individually for analysis —
   only the *resume point* is coarser than the *log*.

4. **`phase` covers welcome/consent too, for bookkeeping.** Originally
   planned to only log tutorial/trial/terminal events (welcome/consent
   didn't need to be *resumable*). Decided to log them anyway — cheap, and
   gives direct evidence of "did they open the link" / "did they actually
   consent" rather than inferring it from later rows existing.

5. **Timeout retries get their own attempt counter, not silent overwrite.**
   A genuine network-retry of one unconfirmed request should overwrite
   itself (that's the whole point of idempotent upsert). A timeout-
   triggered *replay* of the same observation is a different event and
   must not clobber the first attempt's row. Both are handled by an
   `attempt` column in the uniqueness key (see schema).

6. **Row shape mirrors the current JATOS lean-row design, not a JSON blob.**
   Stimulus/response fields are real flattened columns, not a `payload
   jsonb` bag — `SELECT * FROM events WHERE phase='trial'` should already
   look like a usable dataframe, close to a straight rewrite of
   `parse_results.py` rather than "unpack JSON, then reshape." Downstream
   analysis/plotting scripts get rewritten later against this shape — not
   a blocker for backend construction, just noted so the schema is chosen
   with that rewrite in mind.

---

## Schema

### `events`

One append-only table. Every row is either a checkpoint (tutorial/trial
observation) or a bookkeeping/terminal marker (welcome/consent/finished/
terminated).

| Column | Type | Notes |
|---|---|---|
| `id` | `bigserial primary key` | Use `ORDER BY id DESC LIMIT 1` to find "latest state" — don't rely on `created_at` ordering under retries. |
| `prolific_pid` | `text not null` | Real Prolific ID, or a dev/local fallback — **open item, see below** (the old `pilot_${jatos.workerId}` fallback was JATOS-specific and needs a replacement convention). |
| `task` | `text not null check (task in ('numbers','colors'))` | |
| `pool_index` | `int not null` | Deterministic hash of `prolific_pid` — mechanism unchanged from current `poolIndexForParticipant`, just stored per-row same as now for self-containment. |
| `phase` | `text not null check (phase in ('welcome','consent','tutorial','trial','finished','terminated'))` | |
| `trial_index` | `int` | 0–31 for `phase='trial'`; `-1` sentinel for `tutorial`/`welcome`/`consent`/`finished`/`terminated`. **Not `null`** -- Postgres unique constraints treat two `NULL`s as distinct, not equal, so a `null` sentinel would silently break idempotency for exactly the phase (tutorial) this backend exists to fix. Caught during Edge Function implementation; fixed in code (`progress-append` derives/overrides this from `phase` rather than trusting the client), noted here so the doc matches. |
| `observation_index` | `int` | 0–14 for `tutorial`/`trial`; `-1` sentinel for `welcome`/`consent`/`finished`/`terminated`. Same not-null reasoning as `trial_index` above. |
| `attempt` | `int not null default 0` | Increment on a timeout-triggered replay of the same `(trial_index, observation_index)`. Always `0` for non-observation phases. |
| `response` | `numeric` | nullable |
| `timed_out` | `boolean` | nullable |
| `rt` | `numeric` | nullable |
| `value` | `numeric` | nullable |
| `true_mean` | `numeric` | nullable, numbers only |
| `true_std` | `numeric` | nullable, numbers only |
| `true_p` | `numeric` | nullable, colors only |
| `qid` | `int` | nullable |
| `error` | `numeric` | nullable, per-observation bonus error |
| `reward` | `numeric` | nullable, per-observation bonus reward |
| `created_at` | `timestamptz not null default now()` | |
| `updated_at` | `timestamptz not null default now()` | bump on every upsert |

**Unique constraint** (the idempotency mechanism):
`unique (prolific_pid, task, phase, trial_index, observation_index, attempt)`

**Indexes**: at minimum `(prolific_pid, task, id desc)` to make the
progress-check "latest row" lookup fast.

```sql
create table events (
  id                 bigserial primary key,
  prolific_pid       text not null,
  task               text not null check (task in ('numbers','colors')),
  pool_index         int not null,
  phase              text not null check (phase in
                        ('welcome','consent','tutorial','trial','finished','terminated')),
  trial_index        int,
  observation_index  int,
  attempt            int not null default 0,
  response           numeric,
  timed_out          boolean,
  rt                 numeric,
  value              numeric,
  true_mean          numeric,
  true_std           numeric,
  true_p             numeric,
  qid                int,
  error              numeric,
  reward             numeric,
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now(),
  unique (prolific_pid, task, phase, trial_index, observation_index, attempt)
);

create index events_latest_idx on events (prolific_pid, task, id desc);

alter table events enable row level security;
-- Deliberately NO policies created for anon/authenticated — default-deny.
-- Only the service-role key (used server-side in Edge Functions) can
-- read/write. Do not add a permissive policy "to make testing easier."

-- IMPORTANT (found during step 3h testing): disabling "Automatically
-- expose new tables" in the project's Data API settings withholds base
-- table-level GRANTs from ALL Data API roles, including service_role --
-- not just anon/authenticated as decision #1 above assumed. service_role's
-- RLS-bypass is a separate property from having table privileges at all;
-- it still needs an explicit GRANT. anon/authenticated deliberately get
-- nothing (verified via curl in step 2); service_role needs exactly this:
grant select, insert, update on public.events to service_role;
grant usage, select on sequence public.events_id_seq to service_role;

-- SAME GAP, FOUND AGAIN LATER (building the test suite): none of the
-- Edge Functions ever issue a DELETE, so this went unnoticed until the
-- test suite's own cleanup step (a DIRECT DELETE against the REST API
-- using the secret key, for removing test rows -- see tests/helpers.mjs's
-- cleanupTestRows) hit a real 403. Same root cause as above, just a
-- privilege nothing had exercised yet:
grant delete on public.events to service_role;
```

---

## Resume logic (`progress-check`)

Query the latest row for `(prolific_pid, task)`. Branch on `phase`:

- `finished` / `terminated` → return that status + the participant's
  Prolific code (completion vs. earlyExit per `PROLIFIC_CODES`). Client
  skips the timeline entirely.
- `trial` → resume at the start of `trial_index` if its last observation
  (14) isn't logged yet, else `trial_index + 1`.
- `tutorial` → resume at the start of the tutorial.
- `consent` (reached but never entered tutorial) → resume at the start of
  the tutorial, skip welcome/consent again.
- `welcome` only, or no rows at all → full run from welcome.

---

## Open items to resolve during implementation (not blocking table creation)

- ~~**Dev/local fallback for `prolific_pid`.**~~ Resolved: falls back to
  `dev_${Date.now()}` (timestamped, so repeated local runs don't collide)
  when `?PROLIFIC_PID=` isn't present -- see `timeline-builder.js`'s
  `buildAndRun`. No `pilot_`-style prefix; there's no JATOS worker ID to
  fall back to anymore.
- ~~**Consecutive-failure threshold for the visible warning**~~ Resolved:
  `backend-client.js`'s `createCheckpointSender` defaults to a threshold of
  2 consecutive failures, then shows a non-blocking banner
  (`timeline-builder.js`'s `showSaveWarning`/`hideSaveWarning`) until a
  checkpoint succeeds again. Verified for real during the numbers/colors
  rename: a genuine mismatch between client and backend task strings
  produced real `progress-append` failures, and the warning path fired
  correctly rather than failing silently -- an unplanned but useful live
  test of this exact mechanism.
- ~~**`progress-finish` mismatch handling**~~ Resolved as planned: still
  delivers the code, logs a `console.warn` (visible in the Edge Function's
  dashboard logs) for manual reconciliation rather than blocking payment.
- ~~**CORS on Edge Functions**~~ — done in `_shared/cors.ts`, applied in all three functions.
- **Env vars / key format (resolved during implementation, July 2026):**
  the new publishable/secret keys are NOT JWTs, so the platform's default
  `verify_jwt` gateway check rejects them outright if sent as
  `Authorization: Bearer`. All three functions are deployed with
  `verify_jwt = false` (already the CLI's scaffolded default) and do their
  own lightweight check instead (`_shared/auth-check.ts`): the caller must
  send our project's publishable key on the `apikey` header specifically,
  not `Authorization`. Resolved on the client too: `backend-client.js`
  uses plain `fetch` with only the `apikey` header, deliberately not
  `supabase-js`, to sidestep its default header behavior entirely.
  Key values are read from the new JSON-dict env vars
  (`SUPABASE_SECRET_KEYS`/`SUPABASE_PUBLISHABLE_KEYS`, keyed `"default"`),
  falling back to the legacy `SUPABASE_SERVICE_ROLE_KEY`/`SUPABASE_ANON_KEY`
  vars if those aren't set — see `_shared/supabase-admin.ts` and
  `_shared/auth-check.ts`.
- The client should only ever call the three Edge Function endpoints,
  never the table REST endpoints directly — enforced by the deny-all RLS,
  not by convention alone (verified in step 2).

---

## Terminology: numbers/colors, not continuous/binary

The two tasks are referred to as **numbers** and **colors** throughout
`task_backend` (file names, directory names, the `task` column/parameter,
class names, CSS classes, the `isColors` flag) -- renamed from the
old JATOS-era `continuous`/`binary` naming for consistency with the
participant-facing labels already used on the welcome screen and Prolific
itself. This is a deliberate, `task_backend`-only rename: the untouched
`task/` (JATOS) pipeline still uses `continuous`/`binary` internally, and
the two codebases are not expected to share terminology.

One exception, worth remembering if this ever needs re-doing: `bonus-
continuous.js` did NOT mechanically become `bonus-numbers.js` -- it's
shared by both tasks, so it was renamed to the task-neutral `scoring.js`
instead. A blind find-and-replace will get this wrong; fix it by hand.

---

## Next steps

1. ~~Create the Supabase project; run the `events` DDL~~ — done, verified
   (deny-all RLS confirmed via curl, `service_role` grants added after
   catching the "Automatically expose new tables" gap -- see the DDL's
   own comments above).
2. ~~Write the three Edge Functions~~ — done, deployed, and tested directly
   (idempotent upsert confirmed via a real duplicate-call test: retrying
   the same checkpoint left exactly one row, not two).
3. ~~Build client-side resumability~~ — done (`timeline-builder.js`'s
   `buildAndRun`, now `async`, calls `progress-check` first and branches
   into new/resume-tutorial/resume-trial/already-done). Verified with a
   real headless-browser test: the same participant ID, reloaded, landed
   directly on `tutorial_intro` without re-showing welcome/consent.
4. ~~Swap the JATOS save calls for the new endpoints~~ — done. The old
   blanket `on_trial_finish` JATOS-append hook was dropped entirely rather
   than ported (see the port review-pass inventory in chat history for
   why: only 3-4 of the old pipeline's 17 screen tags ever needed a
   network call under the new schema). `jatos-shim.js` and `PROLIFIC_CODES`
   no longer exist client-side at all -- the server hands back the code
   directly from `progress-finish`.
5. ~~Set up the weekly backup process~~ -- DECIDED AGAINST: no scheduled/
   automated backup. Instead, manual backup on demand once real
   participant data actually exists in the table (ask Claude for the
   exact steps at that point -- a `pg_dump` against the project's
   connection string, or the dashboard's own backup/export feature,
   whichever is simpler at the time). Revisit this decision if data
   volume or collection duration ever makes "manual, occasional" feel
   inadequate.
6. ~~Test the three failure modes directly~~ — (a) and (b) now fully done;
   (c) still open, see below.
   - **(a) mid-session resume, including a real trial-boundary case**: DONE,
     verified with a real headless-browser test — completed the tutorial,
     fully completed trial 0, advanced to trial 1 within the same session,
     reloaded as the same participant, and landed exactly at trial 1
     observation 0 (`progress-check` returned `resumeTrialIndex: 1`,
     matching). Also added and verified a UX gap found while testing:
     resuming into the trial loop previously showed NO transition at all
     (silently dropping the participant into an observation) — now always
     shows the same "Trial X / 32 — generating new sequence…" screen used
     for normal between-trial transitions, with the correct trial number,
     whether the session is fresh or resuming
     (`timeline-builder.js` — the `InterTrialPlugin` push moved out of the
     `!skipToTrials` branch to run unconditionally before the trial loop).
   - **The `attempt` counter under a real timeout**: DONE, verified —
     forced three real observation timeouts in a row on the same
     `(trial_index=0, observation_index=0)`; confirmed via the actual
     outgoing request bodies that `attempt` incremented `0 → 1 → 2` with
     `timed_out=true` each time, the same observation replayed each time
     (never silently advancing), and the third timeout correctly
     triggered `terminateSession` (landed on the `terminated` screen).
     Also surfaced that `ItiClockPlugin`'s `timed_out=true` branch
     requires a manual "Repeat" click rather than auto-advancing — not a
     bug, a deliberate anti-tab-visibility-exploit design already in the
     ported code, just one the test needed to account for.
   - **(c) a completed/terminated participant re-visiting the link**:
     DONE, verified. Seeded a fully-complete trial loop and a `terminated`
     row directly via the real API (avoids re-running 480 real
     observations through the UI for something that's really a
     `progress-check` branch test), then loaded each as a fresh page
     load: a finished participant sees "You already completed this
     study" + their completion code; a terminated participant sees "This
     study session already ended" + their (different) earlyExit code.
     Neither ever ran the jsPsych timeline.

**Follow-on UX fix, found while verifying 6c**: the original redirect
for real Prolific participants (`window.location.href = ...`) fired
immediately on success with no on-screen fallback -- if that redirect
ever failed (network hiccup, popup blocker, Prolific-side issue), the
participant would have had no way to know their own code. Fixed via a
new shared `renderCompletionScreen` (`finish-session.js`), used by all
three session-ending paths (finish, terminate, returning-participant):
the code is always shown as visible, selectable text, with a "Continue
to Prolific" button AND a ~4s auto-redirect -- closing the tab or a
failed redirect can never lose the code. Non-Prolific (dev/pilot)
participants see the code too now, for local-testing convenience.
Verified for real: all three paths (finish/terminate/returning) show the
correct, DIFFERENT codes (completion vs. earlyExit) as visible text.

**Real bug caught and fixed while building the above test**: `backend-
client.js`'s `appendProgress` passes extra checkpoint fields through via
`...rest` verbatim — no camelCase-to-snake_case transform. But
`build-trial-timeline.js`/`build-tutorial-timeline.js` were calling
`sendCheckpoint` with `timedOut`/`trueMean`/`trueStd`/`truep` (camelCase),
while `progress-append`'s Edge Function only recognizes the snake_case
`timed_out`/`true_mean`/`true_std`/`true_p`. Every trial/tutorial
checkpoint written before this fix has `NULL` for those four columns,
regardless of the real values — a real, silent data-integrity gap, not a
cosmetic one. Fixed in both files (all four call sites); confirmed fixed
via the real request-body inspection in the test above. **Any rows written
before this fix are wrong on these four columns specifically** — safe to
ignore for pure test/smoke-test rows (already cleaned up), but worth
remembering if this ever needs a real data audit.

Work through the remaining items (dedicated test suite, weekly backup,
hosting deployment) — **sequentially, one at a time** — confirm each
before moving to the next, per how this project has been run so far.

---

## Hosting (a gap nobody noticed until asked about it directly)

**This was never part of the original plan.** CLAUDE.md's "Pilot #3"
investigation and the resulting plan (this doc) were entirely about *why
participant data was being silently lost* -- replacing JATOS's data-saving
job. Nobody -- not CLAUDE.md, not this doc, not several sessions of
building -- ever explicitly asked "who serves the HTML/JS files once
JATOS isn't in the picture?" JATOS/MindProbe did two jobs at once (serve
static files AND track participants/save data); only the second job was
ever discussed. This surfaced only when asked directly, well after the
client port was built and tested.

**Why it doesn't conflict with the original reasoning, though**: Gorilla/
Cognition.run/Labvanced were rejected for being full-stack platforms that
wanted to own task logic, participant tracking, AND data storage all at
once. A plain static host never sees a participant response and never
touches the database -- it's a dumb file server, not a rival to the
Supabase decision. Its failure mode is also qualitatively different from
what motivated this whole redesign: if a static host is down, the page
simply doesn't load, immediately and unmistakably, before a session ever
starts -- nothing like JATOS's actual failure mode (a session running
normally while responses silently vanish underneath it).

**Decision path** (options considered, in order):
1. Netlify / Vercel / Cloudflare Pages / GitHub Pages all evaluated as
   free-tier static hosts for the combined Vite build (see "Build output"
   below).
2. Initially leaned GitHub Pages (repo already on GitHub for version
   control -- zero new account needed) -- until discovering GitHub Pages
   is free only for PUBLIC repos; this repo was private at the time.
   Pivoted to Cloudflare Pages (free regardless of repo visibility, no
   GitHub-Actions-workflow-file to maintain, unlimited free-tier
   bandwidth, no Netlify-style "one project over quota pauses the whole
   account" risk).
3. Decided to make the repo public anyway (no participant data in it --
   verified via `git log --all --full-history` before flipping
   visibility, see below; will need to be public for code-sharing at
   publication regardless). That removes the original GitHub Pages
   blocker, so **reverted to GitHub Pages** -- it satisfies the original
   preference (no new service/account) and this project's traffic is
   nowhere near where Cloudflare's bandwidth/reliability edge would
   actually matter.

**Pre-publicity data-safety check** (done before flipping repo
visibility): `task/pilot3/{numbers,colors}.txt` contain REAL participant
data (raw Prolific worker IDs, full response logs -- the actual pilot #3
incident data CLAUDE.md's history refers to) and were sitting untracked
on disk. Confirmed via `git log --all --full-history -- task/pilot3` and
a content-pattern search that this was NEVER committed to git history.
Added to `task/.gitignore` (alongside the two `compute_bonus*_tmp.py`
scripts that read directly from it) before making the repo public --
see that file's own comments.

**Build output**: combined into ONE deployment, not two -- `vite.config.js`
builds both `index-numbers.html` and `index-colors.html` into a single
`dist/` (plain `npm run build`, no `--mode`; Vite's own code-splitting
even automatically factors the shared `timeline-builder.js` module into
its own chunk between the two entries, a nice side effect of combining
rather than a design goal). `base` is conditionally set to
`/evidence_integration/` in production (GitHub Pages serves this repo as
a project page at that path) vs `/` in local dev -- confirmed both work
via `npm run build` producing correctly-prefixed asset URLs, checked
directly in the built HTML.

**Mechanics** (`.github/workflows/deploy-task-backend.yml`): triggers on
push to `main`, path-filtered to `task_backend/**` so unrelated commits
(`task/`, docs, analysis scripts) don't trigger a rebuild. Needs two
repo secrets (`VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY` -- see
`.env` for the actual values) and **Settings -> Pages -> Source: "GitHub
Actions"** set once in the GitHub UI -- neither is scriptable from this
repo, both are one-time manual steps.

**Verified live** (this session): the deployed site returns `200` at both
`https://psipeter.github.io/evidence_integration/index-numbers.html` and
`.../index-colors.html`, including the JS bundle itself (not a 404) --
confirming the `/evidence_integration/` base path resolved correctly at
build time. A real browser test against the live (not local-dev) site
confirmed the backend calls succeed too (the two secrets baked in
correctly, not `undefined`) and that reload/resume works end-to-end on
the actual deployed URL, not just localhost.

One real point of confusion hit during that verification, worth
recording since it'll recur: a close-tab-then-reopen test using the
**bare** URL (no `?PROLIFIC_PID=`) looked like a resume failure (landed
back on welcome) but wasn't one -- the dev/local fallback
(`dev_${Date.now()}`) deliberately mints a NEW, different participant ID
every page load with no query param present, specifically so repeated
local test runs don't collide with each other's rows. `progress-check`
correctly reported "new" for what was, from the server's perspective,
genuinely a brand-new participant both times. Testing resume on any
deployment (live or local) requires reusing the SAME explicit
`?PROLIFIC_PID=` value across both loads.

---

## Status note for future sessions

As of this note, the Supabase backend and the full `task_backend` client
port are built, deployed, and verified end-to-end against the real
database (not just unit-level) for: welcome/consent checkpointing,
tutorial resume, idempotent upsert, real trial-boundary resume (mid-
session, across a reload), the timeout-retry/`attempt`/terminate path,
and all three session-ending screens (finish/terminate/returning-
participant) showing the correct visible completion code. `CLAUDE.md`'s
"Own-backend decision (Supabase)" section already points here; no
further doc pointer needed.

Two things remain genuinely open, neither blocking the other:
1. ~~A persistent, re-runnable test suite.~~ DONE -- `task_backend/tests/`
   (`@playwright/test`), 5 spec files covering everything verified
   throughout this doc (basic flow, trial-boundary resume, timeout-retry/
   `attempt`, all three completion screens, a colors-task smoke check).
   Runs against the real deployed backend, not mocked. Self-cleaning via
   a local-only `.env.test` secret key (`SUPABASE_SECRET_KEY`) --
   surfaced one more instance of the same GRANT gap from the DDL section
   above (`DELETE` was never granted to `service_role` either, since no
   Edge Function had ever needed it until the suite's own cleanup step
   did); fixed the same way. All 7 tests pass.
2. ~~The weekly backup process~~ -- DECIDED AGAINST (see "Next steps" item
   5 above): manual, on-demand backup once real data exists, not a
   scheduled job. Nothing left to build here unless that decision changes.

Hosting (`Hosting` section above) is now fully done: repo public, GitHub
Pages live, verified against the real deployed URL (not just localhost).

**Everything from CLAUDE.md's original "Pilot #3" incident investigation
through this build-out is now done.** The only remaining open item in
the entire plan is the deferred manual-backup guidance above, which is
intentionally not something to build now -- it's something to ask for
when it's actually needed.

---

## Post-buildout review: deployment readiness (brainstorm session)

A step-back review turned up two real gaps neither of which had been
noticed before, plus a decision on a few more:

1. **Supabase free-tier 7-day auto-pause -- ACCEPTED, not fixed.** A
   project with zero DB activity for 7 days pauses until manually
   resumed; a real participant hitting a paused project during a gap
   would see total silence, arguably worse than the JATOS failure modes
   this backend was built to fix. Decided NOT to build a keep-alive ping
   for this: real data collection is expected to take only a few days
   once the study launches, well under the 7-day window. Revisit if
   collection ever stretches longer or happens in separated waves.
2. **No path from `events` back into the analysis pipeline (`fitting/`,
   `models/`) -- DEFERRED ON PURPOSE.** Confirmed via grep: zero existing
   script has any Supabase awareness. Decided to wait until the database
   is actually populated (even with fake/test data) before building this,
   and to do participant-data anonymization as PART OF that export step
   (distinct from the earlier repo-publicity anonymization, which was
   about `task/pilot3/`'s files, not the live database).
3. ~~**Bonus-payment CSV**~~ DONE -- `task_backend/compute_bonus.py`.
   Matches the original `task/compute_bonus_tmp.py`'s exact convention
   (only `phase='trial'` rows count toward bonus, mirroring the old
   schema's `screen=='observation'` exclusion of tutorial rows; reward in
   cents; $5.00 clip per participant). Verified against real data in the
   live table (paginated fetch, 1443 rows across 8 participants, correct
   per-participant sums and clipping).

   **Extended** to solve the Submission-ID-vs-Participant-ID mismatch
   found while building it: Prolific's dashboard bulk-bonus box wants
   `<submission_id>,<amount>` lines, not `prolific_pid`-keyed ones.
   `--prolific-export <path>` now takes Prolific's own demographic/
   submissions export (Submission ID + Participant ID side by side) and
   joins on `prolific_pid` automatically, printing ready-to-paste lines
   directly (the bulk-bonus box is copy-paste text, not a file upload --
   confirmed by the person testing it against the real UI). Nonzero-
   bonus participants NOT found in the export are excluded from the
   output and explicitly flagged for manual handling, rather than
   silently dropped or included with a blank ID that would fail the
   whole paste.

   Column matching reuses the exact fuzzy-substring-match-with-override
   pattern already proven in `task/reconcile_prolific_jatos.py`
   (`find_col`/`--prolific-id-col`) -- found this existing tool via a
   commit-message search *after* independently arriving at the same
   design, which is reassuring convergence rather than a wasted
   rediscovery. Added the same `--submission-id-col`/`--participant-id-col`
   override flags for consistency and resilience if Prolific's export
   wording ever shifts again (their own docs note it already has, more
   than once).

   Tested against a REAL Prolific demographic export (not synthetic) --
   confirmed correct parsing of the real header row
   (`Submission id,Participant id,Status,...`), correct join behavior in
   both directions (our test participants correctly reported "not found"
   in the real export; the real export's participants correctly reported
   "not found" against our test-only Supabase data), and the nonzero-
   bonus warning path firing correctly for real mismatched data.

4. **URGENT SAFETY GAP FOUND AND FIXED**: the real Prolific export used
   to test the above (containing real Participant IDs, real Submission
   IDs, and -- confirmed via a real `Completion code: C12FEFJU` value in
   one row -- tied to the actual live `colors` study) was placed directly
   in `task_backend/` with **zero gitignore protection**, in a repo
   that's now public. A `git add .` at that moment would have committed
   real participant PII. Fixed immediately: `task_backend/.gitignore`
   now excludes `prolific_*export*.csv` and, preemptively, `compute_bonus.py`'s
   own default output (`bonus_numbers.csv`/`bonus_colors.csv`/`bonus_*.csv`),
   which is exactly as sensitive (real IDs paired with real payment
   amounts) and would hit the identical gap the moment it's ever run for
   real. Confirmed via `git log --all --full-history` that nothing
   matching this was ever actually committed -- caught before any harm,
   not after. **Lesson for future sessions: any new file dropped into
   task_backend/ containing real participant identifiers needs an
   explicit gitignore entry checked immediately, not eventually** -- this
   is the second time in this project a real-data file has needed this
   exact reactive fix (the first was `task/pilot3/`); a proactive habit
   of checking before doing anything else with a newly-added file would
   be better than reacting each time.
5. **Desktop/laptop-only enrollment**: handled entirely on Prolific's
   side (their own device filters screen this out) -- no code needed.
6. **Researcher-facing monitoring and a kill-switch/maintenance-mode**
   -- discussed, deliberately NOT building either. Given the short
   (few-day) collection window: monitoring reduces to "glance at a couple
   of SQL queries during the first hour after launch," and a kill-switch
   is judged not worth building when the existing levers (pause the
   Prolific study to stop new starts; resume already means nobody loses
   progress if a fix requires a pause) are adequate at this scale.

---

## Small-sequence test variants (generate_sequences.py --name)

Added after a real design flaw was caught in an earlier attempt at a
full-session test: seeding most of a session directly via the API and
only driving a few observations for real ends up testing "does my seeded
data model of the app match the app" more than "does the app actually
work" -- exactly the kind of thing a real bug in the trial loop could
slip through undetected. The fix: generate a genuinely small (2-trial)
sequence file and drive the ENTIRE session through it for real, with
zero seeding anywhere. Same idea directly enables fast manual testing
too (locally or through a real Prolific preview), not just automated
tests.

**`generate_sequences.py --name <suffix>`**: builds a small variant
(`TEST_NUMBERS_N_PREFIX=2 x TEST_NUMBERS_N_REPEATS=1` = 2 trials for
numbers, `TEST_COLORS_N_TRIALS=2` for colors) using its OWN separate
constants, never the production `NUMBERS_*`/`COLORS_*` ones -- gated
behind the explicit `--name` flag specifically so there's no way to
accidentally produce a real-shaped production file with a tiny trial
count, or vice versa. Output: `sequences_<task>_<name>.json` -- omitting
`--name` leaves production behavior (`sequences_<task>.json`) completely
unaffected. Verified against real generated output: correct trial count,
correct distinct qids, correct schema.

**Client side** (`src/numbers|colors/config.js`): `VITE_SEQUENCES_VARIANT`
env var selects which file to load, via `import.meta.glob` (auto-
discovers ANY matching `sequences_<task>_*.json` file -- a new variant
never requires editing these files again, directly solving the "awkward
rerunning/copying/renaming" pain point from the old JATOS-era workflow).
Verified live in a real browser: with the variant set, the trial-
transition screen correctly showed "Trial 1 / 2", confirming the small
pool is genuinely loaded at runtime, not just that the build succeeded.

**Real tradeoff, deliberately accepted**: `import.meta.glob(..., {eager:
true})` bundles EVERY matching file at build time regardless of which
one gets selected at runtime -- confirmed directly (bundle size grew
~240KB just from the variant file existing on disk during a build).
Given this, and that the GitHub Actions deploy workflow does a fresh
checkout + build on every push (no "forgot to delete it" protection
possible in CI if a variant file were ever committed), **test-variant
files are gitignored** (`sequences_numbers_*.json` / `sequences_colors_
*.json` in `.gitignore`) -- generated locally/on-demand, never enter git
at all. `playwright.config.mjs` generates the `test2trial` variant
automatically if missing, so the test suite is self-sufficient on a
fresh checkout with no manual setup step.

**`tests/full-session-bonus.spec.mjs` rewritten** using this -- two
participants, each driven through a COMPLETE real session (welcome,
consent, tutorial, both real trials, the genuine `end` screen) with zero
seeding anywhere, using the `test2trial` variant. Captures the ACTUAL
`reward` the app computes per real observation (via the real outgoing
`progress-append` request bodies) rather than trying to hand-predict it
-- the slider has `step="1"` (confirmed directly), so any fractional
target response gets silently rounded by the browser itself, making
exact-value prediction fragile for no real benefit; capturing what the
app actually computed and verifying it survives intact through to
`compute_bonus.py`'s output is the property that actually matters.
Verified: all 10 tests in the full suite pass together (4.7 minutes
total), including this one and every pre-existing test now running
against the small variant instead of the 32-trial production pool.

**One coverage gap accepted, not fixed**: with only 2 trials (max 60
cents raw per participant), `compute_bonus.py`'s $5.00 clip can never
trigger in this test -- that branch was already verified manually
earlier this session against real over-$5 smoke-test data (see the
"Post-buildout review" section's bonus-CSV item above). This test's job
is full-pipeline integrity at a genuinely real, fast scale, not
exhaustive coverage of every reward magnitude.


---

## Tutorial redesign: "Correct answer" panel replaces the old KDE/urn
## figure (numbers + colors), pilot #3's comprehension problem

Motivated by a pilot #3 finding, brainstormed at length before any code:
a substantial fraction of participants appeared to just repeat the most
recently shown stimulus as their response, rather than integrating across
the sequence. Real-task-side interventions (mid-task nudges, always-on
scaffolding during actual trials) were explicitly rejected as too
intrusive / too likely to contaminate the measured behavior itself.
Decision: strengthen the TUTORIAL only, where added clarity can't
contaminate real trial data, and separately rely on post-hoc filtering
(an anchoring-score exclusion criterion) for whatever comprehension gap
remains -- not attempted yet, noted as a follow-up, not a blocker.

**A more fundamental problem found before touching the visuals**: the old
KDE curve (numbers) / urn bar (colors) visually taught the FIXED
population parameter (`true_mean`/`true_p`) as "the answer" -- but
`config-base.js`'s `ERROR_MODE` (`'running_mean'` for numbers, `'running_p'`
for colors) scores against the RUNNING statistic of what's been observed
so far, a materially different, evolving quantity, especially early in a
trial. The tutorial was pointing at the wrong target. Confirmed this
wasn't a stale/leftover config value before proceeding -- current,
intended scoring behavior for both tasks.

### The new "Correct answer" panel

`correct-answer-numbers.js` (numbers) / `correct-answer-colors.js`
(colors) -- both new files, replacing `distribution-numbers.js` +
`numbers-draw-animation.js` (numbers) and `colors-draw-animation.js`
(colors, `urn-colors.js` itself KEPT for its color constants, just no
longer used for the tutorial's own SVG bar) -- **all three deleted
entirely this session**; still present under `task/` if this ever needs
reverting.

- **Numbers**: a thin 0-100 slider-style track. A small BLUE CIRCLE thumb
  (not a bar -- deliberately small enough that the taller red ticks
  protrude visibly above/below it) SLIDES to the RUNNING MEAN's position
  every time a new observation arrives -- directly visualizing "this is
  what should move, and by how much." Red ticks mark every observation
  (faded for history, bold/taller for the current one).
- **Colors**: a thin blue/red bar, split at the RUNNING PROPORTION of
  blue draws so far (blue-left/red-right, SLIDES its split point the same
  way the numbers thumb slides). Small dots accumulate ABOVE the bar --
  one per draw, blue dots packed in from the LEFT edge, red dots packed
  in from the RIGHT edge, so their COUNTS (not a continuous position,
  unlike numbers -- colors' draws have no natural axis position) are
  what's shown. Dot spacing (`100/(N_OBS+1)`) guarantees the two
  sequences can never collide even in the all-one-color extreme.
- Both: NO bubbling animation, NO artificial delay of any kind (the old
  design's ~1s "wait for the bubbles before anything appears" -- found
  and removed for every phase, not just the ones later hidden behind an
  overlay). `SHOW_EXACT_VALUE` (both modules): whether the running
  statistic's exact number is shown as text next to the indicator, or
  left purely visual -- a flag, not a final decision; currently `false`.
  `renderCorrectAnswer(Colors)` reveals its own container and is fully
  self-sufficient for the observation plugin's every-call-already-visible
  usage; the intro plugin's one-time reveal passes `fadeIn:true`.

### Intro plugin (obs 1): three-click progression, redesigned from an
### earlier four-click version

Click 1 (left box 1) -> reveals box 1's own text AND the top-right box
(box 0b) together, plus the centre example number/circle fading in.
Click 2 (left box 2, the goal text) -> reveals box 2's own text AND the
whole correct-answer panel (no more separate click just for the panel).
Click 3 (left box 3, the slider instructions) -> reveals box 3's own text
AND activates the real response slider. The "Sequence history" tracker is
DELIBERATELY never revealed anywhere in either intro plugin file -- it
first becomes visible starting at observation 2 (the observation plugin,
which shows it unconditionally on every call), since sequence history
doesn't meaningfully apply yet with only one value.

Both right-column boxes show a "..." placeholder (matching the LEFT
column's own locked-box convention) until their real content is revealed.

### Box captions + sizing (numbers AND colors)

"Correct answer" / "Sequence history" captions: INSIDE each box,
top-center, absolutely positioned -- iterated through THREE placements
this session (above-as-flow-element, then an in-box overlay, back to
in-box) before landing here; kept for the record since the reasoning
mattered each time (see below). Only render when the box's OWN real
content is genuinely visible (`showCaptions = !hideGraphics && !showClock`
in both observation plugins) -- NOT during phase D (hidden behind an
overlay) or phase E (correct-answer replaced by the clock demo, tracker
hidden behind its own overlay) -- a caption reading "Correct answer" over
an opaque cover or a clock demo doesn't make sense, so the caller skips
rendering the caption's markup entirely rather than fighting stacking
order to hide something still in the DOM.

Six boxes total (3 left text boxes + 3 right boxes: top-box/correct-
answer/tracker), each `flex: 0 0 18vh` -- deliberately EQUAL on both
sides, not independently tuned/guessed per box (an explicit instruction,
after an earlier per-box-clamp() approach kept guessing wrong). Numbers
and colors use SEPARATE classes (`.numbers-tutorial-box` /
`.colors-tutorial-box`, identical properties) rather than one shared
class -- `.tutorial-info-block`/`.tutorial-right-top-box`/-image-box/
-tracker-box are ALL also used by the OTHER task's tutorial, and
coupling their box heights through one class would be a new exception to
this codebase's existing "independently designed tutorials" pattern for
no real benefit.

### Real bugs found and fixed while building this (all confirmed via
### direct DOM/geometry checks in a real browser, not visual guessing)

1. **Observation plugin never revealed the panel's own container.**
   `renderCorrectAnswer(Colors)`'s wrapper starts at CSS `opacity:0`
   (meant to be revealed by a staged-reveal helper); the intro plugin
   called that helper, the OBSERVATION plugin never did, since it has no
   staged reveal at all. Result: ticks/thumb were being positioned
   correctly the whole time (confirmed via computed style) but stayed
   invisible because the PARENT's opacity zeroed out everything inside it
   regardless -- a bug an earlier round of "check the child's own
   opacity" testing completely missed. Fixed by having the render
   function reveal its own container unconditionally.
2. **The box's own white background/border was placed INSIDE the
   hidden-until-revealed content wrapper.** Before reveal, the entire box
   frame vanished along with the content, leaving the bare page
   background showing through (reported directly as "an empty gray
   area"). Fixed by moving `.dist-canvas` (the class providing that
   background) onto the OUTER, always-visible box element, matching the
   left column's own `makeBox` convention (box chrome always visible;
   only the specific real-content span toggles).
3. **The hidden content wrapper had no explicit height**, so
   `correct-answer-outer`'s `height:100%` had nothing real to resolve
   against, collapsing/misplacing the indicator (reported as "a shrunk
   and overlapping box"). Fixed with an explicit `height:100%` on that
   wrapper, verified after the fix via exact pixel geometry (content
   rect height matches the box minus padding; the track's own vertical
   center matches the box's vertical center exactly).
4. **An early height-overflow diagnosis was wrong.** A "boxes taller than
   an 800px window" symptom was initially (incorrectly) attributed to two
   new caption elements; measured scroll-height was BYTE-IDENTICAL before
   and after removing that theory's own fix, proving captions were never
   the cause. The real, much larger factor turned out to be the LEFT
   column's own text boxes, which never had a `vh`-based height budget at
   all (`flex:1`, pure content-driven height, ~20-25vh per box measured
   directly) -- unlike every box in the right column. Surfaced only once
   asked to reason in `vh` terms consistently, per this project's own
   established sizing convention, rather than chasing absolute pixel
   numbers against one arbitrarily-chosen window size.

### Verification discipline

Every claim above was confirmed via real headless-browser checks (exact
`getComputedStyle`/`getBoundingClientRect` values, not just "the code
looks right") -- this was itself a lesson relearned mid-session: an
early screenshot misread ("the centre panel looks blank") turned out
wrong once cross-checked against the DOM directly, so subsequent checks
leaned on precise element geometry/computed style as the primary
evidence, with screenshots as a secondary, lower-confidence sanity check
rather than the main source of truth.


---

## Codebase cleanup pass (post-tutorial-redesign)

After the "Correct answer" panel redesign (previous section) left a lot of
now-inapplicable comments/exports scattered around, asked for a thorough
cleanup: remove dead code and stale comments referencing methods no longer
in use, then consider further consolidation/renaming. Two-part effort,
tackled in that order.

### Part 1: dead code removal -- DONE

Verified via a full sweep of every exported symbol against the WHOLE
`src/` tree (not just `shared/` -- an early version of this check was
wrongly scoped and produced false positives for `pickTutorialExample`/
`buildConfig`, which are used in `numbers/config.js`/`colors/config.js`).

Removed:
- `urn-colors.js`: `buildUrnSVG`, `LAYOUT`, `DIM_BLUE`, `DIM_RED` -- all
  only used by the tutorial's old bar-drawing code (deleted the session
  before this one). File was reduced to 3 color constants, then later
  folded away entirely (see Part 2).
- `tutorial-text-colors.js` / `tutorial-text-numbers.js`: `URN_CAPTION` /
  `DIST_CAPTION` (both dead, both had admitted as much in their own
  "pending a later pass" comments -- that pass never happened) and
  `WARNING_YELLOW` (dead in both files).
- `draw-performance-numbers.js`: `coinGlyph` (never called anywhere, not
  even internally -- confirmed only its own declaration existed) and
  `COIN_STROKE` (only used inside the now-deleted `coinGlyph`).
- Fixed every docstring caught pointing at deleted files/functions
  (`buildHintHTML`, `distribution-numbers.js`, etc.) or describing CSS
  values that no longer apply.
- CSS: found (and confirmed EMPIRICALLY via computed-style checks in a
  real browser, not just reasoning about specificity) that
  `.tutorial-right-image-box`/`.tutorial-right-tracker-box`/
  `.tutorial-right-top-box`'s own `flex`/`overflow` rules are now ALWAYS
  overridden by the newer `.numbers-tutorial-box`/`.colors-tutorial-box`
  classes (2-class selectors beat 1-class ones, and every current usage
  pairs them) -- trimmed the dead lines and the comments explaining
  reasoning that no longer applies. Fixed two stale comment references to
  classes renamed away earlier in the redesign
  (`.correct-answer-box-short`, `.tutorial-right-bottom-box`).

### Part 2: consolidation -- DONE (one significant finding), IN PROGRESS (broader pass)

**Found and fixed**: the same 3 hex colors (`#2563eb` blue / `#ef4444`
red / `#16a34a` green) were independently redeclared as "canonical"
named constants in 5+ different files -- `tutorial-text-numbers.js`'s
GOAL_COLOR/SAMPLE_COLOR/DIST_COLOR, the old `urn-colors.js`'s
SAMPLE_BLUE/SAMPLE_RED/DIST_COLOR, `draw-performance-numbers.js`'s
MEAN_BLUE/SAMPLE_RED/ERROR_GREEN, `draw-performance-colors.js`'s own
local SAMPLE_BLUE/SAMPLE_RED/DIST_COLOR, plus `plugin-observation-
numbers.js` hardcoding the red inline and `plugin-observation-colors.js`
redeclaring blue/red locally instead of importing its own task's
existing canonical source.

New `palette.js` is now the ONE underlying source (`BLUE`/`RED`/`GREEN`).
Every one of the files above now imports from it, keeping its own
semantically-named local alias (e.g. numbers's GOAL_COLOR vs colors's
SAMPLE_BLUE for the identical hex value) -- this fixes there being one
underlying value per color, not that every file must call it the same
thing. Deliberately did NOT extend this to the more incidental one-off
uses of the same hex values (`plugin-iti-clock.js`'s clock rendering,
`create-terminate-session.js`'s "Too slow" message, `slider-colors.js`'s
ruler bands, `observation-timeout-clock.js`'s warning color,
`tutorial-tracker.js`'s default parameter) -- those are standalone UI-
styling choices that happen to reuse the same brand colors, not
canonical named constants multiple files were each separately trying to
own; touching all of those too would be a much bigger, lower-value
refactor. Revisit only if a real inconsistency (not just a style
preference) surfaces there too.

Went one step further: `urn-colors.js`, once stripped to just 3 re-
exported constants, was folded directly INTO `tutorial-text-colors.js`
and deleted entirely -- numbers never had an equivalent separate color
file (its colors are defined directly in `tutorial-text-numbers.js`), so
this gives both tasks the exact same file structure. Redirected all 4
remaining importers (`correct-answer-colors.js`, `plugin-tutorial-intro-
colors.js`, `plugin-tutorial-observation-colors.js`, `plugin-tutorial-
summary-colors.js`), merging duplicate import lines where a file was
already importing separately from both. Verified: zero dangling
imports, every touched file syntax-checks clean.

**Outstanding from this pass** (left exactly where the previous session
ended, for whoever picks this back up):
1. **Naming question, not yet resolved**: `plugin-inter-trial.js` (the
   "Trial X / 40, generating new sequence…" between-trials reset screen)
   vs `plugin-iti-clock.js` -- "inter-trial" vs "ITI" (the standard
   abbreviation for the identical term) used inconsistently across two
   filenames. Had just re-read `plugin-inter-trial.js` (confirmed: the
   reset screen) but had NOT yet read `plugin-iti-clock.js` to determine
   whether it's a genuinely separate concept (names fine as-is) or the
   same thing named two different ways (one should be renamed for
   consistency) before running out of room in that session.
2. **`phases.js`** -- not yet opened/reviewed at all this pass.
3. **Broader naming-consistency pass** not yet done: whether `build-*.js`
   vs `plugin-*.js` naming holds up consistently throughout, and whether
   any OTHER file has the same kind of duplicated-canonical-constant
   pattern the color consolidation just fixed (the color case was found
   by deliberately checking; nothing guarantees it's the only instance).
4. ~~**Nothing has been tested since this whole cleanup/consolidation pass
   began**~~ -- DONE, see "Test suite verification and consolidation"
   below for the full account (real bugs found, colors coverage added,
   and a suite rewrite/consolidation done alongside it).

---

## Test suite verification and consolidation (this session)

Addresses "Outstanding from this pass" item 4 above -- the color
consolidation and dead-code removal (9 files touched, 1 deleted) had
never been exercised in a browser at all. Two real bugs were found doing
this, both in TEST code, not app code; the suite itself was then
consolidated as a separate, explicitly-requested follow-on.

**Real bug #1, found immediately**: `tests/helpers.mjs`'s
`clickThroughTutorialIntro` still drove the OLD 4-click tutorial-intro
flow (`tut-box-0` -> `#tut-image-placeholder` -> `tut-box-1` ->
`tut-box-2`), from before an earlier session's redesign collapsed the
image reveal into box-0's own click (see "Tutorial redesign" section
above, "Intro plugin (obs 1): three-click progression"). `#tut-image-
placeholder` no longer exists in either intro plugin's markup for either
task. Every test that calls `completeTutorial()` (which calls this
function first) hung on Playwright's own actionability wait for a
selector matching zero elements -- surfacing only as a 3-minute
test-level timeout on whichever test happened to reach the tutorial
first in the run order, not as an obvious "element not found" error.
Fixed to the real 3-click sequence (`tests/helpers.mjs`). This was a
STALE TEST, not a real app regression -- the app's own tutorial-intro
plugins were already correct; the test just hadn't been updated to
match them since that earlier redesign.

**Real bug #2, a genuine but UNREPRODUCED anomaly, not confirmed**: one
run of an earlier version of the full-session test found only 28/30 real
trial rows in the database for one of two participants driven through a
complete real UI session (the browser itself successfully reached the
genuine "Session complete" screen for both). Immediately re-ran the same
test 3 more times back to back -- all 3 came back with the full 30/30.
Most likely explanation: a single fire-and-forget `progress-append` call
dropped under Playwright's faster-than-human click pace (a much tighter
request-burst pattern than a real participant's own pacing would ever
produce), not a systematic bug. Also manually verified via a slow,
human-paced local session (`npm run dev:numbers`, `PROLIFIC_PID=test8`)
queried directly against the database afterward: 48/48 rows, fully
gapless, zero loss. **Left as an open, unresolved data point, not
dismissed**: if `happy-path.spec.mjs`'s own row-completeness check (see
below) ever comes back short again, that's a second occurrence worth
escalating -- e.g. checking whether `backend-client.js`'s consecutive-
failure warning banner fired, which it may not have even if this WAS
real loss (2 total dropped rows across a ~60-checkpoint session isn't
necessarily 2 CONSECUTIVE failures for the same participant, the actual
threshold that trips the banner).

**Suite consolidated afterward** (separate, explicitly-requested follow-
on once the above fixes were confirmed working): the pre-existing suite
had grown to 6 spec files driving 6 full tutorial traversals total
(`basic-flow.spec.mjs`, `colors-smoke.spec.mjs`, and
`full-session-bonus.spec.mjs`'s own 2-participant design, each paying for
a fresh multi-minute traversal to attach what was often just one
additional assertion). Rewritten down to 4 files / 4 traversals:

- **`basic-flow.spec.mjs`, `colors-smoke.spec.mjs`, and
  `full-session-bonus.spec.mjs` all DELETED**, replaced by a single new
  **`tests/happy-path.spec.mjs`** -- ONE canonical full-session traversal
  per task (numbers, colors), each `test.describe.serial` block split
  into an explicit two-phase design: test 1 is PURE UI-level (drives the
  real session end to end, asserts only on what the browser can observe
  directly -- screens reached, zero console errors, every checkpoint's
  HTTP status, and a cheap nonzero-reward sanity check guarding against a
  repeat of this project's own past BONUS_DECAY-miscalibration bug, see
  the "Tutorial redesign..." section's bonus-formula history elsewhere in
  this doc); test 2 (database-only, no UI) runs AFTER test 1 and checks
  the resulting rows are the complete, gapless set. `test.describe.serial`
  means a failed test 1 skips test 2 automatically -- a genuine pre-test/
  traversal-then-inspect structure, not just two tests that happen to run
  in file order.
- **Simplified from an earlier 2-participant design to 1** (per explicit
  direction -- "fall back to a single participant, just to confirm the
  bonus calculation is correct"): the second participant in the old
  `full-session-bonus.spec.mjs` existed only to prove two response
  strategies produced two DIFFERENT bonus totals -- a check on the test's
  OWN design, not something a real app bug would trip. One participant
  is enough to verify the reward pipeline carries a real number through
  faithfully end to end.
- **A dedicated `color-rendering.spec.mjs`** (written this session) had
  added `getComputedStyle` checks for `palette.js`'s BLUE/RED at the
  tutorial's correct-answer panel and centre stimulus, plus the first
  real trial's own stimulus color -- confirming the exact color-
  consolidation risk "Outstanding" item 4 above was originally worried
  about. It was DELETED per direction ("the color tests seem
  unnecessary") before ever being run against the fixed suite. Note for
  whoever revisits this: while writing it, one real TEST bug (not an app
  bug) was caught by inspection before deletion -- `#tut-ball`'s (colors
  task) background/border-color fade is a genuine 1000ms CSS transition
  (`plugin-tutorial-intro-colors.js`'s `onBox0`), so a computed-style
  check needs to either wait for the transition to settle
  (`page.waitForFunction` polling for one of the two real palette
  colors, not a fixed timeout) or check something that isn't
  mid-animation -- a fixed-wait version of this check would be flaky by
  construction, not just occasionally slow. If a dedicated color check
  is ever reinstated, don't reintroduce this specific race.
- **`resume.spec.mjs`, `timeout-retry.spec.mjs`, `completion-screens.spec.mjs`
  all UNCHANGED** -- each tests a genuinely distinct code path
  (reload/resume; the `attempt`/terminate path; `progress-check`'s
  branching for already-finished/terminated participants) that can't be
  folded into the happy-path traversal without losing the ability to
  tell "this specific path broke" apart from "everything broke."
  `completion-screens.spec.mjs` got one stale-comment fix (referenced the
  now-deleted `full-session-bonus.spec.mjs` by name).

**Net result, verified**: 10 tests across 4 files (was 12 across 6),
4 full traversals (was 6), full suite run in 4.4 minutes. All 10 pass,
including BOTH tasks' happy-path traversal for the first time ever in
this suite -- colors previously only ever got driven as far as
`tutorial_intro` (`colors-smoke.spec.mjs`'s own stated scope), never a
full tutorial + real trial + database check. This closes that real gap,
not just the original color-consolidation risk.

**Still open from the original "Outstanding from this pass" list**
(items 1-3 above) -- all three now investigated; see "Naming-question
follow-up" below for the full findings and resolution.

---

## Naming-question follow-up (items 1-3 from "Outstanding from this pass")

### 1. `plugin-inter-trial.js` vs `plugin-iti-clock.js` -- RESOLVED (genuinely separate; a real terminology finding, not renamed)

Traced every real usage in `build-trial-timeline.js` to answer this
definitively rather than guessing from filenames alone:

- `ItiClockPlugin` (`plugin-iti-clock.js`) fires **between every
  observation WITHIN a trial** (the `if (o > 0)` guard inside the
  per-observation loop), plus its `timed_out:true` variant as the
  timeout-retry replay screen (`screen: 'iti_replay'`). It never fires
  between trials.
- `InterTrialPlugin` (`plugin-inter-trial.js`) fires **once per trial
  boundary** -- the "Trial X/N -- generating new sequence..." pacing
  screen between the trial-summary screen and the next trial's first
  observation (`screen: 'inter_trial_reset'`).

**Conclusion: NOT the same concept named twice** -- these are genuinely
different screens serving different roles, so no functional duplication
exists. But there IS a real terminology inversion worth recording: in
standard psych-experiment usage, "ITI" specifically means Inter-*Trial*
Interval. The plugin doing the actual between-*trial* pause is named
`inter-trial` (spelled out, no abbreviation used at all), while the
plugin claiming the "ITI" name (`plugin-iti-clock.js`) is doing what's
actually an inter-*observation* pause -- closer to what the literature
calls ISI (Inter-Stimulus Interval). The two files have effectively
swapped which one "deserves" the ITI name. `plugin-iti-clock.js`'s own
docstring even asserts "circular countdown clock between trials," which
is factually wrong about its OWN behavior -- confirmed via the real
usage sites above, not just re-reading the docstring's own claim at
face value.

A separate, smaller wrinkle in the `screen` tag namespace (not the file-
naming question, but adjacent, worth knowing about): `TrialSummaryPlugin`
(a THIRD, unrelated plugin) uses `screen: 'inter_trial'`, while
`InterTrialPlugin` uses `screen: 'inter_trial_reset'` -- two different
plugins, two similarly-prefixed tags, easy to conflate when reading test
code or raw DB rows (the summary/bonus screen is `inter_trial`; the
"generating new sequence" pacing screen is `inter_trial_reset`).

**Decision: documented, NOT renamed.** This is a documentation/precision
issue, not a functional bug -- unlike the color-duplication case,
nothing behaves incorrectly here. A rename would touch
`build-trial-timeline.js`, `build-tutorial-timeline.js`,
`timeline-builder.js`'s imports, three Playwright spec files'
`data-screen` assertions (`iti_replay`, `inter_trial`,
`inter_trial_reset`), and a fair amount of prose in both CLAUDE.md and
this doc -- a real, bounded, but nontrivial blast radius purely for
naming precision. Revisit if this genuinely confuses someone in
practice; not worth doing speculatively.

### 2. `phases.js` -- RESOLVED, clean

Reviewed in full. `PHASES` (welcome/consent/tutorial/trial/finished/
terminated) is a small, well-scoped enum, and a grep across the whole
`src/` tree confirmed nothing else independently redeclares these values
-- no repeat of the color-consolidation's duplicated-canonical-constant
pattern here. (The `screen` tag values that happen to share some of the
same strings, e.g. `screen: 'welcome'` in `build-welcome-screen.js`, are
a deliberately SEPARATE namespace from `phase` -- confirmed via
`timeline-builder.js`'s own checkpoint calls, which pass `PHASES.WELCOME`
independently of whatever `screen` tag the DOM carries; `screen` is a
UI-only bookkeeping label, never sent to the backend at all, so this is
not the same risk category as the color case.)

One trivial, safe fix made: the module's own docstring pointed at
`terminate-session.js (createTerminateSession)`, but the real file is
`create-terminate-session.js` -- fixed (comment-only, zero behavior
change).

### 3. Broader naming-consistency pass -- CHECKED, no further action recommended

Verified programmatically (not by eye) across the whole `src/shared/`
tree: every `build-*.js` file exports a plain builder function with no
jsPsych-plugin shape; every `plugin-*.js` file exports a real jsPsych
plugin (`info` + `trial()`). Zero exceptions in either direction across
all 17 files in each category. `create-terminate-session.js`/
`finish-session.js`/`timeline-builder.js` are a legitimate third
category (hand-rolled, non-jsPsych-trial DOM/orchestration code) --
consistent with the same pattern already established for the old
`task/` pipeline's `create-early-exit.js`, not a new inconsistency.

Also re-checked specifically for the color-consolidation's own failure
signature (a value independently redeclared as a "canonical" constant in
5+ files) against `phase`/`screen` string literals project-wide --
found no second instance.

**Conclusion: the naming conventions that matter for correctness are
fully consistent. What's left (the ITI/inter-trial terminology inversion
above) is cosmetic and already fully documented. Not launching a further
speculative sweep on this basis.**

---

## Deployment-readiness re-check and cutover status (this session)

Prompted by a direct question -- has anything changed since the
"Post-buildout review" and "Status note" sections above, and has the
actual Prolific cutover happened yet? Checked empirically, not by
re-reading old notes at face value.

**Both the site and the backend are live and healthy right now**:
`index-numbers.html`/`index-colors.html` on GitHub Pages both return
`200`; `progress-check` returns `200` (the Supabase project is NOT
currently auto-paused).

**Confirmed the live bundle actually serves real production content, not
test data** -- pulled the deployed JS bundle directly (not just trusted
the build config) and checked: zero occurrences of `test2trial` anywhere
in it, and exactly 6,400 occurrences of `"true_std":10` -- precisely 200
pool members x 32 trials, matching the documented current-production pool
exactly. Traced WHY this is safe by construction, not just lucky this
time: `sequences_numbers.json`/`sequences_colors.json` (the real
200-member pools) ARE tracked in git; only the `_*`-suffixed variant
files are gitignored (see "Small-sequence test variants" section above)
-- so a fresh CI checkout can never accidentally bundle test data, and
`deploy-task-backend.yml` never sets `VITE_SEQUENCES_VARIANT` at all.

**UPDATE (later session): `"true_std":10` above is now stale as a
specific number** -- `NUMBERS_STD_FIXED` (generate_sequences.py) was
deliberately changed 10 -> 15 in a later session (a std=10 pool didn't
leave enough suffix-variance budget for the fixed tutorial example's
high-variance prefix, flatlining the rest of that trial -- see
generate_sequences.py's own comment on NUMBERS_STD_FIXED for the full
reasoning), and the pool was regenerated accordingly. The VERIFICATION
METHOD above (pull the live bundle, count occurrences of the current
true_std value x 6400) is still the right check to re-run after any
future deploy -- just re-run it against whatever `NUMBERS_STD_FIXED`
currently is, don't assume it's still 10. This is exactly the kind of
duplicated-literal drift this project has hit before; avoid re-
hardcoding the specific number in a future note like this one if
avoidable.

**The two "accepted, not fixed" risks from "Post-buildout review"**,
re-checked:
- 7-day auto-pause: not paused right now (incidental -- this session's
  own testing kept it warm, not a fix). The original risk-acceptance
  reasoning assumed real collection would start "within a few days" of
  that review; **worth confirming directly whether that timeline still
  holds**, since the cutover below still hasn't happened -- if launch
  keeps slipping, revisit whether the auto-pause risk is still
  acceptable as-is.
- No `events` -> analysis-pipeline path: still nothing built, still fine
  to defer -- confirmed there's no real data yet to build against (see
  below).

**Cutover status: CONFIRMED NOT DONE, and here's exactly what's
blocking it.** Checked `task_backend`'s Edge Functions directly:
`supabase/functions/_shared/prolific-codes.ts` hands back the EXACT SAME
completion/early-exit codes as the old JATOS pipeline's `PROLIFIC_CODES`
(`C1CNSEMJ`/`C1ARJ6LO` for numbers, `C12FEFJU`/`C1L1GGHT` for colors) --
a deliberate design choice, not an oversight, made specifically so the
cutover never requires creating new Prolific studies. This means the
cutover reduces to ONE manual step per task, done entirely in Prolific's
own dashboard: change each existing study's Study URL field from the
JATOS/MindProbe link to:

    https://psipeter.github.io/evidence_integration/index-numbers.html?PROLIFIC_PID={{%PROLIFIC_PID%}}
    https://psipeter.github.io/evidence_integration/index-colors.html?PROLIFIC_PID={{%PROLIFIC_PID%}}

Simpler than the old URL, too -- confirmed via `timeline-builder.js` that
it only ever reads `?PROLIFIC_PID=` from the query string and has no use
for JATOS's old `STUDY_ID`/`SESSION_ID` params, so those can be dropped
entirely rather than carried over.

Confirmed from the data side too, not just the code side: querying the
live table shows nothing but test-prefixed participant IDs (`test_*`,
`dev_*`, manual smoke-test pids) -- zero real Prolific traffic has ever
reached `task_backend`.

**What's left before that flip, as far as code can verify**: nothing
code-side is blocking it -- backend, hosting, and completion-code
delivery are all confirmed working end-to-end (this session's
`happy-path`/`resume`/`timeout-retry` tests exercise the exact same paths
a real participant would hit). The remaining steps are Prolific-dashboard
actions, not code: (1) update the Study URL field on both existing
studies to the URLs above; (2) a final real "preview as participant"
click-through against the LIVE url (not localhost) before opening to real
traffic -- the same bar this project has held every deployment claim to
since the JATOS-era pre-deployment checklist (see CLAUDE.md's own
checklist, which caught real bugs no amount of local/E2E testing could).

---

## Real Prolific pilots run; cutover now actually done (this session)

**Correction to the section directly above**: "CONFIRMED NOT DONE" was
accurate at the time it was written, but the cutover has since actually
happened. Two real pilot rounds have run against `task_backend` directly:

- **Pilot 4** (`NUMBERS_STD_FIXED=15`): 5 real Prolific participants
  completed BOTH tasks. All 5 confirmed to pass every one of
  `utils.participant_filters.filter_participants`' exclusion criteria
  (no_integration/noncontingent_sign/noncontingent_magnitude) -- zero
  excluded. Two participants each missing exactly one checkpoint (479/480
  observations) -- the same fire-and-forget checkpoint-loss class this
  backend's own reliability hardening (below) targets, not something new.
- **Pilot 5** (`NUMBERS_STD_FIXED=10`, numbers only): 8 finished so far
  as of the last check, all with a clean 480/480 -- no missing-checkpoint
  gaps at all, consistent with the reliability hardening below actually
  helping. 2 more terminated (one via 3 timeouts in a single trial almost
  immediately; one via Prolific's own screening/return mechanism,
  unrelated to any in-app bug -- traced both directly against real
  checkpoint data before concluding this). A few more still in progress.

Bonus payments computed and paid for both rounds via a fresh pull from
Supabase each time (not `compute_bonus.py`'s own CSV path this session --
recomputed directly from raw per-observation `response`/`value` data to
support the running-mean-based error/reward math, and to test different
`BONUS_DECAY` values retroactively against already-collected pilot 5 data
before deciding on a value going forward -- see "Bonus formula" below).

---

## Checkpoint reliability hardening (this session)

Three real fixes, found while reviewing the checkpoint path rather than
waiting for another silent-loss incident:

1. **Bounded retry** (`backend-client.js`'s `createCheckpointSender`):
   `appendProgress` now retries up to 2 more times (300ms, 800ms) before
   counting as a failure, same idempotent payload each attempt. Still
   fire-and-forget from the caller's perspective; `onWarning` only fires
   after ALL retries are exhausted, not on the first transient failure.
2. **End-of-session catch-up resend**: `progress-finish` now computes and
   returns the SPECIFIC missing `(trial_index, observation_index)` pairs
   (not just a boolean) when a "finished" claim's row count comes up
   short. `timeline-builder.js` now maintains a `checkpointLedger` (every
   trial-phase checkpoint payload attempted this session, keyed by
   `trialIndex-observationIndex`) so `finish-session.js` can resend the
   EXACT original payload for whatever's reported missing, without
   needing to re-derive it from sequence data.
3. **A resume-ordering bug found while building the above**: the catch-up
   resend writes trial rows AFTER the session's own 'finished' marker
   (higher `id`s), which would have broken `progress-check`'s "latest row
   = current state" resume logic for anyone who reloaded after a
   successful catch-up resend. Fixed: `progress-check` now checks for an
   existing finished/terminated marker FIRST, independent of row
   recency, before falling through to the general latest-row query --
   robust against any future scenario with rows written after a terminal
   marker, not just this one.

---

## Numbers std reverted 15 -> 10; generation-pipeline repair simplified (this session)

`NUMBERS_STD_FIXED`: 15 -> 10 (second reversal -- see the constant's own
comment in `generate_sequences.py` for the full 10 -> 15 -> 10 history).
Motivated by pilot 4 showing little to no clear |delta response| decay
signal in real participants; testing directly whether std=15 was simply
too noisy a task for genuine evidence-accumulation behavior to show up,
rather than assumed.

**A real, previously-undiagnosed bug found and fixed while investigating
std tolerance**: ~0.77% of a generated pool's trials landed outside the
intended std tolerance band. Traced to specific prefix/target pairings
(from the Hungarian assignment step) with a large mismatch between a
prefix's own mean and its assigned target -- large enough to force
`build_numbers_suffix`'s analytical variance-correction formula negative,
which floors to a near-zero-noise suffix. Confirmed this is NOT fixable
by raising `max_attempts` (tried first, raising 20 -> 100 made zero
difference -- every retry draws from the same near-frozen distribution).
Landed on: for any qid where suffix retries still can't get every repeat
within tolerance, regenerate that qid's WHOLE PREFIX from scratch and
rebuild all repeats against their already-assigned targets (up to 30
attempts) -- reduced outliers to ~0.08-0.09%.

**A separate pairwise target-swap repair (tried first, then removed)**:
before landing on the prefix-regeneration approach above, tried swapping
targets between pairs immediately after Hungarian matching whenever the
analytical formula would go negative. Worked for about half the cases,
but confirmed empirically REDUNDANT once qid-level prefix regeneration
existed (6/6400 vs 5/6400 outliers testing regeneration alone, on the
same real 200-member pool) -- removed entirely rather than keeping two
repair mechanisms reasoning about the same problem from different ends.
The pipeline now asks one direct question repeatedly (does the achieved
std land in tolerance?) instead.

---

## Fixed tutorial sequences (this session)

`choose_tutorial_sequences` (`generate_sequences.py`, run via
`--tutorial`) selects ONE real trial per task from the production pool to
serve as every participant's tutorial example, replacing an earlier
dynamic `pickTutorialExample()` that picked from pool member 0 at load
time. Numbers: two-stage selection (percentile-band on early prefix-
response variability, then best suffix-response variability within that
band), plus a hard filter against any candidate with a repeated raw value
anywhere in its 15 observations. Supports an explicit `exclude` set (and
`--exclude_current` CLI flag) so a "try a different one" request can
guarantee landing on a genuinely different trial, since the selection is
otherwise fully deterministic. Re-run whenever the production pool is
regenerated with different parameters -- confirmed this matters directly
(the numbers std reversion above changed which trial got selected, since
the pool itself changed).

Colors reuses the exact same two-stage selection logic, but needs its own
repeat structure first, since colors' literal `qid` never repeats at all
(see "Colors quasi-qids" below) -- `choose_tutorial_sequences` calls
`add_quasi_qids` internally for colors before scoring candidates.

---

## Distractor system removed; ERROR_MODE confirmed as production default (this session)

Prompted by a production-readiness review that found `config-base.js`'s
`ERROR_MODE` comment describing `'running_mean'` as "set for testing" --
confirmed directly with the PI that this IS the intended production
choice, not a leftover test setting. Comment (and `CLAUDE.md`'s own
Scoring section) updated to say so plainly.

Same review found `DISTRACTOR_TYPE` always `'none'` in both tasks'
configs, with no plan to ever use anything else for this study. Removed
entirely rather than left inert: `config-base.js`'s `DISTRACTOR_TYPE`,
`build-trial-timeline.js`/`timeline-builder.js`'s `distractorType`/
`ITI_DISTRACT_MS` plumbing, and `plugin-iti-clock.js`'s entire "popup
distractor" mechanism (~70 lines: `_circleSize`/`_numFontPx`/
`_placeNoOverlap`/`_spawnPopup`/`_startPopups`/`_stopPopups`, plus the
`distractor_type`/`iti_condition`/`is_colors` params that only that
mechanism used) -- confirmed every reference before removing, not
assumed. `iti_condition` itself (`'control'`/`'distract'`) is still
generated into every sequence (`generate_sequences.py`'s own balanced-
repeat design untouched) but is no longer consumed anywhere client-side
-- inert DATA now, not inert CODE. Bundle size dropped measurably as a
direct result (confirmed: `timeline-builder`'s own JS chunk shrank from
46.73KB to 45.81KB gzipped).

---

## Bonus formula: BONUS_DECAY split per task (this session)

Prompted by a direct observation: pilot 5 (std=10) showed meaningfully
higher performance (lower absolute RMSE) than pilot 4 (std=15) on
numbers. Confirmed this was mostly MECHANICAL, not genuine skill
improvement, before touching anything: `scoring.js`'s reward formula
used a FIXED absolute error tolerance (~6.7 points on the 0-100 scale,
from the shared `BONUS_DECAY=15`), completely independent of
`std_fixed` -- checking error relative to std_fixed showed genuinely
comparable relative precision across both pilots (~0.60 vs ~0.68), so
the absolute improvement was mostly the lower-noise task making the same
fixed tolerance easier to stay inside, not real behavioral change.

Ran a decay sweep (15/22/30/40/50/60) against real pilot 5 responses
(recomputed from raw data, not the already-paid amounts) before choosing
a value: decay=15 and decay=22 both left 4/8 people hitting the $5
manual-payment cap; decay=50 eliminated cap-hitting entirely but crushed
lower performers' pay much harder. Settled on **decay=25** (2/8 still
capped) as a middle ground, after checking a few candidate values against
real data rather than guessing.

Split into `NUMBERS_BONUS_DECAY=25` / `COLORS_BONUS_DECAY=15` (colors
unchanged -- its task design hasn't changed, so no reason to touch it),
not one shared constant -- avoids silently changing colors' payouts as a
side effect of a numbers-specific tuning decision. `computeResponseReward`/
`computeTrialReward` now take `bonusDecay` as a REQUIRED parameter (no
default) -- every one of the 7 call sites across
`build-trial-timeline.js`/`build-tutorial-timeline.js` passes it
explicitly via `isColors`, so a missed call site fails loudly rather than
silently falling back to the wrong task's value. Only affects FUTURE
sessions -- confirmed pilot 4/5's already-paid amounts are unaffected
(recomputed pilot 4 colors fresh from the database under the unchanged
`COLORS_BONUS_DECAY=15`: identical total to what was already paid).

---

## Data pipeline built: Supabase -> figure_soltani_* (this session)

Addresses the "Data pipeline (deferred on purpose)" item from earlier
sections above -- no longer deferred. `scripts/build_task_backend_inputs.py`
pulls real, finished participant data directly from Supabase's `events`
table for an EXPLICIT list of `prolific_pid`s per pilot round (`--pilot
<name> --numbers_pids ... --colors_pids ...`), not "everyone finished so
far" -- different pilot rounds are different people with different
generative parameters (numbers' `std_fixed` changing between pilot 4 and
5), so silently merging them would make cross-pilot comparison
impossible. `--list_candidates <task>` probes current real-participant
status directly from Supabase (finished/terminated/in-progress, plus
`true_std`) for building an accurate pid list without guessing from
memory -- used directly to find pilot 5 had progressed further than last
individually checked (8 finished, not the 3 last confirmed).

Reuses `build_model_inputs.py`'s existing filter/rescale/anonymize/save
pipeline (`build_from_df()`, refactored out of that file's old JATOS-
pilot-file-only `build()` so both sources share one implementation, not
two divergent copies) -- writes to `data/task_continuous_<name>.pkl`/
`task_binary_<name>.pkl`. Real bug found and fixed while building this:
`build_from_df` unconditionally wrote BOTH outputs even when one had zero
input rows (pilot 5 has no colors data at all), producing a meaningless
empty file -- fixed to skip writing when a call's input for that task is
empty, with a clear message, rather than overwrite nothing with junk.

`figure_soltani_{performance,temporal,variability}.py` all take a general
`--datafile <name>` argument (a plain filename suffix, not a pilot-
specific concept, so it generalizes to a future non-pilot experiment
dataset with zero further script changes) pointing at these files; each
degrades to an explicit placeholder, not a crash, when a task has no file
for a given datafile.

---

## figure_soltani_* rebuilt on real data (this session)

All three figures switched from the old raw JATOS-pilot-file path (or,
for performance/temporal, from files that hadn't existed at all yet) to
the data pipeline above. Human-data-only in all three -- model fitting
against real task_backend data hasn't been run yet (estimated at
15-20+ minutes even locally for a full Optuna k-fold-CV pass across 4
models x 2 tasks x 5-8 pids), deliberately deferred to its own pass
rather than attempted as a side effect of building the data pipeline.

Several real, narrow bugs found and fixed by actually running these
against real data rather than by inspection:
- `true_p`/`true_mean` arriving via `json.loads()` (real Supabase JSON)
  can silently end up as `dtype=object` (Python floats) rather than
  `float64`, invisible until a numpy ufunc call on them crashes --
  fixed with explicit `pd.to_numeric()` casts.
- The autocorrelation panel (temporal, col 4) paired residuals by ARRAY
  POSITION, not actual observation index -- wrong wherever a lost
  checkpoint leaves a gap (e.g. observations `[1,3]` logged, `0`/`2`
  missing), since it would treat array-adjacent-but-observation-distant
  values as a "lag=1" pair. Confirmed directly (pid=1's lag=1 pair count
  in pilot 4 dropped from 92 -- 2 spurious -- to the correct 90 after the
  fix) and impact-checked (2/160 prefix-trials in pilot 4, 0/256 in pilot
  5). Fixed by pairing via an explicit observation-index dict per trial.
- The split-half lambda fit (`scipy.optimize.curve_fit`'s bounded
  nonlinear fit, `lam in [0,2]`) reliably degenerated to the `lam=0`
  floor on real, noisy human data (only ~32 trials per pid) -- returning
  `lambda~1e-11` to `1e-21`, a floor artifact, not a genuine estimate.
  Fixed by switching to a log-log linear regression (`lambda = -slope of
  log(delta) vs log(observation)`), which has no boundary to stick to and
  can honestly return a small or negative lambda for a genuinely flat or
  slightly-increasing curve instead of degenerating.

**Colors quasi-qids** (`utils/colors_quasi_qids.py`, new): colors' own
literal `qid` never repeats (every trial gets its own, by design --
confirmed empirically, 0/640 (pid,obs,qid) groups have any repeat). For
every metric that needs a repeat structure (temporal cols 3-4,
variability's row 1 and cross-task panel), derives one empirically
instead: group a participant's trials by their own literal first-4 raw
stimulus values, keep only groups with >=3 repeats. Deliberately does
NOT also require matching on target level (`true_p`) -- numbers' own qid
repeats already only condition on shared prefix identity, not shared
target, so requiring colors to additionally match on target would be a
stricter, non-parallel standard; confirmed empirically that pooling
across target levels (not stratifying) gives a much richer, evenly-spread
sample with no real validity cost. `PREFIX_LENGTH=4`/`MIN_REPEATS=3`
chosen from a full P=1..6 x R=2..4 sweep against real data, not guessed.

**Ground truth changed to running mean**: `figure_soltani_performance.py`
and `figure_soltani_temporal.py`'s performance panels now compare
responses against the RUNNING mean/ratio of the observed stimulus stream,
matching `scoring.js`'s own `ERROR_MODE` and `plot_sequences.py`'s
`gt_mode='running_mean'` -- not the fixed generative parameter, which
these panels used to use.

**Split-half reliability changed to odd/even trials**, in three places
(`figure_soltani_temporal.py`, `figure_soltani_variability.py`,
`plot_sequences.py`'s `split_half_lambda`) -- a strict first-half/
second-half split confounds genuine estimation noise with any systematic
drift in behavior over the session (learning, fatigue); interleaving
odd/even trials isolates noise from drift, the standard recommendation
over a strict chronological split.

---

## Production-readiness review (this session)

A systematic pass before moving beyond piloting, covering: deployment
state (git clean/pushed, live bundle confirmed fresh via commit-vs-
response timestamp comparison, Edge Functions confirmed healthy and
already deployed), RLS policies (verified EMPIRICALLY, not just
documented -- both SELECT and INSERT against the real `events` table
using only the anon/publishable key correctly returned 401), secret-key
handling (confirmed absent from the built `dist/` bundle AND from the
ENTIRE git history, not just current state), and the full Playwright
suite (11/11 passing; one flaky, non-reproducing failure along the way
that passed cleanly on immediate re-run, consistent with this project's
own previously-documented class of rare Playwright timing flakiness, not
a regression).

One thing flagged but explicitly NOT resolved: no SQL migration file is
tracked in the repo for the `events` table/RLS setup -- it exists only in
Supabase's dashboard. Not a security problem, but nothing in git could
rebuild it if the dashboard config were ever lost.

One thing flagged and explicitly left as a known unknown: Supabase plan
tier / rate limits couldn't be verified without dashboard access; rough
math suggests Edge Function invocations (not storage) are the more
likely constraint to watch as real participant N grows, but this depends
on the actual plan tier and total planned N, neither confidently known
from this session.

## Participant exclusion criteria: four candidates, and how we chose (this session)

Started from the hypothesis that the existing filter was TOO AGGRESSIVE (it
excluded 33/60 = 55% of complete_pairs as a union). That hypothesis turned out to
be wrong, but chasing it produced a better criterion and, more usefully, a
definition to justify it. Recording the progression because most of the dead ends
are ones that would be re-attempted otherwise.

### The four candidates

| method | basis | numbers | colors |
|--------|-------|---------|--------|
| `contingency` | three Cohen's f² tests (recency_only, noncontingent sign/magnitude) | 25/60 (42%) | 19/60 (32%) |
| `performance` | carrabin's rule: mean abs error >N SD above the retained mean | 9/60 (15%) | 9/60 (15%) |
| `integration` | skill score vs "copy the latest observation" | 36/61 (59%) | ~1 |
| **`non_integrator`** | **prior observations make no RELIABLE contribution** | **19/61 (31%)** | **17/61 (28%)** |

`non_integrator` is now the default. `performance` and `integration` moved to
`archive/utils/archive_exclusion_criteria.py`. `contingency` retained as a
computed DIAGNOSTIC (it no longer decides anything) because recency_only tests the
same construct as non_integrator by a different method, and their agreement --
23/25 and 18/19 -- is what validates the exclusions.

### Published precedent, which is what prompted the whole investigation

carrabin excluded 4/25 (16%) on ONE model-free quantity: mean |p̂ − p|, with the
excluded group at .263 (SD .0298) against .176 (SD .0132), a >6 SD separation.
yoo excluded 8/46 (17%), of which SEVEN were fMRI-technical (1 structural
abnormality, 6 head motion >3 mm) and exactly ONE was behavioural -- a
post-experiment questionnaire in which the subject said they tracked pairwise
differences rather than the average. Neither used a model-based contingency test.
Applying our own criteria back to their data: carrabin's 21 retained subjects all
score skill 1.00-1.51 (0 would fail ours), but 7 of yoo's 38 (18%) WOULD fail --
so our criterion is stricter than field norms, which is worth stating.

### Why the high rate is real, not an artefact

The `integration` criterion is model-free, shares no quantity with the temporal
panels, and independently reproduced 23/25 and 18/19 of the contingency
exclusions. Roughly half of numbers participants perform worse than reporting
ONLY the latest observation (mean error 8.35 vs an optimal 3.28). The original
"too aggressive" hypothesis is disconfirmed.

### What the bad participants are actually DOING

Assigning each participant the candidate strategy that best predicts their
responses. Zero good participants are best fit by `last_value`; zero bad ones by
`running_mean`. But there are TWO distinct failure modes, not one:

- **Literal copying** (23 of 37 bad numbers participants). Exact-copy rates of
  0.63-1.00, best-fit error down to 0.0-1.6 points. Transcribing the stimulus.
- **Drifting** (11 of 37). Best fit by their OWN PREVIOUS RESPONSE, moving the
  slider constantly (within-trial SD 7.7-27.1 vs 4.4 for good participants) with
  no relation to the evidence. Not a sticky slider -- they move MORE than good
  participants.

That second mode is why a single-axis filter cannot work: copiers sit at one
extreme of any weighting measure and drifters sit in the middle, alongside genuine
integrators.

Contributing context: they are NOT rushing (median inter-observation latency 4.2 s
vs 3.4 s for retained -- SLOWER), not timing out, not leaving the slider still,
and passed the tutorial first try. Comprehension failure, not inattention. Which
explains why five rounds of instruction and bonus tweaks did not move it. The
tutorial teaches but has NO pass/fail gate; gating it on demonstrated performance
is the highest-value fix available.

### Measured and rejected

- **`frac_copy_value`** as a metric. Confounded by `true_std`: at std=10, copying
  the latest value is nearly correct. Superseded by skill/`g_lag0`, which
  normalise per participant.
- **The skill score** (`integration`). Its threshold was defensible -- a 0.29-wide
  empirical void, so any cut in (0.041, 0.334) gave the identical partition -- but
  the METRIC is not monotone in integration depth. On synthetic leaky integrators
  it PEAKS at α=0.20 (+0.745), above a near-optimal α=0.10 (+0.603), because with
  15 observations mild recency overweighting tracks the running mean better than a
  sluggish filter. A genuine α=0.70 integrator with realistic noise scores +0.115,
  a hair above its own 0.10 threshold. It discards inaccurate integrators.
- **`g_lag0`**, the serial-position weight on the latest observation from
  regressing each response on ALL prior observations. Recovers α almost exactly
  (0.100/0.200/0.350/0.494/0.687/0.959 for true α 0.10-1.00) and is nearly immune
  to response noise -- so it IS the right MEASURE of integration depth. But it is
  continuous with no natural cutoff (largest gap 0.076 across 0.03-1.00) and
  CANNOT catch random responders, whose diffuse weights score ~0.12,
  indistinguishable from optimal. Any weight-based test is blind to the "nothing
  predicts them" mode. Report it descriptively; do not filter on it.
- **`gain`** (b_current + b_prior ≈ 1 as a validity check). Catches random and
  frozen responders cleanly (0.00 vs 0.94-1.00) but was rejected as a filter
  component in favour of a single test.
- **A one-sided version** of the final criterion, to catch scale inversion
  (reporting % red for % blue). 1 of 61 numbers and 0 of 61 colors are reliably
  negative, and that one is marginal (b=-0.074, CI [-0.171,-0.013]).
- **Stability across session halves.** 26% of retained numbers participants pass
  pooled but not both halves -- and the asymmetry runs the WRONG way for fatigue:
  12 integrate only in the SECOND half against 4 only in the first. Mostly late
  LEARNING, consistent with error falling 19% from the first 8 to the last 8
  trials. Would penalise a slow start.
- **A trials 8-31 burn-in.** Moves retention by ONE participant per task (numbers
  42→41, colors 43→44), retained sets indistinguishable in accuracy on the same
  late trials (4.90 vs 4.79; 6.94 vs 7.03). Use all 32.

### Fatigue: there is none

Within task, error DECREASES with trial index (numbers -0.075/trial p=0.001;
colors -0.066 p=0.004), first 8 → last 8 trials 9.63 → 7.76 (-19.4%). Across
tasks the second task is if anything better (skill 0.262 → 0.380, p=0.45), with
order well counterbalanced (32 numbers-first, 28 colors-first). Only 3 numbers and
1 colors participant worsen beyond their own noise (per-pid slope t>2); the 22
with positive slopes are the upper half of a null distribution, and 22/60 is BELOW
the 30/60 expected by chance. The slope IS a reliable individual trait (split-half
r=0.758 numbers) -- but the reliable variation is in how much people IMPROVE.
Low-skill participants improve FASTEST (corr(skill, slope) = +0.297, p=0.021),
which is the opposite of "the disengaged fatigue".

Trial order is randomised per participant (54 and 53 distinct orders across 60),
and adjusting each trial's error for that trial's optimal-agent error leaves the
slope unchanged (-0.0751 → -0.0741), so the improvement is not a sequence artefact.

### `require_both_tasks` became the default

A participant failing in either task is dropped from BOTH. Found by following up a
collapse in the within-subject cross-task panels: under per-task exclusion with
`integration`, numbers retained 29 and colors 36, and the 26-pid intersection was
a differently-selected group -- cross-task λ correlation fell to r=0.331 (p=0.099)
from r=0.587 (p=0.0013). NOT power (cp_perf at n=44 gives r=0.572), NOT reliability
(λ split-half was HIGHER, colors 0.836 vs 0.796; attenuation ceilings 0.791 vs
0.780), NOT range restriction (λ SD and range unchanged). Purely the composition
of the intersection.

### Effect on the results

| build | n both | cross-task λ r | col 3 p | col 2 decay | λ numbers |
|-------|--------|----------------|---------|-------------|-----------|
| contingency (`complete_pairs`) | 27 | 0.587 (p=.0013) | 0.00043 | 3.09x | 0.433 |
| **non_integrator (`cp_ni`)** | **35** | **0.508 (p=.0018)** | **0.00051** | **2.46x** | **0.390** |
| integration | 24 | 0.338 (p=.107) | 0.00000 | 2.90x | 0.474 |
| performance | 44 | 0.572 (p<.0001) | 0.033 | 1.23x | 0.266 |
| no filter | 60 | 0.656 (p<.0001) | 0.011 | 1.13x | 0.240 |

`non_integrator` gives 8 MORE participants than the contingency filter with every
result intact -- the only criterion that improves the sample without weakening
anything.

Two asymmetries worth carrying into any write-up. The DECAY results (cols 2-3)
need a filter and weaken monotonically as it loosens. The CROSS-TASK λ correlation
does NOT -- it is strongest with everyone included (r=0.656, n=60), so that finding
depends on no exclusion at all. Different panels have different sensitivity to
exclusion; report the asymmetry rather than smoothing it over.

### Honest limitations of the chosen criterion

Not threshold-free, though I described it that way at one point. It removes the
arbitrary MAGNITUDE threshold and replaces it with a conventional significance
level, whose sensitivity is: ci=90/95/99 → 16/17/24 flagged (numbers), 17/17/20
(colors), so ci=99 adds 7 (+41%). Bootstrap seed moved 2-3 participants at
n_boot=2000, hence the default n_boot=20000 (verified: seeds 0/1/2 identical, ~10 s
per task via a Gram-matrix bootstrap). The PREDICTOR SET is the largest source of
variation -- last-3-lags + older mean gives numbers 23 (churn +10/-4) and colors 15
(+3/-5) -- but `prior_mean` is right on principle: it asks the definitional
question as ONE test, whereas the full-lag version widens every CI (power loss →
more flagged) while giving four uncorrected chances at significance (multiplicity →
fewer flagged); those errors moving in opposite directions across tasks is the
signature of an ill-posed test.

Known gaps, deliberately not engineered around: it retains anyone whose responses
reliably use history, so it does not catch integrating the WRONG STATISTIC (running
sum, max, a subset), SCALE COMPRESSION (right direction, only 40-60 of the slider),
or ANCHORED-WITH-A-NUDGE. The first two are arguably correct to retain; the third
is a real miss. And being a significance test it is POWER-DEPENDENT: the ~30% rate
is tied to this design's 32 trials.

## NoisyRL_lambda: response noise reconciles the fitted/descriptive lambda gap (this session)

Started from a worry that RL_lambda was fitting badly: most fitted `lambda_` above
0.75 (9/35 pinned at exactly 1.0) against a descriptive lambda -- measured from the
decay of |Δresponse| -- below 0.5. Chasing it produced a resolution, a new model,
and three retracted hypotheses of mine along the way. Recording the whole
progression because the dead ends were each plausible.

### The fits were not bad; the comparison was mis-specified

Two things were measured before drawing any conclusion:
- **Parameter recovery works.** Fitting noiseless RL_lambda to noisy RL_lambda data
  recovers lambda almost exactly at every noise level tested (true 0.20 -> 0.192,
  0.40 -> 0.367, 0.60 -> 0.592, 0.80 -> 0.808, even at noise SD 0.10, double the
  human level). So misspecification-via-noise does NOT bias lambda upward.
- **lambda is strongly identified, not weakly.** The RMSE profile along lambda is
  steep and monotone (pid 1: 0.1032 at lambda=0 -> 0.0484 at 0.8). No flat region,
  no boundary drift. The optimiser finds a real minimum.

So the high fitted lambda is a finding about response LEVELS, not an artefact. The
two lambdas measure different things: fitted lambda answers "what weighting
reproduces where the slider ends up", descriptive lambda answers "how fast does the
amount of movement shrink".

### The mechanism, and what "noise" means here

Human |Δresponse| PLATEAUS (~0.06 on [-1,1]) rather than decaying to zero. A
deterministic RL_lambda has |Δ| -> 0 by construction, since |Δ| = alpha(t)*|PE| and
alpha(t) -> 0. Sequence variation keeps |PE| alive but cannot keep |Δ| alive -- the
gain shrinks regardless. This was a genuine confusion worth resolving explicitly:
the relevant noise is RESPONSE noise (slider imprecision, lapses, tick rounding),
NOT sequence variation. The two are separable because within a qid group the stimuli
are IDENTICAL, so within-qid residual SD measures response variation with sequences
held fixed. Measured: humans ~0.055; every math model exactly 0.000e+00.

Decisive test: adding each pid's OWN measured response noise to RL_lambda's fitted
output moves its descriptive lambda from 0.921 to 0.369 against a human 0.294 --
paired gap +0.008, p=0.668, indistinguishable -- with the plateau also matching
(gap -0.0044, p=0.599). For colors the deterministic model was already close and
noise overshoots, so this is largely a numbers-task phenomenon.

### PrimacyRecency does not show the discrepancy, and the reason is structural

At median fitted parameters, the normalised weight on the NEWEST observation:

| t | PrimacyRecency | RL_lambda alpha(t) |
|---|----------------|--------------------|
| 1 | 1.000 | 0.947 |
| 5 | 0.373 | 0.305 |
| 15 | 0.327 | 0.141 |

PR asymptotes to a CONSTANT (~0.33) because for `o=t` the recency factor is
`eps_r^1` at every t, so the newest observation always retains weight. RL_lambda's
alpha decays without bound. Hence PR's |Δ| plateaus by construction and RL_lambda's
cannot. Late-observation plateau, numbers: human 0.0633, PR 0.0445, RL_lambda
0.0223, Mean 0.0131. Descriptive-lambda gap ordered identically: PR +0.219,
RL_lambda +0.396, Mean +0.765. LeakyIntegrator overshoots the other way (plateau
0.0992, gap -0.232) because fixed gamma means constant weight forever. So the three
models BRACKET the human -- a coherent story about wanting a decaying-but-floored
learning rate.

### Retracted along the way

- **"Add an asymptotic term to alpha(t)."** Wrong. Response noise produces an
  |Δ| floor by itself (`E|Δ| -> 1.128*sigma` even when the systematic Δ -> 0), so
  an alpha floor would fit noise with a systematic parameter. Tested per pid: 13/35
  prefer a floored power law, but the fitted floor is statistically
  indistinguishable from the floor PREDICTED by each pid's own response noise
  (numbers median 0.0455 vs 0.0625 predicted, p=0.377, correlated r=0.661 across
  pids). Note this also invalidates the descriptive-lambda level we had been
  quoting: fitting the floored form gives lambda median 1.642 vs 0.294 for the pure
  form, so the noise floor drags the pure-form exponent down.
- **"LeakyIntegrator's `v=0.0` init handicaps it."** Wrong, and checked against the
  task rather than argued: `DEFAULT_VALUE = 50` and `lastResponse` is reset at every
  trial start, so each trial genuinely begins with the slider at the midpoint =
  exactly 0.0 on [-1,1]. The init IS the task's initial condition; initialising at
  x_0 would ignore where the slider actually was.
- **"PrimacyRecency's recency exponent disagrees with its docstring."** Wrong.
  Verified numerically identical under the docstring's stated 1-indexed convention;
  I had misread `eps_r**(n-o)` as giving `eps_r^0` on the last observation when it
  gives `eps_r^1`.

### The model, and what it establishes

`NoisyRL_lambda` = RL_lambda + `sigma_state` (perturbs the estimate, compounds ->
variance growth and autocorrelation, i.e. temporal cols 3-4) + `sigma_resp`
(perturbs only the report, i.i.d. -> a plateau, no autocorrelation). Reduces to
RL_lambda exactly at sigma=0.

RMSE cannot identify either sigma (both collapse to ~0; 24-25 of 35 exactly zero),
so both carry nonzero LOWER BOUNDS chosen by matching human prefix variability and
RMSE-vs-running-mean. With floors in place essentially every pid sits at them, so
the fitted values are not evidence about the noise level.

What it buys, with `alpha_0`/`lambda_` barely moved (numbers lambda 0.704 -> 0.662,
r=0.964):

| numbers | ratio first/last | plateau | descriptive lambda | gap vs human |
|---------|------------------|---------|--------------------|--------------|
| HUMAN | 2.46 | 0.0633 | 0.294 | -- |
| RL_lambda | 7.24 | 0.0223 | 0.921 | +0.382, p<0.0001 |
| NoisyRL_lambda | **2.50** | 0.0537 | 0.405 | **+0.035, p=0.62** |

Colors: plateau becomes exact (0.0853 vs 0.0854) and the per-pid lambda correlation
improves 0.782 -> 0.894.

**Circularity, stated honestly.** sigma_resp's floor was calibrated to the measured
within-qid residual SD, and the plateau is largely a function of that quantity -- so
the plateau match is partly by construction. NOT circular: nothing tied sigma to the
DECAY RATIO or the descriptive lambda, and both landed on target. Two independent
quantities from one calibrated input.

### Still open

- Identical noise for all pids gives human-scale variability but not human
  individual DIFFERENCES: NoisyRL_lambda's prefix-variability distribution is a
  narrow spike (~0.04-0.05) vs the human's broad 0.2-0.5, split-half reliability is
  weaker (numbers r=0.49** vs 0.81****), and the numbers lambda correlation actually
  DROPS 0.644 -> 0.524 even as the level matches. Per-participant sigma_resp fixed
  at each pid's measured value is the obvious next step; MODEL_PARAMS supports a
  `fixed` dict but not per-pid values.
- The observation-0 variability profile cannot be matched by this model family
  (human 0.0093 -> 0.0515 step; model always highest at observation 0). Probably
  task structure rather than a noise process.
- A distributional (NLL) loss remains the only route that would actually FIT the
  noise rather than calibrate it. Tabled: `compute_sim_db_loss` keys cells on the
  FULL sequence tuple, which suits carrabin's repeated sequence pool but not
  soltani's mostly-unique per-participant sequences, and `build_sim_db`
  hand-duplicates each model's implementation (seeding by simulation index rather
  than by trial), so adding a third pair doubles the drift risk.

## RNN as a conditional-mean estimator: works for carrabin, fails for soltani (this session)

Goal was a distributional fit metric to complement RMSE, which cannot see variance
(demonstrated repeatedly this session: NoisyRL_lambda's noise parameters collapse to
their bounds under RMSE). The proposed route -- from earlier discussion -- was to
fit an RNN per participant as a "best possible" conditional-mean estimator, then use
(a) the residual SD as a response-variability metric covering all 15 observations
rather than only the shared prefix, and (b) the RNN prediction as a DENOISED target
for a distributional loss.

Conclusion: the premise is DATASET-DEPENDENT. It holds for carrabin and fails for
soltani, for a structural reason rather than a tuning one. Use the RNN for
carrabin; for soltani keep qid-grouped response std.

### The decisive test, and what it showed

The test is whether the RNN beats simple models on HELD-OUT data. If a 2-parameter
model out-predicts it, it is not a best-possible conditional mean, and its residual
is contaminated with its own prediction error rather than being response noise.

| | trials/pid | obs/trial | sequences | held-out RMSE |
|---|---|---|---|---|
| carrabin | 200 | 5 | repeating pool | RNN 0.1225 BEATS NoisyCounting 0.1324 (15/21 pids), and every other model 21/21 |
| soltani numbers | 32 | 15 | mostly unique | RNN 0.0626 LOSES to RL_lambda 0.0526 (0/4 pids) and to the parameter-free running mean 0.0545 |

Carrabin gives the GRU 6x more trials AND repeating sequences, so a held-out trial
has often been seen in training -- interpolation. Soltani's 32 sequences are unique,
so a held-out trial is genuinely novel -- extrapolation, which is the regime where a
101-parameter model loses to a 2-parameter delta rule.

Worth noting the carrabin numbers came from files already on disk
(data/runs/carrabin/RNN_carrabin_performance.pkl), whose saved `loss` IS
`cv_rmse` -- genuinely out-of-fold at k=5, n_hidden=4.

### Two of my hypotheses were wrong

Asked why the same setup succeeded on carrabin and failed on soltani, I proposed
(a) the old carrabin result was in-sample and (b) it used a different n_hidden.
BOTH wrong: the saved loss is held-out cv_rmse, and n_hidden was 4 in both cases.
The difference is training data and sequence structure, which neither hypothesis
touched.

### n_hidden matters but does not rescue it

Sweep on soltani_numbers, 4 pids, k=8 (28 of 32 trials per fit), held-out RMSE:

| n_hidden | RMSE | vs RL_lambda |
|---|---|---|
| 1 | 0.1751 | +233% |
| 2 | 0.0911 | +73% |
| **3** | **0.0626** | **+19%** |
| 4 | 0.0701 | +33% |
| 5 | 0.0722 | +37% |

A clean U-shape with an interior optimum -- 1 underfits, 4-5 overfit -- so the
default of 4 WAS mistuned, by ~19 percentage points. But even at n_hidden=3
RL_lambda wins on 4/4 pids and the zero-parameter running mean still wins too.

### Consequences for the two applications

- **sigma_RNN cannot replace prefix variability for soltani.** At the best setting
  it is 0.0626 against the qid-repeat estimate of ~0.055 -- only 14% inflated,
  which is tempting -- but the inflation is the GRU's own prediction error, and
  RL_lambda's residual on the same rows would give a lower estimate still. At the
  original settings (k=5, n_hidden=4) sigma_RNN was 0.18, more than 3x the qid
  estimate.
- **The RNN prediction cannot be a denoised target for a distributional loss on
  soltani**, because it is LESS accurate than the models being evaluated. Scoring
  NoisyRL_lambda against a target that RL_lambda predicts better would be perverse.

An NLL needs no conditional-mean estimator anyway: score the observed y under the
model's simulated predictive distribution. That penalises mean AND variance
mismatch together as a proper scoring rule -- the quadratic term punishes a wrong
mean and understated variance, log(sigma) punishes overstated variance. The RNN's
remaining possible roles were (i) a denoised target for the mean term and (ii) a
noise ceiling for normalising NLL across participants; (i) is now ruled out for
soltani, and (ii) would inherit the same contamination.

Note the NLL is UNDEFINED for deterministic models (sigma_m = 0 gives infinite
NLL; compute_sim_db_loss clamps at 1e-3, silently turning it into scaled squared
error with an arbitrary scale). So it applies to stochastic models only and
complements RMSE rather than replacing it.

### Two bugs fixed, both of which made models/RNN.py unusable on soltani

- `build_trial_tensors` derived observations-per-trial from `max(observation)`,
  silently assuming 1-INDEXED data. On soltani (0..14) that gave n_obs=14 while
  every trial has 15 rows, so the `len(td) != n_obs` guard dropped EVERY trial and
  the function failed on an empty stack. Now uses the modal row count
  (index-agnostic) and raises with a clear message if nothing matches. Same class
  of 0-indexing bug as the ones fixed earlier in the activity keying and the lambda
  estimator.
- `generate_rnn_responses` emitted `observation = oi + 1` over `range(n_obs)`,
  hardcoding 1-indexing: on soltani it mislabelled every row and dropped
  observation 0. Now uses each trial's own observation labels.

Also added `cross_validated_predictions()`, stitching OUT-OF-FOLD predictions across
folds so every observation gets a prediction from a model that did not see it, with
nothing discarded. Motivation: in-sample residuals are systematically too small
because the fit absorbs noise, and on soltani that is severe -- in-sample sigma
~0.046-0.056 against 0.18 out-of-fold at k=5. In-sample sigma happens to MATCH the
qid estimate (~0.055), but only because the GRU has memorised each trial, so the
agreement is coincidental rather than validating. The function still uses the
held-out fold for early stopping, making its predictions mildly optimistic; a
nested split was judged not worth a third partition of 32 trials.

### Caveats on the soltani conclusion

4 pids, numbers only, one seed per setting. 0.0626 vs 0.0526 on n=4 is not a strong
separation and a firm decision would want ~10 pids. The ordering was consistent
across all four pids at every n_hidden, which is why the thread was stopped here
rather than powered up.

## NLL fitting infrastructure; response noise split into two mechanisms (this session)

Continuation of the NoisyRL_lambda thread. Goal: a loss function that can
actually IDENTIFY a noise parameter (RMSE cannot -- it collapses every noise
parameter to its floor, since squared error is minimised by the conditional
mean regardless of variance), and a design that lets the noise MECHANISM be
compared rather than just its presence.

### The NLL loss

`fitting.losses.compute_nll` / `nll_from_ensemble`: Gaussian NLL of the single
observed human response at each (pid, trial, observation) under the model's
simulated predictive distribution (mean + SD from n_sims independent draws). A
proper scoring rule -- the quadratic term punishes a wrong mean AND an
understated variance, log(sigma) punishes an overstated one -- so unlike RMSE it
has a genuine interior optimum. Verified directly and unconstrained (floor
0.001, i.e. effectively no floor): NLL fell from 389 at sigma_resp=0.001 to
-2.46 at the optimum (~0.04-0.05) and rose again beyond it. A real U-shape.

`models.math_models.simulate_ensemble(params, n_sims, return_index=False)`: for
a genuinely stochastic model, n_sims realisations without re-simulating from
scratch per (trial, observation) the way run() does (which would cost ~48k
pandas queries per Optuna trial at n_sims=100 on soltani -- not viable). Each
(trial, sim) is ONE forward pass; seeded `_trial_seed(sim, trial)` so
`simulate_ensemble(params, n)[i] == run({**params, "seed": i}).response`
exactly (verified to floating point, <=3.3e-16, across all four datasets).

Wired into `fitting.fit` via `--loss {rmse,nll}` and `--n_sims`. n_sims=100
verified stable (5 reseeded reps of a sigma sweep all picked the identical
argmin; n_sims=25 already agreed) at ~0.45s/eval, ~2.3 min per 300-trial fit.
NLL output files get a `_nll` suffix before `{pid}` so they can never silently
overwrite an RMSE fit of the same model_type -- the two loss scales differ (NLL
can be negative) and a silent overwrite would be a correctness hazard, not a
naming inconvenience.

### The noise mechanism was split into two, at the user's suggestion

Original NoisyRL_lambda had both sigma_state (compounds into the estimate) and
sigma_resp (i.i.d. on the report). Splitting was proposed to avoid depending on
a prior RMSE fit for the deterministic-model comparison, and to isolate the
noise MECHANISM (compounding vs i.i.d.) at EQUAL parameter count rather than one
model simply having more parameters than another:

  NoisyRL_lambda            RL_lambda + sigma_state ONLY (compounding)
  <model>_resp_noise        {Mean,LeakyIntegrator,PrimacyRecency,RL_lambda}
                            + sigma_resp ONLY (i.i.d.), via a NEW generic
                            add_noise() wrapper

`add_noise(params, n_sims, sigma_resp, return_index=False)`: calls the base
model's run() ONCE for its deterministic mean trajectory, then draws n_sims
i.i.d. Gaussian perturbations on top, clipped to [-1,1]. No per-observation loop
needed (i.i.d. noise has no sequential structure to replay), so it is cheaper
than simulate_ensemble's state-noise loop and is entirely generic -- it never
touches per-model branches in math_models.py, so it wraps any of the four
deterministic models without new code per model. Accepts either the bare base
name ("RL_lambda") or the fitting-time suffixed name ("RL_lambda_resp_noise")
identically (base_model_of() strips the suffix), since fit.py's objective
passes the suffixed name.

Verified: reduces EXACTLY to run()'s output at sigma_resp=0 (0.00e+00, all
four datasets x four base models); empirical mean/SD track the requested
values away from the +-1 clipping boundary; bare and suffixed names produce
identical output.

Registered for ALL FOUR DATASETS (carrabin, yoo, soltani_numbers,
soltani_colors) in MODEL_PARAMS, not soltani-only as NoisyRL_lambda originally
was. Floor 0.001 on every noise parameter -- TECHNICAL only, not calibrated,
since NLL was shown to find its own interior optimum unconstrained.

### Extending to carrabin/yoo surfaced two real bugs, both invisible to
### py_compile and to exercising individual branches in isolation

1. **Triplicated code, dormant until now.** Adding NoisyRL_lambda earlier this
   session used an unguarded string-replace on the anchor
   `if model_type == "RL_lambda":`, which appears once in EACH of
   `_run_carrabin`, `_run_yoo`, `_run_soltani_common` (RL_lambda is valid for
   all three). With no occurrence limit, the replace silently duplicated the
   entire branch into all three instead of the intended one. Harmless while
   unreachable (carrabin/yoo never had the model registered); activated the
   moment this request registered it there.

2. **Deduplicating the triplication introduced a WORSE bug.** Refactoring the
   three copies into one shared helper by inserting a top-level `def` string at
   the text position of the FIRST occurrence -- which sat inside
   `_run_carrabin`'s indented body -- caused the bare `def` to dedent out of
   that function. Everything textually after it, including `_run_carrabin`'s
   remaining RL_lambda/LeakyIntegrator/PrimacyRecency branches, became
   unreachable dead code swallowed into the new helper's body.
   `_run_carrabin(..., model_type="RL_lambda", ...)` would have returned None.
   Caught only by testing every carrabin branch directly, not by py_compile.
   Fixed by locating the exact corrupted text and reconstructing both pieces
   (the standalone helper, and _run_carrabin's restored tail) explicitly.

3. **simulate_ensemble labelled columns wrong for 1-indexed datasets.** It used
   a synthetic `range(n_obs)` for the observation index rather than the
   dataset's real values. Coincidentally correct for soltani (0-indexed,
   0..14) but WRONG for carrabin (1-indexed, 1..5): the carrabin Laplace-
   shrinkage formula (t = observation + 1) then received the wrong t, biasing
   the ensemble vs run() by up to 0.167 -- caught only by the direct
   equivalence check, not by any per-branch test.

All three fixed; `scripts/verify_ensemble_invariant.py` (see below) passes
clean on all combinations after the fixes.

### A fourth bug, in the FIX for the model-params update

Adding the split MODEL_PARAMS entries for all four datasets used a
find-and-replace loop that recomputed `s.find('"NoisyRL_lambda": {')` from
scratch after each insertion. The replacement text itself CONTAINS that exact
substring (it is inserting a dict literal whose key is that name), so the loop
kept re-finding the text it had just inserted and never advanced -- a genuine
infinite loop, not a slow computation (confirmed via `timeout`, exit code
124). Fixed by locating all four match positions on the ORIGINAL string before
any insertion, then replacing right-to-left so earlier offsets stay valid.

### scripts/verify_ensemble_invariant.py (new)

Not a pytest suite -- this project has none, and an earlier docstring falsely
claimed one existed in `tests/`; corrected. Run manually after touching
simulate_ensemble, add_noise, any _run_* dispatcher, or
_validate_model_dataset's allowlists, and before trusting a --loss nll fit on
a dataset/model combination not previously checked. Two check families:

- simulate_ensemble vs run(seed=i), for _STOCHASTIC_ENSEMBLE_MODELS.
- add_noise vs run(): sigma=0 exact equality; empirical mean/SD near the
  requested values AWAY FROM THE +-1 BOUNDARY (clip(mu + noise, -1, 1) is
  CORRECTLY biased near the boundary -- confirmed directly on soltani_colors'
  Mean model, which legitimately outputs exactly +-1 on 15.6% of rows: mean
  gap 0.0035 away from the boundary vs 0.0257 on boundary rows, and the
  boundary-row gap does NOT shrink with more sims, confirming bias rather than
  Monte Carlo noise -- so the check correctly excludes those rows rather than
  loosening its tolerance globally); and bare-name vs suffixed-name identity.

All checks currently pass (exit 0) locally across all dataset x model
combinations. NOT yet run on the cluster -- required before trusting any
cluster --loss nll fit, given how many of the above were invisible until
directly tested.

### Not yet done

- No jobs submitted to the cluster.
- The actual NoisyRL_lambda vs RL_lambda_resp_noise NLL comparison -- the
  scientific payoff of this whole thread -- has not been run at real
  n_trials/n_sims on any dataset.
- Response variability (qid-based) as the parallel individual-differences
  metric for carrabin/numbers/colors (yoo has no qid repeats) is planned but
  not implemented as a figure or comparison.

---

## Persistent pid registry; pull_soltani_data.py rename; pilot-4 contamination found and removed (this session)

### The pid-instability bug

`build_model_inputs.py`'s `build_from_df()` used to compute the integer
`pid` fresh on every call:

    all_pids = sorted(df["prolific_pid"].unique())
    pid_map = {p: i + 1 for i, p in enumerate(all_pids)}

That recomputes the mapping from scratch, by alphabetically sorting
whichever `prolific_pid`s are in THAT call's data, every time. The moment
the participant pool changes size, inserting new `prolific_pid` strings
into that sort generally shifts the alphabetical rank of most of the
EXISTING participants too -- not just appends new ones at the end.
Confirmed as a live, not hypothetical, bug this session: `data/
soltani_numbers.pkl` grew from 35 to 45 pids between two builds, and the
model-fit response files (still only pids 1-35, from the older build)
could no longer be safely joined on `pid` against the current human data
-- pid=5 in one file and pid=5 in the other were very likely different
real people. This is exactly what made a new `lambda_model_correlation`
figure's colors/numbers panels come back with near-zero, non-significant
correlations despite `make_response_change`'s own median curves showing
models tracking human decay closely: a MEDIAN is a population statistic,
invariant to which label is attached to which value, so it stayed correct
under the mislabeling; a per-pid CORRELATION depends entirely on the
labels being right, so it was destroyed by it.

**Fix: `utils/pid_registry.py`**, a persistent, append-only
`prolific_pid -> pid` mapping stored at `data/pid_registry.json`. Loads
the existing registry, keeps every known `prolific_pid`'s integer
unchanged, assigns new ones only to genuinely new `prolific_pid`s (sorted
for determinism, continuing from `max(existing) + 1`), then saves the
updated registry back before returning. `build_from_df()` now calls
`get_or_assign_pids()` instead of the from-scratch enumeration. Verified
with a synthetic test that deliberately inserted new IDs alphabetically
BETWEEN existing ones, confirming they don't get reshuffled. Side benefit
noted but not separately tested for: a filtered and an unfiltered build
now assign the SAME pid to the same person, which was also false before
(the old mapping depended on who else was in that specific call's batch).

The registry file contains REAL Prolific participant IDs, unlike every
canonical `data/*.pkl` file, which only ever gets the anonymized integer
-- it must never be committed, and specifically must never go through
GitHub even now that the canonical soltani files do (see below). It's
gitignored explicitly (on top of already being covered by the wholesale
`data/` rule), and syncing it to another machine (e.g. the cluster) is a
manual, non-git responsibility -- copy the one file directly.

### `build_task_backend_inputs.py` renamed to `pull_soltani_data.py`

The old name sounded like it BUILDS INPUTS FOR task_backend (i.e.
configures the task); it does the opposite -- pulls results OUT of
task_backend/Supabase. Renamed to say what it does, and its own module
docstring now explicitly walks through the pipeline's steps 1-3 (pull
from Supabase -> filter to an explicit/derived pid list -> resolve via
the persistent registry), with step 4 (rescale/anonymize/save) staying in
`build_model_inputs.py`'s `build_from_df`, called into rather than
duplicated. Every cross-reference updated (`build_model_inputs.py`,
`figure_soltani_performance.py`, `figure_soltani_variability.py`,
`CLAUDE.md`, `README.md`, `.gitignore`) -- a plain rename, no logic
change in this step.

### `--complete_pairs` was pulling in a stale pilot round

Supabase's `events` table is append-only, so an OLDER pilot round's
participants are STILL present with a perfectly genuine `'finished'` row
long after that round ended. `--complete_pairs`, as originally written,
had no date cutoff and no check that a participant's session used the
CURRENT generative parameters -- it just intersected "everyone finished in
both tasks," for all time. Confirmed directly: an actual pull returned 51
pids, 5 of which had `true_std=15` (pilot 4's fixed numbers-task std) sitting
alongside 46 with the current `true_std=10` -- exactly the failure mode the
module's own docstring already warned about for the explicit-pid-list path,
but never actually guarded against for `--complete_pairs`. Those 5 pids
(`670bd903349d5d24bc92dcb0`, `69163607e65df2b5dbe294fa`,
`697b8bbd3f4ddf0f4e102d42`, `69af34e771ce9d065c0d9d80`,
`6a11c6a18ea3cad18626f8b4`) are literally the same ones the module's own
usage example under `--pilot pilot4` already named -- independent
confirmation this was genuinely pilot 4, not some other anomaly.

Checked whether colors has an equivalent risk: it does not. Every pid's
`true_p` range (including all 5 pilot-4 pids) is identical,
`[0.1333, 0.8667]`, regardless of round -- colors' generative parameters
have apparently never changed, unlike numbers' (`NUMBERS_STD_FIXED`
history: 10 -> 15 -> 10, per `task_backend/generate_sequences.py`'s own
comment).

**Fixed at two levels, deliberately not just one:**

1. Code: `--complete_pairs` now excludes any pid whose numbers session used
   a `true_std` other than `CURRENT_NUMBERS_STD_FIXED` (10, duplicated from
   `generate_sequences.py`'s own constant rather than imported --
   `task_backend/` is a JS/Vite app with that script as its one standalone
   Python utility, not a package this analysis pipeline otherwise reaches
   into; must be kept in sync by hand if that constant ever changes again).
   Intentionally `--complete_pairs`-only, not applied to the explicit
   `--numbers_pids`/`--colors_pids` path, since that path is how a SPECIFIC
   pilot round gets rebuilt on purpose. This is a safety net for the NEXT
   time a generative parameter changes and an old round lingers in the same
   table, not what actually removed the contamination this time (see below).
2. Data: the 5 pilot-4 prolific_pids' rows were deleted directly from
   Supabase's `events` table (both tasks, all phases) after a preview
   `SELECT` confirmed the exact rows affected -- a full, complete session
   each (consent -> tutorial -> ~480 trial rows -> finished), nothing
   unexpected caught. Safe to do irreversibly because pilot 4's data was
   already separately archived (`data/task_results_pilot4.pkl`,
   `data/soltani_{numbers,colors}_pilot4.pkl`) before this session, so
   nothing unique was lost. Re-running `--complete_pairs` afterward found
   nothing to exclude (Supabase itself is clean now), producing the final
   46-pid canonical files directly.

Final state: `data/soltani_numbers.pkl`/`soltani_colors.pkl`, 46 pids
(`complete_pairs` + `non_integrator` exclusion), pid range 1-51 with
`{11, 14, 19, 34, 48}` now absent -- confirmed to be exactly the 5 removed
pilot-4 pids, and confirmed every one of the other 46 kept their pre-
existing registry pid (46 already known, 0 newly assigned).

### Canonical soltani files now tracked in git

`data/soltani_numbers.pkl`/`soltani_colors.pkl` were gitignored; now
tracked, matching `carrabin.pkl`/`yoo.pkl`'s own existing treatment --
GitHub is now the sync channel for these two files specifically between
this machine and the cluster. `data/pid_registry.json` is the deliberate
exception, per above.

**A real git gotcha surfaced doing this, worth remembering.** Per
`gitignore(5)`: "It is not possible to re-include a file if a parent
directory of that file is excluded." `data/` (trailing slash) excludes the
whole directory as a unit, so git never even looks inside it to evaluate a
later `!data/soltani_numbers.pkl` override -- for any file not ALREADY
tracked, that negation line is a silent no-op. `carrabin.pkl`/`yoo.pkl`'s
own identical-looking negation lines were never actually doing anything
either -- those files were already in the git index from before this
wholesale rule existed, and an already-tracked file is immune to
`.gitignore` regardless of any rule. Confirmed directly: `git check-ignore
-v` returns clean (exit 1, untracked-and-not-ignored) for `carrabin.pkl`
but reports the wholesale `data/` rule as still matching for the brand-new
`soltani_numbers.pkl`, and a plain `git add` refuses it with "ignored by
one of your .gitignore files." Fixed with a one-time `git add -f`; after
that, exactly like `carrabin.pkl`, both files are permanently tracked and
immune to the `data/` rule for every future `git add`, no `-f` needed
again.

