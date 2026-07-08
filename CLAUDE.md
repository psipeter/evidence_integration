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
- Continuous task: Normal(mean, std) stimulus; slider response [0–100]; 24 trials × 15 obs
- Binary task: Bernoulli(p) stimulus (blue/red circle); slider response [0–100%]; 24 trials × 15 obs
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
  create-early-exit.js, plugin-observation-continuous.js, plugin-observation-binary.js,
  observation-timeout-clock.js (shared countdown-clock renderer),
  slider-continuous.js, slider-binary.js, distribution-continuous.js (continuous
  tutorial SVG builder), urn-binary.js (binary tutorial SVG builder),
  continuous-draw-animation.js / binary-draw-animation.js (tutorial bubbling
  animations, see below).
Data pipeline: JATOS JSON → task/parse_results.py → data/task_results.pkl
Consent form: verbatim IRB text from task/consent_form.txt — do not paraphrase or edit.
Pilot name field: PILOT ONLY — saves name as prolific_pid substitute.
  Remove before Prolific production (marked // PILOT ONLY in timeline-builder.js and
  # PILOT ONLY in parse_results.py).
Target: ~50–80 participants per task, within-subject (both tasks per participant).
See task/ section in README.md for full details.

Current task status (as of latest session):
- No TEST_MODE/N_TRIALS_TO_RUN toggle anymore (removed along with index-dev.html
  -- see "Task testing architecture" below). Trial count is fully implicit from
  however many trials task/sequences/{task}_sequences.json contains (currently
  24, the 6×4 pilot). BTI_MS=3000ms, DISTRACTOR_TYPE='none' (config-base.js
  DEFAULTS).
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
- 2 boxes (a redundant 3rd repeating Prolific's own timing/pay listing was
  removed), both styled as warnings (red background/border, bold "Warning:"
  label) — stacked vertically, ordered disclosure (box 2 locked with a "· · ·"
  placeholder until box 1 is revealed, mirroring the tutorial-intro pattern).
  Name/checkbox section stays behind its own "· · ·" placeholder until both
  boxes are done.
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
- Layout-shift note: reveal boxes and the name/checkbox section reserve their
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

Sequence design: 6×4 (24 trials) is the CURRENT PILOT production design, generated
  via task/generate_sequences_momentmatch.py (quota/moment-matching, isotonic
  score_mode): continuous seed=175, binary seed=198, std_fixed=15,
  mean_range=[10,90], p_range=[0.1,0.9]. This is the widened-range version --
  we tested how far the moment-matching approach could push toward [0,100]
  before the continuous side's boundary-clipping bias becomes a problem (see
  below); [10,90]/[0.1,0.9] was confirmed as the practical limit and adopted
  for the pilot. Switched from the original rejection-sampling method
  (std_fixed=20) as of the latest session -- this was an explicit choice for
  the pilot specifically, separate from the still-pending 10×4 full-experiment
  decision below.

  For the full 10×4 (40 trials) experiment, three generation methods now exist
  and are under active evaluation — see "Sequence generation methods" below.
  The full-experiment design is NOT yet finalized: pending PI consultation on
  which method (i.i.d. vs moment-matched) to promote to production.

Single master copy in task/sequences/{task}_sequences.{pkl,json}.
task/src/{task}/config.js imports directly from task/sequences/ — no copy step needed.

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
midpoint (50/0.5), subject to a visual-spread check on the first 5 values
(>=2 above/below the mean for continuous; >=2 of each color for binary),
falling through to the next-closest candidate if the top match fails the
spread check. Replaced hand-picked literals (`tutorialValues = [48, 75, 38,
82, 57]`, `TUTORIAL_STD = 20`) that silently drifted out of sync the moment
sequences.json's actual std_fixed changed (this happened once already: the
tutorial kept showing std=20 after the pilot moved to std=15). Never
hand-pick a new tutorial example again — if the pedagogical example ever
needs different qualities, change pickTutorialExample's selection logic,
not the values.

**Participant-data columns** (parse_results.py, build-trial-timeline.js):
only genuinely participant-generated fields are recorded/saved —
prolific_pid, task, trial, observation, value, response, timed_out, rt,
time_elapsed. true_mean, true_std, true_p, qid, prefix_length, iti_ms,
iti_condition are intentionally NOT duplicated into the raw export or the
final pickle — all are fully determined by (task, trial) alone (trial order
is identical for every participant), so they're recovered via a join
against sequences.json when analysis needs them. This was found VIOLATED
once already (build-trial-timeline.js's 'iti' screen was duplicating
iti_ms/iti_condition/distractor_type into every recorded ITI row, despite
the file's own docstring saying otherwise) and fixed — if a new screen/
plugin's `data:{...}` block is ever added, check it against this principle
before assuming it's fine.

### Sequence generation methods (task/)

Three separate scripts, kept deliberately distinct rather than one script with
a tunable knob — see rationale below.

**task/generate_sequences.py** (original; being phased out for new work):
  Draws i.i.d. observations, then rejects and redraws whole blocks until the
  realized sample statistics fall within k × SE of the true parameter
  (rejection sampling). Two problems discovered during the 10×4 push:
  1. The joint constraint (ALL qids must pass simultaneously in one draw)
     scales very badly with n_unique_sequences — going from 6 to 10 qids
     collapsed the binary structural pass rate from ~12% to ~0% at k=0.5,
     and to only ~6% even at the old k=0.7.
  2. At extreme means (e.g. mean=10 or 90 with std=15), the [0,100] bound
     truncates the achievable std so far below the nominal value that NO
     amount of resampling can pass a tight k — a structural mismatch between
     target and bound, not a sampling problem.
  **Key finding**: k-constrained rejection sampling and exact quota sampling
  turned out to be the SAME underlying object at different points on one
  continuum — i.i.d. sampling conditioned on the final composition falling
  within k × SE of the target. Exact enumeration (n=11, p=0.5) showed the
  variance of the LAST observation's predictability at the production k=0.7
  is already ~10x smaller than true i.i.d., ~90% of the way to quota's hard
  zero. There is no way to tighten k for smoothness without buying into
  finite-population predictability — they are the same lever.
  Still used only by whichever pilot/production sequences were last generated
  with it -- currently NOT the active pilot method (the 6×4 pilot switched to
  moment-matching, see "Sequence design" above). Not recommended for new work.

**task/generate_sequences_iid.py** (pure i.i.d. branch):
  Genuinely unconstrained sampling — no k, no plausibility gate, no rejection
  loop, and deliberately NO seed search or best-of-N ranking either (any
  outcome-dependent seed selection is itself a form of conditioning, which
  would pull back toward the same finite-population structure this branch
  exists to avoid). Single draw, save, done. Matches the closest published
  precedent (Nassar/Behrens/Glaze-style predictive-inference tasks draw
  outcomes directly from the generative distribution with no correction).
  `--report` gives a diagnostic (achieved vs target moments) that never
  feeds back into generation.

**task/generate_sequences_momentmatch.py** (quota / moment-matching branch):
  Constructs each block (prefix/suffix) to hit the target sample mean/std
  (continuous, via iterative rescale+clip) or exact blue/red quota (binary),
  then randomizes order/realization. No rejection loop — scales to any
  n_unique_sequences for free, and resolves the mean=10/90+std=15 truncation
  problem directly (verified: achieved std within ~0.5 of nominal even at
  the range edges, vs ~4.5 off under rejection sampling).
  Literature check: no support found for exact quota matching as a
  behaviorally-neutral stimulus-generation choice in the probability-learning/
  evidence-integration literature — every precedent found (gambler's-fallacy,
  probability-matching studies) uses this kind of composition constraint as
  a deliberate, studied manipulation, not a neutral background choice. Real
  methodological tradeoff, not free.
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
      preferred method: at 10×4 (mean_range=[20,80], std_fixed=15,
      p_range=[0.2,0.8], n_tries=300) found residuals ~3.1e-7 (binary, seed=68)
      and ~2.0e-7 (continuous, seed=245) — curves visually indistinguishable
      from perfectly smooth decay.

Current best candidates for the 10×4 full experiment (NOT yet promoted to
production filenames — saved as task/sequences/{task}_momentmatch_sequences.*):
  continuous: seed=245 (momentmatch, isotonic, mean_range=[20,80], std_fixed=15)
  binary:     seed=68  (momentmatch, isotonic, p_range=[0.2,0.8])
std_fixed=15 is the new default for continuous going forward (down from 20
in the 6×4 pilot) — confirmed to resolve within the achievable range via
moment-matching; NOT confirmed compatible with the original rejection-sampling
script at k<0.7 (truncation issues near mean_range edges, see above).

How far can moment-matching push mean_range/p_range toward [0,100]/[0,1]?
Tested empirically (moment_match_continuous/binary directly, 300 draws per
target, std_fixed=15, n=15 obs):

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
shrinks it. [10,90] (bias +0.95, std 14.23) was judged an acceptable tradeoff
and adopted for the 6×4 pilot; below ~mean=10 the bias becomes large enough
to matter. Binary has NO equivalent problem -- quota is exact for any p in
(0,1), the only limitation is 1/n rounding granularity (e.g. p=0.02 at n=15
rounds to an exact 0, same as p=0.00 -- not a degradation, just coarseness),
so p_range can go far more extreme than mean_range if ever needed.

Wider ranges (mean_range=[10,90], p_range=[0.1,0.9]) were tested and found
to fail structurally under rejection sampling (truncation) but work fine
under moment-matching — this is now the adopted 6×4 pilot range (see
"Sequence design" above).

### Open items (as of latest session)

- **Pilot decision made**: 6×4 pilot now uses moment-matched sequences
  (continuous seed=175, mean_range=[10,90]; binary seed=198, p_range=[0.1,0.9];
  std_fixed=15), replacing the previous rejection-sampling pilot sequences.
- **PI decision pending**: i.i.d. vs moment-matched for the 10×4 production
  sequences (see literature-precedent tradeoff above) is still open — the
  pilot decision above does not resolve this; nothing for the 10×4 full
  experiment is promoted to production filenames yet.
- **Summary screens** (task/src/shared/plugin-trial-summary-{continuous,binary}.js):
  running-mean overlay requested (matching inspect_sequences.py's --gt_mode
  running_mean) but explicitly deferred as a separate, bigger UI change —
  not started.

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
- node test_consent_name.mjs — verifies pilot name saved to jsPsych data (fast, ~5s)
- node test_browser.mjs      — Playwright E2E tests across Chromium, Firefox, and
                               WebKit, both tasks (30 scenarios total, full tutorial
                               included). SLOW (~2-3 min, longer than before since
                               the tutorial now runs in full) — only run after big
                               task/ changes or when explicitly asked, not after every
                               small edit. Runs fine via shell tool (spawns the real
                               `vite` dev server against index-test.html, drives it via
                               URL params -- see "Task testing architecture" above --
                               detects screen transitions via body[data-screen="..."]
                               set by on_trial_start in timeline-builder.js, kills the
                               dev server by process group on exit).
                               Run a subset: node test_browser.mjs --task=binary --browser=chromium
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
- Data saved via jatos.endStudyAndRedirect(prolificURL, jsPsych.data.get().json())
- Non-completions: request return on Prolific (not rejection); slot reopens
- Abandoned runs stay as DATA_RETRIEVED in MindProbe — filter by FINISHED state

Pre-deployment checklist (before Prolific production):
  - Confirm task/sequences/{continuous,binary}_sequences.json holds the intended
    final trial count/parameters (no more TEST_MODE/N_TRIALS_TO_RUN switch to check)
  - Remove PILOT ONLY name field (timeline-builder.js + parse_results.py)
  - Fill IRB Protocol Number ([Protocol Number] in timeline-builder.js)
  - Replace EARLY_EXIT_CODE='EARLYEXIT' with real Prolific partial-payment code
  - Obtain binary task completion code (continuous: C3W3TF1O)
  - Fund Prolific wallet; confirm payment rate with PI
  - Run: node test_browser.mjs (in terminal)

Pilot data files:
  data/task_results.pkl          — pilot 3 (40 trials, pilot_undefined)
  data/task_results_pilot4.pkl   — pilot 4 (20 trials, pilot_undefined)
  data/task_results_pilot5.pkl   — pilot 5 (24 trials, pilot_undefined, old jzip)
  dev-results/test6bin.txt etc.  — pilot 6 test (2 trials, name='peter' ✓)

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

See "Sequence generation methods" above for the three scripts. Quick reference:

    # Original rejection sampling (currently unused for the pilot; kept for comparison)
    venv/bin/python task/generate_sequences.py --task continuous --n_unique_sequences 6 --n_repeats 4 --n_tries 500
    venv/bin/python task/generate_sequences.py --task binary    --n_unique_sequences 6 --n_repeats 4 --n_tries 500

    # Pure i.i.d. (single draw, no search)
    venv/bin/python task/generate_sequences_iid.py --task both --seed 0 \
        --n_unique_sequences 10 --n_repeats 4 --mean_range 20 80 --std_fixed 15 --p_range 0.2 0.8

    # Moment-matched / quota (isotonic seed search, default score_mode) --
    # this is what generated the CURRENT 6x4 pilot (seed=175/198, mean_range=[10,90],
    # p_range=[0.1,0.9], n_unique_sequences=6)
    venv/bin/python task/generate_sequences_momentmatch.py --task both --n_tries 300 \
        --n_unique_sequences 6 --n_repeats 4 --mean_range 10 90 --std_fixed 15 --p_range 0.1 0.9 \
        --rl_alpha_0 1.0 --rl_lambda 0.5

    # WARNING (all three scripts): --task both overwrites BOTH sequence files.
    # After a search, regenerate whichever task you want to keep with --seed N.

### Inspect sequences (scripts/inspect_sequences.py)

    venv/bin/python scripts/inspect_sequences.py --alpha_0 1.0 --rl_lambda 0.5 --skip_nef

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
  output to the production {task}_sequences.{pkl,json} filenames without
  explicit go-ahead — PI consultation on i.i.d. vs moment-matched is pending
- Do not add a seed search / best-of-N ranking to generate_sequences_iid.py —
  this was deliberately removed; any outcome-dependent seed selection
  reintroduces the conditioning this branch exists to avoid
- Do not reintroduce dev-only override knobs (testMode, nTrialsDefault,
  trialItiMs, showTutorial) into buildAndRun/timeline-builder.js/config-base.js
  — these were deliberately removed along with index-dev.html; any test-only
  need for different config values belongs in src/test-harness.js building a
  modified config object, never inside the production code path itself
