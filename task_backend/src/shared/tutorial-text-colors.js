/**
 * tutorial-text-colors.js
 * Single shared source for the colors tutorial's instructional copy (box 1/
 * 2/3 text + the probability-bar caption), imported by BOTH
 * plugin-tutorial-intro-colors.js (obs 1, progressive reveal via click) and
 * plugin-tutorial-observation-colors.js (obs 2-15, fully revealed
 * immediately).
 *
 * Those two plugins are legitimately separate FILES -- they have genuinely
 * different reveal mechanics, not just cosmetic differences -- but the TEXT
 * itself should never be two independent copies. That's exactly what
 * caused a real bug: box 3's wording was updated in the intro plugin but an
 * identical hardcoded copy in the observation plugin was silently left
 * stale, so obs 2-5 kept showing the old text. Import BOX0/BOX0B/BOX1/BOX2/
 * URN_CAPTION/RECAP_TEXT_1/RECAP_TEXT_2 from here in both places; never
 * hardcode them locally again.
 *
 * Colors are imported from urn-colors.js (the pre-existing canonical source
 * for the SVG builder) rather than redeclared here, so there's still only
 * one definition of each color, not a third copy.
 *
 * Mirrors tutorial-text-numbers.js's own text (chat history) with
 * "numbers"/"mean" swapped for "balls"/"probability" -- NOT split into a
 * separate BOX0/BOX0B pair the way numbers's BOX0 was (that split moved
 * a second sentence to sit above numbers's KDE figure specifically;
 * colors's right-column box already needed new default content of its own
 * regardless -- see BOX0B below -- so introducing that same structural
 * split wasn't necessary here).
 */
import { SAMPLE_BLUE, SAMPLE_RED, DIST_COLOR } from './urn-colors.js';

// Darker than the warning-yellow box border (#fbbf24) used elsewhere in
// this app -- that shade reads fine as a border/background but is too low-
// contrast as body text on white. Matches tutorial-text-numbers.js's
// own WARNING_YELLOW constant of the same value -- kept as its own
// definition here rather than a cross-import, consistent with how
// SAMPLE_BLUE/RED/DIST_COLOR are ALSO separately defined per task (in
// urn-colors.js vs. distribution-numbers.js) rather than a shared
// cross-task color module.
// Currently UNUSED within this file (RECAP_TEXT_1 below no longer colors
// "will not see" with it -- chat history, dropped along with that text's
// other color/wording changes, mirrors numbers's identical change).
// Left defined, not deleted, in case a future revision wants colored
// warning text again.
const WARNING_YELLOW = '#b45309';

export const BOX0 = `In this task, you'll see a <strong>sequence</strong> of balls,
  presented <em>one at a time</em>.`;

export const BOX1 = `Your <strong>goal</strong> is to estimate the
  <span style="color:${DIST_COLOR};font-weight:bold;">ratio</span> of
  <span style="color:${SAMPLE_RED};font-weight:bold;">red</span> and
  <span style="color:${SAMPLE_BLUE};font-weight:bold;">blue</span> balls in this sequence.`;

export const BOX2 = `<strong>Move</strong> the slider to show your evolving estimate of the
  <span style="color:${DIST_COLOR};font-weight:bold;">ratio</span>.`;

// Default text for the tutorial's top-right box (phase A -- see
// plugin-tutorial-observation-colors.js's rightTopBoxContent) -- a short
// sentence specifically about the figure shown there, same role
// numbers's BOX0B plays, just not built by splitting BOX0 in half (see
// module docstring for why). Originally the second sentence of the old
// combined BOX0, moved here once BOX0 itself was trimmed to just its
// first sentence (chat history) -- "probability" stays the word here
// (describing the hidden GENERATIVE parameter), even though BOX1/BOX2
// above now say "ratio" instead (describing what the participant is
// estimating FROM the observed balls) -- a deliberate distinction, not an
// inconsistency: the two boxes are talking about different things.
export const BOX0B = `Each ball is randomly colored
  <span style="color:${SAMPLE_RED};font-weight:bold;">red</span> or
  <span style="color:${SAMPLE_BLUE};font-weight:bold;">blue</span> according to a
  hidden <span style="color:${DIST_COLOR};font-weight:bold;">probability</span>.`;

export const URN_CAPTION = `This bar shows the true
  <span style="color:${DIST_COLOR};font-weight:bold;">probability</span> of
  <span style="color:${SAMPLE_BLUE};font-weight:bold;">blue</span> vs
  <span style="color:${SAMPLE_RED};font-weight:bold;">red</span>. In the
  experiment, you will only see the colored balls.`;
// ^ Currently UNUSED (chat history, mirrors DIST_CAPTION's own identical
// history in tutorial-text-numbers.js) -- the old static yellow caption
// box this was written for was replaced by the phase system's top-right
// box (which now shows BOX0B/the phase-C reminder/RECAP_TEXT_1+2 instead).
// Left defined here, not deleted, in case that text is wanted again later.

// Recap-phase text (phase D -- see plugin-tutorial-observation-colors.js's
// rightTopBoxContent) -- mirrors tutorial-text-numbers.js's identical
// RECAP_TEXT_1/RECAP_TEXT_2 constants (wording simplified this session
// per explicit direction -- dropped "probability and history"/"of the
// ratio", now identical task-agnostic wording to numbers's own
// version: "these graphics"/"your estimate"). "will not see"/"memory"
// bolded, plain (no color) -- WARNING_YELLOW is no longer used here as a
// result (see its own now-stale-looking but still-used-elsewhere-if-ever
// note above; kept defined regardless).
export const RECAP_TEXT_1 = `In the experiment, you
  <strong>will not see</strong> these graphics.`;
export const RECAP_TEXT_2 = `Use your <strong>memory</strong> of the sequence to update your
  estimate.`;

// SLIDER_REMINDER -- phase B (chat history: reintroduced this session,
// mirrors tutorial-text-numbers.js's identical constant/rationale --
// see that file's own comment and plugin-tutorial-observation-colors.js's
// docstring). "number" swapped for "ball" per this file's own
// terminology; "remembers"/"update" bolded per explicit direction (chat
// history), mirrors numbers's identical change -- plain bold, no
// color.
export const SLIDER_REMINDER = `The slider <strong>remembers</strong> your last estimate. Move it to <strong>update</strong>
  your estimate after seeing each new ball.`;

