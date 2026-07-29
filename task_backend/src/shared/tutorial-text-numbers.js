/**
 * tutorial-text-numbers.js
 * Single shared source for the numbers tutorial's instructional copy
 * (box 0/0b/1/2 text) and its shared colors, imported by BOTH
 * plugin-tutorial-intro-numbers.js (obs 1, progressive reveal via click)
 * and plugin-tutorial-observation-numbers.js (obs 2-15, fully revealed
 * immediately).
 *
 * Those two plugins are legitimately separate FILES -- they have genuinely
 * different reveal mechanics, not just cosmetic differences -- but the TEXT
 * itself should never be two independent copies. This mirrors
 * tutorial-text-colors.js, added after that exact duplication caused a real
 * bug there (box 3's wording updated in one plugin, silently left stale in
 * the other). Import BOX0/BOX0B/BOX1/BOX2 from here in both places; never
 * hardcode them locally again.
 *
 * BOX0 was originally one box (two sentences). Split into BOX0/BOX0B --
 * BOX0 stays in the left-panel box sequence, BOX0B moves to sit above the
 * right panel's correct-answer/tracker boxes (both revealed by a single
 * click on BOX0 -- see plugin-tutorial-intro-numbers.js's onBox0).
 *
 * Colors defined here are the canonical source for this task's own
 * semantic naming -- the underlying hex values themselves come from
 * palette.js (shared across both tasks; see that file's own docstring
 * for why a color palette is fine to share even though most other things
 * in this codebase stay task-specific). SAMPLE_COLOR is also used
 * outside any box text, for the centre number's own color.
 */
import { BLUE, RED } from './palette.js';

export const GOAL_COLOR    = BLUE;    // the correct answer (running mean)
export const SAMPLE_COLOR  = RED;     // current observation

export const BOX0 = `In this task, you'll see a <strong>sequence</strong> of
  <span style="color:${SAMPLE_COLOR};font-weight:bold;">numbers</span>, presented
  <em>one at a time</em>.`;

export const BOX0B = `Each number is
  randomly drawn from a hidden
  distribution.`;

export const BOX1 = `Your <strong>goal</strong> is to estimate the
  <span style="color:${GOAL_COLOR};font-weight:bold;">average</span> of all numbers in the sequence.`;

export const BOX2 = `<strong>Move</strong> the slider to show your evolving estimate of the
  <span style="color:${GOAL_COLOR};font-weight:bold;">average</span>.`;

// RECAP_TEXT_1/2 -- originally written for a dedicated post-summary recap
// screen (plugin-tutorial-recap-numbers.js, since deleted). Now used by
// plugin-tutorial-observation-numbers.js's rightTopBoxContent as the
// phase-D warning text instead (shown DURING the tutorial observations,
// not as a separate screen after) -- same text, same color rules, just a
// different home. Task-agnostic wording ("these graphics"/"your
// estimate") that reads identically for numbers and colors.
// "will not see"/"memory" bolded, plain (no color).
export const RECAP_TEXT_1 = `In the experiment, you
  <strong>will not see</strong> these graphics.`;
export const RECAP_TEXT_2 = `Use your <strong>memory</strong> of the sequence to update your
  estimate.`;

// SLIDER_REMINDER -- phase B. Reminds participants that the slider's
// position PERSISTS between observations (config-base.js's
// SLIDER_DEFAULT='last') rather than resetting, so they should move it
// to reflect their updated estimate rather than assuming it already
// reset for them. "remembers"/"update" bolded, matching RECAP_TEXT_1/2's
// own "bold only" convention above rather than introducing a color.
export const SLIDER_REMINDER = `The slider <strong>remembers</strong> your last estimate. Move it to <strong>update</strong>
  your estimate after seeing each new number.`;
