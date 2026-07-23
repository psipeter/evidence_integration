/**
 * tutorial-text-continuous.js
 * Single shared source for the continuous tutorial's instructional copy
 * (box 1/2/3 text + the distribution caption) and its three tutorial-only
 * colors, imported by BOTH plugin-tutorial-intro-continuous.js (obs 1,
 * progressive reveal via click) and plugin-tutorial-observation-
 * continuous.js (obs 2-5, fully revealed immediately).
 *
 * Those two plugins are legitimately separate FILES -- they have genuinely
 * different reveal mechanics, not just cosmetic differences -- but the TEXT
 * itself should never be two independent copies. This mirrors
 * tutorial-text-binary.js, added after that exact duplication caused a real
 * bug there (box 3's wording updated in one plugin, silently left stale in
 * the other). Import BOX0/BOX0B/BOX1/BOX2/DIST_CAPTION from here in both
 * places; never hardcode them locally again.
 *
 * BOX0 was originally one box (two sentences). Split into BOX0/BOX0B --
 * BOX0 stays in the left-panel box sequence, BOX0B moves to sit above the
 * distribution figure in the right panel (both revealed by a single click
 * on BOX0 -- see plugin-tutorial-intro-continuous.js's onBox0).
 *
 * Unlike binary (which already had a canonical color source in
 * urn-binary.js), continuous had no single existing export for these three
 * colors -- each plugin declared its own local copy. This file is now that
 * canonical source; GOAL_COLOR/SAMPLE_COLOR are also re-exported since both
 * plugins need SAMPLE_COLOR for the centre number's color outside of any
 * box text.
 */
export const GOAL_COLOR    = '#2563eb';   // blue   — true mean
export const SAMPLE_COLOR  = '#ef4444';   // red    — current observation
export const DIST_COLOR    = '#16a34a';   // green  — distribution curve
// Darker than the warning-yellow box border (#fbbf24) used elsewhere in
// this app -- that shade reads fine as a border/background but is too low-
// contrast as body text on white. Same family, legible as text.
// Currently UNUSED within this file (RECAP_TEXT_1 below no longer colors
// "will not see" with it -- chat history, dropped along with that text's
// other color/wording changes). Left defined/exported, not deleted, in
// case a future revision wants colored warning text again.
export const WARNING_YELLOW = '#b45309';

export const BOX0 = `In this task, you'll see a <strong>sequence</strong> of
  <span style="color:${SAMPLE_COLOR};font-weight:bold;">numbers</span>, presented
  <em>one at a time</em>.`;

export const BOX0B = `Each number is
  randomly drawn from a hidden
  <span style="color:${DIST_COLOR};font-weight:bold;">distribution</span>.`;

export const BOX1 = `Your <strong>goal</strong> is to estimate the
  <span style="color:${GOAL_COLOR};font-weight:bold;">mean</span> of all numbers in the sequence.`;

export const BOX2 = `<strong>Move</strong> the slider to show your evolving estimate of the
  <span style="color:${GOAL_COLOR};font-weight:bold;">mean</span>.`;

export const DIST_CAPTION = `This graphic shows the true
  <span style="color:${GOAL_COLOR};font-weight:bold;">mean</span> and the history of
  <span style="color:${SAMPLE_COLOR};font-weight:bold;">numbers</span> for this sequence. In the
  experiment, you will only see the most recent
  <span style="color:${SAMPLE_COLOR};font-weight:bold;">number</span>.`;
// ^ Currently UNUSED (removed from both plugins' rendering/reveal chains --
// the "yellow box" -- pending a later pass that explains what users will
// actually see in the real experiment). Left defined here, not deleted, so
// that later pass can pick the text back up rather than rewrite it from
// scratch.

// RECAP_TEXT_1/2 -- originally written for a dedicated post-summary recap
// screen (plugin-tutorial-recap-continuous.js, since deleted -- chat
// history). Now used by plugin-tutorial-observation-continuous.js's
// buildHintHTML as the phase-D yellow hint-popup text instead (shown
// DURING the tutorial observations, not as a separate screen after) --
// same text, same color rules, just a different home. Wording simplified
// this session per explicit direction (dropped "probability and history"
// and "of the mean" -- just "these graphics"/"your estimate" now, task-
// agnostic wording that happens to read identically for continuous and
// binary). "will not see"/"memory" bolded, plain (no color) -- an
// earlier revision colored "will not see" WARNING_YELLOW, dropped this
// session per explicit direction (bold only, no color).
export const RECAP_TEXT_1 = `In the experiment, you
  <strong>will not see</strong> these graphics.`;
export const RECAP_TEXT_2 = `Use your <strong>memory</strong> of the sequence to update your
  estimate.`;

// SLIDER_REMINDER -- phase B (chat history: reintroduced this session,
// after being designed-then-dropped in an earlier one -- see
// plugin-tutorial-observation-continuous.js's own docstring). Reminds
// participants that the slider's position PERSISTS between observations
// (config-base.js's SLIDER_DEFAULT='last') rather than resetting, so they
// should move it to reflect their updated estimate rather than assuming
// it already reset for them. "remembers"/"update" bolded per explicit
// direction (chat history) -- plain bold, no color, matching RECAP_TEXT_1/
// 2's own "bold only" convention above rather than introducing a color.
export const SLIDER_REMINDER = `The slider <strong>remembers</strong> your last estimate. Move it to <strong>update</strong>
  your estimate after seeing each new number.`;
