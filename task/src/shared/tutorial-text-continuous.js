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
 * the other). Import BOX0/BOX1/BOX2/DIST_CAPTION from here in both places;
 * never hardcode them locally again.
 *
 * Unlike binary (which already had a canonical color source in
 * urn-binary.js), continuous had no single existing export for these three
 * colors -- each plugin declared its own local copy. This file is now that
 * canonical source; GOAL_COLOR/SAMPLE_COLOR are also re-exported since both
 * plugins need SAMPLE_COLOR for the centre number's color outside of any
 * box text.
 */
export const GOAL_COLOR   = '#2563eb';   // blue  — true mean
export const SAMPLE_COLOR = '#ef4444';   // red   — current observation
export const DIST_COLOR   = '#16a34a';   // green — distribution curve

export const BOX0 = `In this task, you'll see a <strong>sequence</strong> of
  <span style="color:${SAMPLE_COLOR};font-weight:bold;">numbers</span>, presented
  <em>one at a time</em>. Each number is
  randomly drawn from a hidden
  <span style="color:${DIST_COLOR};font-weight:bold;">distribution</span>.`;

export const BOX1 = `Your <strong>goal</strong> is to estimate that distribution's
  <span style="color:${GOAL_COLOR};font-weight:bold;">mean</span>, based on
  all the numbers you've seen in this sequence.`;

export const BOX2 = `<strong>Move</strong> the slider to show your evolving estimate of the
  overall <span style="color:${GOAL_COLOR};font-weight:bold;">mean</span> for
  this sequence.`;

export const DIST_CAPTION = `This curve shows the true
  <span style="color:${DIST_COLOR};font-weight:bold;">distribution</span>. In the
  experiment, you will only see the individual
  <span style="color:${SAMPLE_COLOR};font-weight:bold;">numbers</span>.`;
