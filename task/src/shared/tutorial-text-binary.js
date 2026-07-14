/**
 * tutorial-text-binary.js
 * Single shared source for the binary tutorial's instructional copy (box 1/
 * 2/3 text + the probability-bar caption), imported by BOTH
 * plugin-tutorial-intro-binary.js (obs 1, progressive reveal via click) and
 * plugin-tutorial-observation-binary.js (obs 2-5, fully revealed
 * immediately).
 *
 * Those two plugins are legitimately separate FILES -- they have genuinely
 * different reveal mechanics, not just cosmetic differences -- but the TEXT
 * itself should never be two independent copies. That's exactly what
 * caused a real bug: box 3's wording was updated in the intro plugin but an
 * identical hardcoded copy in the observation plugin was silently left
 * stale, so obs 2-5 kept showing the old text. Import BOX0/BOX1/BOX2/
 * URN_CAPTION from here in both places; never hardcode them locally again.
 *
 * Colors are imported from urn-binary.js (the pre-existing canonical source
 * for the SVG builder) rather than redeclared here, so there's still only
 * one definition of each color, not a third copy.
 */
import { SAMPLE_BLUE, SAMPLE_RED, DIST_COLOR } from './urn-binary.js';

export const BOX0 = `In this task, you'll see a <strong>sequence</strong> of balls,
  presented <em>one at a time</em>. Each ball is
  randomly colored
  <span style="color:${SAMPLE_RED};font-weight:bold;">red</span> or
  <span style="color:${SAMPLE_BLUE};font-weight:bold;">blue</span> according to a
  hidden <span style="color:${DIST_COLOR};font-weight:bold;">probability</span>.`;

export const BOX1 = `Your <strong>goal</strong> is to estimate that
  <span style="color:${DIST_COLOR};font-weight:bold;">probability</span>, based on
  all the balls you've seen in this sequence.`;

export const BOX2 = `<strong>Move</strong> the slider to show your evolving estimate of the
  overall <span style="color:${DIST_COLOR};font-weight:bold;">probability</span> for
  this sequence.`;

export const URN_CAPTION = `This bar shows the true
  <span style="color:${DIST_COLOR};font-weight:bold;">probability</span> of
  <span style="color:${SAMPLE_BLUE};font-weight:bold;">blue</span> vs
  <span style="color:${SAMPLE_RED};font-weight:bold;">red</span>. In the
  experiment, you will only see the colored balls.`;
