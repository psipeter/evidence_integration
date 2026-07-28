/**
 * palette.js
 * Canonical color values shared across BOTH tasks' tutorial and real-task
 * screens.
 *
 * Found during a cleanup pass: the same 3 hex values were independently
 * redeclared as "canonical" exports/constants in at least 5 different
 * files -- tutorial-text-numbers.js's GOAL_COLOR/SAMPLE_COLOR/DIST_COLOR,
 * urn-colors.js's SAMPLE_BLUE/SAMPLE_RED/DIST_COLOR,
 * draw-performance-numbers.js's MEAN_BLUE/SAMPLE_RED/ERROR_GREEN,
 * draw-performance-colors.js's local SAMPLE_BLUE/SAMPLE_RED/DIST_COLOR,
 * plus plugin-observation-numbers.js hardcoding the red inline and
 * plugin-observation-colors.js redeclaring blue/red locally rather than
 * importing its own task's existing canonical source. Consolidated here
 * as ONE underlying source; each of those files still exports/uses its
 * own semantically-named alias where that helps readability at the call
 * site (e.g. numbers's GOAL_COLOR for "the correct answer" vs colors's
 * SAMPLE_BLUE for "a blue draw") -- this only fixes there being one
 * underlying value per color, not that every file must call it the same
 * thing.
 *
 * This is a color PALETTE, not task-specific behavior or content, so
 * sharing it across both tasks doesn't couple their otherwise-
 * independent designs the way sharing e.g. box heights or copy text
 * would (those stay separate per task on purpose -- see
 * .numbers-tutorial-box/.colors-tutorial-box's own comment in style.css).
 *
 * Deliberately NOT extended to cover every incidental use of these same
 * hex values elsewhere (plugin-iti-clock.js's clock rendering,
 * create-terminate-session.js's "Too slow" message, slider-colors.js's
 * ruler bands, observation-timeout-clock.js's warning color) --
 * those are standalone UI-styling choices that happen to reuse the same
 * brand colors, not canonical named constants multiple files were each
 * separately trying to be the source of truth for. Left as direct
 * literals; revisit only if a real inconsistency (not just a style
 * preference) ever surfaces there too.
 */
export const BLUE  = '#2563eb';
export const RED   = '#ef4444';
export const GREEN = '#16a34a';
