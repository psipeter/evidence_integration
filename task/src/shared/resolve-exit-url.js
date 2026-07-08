/**
 * resolve-exit-url.js
 * Single source of truth for "where does jatos.endStudyAndRedirect send the
 * participant" -- both timeline-builder.js's on_finish and
 * create-early-exit.js's earlyExit() need this exact same isProlific ?
 * <prolific completion URL> : <local exit-complete.html> decision, and
 * previously each had its own independently-written copy of the URL
 * template. Extracted so there's exactly one place that knows the URL
 * shape; if it's ever wrong, it's wrong in one place, not two that can
 * silently drift apart.
 */

/**
 * @param {boolean} isProlific
 * @param {string}  prolificCode  Prolific completion code for this
 *                                 task/exit-reason (unused when !isProlific)
 * @returns {string} URL to pass as jatos.endStudyAndRedirect's first argument
 */
export function resolveExitUrl(isProlific, prolificCode) {
  return isProlific
    ? `https://app.prolific.com/submissions/complete?cc=${prolificCode}`
    : 'exit-complete.html';
}
