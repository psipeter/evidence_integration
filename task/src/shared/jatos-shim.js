/**
 * jatos-shim.js
 * No-op JATOS shim for local development.
 * Posts result data to the local dev-server.js (port 3099) instead of JATOS.
 * In production, the real jatos object is injected by MindProbe before this
 * script runs, so this block is never executed.
 */
if (typeof jatos === 'undefined') {
  console.warn('[dev mode] jatos not found — using local dev shim');

  // Shared save logic -- endStudy/endStudyAndRedirect both route through
  // this, so testing either path genuinely proves data was saved locally,
  // not just logged.
  const saveData = async (data) => {
    console.log('[dev mode] saving data...');
    try {
      const resp = await fetch('http://localhost:3099/submit', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    typeof data === 'string' ? data : JSON.stringify(data),
      });
      const result = await resp.json();
      console.log('[dev mode] Saved:', result.file);
    } catch (e) {
      console.warn('[dev mode] Could not reach dev server — data logged below');
      console.log(data);
    }
  };

  window.jatos = {
    studySessionData: {},
    submitResultData: async (data, onSuccess) => {
      await saveData(data);
      if (onSuccess) onSuccess();
    },
    // Passing the results-data JSON directly as the argument here (single
    // call, no separate submitResultData) is the historically PROVEN
    // mechanism -- confirmed directly against a real downloaded MindProbe
    // result file (task/dev-results/pilot7cont.txt: a complete 24-trial
    // continuous session, non-Prolific/pilot_undefined, ending in the
    // normal on_finish "Thank you!" screen, not the early-exit path). A
    // prior revision of this shim (and the real app code) switched to a
    // two-call submitResultData-then-endStudyAndRedirect(url) pattern based
    // on an unverified documentation claim that endStudy/endStudyAndRedirect
    // don't accept data -- that claim directly contradicts the real
    // evidence above and was never independently re-confirmed, so it was
    // reverted. Keep this shim's behavior matching the single-call shape
    // exactly, since that's the one with actual proof it works.
    endStudy: async (data) => {
      await saveData(data);
      console.log('[dev mode] endStudy (no redirect requested)');
    },
    // Real jatos.js redirects to `url` after saving -- this mirrors that
    // exactly, rather than just logging, so a real navigation happens
    // even in local/dev testing. Only called for real Prolific participants
    // now (see finish-session.js) -- everyone else gets a DOM update and
    // plain endStudy(data) above, no redirect, since redirecting to a
    // same-origin confirmation page after ending the session was confirmed
    // broken on real JATOS (access-rights error).
    endStudyAndRedirect: async (url, data) => {
      await saveData(data);
      console.log('[dev mode] endStudyAndRedirect — navigating to', url);
      window.location.href = url;
    },
  };
}
