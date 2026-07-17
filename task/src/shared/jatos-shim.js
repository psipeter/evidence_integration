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
    // Real jatos.js only reliably populates workerId/urlQueryParameters
    // etc. AFTER onLoad()'s callback fires -- calling back immediately here
    // (nothing async to actually wait for locally) keeps the local/test
    // code path exercising the exact same jatos.onLoad(...) wrapper
    // timeline-builder.js now uses for real JATOS, rather than a parallel
    // mechanism that could silently drift from it.
    onLoad: (callback) => { callback(); },
    // Mirrors real JATOS's own contract for jatos.urlQueryParameters:
    // "Original query string parameters of the URL that starts the study"
    // -- captured server-side before any internal redirect, so it survives
    // even once window.location.search itself no longer contains them (see
    // timeline-builder.js's own comment on this for the real-JATOS
    // mechanics). Locally there's no such redirect, so window.location.search
    // is still accurate at read time -- this just exposes the same values
    // under the property name the real app code actually reads.
    urlQueryParameters: Object.fromEntries(new URLSearchParams(window.location.search)),
    // Never set at all before this fix -- meant every local/dev run without
    // a PROLIFIC_PID fell through to `pilot_${jatos.workerId}` with
    // workerId literally undefined, producing exactly the "pilot_undefined"
    // participant IDs seen in several real historical pilot files (see
    // CLAUDE.md's "Pilot data files" note) -- a synthetic but non-undefined
    // value here closes that gap for any future local/dev run.
    workerId: 'dev_' + Date.now(),
    submitResultData: async (data, onSuccess) => {
      await saveData(data);
      if (onSuccess) onSuccess();
    },
    // The two functions finish-session.js's save-then-end-then-redirect
    // chain actually calls now (see that file's own docstring for why it
    // moved off submitResultData/endStudy/endStudyAndRedirect below --
    // those three are kept here unmodified since nothing else currently
    // references them, not because they're still load-bearing).
    //
    // appendResultData never overwrites on real JATOS -- saveData() here
    // doesn't distinguish append-vs-overwrite (the local dev-server just
    // writes one file per submission), so this shim can't itself catch a
    // future misuse that accidentally overwrites earlier real-JATOS rows.
    // That's a real, known gap between this shim and production for this
    // one function -- accepted deliberately, since the only thing that
    // matters locally is that a real network round-trip happens, not
    // byte-for-byte append semantics.
    //
    // NOTE: this can't simulate a save FAILURE either -- saveData() below
    // catches its own fetch errors and returns normally rather than
    // rethrowing, so this always resolves/calls onSuccess even if the local
    // dev server is unreachable. Deliberately left this way (discussed and
    // decided not worth fixing -- finish-session.js's save-then-end
    // gating is a production-correctness concern verified against real
    // jatos.js source, not something that needs local repro). If that
    // ever needs testing locally, saveData would need to rethrow instead
    // of swallowing its catch block.
    appendResultData: async (data, onSuccess) => {
      await saveData(data);
      if (onSuccess) onSuccess();
    },
    // Real jatos.endStudyWithoutRedirect just marks the study run finished
    // (no data argument, no redirect) -- data was already sent via the
    // appendResultData call finish-session.js chains before this. Nothing
    // to actually persist here locally; this only needs to resolve.
    endStudyWithoutRedirect: async (successful, message) => {
      console.log(`[dev mode] endStudyWithoutRedirect (successful=${successful})`);
    },
    // Real jatos.log just POSTs a string to JATOS's server-side log --
    // finish-session.js's .catch() calls this on save failure. Needed here
    // or that catch handler throws (jatos.log is not a function) before it
    // ever reaches the innerHTML line that shows the participant an error
    // screen -- console.log is enough locally, there's no server log to
    // route it to.
    log: (msg) => {
      console.log('[dev mode] jatos.log:', msg);
    },
    // Real jatos.catchAndLogErrors forwards uncaught errors/rejections and
    // console.error/warn to JATOS's own server log -- nothing to forward
    // to locally, so this is a no-op beyond confirming it was called.
    catchAndLogErrors: () => {
      console.log('[dev mode] catchAndLogErrors (no-op locally)');
    },
    // Real jatos.addJatosIds mutates/returns obj with studyId, componentId,
    // workerId, studyResultId, etc. Locally there's no real study/component/
    // batch concept, so this just tags enough (workerId + a synthetic
    // per-call studyResultId) to be visibly present and distinguishable as
    // dev-shim output, not a faithful reproduction of every real ID field.
    addJatosIds: (obj) => {
      obj.workerId = jatos.workerId;
      obj.studyResultId = 'dev_result_' + Date.now();
      return obj;
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
