/**
 * jatos-shim.js
 * No-op JATOS shim for local development.
 * In production, the real jatos object is injected by MindProbe/JATOS
 * before this script runs, so this block is never executed.
 */
if (typeof jatos === 'undefined') {
  console.warn('[dev mode] jatos object not found — using no-op shim');
  window.jatos = {
    studySessionData: {},
    submitResultData: (data, onSuccess) => {
      console.log('[dev mode] submitResultData:', data);
      if (onSuccess) onSuccess();
    },
    endStudy: () => console.log('[dev mode] endStudy'),
  };
}
