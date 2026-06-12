/**
 * jatos-shim.js
 * No-op JATOS shim for local development.
 * Posts result data to the local dev-server.js (port 3099) instead of JATOS.
 * In production, the real jatos object is injected by MindProbe before this
 * script runs, so this block is never executed.
 */
if (typeof jatos === 'undefined') {
  console.warn('[dev mode] jatos not found — using local dev shim');
  window.jatos = {
    studySessionData: {},
    submitResultData: async (data, onSuccess) => {
      console.log('[dev mode] submitResultData — posting to local server...');
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
      if (onSuccess) onSuccess();
    },
    endStudy: () => console.log('[dev mode] endStudy'),
  };
}
