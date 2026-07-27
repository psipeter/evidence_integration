// prolific-codes.ts
// Mirrors task/src/shared/timeline-builder.js's PROLIFIC_CODES exactly --
// same real codes from the "Human Mixed Task" Prolific workspace project.
// Kept in one place server-side so the client never has to send/trust a
// completion code itself; progress-check/progress-finish are the only
// things that hand a code back.

export const PROLIFIC_CODES: Record<string, { completion: string; earlyExit: string }> = {
  numbers: { completion: 'C1CNSEMJ', earlyExit: 'C1ARJ6LO' },
  colors: { completion: 'C12FEFJU', earlyExit: 'C1L1GGHT' },
};
