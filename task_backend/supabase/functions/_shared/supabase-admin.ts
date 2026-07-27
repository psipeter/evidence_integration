// supabase-admin.ts
// Server-side Supabase client for all three Edge Functions, using the
// SECRET key (bypasses RLS by design). This is the only thing in the whole
// backend allowed to touch the `events` table -- see TODO.md decision #1
// (locked-down tables, Edge-Functions-only access). Never import this file
// from anything that ends up in the browser bundle.
//
// Key lookup (verified against Supabase's docs, July 2026 -- the
// publishable/secret key migration is actively in-flight and the env var
// story differs across sources):
//   1. SUPABASE_SECRET_KEYS -- new-style, JSON object keyed by name, ours
//      is named "default" (the name given when the key was created).
//   2. SUPABASE_SERVICE_ROLE_KEY -- legacy var name, kept as a fallback
//      since some project configurations still populate it. Logs a
//      warning if this path is taken, so it's visible during 3h testing
//      which one actually fired.

import { createClient } from 'jsr:@supabase/supabase-js@2';

function readSecretKey(): string {
  const raw = Deno.env.get('SUPABASE_SECRET_KEYS');
  if (raw) {
    try {
      const parsed = JSON.parse(raw);
      if (parsed?.default) return parsed.default;
    } catch (e) {
      console.warn('supabase-admin: SUPABASE_SECRET_KEYS present but not parseable JSON', e);
    }
  }

  const legacy = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
  if (legacy) {
    console.warn(
      'supabase-admin: falling back to legacy SUPABASE_SERVICE_ROLE_KEY -- ' +
      'SUPABASE_SECRET_KEYS was not set or not parseable as {"default": "..."}.'
    );
    return legacy;
  }

  throw new Error(
    'No secret key available in environment (checked SUPABASE_SECRET_KEYS.default and SUPABASE_SERVICE_ROLE_KEY).'
  );
}

const SUPABASE_URL = Deno.env.get('SUPABASE_URL');
if (!SUPABASE_URL) {
  throw new Error('SUPABASE_URL not set in environment.');
}

export const supabaseAdmin = createClient(SUPABASE_URL, readSecretKey(), {
  auth: { persistSession: false },
});
