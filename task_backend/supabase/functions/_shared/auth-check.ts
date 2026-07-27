// auth-check.ts
// Because the client's publishable key is not a JWT, the platform's
// verify_jwt gateway check can't validate it (it would reject the request
// before our code ever runs -- see Supabase's docs on the new key system).
// So verify_jwt is set to false for all three functions (config.toml), and
// we do our own light check here instead: the caller must present our
// own project's publishable key on the `apikey` header.
//
// This is NOT meant to be a strong secret -- the publishable key is, by
// design, safe to ship in the client bundle. It just filters out requests
// that aren't coming from our own frontend at all (or from someone poking
// the endpoint with no header), which is roughly what verify_jwt would
// have done if it understood the new key format.

function readPublishableKey(): string {
  const raw = Deno.env.get('SUPABASE_PUBLISHABLE_KEYS');
  if (raw) {
    try {
      const parsed = JSON.parse(raw);
      if (parsed?.default) return parsed.default;
    } catch (e) {
      console.warn('auth-check: SUPABASE_PUBLISHABLE_KEYS present but not parseable JSON', e);
    }
  }
  return Deno.env.get('SUPABASE_ANON_KEY') ?? '';
}

/** Returns a 401 Response if the apikey header doesn't match, else null. */
export function checkApiKey(req: Request): Response | null {
  const expected = readPublishableKey();
  const provided = req.headers.get('apikey') ?? '';
  if (!expected || provided !== expected) {
    return new Response(JSON.stringify({ error: 'unauthorized' }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' },
    });
  }
  return null;
}
