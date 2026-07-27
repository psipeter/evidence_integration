// cors.ts
// Shared CORS handling for all three progress-* Edge Functions -- the
// browser calls these directly (no server proxy), so each needs to answer
// the preflight OPTIONS request and stamp CORS headers on every response,
// including error responses (a 500 without CORS headers just looks like a
// network failure to the browser, which is exactly the kind of ambiguity
// this whole backend exists to eliminate).

export const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

/** Call first in every function. Returns a Response for OPTIONS preflight, or null to continue. */
export function handleOptions(req: Request): Response | null {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }
  return null;
}

/** Wrap every outgoing Response (success or error) so CORS headers are never forgotten. */
export function withCors(res: Response): Response {
  const headers = new Headers(res.headers);
  for (const [k, v] of Object.entries(corsHeaders)) headers.set(k, v);
  return new Response(res.body, { status: res.status, headers });
}
