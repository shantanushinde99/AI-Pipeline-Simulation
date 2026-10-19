import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

export type Policy = "fixed" | "exponential" | "exp-jitter";
export type RLConfig = {
  clients: number;
  rate: number; // tokens per second
  burst: number; // bucket capacity
  policy: Policy;
  baseMs: number;
  retryAfter: boolean; // client honours the Retry-After header
  maxAttempts: number;
  seed: number;
};
export const DEFAULT_RL: RLConfig = { clients: 24, rate: 5, burst: 5, policy: "fixed", baseMs: 500, retryAfter: false, maxAttempts: 6, seed: 11 };
export const POLICY_LABEL: Record<Policy, string> = { fixed: "Fixed delay, no jitter", exponential: "Exponential backoff, no jitter", "exp-jitter": "Exponential backoff + full jitter" };

/** Longest stretch of simulated time shown step by step; the summary still counts everything. */
export const MAX_HORIZON_MS = 16000;
/** Show whole seconds until the last attempt has landed (at least 2 s, at most MAX_HORIZON_MS). */
export const horizonOf = (attempts: Attempt[]) =>
  Math.min(MAX_HORIZON_MS, Math.max(2000, Math.ceil(((attempts.at(-1)?.t ?? 0) + 300) / 1000) * 1000));
/** Step length: short runs get finer steps so each retry wave has its own step; at most 8 window steps. */
export const windowOf = (horizon: number) => (horizon <= 4000 ? 500 : horizon <= 8000 ? 1000 : 2000);
export const BUCKET_MS = 250;
export const MAX_DELAY_MS = 8000;
/** Every client (re)connects in the first 100 ms, as after an outage or a deploy. */
export const START_SPREAD_MS = 100;

export function sanitize(c: RLConfig): RLConfig {
  return {
    ...c,
    clients: Math.round(clamp(c.clients, 1, 40)),
    rate: clamp(c.rate, 0.5, 50),
    burst: Math.round(clamp(c.burst, 1, 20)),
    baseMs: clamp(c.baseMs, 50, 4000),
    maxAttempts: Math.round(clamp(c.maxAttempts, 1, 10)),
  };
}

/** Delay before retry k (k = 1 for the first retry). `u` is a uniform draw used only by the jittered policy. */
export function backoffDelay(policy: Policy, baseMs: number, k: number, u: number): number {
  const exp = Math.min(MAX_DELAY_MS, baseMs * 2 ** (k - 1));
  return policy === "fixed" ? baseMs : policy === "exponential" ? exp : u * exp;
}

/** Tokens in a bucket that held `tokens` at time `last`, refilling at `rate`/s up to `burst`. */
export const refill = (tokens: number, last: number, t: number, rate: number, burst: number) => Math.min(burst, tokens + (rate * (t - last)) / 1000);

export type Attempt = {
  id: number; client: number; n: number; t: number; ok: boolean;
  tokensBefore: number; tokensAfter: number;
  retryAfterMs?: number; backoffMs?: number; nextAt?: number; gaveUp?: boolean;
};

export function runRateLimit(cfgIn: RLConfig) {
  const cfg = sanitize(cfgIn);
  const r = rng(cfg.seed);
  const pending = Array.from({ length: cfg.clients }, (_, client) => ({ client, n: 1, t: r.next() * START_SPREAD_MS }));
  const attempts: Attempt[] = [];
  let tokens = cfg.burst, last = 0;
  while (pending.length) {
    let i = 0;
    for (let j = 1; j < pending.length; j++) if (pending[j].t < pending[i].t) i = j;
    const { client, n, t } = pending.splice(i, 1)[0];
    tokens = refill(tokens, last, t, cfg.rate, cfg.burst);
    last = t;
    const before = tokens;
    const a: Attempt = { id: attempts.length, client, n, t, ok: tokens >= 1 - 1e-9, tokensBefore: before, tokensAfter: before };
    if (a.ok) {
      tokens = Math.max(0, tokens - 1);
      a.tokensAfter = tokens;
    } else {
      // Server tells the client when the next token will exist (rounded up to 10 ms).
      a.retryAfterMs = Math.ceil((((1 - tokens) / cfg.rate) * 1000) / 10) * 10;
      if (n < cfg.maxAttempts) {
        const backoff = backoffDelay(cfg.policy, cfg.baseMs, n, r.next());
        a.backoffMs = backoff;
        a.nextAt = t + (cfg.retryAfter ? Math.max(backoff, a.retryAfterMs) : backoff);
        pending.push({ client, n: n + 1, t: a.nextAt });
      } else a.gaveUp = true;
    }
    attempts.push(a);
  }
  return { cfg, attempts };
}

/** Tokens in the bucket at time t, replayed from the attempt log. */
export function tokensAt(attempts: Attempt[], cfg: RLConfig, t: number): number {
  let tokens = cfg.burst, last = 0;
  for (const a of attempts) {
    if (a.t > t) break;
    tokens = a.tokensAfter;
    last = a.t;
  }
  return refill(tokens, last, t, cfg.rate, cfg.burst);
}

export function summarize(attempts: Attempt[], cfg: RLConfig, until = Number.POSITIVE_INFINITY) {
  const seen = attempts.filter((a) => a.t <= until);
  const ok = seen.filter((a) => a.ok);
  const buckets = new Map<number, number>();
  for (const a of seen) buckets.set(Math.floor(a.t / BUCKET_MS), (buckets.get(Math.floor(a.t / BUCKET_MS)) ?? 0) + 1);
  // Retry spike: the most retries (not first attempts) landing inside any 100 ms window.
  const retries = seen.filter((a) => a.n > 1).map((a) => a.t);
  let retrySpike = 0;
  for (let i = 0, j = 0; i < retries.length; i++) {
    while (retries[i] - retries[j] >= 100) j++;
    retrySpike = Math.max(retrySpike, i - j + 1);
  }
  // Early retries: arrived before the Retry-After the same client was given on its previous 429.
  const prev = new Map<number, Attempt>();
  let early = 0;
  for (const a of seen) {
    const p = prev.get(a.client);
    if (p?.retryAfterMs !== undefined && a.t - p.t < p.retryAfterMs) early++;
    prev.set(a.client, a);
  }
  return {
    attempts: seen.length,
    rejected: seen.length - ok.length,
    rate429: seen.length ? (seen.length - ok.length) / seen.length : 0,
    succeeded: ok.length,
    gaveUp: seen.filter((a) => a.gaveUp).length,
    doneAt: ok.length === cfg.clients ? Math.max(...ok.map((a) => a.t)) : Number.NaN,
    peakPerBucket: Math.max(0, ...buckets.values()),
    retrySpike,
    early,
    wasted: seen.filter((a) => !a.ok).length / Math.max(1, ok.length),
  };
}

/** Attempts per BUCKET_MS over [0, horizon): [accepted, rejected]. */
export function perBucket(attempts: Attempt[], horizon = horizonOf(attempts)) {
  return Array.from({ length: horizon / BUCKET_MS }, (_, i) => {
    const inB = attempts.filter((a) => a.t >= i * BUCKET_MS && a.t < (i + 1) * BUCKET_MS);
    const ok = inB.filter((a) => a.ok).length;
    return { t0: i * BUCKET_MS, t1: (i + 1) * BUCKET_MS, ok, rejected: inB.length - ok };
  });
}

export type RLState = { kind: "window" | "summary"; t0: number; t1: number; attempts: number; rejected: number; maxRetryAfter: number };

export function simulate(cfgIn: RLConfig) {
  const { cfg, attempts } = runRateLimit(cfgIn);
  const sum = summarize(attempts, cfg);
  const steps: SimulationStep<RLState>[] = [];
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "ratelimit", message, severity, tex });
  const horizon = horizonOf(attempts);
  const windowMs = windowOf(horizon);
  for (let t0 = 0; t0 < horizon; t0 += windowMs) {
    const t1 = t0 + windowMs;
    const inW = attempts.filter((a) => a.t >= t0 && a.t < t1);
    const rej = inW.filter((a) => !a.ok);
    const events: SimulationEvent[] = [ev(`w${t0}`, `${inW.length} attempts: ${inW.length - rej.length} accepted, ${rej.length} got 429.`, rej.length > inW.length / 2 ? "warning" : undefined)];
    const firstRej = rej[0];
    if (firstRej?.backoffMs !== undefined) {
      events.push(ev(`r${firstRej.id}`, `client ${firstRej.client + 1}, attempt ${firstRej.n}: 429, Retry-After ${firstRej.retryAfterMs} ms, backoff ${firstRej.backoffMs.toFixed(0)} ms → retries at ${firstRej.nextAt!.toFixed(0)} ms`, undefined,
        `\\text{delay}_{${firstRej.n}} = ${backoffTex(cfg, firstRej)}`));
    }
    const gave = inW.filter((a) => a.gaveUp);
    if (gave.length) events.push(ev(`g${t0}`, `${gave.length} client(s) gave up after ${cfg.maxAttempts} attempts.`, "error"));
    steps.push({
      id: `w${t0}`,
      title: `${t0 / 1000}–${t1 / 1000} s`,
      description: `Simulated ${t0 / 1000}–${t1 / 1000} s. The bucket refills ${cfg.rate} tokens/s up to ${cfg.burst}. Each attempt takes one token or gets HTTP 429.`,
      state: { kind: "window", t0, t1, attempts: inW.length, rejected: rej.length, maxRetryAfter: Math.max(0, ...rej.map((a) => a.retryAfterMs ?? 0)) },
      events,
    });
  }
  steps.push({
    id: "summary",
    title: "Summary",
    description: `${sum.succeeded} of ${cfg.clients} clients succeeded using ${sum.attempts} attempts (${(sum.rate429 * 100).toFixed(0)}% got 429). Worst retry spike: ${sum.retrySpike} retries within 100 ms.`,
    state: { kind: "summary", t0: horizon, t1: horizon, attempts: sum.attempts, rejected: sum.rejected, maxRetryAfter: 0 },
    events: [
      ev("s1", `${sum.rejected} wasted attempts (429) for ${sum.succeeded} successes: ${sum.wasted.toFixed(2)} wasted per success.`, sum.rate429 > 0.5 ? "warning" : undefined),
      ...(sum.gaveUp ? [ev("s2", `${sum.gaveUp} clients gave up.`, "error" as const)] : []),
    ],
  });
  return { cfg, attempts, sum, steps, horizon, windowMs };
}

export function backoffTex(cfg: RLConfig, a: Attempt): string {
  const k = a.n;
  const exp = Math.min(MAX_DELAY_MS, cfg.baseMs * 2 ** (k - 1));
  const core = cfg.policy === "fixed" ? `${cfg.baseMs}` : cfg.policy === "exponential" ? `${cfg.baseMs} \\cdot 2^{${k - 1}} = ${exp}` : `U(0,\\ ${cfg.baseMs} \\cdot 2^{${k - 1}}) = ${(a.backoffMs ?? 0).toFixed(0)}`;
  return cfg.retryAfter ? `\\max(${core},\\ ${a.retryAfterMs}) = ${(a.nextAt! - a.t).toFixed(0)}\\,\\text{ms}` : `${core}\\,\\text{ms}`;
}
