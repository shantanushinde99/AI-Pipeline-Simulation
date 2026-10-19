import { rng } from "../lib/random";

/** Small deterministic service simulations for the production Failure Lab scenarios. Times are simulated ms. */

export const percentile = (xs: number[], q: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.ceil(q * s.length) - 1)] : 0;
};

// ---------------- request latency ----------------
export type LatencyConfig = { rerankCandidates: number; rerankBatch: number; llmProvider: "primary" | "secondary"; pods: number; trafficFactor: number };
export const LATENCY_STAGES = ["gateway", "embed", "retrieve", "rerank", "llm", "post"] as const;
export type Stage = (typeof LATENCY_STAGES)[number];
export type Req = { id: number; hour: number; rps: number; cacheHit: boolean; batches: number[]; ms: Record<Stage, number>; total: number };

/** One simulated day of /ask requests. Each stage's latency is drawn from its own model; the reranker cost scales with candidates. */
export function simulateLatency(cfg: LatencyConfig, seed: number, n = 720): Req[] {
  const r = rng(seed);
  const jitter = (s: number) => Math.exp(s * r.gaussian());
  return Array.from({ length: n }, (_, id) => {
    const hour = Math.floor((id / n) * 24);
    const rps = Math.round(cfg.trafficFactor * (120 + 140 * Math.max(0, Math.sin(((hour - 6) / 24) * 2 * Math.PI))));
    const util = Math.min(0.95, rps / (cfg.pods * 60));
    const cacheHit = r.next() < 0.62;
    const batches: number[] = [];
    for (let left = cfg.rerankCandidates; left > 0; left -= cfg.rerankBatch) batches.push((35 + 5.5 * Math.min(cfg.rerankBatch, left)) * jitter(0.12));
    const ms: Record<Stage, number> = {
      gateway: 3 + 8 * util ** 4 * jitter(0.3),
      embed: cacheHit ? 1 : 22 * jitter(0.25),
      retrieve: 28 * jitter(0.2),
      rerank: batches.reduce((s, b) => s + b, 0),
      llm: (cfg.llmProvider === "primary" ? 520 : 610) * jitter(0.3),
      post: 4 * jitter(0.2),
    };
    return { id, hour, rps, cacheHit, batches, ms, total: LATENCY_STAGES.reduce((s, k) => s + ms[k], 0) };
  });
}

// ---------------- retry storm ----------------
export type RetryPolicy = { maxRetries: number; backoff: "none" | "exp-jitter"; retryBudget?: number; breaker: boolean };
export type Tick = { t: number; capacity: number; newReqs: number; sent: number; retries: number; served: number; dbErrorRate: number; userOk: number; userFail: number; breakerOpen: boolean };

export const STORM = { newRps: 320, capacity: 400, degradedCapacity: 300, outageStart: 60, outageEnd: 90, seconds: 240, rejectCost: 0.18 };

/**
 * One-second ticks of traffic to a vector DB cluster. Rejecting a request still costs the DB 18% of the work of
 * serving one, so offered load above capacity eats into goodput. Retries of failed attempts add to the next ticks' load.
 */
export function simulateStorm(p: RetryPolicy): Tick[] {
  const S = STORM;
  const sched: number[][] = Array.from({ length: S.seconds + 64 }, () => new Array<number>(p.maxRetries + 1).fill(0));
  const ticks: Tick[] = [];
  let openUntil = -1;
  for (let t = 0; t < S.seconds; t++) {
    const capacity = t >= S.outageStart && t < S.outageEnd ? S.degradedCapacity : S.capacity;
    const byAttempt = sched[t].map((c, k) => (k === 0 ? S.newRps : c));
    const breakerOpen = p.breaker && t <= openUntil;
    // While the breaker is open only 5% of calls probe the DB; the rest fail fast and are not retried.
    const admitted = byAttempt.map((c) => (breakerOpen ? c * 0.05 : c));
    let userFail = byAttempt.reduce((s, c, k) => s + (c - admitted[k]), 0);
    const sent = admitted.reduce((s, c) => s + c, 0);
    const served = sent <= capacity ? sent : Math.max(0, (capacity - S.rejectCost * sent) / (1 - S.rejectCost));
    const fail = sent ? 1 - served / sent : 0;
    let userOk = 0;
    let budget = p.retryBudget === undefined ? Infinity : p.retryBudget * S.newRps;
    admitted.forEach((c, k) => {
      userOk += c * (1 - fail);
      const failed = c * fail;
      const retry = k < p.maxRetries ? Math.min(failed, budget) : 0;
      budget -= retry;
      userFail += failed - retry;
      if (!retry) return;
      if (p.backoff === "none") sched[t + 1][k + 1] += retry;
      else {
        const span = 2 ** (k + 1); // full jitter: spread evenly over the next 2^(k+1) seconds
        for (let d = 1; d <= span; d++) sched[t + d][k + 1] += retry / span;
      }
    });
    if (p.breaker && fail > 0.5) openUntil = t + 5;
    ticks.push({ t, capacity, newReqs: S.newRps, sent, retries: sent - admitted[0], served, dbErrorRate: fail, userOk, userFail, breakerOpen });
  }
  return ticks;
}

/** Samples end-to-end latencies for requests starting in ticks [from, to): each attempt fails with that tick's DB error rate. */
export function stormLatencies(ticks: Tick[], p: RetryPolicy, from: number, to: number, seed: number) {
  const r = rng(seed);
  const out: { ms: number; ok: boolean; attempts: { ms: number; ok: boolean }[] }[] = [];
  for (let i = 0; i < 400; i++) {
    const tick = ticks[from + (i % (to - from))];
    const attempts: { ms: number; ok: boolean }[] = [];
    let ms = 25; // gateway + query embedding
    if (tick.breakerOpen && r.next() > 0.05) { out.push({ ms: ms + 2, ok: false, attempts }); continue; }
    for (let k = 0; k <= p.maxRetries; k++) {
      if (k > 0 && p.backoff === "exp-jitter") ms += r.next() * 2 ** k * 100;
      const ok = r.next() >= tick.dbErrorRate;
      const util = Math.min(0.97, tick.served / tick.capacity);
      // Overloaded: a request waits in the admission queue before it is rejected with 503.
      const a = ok ? 30 / (1 - util) : tick.sent > tick.capacity ? 180 + 120 * r.next() : 6;
      attempts.push({ ms: a, ok });
      ms += a;
      if (ok) break;
    }
    const ok = attempts.at(-1)!.ok;
    out.push({ ms: ms + (ok ? 520 * Math.exp(0.3 * r.gaussian()) : 3), ok, attempts });
  }
  return out;
}
