import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

/** What the (simulated) server does on attempt n. The last entry repeats for later attempts. */
export type ServerReply = { latencyMs: number; status: 200 | 400 | 404 | 429 | 500 | 503 };
export type ScenarioId = "flaky" | "slow" | "rateLimited" | "badRequest" | "payment" | "outage";

export const SCENARIOS: Record<ScenarioId, { label: string; note: string; replies: ServerReply[]; sideEffect: boolean }> = {
  flaky: { label: "Flaky service (503, 503, then OK)", note: "Two transient 503s, then the third attempt succeeds.", replies: [{ latencyMs: 300, status: 503 }, { latencyMs: 350, status: 503 }, { latencyMs: 280, status: 200 }], sideEffect: false },
  slow: { label: "Cold start (slow, then fast)", note: "The first calls are slower than the timeout while the service warms up.", replies: [{ latencyMs: 2600, status: 200 }, { latencyMs: 2300, status: 200 }, { latencyMs: 900, status: 200 }], sideEffect: false },
  rateLimited: { label: "Rate limited (429 ×3)", note: "The API rejects bursts with 429 Too Many Requests. Backoff gives it room.", replies: [{ latencyMs: 120, status: 429 }, { latencyMs: 120, status: 429 }, { latencyMs: 120, status: 429 }, { latencyMs: 250, status: 200 }], sideEffect: false },
  badRequest: { label: "Bad request (400)", note: "The request itself is invalid. No number of retries can fix it.", replies: [{ latencyMs: 150, status: 400 }], sideEffect: false },
  payment: { label: "Charge a card (slow first call)", note: "The server charges the card even though the client already gave up waiting. Toggle the idempotency key.", replies: [{ latencyMs: 2600, status: 200 }, { latencyMs: 500, status: 200 }], sideEffect: true },
  outage: { label: "Outage (503 forever)", note: "Every attempt fails. Retries cannot help; the deadline bounds the damage.", replies: [{ latencyMs: 250, status: 503 }], sideEffect: false },
};

export type Jitter = "none" | "full" | "equal";
export type RetryConfig = {
  scenario: ScenarioId;
  maxAttempts: number;
  timeoutMs: number;
  baseBackoffMs: number;
  maxBackoffMs: number;
  jitter: Jitter;
  deadlineMs: number;
  idempotencyKey: boolean;
  retryAllErrors: boolean;
  seed: number;
};
export const DEFAULT_RETRY: RetryConfig = {
  scenario: "flaky", maxAttempts: 4, timeoutMs: 2000, baseBackoffMs: 200, maxBackoffMs: 4000, jitter: "none", deadlineMs: 8000, idempotencyKey: false, retryAllErrors: false, seed: 7,
};

export function sanitize(c: RetryConfig): RetryConfig {
  return {
    ...c,
    maxAttempts: Math.round(clamp(c.maxAttempts, 1, 8)),
    timeoutMs: clamp(c.timeoutMs, 100, 10000),
    baseBackoffMs: clamp(c.baseBackoffMs, 0, 5000),
    maxBackoffMs: clamp(c.maxBackoffMs, 0, 20000),
    deadlineMs: clamp(c.deadlineMs, 200, 60000),
  };
}

export const RETRYABLE = new Set([429, 500, 503]);
export const isRetryable = (outcome: AttemptOutcome) => outcome === "timeout" || (typeof outcome === "number" && RETRYABLE.has(outcome));

/** Backoff before attempt n+1 (n = attempts already made, ≥1): exponential, capped, then jittered. */
export function backoffCeiling(n: number, base: number, cap: number) {
  return Math.min(cap, base * 2 ** (n - 1));
}
export function jittered(ceiling: number, jitter: Jitter, u: number) {
  if (jitter === "full") return Math.round(u * ceiling);
  if (jitter === "equal") return Math.round(ceiling / 2 + (u * ceiling) / 2);
  return ceiling;
}

export type AttemptOutcome = "timeout" | "deadline" | ServerReply["status"];
export type Segment =
  | {
      kind: "attempt"; n: number; start: number; end: number; outcome: AttemptOutcome;
      /** When the server actually finished, even if the client had stopped waiting. */
      serverEnd: number; retryable: boolean;
      /** Payment scenario: a new charge was created / the stored result was replayed. */
      charge?: "new" | "replayed";
    }
  | { kind: "backoff"; n: number; start: number; end: number; ceiling: number; u: number };

export type FinalStatus = "success" | "non-retryable" | "exhausted" | "deadline";
export type RetryRun = { segments: Segment[]; status: FinalStatus; endMs: number; charges: number; attempts: number; reason: string };

export function runRetries(cfgIn: RetryConfig): RetryRun {
  const c = sanitize(cfgIn);
  const sc = SCENARIOS[c.scenario];
  const r = rng(c.seed);
  const segments: Segment[] = [];
  let t = 0;
  let charges = 0;
  let committed = false;
  for (let n = 1; n <= c.maxAttempts; n++) {
    const reply = sc.replies[Math.min(n - 1, sc.replies.length - 1)];
    // With an idempotency key the server recognises a repeat and returns the stored result straight away.
    const replay = sc.sideEffect && c.idempotencyKey && committed;
    const latency = replay ? 80 : reply.latencyMs;
    const limit = Math.min(c.timeoutMs, c.deadlineMs - t);
    const cut = latency > limit;
    const outcome: AttemptOutcome = cut ? (c.timeoutMs <= c.deadlineMs - t ? "timeout" : "deadline") : reply.status;
    let charge: "new" | "replayed" | undefined;
    if (sc.sideEffect && reply.status === 200) {
      // The server does the work whether or not the client is still listening.
      charge = replay ? "replayed" : "new";
      if (!replay) charges++;
      committed = true;
    }
    const end = t + (cut ? limit : latency);
    const retryable = isRetryable(outcome) || (c.retryAllErrors && outcome !== 200 && outcome !== "deadline");
    segments.push({ kind: "attempt", n, start: t, end, outcome, serverEnd: t + latency, retryable, charge });
    t = end;
    if (outcome === 200) return { segments, status: "success", endMs: t, charges, attempts: n, reason: `Attempt ${n} returned 200 OK after ${t} ms in total.` };
    if (outcome === "deadline") return { segments, status: "deadline", endMs: t, charges, attempts: n, reason: `The overall deadline of ${c.deadlineMs} ms cut attempt ${n} short.` };
    if (!retryable) return { segments, status: "non-retryable", endMs: t, charges, attempts: n, reason: `HTTP ${outcome} is not retryable: the same request will fail the same way, so the client stops.` };
    if (n === c.maxAttempts) break;
    const ceiling = backoffCeiling(n, c.baseBackoffMs, c.maxBackoffMs);
    const u = r.next();
    const wait = jittered(ceiling, c.jitter, u);
    if (t + wait >= c.deadlineMs) return { segments, status: "deadline", endMs: t, charges, attempts: n, reason: `Waiting ${wait} ms more would pass the ${c.deadlineMs} ms deadline, so the client gives up now instead of starting a doomed attempt.` };
    segments.push({ kind: "backoff", n, start: t, end: t + wait, ceiling, u });
    t += wait;
  }
  return { segments, status: "exhausted", endMs: t, charges, attempts: c.maxAttempts, reason: `All ${c.maxAttempts} attempts failed with retryable errors; the client returns the last error.` };
}

export type RetryState = { visible: number; now: number; done: boolean; status?: FinalStatus; charges: number };

export function simulate(cfg: RetryConfig): { run: RetryRun; steps: SimulationStep<RetryState>[] } {
  const c = sanitize(cfg);
  const run = runRetries(c);
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "retry", message, severity, tex });
  const steps: SimulationStep<RetryState>[] = [{
    id: "start", title: "Plan the call",
    description: `Up to ${c.maxAttempts} attempts, each allowed ${c.timeoutMs} ms, all inside an overall deadline of ${c.deadlineMs} ms. Retryable: timeouts, 429, 500, 503${c.retryAllErrors ? " and (misconfigured) every other error" : ""}.`,
    state: { visible: 0, now: 0, done: false, charges: 0 },
    events: [ev("cfg", `${SCENARIOS[c.scenario].label}`)],
  }];
  let charges = 0;
  run.segments.forEach((s, i) => {
    if (s.kind === "attempt") {
      if (s.charge === "new") charges++;
      const what = s.outcome === 200 ? "200 OK" : s.outcome === "timeout" ? `timed out at ${c.timeoutMs} ms` : s.outcome === "deadline" ? "cut off by the deadline" : `HTTP ${s.outcome}`;
      steps.push({
        id: `a${s.n}`, title: `Attempt ${s.n}: ${what}`,
        description: s.outcome === "timeout" || s.outcome === "deadline"
          ? `The client stops waiting after ${s.end - s.start} ms. The server is still working and finishes at ${s.serverEnd} ms${s.charge ? ", charging the card anyway" : ""}.`
          : `The server answered in ${s.end - s.start} ms with ${what}.${s.outcome !== 200 ? (s.retryable ? " This error is transient, so a retry may succeed." : " This error is permanent.") : ""}`,
        state: { visible: i + 1, now: s.end, done: false, charges },
        events: [
          ev(`a${s.n}`, `attempt ${s.n}: ${s.start}–${s.end} ms → ${what}`, s.outcome === 200 ? "info" : "warning"),
          ...(s.charge === "new" && charges > 1 ? [ev(`dup${s.n}`, `Duplicate charge #${charges} created by a retry without an idempotency key`, "error")] : []),
          ...(s.charge === "replayed" ? [ev(`rep${s.n}`, "Idempotency key matched: server returned the original charge, no new charge")] : []),
        ],
      });
    } else {
      steps.push({
        id: `b${s.n}`, title: `Back off ${s.end - s.start} ms`,
        description: `Wait before attempt ${s.n + 1}. Ceiling = min(cap, base × 2^${s.n - 1}) = ${s.ceiling} ms${c.jitter === "none" ? "" : `; ${c.jitter} jitter draws u = ${s.u.toFixed(3)}, giving ${s.end - s.start} ms`}.`,
        state: { visible: i + 1, now: s.end, done: false, charges },
        events: [ev(`b${s.n}`, `backoff ${s.end - s.start} ms`, undefined, `\\text{wait}_{${s.n}} = ${c.jitter === "none" ? "" : c.jitter === "full" ? `${s.u.toFixed(3)} \\times ` : `\\tfrac{${s.ceiling}}{2} + ${s.u.toFixed(3)} \\times \\tfrac{${s.ceiling}}{2} \\approx `}${c.jitter === "equal" ? "" : `\\min(${c.maxBackoffMs}, ${c.baseBackoffMs} \\cdot 2^{${s.n - 1}}) = `}${s.end - s.start}\\,\\text{ms}`)],
      });
    }
  });
  steps.push({
    id: "end", title: run.status === "success" ? "Success" : run.status === "non-retryable" ? "Stop: non-retryable error" : run.status === "deadline" ? "Stop: deadline" : "Stop: attempts exhausted",
    description: run.reason + (run.charges > 1 ? ` The customer was charged ${run.charges} times.` : ""),
    state: { visible: run.segments.length, now: run.endMs, done: true, status: run.status, charges: run.charges },
    events: [ev("end", run.reason, run.status === "success" ? "info" : "error"), ...(run.charges > 1 ? [ev("dup", `${run.charges} charges for one purchase`, "error")] : [])],
  });
  return { run, steps };
}

/**
 * Analytic model: every attempt fails independently with probability f; a failure costs the full timeout,
 * a success costs `okMs`. Backoff uses the expected jittered wait.
 */
export function expectedBackoff(n: number, c: RetryConfig) {
  const ceil = backoffCeiling(n, c.baseBackoffMs, c.maxBackoffMs);
  return c.jitter === "full" ? ceil / 2 : c.jitter === "equal" ? (3 * ceil) / 4 : ceil;
}
export function reliability(f: number, attempts: number, c: RetryConfig, okMs = 300) {
  const p = clamp(f, 0, 1);
  const pSuccess = 1 - p ** attempts;
  let expected = 0;
  for (let i = 1; i <= attempts; i++) {
    const reach = p ** (i - 1);
    expected += reach * ((i > 1 ? expectedBackoff(i - 1, c) : 0) + (1 - p) * okMs + p * c.timeoutMs);
  }
  return { pSuccess, expectedMs: expected };
}
