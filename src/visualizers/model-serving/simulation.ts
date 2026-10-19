import { clamp, mean } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { expSample, percentile } from "../latency-and-throughput/queueing";

export type ServingConfig = {
  arrivalRate: number; // req/s
  replicas: number;
  maxBatch: number;
  maxWaitMs: number;
  scaleOut: boolean; // autoscaler adds one replica at SCALE_AT_MS; it needs COLD_START_MS before it can serve
  routeBeforeReady: boolean; // break it: load balancer sends traffic to the new replica before its readiness check passes
  seed: number;
};
export const DEFAULT_SERVING: ServingConfig = { arrivalRate: 300, replicas: 2, maxBatch: 8, maxWaitMs: 10, scaleOut: false, routeBeforeReady: false, seed: 3 };

export const HORIZON_MS = 1200;
export const WINDOW_MS = 150;
export const SCALE_AT_MS = 300;
export const COLD_START_MS = 450;
/** Simulated GPU cost of one forward pass over a batch of b requests: a fixed launch cost plus a small per-item cost. */
export const BATCH_BASE_MS = 20;
export const PER_ITEM_MS = 2.5;
export const batchMs = (b: number) => BATCH_BASE_MS + PER_ITEM_MS * b;

export type BatchReason = "full" | "max wait" | "GPU was busy";
export type Replica = { id: number; createdAt: number; readyAt: number };
export type SReq = { id: number; arrive: number; replica: number; batch: number; start: number; end: number };
export type Batch = { id: number; replica: number; size: number; start: number; end: number; reason: BatchReason; reqIds: number[]; firstArrive: number };

export function sanitize(c: ServingConfig): ServingConfig {
  return {
    ...c,
    arrivalRate: clamp(c.arrivalRate, 1, 600),
    replicas: Math.round(clamp(c.replicas, 1, 4)),
    maxBatch: Math.round(clamp(c.maxBatch, 1, 16)),
    maxWaitMs: clamp(c.maxWaitMs, 0, 100),
  };
}

export function replicasOf(c: ServingConfig): Replica[] {
  const base = Array.from({ length: c.replicas }, (_, id) => ({ id, createdAt: 0, readyAt: 0 }));
  return c.scaleOut ? [...base, { id: c.replicas, createdAt: SCALE_AT_MS, readyAt: SCALE_AT_MS + COLD_START_MS }] : base;
}

/**
 * Dynamic batching per replica. When the GPU is free, a batch launches as soon as either the queue holds `maxBatch`
 * requests or the oldest one has waited `maxWaitMs`. If the GPU is still busy at that moment, the batch launches when
 * it frees up, with whatever is queued (up to maxBatch).
 */
export function batchReplica(arrivals: { id: number; arrive: number }[], readyAt: number, maxBatch: number, maxWaitMs: number) {
  const out: { start: number; end: number; size: number; reason: BatchReason; ids: number[]; firstArrive: number }[] = [];
  let free = readyAt, i = 0;
  while (i < arrivals.length) {
    const head = arrivals[i].arrive;
    const tFull = arrivals[i + maxBatch - 1]?.arrive ?? Number.POSITIVE_INFINITY;
    const tWait = head + maxWaitMs;
    const ready = Math.min(tFull, tWait);
    const start = Math.max(free, ready);
    const ids: number[] = [];
    while (i < arrivals.length && ids.length < maxBatch && arrivals[i].arrive <= start) ids.push(arrivals[i++].id);
    const reason: BatchReason = ids.length === maxBatch ? "full" : start > ready ? "GPU was busy" : "max wait";
    const end = start + batchMs(ids.length);
    out.push({ start, end, size: ids.length, reason, ids, firstArrive: head });
    free = end;
  }
  return out;
}

export type ServingRun = ReturnType<typeof runServing>;

export function runServing(cfgIn: ServingConfig) {
  const cfg = sanitize(cfgIn);
  const r = rng(cfg.seed);
  const replicas = replicasOf(cfg);
  const arrivals: number[] = [];
  for (let t = expSample(r.next(), 1000 / cfg.arrivalRate); t < HORIZON_MS; t += expSample(r.next(), 1000 / cfg.arrivalRate)) arrivals.push(t);
  // Round-robin load balancer over replicas that may receive traffic at that moment.
  let rr = 0;
  const routed = arrivals.map((arrive, id) => {
    const eligible = replicas.filter((p) => (cfg.routeBeforeReady ? p.createdAt : p.readyAt) <= arrive);
    const replica = eligible[rr++ % eligible.length].id;
    return { id, arrive, replica };
  });
  const reqs: SReq[] = routed.map((q) => ({ ...q, batch: -1, start: 0, end: 0 }));
  const batches: Batch[] = [];
  for (const p of replicas) {
    for (const b of batchReplica(routed.filter((q) => q.replica === p.id), p.readyAt, cfg.maxBatch, cfg.maxWaitMs)) {
      const id = batches.length;
      batches.push({ id, replica: p.id, size: b.size, start: b.start, end: b.end, reason: b.reason, reqIds: b.ids, firstArrive: b.firstArrive });
      for (const rid of b.ids) Object.assign(reqs[rid], { batch: id, start: b.start, end: b.end });
    }
  }
  batches.sort((a, b) => a.start - b.start);
  return { cfg, replicas, reqs, batches };
}

/** Metrics over everything that has happened by simulated time t (t = HORIZON_MS gives the run's headline numbers). */
export function metricsAt(run: Pick<ServingRun, "reqs" | "batches" | "replicas">, t: number) {
  const done = run.reqs.filter((q) => q.end <= t);
  const lat = done.map((q) => q.end - q.arrive);
  const busy = run.batches.reduce((s, b) => s + Math.max(0, Math.min(t, b.end) - Math.min(t, b.start)), 0);
  const avail = run.replicas.reduce((s, p) => s + Math.max(0, t - p.readyAt), 0);
  const launched = run.batches.filter((b) => b.start <= t);
  return {
    completed: done.length,
    throughput: t > 0 ? done.length / (t / 1000) : 0,
    p50: percentile(lat, 50),
    p95: percentile(lat, 95),
    gpuUtil: avail > 0 ? busy / avail : 0,
    meanBatch: launched.length ? mean(launched.map((b) => b.size)) : Number.NaN,
    batches: launched.length,
  };
}

export type ServingState = { t0: number; t1: number; closed: Record<BatchReason, number>; coldEvent: "created" | "ready" | null; queuedAtEnd: number };

export function simulate(cfgIn: ServingConfig) {
  const run = runServing(cfgIn);
  const { cfg, reqs, batches, replicas } = run;
  const steps: SimulationStep<ServingState>[] = [];
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "serving", message, severity, tex });
  const scaled = replicas.find((p) => p.createdAt > 0);
  for (let t0 = 0; t0 < HORIZON_MS; t0 += WINDOW_MS) {
    const t1 = t0 + WINDOW_MS;
    const inWin = batches.filter((b) => b.start >= t0 && b.start < t1);
    const closed: Record<BatchReason, number> = { full: 0, "max wait": 0, "GPU was busy": 0 };
    for (const b of inWin) closed[b.reason]++;
    const coldEvent = scaled && scaled.createdAt >= t0 && scaled.createdAt < t1 ? "created" : scaled && scaled.readyAt >= t0 && scaled.readyAt < t1 ? "ready" : null;
    const queuedAtEnd = reqs.filter((q) => q.arrive <= t1 && q.start > t1).length;
    const m = metricsAt(run, t1);
    const events: SimulationEvent[] = [
      ev(`a${t0}`, `${reqs.filter((q) => q.arrive >= t0 && q.arrive < t1).length} requests arrived; ${inWin.length} batches launched (${Object.entries(closed).filter(([, n]) => n).map(([k, n]) => `${n} ${k}`).join(", ") || "none"}).`),
      ...inWin.slice(0, 4).map((b) => ev(`b${b.id}`, `batch #${b.id} on replica ${b.replica + 1}: ${b.size} req, ${b.reason}, ${batchMs(b.size).toFixed(1)} ms`, undefined,
        `t_{\\text{batch}} = ${BATCH_BASE_MS} + ${PER_ITEM_MS} \\times ${b.size} = ${batchMs(b.size).toFixed(1)}\\,\\text{ms}`)),
    ];
    if (coldEvent === "created") events.push(ev("cold", `Autoscaler adds replica ${scaled!.id + 1}; loading weights takes ${COLD_START_MS} ms (cold start).`, "warning"));
    if (coldEvent === "ready") events.push(ev("ready", `Replica ${scaled!.id + 1} passed its readiness check at ${scaled!.readyAt} ms and can serve.`));
    if (queuedAtEnd > 3 * cfg.maxBatch * replicas.length) events.push(ev(`q${t0}`, `${queuedAtEnd} requests waiting: arrivals exceed what the GPUs can process.`, "error"));
    steps.push({
      id: `w${t0}`,
      title: `${t0}–${t1} ms${coldEvent ? (coldEvent === "created" ? " (cold start)" : " (replica ready)") : ""}`,
      description: `Simulated ${t0}–${t1} ms. So far: ${m.completed} responses, p95 ${Number.isFinite(m.p95) ? m.p95.toFixed(0) : "—"} ms, GPU busy ${(m.gpuUtil * 100).toFixed(0)}% of the time, average batch ${Number.isFinite(m.meanBatch) ? m.meanBatch.toFixed(1) : "—"}.`,
      state: { t0, t1, closed, coldEvent, queuedAtEnd },
      events,
    });
  }
  return { ...run, steps, final: metricsAt(run, HORIZON_MS) };
}

/** Same traffic, different max batch sizes: batching trades a little wait for much higher GPU efficiency. */
export function sweepBatch(cfgIn: ServingConfig) {
  return [1, 2, 4, 8, 12, 16].map((maxBatch) => {
    const run = runServing({ ...cfgIn, maxBatch });
    const m = metricsAt(run, HORIZON_MS);
    const all = run.reqs.map((q) => q.end - q.arrive);
    return { maxBatch, throughput: m.throughput, p50: percentile(all, 50), p95: percentile(all, 95), gpuUtil: m.gpuUtil, meanBatch: m.meanBatch };
  });
}
