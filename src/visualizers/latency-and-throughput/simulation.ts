import { clamp, mean } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { expSample, percentile, timeAverage } from "./queueing";

export type ServiceDist = "constant" | "exponential" | "heavy-tail";
export type LTConfig = { arrivalRate: number; serviceMs: number; dist: ServiceDist; workers: number; burst: boolean; seed: number };
export const DEFAULT_LT: LTConfig = { arrivalRate: 10, serviceMs: 250, dist: "exponential", workers: 3, burst: false, seed: 7 };

/** Arrivals are generated in [0, HORIZON); the system then drains so every request has a latency. */
export const HORIZON_MS = 8000;
export const WINDOW_MS = 1000;
/** Burst preset: arrival rate ×4 between these times. */
export const BURST: [number, number] = [2000, 3500];
export const BURST_FACTOR = 4;

export type Req = { id: number; arrive: number; start: number; end: number; worker: number; service: number };

export function sanitize(c: LTConfig): LTConfig {
  return {
    ...c,
    arrivalRate: clamp(c.arrivalRate, 0, 60),
    serviceMs: clamp(c.serviceMs, 10, 2000),
    workers: Math.round(clamp(c.workers, 1, 8)),
  };
}

/** Service time with the configured mean. Heavy tail: 90% fast, 10% are 10× slower (mean preserved). */
export function serviceSample(dist: ServiceDist, meanMs: number, u: number, u2: number): number {
  if (dist === "constant") return meanMs;
  if (dist === "exponential") return expSample(u, meanMs);
  const fast = meanMs / 1.9;
  return (u2 < 0.9 ? fast : fast * 10) * (0.75 + 0.5 * u);
}

export const rateAt = (c: LTConfig, t: number) => (c.burst && t >= BURST[0] && t < BURST[1] ? c.arrivalRate * BURST_FACTOR : c.arrivalRate);

/** FCFS queue with c identical workers (M/G/c): each request goes to whichever worker frees up first. */
export function runQueue(cfgIn: LTConfig): Req[] {
  const c = sanitize(cfgIn);
  const r = rng(c.seed);
  const reqs: Req[] = [];
  const free = new Array<number>(c.workers).fill(0);
  let t = 0;
  for (let id = 0; ; id++) {
    const lam = rateAt(c, t);
    if (lam <= 0) break;
    t += expSample(r.next(), 1000 / lam);
    if (t >= HORIZON_MS) break;
    const service = serviceSample(c.dist, c.serviceMs, r.next(), r.next());
    let w = 0;
    for (let i = 1; i < free.length; i++) if (free[i] < free[w]) w = i;
    const start = Math.max(t, free[w]);
    free[w] = start + service;
    reqs.push({ id, arrive: t, start, end: start + service, worker: w, service });
  }
  return reqs;
}

export type LTSummary = {
  n: number;
  offeredRate: number; // λ, requests per second actually generated
  capacity: number; // c / E[S], requests per second
  rho: number; // offered utilization λ·E[S]/c
  throughput: number; // completions inside the horizon per second
  p50: number; p95: number; p99: number; meanW: number; maxQueue: number;
  // Little's law over the whole run [0, T_end]
  tEnd: number; lambdaEff: number; L: number; lambdaW: number;
};

export function summarize(reqs: Req[], cfgIn: LTConfig): LTSummary {
  const c = sanitize(cfgIn);
  const W = reqs.map((q) => q.end - q.arrive);
  const tEnd = Math.max(HORIZON_MS, ...reqs.map((q) => q.end));
  const L = timeAverage(reqs.flatMap((q) => [{ t: q.arrive, d: 1 }, { t: q.end, d: -1 }]), tEnd);
  const queueEv = reqs.flatMap((q) => [{ t: q.arrive, d: 1 }, { t: q.start, d: -1 }]).sort((a, b) => a.t - b.t || a.d - b.d);
  let qn = 0, maxQueue = 0;
  for (const e of queueEv) maxQueue = Math.max(maxQueue, (qn += e.d));
  const meanS = reqs.length ? mean(reqs.map((q) => q.service)) : c.serviceMs;
  const lambdaEff = reqs.length / (tEnd / 1000);
  const meanW = reqs.length ? mean(W) : Number.NaN;
  return {
    n: reqs.length,
    offeredRate: reqs.length / (HORIZON_MS / 1000),
    capacity: (c.workers * 1000) / c.serviceMs,
    rho: (reqs.length / (HORIZON_MS / 1000)) * (meanS / 1000) / c.workers,
    throughput: reqs.filter((q) => q.end <= HORIZON_MS).length / (HORIZON_MS / 1000),
    p50: percentile(W, 50), p95: percentile(W, 95), p99: percentile(W, 99), meanW, maxQueue,
    tEnd, lambdaEff, L, lambdaW: reqs.length ? lambdaEff * (meanW / 1000) : 0,
  };
}

/** What is in the system at simulated time t. */
export function snapshot(reqs: Req[], t: number) {
  const queued = reqs.filter((q) => q.arrive <= t && q.start > t);
  const serving = reqs.filter((q) => q.start <= t && q.end > t);
  const done = reqs.filter((q) => q.end <= t);
  return { queued, serving, done, arrived: queued.length + serving.length + done.length };
}

export type LTState = { kind: "window" | "summary"; t0: number; t1: number; arrivals: number; completions: number; queueAtEnd: number; inBurst: boolean; p95SoFar: number };

export function simulate(cfgIn: LTConfig) {
  const cfg = sanitize(cfgIn);
  const reqs = runQueue(cfg);
  const sum = summarize(reqs, cfg);
  const steps: SimulationStep<LTState>[] = [];
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id, type: "queue", message, severity, tex });
  for (let t0 = 0; t0 < HORIZON_MS; t0 += WINDOW_MS) {
    const t1 = t0 + WINDOW_MS;
    const arrivals = reqs.filter((q) => q.arrive >= t0 && q.arrive < t1).length;
    const completions = reqs.filter((q) => q.end >= t0 && q.end < t1).length;
    const snap = snapshot(reqs, t1);
    const doneW = snap.done.map((q) => q.end - q.arrive);
    const inBurst = cfg.burst && t0 < BURST[1] && t1 > BURST[0];
    const events: SimulationEvent[] = [
      ev(`a${t0}`, `${arrivals} arrivals and ${completions} completions between ${t0 / 1000} s and ${t1 / 1000} s.`),
      ev(`q${t0}`, `Queue length at ${t1 / 1000} s: ${snap.queued.length}; workers busy: ${snap.serving.length}/${cfg.workers}.`, snap.queued.length > 10 ? "warning" : undefined),
    ];
    if (inBurst) events.push(ev(`b${t0}`, `Burst: arrival rate is ${rateAt(cfg, BURST[0])}/s (×${BURST_FACTOR}) between ${BURST[0] / 1000} s and ${BURST[1] / 1000} s.`, "warning"));
    steps.push({
      id: `w${t0}`,
      title: `${t0 / 1000}–${t1 / 1000} s${inBurst ? " (burst)" : ""}`,
      description: `Simulated time ${t0 / 1000} s to ${t1 / 1000} s. Requests arrive at random (Poisson, λ = ${rateAt(cfg, t0)}/s), wait in one FIFO queue, and the first free worker takes the next one.`,
      state: { kind: "window", t0, t1, arrivals, completions, queueAtEnd: snap.queued.length, inBurst, p95SoFar: percentile(doneW, 95) },
      events,
    });
  }
  steps.push({
    id: "summary",
    title: "Summary: Little's law",
    description: `All ${sum.n} requests have finished (the last at ${(sum.tEnd / 1000).toFixed(2)} s). The average number in the system equals arrival rate × average time in the system.`,
    state: { kind: "summary", t0: 0, t1: sum.tEnd, arrivals: sum.n, completions: sum.n, queueAtEnd: 0, inBurst: false, p95SoFar: sum.p95 },
    events: [
      ev("little", `L = ${sum.L.toFixed(3)} measured; λW = ${sum.lambdaEff.toFixed(3)} × ${(sum.meanW / 1000).toFixed(3)} = ${sum.lambdaW.toFixed(3)}`, undefined,
        `L = ${sum.L.toFixed(3)} \\quad \\lambda W = ${sum.lambdaEff.toFixed(3)} \\times ${(sum.meanW / 1000).toFixed(3)} = ${sum.lambdaW.toFixed(3)}`),
      ev("pct", `p50 = ${sum.p50.toFixed(0)} ms, p95 = ${sum.p95.toFixed(0)} ms, p99 = ${sum.p99.toFixed(0)} ms`, sum.rho >= 1 ? "error" : sum.rho > 0.85 ? "warning" : undefined),
    ],
  });
  return { cfg, reqs, sum, steps };
}

/** Sweep the arrival rate to show throughput saturating at capacity and tail latency exploding near ρ = 1. */
export function sweep(cfgIn: LTConfig, points = 10) {
  const cfg = sanitize({ ...cfgIn, burst: false });
  const cap = (cfg.workers * 1000) / cfg.serviceMs;
  return Array.from({ length: points }, (_, i) => {
    const lam = Math.round(cap * (0.15 + (1.35 * i) / (points - 1)) * 10) / 10;
    const s = summarize(runQueue({ ...cfg, arrivalRate: lam }), { ...cfg, arrivalRate: lam });
    return { lambda: lam, rho: s.rho, throughput: s.throughput, p50: s.p50, p99: s.p99 };
  });
}
