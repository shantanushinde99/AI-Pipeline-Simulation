import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

/** Standard normal CDF via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7). */
export function normCdf(z: number) {
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return z >= 0 ? 0.5 * (1 + erf) : 0.5 * (1 - erf);
}

export type ZTest = { pA: number; pB: number; diff: number; se: number; ci: [number, number]; z: number; p: number };
/** Two-proportion z-test (pooled SE for the test, unpooled SE for the 95% CI of pB − pA). */
export function twoPropTest(nA: number, cA: number, nB: number, cB: number): ZTest {
  const pA = nA ? cA / nA : 0, pB = nB ? cB / nB : 0;
  const pool = nA + nB ? (cA + cB) / (nA + nB) : 0;
  const sePool = Math.sqrt(pool * (1 - pool) * (1 / Math.max(nA, 1) + 1 / Math.max(nB, 1)));
  const se = Math.sqrt((pA * (1 - pA)) / Math.max(nA, 1) + (pB * (1 - pB)) / Math.max(nB, 1));
  const diff = pB - pA;
  const z = sePool > 0 ? diff / sePool : 0;
  return { pA, pB, diff, se, ci: [diff - 1.96 * se, diff + 1.96 * se], z, p: 2 * (1 - normCdf(Math.abs(z))) };
}

/** Users per arm for a two-sided test at α = 0.05 with 80% power to detect an absolute lift `mde` over `p1`. */
export function sampleSizePerArm(p1: number, mde: number) {
  const p2 = p1 + mde;
  return Math.ceil(((1.959964 + 0.841621) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))) / (mde * mde));
}

export type OnlineParams = { baseRate: number; trueLift: number; usersPerDay: number; days: number; peeking: boolean; addedLatency: number; seed: number };
export const DEFAULT_ONLINE: OnlineParams = { baseRate: 0.1, trueLift: 0.015, usersPerDay: 1000, days: 14, peeking: false, addedLatency: 40, seed: 3 };
export const MDE = 0.015;
export const LATENCY_MARGIN = 100;
export const NULL_EXPERIMENTS = 200;

export type DayRow = { day: number; nA: number; cA: number; nB: number; cB: number; test: ZTest; latA: number; latB: number; sample: { arm: "A" | "B"; conv: boolean }[] };
export type Phase = "setup" | "day" | "guardrail" | "decision" | "peeking";
export type OnlineState = { phase: Phase; day: number };

export function clampParams(p: OnlineParams): OnlineParams {
  return {
    ...p,
    baseRate: clamp(p.baseRate, 0.01, 0.6),
    trueLift: clamp(p.trueLift, -0.05, 0.05),
    usersPerDay: clamp(Math.round(p.usersPerDay), 50, 5000),
    days: clamp(Math.round(p.days), 3, 28),
    addedLatency: clamp(p.addedLatency, 0, 400),
  };
}

/** The live experiment: every simulated user is randomly assigned to A or B and converts with their arm's true rate. */
export function runExperiment(p: OnlineParams): DayRow[] {
  const r = rng(p.seed);
  const pB = clamp(p.baseRate + p.trueLift, 0, 1);
  let nA = 0, cA = 0, nB = 0, cB = 0;
  const rows: DayRow[] = [];
  for (let d = 1; d <= p.days; d++) {
    const sample: DayRow["sample"] = [];
    for (let u = 0; u < p.usersPerDay; u++) {
      const arm = r.next() < 0.5 ? "A" : "B";
      const conv = r.next() < (arm === "A" ? p.baseRate : pB);
      if (arm === "A") { nA++; cA += Number(conv); } else { nB++; cB += Number(conv); }
      if (u < 30) sample.push({ arm, conv });
    }
    const latA = 820 + 15 * r.gaussian();
    const latB = 820 + p.addedLatency + 15 * r.gaussian();
    rows.push({ day: d, nA, cA, nB, cB, test: twoPropTest(nA, cA, nB, cB), latA, latB, sample });
  }
  return rows;
}

/** Many A/A experiments (no true difference). Returns, per experiment, the first day p < 0.05 (or null) and whether the final day is significant. */
export function nullExperiments(p: OnlineParams, count = NULL_EXPERIMENTS) {
  const r = rng(p.seed * 7919 + 17);
  const half = Math.max(1, Math.floor(p.usersPerDay / 2));
  const firstSig: (number | null)[] = [];
  const finalSig: boolean[] = [];
  for (let e = 0; e < count; e++) {
    let cA = 0, cB = 0, first: number | null = null, last = false;
    for (let d = 1; d <= p.days; d++) {
      for (let u = 0; u < half; u++) { cA += Number(r.next() < p.baseRate); cB += Number(r.next() < p.baseRate); }
      const sig = twoPropTest(half * d, cA, half * d, cB).p < 0.05;
      if (sig && first === null) first = d;
      if (d === p.days) last = sig;
    }
    firstSig.push(first);
    finalSig.push(last);
  }
  const peekFpr = firstSig.filter((f) => f !== null).length / count;
  const fixedFpr = finalSig.filter(Boolean).length / count;
  const cumulative = Array.from({ length: p.days }, (_, i) => firstSig.filter((f) => f !== null && f <= i + 1).length / count);
  return { firstSig, finalSig, peekFpr, fixedFpr, cumulative };
}

export function guardrail(rows: DayRow[]) {
  const d = rows.map((r) => r.latB - r.latA);
  const n = d.length;
  const mean = d.reduce((s, v) => s + v, 0) / Math.max(n, 1);
  const sd = n > 1 ? Math.sqrt(d.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)) : 0;
  const upper = mean + 1.96 * (sd / Math.sqrt(Math.max(n, 1)));
  return { mean, sd, upper, pass: upper <= LATENCY_MARGIN };
}

export function simulate(input: OnlineParams) {
  const p = clampParams(input);
  const all = runExperiment(p);
  const stopIdx = p.peeking ? all.findIndex((r) => r.test.p < 0.05) : -1;
  const rows = stopIdx >= 0 ? all.slice(0, stopIdx + 1) : all;
  const final = rows[rows.length - 1];
  const fullFinal = all[all.length - 1];
  const guard = guardrail(rows);
  const nulls = nullExperiments(p);
  const nNeeded = sampleSizePerArm(p.baseRate, MDE);
  const daysNeeded = Math.ceil(nNeeded / (p.usersPerDay / 2));
  const significant = final.test.p < 0.05;
  const positive = final.test.diff > 0;
  const decision: "ship" | "blocked" | "no-ship" = significant && positive ? (guard.pass ? "ship" : "blocked") : guard.pass ? "no-ship" : "blocked";
  const stoppedEarly = stopIdx >= 0 && stopIdx < all.length - 1;

  const ev = (id: string, message: string, tex?: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "ab", message, tex, severity });
  const f = (v: number, d = 4) => v.toFixed(d);
  const steps: SimulationStep<OnlineState>[] = [
    { id: "setup", title: "Plan the experiment", state: { phase: "setup", day: 0 },
      description: `Primary metric: conversion. To detect a +${(MDE * 100).toFixed(1)} pt lift on a ${(p.baseRate * 100).toFixed(0)}% base with 80% power you need ${nNeeded.toLocaleString()} users per arm ≈ ${daysNeeded} days at ${p.usersPerDay}/day. Planned: ${p.days} days.`,
      events: [ev("n", `Required n per arm = ${nNeeded}`, `n = \\frac{(z_{0.975}+z_{0.8})^2\\,[p_1(1-p_1)+p_2(1-p_2)]}{\\delta^2} = ${nNeeded}`, daysNeeded > p.days ? "warning" : undefined)] },
    ...rows.map((r, i): SimulationStep<OnlineState> => {
      const stopHere = stoppedEarly && i === rows.length - 1;
      return {
        id: `day-${r.day}`, title: `Day ${r.day}${stopHere ? " (peeker stops)" : ""}`, state: { phase: "day", day: r.day },
        description: `${p.usersPerDay} new users split between the arms. Cumulative: A ${r.cA}/${r.nA}, B ${r.cB}/${r.nB}. Δ = ${(r.test.diff * 100).toFixed(2)} pts, 95% CI [${(r.test.ci[0] * 100).toFixed(2)}, ${(r.test.ci[1] * 100).toFixed(2)}], p = ${r.test.p.toFixed(3)}.`
          + (stopHere ? " The peeker sees p < 0.05 and stops the test here." : ""),
        events: [
          ev(`d${r.day}`, `z = ${f(r.test.z, 3)}, p = ${f(r.test.p)}`, `z = \\frac{${f(r.test.pB)} - ${f(r.test.pA)}}{SE_{pool}} = ${f(r.test.z, 3)},\\; p = ${f(r.test.p)}`, r.test.p < 0.05 ? "warning" : undefined),
          ...(stopHere ? [ev("stop", `Stopped on day ${r.day} of ${p.days} at the first p < 0.05`, undefined, "warning")] : []),
        ],
      };
    }),
    { id: "guardrail", title: "Guardrail: p95 latency", state: { phase: "guardrail", day: final.day },
      description: `B adds ${guard.mean.toFixed(0)} ms to p95 latency on average (upper 95% bound ${guard.upper.toFixed(0)} ms). The guardrail allows at most ${LATENCY_MARGIN} ms.`,
      events: [ev("g", `Guardrail ${guard.pass ? "passed" : "FAILED"}: upper bound ${guard.upper.toFixed(1)} ms vs margin ${LATENCY_MARGIN} ms`, `\\bar{d} + 1.96\\,\\tfrac{s}{\\sqrt{n}} = ${guard.mean.toFixed(1)} + 1.96 \\cdot \\tfrac{${guard.sd.toFixed(1)}}{\\sqrt{${rows.length}}} = ${guard.upper.toFixed(1)}\\text{ ms}`, guard.pass ? undefined : "error")] },
    { id: "decision", title: "Launch decision", state: { phase: "decision", day: final.day },
      description: decision === "ship" ? "Primary metric significantly up and the guardrail holds: ship B." : decision === "blocked" ? "The guardrail fails, so B is blocked regardless of conversion." : "No significant lift: keep A (or run longer with a pre-registered horizon).",
      events: [ev("dec", `Decision: ${decision}`, undefined, decision === "ship" ? undefined : decision === "blocked" ? "error" : "warning")] },
    { id: "peeking", title: `Peeking across ${NULL_EXPERIMENTS} A/A tests`, state: { phase: "peeking", day: p.days },
      description: `${NULL_EXPERIMENTS} simulated experiments where A and B are identical. Checking once at day ${p.days}: ${(nulls.fixedFpr * 100).toFixed(1)}% false positives. Checking every day and stopping at the first p < 0.05: ${(nulls.peekFpr * 100).toFixed(1)}%.`,
      events: [ev("fpr", `False-positive rate: fixed ${f(nulls.fixedFpr, 3)}, peeking ${f(nulls.peekFpr, 3)}`, `\\text{FPR}_{\\text{fixed}} = ${f(nulls.fixedFpr, 3)},\\; \\text{FPR}_{\\text{peek}} = ${f(nulls.peekFpr, 3)}`, nulls.peekFpr > 0.08 ? "warning" : undefined)] },
  ];
  return { params: p, all, rows, final, fullFinal, stoppedEarly, guard, nulls, nNeeded, daysNeeded, decision, steps };
}
export type OnlineSim = ReturnType<typeof simulate>;
