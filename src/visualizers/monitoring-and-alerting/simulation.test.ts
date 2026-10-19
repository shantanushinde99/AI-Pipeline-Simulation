import { describe, expect, it } from "vitest";
import { MINUTES, PRESETS, burnRate, evaluate, generateMetrics, nextState, simulate, summarize, type Window } from "./simulation";

const w = (t: number, errors: number): Window => ({ t, latency: 400, errors, requests: 1000, errorRate: errors / 1000, quality: 0.9 });

describe("monitoring and alerting", () => {
  it("computes burn rate by hand", () => {
    const ws = [w(0, 2), w(1, 3), w(2, 30), w(3, 35)];
    // last 2 windows: 65 errors / 2000 requests = 3.25%; SLO 99.5% allows 0.5% → burn 6.5
    expect(burnRate(ws, 3, 2, 99.5).rate).toBeCloseTo(6.5, 10);
    // lookback longer than history uses what exists
    expect(burnRate(ws, 0, 5, 99.5).rate).toBeCloseTo(0.4, 10);
  });

  it("walks the alert state machine ok → pending → firing → resolved → ok", () => {
    let s = { state: "ok" as const as "ok" | "pending" | "firing" | "resolved", streak: 0 };
    const seen: string[] = [];
    for (const breach of [true, true, true, false, false]) {
      s = nextState(s.state, s.streak, breach, 3);
      seen.push(s.state);
    }
    expect(seen).toEqual(["pending", "pending", "firing", "resolved", "ok"]);
    expect(nextState("ok", 0, true, 1).state).toBe("firing"); // static threshold fires immediately
  });

  it("is deterministic and produces setup + 10 chunks + summary", () => {
    const input = { scenario: "error-outage" as const, seed: 7, spikes: true, config: PRESETS.useful };
    const a = simulate(input), b = simulate(input);
    expect(a).toEqual(b);
    expect(a).toHaveLength(12);
    expect(a.map((s) => s.state.upTo)).toEqual([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 60]);
  });

  it("budget accounting matches the summed errors", () => {
    const ws = generateMetrics("error-outage", 7, false);
    const ev = evaluate(ws, PRESETS.useful);
    const total = ws.reduce((s, x) => s + x.errors, 0);
    expect(ev.at(-1)!.budgetUsed).toBe(total);
    expect(ev[0].budgetTotal).toBeCloseTo(0.005 * 1000 * MINUTES, 6);
    expect(ev.at(-1)!.budgetLeft).toBeLessThan(0); // the outage blows the budget
  });

  it("noisy rules page on healthy spikes; useful rules stay quiet", () => {
    const ws = generateMetrics("healthy-spiky", 7, true);
    const noisy = summarize(evaluate(ws, PRESETS.noisy), "healthy-spiky");
    const useful = summarize(evaluate(ws, PRESETS.useful), "healthy-spiky");
    expect(noisy.pages).toBeGreaterThan(3);
    expect(noisy.noise).toBe(noisy.pages);
    expect(useful.pages).toBe(0);
  });

  it("only a quality rule catches the silent quality regression", () => {
    const ws = generateMetrics("quality-regression", 7, true);
    const withoutQ = summarize(evaluate(ws, { ...PRESETS.useful, qualityOn: false }), "quality-regression");
    const withQ = summarize(evaluate(ws, PRESETS.useful), "quality-regression");
    expect(withoutQ.minutesToDetect).toBeUndefined();
    const noisy = summarize(evaluate(ws, PRESETS.noisy), "quality-regression");
    expect(noisy.pages).toBeGreaterThan(0); // it pages, but only about noise
    expect(noisy.minutesToDetect).toBeUndefined();
    expect(withQ.detectedBy).toBe("quality");
    expect(withQ.minutesToDetect).toBe(PRESETS.useful.qualityFor - 1);
  });

  it("the burn-rate rule catches the outage; a longer for-duration delays latency alerts", () => {
    const out = summarize(evaluate(generateMetrics("error-outage", 7, true), PRESETS.useful), "error-outage");
    expect(out.detectedBy).toBe("errors");
    const slow = generateMetrics("slow-dependency", 7, false);
    const fast = summarize(evaluate(slow, { ...PRESETS.useful, latencyFor: 1 }), "slow-dependency");
    const patient = summarize(evaluate(slow, { ...PRESETS.useful, latencyFor: 5 }), "slow-dependency");
    expect(patient.minutesToDetect! - fast.minutesToDetect!).toBe(4);
  });
});
