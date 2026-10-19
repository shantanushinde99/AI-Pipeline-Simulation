import { describe, expect, it } from "vitest";
import { DEFAULT_ONLINE, guardrail, normCdf, nullExperiments, runExperiment, sampleSizePerArm, simulate, twoPropTest } from "./simulation";

describe("online evaluation (A/B test)", () => {
  it("normal CDF matches known values", () => {
    expect(normCdf(0)).toBeCloseTo(0.5, 7);
    expect(normCdf(1.959964)).toBeCloseTo(0.975, 6);
    expect(normCdf(-1)).toBeCloseTo(0.158655, 6);
  });

  it("two-proportion z-test matches a hand calculation", () => {
    // A: 100/1000, B: 130/1000 → pooled 0.115, SE = sqrt(0.115·0.885·0.002) = 0.014267
    const t = twoPropTest(1000, 100, 1000, 130);
    expect(t.diff).toBeCloseTo(0.03, 12);
    expect(t.z).toBeCloseTo(0.03 / Math.sqrt(0.115 * 0.885 * 0.002), 9);
    expect(t.p).toBeCloseTo(2 * (1 - normCdf(t.z)), 9);
    expect(t.ci[0]).toBeCloseTo(0.03 - 1.96 * Math.sqrt(0.09 / 1000 + 0.13 * 0.87 / 1000), 9);
  });

  it("sample size formula: 10% base, +1.5 pt MDE needs ~6.7k per arm", () => {
    expect(sampleSizePerArm(0.1, 0.015)).toBeGreaterThan(6600);
    expect(sampleSizePerArm(0.1, 0.015)).toBeLessThan(6800);
    expect(sampleSizePerArm(0.1, 0.03)).toBeLessThan(sampleSizePerArm(0.1, 0.015) / 3);
  });

  it("is deterministic and cumulative counts add up", () => {
    const a = runExperiment(DEFAULT_ONLINE), b = runExperiment(DEFAULT_ONLINE);
    expect(a).toEqual(b);
    const last = a.at(-1)!;
    expect(last.nA + last.nB).toBe(DEFAULT_ONLINE.usersPerDay * DEFAULT_ONLINE.days);
    expect(a.length).toBe(DEFAULT_ONLINE.days);
  });

  it("the CI narrows as samples grow", () => {
    const rows = runExperiment(DEFAULT_ONLINE);
    const w = (i: number) => rows[i].test.ci[1] - rows[i].test.ci[0];
    expect(w(13)).toBeLessThan(w(0) / 3);
  });

  it("peeking inflates the false-positive rate of A/A tests", () => {
    const n = nullExperiments({ ...DEFAULT_ONLINE, trueLift: 0 });
    expect(n.peekFpr).toBeGreaterThan(n.fixedFpr * 2);
    expect(n.fixedFpr).toBeLessThan(0.1);
    expect(n.cumulative.at(-1)).toBeCloseTo(n.peekFpr, 12);
  });

  it("peeking stops an A/A test early on a false positive; the guardrail can block a launch", () => {
    const aa = simulate({ ...DEFAULT_ONLINE, trueLift: 0, peeking: true, seed: 8 });
    expect(aa.stoppedEarly).toBe(true);
    expect(aa.final.test.p).toBeLessThan(0.05);
    expect(aa.fullFinal.test.p).toBeGreaterThan(0.05);
    expect(simulate(DEFAULT_ONLINE).decision).toBe("ship");
    expect(simulate({ ...DEFAULT_ONLINE, addedLatency: 180 }).decision).toBe("blocked");
    expect(guardrail(runExperiment({ ...DEFAULT_ONLINE, addedLatency: 0 })).pass).toBe(true);
  });
});
