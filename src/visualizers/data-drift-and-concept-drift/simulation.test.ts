import { describe, expect, it } from "vitest";
import { DEFAULT_INPUT, binShares, ksCritical, ksStat, psi, quantileEdges, simulate, truth } from "./simulation";

describe("data drift and concept drift", () => {
  it("computes PSI and KS by hand", () => {
    // (0.3-0.2)ln(1.5) + (0.2-0.3)ln(2/3) = 0.1·0.405465 + 0.1·0.405465
    expect(psi([0.2, 0.3, 0.5], [0.3, 0.2, 0.5])).toBeCloseTo(0.2 * Math.log(1.5), 10);
    expect(psi([0.25, 0.25, 0.5], [0.25, 0.25, 0.5])).toBe(0);
    expect(ksStat([1, 2, 3, 4], [3, 4, 5, 6])).toBeCloseTo(0.5, 10);
    expect(ksCritical(240, 120)).toBeCloseTo(1.36 * Math.sqrt(360 / 28800), 10);
  });

  it("bins on reference quantiles so each bin holds an equal share", () => {
    const v = Array.from({ length: 100 }, (_, i) => i);
    const e = quantileEdges(v, 5);
    expect(e).toEqual([20, 40, 60, 80]);
    expect(binShares(v, e)).toEqual([0.2, 0.2, 0.2, 0.2, 0.2]);
  });

  it("rotating the concept flips labels of fixed points", () => {
    expect(truth([0, 1], 0)).toBe(1);
    expect(truth([1, 1.5], 0)).toBe(1);
    expect(truth([1, 1.5], 55)).toBe(0);
  });

  it("covariate shift: PSI fires early, accuracy drops later", () => {
    const run = simulate({ ...DEFAULT_INPUT, kind: "covariate" });
    expect(run.steps).toHaveLength(10);
    expect(run.refAcc).toBeGreaterThan(0.9);
    expect(run.firstFeatureAlert).toBeLessThanOrEqual(3);
    const last = run.steps.at(-1)!.state.stats.at(-1)!;
    expect(last.acc).toBeLessThan(0.8);
    expect(run.firstLabelAlert!).toBeGreaterThan(run.firstFeatureAlert!);
  });

  it("concept drift: PSI never fires, the label monitor catches it after the delay", () => {
    const run = simulate({ ...DEFAULT_INPUT, kind: "concept" });
    expect(run.firstFeatureAlert).toBeUndefined();
    const stats = run.steps.at(-1)!.state.stats;
    stats.forEach((s) => expect(Math.max(...s.psi)).toBeLessThan(0.2));
    const firstBad = stats.find((s) => s.acc < DEFAULT_INPUT.accThreshold)!.t;
    expect(run.firstLabelAlert).toBe(firstBad + DEFAULT_INPUT.labelDelay);
    const noDelay = simulate({ ...DEFAULT_INPUT, kind: "concept", labelDelay: 0 });
    expect(noDelay.firstLabelAlert).toBe(firstBad);
  });

  it("no drift: neither detector fires; runs are deterministic", () => {
    const run = simulate({ ...DEFAULT_INPUT, kind: "none" });
    expect(run.firstFeatureAlert).toBeUndefined();
    expect(run.firstLabelAlert).toBeUndefined();
    expect(simulate(DEFAULT_INPUT).steps).toEqual(simulate(DEFAULT_INPUT).steps.map((s) => s));
  });
});
