import { describe, expect, it } from "vitest";
import { DEFAULT_LEAK as LEAKY, PHASES, fitScaler, knnPredict, simulate } from "./simulation";

const DEFAULT_LEAK = { ...LEAKY, leakTarget: false };

describe("data-leakage simulation", () => {
  it("is deterministic and walks the 7 pipeline stages in order", () => {
    const a = simulate(DEFAULT_LEAK), b = simulate(DEFAULT_LEAK);
    expect(a.offline).toEqual(b.offline);
    expect(a.steps.map((s) => s.state.phase)).toEqual(PHASES);
  });

  it("clean pipeline: offline estimate is close to production", () => {
    const s = simulate(DEFAULT_LEAK);
    expect(Math.abs(s.offline.accuracy - s.production.accuracy)).toBeLessThan(0.08);
    expect(s.features).not.toContain("cancel_ticket_logged");
  });

  it("target leakage inflates offline accuracy while production recall collapses", () => {
    const s = simulate({ ...DEFAULT_LEAK, leakTarget: true });
    const leak = s.corr.find((c) => c.name === "cancel_ticket_logged");
    expect(leak?.r).toBeGreaterThan(0.8);
    expect(s.offline.accuracy).toBeGreaterThan(0.9);
    expect(s.production.recall).toBeLessThan(0.1);
    expect(s.offline.accuracy - s.production.accuracy).toBeGreaterThan(0.2);
  });

  it("duplicates before the split put twins in test, which 1-NN memorizes", () => {
    const clean = simulate({ ...DEFAULT_LEAK, model: "knn" });
    const dup = simulate({ ...DEFAULT_LEAK, model: "knn", duplicates: true });
    expect(clean.leakedTwins).toBe(0);
    expect(dup.leakedTwins).toBeGreaterThan(10);
    expect(dup.offline.accuracy - dup.production.accuracy).toBeGreaterThan(0.08);
  });

  it("scaler fit before split changes the statistics but barely moves this model's score", () => {
    const s = simulate({ ...DEFAULT_LEAK, scaleBeforeSplit: true });
    expect(s.scaler.mean[0]).not.toBe(s.scaler.trainOnlyMean[0]);
    expect(Math.abs(s.offline.accuracy - simulate(DEFAULT_LEAK).offline.accuracy)).toBeLessThan(0.02);
  });

  it("scaler and 1-NN match hand calculations", () => {
    const sc = fitScaler([{ x: [1], y: 0 }, { x: [3], y: 1 }], 1);
    expect(sc.mean[0]).toBe(2);
    expect(sc.std[0]).toBe(1);
    expect(knnPredict([{ x: [0, 0], y: 0 }, { x: [5, 5], y: 1 }], [4, 4])).toBe(1);
  });
});
