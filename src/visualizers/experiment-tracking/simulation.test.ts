import { describe, expect, it } from "vitest";
import { DEFAULT_ET, bestBy, makeRun, simulateTracking, standardError, trueAccuracy } from "./simulation";

describe("experiment tracking", () => {
  it("true accuracy surface matches a hand calculation", () => {
    // lr = 0.01 → (log10 0.01 + 2.2)^2 = 0.04 → 0.9 − 0.002; depth 5 → 0.003·4 = 0.012
    expect(trueAccuracy(0.01, 5)).toBeCloseTo(0.9 - 0.002 - 0.012, 12);
    expect(standardError(0.9, 100)).toBeCloseTo(0.03, 12);
  });

  it("validation accuracy is a whole number of correct answers out of n", () => {
    const r = makeRun(1, 0.006, 7, 42, 200);
    expect(Number.isInteger(Math.round(r.valAcc * 200 * 1e6) / 1e6)).toBe(true);
    expect(makeRun(1, 0.006, 7, 42, 200)).toEqual(r);
  });

  it("produces define → runs → best → compare → test", () => {
    const { steps, all } = simulateTracking(DEFAULT_ET);
    expect(steps).toHaveLength(all.length + 4);
    expect(steps[0].id).toBe("define");
    expect(steps.slice(-3).map((s) => s.id)).toEqual(["best", "compare", "test"]);
    // runs accumulate one at a time and the running best is monotone
    steps.slice(1, -3).forEach((s, i) => expect(s.state.runs).toHaveLength(i + 1));
    const bests = steps.slice(1, -3).map((s) => s.state.best!.valAcc);
    bests.forEach((b, i) => i && expect(b).toBeGreaterThanOrEqual(bests[i - 1]));
  });

  it("the default sweep shows the best-by-chance trap", () => {
    const r = simulateTracking(DEFAULT_ET);
    expect(r.best.id).not.toBe(r.bestByTest.id);
    expect(r.best.valAcc - r.best.testAcc).toBeGreaterThan(0.02);
    expect(bestBy(r.all, "valAcc")).toBe(r.best);
  });

  it("a larger validation set shrinks the winner's optimism", () => {
    const optimism = (valSize: number) => { const r = simulateTracking({ ...DEFAULT_ET, valSize }); return r.best.valAcc - r.best.testAcc; };
    expect(optimism(10000)).toBeLessThan(optimism(200));
  });

  it("seed mode repeats one config and reports its spread", () => {
    const r = simulateTracking({ ...DEFAULT_ET, mode: "seeds" });
    expect(new Set(r.all.map((x) => `${x.lr}/${x.depth}`)).size).toBe(1);
    expect(new Set(r.all.map((x) => x.seed)).size).toBe(r.all.length);
    expect(r.seedStats!.max - r.seedStats!.min).toBeGreaterThan(0);
  });

  it("compare uses the requested runs", () => {
    const r = simulateTracking(DEFAULT_ET, [3, 5]);
    expect(r.steps.find((s) => s.id === "compare")!.state.compare!.map((x) => x.id)).toEqual([3, 5]);
  });
});
