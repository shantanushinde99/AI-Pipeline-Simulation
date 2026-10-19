import { describe, expect, it } from "vitest";
import { DEFAULTS, knnAccuracy, makeData, simulate, type Row } from "./simulation";

describe("train / validation / test splits", () => {
  it("k-NN accuracy matches a hand calculation", () => {
    const train: Row[] = [{ id: 0, x: [0, 0], y: 0 }, { id: 1, x: [0.1, 0], y: 0 }, { id: 2, x: [1, 1], y: 1 }];
    const evalRows: Row[] = [{ id: 3, x: [0.05, 0], y: 0 }, { id: 4, x: [0.9, 1], y: 0 }];
    expect(knnAccuracy(train, evalRows, 1)).toBe(0.5); // first right, second wrong
    expect(knnAccuracy(train, evalRows, 3)).toBe(1); // majority of all three is 0
  });

  it("buckets are disjoint and sized from the fractions", () => {
    const r = simulate(DEFAULTS);
    expect(r.train.length + r.val.length + r.test.length).toBe(90);
    expect(r.test.length).toBe(18);
    expect(r.val.length).toBe(18);
    const ids = new Set([...r.train, ...r.val, ...r.test].map((x) => x.id));
    expect(ids.size).toBe(90);
  });

  it("training accuracy flatters k = 1", () => {
    const r = simulate(DEFAULTS);
    expect(r.scores[0].train).toBe(1);
    expect(r.scores[0].val).toBeLessThan(1);
  });

  it("tuning on test can only raise the reported score and inflates it over fresh data", () => {
    const honest = simulate(DEFAULTS);
    const leaky = simulate({ ...DEFAULTS, tuneOnTest: true });
    expect(leaky.reported).toBeGreaterThanOrEqual(honest.reported);
    expect(leaky.reported - leaky.freshAcc).toBeGreaterThan(0.1); // 94.4% reported vs 76.7% fresh for seed 4
  });

  it("k-fold gives every dev row exactly one turn as validation", () => {
    const r = simulate({ ...DEFAULTS, kfold: true, folds: 5 });
    const counts = [0, 0, 0, 0, 0];
    r.dev.forEach((row) => counts[r.fold[row.id]]++);
    expect(counts.reduce((a, b) => a + b, 0)).toBe(r.dev.length);
    expect(Math.max(...counts) - Math.min(...counts)).toBeLessThanOrEqual(1);
    expect(r.steps.filter((s) => s.state.phase === "fold")).toHaveLength(5);
    r.scores.forEach((s) => expect(s.val).toBeCloseTo(s.folds.reduce((a, b) => a + b, 0) / 5, 12));
  });

  it("is deterministic and the seed changes the data", () => {
    expect(simulate(DEFAULTS)).toEqual(simulate(DEFAULTS));
    expect(makeData(1, 10, 0.1)).not.toEqual(makeData(2, 10, 0.1));
    expect(simulate(DEFAULTS).steps.map((s) => s.id)).toEqual(["data", "shuffle", "split", "train", "validate", "select", "test"]);
  });
});
