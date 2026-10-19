import { describe, expect, it } from "vitest";
import { DEFAULT_CM, confusionMatrix, costMatrix, makeExamples, perClass, rowNormalize, simulate, totalCost, type Example } from "./simulation";

const ex = (pairs: [number, number][]): Example[] => pairs.map(([actual, predicted], i) => ({ id: i + 1, actual, predicted }));

describe("confusion matrix", () => {
  it("counts rows = actual, columns = predicted", () => {
    const m = confusionMatrix(ex([[0, 0], [0, 1], [1, 1], [1, 1], [1, 0], [0, 0]]), 2);
    expect(m).toEqual([[2, 1], [1, 2]]);
  });

  it("per-class precision is column-wise and recall is row-wise", () => {
    const m = [[5, 2, 1], [1, 6, 3], [0, 2, 8]];
    const pc = perClass(m);
    expect(pc[0].precision).toBeCloseTo(5 / 6, 12);
    expect(pc[0].recall).toBeCloseTo(5 / 8, 12);
    expect(pc[2].precision).toBeCloseTo(8 / 12, 12);
    expect(pc[2].recall).toBeCloseTo(8 / 10, 12);
  });

  it("row-normalized rows sum to 1 and an empty column gives undefined precision", () => {
    const norm = rowNormalize([[3, 1], [2, 2]]);
    norm.forEach((r) => expect(r[0] + r[1]).toBeCloseTo(1, 12));
    expect(Number.isNaN(perClass([[0, 4], [0, 6]])[0].precision)).toBe(true);
  });

  it("weights errors by the cost matrix", () => {
    const c = costMatrix(2, 10, 1); // missing class 0 costs 10
    expect(c).toEqual([[0, 10], [1, 0]]);
    expect(totalCost([[5, 2], [3, 10]], c)).toBe(2 * 10 + 3 * 1);
  });

  it("is deterministic, keeps exact class counts, and lands every example", () => {
    const a = makeExamples({ ...DEFAULT_CM, imbalanced: true });
    expect(a).toEqual(makeExamples({ ...DEFAULT_CM, imbalanced: true }));
    expect(a.filter((e) => e.actual === 0)).toHaveLength(3); // 15% of 20
    const sim = simulate({ ...DEFAULT_CM, mode: "multi", n: 24 });
    expect(sim.m.flat().reduce((s, v) => s + v, 0)).toBe(24);
    expect(sim.steps.map((s) => s.id)).toEqual(["list", "first", "rest", "diagonal", "precision", "recall", "normalize", "cost"]);
  });

  it("majority-class fault: high accuracy under imbalance, zero recall for the critical class", () => {
    const sim = simulate({ ...DEFAULT_CM, imbalanced: true, majorityFault: true });
    expect(sim.accuracy).toBeCloseTo(17 / 20, 12);
    expect(sim.pc[0].recall).toBe(0);
    expect(sim.total).toBe(3 * DEFAULT_CM.costMiss);
  });

  it("better model quality puts more examples on the diagonal", () => {
    const lo = simulate({ ...DEFAULT_CM, quality: 0.3, n: 36 }).correct;
    const hi = simulate({ ...DEFAULT_CM, quality: 0.95, n: 36 }).correct;
    expect(hi).toBeGreaterThan(lo);
  });
});
