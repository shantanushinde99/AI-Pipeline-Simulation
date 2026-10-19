import { describe, expect, it } from "vitest";
import { averagePrecision, makeExamples, metricsAt, prCurve, rocAuc, rocCurve, simulate, DEFAULT_CLS, type Example } from "./simulation";

const toy: Example[] = [
  { id: 0, score: 0.9, y: 1 },
  { id: 1, score: 0.8, y: 0 },
  { id: 2, score: 0.7, y: 1 },
  { id: 3, score: 0.4, y: 1 },
  { id: 4, score: 0.2, y: 0 },
];

describe("classification metrics", () => {
  it("matches a hand count at threshold 0.5", () => {
    const m = metricsAt(toy, 0.5);
    expect([m.tp, m.fp, m.fn, m.tn]).toEqual([2, 1, 1, 1]);
    expect(m.precision).toBeCloseTo(2 / 3, 12);
    expect(m.recall).toBeCloseTo(2 / 3, 12);
    expect(m.f1).toBeCloseTo(2 / 3, 12);
    expect(m.accuracy).toBeCloseTo(3 / 5, 12);
    expect(m.specificity).toBeCloseTo(1 / 2, 12);
  });

  it("precision is undefined (not 0) when nothing is predicted positive", () => {
    const m = metricsAt(toy, 2);
    expect(m.tp + m.fp).toBe(0);
    expect(Number.isNaN(m.precision)).toBe(true);
    expect(m.recall).toBe(0);
  });

  it("ROC AUC equals the pairwise ranking probability", () => {
    // positives 0.9, 0.7, 0.4 vs negatives 0.8, 0.2: pairs won = (0.9>0.8,0.9>0.2,0.7>0.2,0.4>0.2) = 4 of 6
    const roc = rocCurve(toy);
    expect(roc[0]).toMatchObject({ fpr: 0, tpr: 0 });
    expect(roc.at(-1)).toMatchObject({ fpr: 1, tpr: 1 });
    expect(rocAuc(roc)).toBeCloseTo(4 / 6, 12);
  });

  it("average precision matches the step-wise sum", () => {
    // thresholds 0.9 (R 1/3, P 1), 0.8 (R 1/3, P 1/2), 0.7 (R 2/3, P 2/3), 0.4 (R 1, P 3/4), 0.2 (R 1, P 3/5)
    expect(averagePrecision(prCurve(toy))).toBeCloseTo((1 / 3) * 1 + (1 / 3) * (2 / 3) + (1 / 3) * (3 / 4), 12);
  });

  it("is deterministic and respects prevalence", () => {
    const a = makeExamples({ n: 50, prevalence: 0.1, separation: 2, seed: 3 });
    expect(a).toEqual(makeExamples({ n: 50, prevalence: 0.1, separation: 2, seed: 3 }));
    expect(a.filter((e) => e.y).length).toBe(5);
  });

  it("produces nine ordered steps and better separation raises AUC", () => {
    const sim = simulate(DEFAULT_CLS);
    expect(sim.steps.map((s) => s.id)).toEqual(["scores", "threshold", "count", "precision", "recall", "summary", "roc", "pr", "baseline"]);
    expect(simulate({ ...DEFAULT_CLS, separation: 5 }).auc).toBeGreaterThan(simulate({ ...DEFAULT_CLS, separation: 0.5 }).auc);
  });

  it("imbalanced data: always-negative baseline gets high accuracy with zero recall", () => {
    const sim = simulate({ ...DEFAULT_CLS, n: 50, prevalence: 0.08 });
    expect(sim.allNeg.accuracy).toBeCloseTo(46 / 50, 12);
    expect(sim.allNeg.recall).toBe(0);
  });
});
