import { describe, expect, it } from "vitest";
import { PRESETS, dcgAtK, metricsAt, ndcgAtK, precisionAtK, recallAtK, reciprocalRank, simulate, type Grade } from "./simulation";

const rank7 = PRESETS.find((p) => p.id === "rank7")!;

describe("top-k ranking metrics", () => {
  it("matches hand calculations for precision, recall and reciprocal rank", () => {
    const g: Grade[] = [0, 2, 0, 1, 2];
    expect(precisionAtK(g, 2, 1)).toBe(0.5);
    expect(precisionAtK(g, 5, 2)).toBeCloseTo(2 / 5, 12);
    expect(recallAtK(g, 4, 3, 1)).toBeCloseTo(2 / 3, 12);
    expect(reciprocalRank(g, 5, 1)).toBe(0.5);
    expect(reciprocalRank(g, 1, 1)).toBe(0);
    expect(reciprocalRank(g, 5, 2)).toBe(0.5);
  });

  it("computes DCG and nDCG with graded gains", () => {
    // DCG@3 of [2,0,1] = 3/log2(2) + 0 + 1/log2(4) = 3.5; ideal [2,1,0] = 3 + 1/log2(3)
    expect(dcgAtK([2, 0, 1], 3)).toBeCloseTo(3.5, 12);
    expect(ndcgAtK([2, 0, 1], 3)).toBeCloseTo(3.5 / (3 + 1 / Math.log2(3)), 12);
    expect(ndcgAtK([0, 0], 2)).toBe(0);
  });

  it("the rank-7 preset only reaches full recall at k = 7", () => {
    expect(rank7.candidates[6].grade).toBe(2);
    expect(metricsAt(rank7, 6, 2).recall).toBe(0);
    const m7 = metricsAt(rank7, 7, 2);
    expect(m7.recall).toBe(1);
    expect(m7.rr).toBeCloseTo(1 / 7, 12);
    // DCG@7 = 1/log2 3 + 1/log2 4 + 3/log2 8; IDCG@7 = 3 + 1/log2 3 + 1/log2 4
    expect(m7.ndcg).toBeCloseTo((1 / Math.log2(3) + 0.5 + 1) / (3 + 1 / Math.log2(3) + 0.5), 10);
  });

  it("a relevant chunk missing from the list caps recall below 1", () => {
    const missed = PRESETS.find((p) => p.id === "missed")!;
    expect(metricsAt(missed, 10, 2).recall).toBe(0.5);
    expect(metricsAt(missed, 10, 2).ndcg).toBeLessThan(1);
  });

  it("context tokens grow monotonically with k", () => {
    const toks = rank7.candidates.map((_, i) => metricsAt(rank7, i + 1, 1).tokens);
    toks.slice(1).forEach((t, i) => expect(t).toBeGreaterThan(toks[i]));
  });

  it("produces list + one step per k + pick, deterministically, and responds to parameters", () => {
    const a = simulate({ presetId: "rank7", threshold: 2, chosenK: 4 });
    expect(a).toHaveLength(12);
    expect(a.map((s) => s.state.k)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 4]);
    expect(simulate({ presetId: "rank7", threshold: 2, chosenK: 4 })).toEqual(a);
    const lenient = simulate({ presetId: "rank7", threshold: 1, chosenK: 4 });
    expect(lenient.at(-1)!.state.m.recall).not.toBe(a.at(-1)!.state.m.recall);
    expect(simulate({ presetId: "rank7", threshold: 2, chosenK: 99 }).at(-1)!.state.k).toBe(10);
  });
});
