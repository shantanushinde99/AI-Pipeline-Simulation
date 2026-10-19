import { describe, expect, it } from "vitest";
import { DEFAULTS, LOGITS, VOCAB, pipeline, sampleIndex, simulate, softmaxT, topKMask, topPMask } from "./simulation";

describe("temperature and sampling", () => {
  it("softmax matches a hand calculation and sums to 1", () => {
    const { probs } = softmaxT([0, Math.log(3)], 1);
    expect(probs[0]).toBeCloseTo(0.25, 12);
    expect(probs[1]).toBeCloseTo(0.75, 12);
    const p = softmaxT(LOGITS, 1).probs;
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(p[0]).toBeCloseTo(0.4341, 3);
  });

  it("temperature sharpens (T<1) and flattens (T>1); T=0 is greedy", () => {
    const p1 = softmaxT(LOGITS, 1).probs[0];
    expect(softmaxT(LOGITS, 0.3).probs[0]).toBeGreaterThan(p1);
    expect(softmaxT(LOGITS, 2).probs[0]).toBeLessThan(p1);
    expect(softmaxT(LOGITS, 0).probs).toEqual([1, 0, 0, 0, 0, 0, 0, 0]);
    expect(softmaxT(LOGITS, 0.01).probs[0]).toBeCloseTo(1, 10);
  });

  it("top-k and top-p keep the right tokens", () => {
    const p = softmaxT(LOGITS, 1).probs;
    expect(topKMask(p, 3)).toEqual([true, true, true, false, false, false, false, false]);
    expect(topKMask(p, 0).every(Boolean)).toBe(true);
    // cumulative: .43, .65, .79, .88, .94 → p = 0.9 keeps 5
    expect(topPMask(p, 0.9).mask.filter(Boolean)).toHaveLength(5);
    expect(topPMask(p, 0.01).mask.filter(Boolean)).toHaveLength(1);
  });

  it("renormalizes survivors and samples by inverse CDF", () => {
    const pl = pipeline({ ...DEFAULTS, topK: 2 });
    expect(pl.final.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(pl.final.slice(2).every((v) => v === 0)).toBe(true);
    expect(sampleIndex([0.2, 0.5, 0.3], 0.1)).toBe(0);
    expect(sampleIndex([0.2, 0.5, 0.3], 0.69)).toBe(1);
    expect(sampleIndex([0.2, 0.5, 0.3], 0.7)).toBe(2);
    expect(sampleIndex([0, 1, 0], 0.999999)).toBe(1);
  });

  it("is deterministic per seed, converges with more draws, and is greedy at T=0", () => {
    const a = pipeline(DEFAULTS);
    expect(pipeline(DEFAULTS)).toEqual(a);
    expect(pipeline({ ...DEFAULTS, seed: 8 }).draws).not.toEqual(a.draws);
    const big = pipeline({ ...DEFAULTS, n: 2000 });
    expect(big.checkpoints.at(-1)!.tv).toBeLessThan(pipeline({ ...DEFAULTS, n: 20 }).checkpoints.at(-1)!.tv);
    expect(big.checkpoints.at(-1)!.tv).toBeLessThan(0.05);
    const g = pipeline({ ...DEFAULTS, T: 0 });
    expect(new Set(g.draws)).toEqual(new Set([0]));
    expect(VOCAB[g.pick]).toBe("sunny");
  });

  it("produces the 9 pipeline steps in order", () => {
    expect(simulate(DEFAULTS).map((s) => s.id)).toEqual(["logits", "scale", "exp", "normalize", "topk", "topp", "renorm", "sample", "repeat"]);
    expect(pipeline({ ...DEFAULTS, T: Number.NaN }).cfg.T).toBe(1);
  });
});
