import { describe, expect, it } from "vitest";
import { DEFAULT_PARAMS, X0, gelu, layerNorm, run, simulate } from "./simulation";

describe("transformer block", () => {
  it("LayerNorm matches a hand calculation", () => {
    // x = [1, 2, 3, 6]: mean 3, variance (4 + 1 + 0 + 9) / 4 = 3.5
    const r = layerNorm([1, 2, 3, 6]);
    expect(r.mu).toBe(3);
    expect(r.variance).toBe(3.5);
    expect(r.y[0]).toBeCloseTo(-2 / Math.sqrt(3.5 + 1e-5), 12);
    expect(r.y.reduce((a, b) => a + b, 0)).toBeCloseTo(0, 12);
  });

  it("GELU keeps positives, squashes negatives", () => {
    expect(gelu(0)).toBe(0);
    expect(gelu(3)).toBeCloseTo(2.9964, 3);
    expect(gelu(-3)).toBeCloseTo(-0.0036, 3);
    expect(gelu(1)).toBeCloseTo(0.8412, 3);
  });

  it("records 10 stages per block in order, with residual adds equal to stream + branch", () => {
    const { stages } = run(DEFAULT_PARAMS);
    expect(stages.map((s) => s.phase)).toEqual(["input", "ln1", "attn", "add1", "ln2", "up", "gelu", "down", "add2", "output"]);
    const add1 = stages[3];
    add1.after.forEach((v, i) => v.forEach((x, d) => expect(x).toBeCloseTo(add1.before[i][d] + add1.branch![i][d], 12)));
    expect(stages[5].after[0]).toHaveLength(8);
    expect(stages[0].after).toEqual(X0);
    expect(run({ blocks: 2, residual: true }).stages).toHaveLength(18);
  });

  it("each attention row sums to 1 and LN output has mean 0 / variance ≈ 1", () => {
    const { stages } = run(DEFAULT_PARAMS);
    for (const row of stages[2].attn!) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    for (const v of stages[1].after) {
      expect(v.reduce((a, b) => a + b, 0) / 4).toBeCloseTo(0, 12);
      expect(v.reduce((a, b) => a + b * b, 0) / 4).toBeCloseTo(1, 4);
    }
  });

  it("removing residuals degrades the signal: tokens collapse and lose their input", () => {
    const on = run({ blocks: 2, residual: true }).health;
    const off = run({ blocks: 2, residual: false }).health;
    expect(on.meanKeep).toBeGreaterThan(0.9);
    expect(off.meanKeep).toBeLessThan(0.6);
    expect(off.tokenSimilarity).toBeGreaterThan(0.95);
    expect(on.tokenSimilarity).toBeLessThan(0.2);
  });

  it("simulate is deterministic and step count follows the block count", () => {
    expect(simulate(DEFAULT_PARAMS)).toEqual(simulate(DEFAULT_PARAMS));
    expect(simulate({ blocks: 2, residual: false })).toHaveLength(18);
  });
});
