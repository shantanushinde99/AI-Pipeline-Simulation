import { describe, expect, it } from "vitest";
import { attend, softmax } from "./attention";
import { DEFAULT_PARAMS, TOKENS, compute, simulate } from "./simulation";

describe("self-attention", () => {
  it("softmax matches a hand calculation and handles masks", () => {
    const w = softmax([0, Math.log(3)]);
    expect(w[0]).toBeCloseTo(0.25, 12);
    expect(w[1]).toBeCloseTo(0.75, 12);
    expect(softmax([1, -Infinity])).toEqual([1, 0]);
    expect(softmax([-Infinity, -Infinity])).toEqual([0, 0]);
  });

  it("scaled dot-product attention matches a 2-token hand calculation", () => {
    // q·k = [2, 0] / √2 → softmax → weights; out = w0·v0 + w1·v1
    const r = attend([[1, 1]], [[1, 1], [0, 0]], [[1, 0], [0, 1]]);
    expect(r.scores[0]).toEqual([2, 0]);
    const w0 = Math.exp(2 / Math.SQRT2) / (Math.exp(2 / Math.SQRT2) + 1);
    expect(r.weights[0][0]).toBeCloseTo(w0, 12);
    expect(r.out[0][0]).toBeCloseTo(w0, 12);
    expect(r.out[0][1]).toBeCloseTo(1 - w0, 12);
  });

  it('"it" attends most to "animal" with the designed weights', () => {
    const c = compute(DEFAULT_PARAMS);
    expect(TOKENS[c.i]).toBe("it");
    expect(TOKENS[c.top]).toBe("animal");
    expect(c.row.weights[c.top]).toBeGreaterThan(0.45);
    for (const row of c.weights) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it("output is the weighted sum of value vectors", () => {
    const c = compute(DEFAULT_PARAMS);
    const manual = [0, 1, 2, 3].map((d) => c.row.weights.reduce((s, w, j) => s + w * c.V[j][d], 0));
    manual.forEach((v, d) => expect(c.row.out[d]).toBeCloseTo(v, 12));
    // "it" borrows the animate feature from "animal"
    expect(c.row.out[0]).toBeGreaterThan(c.X[c.i][0]);
  });

  it("removing 1/√d makes the distribution sharper; causal mask zeroes the future", () => {
    const on = compute(DEFAULT_PARAMS);
    const off = compute({ ...DEFAULT_PARAMS, scaled: false });
    expect(off.entropy).toBeLessThan(on.entropy);
    const causal = compute({ ...DEFAULT_PARAMS, causal: true });
    expect(causal.row.weights.slice(8).every((w) => w === 0)).toBe(true);
    expect(TOKENS[causal.top]).toBe("animal");
  });

  it("produces 8 deterministic steps; random weights change the outcome", () => {
    const a = simulate(DEFAULT_PARAMS);
    expect(a.map((s) => s.id)).toEqual(["tokens", "query", "scores", "scale", "softmax", "lines", "mix", "heatmap"]);
    expect(simulate(DEFAULT_PARAMS)).toEqual(a);
    const r = compute({ ...DEFAULT_PARAMS, weights: "random" });
    expect(r.row.weights[1]).not.toBeCloseTo(compute(DEFAULT_PARAMS).row.weights[1], 2);
  });
});
