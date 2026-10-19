import { describe, expect, it } from "vitest";
import { DEFAULT_PARAMS, DEFAULT_WEIGHTS, SWAP, TOKENS, compute, setEntry, simulate } from "./simulation";

describe("query, key and value", () => {
  it("projects the query with a hand-checkable matrix-vector product", () => {
    const c = compute(DEFAULT_PARAMS); // query = "drank", x = [0, 1, 0, 0.2]
    // q0 = 1·1, q1 = 1·0.2, q2 = 1·0.2, q3 = 2·1
    expect(c.q.map((v) => +v.toFixed(10))).toEqual([1, 0.2, 0.2, 2]);
    // k_tea = W_K · [0, 0, 0.2, 1] = [0, 0, 0.3, 2]
    expect(c.K[3].map((v) => +v.toFixed(10))).toEqual([0, 0, 0.3, 2]);
  });

  it("scores, scaling and softmax match a hand calculation", () => {
    const c = compute(DEFAULT_PARAMS);
    // q·k_tea = 0.2·0.3 + 2·2 = 4.06
    expect(c.scores[3]).toBeCloseTo(4.06, 10);
    expect(c.scaled[3]).toBeCloseTo(2.03, 10);
    const e = c.scaled.map(Math.exp);
    expect(c.weights[3]).toBeCloseTo(e[3] / e.reduce((a, b) => a + b, 0), 12);
    expect(TOKENS[c.top]).toBe("tea");
  });

  it("output is the weighted sum of values and the running partial sums end at it", () => {
    const c = compute(DEFAULT_PARAMS);
    const manual = [0, 1, 2, 3].map((d) => c.weights.reduce((s, w, j) => s + w * c.V[j][d], 0));
    manual.forEach((v, d) => expect(c.out[d]).toBeCloseTo(v, 12));
    c.partial.at(-1)!.forEach((v, d) => expect(v).toBeCloseTo(c.out[d], 12));
  });

  it("swapping keys moves attention but keeps values in place", () => {
    const a = compute(DEFAULT_PARAMS);
    const b = compute({ ...DEFAULT_PARAMS, swapKeys: true });
    expect(b.weights[SWAP[0]]).toBeCloseTo(a.weights[SWAP[1]], 12);
    expect(TOKENS[b.top]).toBe("hot");
    expect(b.V).toEqual(a.V);
    expect(b.out).not.toEqual(a.out);
  });

  it("editing one weight-matrix entry changes the outcome; setEntry clamps and does not mutate", () => {
    const w = setEntry(DEFAULT_WEIGHTS, "WQ", 3, 1, 0);
    expect(DEFAULT_WEIGHTS.WQ[3][1]).toBe(2);
    expect(setEntry(DEFAULT_WEIGHTS, "WK", 0, 0, 99).WK[0][0]).toBe(5);
    const b = compute({ ...DEFAULT_PARAMS, weights: w });
    expect(b.weights[3]).toBeLessThan(compute(DEFAULT_PARAMS).weights[3]);
  });

  it("produces 8 ordered, deterministic steps", () => {
    const s = simulate(DEFAULT_PARAMS);
    expect(s.map((x) => x.id)).toEqual(["embed", "project-q", "project-k", "project-v", "dot", "scale", "softmax", "mix"]);
    expect(simulate(DEFAULT_PARAMS)).toEqual(s);
  });
});
