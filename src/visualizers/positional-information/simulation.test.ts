import { describe, expect, it } from "vitest";
import { DEFAULT_PARAMS, encode, orderCheck, ropeDot, rotate, simulate, sinusoidalPE } from "./simulation";

describe("positional information", () => {
  it("sinusoidal PE matches the formula by hand", () => {
    expect(sinusoidalPE(0)).toEqual([0, 1, 0, 1]);
    const p = sinusoidalPE(2); // d = 4: frequencies 1 and 1/100
    expect(p[0]).toBeCloseTo(Math.sin(2), 12);
    expect(p[1]).toBeCloseTo(Math.cos(2), 12);
    expect(p[2]).toBeCloseTo(Math.sin(0.02), 12);
    expect(p[3]).toBeCloseTo(Math.cos(0.02), 12);
  });

  it("attention without positions is order-blind (permutation equivariant)", () => {
    for (const preset of ["dog", "alice"] as const) {
      const r = orderCheck(preset, 0);
      expect(r.maxDiff).toBeLessThan(1e-12);
    }
  });

  it("adding positional encodings makes outputs depend on order", () => {
    expect(orderCheck("dog", 1).maxDiff).toBeGreaterThan(0.01);
    expect(orderCheck("dog", 2).maxDiff).toBeGreaterThan(orderCheck("dog", 0.2).maxDiff);
  });

  it("encode adds scaled PE to the embeddings", () => {
    const e = encode(["dog", "bites"], 0.5);
    expect(e.Xp[1][0]).toBeCloseTo(0 + 0.5 * Math.sin(1), 12);
    expect(e.Xp[0][1]).toBeCloseTo(0.2 + 0.5 * 1, 12);
  });

  it("RoPE preserves length and depends only on the relative distance", () => {
    const r = rotate([3, 4], 1.234);
    expect(Math.hypot(r[0], r[1])).toBeCloseTo(5, 12);
    const a = ropeDot(1, 3, 0.5).dot;
    expect(ropeDot(5, 7, 0.5).dot).toBeCloseTo(a, 12);
    expect(ropeDot(1, 4, 0.5).dot).not.toBeCloseTo(a, 3);
  });

  it("produces 8 deterministic steps", () => {
    const s = simulate(DEFAULT_PARAMS);
    expect(s).toHaveLength(8);
    expect(s.map((x) => x.id)).toEqual(["words", "attn-a", "attn-b", "pe-matrix", "pe-add", "attn-pe", "rope-rotate", "rope-relative"]);
    expect(simulate(DEFAULT_PARAMS)).toEqual(s);
  });
});
