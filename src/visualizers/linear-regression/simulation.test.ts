import { describe, expect, it } from "vitest";
import { DEFAULT_LR, makeData, ols, residuals, simulate, sse } from "./simulation";

describe("linear regression (OLS)", () => {
  it("matches a hand calculation", () => {
    // x̄ = 2, ȳ = 10/3, Sxx = 2, Sxy = (−1)(−4/3) + 0 + (1)(5/3) = 3 → b = 1.5, a = 10/3 − 3 = 1/3
    const f = ols([{ x: 1, y: 2 }, { x: 2, y: 3 }, { x: 3, y: 5 }]);
    expect(f.xm).toBe(2);
    expect(f.ym).toBeCloseTo(10 / 3, 12);
    expect(f.sxx).toBe(2);
    expect(f.sxy).toBeCloseTo(3, 12);
    expect(f.slope).toBeCloseTo(1.5, 12);
    expect(f.intercept).toBeCloseTo(1 / 3, 12);
    // residuals: 2 − 1.8333, 3 − 3.3333, 5 − 4.8333 → 1/6, −1/3, 1/6 → SSE = 1/36 + 1/9 + 1/36 = 1/6
    expect(f.sse).toBeCloseTo(1 / 6, 12);
    expect(f.r2).toBeCloseTo(1 - (1 / 6) / (14 / 3), 12);
  });

  it("is the minimum: nudging slope or intercept raises SSE, and residuals sum to 0", () => {
    const pts = makeData(DEFAULT_LR);
    const f = ols(pts);
    expect(Math.abs(residuals(pts, f.slope, f.intercept).reduce((s, r) => s + r, 0))).toBeLessThan(1e-9);
    for (const [db, da] of [[0.05, 0], [-0.05, 0], [0, 0.1], [0, -0.1]]) expect(sse(pts, f.slope + db, f.intercept + da)).toBeGreaterThan(f.sse);
  });

  it("recovers the hidden line exactly with zero noise", () => {
    const f = ols(makeData({ ...DEFAULT_LR, noise: 0 }));
    expect(f.slope).toBeCloseTo(0.7, 1);
    expect(f.intercept).toBeCloseTo(2, 0);
    expect(f.r2).toBeGreaterThan(0.99);
  });

  it("the outlier drags the slope down and inflates SSE", () => {
    const clean = simulate(DEFAULT_LR).fit;
    const dirty = simulate({ ...DEFAULT_LR, outlier: true }).fit;
    expect(dirty.slope).toBeLessThan(clean.slope - 0.1);
    expect(dirty.sse).toBeGreaterThan(clean.sse * 2);
  });

  it("is deterministic per seed and produces the 9 steps in order", () => {
    const a = simulate(DEFAULT_LR), b = simulate(DEFAULT_LR);
    expect(a.pts).toEqual(b.pts);
    expect(a.steps.map((s) => s.id)).toEqual(["data", "mean-x", "mean-y", "products", "slope", "intercept", "residuals", "squares", "r2"]);
    expect(simulate({ ...DEFAULT_LR, seed: 99 }).pts).not.toEqual(a.pts);
  });

  it("handles degenerate input", () => {
    const flat = ols([{ x: 2, y: 1 }, { x: 2, y: 3 }]);
    expect(flat.slope).toBe(0);
    expect(flat.intercept).toBe(2);
    expect(ols([]).n).toBe(0);
    expect(makeData({ ...DEFAULT_LR, n: 100 })).toHaveLength(20);
  });
});
