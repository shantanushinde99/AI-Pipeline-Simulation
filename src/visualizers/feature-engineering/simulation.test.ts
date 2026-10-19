import { describe, expect, it } from "vitest";
import { ageYears, fitRidge, generateHouses, simulate, type FeConfig } from "./simulation";

const ALL = { date: true, log: true, onehot: true, scale: true, interact: true };
const cfg = (p: Partial<FeConfig> = {}): FeConfig => ({ enabled: ALL, leakScaler: false, lambda: 0.02, seed: 2, ...p });

describe("feature engineering pipeline", () => {
  it("parses a date string into age in years", () => {
    // 2015-01-01 → 2025-01-01 is 3653 days
    expect(ageYears("2015-01-01")).toBeCloseTo(3653 / 365.25, 10);
    expect(ageYears("not a date")).toBeNaN();
  });

  it("ridge with λ → 0 recovers an exact linear relationship", () => {
    const X = [[0, 1], [1, 0], [2, 3], [3, 1], [4, 4]];
    const y = X.map(([a, b]) => 2 + 3 * a - b);
    const m = fitRidge(X, y, 0);
    expect(m.w[0]).toBeCloseTo(2, 5);
    expect(m.w[1]).toBeCloseTo(3, 5);
    expect(m.w[2]).toBeCloseTo(-1, 5);
  });

  it("is deterministic with 7 steps in pipeline order and a 28/12 split", () => {
    const a = simulate(cfg()), b = simulate(cfg());
    expect(a.steps.map((s) => s.id)).toEqual(["raw", "date", "log", "onehot", "scale", "interact", "summary"]);
    expect(a.steps.map((s) => s.state.r2Test)).toEqual(b.steps.map((s) => s.state.r2Test));
    expect(a.trainIdx).toHaveLength(28);
    expect(a.testIdx).toHaveLength(12);
    expect(generateHouses(2).map((h) => h.price)).toEqual(generateHouses(2).map((h) => h.price));
  });

  it("engineering features lifts held-out R² well above the raw baseline", () => {
    const s = simulate(cfg());
    expect(s.steps[0].state.features).toEqual(["sqft"]);
    expect(s.steps.at(-1)!.state.r2Test).toBeGreaterThan(s.steps[0].state.r2Test + 0.3);
  });

  it("a skipped transform leaves the model unchanged; the interaction needs one-hot", () => {
    const s = simulate(cfg({ enabled: { ...ALL, log: false } }));
    expect(s.steps[2].state.r2Test).toBe(s.steps[1].state.r2Test);
    expect(s.steps[2].state.on).toBe(false);
    const noOh = simulate(cfg({ enabled: { ...ALL, onehot: false } }));
    expect(noOh.steps[5].state.on).toBe(false);
    expect(noOh.steps[5].state.features.some((f) => f.includes("Downtown"))).toBe(false);
  });

  it("with plain least squares, standardising does not change R²", () => {
    const s = simulate(cfg({ lambda: 0 }));
    expect(s.steps[4].state.r2Test).toBeCloseTo(s.steps[3].state.r2Test, 6);
  });

  it("the leaky scaler uses held-out rows in its statistics", () => {
    const clean = simulate(cfg()).steps[4].state.scalers;
    const leak = simulate(cfg({ leakScaler: true })).steps[4].state.scalers;
    expect(clean[0].mu).toBe(clean[0].muTrain);
    expect(leak[0].mu).not.toBe(leak[0].muTrain);
    expect(leak[0].muTrain).toBe(clean[0].muTrain);
  });
});
