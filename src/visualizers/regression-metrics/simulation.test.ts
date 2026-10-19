import { describe, expect, it } from "vitest";
import { DEFAULT_REG, makeData, regressionMetrics, simulate } from "./simulation";

describe("regression metrics", () => {
  it("matches a hand calculation", () => {
    // y = [3, 5, 10], ŷ = [2, 5, 13] → e = [1, 0, -3]
    const m = regressionMetrics([{ id: 1, y: 3, yhat: 2 }, { id: 2, y: 5, yhat: 5 }, { id: 3, y: 10, yhat: 13 }]);
    expect(m.mae).toBeCloseTo(4 / 3, 12);
    expect(m.mse).toBeCloseTo(10 / 3, 12);
    expect(m.rmse).toBeCloseTo(Math.sqrt(10 / 3), 12);
    // ȳ = 6, SS_tot = 9 + 1 + 16 = 26
    expect(m.r2).toBeCloseTo(1 - 10 / 26, 12);
    expect(m.mape).toBeCloseTo(((1 / 3 + 0 + 3 / 10) / 3) * 100, 10);
  });

  it("R² is negative when worse than predicting the mean and undefined for constant targets", () => {
    expect(regressionMetrics([{ id: 1, y: 1, yhat: 5 }, { id: 2, y: 3, yhat: -1 }]).r2).toBeLessThan(0);
    expect(Number.isNaN(regressionMetrics([{ id: 1, y: 2, yhat: 1 }, { id: 2, y: 2, yhat: 3 }]).r2)).toBe(true);
  });

  it("is deterministic and produces the eight steps in order", () => {
    expect(makeData(DEFAULT_REG)).toEqual(makeData(DEFAULT_REG));
    expect(simulate(DEFAULT_REG).steps.map((s) => s.id)).toEqual(["data", "residuals", "mae", "mse", "rmse", "r2", "mape", "compare"]);
  });

  it("an outlier moves MSE far more than MAE", () => {
    const clean = simulate(DEFAULT_REG).m;
    const dirty = simulate({ ...DEFAULT_REG, outlier: true }).m;
    expect(dirty.mse / clean.mse).toBeGreaterThan(dirty.mae / clean.mae);
    expect(dirty.mse / clean.mse).toBeGreaterThan(5);
  });

  it("unit scale changes MAE/RMSE linearly and MSE quadratically, but not R² or MAPE", () => {
    const a = simulate(DEFAULT_REG).m;
    const b = simulate({ ...DEFAULT_REG, altUnits: true }).m;
    expect(b.mae / a.mae).toBeCloseTo(1000, 6);
    expect(b.rmse / a.rmse).toBeCloseTo(1000, 6);
    expect(b.mse / a.mse).toBeCloseTo(1e6, 0);
    expect(b.r2).toBeCloseTo(a.r2, 10);
    expect(b.mape).toBeCloseTo(a.mape, 8);
  });

  it("targets near zero make MAPE explode while MAE stays small", () => {
    const e = simulate({ ...DEFAULT_REG, dataset: "energy" }).m;
    const h = simulate(DEFAULT_REG).m;
    expect(e.mape).toBeGreaterThan(50);
    expect(h.mape).toBeLessThan(15);
    expect(e.mae).toBeLessThan(0.5);
  });
});
