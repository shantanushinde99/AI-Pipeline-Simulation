import { describe, expect, it } from "vitest";
import { DEFAULT_LF, LOSS_IDS, bce, bceGrad, fitLine, gradOf, lossOf, makeData, mseGrad, simulate, totalLoss } from "./simulation";

describe("loss functions", () => {
  it("per-point losses and gradients match hand values", () => {
    expect(lossOf("mse", -3, 1)).toBe(9);
    expect(lossOf("mae", -3, 1)).toBe(3);
    expect(lossOf("huber", 0.5, 1)).toBe(0.125); // ½·0.25
    expect(lossOf("huber", -3, 1)).toBe(2.5); // 1·(3 − 0.5)
    expect([gradOf("mse", -3, 1), gradOf("mae", -3, 1), gradOf("huber", -3, 1), gradOf("huber", 0.4, 1)]).toEqual([-6, -1, -1, 0.4]);
  });

  it("gradients match finite differences of the losses", () => {
    const h = 1e-6;
    for (const id of LOSS_IDS) for (const r of [-2.7, -0.3, 0.6, 4.1]) {
      expect(gradOf(id, r, 1.2)).toBeCloseTo((lossOf(id, r + h, 1.2) - lossOf(id, r - h, 1.2)) / (2 * h), 5);
    }
  });

  it("each fitted line is a minimum of its own loss", () => {
    const pts = makeData(2, true, 6);
    for (const id of LOSS_IDS) {
      const l = fitLine(id, pts, 1);
      const best = totalLoss(id, pts, l, 1);
      for (const [ds, di] of [[0.03, 0], [-0.03, 0], [0, 0.1], [0, -0.1], [0.02, -0.15]])
        expect(totalLoss(id, pts, { slope: l.slope + ds, intercept: l.intercept + di }, 1)).toBeGreaterThanOrEqual(best - 1e-9);
    }
  });

  it("the outlier drags the MSE fit much more than MAE or Huber", () => {
    const clean = simulate({ ...DEFAULT_LF, outlier: false }).fits;
    const dirty = simulate(DEFAULT_LF).fits;
    const moved = (id: "mse" | "mae" | "huber") => Math.abs(dirty[id].slope - clean[id].slope);
    expect(moved("mse")).toBeGreaterThan(3 * moved("mae"));
    expect(moved("mse")).toBeGreaterThan(2 * moved("huber"));
    expect(simulate(DEFAULT_LF).share("mse")).toBeGreaterThan(simulate(DEFAULT_LF).share("mae"));
  });

  it("cross-entropy punishes and keeps the gradient of a confident mistake; MSE does not", () => {
    expect(bce(0.02)).toBeCloseTo(3.912, 3);
    expect(bceGrad(0.02)).toBeCloseTo(0.98, 12);
    expect(mseGrad(0.02)).toBeCloseTo(2 * 0.02 * 0.98 ** 2, 12);
    expect(bceGrad(0.02) / mseGrad(0.02)).toBeGreaterThan(20);
  });

  it("produces deterministic regression and classification step lists", () => {
    const r = simulate(DEFAULT_LF);
    expect(r.steps.map((s) => s.id)).toEqual(["data", "candidate", "curves", "bars", "grads", "fit-mse", "fit-mae", "fit-huber", "compare"]);
    expect(simulate(DEFAULT_LF).pts).toEqual(r.pts);
    const c = simulate({ ...DEFAULT_LF, mode: "classification" });
    expect(c.steps.map((s) => s.id)).toEqual(["examples", "bce", "mse", "cbars", "cgrads", "csummary"]);
    expect(c.examples).toHaveLength(7);
    expect(simulate({ ...DEFAULT_LF, mode: "classification", confidentWrong: false }).examples).toHaveLength(6);
  });
});
