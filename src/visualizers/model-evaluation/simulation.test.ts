import { describe, expect, it } from "vitest";
import { diagnose, evaluate, fitPolynomial, generateData, simulate, sweep } from "./simulation";
import { mse } from "../../lib/metrics";

const base = { n: 40, noise: 0.25, degree: 4, testFraction: 0.3, lambda: 0, seed: 3 };

describe("model evaluation", () => {
  it("is deterministic for the same seed", () => {
    expect(generateData(base)).toEqual(generateData(base));
    expect(generateData(base)).not.toEqual(generateData({ ...base, seed: 4 }));
  });

  it("splits by the requested fraction without overlap", () => {
    const { train, test } = generateData(base);
    expect(test).toHaveLength(12);
    expect(train).toHaveLength(28);
  });

  it("recovers an exact line", () => {
    const pts = [0, 0.25, 0.5, 0.75, 1].map((x) => ({ x, y: 2 * x + 1 }));
    const m = fitPolynomial(pts, 1, 0);
    expect(m.w[0]).toBeCloseTo(1, 6);
    expect(m.w[1]).toBeCloseTo(2, 6);
  });

  it("reported metrics match the plotted predictions", () => {
    const { train, test } = generateData(base);
    const e = evaluate(train, test, 4, 0);
    expect(e.testMse).toBeCloseTo(mse(test.map((p) => p.y), test.map((p) => e.model.predict(p.x))), 12);
  });

  it("training error never increases with degree (no regularization)", () => {
    const { train, test } = generateData(base);
    const sw = sweep(train, test, 0);
    for (let i = 1; i < 10; i++) expect(sw[i].trainMse).toBeLessThanOrEqual(sw[i - 1].trainMse + 1e-9);
  });

  it("shows underfitting, a good fit and overfitting", () => {
    const cfg = { n: 20, noise: 0.3, testFraction: 0.3, lambda: 0, seed: 11 };
    expect(simulate({ ...cfg, degree: 1 }).verdict).toBe("underfitting");
    const best = simulate({ ...cfg, degree: 1 }).bestDegree;
    expect(simulate({ ...cfg, degree: best }).verdict).toBe("good fit");
    expect(simulate({ ...cfg, degree: 14 }).verdict).toBe("overfitting");
  });

  it("regularization reduces the overfitting gap", () => {
    const cfg = { n: 20, noise: 0.3, testFraction: 0.3, seed: 11, degree: 14 };
    expect(simulate({ ...cfg, lambda: 0.001 }).testMse).toBeLessThan(simulate({ ...cfg, lambda: 0 }).testMse);
  });

  it("diagnose is relative to the sweep", () => {
    const sw = [{ degree: 1, trainMse: 1, testMse: 1 }, { degree: 2, trainMse: 0.1, testMse: 0.1 }, { degree: 3, trainMse: 0.01, testMse: 0.5 }];
    expect(diagnose(1, 1, sw).verdict).toBe("underfitting");
    expect(diagnose(3, 0.5, sw).verdict).toBe("overfitting");
    expect(diagnose(2, 0.1, sw).verdict).toBe("good fit");
  });
});
