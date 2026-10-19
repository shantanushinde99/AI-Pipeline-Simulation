import { describe, expect, it } from "vitest";
import { mean } from "../../lib/math";
import { fitPolynomial, trueFn } from "../model-evaluation/simulation";
import { GRID, decompose, drawSets, simulate } from "./simulation";

const base = { degree: 3, noise: 0.3, n: 15, sets: 20, seed: 1 };

describe("bias–variance decomposition", () => {
  it("identical training sets have zero variance and bias² equals their squared miss", () => {
    const one = drawSets({ n: 15, noise: 0.3, sets: 1, seed: 4 })[0];
    const d = decompose([one, one, one], 2, 0.3);
    expect(d.variance).toBeCloseTo(0, 12);
    const f = fitPolynomial(one, 2, 0);
    expect(d.bias2).toBeCloseTo(mean(GRID.map((x) => (f.predict(x) - trueFn(x)) ** 2)), 12);
    expect(d.noise2).toBeCloseTo(0.09, 12);
  });

  it("bias² + variance equals the average squared distance of the models from the truth", () => {
    const sets = drawSets(base);
    const d = decompose(sets, 4, 0.3);
    const direct = mean(GRID.map((x, g) => mean(d.curves.map((c) => (c[g] - trueFn(x)) ** 2))));
    expect(d.bias2 + d.variance).toBeCloseTo(direct, 10);
  });

  it("stratified sampling puts one x in each bin of [−1, 1]", () => {
    const [s] = drawSets({ n: 10, noise: 0, sets: 1, seed: 9 });
    s.forEach((p, i) => {
      expect(p.x).toBeGreaterThanOrEqual(-1 + 0.2 * i);
      expect(p.x).toBeLessThan(-1 + 0.2 * (i + 1));
      expect(p.y).toBeCloseTo(trueFn(p.x), 12); // σ = 0 → labels are exact
    });
  });

  it("simple models are biased, flexible models have high variance", () => {
    const s = simulate({ ...base, n: 12 });
    const at = (d: number) => s.sweep[d - 1];
    expect(at(1).bias2).toBeGreaterThan(10 * at(3).bias2);
    expect(at(10).variance).toBeGreaterThan(10 * at(3).variance);
    expect(s.best.degree).toBeGreaterThan(1);
    expect(s.best.degree).toBeLessThan(10);
  });

  it("the decomposition matches a Monte-Carlo test error and is deterministic", () => {
    const a = simulate(base), b = simulate(base);
    expect(a.total).toBe(b.total);
    expect(a.steps.map((s) => s.id)).toEqual(["truth", "one", "many", "mean", "variance", "decompose", "sweep"]);
    expect(Math.abs(a.measured - a.total) / a.total).toBeLessThan(0.1);
  });

  it("more noise raises the floor and the variance", () => {
    const lo = simulate({ ...base, noise: 0.1 }), hi = simulate({ ...base, noise: 0.5 });
    expect(hi.noise2).toBeGreaterThan(lo.noise2);
    expect(hi.variance).toBeGreaterThan(lo.variance);
  });
});
