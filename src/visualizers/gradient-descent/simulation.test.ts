import { describe, expect, it } from "vitest";
import { DEFAULT_GD, SURFACES, run, simulate, type Vec } from "./simulation";

describe("gradient descent", () => {
  it("analytic gradients match finite differences on both surfaces", () => {
    const h = 1e-6;
    for (const s of Object.values(SURFACES)) {
      for (const p of [[-4.5, 2.5], [1.3, -0.7], [0.2, 0.15]] as Vec[]) {
        const g = s.grad(p);
        expect(g[0]).toBeCloseTo((s.f([p[0] + h, p[1]]) - s.f([p[0] - h, p[1]])) / (2 * h), 5);
        expect(g[1]).toBeCloseTo((s.f([p[0], p[1] + h]) - s.f([p[0], p[1] - h])) / (2 * h), 5);
      }
    }
  });

  it("one plain step is p − η∇f (hand calculation)", () => {
    const { iters } = run({ ...DEFAULT_GD, lr: 0.1, steps: 1 });
    // start (−4.5, 2.5): ∇f = (−4.5, 25) → p₁ = (−4.5 + 0.45, 2.5 − 2.5) = (−4.05, 0)
    expect(iters[1].p[0]).toBeCloseTo(-4.05, 12);
    expect(iters[1].p[1]).toBeCloseTo(0, 12);
    expect(iters[1].f).toBeCloseTo(0.5 * 4.05 ** 2, 12);
  });

  it("the learning-rate presets produce slow / good / oscillating / diverging behaviour", () => {
    for (const s of Object.values(SURFACES)) {
      const regime = (k: keyof typeof s.lr) => run({ ...DEFAULT_GD, surface: s.id, lr: s.lr[k], steps: 25 }).regime;
      expect([s.id, regime("slow")]).toEqual([s.id, "slow"]);
      expect([s.id, regime("good")]).toEqual([s.id, "good"]);
      expect([s.id, regime("oscillating")]).toEqual([s.id, "oscillating"]);
      expect([s.id, regime("diverging")]).toEqual([s.id, "diverging"]);
    }
  });

  it("momentum reaches a lower loss than plain GD with a slow learning rate", () => {
    const plain = run({ ...DEFAULT_GD, lr: SURFACES.bowl.lr.slow }).iters.at(-1)!.f;
    const heavy = run({ ...DEFAULT_GD, lr: SURFACES.bowl.lr.slow, momentum: true }).iters.at(-1)!.f;
    expect(heavy).toBeLessThan(plain / 2);
  });

  it("stochastic gradients are seeded: same seed same path, different seed different path", () => {
    const a = run({ ...DEFAULT_GD, stochastic: true });
    expect(run({ ...DEFAULT_GD, stochastic: true }).iters).toEqual(a.iters);
    expect(run({ ...DEFAULT_GD, stochastic: true, seed: 9 }).iters[3].p).not.toEqual(a.iters[3].p);
    expect(a.iters[2].g).not.toEqual(a.iters[2].gTrue);
    expect(run(DEFAULT_GD).iters[2].g).toEqual(run(DEFAULT_GD).iters[2].gTrue);
  });

  it("emits start, one step per iteration, and a summary; stops early when diverging", () => {
    const s = simulate(DEFAULT_GD);
    expect(s.steps).toHaveLength(DEFAULT_GD.steps + 2);
    expect(s.steps[0].id).toBe("start");
    expect(s.steps.at(-1)!.id).toBe("summary");
    const d = simulate({ ...DEFAULT_GD, lr: 0.5, steps: 40 });
    expect(d.diverged).toBe(true);
    expect(d.steps.length).toBeLessThan(42);
  });
});
