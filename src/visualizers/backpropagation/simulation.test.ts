import { describe, expect, it } from "vitest";
import { applyUpdate, backprop, bce, numericGrads, simulate } from "./simulation";
import { randomWeights, type Activation } from "../neural-network/simulation";

const flat = (g: ReturnType<typeof numericGrads>) => [...g.w1.flat(), ...g.b1, ...g.w2, g.b2];

describe("backpropagation", () => {
  it.each<Activation>(["relu", "sigmoid", "tanh", "linear"])("analytic gradients match finite differences (%s)", (activation) => {
    const input = { x: [0.7, -1.2] as [number, number], weights: randomWeights(7), activation };
    for (const t of [0, 1]) {
      const a = flat(backprop(input, t).grads);
      const n = flat(numericGrads(input, t));
      a.forEach((v, i) => expect(v).toBeCloseTo(n[i], 5));
    }
  });

  it("matches a hand calculation at the output", () => {
    const weights = { w1: [[1, 0], [0, 1], [0, 0]], b1: [0, 0, 0], w2: [1, 1, 0], b2: 0 };
    const r = backprop({ x: [1, 1], weights, activation: "linear" }, 1);
    // z_y = 2, ŷ = σ(2) ≈ 0.880797, δ_y = ŷ − 1
    expect(r.dz2).toBeCloseTo(1 / (1 + Math.exp(-2)) - 1, 10);
    expect(r.grads.w2[0]).toBeCloseTo(r.dz2 * 1, 10);
    expect(r.loss).toBeCloseTo(bce(r.f.y, 1), 12);
  });

  it("dead ReLU neurons get zero gradient on their incoming weights", () => {
    const weights = { w1: [[-1, -1], [1, 1], [0.5, 0.5]], b1: [0, 0, 0], w2: [2, 1, 1], b2: 0 };
    const r = backprop({ x: [1, 1], weights, activation: "relu" }, 0);
    expect(r.grads.w1[0]).toEqual([0, 0]);
    expect(r.grads.w1[1][0]).not.toBe(0);
  });

  it("a small step lowers the loss; a huge step can raise it", () => {
    const input = { x: [1, 0] as [number, number], weights: randomWeights(7), activation: "tanh" as const };
    const small = simulate({ ...input, target: 1, lr: 0.1 });
    expect(small.after.loss).toBeLessThan(small.bp.loss);
    const huge = simulate({ ...input, target: 1, lr: 10 });
    expect(huge.next).not.toEqual(small.next);
  });

  it("produces the same deterministic step sequence", () => {
    const input = { x: [1, 0] as [number, number], weights: randomWeights(3), activation: "relu" as const, target: 1 as const, lr: 0.5 };
    const a = simulate(input), b = simulate(input);
    expect(a.steps.map((s) => s.id)).toEqual(["forward", "loss", "delta", "grad-out", "hidden-0", "hidden-1", "hidden-2", "update", "after"]);
    expect(a).toEqual(b);
    expect(a.next).toEqual(applyUpdate(input.weights, a.bp.grads, 0.5));
  });
});
