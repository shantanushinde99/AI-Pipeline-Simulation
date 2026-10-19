import { describe, expect, it } from "vitest";
import { ACTS, PRESETS, chainedGradient, compareAll, computeNeuron, flatRegions, simulate, zoneOf } from "./simulation";

describe("single neuron", () => {
  it("weighted sum matches a hand calculation", () => {
    const r = computeNeuron({ x: [1, 0.5, -1], w: [0.8, -0.4, 0.3], b: 0.5, act: "relu" });
    // 0.8 − 0.2 − 0.3 + 0.5 = 0.8
    expect(r.products.map((v) => +v.toFixed(10))).toEqual([0.8, -0.2, -0.3]);
    expect(r.z).toBeCloseTo(0.8, 12);
    expect(r.a).toBeCloseTo(0.8, 12);
    expect(r.d).toBe(1);
  });

  it("activations and derivatives match known values", () => {
    expect(ACTS.sigmoid.f(0)).toBe(0.5);
    expect(ACTS.sigmoid.d(0)).toBe(0.25);
    expect(ACTS.tanh.d(0)).toBe(1);
    expect(ACTS.leaky.f(-2)).toBeCloseTo(-0.2, 12);
    expect(ACTS.relu.d(-1)).toBe(0);
    expect(ACTS.gelu.f(0)).toBe(0);
    expect(ACTS.gelu.f(1)).toBeCloseTo(0.8412, 3); // exact GELU(1) = 0.8413
    // derivative vs. finite difference for every activation away from kinks
    for (const a of Object.values(ACTS)) for (const z of [-2.3, 0.7, 3.1]) {
      const h = 1e-5;
      expect(a.d(z)).toBeCloseTo((a.f(z + h) - a.f(z - h)) / (2 * h), 5);
    }
  });

  it("classifies dead and vanishing gradients", () => {
    expect(zoneOf("relu", -3.7)).toBe("dead");
    expect(zoneOf("leaky", -3.7)).toBe("healthy");
    expect(zoneOf("sigmoid", 7.5)).toBe("vanishing");
    expect(zoneOf("sigmoid", 0)).toBe("healthy");
    const dead = computeNeuron(PRESETS.find((p) => p.id === "dead")!.input);
    expect(dead.a).toBe(0);
    expect(dead.zone).toBe("dead");
  });

  it("chained gradient shrinks geometrically; flat regions are found", () => {
    expect(chainedGradient(0.25, 5).at(-1)).toBeCloseTo(0.25 ** 5, 12);
    const regions = flatRegions("relu", -5, 5);
    expect(regions).toHaveLength(1);
    expect(regions[0][0]).toBe(-5);
    expect(regions[0][1]).toBeGreaterThan(-0.1);
    expect(flatRegions("leaky", -5, 5)).toHaveLength(0);
  });

  it("produces 6 ordered steps; compare covers all five activations", () => {
    const steps = simulate(PRESETS[0].input);
    expect(steps.map((s) => s.id)).toEqual(["inputs", "sum", "activate", "output", "derivative", "compare"]);
    expect(compareAll(0.8)).toHaveLength(5);
    expect(simulate(PRESETS[0].input)).toEqual(steps); // deterministic
  });

  it("changing the activation changes the outcome and clamps bad inputs", () => {
    const base = PRESETS[0].input;
    expect(computeNeuron({ ...base, act: "sigmoid" }).a).not.toBeCloseTo(computeNeuron(base).a, 3);
    const r = computeNeuron({ x: [100, Number.NaN, 0], w: [1, 1, 1], b: 0, act: "relu" });
    expect(r.z).toBe(2); // x clamped to ±2, NaN → 0
  });
});
