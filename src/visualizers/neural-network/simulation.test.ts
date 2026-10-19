import { describe, expect, it } from "vitest";
import { DATASETS, evaluateDataset, forward, simulate } from "./simulation";
import { sigmoid } from "../../lib/math";

const xor = DATASETS[0];

describe("neural network forward pass", () => {
  it("matches a hand calculation", () => {
    const weights = { w1: [[0.5, -1], [1, 1], [0, 2]], b1: [0.1, -0.5, 0], w2: [1, -2, 0.5], b2: 0.2 };
    const r = forward({ x: [2, 1], weights, activation: "relu" });
    // z1 = [0.5*2 - 1*1 + 0.1, 2 + 1 - 0.5, 0 + 2 + 0] = [0.1, 2.5, 2]
    expect(r.z1.map((v) => +v.toFixed(10))).toEqual([0.1, 2.5, 2]);
    expect(r.h.map((v) => +v.toFixed(10))).toEqual([0.1, 2.5, 2]);
    // z2 = 0.1 - 5 + 1 + 0.2 = -3.7
    expect(r.z2).toBeCloseTo(-3.7, 10);
    expect(r.y).toBeCloseTo(sigmoid(-3.7), 12);
  });

  it("applies the chosen activation", () => {
    const weights = { w1: [[1, 0], [0, 1], [1, 1]], b1: [0, 0, 0], w2: [0, 0, 0], b2: 0 };
    expect(forward({ x: [-1, 2], weights, activation: "relu" }).h).toEqual([0, 2, 1]);
    expect(forward({ x: [-1, 2], weights, activation: "linear" }).h).toEqual([-1, 2, 1]);
    expect(forward({ x: [0, 0], weights, activation: "sigmoid" }).h).toEqual([0.5, 0.5, 0.5]);
  });

  it("the XOR preset classifies all four points", () => {
    const { accuracy } = evaluateDataset(xor, xor.preset, "relu");
    expect(accuracy).toBe(1);
    expect(evaluateDataset(DATASETS[1], DATASETS[1].preset, "relu").accuracy).toBe(1);
  });

  it("produces steps whose final output equals the forward pass", () => {
    const input = { x: [1, 0] as [number, number], weights: xor.preset, activation: "relu" as const };
    const steps = simulate(input);
    expect(steps).toHaveLength(10);
    expect(steps.at(-1)!.state.values.y.a).toBeCloseTo(forward(input).y, 12);
    // hidden values are revealed progressively
    expect(steps[0].state.values.h1).toBeUndefined();
    expect(steps[1].state.values.h1.a).toBeUndefined();
    expect(steps[2].state.values.h1.a).toBe(1);
  });

  it("changing a weight changes the output", () => {
    const w = structuredClone(xor.preset);
    const before = forward({ x: [1, 0], weights: w, activation: "relu" }).y;
    w.w2[0] = 1;
    expect(forward({ x: [1, 0], weights: w, activation: "relu" }).y).not.toBeCloseTo(before, 3);
  });
});
