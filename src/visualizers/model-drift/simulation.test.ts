import { describe, expect, it } from "vitest";
import { psi } from "../data-drift-and-concept-drift/simulation";
import { BOUNDARY0, DEFAULT_DRIFT, T, driftAt, simulate } from "./simulation";

const run = (p: Partial<typeof DEFAULT_DRIFT>) => simulate({ ...DEFAULT_DRIFT, ...p });

describe("model-drift simulation", () => {
  it("is deterministic with one step for training, one per week and a summary", () => {
    const a = run({}), b = run({});
    expect(a.batches.map((x) => x.psi)).toEqual(b.batches.map((x) => x.psi));
    expect(a.steps).toHaveLength(T + 2);
    expect(a.steps[0].state.t).toBe(-1);
    expect(a.steps[T + 1].state.t).toBe(T);
  });

  it("PSI matches a hand calculation and per-bin contributions sum to it", () => {
    // (0.6−0.5)ln(0.6/0.5) + (0.4−0.5)ln(0.4/0.5) = 0.0405
    expect(psi([0.5, 0.5], [0.6, 0.4])).toBeCloseTo(0.0405, 4);
    for (const x of run({ kind: "covariate" }).batches) expect(x.contrib.reduce((s, v) => s + v, 0)).toBeCloseTo(x.psi, 10);
  });

  it("covariate drift trips PSI and KS but accuracy barely moves", () => {
    const s = run({ kind: "covariate" });
    const after = s.batches.filter((x) => x.t >= s.cfg.start);
    expect(after.every((x) => x.inputAlert)).toBe(true);
    expect(Math.min(...after.map((x) => x.acc))).toBeGreaterThan(0.8);
  });

  it("concept drift is invisible to input monitors and only shows up when labels arrive", () => {
    const s = run({ kind: "concept" });
    expect(s.batches.filter((x) => x.t >= s.cfg.start).some((x) => x.inputAlert)).toBe(false);
    expect(s.batches[s.cfg.start].acc).toBeLessThan(0.75);
    expect(s.firstAccAlertKnown).toBe(s.cfg.start + s.cfg.labelDelay);
    expect(driftAt("concept", s.cfg.start, 1, s.cfg.start).b).toBe(BOUNDARY0 - 1);
  });

  it("retraining on recent labelled data moves the boundary and recovers accuracy", () => {
    const off = run({ kind: "concept" }), on = run({ kind: "concept", retrain: true });
    expect(on.retrains.length).toBeGreaterThan(0);
    expect(on.boundaries[1]).toBeLessThan(on.boundaries[0] - 0.5);
    const tail = (s: typeof on) => s.batches.slice(-3).reduce((a, x) => a + x.acc, 0) / 3;
    expect(tail(on) - tail(off)).toBeGreaterThan(0.1);
  });

  it("a longer label delay postpones the accuracy alert", () => {
    expect(run({ labelDelay: 4 }).firstAccAlertKnown).toBe((run({ labelDelay: 1 }).firstAccAlertKnown ?? 0) + 3);
    expect(run({ kind: "none" }).firstInputAlert).toBeUndefined();
  });
});
