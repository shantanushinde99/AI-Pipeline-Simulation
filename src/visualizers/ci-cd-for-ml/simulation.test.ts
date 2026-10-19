import { describe, expect, it } from "vitest";
import { DEFAULT_THRESHOLDS, fmtCheck, percentile, psi, runPipeline } from "./simulation";

const ids = (preset: Parameters<typeof runPipeline>[0], th = DEFAULT_THRESHOLDS) => runPipeline(preset, th).map((s) => s.id);
const last = (preset: Parameters<typeof runPipeline>[0], th = DEFAULT_THRESHOLDS) => runPipeline(preset, th).at(-1)!.state;

describe("CI/CD for ML pipeline", () => {
  it("psi and percentile match hand calculations", () => {
    expect(psi([0.5, 0.5], [0.5, 0.5])).toBe(0);
    // (0.6−0.5)ln(1.2) + (0.4−0.5)ln(0.8) = 0.018232 + 0.022314
    expect(psi([0.5, 0.5], [0.6, 0.4])).toBeCloseTo(0.1 * Math.log(1.2) - 0.1 * Math.log(0.8), 12);
    expect(percentile([5, 1, 4, 2, 3], 0.95)).toBe(5);
    expect(percentile(Array.from({ length: 200 }, (_, i) => i + 1), 0.95)).toBe(190);
    expect(fmtCheck({ unit: "pts", digits: 1 }, 0.0153)).toBe("+1.5 pts");
  });

  it("a healthy change runs every stage and is promoted", () => {
    expect(ids("healthy")).toEqual(["commit", "unit", "data", "train", "eval", "canary", "promote"]);
    expect(last("healthy").outcome).toBe("promoted");
  });

  it("a schema change stops at data validation, after unit tests passed", () => {
    const steps = runPipeline("schema");
    expect(steps.map((s) => s.id)).toEqual(["commit", "unit", "data", "result"]);
    expect(steps[1].state.statuses.unit).toBe("ok");
    expect(steps[2].state.checks.find((c) => c.id === "schema")!.pass).toBe(false);
    expect(steps.at(-1)!.state.statuses.train).toBe("skipped");
  });

  it("drift fails on PSI and a looser threshold lets it through", () => {
    const data = runPipeline("drift")[2].state.checks.find((c) => c.id === "psi")!;
    expect(data.measured!).toBeGreaterThan(0.4);
    expect(last("drift").failedAt).toBe("data");
    expect(last("drift", { ...DEFAULT_THRESHOLDS, maxPsi: 0.5 }).outcome).toBe("promoted");
  });

  it("a slice regression is blocked even though overall accuracy improves; disabling the gate ships it", () => {
    const evalStep = runPipeline("slice").find((s) => s.id === "eval")!.state;
    const acc = evalStep.checks.find((c) => c.id === "acc")!;
    const slice = evalStep.checks.find((c) => c.id === "slice")!;
    expect(acc.measured!).toBeGreaterThan(0); // 1680/1900 − 1655/1900
    expect(acc.measured!).toBeCloseTo(25 / 1900, 12);
    expect(slice.measured!).toBeCloseTo((255 - 228) / 300, 12); // 0.09
    expect(slice.pass).toBe(false);
    expect(last("slice").failedAt).toBe("eval");
    expect(last("slice", { ...DEFAULT_THRESHOLDS, sliceGate: false }).outcome).toBe("promoted-with-regression");
  });

  it("a canary error spike rolls back, and the error rate equals errors / requests", () => {
    const s = last("canary");
    expect(s.outcome).toBe("rolled-back");
    const req = s.canary.minutes.reduce((a, m) => a + m.requests, 0);
    const err = s.canary.minutes.reduce((a, m) => a + m.errors, 0);
    expect(s.canary.minutes.at(-1)!.cumRate).toBeCloseTo(err / req, 12);
    expect(req).toBeGreaterThanOrEqual(s.canary.minRequests);
    expect(runPipeline("canary")).toEqual(runPipeline("canary")); // deterministic
  });
});
