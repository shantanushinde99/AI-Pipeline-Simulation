import { describe, expect, it } from "vitest";
import { DEFAULT_OFFLINE, SLICES, bootstrapDiff, buildDataset, histogram, percentile, simulate } from "./simulation";

describe("offline evaluation", () => {
  it("builds slices with exactly the scenario's correct counts", () => {
    const d = buildDataset(DEFAULT_OFFLINE);
    expect(d).toHaveLength(80);
    const count = (slice: string, m: "a" | "b") => d.filter((e) => e.slice === slice && e[m]).length;
    expect(SLICES.map((s) => count(s.id, "a"))).toEqual([21, 16, 9, 8]);
    expect(SLICES.map((s) => count(s.id, "b"))).toEqual([28, 22, 13, 3]);
  });

  it("aggregate improves while the Non-English slice regresses (the trap)", () => {
    const s = simulate(DEFAULT_OFFLINE);
    expect(s.accA).toBeCloseTo(54 / 80, 12);
    expect(s.accB).toBeCloseTo(66 / 80, 12);
    const nonen = s.slices.find((x) => x.id === "nonen")!;
    expect(nonen.diff).toBeCloseTo(-0.5, 12);
    expect(nonen.regress).toBe(true);
    expect(s.worst.id).toBe("nonen");
    expect(s.boot.ci[0]).toBeGreaterThan(0); // the aggregate win is "statistically solid"
    // re-weighted to 35% Non-English traffic, B is worse than A
    expect(s.prodB).toBeLessThan(s.prodA);
  });

  it("percentile interpolates like numpy's default", () => {
    expect(percentile([1, 2, 3, 4], 0.5)).toBe(2.5);
    expect(percentile([0, 10], 0.25)).toBe(2.5);
    expect(percentile([5], 0.975)).toBe(5);
  });

  it("bootstrap is deterministic and its CI brackets the observed difference", () => {
    const d = buildDataset(DEFAULT_OFFLINE);
    const a = bootstrapDiff(d, 500, 3), b = bootstrapDiff(d, 500, 3);
    expect(a.diffs).toEqual(b.diffs);
    expect(a.ci[0]).toBeLessThan(12 / 80);
    expect(a.ci[1]).toBeGreaterThan(12 / 80);
    // each resample draws n examples in total
    expect(a.firstPicks.reduce((s, v) => s + v, 0)).toBe(80);
  });

  it("histogram snapshots accumulate to the full count", () => {
    const h = histogram([0, 0.1, 0.1, 0.2], 0.1, [0.5, 1]);
    expect(h.counts[1].reduce((s, v) => s + v, 0)).toBe(4);
    expect(h.counts[0].reduce((s, v) => s + v, 0)).toBe(2);
  });

  it("changing B's Non-English score removes the regression", () => {
    const fixed = simulate({ ...DEFAULT_OFFLINE, bNonEn: 9 });
    expect(fixed.slices.find((x) => x.id === "nonen")!.regress).toBe(false);
    expect(fixed.prodB).toBeGreaterThan(fixed.prodA);
    expect(simulate(DEFAULT_OFFLINE).steps.map((s) => s.id)).toEqual(["dataset", "runA", "runB", "aggregate", "resample", "bootstrap", "slices", "reweight"]);
  });
});
