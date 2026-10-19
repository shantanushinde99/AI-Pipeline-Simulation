import { describe, expect, it } from "vitest";
import { contingency, fitNearestCentroid, inertiaOf, kMeans, simulate, type Pt, type Vec } from "./simulation";

describe("supervised vs unsupervised", () => {
  it("nearest centroid averages each class and predicts the closest mean", () => {
    const pts: Pt[] = [{ x: [0, 0], label: 0 }, { x: [2, 0], label: 0 }, { x: [8, 8], label: 1 }, { x: [10, 8], label: 1 }];
    const m = fitNearestCentroid(pts, 2);
    expect(m.centroids).toEqual([[1, 0], [9, 8]]);
    expect(m.predict([4, 3])).toBe(0);
    expect(m.predict([7, 6])).toBe(1);
    expect(m.acc).toBe(1);
  });

  it("k-means converges on two obvious groups and inertia never increases", () => {
    const xs: Vec[] = [[0, 0], [0, 1], [1, 0], [9, 9], [9, 8], [8, 9]];
    const km = kMeans(xs, 2, 3);
    expect(km.converged).toBe(true);
    const groups = new Set([km.assign.slice(0, 3).join(), km.assign.slice(3).join()]);
    expect(groups.size).toBe(2); // first three share a cluster, last three share the other
    expect(km.final.map((c) => c.map((v) => +v.toFixed(6))).sort((a, b) => a[0] - b[0])).toEqual([[1 / 3, 1 / 3].map((v) => +v.toFixed(6)), [26 / 3, 26 / 3].map((v) => +v.toFixed(6))]);
    for (let i = 1; i < km.iters.length; i++) expect(km.iters[i].inertia).toBeLessThanOrEqual(km.iters[i - 1].inertia + 1e-9);
  });

  it("computes inertia and purity by hand", () => {
    expect(inertiaOf([[0, 0], [3, 4]], [0, 0], [[0, 0]])).toBe(25);
    const ct = contingency([0, 0, 1, 1], [0, 1, 1, 1], 2, 2);
    expect(ct.table).toEqual([[1, 1], [0, 2]]);
    expect(ct.purity).toBe(0.75);
  });

  it("is deterministic and orders steps data → supervised → k-means → compare", () => {
    const cfg = { dataset: "aligned" as const, k: 3, dataSeed: 7, initSeed: 2 };
    const a = simulate(cfg), b = simulate(cfg);
    expect(a.steps.map((s) => s.id)).toEqual(b.steps.map((s) => s.id));
    expect(a.steps.slice(0, 4).map((s) => s.state.phase)).toEqual(["data", "sup-means", "sup-boundary", "km-init"]);
    expect(a.steps.at(-1)!.state.phase).toBe("compare");
    expect(a.steps.filter((s) => s.state.phase === "km-iter")).toHaveLength(a.km.iters.length);
  });

  it("clusters match classes on aligned blobs but not on the mismatch dataset", () => {
    const aligned = simulate({ dataset: "aligned", k: 3, dataSeed: 7, initSeed: 2 });
    expect(aligned.ct.purity).toBeGreaterThan(0.9);
    expect(aligned.sup.acc).toBeGreaterThan(0.9);
    const mismatch = simulate({ dataset: "mismatch", k: 2, dataSeed: 7, initSeed: 2 });
    expect(mismatch.sup.acc).toBeGreaterThan(0.85); // labels make left/right easy
    expect(mismatch.ct.purity).toBeLessThan(0.75); // k-means splits top/bottom instead
  });

  it("changing k changes the clustering", () => {
    const k2 = simulate({ dataset: "aligned", k: 2, dataSeed: 7, initSeed: 2 });
    const k4 = simulate({ dataset: "aligned", k: 4, dataSeed: 7, initSeed: 2 });
    expect(k2.km.final).toHaveLength(2);
    expect(k4.km.final).toHaveLength(4);
    expect(k4.km.iters.at(-1)!.inertia).toBeLessThan(k2.km.iters.at(-1)!.inertia);
  });
});
