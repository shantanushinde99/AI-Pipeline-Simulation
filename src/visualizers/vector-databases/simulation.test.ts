import { describe, expect, it } from "vitest";
import { DEFAULT_VDB, buildHnsw, d2, flatSearch, hnswSearch, ivfSearch, makePoints, recallAt, runAll, simulate, trainIvf, tradeoff, voronoiCell, type Pt } from "./simulation";

const pts = makePoints();
const q: Pt = [48, 52];

describe("vector database indexes", () => {
  it("generates 200 deterministic points in the unit box", () => {
    expect(pts).toHaveLength(200);
    expect(makePoints()).toEqual(pts);
    for (const [x, y] of pts) { expect(x).toBeGreaterThanOrEqual(0); expect(y).toBeLessThanOrEqual(100); }
  });

  it("flat search is exact and costs N computations", () => {
    const r = flatSearch(pts, q, 5);
    const brute = pts.map((p, i) => [d2(p, q), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, 5).map((x) => x[1]);
    expect(r.ids).toEqual(brute);
    expect(r.computations).toBe(200);
    expect(recallAt([1, 2, 3], [3, 4, 5])).toBeCloseTo(1 / 3, 12);
  });

  it("k-means assigns every point to its nearest centroid; Voronoi cells contain their centroid", () => {
    const ivf = trainIvf(pts, 8);
    expect(ivf.centroids).toHaveLength(8);
    ivf.assign.forEach((a, i) => { for (const c of ivf.centroids) expect(d2(pts[i], ivf.centroids[a])).toBeLessThanOrEqual(d2(pts[i], c) + 1e-9); });
    const cell = voronoiCell(ivf.centroids, 0);
    const area = Math.abs(cell.reduce((s, p, i) => { const n = cell[(i + 1) % cell.length]; return s + p[0] * n[1] - n[0] * p[1]; }, 0)) / 2;
    expect(area).toBeGreaterThan(0);
    const total = ivf.centroids.reduce((s, _, j) => { const c = voronoiCell(ivf.centroids, j); return s + Math.abs(c.reduce((t, p, i) => { const n = c[(i + 1) % c.length]; return t + p[0] * n[1] - n[0] * p[1]; }, 0)) / 2; }, 0);
    expect(total).toBeCloseTo(10000, 4); // cells tile the box exactly
  });

  it("IVF cost = nlist + scanned vectors, and probing every cell is exact", () => {
    const ivf = trainIvf(pts, 8);
    const one = ivfSearch(pts, ivf, q, 5, 1);
    expect(one.computations).toBe(8 + one.scanned.length);
    const all = ivfSearch(pts, ivf, q, 5, 8);
    expect(all.ids).toEqual(flatSearch(pts, q, 5).ids);
    expect(all.computations).toBe(208);
  });

  it("HNSW layers shrink upward and a big ef reaches full recall cheaper than flat", () => {
    const h = buildHnsw(pts, 5);
    const sizes = [0, 1, 2].map((l) => h.levels.filter((x) => x >= l).length);
    expect(sizes[0]).toBe(200);
    expect(sizes[1]).toBeLessThan(sizes[0]);
    expect(sizes[2]).toBeLessThanOrEqual(sizes[1]);
    const r = hnswSearch(pts, h, q, 5, 64);
    expect(recallAt(r.ids, flatSearch(pts, q, 5).ids)).toBe(1);
    expect(r.computations).toBeLessThan(200);
    expect(r.computations).toBe(1 + r.layers.reduce((s, l) => s + l.evaluated.length, 0));
    const small = hnswSearch(pts, h, q, 5, 5);
    expect(small.computations).toBeLessThan(r.computations);
  });

  it("parameters change the outcome: nprobe 1 vs all, and the step lists per index", () => {
    const low = runAll({ ...DEFAULT_VDB, nprobe: 1 }), high = runAll({ ...DEFAULT_VDB, nprobe: 8 });
    expect(high.iv.recall).toBe(1);
    expect(low.iv.computations).toBeLessThan(high.iv.computations);
    expect(simulate({ ...DEFAULT_VDB, index: "flat" }).steps.map((s) => s.id)).toEqual(["data", "scan", "topk", "compare", "insert"]);
    expect(simulate(DEFAULT_VDB).steps.map((s) => s.id)).toEqual(["data", "train", "assign", "probe", "scan-cells", "topk", "compare", "insert"]);
    const hn = simulate({ ...DEFAULT_VDB, index: "hnsw" }).steps.map((s) => s.id);
    expect(hn[0]).toBe("data");
    expect(hn).toContain("layer-0");
    const t = tradeoff(low);
    expect(t.ivfCurve.at(-1)!.y).toBe(1);
    expect(t.ivfCurve.map((c) => c.x)).toEqual([...t.ivfCurve.map((c) => c.x)].sort((a, b) => a - b));
  });
});
