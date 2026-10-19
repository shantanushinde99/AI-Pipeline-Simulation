import { describe, expect, it } from "vitest";
import { DATASETS, rank, simulate, topK, unit, type SimConfig } from "./simulation";

const base: SimConfig = { dataset: "default", query: [1.6, 1.2], k: 3, metric: "cosine", normalize: false, onePerSource: false };

describe("similarity search", () => {
  it("ranks by hand-checked scores on raw vectors", () => {
    const s = simulate(base);
    expect(s.raw.l2.slice(0, 3).map((r) => r.item.id)).toEqual(["A", "B", "L"]);
    expect(s.raw.l2[0].score).toBeCloseTo(-Math.sqrt(0.2), 10); // ‖[1.6,1.2] − [2,1]‖
    expect(s.raw.dot[0].item.id).toBe("C");
    expect(s.raw.dot[0].score).toBeCloseTo(1.6 * 3.8 + 1.2 * 2.6, 10);
    expect(s.raw.cosine.slice(0, 2).map((r) => r.item.id)).toEqual(["C", "D"]);
  });

  it("after normalization all three metrics produce the same order", () => {
    const s = simulate(base);
    const ids = (m: "cosine" | "dot" | "l2") => s.unitRanks[m].map((r) => r.item.id).join();
    expect(ids("dot")).toBe(ids("cosine"));
    expect(ids("l2")).toBe(ids("cosine"));
    const u = unit([3, 4]);
    expect(u[0]).toBeCloseTo(0.6); expect(u[1]).toBeCloseTo(0.8);
  });

  it("detects a tie at the cutoff and breaks it by id", () => {
    const s = simulate({ ...base, dataset: "ties", query: DATASETS.ties.query, metric: "l2" });
    expect(s.final.map((r) => r.item.id)).toEqual(["B", "A", "C"]);
    expect(s.tieAtCutoff?.item.id).toBe("D");
    expect(s.steps.at(-1)!.events.some((e) => e.type === "tie")).toBe(true);
  });

  it("near-duplicates crowd top-k unless collapsed per source", () => {
    const crowded = simulate({ ...base, dataset: "dupes", query: DATASETS.dupes.query, metric: "l2" });
    expect(crowded.sources).toBe(1);
    const fixed = simulate({ ...base, dataset: "dupes", query: DATASETS.dupes.query, metric: "l2", onePerSource: true });
    expect(fixed.sources).toBe(3);
    expect(topK(rank([1.5, 1.5], DATASETS.dupes.items, "l2"), 2, true).map((r) => r.item.source)).toEqual(["faq", "guide"]);
  });

  it("emits scene, scan, k capture steps, metrics, normalize, cutoff; deterministic; k changes the outcome", () => {
    const s = simulate(base);
    expect(s.steps.map((x) => x.id)).toEqual(["scene", "scan", "capture-1", "capture-2", "capture-3", "metrics", "normalize", "cutoff"]);
    expect(simulate(base)).toEqual(s);
    expect(simulate({ ...base, k: 5 }).steps).toHaveLength(10);
    expect(simulate({ ...base, metric: "dot" }).final[0].item.id).not.toBe(simulate({ ...base, metric: "l2" }).final[0].item.id);
  });

  it("falls back to the preset query when given a zero vector", () => {
    expect(simulate({ ...base, query: [0, 0] }).q).toEqual([1.6, 1.2]);
  });
});
