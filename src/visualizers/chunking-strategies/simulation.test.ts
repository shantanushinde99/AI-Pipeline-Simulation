import { describe, expect, it } from "vitest";
import { DEFAULT_PARAMS, adjacentSimilarity, buildDoc, chunkDoc, compareAll, factSpans, sanitize, simulate, stats } from "./simulation";

const doc = buildDoc();
const P = DEFAULT_PARAMS;

describe("chunking strategies", () => {
  it("fixed size cuts every N words and covers the document exactly once", () => {
    const c = chunkDoc({ ...P, strategy: "fixed", size: 30 });
    expect(c.map((x) => x.start)).toEqual([0, 30, 60, 90, 120, 150]);
    expect(c.at(-1)!.end).toBe(doc.words.length);
    expect(stats(c, []).overlapWords).toBe(0);
  });

  it("overlap repeats the last `overlap` words of the previous chunk", () => {
    const c = chunkDoc({ ...P, strategy: "overlap", size: 30, overlap: 10 });
    for (let i = 1; i < c.length; i++) {
      expect(c[i].start).toBe(c[i - 1].start + 20);
      expect(c[i - 1].end - c[i].start).toBe(10);
    }
    expect(sanitize({ ...P, size: 10, overlap: 50 }).overlap).toBe(9);
  });

  it("sentence and heading strategies only cut at sentence starts; heading-aware always cuts at headings", () => {
    const starts = new Set(doc.sentences.map((s) => s.start));
    for (const strategy of ["sentence", "heading", "semantic"] as const) for (const ch of chunkDoc({ ...P, strategy })) expect(starts.has(ch.start)).toBe(true);
    const h = chunkDoc({ ...P, strategy: "heading", size: 200 });
    expect(h.map((x) => x.start)).toEqual(doc.sentences.filter((s) => s.heading).map((s) => s.start));
  });

  it("semantic chunking cuts where adjacent similarity drops below the threshold", () => {
    const sims = adjacentSimilarity(doc.sentences);
    // Section boundaries share no terms in this toy embedding: similarity 0.
    const sectionStarts = doc.sentences.filter((s) => s.heading && s.start > 0).map((s) => s.start);
    for (const s of sectionStarts) expect(sims.find((x) => x.to === s)!.sim).toBeCloseTo(0, 6);
    const low = chunkDoc({ ...P, strategy: "semantic", size: 200, threshold: 0.05 });
    expect(low.map((x) => x.start)).toEqual([0, ...sectionStarts]);
    const high = chunkDoc({ ...P, strategy: "semantic", size: 200, threshold: 0.3 });
    expect(high.length).toBeGreaterThan(low.length);
  });

  it("counts facts split across a boundary", () => {
    const facts = factSpans(doc.sentences);
    expect(stats([{ index: 0, start: 0, end: doc.words.length }], facts).splitPct).toBe(0);
    const s = stats(chunkDoc({ ...P, strategy: "fixed", size: 30 }), facts);
    expect(s.split.length).toBe(4);
    expect(s.splitPct).toBeCloseTo(0.8, 10);
  });

  it("structure-aware strategies split fewer facts than fixed size at the same budget", () => {
    const all = Object.fromEntries(compareAll({ ...P, size: 30 }).map((r) => [r.strategy, r.stats.split.length]));
    expect(all.heading).toBeLessThan(all.fixed);
    expect(all.semantic).toBeLessThan(all.fixed);
    expect(all.overlap).toBeLessThan(all.fixed);
  });

  it("produces 8 deterministic steps", () => {
    const a = simulate(P), b = simulate(P);
    expect(a.steps.map((s) => s.id)).toEqual(["doc", "candidates", "cut", "overlap", "cards", "sizes", "facts", "compare"]);
    expect(JSON.stringify(a.steps)).toBe(JSON.stringify(b.steps));
    expect(simulate({ ...P, size: 60 }).chunks.length).toBeLessThan(a.chunks.length);
  });
});
