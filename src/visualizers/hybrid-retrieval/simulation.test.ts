import { describe, expect, it } from "vitest";
import { DEFAULT_INPUT, bm25Term, idf, runHybrid, simulate, tokenize } from "./simulation";

const run = (queryId: string, extra = {}) => runHybrid({ ...DEFAULT_INPUT, queryId, ...extra });

describe("hybrid retrieval", () => {
  it("BM25 pieces match hand calculations", () => {
    expect(idf(8, 1)).toBeCloseTo(Math.log(6), 12); // ln(1 + 7.5/1.5)
    // tf=2, |D| = avgdl, k1=1.2: 2·2.2 / (2 + 1.2) = 1.375
    expect(bm25Term(1, 2, 10, 10, { k1: 1.2, b: 0.75 })).toBeCloseTo(1.375, 12);
    // longer than average → smaller contribution; b = 0 ignores length
    expect(bm25Term(1, 1, 20, 10, { k1: 1.2, b: 0.75 })).toBeLessThan(bm25Term(1, 1, 10, 10, { k1: 1.2, b: 0.75 }));
    expect(bm25Term(1, 1, 20, 10, { k1: 1.2, b: 0 })).toBeCloseTo(1, 12);
  });

  it("tokenizer keeps identifiers whole", () => {
    expect(tokenize("Error NW-4013, retry-after!")).toEqual(["error", "nw-4013", "retry-after"]);
  });

  it("exact error code: BM25 ranks the right page first, vectors miss it", () => {
    const r = run("code");
    expect(r.bmList[0].id).toBe("D2");
    expect(r.oov).toContain("nw-4013");
    expect(r.recall.bm25).toBe(1);
    expect(r.recall.vector).toBe(0);
  });

  it("paraphrase: vectors win, BM25 finds nothing relevant", () => {
    const r = run("paraphrase");
    expect(r.vecList[0].id).toBe("D5");
    expect(r.recall.bm25).toBe(0);
    expect(r.recall.vector).toBe(1);
  });

  it("mixed query: RRF finds both relevant docs while each list finds one", () => {
    const r = run("mixed");
    expect(r.recall).toEqual({ bm25: 0.5, vector: 0.5, fused: 1 });
    const d3 = r.fused.find((f) => f.id === "D3")!;
    expect(d3.score).toBeCloseTo(1 / (60 + d3.bm25Rank!) + 1 / (60 + d3.vecRank!), 12);
  });

  it("weighted blend with α = 1 reproduces the BM25 order", () => {
    const r = run("mixed", { mode: "weighted", alpha: 1 });
    expect(r.fused.slice(0, r.bmList.length).map((f) => f.id)).toEqual(r.bmList.map((b) => b.id));
  });

  it("steps are deterministic, ordered, and respond to parameters", () => {
    const a = simulate(DEFAULT_INPUT);
    expect(a.steps.map((s) => s.state.phase)).toEqual(["tokenize", "idf", "bm25", "vector", "lists", "fuse", "compare"]);
    expect(simulate(DEFAULT_INPUT)).toEqual(a);
    const b = simulate({ ...DEFAULT_INPUT, b: 0, k1: 2.5 });
    expect(b.run.bmList.map((x) => x.score)).not.toEqual(a.run.bmList.map((x) => x.score));
  });
});
