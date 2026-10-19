import { describe, expect, it } from "vitest";
import { dot } from "../../lib/math";
import { embed } from "../rag/simulation";
import { CASES, buildCases, rank, simulate } from "./simulation";

describe("retrieval failures", () => {
  it("every case: healthy and fixed put the needed chunk in the top-3, the fault pushes it out", () => {
    for (const c of CASES) {
      expect(c.healthy.relRank, c.id).not.toBeNull();
      expect(c.healthy.relRank!, c.id).toBeLessThanOrEqual(3);
      expect(c.fixed.relRank!, c.id).toBeLessThanOrEqual(3);
      expect(c.broken.relRank === null || c.broken.relRank > 3, c.id).toBe(true);
    }
  });

  it("scores are cosine similarities: the row score equals the dot product of unit vectors", () => {
    const c = CASES.find((x) => x.id === "vocab")!;
    const top = c.healthy.rows[0];
    expect(top.score).toBeCloseTo(dot(embed(c.healthy.query, "embed-v1"), embed(top.text, "embed-v1")), 10);
    const s = c.healthy.rows.map((r) => r.score);
    expect([...s].sort((a, b) => b - a)).toEqual(s);
  });

  it("stale index: the fact is absent; wrong filter: the fact is excluded before ranking", () => {
    expect(CASES.find((x) => x.id === "stale")!.broken.relStatus).toBe("missing");
    const f = CASES.find((x) => x.id === "filter")!.broken;
    expect(f.relStatus).toBe("excluded");
    expect(f.rows.length + f.excluded.length).toBe(f.poolSize);
  });

  it("dedupe removes the mirrored copies", () => {
    const d = CASES.find((x) => x.id === "dup")!;
    expect(d.fixed.excluded.length).toBe(3);
    expect(d.broken.rows.length).toBe(d.fixed.rows.length + 3);
  });

  it("embedding model mismatch collapses similarity", () => {
    const e = CASES.find((x) => x.id === "embed")!;
    expect(e.broken.rows[0].score).toBeLessThan(e.healthy.rows[0].score);
  });

  it("is deterministic and top-k changes the verdict", () => {
    expect(JSON.stringify(buildCases())).toBe(JSON.stringify(CASES));
    const vocab = CASES.find((x) => x.id === "vocab")!;
    const s1 = simulate("vocab", 1), s5 = simulate("vocab", 5);
    expect(s1.inK(vocab.healthy)).toBe(vocab.healthy.relRank === 1);
    expect(s5.inK(vocab.healthy)).toBe(true);
    expect(s1.steps.map((s) => s.id)).toEqual(["healthy", "broken", "diagnose", "fixed", "summary"]);
    expect(simulate("vocab", 99).k).toBe(5);
  });

  it("rank() with an empty pool is safe", () => {
    const r = rank("empty", "anything", [], "x");
    expect(r.relStatus).toBe("missing");
    expect(r.rows).toEqual([]);
  });
});
