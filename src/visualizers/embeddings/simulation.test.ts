import { describe, expect, it } from "vitest";
import { DOCS, PRESET_QUERIES, embedText, precisionAtK, rank, score, simulate } from "./simulation";
import { cosine } from "../../lib/math";

describe("vector search", () => {
  it("cosine similarity is computed from the displayed vectors", () => {
    expect(cosine([1, 0], [0, 1])).toBeCloseTo(0);
    expect(cosine([1, 1], [2, 2])).toBeCloseTo(1);
    expect(cosine([1, 0], [-1, 0])).toBeCloseTo(-1);
    expect(Number.isNaN(cosine([0, 0], [1, 1]))).toBe(true);
    const q = PRESET_QUERIES[0].v, d = DOCS[0].v;
    expect(score("cosine", q, d)).toBeCloseTo((q[0] * d[0] + q[1] * d[1]) / (Math.hypot(...q) * Math.hypot(...d)), 12);
  });

  it("ranks in descending score order with stable ties", () => {
    const r = rank([1, 1], "cosine");
    for (let i = 1; i < r.length; i++) expect(r[i - 1].score).toBeGreaterThanOrEqual(r[i].score);
    expect(r.map((x) => x.rank)).toEqual(DOCS.map((_, i) => i + 1));
  });

  it("the cancel query retrieves a similar but irrelevant doc first", () => {
    const q = PRESET_QUERIES.find((p) => p.id === "cancel")!;
    const r = rank(q.v, "cosine");
    expect(q.relevant).not.toContain(r[0].doc.id);
    expect(r[0].doc.title).toBe("Start a subscription");
    expect(r.findIndex((x) => x.doc.id === "d2")).toBeGreaterThan(0);
    expect(precisionAtK(r, 1, q.relevant)).toBe(0);
  });

  it("dot product favours long vectors where cosine does not", () => {
    const q = PRESET_QUERIES.find((p) => p.id === "refund")!;
    expect(rank(q.v, "cosine")[0].doc.id).toBe("d4");
    expect(rank(q.v, "dot")[0].doc.id).toBe("d13");
  });

  it("embeds free text from the lexicon and reports unknown text", () => {
    expect(embedText("reset my password").v).not.toBeNull();
    expect(rank(embedText("reset my password").v!, "cosine")[0].doc.id).toBe("d5");
    expect(embedText("zzz qqq").v).toBeNull();
  });

  it("produces one scoring step per document", () => {
    const { steps } = simulate([1, 1], "cosine", 3, ["d1"]);
    expect(steps.filter((s) => s.state.phase === "score")).toHaveLength(DOCS.length);
    expect(steps.at(-1)!.state.phase).toBe("judge");
  });
});
