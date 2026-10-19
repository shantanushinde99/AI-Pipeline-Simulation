import { describe, expect, it } from "vitest";
import { DEFAULT_EVAL, compare, evaluateConfig, simulate } from "./simulation";

describe("RAG golden-set evaluation", () => {
  it("baseline scores 1.0 on every metric", () => {
    const a = evaluateConfig("baseline", 4);
    expect(a.rows).toHaveLength(6);
    expect(Object.values(a.means)).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it("recall@k and RR follow their definitions on a hand-checked case", () => {
    // chunk 20 / overlap 5 at k=2: "$49" lives in two overlapping chunks, only one makes the top-2.
    const q = evaluateConfig("chunk20", 2).rows.find((r) => r.qid === "q-price")!;
    expect(q.relevant).toHaveLength(2);
    expect(q.retrieved.filter((h) => h.relevant)).toHaveLength(1);
    expect(q.m.recall).toBe(0.5);
    expect(q.m.rr).toBe(1 / q.firstRelRank!);
    // q-sso: the SSO instruction chunk falls to rank 18 of 19 → not in top-4 → RR 0, answer wrong
    const sso = evaluateConfig("chunk20", 4).rows.find((r) => r.qid === "q-sso")!;
    expect(sso.indexRank).toBe(18);
    expect(sso.m).toEqual({ recall: 0, rr: 0, contextHit: 0, faithful: 1, correct: 0, citation: 1 });
  });

  it("means are averages of per-question scores (MRR = mean RR)", () => {
    const e = evaluateConfig("embed", 4);
    const mrr = e.rows.reduce((s, r) => s + r.m.rr, 0) / e.rows.length;
    expect(e.means.rr).toBeCloseTo(mrr, 12);
    expect(e.means.rr).toBeCloseTo(0.25, 12); // only q-429 rank 1 and q-refund rank 2: (1 + 0.5) / 6
  });

  it("A/B comparison reports deltas and exactly the flipped questions", () => {
    const { deltas, flips } = compare(evaluateConfig("baseline", 4), evaluateConfig("budget60", 4));
    expect(flips.map((f) => `${f.qid}.${f.metric}`)).toEqual(["q-refund.contextHit", "q-refund.correct"]);
    expect(deltas.correct).toBeCloseTo(-1 / 6, 12);
    expect(deltas.recall).toBe(0); // retrieval was fine: the fact was lost in context assembly
    expect(compare(evaluateConfig("baseline", 4), evaluateConfig("norerank", 4)).flips).toHaveLength(0);
  });

  it("faults move the metric they should", () => {
    expect(evaluateConfig("citation", 4).means.citation).toBe(0);
    expect(evaluateConfig("citation", 4).means.correct).toBe(1);
    const p = evaluateConfig("prompt", 4).means;
    expect(p.faithful).toBeCloseTo(4 / 6, 12);
    expect(p.recall).toBe(1);
  });

  it("steps: intro, one per question, aggregate (+ flips in A/B), deterministic", () => {
    const ab = simulate(DEFAULT_EVAL);
    expect(ab.steps.map((s) => s.state.phase)).toEqual(["intro", ...Array(6).fill("question"), "aggregate", "flips"]);
    expect(simulate({ ...DEFAULT_EVAL, mode: "single" }).steps).toHaveLength(8);
    expect(JSON.stringify(simulate(DEFAULT_EVAL).steps)).toBe(JSON.stringify(ab.steps));
    expect(evaluateConfig("baseline", 99).cfg.topK).toBe(6);
  });
});
