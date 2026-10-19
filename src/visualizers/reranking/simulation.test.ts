import { describe, expect, it } from "vitest";
import { sigmoid } from "../../lib/math";
import { LATENCY, WEIGHTS, crossScore, detectIntent, features, rerank, simulate, tradeoff } from "./simulation";

describe("simulated cross-encoder reranking", () => {
  it("extracts features as documented", () => {
    const f = features("Can I get a refund on an annual plan?", "Annual plans can be refunded within 30 days of purchase.");
    expect(f.matched.sort()).toEqual(["annual", "plan", "refund"]);
    expect(f.overlap).toBe(1);
    expect(f.phrase).toBe("annual plan");
    expect(f.intent).toBe(true);
    expect(f.negation).toBeNull();
    expect(features("Can I get a refund?", "Refunds are not available.").negation).toBe("not");
    expect(features("Can I get a refund?", "Refunds are not available.", false).negation).toBeNull();
  });

  it("detects intent from the question form", () => {
    expect(detectIntent("How long are logs kept?")).toBe("quantity");
    expect(detectIntent("How do I enable SSO?")).toBe("procedure");
    expect(detectIntent("Can I get a refund?")).toBe("permission");
    expect(detectIntent("SSO pricing")).toBe("lookup");
  });

  it("scores a pair with the weighted logit and a sigmoid", () => {
    const s = crossScore({ overlap: 2 / 3, matched: [], qTerms: [], phrase: null, intent: true, negation: null });
    const z = WEIGHTS.bias + WEIGHTS.overlap * (2 / 3) + WEIGHTS.intent;
    expect(s.z).toBeCloseTo(z, 12);
    expect(s.p).toBeCloseTo(sigmoid(z), 12);
  });

  it("lifts the answer from first-stage rank 7 to rank 1 and improves nDCG@3", () => {
    const r = rerank({ presetId: "refund", n: 10, useNegation: true });
    expect(r.reranked[0].firstRank).toBe(7);
    expect(r.after).toBeGreaterThan(r.before);
    expect(r.after).toBeCloseTo(1, 12);
  });

  it("cannot recover a passage cut by a small N", () => {
    const r = rerank({ presetId: "refund", n: 5, useNegation: true });
    expect(r.reranked.some((p) => p.grade === 2)).toBe(false);
    const t = tradeoff("refund", true);
    expect(t[9].latency).toBeGreaterThan(t[4].latency);
    expect(t[4].latency).toBeCloseTo(LATENCY.fixedMs + r.scored.reduce((s, p) => s + LATENCY.perPairMs + LATENCY.perTokenMs * p.tokens, 0), 9);
  });

  it("without the negation feature a negated passage wins", () => {
    const on = rerank({ presetId: "deleted", n: 10, useNegation: true });
    const off = rerank({ presetId: "deleted", n: 10, useNegation: false });
    expect(on.reranked[0].id).toBe("P5");
    expect(off.reranked[0].f.negation).toBeNull();
    expect(off.reranked[0].text).toMatch(/cannot/);
    expect(off.after).toBeLessThan(on.after);
  });

  it("has first + N pair steps + sort + compare, deterministically", () => {
    const a = simulate({ presetId: "refund", n: 8, useNegation: true });
    expect(a.steps).toHaveLength(11);
    expect(a.steps.map((s) => s.state.phase)).toEqual(["first", ...Array(8).fill("score"), "sort", "compare"]);
    expect(a.steps[8].state.elapsedMs).toBeCloseTo(a.run.latencyMs, 9);
    expect(simulate({ presetId: "refund", n: 8, useNegation: true })).toEqual(a);
  });
});
