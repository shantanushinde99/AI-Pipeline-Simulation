import { describe, expect, it } from "vitest";
import { CORPUS } from "./corpus";
import { DEFAULT_RAG, answerQuestion, buildIndex, chunk, embed, parse, runGoldenSet, runRag, sanitize } from "./simulation";
import { dot } from "../../lib/math";

describe("RAG indexing", () => {
  it("chunks with the requested size and overlap and covers every word", () => {
    const p = parse(CORPUS[1], "linearize");
    const words = p.text.split(/\s+/);
    const cs = chunk(p, 20, 5);
    expect(cs[0].endWord - cs[0].startWord).toBe(20);
    expect(cs[1].startWord).toBe(15);
    expect(cs.at(-1)!.endWord).toBe(words.length);
    for (let i = 1; i < cs.length; i++) expect(cs[i].startWord).toBeLessThan(cs[i - 1].endWord + 1);
  });

  it("sanitizes invalid configurations", () => {
    const s = sanitize({ ...DEFAULT_RAG, chunkSize: 10, overlap: 50, topK: 2, contextChunks: 9 });
    expect(s.overlap).toBe(9);
    expect(s.contextChunks).toBe(2);
  });

  it("dropping tables removes their content from the index", () => {
    const has = (i: ReturnType<typeof buildIndex>) => i.chunks.some((c) => c.text.includes("$49"));
    expect(has(buildIndex(DEFAULT_RAG))).toBe(true);
    expect(has(buildIndex({ ...DEFAULT_RAG, tables: "drop" }))).toBe(false);
  });

  it("embeddings are deterministic, unit length, and model versions are incompatible", () => {
    const a = embed("API rate limit", "embed-v1");
    expect(a).toEqual(embed("API rate limit", "embed-v1"));
    expect(dot(a, a)).toBeCloseTo(1, 10);
    expect(dot(a, embed("API rate limit", "embed-v1"))).toBeCloseTo(1);
    expect(Math.abs(dot(a, embed("API rate limit", "embed-v2")))).toBeLessThan(0.5);
  });
});

describe("RAG query flow", () => {
  it("the healthy pipeline answers the whole golden set with correct citations", () => {
    const g = runGoldenSet(DEFAULT_RAG);
    expect(g.accuracy).toBe(1);
    expect(g.citationAccuracy).toBe(1);
    expect(g.faithfulness).toBe(1);
  });

  it("retrieval is sorted and respects top-k; context respects budget and chunk count", () => {
    const r = runRag({ ...DEFAULT_RAG, topK: 3, contextChunks: 2, contextBudget: 100 }, "q-refund");
    expect(r.retrieved).toHaveLength(3);
    for (let i = 1; i < r.scored.length; i++) expect(r.scored[i - 1].score).toBeGreaterThanOrEqual(r.scored[i].score);
    expect(r.context.length).toBeLessThanOrEqual(2);
    expect(r.contextTokens).toBeLessThanOrEqual(100);
  });

  it("every context chunk and citation refers to something actually retrieved", () => {
    const r = runRag(DEFAULT_RAG, "q-price");
    const ids = new Set(r.retrieved.map((h) => h.entry.id));
    r.context.forEach((c) => expect(ids.has(c.entry.id)).toBe(true));
    expect(r.prompt).toContain(r.context[0].text);
    const cited = r.context.find((c) => c.label === r.answer.citations[0].label)!;
    expect(cited.text).toContain(r.answer.sentence);
  });

  it("a small context budget truncates the evidence and the answer goes wrong", () => {
    const r = runRag({ ...DEFAULT_RAG, contextBudget: 50 }, "q-refund");
    expect(r.evaluation.retrievalHit).toBe(true);
    expect(r.evaluation.contextHit).toBe(false);
    expect(r.answer.sentence).toContain("14 days");
  });

  it("a stale index answers faithfully but incorrectly", () => {
    const r = runRag({ ...DEFAULT_RAG, indexSnapshot: "stale" }, "q-rate");
    expect(r.answer.sentence).toContain("300 requests");
    expect(r.evaluation.faithful).toBe(true);
    expect(r.evaluation.correct).toBe(false);
  });

  it("a citation off-by-one leaves the text correct but the source wrong", () => {
    const r = runRag({ ...DEFAULT_RAG, citationBug: true }, "q-price");
    expect(r.evaluation.correct).toBe(true);
    expect(r.evaluation.citationCorrect).toBe(false);
  });

  it("an ungrounded prompt lets the simulated model contradict the context", () => {
    const r = runRag({ ...DEFAULT_RAG, promptTemplate: "concise-v2" }, "q-rate");
    expect(r.answer.source).toBe("prior");
    expect(r.evaluation.contextHit).toBe(true);
    expect(r.evaluation.faithful).toBe(false);
  });

  it("small chunks split the SSO instructions", () => {
    const r = runRag({ ...DEFAULT_RAG, chunkSize: 20, overlap: 0 }, "q-sso");
    expect(r.evaluation.correct).toBe(false);
    expect(runRag(DEFAULT_RAG, "q-sso").evaluation.correct).toBe(true);
  });

  it("a query/index embedding mismatch degrades quality while every request succeeds", () => {
    const g = runGoldenSet({ ...DEFAULT_RAG, queryModel: "embed-v2" });
    expect(g.successRate).toBe(1);
    expect(g.accuracy).toBeLessThan(0.5);
    expect(g.meanTop1).toBeLessThan(runGoldenSet(DEFAULT_RAG).meanTop1 / 3);
  });

  it("is deterministic", () => {
    const index = buildIndex(DEFAULT_RAG);
    const q = { id: "x", text: "How long are logs kept?", expect: "days", sourceDoc: "retention" };
    expect(answerQuestion(index, DEFAULT_RAG, q).answer).toEqual(answerQuestion(index, DEFAULT_RAG, q).answer);
  });
});
