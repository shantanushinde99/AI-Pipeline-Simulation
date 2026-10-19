import { describe, expect, it } from "vitest";
import { FIRST_MERGE_ID, UNK_ID, applyMerge, corpusWords, countPairs, encode, pretokenize, simulate, trainBPE } from "./simulation";

describe("BPE tokenizer", () => {
  it("pre-tokenizes GPT-2 style: leading spaces stay with words, digits and punctuation split", () => {
    expect(pretokenize("the cat, 42")).toEqual(["the", " cat", ",", " 42"]);
    expect(pretokenize("a  b")).toEqual(["a", " ", " b"]);
  });

  it("counts pairs weighted by word frequency (hand-checked)", () => {
    const words = [{ text: "aab", count: 3 }, { text: "ab", count: 2 }];
    const pairs = countPairs(words, [["a", "a", "b"], ["a", "b"]]);
    // (a,b): 3 + 2 = 5, (a,a): 3
    expect(pairs[0]).toEqual({ pair: ["a", "b"], count: 5 });
    expect(pairs[1]).toEqual({ pair: ["a", "a"], count: 3 });
    expect(applyMerge(["a", "a", "a"], ["a", "a"]).out).toEqual(["aa", "a"]); // left to right, non-overlapping
  });

  it("learns the most frequent pair first and stops when no pair repeats", () => {
    const t = trainBPE(200);
    expect(t.merges[0].pair).toEqual(["a", "t"]);
    expect(t.merges[0].id).toBe(FIRST_MERGE_ID);
    // first merge count equals the sum of its contributions
    expect(t.merges[0].contributions.reduce((s, c) => s + c.freq * c.occ, 0)).toBe(t.merges[0].count);
    expect(t.exhaustedAt).toBe(t.merges.length);
    expect(t.merges.at(-1)!.count).toBeGreaterThanOrEqual(2);
    for (const m of t.merges) expect(m.top[0].count).toBe(m.count);
  });

  it("more merges never produce more tokens; 0 merges is character-level", () => {
    const text = "the fastest rats were sitting";
    expect(encode(text, trainBPE(0).merges).tokens).toHaveLength(text.length);
    let prev = Infinity;
    for (const n of [0, 5, 10, 20, 39]) {
      const len = encode(text, trainBPE(n).merges).tokens.length;
      expect(len).toBeLessThanOrEqual(prev);
      prev = len;
    }
    expect(encode("the cat", trainBPE(39).merges).tokens.map((t) => t.text)).toEqual(["th", "e", " cat"]);
  });

  it("encodes unseen words into pieces and marks non-ASCII as <unk>", () => {
    const merges = trainBPE(16).merges;
    const e = encode("the fastest café", merges);
    const fastest = e.tokens.filter((t) => t.chunk === 1).map((t) => t.text);
    expect(fastest).toEqual([" ", "f", "a", "st", "est"]);
    expect(e.unseen).toContain(1);
    expect(e.tokens.some((t) => t.id === UNK_ID)).toBe(true);
    // decoding (concatenating) gives back the text, except <unk>
    expect(encode("the cat sat", merges).tokens.map((t) => t.text).join("")).toBe("the cat sat");
  });

  it("builds deterministic steps: corpus, one per merge, split, encode, compare", () => {
    const steps = simulate(5, "the cat");
    expect(steps.map((s) => s.id)).toEqual(["corpus", "merge-0", "merge-1", "merge-2", "merge-3", "merge-4", "split", "encode", "compare"]);
    expect(simulate(5, "the cat")).toEqual(steps);
    expect(corpusWords().find((w) => w.text === " the")!.count).toBeGreaterThan(5);
    expect(simulate(0, "").at(-1)!.state.encoded.tokens).toHaveLength(0);
  });
});
