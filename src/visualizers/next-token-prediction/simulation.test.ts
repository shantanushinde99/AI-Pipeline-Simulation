import { describe, expect, it } from "vitest";
import { generate, occurrences, predict, simulate, tokenize, train } from "./simulation";

const model = train();

describe("n-gram next-token model", () => {
  it("tokenizes to lower-case words and punctuation", () => {
    expect(tokenize("The Cat, sat!")).toEqual(["the", "cat", ",", "sat", "!"]);
    expect(model.tokens).toHaveLength(46);
  });

  it("bigram probabilities equal count / context count (hand-checked)", () => {
    const p = predict(model, ["the"], 2, "backoff");
    expect(p.total).toBe(11);
    expect(p.candidates[0]).toEqual({ token: "cat", count: 4, p: 4 / 11 });
    expect(p.candidates.reduce((s, c) => s + c.p, 0)).toBeCloseTo(1, 12);
    expect(occurrences(model, ["the"])).toHaveLength(11);
  });

  it("trigram uses two words of context", () => {
    const p = predict(model, ["sat", "on"], 3, "backoff");
    expect(p.context).toEqual(["sat", "on"]);
    expect(p.candidates.map((c) => [c.token, c.count])).toEqual([["the", 3], ["a", 1]]);
  });

  it("handles unseen contexts: backoff, add-one, none", () => {
    const back = predict(model, ["the", "zebra"], 2, "backoff");
    expect(back.unseen).toBe(true);
    expect(back.usedOrder).toBe(1); // unigram: 'the' is the most frequent token
    expect(back.candidates[0].token).toBe("the");
    const lap = predict(model, ["zebra"], 2, "laplace");
    expect(lap.candidates).toHaveLength(model.vocab.length);
    expect(lap.candidates[0].p).toBeCloseTo(1 / model.vocab.length, 12);
    expect(predict(model, ["zebra"], 2, "none").candidates).toHaveLength(0);
  });

  it("greedy generation is deterministic and detects the repetition loop", () => {
    const g = generate(model, { prompt: "the cat", order: 2, smoothing: "backoff", n: 8 });
    expect(g.final.join(" ")).toBe("the cat sat on the cat sat on the cat");
    expect(g.loopAt).not.toBeNull();
    expect(generate(model, { prompt: "the cat", order: 2, smoothing: "backoff", n: 8 })).toEqual(g);
    // trigram changes the outcome
    expect(generate(model, { prompt: "the cat", order: 3, smoothing: "backoff", n: 8 }).final.join(" ")).toBe("the cat sat on the mat . the cat sat");
  });

  it("builds count → one step per token → done, and stops when stuck", () => {
    const steps = simulate({ prompt: "the cat", order: 2, smoothing: "backoff", n: 3 }, model);
    expect(steps.map((s) => s.id)).toEqual(["count", "gen-0", "gen-1", "gen-2", "done"]);
    const stuck = simulate({ prompt: "the zebra", order: 2, smoothing: "none", n: 5 }, model);
    expect(stuck.map((s) => s.state.phase)).toEqual(["count", "stuck", "done"]);
    expect(stuck[1].events.some((e) => e.severity === "error")).toBe(true);
  });
});
