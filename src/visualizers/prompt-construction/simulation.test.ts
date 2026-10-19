import { describe, expect, it } from "vitest";
import { QUESTIONS } from "../rag/corpus";
import { DEFAULT_PC, GUARD, simulate, toks } from "./simulation";

const run = (p: Partial<typeof DEFAULT_PC>) => simulate({ ...DEFAULT_PC, ...p });

describe("prompt-construction", () => {
  it("toy tokenizer is 1.3 tokens per word, rounded", () => {
    expect(toks("one two three four five six seven eight nine ten")).toBe(13);
  });

  it("complete prompt: slots in order, one step per slot plus the answer, correct cited answer", () => {
    const s = run({});
    expect(s.slots.map((x) => x.kind)).toEqual(["system", "grounding", "citation", "passage", "passage", "passage", "question", "output"]);
    expect(s.steps).toHaveLength(s.slots.length + 1);
    expect(s.after.text).toBe("The Pro plan allows 600 requests per minute per API key. [1]");
    expect(s.after.correct && s.after.citationOk).toBe(true);
    expect(s.total).toBe(s.slots.reduce((a, x) => a + x.tokens, 0));
    expect(s.steps.at(-2)!.state.used).toBe(s.total);
  });

  it("missing grounding rule → answers from the outdated prior", () => {
    const s = run({ variant: "no-grounding" });
    expect(s.slots.some((x) => x.kind === "grounding")).toBe(false);
    expect(s.after.text).toBe("The Pro plan allows 300 requests per minute.");
    expect(s.after.correct).toBe(false);
    expect(s.before.correct).toBe(true);
  });

  it("no citation format → same sentence, no citation", () => {
    const s = run({ variant: "no-citation" });
    expect(s.after.sentence).toBe(s.before.sentence);
    expect(s.after.cited).toBeNull();
    expect(s.after.citationOk).toBe(false);
  });

  it("passages after the question moves them and changes the answer", () => {
    const s = run({ variant: "passages-after" });
    const qi = s.slots.findIndex((x) => x.kind === "question");
    expect(s.slots.slice(qi + 1).filter((x) => x.kind === "passage")).toHaveLength(3);
    expect(s.after.correct).toBe(false);
  });

  it("injected passage is obeyed unless the data-not-instructions guard is present", () => {
    const bad = run({ variant: "injected" });
    expect(bad.after.from).toBe("injection");
    const guarded = run({ variant: "injected", guard: true });
    expect(guarded.slots.find((x) => x.kind === "grounding")!.text).toContain(GUARD);
    expect(guarded.after.correct).toBe(true);
  });

  it("budget drops passages that do not fit and never exceeds the budget with them", () => {
    for (const q of QUESTIONS) for (const budget of [80, 150, 220, 360]) {
      const s = run({ questionId: q.id, budget });
      const fixed = s.slots.filter((x) => x.kind !== "passage").reduce((a, x) => a + x.tokens, 0);
      expect(s.total).toBeLessThanOrEqual(Math.max(budget, fixed));
    }
    const tight = run({ budget: 150 });
    expect(tight.slots.some((x) => x.kind === "passage" && !x.kept)).toBe(true);
    expect(run({ budget: 80 }).after.from).toBe("none");
  });

  it("is deterministic", () => {
    expect(JSON.stringify(run({ variant: "injected" }))).toBe(JSON.stringify(run({ variant: "injected" })));
  });
});
