import { describe, expect, it } from "vitest";
import { DEFAULT_MEMORY, QUESTION, countTokens, simulate, vectorize, type MemoryConfig } from "./simulation";

const final = (cfg: Partial<MemoryConfig>) => simulate({ ...DEFAULT_MEMORY, ...cfg }).at(-1)!.state;
const slot = (cfg: Partial<MemoryConfig>, s: string) => final(cfg).answer!.slots.find((x) => x.slot === s)!;

describe("short-term and long-term memory", () => {
  it("keyword vectors and cosine scores match a hand calculation", () => {
    // dims: food, date, place, money, person, lodging
    expect(vectorize(QUESTION)).toEqual([2, 2, 1, 0, 0, 1]);
    expect(vectorize("Flight to Lisbon, arrive 12 May")).toEqual([0, 3, 1, 0, 0, 0]);
    const hits = final({}).hits;
    // cos = (2*3 + 1*1) / (sqrt(10) * sqrt(10)) = 0.7
    expect(hits[0].fact.id).toBe("trip");
    expect(hits[0].score).toBeCloseTo(0.7, 10);
    // diet: 6 / (3 * sqrt(10))
    expect(hits[1].score).toBeCloseTo(6 / (3 * Math.sqrt(10)), 10);
    expect(hits.find((h) => h.fact.id === "hotel")!.retrieved).toBe(false); // below the 0.3 threshold
  });

  it("the window never exceeds its token budget and evicts oldest turns first", () => {
    const steps = simulate(DEFAULT_MEMORY);
    for (const s of steps) expect(s.state.used).toBeLessThanOrEqual(DEFAULT_MEMORY.windowTokens);
    const firstEviction = steps.find((s) => s.state.evicted.length)!;
    expect(firstEviction.state.evicted[0].turn.id).toBe("u1");
    expect(countTokens("Hi, I'm Priya.")).toBe(3);
  });

  it("default setup recalls all three facts via summary and long-term memory", () => {
    const a = final({}).answer!;
    expect(a.correct).toBe(3);
    expect(slot({}, "date").source).toBe("long-term");
    expect(slot({}, "diet").source).toBe("summary");
  });

  it("the lossy summary drops the date: without long-term memory it is missing", () => {
    expect(slot({ longTerm: false }, "date")).toMatchObject({ value: null, source: "missing" });
    expect(final({ longTerm: false, summarize: false }).answer!.correct).toBe(0);
  });

  it("a stale fact outranks its correction in retrieval unless the update is still in the window", () => {
    const stale = slot({ conflict: true }, "date");
    expect(stale.value).toBe("12 May");
    expect(stale.correct).toBe(false);
    expect(stale.conflict).toBeDefined();
    const big = slot({ conflict: true, windowTokens: 120 }, "date");
    expect(big).toMatchObject({ value: "19 May", source: "short-term", correct: true });
  });

  it("is deterministic and has one step per exchange plus four", () => {
    expect(simulate(DEFAULT_MEMORY)).toEqual(simulate(DEFAULT_MEMORY));
    expect(simulate(DEFAULT_MEMORY)).toHaveLength(5 + 4);
    expect(simulate({ ...DEFAULT_MEMORY, conflict: true })).toHaveLength(6 + 4);
  });
});
