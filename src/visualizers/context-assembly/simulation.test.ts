import { describe, expect, it } from "vitest";
import { CHUNKS, DEFAULT_INPUT, FACT, SIM, assemble, bothEnds, jaccard, simulate, truncate } from "./simulation";

const run = (p = {}) => assemble({ ...DEFAULT_INPUT, ...p });

describe("context assembly", () => {
  it("jaccard matches a hand calculation and flags the planted duplicates", () => {
    // {refund, annual, plan} vs {refund, monthly, plan}: 2 shared / 4 total
    expect(jaccard("refund annual plan", "refund monthly plan")).toBe(0.5);
    expect(SIM[0][1]).toBeGreaterThan(0.6); // C1 ~ C2
    expect(SIM[2][3]).toBeGreaterThan(0.6); // C3 ~ C4
    expect(SIM[0][4]).toBeLessThan(0.3);
  });

  it("greedy by rank fills the budget with duplicates and truncates the answer away", () => {
    const a = run();
    expect(a.decisions.map((d) => d.action)).toEqual(["add", "add", "add", "add", "truncate"]);
    expect(a.used).toBeLessThanOrEqual(130);
    expect(a.factChunk).toBeUndefined();
  });

  it("dedupe skips near-duplicates and the fact survives", () => {
    const a = run({ strategy: "dedupe" });
    expect(a.dropped.filter((d) => d.kind === "duplicate").map((d) => d.id)).toEqual(["C2", "C4"]);
    expect(a.factChunk?.id).toBe("C5");
    expect(a.used).toBeLessThanOrEqual(130);
  });

  it("MMR score = λ·rel − (1−λ)·max sim, and low λ chases filler", () => {
    const a = run({ strategy: "mmr", lambda: 0.5 });
    const second = a.decisions[1];
    const row = second.mmr!.find((r) => r.id === second.id)!;
    const c = CHUNKS.find((x) => x.id === second.id)!;
    expect(row.score).toBeCloseTo(0.5 * c.rel - 0.5 * row.maxSim, 12);
    expect(run({ strategy: "mmr", lambda: 0.3 }).factChunk).toBeUndefined();
    expect(run({ strategy: "mmr", lambda: 0.7 }).factChunk?.id).toBe("C5");
  });

  it("truncation respects the room and drop mode keeps trying smaller chunks", () => {
    const t = truncate(CHUNKS[4].text, 20);
    expect(t.tokens).toBeLessThanOrEqual(20);
    expect(t.text.includes(FACT)).toBe(false);
    const d = run({ overflow: "drop" });
    expect(d.decisions.length).toBe(CHUNKS.length);
    expect(d.picked.every((p) => !p.truncated)).toBe(true);
  });

  it("both-ends ordering puts ranks 1 and 2 at the edges", () => {
    expect(bothEnds([1, 2, 3, 4, 5])).toEqual([1, 3, 5, 4, 2]);
    expect(run({ strategy: "dedupe", ordering: "ends" }).order[0]).toBe("C1");
  });

  it("steps: candidates, one per decision, order, result; deterministic and parameter-sensitive", () => {
    const a = simulate(DEFAULT_INPUT);
    expect(a.steps.map((s) => s.state.phase)).toEqual(["candidates", ...a.asm.decisions.map(() => "consider"), "order", "result"]);
    expect(simulate(DEFAULT_INPUT)).toEqual(a);
    expect(simulate({ ...DEFAULT_INPUT, budget: 260 }).asm.factChunk?.id).toBe("C5");
  });
});
