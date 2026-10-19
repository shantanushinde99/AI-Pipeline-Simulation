import { describe, expect, it } from "vitest";
import { DEFAULT_LIMITS, PRESETS, callTokens, runAgent, simulate, type PresetId } from "./simulation";

describe("agent termination", () => {
  it("healthy run stops on the final answer with hand-checked token totals", () => {
    const run = runAgent("healthy", DEFAULT_LIMITS);
    expect(run.stop).toBe("final_answer");
    expect(run.iterations).toHaveLength(4);
    // context re-sent each call: 900+0+150, 900+300+150, 900+1200+150, 900+1240+150
    expect(run.iterations.map((i) => i.tokens)).toEqual([1050, 1350, 2250, 2290]);
    expect(run.iterations.at(-1)!.cumTokens).toBe(6940);
    expect(run.cost).toBeCloseTo(0.0694, 10);
  });

  it("each preset trips a different condition first under default limits", () => {
    const fired = (Object.keys(PRESETS) as PresetId[]).map((p) => [p, runAgent(p, DEFAULT_LIMITS).stop]);
    expect(Object.fromEntries(fired)).toEqual({ healthy: "final_answer", loop: "no_progress", slow: "wall_time", expensive: "token_budget" });
  });

  it("without the no-progress detector a loop burns the whole iteration budget", () => {
    const run = runAgent("loop", { ...DEFAULT_LIMITS, noProgressWindow: 0 });
    expect(run.stop).toBe("max_iterations");
    expect(run.iterations).toHaveLength(10);
    expect(run.iterations[4].repeats).toBe(3);
  });

  it("user cancel stops the run at that iteration unless the agent already answered", () => {
    expect(runAgent("slow", { ...DEFAULT_LIMITS, cancelAfter: 2 })).toMatchObject({ stop: "cancelled" });
    expect(runAgent("slow", { ...DEFAULT_LIMITS, cancelAfter: 2 }).iterations).toHaveLength(2);
    expect(runAgent("healthy", { ...DEFAULT_LIMITS, cancelAfter: 4 }).stop).toBe("final_answer");
  });

  it("raising a budget changes which condition fires", () => {
    expect(runAgent("expensive", { ...DEFAULT_LIMITS, maxTokens: 100000 }).stop).toBe("final_answer");
    expect(runAgent("slow", { ...DEFAULT_LIMITS, maxWallMs: 60000 }).stop).toBe("final_answer");
    expect(callTokens(1000)).toBe(2050);
  });

  it("steps: setup, one per iteration, then the stop; deterministic", () => {
    const a = simulate("loop", DEFAULT_LIMITS);
    expect(a.steps).toHaveLength(a.run.iterations.length + 2);
    expect(a.steps.at(-1)!.state.stop).toBe("no_progress");
    expect(a.steps[2].state.usage.iterations).toBe(2);
    expect(simulate("loop", DEFAULT_LIMITS)).toEqual(a);
  });
});
