import { describe, expect, it } from "vitest";
import { MODEL_TURN_MS, simulateCalling, validateArgs } from "./simulation";

describe("tool calling sequence", () => {
  it("sequential two-city run follows parse → validate → execute → result → tool message for each call", () => {
    const run = simulateCalling({ question: "two", mode: "sequential" });
    const kinds = run.steps.map((s) => s.state.arrows.at(-1)!.kind);
    expect(kinds).toEqual(["user", "tool_call", "parse", "validate", "execute", "result", "tool_msg", "tool_call", "parse", "validate", "execute", "result", "tool_msg", "final"]);
    expect(run.final.status).toBe("done");
    expect(run.final.modelTurns).toBe(3);
    // transcript roles: user, assistant(call), tool, assistant(call), tool, assistant(final)
    expect(run.final.transcript.map((m) => m.role)).toEqual(["user", "assistant", "tool", "assistant", "tool", "assistant"]);
    expect(run.final.detail).toEqual({ kind: "final", text: "Tokyo is warmer: 24 °C (sunny) vs 18 °C (cloudy) in Paris." });
  });

  it("parallel mode emits both calls in one turn and waits max(latency), not the sum", () => {
    const seq = simulateCalling({ mode: "sequential" });
    const par = simulateCalling({ mode: "parallel" });
    expect(par.final.modelTurns).toBe(2);
    expect(par.steps[1].state.transcript[1]).toMatchObject({ role: "assistant", tool_calls: [{ id: "call_1" }, { id: "call_2" }] });
    expect(seq.wallToolMs).toBe(120 + 180);
    expect(par.wallToolMs).toBe(Math.max(120, 180));
    expect(seq.final.clockMs).toBe(3 * MODEL_TURN_MS + 300);
    expect(par.final.clockMs).toBe(2 * MODEL_TURN_MS + 180);
    // tool messages keep their own tool_call_id
    expect(par.final.transcript.filter((m) => m.role === "tool").map((m) => m.role === "tool" && m.tool_call_id)).toEqual(["call_1", "call_2"]);
  });

  it("malformed JSON is returned to the model as a tool error and the model recovers", () => {
    const run = simulateCalling({ question: "one", mode: "malformed", onParseError: "return" });
    const parse = run.steps.find((s) => s.state.detail.kind === "parse")!;
    expect(parse.state.detail.kind === "parse" && parse.state.detail.items[0].ok).toBe(false);
    const errMsg = run.final.transcript.find((m) => m.role === "tool" && m.isError);
    expect(errMsg).toBeDefined();
    expect(run.final.status).toBe("done");
    expect(run.final.toolExecutions).toBe(1); // the bad call never executed
    expect(run.final.modelTurns).toBe(3);
  });

  it("malformed JSON with no error handling crashes the run", () => {
    const run = simulateCalling({ question: "one", mode: "malformed", onParseError: "crash" });
    expect(run.final.status).toBe("crashed");
    expect(run.final.toolExecutions).toBe(0);
    expect(run.final.arrows.at(-1)).toMatchObject({ from: "runtime", to: "user", kind: "crash" });
  });

  it("validateArgs flags wrong type, enum and extra fields", () => {
    expect(validateArgs("get_weather", { city: "Paris" }).every((c) => c.ok)).toBe(true);
    const bad = validateArgs("get_weather", { city: 3, units: "f" });
    expect(bad.filter((c) => !c.ok).map((c) => c.label)).toEqual(["city is a string", 'city is one of "Paris", "Tokyo"', "no extra fields (found units)"]);
  });

  it("is deterministic and rows are unique", () => {
    const a = simulateCalling({ mode: "parallel" });
    expect(simulateCalling({ mode: "parallel" })).toEqual(a);
    const rows = a.final.arrows.map((x) => x.row);
    expect(new Set(rows).size).toBe(rows.length);
  });
});
