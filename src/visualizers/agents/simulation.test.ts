import { describe, expect, it } from "vitest";
import { DEFAULT_AGENT, EXPECTED_TOTAL, calculate, runAgent } from "./simulation";

const run = (faults = {}, extra = {}) => runAgent({ ...DEFAULT_AGENT, ...extra, faults });
const tools = (r: ReturnType<typeof runAgent>) => r.toolCalls.map((c) => c.tool);

describe("calculator tool", () => {
  it("evaluates arithmetic with precedence and parentheses", () => {
    expect(calculate("12 * 49 * 12 * 0.85")).toBeCloseTo(EXPECTED_TOTAL);
    expect(calculate("2 + 3 * (4 - 1)")).toBe(11);
    expect(calculate("-2 * 3")).toBe(-6);
  });
  it("rejects identifiers and junk instead of evaluating them", () => {
    expect(() => calculate("12 * PRO_PRICE")).toThrow(/unknown identifier PRO_PRICE/);
    expect(() => calculate("alert(1)")).toThrow();
    expect(() => calculate("2 +")).toThrow();
  });
});

describe("agent loop", () => {
  it("healthy run: search → read → read → calculate → submit, correct total", () => {
    const r = run();
    expect(tools(r)).toEqual(["search_docs", "get_document", "get_document", "calculator", "submit_result"]);
    expect(r.final.status).toBe("success");
    expect(r.correct).toBe(true);
    expect(r.final.text).toContain("5,997.6");
  });

  it("is deterministic", () => {
    expect(run({ slowTool: true })).toEqual(run({ slowTool: true }));
  });

  it("a wrong first tool fails and the error steers the next decision", () => {
    const r = run({ wrongTool: true });
    expect(r.toolCalls[0].tool).toBe("calculator");
    expect(r.toolCalls[0].ok).toBe(false);
    expect(r.toolCalls[1].tool).toBe("search_docs");
    expect(r.correct).toBe(true);
  });

  it("invalid arguments are rejected before execution and the fallback answer is wrong", () => {
    const r = run({ staleToolSpec: true });
    const calc = r.toolCalls.find((c) => c.tool === "calculator")!;
    expect(calc.validation).toMatch(/missing required argument\(s\) expression/);
    expect(calc.attempts).toHaveLength(0);
    expect(r.final.status).toBe("success");
    expect(r.correct).toBe(false);
  });

  it("retrying non-retryable errors loops until the iteration budget", () => {
    const r = run({ staleDocId: true, retryAllErrors: true });
    expect(r.final.status).toBe("budget_exceeded");
    expect(r.toolCalls.filter((c) => c.tool === "get_document").length).toBeGreaterThan(3);
    const fixed = run({ staleDocId: true });
    expect(fixed.correct).toBe(true);
    expect(run({ staleDocId: true, retryAllErrors: true }, { maxIterations: 20 }).final.status).toBe("budget_exceeded");
  });

  it("timeouts retry, and succeed or fail depending on retries and timeout", () => {
    const failing = run({ slowTool: true }, { maxRetries: 1 });
    expect(failing.final.status).toBe("failed");
    expect(failing.toolCalls[1].attempts.map((a) => a.outcome)).toEqual(["timeout", "timeout"]);
    expect(run({ slowTool: true }, { maxRetries: 3 }).correct).toBe(true);
    expect(run({ slowTool: true }, { timeoutMs: 4000 }).correct).toBe(true);
  });

  it("prompt injection: guardrail blocks the destructive call; disabling it executes the deletion", () => {
    const blocked = run({ trustToolOutput: true }, { task: "summarize" });
    expect(tools(blocked)).toContain("delete_records");
    expect(blocked.toolCalls.find((c) => c.tool === "delete_records")!.attempts[0].outcome).toBe("blocked");
    expect(blocked.final.status).toBe("failed");
    const noGuard = run({ trustToolOutput: true }, { task: "summarize", guardrails: false });
    expect(noGuard.toolCalls.find((c) => c.tool === "delete_records")!.ok).toBe(true);
    expect(noGuard.correct).toBe(false);
    const safe = run({}, { task: "summarize" });
    expect(tools(safe)).not.toContain("delete_records");
    expect(safe.final.text).toContain("ignored it");
    expect(safe.correct).toBe(true);
  });
});
