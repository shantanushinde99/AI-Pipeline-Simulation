import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG, RAW, batch, parse, repairSyntax, retryPassProb, runAttempt, simulate, validate } from "./simulation";

describe("structured output validation", () => {
  it("parses valid JSON and locates syntax errors", () => {
    expect(parse(RAW.valid).ok).toBe(true);
    const tc = parse(RAW["trailing-comma"]);
    expect(tc.ok).toBe(false);
    if (!tc.ok) expect(tc.line).toBe(4); // the `"tags": [...],` line (0-based)
    const tr = parse(RAW.truncated);
    if (!tr.ok) expect(tr.pos).toBe(RAW.truncated.length);
  });

  it("the validator walks every node and reports the right problem", () => {
    const issues = (k: keyof typeof RAW) => validate(JSON.parse(RAW[k])).issues.map((i) => i.path);
    expect(validate(JSON.parse(RAW.valid)).visits.map((v) => v.path)).toEqual(["$", "$.title", "$.priority", "$.estimate_hours", "$.tags", "$.tags[0]", "$.tags[1]"]);
    expect(issues("valid")).toEqual([]);
    expect(issues("wrong-type")).toEqual(["$.estimate_hours"]);
    expect(issues("missing-field")).toEqual(["$.priority"]);
    expect(issues("extra-field")).toEqual(["$.confidence"]);
    expect(issues("enum")).toEqual(["$.priority"]);
  });

  it("repair fixes what it safely can and nothing more", () => {
    expect(runAttempt("trailing-comma", true).accepted).toBe(true);
    expect(runAttempt("trailing-comma", false).accepted).toBe(false);
    expect(runAttempt("wrong-type", true).schemaChanges).toEqual(['coerced estimate_hours "3" → 3']);
    expect(runAttempt("extra-field", true).accepted).toBe(true);
    expect(runAttempt("missing-field", true).accepted).toBe(false);
    expect(runAttempt("enum", true).accepted).toBe(false);
    // Truncation repair makes it parseable but two required fields are gone.
    expect(JSON.parse(repairSyntax(RAW.truncated))).toEqual({ title: "Fix login timeout", priority: "high" });
    expect(runAttempt("truncated", true).finalIssues.map((i) => i.path)).toEqual(["$.estimate_hours", "$.tags"]);
  });

  it("each layer in the batch ladder is at least as good as the previous one", () => {
    const { rows, kinds } = batch(DEFAULT_CONFIG);
    expect(Object.values(kinds).reduce((a, b) => a + b, 0)).toBe(200);
    const [parseOnly, validated, repaired, retried, constrained] = rows;
    expect(parseOnly.badAccepted).toBeGreaterThan(0); // silently accepted bad data
    expect(validated.badAccepted).toBe(0);
    expect(validated.validAccepted).toBe(kinds.valid);
    expect(repaired.validAccepted).toBeGreaterThanOrEqual(validated.validAccepted);
    expect(retried.validAccepted).toBeGreaterThanOrEqual(repaired.validAccepted);
    expect(constrained.validAccepted).toBeGreaterThanOrEqual(retried.validAccepted);
    expect(rows.every((r) => r.validAccepted + r.badAccepted + r.rejected === 200)).toBe(true);
  });

  it("is deterministic, and the seed and error rate change the outcome", () => {
    expect(batch(DEFAULT_CONFIG)).toEqual(batch(DEFAULT_CONFIG));
    expect(batch({ ...DEFAULT_CONFIG, seed: 99 }).kinds).not.toEqual(batch(DEFAULT_CONFIG).kinds);
    expect(batch({ ...DEFAULT_CONFIG, errorRate: 0 }).rows.every((r) => r.validAccepted === 200)).toBe(true);
  });

  it("builds the step list from what actually happens", () => {
    const phases = (k: typeof DEFAULT_CONFIG.kind, extra = {}) => simulate({ ...DEFAULT_CONFIG, kind: k, ...extra }).steps.map((s) => s.state.phase);
    expect(phases("valid")).toEqual(["generate", "parse", "validate", "final", "batch"]);
    expect(phases("wrong-type")).toEqual(["generate", "parse", "validate", "repair-schema", "final", "batch"]);
    expect(phases("trailing-comma")).toEqual(["generate", "parse", "repair-syntax", "validate", "final", "batch"]);
    // Constrained decoding turns a would-be trailing comma into valid output.
    expect(phases("trailing-comma", { constrained: true })).toEqual(["generate", "parse", "validate", "final", "batch"]);
    expect(phases("enum", { maxRetries: 0 })).toEqual(["generate", "parse", "validate", "repair-schema", "final", "batch"]);
  });

  it("retry pass probability matches a hand calculation", () => {
    // 3 of 6 error kinds are repairable → q = 0.7 + 0.3 · 3/6 = 0.85; constrained decoding also removes truncation → 4/6.
    expect(retryPassProb({ retryFix: 0.7, repair: true, constrained: false }).q).toBeCloseTo(0.85, 12);
    expect(retryPassProb({ retryFix: 0.7, repair: true, constrained: true }).q).toBeCloseTo(0.9, 12);
    expect(retryPassProb({ retryFix: 0.7, repair: false, constrained: false }).q).toBeCloseTo(0.7, 12);
  });
});
