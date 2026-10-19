import { describe, expect, it } from "vitest";
import { SCENARIOS } from "./registry";
import { scoreAttempt } from "./scoring";

describe.each(SCENARIOS.map((s) => [s.id, s] as const))("scenario %s", (_, s) => {
  const before = s.run();

  it("is broken before any fix and evidence is computed", () => {
    expect(before.resolved).toBe(false);
    expect(before.evidence.length).toBeGreaterThan(3);
    expect(before.evidence.some((e) => e.relevant)).toBe(true);
    before.evidence.forEach((e) => expect(e.lines.length).toBeGreaterThan(0));
  });

  it("has exactly one root cause whose evidence exists", () => {
    expect(s.hypotheses.filter((h) => h.isRootCause)).toHaveLength(1);
    const ids = new Set(before.evidence.map((e) => e.id));
    s.hypotheses.forEach((h) => h.evidenceIds.forEach((id) => expect(ids.has(id), `${h.id} → ${id}`).toBe(true)));
  });

  it("at least one fix resolves it and at least one plausible fix does not", () => {
    const results = s.fixes.map((f) => s.run(f.id).resolved);
    expect(results.some(Boolean)).toBe(true);
    expect(results.some((r) => !r)).toBe(true);
  });

  it("is deterministic", () => {
    expect(JSON.stringify(s.run().metrics)).toBe(JSON.stringify(before.metrics));
  });

  it("scores a correct diagnosis 4/4 and a guess lower", () => {
    const fix = s.fixes.find((f) => s.run(f.id).resolved)!;
    const good = scoreAttempt(s, {
      rootCause: s.hypotheses.find((h) => h.isRootCause)!.id,
      evidence: before.evidence.filter((e) => e.relevant).map((e) => e.id),
      fix: fix.id, verification: s.verifications.find((v) => v.sound)!.id, explanation: "",
    }, before);
    expect(good.total).toBe(4);
    const bad = scoreAttempt(s, {
      rootCause: s.hypotheses.find((h) => !h.isRootCause)!.id, evidence: [],
      fix: s.fixes.find((f) => !s.run(f.id).resolved)!.id, verification: s.verifications.find((v) => !v.sound)!.id, explanation: "",
    }, before);
    expect(bad.total).toBe(0);
  });
});
