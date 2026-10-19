import type { Scenario } from "./types";

export type Attempt = { rootCause: string; evidence: string[]; fix: string; verification: string; explanation: string };
export type Criterion = { label: string; score: number; note: string };

/**
 * Scores reasoning, not phrasing. The fix is judged by actually re-running the simulation with it applied.
 * The written explanation is shown back for self-review but not auto-graded.
 */
export function scoreAttempt(s: Scenario, a: Attempt, before = s.run()) {
  const rc = s.hypotheses.find((h) => h.id === a.rootCause);
  const relevantIds = new Set(before.evidence.filter((e) => e.relevant).map((e) => e.id));
  const rel = a.evidence.filter((id) => relevantIds.has(id)).length;
  const irr = a.evidence.length - rel;
  const evidenceScore = rel === 0 ? 0 : irr <= rel ? 1 : 0.5;
  const fixOutcome = s.run(a.fix);
  const ver = s.verifications.find((v) => v.id === a.verification);
  const criteria: Criterion[] = [
    { label: "Root cause", score: rc?.isRootCause ? 1 : 0, note: rc?.isRootCause ? "Correct." : `“${rc?.text ?? "none"}” is not the cause. ${rc?.explanation ?? ""}` },
    { label: "Supporting evidence", score: evidenceScore, note: rel === 0 ? "None of the evidence you selected bears on the actual cause." : `${rel} relevant, ${irr} not relevant to the cause.${evidenceScore < 1 ? " Focus on observations that distinguish the cause from the alternatives." : ""}` },
    { label: "Fix addresses the cause", score: fixOutcome.resolved ? 1 : 0, note: fixOutcome.resolved ? "Re-running with your fix resolves the incident." : "Re-running with your fix does not resolve the incident." },
    { label: "Verification plan", score: ver?.sound ? 1 : 0, note: ver?.explanation ?? "" },
  ];
  return { criteria, total: criteria.reduce((t, c) => t + c.score, 0), max: criteria.length, after: fixOutcome };
}
