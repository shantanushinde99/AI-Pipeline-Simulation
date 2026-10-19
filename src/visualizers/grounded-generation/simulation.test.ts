import { describe, expect, it } from "vitest";
import { bestSupport, checkSupport, wrap } from "./claims";
import { DEFAULT_GG, evidenceFor, generate, simulate } from "./simulation";

describe("claim support checker", () => {
  it("computes coverage by hand and requires numbers to match in strict mode", () => {
    // claim tokens: pro, plan, allow, 300, request, minute (6); span has all but 300 → 5/6
    const c = checkSupport("The Pro plan allows 300 requests per minute.", "The Pro plan allows 600 requests per minute per API key.");
    expect(c.claimTokens).toEqual(["pro", "plan", "allow", "300", "request", "minute"]);
    expect(c.coverage).toBeCloseTo(5 / 6, 12);
    expect(c.missingNumbers).toEqual(["300"]);
    expect(c.supported).toBe(false);
    expect(checkSupport("The Pro plan allows 300 requests per minute.", "The Pro plan allows 600 requests per minute.", false).supported).toBe(true);
  });

  it("picks the supporting span over a merely similar one", () => {
    const spans = [{ id: "1.1", passage: 1, text: "The Starter plan allows 60 requests per minute." }, { id: "1.2", passage: 1, text: "The Pro plan allows 600 requests per minute." }];
    expect(bestSupport("The Pro plan allows 600 requests per minute.", spans).spanId).toBe("1.2");
    expect(bestSupport("Bananas are yellow.", spans).spanId).toBeNull();
  });

  it("wraps text without exceeding the width", () => {
    const lines = wrap("one two three four five six seven", 10);
    expect(lines).toEqual(["one two", "three four", "five six", "seven"]);
  });
});

describe("grounded generator", () => {
  it("healthy rate-limit answer is fully grounded and links to the right spans", () => {
    const r = generate(DEFAULT_GG);
    expect(r.claims).toBe(2);
    expect(r.groundedness).toBe(1);
    expect(r.sentences[0].text).toContain("600 requests per minute");
    expect(r.sentences.every((s) => s.check?.spanId === s.evidence?.best?.spanId)).toBe(true);
  });

  it("evidence is the share of search terms present", () => {
    const ev = evidenceFor("Pro requests per minute", [{ id: "a", passage: 1, text: "The Starter plan allows 60 requests per minute." }]);
    expect(ev.terms).toEqual(["pro", "request", "minute"]);
    expect(ev.best!.coverage).toBeCloseTo(2 / 3, 12);
  });

  it("missing fact → abstains; with abstention off → an unsupported guess lowers groundedness", () => {
    const abstain = generate({ ...DEFAULT_GG, missingFact: true });
    expect(abstain.sentences[0].kind).toBe("abstain");
    expect(abstain.groundedness).toBe(1); // the remaining claim is still supported
    const guess = generate({ ...DEFAULT_GG, missingFact: true, abstain: false });
    expect(guess.sentences[0].origin).toBe("guess");
    expect(guess.sentences[0].check!.supported).toBe(false);
    expect(guess.groundedness).toBe(0.5);
  });

  it("prior knowledge adds an unsupported claim; low τ lets a wrong-but-grounded sentence through", () => {
    const r = generate({ ...DEFAULT_GG, allowPrior: true });
    expect(r.claims).toBe(3);
    expect(r.groundedness).toBeCloseTo(2 / 3, 12);
    const lowTau = generate({ ...DEFAULT_GG, missingFact: true, tau: 0.5 });
    expect(lowTau.sentences[0].text).toContain("Starter plan"); // grounded, but answers a different question
    expect(lowTau.sentences[0].check!.supported).toBe(true);
  });

  it("steps are ordered, deterministic and end with the score", () => {
    const a = simulate({ ...DEFAULT_GG, allowPrior: true });
    expect(a.steps.map((s) => s.state.phase)).toEqual(["context", "plan", "write", "write", "write", "score"]);
    expect(a.steps.map((s) => s.state.shown)).toEqual([0, 0, 1, 2, 3, 3]);
    expect(JSON.stringify(simulate({ ...DEFAULT_GG, allowPrior: true }).steps)).toBe(JSON.stringify(a.steps));
    const allAbstain = generate({ ...DEFAULT_GG, scenarioId: "sso", tau: 1, missingFact: true });
    expect(allAbstain.claims).toBeGreaterThan(0);
    expect(generate({ ...DEFAULT_GG, tau: 99 }).params.tau).toBe(1);
  });
});
