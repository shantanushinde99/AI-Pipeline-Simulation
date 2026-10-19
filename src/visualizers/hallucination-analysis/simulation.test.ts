import { describe, expect, it } from "vitest";
import { ANSWERS, DEFAULT_CONFIG, checkClaim, claimsOf, simulate } from "./simulation";

const get = (id: string) => ANSWERS.find((a) => a.id === id)!;

describe("hallucination checker", () => {
  it("labels the confident-but-wrong answer by hand", () => {
    const { checks, summary } = simulate(get("confident"), DEFAULT_CONFIG);
    expect(checks.map((c) => c.verdict)).toEqual(["supported", "contradicted", "contradicted", "supported", "contradicted", "contradicted"]);
    expect(checks.map((c) => c.type)).toEqual([null, "entity swap", "number drift", null, "number drift", "number drift"]);
    // |3.6 - 2.4| / 2.4 = 0.5
    expect(checks[2].relDiff).toBeCloseTo(0.5, 12);
    expect(summary.rate).toBeCloseTo(4 / 6, 12);
  });

  it("flags fabricated citations and unsupported generalizations as unverifiable", () => {
    const { checks, summary } = simulate(get("citations"), DEFAULT_CONFIG);
    expect(checks.map((c) => c.type)).toEqual([null, "fabricated citation", "unsupported generalization", null, "fabricated citation"]);
    expect(summary.unverifiable).toBe(3);
    expect(summary.rate).toBeCloseTo(0.6, 12);
    expect(simulate(get("citations"), { ...DEFAULT_CONFIG, countUnverifiable: false }).summary.rate).toBe(0);
  });

  it("tolerance decides whether small numeric drift counts", () => {
    const mirror = claimsOf(get("subtle"))[2]; // 2.5 m vs 2.4 m → 4.17 %
    expect(checkClaim(mirror, { ...DEFAULT_CONFIG, tolerance: 0.05 }).verdict).toBe("supported");
    expect(checkClaim(mirror, { ...DEFAULT_CONFIG, tolerance: 0.03 }).verdict).toBe("contradicted");
    // Years are exact even with a huge tolerance: 1989 vs 1998.
    expect(checkClaim(claimsOf(get("confident"))[5], { ...DEFAULT_CONFIG, tolerance: 1 }).verdict).toBe("contradicted");
  });

  it("a missing source turns true claims into unverifiable, not contradicted", () => {
    const { checks } = simulate(get("grounded"), { ...DEFAULT_CONFIG, missing: "S3" });
    expect(checks.slice(3).map((c) => c.verdict)).toEqual(["unverifiable", "unverifiable"]);
    expect(checks.slice(0, 3).every((c) => c.verdict === "supported")).toBe(true);
  });

  it("produces answer → decompose → one check per claim → rate", () => {
    const { steps, claims } = simulate(get("grounded"), DEFAULT_CONFIG);
    expect(steps).toHaveLength(claims.length + 3);
    expect(steps.map((s) => s.state.phase)).toEqual(["answer", "decompose", ...claims.map(() => "check"), "rate"]);
    expect(steps.at(-1)!.state.checked).toBe(claims.length);
    expect(simulate(get("grounded"), DEFAULT_CONFIG)).toEqual(simulate(get("grounded"), DEFAULT_CONFIG));
  });
});
