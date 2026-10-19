import { describe, expect, it } from "vitest";
import { ATTACKS, DEFAULT_DEFENSES, MATRIX_COLUMNS, NO_DEFENSES, evaluate, matrix, simulate, successRate } from "./simulation";

const get = (id: string) => ATTACKS.find((a) => a.id === id)!;
const cell = (ok: boolean) => (ok ? "S" : "B");

describe("prompt-injection rule model", () => {
  it("every attack succeeds with no defenses", () => {
    expect(ATTACKS.every((a) => evaluate(a, NO_DEFENSES).success)).toBe(true);
    expect(successRate(NO_DEFENSES).rate).toBe(1);
  });

  it("fills the attack × defense matrix as hand-derived", () => {
    expect(MATRIX_COLUMNS.map((c) => c.id)).toEqual(["none", "spotlight", "hierarchy", "tools", "output", "all"]);
    const m = matrix(true).map((row) => row.map((r) => cell(r.success)).join(""));
    expect(m).toEqual([
      "SSBSBB", // direct override: hierarchy + output filter
      "SSSSBB", // disguised leak: only the output filter (canary)
      "SBBSSB", // poisoned page: spotlighting (random markers) + hierarchy
      "SBSBSB", // email tool hijack: spotlighting + tool permissions
      "SBSSBB", // exfil link: spotlighting + output filter
    ]);
  });

  it("static delimiters let a fake closing tag escape spotlighting", () => {
    const page = get("webpage");
    expect(evaluate(page, { ...NO_DEFENSES, spotlight: true, randomMarkers: true }).success).toBe(false);
    expect(evaluate(page, { ...NO_DEFENSES, spotlight: true, randomMarkers: false }).success).toBe(true);
  });

  it("reports the first layer that stops the attack", () => {
    const all = { spotlight: true, hierarchy: true, tools: true, output: true, randomMarkers: true };
    expect(evaluate(get("direct"), all).stoppedBy).toBe("hierarchy");
    expect(evaluate(get("email"), all).stoppedBy).toBe("spotlight");
    expect(evaluate(get("disguised"), all).stoppedBy).toBe("output");
  });

  it("builds an 8-step trace whose final output matches the verdict", () => {
    const leak = simulate(get("disguised"), DEFAULT_DEFENSES);
    expect(leak.steps.map((s) => s.state.phase)).toEqual(["system", "user", "untrusted", "read", "act", "output", "verdict", "matrix"]);
    expect(leak.result.success).toBe(true);
    expect(leak.final).toContain("ZEBRA-42");
    const filtered = simulate(get("disguised"), { ...DEFAULT_DEFENSES, output: true });
    expect(filtered.final).not.toContain("ZEBRA-42");
    expect(successRate(DEFAULT_DEFENSES)).toEqual({ successes: 2, total: 5, rate: 0.4 });
  });
});
