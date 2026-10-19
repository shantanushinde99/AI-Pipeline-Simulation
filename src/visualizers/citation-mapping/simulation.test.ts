import { describe, expect, it } from "vitest";
import { CORPUS } from "../rag/corpus";
import { DEFAULT_CM, anchorFor, mapCitations, simulate } from "./simulation";

const chain = (p = DEFAULT_CM) => mapCitations(p).citations.map((c) => `[${c.marker}]→${c.resolved?.chunkId ?? "none"}:${c.pass ? "pass" : "fail"}`);

describe("citation mapping", () => {
  it("healthy mapping resolves every marker to the chunk the generator read", () => {
    const r = mapCitations(DEFAULT_CM);
    expect(r.prompt.map((p) => p.chunkId)).toEqual(["sso#0", "sso#0", "retention#0", "pricing#0"]);
    expect(r.display.map((p) => p.chunkId)).toEqual(["sso#0", "retention#0", "pricing#0"]);
    expect(chain()).toEqual(["[1]→sso#0:pass", "[3]→retention#0:pass", "[4]→pricing#0:pass"]);
    expect(r.accuracy).toBe(1);
  });

  it("off-by-one: [1] passes by luck (its neighbour is a duplicate), [3] lands on pricing, [4] dangles", () => {
    expect(chain({ ...DEFAULT_CM, bug: "offByOne" })).toEqual(["[1]→sso#0:pass", "[3]→pricing#0:fail", "[4]→none:fail"]);
    expect(mapCitations({ ...DEFAULT_CM, bug: "offByOne", dedupeFirst: true }).accuracy).toBe(0);
  });

  it("renumbering after dedupe shifts markers; deduping before the prompt fixes it", () => {
    expect(chain({ ...DEFAULT_CM, bug: "dedupe" })).toEqual(["[1]→sso#0:pass", "[3]→pricing#0:fail", "[4]→none:fail"]);
    const fixed = mapCitations({ ...DEFAULT_CM, bug: "dedupe", dedupeFirst: true });
    expect(fixed.citations.map((c) => c.marker)).toEqual([1, 2, 3]);
    expect(fixed.accuracy).toBe(1);
  });

  it("a truncated passage fails the strict verifier but fools the lenient one", () => {
    const strict = mapCitations({ ...DEFAULT_CM, bug: "truncated" });
    const c = strict.citations[2];
    expect(c.resolved!.truncated).toBe(true);
    expect(c.resolved!.text.endsWith("Enterprise plan: Price custom pricing;")).toBe(true);
    expect(c.wrongSource).toBe(false);
    expect(c.pass).toBe(false);
    const lenient = mapCitations({ ...DEFAULT_CM, bug: "truncated", strict: false }).citations[2];
    // words: enterprise plan price custom seat unlimited support 24 7 phone → 7 of 10 appear somewhere in the cut passage
    // ("unlimited" from "unlimited projects", "seat"/"support" from other rows); only 24, 7 and phone are missing
    expect(lenient.check!.coverage).toBeCloseTo(0.7, 12);
    expect(lenient.check!.missingNumbers).toEqual(["24", "7"]);
    expect(lenient.pass).toBe(true);
  });

  it("anchors locate the cited span inside the source document", () => {
    const pricing = CORPUS.find((d) => d.id === "pricing")!;
    expect(anchorFor(pricing, "Enterprise plan: Price custom pricing; Seats unlimited; Support 24/7 phone.")).toEqual({ page: 1, anchor: "table “Plan comparison”, row 3" });
    const retention = CORPUS.find((d) => d.id === "retention")!;
    expect(anchorFor(retention, "Deleted projects can be recovered for 7 days before they are permanently removed.").anchor).toBe("#data-retention ¶2");
  });

  it("steps: answer, sources, one resolve per marker, verify, score — deterministic", () => {
    const a = simulate(DEFAULT_CM);
    expect(a.steps.map((s) => s.id)).toEqual(["answer", "sources", "resolve-0", "resolve-1", "resolve-2", "verify", "score"]);
    expect(JSON.stringify(simulate(DEFAULT_CM).steps)).toBe(JSON.stringify(a.steps));
  });
});
