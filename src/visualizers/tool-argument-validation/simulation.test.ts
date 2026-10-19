import { describe, expect, it } from "vitest";
import { CANDIDATES, friendlyMessage, jsonLines, repair, schemaNodes, simulateValidation, validate } from "./simulation";

const strict = { coerce: false, allErrors: true };
const lenient = { coerce: true, allErrors: true };

describe("JSON Schema argument validation", () => {
  it("walks every schema node depth-first and accepts valid arguments", () => {
    expect(schemaNodes().map((n) => n.path)).toEqual(["", "/guest", "/guest/name", "/guest/email", "/check_in", "/nights", "/room", "/breakfast"]);
    const r = validate(CANDIDATES.valid.args, strict);
    expect(r.valid).toBe(true);
    expect(r.visits).toHaveLength(8);
    expect(r.visits.every((v) => v.status === "ok")).toBe(true);
  });

  it("reports each kind of error at the right path", () => {
    const one = (id: keyof typeof CANDIDATES) => validate(CANDIDATES[id].args, strict).errors.map((e) => [e.instancePath, e.keyword]);
    expect(one("missing")).toEqual([["", "required"]]);
    expect(one("wrongType")).toEqual([["/nights", "type"]]);
    expect(one("enum")).toEqual([["/room", "enum"]]);
    expect(one("range")).toEqual([["/nights", "maximum"]]);
    expect(one("extra")).toEqual([["/discount_code", "additionalProperties"]]);
    expect(one("nested")).toEqual([["/guest/email", "type"]]);
  });

  it("coercion mode converts lossless scalars but cannot fix a bad value", () => {
    const t = validate(CANDIDATES.wrongType.args, lenient);
    expect(t.valid).toBe(true);
    expect(t.coercions).toBe(1);
    expect(t.data.nights).toBe(3);
    // 42 → "42" is a string now, but still not an email
    const n = validate(CANDIDATES.nested.args, lenient);
    expect(n.errors.map((e) => [e.instancePath, e.keyword])).toEqual([["/guest/email", "pattern"]]);
  });

  it("stop-at-first-error reports one error and skips the remaining nodes", () => {
    const all = validate(CANDIDATES.several.args, strict);
    expect(all.errors).toHaveLength(5); // extra notes, guest.email, check_in pattern, nights minimum, room enum
    const first = validate(CANDIDATES.several.args, { coerce: false, allErrors: false });
    expect(first.errors).toHaveLength(1);
    expect(first.visits).toHaveLength(1);
  });

  it("produces model-friendly messages and a repair that can change meaning", () => {
    const r = validate(CANDIDATES.range.args, strict);
    const msg = friendlyMessage(r.errors);
    expect(msg).toContain("- nights: must be at most 14 (you sent 30).");
    const fixed = repair(CANDIDATES.range.args, r.errors);
    expect(fixed.args.nights).toBe(14);
    expect(fixed.actions[0].kind).toBe("clamped");
    expect(validate(fixed.args, strict).valid).toBe(true);
    // a missing fact cannot be repaired: the model must ask
    const m = repair(CANDIDATES.missing.args, validate(CANDIDATES.missing.args, strict).errors);
    expect(m.actions).toEqual([{ path: "/check_in", kind: "ask_user", text: "check_in was never given: ask the user" }]);
  });

  it("steps: propose, one per node, then format → repair → revalidate", () => {
    const { steps } = simulateValidation({ candidate: "enum", coerce: false, allErrors: true });
    expect(steps.map((s) => s.state.phase)).toEqual(["propose", ...Array(8).fill("node"), "format", "repair", "revalidate"]);
    expect(steps.at(-1)!.state.verdict?.tone).toBe("ok");
    const valid = simulateValidation({ candidate: "valid", coerce: false, allErrors: true }).steps;
    expect(valid.at(-1)!.state.phase).toBe("execute");
    expect(jsonLines(CANDIDATES.valid.args).filter((l) => l.path).map((l) => l.path)).toEqual(["/guest", "/guest/name", "/guest/email", "/check_in", "/nights", "/room", "/breakfast"]);
  });
});
