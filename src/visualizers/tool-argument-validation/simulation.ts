import type { SimulationEvent, SimulationStep } from "../../types/simulation";

// ---------------- schema ----------------
export type Schema = {
  type: "object" | "string" | "integer" | "number" | "boolean";
  properties?: Record<string, Schema>;
  required?: string[];
  additionalProperties?: boolean;
  enum?: string[];
  minimum?: number;
  maximum?: number;
  minLength?: number;
  pattern?: string;
  /** Human hint used in model-facing error messages for patterns. */
  hint?: string;
  /** Short label for the pattern, shown on the tree node. */
  short?: string;
};

export const TOOL_NAME = "create_booking";
export const SCHEMA: Schema = {
  type: "object",
  required: ["guest", "check_in", "nights", "room"],
  additionalProperties: false,
  properties: {
    guest: {
      type: "object", required: ["name", "email"], additionalProperties: false,
      properties: {
        name: { type: "string", minLength: 1 },
        email: { type: "string", pattern: "^[^@\\s]+@[^@\\s]+\\.[a-z]{2,}$", hint: "an email address like ana@example.com", short: "email" },
      },
    },
    check_in: { type: "string", pattern: "^\\d{4}-\\d{2}-\\d{2}$", hint: "a date in YYYY-MM-DD format", short: "YYYY-MM-DD" },
    nights: { type: "integer", minimum: 1, maximum: 14 },
    room: { type: "string", enum: ["single", "double", "suite"] },
    breakfast: { type: "boolean" },
  },
};

export type TreeNode = { path: string; key: string; depth: number; schema: Schema };
/** Depth-first list of schema nodes: the order the validator walks them. */
export function schemaNodes(schema: Schema = SCHEMA): TreeNode[] {
  const out: TreeNode[] = [];
  const rec = (s: Schema, path: string, key: string, depth: number) => {
    out.push({ path, key, depth, schema: s });
    for (const [k, child] of Object.entries(s.properties ?? {})) rec(child, `${path}/${k}`, k, depth + 1);
  };
  rec(schema, "", "arguments", 0);
  return out;
}

export function constraintText(s: Schema): string {
  if (s.type === "object") return `object · ${(s.required ?? []).length} required${s.additionalProperties === false ? " · no extra fields" : ""}`;
  if (s.enum) return `enum: ${s.enum.join(" | ")}`;
  if (s.minimum !== undefined || s.maximum !== undefined) return `${s.type} · ${s.minimum ?? "−∞"}–${s.maximum ?? "∞"}`;
  if (s.pattern) return `${s.type} · ${s.short ?? "pattern"}`;
  if (s.minLength) return `${s.type} · non-empty`;
  return `${s.type} · optional`;
}

// ---------------- candidates ----------------
type Json = string | number | boolean | null | Json[] | { [k: string]: Json };
export type Args = Record<string, Json>;
const BASE: Args = { guest: { name: "Ana Ruiz", email: "ana@example.com" }, check_in: "2026-11-03", nights: 3, room: "double", breakfast: true };
const without = (o: Args, k: string) => Object.fromEntries(Object.entries(o).filter(([x]) => x !== k));

export const CANDIDATES = {
  valid: { label: "Valid arguments", args: BASE },
  missing: { label: "Missing required field (check_in)", args: without(BASE, "check_in") },
  wrongType: { label: "Wrong type (nights as a string)", args: { ...BASE, nights: "3" } },
  enum: { label: "Enum violation (room: \"Double\")", args: { ...BASE, room: "Double" } },
  range: { label: "Out of range (nights: 30)", args: { ...BASE, nights: 30 } },
  extra: { label: "Extra property (discount_code)", args: { ...BASE, discount_code: "VIP50" } },
  nested: { label: "Nested error (guest.email: 42)", args: { ...BASE, guest: { name: "Ana Ruiz", email: 42 } } },
  several: { label: "Several mistakes at once", args: { guest: { name: "Ana Ruiz" }, check_in: "11/03/2026", nights: 0, room: "deluxe", notes: "late arrival" } },
} satisfies Record<string, { label: string; args: Args }>;
export type CandidateId = keyof typeof CANDIDATES;

// ---------------- validator ----------------
export type Check = { keyword: string; ok: boolean; text: string; coerced?: { from: Json; to: Json } };
export type Visit = { path: string; checks: Check[]; status: "ok" | "error" | "absent" | "skipped" };
export type SchemaError = { instancePath: string; keyword: string; params: Record<string, Json>; message: string; got?: Json };
export type ValidateOptions = { coerce: boolean; allErrors: boolean };
export type ValidationResult = { visits: Visit[]; errors: SchemaError[]; data: Args; valid: boolean; coercions: number };

const show = (v: Json | undefined) => (v === undefined ? "nothing" : JSON.stringify(v));
const kindOf = (v: Json) => (v === null ? "null" : Array.isArray(v) ? "array" : typeof v === "number" && Number.isInteger(v) ? "integer" : typeof v);
function typeOk(t: Schema["type"], v: Json): boolean {
  if (t === "integer") return typeof v === "number" && Number.isInteger(v);
  if (t === "number") return typeof v === "number" && Number.isFinite(v);
  if (t === "object") return typeof v === "object" && v !== null && !Array.isArray(v);
  return typeof v === t;
}
/** ajv-style coerceTypes: only lossless scalar conversions. */
function coerce(t: Schema["type"], v: Json): { ok: true; value: Json } | { ok: false } {
  if ((t === "integer" || t === "number") && typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) {
    const n = Number(v);
    return t === "integer" && !Number.isInteger(n) ? { ok: false } : { ok: true, value: n };
  }
  if (t === "string" && (typeof v === "number" || typeof v === "boolean")) return { ok: true, value: String(v) };
  if (t === "boolean" && (v === "true" || v === "false")) return { ok: true, value: v === "true" };
  return { ok: false };
}

class Stop extends Error {}

export function validate(input: Args, opts: ValidateOptions, schema: Schema = SCHEMA): ValidationResult {
  const data = structuredClone(input) as Args;
  const visits: Visit[] = [];
  const errors: SchemaError[] = [];
  let coercions = 0;
  const nodes = schemaNodes(schema);
  const get = (path: string): Json | undefined => path.split("/").slice(1).reduce<Json | undefined>((o, k) => (o && typeof o === "object" && !Array.isArray(o) ? (o as Record<string, Json>)[k] : undefined), data);
  const set = (path: string, v: Json) => {
    const keys = path.split("/").slice(1);
    const parent = keys.slice(0, -1).reduce<Record<string, Json>>((o, k) => o[k] as Record<string, Json>, data);
    parent[keys.at(-1)!] = v;
  };
  const fail = (checks: Check[], e: SchemaError, text: string) => {
    checks.push({ keyword: e.keyword, ok: false, text });
    errors.push(e);
    if (!opts.allErrors) throw new Stop();
  };

  const checkNode = (node: TreeNode, checks: Check[]): Visit["status"] => {
    const s = node.schema;
    let value = node.path === "" ? (data as Json) : get(node.path);
    const parentPath = node.path.split("/").slice(0, -1).join("/");
    const parentVal = node.path === "" ? data : parentPath === "" ? data : get(parentPath);
    if (value === undefined) {
      const parentRequired = nodes.find((n) => n.path === parentPath)?.schema.required ?? [];
      const missingRequired = parentRequired.includes(node.key) && parentVal !== undefined;
      checks.push({ keyword: "present", ok: !missingRequired, text: missingRequired ? "missing (reported by the parent's required check)" : "not provided: optional or parent missing, nothing to check" });
      return "absent";
    }
    // type (with optional coercion)
    if (typeOk(s.type, value)) checks.push({ keyword: "type", ok: true, text: `type is ${s.type}` });
    else {
      const c = opts.coerce ? coerce(s.type, value) : { ok: false as const };
      if (c.ok) {
        checks.push({ keyword: "type", ok: true, text: `type: ${show(value)} coerced to ${s.type} ${show(c.value)}`, coerced: { from: value, to: c.value } });
        set(node.path, c.value);
        value = c.value;
        coercions++;
      } else {
        fail(checks, { instancePath: node.path, keyword: "type", params: { type: s.type }, message: `must be ${s.type}`, got: value }, `type must be ${s.type}, got ${kindOf(value)} ${show(value)}`);
        return "error";
      }
    }
    if (s.type === "object") {
      const obj = value as Record<string, Json>;
      for (const r of s.required ?? []) {
        if (r in obj) checks.push({ keyword: "required", ok: true, text: `required "${r}" present` });
        else fail(checks, { instancePath: node.path, keyword: "required", params: { missingProperty: r }, message: `must have required property '${r}'` }, `required "${r}" is missing`);
      }
      if (s.additionalProperties === false) {
        const extra = Object.keys(obj).filter((key) => !(key in (s.properties ?? {})));
        if (!extra.length) checks.push({ keyword: "additionalProperties", ok: true, text: "no unknown properties" });
        for (const x of extra) fail(checks, { instancePath: `${node.path}/${x}`, keyword: "additionalProperties", params: { additionalProperty: x }, message: "must NOT have additional properties", got: obj[x] }, `unknown property "${x}"`);
      }
    }
    if (typeof value === "string") {
      if (s.minLength !== undefined) {
        if (value.length >= s.minLength) checks.push({ keyword: "minLength", ok: true, text: `length ${value.length} ≥ ${s.minLength}` });
        else fail(checks, { instancePath: node.path, keyword: "minLength", params: { limit: s.minLength }, message: `must NOT have fewer than ${s.minLength} characters`, got: value }, `length ${value.length} < ${s.minLength}`);
      }
      if (s.pattern) {
        if (new RegExp(s.pattern).test(value)) checks.push({ keyword: "pattern", ok: true, text: `matches ${s.hint ?? s.pattern}` });
        else fail(checks, { instancePath: node.path, keyword: "pattern", params: { pattern: s.pattern }, message: `must match pattern "${s.pattern}"`, got: value }, `${show(value)} is not ${s.hint ?? "matching the pattern"}`);
      }
      if (s.enum) {
        if (s.enum.includes(value)) checks.push({ keyword: "enum", ok: true, text: `${show(value)} is allowed` });
        else fail(checks, { instancePath: node.path, keyword: "enum", params: { allowedValues: s.enum }, message: "must be equal to one of the allowed values", got: value }, `${show(value)} not in {${s.enum.join(", ")}}`);
      }
    }
    if (typeof value === "number") {
      if (s.minimum !== undefined) {
        if (value >= s.minimum) checks.push({ keyword: "minimum", ok: true, text: `${value} ≥ ${s.minimum}` });
        else fail(checks, { instancePath: node.path, keyword: "minimum", params: { limit: s.minimum }, message: `must be >= ${s.minimum}`, got: value }, `${value} < ${s.minimum}`);
      }
      if (s.maximum !== undefined) {
        if (value <= s.maximum) checks.push({ keyword: "maximum", ok: true, text: `${value} ≤ ${s.maximum}` });
        else fail(checks, { instancePath: node.path, keyword: "maximum", params: { limit: s.maximum }, message: `must be <= ${s.maximum}`, got: value }, `${value} > ${s.maximum}`);
      }
    }
    return checks.every((c) => c.ok) ? "ok" : "error";
  };

  for (const node of nodes) {
    const checks: Check[] = [];
    try {
      visits.push({ path: node.path, checks, status: checkNode(node, checks) });
    } catch (e) {
      if (!(e instanceof Stop)) throw e;
      // stop at first error: this node keeps its partial checks; nodes after it are never checked
      visits.push({ path: node.path, checks, status: "error" });
      break;
    }
  }
  return { visits, errors, data, valid: errors.length === 0, coercions };
}

// ---------------- model-facing messages ----------------
export const fieldName = (p: string) => (p === "" ? "(top level)" : p.slice(1).replace(/\//g, "."));
const lookup = (path: string, schema: Schema = SCHEMA) => schemaNodes(schema).find((n) => n.path === path)?.schema;

export function friendlyLine(e: SchemaError): string {
  const f = fieldName(e.instancePath);
  switch (e.keyword) {
    case "required": return `- ${e.instancePath === "" ? "" : `${f}.`}${e.params.missingProperty}: required field is missing.`;
    case "type": return `- ${f}: must be ${e.params.type === "integer" ? "an integer" : `a ${e.params.type}`} (you sent ${show(e.got)}, a ${kindOf(e.got ?? null)}).`;
    case "enum": return `- ${f}: must be one of ${(e.params.allowedValues as string[]).map((v) => `"${v}"`).join(", ")} (you sent ${show(e.got)}).`;
    case "maximum": return `- ${f}: must be at most ${e.params.limit} (you sent ${show(e.got)}).`;
    case "minimum": return `- ${f}: must be at least ${e.params.limit} (you sent ${show(e.got)}).`;
    case "additionalProperties": return `- ${f}: is not a field of this tool; remove it.`;
    case "pattern": return `- ${f}: must be ${lookup(e.instancePath)?.hint ?? "in the expected format"} (you sent ${show(e.got)}).`;
    case "minLength": return `- ${f}: must not be empty.`;
    default: return `- ${f}: ${e.message}.`;
  }
}

export function friendlyMessage(errors: SchemaError[]): string {
  return [`${TOOL_NAME} was NOT called: ${errors.length} invalid argument${errors.length === 1 ? "" : "s"}.`, ...errors.map(friendlyLine), `Fix only these fields and call ${TOOL_NAME} again. If a value is unknown, ask the user instead of guessing.`].join("\n");
}

// ---------------- simulated repair ----------------
export type RepairAction = { path: string; kind: "fixed" | "clamped" | "removed" | "ask_user"; text: string };

/** Rule-based stand-in for the model's next attempt. It can fix form, but it cannot invent facts. */
export function repair(args: Args, errors: SchemaError[]): { args: Args; actions: RepairAction[] } {
  const out = structuredClone(args) as Args;
  const actions: RepairAction[] = [];
  const parentOf = (path: string) => path.split("/").slice(1, -1).reduce<Record<string, Json>>((o, k) => o[k] as Record<string, Json>, out);
  const keyOf = (path: string) => path.split("/").at(-1)!;
  for (const e of errors) {
    const f = fieldName(e.instancePath);
    if (e.keyword === "required") {
      actions.push({ path: `${e.instancePath}/${e.params.missingProperty}`, kind: "ask_user", text: `${fieldName(`${e.instancePath}/${e.params.missingProperty}`)} was never given: ask the user` });
    } else if (e.keyword === "additionalProperties") {
      delete parentOf(e.instancePath)[keyOf(e.instancePath)];
      actions.push({ path: e.instancePath, kind: "removed", text: `removed unknown field ${f}` });
    } else if (e.keyword === "type" && typeof e.got === "string" && (e.params.type === "integer" || e.params.type === "number") && Number.isFinite(Number(e.got))) {
      parentOf(e.instancePath)[keyOf(e.instancePath)] = Number(e.got);
      actions.push({ path: e.instancePath, kind: "fixed", text: `${f}: "${e.got}" → ${Number(e.got)}` });
    } else if (e.keyword === "enum" && typeof e.got === "string") {
      const match = (e.params.allowedValues as string[]).find((v) => v.toLowerCase() === (e.got as string).toLowerCase());
      if (match) { parentOf(e.instancePath)[keyOf(e.instancePath)] = match; actions.push({ path: e.instancePath, kind: "fixed", text: `${f}: "${e.got}" → "${match}"` }); }
      else actions.push({ path: e.instancePath, kind: "ask_user", text: `${f}: "${e.got}" has no close match: ask the user which room` });
    } else if ((e.keyword === "maximum" || e.keyword === "minimum") && typeof e.got === "number") {
      parentOf(e.instancePath)[keyOf(e.instancePath)] = e.params.limit;
      actions.push({ path: e.instancePath, kind: "clamped", text: `${f}: ${e.got} → ${e.params.limit} (passes, but changes what the user asked for)` });
    } else if (e.keyword === "pattern" && typeof e.got === "string" && /^\d{2}\/\d{2}\/\d{4}$/.test(e.got)) {
      const [m, d, y] = e.got.split("/");
      parentOf(e.instancePath)[keyOf(e.instancePath)] = `${y}-${m}-${d}`;
      actions.push({ path: e.instancePath, kind: "fixed", text: `${f}: "${e.got}" → "${y}-${m}-${d}"` });
    } else {
      actions.push({ path: e.instancePath, kind: "ask_user", text: `${f}: ${show(e.got)} cannot be repaired from context: ask the user` });
    }
  }
  return { args: out, actions };
}

// ---------------- JSON lines (for the canvas) ----------------
export type JsonLine = { text: string; path: string | null; depth: number };
export function jsonLines(args: Args): JsonLine[] {
  const lines: JsonLine[] = [{ text: "{", path: null, depth: 0 }];
  const rec = (o: Record<string, Json>, base: string, depth: number) => {
    const entries = Object.entries(o);
    entries.forEach(([k, v], i) => {
      const comma = i < entries.length - 1 ? "," : "";
      if (v && typeof v === "object" && !Array.isArray(v)) {
        lines.push({ text: `"${k}": {`, path: `${base}/${k}`, depth });
        rec(v, `${base}/${k}`, depth + 1);
        lines.push({ text: `}${comma}`, path: null, depth });
      } else lines.push({ text: `"${k}": ${JSON.stringify(v)}${comma}`, path: `${base}/${k}`, depth });
    });
  };
  rec(args, "", 1);
  lines.push({ text: "}", path: null, depth: 0 });
  return lines;
}

// ---------------- steps ----------------
export type Phase = "propose" | "node" | "format" | "repair" | "revalidate" | "execute";
export type ArgState = {
  phase: Phase;
  args: Args;
  /** Visits revealed so far (node steps add one at a time). */
  visits: Visit[];
  current: string | null;
  first: ValidationResult;
  friendly: string | null;
  repaired: { args: Args; actions: RepairAction[] } | null;
  second: ValidationResult | null;
  verdict: { tone: "ok" | "warn" | "err"; text: string } | null;
};
export type ArgConfig = { candidate: CandidateId; coerce: boolean; allErrors: boolean };
export const DEFAULT_ARGS: ArgConfig = { candidate: "range", coerce: false, allErrors: true };

export function simulateValidation(cfg: ArgConfig): { steps: SimulationStep<ArgState>[]; first: ValidationResult } {
  const opts = { coerce: cfg.coerce, allErrors: cfg.allErrors };
  const args = CANDIDATES[cfg.candidate].args;
  const first = validate(args, opts);
  let n = 0;
  const ev = (message: string, severity: SimulationEvent["severity"] = "info"): SimulationEvent => ({ id: `e${n++}`, type: "log", message, severity });
  const base: Omit<ArgState, "phase"> = { args, visits: [], current: null, first, friendly: null, repaired: null, second: null, verdict: null };
  const steps: SimulationStep<ArgState>[] = [{
    id: "propose", title: `Model proposes arguments for ${TOOL_NAME}`, description: "Before anything is executed, the runtime checks the arguments against the tool's JSON Schema, one node at a time.",
    state: { ...base, phase: "propose" }, events: [ev(`candidate: ${JSON.stringify(args)}`)],
  }];
  first.visits.forEach((v, i) => {
    const failed = v.checks.filter((c) => !c.ok);
    steps.push({
      id: `node-${i}`,
      title: `Check ${fieldName(v.path) === "(top level)" ? "the top-level object" : fieldName(v.path)}: ${v.status === "ok" ? "pass" : v.status === "absent" ? "not present" : `${failed.length} error${failed.length > 1 ? "s" : ""}`}`,
      description: v.checks.map((c) => `${c.ok ? "✓" : "✗"} ${c.text}`).join(" · "),
      state: { ...base, phase: "node", visits: first.visits.slice(0, i + 1), current: v.path },
      events: v.checks.map((c) => ev(`${fieldName(v.path)}: ${c.text}`, c.ok ? "info" : "error")),
    });
  });
  const allVisits = first.visits;
  if (first.valid) {
    steps.push({
      id: "execute", title: `All checks pass: ${TOOL_NAME} executes`, description: first.coercions ? `${first.coercions} value(s) were coerced to the schema type before the call.` : "The arguments match the schema exactly.",
      state: { ...base, phase: "execute", visits: allVisits, verdict: { tone: "ok", text: `Executed ${TOOL_NAME} with ${JSON.stringify(first.data)}` } },
      events: [ev(`execute ${TOOL_NAME}`)],
    });
    return { steps, first };
  }
  const friendly = friendlyMessage(first.errors);
  steps.push({
    id: "format", title: `${first.errors.length} error${first.errors.length > 1 ? "s" : ""} rewritten for the model`, description: "Raw validator errors are turned into one short, specific instruction per field, returned as the tool result.",
    state: { ...base, phase: "format", visits: allVisits, friendly }, events: first.errors.map((e) => ev(`${e.instancePath || "/"} ${e.keyword}: ${e.message}`, "error")),
  });
  const repaired = repair(args, first.errors);
  steps.push({
    id: "repair", title: "Simulated model repairs its call", description: repaired.actions.map((a) => a.text).join("; "),
    state: { ...base, phase: "repair", visits: allVisits, friendly, repaired },
    events: repaired.actions.map((a) => ev(a.text, a.kind === "fixed" || a.kind === "removed" ? "info" : "warning")),
  });
  const second = validate(repaired.args, opts);
  const asks = repaired.actions.filter((a) => a.kind === "ask_user");
  const clamped = repaired.actions.filter((a) => a.kind === "clamped");
  const verdict: ArgState["verdict"] = !second.valid
    ? { tone: "err", text: asks.length ? `Still invalid. Ask the user for: ${asks.map((a) => fieldName(a.path)).join(", ")}.` : `Still invalid: ${second.errors.length} more error(s) surfaced. Another round trip is needed.` }
    : clamped.length ? { tone: "warn", text: `Valid, but the repair changed the user's request (${clamped.map((a) => a.text.split(" (")[0]).join("; ")}). Confirm with the user before booking.` }
    : { tone: "ok", text: `Valid after one repair: ${TOOL_NAME} executes.` };
  steps.push({
    id: "revalidate", title: second.valid ? (clamped.length ? "Re-validation passes, but the meaning changed" : "Re-validation passes") : "Re-validation still fails",
    description: verdict.text,
    state: { ...base, phase: "revalidate", visits: second.visits, args: repaired.args, friendly, repaired, second, verdict },
    events: [ev(verdict.text, verdict.tone === "ok" ? "info" : verdict.tone === "warn" ? "warning" : "error")],
  });
  return { steps, first };
}
