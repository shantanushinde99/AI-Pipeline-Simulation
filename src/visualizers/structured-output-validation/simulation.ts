import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep } from "../../types/simulation";

/**
 * Parse → validate → repair → retry on a toy "ticket" schema. Parsing is the real JSON.parse; the validator and
 * the repairs are real (small) code. Only the model is simulated: its outputs are fixed variants, and retries
 * draw a new variant from a seeded random generator.
 */
export const SCHEMA = {
  required: ["title", "priority", "estimate_hours", "tags"],
  properties: {
    title: { type: "string" },
    priority: { type: "string", enum: ["low", "medium", "high"] },
    estimate_hours: { type: "number", minimum: 0, maximum: 100 },
    tags: { type: "array", items: "string" },
  },
} as const;
type Prop = keyof typeof SCHEMA.properties;
const PROPS = Object.keys(SCHEMA.properties) as Prop[];

export const KINDS = ["valid", "trailing-comma", "wrong-type", "missing-field", "extra-field", "truncated", "enum"] as const;
export type Kind = (typeof KINDS)[number];
export const ERROR_KINDS = KINDS.filter((k) => k !== "valid");
export const SYNTAX_KINDS: Kind[] = ["trailing-comma", "truncated"];
export const KIND_LABEL: Record<Kind, string> = {
  valid: "Valid", "trailing-comma": "Trailing comma", "wrong-type": "Wrong type (\"3\")", "missing-field": "Missing field",
  "extra-field": "Extra field", truncated: "Truncated JSON", enum: "Value out of enum",
};

const L = (...lines: string[]) => lines.join("\n");
export const RAW: Record<Kind, string> = {
  valid: L("{", '  "title": "Fix login timeout",', '  "priority": "high",', '  "estimate_hours": 3,', '  "tags": ["auth", "bug"]', "}"),
  "trailing-comma": L("{", '  "title": "Fix login timeout",', '  "priority": "high",', '  "estimate_hours": 3,', '  "tags": ["auth", "bug"],', "}"),
  "wrong-type": L("{", '  "title": "Fix login timeout",', '  "priority": "high",', '  "estimate_hours": "3",', '  "tags": ["auth", "bug"]', "}"),
  "missing-field": L("{", '  "title": "Fix login timeout",', '  "estimate_hours": 3,', '  "tags": ["auth", "bug"]', "}"),
  "extra-field": L("{", '  "title": "Fix login timeout",', '  "priority": "high",', '  "estimate_hours": 3,', '  "tags": ["auth", "bug"],', '  "confidence": 0.9', "}"),
  truncated: L("{", '  "title": "Fix login timeout",', '  "priority": "high",', '  "estimate_ho'),
  enum: L("{", '  "title": "Fix login timeout",', '  "priority": "urgent",', '  "estimate_hours": 3,', '  "tags": ["auth", "bug"]', "}"),
};

export type ParseResult = { ok: true; value: unknown } | { ok: false; pos: number; line: number; message: string };

/** Real JSON.parse; the error position is located ourselves because engines word their messages differently. */
export function parse(raw: string): ParseResult {
  try {
    return { ok: true, value: JSON.parse(raw) };
  } catch {
    const trailing = /,(\s*[}\]])/.exec(raw);
    const pos = trailing ? trailing.index : raw.length;
    return { ok: false, pos, line: raw.slice(0, pos).split("\n").length - 1, message: trailing ? "Unexpected '}' after a trailing comma" : "Unexpected end of input (unterminated string / object)" };
  }
}

export type Visit = { path: string; label: string; ok: boolean; message: string; depth: number };
const typeOf = (v: unknown) => (Array.isArray(v) ? "array" : v === null ? "null" : typeof v);

/** Walks the parsed value in schema order, recording a visit for every node it checks. */
export function validate(value: unknown): { visits: Visit[]; issues: Visit[] } {
  const visits: Visit[] = [];
  if (typeOf(value) !== "object") {
    visits.push({ path: "$", label: "root", ok: false, message: `expected object, got ${typeOf(value)}`, depth: 0 });
    return { visits, issues: visits };
  }
  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj);
  visits.push({ path: "$", label: "{ … } root", ok: true, message: `object with ${keys.length} keys`, depth: 0 });
  for (const p of PROPS) {
    const spec = SCHEMA.properties[p];
    if (!(p in obj)) { visits.push({ path: `$.${p}`, label: p, ok: false, message: "required field is missing", depth: 1 }); continue; }
    const v = obj[p];
    const t = typeOf(v);
    if (t !== spec.type) { visits.push({ path: `$.${p}`, label: p, ok: false, message: `expected ${spec.type}, got ${t} (${JSON.stringify(v)})`, depth: 1 }); continue; }
    if ("enum" in spec && !(spec.enum as readonly string[]).includes(v as string)) { visits.push({ path: `$.${p}`, label: p, ok: false, message: `"${v}" is not one of ${spec.enum.join(" | ")}`, depth: 1 }); continue; }
    if ("minimum" in spec && ((v as number) < spec.minimum || (v as number) > spec.maximum)) { visits.push({ path: `$.${p}`, label: p, ok: false, message: `${v} outside ${spec.minimum}–${spec.maximum}`, depth: 1 }); continue; }
    visits.push({ path: `$.${p}`, label: p, ok: true, message: `${spec.type}${"enum" in spec ? " in enum" : ""} ✓ ${JSON.stringify(v)}`, depth: 1 });
    if (spec.type === "array") (v as unknown[]).forEach((item, i) => visits.push({ path: `$.${p}[${i}]`, label: `[${i}]`, ok: typeof item === "string", message: typeof item === "string" ? `string ✓ ${JSON.stringify(item)}` : `expected string, got ${typeOf(item)}`, depth: 2 }));
  }
  for (const k of keys) if (!(PROPS as string[]).includes(k)) visits.push({ path: `$.${k}`, label: k, ok: false, message: "additional property not allowed", depth: 1 });
  return { visits, issues: visits.filter((x) => !x.ok) };
}

/** Syntax repair: drop trailing commas; if still truncated, cut back to the last complete top-level member and close. */
export function repairSyntax(raw: string): string {
  const s = raw.replace(/,(\s*[}\]])/g, "$1");
  if (parse(s).ok) return s;
  let inStr = false, esc = false, depth = 0, lastComma = -1;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (esc) { esc = false; continue; }
    if (ch === "\\") { esc = true; continue; }
    if (ch === '"') inStr = !inStr;
    else if (!inStr && (ch === "{" || ch === "[")) depth++;
    else if (!inStr && (ch === "}" || ch === "]")) depth--;
    else if (!inStr && ch === "," && depth === 1) lastComma = i;
  }
  return lastComma > 0 ? `${s.slice(0, lastComma)}\n}` : s;
}

/** Schema repair: coerce numeric strings, drop unknown keys. Missing fields and bad enum values cannot be invented. */
export function repairSchema(value: unknown): { value: unknown; changes: string[] } {
  if (typeOf(value) !== "object") return { value, changes: [] };
  const out: Record<string, unknown> = {};
  const changes: string[] = [];
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    if (!(PROPS as string[]).includes(k)) { changes.push(`dropped unknown key "${k}"`); continue; }
    const spec = SCHEMA.properties[k as Prop];
    if (spec.type === "number" && typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) { out[k] = Number(v); changes.push(`coerced ${k} "${v}" → ${Number(v)}`); continue; }
    out[k] = v;
  }
  return { value: out, changes };
}

export type Attempt = {
  kind: Kind; raw: string; parsed: ParseResult; syntaxRepaired?: string; reparsed?: ParseResult;
  visits: Visit[]; issues: Visit[]; schemaChanges: string[]; finalVisits: Visit[]; finalIssues: Visit[]; accepted: boolean;
};

export function runAttempt(kind: Kind, repair: boolean): Attempt {
  const raw = RAW[kind];
  const parsed = parse(raw);
  let value: unknown, syntaxRepaired: string | undefined, reparsed: ParseResult | undefined;
  if (parsed.ok) value = parsed.value;
  else if (repair) {
    syntaxRepaired = repairSyntax(raw);
    reparsed = parse(syntaxRepaired);
    if (reparsed.ok) value = reparsed.value;
  }
  const have = parsed.ok || reparsed?.ok;
  const { visits, issues } = have ? validate(value) : { visits: [], issues: [] };
  let schemaChanges: string[] = [], finalIssues = issues, finalVisits = visits;
  if (have && issues.length && repair) {
    const r = repairSchema(value);
    schemaChanges = r.changes;
    ({ visits: finalVisits, issues: finalIssues } = validate(r.value));
  }
  return { kind, raw, parsed, syntaxRepaired, reparsed, visits, issues, schemaChanges, finalVisits, finalIssues, accepted: Boolean(have) && finalIssues.length === 0 };
}

export type Config = { kind: Kind; repair: boolean; maxRetries: number; constrained: boolean; retryFix: number; errorRate: number; n: number; seed: number };
export const DEFAULT_CONFIG: Config = { kind: "enum", repair: true, maxRetries: 1, constrained: false, retryFix: 0.7, errorRate: 0.35, n: 200, seed: 11 };

/** Constrained decoding (simulated) restricts generation to the JSON grammar: syntax errors cannot be produced. */
const decode = (k: Kind, constrained: boolean): Kind => (constrained && SYNTAX_KINDS.includes(k) ? "valid" : k);
const pick = (r: () => number) => ERROR_KINDS[Math.floor(r() * ERROR_KINDS.length) % ERROR_KINDS.length];

/** The model's answer to "your JSON failed: <errors>, try again". */
function retryKinds(seed: number, n: number, retryFix: number) {
  const g = rng(seed);
  return Array.from({ length: n }, () => (g.next() < retryFix ? "valid" : pick(g.next)) as Kind);
}

export const LADDER = [
  { id: "parse", label: "Parse only" },
  { id: "validate", label: "+ schema validation" },
  { id: "repair", label: "+ repair" },
  { id: "retry", label: "+ retries" },
  { id: "constrained", label: "+ constrained decoding" },
] as const;
export type LadderId = (typeof LADDER)[number]["id"];
export type BatchRow = { id: LadderId; label: string; validAccepted: number; badAccepted: number; rejected: number; calls: number };

/** N simulated outputs pushed through each cumulative layer; the same random draws are reused for every layer. */
export function batch(cfg: Config): { rows: BatchRow[]; kinds: Record<Kind, number> } {
  const n = Math.max(1, Math.min(1000, Math.round(cfg.n)));
  const retries = Math.max(0, Math.min(3, Math.round(cfg.maxRetries)));
  const g = rng(cfg.seed);
  const draws = Array.from({ length: n }, () => {
    const first = (g.next() < cfg.errorRate ? pick(g.next) : "valid") as Kind;
    return [first, ...retryKinds(Math.floor(g.next() * 1e9), retries, cfg.retryFix)];
  });
  const cache = new Map<string, boolean>();
  const pass = (k: Kind, repair: boolean) => {
    const key = `${k}-${repair}`;
    if (!cache.has(key)) cache.set(key, runAttempt(k, repair).accepted);
    return cache.get(key)!;
  };
  const kinds = Object.fromEntries(KINDS.map((k) => [k, draws.filter((d) => d[0] === k).length])) as Record<Kind, number>;
  const rows = LADDER.map(({ id, label }) => {
    let validAccepted = 0, badAccepted = 0, rejected = 0, calls = 0;
    for (const d of draws) {
      if (id === "parse") {
        calls++;
        if (!parse(RAW[d[0]]).ok) rejected++;
        else if (d[0] === "valid") validAccepted++;
        else badAccepted++;
        continue;
      }
      const repair = id !== "validate";
      const tries = id === "retry" || id === "constrained" ? d : [d[0]];
      let ok = false;
      for (const k of tries) {
        calls++;
        if (pass(decode(k, id === "constrained"), repair)) { ok = true; break; }
      }
      if (ok) validAccepted++;
      else rejected++;
    }
    return { id, label, validAccepted, badAccepted, rejected, calls };
  });
  return { rows, kinds };
}

export type SOState = {
  phase: "generate" | "parse" | "repair-syntax" | "validate" | "repair-schema" | "retry" | "final" | "batch";
  attempt: number; // 0 = first output, 1.. = retries
};

export function simulate(cfg: Config) {
  const maxRetries = Math.max(0, Math.min(3, Math.round(cfg.maxRetries)));
  const first = decode(cfg.kind, cfg.constrained);
  const attempts: Attempt[] = [runAttempt(first, cfg.repair)];
  const retryDraws = retryKinds(cfg.seed * 7 + KINDS.indexOf(cfg.kind), maxRetries, cfg.retryFix);
  for (let i = 0; i < maxRetries && !attempts.at(-1)!.accepted; i++) attempts.push(runAttempt(decode(retryDraws[i], cfg.constrained), cfg.repair));
  const a0 = attempts[0];
  const accepted = attempts.at(-1)!.accepted;
  const ev = (id: string, message: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "validation", message, severity });
  const S = (phase: SOState["phase"], attempt = 0): SOState => ({ phase, attempt });
  const steps: SimulationStep<SOState>[] = [
    { id: "generate", title: "Model emits JSON", description: cfg.constrained && SYNTAX_KINDS.includes(cfg.kind) ? `Constrained decoding: the model was going to produce "${KIND_LABEL[cfg.kind]}", but the JSON grammar forbids it, so the output is syntactically valid.` : `The model returns text that is supposed to be JSON (${KIND_LABEL[first]}).`,
      state: S("generate"), events: [ev("gen", `output #1: ${a0.raw.length} chars${cfg.constrained ? " (grammar-constrained)" : ""}`)] },
    { id: "parse", title: "Parse", description: a0.parsed.ok ? "JSON.parse succeeds: the text is well-formed JSON." : `JSON.parse fails at line ${a0.parsed.line + 1}: ${a0.parsed.message}.`,
      state: S("parse"), events: [a0.parsed.ok ? ev("p", "JSON.parse OK") : ev("p", `JSON.parse failed at char ${a0.parsed.pos}: ${a0.parsed.message}`, "error")] },
  ];
  if (!a0.parsed.ok && cfg.repair)
    steps.push({ id: "repair-syntax", title: "Repair syntax", description: a0.reparsed?.ok ? "Removing the trailing comma / closing the truncated object makes it parse." : "Repair could not make it parse.",
      state: S("repair-syntax"), events: [ev("rs", a0.reparsed?.ok ? "syntax repaired, re-parse OK" : "syntax repair failed", a0.reparsed?.ok ? undefined : "error")] });
  if (a0.visits.length)
    steps.push({ id: "validate", title: "Validate against schema", description: a0.issues.length ? `The validator walks the tree and finds ${a0.issues.length} problem(s): ${a0.issues.map((i) => `${i.path} ${i.message}`).join("; ")}.` : "Every node matches the schema.",
      state: S("validate"), events: a0.issues.length ? a0.issues.map((i, j) => ev(`v${j}`, `${i.path}: ${i.message}`, "error")) : [ev("v", "schema OK")] });
  if (a0.issues.length && cfg.repair)
    steps.push({ id: "repair-schema", title: "Repair schema issues", description: a0.schemaChanges.length ? `${a0.schemaChanges.join("; ")}. ${a0.finalIssues.length ? "Still invalid: " + a0.finalIssues.map((i) => i.path).join(", ") : "Now valid."}` : "Nothing safe to repair: missing values and unknown enum values can't be invented.",
      state: S("repair-schema"), events: [...a0.schemaChanges.map((c, j) => ev(`c${j}`, c)), ev("rv", a0.finalIssues.length ? `still invalid (${a0.finalIssues.length})` : "re-validate OK", a0.finalIssues.length ? "warning" : undefined)] });
  attempts.slice(1).forEach((a, i) =>
    steps.push({ id: `retry-${i + 1}`, title: `Retry ${i + 1}`, description: `The errors are sent back to the model, which answers again (${KIND_LABEL[a.kind]}). ${a.accepted ? "This one passes." : "Still failing."}`,
      state: S("retry", i + 1), events: [ev(`r${i}`, `retry ${i + 1}: ${KIND_LABEL[a.kind]} → ${a.accepted ? "accepted" : "rejected"}`, a.accepted ? undefined : "warning")] }));
  steps.push({ id: "final", title: accepted ? "Accepted" : "Rejected", description: accepted ? `A schema-valid object reaches the application after ${attempts.length} model call(s).` : `After ${attempts.length} call(s) the output is still invalid, so the request fails loudly instead of passing bad data on.`,
    state: S("final", attempts.length - 1), events: [ev("f", accepted ? "accepted" : "rejected: raise / fall back", accepted ? undefined : "error")] });
  const b = batch(cfg);
  steps.push({ id: "batch", title: "Across many outputs", description: `${cfg.n} simulated outputs at a ${(cfg.errorRate * 100).toFixed(0)}% error rate, pushed through each layer in turn.`, state: S("batch"),
    events: b.rows.map((r) => ev(r.id, `${r.label}: ${r.validAccepted} valid accepted, ${r.badAccepted} bad accepted, ${r.rejected} rejected, ${r.calls} calls`, r.badAccepted ? "error" : undefined)) });
  return { steps, attempts, accepted, batch: b };
}

/** Probability that one retry passes: a valid answer, or an error kind that repair can fix. */
export function retryPassProb(cfg: Pick<Config, "retryFix" | "repair" | "constrained">) {
  const fixable = ERROR_KINDS.filter((k) => runAttempt(decode(k, cfg.constrained), cfg.repair).accepted).length / ERROR_KINDS.length;
  return { q: cfg.retryFix + (1 - cfg.retryFix) * fixable, fixable };
}
