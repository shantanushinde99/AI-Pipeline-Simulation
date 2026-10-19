import type { SimulationEvent, SimulationStep } from "../../types/simulation";

// ---------------- catalogue ----------------
export type ArgKind = "query" | "text" | "city" | "email" | "amount" | "currency_from" | "currency_to" | "date" | "expression" | "ticket";
export type ToolDef = { name: string; description: string; required: { name: string; kind: ArgKind }[] };

const q = (name = "query"): { name: string; kind: ArgKind } => ({ name, kind: "query" });

const WEB_CLEAR: ToolDef = { name: "web_search", description: "Search the public internet for news, facts and current events. Not for: internal company policy, vacation, benefits, expenses.", required: [q()] };
const HANDBOOK_CLEAR: ToolDef = { name: "search_handbook", description: "Search the internal employee handbook for HR policy: vacation days, PTO, benefits, expenses and holidays per year.", required: [q()] };
const REST: ToolDef[] = [
  { name: "get_weather", description: "Get the current weather or forecast for a city, today or tomorrow.", required: [{ name: "city", kind: "city" }] },
  { name: "send_email", description: "Send an email with a message or report to a recipient address.", required: [{ name: "to", kind: "email" }, { name: "body", kind: "text" }] },
  { name: "convert_currency", description: "Convert an amount of money between currencies such as USD, EUR and JPY.", required: [{ name: "amount", kind: "amount" }, { name: "from", kind: "currency_from" }, { name: "to", kind: "currency_to" }] },
  { name: "calculator", description: "Evaluate an arithmetic expression with numbers.", required: [{ name: "expression", kind: "expression" }] },
];
const DISTRACTORS: ToolDef[] = [
  { name: "get_calendar", description: "List calendar events, holidays and days off for any year or month.", required: [] },
  { name: "search_wiki", description: "Search the engineering wiki for design docs, runbooks and architecture notes.", required: [q()] },
  { name: "search_tickets", description: "Search support tickets by keyword, customer or status.", required: [q()] },
  { name: "get_ticket", description: "Fetch one support ticket by its number, for example #4821.", required: [{ name: "ticket", kind: "ticket" }] },
  { name: "create_ticket", description: "Open a new support ticket with a title and description.", required: [{ name: "title", kind: "text" }] },
  { name: "send_slack", description: "Post a message to a Slack channel or user.", required: [{ name: "text", kind: "text" }] },
  { name: "translate_text", description: "Translate text between languages.", required: [{ name: "text", kind: "text" }] },
  { name: "summarize_document", description: "Summarize a long document or report into bullet points.", required: [{ name: "text", kind: "text" }] },
  { name: "get_user_profile", description: "Look up an employee profile, manager and team by name or email.", required: [{ name: "email", kind: "email" }] },
  { name: "book_meeting_room", description: "Reserve a meeting room for a date and time.", required: [{ name: "date", kind: "date" }] },
  { name: "list_holidays", description: "List public holidays for a country and year.", required: [] },
  { name: "get_exchange_rate", description: "Get today's exchange rate between two currencies, for example USD to EUR.", required: [{ name: "from", kind: "currency_from" }, { name: "to", kind: "currency_to" }] },
  { name: "run_sql", description: "Run a read-only SQL query against the analytics warehouse.", required: [q()] },
  { name: "get_order_status", description: "Check the shipping status of an order by order number.", required: [{ name: "order", kind: "ticket" }] },
  { name: "create_invoice", description: "Create an invoice for a customer with line items and amounts.", required: [{ name: "amount", kind: "amount" }] },
  { name: "reset_password", description: "Send a password reset link to a user's email address.", required: [{ name: "email", kind: "email" }] },
  { name: "weather_alerts", description: "Get severe weather alerts and warnings for a region.", required: [{ name: "city", kind: "city" }] },
  { name: "get_stock_price", description: "Get the latest stock price for a company ticker symbol.", required: [q("symbol")] },
];

export type CatalogueId = "clear" | "overlapping" | "many";
const VAGUE = "Search for information and answer questions.";
export const CATALOGUES: Record<CatalogueId, { label: string; tools: ToolDef[] }> = {
  clear: { label: "Well-described (6 tools)", tools: [WEB_CLEAR, HANDBOOK_CLEAR, ...REST] },
  overlapping: { label: "Overlapping descriptions (6 tools)", tools: [{ ...WEB_CLEAR, description: VAGUE }, { ...HANDBOOK_CLEAR, description: VAGUE }, ...REST] },
  // A real registry with many tools, listed alphabetically.
  many: { label: "Too many tools (24 tools)", tools: [WEB_CLEAR, HANDBOOK_CLEAR, ...REST, ...DISTRACTORS].sort((a, b) => a.name.localeCompare(b.name)) },
};

export const REQUESTS = [
  { id: "vacation", text: "How many vacation days do I get per year?", expected: "search_handbook" },
  { id: "weather", text: "What's the weather in Paris tomorrow?", expected: "get_weather" },
  { id: "email", text: "Email the Q3 report to dana@acme.com", expected: "send_email" },
  { id: "currency", text: "Convert 250 USD to EUR", expected: "convert_currency" },
] as const;

// ---------------- features ----------------
export const WEIGHTS = { overlap: 1, args: 0.3, penalty: 1 };
export const DEFAULT_BUDGET = 160;
const STOP = new Set("a an the is are am be do does did i me my you your we our it its in on at to for of and or with from by per how many much what whats s this that will can could should would please right now get".split(" "));

/** Lowercase, split on non-alphanumerics, drop stopwords, strip a plural "s" (days → day; Paris stays). */
export function tokenize(text: string): string[] {
  return text.toLowerCase().split(/[^a-z0-9#]+/).filter((w) => w && !STOP.has(w)).map(stem);
}
export const stem = (w: string) => (w.length > 3 && w.endsWith("s") && !/(ss|is|us)$/.test(w) ? w.slice(0, -1) : w);
export const isStopword = (w: string) => STOP.has(w.toLowerCase());

const CITIES = ["Paris", "Tokyo", "London", "Berlin"];
/** Deterministic regex extractors: can this argument be filled from the request text? */
export function extract(kind: ArgKind, text: string): string | null {
  const codes = text.match(/\b[A-Z]{3}\b/g) ?? [];
  switch (kind) {
    case "query":
    case "text": return text.trim() ? text.trim() : null;
    case "city": return CITIES.find((c) => new RegExp(`\\b${c}\\b`, "i").test(text)) ?? null;
    case "email": return text.match(/[\w.+-]+@[\w-]+\.[\w.]+/)?.[0] ?? null;
    case "amount": return text.match(/\b\d+(?:\.\d+)?\b/)?.[0] ?? null;
    case "currency_from": return codes[0] ?? null;
    case "currency_to": return codes[1] ?? null;
    case "date": return text.match(/\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|\d{4}-\d{2}-\d{2})\b/i)?.[0] ?? null;
    case "expression": return text.match(/\d+(?:\.\d+)?\s*[-+*/×x]\s*\d+(?:\.\d+)?/)?.[0] ?? null;
    case "ticket": return text.match(/#\d+/)?.[0] ?? null;
  }
}

export type ToolScore = {
  name: string;
  index: number;
  /** Description words in the order the model sees them; `kept` = survived the context budget. */
  words: { w: string; kept: boolean; matched: boolean; excluded: boolean }[];
  matched: string[];
  overlap: number;
  filled: { name: string; value: string | null }[];
  argFit: number;
  exclusionHits: string[];
  penalty: number;
  total: number;
};

export type SelectionResult = {
  request: string;
  queryTokens: string[];
  scores: ToolScore[];
  ranked: string[];
  winner: string;
  runnerUp: string | null;
  margin: number;
  totalWords: number;
  budget: number;
  wordsPerTool: number | null;
  tie: boolean;
  call: { name: string; arguments: Record<string, string> };
};

const r3 = (v: number) => Math.round(v * 1000) / 1000;

export function scoreTools(request: string, tools: ToolDef[], budgetIn = DEFAULT_BUDGET): SelectionResult {
  const budget = Math.max(20, Math.round(budgetIn));
  const Q = [...new Set(tokenize(request))];
  const totalWords = tools.reduce((s, t) => s + t.description.split(/\s+/).length, 0);
  // Over budget: every description is cut to an equal share of words (the model never sees the rest).
  const wordsPerTool = totalWords > budget ? Math.max(1, Math.floor(budget / Math.max(1, tools.length))) : null;
  const scores = tools.map((t, index): ToolScore => {
    const raw = t.description.split(/\s+/);
    const notFor = raw.findIndex((w, i) => w === "Not" && raw[i + 1] === "for:");
    const words = raw.map((w, i) => {
      const kept = wordsPerTool === null || i < wordsPerTool;
      const tok = tokenize(w)[0];
      const inExclusion = notFor >= 0 && i >= notFor;
      return { w, kept, matched: kept && !inExclusion && !!tok && Q.includes(tok), excluded: kept && inExclusion && !!tok && Q.includes(tok) };
    });
    const visibleMain = words.filter((x, i) => x.kept && (notFor < 0 || i < notFor)).map((x) => x.w).join(" ");
    const D = new Set([...tokenize(t.name.replace(/_/g, " ")), ...tokenize(visibleMain)]);
    const matched = Q.filter((tok) => D.has(tok));
    const visibleExcl = new Set(tokenize(words.filter((x, i) => x.kept && notFor >= 0 && i > notFor + 1).map((x) => x.w).join(" ")));
    const exclusionHits = Q.filter((tok) => visibleExcl.has(tok));
    const filled = t.required.map((p) => ({ name: p.name, value: extract(p.kind, request) }));
    const overlap = Q.length ? matched.length / Q.length : 0;
    const argFit = filled.length ? filled.filter((f) => f.value !== null).length / filled.length : 1;
    const penalty = exclusionHits.length ? 1 : 0;
    const total = WEIGHTS.overlap * overlap + WEIGHTS.args * argFit - WEIGHTS.penalty * penalty;
    return { name: t.name, index, words, matched, overlap: r3(overlap), filled, argFit: r3(argFit), exclusionHits, penalty, total: r3(total) };
  });
  // Highest total wins; ties go to the tool listed first (position bias, made explicit).
  const sorted = [...scores].sort((a, b) => b.total - a.total || a.index - b.index);
  const [w, second] = sorted;
  const winner = w.name;
  return {
    request, queryTokens: Q, scores, ranked: sorted.map((s) => s.name), winner, runnerUp: second?.name ?? null,
    margin: second ? r3(w.total - second.total) : w.total, totalWords, budget, wordsPerTool, tie: !!second && second.total === w.total,
    call: { name: winner, arguments: Object.fromEntries(w.filled.map((f) => [f.name, f.value ?? "<missing>"])) },
  };
}

export function accuracyOnSamples(catalogue: CatalogueId, budget: number) {
  const rows = REQUESTS.map((r) => ({ id: r.id, expected: r.expected, picked: scoreTools(r.text, CATALOGUES[catalogue].tools, budget).winner }));
  return { rows, correct: rows.filter((r) => r.picked === r.expected).length, total: rows.length };
}

// ---------------- steps ----------------
export type Phase = "request" | "catalogue" | "overlap" | "args" | "penalty" | "rank" | "pick";
export const PHASES: Phase[] = ["request", "catalogue", "overlap", "args", "penalty", "rank", "pick"];
export type SelectionState = { phase: Phase; result: SelectionResult; expected: string | null };
export type SelectionConfig = { request: string; catalogue: CatalogueId; budget: number };

export function simulateSelection(cfg: SelectionConfig): { steps: SimulationStep<SelectionState>[]; result: SelectionResult; expected: string | null } {
  const tools = CATALOGUES[cfg.catalogue].tools;
  const result = scoreTools(cfg.request, tools, cfg.budget);
  const expected = REQUESTS.find((r) => r.text === cfg.request)?.expected ?? null;
  let n = 0;
  const ev = (message: string, severity: SimulationEvent["severity"] = "info", tex?: string): SimulationEvent => ({ id: `e${n++}`, type: "log", message, severity, tex });
  const byName = (name: string) => result.scores.find((s) => s.name === name)!;
  const w = byName(result.winner);
  const top3 = result.ranked.slice(0, 3).map(byName);
  const def: Record<Phase, { title: string; description: string; events: SimulationEvent[] }> = {
    request: {
      title: "Request is broken into content words",
      description: `Stopwords are dropped and plurals stemmed, leaving ${result.queryTokens.length} words to match: ${result.queryTokens.join(", ") || "(none)"}.`,
      events: [ev(`query tokens: [${result.queryTokens.join(", ")}]`)],
    },
    catalogue: {
      title: result.wordsPerTool ? "Catalogue overflows the budget: descriptions are cut" : "Every tool description fits in the prompt",
      description: result.wordsPerTool
        ? `${tools.length} tools need ${result.totalWords} description words but the budget is ${result.budget}, so each keeps only its first ${result.wordsPerTool} words.`
        : `${tools.length} tools use ${result.totalWords} of ${result.budget} description words; nothing is cut.`,
      events: [ev(`${result.totalWords} words / budget ${result.budget}${result.wordsPerTool ? ` → truncated to ${result.wordsPerTool} words per tool` : ""}`, result.wordsPerTool ? "warning" : "info")],
    },
    overlap: {
      title: "Feature 1: keyword overlap with each description",
      description: `Share of the request's words found in each tool's name and visible description. Best: ${[...result.scores].sort((a, b) => b.overlap - a.overlap)[0].name} (${[...result.scores].sort((a, b) => b.overlap - a.overlap)[0].overlap.toFixed(2)}).`,
      events: result.scores.filter((s) => s.matched.length).map((s) => ev(`${s.name}: matched [${s.matched.join(", ")}] → ${s.matched.length}/${result.queryTokens.length}`, "info", `\\text{overlap} = ${s.matched.length}/${result.queryTokens.length} = ${s.overlap.toFixed(2)}`)),
    },
    args: {
      title: "Feature 2: can the required arguments be filled?",
      description: "A tool whose required arguments cannot be extracted from the request is a worse fit.",
      events: top3.map((s) => ev(`${s.name}: ${s.filled.length ? s.filled.map((f) => `${f.name}=${f.value ?? "missing"}`).join(", ") : "no required args"} → ${s.argFit.toFixed(2)}`)),
    },
    penalty: {
      title: "Feature 3: 'Not for' exclusions subtract",
      description: result.scores.some((s) => s.penalty) ? `${result.scores.filter((s) => s.penalty).map((s) => `${s.name} excludes "${s.exclusionHits.join(", ")}"`).join("; ")}.` : "No visible description excludes these words, so nothing is subtracted.",
      events: result.scores.filter((s) => s.penalty).map((s) => ev(`${s.name}: exclusion hit [${s.exclusionHits.join(", ")}] → −1`, "warning")),
    },
    rank: {
      title: "Scores are summed and the tools re-sort",
      description: `${top3.map((s) => `${s.name} ${s.total.toFixed(2)}`).join(" · ")}. Margin between first and second: ${result.margin.toFixed(2)}${result.tie ? " (a tie, broken by list order)" : ""}.`,
      events: [ev(`ranking: ${result.ranked.slice(0, 5).join(" > ")}`, result.tie ? "warning" : "info"), ...(result.tie ? [ev("tie: the tool listed first wins", "warning")] : [])],
    },
    pick: {
      title: expected ? (result.winner === expected ? `Picked ${result.winner}: correct` : `Picked ${result.winner}: wrong tool (expected ${expected})`) : `Picked ${result.winner}`,
      description: `The highest score wins and its arguments are filled from the request: ${result.winner}(${Object.entries(result.call.arguments).map(([k, v]) => `${k}="${v}"`).join(", ")}).`,
      events: [ev(`select ${result.winner} (score ${w.total.toFixed(2)})`, expected && result.winner !== expected ? "error" : "info")],
    },
  };
  const steps = PHASES.map((phase) => ({ id: phase, ...def[phase], state: { phase, result, expected } }));
  return { steps, result, expected };
}
