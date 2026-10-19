import { Bot, Brain, Calculator, ClipboardList, CreditCard, Database, FileSearch, FileText, Globe, History, Search, Send, User, Users, Utensils, Wrench } from "lucide-react";
import type { TraceSpan } from "../types/simulation";
import { DEFAULT_AGENT, EXPECTED_TOTAL, TASKS, runAgent, type AgentConfig } from "../visualizers/agents/simulation";
import { CATALOGUES, REQUESTS, scoreTools, type ToolDef } from "../visualizers/tool-selection/simulation";
import { DEFAULT_RETRY, runRetries, type RetryConfig, type Segment } from "../visualizers/retries-and-timeouts/simulation";
import { DEFAULT_COORD, simulate as simulateCoord, type AgentId, type CoordConfig } from "../visualizers/multi-agent-coordination/simulation";
import { DEFAULT_MEMORY, QUESTION, SLOTS, expected as expectedSlots, simulate as simulateMemory, type MemoryConfig, type SlotAnswer } from "../visualizers/short-term-and-long-term-memory/simulation";
import type { Evidence, HealthMetric, Scenario } from "./types";

const ev = (id: string, action: string, lines: string[], relevant: boolean, spanId?: string): Evidence => ({ id, action, title: action, lines, relevant, spanId });
const span = (s: Omit<TraceSpan, "events"> & { events?: TraceSpan["events"] }): TraceSpan => ({ events: [], ...s });
const ok = (label: string, value: string, hint?: string): HealthMetric => ({ label, value, status: "ok", kind: "health", hint });
const quality = (label: string, value: string, good: boolean, hint?: string): HealthMetric => ({ label, value, status: good ? "ok" : "error", kind: "quality", hint });

// ======================= Beginner: two search tools with the same description =======================
const ROUTING_SET = [...REQUESTS.map((r) => ({ id: r.id as string, text: r.text as string, expected: r.expected as string })), { id: "news", text: "What's the latest news about the Acme IPO?", expected: "web_search" }];
const VACATION = ROUTING_SET[0];

function catalogue(fixId?: string): ToolDef[] {
  const ov = CATALOGUES.overlapping.tools; // web_search and search_handbook share one generic description
  if (fixId === "f-desc") return CATALOGUES.clear.tools;
  if (fixId === "f-order") return [ov[1], ov[0], ...ov.slice(2)];
  if (fixId === "f-noweb") return ov.slice(1);
  return ov;
}

const toolDescScenario: Scenario = {
  id: "agent-tool-desc-collision", title: "HR assistant answers vacation questions from the internet", system: "Agent", difficulty: "Beginner", relatedTopic: "tool-selection",
  symptom: "“Most US companies offer about 10 vacation days” (handbook says 25)",
  objective: "See how a tool selector decides, and why two tools with the same description can't be told apart.",
  incident: "Since Tuesday, employees asking the HR assistant “How many vacation days do I get per year?” get a generic answer scraped from the web instead of the company handbook. Nothing errors, the handbook index is healthy, and weather, email and currency questions still work.",
  architecture: {
    nodes: [
      { id: "user", label: "Employee", Icon: User, x: 0, y: 110, match: (s) => s.id === "agent.run" },
      { id: "selector", label: "Tool selector", Icon: Bot, x: 190, y: 110, match: (s) => s.component === "selector" },
      { id: "web", label: "web_search", Icon: Globe, x: 400, y: 0, match: (s) => s.component === "tool:web_search" },
      { id: "handbook", label: "search_handbook", Icon: FileSearch, x: 400, y: 110, match: (s) => s.component === "tool:search_handbook" },
      { id: "other", label: "other tools", Icon: Wrench, x: 400, y: 220, match: (s) => s.component.startsWith("tool:") && !["tool:web_search", "tool:search_handbook"].includes(s.component) },
    ],
    edges: [{ source: "user", target: "selector" }, { source: "selector", target: "web" }, { source: "selector", target: "handbook" }, { source: "selector", target: "other" }],
  },
  run(fixId) {
    const tools = catalogue(fixId);
    const rows = ROUTING_SET.map((r) => ({ ...r, sel: scoreTools(r.text, tools) }));
    const vac = rows[0].sel;
    const right = rows.filter((r) => r.sel.winner === r.expected).length;
    const resolved = right === rows.length;
    const byName = (n: string) => vac.scores.find((s) => s.name === n);
    const searchTools = tools.filter((t) => t.name === "web_search" || t.name === "search_handbook");
    const vacOk = vac.winner === VACATION.expected;
    const answer = vacOk ? "Full-time employees get 25 vacation days per year (Employee handbook §4.2)." : "Most US companies offer about 10 vacation days per year, according to recent surveys.";
    const spans = [
      span({ id: "agent.run", name: "agent.run", component: "agent", startMs: 0, durationMs: 1100, status: vacOk ? "ok" : "error", input: { request: VACATION.text }, output: { answer } }),
      span({ id: "select", parentId: "agent.run", name: "select tool (simulated LLM)", component: "selector", startMs: 0, durationMs: 600, status: vacOk ? "ok" : "warning", output: { ranked: vac.ranked.slice(0, 3).join(" > "), margin: vac.margin },
        events: vac.tie ? [{ id: "tie", type: "decision", message: `tie between ${vac.ranked[0]} and ${vac.runnerUp} at ${byName(vac.winner)!.total}; first listed wins`, severity: "warning" }] : [] }),
      span({ id: "tool", parentId: "agent.run", name: `tool.${vac.winner}`, component: `tool:${vac.winner}`, startMs: 600, durationMs: 500, status: vacOk ? "ok" : "error", input: vac.call.arguments, output: { answer } }),
    ];
    return {
      spans, resolved,
      metrics: [ok("Assistant uptime", "100%"), ok("Tool error rate", "0%"), ok("p50 latency", "1.1 s"),
        quality("Routing accuracy (replay set)", `${right}/${rows.length}`, resolved, "requests sent to the expected tool"), quality("Vacation answer source", vacOk ? "handbook" : "web", vacOk)],
      transcript: [{ label: "Employee", text: VACATION.text }, { label: "Assistant", text: answer, tone: vacOk ? "ok" : "err" }],
      evidence: [
        ev("final", "Read the assistant's answer and the tool it used", [`tool called: ${vac.winner}(${JSON.stringify(vac.call.arguments)})`, `answer: “${answer}”`], false, "tool"),
        ev("scores", "Inspect the selector's scores for the vacation question", [`query words: [${vac.queryTokens.join(", ")}]`,
          ...vac.ranked.slice(0, 3).map((n) => { const s = byName(n)!; return `${n}: overlap ${s.overlap} (matched [${s.matched.join(", ")}]) + args ${s.argFit} × 0.3 − penalty ${s.penalty} = ${s.total}`; }),
          vac.tie ? `TIE: ${vac.winner} and ${vac.runnerUp} score the same, so the tool listed first wins` : `margin over ${vac.runnerUp}: ${vac.margin}`], true, "select"),
        ev("descriptions", "Read the descriptions of the two search tools", searchTools.map((t) => `${t.name}: “${t.description}”`).concat(new Set(searchTools.map((t) => t.description)).size < searchTools.length ? ["the two descriptions are identical: neither mentions handbook, vacation or news"] : []), true),
        ev("accuracy", "Replay the routing test set", rows.map((r) => `${r.text} → ${r.sel.winner}${r.sel.winner === r.expected ? " ✓" : `  ✗ expected ${r.expected}`}${r.sel.tie ? " (tie)" : ""}`), true),
        ev("budget", "Check whether tool descriptions are truncated", [`${vac.totalWords} description words across ${tools.length} tools; budget ${vac.budget}`, vac.wordsPerTool ? `truncated to ${vac.wordsPerTool} words per tool` : "nothing truncated: the selector sees every word"], false),
        ev("args", "Check which tools can fill their required arguments", vac.ranked.slice(0, 4).map((n) => { const s = byName(n)!; return `${n}: ${s.filled.length ? s.filled.map((f) => `${f.name}=${f.value ?? "missing"}`).join(", ") : "no required args"}`; }), false),
        ev("handbook", "Query the handbook index directly", ["search_handbook(\"vacation days per year\") → hr-handbook §4.2 “Full-time employees get 25 vacation days per year” (score 0.91)", "index: 214 pages, last sync today 06:00, 0 errors"], false, "tool"),
        ev("errors", "Read tool error logs", ["web_search: 0 errors / 1,204 calls", "search_handbook: 0 errors / 96 calls (down from 1,310 last week)"], false),
        ev("deploys", "Read the deploy log", ["Oct 14 tools-registry: “description cleanup”: search tools now share one generic description (“Search for information and answer questions.”)", "Oct 13 llm-gateway: model minor version bump", "Oct 10 hr-handbook: re-indexed (214 pages)"], true),
      ],
    };
  },
  hypotheses: [
    { id: "h-desc", text: "web_search and search_handbook now have the same generic description, so the selector can't tell them apart and the tie goes to whichever is listed first.", isRootCause: true, evidenceIds: ["scores", "descriptions", "deploys", "accuracy"], explanation: "Both tools score exactly the same for the vacation question: no word of the question appears in either description. Ties are broken by list order and web_search is listed first. The cleanup on Oct 14 made the descriptions identical." },
    { id: "h-llm", text: "The model is too weak for this; switch to a bigger LLM.", isRootCause: false, evidenceIds: ["scores", "descriptions"], explanation: "Any selector, however smart, only knows a tool by its name and description. With identical descriptions there is nothing to choose on; a bigger model just picks between two equal options." },
    { id: "h-index", text: "The handbook index is broken or empty.", isRootCause: false, evidenceIds: ["handbook", "errors"], explanation: "Queried directly, the handbook returns the right page. It isn't being called: its traffic fell from 1,310 to 96 calls a week." },
    { id: "h-trunc", text: "The tool catalogue is too long and descriptions get cut off.", isRootCause: false, evidenceIds: ["budget"], explanation: "Every description fits in the budget; nothing is truncated." },
    { id: "h-web", text: "web_search is returning bad results.", isRootCause: false, evidenceIds: ["final", "accuracy"], explanation: "It returns a reasonable web answer. The question should never have gone to the web." },
  ],
  fixes: [
    { id: "f-desc", label: "Give each search tool its own description, including what it is not for (“internal HR policy: vacation, PTO, benefits”, “public news; not for company policy”)", explanation: "The vacation question matches the handbook and is excluded from web_search; every routing test passes." },
    { id: "f-order", label: "List search_handbook before web_search in the catalogue", explanation: "The vacation question now wins the tie, but news questions lose it and go to the handbook. List order is still the only thing deciding." },
    { id: "f-noweb", label: "Remove web_search from the HR assistant", explanation: "Vacation questions work, but news questions now go to the handbook, which can't answer them." },
  ],
  verifications: [
    { id: "v-replay", label: "Replay the full routing set (vacation, weather, email, currency, news): every request picks its expected tool with no ties; add a CI check that rejects tools with overlapping descriptions", sound: true, explanation: "Checks both search tools in both directions and stops the next cleanup from collapsing descriptions again." },
    { id: "v-once", label: "Ask the vacation question once and check the answer cites the handbook", sound: false, explanation: "Reordering the list passes this and breaks news questions. One request can't show the tie is gone." },
    { id: "v-errors", label: "Confirm the tool error rate is 0%", sound: false, explanation: "It was 0% during the incident. Picking the wrong tool isn't an error." },
  ],
  rootCause: "A tool-registry cleanup gave web_search and search_handbook the same generic description. The selector scores them identically for HR questions and breaks the tie by list order, so vacation questions go to the web. Distinct descriptions with explicit “not for” lines fix it.",
};

// ======================= Beginner: timed-out charge retried without an idempotency key =======================
const PAY_CFG: RetryConfig = { ...DEFAULT_RETRY, scenario: "payment", idempotencyKey: false };
type Attempt = Extract<Segment, { kind: "attempt" }>;

const doubleChargeScenario: Scenario = {
  id: "agent-double-charge", title: "Customers charged twice for one order", system: "Agent", difficulty: "Beginner", relatedTopic: "retries-and-timeouts",
  symptom: "Two identical card charges for a single checkout",
  objective: "See why retrying a call that has side effects is only safe when the server can recognise the repeat.",
  incident: "Since Monday, customers who check out through the shopping agent sometimes see two identical charges on their card. The agent's own log shows one successful payment per order, and the payment provider's status page is green.",
  architecture: {
    nodes: [
      { id: "user", label: "Customer", Icon: User, x: 0, y: 90, match: (s) => s.id === "agent.run" },
      { id: "planner", label: "Checkout agent", Icon: Bot, x: 180, y: 90, match: (s) => s.component === "planner" },
      { id: "client", label: "Retry client", Icon: Wrench, x: 360, y: 90, match: (s) => s.component === "retry-client" },
      { id: "psp", label: "Payment provider", Icon: CreditCard, x: 560, y: 90, match: (s) => s.component === "tool:charge_card" },
    ],
    edges: [{ source: "user", target: "planner" }, { source: "planner", target: "client" }, { source: "client", target: "psp" }],
  },
  run(fixId) {
    const cfg: RetryConfig = { ...PAY_CFG, ...(fixId === "f-idem" ? { idempotencyKey: true } : fixId === "f-noretry" ? { maxAttempts: 1 } : fixId === "f-backoff" ? { baseBackoffMs: 2000 } : {}) };
    const r = runRetries(cfg);
    const attempts = r.segments.filter((s): s is Attempt => s.kind === "attempt");
    const paid = r.status === "success";
    const resolved = paid && r.charges === 1;
    const reply = paid ? "Your order is confirmed and your card was charged $84.20." : "Sorry, the payment didn't go through. Please try again.";
    const t0 = 600;
    const spans: TraceSpan[] = [
      span({ id: "agent.run", name: "agent.run", component: "agent", startMs: 0, durationMs: t0 + r.endMs + 300, status: resolved ? "ok" : "error", input: { request: "Check out my cart" }, output: { reply, charges: r.charges } }),
      span({ id: "decide", parentId: "agent.run", name: "plan next action (simulated LLM)", component: "planner", startMs: 0, durationMs: t0, status: "ok", output: { tool: "charge_card", args: '{"order":"A-1042","amount_usd":84.2}' } }),
      span({ id: "call", parentId: "agent.run", name: "tool.charge_card (with retries)", component: "retry-client", startMs: t0, durationMs: r.endMs, status: paid ? "ok" : "error", metrics: { attempts: attempts.length }, output: { status: r.status } }),
      ...r.segments.map((s, i) => span({
        id: `seg${i}`, parentId: "call", name: s.kind === "attempt" ? `attempt ${s.n}` : `backoff ${s.end - s.start} ms`, component: s.kind === "attempt" ? "tool:charge_card" : "retry-client",
        startMs: t0 + s.start, durationMs: s.end - s.start, status: s.kind === "attempt" && s.outcome !== 200 ? "warning" : "ok",
        ...(s.kind === "attempt" ? { output: { client: String(s.outcome), server_finished_ms: s.serverEnd, charge: s.charge ?? "none" } } : {}),
      })),
    ];
    return {
      spans, resolved,
      metrics: [ok("Agent uptime", "100%"), ok("Provider status", "operational"), ok("Checkout success", paid ? "100%" : "0%"),
        quality("Charges per order", String(r.charges), r.charges === 1), quality("Customer outcome", resolved ? "paid once" : r.charges > 1 ? "double-charged" : "charged but told it failed", resolved)],
      transcript: [{ label: "Customer", text: "Check out my cart, please." }, { label: "Agent", text: reply, tone: resolved ? "ok" : "err" }],
      evidence: [
        ev("final", "Read the agent's reply and final status", [`client status: ${r.status}`, `reply: “${reply}”`, r.reason], false),
        ev("planner", "Count charge_card calls issued by the planner", ["1 tool call: charge_card({order: 'A-1042', amount_usd: 84.2})", "the planner did not call the tool again; repeats below happen inside the retry client"], false, "decide"),
        ev("attempts", "Inspect each attempt the retry client made", attempts.map((a) => `attempt ${a.n}: client ${a.outcome === "timeout" ? `gave up at ${a.end - a.start} ms (timeout ${cfg.timeoutMs} ms)` : `got ${a.outcome} after ${a.end - a.start} ms`}; server finished at ${a.serverEnd - a.start} ms`), true, "call"),
        ev("provider", "Read the payment provider's request log", attempts.map((a) => `${a.charge === "new" ? "NEW charge created" : a.charge === "replayed" ? "repeat key: returned the stored charge, no new charge" : "no charge"} at t=${a.serverEnd} ms (request ${a.n})`).concat([`charges for order A-1042: ${r.charges}`]), true),
        ev("headers", "Inspect the charge_card request headers", [`Idempotency-Key: ${cfg.idempotencyKey ? "order-A-1042-charge" : "(absent)"}`, "Content-Type: application/json", "the provider can only spot a repeat if both requests carry the same key"], true),
        ev("policy", "Check the retry policy configuration", [`max_attempts: ${cfg.maxAttempts}`, `timeout: ${cfg.timeoutMs} ms`, `retry_on: timeout, 429, 5xx`, `backoff: ${cfg.baseBackoffMs} ms × 2^n, jitter ${cfg.jitter}`], false),
        ev("status", "Check the payment provider's status page", ["API: operational (99.99% this month)", "p50 latency 480 ms; p99 2.7 s since fraud screening launched Oct 13"], false),
        ev("backoff", "Measure the gap between attempts", r.segments.filter((s) => s.kind === "backoff").map((s) => `waited ${s.end - s.start} ms before attempt ${s.n + 1}`).concat(r.segments.some((s) => s.kind === "backoff") ? [] : ["no retries were made"]), false),
        ev("deploys", "Read the deploy log", ["Oct 13 payment provider: fraud screening on first charge (slow first calls, up to ~2.6 s)", "Oct 12 checkout-agent: charge_card tool rewritten with a plain HTTP client (the old SDK added Idempotency-Key automatically)", "Oct 09 planner prompt v7"], true),
      ],
    };
  },
  hypotheses: [
    { id: "h-idem", text: "The first charge times out on the client but still succeeds on the server; the retry creates a second charge because requests carry no idempotency key.", isRootCause: true, evidenceIds: ["attempts", "provider", "headers"], explanation: "Attempt 1 is cut off at 2,000 ms, yet the provider finishes it at 2,600 ms and charges the card. Attempt 2 has no Idempotency-Key, so the provider can't tell it's a repeat and charges again." },
    { id: "h-llm", text: "The LLM is calling charge_card twice.", isRootCause: false, evidenceIds: ["planner"], explanation: "The planner issued exactly one tool call. The repeat comes from the retry client underneath it." },
    { id: "h-provider", text: "The payment provider has a bug that processes requests twice.", isRootCause: false, evidenceIds: ["provider", "status"], explanation: "The provider received two separate requests and did what each asked. It can only de-duplicate when told they are the same." },
    { id: "h-backoff", text: "Retries fire too fast, before the first request settles.", isRootCause: false, evidenceIds: ["backoff", "provider"], explanation: "However long the client waits, the second request is still a new request to charge the card." },
  ],
  fixes: [
    { id: "f-idem", label: "Send an Idempotency-Key per order on every charge_card attempt", explanation: "The retry returns the stored charge instead of creating a new one: one charge, order confirmed." },
    { id: "f-noretry", label: "Turn off retries for charge_card", explanation: "The card is still charged once by the timed-out first call, but the agent tells the customer it failed, so they pay again by hand." },
    { id: "f-backoff", label: "Wait longer between retries (2 s base backoff)", explanation: "The retry is later but still a new charge. The customer is still charged twice." },
  ],
  verifications: [
    { id: "v-replay", label: "Replay with a slow first charge: the provider log shows one new charge and one replayed result; add a test that every side-effecting tool sends an idempotency key", sound: true, explanation: "Reproduces the timeout-then-retry path and checks the outcome at the provider, where the money moves." },
    { id: "v-status", label: "Confirm the provider status page is green", sound: false, explanation: "It was green throughout. Nothing was down." },
    { id: "v-reply", label: "Confirm the agent tells customers their order is confirmed", sound: false, explanation: "It said that during the incident, while charging twice." },
  ],
  rootCause: "Fraud screening made first charges slower than the 2 s client timeout. The client gave up and retried, but the provider had already charged the card. Since the rewritten tool dropped the Idempotency-Key, the retry looked like a new payment and charged again.",
};

// ======================= Beginner: termination check matches the planner's own words =======================
const STOP_PATTERN = /total/i;

const earlyStopScenario: Scenario = {
  id: "agent-early-stop", title: "Quote agent replies with a discount fact instead of a price", system: "Agent", difficulty: "Beginner", relatedTopic: "agent-termination",
  symptom: "“Learned: 15% discount applies to 10+ seats” returned as the answer",
  objective: "Find out who decided the run was finished, and why ending on a word match is fragile.",
  incident: "After this week's runtime update, the quoting agent answers “What would 12 Pro seats cost for one year?” with “Learned: 15% discount applies to 10+ seats” and no price. Runs are faster and cheaper than before, and no errors are logged.",
  architecture: {
    nodes: [
      { id: "user", label: "Sales rep", Icon: User, x: 0, y: 110, match: (s) => s.id === "agent.run" },
      { id: "planner", label: "Planner (sim)", Icon: Bot, x: 180, y: 110, match: (s) => s.component === "planner" },
      { id: "stop", label: "Stop check", Icon: ClipboardList, x: 180, y: 240, match: (s) => s.component === "stop-check" },
      { id: "search", label: "search_docs", Icon: FileSearch, x: 400, y: 0, match: (s) => s.component === "tool:search_docs" },
      { id: "getdoc", label: "get_document", Icon: FileText, x: 400, y: 75, match: (s) => s.component === "tool:get_document" },
      { id: "calc", label: "calculator", Icon: Calculator, x: 400, y: 150, match: (s) => s.component === "tool:calculator" },
      { id: "submit", label: "submit_result", Icon: Send, x: 400, y: 225, match: (s) => s.component === "tool:submit_result" },
    ],
    edges: [{ source: "user", target: "planner" }, { source: "planner", target: "stop", sourceHandle: "b", targetHandle: "t" }, ...["search", "getdoc", "calc", "submit"].map((t) => ({ source: "planner", target: t }))],
  },
  run(fixId) {
    const cfg: AgentConfig = { ...DEFAULT_AGENT, task: "quote", ...(fixId === "f-iter" ? { maxIterations: 15 } : {}) };
    const r = runAgent(cfg);
    const textStop = fixId !== "f-structured";
    // The runtime's stop check runs on each planner decision before the tool is called.
    const decisions = r.steps.filter((s) => s.state.phase === "decide" && s.state.decision);
    const hit = textStop ? decisions.find((s) => STOP_PATTERN.test(s.state.decision!.reason)) : undefined;
    const stopIter = hit?.state.iteration;
    const lastObs = stopIter ? r.steps.filter((s) => s.id.endsWith("-observe") && s.state.iteration < stopIter).at(-1)?.description ?? "" : "";
    const answer = stopIter ? lastObs : r.final.text;
    const calls = stopIter ? r.toolCalls.slice(0, stopIter - 1) : r.toolCalls;
    const memory = stopIter ? hit!.state.memory : r.steps.at(-1)!.state.memory;
    const correct = !stopIter && r.correct;
    const iters = stopIter ?? r.iterations;
    const keep = (s: TraceSpan) => { const m = s.id.match(/^iter-(\d+)/); return !m || !stopIter || Number(m[1]) < stopIter || s.id === `iter-${stopIter}` || s.id === `iter-${stopIter}.decide`; };
    const spans = r.spans.filter(keep).map((s) => ({ ...s }));
    if (stopIter) {
      const d = spans.find((s) => s.id === `iter-${stopIter}.decide`)!;
      spans.push(span({ id: "stop", parentId: `iter-${stopIter}`, name: "runtime stop check", component: "stop-check", startMs: d.startMs + d.durationMs, durationMs: 1, status: "error",
        events: [{ id: "stop", type: "termination", message: `stop_when text_matches “total” matched: “${hit!.state.decision!.reason}”`, severity: "error" }] }));
      spans[0] = { ...spans[0], durationMs: d.startMs + d.durationMs + 1, status: "error", output: { status: "stopped_by_rule", answer } };
    }
    const expectedTotal = `$${EXPECTED_TOTAL.toLocaleString("en-US")}`;
    return {
      spans, resolved: correct,
      metrics: [ok("Agent uptime", "100%"), ok("Requests completed", "100%", "returned any response"), ok("Iterations used", `${iters}/${cfg.maxIterations}`), ok("Tool error rate", `${Math.round((calls.filter((c) => !c.ok).length / Math.max(1, calls.length)) * 100)}%`),
        quality("Answer contains a price", /\$\d/.test(answer) ? "yes" : "no", /\$\d/.test(answer)), quality("Task correct", correct ? "yes" : "no", correct)],
      transcript: [{ label: "User", text: TASKS.quote }, { label: "Agent", text: answer, tone: correct ? "ok" : "err" }],
      evidence: [
        ev("final", "Read the final answer and status", [`status: ${stopIter ? "stopped_by_rule" : r.final.status}`, `answer: “${answer}”`, `expected: ${expectedTotal} (12 × $49 × 12 × 0.85)`], false),
        ev("history", "Inspect the tool-call history", calls.map((c, i) => `${i + 1}. ${c.tool}(${JSON.stringify(c.args)}) → ${c.ok ? "ok" : c.attempts.at(-1)?.message}`).concat(calls.some((c) => c.tool === "submit_result") ? [] : ["calculator and submit_result were never called"]), true),
        ev("stoplog", "Read the runtime's termination log", stopIter ? [`iteration ${stopIter}: planner decided “${hit!.state.decision!.reason}” → next tool ${hit!.state.decision!.tool}`, `stop_when.text_matches “total” matched the planner's text → run ended before the tool call`, `the last observation was returned as the answer: “${lastObs}”`] : [`run ended when submit_result succeeded on iteration ${r.iterations}`], true, stopIter ? "stop" : undefined),
        ev("memory", "Inspect the agent's working memory at the end", Object.entries(memory).filter(([, v]) => v !== undefined).map(([k, v]) => `${k} = ${JSON.stringify(v)}`).concat(memory.total === undefined ? ["no total computed yet"] : []), true),
        ev("config", "Read the agent-runtime configuration", [`max_iterations: ${cfg.maxIterations}`, `max_retries: ${cfg.maxRetries}`, textStop ? "stop_when: { text_matches: \"total\" }   # new in runtime 1.8: “end early once the model mentions the total”" : "stop_when: { tool: \"submit_result\", status: \"ok\" }"], true),
        ev("budget", "Check iteration count against the budget", [`iterations used: ${iters} of ${cfg.maxIterations}`, `tool calls: ${calls.length}`], false),
        ev("failing", "Inspect failing tool calls", calls.some((c) => !c.ok) ? calls.filter((c) => !c.ok).map((c) => `${c.tool}: ${c.attempts.at(-1)?.message}`) : ["No tool call failed."], false),
        ev("latency", "Check tool latency", calls.map((c) => `${c.tool}: ${c.attempts.map((a) => `${a.latencyMs} ms`).join(", ")}`).concat([`timeout: ${cfg.timeoutMs} ms`]), false),
        ev("deploys", "Read the deploy log", ["Oct 16 agent-runtime 1.8: new stop_when rule ends runs early when the model's text matches a keyword (saves ~2 iterations per run)", "Oct 15 kb: discount page wording updated", "Oct 14 llm-gateway: upgraded TLS certificates"], true),
      ],
    };
  },
  hypotheses: [
    { id: "h-stop", text: "The new stop rule ends the run whenever the planner's text contains “total”, which happens one step before it actually computes the total.", isRootCause: true, evidenceIds: ["stoplog", "config", "history", "memory"], explanation: "On iteration 4 the planner says “compute the total” and is about to call the calculator. The keyword rule fires on that sentence and returns the previous observation as the answer." },
    { id: "h-llm", text: "The model is confused and answers with the wrong fact.", isRootCause: false, evidenceIds: ["memory", "stoplog"], explanation: "Memory holds the price and the discount, and the planner's next step was the right one. The runtime cut it off; the model never chose that answer." },
    { id: "h-budget", text: "The agent hits its iteration limit; raise the step limit.", isRootCause: false, evidenceIds: ["budget"], explanation: "It used 4 of 8 iterations. Something else ended the run." },
    { id: "h-calc", text: "The calculator is failing.", isRootCause: false, evidenceIds: ["history", "failing"], explanation: "The calculator was never called. No tool failed." },
    { id: "h-kb", text: "The discount page edit broke the quote.", isRootCause: false, evidenceIds: ["memory"], explanation: "The discount was read correctly (0.15). The run just stopped before using it." },
  ],
  fixes: [
    { id: "f-structured", label: "End the run only when submit_result succeeds (a structured signal), not on words in the model's text", explanation: "The planner computes the total and submits $5,997.6." },
    { id: "f-iter", label: "Raise max_iterations to 15", explanation: "The run still ends on iteration 4 because the keyword rule fires first." },
    { id: "f-prompt", label: "Add “always finish by calling submit_result” to the system prompt", explanation: "The planner already plans to finish that way; the runtime stops it before it gets there." },
  ],
  verifications: [
    { id: "v-replay", label: "Replay the quote task: it ends with submit_result($5,997.6); add a test where the planner's intermediate text contains “total” and “done” and the run continues", sound: true, explanation: "Checks the outcome and the specific trap: words in reasoning text must not end the loop." },
    { id: "v-budget", label: "Check that runs stay under the iteration budget", sound: false, explanation: "They were well under budget during the incident. Ending early looks great on this metric." },
    { id: "v-errors", label: "Confirm the tool error rate is 0%", sound: false, explanation: "It was 0% throughout. Nothing failed; the run was stopped." },
  ],
  rootCause: "Runtime 1.8 added a stop rule that ends the run when the model's text matches “total”. The planner says “compute the total” one step before calling the calculator, so every quote stops early and returns the previous observation (the discount fact) as the answer.",
};

// ======================= Advanced: parallel agents overwrite a shared blackboard key =======================
const COORD_CFG: CoordConfig = { ...DEFAULT_COORD, pattern: "orchestrator", failure: "conflict", arbitration: false };
const COORD_ROLE: Record<AgentId, string> = { user: "User", orch: "Orchestrator", research: "Researcher", coder: "Coder", review: "Reviewer" };

const blackboardScenario: Scenario = {
  id: "agent-blackboard-overwrite", title: "Generated API client ships with the wrong rate limit", system: "Agent", difficulty: "Advanced", relatedTopic: "multi-agent-coordination",
  symptom: "client.py throttles to 60 req/min; the API allows 100",
  objective: "Trace a value through a shared blackboard written by parallel agents, and separate a real cause from a coincidental change.",
  incident: "The multi-agent coding assistant generated a Weather API client throttled to 60 requests/min, though the API docs say 100/min, so batch jobs run 40% slower. The reviewer agent approved it. Last week the researcher agent was moved to a cheaper model, and the orchestrator's config was migrated from JSON to YAML.",
  architecture: {
    nodes: [
      { id: "user", label: "User", Icon: User, x: 0, y: 110, match: (s) => s.component === "user" || s.id === "agent.run" },
      { id: "orch", label: "Orchestrator", Icon: Bot, x: 170, y: 110, match: (s) => s.component === "orch" },
      { id: "research", label: "Researcher", Icon: Search, x: 360, y: 0, match: (s) => s.component === "research" },
      { id: "coder", label: "Coder", Icon: Wrench, x: 360, y: 110, match: (s) => s.component === "coder" },
      { id: "review", label: "Reviewer", Icon: Users, x: 360, y: 220, match: (s) => s.component === "review" },
      { id: "bb", label: "Shared blackboard", Icon: Database, x: 560, y: 110, match: (s) => s.component === "blackboard" },
    ],
    edges: [{ source: "user", target: "orch" }, ...["research", "coder", "review"].flatMap((a) => [{ source: "orch", target: a }, { source: a, target: "bb" }])],
  },
  run(fixId) {
    const cfg: CoordConfig = { ...COORD_CFG, ...(fixId === "f-arbitrate" ? { arbitration: true } : fixId === "f-scratch" ? { failure: "none" as const } : fixId === "f-pipeline" ? { pattern: "pipeline" as const } : {}) };
    const { msgs, outcome, steps } = simulateCoord(cfg);
    const bb = steps.at(-1)!.state.bb;
    const resolved = outcome.status === "success" && outcome.limitUsed === 100;
    const final = msgs.find((m) => m.kind === "final")!;
    const end = Math.max(...msgs.map((m) => m.arrive));
    const rate = bb.filter((e) => e.key === "rate_limit");
    const research = msgs.find((m) => m.from === "research" && m.kind === "result");
    const spans: TraceSpan[] = [
      span({ id: "agent.run", name: "multi-agent run", component: "agent", startMs: 0, durationMs: end, status: resolved ? "ok" : "error", input: { task: msgs[0].label }, output: { deliverable: final.label } }),
      ...msgs.filter((m) => m.kind !== "task" || m.from !== "user").map((m) => span({
        id: `m${m.i}`, parentId: "agent.run", name: `${COORD_ROLE[m.from]} → ${COORD_ROLE[m.to]}: ${m.label}`, component: m.from, startMs: m.sent - m.work, durationMs: m.work + 50,
        status: m.bb?.key === "rate_limit" && !m.bb.cited ? "warning" : "ok", output: m.bb ? { [m.bb.key]: m.bb.value, cited: m.bb.cited } : undefined,
      })),
      ...rate.map((e, i) => span({ id: `bb${i}`, parentId: "agent.run", name: `blackboard write rate_limit = ${e.value}`, component: "blackboard", startMs: e.at, durationMs: 10, status: e.status === "conflict" ? "error" : "ok", output: { by: e.by, status: e.status } })),
    ];
    return {
      spans, resolved,
      metrics: [ok("Agents healthy", "4/4"), ok("Hand-offs", `${msgs.filter((m) => m.kind === "handoff").length}/${cfg.maxHandoffs}`), ok("Wall clock", `${(end / 1000).toFixed(1)} s`), ok("Reviewer verdict", "approved"),
        quality("Rate limit in client.py", outcome.limitUsed ? `${outcome.limitUsed}/min` : "none", outcome.limitUsed === 100, "API docs: 100/min"), quality("Open blackboard conflicts", String(bb.filter((e) => e.status === "conflict").length), !bb.some((e) => e.status === "conflict"))],
      transcript: [{ label: "User", text: msgs[0].label }, { label: "Assistant", text: final.label, tone: resolved ? "ok" : "err" }],
      evidence: [
        ev("final", "Read the delivered result", [`deliverable: ${final.label}`, `status: ${outcome.status}`, "API docs §4: 100 requests/min"], false),
        ev("blackboard", "Inspect every write to the rate_limit key on the blackboard", rate.map((e) => `t=${(e.at / 1000).toFixed(2)} s ${COORD_ROLE[e.by]} wrote ${e.value} (${e.cited ? "cited: docs §4" : "uncited assumption"}) → ${e.status}`), true),
        ev("timeline", "Read the message timeline", msgs.map((m) => `${(m.sent / 1000).toFixed(2)} s ${COORD_ROLE[m.from]} → ${COORD_ROLE[m.to]} [${m.kind}] ${m.label}`), true),
        ev("orchconfig", "Read the orchestrator's effective config", [`pattern: ${cfg.pattern}`, `max_handoffs: ${cfg.maxHandoffs}`, `worker_timeout_ms: ${cfg.workerTimeoutMs}`, cfg.arbitration ? "conflict_policy: arbitrate (cited source wins)" : "conflict_policy: (unset → default last_write_wins)   # key was conflictPolicy in JSON, renamed in the YAML schema"], true),
        ev("research", "Read the researcher's output", research ? [`“${research.label}” (cited: ${research.bb?.cited ? "yes" : "no"})`, `worked ${(research.work / 1000).toFixed(1)} s (previous model: 1.1 s)`, "token cost per run: −68% since the model change"] : ["researcher produced no result"], false),
        ev("coder", "Read the coder's first draft", msgs.filter((m) => m.from === "coder").map((m) => `${(m.sent / 1000).toFixed(2)} s: ${m.label}${m.bb ? ` (blackboard ${m.bb.key} = ${m.bb.value}${m.bb.cited ? ", cited" : ", uncited"})` : ""}`), true),
        ev("review", "Read the reviewer's notes", ["approved (2 nits): rename `sleep_s`, add type hints", "reviewer checks the code against the orchestrator's instruction, not against the API docs"], false),
        ev("handoffs", "Check hand-offs and timeouts", [`hand-offs: ${msgs.filter((m) => m.kind === "handoff").length}`, `timeouts: ${msgs.filter((m) => m.kind === "timeout").length}`, "every worker replied within its timeout"], false),
        ev("deploys", "Read the deploy log", ["Oct 15 researcher: model switched to a smaller, cheaper model", "Oct 14 orchestrator: config migrated JSON → YAML (schema v3)", "Oct 11 coder: prompt tweak, “publish working assumptions to the blackboard so others can see them”"], true),
      ],
    };
  },
  hypotheses: [
    { id: "h-overwrite", text: "The coder publishes its uncited 60/min assumption to the same blackboard key as the researcher's cited 100/min, and with arbitration lost in the config migration, the later write wins.", isRootCause: true, evidenceIds: ["blackboard", "orchconfig", "coder", "timeline"], explanation: "Both values sit on rate_limit in conflict. The coder's write lands 0.6 s after the researcher's, and with conflict_policy unset the orchestrator applies the last write. Either factor alone would be harmless: arbitration resolves the conflict, and without the coder's write there is no conflict." },
    { id: "h-model", text: "The cheaper researcher model found the wrong rate limit.", isRootCause: false, evidenceIds: ["research", "blackboard"], explanation: "A real change, but irrelevant: the researcher reported 100/min with a citation. It's a bit slower and much cheaper, and that has nothing to do with the wrong value." },
    { id: "h-review", text: "The reviewer agent is too lenient and should have caught it.", isRootCause: false, evidenceIds: ["review", "timeline"], explanation: "The reviewer checks code against the instruction it got, and the instruction already said 60/min. A stricter reviewer is a safety net, not the cause." },
    { id: "h-coder", text: "The coder LLM hallucinated 60/min.", isRootCause: false, evidenceIds: ["coder", "timeline"], explanation: "Drafting with a placeholder before research arrives is expected. The failure is that the system let an uncited draft value override a cited one." },
    { id: "h-loop", text: "Agents are stuck handing work back and forth.", isRootCause: false, evidenceIds: ["handoffs"], explanation: "Zero hand-offs and zero timeouts; the run is fast and complete." },
  ],
  fixes: [
    { id: "f-arbitrate", label: "Restore conflict_policy: arbitrate under the new YAML key (cited source wins)", explanation: "The orchestrator sees two values for rate_limit, asks the reviewer to arbitrate, and the cited 100/min wins." },
    { id: "f-scratch", label: "Have the coder keep draft assumptions in its own scratch space instead of the shared rate_limit key", explanation: "Only the researcher writes rate_limit, so there's nothing to conflict." },
    { id: "f-model", label: "Move the researcher back to the larger model", explanation: "The researcher was already right. The coder's write still lands last and wins." },
    { id: "f-pipeline", label: "Switch from orchestrator–worker to a pipeline", explanation: "Without arbitration the reviewer still doesn't compare the two values; 60/min ships." },
  ],
  verifications: [
    { id: "v-conflict", label: "Replay with the coder's draft assumption: the blackboard shows the conflict resolved to the cited 100/min and client.py uses 100/min; add a config test that conflict_policy is set after any schema migration", sound: true, explanation: "Exercises the conflicting-write path and guards the setting that was silently lost." },
    { id: "v-research", label: "Re-run the researcher alone and confirm it returns 100/min", sound: false, explanation: "It already returned 100/min. That doesn't test how the two writes are merged." },
    { id: "v-approved", label: "Confirm the reviewer approves the new client", sound: false, explanation: "The reviewer approved the wrong client too." },
  ],
  rootCause: "Two things combined. A coder prompt change made it publish its placeholder 60/min to the shared rate_limit key, and the YAML config migration dropped conflict_policy, so the orchestrator fell back to last-write-wins. The coder's write landed after the researcher's cited 100/min and overrode it. The researcher's model change was a coincidence.",
};

// ======================= Advanced: evicted update loses to a stale, more-similar long-term fact =======================
const MEM_CFG: MemoryConfig = { ...DEFAULT_MEMORY, conflict: true, windowTokens: 44 };

/** Supersede-on-write: among retrieved long-term facts for a slot, the newest one wins. */
function newestWins(slots: SlotAnswer[], hits: ReturnType<typeof simulateMemory>[number]["state"]["hits"], cfg: MemoryConfig): SlotAnswer[] {
  const exp = expectedSlots(cfg);
  return slots.map((s) => {
    if (s.source !== "long-term") return s;
    const newest = hits.filter((h) => h.retrieved && h.fact.values[s.slot]).sort((a, b) => b.fact.turn - a.fact.turn)[0];
    const value = newest.fact.values[s.slot]!;
    return { ...s, value, correct: value === exp[s.slot], conflict: undefined };
  });
}

const staleMemoryScenario: Scenario = {
  id: "agent-stale-memory", title: "Travel agent books dinner for the old arrival date", system: "Agent", difficulty: "Advanced", relatedTopic: "short-term-and-long-term-memory",
  symptom: "Dinner booked for 12 May; the user said her flight moved to 19 May",
  objective: "Follow one fact through the context window, the summary and long-term memory, and see how two memory settings combine to bring back a superseded fact.",
  incident: "Priya told the travel assistant her flight moved and she now arrives 19 May. Two messages later it booked her first-night dinner in Lisbon for 12 May, the original date. Last week the memory team cut the context window from 80 to 44 tokens to save cost; the hotel-budget fact has also been showing low retrieval scores.",
  architecture: {
    nodes: [
      { id: "user", label: "User", Icon: User, x: 0, y: 110, match: (s) => s.id === "agent.run" },
      { id: "window", label: "Context window", Icon: History, x: 180, y: 20, match: (s) => s.component === "window" },
      { id: "store", label: "Long-term store", Icon: Database, x: 180, y: 200, match: (s) => s.component === "longterm" },
      { id: "retriever", label: "Memory retriever", Icon: Brain, x: 380, y: 200, match: (s) => s.component === "retriever" },
      { id: "planner", label: "Planner (sim)", Icon: Bot, x: 380, y: 20, match: (s) => s.component === "planner" },
      { id: "book", label: "book_dinner", Icon: Utensils, x: 580, y: 110, match: (s) => s.component === "tool:book_dinner" },
    ],
    edges: [{ source: "user", target: "window" }, { source: "user", target: "store" }, { source: "store", target: "retriever" }, { source: "retriever", target: "planner" }, { source: "window", target: "planner" }, { source: "planner", target: "book" }],
  },
  run(fixId) {
    const cfg: MemoryConfig = { ...MEM_CFG, ...(fixId === "f-topk" ? { topK: 1 } : fixId === "f-threshold" ? { threshold: 0.5 } : fixId === "f-nolt" ? { longTerm: false } : {}) };
    const steps = simulateMemory(cfg);
    const end = steps.at(-1)!.state;
    const slots = fixId === "f-supersede" ? newestWins(end.answer!.slots, end.hits, cfg) : end.answer!.slots;
    const resolved = slots.every((s) => s.correct);
    const val = (k: string) => slots.find((s) => s.slot === k)?.value;
    const missing = slots.filter((s) => !s.value).map((s) => s.slot);
    const answer = `Booked a ${val("diet") ?? ""} dinner in ${val("city") ?? "?"}${val("date") ? ` on ${val("date")}` : ""}.${missing.length ? ` (I don't know your ${missing.join(" or ")}: could you remind me?)` : ""}`;
    const evicted = steps.flatMap((s) => s.state.evicted);
    const exchangeSteps = steps.filter((s) => s.state.phase === "exchange");
    const replay80 = simulateMemory({ ...cfg, windowTokens: 80 }).at(-1)!.state.answer!.slots.find((s) => s.slot === "date")!;
    const date = slots.find((s) => s.slot === "date")!;
    const t = exchangeSteps.length * 300;
    const spans: TraceSpan[] = [
      span({ id: "agent.run", name: "agent.run", component: "agent", startMs: 0, durationMs: t + 1200, status: resolved ? "ok" : "error", input: { question: QUESTION }, output: { answer } }),
      ...exchangeSteps.map((s, i) => span({ id: `turn${i + 1}`, parentId: "agent.run", name: s.title, component: "window", startMs: i * 300, durationMs: 300, status: s.state.evicted.length ? "warning" : "ok",
        events: s.events.map((e) => ({ ...e })), metrics: { tokens_used: s.state.used, window: cfg.windowTokens } })),
      span({ id: "store", parentId: "agent.run", name: "long-term writes", component: "longterm", startMs: 0, durationMs: t, status: "ok", output: { facts: end.store.map((f) => f.text).join(" · ") } }),
      span({ id: "retrieve", parentId: "agent.run", name: "retrieve memories", component: "retriever", startMs: t, durationMs: 200, status: date.conflict ? "warning" : "ok", output: { retrieved: end.hits.filter((h) => h.retrieved).map((h) => `${h.fact.id}:${h.score.toFixed(2)}`).join(", ") } }),
      span({ id: "plan", parentId: "agent.run", name: "fill booking slots (simulated LLM)", component: "planner", startMs: t + 200, durationMs: 600, status: resolved ? "ok" : "error", output: Object.fromEntries(slots.map((s) => [s.slot, `${s.value ?? "?"} (${s.source})`])) }),
      span({ id: "book", parentId: "agent.run", name: "tool.book_dinner", component: "tool:book_dinner", startMs: t + 800, durationMs: 400, status: resolved ? "ok" : "error", input: { city: val("city"), date: val("date"), diet: val("diet") } }),
    ];
    return {
      spans, resolved,
      metrics: [ok("Assistant uptime", "100%"), ok("Memory store", `${end.store.length} facts, 0 write errors`), ok("Window usage", `${end.used}/${cfg.windowTokens} tokens`), ok("Tool error rate", "0%"),
        quality("Booking date", val("date") ?? "missing", date.correct, "user's latest date: 19 May"), quality("Slots correct", `${slots.filter((s) => s.correct).length}/${SLOTS.length}`, resolved)],
      transcript: [{ label: "User", text: QUESTION }, { label: "Assistant", text: answer, tone: resolved ? "ok" : "err" }],
      evidence: [
        ev("final", "Read the booking the agent made", [`answer: “${answer}”`, `user's latest arrival date: ${expectedSlots(cfg).date}`], false, "book"),
        ev("slots", "See where each booking detail came from", slots.map((s) => `${s.slot} = ${s.value ?? "missing"} from ${s.source}${s.conflict ? `  ⚠ ${s.conflict}` : ""}`), true, "plan"),
        ev("window", "Inspect the context window at question time", [`${end.used}/${cfg.windowTokens} tokens in the window`, ...end.buffer.map((b) => `${b.role}: “${b.text}”`), ...evicted.filter((e) => e.turn.role === "user").map((e) => `evicted earlier: “${e.turn.text}”`)], true),
        ev("retrieval", "Read the long-term retrieval scores", end.hits.map((h) => `${h.fact.text}: cosine ${h.score.toFixed(3)}${h.retrieved ? " (retrieved)" : ""}`).concat(["ranking uses similarity only; it has no notion of which fact is newer"]), true, "retrieve"),
        ev("store", "List the stored long-term facts", end.store.map((f) => `turn ${f.turn}: “${f.text}”${Object.keys(f.values).length ? ` → ${JSON.stringify(f.values)}` : ""}`).concat(["no fact is marked superseded; the old and new dates are both live"]), true, "store"),
        ev("replay", "Replay the same conversation with the old 80-token window", [`date = ${replay80.value} from ${replay80.source}`, "the update was still in the window, so long-term memory was never consulted for the date"], true),
        ev("summary", "Read the running summary", end.summary.map((s) => `“${s.text}”`).concat(["summaries are deliberately short: “Flight was moved.” keeps the event, not the date"]), false),
        ev("hotel", "Investigate the low-scoring hotel-budget fact", end.hits.filter((h) => h.fact.id === "hotel").map((h) => `“${h.fact.text}”: cosine ${h.score.toFixed(3)} vs threshold ${cfg.threshold} → ${h.retrieved ? "retrieved" : "not retrieved"}`).concat(["the dinner question doesn't need the hotel budget"]), false),
        ev("deploys", "Read the deploy log", ["Oct 15 memory: context window 80 → 44 tokens (cost saving; “long-term memory will cover the rest”)", "Oct 12 memory: long-term store writes every salient fact; no update/merge step", "Oct 10 planner prompt v9"], true),
      ],
    };
  },
  hypotheses: [
    { id: "h-stale", text: "The smaller window evicted the date change, and long-term memory keeps both dates with no recency rule, so the older, more similar 12 May fact wins.", isRootCause: true, evidenceIds: ["slots", "window", "retrieval", "store", "replay"], explanation: "At 44 tokens the “now arrive 19 May” turn is gone from the window, so the date comes from long-term memory. Both facts are retrieved, the original trip fact scores higher (0.70 vs 0.63), and nothing marks it superseded. With the old 80-token window the update stayed in short-term memory, which hid the missing recency rule." },
    { id: "h-llm", text: "The model ignored the user's correction.", isRootCause: false, evidenceIds: ["window", "slots"], explanation: "By the time of the question, the correction was no longer in the context window. The model chose between two stored dates by the retriever's ranking." },
    { id: "h-miss", text: "Long-term memory failed to store or retrieve the update.", isRootCause: false, evidenceIds: ["store", "retrieval"], explanation: "The update was stored (turn 4) and retrieved. It was outranked, not missed." },
    { id: "h-threshold", text: "The retrieval threshold is too high: the hotel fact is being dropped.", isRootCause: false, evidenceIds: ["hotel", "slots"], explanation: "A real anomaly but a red herring: the hotel budget isn't needed for this booking, and the date facts both pass the threshold." },
    { id: "h-summary", text: "The summarizer is broken because it dropped the new date.", isRootCause: false, evidenceIds: ["summary"], explanation: "Summaries are lossy by design. Storing facts precisely is long-term memory's job, and that's where the conflict goes unresolved." },
  ],
  fixes: [
    { id: "f-supersede", label: "When a new fact updates a slot, supersede older facts for that slot (newest wins at retrieval)", explanation: "Both dates are still retrieved, but the 19 May update replaces 12 May. The booking is correct at the 44-token window." },
    { id: "f-topk", label: "Retrieve only the single best memory (top_k = 1)", explanation: "The single best match is the old trip fact, so 12 May is still used." },
    { id: "f-threshold", label: "Raise the retrieval threshold to 0.5", explanation: "Both date facts score above 0.5, so the conflict and the ranking stay the same." },
    { id: "f-nolt", label: "Turn off long-term memory to stop stale facts coming back", explanation: "No stale date, but no date at all: the agent has to ask the user again." },
  ],
  verifications: [
    { id: "v-conflict", label: "Replay the date-change conversation at the 44-token window: the date is 19 May from long-term memory with 12 May marked superseded; add a test that a later fact for a slot always beats an older one, whatever their similarity", sound: true, explanation: "Tests the exact path that failed (window too small, conflict in long-term memory) and the rule that fixes it." },
    { id: "v-80", label: "Restore the 80-token window and confirm the booking date is right", sound: false, explanation: "It passes because the update fits in the window again. A slightly longer conversation brings the stale date back." },
    { id: "v-retrieved", label: "Confirm the update fact is retrieved from long-term memory", sound: false, explanation: "It was retrieved during the incident; it still lost the ranking." },
  ],
  rootCause: "Two settings combined. Cutting the window to 44 tokens pushed the “now arrive 19 May” turn out of short-term memory, so the date had to come from long-term memory. The long-term store keeps every fact with no update rule, and ranks by similarity alone, so the original 12 May fact (more similar to the question) beat the newer one.",
};

export const AGENT_MORE_SCENARIOS: Scenario[] = [toolDescScenario, doubleChargeScenario, earlyStopScenario, blackboardScenario, staleMemoryScenario];
