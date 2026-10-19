import { CORPUS, QUESTIONS, type SourceDoc } from "../visualizers/rag/corpus";
import { DEFAULT_RAG, answerQuestion, buildIndex, chunk, embed, parse, splitSentences, wordCount, type RagConfig, type RagIndex, type RagRun } from "../visualizers/rag/simulation";
import { DEFAULT_AGENT, EXPECTED_TOTAL, runAgent, type AgentConfig } from "../visualizers/agents/simulation";
import { agentScenario } from "./agent-scenarios";
import { churnData, dropFeature, evaluateModel, trainLogistic, type Dataset } from "./ml-model";
import { ML_ARCH, healthy, meanOf, pipelineSpans, toEvidence, weightsLines, type Act } from "./ml-scenarios";
import { RAG_ARCH } from "./rag-scenarios";
import type { Evidence, HealthMetric, Outcome, Scenario } from "./types";

const pct = (v: number) => `${Math.round(v * 100)}%`;
const ev = (id: string, action: string, lines: string[], relevant: boolean, spanId?: string): Evidence => ({ id, action, title: action, lines, relevant, spanId });
const q = (id: string) => QUESTIONS.find((x) => x.id === id)!;

// ---------- shared RAG plumbing for scenarios that change what the index holds or what search may see ----------
function indexOf(sources: SourceDoc[], cfg: RagConfig): RagIndex {
  const parsed = sources.map((d) => parse(d, cfg.tables));
  const chunks = parsed.flatMap((p) => chunk(p, cfg.chunkSize, cfg.overlap));
  return { parsed, chunks, sources, model: cfg.indexModel, builtAt: "2026-10-18", entries: chunks.map((c) => ({ ...c, vector: embed(c.text, cfg.indexModel), model: cfg.indexModel })) };
}

function ragOutcome(runs: RagRun[], questionId: string, evidence: (r: RagRun) => Evidence[]): Outcome {
  const r = runs.find((x) => x.question.id === questionId)!;
  const rate = (f: (x: RagRun) => boolean) => runs.filter(f).length / runs.length;
  const accuracy = rate((x) => x.evaluation.correct), faithfulness = rate((x) => x.evaluation.faithful);
  const lat = runs.map((x) => x.spans.find((s) => s.id === "rag.query")!.durationMs).sort((a, b) => a - b);
  return {
    spans: r.spans,
    metrics: [
      { label: "Success rate", value: "100%", status: "ok", kind: "health" },
      { label: "p95 latency", value: `${lat[lat.length - 1].toFixed(0)} ms`, status: "ok", kind: "health" },
      { label: "Golden-set accuracy", value: pct(accuracy), status: accuracy === 1 ? "ok" : "error", kind: "quality" },
      { label: "Faithfulness", value: pct(faithfulness), status: faithfulness === 1 ? "ok" : "error", kind: "quality" },
    ],
    transcript: [{ label: "User", text: r.question.text }, { label: "Assistant", text: r.answer.text, tone: r.evaluation.ok ? "ok" : "err" }],
    resolved: runs.every((x) => x.evaluation.ok),
    evidence: evidence(r),
  };
}

const goldenEv = (runs: RagRun[]) => ev("golden", "Run the golden-set evaluation", [`answer accuracy ${pct(runs.filter((x) => x.evaluation.correct).length / runs.length)}`, ...runs.map((x) => `${x.evaluation.ok ? "PASS" : "FAIL"} · ${x.question.text}`)], false);
const retrievalEv = (r: RagRun, file: (docId: string) => string) => ev("retrieval", "Inspect retrieval results", [`top-${r.cfg.topK} after rerank:`, ...r.reranked.map((h) => `#${h.rank} ${h.entry.id} (${file(h.entry.docId)}) cos=${h.score.toFixed(3)}`), `passages sent to the model: ${r.context.map((c) => `[${c.label}] ${c.entry.id}`).join(", ") || "none"}`], true, "retrieve");
const faithEv = (r: RagRun) => ev("faithfulness", "Run a faithfulness check", [r.answer.source === "none" ? "No answer was given, nothing to check." : r.evaluation.faithful ? "Answer sentence appears verbatim in the context: faithful." : "Answer sentence does NOT appear in the context: unfaithful."], false);
const dashEv = (runs: RagRun[]) => ev("dashboards", "Check latency and error dashboards", ["request success rate 100%, HTTP 5xx 0%", `p95 latency ${Math.max(...runs.map((x) => x.spans.find((s) => s.id === "rag.query")!.durationMs)).toFixed(0)} ms (SLO 1500 ms)`, "No alerts fired in the last 24 h."], false);

// ======================= RAG: a draft document got indexed =======================
const DRAFT: SourceDoc = {
  id: "pricing-draft", file: "drafts/pricing-2027.md", title: "Plans and pricing (DRAFT)", version: "draft", updatedAt: "2026-10-14",
  blocks: [{ type: "heading", text: "Plans and pricing draft" }, { type: "paragraph", text: "Proposal for next year: the Pro plan will cost $59 per month. Not yet approved." }],
};
const fileOf = (d: SourceDoc) => (d.id === DRAFT.id ? d.file : `published/${d.file}`);
const ALL_DOCS = [...CORPUS, DRAFT];
const DRAFT_FIXES: (Scenario["fixes"][number] & { drafts: boolean; cfg?: Partial<RagConfig> })[] = [
  { id: "f-scope", label: "Restrict ingestion to docs/published/** (or status: published) and re-index", explanation: "The draft leaves the index, so the only pricing chunk left is the real one.", drafts: false },
  { id: "f-topk", label: "Raise top-k to 8 and put 5 chunks in context", explanation: "The draft still ranks first, and the generator still answers from it.", drafts: true, cfg: { topK: 8, contextChunks: 5 } },
  { id: "f-reindex", label: "Rebuild the index from scratch", explanation: "The rebuild uses the same include pattern and picks the draft up again.", drafts: true },
];

function draftRun(fixId?: string): Outcome {
  const fix = DRAFT_FIXES.find((f) => f.id === fixId);
  const cfg: RagConfig = { ...DEFAULT_RAG, ...(fix?.cfg ?? {}) };
  const sources = fix?.drafts === false ? CORPUS : ALL_DOCS;
  const index = indexOf(sources, cfg);
  const runs = QUESTIONS.map((x) => answerQuestion(index, cfg, x));
  const file = (docId: string) => fileOf(ALL_DOCS.find((d) => d.id === docId)!);
  const pricing = CORPUS.find((d) => d.id === "pricing")!;
  return ragOutcome(runs, "q-price", (r) => [
    goldenEv(runs),
    ev("answer", "Compare the answer with the source of truth", [`Answer given: “${r.answer.text}”`, `Source of truth (${pricing.file} ${pricing.version}): “${splitSentences(parse(pricing, "linearize").text).find((s) => s.includes(r.question.expect))}”`], false),
    retrievalEv(r, file),
    ev("index", "List the documents in the index", sources.map((d) => `${fileOf(d)} ${d.version} (${d.updatedAt})${d.id === DRAFT.id ? "  ← not a published page" : ""}`), true),
    ev("ingest", "Read the ingestion job log", [`include pattern: ${fix?.drafts === false ? "docs/published/**" : "docs/**"}`, ...sources.map((d) => `ingest docs/${fileOf(d)} → parsed, chunked, embedded`), "job status: SUCCESS"], true),
    ev("parser", "Check parser output for pricing.pdf", [`pricing.pdf: ${index.parsed.find((p) => p.docId === "pricing")!.tablesFound} table found, 0 dropped`, `“${r.question.expect}” present in parsed text: ${index.parsed.find((p) => p.docId === "pricing")!.text.includes(r.question.expect) ? "yes" : "no"}`], false, "parse:pricing"),
    faithEv(r),
    dashEv(runs),
    ev("deploys", "Read the deploy log", ["Oct 14 product: drafts/pricing-2027.md added to the docs repo for review", "Oct 15 ingestion 2.5: include pattern docs/published/** → docs/** (to pick up the new docs/guides folder)", "Oct 15 nightly re-index: SUCCESS (5 → 6 documents)", "Oct 16 web-chat 4.6: dark mode"], true),
  ]);
}

// ======================= RAG: a default search filter hides older pages =======================
const CUTOFF = "2026-06-21";
const FILTER_FIXES: (Scenario["fixes"][number] & { filter: boolean; cfg?: Partial<RagConfig> })[] = [
  { id: "f-filter", label: "Remove the default updated_after filter (apply it only when the user asks for recent changes)", explanation: "Search sees every indexed chunk again, including the SSO guide.", filter: false },
  { id: "f-topk", label: "Raise top-k to 8 and put 5 chunks in context", explanation: "More results from a candidate set that still excludes the SSO guide.", filter: true, cfg: { topK: 8, contextChunks: 5 } },
  { id: "f-reindex", label: "Rebuild the index", explanation: "The SSO guide was already indexed; the rebuild keeps its March date and the filter still drops it.", filter: true },
];

function filterRun(fixId?: string): Outcome {
  const fix = FILTER_FIXES.find((f) => f.id === fixId);
  const cfg: RagConfig = { ...DEFAULT_RAG, ...(fix?.cfg ?? {}) };
  const full = buildIndex(cfg);
  const updated = (docId: string) => CORPUS.find((d) => d.id === docId)!.updatedAt;
  const filtering = fix?.filter !== false;
  const visible = filtering ? { ...full, entries: full.entries.filter((e) => updated(e.docId) >= CUTOFF) } : full;
  const runs = QUESTIONS.map((x) => answerQuestion(visible, cfg, x));
  const fact = "metadata XML";
  return ragOutcome(runs, "q-sso", (r) => [
    goldenEv(runs),
    retrievalEv(r, (id) => CORPUS.find((d) => d.id === id)!.file),
    ev("index", "List the documents in the index", CORPUS.map((d) => `${d.file} ${d.version}, updated ${d.updatedAt}: indexed (${full.chunks.filter((c) => c.docId === d.id).length} chunks)`), true),
    ev("search", `Search every chunk in the index for “${fact}”`, [`${full.chunks.length} chunks in the index`, ...full.chunks.filter((c) => c.text.includes(fact)).map((c) => `${c.id}: “…${c.text.slice(c.text.indexOf("uploads"), c.text.indexOf("uploads") + 60)}…”`)], true, "chunk"),
    ev("request", "Inspect the search request sent to the vector index", [`filter: ${filtering ? `{ updated_after: "${CUTOFF}" }  (default added by search-api 3.0)` : "none"}`, `candidates before filter: ${full.entries.length} chunks, after filter: ${visible.entries.length}`, ...CORPUS.filter((d) => filtering && d.updatedAt < CUTOFF).map((d) => `excluded by filter: ${d.file} (updated ${d.updatedAt})`)], true, "retrieve"),
    ev("models", "Compare query and index embedding models", [`query encoder: ${cfg.queryModel}`, `index vectors: ${full.model}`], false, "embed.query"),
    faithEv(r),
    dashEv(runs),
    ev("deploys", "Read the deploy log", ["Oct 14 search-api 3.0: new default filter updated_after = today − 120 days, so outdated pages stop showing up", "Oct 14 nightly re-index: SUCCESS (5 documents)", "Oct 13 docs: sso-setup.md unchanged since March", "Oct 12 web-chat 4.5: answer feedback buttons"], true),
  ]);
}

// ======================= Agent: iteration budget below the happy path =======================
const BUDGET_CFG: Partial<AgentConfig> = { task: "quote", maxIterations: 4 };
const budgetScenario = agentScenario({
  id: "agent-budget-too-small", title: "Quote agent stops one step before answering", difficulty: "Beginner",
  symptom: "“iteration budget of 4 exhausted”", objective: "Read a clean trace end to end and notice when the limit, not the agent, is the problem.",
  incident: "Since the agent-config migration on Thursday, every seat-quote request ends with “Stopped: iteration budget of 4 exhausted without an answer.” Sales says the agent “seems to be thinking correctly, then gives up”. No tool alerts fired.",
  cfg: BUDGET_CFG, faults: {},
  actions: ["final", "history", "failing", "budget", "memory", "latency", "retryPolicy", "deploys"],
  relevant: ["history", "budget", "memory", "deploys"],
  deploys: ["Oct 16 agent-config 2.0: settings migrated to the new schema; old max_tool_calls: 4 mapped onto max_iterations", "Oct 15 kb: pricing page copy edit", "Oct 14 llm-gateway: upgraded TLS certificates"],
  hypotheses: [
    { id: "h-budget", text: "max_iterations is 4, but a correct quote needs 5 steps (search, 2 reads, calculate, submit), so the run stops just before submitting.", isRootCause: true, evidenceIds: ["history", "budget", "memory", "happy"], explanation: "All 4 calls succeed, memory already holds the correct total, and the run ends on the iteration limit. The same task with a larger limit finishes in exactly 5 iterations." },
    { id: "h-tool", text: "A tool call is failing.", isRootCause: false, evidenceIds: ["failing", "history"], explanation: "No tool call failed; every step returned ok." },
    { id: "h-loop", text: "The agent is stuck in a loop repeating the same call.", isRootCause: false, evidenceIds: ["history"], explanation: "Each tool appears once, in a sensible order. Nothing repeats." },
    { id: "h-math", text: "The model is getting the arithmetic wrong.", isRootCause: false, evidenceIds: ["memory"], explanation: `Memory holds total = ${EXPECTED_TOTAL}, the correct value. The agent never got the chance to submit it.` },
    { id: "h-slow", text: "Tools are slow and the request times out.", isRootCause: false, evidenceIds: ["latency"], explanation: "Every tool answers far within the 2 s timeout. The run ends on step count, not time." },
  ],
  fixes: [
    { id: "f-budget", label: "Set max_iterations back to 8 (headroom above the 5-step happy path)", explanation: "The agent submits the total on iteration 5.", cfg: { maxIterations: 8 } },
    { id: "f-retries", label: "Raise max retries to 4", explanation: "Nothing failed, so nothing is retried; the run still stops after 4 iterations.", cfg: { maxRetries: 4 } },
    { id: "f-timeout", label: "Raise the tool timeout to 5 s", explanation: "No call came close to timing out; the limit is on steps.", cfg: { timeoutMs: 5000 } },
  ],
  verifications: [
    { id: "v-replay", label: "Replay the quote task: it ends with submit_result($5,997.6); add a config test that max_iterations leaves headroom above every task's happy path", sound: true, explanation: "Checks the outcome and stops a future migration from silently shrinking the budget again." },
    { id: "v-errors", label: "Confirm the tool error rate is 0%", sound: false, explanation: "It was 0% during the incident. Nothing was erroring." },
    { id: "v-calc", label: "Unit-test the calculator with 12 * 49 * 12 * 0.85", sound: false, explanation: "The calculator was already right; it just wasn't the last step." },
  ],
  rootCause: "A config migration mapped the old max_tool_calls: 4 onto max_iterations. A correct quote needs 5 iterations, so every run computes the right total on step 4 and is stopped before it can submit it.",
});
const budgetWithHappy: Scenario = {
  ...budgetScenario,
  relatedTopic: "agent-termination",
  run: (fixId) => {
    const o = budgetScenario.run(fixId);
    const happy = runAgent({ ...DEFAULT_AGENT, ...BUDGET_CFG, maxIterations: 20 });
    return { ...o, evidence: [...o.evidence, ev("happy", "Replay the same task with a 20-iteration limit", [`finished in ${happy.iterations} iterations: ${happy.toolCalls.map((c) => c.tool).join(" → ")}`, `answer: “${happy.final.text}”`], true)] };
  },
};

// ======================= ML: label encoding flipped =======================
const DROP = "cancel_ticket_logged";
const flip = (ds: Dataset): Dataset => ({ ...ds, rows: ds.rows.map((r) => ({ ...r, y: 1 - r.y })) });
const churnRate = (ds: Dataset) => ds.rows.filter((r) => r.y).length / ds.rows.length;

function labelsRun(fixId?: string): Outcome {
  const prepare = (ds: Dataset) => (fixId === "f-labels" ? ds : flip(ds)); // the training export's label step
  const raw = dropFeature(churnData(fixId === "f-more" ? 9 : 1, fixId === "f-more" ? 3000 : 800), DROP);
  const tr = prepare(raw), te = prepare(dropFeature(churnData(2, 400), DROP));
  const pr = dropFeature(churnData(3, 400, true), DROP); // real outcomes, 30 days later
  const m = trainLogistic(tr, { epochs: fixId === "f-epochs" ? 1600 : 400 });
  const prev = trainLogistic(dropFeature(churnData(1, 800), DROP));
  const thr = fixId === "f-threshold" ? 0.3 : 0.5;
  const off = evaluateModel(m, te.rows, thr), on = evaluateModel(m, pr.rows, thr);
  const resolved = on.accuracy >= 0.7 && Math.abs(off.accuracy - on.accuracy) < 0.08;
  const p1 = (x: number) => `${(x * 100).toFixed(1)}%`;
  const acts: Act[] = [
    { id: "offline", label: "Review offline evaluation", lines: [`held-out accuracy ${p1(off.accuracy)} (test labels come from the same export as training)`] },
    { id: "online", label: "Review online evaluation (real outcomes after 30 days)", lines: [`online accuracy ${p1(on.accuracy)}; flagged ${on.tp + on.fp} of ${on.n} customers, ${on.tp} of them actually churned`, `if every prediction were inverted, online accuracy would be ${p1(1 - on.accuracy)}`, `always predicting “stays” would score ${p1(1 - on.positives / on.n)}`] },
    { id: "labels", label: "Check the label distribution: training export vs CRM", lines: [`training export: ${p1(churnRate(tr))} of customers labelled churned (is_churned = 1)`, `CRM report for the same customers: ${p1(churnRate(raw))} churned`, `export label values: ${fixId === "f-labels" ? "{active: 0, churned: 1} (explicit map)" : "{churned: 0, retained: 1} (sorted category codes)"}`] },
    { id: "weights", label: "Compare learned weights with the previous model (churn-v6)", lines: m.features.map((f, j) => `${f}: churn-v6 ${prev.weights[j] >= 0 ? "+" : ""}${prev.weights[j].toFixed(2)} → new ${m.weights[j] >= 0 ? "+" : ""}${m.weights[j].toFixed(2)}${Math.sign(prev.weights[j]) !== Math.sign(m.weights[j]) ? "  ← sign flipped" : ""}`) },
    { id: "dist", label: "Compare feature distributions: training vs production", lines: tr.features.map((f, j) => `${f}: train mean ${meanOf(tr, j).toFixed(1)} · prod mean ${meanOf(pr, j).toFixed(1)}`) },
    { id: "thresholds", label: "Sweep the decision threshold on production outcomes", lines: [0.3, 0.5, 0.7].map((t) => `threshold ${t}: online accuracy ${p1(evaluateModel(m, pr.rows, t).accuracy)}`) },
    { id: "serving", label: "Check serving payload and feature order", lines: [`request → [${tr.features.join(", ")}] in training order ✓`, "schema hash matches training", ...weightsLines(m).slice(0, 1).map((l) => `largest weight: ${l}`)] },
    { id: "deploys", label: "Read the change log", lines: ["Oct 02 data-eng: label export rewritten; status {retained, churned} now encoded with sorted category codes", "Oct 04 ml: churn-v7 trained on the new export; offline accuracy 77% (same as v6) → shipped", "Oct 10 crm: retention offers sent to customers with churn score > 0.5"] },
  ];
  return {
    spans: pipelineSpans({ trainRows: tr.rows.length, features: tr.features, offlineAcc: off.accuracy, onlineAcc: on.accuracy, servingOrder: tr.features, prodRows: pr.rows.length, version: "churn-v7" }),
    metrics: [...healthy, { label: "Offline accuracy", value: p1(off.accuracy), status: "ok", kind: "quality" }, { label: "Online accuracy", value: p1(on.accuracy), status: resolved ? "ok" : "error", kind: "quality" }],
    transcript: [{ label: "Retention team", text: `${on.tp + on.fp} customers got retention offers; ${on.tp} of them actually churned. ${on.fn} churners got nothing.`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: toEvidence(acts, ["online", "labels", "weights", "deploys"]),
  };
}

// ======================= Production: gateway output cap applied to every route =======================
const CAP = 12;
const CAP_FIXES: (Scenario["fixes"][number] & { cap: number; cfg?: Partial<RagConfig> })[] = [
  { id: "f-route", label: `Scope max_tokens: ${CAP} to /title-suggest only; /ask goes back to 512`, explanation: "Answers finish with finish_reason=stop again.", cap: 512 },
  { id: "f-provider", label: "Fail over to the secondary LLM provider", explanation: "The cap is set in our gateway, so the other provider gets the same max_tokens.", cap: CAP },
  { id: "f-budget", label: "Raise the context budget to 400 tokens", explanation: "The context already holds the whole sentence; the cut happens on the way out.", cap: CAP, cfg: { contextBudget: 400 } },
];

function capRun(fixId?: string): Outcome {
  const fix = CAP_FIXES.find((f) => f.id === fixId);
  const cfg: RagConfig = { ...DEFAULT_RAG, ...(fix?.cfg ?? {}) };
  const cap = fix?.cap ?? CAP;
  const index = buildIndex(cfg);
  const runs = QUESTIONS.map((x) => {
    const r = answerQuestion(index, cfg, x);
    const words = r.answer.text.split(/\s+/);
    const cut = words.length > cap;
    const text = words.slice(0, cap).join(" ");
    return { r, text, cut, outTokens: Math.min(cap, words.length), ok: r.evaluation.ok && !cut, correct: text.toLowerCase().includes(x.expect.toLowerCase()) };
  });
  const main = runs.find((x) => x.r.question.id === "q-retention")!;
  const accuracy = runs.filter((x) => x.correct).length / runs.length;
  const cutRate = runs.filter((x) => x.cut).length / runs.length;
  const spans = main.r.spans.map((s) => (s.id === "generate" ? { ...s, status: main.cut ? ("warning" as const) : s.status, input: { ...s.input, max_tokens: cap }, output: { ...s.output, answer: main.text }, events: [{ id: "cap", type: "log", message: `finish_reason=${main.cut ? "length" : "stop"}, output_tokens=${main.outTokens}`, severity: main.cut ? ("warning" as const) : ("info" as const) }] } : s));
  const ctx = main.r.context.find((c) => c.text.includes(main.r.question.expect));
  const metrics: HealthMetric[] = [
    { label: "Success rate", value: "100%", status: "ok", kind: "health" },
    { label: "Error rate", value: "0%", status: "ok", kind: "health" },
    { label: "Golden-set accuracy", value: pct(accuracy), status: accuracy === 1 ? "ok" : "error", kind: "quality" },
    { label: "finish_reason=length", value: pct(cutRate), status: cutRate === 0 ? "ok" : "error", kind: "quality" },
  ];
  return {
    spans, metrics,
    transcript: [{ label: "User", text: main.r.question.text }, { label: "Assistant", text: main.text, tone: main.ok ? "ok" : "err" }],
    resolved: runs.every((x) => x.ok),
    evidence: [
      ev("golden", "Run the golden-set evaluation", [`answer accuracy ${pct(accuracy)}`, ...runs.map((x) => `${x.ok ? "PASS" : "FAIL"} · ${x.r.question.text}`)], false),
      ev("answer", "Compare the answer with the source of truth", [`Answer given: “${main.text}”`, `Source of truth: “${splitSentences(parse(CORPUS.find((d) => d.id === "retention")!, "linearize").text).find((s) => s.includes(main.r.question.expect))}”`], false),
      ev("context", "Inspect the assembled context", [`${main.r.context.length} passage(s), ${main.r.contextTokens}/${cfg.contextBudget} tokens, nothing truncated`, ctx ? `[${ctx.label}] ${ctx.entry.id} contains the full sentence including “${main.r.question.expect}”` : "The reference fact is NOT in the context."], true, "context"),
      ev("generate", "Inspect the LLM call parameters and finish_reason", [`request: model sim-gen-1, max_tokens ${cap}`, `response: output_tokens ${main.outTokens}, finish_reason ${main.cut ? "length (the limit was hit)" : "stop"}`], true, "generate"),
      ev("lengths", "Compare output length across the golden set", runs.map((x) => `${x.outTokens} tokens · finish_reason=${x.cut ? "length" : "stop"} · ${x.r.question.text}`), true),
      ev("chunks", "Inspect chunk boundaries in data-retention.md", index.chunks.filter((c) => c.docId === "retention").map((c) => `${c.id} [${c.startWord}–${c.endWord}] (${wordCount(c.text)} words): “${c.text.slice(0, 90)}…”`), false, "chunk"),
      faithEv(main.r),
      ev("provider", "Check the LLM provider's status page and latency", ["status page: all systems operational", "provider-side p95 latency unchanged; no truncation errors reported"], false),
      ev("deploys", "Read the deploy log", [`Oct 17 llm-gateway 1.14: max_tokens: ${CAP} added for the new /title-suggest route (config key set at gateway level)`, "Oct 17 web-chat 4.7: conversation titles", "Oct 16 nightly re-index: SUCCESS", "Oct 15 reranker 2.2: tie-breaking by rank"], true),
    ],
  };
}

export const BEGINNER_SCENARIOS: Scenario[] = [
  {
    id: "rag-draft-indexed", title: "Assistant quotes a price nobody approved", system: "RAG", difficulty: "Beginner", relatedTopic: "document-ingestion",
    symptom: "Says Pro costs $59; the pricing page says $49", objective: "Check where a confident, well-cited answer came from before blaming the model.",
    incident: `Since Wednesday, customers asking “${q("q-price").text}” are told $59 per month, with a citation. The pricing page says $49 and hasn't changed since September. Sales is fielding confused emails.`,
    architecture: RAG_ARCH, run: draftRun,
    hypotheses: [
      { id: "h-draft", text: "An unpublished draft price page was indexed, and it outranks the real pricing page.", isRootCause: true, evidenceIds: ["index", "retrieval", "ingest", "deploys"], explanation: "The index holds drafts/pricing-2027.md, its chunk ranks #1 for the price question, and the ingestion include pattern was widened to docs/** the day it appeared." },
      { id: "h-halluc", text: "The LLM is hallucinating the $59 price.", isRootCause: false, evidenceIds: ["faithfulness", "retrieval"], explanation: "The answer sentence appears word for word in the top passage. The model faithfully repeated what it was given." },
      { id: "h-topk", text: "top-k is too small, so the real pricing chunk isn't retrieved.", isRootCause: false, evidenceIds: ["retrieval"], explanation: "The real pricing chunk is retrieved and even reaches the context. It just ranks below the draft." },
      { id: "h-parser", text: "The parser dropped the pricing table again.", isRootCause: false, evidenceIds: ["parser"], explanation: "pricing.pdf parsed with its table intact; $49 is in the index." },
    ],
    fixes: DRAFT_FIXES.map(({ id, label, explanation }) => ({ id, label, explanation })),
    verifications: [
      { id: "v-published", label: "Re-run the golden set and add an ingestion check that every indexed file is a published page", sound: true, explanation: "Confirms the answer and blocks the next draft or scratch file from reaching users." },
      { id: "v-49", label: "Confirm a chunk containing “$49” is in the index", sound: false, explanation: "It was there all along; the problem was the extra chunk competing with it." },
      { id: "v-once", label: "Ask the price question once in staging", sound: false, explanation: "One question doesn't show which other drafts are in the index." },
    ],
    rootCause: "Ingestion's include pattern was widened from docs/published/** to docs/**, which also pulled in drafts/pricing-2027.md. The draft's short, on-topic chunk outranks the real pricing table, so the assistant faithfully quotes an unapproved $59.",
  },
  {
    id: "rag-recency-filter", title: "Assistant can't find the SSO guide anymore", system: "RAG", difficulty: "Beginner", relatedTopic: "rag-pipeline",
    symptom: "“I couldn't find this in the provided documents.”", objective: "Tell “not in the index” apart from “in the index but hidden from search”.",
    incident: `Since Tuesday, admins asking “${q("q-sso").text}” get “I couldn't find this in the provided documents.” The SSO guide is still on the docs site and the nightly re-index lists it as indexed.`,
    architecture: RAG_ARCH, run: filterRun,
    hypotheses: [
      { id: "h-filter", text: "A new default search filter (updated_after = 120 days) hides pages that haven't been edited recently, including the SSO guide.", isRootCause: true, evidenceIds: ["request", "retrieval", "index", "deploys"], explanation: "The SSO guide is indexed, but the search request carries updated_after 2026-06-21 and sso-setup.md was last edited in March, so its chunks are removed before ranking." },
      { id: "h-missing", text: "The SSO guide was dropped from the index.", isRootCause: false, evidenceIds: ["index", "search"], explanation: "The index lists sso-setup.md and a chunk with “metadata XML” exists." },
      { id: "h-topk", text: "top-k is too small.", isRootCause: false, evidenceIds: ["request", "retrieval"], explanation: "The SSO chunks are filtered out before ranking. No k can retrieve them." },
      { id: "h-llm", text: "The model is being overly cautious and refusing to answer.", isRootCause: false, evidenceIds: ["retrieval", "faithfulness"], explanation: "None of the passages it received mention SSO. Saying “I couldn't find this” is the correct behaviour for that context." },
      { id: "h-embed", text: "The query and index embedding models don't match.", isRootCause: false, evidenceIds: ["models", "golden"], explanation: "Both are embed-v1, and questions about recently edited pages still pass." },
    ],
    fixes: FILTER_FIXES.map(({ id, label, explanation }) => ({ id, label, explanation })),
    verifications: [
      { id: "v-reach", label: "Re-run the golden set and add a test that each golden question's source chunk is reachable with the default search request", sound: true, explanation: "Tests what the user actually sees, not just what the index holds." },
      { id: "v-indexed", label: "Confirm sso-setup.md is listed in the index", sound: false, explanation: "It was listed throughout the incident." },
      { id: "v-200", label: "Confirm the search API returns HTTP 200", sound: false, explanation: "Filtered searches succeed; they just return the wrong candidates." },
    ],
    rootCause: "search-api 3.0 added a default filter that only searches pages updated in the last 120 days. Stable pages like the March SSO guide are still indexed, but every search silently excludes them, so the assistant has nothing relevant to answer from.",
  },
  budgetWithHappy,
  {
    id: "ml-flipped-labels", title: "Churn model sends offers to loyal customers", system: "ML", difficulty: "Beginner", relatedTopic: "logistic-regression",
    symptom: "Online accuracy far below a coin flip", objective: "Recognize a model that learned the opposite of the truth, and trace it to the labels.",
    incident: "The new churn model scored the same offline accuracy as the old one, so it shipped. A month later, retention offers went mostly to customers who stayed, and most customers who left never got one.",
    architecture: ML_ARCH, run: labelsRun,
    hypotheses: [
      { id: "h-flip", text: "The label export swapped churned and retained, so the model learned the opposite of churn.", isRootCause: true, evidenceIds: ["labels", "weights", "online", "deploys"], explanation: "The export labels about two thirds of customers as churned while the CRM says about one third, every weight has the opposite sign from churn-v6, and inverting the predictions would make them good." },
      { id: "h-drift", text: "Customer behaviour drifted after launch.", isRootCause: false, evidenceIds: ["dist"], explanation: "Feature distributions in production match training." },
      { id: "h-threshold", text: "The decision threshold is wrong.", isRootCause: false, evidenceIds: ["thresholds"], explanation: "No threshold helps: the scores point the wrong way, not just too high or too low." },
      { id: "h-overfit", text: "The model is overfitting and needs more data.", isRootCause: false, evidenceIds: ["offline"], explanation: "Held-out accuracy is 77%, the same as churn-v6. The held-out labels went through the same flipped export, so they can't reveal it." },
      { id: "h-serving", text: "Serving sends features in the wrong order.", isRootCause: false, evidenceIds: ["serving"], explanation: "Feature order and schema match training." },
    ],
    fixes: [
      { id: "f-labels", label: "Encode labels with an explicit map {active: 0, churned: 1} and retrain", explanation: "The model learns churn again; online accuracy matches offline." },
      { id: "f-more", label: "Retrain on 3,000 more customers from the same export", explanation: "More data with the same flipped labels teaches the same wrong lesson." },
      { id: "f-epochs", label: "Train 4× longer", explanation: "It converges more precisely to the flipped relationship." },
      { id: "f-threshold", label: "Lower the decision threshold to 0.3", explanation: "Flags even more of the customers who are likely to stay." },
    ],
    verifications: [
      { id: "v-sanity", label: "Assert the training churn rate matches the CRM within a few points and evaluate against real outcomes, not the export", sound: true, explanation: "Catches a flipped or broken label pipeline before training and checks the model against reality." },
      { id: "v-offline", label: "Confirm offline accuracy matches churn-v6", sound: false, explanation: "It did, which is how the bug shipped: the test set had the same flipped labels." },
      { id: "v-uptime", label: "Confirm the model server is healthy", sound: false, explanation: "It always was." },
    ],
    rootCause: "The rewritten label export encoded status with sorted category codes, so churned became 0 and retained became 1. The model learned to predict “stays” as churn. Offline tests used the same flipped labels and looked fine; real outcomes show the model is backwards.",
  },
  {
    id: "prod-max-tokens", title: "Long answers stop after twelve words", system: "Production", difficulty: "Beginner", relatedTopic: "model-serving",
    symptom: "Answers end mid-sentence and lose their citation", objective: "Find where an answer gets cut: before the model (context) or after it (output limit).",
    incident: `Since the conversation-titles release, longer answers stop mid-sentence. Asked “${q("q-retention").text}”, the assistant says “Request logs are retained for 30 days on Starter and Pro plans” and stops. Short answers look fine. No errors, and latency even went down.`,
    architecture: RAG_ARCH, run: capRun,
    hypotheses: [
      { id: "h-cap", text: `A gateway-wide max_tokens: ${CAP}, meant for the new title route, also caps /ask answers.`, isRootCause: true, evidenceIds: ["generate", "lengths", "context", "deploys"], explanation: `The context holds the full sentence; the LLM call is sent with max_tokens ${CAP} and every failing answer is exactly ${CAP} tokens with finish_reason=length.` },
      { id: "h-chunk", text: "Chunking splits the sentence, like the SSO incident.", isRootCause: false, evidenceIds: ["chunks", "context"], explanation: "The retention sentence sits whole inside one chunk, and that chunk is in the context." },
      { id: "h-halluc", text: "The model is hallucinating “30 days”.", isRootCause: false, evidenceIds: ["faithfulness", "context"], explanation: "The words are the start of a real context sentence. The 365-day part was cut off, not invented away." },
      { id: "h-provider", text: "The LLM provider is degraded and truncating responses.", isRootCause: false, evidenceIds: ["provider", "generate"], explanation: "The provider is healthy. finish_reason=length means our own request asked for at most 12 tokens." },
    ],
    fixes: CAP_FIXES.map(({ id, label, explanation }) => ({ id, label, explanation })),
    verifications: [
      { id: "v-length", label: "Re-run the golden set, alert when finish_reason=length exceeds 1% on /ask, and test generation settings per route", sound: true, explanation: "Confirms the fix and makes the next silent truncation loud." },
      { id: "v-errors", label: "Confirm the error rate is 0%", sound: false, explanation: "It was 0% throughout; truncated answers are successful responses." },
      { id: "v-latency", label: "Confirm latency is lower", sound: false, explanation: "Latency dropped because answers got shorter. That's the symptom, not a fix." },
    ],
    rootCause: `llm-gateway 1.14 added max_tokens: ${CAP} for the new title-suggestion route, but the key was set at gateway level and applies to /ask too. Any answer longer than ${CAP} tokens is cut off with finish_reason=length, dropping the end of the sentence and its citation.`,
  },
];
