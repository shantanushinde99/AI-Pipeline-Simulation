import { dot } from "../lib/math";
import type { TraceSpan } from "../types/simulation";
import { CORPUS, QUESTIONS, type Question, type SourceDoc } from "../visualizers/rag/corpus";
import { DEFAULT_RAG, answerQuestion, buildIndex, chunk, embed, parse, splitSentences, wordCount, type RagConfig, type RagIndex, type RagRun } from "../visualizers/rag/simulation";
import { RAG_ARCH } from "./rag-scenarios";
import type { Evidence, HealthMetric, Outcome, Scenario } from "./types";

const pct = (v: number) => `${Math.round(v * 100)}%`;
const ev = (id: string, action: string, lines: string[], relevant: boolean, spanId?: string): Evidence => ({ id, action, title: action, lines, relevant, spanId });
const q = (id: string) => QUESTIONS.find((x) => x.id === id)!;
const share = (xs: boolean[]) => xs.filter(Boolean).length / xs.length;
const latency = (r: RagRun) => r.spans.find((s) => s.id === "rag.query")!.durationMs;
const quality = (label: string, v: number): HealthMetric => ({ label, value: pct(v), status: v === 1 ? "ok" : "error", kind: "quality" });
const health = (spans: number[], extra: HealthMetric[] = []): HealthMetric[] => [
  { label: "Success rate", value: "100%", status: "ok", kind: "health" },
  { label: "p95 latency", value: `${Math.max(...spans).toFixed(0)} ms`, status: Math.max(...spans) < 1500 ? "ok" : "warning", kind: "health" },
  ...extra,
];
const truth = (questionId: string) => {
  const qq = q(questionId);
  const src = CORPUS.find((d) => d.id === qq.sourceDoc)!;
  return `Source of truth (${src.file} ${src.version}): “${splitSentences(parse(src, "linearize").text).find((s) => s.includes(qq.expect))}”`;
};
const faithEv = (r: RagRun) => ev("faithfulness", "Run a faithfulness check", [r.answer.source === "none" ? "No answer was given, nothing to check." : r.evaluation.faithful ? "Answer sentence appears verbatim in the context: faithful." : "Answer sentence does NOT appear in the context: unfaithful."], false);
const modelsEv = (r: RagRun) => ev("models", "Compare query and index embedding models", [`query encoder: ${r.cfg.queryModel}`, `index vectors: ${r.index.model}`], false, "embed.query");
const retrievalEv = (r: RagRun, relevant: boolean, note?: string) => ev("retrieval", "Inspect retrieval results", [`top-${r.cfg.topK} after rerank:`, ...r.reranked.map((h) => `#${h.rank} ${h.entry.id} cos=${h.score.toFixed(3)}`), `passages sent to the model: ${r.context.map((c) => `[${c.label}] ${c.entry.id}`).join(", ") || "none"}`, ...(note ? [note] : [])], relevant, "retrieve");
const searchEv = (index: RagIndex, fact: string) => ev("search", `Search every chunk in the index for “${fact}”`, [`${index.chunks.length} chunks in the index`, `${index.chunks.filter((c) => c.text.includes(fact)).length} chunk(s) contain “${fact}”: ${index.chunks.filter((c) => c.text.includes(fact)).map((c) => c.id).join(", ") || "none"}`], false, "chunk");
const dashEv = (lat: number[], extra: string[] = []) => ev("dashboards", "Check latency and error dashboards", ["request success rate 100%, HTTP 5xx 0%", `p95 latency ${Math.max(...lat).toFixed(0)} ms (SLO 1500 ms)`, ...extra, "No alerts fired in the last 24 h."], false);
const goldenEv = (label: string, rows: { text: string; ok: boolean }[], relevant = false) => ev("golden", label, [`answer accuracy ${pct(share(rows.map((x) => x.ok)))}`, ...rows.map((x) => `${x.ok ? "PASS" : "FAIL"} · ${x.text}`)], relevant);
const chat = (asked: string, answer: string, ok: boolean): Outcome["transcript"] => [{ label: "User", text: asked }, { label: "Assistant", text: answer, tone: ok ? "ok" : "err" }];

/** Same parse → chunk → embed flow as buildIndex, over whatever sources the scenario's ingestion produced. */
function indexOf(sources: SourceDoc[], cfg: RagConfig): RagIndex {
  const parsed = sources.map((d) => parse(d, cfg.tables));
  const chunks = parsed.flatMap((p) => chunk(p, cfg.chunkSize, cfg.overlap));
  return { parsed, chunks, sources, model: cfg.indexModel, builtAt: "2026-10-18", entries: chunks.map((c) => ({ ...c, vector: embed(c.text, cfg.indexModel), model: cfg.indexModel })) };
}

type Fix<T> = Scenario["fixes"][number] & T;
const publicFixes = <T>(fixes: Fix<T>[]) => fixes.map(({ id, label, explanation }) => ({ id, label, explanation }));

// ======================= Beginner: website nav menu and footer crawled into every page =======================
const NAV = "Northwind Cloud docs. Home Pricing Plans Refunds SSO Security API Rate limits Data retention Support.";
const FOOTER = "Contact the support team about your plan, pricing, refunds, billing, API keys or security. Copyright 2026 Northwind Cloud.";
const crawled = (d: SourceDoc): SourceDoc => ({ ...d, blocks: [{ type: "paragraph", text: NAV }, ...d.blocks, { type: "paragraph", text: FOOTER }] });
const NAV_FIXES: Fix<{ strip: boolean; cfg?: Partial<RagConfig> }>[] = [
  { id: "f-main", label: "Extract only the page's <main> content (drop <nav> and <footer>) and re-index", explanation: "Chunks contain only the page's own text again, so the real rate-limit sentences are the best matches.", strip: true },
  { id: "f-topk", label: "Raise top-k to 8 and put 5 chunks in context", explanation: "Every extra chunk carries the same menu line, so the menu still wins answer extraction.", strip: false, cfg: { topK: 8, contextChunks: 5 } },
  { id: "f-rerank", label: "Turn the reranker off", explanation: "The right page is already ranked #1; the menu is inside that page's chunk.", strip: false, cfg: { rerank: false } },
  { id: "f-chunk", label: "Raise chunk size to 200 words so each page is one chunk", explanation: "A bigger chunk still starts with the menu and ends with the footer.", strip: false, cfg: { chunkSize: 200 } },
];

function navRun(fixId?: string): Outcome {
  const fix = NAV_FIXES.find((f) => f.id === fixId);
  const cfg: RagConfig = { ...DEFAULT_RAG, ...(fix?.cfg ?? {}) };
  const sources = fix?.strip ? CORPUS : CORPUS.map(crawled);
  const index = indexOf(sources, cfg);
  const runs = QUESTIONS.map((x) => answerQuestion(index, cfg, x));
  const r = runs.find((x) => x.question.id === "q-rate")!;
  const top = r.context[0];
  const boiler = wordCount(NAV) + wordCount(FOOTER);
  const withMenu = index.chunks.filter((c) => c.text.includes("Home Pricing Plans")).length;
  const answerFrom = r.context.find((c) => c.label === r.answer.citations[0]?.label);
  return {
    spans: r.spans,
    metrics: health(runs.map(latency), [{ label: "Crawl status", value: `${sources.length}/${sources.length} pages 200 OK`, status: "ok", kind: "health" }, quality("Golden-set accuracy", share(runs.map((x) => x.evaluation.ok))), quality("Faithfulness", share(runs.map((x) => x.evaluation.faithful)))]),
    transcript: chat(r.question.text, r.answer.text, r.evaluation.ok),
    resolved: runs.every((x) => x.evaluation.ok),
    evidence: [
      goldenEv("Run the golden-set evaluation", runs.map((x) => ({ text: x.question.text, ok: x.evaluation.ok }))),
      ev("answer", "Compare the answer with the source of truth", [`Answer given: “${r.answer.text}”`, truth("q-rate"), answerFrom ? `The answer sentence starts at word ${answerFrom.text.split(/\s+/).indexOf(r.answer.sentence.split(" ")[0])} of passage [${answerFrom.label}] (${answerFrom.entry.id}).` : "No passage was cited."], false),
      retrievalEv(r, false, `${r.question.sourceDoc} chunk ranks #${r.reranked.findIndex((h) => h.entry.docId === r.question.sourceDoc) + 1}.`),
      ev("chunks", `Read the top passage (${top?.entry.id ?? "none"}) as the model saw it`, top ? [`“${top.text}”`] : ["No passage reached the model."], true, "context"),
      ev("search", "Search every chunk in the index for the menu line “Home Pricing Plans”", [`${withMenu} of ${index.chunks.length} chunks contain it`, `the footer “Copyright 2026 Northwind Cloud” appears in ${index.chunks.filter((c) => c.text.includes("Copyright 2026")).length} chunks`], true, "chunk"),
      ev("parser", "Compare parsed word counts with the source pages", index.parsed.map((p) => { const src = CORPUS.find((d) => d.id === p.docId)!; const own = wordCount(parse(src, "linearize").text); return `${p.file}: source ${own} words, parsed ${wordCount(p.text)} words${wordCount(p.text) > own ? ` (+${wordCount(p.text) - own}, ${pct((wordCount(p.text) - own) / wordCount(p.text))} not from the page body)` : ""}`; }), true, `parse:${r.question.sourceDoc}`),
      faithEv(r),
      modelsEv(r),
      dashEv(runs.map(latency), [`crawler: ${sources.length} pages fetched, 0 errors, ${boiler} words of template per page`]),
      ev("deploys", "Read the deploy log", ["Oct 13 docs: site moved to the new docs.northwind.cloud theme (shared header menu and footer)", `Oct 14 ingestion 3.0: source switched from the markdown repo to crawling docs.northwind.cloud (whole <body> extracted)${fix?.strip ? " → patched to <main> only" : ""}`, "Oct 14 nightly re-index: SUCCESS (5 pages)", "Oct 15 web-chat 4.6: dark mode"], true),
    ],
  };
}

// ======================= Beginner: a PII redactor rewrites the question before search =======================
/** The privacy middleware's rule: a capitalized word that doesn't start the sentence might be a person's name. */
const redact = (text: string, allow: Set<string>) => text.replace(/(?<=\S\s+)[A-Z][A-Za-z0-9]*/g, (w) => (allow.has(w) ? w : "[REDACTED]"));
const REDACT_FIXES: Fix<{ redactQuery: boolean; allow?: string[]; cfg?: Partial<RagConfig> }>[] = [
  { id: "f-logs", label: "Redact only the copy written to the request logs; send the user's original question to search and the model", explanation: "Search and answer extraction see “Pro”, “SSO” and “API” again.", redactQuery: false },
  { id: "f-allow", label: "Add the plan names (Starter, Pro, Enterprise) to the redactor's allowlist", explanation: "Fixes the plan questions, but “SSO” and “API” are still redacted, so the SSO question still finds nothing.", redactQuery: true, allow: ["Starter", "Pro", "Enterprise"] },
  { id: "f-topk", label: "Raise top-k to 8 and put 5 chunks in context", explanation: "More passages, but the question still has no product words for the answer step to match.", redactQuery: true, cfg: { topK: 8, contextChunks: 5 } },
  { id: "f-reindex", label: "Rebuild the index", explanation: "The index was fine; the question is what changed.", redactQuery: true },
];

function redactRun(fixId?: string): Outcome {
  const fix = REDACT_FIXES.find((f) => f.id === fixId);
  const cfg: RagConfig = { ...DEFAULT_RAG, ...(fix?.cfg ?? {}) };
  const allow = new Set(fix?.allow ?? []);
  const sent = (x: Question) => (fix?.redactQuery === false ? x.text : redact(x.text, allow));
  const index = buildIndex(cfg);
  const runs = QUESTIONS.map((x) => answerQuestion(index, cfg, { ...x, text: sent(x) }));
  const r = runs.find((x) => x.question.id === "q-price")!;
  const user = q("q-price").text;
  return {
    spans: r.spans,
    metrics: health(runs.map(latency), [{ label: "PII redaction", value: "active", status: "ok", kind: "health" }, quality("Golden-set accuracy", share(runs.map((x) => x.evaluation.ok))), quality("Faithfulness", share(runs.map((x) => x.evaluation.faithful)))]),
    transcript: chat(user, r.answer.text, r.evaluation.ok),
    resolved: runs.every((x) => x.evaluation.ok),
    evidence: [
      goldenEv("Run the golden-set evaluation", QUESTIONS.map((x, i) => ({ text: x.text, ok: runs[i].evaluation.ok }))),
      ev("answer", "Compare the answer with the source of truth", [`Answer given: “${r.answer.text}”`, truth("q-price")], false),
      ev("query", "Inspect the query the embedder received", QUESTIONS.map((x, i) => `user typed “${x.text}” → search got “${runs[i].question.text}” (terms: ${runs[i].qTokens.join(" ")})`), true, "embed.query"),
      retrievalEv(r, false, `pricing.pdf ranks #${r.reranked.findIndex((h) => h.entry.docId === "pricing") + 1}; the answer step picked “${r.answer.sentence || "nothing"}”.`),
      searchEv(index, "$49"),
      ev("redactor", "Read the privacy middleware log", QUESTIONS.flatMap((x, i) => (runs[i].question.text === x.text ? [] : [`${x.id}: replaced ${x.text.split(" ").filter((w, j) => j > 0 && /^[A-Z]/.test(w) && !allow.has(w.replace(/\W/g, ""))).map((w) => `“${w.replace(/\W/g, "")}”`).join(", ")} (rule: capitalized word mid-sentence = possible person name)`])).concat(fix?.redactQuery === false ? ["redaction applied to log records only"] : []), true),
      faithEv(r),
      modelsEv(r),
      dashEv(runs.map(latency)),
      ev("deploys", "Read the deploy log", ["Oct 13 privacy: PII redaction middleware enabled on POST /ask (runs before embed and search)", "Oct 13 nightly re-index: SUCCESS (5 documents)", "Oct 14 web-chat 4.6: dark mode"], true),
    ],
  };
}

// ======================= Beginner: semantic cache threshold too loose =======================
const TRAFFIC = ["How much does the Starter plan cost per month?", "Can I get a refund on a monthly plan after three weeks?", "How long are request logs kept on the Pro plan?", "What is the API rate limit on the Starter plan?", "How do I disable SSO?", "What happens when I exceed the free tier?"];
const CACHE_FIXES: Fix<{ threshold: number }>[] = [
  { id: "f-threshold", label: "Raise the cache similarity threshold back to 0.95", explanation: "Questions that differ by a plan name score about 0.80 and now miss the cache, so they go through retrieval and generation.", threshold: 0.95 },
  { id: "f-flush", label: "Flush the answer cache", explanation: "The next hour of traffic asks the same Starter/monthly questions and refills it; the loose threshold matches them again.", threshold: 0.75 },
  { id: "f-reindex", label: "Rebuild the vector index", explanation: "Cache hits never reach the index.", threshold: 0.75 },
];
type Served = { run: RagRun; hit?: { question: string; sim: number; answer: string }; correct: boolean; spans: TraceSpan[] };

function cacheRun(fixId?: string): Outcome {
  const fix = CACHE_FIXES.find((f) => f.id === fixId) ?? { threshold: 0.75 };
  const cfg = DEFAULT_RAG;
  const index = buildIndex(cfg);
  const cache: { question: string; vector: number[]; answer: string }[] = [];
  const ask = (x: Question): Served => {
    const fresh = answerQuestion(index, cfg, x);
    const vector = embed(x.text, cfg.queryModel);
    const best = cache.map((c) => ({ c, sim: dot(vector, c.vector) })).sort((a, b) => b.sim - a.sim)[0];
    const embedQ = fresh.spans.find((s) => s.id === "embed.query")!;
    if (best && best.sim >= fix.threshold) {
      const root = fresh.spans.find((s) => s.id === "rag.query")!;
      const t = embedQ.startMs + embedQ.durationMs;
      const spans: TraceSpan[] = [
        ...fresh.spans.filter((s) => s.id === "index.build" || s.parentId === "index.build"),
        { ...root, durationMs: embedQ.durationMs + 4, output: { http: 200, cache: "HIT" } }, embedQ,
        { id: "cache.lookup", parentId: "rag.query", name: "semantic cache lookup", component: "cache", startMs: t, durationMs: 4, status: "ok", input: { threshold: fix.threshold }, output: { hit: true, matched: best.c.question, similarity: +best.sim.toFixed(3) }, events: [{ id: "c0", type: "log", message: `cache HIT: similarity ${best.sim.toFixed(3)} ≥ ${fix.threshold} to “${best.c.question}”`, severity: "info" }] },
      ];
      return { run: fresh, hit: { question: best.c.question, sim: best.sim, answer: best.c.answer }, correct: best.c.answer.toLowerCase().includes(x.expect.toLowerCase()), spans };
    }
    cache.push({ question: x.text, vector, answer: fresh.answer.text });
    return { run: fresh, correct: fresh.evaluation.ok, spans: fresh.spans };
  };
  TRAFFIC.forEach((text) => ask({ id: "traffic", text, expect: "\u0000", sourceDoc: "" }));
  const served = QUESTIONS.map(ask);
  const s = served[0];
  const answer = s.hit?.answer ?? s.run.answer.text;
  const lat = served.map((x) => x.spans.find((sp) => sp.id === "rag.query")!.durationMs);
  const hits = served.filter((x) => x.hit).length;
  return {
    spans: s.spans,
    metrics: health(lat, [{ label: "Cache hit rate", value: pct(hits / served.length), status: "ok", kind: "health" }, quality("Golden-set accuracy", share(served.map((x) => x.correct)))]),
    transcript: chat(s.run.question.text, answer, s.correct),
    resolved: served.every((x) => x.correct),
    evidence: [
      goldenEv("Run the golden-set evaluation (through the live API)", served.map((x) => ({ text: x.run.question.text, ok: x.correct }))),
      ev("answer", "Compare the answer with the source of truth", [`Answer given: “${answer}”`, truth("q-price")], false),
      ev("trace", "Read the request trace", s.hit ? ["spans: POST /ask → embed query → semantic cache lookup (HIT) → response", "no vector search, rerank, context or generate span ran for this request"] : ["spans: POST /ask → embed query → cache MISS → vector search → rerank → context → generate"], true, s.hit ? "cache.lookup" : "retrieve"),
      ev("cache", "Look up which cache entries were served", served.map((x) => (x.hit ? `HIT  “${x.run.question.text}” ← cached “${x.hit.question}” (similarity ${x.hit.sim.toFixed(3)} ≥ ${fix.threshold})` : `MISS “${x.run.question.text}”`)), true),
      ev("bypass", "Re-ask with the cache bypassed (Cache-Control: no-cache)", [`Answer: “${s.run.answer.text}”`, s.run.evaluation.ok ? "Correct, with a valid citation." : "Still wrong."], true, "generate"),
      searchEv(index, "$49"),
      faithEv(s.run),
      modelsEv(s.run),
      dashEv(lat, [`cache hit rate ${pct(hits / served.length)} on the golden set; LLM spend down 41% since Oct 15`]),
      ev("deploys", "Read the deploy log", ["Oct 15 semantic-cache 1.2: similarity threshold lowered 0.95 → 0.75 to raise the hit rate and cut LLM cost", "Oct 15 nightly re-index: SUCCESS (5 documents)", "Oct 16 web-chat 4.6: dark mode"], true),
    ],
  };
}

// ======================= Advanced: query condenser folds the whole conversation into search =======================
const SESSION = ["q-sso", "q-rate", "q-refund", "q-price", "q-429", "q-retention"].map(q);
const HIST_FIXES: Fix<{ condense: (turn: number) => string }>[] = [
  { id: "f-standalone", label: "Condense only when the new message depends on earlier turns (pronouns, “that plan”); otherwise search with the message as typed", explanation: "Every turn in this session is a standalone question, so each one searches with its own words.", condense: (i) => SESSION[i].text },
  { id: "f-window", label: "Keep only the previous turn in the condensed query", explanation: "One earlier question is still enough to pull search and answer extraction toward the previous topic.", condense: (i) => [...SESSION.slice(Math.max(0, i - 1), i), SESSION[i]].map((x) => x.text).join(" ") },
  { id: "f-trunc", label: "Put the new message first and cap the condensed query at 16 words", explanation: "Fixes turns whose question nearly fills the cap; short questions like the 429 one still get 9 words of the previous turn.", condense: (i) => [SESSION[i], ...SESSION.slice(0, i).reverse()].map((x) => x.text).join(" ").split(" ").slice(0, 16).join(" ") },
  { id: "f-topk", label: "Raise top-k to 8 and put 5 chunks in context", explanation: "More passages, but the answer step still scores sentences against words from earlier turns.", condense: (i) => SESSION.slice(0, i + 1).map((x) => x.text).join(" ") },
];
const FAULT_CONDENSE = (i: number) => SESSION.slice(0, i + 1).map((x) => x.text).join(" ");

function historyRun(fixId?: string): Outcome {
  const fix = HIST_FIXES.find((f) => f.id === fixId);
  const cfg: RagConfig = { ...DEFAULT_RAG, ...(fixId === "f-topk" ? { topK: 8, contextChunks: 5 } : {}) };
  const condense = fix?.condense ?? FAULT_CONDENSE;
  const index = buildIndex(cfg);
  const turns = SESSION.map((x, i) => answerQuestion(index, cfg, { ...x, text: condense(i) }));
  const offline = QUESTIONS.map((x) => answerQuestion(index, cfg, x));
  const focus = 2;
  const r = turns[focus];
  const fresh = offline.find((x) => x.question.id === SESSION[focus].id)!;
  const lat = turns.map(latency);
  const refundChunks = index.chunks.filter((c) => c.docId === "refunds");
  return {
    spans: r.spans,
    metrics: health(lat, [quality("Offline golden-set accuracy", share(offline.map((x) => x.evaluation.ok))), quality("Production session accuracy", share(turns.map((x) => x.evaluation.ok)))]),
    transcript: SESSION.slice(0, focus + 1).flatMap((x, i) => chat(x.text, turns[i].answer.text, turns[i].evaluation.ok)),
    resolved: turns.every((x) => x.evaluation.ok),
    evidence: [
      goldenEv("Run the offline golden-set evaluation", offline.map((x) => ({ text: x.question.text, ok: x.evaluation.ok }))),
      ev("session", "Replay the failing production chat session", SESSION.map((x, i) => `turn ${i + 1} ${turns[i].evaluation.ok ? "PASS" : "FAIL"} · ${x.text}`), true),
      ev("query", `Inspect the query the embedder received on turn ${focus + 1}`, [`“${r.question.text}”`, `${wordCount(r.question.text)} words; the user's message is the last ${wordCount(SESSION[focus].text)}`], true, "embed.query"),
      retrievalEv(r, true, `refund-policy.md chunk ranks #${r.reranked.findIndex((h) => h.entry.docId === "refunds") + 1}; the answer step picked “${r.answer.sentence || "nothing"}”.`),
      ev("fresh", `Ask turn ${focus + 1}'s question in a brand-new session`, [`Answer: “${fresh.answer.text}”`, fresh.evaluation.ok ? "Correct, with a valid citation." : "Still wrong."], true),
      ev("chunks", "Inspect chunk boundaries for refund-policy.md", [...refundChunks.map((c) => `${c.id}: words ${c.startWord}–${c.endWord} (${c.endWord - c.startWord} words)`), `“30 days of purchase” is in: ${refundChunks.filter((c) => c.text.includes("30 days of purchase")).map((c) => c.id).join(", ")}`], false, "chunk"),
      faithEv(r),
      modelsEv(r),
      dashEv(lat, [`mean prompt size per turn grew from ${Math.round(offline.reduce((a, x) => a + x.promptTokens, 0) / offline.length)} to ${Math.round(turns.reduce((a, x) => a + x.promptTokens, 0) / turns.length)} tokens`]),
      ev("deploys", "Read the deploy log", ["Oct 16 chat-api 3.4: follow-up support; a query condenser builds the search query from the conversation so far", "Oct 16 eval: offline golden set unchanged (single-turn questions), 100% → shipped", "Oct 17 nightly re-index: SUCCESS (5 documents)", "Oct 17 docs: refund-policy.md re-published (formatting only)"], true),
    ],
  };
}

// ======================= Advanced: partner docs leak across tenants =======================
const PARTNER: SourceDoc[] = [
  { id: "contoso-pricing", file: "contoso/pricing.md", title: "Contoso Edge plans", version: "v1", updatedAt: "2026-10-10", blocks: [{ type: "heading", text: "Contoso Edge plans" }, { type: "paragraph", text: "The Pro plan costs $29 per month and includes 5 seats." }] },
  { id: "contoso-rate-limits", file: "contoso/api-limits.md", title: "Contoso Edge API limits", version: "v1", updatedAt: "2026-10-10", blocks: [{ type: "heading", text: "API limits" }, { type: "paragraph", text: "The Pro plan allows 1000 requests per minute per API key." }] },
  { id: "contoso-retention", file: "contoso/retention.md", title: "Contoso Edge log retention", version: "v1", updatedAt: "2026-10-10", blocks: [{ type: "heading", text: "Log retention" }, { type: "paragraph", text: "Request logs are kept for 90 days on every plan, including the Enterprise plan." }] },
];
const SHARED_DOCS = new Set(["sso"]);
const TENANT_FIXES: Fix<{ tagPartner: boolean; allowShared: boolean; cfg?: Partial<RagConfig> }>[] = [
  { id: "f-tag", label: "Re-ingest the partner upload with tenant: contoso, and make a missing tenant field fail ingestion instead of defaulting", explanation: "Contoso chunks fall outside Northwind's filter while the genuinely shared SSO guide stays visible.", tagPartner: true, allowShared: true },
  { id: "f-noshared", label: "Change the search filter to tenant = northwind only (drop “shared”)", explanation: "Removes the partner chunks, but also hides the shared SSO guide, so SSO questions now find nothing.", tagPartner: false, allowShared: false },
  { id: "f-topk", label: "Raise top-k to 8 and put 5 chunks in context", explanation: "The partner chunks still rank first and the answer step still prefers them.", tagPartner: false, allowShared: true, cfg: { topK: 8, contextChunks: 5 } },
  { id: "f-rerank", label: "Turn the reranker off", explanation: "The partner chunks already rank first by vector similarity.", tagPartner: false, allowShared: true, cfg: { rerank: false } },
];

function tenantRun(fixId?: string): Outcome {
  const fix = TENANT_FIXES.find((f) => f.id === fixId) ?? { tagPartner: false, allowShared: true, cfg: {} };
  const cfg: RagConfig = { ...DEFAULT_RAG, ...(fix.cfg ?? {}) };
  const tenantOf = (docId: string) => (PARTNER.some((d) => d.id === docId) ? (fix.tagPartner ? "contoso" : "shared") : SHARED_DOCS.has(docId) ? "shared" : "northwind");
  const allowed = fix.allowShared ? ["northwind", "shared"] : ["northwind"];
  const full = indexOf([...CORPUS, ...PARTNER], cfg);
  const visible: RagIndex = { ...full, entries: full.entries.filter((e) => allowed.includes(tenantOf(e.docId))) };
  const runs = QUESTIONS.map((x) => answerQuestion(visible, cfg, x));
  const r = runs.find((x) => x.question.id === "q-price")!;
  const file = (docId: string) => full.sources.find((d) => d.id === docId)!.file;
  return {
    spans: r.spans,
    metrics: health(runs.map(latency), [{ label: "Ingestion job", value: "SUCCESS", status: "ok", kind: "health" }, quality("Golden-set accuracy", share(runs.map((x) => x.evaluation.ok))), quality("Faithfulness", share(runs.map((x) => x.evaluation.faithful)))]),
    transcript: chat(r.question.text, r.answer.text, r.evaluation.ok),
    resolved: runs.every((x) => x.evaluation.ok),
    evidence: [
      goldenEv("Run the golden-set evaluation", runs.map((x) => ({ text: x.question.text, ok: x.evaluation.ok }))),
      ev("answer", "Compare the answer with the source of truth", [`Answer given: “${r.answer.text}”`, truth("q-price")], false),
      retrievalEv(r, true, `passage [1] comes from ${r.context[0] ? file(r.context[0].entry.docId) : "nothing"}`),
      ev("index", "List the documents in the index with their tenant tag", full.sources.map((d) => `${d.file}: tenant ${tenantOf(d.id)}${PARTNER.some((p) => p.id === d.id) && !fix.tagPartner ? " (defaulted: upload had no tenant field)" : ""}`), true),
      ev("request", "Inspect the search request sent to the vector index", [`API key tenant: northwind`, `filter: { tenant: { $in: ${JSON.stringify(allowed)} } }`, `candidates after filter: ${visible.entries.length} of ${full.entries.length} chunks`, `partner chunks visible to Northwind: ${visible.entries.filter((e) => PARTNER.some((d) => d.id === e.docId)).length}`], true, "retrieve"),
      ev("ingest", "Read the ingestion job log for Oct 10", PARTNER.map((d) => `ingest ${d.file} → ${fix.tagPartner ? "tenant=contoso" : "no tenant field, defaulting to tenant=shared"}`).concat(["job status: SUCCESS"]), true),
      searchEv(full, "$49"),
      faithEv(r),
      modelsEv(r),
      dashEv(runs.map(latency)),
      ev("deploys", "Read the deploy log", ["Oct 08 ingestion 3.0: documents without a tenant field now default to tenant=shared (previously rejected)", "Oct 10 partnerships: Contoso Edge white-label docs uploaded via the bulk API", "Oct 10 nightly re-index: SUCCESS (5 → 8 documents)", "Oct 12 web-chat 4.6: dark mode"], true),
    ],
  };
}

export const RAG_MORE_SCENARIOS: Scenario[] = [
  {
    id: "rag-nav-boilerplate", title: "Rate-limit answers read like a menu", system: "RAG", difficulty: "Beginner", relatedTopic: "parsing-and-extraction",
    symptom: "“Home Pricing Plans Refunds SSO Security API Rate limits…”", objective: "Read the chunk the model actually received before tuning retrieval.",
    incident: `Since the docs moved to the new website on Monday, customers asking “${q("q-rate").text}” get back a list of menu items instead of a number. The rate-limit page is correct on the website, and the crawler fetched every page without errors.`,
    architecture: RAG_ARCH, run: navRun,
    hypotheses: [
      { id: "h-boiler", text: "The crawler extracts the whole page body, so every chunk starts with the site menu and ends with the footer, and the menu line out-matches the real answer.", isRootCause: true, evidenceIds: ["chunks", "search", "parser", "deploys"], explanation: "The top passage begins with “Home Pricing Plans … API Rate limits”, the same menu sits in every chunk, each page gained the same template words, and ingestion switched to crawling the full <body> on Oct 14." },
      { id: "h-retrieval", text: "Retrieval is returning the wrong page.", isRootCause: false, evidenceIds: ["retrieval"], explanation: "api-rate-limits.md is ranked #1. The problem is the text inside its chunk." },
      { id: "h-halluc", text: "The model is making up a strange answer.", isRootCause: false, evidenceIds: ["faithfulness", "chunks"], explanation: "The answer is copied word for word from the passage. It is the menu text that the crawler put there." },
      { id: "h-topk", text: "Too few chunks reach the model.", isRootCause: false, evidenceIds: ["retrieval", "search"], explanation: "The right chunk already reaches the model; more chunks only add more copies of the menu." },
      { id: "h-embed", text: "The query and index embedding models don't match.", isRootCause: false, evidenceIds: ["models", "golden"], explanation: "Both are embed-v1, and four of six golden questions still pass." },
    ],
    fixes: publicFixes(NAV_FIXES),
    verifications: [
      { id: "v-template", label: "Re-run the golden set and add an ingestion check that flags any sentence appearing on most pages", sound: true, explanation: "Confirms the answers and catches the next theme change that leaks template text into the index." },
      { id: "v-crawl", label: "Confirm every page was crawled with HTTP 200", sound: false, explanation: "They were. The crawl succeeded; it just kept too much of each page." },
      { id: "v-rank", label: "Check that api-rate-limits.md ranks #1", sound: false, explanation: "It already did during the incident." },
    ],
    rootCause: "Ingestion now crawls the website and keeps the whole <body>, so the shared menu (“Home Pricing Plans Refunds SSO Security API Rate limits…”) and footer are added to every page. For rate-limit questions the menu line matches more query words than the real sentence, so the assistant faithfully quotes the menu.",
  },
  {
    id: "rag-query-redaction", title: "Pro customers get the Starter price", system: "RAG", difficulty: "Beginner", relatedTopic: "retrieval-failures",
    symptom: "Says Pro costs $0 per month", objective: "Check what question search actually received, not just what the user typed.",
    incident: `Since the privacy release on Monday, “${q("q-price").text}” is answered with the Starter price, SSO questions get “I couldn't find this”, and the Pro rate limit comes back as a refund rule. Nothing in the docs or the index changed.`,
    architecture: RAG_ARCH, run: redactRun,
    hypotheses: [
      { id: "h-redact", text: "The new PII redactor replaces product words (Pro, SSO, API, Enterprise) with [REDACTED] before the question reaches search and the model.", isRootCause: true, evidenceIds: ["query", "redactor", "deploys"], explanation: "The embedder received “How much does the [REDACTED] plan cost per month?”. The middleware log shows it treating capitalized words as possible names, and it was turned on in front of search on Oct 13." },
      { id: "h-parser", text: "The pricing table was dropped by the parser.", isRootCause: false, evidenceIds: ["search"], explanation: "A chunk containing “$49” is in the index." },
      { id: "h-retrieval", text: "Retrieval returns the wrong document.", isRootCause: false, evidenceIds: ["retrieval"], explanation: "pricing.pdf ranks #1. Without the word “Pro”, every plan row matches equally and the first one, Starter, wins." },
      { id: "h-halluc", text: "The model is hallucinating prices.", isRootCause: false, evidenceIds: ["faithfulness"], explanation: "The answer is copied word for word from the pricing passage. It is just the wrong row." },
      { id: "h-embed", text: "The query and index embedding models don't match.", isRootCause: false, evidenceIds: ["models"], explanation: "Both are embed-v1." },
    ],
    fixes: publicFixes(REDACT_FIXES),
    verifications: [
      { id: "v-golden", label: "Re-run the golden set and add a test that the text sent to search equals the user's question", sound: true, explanation: "Checks every question and stops a future middleware from silently rewriting queries." },
      { id: "v-pricing", label: "Re-ask the Pro price question once", sound: false, explanation: "The plan-name allowlist alone would pass this and still leave SSO broken." },
      { id: "v-redactor", label: "Confirm the redactor's unit tests pass", sound: false, explanation: "They test that names are removed. That is exactly what broke the queries." },
    ],
    rootCause: "A privacy middleware added in front of POST /ask redacts every capitalized word that doesn't start the sentence, treating it as a possible person's name. “Pro”, “SSO”, “API” and “Enterprise” become [REDACTED], so search and answer extraction lose the key word: the price question matches every plan row and returns Starter's $0.",
  },
  {
    id: "rag-semantic-cache", title: "Answers meant for someone else's question", system: "RAG", difficulty: "Beginner", relatedTopic: "similarity-search",
    symptom: "Pro price question answered with the Starter price", objective: "Notice when a response never went through retrieval at all.",
    incident: `Since Thursday, “${q("q-price").text}” returns the Starter plan's $0 price, and annual-refund questions get the monthly rule. Responses are faster than ever and LLM spend is down. The index was rebuilt last night.`,
    architecture: RAG_ARCH, run: cacheRun,
    hypotheses: [
      { id: "h-cache", text: "The semantic cache threshold is too loose: a question that differs by one plan name scores 0.80 and is served another question's cached answer.", isRootCause: true, evidenceIds: ["trace", "cache", "bypass", "deploys"], explanation: "The trace shows a cache HIT with no retrieval or generation, the matched entry is the Starter question at 0.80 ≥ 0.75, the threshold was lowered from 0.95 on Oct 15, and with the cache bypassed the pipeline answers $49." },
      { id: "h-halluc", text: "The model is hallucinating the price.", isRootCause: false, evidenceIds: ["trace", "bypass"], explanation: "No generate step ran for the failing request. When it does run, it answers correctly." },
      { id: "h-index", text: "The index is missing the Pro price.", isRootCause: false, evidenceIds: ["search", "bypass"], explanation: "The index has a chunk with “$49” and a fresh run finds it." },
      { id: "h-embed", text: "The query and index embedding models don't match.", isRootCause: false, evidenceIds: ["models"], explanation: "Both are embed-v1, and retrieval works when it is allowed to run." },
      { id: "h-cost", text: "The cost-saving deploy degraded the model.", isRootCause: false, evidenceIds: ["bypass"], explanation: "The model is unchanged and correct when called. The saving came from skipping it." },
    ],
    fixes: publicFixes(CACHE_FIXES),
    verifications: [
      { id: "v-pairs", label: "Run the golden set through the live API with near-duplicate pairs (Pro vs Starter, annual vs monthly) and check none is served the other's answer", sound: true, explanation: "Tests the cache against the exact kind of question it confused." },
      { id: "v-bypass", label: "Run the golden set with the cache bypassed", sound: false, explanation: "That passes during the incident, because it skips the component that's wrong." },
      { id: "v-hit", label: "Check the cache hit rate went up", sound: false, explanation: "A higher hit rate is what the bad threshold produced." },
    ],
    rootCause: "The semantic cache's similarity threshold was lowered from 0.95 to 0.75 to save cost. Questions that differ only by a plan name or billing period (Pro vs Starter, annual vs monthly) score about 0.80, so they are served a cached answer to someone else's question without any retrieval or generation.",
  },
  {
    id: "rag-history-condense", title: "Follow-up questions get yesterday's topic", system: "RAG", difficulty: "Advanced", relatedTopic: "retrieval-failures",
    symptom: "Refund question answered with the API rate limit", objective: "Explain why production fails while the offline golden set passes, by comparing what each actually sends to search.",
    incident: `Since follow-up support shipped on Wednesday, support agents report odd answers deep into chats: after asking about SSO and the API rate limit, a customer asking “${q("q-refund").text}” was told the Pro plan allows 600 requests per minute. The offline golden set is still 100%, and the refund-policy page was re-published the same week.`,
    architecture: RAG_ARCH, run: historyRun,
    hypotheses: [
      { id: "h-condense", text: "The query condenser joins every earlier user message onto the new one, so search and answer extraction are pulled toward previous topics; the single-turn golden set can't see it.", isRootCause: true, evidenceIds: ["session", "query", "fresh", "retrieval", "deploys"], explanation: "Turn 3's search query contains the SSO and rate-limit questions too, the rate-limit chunk ranks above the refund policy, the same question in a new session is answered correctly, and the offline set never has earlier turns." },
      { id: "h-chunk", text: "The refund-policy edit moved a chunk boundary and split the refund rule.", isRootCause: false, evidenceIds: ["chunks", "fresh"], explanation: "“30 days of purchase” sits whole inside refunds#0, and the question is answered correctly in a fresh session." },
      { id: "h-fine", text: "The pipeline is fine (golden set 100%); users are asking ambiguous questions.", isRootCause: false, evidenceIds: ["golden", "query", "session"], explanation: "The user's message is unambiguous. It is the condensed query, which the offline set never exercises, that mixes topics." },
      { id: "h-topk", text: "top-k is too small for long conversations.", isRootCause: false, evidenceIds: ["retrieval"], explanation: "The refund chunk is retrieved and reaches the model, yet the answer step prefers the rate-limit sentence because the query contains rate-limit words." },
      { id: "h-halluc", text: "The model is mixing up topics on its own.", isRootCause: false, evidenceIds: ["faithfulness"], explanation: "The answer sentence is copied verbatim from a retrieved passage. It was asked a mixed question." },
    ],
    fixes: publicFixes(HIST_FIXES),
    verifications: [
      { id: "v-multiturn", label: "Add multi-turn sessions to the golden set (standalone and genuine follow-ups) and replay this session", sound: true, explanation: "Exercises the condenser, which the single-turn set skips." },
      { id: "v-offline", label: "Re-run the offline golden set", sound: false, explanation: "It was 100% throughout the incident; it never sends conversation history." },
      { id: "v-turn3", label: "Ask the refund question in a new chat", sound: false, explanation: "That works even with the bug: a new chat has no history to fold in." },
    ],
    rootCause: "The new query condenser builds the search query by concatenating every previous user message with the new one. Earlier topics (rate limits, pricing) dominate the vector search and the answer step's word matching, so standalone follow-ups get answers about the previous topic. The offline golden set is single-turn, so it never sends history and stays at 100%.",
  },
  {
    id: "rag-tenant-leak", title: "Customers quoted a partner's prices", system: "RAG", difficulty: "Advanced", relatedTopic: "vector-databases",
    symptom: "Says Pro costs $29 and logs are kept 90 days", objective: "Find how another tenant's documents got past the search filter, and fix it without hiding the docs that really are shared.",
    incident: `Since last weekend, Northwind customers asking “${q("q-price").text}” are told $29, and enterprise retention questions get “90 days”. Neither number appears anywhere on Northwind's site. Search requests carry a tenant filter and ingestion reports success.`,
    architecture: RAG_ARCH, run: tenantRun,
    hypotheses: [
      { id: "h-tenant", text: "The partner's upload had no tenant field, ingestion 3.0 defaulted it to “shared”, and Northwind's filter admits “shared”, so Contoso's docs compete with Northwind's.", isRootCause: true, evidenceIds: ["index", "request", "ingest", "retrieval", "deploys"], explanation: "The Contoso files are tagged tenant=shared by default, the filter allows northwind and shared, 3 partner chunks reach Northwind's search, and contoso/pricing.md is passage [1]. Each factor alone is harmless; together they leak." },
      { id: "h-filter", text: "The tenant filter isn't being sent at all.", isRootCause: false, evidenceIds: ["request"], explanation: "The filter is sent and applied. It allows “shared”, which is where the partner docs landed." },
      { id: "h-parser", text: "The pricing table was dropped, so another number is picked.", isRootCause: false, evidenceIds: ["search"], explanation: "A Northwind chunk with “$49” is in the index." },
      { id: "h-halluc", text: "The model is hallucinating prices.", isRootCause: false, evidenceIds: ["faithfulness", "retrieval"], explanation: "“$29 per month” is copied word for word from a retrieved Contoso passage." },
      { id: "h-rerank", text: "The reranker is promoting the wrong chunk.", isRootCause: false, evidenceIds: ["retrieval"], explanation: "The partner chunk is already #1 by vector similarity; it should never have been a candidate." },
    ],
    fixes: publicFixes(TENANT_FIXES),
    verifications: [
      { id: "v-isolation", label: "Re-run the golden set (including SSO) and add an isolation test: each tenant's search must never return another tenant's chunks", sound: true, explanation: "Checks the leak is closed and the shared SSO guide still works." },
      { id: "v-price", label: "Re-ask the price question", sound: false, explanation: "Dropping “shared” from the filter passes this while breaking SSO answers." },
      { id: "v-job", label: "Confirm the ingestion job reports SUCCESS", sound: false, explanation: "It succeeded while defaulting the partner docs to shared." },
    ],
    rootCause: "Two changes combined: ingestion 3.0 started defaulting documents without a tenant field to tenant=shared instead of rejecting them, and Contoso's white-label docs were bulk-uploaded without that field. Northwind's search filter correctly admits shared docs (like the SSO guide), so Contoso's shorter pricing and retention pages outrank Northwind's own and are quoted to Northwind customers.",
  },
];
