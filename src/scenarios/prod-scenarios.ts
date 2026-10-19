import { Archive, Bot, Cpu, Database, Gauge, GitMerge, ListOrdered, MessageSquare, Network, Repeat, Server, ShieldCheck, User, Binary } from "lucide-react";
import { sigmoid } from "../lib/math";
import { rng } from "../lib/random";
import type { SimulationEvent, TraceSpan } from "../types/simulation";
import { MODEL_PRIOR, QUESTIONS } from "../visualizers/rag/corpus";
import { DEFAULT_RAG, runGoldenSet, runRag } from "../visualizers/rag/simulation";
import { evaluateModel, trainLogistic, type Dataset, type Model, type Row } from "./ml-model";
import { LATENCY_STAGES, STORM, percentile, simulateLatency, simulateStorm, stormLatencies, type LatencyConfig, type Req, type RetryPolicy, type Tick } from "./prod-sim";
import type { Evidence, Outcome, Scenario } from "./types";

const ms = (v: number) => `${Math.round(v)} ms`;
const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
let evN = 0;
const ev = (message: string, severity: SimulationEvent["severity"] = "info"): SimulationEvent => ({ id: `p${evN++}`, type: "log", message, severity });
const evidence = (id: string, action: string, lines: string[], relevant: boolean, spanId?: string): Evidence => ({ id, action, title: action, lines, relevant, spanId });

// ======================= latency regression =======================
const LAT_ARCH: Scenario["architecture"] = {
  nodes: [
    { id: "client", label: "Client", Icon: User, x: 0, y: 110, match: (s) => s.id === "req" },
    { id: "gateway", label: "API gateway", Icon: Network, x: 150, y: 110, match: (s) => s.id === "gateway" },
    { id: "embed", label: "Query embedding", Icon: Binary, x: 300, y: 20, match: (s) => s.id === "embed" },
    { id: "retrieve", label: "Vector search", Icon: Database, x: 450, y: 20, match: (s) => s.id === "retrieve" },
    { id: "rerank", label: "Reranker", Icon: ListOrdered, x: 600, y: 20, match: (s) => s.id.startsWith("rerank") },
    { id: "llm", label: "LLM provider", Icon: Bot, x: 450, y: 200, match: (s) => s.id === "llm" },
    { id: "post", label: "Response", Icon: MessageSquare, x: 300, y: 200, match: (s) => s.id === "post" },
  ],
  edges: [
    { source: "client", target: "gateway" }, { source: "gateway", target: "embed" }, { source: "embed", target: "retrieve" }, { source: "retrieve", target: "rerank" },
    { source: "rerank", target: "llm", sourceHandle: "b", targetHandle: "t" }, { source: "llm", target: "post" },
  ],
};

const SLO = 1500;
const BASE_LAT: LatencyConfig = { rerankCandidates: 20, rerankBatch: 16, llmProvider: "primary", pods: 6, trafficFactor: 1 };
const TODAY_LAT: LatencyConfig = { ...BASE_LAT, rerankCandidates: 150, trafficFactor: 1.18 };
const LAT_FIXES: (Scenario["fixes"][number] & { cfg: Partial<LatencyConfig> })[] = [
  { id: "f-cap", label: "Cap rerank candidates at 30 (and gate the change behind a latency budget)", explanation: "The reranker scores 30 pairs in 2 batches instead of 150 in 10; p95 is back under the SLO.", cfg: { rerankCandidates: 30 } },
  { id: "f-pods", label: "Double the API pods to absorb the extra traffic", explanation: "Gateway queueing was a few ms; the reranker's per-request work is unchanged.", cfg: { pods: 12 } },
  { id: "f-llm", label: "Fail over to the secondary LLM provider", explanation: "The primary provider was never slow, and the secondary is slightly slower.", cfg: { llmProvider: "secondary" } },
];

function reqSpans(q: Req, cfg: LatencyConfig): TraceSpan[] {
  let t = 0;
  const spans: TraceSpan[] = [{ id: "req", name: "POST /ask", component: "api", startMs: 0, durationMs: q.total, status: q.total > SLO ? "warning" : "ok", output: { http: 200 }, events: q.total > SLO ? [ev(`latency ${ms(q.total)} exceeds SLO ${SLO} ms`, "warning")] : [] }];
  const add = (id: string, name: string, component: string, d: number, extra: Partial<TraceSpan> = {}) => { spans.push({ id, parentId: "req", name, component, startMs: t, durationMs: Math.round(d), status: "ok", events: [], ...extra }); t += d; };
  add("gateway", "gateway", "api-gateway", q.ms.gateway, { metrics: { rps: q.rps } });
  add("embed", "embed query", "embedder", q.ms.embed, { output: { cache: q.cacheHit ? "hit" : "miss" } });
  add("retrieve", "vector search top-50", "vector-db", q.ms.retrieve);
  const rerankStart = t;
  spans.push({ id: "rerank", parentId: "req", name: "rerank (cross-encoder)", component: "reranker", startMs: t, durationMs: Math.round(q.ms.rerank), status: "ok", input: { candidates: cfg.rerankCandidates, batch_size: cfg.rerankBatch }, metrics: { batches: q.batches.length }, events: [] });
  q.batches.forEach((b, i) => { spans.push({ id: `rerank.batch${i + 1}`, parentId: "rerank", name: `score batch ${i + 1}`, component: "reranker", startMs: Math.round(rerankStart + q.batches.slice(0, i).reduce((s, x) => s + x, 0)), durationMs: Math.round(b), status: "ok", events: [] }); });
  t += q.ms.rerank;
  add("llm", "generate", "llm-provider", q.ms.llm, { input: { provider: cfg.llmProvider === "primary" ? "provider-a (us-east)" : "provider-b (us-east)" } });
  add("post", "format response", "api", q.ms.post);
  return spans;
}

function latencyRun(fixId?: string): Outcome {
  const cfg = { ...TODAY_LAT, ...(LAT_FIXES.find((f) => f.id === fixId)?.cfg ?? {}) };
  const today = simulateLatency(cfg, 2), yday = simulateLatency(BASE_LAT, 1);
  const tot = (rs: Req[]) => rs.map((r) => r.total);
  const p95 = percentile(tot(today), 0.95), p95y = percentile(tot(yday), 0.95);
  const stage = (rs: Req[], s: (typeof LATENCY_STAGES)[number], q: number) => percentile(rs.map((r) => r.ms[s]), q);
  const sorted = [...today].sort((a, b) => b.total - a.total);
  const rep = sorted[Math.floor(today.length * 0.05)];
  const byRps = [...today].sort((a, b) => a.rps - b.rps);
  const third = Math.floor(byRps.length / 3);
  const resolved = p95 <= SLO;
  const golden = runGoldenSet(DEFAULT_RAG).accuracy;
  const hit = (rs: Req[]) => rs.filter((r) => r.cacheHit).length / rs.length;
  return {
    spans: reqSpans(rep, cfg),
    metrics: [
      { label: "p95 latency", value: ms(p95), status: p95 <= SLO ? "ok" : "error", kind: "health", hint: `SLO ${SLO} ms` },
      { label: "p50 latency", value: ms(percentile(tot(today), 0.5)), status: "ok", kind: "health" },
      { label: "Error rate", value: "0.0%", status: "ok", kind: "health" },
      { label: "Golden-set accuracy", value: pct(golden, 0), status: golden === 1 ? "ok" : "warning", kind: "quality" },
    ],
    transcript: [{ label: "SRE alert", text: `p95 latency ${ms(p95)} over the last 24 h (yesterday ${ms(p95y)}; SLO ${SLO} ms)`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: [
      evidence("dashboards", "Check latency and error dashboards", [`p50 ${ms(percentile(tot(yday), 0.5))} → ${ms(percentile(tot(today), 0.5))} · p95 ${ms(p95y)} → ${ms(p95)} (yesterday → today, SLO ${SLO} ms)`, "error rate 0.0% both days; no 5xx", `requests sampled: ${today.length} per day`], false),
      evidence("breakdown", "Break p95 latency down by span (yesterday → today)", LATENCY_STAGES.map((s) => { const a = stage(yday, s, 0.95), b = stage(today, s, 0.95); return `${s}: p95 ${ms(a)} → ${ms(b)}${b - a > 100 ? `  (+${Math.round(b - a)} ms)` : ""}`; }), true),
      evidence("slowest", "Open the 3 slowest traces", sorted.slice(0, 3).map((r) => `req ${r.id}: ${ms(r.total)} = ${LATENCY_STAGES.map((s) => `${s} ${Math.round(r.ms[s])}`).join(" + ")}`), true, "rerank"),
      evidence("rerank", "Inspect reranker span attributes", [`yesterday: candidates=${BASE_LAT.rerankCandidates}, batch_size=${BASE_LAT.rerankBatch}, batches=${yday[0].batches.length} (sequential), p50 ${ms(stage(yday, "rerank", 0.5))}`, `today: candidates=${cfg.rerankCandidates}, batch_size=${cfg.rerankBatch}, batches=${today[0].batches.length} (sequential), p50 ${ms(stage(today, "rerank", 0.5))}`, `mean time per batch today: ${ms(today[0].batches.reduce((s, b) => s + b, 0) / today[0].batches.length)}`], true, "rerank"),
      evidence("llm", "Check the LLM provider's latency and status page", [`llm span p50 ${ms(stage(yday, "llm", 0.5))} → ${ms(stage(today, "llm", 0.5))}, p95 ${ms(stage(yday, "llm", 0.95))} → ${ms(stage(today, "llm", 0.95))}`, "status page: “Elevated latency in eu-west” (since 09:40); our traffic is served from us-east"], false, "llm"),
      evidence("traffic", "Compare latency at low vs high traffic", [`traffic today vs yesterday: ${pct(TODAY_LAT.trafficFactor - 1, 0)} more requests`, `p95 in the quietest third of the day (≤ ${byRps[third].rps} rps): ${ms(percentile(tot(byRps.slice(0, third)), 0.95))}`, `p95 in the busiest third (≥ ${byRps[2 * third].rps} rps): ${ms(percentile(tot(byRps.slice(2 * third)), 0.95))}`, `gateway p95 ${ms(stage(today, "gateway", 0.95))}`], false),
      evidence("cache", "Check the query-embedding cache", [`hit rate yesterday ${pct(hit(yday), 0)}, today ${pct(hit(today), 0)}`, `embed p95 ${ms(stage(yday, "embed", 0.95))} → ${ms(stage(today, "embed", 0.95))}`], false, "embed"),
      evidence("deploys", "Read the change log", ["08:55 retrieval-config 2.3: rerank_candidates 20 → 150 (+0.4 pt recall@5 on long-tail eval)", "09:30 marketing: product newsletter sent (traffic expected +15–20%)", "10:05 llm-gateway: client library 1.12 → 1.13 (logging only)", "Yesterday 17:00 embed-cache: TTL 24 h → 12 h"], true),
    ],
  };
}

// ======================= retry storm =======================
const STORM_ARCH: Scenario["architecture"] = {
  nodes: [
    { id: "users", label: "Users", Icon: User, x: 0, y: 100, match: () => false },
    { id: "api", label: "Ask API", Icon: Network, x: 160, y: 100, match: (s) => s.id === "req" },
    { id: "retry", label: "Retry middleware", Icon: Repeat, x: 330, y: 100, match: (s) => s.id === "retry" },
    { id: "vdb", label: "Vector DB (4 replicas)", Icon: Database, x: 520, y: 100, match: (s) => s.component === "vector-db" },
    { id: "llm", label: "LLM provider", Icon: Bot, x: 520, y: 230, match: (s) => s.id === "llm" },
  ],
  edges: [{ source: "users", target: "api" }, { source: "api", target: "retry" }, { source: "retry", target: "vdb" }, { source: "api", target: "llm", sourceHandle: "b", targetHandle: "l" }],
};
const BASE_POLICY: RetryPolicy = { maxRetries: 5, backoff: "none", breaker: false };
const STORM_FIXES: (Scenario["fixes"][number] & { policy: Partial<RetryPolicy> })[] = [
  { id: "f-backoff", label: "Exponential backoff with full jitter and a retry budget (retries ≤ 10% of requests)", explanation: "Retry traffic can never exceed 10% of real traffic, so the DB stays under capacity once the replica is back.", policy: { backoff: "exp-jitter", retryBudget: 0.1 } },
  { id: "f-breaker", label: "Add a circuit breaker: when > 50% of calls fail, fail fast for 5 s, then probe", explanation: "The breaker sheds load whenever the DB is overloaded, so it can recover.", policy: { breaker: true } },
  { id: "f-retries", label: "Raise max retries to 8 so more requests eventually succeed", explanation: "Each failure now generates even more load; the storm gets worse.", policy: { maxRetries: 8 } },
];

function stormSpans(ticks: Tick[], p: RetryPolicy): TraceSpan[] {
  const sample = stormLatencies(ticks, p, 200, 201, 3)[0];
  const spans: TraceSpan[] = [
    { id: "req", name: "POST /ask", component: "api", startMs: 0, durationMs: Math.round(sample.ms), status: sample.ok ? "ok" : "error", output: { http: sample.ok ? 200 : 503 }, events: [] },
    { id: "embed", parentId: "req", name: "embed query", component: "embedder", startMs: 0, durationMs: 25, status: "ok", events: [] },
    { id: "retry", parentId: "req", name: "retry middleware", component: "retry", startMs: 25, durationMs: 0, status: sample.ok ? "ok" : "error", input: { max_retries: p.maxRetries, backoff: p.backoff, breaker: p.breaker ? "on" : "off" }, events: [] },
  ];
  let t = 25;
  if (!sample.attempts.length) spans[2].events.push(ev("circuit open: failing fast without calling the vector DB", "warning"));
  sample.attempts.forEach((a, i) => {
    spans.push({ id: `vdb${i + 1}`, parentId: "retry", name: `vector search attempt ${i + 1}`, component: "vector-db", startMs: Math.round(t), durationMs: Math.round(a.ms), status: a.ok ? "ok" : "error", output: { status: a.ok ? 200 : 503 }, events: a.ok ? [] : [ev("503 Service Unavailable: admission queue full", "error")] });
    t += a.ms;
  });
  spans[2].durationMs = Math.round(t - 25);
  if (sample.ok) spans.push({ id: "llm", parentId: "req", name: "generate", component: "llm-provider", startMs: Math.round(t), durationMs: Math.round(sample.ms - t), status: "ok", events: [] });
  return spans;
}

function stormRun(fixId?: string): Outcome {
  const policy = { ...BASE_POLICY, ...(STORM_FIXES.find((f) => f.id === fixId)?.policy ?? {}) };
  const ticks = simulateStorm(policy);
  const win = (a: number, b: number) => ticks.slice(a, b);
  const errRate = (ts: Tick[]) => ts.reduce((s, t) => s + t.userFail, 0) / ts.reduce((s, t) => s + t.newReqs, 0);
  const avg = (ts: Tick[], f: (t: Tick) => number) => ts.reduce((s, t) => s + f(t), 0) / ts.length;
  const before = win(0, STORM.outageStart), outage = win(STORM.outageStart, STORM.outageEnd), after = win(STORM.outageEnd + 30, STORM.seconds);
  const lat = (a: number, b: number) => percentile(stormLatencies(ticks, policy, a, b, 7).map((x) => x.ms), 0.95);
  const errAfter = errRate(after);
  const resolved = errAfter < 0.01;
  const now = ticks.at(-1)!;
  return {
    spans: stormSpans(ticks, policy),
    metrics: [
      { label: "User error rate (last 2 min)", value: pct(errAfter), status: errAfter < 0.01 ? "ok" : "error", kind: "health" },
      { label: "p95 latency (last 2 min)", value: ms(lat(STORM.outageEnd + 30, STORM.seconds)), status: lat(STORM.outageEnd + 30, STORM.seconds) <= 1500 ? "ok" : "warning", kind: "health" },
      { label: "Vector DB requests/s", value: Math.round(now.sent).toString(), status: now.sent > STORM.capacity ? "error" : "ok", kind: "health", hint: `capacity ${STORM.capacity}` },
      { label: "Answers served", value: pct(1 - errAfter), status: errAfter < 0.01 ? "ok" : "error", kind: "quality" },
    ],
    transcript: [{ label: "User", text: QUESTIONS.find((q) => q.id === "q-retention")!.text }, { label: "Assistant", text: errAfter < 0.01 ? runRag(DEFAULT_RAG, "q-retention").answer.text : "Something went wrong (503: vector search unavailable). Please try again.", tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: [
      evidence("dashboards", "Check error-rate and latency dashboards by phase", [[before, "14:20:00–14:21:00 (normal)", 0, STORM.outageStart], [outage, "14:21:00–14:21:30 (replica vdb-3 down)", STORM.outageStart, STORM.outageEnd], [after, "14:22:00–14:24:00 (vdb-3 back)", STORM.outageEnd + 30, STORM.seconds]].map(([ts, label, a, b]) => `${label}: user errors ${pct(errRate(ts as Tick[]))}, p95 ${ms(lat(a as number, b as number))}`), false),
      evidence("vdb", "Read vector DB server metrics", [`capacity ${STORM.capacity} req/s with 4 replicas (${STORM.degradedCapacity} with 3)`, ...[[before, "normal"], [outage, "replica down"], [after, "replica back"]].map(([ts, l]) => `${l}: received ${Math.round(avg(ts as Tick[], (t) => t.sent))} req/s, served ${Math.round(avg(ts as Tick[], (t) => t.served))}, rejected ${pct(avg(ts as Tick[], (t) => t.dbErrorRate), 0)} with 503`), "CPU 100% since 14:21:08; most CPU spent accepting and rejecting connections"], true, "vdb1"),
      evidence("retries", "Break vector DB traffic down by attempt number", [`user requests: ${STORM.newRps}/s throughout`, `last 2 min: ${Math.round(avg(after, (t) => t.sent))} DB calls/s, of which ${pct(avg(after, (t) => t.retries / Math.max(1, t.sent)), 0)} are retries`, `amplification: ${(avg(after, (t) => t.sent) / STORM.newRps).toFixed(1)} DB calls per user request`], true),
      evidence("policy", "Read the retry middleware configuration", [`max_retries: ${policy.maxRetries}`, `backoff: ${policy.backoff === "none" ? "none (retry immediately)" : "exponential, full jitter"}`, `retry budget: ${policy.retryBudget ? `${policy.retryBudget * 100}% of requests` : "unlimited"}`, `circuit breaker: ${policy.breaker ? "enabled" : "none"}`], true, "retry"),
      evidence("replicas", "Check vector DB replica health", ["vdb-1, vdb-2, vdb-4: healthy all along", "vdb-3: crashed 14:21:00 (OOM), replaced and healthy since 14:21:30", "replication lag 0 s; no data errors"], false),
      evidence("traffic", "Check incoming user traffic", [`new user requests: ${STORM.newRps}/s, flat for the whole window (no spike)`, "top clients unchanged; no new API keys"], false),
      evidence("llm", "Check LLM provider latency", [`generate p95 ${ms(520 * Math.exp(0.3 * 1.645))}, unchanged`, "LLM is only called after retrieval succeeds; its call volume dropped with retrieval successes"], false),
      evidence("deploys", "Read the change log", ["14:21 vector-db: vdb-3 OOM-killed, auto-replaced at 14:21:30", "Oct 02 ask-api 3.4: retry middleware max_retries=5, no backoff (“retrieval should never fail a request”)", "13:50 llm-gateway: rotated API key", "Oct 15 vector-db: upgraded to 4 replicas"], true),
    ],
  };
}

// ======================= stale fallback =======================
const FB_ARCH: Scenario["architecture"] = {
  nodes: [
    { id: "user", label: "User", Icon: User, x: 0, y: 100, match: () => false },
    { id: "api", label: "Ask API", Icon: Network, x: 170, y: 100, match: (s) => s.id === "req" },
    { id: "retrieval", label: "Retrieval (index v3)", Icon: Database, x: 370, y: 10, match: (s) => s.id === "primary.retrieve" },
    { id: "llm", label: "Primary LLM", Icon: Bot, x: 580, y: 10, match: (s) => s.id === "primary.generate" },
    { id: "cache", label: "Response cache", Icon: Archive, x: 370, y: 200, match: (s) => s.id === "fallback" },
  ],
  edges: [{ source: "user", target: "api" }, { source: "api", target: "retrieval" }, { source: "retrieval", target: "llm" }, { source: "api", target: "cache" }],
};
const MIX: [string, number][] = [["q-rate", 0.3], ["q-price", 0.2], ["q-refund", 0.15], ["q-429", 0.15], ["q-sso", 0.1], ["q-retention", 0.1]];
const POLICY_DOCS = new Set(["pricing", "refunds", "rate-limits", "retention"]);
const FB_FIXES: Scenario["fixes"] = [
  { id: "f-invalidate", label: "Key cache entries by source-document version and invalidate them when a document is republished", explanation: "When api-rate-limits.md became v3, its cached answers were regenerated; the fallback serves current answers." },
  { id: "f-failclosed", label: "For pricing and policy questions, fail closed: say the answer can't be confirmed and link the doc", explanation: "No wrong policy answers are served. Fewer requests get an answer, which the health dashboard will now show honestly." },
  { id: "f-timeout", label: "Raise the primary timeout from 2 s to 4 s", explanation: "Fewer requests fall back, but those that do still get the stale answer." },
  { id: "f-small", label: "Use a small, fast model (no retrieval) as the fallback instead of the cache", explanation: "The small model answers from its training data, which is also from before the change." },
];
type Served = { q: string; path: "primary" | "cache" | "declined" | "small"; text: string; wrong: boolean; ms: number; primaryMs: number };

function fallbackRun(fixId?: string): Outcome {
  const timeout = fixId === "f-timeout" ? 4000 : 2000;
  const current = Object.fromEntries(QUESTIONS.map((q) => [q.id, runRag(DEFAULT_RAG, q.id)]));
  const cached = fixId === "f-invalidate" ? current : Object.fromEntries(QUESTIONS.map((q) => [q.id, runRag({ ...DEFAULT_RAG, indexSnapshot: "stale" }, q.id)]));
  const r = rng(77);
  const served: Served[] = Array.from({ length: 400 }, () => {
    let u = r.next();
    const q = MIX.find(([, w]) => (u -= w) < 0)?.[0] ?? "q-rate";
    const primaryMs = 1450 * Math.exp(0.45 * r.gaussian());
    const doc = QUESTIONS.find((x) => x.id === q)!;
    if (primaryMs <= timeout) return { q, path: "primary", text: current[q].answer.text, wrong: !current[q].evaluation.correct, ms: primaryMs, primaryMs };
    if (fixId === "f-failclosed" && POLICY_DOCS.has(doc.sourceDoc)) return { q, path: "declined", text: "I can't confirm this right now. Please check the documentation page linked below.", wrong: false, ms: timeout + 4, primaryMs };
    if (fixId === "f-small") { const prior = MODEL_PRIOR[q]; return { q, path: "small", text: prior ?? "I'm not sure. Please check the documentation.", wrong: !!prior && !prior.includes(doc.expect), ms: timeout + 300, primaryMs }; }
    return { q, path: "cache", text: cached[q].answer.text, wrong: !cached[q].evaluation.correct, ms: timeout + 6, primaryMs };
  });
  const n = served.length;
  const by = (p: Served["path"]) => served.filter((s) => s.path === p);
  const wrong = served.filter((s) => s.wrong);
  const answered = served.filter((s) => s.path !== "declined").length;
  const fb = served.filter((s) => s.path !== "primary");
  const p95 = percentile(served.map((s) => s.ms), 0.95);
  const audit = served.filter((_, i) => i % 10 === 0);
  const rep = served.find((s) => s.q === "q-rate" && s.path !== "primary") ?? served.find((s) => s.q === "q-rate")!;
  const qText = (id: string) => QUESTIONS.find((x) => x.id === id)!.text;
  const spans: TraceSpan[] = [
    { id: "req", name: "POST /ask", component: "api", startMs: 0, durationMs: Math.round(rep.ms), status: "ok", input: { question: qText(rep.q) }, output: { http: 200, served_by: rep.path }, events: [] },
    { id: "primary.retrieve", parentId: "req", name: "retrieve (index v3)", component: "vector-index", startMs: 0, durationMs: 42, status: "ok", output: { top1: current[rep.q].retrieved[0].entry.id }, events: [] },
    { id: "primary.generate", parentId: "req", name: "generate (primary LLM)", component: "llm", startMs: 42, durationMs: Math.round(Math.min(rep.primaryMs, timeout)) - 42, status: rep.path === "primary" ? "ok" : "error", events: rep.path === "primary" ? [] : [ev(`timeout after ${timeout} ms`, "error")] },
  ];
  if (rep.path !== "primary") spans.push({ id: "fallback", parentId: "req", name: rep.path === "cache" ? "fallback: response cache" : rep.path === "small" ? "fallback: small model" : "fallback: fail closed", component: "cache", startMs: timeout, durationMs: Math.round(rep.ms - timeout), status: "ok", output: rep.path === "cache" ? { hit: true, created: fixId === "f-invalidate" ? "2026-10-01" : "2026-09-28", ttl: "30d" } : { answer: rep.text.slice(0, 60) }, events: [] });
  const resolved = wrong.length === 0;
  const cacheEntry = cached["q-rate"];
  return {
    spans,
    metrics: [
      { label: "Requests answered", value: pct(answered / n), status: answered === n ? "ok" : "warning", kind: "health" },
      { label: "HTTP 5xx", value: "0.0%", status: "ok", kind: "health" },
      { label: "p95 latency", value: ms(p95), status: p95 <= 2500 ? "ok" : "warning", kind: "health" },
      { label: "Fallback share", value: pct(fb.length / n), status: fb.length / n > 0.1 ? "warning" : "ok", kind: "health" },
      { label: "Answer accuracy (graded)", value: pct(1 - wrong.length / Math.max(1, answered)), status: wrong.length ? "error" : "ok", kind: "quality", hint: "graded offline against current docs" },
    ],
    transcript: [{ label: "User", text: qText(rep.q) }, { label: "Assistant", text: rep.text, tone: rep.wrong ? "err" : "ok" }],
    resolved,
    evidence: [
      evidence("dashboards", "Check availability and latency dashboards", [`answered ${pct(answered / n)} of ${n} requests, HTTP 5xx 0.0%`, `p95 latency ${ms(p95)}`, "no alerts fired"], false),
      evidence("audit", "Audit a sample of served answers against current docs", [`audited ${audit.length} conversations: ${audit.filter((s) => s.wrong).length} wrong`, ...audit.filter((s) => s.wrong).slice(0, 3).map((s) => `WRONG · “${s.text}” (served by ${s.path})`)], true),
      evidence("paths", "Break down answers by serving path", [["primary", "primary LLM"], ["cache", "fallback cache"], ["small", "fallback small model"], ["declined", "fail-closed"]].filter(([p]) => by(p as Served["path"]).length).map(([p, l]) => { const xs = by(p as Served["path"]); return `${l}: ${xs.length} requests (${pct(xs.length / n)}), ${xs.filter((s) => s.wrong).length} wrong when graded against current docs`; }), true),
      evidence("cache", "Inspect the cache entry for the rate-limit question", [`key: normalize(“${qText("q-rate")}”)`, `created ${fixId === "f-invalidate" ? "2026-10-01 (regenerated on doc update)" : "2026-09-28"}, TTL 30 days`, `cached answer: “${cacheEntry.answer.text}”`, `built from ${cacheEntry.index.sources.find((s) => s.id === "rate-limits")!.file} ${cacheEntry.index.sources.find((s) => s.id === "rate-limits")!.version}; source of truth today: v3 (2026-10-01)`], true, "fallback"),
      evidence("primary", "Check primary LLM latency", [`primary p50 ${ms(percentile(served.map((s) => s.primaryMs), 0.5))}, p95 ${ms(percentile(served.map((s) => s.primaryMs), 0.95))}; timeout ${timeout} ms`, `timed out: ${pct(served.filter((s) => s.primaryMs > timeout).length / n)} of requests`, "provider status page: “degraded performance” since Oct 18"], false, "primary.generate"),
      evidence("index", "Check index freshness", ["vector index built 2026-10-02 from current sources", ...current["q-rate"].index.sources.map((s) => `${s.file}: ${s.version} (current)`)], false, "primary.retrieve"),
      evidence("faithful", "Check primary answers for faithfulness", [`primary answers graded: ${by("primary").filter((s) => !s.wrong).length}/${by("primary").length} correct`, `primary rate-limit answer: “${current["q-rate"].answer.text}”`], false),
      evidence("deploys", "Read the change log", ["Oct 18 llm-provider: status page reports degraded performance", "Oct 01 docs: api-rate-limits.md v3 published (Pro 300 → 600 req/min)", "Sep 10 ask-api 3.0: on primary timeout, serve cached answer (key = normalized question, TTL 30 d)", "Oct 17 web-chat: new theme"], true),
    ],
  };
}

// ======================= bad deploy (slice regression) =======================
const DEPLOY_ARCH: Scenario["architecture"] = {
  nodes: [
    { id: "data", label: "Orders + chargebacks", Icon: Database, x: 0, y: 20, match: (s) => s.id === "load" },
    { id: "labels", label: "Label join", Icon: GitMerge, x: 170, y: 20, match: (s) => s.id === "labels" },
    { id: "train", label: "Train risk-v2", Icon: Cpu, x: 340, y: 20, match: (s) => s.id === "train" },
    { id: "gate", label: "Canary gate", Icon: ShieldCheck, x: 510, y: 20, match: (s) => s.id === "canary" },
    { id: "v1", label: "risk-v1 (previous)", Icon: Archive, x: 340, y: 190, match: () => false },
    { id: "server", label: "Model server", Icon: Server, x: 510, y: 190, match: (s) => s.id === "serve" },
    { id: "monitor", label: "Chargeback monitor", Icon: Gauge, x: 680, y: 190, match: (s) => s.id === "online" },
  ],
  edges: [
    { source: "data", target: "labels" }, { source: "labels", target: "train" }, { source: "train", target: "gate" },
    { source: "gate", target: "server", sourceHandle: "b", targetHandle: "t" }, { source: "v1", target: "server" }, { source: "server", target: "monitor" },
  ],
};

/** Order-risk data. `walletLabelBug`: chargebacks are joined on card_id, which wallet orders don't have, so wallet fraud is labelled 0. */
function orderData(seed: number, n: number, walletLabelBug = false): Dataset {
  const r = rng(seed);
  const rows: Row[] = Array.from({ length: n }, () => {
    const amount = 20 + 300 * r.next() ** 2, age = r.next() * 720, wallet = r.next() < 0.18 ? 1 : 0, device = r.next() * 365;
    const z = 0.012 * (amount - 110) - 0.004 * (age - 360) - 0.012 * (device - 180) + 0.3 * wallet - 0.9;
    const y = r.next() < sigmoid(z) ? 1 : 0;
    return { x: [Math.round(amount), Math.round(age), wallet, Math.round(device)], y: walletLabelBug && wallet ? 0 : y };
  });
  return { features: ["amount_usd", "account_age_days", "is_wallet", "device_age_days"], rows };
}
const withoutDevice = (ds: Dataset): Dataset => ({ features: ds.features.slice(0, 3), rows: ds.rows.map((r) => ({ ...r, x: r.x.slice(0, 3) })) });
type Served2 = { name: string; m: Model; rows: (rs: Row[]) => Row[] };

function deployRun(fixId?: string): Outcome {
  const v1: Served2 = { name: "risk-v1", m: trainLogistic(withoutDevice(orderData(41, 2500))), rows: (rs) => rs.map((r) => ({ ...r, x: r.x.slice(0, 3) })) };
  const v2train = orderData(fixId === "f-retrain" ? 52 : 42, 2500, true);
  const v2: Served2 = { name: "risk-v2", m: trainLogistic(v2train, { epochs: fixId === "f-retrain" ? 1200 : 400 }), rows: (rs) => rs };
  const thr = fixId === "f-threshold" ? 0.3 : 0.5;
  const prod = orderData(43, 3000).rows;
  const live = fixId === "f-rollback" ? v1 : v2;
  const evalOn = (s: Served2, rs: Row[], t = 0.5) => evaluateModel(s.m, s.rows(rs), t);
  const wallet = prod.filter((r) => r.x[2] === 1), card = prod.filter((r) => r.x[2] === 0);
  const agg = { v1: evalOn(v1, prod), v2: evalOn(v2, prod), live: evalOn(live, prod, live === v2 ? thr : 0.5) };
  const sl = (s: Served2, t = 0.5) => ({ wallet: evalOn(s, wallet, t), card: evalOn(s, card, t) });
  const liveSlices = sl(live, live === v2 ? thr : 0.5);
  const resolved = liveSlices.wallet.recall >= 0.5 && agg.live.accuracy >= agg.v1.accuracy - 0.02;
  const posRate = (rows: Row[], w: number) => { const xs = rows.filter((r) => r.x[2] === w); return xs.filter((r) => r.y).length / xs.length; };
  const v1train = orderData(41, 2500);
  const slicesLine = (s: Served2) => { const x = sl(s); return `${s.name}: card accuracy ${pct(x.card.accuracy)}, recall ${pct(x.card.recall)} · wallet accuracy ${pct(x.wallet.accuracy)}, recall ${pct(x.wallet.recall)}`; };
  const wIdx = v2.m.features.indexOf("is_wallet");
  const spans: TraceSpan[] = [
    { id: "load", name: "load orders + chargebacks", component: "ml-pipeline", startMs: 0, durationMs: 1400, status: "ok", output: { rows: v2train.rows.length }, events: [] },
    { id: "labels", name: "join chargebacks on card_id", component: "ml-pipeline", startMs: 1400, durationMs: 600, status: "ok", output: { positives: v2train.rows.filter((r) => r.y).length, wallet_positives: v2train.rows.filter((r) => r.y && r.x[2]).length }, events: [] },
    { id: "train", name: `train ${v2.name}`, component: "ml-pipeline", startMs: 2000, durationMs: 4000, status: "ok", input: { features: v2.m.features.join(", ") }, events: [] },
    { id: "canary", name: "canary gate (aggregate accuracy)", component: "ml-pipeline", startMs: 6000, durationMs: 800, status: "ok", metrics: { v1_accuracy: +agg.v1.accuracy.toFixed(3), v2_accuracy: +agg.v2.accuracy.toFixed(3) }, events: [ev(`gate passed: v2 accuracy ${pct(agg.v2.accuracy)} ≥ v1 ${pct(agg.v1.accuracy)} − 1 pt`)] },
    { id: "serve", name: `serving ${live.name}`, component: "serving", startMs: 6800, durationMs: 900, status: "ok", output: { http_200: "100%", p95_ms: 11 }, events: [] },
    { id: "online", name: "online eval (chargebacks after 14 days)", component: "monitoring", startMs: 7700, durationMs: 300, status: resolved ? "ok" : "error", metrics: { accuracy: +agg.live.accuracy.toFixed(3), wallet_recall: +liveSlices.wallet.recall.toFixed(3) }, events: resolved ? [] : [ev(`wallet-order fraud recall ${pct(liveSlices.wallet.recall)}`, "error")] },
  ];
  const walletFraud = wallet.filter((r) => r.y).length;
  return {
    spans,
    metrics: [
      { label: "Model server uptime", value: "100%", status: "ok", kind: "health" },
      { label: "p95 inference latency", value: "11 ms", status: "ok", kind: "health" },
      { label: "Accuracy (all orders)", value: pct(agg.live.accuracy), status: "ok", kind: "quality" },
      { label: "Fraud recall (all orders)", value: pct(agg.live.recall), status: agg.live.recall >= 0.5 ? "ok" : "warning", kind: "quality" },
      { label: "Fraud recall (wallet orders)", value: pct(liveSlices.wallet.recall), status: liveSlices.wallet.recall >= 0.5 ? "ok" : "error", kind: "quality" },
    ],
    transcript: [{ label: "Payments ops", text: `${liveSlices.wallet.fn} of ${walletFraud} fraudulent wallet orders were approved this period (${live.name} serving).`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: [
      evidence("canary", "Read the canary report", [`canary: 5% of traffic for 24 h, scored on a labelled replay set; gate = aggregate accuracy ≥ v1 − 1 pt`, `risk-v1 accuracy ${pct(agg.v1.accuracy)}`, `risk-v2 accuracy ${pct(agg.v2.accuracy)}`, "result: PASSED → promoted to 100%"], false, "canary"),
      evidence("slices", "Compare v1 and v2 by payment-method slice", [slicesLine(v1), slicesLine(v2), `wallet orders are ${pct(wallet.length / prod.length)} of traffic`], true, "online"),
      evidence("labels", "Compare training labels by slice (v1 vs v2 training data)", [`v1 training data: fraud rate card ${pct(posRate(v1train.rows, 0))}, wallet ${pct(posRate(v1train.rows, 1))}`, `v2 training data: fraud rate card ${pct(posRate(v2train.rows, 0))}, wallet ${pct(posRate(v2train.rows, 1))}`], true, "labels"),
      evidence("weights", "Compare learned weights (v1 vs v2)", [`is_wallet: v1 ${v1.m.weights[2].toFixed(2)} → v2 ${v2.m.weights[wIdx].toFixed(2)} (standardized)`, ...v2.m.features.filter((f) => f !== "is_wallet").map((f, i) => `${f}: v1 ${i < 2 ? v1.m.weights[v1.m.features.indexOf(f)].toFixed(2) : "n/a"} → v2 ${v2.m.weights[v2.m.features.indexOf(f)].toFixed(2)}`)], true),
      evidence("mix", "Check traffic mix by payment method", [`wallet share: launch week ${pct(orderData(44, 3000).rows.filter((r) => r.x[2]).length / 3000)} · this week ${pct(wallet.length / prod.length)}`, "no new merchants or payment methods"], false),
      evidence("device", "Evaluate v2's new feature, device_age_days", [`v2 on card orders: recall ${pct(sl(v2).card.recall)} vs v1 ${pct(sl(v1).card.recall)}`, "device_age_days: identical definition in training and serving; no nulls"], false),
      evidence("health", "Check model server health", ["uptime 100%, p95 11 ms, 0 errors", "feature schema check: pass"], false, "serve"),
      evidence("deploys", "Read the change log", ["Oct 06 data-eng: training query rewritten; chargebacks now joined on card_id (was order_id)", "Oct 08 ml: risk-v2 adds device_age_days; canary passed on aggregate accuracy → 100%", "Oct 09 payments: wallet checkout button redesigned", "Oct 12 model-server: autoscaler tuning"], true),
    ],
  };
}

export const PROD_SCENARIOS: Scenario[] = [
  {
    id: "prod-latency-regression", title: "p95 latency doubled overnight", system: "Production", difficulty: "Beginner", relatedTopic: "latency-and-throughput",
    symptom: "p95 latency far over the 1.5 s SLO; no errors", objective: "Use per-span timings to find which stage got slower, and rule out the usual suspects with evidence.",
    incident: "The latency alert fired this morning: p95 for /ask is roughly double yesterday's and well over the 1.5 s SLO. No errors. The LLM provider has an open incident on its status page, and a newsletter went out today.",
    architecture: LAT_ARCH, run: latencyRun,
    hypotheses: [
      { id: "h-rerank", text: "The reranker now scores 150 candidates instead of 20, and its cost grows with every candidate.", isRootCause: true, evidenceIds: ["breakdown", "slowest", "rerank", "deploys"], explanation: "The rerank span accounts for almost all of the increase: 10 sequential batches instead of 2, starting with the retrieval-config change at 08:55." },
      { id: "h-llm", text: "The LLM provider is degraded.", isRootCause: false, evidenceIds: ["llm", "breakdown"], explanation: "LLM span latency is unchanged; the provider incident is in a region we don't use." },
      { id: "h-traffic", text: "The newsletter traffic spike is overloading the service.", isRootCause: false, evidenceIds: ["traffic"], explanation: "p95 is the same in the quietest and busiest hours; gateway time is a few ms." },
      { id: "h-cache", text: "The embedding cache stopped working.", isRootCause: false, evidenceIds: ["cache"], explanation: "Hit rate and embed latency are unchanged." },
    ],
    fixes: LAT_FIXES.map(({ id, label, explanation }) => ({ id, label, explanation })),
    verifications: [
      { id: "v-span", label: "Replay a day of traffic, compare per-span p95 with the baseline, and add a CI latency budget for retrieval-config changes", sound: true, explanation: "Confirms the fix where the time went and catches the next config change before it ships." },
      { id: "v-status", label: "Wait for the provider's status page to turn green", sound: false, explanation: "The provider was never the cause." },
      { id: "v-errors", label: "Confirm the error rate stays at 0%", sound: false, explanation: "It was 0% throughout. Latency regressions don't show up as errors." },
    ],
    rootCause: "A retrieval-config change raised rerank candidates from 20 to 150 for a small recall gain. The cross-encoder scores candidates in sequential batches, so rerank time grew roughly 6×, doubling end-to-end p95. The LLM incident and the traffic bump were coincidences.",
  },
  {
    id: "prod-dependency-errors", title: "Errors persist after the database recovered", system: "Production", difficulty: "Advanced", relatedTopic: "rate-limits-and-retries",
    symptom: "503s and slow answers continue after the replica came back", objective: "Recognize a retry storm: retries that turn a short dependency blip into a lasting overload.",
    incident: "At 14:21 one vector DB replica crashed and was replaced within 30 seconds. Users are still getting 'vector search unavailable' errors minutes later. The DB team says all replicas are healthy, yet the cluster is at 100% CPU.",
    architecture: STORM_ARCH, run: stormRun,
    hypotheses: [
      { id: "h-storm", text: "Immediate, unlimited retries keep the vector DB overloaded: a retry storm that outlives the original failure.", isRootCause: true, evidenceIds: ["vdb", "retries", "policy", "deploys"], explanation: "User traffic is flat, but the DB receives several times that, mostly retries. Rejections still cost CPU, so goodput stays below demand and every failure creates more retries." },
      { id: "h-db", text: "The vector DB is still broken.", isRootCause: false, evidenceIds: ["replicas", "vdb"], explanation: "All replicas are healthy with full capacity. The DB is rejecting excess load, not failing." },
      { id: "h-traffic", text: "A traffic spike is overloading the database.", isRootCause: false, evidenceIds: ["traffic", "retries"], explanation: "User requests are flat; the extra load is generated by our own retries." },
      { id: "h-llm", text: "The LLM provider is failing requests.", isRootCause: false, evidenceIds: ["llm"], explanation: "LLM latency is unchanged; the 503s come from vector search." },
      { id: "h-capacity", text: "The cluster is under-provisioned for normal traffic.", isRootCause: false, evidenceIds: ["dashboards", "vdb"], explanation: "Before 14:21 it served all traffic at 80% utilization with zero errors." },
    ],
    fixes: STORM_FIXES.map(({ id, label, explanation }) => ({ id, label, explanation })),
    verifications: [
      { id: "v-chaos", label: "Chaos test: kill one replica under production-like load and require user errors to return to 0% within 30 s of recovery and DB load to stay under capacity", sound: true, explanation: "Reproduces the trigger and checks the property that failed: recovery." },
      { id: "v-replica", label: "Confirm all four replicas report healthy", sound: false, explanation: "They did during the incident." },
      { id: "v-restart", label: "Restart the API pods and confirm errors stop", sound: false, explanation: "Restarting drops the queued retries so the system recovers, but the next blip will start the storm again." },
    ],
    rootCause: "A 30-second replica loss pushed the vector DB just over capacity. The retry middleware retried every 503 immediately, up to 5 times, multiplying load. Rejecting requests still costs the DB CPU, so goodput fell further and created more retries. After the replica returned, retry traffic alone kept the cluster overloaded.",
  },
  {
    id: "prod-stale-fallback", title: "100% answered, but quoting an old rate limit", system: "Production", difficulty: "Intermediate", relatedTopic: "fallback-strategies",
    symptom: "Some users told 300 req/min; docs say 600", objective: "See how a fallback can keep health metrics green while serving stale answers, and choose a safe fallback policy.",
    incident: "Several Pro customers say the assistant told them their rate limit is 300 requests per minute; the docs say 600. Others asking the same question get 600. The availability dashboard shows 100% of requests answered and no errors.",
    architecture: FB_ARCH, run: fallbackRun,
    hypotheses: [
      { id: "h-cache", text: "When the primary times out, the fallback cache serves answers generated before the rate-limit change; nothing invalidates them.", isRootCause: true, evidenceIds: ["audit", "paths", "cache", "deploys"], explanation: "Every wrong answer came from the fallback cache. The cached entry was built from api-rate-limits.md v2 in September; v3 was published Oct 01 and the cache key ignores document versions." },
      { id: "h-provider", text: "The primary LLM is giving wrong answers.", isRootCause: false, evidenceIds: ["faithful", "paths"], explanation: "Every primary answer is correct. The primary's only problem is latency, which decides how often the fallback is used." },
      { id: "h-index", text: "The vector index is stale.", isRootCause: false, evidenceIds: ["index"], explanation: "The index holds v3 of every document." },
      { id: "h-timeout", text: "The 2 s primary timeout is simply too short.", isRootCause: false, evidenceIds: ["primary", "paths"], explanation: "It explains how often the fallback runs, not why its answers are wrong. Any fallback request can still get the stale answer." },
    ],
    fixes: FB_FIXES,
    verifications: [
      { id: "v-quality", label: "Force the fallback path in a test, grade its answers against current docs, and add answer-accuracy-by-path to the dashboard", sound: true, explanation: "Exercises the path that broke and makes its quality visible next to availability." },
      { id: "v-answered", label: "Confirm 100% of requests are answered", sound: false, explanation: "That was true during the incident. Availability isn't correctness." },
      { id: "v-provider", label: "Wait for the provider's latency to recover and confirm fewer fallbacks", sound: false, explanation: "Hides the stale cache until the next slowdown." },
    ],
    rootCause: "On primary timeouts the API serves a cached answer keyed only by the question, with a 30-day TTL. The rate-limit page changed on Oct 01, but the cached answer was built from the old page in September. While the provider is slow, about a quarter of requests fall back, and rate-limit questions get 300.",
  },
  {
    id: "prod-bad-deploy", title: "New risk model passed canary; wallet fraud is getting through", system: "Production", difficulty: "Advanced", relatedTopic: "rollbacks",
    symptom: "Aggregate metrics fine; wallet chargebacks up", objective: "Find a regression hidden by aggregate metrics and fix it with a rollback plus a slice-level release gate.",
    incident: "risk-v2 passed its canary on aggregate accuracy and was promoted to 100% last week. Chargebacks on digital-wallet orders have climbed since. The overall accuracy dashboard is unchanged.",
    architecture: DEPLOY_ARCH, run: deployRun,
    hypotheses: [
      { id: "h-labels", text: "v2's training labels lost wallet fraud: the new chargeback join uses card_id, which wallet orders don't have.", isRootCause: true, evidenceIds: ["slices", "labels", "weights", "deploys"], explanation: "v2's training data shows 0% fraud on wallet orders, v2 learned a strongly negative is_wallet weight, and its wallet recall collapsed while card recall improved." },
      { id: "h-device", text: "The new device_age_days feature is broken.", isRootCause: false, evidenceIds: ["device"], explanation: "It helps: card-order recall improved, and its definition matches between training and serving." },
      { id: "h-mix", text: "More orders are now paid by wallet (traffic mix shift).", isRootCause: false, evidenceIds: ["mix"], explanation: "Wallet share is the same as at launch." },
      { id: "h-serving", text: "The model server is misbehaving.", isRootCause: false, evidenceIds: ["health"], explanation: "Serving is healthy and the schema check passes." },
    ],
    fixes: [
      { id: "f-rollback", label: "Roll back to risk-v1 and add a per-slice recall gate (card, wallet) to the canary", explanation: "v1 still catches wallet fraud (aggregate accuracy returns to v1's level); the slice gate would have blocked v2. Next: fix the join and retrain." },
      { id: "f-threshold", label: "Keep v2 but lower its decision threshold to 0.3", explanation: "v2 learned that wallet orders are never fraud; a lower threshold flags more card orders but barely touches wallets." },
      { id: "f-retrain", label: "Retrain v2 for longer on a fresh sample from the same pipeline", explanation: "The same join produces the same missing wallet labels." },
    ],
    verifications: [
      { id: "v-slices", label: "Evaluate every candidate per slice (payment method, region) and require no slice to regress beyond tolerance before promotion", sound: true, explanation: "Aggregate metrics average a 18% slice away; slice gates don't." },
      { id: "v-agg", label: "Confirm aggregate accuracy is at least as good as before", sound: false, explanation: "That's the gate v2 passed." },
      { id: "v-latency", label: "Confirm inference latency and error rate are normal", sound: false, explanation: "They were normal throughout." },
    ],
    rootCause: "A rewritten training query joined chargebacks on card_id. Wallet orders have no card_id, so every wallet fraud case was labelled legitimate. risk-v2 learned that wallet orders are safe. Its new feature improved card orders enough that aggregate accuracy held, so the canary passed while wallet fraud recall collapsed.",
  },
];
