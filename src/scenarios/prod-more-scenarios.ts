import { Archive, Boxes, Cpu, Database, Gauge, Hourglass, Network, Server, ShieldCheck, Tag, User, Bot } from "lucide-react";
import { rng } from "../lib/random";
import type { SimulationEvent, TraceSpan } from "../types/simulation";
import { IN_TOKENS, QUALITY, requestCost, type ModelId } from "../visualizers/cost-optimization/simulation";
import { COLD_START_MS, SCALE_AT_MS, batchMs, runServing, type ServingConfig, type ServingRun } from "../visualizers/model-serving/simulation";
import { evaluateModel, fraudData, trainLogistic, type Row } from "./ml-model";
import { percentile } from "./prod-sim";
import type { Evidence, Outcome, Scenario } from "./types";

const ms = (v: number) => `${Math.round(v).toLocaleString("en-US")} ms`;
const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
const usd = (v: number) => `$${Math.round(v).toLocaleString("en-US")}`;
const hhmm = (sec: number) => `${String(Math.floor(sec / 3600) % 24).padStart(2, "0")}:${String(Math.floor((sec % 3600) / 60)).padStart(2, "0")}`;
let evN = 0;
const ev = (message: string, severity: SimulationEvent["severity"] = "info"): SimulationEvent => ({ id: `pm${evN++}`, type: "log", message, severity });
const evidence = (id: string, action: string, lines: string[], relevant: boolean, spanId?: string): Evidence => ({ id, action, title: action, lines, relevant, spanId });
type Fix<C> = Scenario["fixes"][number] & { cfg: Partial<C> };
const publicFixes = <C,>(fs: Fix<C>[]) => fs.map(({ id, label, explanation }) => ({ id, label, explanation }));

// ======================= autoscaler max replicas (Beginner) =======================
/**
 * One day of a self-hosted summarization service in 10 s ticks. Each pod serves POD_RPS; the autoscaler targets a
 * utilization and new pods take 60 s to load. Requests queue at the gateway (fluid model, Little's law W = L / X) and
 * time out with 504 once their wait would exceed the gateway timeout. Timed-out requests may be retried by clients.
 */
type ScaleCfg = { maxReplicas: number; target: number; timeoutS: number; clientRetry: boolean };
const POD_RPS = 30, DT = 10, COLD_TICKS = 6, MIN_PODS = 2, SCALE_SLO = 1000;
const SCALE_BASE: ScaleCfg = { maxReplicas: 16, target: 0.7, timeoutS: 10, clientRetry: false };
const SCALE_TODAY: ScaleCfg = { ...SCALE_BASE, maxReplicas: 6 };
const SCALE_FIXES: Fix<ScaleCfg>[] = [
  { id: "f-max", label: "Raise the autoscaler's max replicas back to 16 (and alert when desired replicas exceed the max)", explanation: "At peak the autoscaler asks for 13 pods and now gets them; the queue never builds.", cfg: { maxReplicas: 16 } },
  { id: "f-target", label: "Make the autoscaler more aggressive: target 50% utilization instead of 70%", explanation: "It already wants more pods than it is allowed. A lower target raises the desired count, but the cap still holds it at 6.", cfg: { target: 0.5 } },
  { id: "f-timeout", label: "Raise the gateway timeout from 10 s to 30 s so fewer requests fail", explanation: "Fewer 504s, but the queue simply grows longer: requests wait up to 30 s instead of failing at 10 s.", cfg: { timeoutS: 30 } },
  { id: "f-retry", label: "Have clients retry timed-out requests automatically", explanation: "Every retry is another arrival into a queue that is already over capacity; more requests time out.", cfg: { clientRetry: true } },
];
const rpsAt = (sec: number) => { const h = sec / 3600; return 40 + 200 * Math.max(0, Math.sin(((h - 7) / 14) * Math.PI)); };

function simulateScaling(cfg: ScaleCfg, seed: number) {
  const r = rng(seed);
  const ticks: { t: number; rps: number; desired: number; ready: number; queue: number; wait: number; dropped: number; arrivals: number }[] = [];
  const samples: { t: number; ms: number; ok: boolean; wait: number; service: number }[] = [];
  const pending: number[] = []; // tick at which each requested pod becomes ready
  let ready = MIN_PODS, queue = 0, retries = 0;
  for (let i = 0; i < (24 * 3600) / DT; i++) {
    const t = i * DT, rps = rpsAt(t) * (1 + 0.03 * r.gaussian());
    const desired = Math.min(cfg.maxReplicas, Math.max(MIN_PODS, Math.ceil(rps / (POD_RPS * cfg.target))));
    while (pending.length && pending[0] <= i) { pending.shift(); ready++; }
    if (ready + pending.length < desired) for (let k = ready + pending.length; k < desired; k++) pending.push(i + COLD_TICKS);
    if (ready > desired && !pending.length) ready = Math.max(desired, ready - 1); // scale down one pod per tick
    const cap = ready * POD_RPS, arrivals = rps * DT + retries;
    const served = Math.min(queue + arrivals, cap * DT);
    queue += arrivals - served;
    const dropped = Math.max(0, queue - cap * cfg.timeoutS);
    queue -= dropped;
    retries = cfg.clientRetry ? dropped : 0;
    const wait = (queue / cap) * 1000;
    ticks.push({ t, rps, desired: Math.min(cfg.maxReplicas, Math.max(MIN_PODS, Math.ceil(rps / (POD_RPS * cfg.target)))), ready, queue, wait, dropped, arrivals });
    for (let k = 0; k < Math.round(rps / 8); k++) {
      const service = 180 * Math.exp(0.25 * r.gaussian());
      const ok = r.next() >= dropped / arrivals;
      samples.push({ t, ms: ok ? wait * r.next() * 2 + service : cfg.timeoutS * 1000, ok, wait: ok ? wait : cfg.timeoutS * 1000, service });
    }
  }
  return { ticks, samples };
}

const SCALE_ARCH: Scenario["architecture"] = {
  nodes: [
    { id: "client", label: "Clients", Icon: User, x: 0, y: 100, match: (s) => s.id === "req" },
    { id: "gateway", label: "API gateway", Icon: Network, x: 160, y: 100, match: (s) => s.id === "gateway" },
    { id: "queue", label: "Request queue", Icon: Hourglass, x: 330, y: 100, match: (s) => s.id === "queue" },
    { id: "pods", label: "Model pods (GPU)", Icon: Boxes, x: 510, y: 100, match: (s) => s.id === "infer" },
    { id: "hpa", label: "Autoscaler", Icon: Gauge, x: 510, y: 230, match: () => false },
  ],
  edges: [{ source: "client", target: "gateway" }, { source: "gateway", target: "queue" }, { source: "queue", target: "pods" }, { source: "hpa", target: "pods", sourceHandle: "r", targetHandle: "l" }],
};

function scalingRun(fixId?: string): Outcome {
  const cfg = { ...SCALE_TODAY, ...(SCALE_FIXES.find((f) => f.id === fixId)?.cfg ?? {}) };
  const { ticks, samples } = simulateScaling(cfg, 5);
  const lastWeek = simulateScaling(SCALE_BASE, 4);
  const lat = (xs: typeof samples) => xs.map((s) => s.ms);
  const p95 = percentile(lat(samples), 0.95), fail = samples.filter((s) => !s.ok).length / samples.length;
  const band = (xs: typeof samples, a: number, b: number) => xs.filter((s) => s.t >= a * 3600 && s.t < b * 3600);
  const peak = ticks.reduce((m, t) => (t.rps > m.rps ? t : m), ticks[0]);
  const peakRps = Math.max(...ticks.map((t) => t.rps)), peakRpsLw = Math.max(...lastWeek.ticks.map((t) => t.rps));
  const capped = ticks.filter((t) => Math.ceil(t.rps / (POD_RPS * cfg.target)) > cfg.maxReplicas);
  const resolved = p95 <= SCALE_SLO && fail < 0.005;
  const rep = [...samples].filter((s) => s.ok).sort((a, b) => b.ms - a.ms)[Math.floor(samples.length * 0.03)] ?? samples[0];
  const repTick = ticks[Math.round(rep.t / DT)];
  const qWait = rep.ms - rep.service;
  const spans: TraceSpan[] = [
    { id: "req", name: "POST /summarize", component: "api", startMs: 0, durationMs: Math.round(rep.ms + 4), status: rep.ms > SCALE_SLO ? "warning" : "ok", output: { http: 200 }, events: rep.ms > SCALE_SLO ? [ev(`latency ${ms(rep.ms)} exceeds SLO ${SCALE_SLO} ms`, "warning")] : [] },
    { id: "gateway", parentId: "req", name: "gateway", component: "api-gateway", startMs: 0, durationMs: 2, status: "ok", metrics: { rps: Math.round(repTick.rps) }, events: [] },
    { id: "queue", parentId: "req", name: "queue: wait for a free pod", component: "gateway-queue", startMs: 2, durationMs: Math.round(qWait), status: qWait > 500 ? "warning" : "ok", metrics: { queue_depth: Math.round(repTick.queue), ready_pods: repTick.ready }, events: qWait > 500 ? [ev(`${Math.round(repTick.queue)} requests ahead in the queue; ${repTick.ready} pods busy`, "warning")] : [] },
    { id: "infer", parentId: "req", name: "summarize (model pod)", component: "model-pod", startMs: Math.round(2 + qWait), durationMs: Math.round(rep.service), status: "ok", input: { model: "summarizer-v4", pod: `summarizer-${(rep.t % 7) + 1}` }, events: [] },
    { id: "post", parentId: "req", name: "format response", component: "api", startMs: Math.round(2 + rep.ms), durationMs: 2, status: "ok", events: [] },
  ];
  const svc = samples.map((s) => s.service), svcLw = lastWeek.samples.map((s) => s.service);
  return {
    spans,
    metrics: [
      { label: "p95 latency (24 h)", value: ms(p95), status: p95 <= SCALE_SLO ? "ok" : "error", kind: "health", hint: `SLO ${SCALE_SLO} ms` },
      { label: "504 timeouts", value: pct(fail, 2), status: fail < 0.005 ? "ok" : "error", kind: "health" },
      { label: "Pods at peak", value: `${Math.max(...ticks.map((t) => t.ready))} / ${cfg.maxReplicas} max`, status: capped.length ? "warning" : "ok", kind: "health" },
      { label: "Summary quality (sampled)", value: "4.6 / 5", status: "ok", kind: "quality", hint: "human-rated sample of served summaries" },
    ],
    transcript: [{ label: "Customer ticket", text: resolved ? "Summaries are quick again this afternoon, thanks." : `Every afternoon summaries take ages and some fail with “504 Gateway Timeout”. Mornings are fine.`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: [
      evidence("dashboards", "Check latency and error dashboards by time of day", [[0, 9, "00:00–09:00"], [9, 18, "09:00–18:00"], [18, 24, "18:00–24:00"]].map(([a, b, l]) => { const xs = band(samples, a as number, b as number); return `${l}: p95 ${ms(percentile(lat(xs), 0.95))}, 504s ${pct(xs.filter((s) => !s.ok).length / xs.length, 2)}`; }), false),
      evidence("replicas", "Compare desired vs running pods over the day", [`peak traffic ${Math.round(peakRps)} req/s needs ${Math.ceil(peakRps / (POD_RPS * cfg.target))} pods at the ${pct(cfg.target, 0)} target (${POD_RPS} req/s per pod)`, `running pods at peak: ${Math.max(...ticks.map((t) => t.ready))}`, capped.length ? `desired > max_replicas (${cfg.maxReplicas}) from ${hhmm(capped[0].t)} to ${hhmm(capped.at(-1)!.t)}` : `desired never exceeded max_replicas (${cfg.maxReplicas})`], true),
      evidence("queue", "Measure the request queue at the worst moment (Little's law)", [`at ${hhmm(peak.t)}: queue L = ${Math.round(peak.queue)} requests, throughput X = ${peak.ready} pods × ${POD_RPS} = ${peak.ready * POD_RPS} req/s, arrivals ${Math.round(peak.rps)} req/s`, `wait W = L / X = ${Math.round(peak.queue)} / ${peak.ready * POD_RPS} = ${(peak.queue / (peak.ready * POD_RPS)).toFixed(1)} s`, `arrivals exceed capacity by ${Math.round(Math.max(0, peak.rps - peak.ready * POD_RPS))} req/s, so the queue grows until requests hit the ${cfg.timeoutS} s gateway timeout`], true, "queue"),
      evidence("pods", "Check per-pod inference time and GPU utilization", [`inference time per request: p50 ${ms(percentile(svcLw, 0.5))} → ${ms(percentile(svc, 0.5))}, p95 ${ms(percentile(svcLw, 0.95))} → ${ms(percentile(svc, 0.95))} (last week → today)`, `GPU utilization at peak: ${pct(Math.min(1, peak.rps / (peak.ready * POD_RPS)), 0)} on every pod`, "no OOMs, no restarts"], true, "infer"),
      evidence("hpa", "Read the autoscaler events", [`${capped.length ? hhmm(capped[0].t) : "—"}: “desired replicas ${Math.ceil(peakRps / (POD_RPS * cfg.target))}, capped at maxReplicas ${cfg.maxReplicas}” (warning, logged on every 10 s sync until ${capped.length ? hhmm(capped.at(-1)!.t) : "—"})`, "scale-up decisions arrive within one 10 s sync period; new pods ready after 60 s", "autoscaler controller healthy"], true),
      evidence("traffic", "Compare traffic with the same day last week", [`peak ${Math.round(peakRpsLw)} → ${Math.round(peakRps)} req/s`, `daily requests ${Math.round(lastWeek.ticks.reduce((s, t) => s + t.rps * DT, 0)).toLocaleString("en-US")} → ${Math.round(ticks.reduce((s, t) => s + t.rps * DT, 0)).toLocaleString("en-US")}`, "top clients unchanged; no bot traffic flagged"], false),
      evidence("deps", "Check the document store the service reads from", ["doc-store p95 18 ms, unchanged", "error rate 0.0%"], false),
      evidence("model", "Check the deployed model version", ["summarizer-v4 since Sep 30 (unchanged)", "weekly quality review: 4.6 / 5, same as last week"], false),
      evidence("deploys", "Read the change log", ["Mon 09:10 infra cost review: summarizer HPA maxReplicas 16 → 6 (“average usage is 5 pods”)", "Mon 14:00 gateway: TLS certificate renewed", "Sep 30 summarizer-v4 released"], true),
    ],
  };
}

// ======================= cache key includes a timestamp (Beginner) =======================
/**
 * One day of a support assistant's traffic (sampled at 4,000 requests standing for 66,000). Questions repeat (Zipf)
 * plus a long tail of one-offs. The response cache key is a hash of either the normalized question or the full prompt.
 */
type CacheCfg = { promptHasTime: boolean; keyOn: "prompt" | "question"; ttlS: number; model: ModelId };
const CACHE_YDAY: CacheCfg = { promptHasTime: false, keyOn: "prompt", ttlS: 3600, model: "large" };
const CACHE_TODAY: CacheCfg = { ...CACHE_YDAY, promptHasTime: true };
const CACHE_FIXES: Fix<CacheCfg>[] = [
  { id: "f-key", label: "Build the cache key from the normalized question (plus doc version), not the full prompt, so the timestamp line doesn't change it", explanation: "Repeated questions hit the cache again; LLM calls and cost fall back to yesterday's level.", cfg: { keyOn: "question" } },
  { id: "f-ttl", label: "Raise the cache TTL from 1 hour to 24 hours", explanation: "Entries live longer, but no request ever produces the same key twice, so none is ever read.", cfg: { ttlS: 86400 } },
  { id: "f-small", label: "Route every request to the small model to bring the bill down", explanation: "The bill falls, but medium and hard questions get noticeably worse answers, and the cache still never hits.", cfg: { model: "small" } },
];
const DAILY = 66_000, N_CACHE = 4000, FAQS = 60;
const diffOf = (q: number) => (q % 20 < 12 ? 0 : q % 20 < 17 ? 1 : 2);
const outOf = (q: number) => [220, 480, 1000][diffOf(q)] + ((q * 37) % 60);

function simulateCache(cfg: CacheCfg, seed: number) {
  const r = rng(seed);
  const zipf = Array.from({ length: FAQS }, (_, k) => 1 / (k + 1)), zsum = zipf.reduce((s, v) => s + v, 0);
  const cache = new Map<string, number>();
  let reads = 0;
  const reqs = Array.from({ length: N_CACHE }, (_, i) => {
    const t = Math.floor(6 * 3600 + (i / N_CACHE) * 16 * 3600 + r.next() * 10);
    let q: number;
    if (r.next() < 0.3) q = FAQS + i; // one-off question
    else { let u = r.next() * zsum; q = zipf.findIndex((w) => (u -= w) < 0); }
    const question = `q${q}`;
    const prompt = `${cfg.promptHasTime ? `Current time: 2026-10-19T${hhmm(t)}:${String(t % 60).padStart(2, "0")}Z\n` : ""}You are the support assistant…\n${question}`;
    const key = cfg.keyOn === "question" ? `v3:${question}` : prompt;
    const exp = cache.get(key);
    const hit = exp !== undefined && exp > t;
    if (hit) reads++; else cache.set(key, t + cfg.ttlS);
    const inTok = IN_TOKENS + (cfg.promptHasTime ? 12 : 0), outTok = outOf(q);
    const cost = hit ? 0 : requestCost(cfg.model, inTok, outTok);
    const answeredBy: ModelId = hit ? "large" : cfg.model; // ponytail: assumes cached answers were made by the large model
    return { t, q, key, hit, inTok, outTok, cost, quality: QUALITY[answeredBy][diffOf(q)], ms: hit ? 8 : 300 + outTok * 2.2 * (cfg.model === "small" ? 0.4 : 1) };
  });
  const scale = DAILY / N_CACHE;
  const day = reqs.reduce((s, x) => s + x.cost, 0) * scale;
  return { reqs, day, hitRate: reqs.filter((x) => x.hit).length / reqs.length, entries: cache.size, reads, quality: reqs.reduce((s, x) => s + x.quality, 0) / reqs.length };
}

const CACHE_ARCH: Scenario["architecture"] = {
  nodes: [
    { id: "user", label: "User", Icon: User, x: 0, y: 100, match: (s) => s.id === "req" },
    { id: "prompt", label: "Prompt builder", Icon: Server, x: 160, y: 100, match: (s) => s.id === "prompt" },
    { id: "cache", label: "Response cache", Icon: Archive, x: 330, y: 20, match: (s) => s.id.startsWith("cache") },
    { id: "llm", label: "LLM provider", Icon: Bot, x: 330, y: 200, match: (s) => s.id === "llm" },
    { id: "billing", label: "Billing", Icon: Gauge, x: 520, y: 200, match: () => false },
  ],
  edges: [{ source: "user", target: "prompt" }, { source: "prompt", target: "cache" }, { source: "prompt", target: "llm" }, { source: "llm", target: "billing" }],
};

function cacheRun(fixId?: string): Outcome {
  const cfg = { ...CACHE_TODAY, ...(CACHE_FIXES.find((f) => f.id === fixId)?.cfg ?? {}) };
  const today = simulateCache(cfg, 12), yday = simulateCache(CACHE_YDAY, 11);
  const budget = yday.day * 1.15;
  const resolved = today.day <= budget && today.quality >= 0.9;
  const pw = today.reqs.filter((x) => x.q === 1); // “How do I reset my password?”
  const rep = pw[3] ?? today.reqs[0];
  const lat = (rs: typeof today.reqs, q: number) => percentile(rs.map((x) => x.ms), q);
  const hourly = (rs: typeof today.reqs, h: number) => { const xs = rs.filter((x) => Math.floor(x.t / 3600) === h); return xs.length ? pct(xs.filter((x) => x.hit).length / xs.length, 0) : "—"; };
  const tok = (rs: typeof today.reqs, f: (x: (typeof rs)[number]) => number) => Math.round(rs.reduce((s, x) => s + f(x), 0) / rs.length);
  const shortKey = (k: string) => (k.startsWith("v3:") ? k : `sha256(full prompt) of “${k.split("\n")[0].slice(0, 34)}…”`);
  const spans: TraceSpan[] = [
    { id: "req", name: "POST /chat", component: "api", startMs: 0, durationMs: Math.round(rep.ms + 6), status: "ok", input: { question: "How do I reset my password?" }, output: { http: 200, cost_usd: Number(rep.cost.toFixed(4)) }, events: [] },
    { id: "prompt", parentId: "req", name: "build prompt (template v8)", component: "prompt-builder", startMs: 0, durationMs: 2, status: "ok", output: { first_line: cfg.promptHasTime ? `Current time: …${hhmm(rep.t)}Z` : "You are the support assistant…", tokens: rep.inTok }, events: [] },
    { id: "cache.get", parentId: "req", name: "cache lookup", component: "response-cache", startMs: 2, durationMs: 2, status: rep.hit ? "ok" : "warning", input: { key: shortKey(rep.key) }, output: { hit: rep.hit }, events: rep.hit ? [] : [ev("cache miss", "warning")] },
  ];
  if (!rep.hit) {
    spans.push({ id: "llm", parentId: "req", name: `generate (${cfg.model} model)`, component: "llm-provider", startMs: 4, durationMs: Math.round(rep.ms), status: "ok", metrics: { input_tokens: rep.inTok, output_tokens: rep.outTok }, events: [] });
    spans.push({ id: "cache.set", parentId: "req", name: "cache write", component: "response-cache", startMs: Math.round(4 + rep.ms), durationMs: 2, status: "ok", input: { ttl_s: cfg.ttlS }, events: [] });
  }
  return {
    spans,
    metrics: [
      { label: "LLM spend (today, projected)", value: usd(today.day), status: today.day <= budget ? "ok" : "error", kind: "health", hint: `budget ${usd(budget)} / day` },
      { label: "Cache hit rate", value: pct(today.hitRate, 0), status: today.hitRate >= 0.3 ? "ok" : "error", kind: "health" },
      { label: "p50 latency", value: ms(lat(today.reqs, 0.5)), status: lat(today.reqs, 0.5) <= lat(yday.reqs, 0.5) * 1.5 ? "ok" : "warning", kind: "health" },
      { label: "Answer quality (graded)", value: pct(today.quality, 0), status: today.quality >= 0.9 ? "ok" : "error", kind: "quality" },
    ],
    transcript: [{ label: "Finance", text: resolved ? `Projected LLM spend today ${usd(today.day)}, within budget.` : `Projected LLM spend today is ${usd(today.day)} (yesterday ${usd(yday.day)}). Did traffic grow?`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: [
      evidence("billing", "Read the provider invoice for today", [`projected spend ${usd(yday.day)} → ${usd(today.day)} per day`, "per-token prices unchanged: $3 / 1M input, $15 / 1M output (large model)", "no new models or features billed"], false),
      evidence("calls", "Count LLM calls per request", [`yesterday: ${Math.round((1 - yday.hitRate) * DAILY).toLocaleString("en-US")} LLM calls for ${DAILY.toLocaleString("en-US")} requests`, `today: ${Math.round((1 - today.hitRate) * DAILY).toLocaleString("en-US")} LLM calls for ${DAILY.toLocaleString("en-US")} requests`], true, "llm"),
      evidence("hits", "Plot cache hit rate by hour", [9, 12, 15, 18].map((h) => `${h}:00 hit rate: yesterday ${hourly(yday.reqs, h)}, today ${hourly(today.reqs, h)}`), true, "cache.get"),
      evidence("keys", "Compare cache keys for the same question", [`“How do I reset my password?” asked ${pw.length} times today`, ...pw.slice(0, 3).map((x) => `${hhmm(x.t)} → key ${shortKey(x.key)}${x.hit ? " (hit)" : " (miss)"}`), `distinct keys for this one question: ${new Set(pw.map((x) => x.key)).size}`], true, "cache.get"),
      evidence("store", "Check cache cluster health", ["Redis: 0 errors, latency 0.4 ms, memory 41%", `entries written today: ${today.entries.toLocaleString("en-US")} for ${N_CACHE.toLocaleString("en-US")} sampled requests; entries ever read: ${today.reads}`], true, "cache.set"),
      evidence("tokens", "Compare tokens per request", [`input tokens per call: ${tok(yday.reqs, (x) => x.inTok)} → ${tok(today.reqs, (x) => x.inTok)}`, `output tokens per call: ${tok(yday.reqs, (x) => x.outTok)} → ${tok(today.reqs, (x) => x.outTok)}`], false, "llm"),
      evidence("traffic", "Compare request volume with yesterday", [`requests: ${DAILY.toLocaleString("en-US")} → ${DAILY.toLocaleString("en-US")} (flat)`, "share of repeated questions: 70% both days"], false),
      evidence("quality", "Grade a sample of today's answers", [`graded quality: yesterday ${pct(yday.quality, 0)}, today ${pct(today.quality, 0)}`, "no complaints about answer content"], false),
      evidence("deploys", "Read the change log", ["08:00 prompt template v8: first line “Current time: <ISO timestamp>” so the bot can answer “is support open now?”", "Mar 02 response cache: key = sha256(full prompt), TTL 1 h", "07:30 web-chat: new emoji picker"], true),
    ],
  };
}

// ======================= rate limit keyed on the load balancer's IP (Beginner) =======================
/** Five minutes of per-second traffic from 12 tenants through a token-bucket limiter (rate and burst per key). */
type RLCfg = { keyBy: "remote_addr" | "api_key"; behindLb: boolean; rate: number; clientRetry: boolean };
const TENANTS: [string, number][] = [["acme", 42], ["globex", 38], ["initech", 35], ["umbrella", 30], ["hooli", 28], ["stark", 25], ["wayne", 22], ["wonka", 20], ["tyrell", 18], ["cyberdyne", 15], ["soylent", 10], ["piedpiper", 3]];
const LB_IP = "10.0.4.12";
const RL_TODAY: RLCfg = { keyBy: "remote_addr", behindLb: true, rate: 60, clientRetry: false };
const RL_FIXES: Fix<RLCfg>[] = [
  { id: "f-key", label: "Key the limiter on the API key (the tenant), not the connection's source IP", explanation: "Each tenant gets its own 60 req/s bucket again; every tenant is under quota, so 429s disappear.", cfg: { keyBy: "api_key" } },
  { id: "f-raise", label: "Double the rate limit to 120 req/s", explanation: "All tenants still share one bucket. 120 req/s is still far below the ~290 req/s they send together.", cfg: { rate: 120 } },
  { id: "f-retry", label: "Tell clients to retry 429s with backoff", explanation: "Retries add more requests to the same shared bucket; the 429 rate goes up, not down.", cfg: { clientRetry: true } },
];

function simulateLimiter(cfg: RLCfg, seed: number) {
  const r = rng(seed);
  const buckets = new Map<string, number>();
  const per = TENANTS.map(() => ({ sent: 0, limited: 0 }));
  let retry = TENANTS.map(() => 0);
  const keyOf = (i: number) => (cfg.keyBy === "api_key" ? `key:${TENANTS[i][0]}` : cfg.behindLb ? LB_IP : `203.0.113.${10 + i}`);
  for (let t = 0; t < 300; t++) {
    const counts = TENANTS.map(([, rate], i) => Math.max(0, Math.round(rate + Math.sqrt(rate) * r.gaussian())) + retry[i]);
    const byKey = new Map<string, number[]>();
    counts.forEach((_, i) => byKey.set(keyOf(i), [...(byKey.get(keyOf(i)) ?? []), i]));
    const next = TENANTS.map(() => 0);
    byKey.forEach((members, key) => {
      const tokens = Math.min(cfg.rate * 2, (buckets.get(key) ?? cfg.rate * 2) + (t ? cfg.rate : 0));
      const want = members.reduce((s, i) => s + counts[i], 0);
      const allowed = Math.min(want, Math.floor(tokens));
      buckets.set(key, tokens - allowed);
      members.forEach((i) => {
        const lim = want ? Math.round(counts[i] * (1 - allowed / want)) : 0;
        per[i].sent += counts[i]; per[i].limited += lim;
        next[i] = cfg.clientRetry ? lim : 0;
      });
    });
    retry = next;
  }
  const sent = per.reduce((s, p) => s + p.sent, 0), limited = per.reduce((s, p) => s + p.limited, 0);
  return { per, rate429: limited / sent, sent, keys: new Set(TENANTS.map((_, i) => keyOf(i))) };
}

const RL_ARCH: Scenario["architecture"] = {
  nodes: [
    { id: "tenants", label: "12 tenants", Icon: User, x: 0, y: 100, match: (s) => s.id === "req" },
    { id: "lb", label: "L7 load balancer", Icon: Network, x: 160, y: 100, match: (s) => s.id === "lb" },
    { id: "limiter", label: "Rate limiter", Icon: ShieldCheck, x: 330, y: 100, match: (s) => s.id === "ratelimit" },
    { id: "api", label: "Completion API", Icon: Server, x: 500, y: 100, match: (s) => s.id === "api" },
    { id: "llm", label: "LLM provider", Icon: Bot, x: 500, y: 230, match: (s) => s.id === "llm" },
  ],
  edges: [{ source: "tenants", target: "lb" }, { source: "lb", target: "limiter" }, { source: "limiter", target: "api" }, { source: "api", target: "llm", sourceHandle: "b", targetHandle: "t" }],
};

function limiterRun(fixId?: string): Outcome {
  const cfg = { ...RL_TODAY, ...(RL_FIXES.find((f) => f.id === fixId)?.cfg ?? {}) };
  const today = simulateLimiter(cfg, 21), yday = simulateLimiter({ ...RL_TODAY, behindLb: false }, 20);
  const resolved = today.rate429 < 0.01;
  const pp = today.per[TENANTS.length - 1];
  const limitedRep = pp.limited / Math.max(1, pp.sent) > 0.5;
  const key = cfg.keyBy === "api_key" ? "key:piedpiper" : LB_IP;
  const spans: TraceSpan[] = [
    { id: "req", name: "POST /v1/complete (tenant piedpiper)", component: "client", startMs: 0, durationMs: limitedRep ? 5 : 642, status: limitedRep ? "error" : "ok", output: { http: limitedRep ? 429 : 200 }, events: [] },
    { id: "lb", parentId: "req", name: "L7 load balancer", component: "load-balancer", startMs: 0, durationMs: 2, status: "ok", output: { "X-Forwarded-For": "203.0.113.21", upstream_remote_addr: cfg.behindLb ? LB_IP : "203.0.113.21" }, events: [] },
    { id: "ratelimit", parentId: "req", name: "rate limit check", component: "api-gateway", startMs: 2, durationMs: 1, status: limitedRep ? "error" : "ok", input: { key_by: cfg.keyBy, key, limit: `${cfg.rate}/s` }, output: { remaining: limitedRep ? 0 : 41 }, events: limitedRep ? [ev(`429 Too Many Requests: bucket “${key}” empty`, "error")] : [] },
  ];
  if (!limitedRep) {
    spans.push({ id: "api", parentId: "req", name: "completion handler", component: "api", startMs: 3, durationMs: 637, status: "ok", events: [] });
    spans.push({ id: "llm", parentId: "api", name: "generate", component: "llm-provider", startMs: 8, durationMs: 610, status: "ok", events: [] });
  }
  const show = [0, 5, 10, 11];
  return {
    spans,
    metrics: [
      { label: "429 rate", value: pct(today.rate429), status: today.rate429 < 0.01 ? "ok" : "error", kind: "health" },
      { label: "Tenants over their quota", value: "0 / 12", status: "ok", kind: "health", hint: "quota 60 req/s per tenant" },
      { label: "p95 latency (allowed requests)", value: ms(642), status: "ok", kind: "health" },
      { label: "Completed requests", value: pct(1 - today.rate429), status: today.rate429 < 0.01 ? "ok" : "error", kind: "quality" },
    ],
    transcript: [{ label: "Customer (piedpiper)", text: limitedRep ? "We send about 3 requests a second and our plan allows 60. Why are most of them getting 429 Too Many Requests?" : "429s are gone, thanks.", tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: [
      evidence("dashboards", "Check the 429 dashboard", [`429 rate: yesterday ${pct(yday.rate429)}, today ${pct(today.rate429)} (since 09:00)`, "5xx: 0.0%"], false),
      evidence("bytenant", "Break 429s down by tenant", show.map((i) => `${TENANTS[i][0]}: sends ${TENANTS[i][1]} req/s (quota 60), ${pct(today.per[i].limited / today.per[i].sent, 0)} rejected`), true),
      evidence("keys", "List the limiter's active bucket keys", [`yesterday: ${yday.keys.size} distinct keys (${[...yday.keys].slice(0, 2).join(", ")}, …)`, `today: ${today.keys.size} distinct key${today.keys.size === 1 ? "" : "s"} (${[...today.keys].slice(0, 2).join(", ")}${today.keys.size > 2 ? ", …" : ""})`, `${LB_IP} is the internal address of the new load balancer`], true, "ratelimit"),
      evidence("headers", "Inspect a 429 response", ["returned by: api-gateway (not the LLM provider)", `X-RateLimit-Limit: ${cfg.rate}`, "X-RateLimit-Remaining: 0", `X-RateLimit-Key: ${cfg.keyBy}`], true, "ratelimit"),
      evidence("config", "Read the rate limiter configuration", [`key: ${cfg.keyBy}`, `rate: ${cfg.rate} req/s, burst ${cfg.rate * 2}`, "last changed: Jun 12 (unchanged since)"], true),
      evidence("provider", "Check the LLM provider's rate limits", ["provider 429s today: 0", "provider quota used at peak: 31%", "status page: all systems operational"], false, "llm"),
      evidence("traffic", "Compare tenant traffic with yesterday", [`total ${Math.round(yday.sent / 300)} → ${Math.round(today.sent / 300)} req/s offered`, `largest tenant: ${TENANTS[0][0]} at ${TENANTS[0][1]} req/s; no tenant above its 60 req/s quota`], false),
      evidence("latency", "Check latency of allowed requests", ["p50 520 ms, p95 642 ms (unchanged)"], false, "api"),
      evidence("deploys", "Read the change log", ["09:00 network: traffic moved to the new L7 load balancer (it proxies connections; the client IP is in X-Forwarded-For)", "08:40 api: completion handler 2.8 (logging)", "Yesterday: piedpiper signed up (Starter plan)"], true),
    ],
  };
}

// ======================= cold replica receives traffic (Advanced) =======================
/** Replays one autoscaler scale-out on the model-serving simulation; `routeBeforeReady` = readiness probe on /healthz. */
const SERVE_SLO = 150;
const SERVE_TODAY: ServingConfig = { arrivalRate: 300, replicas: 2, maxBatch: 16, maxWaitMs: 10, scaleOut: true, routeBeforeReady: true, seed: 9 };
const SERVE_FIXES: Fix<ServingConfig>[] = [
  { id: "f-probe", label: "Point the readiness probe at /v1/ready, which only passes once the weights are loaded and a warm-up batch has run", explanation: "The load balancer waits until the new replica can actually serve; scale-outs no longer create a slow burst.", cfg: { routeBeforeReady: false } },
  { id: "f-batch", label: "Revert max batch size from 16 to 8", explanation: "Batches were rarely full, so the batch size barely mattered. Requests still queue on the loading replica.", cfg: { maxBatch: 8 } },
  { id: "f-replicas", label: "Run 3 replicas at minimum so there's more headroom", explanation: "The autoscaler still adds a replica when load rises, and the new one still gets a quarter of traffic before its weights are loaded.", cfg: { replicas: 3 } },
  { id: "f-wait", label: "Lower max batch wait from 10 ms to 2 ms", explanation: "Shaves a few ms off healthy replicas. The slow requests are waiting for a model that isn't loaded yet.", cfg: { maxWaitMs: 2 } },
];

const SERVE_ARCH: Scenario["architecture"] = {
  nodes: [
    { id: "client", label: "Clients", Icon: User, x: 0, y: 110, match: (s) => s.id === "req" },
    { id: "lb", label: "Load balancer", Icon: Network, x: 160, y: 110, match: (s) => s.id === "lb" },
    { id: "warm", label: "Warm replicas", Icon: Cpu, x: 340, y: 20, match: () => false },
    { id: "new", label: "New replica", Icon: Cpu, x: 340, y: 200, match: (s) => s.id === "queue" || s.id === "batch" },
    { id: "hpa", label: "Autoscaler", Icon: Gauge, x: 520, y: 200, match: () => false },
    { id: "probe", label: "Readiness probe", Icon: ShieldCheck, x: 520, y: 20, match: (s) => s.id === "probe" },
  ],
  edges: [{ source: "client", target: "lb" }, { source: "lb", target: "warm" }, { source: "lb", target: "new" }, { source: "hpa", target: "new", sourceHandle: "r", targetHandle: "l" }, { source: "probe", target: "lb", sourceHandle: "r", targetHandle: "t" }],
};

function servingRun(fixId?: string): Outcome {
  const cfg = { ...SERVE_TODAY, ...(SERVE_FIXES.find((f) => f.id === fixId)?.cfg ?? {}) };
  const run = runServing(cfg), steady = runServing({ ...cfg, scaleOut: false });
  const lat = (rn: ServingRun) => rn.reqs.map((q) => q.end - q.arrive);
  const p = (rn: ServingRun, q: number) => percentile(lat(rn), q);
  const fresh = run.replicas.at(-1)!;
  const onFresh = run.reqs.filter((q) => q.replica === fresh.id);
  const slow = run.reqs.filter((q) => q.end - q.arrive > SERVE_SLO);
  const resolved = p(run, 0.95) <= SERVE_SLO;
  const rep = [...onFresh].sort((a, b) => b.end - b.arrive - (a.end - a.arrive))[Math.floor(onFresh.length * 0.1)] ?? run.reqs[0];
  const early = onFresh.filter((q) => q.arrive < fresh.readyAt);
  const full = run.batches.filter((b) => b.reason === "full").length;
  const sizes = run.batches.map((b) => b.size);
  const freshBatch = run.batches.filter((b) => b.replica === fresh.id && b.start >= fresh.readyAt + 50);
  const warmBatch = run.batches.filter((b) => b.replica !== fresh.id);
  const avg = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / Math.max(1, xs.length);
  const routedBefore = rep.arrive < fresh.readyAt;
  const spans: TraceSpan[] = [
    { id: "req", name: "POST /v1/embed", component: "client", startMs: 0, durationMs: Math.round(rep.end - rep.arrive), status: rep.end - rep.arrive > SERVE_SLO ? "warning" : "ok", output: { http: 200 }, events: [] },
    { id: "lb", parentId: "req", name: "load balancer: pick replica", component: "load-balancer", startMs: 0, durationMs: 1, status: "ok", output: { replica: `replica-${rep.replica + 1}`, readiness: "Ready" }, events: [] },
    { id: "probe", parentId: "lb", name: "readiness (last probe)", component: "kubelet", startMs: 0, durationMs: 1, status: "ok", input: { path: cfg.routeBeforeReady ? "/healthz" : "/v1/ready" }, output: { http: 200 }, events: [] },
    { id: "queue", parentId: "req", name: `queue on replica-${rep.replica + 1}`, component: "model-server", startMs: 1, durationMs: Math.round(rep.start - rep.arrive), status: rep.start - rep.arrive > 100 ? "warning" : "ok", events: routedBefore ? [ev(`replica-${rep.replica + 1} still loading weights (${Math.round((rep.arrive - fresh.createdAt) / COLD_START_MS * 100)}% done when the request arrived)`, "warning")] : [] },
    { id: "batch", parentId: "req", name: `GPU batch (${run.batches[rep.batch]?.size ?? 1} requests)`, component: "gpu", startMs: Math.round(1 + rep.start - rep.arrive), durationMs: Math.round(rep.end - rep.start), status: "ok", events: [] },
  ];
  return {
    spans,
    metrics: [
      { label: "p95 latency (scale-out window)", value: ms(p(run, 0.95)), status: p(run, 0.95) <= SERVE_SLO ? "ok" : "error", kind: "health", hint: `SLO ${SERVE_SLO} ms` },
      { label: "p50 latency", value: ms(p(run, 0.5)), status: "ok", kind: "health" },
      { label: "Error rate", value: "0.0%", status: "ok", kind: "health" },
      { label: "Embedding parity vs reference", value: "100%", status: "ok", kind: "quality", hint: "outputs match the offline model bit-for-bit" },
    ],
    transcript: [{ label: "Search team", text: resolved ? "No latency blips during this afternoon's scale-outs." : `Every time embed-svc scales out we get a burst of embedding calls taking ${ms(p(run, 0.99))}+, and our search times out. It started after last week's release.`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: [
      evidence("dashboards", "Compare latency with and without a scale-out", [`replay with a scale-out at +${SCALE_AT_MS} ms: p50 ${ms(p(run, 0.5))}, p95 ${ms(p(run, 0.95))}, p99 ${ms(p(run, 0.99))}`, `same traffic, no scale-out: p50 ${ms(p(steady, 0.5))}, p95 ${ms(p(steady, 0.95))}, p99 ${ms(p(steady, 0.99))}`], true),
      evidence("byreplica", "Break latency down by replica", [...run.replicas.map((rp) => { const xs = run.reqs.filter((q) => q.replica === rp.id).map((q) => q.end - q.arrive); return `replica-${rp.id + 1}${rp.createdAt ? " (new)" : ""}: ${xs.length} requests, p50 ${ms(percentile(xs, 0.5))}, p95 ${ms(percentile(xs, 0.95))}`; }), `${pct(slow.filter((q) => q.replica === fresh.id).length / Math.max(1, slow.length), 0)} of requests over ${SERVE_SLO} ms were served by replica-${fresh.id + 1}`], true, "queue"),
      evidence("timeline", "Line up slow requests with the new replica's lifecycle", [`replica-${fresh.id + 1}: pod created +${fresh.createdAt} ms, readiness passed +${cfg.routeBeforeReady ? fresh.createdAt + 5 : fresh.readyAt} ms, weights loaded +${fresh.readyAt} ms`, `requests routed to it before weights loaded: ${early.length} (they waited up to ${ms(Math.max(0, ...early.map((q) => q.start - q.arrive)))})`, `slowest request arrived at +${Math.round([...run.reqs].sort((a, b) => b.end - b.arrive - (a.end - a.arrive))[0].arrive)} ms`], true, "queue"),
      evidence("probe", "Inspect the readiness probe", [`readinessProbe: GET ${cfg.routeBeforeReady ? "/healthz" : "/v1/ready"} every 1 s`, cfg.routeBeforeReady ? "/healthz returns 200 as soon as the HTTP server starts; it doesn't check the model" : "/v1/ready returns 503 until weights are on the GPU and a warm-up batch ran", `model load time on a fresh node: ${COLD_START_MS} ms (weights 1.3 GB)`], true, "probe"),
      evidence("batches", "Look at batch sizes and GPU time per batch", [`max batch ${cfg.maxBatch}; mean batch ${avg(sizes).toFixed(1)}; ${pct(full / run.batches.length, 0)} of batches closed full`, `GPU time per batch: ${batchMs(avg(sizes)).toFixed(1)} ms (warm replicas ${avg(warmBatch.map((b) => b.end - b.start)).toFixed(1)} ms, new replica once loaded ${avg(freshBatch.map((b) => b.end - b.start)).toFixed(1)} ms)`], false, "batch"),
      evidence("gpu", "Check GPU node health", ["all nodes: same GPU model, driver 550.54, no ECC errors", "new replica's node: GPU at 0% for its first few hundred ms, then normal"], false),
      evidence("traffic", "Check incoming traffic during the window", [`${cfg.arrivalRate} req/s steady (${run.reqs.length} requests in the ${(1200 / 1000).toFixed(1)} s replay)`, "autoscaler fired on GPU utilization > 70%, as configured"], false),
      evidence("deploys", "Read the change log", ["last week embed-svc chart 3.0: max_batch_size 8 → 16 (throughput)", "last week embed-svc chart 3.0: probes consolidated: readiness and liveness both use /healthz", "Oct 10 HPA: scale on GPU utilization"], true),
    ],
  };
}

// ======================= mutable "latest" tag + autoscaling (Advanced) =======================
/**
 * One day of a fraud-scoring deployment. Pods pull `fraud-model:<tag>` when they are created. At 10:07 a staging CI job
 * pushes a release candidate to `:latest`; only pods created after that (autoscaled at peak) get it. The candidate was
 * trained for feature schema v2 (amount in cents instead of dollars), which production doesn't send yet.
 */
type TagCfg = { pinDigest: boolean; restartAt?: number; maxReplicas: number; autoscale: boolean };
const TAG_TODAY: TagCfg = { pinDigest: false, maxReplicas: 10, autoscale: true };
const TAG_FIXES: Fix<TagCfg>[] = [
  { id: "f-pin", label: "Reference the image by immutable digest (fraud-model@sha256:9f2c…, v7) and roll the deployment", explanation: "Every pod, including ones the autoscaler adds later, runs v7; CI pushing to :latest no longer changes production.", cfg: { pinDigest: true, restartAt: 12 } },
  { id: "f-restart", label: "Rolling-restart the deployment so all pods are identical", explanation: "They become identical: every pod re-pulls :latest and now runs the release candidate. Recall drops on every pod.", cfg: { restartAt: 12 } },
  { id: "f-noscale", label: "Disable the autoscaler and run a fixed 3 replicas", explanation: "No new pods pull the candidate, but 3 pods can't carry the afternoon peak: latency blows through the SLO.", cfg: { autoscale: false, restartAt: 12 } },
  { id: "f-more", label: "Raise max replicas to 14 so pods are less loaded at peak", explanation: "Pods aren't overloaded, so the autoscaler never needs the extra headroom; any pod it does add still pulls the candidate.", cfg: { maxReplicas: 14 } },
];
const TAG_SLO = 250, TAG_POD_RPS = 100, RC_PUSH = 10 + 7 / 60, THRESH = 0.3;
const toV2 = (rows: Row[]): Row[] => rows.map((r) => ({ ...r, x: [r.x[0] * 100, ...r.x.slice(1)] })); // schema v2: amount_cents

let models: { v7: ReturnType<typeof trainLogistic>; rc: ReturnType<typeof trainLogistic> } | undefined;
const getModels = () => (models ??= {
  v7: trainLogistic(fraudData(31, 4000), { positiveWeight: 6 }),
  rc: trainLogistic({ features: ["amount_cents", "is_foreign", "night_hour", "txn_velocity_1h"], rows: toV2(fraudData(32, 4000).rows) }, { positiveWeight: 6 }),
});
const tagRps = (h: number) => 100 + 340 * Math.max(0, Math.sin(((h - 9) / 13) * Math.PI));

function simulateTag(cfg: TagCfg) {
  const { v7, rc } = getModels();
  const traffic = fraudData(33, 12000).rows;
  type Pod = { id: number; created: number; image: "v7" | "rc" };
  const imageAt = (h: number): Pod["image"] => (cfg.pinDigest || h < RC_PUSH ? "v7" : "rc");
  let pods: Pod[] = [0, 1, 2].map((id) => ({ id, created: -14, image: "v7" }));
  let nextId = 3, row = 0;
  const hours: { h: number; rps: number; pods: Pod[]; util: number; p95: number }[] = [];
  const scored: { h: number; pod: Pod; y: number; flagged: boolean }[] = [];
  for (let h = 0; h < 24; h++) {
    if (cfg.restartAt === h) pods = pods.map((p) => ({ ...p, id: nextId++, created: h + 0.2, image: imageAt(h + 0.2) }));
    if (!cfg.autoscale && cfg.restartAt !== undefined && h >= cfg.restartAt) pods = pods.slice(0, 3);
    const rps = tagRps(h);
    const want = cfg.autoscale || cfg.restartAt === undefined || h < cfg.restartAt ? Math.min(cfg.maxReplicas, Math.max(3, Math.ceil(rps / (TAG_POD_RPS * 0.7)))) : 3;
    while (pods.length < want) pods.push({ id: nextId++, created: h + 0.05, image: imageAt(h + 0.05) });
    while (pods.length > want) pods.pop(); // scale down removes the newest pod
    const util = rps / (pods.length * TAG_POD_RPS);
    hours.push({ h, rps, pods: [...pods], util, p95: util >= 0.98 ? 4000 : (3 * 22) / (1 - util) });
    const n = Math.round(rps * 1.3);
    for (let k = 0; k < n; k++) {
      const r = traffic[row++ % traffic.length], pod = pods[k % pods.length];
      const prob = pod.image === "v7" ? v7.prob(r.x) : rc.prob(r.x); // production still sends schema v1
      scored.push({ h, pod, y: r.y, flagged: prob >= THRESH });
    }
  }
  const recall = (xs: typeof scored) => { const pos = xs.filter((s) => s.y); return pos.length ? pos.filter((s) => s.flagged).length / pos.length : Number.NaN; };
  const flagRate = (xs: typeof scored) => xs.filter((s) => s.flagged).length / Math.max(1, xs.length);
  return { hours, scored, recall, flagRate, peakP95: Math.max(...hours.map((x) => x.p95)) };
}

const TAG_ARCH: Scenario["architecture"] = {
  nodes: [
    { id: "checkout", label: "Checkout", Icon: User, x: 0, y: 100, match: (s) => s.id === "req" },
    { id: "features", label: "Feature service", Icon: Database, x: 160, y: 100, match: (s) => s.id === "features" },
    { id: "pods", label: "Scorer pods", Icon: Boxes, x: 340, y: 100, match: (s) => s.id === "predict" },
    { id: "hpa", label: "Autoscaler", Icon: Gauge, x: 340, y: 230, match: () => false },
    { id: "registry", label: "Image registry (:latest)", Icon: Tag, x: 540, y: 100, match: (s) => s.id === "pull" },
    { id: "ci", label: "Staging CI", Icon: Cpu, x: 540, y: 230, match: () => false },
  ],
  edges: [{ source: "checkout", target: "features" }, { source: "features", target: "pods" }, { source: "hpa", target: "pods", sourceHandle: "r", targetHandle: "l" }, { source: "registry", target: "pods" }, { source: "ci", target: "registry", sourceHandle: "r", targetHandle: "l" }],
};

function tagRun(fixId?: string): Outcome {
  const cfg = { ...TAG_TODAY, ...(TAG_FIXES.find((f) => f.id === fixId)?.cfg ?? {}) };
  const sim = simulateTag(cfg), base = simulateTag({ ...TAG_TODAY, pinDigest: true });
  const { v7, rc } = getModels();
  const recall = sim.recall(sim.scored), baseRecall = base.recall(base.scored);
  const resolved = recall >= baseRecall - 0.02 && sim.peakP95 <= TAG_SLO;
  const band = (s: typeof sim, a: number, b: number) => s.scored.filter((x) => x.h >= a && x.h < b);
  const peak = sim.hours[14];
  const podsAt = peak.pods;
  const rep = sim.scored.find((x) => x.h === 14 && x.pod.image === "rc" && x.y === 1) ?? sim.scored.find((x) => x.h === 14 && x.y === 1)!;
  const digest = (img: "v7" | "rc") => (img === "v7" ? "sha256:9f2c…e1 (fraud-model v7)" : "sha256:41ab…7d (fraud-rc-v8)");
  const held = fraudData(34, 3000);
  const rcOffline = evaluateModel(rc, toV2(held.rows), THRESH).recall, rcProd = evaluateModel(rc, held.rows, THRESH).recall, v7Offline = evaluateModel(v7, held.rows, THRESH).recall;
  const spans: TraceSpan[] = [
    { id: "req", name: "POST /score (order #88412)", component: "checkout", startMs: 0, durationMs: 31, status: rep.flagged ? "ok" : "warning", output: { decision: rep.flagged ? "review" : "approve", label_next_day: "chargeback" }, events: rep.flagged ? [] : [ev("fraudulent order approved", "warning")] },
    { id: "features", parentId: "req", name: "fetch features (schema v1)", component: "feature-service", startMs: 0, durationMs: 9, status: "ok", output: { schema: "v1", columns: "amount_usd, is_foreign, night_hour, txn_velocity_1h" }, events: [] },
    { id: "predict", parentId: "req", name: `predict on scorer-${rep.pod.id + 1}`, component: "model-server", startMs: 9, durationMs: 18, status: rep.flagged ? "ok" : "warning", input: { pod_created: hhmm(Math.round(rep.pod.created * 3600)), threshold: THRESH }, output: { model: rep.pod.image === "v7" ? "fraud-model v7" : "fraud-rc-v8", flagged: rep.flagged }, events: [] },
    { id: "pull", parentId: "predict", name: "image (pulled at pod start)", component: "registry", startMs: 9, durationMs: 0, status: "ok", output: { ref: cfg.pinDigest ? "fraud-model@sha256:9f2c…e1" : "fraud-model:latest", digest: digest(rep.pod.image) }, events: [] },
  ];
  return {
    spans,
    metrics: [
      { label: "p95 scoring latency (peak hour)", value: ms(sim.peakP95), status: sim.peakP95 <= TAG_SLO ? "ok" : "error", kind: "health", hint: `SLO ${TAG_SLO} ms` },
      { label: "Error rate", value: "0.0%", status: "ok", kind: "health" },
      { label: "Fraud recall (labelled next day)", value: pct(recall, 0), status: recall >= baseRecall - 0.02 ? "ok" : "error", kind: "quality", hint: `v7 baseline ${pct(baseRecall, 0)}` },
      { label: "Orders flagged for review", value: pct(sim.flagRate(sim.scored)), status: "ok", kind: "quality" },
    ],
    transcript: [{ label: "Risk team", text: resolved ? `Next-day labels: fraud recall ${pct(recall, 0)}, back to normal.` : `Yesterday's chargebacks: we caught only ${pct(recall, 0)} of fraud (normally ${pct(baseRecall, 0)}). Misses cluster in the afternoon. Is the model buckling under peak load?`, tone: resolved ? "ok" : "err" }],
    resolved,
    evidence: [
      evidence("dashboards", "Plot fraud recall by time of day", [[0, 10, "00:00–10:00"], [10, 13, "10:00–13:00"], [13, 19, "13:00–19:00 (peak)"], [19, 24, "19:00–24:00"]].map(([a, b, l]) => `${l}: recall ${pct(sim.recall(band(sim, a as number, b as number)), 0)}, ${Math.round(tagRps(((a as number) + (b as number)) / 2))} req/s`), false),
      evidence("bypod", "Break recall down by pod (14:00–15:00)", [...new Map(podsAt.map((p) => [p.image, p])).values()].map((p) => { const xs = sim.scored.filter((x) => x.h === 14 && x.pod.image === p.image); const ids = podsAt.filter((q) => q.image === p.image).map((q) => q.id + 1); return `pods ${ids.join(", ")}: recall ${pct(sim.recall(xs), 0)}, flag rate ${pct(sim.flagRate(xs))}, utilization ${pct(peak.util, 0)}`; }), true, "predict"),
      evidence("digests", "List pods with creation time and image digest", podsAt.slice(0, 8).map((p) => `scorer-${p.id + 1}: created ${p.created < 0 ? "yesterday 10:00" : hhmm(Math.round(p.created * 3600))}, ${digest(p.image)}`), true, "pull"),
      evidence("registry", "Read the tag history of fraud-model:latest", ["Oct 12 09:55 → sha256:9f2c…e1 (v7, promoted by release pipeline)", `Today ${hhmm(Math.round(RC_PUSH * 3600))} → sha256:41ab…7d (pushed by ci/staging-train: “fraud-rc-v8 for schema v2”)`], true, "pull"),
      evidence("manifest", "Read the scorer deployment manifest", [`image: ${cfg.pinDigest ? "fraud-model@sha256:9f2c…e1" : "fraud-model:latest"}`, "imagePullPolicy: Always", "no deploys to fraud-scorer since Oct 12"], true),
      evidence("hpa", "Check autoscaler activity", [9, 12, 14, 17, 21].map((h) => `${String(h).padStart(2, "0")}:00: ${sim.hours[h].pods.length} pods for ${Math.round(sim.hours[h].rps)} req/s`), true),
      evidence("load", "Check pod CPU, memory and latency at peak", [`utilization at 14:00: ${pct(peak.util, 0)} on every pod; p95 ${ms(peak.p95)}`, "no OOMs, no throttling; old and new pods equally loaded"], false),
      evidence("drift", "Compare feature distributions at peak vs overnight", ["PSI amount_usd 0.02, is_foreign 0.01, night_hour 0.03, txn_velocity_1h 0.02 (all stable)", "fraud base rate 3.1% overnight vs 3.0% at peak"], false, "features"),
      evidence("rc", "Evaluate fraud-rc-v8 offline", [`on schema v2 inputs (what it was trained for): recall ${pct(rcOffline, 0)}`, `on schema v1 inputs (what production sends): recall ${pct(rcProd, 0)}`, `v7 on schema v1: recall ${pct(v7Offline, 0)}`], true),
    ],
  };
}

export const PROD_MORE_SCENARIOS: Scenario[] = [
  {
    id: "prod-autoscaler-cap", title: "Slow every afternoon, fine every morning", system: "Production", difficulty: "Beginner", relatedTopic: "latency-and-throughput",
    symptom: "Afternoon p95 in the seconds and 504 timeouts", objective: "Use Little's law to see a queue that grows because capacity is capped below peak demand.",
    incident: "Since Monday, the summarize API is slow every afternoon and some requests fail with 504 Gateway Timeout. Mornings and evenings are fine. The model hasn't changed, and the autoscaler is running.",
    architecture: SCALE_ARCH, run: scalingRun,
    hypotheses: [
      { id: "h-cap", text: "The autoscaler's max replicas is set below what the afternoon peak needs, so arrivals exceed capacity and the queue grows.", isRootCause: true, evidenceIds: ["replicas", "queue", "pods", "hpa", "deploys"], explanation: "The autoscaler wants 13 pods at peak but is capped at 6 since Monday's cost review. 6 pods serve 180 req/s against ~260 arriving, so the queue grows until requests hit the 10 s timeout (W = L / X)." },
      { id: "h-model", text: "The model got slower.", isRootCause: false, evidenceIds: ["pods", "model"], explanation: "Per-request inference time is unchanged. The time is spent waiting in the queue, not on the GPU." },
      { id: "h-traffic", text: "A traffic spike or bot is flooding the service.", isRootCause: false, evidenceIds: ["traffic"], explanation: "Traffic matches the same day last week. The service handled it then with up to 13 pods." },
      { id: "h-hpa", text: "The autoscaler is broken and isn't reacting.", isRootCause: false, evidenceIds: ["hpa", "replicas"], explanation: "It reacts within seconds; it simply isn't allowed past 6 pods." },
      { id: "h-deps", text: "The document store is slow.", isRootCause: false, evidenceIds: ["deps"], explanation: "Doc-store latency is unchanged at 18 ms." },
    ],
    fixes: publicFixes(SCALE_FIXES),
    verifications: [
      { id: "v-replay", label: "Replay a full day of traffic, confirm peak queue wait stays under 1 s, and alert whenever desired replicas exceed max replicas", sound: true, explanation: "Tests the peak that broke, and catches the next cap before users do." },
      { id: "v-morning", label: "Check p95 at 09:00 the next morning", sound: false, explanation: "Mornings were always fine; the peak is in the afternoon." },
      { id: "v-model", label: "Benchmark a single pod's inference time", sound: false, explanation: "A single pod was never slow; the problem is how many pods there are." },
    ],
    rootCause: "A cost review lowered the autoscaler's max replicas from 16 to 6 based on average usage. The afternoon peak needs about 13 pods. With 6, arrivals exceed throughput by up to ~80 req/s, so the queue grows (W = L / X) until requests wait 10 s and the gateway returns 504.",
  },
  {
    id: "prod-cache-key-time", title: "The LLM bill doubled overnight", system: "Production", difficulty: "Beginner", relatedTopic: "cost-optimization",
    symptom: "Spend up ~2× with flat traffic", objective: "Find why a response cache stopped hitting, and tell a cost problem from a pricing or traffic problem.",
    incident: "Finance flagged that today's projected LLM spend is about double yesterday's. Traffic is flat, the provider didn't change prices, and answers look the same. Nobody touched the cache.",
    architecture: CACHE_ARCH, run: cacheRun,
    hypotheses: [
      { id: "h-key", text: "The new prompt line with the current time is part of the cache key, so every request gets a unique key and the cache never hits.", isRootCause: true, evidenceIds: ["calls", "hits", "keys", "store", "deploys"], explanation: "The cache key is a hash of the full prompt. Template v8 added “Current time: …” to the first line, so the same question gets a new key every second. Every entry is written and never read." },
      { id: "h-price", text: "The provider raised its prices.", isRootCause: false, evidenceIds: ["billing"], explanation: "Per-token prices on the invoice are unchanged." },
      { id: "h-traffic", text: "Traffic grew.", isRootCause: false, evidenceIds: ["traffic"], explanation: "Request volume is flat. LLM calls per request went up, not requests." },
      { id: "h-cachedown", text: "The cache cluster is down.", isRootCause: false, evidenceIds: ["store"], explanation: "The cache is healthy and accepts writes. It's just never asked for the same key twice." },
      { id: "h-long", text: "Answers got longer.", isRootCause: false, evidenceIds: ["tokens"], explanation: "Output tokens per call are unchanged; input grew by only 12 tokens." },
    ],
    fixes: publicFixes(CACHE_FIXES),
    verifications: [
      { id: "v-hit", label: "Send the same question twice through the real prompt builder and assert the second is a cache hit; alert on hit rate and LLM calls per request", sound: true, explanation: "Tests the key the cache actually sees, and makes the next silent drop visible before the invoice." },
      { id: "v-invoice", label: "Wait for tomorrow's invoice", sound: false, explanation: "It confirms the cost after another day of overspending and says nothing about why." },
      { id: "v-redis", label: "Confirm Redis is up and writes succeed", sound: false, explanation: "It was up the whole time; writes succeeding is part of the problem." },
    ],
    rootCause: "Prompt template v8 added “Current time: <timestamp>” to the first line of every prompt. The response cache keys on a hash of the full prompt, so identical questions now produce a different key every second. Hit rate fell from ~55% to 0%, and every request became a paid LLM call.",
  },
  {
    id: "prod-shared-rate-limit", title: "429s for customers far below their quota", system: "Production", difficulty: "Beginner", relatedTopic: "rate-limits-and-retries",
    symptom: "Most requests get 429 Too Many Requests", objective: "Check what a rate limiter is actually keyed on before raising limits.",
    incident: "Since 09:00 most API calls fail with 429 Too Many Requests. A new customer sending 3 requests per second on a 60 req/s plan gets the same 429s as everyone else. The LLM provider says it isn't rate-limiting us.",
    architecture: RL_ARCH, run: limiterRun,
    hypotheses: [
      { id: "h-key", text: "The limiter is keyed on the connection's source IP, which is now the load balancer's IP for everyone, so all tenants share one 60 req/s bucket.", isRootCause: true, evidenceIds: ["bytenant", "keys", "headers", "config", "deploys"], explanation: "Every tenant is rejected at the same rate regardless of its own usage. The limiter has one active key, 10.0.4.12, the new load balancer, because remote_addr is now the proxy, not the client." },
      { id: "h-provider", text: "The LLM provider is rate-limiting us.", isRootCause: false, evidenceIds: ["provider", "headers"], explanation: "The 429s come from our own gateway; the provider returned none." },
      { id: "h-abuse", text: "One tenant is abusing the API and using up capacity.", isRootCause: false, evidenceIds: ["traffic", "bytenant"], explanation: "No tenant is over its quota, and even the largest gets rejected at the same rate as the smallest." },
      { id: "h-quota", text: "The 60 req/s limit is too low for our growth.", isRootCause: false, evidenceIds: ["traffic", "config"], explanation: "Every tenant is under 60 req/s. The limit is fine; it's being applied to everyone at once." },
    ],
    fixes: publicFixes(RL_FIXES),
    verifications: [
      { id: "v-two", label: "Send traffic as two tenants through the load balancer: one over quota, one under; assert only the first gets 429s, and alert if limiter key cardinality drops", sound: true, explanation: "Tests that buckets are separate per tenant on the real network path." },
      { id: "v-direct", label: "Call the API directly from a laptop, bypassing the load balancer", sound: false, explanation: "That path still uses the laptop's IP as the key, so it works even with the bug." },
      { id: "v-provider", label: "Confirm with the provider that we're under their limit", sound: false, explanation: "We were; the provider was never involved." },
    ],
    rootCause: "The rate limiter keys buckets on remote_addr. At 09:00 traffic moved behind a new L7 load balancer, which proxies connections, so every request now arrives from 10.0.4.12. All 12 tenants share one 60 req/s bucket while together sending ~290 req/s, so about 80% of everyone's requests are rejected.",
  },
  {
    id: "prod-cold-replica", title: "Latency bursts every time the service scales out", system: "Production", difficulty: "Advanced", relatedTopic: "model-serving",
    symptom: "p95 spikes during autoscaler scale-outs", objective: "Separate a red-herring config change from the real cause, by combining per-replica latency with the replica's lifecycle.",
    incident: "Since last week's embed-svc chart 3.0 release, the search team sees bursts of very slow embedding calls whenever the service scales out. The release also doubled the max batch size, and most of the team suspects that.",
    architecture: SERVE_ARCH, run: servingRun,
    hypotheses: [
      { id: "h-probe", text: "The readiness probe now passes before the model is loaded, so the load balancer sends traffic to each new replica while its weights are still loading.", isRootCause: true, evidenceIds: ["dashboards", "byreplica", "timeline", "probe", "deploys"], explanation: "Slow requests occur only with a scale-out and almost all were served by the new replica, arriving after its pod started but before its weights finished loading. /healthz returns 200 as soon as the HTTP server is up." },
      { id: "h-batch", text: "The larger max batch size (16) makes requests wait for bigger batches.", isRootCause: false, evidenceIds: ["batches", "dashboards"], explanation: "Batches rarely fill, and latency is fine without a scale-out with the same batch setting." },
      { id: "h-capacity", text: "Two replicas aren't enough; we need more headroom.", isRootCause: false, evidenceIds: ["byreplica", "traffic"], explanation: "The warm replicas stay fast. Adding replicas doesn't stop the next new one getting traffic too early." },
      { id: "h-gpu", text: "The new replica landed on a faulty GPU node.", isRootCause: false, evidenceIds: ["gpu", "batches"], explanation: "Once loaded, its batch times match the warm replicas; all nodes are healthy." },
      { id: "h-autoscaler", text: "The autoscaler is misconfigured and scaling when it shouldn't.", isRootCause: false, evidenceIds: ["traffic"], explanation: "It scales at 70% GPU utilization, as intended. Scaling out is fine; routing to an unready replica isn't." },
    ],
    fixes: publicFixes(SERVE_FIXES),
    verifications: [
      { id: "v-scale", label: "Under production-like load, force a scale-out and assert p95 stays in SLO and no request reaches a replica before its model is loaded", sound: true, explanation: "Reproduces the trigger and checks the property that broke." },
      { id: "v-steady", label: "Load-test at steady traffic and confirm p95 is in SLO", sound: false, explanation: "It always was without a scale-out." },
      { id: "v-batch", label: "Benchmark batch sizes 8 vs 16 offline", sound: false, explanation: "It measures the red herring; batch size isn't what the slow requests wait for." },
    ],
    rootCause: "Chart 3.0 pointed the readiness probe at /healthz, which returns 200 as soon as the HTTP server starts. When the autoscaler adds a replica, the load balancer starts sending it traffic while its weights are still loading, and those requests sit in its queue until the model is ready. The batch-size change in the same release is unrelated.",
  },
  {
    id: "prod-latest-tag", title: "Fraud slips through at peak hours", system: "Production", difficulty: "Advanced", relatedTopic: "model-versioning",
    symptom: "Fraud recall drops in the afternoon, recovers at night", objective: "Look past a load correlation: combine per-pod quality, image digests and tag history to find a model version you never deployed.",
    incident: "Next-day chargeback labels show the fraud scorer missing far more fraud than usual, mostly in the afternoon peak. Nobody has deployed fraud-scorer since Oct 12. The on-call thinks the model degrades under load and wants to add replicas.",
    architecture: TAG_ARCH, run: tagRun,
    hypotheses: [
      { id: "h-tag", text: "The deployment pulls the mutable :latest tag; a staging CI job retagged it to a release candidate, and only pods the autoscaler creates afterwards run it.", isRootCause: true, evidenceIds: ["bypod", "digests", "registry", "manifest", "hpa", "rc"], explanation: "At the same hour and load, the original pods catch fraud normally and the newer pods barely do. The newer pods were created after 10:07, when ci/staging-train pushed fraud-rc-v8 to :latest. That candidate expects schema v2 amounts in cents; fed production's dollar amounts, every order looks 100× smaller and far less gets flagged. Scale-down removes the newest pods, so recall recovers at night." },
      { id: "h-load", text: "The model degrades under peak load.", isRootCause: false, evidenceIds: ["load", "bypod"], explanation: "Old and new pods carry the same load; only the new ones miss fraud." },
      { id: "h-drift", text: "Fraud patterns are different at peak hours (drift).", isRootCause: false, evidenceIds: ["drift"], explanation: "Feature distributions and fraud base rate at peak match overnight." },
      { id: "h-decay", text: "v7 has decayed and needs retraining.", isRootCause: false, evidenceIds: ["rc", "bypod"], explanation: "Pods running v7 perform at their usual recall." },
      { id: "h-features", text: "The feature service returns stale features at peak.", isRootCause: false, evidenceIds: ["drift", "bypod"], explanation: "All pods get features from the same service; only some pods are wrong." },
    ],
    fixes: publicFixes(TAG_FIXES),
    verifications: [
      { id: "v-digest", label: "Assert every running pod reports the digest in the release record, scale out under load, and check new pods' recall on a labelled replay", sound: true, explanation: "Checks the exact property that broke (what new pods run) and that quality holds after scale-outs." },
      { id: "v-latency", label: "Confirm peak latency and CPU are within limits", sound: false, explanation: "They were within limits throughout; the problem is quality, not load." },
      { id: "v-oldpods", label: "Spot-check scores on scorer-1", sound: false, explanation: "scorer-1 is an original pod and always ran v7." },
    ],
    rootCause: "The scorer deployment references fraud-model:latest with imagePullPolicy Always. At 10:07 a staging CI job pushed fraud-rc-v8, built for a future feature schema with amounts in cents, to :latest. Existing pods kept v7, but every pod the autoscaler added at peak pulled the candidate, which reads production's dollar amounts as 100× smaller and flags far fewer risky orders. Recall falls with each scale-up and recovers as scale-down removes the newest pods, so it looks like a load problem.",
  },
];
