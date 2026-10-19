import { clamp } from "../../lib/math";
import { rng } from "../../lib/random";
import type { SimulationEvent, SimulationStep, TraceSpan } from "../../types/simulation";

export type ServiceId = "gateway" | "retriever" | "vector-db" | "reranker" | "llm" | "tool";
export const SERVICES: { id: ServiceId; label: string }[] = [
  { id: "gateway", label: "API gateway" },
  { id: "retriever", label: "retriever" },
  { id: "vector-db", label: "vector DB ×3" },
  { id: "reranker", label: "reranker" },
  { id: "llm", label: "LLM" },
  { id: "tool", label: "tool: orders API" },
];

export type Preset = "healthy" | "slow-shard" | "tool-error";
export const PRESETS: { id: Preset; name: string }[] = [
  { id: "healthy", name: "Healthy request" },
  { id: "slow-shard", name: "Slow vector-DB shard (cold cache)" },
  { id: "tool-error", name: "Tool times out, then retries" },
];

export type TraceInput = { preset: Preset; parallel: boolean; samplingRate: number; tailSampling: boolean; seed: number };
export const DEFAULT_INPUT: TraceInput = { preset: "slow-shard", parallel: true, samplingRate: 0.2, tailSampling: false, seed: 7 };

export const SLOW_MS = 1800; // a trace slower than this counts as "slow" for tail sampling
export const POPULATION = 24;

const hex = (r: ReturnType<typeof rng>, n: number) => Array.from({ length: n }, () => Math.floor(r.next() * 16).toString(16)).join("");
export const traceparent = (traceId: string, parentSpanId: string, sampled: boolean) => `00-${traceId}-${parentSpanId}-${sampled ? "01" : "00"}`;
export const spanEnd = (s: TraceSpan) => s.startMs + s.durationMs;

/** Builds one request's spans. Timings are simulated milliseconds with seeded jitter. */
export function buildTrace(seed: number, opts: { slowShard: boolean; toolError: boolean; parallel: boolean }) {
  const r = rng(seed);
  const traceId = hex(r, 32);
  const id = () => hex(r, 16);
  const j = (base: number, spread: number) => Math.round(base + (r.next() * 2 - 1) * spread);
  const spans: TraceSpan[] = [];
  const ev = (sid: string, message: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id: `${sid}-${message.slice(0, 12)}`, type: "log", message, severity });
  const add = (s: Omit<TraceSpan, "events" | "status"> & { status?: TraceSpan["status"]; events?: SimulationEvent[] }) => {
    const span: TraceSpan = { status: "ok", events: [], ...s };
    spans.push(span);
    return span;
  };

  const rootId = id();
  const retr = add({ id: id(), parentId: rootId, name: "retrieve", component: "retriever", startMs: 8, durationMs: 0, input: { query: "where is my order 1182?" } });
  const embed = add({ id: id(), parentId: retr.id, name: "embed_query", component: "retriever", startMs: 10, durationMs: j(28, 4), metrics: { dims: 384 } });
  let cursor = spanEnd(embed) + 2;
  const shardEnds: number[] = [];
  for (let k = 0; k < 3; k++) {
    const dur = opts.slowShard && k === 2 ? j(820, 40) : j(48, 10);
    const s = add({
      id: id(), parentId: retr.id, name: `ann_search shard-${k}`, component: "vector-db", startMs: opts.parallel ? spanEnd(embed) + 2 : cursor, durationMs: dur,
      status: dur > 400 ? "warning" : "ok", output: { hits: 4 }, metrics: { top_k: 4, cache: dur > 400 ? "cold" : "warm" },
      events: dur > 400 ? [ev(`s${k}`, "cache miss: loading index segment from disk", "warning")] : [],
    });
    cursor = spanEnd(s) + 2;
    shardEnds.push(spanEnd(s));
  }
  retr.durationMs = Math.max(...shardEnds) + 3 - retr.startMs;
  retr.output = { chunks: 12 };
  const rerank = add({ id: id(), parentId: rootId, name: "rerank", component: "reranker", startMs: spanEnd(retr) + 2, durationMs: j(85, 10), input: { candidates: 12 }, output: { kept: 4 } });
  const llm1 = add({ id: id(), parentId: rootId, name: "llm.generate #1", component: "llm", startMs: spanEnd(rerank) + 2, durationMs: j(640, 40),
    metrics: { input_tokens: 1850, output_tokens: 42 }, output: { tool_call: "lookup_order(1182)" } });
  let toolEnd: number;
  if (opts.toolError) {
    const t1 = add({ id: id(), parentId: rootId, name: "tool.lookup_order", component: "tool", startMs: spanEnd(llm1) + 2, durationMs: 300, status: "error",
      events: [ev("t1", "timeout after 300 ms", "error")], metrics: { attempt: 1 } });
    const t2 = add({ id: id(), parentId: rootId, name: "tool.lookup_order (retry)", component: "tool", startMs: spanEnd(t1) + 40, durationMs: j(130, 15),
      output: { status: "shipped" }, metrics: { attempt: 2 }, events: [ev("t2", "retry after 40 ms backoff")] });
    toolEnd = spanEnd(t2);
  } else {
    const t = add({ id: id(), parentId: rootId, name: "tool.lookup_order", component: "tool", startMs: spanEnd(llm1) + 2, durationMs: j(120, 15), output: { status: "shipped" }, metrics: { attempt: 1 } });
    toolEnd = spanEnd(t);
  }
  const llm2 = add({ id: id(), parentId: rootId, name: "llm.generate #2", component: "llm", startMs: toolEnd + 2, durationMs: j(520, 40), metrics: { input_tokens: 1930, output_tokens: 96 }, output: { answer: "Order 1182 shipped yesterday." } });
  const root: TraceSpan = {
    id: rootId, name: "POST /ask", component: "gateway", startMs: 0, durationMs: spanEnd(llm2) + 4, status: spans.some((s) => s.status === "error") ? "warning" : "ok",
    input: { route: "/ask" }, output: { http_status: 200 }, events: [],
  };
  return { traceId, spans: [root, ...spans] };
}

/** The chain of spans that determines end-to-end latency, with each span's exclusive ("self") time on that chain. Self times sum to the root duration. */
export function criticalPath(spans: TraceSpan[]): { id: string; self: number }[] {
  const root = spans.find((s) => !s.parentId);
  if (!root) return [];
  const rec = (s: TraceSpan): { id: string; self: number }[] => {
    const kids = spans.filter((c) => c.parentId === s.id).sort((a, b) => spanEnd(b) - spanEnd(a));
    let cur = spanEnd(s);
    const chosen: TraceSpan[] = [];
    for (const c of kids) if (spanEnd(c) <= cur + 1e-9) { chosen.push(c); cur = c.startMs; }
    const covered = chosen.reduce((t, c) => t + c.durationMs, 0);
    return [{ id: s.id, self: s.durationMs - covered }, ...chosen.reverse().flatMap(rec)];
  };
  return rec(root);
}

export type PopTrace = { i: number; traceId: string; u: number; totalMs: number; error: boolean; slow: boolean; head: boolean; kept: boolean };

/** Head sampling keeps a trace when a hash of its trace id is below the rate; tail sampling also keeps every error or slow trace. */
export function samplePopulation(input: TraceInput): PopTrace[] {
  const r = rng(input.seed * 7919);
  return Array.from({ length: POPULATION }, (_, i) => {
    const roll = r.next();
    const slowShard = input.preset === "slow-shard" && (i === 0 || roll < 0.25);
    const toolError = input.preset === "tool-error" && (i === 0 || roll < 0.2);
    const t = buildTrace(input.seed + i * 101, { slowShard, toolError, parallel: input.parallel });
    const u = parseInt(t.traceId.slice(-8), 16) / 0xffffffff; // deterministic hash of the trace id
    const totalMs = t.spans[0].durationMs;
    const error = t.spans.some((s) => s.status === "error");
    const slow = totalMs > SLOW_MS;
    const head = u < clamp(input.samplingRate, 0, 1);
    return { i, traceId: t.traceId, u, totalMs, error, slow, head, kept: head || (input.tailSampling && (error || slow)) };
  });
}

export type HopView = { from: ServiceId; to: ServiceId; spanId: string; header: string };
export type TraceState = {
  phase: "hop" | "bottleneck" | "sampling";
  t0: number; t1: number; // simulated time window revealed by this step
  hops: HopView[];
  focus: string[]; // span ids this step is about
  headline: string;
  sub: string;
};

export type TraceRun = { traceId: string; sampled: boolean; spans: TraceSpan[]; total: number; critical: { id: string; self: number }[]; population: PopTrace[]; steps: SimulationStep<TraceState>[] };

export function simulate(raw: TraceInput): TraceRun {
  const input = { ...raw, samplingRate: clamp(raw.samplingRate, 0, 1) };
  const population = samplePopulation(input);
  const main = population[0];
  const { traceId, spans } = buildTrace(input.seed, { slowShard: input.preset === "slow-shard", toolError: input.preset === "tool-error", parallel: input.parallel });
  // With tail sampling every span is recorded and the collector decides later, so the header's sampled flag is 1.
  const sampled = input.tailSampling || main.head;
  const by = (name: string) => spans.find((s) => s.name === name)!;
  const root = spans[0], retr = by("retrieve"), embed = by("embed_query");
  const shards = spans.filter((s) => s.component === "vector-db");
  const rerank = by("rerank"), llm1 = by("llm.generate #1"), llm2 = by("llm.generate #2");
  const tools = spans.filter((s) => s.component === "tool");
  const total = root.durationMs;
  const critical = criticalPath(spans);
  const hdr = (parent: TraceSpan) => traceparent(traceId, parent.id, sampled);
  const hop = (from: ServiceId, to: ServiceId, parent: TraceSpan, child: TraceSpan): HopView => ({ from, to, spanId: child.id, header: hdr(parent) });
  const log = (id: string, message: string, severity?: SimulationEvent["severity"]): SimulationEvent => ({ id, type: "trace", message, severity });
  const steps: SimulationStep<TraceState>[] = [];
  const push = (id: string, title: string, description: string, state: TraceState, events: SimulationEvent[]) => steps.push({ id, title, description, state, events });
  const shardStart = Math.min(...shards.map((s) => s.startMs));
  const slowest = shards.reduce((a, b) => (b.durationMs > a.durationMs ? b : a));

  push("gateway", "Request hits the gateway",
    `The gateway starts the root span "POST /ask" and mints trace id ${traceId.slice(0, 8)}…. Head sampling hashes the trace id: u = ${main.u.toFixed(3)} ${main.head ? "<" : "≥"} rate ${input.samplingRate}, so the trace is ${main.head ? "sampled" : "not head-sampled"}${input.tailSampling ? " (tail sampling records everything and decides at the collector)" : ""}.`,
    { phase: "hop", t0: 0, t1: retr.startMs, hops: [], focus: [root.id], headline: "A request arrives: the gateway starts the trace", sub: `trace id ${traceId.slice(0, 8)}… · sampled flag ${sampled ? "01" : "00"}` },
    [log("root", `span start POST /ask (trace ${traceId.slice(0, 8)}…)`), log("sample", `head sampling: u=${main.u.toFixed(3)} vs rate ${input.samplingRate} → ${main.head ? "keep" : "drop"}`)]);

  push("retriever", "gateway → retriever",
    `The gateway calls the retriever and injects the header traceparent: ${hdr(root)}. The retriever extracts it, starts a child span "retrieve" whose parent is the gateway span, then embeds the query (${embed.durationMs} ms).`,
    { phase: "hop", t0: retr.startMs, t1: shardStart, hops: [hop("gateway", "retriever", root, retr)], focus: [retr.id, embed.id], headline: "The trace context hops to the retriever in a header", sub: "Same trace id, new parent id: the gateway's span id." },
    [log("h1", `traceparent: ${hdr(root)}`), log("embed", `embed_query ${embed.durationMs} ms`)]);

  push("vector-db", input.parallel ? "Fan-out to 3 shards" : "3 shards, one after another",
    input.parallel
      ? `The retriever queries 3 vector-DB shards in parallel. All three spans share the parent "retrieve". The retriever waits for the slowest (${slowest.name}, ${slowest.durationMs} ms).`
      : `Fan-out is disabled: the 3 shard queries run one after another, so their times add up (${shards.map((s) => s.durationMs).join(" + ")} ms).`,
    { phase: "hop", t0: shardStart, t1: spanEnd(retr), hops: shards.map((s) => hop("retriever", "vector-db", retr, s)), focus: shards.map((s) => s.id),
      headline: input.parallel ? "Fan-out: three shard spans run side by side" : "No fan-out: shard spans run in sequence", sub: `retrieve finishes after ${retr.durationMs} ms${slowest.durationMs > 400 ? ` because ${slowest.name} took ${slowest.durationMs} ms` : ""}` },
    shards.map((s) => log(s.id, `${s.name}: ${s.durationMs} ms${s.durationMs > 400 ? " (cache miss)" : ""}`, s.durationMs > 400 ? "warning" : undefined)));

  push("reranker", "gateway → reranker",
    `With 12 candidate chunks back, the gateway calls the reranker (${rerank.durationMs} ms), which keeps the best 4.`,
    { phase: "hop", t0: rerank.startMs, t1: spanEnd(rerank), hops: [hop("gateway", "reranker", root, rerank)], focus: [rerank.id], headline: "The reranker scores 12 chunks and keeps 4", sub: `${rerank.durationMs} ms · parent = gateway span` },
    [log("rr", `rerank ${rerank.durationMs} ms`)]);

  push("llm1", "gateway → LLM",
    `The LLM call gets the question plus 4 chunks (${llm1.metrics?.input_tokens} input tokens) and answers with a tool call after ${llm1.durationMs} ms.`,
    { phase: "hop", t0: llm1.startMs, t1: spanEnd(llm1), hops: [hop("gateway", "llm", root, llm1)], focus: [llm1.id], headline: "The LLM span: the longest single call so far", sub: `${llm1.durationMs} ms · decides to call lookup_order(1182)` },
    [log("l1", `llm.generate #1 ${llm1.durationMs} ms → tool_call lookup_order`)]);

  push("tool", tools.length > 1 ? "Tool fails, retries" : "gateway → tool",
    tools.length > 1
      ? `The first tool call times out after 300 ms (status error). The gateway retries after 40 ms; the retry succeeds in ${tools[1].durationMs} ms. Both attempts are visible as separate spans.`
      : `The gateway runs the tool the LLM asked for: lookup_order (${tools[0].durationMs} ms).`,
    { phase: "hop", t0: tools[0].startMs, t1: spanEnd(tools.at(-1)!), hops: tools.map((t) => hop("gateway", "tool", root, t)), focus: tools.map((t) => t.id),
      headline: tools.length > 1 ? "The tool call times out; the retry shows up as a second span" : "The tool call is its own span", sub: tools.length > 1 ? "Without tracing, this would look like a mysteriously slow request." : `${tools[0].durationMs} ms` },
    tools.flatMap((t) => t.events.length ? t.events : [log(t.id, `${t.name} ${t.durationMs} ms`)]));

  push("llm2", "gateway → LLM (final answer)",
    `The LLM writes the final answer from the tool result (${llm2.durationMs} ms). The gateway ends the root span at ${total} ms.`,
    { phase: "hop", t0: llm2.startMs, t1: total, hops: [hop("gateway", "llm", root, llm2)], focus: [llm2.id, root.id], headline: `The answer is written and the root span closes at ${total} ms`, sub: `${spans.length} spans, one trace id` },
    [log("l2", `llm.generate #2 ${llm2.durationMs} ms`), log("end", `span end POST /ask: ${total} ms`)]);

  const top = critical.filter((c) => c.id !== root.id).reduce((a, b) => (b.self > a.self ? b : a));
  const topSpan = spans.find((s) => s.id === top.id)!;
  push("bottleneck", "Find the bottleneck",
    `The critical path is the chain of spans that sets end-to-end latency. Its self times add up to ${total} ms. The biggest is ${topSpan.name} at ${top.self} ms (${Math.round((top.self / total) * 100)}%).`,
    { phase: "bottleneck", t0: total, t1: total, hops: [], focus: critical.map((c) => c.id), headline: `Bottleneck: ${topSpan.name} (${Math.round((top.self / total) * 100)}% of the request)`, sub: "Highlighted spans are on the critical path; speeding up anything else changes nothing." },
    critical.map((c) => log(`cp-${c.id}`, `critical: ${spans.find((s) => s.id === c.id)!.name} self ${c.self} ms`)));

  const kept = population.filter((p) => p.kept);
  const bad = population.filter((p) => p.error || p.slow);
  const badKept = bad.filter((p) => p.kept).length;
  push("sampling", "Which traces are kept?",
    `${POPULATION} similar requests. Head sampling at ${Math.round(input.samplingRate * 100)}% keeps ${population.filter((p) => p.head).length}${input.tailSampling ? `; tail sampling adds every error or slow (> ${SLOW_MS} ms) trace, for ${kept.length} in total` : ""}. Of the ${bad.length} slow or failed traces, ${badKept} ${badKept === 1 ? "was" : "were"} kept.`,
    { phase: "sampling", t0: total, t1: total, hops: [], focus: [], headline: `${kept.length} of ${POPULATION} traces kept · ${badKept} of ${bad.length} problem traces saved`, sub: input.tailSampling ? "Tail sampling decides after the trace ends, so it can keep the interesting ones." : "Head sampling decides at the start, before anyone knows the request will be slow." },
    [log("s", `kept ${kept.length}/${POPULATION}; problem traces kept ${badKept}/${bad.length}`, bad.length && badKept < bad.length ? "warning" : undefined)]);

  return { traceId, sampled, spans, total, critical, population, steps };
}
