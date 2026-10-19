import { dot } from "../../lib/math";
import type { SimulationEvent, TraceSpan } from "../../types/simulation";
import { CORPUS, MODEL_PRIOR, QUESTIONS, RATE_LIMITS_V2, type Block, type Question, type SourceDoc } from "./corpus";

export type EmbedModel = "embed-v1" | "embed-v2";
export type PromptTemplate = "grounded-v1" | "concise-v2";

export type RagConfig = {
  chunkSize: number;
  overlap: number;
  topK: number;
  rerank: boolean;
  contextChunks: number;
  contextBudget: number;
  // Fault knobs used by the Failure Lab. Defaults are the healthy system.
  tables: "linearize" | "drop";
  indexSnapshot: "current" | "stale";
  queryModel: EmbedModel;
  indexModel: EmbedModel;
  citationBug: boolean;
  promptTemplate: PromptTemplate;
};

export const DEFAULT_RAG: RagConfig = {
  chunkSize: 80, overlap: 15, topK: 4, rerank: true, contextChunks: 3, contextBudget: 300,
  tables: "linearize", indexSnapshot: "current", queryModel: "embed-v1", indexModel: "embed-v1", citationBug: false, promptTemplate: "grounded-v1",
};

/** Clamp user-controlled parameters into valid ranges (overlap must be smaller than chunk size, etc.). */
export function sanitize(cfg: RagConfig): RagConfig {
  const chunkSize = Math.max(5, Math.round(cfg.chunkSize));
  const topK = Math.max(1, Math.round(cfg.topK));
  return {
    ...cfg, chunkSize, topK,
    overlap: Math.max(0, Math.min(Math.round(cfg.overlap), chunkSize - 1)),
    contextChunks: Math.max(1, Math.min(Math.round(cfg.contextChunks), topK)),
    contextBudget: Math.max(10, Math.round(cfg.contextBudget)),
  };
}

// ---------- text processing ----------
const STOP = new Set("a an the is are was were be been of to in on for and or with by at as it its this that how do does i my me can what when which who much many after before from your you our we any if than then so up per get".split(" "));
const SYN: Record<string, string> = {
  cost: "price", costs: "price", pricing: "price", priced: "price", kept: "retain", keep: "retain", retained: "retain",
  refunded: "refund", refundable: "refund", exceed: "above", exceeding: "above", enabled: "enable",
};
export function tokens(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9$]+/g) ?? [])
    .map((w) => SYN[w] ?? w)
    .map((w) => (w.length > 3 && w.endsWith("s") && !w.endsWith("ss") ? w.slice(0, -1) : w))
    .filter((w) => !STOP.has(w));
}
export const splitSentences = (text: string) => text.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
export const wordCount = (text: string) => text.split(/\s+/).filter(Boolean).length;

// ---------- embedding ----------
export const DIM = 256;
function fnv1a(s: string, seed: number) {
  let h = (2166136261 ^ seed) >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}
/**
 * Deterministic hashed bag-of-words "embedding": each normalized token is hashed into one of 256 signed buckets.
 * Different model versions use different hash seeds, so their vector spaces are incompatible, just like real models.
 */
export function embed(text: string, model: EmbedModel): number[] {
  const seed = model === "embed-v1" ? 1 : 0x9e3779b9;
  const counts = new Map<string, number>();
  for (const t of tokens(text)) counts.set(t, (counts.get(t) ?? 0) + 1);
  const v = new Array<number>(DIM).fill(0);
  for (const [t, c] of counts) {
    const h = fnv1a(t, seed);
    v[h % DIM] += (h & 0x10000 ? 1 : -1) * (1 + Math.log(c));
  }
  const n = Math.hypot(...v);
  return n ? v.map((x) => x / n) : v;
}

// ---------- indexing flow ----------
export type ParsedDoc = { docId: string; file: string; version: string; updatedAt: string; text: string; tablesFound: number; tablesDropped: number; blocks: number };

const linearize = (b: Extract<Block, { type: "table" }>) =>
  b.rows.map((r) => `${r[0]} plan: ${b.headers.slice(1).map((h, i) => `${h} ${r[i + 1]}`).join("; ")}.`).join(" ");

export function parse(doc: SourceDoc, tables: RagConfig["tables"]): ParsedDoc {
  const tablesFound = doc.blocks.filter((b) => b.type === "table").length;
  const parts = doc.blocks.flatMap((b) => (b.type === "table" ? (tables === "drop" ? [] : [linearize(b)]) : [b.type === "heading" ? `${b.text}.` : b.text]));
  return { docId: doc.id, file: doc.file, version: doc.version, updatedAt: doc.updatedAt, text: parts.join(" "), tablesFound, tablesDropped: tables === "drop" ? tablesFound : 0, blocks: doc.blocks.length };
}

export type Chunk = { id: string; docId: string; index: number; text: string; startWord: number; endWord: number; docVersion: string };

export function chunk(p: ParsedDoc, size: number, overlap: number): Chunk[] {
  const words = p.text.split(/\s+/).filter(Boolean);
  const step = Math.max(1, size - overlap);
  const out: Chunk[] = [];
  for (let start = 0; start < words.length; start += step) {
    const end = Math.min(start + size, words.length);
    out.push({ id: `${p.docId}#${out.length}`, docId: p.docId, index: out.length, text: words.slice(start, end).join(" "), startWord: start, endWord: end, docVersion: p.version });
    if (end === words.length) break;
  }
  return out;
}

export type IndexEntry = Chunk & { vector: number[]; model: EmbedModel };
export type RagIndex = { parsed: ParsedDoc[]; chunks: Chunk[]; entries: IndexEntry[]; model: EmbedModel; builtAt: string; sources: SourceDoc[] };

export function buildIndex(cfgIn: RagConfig): RagIndex {
  const cfg = sanitize(cfgIn);
  const sources = cfg.indexSnapshot === "stale" ? CORPUS.map((d) => (d.id === RATE_LIMITS_V2.id ? RATE_LIMITS_V2 : d)) : CORPUS;
  const parsed = sources.map((d) => parse(d, cfg.tables));
  const chunks = parsed.flatMap((p) => chunk(p, cfg.chunkSize, cfg.overlap));
  return {
    parsed, chunks, sources, model: cfg.indexModel,
    builtAt: cfg.indexSnapshot === "stale" ? "2026-04-20" : "2026-10-02",
    entries: chunks.map((c) => ({ ...c, vector: embed(c.text, cfg.indexModel), model: cfg.indexModel })),
  };
}

// ---------- query flow ----------
export type Hit = { entry: IndexEntry; score: number; rank: number; rerankScore?: number; overlap?: number };
export type ContextItem = { label: number; entry: IndexEntry; text: string; tokens: number; truncated: boolean; droppedTokens: number };
export type Answer = { text: string; sentence: string; source: "context" | "prior" | "none"; citations: { label: number; chunkId: string | null }[] };
export type RagEval = { correct: boolean; retrievalHit: boolean; contextHit: boolean; faithful: boolean; citationCorrect: boolean; ok: boolean };

export type RagRun = {
  cfg: RagConfig; question: Question; index: RagIndex;
  qTokens: string[]; qVector: number[];
  scored: Hit[]; retrieved: Hit[]; reranked: Hit[];
  context: ContextItem[]; contextTokens: number;
  prompt: string; promptTokens: number;
  answer: Answer; evaluation: RagEval; spans: TraceSpan[];
};

const overlapCount = (q: string[], text: string) => {
  const t = new Set(tokens(text));
  return new Set(q.filter((w) => t.has(w))).size;
};

export const TEMPLATES: Record<PromptTemplate, string> = {
  "grounded-v1": "Answer the question using ONLY the numbered context passages. Cite the passage you used as [n]. If the context does not contain the answer, say you don't know.",
  "concise-v2": "You are a helpful Northwind Cloud assistant. Answer briefly and cite a source.",
};

export function answerQuestion(index: RagIndex, cfgIn: RagConfig, question: Question): RagRun {
  const cfg = sanitize(cfgIn);
  const qTokens = [...new Set(tokens(question.text))];
  const qVector = embed(question.text, cfg.queryModel);

  const scored: Hit[] = index.entries
    .map((entry) => ({ entry, score: dot(qVector, entry.vector), rank: 0 }))
    .sort((a, b) => b.score - a.score || a.entry.id.localeCompare(b.entry.id))
    .map((h, i) => ({ ...h, rank: i + 1 }));
  const retrieved = scored.slice(0, cfg.topK);

  // Simulated reranker: query-term overlap plus a small share of the vector score. Not a neural cross-encoder.
  const reranked = cfg.rerank
    ? retrieved
        .map((h) => { const ov = overlapCount(qTokens, h.entry.text); return { ...h, overlap: ov, rerankScore: ov + 0.5 * h.score }; })
        .sort((a, b) => b.rerankScore! - a.rerankScore! || a.rank - b.rank)
    : retrieved;

  const context: ContextItem[] = [];
  let used = 0;
  for (const h of reranked.slice(0, cfg.contextChunks)) {
    const words = h.entry.text.split(/\s+/);
    const room = cfg.contextBudget - used;
    if (room <= 0) break;
    const take = Math.min(room, words.length);
    context.push({ label: context.length + 1, entry: h.entry, text: words.slice(0, take).join(" "), tokens: take, truncated: take < words.length, droppedTokens: words.length - take });
    used += take;
    if (take < words.length) break;
  }

  const prompt = [
    `System: ${TEMPLATES[cfg.promptTemplate]}`, "", "Context:",
    ...context.map((c) => `[${c.label}] (${c.entry.id}) ${c.text}`), "", `Question: ${question.text}`, "Answer:",
  ].join("\n");

  // Simulated generator. Rule-based extraction, not a language model.
  let answer: Answer;
  const prior = MODEL_PRIOR[question.id];
  if (cfg.promptTemplate === "concise-v2" && prior && context.length) {
    // Without the grounding instruction the simulated model answers from its (outdated) memorized fact.
    answer = { text: `${prior} [1]`, sentence: prior, source: "prior", citations: [{ label: 1, chunkId: context[0].entry.id }] };
  } else {
    // Headings and fragments under 5 words are not answers.
    const candidates = context.flatMap((c) => splitSentences(c.text).filter((s) => wordCount(s) >= 5).map((s) => ({ s, label: c.label, score: overlapCount(qTokens, s) })));
    const best = candidates.reduce<(typeof candidates)[number] | null>((a, b) => (!a || b.score > a.score ? b : a), null);
    if (!best || best.score < 2) {
      answer = { text: "I couldn't find this in the provided documents.", sentence: "", source: "none", citations: [] };
    } else {
      const label = cfg.citationBug ? best.label + 1 : best.label;
      const cited = context.find((c) => c.label === label);
      answer = { text: `${best.s} [${label}]`, sentence: best.s, source: "context", citations: [{ label, chunkId: cited?.entry.id ?? null }] };
    }
  }

  const has = (t: string) => t.toLowerCase().includes(question.expect.toLowerCase());
  const citedItem = answer.citations[0] ? context.find((c) => c.label === answer.citations[0].label) : undefined;
  const evaluation: RagEval = {
    correct: has(answer.sentence),
    retrievalHit: retrieved.some((h) => has(h.entry.text)),
    contextHit: context.some((c) => has(c.text)),
    faithful: answer.source === "none" || context.some((c) => c.text.includes(answer.sentence)),
    citationCorrect: answer.source === "none" || (!!citedItem && citedItem.text.includes(answer.sentence)),
    ok: false,
  };
  evaluation.ok = evaluation.correct && evaluation.faithful && evaluation.citationCorrect;

  const contextTokens = context.reduce((s, c) => s + c.tokens, 0);
  const promptTokens = wordCount(prompt);
  const run: RagRun = { cfg, question, index, qTokens, qVector, scored, retrieved, reranked, context, contextTokens, prompt, promptTokens, answer, evaluation, spans: [] };
  run.spans = traceFor(run);
  return run;
}

// ---------- trace ----------
function traceFor(run: RagRun): TraceSpan[] {
  const { cfg, index } = run;
  const spans: TraceSpan[] = [];
  let t = 0;
  let n = 0;
  const ev = (message: string, severity: SimulationEvent["severity"] = "info", type = "log"): SimulationEvent => ({ id: `e${n++}`, type, message, severity });
  const add = (s: Omit<TraceSpan, "startMs">, advance = true) => {
    spans.push({ ...s, startMs: t });
    if (advance) t += s.durationMs;
  };

  // Indexing job (ran before this request; shown for completeness).
  const idxStart = t;
  spans.push({ id: "index.build", name: "index.build", component: "indexer", startMs: 0, durationMs: 0, status: "ok", events: [ev(`index built ${index.builtAt} with ${index.model}`)], output: { chunks: index.entries.length, model: index.model, builtAt: index.builtAt } });
  for (const p of index.parsed) {
    add({
      id: `parse:${p.docId}`, parentId: "index.build", name: `parse ${p.file}`, component: "parser", durationMs: 3 + 2 * p.blocks, status: "ok",
      input: { file: p.file, version: p.version, updatedAt: p.updatedAt }, output: { words: wordCount(p.text), tablesFound: p.tablesFound, tablesDropped: p.tablesDropped },
      events: p.tablesDropped ? [ev(`skipped ${p.tablesDropped} element(s) of type 'table' (parser mode: text-only)`)] : [],
    });
  }
  add({ id: "chunk", parentId: "index.build", name: "chunk documents", component: "chunker", durationMs: 1 + index.chunks.length * 0.1, status: "ok", input: { chunkSize: cfg.chunkSize, overlap: cfg.overlap }, output: { chunks: index.chunks.length }, events: [] });
  add({ id: "embed.index", parentId: "index.build", name: "embed chunks", component: "embedder", durationMs: 5 + index.chunks.length * 0.4, status: "ok", input: { model: index.model }, output: { vectors: index.entries.length, dim: DIM }, events: [] });
  add({ id: "index.write", parentId: "index.build", name: "write vector index", component: "vector-index", durationMs: 4, status: "ok", output: { entries: index.entries.length }, events: [] });
  spans[0].durationMs = t - idxStart;

  // The request itself.
  const reqStart = t;
  const top = run.retrieved[0]?.score ?? 0;
  spans.push({ id: "rag.query", name: "POST /ask", component: "api", startMs: t, durationMs: 0, status: "ok", input: { question: run.question.text }, output: { http: 200 }, events: [] });
  add({ id: "embed.query", parentId: "rag.query", name: "embed query", component: "embedder", durationMs: 9, status: "ok", input: { text: run.question.text, model: cfg.queryModel }, output: { dim: DIM, tokens: run.qTokens.join(" ") }, events: [] });
  add({
    id: "retrieve", parentId: "rag.query", name: `vector search top-${cfg.topK}`, component: "vector-index", durationMs: 3 + index.entries.length * 0.05, status: "ok",
    input: { topK: cfg.topK, indexModel: index.model, queryModel: cfg.queryModel }, output: { hits: run.retrieved.map((h) => `${h.entry.id} (${h.score.toFixed(3)})`).join(", ") },
    metrics: { top1_score: +top.toFixed(3) }, events: [ev(`top-1 similarity ${top.toFixed(3)}`)],
  });
  if (cfg.rerank) add({ id: "rerank", parentId: "rag.query", name: "rerank", component: "reranker", durationMs: 12 * run.retrieved.length, status: "ok", output: { order: run.reranked.map((h) => h.entry.id).join(", ") }, events: [] });
  const trunc = run.context.find((c) => c.truncated);
  add({
    id: "context", parentId: "rag.query", name: "assemble context", component: "context-builder", durationMs: 1, status: trunc ? "warning" : "ok",
    input: { budget: cfg.contextBudget, maxChunks: cfg.contextChunks }, output: { chunks: run.context.map((c) => c.entry.id).join(", "), tokens: run.contextTokens },
    events: trunc ? [ev(`${trunc.entry.id} truncated: kept ${trunc.tokens} tokens, dropped ${trunc.droppedTokens}`, "warning")] : [],
  });
  add({ id: "prompt", parentId: "rag.query", name: "build prompt", component: "prompt-builder", durationMs: 1, status: "ok", input: { template: cfg.promptTemplate }, output: { tokens: run.promptTokens }, events: [] });
  const ansTokens = wordCount(run.answer.text);
  add({
    id: "generate", parentId: "rag.query", name: "generate (simulated LLM)", component: "llm", durationMs: Math.round(350 + 1.5 * run.promptTokens + 8 * ansTokens), status: "ok",
    input: { model: "sim-gen-1", promptTokens: run.promptTokens }, output: { answer: run.answer.text, citations: run.answer.citations.map((c) => `[${c.label}]→${c.chunkId ?? "none"}`).join(", ") || "none" },
    events: [ev(`finish_reason=stop, output_tokens=${ansTokens}`)],
  });
  const root = spans.find((s) => s.id === "rag.query")!;
  root.durationMs = t - reqStart;
  return spans;
}

// ---------- golden-set evaluation ----------
export function runGoldenSet(cfg: RagConfig) {
  const index = buildIndex(cfg);
  const runs = QUESTIONS.map((q) => answerQuestion(index, cfg, q));
  const rate = (f: (r: RagRun) => boolean) => runs.filter(f).length / runs.length;
  const latencies = runs.map((r) => r.spans.find((s) => s.id === "rag.query")!.durationMs).sort((a, b) => a - b);
  return {
    runs,
    accuracy: rate((r) => r.evaluation.correct),
    retrievalRecall: rate((r) => r.evaluation.retrievalHit),
    faithfulness: rate((r) => r.evaluation.faithful),
    citationAccuracy: rate((r) => r.evaluation.citationCorrect),
    p95LatencyMs: latencies[Math.min(latencies.length - 1, Math.ceil(0.95 * latencies.length) - 1)],
    meanTop1: runs.reduce((s, r) => s + (r.retrieved[0]?.score ?? 0), 0) / runs.length,
    successRate: 1,
  };
}

export const runRag = (cfg: RagConfig, questionId: string) => answerQuestion(buildIndex(cfg), cfg, QUESTIONS.find((q) => q.id === questionId)!);
