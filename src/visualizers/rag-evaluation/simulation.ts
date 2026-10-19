import type { SimulationEvent, SimulationStep } from "../../types/simulation";
import { DEFAULT_RAG, runGoldenSet, type RagConfig, type RagRun } from "../rag/simulation";

export const CONFIGS = {
  baseline: { label: "Baseline: chunk 80, rerank on", cfg: {} },
  chunk20: { label: "Chunk size 20 (overlap 5)", cfg: { chunkSize: 20, overlap: 5 } },
  norerank: { label: "Rerank off", cfg: { rerank: false } },
  chunk20norerank: { label: "Chunk 20, rerank off", cfg: { chunkSize: 20, overlap: 5, rerank: false } },
  budget60: { label: "Context budget 60 tokens", cfg: { contextBudget: 60 } },
  stale: { label: "Fault: stale index snapshot", cfg: { indexSnapshot: "stale" } },
  embed: { label: "Fault: query encoder embed-v2", cfg: { queryModel: "embed-v2" } },
  citation: { label: "Fault: citation off-by-one", cfg: { citationBug: true } },
  prompt: { label: "Fault: grounding instruction removed", cfg: { promptTemplate: "concise-v2" } },
} satisfies Record<string, { label: string; cfg: Partial<RagConfig> }>;
export type ConfigId = keyof typeof CONFIGS;

export const METRICS = [
  { id: "recall", label: "recall@k", short: "R@k" },
  { id: "rr", label: "reciprocal rank", short: "RR" },
  { id: "contextHit", label: "context hit", short: "ctx" },
  { id: "faithful", label: "faithfulness", short: "faith" },
  { id: "correct", label: "answer correctness", short: "correct" },
  { id: "citation", label: "citation accuracy", short: "cite" },
] as const;
export type Metric = (typeof METRICS)[number]["id"];

export type QEval = {
  qid: string; question: string; expect: string; k: number;
  relevant: string[]; // chunk ids containing the expected fact
  retrieved: { id: string; score: number; relevant: boolean }[]; // final order (after rerank), top-k
  firstRelRank: number | null; // rank of first relevant chunk in the final top-k order
  indexRank: number | null; // rank of the first relevant chunk among ALL chunks by vector score
  answer: string; source: RagRun["answer"]["source"];
  m: Record<Metric, number>;
};
export type ConfigEval = { id: ConfigId; label: string; cfg: RagConfig; rows: QEval[]; means: Record<Metric, number> };

const has = (text: string, expect: string) => text.toLowerCase().includes(expect.toLowerCase());

export function evaluateRun(run: RagRun): QEval {
  const e = run.question.expect;
  const relevant = run.index.entries.filter((x) => has(x.text, e)).map((x) => x.id);
  const retrieved = run.reranked.map((h) => ({ id: h.entry.id, score: h.rerankScore ?? h.score, relevant: relevant.includes(h.entry.id) }));
  const firstIdx = retrieved.findIndex((h) => h.relevant);
  const idxRank = run.scored.findIndex((h) => relevant.includes(h.entry.id));
  const inTopK = retrieved.filter((h) => h.relevant).length;
  return {
    qid: run.question.id, question: run.question.text, expect: e, k: run.cfg.topK, relevant, retrieved,
    firstRelRank: firstIdx >= 0 ? firstIdx + 1 : null, indexRank: idxRank >= 0 ? idxRank + 1 : null,
    answer: run.answer.text, source: run.answer.source,
    m: {
      recall: relevant.length ? inTopK / relevant.length : 0,
      rr: firstIdx >= 0 ? 1 / (firstIdx + 1) : 0,
      contextHit: run.evaluation.contextHit ? 1 : 0,
      faithful: run.evaluation.faithful ? 1 : 0,
      correct: run.evaluation.correct ? 1 : 0,
      citation: run.evaluation.citationCorrect ? 1 : 0,
    },
  };
}

export function evaluateConfig(id: ConfigId, topK: number): ConfigEval {
  const k = Math.max(1, Math.min(6, Math.round(topK)));
  const cfg: RagConfig = { ...DEFAULT_RAG, ...CONFIGS[id].cfg, topK: k, contextChunks: Math.min(DEFAULT_RAG.contextChunks, k) };
  const rows = runGoldenSet(cfg).runs.map(evaluateRun);
  const means = Object.fromEntries(METRICS.map(({ id: m }) => [m, rows.reduce((s, r) => s + r.m[m], 0) / rows.length])) as Record<Metric, number>;
  return { id, label: CONFIGS[id].label, cfg, rows, means };
}

export type Flip = { qid: string; question: string; metric: Metric; from: number; to: number };
export function compare(a: ConfigEval, b: ConfigEval) {
  const deltas = Object.fromEntries(METRICS.map(({ id }) => [id, b.means[id] - a.means[id]])) as Record<Metric, number>;
  const flips: Flip[] = a.rows.flatMap((ra, i) => METRICS.filter(({ id }) => Math.abs(b.rows[i].m[id] - ra.m[id]) > 1e-9).map(({ id }) => ({ qid: ra.qid, question: ra.question, metric: id, from: ra.m[id], to: b.rows[i].m[id] })));
  return { deltas, flips };
}

export type EvalParams = { mode: "single" | "ab"; a: ConfigId; b: ConfigId; topK: number };
export const DEFAULT_EVAL: EvalParams = { mode: "ab", a: "baseline", b: "chunk20", topK: 4 };
export type EvalState = { phase: "intro" | "question" | "aggregate" | "flips"; q: number | null; filled: number };

export function simulate(params: EvalParams) {
  const A = evaluateConfig(params.a, params.topK);
  const B = params.mode === "ab" ? evaluateConfig(params.b, params.topK) : null;
  const cmp = B ? compare(A, B) : null;
  let n = 0;
  const ev = (message: string, severity?: SimulationEvent["severity"], tex?: string): SimulationEvent => ({ id: `e${n++}`, type: "eval", message, severity, tex });
  const f = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2));
  const steps: SimulationStep<EvalState>[] = [];
  steps.push({
    id: "intro", title: "Golden set and configuration",
    description: `${A.rows.length} questions, each with one expected fact and the document that should supply it. ${B ? `Two configurations run the same questions: A = ${A.label}, B = ${B.label}.` : `Configuration: ${A.label}.`} top-k = ${A.cfg.topK}.`,
    state: { phase: "intro", q: null, filled: 0 },
    events: [ev(`A: chunk ${A.cfg.chunkSize}, overlap ${A.cfg.overlap}, rerank ${A.cfg.rerank ? "on" : "off"}, k=${A.cfg.topK}`), ...(B ? [ev(`B: chunk ${B.cfg.chunkSize}, overlap ${B.cfg.overlap}, rerank ${B.cfg.rerank ? "on" : "off"}, k=${B.cfg.topK}`)] : [])],
  });
  A.rows.forEach((ra, i) => {
    const rb = B?.rows[i];
    const line = (r: QEval, tag: string): SimulationEvent[] => [
      ev(`${tag}recall@${r.k} = ${r.retrieved.filter((h) => h.relevant).length}/${r.relevant.length || 0}${r.relevant.length ? "" : " (fact not in index)"}, first relevant at rank ${r.firstRelRank ?? "—"}`, r.m.recall < 1 ? "warning" : undefined,
        `\\text{${tag}RR} = ${r.firstRelRank ? `\\tfrac{1}{${r.firstRelRank}} = ${f(r.m.rr)}` : "0"}`),
      ev(`${tag}answer: "${r.answer}" → correct ${r.m.correct ? "yes" : "no"}, faithful ${r.m.faithful ? "yes" : "no"}, citation ${r.m.citation ? "ok" : "wrong"}`, r.m.correct && r.m.faithful && r.m.citation ? undefined : "error"),
    ];
    steps.push({
      id: `q-${ra.qid}`, title: `Q${i + 1}: ${ra.question}`,
      description: `Expected fact: “${ra.expect}”. ${ra.relevant.length ? `It lives in ${ra.relevant.join(", ")}.` : "No chunk in A's index contains it."} Retrieval, context, answer and citation are each scored for this question.`,
      state: { phase: "question", q: i, filled: i + 1 },
      events: [...line(ra, B ? "A: " : ""), ...(rb ? line(rb, "B: ") : [])],
    });
  });
  steps.push({
    id: "aggregate", title: "Aggregate the metrics",
    description: B
      ? `Each metric is averaged over the ${A.rows.length} questions. Deltas (B − A): ${METRICS.map(({ id, short }) => `${short} ${cmp!.deltas[id] >= 0 ? "+" : ""}${cmp!.deltas[id].toFixed(2)}`).join(", ")}.`
      : `Each metric is averaged over the ${A.rows.length} questions: ${METRICS.map(({ id, short }) => `${short} ${A.means[id].toFixed(2)}`).join(", ")}.`,
    state: { phase: "aggregate", q: null, filled: A.rows.length },
    events: METRICS.map(({ id, label }) => ev(`${label}: A ${A.means[id].toFixed(2)}${B ? `, B ${B.means[id].toFixed(2)}` : ""}`, B && cmp!.deltas[id] < 0 ? "warning" : undefined,
      `\\overline{\\text{${label}}} = \\tfrac{1}{${A.rows.length}}\\textstyle\\sum_q m_q = ${A.means[id].toFixed(2)}${B ? `\\ \\to\\ ${B.means[id].toFixed(2)}` : ""}`)),
  });
  if (B) {
    const qs = [...new Set(cmp!.flips.map((x) => x.qid))];
    steps.push({
      id: "flips", title: "Which questions flipped",
      description: qs.length ? `${qs.length} question(s) changed on at least one metric. Averages hide which ones; the flip list tells you where to look.` : "No question changed on any metric: on this golden set the two configurations are indistinguishable.",
      state: { phase: "flips", q: null, filled: A.rows.length },
      events: cmp!.flips.length ? cmp!.flips.map((x) => ev(`${x.qid} ${x.metric}: ${f(x.from)} → ${f(x.to)}`, x.to < x.from ? "error" : undefined)) : [ev("no flips")],
    });
  }
  return { A, B, cmp, steps };
}
