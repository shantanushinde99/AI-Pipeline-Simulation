import { CORPUS, QUESTIONS } from "../visualizers/rag/corpus";
import { DEFAULT_RAG, TEMPLATES, answerQuestion, chunk, embed, parse, splitSentences, type RagConfig, type RagIndex } from "../visualizers/rag/simulation";
import { RAG_ARCH } from "./rag-scenarios";
import type { Evidence, Outcome, Scenario } from "./types";

const pct = (v: number) => `${Math.round(v * 100)}%`;
const globRe = (g: string) => new RegExp(`^${g.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*")}$`);
const excludedBy = (file: string, rules: string[]) => rules.find((g) => globRe(g).test(file));

/** The same parse → chunk → embed flow as buildIndex, but over the files the ingestion filter lets through. */
function buildFilteredIndex(cfg: RagConfig, exclude: string[]): RagIndex {
  const sources = CORPUS.filter((d) => !excludedBy(d.file, exclude));
  const parsed = sources.map((d) => parse(d, cfg.tables));
  const chunks = parsed.flatMap((p) => chunk(p, cfg.chunkSize, cfg.overlap));
  return { parsed, chunks, sources, model: cfg.indexModel, builtAt: "2026-10-18", entries: chunks.map((c) => ({ ...c, vector: embed(c.text, cfg.indexModel), model: cfg.indexModel })) };
}

const FAULT_EXCLUDE = ["data-*", "*.csv"];
const FIXES: (Scenario["fixes"][number] & { exclude?: string[]; cfg?: Partial<RagConfig> })[] = [
  { id: "f-filter", label: "Narrow the exclude rule to data-export-* and re-index", explanation: "data-retention.md passes the filter, is chunked and embedded, and the fact becomes retrievable.", exclude: ["data-export-*", "*.csv"] },
  { id: "f-topk", label: "Raise top-k to 8 and put 5 chunks in context", explanation: "More chunks from an index that doesn't contain the document.", cfg: { topK: 8, contextChunks: 5 } },
  { id: "f-reindex", label: "Re-run the nightly ingestion job", explanation: "The same filter skips the same file again.", exclude: FAULT_EXCLUDE },
];

function missingDocRun(fixId?: string): Outcome {
  const fix = FIXES.find((f) => f.id === fixId);
  const cfg: RagConfig = { ...DEFAULT_RAG, ...(fix?.cfg ?? {}) };
  const exclude = fix?.exclude ?? FAULT_EXCLUDE;
  const index = buildFilteredIndex(cfg, exclude);
  const runs = QUESTIONS.map((q) => answerQuestion(index, cfg, q));
  const r = runs.find((x) => x.question.id === "q-retention")!;
  const rate = (f: (x: (typeof runs)[number]) => boolean) => runs.filter(f).length / runs.length;
  const accuracy = rate((x) => x.evaluation.correct), faithfulness = rate((x) => x.evaluation.faithful);
  const fact = r.question.expect;
  const src = CORPUS.find((d) => d.id === r.question.sourceDoc)!;
  const lat = runs.map((x) => x.spans.find((s) => s.id === "rag.query")!.durationMs).sort((a, b) => a - b);
  const ev = (id: string, action: string, lines: string[], relevant: boolean, spanId?: string): Evidence => ({ id, action, title: action, lines, relevant, spanId });
  const evidence: Evidence[] = [
    ev("golden", "Run the golden-set evaluation", [`answer accuracy ${pct(accuracy)}, faithfulness ${pct(faithfulness)}`, ...runs.map((x) => `${x.evaluation.ok ? "PASS" : "FAIL"} · ${x.question.text}`)], false),
    ev("answer", "Compare the answer with the source of truth", [`Answer given: “${r.answer.text}”`, `Source of truth (${src.file} ${src.version}): “${splitSentences(parse(src, "linearize").text).find((s) => s.includes(fact))}”`], false),
    ev("retrieval", "Inspect retrieval results", [`top-${r.cfg.topK}:`, ...r.retrieved.map((h) => `#${h.rank} ${h.entry.id} cos=${h.score.toFixed(3)}`), r.evaluation.retrievalHit ? "A retrieved chunk contains the reference fact." : `No retrieved chunk contains “${fact}”.`], true, "retrieve"),
    ev("search", `Search every chunk in the index for “${fact}”`, [`${index.chunks.length} chunks from ${index.sources.length} documents`, `${index.chunks.filter((c) => c.text.includes(fact)).length} chunk(s) contain “${fact}”`], true, "chunk"),
    ev("index", "List the documents in the index vs the docs repository", CORPUS.map((d) => `${d.file} ${d.version}: ${index.sources.some((s) => s.id === d.id) ? "indexed" : "NOT IN INDEX"}`), true),
    ev("ingest", "Read the ingestion job log", [`exclude rules: ${JSON.stringify(exclude)}`, ...CORPUS.map((d) => { const rule = excludedBy(d.file, exclude); return rule ? `skip ${d.file} (matched exclude '${rule}')` : `ingest ${d.file} → parsed, chunked, embedded`; }), "job status: SUCCESS"], true),
    ev("models", "Compare query and index embedding models", [`query encoder: ${cfg.queryModel}`, `index vectors: ${index.model}`, `other golden questions pass: ${runs.filter((x) => x !== r && x.evaluation.ok).length}/${runs.length - 1}`], false, "embed.query"),
    ev("faithfulness", "Run a faithfulness check", [r.answer.source === "none" ? "No answer was given, nothing to check." : r.evaluation.faithful ? "Answer sentence appears verbatim in the context: faithful." : "Answer sentence does NOT appear in the context: unfaithful.", `template: ${cfg.promptTemplate} (“${TEMPLATES[cfg.promptTemplate].slice(0, 60)}…”)`], false),
    ev("dashboards", "Check latency and error dashboards", [`request success rate 100%, HTTP 5xx 0%`, `p95 latency ${lat[lat.length - 1].toFixed(0)} ms (SLO 1500 ms)`, "ingestion job: last 7 runs SUCCESS"], false),
    ev("deploys", "Read the deploy log", ["Oct 16 ingestion 2.4: exclude rule 'data-*' added so nightly data-export-*.csv dumps stop polluting the index", "Oct 16 nightly re-index: SUCCESS (5 → 4 documents)", "Oct 17 web-chat 4.5: answer feedback buttons", "Oct 18 reranker 2.2: tie-breaking by rank"], true),
  ];
  return {
    spans: r.spans,
    metrics: [
      { label: "Success rate", value: "100%", status: "ok", kind: "health" },
      { label: "p95 latency", value: `${lat[lat.length - 1].toFixed(0)} ms`, status: "ok", kind: "health" },
      { label: "Ingestion job", value: "SUCCESS", status: "ok", kind: "health" },
      { label: "Golden-set accuracy", value: pct(accuracy), status: accuracy === 1 ? "ok" : "error", kind: "quality" },
      { label: "Faithfulness", value: pct(faithfulness), status: faithfulness === 1 ? "ok" : "error", kind: "quality" },
    ],
    transcript: [{ label: "User", text: r.question.text }, { label: "Assistant", text: r.answer.text, tone: r.evaluation.ok ? "ok" : "err" }],
    resolved: r.evaluation.ok && accuracy === 1,
    evidence,
  };
}

export const RAG_EXTRA_SCENARIOS: Scenario[] = [
  {
    id: "rag-missing-document", title: "Assistant can't answer log-retention questions", system: "RAG", difficulty: "Beginner", relatedTopic: "rag-pipeline",
    symptom: "Enterprise retention questions get an off-topic answer", objective: "Check what is actually in the index before tuning retrieval.",
    incident: `Since Thursday, enterprise customers asking “${QUESTIONS.find((q) => q.id === "q-retention")!.text}” get an answer about SSO instead of the retention period. The data-retention page hasn't changed in months. The nightly ingestion job reports success.`,
    architecture: RAG_ARCH, run: missingDocRun,
    hypotheses: [
      { id: "h-missing", text: "data-retention.md was never indexed: the new ingestion exclude rule matches its filename.", isRootCause: true, evidenceIds: ["index", "ingest", "search", "deploys"], explanation: "The ingestion log shows 'skip data-retention.md (matched exclude data-*)', the index holds 4 of 5 documents, and no chunk anywhere contains “365 days”." },
      { id: "h-topk", text: "top-k is too small, so the retention chunk isn't retrieved.", isRootCause: false, evidenceIds: ["search", "retrieval"], explanation: "There is no retention chunk to retrieve: zero chunks in the whole index contain the fact." },
      { id: "h-halluc", text: "The generator is ignoring the context and hallucinating.", isRootCause: false, evidenceIds: ["faithfulness", "answer"], explanation: "The answer is a sentence copied from the context. It's the best match among the wrong documents." },
      { id: "h-embed", text: "The query and index embedding models don't match.", isRootCause: false, evidenceIds: ["models", "golden"], explanation: "Both are embed-v1 and every other golden question passes." },
      { id: "h-parser", text: "The parser dropped the retention paragraph.", isRootCause: false, evidenceIds: ["ingest"], explanation: "The file never reached the parser; there is no parse step for it at all." },
    ],
    fixes: FIXES.map(({ id, label, explanation }) => ({ id, label, explanation })),
    verifications: [
      { id: "v-coverage", label: "Re-run the golden set and add an ingestion check that every file in the docs repo is either indexed or on an explicit, reviewed skip list", sound: true, explanation: "Confirms the fix and turns a silent skip into a failing check." },
      { id: "v-job", label: "Confirm the ingestion job reports SUCCESS", sound: false, explanation: "It reported SUCCESS while skipping the file." },
      { id: "v-topk", label: "Check that retrieval returns 8 chunks", sound: false, explanation: "Volume of retrieved chunks says nothing about whether the document exists." },
    ],
    rootCause: "An exclude rule meant for nightly data-export CSV dumps ('data-*') also matches data-retention.md. The file is skipped before parsing, so the index has no chunk with the retention period, and retrieval returns the closest unrelated passages. The job still reports success.",
  },
];
